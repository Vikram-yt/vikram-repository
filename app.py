from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import fpgrowth
import io

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "smartcart_secret"

# ---------------- MongoDB connection ----------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["grocerydb"]
users_collection = db["users"]
orders_collection = db["orders"]

# ---------------- Serve HTML pages ----------------
@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")

@app.route("/index_page")
def index_page():
    return render_template("index.html")

# ---------------- Signup API ----------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already registered"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_pw
    })
    return jsonify({"message": "Signup successful"}), 200

# ---------------- Login API ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name")
    password = data.get("password")

    user = users_collection.find_one({"name": name})
    if not user:
        return jsonify({"message": "User not found"}), 400

    if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        token = jwt.encode({
            "id": str(user["_id"]),
            "name": user["name"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        return jsonify({"message": "Login successful", "token": token}), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 400

# ---------------- Verify JWT ----------------
@app.route("/verify", methods=["POST"])
def verify():
    data = request.json
    token = data.get("token")
    if not token:
        return jsonify({"message": "No token provided"}), 400
    try:
        decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return jsonify({"message": "Token valid", "name": decoded["name"]}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token expired"}), 401
    except Exception:
        return jsonify({"message": "Invalid token"}), 401

# ---------------- Import Dataset API ----------------
@app.route("/import_dataset", methods=["POST"])
def import_dataset():
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded"}), 400
    try:
        filename = file.filename.lower()
        content = file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return jsonify({"message": "Unsupported file type"}), 400

        df = df.where(pd.notnull(df), None)
        product_cols = [c for c in df.columns if c.lower().startswith("product")]
        for c in product_cols:
            df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)

        records = df.to_dict(orient="records")
        if records:
            orders_collection.insert_many(records)

        return jsonify({"message": f"Inserted {len(records)} records into database"}), 200
    except Exception as e:
        return jsonify({"message": f"Error importing dataset: {str(e)}"}), 500

# ---------------- Helper: build customer-product matrix ----------------
def build_customer_product_matrix():
    orders = list(orders_collection.find({}, {"_id": 0}))
    if not orders:
        return None, [], None

    df = pd.DataFrame(orders)
    product_cols = [col for col in df.columns if col.lower().startswith("product") or col.lower().startswith("item")]
    if not product_cols:
        return None, [], df

    df['all_products'] = df[product_cols].apply(
        lambda r: [str(x).strip() for x in r if x and str(x).upper() != "NULL"], axis=1
    )
    all_products = sorted({p for sublist in df['all_products'] for p in sublist})

    if 'customer_name' not in df.columns:
        candidate = next((c for c in df.columns if c.lower() in ('customer', 'name', 'customer_name')), None)
        df['customer_name'] = df[candidate] if candidate else None

    product_matrix = pd.DataFrame({p: [1 if p in row else 0 for row in df['all_products']] for p in all_products})
    product_matrix['customer_name'] = df['customer_name']
    customer_history = product_matrix.groupby('customer_name').max()
    return customer_history, all_products, df

# ---------------- FP-Growth based recommendation ----------------
def fp_growth_recommend(df_orders, current_order, min_support=0.05):
    # Build one-hot encoded DataFrame
    all_products = sorted({p for sublist in df_orders['all_products'] for p in sublist})
    ohe_df = pd.DataFrame([{p: 1 if p in row else 0 for p in all_products} for row in df_orders['all_products']])
    
    # Run FP-Growth
    freq_items = fpgrowth(ohe_df, min_support=min_support, use_colnames=True)
    freq_items = freq_items.sort_values('support', ascending=False)

    # Collect candidate products based on current order
    candidates = set()
    for products in freq_items['itemsets']:
        if any(p in products for p in current_order):
            candidates.update(products)
    candidates -= set(current_order)
    return list(candidates)

# ---------------- Recommendations API ----------------
@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        data = request.json or {}
        customer_name = str(data.get("customer_name", "")).strip()
        current_order = [str(p).strip() for p in data.get("current_order", []) if p]

        customer_history, all_products, df_orders = build_customer_product_matrix()
        if customer_history is None or len(all_products) == 0:
            return jsonify({"missed_products": []}), 200

        # Map product names to prices
        product_prices = {}
        if df_orders is not None:
            for p in all_products:
                row = df_orders[df_orders['all_products'].apply(lambda lst: p in lst)]
                if not row.empty and 'price' in row.columns:
                    product_prices[p] = float(row['price'].iloc[0])
                else:
                    product_prices[p] = 50.0
        else:
            product_prices = {p: 50.0 for p in all_products}

        # FP-Growth candidates
        fp_candidates = fp_growth_recommend(df_orders, current_order)

        # Collaborative filtering fallback
        if customer_name in customer_history.index:
            X = customer_history.fillna(0).astype(float)
            if len(X) > 1:
                n_neighbors = min(5, len(X))
                knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
                knn.fit(X)
                customer_vector = X.loc[[customer_name]]
                distances, indices = knn.kneighbors(customer_vector)
                distances = distances.flatten()
                indices = indices.flatten()
                eps = 1e-6
                weights = 1.0 / (distances + eps)
                neighbor_rows = X.iloc[indices]
                weighted_scores = (neighbor_rows.T * weights).T.sum(axis=0)

                user_history = X.loc[customer_name]
                user_products = [p for p in X.columns if int(user_history[p]) == 1]
                candidate_products = [p for p in user_products if p not in current_order]
                candidate_products = list(set(candidate_products) | set(fp_candidates))
                candidate_scores = {p: float(weighted_scores.get(p, 0.0)) for p in candidate_products}
                ranked = [p for p, s in sorted(candidate_scores.items(), key=lambda x: -x[1])]
                missed = ranked[:5]
                return jsonify({"missed_products": [{"name": p, "price": product_prices.get(p,50)} for p in missed]}), 200

        # For new customers or fallback -> top popular + FP-growth
        popular_scores = {p: df_orders['all_products'].apply(lambda lst: 1 if p in lst else 0).sum() for p in all_products}
        popular_sorted = [p for p, s in sorted(popular_scores.items(), key=lambda x: -x[1]) if p not in current_order]
        candidates = list(dict.fromkeys(fp_candidates + popular_sorted))
        return jsonify({"missed_products": [{"name": p, "price": product_prices.get(p,50)} for p in candidates[:5]]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
