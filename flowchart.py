# turbo_runner_clean_small_boxes.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Draw small box
def draw_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25",
        linewidth=1.5, edgecolor="black",
        facecolor="#c9e9ff"
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, weight="bold", wrap=True)

# Draw arrow
def draw_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=1.4, color="black"
    ))

# Create figure (tall + plenty of space)
fig, ax = plt.subplots(figsize=(10, 22), dpi=200)
ax.set_xlim(0, 20)
ax.set_ylim(0, 40)
ax.axis("off")

# -------- MAIN FLOW (small boxes) --------
boxes = [
    (6, 37, 8, 1.2, "Run Start"),
    (6, 34, 8, 1.2, "Learn controls & obstacles"),
    (6, 31, 8, 1.2, "Enter main running track"),
    (6, 28, 8, 1.2, "Primary objective challenge"),
    (6, 25, 8, 1.2, "Speed increase phase"),
    (6, 22, 8, 1.2, "High-speed final challenge"),
    (6, 19, 8, 1.2, "Run End"),
    (6, 16, 8, 1.4, "Score summary / retry / main menu")
]

# Draw main boxes
for b in boxes:
    draw_box(ax, *b)

# Vertical arrows
for i in range(len(boxes) - 1):
    x = boxes[i][0] + boxes[i][2]/2
    draw_arrow(ax, x, boxes[i][1], x, boxes[i+1][1] + boxes[i+1][3])

# -------- LEFT SIDE SMALL BOXES --------
opt = (1, 27, 4, 1.2, "Optional\ntasks")
sec = (1, 23.5, 4, 1.2, "Secondary\nrewards")

draw_box(ax, *opt)
draw_box(ax, *sec)

# Optional → Final Challenge
draw_arrow(ax,
           opt[0] + opt[2], opt[1] + opt[3]/2,
           boxes[5][0], boxes[5][1] + boxes[5][3]/2)

# Optional → Secondary rewards
draw_arrow(ax,
           opt[0] + opt[2]/2, opt[1],
           sec[0] + sec[2]/2, sec[1] + sec[3])

# Secondary → Score summary
draw_arrow(ax,
           sec[0] + sec[2], sec[1] + sec[3]/2,
           boxes[-1][0] + 1, boxes[-1][1] + boxes[-1][3]/2)

# Speed increase → Score summary loop
speed = boxes[4]
draw_arrow(ax,
           speed[0] + speed[2], speed[1] + speed[3]/2,
           18, boxes[-1][1] + boxes[-1][3]/2)

plt.title("Turbo Runner — Run Flowchart", fontsize=18, weight="bold")
plt.tight_layout()
plt.savefig("turbo_runner_flowchart_small_boxes.png", dpi=300)
plt.show()

print("Saved: turbo_runner_flowchart_small_boxes.png")
