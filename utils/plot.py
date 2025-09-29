import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 10))

def draw_block(ax, text, xy, size=(2.5, 0.8), color="skyblue"):
    rect = mpatches.FancyBboxPatch(xy, size[0], size[1],
                                   boxstyle="round,pad=0.1",
                                   edgecolor="black", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + size[0]/2, xy[1] + size[1]/2, text,
            ha="center", va="center", fontsize=9)

def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xycoords="data", 
                xytext=start, textcoords="data",
                arrowprops=dict(arrowstyle="->", lw=1.2))

# Define blocks in sequence
blocks = [
    ("Input\n784", "input"),
    ("Linear\n512", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n256", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n128", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n64", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n32", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n16", "linear"),
    ("ReLU", "relu"),
    ("Dropout 0.3", "dropout"),
    ("Linear\n10", "output")
]

# Colors
colors = {
    "input": "lightgray",
    "linear": "skyblue",
    "relu": "khaki",
    "dropout": "lightcoral",
    "output": "lightgreen"
}

# Y positions
y_positions = list(range(len(blocks)*2, -2, -2))

# Draw blocks
for i, (label, btype) in enumerate(blocks):
    y = y_positions[i]
    draw_block(ax, label, (0, y), size=(3, 1), color=colors[btype])
    if i > 0:
        draw_arrow(ax, (1.5, y_positions[i-1]), (1.5, y+1))

ax.set_xlim(-1, 5)
ax.set_ylim(-2, y_positions[0]+2)
ax.axis("off")
plt.show()
