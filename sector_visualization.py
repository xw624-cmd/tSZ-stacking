# The major and minor sector visualization

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

fig, ax = plt.subplots(figsize=(6, 6))

top = [(-1, 1), (1, 1), (0, 0)]
bottom = [(-1, -1), (1, -1), (0, 0)]
left = [(-1, 1), (-1, -1), (0, 0)]
right = [(1, 1), (1, -1), (0, 0)]

major_color = "#e58a8a"
minor_color = "#9eb8e6"

ax.add_patch(Polygon(top, closed=True, facecolor=major_color, edgecolor="black", linewidth=1.5))
ax.add_patch(Polygon(bottom, closed=True, facecolor=major_color, edgecolor="black", linewidth=1.5))
ax.add_patch(Polygon(left, closed=True, facecolor=minor_color, edgecolor="black", linewidth=1.5))
ax.add_patch(Polygon(right, closed=True, facecolor=minor_color, edgecolor="black", linewidth=1.5))

ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="black", linewidth=1.5))
ax.plot([-1, 1], [1, -1], color="black", linewidth=1.5)
ax.plot([1, -1], [1, -1], color="black", linewidth=1.5)

ax.text(0, 0.55, "Major axis", ha="center", va="center", fontsize=18)
ax.text(0, -0.55, "Major axis", ha="center", va="center", fontsize=18)
ax.text(-0.58, 0, "Minor axis", ha="center", va="center", fontsize=18)
ax.text(0.58, 0, "Minor axis", ha="center", va="center", fontsize=18)

ax.set_aspect("equal")
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-1.12, 1.12)
ax.axis("off")

fig.savefig("oriented_sectors_schematic.pdf", bbox_inches="tight")
plt.show()
