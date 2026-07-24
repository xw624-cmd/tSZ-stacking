from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon

# ---------------------------------------------------------------------
# Reconstructed version of the "Unoriented / Oriented Stacking" figure.
# The coordinates are in pixels so that the output closely matches the
# original 917 x 933 PNG.
# ---------------------------------------------------------------------

FIG_W, FIG_H = 917, 933
DPI = 100

fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(FIG_H, 0)  # Image-like coordinates: y increases downward
ax.axis("off")

# Styling
face = "#90D5FF"
edge = "#4D6DAA"
panel_edge = "#808080"

source_alpha = 0.95
stack_alpha = 0.28
ellipse_lw = 1.8

# Panel outlines
ax.add_patch(Rectangle((67, 77), 783, 335, fill=False,
                       edgecolor=panel_edge, linewidth=2.0))
ax.add_patch(Rectangle((67, 502), 783, 336, fill=False,
                       edgecolor=panel_edge, linewidth=2.0))

# Titles
title_kw = dict(
    ha="center",
    va="center",
    fontsize=40,
    fontweight="bold",
    fontfamily="DejaVu Sans",
    color="black",
)
ax.text(458.5, 39, "Unoriented Stacking", **title_kw)
ax.text(458.5, 465, "Oriented Stacking", **title_kw)

# Five galaxies: centers and dimensions reconstructed from the PNG
centers_top = np.array([
    [113.9, 163.5],
    [301.8, 153.4],
    [207.9, 220.4],
    [278.3, 307.5],
    [199.9, 357.8],
])

panel_shift = 425.6
centers_bottom = centers_top + np.array([0.0, panel_shift])

major = np.array([66.2, 70.1, 77.0, 68.7, 78.8])
minor = np.array([34.5, 34.4, 36.2, 33.3, 37.2])

# Matplotlib angles; positive angles look counterclockwise in the final figure.
angles_unoriented = np.array([28.4, 40.9, -35.7, 62.8, -18.4])
angles_oriented = np.full(5, 90.0)

def add_galaxies(centers, angles, alpha):
    for (x, y), a, b, angle in zip(centers, major, minor, angles):
        ax.add_patch(Ellipse(
            (x, y),
            width=a,
            height=b,
            angle=angle,
            facecolor=face,
            edgecolor=edge,
            linewidth=ellipse_lw,
            alpha=alpha,
        ))

# Input galaxies
add_galaxies(centers_top, angles_unoriented, source_alpha)
add_galaxies(centers_bottom, angles_oriented, source_alpha)

# Labels
label_kw = dict(
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    fontfamily="DejaVu Sans",
    color="black",
    linespacing=0.90,
)
ax.text(419, 198, "Stack galaxies\nwithout rotation", **label_kw)
ax.text(419, 624, "Stack galaxies\nwith rotation", **label_kw)

# Arrows, drawn as polygons for a close match to the original
def add_arrow(y_center):
    vertices = np.array([
        [407, y_center - 7],
        [443, y_center - 7],
        [443, y_center - 14],
        [471, y_center],
        [443, y_center + 14],
        [443, y_center + 7],
        [407, y_center + 7],
    ])
    ax.add_patch(Polygon(vertices, closed=True, facecolor="black", edgecolor="black"))

add_arrow(251)
add_arrow(677)

# Stacked galaxies. A slight diagonal offset makes all five layers visible.
stack_x0, stack_y0 = 618.35, 214.39
dx, dy = 10.31, 8.25
stack_centers_top = np.array([
    [stack_x0 + i * dx, stack_y0 + i * dy] for i in range(5)
])
stack_centers_bottom = stack_centers_top + np.array([0.0, panel_shift])

add_galaxies(stack_centers_top, angles_unoriented, stack_alpha)
add_galaxies(stack_centers_bottom, angles_oriented, stack_alpha)

# Exact-size PNG; PDF is also produced for publication use.
output_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path("/mnt/data")
fig.savefig(output_dir / "galaxy_stacking_schematic_reconstructed.png", dpi=DPI,
            facecolor="white", edgecolor="none")
fig.savefig(output_dir / "galaxy_stacking_schematic_reconstructed.pdf",
            facecolor="white", edgecolor="none")
plt.close(fig)
