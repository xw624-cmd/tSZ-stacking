from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrow

# ---------------------------------------------------------------------
# Horizontal, side-by-side version of the stacking schematic.
# ---------------------------------------------------------------------

DPI = 100
FIG_W, FIG_H = 1700, 520

output_dir = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path("/mnt/data")
)

fig = plt.figure(
    figsize=(FIG_W / DPI, FIG_H / DPI),
    dpi=DPI,
    facecolor="white",
)

# Colors and line styles
face = "#90D5FF"
edge = "#4D6DAA"
panel_edge = "#808080"

source_alpha = 0.95
stack_alpha = 0.28
ellipse_lw = 1.8

# Each panel gets its own axes. The two panels are arranged horizontally.
left_ax = fig.add_axes([0.025, 0.06, 0.46, 0.88])
right_ax = fig.add_axes([0.515, 0.06, 0.46, 0.88])

for ax in (left_ax, right_ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel border
    ax.add_patch(
        Rectangle(
            (0.02, 0.04),
            0.96,
            0.78,
            fill=False,
            edgecolor=panel_edge,
            linewidth=2.0,
        )
    )

title_kw = dict(
    ha="center",
    va="center",
    fontsize=34,
    fontweight="bold",
    fontfamily="DejaVu Sans",
    color="black",
)

label_kw = dict(
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    fontfamily="DejaVu Sans",
    color="black",
    linespacing=0.90,
)

left_ax.text(0.50, 0.91, "Unoriented Stacking", **title_kw)
right_ax.text(0.50, 0.91, "Oriented Stacking", **title_kw)

# Relative locations of the five input galaxies within each panel
galaxy_centers = np.array([
    [0.10, 0.61],
    [0.34, 0.64],
    [0.22, 0.48],
    [0.31, 0.29],
    [0.18, 0.18],
])

# Ellipse dimensions in axes-coordinate units
major = np.array([0.082, 0.087, 0.096, 0.085, 0.098])
minor = np.array([0.043, 0.043, 0.045, 0.042, 0.046])

angles_unoriented = np.array([28.4, 40.9, -35.7, 62.8, -18.4])
angles_oriented = np.full(5, 90.0)


def add_galaxies(ax, centers, angles, alpha):
    """Draw a set of elliptical galaxies on one panel."""
    for (x, y), a, b, angle in zip(centers, major, minor, angles):
        ax.add_patch(
            Ellipse(
                (x, y),
                width=a,
                height=b,
                angle=angle,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=ellipse_lw,
                alpha=alpha,
            )
        )


def add_arrow(ax):
    """Draw the central stacking arrow."""
    ax.add_patch(
        FancyArrow(
            0.44,
            0.40,
            0.095,
            0.0,
            width=0.026,
            head_width=0.065,
            head_length=0.038,
            length_includes_head=True,
            transform=ax.transAxes,
            facecolor="black",
            edgecolor="black",
        )
    )


# Input galaxies
add_galaxies(left_ax, galaxy_centers, angles_unoriented, source_alpha)
add_galaxies(right_ax, galaxy_centers, angles_oriented, source_alpha)

# Text and arrows
left_ax.text(0.49, 0.55, "Stack galaxies\nwithout rotation", **label_kw)
right_ax.text(0.49, 0.55, "Stack galaxies\nwith rotation", **label_kw)

add_arrow(left_ax)
add_arrow(right_ax)

# Slight offsets make the individual stacked layers visible
stack_centers = np.array([
    [0.70 + 0.014 * i, 0.48 - 0.018 * i]
    for i in range(5)
])

add_galaxies(left_ax, stack_centers, angles_unoriented, stack_alpha)
add_galaxies(right_ax, stack_centers, angles_oriented, stack_alpha)

png_path = output_dir / "galaxy_stacking_schematic_horizontal.png"
pdf_path = output_dir / "galaxy_stacking_schematic_horizontal.pdf"

fig.savefig(
    png_path,
    dpi=DPI,
    facecolor="white",
    edgecolor="none",
)
fig.savefig(
    pdf_path,
    facecolor="white",
    edgecolor="none",
)
plt.close(fig)
