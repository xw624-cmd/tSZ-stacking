from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle


# ---------------------------------------------------------------------
# Reconstructed "Unoriented / Oriented Stacking" schematic.
#
# The figure is initially drawn using pixel-like coordinates, then cropped
# tightly around all visible objects when saved.
# ---------------------------------------------------------------------

FIG_W = 917
FIG_H = 933
DPI = 100

fig = plt.figure(
    figsize=(FIG_W / DPI, FIG_H / DPI),
    dpi=DPI,
    facecolor="white",
)

ax = fig.add_axes([0, 0, 1, 1])

# Use image-style coordinates, where y increases downward.
ax.set_xlim(0, FIG_W)
ax.set_ylim(FIG_H, 0)
ax.axis("off")


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

face_color = "#90D5FF"
edge_color = "#4D6DAA"
panel_edge_color = "#808080"

source_alpha = 0.95
stack_alpha = 0.28
ellipse_linewidth = 1.8


# ---------------------------------------------------------------------
# Panel outlines
# ---------------------------------------------------------------------

ax.add_patch(
    Rectangle(
        (67, 77),
        783,
        335,
        fill=False,
        edgecolor=panel_edge_color,
        linewidth=2.0,
    )
)

ax.add_patch(
    Rectangle(
        (67, 502),
        783,
        336,
        fill=False,
        edgecolor=panel_edge_color,
        linewidth=2.0,
    )
)


# ---------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------

title_style = {
    "ha": "center",
    "va": "center",
    "fontsize": 40,
    "fontweight": "bold",
    "fontfamily": "DejaVu Sans",
    "color": "black",
}

ax.text(
    458.5,
    39,
    "Unoriented Stacking",
    **title_style,
)

ax.text(
    458.5,
    465,
    "Oriented Stacking",
    **title_style,
)


# ---------------------------------------------------------------------
# Galaxy positions and shapes
# ---------------------------------------------------------------------

centers_top = np.array(
    [
        [113.9, 163.5],
        [301.8, 153.4],
        [207.9, 220.4],
        [278.3, 307.5],
        [199.9, 357.8],
    ]
)

panel_shift = 425.6

centers_bottom = centers_top + np.array([0.0, panel_shift])

major_axes = np.array([66.2, 70.1, 77.0, 68.7, 78.8])
minor_axes = np.array([34.5, 34.4, 36.2, 33.3, 37.2])

# Matplotlib ellipse angles are measured counterclockwise.
angles_unoriented = np.array([28.4, 40.9, -35.7, 62.8, -18.4])
angles_oriented = np.full(5, 90.0)


def add_galaxies(centers, angles, alpha):
    """Add a collection of elliptical galaxy shapes to the axes."""

    for center, major, minor, angle in zip(
        centers,
        major_axes,
        minor_axes,
        angles,
    ):
        x, y = center

        galaxy = Ellipse(
            xy=(x, y),
            width=major,
            height=minor,
            angle=angle,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=ellipse_linewidth,
            alpha=alpha,
        )

        ax.add_patch(galaxy)


# Input galaxies
add_galaxies(
    centers=centers_top,
    angles=angles_unoriented,
    alpha=source_alpha,
)

add_galaxies(
    centers=centers_bottom,
    angles=angles_oriented,
    alpha=source_alpha,
)


# ---------------------------------------------------------------------
# Explanatory labels
# ---------------------------------------------------------------------

label_style = {
    "ha": "center",
    "va": "center",
    "fontsize": 20,
    "fontweight": "bold",
    "fontfamily": "DejaVu Sans",
    "color": "black",
    "linespacing": 0.90,
}

ax.text(
    419,
    218,
    "Stack galaxies\nwithout rotation",
    **label_style,
)

ax.text(
    419,
    644,
    "Stack galaxies\nwith rotation",
    **label_style,
)


# ---------------------------------------------------------------------
# Arrows
# ---------------------------------------------------------------------

def add_arrow(y_center):
    """Add a right-pointing polygon arrow."""

    vertices = np.array(
        [
            [407, y_center - 7],
            [443, y_center - 7],
            [443, y_center - 14],
            [471, y_center],
            [443, y_center + 14],
            [443, y_center + 7],
            [407, y_center + 7],
        ]
    )

    arrow = Polygon(
        vertices,
        closed=True,
        facecolor="black",
        edgecolor="black",
    )

    ax.add_patch(arrow)


add_arrow(261)
add_arrow(687)


# ---------------------------------------------------------------------
# Stacked galaxies
# ---------------------------------------------------------------------

stack_x0 = 618.35
stack_y0 = 214.39

stack_dx = 10.31
stack_dy = 8.25

stack_centers_top = np.array(
    [
        [
            stack_x0 + i * stack_dx,
            stack_y0 + i * stack_dy,
        ]
        for i in range(5)
    ]
)

stack_centers_bottom = stack_centers_top + np.array(
    [0.0, panel_shift]
)

add_galaxies(
    centers=stack_centers_top,
    angles=angles_unoriented,
    alpha=stack_alpha,
)

add_galaxies(
    centers=stack_centers_bottom,
    angles=angles_oriented,
    alpha=stack_alpha,
)


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

if "__file__" in globals():
    output_dir = Path(__file__).resolve().parent
else:
    output_dir = Path.cwd()

png_path = output_dir / "galaxy_stacking_schematic_reconstructed.png"
pdf_path = output_dir / "galaxy_stacking_schematic_reconstructed.pdf"


# ---------------------------------------------------------------------
# Calculate a tight bounding box around all visible objects.
#
# This avoids the extra whitespace that can remain when using only
# tight_layout() or bbox_inches="tight".
# ---------------------------------------------------------------------

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

visible_artists = [
    *ax.patches,
    *ax.texts,
]

artist_bounding_boxes = [
    artist.get_window_extent(renderer=renderer)
    for artist in visible_artists
    if artist.get_visible()
]

tight_bbox_pixels = transforms.Bbox.union(artist_bounding_boxes)

# Small border around the outermost visible objects.
# Set this to 0 for no border at all.
padding_pixels = 4

tight_bbox_pixels = transforms.Bbox.from_extents(
    tight_bbox_pixels.x0 - padding_pixels,
    tight_bbox_pixels.y0 - padding_pixels,
    tight_bbox_pixels.x1 + padding_pixels,
    tight_bbox_pixels.y1 + padding_pixels,
)

# savefig expects the bounding box in inches, not pixels.
tight_bbox_inches = tight_bbox_pixels.transformed(
    fig.dpi_scale_trans.inverted()
)


# ---------------------------------------------------------------------
# Save tightly cropped PNG and PDF files
# ---------------------------------------------------------------------

fig.savefig(
    png_path,
    dpi=DPI,
    facecolor="white",
    edgecolor="none",
    bbox_inches=tight_bbox_inches,
    pad_inches=0,
)

fig.savefig(
    pdf_path,
    facecolor="white",
    edgecolor="none",
    bbox_inches=tight_bbox_inches,
    pad_inches=0,
)

plt.close(fig)

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
