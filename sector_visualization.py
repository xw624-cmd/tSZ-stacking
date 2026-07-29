import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Arc

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 1.2,
})

# --- Angle convention -------------------------------------------------------
# phi is measured counterclockwise from the UPWARD VERTICAL direction of the
# rotated cutout. Matplotlib's polar convention measures counterclockwise from
# +x, so the single mapping below is the only place the convention enters.
PHI_OFFSET = 90.0


def phi_to_plot(phi_deg):
    """Convert cutout angle phi (CCW from +y) to plot angle (CCW from +x)."""
    return phi_deg + PHI_OFFSET


def ray_to_square(phi_deg, L=1.0):
    theta = np.deg2rad(phi_to_plot(phi_deg))
    c, s = np.cos(theta), np.sin(theta)
    tx = np.inf if np.isclose(c, 0.0) else L / abs(c)
    ty = np.inf if np.isclose(s, 0.0) else L / abs(s)
    return min(tx, ty) * np.array([c, s])


def sector_polygon(phi1, phi2, L=1.0):
    return np.array([[0.0, 0.0], ray_to_square(phi1, L), ray_to_square(phi2, L)])


fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=300)
L = 1.0

major_color = "#771704"
minor_color = "#5b7fa6"
unused_color = "#717171"
frame_color = "#444444"

ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L,
                       facecolor=unused_color, edgecolor="none", zorder=0))

# Sector definitions in phi (the new convention)
major_ranges = [(-15, 15), (165, 195)]
minor_ranges = [(75, 105), (255, 285)]

for phi1, phi2 in major_ranges:
    ax.add_patch(Polygon(sector_polygon(phi1, phi2, L), closed=True,
                         facecolor=major_color, edgecolor=major_color,
                         linewidth=1.0, joinstyle="miter", zorder=2))
for phi1, phi2 in minor_ranges:
    ax.add_patch(Polygon(sector_polygon(phi1, phi2, L), closed=True,
                         facecolor=minor_color, edgecolor=minor_color,
                         linewidth=1.0, joinstyle="miter", zorder=2))

for phi in [b for r in major_ranges + minor_ranges for b in r]:
    p = ray_to_square(phi, L)
    ax.plot([0, p[0]], [0, p[1]], color=frame_color, lw=0.7, zorder=3)

ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False,
                       edgecolor=frame_color, linewidth=0.9, zorder=4))

# --- phi reference: dashed line along phi = 0 (upward vertical), arc to +15 --
ref = ray_to_square(0.0, L) * 0.72
ax.plot([0, ref[0]], [0, ref[1]], color="white", lw=0.7, ls=(0, (4, 3)), zorder=3)
ax.add_patch(Arc((0, 0), 0.34, 0.34, angle=0,
                 theta1=phi_to_plot(0), theta2=phi_to_plot(15),
                 color="white", lw=0.9, zorder=3))
phi_lab = np.deg2rad(phi_to_plot(7.5))
ax.text(0.27 * np.cos(phi_lab), 0.27 * np.sin(phi_lab), r"$\phi$",
        ha="center", va="center", fontsize=12, color="white", zorder=5)

# --- Sector labels, stated in phi -------------------------------------------
# (phi, text, color, radius, dx) -- dx is a small optical-centering nudge in
# data units. mathtext centers the full string including the raised degree
# symbol, which leaves the numerals sitting visibly left of the sector axis;
# dx shifts each label so its ink balances on the axis instead.
label_defs = [
    (0,   r"$0^\circ$",   major_color, 0.82, 0.025),
    (180, r"$180^\circ$", major_color, 0.82, 0.020),
    (90,  r"$90^\circ$",  minor_color, 0.85, 0.013),
    (270, r"$270^\circ$", minor_color, 0.85, 0.013),
]
for phi, label, color, r, dx in label_defs:
    theta = np.deg2rad(phi_to_plot(phi))
    ax.text(r * np.cos(theta) + dx, r * np.sin(theta), label,
            ha="center", va="center", fontsize=11, color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                      edgecolor="none", alpha=0.9))

ax.plot(0, 0, marker="o", ms=2.0, color=frame_color, zorder=6)
ax.set_aspect("equal")
ax.set_xlim(-1.04, 1.04)
ax.set_ylim(-1.04, 1.04)
ax.axis("off")

fig.savefig("oriented_sectors_schematic.pdf", bbox_inches="tight",
            pad_inches=0, facecolor="white")
fig.savefig("oriented_sectors_schematic.png", bbox_inches="tight",
            pad_inches=0, facecolor="white")
