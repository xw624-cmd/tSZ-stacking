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

def ray_to_square(theta_deg, L=1.0):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    tx = np.inf if np.isclose(c, 0.0) else L / abs(c)
    ty = np.inf if np.isclose(s, 0.0) else L / abs(s)
    t = min(tx, ty)
    return np.array([t * c, t * s])

def sector_polygon(theta1, theta2, L=1.0):
    p1 = ray_to_square(theta1, L)
    p2 = ray_to_square(theta2, L)
    return np.array([[0.0, 0.0], p1, p2])

fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=300)

L = 1.0
major_color = "#c1666b"
minor_color = "#5b7fa6"
unused_color = "#e6e6e6"
frame_color = "#444444"

ax.add_patch(Rectangle((-L, -L), 2*L, 2*L, facecolor=unused_color, edgecolor="none", zorder=0))

major_ranges = [(75, 105), (255, 285)]
minor_ranges = [(-15, 15), (165, 195)]

for th1, th2 in major_ranges:
    ax.add_patch(Polygon(sector_polygon(th1, th2, L), closed=True,
                          facecolor=major_color, edgecolor=major_color,
                          linewidth=1.0, joinstyle="miter", zorder=2))

for th1, th2 in minor_ranges:
    ax.add_patch(Polygon(sector_polygon(th1, th2, L), closed=True,
                          facecolor=minor_color, edgecolor=minor_color,
                          linewidth=1.0, joinstyle="miter", zorder=2))

boundary_angles = [75, 105, 255, 285, -15, 15, 165, 195]
for ang in boundary_angles:
    p = ray_to_square(ang, L)
    ax.plot([0, p[0]], [0, p[1]], color=frame_color, lw=0.7, zorder=3)

ax.add_patch(Rectangle((-L, -L), 2*L, 2*L, fill=False, edgecolor=frame_color,
                        linewidth=0.9, zorder=4))

ax.plot([0, 0.9*L], [0, 0], color="white", lw=0.7, ls=(0, (4, 3)), zorder=3)

arc = Arc((0, 0), 0.34, 0.34, angle=0, theta1=0, theta2=15, color="white", lw=0.9, zorder=3)
ax.add_patch(arc)
phi_ang = np.deg2rad(7.5)
phi_r = 0.27
ax.text(phi_r*np.cos(phi_ang), phi_r*np.sin(phi_ang), r"$\phi$",
        ha="center", va="center", fontsize=12, color="white")

label_defs = [
    (90, "90°", major_color, 0.55),
    (270, "270°", major_color, 0.55),
    (0, "0°", minor_color, 0.85),
    (180, "180°", minor_color, 0.85),
]
for ang, label, color, r in label_defs:
    theta = np.deg2rad(ang)
    x, y = r*np.cos(theta), r*np.sin(theta)
    ax.text(x, y, label, ha="center", va="center", fontsize=11,
            color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none", alpha=0.9))

ax.plot(0, 0, marker='o', ms=2.0, color=frame_color, zorder=6)

ax.set_aspect("equal")
ax.set_xlim(-1.04, 1.04)
ax.set_ylim(-1.04, 1.04)
ax.axis("off")

fig.savefig("oriented_sectors_schematic.pdf", bbox_inches="tight", pad_inches=0, facecolor="white")
plt.show()
