#!/usr/bin/env python3
"""Animate convergence of a cumulative galaxy stack for 11.5 < log10(M*/Msun) <= 12.0.

This script reuses the selection, cache, and rotation functions from
``oriented_stacking_separate_maps.py``. By default it animates the oriented
stack; set STACK_MODE = "unoriented" to animate the unrotated stack instead.
"""

from pathlib import Path
import importlib.util

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PIPELINE_PATH = Path(__file__).with_name("oriented_stacking_separate_maps.py")
OUTPUT_PATH = Path("run_output/stack_convergence_11p5_12p0_oriented.gif")

MASS_LO = 11.5
MASS_HI = 12.0
STACK_MODE = "unoriented"       # "oriented" or "unoriented"
N_FRAMES = 150
FPS = 8
SEED = 42
SHUFFLE_GALAXIES = True
MAX_GALAXIES = None            # e.g. 5000 for a faster demonstration run
COLOR_PERCENTILES = (1.0, 99.0)


def load_pipeline(path: Path):
    """Import the main pipeline from an explicit file path."""
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    spec = importlib.util.spec_from_file_location("oriented_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_checkpoints(n_galaxies: int, n_frames: int) -> np.ndarray:
    """Use dense early sampling and progressively wider late checkpoints."""
    if n_galaxies < 1:
        return np.array([], dtype=int)
    n_frames = max(2, min(int(n_frames), n_galaxies))
    geometric = np.geomspace(1, n_galaxies, n_frames)
    checkpoints = np.unique(np.rint(geometric).astype(int))
    if checkpoints[-1] != n_galaxies:
        checkpoints = np.append(checkpoints, n_galaxies)
    return checkpoints


def cumulative_stack_snapshots(pipeline, h5f: h5py.File, indices: np.ndarray,
                               checkpoints: np.ndarray, stack_mode: str):
    """Build cumulative-mean snapshots at the requested galaxy counts."""
    ny = int(h5f.attrs["ny"])
    nx = int(h5f.attrs["nx"])
    pixscale = 2.0 * pipeline.STAMP_RADIUS_ARCMIN / (ny - 1)
    pa_all = h5f["pa"][:]

    running_sum = np.zeros((ny, nx), dtype=np.float64)
    snapshots = []
    next_frame = 0

    for j, cache_idx in enumerate(indices, start=1):
        source_stamp = np.asarray(h5f["stamps"][cache_idx], dtype=np.float64)
        if stack_mode == "oriented":
            angle_deg = -float(pa_all[cache_idx])
        elif stack_mode == "unoriented":
            angle_deg = 0.0
        else:
            raise ValueError('STACK_MODE must be "oriented" or "unoriented"')

        stamp = pipeline.sample_large_stamp_to_output(
            source_stamp,
            angle_deg,
            ny,
            nx,
            pixscale,
        )
        running_sum += stamp

        if next_frame < len(checkpoints) and j == checkpoints[next_frame]:
            snapshots.append((j, (running_sum / j).copy()))
            next_frame += 1

    return snapshots, pixscale


def make_animation(snapshots, pixscale: float, output_path: Path, stack_mode: str):
    """Render a GIF with one fixed color scale based on the final stack."""
    if not snapshots:
        raise ValueError("No snapshots were generated")

    final_stack = snapshots[-1][1]
    finite = final_stack[np.isfinite(final_stack)]
    if finite.size == 0:
        raise ValueError("Final stack contains no finite pixels")

    lo, hi = np.nanpercentile(finite, COLOR_PERCENTILES)
    vabs = max(abs(float(lo)), abs(float(hi)), 1e-30)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    ny, nx = final_stack.shape
    extent = [
        -0.5 * nx * pixscale,
        0.5 * nx * pixscale,
        -0.5 * ny * pixscale,
        0.5 * ny * pixscale,
    ]

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    image = ax.imshow(
        snapshots[0][1],
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
        extent=extent,
        animated=True,
    )
    ax.set_xlabel(r"$x\ [{\rm arcmin}]$")
    ax.set_ylabel(r"$y\ [{\rm arcmin}]$")
    title = ax.set_title("")
    count_text = ax.text(
        0.04,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"Compton $y$")

    label = "Oriented" if stack_mode == "oriented" else "Unoriented"

    def update(frame_index):
        n_used, stack = snapshots[frame_index]
        image.set_data(stack)
        count_text.set_text(rf"$N_{{\rm stacked}}={n_used:,}$")
        return image, title, count_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=140)
    plt.close(fig)
    print(f"Animation written to: {output_path}")


def main():
    pipeline = load_pipeline(PIPELINE_PATH)

    # This applies the same catalog, redshift, shape, radio, and map-footprint
    # selection as the main pipeline and ensures the stamp cache exists.
    h5f, base_fits_idx, _radio_fits_idx = pipeline.build_selection_and_cache()
    try:
        cache_ids = h5f["fits_idx"][:]
        logm = h5f["logm"][:]
        stamp_valid = h5f["stamp_valid"][:]

        selected = (
            np.isin(cache_ids, base_fits_idx)
            & stamp_valid
            & np.isfinite(logm)
            & (logm > MASS_LO)
            & (logm <= MASS_HI)
        )
        indices = np.flatnonzero(selected)
        if indices.size == 0:
            raise RuntimeError("No valid galaxies satisfy the animation selection")

        rng = np.random.default_rng(SEED)
        if SHUFFLE_GALAXIES:
            indices = rng.permutation(indices)
        if MAX_GALAXIES is not None:
            indices = indices[: int(MAX_GALAXIES)]

        checkpoints = frame_checkpoints(len(indices), N_FRAMES)
        print(f"Selected {len(indices):,} galaxies; generating {len(checkpoints)} frames")
        snapshots, pixscale = cumulative_stack_snapshots(
            pipeline,
            h5f,
            indices,
            checkpoints,
            STACK_MODE,
        )
    finally:
        h5f.close()

    make_animation(snapshots, pixscale, OUTPUT_PATH, STACK_MODE)


if __name__ == "__main__":
    main()
