#!/usr/bin/env python3
"""
Oriented tSZ stacking pipeline.

This code produces five summary PDFs:

1. summary_oriented_full_stack_2x3.pdf
2. summary_stellar_age_stack_4x3.pdf
3. summary_oriented_full_stack_cap_profiles_1x3.pdf
4. summary_stellar_age_cap_profiles_1x3.pdf
5. summary_radio_full_stack_1x3.pdf

As well as these histogram/weight diagnostic PDFs:
1. summary_stellar_age_hist_mass_1x3.pdf
2. summary_stellar_age_hist_mass_weighted_1x3.pdf
3. summary_stellar_age_mass_weights_1x3.pdf
4. summary_stellar_age_hist_ebv_1x3.pdf
"""

import os
import time
import json
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from astropy.io import fits
from pixell import enmap, reproject, utils
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline
from scipy import spatial
import h5py

os.environ["PATH"] = (
    "/usr/local/texlive/2025/bin/universal-darwin:"
    "/Library/TeX/texbin:"
    + os.environ.get("PATH", "")
)

# Use the local LaTeX installation for all text in the figures.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
})


# ============================================================
# CONFIG
# ============================================================
FIREFLY_PATH = "/Users/jerrywang/Documents/Battaglia_research/Project_tsz_stacking/sdss_firefly-26.fits"
PHOTO_PATH   = "/Users/jerrywang/Documents/Battaglia_research/Project_tsz_stacking/photoPosPlate-dr17.fits"
TSZ_MAP_PATH = "/Users/jerrywang/Documents/Battaglia_research/Project_tsz_stacking/act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.2_24.0.fits"
FIRST_PATH   = "/Users/jerrywang/Documents/Battaglia_research/Project_tsz_stacking/first_14dec17.fits"

# Photo shape settings. Keep this on for oriented stacks.
USE_PHOTO_SHAPE      = True
PHOTO_MATCH_ARCSEC   = 1.0
PHOTO_FRACDEV_THRESH = 0.5
PHOTO_TYPE_GALAXY    = 3
BA_MAX               = 1.0
INTERPOLATION_ORDER = 1

# Orientation and sector settings.
WEDGE_HALF_DEG   = 45.0
MAJOR_AXIS_ANGLE = 90.0
MINOR_AXIS_ANGLE = 0.0

# CAP settings.
CAP_N_AP          = 6
CAP_AP_MIN_ARCMIN = 1.0
CAP_AP_MAX_ARCMIN = 6.0
CAP_RADII_ARCMIN  = np.linspace(CAP_AP_MIN_ARCMIN, CAP_AP_MAX_ARCMIN, CAP_N_AP)

# CAP units. Pixell/reproject handles the sky projection when extracting
# each thumbnail. After extraction, the local cutout is treated as a flat
# angular grid, so CAP values are reported in y arcmin^2.
CAP_UNIT_LABEL = r"$\mathrm{CAP}\ [y\,\mathrm{arcmin}^{2}]$"

# Catalog cuts.
MASS_BINS = [
    (11.0, 11.4),
    (11.4, 11.7),
    (11.7, 12.0),
]
CACHE_LOG_MASS_MIN = min(lo for lo, hi in MASS_BINS)
CACHE_LOG_MASS_MAX = max(hi for lo, hi in MASS_BINS)

USE_REDSHIFT_CUT = True
Z_MIN = 0.2
Z_MAX = 0.6

USE_EBV_CUT = True
EBV_MAX = 0.1

RADIO_ONLY = False
EXCLUDE_RADIO = True
FIRST_MATCH_ARCSEC = 60.0
SAFETY = 1 

# Split settings.
SPLIT_REMOVE_MIDDLE_PCT = 30.0
SPLIT_TAIL_PCT = 0.5 * (100.0 - SPLIT_REMOVE_MIDDLE_PCT)
ACTIVE_SPLIT_SCHEMES = ["mass_age", "light_age"]

SCHEME_META = {
    "mass_age": {
        "field": "age_massW",
        "label": "Mass weighted stellar age",
        "lo": "Young mass weighted age",
        "hi": "Old mass weighted age",
    },
    "light_age": {
        "field": "age_lightW",
        "label": "Light weighted stellar age",
        "lo": "Young light weighted age",
        "hi": "Old light weighted age",
    },
}

# Stamp and cache settings.
# STAMP_RADIUS_ARCMIN is the final plotted/CAP stamp half width.
# STAMP_SOURCE_RADIUS_ARCMIN is the larger source thumbnail half width used
# for oriented coordinate-remapping rotations.  sqrt(2) is the geometric
# no-edge-loss radius for rotating a square output grid by any angle. REMEBER to round up to the nearest 0.25 multiple
STAMP_RADIUS_ARCMIN = 15.0
STAMP_SOURCE_RADIUS_ARCMIN = 21.25
CACHE_DIR  = "./stamp_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "stamps.h5")
EXTRACTION_BATCH_LOG_EVERY = 2000

# Unweighted bootstrap errors for CAP points. This is not mass weighting.
RUN_BOOTSTRAP = True
N_BOOT = 10000
SEED = 42

# Output folder for the four requested PDFs.
SUMMARY_DIR = "./mass_weighted_oriented_runs_3bins/summary"

# Radio-only summary stack row. Use unoriented by default in this pipeline.
RADIO_STACK_KEY = "stack_unori"

# Split-stack mass weighting.
# This affects only the age-split stacks, not the full oriented stacks or radio stacks.
USE_SPLIT_MASS_WEIGHTS = True

# Single source of truth for the mass-weighting bin count.
# Change this one number only.
MASS_WEIGHT_N_BINS = 20
SPLIT_MASS_WEIGHT_N_BINS = MASS_WEIGHT_N_BINS

# Histogram settings.
# Force all diagnostic histograms to use the same bin count as the mass-weighting bins.
HIST_N_BINS = MASS_WEIGHT_N_BINS

# Set this to something like 5.0 if sparse edge bins get dangerously large weights.
SPLIT_MASS_WEIGHT_CLIP = None


HIST_YLIMS = {
    "logm": (0.0, 0.15),
    "z":    (0.0, 0.4),
    "EBV":  (0.0, 0.6),
    "age_log": (0.0, 1),
    "ba_selected": (0.0, 0.15),
    "delta_ba": (0.0, 1),
    "pa_folded_diff": (0.0, 1),
}

HIST_AGE_LOG_MIN = 9.2
HIST_AGE_LOG_MAX = 10.25
BA_HIST_MIN = 0.0
BA_HIST_MAX = 1.0
DELTA_BA_HIST_MIN = -0.24
DELTA_BA_HIST_MAX = 0.24
PA_DIFF_HIST_MIN = 0.0
PA_DIFF_HIST_MAX = 90.0


# ============================================================
# PLOTTING AESTHETICS
# ============================================================
# All figure polish knobs are centralized here so later tuning does not
# require digging through the plotting functions.

# colors
STACK_COLOR_PERCENTILE_LOW = 1
STACK_COLOR_PERCENTILE_HIGH = 99

# Global save settings.  pad_inches=0 trims boundary whitespace aggressively.
SAVEFIG_DPI = 220
SAVEFIG_BBOX_INCHES = "tight"
SAVEFIG_PAD_INCHES = 0.0

# Shared stack-image aesthetics.
STACK_PANEL_TITLE_SIZE = 10
STACK_PANEL_TITLE_PAD = 14
STACK_SUPTITLE_SIZE = 15
STACK_AXIS_LABEL_SIZE = 8
STACK_TICK_LABEL_SIZE = 7
STACK_N_LABEL_SIZE = 8
STACK_NO_DATA_SIZE = 9
STACK_ROW_LABEL_SIZE = 11
STACK_XY_TICK_NBINS = 5
STACK_COLORBAR_TICK_NBINS = 7
STACK_COLORBAR_LABEL_SIZE = 11
STACK_COLORBAR_TICK_SIZE = 9
STACK_COLORBAR_WIDTH = 0.018
STACK_COLORBAR_PAD = 0.012
STACK_N_LABEL_X = 0.04
STACK_N_LABEL_Y = 0.94
STACK_N_LABEL_BBOX_ALPHA = 0.75
STACK_N_LABEL_BBOX_PAD = 1.5

# Full-stack summary grid.
STACK_FULL_FIG_WIDTH_PER_COL = 4.0
STACK_FULL_FIG_HEIGHT_PER_ROW = 3.75
STACK_FULL_GRID_LEFT = 0.075
STACK_FULL_GRID_BOTTOM = 0.08
STACK_FULL_GRID_TOP = 0.86
STACK_FULL_ROW_LABEL_X = -0.18
STACK_FULL_CBAR_BOTTOM = 0.18
STACK_FULL_CBAR_HEIGHT = 0.62
STACK_FULL_SUPTITLE_Y = 0.94

# Stellar-age stack grid.
STACK_AGE_FIG_WIDTH_PER_COL = 3.75
STACK_AGE_FIG_HEIGHT_PER_ROW = 3.35
STACK_AGE_GRID_LEFT = 0.105
STACK_AGE_GRID_BOTTOM = 0.06
STACK_AGE_GRID_TOP = 0.90
STACK_AGE_ROW_LABEL_X = -0.20
STACK_AGE_CBAR_BOTTOM = 0.16
STACK_AGE_CBAR_HEIGHT = 0.66
STACK_AGE_SUPTITLE_Y = 0.945

# Radio-stack summary grid.
STACK_RADIO_FIG_WIDTH_PER_COL = 3.5
STACK_RADIO_FIG_HEIGHT = 3.75
STACK_RADIO_GRID_LEFT = 0.075
STACK_RADIO_GRID_BOTTOM = 0.12
STACK_RADIO_GRID_TOP = 0.80
STACK_RADIO_CBAR_BOTTOM = 0.20
STACK_RADIO_CBAR_HEIGHT = 0.56
STACK_RADIO_SUPTITLE_Y = 0.965

# CAP-profile aesthetics.
CAP_FIG_WIDTH_PER_COL = 6.35
CAP_FIG_HEIGHT = 5.25
CAP_LEFT = 0.08
CAP_RIGHT = 0.98
CAP_BOTTOM = 0.15
CAP_TOP = 0.80
CAP_WSPACE = 0.0
CAP_PANEL_TITLE_SIZE = 10
CAP_PANEL_TITLE_PAD = 14
CAP_SUPTITLE_SIZE = 15
CAP_SUPTITLE_Y = 0.925
CAP_AXIS_LABEL_SIZE_SECTOR = 16
CAP_AXIS_LABEL_SIZE_AGE = 11
CAP_TICK_LABEL_SIZE = 9
CAP_Y_TICK_NBINS = 10
CAP_LEGEND_LOC_SECTOR = "upper left"
CAP_LEGEND_SIZE_SECTOR = 15
CAP_LEGEND_LOC_AGE = "upper left"
CAP_LEGEND_SIZE_AGE = 5
CAP_ERROR_CAPSIZE = 3
CAP_ERROR_LW = 1.3
CAP_ERROR_MARKER_SIZE = 5
CAP_ZERO_LINE_COLOR = "0.5"
CAP_ZERO_LINE_WIDTH = 0.6
CAP_ZERO_LINE_STYLE = "--"

# Histogram aesthetics.
HIST_FIG_WIDTH_PER_COL = 6.35
HIST_FIG_HEIGHT = 5.25
HIST_LEFT = 0.08
HIST_RIGHT = 0.98
HIST_BOTTOM = 0.15
HIST_TOP = 0.80
HIST_WSPACE = 0.0
HIST_PANEL_TITLE_SIZE = 10
HIST_PANEL_TITLE_PAD = 14
HIST_SUPTITLE_SIZE = 15
HIST_SUPTITLE_Y = 0.925
HIST_AXIS_LABEL_SIZE = 11
HIST_TICK_LABEL_SIZE = 9
HIST_X_TICK_NBINS = 10
HIST_LEGEND_LOC = "upper right"
HIST_LEGEND_SIZE = 7
HIST_LINEWIDTH = 1.8
HIST_ZERO_LINE_COLOR = "0.5"
HIST_ZERO_LINE_WIDTH = 0.6
HIST_ZERO_LINE_STYLE = "--"

# Median annotations for the stellar-age split histograms.
# These affect only the still-enabled age-split histogram PDFs:
#   summary_stellar_age_hist_mass_1x3.pdf
#   summary_stellar_age_hist_ebv_1x3.pdf
HIST_SHOW_MEDIAN_IN_LEGEND = True
HIST_MEDIAN_SCI_PRECISION = 2
HIST_MEDIAN_TEXT = r"{\rm med}"
HIST_AGE_LOG_LEGEND_MEDIAN_IN_YEARS = True

# ============================================================
# SMALL HELPERS
# ============================================================


def _savefig(path):
    """Save figures using centralized trimming settings."""
    plt.savefig(
        path,
        dpi=SAVEFIG_DPI,
        bbox_inches=SAVEFIG_BBOX_INCHES,
        pad_inches=SAVEFIG_PAD_INCHES,
    )


def _fmt_pct(n, d):
    """Format n/d as a percentage string."""
    if d == 0:
        return "nan%"
    return f"{100.0 * n / d:.2f}%"


def _mass_bin_mask(logm, mass_lo, mass_hi):
    """Mass bin mask using nonoverlapping bins: mass_lo < logM <= mass_hi."""
    return np.isfinite(logm) & (logm > mass_lo) & (logm <= mass_hi)


def _mass_bin_mask_for_accounting(logm, mass_lo, mass_hi):
    """Mass bin mask used only for printed accounting."""
    return _mass_bin_mask(logm, mass_lo, mass_hi)

def print_mass_bin_accounting(label, mask, logm, previous_mask=None, indent="  "):
    """
    Print total and per mass bin counts for a Boolean mask.

    If previous_mask is provided, also print how many objects were lost
    relative to that previous cumulative stage.
    """
    mask = np.asarray(mask, dtype=bool)
    logm = np.asarray(logm)
    total = int(np.sum(mask))

    if previous_mask is None:
        print(f"{indent}{label}: {total:,}")
    else:
        previous_mask = np.asarray(previous_mask, dtype=bool)
        previous_total = int(np.sum(previous_mask))
        lost = previous_total - total
        print(
            f"{indent}{label}: {total:,} "
            f"(lost {lost:,} from previous, kept {_fmt_pct(total, previous_total)})"
        )

    for mass_lo, mass_hi in MASS_BINS:
        in_bin = _mass_bin_mask_for_accounting(logm, mass_lo, mass_hi)
        n_bin = int(np.sum(mask & in_bin))

        if previous_mask is None:
            print(f"{indent}  logM ({mass_lo:.1f}, {mass_hi:.1f}]: {n_bin:,}")
        else:
            prev_bin = int(np.sum(previous_mask & in_bin))
            lost_bin = prev_bin - n_bin
            print(
                f"{indent}  logM ({mass_lo:.1f}, {mass_hi:.1f}]: {n_bin:,} "
                f"(lost {lost_bin:,}, kept {_fmt_pct(n_bin, prev_bin)})"
            )

def _cap_to_plot_units(values):
    """CAP values are already in y arcmin^2."""
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _cap_plot_axis_label():
    """CAP axis label for the y arcmin^2 convention."""
    return CAP_UNIT_LABEL


def _cap_pixel_area_arcmin2(pixscale_arcmin):
    """Projected local cutout pixel area in arcmin^2."""
    return float(pixscale_arcmin) ** 2


def _write_cap_area_metadata(h5f, pixscale_arcmin):
    """Store CAP unit metadata in the HDF5 cache."""
    h5f.attrs["cap_pixel_area_arcmin2"] = _cap_pixel_area_arcmin2(pixscale_arcmin)
    h5f.attrs["cap_internal_unit"] = "y_arcmin2"
    h5f.attrs["cap_plot_unit"] = "yarcmin2"
    h5f.attrs["cap_plot_scale_from_internal"] = 1.0
    h5f.flush()


def _scheme_field(sk):
    return SCHEME_META[sk]["field"]


def _scheme_label(sk):
    return SCHEME_META[sk]["label"]


def _lo_label(sk):
    return SCHEME_META[sk]["lo"]


def _hi_label(sk):
    return SCHEME_META[sk]["hi"]


def _centered_pixel_axis(n, pixscale_arcmin):
    """Pixel-center coordinates in arcmin, centered on zero."""
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(pixscale_arcmin)


def sample_large_stamp_to_output(
    source_stamp,
    angle_deg,
    out_ny,
    out_nx,
    out_pixscale_arcmin,
    source_radius_arcmin=STAMP_SOURCE_RADIUS_ARCMIN,
    fill_value=0.0,
):
    """Sample a larger source thumbnail onto the final output grid.

    This is the coordinate-remapping version of rotation, close to the
    oriented_superclustering/ThumbStack pattern:

        1. keep the final output grid fixed,
        2. rotate output coordinates backward into the larger source thumbnail,
        3. bilinearly interpolate the source thumbnail at those coordinates.

    angle_deg follows scipy.ndimage.rotate semantics: positive values rotate
    the image counterclockwise.  The interpolation is linear because
    RectBivariateSpline is used with kx=1, ky=1.
    """
    src = np.asarray(source_stamp, dtype=np.float64)
    src_ny, src_nx = src.shape

    src_pixscale_y = 2.0 * float(source_radius_arcmin) / (src_ny - 1)
    src_pixscale_x = 2.0 * float(source_radius_arcmin) / (src_nx - 1)
    
    y_src = _centered_pixel_axis(src_ny, src_pixscale_y)
    x_src = _centered_pixel_axis(src_nx, src_pixscale_x)

    y_out = _centered_pixel_axis(out_ny, out_pixscale_arcmin)
    x_out = _centered_pixel_axis(out_nx, out_pixscale_arcmin)
    xg_out, yg_out = np.meshgrid(x_out, y_out)

    # Inverse mapping: to create an output image rotated by +angle,
    # sample the input image at coordinates rotated by -angle.
    theta = -np.deg2rad(angle_deg)
    ca = np.cos(theta)
    sa = np.sin(theta)
    xg_src = ca * xg_out - sa * yg_out
    yg_src = sa * xg_out + ca * yg_out

    inside = (
        (yg_src >= y_src[0]) & (yg_src <= y_src[-1])
        & (xg_src >= x_src[0]) & (xg_src <= x_src[-1])
    )

    out = np.full((out_ny, out_nx), fill_value, dtype=np.float64)
    if np.any(inside):
        interp = RectBivariateSpline(y_src, x_src, src, kx=1, ky=1)
        out[inside] = interp(yg_src[inside], xg_src[inside], grid=False)

    return out.astype(np.float64)


def rotate_stamp(stamp, angle_deg):
    """Legacy fallback: rotate a same-size stamp with scipy."""
    return rotate(
        stamp,
        angle_deg,
        reshape=False,
        order=INTERPOLATION_ORDER,
        mode="constant",
        cval=0.0,
    )


def apply_percentile_split(values, remove_middle_pct=30.0):
    """Return low and high tail masks after removing the middle percentage."""
    lo_pct = (100.0 - remove_middle_pct) / 2.0
    finite_idx = np.where(np.isfinite(values))[0]
    n = len(finite_idx)
    if n == 0:
        return (
            np.zeros(len(values), dtype=bool),
            np.zeros(len(values), dtype=bool),
            np.nan,
            np.nan,
        )

    n_tail = int(np.round(n * lo_pct / 100.0))
    n_tail = max(1, min(n_tail, n // 2))

    sorted_idx = finite_idx[np.argsort(values[finite_idx])]
    is_lo = np.zeros(len(values), dtype=bool)
    is_hi = np.zeros(len(values), dtype=bool)
    is_lo[sorted_idx[:n_tail]] = True
    is_hi[sorted_idx[-n_tail:]] = True

    p_lo_val = values[sorted_idx[n_tail - 1]]
    p_hi_val = values[sorted_idx[-n_tail]]
    return is_lo, is_hi, p_lo_val, p_hi_val


def mean_profile_and_covariance(profiles, weights=None, seed=SEED):
    """
    Mean CAP profile plus bootstrap covariance on the mean.

    If weights is None, this is the original unweighted calculation.
    If weights is supplied, the mean is a weighted mean and the bootstrap
    resamples galaxies with their corresponding weights.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    n_ap = profiles.shape[1]

    valid = np.all(np.isfinite(profiles), axis=1)

    if weights is None:
        w_all = np.ones(profiles.shape[0], dtype=np.float64)
    else:
        w_all = np.asarray(weights, dtype=np.float64)
        if w_all.shape[0] != profiles.shape[0]:
            raise ValueError("weights must have the same length as profiles")
        valid &= np.isfinite(w_all) & (w_all > 0.0)

    p = profiles[valid]
    w = w_all[valid]
    n = len(p)

    if n == 0 or np.sum(w) <= 0.0:
        mean = np.full(n_ap, np.nan, dtype=np.float64)
        std = np.full(n_ap, np.nan, dtype=np.float64)
        cov = np.full((n_ap, n_ap), np.nan, dtype=np.float64)
        return mean, std, cov, 0

    mean = np.average(p, axis=0, weights=w)

    if not RUN_BOOTSTRAP or n < 2:
        std = np.zeros(n_ap, dtype=np.float64)
        cov = np.zeros((n_ap, n_ap), dtype=np.float64)
        return mean.astype(np.float64), std.astype(np.float64), cov.astype(np.float64), n

    rng = np.random.default_rng(seed)
    boot = np.empty((N_BOOT, n_ap), dtype=np.float64)
    for b in range(N_BOOT):
        draw = rng.integers(0, n, size=n)
        boot[b] = np.average(p[draw], axis=0, weights=w[draw])

    cov = np.cov(boot, rowvar=False)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return mean.astype(np.float64), std.astype(np.float64), cov.astype(np.float64), n

# Backward-compatible wrapper for any future call sites that still expect
# only mean, one-sigma errors, and sample size.
def mean_profile_and_error(profiles, seed=SEED, weights=None):
    mean, std, cov, n = mean_profile_and_covariance(profiles, weights=weights, seed=seed)
    return mean, std, n


# ============================================================
# GEOMETRY AND CAP
# ============================================================

def make_angle_map(ny, nx):
    """Angle from stamp center: 0 is right, 90 is up."""
    cy, cx = ny // 2, nx // 2
    y, x = np.mgrid[:ny, :nx]
    return (np.degrees(np.arctan2(y - cy, x - cx)) % 360.0).astype(np.float64)


def sector_mask(angle_map, center_deg, half_width_deg):
    """Boolean mask for a bidirectional wedge."""
    mask = np.zeros_like(angle_map, dtype=bool)
    for c in [center_deg % 360.0, (center_deg + 180.0) % 360.0]:
        lo = (c - half_width_deg) % 360.0
        hi = (c + half_width_deg) % 360.0
        if lo < hi:
            mask |= (angle_map >= lo) & (angle_map < hi)
        else:
            mask |= (angle_map >= lo) | (angle_map < hi)
    return mask


def compute_cap_values(
    image,
    r_map,
    pixscale,
    cap_radii_arcmin,
    pixel_area,
    sec_mask=None,
):
    """Compensated aperture photometry on one image.

    The returned CAP values are in y arcmin^2. Pixell/reproject handles
    the map projection during thumbnail extraction; this routine then works
    on the local projected cutout with pixel_area = pixscale_arcmin^2.
    """
    cap = np.full(len(cap_radii_arcmin), np.nan, dtype=np.float64)

    for i, r_ap in enumerate(cap_radii_arcmin):
        r_disc_pix = r_ap / pixscale
        r_ring_pix = r_disc_pix * np.sqrt(2.0)

        disc = r_map <= r_disc_pix
        ring = (r_map > r_disc_pix) & (r_map <= r_ring_pix)

        if sec_mask is not None:
            disc = disc & sec_mask
            ring = ring & sec_mask

        n_disc = int(np.sum(np.isfinite(image[disc])))
        n_ring = int(np.sum(np.isfinite(image[ring])))
        if n_disc == 0 or n_ring == 0:
            continue

        disc_sum = float(np.nansum(image[disc]))
        ring_sum = float(np.nansum(image[ring]))
        cap[i] = (disc_sum - ring_sum * n_disc / n_ring) * pixel_area

    return cap


def full_stamp_inside_map(ra_deg, dec_deg, emap, STAMP_SOURCE_RADIUS_ARCMIN):
    """Check that the full thumbnail footprint stays inside the map."""
    dec_rad = np.deg2rad(dec_deg)
    ra_rad = np.deg2rad(ra_deg)
    r_rad = np.full(len(ra_deg), np.deg2rad(STAMP_SOURCE_RADIUS_ARCMIN / 60.0))
    keep = np.ones(len(ra_deg), dtype=bool)

    offsets_dec = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.float64)
    offsets_ra_factor = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.float64)

    for j in range(8):
        test_dec = dec_rad + offsets_dec[j] * r_rad
        test_ra = ra_rad + offsets_ra_factor[j] * r_rad
        keep &= emap.contains(np.vstack([test_dec, test_ra]))

    return keep


# ============================================================
# CATALOG LOADERS
# ============================================================

def load_firefly_full(filepath):
    """Load the full Firefly catalog with no cuts applied."""
    with fits.open(filepath, memmap=True) as hdu:
        data = hdu[1].data
        n_rows = len(data)
        ra = data["PLUG_RA"].astype(np.float64)
        dec = data["PLUG_DEC"].astype(np.float64)
        mstar = data["Chabrier_MILES_stellar_mass"].astype(np.float64)
        z = data["Z"].astype(np.float64)
        age_massW = data["Chabrier_MILES_age_massW"].astype(np.float64)
        age_lightW = data["Chabrier_MILES_age_lightW"].astype(np.float64)
        Z_massW = data["Chabrier_MILES_metallicity_massW"].astype(np.float64)
        Z_lightW = data["Chabrier_MILES_metallicity_lightW"].astype(np.float64)
        EBV = data["Chabrier_MILES_spm_EBV"].astype(np.float64)

    log_mstar = np.log10(mstar)
    fits_idx = np.arange(n_rows, dtype=np.int64)
    print(f"  Firefly total: {n_rows:,}")

    return (
        ra, dec, log_mstar, z, fits_idx,
        age_massW, age_lightW, Z_massW, Z_lightW, EBV,
    )


def load_photo_shapes(
    firefly_ra,
    firefly_dec,
    photo_path,
    match_arcsec=1.0,
    fracdev_thresh=0.5,
    type_galaxy=3,
    safety=1,
):
    """Cross match to SDSS photometric catalog for PA and axis ratio.

    Returns both the selected shape used for oriented stacking and the
    de Vaucouleurs/exponential model quantities needed for diagnostics.
    """
    print(f"  Loading photo catalog: {photo_path}")
    with fits.open(photo_path, memmap=True) as hdu:
        photo = hdu[1].data
        ra_p = photo["RA"].astype(np.float64)
        dec_p = photo["DEC"].astype(np.float64)
        phi_dev = photo["PHI_DEV_DEG"][:, 2].astype(np.float64)
        phi_exp = photo["PHI_EXP_DEG"][:, 2].astype(np.float64)
        ab_dev = photo["AB_DEV"][:, 2].astype(np.float64)
        ab_exp = photo["AB_EXP"][:, 2].astype(np.float64)
        fracdev = photo["FRACDEV"][:, 2].astype(np.float64)
        ptype = photo["TYPE"][:, 2].astype(np.int16)

    tol = np.deg2rad(match_arcsec / 3600.0)
    pos1 = np.deg2rad(np.column_stack([firefly_ra, firefly_dec]))
    pos2 = np.deg2rad(np.column_stack([ra_p, dec_p]))
    tree1 = spatial.KDTree(pos1)
    tree2 = spatial.KDTree(pos2)
    groups = tree1.query_ball_tree(tree2, tol * safety)

    n_ff = len(firefly_ra)
    idx = np.zeros(n_ff, dtype=np.int64)
    good = np.zeros(n_ff, dtype=bool)

    for gi, group in enumerate(groups):
        if len(group) == 0:
            continue
        group = np.array(group)
        dists = utils.angdist(pos1[gi, :, None], pos2[group, :].T)
        best = np.argmin(dists)
        if dists[best] > tol:
            continue
        idx[gi] = group[best]
        good[gi] = True

    print(f"  Photo matched: {good.sum():,} / {n_ff:,}")

    use_dev = fracdev[idx] > fracdev_thresh
    pa = np.where(use_dev, phi_dev[idx], phi_exp[idx])
    ab = np.where(use_dev, ab_dev[idx], ab_exp[idx])

    valid_shape = (
        good
        & (pa > -900)
        & (pa < 360)
        & (ab > 0)
        & (ab <= 1)
        & (ptype[idx] == type_galaxy)
    )
    print(f"  Valid shapes: {valid_shape.sum():,}")

    pa_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    ab_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    pa_dev_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    pa_exp_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    ab_dev_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    ab_exp_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)
    fracdev_out = np.full(len(firefly_ra), np.nan, dtype=np.float64)

    pa_out[valid_shape] = pa[valid_shape]
    ab_out[valid_shape] = ab[valid_shape]
    pa_dev_out[valid_shape] = phi_dev[idx[valid_shape]]
    pa_exp_out[valid_shape] = phi_exp[idx[valid_shape]]
    ab_dev_out[valid_shape] = ab_dev[idx[valid_shape]]
    ab_exp_out[valid_shape] = ab_exp[idx[valid_shape]]
    fracdev_out[valid_shape] = fracdev[idx[valid_shape]]

    return (
        pa_out,
        ab_out,
        valid_shape,
        pa_dev_out,
        pa_exp_out,
        ab_dev_out,
        ab_exp_out,
        fracdev_out,
    )

def crossmatch_to_first(ra, dec, first_path, match_arcsec, safety=1):
    """Return a mask for galaxies with a nearby FIRST galaxy source."""
    print("  Loading FIRST catalog...")
    with fits.open(first_path, memmap=True) as hdu:
        f = hdu[1].data
        ra_first = f["RA"].astype(np.float64)
        dec_first = f["DEC"].astype(np.float64)
        sdss_cls = np.char.strip(f["SDSS_CLASS"].astype(str))

    is_gal = sdss_cls == "g"
    ra_first = ra_first[is_gal]
    dec_first = dec_first[is_gal]

    tol = np.deg2rad(match_arcsec / 3600.0)
    pos1 = np.deg2rad(np.column_stack([ra, dec]))
    pos2 = np.deg2rad(np.column_stack([ra_first, dec_first]))
    tree1 = spatial.KDTree(pos1)
    tree2 = spatial.KDTree(pos2)
    groups = tree1.query_ball_tree(tree2, tol * safety)

    has_radio = np.zeros(len(ra), dtype=bool)
    for gi, group in enumerate(groups):
        if len(group) == 0:
            continue
        group = np.array(group)
        dists = utils.angdist(pos1[gi, :, None], pos2[group, :].T)
        best = np.argmin(dists)
        if dists[best] <= tol:
            has_radio[gi] = True

    print(f"  FIRST cross match ({match_arcsec:.0f} arcsec): {has_radio.sum():,} / {len(ra):,}")
    return has_radio


def _split_tail_pct_label():
    if abs(SPLIT_TAIL_PCT - round(SPLIT_TAIL_PCT)) < 1e-8:
        return f"{int(round(SPLIT_TAIL_PCT))}"
    return f"{SPLIT_TAIL_PCT:.1f}".rstrip("0").rstrip(".")

# ============================================================
# HDF5 STAMP CACHE
# ============================================================

def _extraction_config_dict():
    return {
        "cache_version": "large_source_coordinate_remap_v1",
        "tsz_map_path": TSZ_MAP_PATH,
        "stamp_radius_arcmin": STAMP_RADIUS_ARCMIN,
        "stamp_source_radius_arcmin": float(STAMP_SOURCE_RADIUS_ARCMIN),
        "rotation_method": "large_source_rectbivariatespline_k1",
    }


def _extraction_config_hash():
    cfg = _extraction_config_dict()
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _open_or_create_cache(cache_path, ny, nx, ny_src, nx_src, config_hash):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        h5f = h5py.File(cache_path, "a")
        stored_hash = h5f.attrs.get("config_hash", "")
        if stored_hash != config_hash:
            print("  [cache] Config hash mismatch, rebuilding cache.")
            h5f.close()
            os.remove(cache_path)
            return _open_or_create_cache(cache_path, ny, nx, ny_src, nx_src, config_hash)
        existing_ids = set(h5f["fits_idx"][:].tolist())
        print(f"  [cache] Opened existing cache with {len(existing_ids):,} galaxies")
        return h5f, existing_ids

    print(f"  [cache] Creating new cache: {cache_path}")
    h5f = h5py.File(cache_path, "w")
    h5f.attrs["config_hash"] = config_hash
    h5f.attrs["config_json"] = json.dumps(_extraction_config_dict(), sort_keys=True)
    h5f.attrs["ny"] = ny
    h5f.attrs["nx"] = nx
    h5f.attrs["ny_src"] = ny_src
    h5f.attrs["nx_src"] = nx_src
    h5f.attrs["stamp_radius_arcmin"] = float(STAMP_RADIUS_ARCMIN)
    h5f.attrs["stamp_source_radius_arcmin"] = float(STAMP_SOURCE_RADIUS_ARCMIN)

    scalar_keys = [
        "fits_idx", "ra", "dec", "z", "logm", "pa", "ab",
        "pa_dev", "pa_exp", "ab_dev", "ab_exp", "fracdev",
        "age_massW", "age_lightW", "Z_massW", "Z_lightW", "EBV",
        "has_radio", "extract_attempted", "stamp_valid",
    ]

    for key in scalar_keys:
        dtype = np.float64
        if key == "fits_idx":
            dtype = np.int64
        elif key in ("has_radio", "extract_attempted", "stamp_valid"):
            dtype = bool
        h5f.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)

    h5f.create_dataset(
        "stamps",
        shape=(0, ny_src, nx_src),
        maxshape=(None, ny_src, nx_src),
        dtype=np.float64,
        chunks=(1, ny_src, nx_src),
        compression="gzip",
        compression_opts=4,
    )

    return h5f, set()


def _ensure_cache_column(h5f, key, dtype=np.float64, fill_value=np.nan):
    """Create a resizable 1D cache column if an older cache is missing it."""
    if key in h5f:
        return
    n_rows = h5f["fits_idx"].shape[0]
    ds = h5f.create_dataset(key, shape=(n_rows,), maxshape=(None,), dtype=dtype)
    if n_rows > 0:
        ds[:] = fill_value
    h5f.flush()


def _ensure_photo_model_cache_columns(h5f):
    """Older stamp caches may not have the model-specific photo columns."""
    for key in ["pa_dev", "pa_exp", "ab_dev", "ab_exp", "fracdev"]:
        _ensure_cache_column(h5f, key, dtype=np.float64, fill_value=np.nan)


def _append_to_cache(
    h5f,
    n_new,
    fits_idx,
    ra,
    dec,
    z,
    logm,
    pa,
    ab,
    pa_dev,
    pa_exp,
    ab_dev,
    ab_exp,
    fracdev,
    age_massW,
    age_lightW,
    Z_massW,
    Z_lightW,
    EBV,
    has_radio,
):
    n_existing = h5f["fits_idx"].shape[0]
    n_total = n_existing + n_new

    keys = [
        "fits_idx", "ra", "dec", "z", "logm", "pa", "ab",
        "pa_dev", "pa_exp", "ab_dev", "ab_exp", "fracdev",
        "age_massW", "age_lightW", "Z_massW", "Z_lightW", "EBV",
        "has_radio", "extract_attempted", "stamp_valid",
    ]
    for key in keys:
        h5f[key].resize(n_total, axis=0)
    h5f["stamps"].resize(n_total, axis=0)

    sl = slice(n_existing, n_total)
    h5f["fits_idx"][sl] = fits_idx
    h5f["ra"][sl] = ra
    h5f["dec"][sl] = dec
    h5f["z"][sl] = z
    h5f["logm"][sl] = logm
    h5f["pa"][sl] = pa
    h5f["ab"][sl] = ab
    h5f["pa_dev"][sl] = pa_dev
    h5f["pa_exp"][sl] = pa_exp
    h5f["ab_dev"][sl] = ab_dev
    h5f["ab_exp"][sl] = ab_exp
    h5f["fracdev"][sl] = fracdev
    h5f["age_massW"][sl] = age_massW
    h5f["age_lightW"][sl] = age_lightW
    h5f["Z_massW"][sl] = Z_massW
    h5f["Z_lightW"][sl] = Z_lightW
    h5f["EBV"][sl] = EBV
    h5f["has_radio"][sl] = has_radio
    h5f["extract_attempted"][sl] = False
    h5f["stamp_valid"][sl] = False
    h5f.flush()


def extract_to_cache(comptony, h5f):
    n_total = h5f["fits_idx"].shape[0]
    ny_src = int(h5f.attrs.get("ny_src", h5f.attrs["ny"]))
    nx_src = int(h5f.attrs.get("nx_src", h5f.attrs["nx"]))
    attempted = h5f["extract_attempted"][:]
    n_todo = int((~attempted).sum())

    if n_todo == 0:
        print("  [extract] All galaxies already cached.")
        return

    print(f"  [extract] {n_todo:,} galaxies to extract")
    ra_all = h5f["ra"][:]
    dec_all = h5f["dec"][:]
    r_rad = STAMP_SOURCE_RADIUS_ARCMIN * np.pi / 180.0 / 60.0

    t0 = time.time()
    n_done = 0
    n_success = 0
    n_shape_mismatch = 0

    for i in range(n_total):
        if attempted[i]:
            continue

        coords = np.deg2rad([dec_all[i], ra_all[i]])
        stamp = reproject.thumbnails(comptony, coords=coords, r=r_rad)
        h5f["extract_attempted"][i] = True

        if stamp is None:
            h5f["stamp_valid"][i] = False
            n_done += 1
            continue

        arr = np.array(stamp, dtype=np.float64)
        if arr.shape != (ny_src, nx_src):
            n_shape_mismatch += 1
            old_shape = arr.shape
            arr = arr[:ny_src, :nx_src]
            print(
                f"  [extract] shape mismatch at cache row {i}: "
                f"got {old_shape}, expected {(ny_src, nx_src)}, "
                f"after crop {arr.shape}"
            )

        if arr.shape != (ny_src, nx_src) or np.all(arr == 0):
            h5f["stamp_valid"][i] = False
            n_done += 1
            continue

        h5f["stamps"][i] = arr
        h5f["stamp_valid"][i] = True
        n_done += 1
        n_success += 1

        if n_done % EXTRACTION_BATCH_LOG_EVERY == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-6)
            print(f"    extracted {n_done:,}/{n_todo:,}, success={n_success:,}, rate={rate:.0f}/s")
            h5f.flush()

    h5f.flush()
    dt = time.time() - t0
    print(f"  [extract] Done: {n_success:,} successful / {n_done:,} attempted in {dt / 60:.1f} min")
    print(f"  [extract] Shape mismatches encountered: {n_shape_mismatch:,}")


# ============================================================
# STACKING
# ============================================================

def stack_from_cache(h5f, mask, label="", weights=None):
    """Stack for a selected mask, optionally using per-galaxy weights.

    Cached stamps are larger source thumbnails.  Each selected source
    thumbnail is sampled onto the final output grid, once with angle 0 for
    the unoriented stack and once with -PA for the oriented stack.
    """
    ny = int(h5f.attrs["ny"])
    nx = int(h5f.attrs["nx"])

    stamp_valid = h5f["stamp_valid"][:]
    effective_mask = mask & stamp_valid
    candidate_indices = np.where(mask)[0]
    indices = np.where(effective_mask)[0]
    n_candidates = len(candidate_indices)
    n_sel = len(indices)
    n_rejected_stamp_valid = int(np.sum(mask & ~stamp_valid))

    if weights is None:
        selected_weights = np.ones(n_sel, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != len(mask):
            raise ValueError("weights must have the same length as mask")
        selected_weights = weights[indices].astype(np.float64)
        bad_weight = ~np.isfinite(selected_weights) | (selected_weights <= 0.0)
        if np.any(bad_weight):
            print(f"  [stack accounting: {label}] rejecting {int(np.sum(bad_weight)):,} nonpositive or nonfinite weights")
            keep = ~bad_weight
            indices = indices[keep]
            selected_weights = selected_weights[keep]
            n_sel = len(indices)
            if n_sel == 0:
                return {"n_success": 0, "effective_mask": effective_mask}
        
        effective_mask = np.zeros_like(mask, dtype=bool)
        effective_mask[indices] = True

    sum_w = float(np.sum(selected_weights))

    label_txt = label if label else "unnamed selection"
    print(f"  [stack accounting: {label_txt}] candidates before stamp-valid cut: {n_candidates:,}")
    print(f"  [stack accounting: {label_txt}] rejected by stamp_valid=False: {n_rejected_stamp_valid:,}")
    print(f"  [stack accounting: {label_txt}] used for stacking: {n_sel:,}")
    if weights is not None and n_sel > 0:
        print(
            f"  [stack accounting: {label_txt}] mass weights: "
            f"sum={sum_w:.6e}, min={np.nanmin(selected_weights):.3e}, "
            f"median={np.nanmedian(selected_weights):.3e}, max={np.nanmax(selected_weights):.3e}"
        )

    if n_sel == 0:
        return {"n_success": 0, "effective_mask": effective_mask}

    pixscale = 2.0 * STAMP_RADIUS_ARCMIN / (ny - 1)
    cap_pixel_area = _cap_pixel_area_arcmin2(pixscale)
    print(
        f"  [CAP units: {label_txt}] internal unit=y arcmin^2, "
        f"pixel_area={cap_pixel_area:.6e} arcmin^2"
    )
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[:ny, :nx]
    r_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float64)

    angle_map = make_angle_map(ny, nx)
    major_mask = sector_mask(angle_map, MAJOR_AXIS_ANGLE, WEDGE_HALF_DEG)
    minor_mask = sector_mask(angle_map, MINOR_AXIS_ANGLE, WEDGE_HALF_DEG)

    pa_all = h5f["pa"][:]

    stack_sum_unori = np.zeros((ny, nx), dtype=np.float64)
    stack_sum_ori = np.zeros((ny, nx), dtype=np.float64)

    n_ap = len(CAP_RADII_ARCMIN)
    cap_full_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)
    cap_major_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)
    cap_minor_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)

    t0 = time.time()

    for j, cache_idx in enumerate(indices):
        source_stamp = np.asarray(h5f["stamps"][cache_idx], dtype=np.float64)
        pa_j = float(pa_all[cache_idx])

        # Same coordinate-remapping sampler for both cases.  Angle 0 gives
        # the central final-size thumbnail from the larger source stamp.
        stamp = sample_large_stamp_to_output(
            source_stamp,
            0.0,
            ny,
            nx,
            pixscale,
        )
        stamp_rot = sample_large_stamp_to_output(
            source_stamp,
            -pa_j,
            ny,
            nx,
            pixscale,
        )

        w_j = selected_weights[j]
        stack_sum_unori += w_j * stamp.astype(np.float64)
        stack_sum_ori += w_j * stamp_rot.astype(np.float64)

        cap_full_values[j] = compute_cap_values(
            stamp, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area,
        )
        cap_major_values[j] = compute_cap_values(
            stamp_rot, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area, sec_mask=major_mask,
        )
        cap_minor_values[j] = compute_cap_values(
            stamp_rot, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area, sec_mask=minor_mask,
        )

        if (j + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f"    stacked {j + 1:,}/{n_sel:,}, rate={(j + 1) / max(dt, 1e-6):.0f}/s")

    print(f"  [stack: {label_txt}] {n_sel:,} galaxies used in final stack")

    return {
        "n_success": n_sel,
        "ny": ny,
        "nx": nx,
        "pixscale": pixscale,
        "stack_unori": (stack_sum_unori / sum_w).astype(np.float64),
        "stack_ori": (stack_sum_ori / sum_w).astype(np.float64),
        "weights": selected_weights.astype(np.float64),
        "sum_weights": sum_w,
        "cap_full_values": cap_full_values,
        "cap_major_values": cap_major_values,
        "cap_minor_values": cap_minor_values,
        "effective_mask": effective_mask,
    }


# ============================================================
# SUMMARY PLOT OUTPUTS
# ============================================================

SUMMARY_PDF_NAMES = [
    "summary_oriented_full_stack_2x3.pdf",
    "summary_stellar_age_stack_4x3.pdf",
    "summary_oriented_full_stack_cap_profiles_1x3.pdf",
    "summary_stellar_age_cap_profiles_1x3.pdf",
    "summary_radio_full_stack_1x3.pdf",
    "summary_stellar_age_hist_mass_1x3.pdf",
    "summary_stellar_age_hist_mass_weighted_1x3.pdf",
    "summary_stellar_age_mass_weights_1x3.pdf",
    "summary_stellar_age_hist_ebv_1x3.pdf",
]


def _tex_sci_notation(x, pos=None):
    """Format tick values as explicit LaTeX scientific notation, with no axis offset."""
    if not np.isfinite(x):
        return ""
    if abs(x) < 1e-99:
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0 ** exponent)

    if abs(mantissa - round(mantissa)) < 1e-8:
        mantissa_str = f"{int(round(mantissa))}"
    else:
        mantissa_str = f"{mantissa:.2f}".rstrip("0").rstrip(".")

    if exponent == 0:
        return rf"${mantissa_str}$"
    return rf"${mantissa_str}\times 10^{{{exponent}}}$"


def _apply_scientific_y_ticks(ax, nbins=CAP_Y_TICK_NBINS):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_formatter(FuncFormatter(_tex_sci_notation))
    ax.yaxis.get_offset_text().set_visible(False)

def _shared_cap_ylim(mean_err_pairs, pad_frac=0.08):
    """Return shared y limits including error bars and zero."""
    vals = [np.array([0.0])]

    for mean, err in mean_err_pairs:
        if mean is None or err is None:
            continue

        mean = np.asarray(mean, dtype=np.float64)
        err = np.asarray(err, dtype=np.float64)

        lo = mean - err
        hi = mean + err

        vals.append(lo[np.isfinite(lo)])
        vals.append(hi[np.isfinite(hi)])

    vals = [v for v in vals if len(v) > 0]
    if len(vals) == 0:
        return None

    vals = np.concatenate(vals)
    ylo = float(np.nanmin(vals))
    yhi = float(np.nanmax(vals))

    if not np.isfinite(ylo) or not np.isfinite(yhi):
        return None

    if ylo == yhi:
        pad = max(abs(ylo), 1e-30) * 0.1
    else:
        pad = pad_frac * (yhi - ylo)

    return ylo - pad, yhi + pad


def _format_colorbar(cb, label):
    cb.locator = MaxNLocator(nbins=STACK_COLORBAR_TICK_NBINS)
    cb.formatter = FuncFormatter(_tex_sci_notation)
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.set_label(label, fontsize=STACK_COLORBAR_LABEL_SIZE)
    cb.ax.tick_params(labelsize=STACK_COLORBAR_TICK_SIZE)


def _set_touching_square_grid(fig, axes, left, bottom, top):
    """Position image axes as a touching grid with square panels.

    This prevents imshow panels from leaving horizontal gutters when the
    figure is wider than the square image grid requires.  Returns the
    right edge of the grid in figure coordinates, useful for placing a
    close colorbar.
    """
    axes = np.asarray(axes)
    if axes.ndim != 2:
        raise ValueError("axes must be a 2D array")

    nrows, ncols = axes.shape
    fig_w, fig_h = fig.get_size_inches()
    panel_h = (top - bottom) / nrows
    panel_w = panel_h * fig_h / fig_w

    for r in range(nrows):
        y0 = top - (r + 1) * panel_h
        for c in range(ncols):
            x0 = left + c * panel_w
            axes[r, c].set_position([x0, y0, panel_w, panel_h])

    return left + ncols * panel_w


def _prune_touching_x_ticks(ax, nbins=HIST_X_TICK_NBINS):
    """Avoid overlapping tick labels at zero-spacing panel boundaries."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="both"))


def _prune_touching_xy_ticks(ax, nbins=STACK_XY_TICK_NBINS):
    """Avoid overlapping x and y tick labels on touching image grids."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="both"))


def _mass_bin_label(mass_lo, mass_hi):
    return (
        rf"${mass_lo:.1f} < \log_{{10}}\!\left(\frac{{M_\ast}}{{M_\odot}}\right) "
        rf"\leq {mass_hi:.1f}$"
    )


def _n_stacked_label(n_success):
    return rf"$N_{{\rm stacked}} = {int(n_success):,}$"


def _collect_stack_values_from_results(all_bin_results):
    """Collect every stack image used by the two stack summary figures."""
    chunks = []

    for bin_result in all_bin_results:
        full = bin_result.get("full_stack", {})
        if isinstance(full, dict):
            for key in ["stack_unori", "stack_ori"]:
                stack = full.get(key)
                if stack is not None:
                    vals = stack[np.isfinite(stack)]
                    if len(vals) > 0:
                        chunks.append(vals)

        radio_full = bin_result.get("radio_full_stack", {})
        if isinstance(radio_full, dict):
            for key in ["stack_unori", "stack_ori"]:
                stack = radio_full.get(key)
                if stack is not None:
                    vals = stack[np.isfinite(stack)]
                    if len(vals) > 0:
                        chunks.append(vals)

        for scheme_key in ACTIVE_SPLIT_SCHEMES:
            for split_key in ["lo", "hi"]:
                res = bin_result.get(scheme_key, {}).get(split_key, {})
                stack = res.get("stack_unori") if isinstance(res, dict) else None
                if stack is not None:
                    vals = stack[np.isfinite(stack)]
                    if len(vals) > 0:
                        chunks.append(vals)

    return chunks


def _shared_stack_norm(finite_chunks):
    """Use the full dynamic range of all stacked-map pixels.
    The colorbar is symmetric around zero.
    """
    if len(finite_chunks) == 0:
        return None

    vals = np.concatenate(finite_chunks)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return None

    lo = float(np.nanpercentile(vals, STACK_COLOR_PERCENTILE_LOW))
    hi = float(np.nanpercentile(vals, STACK_COLOR_PERCENTILE_HIGH))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))

    vabs = max(abs(lo), abs(hi), 1e-30)

    print(
        f"  [stack color norm] robust symmetric range: "
        f"p1={lo:.3e}, p99={hi:.3e}, "
        f"vmin={-vabs:.3e}, vmax={vabs:.3e}"
    )

    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)



def _imshow_stack_panel(ax, stack, pixscale, norm, n_success=None):
    if stack is None or pixscale is None or norm is None:
        ax.text(
            0.5,
            0.5,
            r"{\rm no\ data}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
            fontsize=STACK_NO_DATA_SIZE,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    ny, nx = stack.shape
    ext = [
        -0.5 * nx * pixscale,
        0.5 * nx * pixscale,
        -0.5 * ny * pixscale,
        0.5 * ny * pixscale,
    ]
    im = ax.imshow(stack, origin="lower", cmap="RdBu_r", norm=norm, extent=ext)
    ax.set_xlabel(r"$x\ {\rm [arcmin]}$", fontsize=STACK_AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$y\ {\rm [arcmin]}$", fontsize=STACK_AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=STACK_TICK_LABEL_SIZE)

    if n_success is not None:
        ax.text(
            STACK_N_LABEL_X,
            STACK_N_LABEL_Y,
            _n_stacked_label(n_success),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=STACK_N_LABEL_SIZE,
            bbox=dict(facecolor="white", alpha=STACK_N_LABEL_BBOX_ALPHA, edgecolor="none", pad=STACK_N_LABEL_BBOX_PAD),
        )

    return im


def plot_summary_full_mass_stacks(all_bin_results, out_dir, stack_norm):
    """Two rows by three columns: full mass bin unoriented and oriented."""
    if len(all_bin_results) == 0:
        return

    nrows = 2
    ncols = len(MASS_BINS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(STACK_FULL_FIG_WIDTH_PER_COL * ncols, STACK_FULL_FIG_HEIGHT_PER_ROW * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    grid_left = STACK_FULL_GRID_LEFT
    grid_bottom = STACK_FULL_GRID_BOTTOM
    grid_top = STACK_FULL_GRID_TOP
    grid_right = _set_touching_square_grid(fig, axes, grid_left, grid_bottom, grid_top)

    row_defs = [
        ("stack_unori", r"{\rm Unoriented}"),
        ("stack_ori", r"{\rm Oriented}"),
    ]

    last_im = None
    for r, (stack_key, row_label) in enumerate(row_defs):
        for c, bin_result in enumerate(all_bin_results):
            ax = axes[r, c]
            mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
            mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
            res = bin_result.get("full_stack", {})
            stack = res.get(stack_key) if isinstance(res, dict) else None
            pixscale = res.get("pixscale") if isinstance(res, dict) else None
            im = _imshow_stack_panel(ax, stack, pixscale, stack_norm, res.get("n_success", 0))
            if im is not None:
                last_im = im

            _prune_touching_xy_ticks(ax)

            if c > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if r < nrows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if r == 0:
                ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=STACK_PANEL_TITLE_SIZE, pad=STACK_PANEL_TITLE_PAD)

        axes[r, 0].text(
            STACK_FULL_ROW_LABEL_X,
            0.5,
            row_label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=STACK_ROW_LABEL_SIZE,
        )

    if last_im is not None:
        cax = fig.add_axes([grid_right + STACK_COLORBAR_PAD, STACK_FULL_CBAR_BOTTOM, STACK_COLORBAR_WIDTH, STACK_FULL_CBAR_HEIGHT])
        cb = fig.colorbar(last_im, cax=cax)
        _format_colorbar(cb, r"$y\ {\rm [dimensionless]}$")

    fig.suptitle(r"{\rm Oriented\ Full\ Stack}", fontsize=STACK_SUPTITLE_SIZE, y=STACK_FULL_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_oriented_full_stack_2x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary full stacks] {path}")


def plot_summary_age_split_stacks(all_bin_results, out_dir, stack_norm):
    """Four rows by three columns: age populations by mass bins."""
    if len(all_bin_results) == 0:
        return

    tail_pct = _split_tail_pct_label()

    row_defs = [
        ("mass_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("mass_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("light_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ light\ weighted\ age}}"),
        ("light_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ light\ weighted\ age}}"),
    ]

    nrows = len(row_defs)
    ncols = len(MASS_BINS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(STACK_AGE_FIG_WIDTH_PER_COL * ncols, STACK_AGE_FIG_HEIGHT_PER_ROW * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    grid_left = STACK_AGE_GRID_LEFT
    grid_bottom = STACK_AGE_GRID_BOTTOM
    grid_top = STACK_AGE_GRID_TOP
    grid_right = _set_touching_square_grid(fig, axes, grid_left, grid_bottom, grid_top)

    last_im = None
    for r, (scheme_key, split_key, row_label) in enumerate(row_defs):
        for c, bin_result in enumerate(all_bin_results):
            ax = axes[r, c]
            mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
            mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
            res = bin_result.get(scheme_key, {}).get(split_key, {})
            stack = res.get("stack_unori") if isinstance(res, dict) else None
            pixscale = res.get("pixscale") if isinstance(res, dict) else None
            im = _imshow_stack_panel(ax, stack, pixscale, stack_norm, res.get("n_success", 0))
            if im is not None:
                last_im = im

            _prune_touching_xy_ticks(ax)

            if c > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if r < nrows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if r == 0:
                ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=STACK_PANEL_TITLE_SIZE, pad=STACK_PANEL_TITLE_PAD)

        axes[r, 0].text(
            STACK_AGE_ROW_LABEL_X,
            0.5,
            row_label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=STACK_ROW_LABEL_SIZE,
        )

    if last_im is not None:
        cax = fig.add_axes([grid_right + STACK_COLORBAR_PAD, STACK_AGE_CBAR_BOTTOM, STACK_COLORBAR_WIDTH, STACK_AGE_CBAR_HEIGHT])
        cb = fig.colorbar(last_im, cax=cax)
        _format_colorbar(cb, r"$y\ {\rm [dimensionless]}$")

    fig.suptitle(r"{\rm Stellar\ Age\ Stack}", fontsize=STACK_SUPTITLE_SIZE, y=STACK_AGE_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_stellar_age_stack_4x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary age stacks] {path}")


def plot_summary_sector_cap_profiles(all_bin_results, out_dir):
    """One row by three columns: major versus minor sector CAP in each mass bin."""
    if len(all_bin_results) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(CAP_FIG_WIDTH_PER_COL * len(MASS_BINS), CAP_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=CAP_LEFT, right=CAP_RIGHT, bottom=CAP_BOTTOM, top=CAP_TOP, wspace=CAP_WSPACE)

    ylim_pairs = []

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        res = bin_result.get("full_stack", {})

        maj_m = _cap_to_plot_units(res.get("cap_major_mean"))
        maj_s = _cap_to_plot_units(res.get("cap_major_std"))
        min_m = _cap_to_plot_units(res.get("cap_minor_mean"))
        min_s = _cap_to_plot_units(res.get("cap_minor_std"))

        if maj_m is None or maj_s is None or min_m is None or min_s is None:
            ax.text(
                0.5,
                0.5,
                r"{\rm no\ sector\ CAP\ data}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
        else:
            ylim_pairs.extend([
                (maj_m, maj_s),
                (min_m, min_s),
            ])

            ax.errorbar(
                CAP_RADII_ARCMIN - 0.04,
                maj_m,
                yerr=maj_s,
                fmt="P",
                linestyle="none",
                capsize=CAP_ERROR_CAPSIZE,
                lw=CAP_ERROR_LW,
                ms=CAP_ERROR_MARKER_SIZE,
                label=r"{\rm Major\ sector}",
            )
            ax.errorbar(
                CAP_RADII_ARCMIN + 0.04,
                min_m,
                yerr=min_s,
                fmt="X",
                linestyle="none",
                capsize=CAP_ERROR_CAPSIZE,
                lw=CAP_ERROR_LW,
                ms=CAP_ERROR_MARKER_SIZE,
                label=r"{\rm Minor\ sector}",
            )

        ax.axhline(0, color=CAP_ZERO_LINE_COLOR, lw=CAP_ZERO_LINE_WIDTH, ls=CAP_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=CAP_PANEL_TITLE_SIZE, pad=CAP_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=CAP_TICK_LABEL_SIZE)
        ax.set_xlabel(r"$\theta_d\ {\rm [arcmin]}$", fontsize=CAP_AXIS_LABEL_SIZE_SECTOR)
        if c == 0:
            ax.set_ylabel(_cap_plot_axis_label(), fontsize=CAP_AXIS_LABEL_SIZE_SECTOR)
        else:
            ax.tick_params(labelleft=False)
        ax.legend(loc=CAP_LEGEND_LOC_SECTOR, fontsize=CAP_LEGEND_SIZE_SECTOR)

    shared_ylim = _shared_cap_ylim(ylim_pairs)
    if shared_ylim is not None:
        for ax in axes:
            ax.set_ylim(shared_ylim)
            _apply_scientific_y_ticks(ax)
    else:
        for ax in axes:
            _apply_scientific_y_ticks(ax)

    fig.suptitle(r"{\rm Oriented\ Full\ Stack\ CAP\ profiles}", fontsize=CAP_SUPTITLE_SIZE, y=CAP_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_oriented_full_stack_cap_profiles_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary sector CAP] {path}")


def plot_summary_age_split_cap_profiles(all_bin_results, out_dir):
    """One row by three columns: four age split CAP profiles in each mass bin."""
    if len(all_bin_results) == 0:
        return

    tail_pct = _split_tail_pct_label()

    curve_defs = [
        ("mass_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ mass\ weighted\ age}}", -0.09, "o"),
        ("mass_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ mass\ weighted\ age}}", -0.03, "s"),
        ("light_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ light\ weighted\ age}}", 0.03, "^"),
        ("light_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ light\ weighted\ age}}", 0.09, "D"),
    ]

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(CAP_FIG_WIDTH_PER_COL * len(MASS_BINS), CAP_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=CAP_LEFT, right=CAP_RIGHT, bottom=CAP_BOTTOM, top=CAP_TOP, wspace=CAP_WSPACE)

    ylim_pairs = []

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])

        for scheme_key, split_key, label, jitter, marker in curve_defs:
            res = bin_result.get(scheme_key, {}).get(split_key, {})
            cap_m = _cap_to_plot_units(res.get("cap_mean")) if isinstance(res, dict) else None
            cap_s = _cap_to_plot_units(res.get("cap_std")) if isinstance(res, dict) else None
            if cap_m is None or cap_s is None:
                continue

            ylim_pairs.append((cap_m, cap_s))

            ax.errorbar(
                CAP_RADII_ARCMIN + jitter,
                cap_m,
                yerr=cap_s,
                fmt=marker,
                linestyle="none",
                capsize=CAP_ERROR_CAPSIZE,
                lw=CAP_ERROR_LW,
                ms=CAP_ERROR_MARKER_SIZE,
                label=label,
            )

        ax.axhline(0, color=CAP_ZERO_LINE_COLOR, lw=CAP_ZERO_LINE_WIDTH, ls=CAP_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=CAP_PANEL_TITLE_SIZE, pad=CAP_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=CAP_TICK_LABEL_SIZE)
        ax.set_xlabel(r"$\theta_d\ {\rm [arcmin]}$", fontsize=CAP_AXIS_LABEL_SIZE_AGE)
        if c == 0:
            ax.set_ylabel(_cap_plot_axis_label(), fontsize=CAP_AXIS_LABEL_SIZE_AGE)
        else:
            ax.tick_params(labelleft=False)
        ax.legend(loc=CAP_LEGEND_LOC_AGE, fontsize=CAP_LEGEND_SIZE_AGE)

    shared_ylim = _shared_cap_ylim(ylim_pairs)
    if shared_ylim is not None:
        for ax in axes:
            ax.set_ylim(shared_ylim)
            _apply_scientific_y_ticks(ax)
    else:
        for ax in axes:
            _apply_scientific_y_ticks(ax)

    fig.suptitle(r"{\rm Stellar\ Age\ CAP\ profiles}", fontsize=CAP_SUPTITLE_SIZE, y=CAP_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_stellar_age_cap_profiles_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary age CAP] {path}")



def plot_summary_radio_full_stacks(all_bin_results, out_dir, stack_norm):
    """One row by three columns: radio-only full stacks by mass bin."""
    if len(all_bin_results) == 0:
        return

    nrows = 1
    ncols = len(MASS_BINS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(STACK_RADIO_FIG_WIDTH_PER_COL * ncols, STACK_RADIO_FIG_HEIGHT),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    grid_left = STACK_RADIO_GRID_LEFT
    grid_bottom = STACK_RADIO_GRID_BOTTOM
    grid_top = STACK_RADIO_GRID_TOP
    grid_right = _set_touching_square_grid(fig, axes, grid_left, grid_bottom, grid_top)
    axes = axes.ravel()

    last_im = None
    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        res = bin_result.get("radio_full_stack", {})
        stack = res.get(RADIO_STACK_KEY) if isinstance(res, dict) else None
        pixscale = res.get("pixscale") if isinstance(res, dict) else None
        im = _imshow_stack_panel(ax, stack, pixscale, stack_norm, res.get("n_success", 0))
        if im is not None:
            last_im = im

        _prune_touching_xy_ticks(ax)

        if c > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=STACK_PANEL_TITLE_SIZE, pad=STACK_PANEL_TITLE_PAD)

    if last_im is not None:
        cax = fig.add_axes([grid_right + STACK_COLORBAR_PAD, STACK_RADIO_CBAR_BOTTOM, STACK_COLORBAR_WIDTH, STACK_RADIO_CBAR_HEIGHT])
        cb = fig.colorbar(last_im, cax=cax)
        _format_colorbar(cb, r"$y\ {\rm [dimensionless]}$")

    fig.suptitle(r"{\rm Radio\ Full\ Stack}", fontsize=STACK_SUPTITLE_SIZE, y=STACK_RADIO_SUPTITLE_Y)

    path = os.path.join(out_dir, "summary_radio_full_stack_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary radio full stacks] {path}")


def save_cap_covariances(all_bin_results, out_dir):
    """Save CAP means, one-sigma errors, and full covariance matrices."""
    payload = {}

    for i_bin, bin_result in enumerate(all_bin_results):
        mass_lo = bin_result.get("mass_lo", MASS_BINS[i_bin][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[i_bin][1])
        tag = f"logm_{mass_lo:.1f}_{mass_hi:.1f}".replace(".", "p")

        full = bin_result.get("full_stack", {})
        if isinstance(full, dict):
            for name in ["cap", "cap_major", "cap_minor"]:
                mean_key = f"{name}_mean" if name != "cap" else "cap_mean"
                std_key = f"{name}_std" if name != "cap" else "cap_std"
                cov_key = f"{name}_cov" if name != "cap" else "cap_cov"
                if cov_key in full:
                    payload[f"{tag}_full_{name}_mean"] = full[mean_key]
                    payload[f"{tag}_full_{name}_std"] = full[std_key]
                    payload[f"{tag}_full_{name}_cov"] = full[cov_key]

        for scheme_key in ACTIVE_SPLIT_SCHEMES:
            for split_key in ["lo", "hi"]:
                res = bin_result.get(scheme_key, {}).get(split_key, {})
                if isinstance(res, dict) and "cap_cov" in res:
                    prefix = f"{tag}_{scheme_key}_{split_key}"
                    payload[f"{prefix}_mean"] = res["cap_mean"]
                    payload[f"{prefix}_std"] = res["cap_std"]
                    payload[f"{prefix}_cov"] = res["cap_cov"]

    payload["cap_radii_arcmin"] = CAP_RADII_ARCMIN
    payload["cap_internal_unit"] = np.array("y arcmin^2")
    payload["cap_plot_unit"] = np.array("yarcmin2")
    payload["cap_plot_scale_from_internal"] = np.array(1.0)
    path = os.path.join(out_dir, "cap_profile_covariances.npz")
    np.savez(path, **payload)
    print(f"  [CAP covariance] {path}")



def _cap_sig(value, error):
    value = float(value)
    error = float(error)
    if not np.isfinite(value) or not np.isfinite(error) or error <= 0.0:
        return np.nan
    return value / error


def _cap_diff_sig(value_a, error_a, value_b, error_b):
    value_a = float(value_a)
    error_a = float(error_a)
    value_b = float(value_b)
    error_b = float(error_b)
    diff = value_a - value_b
    diff_error = np.sqrt(error_a ** 2 + error_b ** 2)
    if not np.isfinite(diff) or not np.isfinite(diff_error) or diff_error <= 0.0:
        return diff, diff_error, np.nan
    return diff, diff_error, diff / diff_error


def _cap_fmt_value(x):
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6e}"


def _cap_fmt_sigma(x):
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _print_cap_table(title, columns, rows):
    print("\n" + title)
    print("\t".join(columns))
    if len(rows) == 0:
        print("no data")
        return
    for row in rows:
        print("\t".join(row))


def _cap_total_sigma(sigmas):
    sigmas = np.asarray(sigmas, dtype=np.float64)
    good = np.isfinite(sigmas)
    if not np.any(good):
        return np.nan
    return float(np.sqrt(np.sum(sigmas[good] ** 2)))


def _cap_sig_array(values, errors):
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    out[good] = values[good] / errors[good]
    return out


def _cap_diff_sig_array(values_a, errors_a, values_b, errors_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    errors_a = np.asarray(errors_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    diff = values_a - values_b
    diff_error = np.sqrt(errors_a ** 2 + errors_b ** 2)
    out = np.full(diff.shape, np.nan, dtype=np.float64)
    good = np.isfinite(diff) & np.isfinite(diff_error) & (diff_error > 0.0)
    out[good] = diff[good] / diff_error[good]
    return out


def print_cap_significance_tables(all_bin_results):
    sector_columns = [
        "mass_bin",
        "theta_arcmin",
        "major_sigma0",
        "minor_sigma0",
        "difference_sigma",
    ]
    sector_rows = []
    total_zero_rows = []
    total_difference_rows = []

    for i_bin, bin_result in enumerate(all_bin_results):
        mass_lo = bin_result.get("mass_lo", MASS_BINS[i_bin][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[i_bin][1])
        mass_label = f"({mass_lo:.1f}, {mass_hi:.1f}]"
        full = bin_result.get("full_stack", {})
        if not isinstance(full, dict):
            continue

        maj_m = _cap_to_plot_units(full.get("cap_major_mean"))
        maj_s = _cap_to_plot_units(full.get("cap_major_std"))
        min_m = _cap_to_plot_units(full.get("cap_minor_mean"))
        min_s = _cap_to_plot_units(full.get("cap_minor_std"))
        if maj_m is None or maj_s is None or min_m is None or min_s is None:
            continue

        maj_sig = _cap_sig_array(maj_m, maj_s)
        min_sig = _cap_sig_array(min_m, min_s)
        diff_sig = _cap_diff_sig_array(maj_m, maj_s, min_m, min_s)

        for theta, a_sig, b_sig, d_sig in zip(CAP_RADII_ARCMIN, maj_sig, min_sig, diff_sig):
            sector_rows.append([
                mass_label,
                f"{float(theta):.2f}",
                _cap_fmt_sigma(a_sig),
                _cap_fmt_sigma(b_sig),
                _cap_fmt_sigma(abs(d_sig)),
            ])

        total_zero_rows.append([
            mass_label,
            "major",
            _cap_fmt_sigma(_cap_total_sigma(maj_sig)),
        ])
        total_zero_rows.append([
            mass_label,
            "minor",
            _cap_fmt_sigma(_cap_total_sigma(min_sig)),
        ])
        total_difference_rows.append([
            mass_label,
            "major_vs_minor",
            _cap_fmt_sigma(_cap_total_sigma(diff_sig)),
        ])

    _print_cap_table("CAP significance table: major versus minor sector", sector_columns, sector_rows)

    age_columns = [
        "mass_bin",
        "theta_arcmin",
        "low_sigma0",
        "high_sigma0",
        "difference_sigma",
    ]

    for scheme_key, title in [
        ("mass_age", "CAP significance table: lowest versus highest mass weighted stellar age"),
        ("light_age", "CAP significance table: lowest versus highest light weighted stellar age"),
    ]:
        age_rows = []
        for i_bin, bin_result in enumerate(all_bin_results):
            mass_lo = bin_result.get("mass_lo", MASS_BINS[i_bin][0])
            mass_hi = bin_result.get("mass_hi", MASS_BINS[i_bin][1])
            mass_label = f"({mass_lo:.1f}, {mass_hi:.1f}]"
            low = bin_result.get(scheme_key, {}).get("lo", {})
            high = bin_result.get(scheme_key, {}).get("hi", {})
            if not isinstance(low, dict) or not isinstance(high, dict):
                continue

            low_m = _cap_to_plot_units(low.get("cap_mean"))
            low_s = _cap_to_plot_units(low.get("cap_std"))
            high_m = _cap_to_plot_units(high.get("cap_mean"))
            high_s = _cap_to_plot_units(high.get("cap_std"))
            if low_m is None or low_s is None or high_m is None or high_s is None:
                continue

            low_sig = _cap_sig_array(low_m, low_s)
            high_sig = _cap_sig_array(high_m, high_s)
            diff_sig = _cap_diff_sig_array(low_m, low_s, high_m, high_s)

            for theta, a_sig, b_sig, d_sig in zip(CAP_RADII_ARCMIN, low_sig, high_sig, diff_sig):
                age_rows.append([
                    mass_label,
                    f"{float(theta):.2f}",
                    _cap_fmt_sigma(a_sig),
                    _cap_fmt_sigma(b_sig),
                    _cap_fmt_sigma(abs(d_sig)),
                ])

            if scheme_key == "mass_age":
                low_label = "lowest_mass_weighted_age"
                high_label = "highest_mass_weighted_age"
                diff_label = "lowest_vs_highest_mass_weighted_age"
            else:
                low_label = "lowest_light_weighted_age"
                high_label = "highest_light_weighted_age"
                diff_label = "lowest_vs_highest_light_weighted_age"

            total_zero_rows.append([
                mass_label,
                low_label,
                _cap_fmt_sigma(_cap_total_sigma(low_sig)),
            ])
            total_zero_rows.append([
                mass_label,
                high_label,
                _cap_fmt_sigma(_cap_total_sigma(high_sig)),
            ])
            total_difference_rows.append([
                mass_label,
                diff_label,
                _cap_fmt_sigma(_cap_total_sigma(diff_sig)),
            ])

        _print_cap_table(title, age_columns, age_rows)

    _print_cap_table(
        "CAP total significance from zero",
        ["mass_bin", "profile", "total_sigma0"],
        total_zero_rows,
    )
    _print_cap_table(
        "CAP total significance of differences",
        ["mass_bin", "comparison", "total_difference_sigma"],
        total_difference_rows,
    )

# ============================================================
# MAIN PIPELINE
# ============================================================

def build_selection_and_cache():
    """Load catalogs, make the current selection, and fill the stamp cache."""
    if RADIO_ONLY and EXCLUDE_RADIO:
        raise ValueError("RADIO_ONLY and EXCLUDE_RADIO cannot both be True.")
    if FIRST_PATH is None and (RADIO_ONLY or EXCLUDE_RADIO):
        raise ValueError("FIRST_PATH is None but a radio selection is enabled.")

    print("\nLoading tSZ map...")
    comptony = enmap.read_map(TSZ_MAP_PATH)

    print("\nLoading Firefly catalog...")
    (
        ra_ff,
        dec_ff,
        logm_ff,
        z_ff,
        fits_idx_ff,
        age_massW_ff,
        age_lightW_ff,
        Z_massW_ff,
        Z_lightW_ff,
        EBV_ff,
    ) = load_firefly_full(FIREFLY_PATH)

    n_firefly = len(ra_ff)
    if n_firefly == 0:
        raise RuntimeError("Firefly catalog is empty.")

    finite_data_mask = (
        np.isfinite(ra_ff)
        & np.isfinite(dec_ff)
        & np.isfinite(logm_ff)
        & np.isfinite(z_ff)
        & np.isfinite(age_massW_ff)
        & np.isfinite(age_lightW_ff)
        & np.isfinite(Z_massW_ff)
        & np.isfinite(Z_lightW_ff)
        & np.isfinite(EBV_ff)
    )
    mass_range_mask = (
        np.isfinite(logm_ff)
        & (logm_ff > CACHE_LOG_MASS_MIN)
        & (logm_ff <= CACHE_LOG_MASS_MAX)
    )
    redshift_mask = np.ones(n_firefly, dtype=bool)
    if USE_REDSHIFT_CUT:
        redshift_mask = np.isfinite(z_ff) & (z_ff >= Z_MIN) & (z_ff <= Z_MAX)

    ebv_mask = np.isfinite(EBV_ff)
    if USE_EBV_CUT:
        ebv_mask &= (EBV_ff >= 0.0) & (EBV_ff <= EBV_MAX)

    broad_mask = finite_data_mask & mass_range_mask & redshift_mask & ebv_mask

    print("\nCatalog cut flow:")
    all_rows_mask = np.ones(n_firefly, dtype=bool)
    print_mass_bin_accounting("raw Firefly rows", all_rows_mask, logm_ff)

    running_mask = all_rows_mask & finite_data_mask
    print_mass_bin_accounting("after finite RA/Dec/logM/z/age/metallicity/E(B-V) data", running_mask, logm_ff, all_rows_mask)

    prev_mask = running_mask
    running_mask = running_mask & mass_range_mask
    print_mass_bin_accounting(
        f"after mass range ({CACHE_LOG_MASS_MIN:.1f}, {CACHE_LOG_MASS_MAX:.1f}]",
        running_mask,
        logm_ff,
        prev_mask,
    )

    if USE_REDSHIFT_CUT:
        prev_mask = running_mask
        running_mask = running_mask & redshift_mask
        print_mass_bin_accounting(
            f"after redshift cut [{Z_MIN:.2f}, {Z_MAX:.2f}]",
            running_mask,
            logm_ff,
            prev_mask,
        )

    if USE_EBV_CUT:
        prev_mask = running_mask
        running_mask = running_mask & ebv_mask
        print_mass_bin_accounting(
            f"after E(B-V) cut [0.000, {EBV_MAX:.3f}]",
            running_mask,
            logm_ff,
            prev_mask,
        )

    print_mass_bin_accounting("broad sample", broad_mask, logm_ff)

    pa_ff = np.zeros(n_firefly, dtype=np.float64)
    ab_ff = np.ones(n_firefly, dtype=np.float64)
    pa_dev_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    pa_exp_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    ab_dev_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    ab_exp_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    fracdev_ff = np.full(n_firefly, np.nan, dtype=np.float64)

    if USE_PHOTO_SHAPE:
        if PHOTO_PATH is None or not os.path.exists(PHOTO_PATH):
            raise FileNotFoundError(f"USE_PHOTO_SHAPE=True but PHOTO_PATH was not found: {PHOTO_PATH}")

        valid_shape_ff = np.zeros(n_firefly, dtype=bool)
        photo_candidates = broad_mask.copy()
        (
            pa_tmp,
            ab_tmp,
            valid_tmp,
            pa_dev_tmp,
            pa_exp_tmp,
            ab_dev_tmp,
            ab_exp_tmp,
            fracdev_tmp,
        ) = load_photo_shapes(
            ra_ff[photo_candidates],
            dec_ff[photo_candidates],
            PHOTO_PATH,
            match_arcsec=PHOTO_MATCH_ARCSEC,
            fracdev_thresh=PHOTO_FRACDEV_THRESH,
            type_galaxy=PHOTO_TYPE_GALAXY,
            safety=SAFETY,
        )
        pa_ff[photo_candidates] = pa_tmp
        ab_ff[photo_candidates] = ab_tmp
        pa_dev_ff[photo_candidates] = pa_dev_tmp
        pa_exp_ff[photo_candidates] = pa_exp_tmp
        ab_dev_ff[photo_candidates] = ab_dev_tmp
        ab_exp_ff[photo_candidates] = ab_exp_tmp
        fracdev_ff[photo_candidates] = fracdev_tmp
        valid_shape_ff[photo_candidates] = valid_tmp

        finite_shape_mask = (
            np.isfinite(pa_ff)
            & np.isfinite(ab_ff)
            & np.isfinite(pa_dev_ff)
            & np.isfinite(pa_exp_ff)
            & np.isfinite(ab_dev_ff)
            & np.isfinite(ab_exp_ff)
            & np.isfinite(fracdev_ff)
        )

        shape_mask = valid_shape_ff & finite_shape_mask & (ab_ff < BA_MAX)

        print(f"  valid photo shape and 0 < selected b/a < {BA_MAX:.3f}: {(shape_mask & broad_mask).sum():,}")
    else:
        shape_mask = np.ones(n_firefly, dtype=bool)

    shape_sel = broad_mask & shape_mask
    print("\nPhoto-shape accounting:")
    print_mass_bin_accounting("before photo shape selection", broad_mask, logm_ff)
    print_mass_bin_accounting(f"after photo shape and 0 < selected b/a < {BA_MAX:.3f}", shape_sel, logm_ff, broad_mask)

    print("\nChecking map coverage...")
    inside_map = np.zeros(n_firefly, dtype=bool)
    map_candidates = broad_mask & shape_mask
    inside_map[map_candidates] = full_stamp_inside_map(
        ra_ff[map_candidates],
        dec_ff[map_candidates],
        comptony,
        STAMP_SOURCE_RADIUS_ARCMIN,
    )
    print(f"  inside map: {(inside_map & map_candidates).sum():,} / {map_candidates.sum():,}")

    map_sel = broad_mask & shape_mask & inside_map
    print("\nMap-footprint accounting:")
    print_mass_bin_accounting("before full-stamp map-footprint cut", shape_sel, logm_ff)
    print_mass_bin_accounting("after full-stamp map-footprint cut", map_sel, logm_ff, shape_sel)

    has_radio_ff = np.zeros(n_firefly, dtype=bool)
    if FIRST_PATH is not None:
        finite_radio_match_mask = np.isfinite(ra_ff) & np.isfinite(dec_ff)
        radio_candidates = broad_mask & shape_mask & inside_map & finite_radio_match_mask
        has_radio_ff[radio_candidates] = crossmatch_to_first(
            ra_ff[radio_candidates],
            dec_ff[radio_candidates],
            FIRST_PATH,
            FIRST_MATCH_ARCSEC,
            safety=SAFETY,
        )

        print("\nFIRST radio accounting:")
        print_mass_bin_accounting("radio-candidate parent sample", radio_candidates, logm_ff)
        print_mass_bin_accounting(
            f"FIRST-matched galaxies within {FIRST_MATCH_ARCSEC:.1f} arcsec",
            radio_candidates & has_radio_ff,
            logm_ff,
        )
        print_mass_bin_accounting(
            "non-radio galaxies after FIRST exclusion",
            radio_candidates & ~has_radio_ff,
            logm_ff,
            radio_candidates,
        )

    if RADIO_ONLY:
        radio_mask = has_radio_ff
    elif EXCLUDE_RADIO:
        radio_mask = ~has_radio_ff
    else:
        radio_mask = np.ones(n_firefly, dtype=bool)

    cache_sel = broad_mask & shape_mask & inside_map
    base_sel = cache_sel & radio_mask
    radio_sel = cache_sel & has_radio_ff

    print("\nFinal samples:")
    print(f"  main selected galaxies: {base_sel.sum():,} / {n_firefly:,}")
    print(f"  radio galaxies: {radio_sel.sum():,} / {n_firefly:,}")
    print(f"  cached galaxies: {cache_sel.sum():,} / {n_firefly:,}")

    print("\nFinal sample accounting by mass bin:")
    print_mass_bin_accounting("cache selection before radio split", cache_sel, logm_ff)
    print_mass_bin_accounting("main non-radio selection", base_sel, logm_ff, cache_sel)
    print_mass_bin_accounting("radio-only selection", radio_sel, logm_ff)
    if base_sel.sum() == 0:
        raise RuntimeError("No galaxies survive the main full selection.")
    if cache_sel.sum() == 0:
        raise RuntimeError("No galaxies survive the cache selection.")

    r_rad_test = STAMP_RADIUS_ARCMIN * np.pi / 180.0 / 60.0
    r_rad_source_test = STAMP_SOURCE_RADIUS_ARCMIN * np.pi / 180.0 / 60.0
    test = None
    test_source = None
    for j in np.where(cache_sel)[0][:50]:
        coords = np.deg2rad([dec_ff[j], ra_ff[j]])
        test = reproject.thumbnails(comptony, coords=coords, r=r_rad_test)
        test_source = reproject.thumbnails(comptony, coords=coords, r=r_rad_source_test)
        if test is not None and test_source is not None:
            break
    if test is None or test_source is None:
        raise RuntimeError("Could not extract both final and source test thumbnails.")

    ny, nx = np.array(test, dtype=np.float64).shape
    ny_src, nx_src = np.array(test_source, dtype=np.float64).shape
    test_pixscale_arcmin = 2.0 * STAMP_RADIUS_ARCMIN / (ny - 1)
    test_source_pixscale_arcmin = 2.0 * STAMP_SOURCE_RADIUS_ARCMIN / (ny_src - 1)
    test_pixel_area_arcmin2 = test_pixscale_arcmin ** 2
    print(f"  Final stamp dimensions: {ny} x {nx}")
    print(f"  Source stamp dimensions: {ny_src} x {nx_src}")
    print(f"  Final stamp radius: {STAMP_RADIUS_ARCMIN:.6f} arcmin")
    print(f"  Source stamp radius: {STAMP_SOURCE_RADIUS_ARCMIN:.6f} arcmin")
    print(f"  Final thumbnail pixel scale: {test_pixscale_arcmin:.6f} arcmin per pixel")
    print(f"  Source thumbnail pixel scale: {test_source_pixscale_arcmin:.6f} arcmin per pixel")
    print(f"  CAP pixel area used: {test_pixel_area_arcmin2:.6e} arcmin^2")

    print("\nBuilding or opening stamp cache...")
    config_hash = _extraction_config_hash()
    h5f, existing_ids = _open_or_create_cache(CACHE_FILE, ny, nx, ny_src, nx_src, config_hash)
    _ensure_photo_model_cache_columns(h5f)
    _write_cap_area_metadata(h5f, test_pixscale_arcmin)

    cache_indices = np.where(cache_sel)[0]
    cache_fits_idx = fits_idx_ff[cache_indices]
    new_mask = np.array([fid not in existing_ids for fid in cache_fits_idx])
    n_new = int(new_mask.sum())

    if n_new > 0:
        print(f"  [cache] Adding {n_new:,} new galaxies")
        new_local = cache_indices[new_mask]
        print("THIS IS ME TESTING CACHE Before they are stored")
        print("fits_idx before:", fits_idx_ff[new_local][:10])
        print("ra before:", ra_ff[new_local][:10])


        _append_to_cache(
            h5f,
            n_new,
            fits_idx=fits_idx_ff[new_local],
            ra=ra_ff[new_local],
            dec=dec_ff[new_local],
            z=z_ff[new_local],
            logm=logm_ff[new_local],
            pa=pa_ff[new_local],
            ab=ab_ff[new_local],
            pa_dev=pa_dev_ff[new_local],
            pa_exp=pa_exp_ff[new_local],
            ab_dev=ab_dev_ff[new_local],
            ab_exp=ab_exp_ff[new_local],
            fracdev=fracdev_ff[new_local],
            age_massW=age_massW_ff[new_local],
            age_lightW=age_lightW_ff[new_local],
            Z_massW=Z_massW_ff[new_local],
            Z_lightW=Z_lightW_ff[new_local],
            EBV=EBV_ff[new_local],
            has_radio=has_radio_ff[new_local],
        )
        print("THIS IS ME TESTING CACHE AFTER they are stored")
        print("fits_idx cache:", h5f["fits_idx"][-n_new:][:10])
        print("ra cache:", h5f["ra"][-n_new:][:10])
    else:
        print(f"  [cache] All {len(cache_fits_idx):,} cache galaxies are already in cache")

    base_indices = np.where(base_sel)[0]
    base_fits_idx = fits_idx_ff[base_indices]
    radio_indices = np.where(radio_sel)[0]
    radio_fits_idx = fits_idx_ff[radio_indices]

    if USE_PHOTO_SHAPE:
        cache_fits_idx = h5f["fits_idx"][:]
        ff_fid_to_idx = {int(fid): i for i, fid in enumerate(fits_idx_ff)}
        n_updated = 0
        for ci, fid in enumerate(cache_fits_idx):
            li = ff_fid_to_idx.get(int(fid))
            if li is None:
                continue
            h5f["pa"][ci] = pa_ff[li]
            h5f["ab"][ci] = ab_ff[li]
            h5f["pa_dev"][ci] = pa_dev_ff[li]
            h5f["pa_exp"][ci] = pa_exp_ff[li]
            h5f["ab_dev"][ci] = ab_dev_ff[li]
            h5f["ab_exp"][ci] = ab_exp_ff[li]
            h5f["fracdev"][ci] = fracdev_ff[li]
            n_updated += 1
        h5f.flush()
        print(f"  [cache] Updated PA and b/a for {n_updated:,} galaxies")

    extract_to_cache(comptony, h5f)

    print("\nCached stamp accounting after extraction:")
    cache_fits_idx_now = h5f["fits_idx"][:]
    cache_logm_now = h5f["logm"][:]
    cache_stamp_valid_now = h5f["stamp_valid"][:]
    cache_base_rows = np.isin(cache_fits_idx_now, base_fits_idx)
    cache_radio_rows = np.isin(cache_fits_idx_now, radio_fits_idx)

    print_mass_bin_accounting("all rows currently in stamp cache", np.ones(len(cache_fits_idx_now), dtype=bool), cache_logm_now)
    print_mass_bin_accounting("main-selection cache rows before stamp-valid cut", cache_base_rows, cache_logm_now)
    print_mass_bin_accounting("main-selection cache rows after stamp-valid cut", cache_base_rows & cache_stamp_valid_now, cache_logm_now, cache_base_rows)
    print_mass_bin_accounting("radio-selection cache rows before stamp-valid cut", cache_radio_rows, cache_logm_now)
    print_mass_bin_accounting("radio-selection cache rows after stamp-valid cut", cache_radio_rows & cache_stamp_valid_now, cache_logm_now, cache_radio_rows)

    return h5f, base_fits_idx, radio_fits_idx


def add_full_stack_results(h5f, bin_result, mass_mask):
    """Compute full mass bin stack and CAP quantities."""
    mass_lo = bin_result.get("mass_lo", np.nan)
    mass_hi = bin_result.get("mass_hi", np.nan)
    result = stack_from_cache(
        h5f,
        mass_mask,
        label=f"full non-radio stack logM ({mass_lo:.1f}, {mass_hi:.1f}]",
    )

    if result["n_success"] == 0:
        bin_result["full_stack"] = {"n_success": 0}
        return

    cap_m, cap_s, cap_cov, n_cap = mean_profile_and_covariance(
        result["cap_full_values"], seed=SEED
    )
    cap_maj_m, cap_maj_s, cap_maj_cov, _ = mean_profile_and_covariance(
        result["cap_major_values"], seed=SEED
    )
    cap_min_m, cap_min_s, cap_min_cov, _ = mean_profile_and_covariance(
        result["cap_minor_values"], seed=SEED
    )

    print(f"    full CAP profiles from {n_cap:,} galaxies")

    bin_result["full_stack"] = {
        "n_success": result["n_success"],
        "stack_unori": result["stack_unori"],
        "stack_ori": result["stack_ori"],
        "pixscale": result["pixscale"],
        "cap_mean": cap_m,
        "cap_std": cap_s,
        "cap_cov": cap_cov,
        "cap_major_mean": cap_maj_m,
        "cap_major_std": cap_maj_s,
        "cap_major_cov": cap_maj_cov,
        "cap_minor_mean": cap_min_m,
        "cap_minor_std": cap_min_s,
        "cap_minor_cov": cap_min_cov,
        "effective_mask": result["effective_mask"],
    }


def add_radio_stack_results(h5f, bin_result, mass_mask):
    """Compute radio-only full mass bin stack."""
    mass_lo = bin_result.get("mass_lo", np.nan)
    mass_hi = bin_result.get("mass_hi", np.nan)
    result = stack_from_cache(
        h5f,
        mass_mask,
        label=f"radio-only stack logM ({mass_lo:.1f}, {mass_hi:.1f}]",
    )

    if result["n_success"] == 0:
        bin_result["radio_full_stack"] = {"n_success": 0}
        return

    bin_result["radio_full_stack"] = {
        "n_success": result["n_success"],
        "stack_unori": result["stack_unori"],
        "stack_ori": result["stack_ori"],
        "pixscale": result["pixscale"],
    }


def mass_uniform_weights(logm_all, split_mask, mass_lo, mass_hi, n_bins=SPLIT_MASS_WEIGHT_N_BINS):
    """Return per-row weights that flatten one split sample in stellar mass.

    The weights are defined on the objects selected by split_mask.  For the
    tSZ stacks, split_mask should already include stamp_valid, otherwise the
    later stamp-valid cut can spoil the intended mass flattening.

    Within the parent mass bin, the mass interval is divided into n_bins
    equal-width bins.  A galaxy in nonempty bin b receives weight proportional
    to

        target_count / N_b = (N_selected / n_bins) / N_b.

    Empty bins cannot be fixed, because there are no galaxies to reweight.
    The final weights are renormalized so the mean weight of selected
    galaxies is 1, which keeps weight values easy to read.  This
    renormalization does not change any weighted mean stack.
    """
    logm_all = np.asarray(logm_all, dtype=np.float64)
    split_mask = np.asarray(split_mask, dtype=bool)
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    weights = np.zeros(len(logm_all), dtype=np.float64)
    edges = np.linspace(mass_lo, mass_hi, n_bins + 1)

    selected = (
        split_mask
        & np.isfinite(logm_all)
        & (logm_all > mass_lo)
        & (logm_all <= mass_hi)
    )
    n_selected = int(np.sum(selected))
    if n_selected == 0:
        return weights, edges, np.zeros(n_bins, dtype=int)

    counts, _ = np.histogram(logm_all[selected], bins=edges)
    target = float(n_selected) / float(n_bins)

    bin_index = np.searchsorted(edges, logm_all[selected], side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    selected_idx = np.where(selected)[0]
    for b in range(n_bins):
        in_bin = bin_index == b
        if counts[b] <= 0:
            continue
        weights[selected_idx[in_bin]] = target / float(counts[b])

    if SPLIT_MASS_WEIGHT_CLIP is not None:
        positive = weights[selected]
        positive = positive[positive > 0.0]
        if len(positive) > 0:
            med = float(np.median(positive))
            weights[selected] = np.clip(weights[selected], 0.0, SPLIT_MASS_WEIGHT_CLIP * med)

    # Normalize to mean weight 1 over the selected split population.
    sum_w = float(np.sum(weights[selected]))
    if sum_w > 0.0:
        weights[selected] *= n_selected / sum_w

    return weights, edges, counts

def add_age_split_results(h5f, bin_result, mass_mask, cache_fields):
    """Compute young and old age split stacks for one mass bin."""
    logm_all = h5f["logm"][:]
    stamp_valid = h5f["stamp_valid"][:]

    for scheme_key in ACTIVE_SPLIT_SCHEMES:
        values = cache_fields[_scheme_field(scheme_key)].copy()
        values[~mass_mask] = np.nan

        is_lo, is_hi, p_lo_val, p_hi_val = apply_percentile_split(
            values,
            remove_middle_pct=SPLIT_REMOVE_MIDDLE_PCT,
        )
        is_lo &= mass_mask
        is_hi &= mass_mask

        print(
            f"    {_scheme_label(scheme_key)}: "
            f"{_lo_label(scheme_key)} N={is_lo.sum():,}, "
            f"{_hi_label(scheme_key)} N={is_hi.sum():,}, "
            f"cut values {p_lo_val:.3e}, {p_hi_val:.3e}"
        )

        bin_result[scheme_key] = {}
        split_defs = [
            ("lo", is_lo, _lo_label(scheme_key), SEED),
            ("hi", is_hi, _hi_label(scheme_key), SEED),
        ]

        for split_key, split_mask, label, seed in split_defs:
            if int(split_mask.sum()) == 0:
                bin_result[scheme_key][split_key] = {"n_success": 0}
                continue

            mass_lo = bin_result.get("mass_lo", np.nan)
            mass_hi = bin_result.get("mass_hi", np.nan)
            split_weights = None
            weight_edges = None
            weight_counts = None
            if USE_SPLIT_MASS_WEIGHTS:
                # Weight the same population that can actually enter the stack.
                # If weights are computed before the stamp-valid cut, the final
                # stacked sample can stop being flat in stellar mass.
                split_weight_mask = split_mask & stamp_valid
                split_weights, weight_edges, weight_counts = mass_uniform_weights(
                    logm_all,
                    split_weight_mask,
                    mass_lo,
                    mass_hi,
                    n_bins=SPLIT_MASS_WEIGHT_N_BINS,
                )
                w_sel = split_weights[split_weight_mask]
                w_sel = w_sel[np.isfinite(w_sel) & (w_sel > 0.0)]
                n_weight_bins = int(len(weight_counts)) if weight_counts is not None else int(SPLIT_MASS_WEIGHT_N_BINS)
                target_fraction = 1.0 / float(n_weight_bins)
                if len(w_sel) > 0:
                    print(
                        f"      mass weighting {label}: "
                        f"target fraction per bin={target_fraction:.3f}, "
                        f"min/median/max weight={np.min(w_sel):.3e}/"
                        f"{np.median(w_sel):.3e}/{np.max(w_sel):.3e}"
                    )

            print(f"      stacking {label}: {split_mask.sum():,} candidates")
            result = stack_from_cache(
                h5f,
                split_mask,
                label=f"{label} logM ({mass_lo:.1f}, {mass_hi:.1f}]",
                weights=split_weights,
            )
            if result["n_success"] == 0:
                bin_result[scheme_key][split_key] = {"n_success": 0}
                continue

            cap_m, cap_s, cap_cov, n_cap = mean_profile_and_covariance(
                result["cap_full_values"], weights=result.get("weights"), seed=seed
            )
            bin_result[scheme_key][split_key] = {
                "n_success": result["n_success"],
                "stack_unori": result["stack_unori"],
                "stack_ori": result["stack_ori"],
                "pixscale": result["pixscale"],
                "cap_mean": cap_m,
                "cap_std": cap_s,
                "cap_cov": cap_cov,
                "n_cap": n_cap,
                "effective_mask": result["effective_mask"],
                "mass_weights": result.get("weights"),
                "mass_weight_edges": weight_edges,
                "mass_weight_counts": weight_counts,
                "sum_weights": result.get("sum_weights"),
            }


def _hist_bin_edges(var_name, mass_lo=None, mass_hi=None):
    if var_name == "logm":
        return np.linspace(mass_lo, mass_hi, MASS_WEIGHT_N_BINS + 1)
    if var_name == "z":
        return np.linspace(Z_MIN, Z_MAX, HIST_N_BINS + 1)
    if var_name == "EBV":
        return np.linspace(0.0, EBV_MAX, HIST_N_BINS + 1)
    if var_name == "age_log":
        return np.linspace(HIST_AGE_LOG_MIN, HIST_AGE_LOG_MAX, HIST_N_BINS + 1)
    raise ValueError(f"Unknown histogram variable: {var_name}")

def _hist_xlabel(var_name):
    if var_name == "logm":
        return r"$\log_{10}\!\left(M_\ast/M_\odot\right)$"
    if var_name == "z":
        return r"$\mathrm{Redshift}\ (z)$"
    if var_name == "EBV":
        return r"$E(B-V)$"
    if var_name == "age_log":
        return r"$\log_{10}\!\left(t_{\rm age}/{\rm yr}\right)$"
    raise ValueError(f"Unknown histogram variable: {var_name}")

def _hist_title(var_name):
    if var_name == "logm":
        return r"{\rm Stellar\ Mass\ Distribution}"
    if var_name == "z":
        return r"{\rm Redshift\ Distribution}"
    if var_name == "EBV":
        return r"{\rm Dust\ Reddening\ Distribution}"
    if var_name == "age_log":
        return r"{\rm Stellar\ Age\ Distribution}"
    raise ValueError(f"Unknown histogram variable: {var_name}")

def _latex_sci_value(value, precision=HIST_MEDIAN_SCI_PRECISION):
    """Return a compact LaTeX scientific notation string without dollar signs."""
    value = float(value)
    if not np.isfinite(value):
        return r"\mathrm{nan}"
    if value == 0.0:
        return r"0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    mantissa_str = f"{mantissa:.{precision}f}"

    if exponent == 0:
        return mantissa_str
    return rf"{mantissa_str}\times 10^{{{exponent}}}"

def _hist_legend_label_with_median(label, var_name, median_value, median_raw=None):
    """Append median value to the stellar-age split histogram legend label."""
    if not HIST_SHOW_MEDIAN_IN_LEGEND:
        return label

    if var_name == "age_log" and HIST_AGE_LOG_LEGEND_MEDIAN_IN_YEARS and median_raw is not None:
        display_value = median_raw
        unit = r"\ {\rm yr}"
    else:
        display_value = median_value
        unit = ""

    if not np.isfinite(display_value):
        return label

    median_str = _latex_sci_value(display_value)
    return rf"${label},\ {HIST_MEDIAN_TEXT}={median_str}{unit}$"

def plot_summary_age_split_histograms(all_bin_results, h5f, out_dir, var_name):
    """
    Plot 1x3 histogram summary for one variable:
    logm, z, EBV, or age_log.

    Each panel is one stellar mass bin.
    Each panel overlays the four age-selected subsamples.
    The legend gives each subsample median in scientific notation.
    The histogram height is relative frequency per bin:
        N_bin / N_total_subsample.
    """
    if len(all_bin_results) == 0:
        return

    tail_pct = _split_tail_pct_label()

    curve_defs = [
        ("mass_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("mass_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("light_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ light\ weighted\ age}}"),
        ("light_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ light\ weighted\ age}}"),
    ]
    
    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    # Pull cached catalog arrays once.
    logm_all = h5f["logm"][:]
    z_all = h5f["z"][:]
    ebv_all = h5f["EBV"][:]
    age_massW_all = h5f["age_massW"][:]
    age_lightW_all = h5f["age_lightW"][:]

    var_map = {
        "logm": logm_all,
        "z": z_all,
        "EBV": ebv_all,
    }

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])

        bins = _hist_bin_edges(var_name, mass_lo=mass_lo, mass_hi=mass_hi)

        for scheme_key, split_key, label in curve_defs:
            res = bin_result.get(scheme_key, {}).get(split_key, {})
            if not isinstance(res, dict):
                continue

            mask = res.get("effective_mask")
            if mask is None:
                continue

            median_raw = None
            if var_name == "age_log":
                if scheme_key == "mass_age":
                    x_raw = age_massW_all[mask]
                elif scheme_key == "light_age":
                    x_raw = age_lightW_all[mask]
                else:
                    raise ValueError(f"Unknown age split scheme: {scheme_key}")

                x_raw = x_raw[np.isfinite(x_raw) & (x_raw > 0.0)]
                if len(x_raw) == 0:
                    continue

                median_raw = float(np.nanmedian(x_raw))
                x = np.log10(x_raw)

            else:
                values_all = var_map[var_name]
                x = values_all[mask]
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue

            median_value = float(np.nanmedian(x))
            label_with_median = _hist_legend_label_with_median(
                label,
                var_name,
                median_value,
                median_raw=median_raw,
            )

            weights = np.ones_like(x, dtype=np.float64) / len(x)

            ax.hist(
                x,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=HIST_LINEWIDTH,
                label=label_with_median,
            )

        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(_hist_xlabel(var_name), fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Fraction\ per\ bin}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)

        ax.set_ylim(*HIST_YLIMS[var_name])
        _prune_touching_x_ticks(ax)
        ax.legend(loc=HIST_LEGEND_LOC, fontsize=HIST_LEGEND_SIZE)

    fig.suptitle(_hist_title(var_name), fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)

    filename_map = {
        "logm": "summary_stellar_age_hist_mass_1x3.pdf",
        "z": "summary_stellar_age_hist_redshift_1x3.pdf",
        "EBV": "summary_stellar_age_hist_ebv_1x3.pdf",
        "age_log": "summary_stellar_age_hist_age_log_1x3.pdf",
    }

    path = os.path.join(out_dir, filename_map[var_name])
    _savefig(path)
    plt.close(fig)
    print(f"  [summary histogram: {var_name}] {path}")


def plot_summary_age_split_weighted_mass_histograms(all_bin_results, h5f, out_dir):
    """Plot mass histograms after applying split-stack mass weights."""
    if len(all_bin_results) == 0:
        return

    tail_pct = _split_tail_pct_label()
    curve_defs = [
        ("mass_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("mass_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("light_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ light\ weighted\ age}}"),
        ("light_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ light\ weighted\ age}}"),
    ]

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    logm_all = h5f["logm"][:]

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        bins = _hist_bin_edges("logm", mass_lo=mass_lo, mass_hi=mass_hi)

        for scheme_key, split_key, label in curve_defs:
            res = bin_result.get(scheme_key, {}).get(split_key, {})
            if not isinstance(res, dict):
                continue
            mask = res.get("effective_mask")
            w = res.get("mass_weights")
            if mask is None or w is None:
                continue

            x = logm_all[mask]
            w = np.asarray(w, dtype=np.float64)
            good = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
            x = x[good]
            w = w[good]
            if len(x) == 0 or np.sum(w) <= 0.0:
                continue

            ax.hist(
                x,
                bins=bins,
                weights=w / np.sum(w),
                histtype="step",
                linewidth=HIST_LINEWIDTH,
                label=rf"${label}$",
            )

        target_fraction = 1.0 / float(len(bins) - 1)
        ax.axhline(target_fraction, color=HIST_ZERO_LINE_COLOR, lw=HIST_ZERO_LINE_WIDTH, ls=HIST_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(_hist_xlabel("logm"), fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Weighted\ fraction\ per\ bin}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(*HIST_YLIMS["logm"])
        _prune_touching_x_ticks(ax)
        ax.legend(loc=HIST_LEGEND_LOC, fontsize=HIST_LEGEND_SIZE)

    fig.suptitle(r"{\rm Stellar\ Mass\ Distribution\ After\ Mass\ Weighting}", fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_stellar_age_hist_mass_weighted_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary weighted mass histogram] {path}")


def plot_summary_age_split_mass_weight_values(all_bin_results, h5f, out_dir):
    """Plot the mean galaxy weight assigned in each mass sub-bin."""
    if len(all_bin_results) == 0:
        return

    tail_pct = _split_tail_pct_label()
    curve_defs = [
        ("mass_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("mass_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ mass\ weighted\ age}}"),
        ("light_age", "lo", rf"{{\rm Lowest\ {tail_pct}\%\ light\ weighted\ age}}"),
        ("light_age", "hi", rf"{{\rm Highest\ {tail_pct}\%\ light\ weighted\ age}}"),
    ]

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    logm_all = h5f["logm"][:]
    all_weight_values = []

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        bins = _hist_bin_edges("logm", mass_lo=mass_lo, mass_hi=mass_hi)
        centers = 0.5 * (bins[:-1] + bins[1:])

        for scheme_key, split_key, label in curve_defs:
            res = bin_result.get(scheme_key, {}).get(split_key, {})
            if not isinstance(res, dict):
                continue
            mask = res.get("effective_mask")
            w = res.get("mass_weights")
            if mask is None or w is None:
                continue
            x = logm_all[mask]
            w = np.asarray(w, dtype=np.float64)
            good = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
            x = x[good]
            w = w[good]
            if len(x) == 0:
                continue

            bin_idx = np.searchsorted(bins, x, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            mean_w = np.full(len(bins) - 1, np.nan, dtype=np.float64)
            for b in range(len(mean_w)):
                in_b = bin_idx == b
                if np.any(in_b):
                    mean_w[b] = np.mean(w[in_b])
            all_weight_values.append(mean_w[np.isfinite(mean_w)])
            ax.step(centers, mean_w, where="mid", linewidth=HIST_LINEWIDTH, label=rf"${label}$")

        ax.axhline(1.0, color=HIST_ZERO_LINE_COLOR, lw=HIST_ZERO_LINE_WIDTH, ls=HIST_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(_hist_xlabel("logm"), fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Mean\ mass\ weight}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)
        _prune_touching_x_ticks(ax)
        ax.legend(loc=HIST_LEGEND_LOC, fontsize=HIST_LEGEND_SIZE)

    finite_weight_values = [v for v in all_weight_values if len(v) > 0]
    if len(finite_weight_values) > 0:
        vals = np.concatenate(finite_weight_values)
        ymax = float(np.nanpercentile(vals, 99)) * 1.15
        ymax = max(ymax, 1.2)
        for ax in axes:
            ax.set_ylim(0.0, ymax)

    fig.suptitle(r"{\rm Applied\ Split\ Stack\ Mass\ Weights}", fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)
    path = os.path.join(out_dir, "summary_stellar_age_mass_weights_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary mass weights] {path}")


def _fraction_step_hist(ax, x, bins, label=None, linewidth=HIST_LINEWIDTH):
    """Plot a relative-frequency step histogram."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return False

    weights = np.ones_like(x, dtype=np.float64) / len(x)
    ax.hist(
        x,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=linewidth,
        label=label,
    )
    return True


def _folded_pa_difference_deg(pa1_deg, pa2_deg):
    """Axial PA difference in degrees, folded into [0, 90]."""
    return np.abs(((pa1_deg - pa2_deg + 90.0) % 180.0) - 90.0)


def plot_summary_oriented_selected_ba_histograms(all_bin_results, h5f, out_dir):
    """Plot selected oriented-sample b/a distributions by mass bin."""
    if len(all_bin_results) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    ab_all = h5f["ab"][:]
    bins = np.linspace(BA_HIST_MIN, BA_HIST_MAX, HIST_N_BINS + 1)

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        res = bin_result.get("full_stack", {})
        mask = res.get("effective_mask") if isinstance(res, dict) else None

        if mask is not None:
            x = ab_all[mask]
            x = x[np.isfinite(x) & (x > 0.0) & (x < BA_MAX)]
            _fraction_step_hist(ax, x, bins)

        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(r"$b/a$", fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Fraction\ per\ bin}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlim(BA_HIST_MIN, BA_HIST_MAX)
        ax.set_ylim(*HIST_YLIMS["ba_selected"])
        _prune_touching_x_ticks(ax)

    fig.suptitle(r"{\rm Nonradio\ Oriented\ Sample\ Axis\ Ratio}", fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)

    path = os.path.join(out_dir, "summary_oriented_hist_ba_selected_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary histogram: selected b/a] {path}")


def plot_summary_photo_model_delta_ba_histograms(all_bin_results, h5f, out_dir):
    """Plot de Vaucouleurs minus exponential b/a by mass bin."""
    if len(all_bin_results) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    ab_dev_all = h5f["ab_dev"][:]
    ab_exp_all = h5f["ab_exp"][:]
    bins = np.linspace(DELTA_BA_HIST_MIN, DELTA_BA_HIST_MAX, HIST_N_BINS + 1)

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        res = bin_result.get("full_stack", {})
        mask = res.get("effective_mask") if isinstance(res, dict) else None

        if mask is not None:
            ab_dev = ab_dev_all[mask]
            ab_exp = ab_exp_all[mask]
            good = (
                np.isfinite(ab_dev)
                & np.isfinite(ab_exp)
                & (ab_dev > 0.0)
                & (ab_dev <= 1.0)
                & (ab_exp > 0.0)
                & (ab_exp <= 1.0)
            )
            x = ab_dev[good] - ab_exp[good]
            _fraction_step_hist(ax, x, bins)

        ax.axvline(0.0, color=HIST_ZERO_LINE_COLOR, lw=HIST_ZERO_LINE_WIDTH, ls=HIST_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(r"$(b/a)_{\rm deV}-(b/a)_{\rm exp}$", fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Fraction\ per\ bin}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlim(DELTA_BA_HIST_MIN, DELTA_BA_HIST_MAX)
        ax.set_ylim(*HIST_YLIMS["delta_ba"])
        _prune_touching_x_ticks(ax)

    fig.suptitle(r"{\rm Nonradio\ deVaucouleurs\ minus\ Exponential\ Axis\ Ratio}", fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)

    path = os.path.join(out_dir, "summary_photo_model_hist_delta_ba_dev_minus_exp_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary histogram: delta b/a] {path}")


def plot_summary_photo_model_folded_pa_difference_histograms(all_bin_results, h5f, out_dir):
    """Plot folded de Vaucouleurs versus exponential PA differences by mass bin."""
    if len(all_bin_results) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(MASS_BINS),
        figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)

    pa_dev_all = h5f["pa_dev"][:]
    pa_exp_all = h5f["pa_exp"][:]
    bins = np.linspace(PA_DIFF_HIST_MIN, PA_DIFF_HIST_MAX, HIST_N_BINS + 1)

    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get("mass_lo", MASS_BINS[c][0])
        mass_hi = bin_result.get("mass_hi", MASS_BINS[c][1])
        res = bin_result.get("full_stack", {})
        mask = res.get("effective_mask") if isinstance(res, dict) else None

        if mask is not None:
            pa_dev = pa_dev_all[mask]
            pa_exp = pa_exp_all[mask]
            good = (
                np.isfinite(pa_dev)
                & np.isfinite(pa_exp)
                & (pa_dev > -900.0)
                & (pa_exp > -900.0)
            )
            x = _folded_pa_difference_deg(pa_dev[good], pa_exp[good])
            _fraction_step_hist(ax, x, bins)

        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(r"$|\Delta{\rm PA}|_{\rm folded}\ [{\rm deg}]$", fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Fraction\ per\ bin}$", fontsize=HIST_AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlim(PA_DIFF_HIST_MIN, PA_DIFF_HIST_MAX)
        ax.set_ylim(*HIST_YLIMS["pa_folded_diff"])
        _prune_touching_x_ticks(ax)

    fig.suptitle(r"{\rm Nonradio\ Folded\ deVaucouleurs\ versus\ Exponential\ PA\ Difference}", fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)

    path = os.path.join(out_dir, "summary_photo_model_hist_folded_pa_difference_1x3.pdf")
    _savefig(path)
    plt.close(fig)
    print(f"  [summary histogram: folded PA difference] {path}")

def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    lo_pct = (100.0 - SPLIT_REMOVE_MIDDLE_PCT) / 2.0
    print("=" * 70)
    print("ORIENTED TSZ STACKING, SUMMARY FIGURES AND DIAGNOSTIC HISTOGRAMS")
    print("  Mass bin convention: mass_lo < log10(M*/Msun) <= mass_hi")
    print(f"  Mass bins: {MASS_BINS}")
    print(f"  Split tails: bottom/top {lo_pct:.0f} percent")
    print(f"  CAP radii: {CAP_RADII_ARCMIN}")
    print("  CAP unit convention: y arcmin^2")
    print("  Projection handling: pixell/reproject extracts local cutouts from the CAR map")
    print(f"  Bootstrap errors: {'ON' if RUN_BOOTSTRAP else 'OFF'}, N_BOOT={N_BOOT:,}")
    print(f"  Split-stack mass weights: {'ON' if USE_SPLIT_MASS_WEIGHTS else 'OFF'}, bins={MASS_WEIGHT_N_BINS}")
    print("=" * 70)

    h5f, base_fits_idx, radio_fits_idx = build_selection_and_cache()

    cache_stamp_valid = h5f["stamp_valid"][:]
    cache_logm = h5f["logm"][:]
    cache_fits_idx = h5f["fits_idx"][:]
    cache_in_current_selection = np.isin(cache_fits_idx, base_fits_idx)
    cache_in_radio_selection = np.isin(cache_fits_idx, radio_fits_idx)

    cache_fields = {
        "age_massW": h5f["age_massW"][:],
        "age_lightW": h5f["age_lightW"][:],
        "Z_massW": h5f["Z_massW"][:],
        "Z_lightW": h5f["Z_lightW"][:],
        "EBV": h5f["EBV"][:],
    }

    print("\nCache summary:")
    print(f"  total cache rows: {len(cache_fits_idx):,}")
    print(f"  valid stamps: {cache_stamp_valid.sum():,}")
    print(f"  current selection rows: {cache_in_current_selection.sum():,}")
    print(f"  radio selection rows: {cache_in_radio_selection.sum():,}")

    print("\nCache selection accounting by mass bin:")
    print_mass_bin_accounting("all cache rows", np.ones(len(cache_fits_idx), dtype=bool), cache_logm)
    print_mass_bin_accounting("all valid-stamp cache rows", cache_stamp_valid, cache_logm)
    print_mass_bin_accounting("main selection before stamp-valid cut", cache_in_current_selection, cache_logm)
    print_mass_bin_accounting("main selection after stamp-valid cut", cache_in_current_selection & cache_stamp_valid, cache_logm, cache_in_current_selection)
    print_mass_bin_accounting("radio selection before stamp-valid cut", cache_in_radio_selection, cache_logm)
    print_mass_bin_accounting("radio selection after stamp-valid cut", cache_in_radio_selection & cache_stamp_valid, cache_logm, cache_in_radio_selection)

    all_bin_results = []

    for mass_lo, mass_hi in MASS_BINS:
        print("\n" + "=" * 70)
        print(f"MASS BIN: log stellar mass ({mass_lo}, {mass_hi}]")
        print("=" * 70)

        in_mass_bin = _mass_bin_mask(cache_logm, mass_lo, mass_hi)
        
        main_bin_before_stamp = cache_in_current_selection & in_mass_bin
        main_bin_after_stamp = main_bin_before_stamp & cache_stamp_valid
        radio_bin_before_stamp = cache_in_radio_selection & in_mass_bin
        radio_bin_after_stamp = radio_bin_before_stamp & cache_stamp_valid

        mass_mask = main_bin_after_stamp
        radio_mass_mask = radio_bin_after_stamp

        print("  Main sample accounting inside this mass bin:")
        print(f"    before stamp-valid cut: {int(np.sum(main_bin_before_stamp)):,}")
        print(f"    rejected by stamp_valid=False: {int(np.sum(main_bin_before_stamp & ~cache_stamp_valid)):,}")
        print(f"    after stamp-valid cut: {int(np.sum(main_bin_after_stamp)):,}")

        print("  Radio sample accounting inside this mass bin:")
        print(f"    before stamp-valid cut: {int(np.sum(radio_bin_before_stamp)):,}")
        print(f"    rejected by stamp_valid=False: {int(np.sum(radio_bin_before_stamp & ~cache_stamp_valid)):,}")
        print(f"    after stamp-valid cut: {int(np.sum(radio_bin_after_stamp)):,}")
        bin_result = {"mass_lo": mass_lo, "mass_hi": mass_hi}

        if int(mass_mask.sum()) == 0:
            bin_result["full_stack"] = {"n_success": 0}
            for scheme_key in ACTIVE_SPLIT_SCHEMES:
                bin_result[scheme_key] = {}
        else:
            print("  Full mass bin stack")
            add_full_stack_results(h5f, bin_result, mass_mask)

            print("  Age split stacks")
            add_age_split_results(h5f, bin_result, mass_mask, cache_fields)

        if int(radio_mass_mask.sum()) == 0:
            bin_result["radio_full_stack"] = {"n_success": 0}
        else:
            print("  Radio full stack")
            add_radio_stack_results(h5f, bin_result, radio_mass_mask)

        all_bin_results.append(bin_result)

    print("\nWriting summary PDFs...")
    stack_norm = _shared_stack_norm(_collect_stack_values_from_results(all_bin_results))
    plot_summary_full_mass_stacks(all_bin_results, SUMMARY_DIR, stack_norm)
    plot_summary_age_split_stacks(all_bin_results, SUMMARY_DIR, stack_norm)
    plot_summary_sector_cap_profiles(all_bin_results, SUMMARY_DIR)
    plot_summary_age_split_cap_profiles(all_bin_results, SUMMARY_DIR)
    plot_summary_radio_full_stacks(all_bin_results, SUMMARY_DIR, stack_norm)
    plot_summary_age_split_histograms(all_bin_results, h5f, SUMMARY_DIR, "logm")
    plot_summary_age_split_weighted_mass_histograms(all_bin_results, h5f, SUMMARY_DIR)
    plot_summary_age_split_mass_weight_values(all_bin_results, h5f, SUMMARY_DIR)
    plot_summary_age_split_histograms(all_bin_results, h5f, SUMMARY_DIR, "EBV")
    save_cap_covariances(all_bin_results, SUMMARY_DIR)
    print_cap_significance_tables(all_bin_results)

    print("\nDone. The only PDFs written by this script are:")
    for pdf_name in SUMMARY_PDF_NAMES:
        print(f"  {pdf_name}")


if __name__ == "__main__":
    main()
