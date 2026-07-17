#!/usr/bin/env python3

"""Oriented tSZ stacking pipeline.

CAP fitting follows the emcee line-fitting tutorial
(https://emcee.readthedocs.io/en/stable/tutorials/line/):
maximum-likelihood quadratic fit, walkers initialized near the ML solution,
posterior percentiles and correlations from the samples, corner plots,
posterior curve draws, and printed/exported coefficients.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from astropy.io import fits
from pixell import enmap, reproject, utils
from scipy.interpolate import RectBivariateSpline
from scipy import spatial
from scipy import stats as scipy_stats
import h5py

try:
    import emcee
    import corner
except ImportError:
    emcee = None
    corner = None


os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "text.latex.preamble": "\n".join([
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{textcomp}",
        r"\usepackage{amsmath}",
    ]),
})


# ============================================================
# CONFIGURATION
# ============================================================


FIREFLY_PATH = '/Users/jerrywang/Documents/Battaglia_research/Project1/Catalogs/sdss_firefly-26.fits'

PHOTO_PATH = '/Users/jerrywang/Documents/Battaglia_research/Project1/Catalogs/photoPosPlate-dr17.fits'

TSZ_MAP_PATH = '/Users/jerrywang/Documents/Battaglia_research/Project1/Catalogs/act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.2_24.0.fits'

FIRST_PATH = '/Users/jerrywang/Documents/Battaglia_research/Project1/Catalogs/first_14dec17.fits'

FIREFLY_CLASS_FIELD = 'CLASS'

FIREFLY_GALAXY_CLASS = 'GALAXY'

PHOTO_MATCH_ARCSEC = 1.0

PHOTO_FRACDEV_THRESH = 0.5

PHOTO_TYPE_GALAXY = 3

BA_MAX = 0.75

WEDGE_HALF_DEG = 15.0

MAJOR_AXIS_ANGLE = 90.0

MINOR_AXIS_ANGLE = 0.0

CAP_N_AP = 9

CAP_AP_MIN_ARCMIN = 2.0

CAP_AP_MAX_ARCMIN = 6.0

CAP_RADII_ARCMIN = np.linspace(CAP_AP_MIN_ARCMIN, CAP_AP_MAX_ARCMIN, CAP_N_AP)

MASS_BINS = [(11.0, 11.4), (11.4, 11.7), (11.7, 12.0)]

CACHE_LOG_MASS_MIN = min((lo for lo, hi in MASS_BINS))

CACHE_LOG_MASS_MAX = max((hi for lo, hi in MASS_BINS))

USE_REDSHIFT_CUT = True

Z_MIN = 0.2

Z_MAX = 0.6

RADIO_ONLY = False

EXCLUDE_RADIO = False

FIRST_MATCH_ARCSEC = 60.0

SAFETY = 1

STAMP_RADIUS_ARCMIN = 15.0

STAMP_SOURCE_RADIUS_ARCMIN = 21.25

CACHE_DIR = './stamp_cache'

EXTRACTION_BATCH_LOG_EVERY = 2000

RUN_BOOTSTRAP = True

N_BOOT = 10000

SEED = 42

CAP_SHAPE_FIT_MODEL = 'linear'

_CAP_SHAPE_FIT_DEGREES = {'linear': 1, 'quadratic': 2}

CAP_SHAPE_FIT_DEGREE = _CAP_SHAPE_FIT_DEGREES[CAP_SHAPE_FIT_MODEL]

RUN_CAP_MCMC = True

CAP_MCMC_N_WALKERS = 64

CAP_MCMC_N_STEPS = 6000

CAP_MCMC_BURN_IN = 1500

CAP_MCMC_THIN = 5

CAP_MCMC_INIT_BALL_SCALE = 0.0001

CAP_MCMC_SEED = SEED

CAP_MCMC_PROGRESS = True

CAP_POSTERIOR_N_CURVES = 100

CAP_POSTERIOR_CURVE_ALPHA = 0.08

CAP_POSTERIOR_CURVE_LW = 0.8

SUMMARY_DIR = './run_output'

RADIO_STACK_KEY = 'stack_unori'

MASS_WEIGHT_N_BINS = 20

HIST_N_BINS = MASS_WEIGHT_N_BINS

HIST_YLIMS = {'ba_selected': (0.0, 0.8)}

BA_HIST_MIN = 0.0

BA_HIST_MAX = 1.0

STACK_COLOR_PERCENTILE_LOW = 1

STACK_COLOR_PERCENTILE_HIGH = 99

SAVEFIG_DPI = 220

SAVEFIG_BBOX_INCHES = 'tight'

SAVEFIG_PAD_INCHES = 0.0

DISPLAY_Y_SCALE = 1000000.0

STACK_NO_DATA_SIZE = 9

STACK_XY_TICKS = np.array([-12, -8, -4, 0, 4, 8, 12])

STACK_COLORBAR_TICK_NBINS = 9

STACK_COLORBAR_LABEL_SIZE = 13

STACK_COLORBAR_TICK_SIZE = 11

STACK_COLORBAR_EXPONENT_LABEL = '$10^{\\text{\\textminus} 6}$'

STACK_COLORBAR_EXPONENT_SIZE = 13

STACK_COLORBAR_EXPONENT_PAD = 6

# Display every CAP MCMC coefficient in fixed 10^{-6} units.  Scaling the
# corner samples before plotting prevents Matplotlib/corner from introducing
# automatic 1e-7, e-7, or other e-notation anywhere in the PDF.
CAP_MCMC_DISPLAY_SCALE = DISPLAY_Y_SCALE

CAP_MCMC_EXPONENT = -6

CAP_MCMC_TITLE_DECIMALS = 4

# Corner-plot diagonal title size.  This was hardcoded to 8, which pushed the
# scriptsize super/subscripts down to ~5.6pt, where Computer Modern's minus
# thins to a hairline and PDF viewers drop it.  Keep this at 11 or above.
CAP_MCMC_TITLE_SIZE = 8

CAP_MCMC_SUPTITLE_SIZE = 16

CAP_MCMC_TICK_DECIMALS = 4

STACK_COLORBAR_WIDTH = 0.018

STACK_COLORBAR_PAD = 0.012

STACK_N_LABEL_X = 0.04

STACK_N_LABEL_Y = 0.94

STACK_N_LABEL_BBOX_ALPHA = 0.75

STACK_N_LABEL_BBOX_PAD = 1.5

STACK_FULL_PANEL_TITLE_SIZE = 12

STACK_FULL_PANEL_TITLE_PAD = 18

STACK_FULL_SUPTITLE_SIZE = 16

STACK_FULL_AXIS_LABEL_SIZE = 15

STACK_FULL_TICK_LABEL_SIZE = 12

STACK_FULL_N_LABEL_SIZE = 10

STACK_FULL_ROW_LABEL_SIZE = 16

STACK_RADIO_PANEL_TITLE_SIZE = 12

STACK_RADIO_PANEL_TITLE_PAD = 18

STACK_RADIO_SUPTITLE_SIZE = 16

STACK_RADIO_AXIS_LABEL_SIZE = 15

STACK_RADIO_TICK_LABEL_SIZE = 12

STACK_RADIO_N_LABEL_SIZE = 10

STACK_FULL_FIG_WIDTH_PER_COL = 4.0

STACK_FULL_FIG_HEIGHT_PER_ROW = 3.75

STACK_FULL_GRID_LEFT = 0.075

STACK_FULL_GRID_BOTTOM = 0.08

STACK_FULL_GRID_TOP = 0.86

STACK_FULL_ROW_LABEL_X = -0.25

STACK_FULL_CBAR_BOTTOM = 0.18

STACK_FULL_CBAR_HEIGHT = 0.62

STACK_FULL_SUPTITLE_Y = 0.96

STACK_RADIO_FIG_WIDTH_PER_COL = 3.5

STACK_RADIO_FIG_HEIGHT = 3.75

STACK_RADIO_GRID_LEFT = 0.07

STACK_RADIO_GRID_BOTTOM = 0.12

STACK_RADIO_GRID_TOP = 0.78

STACK_RADIO_CBAR_BOTTOM = 0.2

STACK_RADIO_CBAR_HEIGHT = 0.56

STACK_RADIO_SUPTITLE_Y = 0.975

CAP_FIG_WIDTH_PER_COL = 6.35

CAP_FIG_HEIGHT = 5.25

CAP_LEFT = 0.08

CAP_RIGHT = 0.98

CAP_BOTTOM = 0.15

CAP_SECTOR_TOP = 0.78

CAP_WSPACE = 0.0

CAP_PANEL_TITLE_SIZE = 18

CAP_PANEL_TITLE_PAD = 14

CAP_SUPTITLE_SIZE = 20

CAP_SECTOR_SUPTITLE_Y = 0.94

CAP_AXIS_LABEL_SIZE_SECTOR = 18

CAP_TICK_LABEL_SIZE = 16

CAP_Y_TICK_NBINS = 10

CAP_YLABEL_X = -0.1

CAP_YLABEL_Y = 0.57

CAP_LEGEND_LOC_SECTOR = 'upper left'

CAP_LEGEND_SIZE_SECTOR = 15

CAP_ERROR_CAPSIZE = 3

CAP_ERROR_LW = 1.3

CAP_ERROR_MARKER_SIZE = 5

CAP_ZERO_LINE_COLOR = '0.5'

CAP_ZERO_LINE_WIDTH = 0.6

CAP_ZERO_LINE_STYLE = '--'

HIST_FIG_WIDTH_PER_COL = 6.35

HIST_FIG_HEIGHT = 5.25

HIST_LEFT = 0.08

HIST_RIGHT = 0.98

HIST_BOTTOM = 0.15

HIST_TOP = 0.8

HIST_WSPACE = 0.0

HIST_PANEL_TITLE_SIZE = 17

HIST_PANEL_TITLE_PAD = 14

HIST_SUPTITLE_SIZE = 22

HIST_SUPTITLE_Y = 0.965

HIST_AXIS_LABEL_SIZE = 20

HIST_TICK_LABEL_SIZE = 15

HIST_X_TICK_NBINS = 10

HIST_YLABEL_X = -0.09

HIST_YLABEL_Y = 0.57

HIST_LINEWIDTH = 1.8

HIST_MEDIAN_LINEWIDTH = 1.8

HIST_MEDIAN_LINESTYLE = '--'

HIST_MEDIAN_LABEL_SIZE = 14

CAP_CORR_CMAP = 'YlOrRd'

CAP_CORR_VMIN = 0.0

CAP_CORR_VMAX = 1.0

CAP_CORR_FIG_WIDTH_PER_COL = 5.4

CAP_CORR_FIG_HEIGHT_PER_ROW = 5.4

CAP_CORR_WSPACE = 0.12

CAP_CORR_N_TICKS = 5

CAP_CORR_TICK_LABEL_SIZE = 12

CAP_CORR_AXIS_LABEL_SIZE = 14

CAP_CORR_PANEL_TITLE_SIZE = 16

CAP_CORR_PANEL_TITLE_PAD = 10

CAP_CORR_CBAR_LABEL_SIZE = 14

CAP_CORR_SUPTITLE = '{\\rm Oriented-stack\\ CAP\\ Correlation\\ Matrices}'

CAP_CORR_SUPTITLE_Y = 0.98

MASS_BIN_LABEL_TEMPLATE = '${mass_lo:.1f} < \\log_{{10}}\\!\\left(\\frac{{M_\\ast}}{{M_\\odot}}\\right) \\leq {mass_hi:.1f}$'

N_STACKED_LABEL_TEMPLATE = '$N_{{\\rm stacked}} = {n_stacked:,}$'

STACK_X_LABEL = '$x\\ {\\rm [arcmin]}$'

STACK_Y_LABEL = '$y\\ {\\rm [arcmin]}$'

STACK_COLORBAR_LABEL = '$y\\ {\\rm [dimensionless]}$'

STACK_NO_DATA_LABEL = '{\\rm no\\ data}'

STACK_ROW_LABEL_UNORIENTED = '{\\rm Unoriented}'

STACK_ROW_LABEL_ORIENTED = '{\\rm Oriented}'

STACK_FULL_SUPTITLE = '{\\rm Oriented\\ Stacked\\ Compton\\ }$y${\\rm\\ Maps}'

STACK_RADIO_SUPTITLE = '{\\rm FIRST-matched\\ Radio-source\\ Stacked\\ Compton\\ }$y${\\rm\\ Maps}'

CAP_X_LABEL = '$\\theta_{\\rm d}\\ [{\\rm arcmin}]$'

CAP_Y_LABEL = '$y_{\\mathrm{CAP}}\\ [10^{\\text{\\textminus} 6}\\,\\mathrm{arcmin}^{2}]$'

CAP_NO_SECTOR_DATA_LABEL = '{\\rm no\\ sector\\ CAP\\ data}'

CAP_MAJOR_SECTOR_LABEL = '{\\rm Major-axis\\ sector}'

CAP_MINOR_SECTOR_LABEL = '{\\rm Minor-axis\\ sector}'

CAP_SECTOR_SUPTITLE = '{\\rm Oriented-stack\\ CAP\\ Profiles:\\ Major\\ vs.\\ Minor\\ Axis}'

HIST_FRACTION_Y_LABEL = '$\\mathrm{Fraction\\ per\\ bin}$'

HIST_BA_X_LABEL = '$b/a$'

HIST_SELECTED_BA_SUPTITLE = '{\\rm Oriented\\ Sample\\ Axis\\ Ratio\\ Distribution}'

CAP_FIT_LINE_LW = 2.0

_CAP_FIT_MODEL_TEX = {'linear': 'Linear', 'quadratic': 'Quadratic'}[CAP_SHAPE_FIT_MODEL]

CAP_FIT_SECTOR_SUPTITLE = '{\\rm Oriented-stack\\ CAP\\ ' + _CAP_FIT_MODEL_TEX + '\\ Fit:\\ ML\\ Curve\\ and\\ Posterior\\ Draws}'

CACHE_FILE = os.path.join(CACHE_DIR, 'stamps.h5')

ORIENTED_PDF_NAMES = ['summary_oriented_full_stack_2x3.pdf', 'summary_oriented_full_stack_cap_profiles_1x3.pdf', 'summary_oriented_sector_cap_correlation_2x3.pdf', 'summary_oriented_sector_posterior_curves_1x3.pdf', 'summary_oriented_hist_ba_selected_1x3.pdf', 'summary_radio_full_stack_1x3.pdf']

if CAP_SHAPE_FIT_MODEL not in _CAP_SHAPE_FIT_DEGREES:
    raise ValueError(
        "CAP_SHAPE_FIT_MODEL must be either 'linear' or 'quadratic'; "
        f"got {CAP_SHAPE_FIT_MODEL!r}."
    )
    
    
# --- Combined corner-overlay page (not strict A4 — width fixed, height auto-cropped) ---
CAP_MCMC_A4_WIDTH_IN = 8.27
CAP_MCMC_A4_HEIGHT_IN = 9.0            # scratch height; bbox_inches='tight' crops the real output
CAP_MCMC_A4_NCOLS = 2                   # 3 bins -> 2x2 grid, one empty cell
CAP_MCMC_A4_WSPACE = 0.04
CAP_MCMC_A4_HSPACE = -0.06              # negative pulls rows together, closing the row gap
CAP_MCMC_A4_CORNER_WHSPACE = 0.06
CAP_MCMC_A4_TITLE_SIZE = 7
CAP_MCMC_A4_MASSLABEL_SIZE = 13
CAP_MCMC_A4_MASSLABEL_X = 0.82           # moved right from 0.70
CAP_MCMC_A4_MASSLABEL_Y = 0.82
CAP_MCMC_A4_LEGEND_CELL = 1              # flattened row-major grid index -> top-right in a 2x2
CAP_MCMC_A4_SUPTITLE = '{\\rm Oriented-stack\\ CAP\\ Posterior\\ Overlays}'
CAP_MCMC_A4_SUPTITLE_Y = 1.005


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def _savefig(path):
    """Save figures using centralized trimming settings."""
    plt.savefig(path, dpi=SAVEFIG_DPI, bbox_inches=SAVEFIG_BBOX_INCHES, pad_inches=SAVEFIG_PAD_INCHES)

def _fmt_pct(n, d):
    """Format n/d as a percentage string."""
    if d == 0:
        return 'nan%'
    return f'{100.0 * n / d:.2f}%'

def _mass_bin_mask(logm, mass_lo, mass_hi):
    """Mass bin mask using nonoverlapping bins: mass_lo < logM <= mass_hi."""
    return np.isfinite(logm) & (logm > mass_lo) & (logm <= mass_hi)

def _mass_bin_mask_for_accounting(logm, mass_lo, mass_hi):
    """Mass bin mask used only for printed accounting."""
    return _mass_bin_mask(logm, mass_lo, mass_hi)

def print_mass_bin_accounting(label, mask, logm, previous_mask=None, indent='  '):
    """
    Print total and per mass bin counts for a Boolean mask.

    If previous_mask is provided, also print how many objects were lost
    relative to that previous cumulative stage.
    """
    mask = np.asarray(mask, dtype=bool)
    logm = np.asarray(logm)
    total = int(np.sum(mask))
    if previous_mask is None:
        print(f'{indent}{label}: {total:,}')
    else:
        previous_mask = np.asarray(previous_mask, dtype=bool)
        previous_total = int(np.sum(previous_mask))
        lost = previous_total - total
        print(f'{indent}{label}: {total:,} (lost {lost:,} from previous, kept {_fmt_pct(total, previous_total)})')
    for mass_lo, mass_hi in MASS_BINS:
        in_bin = _mass_bin_mask_for_accounting(logm, mass_lo, mass_hi)
        n_bin = int(np.sum(mask & in_bin))
        if previous_mask is None:
            print(f'{indent}  logM ({mass_lo:.1f}, {mass_hi:.1f}]: {n_bin:,}')
        else:
            prev_bin = int(np.sum(previous_mask & in_bin))
            lost_bin = prev_bin - n_bin
            print(f'{indent}  logM ({mass_lo:.1f}, {mass_hi:.1f}]: {n_bin:,} (lost {lost_bin:,}, kept {_fmt_pct(n_bin, prev_bin)})')

def print_cumulative_cut_table(stages, logm, final_label=None, mass_bins=MASS_BINS):
    """Print a compact cumulative selection table.

    stages : list of (label, cumulative_boolean_mask). Each mask is the
        running survivor set up to and including that stage, defined on the
        full Firefly row array. "Total" is the mask sum; the mass-bin columns
        split that survivor set by stellar mass.
    logm : per-row log10 stellar mass, used only for the mass-bin split.
    final_label : if given, repeat the last stage's mask under this label.
    """
    logm = np.asarray(logm)
    bin_masks = [_mass_bin_mask(logm, lo, hi) for lo, hi in mass_bins]
    bin_headers = [f'({lo:.1f},{hi:.1f}]' for lo, hi in mass_bins]
    label_w = max([len('Cut stage')] + [len(lbl) for lbl, _ in stages]) + 2
    if final_label is not None:
        label_w = max(label_w, len(final_label) + 2)
    total_w = 12
    bin_w = 13
    header = f"{'Cut stage':<{label_w}}{'Total':>{total_w}}" + ''.join((f'{h:>{bin_w}}' for h in bin_headers))
    rule = '-' * len(header)

    def _row(label, mask):
        mask = np.asarray(mask, dtype=bool)
        total = int(np.sum(mask))
        cells = ''.join((f'{int(np.sum(mask & bm)):>{bin_w},}' for bm in bin_masks))
        print(f'{label:<{label_w}}{total:>{total_w},}{cells}')
    print('\nCumulative selection (mass bins: mass_lo < log10(M*/Msun) <= mass_hi)')
    print(header)
    print(rule)
    for label, mask in stages:
        _row(label, mask)
    if final_label is not None:
        print(rule)
        _row(final_label, stages[-1][1])

def _cap_to_plot_units(values):
    """CAP values are already in y arcmin^2."""
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)

def _cap_plot_axis_label():
    """CAP axis label for the y arcmin^2 convention."""
    return CAP_Y_LABEL

def print_cap_profile_mean_sd_table(mass_lo, mass_hi, major_mean, major_std, minor_mean, minor_std):
    """Print the nine plotted major/minor CAP means and bootstrap SDs.

    Values are multiplied by ``DISPLAY_Y_SCALE`` so the numbers use the same
    10^{-6} convention shown on the CAP-profile plot axis.
    """
    arrays = [major_mean, major_std, minor_mean, minor_std]
    if any(values is None for values in arrays):
        print('    [CAP profile table] unavailable: one or more profile arrays are missing')
        return

    major_mean, major_std, minor_mean, minor_std = [
        np.asarray(values, dtype=np.float64) for values in arrays
    ]
    n_ap = len(CAP_RADII_ARCMIN)
    if any(values.shape != (n_ap,) for values in (major_mean, major_std, minor_mean, minor_std)):
        shapes = [values.shape for values in (major_mean, major_std, minor_mean, minor_std)]
        raise ValueError(
            f'CAP profile arrays must each have shape ({n_ap},); got {shapes}'
        )

    scale = float(DISPLAY_Y_SCALE)
    print(
        f'\n    Plotted CAP mean and bootstrap SD: '
        f'logM ({mass_lo:.1f}, {mass_hi:.1f}] '
        f'[values in 10^{{-6}} y arcmin^2]'
    )
    header = (
        f"    {'theta [arcmin]':>14}"
        f"{'major mean':>16}{'major SD':>16}"
        f"{'minor mean':>16}{'minor SD':>16}"
    )
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for radius, maj_m, maj_s, min_m, min_s in zip(
        CAP_RADII_ARCMIN, major_mean, major_std, minor_mean, minor_std
    ):
        print(
            f'{radius:18.2f}'
            f'{maj_m * scale:16.8f}{maj_s * scale:16.8f}'
            f'{min_m * scale:16.8f}{min_s * scale:16.8f}'
        )

def _cap_pixel_area_arcmin2(pixscale_arcmin):
    """Projected local cutout pixel area in arcmin^2."""
    return float(pixscale_arcmin) ** 2

def _write_cap_area_metadata(h5f, pixscale_arcmin):
    """Store CAP unit metadata in the HDF5 cache."""
    h5f.attrs['cap_pixel_area_arcmin2'] = _cap_pixel_area_arcmin2(pixscale_arcmin)
    h5f.attrs['cap_internal_unit'] = 'y_arcmin2'
    h5f.attrs['cap_plot_unit'] = 'yarcmin2'
    h5f.attrs['cap_plot_scale_from_internal'] = 1.0
    h5f.flush()

def _centered_pixel_axis(n, pixscale_arcmin):
    """Pixel-center coordinates in arcmin, centered on zero."""
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(pixscale_arcmin)

def sample_large_stamp_to_output(source_stamp, angle_deg, out_ny, out_nx, out_pixscale_arcmin, source_radius_arcmin=STAMP_SOURCE_RADIUS_ARCMIN, fill_value=0.0):
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
    theta = -np.deg2rad(angle_deg)
    ca = np.cos(theta)
    sa = np.sin(theta)
    xg_src = ca * xg_out - sa * yg_out
    yg_src = sa * xg_out + ca * yg_out
    inside = (yg_src >= y_src[0]) & (yg_src <= y_src[-1]) & (xg_src >= x_src[0]) & (xg_src <= x_src[-1])
    out = np.full((out_ny, out_nx), fill_value, dtype=np.float64)
    if np.any(inside):
        interp = RectBivariateSpline(y_src, x_src, src, kx=1, ky=1)
        out[inside] = interp(yg_src[inside], xg_src[inside], grid=False)
    return out.astype(np.float64)

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
            raise ValueError('weights must have the same length as profiles')
        valid &= np.isfinite(w_all) & (w_all > 0.0)
    p = profiles[valid]
    w = w_all[valid]
    n = len(p)
    if n == 0 or np.sum(w) <= 0.0:
        mean = np.full(n_ap, np.nan, dtype=np.float64)
        std = np.full(n_ap, np.nan, dtype=np.float64)
        cov = np.full((n_ap, n_ap), np.nan, dtype=np.float64)
        return (mean, std, cov, 0)
    mean = np.average(p, axis=0, weights=w)
    if not RUN_BOOTSTRAP or n < 2:
        std = np.zeros(n_ap, dtype=np.float64)
        cov = np.zeros((n_ap, n_ap), dtype=np.float64)
        return (mean.astype(np.float64), std, cov, n)
    rng = np.random.default_rng(seed)
    boot = np.empty((N_BOOT, n_ap), dtype=np.float64)
    for b in range(N_BOOT):
        draw = rng.integers(0, n, size=n)
        boot[b] = np.average(p[draw], axis=0, weights=w[draw])
    cov = np.cov(boot, rowvar=False)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return (mean.astype(np.float64), std.astype(np.float64), cov.astype(np.float64), n)

def make_angle_map(ny, nx):
    """Angle from stamp center: 0 is right, 90 is up."""
    cy, cx = (ny // 2, nx // 2)
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

def compute_cap_values(image, r_map, pixscale, cap_radii_arcmin, pixel_area, sec_mask=None):
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

def _normalize_fits_text(values):
    """Return an uppercase, whitespace-stripped Unicode FITS text column."""
    values = np.asarray(values)
    if values.dtype.kind == 'S':
        text = np.char.decode(values, 'utf-8', errors='ignore')
    else:
        text = values.astype(str)
    return np.char.upper(np.char.strip(text))

def load_firefly_full(filepath, class_field=FIREFLY_CLASS_FIELD, galaxy_class=FIREFLY_GALAXY_CLASS):
    """Load only the Firefly columns needed by the oriented pipeline."""
    with fits.open(filepath, memmap=True) as hdu:
        data = hdu[1].data
        n_rows = len(data)
        field_lookup = {name.upper(): name for name in data.names}
        class_key = field_lookup.get(class_field.upper())
        if class_key is None:
            raise KeyError(f"Firefly catalog has no {class_field!r} column. Available columns include: {', '.join(data.names[:20])}")
        object_class = _normalize_fits_text(data[class_key])
        is_galaxy = object_class == galaxy_class.upper()
        ra = data['PLUG_RA'].astype(np.float64)
        dec = data['PLUG_DEC'].astype(np.float64)
        mstar = data['Chabrier_MILES_stellar_mass'].astype(np.float64)
        z = data['Z'].astype(np.float64)
    log_mstar = np.log10(mstar)
    fits_idx = np.arange(n_rows, dtype=np.int64)
    print(f'  Firefly rows read: {n_rows:,}')
    print(f'  Firefly {class_key}={galaxy_class.upper()} galaxies: {int(np.sum(is_galaxy)):,}')
    return (ra, dec, log_mstar, z, fits_idx, is_galaxy)

def load_photo_shapes(firefly_ra, firefly_dec, photo_path, match_arcsec=1.0, fracdev_thresh=0.5, type_galaxy=3, safety=1):
    """Cross match to SDSS photometric catalog for PA and axis ratio.

    Returns both the selected shape used for oriented stacking and the
    de Vaucouleurs/exponential model quantities needed for diagnostics.
    """
    print(f'  Loading photo catalog: {photo_path}')
    with fits.open(photo_path, memmap=True) as hdu:
        photo = hdu[1].data
        ra_p = photo['RA'].astype(np.float64)
        dec_p = photo['DEC'].astype(np.float64)
        phi_dev = photo['PHI_DEV_DEG'][:, 2].astype(np.float64)
        phi_exp = photo['PHI_EXP_DEG'][:, 2].astype(np.float64)
        ab_dev = photo['AB_DEV'][:, 2].astype(np.float64)
        ab_exp = photo['AB_EXP'][:, 2].astype(np.float64)
        fracdev = photo['FRACDEV'][:, 2].astype(np.float64)
        ptype = photo['TYPE'][:, 2].astype(np.int16)
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
    print(f'  Photo matched: {good.sum():,} / {n_ff:,}')
    use_dev = fracdev[idx] > fracdev_thresh
    pa = np.where(use_dev, phi_dev[idx], phi_exp[idx])
    ab = np.where(use_dev, ab_dev[idx], ab_exp[idx])
    valid_shape = good & (pa > -900) & (pa < 360) & (ab > 0) & (ab <= 1) & (ptype[idx] == type_galaxy)
    print(f'  Valid shapes: {valid_shape.sum():,}')
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
    return (pa_out, ab_out, valid_shape, pa_dev_out, pa_exp_out, ab_dev_out, ab_exp_out, fracdev_out)

def crossmatch_to_first(ra, dec, first_path, match_arcsec, safety=1):
    """Return a mask for galaxies with a nearby FIRST galaxy source."""
    print('  Loading FIRST catalog...')
    with fits.open(first_path, memmap=True) as hdu:
        f = hdu[1].data
        ra_first = f['RA'].astype(np.float64)
        dec_first = f['DEC'].astype(np.float64)
        sdss_cls = np.char.strip(f['SDSS_CLASS'].astype(str))
    is_gal = sdss_cls == 'g'
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
    print(f'  FIRST cross match ({match_arcsec:.0f} arcsec): {has_radio.sum():,} / {len(ra):,}')
    return has_radio

def _extraction_config_dict():
    return {'cache_version': 'oriented_standalone_v1', 'pipeline': 'oriented', 'tsz_map_path': TSZ_MAP_PATH, 'stamp_radius_arcmin': STAMP_RADIUS_ARCMIN, 'stamp_source_radius_arcmin': float(STAMP_SOURCE_RADIUS_ARCMIN), 'rotation_method': 'large_source_rectbivariatespline_k1'}

def _extraction_config_hash():
    cfg = _extraction_config_dict()
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

def _open_or_create_cache(cache_path, ny, nx, ny_src, nx_src, config_hash):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        h5f = h5py.File(cache_path, 'a')
        stored_hash = h5f.attrs.get('config_hash', '')
        if stored_hash != config_hash:
            print('  [cache] Config hash mismatch, rebuilding oriented cache.')
            h5f.close()
            os.remove(cache_path)
            return _open_or_create_cache(cache_path, ny, nx, ny_src, nx_src, config_hash)
        existing_ids = set(h5f['fits_idx'][:].tolist())
        print(f'  [cache] Opened oriented cache with {len(existing_ids):,} galaxies')
        return (h5f, existing_ids)
    print(f'  [cache] Creating oriented cache: {cache_path}')
    h5f = h5py.File(cache_path, 'w')
    h5f.attrs['config_hash'] = config_hash
    h5f.attrs['config_json'] = json.dumps(_extraction_config_dict(), sort_keys=True)
    h5f.attrs['ny'] = ny
    h5f.attrs['nx'] = nx
    h5f.attrs['ny_src'] = ny_src
    h5f.attrs['nx_src'] = nx_src
    h5f.attrs['stamp_radius_arcmin'] = float(STAMP_RADIUS_ARCMIN)
    h5f.attrs['stamp_source_radius_arcmin'] = float(STAMP_SOURCE_RADIUS_ARCMIN)
    scalar_keys = ['fits_idx', 'ra', 'dec', 'z', 'logm', 'pa', 'ab', 'pa_dev', 'pa_exp', 'ab_dev', 'ab_exp', 'fracdev', 'has_radio', 'extract_attempted', 'stamp_valid']
    for key in scalar_keys:
        dtype = np.float64
        if key == 'fits_idx':
            dtype = np.int64
        elif key in ('has_radio', 'extract_attempted', 'stamp_valid'):
            dtype = bool
        h5f.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)
    h5f.create_dataset('stamps', shape=(0, ny_src, nx_src), maxshape=(None, ny_src, nx_src), dtype=np.float64, chunks=(1, ny_src, nx_src), compression='gzip', compression_opts=4)
    return (h5f, set())

def _append_to_cache(h5f, n_new, fits_idx, ra, dec, z, logm, pa, ab, pa_dev, pa_exp, ab_dev, ab_exp, fracdev, has_radio):
    n_existing = h5f['fits_idx'].shape[0]
    n_total = n_existing + n_new
    keys = ['fits_idx', 'ra', 'dec', 'z', 'logm', 'pa', 'ab', 'pa_dev', 'pa_exp', 'ab_dev', 'ab_exp', 'fracdev', 'has_radio', 'extract_attempted', 'stamp_valid']
    for key in keys:
        h5f[key].resize(n_total, axis=0)
    h5f['stamps'].resize(n_total, axis=0)
    sl = slice(n_existing, n_total)
    h5f['fits_idx'][sl] = fits_idx
    h5f['ra'][sl] = ra
    h5f['dec'][sl] = dec
    h5f['z'][sl] = z
    h5f['logm'][sl] = logm
    h5f['pa'][sl] = pa
    h5f['ab'][sl] = ab
    h5f['pa_dev'][sl] = pa_dev
    h5f['pa_exp'][sl] = pa_exp
    h5f['ab_dev'][sl] = ab_dev
    h5f['ab_exp'][sl] = ab_exp
    h5f['fracdev'][sl] = fracdev
    h5f['has_radio'][sl] = has_radio
    h5f['extract_attempted'][sl] = False
    h5f['stamp_valid'][sl] = False
    h5f.flush()

def extract_to_cache(comptony, h5f):
    n_total = h5f['fits_idx'].shape[0]
    ny_src = int(h5f.attrs.get('ny_src', h5f.attrs['ny']))
    nx_src = int(h5f.attrs.get('nx_src', h5f.attrs['nx']))
    attempted = h5f['extract_attempted'][:]
    n_todo = int((~attempted).sum())
    if n_todo == 0:
        print('  [extract] All galaxies already cached.')
        return
    print(f'  [extract] {n_todo:,} galaxies to extract')
    ra_all = h5f['ra'][:]
    dec_all = h5f['dec'][:]
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
        h5f['extract_attempted'][i] = True
        if stamp is None:
            h5f['stamp_valid'][i] = False
            n_done += 1
            continue
        arr = np.array(stamp, dtype=np.float64)
        if arr.shape != (ny_src, nx_src):
            n_shape_mismatch += 1
            old_shape = arr.shape
            arr = arr[:ny_src, :nx_src]
            print(f'  [extract] shape mismatch at cache row {i}: got {old_shape}, expected {(ny_src, nx_src)}, after crop {arr.shape}')
        if arr.shape != (ny_src, nx_src) or np.all(arr == 0):
            h5f['stamp_valid'][i] = False
            n_done += 1
            continue
        h5f['stamps'][i] = arr
        h5f['stamp_valid'][i] = True
        n_done += 1
        n_success += 1
        if n_done % EXTRACTION_BATCH_LOG_EVERY == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-06)
            print(f'    extracted {n_done:,}/{n_todo:,}, success={n_success:,}, rate={rate:.0f}/s')
            h5f.flush()
    h5f.flush()
    dt = time.time() - t0
    print(f'  [extract] Done: {n_success:,} successful / {n_done:,} attempted in {dt / 60:.1f} min')
    print(f'  [extract] Shape mismatches encountered: {n_shape_mismatch:,}')

def stack_from_cache(h5f, mask, label='', weights=None):
    """Stack for a selected mask, optionally using per-galaxy weights.

    Cached stamps are larger source thumbnails.  Each selected source
    thumbnail is sampled onto the final output grid, once with angle 0 for
    the unoriented stack and once with -PA for the oriented stack.
    """
    ny = int(h5f.attrs['ny'])
    nx = int(h5f.attrs['nx'])
    stamp_valid = h5f['stamp_valid'][:]
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
            raise ValueError('weights must have the same length as mask')
        selected_weights = weights[indices].astype(np.float64)
        bad_weight = ~np.isfinite(selected_weights) | (selected_weights <= 0.0)
        if np.any(bad_weight):
            print(f'  [stack accounting: {label}] rejecting {int(np.sum(bad_weight)):,} nonpositive or nonfinite weights')
            keep = ~bad_weight
            indices = indices[keep]
            selected_weights = selected_weights[keep]
            n_sel = len(indices)
            if n_sel == 0:
                return {'n_success': 0, 'effective_mask': effective_mask}
        effective_mask = np.zeros_like(mask, dtype=bool)
        effective_mask[indices] = True
    sum_w = float(np.sum(selected_weights))
    label_txt = label if label else 'unnamed selection'
    print(f'  [stack accounting: {label_txt}] candidates before stamp-valid cut: {n_candidates:,}')
    print(f'  [stack accounting: {label_txt}] rejected by stamp_valid=False: {n_rejected_stamp_valid:,}')
    print(f'  [stack accounting: {label_txt}] used for stacking: {n_sel:,}')
    if weights is not None and n_sel > 0:
        print(f'  [stack accounting: {label_txt}] mass weights: sum={sum_w:.6e}, min={np.nanmin(selected_weights):.3e}, median={np.nanmedian(selected_weights):.3e}, max={np.nanmax(selected_weights):.3e}')
    if n_sel == 0:
        return {'n_success': 0, 'effective_mask': effective_mask}
    pixscale = 2.0 * STAMP_RADIUS_ARCMIN / (ny - 1)
    cap_pixel_area = _cap_pixel_area_arcmin2(pixscale)
    print(f'  [CAP units: {label_txt}] internal unit=y arcmin^2, pixel_area={cap_pixel_area:.6e} arcmin^2')
    cy, cx = (ny // 2, nx // 2)
    yy, xx = np.mgrid[:ny, :nx]
    r_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float64)
    angle_map = make_angle_map(ny, nx)
    major_mask = sector_mask(angle_map, MAJOR_AXIS_ANGLE, WEDGE_HALF_DEG)
    minor_mask = sector_mask(angle_map, MINOR_AXIS_ANGLE, WEDGE_HALF_DEG)
    pa_all = h5f['pa'][:]
    stack_sum_unori = np.zeros((ny, nx), dtype=np.float64)
    stack_sum_ori = np.zeros((ny, nx), dtype=np.float64)
    n_ap = len(CAP_RADII_ARCMIN)
    cap_full_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)
    cap_major_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)
    cap_minor_values = np.full((n_sel, n_ap), np.nan, dtype=np.float64)
    t0 = time.time()
    for j, cache_idx in enumerate(indices):
        source_stamp = np.asarray(h5f['stamps'][cache_idx], dtype=np.float64)
        pa_j = float(pa_all[cache_idx])
        stamp = sample_large_stamp_to_output(source_stamp, 0.0, ny, nx, pixscale)
        stamp_rot = sample_large_stamp_to_output(source_stamp, -pa_j, ny, nx, pixscale)
        w_j = selected_weights[j]
        stack_sum_unori += w_j * stamp.astype(np.float64)
        stack_sum_ori += w_j * stamp_rot.astype(np.float64)
        cap_full_values[j] = compute_cap_values(stamp, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area)
        cap_major_values[j] = compute_cap_values(stamp_rot, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area, sec_mask=major_mask)
        cap_minor_values[j] = compute_cap_values(stamp_rot, r_map, pixscale, CAP_RADII_ARCMIN, cap_pixel_area, sec_mask=minor_mask)
        if (j + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f'    stacked {j + 1:,}/{n_sel:,}, rate={(j + 1) / max(dt, 1e-06):.0f}/s')
    print(f'  [stack: {label_txt}] {n_sel:,} galaxies used in final stack')
    return {'n_success': n_sel, 'ny': ny, 'nx': nx, 'pixscale': pixscale, 'stack_unori': (stack_sum_unori / sum_w).astype(np.float64), 'stack_ori': (stack_sum_ori / sum_w).astype(np.float64), 'weights': selected_weights.astype(np.float64), 'sum_weights': sum_w, 'cap_full_values': cap_full_values, 'cap_major_values': cap_major_values, 'cap_minor_values': cap_minor_values, 'effective_mask': effective_mask}

def _visible_tex_minus():
    return '\\text{\\textminus}'

def _tex_visible_exponent(exponent):
    """Render an integer exponent for use inside ``10^{...}`` with the
    visible minus sign instead of a bare ``-``."""
    exponent = int(exponent)
    if exponent < 0:
        return f'{_visible_tex_minus()}{abs(exponent)}'
    return f'{exponent}'

def _latex_visible_minus_number(value, precision=3, zero_tol=1e-12):
    """Return a LaTeX math-mode number using the visible minus glyph.

    See _visible_tex_minus for why the ordinary math minus is avoided.
    """
    if not np.isfinite(value):
        return ''
    value = float(value)
    if abs(value) < zero_tol:
        return '$0$'
    sign = _visible_tex_minus() if value < 0.0 else ''
    value_abs = abs(value)
    if abs(value_abs - round(value_abs)) < 1e-08:
        body = f'{int(round(value_abs))}'
    else:
        body = f'{value_abs:.{precision}g}'
    return f'${sign}{body}$'

def _tex_scaled_tick(x, pos=None):
    """Format tick values after scaling by DISPLAY_Y_SCALE.

    The plotted data remain in their original units.  Only the tick labels
    are multiplied by 1e6, while the axis/colorbar labels carry the 10^{-6}
    factor.
    """
    return _latex_visible_minus_number(x * DISPLAY_Y_SCALE, precision=3)

def _tex_unscaled_tick(x, pos=None):
    """Format ordinary stacked-map x/y ticks with visible minus signs."""
    return _latex_visible_minus_number(x, precision=3)

def _fixed_decimal_body(value, decimals, strip_zeros):
    """Fixed-point number, visible minus, never e notation.

    Returns bare TeX (no surrounding $ ... $) so it can be dropped either into
    its own math span or into a larger one such as a corner-plot title.
    """
    value = float(value)
    decimals = int(decimals)
    if abs(value) < 0.5 * 10.0 ** (-decimals):
        return '0' if strip_zeros else f'{0.0:.{decimals}f}'
    sign = _visible_tex_minus() if value < 0.0 else ''
    body = f'{abs(value):.{decimals}f}'
    if strip_zeros:
        body = body.rstrip('0').rstrip('.') or '0'
    return f'{sign}{body}'

def _tex_cap_mcmc_tick(x, pos=None):
    """Tick formatter for the already-scaled CAP MCMC corner axes."""
    if not np.isfinite(x):
        return ''
    return f'${_fixed_decimal_body(x, CAP_MCMC_TICK_DECIMALS, strip_zeros=True)}$'

def _cap_mcmc_parameter_label(name, sector_subscript, degree):
    """Corner-axis label with the fixed 10^{-6} coefficient unit.

    For y_CAP in y arcmin^2 and a theta^p term, the coefficient carries
    arcmin^(2-p).  The displayed numerical value is the physical coefficient
    divided by 10^{-6}.
    """
    exponent_tex = _tex_visible_exponent(CAP_MCMC_EXPONENT)
    names = _cap_coefficient_names(degree)
    try:
        index = names.index(name)
    except ValueError:
        return f'${name}_{{\\rm {sector_subscript}}}\\,[10^{{{exponent_tex}}}]$'
    power = degree - index
    arcmin_power = 2 - power
    if arcmin_power == 0:
        unit = ''
    elif arcmin_power == 1:
        unit = '\\,\\mathrm{arcmin}'
    else:
        unit = f'\\,\\mathrm{{arcmin}}^{{{arcmin_power}}}'
    return f'${name}_{{\\rm {sector_subscript}}}\\,[10^{{{exponent_tex}}}{unit}]$'

def _set_cap_mcmc_corner_titles(fig, summary, labels, fontsize=CAP_MCMC_TITLE_SIZE):
    """Set the diagonal titles to "<label> = median +err -err".

    ``corner`` can build these itself via show_titles=True, but it would
    re-derive the label text and use a bare math minus.  Building them here
    reuses ``labels`` verbatim from the axis-label list and the already
    computed posterior percentiles, so the label and the minus sign are
    identical to the ones on the side axes.

    summary : the dict from _posterior_percentile_summary, with its values
        already multiplied by CAP_MCMC_DISPLAY_SCALE.
    """
    ndim = len(labels)
    axes = np.asarray(fig.axes, dtype=object)
    if axes.size != ndim * ndim:
        return
    axes = axes.reshape((ndim, ndim))
    fmt = lambda v: _fixed_decimal_body(v, CAP_MCMC_TITLE_DECIMALS, strip_zeros=False)
    minus = _visible_tex_minus()
    for i in range(ndim):
        median = fmt(summary['median'][i])
        plus = fmt(summary['err_plus'][i])
        low = fmt(summary['err_minus'][i])
        axes[i, i].set_title(
            f'{labels[i]} $= {median}^{{+{plus}}}_{{{minus}{low}}}$',
            fontsize=fontsize,
        )

def _set_cap_mcmc_corner_titles_overlay(fig, major_summary, minor_summary, names, degree, fontsize=CAP_MCMC_TITLE_SIZE):
    """Set diagonal titles for an overlaid major/minor corner plot.

    Each diagonal title gets up to two lines:
      1. major-sector posterior summary
      2. minor-sector posterior summary

    The summary values should already be multiplied by CAP_MCMC_DISPLAY_SCALE.
    """
    ndim = len(names)
    axes = np.asarray(fig.axes, dtype=object)
    if axes.size != ndim * ndim:
        return
    axes = axes.reshape((ndim, ndim))

    fmt = lambda v: _fixed_decimal_body(v, CAP_MCMC_TITLE_DECIMALS, strip_zeros=False)
    minus = _visible_tex_minus()

    for i, name in enumerate(names):
        lines = []

        if major_summary is not None:
            label_maj = _cap_mcmc_parameter_label(name, 'maj', degree)
            median = fmt(major_summary['median'][i])
            plus = fmt(major_summary['err_plus'][i])
            low = fmt(major_summary['err_minus'][i])
            lines.append(
                f'{label_maj} $= {median}^{{+{plus}}}_{{{minus}{low}}}$'
            )

        if minor_summary is not None:
            label_min = _cap_mcmc_parameter_label(name, 'min', degree)
            median = fmt(minor_summary['median'][i])
            plus = fmt(minor_summary['err_plus'][i])
            low = fmt(minor_summary['err_minus'][i])
            lines.append(
                f'{label_min} $= {median}^{{+{plus}}}_{{{minus}{low}}}$'
            )

        axes[i, i].set_title('\n'.join(lines), fontsize=fontsize)

def _format_cap_mcmc_corner_axes(fig, ndim):
    """Apply the no-e-notation, visible-minus tick formatter to a corner plot.

    ``corner`` already does the axis work that used to be reimplemented here:
    it gives each parameter one common numerical range wherever that parameter
    appears (horizontally in its column and vertically in its row), and it
    installs a NullFormatter on the interior panels so that only the bottom
    row shows x tick labels and only the left column shows y tick labels.

    So the only thing left to do is swap corner's default ScalarFormatter for
    ours on exactly the panels that actually display labels -- the bottom row
    and the left column.  Touching the interior panels would overwrite their
    NullFormatter and make the hidden labels reappear.
    """
    axes = np.asarray(fig.axes, dtype=object)
    if axes.size != ndim * ndim:
        return
    axes = axes.reshape((ndim, ndim))
    formatter = FuncFormatter(_tex_cap_mcmc_tick)
    for col in range(ndim):
        ax = axes[ndim - 1, col]
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.get_offset_text().set_visible(False)
    for row in range(1, ndim):
        ax = axes[row, 0]
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_visible(False)

def _apply_scientific_y_ticks(ax, nbins=CAP_Y_TICK_NBINS):
    """Apply scaled CAP y-axis tick labels.

    Function name is kept unchanged so existing plotting calls still work.
    """
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_formatter(FuncFormatter(_tex_scaled_tick))
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
    return (ylo - pad, yhi + pad)

def _format_colorbar(cb, label):
    cb.locator = MaxNLocator(nbins=STACK_COLORBAR_TICK_NBINS)
    cb.formatter = FuncFormatter(_tex_scaled_tick)
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.set_label(label, fontsize=STACK_COLORBAR_LABEL_SIZE)
    cb.ax.tick_params(labelsize=STACK_COLORBAR_TICK_SIZE)
    cb.ax.set_title(STACK_COLORBAR_EXPONENT_LABEL, fontsize=STACK_COLORBAR_EXPONENT_SIZE, pad=STACK_COLORBAR_EXPONENT_PAD)

def _set_touching_square_grid(fig, axes, left, bottom, top):
    """Position image axes as a touching grid with square panels.

    This prevents imshow panels from leaving horizontal gutters when the
    figure is wider than the square image grid requires.  Returns the
    right edge of the grid in figure coordinates, useful for placing a
    close colorbar.
    """
    axes = np.asarray(axes)
    if axes.ndim != 2:
        raise ValueError('axes must be a 2D array')
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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

def _prune_touching_xy_ticks(ax):
    """Set fixed x/y ticks for stacked image panels, with visible minus signs."""
    ax.set_xticks(STACK_XY_TICKS)
    ax.set_yticks(STACK_XY_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(_tex_unscaled_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_tex_unscaled_tick))

def _set_hist_ylabel(ax, label):
    """Set histogram y-label with centralized label coordinates."""
    ax.set_ylabel(label, fontsize=HIST_AXIS_LABEL_SIZE)
    ax.yaxis.set_label_coords(HIST_YLABEL_X, HIST_YLABEL_Y)

def _set_cap_ylabel(ax, label, fontsize):
    """Set CAP-profile y-label with centralized label coordinates."""
    ax.set_ylabel(label, fontsize=fontsize)
    ax.yaxis.set_label_coords(CAP_YLABEL_X, CAP_YLABEL_Y)


def _legend_label_with_reduced_chi2(base_label, mcmc):
    """Legend label optionally augmented with reduced chi^2."""
    if not isinstance(mcmc, dict):
        return base_label
    chi2 = mcmc.get('chi2_gof')
    dof = mcmc.get('dof_gof')
    if np.isfinite(chi2) and np.isfinite(dof) and dof > 0:
        return rf'{base_label} ($\chi^2_\nu = {chi2 / dof:.3f}$)'
    return base_label
def _mass_bin_label(mass_lo, mass_hi):
    return MASS_BIN_LABEL_TEMPLATE.format(mass_lo=mass_lo, mass_hi=mass_hi)

def _n_stacked_label(n_success):
    return N_STACKED_LABEL_TEMPLATE.format(n_stacked=int(n_success))

def _collect_stack_values_from_results(all_bin_results):
    """Collect only oriented and radio stack pixels for a shared color scale."""
    chunks = []
    for bin_result in all_bin_results:
        for result_key in ('full_stack', 'radio_full_stack'):
            result = bin_result.get(result_key, {})
            if not isinstance(result, dict):
                continue
            for stack_key in ('stack_unori', 'stack_ori'):
                stack = result.get(stack_key)
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
    print(f'  [stack color norm] robust symmetric range: p1={lo:.3e}, p99={hi:.3e}, vmin={-vabs:.3e}, vmax={vabs:.3e}')
    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

def _imshow_stack_panel(ax, stack, pixscale, norm, n_success=None, axis_label_size=15, tick_label_size=12, n_label_size=10):
    if stack is None or pixscale is None or norm is None:
        ax.text(0.5, 0.5, STACK_NO_DATA_LABEL, transform=ax.transAxes, ha='center', va='center', color='0.35', fontsize=STACK_NO_DATA_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    ny, nx = stack.shape
    ext = [-0.5 * nx * pixscale, 0.5 * nx * pixscale, -0.5 * ny * pixscale, 0.5 * ny * pixscale]
    im = ax.imshow(stack, origin='lower', cmap='RdBu_r', norm=norm, extent=ext)
    ax.set_xlabel(STACK_X_LABEL, fontsize=axis_label_size)
    ax.set_ylabel(STACK_Y_LABEL, fontsize=axis_label_size)
    ax.tick_params(labelsize=tick_label_size)
    if n_success is not None:
        ax.text(STACK_N_LABEL_X, STACK_N_LABEL_Y, _n_stacked_label(n_success), transform=ax.transAxes, ha='left', va='top', fontsize=n_label_size, bbox=dict(facecolor='white', alpha=STACK_N_LABEL_BBOX_ALPHA, edgecolor='none', pad=STACK_N_LABEL_BBOX_PAD))
    return im

def plot_summary_full_mass_stacks(all_bin_results, out_dir, stack_norm):
    """Two rows by three columns: full mass bin unoriented and oriented."""
    if len(all_bin_results) == 0:
        return
    nrows = 2
    ncols = len(MASS_BINS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(STACK_FULL_FIG_WIDTH_PER_COL * ncols, STACK_FULL_FIG_HEIGHT_PER_ROW * nrows), squeeze=False, sharex=True, sharey=True)
    grid_left = STACK_FULL_GRID_LEFT
    grid_bottom = STACK_FULL_GRID_BOTTOM
    grid_top = STACK_FULL_GRID_TOP
    grid_right = _set_touching_square_grid(fig, axes, grid_left, grid_bottom, grid_top)
    row_defs = [('stack_unori', STACK_ROW_LABEL_UNORIENTED), ('stack_ori', STACK_ROW_LABEL_ORIENTED)]
    last_im = None
    for r, (stack_key, row_label) in enumerate(row_defs):
        for c, bin_result in enumerate(all_bin_results):
            ax = axes[r, c]
            mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
            mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
            res = bin_result.get('full_stack', {})
            stack = res.get(stack_key) if isinstance(res, dict) else None
            pixscale = res.get('pixscale') if isinstance(res, dict) else None
            im = _imshow_stack_panel(ax, stack, pixscale, stack_norm, res.get('n_success', 0), axis_label_size=STACK_FULL_AXIS_LABEL_SIZE, tick_label_size=STACK_FULL_TICK_LABEL_SIZE, n_label_size=STACK_FULL_N_LABEL_SIZE)
            if im is not None:
                last_im = im
            _prune_touching_xy_ticks(ax)
            if c > 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
            if r < nrows - 1:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
            if r == 0:
                ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=STACK_FULL_PANEL_TITLE_SIZE, pad=STACK_FULL_PANEL_TITLE_PAD)
        axes[r, 0].text(STACK_FULL_ROW_LABEL_X, 0.5, row_label, transform=axes[r, 0].transAxes, ha='right', va='center', rotation=90, fontsize=STACK_FULL_ROW_LABEL_SIZE)
    if last_im is not None:
        cax = fig.add_axes([grid_right + STACK_COLORBAR_PAD, STACK_FULL_CBAR_BOTTOM, STACK_COLORBAR_WIDTH, STACK_FULL_CBAR_HEIGHT])
        cb = fig.colorbar(last_im, cax=cax)
        _format_colorbar(cb, STACK_COLORBAR_LABEL)
    fig.suptitle(STACK_FULL_SUPTITLE, fontsize=STACK_FULL_SUPTITLE_SIZE, y=STACK_FULL_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_oriented_full_stack_2x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary full stacks] {path}')

def _cov_to_correlation(cov):
    """Normalize a bootstrap covariance matrix into a correlation matrix.

    Guards zero/degenerate variance entries (returns NaN there instead of
    dividing by zero) and forces the diagonal to exactly 1.0.
    """
    cov = np.asarray(cov, dtype=np.float64)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(std, std)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0.0, cov / denom, np.nan)
    np.fill_diagonal(corr, 1.0)
    return corr

def plot_summary_sector_cap_correlation(all_bin_results, out_dir):
    """2 rows (major top, minor bottom) x N mass-bin columns of CAP correlation matrices.

    Fully scalable: the tick count/positions and matrix size follow
    CAP_RADII_ARCMIN, and the column count follows MASS_BINS, so changing
    CAP_N_AP, the aperture range, or the number of mass bins requires no
    edits here.
    """
    if len(all_bin_results) == 0:
        return
    n_cols = len(MASS_BINS)
    n_ap = len(CAP_RADII_ARCMIN)
    fig, axes = plt.subplots(2, n_cols, figsize=(CAP_CORR_FIG_WIDTH_PER_COL * n_cols, CAP_CORR_FIG_HEIGHT_PER_ROW * 2), squeeze=False)
    tick_idx = np.unique(np.linspace(0, n_ap - 1, min(n_ap, CAP_CORR_N_TICKS)).round().astype(int))
    tick_labels = [f'{CAP_RADII_ARCMIN[i]:.1f}' for i in tick_idx]
    sector_rows = [('cap_major_cov', CAP_MAJOR_SECTOR_LABEL), ('cap_minor_cov', CAP_MINOR_SECTOR_LABEL)]
    im = None
    for row, (cov_key, sector_label) in enumerate(sector_rows):
        for c, bin_result in enumerate(all_bin_results):
            ax = axes[row][c]
            mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
            mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
            res = bin_result.get('full_stack', {})
            cov = res.get(cov_key)
            if cov is None or not np.all(np.isfinite(cov)):
                ax.text(0.5, 0.5, CAP_NO_SECTOR_DATA_LABEL, transform=ax.transAxes, ha='center', va='center', color='0.35')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            corr = _cov_to_correlation(cov)
            im = ax.imshow(corr, origin='lower', cmap=CAP_CORR_CMAP, vmin=CAP_CORR_VMIN, vmax=CAP_CORR_VMAX, aspect='equal')
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels(tick_labels, fontsize=CAP_CORR_TICK_LABEL_SIZE)
            ax.set_yticklabels(tick_labels, fontsize=CAP_CORR_TICK_LABEL_SIZE)
            ax.set_xlabel(CAP_X_LABEL, fontsize=CAP_CORR_AXIS_LABEL_SIZE)
            ax.set_ylabel(CAP_X_LABEL, fontsize=CAP_CORR_AXIS_LABEL_SIZE)
            title = f'{sector_label}\n{_mass_bin_label(mass_lo, mass_hi)}'
            ax.set_title(title, fontsize=CAP_CORR_PANEL_TITLE_SIZE, pad=CAP_CORR_PANEL_TITLE_PAD)
    if im is not None:
        fig.subplots_adjust(left=0.07, right=0.9, bottom=0.06, top=0.88, wspace=CAP_CORR_WSPACE, hspace=0.4)
        cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('${\\rm Correlation}$', fontsize=CAP_CORR_CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=CAP_CORR_TICK_LABEL_SIZE)
    fig.suptitle(CAP_CORR_SUPTITLE, fontsize=CAP_SUPTITLE_SIZE, y=CAP_CORR_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_oriented_sector_cap_correlation_2x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary sector CAP correlation] {path}')

def plot_summary_sector_cap_profiles(all_bin_results, out_dir):
    """One row by three columns: major versus minor sector CAP in each mass bin."""
    if len(all_bin_results) == 0:
        return
    fig, axes = plt.subplots(1, len(MASS_BINS), figsize=(CAP_FIG_WIDTH_PER_COL * len(MASS_BINS), CAP_FIG_HEIGHT), squeeze=False, sharey=True)
    axes = axes.ravel()
    fig.subplots_adjust(left=CAP_LEFT, right=CAP_RIGHT, bottom=CAP_BOTTOM, top=CAP_SECTOR_TOP, wspace=CAP_WSPACE)
    ylim_pairs = []
    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
        res = bin_result.get('full_stack', {})
        maj_m = _cap_to_plot_units(res.get('cap_major_mean'))
        maj_s = _cap_to_plot_units(res.get('cap_major_std'))
        min_m = _cap_to_plot_units(res.get('cap_minor_mean'))
        min_s = _cap_to_plot_units(res.get('cap_minor_std'))
        if maj_m is None or maj_s is None or min_m is None or (min_s is None):
            ax.text(0.5, 0.5, CAP_NO_SECTOR_DATA_LABEL, transform=ax.transAxes, ha='center', va='center', color='0.35')
        else:
            ylim_pairs.extend([(maj_m, maj_s), (min_m, min_s)])
            ax.errorbar(CAP_RADII_ARCMIN - 0.04, maj_m, yerr=maj_s, fmt='P', linestyle='none', capsize=CAP_ERROR_CAPSIZE, lw=CAP_ERROR_LW, ms=CAP_ERROR_MARKER_SIZE, label=CAP_MAJOR_SECTOR_LABEL)
            ax.errorbar(CAP_RADII_ARCMIN + 0.04, min_m, yerr=min_s, fmt='X', linestyle='none', capsize=CAP_ERROR_CAPSIZE, lw=CAP_ERROR_LW, ms=CAP_ERROR_MARKER_SIZE, label=CAP_MINOR_SECTOR_LABEL)
        ax.axhline(0, color=CAP_ZERO_LINE_COLOR, lw=CAP_ZERO_LINE_WIDTH, ls=CAP_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=CAP_PANEL_TITLE_SIZE, pad=CAP_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=CAP_TICK_LABEL_SIZE)
        ax.set_xlabel(CAP_X_LABEL, fontsize=CAP_AXIS_LABEL_SIZE_SECTOR)
        if c == 0:
            _set_cap_ylabel(ax, _cap_plot_axis_label(), CAP_AXIS_LABEL_SIZE_SECTOR)
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
    fig.suptitle(CAP_SECTOR_SUPTITLE, fontsize=CAP_SUPTITLE_SIZE, y=CAP_SECTOR_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_oriented_full_stack_cap_profiles_1x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary sector CAP] {path}')

def plot_summary_sector_posterior_curves(all_bin_results, out_dir):
    """One row by three columns: major/minor CAP data with the ML polynomial
    curve and random posterior draws overlaid (emcee line-tutorial style).

    The spread of the posterior curves around the ML curve is the visual
    statement of the parameter uncertainties and correlations.
    """
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
    fig.subplots_adjust(left=CAP_LEFT, right=CAP_RIGHT, bottom=CAP_BOTTOM, top=CAP_SECTOR_TOP, wspace=CAP_WSPACE)
    theta = np.asarray(CAP_RADII_ARCMIN, dtype=np.float64)
    theta_line = np.linspace(theta.min(), theta.max(), 100)
    rng = np.random.default_rng(SEED)
    ylim_pairs = []
    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
        res = bin_result.get('full_stack', {})
        maj_m = _cap_to_plot_units(res.get('cap_major_mean'))
        maj_s = _cap_to_plot_units(res.get('cap_major_std'))
        min_m = _cap_to_plot_units(res.get('cap_minor_mean'))
        min_s = _cap_to_plot_units(res.get('cap_minor_std'))
        maj_fit = res.get('mcmc_major')
        min_fit = res.get('mcmc_minor')
        if maj_m is None or maj_s is None or min_m is None or (min_s is None):
            ax.text(0.5, 0.5, CAP_NO_SECTOR_DATA_LABEL, transform=ax.transAxes, ha='center', va='center', color='0.35')
        else:
            ylim_pairs.extend([(maj_m, maj_s), (min_m, min_s)])
            maj_label = _legend_label_with_reduced_chi2(CAP_MAJOR_SECTOR_LABEL, maj_fit)
            min_label = _legend_label_with_reduced_chi2(CAP_MINOR_SECTOR_LABEL, min_fit)
            maj_container = ax.errorbar(
                theta - 0.04,
                maj_m,
                yerr=maj_s,
                fmt='P',
                linestyle='none',
                capsize=CAP_ERROR_CAPSIZE,
                lw=CAP_ERROR_LW,
                ms=CAP_ERROR_MARKER_SIZE,
                label=maj_label,
            )
            min_container = ax.errorbar(
                theta + 0.04,
                min_m,
                yerr=min_s,
                fmt='X',
                linestyle='none',
                capsize=CAP_ERROR_CAPSIZE,
                lw=CAP_ERROR_LW,
                ms=CAP_ERROR_MARKER_SIZE,
                label=min_label,
            )
            maj_color = maj_container.lines[0].get_color()
            min_color = min_container.lines[0].get_color()
            for mcmc, color in [(maj_fit, maj_color), (min_fit, min_color)]:
                if not isinstance(mcmc, dict) or mcmc.get('samples') is None:
                    continue
                X_line = _design_matrix_polynomial(theta_line, mcmc['degree'])
                samples = mcmc['samples']
                n_draw = min(CAP_POSTERIOR_N_CURVES, len(samples))
                for i in rng.integers(0, len(samples), size=n_draw):
                    ax.plot(theta_line, X_line @ samples[i], color=color, lw=CAP_POSTERIOR_CURVE_LW, alpha=CAP_POSTERIOR_CURVE_ALPHA)
                if np.all(np.isfinite(mcmc['beta_ml'])):
                    ax.plot(theta_line, X_line @ mcmc['beta_ml'], color=color, lw=CAP_FIT_LINE_LW, ls='-')
        ax.axhline(0, color=CAP_ZERO_LINE_COLOR, lw=CAP_ZERO_LINE_WIDTH, ls=CAP_ZERO_LINE_STYLE)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=CAP_PANEL_TITLE_SIZE, pad=CAP_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=CAP_TICK_LABEL_SIZE)
        ax.set_xlabel(CAP_X_LABEL, fontsize=CAP_AXIS_LABEL_SIZE_SECTOR)
        if c == 0:
            _set_cap_ylabel(ax, _cap_plot_axis_label(), CAP_AXIS_LABEL_SIZE_SECTOR)
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
    fig.suptitle(CAP_FIT_SECTOR_SUPTITLE, fontsize=CAP_SUPTITLE_SIZE, y=CAP_SECTOR_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_oriented_sector_posterior_curves_1x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary sector posterior curves] {path}')

def plot_summary_radio_full_stacks(all_bin_results, out_dir, stack_norm):
    """One row by three columns: radio-only full stacks by mass bin."""
    if len(all_bin_results) == 0:
        return
    nrows = 1
    ncols = len(MASS_BINS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(STACK_RADIO_FIG_WIDTH_PER_COL * ncols, STACK_RADIO_FIG_HEIGHT), squeeze=False, sharex=True, sharey=True)
    grid_left = STACK_RADIO_GRID_LEFT
    grid_bottom = STACK_RADIO_GRID_BOTTOM
    grid_top = STACK_RADIO_GRID_TOP
    grid_right = _set_touching_square_grid(fig, axes, grid_left, grid_bottom, grid_top)
    axes = axes.ravel()
    last_im = None
    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
        res = bin_result.get('radio_full_stack', {})
        stack = res.get(RADIO_STACK_KEY) if isinstance(res, dict) else None
        pixscale = res.get('pixscale') if isinstance(res, dict) else None
        im = _imshow_stack_panel(ax, stack, pixscale, stack_norm, res.get('n_success', 0), axis_label_size=STACK_RADIO_AXIS_LABEL_SIZE, tick_label_size=STACK_RADIO_TICK_LABEL_SIZE, n_label_size=STACK_RADIO_N_LABEL_SIZE)
        if im is not None:
            last_im = im
        _prune_touching_xy_ticks(ax)
        if c > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=STACK_RADIO_PANEL_TITLE_SIZE, pad=STACK_RADIO_PANEL_TITLE_PAD)
    if last_im is not None:
        cax = fig.add_axes([grid_right + STACK_COLORBAR_PAD, STACK_RADIO_CBAR_BOTTOM, STACK_COLORBAR_WIDTH, STACK_RADIO_CBAR_HEIGHT])
        cb = fig.colorbar(last_im, cax=cax)
        _format_colorbar(cb, STACK_COLORBAR_LABEL)
    fig.suptitle(STACK_RADIO_SUPTITLE, fontsize=STACK_RADIO_SUPTITLE_SIZE, y=STACK_RADIO_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_radio_full_stack_1x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary radio full stacks] {path}')

def _cap_shape_fit_model_description(degree=CAP_SHAPE_FIT_DEGREE):
    """Human-readable description of the active polynomial fit."""
    if degree == 1:
        return 'linear fit, CAP ~ m*theta + c'
    if degree == 2:
        return 'quadratic fit, CAP ~ a*theta^2 + b*theta + c'
    return f'degree-{degree} polynomial fit'

def _design_matrix_polynomial(theta, degree):
    """Design matrix for a polynomial in theta.

    np.vander(theta, degree + 1) already returns the columns in exactly the
    order this pipeline wants -- [theta^degree, ..., theta, 1] -- with the
    constant/offset term last.
    """
    return np.vander(np.asarray(theta, dtype=np.float64), degree + 1)

def _cap_coefficient_names(degree=CAP_SHAPE_FIT_DEGREE):
    """Plain-text coefficient names in design-matrix order."""
    if degree == 1:
        return ['m', 'c']
    if degree == 2:
        return ['a', 'b', 'c']
    return [*[f'theta^{power}' if power > 1 else 'theta' for power in range(degree, 0, -1)], 'c']

def _posterior_percentile_summary(samples):
    """Return median and asymmetric 68% uncertainties for each column."""
    samples = np.asarray(samples, dtype=np.float64)
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    return {'q16': q16, 'median': q50, 'q84': q84, 'err_minus': q50 - q16, 'err_plus': q84 - q50}

def run_quadratic_mcmc(cap_mean, cap_cov, theta=None, degree=CAP_SHAPE_FIT_DEGREE, seed=CAP_MCMC_SEED):
    """Fit one CAP profile with the emcee line-fitting tutorial recipe
    (https://emcee.readthedocs.io/en/stable/tutorials/line/).

    1. Maximum-likelihood fit: the polynomial model is linear in its
       coefficients and the likelihood is Gaussian, so the ML solution is
       the closed-form weighted-least-squares solution -- no optimizer.
    2. Walkers start in a tiny Gaussian ball around the ML solution.
    3. Flat prior: the log-probability is just the Gaussian log-likelihood
       built from the full bootstrap covariance of the CAP profile.
    4. Uncertainties are 16/50/84 posterior percentiles and the parameter
       correlation matrix comes directly from the flattened samples.

    Goodness of fit: chi2_gof = r^T C^-1 r at the ML solution, with
    dof_gof = n_apertures - n_parameters and pte_gof the probability of a
    larger chi2 by chance. chi2/dof near 1 (PTE not extreme in either
    direction) means the quadratic describes the profile adequately;
    chi2/dof >> 1 (tiny PTE) means it does not, and the parameter
    uncertainties below should not be trusted.
    """
    n_params = degree + 1
    nan_vec = np.full(n_params, np.nan)
    out = {'degree': degree, 'parameter_names': _cap_coefficient_names(degree), 'beta_ml': nan_vec.copy(), 'samples': None, 'summary': {k: nan_vec.copy() for k in ('q16', 'median', 'q84', 'err_minus', 'err_plus')}, 'param_corr': np.full((n_params, n_params), np.nan), 'acceptance_fraction': np.nan, 'autocorr_time': nan_vec.copy(), 'n_posterior_samples': 0, 'chi2_gof': np.nan, 'dof_gof': 0, 'pte_gof': np.nan}
    if not RUN_CAP_MCMC:
        return out
    if emcee is None or corner is None:
        raise ImportError("RUN_CAP_MCMC=True requires the 'emcee' and 'corner' packages. Install them with: python -m pip install emcee corner")
    if theta is None:
        theta = CAP_RADII_ARCMIN
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(cap_mean, dtype=np.float64)
    cov = np.asarray(cap_cov, dtype=np.float64)
    if y.ndim != 1 or theta.size != y.size or cov.shape != (y.size, y.size) or (not np.all(np.isfinite(y))) or (not np.all(np.isfinite(cov))):
        return out
    burn = int(CAP_MCMC_BURN_IN)
    thin = int(CAP_MCMC_THIN)
    if burn < 0 or burn >= CAP_MCMC_N_STEPS:
        raise ValueError('CAP_MCMC_BURN_IN must satisfy 0 <= burn-in < n_steps.')
    if thin <= 0:
        raise ValueError('CAP_MCMC_THIN must be positive.')
    X = _design_matrix_polynomial(theta, degree)
    cov_inv = np.linalg.inv(cov)
    # Closed-form ML/weighted-least-squares solution and its parameter
    # covariance (the inverse Fisher matrix), both from a single inverse.
    beta_cov = np.linalg.inv(X.T @ cov_inv @ X)
    beta_ml = beta_cov @ (X.T @ cov_inv @ y)
    out['beta_ml'] = beta_ml
    resid_ml = y - X @ beta_ml
    out['chi2_gof'] = float(resid_ml @ cov_inv @ resid_ml)
    out['dof_gof'] = int(y.size - n_params)
    if out['dof_gof'] > 0:
        out['pte_gof'] = float(scipy_stats.chi2.sf(out['chi2_gof'], df=out['dof_gof']))
    beta_sigma = np.sqrt(np.clip(np.diag(beta_cov), 0.0, None))
    beta_sigma = np.where(beta_sigma > 0.0, beta_sigma, np.maximum(np.abs(beta_ml), np.finfo(np.float64).eps))

    def log_probability(beta):
        resid = y - X @ beta
        return -0.5 * float(resid @ cov_inv @ resid)

    rng = np.random.default_rng(seed)
    nwalkers = max(int(CAP_MCMC_N_WALKERS), 2 * n_params + 2)
    if nwalkers % 2 != 0:
        nwalkers += 1
    pos = beta_ml + CAP_MCMC_INIT_BALL_SCALE * beta_sigma * rng.standard_normal((nwalkers, n_params))
    sampler = emcee.EnsembleSampler(nwalkers, n_params, log_probability)
    sampler.run_mcmc(pos, int(CAP_MCMC_N_STEPS), progress=CAP_MCMC_PROGRESS)
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    out['samples'] = flat
    out['summary'] = _posterior_percentile_summary(flat)
    out['param_corr'] = np.corrcoef(flat, rowvar=False)
    out['acceptance_fraction'] = float(np.mean(sampler.acceptance_fraction))
    out['n_posterior_samples'] = int(flat.shape[0])
    try:
        out['autocorr_time'] = np.asarray(sampler.get_autocorr_time(tol=0), dtype=np.float64)
    except Exception as exc:
        print(f'  [MCMC] autocorrelation-time estimate unavailable: {exc}')
    return out

def _format_posterior_value(median, err_minus, err_plus):
    """Compact median with asymmetric 68% credible interval."""
    if not (np.isfinite(median) and np.isfinite(err_minus) and np.isfinite(err_plus)):
        return 'nan'
    return f'{median:.6e} -{err_minus:.2e} +{err_plus:.2e}'

def print_cap_mcmc_fit_tables(all_bin_results):
    """Print ML values, posterior medians with 16th/84th-percentile
    uncertainties, and the posterior parameter correlation matrix for each
    mass bin and sector."""
    if not RUN_CAP_MCMC:
        return
    for i_bin, bin_result in enumerate(all_bin_results):
        mass_lo = bin_result.get('mass_lo', MASS_BINS[i_bin][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[i_bin][1])
        full = bin_result.get('full_stack', {})
        if not isinstance(full, dict):
            continue
        for key, sector_word in [('mcmc_major', 'major'), ('mcmc_minor', 'minor')]:
            mcmc = full.get(key)
            if not isinstance(mcmc, dict) or mcmc.get('samples') is None:
                continue
            print('\n' + '=' * 88)
            print(f"CAP MCMC posterior ({sector_word} sector): logM ({mass_lo:.1f}, {mass_hi:.1f}] [{_cap_shape_fit_model_description(mcmc['degree'])}]")
            if np.isfinite(mcmc['chi2_gof']) and mcmc['dof_gof'] > 0:
                print(f"Goodness of fit at ML: chi2 = {mcmc['chi2_gof']:.3f}, dof = {mcmc['dof_gof']}, chi2/dof = {mcmc['chi2_gof'] / mcmc['dof_gof']:.3f}, PTE = {mcmc['pte_gof']:.3g}")
            print(f"Mean acceptance fraction: {mcmc['acceptance_fraction']:.3f}")
            tau = np.asarray(mcmc['autocorr_time'], dtype=np.float64)
            if np.all(np.isfinite(tau)):
                print('Autocorrelation times: ' + ', '.join((f'{x:.1f}' for x in tau)))
            print(f"Flattened posterior samples: {mcmc['n_posterior_samples']:,}")
            names = mcmc['parameter_names']
            s = mcmc['summary']
            print(f"{'parameter':<12}{'ML':>16}{'posterior median -16/+84':>34}")
            print('-' * 62)
            for j, name in enumerate(names):
                post_txt = _format_posterior_value(s['median'][j], s['err_minus'][j], s['err_plus'][j])
                print(f"{name:<12}{mcmc['beta_ml'][j]:>16.6e}{post_txt:>34}")
            corr = np.asarray(mcmc['param_corr'], dtype=np.float64)
            print('Posterior parameter correlation matrix:')
            print(' ' * 10 + ''.join((f'{n:>10}' for n in names)))
            for j, name in enumerate(names):
                print(f'{name:<10}' + ''.join((f'{corr[j, k]:>10.3f}' for k in range(len(names)))))
            print('=' * 88)

def plot_cap_mcmc_corner_pdf(all_bin_results, out_dir):
    """Write ALL mass-bin overlaid major/minor corner plots onto ONE page.

    The per-bin overlaid corners are laid out as a grid of matplotlib
    subfigures (two columns by default). Each panel is forced square so the
    posterior contours show correlation rather than the cell aspect ratio.
    Any leftover cell (e.g. the 4th slot in a 3-bin, 2x2 grid) holds the
    shared legend at CAP_MCMC_A4_LEGEND_CELL. The page is not literal A4:
    width is fixed but height is generous scratch space, cropped tightly by
    bbox_inches='tight' at save time. Returns a one-element list with the
    combined PDF path.
    """
    if not RUN_CAP_MCMC:
        return []
    if corner is None:
        raise ImportError('corner is required when RUN_CAP_MCMC=True')

    major_color = 'C0'
    minor_color = 'C1'

    plottable = []
    for i_bin, bin_result in enumerate(all_bin_results):
        full = bin_result.get('full_stack', {})
        if not isinstance(full, dict):
            continue
        major = full.get('mcmc_major')
        minor = full.get('mcmc_minor')
        if not isinstance(major, dict) or major.get('samples') is None:
            major = None
        if not isinstance(minor, dict) or minor.get('samples') is None:
            minor = None
        if major is None and minor is None:
            continue
        mass_lo = bin_result.get('mass_lo', MASS_BINS[i_bin][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[i_bin][1])
        plottable.append((mass_lo, mass_hi, major, minor))

    if not plottable:
        return []

    n_bins = len(plottable)
    ncols = 1 if n_bins == 1 else CAP_MCMC_A4_NCOLS
    nrows = -(-n_bins // ncols)  # ceil division

    n_cells = nrows * ncols
    legend_idx = CAP_MCMC_A4_LEGEND_CELL if n_cells > n_bins else None
    bin_cell_idx = [i for i in range(n_cells) if i != legend_idx][:n_bins]

    fig = plt.figure(figsize=(CAP_MCMC_A4_WIDTH_IN, CAP_MCMC_A4_HEIGHT_IN))
    grid = fig.subfigures(nrows, ncols, wspace=CAP_MCMC_A4_WSPACE, hspace=CAP_MCMC_A4_HSPACE)
    cells = np.atleast_1d(grid).ravel()

    for cell_i, (mass_lo, mass_hi, major, minor) in zip(bin_cell_idx, plottable):
        cell = cells[cell_i]
        ref = major if major is not None else minor
        names = ref['parameter_names']
        degree = int(ref['degree'])
        ndim = len(names)
        labels = [_cap_mcmc_parameter_label(name, 'fit', degree) for name in names]

        cell.subplots(
            ndim, ndim,
            gridspec_kw=dict(wspace=CAP_MCMC_A4_CORNER_WHSPACE,
                             hspace=CAP_MCMC_A4_CORNER_WHSPACE),
        )

        for mcmc, color in [(major, major_color), (minor, minor_color)]:
            if mcmc is None:
                continue
            scaled_samples = np.asarray(mcmc['samples'], dtype=np.float64) * CAP_MCMC_DISPLAY_SCALE
            scaled_truths = np.asarray(mcmc['beta_ml'], dtype=np.float64) * CAP_MCMC_DISPLAY_SCALE
            corner.corner(
                scaled_samples,
                fig=cell,
                labels=labels,
                truths=scaled_truths,
                truth_color=color,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=False,
                bins=35,
                smooth=1.0,
                smooth1d=1.0,
                plot_datapoints=False,
                fill_contours=False,
                plot_density=False,
                color=color,
                hist_kwargs={'linewidth': 1.8},
                levels=(0.393, 0.864),
            )

        _format_cap_mcmc_corner_axes(cell, ndim)

        major_summary_scaled = None
        if major is not None:
            major_summary_scaled = {
                k: np.asarray(v, dtype=np.float64) * CAP_MCMC_DISPLAY_SCALE
                for k, v in major['summary'].items()
            }
        minor_summary_scaled = None
        if minor is not None:
            minor_summary_scaled = {
                k: np.asarray(v, dtype=np.float64) * CAP_MCMC_DISPLAY_SCALE
                for k, v in minor['summary'].items()
            }

        _set_cap_mcmc_corner_titles_overlay(
            cell,
            major_summary_scaled,
            minor_summary_scaled,
            names,
            degree,
            fontsize=CAP_MCMC_A4_TITLE_SIZE,
        )

        for ax in cell.axes:
            ax.set_box_aspect(1)

        cell.text(
            CAP_MCMC_A4_MASSLABEL_X,
            CAP_MCMC_A4_MASSLABEL_Y,
            '$\\log_{10}\\!\\left(M_\\ast/M_\\odot\\right)$\n'
            + f'$\\in({mass_lo:.1f},{mass_hi:.1f}]$',
            ha='center',
            va='center',
            fontsize=CAP_MCMC_A4_MASSLABEL_SIZE,
        )

    legend_handles = [
        Line2D([0], [0], color=major_color, lw=2.0, label=CAP_MAJOR_SECTOR_LABEL),
        Line2D([0], [0], color=minor_color, lw=2.0, label=CAP_MINOR_SECTOR_LABEL),
    ]
    if legend_idx is not None:
        cells[legend_idx].legend(
            handles=legend_handles,
            loc='center',
            fontsize=CAP_LEGEND_SIZE_SECTOR,
            frameon=True,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc='upper right',
            fontsize=CAP_LEGEND_SIZE_SECTOR,
            frameon=True,
        )
    fig.suptitle(CAP_MCMC_A4_SUPTITLE, fontsize=CAP_MCMC_SUPTITLE_SIZE, y=CAP_MCMC_A4_SUPTITLE_Y)

    path = os.path.join(out_dir, 'summary_oriented_sector_mcmc_corner_overlay_A4.pdf')
    fig.savefig(path, dpi=SAVEFIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  [summary MCMC corner overlay] {path}')
    return [path]
     
def build_selection_and_cache():
    """Build the oriented sample without any stellar-age dependency."""
    if RADIO_ONLY and EXCLUDE_RADIO:
        raise ValueError('RADIO_ONLY and EXCLUDE_RADIO cannot both be True.')
    if FIRST_PATH is None and (RADIO_ONLY or EXCLUDE_RADIO):
        raise ValueError('FIRST_PATH is None but a radio selection is enabled.')
    print('\nLoading tSZ map...')
    comptony = enmap.read_map(TSZ_MAP_PATH)
    print('\nLoading Firefly catalog...')
    ra_ff, dec_ff, logm_ff, z_ff, fits_idx_ff, is_galaxy_ff = load_firefly_full(FIREFLY_PATH)
    n_firefly = len(ra_ff)
    galaxy_class_mask = np.asarray(is_galaxy_ff, dtype=bool)
    n_firefly_galaxies = int(np.sum(galaxy_class_mask))
    if n_firefly == 0 or n_firefly_galaxies == 0:
        raise RuntimeError('Firefly contains no usable GALAXY rows.')
    finite_data_mask = np.isfinite(ra_ff) & np.isfinite(dec_ff) & np.isfinite(logm_ff) & np.isfinite(z_ff)
    mass_range_mask = np.isfinite(logm_ff) & (logm_ff > CACHE_LOG_MASS_MIN) & (logm_ff <= CACHE_LOG_MASS_MAX)
    redshift_mask = np.ones(n_firefly, dtype=bool)
    if USE_REDSHIFT_CUT:
        redshift_mask = np.isfinite(z_ff) & (z_ff >= Z_MIN) & (z_ff <= Z_MAX)
    cum_galaxy = galaxy_class_mask
    cum_mass = cum_galaxy & finite_data_mask & mass_range_mask
    cum_z = cum_mass & redshift_mask
    broad_mask = cum_z
    if PHOTO_PATH is None or not os.path.exists(PHOTO_PATH):
        raise FileNotFoundError(f'PHOTO_PATH was not found: {PHOTO_PATH}')
    pa_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    ab_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    pa_dev_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    pa_exp_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    ab_dev_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    ab_exp_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    fracdev_ff = np.full(n_firefly, np.nan, dtype=np.float64)
    valid_shape_ff = np.zeros(n_firefly, dtype=bool)
    photo_candidates = broad_mask.copy()
    pa_tmp, ab_tmp, valid_tmp, pa_dev_tmp, pa_exp_tmp, ab_dev_tmp, ab_exp_tmp, fracdev_tmp = load_photo_shapes(ra_ff[photo_candidates], dec_ff[photo_candidates], PHOTO_PATH, match_arcsec=PHOTO_MATCH_ARCSEC, fracdev_thresh=PHOTO_FRACDEV_THRESH, type_galaxy=PHOTO_TYPE_GALAXY, safety=SAFETY)
    pa_ff[photo_candidates] = pa_tmp
    ab_ff[photo_candidates] = ab_tmp
    pa_dev_ff[photo_candidates] = pa_dev_tmp
    pa_exp_ff[photo_candidates] = pa_exp_tmp
    ab_dev_ff[photo_candidates] = ab_dev_tmp
    ab_exp_ff[photo_candidates] = ab_exp_tmp
    fracdev_ff[photo_candidates] = fracdev_tmp
    valid_shape_ff[photo_candidates] = valid_tmp
    finite_shape_mask = np.isfinite(pa_ff) & np.isfinite(ab_ff) & np.isfinite(pa_dev_ff) & np.isfinite(pa_exp_ff) & np.isfinite(ab_dev_ff) & np.isfinite(ab_exp_ff) & np.isfinite(fracdev_ff)
    shape_mask = valid_shape_ff & finite_shape_mask & (ab_ff < BA_MAX)
    cum_photo = broad_mask & shape_mask
    print(f'  valid photo shape and 0 < selected b/a < {BA_MAX:.3f}: {cum_photo.sum():,}')
    print('\nChecking map coverage...')
    inside_map = np.zeros(n_firefly, dtype=bool)
    inside_map[cum_photo] = full_stamp_inside_map(ra_ff[cum_photo], dec_ff[cum_photo], comptony, STAMP_SOURCE_RADIUS_ARCMIN)
    cum_map = cum_photo & inside_map
    print(f'  inside map: {cum_map.sum():,} / {cum_photo.sum():,}')
    has_radio_ff = np.zeros(n_firefly, dtype=bool)
    if FIRST_PATH is not None:
        has_radio_ff[cum_map] = crossmatch_to_first(ra_ff[cum_map], dec_ff[cum_map], FIRST_PATH, FIRST_MATCH_ARCSEC, safety=SAFETY)
    if RADIO_ONLY:
        radio_mask = has_radio_ff
    elif EXCLUDE_RADIO:
        radio_mask = ~has_radio_ff
    else:
        radio_mask = np.ones(n_firefly, dtype=bool)
    cache_sel = cum_map
    base_sel = cache_sel & radio_mask
    radio_sel = cache_sel & has_radio_ff
    stages = [(f'1. Firefly {FIREFLY_CLASS_FIELD}={FIREFLY_GALAXY_CLASS}', cum_galaxy), (f'2. Mass cut ({CACHE_LOG_MASS_MIN:.1f}, {CACHE_LOG_MASS_MAX:.1f}]', cum_mass)]
    stage_number = 2
    if USE_REDSHIFT_CUT:
        stage_number += 1
        stages.append((f'{stage_number}. Redshift cut [{Z_MIN:.2f}, {Z_MAX:.2f}]', cum_z))
    stage_number += 1
    stages.append((f'{stage_number}. SDSS DR17 photo shape and b/a cut', cum_photo))
    stage_number += 1
    stages.append((f'{stage_number}. ACT-Planck map footprint', cum_map))
    if RADIO_ONLY:
        stage_number += 1
        stages.append((
            f'{stage_number}. Require FIRST radio match (<= {FIRST_MATCH_ARCSEC:.0f} arcsec)',
            base_sel,
        ))
    elif EXCLUDE_RADIO:
        stage_number += 1
        stages.append((
            f'{stage_number}. Exclude FIRST radio matches (<= {FIRST_MATCH_ARCSEC:.0f} arcsec)',
            base_sel,
        ))
    print_cumulative_cut_table(stages, logm_ff, final_label='FINAL oriented sample')
    print('\nFinal oriented samples (denominator: Firefly CLASS=GALAXY):')
    print(f'  main selected galaxies: {base_sel.sum():,} / {n_firefly_galaxies:,}')
    print(f'  radio galaxies: {radio_sel.sum():,} / {n_firefly_galaxies:,}')
    if base_sel.sum() == 0:
        raise RuntimeError('No galaxies survive the oriented selection.')
    r_final = STAMP_RADIUS_ARCMIN * np.pi / 180.0 / 60.0
    r_source = STAMP_SOURCE_RADIUS_ARCMIN * np.pi / 180.0 / 60.0
    test = test_source = None
    for j in np.where(cache_sel)[0][:50]:
        coords = np.deg2rad([dec_ff[j], ra_ff[j]])
        test = reproject.thumbnails(comptony, coords=coords, r=r_final)
        test_source = reproject.thumbnails(comptony, coords=coords, r=r_source)
        if test is not None and test_source is not None:
            break
    if test is None or test_source is None:
        raise RuntimeError('Could not extract test thumbnails.')
    ny, nx = np.asarray(test).shape
    ny_src, nx_src = np.asarray(test_source).shape
    pixscale = 2.0 * STAMP_RADIUS_ARCMIN / (ny - 1)
    print(f'  Final stamp dimensions: {ny} x {nx}')
    print(f'  Source stamp dimensions: {ny_src} x {nx_src}')
    print(f'  Final thumbnail pixel scale: {pixscale:.6f} arcmin per pixel')
    print('\nBuilding or opening oriented stamp cache...')
    h5f, existing_ids = _open_or_create_cache(CACHE_FILE, ny, nx, ny_src, nx_src, _extraction_config_hash())
    _write_cap_area_metadata(h5f, pixscale)
    cache_indices = np.where(cache_sel)[0]
    cache_fids = fits_idx_ff[cache_indices]
    new_mask = np.array([fid not in existing_ids for fid in cache_fids])
    if np.any(new_mask):
        new_local = cache_indices[new_mask]
        print(f'  [cache] Adding {len(new_local):,} new galaxies')
        _append_to_cache(h5f, len(new_local), fits_idx_ff[new_local], ra_ff[new_local], dec_ff[new_local], z_ff[new_local], logm_ff[new_local], pa_ff[new_local], ab_ff[new_local], pa_dev_ff[new_local], pa_exp_ff[new_local], ab_dev_ff[new_local], ab_exp_ff[new_local], fracdev_ff[new_local], has_radio_ff[new_local])
    else:
        print(f'  [cache] All {len(cache_fids):,} selected galaxies are already cached')
    base_fits_idx = fits_idx_ff[base_sel]
    radio_fits_idx = fits_idx_ff[radio_sel]
    cache_fids_now = h5f['fits_idx'][:]
    ff_lookup = {int(fid): i for i, fid in enumerate(fits_idx_ff)}
    n_updated = 0
    for ci, fid in enumerate(cache_fids_now):
        li = ff_lookup.get(int(fid))
        if li is None or not shape_mask[li]:
            continue
        h5f['pa'][ci] = pa_ff[li]
        h5f['ab'][ci] = ab_ff[li]
        h5f['pa_dev'][ci] = pa_dev_ff[li]
        h5f['pa_exp'][ci] = pa_exp_ff[li]
        h5f['ab_dev'][ci] = ab_dev_ff[li]
        h5f['ab_exp'][ci] = ab_exp_ff[li]
        h5f['fracdev'][ci] = fracdev_ff[li]
        h5f['has_radio'][ci] = has_radio_ff[li]
        n_updated += 1
    h5f.flush()
    print(f'  [cache] Refreshed shape/radio metadata for {n_updated:,} galaxies')
    extract_to_cache(comptony, h5f)
    cache_fids_now = h5f['fits_idx'][:]
    cache_logm_now = h5f['logm'][:]
    valid_now = h5f['stamp_valid'][:]
    main_rows = np.isin(cache_fids_now, base_fits_idx)
    radio_rows = np.isin(cache_fids_now, radio_fits_idx)
    print('\nCached stamp accounting after extraction:')
    print_mass_bin_accounting('oriented galaxies before stamp-valid cut', main_rows, cache_logm_now)
    print_mass_bin_accounting('oriented galaxies after stamp-valid cut', main_rows & valid_now, cache_logm_now, main_rows)
    print_mass_bin_accounting('radio galaxies before stamp-valid cut', radio_rows, cache_logm_now)
    print_mass_bin_accounting('radio galaxies after stamp-valid cut', radio_rows & valid_now, cache_logm_now, radio_rows)
    return (h5f, base_fits_idx, radio_fits_idx)

def add_full_stack_results(h5f, bin_result, mass_mask):
    """Compute full mass bin stack and CAP quantities."""
    mass_lo = bin_result.get('mass_lo', np.nan)
    mass_hi = bin_result.get('mass_hi', np.nan)
    result = stack_from_cache(h5f, mass_mask, label=f'main selected stack logM ({mass_lo:.1f}, {mass_hi:.1f}]')
    if result['n_success'] == 0:
        bin_result['full_stack'] = {'n_success': 0}
        return
    cap_m, cap_s, cap_cov, n_cap = mean_profile_and_covariance(result['cap_full_values'], seed=SEED)
    cap_maj_m, cap_maj_s, cap_maj_cov, _ = mean_profile_and_covariance(result['cap_major_values'], seed=SEED)
    cap_min_m, cap_min_s, cap_min_cov, _ = mean_profile_and_covariance(result['cap_minor_values'], seed=SEED)
    print(f'    full CAP profiles from {n_cap:,} galaxies')
    print_cap_profile_mean_sd_table(
        mass_lo,
        mass_hi,
        cap_maj_m,
        cap_maj_s,
        cap_min_m,
        cap_min_s,
    )
    bin_result['full_stack'] = {'n_success': result['n_success'], 'stack_unori': result['stack_unori'], 'stack_ori': result['stack_ori'], 'pixscale': result['pixscale'], 'cap_mean': cap_m, 'cap_std': cap_s, 'cap_cov': cap_cov, 'cap_major_mean': cap_maj_m, 'cap_major_std': cap_maj_s, 'cap_major_cov': cap_maj_cov, 'cap_minor_mean': cap_min_m, 'cap_minor_std': cap_min_s, 'cap_minor_cov': cap_min_cov, 'effective_mask': result['effective_mask']}
    if RUN_CAP_MCMC:
        full = bin_result['full_stack']
        seed_bin = CAP_MCMC_SEED + int(round(10.0 * mass_lo))
        print('    running major-sector CAP MCMC')
        full['mcmc_major'] = run_quadratic_mcmc(cap_maj_m, cap_maj_cov, seed=seed_bin)
        print('    running minor-sector CAP MCMC')
        full['mcmc_minor'] = run_quadratic_mcmc(cap_min_m, cap_min_cov, seed=seed_bin + 1)

def add_radio_stack_results(h5f, bin_result, mass_mask):
    """Compute radio-only full mass bin stack."""
    mass_lo = bin_result.get('mass_lo', np.nan)
    mass_hi = bin_result.get('mass_hi', np.nan)
    result = stack_from_cache(h5f, mass_mask, label=f'radio-only stack logM ({mass_lo:.1f}, {mass_hi:.1f}]')
    if result['n_success'] == 0:
        bin_result['radio_full_stack'] = {'n_success': 0}
        return
    bin_result['radio_full_stack'] = {'n_success': result['n_success'], 'stack_unori': result['stack_unori'], 'stack_ori': result['stack_ori'], 'pixscale': result['pixscale']}

def plot_summary_oriented_selected_ba_histograms(all_bin_results, h5f, out_dir):
    """Plot selected oriented-sample b/a distributions by mass bin."""
    if len(all_bin_results) == 0:
        return
    fig, axes = plt.subplots(1, len(MASS_BINS), figsize=(HIST_FIG_WIDTH_PER_COL * len(MASS_BINS), HIST_FIG_HEIGHT), squeeze=False, sharey=True)
    axes = axes.ravel()
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP, wspace=HIST_WSPACE)
    ab_all = h5f['ab'][:]
    bins = np.linspace(BA_HIST_MIN, BA_HIST_MAX, HIST_N_BINS + 1)
    for c, bin_result in enumerate(all_bin_results):
        ax = axes[c]
        mass_lo = bin_result.get('mass_lo', MASS_BINS[c][0])
        mass_hi = bin_result.get('mass_hi', MASS_BINS[c][1])
        res = bin_result.get('full_stack', {})
        mask = res.get('effective_mask') if isinstance(res, dict) else None
        if mask is not None:
            x = ab_all[mask]
            x = x[np.isfinite(x) & (x > BA_HIST_MIN) & (x <= BA_HIST_MAX)]
            if len(x) > 0:
                weights = np.ones_like(x, dtype=np.float64) / len(x)
                ax.hist(x, bins=bins, weights=weights, histtype='step', linewidth=HIST_LINEWIDTH)
                median_ba = float(np.median(x))
                ax.axvline(
                    median_ba,
                    linestyle=HIST_MEDIAN_LINESTYLE,
                    linewidth=HIST_MEDIAN_LINEWIDTH,
                    color='black',
                    label=rf'$\mathrm{{Median}}:\ b/a = {median_ba:.3f}$',
                )
                ax.legend(
                    loc='upper left',
                    fontsize=HIST_MEDIAN_LABEL_SIZE,
                    frameon=True,
                )
        ax.set_title(_mass_bin_label(mass_lo, mass_hi), fontsize=HIST_PANEL_TITLE_SIZE, pad=HIST_PANEL_TITLE_PAD)
        ax.tick_params(labelsize=HIST_TICK_LABEL_SIZE)
        ax.set_xlabel(HIST_BA_X_LABEL, fontsize=HIST_AXIS_LABEL_SIZE)
        if c == 0:
            _set_hist_ylabel(ax, HIST_FRACTION_Y_LABEL)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlim(BA_HIST_MIN, BA_HIST_MAX)
        ax.set_ylim(*HIST_YLIMS['ba_selected'])
        _prune_touching_x_ticks(ax)
    fig.suptitle(HIST_SELECTED_BA_SUPTITLE, fontsize=HIST_SUPTITLE_SIZE, y=HIST_SUPTITLE_Y)
    path = os.path.join(out_dir, 'summary_oriented_hist_ba_selected_1x3.pdf')
    _savefig(path)
    plt.close(fig)
    print(f'  [summary histogram: selected b/a] {path}')

def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    print('=' * 70)
    print('ORIENTED-STACK STANDALONE PIPELINE')
    print('  Stellar-age procedures: NOT PRESENT')
    print(f'  Summary path: {SUMMARY_DIR}')
    print(f'  Oriented cache: {CACHE_FILE}')
    print('  Dust cut: OFF')
    print('  photoPosPlate cross-match: ON')
    print(f'  Axis-ratio cut: 0 < b/a < {BA_MAX}')
    print(f"  CAP MCMC: {('ON' if RUN_CAP_MCMC else 'OFF')}")
    if RADIO_ONLY:
        print(f'  Radio selection: REQUIRE FIRST match within {FIRST_MATCH_ARCSEC:.0f} arcsec')
    elif EXCLUDE_RADIO:
        print(f'  Radio selection: EXCLUDE FIRST matches within {FIRST_MATCH_ARCSEC:.0f} arcsec')
    else:
        print('  Radio selection: no cut')
    print('=' * 70)
    h5f, base_fits_idx, radio_fits_idx = build_selection_and_cache()
    try:
        cache_stamp_valid = h5f['stamp_valid'][:]
        cache_logm = h5f['logm'][:]
        cache_fits_idx = h5f['fits_idx'][:]
        in_main = np.isin(cache_fits_idx, base_fits_idx)
        in_radio = np.isin(cache_fits_idx, radio_fits_idx)
        all_bin_results = []
        for mass_lo, mass_hi in MASS_BINS:
            print('\n' + '=' * 70)
            print(f'MASS BIN: log stellar mass ({mass_lo}, {mass_hi}]')
            print('=' * 70)
            in_mass = _mass_bin_mask(cache_logm, mass_lo, mass_hi)
            main_mask = in_main & in_mass
            radio_mask = in_radio & in_mass
            bin_result = {'mass_lo': mass_lo, 'mass_hi': mass_hi}
            if np.any(main_mask & cache_stamp_valid):
                add_full_stack_results(h5f, bin_result, main_mask)
            else:
                bin_result['full_stack'] = {'n_success': 0}
            if np.any(radio_mask & cache_stamp_valid):
                add_radio_stack_results(h5f, bin_result, radio_mask)
            else:
                bin_result['radio_full_stack'] = {'n_success': 0}
            all_bin_results.append(bin_result)
        print('\nWriting oriented-stack products...')
        stack_norm = _shared_stack_norm(_collect_stack_values_from_results(all_bin_results))
        plot_summary_full_mass_stacks(all_bin_results, SUMMARY_DIR, stack_norm)
        plot_summary_sector_cap_profiles(all_bin_results, SUMMARY_DIR)
        plot_summary_sector_cap_correlation(all_bin_results, SUMMARY_DIR)
        plot_summary_sector_posterior_curves(all_bin_results, SUMMARY_DIR)
        corner_pdf_paths = plot_cap_mcmc_corner_pdf(all_bin_results, SUMMARY_DIR)
        plot_summary_radio_full_stacks(all_bin_results, SUMMARY_DIR, stack_norm)
        plot_summary_oriented_selected_ba_histograms(all_bin_results, h5f, SUMMARY_DIR)
        print_cap_mcmc_fit_tables(all_bin_results)
    finally:
        h5f.close()
    print('\nDone. PDFs written by this pipeline:')
    for pdf_name in ORIENTED_PDF_NAMES:
        print(f'  {pdf_name}')
    for pdf_path in corner_pdf_paths:
        print(f'  {os.path.basename(pdf_path)}')


if __name__ == "__main__":
    main()
