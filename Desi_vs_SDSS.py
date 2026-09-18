import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from pixell import enmap, utils
from scipy import spatial


# ============================================================
# Paths
# ============================================================

FIREFLY_PATH = (
    "/Users/jerrywang/Documents/Battaglia_research/"
    "Project1/Catalogs/sdss_firefly-26.fits"
)

PHOTO_PATH = (
    "/Users/jerrywang/Documents/Battaglia_research/"
    "Project1/Catalogs/photoPosPlate-dr17.fits"
)

DESI_PATH = (
    "/Users/jerrywang/Documents/Battaglia_research/"
    "Project1/Catalogs/dr1_galaxy_stellarmass_lineinfo_v1.0.fits"
)

TSZ_MAP_PATH = (
    "/Users/jerrywang/Documents/Battaglia_research/"
    "Project1/Catalogs/act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.2_24.0.fits"
)

FIRST_PATH = (
    "/Users/jerrywang/Documents/Battaglia_research/"
    "Project1/Catalogs/first_14dec17.fits"
)


# ============================================================
# Configuration
# ============================================================

MATCH_RADIUS_ARCSEC = 1.0

# Match the DESI oriented-stacking parent-sample cuts exactly.
MASS_MIN = 11.0
MASS_MAX = 12.0
USE_REDSHIFT_CUT = True
Z_MIN = 0.2
Z_MAX = 0.6
STACK_BA_MAX = 0.75
STAMP_SOURCE_RADIUS_ARCMIN = 21.25

# These are False in oriented_stacking_desi(4).py. They are included here
# so this script follows the same logic if you later switch one on.
RADIO_ONLY = False
EXCLUDE_RADIO = False
FIRST_MATCH_ARCSEC = 60.0
SAFETY = 1

# PA is unreliable for nearly round galaxies, so retain the original
# elongated-galaxy cut for the PA comparison only.
PA_BA_MAX = 0.5


# ============================================================
# Helper: paired scatter + difference histogram
# ============================================================


def full_stamp_inside_map(ra_deg, dec_deg, emap, stamp_source_radius_arcmin):
    """Same map-footprint test used by oriented_stacking_desi(4).py."""
    dec_rad = np.deg2rad(dec_deg)
    ra_rad = np.deg2rad(ra_deg)
    r_rad = np.full(len(ra_deg), np.deg2rad(stamp_source_radius_arcmin / 60.0))
    keep = np.ones(len(ra_deg), dtype=bool)

    offsets_dec = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.float64)
    offsets_ra_factor = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.float64)

    for j in range(8):
        test_dec = dec_rad + offsets_dec[j] * r_rad
        test_ra = ra_rad + offsets_ra_factor[j] * r_rad
        keep &= emap.contains(np.vstack([test_dec, test_ra]))

    return keep


def crossmatch_to_first(ra, dec, first_path, match_arcsec, safety=1):
    """Same FIRST-selection logic used by oriented_stacking_desi(4).py."""
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
        group = np.asarray(group)
        dists = utils.angdist(pos1[gi, :, None], pos2[group, :].T)
        best = np.argmin(dists)
        if dists[best] <= tol:
            has_radio[gi] = True

    return has_radio


def scatter_hist_figure(
    x,
    y,
    x_label,
    y_label,
    diff_label,
    title,
    scatter_limits=None,
    hist_bins=80,
    diff=None,
    reference_line=True,
    selection_cut=None,
):
    """Make one figure containing a scatter plot and a histogram.

    Left:  catalog-to-catalog scatter.
    Right: histogram of DESI - SDSS differences (or supplied `diff`).
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if diff is None:
        d = y - x
    else:
        d = np.asarray(diff, dtype=float)[good]

    fig, (ax_scatter, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    # --------------------
    # Scatter panel
    # --------------------
    ax_scatter.scatter(x, y, s=5, alpha=0.15)

    if scatter_limits is None:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
    else:
        lo, hi = scatter_limits
        ax_scatter.set_xlim(lo, hi)
        ax_scatter.set_ylim(lo, hi)

    if reference_line:
        ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")

    if selection_cut is not None:
        ax_scatter.axvline(selection_cut, color="gray", ls=":", lw=1)
        ax_scatter.axhline(selection_cut, color="gray", ls=":", lw=1)

    ax_scatter.set_xlabel(x_label)
    ax_scatter.set_ylabel(y_label)
    ax_scatter.set_title("Scatter")

    if reference_line:
        ax_scatter.legend()

    # --------------------
    # Histogram panel
    # --------------------
    ax_hist.hist(d[np.isfinite(d)], bins=hist_bins)
    ax_hist.axvline(0.0, color="k", ls="--", lw=1)
    ax_hist.set_xlabel(diff_label)
    ax_hist.set_ylabel("Number of galaxies")
    ax_hist.set_title("Difference histogram")

    fig.suptitle(title)
    plt.show()


# ============================================================
# 1. Load SDSS / FIREFLY galaxies
# ============================================================

with fits.open(FIREFLY_PATH, memmap=True) as hdul:
    ff = hdul[1].data

    ra_ff = ff["PLUG_RA"].astype(float)
    dec_ff = ff["PLUG_DEC"].astype(float)

    # Quantities to compare with DESI
    z_ff = ff["Z"].astype(float)
    mstar_ff = ff["Chabrier_MILES_stellar_mass"].astype(float)

    object_class = np.char.upper(
        np.char.strip(ff["CLASS"].astype(str))
    )
    ff_galaxy = object_class == "GALAXY"


print("FIREFLY galaxies:", np.sum(ff_galaxy))


# ============================================================
# 2. Load SDSS photometric shapes
# ============================================================

with fits.open(PHOTO_PATH, memmap=True) as hdul:
    photo = hdul[1].data

    ra_photo = photo["RA"].astype(float)
    dec_photo = photo["DEC"].astype(float)

    # r-band = index 2, matching the SDSS pipeline
    phi_dev = photo["PHI_DEV_DEG"][:, 2].astype(float)
    phi_exp = photo["PHI_EXP_DEG"][:, 2].astype(float)

    ab_dev = photo["AB_DEV"][:, 2].astype(float)
    ab_exp = photo["AB_EXP"][:, 2].astype(float)

    fracdev = photo["FRACDEV"][:, 2].astype(float)
    ptype = photo["TYPE"][:, 2].astype(int)


# ============================================================
# 3. Match FIREFLY -> SDSS photoPosPlate
# ============================================================

ff_coord = SkyCoord(
    ra=ra_ff[ff_galaxy] * u.deg,
    dec=dec_ff[ff_galaxy] * u.deg,
)

photo_coord = SkyCoord(
    ra=ra_photo * u.deg,
    dec=dec_photo * u.deg,
)

idx_photo, sep_photo, _ = ff_coord.match_to_catalog_sky(photo_coord)
matched_photo = sep_photo.arcsec < MATCH_RADIUS_ARCSEC


# Pick deV or exponential exactly like the original SDSS code
use_dev = fracdev[idx_photo] > 0.5

pa_sdss = np.where(
    use_dev,
    phi_dev[idx_photo],
    phi_exp[idx_photo],
)

ba_sdss = np.where(
    use_dev,
    ab_dev[idx_photo],
    ab_exp[idx_photo],
)


valid_sdss = (
    matched_photo
    & np.isfinite(pa_sdss)
    & np.isfinite(ba_sdss)
    & (pa_sdss > -900)
    & (pa_sdss < 360)
    & (ba_sdss > 0)
    & (ba_sdss <= 1)
    & (ptype[idx_photo] == 3)
)


# FIREFLY values corresponding to the galaxy-class rows
ra_sdss = ra_ff[ff_galaxy]
dec_sdss = dec_ff[ff_galaxy]
z_sdss_allrows = z_ff[ff_galaxy]
mstar_sdss_allrows = mstar_ff[ff_galaxy]


print("Valid SDSS shapes:", np.sum(valid_sdss))


# ============================================================
# 4. Load DESI and calculate DESI PA and b/a
# ============================================================

with fits.open(DESI_PATH, memmap=True) as hdul:
    if "EMLINES_MASS" in hdul:
        desi = hdul["EMLINES_MASS"].data
    else:
        desi = hdul[1].data

    targetid_desi = desi["TARGETID"].astype(np.int64)
    ra_desi = desi["TARGET_RA"].astype(float)
    dec_desi = desi["TARGET_DEC"].astype(float)
    z_desi = desi["Z"].astype(float)
    mstar_desi = desi["MASS_CG"].astype(float)
    e1 = desi["SHAPE_E1"].astype(float)
    e2 = desi["SHAPE_E2"].astype(float)

logm_desi = np.full(mstar_desi.shape, np.nan, dtype=float)
positive_mass = np.isfinite(mstar_desi) & (mstar_desi > 0)
logm_desi[positive_mass] = np.log10(mstar_desi[positive_mass])

e = np.hypot(e1, e2)
ba_desi = (1.0 - e) / (1.0 + e)
phi_desi = 0.5 * np.degrees(np.arctan2(e2, e1))
pa_desi = np.mod(phi_desi, 180.0)

valid_shape_desi = (
    np.isfinite(e1)
    & np.isfinite(e2)
    & np.isfinite(e)
    & np.isfinite(ba_desi)
    & np.isfinite(pa_desi)
    & (e >= 0.0)
    & (e < 1.0)
    & (ba_desi > 0.0)
    & (ba_desi <= 1.0)
)


# ============================================================
# 5. Reproduce the DESI oriented-stacking PARENT selection
#
# IMPORTANT: this intentionally stops BEFORE stamp_valid.
# These are the galaxies selected by the science cuts, whether or not
# thumbnail extraction later succeeded.
# ============================================================

n_desi = len(targetid_desi)

finite_data_mask = (
    np.isfinite(ra_desi)
    & np.isfinite(dec_desi)
    & np.isfinite(logm_desi)
    & np.isfinite(z_desi)
)

mass_range_mask = (
    np.isfinite(logm_desi)
    & (logm_desi > MASS_MIN)
    & (logm_desi <= MASS_MAX)
)

redshift_mask = np.ones(n_desi, dtype=bool)
if USE_REDSHIFT_CUT:
    redshift_mask = (
        np.isfinite(z_desi)
        & (z_desi >= Z_MIN)
        & (z_desi <= Z_MAX)
    )

cum_finite = finite_data_mask
cum_mass = cum_finite & mass_range_mask
cum_z = cum_mass & redshift_mask

shape_mask = (
    valid_shape_desi
    & np.isfinite(pa_desi)
    & np.isfinite(ba_desi)
    & (ba_desi > 0.0)
    & (ba_desi < STACK_BA_MAX)
)

cum_shape = cum_z & shape_mask

print("Loading ACT-Planck map for the same footprint cut used by the stack...")
comptony = enmap.read_map(TSZ_MAP_PATH)
inside_map = np.zeros(n_desi, dtype=bool)
inside_map[cum_shape] = full_stamp_inside_map(
    ra_desi[cum_shape],
    dec_desi[cum_shape],
    comptony,
    STAMP_SOURCE_RADIUS_ARCMIN,
)
cum_map = cum_shape & inside_map

# Same radio-selection logic as the stacking script.
# In the uploaded run RADIO_ONLY=False and EXCLUDE_RADIO=False, so this
# reduces to no radio cut.
if RADIO_ONLY and EXCLUDE_RADIO:
    raise ValueError("RADIO_ONLY and EXCLUDE_RADIO cannot both be True")

if RADIO_ONLY or EXCLUDE_RADIO:
    has_radio_desi = np.zeros(n_desi, dtype=bool)
    has_radio_desi[cum_map] = crossmatch_to_first(
        ra_desi[cum_map],
        dec_desi[cum_map],
        FIRST_PATH,
        FIRST_MATCH_ARCSEC,
        safety=SAFETY,
    )

    if RADIO_ONLY:
        radio_mask = has_radio_desi
    else:
        radio_mask = ~has_radio_desi
else:
    radio_mask = np.ones(n_desi, dtype=bool)

selected_desi = cum_map & radio_mask

print()
print("DESI parent-sample accounting")
print("--------------------------------")
print(f"All DESI rows                         = {n_desi:,}")
print(f"Finite RA/Dec/z/MASS_CG              = {cum_finite.sum():,}")
print(f"Mass cut ({MASS_MIN:.1f}, {MASS_MAX:.1f}]                 = {cum_mass.sum():,}")
if USE_REDSHIFT_CUT:
    print(f"Redshift cut [{Z_MIN:.1f}, {Z_MAX:.1f}]                  = {cum_z.sum():,}")
print(f"Shape + 0 < b/a < {STACK_BA_MAX:.2f}              = {cum_shape.sum():,}")
print(f"ACT-Planck map footprint              = {cum_map.sum():,}")
print(f"FINAL DESI comparison parent sample   = {selected_desi.sum():,}")
print("(No stamp_valid cut is applied.)")


# ============================================================
# 6. Crossmatch ONLY the selected DESI parent sample -> SDSS/FIREFLY
# ============================================================

sdss_coord = SkyCoord(
    ra=ra_sdss[valid_sdss] * u.deg,
    dec=dec_sdss[valid_sdss] * u.deg,
)

desi_coord = SkyCoord(
    ra=ra_desi[selected_desi] * u.deg,
    dec=dec_desi[selected_desi] * u.deg,
)

idx_sdss, sep, _ = desi_coord.match_to_catalog_sky(sdss_coord)
match = sep.arcsec < MATCH_RADIUS_ARCSEC

# Indices into the valid subsets
sdss_match_idx = idx_sdss[match]


# DESI matched values
pa_d_all = pa_desi[selected_desi][match]
ba_d_all = ba_desi[selected_desi][match]
z_d_all = z_desi[selected_desi][match]
mstar_d_all = mstar_desi[selected_desi][match]


# SDSS/FIREFLY matched values
pa_s_all = pa_sdss[valid_sdss][sdss_match_idx]
ba_s_all = ba_sdss[valid_sdss][sdss_match_idx]
z_s_all = z_sdss_allrows[valid_sdss][sdss_match_idx]
mstar_s_all = mstar_sdss_allrows[valid_sdss][sdss_match_idx]


print()
print("DESI-SDSS matches:", len(pa_d_all))


# ============================================================
# 7. Stellar-mass comparison
# ============================================================

mass_good = (
    np.isfinite(mstar_s_all)
    & np.isfinite(mstar_d_all)
    & (mstar_s_all > 0)
    & (mstar_d_all > 0)
)

logm_s = np.log10(mstar_s_all[mass_good])
logm_d = np.log10(mstar_d_all[mass_good])

dlogm = logm_d - logm_s

print()
print("Stellar-mass agreement")
print("--------------------------------")
print(f"N                 = {len(logm_s):,}")
print(f"median ΔlogM      = {np.median(dlogm):.4f} dex")
print(f"median |ΔlogM|    = {np.median(np.abs(dlogm)):.4f} dex")

scatter_hist_figure(
    logm_s,
    logm_d,
    x_label=r"FIREFLY $\log_{10}(M_*/M_\odot)$",
    y_label=r"DESI $\log_{10}(M_*/M_\odot)$",
    diff_label=r"$\log M_{*,\rm DESI} - \log M_{*,\rm FIREFLY}$ [dex]",
    title="Stellar mass: DESI vs FIREFLY",
    hist_bins=80,
)


# ============================================================
# 8. Redshift comparison
# ============================================================

z_good = (
    np.isfinite(z_s_all)
    & np.isfinite(z_d_all)
    & (z_s_all >= 0)
    & (z_d_all >= 0)
)

z_s = z_s_all[z_good]
z_d = z_d_all[z_good]
dz = z_d - z_s

print()
print("Redshift agreement")
print("--------------------------------")
print(f"N                 = {len(z_s):,}")
print(f"median Δz         = {np.median(dz):.6g}")
print(f"median |Δz|       = {np.median(np.abs(dz)):.6g}")

scatter_hist_figure(
    z_s,
    z_d,
    x_label="FIREFLY redshift",
    y_label="DESI redshift",
    diff_label=r"$z_{\rm DESI} - z_{\rm FIREFLY}$",
    title="Redshift: DESI vs FIREFLY",
    hist_bins=80,
)


# ============================================================
# 9. b/a comparison
# ============================================================

dba = ba_d_all - ba_s_all

print()
print("b/a agreement")
print("--------------------------------")
print(f"N                 = {len(dba):,}")
print(f"median Δ(b/a)     = {np.median(dba):.4f}")
print(f"median |Δ(b/a)|   = {np.median(np.abs(dba)):.4f}")

scatter_hist_figure(
    ba_s_all,
    ba_d_all,
    x_label=r"SDSS $b/a$",
    y_label=r"DESI $b/a$",
    diff_label=r"$(b/a)_{\rm DESI} - (b/a)_{\rm SDSS}$",
    title=r"Axis ratio $b/a$: DESI vs SDSS",
    scatter_limits=(0, 1),
    hist_bins=np.linspace(-0.6, 0.6, 121),
    selection_cut=STACK_BA_MAX,
)


# ============================================================
# 10. PA comparison
# ============================================================

# Restrict to clearly elongated galaxies in BOTH catalogs.
# PAs of nearly round galaxies are poorly defined.
elongated = (
    (ba_d_all < PA_BA_MAX)
    & (ba_s_all < PA_BA_MAX)
)

pa_d = pa_d_all[elongated]
pa_s = pa_s_all[elongated]

# PAs are axes, so 0 deg == 180 deg. Wrap DESI-SDSS into [-90, +90).
dpa = ((pa_d - pa_s + 90.0) % 180.0) - 90.0

print()
print("PA agreement")
print("--------------------------------")
print(f"N                 = {len(pa_d):,}")
print(f"median ΔPA        = {np.median(dpa):.2f} deg")
print(f"median |ΔPA|      = {np.median(np.abs(dpa)):.2f} deg")

scatter_hist_figure(
    pa_s,
    pa_d,
    x_label="SDSS PA [deg]",
    y_label="DESI PA [deg]",
    diff_label=r"wrapped $(PA_{\rm DESI} - PA_{\rm SDSS})$ [deg]",
    title=rf"Position angle: DESI vs SDSS (both $b/a < {PA_BA_MAX}$)",
    scatter_limits=(0, 180),
    hist_bins=np.arange(-90, 91, 2),
    diff=dpa,
)
