import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


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


# ============================================================
# Configuration
# ============================================================

MATCH_RADIUS_ARCSEC = 1.0

# PA is unreliable for nearly round galaxies, so retain the
# original elongated-galaxy cut for the PA comparison only.
PA_BA_MAX = 0.5

# Kept because it is useful to show on the b/a comparison.
STACK_BA_MAX = 0.75


# ============================================================
# Helper: paired scatter + difference histogram
# ============================================================

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

    ra_desi = desi["TARGET_RA"].astype(float)
    dec_desi = desi["TARGET_DEC"].astype(float)

    z_desi = desi["Z"].astype(float)
    mstar_desi = desi["MASS_CG"].astype(float)

    e1 = desi["SHAPE_E1"].astype(float)
    e2 = desi["SHAPE_E2"].astype(float)


e = np.hypot(e1, e2)
ba_desi = (1.0 - e) / (1.0 + e)

phi_desi = 0.5 * np.degrees(np.arctan2(e2, e1))
pa_desi = np.mod(phi_desi, 180.0)


valid_desi = (
    np.isfinite(pa_desi)
    & np.isfinite(ba_desi)
    & np.isfinite(e)
    & (e >= 0)
    & (e < 1)
    & (ba_desi > 0)
    & (ba_desi <= 1)
)


print("Valid DESI shapes:", np.sum(valid_desi))


# ============================================================
# 5. Crossmatch DESI -> SDSS/FIREFLY
# ============================================================

sdss_coord = SkyCoord(
    ra=ra_sdss[valid_sdss] * u.deg,
    dec=dec_sdss[valid_sdss] * u.deg,
)

desi_coord = SkyCoord(
    ra=ra_desi[valid_desi] * u.deg,
    dec=dec_desi[valid_desi] * u.deg,
)

idx_sdss, sep, _ = desi_coord.match_to_catalog_sky(sdss_coord)
match = sep.arcsec < MATCH_RADIUS_ARCSEC


# Indices into the valid subsets
sdss_match_idx = idx_sdss[match]


# DESI matched values
pa_d_all = pa_desi[valid_desi][match]
ba_d_all = ba_desi[valid_desi][match]
z_d_all = z_desi[valid_desi][match]
mstar_d_all = mstar_desi[valid_desi][match]


# SDSS/FIREFLY matched values
pa_s_all = pa_sdss[valid_sdss][sdss_match_idx]
ba_s_all = ba_sdss[valid_sdss][sdss_match_idx]
z_s_all = z_sdss_allrows[valid_sdss][sdss_match_idx]
mstar_s_all = mstar_sdss_allrows[valid_sdss][sdss_match_idx]


print()
print("DESI-SDSS matches:", len(pa_d_all))


# ============================================================
# 6. Stellar-mass comparison
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
# 7. Redshift comparison
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
# 8. b/a comparison
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
# 9. PA comparison
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
    hist_bins=np.arange(-25, 26, 1),
    diff=dpa,
)
