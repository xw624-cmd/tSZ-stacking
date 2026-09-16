# Oriented tSZ Stacking

Python pipeline for oriented stacking of ACT-Planck Compton-$y$ cutouts around massive SDSS galaxies. The code aligns each galaxy by its optical position angle, measures the stacked signal along the projected major and minor axes, and computes sector-based compensated aperture photometry (CAP) profiles with bootstrap covariance estimates.

The analysis is designed for the oriented-stacking measurements described in *A Search for Thermal Sunyaev-Zel'dovich Anisotropy Around Massive Galaxies with Oriented Stacking*.

## Overview

The pipeline:

1. loads the FIREFLY galaxy catalog;
2. applies stellar-mass and redshift cuts;
3. cross-matches galaxies to SDSS DR17 `photoPosPlate` measurements;
4. selects the optical position angle and axis ratio from the de Vaucouleurs or exponential fit according to `fracDeV`;
5. applies the projected axis-ratio cut;
6. checks coverage in the ACT-Planck Compton-$y$ map;
7. optionally identifies galaxies with nearby FIRST radio sources;
8. extracts and caches Compton-$y$ thumbnails;
9. rotates each thumbnail so that the projected galaxy major axis is vertical;
10. stacks the unoriented and oriented thumbnails in stellar-mass bins;
11. measures major- and minor-axis CAP profiles in $\pm15^\circ$ sectors;
12. estimates profile uncertainties and covariances with bootstrap resampling;
13. writes figures and paired-bootstrap products for downstream modeling.

## Input data

The file paths are set near the top of `oriented_stacking.py`.

- **FIREFLY DR16 value-added catalog**  
  https://data.sdss.org/sas/dr16/eboss/spectro/firefly/v1_1_0/

- **SDSS DR17 photoPosPlate catalog**  
  https://www.sdss4.org/dr17/spectro/spectro_access/

- **ACT DR6.02 / Planck Compton-$y$ map**  
  https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_nilc_get.html

- **FIRST catalog**  
  https://sundog.stsci.edu/first/catalogs/readme_14dec17.html

The current configuration uses the ACT DR6.02 NILC Compton-$y$ map with CIB deprojection parameters $\beta=1.2$ and $T_{\rm dust}=24\,$K.

## Requirements

The pipeline requires Python and the following packages:

```text
numpy
matplotlib
astropy
pixell
scipy
h5py
```

Matplotlib is configured to use LaTeX for figure text, so a working TeX installation is also required.

A minimal environment can be created with, for example,

```bash
pip install numpy matplotlib astropy pixell scipy h5py
```

## Configuration

The main analysis choices are defined at the top of the script. The default configuration includes:

```python
MASS_BINS = [(11.0, 11.4), (11.4, 11.7), (11.7, 12.0)]
Z_MIN = 0.2
Z_MAX = 0.6
BA_MAX = 0.75
WEDGE_HALF_DEG = 15.0
CAP_INNER_CUT_ARCMIN = 1.0
CAP_RADII_ARCMIN = np.linspace(2.0, 8.0, 7)
N_BOOT = 10000
```

The mass bins use

$$
M_{\rm low} < \log_{10}(M_*/M_\odot) \le M_{\rm high}.
$$

Radio-associated galaxies are retained by default. `RADIO_ONLY` and `EXCLUDE_RADIO` can be used for radio-source tests.

## Oriented stacking

For each selected galaxy, the pipeline extracts a $42.5'\times42.5'$ source thumbnail from the Compton-$y$ map. The thumbnail is then sampled onto a final $30'\times30'$ grid using bilinear interpolation.

Two versions are accumulated:

- an **unoriented stack**, with no rotation;
- an **oriented stack**, rotated by the galaxy position angle so that the projected optical major axis is vertical.

The major-axis sectors are centered on the vertical direction and the minor-axis sectors on the horizontal direction. Each axis is represented by two opposite wedges with a half-opening angle of $15^\circ$.

Pixel-center coordinates determine radial and angular membership in the CAP masks.

## Ring-ring CAP filter

The profile measurement uses a ring-ring compensated aperture filter with an inner exclusion radius $\theta_0=1'$.

For each aperture radius $\theta_d$, the positive region spans

$$
\theta_0 < \theta < \theta_d,
$$

and the surrounding subtraction annulus extends to

$$
\theta_{\rm out}=\sqrt{2\theta_d^2-\theta_0^2}.
$$

On the discrete pixel grid, the CAP measurement is

$$
y_{\rm CAP}=\left(\sum_{\rm inner}y-\frac{N_{\rm inner}}{N_{\rm outer}}\sum_{\rm outer}y\right)A_{\rm pix},
$$

where $N_{\rm inner}$ and $N_{\rm outer}$ are the numbers of finite pixels in the two regions. This pixel-count normalization makes the discrete filter exactly compensated for a constant background.

The sector CAP measurements use the same filter after restricting both regions to the chosen major- or minor-axis wedges.

## Bootstrap covariance

The code performs `N_BOOT = 10000` bootstrap realizations by resampling galaxies with replacement.

Major- and minor-axis profiles use the same bootstrap draw in each realization. This preserves the cross-covariance between the two sectors. The pipeline stores the individual sector covariances, the major-minor cross-covariance, and the full joint covariance matrix.

## Running the pipeline

After setting the input paths,

```bash
python oriented_stacking.py
```

The first run builds an HDF5 thumbnail cache in

```text
./stamp_cache/stamps.h5
```

and subsequent runs reuse cached thumbnails when the extraction configuration is unchanged.

Analysis products are written to

```text
./run_output/
```

## Output products

The pipeline currently writes four summary figures:

```text
summary_oriented_full_stack_2x3.pdf
summary_oriented_full_stack_cap_profiles_1x3.pdf
summary_oriented_sector_cap_correlation_2x3.pdf
summary_oriented_hist_ba_selected_1x3.pdf
```

It also exports paired major/minor bootstrap products for downstream profile modeling:

```text
paired_sector_cap_bootstraps.npz
paired_sector_cap_bootstraps.json
```

The NPZ file contains the measured profiles, bootstrap realizations, covariance matrices, and the galaxy metadata associated with each stellar-mass bin. The JSON file records the filter definition, aperture radii, sector geometry, units, and array conventions.

At the end of each run, the script prints additional diagnostics including:

- the median redshift in each stellar-mass bin;
- the major-minor CAP difference significance at $2'$;
- the positive (`raw`) and subtraction components of the major- and minor-axis CAP measurements at each aperture radius.

For the last diagnostic,

$$
Y_{\rm raw}=\left(\sum_{\rm inner}y\right)A_{\rm pix},
$$

and

$$
Y_{\rm sub}=\left(\frac{N_{\rm inner}}{N_{\rm outer}}\sum_{\rm outer}y\right)A_{\rm pix},
$$

so that $Y_{\rm CAP}=Y_{\rm raw}-Y_{\rm sub}$.

## Related software

The stacking and aperture-photometry workflow follows methods used in ACT stacking analyses and is closely related to the public **ThumbStack** code developed by Emmanuel Schaan:

https://github.com/EmmanuelSchaan/ThumbStack

ThumbStack produces stacked maps and aperture-photometry profiles with bootstrap covariance estimates for thermal and kinematic Sunyaev-Zel'dovich analyses.

## Citation

If you use this code or results derived from it, please cite the associated analysis paper when available. The CAP and stacking methodology also builds on the literature referenced in the manuscript and on the ThumbStack framework above.

## Contact

Jerry Wang  
Department of Astronomy, Cornell University
