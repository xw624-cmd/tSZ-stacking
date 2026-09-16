# Oriented tSZ Stacking

Python pipeline for oriented stacking of ACT Compton-y map cutouts around massive galaxies. The code aligns each galaxy by its optical position angle, measures the stacked signal along the projected major and minor axes, and computes sector-based ring-ring CAP profiles with bootstrap covariance estimates.

ThumbStack was used in the following publication:

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

## Dependencies

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


Please feel free to use this code in your work, and contact me with any questions or suggestions (xw624@cornell.edu). If you use it in your publication, please cite the paper above.

