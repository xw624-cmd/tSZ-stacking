#!/usr/bin/env python3

"""Robustness checks for the 2 arcmin oriented CAP measurement.

The script compares the major- and minor-axis CAP values across alternative
Compton-y maps, sector half-opening angles, and radio-source selections. The
same galaxy sample is used within each comparison. It expects
``oriented_stacking.py`` in the same working directory and uses its catalog
selection and thumbnail-remapping utilities.
"""

import hashlib
import json
import os

import h5py
import numpy as np
from pixell import enmap, reproject

import oriented_stacking as ost


# Configuration

PROJECT_DIR = '/Users/jerrywang/Documents/Battaglia_research/Project1'
CATALOG_DIR = os.path.join(PROJECT_DIR, 'Catalogs')
OTHER_MAP_DIR = os.path.join(PROJECT_DIR, 'Other_maps')

MAPS = [
    (
        '1.2, 24.0 (original)',
        os.path.join(
            CATALOG_DIR,
            'act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.2_24.0.fits',
        ),
    ),
    (
        '1.6, 10.7',
        os.path.join(
            OTHER_MAP_DIR,
            'act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.6_10.7.fits',
        ),
    ),
    (
        '1.6, 24.0',
        os.path.join(
            OTHER_MAP_DIR,
            'act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.6_24.0.fits',
        ),
    ),
    (
        '1.8, 24.0',
        os.path.join(
            OTHER_MAP_DIR,
            'act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.8_24.0.fits',
        ),
    ),
]

TEST_RADIUS_ARCMIN = 2.0
MAP_TEST_HALF_ANGLE_DEG = 15.0
ANGLE_TESTS = [
    (15.0, 3.0),
    (30.0, 1.5),
    (45.0, 1.0),
]

CACHE_DIR = './robustness_cache'
EXTRACTION_LOG_EVERY = 2000
BOOTSTRAP_BATCH_SIZE = 25
BOOTSTRAP_SEED = ost.SEED
DISPLAY_SCALE = 1.0e6


# Sample selection

def _file_signature(path):
    stat = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()[:20]


def validate_paths():
    paths = [ost.FIREFLY_PATH, ost.PHOTO_PATH, ost.FIRST_PATH] + [path for _, path in MAPS]
    missing = [path for path in paths if path is None or not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('Missing required input(s):\n  ' + '\n  '.join(missing))


def load_catalog_selection():
    """Apply the catalog cuts used by the main oriented-stacking analysis."""
    ra, dec, logm, redshift, fits_idx, is_galaxy = ost.load_firefly_full(
        ost.FIREFLY_PATH
    )
    n = len(ra)

    finite = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(logm)
        & np.isfinite(redshift)
    )
    mass_cut = (
        np.isfinite(logm)
        & (logm > ost.CACHE_LOG_MASS_MIN)
        & (logm <= ost.CACHE_LOG_MASS_MAX)
    )
    redshift_cut = np.ones(n, dtype=bool)
    if ost.USE_REDSHIFT_CUT:
        redshift_cut = (
            np.isfinite(redshift)
            & (redshift >= ost.Z_MIN)
            & (redshift <= ost.Z_MAX)
        )

    broad = np.asarray(is_galaxy, dtype=bool) & finite & mass_cut & redshift_cut

    pa = np.full(n, np.nan, dtype=np.float64)
    ab = np.full(n, np.nan, dtype=np.float64)
    pa_dev = np.full(n, np.nan, dtype=np.float64)
    pa_exp = np.full(n, np.nan, dtype=np.float64)
    ab_dev = np.full(n, np.nan, dtype=np.float64)
    ab_exp = np.full(n, np.nan, dtype=np.float64)
    fracdev = np.full(n, np.nan, dtype=np.float64)
    valid_shape = np.zeros(n, dtype=bool)

    out = ost.load_photo_shapes(
        ra[broad],
        dec[broad],
        ost.PHOTO_PATH,
        match_arcsec=ost.PHOTO_MATCH_ARCSEC,
        fracdev_thresh=ost.PHOTO_FRACDEV_THRESH,
        type_galaxy=ost.PHOTO_TYPE_GALAXY,
        safety=ost.SAFETY,
    )
    (
        pa_tmp,
        ab_tmp,
        valid_tmp,
        pa_dev_tmp,
        pa_exp_tmp,
        ab_dev_tmp,
        ab_exp_tmp,
        fracdev_tmp,
    ) = out

    pa[broad] = pa_tmp
    ab[broad] = ab_tmp
    pa_dev[broad] = pa_dev_tmp
    pa_exp[broad] = pa_exp_tmp
    ab_dev[broad] = ab_dev_tmp
    ab_exp[broad] = ab_exp_tmp
    fracdev[broad] = fracdev_tmp
    valid_shape[broad] = valid_tmp

    finite_shape = (
        np.isfinite(pa)
        & np.isfinite(ab)
        & np.isfinite(pa_dev)
        & np.isfinite(pa_exp)
        & np.isfinite(ab_dev)
        & np.isfinite(ab_exp)
        & np.isfinite(fracdev)
    )
    selected = broad & valid_shape & finite_shape & (ab < ost.BA_MAX)

    has_radio = np.zeros(n, dtype=bool)
    if ost.FIRST_PATH is not None and np.any(selected):
        has_radio[selected] = ost.crossmatch_to_first(
            ra[selected],
            dec[selected],
            ost.FIRST_PATH,
            ost.FIRST_MATCH_ARCSEC,
            safety=ost.SAFETY,
        )

    print(f'Catalog-selected galaxies before map coverage: {selected.sum():,}')
    print(f'FIRST-matched galaxies in that sample: {np.sum(selected & has_radio):,}')
    return {
        'ra': ra,
        'dec': dec,
        'logm': logm,
        'redshift': redshift,
        'fits_idx': fits_idx,
        'pa': pa,
        'has_radio': has_radio,
        'selected': selected,
    }


def apply_common_map_coverage(catalog):
    """Restrict the catalog to the source-thumbnail footprint of every map."""
    common = np.asarray(catalog['selected'], dtype=bool).copy()
    base = common.copy()

    print('\nMap-footprint checks:')
    for label, path in MAPS:
        emap = enmap.read_map(path)
        inside = np.zeros_like(common)
        inside[base] = ost.full_stamp_inside_map(
            catalog['ra'][base],
            catalog['dec'][base],
            emap,
            ost.STAMP_SOURCE_RADIUS_ARCMIN,
        )
        common &= inside
        print(f'  {label:<22} {inside[base].sum():,} / {base.sum():,}')
        del emap

    print(f'Common map-footprint sample: {common.sum():,}')
    return common


# Thumbnail cache

def _find_thumbnail_geometry(emap, ra, dec, radius_arcmin):
    radius_rad = np.deg2rad(radius_arcmin / 60.0)
    for ra_i, dec_i in zip(ra, dec):
        stamp = reproject.thumbnails(
            emap,
            coords=np.deg2rad([dec_i, ra_i]),
            r=radius_rad,
        )
        if stamp is not None:
            return np.asarray(stamp).shape
    raise RuntimeError('Could not extract a test thumbnail.')


def determine_output_geometry(catalog, selection):
    """Use the original map to define the fixed final Cartesian grid."""
    original_map = enmap.read_map(MAPS[0][1])
    try:
        shape = _find_thumbnail_geometry(
            original_map,
            catalog['ra'][selection],
            catalog['dec'][selection],
            ost.STAMP_RADIUS_ARCMIN,
        )
    finally:
        del original_map

    ny, nx = shape
    pixscale = 2.0 * ost.STAMP_RADIUS_ARCMIN / (ny - 1)
    print(
        f'Fixed output grid: {ny} x {nx}, '
        f'{pixscale:.6f} arcmin per pixel'
    )
    return ny, nx, pixscale


def _cache_path(map_path):
    name = os.path.splitext(os.path.basename(map_path))[0]
    return os.path.join(CACHE_DIR, f'{name}_stamps.h5')


def _cache_config(map_path, fits_idx, ra, dec, source_shape):
    return {
        'format_version': 1,
        'map': _file_signature(map_path),
        'firefly': _file_signature(ost.FIREFLY_PATH),
        'source_radius_arcmin': float(ost.STAMP_SOURCE_RADIUS_ARCMIN),
        'source_shape': [int(source_shape[0]), int(source_shape[1])],
        'sample_hash': _array_hash(fits_idx, ra, dec),
    }


def open_map_cache(map_path, fits_idx, ra, dec, source_shape):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(map_path)
    config = _cache_config(map_path, fits_idx, ra, dec, source_shape)
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:20]

    if os.path.exists(path):
        h5f = h5py.File(path, 'a')
        if h5f.attrs.get('config_hash', '') == config_hash:
            print(f'  cache: {path}')
            return h5f
        h5f.close()
        os.remove(path)

    ny_src, nx_src = source_shape
    h5f = h5py.File(path, 'w')
    h5f.attrs['config_hash'] = config_hash
    h5f.attrs['config_json'] = config_json
    h5f.create_dataset('fits_idx', data=np.asarray(fits_idx, dtype=np.int64))
    h5f.create_dataset('attempted', data=np.zeros(len(fits_idx), dtype=bool))
    h5f.create_dataset('stamp_valid', data=np.zeros(len(fits_idx), dtype=bool))
    h5f.create_dataset(
        'stamps',
        shape=(len(fits_idx), ny_src, nx_src),
        dtype=np.float64,
        chunks=(1, ny_src, nx_src),
        compression='gzip',
        compression_opts=4,
    )
    h5f.flush()
    print(f'  new cache: {path}')
    return h5f


def fill_map_cache(emap, h5f, ra, dec):
    attempted = h5f['attempted'][:]
    n_todo = int(np.sum(~attempted))
    if n_todo == 0:
        return

    ny_src, nx_src = h5f['stamps'].shape[1:]
    radius_rad = np.deg2rad(ost.STAMP_SOURCE_RADIUS_ARCMIN / 60.0)
    done = 0
    success = 0

    for i in np.where(~attempted)[0]:
        stamp = reproject.thumbnails(
            emap,
            coords=np.deg2rad([dec[i], ra[i]]),
            r=radius_rad,
        )
        h5f['attempted'][i] = True
        done += 1

        if stamp is None:
            continue
        arr = np.asarray(stamp, dtype=np.float64)
        if arr.shape != (ny_src, nx_src) or np.all(arr == 0):
            continue

        h5f['stamps'][i] = arr
        h5f['stamp_valid'][i] = True
        success += 1

        if done % EXTRACTION_LOG_EVERY == 0:
            print(f'    extracted {done:,}/{n_todo:,}; valid={success:,}')
            h5f.flush()

    h5f.flush()
    print(f'    extraction complete: {success:,}/{done:,} newly valid')


def prepare_caches(catalog, nominal_selection):
    """Extract source thumbnails and return the sample valid in every map."""
    idx = np.where(nominal_selection)[0]
    fits_idx = catalog['fits_idx'][idx]
    ra = catalog['ra'][idx]
    dec = catalog['dec'][idx]

    caches = []
    valid_masks = []
    print('\nPreparing map caches:')

    for label, path in MAPS:
        print(f'\n{label}')
        emap = enmap.read_map(path)
        source_shape = _find_thumbnail_geometry(
            emap,
            ra,
            dec,
            ost.STAMP_SOURCE_RADIUS_ARCMIN,
        )
        h5f = open_map_cache(path, fits_idx, ra, dec, source_shape)
        fill_map_cache(emap, h5f, ra, dec)
        valid = np.asarray(h5f['stamp_valid'][:], dtype=bool)
        print(f'  valid source thumbnails: {valid.sum():,} / {len(valid):,}')
        caches.append((label, path, h5f))
        valid_masks.append(valid)
        del emap

    common_valid = np.logical_and.reduce(valid_masks)
    print(f'\nCommon valid thumbnail sample: {common_valid.sum():,} / {len(common_valid):,}')
    return idx, caches, common_valid


# CAP measurement

def make_geometry(ny, nx, pixscale):
    x = ost._centered_pixel_axis(nx, pixscale)
    y = ost._centered_pixel_axis(ny, pixscale)
    xg, yg = np.meshgrid(x, y)
    radius = np.hypot(xg, yg)
    angle = np.degrees(np.arctan2(yg, xg)) % 360.0
    return radius, angle


def cap_masks(radius, angle, half_angle_deg):
    outer_radius = np.sqrt(
        2.0 * TEST_RADIUS_ARCMIN**2 - ost.CAP_INNER_CUT_ARCMIN**2
    )
    positive = (
        (radius > ost.CAP_INNER_CUT_ARCMIN)
        & (radius < TEST_RADIUS_ARCMIN)
    )
    negative = (radius > TEST_RADIUS_ARCMIN) & (radius < outer_radius)

    major_sector = ost.sector_mask(
        angle, ost.MAJOR_AXIS_ANGLE, half_angle_deg
    )
    minor_sector = ost.sector_mask(
        angle, ost.MINOR_AXIS_ANGLE, half_angle_deg
    )
    return {
        'major': (positive & major_sector, negative & major_sector),
        'minor': (positive & minor_sector, negative & minor_sector),
    }


def cap_value(image, masks, pixel_area):
    positive, negative = masks
    pos_valid = positive & np.isfinite(image)
    neg_valid = negative & np.isfinite(image)
    n_pos = int(np.sum(pos_valid))
    n_neg = int(np.sum(neg_valid))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    pos_sum = float(np.sum(image[pos_valid]))
    neg_sum = float(np.sum(image[neg_valid]))
    return (pos_sum - neg_sum * n_pos / n_neg) * pixel_area


def measure_map(h5f, pa, valid_rows, ny, nx, pixscale, half_angles):
    """Measure major/minor 2 arcmin CAP values for the requested sectors."""
    rows = np.where(valid_rows)[0]
    radius, angle = make_geometry(ny, nx, pixscale)
    masks = {half: cap_masks(radius, angle, half) for half in half_angles}
    pixel_area = pixscale**2

    result = {
        half: {
            'major': np.full(len(valid_rows), np.nan, dtype=np.float64),
            'minor': np.full(len(valid_rows), np.nan, dtype=np.float64),
        }
        for half in half_angles
    }

    for j, row in enumerate(rows):
        source = np.asarray(h5f['stamps'][row], dtype=np.float64)
        rotated = ost.sample_large_stamp_to_output(
            source,
            -float(pa[row]),
            ny,
            nx,
            pixscale,
        )
        for half in half_angles:
            result[half]['major'][row] = cap_value(
                rotated, masks[half]['major'], pixel_area
            )
            result[half]['minor'][row] = cap_value(
                rotated, masks[half]['minor'], pixel_area
            )

        if (j + 1) % 5000 == 0:
            print(f'    measured {j + 1:,}/{len(rows):,}')

    return result


# Bootstrap and tables

def _condition_names():
    names = []
    for label, _ in MAPS:
        names.extend([
            f'map:{label}:major',
            f'map:{label}:minor',
            f'map:{label}:difference',
        ])
    names.extend([
        'angle:30:major',
        'angle:30:minor',
        'angle:30:difference',
        'angle:45:major',
        'angle:45:minor',
        'angle:45:difference',
    ])
    return names


def build_condition_matrix(map_results, sample_mask):
    """Build major, minor, and direct minor-major columns for every test."""
    columns = []

    for label, _ in MAPS:
        r = map_results[label][MAP_TEST_HALF_ANGLE_DEG]
        major = r['major'][sample_mask]
        minor = r['minor'][sample_mask]
        difference = minor - major
        columns.extend([major, minor, difference])

    original_label = MAPS[0][0]
    for half in (30.0, 45.0):
        r = map_results[original_label][half]
        major = r['major'][sample_mask]
        minor = r['minor'][sample_mask]
        difference = minor - major
        columns.extend([major, minor, difference])

    return np.column_stack(columns)


def bootstrap_matrix(matrix, n_boot, seed, batch_size=BOOTSTRAP_BATCH_SIZE):
    """Bootstrap all columns with common galaxy resamples.

    The minor-major difference is an explicit per-galaxy column, so its
    uncertainty is obtained by bootstrapping y_minor - y_major directly.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.all(np.isfinite(matrix), axis=1)
    data = matrix[finite]
    n, n_col = data.shape
    if n == 0:
        return (
            np.full(n_col, np.nan),
            np.full(n_col, np.nan),
            0,
        )

    means = np.mean(data, axis=0)
    if n < 2 or n_boot <= 1:
        return means, np.zeros(n_col, dtype=np.float64), n

    rng = np.random.default_rng(seed)
    boot_sum = np.zeros(n_col, dtype=np.float64)
    boot_sumsq = np.zeros(n_col, dtype=np.float64)
    completed = 0

    while completed < n_boot:
        batch = min(batch_size, n_boot - completed)
        draw = rng.integers(0, n, size=(batch, n), dtype=np.int32)
        boot_means = np.mean(data[draw], axis=1)
        boot_sum += np.sum(boot_means, axis=0)
        boot_sumsq += np.sum(boot_means**2, axis=0)
        completed += batch

    variance = (
        boot_sumsq - boot_sum**2 / float(n_boot)
    ) / float(n_boot - 1)
    std = np.sqrt(np.clip(variance, 0.0, None))
    return means, std, n


def format_value(mean, std, scale=1.0):
    mean = mean * scale * DISPLAY_SCALE
    std = std * scale * DISPLAY_SCALE
    return f'{mean:.4f} +/- {std:.4f}'


def format_significance(value):
    return f'{value:.4f} sigma' if np.isfinite(value) else 'nan sigma'


def format_pair_result(major, minor, difference, scale=1.0):
    """Format major, minor, direct-bootstrap difference, and significance."""
    diff_mean, diff_std = difference
    significance = (
        abs(diff_mean) / diff_std
        if np.isfinite(diff_std) and diff_std > 0.0
        else np.nan
    )
    return (
        format_value(*major, scale=scale),
        format_value(*minor, scale=scale),
        format_value(diff_mean, diff_std, scale=scale),
        format_significance(significance),
    )


def print_grouped_table(title, row_labels, row_values):
    """Print three mass-bin groups with major/minor/difference/significance."""
    cell_w = 22
    row_w = max(25, max(len(label) for label in row_labels) + 2)

    print('\n' + title)
    print('Major, minor, and minor-major are in 10^-6 y arcmin^2; S_delta is in sigma.')

    header1 = ' ' * row_w
    for lo, hi in ost.MASS_BINS:
        label = f'logM ({lo:.1f}, {hi:.1f}]'
        header1 += f'{label:^{4 * cell_w}}'
    print(header1)

    header2 = f'{"":<{row_w}}'
    for _ in ost.MASS_BINS:
        header2 += (
            f'{"Major":^{cell_w}}'
            f'{"Minor":^{cell_w}}'
            f'{"Minor-Major":^{cell_w}}'
            f'{"S_delta":^{cell_w}}'
        )
    print(header2)
    print('-' * (row_w + len(ost.MASS_BINS) * 4 * cell_w))

    for label, values in zip(row_labels, row_values):
        line = f'{label:<{row_w}}'
        for major, minor, difference, significance in values:
            line += (
                f'{major:^{cell_w}}'
                f'{minor:^{cell_w}}'
                f'{difference:^{cell_w}}'
                f'{significance:^{cell_w}}'
            )
        print(line)


def main():
    validate_paths()

    print('=' * 78)
    print('ORIENTED CAP ROBUSTNESS CHECKS')
    print(f'CAP radius: {TEST_RADIUS_ARCMIN:g} arcmin')
    print(f'Bootstrap realizations: {ost.N_BOOT:,}')
    print('=' * 78)

    catalog = load_catalog_selection()
    nominal_selection = apply_common_map_coverage(catalog)
    ny, nx, pixscale = determine_output_geometry(catalog, nominal_selection)

    selected_idx, caches, common_valid = prepare_caches(
        catalog, nominal_selection
    )
    try:
        logm = catalog['logm'][selected_idx]
        pa = catalog['pa'][selected_idx]
        has_radio = catalog['has_radio'][selected_idx]

        map_results = {}
        original_label = MAPS[0][0]

        print('\nMeasuring 2 arcmin CAP values:')
        for label, _, h5f in caches:
            print(f'\n{label}')
            half_angles = [MAP_TEST_HALF_ANGLE_DEG]
            if label == original_label:
                half_angles = [15.0, 30.0, 45.0]
            map_results[label] = measure_map(
                h5f,
                pa,
                common_valid,
                ny,
                nx,
                pixscale,
                half_angles,
            )

        condition_names = _condition_names()
        stats = []

        print('\nBootstrap sample sizes:')
        for bin_index, (lo, hi) in enumerate(ost.MASS_BINS):
            mass_mask = (
                common_valid
                & np.isfinite(logm)
                & (logm > lo)
                & (logm <= hi)
            )
            matrix = build_condition_matrix(map_results, mass_mask)
            means, stds, n_used = bootstrap_matrix(
                matrix,
                n_boot=ost.N_BOOT,
                seed=BOOTSTRAP_SEED,
            )
            stats.append({
                name: (means[i], stds[i])
                for i, name in enumerate(condition_names)
            })
            print(f'  logM ({lo:.1f}, {hi:.1f}]: {n_used:,}')

        map_rows = []
        for label, _ in MAPS:
            values = []
            major_name = f'map:{label}:major'
            minor_name = f'map:{label}:minor'
            difference_name = f'map:{label}:difference'
            for bin_stats in stats:
                values.append(format_pair_result(
                    bin_stats[major_name],
                    bin_stats[minor_name],
                    bin_stats[difference_name],
                ))
            map_rows.append(values)

        print_grouped_table(
            f'MAP ROBUSTNESS: theta_d = {TEST_RADIUS_ARCMIN:g} arcmin, '
            f'half-opening angle = {MAP_TEST_HALF_ANGLE_DEG:g} deg',
            [label for label, _ in MAPS],
            map_rows,
        )

        angle_rows = []
        angle_labels = []
        for half, rescale in ANGLE_TESTS:
            values = []
            if half == 15.0:
                major_name = f'map:{original_label}:major'
                minor_name = f'map:{original_label}:minor'
                difference_name = f'map:{original_label}:difference'
            else:
                major_name = f'angle:{int(half)}:major'
                minor_name = f'angle:{int(half)}:minor'
                difference_name = f'angle:{int(half)}:difference'
            for bin_stats in stats:
                values.append(format_pair_result(
                    bin_stats[major_name],
                    bin_stats[minor_name],
                    bin_stats[difference_name],
                    scale=rescale,
                ))
            angle_rows.append(values)
            angle_labels.append(f'{half:g} deg (x{rescale:g})')

        print_grouped_table(
            f'HALF-OPENING-ANGLE ROBUSTNESS: theta_d = {TEST_RADIUS_ARCMIN:g} arcmin '
            '(original map; rescaled to the 45 deg sector area)',
            angle_labels,
            angle_rows,
        )

        radio_stats = []
        original_15 = map_results[original_label][15.0]
        print('\nRadio-removal sample sizes:')
        for bin_index, (lo, hi) in enumerate(ost.MASS_BINS):
            in_mass = (
                common_valid
                & np.isfinite(logm)
                & (logm > lo)
                & (logm <= hi)
            )
            selections = {
                'All galaxies': in_mass,
                'FIRST removed': in_mass & ~has_radio,
            }
            bin_stats = {}
            for selection_index, (label, sample_mask) in enumerate(selections.items()):
                major_values = original_15['major'][sample_mask]
                minor_values = original_15['minor'][sample_mask]
                difference_values = minor_values - major_values
                matrix = np.column_stack([
                    major_values,
                    minor_values,
                    difference_values,
                ])
                means, stds, n_used = bootstrap_matrix(
                    matrix,
                    n_boot=ost.N_BOOT,
                    seed=BOOTSTRAP_SEED,
                )
                bin_stats[label] = {
                    'major': (means[0], stds[0]),
                    'minor': (means[1], stds[1]),
                    'difference': (means[2], stds[2]),
                    'n': n_used,
                }
            radio_stats.append(bin_stats)
            print(
                f'  logM ({lo:.1f}, {hi:.1f}]: '
                f'all={bin_stats["All galaxies"]["n"]:,}, '
                f'FIRST removed={bin_stats["FIRST removed"]["n"]:,}'
            )

        radio_rows = []
        for label in ('All galaxies', 'FIRST removed'):
            values = []
            for bin_stats in radio_stats:
                values.append(format_pair_result(
                    bin_stats[label]['major'],
                    bin_stats[label]['minor'],
                    bin_stats[label]['difference'],
                ))
            radio_rows.append(values)

        print_grouped_table(
            f'RADIO ROBUSTNESS: theta_d = {TEST_RADIUS_ARCMIN:g} arcmin, '
            'half-opening angle = 15 deg (original map; no angular rescaling)',
            ['All galaxies', 'FIRST removed'],
            radio_rows,
        )

    finally:
        for _, _, h5f in caches:
            h5f.close()


if __name__ == '__main__':
    main()
