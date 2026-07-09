#!/usr/bin/env python3
"""Run only the stellar-age split-stack products.

Outputs written to SUMMARY_DIR, using the shared CACHE_FILE:
  - summary_stellar_age_stack_4x3.pdf
  - summary_stellar_age_cap_profiles_1x3.pdf
  - summary_stellar_age_hist_ebv_1x3.pdf
  - summary_stellar_age_hist_mass_1x3.pdf
  - summary_stellar_age_hist_mass_weighted_1x3.pdf
  - summary_stellar_age_mass_weights_1x3.pdf

Selection policy for this driver:
  - keeps the E(B-V) dust cut
  - does not cross-match to SDSS photoPosPlate
  - therefore does not apply the FRACDEV/disk cut
"""

import os
import numpy as np

import tsz_stack_common as pipe


# ============================================================
# DRIVER-SPECIFIC SELECTION
# ============================================================
# Stellar-age stacks keep the dust cut, but intentionally do not depend on
# photoPosPlate morphology.  This also guarantees no FRACDEV cut is applied.
pipe.USE_EBV_CUT = True
pipe.USE_PHOTO_SHAPE = False
pipe.USE_DISK_CUT = False

AGE_PDF_NAMES = [
    "summary_stellar_age_stack_4x3.pdf",
    "summary_stellar_age_cap_profiles_1x3.pdf",
    "summary_stellar_age_hist_ebv_1x3.pdf",
    "summary_stellar_age_hist_mass_1x3.pdf",
    "summary_stellar_age_hist_mass_weighted_1x3.pdf",
    "summary_stellar_age_mass_weights_1x3.pdf",
]
pipe.SUMMARY_PDF_NAMES = AGE_PDF_NAMES


def main():
    os.makedirs(pipe.SUMMARY_DIR, exist_ok=True)
    lo_pct = (100.0 - pipe.SPLIT_REMOVE_MIDDLE_PCT) / 2.0
    print("=" * 70)
    print("STELLAR-AGE SPLIT-STACK DRIVER")
    print("  Outputs: age maps/CAP and age-split mass/dust/weight diagnostics")
    print(f"  Summary path: {pipe.SUMMARY_DIR}")
    print(f"  Shared cache: {pipe.CACHE_FILE}")
    print("  Dust cut: ON for this driver")
    print("  photoPosPlate cross-match: OFF")
    print("  Disk/FRACDEV cut: OFF")
    print(f"  Split tails: bottom/top {lo_pct:.0f} percent")
    print(f"  Split-stack mass weights: {'ON' if pipe.USE_SPLIT_MASS_WEIGHTS else 'OFF'}, bins={pipe.MASS_WEIGHT_N_BINS}")
    print("=" * 70)

    h5f, base_fits_idx, _radio_fits_idx = pipe.build_selection_and_cache()

    cache_stamp_valid = h5f["stamp_valid"][:]
    cache_logm = h5f["logm"][:]
    cache_fits_idx = h5f["fits_idx"][:]
    cache_in_current_selection = np.isin(cache_fits_idx, base_fits_idx)

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
    print(f"  current age-selection rows: {cache_in_current_selection.sum():,}")

    print("\nAge-selection accounting by mass bin:")
    pipe.print_mass_bin_accounting("age selection before stamp-valid cut", cache_in_current_selection, cache_logm)
    pipe.print_mass_bin_accounting(
        "age selection after stamp-valid cut",
        cache_in_current_selection & cache_stamp_valid,
        cache_logm,
        cache_in_current_selection,
    )

    all_bin_results = []

    for mass_lo, mass_hi in pipe.MASS_BINS:
        print("\n" + "=" * 70)
        print(f"MASS BIN: log stellar mass ({mass_lo}, {mass_hi}]")
        print("=" * 70)

        in_mass_bin = pipe._mass_bin_mask(cache_logm, mass_lo, mass_hi)
        age_bin_before_stamp = cache_in_current_selection & in_mass_bin
        age_bin_after_stamp = age_bin_before_stamp & cache_stamp_valid

        print("  Age sample accounting inside this mass bin:")
        print(f"    before stamp-valid cut: {int(np.sum(age_bin_before_stamp)):,}")
        print(f"    rejected by stamp_valid=False: {int(np.sum(age_bin_before_stamp & ~cache_stamp_valid)):,}")
        print(f"    after stamp-valid cut: {int(np.sum(age_bin_after_stamp)):,}")

        bin_result = {"mass_lo": mass_lo, "mass_hi": mass_hi}

        if int(age_bin_after_stamp.sum()) == 0:
            for scheme_key in pipe.ACTIVE_SPLIT_SCHEMES:
                bin_result[scheme_key] = {}
        else:
            print("  Age split stacks")
            pipe.add_age_split_results(h5f, bin_result, age_bin_after_stamp, cache_fields)

        all_bin_results.append(bin_result)

    print("\nWriting stellar-age split PDFs...")
    stack_norm = pipe._shared_stack_norm(pipe._collect_stack_values_from_results(all_bin_results))
    pipe.plot_summary_age_split_stacks(all_bin_results, pipe.SUMMARY_DIR, stack_norm)
    pipe.plot_summary_age_split_cap_profiles(all_bin_results, pipe.SUMMARY_DIR)
    pipe.plot_summary_age_split_histograms(all_bin_results, h5f, pipe.SUMMARY_DIR, "logm")
    pipe.plot_summary_age_split_weighted_mass_histograms(all_bin_results, h5f, pipe.SUMMARY_DIR)
    pipe.plot_summary_age_split_mass_weight_values(all_bin_results, h5f, pipe.SUMMARY_DIR)
    pipe.plot_summary_age_split_histograms(all_bin_results, h5f, pipe.SUMMARY_DIR, "EBV")
    pipe.print_cap_significance_tables(all_bin_results)

    print("\nDone. PDFs written by this driver:")
    for pdf_name in AGE_PDF_NAMES:
        print(f"  {pdf_name}")


if __name__ == "__main__":
    main()
