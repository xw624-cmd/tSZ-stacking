#!/usr/bin/env python3
"""Run only the oriented-stack products.

Outputs written to SUMMARY_DIR, using the shared CACHE_FILE:
  - summary_oriented_full_stack_2x3.pdf
  - summary_oriented_full_stack_cap_profiles_1x3.pdf
  - summary_oriented_hist_ba_selected_1x3.pdf
  - summary_radio_full_stack_1x3.pdf

Selection policy for this driver:
  - no E(B-V) dust cut
  - uses SDSS photoPosPlate cross-match for PA/b/a
  - no disk/FRACDEV cut
"""

import os
import numpy as np

import tsz_stack_common as pipe


# ============================================================
# DRIVER-SPECIFIC SELECTION
# ============================================================
# User-requested split: oriented stacks should not use the dust cut.
pipe.USE_EBV_CUT = False

# Oriented stacks need PA/b/a, so keep the photo cross-match.
pipe.USE_PHOTO_SHAPE = True

# Do not apply any disk/FRACDEV cut in this driver.
pipe.USE_DISK_CUT = False

ORIENTED_PDF_NAMES = [
    "summary_oriented_full_stack_2x3.pdf",
    "summary_oriented_full_stack_cap_profiles_1x3.pdf",
    "summary_oriented_hist_ba_selected_1x3.pdf",
    "summary_radio_full_stack_1x3.pdf",
]
pipe.SUMMARY_PDF_NAMES = ORIENTED_PDF_NAMES


def main():
    os.makedirs(pipe.SUMMARY_DIR, exist_ok=True)
    print("=" * 70)
    print("ORIENTED-STACK DRIVER")
    print("  Outputs: oriented maps/CAP, selected b/a histogram, radio stack")
    print(f"  Summary path: {pipe.SUMMARY_DIR}")
    print(f"  Shared cache: {pipe.CACHE_FILE}")
    print("  Dust cut: OFF for this driver")
    print("  photoPosPlate cross-match: ON")
    print("  Disk/FRACDEV cut: OFF")
    print("=" * 70)

    h5f, base_fits_idx, radio_fits_idx = pipe.build_selection_and_cache()

    cache_stamp_valid = h5f["stamp_valid"][:]
    cache_logm = h5f["logm"][:]
    cache_fits_idx = h5f["fits_idx"][:]
    cache_in_current_selection = np.isin(cache_fits_idx, base_fits_idx)
    cache_in_radio_selection = np.isin(cache_fits_idx, radio_fits_idx)

    print("\nCache summary:")
    print(f"  total cache rows: {len(cache_fits_idx):,}")
    print(f"  valid stamps: {cache_stamp_valid.sum():,}")
    print(f"  current oriented selection rows: {cache_in_current_selection.sum():,}")
    print(f"  radio selection rows: {cache_in_radio_selection.sum():,}")

    all_bin_results = []

    for mass_lo, mass_hi in pipe.MASS_BINS:
        print("\n" + "=" * 70)
        print(f"MASS BIN: log stellar mass ({mass_lo}, {mass_hi}]")
        print("=" * 70)

        in_mass_bin = pipe._mass_bin_mask(cache_logm, mass_lo, mass_hi)
        main_bin_before_stamp = cache_in_current_selection & in_mass_bin
        main_bin_after_stamp = main_bin_before_stamp & cache_stamp_valid
        radio_bin_before_stamp = cache_in_radio_selection & in_mass_bin
        radio_bin_after_stamp = radio_bin_before_stamp & cache_stamp_valid

        print("  Oriented sample accounting inside this mass bin:")
        print(f"    before stamp-valid cut: {int(np.sum(main_bin_before_stamp)):,}")
        print(f"    rejected by stamp_valid=False: {int(np.sum(main_bin_before_stamp & ~cache_stamp_valid)):,}")
        print(f"    after stamp-valid cut: {int(np.sum(main_bin_after_stamp)):,}")

        print("  Radio sample accounting inside this mass bin:")
        print(f"    before stamp-valid cut: {int(np.sum(radio_bin_before_stamp)):,}")
        print(f"    rejected by stamp_valid=False: {int(np.sum(radio_bin_before_stamp & ~cache_stamp_valid)):,}")
        print(f"    after stamp-valid cut: {int(np.sum(radio_bin_after_stamp)):,}")

        bin_result = {"mass_lo": mass_lo, "mass_hi": mass_hi}

        if int(main_bin_after_stamp.sum()) == 0:
            bin_result["full_stack"] = {"n_success": 0}
        else:
            print("  Full oriented mass-bin stack")
            pipe.add_full_stack_results(h5f, bin_result, main_bin_after_stamp)

        if int(radio_bin_after_stamp.sum()) == 0:
            bin_result["radio_full_stack"] = {"n_success": 0}
        else:
            print("  Radio full stack")
            pipe.add_radio_stack_results(h5f, bin_result, radio_bin_after_stamp)

        all_bin_results.append(bin_result)

    print("\nWriting oriented-stack PDFs...")
    stack_norm = pipe._shared_stack_norm(pipe._collect_stack_values_from_results(all_bin_results))
    pipe.plot_summary_full_mass_stacks(all_bin_results, pipe.SUMMARY_DIR, stack_norm)
    pipe.plot_summary_sector_cap_profiles(all_bin_results, pipe.SUMMARY_DIR)
    pipe.plot_summary_radio_full_stacks(all_bin_results, pipe.SUMMARY_DIR, stack_norm)
    pipe.plot_summary_oriented_selected_ba_histograms(all_bin_results, h5f, pipe.SUMMARY_DIR)
    pipe.print_cap_significance_tables(all_bin_results)

    print("\nDone. PDFs written by this driver:")
    for pdf_name in ORIENTED_PDF_NAMES:
        print(f"  {pdf_name}")


if __name__ == "__main__":
    main()
