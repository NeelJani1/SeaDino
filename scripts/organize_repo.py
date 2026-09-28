#!/usr/bin/env python3
"""
SeaDino Repository Organization Utility
======================================
Cleans up the SeaDino workspace into a modular, production-grade structure:
  • data/splits/           - Spatial splits (Substrate, German Bank, Biota)
  • data/manifests/        - Benchmark masks, clean manifests, hashes
  • results/               - Master publication tables, benchmark sweeps
  • scripts/audits/        - Leakage diagnostic & spatial split generators
  • scripts/benchmarks/    - Impact comparison & probe evaluation tools
  • scripts/visualization/ - Augmentation visualizers, notebooks, figures
  • scripts/diagnostics/   - Weight drift & LR decay checkers
  • docs/                  - Investigation briefs and reports

Supports dry-run (--dry_run) and backward-compatible symlinking (--create_symlinks).
"""

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# File Mapping Plan: Source -> Destination Directory
FILE_MAPPING = {
    # Data Splits
    "substrate_depth_2_spatial_split.csv": "data/splits",
    "german_bank_2010_spatial_split.csv": "data/splits",
    "biota_spatial_split.csv": "data/splits",
    "substrate_depth_2_data_regional_split.csv": "data/splits",
    "substrate_depth_2_data_site_split.csv": "data/splits",

    # Data Manifests & Hashes
    "coralmask_test_clean.txt": "data/manifests",
    "coralmask_test_leakage_ids.txt": "data/manifests",
    "benthicnet_shard_hashes.npz": "data/manifests",
    "coralmask_test_min_dists.npz": "data/manifests",

    # Results & Publication Master Tables
    "paper_ablation_tables.csv": "results",
    "master_coral_all_epochs_full_clean.csv": "results",
    "master_coral_all_epochs_full.csv": "results",
    "master_substrate_germanbank_all_epochs_clean.csv": "results",
    "stage2_spotcheck.csv": "results",
    "eval_results_biota_original_split.csv": "results",
    "eval_results_biota_spatial_split.csv": "results",
    "eval_results_original_split.csv": "results",
    "eval_results_spatial_split.csv": "results",
    "substrate_and_german_bank_with_f1.csv": "results",
    "tau_994_vs_099_control.csv": "results",
    "Marine_With_physics_tunning_invetigation.csv": "results",
    "Marine_With_physics_tunning_invetigation_3e-4.csv": "results",
    "Marine_with_physics_check.csv": "results",

    # Audits & Leakage Resolution
    "solve_benthicnet_leakage.py": "scripts/audits",
    "solve_biota_leakage.py": "scripts/audits",
    "cross_match_shards.py": "scripts/audits",
    "coralmask_threshold_sweep.py": "scripts/audits",
    "coralmask_leakage_diagnostic.py": "scripts/audits",
    "task1_diagnose_zero_matches.py": "scripts/audits",
    "verify_dataset_independence.py": "scripts/audits",

    # Evaluation Utilities
    "compare_leakage_impact.py": "scripts/benchmarks",
    "compare_biota_impact.py": "scripts/benchmarks",
    "inspect_ckpt_norm.py": "scripts/benchmarks",

    # Visualization & Media
    "visualize_augmentations.py": "scripts/visualization",
    "embed_gallery_images.py": "scripts/visualization",
    "Visualize_augmentation.ipynb": "scripts/visualization",
    "coralmask_benthicnet_leakage_evidence.png": "scripts/visualization",

    # Diagnostics
    "check_lr_decay.py": "scripts/diagnostics",
    "weight_drift.py": "scripts/diagnostics",

    # Docs
    "Benthicnet leakage task brief.md": "docs",
}


def organize_repo(dry_run: bool = False, create_symlinks: bool = True):
    print("=" * 80)
    print(f"SeaDino Workspace Cleaner {'[DRY RUN]' if dry_run else ''}")
    print(f"Repo Root: {REPO_ROOT}")
    print("=" * 80)

    moved_count = 0
    missing_count = 0

    for filename, rel_target_dir in FILE_MAPPING.items():
        src_path = REPO_ROOT / filename
        target_dir = REPO_ROOT / rel_target_dir
        dest_path = target_dir / filename

        if not src_path.exists():
            # Check if it was already moved
            if dest_path.exists():
                continue
            missing_count += 1
            continue

        print(f"• Moving: {filename} -> {rel_target_dir}/")
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))

            if create_symlinks:
                try:
                    src_path.symlink_to(dest_path)
                    print(f"    ↳ Linked root symlink for backward compatibility.")
                except Exception as e:
                    pass

        moved_count += 1

    print("\n" + "-" * 80)
    print(f"Organization Complete! Files processed: {moved_count}, Missing/Already moved: {missing_count}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize SeaDino repo structure")
    parser.add_argument("--dry_run", action="store_true", help="Simulate without modifying files.")
    parser.add_argument("--no_symlinks", action="store_true", help="Do not create root backward-compatible symlinks.")
    args = parser.parse_args()

    organize_repo(dry_run=args.dry_run, create_symlinks=not args.no_symlinks)
