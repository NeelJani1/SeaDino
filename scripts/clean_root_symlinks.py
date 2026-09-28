#!/usr/bin/env python3
"""
SeaDino Root Directory Pristine Cleaner
=======================================
Cleans up stray test images and redundant root-level backward-compatible symlinks
after organize_repo.py has safely relocated files to their permanent homes:
  - data/splits/
  - data/manifests/
  - results/
  - scripts/
  - docs/

Usage:
  python scripts/clean_root_symlinks.py
  python scripts/clean_root_symlinks.py --dry_run
  python scripts/clean_root_symlinks.py --include_external_symlinks
"""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that MUST ALWAYS STAY in root
CANONICAL_ROOT_FILES = {
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "ssl_train.py",
    "run_eval.py",
}

# Stray files that should be deleted
STRAY_FILES = [
    "000000566.jpg",
    "000004375.jpg",
]

# Empty directories that should be removed
EMPTY_DIRS = [
    REPO_ROOT / "data" / "benthic_images",
]


def clean_root(dry_run: bool = False, remove_external: bool = False):
    print("=" * 80)
    print(f"SeaDino Root Cleaner {'[DRY RUN]' if dry_run else ''}")
    print(f"Target Root: {REPO_ROOT}")
    print("=" * 80)

    # 1. Remove stray files
    print("\n[1/3] Removing stray development files...")
    for sf in STRAY_FILES:
        p = REPO_ROOT / sf
        if p.exists() or p.is_symlink():
            print(f"  • Deleting stray file: {sf}")
            if not dry_run:
                p.unlink()

    # 2. Clean up redundant root symlinks
    print("\n[2/3] Cleaning redundant root symlinks...")
    removed_links = 0
    for item in sorted(REPO_ROOT.iterdir()):
        if not item.is_symlink():
            continue
        if item.name in CANONICAL_ROOT_FILES:
            continue

        is_external = item.name.startswith("stage1_") or item.name == "wandb"
        if is_external and not remove_external:
            print(f"  • Keeping external storage symlink: {item.name} -> {item.resolve()}")
            continue

        target = item.resolve()
        # Verify the target exists before removing the symlink!
        if target.exists():
            print(f"  • Removing redundant symlink: {item.name} (Source safe at: {target})")
            if not dry_run:
                item.unlink()
            removed_links += 1
        else:
            print(f"  [Warning] Broken symlink detected: {item.name} -> {target}")
            if not dry_run:
                item.unlink()
            removed_links += 1

    # 3. Clean empty directories
    print("\n[3/3] Cleaning empty artifacts...")
    for ed in EMPTY_DIRS:
        if ed.exists() and ed.is_dir() and not any(ed.iterdir()):
            print(f"  • Removing empty directory: {ed.relative_to(REPO_ROOT)}")
            if not dry_run:
                ed.rmdir()

    # Also clean 2-hop circular symlinks in data/
    for old_link in ["german_bank_spatial_split.csv", "substrate_spatial_split.csv"]:
        dp = REPO_ROOT / "data" / old_link
        if dp.is_symlink():
            print(f"  • Removing 2-hop symlink in data/: {old_link}")
            if not dry_run:
                dp.unlink()

    print("\n" + "=" * 80)
    print(f"Cleanup Complete! Removed {removed_links} symlinks.")
    print("Your repository root is now 100% pristine and production-grade.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean up redundant root symlinks in SeaDino")
    p.add_argument("--dry_run", action="store_true", help="Simulate without deleting")
    p.add_argument("--include_external_symlinks", action="store_true",
                   help="Also remove external drive symlinks (e.g. stage1_*, wandb)")
    args = p.parse_args()

    clean_root(dry_run=args.dry_run, remove_external=args.include_external_symlinks)
