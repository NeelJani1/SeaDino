#!/usr/bin/env python3
"""
SeaDino External Drive Migration Utility
========================================
Assists in safely moving heavy model checkpoints, training logs, and benchmark
results to an external drive (e.g. /mnt/d/... or secondary NVMe/HDD storage),
while preserving code compatibility via automatic symlinks and environment setup.

Usage:
  python scripts/migrate_to_external_drive.py --target_dir /mnt/external_drive/SeaDino_Storage
  python scripts/migrate_to_external_drive.py --target_dir /path/to/drive --dry_run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def migrate_data(target_dir: Path, move_ckpts: bool = True, move_logs: bool = True,
                 move_results: bool = True, dry_run: bool = False):
    print("=" * 80)
    print(f"SeaDino External Drive Migration {'[DRY RUN]' if dry_run else ''}")
    print(f"Source Repo:  {REPO_ROOT}")
    print(f"Target Drive: {target_dir}")
    print("=" * 80)

    target_dir.mkdir(parents=True, exist_ok=True)
    target_ckpts = target_dir / "checkpoints"
    target_logs = target_dir / "logs"
    target_results = target_dir / "results"

    # 1. Migrate In-Repo Checkpoints
    if move_ckpts:
        target_ckpts.mkdir(parents=True, exist_ok=True)
        ckpt_dirs = [d for d in REPO_ROOT.iterdir() if d.is_dir() and d.name.startswith("stage1_") and not d.name.endswith("_logs")]
        print(f"\n[1/3] Processing {len(ckpt_dirs)} Checkpoint Directories...")
        for cdir in sorted(ckpt_dirs):
            dst = target_ckpts / cdir.name
            print(f"  • {cdir.name} -> {dst}")
            if not dry_run:
                if dst.exists():
                    print(f"    ↳ Destination already exists. Skipping move.")
                else:
                    shutil.move(str(cdir), str(dst))
                    # Leave symlink so code referencing local path never breaks
                    cdir.symlink_to(dst)
                    print(f"    ↳ Moved & linked symlink at original location.")

    # 2. Migrate Training Logs
    if move_logs:
        target_logs.mkdir(parents=True, exist_ok=True)
        log_dirs = [d for d in REPO_ROOT.iterdir() if d.is_dir() and (d.name.endswith("_logs") or d.name == "wandb")]
        print(f"\n[2/3] Processing {len(log_dirs)} Log & WandB Directories...")
        for ldir in sorted(log_dirs):
            dst = target_logs / ldir.name
            print(f"  • {ldir.name} -> {dst}")
            if not dry_run:
                if dst.exists():
                    print(f"    ↳ Destination already exists. Skipping move.")
                else:
                    shutil.move(str(ldir), str(dst))
                    ldir.symlink_to(dst)
                    print(f"    ↳ Moved & linked symlink at original location.")

    # 3. Migrate Results
    if move_results:
        target_results.mkdir(parents=True, exist_ok=True)
        results_dir = REPO_ROOT / "results"
        print(f"\n[3/3] Processing Results Directory...")
        if results_dir.exists():
            print(f"  • results/ -> {target_results}")
            if not dry_run:
                # Copy or sync results
                for f in results_dir.glob("*.csv"):
                    dest_f = target_results / f.name
                    if not dest_f.exists():
                        shutil.copy2(str(f), str(dest_f))
                        print(f"    ↳ Copied {f.name}")

    # 4. Generate Environment Config Script
    env_file = target_dir / "seadino_env.sh"
    print(f"\n[Config] Generating environment configuration script -> {env_file}")
    if not dry_run:
        with open(env_file, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("# Source this file to route SeaDino to your external storage:\n")
            f.write(f"export SEADINO_CHECKPOINTS_DIR=\"{target_ckpts.resolve()}\"\n")
            f.write(f"export SEADINO_LOGS_DIR=\"{target_logs.resolve()}\"\n")
            f.write(f"export SEADINO_RESULTS_DIR=\"{target_results.resolve()}\"\n")
            f.write("echo \"[SeaDino] External drive paths active:\"\n")
            f.write("echo \"  Checkpoints: $SEADINO_CHECKPOINTS_DIR\"\n")
            f.write("echo \"  Logs:        $SEADINO_LOGS_DIR\"\n")
            f.write("echo \"  Results:     $SEADINO_RESULTS_DIR\"\n")
        print("  ↳ Done! You can do: source " + str(env_file))

    print("\n" + "=" * 80)
    print("Migration Complete! Local symlinks ensure 100% backward compatibility.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Migrate SeaDino artifacts to an external storage drive")
    p.add_argument("--target_dir", type=str, required=True, help="Destination directory path on the external drive")
    p.add_argument("--no_checkpoints", action="store_true", help="Skip migrating checkpoint files")
    p.add_argument("--no_logs", action="store_true", help="Skip migrating log files")
    p.add_argument("--no_results", action="store_true", help="Skip copying results")
    p.add_argument("--dry_run", action="store_true", help="Simulate without copying/moving files")
    args = p.parse_args()

    migrate_data(
        target_dir=Path(args.target_dir),
        move_ckpts=not args.no_checkpoints,
        move_logs=not args.no_logs,
        move_results=not args.no_results,
        dry_run=args.dry_run
    )
