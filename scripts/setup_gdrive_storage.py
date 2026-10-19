#!/usr/bin/env python3
"""Initializes external G-Drive storage layout for SeaDino datasets, checkpoints, and results."""

import argparse
import os
from pathlib import Path
import shutil
import sys

# Candidate mount points for G-drive (lowercase and uppercase for WSL compatibility)
DEFAULT_CANDIDATES = [
    Path("/mnt/g"),
    Path("/mnt/G"),
]

BENCHMARK_FOLDERS = [
    "deepfish", "fathomnet", "brackish", "coralmask", "substrate", "german_bank", "biota"
]


def find_accessible_gdrive(custom_path=None) -> Path | None:
    if custom_path:
        p = Path(custom_path)
        try:
            if p.is_dir():
                return p.resolve()
        except (OSError, PermissionError):
            pass

    env_mount = os.getenv("GDRIVE_ROOT") or os.getenv("GDRIVE_DIR")
    if env_mount:
        p = Path(env_mount)
        try:
            if p.is_dir():
                return p.resolve()
        except (OSError, PermissionError):
            pass

    for cand in DEFAULT_CANDIDATES:
        try:
            if cand.is_dir():
                return cand.resolve()
        except (OSError, PermissionError):
            continue
    return None


def setup(custom_root=None):
    gdrive_root = find_accessible_gdrive(custom_root)

    if gdrive_root is None:
        print("\n" + "=" * 70)
        print("❌ ERROR: External G-Drive mount is not accessible in WSL.")
        print("=" * 70)
        print("In WSL2, external/removable USB drives (like G:) are not always")
        print("auto-mounted by default, or an old mount point may be stale.")
        print("\nTo mount Windows drive G: into WSL, run the following in your terminal:\n")
        print("  sudo umount /mnt/g 2>/dev/null")
        print("  sudo mkdir -p /mnt/g")
        print("  sudo mount -t drvfs G: /mnt/g")
        print("\nThen re-run this script:")
        print("  python scripts/setup_gdrive_storage.py\n")
        print("=" * 70)
        return 1

    target_data = gdrive_root / "data"
    print(f"✅ Found accessible G-Drive at: {gdrive_root}")
    print(f"Initializing SeaDino storage in {target_data}...")
    try:
        target_data.mkdir(parents=True, exist_ok=True)
        (gdrive_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (gdrive_root / "results").mkdir(parents=True, exist_ok=True)
        (target_data / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)

        for b in BENCHMARK_FOLDERS:
            (target_data / "benchmarks" / b).mkdir(parents=True, exist_ok=True)

        # Copy local spatial splits to G drive if present
        repo_data = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
        if repo_data.is_dir():
            for csv_file in repo_data.glob("**/*.csv"):
                rel = csv_file.relative_to(repo_data)
                dst = target_data / "benchmarks" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_size == 0:
                    try:
                        shutil.copyfile(csv_file, dst)
                        print(f"  Copied split: {rel} -> G-Drive")
                    except Exception:
                        try:
                            with open(csv_file, "rb") as f_in, open(dst, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                            print(f"  Copied split: {rel} -> G-Drive")
                        except Exception as ce:
                            print(f"  [Warning: Could not copy {rel}]: {ce}")

        env_sh = Path(__file__).resolve().parent.parent / "env_gdrive.sh"
        with open(env_sh, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f'export SEADINO_DATA_DIR="{target_data}"\n')
            f.write(f'export SEADINO_CHECKPOINTS_DIR="{gdrive_root / "checkpoints"}"\n')
            f.write(f'export SEADINO_RESULTS_DIR="{gdrive_root / "results"}"\n')
            f.write(f'export HF_HOME="{target_data / ".cache" / "huggingface"}"\n')
            f.write(f'export PYTHONPATH="{Path(__file__).resolve().parent.parent}:${{PYTHONPATH}}"\n')
            f.write('echo "[SeaDino] G-Drive storage active: SEADINO_DATA_DIR=$SEADINO_DATA_DIR"\n')
        try:
            os.chmod(env_sh, 0o755)
        except Exception:
            pass
        print(f"\n🎉 Successfully initialized G-Drive storage layout!")
        print(f"Environment setup script created at: {env_sh}")
        print("To activate: source env_gdrive.sh")
        return 0
    except (PermissionError, OSError) as pe:
        print(f"\n❌ Permission error writing to G drive: {pe}")
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Initialize external G-Drive storage layout")
    p.add_argument("--gdrive_root", type=str, default=None, help="Custom mount path for G-Drive")
    args = p.parse_args()
    sys.exit(setup(args.gdrive_root))
