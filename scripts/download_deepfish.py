#!/usr/bin/env python3
"""Downloads the 7.1GB DeepFish dataset from Hugging Face directly to G-Drive."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_suite.config import GDRIVE_CANDIDATE_PATHS, DATA_DIR


def default_dest():
    for g in GDRIVE_CANDIDATE_PATHS:
        try:
            if g.is_dir():
                return g / "benchmarks" / "deepfish"
        except (PermissionError, OSError):
            continue
    return DATA_DIR / "benchmarks" / "deepfish"


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    print(f"Verifying SHA-256 for {file_path.name}...")
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            sha.update(chunk)
    digest = sha.hexdigest().strip()
    return digest.lower() == expected_hash.strip().lower()


def main():
    p = argparse.ArgumentParser(description="Download DeepFish directly to G-Drive")
    p.add_argument("--dest_dir", type=str, default=str(default_dest()),
                   help="Destination directory on G drive.")
    args = p.parse_args()

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "DeepFish.tar"

    print(f"Destination: {dest.resolve()}")
    if not (dest / "Classification").is_dir():
        if not tar_path.exists():
            print("Downloading Alzayats/DeepFish from Hugging Face Hub...")
            cmd = [
                "hf", "download", "Alzayats/DeepFish", "DeepFish.tar", "DeepFish.tar.sha256",
                "--repo-type", "dataset", "--local-dir", str(dest)
            ]
            subprocess.run(cmd, check=True)

        sha_file = dest / "DeepFish.tar.sha256"
        if sha_file.exists():
            with open(sha_file, "r") as f:
                expected_sha = f.read().split()[0]
            if not verify_sha256(tar_path, expected_sha):
                print("Error: SHA256 checksum mismatch!")
                return 1

        print("Extracting DeepFish.tar...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dest)
        print("Extraction complete.")
    else:
        print("DeepFish Classification directory already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
