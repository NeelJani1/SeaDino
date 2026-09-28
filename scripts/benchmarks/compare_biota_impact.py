#!/usr/bin/env python3
"""
Task 4: Quantify the Impact of Spatial Leakage for Biota
--------------------------------------------------------
Runs run_eval.py on both:
  1. The Original Leaky Split
  2. The New Spatially Isolated Split (with 250m/50m buffer)

Then computes and displays the exact before/after deltas for:
  - Off-the-shelf DINOv3 (zero-shot foundation baseline)
  - Trained SeaDino Checkpoint
"""

import os
import sys
from pathlib import Path
import subprocess
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_suite.config import DATA_DIR, RESULTS_DIR, _find_file, DEFAULT_PATHS

ORIGINAL_BIOTA_CSV = os.getenv(
    "ORIGINAL_BIOTA_CSV",
    "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/benthicnet_nn.csv"
)
SPATIAL_BIOTA_CSV = _find_file("biota_spatial_split.csv", ["splits", ""])
DEFAULT_CKPT = os.getenv(
    "DEFAULT_CKPT",
    str(REPO_ROOT / "stage1_marine_with_physics" / "benthic-ssl-epoch=03-ssl_loss=12.79.ckpt")
)
RUN_EVAL_SCRIPT = str(REPO_ROOT / "run_eval.py")


def run_cmd(cmd):
    print("\n" + "=" * 80)
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Command exited with code {ret.returncode}")
        sys.exit(ret.returncode)


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    print(f"Target Checkpoint: {ckpt_path}")

    # Check if spatial splits exist
    if not os.path.exists(SPATIAL_BIOTA_CSV):
        print(f"Spatial split not found! Please run solve_biota_leakage.py first.")
        sys.exit(1)

    out_orig = str(RESULTS_DIR / "eval_results_biota_original_split.csv")
    out_spatial = str(RESULTS_DIR / "eval_results_biota_spatial_split.csv")

    # 1. Evaluate on Original Split
    print("\n>>> STEP 1: Evaluating on ORIGINAL Split...")
    run_cmd([
        sys.executable, RUN_EVAL_SCRIPT,
        "--include_off_the_shelf",
        "--checkpoints", ckpt_path,
        "--datasets", "biota",
        "--biota_csv", ORIGINAL_BIOTA_CSV,
        "--output_csv", out_orig,
    ])

    # 2. Evaluate on Spatial Split
    print("\n>>> STEP 2: Evaluating on SPATIAL Split (Buffer Protected)...")
    run_cmd([
        sys.executable, RUN_EVAL_SCRIPT,
        "--include_off_the_shelf",
        "--checkpoints", ckpt_path,
        "--datasets", "biota",
        "--biota_csv", SPATIAL_BIOTA_CSV,
        "--output_csv", out_spatial,
    ])

    # 3. Compute and Print Deltas
    print("\n" + "=" * 90)
    print("TASK 4 IMPACT QUANTIFICATION: ORIGINAL VS SPATIAL SPLIT (BIOTA)")
    print("=" * 90)

    df_orig = pd.read_csv(out_orig)
    df_spatial = pd.read_csv(out_spatial)

    print("\n--- ORIGINAL SPLIT (Leaky) ---")
    print(df_orig.to_string(index=False))

    print("\n--- SPATIAL SPLIT (Clean Buffer Protected) ---")
    print(df_spatial.to_string(index=False))

    print("\n" + "=" * 90)
    print("DELTAS (Spatial Split - Original Split)")
    print("=" * 90)

    def parse_pct(val):
        if pd.isna(val) or val == "-":
            return None
        return float(str(val).replace("%", "").strip())

    metrics = [
        ("Biota Lin mAP (%)", "Biota Linear mAP"),
        ("Biota Lin F1-Macro (%)", "Biota Linear Macro-F1"),
        ("Biota kNN mAP (%)", "Biota kNN mAP"),
    ]

    for idx in range(len(df_orig)):
        model_name = df_orig.iloc[idx]["Model / Label"]
        print(f"\nModel: {model_name}")
        for col, label in metrics:
            if col in df_orig.columns and col in df_spatial.columns:
                v_orig = parse_pct(df_orig.iloc[idx][col])
                v_spat = parse_pct(df_spatial.iloc[idx][col])
                if v_orig is not None and v_spat is not None:
                    delta = v_spat - v_orig
                    sign = "+" if delta >= 0 else ""
                    print(f"  {label:<24}: {v_orig:>6.2f}% -> {v_spat:>6.2f}%  (Delta: {sign}{delta:>6.2f}%)")

if __name__ == "__main__":
    main()
