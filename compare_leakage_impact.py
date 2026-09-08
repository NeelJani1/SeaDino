#!/usr/bin/env python3
"""
Task 4: Quantify the Impact of Spatial Leakage
----------------------------------------------
Runs run_eval.py on both:
  1. The Original Leaky Split
  2. The New Spatially Isolated Split (with 100m/50m buffer)

Then computes and displays the exact before/after deltas for:
  - Off-the-shelf DINOv3 (zero-shot foundation baseline)
  - Trained SeaDino Checkpoint (e.g. stage1_marine_with_physics_lr_1e-4_no_pixel/benthic-ssl-best.ckpt)
"""

import os
import sys
import subprocess
import pandas as pd

ORIGINAL_SUBSTRATE_CSV = "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/substrate_depth_2/substrate_depth_2_data.csv"
ORIGINAL_GERMAN_BANK_CSV = "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/german_bank_2010/german_bank_2010_data.csv"

SPATIAL_SUBSTRATE_CSV = "/home/njan320/SeaDino/substrate_depth_2_spatial_split.csv"
SPATIAL_GERMAN_BANK_CSV = "/home/njan320/SeaDino/german_bank_2010_spatial_split.csv"

DEFAULT_CKPT = "/home/njan320/stage1_marine_with_physics/benthic-ssl-epoch=03-ssl_loss=12.79.ckpt"


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
    if not os.path.exists(SPATIAL_SUBSTRATE_CSV) or not os.path.exists(SPATIAL_GERMAN_BANK_CSV):
        print(f"Spatial splits not found! Please run solve_benthicnet_leakage.py first.")
        sys.exit(1)

    out_orig = "eval_results_original_split.csv"
    out_spatial = "eval_results_spatial_split.csv"

    # 1. Evaluate on Original Split
    print("\n>>> STEP 1: Evaluating on ORIGINAL Split...")
    run_cmd([
        sys.executable, "run_eval.py",
        "--include_off_the_shelf",
        "--checkpoints", ckpt_path,
        "--datasets", "substrate", "german_bank",
        "--substrate_csv", ORIGINAL_SUBSTRATE_CSV,
        "--german_bank_csv", ORIGINAL_GERMAN_BANK_CSV,
        "--output_csv", out_orig,
    ])

    # 2. Evaluate on Spatial Split
    print("\n>>> STEP 2: Evaluating on SPATIAL Split (Buffer Protected)...")
    run_cmd([
        sys.executable, "run_eval.py",
        "--include_off_the_shelf",
        "--checkpoints", ckpt_path,
        "--datasets", "substrate", "german_bank",
        "--substrate_csv", SPATIAL_SUBSTRATE_CSV,
        "--german_bank_csv", SPATIAL_GERMAN_BANK_CSV,
        "--output_csv", out_spatial,
    ])

    # 3. Compute and Print Deltas
    print("\n" + "=" * 90)
    print("TASK 4 IMPACT QUANTIFICATION: ORIGINAL VS SPATIAL SPLIT")
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
        ("Substrate Acc (%)", "Substrate Accuracy"),
        ("Substrate F1-Macro (%)", "Substrate Macro-F1"),
        ("German Bank Acc (%)", "German Bank Accuracy"),
        ("German Bank F1-Macro (%)", "German Bank Macro-F1"),
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
