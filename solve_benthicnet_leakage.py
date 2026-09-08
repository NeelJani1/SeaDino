#!/usr/bin/env python3
"""
BenthicNet Eval Leakage Resolution: Tasks 1, 2, and 3
------------------------------------------------------
Task 1: Diagnose the 0.0m / near-distance train/test matches in Substrate Depth 2 and German Bank 2010.
Task 2: Construct spatial block train/test splits with explicit buffer zones.
Task 3: Validate the new splits (spatial isolation, 0% leakage within buffer, class balance).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Paths
BASE_DIR = "/home/njan320"
MASTER_CSV = f"{BASE_DIR}/Neel/BenthicNet/csvs/finalized_csvs/benthicnet_labelled.csv"
SUBSTRATE_CSV = f"{BASE_DIR}/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/substrate_depth_2/substrate_depth_2_data.csv"
GERMAN_BANK_CSV = f"{BASE_DIR}/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/german_bank_2010/german_bank_2010_data.csv"

OUTPUT_SUBSTRATE_CSV = f"{BASE_DIR}/SeaDino/substrate_depth_2_spatial_split.csv"
OUTPUT_GERMAN_BANK_CSV = f"{BASE_DIR}/SeaDino/german_bank_2010_spatial_split.csv"

EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Helper: Haversine distance using sklearn BallTree
# ---------------------------------------------------------------------------
def compute_nearest_haversine(train_df, test_df):
    """
    Returns array of distances (in meters) from each test point to its nearest train point,
    and the indices of the corresponding nearest train point.
    """
    train_rad = np.radians(train_df[["latitude", "longitude"]].to_numpy())
    test_rad = np.radians(test_df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(train_rad, metric="haversine")
    dists_rad, idxs = tree.query(test_rad, k=1)
    dists_m = dists_rad.flatten() * EARTH_RADIUS_M
    return dists_m, idxs.flatten()


# ---------------------------------------------------------------------------
# Task 1: Diagnose 0.0m and Near-Distance Matches
# ---------------------------------------------------------------------------
def task1_diagnose(df_merged, benchmark_name, label_col):
    print("\n" + "=" * 80)
    print(f"TASK 1 DIAGNOSIS: {benchmark_name}")
    print("=" * 80)

    train = df_merged[df_merged["partition"].str.lower() == "train"].reset_index(drop=True)
    test = df_merged[df_merged["partition"].str.lower().isin(["test", "val"])].reset_index(drop=True)

    print(f"Total merged samples: {len(df_merged):,}")
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    dists_m, nearest_train_idx = compute_nearest_haversine(train, test)

    # Distance percentiles
    print("\nDistance percentiles (Test -> Nearest Train):")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        print(f"  {p:>3}th percentile: {np.percentile(dists_m, p):>10.2f} meters")

    # Threshold breakdown
    print("\nTest points within distance thresholds of ANY train point:")
    thresholds = [0.001, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    for t in thresholds:
        cnt = (dists_m < t).sum()
        pct = (cnt / len(dists_m)) * 100.0
        label = f"< {t}m" if t >= 1.0 else "== 0.0m (exact)"
        print(f"  {label:<18}: {cnt:>6,} / {len(dists_m):,} ({pct:>5.2f}%)")

    # Inspect 0.0m / < 1.0m matches
    zero_mask = dists_m < 1.0
    n_zero = zero_mask.sum()
    print(f"\nAnalyzing all {n_zero} test points within 1.0m of a train point:")

    same_img_count = 0
    same_site_count = 0
    same_label_count = 0

    zero_test_indices = np.where(zero_mask)[0]
    for i in zero_test_indices:
        t_row = test.iloc[i]
        tr_row = train.iloc[nearest_train_idx[i]]

        is_same_img = (t_row["image"] == tr_row["image"])
        is_same_site = (t_row["dataset"] == tr_row["dataset"] and t_row["site"] == tr_row["site"])
        is_same_lbl = (t_row[label_col] == tr_row[label_col])

        if is_same_img:
            same_img_count += 1
        if is_same_site:
            same_site_count += 1
        if is_same_lbl:
            same_label_count += 1

    print(f"  Exact same image filename (duplicate rows): {same_img_count} ({same_img_count/max(1,n_zero)*100:.1f}%)")
    print(f"  Same transect / site (interleaved tow frames): {same_site_count} ({same_site_count/max(1,n_zero)*100:.1f}%)")
    print(f"  Different site at exact same GPS coordinate:  {n_zero - same_site_count} ({(n_zero - same_site_count)/max(1,n_zero)*100:.1f}%)")
    print(f"  Same ground-truth class label:                 {same_label_count} ({same_label_count/max(1,n_zero)*100:.1f}%)")

    # Print sample pairs
    print(f"\nSample 0.0m pairs (showing up to 5):")
    for k, i in enumerate(zero_test_indices[:5]):
        t_row = test.iloc[i]
        tr_row = train.iloc[nearest_train_idx[i]]
        dist = dists_m[i]
        print(f"\n  [Pair #{k+1}] Dist = {dist:.4f}m:")
        print(f"    TEST : dataset={t_row['dataset']} | site={t_row['site']}")
        print(f"           img={t_row['image']} | lat={t_row['latitude']:.7f}, lon={t_row['longitude']:.7f} | label={t_row[label_col]}")
        print(f"    TRAIN: dataset={tr_row['dataset']} | site={tr_row['site']}")
        print(f"           img={tr_row['image']} | lat={tr_row['latitude']:.7f}, lon={tr_row['longitude']:.7f} | label={tr_row[label_col]}")

    return dists_m


# ---------------------------------------------------------------------------
# Task 2: Build Spatial Block Split with Buffer Discard
# ---------------------------------------------------------------------------
def build_spatial_block_split(df, cell_size_m=500.0, buffer_m=100.0, test_frac=0.20, seed=42, label_col="label"):
    """
    Constructs a spatial block split:
    1. Projects lat/lon onto regular grid cells of size ~cell_size_m (correcting for cosine of latitude).
    2. Randomly assigns cells to train vs test to achieve ~test_frac test images.
    3. Discards all train points within buffer_m (haversine) of ANY test point.
    """
    df = df.copy().reset_index(drop=True)

    # 1. Grid projection correcting for latitude longitude compression
    # 1 deg lat = ~111,139 m
    # 1 deg lon = ~111,139 * cos(lat) m
    lat_rad = np.radians(df["latitude"].to_numpy())
    m_per_deg_lat = 111_139.0
    m_per_deg_lon = 111_139.0 * np.cos(lat_rad)

    # Local grid indices
    df["grid_y"] = (df["latitude"] * m_per_deg_lat / cell_size_m).astype(int)
    df["grid_x"] = (df["longitude"] * m_per_deg_lon / cell_size_m).astype(int)
    df["spatial_cell"] = df["dataset"].astype(str) + "_" + df["grid_y"].astype(str) + "_" + df["grid_x"].astype(str)

    # 2. Assign cells to train / test
    rng = np.random.default_rng(seed)
    unique_cells = np.array(sorted(df["spatial_cell"].unique()))
    rng.shuffle(unique_cells)

    # Greedy allocation to achieve target test fraction while keeping whole cells intact
    total_images = len(df)
    target_test_images = int(total_images * test_frac)

    cell_counts = df["spatial_cell"].value_counts().to_dict()

    test_cells = set()
    accum_test = 0
    for cell in unique_cells:
        test_cells.add(cell)
        accum_test += cell_counts[cell]
        if accum_test >= target_test_images:
            break

    df["initial_partition"] = np.where(df["spatial_cell"].isin(test_cells), "test", "train")

    # 3. Buffer discard: drop train points within buffer_m of ANY test point
    train_subset = df[df["initial_partition"] == "train"]
    test_subset = df[df["initial_partition"] == "test"]

    print(f"  Pre-buffer: {len(train_subset):,} train in {len(unique_cells)-len(test_cells)} cells, "
          f"{len(test_subset):,} test in {len(test_cells)} cells ({len(test_subset)/total_images*100:.1f}%)")

    # BallTree on test points to query distances for all train points
    test_rad = np.radians(test_subset[["latitude", "longitude"]].to_numpy())
    train_rad = np.radians(train_subset[["latitude", "longitude"]].to_numpy())

    tree = BallTree(test_rad, metric="haversine")
    dists_rad, _ = tree.query(train_rad, k=1)
    train_to_test_m = dists_rad.flatten() * EARTH_RADIUS_M

    train_keep_mask = train_to_test_m >= buffer_m
    dropped_train_count = (~train_keep_mask).sum()

    # Assign final partitions
    train_indices = train_subset.index.to_numpy()
    kept_train_indices = set(train_indices[train_keep_mask])

    final_partition = []
    for idx in df.index:
        if idx in test_subset.index:
            final_partition.append("test")
        elif idx in kept_train_indices:
            final_partition.append("train")
        else:
            final_partition.append("dropped")

    df["partition"] = final_partition
    print(f"  Buffer filtering (buffer = {buffer_m}m): dropped {dropped_train_count:,} buffer-zone train images")
    print(f"  Final split: {len(df[df['partition']=='train']):,} Train | "
          f"{len(df[df['partition']=='test']):,} Test | "
          f"{len(df[df['partition']=='dropped']):,} Dropped")

    return df


# ---------------------------------------------------------------------------
# Task 3: Validate the New Split
# ---------------------------------------------------------------------------
def task3_validate(df_split, benchmark_name, buffer_m, label_col):
    print("\n" + "=" * 80)
    print(f"TASK 3 VALIDATION: {benchmark_name}")
    print("=" * 80)

    train = df_split[df_split["partition"] == "train"].reset_index(drop=True)
    test = df_split[df_split["partition"] == "test"].reset_index(drop=True)
    dropped = df_split[df_split["partition"] == "dropped"].reset_index(drop=True)

    print(f"Partition counts:")
    print(f"  Train:   {len(train):>6,} ({len(train)/len(df_split)*100:>5.1f}%)")
    print(f"  Test:    {len(test):>6,} ({len(test)/len(df_split)*100:>5.1f}%)")
    print(f"  Dropped: {len(dropped):>6,} ({len(dropped)/len(df_split)*100:>5.1f}%)")

    # Check 1: No cell overlap between train and test
    train_cells = set(train["spatial_cell"].unique())
    test_cells = set(test["spatial_cell"].unique())
    cell_overlap = train_cells & test_cells
    print(f"\n[Check 1] Spatial cell overlap between Train and Test: {len(cell_overlap)}")
    assert len(cell_overlap) == 0, f"FAILED: {len(cell_overlap)} overlapping cells found!"
    print("  -> PASSED: Zero cell overlap.")

    # Check 2: Minimum haversine distance between any test point and any train point
    dists_m, _ = compute_nearest_haversine(train, test)
    min_dist = dists_m.min()
    pct_within_buffer = (dists_m < buffer_m).sum() / len(dists_m) * 100.0

    print(f"\n[Check 2] Minimum test -> train distance: {min_dist:.2f} meters (target buffer: {buffer_m}m)")
    print(f"  Test points within {buffer_m}m of a train point: {(dists_m < buffer_m).sum()} ({pct_within_buffer:.2f}%)")
    assert (dists_m < buffer_m).sum() == 0, f"FAILED: {(dists_m < buffer_m).sum()} points violated buffer!"
    print("  -> PASSED: Exactly 0.0% of test points are within the buffer zone of any train point!")

    # Check 3: Distance percentiles under new split
    print("\n[Check 3] New distance distribution (Test -> Nearest Train):")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 100]:
        print(f"  {p:>3}th percentile: {np.percentile(dists_m, p):>10.2f} meters")

    # Check 4: Class distribution preservation
    print(f"\n[Check 4] Class distribution across Train and Test ({label_col}):")
    tr_dist = train[label_col].value_counts(normalize=True).sort_index()
    te_dist = test[label_col].value_counts(normalize=True).sort_index()
    all_classes = sorted(list(set(tr_dist.index) | set(te_dist.index)))

    print(f"  {'Class':<20} | {'Train Count':<12} | {'Train %':<8} | {'Test Count':<12} | {'Test %':<8}")
    print("  " + "-" * 70)
    for c in all_classes:
        tr_c = (train[label_col] == c).sum()
        te_c = (test[label_col] == c).sum()
        tr_p = tr_dist.get(c, 0.0) * 100.0
        te_p = te_dist.get(c, 0.0) * 100.0
        print(f"  {str(c):<20} | {tr_c:>12,} | {tr_p:>7.2f}% | {te_c:>12,} | {te_p:>7.2f}%")

    print("\n  -> ALL VALIDATION CHECKS PASSED.")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("BENTHICNET EVAL LEAKAGE AUDIT & SPATIAL RE-SPLIT")
    print("=" * 80)

    # 1. Load Master Coordinates Lookup
    print(f"\nStep 1: Loading coordinates from {MASTER_CSV}...")
    coords_df = pd.read_csv(
        MASTER_CSV,
        low_memory=False,
        usecols=["image", "longitude", "latitude", "dataset", "site"]
    ).dropna(subset=["longitude", "latitude"]).drop_duplicates(subset=["image"])
    print(f"  Loaded {len(coords_df):,} unique images with coordinates.")

    # 2. Process Substrate Depth 2
    print(f"\nStep 2: Processing Substrate Depth 2 ({SUBSTRATE_CSV})...")
    sub_raw = pd.read_csv(SUBSTRATE_CSV, low_memory=False)
    print(f"  Raw substrate rows: {len(sub_raw):,}")

    sub_merged = sub_raw.merge(coords_df[["image", "longitude", "latitude"]], on="image", how="inner")
    print(f"  Merged with coords: {len(sub_merged):,} rows")

    # Task 1 for Substrate Depth 2
    task1_diagnose(sub_merged, "Substrate Depth 2 (Original Official Split)", "catami_substrate")

    # Task 2 for Substrate Depth 2: cell_size = 500m, buffer = 100m, test_frac = 0.20
    print("\n" + "-" * 80)
    print("Building Spatial Block Split for Substrate Depth 2 (cell_size=500m, buffer=100m)...")
    sub_split = build_spatial_block_split(
        sub_merged,
        cell_size_m=500.0,
        buffer_m=100.0,
        test_frac=0.20,
        seed=42,
        label_col="catami_substrate"
    )

    # Task 3 for Substrate Depth 2
    task3_validate(sub_split, "Substrate Depth 2 (New Spatial Split)", buffer_m=100.0, label_col="catami_substrate")

    # Save corrected Substrate Depth 2 CSV
    # Preserving exact columns expected by run_eval.py / single_label.py
    # partition will be 'train', 'test', or 'dropped' (single_label.py will automatically load train and test)
    out_cols_sub = ["dataset", "site", "image", "url", "catami_substrate", "partition", "longitude", "latitude"]
    sub_split[out_cols_sub].to_csv(OUTPUT_SUBSTRATE_CSV, index=False)
    print(f"  Saved corrected Substrate Depth 2 CSV -> {OUTPUT_SUBSTRATE_CSV}")

    # 3. Process German Bank 2010
    print(f"\nStep 3: Processing German Bank 2010 ({GERMAN_BANK_CSV})...")
    gb_raw = pd.read_csv(GERMAN_BANK_CSV, low_memory=False)
    print(f"  Raw German Bank rows: {len(gb_raw):,}")

    gb_merged = gb_raw.merge(coords_df[["image", "longitude", "latitude"]], on="image", how="inner")
    print(f"  Merged with coords: {len(gb_merged):,} rows")

    # Task 1 for German Bank 2010
    task1_diagnose(gb_merged, "German Bank 2010 (Original Official Split)", "substrate")

    # Task 2 for German Bank: smaller dataset (3,181 rows).
    # Use cell_size = 250m, buffer = 50m to retain enough test images
    print("\n" + "-" * 80)
    print("Building Spatial Block Split for German Bank 2010 (cell_size=250m, buffer=50m)...")
    gb_split = build_spatial_block_split(
        gb_merged,
        cell_size_m=250.0,
        buffer_m=50.0,
        test_frac=0.20,
        seed=42,
        label_col="substrate"
    )

    # Task 3 for German Bank 2010
    task3_validate(gb_split, "German Bank 2010 (New Spatial Split)", buffer_m=50.0, label_col="substrate")

    # Save corrected German Bank CSV
    out_cols_gb = ["dataset", "site", "image", "url", "substrate", "partition", "longitude", "latitude"]
    gb_split[out_cols_gb].to_csv(OUTPUT_GERMAN_BANK_CSV, index=False)
    print(f"  Saved corrected German Bank CSV -> {OUTPUT_GERMAN_BANK_CSV}")

    print("\n" + "=" * 80)
    print("TASKS 1, 2, AND 3 COMPLETE!")
    print(f"Output files ready:")
    print(f"  1. {OUTPUT_SUBSTRATE_CSV}")
    print(f"  2. {OUTPUT_GERMAN_BANK_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
