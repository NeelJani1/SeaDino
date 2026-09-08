# BenthicNet Eval Leakage Investigation — Task Brief

## Context (read first)

Project: continued DINOv3 SSL pretraining on BenthicNet seafloor imagery, evaluated
downstream via k-NN probing on two labelled benchmarks: **Substrate Depth 2**
(75,537 images, 5-class) and **German Bank 2010** (3,181 images, 5-class), each with
its own official `train`/`test` partition column.

A prior investigation (outside this brief) confirmed the SSL pretraining pool
(189,101 images, in `/home/njan320/ssl_pretrain_shards`) does **not** contain any
image from either benchmark's `test` partition, at the image level or the
`dataset/site` transect level. That part is solid — do not re-litigate it.

**What's broken instead is the benchmarks' own internal train/test split.** It was
found to leak in two compounding ways, discovered in this order:

1. **Frame-level leakage within a transect.** E.g. Substrate Depth 2, site
   `20191020_Broughton_Island/20191020_T001`: test frame at `21:32:18`, then train
   frames at `21:33:08` through `21:38:06` — all one continuous ~6-minute tow,
   interleaved train/test by what looks like stratified-by-label sampling with no
   spatial/temporal awareness.
2. **Cross-transect spatial leakage**, confirmed even after grouping by
   `dataset/site` (transect ID) as the split unit: **65.0% of test points have a
   train point within 50m** (haversine distance, properly correcting for
   longitude compression at ~-32.5° latitude). The single closest train-test pair
   is **0.0m apart** — exact coordinate match, uninvestigated as to why (duplicate
   row? stationary camera producing near-identical frames? — see Task 1).

**Net effect:** every k-NN accuracy number produced so far for these two
benchmarks (baseline off-the-shelf DINOv3 included) may be partly measuring
near-duplicate-frame retrieval within already-seen ground, not genuine
generalization to unseen seafloor. This needs to be fixed and re-measured before
any of those numbers are treated as reliable.

## Key files / paths

- Main labelled CSV: `/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/benthicnet_labelled.csv`
- Substrate Depth 2 CSV: `/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/substrate_depth_2/substrate_depth_2_data.csv`
- German Bank 2010 CSV: `/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/german_bank_2010/german_bank_2010_data.csv`
- Both CSVs have `longitude`/`latitude` columns (confirm exact column names on load —
  they may need pulling from the main labelled CSV via an `image`-column join, as
  the probe CSVs may not carry coordinates directly).
- `run_eval.py` — the eval harness (source not yet reviewed by this brief's author;
  Task 4 depends on finding/reading it first).
- Existing checkpoints to re-evaluate once the split is fixed: at minimum
  off-the-shelf DINOv3 (`facebook/dinov3-vits16-pretrain-lvd1689m`, zero-shot) and
  one trained arm, e.g. `/home/njan320/stage1_marine_with_physics/benthic-ssl-epoch=03-ssl_loss=12.79.ckpt`.

## Task 1 — Diagnose the 0.0m matches (fast, do first)

Find out what the exact-coordinate train/test pairs actually are before designing
the fix around them.

```python
import pandas as pd, numpy as np
from sklearn.neighbors import BallTree

df = pd.read_csv("/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/one_hots/substrate_depth_2/substrate_depth_2_data.csv", low_memory=False)
df_master = pd.read_csv("/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/benthicnet_labelled.csv", low_memory=False)
coords_lookup = df_master[["image", "longitude", "latitude"]].dropna().drop_duplicates("image")
df = df.merge(coords_lookup, on="image", how="inner")

train = df[df["partition"] == "train"].reset_index(drop=True)
test  = df[df["partition"] == "test"].reset_index(drop=True)

train_rad = np.radians(train[["latitude", "longitude"]].to_numpy())
test_rad  = np.radians(test[["latitude", "longitude"]].to_numpy())
tree = BallTree(train_rad, metric="haversine")
dists_rad, idx = tree.query(test_rad, k=1)
dists_m = dists_rad.flatten() * 6_371_000

zero_mask = dists_m < 1.0
print(f"{zero_mask.sum()} test points within 1m of a train point")
for i in np.where(zero_mask)[0][:10]:
    j = idx[i][0]
    print("TEST :", test.iloc[i]["image"], test.iloc[i][["longitude","latitude"]].values)
    print("TRAIN:", train.iloc[j]["image"], train.iloc[j][["longitude","latitude"]].values)
    print()
```

**Report back:** are these duplicate rows (same/near-identical filename or
image), or distinct filenames at identical GPS (e.g. a stationary or slow-drift
camera producing many frames at ~one spot)? This affects whether project-wide
deduplication is also needed, separate from the split fix.

## Task 2 — Build a genuine spatial block split with buffer

Grid-based splitting alone won't fully solve this (boundary points between
adjacent cells will still be close). Use grid blocking **plus** an explicit
buffer discard. Apply to **both** Substrate Depth 2 and German Bank — German
Bank is smaller (3,181 images, 14 test "sites" originally) so tune cell size
accordingly; don't just copy Substrate Depth 2's parameters blindly.

```python
import pandas as pd, numpy as np
from sklearn.neighbors import BallTree

def build_spatial_split(df, cell_size_m=500, buffer_m=100, test_frac=0.2, seed=42):
    df = df.copy()
    cell_deg = cell_size_m / 111_000
    df["grid_lon"] = (df["longitude"] // cell_deg).astype(int)
    df["grid_lat"] = (df["latitude"] // cell_deg).astype(int)
    df["grid_cell"] = df["grid_lon"].astype(str) + "_" + df["grid_lat"].astype(str)

    rng = np.random.default_rng(seed)
    cells = df["grid_cell"].unique()
    rng.shuffle(cells)
    n_test = int(len(cells) * test_frac)
    test_cells = set(cells[:n_test])
    df["partition_v2"] = np.where(df["grid_cell"].isin(test_cells), "test", "train")

    # buffer discard: drop train points within buffer_m of ANY test point
    train_df = df[df["partition_v2"] == "train"]
    test_df  = df[df["partition_v2"] == "test"]
    train_rad = np.radians(train_df[["latitude", "longitude"]].to_numpy())
    test_rad  = np.radians(test_df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(test_rad, metric="haversine")
    dists_rad, _ = tree.query(train_rad, k=1)
    dists_m = dists_rad.flatten() * 6_371_000
    keep_train_mask = dists_m >= buffer_m

    kept_train_idx = train_df.index[keep_train_mask]
    final_idx = list(kept_train_idx) + list(test_df.index)
    df["partition_v2_final"] = np.where(
        df.index.isin(test_df.index), "test",
        np.where(df.index.isin(kept_train_idx), "train", "dropped")
    )
    return df

# run for substrate depth 2 and german bank separately, save each result
```

**Done when:** re-running Task 1's haversine-distance check against
`partition_v2_final` shows **0%** (or as close to 0 as achievable without
discarding excessive train data) of test points within `buffer_m` of a train
point. Report the final train/test counts after buffer discard — expect
meaningful shrinkage of the train pool, that's the correct trade-off here.

## Task 3 — Validate before trusting

- Confirm no `grid_cell` appears in both partitions (sanity check on the
  grouping itself).
- Re-run the haversine nearest-neighbor check from Task 1 against the new split
  and confirm the target buffer distance is respected.
- Do this for **both** benchmarks independently; German Bank's much smaller
  test-site count (14 originally) means it may need a larger `test_frac` or
  smaller `cell_size_m` to retain enough test images to be statistically usable
  — check final test-set size isn't too small to trust (e.g. don't let it drop
  below a few hundred images without flagging it).

## Task 4 — Quantify the impact

Once both corrected splits exist and pass Task 3:

1. Locate and read `run_eval.py` — confirm how it currently loads
   train/test partitions (does it read the `partition` column directly from the
   CSV, or is there a hardcoded path/logic that needs pointing at the new
   split?). Also check whether it applies each checkpoint's own
   `benthic_norm` setting (stored in `checkpoint["config"]["benthic_norm"]`) or
   a single fixed normalization regardless of which checkpoint is scored —
   report this either way, it's an open question from the broader project.
2. Re-run eval for **off-the-shelf DINOv3** and **one trained checkpoint**
   (`stage1_marine_with_physics` epoch 3, path above) on both the original split
   and the new corrected split, same benchmarks (Substrate Depth 2, German Bank).
3. Report the deltas. A large drop and/or a shrinking gap between off-the-shelf
   and the trained arm would confirm how much the original leakage was inflating
   numbers. A small move would suggest the leak's practical effect was smaller
   than the 65%/50m figure implies — also useful to know either way.

## Model routing suggestion

- Tasks 1–3 are straightforward, deterministic pandas/geometry scripts — a
  fast/cheap model is fine, but double-check the haversine math and column
  names are correct since a subtly wrong distance metric here would quietly
  reintroduce the exact bug the prior investigation caught (an earlier version
  of this check used flat degree-to-meter conversion and overstated distances
  by ~19% at this latitude — verify whichever model does this doesn't repeat
  that mistake).
- Task 4 (reading and possibly modifying `run_eval.py`, interpreting before/after
  deltas) is higher-stakes and more ambiguous — use your strongest available
  reasoning model for it, and have it show its work (print statements, not just
  a final table) so the deltas are checkable rather than taken on faith.

## Do NOT touch / re-litigate

- The SSL-pretraining-pool-vs-test-set leakage check (image-level and
  transect-level) is already confirmed clean — no need to redo it.
- Don't conflate this with the separate, still-open LR-sweep/masking-ablation
  questions from the broader project (a 3e-4 midpoint training run, a
  masking-on/off minimal-pair rerun, a `5e-4` run's log path for NaN/Inf
  gradient warnings, and a `.layer.`-name-matching bug in `get_vit_lr_decay_rate`
  that makes `--lr_decay_rate` a no-op for this HF model). Those are real and
  still queued, but they're downstream of this leakage question and shouldn't
  be worked in parallel until the split is fixed — re-running LR/masking sweeps
  against a leaky benchmark just produces more numbers that need redoing later.