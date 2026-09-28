# SeaDino SSL Pretraining Data Specifications

This directory contains metadata, shard specifications, and deduplication records for the SeaDino self-supervised pretraining pipeline.

---

## 1. Dataset Overview

SeaDino is pretrained on **BenthicNet**, an extensive collection of seafloor imagery aggregated from autonomous underwater vehicles (AUVs), remotely operated vehicles (ROVs), towed cameras, and drop frames worldwide.

* **Image Modality**: Color benthic seafloor photographs.
* **Storage Format**: Sharded [WebDataset](https://github.com/webdataset/webdataset) `.tar` archives.
* **Shard Size**: Typically ~1,000 to 2,000 images per tar file (`benthicnet_shard_000000.tar`, etc.).
* **Image Content**: Each sample contains a JPEG image (`.jpg` or `.png`) and corresponding metadata JSON (`.json`) with camera platform, location, and survey identifiers.

---

## 2. Ingestion & Pretraining

Pretraining is executed via `ssl_train.py` using high-throughput streaming through PyTorch:

```bash
python ssl_train.py \
    --use_webdataset \
    --shard_dir /path/to/benthicnet_shards \
    --output_dir ./checkpoints/seadino_vits16_stage1 \
    --model_id facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --marine_aug \
    --benthic_norm \
    --use_bf16 \
    --epochs 15
```

---

## 3. Decontamination & Perceptual Hashing

To strictly prevent data leakage into downstream evaluation benchmarks (such as CoralMask), perceptual hashes (pHash) of pretraining shards were computed and cross-matched:

* `benthicnet_shard_hashes.npz`: Compact array of 64-bit perceptual hashes for all 189,101 pretraining images across the shard corpus (gitignored due to size; preserved on external storage vault).
* Cross-matching against CoralMask test images revealed exactly 7 leaked frames ($d \le 4$ Hamming distance), which are decontaminated in downstream evaluation via `data/benchmarks/coralmask/coralmask_test_clean.txt`.
