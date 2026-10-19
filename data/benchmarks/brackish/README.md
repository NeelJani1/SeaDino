# Brackish Underwater Marine Fauna Benchmark

This directory documents the integration of the **Brackish** dataset into the SeaDino evaluation suite.

---

## 1. Overview

The [Brackish Dataset](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset) (*Pedersen et al., IEEE CVPR Workshops 2019*) is an underwater visual dataset recorded in turbid Limfjorden coastal waters in northern Denmark.

Unlike clear tropical waters, Brackish features high particulate backscatter, green/brown organic water attenuation, low light, and natural camouflaged benthic fauna across 6 primary taxonomic categories:
* `crab`
* `fish`
* `jellyfish`
* `shrimp`
* `small_fish`
* `starfish`

---

## 2. Quick Download to G-Drive

To download the Brackish dataset directly to your external G-Drive storage (`/mnt/g/data/benchmarks/brackish`):

```bash
python scripts/download_brackish.py
```

Expected directory layout:
```
data/benchmarks/brackish/
├── train/
│   ├── crab/
│   ├── fish/
│   ├── jellyfish/
│   ├── shrimp/
│   ├── small_fish/
│   └── starfish/
└── test/
    ├── crab/
    ...
```

---

## 3. Evaluation in SeaDino

To evaluate SeaDino foundation representations on Brackish:

```bash
# Fast evaluation with 2,000 train frames and 1,000 test frames
python run_eval.py \
    --include_off_the_shelf \
    --datasets brackish \
    --brackish_max_train_samples 2000 \
    --brackish_max_test_samples 1000 \
    --brackish_epochs 10

# Multi-benchmark evaluation with DeepFish and FathomNet
python run_eval.py \
    --datasets deepfish fathomnet brackish \
    --include_off_the_shelf
```

### Metrics Reported:
* **Brackish Lin Acc (%)**: Linear probe classification accuracy across 6 classes
* **Brackish Lin F1-Macro (%)**: Macro-averaged F1 score
* **Brackish kNN Acc (%)**: Cosine-similarity weighted $k$-NN accuracy ($k=20$)
* **Brackish kNN F1-Macro (%)**: $k$-NN macro-averaged F1 score
* **Normalized Confusion Matrix**: Plotted and saved to `eval_output/confusion_matrices/`
