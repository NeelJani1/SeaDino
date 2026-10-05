# DeepFish Benthic Fish-Habitat Benchmark

This directory documents the integration of the **DeepFish** dataset into the SeaDino evaluation suite.

---

## 1. Overview

[DeepFish](https://alzayats.github.io/DeepFish/) (*Saleh et al., Nature Scientific Reports 2020*) is a realistic, large-scale fish-habitat dataset collected from 20 coastal marine habitats in tropical Australia (mangrove roots, seagrass beds, coral reefs, bouldered seabed).

Unlike datasets filmed in clear, aquarium-like water, DeepFish features fish in their natural environmental conditions: camouflaged, small, partially hidden, and with varying water turbidity.

* **Total Images**: 39,766 high-resolution frames (1920 × 1080)
* **Habitats**: 20 distinct marine habitats
* **Primary Task**: Binary Fish / No-Fish Classification across complex seafloor backgrounds
* **License**: CC BY 4.0

---

## 2. Quick Download from Hugging Face

The dataset is mirrored as a single 7.1 GB archive on Hugging Face ([`Alzayats/DeepFish`](https://huggingface.co/datasets/Alzayats/DeepFish)).

Run the following commands to download and extract into this directory:

```bash
# 1. Install compatible huggingface_hub
pip install "huggingface-hub<2.0,>=1.3.0"

# 2. Download the archive (resumes automatically if interrupted)
hf download Alzayats/DeepFish DeepFish.tar DeepFish.tar.sha256 --repo-type dataset --local-dir data/benchmarks/deepfish

# 3. Verify checksum and extract
cd data/benchmarks/deepfish
sha256sum -c DeepFish.tar.sha256
tar -xf DeepFish.tar
```

After extraction, the directory structure will be:
```
data/benchmarks/deepfish/
├── Classification/
│   ├── train.csv         # Official training split
│   ├── val.csv           # Official validation split
│   ├── test.csv          # Official test split
│   └── images/           # JPEG video frames organized by habitat
├── Localization/         # Point-level fish annotations
└── Segmentation/         # Dense pixel fish masks
```

---

## 3. Evaluation in SeaDino

To evaluate SeaDino foundation representations on DeepFish:

```bash
# Fast evaluation with 2,000 train frames and 1,000 test frames
python run_eval.py \
    --include_off_the_shelf \
    --datasets deepfish \
    --deepfish_max_train_samples 2000 \
    --deepfish_max_test_samples 1000 \
    --deepfish_epochs 10

# Full evaluation across all 39k frames
python run_eval.py \
    --checkpoint_dirs /path/to/checkpoints \
    --datasets deepfish \
    --deepfish_split test \
    --use_bf16
```

### Metrics Reported:
* **DeepFish Lin Acc (%)**: Linear probe classification accuracy
* **DeepFish Lin F1-Macro (%)**: Linear probe macro-averaged F1
* **DeepFish Lin Fish-F1 (%)**: Linear probe F1 on positive Fish frames
* **DeepFish kNN Acc (%)**: Cosine-similarity weighted $k$-NN accuracy
* **DeepFish kNN F1-Macro (%)**: $k$-NN macro-averaged F1
* **DeepFish kNN Fish-F1 (%)**: $k$-NN F1 on positive Fish frames
* **Normalized Confusion Matrix**: Saved to `eval_output/confusion_matrices/`
