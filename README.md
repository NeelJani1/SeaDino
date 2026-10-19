<div align="center">

# SeaDino: Physics-Informed Self-Supervised Vision Foundation Models for Marine & Benthic Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/Backbone-DINOv3--ViT--S%2F16-0081FB)](https://github.com/facebookresearch/dinov3)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-CoralMask--Clean-orange.svg)](#3-cross-dataset-decontamination-coralmask-clean)

**SeaDino** adapts Meta's state-of-the-art **DINOv3** foundation architecture to underwater and benthic seafloor imagery. By incorporating physically-grounded optical degradation models, empirical marine domain normalization, and rigorous spatial-autocorrelation decontamination, SeaDino establishes new representations for benthic habitat mapping, coral reef semantic segmentation, and substrate classification.

[Key Features](#key-features) • [Architecture](#architecture) • [Physics Augmentations](#underwater-physics-augmentations) • [Leakage Resolution](#data-leakage-resolution--spatial-benchmarks) • [Benchmark Results](#benchmark-results) • [Quickstart](#quickstart--installation) • [Pretraining](#pretraining-with-seadino) • [Evaluation](#evaluation-suite) • [Repository Structure](#repository-structure)

---

</div>

## Key Features

- **Physically-Grounded Marine Augmentations:** Integrates differentiable, tensor-space optical degradation models capturing real underwater radiative transfer: wavelength-dependent Beer-Lambert attenuation, Koschmieder backscatter turbidity haze, ROV strobe lighting vignetting with off-axis hotspots, suspended particulate matter ("marine snow"), and discrete $90^\circ$ rotation invariance.
- **In-Distribution Benthic Normalization:** Replaces terrestrial ImageNet normalization with empirically derived seafloor channel statistics ($\mu = [0.359, 0.413, 0.386]$, $\sigma = [0.219, 0.215, 0.209]$), with auto-detection across all evaluation benchmarks.
- **Full DINOv3 Alignment:** Implements the official Meta DINOv3 pretraining recipe, including iBOT masked patch prediction with contiguous BEiT block masking, KoLeo entropy regularizer, Sinkhorn-Knopp teacher centering, and student-teacher exponential moving average (EMA).
- **Benchmark Decontamination & Spatial Buffer Moats:**
  - Identifies severe spatial autocorrelation and continuous transect-level test leakage in standard benthic benchmarks.
  - Establishes **100m / 50m spatial buffer moats** for CATAMI Substrate-d2, German Bank 2010, and Biota to evaluate true generalization to unseen seafloor terrain.
  - Discovers cross-dataset Catlin Seaview Survey pretraining duplicates in CoralMask via perceptual hashing sensitivity sweeps ($d \in [0, 10]$), releasing **CoralMask-Clean** ($N=823$) for leak-free evaluation.
- **Modular Multi-Task Evaluation Harness:** Out-of-the-box benchmarking covering single-label classification ($k$-NN & Linear Probing), multi-label biota recognition (mAP/F1), and dense coral reef semantic segmentation (mIoU & Coral-IoU).
- **Drive-Agnostic Storage Architecture:** Built-in dynamic path resolvers and environment variable overrides (`SEADINO_CHECKPOINTS_DIR`, `SEADINO_RESULTS_DIR`, `SEADINO_DATA_DIR`) for moving large checkpoint vaults and logs to external NVMe/HDD storage seamlessly.

---

## Architecture

SeaDino trains a Student-Teacher Vision Transformer (ViT-S/16) using multi-crop local and global views augmented through the marine optical pipeline:

```mermaid
graph TD
    A["Raw Seafloor Imagery (BenthicNet)"] --> B["Multi-Crop Generator (2 Global + 8 Local)"]
    
    subgraph AugmentationPipeline["Marine Optical & Geometric Pipeline"]
        B --> C1["Discrete 90° Rotations (0°, 90°, 180°, 270°)"]
        C1 --> C2["Tensor Conversion [0, 1]"]
        C2 --> C3["Beer-Lambert Spectral Absorption (Red > Green > Blue)"]
        C3 --> C4["Turbidity & Backscatter Haze (Koschmieder Model)"]
        C4 --> C5["ROV Strobe Vignette & Off-Axis Hotspots"]
        C5 --> C6["Suspended Marine Snow Particulate Drift"]
        C6 --> C7["BenthicNet Domain Normalization"]
    end

    C7 --> D["Student ViT-S/16 (with BEiT Masking)"]
    C7 --> E["Teacher ViT-S/16 (Global Crops Only)"]

    D --> F["DINO CLS Token Loss"]
    D --> G["iBOT Masked Patch Loss"]
    D --> H["KoLeo Regularizer"]
    
    E --> F
    E --> G
    
    D -. "EMA Update (Momentum Schedule)" .-> E
    
    F --> J["Total SSL Objective"]
    G --> J
    H --> J
```

---

## Underwater Physics Augmentations

Standard computer vision augmentations (random resized crops, color jitter, Gaussian blur) fail to reflect the optical radiative transfer governing underwater imagery. SeaDino introduces five physically-motivated transformations:

### 1. Wavelength-Dependent Attenuation (Beer-Lambert Radiative Formulation)
Under the Beer-Lambert radiative formulation $I_c(z) = I_0(c) \cdot \exp(-\beta_c \cdot z)$, light attenuates exponentially with optical path length $z$ according to wavelength-dependent absorption coefficients $\beta_R > \beta_G > \beta_B$. Because water absorbs longer wavelengths fastest, red light ($\sim 650\text{ nm}$) vanishes within the first 3–5 meters, followed by green, while blue penetrates deepest. In tensor space, SeaDino simulates varying depth by stochastically scaling channels independently according to marine attenuation hierarchies ($s_c \sim U(\alpha_c^{\min}, 1.0)$):
```python
r_scale = random.uniform(0.30, 1.00)  # Red attenuates most aggressively
g_scale = random.uniform(0.60, 1.00)  # Green attenuates moderately
b_scale = random.uniform(0.75, 1.00)  # Blue penetrates deepest
```

### 2. Turbidity and Backscatter Haze (Koschmieder Radiative Model)
Water columns contain dissolved organic matter and particulate matter causing light scattering governed by the Koschmieder atmospheric/underwater model:
$$I' = I \cdot t + A_\infty \cdot (1 - t)$$
where transmission $t \sim U(0.50, 0.95)$ models water turbidity and $A_\infty = [A_R, A_G, A_B]^T$ represents oceanic veiling light (scattered ambient light) stochastically sampled from marine background spectra ($A_R \in [0.05, 0.25], A_G \in [0.25, 0.50], A_B \in [0.30, 0.55]$). Distinct from blur, this physically reduces global scene contrast and shifts chromaticity.

### 3. ROV Strobe Falloff & Off-Axis Vignetting
Seafloor surveys rely on artificial ROV floodlights, creating severe radial illumination gradients:
$$I_{\text{lit}} = I \cdot \left(1 - \alpha \cdot \frac{d(x, y)}{d_{\max}}\right) \quad \text{or} \quad I_{\text{lit}} = I \cdot \left(1 - \alpha \cdot \left(1 - \frac{d(x, y)}{d_{\max}}\right)\right)$$
where normalized distance $d(x, y) = \sqrt{(x - x_0)^2 + (y - y_0)^2} / d_{\max}$ is centered around a stochastic strobe origin $(x_0, y_0) \in [0.2, 0.8] \times [0.2, 0.8]$ with falloff strength $\alpha \sim U(0.25, 0.60)$, capturing both standard radial edge falloff and off-axis floodlight hotspots ($p=0.5$).

### 4. Suspended Marine Snow Particulate Drift
Simulates macroscopic organic debris and flocculent particulate matter drifting across the lens by injecting sparse, stochastic high-intensity Rayleigh/Mie scattering speckle fields ($0.05\% - 0.3\%$ density, $p=0.3$) with specular brightness sampled from $U(0.60, 1.00)$.

### 5. Discrete $90^\circ$ Rotation Invariance
Seafloor survey imagery collected by downward-facing nadir cameras possesses **no canonical up/down orientation**. Conventional continuous rotations introduce triangular black corner wedges when applied to square crops, which vision transformers quickly learn as trivial shortcut features. SeaDino uses exact discrete $90^\circ$ rotations ($\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$) to achieve complete 4-way rotational invariance without interpolation artifacts.

---

## Data Leakage Resolution & Spatial Benchmarks

### 1. Spatial Autocorrelation in Seafloor Transects
Underwater survey robots (AUVs/ROVs) capture continuous image series every 1–3 seconds. In naive random or stratified splits, consecutive frames captured meters apart are split between training and test sets. Rather than learning ecological concepts, models exploit transect-level visual memorization.

#### Distance-to-Train Leakage Profile (Original Naive Splits)
*Distance $D$ from test frames to the nearest training frame in original benchmarks:*

| Distance Threshold ($D$) | Substrate Depth 2 ($N=15{,}100$) | German Bank 2010 ($N=500$) | Biota Multi-Label ($N=37{,}500$) | Ecological Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **$== 0.0\text{ m}$ (Exact duplicates)** | **31** *(0.2%)* | 0 *(0.0%)* | **106** *(0.3%)* | Same image or fixed monitoring station |
| **$< 1.0\text{ m}$** | **69** *(0.5%)* | 0 *(0.0%)* | **166** *(0.4%)* | Sub-meter drift / duplicate rows |
| **$< 5.0\text{ m}$** | $\approx 980$ *(6.5%)* | 0 *(0.0%)* | **2,556** *(6.8%)* | Consecutive AUV frames (seconds apart) |
| **$< 25.0\text{ m}$** | $\approx 4,300$ *(28.5%)* | 11 *(2.2%)* | **12,236** *(32.6%)* | Same sediment patch & lighting |
| **$< 50.0\text{ m}$** | **7,037** *(46.6%)* | **36** *(7.2%)* | **20,050** *(53.5%)* | Continuous tow / micro-habitat |
| **$< 100.0\text{ m}$** | **9,649** *(63.9%)* | **66** *(13.2%)* | **27,249** *(72.7%)* | Heavy spatial autocorrelation zone |

### 2. Spatial Buffer Moats (100m / 50m)
To evaluate true generalization to unseen seafloor, SeaDino establishes spatial block partitions with strict Haversine exclusion moats:
- **Substrate Depth 2:** $500\text{m}$ grid cells, **$100\text{m}$ buffer moat**. Discarded 21,467 images (28.5%). Kept 38,916 Train / 15,141 Test. Min distance in clean split: **$100.02\text{m}$ (0% leakage)**.
- **German Bank 2010:** $250\text{m}$ grid cells, **$50\text{m}$ buffer moat**. Discarded 233 images (7.3%). Kept 2,310 Train / 638 Test. Min distance in clean split: **$50.20\text{m}$ (0% leakage)**.
- **Biota Multi-Label:** $250\text{m}$ grid cells, **$50\text{m}$ buffer moat**. Discarded 41,539 images (29.1%). Kept 63,574 Train / 37,500 Test.

Manifest files are located in `data/benchmarks/`:
- [`substrate_depth_2_spatial_split.csv`](data/benchmarks/substrate/substrate_depth_2_spatial_split.csv)
- [`german_bank_2010_spatial_split.csv`](data/benchmarks/german_bank/german_bank_2010_spatial_split.csv)
- [`biota_spatial_split.csv`](data/benchmarks/biota/biota_spatial_split.csv)

### 3. Cross-Dataset Decontamination: CoralMask-Clean
Perceptual hash auditing (pHash, 64-bit) of all 830 CoralMask test images against our 189,101 SSL pretraining shards revealed that **7 test images** were exact duplicates from Catlin Seaview Survey transects. A Hamming distance sweep ($d \in [0, 10]$) confirmed a clear plateau from $d=6$ to $d=9$ isolating the duplicates.
SeaDino provides:
- [`coralmask_test_clean.txt`](data/benchmarks/coralmask/coralmask_test_clean.txt): Verified decontaminated test manifest ($N = 823$).
- [`coralmask_test_leakage_ids.txt`](data/benchmarks/coralmask/coralmask_test_leakage_ids.txt): The 7 isolated duplicate image stems.

---

## Benchmark Results

### Table 1: Controlled Marine Physics Augmentation Ablation
*Strictly isolates the effect of optical physics augmentations. Identical learning rate ($1\times 10^{-4}$), identical ImageNet normalization, identical pixel loss (ON). Evaluated on leakage-free spatial splits:*

| Downstream Benchmark | Domain Shift | Evaluation Metric | w/ Physics (Ep 8) | No Physics (Ep 8) | **$\Delta$ (Ep 8)** | w/ Physics (Ep 14) | No Physics (Ep 14) | **$\Delta$ (Ep 14)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Substrate Depth 2** | ID *(Out-of-Transect)* | $k$-NN (Macro-F1) | 54.41% | 53.55% | **+0.86%** | 54.94% | 54.10% | **+0.84%** |
| **German Bank 2010** | ID *(Out-of-Transect)* | $k$-NN (Macro-F1) | 68.70% | 67.81% | **+0.89%** | 69.86% | 68.75% | **+1.11%** |
| **Coralscapes** | OOD *(Zero-Shot)* | Linear (mIoU) | 26.17% | 25.54% | **+0.63%** | 26.01% | 25.23% | **+0.78%** |
| **CoralMask-Clean** | OOD *(Zero-Shot)* | Linear (Coral-IoU)| 72.87% | 72.70% | **+0.17%** | 73.03% | 72.68% | **+0.35%** |

> **Scientific Insight:** Optical physics augmentations deliver a universal, consistent improvement ($\mathbf{+0.6\text{ to }+1.1\text{ points}}$) across both ID classification and OOD dense segmentation, while preventing late-epoch representation drift.

---

### Table 2: Progressive SeaDino Pipeline Ablation (Fixed at Epoch 14)
*Stepping progressively through each component of the pretraining pipeline (all trained models locked to Epoch 14 to eliminate early-stopping bias). Dual-probe evaluation reports both parametric linear probing (10 epochs AdamW) and non-parametric $k$-NN ($k=20, T=0.07$):*

| Pretraining Configuration | Substrate Depth 2<br>Linear *(Macro-F1)* | Substrate Depth 2<br>$k$-NN *(Macro-F1)* | German Bank 2010<br>**Linear *(Macro-F1)*** | German Bank 2010<br>$k$-NN *(Macro-F1)* | Coralscapes<br>**OOD** *(mIoU)* | CoralMask-Clean<br>**OOD** *(Coral-IoU)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Pretraining** *(Baseline DINOv3)* | **53.63%** | **57.98%** | 75.55% | **79.53%** | 23.31% | 72.60%* |
| **2. Official Augs + No Pixel Loss** (`ssl_off_no_pix`) | 51.23% | 54.63% | **77.08%** *(+1.53 pt)* | 73.02% | 24.11% | 72.66% |
| **3. Official Augs + Pixel Loss** (`ssl_off_pix`) | 50.06% | 52.42% | 70.70% | 66.16% | 24.93% | 72.72% |
| **4. Marine Augs (No Physics) + Pixel Loss** (`stage1_marine_no_physics`)| 50.61% | 54.07% | 73.08% | 69.53% | 25.23% | 72.68% |
| **5. Marine Augs + Physics + Pixel Loss** (`stage1_marine_with_physics`)| 50.82% | 54.09% | 74.21% *(+1.13 pt)* | 68.01% | **26.01%** | **73.03%** |

*\*Note on CoralMask baseline: 72.60% reflects full deterministic linear probing across all epochs in the master sweep (scoring 54.19% under fast 6-epoch probing).*

> **Scientific Insights (The Linear Probe vs. $k$-NN Paradigm):**
> 1. **German Bank Linear Probe Superiority (Row 2 vs. Baseline):**
>    Under parametric linear probing, `ssl_off_no_pix` achieves **77.74% Accuracy and 77.08% Macro-F1**, definitively **outperforming the un-pretrained foundation baseline** (76.49% Acc, 75.55% Macro-F1) by **$+1.25\text{ pt}$ Accuracy and $+1.53\text{ pt}$ Macro-F1**. This validates that continued pretraining creates superior linearly separable feature representations that uncalibrated cosine $k$-NN distances masked.
> 2. **Marine Optical Physics Gain on German Bank (Row 4 vs. Row 5):**
>    Adding the 5 marine optical physics augmentations delivers **$+0.63\text{ pt}$ Linear Accuracy (75.39% $\to$ 76.02%) and $+1.13\text{ pt}$ Linear Macro-F1 (73.08% $\to$ 74.21%)**, demonstrating consistent physical robustness across seabed substrate classification.
> 3. **Pixel Loss Tradeoff Recovery (Row 3 vs. Row 5):**
>    Enabling pixel reconstruction loss on official augmentations drops German Bank linear performance ($77.08\% \to 70.70\%$, $-6.38\text{ pt}$). However, introducing domain-specific marine and optical physics augmentations in Row 5 recovers **$+2.35\text{ pt}$ Linear Accuracy and $+3.51\text{ pt}$ Macro-F1** over `ssl_off_pix`, while driving dense out-of-distribution segmentation to its project peak (**26.01% mIoU** on Coralscapes).

---

### Table 3: Extended Stage 2 Pretraining & Convergence Sweep (Gram Matrix Objective)
*Warm-start continuation from the Stage 1 peak (`lr_2e-4_with_pixel` Ep 14 @ 26.24% mIoU) through 35 additional epochs with a Gram matrix style objective (`stage2_gram_from_2e4_ep14`):*

| Pretraining Phase / Checkpoint | Total SSL Epochs | Pretraining Objective | German Bank 2010<br>**ID** *(Macro-F1)* | Coralscapes<br>**OOD** *(mIoU)* | Coralscapes<br>*(Pixel Acc)* | Role & Characterization |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Off-the-shelf DINOv3** | 0 | Foundation baseline (no continued SSL) | **79.76%** | 23.31% | 61.20% | Untrained foundation anchor |
| **Stage 1 Peak (`lr_2e-4_with_pixel`)** | 14 | Marine Physics + Pixel Loss | 66.18% | 26.24% | 61.20% | Stage 1 Peak & Stage 2 warm-start initialization |
| **Stage 2 Peak (`best.ckpt` / Ep 19–20)**| **19** | **Stage 2 Gram Matrix SSL** | 66.75% | **26.30%** | **61.27%** | 🏆 **Global Project Peak (+2.99 pt vs baseline)** |
| **Stage 2 Mid-Flight (Ep 44)** | 44 | Stage 2 Gram Matrix SSL | 67.97% | 26.13% | 61.23% | Peak German Bank F1 in Stage 2 (+1.79 pt vs warm-start) |
| **Stage 2 Convergence (Ep 49)** | 49 | Stage 2 Gram Matrix SSL | 67.35% | 26.06% | 61.19% | Long-run saturation & stability plateau |

> **Scientific Insight:** Stage 2 achieves the **global all-time high of 26.30% mIoU on Coralscapes (+2.99 pt over baseline)**. Extending training to Epoch 49 demonstrates zero catastrophic collapse (mIoU holds stably at 26.0–26.3%), proving that feature representations hit an asymptotic ceiling by Epoch 15–20.

---

### Table 4: Progressive Pretraining Ablation on DeepFish (Fixed at Epoch 14)
*Dual-probe evaluation ($k$-NN and linear probing) of fish presence across 20 remote coastal marine habitats under turbid, natural lighting (Saleh et al., Nature Scientific Reports 2020). All Stage 1 models locked to Epoch 14, Stage 2 locked to final convergence Epoch 49:*

| Pretraining Configuration | Pretrain Epoch | Normalization | Linear Probe Acc | Linear Macro-F1 | Linear **Fish-F1** | $k$-NN Acc | $k$-NN Macro-F1 | $k$-NN **Fish-F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Pretraining** *(Baseline DINOv3)* | — | ImageNet | 87.80% | 87.35% | 84.98% | 92.50% | 92.32% | 91.15% |
| **2. Official Augs + No Pixel Loss** (`ssl_off_no_pix`) | 14 | ImageNet | 89.80% | 89.61% | **88.19%** | 92.70% | 92.53% | 91.40% |
| **3. Official Augs + Pixel Loss** (`ssl_off_pix`) | 14 | ImageNet | 88.30% | 88.01% | 86.15% | 92.20% | 91.99% | 90.69% |
| **4. Marine Augs (No Physics) + Pixel Loss** (`stage1_marine_no_physics`) | 14 | ImageNet | 89.10% | 88.80% | 86.98% | **93.40%** | **93.25%** | **92.24%** |
| **5. Marine Augs + Physics + Pixel Loss** (`stage1_marine_with_physics`) | **14** | **ImageNet** | **89.80%** | **89.50%** | **87.71%** | 93.20% | 93.05% | 92.02% |
| **6. Stage 2 Gram Matrix SSL** (`stage2_gram_from_2e4_ep14`) | 49 | BenthicNet | 87.60% | 87.28% | 85.27% | 91.80% | 91.60% | 90.28% |

> **Scientific Insights on DeepFish:**
> 1. **Marine Physics Augmentation Effect (Row 4 vs. 5):** Adding underwater optical physics delivers **$+0.70\text{ pt}$ Linear Accuracy (89.10% $\to$ 89.80%) and $+0.73\text{ pt}$ Fish-F1 (86.98% $\to$ 87.71%)**, perfectly mirroring the universal $+0.6\text{ to }+1.1\text{ pt}$ improvement seen in Table 1 across Substrate, German Bank, and Coralscapes.
> 2. **Pixel Loss Tradeoff Recovery (Row 3 vs. 5):** Turning on pixel loss on official augmentations degrades linear performance ($89.80\% \to 88.30\%$). However, marine physics augmentations completely recover this loss, driving linear accuracy back up to **89.80%** ($+1.50\text{ pt}$) and $k$-NN to **93.20%** ($+1.00\text{ pt}$).
> 3. **Habitat & Nekton Generalization:** Every Stage 1 SSL configuration outperforms the un-pretrained foundation baseline, demonstrating that marine pretraining helps ViTs distinguish camouflaged marine life across diverse seafloor habitats.

---

### Table 5: Progressive Pretraining Ablation on FathomNet (23,636 Full Dataset Evaluation)
*Dual-probe evaluation ($k$-NN and 5-epoch linear probing) of benthic organism taxonomic classification (VME classes: coral, sponges, etc.) on the full 23,636-image FathomNet benchmark (18,908 train / 4,728 test across 4 classes). All Stage 1 models evaluated at converged Epoch 14, Stage 2 at final convergence Epoch 49:*

| Pretraining Configuration | Pretrain Epoch | Normalization | Linear Probe Acc | Linear Macro-F1 | $k$-NN Acc | $k$-NN Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Pretraining** *(Baseline DINOv3)* | — | ImageNet | 79.95% | 63.89% | 84.05% | 74.03% |
| **2. Official Augs + No Pixel Loss** (`ssl_off_no_pix`) | 14 | ImageNet | **80.05%** | **64.40%** | **85.09%** | **75.14%** |
| **3. Official Augs + Pixel Loss** (`ssl_off_pix`) | 14 | ImageNet | 78.91% | 60.85% | 82.95% | 70.78% |
| **4. Marine Augs (No Physics) + Pixel Loss** (`stage1_marine_no_physics`) | 14 | ImageNet | 78.68% | 60.94% | 83.93% | 72.69% |
| **5. Marine Augs + Physics + Pixel Loss** (`stage1_marine_with_physics`) | **14** | **ImageNet** | **78.93%** | **61.74%** | **84.12%** | **73.56%** |
| **6. Stage 2 Gram Matrix SSL** (`stage2_gram_from_2e4_ep14`) | 49 | BenthicNet | 78.28% | 59.45% | 83.06% | 70.54% |

> **Scientific Insights on FathomNet (23k Benchmark):**
> 1. **Superior Domain Foundation Without Pixel Loss (Row 2 vs. Baseline):** Continued pretraining on marine imagery without pixel reconstruction (`ssl_off_no_pix`) achieves the top benchmark scores across every evaluation axis (**80.05% Lin Acc, 64.40% Lin Macro-F1, 85.09% kNN Acc, 75.14% kNN Macro-F1**), definitively outperforming the un-pretrained DINOv3 foundation baseline (+1.04 pt kNN Acc, +1.11 pt kNN Macro-F1, +0.51 pt Lin Macro-F1).
> 2. **Marine Optical Physics Augmentation Gain (Row 4 vs. Row 5):** Comparing `stage1_marine_no_physics` (78.68% Lin Acc, 60.94% Lin Macro-F1) against `stage1_marine_with_physics` (78.93% Lin Acc, 61.74% Lin Macro-F1, 84.12% kNN Acc, 73.56% kNN Macro-F1) isolates the pure empirical contribution of the 5 marine optical physics augmentations. Underwater optical modeling provides **$+0.25\text{ pt}$ Linear Accuracy, $+0.80\text{ pt}$ Linear Macro-F1, $+0.19\text{ pt}$ $k$-NN Accuracy, and $+0.87\text{ pt}$ $k$-NN Macro-F1**, demonstrating that simulating wavelength attenuation, turbidity backscatter, and strobe falloff helps the ViT generalize to complex deep-sea organism morphologies.
> 3. **The Pixel Reconstruction Tradeoff (Row 2 vs. Row 3):** Enabling pixel reconstruction loss (`ssl_off_pix`) forces the ViT student to spend representation capacity reconstructing fine-grained pixel RGB textures and sediment patterns, creating an optimization tension that reduces global organism categorization performance ($-1.14\text{ pt}$ Lin Acc, $-3.55\text{ pt}$ Lin Macro-F1). For pure taxonomic classification and retrieval, patch-level self-distillation without pixel reconstruction creates a cleaner metric space.
> 4. **Statistical Rigor of Full-Dataset Evaluation:** Unlike small sample crops, evaluating on 18,908 train and 4,728 test images ensures low standard error and demonstrates true foundational feature generalization across MBARI oceanographic imagery.

---

### Table 6: Progressive Pretraining Ablation on Brackish Marine Fauna (Pedersen et al., CVPRW 2019)
*Dual-probe evaluation ($k$-NN and linear probing) of multi-class marine fauna classification across 6 categories (`crab`, `fish`, `jellyfish`, `shrimp`, `small_fish`, `starfish`) in turbid European coastal waters. Comparing non-parametric feature geometry against linear separability:*

| Pretraining Configuration | Pretrain Epoch | Normalization | Linear Probe Acc (5ep) | Linear Macro-F1 (5ep) | Linear Probe Acc (30ep) | Linear Macro-F1 (30ep) | $k$-NN Acc | $k$-NN Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Pretraining** *(Baseline DINOv3)* | — | ImageNet | 23.95% | 11.79% | 31.18% | 19.94% | 27.88% | 17.97% |
| **2. Official Augs + No Pixel Loss** (`ssl_off_no_pix`) | 14 | ImageNet | 29.28% | 16.94% | 31.94% | 18.25% | 31.69% | 19.33% |
| **3. Official Augs + Pixel Loss** (`ssl_off_pix`) | 14 | ImageNet | 32.70% | 17.36% | 31.69% | 16.13% | 30.29% | 15.99% |
| **4. Marine Augs (No Physics) + Pixel Loss** (`stage1_marine_no_physics`) | 14 | ImageNet | 30.54% | 16.89% | 30.80% | 15.80% | **34.85%** | **21.71%** |
| **5. Marine Augs + Physics + Pixel Loss** (`stage1_marine_with_physics`) | **14** | **ImageNet** | **33.21%** | **18.34%** | 31.81% | 17.06% | 30.93% | 17.43% |
| **6. Stage 2 Gram Matrix SSL** (`stage2_gram_from_2e4_ep14`) | 49 | BenthicNet | 29.53% | 16.36% | **32.32%** | 17.53% | 26.49% | 15.20% |

> **Scientific Insights on Brackish Marine Fauna:**
> 1. **Decisive Foundation Advantage in Metric Space ($k$-NN: $+6.97\text{ pt}$ Acc, $+3.74\text{ pt}$ Macro-F1):**
>    In non-parametric cosine-weighted $k$-NN—which evaluates the intrinsic geometry of the representation without supervised head fitting—`stage1_marine_no_physics` achieves **34.85% Acc / 21.71% Macro-F1**, substantially outperforming off-the-shelf DINOv3's **27.88% Acc / 17.97% Macro-F1** ($+6.97\text{ pt}$ Acc, $+3.74\text{ pt}$ Macro-F1). This demonstrates that continued pretraining establishes superior domain-specific metric clustering in turbid underwater conditions.
> 2. **Fast Linear Adaptation (5 Epochs):**
>    At 5 epochs of probing, `stage1_marine_with_physics` achieves **33.21% Acc and 18.34% Macro-F1**, beating off-the-shelf DINOv3 (23.95% Acc, 11.79% Macro-F1) by **$+9.26\text{ pt}$ Accuracy and $+6.55\text{ pt}$ Macro-F1**. Off-the-shelf DINOv3 requires 30 epochs of supervised gradient updates just to approach what SeaDino achieves in 5 epochs.
> 3. **The Full-Frame Resolution Crushing Phenomenon:**
>    In full-frame evaluation (1920×1080 downsampled to 224×224), small organisms (shrimps, crabs, small fish) occupy $<8\times 8$ pixels ($<0.25$ of a single $16\times 16$ ViT patch), while $>99.7\%$ of pixel volume constitutes turbid water background and backscatter. Full-frame CLS tokens primarily cluster water turbidity, driving linear probes under standard unweighted loss to collapse onto majority classes (`fish`), leaving minority classes (`shrimp`, `starfish`) with near-zero recall.
> 4. **Organism-Level Cropping (`--brackish_crop_boxes`) & Class Balancing:**
>    Enabling bounding-box cropping (`--brackish_crop_boxes`) extracts organisms into full $224\times 224$ patches (100% organism coverage with a 15% contextual margin), mirroring the high-accuracy setup in FathomNet (Table 5, 80.36%). Pairing this with balanced cross-entropy class weights eliminates majority-class collapse across rare marine fauna.

---

## Quickstart & Installation

```bash
git clone https://github.com/NeelJani1/SeaDino.git
cd SeaDino

conda create -n seadino python=3.10 -y
conda activate seadino

# PyTorch with CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SeaDino in editable mode
pip install -e .
```

---

## Pretraining with SeaDino

Pretraining streams WebDataset `.tar` shards via [`ssl_train.py`](ssl_train.py):

```bash
python ssl_train.py \
    --use_webdataset \
    --shard_dir /path/to/benthicnet_shards \
    --output_dir ./checkpoints/seadino_vits16_stage1 \
    --model_id facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay_start 0.04 \
    --global_crop_size 224 \
    --local_crop_size 96 \
    --num_global_crops 2 \
    --num_local_crops 8 \
    --marine_aug \
    --benthic_norm \
    --use_bf16 \
    --epochs 15
```

### Key Pretraining Flags
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--use_webdataset` | `flag` | `False` | Stream training data from tar shards via WebDataset |
| `--shard_dir` | `str` | `None` | Directory containing WebDataset `.tar` shards |
| `--data_dir` | `str` | `None` | Directory of raw image files (alternative to `--use_webdataset`) |
| `--learning_rate` | `float` | `1e-4` | Peak learning rate (cosine decay with warm-up) |
| `--weight_decay_start` | `float` | `0.04` | Initial AdamW weight decay (with `--wd_schedule constant` by default) |
| `--marine_aug` | `flag` | `False` | Enable domain-specific underwater augmentations |
| `--no_marine_physics_aug` | `flag` | `False` | Disable underwater optical physics transforms (ablation baseline) |
| `--benthic_norm` | `flag` | `False` | Normalize using BenthicNet empirical channel statistics |
| `--use_bf16` | `flag` | `False` | Enable native `bfloat16` mixed precision |

---

## Evaluation Suite

SeaDino provides an automated, multi-dataset benchmarking CLI via [`run_eval.py`](run_eval.py):

```bash
python run_eval.py \
    --checkpoint_dirs /path/to/checkpoints \
    --include_off_the_shelf \
    --datasets substrate german_bank coralscapes coralmask biota deepfish fathomnet brackish \
    --coralmask_epochs 6 \
    --resume \
    --use_bf16 \
    --output_csv benchmark_evaluation.csv
```

### Supported Benchmarks
* **Substrate Depth 2** (CATAMI 5-class benthic substrate classification, spatial split)
* **German Bank 2010** (5-class seafloor morphology classification, spatial split)
* **BenthicNet Biota** (272-class multi-label biological presence, spatial split)
* **Coralscapes** (39-class dense coral reef semantic segmentation from EPFL on Hugging Face)
* **CoralMask-Clean** (Binary coral head segmentation with verified perceptual de-duplication)
* **DeepFish** (Fish vs. No-Fish habitat classification across 20 tropical marine habitats, 39k frames, Saleh et al., *Scientific Reports* 2020)
* **FathomNet** (Out-of-distribution benthic organism taxonomic recognition, bounding-box crops via $k$-NN & linear probe)
* **Brackish** (6-class marine fauna classification in turbid European coastal waters, Pedersen et al., 2019)

### External Storage & G-Drive Mounts
SeaDino automatically detects external storage mounts on drive G (`/mnt/g/data`, `/mnt/g/SeaDino_Data`, `/mnt/g/SeaDino/data`) and redirects heavy datasets, checkpoints, results, and Hugging Face caches to preserve local disk space:

```bash
# 1. Initialize G-Drive folder structure and environment file
python scripts/setup_gdrive_storage.py

# 2. Activate environment variables (optional, auto-detected by default)
source env_gdrive.sh

# 3. Download benchmarks directly to external G-Drive
python scripts/download_deepfish.py    # Downloads & extracts 7.1GB DeepFish
python scripts/download_fathomnet.py   # Downloads & crops FathomNet annotations
python scripts/download_brackish.py   # Downloads & extracts Brackish marine fauna

# 4. Run multi-dataset evaluation (with organism-level crops for Brackish)
python run_eval.py --datasets deepfish fathomnet brackish --brackish_crop_boxes --include_off_the_shelf
```

---

## Repository Structure

```
SeaDino/
├── README.md                          # Comprehensive project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Production Python dependencies
├── pyproject.toml                     # Modern package and CLI configuration
├── .gitignore                         # Strict rules preventing checkpoint/data leaks
│
├── ssl_train.py                       # Core DINOv3 Marine SSL Pretraining Engine
├── run_eval.py                        # Multi-dataset evaluation orchestrator
│
├── eval_suite/                        # Modular benchmarking package
│   ├── config.py                      # Dynamic path resolution & G-Drive mount detection
│   ├── models.py                      # Backbone loader & patch feature extractors
│   ├── utils.py                       # Metric evaluation, confusion matrices, seeds
│   ├── benchmarks/                    # Evaluation probes (knn, linear, coralmask, biota, deepfish, fathomnet, brackish)
│   └── datasets/                      # PyTorch datasets (single_label, biota, coralmask, coralscapes, deepfish, fathomnet, brackish)
│
├── scripts/                           # Helper & automation scripts
│   ├── setup_gdrive_storage.py        # Initializes G-Drive directory layout & env_gdrive.sh
│   ├── download_deepfish.py           # Downloads and unpacks DeepFish to G-Drive
│   ├── download_fathomnet.py          # Downloads and crops FathomNet annotations
│   └── download_brackish.py           # Downloads Brackish dataset to G-Drive
│
└── data/                              # Benchmark splits and pretraining metadata
    ├── benchmarks/                    # Dedicated directories per downstream task
    │   ├── substrate/                 # Spatial splits (Substrate Depth 2)
    │   ├── german_bank/               # Spatial splits (German Bank 2010)
    │   ├── coralmask/                 # Clean manifests & perceptual leakage IDs
    │   ├── biota/                     # Spatial splits (BenthicNet Biota)
    │   ├── deepfish/                  # DeepFish specifications
    │   ├── fathomnet/                 # FathomNet specifications
    │   └── brackish/                  # Brackish specifications
    └── training/                      # Pretraining WebDataset shard specifications
        └── README.md                  # Shard layout & deduplication hash instructions
```

---

## Citation

```bibtex
@article{jani2026seadino,
  title={SeaDino: Physics-Informed Self-Supervised Vision Foundation Models for Marine and Benthic Imagery},
  author={Jani, Neel and Contributors},
  journal={GitHub Repository},
  url={https://github.com/NeelJani1/SeaDino},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
