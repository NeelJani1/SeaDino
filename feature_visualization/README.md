# Frozen backbone feature explorer

Independent CLI for the nine SeaDino benchmark encoders. Each scatter point is
one **whole image**. Feature similarity determines position; annotations only
determine color. A mostly fish image may group with other fish images, but a
small fish against seabed can also retain background structure. The program does
not infer which object occupies most of an image. Dataset labels may describe an
annotated organism rather than the dominant pixels.

No SSL heads, classifiers, prompt decoders, or additional backbone training are
used. Optional attention inspection uses the same frozen encoder weights. The global feature matches the benchmark: CLS for
DINO/CLIP and adapter-provided mean patch pooling for SAM3.1/MarineInst. SeaDino
uses the same EMA teacher and normalization as its benchmark checkpoint. Custom
backbone tensors must all match the loaded checkpoint; silent fallback is an error.

## Run in WSL

```bash
cd /home/njan320/SeaDino
conda activate SeaDino
python -m pip install -r feature_visualization/requirements.txt
python -m feature_visualization \
  --dataset fathomnet --split train \
  --models all --methods pca umap tsne --dims 2 3 \
  --max_images 2000 --patch_images 20 --seed 42
```

Outputs default to `/mnt/d/Neel/FeatureVisualizations/<UTC timestamp>_<dataset>/`.
Open `index.html` in a browser. HTML assets are local, with no CDN, web service,
telemetry, or model download needed. Keep the output folder together when sharing.
Click a scatter point to inspect its thumbnail, original path, label, and split.
Switch Dataset labels / K-means clusters without changing any coordinates.

The first completed demonstration is
`/mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019/index.html`:
2,000 train images (500 each of coral/crinoid/fish/sponge), 54 interactive views,
and 20 shared patch-image comparisons across all nine encoders. The validation
smoke folder has fewer images and is not the main report.

For a localhost preview, serve the output root on loopback only:

```bash
python -m http.server 8767 --bind 127.0.0.1 \
  --directory /mnt/d/Neel/FeatureVisualizations
```

Then open `http://localhost:8767/fathomnet_all9_20261019/index.html`. A preview
server must remain running for localhost links; the exported HTML also works
directly from disk without a server.

Use `--models dinov3_s seadino_s dinov3_b seadino_b` for a subset. Available keys
are in `models.json`; `--model_config` accepts an alternative file with the same
structure, including checkpoint overrides. The defaults pin the exact SeaDino
checkpoints in the comparison tables and the SAM3.1 local snapshot. Frozen encoder
parameter counts and pooling are recorded in each model's `result.json`.

```bash
python -m feature_visualization --input_dir /path/to/images \
  --folder_labels auto --models dinov3_s seadino_s --max_images 500
```

`auto` labels images by the first subfolder under the input root; images directly
in the root are unlabeled. `parent` uses the immediate containing folder. `none`
disables folder labels. Folder names are user labels, not verified annotations.
Different files sharing a filename remain distinct. Identical paths are deduplicated;
conflicting labels on one path cause an error. Unreadable images fail with their ID
and path; they are never silently substituted.

For one image and custom resolution:

Use an absolute WSL drive path starting with `/mnt/d/`, including the leading
slash. `--image_path` selects one file; `--input_dir` also accepts a single image
as an alias, or recursively scans a folder. Single-file inputs remain unlabeled.

```bash
python -m feature_visualization --image_path /mnt/d/Neel/my_images/fish.jpg \
  --models all --image_size 448 --patch_longest 768 --patch_images 1 \
  --batch_size 1 --output_dir /mnt/d/Neel/FeatureVisualizations/my_fish
```

`--image_size` controls square whole-image features (default 224 x 224).
`--patch_longest` controls the longest edge for spatial PCA (default 512), preserving
aspect ratio. Inputs are padded to each backbone's patch multiple; e.g. a requested
512 x 512 global image becomes 518 x 518 for patch-14 models. Exact requested/padded
sizes are recorded. Use the same requested resolution across models. A single image
produces useful patch maps but its scatter contains only one point. Use a folder of
multiple images to inspect semantic groups. The HTML report is an offline viewer;
new feature extraction is launched through the CLI. Changing resolution requires a
new extraction, and results at custom sizes differ from the standard 224 benchmark.
For one image, keep `--image_size 224` unless its whole-image embedding is needed
at a different resolution: this does not limit `--patch_longest`. Spatial patch
extraction runs one image at a time regardless of `--batch_size`. A 4096-pixel
longest edge has far more tokens than the default 512 and can exhaust GPU memory;
upsampling a smaller source image does not recover additional image detail.

## Benchmarks and sampling

- Image projections: Substrate, German Bank, FathomNet, Brackish, DeepFish.
- Patch PCA only: CoralMask and Coralscapes. No image-level semantic labels are
  inferred from their segmentation annotations. Their selection is capped at
  `--patch_images`, avoiding extraction for unused images.
- The first four classification benchmarks and Coralscapes read **existing**
  explicit-v1 manifests. No splits or benchmark results are rewritten.
- DeepFish reads the requested official CSV (`validation` maps to `val`). CoralMask
  supports train/test and requires the clean test manifest for test images.
- Default: seed 42, train, maximum 2,000 images. Labeled inputs are sampled in
  class round-robin order without duplication, filling remaining capacity when a
  class is exhausted. Unlabeled inputs use uniform sampling. The saved class counts
  describe the actual selection, which may differ from natural class proportions.
- Every model uses the same ordered records. Inputs are staged on D: without JPEG
  re-encoding; thumbnails and source hashes are recorded. Test plots are exploratory,
  not a basis for selecting training settings for a claimed untouched test result.

## Methods

Global features: benchmark bicubic antialiased resize to 224 x 224, model-specific
normalization, FP32, then L2 normalization. Independent PCA/UMAP/t-SNE fits for
each model; **axes and intercluster distances are not aligned across models**.

PCA uses centered features. UMAP uses cosine distance, 15 neighbors, min_dist 0.1,
and seeded random initialization. t-SNE uses PCA preprocessing to at most 50
dimensions, perplexity 30, 1,000 iterations and automatic learning rate. Smaller
inputs receive recorded neighbor/perplexity adjustments; fewer than four images or
constant features receive a clearly recorded PCA/zero-coordinate fallback.

K-means fits original normalized features, never 2D/3D coordinates, using ten
initializations and the run seed. Default K is the known class count, or eight for
unlabeled data, bounded by samples and unique feature vectors. `--n_clusters` overrides
it. Clusters stay numbered; they are not renamed into ground-truth classes. Metrics:
cosine silhouette (up to 2,000 samples), ARI and NMI on labeled samples. Undefined
metrics are null, not zero. Nice-looking plots alone do not establish representation
quality; compare the existing linear-probe and kNN results as well.

Patch maps default to the first 20 shared sampled images, a 512-pixel longest side,
preserved aspect ratio, and reflection padding to the model's patch multiple.
Register/CLS tokens are excluded by the spatial suffix. Maps interpolate floating
PCA scores before scaling and remove padded pixels. A single PCA basis is fit across
selected images within each model, using at most 50,000 sampled patches, with a
deterministic sign convention and shared MAD-based color scaling. Colors are
**feature directions**, not fish/coral predictions; color identities do not transfer
across models. `--pc_indices 3 4 5` selects different 1-based RGB components.

## Caches and exports

`cache/` holds raw and normalized global features plus bounded patch tokens. Identity
includes source weight SHA256 hashes, extraction/adapter code, image checksums and
order, transforms, precision, batch size, geometry, and relevant package versions.
Feature arrays have a separate checksum. Extraction is sequential, releasing each
model before analysis and the next model load.

`<model>/` contains offline interactive HTML, coordinates CSV, method metadata,
label-colored and cluster-colored PNG/PDF plots, patch maps, fitted PCA bases, and
model metadata. `patch_comparisons/` compares identical images across backbones.
The root includes `run.json`, `samples.csv`, `metrics.csv`, and `index.html`.

Rerender or change projection settings without loading models or accessing CUDA:

```bash
python -m feature_visualization --render_only \
  --output_dir /mnt/d/Neel/FeatureVisualizations/YOUR_RUN \
  --methods pca umap tsne --dims 2 3
```

Extraction provenance remains attached to the cached arrays. The last render's code,
versions, and arguments are separately recorded. Use a new run directory for a new
sample selection. Model weights must be cached locally; offline mode is enabled.

## Validation

```bash
python -m unittest discover -s feature_visualization/tests -v
```

Synthetic tests cover sample alignment, label independence, token suffix/padding,
constant/tiny inputs, strict checkpoint audits, and cache invalidation. Real-model
smoke comparisons additionally check benchmark global features for all nine encoders.

## Attention maps

Attach maps to an existing report without recomputing its embeddings or clustering:

```bash
python -m feature_visualization --attention_only \
  --output_dir /mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019 \
  --models all --attention_images 20 --attention_longest 512
```

For a new run, add `--attention_maps`. The attention resolution is independent of
`--image_size` and `--patch_longest`: `--attention_longest` preserves aspect ratio
and pads to each model's patch multiple. The selection starts with the shared patch
images so side-by-side comparisons use identical inputs. If no patch images were
selected, it uses the first shared image records. The requested attention count is
capped by the available shared patch selection.

- DINOv3 S/B, SeaDino S/B, DINOv2 S/B and CLIP: genuine last-layer CLS-to-patch
  attention, separate head overlays, and residual attention rollout. All encoder
  tokens participate in rollout; CLS/register tokens are omitted from the final
  spatial grid. The rollout is a head-averaged attention-flow approximation, not
  an exact attribution through MLPs and residual branches.
- SAM3.1: exact last global-layer QK + 2D RoPE softmax, averaged over spatial
  queries to show attention received by each key. The encoder stays on SDPA;
  capturing its probabilities does not change its output. A synthetic test
  reconstructs the installed HF implementation, including rotary positions.
- MarineInst: last global-layer attention reconstructed with the exact native
  QK scaling and decomposed relative positions, with received-patch and head views.
  A synthetic parity test reconstructs its native attention output from these weights.
- Native SAM/Marine windowed encoders do not receive a fabricated CLS rollout.
  Their query semantics are visibly different in comparison panels.

The extractor uses eager FP32 attention for CLS encoders and the native attention
formulas for SAM3.1/MarineInst. It verifies probability rows and, when resolution/selection
matches the saved patch cache, compares attention-enabled forward features with
the earlier benchmark-style frozen encoder features. Source weight and staged image
checksums must still match the original report. Hooks are removed after each use.

`<model>/attention/index.html` provides overlays, individual heads, raw unscaled
NPZ arrays, PNG/PDF panels, and JSON metadata. `attention_comparisons/index.html`
compares the same images across all models; the main report links these views.
`attention_run.json` records attention-specific provenance without replacing the
original embedding extraction provenance. Checksums protect raw attention caches.

Display contrast uses per-image 2nd/98th percentiles (independently per head).
Uniform maps remain uniform; interpolation roundoff is not amplified. Brightness
is not comparable as an absolute probability across panels. Attention is internal
token mixing, is not class-conditioned here, and is not a segmentation prediction
or causal proof. For FathomNet panels the dashed cyan box is the stored COCO
annotation; it is never fed to the model or treated as a prediction.

The report also includes `quality_guide.html`, a sourced matrix covering frozen
probes, dense prediction, retrieval, spatial PCA, attention, feature health,
clustering and representation comparison. These are complementary checks; the
matrix does not claim that every listed metric has been run in this tool.

## Dense Coralscapes comparison

```bash
python -m feature_visualization --dataset coralscapes --split train \
  --models all --max_images 20 --patch_images 20 --patch_longest 896 \
  --attention_maps --attention_images 20 --attention_longest 896 \
  --batch_size 4 --seed 42 \
  --output_dir /mnt/d/Neel/FeatureVisualizations/coralscapes_all9_20261019
```

Coralscapes uses the existing official train manifest and its exact cached HF row
fingerprints. Original integer masks are preserved losslessly and checked against
image dimensions. Reports show the original, official-palette ground truth, PC1,
PCA RGB, PCA with annotation boundaries, and an image/feature overlay. Per-model
spatial pages include class legends and original-resolution pixel coverage. Every
model shares the same sampled images. IDs 0/255 remain unmapped/void; annotation
boundaries exclude these pixels. Masks are resized with nearest-neighbor only to
the unpadded geometry, including for the attention reference panel.

This spatial report deliberately has no whole-image semantic classes or scatter
plots. Neither PCA nor attention is fitted to masks. No semantic IoU is computed
between arbitrary PCA colors and ground-truth IDs; quantitative segmentation
quality remains the job of a separately trained, held-out evaluated decoder.

Model attention links appear as each encoder finishes. The all-model attention
gallery is linked after it exists. If attaching maps, the runner only hashes the
selected images, rather than every original in a larger image-scatter report.

FP32 eager attention and SDPA accumulate differently. Feature agreement is checked
on complete patch vectors (maximum relative L2 5e-4, overall relative L2 1e-4),
with actual errors recorded. An audit on all twenty FathomNet CLIP samples found
bitwise-identical fresh SDPA features to the saved cache; eager differences were
at most 0.00333 per element / 0.000204 relative L2 per patch. This avoids a brittle
near-zero element criterion while still rejecting changed encoder outputs.

References: [UMAP clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html),
[DINOv2 patch PCA](https://arxiv.org/abs/2304.07193),
[scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

## Loading/rendering audit and SAM rectangular correction (19 October 2026)

`attention_diagnostics/index.html` provides a concrete teacher-weight audit,
matched-scale base/adapted attention differences, SAM before/after figures, and
interactive patch-block versus bilinear views with zero-to-peak or percentile
contrast and adjustable overlay opacity. Generate it from saved outputs on CPU:

```bash
python -m feature_visualization.diagnostics /mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019
```

The audit on twenty FathomNet frames found 211/211 loaded SeaDino tensors match
the selected teacher checkpoint, and 210/211 differ from the corresponding
foundation model. Mean raw attention correlation is 0.623 for S and 0.959 for B
under the saved normalization policies; it is not a pure weight-only ablation.
MarineInst's mean normalized spatial attention entropy is 0.977. Its received
attention is averaged over all patch queries and heads, unlike a CLS query.

An actual non-square SAM adapter error was found: HF 5.3's rectangular buffer
construction uses height as `end_x`, but flattened NHWC patches advance x modulo
width. `SAM3BackboneWrapper` now explicitly uses `end_x=width, end_y=height`, with
the existing isotropic window/height scale. Square inputs and learned weights
are unchanged. Independent meshgrid tests cover landscape/portrait geometries;
matching a second path through the same HF constructor had missed the bug.

The adapter identity is now `sam3-vit-patches-mean-pool-v2`. Only SAM spatial
features/attention in the two reports were refreshed. Previous spatial artifacts
are preserved under `corrections/sam_rectangular_rope_before/`; per-model code
provenance is saved with the new cache. Square global features were reused only
after a fresh forward matched four saved samples exactly in each report.
**Earlier SAM segmentation scores obtained with rectangular inputs need a fresh
probe evaluation. This visual refresh does not replace those benchmark scores.**

DINOv2/CLIP artifacts are discussed in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
and [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010).
This sample has strong CLIP norm outliers; additional pre-output-LayerNorm checks
found no comparably large norm outliers in its DINOv2 S/B features. These checks
do not establish one universal cause for every irregular attention patch.
