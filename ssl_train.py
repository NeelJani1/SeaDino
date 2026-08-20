"""
Self-Supervised Learning (SSL) Training for DINOv3 on Seafloor Imagery
========================================================================
Alignment status vs official facebookresearch/dinov3 (main @ 6876159, 2026-07-15):

  This revision fixes several places where the previous version's own alignment
  checklist didn't match what the official repo actually does. See "FIXES IN
  THIS REVISION" below for the full list; the checklist itself is now split
  into what's genuinely confirmed correct vs. what was changed.

  CONFIRMED CORRECT (verified line-by-line against official source):
  ✅ DINO loss – global/local pair-count scaling (n_gg=G*(G-1), n_lg=G*L),
     ignore_diagonal=True for global / False for local
  ✅ iBOT patch loss – separate head from DINO head
  ✅ KoLeo – PRE-HEAD backbone CLS tokens, float32 cast
  ✅ Teacher temperature warm-up (0.04 -> target, then flat)
  ✅ EMA update formula (teacher = teacher*m + student*(1-m))
  ✅ DINOHead trunc_normal_ init
  ✅ bfloat16 AMP – scaler correctly bypassed (only needed for float16)
  ✅ Loss weight/pair-count arithmetic generally

  FIXES IN THIS REVISION (previously wrong or only partially matching official):
  🔧 Sinkhorn-Knopp centering: official's `sinkhorn_knopp_teacher` (both DINO and
     iBOT losses) never subtracts a running `center` -- that's only used by the
     alternate `softmax_center_teacher` path, which upstream never actually calls
     (ssl_meta_arch.py asserts `centering == "sinkhorn_knopp"`). We stopped
     subtracting/updating `center` in the Sinkhorn path to match.
  🔧 iBOT masking: official does NOT sample a Gaussian ratio per sample. It (a)
     gates what fraction of the batch gets masked at all via mask_sample_probability
     (the rest get an empty mask), (b) uses linearly-spaced ratios across the masked
     subset (torch.linspace), not Gaussian, and (c) places BEiT-style contiguous
     rectangular blocks on the real 2D patch grid, not an i.i.d. scatter of patches.
     MaskingGenerator is now a faithful port of dinov3/data/masking.py, and batch-level
     gating matches dinov3/data/collate.py's collate_data_and_cast. iBOTPatchLoss now
     also applies official's per-sample masks_weight reweighting (forward_masked),
     since our earlier plain `.mean()` implicitly over-weighted heavily-masked samples.
     NOTE: official pools the masking budget across BOTH global crops together
     (B = 2*batch_size in collate_data_and_cast) before drawing ratios and shuffling;
     this script still draws each global crop's mask independently (two separate
     ~half-batch-sized pools). Converges to the same thing in expectation, but the
     per-crop split isn't stochastic the same way official's is. Left as-is --
     low practical impact, flagged for completeness.
  🔧 KoLeo: official computes KoLeo separately per global-crop slot and sums the
     results (the /n_global_crops and *n_global_crops in ssl_meta_arch.py cancel
     out to a plain sum) -- it never pools different crops' embeddings into one
     nearest-neighbor search. We were concatenating both global crops before running
     KoLeo, which lets a same-image crop pair become each other's nearest neighbor
     and weakens the anti-collapse effect. Fixed to sum per-crop losses.
  🔧 GramLoss: every real training recipe that touches Gram (gram_anchor,
     high_res_adapt, vitl16_distilled) sets remove_neg=False AND
     remove_only_teacher_neg=False. Default flipped to match; both are still
     available as opt-in flags for experimentation.
  🔧 DINOHead: `weight_norm` does not appear anywhere in the current dinov3 repo's
     Python source (grepped the whole tree) -- dino_head.py just uses a plain
     nn.Linear for the last layer. Default flipped to off; kept as an opt-in
     (`use_weight_norm`) for anyone deliberately replicating the older DINOv2 head.
  🔧 EMA momentum / weight decay schedules: the *shape* of our cosine formula was
     already algebraically identical to official's, but the real recipes
     (dinov3_vit7b16_pretrain.yaml, _gram_anchor.yaml, _high_res_adapt.yaml) set
     start==peak==end, i.e. CONSTANT momentum/weight-decay within each phase
     (momentum 0.994 during main pretrain, 0.999 during gram-anchor/high-res
     stages; weight decay constant 0.04 throughout), not a ramp across the whole
     run. Default behavior is now "constant"; the old cosine ramp is kept as an
     opt-in (`--momentum_schedule cosine` / `--wd_schedule cosine`).
  🔧 official_dinov3_aug: GaussianBlur kernel_size is a constant 9 in official
     (all crop sizes), not 23/5. RandomResizedCrop uses bicubic interpolation
     explicitly; torchvision's default is bilinear. Both fixed.
  🔧 koleo_loss_weight default changed 1e-4 -> 0.1 (official default everywhere
     it's set explicitly).
  🔧 Layer-wise LR decay: official decays learning rate by depth (deeper layers
     learn slower). Implemented via get_vit_lr_decay_rate(), matching official's
     dinov3/train/param_groups.py exactly. Separate patch-embed LR multiplier
     also available (--patch_embed_lr_mult).
  🔧 patch_embed_lr_mult default changed 1.0 -> 0.2. The bare function signature
     in official's param_groups.py defaults to 1.0, but every shipped recipe that
     sets it explicitly (ssl_default_config.yaml, pretrain, gram_anchor,
     high_res_adapt, distilled) overrides it to 0.2. That's the value actually
     used in practice.
  🔧 Weight-decay exemption list corrected to match official exactly. Official's
     real condition (param_groups.py) is:
         name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name
     Official does NOT exempt cls_token/mask_token/register_tokens/storage_tokens
     from weight decay -- those tokens get zero LR decay-exemption treatment
     (handled separately, only affecting the LR multiplier via layer_id=0) but
     STILL receive normal weight decay. Previous revision's no_decay_keywords
     incorrectly zeroed weight decay on these tokens. Fixed to match official's
     exact condition; the len(p.shape)==1 fallback is kept as a safety net for
     legitimately-1D params (e.g. LayerScale's `lambda1` in the HF port, which
     doesn't literally contain "gamma") since cls_token/mask_token/register_tokens
     are 3D tensors and are never accidentally caught by that fallback.
  🔧 AdamW betas exposed via --adam_beta1/--adam_beta2 instead of hardcoded.
     beta2=0.99 is NOT a universal "official DINOv3" constant -- it's specific to
     the 7B-scale main recipes (pretrain/gram_anchor/high_res_adapt). The base
     ssl_default_config.yaml default, the distilled recipe, and the linear-probe
     recipe (vitl_im1k_lin834.yaml) all use the standard AdamW 0.999. Defaults
     here stay at (0.9, 0.99) since that's what the recipes closest to this
     script's use case (continued SSL pretraining, not distillation or linear
     probing) actually use -- but it's now a CLI-overridable choice, not a
     silently-hardcoded "official" value.
  🔧 official_dinov3_aug: horizontal flip removed from flip_and_color_jitter.
     Checked across every real SSL pretraining recipe -- pretrain, gram_anchor,
     high_res_adapt, and distilled all explicitly set horizontal_flips: false.
     True only appears in the base placeholder ssl_default_config.yaml and in
     vitl_im1k_lin834.yaml, a downstream linear-probe eval config, not a
     pretraining recipe -- so it was never the value official actually trains
     SSL with. (This was originally flagged early in this file's review history
     but never actually applied across the subsequent rounds of fixes -- fixed
     now.)
  🔧 official_dinov3_aug: documented (docstring + runtime warning) that the
     fixed local_crops_size=112 only pairs correctly with image_size=256 --
     official's real global_crops_size for pretrain/gram_anchor. Running this
     flag at other --image_size values (e.g. this script's own 224 default)
     doesn't correspond to any single real recipe's crop-size pairing; still a
     reasonable ablation, just not an official-parity baseline unless paired
     with --image_size 256.

  KEPT AS INTENTIONAL, DOCUMENTED DEVIATIONS (not "fixed", by design):
  ⚠️  Pixel-space masking instead of official's embedding-space mask_token
      substitution (prepare_tokens_with_masks in vision_transformer.py). Retained
      intentionally: models sensor occlusion / turbidity in benthic imagery. The
      new official-faithful mask CONTENT (block-wise, gated, linspace ratios) now
      drives this pixel-zeroing directly -- see `apply_pixel_masking` below --
      so this deviation composes correctly with the masking fix rather than
      fighting it. NOTE: HF's DINOv3ViTModel exposes bool_masked_pos directly,
      which performs official's real embedding-space mask_token substitution --
      swapping to that is a one-line ablation worth running against this deviation.
  ⚠️  Gram teacher lifecycle: official can periodically refresh the Gram teacher
      from the live EMA teacher mid-run (gram.rep_update/update_frequency/
      max_updates); we keep the simpler static two-stage design (load once from
      a Stage-1 checkpoint, frozen thereafter). Not implemented -- optional/advanced.
  ⚠️  proj_dim=16384 (official uses 262144 for DINO, 98304 for iBOT). Smaller
      value appropriate for ViT-S + single-GPU training.
  ⚠️  No batch-size-dependent LR scaling (official's scaling_rule: sqrt_wrt_1024
      scales LR peak/end by sqrt(global_batch_size/1024)). Not implemented --
      --learning_rate is a flat absolute value here regardless of achieved batch
      size. Fine for a single fixed-batch-size setup; just don't compare this
      script's LR values directly against official's without accounting for it.

Recommended two-stage launch:
  Stage 1 – no Gram, build a benthic-adapted teacher (~15 epochs)
    python ssl_training.py ... --use_ibot --use_koleo --output_dir ./stage1

  Stage 2 – add Gram anchoring with the Stage-1 checkpoint as the Gram teacher.
    Official uses a higher (more stable) EMA momentum during this phase (0.999
    vs 0.994 during main pretraining) -- pass --teacher_ema_tau 0.999 here.
    python ssl_training.py ... --use_ibot --use_koleo --use_gram \
        --use_gram_teacher --gram_teacher_ckpt ./stage1/benthic-ssl-best.ckpt \
        --teacher_ema_tau 0.999 \
        --resume ./stage1/benthic-ssl-best.ckpt --output_dir ./stage2
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, ExponentialLR
)
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple
import argparse
from tqdm import tqdm
import warnings
import copy
import random
from torch.nn.init import trunc_normal_
import sys
import glob
import csv

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

warnings.filterwarnings("ignore")
import colorama
colorama.init()

# ============================================================================
# LAYER-WISE LR DECAY (official dinov3/train/param_groups.py)
# ============================================================================

def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 0.98,
    num_layers: int = 12,
) -> float:
    """
    Calculate learning rate decay factor for different ViT blocks.
    Deeper layers (higher layer_id) get smaller LR multipliers.
    
    Supports both:
    - Official dinov3 repo structure: model.backbone.blocks.N
    - Hugging Face transformers structure: model.encoder.layer.N
    
    Args:
        name: parameter name from model.named_parameters()
        lr_decay_rate: base decay rate (official default 0.98, use 1.0 for no decay)
        num_layers: total number of transformer blocks in the backbone
    
    Returns:
        Multiplicative factor for learning rate of this parameter.
        Example: if lr_base=1e-3 and decay_rate=0.98, layer 11 gets lr=1e-3 * 0.98^1,
        layer 0 gets lr=1e-3 * 0.98^12.
    """
    layer_id = num_layers + 1
    
    # Embeddings and tokens get layer_id=0 (highest LR decay, lowest LR)
    if any(x in name.lower() for x in [
        "pos_embed", "patch_embed", "mask_token", "cls_token", "storage_tokens", "embeddings"
    ]):
        layer_id = 0
    # Official dinov3 repo structure: backbone.blocks.N
    elif ".blocks." in name:
        try:
            block_idx = int(name.split(".blocks.")[1].split(".")[0])
            layer_id = block_idx + 1
        except (ValueError, IndexError):
            layer_id = num_layers + 1
    # Hugging Face transformers structure: encoder.layer.N
    elif ".layer." in name:
        try:
            layer_idx = int(name.split(".layer.")[1].split(".")[0])
            layer_id = layer_idx + 1
        except (ValueError, IndexError):
            layer_id = num_layers + 1
    
    # Return decay factor: deeper layers get smaller multiplier
    return lr_decay_rate ** (num_layers + 1 - layer_id)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ============================================================================
# DATASET
# ============================================================================

class BenthicImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        extensions: Tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths: list[Path] = []
        print(f"Scanning {root_dir} for images…")
        for ext in extensions:
            self.image_paths.extend(self.root_dir.rglob(f"*{ext}"))
        self.image_paths = sorted(
            {p for p in self.image_paths if p.is_file()}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")
        print(f"✓ Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"⚠️  Failed to load {self.image_paths[idx]}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))
        return self.transform(img) if self.transform else img


# ============================================================================
# WEBDATASET LOADING (tar shards — avoids HDD random-read bottlenecks)
# ============================================================================

def count_webdataset_samples(shard_urls: list) -> int:
    """One-time sample count via a raw (non-decoding) pass over the shard tars.
    Pass --webdataset_num_samples on later runs to skip this step."""
    if not WEBDATASET_AVAILABLE:
        raise ImportError("webdataset is not installed. Install with: pip install webdataset")
    print(f"Counting samples across {len(shard_urls)} shard(s) (one-time, sequential read)...")
    count = 0
    for _ in wds.WebDataset(shard_urls, shardshuffle=False):
        count += 1
    print(f"✓ Counted {count} samples")
    return count


class WebDatasetImageTransform:
    """Extracts the decoded image from a WebDataset sample dict and applies the
    multi-crop transform. Handles two tar-member-key conventions, since
    BenthicNet's constituent per-dataset tars aren't all written the same way:
    the standard short-extension key (jpg/jpeg/png), and tar members whose key
    is the *original, full filename* instead (e.g. '0mbowensislandsw (20).jpg').
    .decode('pil') still decodes the latter correctly (it matches on the
    trailing extension), it just leaves the odd key name in place.

    Module-level (not a nested closure) so it can be pickled by Windows' 'spawn'
    multiprocessing start method when DataLoader num_workers > 0.
    """
    _META_KEYS = {'__key__', '__url__', '__local_path__', '__corrupted__', '__worker__', '__rank__'}
    _SHORT_KEYS = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
    _IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    def __init__(self, transform):
        self.transform = transform

    def _extract_image(self, sample):
        for key in self._SHORT_KEYS:
            if key in sample:
                return sample[key]
        for key, value in sample.items():
            if key in self._META_KEYS:
                continue
            if isinstance(value, Image.Image):
                return value
            if isinstance(key, str) and key.lower().endswith(self._IMAGE_EXTS):
                return value
        raise ValueError(
            f"No decodable image found in sample (key={sample.get('__key__')}, "
            f"url={sample.get('__url__')}); available keys: {list(sample.keys())}"
        )

    def __call__(self, sample):
        try:
            img = self._extract_image(sample)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return self.transform(img)
        except Exception as e:
            key = sample.get('__key__', '?') if isinstance(sample, dict) else '?'
            url = sample.get('__url__', '?') if isinstance(sample, dict) else '?'
            print(f"⚠️  Skipping unreadable WebDataset sample (key={key}, shard={url}): {e}")
            return None


class _NotNone:
    """Picklable replacement for `lambda x: x is not None`, used to filter out
    samples WebDatasetImageTransform couldn't decode. A lambda defined inside a
    function hits the same Windows 'spawn' pickling problem as a nested class."""
    def __call__(self, x):
        return x is not None


def build_webdataset_loader(
    shard_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    shard_pattern: str = "shard-*.tar",
    shuffle_buffer: int = 2000,
    num_samples: Optional[int] = None,
    distributed: bool = False,
    seed: int = 0,
):
    """Build a loader reading directly from tar shards instead of scanning
    individual image files. Returns (loader, steps_per_epoch)."""
    if not WEBDATASET_AVAILABLE:
        raise ImportError("webdataset is not installed. Install with: pip install webdataset")

    shard_urls_abs = sorted(glob.glob(os.path.join(shard_dir, shard_pattern)))
    if not shard_urls_abs:
        raise FileNotFoundError(
            f"No shards matching '{shard_pattern}' found in {shard_dir}. "
            f"Check --shard_dir/--shard_pattern, and make sure the resharding step has run."
        )
    print(f"Found {len(shard_urls_abs)} shard file(s) in {shard_dir}")

    cwd = os.getcwd()
    try:
        shard_urls = [os.path.relpath(p, start=cwd) for p in shard_urls_abs]
    except ValueError as e:
        raise RuntimeError(
            f"Could not express shard paths relative to the current working directory "
            f"({cwd}). On Windows this happens when --shard_dir is on a different drive "
            f"than the one you're running the script from. Run the script from the same "
            f"drive as --shard_dir, or move the shards onto the same drive. "
            f"Original error: {e}"
        ) from e

    if num_samples is None:
        num_samples = count_webdataset_samples(shard_urls)
    if num_samples < batch_size:
        raise ValueError(f"Only {num_samples} samples found, which is fewer than batch_size={batch_size}")

    steps_per_epoch = num_samples // batch_size  # mirrors drop_last=True

    nodesplitter = wds.split_by_node if distributed else wds.single_node_only

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True, nodesplitter=nodesplitter, seed=seed)
        .repeat()
        .shuffle(shuffle_buffer)
        .decode("pil", handler=wds.warn_and_continue)
        .map(WebDatasetImageTransform(transform))
        .select(_NotNone())
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = wds.WebLoader(dataset, **loader_kwargs).with_epoch(steps_per_epoch).with_length(steps_per_epoch)
    return loader, steps_per_epoch


# ============================================================================
# AUGMENTATION
# ============================================================================

class MultiCropTransform:
    """Wrapper that returns a list of global + local crops.

    global_aug can be either a single transform (applied to every global crop,
    the original behavior) or a list of transforms, one per "slot" -- if there
    are more global crops than transforms in the list, the last transform is
    reused for the remaining crops. This is needed for the official DINOv3
    recipe, where the first global crop gets blur-only treatment and later
    global crops get light-blur + solarize.
    """
    def __init__(self, num_global, num_local, global_aug, local_aug):
        self.num_global = num_global
        self.num_local = num_local
        self.global_augs = global_aug if isinstance(global_aug, (list, tuple)) else [global_aug]
        self.local_aug = local_aug

    def __call__(self, img):
        crops = []
        for i in range(self.num_global):
            t = self.global_augs[i] if i < len(self.global_augs) else self.global_augs[-1]
            crops.append(t(img))
        crops.extend(self.local_aug(img) for _ in range(self.num_local))
        return crops


class Solarization:
    """Solarization as used in the official DINO/DINOv2/DINOv3 augmentation recipe.
    Applied to a PIL image (i.e. before ToTensor), with probability p."""
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            from PIL import ImageOps
            return ImageOps.solarize(img)
        return img


class RandomDiscreteRotation:
    """Rotate by a random multiple of 90 degrees (0/90/180/270), applied to a
    PIL image (i.e. before ToTensor). FIX: replaces continuous
    T.RandomRotation, which -- when applied after RandomResizedCrop, on an
    already-square crop -- leaves triangular corner regions with no source
    pixels that torchvision fills with solid black by default. Those black
    wedges become a permanent, semantically meaningless artifact the model
    can learn to detect as a trivial shortcut. A square rotated by an exact
    multiple of 90 degrees maps perfectly onto itself with zero fill needed,
    while still giving full 4-way "no canonical up/down" orientation
    coverage -- the actual stated goal of this augmentation."""
    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img = img.rotate(angle, expand=False)
        return img


class RandomVignette:
    """Simulates uneven artificial lighting (ROV floodlight/strobe falloff,
    including off-axis hotspots) as a multiplicative radial gradient
    centered at a random point -- not necessarily image center. Operates on
    a [C,H,W] float tensor in [0,1] (i.e. after ToTensor, before Normalize)."""
    def __init__(self, p: float = 0.5, strength_range: Tuple[float, float] = (0.25, 0.6)):
        self.p = p
        self.strength_range = strength_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        _, H, W = x.shape
        cy = random.uniform(0.2, 0.8) * H
        cx = random.uniform(0.2, 0.8) * W
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = math.sqrt(H ** 2 + W ** 2) / 2
        dist = (dist / max_dist).clamp(0, 1)
        strength = random.uniform(*self.strength_range)
        if random.random() < 0.5:
            gradient = 1.0 - strength * dist          # darker at edges (typical falloff)
        else:
            gradient = 1.0 - strength * (1.0 - dist)  # darker at center (off-axis hotspot)
        return (x * gradient.unsqueeze(0)).clamp(0.0, 1.0)


class RandomTurbidity:
    """Simulates water-column turbidity/haze via the standard underwater/
    atmospheric-scattering blend: I' = I*t + A*(1-t), where A is a
    blue-green-tinted veiling-light color (scattered ambient light) and t
    is the transmission (lower t = more turbid/hazy). Distinct from blur --
    this reduces contrast and shifts color rather than softening edges.
    Operates on a [C,H,W] float tensor in [0,1]."""
    def __init__(self, p: float = 0.4, transmission_range: Tuple[float, float] = (0.5, 0.95)):
        self.p = p
        self.transmission_range = transmission_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        t = random.uniform(*self.transmission_range)
        veil = torch.tensor(
            [random.uniform(0.05, 0.25), random.uniform(0.25, 0.5), random.uniform(0.3, 0.55)],
            dtype=x.dtype,
        ).view(3, 1, 1)
        return (x * t + veil * (1 - t)).clamp(0.0, 1.0)


class RandomChannelAttenuation:
    """Simulates depth-dependent wavelength attenuation: red light is
    absorbed fastest underwater, green next, blue slowest. Stochastically
    scales each channel independently, biased so red attenuates most
    aggressively on average -- a rough proxy for varying simulated depth.
    Distinct from ColorJitter's hue/saturation, which don't independently
    attenuate per-channel in a physically-motivated direction. Operates on
    a [C,H,W] float tensor in [0,1]."""
    def __init__(
        self,
        p: float = 0.5,
        r_range: Tuple[float, float] = (0.3, 1.0),
        g_range: Tuple[float, float] = (0.6, 1.0),
        b_range: Tuple[float, float] = (0.75, 1.0),
    ):
        self.p = p
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        scale = torch.tensor(
            [random.uniform(*self.r_range), random.uniform(*self.g_range), random.uniform(*self.b_range)],
            dtype=x.dtype,
        ).view(3, 1, 1)
        return (x * scale).clamp(0.0, 1.0)


class RandomMarineSnow:
    """Simulates suspended particulate matter ('marine snow') drifting in
    the water column -- small bright specks scattered across the frame.
    (Best-guess implementation for what was requested as a 'gravity'
    effect -- flag if a different effect was intended, e.g. directional
    sediment drift from current rather than a static speckle field, and
    this can be adjusted.) Operates on a [C,H,W] float tensor in [0,1]."""
    def __init__(self, p: float = 0.3, density_range: Tuple[float, float] = (0.0005, 0.003)):
        self.p = p
        self.density_range = density_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        _, H, W = x.shape
        density = random.uniform(*self.density_range)
        n_particles = int(density * H * W)
        if n_particles == 0:
            return x
        ys = torch.randint(0, H, (n_particles,))
        xs = torch.randint(0, W, (n_particles,))
        brightness = torch.empty(n_particles).uniform_(0.6, 1.0)
        out = x.clone()
        out[:, ys, xs] = brightness.unsqueeze(0)
        return out


class MarinePhysicsAugment:
    """Bundles the underwater-physics-motivated augmentations (lighting
    vignette, turbidity/haze, per-channel attenuation, marine snow) into one
    callable. Applied after ToTensor and before Normalize, tensor-space only
    -- kept separate from the PIL-space color_and_distortions pipeline since
    these operate on [0,1] float tensors, not PIL images."""
    def __init__(
        self,
        vignette_p: float = 0.5,
        turbidity_p: float = 0.4,
        channel_p: float = 0.5,
        snow_p: float = 0.3,
    ):
        self.vignette = RandomVignette(p=vignette_p)
        self.turbidity = RandomTurbidity(p=turbidity_p)
        self.channel_atten = RandomChannelAttenuation(p=channel_p)
        self.snow = RandomMarineSnow(p=snow_p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vignette(x)
        x = self.turbidity(x)
        x = self.channel_atten(x)
        x = self.snow(x)
        return x

BENTHIC_MEAN = (0.359, 0.413, 0.386)
BENTHIC_STD = (0.219, 0.215, 0.209)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transform(
    image_size: int = 224,
    num_global_crops: int = 2,
    num_local_crops: int = 8,
    marine_aug: bool = False,
    underwater_orientation_aug: bool = True,
    official_dinov3_aug: bool = False,
    benthic_norm: bool = False,
    local_crop_size: Optional[int] = None,
):
    """
    underwater_orientation_aug: when True (default), adds RandomVerticalFlip + full
    180-degree RandomRotation, since seafloor imagery has no canonical up/down
    orientation. Set False for an ablation using only standard, orientation-preserving
    augmentation (horizontal flip only).

    official_dinov3_aug: when True, ignores marine_aug/underwater_orientation_aug and
    builds the exact official DINO/DINOv2/DINOv3 multi-crop recipe: bicubic-interpolated
    resized crop, NO horizontal flip (every real pretraining recipe sets
    horizontal_flips: false -- only the base placeholder config and an unrelated
    linear-probe eval config enable it), RandomApply(ColorJitter, p=0.8),
    RandomGrayscale(p=0.2), and per-crop Gaussian blur (kernel_size=9, official's
    constant across all crop sizes) / solarization probabilities (first global crop:
    blur p=1.0; later global crops: blur p=0.1 + solarize p=0.2; local crops: blur
    p=0.5). No vertical flip, rotation, or affine translate. Local crop size is
    fixed at 112 regardless of `image_size` (official's real value across
    pretrain/gram_anchor/high_res_adapt), which only pairs correctly with
    `image_size=256` (official's actual global_crops_size for pretrain/gram_anchor) --
    pass --image_size 256 alongside this flag for a genuinely faithful reproduction;
    at other image_size values this remains a reasonable ablation, just not one
    that corresponds to any single real recipe's crop-size pairing.

    marine_aug: when True, uses dedicated marine/underwater specific color, lighting,
    and turbidity augmentations (stronger underwater color jitter, gaussian blur,
    solarization/grayscale) tailored for benthic sea floor imagery.

    benthic_norm: when True, normalizes with BenthicNet-specific mean/std
    (mean=[0.359,0.413,0.386], std=[0.219,0.215,0.209]) instead of ImageNet
    defaults. Also drives the RandomAffine fill color so exposed-canvas regions
    blend with whichever normalization is active instead of defaulting to an
    ImageNet-toned fill. Ignored when official_dinov3_aug is set (that path has
    no RandomAffine and always normalizes with `normalize`, so it still honors
    this flag through norm_mean/norm_std).

    local_crop_size: overrides the local-crop pixel size for the marine_aug and
    default paths (ignored when official_dinov3_aug is set, which has its own
    fixed 112px local crop). Defaults to 96px if omitted -- override for
    non-default image_size runs (e.g. 384) to preserve the intended
    local:global crop-size ratio. Should stay divisible by the model's patch
    size (16).
    """
    norm_mean = BENTHIC_MEAN if benthic_norm else IMAGENET_MEAN
    norm_std = BENTHIC_STD if benthic_norm else IMAGENET_STD
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    affine_fill = tuple(round(m * 255) for m in norm_mean)
    local_size = local_crop_size or 96   # was a bare 96 literal in two places below, untethered from --image_size

    if official_dinov3_aug:
        # FIX: official's custom GaussianBlur hardcodes kernel_size=9 for every crop
        # (global or local) -- it does not scale kernel size with crop resolution.
        OFFICIAL_BLUR_KERNEL = 9
        # FIX: official explicitly passes interpolation=BICUBIC to RandomResizedCrop;
        # torchvision's default (silently used before) is bilinear.
        BICUBIC = T.InterpolationMode.BICUBIC

        flip_and_color_jitter = T.Compose([
            # FIX: horizontal_flips is False in every real SSL pretraining recipe
            # (pretrain/gram_anchor/high_res_adapt/distilled all set it explicitly
            # to false). True only appears in the base placeholder config and in
            # vitl_im1k_lin834.yaml, which is a downstream linear-probe eval config,
            # not a pretraining recipe -- so it's not the value actually trained with.
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ])
        global_transfo1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.32, 1.0), ratio=(0.75, 1.333), interpolation=BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([T.GaussianBlur(kernel_size=OFFICIAL_BLUR_KERNEL, sigma=(0.1, 2.0))], p=1.0),
            normalize,
        ])
        global_transfo2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.32, 1.0), ratio=(0.75, 1.333), interpolation=BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([T.GaussianBlur(kernel_size=OFFICIAL_BLUR_KERNEL, sigma=(0.1, 2.0))], p=0.1),
            Solarization(p=0.2),
            normalize,
        ])
        local_aug = T.Compose([
            T.RandomResizedCrop(112, scale=(0.05, 0.32), ratio=(0.75, 1.333), interpolation=BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([T.GaussianBlur(kernel_size=OFFICIAL_BLUR_KERNEL, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
        return MultiCropTransform(num_global_crops, num_local_crops, [global_transfo1, global_transfo2], local_aug)

    if marine_aug:
        BICUBIC = T.InterpolationMode.BICUBIC
        to_tensor = T.ToTensor()
        norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        physics_aug = MarinePhysicsAugment()  # vignette + turbidity + channel attenuation + marine snow

        marine_orientation_transforms = [T.RandomHorizontalFlip(p=0.5)]
        if underwater_orientation_aug:
            marine_orientation_transforms.extend([
                T.RandomVerticalFlip(p=0.3),
                RandomDiscreteRotation(),   # FIX: was T.RandomRotation(degrees=180), which
                                            # leaves black-fill triangular corners on every crop.
            ])
        marine_geo = T.Compose(marine_orientation_transforms)

        color_and_distortions = T.Compose([
            marine_geo,
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.15)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ])

        global_transfo1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.32, 1.0), ratio=(0.75, 1.333), interpolation=BICUBIC),
            color_and_distortions,
            T.RandomApply([T.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.8),
            to_tensor,
            physics_aug,
            norm,
        ])
        global_transfo2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.32, 1.0), ratio=(0.75, 1.333), interpolation=BICUBIC),
            color_and_distortions,
            T.RandomApply([T.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.3),
            Solarization(p=0.25),
            to_tensor,
            physics_aug,
            norm,
        ])
        local_aug = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.32), ratio=(0.75, 1.333), interpolation=BICUBIC),
            color_and_distortions,
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            to_tensor,
            physics_aug,
            norm,
        ])
        return MultiCropTransform(num_global_crops, num_local_crops, [global_transfo1, global_transfo2], local_aug)

    global_aug_list = [
        T.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if underwater_orientation_aug:
        global_aug_list.append(T.RandomVerticalFlip(p=0.3))
        global_aug_list.append(RandomDiscreteRotation())   # was T.RandomRotation(degrees=180)
    global_aug_list.extend([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=affine_fill),  # was missing fill
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        normalize,
    ])
    global_aug = T.Compose(global_aug_list)

    local_aug_list = [
        T.RandomResizedCrop(local_size, scale=(0.05, 0.4), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if underwater_orientation_aug:
        local_aug_list.append(T.RandomVerticalFlip(p=0.3))
        local_aug_list.append(RandomDiscreteRotation()) 
    local_aug_list.extend([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        normalize,
    ])
    local_aug = T.Compose(local_aug_list)
    return MultiCropTransform(num_global_crops, num_local_crops, global_aug, local_aug)


# ============================================================================
# MASKING — faithful port of official dinov3/data/masking.py +
# dinov3/data/collate.py's collate_data_and_cast batch-level gating.
# ============================================================================

class MaskingGenerator:
    """BEiT-style block-wise masking generator over a 2D patch grid.

    This is a near line-for-line port of the official MaskingGenerator: random
    rectangular blocks (log-uniform aspect ratio) are placed and grown until the
    requested patch count is reached; any shortfall is filled in by randomly
    flipping additional individual (non-contiguous) patches so the exact target
    count is always hit. This produces spatially contiguous masked regions,
    unlike an i.i.d. scatter of individually-chosen patches.
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        self.height = height
        self.width = width
        self.num_patches = height * width
        self.min_num_patches = min_num_patches
        self.max_num_patches = self.num_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top:top + h, left:left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        mask = np.zeros(shape=(self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = min(num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return self._complete_mask_randomly(mask, num_masking_patches)

    @staticmethod
    def _complete_mask_randomly(mask: np.ndarray, num_masking_patches: int) -> np.ndarray:
        shape = mask.shape
        m2 = mask.flatten()
        deficit = int(num_masking_patches - m2.sum())
        if deficit > 0:
            candidates = np.where(~m2)[0]
            deficit = min(deficit, len(candidates))  # defensive; official assumes this never binds
            if deficit > 0:
                to_add = np.random.choice(candidates, size=deficit, replace=False)
                m2[to_add] = True
        return m2.reshape(shape)


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=bias)])
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """DINOv3 head. trunc_normal_ init matches official. weight_norm on the last
    layer is OFF by default -- grepping the current official repo shows
    weight_norm doesn't appear anywhere in it; DINOHead's last layer is a plain
    nn.Linear. This differs from DINOv2, which did use weight_norm there. Kept
    as an opt-in (use_weight_norm=True) for anyone deliberately targeting the
    older DINOv2-style head.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.mlp = _build_mlp(
            max(nlayers, 1), in_dim, bottleneck_dim,
            hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init_weights)

        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.last_layer = weight_norm(self.last_layer)
            self.last_layer.weight_g.data.fill_(1)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def _ce(t, s, temp):
    """Cross-entropy between teacher probs t and student logits s."""
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic regulariser.
    Applied to PRE-HEAD backbone CLS tokens (matching official DINOv3).
    Explicit float32 cast prevents NaN from bfloat16 / float16 pdist.
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-8)

    def _nearest_neighbours(self, x: torch.Tensor) -> torch.Tensor:
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: n + 1].fill_(-1)          # zero diagonal
        return torch.max(dots, dim=1).indices

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            x = F.normalize(x.float(), p=2, dim=-1, eps=eps)
            idx = self._nearest_neighbours(x)
            d = self.pdist(x, x[idx])
            # Add clamp just in case exact identical arrays bypass pdist stability
            return -torch.log(d.clamp(min=eps)).mean()


class GramLoss(nn.Module):
    """
    Gram matrix regularisation loss.
    img_level=True -> per-image Gram matrices (matches official default).

    FIX: remove_neg / remove_only_teacher_neg now both default to False. Every
    real training recipe that turns Gram on (dinov3_vit7b16_gram_anchor.yaml,
    dinov3_vit7b16_high_res_adapt.yaml, dinov3_vitl16_lvd1689m_distilled.yaml)
    sets both flags False -- i.e. plain MSE on the (optionally L2-normalized)
    Gram matrices, no special-casing of negative similarities. Both flags are
    kept available for experimentation.

    Note on official's own remove_only_teacher_neg branch, for anyone comparing
    line-by-line: as literally written upstream (`target_sim[target_sim<0]=0`
    followed by `student_sim[(student_sim<0)&(target_sim<0)]=0`), the second
    line checks target_sim *after* it was already zeroed in place on the first
    line, so that check is always False and the student_sim line is a no-op --
    official's real effective behavior in that branch is just "zero the
    teacher's negatives, leave student untouched everywhere". Our
    implementation below (when this flag is turned on) intentionally does
    something more useful instead: it excludes positions where both teacher and
    student already agree the patches are anti-correlated from contributing any
    loss at all, only pulling the student down when it disagrees (positive)
    with a teacher-negative anchor. Since no shipped recipe actually enables
    this branch, exact bug-for-bug parity didn't seem worth chasing -- flagged
    here so it's a documented choice rather than a silent divergence.
    """

    def __init__(
        self,
        apply_norm: bool = True,
        img_level: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor):
        s = student_feats.float()
        t = teacher_feats.float()

        if self.apply_norm:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        if self.img_level:
            # [B, P, D] → [B, P, P] per-image Gram matrices
            t_sim = torch.matmul(t, t.transpose(-1, -2))
            s_sim = torch.matmul(s, s.transpose(-1, -2))
        else:
            t = t.flatten(0, 1)
            s = s.flatten(0, 1)
            t_sim = torch.mm(t, t.t())
            s_sim = torch.mm(s, s.t())

        if self.remove_neg:
            t_sim = F.relu(t_sim)
            s_sim = F.relu(s_sim)
        elif self.remove_only_teacher_neg:
            teacher_neg_mask = t_sim < 0
            t_sim = F.relu(t_sim)
            s_sim = torch.where(teacher_neg_mask, s_sim.clamp(min=0.0), s_sim)

        return self.mse(s_sim, t_sim)


class _SinkhornKnopp(nn.Module):
    """Sinkhorn-Knopp assignment used for both DINO and iBOT teacher targets."""

    @torch.no_grad()
    def forward(self, logits, temp, n_sink=3, B_override=None):
        logits = logits.float() / temp
        # Subtract max along classification/feature dimension to avoid inf exp overflow issues
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        Q = torch.exp(logits).t()   # [K, N]
        K, N = Q.shape
        B = int(B_override.item()) if isinstance(B_override, torch.Tensor) else (B_override or N)
        Q /= Q.sum()
        for _ in range(n_sink):
            Q /= Q.sum(dim=1, keepdim=True) * K
            Q /= Q.sum(dim=0, keepdim=True) * B
        return (Q * B).t()                 # [N, K]


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self._sk = _SinkhornKnopp()
        # Vestigial: kept only for parity with official's own unused softmax-centering
        # path / checkpoint-shape compatibility. Never referenced by
        # sinkhorn_knopp_teacher below -- see the note there.
        self.register_buffer("center", torch.zeros(1, out_dim))

    def to(self, device):
        super().to(device)
        self.center = self.center.to(device)
        return self

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, logits, teacher_temp, n_iterations=3):
        # FIX: official's sinkhorn_knopp_teacher never subtracts self.center -- that
        # buffer only feeds the alternate softmax_center_teacher path, which upstream
        # asserts is never used (ssl_meta_arch.py: `assert cfg.train.centering ==
        # "sinkhorn_knopp"`). Sinkhorn's own row/column normalization is the only
        # "centering" that officially happens. Previously this subtracted `center`,
        # which doesn't match official behavior.
        return self._sk(logits, teacher_temp, n_sink=n_iterations)

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        # student_logits: [S, B, K]   teacher_probs: [T, B, K]
        S, B, K = student_logits.shape
        T = teacher_probs.shape[0]
        s = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("sbk,tbk->", s, teacher_probs)
            return loss / (B * S * T)
        else:
            loss = -torch.einsum("sbk,tbk->st", s, teacher_probs)   # [S, T]
            n = min(S, T)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(n))
            return loss.sum() / (B * S * T - B * n)

    @torch.no_grad()
    def update_center(self, teacher_global):
        # Vestigial -- not called anywhere in DINOTrainer now (see note on
        # sinkhorn_knopp_teacher above). Kept only so any code/checkpoints that
        # reference `center` don't break.
        batch_center = teacher_global.mean(dim=0, keepdim=True)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self._sk = _SinkhornKnopp()
        self.register_buffer("center", torch.zeros(1, patch_out_dim))  # vestigial, see DINOLoss note

    def to(self, device):
        super().to(device)
        self.center = self.center.to(device)
        return self

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, logits, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        # FIX: see DINOLoss.sinkhorn_knopp_teacher -- no self.center subtraction here either.
        return self._sk(logits, teacher_temp, n_sink=n_iterations, B_override=n_masked_patches_tensor)

    def forward(self, student_patches, teacher_patches, masks_weight=None, n_samples=None):
        """FIX: official's forward_masked reweights each masked patch's loss by
        1/(number of masked patches in that sample), then divides the sum by the
        number of sample-slots (not the number of masked patches) -- so every
        sample-slot contributes equally to the loss regardless of how many patches
        it had masked. Our earlier plain `.mean()` over all masked patches pooled
        together implicitly gave more weight to heavily-masked samples, since our
        mask ratio varies per sample. `masks_weight`/`n_samples` are produced by
        DINOTrainer._generate_batch_masks; forward() falls back to a plain mean
        if they're not supplied (e.g. for quick unit tests).
        """
        loss = _ce(teacher_patches, student_patches, self.student_temp)
        if masks_weight is not None:
            loss = loss * masks_weight
            denom = float(n_samples) if n_samples else max(float(loss.shape[0]), 1.0)
            return -loss.sum() / denom
        return -loss.mean()

    @torch.no_grad()
    def update_center(self, teacher_patches):
        # Vestigial, see DINOLoss.update_center note.
        batch_center = teacher_patches.mean(dim=0, keepdim=True)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))


# ============================================================================
# TEACHER TEMPERATURE SCHEDULE
# ============================================================================

def _make_temp_schedule(target_temp, warmup_epochs, iter_per_epoch):
    return {
        "warmup_start": 0.04,
        "target": target_temp,
        "warmup_iters": warmup_epochs * iter_per_epoch,
    }


def _get_temp(schedule, current_iter):
    wi = schedule["warmup_iters"]
    if current_iter >= wi:
        return schedule["target"]
    p = current_iter / max(1, wi)
    return schedule["warmup_start"] + p * (schedule["target"] - schedule["warmup_start"])


# ============================================================================
# GRAM TEACHER INIT HELPER
# ============================================================================

def _load_gram_teacher(trainer, ckpt_path: Optional[str], device):
    """
    Load Gram teacher from a benthic-adapted Stage-1 checkpoint.
    Without this, the Gram loss pulls student features toward ImageNet-style
    correlations in early epochs (counter-productive for domain adaptation).
    """
    if not ckpt_path or not Path(ckpt_path).is_file():
        trainer.logger.warning(
            "No gram_teacher_ckpt supplied — Gram teacher starts from the "
            "ImageNet-pretrained init copy. This diverges from the official "
            "two-stage training strategy. Run Stage 1 without Gram (~15 epochs)"
            " then resume with --gram_teacher_ckpt pointing to that checkpoint."
        )
        return

    trainer.logger.info(f"Loading Gram teacher from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    miss, unexp = trainer.gram_teacher.load_state_dict(sd, strict=False)
    if miss:
        trainer.logger.warning(f"Gram teacher missing keys: {miss}")
    if unexp:
        trainer.logger.warning(f"Gram teacher unexpected keys: {unexp}")
    trainer.gram_teacher.requires_grad_(False)
    trainer.gram_teacher.eval()
    trainer.logger.info("Gram teacher loaded from benthic-adapted checkpoint.")


# ============================================================================
# TRAINER
# ============================================================================

class DINOTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        output_dir: str = "./ssl_checkpoints",
        logger=None,
        use_amp: bool = True,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        base_lr: float = 1e-3,
        head_lr_mult: float = 10.0,
        lr_decay_rate: float = 1.0,
        patch_embed_lr_mult: float = 0.2,
        use_ibot: bool = True,
        proj_dim: int = 16384,
        # FIX: default changed 0.996 -> 0.994 to match the constant momentum used in
        # official's real main-pretrain recipes (dinov3_vit7b16_pretrain.yaml). Use
        # 0.999 for a Gram-anchor/high-res-adapt-style stage (their recipes use that
        # higher, more-stable value). See momentum_schedule below for the ramp option.
        teacher_ema_tau: float = 0.994,
        teacher_ema_tau_final: float = 1.0,
        momentum_schedule: str = "constant",  # "constant" (official real recipes) | "cosine" (DINOv2-style ramp, old default here)
        num_global_crops: int = 2,
        num_local_crops: int = 8,
        use_koleo: bool = True,
        koleo_loss_weight: float = 1e-4,
        use_gram: bool = True,
        gram_loss_weight: float = 1.0,
        gram_img_level: bool = True,
        gram_remove_neg: bool = False,
        gram_remove_only_teacher_neg: bool = False,
        lr_schedule: str = "cosine_warmup",
        teacher_temp_base: float = 0.07,
        use_gram_teacher: bool = False,
        gram_teacher_ckpt: Optional[str] = None,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        mask_sample_probability: float = 0.5,
        apply_pixel_masking: bool = True,
        weight_decay_start: float = 0.04,
        weight_decay_end: float = 0.4,
        wd_schedule: str = "constant",  # "constant" (official real recipes) | "cosine" (old default here)
        dino_head_weight_norm: bool = False,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        log_every_n_steps: int = 1,
    ):
        # ---- Student / EMA teacher ----
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        # ---- Static Gram teacher ----
        self.use_gram_teacher = use_gram_teacher
        if use_gram_teacher:
            self.gram_teacher = copy.deepcopy(self.teacher)
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        else:
            self.gram_teacher = None

        self.train_loader = train_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.use_amp = use_amp
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.head_lr_mult = head_lr_mult
        self.lr_decay_rate = lr_decay_rate
        self.patch_embed_lr_mult = patch_embed_lr_mult
        self.use_ibot = use_ibot
        self.proj_dim = proj_dim
        self.teacher_ema_tau = teacher_ema_tau
        self.teacher_ema_tau_final = teacher_ema_tau_final
        self.momentum_schedule = momentum_schedule
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.use_koleo = use_koleo
        self.koleo_loss_weight = koleo_loss_weight
        self.use_gram = use_gram
        self._base_gram_weight = gram_loss_weight
        self.gram_loss_weight = gram_loss_weight
        self.gram_img_level = gram_img_level
        self.dino_head_weight_norm = dino_head_weight_norm
        self.lr_schedule = lr_schedule
        self.current_iter = 0

        # Masking config (see MaskingGenerator / _generate_batch_masks below).
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_sample_probability = mask_sample_probability
        self.apply_pixel_masking = apply_pixel_masking
        self._mask_gen_cache: dict[Tuple[int, int], MaskingGenerator] = {}

        # Weight-decay schedule config.
        self.weight_decay_start = weight_decay_start
        self.weight_decay_end = weight_decay_end
        self.wd_schedule = wd_schedule

        # Gram ramp-up window (set to start_epoch on resume)
        self.gram_ramp_start_epoch = 0
        self.gram_ramp_end_epoch = 5

        self.teacher_temp_schedule = _make_temp_schedule(
            teacher_temp_base, warmup_epochs, len(train_loader)
        )

        # ---- Optimiser ----
        # FIX: betas exposed via adam_beta1/adam_beta2 args instead of hardcoded.
        # 0.99 for beta2 is NOT a universal "official DINOv3" value -- it's specific
        # to the 7B-scale pretrain/gram_anchor/high_res_adapt recipes. The base
        # ssl_default_config.yaml, the distillation recipe, and the linear-probe
        # recipe (vitl_im1k_lin834.yaml) all use the standard AdamW default 0.999.
        self.optimizer = AdamW(
            self._param_groups(),
            lr=base_lr,
            weight_decay=weight_decay_start,
            betas=(adam_beta1, adam_beta2),
        )
        self._setup_lr_scheduler()

        # GradScaler should be bound any time precision uses torch.float16, regardless of fp32 parameter layout
        dtype = torch.bfloat16 if next(self.student.parameters()).dtype == torch.bfloat16 else torch.float16
        use_f16 = self.use_amp and (dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler() if use_f16 else None

        # ---- Loss modules ----
        self.dino_loss = DINOLoss(proj_dim).to(device)
        self.ibot_patch_loss = (
            iBOTPatchLoss(proj_dim).to(device) if use_ibot else None
        )
        self.koleo_loss_fn = KoLeoLoss() if use_koleo else None
        self.gram_loss_fn = (
            GramLoss(
                img_level=gram_img_level,
                remove_neg=gram_remove_neg,
                remove_only_teacher_neg=gram_remove_only_teacher_neg,
            )
            if use_gram else None
        )

        self.dino_loss_weight = 1.0
        self.ibot_loss_weight = 1.0

        self.train_losses: list[float] = []
        self.best_loss = float("inf")
        self.best_knn_acc = -1.0  # For k-NN validation (ranges 0.0-1.0, init to -1 to ensure first probe is "best")
        self._last_loss_components = {}
        self.loss_history_csv = self.output_dir / "loss_history.csv"

        # ---- Experiment tracking (wandb / TensorBoard) ----
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.use_wandb = use_wandb
        if use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("--use_wandb was set but wandb is not installed "
                                 "(pip install wandb); continuing without it.")
            self.use_wandb = False
        self.tb_writer = None
        if use_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                self.logger.warning("--use_tensorboard was set but tensorboard is not installed "
                                     "(pip install tensorboard); continuing without it.")
            else:
                tb_dir = tensorboard_dir or str(self.output_dir / "tensorboard")
                self.tb_writer = SummaryWriter(tb_dir)
                self.logger.info(f"TensorBoard logging to: {tb_dir}")

        # Load Gram teacher from checkpoint (after gram_teacher exists)
        if use_gram_teacher:
            _load_gram_teacher(self, gram_teacher_ckpt, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _param_groups(self):
        """Build parameter groups with layer-wise LR decay.
        
        Implements official dinov3/train/param_groups.py logic:
        - Deeper ViT layers get smaller LR (exponential decay by depth)
        - Bias/norm/gamma/fourier_w params get no weight decay (official's exact
          condition -- learned tokens (cls/mask/register) are NOT exempted, they
          only get the layer_id=0 LR-decay treatment, not a WD exemption)
        - Heads get separate (larger) LR multiplier
        
        Supports both dinov3 repo and HuggingFace model structures.
        
        Returns:
            List of param groups with specific lr and weight_decay per parameter.
        """
        # Get backbone depth for LR decay
        # Works with both official repo structure (backbone.blocks) and HF structure (encoder.layer)
        # FIX: more robust than a bare hasattr check -- falls through to
        # base_model.config for PEFT/LoRA-wrapped students where the simple
        # hasattr(self.student, "config") check could plausibly miss the
        # proxied config, and defaults to 12 (ViT-S/16's real depth) rather
        # than 0 if truly nothing is found. Defaulting to 0 previously turned
        # layer-wise decay into an exponential LR *boost* for deep blocks
        # (0.98**-11 ~= 1.25) rather than a decay -- 12 fails safe instead.
        config = getattr(self.student, "config", None)
        if config is None and hasattr(self.student, "base_model"):
            config = getattr(self.student.base_model, "config", None)
        n_blocks = getattr(config, "num_hidden_layers", 12) or 12
        
        # FIX: matches official's param_groups.py exact condition:
        #   name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name
        # Previous revision incorrectly added cls_token/mask_token/register_tokens/beta
        # to this list -- official does NOT exempt learned tokens from weight decay.
        # The len(p.shape) == 1 fallback is kept as a safety net for legitimately-1D
        # params that don't literally contain "gamma" in their name (e.g. HF's
        # LayerScale parameter is named "lambda1"). It does NOT accidentally catch
        # cls_token/mask_token/register_tokens since those are 3D tensors
        # (shape (1, 1, D) or (1, N, D) in the HF port), not 1D.
        no_decay_keywords = ["bias", "norm", "gamma", "fourier_w"]
        
        # Collect params with metadata
        param_specs = []  # List[{params: [p], name: str, lr_mult: float, wd: float}]
        
        for name, p in self.student.named_parameters():
            if not p.requires_grad or not p.is_leaf:
                continue
            
            # Compute LR multiplier
            if "dino_head" in name or "ibot_head" in name:
                # Head gets larger LR
                lr_mult = self.head_lr_mult
            else:
                # Backbone: use layer-wise decay
                lr_mult = get_vit_lr_decay_rate(
                    name, 
                    lr_decay_rate=self.lr_decay_rate,
                    num_layers=n_blocks
                )
                # Apply optional patch_embed LR multiplier on top of layer decay
                if "patch_embed" in name.lower():
                    lr_mult *= self.patch_embed_lr_mult
            
            # Compute weight decay multiplier
            has_no_decay = any(nd in name.lower() for nd in no_decay_keywords) or len(p.shape) == 1
            wd_mult = 0.0 if has_no_decay else 1.0
            
            param_specs.append({
                "params": [p],
                "name": name,
                "lr_mult": lr_mult,
                "wd_mult": wd_mult,
            })
        
        # Group by (lr_mult, wd_mult) to minimize number of param groups
        # (PyTorch optimizers are faster with fewer groups)
        groups_dict = {}  # {(lr_mult, wd_mult): {params: [...], names: [...]}}
        for spec in param_specs:
            key = (spec["lr_mult"], spec["wd_mult"])
            if key not in groups_dict:
                groups_dict[key] = {"params": [], "names": []}
            groups_dict[key]["params"].extend(spec["params"])
            groups_dict[key]["names"].append(spec["name"])
        
        # Build final param groups for optimizer
        param_groups = []
        for (lr_mult, wd_mult), group_data in sorted(groups_dict.items()):
            pg = {
                "params": group_data["params"],
                "lr": self.base_lr * lr_mult,
                "weight_decay": self.weight_decay_start * wd_mult,
            }
            param_groups.append(pg)
        
        self.logger.debug(
            f"Built {len(param_groups)} param groups: "
            f"{', '.join(f'({lr_m:.3f}x LR, {wd_m:.1f}x WD)' for lr_m, wd_m in sorted(groups_dict.keys()))}"
        )
        
        return param_groups

    def _setup_lr_scheduler(self):
        """Setup LR scheduler with per-iteration stepping (not per-epoch)."""
        # Calculate total iterations
        total_iters = self.total_epochs * len(self.train_loader)
        warmup_iters = self.warmup_epochs * len(self.train_loader)
        
        s = self.lr_schedule
        if s == "cosine_warmup":
            if self.warmup_epochs > 0:
                wu = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
                main = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, total_iters - warmup_iters),
                    eta_min=self.base_lr * 0.01,
                )
                self.scheduler = SequentialLR(self.optimizer, [wu, main], milestones=[warmup_iters])
            else:
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_iters, eta_min=self.base_lr * 0.01)
        elif s == "linear_decay":
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_iters)
        elif s == "exponential_decay":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        else:
            self.scheduler = None

    def _teacher_temp(self):
        return _get_temp(self.teacher_temp_schedule, self.current_iter)

    def _get_ema_tau(self):
        if self.momentum_schedule == "constant":
            # FIX: matches the real DINOv3 recipes (dinov3_vit7b16_pretrain.yaml etc.
            # all set schedules.momentum start==peak==end, i.e. flat within a phase).
            return self.teacher_ema_tau
        # Old behavior, kept as an opt-in: DINOv2-style cosine ramp base_tau -> final_tau.
        # Mathematically this is the same shape as official's CosineScheduler, it's just
        # not what the actual shipped recipes use (they hold momentum constant per phase).
        base_tau = self.teacher_ema_tau
        final_tau = self.teacher_ema_tau_final
        total_iters = self.total_epochs * len(self.train_loader)
        progress = self.current_iter / max(1, total_iters)
        return final_tau - (final_tau - base_tau) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def _ema_update(self):
        tau = self._get_ema_tau()
        for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
            tp.data.mul_(tau).add_(sp.data * (1.0 - tau))

    # ------------------------------------------------------------------
    # MASKING
    # ------------------------------------------------------------------

    def _generate_batch_masks(self, batch_size: int, n_h: int, n_w: int, device):
        """Official semantics (dinov3/data/collate.py:collate_data_and_cast):
        only `mask_sample_probability` fraction of the batch gets a non-empty
        mask at all; the rest get an all-False (empty) mask. Ratios for the
        masked subset are linearly spaced across [mask_ratio_min, mask_ratio_max]
        (NOT Gaussian), and the assignment of which sample-slot gets which ratio
        (including zero) is shuffled. Each non-empty mask is a BEiT-style
        contiguous block placement via MaskingGenerator, not a random scatter.

        NOTE: official pools this budget across BOTH global crops jointly
        (B = 2*batch_size in collate_data_and_cast, drawn/shuffled once). This
        function is called once per global crop independently instead, so the
        masked-count split between crop 0 and crop 1 isn't stochastic the same
        way official's is. Converges to the same aggregate statistics; flagged
        as a known, low-impact deviation rather than fixed, since fixing it
        would require restructuring _compute_loss to generate both crops' masks
        from one shared pool.

        Returns:
            mask:   [batch_size, n_h*n_w] bool tensor on `device`.
            weight: 1D float tensor, one entry per True position in `mask` (in
                    row-major flattened order), each equal to 1/(number of True
                    positions in that sample's row). This is official's
                    `masks_weight` -- pass it straight into iBOTPatchLoss.forward
                    so heavily-masked samples don't dominate the loss.
        """
        key = (n_h, n_w)
        gen = self._mask_gen_cache.get(key)
        if gen is None:
            gen = MaskingGenerator(n_h, n_w, max_num_patches=max(1, int(0.5 * n_h * n_w)))
            self._mask_gen_cache[key] = gen

        n_patches = n_h * n_w
        n_masked = int(batch_size * self.mask_sample_probability)
        ratios = (
            torch.linspace(self.mask_ratio_min, self.mask_ratio_max, n_masked + 1).tolist()
            if n_masked > 0 else []
        )

        masks = [gen(int(n_patches * ratios[i + 1])) for i in range(n_masked)]
        masks.extend(np.zeros((n_h, n_w), dtype=bool) for _ in range(batch_size - n_masked))
        random.shuffle(masks)

        mask_t = torch.from_numpy(np.stack([m.reshape(-1) for m in masks], axis=0)).to(device)
        weight = (1.0 / mask_t.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(mask_t)[mask_t]
        return mask_t, weight

    # ------------------------------------------------------------------
    # LOSS COMPUTATION
    # ------------------------------------------------------------------

    def _compute_loss(self, crops_batch):
        B = crops_batch[0].shape[0]
        ps = self.student.config.patch_size
        G = self.num_global_crops

        # Accumulators
        s_global_post: list[torch.Tensor] = []   # post-head CLS → DINO
        s_global_pre:  list[torch.Tensor] = []   # pre-head CLS → KoLeo (kept per-crop, NOT pooled)
        s_local:       list[torch.Tensor] = []   # post-head CLS → DINO local
        s_masked_ibot: list[torch.Tensor] = []   # masked patch logits → iBOT
        s_masked_weights: list[torch.Tensor] = []  # per-position official masks_weight, paired 1:1 with s_masked_ibot
        s_patches:     list[torch.Tensor] = []   # all patch tokens → Gram

        crop_masks: list[Optional[torch.Tensor]] = []

        # ---- STUDENT FORWARD ----------------------------------------
        for idx, crop in enumerate(crops_batch):
            H, W = crop.shape[-2], crop.shape[-1]
            n_h, n_w = H // ps, W // ps
            n_patches = n_h * n_w

            if idx < G:
                if self.use_ibot:
                    mask, mask_weight = self._generate_batch_masks(B, n_h, n_w, self.device)
                else:
                    mask, mask_weight = None, None
                crop_masks.append(mask)

                if self.use_ibot and self.apply_pixel_masking and mask is not None:
                    # ⚠️  Pixel-space masking (diverges from official's embedding-space
                    #    mask_token substitution in prepare_tokens_with_masks). Retained
                    #    intentionally: models sensor occlusion/turbidity in benthic
                    #    imagery. Works correctly with any mask content (block-wise or
                    #    scattered) since it's just a multiplicative zero-out -- the
                    #    official-faithful mask generated above drives it directly, no
                    #    separate coin-flip needed (that used to double-gate against the
                    #    per-sample probability already baked into mask generation).
                    pmask = (
                        mask.view(B, n_h, n_w)
                        .repeat_interleave(ps, dim=1)
                        .repeat_interleave(ps, dim=2)
                        .unsqueeze(1)
                        .to(crop.device)
                    )
                    inp = crop * (~pmask)
                else:
                    inp = crop
                out = self.student(pixel_values=inp)
            else:
                out = self.student(pixel_values=crop)

            # Extract tokens
            cls_pre  = out.last_hidden_state[:, 0, :]              # backbone CLS (pre-head)
            cls_post = self.student.dino_head(cls_pre)              # prototype space
            patches  = out.last_hidden_state[:, -n_patches:, :]

            if idx < G:
                s_global_post.append(cls_post)
                s_global_pre.append(cls_pre)          # kept per-crop for KoLeo
                s_patches.append(patches)

                if self.use_ibot and mask is not None:
                    flat_p = patches.reshape(B * n_patches, -1)
                    flat_m = mask.reshape(B * n_patches)
                    masked = flat_p[flat_m]
                    if masked.numel() > 0:
                        s_masked_ibot.append(self.student.ibot_head(masked))
                        s_masked_weights.append(mask_weight)
            else:
                s_local.append(cls_post)

        s_global_post = torch.stack(s_global_post, dim=0)   # [G, B, K]
        s_local       = torch.stack(s_local, dim=0)          # [L, B, K]

        # ---- TEACHER FORWARD (unmasked, no grad) --------------------
        with torch.no_grad():
            t_temp = self._teacher_temp()
            t_global:  list[torch.Tensor] = []
            t_patches: list[torch.Tensor] = []
            t_masked:  list[torch.Tensor] = []

            for idx, crop in enumerate(crops_batch[:G]):
                n_patches = (crop.shape[-2] // ps) * (crop.shape[-1] // ps)
                out = self.teacher(pixel_values=crop)
                cls = out.last_hidden_state[:, 0, :]
                t_global.append(self.teacher.dino_head(cls))

                pt = out.last_hidden_state[:, -n_patches:, :]
                t_patches.append(pt)

                if self.use_ibot and crop_masks[idx] is not None:
                    flat_p = pt.reshape(B * n_patches, -1)
                    flat_m = crop_masks[idx].reshape(B * n_patches)
                    masked = flat_p[flat_m]
                    if masked.numel() > 0:
                        t_masked.append(self.teacher.ibot_head(masked))

            # DINO teacher targets via Sinkhorn-Knopp (no running-center subtraction --
            # see DINOLoss.sinkhorn_knopp_teacher note)
            t_global_stack = torch.stack(t_global, dim=0)            # [G, B, K]
            t_flat = t_global_stack.flatten(0, 1)                    # [G*B, K]
            t_centered = self.dino_loss.sinkhorn_knopp_teacher(
                t_flat, teacher_temp=t_temp
            ).view(G, B, -1)

            # iBOT teacher targets (same: no center subtraction)
            if self.use_ibot and t_masked:
                t_masked_cat = torch.cat(t_masked, dim=0)
                t_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    t_masked_cat, teacher_temp=t_temp,
                    n_masked_patches_tensor=t_masked_cat.shape[0],
                )
            else:
                t_patch_centered = None

        # ---- LOSS SUM -----------------------------------------------

        # DINO global (ignore diagonal – each global view predicts the other)
        dino_global = self.dino_loss(s_global_post, t_centered, ignore_diagonal=True)
        # DINO local
        dino_local  = self.dino_loss(s_local, t_centered, ignore_diagonal=False)

        # Correct pair-count scaling (not a fixed 0.5/0.5 split) -- confirmed to match
        # official's ssl_meta_arch.py exactly.
        n_gg = G * (G - 1)                          # global→global pairs
        n_lg = G * self.num_local_crops              # local→global pairs
        n_total = n_gg + n_lg
        g_scale = n_gg / n_total
        l_scale = n_lg / n_total

        dino_term = self.dino_loss_weight * (g_scale * dino_global + l_scale * dino_local)
        total = dino_term

        # Per-component scalar tracking for logging/CSV curves. Uses .item() on
        # already-computed terms only -- doesn't touch the graph for `total`,
        # which is what actually gets backpropagated below in train_epoch.
        components = {"dino": dino_term.item(), "ibot": 0.0, "koleo": 0.0, "gram": 0.0}

        # iBOT (FIX: now uses official's per-sample masks_weight reweighting instead
        # of a plain .mean() over pooled masked patches)
        if self.use_ibot and s_masked_ibot and t_patch_centered is not None:
            s_pat = torch.cat(s_masked_ibot, dim=0)
            weights_cat = torch.cat(s_masked_weights, dim=0)
            ibot_term = self.ibot_loss_weight * self.ibot_patch_loss(
                s_pat, t_patch_centered, masks_weight=weights_cat, n_samples=G * B
            )
            total = total + ibot_term
            components["ibot"] = ibot_term.item()

        # KoLeo (FIX: sum of per-global-crop losses, matching official's cancelled-out
        # `sum(koleo(x) for x in crops) / n_global_crops * koleo_scale(=n_global_crops)`
        # -- previously this concatenated all global crops together first, which lets a
        # same-image crop pair become each other's nearest neighbor and blunts the
        # anti-collapse effect)
        if self.use_koleo and s_global_pre:
            koleo_term = self.koleo_loss_weight * sum(self.koleo_loss_fn(x) for x in s_global_pre)
            total = total + koleo_term
            components["koleo"] = koleo_term.item()

        # Gram
        if self.use_gram and s_patches:
            gram_src = (
                self.gram_teacher
                if (self.use_gram_teacher and self.gram_teacher is not None)
                else self.teacher
            )
            with torch.no_grad():
                tg_patches = []
                for crop in crops_batch[:G]:
                    n_p = (crop.shape[-2] // ps) * (crop.shape[-1] // ps)
                    o = gram_src(pixel_values=crop)
                    tg_patches.append(o.last_hidden_state[:, -n_p:, :])

            s_gram = torch.cat(s_patches, dim=0)
            t_gram = torch.cat(tg_patches, dim=0)
            gram_term = self.gram_loss_weight * self.gram_loss_fn(s_gram, t_gram)
            total = total + gram_term
            components["gram"] = gram_term.item()

        components["total"] = total.item()
        self._last_loss_components = components

        return total

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------

    if sys.platform == "win32":
        import colorama
        colorama.init()

    def train_epoch(self, epoch: int) -> float:
        self.student.train()

        # Gram weight ramp-up (avoids sudden shock when Gram is first enabled)
        if self.use_gram:
            ramp_epochs = self.gram_ramp_end_epoch - self.gram_ramp_start_epoch
            if epoch < self.gram_ramp_end_epoch:
                p = max(0.0, min(1.0, (epoch - self.gram_ramp_start_epoch) / max(1, ramp_epochs)))
                self.gram_loss_weight = self._base_gram_weight * (0.1 + 0.9 * p)
            else:
                self.gram_loss_weight = self._base_gram_weight

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.total_epochs}",
            ncols=80,
            ascii=" >=",
            mininterval=1.0,
            file=sys.stdout,
        )
        total_loss, n = 0.0, 0
        component_sums = {"dino": 0.0, "ibot": 0.0, "koleo": 0.0, "gram": 0.0}

        for crops_batch in pbar:
            # Weight-decay schedule: FIX -- official's real recipes hold weight decay
            # constant (0.04) within a phase; the 0.04->0.4 cosine ramp is kept as an
            # opt-in (`--wd_schedule cosine`) but is no longer the default.
            if self.wd_schedule == "constant":
                current_wd = self.weight_decay_start
            else:
                start_wd, end_wd = self.weight_decay_start, self.weight_decay_end
                total_iters = self.total_epochs * len(self.train_loader)
                wd_progress = self.current_iter / max(1, total_iters)
                current_wd = end_wd - (end_wd - start_wd) * (math.cos(math.pi * wd_progress) + 1.0) / 2.0

            for param_group in self.optimizer.param_groups:
                if "weight_decay" in param_group:
                    param_group["weight_decay"] = current_wd

            crops_batch = (
                [c.to(self.device) for c in crops_batch]
                if isinstance(crops_batch, (list, tuple))
                else crops_batch.to(self.device)
            )
            self.optimizer.zero_grad()

            if self.use_amp:
                dtype = (
                    torch.bfloat16
                    if next(self.student.parameters()).dtype == torch.bfloat16
                    else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = self._compute_loss(crops_batch)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
                    # Protects weights from NaN/Inf poisoning typical of sudden loss spikes
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        self.logger.warning(f"NaN/Inf gradient detected (norm={grad_norm:.2f}). Skipping step.")
                        self.optimizer.zero_grad(set_to_none=True)
            else:
                loss = self._compute_loss(crops_batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                else:
                    self.logger.warning(f"NaN/Inf gradient detected (norm={grad_norm:.2f}). Skipping step.")
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Step LR scheduler per iteration (not per epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            self._ema_update()
            self.current_iter += 1
            total_loss += loss.item()
            for key in component_sums:
                component_sums[key] += self._last_loss_components.get(key, 0.0)
            n += 1
            pbar.set_postfix({"loss": f"{total_loss / n:.4f}"})

            if (self.use_wandb or self.tb_writer is not None) and \
                    (self.current_iter % self.log_every_n_steps == 0):
                self._log_metrics({
                    "loss/total": loss.item(),
                    "loss/dino": self._last_loss_components.get("dino", 0.0),
                    "loss/ibot": self._last_loss_components.get("ibot", 0.0),
                    "loss/koleo": self._last_loss_components.get("koleo", 0.0),
                    "loss/gram": self._last_loss_components.get("gram", 0.0),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "teacher_temp": self._teacher_temp(),
                    "gram_weight": self.gram_loss_weight,
                    "ema_tau": self._get_ema_tau(),
                    "weight_decay": current_wd,
                }, step=self.current_iter)

        epoch_loss = total_loss / n
        component_avgs = {key: val / n for key, val in component_sums.items()}
        self.train_losses.append(epoch_loss)
        self.logger.info(
            f"Epoch {epoch + 1}/{self.total_epochs}  "
            f"loss={epoch_loss:.4f}  "
            f"dino={component_avgs['dino']:.4f}  "
            f"ibot={component_avgs['ibot']:.4f}  "
            f"koleo={component_avgs['koleo']:.4f}  "
            f"gram={component_avgs['gram']:.4f}  "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
            f"t_temp={self._teacher_temp():.4f}  "
            f"gram_w={self.gram_loss_weight:.4f}"
        )
        self._append_loss_history_row(epoch, epoch_loss, component_avgs)
        if self.use_wandb or self.tb_writer is not None:
            self._log_metrics({
                "epoch/total_loss": epoch_loss,
                "epoch/dino_loss": component_avgs["dino"],
                "epoch/ibot_loss": component_avgs["ibot"],
                "epoch/koleo_loss": component_avgs["koleo"],
                "epoch/gram_loss": component_avgs["gram"],
                "epoch/lr": self.optimizer.param_groups[0]["lr"],
                "epoch/teacher_temp": self._teacher_temp(),
                "epoch/gram_weight": self.gram_loss_weight,
                "epoch/epoch_num": epoch + 1,
            }, step=self.current_iter)
        return epoch_loss

    def _log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Send a dict of metrics to whichever trackers are enabled. Never lets a
        tracker error interrupt training -- logging failures are only warned about."""
        payload = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
        if self.use_wandb:
            try:
                wandb.log(payload, step=step)
            except Exception as e:
                self.logger.warning(f"wandb.log failed (continuing without it this step): {e}")
        if self.tb_writer is not None:
            try:
                for k, v in payload.items():
                    self.tb_writer.add_scalar(k, v, global_step=step)
            except Exception as e:
                self.logger.warning(f"TensorBoard logging failed (continuing without it this step): {e}")

    def _append_loss_history_row(self, epoch: int, epoch_loss: float, component_avgs: dict):
        """Append one row of per-epoch loss-curve data to output_dir/loss_history.csv,
        writing a header only on first creation. Safe across resumed runs: rows just
        keep appending, so the full curve (including pre-resume epochs) stays in one
        file rather than being reset."""
        file_exists = self.loss_history_csv.exists()
        with open(self.loss_history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "epoch", "total_loss", "dino_loss", "ibot_loss", "koleo_loss",
                    "gram_loss", "lr", "teacher_temp", "gram_weight",
                ])
            writer.writerow([
                epoch + 1,
                f"{epoch_loss:.6f}",
                f"{component_avgs['dino']:.6f}",
                f"{component_avgs['ibot']:.6f}",
                f"{component_avgs['koleo']:.6f}",
                f"{component_avgs['gram']:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.8e}",
                f"{self._teacher_temp():.6f}",
                f"{self.gram_loss_weight:.6f}",
            ])

    # ------------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, extra_data: dict = None):
        fname = f"benthic-ssl-epoch={epoch:02d}-ssl_loss={loss:.2f}.ckpt"
        fpath = self.output_dir / fname
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(), # ✅ FIX: Saved Teacher Weights
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "loss": loss,
            "current_iter": self.current_iter,
            "config": {
                "warmup_epochs": self.warmup_epochs,
                "total_epochs": self.total_epochs,
                "base_lr": self.base_lr,
            },
        }
        if self.use_gram_teacher and self.gram_teacher is not None:
            ckpt["gram_teacher_state_dict"] = self.gram_teacher.state_dict()
        
        # Merge any extra data (e.g., k-NN accuracy)
        if extra_data:
            ckpt.update(extra_data)
        
        torch.save(ckpt, fpath)
        self.logger.info(f"Saved: {fpath}")
        if is_best:
            best = self.output_dir / "benthic-ssl-best.ckpt"
            torch.save(ckpt, best)
            self.logger.info(f"Best:  {best}")


# ============================================================================
# k-NN VALIDATION PROBE (BATCHED & EXTENDED LOOKUP)
# ============================================================================

class KNNProbeDataset(Dataset):
    """Batched Dataset for k-NN probe supporting full image extension matching."""
    def __init__(self, df, image_root: Optional[str], transform):
        self.df = df
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        extensions = ("", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        
        for _, row in df.iterrows():
            image_id = str(row['image'])
            label_id = row['label_id']
            img_path = None
            
            # 1. Try relative to image_root
            if image_root:
                for ext in extensions:
                    candidate = os.path.join(image_root, f"{image_id}{ext}")
                    if os.path.isfile(candidate):
                        img_path = candidate
                        break
            # 2. Try relative to working directory or direct path
            if not img_path:
                for ext in extensions:
                    candidate = f"{image_id}{ext}"
                    if os.path.isfile(candidate):
                        img_path = candidate
                        break
            
            if img_path:
                self.samples.append((img_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return None, label


def _collate_knn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def knn_validation_probe(trainer, args, logger, device, k=20, batch_size=64):
    """
    Validate on a held-out test subset using k-NN classification on frozen teacher CLS features.
    
    Uses the test partition from the CSV (if --knn_csv_path provided) to evaluate the teacher
    backbone's learned representations. Returns top-1 k-NN accuracy on catami_substrate labels.
    Batched via DataLoader for GPU speedup.
    """
    if not hasattr(args, 'knn_csv_path') or not args.knn_csv_path or not os.path.exists(args.knn_csv_path):
        return None
    
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping k-NN probe. Install with 'pip install pandas'.")
        return None
    
    trainer.teacher.eval()
    
    logger.info(f"Loading k-NN validation set from {args.knn_csv_path} (test partition)...")
    df = pd.read_csv(args.knn_csv_path, low_memory=False)
    test_df = df[df['partition'] == 'test'].copy()
    test_df = test_df[test_df['catami_substrate'].notna()].copy()
    
    if len(test_df) == 0:
        logger.warning("No test samples with valid catami_substrate labels found.")
        return None
    
    substrate_labels = sorted(test_df['catami_substrate'].unique())
    label_map = {label: idx for idx, label in enumerate(substrate_labels)}
    test_df['label_id'] = test_df['catami_substrate'].map(label_map)
    
    logger.info(f"Found {len(test_df)} test samples, {len(substrate_labels)} substrate classes.")
    
    image_root = args.knn_image_root if hasattr(args, 'knn_image_root') and args.knn_image_root else None
    
    # In knn_validation_probe():
    _mean = BENTHIC_MEAN if getattr(args, 'benthic_norm', False) else IMAGENET_MEAN
    _std = BENTHIC_STD if getattr(args, 'benthic_norm', False) else IMAGENET_STD
    val_transform = T.Compose([
        T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=_mean, std=_std),
    ])
    
    probe_dataset = KNNProbeDataset(test_df, image_root, val_transform)
    if len(probe_dataset) == 0:
        logger.warning("No valid test images could be located on disk for k-NN probe.")
        return None

    failed_count = len(test_df) - len(probe_dataset)
    if failed_count > 0:
        logger.info(f"k-NN validation: {len(probe_dataset)} valid images queued, {failed_count} missing from disk.")

    eval_batch_size = getattr(args, 'batch_size', batch_size)
    num_workers = getattr(args, 'num_workers', 4)
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_knn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    feats_list, labels_list = [], []
    for imgs, lbls in tqdm(probe_loader, desc="k-NN validation", leave=False):
        if imgs is None:
            continue
        imgs = imgs.to(device)
        with torch.no_grad():
            out = trainer.teacher(pixel_values=imgs)
            cls_feat = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        feats_list.append(cls_feat.cpu())
        labels_list.append(lbls)
    
    if not feats_list:
        logger.warning("No test images loaded successfully during k-NN probe.")
        return None
    
    feats = torch.cat(feats_list, dim=0)  # [N, D]
    labels = torch.cat(labels_list, dim=0)  # [N]
    
    # Compute k-NN accuracy
    with torch.no_grad():
        sims = feats @ feats.t()  # [N, N]
        sims.fill_diagonal_(-1e9)  # exclude self
        topk_indices = sims.topk(k, dim=1).indices  # [N, k]
        topk_labels = labels[topk_indices]  # [N, k]
        
        try:
            preds = torch.mode(topk_labels, dim=1).values
        except Exception:
            preds = torch.mode(topk_labels, dim=1)[0]
        
        acc = (preds == labels).float().mean().item()
    
    logger.info(f"k-NN validation (k={k}): accuracy={acc:.4f} on {len(feats)} test samples")
    return acc


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.log_dir)
    logger.info("=" * 70)
    logger.info("DINOv3 SSL — Benthic Seafloor Imagery (v7 — benthic_norm/local_crop_size now real get_transform() params, "
                "fixing the unconditional TypeError crash on every run)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    logger.info(f"Mask sample probability: {args.mask_sample_probability:.2f} "
                f"(fraction of the batch that gets any masking at all -- official semantics)")
    logger.info(f"Pixel-space masking of masked positions: "
                f"{'DISABLED (ablation)' if args.no_pixel_masking else 'ENABLED (intentional deviation from official mask_token substitution)'}")
    
    if args.official_dinov3_aug:
        logger.info("Augmentation pipeline: OFFICIAL DINOv3 recipe (bicubic resize, grayscale + "
                     "solarize + probability-gated jitter/blur with kernel_size=9, no flip, no "
                     "rotation/affine) — --no_underwater_aug and --marine_aug are ignored")
        if args.image_size != 256:
            logger.warning(
                f"--official_dinov3_aug is set with --image_size {args.image_size}, but local crops "
                f"are fixed at 112 (official's real local_crops_size). Official only ever pairs "
                f"112 local crops with image_size=256 global crops (pretrain/gram_anchor) -- this "
                f"combination doesn't match any real recipe's crop-size pairing. Pass --image_size 256 "
                f"for a genuinely faithful reproduction, or treat this run as an ablation rather than "
                f"an official-parity baseline."
            )
            
    elif args.marine_aug:
        logger.info(f"Augmentation pipeline: MARINE-SPECIFIC recipe (marine water-column color jitter, "
                     f"turbidity blur, solarization, orientation augs={'ENABLED' if not args.no_underwater_aug else 'DISABLED (ablation)'})")
        if args.local_crop_size is None and args.image_size != 224:
            logger.warning(
                f"marine_aug's local crop is a fixed 96px, independent of --image_size "
                f"({args.image_size}). At other sizes local crops will be disproportionate "
                f"to global crops unless you pass --local_crop_size explicitly."
        )
    else:
        logger.info(f"Augmentation pipeline: BENTHIC DEFAULT (orientation augs="
                     f"{'ENABLED' if not args.no_underwater_aug else 'DISABLED (ablation)'})")

    transform = get_transform(
        image_size=args.image_size,
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        marine_aug=args.marine_aug,
        underwater_orientation_aug=not args.no_underwater_aug,
        official_dinov3_aug=args.official_dinov3_aug,
        benthic_norm=args.benthic_norm,
        local_crop_size=args.local_crop_size,   # was missing -- CLI flag reached nothing without this
    )

    logger.info(f"Normalization: {'BenthicNet-specific' if args.benthic_norm else 'ImageNet default'}")

    if args.use_webdataset:
        if not WEBDATASET_AVAILABLE:
            raise ImportError("--use_webdataset requires the 'webdataset' package: pip install webdataset")
        if not args.shard_dir:
            raise ValueError("--use_webdataset requires --shard_dir to be set")
        logger.info(f"Data loading: WebDataset (tar shards) from {args.shard_dir}")
        train_loader, steps_per_epoch = build_webdataset_loader(
            shard_dir=args.shard_dir,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            shard_pattern=args.shard_pattern,
            shuffle_buffer=args.webdataset_shuffle_buffer,
            num_samples=args.webdataset_num_samples,
            distributed=args.distributed,
            seed=args.seed,
        )
        logger.info(f"WebDataset loader ready: {steps_per_epoch} steps/epoch")
        sampler = None
    else:
        logger.info(f"Data loading: folder scan from {args.data_dir}")
        dataset = BenthicImageDataset(args.data_dir, transform=transform)

        sampler = None
        if args.distributed:
            sampler = DistributedSampler(dataset)

        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        if args.num_workers > 0:
            # persistent_workers avoids respawning worker processes every epoch
            # (expensive on Windows, which uses 'spawn' rather than 'fork').
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = args.prefetch_factor
        train_loader = DataLoader(
            dataset,
            shuffle=(sampler is None),
            sampler=sampler,
            **loader_kwargs,
        )

    # ✅ FIX: Handle bfloat16 properly at initialization time
    if args.use_bf16:
        torch_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision (GradScaler will be bypassed automatically)")
    elif args.use_fp16:
        torch_dtype = torch.float16
        logger.info("Using float16 precision")
    else:
        torch_dtype = torch.float32

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        args.model_id,
        attn_implementation="sdpa",
        dtype=torch_dtype,
    )

    hidden_dim = model.config.hidden_size
    model.dino_head = DINOHead(hidden_dim, args.proj_dim, hidden_dim=2048, bottleneck_dim=256,
                                nlayers=3, use_weight_norm=args.dino_head_weight_norm)
    if args.use_ibot:
        model.ibot_head = DINOHead(hidden_dim, args.proj_dim, hidden_dim=2048, bottleneck_dim=256,
                                    nlayers=3, use_weight_norm=args.dino_head_weight_norm)

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("pip install peft")
        save_mods = ["dino_head", "ibot_head"] if args.use_ibot else ["dino_head"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                modules_to_save=save_mods,
            ),
        )

    model = model.to(device)

    wandb_active = args.use_wandb
    if wandb_active and not WANDB_AVAILABLE:
        logger.warning("--use_wandb requested but wandb is not installed (pip install wandb); "
                        "continuing without it.")
        wandb_active = False
    if wandb_active:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or Path(args.output_dir).name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )
        logger.info(f"W&B logging active: project={args.wandb_project} "
                    f"run={args.wandb_run_name or Path(args.output_dir).name}")

    trainer = DINOTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        output_dir=args.output_dir,
        logger=logger,
        use_amp=args.use_amp,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.learning_rate,
        head_lr_mult=args.head_lr_mult,
        lr_decay_rate=args.lr_decay_rate,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
        use_ibot=args.use_ibot,
        proj_dim=args.proj_dim,
        teacher_ema_tau=args.teacher_ema_tau,
        teacher_ema_tau_final=args.teacher_ema_tau_final,
        momentum_schedule=args.momentum_schedule,
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        use_koleo=args.use_koleo,
        koleo_loss_weight=args.koleo_loss_weight,
        use_gram=args.use_gram,
        gram_loss_weight=args.gram_loss_weight,
        gram_img_level=args.gram_img_level,
        gram_remove_neg=args.gram_remove_neg,
        gram_remove_only_teacher_neg=args.gram_remove_only_teacher_neg,
        lr_schedule=args.lr_schedule,
        teacher_temp_base=args.teacher_temp_base,
        use_gram_teacher=args.use_gram_teacher,
        gram_teacher_ckpt=args.gram_teacher_ckpt,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        mask_sample_probability=args.mask_sample_probability,
        apply_pixel_masking=not args.no_pixel_masking,
        weight_decay_start=args.weight_decay_start,
        weight_decay_end=args.weight_decay_end,
        wd_schedule=args.wd_schedule,
        dino_head_weight_norm=args.dino_head_weight_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        use_wandb=wandb_active,
        use_tensorboard=args.use_tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        log_every_n_steps=args.log_every_n_steps,
    )

    if args.init_from_ckpt:
        logger.info(f"Initializing student/teacher from {args.init_from_ckpt}")
        ckpt = torch.load(args.init_from_ckpt, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
        # Load into both student and teacher (EMA teacher will adjust quickly)
        trainer.student.load_state_dict(sd, strict=False)
        trainer.teacher.load_state_dict(sd, strict=False)
        logger.info("Student and teacher initialized from checkpoint (new training run).")

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)

        # Load Student
        sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in ckpt["model_state_dict"].items()
        }
        trainer.student.load_state_dict(sd, strict=False)

        # ✅ FIX: Load Teacher safely
        if "teacher_state_dict" in ckpt:
            t_sd = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in ckpt["teacher_state_dict"].items()
            }
            trainer.teacher.load_state_dict(t_sd, strict=False)
        else:
            trainer.logger.warning("No teacher state in checkpoint; initializing teacher as exact copy of student to avoid random init.")
            trainer.teacher.load_state_dict(trainer.student.state_dict())

        try:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            logger.warning("Optimizer state not loaded — starting fresh.")
        if trainer.scheduler:
            try:
                trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                logger.warning("Scheduler state not loaded — restarting.")
        trainer.current_iter = ckpt.get("current_iter", 0)
        if args.use_gram_teacher and "gram_teacher_state_dict" in ckpt:
            trainer.gram_teacher.load_state_dict(ckpt["gram_teacher_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        
        # Initialize best metric tracking based on whether k-NN validation is enabled
        if args.use_knn_validation:
            trainer.best_knn_acc = ckpt.get("knn_acc", -1.0)
        else:
            trainer.best_loss = ckpt.get("loss", float("inf"))

        # Resume protections
        trainer.gram_ramp_start_epoch = start_epoch
        trainer.gram_ramp_end_epoch   = start_epoch + 9  # 9-epoch ramp
        logger.info(f"Gram loss ramping {args.gram_loss_weight:.2f} x [0.1 -> 1.0] over epochs {start_epoch}-{trainer.gram_ramp_end_epoch}")

        logger.info(f"Resumed from epoch {start_epoch - 1}, iter {trainer.current_iter}")
    else:
        # Initialize best metric tracking for new run
        if args.use_knn_validation:
            trainer.best_knn_acc = -1.0
        else:
            trainer.best_loss = float("inf")

    # ---- Train ----
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss = trainer.train_epoch(epoch)
        
        # Determine if this is the best checkpoint
        is_best = False
        knn_acc = None
        extra_ckpt_data = {}
        
        if args.use_knn_validation:
            # k-NN validation is the source of truth for best checkpoint
            if args.knn_freq_epochs > 0 and (epoch + 1) % args.knn_freq_epochs == 0:
                knn_acc = knn_validation_probe(trainer, args, logger, device, k=args.knn_k)
                if knn_acc is not None:
                    is_best = knn_acc > trainer.best_knn_acc
                    if is_best:
                        trainer.best_knn_acc = knn_acc
                        logger.info(f"New best k-NN accuracy: {knn_acc:.4f}")
                    extra_ckpt_data['knn_acc'] = knn_acc
            # else: not a probe epoch -- is_best stays False, no fallback to loss
        else:
            # Training loss is the source of truth for best checkpoint
            is_best = loss < trainer.best_loss
            if is_best:
                trainer.best_loss = loss
        
        # Save on interval, at end of training, or immediately when a new best is found
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1 or is_best:
            trainer.save_checkpoint(epoch, loss, is_best=is_best, extra_data=extra_ckpt_data)

    if trainer.tb_writer is not None:
        trainer.tb_writer.close()
    if wandb_active:
        wandb.finish()


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 SSL — Benthic Seafloor Imagery")
    p.add_argument("--data_dir",       type=str, default=None,
                    help="Folder of individual images to scan recursively. Required "
                         "unless --use_webdataset is set (validated at runtime).")
    p.add_argument("--image_size",     type=int, default=224)
    p.add_argument("--model_id",       type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--proj_dim",       type=int, default=16384)

    # LoRA
    p.add_argument("--use_lora",               action="store_true")
    p.add_argument("--lora_r",                 type=int,   default=16)
    p.add_argument("--lora_alpha",             type=int,   default=32)
    p.add_argument("--lora_dropout",           type=float, default=0.1)
    p.add_argument("--lora_target_modules",    nargs="+",  default=["q_proj", "v_proj"])

    # Training schedule
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--head_lr_mult",    type=float, default=5.0,
                    help="LR multiplier for DINO/iBOT heads vs backbone.")
    p.add_argument("--lr_decay_rate",   type=float, default=0.98,
                    help="Layer-wise LR decay rate (official default: 0.98 for the 7B-scale "
                         "main recipes; the base ssl_default_config.yaml itself defaults to 0.9). "
                         "1.0 = no decay (all layers same LR). "
                         "< 1.0 = deeper layers get progressively smaller LR.")
    p.add_argument("--patch_embed_lr_mult", type=float, default=0.2,
                    help="LR multiplier for patch-embedding params, applied on top of layer-wise "
                         "decay. FIX: the bare function default in official's param_groups.py is "
                         "1.0, but every shipped recipe that sets it explicitly (base config, "
                         "pretrain, gram_anchor, high_res_adapt, distilled) overrides it to 0.2 -- "
                         "that's the value actually used in practice, and now the default here too.")
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--lr_schedule",     type=str,   default="cosine_warmup")
    p.add_argument("--teacher_temp_base", type=float, default=0.07)

    # EMA momentum -- FIX: now actually exposed via CLI (previously hardcoded at 0.996
    # inside DINOTrainer with no way to override it), default changed to match official's
    # real constant-momentum recipes.
    p.add_argument("--teacher_ema_tau", type=float, default=0.994,
                   help="EMA momentum for the teacher. Official's real recipes use a "
                        "constant 0.994 during main pretraining and 0.999 during a "
                        "Gram-anchor/high-res-adapt-style stage.")
    p.add_argument("--teacher_ema_tau_final", type=float, default=1.0,
                   help="Only used when --momentum_schedule cosine.")
    p.add_argument("--momentum_schedule", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="'constant' (default) matches official's shipped recipes. "
                        "'cosine' is the older DINOv2-style ramp from --teacher_ema_tau "
                        "to --teacher_ema_tau_final over the whole run.")

    # Weight decay -- FIX: default changed to a constant, matching official's real
    # recipes (the 0.04->0.4 ramp is kept as an opt-in).
    p.add_argument("--weight_decay_start", type=float, default=0.04)
    p.add_argument("--weight_decay_end",   type=float, default=0.4,
                   help="Only used when --wd_schedule cosine.")
    p.add_argument("--wd_schedule", type=str, default="constant", choices=["constant", "cosine"],
                   help="'constant' (default) matches official's shipped recipes. 'cosine' "
                        "is the older 0.04->0.4 ramp.")

    # AdamW betas -- FIX: exposed via CLI instead of hardcoded. 0.99 is NOT a universal
    # "official DINOv3" value; it's specific to the 7B-scale pretrain/gram_anchor/
    # high_res_adapt recipes. The base config, distillation recipe, and linear-probe
    # recipe all use the standard AdamW default of 0.999.
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.99,
                   help="AdamW beta2. Official's 7B-scale main recipes (pretrain/gram_anchor/"
                        "high_res_adapt) use 0.99; the base ssl_default_config.yaml, the "
                        "distillation recipe, and the linear-probe recipe all use the standard "
                        "0.999. Neither is universally 'the' official value -- pick per your "
                        "closest analogous recipe, or sweep it.")

    # AMP / precision
    p.add_argument("--use_amp",  action="store_true")
    p.add_argument("--use_bf16", action="store_true", help="✅ Use bfloat16 precision instead of float32/float16")
    p.add_argument("--use_fp16", action="store_true", help="Use float16 precision (requires GradScaler)")

    # Losses
    p.add_argument("--use_ibot",            action="store_true")
    p.add_argument("--use_koleo",           action="store_true")
    p.add_argument("--koleo_loss_weight",   type=float, default=0.1,
                   help="FIX: default changed 1e-4 -> 0.1 to match official.")
    p.add_argument("--use_gram",            action="store_true")
    p.add_argument("--gram_loss_weight",    type=float, default=1.0)
    p.add_argument("--gram_img_level",      action="store_true", default=True)
    p.add_argument("--gram_remove_neg", action="store_true", default=False,
                   help="FIX: default False now, matching every shipped Gram recipe.")
    p.add_argument("--gram_remove_only_teacher_neg", action="store_true", default=False,
                   help="FIX: default False now, matching every shipped Gram recipe.")
    p.add_argument("--use_gram_teacher",    action="store_true")
    p.add_argument("--gram_teacher_ckpt",   type=str, default=None,
                   help="Checkpoint from Stage-1 (no Gram) to use as the static Gram teacher.")

    # DINOHead
    p.add_argument("--dino_head_weight_norm", action="store_true", default=False,
                   help="FIX: default False now -- weight_norm does not appear anywhere "
                        "in the current official dinov3 repo (it was a DINOv2-era detail). "
                        "Opt in if you deliberately want the older DINOv2-style head.")

    # Masking
    p.add_argument("--mask_ratio_min", type=float, default=0.1)
    p.add_argument("--mask_ratio_max", type=float, default=0.5)
    p.add_argument("--mask_sample_probability", type=float, default=0.5,
                    help="FIX (was --pixel_mask_prob, wrongly used as a Gaussian-ratio "
                         "generator with a per-crop pixel-zeroing coin-flip). Now matches "
                         "official's mask_sample_probability exactly: this fraction of the "
                         "batch gets a non-empty (block-wise, linearly-spaced-ratio) mask; "
                         "the rest get no masking at all.")
    p.add_argument("--no_pixel_masking", action="store_true", default=False,
                    help="Ablation: never zero out pixels in pixel space, while still "
                         "computing the iBOT token loss on the same mask positions "
                         "(student simply saw the full image for those crops). Independent "
                         "of --mask_sample_probability, which controls which positions are "
                         "selected as 'masked' in the first place.")

    # Crops
    p.add_argument("--num_global_crops", type=int, default=2)
    p.add_argument("--num_local_crops",  type=int, default=8)
    p.add_argument("--local_crop_size", type=int, default=None,
                help="Local crop size (px) for marine_aug/default paths -- ignored when "
                     "--official_dinov3_aug is set. Defaults to 96px if omitted, independent "
                     "of --image_size. Override for non-default --image_size runs (e.g. "
                     "Stage 3's 384) to preserve the intended local:global ratio. Should stay "
                     "divisible by the model's patch size (16).")

    # Augmentation ablations
    p.add_argument("--no_underwater_aug", action="store_true", default=False,
                    help="Ablation: disable underwater-specific orientation augmentation "
                         "(RandomVerticalFlip p=0.3 and full 180-degree RandomRotation), "
                         "leaving only standard, orientation-preserving DINO-style augmentation.")
    p.add_argument("--official_dinov3_aug", action="store_true", default=False,
                    help="Ablation: replace the marine-tuned augmentation pipeline with the "
                         "exact official DINO/DINOv2/DINOv3 multi-crop recipe (bicubic resize, "
                         "kernel_size=9 blur, local crop size 112 -- matches official's real "
                         "local_crops_size across pretrain/gram_anchor/high_res_adapt, not "
                         "DINOv2's 96). Ignores --marine_aug and --no_underwater_aug when set.")
    p.add_argument("--benthic_norm", action="store_true", default=False,
                help="Use BenthicNet-specific normalization (mean=[0.359,0.413,0.386], "
                     "std=[0.219,0.215,0.209], from DalhousieAI/ssl-bentho) instead of "
                     "ImageNet defaults. Applies to training aug AND the k-NN eval transform "
                     "consistently. Recommended: recompute on your own 189,101-image pool "
                     "before trusting this third-party number for a real run.")

    # WebDataset (tar-shard) loading
    p.add_argument("--use_webdataset", action="store_true", default=False,
                    help="Read training data from tar shards via webdataset instead of "
                         "scanning --data_dir for individual image files. Requires --shard_dir "
                         "and 'pip install webdataset'.")
    p.add_argument("--shard_dir", type=str, default=None,
                    help="Directory containing tar shards. Required when --use_webdataset is set.")
    p.add_argument("--shard_pattern", type=str, default="shard-*.tar",
                    help="Glob pattern (relative to --shard_dir) matching shard tar files.")
    p.add_argument("--webdataset_shuffle_buffer", type=int, default=2000,
                    help="Sample-level shuffle buffer size for the WebDataset pipeline.")
    p.add_argument("--webdataset_num_samples", type=int, default=None,
                    help="Total sample count across all shards, used to fix the epoch length. "
                         "If omitted, counted automatically on startup (one-time sequential pass).")
    p.add_argument("--prefetch_factor", type=int, default=2,
                    help="Batches each DataLoader worker prefetches ahead of time. Only used "
                         "when --num_workers > 0. Only raise this if you have RAM headroom to "
                         "spare -- it does not help if the bottleneck is disk I/O or GPU compute.")

    # Misc
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--distributed",   action="store_true")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--output_dir",    type=str, default="./ssl_checkpoints")
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--log_dir",       type=str, default="./logs")
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--marine_aug",    action="store_true")

    # k-NN Validation for best.ckpt selection (alternative to training loss)
    p.add_argument("--use_knn_validation", action="store_true", default=False,
                    help="Select best.ckpt based on held-out k-NN accuracy instead of training loss. "
                         "Requires --knn_csv_path with a 'partition' column (train/test) and "
                         "'catami_substrate' labels. Run validation every --knn_freq_epochs epochs.")
    p.add_argument("--knn_csv_path", type=str, default=None,
                    help="Path to BenthicNet CSV with 'partition' and 'catami_substrate' columns. "
                         "Used for held-out k-NN validation when --use_knn_validation is set.")
    p.add_argument("--knn_image_root", type=str, default=None,
                    help="Root directory where test images are located (by image ID). "
                         "If omitted, will attempt to locate images relative to current directory.")
    p.add_argument("--knn_k", type=int, default=20,
                    help="Number of neighbors for k-NN validation accuracy (default 20).")
    p.add_argument("--knn_freq_epochs", type=int, default=5,
                    help="Run k-NN validation every N epochs (default 5). Set to 0 to disable periodic validation.")

    p.add_argument('--init_from_ckpt', type=str, default=None,
                    help='Initialize student and teacher from a checkpoint (without optimizer/scheduler)')

    # Experiment tracking
    p.add_argument("--use_wandb", action="store_true", default=False,
                    help="Log metrics to Weights & Biases. Requires 'pip install wandb' and "
                         "either 'wandb login' once beforehand or a WANDB_API_KEY env var. "
                         "For runs without network access to wandb.ai, set env var "
                         "WANDB_MODE=offline and sync later with 'wandb sync'.")
    p.add_argument("--wandb_project", type=str, default="benthic-dinov3",
                    help="W&B project name.")
    p.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity (team/username). Uses your account default if not set.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name. Defaults to the --output_dir folder name.")
    p.add_argument("--wandb_run_id", type=str, default=None,
                    help="W&B run ID to resume into, keeping metric history continuous across "
                         "--resume instead of starting a disconnected new run. Omit for a fresh run.")
    p.add_argument("--use_tensorboard", action="store_true", default=False,
                    help="Log metrics to TensorBoard. Requires 'pip install tensorboard'.")
    p.add_argument("--tensorboard_dir", type=str, default=None,
                    help="TensorBoard log directory. Defaults to <output_dir>/tensorboard.")
    p.add_argument("--log_every_n_steps", type=int, default=1,
                    help="Log iteration-level metrics to W&B/TensorBoard every N optimizer "
                         "steps (epoch-level summaries always log regardless). Raise this if "
                         "logging overhead becomes noticeable on very long runs.")

    args = p.parse_args()

    if not args.use_webdataset and not args.data_dir:
        p.error("--data_dir is required unless --use_webdataset is set")
    if args.use_webdataset and not args.shard_dir:
        p.error("--use_webdataset requires --shard_dir")

    return args


if __name__ == "__main__":
    main(parse_args())