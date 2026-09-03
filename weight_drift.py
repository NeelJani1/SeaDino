#!/usr/bin/env python3
"""
Measures how far each saved checkpoint's backbone has drifted from the
off-the-shelf pretrained initialization, epoch by epoch -- a direct,
quantitative test of the "continued pretraining is forgetting/narrowing past
epoch 3" hypothesis. If drift keeps growing through later epochs at roughly
the same rate even as k-NN accuracy is already falling, that's hard evidence
for the forgetting story, not just a plausible-sounding narrative -- and if
drift is still accelerating rather than plateauing, that also tells you the
LR schedule hasn't naturally "settled" by the time downstream accuracy peaks.

Usage:
    python weight_drift.py --checkpoint_dir stage1_marine_with_physics
    python weight_drift.py --checkpoint_dir stage1_marine_with_physics \\
        --model_id facebook/dinov3-vits16-pretrain-lvd1689m
"""
import argparse
import glob
from pathlib import Path

import torch


def load_state_dict(checkpoint_path, model_id, device):
    """checkpoint_path=None returns the off-the-shelf reference weights."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", dtype=torch.float32)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in ckpt["teacher_state_dict"].items()
        }
        # strict=False: dino_head/ibot_head keys are expected to be ignored here --
        # this script only cares about the backbone, same as the k-NN eval does.
        model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return {k: v.float() for k, v in model.state_dict().items() if v.is_floating_point()}


def relative_drift(sd_a, sd_b):
    """Relative L2 drift: ||a - b|| / ||a|| aggregated over every shared
    floating-point backbone parameter."""
    total_sq_diff, total_sq_norm, n = 0.0, 0.0, 0
    for k in set(sd_a) & set(sd_b):
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape:
            continue
        total_sq_diff += (a - b).pow(2).sum().item()
        total_sq_norm += a.pow(2).sum().item()
        n += 1
    return (total_sq_diff ** 0.5) / (total_sq_norm ** 0.5 + 1e-12), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True,
                    help="Directory containing benthic-ssl-epoch=NN-*.ckpt files.")
    p.add_argument("--model_id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading off-the-shelf reference weights...")
    ref_sd = load_state_dict(None, args.model_id, device)

    ckpts = sorted(glob.glob(str(Path(args.checkpoint_dir) / "benthic-ssl-epoch=*.ckpt")))
    if not ckpts:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"\n{'checkpoint':45s} {'drift from off-the-shelf':>26s} {'shared params':>14s}   incremental drift since previous")
    prev_sd, prev_label = None, None
    for c in ckpts:
        label = Path(c).stem
        sd = load_state_dict(c, args.model_id, device)
        d, n = relative_drift(ref_sd, sd)
        step_str = ""
        if prev_sd is not None:
            d_step, _ = relative_drift(prev_sd, sd)
            step_str = f"   Δ since {prev_label}: {d_step:.4%}"
        print(f"{label:45s} {d:>25.4%} {n:>14d}   {step_str}")
        prev_sd, prev_label = sd, label

    print(
        "\nRead: if 'drift from off-the-shelf' keeps climbing at a similar (not "
        "shrinking) rate past the epoch where k-NN accuracy already peaked, the "
        "model is continuing to move away from the good initialization without "
        "anything correcting for it -- supports lowering the LR / raising EMA "
        "momentum / trying LoRA rather than just training fewer epochs."
    )


if __name__ == "__main__":
    main()