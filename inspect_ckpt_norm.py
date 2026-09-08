#!/usr/bin/env python3
"""Quick sanity inspector to check benthic_norm and loss across all 10 checkpoint arms."""

import glob
from pathlib import Path
import torch

checkpoint_dirs = [
    "/home/njan320/SeaDino/stage1_marine_with_physics_lr_1e-4_no_pixel",
    "/home/njan320/SeaDino/stage1_marine_with_physics_lr_1e-5_with_pixel",
    "/home/njan320/SeaDino/stage1_marine_with_physics_lr_2e-4_with_pixel",
    "/home/njan320/SeaDino/stage1_marine_with_physics_lr_5e-4_with_pixel",
    "/home/njan320/SeaDino/stage1_marine_with_physics_lr_1e-4_no_pixel_tau_99",
    "/home/njan320/stage1_marine_with_physics",
    "/home/njan320/Neel/stage1_marine_with_physics_lr_3e-4_pixel",
    "/home/njan320/Neel/Checkpoints/stage1_marine_no_physics",
    "/home/njan320/Neel/Checkpoints/ssl_off_pix",
    "/home/njan320/Neel/Checkpoints/ssl_off_no_pix",
]

print(f"{'Directory / Arm':<60} {'Epoch':<7} {'Loss':<8} {'benthic_norm':<12} {'Ckpt File'}")
print("=" * 115)

for d in checkpoint_dirs:
    p_dir = Path(d)
    ckpts = sorted(p_dir.glob("*.ckpt"))
    if not ckpts:
        print(f"{p_dir.name:<60} [NO CHECKPOINTS FOUND]")
        continue

    # Prioritize best checkpoint or last epoch
    best_ckpts = [c for c in ckpts if "best" in c.name]
    target_ckpt = best_ckpts[0] if best_ckpts else ckpts[-1]

    try:
        data = torch.load(target_ckpt, map_location="cpu", weights_only=False)
        epoch = str(data.get("epoch", "-"))
        loss_val = data.get("loss", data.get("ssl_loss", "-"))
        loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
        cfg = data.get("config", {})
        b_norm = cfg.get("benthic_norm", False) if isinstance(cfg, dict) else "N/A"
        print(f"{p_dir.name:<60} {epoch:<7} {loss_str:<8} {str(b_norm):<12} {target_ckpt.name}")
    except Exception as e:
        print(f"{p_dir.name:<60} ERROR: {e}")

