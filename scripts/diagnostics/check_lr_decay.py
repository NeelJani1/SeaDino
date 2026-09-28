#!/usr/bin/env python3
"""
Layer-Wise Learning Rate Decay Diagnostic
=========================================
Tests and verifies layer-wise learning rate decay (LLRD) multipliers for the
DINOv3 ViT backbone across both:
  1. Hugging Face ViT parameter structure (model.encoder.layer.N)
  2. Official Meta DINOv3 parameter structure (backbone.blocks.N)

Verifies that:
  - Patch embeddings and tokens receive layer_id=0 (decay^13)
  - Transformer blocks 0..11 receive progressive multipliers (decay^(12 - layer))
  - Output heads receive layer_id=13 (multiplier=1.0)
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssl_train import get_vit_lr_decay_rate


def test_lr_decay(decay_rate: float = 0.98, num_layers: int = 12):
    print("=" * 80)
    print(f"DINOv3 Layer-Wise Learning Rate Decay Verification")
    print(f"Base Decay Rate: {decay_rate} | Total Backbone Layers: {num_layers}")
    print("=" * 80)

    # Test parameter names across architectures
    test_cases = [
        # Embeddings & Special Tokens
        ("model.embeddings.patch_embeddings.projection.weight", 0),
        ("model.embeddings.cls_token", 0),
        ("model.embeddings.position_embeddings", 0),
        ("model.embeddings.mask_token", 0),
        
        # Hugging Face Transformer Blocks (0 to 11)
        ("model.encoder.layer.0.attention.attention.query.weight", 1),
        ("model.encoder.layer.3.intermediate.dense.weight", 4),
        ("model.encoder.layer.7.output.dense.weight", 8),
        ("model.encoder.layer.11.attention.output.dense.weight", 12),
        
        # Meta Official DINOv3 Blocks (0 to 11)
        ("backbone.blocks.0.attn.qkv.weight", 1),
        ("backbone.blocks.3.mlp.fc1.weight", 4),
        ("backbone.blocks.7.mlp.fc2.weight", 8),
        ("backbone.blocks.11.norm2.weight", 12),
        
        # DINO Heads (No decay / top-level)
        ("dino_head.last_layer.weight", 13),
        ("ibot_head.last_layer.weight", 13),
    ]

    print(f"\n{'Parameter Name':<60} {'Layer ID':<10} {'Multiplier':<12} {'Status'}")
    print("-" * 90)

    all_passed = True
    for name, expected_id in test_cases:
        actual_mult = get_vit_lr_decay_rate(name, decay_rate=decay_rate, num_layers=num_layers)
        expected_mult = decay_rate ** (num_layers + 1 - expected_id)
        
        passed = abs(actual_mult - expected_mult) < 1e-7
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        print(f"{name:<60} {expected_id:<10} {actual_mult:<12.5f} {status}")

    print("-" * 90)
    if all_passed:
        print("🎉 ALL TEST CASES PASSED: Layer-wise LR decay is correctly mapped for all ViT blocks.")
    else:
        print("⚠️ SOME TEST CASES FAILED: Check layer name parsing regex in ssl_train.py.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify ViT layer-wise learning rate decay mappings")
    p.add_argument("--decay_rate", type=float, default=0.98, help="Base layer decay rate (default: 0.98)")
    p.add_argument("--num_layers", type=int, default=12, help="Number of ViT backbone layers (default: 12)")
    args = p.parse_args()

    test_lr_decay(decay_rate=args.decay_rate, num_layers=args.num_layers)
