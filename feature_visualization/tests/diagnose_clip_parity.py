"""Inspect FP32 eager/SDPA agreement on the saved shared images; no rendering."""
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from eval_suite.config import REPO_ROOT
import numpy as np
from PIL import Image
import torch
from feature_visualization.backbones import load_backbone, patch_input, forward_tokens
from feature_visualization.common import write_json


@torch.inference_mode()
def main():
    root = Path("/mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019")
    run = json.loads((root / "run.json").read_text())
    model, meta = load_backbone(run["models"]["clip"], torch.device("cuda"), "eager")
    encoder = getattr(model, "raw_model", model)
    summary = []
    with np.load(run["caches"]["clip"]["arrays"], allow_pickle=False) as cache:
        for n, index in enumerate(run["patch_indices"]):
            record = run["records"][index]
            with Image.open(record["staged_image"]) as im:
                x, _ = patch_input(im.convert("RGB"), 512, meta["patch_size"], **meta["normalization"])
            encoder.set_attn_implementation("eager")
            tokens = forward_tokens(model, x.unsqueeze(0).cuda())
            count = x.shape[1] // meta["patch_size"] * (x.shape[2] // meta["patch_size"])
            eager = tokens[0, -count:].cpu().numpy()
            ref = cache[f"patch_{n}"]
            delta = eager - ref
            rel = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
            encoder.set_attn_implementation("sdpa")
            fresh = forward_tokens(model, x.unsqueeze(0).cuda())[0, -count:].cpu().numpy()
            summary.append(dict(sample_id=record["sample_id"], maximum_absolute_error=float(np.abs(delta).max()),
                                maximum_patch_relative_l2=float(rel.max()),
                                fresh_sdpa_to_cache_max_abs=float(np.max(np.abs(fresh-ref))),
                                eager_to_fresh_sdpa_max_abs=float(np.max(np.abs(eager-fresh)))))
            print(n, summary[-1], flush=True)
    write_json(root / "clip_attention_numerical_diagnostics.json", summary)


if __name__ == "__main__":
    main()
