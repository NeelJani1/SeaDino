"""Inspect pre-output-LayerNorm tokens on the exact cached attention input."""
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
from eval_suite.config import REPO_ROOT
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch

from feature_visualization.backbones import load_backbone, patch_input, forward_tokens
from feature_visualization.common import write_json


def main():
    root = Path("/mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019")
    gate = root.parent/"coralscapes_all9_20261019/run.json"
    for _ in range(240):
        status = json.loads(gate.read_text()).get("corrections",[])
        if status and status[-1]["state"] == "complete":
            break
        time.sleep(5)
    else:
        raise RuntimeError("SAM refresh did not complete within twenty minutes")
    run = json.loads((root/"run.json").read_text())
    attention = json.loads((root/"attention_run.json").read_text())
    report = {}
    for key in ("dinov2_s","dinov2_b"):
        model, metadata = load_backbone(run["models"][key],torch.device("cuda"))
        native = getattr(model,"raw_model",model)
        first = attention["results"][key]["items"][0]
        details = json.loads((root/key/"attention"/first["metadata"]).read_text())
        count = details["geometry"]["grid_h"]*details["geometry"]["grid_w"]
        captured = {}
        def hook(module,arguments):
            captured["norms"] = arguments[0][0,-count:].float().norm(dim=-1).detach().cpu().numpy()
        handle = native.layernorm.register_forward_pre_hook(hook)
        try:
            record = run["records"][attention["sample_indices"][0]]
            norm = metadata["normalization"]
            with Image.open(record["staged_image"]) as image:
                x,_ = patch_input(image.convert("RGB"),attention["longest_side"],metadata["patch_size"],norm["mean"],norm["std"])
            with torch.inference_mode():
                output = forward_tokens(model,x.unsqueeze(0).cuda())[0,-count:].cpu().numpy()
        finally:
            handle.remove()
        with np.load(run["caches"][key]["arrays"]) as cache:
            reference = cache["patch_0"]
            np.testing.assert_allclose(output,reference,atol=1e-4,rtol=1e-5)
        with np.load(root/key/"attention"/first["arrays"]) as arrays:
            weights = arrays["last"].copy()
        norms = captured["norms"]
        maximum = int(np.argmax(norms)); peak = int(np.argmax(weights))
        report[key] = dict(sample_id=record["sample_id"], layer="encoder output before final LayerNorm",
            maximum_norm_over_median=float(norms.max()/np.median(norms)),
            maximum_norm_patch_index=maximum,maximum_attention_patch_index=peak,
            maximum_attention_patch_norm_over_median=float(norms[peak]/np.median(norms)),
            norm_attention_spearman=float(spearmanr(norms,weights).statistic),
            cached_patch_feature_parity_passed=True, maximum_cached_feature_error=float(np.max(np.abs(output-reference))))
        print(key,report[key],flush=True)
        del model, native
        gc.collect();torch.cuda.empty_cache()
    write_json(root/"dinov2_pre_norm_diagnostics.json",report)


if __name__ == "__main__":
    main()
