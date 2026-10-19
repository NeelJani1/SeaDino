"""Compare cached global features to the benchmark extractor, one real model at a time.

python -m feature_visualization.tests.validate_real --run_dir /path/to/completed/run
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from feature_visualization.backbones import load_backbone, transform
from feature_visualization.common import write_json
from feature_visualization.runner import load_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = Path(args.run_dir)
    run = json.loads((root / "run.json").read_text())
    from eval_suite.benchmarks.explicit_classification import StrictImages, extract_features
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, spec in run["models"].items():
        print(f"Reference check: {key}", flush=True)
        saved = run["caches"][key]
        raw, normalized, _, _ = load_cache(saved)
        model, metadata = load_backbone(spec, device)
        rows = []
        for row in run["records"][:4]:
            # Same decoded source pixels, staged locally to avoid network variability.
            rows.append(dict(row, image=row["staged_image"], label=0))
        norm = metadata["normalization"]
        resolution = run.get("arguments", {}).get("image_size", 224)
        dataset = StrictImages(rows, transform(norm["mean"], norm["std"], resolution, metadata["patch_size"]), "reference")
        reference, _ = extract_features(model, dataset, device, torch.float32, 4, 0, run["seed"], key)
        error = float(np.max(np.abs(normalized[:len(rows)] - reference.numpy())))
        np.testing.assert_allclose(normalized[:len(rows)], reference.numpy(), atol=2e-5, rtol=2e-4)
        if metadata["frozen_parameters"] != metadata["parameters"]:
            raise AssertionError("A backbone contains trainable parameters")
        results[key] = dict(passed=True, maximum_absolute_error=error, shape=list(normalized.shape),
                            parameters=metadata["parameters"], pooling=metadata["pooling"], checkpoint_audit=metadata["checkpoint_audit"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_json(root / "benchmark_feature_validation.json", results)
    print(json.dumps({"validated_models": len(results), "all_passed": all(r["passed"] for r in results.values())}), flush=True)


if __name__ == "__main__":
    main()
