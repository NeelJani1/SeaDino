"""Read-only audit of base/adapted weights, saved attention and overlay geometry."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from scipy.stats import spearmanr

from feature_visualization.backbones import checkpoint_state
from feature_visualization.common import write_json


def pair_weights(run, base, adapted):
    path = run["models"][adapted]["checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    selected, state = checkpoint_state(checkpoint)
    cleaned = {k.replace("_orig_mod.", "").replace("backbone.", "").replace("teacher.", ""): v for k, v in state.items() if torch.is_tensor(v)}
    foundation = next(item["path"] for item in run["caches"][base]["source"]["files"] if item["path"].endswith(".safetensors"))
    changed, total, changed_values, elements = [], 0, 0, 0
    delta_sq, base_sq = 0., 0.
    with safe_open(foundation, framework="pt", device="cpu") as original:
        for key in original.keys():
            if key not in cleaned:
                raise ValueError(f"Missing teacher parameter {key}")
            ref, trained = original.get_tensor(key).float(), cleaned[key].float()
            assert ref.shape == trained.shape
            changed_count = int(torch.count_nonzero(ref != trained))
            if changed_count:
                changed.append(key)
            total += 1; elements += ref.numel(); changed_values += changed_count
            delta_sq += float(torch.sum((trained.double() - ref.double()).square()))
            base_sq += float(torch.sum(ref.double().square()))
    return dict(checkpoint=path, selected_state=selected, compared_tensors=total, tensors_different_from_foundation=len(changed),
                different_values=changed_values, total_values=elements, relative_weight_l2=float(np.sqrt(delta_sq/base_sq)),
                existing_loaded_tensor_audit=run["caches"][adapted]["model_metadata"]["checkpoint_audit"])


def main():
    torch.set_num_threads(2)
    root = Path("/mnt/d/Neel/FeatureVisualizations/fathomnet_all9_20261019")
    run = json.loads((root/"run.json").read_text())
    attention = json.loads((root/"attention_run.json").read_text())
    report = {"weights": {}, "attention_pairs": {}, "rendering": {}}
    for base, adapted in (("dinov3_s", "seadino_s"), ("dinov3_b", "seadino_b")):
        report["weights"][adapted] = pair_weights(run, base, adapted)
        correlations, differences = [], []
        for a, b in zip(attention["results"][base]["items"], attention["results"][adapted]["items"]):
            assert a["sample_id"] == b["sample_id"]
            with np.load(root/base/"attention"/a["arrays"]) as x, np.load(root/adapted/"attention"/b["arrays"]) as y:
                correlations.append(float(np.corrcoef(x["last"], y["last"])[0,1]))
                differences.append(float(np.max(np.abs(x["last"]-y["last"]))))
        report["attention_pairs"][adapted] = dict(images=len(correlations), identical_maps=int(sum(x == 0 for x in differences)),
            pearson_mean=float(np.mean(correlations)), pearson_range=[min(correlations),max(correlations)])
    for key in run["models"]:
        rows = []
        with np.load(run["caches"][key]["arrays"]) as cache:
            for n, item in enumerate(attention["results"][key]["items"]):
                directory = root/key/"attention"
                metadata = json.loads((directory/item["metadata"]).read_text())
                stem = item["sample_id"][:20]
                original = np.array(Image.open(item["original"]).convert("RGB"))
                heat = np.array(Image.open(directory/f"{stem}_last_heatmap.png").convert("RGB"))
                overlay = np.array(Image.open(item["last_overlay"]).convert("RGB"))
                expected = (.55*original + .45*heat).round().astype(np.uint8)
                error = int(np.max(np.abs(overlay.astype(int)-expected.astype(int))))
                with np.load(directory/item["arrays"]) as arrays:
                    values = arrays["last"].astype(float)
                    probability = values/values.sum()
                    entropy = float(-np.sum(probability*np.log(np.maximum(probability,1e-30)))/np.log(len(probability)))
                    norm = np.linalg.norm(cache[f"patch_{n}"],axis=1)
                    rows.append(dict(sample_id=item["sample_id"], geometry=metadata["geometry"], overlay_reconstruction_error_8bit=error,
                        normalized_spatial_entropy=entropy, maximum_patch_norm_over_median=float(norm.max()/np.median(norm)),
                        norm_attention_spearman=float(spearmanr(norm,values).statistic)))
        report["rendering"][key] = dict(images=len(rows), maximum_overlay_rounding_error=max(r["overlay_reconstruction_error_8bit"] for r in rows),
            mean_entropy=float(np.mean([r["normalized_spatial_entropy"] for r in rows])),
            mean_maximum_patch_norm_over_median=float(np.mean([r["maximum_patch_norm_over_median"] for r in rows])),
            mean_norm_attention_spearman=float(np.mean([r["norm_attention_spearman"] for r in rows])), first_image=rows[0])
    write_json(root/"attention_loading_diagnostics.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
