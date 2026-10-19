"""Refresh only SAM spatial artifacts after correcting rectangular RoPE geometry.

Square global features are preserved and checked against a fresh forward pass.
Original caches and spatial artifacts remain in an explicit correction archive.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from eval_suite.config import REPO_ROOT
from argparse import Namespace
import gc
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch

from feature_visualization.backbones import load_backbone, patch_input, forward_tokens, spatial_tokens, transform
from feature_visualization.common import code_identity, file_sha, write_json
from feature_visualization.runner import feature_key, load_cache, save_cache
from feature_visualization.render import render_comparisons, render_index, render_patches
from feature_visualization.attention import run_attention


def main():
    base = Path("/mnt/d/Neel/FeatureVisualizations")
    roots = [base/name for name in ("fathomnet_all9_20261019", "coralscapes_all9_20261019")]
    key = "sam31"
    # Mark both reports before spending time on the first one.
    for root in roots:
        run = json.loads((root/"run.json").read_text())
        archive = root/"corrections/sam_rectangular_rope_before"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("run.json", "attention_run.json", "spatial_attention_validation.json"):
            if (root/name).exists() and not (archive/name).exists():
                shutil.copy2(root/name,archive/name)
        for name in ("attention", "patches"):
            if not (archive/name).exists():
                shutil.copytree(root/key/name,archive/name)
        correction = dict(id="sam-rectangular-rope-v2", state="refreshing",
            description="SAM rectangular RoPE now follows image rows: x=index%grid_width, y=index//grid_width. Square classification preprocessing and features are unchanged.",
            benchmark_note="Previous SAM segmentation scores from rectangular inputs require a fresh probe evaluation; this refresh corrects spatial visualizations only.",
            archive=str(archive), started_utc=datetime.now(timezone.utc).isoformat())
        run["corrections"] = [x for x in run.get("corrections",[]) if x.get("id") != correction["id"]]+[correction]
        attention = json.loads((root/"attention_run.json").read_text())
        attention["results"][key]["state"] = "refreshing"
        attention["state"] = "running"
        write_json(root/"run.json",run)
        write_json(root/"attention_run.json",attention)
        render_index(root,run,list(run["results"].values()),[f"patch_comparisons/{run['records'][i]['sample_id'][:20]}.png" for i in run["patch_indices"]])
    for root in roots:
        print(f"\nSAM RECTANGULAR REFRESH: {root.name}",flush=True)
        run = json.loads((root/"run.json").read_text())
        saved_before = run["caches"][key]
        raw, normalized, previous, _ = load_cache(saved_before)
        device = torch.device("cuda")
        model, metadata = load_backbone(run["models"][key],device)
        norm, patch = metadata["normalization"],metadata["patch_size"]
        convert = transform(norm["mean"],norm["std"],run["arguments"].get("image_size",224),patch)
        images = []
        for record in run["records"][:4]:
            with Image.open(record["staged_image"]) as image:
                images.append(convert(image.convert("RGB")))
        with torch.inference_mode():
            fresh = forward_tokens(model,torch.stack(images).to(device))[:,0].cpu().numpy()
        np.testing.assert_allclose(fresh,raw[:4],atol=1e-4,rtol=1e-5,err_msg="Square global features changed")
        patches, geometries, drift = [],[],[]
        longest = run["arguments"]["patch_longest"]
        for n,index in enumerate(run["patch_indices"]):
            record = run["records"][index]
            assert file_sha(record["staged_image"]) == record["image_sha256"]
            with Image.open(record["staged_image"]) as image:
                x,geometry = patch_input(image.convert("RGB"),longest,patch,norm["mean"],norm["std"])
            with torch.inference_mode():
                tokens = forward_tokens(model,x.unsqueeze(0).to(device))
            values,gh,gw = spatial_tokens(tokens,*x.shape[-2:],patch)
            array = values[0].cpu().numpy()
            patches.append(array)
            geometry.update(grid_h=gh,grid_w=gw,index=index)
            geometries.append(geometry)
            drift.append(float(np.linalg.norm(array-previous[n])/max(np.linalg.norm(previous[n]),1e-12)))
            print(f"  Corrected SAM patch image {n+1}/{len(run['patch_indices'])}",flush=True)
        metadata.update(device="cuda",precision="float32",batch_size=run["arguments"]["batch_size"],
                        native_adapter_identity=model.benchmark_identity,
                        square_global_feature_check=dict(samples=4,maximum_absolute_error=float(np.abs(fresh-raw[:4]).max()),passed=True),
                        corrected_rectangular_patch_relative_l2=drift)
        current_code = code_identity()
        producing_run = dict(run,code=current_code)
        cache_key = feature_key(saved_before["source"],producing_run,key,Namespace(**run["arguments"]))
        saved = save_cache(root/"cache",cache_key,raw,normalized,patches,geometries,metadata,saved_before["source"])
        saved["extraction_provenance"] = dict(code=current_code, correction="sam-rectangular-rope-v2",global_features_reused_from=saved_before["key"])
        write_json(root/"cache"/f"{cache_key}.json",saved)
        run["caches"][key] = saved
        run.setdefault("model_extraction_overrides",{})[key] = saved["extraction_provenance"]
        del model,tokens,values
        gc.collect();torch.cuda.empty_cache()
        maps = render_patches(root/key,run["records"],patches,geometries,run["arguments"]["pc_indices"],run["seed"],run["models"][key]["name"])
        run["results"][key].update(patch_maps=maps,model_metadata=metadata)
        write_json(root/key/"result.json",run["results"][key])
        write_json(root/"run.json",run)
        results = [run["results"][k] for k in run["models"]]
        comparisons = render_comparisons(root,run["records"],results)
        render_index(root,run,results,comparisons)
        del patches,previous,raw,normalized
        gc.collect()
        attention = run_attention(root,run,[key],run.get("attention",{}).get("longest_side",longest),20,"cuda")
        run["attention"] = dict(state=attention["state"],models=list(attention["results"]),longest_side=attention["longest_side"])
        run["corrections"][-1].update(state="complete",finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/"run.json",run)
        render_index(root,run,results,comparisons)
        print(f"CORRECTED SAM COMPLETE: {root.name}",flush=True)


if __name__ == "__main__":
    main()
