"""Single command for selection, extraction, CPU analysis, and offline reporting."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .common import code_identity, digest, file_sha, write_json
from .data import CLASSIFICATION, SEGMENTATION, benchmark_records, folder_records, image_records, sample_summary, select_records, stage_images


def parser():
    p = argparse.ArgumentParser(description="Frozen backbone feature explorer: image projections and spatial patch PCA")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=CLASSIFICATION + SEGMENTATION)
    source.add_argument("--input_dir", help="Recursively scan an image folder, or accept a single image file")
    source.add_argument("--image_path", help="A single image, useful for patch PCA maps")
    p.add_argument("--split", choices=("train", "validation", "test"), default="train")
    p.add_argument("--folder_labels", choices=("auto", "none", "parent"), default="auto")
    p.add_argument("--manifest_dir", help="Existing explicit-v1 manifest directory")
    p.add_argument("--data_root", help="Override DeepFish or CoralMask data location")
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--model_config", help="Alternative models.json, including custom checkpoint paths")
    p.add_argument("--methods", nargs="+", choices=("pca", "umap", "tsne"), default=["pca", "umap", "tsne"])
    p.add_argument("--dims", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--patch_images", type=int, default=20)
    p.add_argument("--patch_longest", type=int, default=512)
    p.add_argument("--image_size", type=int, default=224, help="Square global-feature resolution; padded to each model's patch multiple")
    p.add_argument("--pc_indices", type=int, nargs=3, default=[1, 2, 3])
    p.add_argument("--n_clusters", type=int, default=0, help="0 = class count, or 8 for unlabeled images")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--output_dir", help="Default: /mnt/d/Neel/FeatureVisualizations/<UTC timestamp>_<source>")
    p.add_argument("--render_only", action="store_true", help="Use the verified caches in --output_dir; no model loading or GPU access")
    p.add_argument("--attention_maps", action="store_true", help="Add real encoder attention views after feature visualization")
    p.add_argument("--attention_only", action="store_true", help="Attach attention maps to the existing --output_dir without recomputing embeddings")
    p.add_argument("--attention_images", type=int, default=20)
    p.add_argument("--attention_longest", type=int, default=512, help="Aspect-preserved attention input longest edge")
    return p


def versions():
    names = ("torch", "torchvision", "transformers", "numpy", "Pillow", "scikit-learn", "umap-learn", "plotly", "matplotlib")
    return {name: importlib.metadata.version(name) for name in names}


def feature_key(source, run, key, args):
    code = run["code"]
    extraction_code = {k: v for k, v in code.items() if k.startswith("eval_suite/") or Path(k).name in ("backbones.py", "data.py", "common.py")}
    runtime = {k: v for k, v in run["versions"].items() if k in ("torch", "torchvision", "transformers", "numpy", "Pillow")}
    return digest(dict(model=source, images=[(r["sample_id"], r["image_sha256"]) for r in run["records"]],
                       patch_indices=run["patch_indices"], patch_longest=args.patch_longest,
                       global_size=[getattr(args, "image_size", 224)] * 2, resize="bicubic-antialiased; reflection pad to patch multiple", precision="float32", batch_size=args.batch_size,
                       code=extraction_code, runtime=runtime, model_key=key))


def save_cache(directory, cache_key, raw, norm, patches, geometries, metadata, source):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = directory / f"{cache_key}.npz"
    temporary = arrays.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, raw=raw, normalized=norm, **{f"patch_{i}": p for i, p in enumerate(patches)})
    temporary.replace(arrays)
    saved = dict(key=cache_key, arrays=str(arrays), arrays_sha256=file_sha(arrays), geometry=geometries,
                 model_metadata=metadata, source=source, patch_count=len(patches))
    write_json(directory / f"{cache_key}.json", saved)
    return saved


def load_cache(saved):
    if file_sha(saved["arrays"]) != saved["arrays_sha256"]:
        raise ValueError(f"Feature cache checksum failed: {saved['arrays']}")
    with np.load(saved["arrays"], allow_pickle=False) as arrays:
        raw, norm = arrays["raw"].copy(), arrays["normalized"].copy()
        patches = [arrays[f"patch_{i}"].copy() for i in range(saved["patch_count"])]
    if not np.isfinite(raw).all() or not np.isfinite(norm).all() or any(not np.isfinite(p).all() for p in patches):
        raise ValueError("Feature cache contains non-finite values")
    return raw, norm, patches, saved["geometry"]


def write_samples(root, records):
    keys = ("sample_id", "image", "label_name", "partition", "image_sha256", "width", "height")
    with (root / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.patch_images < 0 or args.patch_longest < 1 or args.image_size < 1 or args.n_clusters < 0:
        parser().error("batch_size/patch_longest must be positive; patch_images/n_clusters cannot be negative")
    if args.attention_only and args.render_only:
        parser().error("--attention_only computes maps; --render_only reads cached outputs. Choose one.")
    if (args.attention_maps or args.attention_only) and (args.attention_images < 1 or args.attention_longest < 1):
        parser().error("attention_images and attention_longest must be positive")
    if not args.render_only and not args.attention_only and not (args.dataset or args.input_dir or args.image_path):
        parser().error("Provide --dataset, --input_dir, or --image_path")
    if (args.render_only or args.attention_only) and not args.output_dir:
        parser().error("--render_only/--attention_only requires --output_dir")
    # Models and datasets are already cached for this project; never trigger background downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/seadino_feature_mpl")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    from .backbones import selected_specs
    from .render import prepare_assets, render_comparisons, render_index, render_patches, render_scatter
    from .analysis import cluster, projection
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = Path(args.output_dir or f"/mnt/d/Neel/FeatureVisualizations/{stamp}_{args.dataset or 'folder'}").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / "run.json"
    if args.render_only or args.attention_only:
        run = json.loads(run_path.read_text())
        specs = [(key, value) for key, value in run["models"].items()]
        if args.models != ["all"]:
            specs = [(key, value) for key, value in specs if key in args.models]
            if len(specs) != len(args.models):
                raise ValueError("Requested models are not available in this run")
        args.seed = run["seed"]
        records = run["records"]
        # Attention verifies its selected images itself; do not hash 2,000 unused
        # originals or rebuild Plotly assets when attaching maps to twenty images.
        if not args.attention_only:
            for record in records:
                if file_sha(record["staged_image"]) != record["image_sha256"]:
                    raise ValueError(f"Staged image changed: {record['sample_id']}")
                if record.get("mask_path") and file_sha(record["mask_path"]) != record["mask_sha256"]:
                    raise ValueError(f"Semantic mask changed: {record['sample_id']}")
    else:
        specs = selected_specs(args.models, args.model_config)
        if args.dataset:
            original, source = benchmark_records(args.dataset, args.split, args.manifest_dir, args.data_root)
        elif args.image_path:
            original, source = image_records(args.image_path)
        else:
            original, source = folder_records(args.input_dir, args.folder_labels)
        selection_limit = args.max_images
        if source["task"] == "segmentation":
            if args.patch_images < 1:
                raise ValueError("Segmentation inputs provide patch maps only; patch_images must be positive")
            selection_limit = min(selection_limit, args.patch_images)
            source["visualization_mode"] = "patch-only; selection capped at patch_images"
        selected = select_records(original, selection_limit, args.seed)
        records = stage_images(selected, root)
        if args.dataset == "coralscapes":
            from .dense import stage_ground_truth
            stage_ground_truth(records, root, source)
        patch_indices = list(range(min(args.patch_images, len(records))))
        run = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), state="running", seed=args.seed,
                   source_name=args.dataset or Path(args.input_dir or args.image_path).name, source=source, arguments=vars(args),
                   records=records, sample_summary=sample_summary(records), patch_indices=patch_indices,
                   models=dict(specs), caches={}, results={}, code=code_identity(), versions=versions())
        write_samples(root, records)
        write_json(run_path, run)
    if not args.attention_only:
        prepare_assets(root, records)
    print(f"OUTPUT_DIR={root}", flush=True)
    if args.attention_only:
        from .attention import run_attention
        state = run_attention(root, run, [key for key, _ in specs], args.attention_longest, args.attention_images, args.device)
        run["attention"] = dict(state=state["state"], models=list(state["results"]), longest_side=args.attention_longest)
        write_json(run_path, run)
        results = [run["results"][key] for key in run["models"] if key in run["results"]]
        comparisons = [str(p.relative_to(root)) for p in sorted((root / "patch_comparisons").glob("*.png"))]
        render_index(root, run, results, comparisons)
        print(f"ATTENTION COMPLETE: {root / 'attention_comparisons/index.html'}", flush=True)
        return
    segmentation = run["source"]["task"] == "segmentation"
    labels = [r.get("label_name") for r in records]
    all_results = []
    started = time.monotonic()
    try:
        for key, spec in specs:
            print(f"\n=== {spec['name']} ===", flush=True)
            if args.render_only:
                if key not in run["caches"]:
                    raise ValueError(f"No completed extraction cache for {key}")
                saved = run["caches"][key]
            else:
                import torch
                from .backbones import extract, load_backbone, source_identity
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                source = source_identity(spec)
                cache_key = feature_key(source, run, key, args)
                saved_path = root / "cache" / f"{cache_key}.json"
                if saved_path.exists():
                    saved = json.loads(saved_path.read_text())
                    print("Verified extraction cache hit", flush=True)
                else:
                    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
                    model, metadata = load_backbone(spec, device)
                    actual_size = args.image_size + (-args.image_size) % metadata["patch_size"]
                    metadata["global_resolution"] = dict(requested=[args.image_size] * 2, padded=[actual_size] * 2, benchmark_default=args.image_size == 224)
                    raw, norm, patches, geometry = extract(model, metadata, records, run["patch_indices"], args.batch_size, args.patch_longest, device, args.image_size)
                    metadata.update(device=str(device), precision="float32", batch_size=args.batch_size)
                    saved = save_cache(root / "cache", cache_key, raw, norm, patches, geometry, metadata, source)
                    del model, raw, norm, patches
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                run["caches"][key] = saved
                write_json(run_path, run)
            raw, features, patches, geometries = load_cache(saved)
            if len(features) != len(records):
                raise ValueError("Feature/sample alignment mismatch")
            ids, metrics = cluster(features, labels, args.n_clusters, args.seed)
            model_root = root / key
            model_root.mkdir(exist_ok=True)
            plots = []
            if not segmentation:
                for method in dict.fromkeys(args.methods):
                    for dimensions in dict.fromkeys(args.dims):
                        print(f"  {method.upper()} {dimensions}D", flush=True)
                        coordinates, settings = projection(features, method, dimensions, args.seed)
                        settings.update(feature_cache_key=saved["key"], label_fitting=False, metrics=metrics,
                                        versions={k: run["versions"][k] for k in ("scikit-learn", "umap-learn", "numpy")})
                        plots.append(render_scatter(model_root, root, spec["name"], coordinates, records, ids, method, settings))
            maps = render_patches(model_root, records, patches, geometries, args.pc_indices, args.seed, spec["name"])
            result = dict(name=spec["name"], size=spec["size"], metrics=metrics, plots=plots, patch_maps=maps, model_metadata=saved["model_metadata"])
            run["results"][key] = result
            # A subset rerender must retain the other completed model cards and maps.
            all_results = [run["results"][model_key] for model_key in run["models"] if model_key in run["results"]]
            write_json(model_root / "result.json", result)
            write_json(run_path, run)
            comparisons = render_comparisons(root, records, all_results)
            render_index(root, run, all_results, comparisons)
            del raw, features, patches
            gc.collect()
        run.update(state="complete", finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic() - started, 2))
        if args.render_only:
            run["last_render"] = dict(arguments=vars(args), code=code_identity(), versions=versions())
        if args.attention_maps:
            from .attention import run_attention
            state = run_attention(root, run, [key for key, _ in specs], args.attention_longest, args.attention_images, args.device, cached_only=args.render_only)
            run["attention"] = dict(state=state["state"], models=list(state["results"]), longest_side=args.attention_longest)
            render_index(root, run, all_results, comparisons)
        write_json(run_path, run)
        print(f"\nCOMPLETE: {root / 'index.html'}", flush=True)
    except Exception as exc:
        run.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(run_path, run)
        raise


if __name__ == "__main__":
    main()
