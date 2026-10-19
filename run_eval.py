#!/usr/bin/env python3
"""Main CLI Orchestrator with High-Quality Bicubic Antialiased Transforms."""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import sys

import torch
import torchvision.transforms as T

from eval_suite.benchmarks.biota_probe import evaluate_biota
from eval_suite.benchmarks.brackish_probe import evaluate_brackish
from eval_suite.benchmarks.coralmask_probe import evaluate_coralmask
from eval_suite.benchmarks.coralscapes_probe import evaluate_coralscapes
from eval_suite.benchmarks.deepfish_probe import evaluate_deepfish
from eval_suite.benchmarks.fathomnet_probe import evaluate_fathomnet
from eval_suite.benchmarks.single_label import evaluate_knn, evaluate_single_label
from eval_suite.config import (
    BENTHIC_MEAN,
    BENTHIC_STD,
    BRACKISH_CLASS_NAMES,
    CHECKPOINTS_DIR,
    DEEPFISH_CLASS_NAMES,
    DEFAULT_PATHS,
    GERMAN_BANK_CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
    SUBSTRATE_CLASS_NAMES,
)
from eval_suite.datasets.biota import prepare_biota_data
from eval_suite.datasets.brackish import prepare_brackish_data
from eval_suite.datasets.deepfish import prepare_deepfish_data
from eval_suite.datasets.fathomnet import prepare_fathomnet_data
from eval_suite.datasets.single_label import prepare_single_label_data
from eval_suite.models import CLIP_MEAN, CLIP_STD, load_baseline_backbone, load_dinov3_backbone
from eval_suite.utils import plot_and_save_confusion_matrix, set_seed


def ensure_spatial_split_symlinks():
    """Ensures data/ has symlinks to the leakage-free spatial split files if present in data/benchmarks, data/splits, or root."""
    try:
        from eval_suite.config import REPO_ROOT, DATA_DIR
        data_dir = DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        links = [
            ("german_bank_2010_spatial_split.csv", "german_bank_spatial_split.csv", "benchmarks/german_bank"),
            ("substrate_depth_2_spatial_split.csv", "substrate_spatial_split.csv", "benchmarks/substrate"),
        ]
        for src_name, dst_name, rel_sub in links:
            candidates = [
                data_dir / rel_sub / src_name,
                REPO_ROOT / "data" / rel_sub / src_name,
                data_dir / "splits" / src_name,
                REPO_ROOT / "data" / "splits" / src_name,
                REPO_ROOT / src_name,
            ]
            src = next((c for c in candidates if c.exists()), None)
            dst = data_dir / dst_name
            if src and src.exists():
                if dst.is_symlink() or not dst.exists():
                    try:
                        dst.unlink(missing_ok=True)
                    except Exception:
                        pass
                if not dst.exists():
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        import shutil
                        shutil.copyfile(src, dst)
    except Exception:
        pass


def resolve_csv_path(path_str, fallback_names):
    """Resolves CSV or manifest path across standard repo and external data locations."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    from eval_suite.config import REPO_ROOT, DATA_DIR, GDRIVE_CANDIDATE_PATHS
    search_dirs = [
        DATA_DIR / "benchmarks" / "substrate",
        DATA_DIR / "benchmarks" / "german_bank",
        DATA_DIR / "benchmarks" / "coralmask",
        DATA_DIR / "benchmarks" / "biota",
        DATA_DIR / "benchmarks" / "deepfish",
        DATA_DIR / "benchmarks" / "fathomnet",
        DATA_DIR / "benchmarks" / "brackish",
        DATA_DIR / "benchmarks",
        DATA_DIR / "training",
        DATA_DIR / "splits",
        DATA_DIR / "manifests",
        DATA_DIR,
        REPO_ROOT / "data" / "benchmarks" / "substrate",
        REPO_ROOT / "data" / "benchmarks" / "german_bank",
        REPO_ROOT / "data" / "benchmarks" / "coralmask",
        REPO_ROOT / "data" / "benchmarks" / "biota",
        REPO_ROOT / "data" / "benchmarks" / "deepfish",
        REPO_ROOT / "data" / "benchmarks" / "fathomnet",
        REPO_ROOT / "data" / "benchmarks" / "brackish",
        REPO_ROOT / "data" / "benchmarks",
        REPO_ROOT / "data" / "training",
        REPO_ROOT / "data" / "splits",
        REPO_ROOT / "data" / "manifests",
        REPO_ROOT / "data",
        REPO_ROOT,
    ]
    for gd in GDRIVE_CANDIDATE_PATHS:
        search_dirs.extend([
            gd / "benchmarks" / "substrate",
            gd / "benchmarks" / "german_bank",
            gd / "benchmarks" / "coralmask",
            gd / "benchmarks" / "biota",
            gd / "benchmarks" / "deepfish",
            gd / "benchmarks" / "fathomnet",
            gd / "benchmarks" / "brackish",
            gd / "benchmarks",
            gd / "splits",
            gd / "manifests",
            gd,
        ])
    env_data = os.getenv("SEADINO_DATA_DIR")
    if env_data and env_data.strip():
        env_p = Path(env_data.strip())
        search_dirs.extend([
            env_p / "benchmarks" / "substrate",
            env_p / "benchmarks" / "german_bank",
            env_p / "benchmarks" / "coralmask",
            env_p / "benchmarks" / "biota",
            env_p / "benchmarks" / "deepfish",
            env_p / "benchmarks" / "fathomnet",
            env_p / "benchmarks" / "brackish",
            env_p / "benchmarks",
            env_p / "splits",
            env_p / "manifests",
            env_p,
        ])

    for fb in fallback_names:
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            cand = sdir / fb
            if cand.exists():
                print(f"  [Path Resolver] '{path_str}' not found -> using '{cand}'")
                return str(cand.resolve())
    return str(p)


def main():
    ensure_spatial_split_symlinks()

    p = argparse.ArgumentParser(description="Modular Marine Foundation Model Evaluator")
    p.add_argument("--checkpoints", nargs="+", default=[], help="Specific .ckpt files to score.")
    p.add_argument("--checkpoint_dirs", "--checkpoint_dir", nargs="+", default=[],
                   help="Directories containing *.ckpt files (accepts both --checkpoint_dirs and --checkpoint_dir).")
    p.add_argument("--include_off_the_shelf", action="store_true", default=False)
    p.add_argument("--split_protocol", choices=["legacy", "explicit-v1"], default="legacy",
                   help="Explicit-v1 separates training, validation selection and final test; legacy preserves previous commands.")
    p.add_argument("--eval_split", choices=["validation", "test"], default="validation",
                   help="Reported partition for explicit-v1 (default: validation).")
    p.add_argument("--prepare_splits_only", action="store_true",
                   help="Create/check explicit-v1 manifests without loading models or fitting probes.")
    p.add_argument("--split_dir", default=str(REPO_ROOT / "data/benchmarks/splits/explicit-v1"))
    p.add_argument("--probe_dir", default="results/probes/explicit-v1")
    p.add_argument("--coralscapes_epochs", type=int, default=10)
    p.add_argument("--coralscapes_batch_size", type=int, default=16)
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=["all", "substrate", "german_bank", "biota", "coralscapes", "coralmask", "deepfish", "fathomnet", "brackish"])

    # Path overrides
    p.add_argument("--data_dir", "--gdrive_dir", type=str, default=DEFAULT_PATHS.get("data_dir"),
                   help="Root data directory (can point to G drive, e.g. /mnt/g/data).")
    p.add_argument("--benthic_img_root", type=str, default=DEFAULT_PATHS["benthic_img_root"])
    p.add_argument("--substrate_csv", type=str, default=DEFAULT_PATHS["substrate_csv"])
    p.add_argument("--german_bank_csv", type=str, default=DEFAULT_PATHS["german_bank_csv"])
    p.add_argument("--biota_csv", type=str, default=DEFAULT_PATHS["biota_csv"])
    p.add_argument("--coralmask_dir", type=str, default=DEFAULT_PATHS["coralmask_dir"])
    p.add_argument("--coralmask_test_manifest", type=str, default=DEFAULT_PATHS["coralmask_test_manifest"],
                   help="Path to clean test manifest file for CoralMask (e.g. coralmask_test_clean.txt).")
    p.add_argument("--deepfish_dir", type=str, default=DEFAULT_PATHS.get("deepfish_dir", "data/benchmarks/deepfish"),
                   help="Directory containing DeepFish dataset (with Classification/ subdirectory or train.csv).")
    p.add_argument("--deepfish_split", type=str, default="test", choices=["val", "test"],
                   help="Evaluation split for DeepFish: 'test' (default) or 'val'.")
    p.add_argument("--fathomnet_dir", type=str, default=DEFAULT_PATHS.get("fathomnet_dir", "FathomNet/2024-vme-benchmark"),
                   help="Directory with FathomNet crops / metadata.csv or Hugging Face dataset name.")
    p.add_argument("--fathomnet_train_split", type=str, default="train")
    p.add_argument("--fathomnet_test_split", type=str, default="validation")
    p.add_argument("--fathomnet_max_train_samples", type=int, default=None,
                   help="Limit FathomNet train samples (e.g. 5000 for fast eval; None for full dataset).")
    p.add_argument("--fathomnet_max_test_samples", type=int, default=None,
                   help="Limit FathomNet test samples (e.g. 1000 for fast eval; None for full dataset).")
    p.add_argument("--fathomnet_epochs", type=int, default=20,
                   help="Number of epochs to train FathomNet linear probe head (default: 20).")
    p.add_argument("--fathomnet_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Cache exact resized FathomNet pixels and backbone features locally (default: enabled; --no-fathomnet_cache disables).")
    p.add_argument("--fathomnet_cache_dir", type=str, default=None,
                   help="FathomNet cache directory (default: ~/.cache/seadino/fathomnet, or SEADINO_FATHOMNET_CACHE). Prefer the WSL/Linux disk.")
    p.add_argument("--brackish_dir", type=str, default=DEFAULT_PATHS.get("brackish_dir", "data/benchmarks/brackish"),
                   help="Directory containing Brackish dataset (with train/test class folders or metadata.csv).")
    p.add_argument("--brackish_max_train_samples", type=int, default=None,
                   help="Limit Brackish train samples (e.g. 2000 for fast eval; None for full dataset).")
    p.add_argument("--brackish_max_test_samples", type=int, default=None,
                   help="Limit Brackish test samples (e.g. 1000 for fast eval; None for full dataset).")
    p.add_argument("--brackish_epochs", type=int, default=10,
                   help="Number of epochs to train Brackish linear probe head (default: 10).")
    p.add_argument("--model_id", type=str, default=DEFAULT_PATHS["model_id"])

    # Normalization strategy
    p.add_argument("--norm", type=str, default="auto", choices=["auto", "imagenet", "benthic"],
                   help="Normalization strategy: 'auto' (detect from checkpoint config), 'imagenet', or 'benthic'.")

    p.add_argument("--coralmask_max_train_samples", type=int, default=None,
                   help="Limit CoralMask train samples (e.g. 3000 for fast eval; None for full 38k dataset).")
    p.add_argument("--coralmask_epochs", type=int, default=6,
                   help="Number of epochs to train CoralMask linear probe head (default: 6).")
    p.add_argument("--deepfish_max_train_samples", type=int, default=None,
                   help="Limit DeepFish train samples (e.g. 5000 for fast eval; None for full dataset).")
    p.add_argument("--deepfish_max_test_samples", type=int, default=None,
                   help="Limit DeepFish test samples (e.g. 1000 for fast eval; None for full dataset).")
    p.add_argument("--deepfish_epochs", type=int, default=10,
                   help="Number of epochs to train DeepFish linear probe head (default: 10).")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume evaluation: skip checkpoints already recorded in output_csv.")
    p.add_argument("--use_bf16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42, help="Fixed random seed for determinism.")
    p.add_argument("--image_size", type=int, default=224,
                   help="Evaluation input resolution for ViT backbone (default: 224; supports higher resolutions like 448, 518 via dynamic bicubic pos-embed interpolation).")
    p.add_argument("--fathomnet_crop_boxes", action="store_true", default=False,
                   help="Crop bounding boxes for FathomNet instead of full original images (default: False).")
    p.add_argument("--brackish_crop_boxes", action=argparse.BooleanOptionalAction, default=False,
                   help="Crop bounding boxes for Brackish fauna instead of full original 1080p frames.")
    p.add_argument("--substrate_epochs", type=int, default=10,
                   help="Number of epochs to train Substrate Depth 2 linear probe head (default: 10).")
    p.add_argument("--german_bank_epochs", type=int, default=10,
                   help="Number of epochs to train German Bank linear probe head (default: 10).")
    p.add_argument("--baselines", nargs="+", default=[],
                   help="External foundation baselines to score (e.g. bioclip dinov2-small dinov2-base clip sam3-encoder sam3.1-encoder marineinst-encoder).")
    from eval_suite.sam3_backbone import SAM3_REVISION
    p.add_argument("--marineinst_checkpoint", default=None, help="Official MarineInst ViT-H Stage 1 .pth checkpoint (default: local SeaDino model cache).")
    p.add_argument("--marineinst_microbatch_size", type=int, default=4, help="Frozen MarineInst encoder microbatch; probe optimizer batch is unchanged.")
    p.add_argument("--sam3_revision", default=None, help="Override the release-specific pinned SAM 3 / SAM 3.1 revision.")
    p.add_argument("--sam3_model_path", default=None, help="Authorized local SAM directory: config.json plus model.safetensors (SAM 3) or sam3.1_multiplex.pt (SAM 3.1).")
    p.add_argument("--sam3_microbatch_size", type=int, default=2, help="Frozen SAM 3 encoder microbatch; probe optimizer batch is unchanged.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_cm_plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Save confusion matrix plots (default: True; use --no-save_cm_plots to disable).")
    p.add_argument("--cm_output_dir", type=str, default="eval_output/confusion_matrices")
    p.add_argument("--output_csv", type=str, default="master_benchmark_results.csv")
    args = p.parse_args()

    if args.prepare_splits_only and args.split_protocol != "explicit-v1":
        p.error("--prepare_splits_only requires --split_protocol explicit-v1")
    if args.eval_split == "test" and args.split_protocol != "explicit-v1":
        p.error("--eval_split test requires --split_protocol explicit-v1")
    if not args.prepare_splits_only and not args.checkpoints and not args.checkpoint_dirs and not args.include_off_the_shelf and not args.baselines:
        sys.exit("Error: Please pass --checkpoints, --checkpoint_dirs (or --checkpoint_dir), --include_off_the_shelf, or --baselines")

    # Resolve CSV and manifest paths with graceful fallback
    args.substrate_csv = resolve_csv_path(
        args.substrate_csv,
        ["substrate_depth_2_spatial_split.csv", "substrate_spatial_split.csv"]
    )
    args.german_bank_csv = resolve_csv_path(
        args.german_bank_csv,
        ["german_bank_2010_spatial_split.csv", "german_bank_spatial_split.csv"]
    )
    if args.biota_csv:
        args.biota_csv = resolve_csv_path(
            args.biota_csv,
            ["biota_spatial_split.csv", "benthicnet_nn.csv"]
        )
    if args.coralmask_test_manifest:
        args.coralmask_test_manifest = resolve_csv_path(
            args.coralmask_test_manifest,
            ["coralmask_test_clean.txt"]
        )
    else:
        auto_cm = resolve_csv_path("coralmask_test_clean.txt", ["coralmask_test_clean.txt"])
        if Path(auto_cm).exists():
            args.coralmask_test_manifest = auto_cm
            print(f"  [Manifest Resolver] Auto-detected clean test manifest -> '{args.coralmask_test_manifest}'")

    if args.split_protocol == "explicit-v1":
        from eval_suite.explicit_evaluation import run_explicit_evaluation
        run_explicit_evaluation(args)
        return
    print("Protocol: legacy (classification hold-outs; Coralscapes validation). Use explicit-v1 for separate validation/test reporting.")

    if args.output_csv:
        out_dir = Path(args.output_csv).parent
        if str(out_dir) and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32
    print(f"Device: {device} | Dtype: {dtype} | Master Seed: {args.seed} | Workers: {args.num_workers}")
    print(f"ViT Input Resolution: {args.image_size}x{args.image_size} (Bicubic antialiased)")
    print(f"Norm Policy: {args.norm.upper()} (auto = benthic if trained with --benthic_norm, else ImageNet)")

    selected = set(args.datasets)
    run_all = "all" in selected

    print("\n" + "="*80)
    print("STEP 1: PRE-INDEXING SELECTED BENCHMARKS")
    print("="*80)

    sub_train, sub_test, sub_num_cls = None, None, 0
    if run_all or "substrate" in selected:
        print("  • Indexing Substrate Depth 2...")
        sub_train, sub_test, sub_num_cls = prepare_single_label_data(args.substrate_csv, args.benthic_img_root, "catami_substrate")
        print(f"    Loaded {len(sub_train):,} Train | {len(sub_test):,} Test ({sub_num_cls} classes)")

    gb_train, gb_test, gb_num_cls = None, None, 0
    if run_all or "german_bank" in selected:
        print("  • Indexing German Bank 2010...")
        gb_train, gb_test, gb_num_cls = prepare_single_label_data(args.german_bank_csv, args.benthic_img_root, "substrate")
        print(f"    Loaded {len(gb_train):,} Train | {len(gb_test):,} Test ({gb_num_cls} classes)")

    bio_train, bio_test, bio_num_cls = None, None, 0
    if run_all or "biota" in selected:
        print("  • Indexing BenthicNet Biota (272 classes)...")
        bio_train, bio_test, bio_num_cls = prepare_biota_data(args.biota_csv, args.benthic_img_root)
        print(f"    Loaded {len(bio_train):,} Train | {len(bio_test):,} Test ({bio_num_cls} classes)")

    cs_train, cs_val, cs_num_cls = None, None, 0
    if run_all or "coralscapes" in selected:
        print("  • Loading Coralscapes from Hugging Face...")
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        cs_meta = hf_hub_download(repo_id="EPFL-ECEO/coralscapes", repo_type="dataset", filename="id2label.json")
        with open(cs_meta, "r") as f:
            cs_id2label = json.load(f)
        cs_num_cls = max([int(k) for k in cs_id2label.keys()]) + 1
        cs_ds = load_dataset("EPFL-ECEO/coralscapes")
        cs_train, cs_val = cs_ds["train"], cs_ds["validation"]
        print(f"    Loaded {len(cs_train):,} Train | {len(cs_val):,} Val ({cs_num_cls} classes)")

    df_train, df_test, df_num_cls, df_class_names = None, None, 0, []
    if run_all or "deepfish" in selected:
        print("  • Indexing DeepFish Classification (Fish vs No-Fish across 20 habitats)...")
        try:
            df_train, df_test, df_num_cls, df_class_names = prepare_deepfish_data(
                args.deepfish_dir,
                split_test=args.deepfish_split,
                max_train_samples=args.deepfish_max_train_samples,
                max_test_samples=args.deepfish_max_test_samples,
                seed=args.seed,
            )
            print(f"    Loaded {len(df_train):,} Train | {len(df_test):,} Test ({df_num_cls} classes)")
        except Exception as e:
            print(f"  [Warning: DeepFish indexing failed] {e}")
            if "deepfish" in selected and not run_all:
                raise

    fn_train, fn_test, fn_num_cls, fn_class_names = None, None, 0, []
    if run_all or "fathomnet" in selected:
        print("  • Indexing FathomNet Organism Classification...")
        try:
            fn_train, fn_test, fn_num_cls, fn_class_names = prepare_fathomnet_data(
                repo_or_dir=args.fathomnet_dir,
                split_train=args.fathomnet_train_split,
                split_test=args.fathomnet_test_split,
                max_train_samples=args.fathomnet_max_train_samples,
                max_test_samples=args.fathomnet_max_test_samples,
                seed=args.seed,
            )
            print(f"    Loaded {len(fn_train):,} Train | {len(fn_test):,} Test ({fn_num_cls} classes)")
        except Exception as e:
            print(f"  [Warning: FathomNet indexing failed] {e}")
            if "fathomnet" in selected and not run_all:
                raise

    br_train, br_test, br_num_cls, br_class_names = None, None, 0, []
    if run_all or "brackish" in selected:
        print("  • Indexing Brackish Marine Fauna Classification...")
        try:
            br_train, br_test, br_num_cls, br_class_names = prepare_brackish_data(
                data_dir=args.brackish_dir,
                max_train_samples=args.brackish_max_train_samples,
                max_test_samples=args.brackish_max_test_samples,
                seed=args.seed,
            )
            print(f"    Loaded {len(br_train):,} Train | {len(br_test):,} Test ({br_num_cls} classes)")
        except Exception as e:
            print(f"  [Warning: Brackish indexing failed] {e}")
            if "brackish" in selected and not run_all:
                raise

    # 2. Setup Jobs
    def _resolve_ckpt_path(c_str: str) -> str:
        cp = Path(c_str)
        if cp.is_file():
            return str(cp.resolve())
        for base in [CHECKPOINTS_DIR, Path("/mnt/d/Neel/Checkpoints"), REPO_ROOT, REPO_ROOT / "checkpoints"]:
            cand = base / c_str
            if cand.is_file():
                print(f"  [Checkpoint Resolver] '{c_str}' -> '{cand}'")
                return str(cand.resolve())
        return c_str

    def _resolve_ckpt_dir(d_str: str) -> str:
        dp = Path(d_str)
        if dp.is_dir():
            return str(dp.resolve())
        for base in [CHECKPOINTS_DIR, Path("/mnt/d/Neel/Checkpoints"), REPO_ROOT, REPO_ROOT / "checkpoints"]:
            cand = base / d_str
            if cand.is_dir():
                print(f"  [Checkpoint Dir Resolver] '{d_str}' -> '{cand}'")
                return str(cand.resolve())
        return d_str

    jobs = []
    for b in args.baselines:
        jobs.append((f"{b} (baseline)", ("baseline", b)))
    for c in args.checkpoints:
        resolved_c = _resolve_ckpt_path(c)
        jobs.append((f"{Path(resolved_c).parent.name}/{Path(resolved_c).name}", ("ckpt", resolved_c)))
    for d in args.checkpoint_dirs:
        resolved_d = _resolve_ckpt_dir(d)
        found_ckpts = sorted(glob.glob(str(Path(resolved_d) / "*.ckpt")))
        # Auto-redirect ssl_off to ssl_off_pix if empty
        if not found_ckpts and resolved_d.endswith("ssl_off") and Path(f"{resolved_d}_pix").is_dir():
            print(f"  [Auto-Redirect] '{resolved_d}' contains no checkpoints -> redirecting to '{resolved_d}_pix'")
            resolved_d = f"{resolved_d}_pix"
            found_ckpts = sorted(glob.glob(str(Path(resolved_d) / "*.ckpt")))
        for c in found_ckpts:
            jobs.append((f"{Path(resolved_d).name}/{Path(c).name}", ("ckpt", c)))
    if args.include_off_the_shelf:
        jobs.append(("off-the-shelf (no continued pretraining)", ("ckpt", None)))

    # Check already evaluated jobs if resume
    completed_labels = set()
    if args.resume and args.output_csv and os.path.isfile(args.output_csv):
        try:
            with open(args.output_csv, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    l = r.get("Model / Label", "")
                    if l:
                        completed_labels.add(l)
            print(f"  [Resume] Loaded {len(completed_labels)} already-evaluated models from {args.output_csv}")
        except Exception as e:
            print(f"  [Resume Warning] Could not read {args.output_csv}: {e}")

    print(f"\nTotal evaluation jobs queued: {len(jobs)}")

    # 3. Evaluate Models
    results = []
    for job_idx, (label, job_info) in enumerate(jobs, 1):
        if args.resume and label in completed_labels:
            print(f"\n[{job_idx}/{len(jobs)}] [Resume Skipping] Model [{label}] already recorded in {args.output_csv}")
            continue

        job_type, target = job_info if isinstance(job_info, tuple) else ("ckpt", job_info)
        ckpt_desc = target if target else ("(baseline)" if job_type == "baseline" else "(off-the-shelf baseline)")
        safe_label = label.replace(" ", "_").replace("/", "_")
        print(f"\n{'='*80}\n[{job_idx}/{len(jobs)}] EVALUATING MODEL: [{label}]\nPath: {ckpt_desc}\n{'='*80}")

        try:
            if job_type == "baseline":
                model, epoch, train_loss, ckpt_benthic_norm = load_baseline_backbone(
                    target, device, dtype, sam3_revision=args.sam3_revision,
                    sam3_model_path=args.sam3_model_path, sam3_microbatch_size=args.sam3_microbatch_size,
                    marineinst_checkpoint=args.marineinst_checkpoint,
                    marineinst_microbatch_size=args.marineinst_microbatch_size)
                use_benthic = False
                if getattr(model, "norm_mean", None) is not None:
                    active_mean, active_std = model.norm_mean, model.norm_std
                    norm_label = getattr(model, "norm_label", "CLIP")
                else:
                    active_mean, active_std = IMAGENET_MEAN, IMAGENET_STD
                    norm_label = "ImageNet"
            else:
                model, epoch, train_loss, ckpt_benthic_norm = load_dinov3_backbone(target, args.model_id, device, dtype)
                # Determine normalization strategy per model
                if args.norm == "auto":
                    use_benthic = ckpt_benthic_norm
                elif args.norm == "benthic":
                    use_benthic = True
                else:
                    use_benthic = False

                active_mean = BENTHIC_MEAN if use_benthic else IMAGENET_MEAN
                active_std = BENTHIC_STD if use_benthic else IMAGENET_STD
                norm_label = "BenthicNet" if use_benthic else "ImageNet"

            print(f"  --> Normalization: {norm_label} (mean={active_mean}, std={active_std})")

            # Dynamic Bicubic Antialiased transform at specified resolution with model-appropriate normalization
            eval_size = (args.image_size, args.image_size)
            val_transform_224 = T.Compose([
                T.Resize(eval_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=active_mean, std=active_std),
            ])

            row = {
                "Model / Label": label,
                "Epoch": epoch if epoch is not None else "-",
                "Norm": norm_label,
            }

            # Substrate Depth 2
            if run_all or "substrate" in selected:
                print(f"  --> Evaluating Substrate Depth 2 (k-NN + {args.substrate_epochs}ep Linear Probe)...")
                sub_res = evaluate_single_label(
                    model, sub_train, sub_test, sub_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.substrate_epochs, seed=args.seed
                )
                row["Substrate Lin Acc (%)"] = f"{sub_res['lin_accuracy']:.2f}%"
                row["Substrate Lin F1-Macro (%)"] = f"{sub_res['lin_macro_f1']:.2f}%"
                row["Substrate kNN Acc (%)"] = f"{sub_res['knn_accuracy']:.2f}%"
                row["Substrate kNN F1-Macro (%)"] = f"{sub_res['knn_macro_f1']:.2f}%"
                row["Substrate Acc (%)"] = f"{sub_res['knn_accuracy']:.2f}%"
                row["Substrate F1-Macro (%)"] = f"{sub_res['knn_macro_f1']:.2f}%"
                print(f"      Substrate Lin Acc: {sub_res['lin_accuracy']:.2f}% | Lin Macro-F1: {sub_res['lin_macro_f1']:.2f}%")
                print(f"      Substrate k-NN Acc: {sub_res['knn_accuracy']:.2f}% | k-NN Macro-F1: {sub_res['knn_macro_f1']:.2f}%")

                if args.save_cm_plots:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_substrate_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        sub_res["confusion_matrix_norm"], SUBSTRATE_CLASS_NAMES, cm_path,
                        title=f"CATAMI Substrate-d2: {label} ({norm_label} Norm)"
                    )

            # German Bank 2010
            if run_all or "german_bank" in selected:
                print(f"  --> Evaluating German Bank 2010 (k-NN + {args.german_bank_epochs}ep Linear Probe)...")
                gb_res = evaluate_single_label(
                    model, gb_train, gb_test, gb_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.german_bank_epochs, seed=args.seed
                )
                row["German Bank Lin Acc (%)"] = f"{gb_res['lin_accuracy']:.2f}%"
                row["German Bank Lin F1-Macro (%)"] = f"{gb_res['lin_macro_f1']:.2f}%"
                row["German Bank kNN Acc (%)"] = f"{gb_res['knn_accuracy']:.2f}%"
                row["German Bank kNN F1-Macro (%)"] = f"{gb_res['knn_macro_f1']:.2f}%"
                row["German Bank Acc (%)"] = f"{gb_res['knn_accuracy']:.2f}%"
                row["German Bank F1-Macro (%)"] = f"{gb_res['knn_macro_f1']:.2f}%"
                print(f"      German Bank Lin Acc: {gb_res['lin_accuracy']:.2f}% | Lin Macro-F1: {gb_res['lin_macro_f1']:.2f}%")
                print(f"      German Bank k-NN Acc: {gb_res['knn_accuracy']:.2f}% | k-NN Macro-F1: {gb_res['knn_macro_f1']:.2f}%")

                if args.save_cm_plots:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_german_bank_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        gb_res["confusion_matrix_norm"], GERMAN_BANK_CLASS_NAMES, cm_path,
                        title=f"German Bank 2010: {label} ({norm_label} Norm)"
                    )

            # Biota
            if run_all or "biota" in selected:
                print("  --> Evaluating BenthicNet Biota (k-NN + Linear Probe, Bicubic Antialiased)...")
                bio_res = evaluate_biota(model, bio_train, bio_test, bio_num_cls, val_transform_224, device, dtype, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
                row["Biota Lin mAP (%)"] = f"{bio_res['biota_lin_mAP']:.2f}%"
                row["Biota Lin F1-Macro (%)"] = f"{bio_res['biota_lin_macro_f1']:.2f}%"
                row["Biota kNN mAP (%)"] = f"{bio_res['biota_knn_mAP']:.2f}%"

            # Coralscapes
            if run_all or "coralscapes" in selected:
                print("  --> Evaluating Coralscapes Semantic Segmentation (10 epochs)...")
                cs_res = evaluate_coralscapes(model, cs_train, cs_val, cs_num_cls, device, dtype, batch_size=16, num_workers=args.num_workers, seed=args.seed, mean=active_mean, std=active_std)
                row["Coralscapes mIoU (%)"] = f"{cs_res['coralscapes_mIoU']:.2f}%"
                row["Coralscapes Pixel Acc (%)"] = f"{cs_res['coralscapes_pixel_acc']:.2f}%"
                print(f"      Coralscapes mIoU: {cs_res['coralscapes_mIoU']:.2f}% | Pix Acc: {cs_res['coralscapes_pixel_acc']:.2f}%")

            # CoralMask
            if run_all or "coralmask" in selected:
                manifest_str = f" [Manifest: {Path(args.coralmask_test_manifest).name}]" if args.coralmask_test_manifest else ""
                print(f"  --> Evaluating CoralMask Dense Binary Segmentation ({args.coralmask_epochs} epochs){manifest_str}...")
                cm_res = evaluate_coralmask(
                    model, args.coralmask_dir, device, dtype,
                    batch_size=16, num_workers=min(4, args.num_workers), seed=args.seed,
                    epochs=args.coralmask_epochs,
                    mean=active_mean, std=active_std,
                    test_manifest=args.coralmask_test_manifest,
                    max_train_samples=args.coralmask_max_train_samples,
                )
                row["CoralMask Coral-IoU (%)"] = f"{cm_res['coralmask_coral_iou']:.2f}%"
                row["CoralMask mIoU (%)"] = f"{cm_res['coralmask_mIoU']:.2f}%"
                print(f"      CoralMask mIoU: {cm_res['coralmask_mIoU']:.2f}% | Coral-IoU: {cm_res['coralmask_coral_iou']:.2f}%")

            # DeepFish
            if (run_all or "deepfish" in selected) and df_train is not None and df_test is not None:
                print(f"  --> Evaluating DeepFish Fish Classification (k-NN + {args.deepfish_epochs}ep Linear Probe)...")
                df_res = evaluate_deepfish(
                    model, df_train, df_test, df_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    seed=args.seed, epochs=args.deepfish_epochs,
                )
                row["DeepFish Lin Acc (%)"] = f"{df_res['deepfish_lin_acc']:.2f}%"
                row["DeepFish Lin F1-Macro (%)"] = f"{df_res['deepfish_lin_macro_f1']:.2f}%"
                row["DeepFish Lin Fish-F1 (%)"] = f"{df_res['deepfish_lin_fish_f1']:.2f}%"
                row["DeepFish kNN Acc (%)"] = f"{df_res['deepfish_knn_acc']:.2f}%"
                row["DeepFish kNN F1-Macro (%)"] = f"{df_res['deepfish_knn_macro_f1']:.2f}%"
                row["DeepFish kNN Fish-F1 (%)"] = f"{df_res['deepfish_knn_fish_f1']:.2f}%"
                print(f"      DeepFish Lin Acc: {df_res['deepfish_lin_acc']:.2f}% | Lin Macro-F1: {df_res['deepfish_lin_macro_f1']:.2f}% | Fish-F1: {df_res['deepfish_lin_fish_f1']:.2f}%")
                print(f"      DeepFish k-NN Acc: {df_res['deepfish_knn_acc']:.2f}% | k-NN Macro-F1: {df_res['deepfish_knn_macro_f1']:.2f}% | Fish-F1: {df_res['deepfish_knn_fish_f1']:.2f}%")

                if args.save_cm_plots:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_deepfish_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        df_res["confusion_matrix_norm"], df_class_names, cm_path,
                        title=f"DeepFish: {label} ({norm_label} Norm)"
                    )

            # FathomNet
            if (run_all or "fathomnet" in selected) and fn_train is not None and fn_test is not None:
                print(f"  --> Evaluating FathomNet Organism Classification (k-NN + {args.fathomnet_epochs}ep Linear Probe)...")
                fn_res = evaluate_fathomnet(
                    model, fn_train, fn_test, fn_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    seed=args.seed, epochs=args.fathomnet_epochs,
                    crop_boxes=args.fathomnet_crop_boxes,
                    cache_enabled=args.fathomnet_cache, cache_dir=args.fathomnet_cache_dir,
                )
                row["FathomNet Lin Acc (%)"] = f"{fn_res['fathomnet_lin_acc']:.2f}%"
                row["FathomNet Lin F1-Macro (%)"] = f"{fn_res['fathomnet_lin_macro_f1']:.2f}%"
                row["FathomNet kNN Acc (%)"] = f"{fn_res['fathomnet_knn_acc']:.2f}%"
                row["FathomNet kNN F1-Macro (%)"] = f"{fn_res['fathomnet_knn_macro_f1']:.2f}%"
                print(f"      FathomNet Lin Acc: {fn_res['fathomnet_lin_acc']:.2f}% | Lin Macro-F1: {fn_res['fathomnet_lin_macro_f1']:.2f}%")
                print(f"      FathomNet k-NN Acc: {fn_res['fathomnet_knn_acc']:.2f}% | k-NN Macro-F1: {fn_res['fathomnet_knn_macro_f1']:.2f}%")

                if args.save_cm_plots and "confusion_matrix_norm" in fn_res and fn_class_names and len(fn_class_names) <= 50:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_fathomnet_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        fn_res["confusion_matrix_norm"], fn_class_names, cm_path,
                        title=f"FathomNet: {label} ({norm_label} Norm)"
                    )

            # Brackish
            if (run_all or "brackish" in selected) and br_train is not None and br_test is not None:
                br_desc = "Cropped Bounding Boxes" if args.brackish_crop_boxes else "Full 1080p Frames"
                print(f"  --> Evaluating Brackish Marine Fauna Classification [{br_desc}] (k-NN + {args.brackish_epochs}ep Linear Probe)...")
                br_res = evaluate_brackish(
                    model, br_train, br_test, br_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=min(4, args.num_workers),
                    seed=args.seed, epochs=args.brackish_epochs,
                    crop_boxes=args.brackish_crop_boxes,
                )
                row["Brackish Lin Acc (%)"] = f"{br_res['brackish_lin_acc']:.2f}%"
                row["Brackish Lin F1-Macro (%)"] = f"{br_res['brackish_lin_macro_f1']:.2f}%"
                row["Brackish kNN Acc (%)"] = f"{br_res['brackish_knn_acc']:.2f}%"
                row["Brackish kNN F1-Macro (%)"] = f"{br_res['brackish_knn_macro_f1']:.2f}%"
                print(f"      Brackish Lin Acc: {br_res['brackish_lin_acc']:.2f}% | Lin Macro-F1: {br_res['brackish_lin_macro_f1']:.2f}%")
                print(f"      Brackish k-NN Acc: {br_res['brackish_knn_acc']:.2f}% | k-NN Macro-F1: {br_res['brackish_knn_macro_f1']:.2f}%")

                if args.save_cm_plots and "confusion_matrix_norm" in br_res and br_class_names:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_brackish_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        br_res["confusion_matrix_norm"], br_class_names, cm_path,
                        title=f"Brackish: {label} ({norm_label} Norm)"
                    )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append(row)

            # Incremental save so progress is never lost
            if args.output_csv:
                try:
                    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
                    file_exists = os.path.isfile(args.output_csv)
                    fieldnames = list(row.keys())
                    if file_exists:
                        with open(args.output_csv, "r") as f:
                            header_line = f.readline().strip()
                            if header_line:
                                fieldnames = [col.strip() for col in header_line.split(",")]
                                for k in row.keys():
                                    if k not in fieldnames:
                                        fieldnames.append(k)
                    with open(args.output_csv, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception as ex:
                    print(f"  [Warning: CSV incremental write failed] {ex}")

        except Exception as e:
            print(f"  ❌ Error evaluating {ckpt_desc}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Master Table Output
    if not results:
        print("\nNo newly evaluated results to display.")
        return

    print("\n" + "="*140)
    print("MASTER BENCHMARK RESULTS TABLE")
    print("="*140)

    keys = list(results[0].keys())
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in results)) + 2 for k in keys}
    header_str = "".join([f"{k:<{col_widths[k]}}" for k in keys])
    print(header_str)
    print("-" * len(header_str))

    for r in results:
        row_str = "".join([f"{str(r.get(k, '')):<{col_widths[k]}}" for k in keys])
        print(row_str)
    print("="*140)

    if args.output_csv:
        print(f"\nAll results saved to: {args.output_csv}")
        if args.save_cm_plots:
            print(f"Saved confusion matrix heatmaps in: {args.cm_output_dir}/")


if __name__ == "__main__":
    main()
