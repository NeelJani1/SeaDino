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
from eval_suite.benchmarks.coralmask_probe import evaluate_coralmask
from eval_suite.benchmarks.coralscapes_probe import evaluate_coralscapes
from eval_suite.benchmarks.knn_probe import evaluate_knn
from eval_suite.config import (
    BENTHIC_MEAN,
    BENTHIC_STD,
    DEFAULT_PATHS,
    GERMAN_BANK_CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUBSTRATE_CLASS_NAMES,
)
from eval_suite.datasets.biota import prepare_biota_data
from eval_suite.datasets.single_label import prepare_single_label_data
from eval_suite.models import load_dinov3_backbone
from eval_suite.utils import plot_and_save_confusion_matrix, set_seed


def ensure_spatial_split_symlinks():
    """Ensures data/ has symlinks to the leakage-free spatial split files if present in root."""
    try:
        import shutil
        data_dir = Path("/home/njan320/SeaDino/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        links = [
            ("german_bank_2010_spatial_split.csv", "german_bank_spatial_split.csv"),
            ("substrate_depth_2_spatial_split.csv", "substrate_spatial_split.csv"),
        ]
        for src_name, dst_name in links:
            src = Path("/home/njan320/SeaDino") / src_name
            dst = data_dir / dst_name
            if src.exists() and not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
    except Exception:
        pass


def resolve_csv_path(path_str, fallback_names):
    """Resolves CSV path, checking fallbacks if user-provided path does not exist."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    for fb in fallback_names:
        for candidate in [
            Path("/home/njan320/SeaDino") / fb,
            Path("/home/njan320/SeaDino/data") / fb,
            Path(fb),
        ]:
            if candidate.exists():
                print(f"  [Path Resolver] '{path_str}' not found -> using '{candidate}'")
                return str(candidate)
    return str(p)


def main():
    ensure_spatial_split_symlinks()

    p = argparse.ArgumentParser(description="Modular Marine Foundation Model Evaluator")
    p.add_argument("--checkpoints", nargs="+", default=[], help="Specific .ckpt files to score.")
    p.add_argument("--checkpoint_dirs", "--checkpoint_dir", nargs="+", default=[],
                   help="Directories containing *.ckpt files (accepts both --checkpoint_dirs and --checkpoint_dir).")
    p.add_argument("--include_off_the_shelf", action="store_true", default=False)
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=["all", "substrate", "german_bank", "biota", "coralscapes", "coralmask"])

    # Path overrides
    p.add_argument("--benthic_img_root", type=str, default=DEFAULT_PATHS["benthic_img_root"])
    p.add_argument("--substrate_csv", type=str, default=DEFAULT_PATHS["substrate_csv"])
    p.add_argument("--german_bank_csv", type=str, default=DEFAULT_PATHS["german_bank_csv"])
    p.add_argument("--biota_csv", type=str, default=DEFAULT_PATHS["biota_csv"])
    p.add_argument("--coralmask_dir", type=str, default=DEFAULT_PATHS["coralmask_dir"])
    p.add_argument("--coralmask_test_manifest", type=str, default=None,
                   help="Path to clean test manifest file for CoralMask (e.g. coralmask_test_clean.txt).")
    p.add_argument("--model_id", type=str, default=DEFAULT_PATHS["model_id"])

    # Normalization strategy
    p.add_argument("--norm", type=str, default="auto", choices=["auto", "imagenet", "benthic"],
                   help="Normalization strategy: 'auto' (detect from checkpoint config), 'imagenet', or 'benthic'.")

    p.add_argument("--coralmask_max_train_samples", type=int, default=None,
                   help="Limit CoralMask train samples (e.g. 3000 for fast eval; None for full 38k dataset).")
    p.add_argument("--coralmask_epochs", type=int, default=6,
                   help="Number of epochs to train CoralMask linear probe head (default: 6).")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume evaluation: skip checkpoints already recorded in output_csv.")
    p.add_argument("--use_bf16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42, help="Fixed random seed for determinism.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_cm_plots", action="store_true", default=True)
    p.add_argument("--cm_output_dir", type=str, default="eval_output/confusion_matrices")
    p.add_argument("--output_csv", type=str, default="master_benchmark_results.csv")
    args = p.parse_args()

    if not args.checkpoints and not args.checkpoint_dirs and not args.include_off_the_shelf:
        sys.exit("Error: Please pass --checkpoints, --checkpoint_dirs (or --checkpoint_dir), or --include_off_the_shelf")

    # Resolve CSV paths with graceful fallback
    args.substrate_csv = resolve_csv_path(
        args.substrate_csv,
        ["substrate_depth_2_spatial_split.csv", "substrate_spatial_split.csv"]
    )
    args.german_bank_csv = resolve_csv_path(
        args.german_bank_csv,
        ["german_bank_2010_spatial_split.csv", "german_bank_spatial_split.csv"]
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32
    print(f"Device: {device} | Dtype: {dtype} | Master Seed: {args.seed} | Workers: {args.num_workers}")
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

    # 2. Setup Jobs
    jobs = []
    for c in args.checkpoints:
        jobs.append((Path(c).parent.name, c))
    for d in args.checkpoint_dirs:
        found_ckpts = sorted(glob.glob(str(Path(d) / "*.ckpt")))
        # Auto-redirect ssl_off to ssl_off_pix if empty
        if not found_ckpts and d.endswith("ssl_off") and Path(f"{d}_pix").is_dir():
            print(f"  [Auto-Redirect] '{d}' contains no checkpoints -> redirecting to '{d}_pix'")
            found_ckpts = sorted(glob.glob(str(Path(f"{d}_pix") / "*.ckpt")))
            d = f"{d}_pix"
        for c in found_ckpts:
            jobs.append((f"{Path(d).name}/{Path(c).name}", c))
    if args.include_off_the_shelf:
        jobs.append(("off-the-shelf (no continued pretraining)", None))

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
    for job_idx, (label, ckpt_path) in enumerate(jobs, 1):
        if args.resume and label in completed_labels:
            print(f"\n[{job_idx}/{len(jobs)}] [Resume Skipping] Model [{label}] already recorded in {args.output_csv}")
            continue

        ckpt_desc = ckpt_path if ckpt_path else "(off-the-shelf baseline)"
        safe_label = label.replace(" ", "_").replace("/", "_")
        print(f"\n{'='*80}\n[{job_idx}/{len(jobs)}] EVALUATING MODEL: [{label}]\nPath: {ckpt_desc}\n{'='*80}")

        try:
            model, epoch, train_loss, ckpt_benthic_norm = load_dinov3_backbone(ckpt_path, args.model_id, device, dtype)

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

            # Dynamic 224x224 Bicubic Antialiased transform with model-appropriate normalization
            val_transform_224 = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
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
                print("  --> Evaluating Substrate Depth 2 (k-NN, Bicubic Antialiased)...")
                sub_res = evaluate_knn(model, sub_train, sub_test, sub_num_cls, val_transform_224, device, dtype, batch_size=args.batch_size, num_workers=args.num_workers)
                row["Substrate Acc (%)"] = f"{sub_res['accuracy']:.2f}%"
                row["Substrate F1-Macro (%)"] = f"{sub_res['macro_f1']:.2f}%"
                print(f"      Substrate Acc: {sub_res['accuracy']:.2f}% | Macro-F1: {sub_res['macro_f1']:.2f}%")

                if args.save_cm_plots:
                    cm_path = os.path.join(args.cm_output_dir, f"cm_substrate_{safe_label}.png")
                    plot_and_save_confusion_matrix(
                        sub_res["confusion_matrix_norm"], SUBSTRATE_CLASS_NAMES, cm_path,
                        title=f"CATAMI Substrate-d2: {label} ({norm_label} Norm)"
                    )

            # German Bank 2010
            if run_all or "german_bank" in selected:
                print("  --> Evaluating German Bank 2010 (k-NN, Bicubic Antialiased)...")
                gb_res = evaluate_knn(model, gb_train, gb_test, gb_num_cls, val_transform_224, device, dtype, batch_size=args.batch_size, num_workers=args.num_workers)
                row["German Bank Acc (%)"] = f"{gb_res['accuracy']:.2f}%"
                row["German Bank F1-Macro (%)"] = f"{gb_res['macro_f1']:.2f}%"
                print(f"      German Bank Acc: {gb_res['accuracy']:.2f}% | Macro-F1: {gb_res['macro_f1']:.2f}%")

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

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append(row)

            # Incremental save so progress is never lost
            if args.output_csv:
                try:
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
