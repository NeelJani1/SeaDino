#!/usr/bin/env python3
"""Main CLI Orchestrator with High-Quality Bicubic Antialiased Transforms."""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

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
    setup_hf_cache,
)
from eval_suite.datasets.biota import prepare_biota_data
from eval_suite.datasets.brackish import prepare_brackish_data
from eval_suite.datasets.deepfish import prepare_deepfish_data
from eval_suite.datasets.fathomnet import prepare_fathomnet_data
from eval_suite.datasets.single_label import prepare_single_label_data
from eval_suite.marineinst_backbone import MARINEINST_FILENAME
from eval_suite.models import (
    BASELINE_MAP,
    CLIP_MEAN,
    CLIP_STD,
    load_baseline_backbone,
    load_dinov3_backbone,
)
from eval_suite.utils import plot_and_save_confusion_matrix, set_seed


DATASET_COMPLETION_METRICS = {
    "substrate": ["Substrate Lin Acc (%)", "Substrate kNN Acc (%)"],
    "german_bank": ["German Bank Lin Acc (%)", "German Bank kNN Acc (%)"],
    "coralscapes": ["Coralscapes mIoU (%)"],
    "coralmask": ["CoralMask mIoU (%)", "CoralMask Coral-IoU (%)"],
    "biota": ["Biota Lin mAP (%)", "Biota kNN mAP (%)"],
    "deepfish": ["DeepFish Lin Acc (%)", "DeepFish kNN Acc (%)"],
    "fathomnet": ["FathomNet Lin Acc (%)", "FathomNet kNN Acc (%)"],
    "brackish": ["Brackish Lin Acc (%)", "Brackish kNN Acc (%)"],
}


def get_file_sha256(path_str: Optional[str]) -> str:
    if not path_str:
        return "-"
    try:
        p = Path(path_str)
        if not p.is_file():
            return "-"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "-"


def is_valid_metric_value(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    if not s or s in ("-", "None", "nan", "NaN", "null", "inf", "-inf"):
        return False
    s_clean = s.rstrip("%").strip()
    try:
        v = float(s_clean)
        return not (math.isnan(v) or math.isinf(v))
    except (ValueError, TypeError):
        return False


def compute_dataset_fingerprint(train_records: Any, test_records: Any, extra_info: Optional[str] = None) -> str:
    """Computes a deterministic SHA-256 fingerprint of dataset samples, partitions, and labels."""
    def _serialize_record(item: Any) -> Optional[str]:
        if isinstance(item, tuple):
            parts = []
            for x in item:
                if hasattr(x, "item") and callable(x.item):
                    try:
                        x = x.item()
                    except Exception:
                        pass
                parts.append(str(x))
            return ":".join(parts)
        if isinstance(item, dict):
            parts = []
            for k in sorted(item.keys()):
                v = item[k]
                if hasattr(v, "item") and callable(v.item):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                if hasattr(v, "tolist") and callable(v.tolist):
                    try:
                        v = v.tolist()
                    except Exception:
                        pass
                if v is None:
                    val_str = "None"
                elif isinstance(v, (str, int, float, bool)):
                    val_str = str(v)
                elif isinstance(v, (bytes, bytearray)):
                    val_str = f"bytes:{hashlib.sha256(v).hexdigest()[:16]}"
                elif isinstance(v, (Path, os.PathLike)):
                    val_str = os.fspath(v)
                elif isinstance(v, (list, tuple)):
                    try:
                        val_str = json.dumps(v, sort_keys=True, default=str)
                    except Exception:
                        val_str = str(v)
                elif hasattr(v, "filename") and getattr(v, "filename", None):
                    val_str = str(getattr(v, "filename"))
                elif isinstance(v, dict):
                    if "path" in v and v["path"]:
                        val_str = str(v["path"])
                    elif "bytes" in v and v["bytes"]:
                        b = v["bytes"]
                        val_str = f"bytes:{hashlib.sha256(b).hexdigest()[:16]}"
                    else:
                        try:
                            val_str = json.dumps(v, sort_keys=True, default=str)
                        except Exception:
                            val_str = str(v)
                else:
                    return None  # Unverifiable representation
                parts.append(f"{k}={val_str}")
            return ":".join(parts)
        if isinstance(item, (str, int, float, bool)):
            return str(item)
        if isinstance(item, (bytes, bytearray)):
            return f"bytes:{hashlib.sha256(item).hexdigest()[:16]}"
        if isinstance(item, (Path, os.PathLike)):
            return os.fspath(item)
        return None

    def _hash_records(records) -> str:
        if records is None:
            return "none"
        if isinstance(records, (list, tuple)):
            lines = []
            for item in records:
                s = _serialize_record(item)
                if s is None:
                    return "unverifiable"
                lines.append(s)
            lines.sort()
            return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        elif hasattr(records, "_fingerprint") and getattr(records, "_fingerprint", None):
            return hashlib.sha256(f"hf_fp={records._fingerprint}:len={len(records)}".encode("utf-8")).hexdigest()[:16]
        return "unverifiable"

    tr_hash = _hash_records(train_records)
    te_hash = _hash_records(test_records)
    if tr_hash == "unverifiable" or te_hash == "unverifiable":
        return "unverifiable"
    combined = f"tr={tr_hash}:te={te_hash}"
    if extra_info:
        combined += f":{extra_info}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class BaselineSourceIdentity(NamedTuple):
    target_path: str
    weight_sha: str
    effective_model_id: str
    config_sha: str = "-"


def resolve_baseline_source_and_weights(target: Any, args) -> BaselineSourceIdentity:
    """Resolves baseline model source path, weight SHA256, effective model ID, and consumed config SHA256.

    Returns:
        BaselineSourceIdentity(target_path, weight_sha, effective_model_id, config_sha)
    """
    raw_target = str(target).strip()
    base_id = raw_target.lower()
    canonical_id = BASELINE_MAP.get(base_id, base_id)

    # SAM 3 / SAM 3.1: local model directory takes precedence over remote revision (eval_suite/sam3_backbone.py:212-218)
    if canonical_id in ("facebook/sam3", "facebook/sam3.1"):
        model_path = getattr(args, "sam3_model_path", None)
        if model_path:
            p = Path(model_path).expanduser().resolve()
            eff_id = f"{raw_target}:{model_path}"
            is_sam31 = (canonical_id == "facebook/sam3.1" or "3.1" in base_id or "sam31" in base_id)
            weight_name = "sam3.1_multiplex.pt" if is_sam31 else "model.safetensors"
            w_file = (p / weight_name) if p.is_dir() else p
            cfg_file = (p / "config.json") if p.is_dir() else (p.parent / "config.json")

            # Check that exact release weight file exists and matches the required release filename
            if not w_file.is_file() or w_file.name != weight_name:
                w_sha = "-"
            else:
                w_sha = get_file_sha256(str(w_file))

            # Check that consumed config.json exists
            if not cfg_file.is_file():
                cfg_sha = "-"
            else:
                cfg_sha = get_file_sha256(str(cfg_file))

            # Missing or unreadable weight or config must fail closed
            if w_sha == "-" or cfg_sha == "-":
                return BaselineSourceIdentity(str(p), "-", eff_id, "-")
            return BaselineSourceIdentity(str(p), w_sha, eff_id, cfg_sha)

        elif getattr(args, "sam3_revision", None):
            return BaselineSourceIdentity("-", "-", f"{raw_target}@{args.sam3_revision}", "-")
        return BaselineSourceIdentity("-", "-", raw_target, "-")

    # MarineInst
    elif canonical_id == "marineinst-vit-h-stage1":
        ckpt = getattr(args, "marineinst_checkpoint", None)
        if ckpt:
            p = Path(ckpt).expanduser().resolve()
            eff_id = f"{raw_target}:{ckpt}"
            w_sha = get_file_sha256(str(p)) if p.is_file() else "-"
            if w_sha == "-":
                return BaselineSourceIdentity(str(p), "-", eff_id, "-")
            return BaselineSourceIdentity(str(p), w_sha, eff_id, "-")
        else:
            default_p = (Path.home() / ".cache" / "seadino" / "checkpoints" / MARINEINST_FILENAME).resolve()
            eff_id = f"{raw_target}:{default_p}"
            w_sha = get_file_sha256(str(default_p)) if default_p.is_file() else "-"
            if w_sha == "-":
                return BaselineSourceIdentity(str(default_p), "-", eff_id, "-")
            return BaselineSourceIdentity(str(default_p), w_sha, eff_id, "-")

    # Foundation baselines (CLIP, BioCLIP, DINOv2)
    return BaselineSourceIdentity("-", "-", raw_target, "-")


def derive_effective_model_id(job_info, args) -> str:
    """Derives effective model/weights identity based on job type and CLI options."""
    if job_info is not None:
        job_type, target = job_info if isinstance(job_info, tuple) else ("ckpt", job_info)
        if job_type == "baseline":
            ident = resolve_baseline_source_and_weights(target, args)
            return ident.effective_model_id
        elif job_type == "ckpt":
            return str(getattr(args, "model_id", "facebook/dinov3-vit-base")).strip()
    return str(getattr(args, "model_id", "facebook/dinov3-vit-base")).strip()


def is_dataset_completed_in_row(row: dict, dataset: str, args=None, job_info=None) -> bool:
    metrics = DATASET_COMPLETION_METRICS.get(dataset, [])
    if not metrics:
        return False
    for col in metrics:
        if col not in row or not is_valid_metric_value(row[col]):
            return False

    if args is not None:
        if dataset == "coralmask":
            expected_cm_proto = getattr(args, "coralmask_protocol", "legacy") or "legacy"
            if expected_cm_proto in ("heldout-v1", "coralmask-heldout-v1"):
                row_cm_proto = (row.get("CoralMask Protocol") or "").strip()
                if row_cm_proto not in ("heldout-v1", "coralmask-heldout-v1"):
                    return False
                val_miou = row.get("CoralMask Val mIoU (%)")
                if not is_valid_metric_value(val_miou):
                    return False
                # CoralMask val ratio
                expected_ratio = getattr(args, "coralmask_val_ratio", 0.2)
                row_ratio = row.get("CoralMask Val Ratio")
                if not row_ratio or str(row_ratio).strip() in ("", "-"):
                    return False
                try:
                    if abs(float(row_ratio) - float(expected_ratio)) > 1e-5:
                        return False
                except (ValueError, TypeError):
                    return False
                # CoralMask sample cap
                expected_cap = getattr(args, "coralmask_max_train_samples", None)
                expected_cap_str = "-" if (expected_cap is None or str(expected_cap).strip() in ("", "-")) else str(expected_cap).strip()
                row_cap = row.get("CoralMask Max Train Samples")
                row_cap_str = "-" if (row_cap is None or str(row_cap).strip() in ("", "-")) else str(row_cap).strip()
                if row_cap_str != expected_cap_str:
                    return False
                # CoralMask manifest
                expected_manifest = getattr(args, "coralmask_test_manifest", None)
                expected_manifest_str = str(expected_manifest).strip() if (expected_manifest and str(expected_manifest).strip() not in ("", "-")) else "-"
                row_manifest = row.get("CoralMask Test Manifest")
                row_manifest_str = str(row_manifest).strip() if (row_manifest and str(row_manifest).strip() not in ("", "-")) else "-"

                if expected_manifest_str != "-":
                    if not os.path.isfile(expected_manifest_str):
                        return False
                    expected_manifest_name = Path(expected_manifest_str).name
                    row_manifest_name = Path(row_manifest_str).name if row_manifest_str != "-" else "-"
                    if row_manifest_name != expected_manifest_name:
                        return False
                    cur_m_sha = get_file_sha256(expected_manifest_str)
                    row_sha_str = str(row.get("CoralMask Manifest SHA256", "-")).strip()
                    if cur_m_sha == "-" or row_sha_str != cur_m_sha:
                        return False
                else:
                    if row_manifest_str != "-":
                        return False

                # CoralMask epochs
                expected_cm_ep = str(getattr(args, "coralmask_epochs", 10)).strip()
                row_cm_ep = row.get("CoralMask Epochs")
                if not row_cm_ep or str(row_cm_ep).strip() in ("", "-"):
                    return False
                if str(row_cm_ep).strip() != expected_cm_ep:
                    return False

                # CoralMask split fingerprint: must match current validated split
                row_fp = str(row.get("CoralMask Split Fingerprint", "-")).strip()
                if not row_fp or row_fp == "-":
                    return False
                cur_fp = getattr(args, "_current_cm_fingerprint", None)
                if cur_fp is None:
                    if hasattr(args, "coralmask_dir") and args.coralmask_dir and os.path.isdir(args.coralmask_dir):
                        try:
                            from eval_suite.datasets.coralmask import prepare_coralmask_splits
                            _, _, _, prov = prepare_coralmask_splits(
                                data_dir=args.coralmask_dir,
                                val_ratio=args.coralmask_val_ratio,
                                split_seed=args.seed,
                                test_manifest=args.coralmask_test_manifest,
                                max_train_samples=args.coralmask_max_train_samples,
                            )
                            cur_fp = prov.get("coralmask_split_fingerprint")
                            args._current_cm_fingerprint = cur_fp
                        except Exception:
                            return False
                    else:
                        return False
                if not cur_fp or row_fp != cur_fp:
                    return False
            else:
                row_cm_proto = (row.get("CoralMask Protocol") or "").strip()
                if row_cm_proto in ("heldout-v1", "coralmask-heldout-v1"):
                    return False

                expected_cm_ep = str(getattr(args, "coralmask_epochs", 10)).strip()
                row_cm_ep = row.get("CoralMask Epochs")
                if row_cm_ep and str(row_cm_ep).strip() not in ("", "-"):
                    if str(row_cm_ep).strip() != expected_cm_ep:
                        return False

        elif dataset == "deepfish":
            expected_df_ep = str(getattr(args, "deepfish_epochs", 10)).strip()
            row_df_ep = row.get("DeepFish Epochs")
            if not row_df_ep or str(row_df_ep).strip() in ("", "-"):
                return False
            if str(row_df_ep).strip() != expected_df_ep:
                return False

            # DeepFish split matching (test vs val): mandatory field, no default assumption
            expected_df_split = str(getattr(args, "deepfish_split", "test") or "test").strip()
            row_df_split = row.get("DeepFish Split")
            if not row_df_split or str(row_df_split).strip() in ("", "-"):
                return False
            if str(row_df_split).strip() != expected_df_split:
                return False

            # DeepFish sample caps
            expected_max_train = str(getattr(args, "deepfish_max_train_samples", None) or "-").strip()
            row_max_train = str(row.get("DeepFish Max Train Samples", "-") or "-").strip()
            if row_max_train != "-" or expected_max_train != "-":
                if row_max_train != expected_max_train:
                    return False

            expected_max_test = str(getattr(args, "deepfish_max_test_samples", None) or "-").strip()
            row_max_test = str(row.get("DeepFish Max Test Samples", "-") or "-").strip()
            if row_max_test != "-" or expected_max_test != "-":
                if row_max_test != expected_max_test:
                    return False

            # DeepFish source directory: mandatory field, no default assumption
            cur_df_dir = str(Path(getattr(args, "deepfish_dir", "data/benchmarks/deepfish")).resolve()).strip()
            row_df_dir = row.get("DeepFish Directory") or row.get("DeepFish Dir")
            if not row_df_dir or str(row_df_dir).strip() in ("", "-"):
                return False
            if str(Path(str(row_df_dir).strip()).resolve()) != cur_df_dir:
                return False

            # DeepFish dataset fingerprint: mandatory sample/label/partition identity
            row_df_fp = row.get("DeepFish Dataset Fingerprint") or row.get("DeepFish Fingerprint")
            if not row_df_fp or str(row_df_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            cur_df_fp = getattr(args, "_current_deepfish_fingerprint", None)
            if cur_df_fp is None:
                if hasattr(args, "deepfish_dir") and args.deepfish_dir and os.path.isdir(args.deepfish_dir):
                    try:
                        tr, te, _, _ = prepare_deepfish_data(
                            args.deepfish_dir,
                            split_test=getattr(args, "deepfish_split", "test") or "test",
                            max_train_samples=getattr(args, "deepfish_max_train_samples", None),
                            max_test_samples=getattr(args, "deepfish_max_test_samples", None),
                            seed=getattr(args, "seed", 42),
                        )
                        cur_df_fp = compute_dataset_fingerprint(tr, te)
                        args._current_deepfish_fingerprint = cur_df_fp
                    except Exception:
                        return False
                else:
                    return False
            if not cur_df_fp or str(cur_df_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            if str(row_df_fp).strip() != str(cur_df_fp).strip():
                return False

        elif dataset == "german_bank":
            expected_gb_ep = str(getattr(args, "german_bank_epochs", 10)).strip()
            row_gb_ep = row.get("German Bank Epochs")
            if not row_gb_ep or str(row_gb_ep).strip() in ("", "-"):
                return False
            if str(row_gb_ep).strip() != expected_gb_ep:
                return False

            row_gb_fp = row.get("German Bank Dataset Fingerprint") or row.get("German Bank Fingerprint")
            cur_gb_fp = getattr(args, "_current_german_bank_fingerprint", None)
            if cur_gb_fp is not None:
                if not row_gb_fp or str(row_gb_fp).strip() in ("", "-", "none", "unverifiable") or str(row_gb_fp).strip() != str(cur_gb_fp).strip():
                    return False

        elif dataset == "substrate":
            expected_sub_ep = str(getattr(args, "substrate_epochs", 10)).strip()
            row_sub_ep = row.get("Substrate Epochs")
            if not row_sub_ep or str(row_sub_ep).strip() in ("", "-"):
                return False
            if str(row_sub_ep).strip() != expected_sub_ep:
                return False

            row_sub_fp = row.get("Substrate Dataset Fingerprint") or row.get("Substrate Fingerprint")
            cur_sub_fp = getattr(args, "_current_substrate_fingerprint", None)
            if cur_sub_fp is not None:
                if not row_sub_fp or str(row_sub_fp).strip() in ("", "-", "none", "unverifiable") or str(row_sub_fp).strip() != str(cur_sub_fp).strip():
                    return False

        elif dataset == "fathomnet":
            expected_fn_ep = str(getattr(args, "fathomnet_epochs", 20)).strip()
            row_fn_ep = row.get("FathomNet Epochs")
            if not row_fn_ep or str(row_fn_ep).strip() in ("", "-"):
                return False
            if str(row_fn_ep).strip() != expected_fn_ep:
                return False

            expected_fn_train = str(getattr(args, "fathomnet_train_split", "train") or "train").strip()
            row_fn_train = row.get("FathomNet Train Split")
            if not row_fn_train or str(row_fn_train).strip() in ("", "-") or str(row_fn_train).strip() != expected_fn_train:
                return False

            expected_fn_test = str(getattr(args, "fathomnet_test_split", "validation") or "validation").strip()
            row_fn_test = row.get("FathomNet Test Split")
            if not row_fn_test or str(row_fn_test).strip() in ("", "-") or str(row_fn_test).strip() != expected_fn_test:
                return False

            expected_fn_cap = str(getattr(args, "fathomnet_max_train_samples", None) or "-").strip()
            row_fn_cap = str(row.get("FathomNet Max Train Samples", "-") or "-").strip()
            if row_fn_cap != expected_fn_cap:
                return False

            expected_crop = str(getattr(args, "fathomnet_crop_boxes", False)).strip()
            row_crop = row.get("FathomNet Crop Boxes")
            if not row_crop or str(row_crop).strip() in ("", "-") or str(row_crop).strip() != expected_crop:
                return False

            row_fn_fp = row.get("FathomNet Dataset Fingerprint") or row.get("FathomNet Fingerprint")
            if not row_fn_fp or str(row_fn_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            cur_fn_fp = getattr(args, "_current_fathomnet_fingerprint", None)
            if cur_fn_fp is None:
                if hasattr(args, "fathomnet_dir") and args.fathomnet_dir and (os.path.isdir(args.fathomnet_dir) or os.path.isfile(args.fathomnet_dir)):
                    try:
                        tr, te, _, _ = prepare_fathomnet_data(
                            repo_or_dir=args.fathomnet_dir,
                            split_train=getattr(args, "fathomnet_train_split", "train"),
                            split_test=getattr(args, "fathomnet_test_split", "validation"),
                            max_train_samples=getattr(args, "fathomnet_max_train_samples", None),
                            max_test_samples=getattr(args, "fathomnet_max_test_samples", None),
                            seed=getattr(args, "seed", 42),
                        )
                        cur_fn_fp = compute_dataset_fingerprint(
                            tr, te, extra_info=f"crops={getattr(args, 'fathomnet_crop_boxes', False)}"
                        )
                        args._current_fathomnet_fingerprint = cur_fn_fp
                    except Exception:
                        return False
                else:
                    return False
            if not cur_fn_fp or str(cur_fn_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            if str(row_fn_fp).strip() != str(cur_fn_fp).strip():
                return False

        elif dataset == "brackish":
            expected_br_ep = str(getattr(args, "brackish_epochs", 10)).strip()
            row_br_ep = row.get("Brackish Epochs")
            if not row_br_ep or str(row_br_ep).strip() in ("", "-"):
                return False
            if str(row_br_ep).strip() != expected_br_ep:
                return False

            expected_br_cap = str(getattr(args, "brackish_max_train_samples", None) or "-").strip()
            row_br_cap = str(row.get("Brackish Max Train Samples", "-") or "-").strip()
            if row_br_cap != expected_br_cap:
                return False

            expected_crop = str(getattr(args, "brackish_crop_boxes", False)).strip()
            row_crop = row.get("Brackish Crop Boxes")
            if not row_crop or str(row_crop).strip() in ("", "-") or str(row_crop).strip() != expected_crop:
                return False

            row_br_fp = row.get("Brackish Dataset Fingerprint") or row.get("Brackish Fingerprint")
            if not row_br_fp or str(row_br_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            cur_br_fp = getattr(args, "_current_brackish_fingerprint", None)
            if cur_br_fp is None:
                if hasattr(args, "brackish_dir") and args.brackish_dir and os.path.isdir(args.brackish_dir):
                    try:
                        tr, te, _, _ = prepare_brackish_data(
                            data_dir=args.brackish_dir,
                            max_train_samples=getattr(args, "brackish_max_train_samples", None),
                            max_test_samples=getattr(args, "brackish_max_test_samples", None),
                            seed=getattr(args, "seed", 42),
                        )
                        cur_br_fp = compute_dataset_fingerprint(
                            tr, te, extra_info=f"crops={getattr(args, 'brackish_crop_boxes', False)}"
                        )
                        args._current_brackish_fingerprint = cur_br_fp
                    except Exception:
                        return False
                else:
                    return False
            if not cur_br_fp or str(cur_br_fp).strip() in ("", "-", "none", "unverifiable"):
                return False
            if str(row_br_fp).strip() != str(cur_br_fp).strip():
                return False

        elif dataset == "coralscapes":
            expected_cs_ep = str(getattr(args, "coralscapes_epochs", 10)).strip()
            row_cs_ep = row.get("Coralscapes Epochs")
            if not row_cs_ep or str(row_cs_ep).strip() in ("", "-"):
                return False
            if str(row_cs_ep).strip() != expected_cs_ep:
                return False

            row_cs_fp = row.get("Coralscapes Dataset Fingerprint") or row.get("Coralscapes Fingerprint")
            cur_cs_fp = getattr(args, "_current_coralscapes_fingerprint", None)
            if cur_cs_fp is not None:
                if not row_cs_fp or str(row_cs_fp).strip() in ("", "-", "none", "unverifiable") or str(row_cs_fp).strip() != str(cur_cs_fp).strip():
                    return False

        elif dataset == "biota":
            row_bio_fp = row.get("Biota Dataset Fingerprint") or row.get("Biota Fingerprint")
            cur_bio_fp = getattr(args, "_current_biota_fingerprint", None)
            if cur_bio_fp is not None:
                if not row_bio_fp or str(row_bio_fp).strip() in ("", "-", "none", "unverifiable") or str(row_bio_fp).strip() != str(cur_bio_fp).strip():
                    return False

    return True


def row_matches_job_config(row: dict, label: str, args, job_info=None) -> bool:
    if row.get("Model / Label", "").strip() != label.strip():
        return False

    job_type = "ckpt"
    target = None
    if job_info is not None:
        job_type, target = job_info if isinstance(job_info, tuple) else ("ckpt", job_info)

    # Checkpoint identity (full path & content bytes)
    if job_type == "ckpt" and target is not None:
        target_abs = os.path.abspath(target)
        row_ckpt = row.get("Checkpoint Path")
        if not row_ckpt or str(row_ckpt).strip() in ("", "-"):
            return False
        if os.path.abspath(str(row_ckpt).strip()) != target_abs:
            return False

        if not os.path.isfile(target_abs):
            return False

        target_sha = get_file_sha256(target_abs)
        if not target_sha or target_sha == "-":
            return False

        row_sha = row.get("Checkpoint SHA256")
        if not row_sha or str(row_sha).strip() in ("", "-"):
            return False
        if str(row_sha).strip() != target_sha:
            return False
    elif job_type == "ckpt" and target is None:
        # off-the-shelf
        pass
    elif job_type == "baseline":
        ident = resolve_baseline_source_and_weights(target, args)
        base_target_path = ident.target_path
        base_sha = ident.weight_sha
        base_cfg_sha = ident.config_sha
        if base_target_path != "-":
            row_ckpt = row.get("Checkpoint Path")
            if not row_ckpt or str(row_ckpt).strip() in ("", "-"):
                return False
            if os.path.abspath(str(row_ckpt).strip()) != os.path.abspath(base_target_path):
                return False
            if not os.path.exists(base_target_path):
                return False
            if not base_sha or base_sha == "-":
                return False
            row_sha = row.get("Checkpoint SHA256")
            if not row_sha or str(row_sha).strip() in ("", "-"):
                return False
            if str(row_sha).strip() != base_sha:
                return False
            if base_cfg_sha != "-":
                row_cfg_sha = row.get("Config SHA256")
                if not row_cfg_sha or str(row_cfg_sha).strip() in ("", "-"):
                    return False
                if str(row_cfg_sha).strip() != base_cfg_sha:
                    return False
            else:
                row_cfg_sha = row.get("Config SHA256")
                if row_cfg_sha and str(row_cfg_sha).strip() not in ("", "-"):
                    return False
        else:
            row_sha = row.get("Checkpoint SHA256")
            if row_sha and str(row_sha).strip() not in ("", "-"):
                return False
            row_cfg_sha = row.get("Config SHA256")
            if row_cfg_sha and str(row_cfg_sha).strip() not in ("", "-"):
                return False
    else:
        # Fallback when job_info not passed: check Checkpoint Path/SHA256 if present
        row_ckpt = row.get("Checkpoint Path")
        if row_ckpt and str(row_ckpt).strip() not in ("", "-"):
            if not os.path.exists(str(row_ckpt).strip()):
                return False
            target_sha = get_file_sha256(str(row_ckpt).strip())
            row_sha = row.get("Checkpoint SHA256")
            if not row_sha or str(row_sha).strip() in ("", "-") or target_sha == "-" or str(row_sha).strip() != target_sha:
                return False

    # Model ID matching (derive effective model identity by job type)
    expected_model_id = derive_effective_model_id(job_info, args)
    row_model_id = row.get("Model ID") or row.get("Backbone Model ID")
    if not row_model_id or str(row_model_id).strip() in ("", "-"):
        return False
    if str(row_model_id).strip() != expected_model_id:
        return False

    # Resolution matching: reject missing or mismatched
    row_res = row.get("Resolution") or row.get("Image Size")
    if not row_res or str(row_res).strip() in ("", "-"):
        return False
    expected_size = str(getattr(args, "image_size", 224)).strip()
    if str(row_res).strip() != expected_size:
        return False

    # Epoch matching: reject missing
    if "Epoch" not in row or str(row.get("Epoch")).strip() == "":
        return False

    # Norm policy matching: reject missing or mismatched
    row_norm_policy = row.get("Norm Policy")
    if not row_norm_policy or str(row_norm_policy).strip() in ("", "-"):
        return False
    expected_norm = str(getattr(args, "norm", "auto")).strip()
    if str(row_norm_policy).strip() != expected_norm:
        return False

    row_norm = row.get("Norm")
    if not row_norm or str(row_norm).strip() in ("", "-"):
        return False
    if expected_norm == "benthic" and str(row_norm).strip() != "BenthicNet":
        return False
    if expected_norm == "imagenet" and str(row_norm).strip() not in ("ImageNet", "CLIP"):
        return False

    # Seed matching: reject missing or mismatched
    row_seed = row.get("Seed")
    if not row_seed or str(row_seed).strip() in ("", "-"):
        return False
    if str(row_seed).strip() != str(getattr(args, "seed", 42)).strip():
        return False

    # Split protocol matching: reject missing or mismatched
    row_split = row.get("Split Protocol")
    if not row_split or str(row_split).strip() in ("", "-"):
        return False
    expected_split_proto = getattr(args, "split_protocol", "legacy") or "legacy"
    if str(row_split).strip() != expected_split_proto:
        return False

    # Eval split matching: reject missing or mismatched
    expected_eval_split = getattr(args, "eval_split", "test") or "test"
    row_eval_split = (row.get("Eval Split") or "").strip()
    if not row_eval_split or row_eval_split in ("", "-"):
        return False
    if row_eval_split != expected_eval_split:
        return False

    # Batch size matching: mandatory field, no default assumption
    expected_bs = str(getattr(args, "batch_size", 256)).strip()
    row_bs = row.get("Batch Size")
    if not row_bs or str(row_bs).strip() in ("", "-"):
        return False
    if str(row_bs).strip() != expected_bs:
        return False

    # Coralmask protocol matching
    expected_cm_proto = getattr(args, "coralmask_protocol", "legacy") or "legacy"
    if expected_cm_proto in ("heldout-v1", "coralmask-heldout-v1"):
        expected_cm_proto = "coralmask-heldout-v1"
    effective_cm_proto = (row.get("CoralMask Protocol") or "").strip() or "legacy"
    if effective_cm_proto in ("heldout-v1", "coralmask-heldout-v1"):
        effective_cm_proto = "coralmask-heldout-v1"
    if effective_cm_proto != expected_cm_proto:
        return False

    return True


def is_legacy_job_completed(existing_rows: list, label: str, active_datasets: list, args, job_info=None) -> bool:
    if not active_datasets:
        return False
    matching_rows = [r for r in existing_rows if row_matches_job_config(r, label, args, job_info=job_info)]
    if not matching_rows:
        return False
    for ds in active_datasets:
        if not any(is_dataset_completed_in_row(r, ds, args, job_info=job_info) for r in matching_rows):
            return False
    return True


def validate_csv_rectangularity(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """Validates that an existing CSV file has a non-empty header, unique columns, and uniform rectangular rows."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)
    if not raw_rows:
        return [], []
    existing_header = raw_rows[0]
    if not existing_header or all(not col.strip() for col in existing_header):
        raise ValueError(f"Corrupted CSV in {csv_path}: empty or missing header line.")
    seen_headers = set()
    for col_idx, col in enumerate(existing_header):
        norm_col = col.strip()
        if not norm_col:
            raise ValueError(f"Corrupted CSV in {csv_path}: header column at index {col_idx} is empty.")
        if norm_col in seen_headers:
            raise ValueError(f"Corrupted CSV in {csv_path}: duplicate header column '{norm_col}' at index {col_idx}.")
        seen_headers.add(norm_col)
    num_cols = len(existing_header)
    data_rows = raw_rows[1:]
    for line_idx, r in enumerate(data_rows, start=2):
        if not r or (len(r) == 1 and not r[0].strip()):
            continue
        if len(r) != num_cols:
            raise ValueError(
                f"Corrupted CSV in {csv_path}: line {line_idx} has {len(r)} columns, expected {num_cols}."
            )
    return existing_header, data_rows


def save_incremental_row_to_csv(output_csv: str, row: dict) -> None:
    csv_path = Path(output_csv)
    if csv_path.parent and not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_header, data_rows = validate_csv_rectangularity(csv_path)

    if not existing_header:
        fieldnames = list(row.keys())
        temp_file = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}_{time.time_ns()}")
        try:
            with open(temp_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerow(row)
            os.replace(temp_file, csv_path)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        return

    new_keys = [k for k in row.keys() if k not in existing_header]
    superset_header = list(existing_header) + new_keys

    padded_rows = []
    for r in data_rows:
        if not r or (len(r) == 1 and not r[0].strip()):
            continue
        r_dict = dict(zip(existing_header, r))
        for k in new_keys:
            r_dict[k] = ""
        padded_rows.append(r_dict)
    padded_rows.append(row)

    temp_file = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=superset_header, extrasaction="ignore", restval="")
            writer.writeheader()
            for pr in padded_rows:
                writer.writerow(pr)
        os.replace(temp_file, csv_path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


def ensure_spatial_split_symlinks():
    """Ensures data/ has symlinks to the leakage-free spatial split files if present in data/benchmarks, data/splits, or root."""
    try:
        from eval_suite.config import REPO_ROOT, DATA_DIR
        data_dir = DATA_DIR

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
                if not data_dir.exists():
                    data_dir.mkdir(parents=True, exist_ok=True)
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


def get_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--coralmask_protocol", choices=["legacy", "heldout-v1", "coralmask-heldout-v1"], default="legacy",
                   help="CoralMask evaluation protocol: 'legacy' (exploratory test epoch selection) or 'heldout-v1' (deterministic 80/20 train/val split, single test pass).")
    p.add_argument("--coralmask_val_ratio", type=float, default=0.2,
                   help="Validation split ratio for CoralMask held-out protocol (default: 0.2).")
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
    return p


def main():
    p = get_parser()
    args = p.parse_args()

    if args.prepare_splits_only and args.split_protocol != "explicit-v1":
        p.error("--prepare_splits_only requires --split_protocol explicit-v1")
    if args.eval_split == "test" and args.split_protocol != "explicit-v1":
        p.error("--eval_split test requires --split_protocol explicit-v1")
    if not args.prepare_splits_only and not args.checkpoints and not args.checkpoint_dirs and not args.include_off_the_shelf and not args.baselines:
        sys.exit("Error: Please pass --checkpoints, --checkpoint_dirs (or --checkpoint_dir), --include_off_the_shelf, or --baselines")

    setup_hf_cache()
    ensure_spatial_split_symlinks()

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
        # Upfront rectangularity validation if output_csv already exists
        if os.path.isfile(args.output_csv):
            validate_csv_rectangularity(Path(args.output_csv))

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
        args._current_substrate_fingerprint = compute_dataset_fingerprint(sub_train, sub_test)

    gb_train, gb_test, gb_num_cls = None, None, 0
    if run_all or "german_bank" in selected:
        print("  • Indexing German Bank 2010...")
        gb_train, gb_test, gb_num_cls = prepare_single_label_data(args.german_bank_csv, args.benthic_img_root, "substrate")
        print(f"    Loaded {len(gb_train):,} Train | {len(gb_test):,} Test ({gb_num_cls} classes)")
        args._current_german_bank_fingerprint = compute_dataset_fingerprint(gb_train, gb_test)

    bio_train, bio_test, bio_num_cls = None, None, 0
    if run_all or "biota" in selected:
        print("  • Indexing BenthicNet Biota (272 classes)...")
        bio_train, bio_test, bio_num_cls = prepare_biota_data(args.biota_csv, args.benthic_img_root)
        print(f"    Loaded {len(bio_train):,} Train | {len(bio_test):,} Test ({bio_num_cls} classes)")
        args._current_biota_fingerprint = compute_dataset_fingerprint(bio_train, bio_test)

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
        args._current_coralscapes_fingerprint = compute_dataset_fingerprint(cs_train, cs_val)

    indexing_failures = []

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
            args._current_deepfish_fingerprint = compute_dataset_fingerprint(df_train, df_test)
        except Exception as e:
            err_msg = f"DeepFish indexing failed: {e}"
            print(f"  ❌ {err_msg}")
            if "deepfish" in selected and not run_all:
                raise
            indexing_failures.append(err_msg)

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
            args._current_fathomnet_fingerprint = compute_dataset_fingerprint(
                fn_train, fn_test, extra_info=f"crops={args.fathomnet_crop_boxes}"
            )
        except Exception as e:
            err_msg = f"FathomNet indexing failed: {e}"
            print(f"  ❌ {err_msg}")
            if "fathomnet" in selected and not run_all:
                raise
            indexing_failures.append(err_msg)

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
            args._current_brackish_fingerprint = compute_dataset_fingerprint(
                br_train, br_test, extra_info=f"crops={args.brackish_crop_boxes}"
            )
        except Exception as e:
            err_msg = f"Brackish indexing failed: {e}"
            print(f"  ❌ {err_msg}")
            if "brackish" in selected and not run_all:
                raise
            indexing_failures.append(err_msg)

    if indexing_failures and run_all:
        formatted_errors = "\n".join(f"    - {fail}" for fail in indexing_failures)
        raise RuntimeError(
            f"Dataset preparation failed under '--datasets all' ({len(indexing_failures)} error(s)):\n"
            f"{formatted_errors}\n"
            f"Aborting evaluation run before model execution."
        )

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

    # Check already evaluated jobs if resume (composite identity check)
    existing_rows = []
    if args.resume and args.output_csv and os.path.isfile(args.output_csv):
        try:
            with open(args.output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_rows.append(r)
            print(f"  [Resume] Loaded {len(existing_rows)} record(s) from {args.output_csv}")
        except Exception as e:
            print(f"  [Resume Warning] Could not read {args.output_csv}: {e}")

    active_datasets = [ds for ds in selected if ds != "all"]
    if run_all:
        active_datasets = ["substrate", "german_bank", "coralscapes", "coralmask", "biota", "deepfish", "fathomnet", "brackish"]

    print(f"\nTotal evaluation jobs queued: {len(jobs)}")

    # 3. Evaluate Models
    results = []
    for job_idx, (label, job_info) in enumerate(jobs, 1):
        if args.resume and is_legacy_job_completed(existing_rows, label, active_datasets, args, job_info=job_info):
            print(f"\n[{job_idx}/{len(jobs)}] [Resume Skipping] Model [{label}] already has completed results for {active_datasets} in {args.output_csv}")
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

            if job_type == "baseline":
                ident = resolve_baseline_source_and_weights(target, args)
                job_target_path = ident.target_path
                ckpt_file_sha = ident.weight_sha
                effective_model_id = ident.effective_model_id
                config_file_sha = ident.config_sha
            else:
                job_target_path = target if (job_type == "ckpt" and target is not None) else "-"
                ckpt_file_sha = get_file_sha256(job_target_path) if job_target_path != "-" else "-"
                effective_model_id = derive_effective_model_id((job_type, target), args)
                config_file_sha = "-"

            manifest_p = args.coralmask_test_manifest
            manifest_file_sha = get_file_sha256(manifest_p) if (manifest_p and os.path.isfile(manifest_p)) else "-"

            row = {
                "Model / Label": label,
                "Checkpoint Path": str(Path(job_target_path).resolve()) if job_target_path != "-" else "-",
                "Checkpoint SHA256": ckpt_file_sha,
                "Config SHA256": config_file_sha if config_file_sha != "-" else "-",
                "Model ID": effective_model_id,
                "Resolution": str(args.image_size),
                "Epoch": epoch if epoch is not None else "-",
                "Norm": norm_label,
                "Norm Policy": args.norm,
                "Seed": str(args.seed),
                "Split Protocol": args.split_protocol,
                "Eval Split": getattr(args, "eval_split", "test"),
                "Batch Size": str(args.batch_size),
            }

            # Substrate Depth 2
            if run_all or "substrate" in selected:
                print(f"  --> Evaluating Substrate Depth 2 (k-NN + {args.substrate_epochs}ep Linear Probe)...")
                sub_res = evaluate_single_label(
                    model, sub_train, sub_test, sub_num_cls, val_transform_224, device, dtype,
                    batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.substrate_epochs, seed=args.seed
                )
                row["Substrate Epochs"] = str(args.substrate_epochs)
                row["Substrate Dataset Fingerprint"] = getattr(args, "_current_substrate_fingerprint", "-")
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
                row["German Bank Epochs"] = str(args.german_bank_epochs)
                row["German Bank Dataset Fingerprint"] = getattr(args, "_current_german_bank_fingerprint", "-")
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
                row["Biota Dataset Fingerprint"] = getattr(args, "_current_biota_fingerprint", "-")
                row["Biota Lin mAP (%)"] = f"{bio_res['biota_lin_mAP']:.2f}%"
                row["Biota Lin F1-Macro (%)"] = f"{bio_res['biota_lin_macro_f1']:.2f}%"
                row["Biota kNN mAP (%)"] = f"{bio_res['biota_knn_mAP']:.2f}%"

            # Coralscapes
            if run_all or "coralscapes" in selected:
                print("  --> Evaluating Coralscapes Semantic Segmentation (10 epochs)...")
                cs_res = evaluate_coralscapes(model, cs_train, cs_val, cs_num_cls, device, dtype, batch_size=16, num_workers=args.num_workers, seed=args.seed, mean=active_mean, std=active_std)
                row["Coralscapes Epochs"] = "10"
                row["Coralscapes Dataset Fingerprint"] = getattr(args, "_current_coralscapes_fingerprint", "-")
                row["Coralscapes mIoU (%)"] = f"{cs_res['coralscapes_mIoU']:.2f}%"
                row["Coralscapes Pixel Acc (%)"] = f"{cs_res['coralscapes_pixel_acc']:.2f}%"
                print(f"      Coralscapes mIoU: {cs_res['coralscapes_mIoU']:.2f}% | Pix Acc: {cs_res['coralscapes_pixel_acc']:.2f}%")

            # CoralMask
            if run_all or "coralmask" in selected:
                manifest_str = f" [Manifest: {Path(args.coralmask_test_manifest).name}]" if args.coralmask_test_manifest else ""
                proto_str = f" [Protocol: {args.coralmask_protocol}]"
                print(f"  --> Evaluating CoralMask Dense Binary Segmentation ({args.coralmask_epochs} epochs){manifest_str}{proto_str}...")
                cm_res = evaluate_coralmask(
                    model, args.coralmask_dir, device, dtype,
                    batch_size=16, num_workers=min(4, args.num_workers), seed=args.seed,
                    epochs=args.coralmask_epochs,
                    mean=active_mean, std=active_std,
                    test_manifest=args.coralmask_test_manifest,
                    max_train_samples=args.coralmask_max_train_samples,
                    protocol=args.coralmask_protocol,
                    val_ratio=args.coralmask_val_ratio,
                )
                row["CoralMask Epochs"] = str(args.coralmask_epochs)
                if args.coralmask_protocol in ("heldout-v1", "coralmask-heldout-v1"):
                    row["CoralMask Protocol"] = "coralmask-heldout-v1"
                    row["CoralMask Val Ratio"] = str(args.coralmask_val_ratio)
                    row["CoralMask Max Train Samples"] = str(args.coralmask_max_train_samples) if args.coralmask_max_train_samples is not None else "-"
                    row["CoralMask Test Manifest"] = str(args.coralmask_test_manifest) if args.coralmask_test_manifest is not None else "-"
                    row["CoralMask Manifest SHA256"] = manifest_file_sha
                    row["CoralMask Split Fingerprint"] = cm_res.get("coralmask_split_fingerprint", "-")
                    row["CoralMask Selection Rule"] = cm_res.get("coralmask_selection_rule", "best_val_miou")
                    row["CoralMask Val mIoU (%)"] = f"{cm_res['coralmask_val_mIoU']:.2f}%"
                    row["CoralMask Selected Epoch"] = str(cm_res.get("coralmask_selected_epoch", "-"))
                    row["CoralMask Coral-IoU (%)"] = f"{cm_res['coralmask_coral_iou']:.2f}%"
                    row["CoralMask mIoU (%)"] = f"{cm_res['coralmask_mIoU']:.2f}%"
                    print(f"      CoralMask Held-Out Protocol: Selected Ep {cm_res.get('coralmask_selected_epoch', '-')} (Val mIoU: {cm_res['coralmask_val_mIoU']:.2f}%) | Final Test mIoU: {cm_res['coralmask_mIoU']:.2f}% | Coral-IoU: {cm_res['coralmask_coral_iou']:.2f}%")
                else:
                    row["CoralMask Protocol"] = "legacy"
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
                row["DeepFish Epochs"] = str(args.deepfish_epochs)
                row["DeepFish Split"] = str(args.deepfish_split)
                row["DeepFish Max Train Samples"] = str(args.deepfish_max_train_samples) if args.deepfish_max_train_samples is not None else "-"
                row["DeepFish Directory"] = str(Path(args.deepfish_dir).resolve())
                row["DeepFish Dataset Fingerprint"] = getattr(args, "_current_deepfish_fingerprint", "-")
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
                row["FathomNet Epochs"] = str(args.fathomnet_epochs)
                row["FathomNet Train Split"] = str(args.fathomnet_train_split)
                row["FathomNet Test Split"] = str(args.fathomnet_test_split)
                row["FathomNet Max Train Samples"] = str(args.fathomnet_max_train_samples) if args.fathomnet_max_train_samples is not None else "-"
                row["FathomNet Crop Boxes"] = str(args.fathomnet_crop_boxes)
                row["FathomNet Dataset Fingerprint"] = getattr(args, "_current_fathomnet_fingerprint", "-")
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
                row["Brackish Epochs"] = str(args.brackish_epochs)
                row["Brackish Max Train Samples"] = str(args.brackish_max_train_samples) if args.brackish_max_train_samples is not None else "-"
                row["Brackish Crop Boxes"] = str(args.brackish_crop_boxes)
                row["Brackish Dataset Fingerprint"] = getattr(args, "_current_brackish_fingerprint", "-")
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

        except Exception as e:
            print(f"  ❌ Error evaluating {ckpt_desc}: {e}")
            import traceback
            traceback.print_exc()
            continue

        results.append(row)

        # Incremental save so progress is never lost (rectangular atomic writer)
        if args.output_csv:
            try:
                save_incremental_row_to_csv(args.output_csv, row)
                existing_rows.append(row)
            except Exception as ex:
                print(f"  ❌ [FATAL CSV persistence failure] {ex}")
                raise

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
