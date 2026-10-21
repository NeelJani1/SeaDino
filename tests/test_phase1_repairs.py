"""Comprehensive CPU-Only Test Suite for Phase 1 Repairs (Tasks T01 - T05b).

Validates:
- T01: Legacy Resume Composite Identity, Checkpoint Fingerprinting & Strict Metric Validation (run_eval.py)
- T02: Rectangular Atomic CSV Writer & Duplicate Header Detection (run_eval.py)
- T03: Fail-Fast Training Resume Path Normalization & Readability (ssl_train.py)
- T04: Strict Error Propagation Under --datasets all (run_eval.py)
- T05a: CoralMask Exploratory Selection Documentation & Console Disclaimers
- T05b: Held-Out CoralMask Partition Disjointness, Provenance & Single Test Scoring
"""

import argparse
import contextlib
from contextlib import ExitStack
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_eval
from run_eval import (
    DATASET_COMPLETION_METRICS,
    BaselineSourceIdentity,
    compute_dataset_fingerprint,
    derive_effective_model_id,
    get_file_sha256,
    is_dataset_completed_in_row,
    is_legacy_job_completed,
    is_valid_metric_value,
    resolve_baseline_source_and_weights,
    row_matches_job_config,
    save_incremental_row_to_csv,
    validate_csv_rectangularity,
)
from eval_suite.datasets.coralmask import CoralMaskDataset, prepare_coralmask_splits
from eval_suite.benchmarks.coralmask_probe import evaluate_coralmask, _eval_loader


class TestTaskT01LegacyResumeIdentity(unittest.TestCase):
    """T01 / RR01: Verify composite identity, checkpoint hashing, and setting-aware completion in legacy resume."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.temp_dir, "benthic-ssl-best.ckpt")
        with open(self.ckpt_path, "wb") as f:
            f.write(b"checkpoint_bytes_v1")
        self.ckpt_sha = get_file_sha256(self.ckpt_path)

        self.label = "stage1_marine_with_physics/benthic-ssl-best.ckpt"
        self.job_info = ("ckpt", self.ckpt_path)

        self.df_fingerprint = "df_test_fingerprint_abc123"
        self.sub_fingerprint = "sub_test_fingerprint_def456"
        self.gb_fingerprint = "gb_test_fingerprint_ghi789"

        self.args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            coralmask_protocol="legacy",
            image_size=224,
            model_id="facebook/dinov3-vit-base",
            eval_split="test",
            batch_size=256,
            deepfish_split="test",
            deepfish_max_train_samples=None,
            deepfish_max_test_samples=None,
            deepfish_dir="data/benchmarks/deepfish",
            substrate_epochs=10,
            german_bank_epochs=10,
            deepfish_epochs=10,
            coralmask_epochs=10,
            coralmask_val_ratio=0.2,
            coralmask_max_train_samples=None,
            coralmask_test_manifest=None,
            _current_deepfish_fingerprint=self.df_fingerprint,
            _current_substrate_fingerprint=self.sub_fingerprint,
            _current_german_bank_fingerprint=self.gb_fingerprint,
        )

        self.completed_row = {
            "Model / Label": self.label,
            "Checkpoint Path": str(Path(self.ckpt_path).resolve()),
            "Checkpoint SHA256": self.ckpt_sha,
            "Model ID": "facebook/dinov3-vit-base",
            "Resolution": "224",
            "Epoch": "14",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "test",
            "Batch Size": "256",
            "Substrate Epochs": "10",
            "Substrate Dataset Fingerprint": self.sub_fingerprint,
            "Substrate Lin Acc (%)": "54.41%",
            "Substrate kNN Acc (%)": "53.55%",
            "German Bank Epochs": "10",
            "German Bank Dataset Fingerprint": self.gb_fingerprint,
            "German Bank Lin Acc (%)": "69.86%",
            "German Bank kNN Acc (%)": "68.75%",
            "DeepFish Epochs": "10",
            "DeepFish Split": "test",
            "DeepFish Max Train Samples": "-",
            "DeepFish Directory": str(Path("data/benchmarks/deepfish").resolve()),
            "DeepFish Dataset Fingerprint": self.df_fingerprint,
            "DeepFish Lin Acc (%)": "78.20%",
            "DeepFish Lin Fish-F1 (%)": "76.50%",
            "DeepFish kNN Acc (%)": "77.10%",
            "DeepFish kNN Fish-F1 (%)": "75.30%",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_identical_completed_job_skips(self):
        """Exact identical configuration and checkpoint must skip."""
        self.assertTrue(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

    def test_completed_substrate_does_not_skip_german_bank(self):
        row = dict(self.completed_row)
        del row["German Bank Lin Acc (%)"]
        del row["German Bank kNN Acc (%)"]

        self.assertFalse(is_legacy_job_completed([row], self.label, ["german_bank"], self.args, job_info=self.job_info))
        self.assertTrue(is_legacy_job_completed([row], self.label, ["substrate"], self.args, job_info=self.job_info))
        self.assertFalse(is_legacy_job_completed([row], self.label, ["substrate", "german_bank"], self.args, job_info=self.job_info))

    def test_nan_metric_prevents_skipping(self):
        """Values like nan% or nan must never be treated as completed."""
        row = dict(self.completed_row)
        row["DeepFish Lin Acc (%)"] = "nan%"
        self.assertFalse(is_legacy_job_completed([row], self.label, ["deepfish"], self.args, job_info=self.job_info))

        row["DeepFish Lin Acc (%)"] = "nan"
        self.assertFalse(is_legacy_job_completed([row], self.label, ["deepfish"], self.args, job_info=self.job_info))

        row["DeepFish Lin Acc (%)"] = ""
        self.assertFalse(is_legacy_job_completed([row], self.label, ["deepfish"], self.args, job_info=self.job_info))

    def test_missing_seed_or_norm_metadata_prevents_skipping(self):
        """Ambiguous older rows missing Seed, Norm, or Split Protocol must not skip."""
        row_no_seed = dict(self.completed_row)
        del row_no_seed["Seed"]
        self.assertFalse(is_legacy_job_completed([row_no_seed], self.label, ["deepfish"], self.args, job_info=self.job_info))

        row_no_norm = dict(self.completed_row)
        del row_no_norm["Norm"]
        self.assertFalse(is_legacy_job_completed([row_no_norm], self.label, ["deepfish"], self.args, job_info=self.job_info))

        row_no_split = dict(self.completed_row)
        del row_no_split["Split Protocol"]
        self.assertFalse(is_legacy_job_completed([row_no_split], self.label, ["deepfish"], self.args, job_info=self.job_info))

    def test_epoch_budget_change_prevents_skipping(self):
        """Changing DeepFish epoch budget (e.g. to 99) must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.deepfish_epochs = 99
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_eval_split_change_prevents_skipping(self):
        """Changing requested eval split (e.g. to 'val') must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.eval_split = "val"
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_resolution_change_prevents_skipping(self):
        """Changing resolution (e.g. to 448) must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.image_size = 448
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_model_id_change_prevents_skipping(self):
        """Changing backbone model ID must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.model_id = "facebook/dinov3-vit-large"
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_checkpoint_path_mismatch_prevents_skipping(self):
        """Different checkpoint file sharing same display label must cause a rerun."""
        other_path = os.path.join(self.temp_dir, "other_dir", "benthic-ssl-best.ckpt")
        os.makedirs(os.path.dirname(other_path), exist_ok=True)
        with open(other_path, "wb") as f:
            f.write(b"other_bytes")
        other_job_info = ("ckpt", other_path)

        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], self.args, job_info=other_job_info)
        )

    def test_checkpoint_bytes_in_place_change_prevents_skipping(self):
        """Overwriting checkpoint in place with different bytes must cause a rerun."""
        with open(self.ckpt_path, "wb") as f:
            f.write(b"modified_checkpoint_bytes_v2")

        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

    def test_coralmask_heldout_settings_mismatch_prevents_skipping(self):
        """Changed CoralMask held-out settings (ratio, manifest, epochs, cap) must cause a rerun."""
        heldout_row = {
            "Model / Label": self.label,
            "Checkpoint Path": str(Path(self.ckpt_path).resolve()),
            "Checkpoint SHA256": self.ckpt_sha,
            "Model ID": "facebook/dinov3-vit-base",
            "Resolution": "224",
            "Epoch": "14",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "test",
            "Batch Size": "256",
            "CoralMask Protocol": "coralmask-heldout-v1",
            "CoralMask Epochs": "10",
            "CoralMask Val Ratio": "0.2",
            "CoralMask Max Train Samples": "-",
            "CoralMask Test Manifest": "-",
            "CoralMask Manifest SHA256": "-",
            "CoralMask Split Fingerprint": "cm_split_fingerprint_test_v1",
            "CoralMask Val mIoU (%)": "71.00%",
            "CoralMask Selected Epoch": "2",
            "CoralMask Coral-IoU (%)": "72.87%",
            "CoralMask mIoU (%)": "73.03%",
        }

        heldout_args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            coralmask_protocol="heldout-v1",
            image_size=224,
            model_id="facebook/dinov3-vit-base",
            eval_split="test",
            batch_size=256,
            coralmask_epochs=10,
            coralmask_val_ratio=0.2,
            coralmask_max_train_samples=None,
            coralmask_test_manifest=None,
            _current_cm_fingerprint="cm_split_fingerprint_test_v1",
        )

        # Matching held-out row skips
        self.assertTrue(is_legacy_job_completed([heldout_row], self.label, ["coralmask"], heldout_args, job_info=self.job_info))

        # Changed validation ratio causes rerun
        ratio_args = argparse.Namespace(**vars(heldout_args))
        ratio_args.coralmask_val_ratio = 0.3
        self.assertFalse(is_legacy_job_completed([heldout_row], self.label, ["coralmask"], ratio_args, job_info=self.job_info))

        # Changed epoch budget causes rerun
        ep_args = argparse.Namespace(**vars(heldout_args))
        ep_args.coralmask_epochs = 20
        self.assertFalse(is_legacy_job_completed([heldout_row], self.label, ["coralmask"], ep_args, job_info=self.job_info))

        # Changed sample cap causes rerun
        cap_args = argparse.Namespace(**vars(heldout_args))
        cap_args.coralmask_max_train_samples = 50
        self.assertFalse(is_legacy_job_completed([heldout_row], self.label, ["coralmask"], cap_args, job_info=self.job_info))

    def test_batch_size_change_prevents_skipping(self):
        """Changing probe batch size (e.g. to 1) must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.batch_size = 1
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_deepfish_split_change_prevents_skipping(self):
        """Changing DeepFish eval split (val vs test) must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.deepfish_split = "val"
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_deepfish_sample_cap_change_prevents_skipping(self):
        """Changing DeepFish sample cap must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.deepfish_max_train_samples = 1
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_deepfish_source_directory_change_prevents_skipping(self):
        """Changing DeepFish data directory must cause a rerun."""
        args_modified = argparse.Namespace(**vars(self.args))
        args_modified.deepfish_dir = str(Path(self.temp_dir) / "other_deepfish")
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], args_modified, job_info=self.job_info)
        )

    def test_ambiguous_row_missing_required_fields_prevents_skipping(self):
        """Popping any required configuration/provenance column must prevent skipping."""
        for field in [
            "Checkpoint SHA256", "Checkpoint Path", "Model ID", "Resolution",
            "DeepFish Epochs", "Norm Policy", "DeepFish Split", "DeepFish Directory",
            "Batch Size", "DeepFish Dataset Fingerprint"
        ]:
            ambiguous = dict(self.completed_row)
            del ambiguous[field]
            self.assertFalse(
                is_legacy_job_completed([ambiguous], self.label, ["deepfish"], self.args, job_info=self.job_info),
                f"Missing field '{field}' should have prevented resume skip but did not."
            )

    def test_deleted_checkpoint_prevents_skipping(self):
        """If the checkpoint file no longer exists on disk, resume must not skip."""
        os.unlink(self.ckpt_path)
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

    def test_baseline_resume_skips_identical_and_reruns_on_changed_identity(self):
        """Baseline evaluations must resume-skip on exact repeat and rerun on model ID change."""
        baseline_row = {
            "Model / Label": "clip (baseline)",
            "Checkpoint Path": "-",
            "Checkpoint SHA256": "-",
            "Model ID": "clip",
            "Resolution": "224",
            "Epoch": "1",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "validation",
            "Batch Size": "256",
            "DeepFish Epochs": "10",
            "DeepFish Split": "test",
            "DeepFish Max Train Samples": "-",
            "DeepFish Directory": str(Path("data/benchmarks/deepfish").resolve()),
            "DeepFish Dataset Fingerprint": "df_test_fingerprint_abc123",
            "DeepFish Lin Acc (%)": "50.00%",
            "DeepFish kNN Acc (%)": "50.00%",
        }
        baseline_args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            image_size=224,
            model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
            eval_split="validation",
            batch_size=256,
            deepfish_epochs=10,
            deepfish_split="test",
            deepfish_max_train_samples=None,
            deepfish_max_test_samples=None,
            deepfish_dir="data/benchmarks/deepfish",
            _current_deepfish_fingerprint="df_test_fingerprint_abc123",
        )
        # Exact repeat skips
        self.assertTrue(
            is_legacy_job_completed([baseline_row], "clip (baseline)", ["deepfish"], baseline_args, job_info=("baseline", "clip"))
        )
        # Changed baseline ID reruns
        self.assertFalse(
            is_legacy_job_completed([baseline_row], "bioclip (baseline)", ["deepfish"], baseline_args, job_info=("baseline", "bioclip"))
        )
        # Independence from unused DINO model ID: changing args.model_id does not invalidate baseline repeat
        dino_args = argparse.Namespace(**vars(baseline_args))
        dino_args.model_id = "irrelevant-dinov3-model"
        self.assertTrue(
            is_legacy_job_completed([baseline_row], "clip (baseline)", ["deepfish"], dino_args, job_info=("baseline", "clip"))
        )

    def test_coralmask_missing_requested_manifest_prevents_skipping(self):
        """If a requested test manifest was unlinked or does not exist, resume must not skip."""
        manifest = Path(self.temp_dir) / "requested_test_manifest.txt"
        manifest.write_text("test_00.jpg\n", encoding="utf-8")
        manifest_sha = get_file_sha256(str(manifest))

        cm_row = {
            "Model / Label": self.label,
            "Checkpoint Path": str(Path(self.ckpt_path).resolve()),
            "Checkpoint SHA256": self.ckpt_sha,
            "Model ID": "facebook/dinov3-vit-base",
            "Resolution": "224",
            "Epoch": "14",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "test",
            "Batch Size": "256",
            "CoralMask Protocol": "coralmask-heldout-v1",
            "CoralMask Epochs": "10",
            "CoralMask Val Ratio": "0.2",
            "CoralMask Max Train Samples": "-",
            "CoralMask Test Manifest": str(manifest),
            "CoralMask Manifest SHA256": manifest_sha,
            "CoralMask Split Fingerprint": "abcd1234ef567890",
            "CoralMask Val mIoU (%)": "71.00%",
            "CoralMask Selected Epoch": "2",
            "CoralMask Coral-IoU (%)": "72.87%",
            "CoralMask mIoU (%)": "73.03%",
        }

        cm_args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            coralmask_protocol="heldout-v1",
            image_size=224,
            model_id="facebook/dinov3-vit-base",
            eval_split="test",
            batch_size=256,
            coralmask_epochs=10,
            coralmask_val_ratio=0.2,
            coralmask_max_train_samples=None,
            coralmask_test_manifest=str(manifest),
            _current_cm_fingerprint="abcd1234ef567890",
        )

        # Before unlinking manifest, row matches and skips
        self.assertTrue(is_legacy_job_completed([cm_row], self.label, ["coralmask"], cm_args, job_info=self.job_info))

        # Once manifest is unlinked, resume must not skip
        manifest.unlink()
        self.assertFalse(is_legacy_job_completed([cm_row], self.label, ["coralmask"], cm_args, job_info=self.job_info))

    def test_deepfish_dataset_mutation_prevents_skipping(self):
        """Changing DeepFish dataset records or labels must cause a rerun."""
        # 1. Changed fingerprint in current runtime
        mutated_args = argparse.Namespace(**vars(self.args))
        mutated_args._current_deepfish_fingerprint = "mutated_records_fingerprint"
        self.assertFalse(
            is_legacy_job_completed([self.completed_row], self.label, ["deepfish"], mutated_args, job_info=self.job_info)
        )

        # 2. Historical row lacking dataset fingerprint must not skip
        unfingerprinted_row = dict(self.completed_row)
        del unfingerprinted_row["DeepFish Dataset Fingerprint"]
        self.assertFalse(
            is_legacy_job_completed([unfingerprinted_row], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

        # 3. compute_dataset_fingerprint sensitivity to member and label mutations
        train_records_orig = [("a.jpg", 0), ("b.jpg", 1)]
        test_records_orig = [("c.jpg", 1)]
        fp_orig = compute_dataset_fingerprint(train_records_orig, test_records_orig)

        # Label mutation
        test_records_label_mutated = [("c.jpg", 0)]
        fp_label_mutated = compute_dataset_fingerprint(train_records_orig, test_records_label_mutated)
        self.assertNotEqual(fp_orig, fp_label_mutated)

        # Member addition
        train_records_added = [("a.jpg", 0), ("b.jpg", 1), ("d.jpg", 0)]
        fp_added = compute_dataset_fingerprint(train_records_added, test_records_orig)
        self.assertNotEqual(fp_orig, fp_added)

    def test_missing_deepfish_split_prevents_skipping(self):
        """Missing or empty DeepFish Split in row must prevent skipping."""
        row_no_split = dict(self.completed_row)
        del row_no_split["DeepFish Split"]
        self.assertFalse(
            is_legacy_job_completed([row_no_split], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

        row_empty_split = dict(self.completed_row)
        row_empty_split["DeepFish Split"] = ""
        self.assertFalse(
            is_legacy_job_completed([row_empty_split], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

    def test_missing_batch_size_prevents_skipping(self):
        """Missing or empty Batch Size in row must prevent skipping (no default assumption)."""
        row_no_bs = dict(self.completed_row)
        del row_no_bs["Batch Size"]
        self.assertFalse(
            is_legacy_job_completed([row_no_bs], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

        row_empty_bs = dict(self.completed_row)
        row_empty_bs["Batch Size"] = "-"
        self.assertFalse(
            is_legacy_job_completed([row_empty_bs], self.label, ["deepfish"], self.args, job_info=self.job_info)
        )

    def test_marineinst_local_weights_mutation_prevents_skipping(self):
        """MarineInst local weights replacement must trigger a rerun."""
        marine_path = os.path.join(self.temp_dir, "marine.pth")
        with open(marine_path, "wb") as f:
            f.write(b"marine_weights_v1")
        marine_sha = get_file_sha256(marine_path)

        marine_args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            image_size=224,
            model_id="facebook/dinov3-vit-base",
            eval_split="test",
            batch_size=256,
            deepfish_epochs=10,
            deepfish_split="test",
            deepfish_max_train_samples=None,
            deepfish_max_test_samples=None,
            deepfish_dir="data/benchmarks/deepfish",
            marineinst_checkpoint=marine_path,
            _current_deepfish_fingerprint=self.df_fingerprint,
        )

        marine_row = {
            "Model / Label": "marineinst (baseline)",
            "Checkpoint Path": str(Path(marine_path).resolve()),
            "Checkpoint SHA256": marine_sha,
            "Model ID": f"marineinst:{marine_path}",
            "Resolution": "224",
            "Epoch": "1",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "test",
            "Batch Size": "256",
            "DeepFish Epochs": "10",
            "DeepFish Split": "test",
            "DeepFish Max Train Samples": "-",
            "DeepFish Directory": str(Path("data/benchmarks/deepfish").resolve()),
            "DeepFish Dataset Fingerprint": self.df_fingerprint,
            "DeepFish Lin Acc (%)": "60.00%",
            "DeepFish kNN Acc (%)": "60.00%",
        }

        # Exact repeat skips
        self.assertTrue(
            is_legacy_job_completed([marine_row], "marineinst (baseline)", ["deepfish"], marine_args, job_info=("baseline", "marineinst"))
        )

        # Mutate MarineInst local weights file in place
        with open(marine_path, "wb") as f:
            f.write(b"marine_weights_v2_mutated")

        # Now resume must not skip because Checkpoint SHA256 no longer matches new file bytes
        self.assertFalse(
            is_legacy_job_completed([marine_row], "marineinst (baseline)", ["deepfish"], marine_args, job_info=("baseline", "marineinst"))
        )

    def test_sam_local_weights_mutation_and_source_precedence(self):
        """SAM 3.1 local weights replacement and local directory precedence."""
        sam_dir = Path(self.temp_dir) / "sam_local"
        sam_dir.mkdir(parents=True, exist_ok=True)
        (sam_dir / "config.json").write_text("{}", encoding="utf-8")
        sam_weights = sam_dir / "sam3.1_multiplex.pt"
        sam_weights.write_bytes(b"sam_weights_original")
        sam_sha = get_file_sha256(str(sam_weights))

        sam_args = argparse.Namespace(
            seed=42,
            norm="auto",
            split_protocol="legacy",
            image_size=224,
            model_id="facebook/dinov3-vit-base",
            eval_split="test",
            batch_size=256,
            deepfish_epochs=10,
            deepfish_split="test",
            deepfish_max_train_samples=None,
            deepfish_max_test_samples=None,
            deepfish_dir="data/benchmarks/deepfish",
            sam3_model_path=str(sam_dir),
            sam3_revision="revision-A",
            _current_deepfish_fingerprint=self.df_fingerprint,
        )

        sam_cfg_sha = get_file_sha256(str(sam_dir / "config.json"))
        sam_row = {
            "Model / Label": "sam3.1 (baseline)",
            "Checkpoint Path": str(sam_dir.resolve()),
            "Checkpoint SHA256": sam_sha,
            "Config SHA256": sam_cfg_sha,
            "Model ID": f"sam3.1:{sam_dir}",
            "Resolution": "224",
            "Epoch": "1",
            "Norm": "ImageNet",
            "Norm Policy": "auto",
            "Seed": "42",
            "Split Protocol": "legacy",
            "Eval Split": "test",
            "Batch Size": "256",
            "DeepFish Epochs": "10",
            "DeepFish Split": "test",
            "DeepFish Max Train Samples": "-",
            "DeepFish Directory": str(Path("data/benchmarks/deepfish").resolve()),
            "DeepFish Dataset Fingerprint": self.df_fingerprint,
            "DeepFish Lin Acc (%)": "65.00%",
            "DeepFish kNN Acc (%)": "65.00%",
        }

        # Exact repeat skips
        self.assertTrue(
            is_legacy_job_completed([sam_row], "sam3.1 (baseline)", ["deepfish"], sam_args, job_info=("baseline", "sam3.1"))
        )

        # 1. Mutate weights file in place -> must not skip
        sam_weights.write_bytes(b"sam_weights_mutated")
        self.assertFalse(
            is_legacy_job_completed([sam_row], "sam3.1 (baseline)", ["deepfish"], sam_args, job_info=("baseline", "sam3.1"))
        )

        # Restore weights
        sam_weights.write_bytes(b"sam_weights_original")

        # 2. Mutate consumed config.json in place -> must not skip
        (sam_dir / "config.json").write_text('{"image_size": 896}', encoding="utf-8")
        self.assertFalse(
            is_legacy_job_completed([sam_row], "sam3.1 (baseline)", ["deepfish"], sam_args, job_info=("baseline", "sam3.1"))
        )

        # 3. Deleted config.json -> must not skip
        (sam_dir / "config.json").unlink()
        self.assertFalse(
            is_legacy_job_completed([sam_row], "sam3.1 (baseline)", ["deepfish"], sam_args, job_info=("baseline", "sam3.1"))
        )

        # Restore config.json
        (sam_dir / "config.json").write_text("{}", encoding="utf-8")

        # 4. Local directory precedence over revision:
        # If user points to a different local directory, even with same revision-A, it must not skip
        other_sam_dir = Path(self.temp_dir) / "sam_other"
        other_sam_dir.mkdir(parents=True, exist_ok=True)
        (other_sam_dir / "config.json").write_text("{}", encoding="utf-8")
        (other_sam_dir / "sam3.1_multiplex.pt").write_bytes(b"sam_other_weights")

        other_sam_args = argparse.Namespace(**vars(sam_args))
        other_sam_args.sam3_model_path = str(other_sam_dir)
        self.assertFalse(
            is_legacy_job_completed([sam_row], "sam3.1 (baseline)", ["deepfish"], other_sam_args, job_info=("baseline", "sam3.1"))
        )


class TestTaskT02RectangularCSVPersistence(unittest.TestCase):
    """T02 / RR05: Verify rectangular atomic CSV writer and duplicate header rejection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_results.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_duplicate_headers_rejected_and_file_unchanged(self):
        """A CSV with duplicate column names must raise ValueError and stay byte-for-byte unchanged."""
        original_content = "Model / Label,Metric,Metric\nold,11%,99%\n".encode("utf-8")
        with open(self.csv_path, "wb") as f:
            f.write(original_content)

        with self.assertRaises(ValueError) as ctx:
            save_incremental_row_to_csv(self.csv_path, {"Model / Label": "new", "Metric": "50%"})
        self.assertIn("duplicate header column", str(ctx.exception))

        with open(self.csv_path, "rb") as f:
            content_after = f.read()
        self.assertEqual(content_after, original_content, "Corrupted file was rewritten instead of preserved!")

    def test_heterogeneous_sequential_appends_stay_rectangular(self):
        row1 = {
            "Model / Label": "ModelA",
            "Epoch": "1",
            "Norm": "ImageNet",
            "Substrate Lin Acc (%)": "54.00%",
            "Substrate kNN Acc (%)": "55.00%",
        }
        save_incremental_row_to_csv(self.csv_path, row1)

        row2 = {
            "Model / Label": "ModelA",
            "Epoch": "1",
            "Norm": "ImageNet",
            "German Bank Lin Acc (%)": "70.00%",
            "German Bank kNN Acc (%)": "72.00%",
            "German Bank Acc (%)": "72.00%",
        }
        save_incremental_row_to_csv(self.csv_path, row2)

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)

        expected_cols = [
            "Model / Label", "Epoch", "Norm",
            "Substrate Lin Acc (%)", "Substrate kNN Acc (%)",
            "German Bank Lin Acc (%)", "German Bank kNN Acc (%)", "German Bank Acc (%)"
        ]
        self.assertEqual(header, expected_cols)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Substrate Lin Acc (%)"], "54.00%")
        self.assertEqual(rows[0]["German Bank Lin Acc (%)"], "")
        self.assertEqual(rows[1]["Substrate Lin Acc (%)"], "")
        self.assertEqual(rows[1]["German Bank Lin Acc (%)"], "70.00%")

    def test_corrupted_csv_fails_visibly(self):
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("Col1,Col2,Col3\n")
            f.write("val1,val2\n")  # corrupted line: only 2 columns

        with self.assertRaises(ValueError) as ctx:
            save_incremental_row_to_csv(self.csv_path, {"Col1": "a", "Col2": "b", "Col3": "c"})
        self.assertIn("Corrupted CSV", str(ctx.exception))


class TestTaskT03FailFastResumeValidation(unittest.TestCase):
    """T03 / RR02: Fail-fast validation & normalization of training --resume path."""

    def test_resume_path_normalization_and_early_validation(self):
        ssl_train_path = REPO_ROOT / "ssl_train.py"
        if not ssl_train_path.is_file():
            self.skipTest("ssl_train.py exists only in SeaDino-Core")

        import ssl_train

        # 1. Nonexistent path raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            ssl_train.validate_and_normalize_resume("/nonexistent/checkpoint_epoch99.ckpt")

        # 2. Existing file is normalized to absolute path
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            f.write(b"dummy_weights")
            f_path = f.name
        try:
            resolved = ssl_train.validate_and_normalize_resume(f_path)
            self.assertEqual(resolved, os.path.abspath(f_path))

            # 3. Quoted tilde path normalization
            fake_tilde = os.path.join("~", os.path.basename(f_path))
            with patch("os.path.expanduser", return_value=f_path):
                norm_tilde = ssl_train.validate_and_normalize_resume(fake_tilde)
                self.assertEqual(norm_tilde, os.path.abspath(f_path))

            # 4. Calling main(args) with missing resume raises before setup
            missing_args = argparse.Namespace(resume="/nonexistent_ckpt_path.ckpt")
            with self.assertRaises(FileNotFoundError):
                ssl_train.main(missing_args)
        finally:
            if os.path.exists(f_path):
                os.unlink(f_path)


class TestTaskT04StrictErrorPropagation(unittest.TestCase):
    """T04: Verify missing dataset indexing fails visibly under --datasets all."""

    def test_indexing_failure_aborts_under_run_all(self):
        """Calls production run_eval dataset preparation loop and asserts RuntimeError abort."""
        import types
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as td:
            temp_path = Path(td)
            out_csv = temp_path / "failure_results.csv"
            meta = temp_path / "id2label.json"
            meta.write_text('{"0":"fake"}', encoding="utf-8")

            ds = types.ModuleType("datasets")
            ds.load_dataset = lambda *a, **k: {"train": [1], "validation": [2]}
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = lambda *a, **k: str(meta)

            mock_load = MagicMock(return_value=(MagicMock(), 1, 0.0, False))

            index_defaults = {
                "setup_hf_cache": MagicMock(),
                "ensure_spatial_split_symlinks": MagicMock(),
                "resolve_csv_path": lambda p, f: p,
                "set_seed": lambda s: None,
                "load_dinov3_backbone": mock_load,
                "load_baseline_backbone": mock_load,
                "prepare_single_label_data": lambda *a, **k: ([1], [2], 2),
                "prepare_biota_data": lambda *a, **k: ([1], [2], 2),
                "prepare_deepfish_data": lambda *a, **k: ([1], [2], 2, ["a", "b"]),
                "prepare_fathomnet_data": lambda *a, **k: ([1], [2], 2, ["a", "b"]),
                "prepare_brackish_data": lambda *a, **k: ([1], [2], 2, ["a", "b"]),
            }

            for ds_name in ["deepfish", "fathomnet", "brackish"]:
                mock_load.reset_mock()
                overrides = dict(index_defaults)
                overrides[f"prepare_{ds_name}_data"] = MagicMock(side_effect=FileNotFoundError(f"injected indexing error for {ds_name}"))

                test_argv = [
                    "run_eval.py",
                    "--include_off_the_shelf",
                    "--datasets", "all",
                    "--no-save_cm_plots",
                    "--output_csv", str(out_csv),
                ]

                stdout = io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", test_argv))
                    stack.enter_context(patch.dict(sys.modules, {"datasets": ds, "huggingface_hub": hub}))
                    stack.enter_context(contextlib.redirect_stdout(stdout))
                    for name, value in overrides.items():
                        stack.enter_context(patch.object(run_eval, name, value))

                    with self.assertRaises(RuntimeError) as ctx:
                        run_eval.main()

                    self.assertIn("Dataset preparation failed under '--datasets all'", str(ctx.exception))
                    ds_labels = {"deepfish": "DeepFish", "fathomnet": "FathomNet", "brackish": "Brackish"}
                    self.assertIn(f"{ds_labels[ds_name]} indexing failed", str(ctx.exception))
                    self.assertEqual(mock_load.call_count, 0, "Model should never be loaded on indexing failure")
                    self.assertFalse(out_csv.exists(), "CSV should never be created on indexing failure")
                    self.assertNotIn("All results saved to:", stdout.getvalue(), "Success message should never be printed on indexing failure")


class TestTaskT05aConsoleLoggingAndReadme(unittest.TestCase):
    """T05a: Disclose legacy CoralMask exploratory selection."""

    def test_probe_console_label_updated(self):
        probe_source = (REPO_ROOT / "eval_suite/benchmarks/coralmask_probe.py").read_text(encoding="utf-8")
        self.assertIn("Test mIoU (exploratory legacy selection)", probe_source)
        self.assertNotIn("Val mIoU: {test_miou:.2f}%", probe_source)

    def test_readme_contains_exact_qualification_text(self):
        readme_source = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        exact_text = (
            "Historical result; exact run provenance unverified. "
            "The current legacy evaluator selects probe epoch using test mIoU, "
            "so these figures are not established as untouched final-test estimates."
        )
        self.assertIn(exact_text, readme_source)


class TestTaskT05bHeldOutCoralMaskProtocol(unittest.TestCase):
    """T05b / RR03 & RR04 & RR07: Held-out CoralMask deterministic disjoint split, manifest validation & provenance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_img_dir = Path(self.temp_dir) / "train" / "images"
        self.train_json_dir = Path(self.temp_dir) / "train" / "jsons"
        self.test_img_dir = Path(self.temp_dir) / "test" / "images"
        self.test_json_dir = Path(self.temp_dir) / "test" / "jsons"
        for d in [self.train_img_dir, self.train_json_dir, self.test_img_dir, self.test_json_dir]:
            d.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        for i in range(10):
            img_path = self.train_img_dir / f"train_{i:02d}.jpg"
            json_path = self.train_json_dir / f"train_{i:02d}.json"
            Image.new("RGB", (64, 64), color=(i * 10, i * 5, i * 20)).save(img_path)
            json_path.write_text(json.dumps({"annotations": []}), encoding="utf-8")

        for i in range(3):
            img_path = self.test_img_dir / f"test_{i:02d}.jpg"
            json_path = self.test_json_dir / f"test_{i:02d}.json"
            Image.new("RGB", (64, 64), color=(200 + i * 10, 50, 50)).save(img_path)
            json_path.write_text(json.dumps({"annotations": []}), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_80_20_split_from_train_partition(self):
        train_s, val_s, test_s, prov = prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, split_seed=42)

        self.assertEqual(len(train_s), 8)
        self.assertEqual(len(val_s), 2)
        self.assertEqual(len(test_s), 3)

        train_paths = {p[0] for p in train_s}
        val_paths = {p[0] for p in val_s}
        test_paths = {p[0] for p in test_s}

        self.assertEqual(train_paths & val_paths, set())
        self.assertEqual(train_paths & test_paths, set())
        self.assertEqual(val_paths & test_paths, set())

        self.assertIn("coralmask_split_fingerprint", prov)
        self.assertEqual(prov["coralmask_selection_rule"], "best_val_miou")

    def test_identical_content_images_stay_in_same_partition(self):
        """Images with identical content hashes must never be split across train and validation."""
        from PIL import Image
        img_dup1 = self.train_img_dir / "train_00.jpg"
        img_dup2 = self.train_img_dir / "train_01.jpg"
        Image.new("RGB", (64, 64), color=(42, 42, 42)).save(img_dup1)
        Image.new("RGB", (64, 64), color=(42, 42, 42)).save(img_dup2)

        train_s, val_s, _, _ = prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, split_seed=42)
        train_paths = {p[0] for p in train_s}
        val_paths = {p[0] for p in val_s}

        # Either both are in train, or both are in val
        dup1_in_train = str(img_dup1.resolve()) in train_paths
        dup2_in_train = str(img_dup2.resolve()) in train_paths
        self.assertEqual(dup1_in_train, dup2_in_train, "Identical content images were split across train and val!")

    def test_duplicate_manifest_entries_fail_fast(self):
        """Duplicate entries in test manifest must raise ValueError."""
        manifest = Path(self.temp_dir) / "dup_manifest.txt"
        img_path = self.test_img_dir / "test_00.jpg"
        manifest.write_text(f"{img_path}\n{img_path}\n", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, test_manifest=str(manifest))
        self.assertIn("Duplicate entry found in CoralMask test manifest", str(ctx.exception))

    def test_missing_manifest_fails_fast(self):
        """Missing requested test manifest must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, test_manifest="/nonexistent_manifest.txt")

    def test_cross_partition_leakage_detected_and_rejected(self):
        """A test image that appears in the training partition must raise ValueError."""
        manifest = Path(self.temp_dir) / "leaking_manifest.txt"
        train_img = self.train_img_dir / "train_00.jpg"
        manifest.write_text(f"{train_img}\n", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, test_manifest=str(manifest))
        self.assertIn("Cross-partition leakage detected", str(ctx.exception))

    def test_invalid_parameters_raise(self):
        """Invalid val_ratio or zero epochs must raise ValueError."""
        with self.assertRaises(ValueError):
            prepare_coralmask_splits(self.temp_dir, val_ratio=-0.1)
        with self.assertRaises(ValueError):
            prepare_coralmask_splits(self.temp_dir, val_ratio=1.0)
        with self.assertRaises(ValueError):
            prepare_coralmask_splits(self.temp_dir, val_ratio=1.5)

    def test_heldout_evaluator_event_trace_and_single_test_call(self):
        """Asserts validation selection across epochs and exactly one test scoring call."""
        class MockConfig:
            hidden_size = 32

        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

            def forward(self, x):
                b = x.shape[0]
                return {"last_hidden_state": torch.zeros(b, 16, 32)}

        backbone = MockBackbone()

        call_log = []
        original_eval_loader = _eval_loader

        def spying_eval_loader(model, loader, out_size, device, dtype, desc=""):
            call_log.append(desc)
            return original_eval_loader(model, loader, out_size, device, dtype, desc=desc)

        with patch("eval_suite.benchmarks.coralmask_probe._eval_loader", side_effect=spying_eval_loader):
            res = evaluate_coralmask(
                backbone, self.temp_dir, device=torch.device("cpu"), dtype=torch.float32,
                batch_size=2, num_workers=0, epochs=2, width=64, height=64,
                protocol="coralmask-heldout-v1", val_ratio=0.2
            )

        self.assertEqual(res["coralmask_protocol"], "coralmask-heldout-v1")
        self.assertIn("coralmask_val_mIoU", res)
        self.assertIn("coralmask_selected_epoch", res)
        self.assertIn("coralmask_mIoU", res)
        self.assertIn("coralmask_split_fingerprint", res)
        self.assertEqual(res["coralmask_selection_rule"], "best_val_miou")

        # Verify call log: 2 validation evaluations during training, exactly 1 final test evaluation
        val_calls = [c for c in call_log if "[Val]" in c]
        test_calls = [c for c in call_log if "[Final Test Eval]" in c]

        self.assertEqual(len(val_calls), 2, f"Expected 2 val calls, got: {val_calls}")
        self.assertEqual(len(test_calls), 1, f"Expected exactly 1 final test call, got: {test_calls}")

    def test_connected_component_clustering_chain(self):
        """Chained equivalence (sample A & B share image content, B & C share group metadata)
        must place all three samples into the exact same partition via connected component clustering."""
        from PIL import Image
        img_a = self.train_img_dir / "chain_a.jpg"
        img_b = self.train_img_dir / "chain_b.jpg"
        img_c = self.train_img_dir / "chain_c.jpg"

        # A and B share exact image content
        shared_pixels = Image.new("RGB", (64, 64), color=(77, 88, 99))
        shared_pixels.save(img_a)
        shared_pixels.save(img_b)

        # C has distinct pixels
        Image.new("RGB", (64, 64), color=(11, 22, 33)).save(img_c)

        # B and C share group metadata
        (self.train_json_dir / "chain_a.json").write_text(json.dumps({"annotations": [], "group_id": "group_a"}), encoding="utf-8")
        (self.train_json_dir / "chain_b.json").write_text(json.dumps({"annotations": [], "group_id": "group_bc"}), encoding="utf-8")
        (self.train_json_dir / "chain_c.json").write_text(json.dumps({"annotations": [], "group_id": "group_bc"}), encoding="utf-8")

        train_s, val_s, test_s, prov = prepare_coralmask_splits(self.temp_dir, val_ratio=0.3, split_seed=42)
        train_paths = {p[0] for p in train_s}
        val_paths = {p[0] for p in val_s}

        a_in_train = str(img_a.resolve()) in train_paths
        b_in_train = str(img_b.resolve()) in train_paths
        c_in_train = str(img_c.resolve()) in train_paths

        # Either all 3 are in train, or all 3 are in val
        self.assertEqual(a_in_train, b_in_train, "Sample A and B were split across partitions!")
        self.assertEqual(b_in_train, c_in_train, "Sample B and C were split across partitions!")

    def test_conflicting_group_ids_and_duplicate_bytes(self):
        """RC02: Conflicting group IDs with duplicate image bytes must stay in the same partition and never cross train/val."""
        from PIL import Image
        img_dup1 = self.train_img_dir / "train_00.jpg"
        img_dup2 = self.train_img_dir / "train_01.jpg"
        # Overwrite both images with exact duplicate bytes
        Image.new("RGB", (64, 64), color=(42, 42, 42)).save(img_dup1)
        Image.new("RGB", (64, 64), color=(42, 42, 42)).save(img_dup2)

        # Assign conflicting group IDs in their annotations
        (self.train_json_dir / "train_00.json").write_text(json.dumps({"annotations": [], "group_id": "group_alpha"}), encoding="utf-8")
        (self.train_json_dir / "train_01.json").write_text(json.dumps({"annotations": [], "group_id": "group_beta"}), encoding="utf-8")

        train_s, val_s, test_s, prov = prepare_coralmask_splits(self.temp_dir, val_ratio=0.2, split_seed=42)
        train_paths = {p[0] for p in train_s}
        val_paths = {p[0] for p in val_s}

        dup1_in_train = str(img_dup1.resolve()) in train_paths
        dup2_in_train = str(img_dup2.resolve()) in train_paths
        self.assertEqual(dup1_in_train, dup2_in_train, "Images with duplicate content but conflicting groups were split across train and val!")

        # Verify content disjointness between train and val
        train_hashes = {hashlib.sha256(Path(p[0]).read_bytes()).hexdigest() for p in train_s}
        val_hashes = {hashlib.sha256(Path(p[0]).read_bytes()).hexdigest() for p in val_s}
        self.assertEqual(len(train_hashes & val_hashes), 0, "Train and validation partitions contain overlapping image content hashes!")

    def test_direct_train_zero_ratio_preserves_all_samples(self):
        """Direct train split with val_ratio=0.0 must return all candidate samples without allocating validation."""
        ds = CoralMaskDataset(self.temp_dir, split="train", val_ratio=0.0)
        img_files = [f for f in os.listdir(self.train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.assertEqual(len(ds), len(img_files))

        # split='val' with val_ratio=0.0 must raise ValueError
        with self.assertRaises(ValueError):
            CoralMaskDataset(self.temp_dir, split="val", val_ratio=0.0)


class TestTaskRV01FormatAwareDatasetIdentity(unittest.TestCase):
    """RV01 / P1: Format-aware dataset fingerprinting in run_eval.py.

    Verifies:
    1. Preserving actual image identity for dictionary records (e.g. FathomNet and Brackish).
    2. Swapping members, label reassignment, partition change, or source directory change produces distinct fingerprints.
    3. Rejection of unverifiable representations (missing image, length-alone hashing).
    4. Exact repeats skip; mutations and missing stored fingerprints never skip.
    5. Actual CLI/main test with synthetic metadata CSVs and inert models/probes reproducing rv01_dict_identity.py.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tmp = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dict_records_fingerprint_sensitivity(self):
        """Format-aware fingerprinting must preserve actual image source, label, bbox, and partition identity."""
        train_base = [
            {"image": "/path/train/a.jpg", "raw_label": "crab", "bbox": [0, 0, 10, 10]},
            {"image": "/path/train/b.jpg", "raw_label": "fish", "bbox": None},
        ]
        test_base = [
            {"image": "/path/test/c.jpg", "raw_label": "crab", "bbox": None},
            {"image": "/path/test/d.jpg", "raw_label": "fish", "bbox": [5, 5, 15, 15]},
        ]
        fp_base = compute_dataset_fingerprint(train_base, test_base, extra_info="crops=False")
        self.assertNotIn(fp_base, ("", "-", "none", "unverifiable"))

        # Exact repeat produces identical fingerprint
        fp_exact = compute_dataset_fingerprint(copy.deepcopy(train_base), copy.deepcopy(test_base), extra_info="crops=False")
        self.assertEqual(fp_base, fp_exact)

        # Replacing test member with another file of same label must change fingerprint
        test_replaced = copy.deepcopy(test_base)
        test_replaced[0]["image"] = "/path/test/c_replacement.jpg"
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_base, test_replaced, extra_info="crops=False"))

        # Replacing train member with another file of same label must change fingerprint
        train_replaced = copy.deepcopy(train_base)
        train_replaced[0]["image"] = "/path/train/a_replacement.jpg"
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_replaced, test_base, extra_info="crops=False"))

        # Swapping same-label members between train and test partitions must change fingerprint
        train_swapped = [copy.deepcopy(test_base[0]), copy.deepcopy(train_base[1])]
        test_swapped = [copy.deepcopy(train_base[0]), copy.deepcopy(test_base[1])]
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_swapped, test_swapped, extra_info="crops=False"))

        # Swapping labels between test images must change fingerprint
        test_labels_swapped = copy.deepcopy(test_base)
        test_labels_swapped[0]["raw_label"] = "fish"
        test_labels_swapped[1]["raw_label"] = "crab"
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_base, test_labels_swapped, extra_info="crops=False"))

        # Modifying bounding box must change fingerprint
        test_bbox_changed = copy.deepcopy(test_base)
        test_bbox_changed[1]["bbox"] = [5, 5, 20, 20]
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_base, test_bbox_changed, extra_info="crops=False"))

        # Modifying source directory path must change fingerprint
        train_other_dir = [
            {"image": "/other/train/a.jpg", "raw_label": "crab", "bbox": [0, 0, 10, 10]},
            {"image": "/other/train/b.jpg", "raw_label": "fish", "bbox": None},
        ]
        test_other_dir = [
            {"image": "/other/test/c.jpg", "raw_label": "crab", "bbox": None},
            {"image": "/other/test/d.jpg", "raw_label": "fish", "bbox": [5, 5, 15, 15]},
        ]
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_other_dir, test_other_dir, extra_info="crops=False"))

        # Extra info sensitivity (e.g. crop boxes enabled vs disabled)
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(train_base, test_base, extra_info="crops=True"))

        # Raw bytes image support in records
        train_bytes = [{"image": b"synthetic_image_bytes_train", "label": 0}]
        test_bytes = [{"image": b"synthetic_image_bytes_test", "label": 1}]
        fp_bytes = compute_dataset_fingerprint(train_bytes, test_bytes)
        self.assertNotIn(fp_bytes, ("", "-", "none", "unverifiable"))
        test_bytes_mutated = [{"image": b"synthetic_image_bytes_mutated", "label": 1}]
        self.assertNotEqual(fp_bytes, compute_dataset_fingerprint(train_bytes, test_bytes_mutated))

        # Tuple with raw bytes support
        fp_tuple = compute_dataset_fingerprint([("/a.jpg", 0)], [("/b.jpg", 1)])
        self.assertNotIn(fp_tuple, ("", "-", "none", "unverifiable"))

    def test_unverifiable_representations_fail_closed(self):
        """Reject unhashable object records and objects only exposing length without _fingerprint."""
        class LengthOnlyDataset:
            def __len__(self):
                return 42

        fp_len = compute_dataset_fingerprint(LengthOnlyDataset(), LengthOnlyDataset())
        self.assertEqual(fp_len, "unverifiable")

        # Unknown unhashable item representation in records
        unhashable_records = [{"image": object(), "label": 1}]
        self.assertEqual(compute_dataset_fingerprint(unhashable_records, unhashable_records), "unverifiable")

        # is_dataset_completed_in_row must reject 'unverifiable', '-', and empty fingerprints
        dummy_row = {
            "FathomNet Lin Acc (%)": "50.00%",
            "FathomNet kNN Acc (%)": "50.00%",
            "FathomNet Epochs": "20",
            "FathomNet Train Split": "train",
            "FathomNet Test Split": "validation",
            "FathomNet Max Train Samples": "-",
            "FathomNet Crop Boxes": "False",
            "FathomNet Dataset Fingerprint": "unverifiable",
        }
        dummy_args = argparse.Namespace(
            fathomnet_epochs=20,
            fathomnet_train_split="train",
            fathomnet_test_split="validation",
            fathomnet_max_train_samples=None,
            fathomnet_crop_boxes=False,
            _current_fathomnet_fingerprint="unverifiable",
        )
        self.assertFalse(is_dataset_completed_in_row(dummy_row, "fathomnet", dummy_args))

    def test_rv01_cli_main_recheck_controls(self):
        """Full CLI/main execution with synthetic metadata CSVs and inert probes for FathomNet & Brackish."""
        ckpt = self.tmp / "model.ckpt"
        ckpt.write_bytes(b"inert_ckpt_bytes")

        rows = [
            {"file_name": "a.jpg", "label": "crab", "split": "train"},
            {"file_name": "b.jpg", "label": "fish", "split": "train"},
            {"file_name": "c.jpg", "label": "crab", "split": "train"},
            {"file_name": "d.jpg", "label": "fish", "split": "train"},
            {"file_name": "e.jpg", "label": "crab", "split": "test"},
            {"file_name": "f.jpg", "label": "fish", "split": "test"},
        ]

        def write_meta(root_dir, data_rows):
            root_dir.mkdir(parents=True, exist_ok=True)
            with (root_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["file_name", "label", "split"])
                w.writeheader()
                w.writerows(data_rows)
            for r in data_rows:
                img_p = root_dir / r["file_name"]
                img_p.write_bytes(f"dummy-pixels:{root_dir}:{r['file_name']}".encode("utf-8"))

        def run_cli(dataset, argv):
            prefix = "fathomnet" if dataset == "fathomnet" else "brackish"
            metric_names = [f"{prefix}_{s}" for s in ["lin_acc", "lin_macro_f1", "knn_acc", "knn_macro_f1"]]
            metrics = {name: 50.0 for name in metric_names}
            loader = MagicMock(return_value=(types.SimpleNamespace(), 1, 0.0, False))
            probe = MagicMock(return_value=metrics)
            stdout, stderr = io.StringIO(), io.StringIO()
            overrides = {
                "setup_hf_cache": MagicMock(),
                "ensure_spatial_split_symlinks": MagicMock(),
                "resolve_csv_path": lambda p, f: p,
                "set_seed": lambda s: None,
                "load_dinov3_backbone": loader,
                "load_baseline_backbone": loader,
                f"evaluate_{dataset}": probe,
            }
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["run_eval.py"] + argv))
                stack.enter_context(patch.object(run_eval.torch.cuda, "is_available", return_value=False))
                for name, fn in overrides.items():
                    stack.enter_context(patch.object(run_eval, name, fn))
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    run_eval.main()
            return {
                "skipped": "[Resume Skipping]" in stdout.getvalue(),
                "loader_calls": loader.call_count,
                "probe_calls": probe.call_count,
                "stdout": stdout.getvalue(),
            }

        def save_row(path, row_dict):
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
                w.writeheader()
                w.writerow(row_dict)

        for dataset in ["fathomnet", "brackish"]:
            title = "FathomNet" if dataset == "fathomnet" else "Brackish"
            src_dir = self.tmp / dataset
            out_csv = self.tmp / f"{dataset}_results.csv"
            write_meta(src_dir, rows)

            base_args = [
                "--checkpoints", str(ckpt),
                "--datasets", dataset,
                f"--{dataset}_dir", str(src_dir),
                "--no-save_cm_plots",
                "--output_csv", str(out_csv),
            ]
            if dataset == "fathomnet":
                base_args += ["--no-fathomnet_cache"]

            # Initial evaluation: probe must execute, output CSV generated
            initial = run_cli(dataset, base_args)
            self.assertFalse(initial["skipped"])
            self.assertEqual(initial["loader_calls"], 1)
            self.assertEqual(initial["probe_calls"], 1)
            self.assertTrue(out_csv.is_file())

            with out_csv.open("r", newline="", encoding="utf-8") as f:
                saved_row = next(csv.DictReader(f))
            initial_fp = saved_row.get(f"{title} Dataset Fingerprint")
            self.assertNotIn(initial_fp, (None, "", "-", "none", "unverifiable"))

            # Exact repeat with --resume: MUST skip!
            exact = run_cli(dataset, base_args + ["--resume"])
            self.assertTrue(exact["skipped"], f"{dataset} exact repeat failed to resume-skip")
            self.assertEqual(exact["loader_calls"], 0)
            self.assertEqual(exact["probe_calls"], 0)

            # Mutation 1: Replace test member of same label -> MUST NOT skip!
            mutated_rows = copy.deepcopy(rows)
            mutated_rows[4]["file_name"] = "replacement_test.jpg"
            write_meta(src_dir, mutated_rows)
            save_row(out_csv, saved_row)
            res = run_cli(dataset, base_args + ["--resume"])
            self.assertFalse(res["skipped"], f"{dataset} member replacement incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)

            # Mutation 2: Replace train member of same label -> MUST NOT skip!
            mutated_rows = copy.deepcopy(rows)
            mutated_rows[0]["file_name"] = "replacement_train.jpg"
            write_meta(src_dir, mutated_rows)
            save_row(out_csv, saved_row)
            res = run_cli(dataset, base_args + ["--resume"])
            self.assertFalse(res["skipped"], f"{dataset} train member replacement incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)

            # Mutation 3: Swap members between train and test partitions -> MUST NOT skip!
            mutated_rows = copy.deepcopy(rows)
            mutated_rows[0]["split"] = "test"
            mutated_rows[4]["split"] = "train"
            write_meta(src_dir, mutated_rows)
            save_row(out_csv, saved_row)
            res = run_cli(dataset, base_args + ["--resume"])
            self.assertFalse(res["skipped"], f"{dataset} partition swap incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)

            # Mutation 4: Swap labels between test images -> MUST NOT skip!
            mutated_rows = copy.deepcopy(rows)
            mutated_rows[4]["label"] = "fish"
            mutated_rows[5]["label"] = "crab"
            write_meta(src_dir, mutated_rows)
            save_row(out_csv, saved_row)
            res = run_cli(dataset, base_args + ["--resume"])
            self.assertFalse(res["skipped"], f"{dataset} label swap incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)

            # Mutation 5: Different source directory -> MUST NOT skip!
            write_meta(src_dir, rows)
            alt_dir = self.tmp / f"{dataset}_alternate"
            write_meta(alt_dir, rows)
            save_row(out_csv, saved_row)
            alt_args = [
                "--checkpoints", str(ckpt),
                "--datasets", dataset,
                f"--{dataset}_dir", str(alt_dir),
                "--no-save_cm_plots",
                "--output_csv", str(out_csv),
                "--resume",
            ]
            if dataset == "fathomnet":
                alt_args += ["--no-fathomnet_cache"]
            res = run_cli(dataset, alt_args)
            self.assertFalse(res["skipped"], f"{dataset} source directory change incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)

            # Mutation 6: Absent stored fingerprint in CSV -> MUST NOT skip!
            row_no_fp = dict(saved_row)
            row_no_fp[f"{title} Dataset Fingerprint"] = "-"
            save_row(out_csv, row_no_fp)
            write_meta(src_dir, rows)
            res = run_cli(dataset, base_args + ["--resume"])
            self.assertFalse(res["skipped"], f"{dataset} absent fingerprint incorrectly skipped")
            self.assertEqual(res["probe_calls"], 1)


class TestTaskRV02EffectiveBaselineSourceIdentity(unittest.TestCase):
    """RV02 / P1: Effective baseline source resolution in run_eval.py.

    Verifies:
    1. Default MarineInst checkpoint resolved to ~/.cache/seadino/checkpoints/marineinst_vit-h_stage1.pth.
    2. Missing/unreadable checkpoint or weights fail closed (never reuse old rows).
    3. MarineInst explicit checkpoint byte mutations and unlinking prevent skipping.
    4. Canonical alias normalization (e.g., MARINEINST, marineinst-encoder, SAM3.1, etc.).
    5. Release-specific exact SAM weights (model.safetensors for SAM 3, sam3.1_multiplex.pt for SAM 3.1) with NO fallback.
    6. Consumed config.json SHA256 tracking and mutation detection.
    7. Local SAM model path precedence over revision.
    8. Unchanged CLIP baseline independence from irrelevant DINO model ID.
    9. Real main() execution and CSV persistence matching rv02_verify.py controls.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tmp = Path(self.temp_dir)
        self.fake_home = self.tmp / "home"
        self.fake_home.mkdir(parents=True, exist_ok=True)
        self.df_dir = self.tmp / "deepfish"
        self.df_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            (self.df_dir / f"{split}.csv").write_text(f"ID,label\n{split}.jpg,0\n", encoding="utf-8")
            (self.df_dir / f"{split}.jpg").write_bytes(b"deepfish_dummy_pixels")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_source_resolution_unit(self):
        """Unit test source resolution, release weight checks, and config tracking."""
        # 1. MarineInst explicit
        marine_pth = self.tmp / "marine_explicit.pth"
        marine_pth.write_bytes(b"marine_bytes_explicit")
        args_marine = argparse.Namespace(marineinst_checkpoint=str(marine_pth))
        for alias in ["marineinst", "marineinst-encoder", "marineinst-vit-h-stage1", "MARINEINST"]:
            ident = resolve_baseline_source_and_weights(alias, args_marine)
            self.assertEqual(ident.target_path, str(marine_pth.resolve()))
            self.assertEqual(ident.weight_sha, get_file_sha256(str(marine_pth)))
            self.assertEqual(ident.effective_model_id, f"{alias}:{marine_pth}")
            self.assertEqual(ident.config_sha, "-")

        # 2. MarineInst default cache path (lowercase and uppercase aliases)
        default_p = self.fake_home / ".cache" / "seadino" / "checkpoints" / "marineinst_vit-h_stage1.pth"
        default_p.parent.mkdir(parents=True, exist_ok=True)
        default_p.write_bytes(b"marine_default_cache_bytes")
        args_default = argparse.Namespace(marineinst_checkpoint=None)
        with patch.object(Path, "home", return_value=self.fake_home):
            for alias in ["marineinst", "MARINEINST"]:
                ident = resolve_baseline_source_and_weights(alias, args_default)
                self.assertEqual(ident.target_path, str(default_p.resolve()))
                self.assertEqual(ident.weight_sha, get_file_sha256(str(default_p)))
                self.assertEqual(ident.config_sha, "-")

        # 3. MarineInst missing file fails closed (weight_sha = "-")
        args_missing = argparse.Namespace(marineinst_checkpoint=str(self.tmp / "nonexistent.pth"))
        ident_missing = resolve_baseline_source_and_weights("marineinst", args_missing)
        self.assertEqual(ident_missing.weight_sha, "-")

        # 4. SAM 3 release-specific weights (model.safetensors) and consumed config
        sam3_dir = self.tmp / "sam3_local_unit"
        sam3_dir.mkdir(parents=True, exist_ok=True)
        sam3_cfg = sam3_dir / "config.json"
        sam3_cfg.write_text('{"model_type": "sam3_vit_model"}', encoding="utf-8")
        sam3_w = sam3_dir / "model.safetensors"
        sam3_w.write_bytes(b"sam3_weights_unit")
        args_sam3 = argparse.Namespace(sam3_model_path=str(sam3_dir), sam3_revision=None)
        for alias in ["sam3", "sam3-encoder", "facebook/sam3", "SAM3"]:
            ident = resolve_baseline_source_and_weights(alias, args_sam3)
            self.assertEqual(ident.weight_sha, get_file_sha256(str(sam3_w)))
            self.assertEqual(ident.config_sha, get_file_sha256(str(sam3_cfg)))
            self.assertEqual(ident.effective_model_id, f"{alias}:{sam3_dir}")

        # SAM 3 fallback rejection: sam3.1_multiplex.pt present but model.safetensors absent
        sam3_w.unlink()
        (sam3_dir / "sam3.1_multiplex.pt").write_bytes(b"wrong_release_weights_31")
        ident_sam3_no_fb = resolve_baseline_source_and_weights("sam3", args_sam3)
        self.assertEqual(ident_sam3_no_fb.weight_sha, "-", "SAM 3 must reject sam3.1_multiplex.pt fallback")

        # 5. SAM 3.1 release-specific weights and consumed config
        sam31_dir = self.tmp / "sam31_local"
        sam31_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = sam31_dir / "config.json"
        cfg_file.write_text('{"image_size": 1008}', encoding="utf-8")
        w_file = sam31_dir / "sam3.1_multiplex.pt"
        w_file.write_bytes(b"sam31_weights")

        args_sam31 = argparse.Namespace(sam3_model_path=str(sam31_dir), sam3_revision=None)
        for alias in ["sam3.1", "sam3.1-encoder", "facebook/sam3.1", "SAM3.1"]:
            ident = resolve_baseline_source_and_weights(alias, args_sam31)
            self.assertEqual(ident.weight_sha, get_file_sha256(str(w_file)))
            self.assertEqual(ident.config_sha, get_file_sha256(str(cfg_file)))
            self.assertEqual(ident.effective_model_id, f"{alias}:{sam31_dir}")

        # 6. SAM 3.1 fallback rejection: model.safetensors present but sam3.1_multiplex.pt absent
        w_file.unlink()
        (sam31_dir / "model.safetensors").write_bytes(b"wrong_release_weights")
        ident_no_fallback = resolve_baseline_source_and_weights("sam3.1", args_sam31)
        self.assertEqual(ident_no_fallback.weight_sha, "-", "SAM 3.1 must reject model.safetensors fallback")

        # 7. SAM config missing fails closed
        w_file.write_bytes(b"sam31_weights")
        cfg_file.unlink()
        ident_no_cfg = resolve_baseline_source_and_weights("sam3.1", args_sam31)
        self.assertEqual(ident_no_cfg.config_sha, "-")
        self.assertEqual(ident_no_cfg.weight_sha, "-")

    def test_rv02_cli_main_recheck_controls(self):
        """Full CLI/main execution matching rv02_verify.py with synthetic local baselines."""
        fake_metrics = {
            k: 50.0 for k in [
                "deepfish_lin_acc", "deepfish_lin_macro_f1", "deepfish_lin_fish_f1",
                "deepfish_knn_acc", "deepfish_knn_macro_f1", "deepfish_knn_fish_f1",
            ]
        }

        def run_cli(argv, custom_home=None):
            loader = MagicMock(return_value=(types.SimpleNamespace(), 1, 0.0, False))
            probe = MagicMock(return_value=fake_metrics)
            stdout, stderr = io.StringIO(), io.StringIO()
            overrides = {
                "setup_hf_cache": MagicMock(),
                "ensure_spatial_split_symlinks": MagicMock(),
                "resolve_csv_path": lambda p, f: p,
                "set_seed": lambda s: None,
                "load_baseline_backbone": loader,
                "evaluate_deepfish": probe,
            }
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["run_eval.py"] + argv))
                stack.enter_context(patch.object(run_eval.torch.cuda, "is_available", return_value=False))
                if custom_home is not None:
                    stack.enter_context(patch.object(Path, "home", return_value=custom_home))
                for name, fn in overrides.items():
                    stack.enter_context(patch.object(run_eval, name, fn))
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    run_eval.main()
            return {
                "skipped": "[Resume Skipping]" in stdout.getvalue(),
                "loader_calls": loader.call_count,
                "probe_calls": probe.call_count,
                "stdout": stdout.getvalue(),
            }

        def save_row(path, row_dict):
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
                w.writeheader()
                w.writerow(row_dict)

        def run_case(name, alias, options, change_fn=None, new_options=None, custom_home=None):
            out_csv = self.tmp / f"{name}.csv"
            prefix = [
                "--baselines", alias,
                "--datasets", "deepfish",
                "--deepfish_dir", str(self.df_dir),
                "--no-save_cm_plots",
                "--output_csv", str(out_csv),
            ]
            initial = run_cli(prefix + options, custom_home=custom_home)
            self.assertFalse(initial["skipped"])
            self.assertEqual(initial["loader_calls"], 1)
            with out_csv.open("r", newline="", encoding="utf-8") as f:
                saved_row = next(csv.DictReader(f))

            exact = run_cli(prefix + options + ["--resume"], custom_home=custom_home)
            self.assertTrue(exact["skipped"], f"{name} exact repeat failed to resume-skip")
            self.assertEqual(exact["loader_calls"], 0)

            save_row(out_csv, saved_row)
            if change_fn:
                change_fn()
            after_opts = options if new_options is None else new_options
            after = run_cli(prefix + after_opts + ["--resume"], custom_home=custom_home)
            return initial, exact, after, saved_row

        # 1. Marine explicit changed bytes -> does NOT skip
        marine = self.tmp / "marine.pth"
        marine.write_bytes(b"marine-A")
        _, _, after, _ = run_case(
            "marine_explicit_changed_bytes", "marineinst",
            ["--marineinst_checkpoint", str(marine)],
            change_fn=lambda: marine.write_bytes(b"marine-B")
        )
        self.assertFalse(after["skipped"])
        self.assertEqual(after["probe_calls"], 1)

        # 2. Marine explicit missing file -> does NOT skip
        marine.write_bytes(b"marine-A")
        _, _, after, _ = run_case(
            "marine_explicit_missing", "marineinst",
            ["--marineinst_checkpoint", str(marine)],
            change_fn=lambda: marine.unlink()
        )
        self.assertFalse(after["skipped"])

        # 3. Marine uppercase alias -> exact repeat skips, byte change does NOT skip
        marine.write_bytes(b"marine-A")
        _, _, after, _ = run_case(
            "marine_alias_uppercase", "MARINEINST",
            ["--marineinst_checkpoint", str(marine)],
            change_fn=lambda: marine.write_bytes(b"marine-B")
        )
        self.assertFalse(after["skipped"])

        # 4. Marine default cache path changed bytes -> does NOT skip
        default_ckpt = self.fake_home / ".cache/seadino/checkpoints/marineinst_vit-h_stage1.pth"
        default_ckpt.parent.mkdir(parents=True, exist_ok=True)
        default_ckpt.write_bytes(b"default-A")
        _, _, after, row = run_case(
            "marine_default_changed_bytes", "marineinst", [],
            change_fn=lambda: default_ckpt.write_bytes(b"default-B"),
            custom_home=self.fake_home
        )
        self.assertEqual(row.get("Checkpoint Path"), str(default_ckpt.resolve()))
        self.assertFalse(after["skipped"])

        # 5. Marine default missing -> does NOT skip
        default_ckpt.write_bytes(b"default-A")
        _, _, after, _ = run_case(
            "marine_default_missing", "marineinst", [],
            change_fn=lambda: default_ckpt.unlink(),
            custom_home=self.fake_home
        )
        self.assertFalse(after["skipped"])

        # 6. SAM 3 (model.safetensors) weights changed -> does NOT skip
        sam3_dir = self.tmp / "sam3-local"
        sam3_dir.mkdir(parents=True, exist_ok=True)
        (sam3_dir / "config.json").write_text('{"model_type": "sam3_vit_model"}', encoding="utf-8")
        sam3_w = sam3_dir / "model.safetensors"
        sam3_w.write_bytes(b"sam3-weights-A")
        _, _, after, _ = run_case(
            "sam3_changed_bytes", "sam3",
            ["--sam3_model_path", str(sam3_dir)],
            change_fn=lambda: sam3_w.write_bytes(b"sam3-weights-B")
        )
        self.assertFalse(after["skipped"])

        # 7. SAM 3 missing expected weights -> does NOT skip
        sam3_w.write_bytes(b"sam3-weights-A")
        _, _, after, _ = run_case(
            "sam3_missing_weights", "sam3",
            ["--sam3_model_path", str(sam3_dir)],
            change_fn=lambda: sam3_w.unlink()
        )
        self.assertFalse(after["skipped"])

        # 8. SAM 3.1 (sam3.1_multiplex.pt) weights changed -> does NOT skip
        sam31_dir = self.tmp / "sam31-local"
        sam31_dir.mkdir(parents=True, exist_ok=True)
        sam31_cfg = sam31_dir / "config.json"
        sam31_cfg.write_text('{"model_type": "sam3_vit_model"}', encoding="utf-8")
        sam31_w = sam31_dir / "sam3.1_multiplex.pt"
        sam31_w.write_bytes(b"sam31-weights-A")
        _, _, after, _ = run_case(
            "sam31_changed_bytes", "SAM3.1",
            ["--sam3_model_path", str(sam31_dir)],
            change_fn=lambda: sam31_w.write_bytes(b"sam31-weights-B")
        )
        self.assertFalse(after["skipped"])

        # 9. SAM 3.1 missing expected weights -> does NOT skip
        sam31_w.write_bytes(b"sam31-weights-A")
        _, _, after, _ = run_case(
            "sam31_missing_weights", "sam3.1",
            ["--sam3_model_path", str(sam31_dir)],
            change_fn=lambda: sam31_w.unlink()
        )
        self.assertFalse(after["skipped"])

        # 10. SAM consumed config.json changed -> does NOT skip
        sam31_w.write_bytes(b"sam31-weights-A")
        _, _, after, row = run_case(
            "sam_config_changed", "sam3.1",
            ["--sam3_model_path", str(sam31_dir)],
            change_fn=lambda: sam31_cfg.write_text('{"model_type": "sam3_vit_model", "image_size": 896}', encoding="utf-8")
        )
        self.assertNotIn(row.get("Config SHA256"), (None, "", "-"))
        self.assertFalse(after["skipped"])

        # 11. SAM 3.1 renaming to model.safetensors (fallback rejection) -> does NOT skip
        sam_fb_dir = self.tmp / "sam-fallback"
        sam_fb_dir.mkdir(parents=True, exist_ok=True)
        (sam_fb_dir / "config.json").write_text('{"model_type": "sam3_vit_model"}', encoding="utf-8")
        fb_w = sam_fb_dir / "sam3.1_multiplex.pt"
        fb_w.write_bytes(b"sam31-expected-weights")
        _, _, after, _ = run_case(
            "sam31_fallback_rejected", "sam3.1",
            ["--sam3_model_path", str(sam_fb_dir)],
            change_fn=lambda: fb_w.rename(sam_fb_dir / "model.safetensors")
        )
        self.assertFalse(after["skipped"], "SAM 3.1 must not accept renamed model.safetensors fallback")

        # 12. SAM local revision is unused (local path takes precedence) -> DOES skip!
        sam_rev_dir = self.tmp / "sam-rev"
        sam_rev_dir.mkdir(parents=True, exist_ok=True)
        (sam_rev_dir / "config.json").write_text('{"model_type": "sam3_vit_model"}', encoding="utf-8")
        (sam_rev_dir / "sam3.1_multiplex.pt").write_bytes(b"sam-rev-weights")
        _, _, after, _ = run_case(
            "sam_local_rev_unused", "sam3.1",
            ["--sam3_model_path", str(sam_rev_dir), "--sam3_revision", "rev-A"],
            new_options=["--sam3_model_path", str(sam_rev_dir), "--sam3_revision", "rev-B"]
        )
        self.assertTrue(after["skipped"], "Local SAM directory should take precedence over remote revision")

        # 13. SAM remote revision changed without local path -> does NOT skip
        _, _, after, _ = run_case(
            "sam_remote_rev_changed", "sam3.1",
            ["--sam3_revision", "rev-A"],
            new_options=["--sam3_revision", "rev-B"]
        )
        self.assertFalse(after["skipped"], "Changing remote revision should trigger rerun")

        # 14. CLIP baseline repeat independent from irrelevant DINO model ID -> DOES skip!
        _, _, after, _ = run_case(
            "clip_repeat_irrelevant_dino", "clip", [],
            new_options=["--model_id", "irrelevant-dinov3-base"]
        )
        self.assertTrue(after["skipped"], "CLIP baseline should be independent from DINO model ID")


if __name__ == "__main__":
    unittest.main()
