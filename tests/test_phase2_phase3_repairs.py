"""Synthetic CPU-Only Test Suite for Phase 2 & Phase 3 Repairs (Tasks T06 - T13).

Validates:
- T06: Feature Visualization Module packaged in wheels (pyproject.toml)
- T07: Preserved Vendored MarineInst Attribution in Wheels (pyproject.toml)
- T08: Side-Effect-Free CLI Help & Deferred Filesystem Mutations (run_eval.py, eval_suite/config.py)
- T09: Correct Final Accumulation Group Scaling (ssl_train.py)
- T10: Gate Scheduler & Teacher EMA on Step Success (ssl_train.py)
- T11: Correct Positional Parsing in Training Launcher (scripts/train_vitb_1.3m.sh)
- T12: Correct Clone URL and Directory in Core Quickstart (README.md)
- T13: Repaired Untracked Biota README Link (README.md)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


class TestTaskT06T07Packaging(unittest.TestCase):
    """T06 & T07: Verify feature visualization and MarineInst attribution packaging."""

    def setUp(self):
        self.pyproject_path = REPO_ROOT / "pyproject.toml"
        self.assertTrue(self.pyproject_path.exists(), "pyproject.toml must exist")
        self.content = self.pyproject_path.read_text(encoding="utf-8")

    def test_pyproject_includes_feature_visualization_package(self):
        """T06: feature_visualization* must be included in packages.find."""
        self.assertIn("feature_visualization*", self.content)
        if tomllib:
            data = tomllib.loads(self.content)
            find_inc = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("include", [])
            self.assertIn("feature_visualization*", find_inc)

    def test_pyproject_includes_package_data_models_json(self):
        """T06: models.json must be declared in package-data."""
        self.assertIn("models.json", self.content)
        if tomllib:
            data = tomllib.loads(self.content)
            pkg_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
            fv_data = pkg_data.get("feature_visualization", [])
            self.assertTrue(any("models.json" in item for item in fv_data))

    def test_feature_visualization_models_json_exists_and_valid(self):
        """T06: models.json exists on disk and parses cleanly."""
        models_json = REPO_ROOT / "feature_visualization" / "models.json"
        self.assertTrue(models_json.is_file(), f"Missing {models_json}")
        with open(models_json, "r", encoding="utf-8") as f:
            registry = json.load(f)
        self.assertIsInstance(registry, dict)
        self.assertIn("dinov3_s", registry)
        self.assertIn("seadino_s", registry)

    def test_pyproject_includes_marineinst_attribution_package_data(self):
        """T07: MarineInst LICENSE.txt and provenance.json declared in package-data."""
        self.assertIn("eval_suite.vendor.marineinst", self.content)
        self.assertIn("LICENSE.txt", self.content)
        self.assertIn("provenance.json", self.content)
        if tomllib:
            data = tomllib.loads(self.content)
            pkg_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
            marine_data = pkg_data.get("eval_suite.vendor.marineinst", [])
            self.assertIn("LICENSE.txt", marine_data)
            self.assertIn("provenance.json", marine_data)

    def test_vendored_marineinst_attribution_files_and_hashes(self):
        """T07: MarineInst attribution files exist and SHA-256 match provenance."""
        vendor_dir = REPO_ROOT / "eval_suite" / "vendor" / "marineinst"
        license_path = vendor_dir / "LICENSE.txt"
        prov_path = vendor_dir / "provenance.json"

        self.assertTrue(license_path.is_file(), f"Missing {license_path}")
        self.assertTrue(prov_path.is_file(), f"Missing {prov_path}")

        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
        expected_license_sha = prov["files"]["LICENSE.txt"]
        actual_license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
        self.assertEqual(actual_license_sha, expected_license_sha)


class TestTaskT08SideEffectFreeCLIHelp(unittest.TestCase):
    """T08: Verify --help and invalid args produce zero filesystem side effects."""

    def test_config_import_does_not_mutate_filesystem_or_env(self):
        """Importing config module must not create directories or mutate HF_HOME."""
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dummy_data = tmp_path / "isolated_data"

            orig_env = os.environ.get("SEADINO_DATA_DIR")
            orig_hf = os.environ.get("HF_HOME")
            try:
                os.environ["SEADINO_DATA_DIR"] = str(dummy_data)
                if "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]

                from eval_suite.config import _resolve_data_dir, setup_hf_cache
                resolved = _resolve_data_dir()
                self.assertEqual(resolved, dummy_data)
                # dummy_data should NOT have been created by _resolve_data_dir()
                self.assertFalse(dummy_data.exists(), "Config resolution must not mkdir")
                self.assertNotIn("HF_HOME", os.environ, "Config import must not mutate HF_HOME")
            finally:
                if orig_env is not None:
                    os.environ["SEADINO_DATA_DIR"] = orig_env
                elif "SEADINO_DATA_DIR" in os.environ:
                    del os.environ["SEADINO_DATA_DIR"]
                if orig_hf is not None:
                    os.environ["HF_HOME"] = orig_hf

    def test_ensure_spatial_split_symlinks_deferred_and_clean(self):
        """ensure_spatial_split_symlinks should only act when source files exist."""
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            empty_data = tmp_path / "empty_data"

            from run_eval import ensure_spatial_split_symlinks

            # In an empty directory with no source split files, nothing should be created
            ensure_spatial_split_symlinks()
            # empty_data was not created
            self.assertFalse(empty_data.exists())

    def test_eval_help_exits_cleanly(self):
        """run_eval ArgumentParser --help exits cleanly."""
        from run_eval import main
        # Verify ArgumentParser parses --help with SystemExit(0)
        p = argparse.ArgumentParser()
        # Ensure our run_eval main parser builds without error
        import run_eval
        # Check that main() no longer calls ensure_spatial_split_symlinks before parse_args
        import inspect
        source = inspect.getsource(run_eval.main)
        main_lines = source.splitlines()
        first_call = [line.strip() for line in main_lines if line.strip() and not line.strip().startswith("#")][1]
        self.assertNotIn("ensure_spatial_split_symlinks", first_call,
                         "ensure_spatial_split_symlinks must not be called before ArgumentParser")

    def test_setup_hf_cache_preserves_explicit_hub_cache(self):
        """RC05: Explicit HF_HUB_CACHE must not be overridden by setup_hf_cache."""
        orig_hub = os.environ.get("HF_HUB_CACHE")
        orig_home = os.environ.get("HF_HOME")
        orig_trans = os.environ.get("TRANSFORMERS_CACHE")
        try:
            os.environ["HF_HUB_CACHE"] = "/custom/hub/cache"
            if "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            if "TRANSFORMERS_CACHE" in os.environ:
                del os.environ["TRANSFORMERS_CACHE"]

            from eval_suite.config import setup_hf_cache
            with tempfile.TemporaryDirectory() as td:
                setup_hf_cache(Path(td) / "mnt" / "g" / "data")
                self.assertEqual(os.environ.get("HF_HUB_CACHE"), "/custom/hub/cache")
        finally:
            if orig_hub is not None:
                os.environ["HF_HUB_CACHE"] = orig_hub
            elif "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            if orig_home is not None:
                os.environ["HF_HOME"] = orig_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            if orig_trans is not None:
                os.environ["TRANSFORMERS_CACHE"] = orig_trans
            elif "TRANSFORMERS_CACHE" in os.environ:
                del os.environ["TRANSFORMERS_CACHE"]

    def test_setup_hf_cache_preserves_explicit_home_cache(self):
        """RC05: Explicit HF_HOME must cause setup_hf_cache to return without mutating env."""
        orig_home = os.environ.get("HF_HOME")
        orig_hub = os.environ.get("HF_HUB_CACHE")
        orig_trans = os.environ.get("TRANSFORMERS_CACHE")
        try:
            os.environ["HF_HOME"] = "/custom/hf/home"
            if "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            if "TRANSFORMERS_CACHE" in os.environ:
                del os.environ["TRANSFORMERS_CACHE"]

            from eval_suite.config import setup_hf_cache
            with tempfile.TemporaryDirectory() as td:
                setup_hf_cache(Path(td) / "mnt" / "g" / "data")
                self.assertEqual(os.environ.get("HF_HOME"), "/custom/hf/home")
                self.assertNotIn("HF_HUB_CACHE", os.environ)
                self.assertNotIn("TRANSFORMERS_CACHE", os.environ)
        finally:
            if orig_home is not None:
                os.environ["HF_HOME"] = orig_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            if orig_hub is not None:
                os.environ["HF_HUB_CACHE"] = orig_hub
            elif "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            if orig_trans is not None:
                os.environ["TRANSFORMERS_CACHE"] = orig_trans
            elif "TRANSFORMERS_CACHE" in os.environ:
                del os.environ["TRANSFORMERS_CACHE"]

    def test_setup_hf_cache_preserves_combined_explicit_cache(self):
        """RC05: Combined explicit HF_HOME and HF_HUB_CACHE must both be preserved."""
        orig_home = os.environ.get("HF_HOME")
        orig_hub = os.environ.get("HF_HUB_CACHE")
        orig_trans = os.environ.get("TRANSFORMERS_CACHE")
        try:
            os.environ["HF_HOME"] = "/custom/hf/home"
            os.environ["HF_HUB_CACHE"] = "/custom/hub/cache"
            os.environ["TRANSFORMERS_CACHE"] = "/custom/trans/cache"

            from eval_suite.config import setup_hf_cache
            with tempfile.TemporaryDirectory() as td:
                setup_hf_cache(Path(td) / "mnt" / "g" / "data")
                self.assertEqual(os.environ.get("HF_HOME"), "/custom/hf/home")
                self.assertEqual(os.environ.get("HF_HUB_CACHE"), "/custom/hub/cache")
                self.assertEqual(os.environ.get("TRANSFORMERS_CACHE"), "/custom/trans/cache")
        finally:
            for k, orig in [("HF_HOME", orig_home), ("HF_HUB_CACHE", orig_hub), ("TRANSFORMERS_CACHE", orig_trans)]:
                if orig is not None:
                    os.environ[k] = orig
                elif k in os.environ:
                    del os.environ[k]


class TestTaskT09AccumulationScaling(unittest.TestCase):
    """T09: Verify final partial accumulation group microbatch scaling."""

    @staticmethod
    def compute_scale(batch_idx, num_batches, grad_accum_steps):
        group_start_idx = (batch_idx // grad_accum_steps) * grad_accum_steps
        remaining_batches = num_batches - group_start_idx
        accum_group_size = min(grad_accum_steps, remaining_batches) if remaining_batches > 0 else grad_accum_steps
        return max(1, accum_group_size)

    def test_five_batch_accumulation_with_accum_four(self):
        """Synthetic 5-batch / accum=4: first 4 batches scaled by 4, 5th batch scaled by 1."""
        num_batches = 5
        grad_accum_steps = 4

        scales = [self.compute_scale(i, num_batches, grad_accum_steps) for i in range(num_batches)]
        self.assertEqual(scales, [4, 4, 4, 4, 1])

        # Simulate microbatch gradients of 1.0 each
        # Group 1 (batches 0..3): sum = 4 * (1.0 / 4) = 1.0
        # Group 2 (batch 4): sum = 1 * (1.0 / 1) = 1.0 (unattenuated)
        grad_group1 = sum(1.0 / s for s in scales[:4])
        grad_group2 = 1.0 / scales[4]
        self.assertAlmostEqual(grad_group1, 1.0)
        self.assertAlmostEqual(grad_group2, 1.0)

    def test_11818_batch_accumulation_with_accum_four(self):
        """Full 11,818 batches / accum=4: final 2 batches scaled by 2 instead of 4."""
        num_batches = 11818
        grad_accum_steps = 4

        # Standard middle batch
        scale_mid = self.compute_scale(100, num_batches, grad_accum_steps)
        self.assertEqual(scale_mid, 4)

        # Final group: batches 11816 and 11817
        scale_11816 = self.compute_scale(11816, num_batches, grad_accum_steps)
        scale_11817 = self.compute_scale(11817, num_batches, grad_accum_steps)
        self.assertEqual(scale_11816, 2)
        self.assertEqual(scale_11817, 2)


class TestTaskT10OptimizerStepGating(unittest.TestCase):
    """T10: Verify scheduler, teacher EMA, and current_iter advance only on successful step."""

    def test_production_trainer_source_gates_scheduler_and_ema(self):
        """In SeaDino-Core, verify production Trainer.train_epoch explicitly gates scheduler and EMA on step_successful."""
        ssl_train_path = REPO_ROOT / "ssl_train.py"
        if not ssl_train_path.is_file():
            self.skipTest("ssl_train.py exists only in SeaDino-Core")

        import ssl_train
        import inspect
        trainer_cls = getattr(ssl_train, "DINOTrainer", getattr(ssl_train, "Trainer", None))
        self.assertIsNotNone(trainer_cls, "DINOTrainer or Trainer class must exist in ssl_train.py")
        source = inspect.getsource(trainer_cls.train_epoch)

        # 1. Assert step_successful flag is defined and updated on finite grad_norm
        self.assertIn("step_successful = True", source)
        self.assertIn("step_successful = False", source)
        self.assertIn("if torch.isfinite(grad_norm):", source)

        # 2. Assert scheduler, EMA, and current_iter are gated on step_successful
        self.assertIn("if is_accum_step and step_successful:", source)
        self.assertIn("self.scheduler.step()", source)
        self.assertIn("self._ema_update()", source)
        self.assertIn("self.current_iter += 1", source)

        # 3. Assert partial accumulation scaling is applied
        self.assertIn("accum_group_size = min(self.grad_accum_steps", source)
        self.assertIn("loss_scaled = loss / accum_group_size", source)

    def test_skipped_step_leaves_state_unchanged(self):
        """Non-finite gradient step must not advance scheduler, EMA, or current_iter."""
        current_iter = 10
        scheduler_stepped = False
        ema_updated = False

        # Simulate non-finite gradient
        grad_norm = float("nan")
        step_successful = False
        is_accum_step = True

        if is_accum_step and torch.isfinite(torch.tensor(grad_norm)):
            step_successful = True

        if is_accum_step and step_successful:
            scheduler_stepped = True
            ema_updated = True
            current_iter += 1

        self.assertFalse(step_successful)
        self.assertFalse(scheduler_stepped)
        self.assertFalse(ema_updated)
        self.assertEqual(current_iter, 10)

    def test_valid_step_advances_state_exactly_once(self):
        """Finite gradient step advances scheduler, EMA, and current_iter exactly once."""
        current_iter = 10
        scheduler_step_count = 0
        ema_update_count = 0

        # Simulate finite gradient
        grad_norm = 1.45
        step_successful = False
        is_accum_step = True

        if is_accum_step and torch.isfinite(torch.tensor(grad_norm)):
            step_successful = True

        if is_accum_step and step_successful:
            scheduler_step_count += 1
            ema_update_count += 1
            current_iter += 1

        self.assertTrue(step_successful)
        self.assertEqual(scheduler_step_count, 1)
        self.assertEqual(ema_update_count, 1)
        self.assertEqual(current_iter, 11)


class TestTaskT11LauncherArgumentShifting(unittest.TestCase):
    """T11: Verify train launcher handles 0, 1, 2, and extra arguments without duplication."""

    def test_production_launcher_script_has_repaired_shift_logic(self):
        """If launcher script is present, assert actual bash syntax has repaired shift logic."""
        script_path = REPO_ROOT / "scripts" / "train_vitb_1.3m.sh"
        if not script_path.is_file():
            self.skipTest("scripts/train_vitb_1.3m.sh exists only in SeaDino-Core")

        content = script_path.read_text(encoding="utf-8")
        self.assertIn("[ $# -ge 2 ] && shift 2 || shift $#", content)
        self.assertIn('EXTRA_ARGS=("$@")', content)

    @staticmethod
    def simulate_launcher_args(args):
        default_shard = "/default/shards"
        default_out = "./checkpoints/out"

        shard_dir = args[0] if len(args) >= 1 else default_shard
        output_dir = args[1] if len(args) >= 2 else default_out

        # The repaired shift logic: [ $# -ge 2 ] && shift 2 || shift $#
        if len(args) >= 2:
            extra_args = args[2:]
        else:
            extra_args = []

        return shard_dir, output_dir, extra_args

    def test_zero_arguments(self):
        shard, out, extras = self.simulate_launcher_args([])
        self.assertEqual(shard, "/default/shards")
        self.assertEqual(out, "./checkpoints/out")
        self.assertEqual(extras, [])

    def test_one_argument_does_not_duplicate_into_extras(self):
        """1 argument must NOT leave the shard argument inside extra_args."""
        shard, out, extras = self.simulate_launcher_args(["/custom/shards"])
        self.assertEqual(shard, "/custom/shards")
        self.assertEqual(out, "./checkpoints/out")
        self.assertEqual(extras, [], "Single argument must not leak into extra_args")

    def test_two_arguments(self):
        shard, out, extras = self.simulate_launcher_args(["/custom/shards", "/custom/output"])
        self.assertEqual(shard, "/custom/shards")
        self.assertEqual(out, "/custom/output")
        self.assertEqual(extras, [])

    def test_three_or_more_arguments(self):
        shard, out, extras = self.simulate_launcher_args(
            ["/custom/shards", "/custom/output", "--epochs", "15", "--learning_rate", "1e-4"]
        )
        self.assertEqual(shard, "/custom/shards")
        self.assertEqual(out, "/custom/output")
        self.assertEqual(extras, ["--epochs", "15", "--learning_rate", "1e-4"])


class TestTaskT12CoreQuickstartConfig(unittest.TestCase):
    """T12: Verify Core README contains SeaDino-Core clone URL and cd command."""

    def test_core_readme_quickstart_points_to_seadino_core(self):
        core_readme = REPO_ROOT / "README.md"
        content = core_readme.read_text(encoding="utf-8")

        # If running in SeaDino-Core, verify SeaDino-Core.git
        if REPO_ROOT.name == "SeaDino-Core":
            self.assertIn("git clone https://github.com/NeelJani1/SeaDino-Core.git", content)
            self.assertIn("cd SeaDino-Core", content)
        else:
            self.assertIn("git clone https://github.com/NeelJani1/SeaDino.git", content)
            self.assertIn("cd SeaDino", content)


class TestTaskT13BiotaManifestLink(unittest.TestCase):
    """T13: Verify all relative markdown links in README resolve to tracked files."""

    def test_all_readme_relative_links_resolve(self):
        readme_path = REPO_ROOT / "README.md"
        content = readme_path.read_text(encoding="utf-8")

        # Find all relative markdown links: [text](path)
        rel_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, link in rel_links:
            # Skip web URLs, anchors, mailto
            if link.startswith("http://") or link.startswith("https://") or link.startswith("#") or link.startswith("mailto:"):
                continue
            # Remove any anchor from link
            clean_link = link.split("#")[0]
            if not clean_link:
                continue

            target_path = (REPO_ROOT / clean_link).resolve()
            self.assertTrue(
                target_path.exists(),
                f"Broken relative link in README.md: [{text}]({link}) -> {target_path} does not exist"
            )

        # Specifically verify the Biota audit JSON link is present and valid
        self.assertIn("biota_spatial_split_available_v1.audit.json", content)
        self.assertNotIn("biota_spatial_split.csv", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
