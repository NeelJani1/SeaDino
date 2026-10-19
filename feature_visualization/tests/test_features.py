import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from feature_visualization.analysis import cluster, fit_patch_pca, projection, scale_rgb
from feature_visualization.backbones import audit_backbone, patch_input, spatial_tokens, transform
from feature_visualization.common import digest, normalize_features
from feature_visualization.data import folder_records, image_records, open_image, select_records
from feature_visualization.runner import feature_key, load_cache, save_cache


class FeatureVisualizationTests(unittest.TestCase):
    def test_single_image_accepts_both_input_options_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "sample.JPG"
            Image.new("RGB", (16, 8)).save(image)
            rows, source = image_records(image)
            self.assertEqual(folder_records(image), (rows, source))
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["label_name"])
            self.assertEqual(rows[0]["partition"], "single-image")
            self.assertEqual(rows[0]["image"], str(image.resolve()))
            with self.assertRaisesRegex(ValueError, "use --input_dir"):
                image_records(folder)

    def test_missing_wsl_mount_path_explains_leading_slash(self):
        from unittest.mock import patch
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "leading '/': /mnt/d/Neel/sample.jpg"):
                folder_records("mnt/d/Neel/sample.jpg")
            with self.assertRaisesRegex(FileNotFoundError, "leading '/'"):
                image_records("mnt/d/Neel/sample.jpg")

    def test_balanced_selection_shared_order_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label, count in (("fish", 8), ("coral", 2)):
                (root / label).mkdir()
                for i in range(count):
                    Image.new("RGB", (8, 8), (i, 20, 30)).save(root / label / f"{i}.jpg")
            rows, _ = folder_records(root)
            chosen = select_records(rows + rows, 4, 42)
            self.assertEqual([r["label_name"] for r in chosen], ["coral", "fish", "coral", "fish"])
            self.assertEqual(chosen, select_records(list(reversed(rows)), 4, 42))
            self.assertEqual(len({r["sample_id"] for r in rows}), 10)
            self.assertEqual(len(select_records(rows, 100, 42)), 10)

    def test_unreadable_image_never_substituted(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "broken.jpg"
            broken.write_bytes(b"broken")
            with self.assertRaisesRegex(RuntimeError, "Unreadable sample bad-id"):
                open_image(dict(image=str(broken), sample_id="bad-id"))

    def test_spatial_suffix_excludes_registers(self):
        tokens = torch.arange(2 * 11 * 3).reshape(2, 11, 3)
        spatial, height, width = spatial_tokens(tokens, 32, 48, 16)
        self.assertEqual((height, width), (2, 3))
        torch.testing.assert_close(spatial, tokens[:, 5:, :])
        with self.assertRaises(ValueError):
            spatial_tokens(tokens, 31, 48, 16)

    def test_patch_padding_preserves_display_aspect(self):
        image = Image.new("RGB", (400, 200), (128, 100, 80))
        tensor, geometry = patch_input(image, 512, 14, [0, 0, 0], [1, 1, 1])
        self.assertEqual((geometry["width"], geometry["height"]), (512, 256))
        self.assertEqual(tuple(tensor.shape), (3, 266, 518))
        tokens = torch.zeros(1, 1 + 19 * 37, 8)
        self.assertEqual(spatial_tokens(tokens, 266, 518, 14)[0].shape[1], 19 * 37)

    def test_custom_global_resolution_padding_and_default_parity(self):
        image = Image.new("RGB", (713, 368), (128, 100, 80))
        default = transform([0, 0, 0], [1, 1, 1])(image)
        original = transform([0, 0, 0], [1, 1, 1], 224, 14)(image)
        torch.testing.assert_close(default, original, rtol=0, atol=0)
        custom = transform([0, 0, 0], [1, 1, 1], 256, 14)(image)
        self.assertEqual(tuple(custom.shape), (3, 266, 266))
        exact = transform([0, 0, 0], [1, 1, 1], 448, 16)(image)
        self.assertEqual(tuple(exact.shape), (3, 448, 448))

    def test_strict_teacher_loading_rejects_partial_and_wrong_weights(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.ckpt"
            state = {f"backbone.{k}": v.clone() for k, v in model.state_dict().items()}
            state["dino_head.weight"] = torch.zeros(2, 2)
            torch.save(dict(teacher_state_dict=state, model_state_dict={}), path)
            report = audit_backbone(model, path)
            self.assertEqual(report["selected_state"], "teacher_state_dict")
            self.assertEqual(report["matched_tensors"], 2)
            state.pop("backbone.bias")
            torch.save(dict(teacher_state_dict=state), path)
            with self.assertRaisesRegex(ValueError, "Incomplete backbone"):
                audit_backbone(model, path)
            state["backbone.bias"] = model.bias.detach().clone() + 1
            torch.save(dict(teacher_state_dict=state), path)
            with self.assertRaisesRegex(ValueError, "differs"):
                audit_backbone(model, path)

    def test_clusters_do_not_use_label_values(self):
        x = np.random.default_rng(0).normal(size=(30, 5))
        first, _ = cluster(x, ["fish"] * 15 + ["coral"] * 15, 2, 42)
        second, _ = cluster(x, ["arbitrary"] * 30, 2, 42)
        np.testing.assert_array_equal(first, second)

    def test_tiny_and_constant_inputs_are_finite_and_explicit(self):
        for n in (1, 2, 3, 4):
            x = np.ones((n, 5))
            for method in ("pca", "umap", "tsne"):
                for dimensions in (2, 3):
                    coords, metadata = projection(x, method, dimensions, 42)
                    self.assertEqual(coords.shape, (n, dimensions))
                    self.assertTrue(np.isfinite(coords).all())
                    self.assertIn("fallback", metadata)
            ids, metrics = cluster(x, [None] * n, 0, 42)
            self.assertEqual(metrics["clusters_effective"], 1)
            self.assertIsNone(metrics["silhouette_cosine"])
            self.assertIsNone(metrics["adjusted_rand_index"])

    def test_small_nonconstant_umap_and_tsne_both_dimensions(self):
        features = np.random.default_rng(42).normal(size=(6, 8))
        for method in ("umap", "tsne"):
            for dimensions in (2, 3):
                coordinates, metadata = projection(features, method, dimensions, 42)
                self.assertTrue(np.isfinite(coordinates).all())
                self.assertEqual(coordinates.shape, (6, dimensions))

    def test_patch_basis_shared_across_images_and_constant_safe(self):
        x = np.random.default_rng(1).normal(size=(40, 8)).astype(np.float32)
        scores, metadata = fit_patch_pca([x, x.copy()])
        np.testing.assert_allclose(scores[0], scores[1])
        self.assertTrue(np.isfinite(scale_rgb(scores[0], metadata["color_limits"])).all())
        scores, metadata = fit_patch_pca([np.ones((10, 5), dtype=np.float32)])
        self.assertTrue(np.isfinite(scale_rgb(scores[0], metadata["color_limits"])).all())

    def test_cache_fingerprint_changes_for_weights_images_and_transforms(self):
        from types import SimpleNamespace
        run = dict(code={"feature_visualization/backbones.py": "extract-v1", "feature_visualization/render.py": "render-v1"},
                   versions={"torch": "test"}, records=[dict(sample_id="sample", image_sha256="pixels1")], patch_indices=[0])
        args = SimpleNamespace(patch_longest=512, batch_size=16)
        original = feature_key({"weights": "one"}, run, "s", args)
        self.assertNotEqual(original, feature_key({"weights": "two"}, run, "s", args))
        run["records"][0]["image_sha256"] = "pixels2"
        self.assertNotEqual(original, feature_key({"weights": "one"}, run, "s", args))
        run["records"][0]["image_sha256"] = "pixels1"
        run["code"]["feature_visualization/render.py"] = "render-v2"
        self.assertEqual(original, feature_key({"weights": "one"}, run, "s", args))
        args.patch_longest = 1024
        self.assertNotEqual(original, feature_key({"weights": "one"}, run, "s", args))

    def test_cache_detects_tampering(self):
        x = np.eye(3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            saved = save_cache(folder, "key", x, x, [x], [dict(index=0)], {}, {})
            np.testing.assert_array_equal(load_cache(saved)[0], x)
            with Path(saved["arrays"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_cache(saved)

    def test_subset_rerender_preserves_all_model_cards_without_loading_models(self):
        from unittest.mock import patch
        from feature_visualization.common import file_sha, write_json
        from feature_visualization.runner import main, versions
        x = normalize_features(np.random.default_rng(42).normal(size=(6, 8)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "sample.jpg"
            Image.new("RGB", (20, 10), (10, 20, 30)).save(image)
            records = [dict(sample_id=f"sample-{i}", image=str(image), staged_image=str(image), thumbnail=str(image),
                            image_sha256=file_sha(image), label_name="fish" if i < 3 else "coral", partition="train", width=20, height=10) for i in range(6)]
            saved = save_cache(root / "cache", "cache", x, x, [], [], {}, {})
            null_metrics = dict(clusters_effective=2, silhouette_cosine=None, adjusted_rand_index=None, normalized_mutual_information=None)
            run = dict(models={"one": {"name": "Model One", "size": "S"}, "two": {"name": "Model Two", "size": "B"}},
                       caches={"one": saved, "two": saved}, records=records, seed=42, versions=versions(),
                       source_name="test", source={"task": "classification", "split": "train"}, sample_summary={"labels": {"fish": 3, "coral": 3}},
                       results={"two": dict(name="Model Two", size="B", metrics=null_metrics, plots=[], patch_maps=[])})
            write_json(root / "run.json", run)
            with patch("feature_visualization.backbones.load_backbone", side_effect=AssertionError("Render-only loaded a model")):
                main(["--render_only", "--output_dir", str(root), "--models", "one", "--methods", "pca", "--dims", "2"])
            page = (root / "index.html").read_text()
            self.assertIn("Model One", page)
            self.assertIn("Model Two", page)
            self.assertTrue((root / "one/pca_2d.html").exists())


if __name__ == "__main__":
    unittest.main()
