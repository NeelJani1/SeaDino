import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from feature_visualization.common import file_sha
from feature_visualization.dense import mask_view, stage_ground_truth, validate_mask
from feature_visualization.render import render_index


class DenseTests(unittest.TestCase):
    def test_ids_preserved_and_nearest_resize_has_no_new_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = np.array([[0, 0, 9, 9], [5, 5, 39, 255]], dtype=np.uint8)
            records = [dict(sample_id="one", image="hf://source", width=4, height=2)]
            with patch("feature_visualization.data.hf_item", return_value={"label": Image.fromarray(mask)}):
                stage_ground_truth(records, folder, {"class_names": [f"class {i}" for i in range(40)]})
            row = records[0]
            np.testing.assert_array_equal(np.asarray(Image.open(row["mask_path"])), mask)
            rgb, edges = mask_view(row, (8, 4))
            self.assertEqual(rgb.shape, (4, 8, 3))
            self.assertEqual(edges.shape, (4, 8))
            self.assertEqual(len(np.unique(rgb.reshape(-1, 3), axis=0)), 5)
            self.assertEqual(sum(row["mask_class_pixels"].values()), 8)
            self.assertEqual(row["segmentation_classes"]["0"], "unmapped / unlabeled")
            # Boundary drawing does not invent an object boundary at void pixels.
            self.assertFalse(edges[-1, -1])
            Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(row["mask_path"])
            with self.assertRaisesRegex(ValueError, "Semantic mask changed"):
                mask_view(row, (8, 4))

    def test_mask_alignment_and_unknown_ids_fail_visibly(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_mask(np.zeros((2, 3), dtype=np.uint8), (4, 2))
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_mask(np.full((2, 3), 70, dtype=np.uint8), (3, 2))

    def test_attention_progress_links_only_to_existing_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "attention_run.json").write_text(json.dumps({"state": "running", "results": {
                "one": {"name": "Model One", "state": "complete", "page": "one/attention/index.html"}}}))
            result = dict(name="Model One", plots=[], metrics={})
            run = dict(models={"one": {}, "two": {}}, results={"one": result}, records=[], source_name="coralscapes",
                       source={"task": "segmentation", "split": "train"}, seed=42, sample_summary={"labels": {}})
            render_index(root, run, [result], [])
            page = (root / "index.html").read_text()
            self.assertIn("Attention: 1/2 models ready", page)
            self.assertIn("one/attention/index.html", page)
            self.assertNotIn('href="attention_comparisons/index.html"', page)
            self.assertIn("Dense spatial feature comparison", page)
            self.assertNotIn("Cosine silhouette:", page)


if __name__ == "__main__":
    unittest.main()
