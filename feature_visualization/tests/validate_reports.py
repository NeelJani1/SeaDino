"""Check generated all-model reports, artifact links, geometries and parity evidence."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

from feature_visualization.common import file_sha, write_json
from feature_visualization.dense import mask_view


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value and not value.startswith(("http:", "https:", "data:", "#")):
                self.paths.append(value)


def validate(root):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    attention = json.loads((root / "attention_run.json").read_text())
    assert run["state"] == attention["state"] == "complete"
    assert len(run["results"]) == len(attention["results"]) == 9
    assert len(set(r["sample_id"] for r in run["records"])) == len(run["records"])
    dense = run["source"]["task"] == "segmentation"
    if dense:
        assert len(run["records"]) == 20
        assert {r["partition"] for r in run["records"]} == {"train"}
        for record in run["records"]:
            rgb, _ = mask_view(record, (896, 448))
            assert rgb.shape == (448, 896, 3)
    pages = [root / "index.html", root / "attention_comparisons/index.html"]
    comparisons = root / "patch_comparisons"
    assert len(list(comparisons.glob("*.png"))) == len(run["patch_indices"])
    report = {"state": "passed", "models": {}, "dense_ground_truth": dense}
    if run.get("corrections"):
        report["corrections"] = run["corrections"]
        report["sam_square_feature_check"] = run["caches"]["sam31"]["model_metadata"]["square_global_feature_check"]
    selected = attention["sample_indices"]
    for key in run["models"]:
        result = run["results"][key]
        saved = run["caches"][key]
        metadata = saved["model_metadata"]
        assert metadata["parameters"] == metadata["frozen_parameters"]
        assert file_sha(saved["arrays"]) == saved["arrays_sha256"]
        assert len(result["patch_maps"]) == len(run["patch_indices"])
        with np.load(saved["arrays"], allow_pickle=False) as arrays:
            for n, geometry in enumerate(saved["geometry"]):
                assert arrays[f"patch_{n}"].shape[0] == geometry["grid_h"] * geometry["grid_w"]
                assert np.isfinite(arrays[f"patch_{n}"]).all()
                if dense:
                    assert (geometry["width"], geometry["height"]) == (896, 448)
                path = Path(result["patch_maps"][n]["rgb"])
                with Image.open(path) as image:
                    assert image.size == (geometry["width"], geometry["height"])
        found = attention["results"][key]
        assert len(found["items"]) == len(selected)
        assert [i["sample_id"] for i in found["items"]] == [run["records"][i]["sample_id"] for i in selected]
        errors = []
        for item in found["items"]:
            directory = root / key / "attention"
            for field in ("arrays", "metadata"):
                assert file_sha(directory / item[field]) == item[field + "_sha256"]
            details = json.loads((directory / item["metadata"]).read_text())
            assert details["maximum_row_sum_error"] < 2e-4
            assert details["feature_parity"]["passed"]
            errors.append(details["feature_parity"]["maximum_absolute_error"])
            if dense:
                assert details["semantic_reference"]["used_as_model_input"] is False
                assert details["geometry"]["width"] == 896
                assert details["geometry"]["height"] == 448
            for stem in ("comparison", "heads"):
                assert (directory / item[stem]).is_file()
                assert (directory / item[stem]).with_suffix(".pdf").is_file()
        pages.append(root / found["page"])
        if dense:
            pages.append(root / key / "patches.html")
            assert not result["plots"]
        report["models"][key] = {"patch_maps": len(result["patch_maps"]), "attention_maps": len(found["items"]),
                                  "all_parameters_frozen": True, "all_feature_parity_checks_passed": True,
                                  "maximum_absolute_feature_error": max(errors)}
    for page in pages:
        parser = Links()
        parser.feed(page.read_text())
        for link in parser.paths:
            assert (page.parent / unquote(link.split("#")[0])).exists(), f"Broken link: {page}: {link}"
    report["html_pages_checked"] = len(pages)
    if (root / "attention_baseline_integrity.json").exists():
        hashes = json.loads((root / "attention_baseline_integrity.json").read_text())
        assert all(file_sha(root / p) == sha for p, sha in hashes.items())
        report["existing_feature_coordinate_metric_files_unchanged"] = len(hashes)
    write_json(root / "spatial_attention_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    validate(p.parse_args().output_dir)
