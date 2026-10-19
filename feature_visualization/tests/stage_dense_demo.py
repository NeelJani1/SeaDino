"""CPU preparation and reference inspection for the shared Coralscapes demo."""
import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from pathlib import Path
from feature_visualization.data import benchmark_records, select_records, stage_images
from feature_visualization.dense import stage_ground_truth
from feature_visualization.common import write_json


def main():
    root = Path("/mnt/d/Neel/FeatureVisualizations/coralscapes_all9_20261019")
    root.mkdir(parents=True, exist_ok=True)
    records, source = benchmark_records("coralscapes", "train")
    chosen = select_records(records, 20, 42)
    staged = stage_images(chosen, root)
    stage_ground_truth(staged, root, source)
    write_json(root / "prepared_annotations.json", dict(records=staged, source=source, seed=42))
    print("Prepared", len(staged), "aligned image/mask pairs", flush=True)


if __name__ == "__main__":
    main()
