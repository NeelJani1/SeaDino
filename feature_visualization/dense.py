"""Lossless semantic references for spatial feature inspection.

Annotations are displayed after extraction. They never fit the PCA basis or
attention and do not turn unsupervised feature colors into class predictions.
"""
import html
from pathlib import Path

import numpy as np
from PIL import Image

from .common import file_sha, write_json

# Official Coralscapes palette, indexed by the original semantic IDs (1..39).
# https://huggingface.co/datasets/EPFL-ECEO/coralscapes
CORALSCAPES_COLORS = np.array([
    (0, 0, 0), (125, 222, 125), (255, 0, 134), (114, 60, 61),
    (250, 224, 225), (194, 178, 128), (224, 118, 119), (255, 0, 0),
    (8, 205, 12), (255, 255, 0), (125, 163, 125), (0, 255, 255),
    (125, 125, 125), (29, 162, 216), (31, 31, 31), (0, 255, 0),
    (255, 248, 228), (236, 150, 21), (161, 153, 128), (252, 231, 240),
    (123, 50, 86), (244, 150, 115), (226, 91, 157), (134, 86, 18),
    (189, 255, 234), (236, 128, 255), (0, 231, 255), (228, 255, 119),
    (189, 119, 255), (240, 80, 80), (0, 255, 189), (255, 146, 150),
    (85, 53, 116), (251, 243, 216), (255, 111, 194), (0, 142, 255),
    (230, 193, 0), (119, 100, 14), (179, 245, 234), (89, 155, 134),
], dtype=np.uint8)


def validate_mask(mask, size):
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind not in "ui" or array.shape != (size[1], size[0]):
        raise ValueError("Semantic mask must contain integer IDs aligned with the original image")
    if not np.isin(array, list(range(40)) + [255]).all():
        raise ValueError("Coralscapes mask contains an unknown semantic ID")
    return array


def color_mask(ids):
    rgb = np.full((*ids.shape, 3), 128, dtype=np.uint8)
    known = ids < len(CORALSCAPES_COLORS)
    rgb[known] = CORALSCAPES_COLORS[ids[known]]
    return rgb


def mask_view(record, size):
    """Nearest-neighbor labels in the exact unpadded feature-map geometry."""
    if file_sha(record["mask_path"]) != record["mask_sha256"]:
        raise ValueError(f"Semantic mask changed: {record['sample_id']}")
    with Image.open(record["mask_path"]) as mask:
        validate_mask(mask, (record["width"], record["height"]))
        ids = np.asarray(mask.resize(size, Image.Resampling.NEAREST))
    valid = (ids > 0) & (ids < 40)
    edges = np.zeros(ids.shape, dtype=bool)
    edges[:, 1:] |= (ids[:, 1:] != ids[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edges[1:, :] |= (ids[1:, :] != ids[:-1, :]) & valid[1:, :] & valid[:-1, :]
    return color_mask(ids) / 255., edges


def legend_html(record):
    entries = []
    total = record["width"] * record["height"]
    for key, count in sorted(record.get("mask_class_pixels", {}).items(), key=lambda x: int(x[0])):
        index = int(key)
        name = record["segmentation_classes"].get(key, f"ID {key}")
        color = CORALSCAPES_COLORS[index] if index < 40 else (128, 128, 128)
        css = "#" + "".join(f"{int(v):02x}" for v in color)
        entries.append(f'<span style="display:inline-block;margin:4px 12px 4px 0"><i style="display:inline-block;width:12px;height:12px;border:1px solid #777;background:{css};margin-right:5px"></i>{key}: {html.escape(name)} ({100 * count / total:.1f}%)</span>')
    return '<div class="semantic-legend">' + ''.join(entries) + '</div>'


def stage_ground_truth(records, root, source):
    from .data import hf_item
    directory = Path(root) / "annotations"
    directory.mkdir(exist_ok=True)
    classes = {str(i): name for i, name in enumerate(source["class_names"])}
    classes["0"] = "unmapped / unlabeled"
    classes["255"] = "void (if present)"
    support = {}
    for record in records:
        mask = hf_item(record["image"])["label"]
        ids = validate_mask(mask, (record["width"], record["height"]))
        stem = record["sample_id"][:20]
        path, colored = directory / f"{stem}_ids.png", directory / f"{stem}_color.png"
        Image.fromarray(ids).save(path)
        Image.fromarray(color_mask(ids)).save(colored)
        labels, counts = np.unique(ids, return_counts=True)
        histogram = {str(int(k)): int(v) for k, v in zip(labels, counts)}
        for key, count in histogram.items():
            support[key] = support.get(key, 0) + count
        record.update(mask_path=str(path), mask_sha256=file_sha(path), mask_color=str(colored),
                      mask_class_pixels=histogram, segmentation_classes=classes)
    metadata = dict(classes=classes, palette=CORALSCAPES_COLORS.tolist(), unmapped_ids=[0, 255],
                    original_ids_preserved=True, resizing="nearest neighbor only", class_pixel_counts=support,
                    palette_source="https://huggingface.co/datasets/EPFL-ECEO/coralscapes",
                    note="Masks are reference annotations. No PCA or attention color is a semantic prediction.")
    source["dense_annotations"] = metadata
    write_json(Path(root) / "semantic_labels.json", metadata)
    return records


def render_patch_page(directory, records, maps, title):
    directory = Path(directory)
    cards = []
    for n, item in enumerate(maps):
        record = records[item["index"]]
        stem = record["sample_id"][:20]
        legend = legend_html(record) if record.get("mask_path") else ""
        with Image.open(directory / "patches" / f"{stem}_comparison.png") as image:
            width, height = image.size
        loading = "eager" if n == 0 else "lazy"
        cards.append(f'<article><a href="patches/{stem}_comparison.png"><img width="{width}" height="{height}" style="width:100%;height:auto" loading="{loading}" src="patches/{stem}_comparison.png" alt="Spatial patch PCA comparison"></a>{legend}<p><a href="patches/{stem}_comparison.pdf">PDF</a> · <a href="patches/{stem}_rgb.png">PCA RGB</a></p></article>')
    semantic_note = " Ground truth uses the official class palette. White boundaries come from annotations; IDs 0/255 are excluded. Percentages describe original-resolution annotation coverage." if any(records[item["index"]].get("mask_path") for item in maps) else ""
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} spatial features</title><style>body{{font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;background:#f4f7fa;color:#18283c}}article{{background:white;padding:16px;margin:24px 0;border-radius:12px}}img{{width:100%}}a{{color:#0874a7}}p{{line-height:1.6}}</style></head><body><a href="../index.html">All models</a><h1>{html.escape(title)} spatial features</h1><p>One shared PCA basis and color scale per model, fitted across the selected images without labels. PCA RGB colors are feature directions, independent across models.{semantic_note}</p>{''.join(cards)}</body></html>'''
    (directory / "patches.html").write_text(page)
