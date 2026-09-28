#!/usr/bin/env python3
"""
Embed 20 sample images (10 clean + 10 dropped buffer) directly into german_bank_sample_gallery.html as base64 data URIs.
"""

import argparse
import base64
import os
import re
from pathlib import Path

DEFAULT_HTML_PATH = os.getenv(
    "GALLERY_HTML_PATH",
    "/home/njan320/snap/antigravity-cli/common/.gemini/antigravity-cli/brain/84c20516-fb78-4ca2-ac2d-1fb03ad8f2e3/german_bank_sample_gallery.html"
)
DEFAULT_IMG_ROOT = os.getenv(
    "BENTHIC_IMG_ROOT",
    "/home/njan320/Neel/BenthicNet/01_BenthicNet/images/labelled/full_labelled_512px/compiled_labelled_512pix/German_Bank_2010"
)

IMAGE_IDS = [
    # Clean Samples (Part 1)
    ("EX01", "EX01_262151309"),
    ("EX01", "EX01_262152108"),
    ("EX01", "EX01_262151529"),
    ("EX01", "EX01_262152213"),
    ("Hp16", "Hp16_259224712"),
    ("EX01", "EX01_262152151"),
    ("HP04", "HP04_258122030"),
    ("HP03", "HP03_258110447"),
    ("Hp17", "Hp17_259210803"),
    ("Hp17", "Hp17_259205625"),

    # Dropped Buffer Samples (Part 2)
    ("EX01", "EX01_262152018"),
    ("EX01", "EX01_262152033"),
    ("EX01", "EX01_262152044"),
    ("EX01", "EX01_262152336"),
    ("EX01", "EX01_262152407"),
    ("HP04", "HP04_258121940"),
    ("HP04", "HP04_258121958"),
    ("HP04", "HP04_258122015"),
    ("Hp16", "Hp16_259224656"),
    ("Hp17", "Hp17_259210747"),
]

def main():
    p = argparse.ArgumentParser(description="Embed base64 images into sample gallery HTML")
    p.add_argument("--html_path", type=str, default=DEFAULT_HTML_PATH, help="Path to gallery HTML file")
    p.add_argument("--img_root", type=str, default=DEFAULT_IMG_ROOT, help="Root path to German Bank images")
    args = p.parse_args()

    html_path = args.html_path
    base_img_root = args.img_root

    print(f"Reading HTML template from: {html_path}")
    if not os.path.exists(html_path):
        print(f"Error: HTML template not found: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    success_count = 0
    for site, img_id in IMAGE_IDS:
        img_path = f"{base_img_root}/{site}/{img_id}.jpg"
        if not os.path.exists(img_path):
            img_path = f"{base_img_root}/{site}/{img_id}.JPG"

        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found: {img_path}")
            continue

        with open(img_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_data}"

        pattern = rf'file:///[^"\'>]+{img_id}\.jpg'
        html, count = re.subn(pattern, data_uri, html)
        if count > 0:
            success_count += 1
            print(f"  [+] Embedded {img_id} ({len(b64_data)/1024:.1f} KB base64)")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nSUCCESS: Embedded {success_count} / {len(IMAGE_IDS)} images directly into HTML!")
    print(f"File updated: {html_path}")
    print("Refresh your browser tab to see all 20 images.")

if __name__ == "__main__":
    main()
