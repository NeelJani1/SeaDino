"""Small, deterministic serialization and provenance helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def code_identity():
    root = Path(__file__).parent
    paths = list(root.glob("*.py")) + [root / "models.json"]
    shared = root.parent / "eval_suite"
    paths += [shared / n for n in ("models.py", "sam3_backbone.py", "marineinst_backbone.py", "config.py")]
    return {str(p.relative_to(root.parent)): file_sha(p) for p in sorted(paths) if p.is_file()}


def normalize_features(features):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Features must be a nonempty finite N x D matrix")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
