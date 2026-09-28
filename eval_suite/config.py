"""Global constants, paths, and normalization parameters with dynamic location resolution."""

import os
from pathlib import Path

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SEADINO_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("SEADINO_RESULTS_DIR", REPO_ROOT / "results"))
CHECKPOINTS_DIR = Path(os.getenv("SEADINO_CHECKPOINTS_DIR", REPO_ROOT / "checkpoints"))

# Normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENTHIC_MEAN = [0.359, 0.413, 0.386]
BENTHIC_STD = [0.219, 0.215, 0.209]


def _find_file(filename: str, subdirs: list[str] | None = None) -> str:
    """Finds a file across candidate locations (env var, subdirs, root)."""
    candidates = []
    if subdirs:
        for sd in subdirs:
            if sd:
                candidates.append(DATA_DIR / sd / filename)
                candidates.append(REPO_ROOT / "data" / sd / filename)
                candidates.append(REPO_ROOT / sd / filename)
    candidates.extend([
        DATA_DIR / filename,
        REPO_ROOT / "data" / filename,
        REPO_ROOT / filename,
        Path(filename),
    ])
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # Return preferred standard location if not yet created
    if subdirs and subdirs[0]:
        return str((DATA_DIR / subdirs[0] / filename).resolve())
    return str((DATA_DIR / filename).resolve())


def _resolve_img_root() -> str:
    env_root = os.getenv("BENTHIC_IMG_ROOT")
    if env_root and Path(env_root).exists():
        return env_root
    default_server = "/home/njan320/Neel/BenthicNet/01_BenthicNet/images/labelled/full_labelled_512px/compiled_labelled_512pix"
    if Path(default_server).exists():
        return default_server
    local_candidate = DATA_DIR / "benthic_images"
    if local_candidate.exists() and any(local_candidate.iterdir()):
        return str(local_candidate.resolve())
    return default_server


def _resolve_coralmask_dir() -> str:
    env_cm = os.getenv("CORALMASK_DIR")
    if env_cm and Path(env_cm).exists():
        return env_cm
    default_server = "/home/njan320/Neel/CoralMaskv1/CoralMask"
    if Path(default_server).exists():
        return default_server
    local_candidate = DATA_DIR / "CoralMask"
    if local_candidate.exists() and any(local_candidate.iterdir()):
        return str(local_candidate.resolve())
    return default_server


def _resolve_biota_csv() -> str:
    spatial_biota = _find_file("biota_spatial_split.csv", ["splits", ""])
    if Path(spatial_biota).exists():
        return spatial_biota
    raw_biota = _find_file("benthicnet_nn.csv", ["splits", ""])
    if Path(raw_biota).exists():
        return raw_biota
    env_biota = os.getenv("BIOTA_CSV")
    if env_biota and Path(env_biota).exists():
        return env_biota
    default_server = "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/benthicnet_nn.csv"
    if Path(default_server).exists():
        return default_server
    return str(DATA_DIR / "splits" / "biota_spatial_split.csv")


DEFAULT_PATHS = {
    "benthic_img_root": _resolve_img_root(),
    "substrate_csv": _find_file("substrate_depth_2_spatial_split.csv", ["splits", ""]),
    "german_bank_csv": _find_file("german_bank_2010_spatial_split.csv", ["splits", ""]),
    "biota_csv": _resolve_biota_csv(),
    "coralmask_dir": _resolve_coralmask_dir(),
    "coralmask_test_manifest": _find_file("coralmask_test_clean.txt", ["manifests", ""]),
    "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "results_dir": str(RESULTS_DIR),
}

# Official class labels matching Fig 7 & Fig 8 in the BenthicNet paper
SUBSTRATE_CLASS_NAMES = ["Boulders", "Cobbles", "Rock", "Pebble/Gravel", "Sand/Mud (<2mm)"]
GERMAN_BANK_CLASS_NAMES = ["silt/mud", "silt with bedforms", "reef", "glacial till", "sand with bedforms"]
