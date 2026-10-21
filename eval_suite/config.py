"""Global constants, paths, and normalization parameters with dynamic location resolution."""

import os
from pathlib import Path
import sys
from typing import List, Optional

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# Candidate external storage mount points on G drive
# (Strictly isolated to subfolders to avoid polluting root /mnt/g)
GDRIVE_CANDIDATE_PATHS: List[Path] = [
    Path("/mnt/g/data"),
    Path("/mnt/g/SeaDino_Data"),
    Path("/mnt/g/SeaDino/data"),
    Path("/mnt/G/data"),
    Path("/mnt/G/SeaDino_Data"),
    Path("/mnt/G/SeaDino/data"),
]


def detect_gdrive_root() -> Optional[Path]:
    """Detects and returns active G-Drive data directory if mounted and accessible."""
    env_gdrive = os.getenv("GDRIVE_DATA_DIR") or os.getenv("GDRIVE_DIR")
    if env_gdrive and env_gdrive.strip():
        p = Path(env_gdrive.strip())
        try:
            if p.is_dir():
                return p.resolve()
        except (PermissionError, OSError):
            pass

    for cand in GDRIVE_CANDIDATE_PATHS:
        try:
            if cand.is_dir():
                return cand.resolve()
        except (PermissionError, OSError):
            continue

    # Check if /mnt/g is mounted; if so, default to /mnt/g/data
    try:
        g_mount = Path("/mnt/g")
        if g_mount.is_dir():
            target = g_mount / "data"
            try:
                if target.is_dir():
                    return target.resolve()
                return target
            except (PermissionError, OSError):
                pass
    except (PermissionError, OSError):
        pass

    return None


def ensure_gdrive_data_dir(prefer_subfolder: str = "data") -> Path:
    """Auto-creates and returns target data directory on G drive if mounted; falls back to local data/."""
    try:
        g_mount = Path("/mnt/g")
        if g_mount.is_dir():
            target = g_mount / prefer_subfolder
            target.mkdir(parents=True, exist_ok=True)
            return target.resolve()
    except (PermissionError, OSError):
        pass
    return _resolve_data_dir()


def _resolve_data_dir() -> Path:
    """Dynamically resolves primary data directory prioritizing env vars, external G-drive mounts, or local repo data/."""
    env_dir = os.getenv("SEADINO_DATA_DIR")
    if env_dir and env_dir.strip():
        p = Path(env_dir.strip())
        try:
            if p.is_dir():
                return p.resolve()
            return p
        except (PermissionError, OSError):
            pass

    gdrive_root = detect_gdrive_root()
    if gdrive_root is not None:
        return gdrive_root

    local_data = REPO_ROOT / "data"
    try:
        if local_data.is_dir():
            return local_data.resolve()
        return local_data
    except (PermissionError, OSError):
        return local_data


DATA_DIR = _resolve_data_dir()
RESULTS_DIR = Path(os.getenv("SEADINO_RESULTS_DIR", REPO_ROOT / "results"))
CHECKPOINTS_DIR = Path(
    os.getenv(
        "SEADINO_CHECKPOINTS_DIR",
        Path("/mnt/d/Neel/Checkpoints")
        if Path("/mnt/d/Neel/Checkpoints").is_dir()
        else REPO_ROOT / "checkpoints",
    )
)


def setup_hf_cache(data_dir: Optional[Path] = None):
    """Configures HF cache directories on external storage if active and permitted,
    preserving explicit cache variables (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE)."""
    target_data_dir = data_dir or DATA_DIR
    if "mnt/g" in str(target_data_dir) or os.getenv("GDRIVE_DATA_DIR"):
        has_home_override = "HF_HOME" in os.environ
        has_hub_override = "HF_HUB_CACHE" in os.environ or "HUGGINGFACE_HUB_CACHE" in os.environ
        has_trans_override = "TRANSFORMERS_CACHE" in os.environ

        if has_home_override:
            return

        set_home = not has_home_override
        set_hub = not has_hub_override
        set_trans = not has_trans_override and not has_hub_override

        if not (set_home or set_hub or set_trans):
            return

        hf_cache = target_data_dir / ".cache" / "huggingface"
        hf_hub = hf_cache / "hub"
        try:
            hf_cache_str = str(hf_cache)
            hf_hub_str = str(hf_hub)

            if set_home:
                hf_cache.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = hf_cache_str
            if set_hub:
                hf_hub.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HUB_CACHE"] = hf_hub_str
            if set_trans:
                os.environ["TRANSFORMERS_CACHE"] = hf_hub_str

            effective_hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or (hf_hub_str if set_hub else None)
            effective_home = os.environ.get("HF_HOME") or (hf_cache_str if set_home else None)
            effective_trans = os.environ.get("TRANSFORMERS_CACHE") or (hf_hub_str if set_trans else None)

            # Update huggingface_hub constants if already imported or if constants module is active
            if "huggingface_hub.constants" in sys.modules:
                c = sys.modules["huggingface_hub.constants"]
                if effective_home and hasattr(c, "HF_HOME"):
                    c.HF_HOME = effective_home
                if effective_hub:
                    if hasattr(c, "HF_HUB_CACHE"):
                        c.HF_HUB_CACHE = effective_hub
                    if hasattr(c, "HUGGINGFACE_HUB_CACHE"):
                        c.HUGGINGFACE_HUB_CACHE = effective_hub
                if hasattr(c, "default_cache_path") and set_home:
                    c.default_cache_path = os.path.join(hf_cache_str, "hub")
            if "huggingface_hub" in sys.modules:
                hh = sys.modules["huggingface_hub"]
                if hasattr(hh, "constants"):
                    if effective_home and hasattr(hh.constants, "HF_HOME"):
                        hh.constants.HF_HOME = effective_home
                    if effective_hub:
                        if hasattr(hh.constants, "HF_HUB_CACHE"):
                            hh.constants.HF_HUB_CACHE = effective_hub
                        if hasattr(hh.constants, "HUGGINGFACE_HUB_CACHE"):
                            hh.constants.HUGGINGFACE_HUB_CACHE = effective_hub
            if "transformers.utils.hub" in sys.modules and effective_trans:
                t = sys.modules["transformers.utils.hub"]
                if hasattr(t, "TRANSFORMERS_CACHE"):
                    t.TRANSFORMERS_CACHE = effective_trans
        except (PermissionError, OSError):
            pass

# Normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENTHIC_MEAN = [0.359, 0.413, 0.386]
BENTHIC_STD = [0.219, 0.215, 0.209]


def _find_file(filename: str, subdirs: Optional[List[str]] = None) -> str:
    """Finds a file across candidate locations (env var, subdirs, G drive, repo root)."""
    candidates = []
    if subdirs:
        for sd in subdirs:
            if sd:
                candidates.append(DATA_DIR / sd / filename)
                for gd in GDRIVE_CANDIDATE_PATHS:
                    candidates.append(gd / sd / filename)
                candidates.append(REPO_ROOT / "data" / sd / filename)
                candidates.append(REPO_ROOT / sd / filename)
    candidates.extend([
        DATA_DIR / filename,
        REPO_ROOT / "data" / filename,
    ])
    for gd in GDRIVE_CANDIDATE_PATHS:
        candidates.append(gd / filename)
    candidates.extend([
        REPO_ROOT / filename,
        Path(filename),
    ])
    for c in candidates:
        try:
            if c.exists():
                return str(c.resolve())
        except (PermissionError, OSError):
            continue
    if subdirs and subdirs[0]:
        return str((DATA_DIR / subdirs[0] / filename))
    return str((DATA_DIR / filename))


def _resolve_img_root() -> str:
    env_root = os.getenv("BENTHIC_IMG_ROOT")
    if env_root:
        try:
            if Path(env_root).exists():
                return env_root
        except (PermissionError, OSError):
            pass
    default_server = "/home/njan320/Neel/BenthicNet/01_BenthicNet/images/labelled/full_labelled_512px/compiled_labelled_512pix"
    try:
        if Path(default_server).exists():
            return default_server
    except (PermissionError, OSError):
        pass

    candidates = [
        DATA_DIR / "benthic_images",
        Path("/mnt/g/data/benthic_images"),
        Path("/mnt/g/SeaDino_Data/benthic_images"),
        Path("/mnt/g/benthic_images"),
    ]
    for cand in candidates:
        try:
            if cand.exists() and any(cand.iterdir()):
                return str(cand.resolve())
        except (PermissionError, OSError):
            continue
    return default_server


def _resolve_coralmask_dir() -> str:
    env_cm = os.getenv("CORALMASK_DIR")
    if env_cm:
        try:
            if Path(env_cm).exists():
                return env_cm
        except (PermissionError, OSError):
            pass
    default_server = "/home/njan320/Neel/CoralMaskv1/CoralMask"
    try:
        if Path(default_server).exists():
            return default_server
    except (PermissionError, OSError):
        pass

    candidates = [
        DATA_DIR / "CoralMask",
        DATA_DIR / "benchmarks" / "coralmask" / "CoralMask",
        Path("/mnt/g/data/CoralMask"),
        Path("/mnt/g/data/benchmarks/coralmask/CoralMask"),
        Path("/mnt/g/SeaDino_Data/benchmarks/coralmask/CoralMask"),
    ]
    for cand in candidates:
        try:
            if cand.exists() and any(cand.iterdir()):
                return str(cand.resolve())
        except (PermissionError, OSError):
            continue
    return default_server


def _resolve_biota_csv() -> str:
    spatial_biota = _find_file("biota_spatial_split.csv", ["benchmarks/biota", "splits", ""])
    try:
        if Path(spatial_biota).exists():
            return spatial_biota
    except (PermissionError, OSError):
        pass
    raw_biota = _find_file("benthicnet_nn.csv", ["benchmarks/biota", "splits", ""])
    try:
        if Path(raw_biota).exists():
            return raw_biota
    except (PermissionError, OSError):
        pass
    env_biota = os.getenv("BIOTA_CSV")
    if env_biota:
        try:
            if Path(env_biota).exists():
                return env_biota
        except (PermissionError, OSError):
            pass
    default_server = "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/benthicnet_nn.csv"
    try:
        if Path(default_server).exists():
            return default_server
    except (PermissionError, OSError):
        pass
    return str(DATA_DIR / "benchmarks" / "biota" / "biota_spatial_split.csv")


def _resolve_deepfish_dir() -> str:
    env_df = os.getenv("DEEPFISH_DIR")
    if env_df:
        try:
            if Path(env_df).exists():
                return env_df
        except (PermissionError, OSError):
            pass
    candidates = [
        DATA_DIR / "benchmarks" / "deepfish" / "DeepFish",
        DATA_DIR / "benchmarks" / "deepfish",
        DATA_DIR / "DeepFish",
        Path("/mnt/g/data/benchmarks/deepfish/DeepFish"),
        Path("/mnt/g/data/benchmarks/deepfish"),
        Path("/mnt/g/data/DeepFish"),
        Path("/mnt/g/SeaDino_Data/benchmarks/deepfish"),
        REPO_ROOT / "data" / "benchmarks" / "deepfish" / "DeepFish",
        REPO_ROOT / "data" / "benchmarks" / "deepfish",
        REPO_ROOT / "DeepFish",
        Path("/home/njan320/Neel/DeepFish"),
        Path("/home/njan320/DeepFish"),
    ]
    for c in candidates:
        try:
            if c.exists() and ((c / "Classification").exists() or (c / "train.csv").exists()):
                return str(c.resolve())
        except (PermissionError, OSError):
            continue
    return str((DATA_DIR / "benchmarks" / "deepfish").resolve())


def _resolve_fathomnet_dir() -> str:
    env_fn = os.getenv("FATHOMNET_DIR")
    if env_fn:
        try:
            if Path(env_fn).exists():
                return env_fn
        except (PermissionError, OSError):
            pass
    candidates = [
        DATA_DIR / "benchmarks" / "fathomnet",
        DATA_DIR / "fathomnet",
        Path("/mnt/g/data/benchmarks/fathomnet"),
        Path("/mnt/g/data/fathomnet"),
        Path("/mnt/g/SeaDino_Data/benchmarks/fathomnet"),
        REPO_ROOT / "data" / "benchmarks" / "fathomnet",
        REPO_ROOT / "fathomnet",
    ]
    for c in candidates:
        try:
            if c.is_dir() and (
                (c / "train").exists()
                or (c / "metadata.csv").exists()
                or (c / "images").exists()
                or (c / "annotations.csv").exists()
            ):
                return str(c.resolve())
        except (PermissionError, OSError):
            continue
    return "FathomNet/2024-vme-benchmark"


def _resolve_brackish_dir() -> str:
    env_br = os.getenv("BRACKISH_DIR")
    if env_br:
        try:
            if Path(env_br).exists():
                return env_br
        except (PermissionError, OSError):
            pass
    candidates = [
        DATA_DIR / "benchmarks" / "brackish",
        DATA_DIR / "brackish",
        Path("/mnt/g/data/benchmarks/brackish"),
        Path("/mnt/g/data/brackish"),
        Path("/mnt/g/SeaDino_Data/benchmarks/brackish"),
        REPO_ROOT / "data" / "benchmarks" / "brackish",
        REPO_ROOT / "brackish",
    ]
    for c in candidates:
        try:
            if c.is_dir() and (
                (c / "train").exists()
                or (c / "images").exists()
                or (c / "metadata.csv").exists()
            ):
                return str(c.resolve())
        except (PermissionError, OSError):
            continue
    return str((DATA_DIR / "benchmarks" / "brackish").resolve())


DEFAULT_PATHS = {
    "benthic_img_root": _resolve_img_root(),
    "substrate_csv": _find_file("substrate_depth_2_spatial_split.csv", ["benchmarks/substrate", "splits", ""]),
    "german_bank_csv": _find_file("german_bank_2010_spatial_split.csv", ["benchmarks/german_bank", "splits", ""]),
    "biota_csv": _resolve_biota_csv(),
    "coralmask_dir": _resolve_coralmask_dir(),
    "coralmask_test_manifest": _find_file("coralmask_test_clean.txt", ["benchmarks/coralmask", "manifests", ""]),
    "deepfish_dir": _resolve_deepfish_dir(),
    "fathomnet_dir": _resolve_fathomnet_dir(),
    "brackish_dir": _resolve_brackish_dir(),
    "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "results_dir": str(RESULTS_DIR),
    "data_dir": str(DATA_DIR),
}

SUBSTRATE_CLASS_NAMES = ["Boulders", "Cobbles", "Rock", "Pebble/Gravel", "Sand/Mud (<2mm)"]
GERMAN_BANK_CLASS_NAMES = ["silt/mud", "silt with bedforms", "reef", "glacial till", "sand with bedforms"]
DEEPFISH_CLASS_NAMES = ["No-Fish", "Fish"]
BRACKISH_CLASS_NAMES = ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"]
