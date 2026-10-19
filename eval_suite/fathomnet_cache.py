"""Exact local pixel/feature caches for deterministic FathomNet evaluation."""
import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torchvision
import torchvision.transforms as T

VERSION = 'fathomnet-cache-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _directory_index(directory):
    """Batch Windows metadata via native enumeration, avoiding one WSL RPC per file."""
    match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', str(directory))
    powershell = shutil.which('powershell.exe') if match else None
    if powershell:
        winpath = match[1].upper() + ':\\' + match[2].replace('/', '\\')
        literal = winpath.replace("'", "''")
        script = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
                  "$ErrorActionPreference='Stop'; $items=@(Get-ChildItem -LiteralPath '" + literal +
                  "' -File | ForEach-Object { [pscustomobject]@{Name=$_.Name;Size=$_.Length;"
                  "Modified=$_.LastWriteTimeUtc.Ticks} }); "
                  "ConvertTo-Json -InputObject $items -Compress -Depth 3")
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                                check=True, capture_output=True, timeout=90)
        rows = json.loads(result.stdout.decode('utf-8-sig'))
        return {str(directory / row['Name']): [int(row['Size']), int(row['Modified'])] for row in rows}
    if match and 'microsoft' in Path('/proc/version').read_text().lower():
        raise RuntimeError('Native Windows file indexing is unavailable; use a local Linux image directory or disable caching')
    result = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                info = entry.stat()
                result[str(directory / entry.name)] = [info.st_size, info.st_mtime_ns]
    return result


def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FathomNetCache:
    def __init__(self, samples, transform, directory=None):
        steps = getattr(transform, 'transforms', ())
        if len(steps) != 3 or type(steps[0]) is not T.Resize or type(steps[1]) is not T.ToTensor or type(steps[2]) is not T.Normalize:
            raise ValueError('Pixel caching requires deterministic Resize / ToTensor / Normalize')
        if not samples or any(not isinstance(item.get('image'), (str, Path)) for item in samples):
            raise ValueError('Caching requires local image paths')
        self.root = Path(directory or os.getenv('SEADINO_FATHOMNET_CACHE', str(Path.home() / '.cache' / 'seadino' / 'fathomnet'))).expanduser()
        self.resize = steps[0]
        self.transform = repr(transform)
        self.source_index = {}
        for parent in sorted({Path(item['image']).absolute().parent for item in samples}, key=str):
            self.source_index.update(_directory_index(parent))
        if any(str(Path(item['image']).absolute()) not in self.source_index for item in samples):
            raise FileNotFoundError('Some FathomNet source images are absent from the file index')
        self._model_stamp = self._model_hash = None
        self.root.mkdir(parents=True, exist_ok=True)

    def pixel_path(self, item, crop_boxes, crop_margin):
        source = str(Path(item['image']).absolute())
        key = digest(dict(version=VERSION, source=source, file=self.source_index[source],
                          bbox=item.get('bbox') if crop_boxes else None,
                          crop_boxes=crop_boxes, crop_margin=crop_margin if crop_boxes else 0,
                          resize=repr(self.resize), pillow=pillow_version, torchvision=torchvision.__version__))
        return self.root / 'pixels' / key[:2] / (key + '.npy')

    def image(self, item, load, crop_boxes, crop_margin):
        path = self.pixel_path(item, crop_boxes, crop_margin)
        try:
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError('Invalid cached RGB pixels')
            return Image.fromarray(array)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, EOFError) as exc:
            print(f'  [FathomNet cache] Rebuilding unreadable pixels: {path.name}: {exc}')
        image = self.resize(load(item))
        array = np.asarray(image, dtype=np.uint8)
        try:
            _atomic_write(path, lambda stream: np.save(stream, array, allow_pickle=False))
        except OSError as exc:
            print(f'  [FathomNet cache] Could not save pixels: {exc}')
        return image

    def _weights_hash(self, model):
        state = model.state_dict()
        stamp = tuple((key, value.data_ptr(), value._version) for key, value in state.items())
        if stamp != self._model_stamp:
            sha = hashlib.sha256()
            for key, value in state.items():
                sha.update(f'{key}|{value.dtype}|{tuple(value.shape)}'.encode())
                sha.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self._model_stamp, self._model_hash = stamp, sha.hexdigest()
        return self._model_hash

    def feature_path(self, model, samples, crop_boxes, device, dtype, batch_size, pipeline):
        records = []
        for item in samples:
            source = str(Path(item['image']).absolute())
            if source not in self.source_index:
                raise ValueError(f'Source not indexed: {source}')
            records.append([source, self.source_index[source], int(item['label']), item.get('bbox') if crop_boxes else None])
        code_files = [Path(__file__), Path(__file__).parent / 'datasets' / 'fathomnet.py',
                      Path(__file__).parent / 'benchmarks' / ('explicit_classification.py' if pipeline == 'explicit' else 'fathomnet_probe.py'),
                      Path(__file__).parent / 'models.py']
        code_files.extend(Path(path) for path in getattr(model, 'feature_cache_source_files', ()))
        code = [hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files]
        key = digest(dict(version=VERSION, pipeline=pipeline, code=code, records=records, crop_boxes=crop_boxes,
                          crop_margin=.05, transform=self.transform, model=self._weights_hash(model),
                          model_class=type(model).__qualname__, adapter=getattr(model, 'benchmark_identity', {}),
                          dtype=str(dtype), batch_size=batch_size,
                          device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                          torch=torch.__version__, torchvision=torchvision.__version__, pillow=pillow_version,
                          default_dtype=str(torch.get_default_dtype()),
                          tf32=torch.backends.cuda.matmul.allow_tf32, float32_precision=torch.get_float32_matmul_precision()))
        return self.root / 'features' / (key + '.pt')

    def load_features(self, path, samples, description):
        started = time.perf_counter()
        try:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            features, labels = saved['features'], saved['labels']
            expected = torch.tensor([int(item['label']) for item in samples], dtype=torch.long)
            if (features.ndim != 2 or len(features) != len(samples) or features.dtype != torch.float32
                    or not torch.equal(labels, expected) or not torch.isfinite(features).all()):
                raise ValueError('Invalid cached features or label order')
            print(f'  [FathomNet cache] Features HIT: {description}, {len(samples):,} images ({time.perf_counter()-started:.2f}s)')
            return features, labels
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError, RuntimeError, KeyError, TypeError, pickle.UnpicklingError) as exc:
            print(f'  [FathomNet cache] Rebuilding unreadable features: {path.name}: {exc}')
            return None

    def save_features(self, path, features, labels):
        try:
            _atomic_write(path, lambda stream: torch.save(dict(features=features.cpu(), labels=labels.cpu()), stream))
        except OSError as exc:
            print(f'  [FathomNet cache] Could not save features: {exc}')


def make_fathomnet_cache(samples, transform, directory=None, enabled=True):
    if not enabled:
        return None
    started = time.perf_counter()
    try:
        cache = FathomNetCache(samples, transform, directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'  [FathomNet cache] Disabled for this run: {exc}')
        return None
    print(f'  [FathomNet cache] Local pixels + features: {cache.root} | source index {time.perf_counter()-started:.2f}s')
    return cache
