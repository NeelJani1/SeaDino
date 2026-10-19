"""Versioned train/validation/test manifests for the five transfer benchmarks."""
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree

VERSION = 'explicit-v1'
PARTS = ('train', 'validation', 'test')


@dataclass
class SplitBundle:
    name: str
    rows: list
    class_names: list
    metadata: dict
    manifest: str = ''

    def records(self, partition):
        return [r for r in self.rows if r['partition'] == partition]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def assert_disjoint(rows, enforce_groups=True):
    ids, images, groups = {}, {}, {}
    for r in rows:
        part = r['partition']
        if part not in PARTS:
            continue
        for mapping, key, kind in [(ids, r['sample_id'], 'sample'), (images, r['image'], 'image')]:
            if key in mapping and mapping[key] != part:
                raise ValueError(f'{kind} crosses partitions: {key}')
            mapping[key] = part
        if enforce_groups:
            key = r['group_id']
            if key in groups and groups[key] != part:
                raise ValueError(f'group crosses partitions: {key}')
            groups[key] = part


def grouped_partition(rows, seed=42, validation_fraction=.1, test_fraction=.2):
    """Assign whole groups; choose folds using label support, never model scores."""
    rows = sorted(rows, key=lambda r: r['sample_id'])
    group_count = len({r['group_id'] for r in rows})
    if group_count < (3 if test_fraction else 2):
        raise ValueError('Not enough independent groups for the requested partitions')
    folds = min(10, group_count)
    n_val = max(1, round(validation_fraction * folds))
    n_test = max(1, round(test_fraction * folds)) if test_fraction else 0
    if n_val + n_test >= folds:
        raise ValueError('Split fractions leave no training groups')
    y = np.array([r['label'] for r in rows], dtype=int)
    groups = np.array([r['group_id'] for r in rows])
    classes = np.unique(y)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = np.zeros(len(rows), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for fold, (_, held) in enumerate(cv.split(np.zeros(len(rows)), y, groups)):
            assignments[held] = fold
    hist = np.array([[np.sum(y[assignments == f] == c) for c in classes] for f in range(folds)])
    totals = hist.sum(axis=0)
    target = np.array([1 - validation_fraction - test_fraction, validation_fraction, test_fraction])
    best = None
    for val_folds in itertools.combinations(range(folds), n_val):
        remaining = [f for f in range(folds) if f not in val_folds]
        for test_folds in itertools.combinations(remaining, n_test):
            vc = hist[list(val_folds)].sum(axis=0)
            tc = hist[list(test_folds)].sum(axis=0) if test_folds else np.zeros_like(vc)
            counts = np.stack([totals - vc - tc, vc, tc])
            active = 3 if n_test else 2
            missing = np.sum(counts[:active] == 0)
            score = 1000 * missing + np.square(counts / totals - target[:, None]).sum()
            candidate = (float(score), val_folds, test_folds)
            if best is None or candidate < best:
                best = candidate
    _, val_folds, test_folds = best
    for r, fold in zip(rows, assignments):
        r['partition'] = 'validation' if fold in val_folds else 'test' if fold in test_folds else 'train'
    assert_disjoint(rows)
    train_labels = {r['label'] for r in rows if r['partition'] == 'train'}
    if set(classes) - train_labels:
        raise ValueError('Grouping leaves an observed class absent from training; inspect group support')
    return rows


def _row(image, label, group, source_part, bbox=None, **extra):
    r = dict(image=str(image), label=int(label), bbox=bbox, group_id=group,
             source_partition=source_part, partition=source_part, **extra)
    r['sample_id'] = stable_hash([r['image'], r['label'], bbox])
    return r


def _bbox(value):
    if not value or value == 'nan':
        return None
    return list(ast.literal_eval(value))


@lru_cache(maxsize=1024)
def _image_names(directory):
    try:
        return frozenset(name for name in os.listdir(directory) if Path(name).suffix.lower() in ['.jpg', '.jpeg', '.png'])
    except FileNotFoundError:
        return frozenset()


def _valid_path(path):
    p = Path(path)
    names = _image_names(str(p.parent))
    if p.name in names:
        return str(p)
    for suffix in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
        alt = p.with_suffix(suffix)
        if alt.name in names:
            return str(alt)
    return None


def _spatial_guard(rows, buffer_m):
    val = [r for r in rows if r['partition'] == 'validation']
    train = [r for r in rows if r['partition'] == 'train']
    coords = lambda records: np.radians([[float(r['latitude']), float(r['longitude'])] for r in records])
    if not np.isfinite(coords(val + train)).all():
        raise ValueError('Missing or invalid coordinates in spatial split')
    nearest = BallTree(coords(val), metric='haversine').query(coords(train), k=1)[0][:, 0] * 6371000
    for r, distance in zip(train, nearest):
        if distance < buffer_m:
            r['partition'], r['reason'] = 'dropped', 'validation spatial buffer'
    minimum = {}
    for a, b in itertools.combinations(PARTS, 2):
        aa, bb = [r for r in rows if r['partition'] == a], [r for r in rows if r['partition'] == b]
        if not aa or not bb:
            raise ValueError(f'Empty {a}/{b} after spatial buffering')
        distance = float(BallTree(coords(aa), metric='haversine').query(coords(bb), k=1)[0].min() * 6371000)
        if distance < buffer_m - 1e-6:
            raise ValueError(f'{a}/{b} violates the {buffer_m}m buffer: {distance}')
        minimum[f'{a}_to_{b}'] = distance
    return minimum


def prepare_benthic(name, csv_path, image_root, label_col, class_names, seed):
    raw = sorted(csv.DictReader(Path(csv_path).open()), key=lambda r: (r['dataset'], r['site'], r['image']))
    label_values = sorted({int(ast.literal_eval(r[label_col])[0]) for r in raw if r['partition'].lower() in ['train', 'test']})
    mapping = {label: i for i, label in enumerate(label_values)}
    rows = []
    for r in raw:
        part = r['partition'].lower()
        label = int(ast.literal_eval(r[label_col])[0])
        if label not in mapping:
            continue
        path = Path(image_root) / r['dataset'].strip() / r['site'].strip() / (r['image'].strip() + '.jpg')
        record = _row(path, mapping[label], r['dataset'] + '/' + r['site'], part,
                      latitude=r['latitude'], longitude=r['longitude'])
        if part in ['train', 'test']:
            valid = _valid_path(path)
            if valid is None:
                record['partition'], record['reason'] = 'excluded', 'image not found'
            else:
                record['image'] = valid
        rows.append(record)
    original_test = {r['sample_id'] for r in rows if r['partition'] == 'test'}
    train = [r for r in rows if r['partition'] == 'train']
    grouped_partition(train, seed, test_fraction=0)
    buffer_m = 100 if name == 'substrate' else 50
    distances = _spatial_guard(rows, buffer_m)
    assert original_test == {r['sample_id'] for r in rows if r['partition'] == 'test'}
    assert_disjoint(rows, enforce_groups=False)
    active = [r for r in rows if r['partition'] in ['train', 'validation']]
    assert_disjoint(active)
    return SplitBundle(name, rows, list(class_names), dict(
        source=str(csv_path), grouping='dataset/site for train-validation; original spatial test retained',
        spatial_buffer_m=buffer_m, minimum_distances_m=distances,
        original_test_preserved=True, history='Existing test scores have already been inspected.'))


CLIP = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})to(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')


def video_groups(names):
    """Keep overlapping or <=60-second-separated clips together."""
    intervals = {}
    for name in names:
        m = CLIP.match(Path(name).name)
        if not m:
            raise ValueError(f'Cannot identify Brackish clip: {name}')
        intervals[m.group(0)] = tuple(datetime.strptime(x, '%Y-%m-%d_%H-%M-%S') for x in m.groups())
    mapping, group, end = {}, None, None
    for key, (start, stop) in sorted(intervals.items(), key=lambda item: item[1]):
        if end is None or (start - end).total_seconds() > 60:
            group = 'clip-block:' + key
        end = max(end, stop) if end is not None else stop
        mapping[key] = group
    return {name: mapping[CLIP.match(Path(name).name).group(0)] for name in names}


def prepare_local(name, directory, seed, brackish_names=None):
    directory = Path(directory)
    source = directory if directory.suffix == '.csv' else directory / 'metadata.csv'
    raw = list(csv.DictReader(source.open()))
    base = source.parent
    class_names = list(brackish_names) if name == 'brackish' else sorted({r['label'] for r in raw})
    labels = {c: i for i, c in enumerate(class_names)}
    clips = video_groups([r['file_name'] for r in raw]) if name == 'brackish' else {}
    rows, missing_groups = [], 0
    for r in raw:
        image = base / r['file_name']
        if name == 'brackish':
            group = clips[r['file_name']]
        else:
            group = next((r[k] for k in ['sequence_id', 'video_id', 'dive_id'] if r.get(k)), None)
            url = urlsplit(r.get('url', ''))
            if not group and url.netloc:
                group = url.netloc + url.path.rsplit('/', 1)[0]
            if not group:
                group, missing_groups = str(image), missing_groups + 1
        record = _row(image, labels[r['label']], group, r.get('split', 'train'), _bbox(r.get('bbox')))
        valid = _valid_path(image)
        if valid:
            record['image'] = valid
        else:
            record['partition'], record['reason'] = 'excluded', 'image not found'
        rows.append(record)
    active = [r for r in rows if r['partition'] != 'excluded']
    grouped_partition(active, seed)
    assert_disjoint(rows)
    return SplitBundle(name, rows, class_names, dict(
        source=str(source), grouping='video clip blocks (60-second gap)' if name == 'brackish' else 'source URL directory / sequence identifier',
        nominal_fractions={'train': .7, 'validation': .1, 'test': .2},
        image_only_group_fallbacks=missing_groups, official_test=False,
        history='New local grouped protocol; not comparable to legacy partitions and not a pristine unseen historical test.'))


def save_manifest(bundle, directory, seed):
    directory = Path(directory) / f'seed-{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(bundle.rows, key=lambda r: r['sample_id'])
    signature = stable_hash({'version': VERSION, 'seed': seed, 'rows': rows, 'classes': bundle.class_names})
    path = directory / (bundle.name + '.csv')
    metadata_path = path.with_suffix('.json')
    metadata = dict(bundle.metadata, protocol=VERSION, seed=seed, manifest_hash=signature,
                    class_names=bundle.class_names, counts=dict(Counter(r['partition'] for r in rows)),
                    class_support=None if bundle.name == 'coralscapes' else {
                        p: {c: sum(r['partition'] == p and r['label'] == i for r in rows) for i, c in enumerate(bundle.class_names)} for p in PARTS})
    if path.exists() or metadata_path.exists():
        if not (path.exists() and metadata_path.exists()):
            raise ValueError(f'Incomplete existing manifest: {path}')
        previous = json.loads(metadata_path.read_text())
        if previous['manifest_hash'] != signature or previous['csv_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Manifest/source changed: {path}. Use a new --split_dir; existing identities were not overwritten.')
    else:
        fields = sorted({k for r in rows for k in r})
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
        metadata['csv_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with metadata_path.open('x') as f:
            json.dump(metadata, f, indent=2)
    bundle.manifest, bundle.metadata = str(path), metadata
    for part in PARTS:
        if not bundle.records(part):
            raise ValueError(f'{bundle.name}: empty {part}')
    print(f"  [{bundle.name}] Train={metadata['counts'].get('train', 0):,} | Validation={metadata['counts'].get('validation', 0):,} | Test={metadata['counts'].get('test', 0):,}")
    print(f"    Manifest: {path} | grouping: {metadata['grouping']}")
    absent = {p: [c for c, n in metadata['class_support'][p].items() if not n] for p in PARTS} if metadata['class_support'] else {}
    if any(absent.values()):
        print(f'    Absent classes by partition: {absent}')
    return bundle
