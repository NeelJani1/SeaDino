"""Explicit split protocol entry point, dispatched by run_eval.py."""
import csv
import gc
import glob
import hashlib
import json
from pathlib import Path

import torch
import torchvision.transforms as T

from eval_suite.benchmarks.explicit_classification import evaluate_explicit_classification
from eval_suite.benchmarks.explicit_coralscapes import evaluate_explicit_coralscapes
from eval_suite.config import (BENTHIC_MEAN, BENTHIC_STD, BRACKISH_CLASS_NAMES, CHECKPOINTS_DIR,
                               GERMAN_BANK_CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, REPO_ROOT,
                               SUBSTRATE_CLASS_NAMES)
from eval_suite.models import load_baseline_backbone, load_dinov3_backbone
from eval_suite.split_protocol import (VERSION, PARTS, SplitBundle, _row, prepare_benthic,
                                     prepare_local, save_manifest, stable_hash)

SUPPORTED = ('substrate', 'german_bank', 'fathomnet', 'brackish', 'coralscapes')
FIELDS = ['Run ID', 'Protocol', 'Dataset', 'Evaluation Split', 'Model / Label', 'Model ID', 'Model Revision',
          'Checkpoint SHA256', 'Checkpoint Epoch', 'Normalization', 'Seed', 'Image Size',
          'Precision', 'Batch Size', 'Probe Epoch Budget', 'Selected Probe Epoch',
          'Train Count', 'Validation Count', 'Test Count', 'Manifest', 'Manifest Hash',
          'Linear Accuracy (%)', 'Linear Macro-F1 (%)', 'kNN Accuracy (%)', 'kNN Macro-F1 (%)',
          'Validation Linear Macro-F1 (%)', 'Validation kNN Macro-F1 (%)', 'mIoU (%)',
          'Pixel Accuracy (%)', 'Validation mIoU (%)', 'Probe File']


def class_names_from_id_map(labels):
    # Preserve raw mask IDs, including gaps; never compact or shift mask labels.
    return [labels.get(str(i), f'unmapped_id_{i}') for i in range(max(map(int, labels)) + 1)]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    if existing:
        with path.open(newline='') as f:
            if next(csv.reader(f)) != FIELDS:
                raise ValueError(f'Result schema mismatch: {path}. Use a new output CSV for {VERSION}.')
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def load_completed(path):
    if not Path(path).exists() or not Path(path).stat().st_size:
        return set()
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Result schema mismatch: {path}. Use a new output CSV.')
        return {r['Run ID'] for r in reader}


def prepare_selected(args):
    selected = list(SUPPORTED) if 'all' in args.datasets else [n for n in SUPPORTED if n in args.datasets]
    unsupported = set(args.datasets) - set(SUPPORTED) - {'all'}
    if unsupported:
        raise ValueError(f'{VERSION} currently covers {SUPPORTED}; unsupported: {sorted(unsupported)}')
    bundles, coral_splits = {}, None
    for name in selected:
        if name in ['substrate', 'german_bank']:
            bundle = prepare_benthic(name, getattr(args, name + '_csv'), args.benthic_img_root,
                                     'catami_substrate' if name == 'substrate' else 'substrate',
                                     SUBSTRATE_CLASS_NAMES if name == 'substrate' else GERMAN_BANK_CLASS_NAMES, args.seed)
        elif name in ['fathomnet', 'brackish']:
            bundle = prepare_local(name, getattr(args, name + '_dir'), args.seed, BRACKISH_CLASS_NAMES)
        else:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download
            id_path = hf_hub_download(repo_id='EPFL-ECEO/coralscapes', repo_type='dataset', filename='id2label.json')
            labels = json.loads(Path(id_path).read_text())
            coral_splits = load_dataset('EPFL-ECEO/coralscapes')
            if not set(PARTS) <= set(coral_splits):
                raise ValueError('Coralscapes must supply all three official splits')
            rows = []
            for part in PARTS:
                fingerprint = coral_splits[part]._fingerprint
                for i in range(len(coral_splits[part])):
                    image = f'hf://EPFL-ECEO/coralscapes/{fingerprint}/{part}/{i}'
                    rows.append(_row(image, -1, image, part))
            bundle = SplitBundle(name, rows, class_names_from_id_map(labels),
                                 dict(grouping='official dataset splits', official_test=True,
                                      dataset_fingerprints={p: coral_splits[p]._fingerprint for p in PARTS},
                                      history='Validation was previously inspected; official test was not used by the legacy probe.'))
        bundles[name] = save_manifest(bundle, args.split_dir, args.seed)
    return bundles, coral_splits


def _jobs(args):
    paths = list(args.checkpoints)
    for directory in args.checkpoint_dirs:
        paths.extend(sorted(glob.glob(str(Path(directory) / '*.ckpt'))))
    jobs, seen = [], set()
    for path in paths:
        candidates = [Path(path), Path(CHECKPOINTS_DIR) / path]
        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(path)
        if str(resolved) not in seen:
            seen.add(str(resolved))
            jobs.append((resolved.parent.name + '/' + resolved.name, 'checkpoint', str(resolved)))
    jobs.extend((name + ' (baseline)', 'external', name) for name in args.baselines)
    if args.include_off_the_shelf:
        jobs.append(('off-the-shelf (no continued pretraining)', 'checkpoint', None))
    return jobs


def run_explicit_evaluation(args):
    print(f'Protocol: {VERSION} | Reported partition: {args.eval_split.upper()}')
    print('Training fits the probe; validation selects its epoch; final test is scored only after selection.')
    caps = [getattr(args, name + '_max_' + part + '_samples') for name in ['fathomnet', 'brackish'] for part in ['train', 'test']]
    if any(value is not None for value in caps):
        raise ValueError('Explicit-v1 uses complete versioned manifests; remove sample-limit flags or create a separate protocol.')
    if min(getattr(args, name + '_epochs') for name in SUPPORTED) < 1:
        raise ValueError('Probe epoch budgets must be positive')
    bundles, coral_splits = prepare_selected(args)
    if args.prepare_splits_only:
        print('Manifests prepared and checked. No models or probes were run.')
        return
    completed = load_completed(args.output_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.use_bf16 and device.type == 'cuda' else torch.float32
    code_hash = stable_hash([(str(p.relative_to(REPO_ROOT)), file_hash(p)) for p in sorted((REPO_ROOT / 'eval_suite').rglob('*.py'))])
    budgets = {n: getattr(args, n + '_epochs') for n in SUPPORTED}
    print(f'Probe budgets: {budgets}')
    for label, kind, target in _jobs(args):
        checkpoint_hash = file_hash(target) if target and kind == 'checkpoint' else ''
        if kind == 'external':
            model, epoch, _, _ = load_baseline_backbone(
                target, device, dtype, sam3_revision=getattr(args, 'sam3_revision', None),
                sam3_model_path=getattr(args, 'sam3_model_path', None),
                sam3_microbatch_size=getattr(args, 'sam3_microbatch_size', 2),
                marineinst_checkpoint=getattr(args, 'marineinst_checkpoint', None),
                marineinst_microbatch_size=getattr(args, 'marineinst_microbatch_size', 4))
            mean, std = getattr(model, 'norm_mean', IMAGENET_MEAN), getattr(model, 'norm_std', IMAGENET_STD)
            mean = IMAGENET_MEAN if mean is None else mean
            std = IMAGENET_STD if std is None else std
            norm = getattr(model, 'norm_label', 'foundation-native')
        else:
            model, epoch, _, benthic = load_dinov3_backbone(target, args.model_id, device, dtype)
            benthic = benthic if args.norm == 'auto' else args.norm == 'benthic'
            mean, std = (BENTHIC_MEAN, BENTHIC_STD) if benthic else (IMAGENET_MEAN, IMAGENET_STD)
            norm = 'BenthicNet' if benthic else 'ImageNet'
        transform = T.Compose([T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                               T.ToTensor(), T.Normalize(mean, std)])
        revision = getattr(getattr(model, 'config', None), '_commit_hash', '') or ''
        for name, bundle in bundles.items():
            identity = dict(protocol=VERSION, code_hash=code_hash, dataset=name, manifest_hash=bundle.metadata['manifest_hash'],
                            model=target or args.model_id, checkpoint_hash=checkpoint_hash, model_id=args.model_id,
                            model_revision=revision, adapter=getattr(model, 'benchmark_identity', {}),
                            seed=args.seed, eval_split=args.eval_split, norm=[list(mean), list(std)],
                            epochs=budgets[name], image_size=args.image_size, precision=str(dtype),
                            batch_size=args.batch_size, coralscapes_batch_size=args.coralscapes_batch_size,
                            coralscapes_resolution=[448, 896], num_workers=args.num_workers,
                            fathomnet_crop_boxes=args.fathomnet_crop_boxes, brackish_crop_boxes=args.brackish_crop_boxes)
            run_id = stable_hash(identity)
            if args.resume and run_id in completed:
                print(f'[Resume] Completed {label} / {name} / {args.eval_split}: {run_id[:12]}')
                continue
            probe_path = str(Path(args.probe_dir) / f'{name}-{run_id}.pt')
            print(f'\nEvaluating {label} | {name} | {args.eval_split.upper()} | {norm}')
            if name == 'coralscapes':
                metrics = evaluate_explicit_coralscapes(model, coral_splits, len(bundle.class_names), device, dtype,
                    args.eval_split, budgets[name], args.coralscapes_batch_size, args.num_workers, args.seed,
                    mean, std, probe_path, bundle.metadata['manifest_hash'])
            else:
                metrics = evaluate_explicit_classification(model, bundle, transform, device, dtype,
                    args.eval_split, budgets[name], args.batch_size, args.num_workers, args.seed, probe_path,
                    crop_boxes=getattr(args, name + '_crop_boxes', False),
                    cache_enabled=getattr(args, 'fathomnet_cache', True),
                    cache_dir=getattr(args, 'fathomnet_cache_dir', None))
            row = {field: '' for field in FIELDS}
            row.update({'Run ID': run_id, 'Protocol': VERSION, 'Dataset': name, 'Evaluation Split': args.eval_split,
                        'Model / Label': label, 'Model ID': args.model_id if kind == 'checkpoint' else target,
                        'Model Revision': revision,
                        'Checkpoint SHA256': checkpoint_hash, 'Checkpoint Epoch': epoch if epoch is not None else '-',
                        'Normalization': norm, 'Seed': args.seed, 'Image Size': '448x896' if name == 'coralscapes' else str(args.image_size), 'Precision': str(dtype),
                        'Batch Size': args.coralscapes_batch_size if name == 'coralscapes' else args.batch_size,
                        'Probe Epoch Budget': budgets[name], 'Selected Probe Epoch': metrics['selected_probe_epoch'],
                        'Train Count': len(bundle.records('train')), 'Validation Count': len(bundle.records('validation')),
                        'Test Count': len(bundle.records('test')), 'Manifest': bundle.manifest,
                        'Manifest Hash': bundle.metadata['manifest_hash'], 'Probe File': probe_path})
            columns = {'linear_accuracy': 'Linear Accuracy (%)', 'linear_macro_f1': 'Linear Macro-F1 (%)',
                       'knn_accuracy': 'kNN Accuracy (%)', 'knn_macro_f1': 'kNN Macro-F1 (%)',
                       'validation_linear_macro_f1': 'Validation Linear Macro-F1 (%)',
                       'validation_knn_macro_f1': 'Validation kNN Macro-F1 (%)',
                       'miou': 'mIoU (%)', 'pixel_accuracy': 'Pixel Accuracy (%)', 'validation_miou': 'Validation mIoU (%)'}
            row.update({columns[k]: f'{v:.4f}' for k, v in metrics.items() if k in columns})
            write_result(args.output_csv, row)
            print(f'  Saved {args.eval_split.upper()} result: {args.output_csv} | selected probe epoch {metrics["selected_probe_epoch"]}')
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        del model
        gc.collect()
