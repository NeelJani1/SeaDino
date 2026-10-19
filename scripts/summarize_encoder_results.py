"""Collate saved encoder benchmark scores without training or editing raw CSVs.

Run after the benchmark queue finishes:
    python scripts/summarize_encoder_results.py --require_complete

Missing scores remain pending. --require_complete writes the same truthful report
and exits 2 until MarineInst and SAM3.1 each have all eight benchmark results.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT_DEFAULT = Path('/home/njan320/SeaDino')
ART_DEFAULT = Path('/mnt/c/Users/njan320/.codex/visualizations/2026/10/08/01a11947-acbf-7082-8bd6-ca6e9be30766')
EXPLICIT = ('substrate', 'german_bank', 'fathomnet', 'brackish', 'coralscapes')
LEGACY = ('deepfish', 'coralmask', 'biota')
NEW = ('marineinst', 'sam31')
NAMES = {'substrate': 'Substrate', 'german_bank': 'German Bank', 'fathomnet': 'FathomNet',
         'brackish': 'Brackish', 'coralscapes': 'Coralscapes', 'deepfish': 'DeepFish',
         'coralmask': 'CoralMask', 'biota': 'Biota'}
EXPLICIT_FILES = ('results/vitb_explicit_test.csv', 'results/vitb_baselines_explicit_test.csv',
                  'results/vits_explicit_test.csv', 'results/marineinst_encoder/explicit_test.csv',
                  'results/sam31_encoder/explicit_test.csv')
LEGACY_FILES = {
    'deepfish': ('results/vitb_deepfish_eval.csv', 'results/marineinst_encoder/deepfish_legacy.csv',
                'results/sam31_encoder/deepfish_legacy.csv',
                'results/deepfish_refresh_20261016/dinov3_s.csv',
                'results/deepfish_refresh_20261016/seadino_s_physics.csv',
                'results/deepfish_refresh_20261016/dinov3_b.csv',
                'results/deepfish_refresh_20261016/seadino_b.csv',
                'results/deepfish_refresh_20261016/dinov2_s.csv',
                'results/deepfish_refresh_20261016/dinov2_b.csv',
                'results/deepfish_refresh_20261016/clip.csv',
                'results/deepfish_refresh_20261016/sam31.csv',
                'results/deepfish_refresh_20261016/marineinst.csv'),
    'coralmask': ('results/vitb_coralmask_eval.csv', 'results/marineinst_encoder/coralmask_legacy.csv',
                 'results/sam31_encoder/coralmask_legacy.csv',
                 'results/coralmask_missing_models_20261015/dinov3_s.csv',
                 'results/coralmask_missing_models_20261015/seadino_s_physics.csv',
                 'results/coralmask_missing_models_20261015/dinov2_s.csv',
                 'results/coralmask_missing_models_20261015/dinov2_b.csv',
                 'results/coralmask_missing_models_20261015/clip.csv',
                 'results/coralmask_refresh_20261016/dinov3_b.csv',
                 'results/coralmask_refresh_20261016/seadino_b.csv',
                 'results/coralmask_refresh_20261016/sam31.csv',
                 'results/coralmask_refresh_20261016/marineinst.csv'),
    'biota': ('results/vitb_biota_eval.csv', 'results/marineinst_biota_available_v1/biota_legacy.csv',
             'results/sam31_encoder/biota_legacy.csv'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def numeric(value):
    try:
        result = float(str(value).strip().removesuffix('%'))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def fmt(value, missing='pending'):
    result = numeric(value)
    if result is None:
        return missing
    # CSVs retain decimal strings. Avoid a binary-float half-cent artifact such
    # as formatting the saved 56.5350 as 56.53.
    decimal = Decimal(str(value).strip().removesuffix('%'))
    return str(decimal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def link(path, label=None):
    target = str(path).replace('\\', '/')
    if target.startswith('/mnt/c/'):
        target = 'C:/' + target[len('/mnt/c/'):]
    return f'[{label or Path(path).name}](<{target}>)'


def family(row):
    model = row.get('Model ID', '').lower()
    label = row.get('Model / Label', '').lower()
    if 'bioclip' in model or 'bioclip' in label:
        return 'invalid_bioclip'
    if 'marineinst' in model or 'marineinst' in label:
        return 'marineinst'
    if 'sam3.1' in model or 'sam31' in model or 'sam3.1' in label or 'sam31' in label:
        return 'sam31'
    if 'dinov3-vitb16' in model:
        return 'dinov3_b'
    if 'dinov3-vits16' in model:
        return 'dinov3_s'
    if 'dinov2-base' in model:
        return 'dinov2_b'
    if 'dinov2-small' in model:
        return 'dinov2_s'
    if model == 'clip':
        return 'clip'
    return 'other'


def is_original(row):
    # Model ID disambiguates the otherwise identical original-S/original-B label.
    return family(row) in ('dinov3_b', 'dinov3_s') and (
        row.get('Checkpoint Epoch', '-') in ('', '-') and
        not row.get('Checkpoint SHA256', '') and
        row.get('Model / Label', '').startswith('off-the-shelf'))


def epoch(row):
    value = row.get('Checkpoint Epoch', row.get('Epoch', '-'))
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return str(value)


def condition(row):
    return (row.get('Model ID', ''), row.get('Model / Label', ''),
            row.get('Normalization', row.get('Norm', '')), epoch(row),
            row.get('Checkpoint SHA256', ''), row.get('Model Revision', ''), row.get('Seed', ''))


def condition_name(row):
    kind = family(row)
    if kind == 'marineinst': return 'MarineInst ViT-H encoder'
    if kind == 'sam31': return 'SAM3.1 ViT encoder'
    if is_original(row): return 'Original DINOv3-' + ('B' if kind == 'dinov3_b' else 'S')
    if kind == 'dinov2_b': return 'DINOv2-B'
    if kind == 'dinov2_s': return 'DINOv2-S'
    if kind == 'clip': return 'CLIP-B'
    label = row.get('Model / Label', '')
    if kind == 'dinov3_b': return 'SeaDino-B checkpoint ' + epoch(row).zfill(2)
    if kind == 'dinov3_s':
        prefix = label.split('/')[0]
        known = {'stage1_marine_with_physics': 'SeaDino-S marine physics',
                 'stage1_marine_no_physics': 'SeaDino-S marine no physics',
                 'ssl_off_pix': 'SeaDino-S input masking',
                 'ssl_off_no_pix': 'SeaDino-S no input masking',
                 'stage2_gram_from_2e4_ep14': 'SeaDino-S Stage 2 Gram'}
        return known.get(prefix, prefix) + ' checkpoint ' + epoch(row).zfill(2)
    return label or row.get('Model ID', 'Unknown model')


def ordering(row):
    kind = family(row)
    order = {'dinov3_s': 0, 'dinov2_s': 1, 'dinov3_b': 2, 'dinov2_b': 3,
             'clip': 4, 'marineinst': 5, 'sam31': 6, 'other': 7}
    return order.get(kind, 8), 0 if is_original(row) else 1, condition_name(row)


def load_csv(root, relative, sources):
    path = root / relative
    info = {'path': str(path), 'exists': path.is_file()}
    if not path.is_file():
        sources.append(info)
        return []
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    info.update(sha256=digest(data), bytes=len(data), row_count=len(rows), fields=reader.fieldnames)
    sources.append(info)
    return [{**row, '_source_file': str(path), '_source_csv_line': index + 2}
            for index, row in enumerate(rows)]


def load_json(path, attachments):
    if not path.is_file(): return None
    data = path.read_bytes()
    attachments.append({'path': str(path), 'sha256': digest(data), 'bytes': len(data)})
    return json.loads(data)


def pair(row, linear, knn, missing='pending'):
    return f'{fmt(row.get(linear), missing)} / {fmt(row.get(knn), missing)}' if row else missing


def explicit_complete(row, dataset):
    keys = ('mIoU (%)',) if dataset == 'coralscapes' else ('Linear Macro-F1 (%)', 'kNN Macro-F1 (%)')
    return bool(row) and all(numeric(row.get(key)) is not None for key in keys)


def legacy_complete(row, dataset):
    keys = {'deepfish': ('DeepFish Lin F1-Macro (%)', 'DeepFish kNN F1-Macro (%)'),
            'coralmask': ('CoralMask mIoU (%)',),
            'biota': ('Biota Lin mAP (%)', 'Biota kNN mAP (%)')}[dataset]
    return bool(row) and all(numeric(row.get(key)) is not None for key in keys)


def legacy_identity(row, relative, dataset):
    label = row.get('Model / Label', '')
    kind = family(row)
    if kind in NEW: return kind
    if dataset == 'coralmask' and Path(relative).parent.name == 'coralmask_refresh_20261016':
        expected = {
            'dinov3_b.csv': ('original_b', 'off-the-shelf (no continued pretraining)'),
            'seadino_b.csv': ('seadino_b10', 'stage1_vitb_marine_lr1e-5_head1/benthic-ssl-epoch=10-ssl_loss=14.07.ckpt'),
        }.get(Path(relative).name)
        if expected and label == expected[1] and row.get('Norm') in ('ImageNet','BenthicNet'):
            return expected[0]
        return None
    if dataset == 'coralmask' and Path(relative).parent.name == 'coralmask_missing_models_20261015':
        expected = {
            'dinov3_s.csv': ('original_s', 'off-the-shelf (no continued pretraining)'),
            'seadino_s_physics.csv': ('seadino_s_physics', 'stage1_marine_with_physics/benthic-ssl-epoch=14-ssl_loss=12.84.ckpt'),
            'dinov2_s.csv': ('dinov2_s', 'dinov2-small (baseline)'),
            'dinov2_b.csv': ('dinov2_b', 'dinov2-base (baseline)'),
            'clip.csv': ('clip', 'clip (baseline)'),
        }.get(Path(relative).name)
        if expected and label == expected[1] and row.get('Norm') in ('ImageNet','CLIP'):
            return expected[0]
        return None
    if dataset == 'deepfish' and Path(relative).parent.name == 'deepfish_refresh_20261016':
        expected = {
            'dinov3_s.csv': ('original_s', 'off-the-shelf (no continued pretraining)'),
            'seadino_s_physics.csv': ('seadino_s_physics', 'stage1_marine_with_physics/benthic-ssl-epoch=14-ssl_loss=12.84.ckpt'),
            'dinov3_b.csv': ('original_b', 'off-the-shelf (no continued pretraining)'),
            'seadino_b.csv': ('seadino_b10', 'stage1_vitb_marine_lr1e-5_head1/benthic-ssl-epoch=10-ssl_loss=14.07.ckpt'),
            'dinov2_s.csv': ('dinov2_s', 'dinov2-small (baseline)'),
            'dinov2_b.csv': ('dinov2_b', 'dinov2-base (baseline)'),
            'clip.csv': ('clip', 'clip (baseline)'),
        }.get(Path(relative).name)
        if expected and label == expected[1] and row.get('Norm') in ('ImageNet','BenthicNet','CLIP'):
            return expected[0]
        return None
    # Legacy exports omitted Model ID; only these specifically named ViT-B CSVs
    # are admitted. Ambiguous off-the-shelf rows from ViT-S files are not inferred.
    if Path(relative).name.startswith('vitb_') and label.startswith('off-the-shelf'):
        return 'original_b'
    if 'stage1_vitb_marine_lr1e-5_head1/' in label and epoch(row) == '10':
        return 'seadino_b10'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--artifact_dir', type=Path, default=ART_DEFAULT)
    parser.add_argument('--require_complete', action='store_true')
    args = parser.parse_args()
    root, art = args.root.resolve(), args.artifact_dir.resolve()
    sources, excluded, selected, replacements = [], [], {}, []
    for relative in EXPLICIT_FILES:
        for row in load_csv(root, relative, sources):
            if family(row) == 'invalid_bioclip':
                excluded.append({'reason': 'Known invalid BioCLIP batch/token-axis wrapper', 'row': row})
                continue
            if row.get('Protocol') != 'explicit-v1' or row.get('Evaluation Split') != 'test' or row.get('Dataset') not in EXPLICIT:
                excluded.append({'reason': 'Not a requested explicit-v1 TEST result', 'row': row})
                continue
            key = condition(row), row['Dataset']
            if key in selected:
                replacements.append({'discarded': selected[key], 'selected_later_row': row})
            selected[key] = row
    conditions = {}
    for (key, dataset), row in selected.items():
        conditions.setdefault(key, {})[dataset] = row
    # Pending new-model rows remain visible even before the first CSV is written.
    for kind, model in (('marineinst', 'marineinst-encoder'), ('sam31', 'sam3.1-encoder')):
        if not any(family(next(iter(rows.values()))) == kind for rows in conditions.values()):
            placeholder = {'Model ID': model, 'Model / Label': model + ' (baseline)'}
            conditions[condition(placeholder)] = {}

    legacy = {}
    for dataset, relatives in LEGACY_FILES.items():
        for relative in relatives:
            for row in load_csv(root, relative, sources):
                identity = legacy_identity(row, relative, dataset)
                if identity:
                    key = identity, dataset
                    if key in legacy:
                        replacements.append({'discarded': legacy[key], 'selected_later_row': row})
                    legacy[key] = row
                else:
                    excluded.append({'reason': 'Legacy model outside requested B/new-model comparison', 'row': row})

    newest_explicit = {}
    for row in selected.values():
        kind = family(row)
        if kind in NEW:
            newest_explicit[kind, row['Dataset']] = row
    completeness = {}
    for kind in NEW:
        checks = {dataset: explicit_complete(newest_explicit.get((kind, dataset)), dataset) for dataset in EXPLICIT}
        checks.update({dataset: legacy_complete(legacy.get((kind, dataset)), dataset) for dataset in LEGACY})
        completeness[kind] = {'complete': all(checks.values()), 'completed': sum(checks.values()),
                              'requested': 8, 'datasets': checks,
                              'pending': [name for name, complete in checks.items() if not complete]}

    attachments = []
    integration = art / 'sam3-integration'
    downloads = load_json(integration / 'sam3-approved-downloads.json', attachments)
    smoke = load_json(integration / 'sam31-real-weight-smoke.json', attachments)
    conversion = load_json(integration / 'sam31-weight-conversion-evidence.json', attachments)
    parity = load_json(integration / 'native31-parity/native31-parity-results.json', attachments)
    marine_smoke = load_json(integration / 'marineinst-real-weight-smoke.json', attachments)
    biota_audit = load_json(root / 'data/benchmarks/biota/biota_spatial_split_available_v1.audit.json', attachments)
    test_path = integration / 'sam31-verification-tests.txt'
    tests = None
    if test_path.is_file():
        data = test_path.read_bytes()
        attachments.append({'path': str(test_path), 'sha256': digest(data), 'bytes': len(data)})
        count = re.search(r'Ran (\d+) tests', data.decode(errors='replace'))
        tests = {'count': int(count.group(1)) if count else None,
                 'status': 'PASS' if data.rstrip().endswith(b'OK') else 'UNCONFIRMED', 'path': str(test_path)}

    now = datetime.now(timezone.utc)
    evidence = {'generated_utc': now.isoformat(), 'generated_pacific_auckland': now.astimezone(ZoneInfo('Pacific/Auckland')).isoformat(),
                'selection': 'Last CSV row per model condition/dataset in the documented source-file order; no maxima or averages',
                'sources': sources, 'attachments': attachments,
                'explicit_selected_rows': list(selected.values()), 'legacy_selected_rows': list(legacy.values()),
                'discarded_or_excluded_rows': excluded, 'replaced_duplicate_rows': replacements,
                'new_model_completeness': completeness,
                'verification_tests': tests, 'sam31_real_weight_smoke': smoke,
                'sam31_conversion_summary': {k: v for k, v in (conversion or {}).items() if k != 'differences'},
                'native_parity_summary': {'status': (parity or {}).get('status'),
                                          'native_commit': (parity or {}).get('native_commit'),
                                          'notes': (parity or {}).get('notes'),
                                          'model_cases': len((parity or {}).get('models', []))},
                'approved_downloads': downloads, 'marineinst_real_weight_smoke': marine_smoke,
                'biota_availability_audit': biota_audit,
                'raw_csvs_modified': False}
    report_path = art / 'SeaDino-SAM31-MarineInst-results-2026-10-14.md'
    evidence_path = art / 'SeaDino-SAM31-MarineInst-results-2026-10-14.evidence.json'
    live_path = root / 'results/encoder_baseline_report.md'
    lines = ['# SeaDino encoder benchmark comparison', '',
             f"Generated {evidence['generated_pacific_auckland']} (Pacific/Auckland).",
             f"MarineInst: **{completeness['marineinst']['completed']}/8 complete**. SAM3.1: **{completeness['sam31']['completed']}/8 complete**.", '',
             'Classification cells show **linear / kNN macro-F1 (%)**; Coralscapes shows **mIoU (%)**. Each saved row is an explicit-v1 TEST score after validation selected the probe epoch. Pending cells contain no measured result.', '',
             '| Model condition | Substrate | German Bank | FathomNet | Brackish | Coralscapes |',
             '|---|---:|---:|---:|---:|---:|']
    def representative(key, rows):
        return next(iter(rows.values()), {'Model ID': key[0], 'Model / Label': key[1]})
    name_counts = {}
    for key, rows in conditions.items():
        name = condition_name(representative(key, rows)); name_counts[name] = name_counts.get(name, 0) + 1
    ordered = sorted(conditions.items(), key=lambda entry: ordering(representative(*entry)))
    for key, rows in ordered:
        row = representative(key, rows)
        name = condition_name(row)
        if name_counts[name] > 1:
            name += f" ({row.get('Normalization', '?')}; seed {row.get('Seed', '?')}; revision {row.get('Model Revision', '?')[:8]})"
        cells = [pair(rows.get(dataset), 'Linear Macro-F1 (%)', 'kNN Macro-F1 (%)',
                      'pending' if family(row) in NEW else '—') for dataset in EXPLICIT[:-1]]
        coral = rows.get('coralscapes', {})
        cells.append(fmt(coral.get('mIoU (%)'), 'pending' if family(row) in NEW else '—'))
        lines.append('| ' + ' | '.join([name, *cells]) + ' |')
    lines += ['', 'Legacy and exploratory results are shown separately. DeepFish cells show linear / kNN macro-F1; Biota cells show linear / kNN **mAP**, because it is multi-label; CoralMask shows mIoU. Missing older-model results are marked —.', '',
              '| Model | DeepFish F1 | CoralMask mIoU | Biota mAP |', '|---|---:|---:|---:|']
    for kind, name in (('original_s', 'Original DINOv3-S'), ('seadino_s_physics', 'SeaDino-S marine physics checkpoint 14'),
                       ('original_b', 'Original DINOv3-B'), ('seadino_b10', 'SeaDino-B checkpoint 10'),
                       ('dinov2_s', 'DINOv2-S'), ('dinov2_b', 'DINOv2-B'), ('clip', 'CLIP-B'),
                       ('marineinst', 'MarineInst ViT-H encoder'), ('sam31', 'SAM3.1 ViT encoder')):
        missing = 'pending' if kind in NEW else '—'
        deep = pair(legacy.get((kind, 'deepfish')), 'DeepFish Lin F1-Macro (%)', 'DeepFish kNN F1-Macro (%)', missing)
        coral = fmt(legacy.get((kind, 'coralmask'), {}).get('CoralMask mIoU (%)'), missing)
        biota = pair(legacy.get((kind, 'biota')), 'Biota Lin mAP (%)', 'Biota kNN mAP (%)', missing)
        lines.append('| ' + ' | '.join((name, deep, coral, biota)) + ' |')
    lines += ['', 'These comparisons use frozen image encoders with the existing probes. SAM3.1 and MarineInst use mean-pooled patch features for classification and spatial features for segmentation. Their prompted mask decoders are not evaluated by these scores.', '',
              '- The native SAM3.1 release uses a 1008-pixel image input. These probes use matched 224×224 classification inputs and 448×896 Coralscapes inputs. The SAM3.1 adapter pads 512×512 CoralMask inputs to 518×518 for its 14-pixel patch grid; its native-resolution prompted performance is a different evaluation.',
              '- MarineInst Stage 1 uses a ViT-H encoder and native 1024-pixel inputs. Its published training sources include **Brackish, DeepFish and FathomNet**; exact overlap with these benchmark records is unresolved.',
              '- **Biota availability-v1** removes exactly one unavailable TRAIN record, preserving every other original record and the entire original holdout. It has 72,248 train / 28,525 test samples and the unchanged 272-class taxonomy. The original manifest remains intact; no thumbnail or neighboring image was substituted.',
              '- **CoralMask remains legacy and test-selected**: its reported best mIoU was selected using test scores across probe epochs. It is exploratory, not an explicit-v1 final test score. DeepFish and Biota also retain their legacy split/evaluation implementations.',
              '- **CLIP Coralscapes used batch 4; the other completed conditions used batch 16.** This changes optimizer updates per epoch, so that comparison lacks a matched training budget.',
              '- SeaDino-B uses BenthicNet normalization while original DINOv3-B uses ImageNet normalization. Treat adaptation differences as model-plus-normalization comparisons until controlled reruns exist.',
              '- Results are from seed 42. These inspected test results are exploratory; checkpoint and setting selection for subsequent experiments should use validation. The tables do not establish statistical significance or prove SSL/test content independence.',
              '- BioCLIP is excluded because its existing wrapper swaps batch/token axes. Duplicate rows are replaced by the last saved row per condition/dataset; repeated rows are not additional seeds.', '']
    if tests and tests['status'] == 'PASS':
        lines += [f"Integration verification: **{tests['count']} tests passed**, as recorded in {link(test_path)}.", '']
    if smoke and conversion:
        identity = smoke.get('identity', {})
        lines += [f"SAM3.1's frozen ViT has **{smoke.get('parameters', 0):,} parameters** ({smoke.get('parameters', 0)/1e6:.2f}M). Its official native checkpoint was converted by explicit tensor mapping; all {conversion.get('keys')} encoder tensors loaded. LayerNorm epsilon is **{identity.get('layer_norm_eps')}**, matching native SAM3.1. This is the encoder count, not the entire prompted SAM3.1 model.", '',
                  f"SAM3.1 model: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), pinned revision `{identity.get('revision')}`, encoder SHA256 `{identity.get('encoder_sha256')}`.", '']
    if downloads:
        for item in downloads:
            if item.get('model') == 'facebook/sam3.1':
                lines += [f"Approved official weights: `{item['filename']}`, SHA256 `{item['sha256']}`; {link(item['path'], 'local checkpoint')}.", '']
    if parity and parity.get('status') == 'PASS':
        lines += [f"Official native/Hugging Face adapter parity checks passed on {len(parity.get('models', []))} reduced model cases and their rotary-position cases. Native source commit `{parity.get('native_commit')}`. Complete model parity used BF16 autocast because the current native fused MLP computes its first layer and GELU in BF16. Rectangular cases explicitly rebuilt native rotary positions; this is reference parity, not evidence that Meta validated these benchmark shapes. See {link(integration/'native31-parity/native31-parity-results.json') }.", '']
    if marine_smoke:
        identity = marine_smoke.get('identity', {})
        lines += [f"MarineInst checkpoint: {link(identity.get('checkpoint_path', ''), 'marineinst_vit-h_stage1.pth')}; SHA256 `{identity.get('checkpoint_sha256')}`; code revision `{identity.get('code_revision')}`.", '']
    lines += [f"{link(evidence_path, 'Machine-readable evidence')} contains every selected raw row, run ID, manifest hash, normalization, checkpoint/revision, probe budget, source-file SHA256, and discarded duplicate/exclusion record.", '',
              'Raw result sources, preserved unchanged:', '']
    for item in sources:
        status = f"{item['row_count']} rows; SHA256 `{item['sha256']}`" if item['exists'] else 'pending; file absent'
        lines.append(f"- {link(item['path'])}: {status}.")
    lines += ['', 'Regenerate after the queue completes:', '',
              '```bash', 'python scripts/summarize_encoder_results.py --require_complete', '```', '',
              '`--require_complete` still writes this report when incomplete, then exits 2 until both new models have scores for all eight benchmarks.', '']
    art.mkdir(parents=True, exist_ok=True)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines)
    report_path.write_text(report, encoding='utf-8')
    live_path.write_text(report, encoding='utf-8')
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    complete = all(value['complete'] for value in completeness.values())
    print(json.dumps({'complete': complete, 'progress': completeness, 'report': str(report_path),
                      'live_report': str(live_path), 'evidence': str(evidence_path)}, indent=2))
    return 2 if args.require_complete and not complete else 0


if __name__ == '__main__':
    sys.exit(main())
