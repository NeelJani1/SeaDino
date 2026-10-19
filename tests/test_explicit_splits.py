import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eval_suite.split_protocol import (SplitBundle, _row, _spatial_guard, assert_disjoint, grouped_partition,
                                     save_manifest, video_groups)
from eval_suite.benchmarks.explicit_classification import fit_linear_probe, classification_metrics
from eval_suite.benchmarks.explicit_coralscapes import evaluate_explicit_coralscapes
from eval_suite.explicit_evaluation import FIELDS, class_names_from_id_map, load_completed, write_result


class SplitTests(unittest.TestCase):
    def test_segmentation_class_ids_are_not_shifted_when_zero_is_absent(self):
        self.assertEqual(class_names_from_id_map({'1': 'coral', '3': 'sand'}),
                         ['unmapped_id_0', 'coral', 'unmapped_id_2', 'sand'])
    def records(self):
        return [_row(f'/images/g{group}-{i}.jpg', group % 3, f'group-{group}', 'train')
                for group in range(30) for i in range(3)]

    def test_group_split_is_order_independent_and_nonoverlapping(self):
        a, b = self.records(), list(reversed(self.records()))
        grouped_partition(a)
        grouped_partition(b)
        self.assertEqual({r['sample_id']: r['partition'] for r in a}, {r['sample_id']: r['partition'] for r in b})
        assert_disjoint(a)
        for part in ['train', 'validation', 'test']:
            self.assertEqual({r['label'] for r in a if r['partition'] == part}, {0, 1, 2})

    def test_overlap_detection_uses_images_not_just_object_labels(self):
        rows = [_row('/same.jpg', 0, 'g1', 'train'), _row('/same.jpg', 1, 'g2', 'test')]
        with self.assertRaisesRegex(ValueError, 'image crosses'):
            assert_disjoint(rows)

    def test_neighbouring_and_overlapping_clips_stay_together(self):
        names = ['2019-03-21_03-26-50to2019-03-21_03-27-04_1-1.jpg',
                 '2019-03-21_03-27-02to2019-03-21_03-27-17_1-1.jpg',
                 '2019-03-21_05-01-24to2019-03-21_05-01-31_1-1.jpg']
        groups = video_groups(names)
        self.assertEqual(groups[names[0]], groups[names[1]])
        self.assertNotEqual(groups[names[0]], groups[names[2]])

    def test_manifest_is_immutable_and_tampering_is_detected(self):
        rows = self.records()
        grouped_partition(rows)
        with tempfile.TemporaryDirectory() as tmp:
            make = lambda: SplitBundle('fixture', copy.deepcopy(rows), ['a', 'b', 'c'], {'grouping': 'fixture'})
            first = save_manifest(make(), tmp, 42)
            second = save_manifest(make(), tmp, 42)
            self.assertEqual(first.metadata['manifest_hash'], second.metadata['manifest_hash'])
            with Path(first.manifest).open('a') as f:
                f.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'Manifest/source changed'):
                save_manifest(make(), tmp, 42)

    def test_fixed_taxonomy_includes_absent_class(self):
        self.assertAlmostEqual(classification_metrics([0, 1], [0, 1], 3)['macro_f1'], 200 / 3)

    def test_validation_buffer_drops_near_train_and_retains_test(self):
        rows = [_row('/near.jpg', 0, 'near', 'train', latitude='-43', longitude='147.0001'),
                _row('/far.jpg', 0, 'far', 'train', latitude='-43', longitude='147.1'),
                _row('/val.jpg', 0, 'val', 'validation', latitude='-43', longitude='147'),
                _row('/test.jpg', 0, 'test', 'test', latitude='-43', longitude='147.2')]
        test_id = rows[-1]['sample_id']
        distances = _spatial_guard(rows, 100)
        self.assertEqual(rows[0]['partition'], 'dropped')
        self.assertEqual(rows[-1]['partition'], 'test')
        self.assertEqual(rows[-1]['sample_id'], test_id)
        self.assertTrue(all(d >= 100 for d in distances.values()))

    def test_csv_refuses_legacy_schema_and_resume_uses_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.csv'
            row = {k: '' for k in FIELDS}
            row['Run ID'] = 'validation-config'
            write_result(path, row)
            self.assertEqual(load_completed(path), {'validation-config'})
            row['Run ID'] = 'test-config'
            write_result(path, row)
            self.assertEqual(load_completed(path), {'validation-config', 'test-config'})
            path.write_text('Model / Label,Epoch\nold,1\n')
            with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                write_result(path, row)

    def test_linear_probe_selects_using_validation_only(self):
        torch.set_num_threads(1)
        train_f = torch.tensor([[1., 0.], [0., 1.], [1., .1], [.1, 1.]])
        train_y = torch.tensor([0, 1, 0, 1])
        head, epoch, metrics = fit_linear_probe(train_f, train_y, train_f, train_y, 2, torch.device('cpu'), 3, 42)
        self.assertGreaterEqual(epoch, 1)
        self.assertLessEqual(epoch, 3)
        self.assertEqual(metrics['macro_f1'], classification_metrics(train_y.numpy(), head(train_f).argmax(1).detach().numpy(), 2)['macro_f1'])

    def test_coralscapes_test_access_happens_after_validation_selection(self):
        from unittest.mock import patch
        import eval_suite.benchmarks.explicit_coralscapes as module
        import torch.nn as nn
        events = []
        class TinyData(torch.utils.data.Dataset):
            def __init__(self, split, **kwargs):
                self.split = split
                events.append('construct:' + split)
            def __len__(self): return 2
            def __getitem__(self, index): return torch.ones(1, 2, 2), torch.zeros(2, 2, dtype=torch.long)
        class TinySegmenter(nn.Module):
            def __init__(self, backbone, **kwargs):
                super().__init__()
                self.head = nn.Conv2d(1, 2, 1)
            def forward(self, x, out_size=None): return self.head(x)
        counter = {'validation': 0}
        selected_state = {}
        def score(model, loader, *args):
            part = loader.dataset.split
            events.append('score:' + part)
            if part == 'validation':
                counter[part] += 1
                value = 90 if counter[part] == 1 else 10
                if counter[part] == 1:
                    selected_state.update({k: v.detach().clone() for k, v in model.head.state_dict().items()})
                return {'miou': value, 'pixel_accuracy': value}
            self.assertEqual(counter['validation'], 2)
            for key, value in model.head.state_dict().items():
                self.assertTrue(torch.equal(value, selected_state[key]), 'Final test did not use the validation-selected state')
            return {'miou': 25, 'pixel_accuracy': 30}
        backbone = type('Backbone', (), {'config': type('Config', (), {'hidden_size': 1})()})()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'CoralscapesDataset', TinyData), patch.object(module, 'LinearSegmenter', TinySegmenter), patch.object(module, 'score_segmentation', score):
            result = evaluate_explicit_coralscapes(backbone, {p:p for p in ['train', 'validation', 'test']}, 2,
                torch.device('cpu'), torch.float32, 'test', 2, 2, 0, 42, [0], [1], str(Path(tmp)/'head.pt'), 'fixture', 2, 2)
            self.assertEqual(result['selected_probe_epoch'], 1)
            self.assertEqual(result['validation_miou'], 90)
            self.assertEqual(result['miou'], 25)
            self.assertGreater(events.index('construct:test'), events.index('score:validation'))
            self.assertEqual(events.count('score:test'), 1)


if __name__ == '__main__':
    unittest.main()
