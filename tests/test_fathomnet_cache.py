import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from eval_suite.fathomnet_cache import FathomNetCache, make_fathomnet_cache
from eval_suite.datasets.fathomnet import FathomNetDataset
from eval_suite.benchmarks.explicit_classification import StrictImages, extract_features, evaluate_explicit_classification
from eval_suite.benchmarks.fathomnet_probe import evaluate_fathomnet


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 5)
        self.calls = 0

    def forward(self, images, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.projection(images.mean(dim=(2, 3))).unsqueeze(1))


class FathomNetCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.samples = []
        rng = np.random.default_rng(42)
        for i in range(12):
            path = self.root / f'{i}.png'
            Image.fromarray(rng.integers(0, 256, (29, 37, 3), dtype=np.uint8)).save(path)
            self.samples.append(dict(image=str(path), label=i % 2, bbox=[3, 2, 15, 19],
                                     partition='train', sample_id=str(i)))
        self.transform = T.Compose([T.Resize((16, 16), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                                    T.ToTensor(), T.Normalize([.359, .413, .386], [.219, .215, .209])])
        self.cache_dir = self.root / 'cache'

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, samples=None, transform=None):
        return FathomNetCache(samples or self.samples, transform or self.transform, self.cache_dir)

    def test_full_frame_and_crop_pixels_are_bit_identical_and_hit_avoids_decode(self):
        cache = self.cache()
        for crop in [False, True]:
            original = FathomNetDataset(self.samples, transform=self.transform, crop_boxes=crop)
            cached = FathomNetDataset(self.samples, transform=self.transform, crop_boxes=crop, cache=cache)
            expected, label = original[0]
            actual, actual_label = cached[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(label, actual_label)
            with patch.object(cached, '_load_uncached_image', side_effect=AssertionError('Cache hit decoded source')):
                self.assertTrue(torch.equal(expected, cached[0][0]))

    def test_pixel_cache_is_shared_across_normalizations_but_not_crops(self):
        cached = FathomNetDataset(self.samples, self.transform, cache=self.cache())
        cached[0]
        other = T.Compose([self.transform.transforms[0], T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
        second = FathomNetDataset(self.samples, other, cache=self.cache(transform=other))
        with patch.object(second, '_load_uncached_image', side_effect=AssertionError('Normalization reread image')):
            actual = second[0][0]
        self.assertTrue(torch.equal(actual, FathomNetDataset(self.samples, other)[0][0]))
        cache = self.cache()
        self.assertNotEqual(cache.pixel_path(self.samples[0], False, .05), cache.pixel_path(self.samples[0], True, .05))

    def test_source_change_invalidates_pixels_and_features(self):
        cache = self.cache()
        model = TinyBackbone()
        old_pixel = cache.pixel_path(self.samples[0], False, .05)
        old_feature = cache.feature_path(model, self.samples, False, torch.device('cpu'), torch.float32, 4, 'explicit')
        time.sleep(.002)
        Image.new('RGB', (37, 29), (255, 0, 0)).save(self.samples[0]['image'])
        changed = self.cache()
        self.assertNotEqual(old_pixel, changed.pixel_path(self.samples[0], False, .05))
        self.assertNotEqual(old_feature, changed.feature_path(model, self.samples, False, torch.device('cpu'), torch.float32, 4, 'explicit'))

    def test_feature_key_changes_for_weights_labels_order_norm_precision_crop_and_batch(self):
        cache, model = self.cache(), TinyBackbone()
        key = lambda c=cache, rows=self.samples, crop=False, dtype=torch.float32, batch=4: c.feature_path(model, rows, crop, torch.device('cpu'), dtype, batch, 'explicit')
        original = key()
        self.assertNotEqual(original, key(rows=list(reversed(self.samples))))
        changed = [dict(r) for r in self.samples]
        changed[0]['label'] = 1
        self.assertNotEqual(original, key(rows=changed))
        self.assertNotEqual(original, key(crop=True))
        self.assertNotEqual(original, key(dtype=torch.bfloat16))
        self.assertNotEqual(original, key(batch=3))
        other = T.Compose([self.transform.transforms[0], T.ToTensor(), T.Normalize([0, 0, 0], [1, 1, 1])])
        self.assertNotEqual(original, key(c=self.cache(transform=other)))
        with torch.no_grad():
            model.projection.weight.add_(.1)
        self.assertNotEqual(original, key())

    def test_corrupt_pixels_and_features_are_rebuilt(self):
        cache = self.cache()
        dataset = FathomNetDataset(self.samples, self.transform, cache=cache)
        expected = dataset[0][0]
        cache.pixel_path(self.samples[0], False, .05).write_bytes(b'incomplete')
        self.assertTrue(torch.equal(dataset[0][0], expected))
        model = TinyBackbone()
        path = cache.feature_path(model, self.samples, False, torch.device('cpu'), torch.float32, 4, 'explicit')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'incomplete')
        self.assertIsNone(cache.load_features(path, self.samples, 'fixture'))

    def test_feature_hit_is_identical_and_avoids_model_forward(self):
        cache, model = self.cache(), TinyBackbone()
        dataset = StrictImages(self.samples, self.transform, 'fathomnet', cache=cache)
        args = (model, dataset, torch.device('cpu'), torch.float32, 4, 0, 42, 'fixture')
        first = extract_features(*args, cache=cache)
        calls = model.calls
        with patch.object(model, 'forward', side_effect=AssertionError('Feature hit ran backbone')):
            second = extract_features(*args, cache=cache)
        self.assertEqual(model.calls, calls)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        uncached = extract_features(model, StrictImages(self.samples, self.transform, 'fathomnet'),
                                    torch.device('cpu'), torch.float32, 4, 0, 42, 'uncached')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, uncached)))

    def test_legacy_metrics_match_cache_disabled_and_second_run_skips_backbone(self):
        model = TinyBackbone()
        args = (model, self.samples[:8], self.samples[8:], 2, self.transform, torch.device('cpu'), torch.float32)
        kwargs = dict(batch_size=4, num_workers=0, epochs=2)
        plain = evaluate_fathomnet(*args, **kwargs, cache_enabled=False)
        first = evaluate_fathomnet(*args, **kwargs, cache_dir=str(self.cache_dir))
        with patch.object(model, 'forward', side_effect=AssertionError('Cached legacy run ran backbone')):
            second = evaluate_fathomnet(*args, **kwargs, cache_dir=str(self.cache_dir))
        for key in plain:
            np.testing.assert_array_equal(plain[key], first[key])
            np.testing.assert_array_equal(plain[key], second[key])

    def test_explicit_validation_then_test_reuses_only_selected_partition_features(self):
        parts = {'train': self.samples[:6], 'validation': self.samples[6:9], 'test': self.samples[9:]}
        for part, rows in parts.items():
            for row in rows:
                row['partition'] = part
        bundle = SimpleNamespace(name='fathomnet', class_names=['a', 'b'], records=lambda p: parts[p], metadata={'manifest_hash': 'fixture'})
        model = TinyBackbone()
        args = (model, bundle, self.transform, torch.device('cpu'), torch.float32)
        def run(part, caching):
            return evaluate_explicit_classification(*args, part, 2, 3, 0, 42,
                str(self.root / (part + '.pt')), cache_enabled=caching, cache_dir=str(self.cache_dir))
        val = run('validation', True)
        calls = model.calls
        final = run('test', True)
        self.assertEqual(model.calls - calls, 1, 'Train/validation features were unnecessarily reextracted')
        plain = run('test', False)
        self.assertEqual(final, plain)
        self.assertEqual(val['selected_probe_epoch'], final['selected_probe_epoch'])

    def test_random_transform_disables_caching(self):
        transform = T.Compose([T.RandomResizedCrop(16), T.ToTensor(), T.Normalize([0]*3, [1]*3)])
        self.assertIsNone(make_fathomnet_cache(self.samples, transform, self.cache_dir))


if __name__ == '__main__':
    unittest.main()
