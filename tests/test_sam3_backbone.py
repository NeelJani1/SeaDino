"""Offline SAM 3 adapter tests; tiny random models are never benchmark results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from transformers import Sam3Config, Sam3ViTConfig, Sam3ViTModel, Sam3VisionConfig

from eval_suite.models import BASELINE_MAP, LinearSegmenter, load_baseline_backbone
from eval_suite.sam3_backbone import (SAM3BackboneWrapper, _read_encoder_weights,
                                     load_sam3_backbone)


def tiny_encoder(size=32):
    config = Sam3ViTConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=2, patch_size=2, image_size=size,
                          pretrain_image_size=16, window_size=4, global_attn_indexes=[1])
    config._attn_implementation = 'sdpa'
    return Sam3ViTModel(config).eval()


class SAM3AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.encoder = tiny_encoder()
        self.adapter = SAM3BackboneWrapper(self.encoder, microbatch_size=2).eval()

    def test_mean_pool_and_patch_tokens_match_native_encoder(self):
        images = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = self.encoder(images).last_hidden_state
        actual = self.adapter(images).last_hidden_state
        self.assertEqual(actual.shape, (3, 257, 16))
        torch.testing.assert_close(actual[:, 1:], expected)
        torch.testing.assert_close(actual[:, 0], expected.mean(1))
        self.assertGreater((actual[0, 0] - actual[1, 0]).abs().max().item(), 1e-5)

    def test_rectangular_and_changing_resolutions_match_independent_row_major_reference(self):
        state = copy.deepcopy(self.encoder.state_dict())
        for size in ((16, 32), (18, 18), (32, 32), (32, 16)):
            config = copy.deepcopy(self.encoder.config)
            config.image_size = list(size)
            native = Sam3ViTModel(config).eval()
            native.load_state_dict(state, strict=True)
            # Independent meshgrid reference. Reusing the HF rectangular
            # constructor alone would repeat its height-as-row-stride error.
            for layer in native.layers:
                h, w = (layer.window_size, layer.window_size) if layer.window_size else (size[0]//2, size[1]//2)
                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                dim = config.hidden_size // config.num_attention_heads
                omega = 1. / (config.rope_theta ** (torch.arange(0, dim, 4).float() / dim))
                scale = config.window_size / h
                phase = torch.cat((xx.flatten()[:, None]*omega*scale,
                                   yy.flatten()[:, None]*omega*scale), dim=-1).repeat_interleave(2, dim=-1)
                layer.rotary_emb.rope_embeddings_cos.copy_(phase.cos())
                layer.rotary_emb.rope_embeddings_sin.copy_(phase.sin())
            images = torch.randn(2, 3, *size)
            with torch.no_grad():
                expected = native(images).last_hidden_state
            actual = self.adapter(images).last_hidden_state[:, 1:]
            torch.testing.assert_close(actual, expected)

    def test_rectangular_rotary_coordinates_follow_image_rows_and_columns(self):
        self.adapter._set_rotary_grid(6, 10, torch.device('cpu'))
        rotary = self.encoder.layers[1].rotary_emb
        self.assertEqual((rotary.end_x, rotary.end_y), (5, 3))
        cos, sin = rotary()
        half = cos.shape[-1] // 2
        # One horizontal step changes x only; one complete image row changes y only.
        torch.testing.assert_close(cos[0, half:], cos[1, half:], atol=0, rtol=0)
        torch.testing.assert_close(cos[0, :half], cos[5, :half], atol=0, rtol=0)
        self.assertFalse(torch.equal(sin[0, :half], sin[1, :half]))
        self.assertFalse(torch.equal(sin[0, half:], sin[5, half:]))

    def test_features_are_independent_of_batch_companions_and_microbatch_size(self):
        images = torch.randn(5, 3, 32, 32)
        expected = self.adapter(images).last_hidden_state
        for microbatch in (1, 3, 5):
            self.adapter.microbatch_size = microbatch
            actual = self.adapter(images.flip(0)).last_hidden_state.flip(0)
            torch.testing.assert_close(actual, expected)

    def test_segmentation_grid_and_head_gradients_with_padding(self):
        model = LinearSegmenter(self.adapter, feat_dim=16, num_classes=3)
        images = torch.randn(2, 3, 17, 31)
        labels = torch.randint(3, (2, 17, 31))
        logits = model(images, out_size=(17, 31))
        self.assertEqual(logits.shape, (2, 3, 17, 31))
        torch.nn.functional.cross_entropy(logits, labels).backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_classifier_and_multilabel_heads_can_train(self):
        images = torch.randn(4, 3, 32, 32, requires_grad=True)
        features = self.adapter(images).last_hidden_state[:, 0]
        self.assertFalse(features.requires_grad)
        for criterion, targets in (
            (torch.nn.CrossEntropyLoss(), torch.tensor([0, 1, 0, 1])),
            (torch.nn.BCEWithLogitsLoss(), torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])),
        ):
            head = torch.nn.Linear(16, 2)
            loss = criterion(head(features), targets)
            loss.backward()
            self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_invalid_shapes_and_attention_requests_fail_clearly(self):
        for shape in ((0, 3, 32, 32), (1, 3, 17, 32), (1, 1, 32, 32)):
            with self.assertRaises(ValueError):
                self.adapter(torch.randn(*shape))
        with self.assertRaises(NotImplementedError):
            self.adapter(torch.randn(1, 3, 32, 32), output_attentions=True)
        with self.assertRaises(ValueError):
            SAM3BackboneWrapper(self.encoder, microbatch_size=0)

    def _export(self, root, prefix, video=False):
        vision = Sam3VisionConfig(backbone_config=self.encoder.config.to_dict())
        config = Sam3Config(vision_config=vision.to_dict())
        if video:
            from transformers import Sam3VideoConfig
            config = Sam3VideoConfig(detector_config=config.to_dict())
        config.save_pretrained(root)
        weights = {prefix + k: v.contiguous() for k, v in self.encoder.state_dict().items()}
        if prefix:
            weights['unused_decoder.weight'] = torch.ones(2, 2)
        save_file(weights, str(Path(root) / 'model.safetensors'))

    def test_image_and_video_hf_exports_load_only_exact_encoder_weights(self):
        for prefix, video in [('vision_encoder.backbone.', False),
                              ('detector_model.vision_encoder.backbone.', True)]:
            with tempfile.TemporaryDirectory() as directory:
                self._export(directory, prefix, video)
                loaded, *_ = load_sam3_backbone(torch.device('cpu'), torch.float32,
                                                model_path=directory)
                images = torch.randn(2, 3, 16, 32)
                torch.testing.assert_close(loaded(images).last_hidden_state,
                                           self.adapter(images).last_hidden_state)
                self.assertTrue(all(not p.is_meta for p in loaded.parameters()))
                self.assertTrue(all(not b.is_meta for b in loaded.buffers()))
                self.assertEqual(loaded.norm_mean, [.5, .5, .5])
                self.assertEqual(len(loaded.benchmark_identity['encoder_sha256']), 64)

    def test_missing_encoder_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._export(directory, 'vision_encoder.backbone.')
            weights = {f'vision_encoder.backbone.{k}': v for k, v in self.encoder.state_dict().items()
                       if k != 'layer_norm.weight'}
            save_file(weights, str(Path(directory) / 'model.safetensors'))
            with self.assertRaises(RuntimeError):
                load_sam3_backbone(torch.device('cpu'), torch.float32, model_path=directory)

    def test_unrecognized_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'model.safetensors'
            save_file({'unrelated': torch.ones(2, 2)}, str(filename))
            with self.assertRaises(ValueError):
                _read_encoder_weights(filename)

    def test_alias_loader_and_normalization(self):
        self.assertEqual(BASELINE_MAP['sam3'], 'facebook/sam3')
        self.assertEqual(BASELINE_MAP['sam3-encoder'], 'facebook/sam3')
        with tempfile.TemporaryDirectory() as directory:
            self._export(directory, 'vision_encoder.backbone.')
            loaded, *_ = load_baseline_backbone('sam3', torch.device('cpu'), torch.float32,
                                                sam3_model_path=directory, sam3_microbatch_size=1)
            self.assertEqual(loaded.microbatch_size, 1)
            self.assertEqual(loaded.norm_label, 'SAM3')


if __name__ == '__main__':
    unittest.main()
