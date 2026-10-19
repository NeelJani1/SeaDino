"""Offline parity and feature-interface tests, using a tiny random encoder."""
from functools import partial
import unittest

import torch
from torch import nn

from eval_suite.marineinst_backbone import MarineInstBackboneWrapper
from eval_suite.models import BASELINE_MAP, LinearSegmenter
from eval_suite.vendor.marineinst.image_encoder import ImageEncoderViT


class MarineInstTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.encoder = ImageEncoderViT(img_size=32, patch_size=2, embed_dim=16,
            depth=2, num_heads=2, mlp_ratio=2, out_chans=8, use_rel_pos=True,
            global_attn_indexes=[1], window_size=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
        self.encoder.pos_embed.data.normal_(std=.02)
        self.adapter = MarineInstBackboneWrapper(self.encoder, microbatch_size=2)

    def test_native_features_reproduce_upstream_encoder_output(self):
        images = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            native = self.encoder(images)
            adapted = self.adapter(images).last_hidden_state
            patches = adapted[:, 1:].transpose(1, 2).reshape(3, 16, 16, 16)
            reconstructed = self.encoder.neck(patches)
        torch.testing.assert_close(reconstructed, native)
        torch.testing.assert_close(adapted[:, 0], adapted[:, 1:].mean(1))

    def test_nonsquare_features_and_microbatch_independence(self):
        images = torch.randn(5, 3, 16, 30)
        reference = self.adapter(images).last_hidden_state
        self.assertEqual(reference.shape, (5, 121, 16))
        for size in (1, 3, 5):
            self.adapter.microbatch_size = size
            torch.testing.assert_close(self.adapter(images.flip(0)).last_hidden_state.flip(0), reference)
        self.assertGreater((reference[0, 0] - reference[1, 0]).abs().sum().item(), 1e-4)

    def test_padded_segmentation_head_trains_without_backbone_gradients(self):
        segmenter = LinearSegmenter(self.adapter, 16, 3)
        output = segmenter(torch.randn(2, 3, 17, 31), (17, 31))
        target = torch.randint(3, (2, 17, 31))
        nn.CrossEntropyLoss()(output, target).backward()
        self.assertTrue(torch.isfinite(segmenter.head.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_aliases_and_native_normalization(self):
        self.assertEqual(BASELINE_MAP['marineinst'], 'marineinst-vit-h-stage1')
        self.assertEqual(BASELINE_MAP['marineinst-encoder'], 'marineinst-vit-h-stage1')
        torch.testing.assert_close(torch.tensor(self.adapter.norm_mean),
                                   torch.tensor([123.675/255, 116.28/255, 103.53/255]))
        torch.testing.assert_close(torch.tensor(self.adapter.norm_std),
                                   torch.tensor([58.395/255, 57.12/255, 57.375/255]))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.adapter(torch.randn(1, 3, 17, 32))
        with self.assertRaises(ValueError):
            MarineInstBackboneWrapper(self.encoder, microbatch_size=0)


if __name__ == '__main__':
    unittest.main()
