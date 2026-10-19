import unittest

import numpy as np
import torch

from feature_visualization.attention import color_attention, dense_map, feature_parity, marine_attention_weights, sam_attention_weights, summarize_cls, summarize_received, validate_attention


class AttentionTests(unittest.TestCase):
    def test_patch_vector_parity_accepts_roundoff_but_rejects_encoder_changes(self):
        ref = np.ones((8, 768), dtype=np.float32)
        actual = ref.copy()
        actual[0, 0] += .003
        self.assertTrue(feature_parity(actual, ref)["passed"])
        with self.assertRaisesRegex(ValueError, "differs"):
            feature_parity(ref * 1.01, ref)
        with self.assertRaisesRegex(ValueError, "differs"):
            bad = ref.copy()
            bad[0] *= 1.001
            feature_parity(bad, ref)

    def test_probability_validation_rejects_fabricated_or_invalid_weights(self):
        weights = torch.ones(1, 2, 5, 5) / 5
        self.assertLess(validate_attention(weights), 1e-6)
        with self.assertRaises(ValueError):
            validate_attention(weights * 2)
        with self.assertRaises(ValueError):
            validate_attention(torch.tensor([float("nan")]).reshape(1, 1, 1, 1))

    def test_cls_spatial_suffix_and_rollout_keep_register_paths(self):
        # Token 0=CLS, token 1=register, tokens 2/3=patches.
        first = torch.eye(4).reshape(1, 1, 4, 4)
        first[0, 0, 1] = torch.tensor([0., 0., .8, .2])
        second = torch.eye(4).reshape(1, 1, 4, 4)
        second[0, 0, 0] = torch.tensor([0., 1., 0., 0.])
        arrays, meta = summarize_cls([first, second], 2)
        np.testing.assert_array_equal(arrays["last"], [0, 0])
        np.testing.assert_allclose(arrays["rollout"], [.2, .05], atol=1e-7)
        self.assertEqual(arrays["heads"].shape, (1, 2))
        self.assertTrue(meta["registers_retained_in_rollout"])

    def test_received_patch_attention_averages_queries_not_keys(self):
        weights = torch.tensor([[[[.9, .1], [.7, .3]]]])
        arrays, meta = summarize_received(weights, 2, 31, "native")
        np.testing.assert_allclose(arrays["last"], [.8, .2])
        self.assertNotIn("rollout", arrays)
        self.assertIn("No learned CLS", meta["rollout_unavailable"])

    def test_padding_is_removed_from_dense_attention_display(self):
        geometry = dict(grid_h=2, grid_w=3, padded_height=28, padded_width=42, height=23, width=39)
        output = dense_map(np.ones(6, dtype=np.float32), geometry)
        self.assertEqual(output.shape, (23, 39))
        np.testing.assert_allclose(output, 1, atol=2e-7)

    def test_uniform_attention_does_not_gain_false_interpolation_contrast(self):
        geometry = dict(grid_h=2, grid_w=3, padded_height=28, padded_width=42, height=23, width=39)
        rgb, metadata = color_attention(np.ones(6, dtype=np.float32), geometry)
        self.assertTrue(metadata["constant"])
        self.assertEqual(len(np.unique(rgb.reshape(-1, 3), axis=0)), 1)

    def test_marine_relative_position_attention_reconstructs_native_output(self):
        from eval_suite.vendor.marineinst.image_encoder import Attention
        torch.manual_seed(42)
        module = Attention(dim=8, num_heads=2, use_rel_pos=True, input_size=(2, 3)).eval()
        with torch.no_grad():
            module.rel_pos_h.normal_(0, .1)
            module.rel_pos_w.normal_(0, .1)
            x = torch.randn(1, 2, 3, 8)
            weights = marine_attention_weights(module, x)
            self.assertLess(validate_attention(weights), 1e-6)
            qkv = module.qkv(x).reshape(1, 6, 3, 2, -1).permute(2, 0, 3, 1, 4)
            values = qkv[2]
            reconstructed = (weights @ values).view(1, 2, 2, 3, -1).permute(0, 2, 3, 1, 4).reshape(1, 2, 3, 8)
            reconstructed = module.proj(reconstructed)
            torch.testing.assert_close(reconstructed, module(x), atol=1e-7, rtol=1e-6)

    def test_sam_rope_attention_reconstructs_installed_native_output(self):
        from transformers.models.sam3.modeling_sam3 import Sam3ViTConfig, Sam3ViTRoPEAttention
        torch.manual_seed(42)
        config = Sam3ViTConfig(hidden_size=8, num_attention_heads=2, attention_dropout=0.)
        config._attn_implementation = "eager"
        module = Sam3ViTRoPEAttention(config).eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 3, 8)
            theta = torch.randn(6, 4)
            position = (theta.cos(), theta.sin())
            weights = sam_attention_weights(module, x, position)
            output, native_weights = module(x, position)
            torch.testing.assert_close(weights, native_weights, atol=0, rtol=0)
            values = module.v_proj(x).view(1, 6, 2, 4).transpose(1, 2)
            reconstructed = module.o_proj((weights @ values).transpose(1, 2).reshape(1, 2, 3, 8))
            torch.testing.assert_close(reconstructed, output, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
