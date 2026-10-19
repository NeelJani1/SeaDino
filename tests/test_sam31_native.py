"""Check native SAM 3.1 conversion independently of its downloadable checkpoint."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Sam3Config, Sam3VideoConfig, Sam3VisionConfig

from eval_suite.models import BASELINE_MAP, load_baseline_backbone
from eval_suite.sam3_backbone import _convert_native_encoder_weights, load_sam3_backbone
from test_sam3_backbone import tiny_encoder


def native_export(encoder):
    """Represent a tiny model in the documented Meta checkpoint layout."""
    native={}
    prefix='detector.backbone.vision_backbone.trunk.'
    state=encoder.state_dict()
    native[prefix+'pos_embed']=torch.cat((torch.randn(1,1,encoder.config.hidden_size),
                                          state['embeddings.position_embeddings']),dim=1)
    native[prefix+'patch_embed.proj.weight']=state['embeddings.patch_embeddings.projection.weight']
    for suffix in ('weight','bias'):native[prefix+'ln_pre.'+suffix]=state['layer_norm.'+suffix]
    for index,layer in enumerate(encoder.layers):
        a=f'layers.{index}.'
        b=prefix+f'blocks.{index}.'
        for suffix in ('weight','bias'):
            native[b+'attn.qkv.'+suffix]=torch.cat([state[a+'attention.'+part+'_proj.'+suffix] for part in ('q','k','v')],dim=0)
            for old,new in [('layer_norm1','norm1'),('layer_norm2','norm2'),
                            ('attention.o_proj','attn.proj'),('mlp.fc1','mlp.fc1'),('mlp.fc2','mlp.fc2')]:
                native[b+new+'.'+suffix]=state[a+old+'.'+suffix]
        native[b+'attn.freqs_cis']=torch.complex(layer.rotary_emb.rope_embeddings_cos[:,::2],
                                              layer.rotary_emb.rope_embeddings_sin[:,::2])
    return native


class NativeSAM31Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.encoder=tiny_encoder()
        self.native=native_export(self.encoder)

    def test_all_learned_tensors_are_preserved_and_cls_position_is_removed(self):
        converted=_convert_native_encoder_weights(self.native,self.encoder.config)
        expected=self.encoder.state_dict()
        self.assertEqual(set(converted),set(expected))
        for key in expected:torch.testing.assert_close(converted[key],expected[key],rtol=0,atol=0)

    def test_qkv_order_is_checked_and_unknown_tensors_fail(self):
        prefix='detector.backbone.vision_backbone.trunk.'
        state=dict(self.native)
        state[prefix+'blocks.0.attn.qkv.bias']=torch.arange(48).float()
        result=_convert_native_encoder_weights(state,self.encoder.config)
        for i,part in enumerate(('q','k','v')):
            torch.testing.assert_close(result[f'layers.0.attention.{part}_proj.bias'],torch.arange(i*16,(i+1)*16).float())
        state[prefix+'blocks.0.unknown_weight']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'Unknown native'):
            _convert_native_encoder_weights(state,self.encoder.config)

    def test_invalid_rotary_geometry_fails(self):
        state=dict(self.native)
        state['detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis']=torch.ones(1,1,dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError,'RoPE layout'):
            _convert_native_encoder_weights(state,self.encoder.config)

    def test_native_export_loads_strictly_with_native_epsilon_and_release_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            vision=Sam3VisionConfig(backbone_config=self.encoder.config.to_dict())
            config=Sam3VideoConfig(detector_config=Sam3Config(vision_config=vision.to_dict()).to_dict())
            config.save_pretrained(directory)
            torch.save(self.native,Path(directory)/'sam3.1_multiplex.pt')
            loaded,*_=load_baseline_backbone('sam3.1-encoder',torch.device('cpu'),torch.float32,
                                            sam3_model_path=directory)
            self.assertEqual(loaded.norm_label,'SAM3.1')
            self.assertEqual(loaded.config.layer_norm_eps,1e-5)
            self.assertEqual(loaded.benchmark_identity['model_id'],'facebook/sam3.1')
            self.assertEqual(loaded.benchmark_identity['weight_format'],'.pt')
            inputs=torch.randn(2,3,32,32)
            actual=loaded(inputs).last_hidden_state
            self.assertEqual(actual.shape,(2,257,16))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(all(not p.is_meta for p in loaded.parameters()))
        self.assertEqual(BASELINE_MAP['sam3.1'],'facebook/sam3.1')


if __name__=='__main__':unittest.main()
