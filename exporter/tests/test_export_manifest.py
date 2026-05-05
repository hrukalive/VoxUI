import unittest
from pathlib import Path

import numpy as np

from exporter.export_voxcpm import (
    build_manifest,
    partition_weights,
    resolve_quant_args,
    validate_required_tensors,
)


class ExportManifestTests(unittest.TestCase):
    def test_partition_uses_python_component_names(self):
        weights = {
            "base_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "residual_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_encoder.in_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_decoder.input_embed.weight": np.zeros((2, 2), dtype=np.float32),
            "fsq_layer.project_in.weight": np.zeros((2, 2), dtype=np.float32),
        }
        buckets = partition_weights(weights, None)
        self.assertIn("base_lm.gguf", buckets)
        self.assertIn("residual_lm.gguf", buckets)
        self.assertIn("feat_encoder.gguf", buckets)
        self.assertIn("feat_decoder.gguf", buckets)
        self.assertIn("projections.gguf", buckets)
        names = {name for name, _ in buckets["feat_encoder.gguf"]}
        self.assertIn("feat_encoder.in_proj.weight", names)

    def test_missing_required_tensor_is_hard_error(self):
        buckets = {
            "base_lm.gguf": [("base_lm.norm.weight", np.zeros(2, dtype=np.float32))],
        }
        with self.assertRaisesRegex(ValueError, "missing required tensor"):
            validate_required_tensors(buckets, variant="2.0")

    def test_q4_lm_profile_for_variant_2_keeps_vae_f32(self):
        quant_args = resolve_quant_args(
            variant="2.0",
            profile="q4-lm",
            quant_lm=None,
            quant_encoder=None,
            quant_dit=None,
            quant_vae=None,
        )
        self.assertEqual(
            quant_args,
            {
                "quant_lm": "q4",
                "quant_encoder": "fp16",
                "quant_dit": "fp16",
                "quant_vae": "f32",
            },
        )

    def test_q4_lm_profile_for_variant_15_uses_fp16_vae(self):
        quant_args = resolve_quant_args(
            variant="1.5",
            profile="q4-lm",
            quant_lm=None,
            quant_encoder=None,
            quant_dit=None,
            quant_vae=None,
        )
        self.assertEqual(
            quant_args,
            {
                "quant_lm": "q4",
                "quant_encoder": "fp16",
                "quant_dit": "fp16",
                "quant_vae": "fp16",
            },
        )

    def test_fp16_profile_for_variant_2_keeps_vae_f32(self):
        quant_args = resolve_quant_args(
            variant="2.0",
            profile="fp16",
            quant_lm=None,
            quant_encoder=None,
            quant_dit=None,
            quant_vae=None,
        )
        self.assertEqual(
            quant_args,
            {
                "quant_lm": "fp16",
                "quant_encoder": "fp16",
                "quant_dit": "fp16",
                "quant_vae": "f32",
            },
        )

    def test_explicit_component_quantization_overrides_profile_defaults(self):
        quant_args = resolve_quant_args(
            variant="2.0",
            profile="q4-lm",
            quant_lm=None,
            quant_encoder=None,
            quant_dit="q8",
            quant_vae="fp16",
        )
        self.assertEqual(quant_args["quant_lm"], "q4")
        self.assertEqual(quant_args["quant_encoder"], "fp16")
        self.assertEqual(quant_args["quant_dit"], "q8")
        self.assertEqual(quant_args["quant_vae"], "fp16")

    def test_manifest_records_component_files_and_special_tokens(self):
        config = {
            "architecture": "voxcpm2",
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0,
            "audio_vae_config": {
                "sample_rate": 16000,
                "out_sample_rate": 48000,
                "latent_dim": 64,
                "chunk_size": 20,
                "decode_chunk_size": 240,
                "encoder_rates": [2, 5, 8, 8],
                "decoder_rates": [8, 6, 5, 2, 2, 2],
            },
            "lm_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "kv_channels": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "use_mup": True,
                "scale_emb": 12,
                "scale_depth": 1.4,
            },
        }
        manifest = build_manifest(
            model_dir=Path("VoxCPM/models/VoxCPM2"),
            config=config,
            variant="2.0",
            source_weight_format="safetensors",
            component_quantization={"base_lm.gguf": "fp16"},
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["architecture"], "voxcpm2")
        self.assertEqual(manifest["special_tokens"]["audio_start"], 101)
        self.assertEqual(manifest["special_tokens"]["ref_audio_start"], 103)
        self.assertEqual(manifest["components"]["feat_decoder"], "feat_decoder.gguf")


if __name__ == "__main__":
    unittest.main()
