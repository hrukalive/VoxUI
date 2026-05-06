import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from exporter.export_voxcpm import (
    BASE_MODEL_FILE,
    partition_weights,
    resolve_quant_args,
    validate_required_tensors,
    export,
)


class RecordingWriter:
    instances = []

    def __init__(self):
        self.metadata = {}
        self.tensors = []
        self.path = None
        RecordingWriter.instances.append(self)

    def add_metadata(self, key, value, value_type=None):
        self.metadata[key] = value

    def add_tensor(self, name, data, shape, dtype):
        self.tensors.append((name, data, shape, dtype))

    def write(self, path):
        self.path = Path(path)
        self.path.write_bytes(b"recorded gguf")


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
        self.assertIn("base_lm", buckets)
        self.assertIn("residual_lm", buckets)
        self.assertIn("feat_encoder", buckets)
        self.assertIn("feat_decoder", buckets)
        self.assertIn("projections", buckets)
        names = {name for name, _ in buckets["feat_encoder"]}
        self.assertIn("feat_encoder.in_proj.weight", names)

    def test_missing_required_tensor_is_hard_error(self):
        buckets = {
            "base_lm": [("base_lm.norm.weight", np.zeros(2, dtype=np.float32))],
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

    def test_base_export_writes_single_model_gguf_without_manifest(self):
        main_weights = {
            "base_lm.norm.weight": np.zeros(2, dtype=np.float32),
            "base_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "residual_lm.norm.weight": np.zeros(2, dtype=np.float32),
            "residual_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_encoder.in_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_encoder.special_token": np.zeros(2, dtype=np.float32),
            "feat_decoder.input_embed.weight": np.zeros((2, 2), dtype=np.float32),
            "fsq_layer.project_in.weight": np.zeros((2, 2), dtype=np.float32),
            "enc_to_lm_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "lm_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "res_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "stop_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "stop_head.weight": np.zeros((2, 2), dtype=np.float32),
        }
        vae_weights = {"encoder.conv.weight": np.zeros((2, 2), dtype=np.float32)}
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

        with TemporaryDirectory() as model_tmp, TemporaryDirectory() as output_tmp:
            model_dir = Path(model_tmp)
            output_dir = Path(output_tmp)
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            RecordingWriter.instances = []
            with (
                patch("exporter.export_voxcpm.GGUFWriter", RecordingWriter),
                patch("exporter.export_voxcpm.load_weights", return_value=(main_weights, vae_weights, "safetensors")),
            ):
                summary = export(
                    model_dir,
                    output_dir,
                    {
                        "quant_lm": "q4",
                        "quant_encoder": "fp16",
                        "quant_dit": "q8",
                        "quant_vae": "f32",
                    },
                    "2.0",
                )

            self.assertEqual([writer.path.name for writer in RecordingWriter.instances], [BASE_MODEL_FILE])
            self.assertTrue((output_dir / BASE_MODEL_FILE).exists())
            self.assertTrue((output_dir / "tokenizer.json").exists())
            self.assertFalse((output_dir / "manifest.json").exists())

            writer = RecordingWriter.instances[0]
            self.assertEqual(writer.metadata["voxcpm.schema_version"], 2)
            self.assertEqual(writer.metadata["voxcpm.kind"], "base")
            self.assertEqual(writer.metadata["voxcpm.architecture"], "voxcpm2")
            self.assertEqual(writer.metadata["voxcpm.variant"], "2.0")
            self.assertEqual(writer.metadata["voxcpm.quant_profile"], "manual")
            self.assertEqual(writer.metadata["voxcpm.source_weight_format"], "safetensors")
            self.assertEqual(writer.metadata["voxcpm.quantization.base_lm"], "q4")
            self.assertEqual(writer.metadata["voxcpm.quantization.feat_encoder"], "fp16")
            self.assertEqual(writer.metadata["voxcpm.quantization.feat_decoder"], "q8")
            self.assertEqual(writer.metadata["voxcpm.quantization.audio_vae"], "f32")
            self.assertIn("audio_vae.encoder.conv.weight", {name for name, *_ in writer.tensors})

            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["model_file"], BASE_MODEL_FILE)

    def test_base_export_removes_stale_legacy_base_outputs_only(self):
        main_weights = {
            "base_lm.norm.weight": np.zeros(2, dtype=np.float32),
            "base_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "residual_lm.norm.weight": np.zeros(2, dtype=np.float32),
            "residual_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_encoder.in_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "feat_encoder.special_token": np.zeros(2, dtype=np.float32),
            "feat_decoder.input_embed.weight": np.zeros((2, 2), dtype=np.float32),
            "fsq_layer.project_in.weight": np.zeros((2, 2), dtype=np.float32),
            "enc_to_lm_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "lm_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "res_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "stop_proj.weight": np.zeros((2, 2), dtype=np.float32),
            "stop_head.weight": np.zeros((2, 2), dtype=np.float32),
        }
        vae_weights = {"encoder.conv.weight": np.zeros((2, 2), dtype=np.float32)}
        config = {"architecture": "voxcpm2"}
        stale_names = [
            "manifest.json",
            "base_lm.gguf",
            "residual_lm.gguf",
            "feat_encoder.gguf",
            "feat_decoder.gguf",
            "audio_vae.gguf",
            "projections.gguf",
        ]

        with TemporaryDirectory() as model_tmp, TemporaryDirectory() as output_tmp:
            model_dir = Path(model_tmp)
            output_dir = Path(output_tmp)
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for name in stale_names + ["notes.txt", "lora_manifest.json", "lora_base_lm.gguf"]:
                (output_dir / name).write_text("stale", encoding="utf-8")

            RecordingWriter.instances = []
            with (
                patch("exporter.export_voxcpm.GGUFWriter", RecordingWriter),
                patch("exporter.export_voxcpm.load_weights", return_value=(main_weights, vae_weights, "safetensors")),
            ):
                export(
                    model_dir,
                    output_dir,
                    {
                        "quant_lm": "fp16",
                        "quant_encoder": "fp16",
                        "quant_dit": "fp16",
                        "quant_vae": "f32",
                    },
                    "2.0",
                )

            self.assertTrue((output_dir / BASE_MODEL_FILE).exists())
            for name in stale_names:
                self.assertFalse((output_dir / name).exists(), name)
            self.assertTrue((output_dir / "notes.txt").exists())
            self.assertTrue((output_dir / "lora_manifest.json").exists())
            self.assertTrue((output_dir / "lora_base_lm.gguf").exists())


if __name__ == "__main__":
    unittest.main()
