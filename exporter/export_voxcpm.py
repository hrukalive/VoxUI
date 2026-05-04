"""Export VoxCPM model weights to native multi-file GGUF bundles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exporter.gguf_writer import (
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUFWriter,
)
from exporter.quantize import quantize_fp16, quantize_q4_0, quantize_q8_0


QUANT_MAP = {
    "fp16": (quantize_fp16, GGML_TYPE_F16),
    "q8": (quantize_q8_0, GGML_TYPE_Q8_0),
    "q4": (quantize_q4_0, GGML_TYPE_Q4_0),
}

COMPONENT_FILES = {
    "base_lm": "base_lm.gguf",
    "residual_lm": "residual_lm.gguf",
    "feat_encoder": "feat_encoder.gguf",
    "feat_decoder": "feat_decoder.gguf",
    "audio_vae": "audio_vae.gguf",
    "projections": "projections.gguf",
}

PROJECTION_PREFIXES = (
    "fsq_layer.",
    "enc_to_lm_proj.",
    "lm_to_dit_proj.",
    "res_to_dit_proj.",
    "fusion_concat_proj.",
    "stop_proj.",
    "stop_head.",
)

QUANT_ARG_MAP = {
    "base_lm.gguf": "quant_lm",
    "residual_lm.gguf": "quant_lm",
    "feat_encoder.gguf": "quant_encoder",
    "feat_decoder.gguf": "quant_dit",
    "audio_vae.gguf": "quant_vae",
    "projections.gguf": "quant_lm",
}

REQUIRED_PREFIXES = {
    "base_lm.gguf": [
        "base_lm.norm.weight",
        "base_lm.layers.0.self_attn.q_proj.weight",
    ],
    "residual_lm.gguf": [
        "residual_lm.norm.weight",
        "residual_lm.layers.0.self_attn.q_proj.weight",
    ],
    "feat_encoder.gguf": [
        "feat_encoder.in_proj.weight",
        "feat_encoder.special_token",
    ],
    "feat_decoder.gguf": ["feat_decoder."],
    "audio_vae.gguf": ["audio_vae."],
    "projections.gguf": [
        "fsq_layer.",
        "enc_to_lm_proj.weight",
        "lm_to_dit_proj.weight",
        "res_to_dit_proj.weight",
        "stop_proj.weight",
        "stop_head.weight",
    ],
}


def get_component_for_key(key: str):
    if key.startswith("base_lm."):
        return "base_lm.gguf", lambda k: k, "lm"
    if key.startswith("residual_lm."):
        return "residual_lm.gguf", lambda k: k, "lm"
    if key.startswith("feat_encoder."):
        return "feat_encoder.gguf", lambda k: k, "encoder"
    if key.startswith("feat_decoder."):
        return "feat_decoder.gguf", lambda k: k, "dit"
    if key.startswith(PROJECTION_PREFIXES):
        return "projections.gguf", lambda k: k, "projections"
    return None, None, None


def tensor_to_f32_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().to("cpu").to(dtype=__import__("torch").float32).contiguous().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def load_weights(model_dir: str | Path):
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    vae_safetensors_path = model_dir / "audiovae.safetensors"
    vae_pth_path = model_dir / "audiovae.pth"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        main_weights = load_file(str(safetensors_path), device="cpu")
        source_weight_format = "safetensors"
    elif bin_path.exists():
        import torch

        data = torch.load(bin_path, map_location="cpu", weights_only=True)
        main_weights = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        source_weight_format = "pytorch_model.bin"
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. Expected model.safetensors or pytorch_model.bin"
        )

    vae_weights = None
    if vae_safetensors_path.exists():
        from safetensors.torch import load_file

        vae_weights = load_file(str(vae_safetensors_path), device="cpu")
    elif vae_pth_path.exists():
        import torch

        data = torch.load(vae_pth_path, map_location="cpu", weights_only=True)
        vae_weights = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data

    return main_weights, vae_weights, source_weight_format


def partition_weights(main_weights: dict[str, Any], vae_weights: dict[str, Any] | None):
    buckets: dict[str, list[tuple[str, Any]]] = {}
    unmapped_keys: list[str] = []

    for key, tensor in main_weights.items():
        filename, transform, _ = get_component_for_key(key)
        if filename is None or transform is None:
            unmapped_keys.append(key)
            continue
        buckets.setdefault(filename, []).append((transform(key), tensor))

    if vae_weights:
        for key, tensor in vae_weights.items():
            buckets.setdefault("audio_vae.gguf", []).append((f"audio_vae.{key}", tensor))

    if unmapped_keys:
        sample = ", ".join(unmapped_keys[:10])
        raise ValueError(f"unmapped tensor keys ({len(unmapped_keys)}): {sample}")

    return buckets


def _matches_required(name: str, required: str) -> bool:
    return name == required or name.startswith(required)


def validate_required_tensors(buckets: dict[str, list[tuple[str, Any]]], variant: str):
    for filename, required_names in REQUIRED_PREFIXES.items():
        tensor_names = [name for name, _ in buckets.get(filename, [])]
        if not tensor_names:
            raise ValueError(f"missing required tensor component {filename} for variant {variant}")

        seen: set[str] = set()
        for name in tensor_names:
            if name in seen:
                raise ValueError(f"duplicate tensor {name!r} in {filename}")
            seen.add(name)

        for required in required_names:
            if not any(_matches_required(name, required) for name in tensor_names):
                raise ValueError(f"missing required tensor {required!r} in {filename}")


def _audio_vae_manifest(config: dict[str, Any]) -> dict[str, Any]:
    vae = dict(config.get("audio_vae_config") or {})
    vae.setdefault("sample_rate", 16000 if config.get("architecture") == "voxcpm2" else 44100)
    if config.get("architecture") == "voxcpm2":
        vae.setdefault("out_sample_rate", 48000)
    vae.setdefault("latent_dim", config.get("feat_dim", 64))
    vae.setdefault("chunk_size", 20)
    vae.setdefault("decode_chunk_size", 240)
    vae.setdefault("encoder_rates", [])
    vae.setdefault("decoder_rates", [])
    return vae


def build_manifest(
    *,
    model_dir: Path,
    config: dict[str, Any],
    variant: str,
    source_weight_format: str,
    component_quantization: dict[str, str],
) -> dict[str, Any]:
    architecture = config.get("architecture", "voxcpm")
    special_tokens = {
        "audio_start": 101,
        "audio_end": 102,
    }
    if variant == "2.0" or architecture == "voxcpm2":
        special_tokens["ref_audio_start"] = 103
        special_tokens["ref_audio_end"] = 104

    manifest = {
        "schema_version": 1,
        "architecture": architecture,
        "variant": variant,
        "source_model_dir": str(Path(model_dir).resolve()),
        "source_weight_format": source_weight_format,
        "special_tokens": special_tokens,
        "patch_size": config.get("patch_size", 4),
        "feat_dim": config.get("feat_dim", 64),
        "scalar_quantization_latent_dim": config.get(
            "scalar_quantization_latent_dim",
            512 if architecture == "voxcpm2" else 256,
        ),
        "scalar_quantization_scale": float(config.get("scalar_quantization_scale", 9.0)),
        "audio_vae": _audio_vae_manifest(config),
        "lm_config": config.get("lm_config", {}),
        "encoder_config": config.get("encoder_config", {}),
        "dit_config": config.get("dit_config", {}),
        "residual_lm_num_layers": config.get("residual_lm_num_layers"),
        "residual_lm_no_rope": config.get("residual_lm_no_rope", architecture == "voxcpm2"),
        "components": dict(COMPONENT_FILES),
        "quantization": component_quantization,
    }
    return manifest


def _add_metadata_if_supported(writer: GGUFWriter, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, list) and not value:
        return
    if isinstance(value, dict):
        writer.add_metadata(key, json.dumps(value, ensure_ascii=False))
        return
    writer.add_metadata(key, value)


def add_common_metadata(writer: GGUFWriter, config: dict[str, Any], component_name: str, quant_name: str) -> None:
    writer.add_metadata("voxcpm.architecture", config.get("architecture", "voxcpm"))
    writer.add_metadata("voxcpm.component", component_name)
    writer.add_metadata("voxcpm.quantization", quant_name)
    writer.add_metadata("voxcpm.patch_size", int(config.get("patch_size", 4)))
    writer.add_metadata("voxcpm.feat_dim", int(config.get("feat_dim", 64)))


def add_lm_metadata(writer: GGUFWriter, config: dict[str, Any], component_name: str, quant_name: str) -> None:
    add_common_metadata(writer, config, component_name, quant_name)
    lm = config.get("lm_config", {})
    for key, value in lm.items():
        _add_metadata_if_supported(writer, f"voxcpm.{component_name}.{key}", value)
    if component_name == "residual_lm":
        _add_metadata_if_supported(writer, "voxcpm.residual_lm.num_layers", config.get("residual_lm_num_layers"))
        _add_metadata_if_supported(writer, "voxcpm.residual_lm.no_rope", config.get("residual_lm_no_rope"))


def add_encoder_metadata(writer: GGUFWriter, config: dict[str, Any], quant_name: str) -> None:
    add_common_metadata(writer, config, "feat_encoder", quant_name)
    for key, value in config.get("encoder_config", {}).items():
        _add_metadata_if_supported(writer, f"voxcpm.feat_encoder.{key}", value)


def add_dit_metadata(writer: GGUFWriter, config: dict[str, Any], quant_name: str) -> None:
    add_common_metadata(writer, config, "feat_decoder", quant_name)
    for key, value in config.get("dit_config", {}).items():
        _add_metadata_if_supported(writer, f"voxcpm.feat_decoder.{key}", value)


def add_vae_metadata(writer: GGUFWriter, config: dict[str, Any], quant_name: str) -> None:
    add_common_metadata(writer, config, "audio_vae", quant_name)
    for key, value in _audio_vae_manifest(config).items():
        _add_metadata_if_supported(writer, f"voxcpm.audio_vae.{key}", value)


def add_projections_metadata(writer: GGUFWriter, config: dict[str, Any], quant_name: str) -> None:
    add_common_metadata(writer, config, "projections", quant_name)
    for key in ("scalar_quantization_latent_dim", "scalar_quantization_scale", "patch_size", "feat_dim"):
        _add_metadata_if_supported(writer, f"voxcpm.projections.{key}", config.get(key))


METADATA_ADDERS = {
    "base_lm.gguf": lambda w, c, q: add_lm_metadata(w, c, "base_lm", q),
    "residual_lm.gguf": lambda w, c, q: add_lm_metadata(w, c, "residual_lm", q),
    "feat_encoder.gguf": add_encoder_metadata,
    "feat_decoder.gguf": add_dit_metadata,
    "audio_vae.gguf": add_vae_metadata,
    "projections.gguf": add_projections_metadata,
}


def write_component_gguf(
    *,
    output_dir: Path,
    filename: str,
    tensors: list[tuple[str, Any]],
    config: dict[str, Any],
    quant_name: str,
) -> None:
    if quant_name not in QUANT_MAP:
        raise ValueError(f"Unknown quantization {quant_name!r}; expected one of {sorted(QUANT_MAP)}")
    quant_fn, ggml_dtype = QUANT_MAP[quant_name]

    writer = GGUFWriter()
    meta_fn = METADATA_ADDERS.get(filename)
    if meta_fn:
        meta_fn(writer, config, quant_name)

    for tensor_name, tensor in tensors:
        arr = tensor_to_f32_numpy(tensor)
        writer.add_tensor(tensor_name, quant_fn(arr), list(arr.shape), ggml_dtype)

    writer.write(str(output_dir / filename))


def copy_bundle_files(model_dir: Path, output_dir: Path) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        src = model_dir / name
        if not src.exists():
            if name == "config.json":
                raise FileNotFoundError(f"{name} not found in {model_dir}")
            continue
        shutil.copy2(src, output_dir / name)


def export(model_dir: str | Path, output_dir: str | Path, quant_args: dict[str, str], variant: str) -> dict[str, Any]:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    main_weights, vae_weights, source_weight_format = load_weights(model_dir)
    buckets = partition_weights(main_weights, vae_weights)
    validate_required_tensors(buckets, variant=variant)

    output_dir.mkdir(parents=True, exist_ok=True)
    component_quantization: dict[str, str] = {}
    for filename, tensors in sorted(buckets.items()):
        quant_key = QUANT_ARG_MAP.get(filename, "quant_lm")
        quant_name = quant_args.get(quant_key, "fp16")
        component_quantization[filename] = quant_name
        print(f"Writing {filename} ({len(tensors)} tensors, quant={quant_name})")
        write_component_gguf(
            output_dir=output_dir,
            filename=filename,
            tensors=tensors,
            config=config,
            quant_name=quant_name,
        )

    copy_bundle_files(model_dir, output_dir)
    manifest = build_manifest(
        model_dir=model_dir,
        config=config,
        variant=variant,
        source_weight_format=source_weight_format,
        component_quantization=component_quantization,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _lora_key_transform(key: str):
    if key.startswith("base_lm."):
        return "base_lm", key
    if key.startswith("residual_lm."):
        return "residual_lm", key
    if key.startswith("feat_decoder."):
        return "feat_decoder", key
    if key.startswith(PROJECTION_PREFIXES):
        return "projections", key
    return None, None


def _safe_lora_dir_name(lora_dir: Path) -> str:
    name = lora_dir.name
    if name == "latest" and lora_dir.parent.name:
        name = lora_dir.parent.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "adapter"
    return f"lora_{name}"


def export_lora(lora_dir: str | Path, output_dir: str | Path, config_path: str | Path, variant: str) -> dict[str, Any]:
    lora_dir = Path(lora_dir)
    output_dir = Path(output_dir) / _safe_lora_dir_name(lora_dir)
    config_path = Path(config_path)
    lora_config_path = lora_dir / "lora_config.json"
    lora_weights_path = lora_dir / "lora_weights.safetensors"
    if not lora_config_path.exists():
        raise FileNotFoundError(f"lora_config.json not found in {lora_dir}")
    if not lora_weights_path.exists():
        raise FileNotFoundError(f"lora_weights.safetensors not found in {lora_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    lora_config = json.loads(lora_config_path.read_text(encoding="utf-8"))
    lc = lora_config.get("lora_config", lora_config)

    from safetensors.torch import load_file

    lora_weights = load_file(str(lora_weights_path), device="cpu")
    buckets: dict[str, list[tuple[str, Any]]] = {}
    for key, tensor in lora_weights.items():
        component, new_name = _lora_key_transform(key)
        if component is None:
            raise ValueError(f"unmapped LoRA tensor key: {key}")
        buckets.setdefault(component, []).append((new_name, tensor))

    output_dir.mkdir(parents=True, exist_ok=True)
    components: dict[str, str] = {}
    for component, tensors in sorted(buckets.items()):
        filename = f"lora_{component}.gguf"
        components[component] = filename
        writer = GGUFWriter()
        writer.add_metadata("voxcpm.architecture", config.get("architecture", "voxcpm"))
        writer.add_metadata("voxcpm.component", "lora")
        writer.add_metadata("voxcpm.lora.target_component", component)
        writer.add_metadata("voxcpm.lora.rank", int(lc.get("r", 0)))
        writer.add_metadata("voxcpm.lora.alpha", int(lc.get("alpha", lc.get("r", 0))))
        writer.add_metadata("voxcpm.quantization", "fp16")
        for tensor_name, tensor in tensors:
            arr = tensor_to_f32_numpy(tensor)
            writer.add_tensor(tensor_name, quantize_fp16(arr), list(arr.shape), GGML_TYPE_F16)
        writer.write(str(output_dir / filename))

    manifest = {
        "schema_version": 1,
        "architecture": config.get("architecture", "voxcpm"),
        "variant": variant,
        "source_lora_dir": str(lora_dir.resolve()),
        "rank": int(lc.get("r", 0)),
        "alpha": int(lc.get("alpha", lc.get("r", 0))),
        "enabled": {
            "lm": bool(lc.get("enable_lm", False)),
            "dit": bool(lc.get("enable_dit", False)),
            "projections": bool(lc.get("enable_proj", False)),
        },
        "target_modules": {
            "lm": lc.get("target_modules_lm", []),
            "dit": lc.get("target_modules_dit", []),
            "projections": lc.get("target_proj_modules", []),
        },
        "components": components,
    }
    (output_dir / "lora_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(lora_config_path, output_dir / "lora_config.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoxCPM model to native multi-file GGUF bundle")
    parser.add_argument("--model-dir", required=True, help="Path to VoxCPM model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for GGUF files")
    parser.add_argument("--variant", required=True, choices=["0.5", "1.5", "2.0"], help="VoxCPM variant")
    parser.add_argument("--lora-dir", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--quant-lm", default="fp16", choices=["fp16", "q8", "q4"])
    parser.add_argument("--quant-encoder", default="fp16", choices=["fp16", "q8", "q4"])
    parser.add_argument("--quant-dit", default="fp16", choices=["fp16", "q8", "q4"])
    parser.add_argument("--quant-vae", default="fp16", choices=["fp16", "q8", "q4"])
    args = parser.parse_args()

    quant_args = {
        "quant_lm": args.quant_lm,
        "quant_encoder": args.quant_encoder,
        "quant_dit": args.quant_dit,
        "quant_vae": args.quant_vae,
    }

    export(args.model_dir, args.output_dir, quant_args, args.variant)
    if args.lora_dir:
        export_lora(args.lora_dir, args.output_dir, Path(args.model_dir) / "config.json", args.variant)


if __name__ == "__main__":
    main()
