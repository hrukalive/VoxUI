"""Export VoxCPM model weights to native GGUF bundles."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exporter.gguf_writer import (
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUFWriter,
)
from exporter.quantize import quantize_f32, quantize_fp16, quantize_q4_0, quantize_q8_0


QUANT_MAP = {
    "f32": (quantize_f32, GGML_TYPE_F32),
    "fp16": (quantize_fp16, GGML_TYPE_F16),
    "q8": (quantize_q8_0, GGML_TYPE_Q8_0),
    "q4": (quantize_q4_0, GGML_TYPE_Q4_0),
}

QUANT_PROFILES = ("manual", "fp16", "q4-lm")

BASE_MODEL_FILE = "model.gguf"

BASE_LM = "base_lm"
RESIDUAL_LM = "residual_lm"
FEAT_ENCODER = "feat_encoder"
FEAT_DECODER = "feat_decoder"
AUDIO_VAE = "audio_vae"
PROJECTIONS = "projections"

BASE_COMPONENTS = (
    BASE_LM,
    RESIDUAL_LM,
    FEAT_ENCODER,
    FEAT_DECODER,
    AUDIO_VAE,
    PROJECTIONS,
)

STALE_BASE_OUTPUTS = (
    "manifest.json",
    "base_lm.gguf",
    "residual_lm.gguf",
    "feat_encoder.gguf",
    "feat_decoder.gguf",
    "audio_vae.gguf",
    "projections.gguf",
)

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
    BASE_LM: "quant_lm",
    RESIDUAL_LM: "quant_lm",
    FEAT_ENCODER: "quant_encoder",
    FEAT_DECODER: "quant_dit",
    AUDIO_VAE: "quant_vae",
    PROJECTIONS: "quant_lm",
}


def profile_default_quant_args(profile: str, variant: str) -> dict[str, str]:
    if profile not in QUANT_PROFILES:
        raise ValueError(f"Unknown quantization profile {profile!r}; expected one of {QUANT_PROFILES}")
    if profile == "manual":
        return {
            "quant_lm": "fp16",
            "quant_encoder": "fp16",
            "quant_dit": "fp16",
            "quant_vae": "f32",
        }
    if profile == "fp16":
        return {
            "quant_lm": "fp16",
            "quant_encoder": "fp16",
            "quant_dit": "fp16",
            "quant_vae": "f32" if variant == "2.0" else "fp16",
        }
    return {
        "quant_lm": "q4",
        "quant_encoder": "fp16",
        "quant_dit": "fp16",
        "quant_vae": "f32" if variant == "2.0" else "fp16",
    }


def resolve_quant_args(
    *,
    variant: str,
    profile: str,
    quant_lm: str | None,
    quant_encoder: str | None,
    quant_dit: str | None,
    quant_vae: str | None,
) -> dict[str, str]:
    quant_args = profile_default_quant_args(profile, variant)
    overrides = {
        "quant_lm": quant_lm,
        "quant_encoder": quant_encoder,
        "quant_dit": quant_dit,
        "quant_vae": quant_vae,
    }
    for key, value in overrides.items():
        if value is not None:
            if value not in QUANT_MAP:
                raise ValueError(f"Unknown quantization {value!r}; expected one of {sorted(QUANT_MAP)}")
            quant_args[key] = value
    return quant_args


REQUIRED_PREFIXES = {
    BASE_LM: [
        "base_lm.norm.weight",
        "base_lm.layers.0.self_attn.q_proj.weight",
    ],
    RESIDUAL_LM: [
        "residual_lm.norm.weight",
        "residual_lm.layers.0.self_attn.q_proj.weight",
    ],
    FEAT_ENCODER: [
        "feat_encoder.in_proj.weight",
        "feat_encoder.special_token",
    ],
    FEAT_DECODER: ["feat_decoder."],
    AUDIO_VAE: ["audio_vae."],
    PROJECTIONS: [
        "fsq_layer.",
        "enc_to_lm_proj.weight",
        "lm_to_dit_proj.weight",
        "res_to_dit_proj.weight",
        "stop_proj.weight",
        "stop_head.weight",
    ],
}


def classify_tensor_key(key: str) -> tuple[str | None, str | None]:
    if key.startswith("base_lm."):
        return BASE_LM, key
    if key.startswith("residual_lm."):
        return RESIDUAL_LM, key
    if key.startswith("feat_encoder."):
        return FEAT_ENCODER, key
    if key.startswith("feat_decoder."):
        return FEAT_DECODER, key
    if key.startswith(PROJECTION_PREFIXES):
        return PROJECTIONS, key
    return None, None


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
        component, tensor_name = classify_tensor_key(key)
        if component is None or tensor_name is None:
            unmapped_keys.append(key)
            continue
        buckets.setdefault(component, []).append((tensor_name, tensor))

    if vae_weights:
        for key, tensor in vae_weights.items():
            buckets.setdefault(AUDIO_VAE, []).append((f"audio_vae.{key}", tensor))

    if unmapped_keys:
        sample = ", ".join(unmapped_keys[:10])
        raise ValueError(f"unmapped tensor keys ({len(unmapped_keys)}): {sample}")

    return buckets


def _matches_required(name: str, required: str) -> bool:
    return name == required or name.startswith(required)


def validate_required_tensors(buckets: dict[str, list[tuple[str, Any]]], variant: str):
    for component, required_names in REQUIRED_PREFIXES.items():
        tensor_names = [name for name, _ in buckets.get(component, [])]
        if not tensor_names:
            raise ValueError(f"missing required tensor component {component} for variant {variant}")

        seen: set[str] = set()
        for name in tensor_names:
            if name in seen:
                raise ValueError(f"duplicate tensor {name!r} in {component}")
            seen.add(name)

        for required in required_names:
            if not any(_matches_required(name, required) for name in tensor_names):
                raise ValueError(f"missing required tensor {required!r} in {component}")


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
    BASE_LM: lambda w, c, q: add_lm_metadata(w, c, BASE_LM, q),
    RESIDUAL_LM: lambda w, c, q: add_lm_metadata(w, c, RESIDUAL_LM, q),
    FEAT_ENCODER: add_encoder_metadata,
    FEAT_DECODER: add_dit_metadata,
    AUDIO_VAE: add_vae_metadata,
    PROJECTIONS: add_projections_metadata,
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


def add_base_metadata(
    writer: GGUFWriter,
    *,
    model_dir: Path,
    config: dict[str, Any],
    variant: str,
    quant_profile: str,
    source_weight_format: str,
    component_quantization: dict[str, str],
) -> None:
    architecture = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.schema_version", 2)
    writer.add_metadata("voxcpm.kind", "base")
    writer.add_metadata("voxcpm.architecture", architecture)
    writer.add_metadata("voxcpm.variant", variant)
    writer.add_metadata("voxcpm.quant_profile", quant_profile)
    writer.add_metadata("voxcpm.source_model_dir", str(model_dir.resolve()))
    writer.add_metadata("voxcpm.source_weight_format", source_weight_format)
    writer.add_metadata("voxcpm.patch_size", int(config.get("patch_size", 4)))
    writer.add_metadata("voxcpm.feat_dim", int(config.get("feat_dim", 64)))
    _add_metadata_if_supported(
        writer,
        "voxcpm.scalar_quantization_latent_dim",
        config.get("scalar_quantization_latent_dim", 512 if architecture == "voxcpm2" else 256),
    )
    _add_metadata_if_supported(
        writer,
        "voxcpm.scalar_quantization_scale",
        float(config.get("scalar_quantization_scale", 9.0)),
    )
    _add_metadata_if_supported(writer, "voxcpm.audio_vae", _audio_vae_manifest(config))
    _add_metadata_if_supported(writer, "voxcpm.lm_config", config.get("lm_config", {}))
    _add_metadata_if_supported(writer, "voxcpm.encoder_config", config.get("encoder_config", {}))
    _add_metadata_if_supported(writer, "voxcpm.dit_config", config.get("dit_config", {}))
    _add_metadata_if_supported(writer, "voxcpm.residual_lm_num_layers", config.get("residual_lm_num_layers"))
    _add_metadata_if_supported(
        writer,
        "voxcpm.residual_lm_no_rope",
        config.get("residual_lm_no_rope", architecture == "voxcpm2"),
    )
    for component in BASE_COMPONENTS:
        _add_metadata_if_supported(
            writer,
            f"voxcpm.quantization.{component}",
            component_quantization.get(component),
        )


def write_base_gguf(
    *,
    output_dir: Path,
    buckets: dict[str, list[tuple[str, Any]]],
    config: dict[str, Any],
    quant_args: dict[str, str],
    model_dir: Path,
    variant: str,
    quant_profile: str,
    source_weight_format: str,
) -> dict[str, str]:
    component_quantization: dict[str, str] = {}
    for component in BASE_COMPONENTS:
        if component in buckets:
            quant_key = QUANT_ARG_MAP.get(component, "quant_lm")
            component_quantization[component] = quant_args.get(quant_key, "fp16")

    writer = GGUFWriter()
    add_base_metadata(
        writer,
        model_dir=model_dir,
        config=config,
        variant=variant,
        quant_profile=quant_profile,
        source_weight_format=source_weight_format,
        component_quantization=component_quantization,
    )

    for component in BASE_COMPONENTS:
        tensors = buckets.get(component, [])
        if not tensors:
            continue
        quant_name = component_quantization[component]
        if quant_name not in QUANT_MAP:
            raise ValueError(f"Unknown quantization {quant_name!r}; expected one of {sorted(QUANT_MAP)}")
        quant_fn, ggml_dtype = QUANT_MAP[quant_name]
        for tensor_name, tensor in tensors:
            arr = tensor_to_f32_numpy(tensor)
            writer.add_tensor(tensor_name, quant_fn(arr), list(arr.shape), ggml_dtype)

    writer.write(str(output_dir / BASE_MODEL_FILE))
    return component_quantization


def copy_bundle_files(model_dir: Path, output_dir: Path) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        src = model_dir / name
        if not src.exists():
            if name == "config.json":
                raise FileNotFoundError(f"{name} not found in {model_dir}")
            continue
        shutil.copy2(src, output_dir / name)


def remove_stale_base_outputs(output_dir: Path) -> None:
    for name in STALE_BASE_OUTPUTS:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def export(
    model_dir: str | Path,
    output_dir: str | Path,
    quant_args: dict[str, str],
    variant: str,
    quant_profile: str = "manual",
) -> dict[str, Any]:
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
    remove_stale_base_outputs(output_dir)
    print(f"Writing {BASE_MODEL_FILE} ({sum(len(tensors) for tensors in buckets.values())} tensors)")
    component_quantization = write_base_gguf(
        output_dir=output_dir,
        buckets=buckets,
        config=config,
        quant_args=quant_args,
        model_dir=model_dir,
        variant=variant,
        quant_profile=quant_profile,
        source_weight_format=source_weight_format,
    )

    copy_bundle_files(model_dir, output_dir)
    return {
        "schema_version": 2,
        "kind": "base",
        "architecture": config.get("architecture", "voxcpm"),
        "variant": variant,
        "model_file": BASE_MODEL_FILE,
        "source_model_dir": str(model_dir.resolve()),
        "source_weight_format": source_weight_format,
        "quant_profile": quant_profile,
        "quantization": component_quantization,
    }


def _validate_lora_key(key: str) -> str:
    if key.startswith(("base_lm.", "residual_lm.", "feat_decoder.")):
        return key
    if key.startswith(PROJECTION_PREFIXES):
        return key
    raise ValueError(f"unmapped LoRA tensor key: {key}")


def _safe_lora_name(lora_dir: Path) -> str:
    name = lora_dir.name
    if name == "latest" and lora_dir.parent.name:
        name = lora_dir.parent.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "adapter"
    return name


def _safe_lora_file_name(lora_dir: Path) -> str:
    return f"lora_{_safe_lora_name(lora_dir)}.gguf"


def validate_lora_pairs(tensor_names: list[str], rank: int) -> None:
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")

    names = set(tensor_names)
    for name in tensor_names:
        if ".lora_A." in name:
            pair_name = name.replace(".lora_A.", ".lora_B.", 1)
            if pair_name not in names:
                raise ValueError(f"missing LoRA B tensor for {name!r}")
            continue
        if ".lora_B." in name:
            pair_name = name.replace(".lora_B.", ".lora_A.", 1)
            if pair_name not in names:
                raise ValueError(f"missing LoRA A tensor for {name!r}")
            continue
        raise ValueError(f"LoRA tensor key must contain .lora_A. or .lora_B.: {name}")


def export_lora(lora_dir: str | Path, output_dir: str | Path, config_path: str | Path, variant: str) -> dict[str, Any]:
    lora_dir = Path(lora_dir)
    output_dir = Path(output_dir)
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
    tensors: list[tuple[str, Any]] = []
    for key, tensor in lora_weights.items():
        tensors.append((_validate_lora_key(key), tensor))

    rank = int(lc.get("r", 0))
    alpha = lc.get("alpha", rank)
    tensor_names = [name for name, _ in tensors]
    validate_lora_pairs(tensor_names, rank)

    filename = _safe_lora_file_name(lora_dir)
    lora_name = _safe_lora_name(lora_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter()
    writer.add_metadata("voxcpm.schema_version", 2)
    writer.add_metadata("voxcpm.kind", "lora")
    writer.add_metadata("voxcpm.architecture", config.get("architecture", "voxcpm"))
    writer.add_metadata("voxcpm.variant", variant)
    writer.add_metadata("voxcpm.lora.name", lora_name)
    writer.add_metadata("voxcpm.lora.rank", rank)
    writer.add_metadata("voxcpm.lora.alpha", alpha)
    enabled_targets = {
        "lm": bool(lc.get("enable_lm", False)),
        "dit": bool(lc.get("enable_dit", False)),
        "projections": bool(lc.get("enable_proj", False)),
    }
    target_modules = {
        "lm": lc.get("target_modules_lm", []),
        "dit": lc.get("target_modules_dit", []),
        "projections": lc.get("target_proj_modules", []),
    }
    writer.add_metadata("voxcpm.lora.enabled_targets", json.dumps(enabled_targets, ensure_ascii=False))
    writer.add_metadata("voxcpm.lora.target_modules", json.dumps(target_modules, ensure_ascii=False))
    writer.add_metadata("voxcpm.quantization", "fp16")

    for tensor_name, tensor in sorted(tensors, key=lambda item: item[0]):
        arr = tensor_to_f32_numpy(tensor)
        writer.add_tensor(tensor_name, quantize_fp16(arr), list(arr.shape), GGML_TYPE_F16)
    writer.write(str(output_dir / filename))

    return {
        "schema_version": 2,
        "kind": "lora",
        "architecture": config.get("architecture", "voxcpm"),
        "variant": variant,
        "file": filename,
        "source_lora_dir": str(lora_dir.resolve()),
        "rank": rank,
        "alpha": alpha,
        "enabled_targets": enabled_targets,
        "target_modules": target_modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoxCPM model to native GGUF bundle")
    parser.add_argument("--model-dir", required=True, help="Path to VoxCPM model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for GGUF files")
    parser.add_argument("--variant", required=True, choices=["0.5", "1.5", "2.0"], help="VoxCPM variant")
    parser.add_argument("--lora-dir", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--quant-profile", default="manual", choices=QUANT_PROFILES)
    parser.add_argument("--quant-lm", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-encoder", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-dit", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-vae", default=None, choices=sorted(QUANT_MAP))
    args = parser.parse_args()

    quant_args = resolve_quant_args(
        variant=args.variant,
        profile=args.quant_profile,
        quant_lm=args.quant_lm,
        quant_encoder=args.quant_encoder,
        quant_dit=args.quant_dit,
        quant_vae=args.quant_vae,
    )

    export(args.model_dir, args.output_dir, quant_args, args.variant, quant_profile=args.quant_profile)
    if args.lora_dir:
        export_lora(args.lora_dir, args.output_dir, Path(args.model_dir) / "config.json", args.variant)


if __name__ == "__main__":
    main()
