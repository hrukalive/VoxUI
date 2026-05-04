"""Export VoxCPM model weights to multi-file GGUF format."""

import argparse
import json
import os
import sys
import numpy as np

from exporter.gguf_writer import (
    GGUFWriter,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
)
from exporter.quantize import quantize_fp16, quantize_q8_0, quantize_q4_0

# Quantization name -> (function, ggml_type)
QUANT_MAP = {
    "fp16": (quantize_fp16, GGML_TYPE_F16),
    "q8": (quantize_q8_0, GGML_TYPE_Q8_0),
    "q4": (quantize_q4_0, GGML_TYPE_Q4_0),
}

# Component definitions: (gguf_filename, source_prefixes, key_transform)
# key_transform: callable(source_key) -> gguf_tensor_name or None to skip
PROJECTION_PREFIXES = (
    "fsq_layer.", "enc_to_lm_proj.", "lm_to_dit_proj.",
    "res_to_dit_proj.", "fusion_concat_proj.", "stop_proj.", "stop_head.",
)


def _make_lm_transform(prefix):
    """Strip .model. from base_lm.model.xxx or residual_lm.model.xxx."""
    component = prefix.split(".")[0]  # "base_lm" or "residual_lm"

    def transform(key):
        if not key.startswith(prefix):
            return None
        remainder = key[len(prefix):]
        return f"{component}.{remainder}"

    return transform


def _prefix_transform(src_prefix, dst_prefix):
    def transform(key):
        if not key.startswith(src_prefix):
            return None
        return dst_prefix + key[len(src_prefix):]
    return transform


def _projection_transform(key):
    for p in PROJECTION_PREFIXES:
        if key.startswith(p):
            return key  # keep as-is
    return None


def _vae_transform(key):
    return f"audiovae.{key}"




def get_component_for_key(key):
    """Return (gguf_filename, transform_fn, quant_group) for a key from the main weights.
    
    Note: VoxCPM saves safetensors with keys like 'base_lm.layers.0...' (no '.model.' prefix).
    The .model. is stripped during save_checkpoint in VoxCPM's training code.
    """
    if key.startswith("base_lm."):
        return "base_lm.gguf", lambda k: k, "lm"  # keep as-is
    if key.startswith("residual_lm."):
        return "residual_lm.gguf", lambda k: k, "lm"  # keep as-is
    if key.startswith("feat_encoder."):
        return "encoder.gguf", _prefix_transform("feat_encoder.", "encoder."), "encoder"
    if key.startswith("feat_decoder."):
        return "dit.gguf", _prefix_transform("feat_decoder.", "dit."), "dit"
    for p in PROJECTION_PREFIXES:
        if key.startswith(p):
            return "projections.gguf", _projection_transform, "projections"
    return None, None, None


def tensor_to_f32_numpy(tensor):
    """Convert a tensor (torch or numpy) to float32 numpy array."""
    if hasattr(tensor, "numpy"):
        # PyTorch tensor
        import torch
        return tensor.to(torch.float32).numpy()
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def load_weights(model_dir):
    """Load main weights and optionally VAE weights. Returns (state_dict, vae_dict_or_None)."""
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    vae_path = os.path.join(model_dir, "audiovae.pth")

    main_weights = {}
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        main_weights = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        import torch
        data = torch.load(bin_path, map_location="cpu", weights_only=True)
        # pytorch_model.bin may be a nested dict with 'state_dict' key
        if isinstance(data, dict) and "state_dict" in data:
            main_weights = data["state_dict"]
        else:
            main_weights = data
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    vae_weights = None
    if os.path.exists(vae_path):
        import torch
        vae_data = torch.load(vae_path, map_location="cpu", weights_only=True)
        # audiovae.pth may be a nested dict with 'state_dict' key
        if isinstance(vae_data, dict) and "state_dict" in vae_data:
            vae_weights = vae_data["state_dict"]
        else:
            vae_weights = vae_data

    return main_weights, vae_weights


def add_lm_metadata(writer, config, component_name, quant_name):
    """Add metadata for base_lm or residual_lm."""
    arch = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.architecture", arch)
    writer.add_metadata("voxcpm.component", component_name)
    writer.add_metadata("voxcpm.quantization", quant_name)

    lm = config.get("lm_config", {})

    # For residual_lm, override num_hidden_layers with the actual residual layer count
    if component_name == "residual_lm":
        actual_num_layers = config.get("residual_lm_num_layers", lm.get("num_hidden_layers", 28))
        for k in ("hidden_size", "num_attention_heads",
                  "num_key_value_heads", "intermediate_size", "vocab_size",
                  "max_position_embeddings", "kv_channels"):
            if k in lm:
                writer.add_metadata(f"voxcpm.{component_name}.{k}", lm[k])
        writer.add_metadata(f"voxcpm.{component_name}.num_hidden_layers", actual_num_layers)
    else:
        for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "intermediate_size", "vocab_size",
                  "max_position_embeddings", "kv_channels"):
            if k in lm:
                writer.add_metadata(f"voxcpm.{component_name}.{k}", lm[k])

    for k in ("rms_norm_eps", "rope_theta"):
        if k in lm:
            writer.add_metadata(f"voxcpm.{component_name}.{k}", float(lm[k]))

    for k in ("scale_emb", "scale_depth"):
        if k in lm:
            v = lm[k]
            writer.add_metadata(f"voxcpm.{component_name}.{k}",
                                float(v) if isinstance(v, (int, float)) else v)

    # Rope scaling factors
    rope = lm.get("rope_scaling", {})
    if "long_factor" in rope:
        writer.add_metadata(f"voxcpm.{component_name}.rope_long_factor",
                            [float(x) for x in rope["long_factor"]])
    if "short_factor" in rope:
        writer.add_metadata(f"voxcpm.{component_name}.rope_short_factor",
                            [float(x) for x in rope["short_factor"]])

    if component_name == "residual_lm":
        if "residual_lm_num_layers" in config:
            writer.add_metadata("voxcpm.residual_lm.num_layers",
                                config["residual_lm_num_layers"])
        if "residual_lm_no_rope" in config:
            writer.add_metadata("voxcpm.residual_lm.no_rope",
                                config["residual_lm_no_rope"])


def add_encoder_metadata(writer, config, quant_name):
    arch = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.architecture", arch)
    writer.add_metadata("voxcpm.component", "encoder")
    writer.add_metadata("voxcpm.quantization", quant_name)

    enc = config.get("encoder_config", {})
    for k in ("hidden_dim", "ffn_dim", "num_heads", "num_layers", "kv_channels"):
        if k in enc:
            writer.add_metadata(f"voxcpm.encoder.{k}", enc[k])


def add_dit_metadata(writer, config, quant_name):
    arch = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.architecture", arch)
    writer.add_metadata("voxcpm.component", "dit")
    writer.add_metadata("voxcpm.quantization", quant_name)

    dit = config.get("dit_config", {})
    for k in ("hidden_dim", "ffn_dim", "num_heads", "num_layers", "kv_channels"):
        if k in dit:
            writer.add_metadata(f"voxcpm.dit.{k}", dit[k])

    cfm = dit.get("cfm_config", {})
    if "sigma_min" in cfm:
        writer.add_metadata("voxcpm.dit.cfm_sigma_min", float(cfm["sigma_min"]))
    if "inference_cfg_rate" in cfm:
        writer.add_metadata("voxcpm.dit.cfm_cfg_rate", float(cfm["inference_cfg_rate"]))


def add_vae_metadata(writer, config, quant_name):
    arch = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.architecture", arch)
    writer.add_metadata("voxcpm.component", "audiovae")
    writer.add_metadata("voxcpm.quantization", quant_name)

    vae = config.get("audio_vae_config", {})
    for k in ("decoder_dim", "sample_rate", "out_sample_rate", "latent_dim", "encoder_dim"):
        if k in vae:
            writer.add_metadata(f"voxcpm.audiovae.{k}", vae[k])
    for k in ("decoder_rates", "encoder_rates"):
        if k in vae:
            writer.add_metadata(f"voxcpm.audiovae.{k}", vae[k])


def add_projections_metadata(writer, config, quant_name):
    arch = config.get("architecture", "voxcpm")
    writer.add_metadata("voxcpm.architecture", arch)
    writer.add_metadata("voxcpm.component", "projections")
    writer.add_metadata("voxcpm.quantization", quant_name)

    for k in ("scalar_quantization_latent_dim", "scalar_quantization_scale",
              "patch_size", "feat_dim"):
        if k in config:
            writer.add_metadata(f"voxcpm.projections.{k}", config[k])


METADATA_ADDERS = {
    "base_lm.gguf": lambda w, c, q: add_lm_metadata(w, c, "base_lm", q),
    "residual_lm.gguf": lambda w, c, q: add_lm_metadata(w, c, "residual_lm", q),
    "encoder.gguf": add_encoder_metadata,
    "dit.gguf": add_dit_metadata,
    "audiovae.gguf": add_vae_metadata,
    "projections.gguf": add_projections_metadata,
}

# Map gguf filename -> quant arg name
QUANT_ARG_MAP = {
    "base_lm.gguf": "quant_lm",
    "residual_lm.gguf": "quant_lm",
    "encoder.gguf": "quant_encoder",
    "dit.gguf": "quant_dit",
    "audiovae.gguf": "quant_vae",
    "projections.gguf": "quant_lm",  # small tensors, use LM quant or default fp16
}


def partition_weights(main_weights, vae_weights):
    """Partition all weights into per-component buckets.

    Returns dict: gguf_filename -> list of (gguf_tensor_name, tensor)
    """
    buckets = {}
    unmapped_keys = []

    for key, tensor in main_weights.items():
        filename, transform, _ = get_component_for_key(key)
        if filename is None:
            print(f"  WARNING: unmapped key {key}, skipping")
            unmapped_keys.append(key)
            continue
        if callable(transform):
            new_name = transform(key)
        else:
            new_name = None
        if new_name is None:
            continue
        buckets.setdefault(filename, []).append((new_name, tensor))

    if vae_weights:
        for key, tensor in vae_weights.items():
            new_name = _vae_transform(key)
            buckets.setdefault("audiovae.gguf", []).append((new_name, tensor))

    if unmapped_keys:
        print(f"\n  WARNING: {len(unmapped_keys)} unmapped key(s) were skipped.")

    return buckets


def export(model_dir, output_dir, quant_args):
    """Main export logic."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Architecture: {config.get('architecture', 'unknown')}")
    print(f"Loading weights from {model_dir}...")
    main_weights, vae_weights = load_weights(model_dir)
    print(f"  Main weights: {len(main_weights)} tensors")
    if vae_weights:
        print(f"  VAE weights: {len(vae_weights)} tensors")

    print("Partitioning weights...")
    buckets = partition_weights(main_weights, vae_weights)

    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for filename, tensors in sorted(buckets.items()):
        quant_key = QUANT_ARG_MAP.get(filename, "quant_lm")
        quant_name = quant_args.get(quant_key, "fp16")
        if quant_name not in QUANT_MAP:
            raise ValueError(
                f"Unknown quantization '{quant_name}' for {filename}. "
                f"Valid options: {', '.join(sorted(QUANT_MAP))}"
            )
        quant_fn, ggml_dtype = QUANT_MAP[quant_name]

        print(f"\nWriting {filename} ({len(tensors)} tensors, quant={quant_name})...")

        writer = GGUFWriter()

        # Add metadata
        meta_fn = METADATA_ADDERS.get(filename)
        if meta_fn:
            meta_fn(writer, config, quant_name)

        for tensor_name, tensor in tensors:
            arr = tensor_to_f32_numpy(tensor)
            shape = list(arr.shape)
            data = quant_fn(arr)
            writer.add_tensor(tensor_name, data, shape, ggml_dtype)

        out_path = os.path.join(output_dir, filename)
        writer.write(out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        summary.append((filename, len(tensors), size_mb, quant_name))
        print(f"  -> {out_path} ({size_mb:.1f} MB)")

    # Print summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"{'File':<25} {'Tensors':>8} {'Size (MB)':>10} {'Quant':>6}")
    print("-" * 60)
    total_tensors = 0
    total_size = 0.0
    for fname, ntens, size, qname in summary:
        print(f"{fname:<25} {ntens:>8} {size:>10.1f} {qname:>6}")
        total_tensors += ntens
        total_size += size
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_tensors:>8} {total_size:>10.1f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files written: {len(summary)}")

    # Copy tokenizer files to output directory
    import shutil
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    copied = 0
    for tf in tokenizer_files:
        src = os.path.join(model_dir, tf)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, tf))
            copied += 1
    if copied:
        print(f"Copied {copied} tokenizer file(s) to output directory")


def _lora_key_transform(key):
    """Transform LoRA weight key to GGUF tensor name, matching main export renaming.
    
    Note: LoRA safetensors keys already have .model. stripped (e.g., 'base_lm.layers.0...').
    """
    if key.startswith("base_lm."):
        return "base_lm", key  # keep as-is
    if key.startswith("residual_lm."):
        return "residual_lm", key  # keep as-is
    if key.startswith("feat_decoder."):
        remainder = key[len("feat_decoder."):]
        return "dit", f"dit.{remainder}"
    if key.startswith("feat_encoder."):
        remainder = key[len("feat_encoder."):]
        return "encoder", f"encoder.{remainder}"
    for p in PROJECTION_PREFIXES:
        if key.startswith(p):
            return "projections", key
    return None, None


def export_lora(lora_dir, output_dir, config_path):
    """Export LoRA weights as separate per-component GGUF files."""
    lora_config_path = os.path.join(lora_dir, "lora_config.json")
    lora_weights_path = os.path.join(lora_dir, "lora_weights.safetensors")

    if not os.path.exists(lora_config_path):
        raise FileNotFoundError(f"lora_config.json not found in {lora_dir}")
    if not os.path.exists(lora_weights_path):
        raise FileNotFoundError(f"lora_weights.safetensors not found in {lora_dir}")

    with open(lora_config_path, "r", encoding="utf-8") as f:
        lora_config = json.load(f)

    # Load main config for architecture info
    main_config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            main_config = json.load(f)

    lc = lora_config.get("lora_config", lora_config)
    rank = lc.get("r", 0)
    alpha = lc.get("alpha", rank)

    # Build target_modules map per component
    target_modules_map = {}
    if lc.get("enable_lm"):
        mods = lc.get("target_modules_lm", [])
        target_modules_map["base_lm"] = mods
        target_modules_map["residual_lm"] = mods
    if lc.get("enable_dit"):
        target_modules_map["dit"] = lc.get("target_modules_dit", [])
    if lc.get("enable_proj"):
        target_modules_map["projections"] = lc.get("target_proj_modules", [])

    # Load weights
    from safetensors.torch import load_file
    print(f"\nLoading LoRA weights from {lora_dir}...")
    lora_weights = load_file(lora_weights_path, device="cpu")
    print(f"  LoRA weights: {len(lora_weights)} tensors")

    # Partition by component
    buckets = {}
    for key, tensor in lora_weights.items():
        component, new_name = _lora_key_transform(key)
        if component is None:
            print(f"  WARNING: unmapped LoRA key {key}, skipping")
            continue
        buckets.setdefault(component, []).append((new_name, tensor))

    os.makedirs(output_dir, exist_ok=True)
    arch = main_config.get("architecture", "voxcpm")
    quant_fn, ggml_dtype = QUANT_MAP["fp16"]

    summary = []
    for component, tensors in sorted(buckets.items()):
        filename = f"lora_{component}.gguf"
        print(f"\nWriting {filename} ({len(tensors)} tensors, quant=fp16)...")

        writer = GGUFWriter()
        writer.add_metadata("voxcpm.architecture", arch)
        writer.add_metadata("voxcpm.component", "lora")
        writer.add_metadata("voxcpm.lora.target_component", component)
        writer.add_metadata("voxcpm.lora.rank", rank)
        writer.add_metadata("voxcpm.lora.alpha", alpha)
        mods = target_modules_map.get(component, [])
        writer.add_metadata("voxcpm.lora.target_modules", ",".join(mods))
        writer.add_metadata("voxcpm.quantization", "fp16")

        for tensor_name, tensor in tensors:
            arr = tensor_to_f32_numpy(tensor)
            shape = list(arr.shape)
            data = quant_fn(arr)
            writer.add_tensor(tensor_name, data, shape, ggml_dtype)

        out_path = os.path.join(output_dir, filename)
        writer.write(out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        summary.append((filename, len(tensors), size_mb))
        print(f"  -> {out_path} ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("LoRA Export Summary")
    print("=" * 60)
    print(f"{'File':<30} {'Tensors':>8} {'Size (MB)':>10}")
    print("-" * 60)
    total_tensors = 0
    total_size = 0.0
    for fname, ntens, size in summary:
        print(f"{fname:<30} {ntens:>8} {size:>10.1f}")
        total_tensors += ntens
        total_size += size
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_tensors:>8} {total_size:>10.1f}")
    print(f"\nLoRA output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export VoxCPM model to multi-file GGUF format"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to VoxCPM model directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for GGUF files")
    parser.add_argument("--lora-dir", default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--quant-lm", default="fp16", choices=["fp16", "q8", "q4"],
                        help="Quantization for LM components (default: fp16)")
    parser.add_argument("--quant-encoder", default="fp16", choices=["fp16", "q8", "q4"],
                        help="Quantization for encoder (default: fp16)")
    parser.add_argument("--quant-dit", default="fp16", choices=["fp16", "q8", "q4"],
                        help="Quantization for DiT decoder (default: fp16)")
    parser.add_argument("--quant-vae", default="fp16", choices=["fp16", "q8", "q4"],
                        help="Quantization for audio VAE (default: fp16)")

    args = parser.parse_args()

    quant_args = {
        "quant_lm": args.quant_lm,
        "quant_encoder": args.quant_encoder,
        "quant_dit": args.quant_dit,
        "quant_vae": args.quant_vae,
    }

    export(args.model_dir, args.output_dir, quant_args)

    if args.lora_dir:
        config_path = os.path.join(args.model_dir, "config.json")
        export_lora(args.lora_dir, args.output_dir, config_path)


if __name__ == "__main__":
    main()
