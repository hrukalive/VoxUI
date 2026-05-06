# VoxCPM Single-GGUF Export And Runtime Simplification

## Context

The current VoxCPM exporter writes a base model as multiple GGUF component files plus a manifest, tokenizer/config sidecars, and optional LoRA adapter directories containing multiple LoRA component files. The Rust inference engine mirrors that layout by opening separate GGUF loaders for `base_lm`, `residual_lm`, `feat_encoder`, `feat_decoder`, `audio_vae`, and `projections`.

This design removes backward compatibility with the multi-file layout. The new target is a base model represented by one GGUF tensor file and a LoRA adapter represented by one GGUF tensor file, while keeping tokenizer and model config files as sidecars.

## Goals

- Export each base VoxCPM model as one `model.gguf`.
- Export each LoRA adapter as one direct `lora_<name>.gguf`.
- Keep `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` as sidecar files.
- Simplify Rust inference so it opens one base GGUF tensor store and uses it for all model components.
- Simplify model discovery to directories containing `model.gguf`.
- Simplify LoRA discovery to direct files matching `lora_*.gguf`.
- Remove the runtime component-file contract and manifest-based component map.

## Non-Goals

- Backward compatibility with existing multi-GGUF model directories.
- Embedding tokenizer/config JSON into GGUF.
- Supporting old LoRA adapter directories in the new runtime.
- Changing VoxCPM synthesis semantics, request validation, or parity requirements.
- Multi-LoRA composition.

## Export Layout

A base export directory contains:

```text
model.gguf
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Optional LoRA adapters live directly beside the base model:

```text
lora_ft2.gguf
lora_some_voice.gguf
```

Base GGUF tensor names remain globally qualified and source-faithful:

- `base_lm.*`
- `residual_lm.*`
- `feat_encoder.*`
- `feat_decoder.*`
- `audio_vae.*`
- `fsq_layer.*`
- `enc_to_lm_proj.*`
- `lm_to_dit_proj.*`
- `res_to_dit_proj.*`
- `fusion_concat_proj.*`
- `stop_proj.*`
- `stop_head.*`

LoRA GGUF tensor names remain fully qualified target names ending in `.lora_A` and `.lora_B`.

## Exporter Contract

The base export command keeps the current shape:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16
```

It writes a single `model.gguf` plus sidecar tokenizer/config files. It does not write `manifest.json`, component GGUF files, or a component file map.

The exporter may still classify tensors logically for validation and quantization policy, but all tensors are written to one GGUF file. Required tensor checks remain strict. Missing required base tensors, unmapped tensors, duplicate tensor names, missing LoRA A/B pairs, and LoRA rank mismatches are hard errors.

Base GGUF metadata includes runtime-useful identifiers and audit fields:

```text
voxcpm.schema_version = 2
voxcpm.kind = "base"
voxcpm.architecture = "voxcpm" | "voxcpm2"
voxcpm.variant = "0.5" | "1.5" | "2.0"
voxcpm.quant_profile = "fp16" | "q4-lm" | "manual"
voxcpm.source_model_dir = ...
```

Detailed architecture shape and model parameters remain in `config.json`.

When `--lora-dir` is provided, the exporter writes one direct adapter file:

```text
lora_<name>.gguf
```

LoRA GGUF metadata includes:

```text
voxcpm.schema_version = 2
voxcpm.kind = "lora"
voxcpm.lora.name = "<name>"
voxcpm.lora.rank = <rank>
voxcpm.lora.alpha = <alpha>
voxcpm.architecture = "voxcpm" | "voxcpm2"
voxcpm.variant = "0.5" | "1.5" | "2.0"
voxcpm.lora.enabled_targets = ...
voxcpm.lora.target_modules = ...
```

## Runtime Loading

`VoxCPMEngine::load(model_dir, device)` requires:

```text
model_dir/model.gguf
model_dir/config.json
model_dir/tokenizer.json
```

The engine opens one GGUF tensor store for `model.gguf`, reads config/tokenizer sidecars, and builds every subsystem from the same store:

```text
BaseLM
ResidualLM
LocalEncoder
DiT
AudioVAE
FSQLayer
projections
```

The runtime will introduce a shared internal model store that provides:

- GGUF metadata access.
- Tensor lookup by global tensor name.
- `load_tensor_optimal(name)` on the configured device.
- Tensor caching so repeated access does not re-dequantize.

The implementation will use one coordinated model load pass. It may either eagerly preload required tensors or lazily load subsystem tensors, but both paths use the same shared loader/cache.

This removes:

- `ComponentFiles`.
- `BundleManifest::component_path()`.
- Six component file existence checks.
- Six component GGUF loaders.
- Component-path plumbing through the engine.

Load progress becomes coarse phases rather than file components: config/tokenizer, open GGUF, LM, residual LM, encoder, DiT, VAE/projections.

## Discovery

Model discovery scans model roots for directories containing:

```text
model.gguf
```

LoRA discovery scans the selected model directory for direct child files matching:

```text
lora_*.gguf
```

It returns a `None` option plus sorted LoRA entries. Display names come from `voxcpm.lora.name` metadata when available, otherwise from the filename stem with the `lora_` prefix removed.

`VoxCPMEngine::load_lora(path)` accepts a `.gguf` file path only. It loads one LoRA tensor store, validates the adapter metadata against the loaded base model architecture and variant, then loads all `.lora_A` / `.lora_B` tensor pairs from that same file.

## Tests

Exporter tests will be updated around the new file contract:

- Replace component partition tests with logical tensor classification and required-prefix validation tests.
- Replace manifest tests with GGUF metadata tests.
- Add a test that base export writes only one `.gguf` named `model.gguf`.
- Add a test that LoRA export writes one direct `lora_<name>.gguf`.
- Keep quantization profile tests.

Rust inference tests will move off the manifest/component map:

- Replace manifest-loader tests with model layout and model metadata tests.
- Update parity tests to load `models/*/model.gguf`.
- Update LoRA tests to discover direct `lora_*.gguf`.
- Keep runtime purity, request validation, and generation parity tests.

Desktop and TUI tests will assert:

- Model discovery returns directories containing `model.gguf`.
- LoRA discovery returns direct `lora_*.gguf` files.
- Loading a model directory without `model.gguf` fails clearly.

## Migration And Cleanup

Generated model directories will be regenerated into the clean target layout. Old component files are not part of the new runtime contract and will be removed from regenerated directories.

Cleanup scope:

- Remove exporter manifest generation.
- Remove runtime `ComponentFiles`.
- Remove component path resolution.
- Remove component-specific GGUF loading in the engine.
- Update `README.txt` export and verify commands.

## Acceptance Criteria

- Fresh base exports contain one base GGUF named `model.gguf` plus sidecar tokenizer/config files.
- Fresh LoRA exports contain one direct GGUF named `lora_<name>.gguf`.
- Rust runtime loads all model components from one shared base GGUF tensor store.
- Desktop and TUI model discovery use `model.gguf`.
- Desktop and TUI LoRA discovery use direct `lora_*.gguf` files.
- Loader, LoRA, and existing targeted inference/parity tests pass on CPU.
- CUDA compile path is checked when the crate supports CUDA in this workspace.
