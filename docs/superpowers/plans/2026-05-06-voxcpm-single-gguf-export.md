# VoxCPM Single-GGUF Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multi-GGUF VoxCPM export/runtime contract with one base `model.gguf` and one direct `lora_<name>.gguf` per adapter, while keeping tokenizer/config sidecars.

**Architecture:** The exporter keeps logical tensor classification for validation and quantization, but writes all base tensors into one GGUF and all LoRA tensors into one GGUF. Rust runtime reads `config.json` plus metadata from `model.gguf`, opens one shared GGUF-backed tensor store, and builds all model subsystems from it. Desktop and TUI discovery move from `manifest.json` and LoRA directories to `model.gguf` and direct `lora_*.gguf` files.

**Tech Stack:** Python exporter with NumPy/safetensors/PyTorch, custom GGUF writer/verifier, Rust 2021, Candle, Tauri desktop commands, Leptos/TUI app surfaces, PowerShell verification commands.

---

## File Structure

- Modify `D:/Sandbox_Share/VoxUI/exporter/export_voxcpm.py`: collapse base export to one `model.gguf`, collapse LoRA export to one `lora_<name>.gguf`, remove manifest writing, add GGUF metadata.
- Modify `D:/Sandbox_Share/VoxUI/exporter/tests/test_export_manifest.py`: replace manifest/component-file assertions with single-GGUF metadata and file-layout tests.
- Modify `D:/Sandbox_Share/VoxUI/exporter/verify_gguf.py`: keep directory verification, but make single `model.gguf` and `lora_*.gguf` outputs easy to inspect.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/manifest.rs`: replace bundle-manifest/component map with config-derived `ModelConfig`, `AudioVaeManifest`, and `ModelVariant`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/model_loader.rs`: add a cached shared `GgufTensorStore` and keep `GgufModelLoader` as the subsystem-facing API.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/base_lm.rs`: read `BaseLMConfig` from `ModelConfig` instead of `BundleManifest`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/encoder.rs`: load encoder config from `ModelConfig`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/dit.rs`: load DiT config from `ModelConfig`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/audiovae.rs`: load AudioVAE from `ModelConfig.audio_vae`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`: load `model.gguf`, remove component paths, build all components from one loader/store, validate GGUF metadata.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lora.rs`: load one `.gguf` LoRA file with adapter metadata and all tensor pairs.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/request.rs`: keep `ModelVariant` import working after manifest refactor.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`: update public exports after manifest refactor.
- Modify tests under `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/`: replace manifest/component paths with `model.gguf` and direct `lora_*.gguf`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs`: model discovery uses `model.gguf`, LoRA discovery uses direct `lora_*.gguf`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: pass LoRA file paths to `engine.load_lora`.
- Modify `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/main.rs`: startup model existence uses `model.gguf`.
- Modify `D:/Sandbox_Share/VoxUI/README.txt`: update export, verify, and test commands.

---

### Task 1: Exporter Single-GGUF Base Layout

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/exporter/export_voxcpm.py`
- Modify: `D:/Sandbox_Share/VoxUI/exporter/tests/test_export_manifest.py`

- [ ] **Step 1: Write failing exporter layout tests**

In `exporter/tests/test_export_manifest.py`, replace the current component-file assertions with tests that assert logical grouping still exists, but file output is single-GGUF. Add these imports:

```python
import json
import tempfile
from unittest.mock import patch
```

Add this test helper near the top of the file:

```python
class RecordingWriter:
    writes: list[Path] = []
    metadata: dict[str, object] = {}
    tensors: list[str] = []

    def __init__(self):
        self.metadata = {}
        self.tensors = []

    def add_metadata(self, key, value, value_type=None):
        self.metadata[key] = value

    def add_tensor(self, name, data, shape, dtype):
        self.tensors.append(name)

    def write(self, path):
        out = Path(path)
        out.write_bytes(b"GGUF-test")
        RecordingWriter.writes.append(out)
        RecordingWriter.metadata = dict(self.metadata)
        RecordingWriter.tensors = list(self.tensors)
```

Add these tests:

```python
def test_base_export_writes_single_model_gguf(self):
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
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 4096,
        },
    }
    main_weights = {
        "base_lm.norm.weight": np.zeros(2, dtype=np.float32),
        "base_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "residual_lm.norm.weight": np.zeros(2, dtype=np.float32),
        "residual_lm.layers.0.self_attn.q_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "feat_encoder.in_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "feat_encoder.special_token": np.zeros((1, 1, 1, 2), dtype=np.float32),
        "feat_decoder.input_embed.weight": np.zeros((2, 2), dtype=np.float32),
        "fsq_layer.project_in.weight": np.zeros((2, 2), dtype=np.float32),
        "enc_to_lm_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "lm_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "res_to_dit_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "stop_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "stop_head.weight": np.zeros((2, 2), dtype=np.float32),
    }
    vae_weights = {"decoder.model.0.weight_v": np.zeros((2, 2, 2), dtype=np.float32)}

    with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
        source_dir = Path(source)
        output_dir = Path(output)
        (source_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (source_dir / name).write_text("{}", encoding="utf-8")
        RecordingWriter.writes = []
        with patch("exporter.export_voxcpm.load_weights", return_value=(main_weights, vae_weights, "unit")):
            with patch("exporter.export_voxcpm.GGUFWriter", RecordingWriter):
                export(source_dir, output_dir, profile_default_quant_args("fp16", "2.0"), "2.0")

    self.assertEqual([path.name for path in RecordingWriter.writes], ["model.gguf"])
    self.assertEqual(RecordingWriter.metadata["voxcpm.schema_version"], 2)
    self.assertEqual(RecordingWriter.metadata["voxcpm.kind"], "base")
    self.assertEqual(RecordingWriter.metadata["voxcpm.variant"], "2.0")
    self.assertIn("base_lm.norm.weight", RecordingWriter.tensors)
    self.assertIn("audio_vae.decoder.model.0.weight_v", RecordingWriter.tensors)

def test_manifest_is_not_written_for_single_gguf_export(self):
    with tempfile.TemporaryDirectory() as output:
        output_dir = Path(output)
        self.assertFalse((output_dir / "manifest.json").exists())
```

Update the existing partition test name from `test_partition_uses_python_component_names` to `test_classification_uses_python_component_names` after implementing Step 3.

- [ ] **Step 2: Run exporter tests to verify failure**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: fail because `export` still writes multiple component files and `manifest.json`.

- [ ] **Step 3: Replace component filenames with logical component classes**

In `exporter/export_voxcpm.py`, replace `COMPONENT_FILES`, `QUANT_ARG_MAP`, and filename-returning `get_component_for_key` with this logical classification:

```python
BASE_MODEL_FILE = "model.gguf"

LOGICAL_COMPONENTS = (
    "base_lm",
    "residual_lm",
    "feat_encoder",
    "feat_decoder",
    "audio_vae",
    "projections",
)

QUANT_COMPONENT_MAP = {
    "base_lm": "quant_lm",
    "residual_lm": "quant_lm",
    "feat_encoder": "quant_encoder",
    "feat_decoder": "quant_dit",
    "audio_vae": "quant_vae",
    "projections": "quant_lm",
}

def classify_tensor_key(key: str):
    if key.startswith("base_lm."):
        return "base_lm", key
    if key.startswith("residual_lm."):
        return "residual_lm", key
    if key.startswith("feat_encoder."):
        return "feat_encoder", key
    if key.startswith("feat_decoder."):
        return "feat_decoder", key
    if key.startswith(PROJECTION_PREFIXES):
        return "projections", key
    return None, None
```

Update `partition_weights` so it returns `dict[str, list[tuple[str, Any]]]` keyed by logical component names:

```python
def partition_weights(main_weights: dict[str, Any], vae_weights: dict[str, Any] | None):
    buckets: dict[str, list[tuple[str, Any]]] = {}
    unmapped_keys: list[str] = []

    for key, tensor in main_weights.items():
        component, new_name = classify_tensor_key(key)
        if component is None or new_name is None:
            unmapped_keys.append(key)
            continue
        buckets.setdefault(component, []).append((new_name, tensor))

    if vae_weights:
        for key, tensor in vae_weights.items():
            buckets.setdefault("audio_vae", []).append((f"audio_vae.{key}", tensor))

    if unmapped_keys:
        sample = ", ".join(unmapped_keys[:10])
        raise ValueError(f"unmapped tensor keys ({len(unmapped_keys)}): {sample}")

    return buckets
```

Change `REQUIRED_PREFIXES` keys from filenames to logical names:

```python
REQUIRED_PREFIXES = {
    "base_lm": ["base_lm.norm.weight", "base_lm.layers.0.self_attn.q_proj.weight"],
    "residual_lm": ["residual_lm.norm.weight", "residual_lm.layers.0.self_attn.q_proj.weight"],
    "feat_encoder": ["feat_encoder.in_proj.weight", "feat_encoder.special_token"],
    "feat_decoder": ["feat_decoder."],
    "audio_vae": ["audio_vae."],
    "projections": [
        "fsq_layer.",
        "enc_to_lm_proj.weight",
        "lm_to_dit_proj.weight",
        "res_to_dit_proj.weight",
        "stop_proj.weight",
        "stop_head.weight",
    ],
}
```

- [ ] **Step 4: Write single base GGUF metadata and writer helpers**

In `exporter/export_voxcpm.py`, replace component metadata adders with a single base metadata function:

```python
def add_base_metadata(
    writer: GGUFWriter,
    *,
    config: dict[str, Any],
    variant: str,
    quant_profile: str,
    source_model_dir: Path,
    source_weight_format: str,
) -> None:
    writer.add_metadata("voxcpm.schema_version", 2)
    writer.add_metadata("voxcpm.kind", "base")
    writer.add_metadata("voxcpm.architecture", config.get("architecture", "voxcpm"))
    writer.add_metadata("voxcpm.variant", variant)
    writer.add_metadata("voxcpm.quant_profile", quant_profile)
    writer.add_metadata("voxcpm.source_model_dir", str(source_model_dir.resolve()))
    writer.add_metadata("voxcpm.source_weight_format", source_weight_format)
```

Add a helper that writes all logical buckets into one file:

```python
def write_base_gguf(
    *,
    output_dir: Path,
    buckets: dict[str, list[tuple[str, Any]]],
    config: dict[str, Any],
    quant_args: dict[str, str],
    variant: str,
    quant_profile: str,
    model_dir: Path,
    source_weight_format: str,
) -> dict[str, str]:
    writer = GGUFWriter()
    add_base_metadata(
        writer,
        config=config,
        variant=variant,
        quant_profile=quant_profile,
        source_model_dir=model_dir,
        source_weight_format=source_weight_format,
    )
    component_quantization: dict[str, str] = {}
    for component in LOGICAL_COMPONENTS:
        tensors = buckets.get(component, [])
        if not tensors:
            continue
        quant_key = QUANT_COMPONENT_MAP.get(component, "quant_lm")
        quant_name = quant_args.get(quant_key, "fp16")
        component_quantization[component] = quant_name
        quant_fn, ggml_dtype = QUANT_MAP[quant_name]
        print(f"Adding {component} ({len(tensors)} tensors, quant={quant_name})")
        writer.add_metadata(f"voxcpm.quantization.{component}", quant_name)
        for tensor_name, tensor in tensors:
            arr = tensor_to_f32_numpy(tensor)
            writer.add_tensor(tensor_name, quant_fn(arr), list(arr.shape), ggml_dtype)
    writer.write(str(output_dir / BASE_MODEL_FILE))
    return component_quantization
```

- [ ] **Step 5: Update export entrypoint**

Change the `export` signature to accept `quant_profile`:

```python
def export(
    model_dir: str | Path,
    output_dir: str | Path,
    quant_args: dict[str, str],
    variant: str,
    quant_profile: str = "manual",
) -> dict[str, Any]:
```

Replace the loop that writes each component and the `manifest.json` write with:

```python
output_dir.mkdir(parents=True, exist_ok=True)
component_quantization = write_base_gguf(
    output_dir=output_dir,
    buckets=buckets,
    config=config,
    quant_args=quant_args,
    variant=variant,
    quant_profile=quant_profile,
    model_dir=model_dir,
    source_weight_format=source_weight_format,
)
copy_bundle_files(model_dir, output_dir)
return {
    "schema_version": 2,
    "architecture": config.get("architecture", "voxcpm"),
    "variant": variant,
    "model_file": BASE_MODEL_FILE,
    "quantization": component_quantization,
}
```

Update `main()` to pass the profile:

```python
export(args.model_dir, args.output_dir, quant_args, args.variant, args.quant_profile)
```

- [ ] **Step 6: Update tests for logical bucket keys**

In `exporter/tests/test_export_manifest.py`, update assertions:

```python
self.assertIn("base_lm", buckets)
self.assertIn("residual_lm", buckets)
self.assertIn("feat_encoder", buckets)
self.assertIn("feat_decoder", buckets)
self.assertIn("projections", buckets)
names = {name for name, _ in buckets["feat_encoder"]}
self.assertIn("feat_encoder.in_proj.weight", names)
```

Update the missing-required test bucket:

```python
buckets = {
    "base_lm": [("base_lm.norm.weight", np.zeros(2, dtype=np.float32))],
}
```

Remove the old `test_manifest_records_component_files_and_special_tokens` test or rewrite it to inspect metadata through `RecordingWriter`.

- [ ] **Step 7: Run exporter tests**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: `OK`.

- [ ] **Step 8: Commit exporter base layout**

Run:

```powershell
git add exporter/export_voxcpm.py exporter/tests/test_export_manifest.py
git commit -m "feat(exporter): write VoxCPM base model as single GGUF"
```

Expected: commit succeeds and does not stage unrelated Rust or desktop changes.

---

### Task 2: Exporter Single-GGUF LoRA Layout

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/exporter/export_voxcpm.py`
- Modify: `D:/Sandbox_Share/VoxUI/exporter/tests/test_export_manifest.py`

- [ ] **Step 1: Add failing LoRA single-file test**

In `exporter/tests/test_export_manifest.py`, add:

```python
def test_lora_export_writes_one_direct_gguf(self):
    lora_config = {
        "lora_config": {
            "r": 8,
            "alpha": 16,
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "target_modules_lm": ["q_proj"],
            "target_modules_dit": ["q_proj"],
            "target_proj_modules": [],
        }
    }
    config = {"architecture": "voxcpm2"}
    lora_weights = {
        "base_lm.layers.0.self_attn.q_proj.lora_A": np.zeros((8, 4), dtype=np.float32),
        "base_lm.layers.0.self_attn.q_proj.lora_B": np.zeros((4, 8), dtype=np.float32),
        "feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.lora_A": np.zeros((8, 4), dtype=np.float32),
        "feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.lora_B": np.zeros((4, 8), dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as lora, tempfile.TemporaryDirectory() as output:
        lora_dir = Path(lora) / "ft_unit"
        lora_dir.mkdir()
        output_dir = Path(output)
        config_path = Path(lora) / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (lora_dir / "lora_config.json").write_text(json.dumps(lora_config), encoding="utf-8")
        (lora_dir / "lora_weights.safetensors").write_bytes(b"placeholder")
        RecordingWriter.writes = []
        with patch("safetensors.torch.load_file", return_value=lora_weights):
            with patch("exporter.export_voxcpm.GGUFWriter", RecordingWriter):
                manifest = export_lora(lora_dir, output_dir, config_path, "2.0")

    self.assertEqual([path.name for path in RecordingWriter.writes], ["lora_ft_unit.gguf"])
    self.assertEqual(manifest["file"], "lora_ft_unit.gguf")
    self.assertEqual(RecordingWriter.metadata["voxcpm.kind"], "lora")
    self.assertEqual(RecordingWriter.metadata["voxcpm.lora.rank"], 8)
    self.assertIn("base_lm.layers.0.self_attn.q_proj.lora_A", RecordingWriter.tensors)
```

- [ ] **Step 2: Run exporter tests to verify failure**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: fail because `export_lora` still writes an adapter subdirectory, `lora_manifest.json`, and multiple component files.

- [ ] **Step 3: Replace LoRA component buckets with one writer**

In `exporter/export_voxcpm.py`, replace `_safe_lora_dir_name` with:

```python
def _safe_lora_name(lora_dir: Path) -> str:
    name = lora_dir.name
    if name == "latest" and lora_dir.parent.name:
        name = lora_dir.parent.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "adapter"
    return name

def _safe_lora_file_name(lora_dir: Path) -> str:
    return f"lora_{_safe_lora_name(lora_dir)}.gguf"
```

Replace `_lora_key_transform` with a function that validates known prefixes but does not bucket by component:

```python
def _validate_lora_key(key: str) -> str:
    if key.startswith(("base_lm.", "residual_lm.", "feat_decoder.")):
        return key
    if key.startswith(PROJECTION_PREFIXES):
        return key
    raise ValueError(f"unmapped LoRA tensor key: {key}")
```

Add pair validation:

```python
def validate_lora_pairs(tensor_names: list[str], rank: int) -> None:
    a_targets = {name.removesuffix(".lora_A") for name in tensor_names if name.endswith(".lora_A")}
    b_targets = {name.removesuffix(".lora_B") for name in tensor_names if name.endswith(".lora_B")}
    missing_b = sorted(a_targets - b_targets)
    missing_a = sorted(b_targets - a_targets)
    if missing_b:
        raise ValueError(f"missing lora_B tensors for {missing_b[:5]}")
    if missing_a:
        raise ValueError(f"missing lora_A tensors for {missing_a[:5]}")
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
```

- [ ] **Step 4: Rewrite `export_lora`**

Replace the body after loading `lora_weights` with:

```python
lora_name = _safe_lora_name(lora_dir)
filename = _safe_lora_file_name(lora_dir)
output_dir.mkdir(parents=True, exist_ok=True)

tensors: list[tuple[str, Any]] = []
for key, tensor in lora_weights.items():
    tensors.append((_validate_lora_key(key), tensor))

rank = int(lc.get("r", 0))
alpha = int(lc.get("alpha", lc.get("r", 0)))
validate_lora_pairs([name for name, _ in tensors], rank)

writer = GGUFWriter()
writer.add_metadata("voxcpm.schema_version", 2)
writer.add_metadata("voxcpm.kind", "lora")
writer.add_metadata("voxcpm.architecture", config.get("architecture", "voxcpm"))
writer.add_metadata("voxcpm.variant", variant)
writer.add_metadata("voxcpm.lora.name", lora_name)
writer.add_metadata("voxcpm.lora.rank", rank)
writer.add_metadata("voxcpm.lora.alpha", alpha)
writer.add_metadata("voxcpm.lora.enabled_targets", json.dumps({
    "lm": bool(lc.get("enable_lm", False)),
    "dit": bool(lc.get("enable_dit", False)),
    "projections": bool(lc.get("enable_proj", False)),
}, ensure_ascii=False))
writer.add_metadata("voxcpm.lora.target_modules", json.dumps({
    "lm": lc.get("target_modules_lm", []),
    "dit": lc.get("target_modules_dit", []),
    "projections": lc.get("target_proj_modules", []),
}, ensure_ascii=False))

for tensor_name, tensor in sorted(tensors):
    arr = tensor_to_f32_numpy(tensor)
    writer.add_tensor(tensor_name, quantize_fp16(arr), list(arr.shape), GGML_TYPE_F16)
writer.write(str(output_dir / filename))

return {
    "schema_version": 2,
    "architecture": config.get("architecture", "voxcpm"),
    "variant": variant,
    "source_lora_dir": str(lora_dir.resolve()),
    "name": lora_name,
    "file": filename,
    "rank": rank,
    "alpha": alpha,
}
```

Remove `lora_manifest.json` and `lora_config.json` copies from `export_lora`.

- [ ] **Step 5: Run exporter tests**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: `OK`.

- [ ] **Step 6: Commit exporter LoRA layout**

Run:

```powershell
git add exporter/export_voxcpm.py exporter/tests/test_export_manifest.py
git commit -m "feat(exporter): write VoxCPM LoRA as single GGUF"
```

Expected: commit succeeds.

---

### Task 3: Rust Config Loader Without Component Manifest

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/manifest.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/request.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/manifest_loader.rs`

- [ ] **Step 1: Replace manifest tests with config/layout tests**

In `voxui/crates/voxui-inference/tests/manifest_loader.rs`, replace the file contents with:

```rust
use std::fs;

use voxui_inference::{ModelConfig, ModelVariant};

#[test]
fn model_config_parses_variant_and_audio_vae_from_config_json() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
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
                "encoder_rates": [2,5,8,8],
                "decoder_rates": [8,6,5,2,2,2]
            },
            "lm_config": {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16},
            "encoder_config": {},
            "dit_config": {}
        }"#,
    )
    .unwrap();

    let config = ModelConfig::load(dir.path(), ModelVariant::VoxCpm2).unwrap();
    assert_eq!(config.variant, ModelVariant::VoxCpm2);
    assert_eq!(config.architecture, "voxcpm2");
    assert_eq!(config.special_tokens.audio_start, 101);
    assert_eq!(config.special_tokens.ref_audio_start, Some(103));
    assert_eq!(config.output_sample_rate(), 48000);
}

#[test]
fn model_config_rejects_v2_variant_with_non_v2_architecture() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
            "architecture": "voxcpm",
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0,
            "audio_vae_config": {"sample_rate":16000,"latent_dim":64,"chunk_size":20,"decode_chunk_size":240},
            "lm_config": {},
            "encoder_config": {},
            "dit_config": {}
        }"#,
    )
    .unwrap();

    let err = ModelConfig::load(dir.path(), ModelVariant::VoxCpm2).unwrap_err();
    assert!(err.to_string().contains("VoxCPM2 variant requires voxcpm2 architecture"));
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test manifest_loader
```

Expected: fail because `ModelConfig` does not exist.

- [ ] **Step 3: Replace manifest structs with config structs**

In `voxui/crates/voxui-inference/src/manifest.rs`, keep `ModelVariant`, `SpecialTokens`, and `AudioVaeManifest`, delete `ComponentFiles` and `BundleManifest`, and add:

```rust
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub schema_version: u32,
    pub architecture: String,
    pub variant: ModelVariant,
    pub special_tokens: SpecialTokens,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: f32,
    pub audio_vae: AudioVaeManifest,
    pub lm_config: serde_json::Value,
    pub encoder_config: serde_json::Value,
    pub dit_config: serde_json::Value,
    pub residual_lm_num_layers: Option<usize>,
    pub residual_lm_no_rope: Option<bool>,
}
```

Add `ModelConfig::load`:

```rust
impl ModelConfig {
    pub fn load(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let architecture = config
            .get("architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("voxcpm")
            .to_string();
        let special_tokens = match variant {
            ModelVariant::VoxCpm2 => SpecialTokens {
                audio_start: 101,
                audio_end: 102,
                ref_audio_start: Some(103),
                ref_audio_end: Some(104),
            },
            _ => SpecialTokens {
                audio_start: 101,
                audio_end: 102,
                ref_audio_start: None,
                ref_audio_end: None,
            },
        };
        let audio_vae = audio_vae_from_config(&config);
        let model = Self {
            schema_version: 2,
            architecture,
            variant,
            special_tokens,
            patch_size: value_usize(&config, "patch_size", 4),
            feat_dim: value_usize(&config, "feat_dim", 64),
            scalar_quantization_latent_dim: value_usize(
                &config,
                "scalar_quantization_latent_dim",
                if variant == ModelVariant::VoxCpm2 { 512 } else { 256 },
            ),
            scalar_quantization_scale: config
                .get("scalar_quantization_scale")
                .and_then(|v| v.as_f64())
                .unwrap_or(9.0) as f32,
            audio_vae,
            lm_config: config.get("lm_config").cloned().unwrap_or_default(),
            encoder_config: config.get("encoder_config").cloned().unwrap_or_default(),
            dit_config: config.get("dit_config").cloned().unwrap_or_default(),
            residual_lm_num_layers: config
                .get("residual_lm_num_layers")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            residual_lm_no_rope: config.get("residual_lm_no_rope").and_then(|v| v.as_bool()),
        };
        model.validate()?;
        Ok(model)
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.audio_vae
            .out_sample_rate
            .unwrap_or(self.audio_vae.sample_rate)
    }

    fn validate(&self) -> Result<()> {
        if self.variant == ModelVariant::VoxCpm2 && self.architecture != "voxcpm2" {
            bail!("VoxCPM2 variant requires voxcpm2 architecture");
        }
        if self.variant != ModelVariant::VoxCpm2 && self.architecture == "voxcpm2" {
            bail!("voxcpm2 architecture requires VoxCPM2 variant");
        }
        Ok(())
    }
}
```

Add helpers:

```rust
fn value_usize(config: &serde_json::Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn array_usize(config: &serde_json::Value, key: &str) -> Vec<usize> {
    config
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default()
}

fn audio_vae_from_config(config: &serde_json::Value) -> AudioVaeManifest {
    let vae = config.get("audio_vae_config").unwrap_or(&serde_json::Value::Null);
    let architecture = config
        .get("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("voxcpm");
    AudioVaeManifest {
        encoder_dim: vae.get("encoder_dim").and_then(|v| v.as_u64()).map(|v| v as usize),
        decoder_dim: vae.get("decoder_dim").and_then(|v| v.as_u64()).map(|v| v as usize),
        sample_rate: vae
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(if architecture == "voxcpm2" { 16000 } else { 44100 }) as u32,
        out_sample_rate: vae
            .get("out_sample_rate")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        latent_dim: vae
            .get("latent_dim")
            .and_then(|v| v.as_u64())
            .or_else(|| config.get("feat_dim").and_then(|v| v.as_u64()))
            .unwrap_or(64) as usize,
        chunk_size: vae.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(20) as usize,
        decode_chunk_size: vae
            .get("decode_chunk_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(240) as usize,
        encoder_rates: array_usize(vae, "encoder_rates"),
        decoder_rates: array_usize(vae, "decoder_rates"),
    }
}
```

- [ ] **Step 4: Update public exports and request import**

In `voxui/crates/voxui-inference/src/lib.rs`, replace:

```rust
pub use manifest::{BundleManifest, ComponentFiles, ModelVariant};
```

with:

```rust
pub use manifest::{AudioVaeManifest, ModelConfig, ModelVariant, SpecialTokens};
```

In `request.rs`, keep:

```rust
use crate::manifest::ModelVariant;
```

- [ ] **Step 5: Run manifest tests**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test manifest_loader
```

Expected: `OK` for `manifest_loader`, while other inference tests may still fail until later tasks.

- [ ] **Step 6: Commit config loader**

Run:

```powershell
git add voxui/crates/voxui-inference/src/manifest.rs voxui/crates/voxui-inference/src/request.rs voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/tests/manifest_loader.rs
git commit -m "refactor(inference): load VoxCPM config without component manifest"
```

Expected: commit succeeds.

---

### Task 4: Shared Cached GGUF Tensor Store

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/model_loader.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Test: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/manifest_loader.rs`

- [ ] **Step 1: Add layout test for `model.gguf` requirement**

In `voxui/crates/voxui-inference/tests/manifest_loader.rs`, add:

```rust
use voxui_inference::GgufModelLoader;

#[test]
fn model_loader_requires_model_gguf_in_directory() {
    let dir = tempfile::tempdir().unwrap();
    let err = GgufModelLoader::from_model_dir(dir.path(), candle_core::Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("model.gguf"));
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test manifest_loader
```

Expected: fail because `GgufModelLoader::from_model_dir` does not exist.

- [ ] **Step 3: Add cached store implementation**

In `model_loader.rs`, replace the struct fields with an `Arc` store:

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use voxui_gguf::{GgufFile, MetadataValue, TensorInfo};

struct GgufTensorStore {
    gguf: GgufFile,
    cache: Mutex<HashMap<String, Tensor>>,
    path: PathBuf,
}

#[derive(Clone)]
pub struct GgufModelLoader {
    store: Arc<GgufTensorStore>,
    device: Device,
}
```

Update constructors:

```rust
impl GgufModelLoader {
    pub fn new(path: &Path, device: Device) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Ok(Self {
            store: Arc::new(GgufTensorStore {
                gguf,
                cache: Mutex::new(HashMap::new()),
                path: path.to_path_buf(),
            }),
            device,
        })
    }

    pub fn from_model_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let path = model_dir.join("model.gguf");
        if !path.is_file() {
            anyhow::bail!("missing model.gguf at {}", path.display());
        }
        Self::new(&path, device)
    }
```

Update `load_tensor` to use the cache:

```rust
    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        if let Some(tensor) = self
            .store
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF tensor cache lock poisoned"))?
            .get(name)
            .cloned()
        {
            return Ok(tensor);
        }
        let info = self
            .store
            .gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found in GGUF file", name))?;
        let data = self.store.gguf.tensor_f32(name)?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let tensor = Tensor::from_vec(data, shape.as_slice(), &self.device)
            .with_context(|| format!("load tensor `{name}` from {}", self.store.path.display()))?;
        self.store
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF tensor cache lock poisoned"))?
            .insert(name.to_string(), tensor.clone());
        Ok(tensor)
    }
```

Update metadata/tensor methods to use `self.store.gguf`.

- [ ] **Step 4: Run layout test**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test manifest_loader
```

Expected: `OK`.

- [ ] **Step 5: Commit shared loader**

Run:

```powershell
git add voxui/crates/voxui-inference/src/model_loader.rs voxui/crates/voxui-inference/tests/manifest_loader.rs
git commit -m "refactor(inference): add shared cached GGUF model loader"
```

Expected: commit succeeds.

---

### Task 5: Load Engine Components From One `model.gguf`

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/base_lm.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/encoder.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/dit.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/audiovae.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Modify tests under `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/`

- [ ] **Step 1: Update tests to expect `model.gguf`**

In these tests, replace `manifest.component_path(&model_dir, "...")` and `BundleManifest::load` usage with `ModelConfig::load` and `GgufModelLoader::from_model_dir`:

`audiovae_parity.rs` helper becomes:

```rust
fn load_voxcpm2_vae(root: &Path) -> AudioVAE {
    let model_dir = root.join("models/voxcpm2-fp16");
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
    AudioVAE::load_from_config(&loader, &config.audio_vae).unwrap()
}
```

Update imports:

```rust
use voxui_inference::{AudioVAE, GgufModelLoader, ModelConfig, ModelVariant};
```

In `dit_parity.rs`, change loader creation to:

```rust
let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
let config = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
let dit = DiT::load_from_config(&loader, &config).unwrap();
```

In `local_encoder_parity.rs`, use:

```rust
let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
let config = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
let encoder = LocalEncoder::load_from_config(&loader, &config).unwrap();
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test audiovae_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test dit_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test local_encoder_parity -- --nocapture
```

Expected: fail because `load_from_config` methods do not exist and local model dirs still need regeneration in Task 9.

- [ ] **Step 3: Update BaseLM config constructor**

In `base_lm.rs`, replace:

```rust
use crate::manifest::BundleManifest;
```

with:

```rust
use crate::manifest::ModelConfig;
```

Rename `BaseLMConfig::from_manifest` to:

```rust
pub fn from_model_config(model: &ModelConfig, component: &str) -> Result<Self>
```

Replace every `manifest.` reference in that function with `model.`. Keep the component behavior unchanged:

```rust
let cfg = match component {
    "base_lm" | "residual_lm" => &model.lm_config,
    "feat_encoder" => &model.encoder_config,
    other => bail!("unsupported MiniCPM component `{other}`"),
};
```

- [ ] **Step 4: Update encoder/DiT/AudioVAE loaders**

In `encoder.rs`, replace `load_from_manifest` with:

```rust
pub fn load_from_config(loader: &GgufModelLoader, model: &crate::ModelConfig) -> Result<Self> {
    let config = BaseLMConfig::from_model_config(model, "feat_encoder")?;
    Self::load(loader, config, loader.device())
}
```

In `dit.rs`, replace `load_from_manifest` with:

```rust
pub fn load_from_config(loader: &GgufModelLoader, model: &crate::ModelConfig) -> Result<Self> {
    let dit = &model.dit_config;
    let lm = &model.lm_config;
    let hidden_dim = get_usize(dit, &["hidden_dim", "hidden_size"], 1024);
    let num_heads = get_usize(dit, &["num_heads", "num_attention_heads"], 16);
    let head_dim = dit
        .get("kv_channels")
        .or_else(|| lm.get("kv_channels"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_dim / num_heads);
    let num_kv_heads = lm
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(num_heads);
    let rope_scaling = lm.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
    let cfm = dit.get("cfm_config").unwrap_or(&serde_json::Value::Null);
    let config = DiTConfig {
        prefix: "feat_decoder.estimator".to_string(),
        hidden_dim,
        num_layers: get_usize(dit, &["num_layers", "num_hidden_layers"], 12),
        num_heads,
        num_kv_heads,
        head_dim,
        ffn_dim: get_usize(dit, &["ffn_dim", "intermediate_size"], 4096),
        rms_norm_eps: get_f64(lm, "rms_norm_eps", 1e-5),
        scale_depth: get_f64(lm, "scale_depth", 1.0),
        use_mup: lm.get("use_mup").and_then(|v| v.as_bool()).unwrap_or(false),
        rope_theta: get_f64(lm, "rope_theta", 10000.0),
        original_max_position_embeddings: rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        rope_short_factors: read_f32_array(rope_scaling, "short_factor", head_dim / 2),
        rope_long_factors: read_f32_array(rope_scaling, "long_factor", head_dim / 2),
        cfg_value: get_f64(cfm, "inference_cfg_rate", 1.0),
        n_steps: 10,
        sway_coef: get_f64(dit, "sway_sampling_coef", 1.0),
        latent_dim: model.feat_dim,
    };
    Self::load(loader, config, loader.device())
}
```

In `audiovae.rs`, rename `load_from_manifest` to:

```rust
pub fn load_from_config(loader: &GgufModelLoader, manifest: &AudioVaeManifest) -> Result<Self>
```

Keep the function body unchanged.

- [ ] **Step 5: Update engine load**

In `engine.rs`, replace manifest loading:

```rust
let manifest = BundleManifest::load(model_dir)?;
```

with:

```rust
let base_loader = GgufModelLoader::from_model_dir(model_dir, device.clone())?;
let variant = read_variant_from_loader(&base_loader)?;
let manifest = ModelConfig::load(model_dir, variant)?;
validate_base_metadata(&base_loader, &manifest)?;
```

Add helpers near the bottom of `engine.rs`:

```rust
fn read_variant_from_loader(loader: &GgufModelLoader) -> Result<ModelVariant> {
    let value = loader
        .metadata()
        .get("voxcpm.variant")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("model.gguf missing voxcpm.variant metadata"))?;
    match value {
        "0.5" => Ok(ModelVariant::VoxCpm05),
        "1.5" => Ok(ModelVariant::VoxCpm15),
        "2.0" => Ok(ModelVariant::VoxCpm2),
        other => anyhow::bail!("unsupported voxcpm.variant `{other}`"),
    }
}

fn validate_base_metadata(loader: &GgufModelLoader, model: &ModelConfig) -> Result<()> {
    let kind = loader
        .metadata()
        .get("voxcpm.kind")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if kind != "base" {
        anyhow::bail!("model.gguf voxcpm.kind must be `base`, got `{kind}`");
    }
    let architecture = loader
        .metadata()
        .get("voxcpm.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if architecture != model.architecture {
        anyhow::bail!(
            "model.gguf architecture `{}` does not match config `{}`",
            architecture,
            model.architecture
        );
    }
    Ok(())
}
```

Replace component path/load blocks with one shared loader:

```rust
let base_lm_config = BaseLMConfig::from_model_config(&manifest, "base_lm")?;
let base_lm = BaseLM::load(&base_loader, base_lm_config, &device)?;

let residual_lm_config = BaseLMConfig::from_model_config(&manifest, "residual_lm")?;
let residual_lm = BaseLM::load(&base_loader, residual_lm_config, &device)?;

let encoder = LocalEncoder::load_from_config(&base_loader, &manifest)?;
let dit = DiT::load_from_config(&base_loader, &manifest)?;
let vae = AudioVAE::load_from_config(&base_loader, &manifest.audio_vae)?;

let fsq = FSQLayer::load(
    &base_loader,
    manifest.scalar_quantization_latent_dim,
    manifest.scalar_quantization_scale as f64,
)?;
let lm_to_dit_proj = load_projection(&base_loader, "lm_to_dit_proj")?;
let res_to_dit_proj = load_projection(&base_loader, "res_to_dit_proj")?;
let enc_to_lm_proj = load_projection(&base_loader, "enc_to_lm_proj")?;
let fusion_concat_proj = if base_loader.has_tensor("fusion_concat_proj.weight") {
    Some(load_projection(&base_loader, "fusion_concat_proj")?)
} else {
    None
};
let stop_proj = load_projection(&base_loader, "stop_proj")?;
let stop_head = load_projection(&base_loader, "stop_head")?;
```

Keep the field name `manifest` in `VoxCPMEngine` if minimizing churn, but change its type to `ModelConfig`.

Update `check_cancel` total steps from `6` to `7` and call progress between the logical phases.

- [ ] **Step 6: Update remaining `BundleManifest` references**

Run:

```powershell
rg -n "BundleManifest|ComponentFiles|component_path|load_from_manifest|from_manifest" voxui/crates/voxui-inference
```

Replace each match with the corresponding `ModelConfig`, `load_from_config`, or `from_model_config` API from this task.

- [ ] **Step 7: Run cargo check**

Run:

```powershell
cd voxui; cargo check -p voxui-inference
```

Expected: pass after code references are updated. Parity tests may still fail until models are regenerated.

- [ ] **Step 8: Commit single-loader runtime**

Run:

```powershell
git add voxui/crates/voxui-inference/src/base_lm.rs voxui/crates/voxui-inference/src/encoder.rs voxui/crates/voxui-inference/src/dit.rs voxui/crates/voxui-inference/src/audiovae.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/tests
git commit -m "refactor(inference): load VoxCPM components from one GGUF"
```

Expected: commit succeeds.

---

### Task 6: Single-File LoRA Runtime

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lora.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/lora_parity.rs`

- [ ] **Step 1: Update LoRA tests for direct GGUF files**

In `lora_parity.rs`, replace adapter discovery:

```rust
let lora_file = std::fs::read_dir(&model_dir)
    .unwrap()
    .flatten()
    .map(|e| e.path())
    .find(|p| {
        p.is_file()
            && p.extension().and_then(|v| v.to_str()) == Some("gguf")
            && p.file_stem()
                .map(|s| s.to_string_lossy().starts_with("lora_"))
                .unwrap_or(false)
    });
let Some(lora_file) = lora_file else {
    eprintln!("skip: no single-file LoRA adapter exported");
    return;
};
```

Change load call:

```rust
engine.load_lora(&lora_file).unwrap();
```

- [ ] **Step 2: Run LoRA tests to verify failure**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: fail because `LoraAdapter::load_from_dir_for_model` still expects old directory manifests.

- [ ] **Step 3: Replace LoRA manifest structs**

In `lora.rs`, delete `LoraManifest` fields that describe component files. Replace with:

```rust
#[derive(Debug)]
struct LoraMetadata {
    architecture: String,
    variant: ModelVariant,
    name: String,
    rank: usize,
    alpha: f32,
    target_modules: LoraTargetModules,
}
```

Add metadata loader:

```rust
impl LoraMetadata {
    fn from_loader(loader: &GgufModelLoader) -> Result<Self> {
        let metadata = loader.metadata();
        let kind = metadata
            .get("voxcpm.kind")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if kind != "lora" {
            bail!("LoRA GGUF voxcpm.kind must be `lora`, got `{kind}`");
        }
        let architecture = metadata
            .get("voxcpm.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("LoRA GGUF missing voxcpm.architecture"))?
            .to_string();
        let variant = match metadata
            .get("voxcpm.variant")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("LoRA GGUF missing voxcpm.variant"))?
        {
            "0.5" => ModelVariant::VoxCpm05,
            "1.5" => ModelVariant::VoxCpm15,
            "2.0" => ModelVariant::VoxCpm2,
            other => bail!("unsupported LoRA variant `{other}`"),
        };
        let name = metadata
            .get("voxcpm.lora.name")
            .and_then(|v| v.as_str())
            .unwrap_or("adapter")
            .to_string();
        let rank = metadata
            .get("voxcpm.lora.rank")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize;
        let alpha = metadata
            .get("voxcpm.lora.alpha")
            .and_then(|v| v.as_f32())
            .unwrap_or(rank as f32);
        let target_modules = metadata
            .get("voxcpm.lora.target_modules")
            .and_then(|v| v.as_str())
            .and_then(|text| serde_json::from_str::<LoraTargetModules>(text).ok())
            .unwrap_or_default();
        Ok(Self {
            architecture,
            variant,
            name,
            rank,
            alpha,
            target_modules,
        })
    }

    fn validate(&self, model: &ModelConfig) -> Result<()> {
        if self.rank == 0 {
            bail!("LoRA rank must be positive");
        }
        if self.alpha <= 0.0 {
            bail!("LoRA alpha must be positive");
        }
        if self.architecture != model.architecture {
            bail!(
                "LoRA architecture `{}` does not match model `{}`",
                self.architecture,
                model.architecture
            );
        }
        if self.variant != model.variant {
            bail!("LoRA variant {:?} does not match model {:?}", self.variant, model.variant);
        }
        Ok(())
    }
}
```

Import `ModelConfig`:

```rust
use crate::manifest::{ModelConfig, ModelVariant};
```

- [ ] **Step 4: Replace `load_from_dir_for_model` with file loader**

In `impl LoraAdapter`, replace directory loaders with:

```rust
pub fn load_file_for_model(path: &Path, device: &Device, model: &ModelConfig) -> Result<Self> {
    if path.extension().and_then(|v| v.to_str()) != Some("gguf") {
        bail!("LoRA path must be a .gguf file: {}", path.display());
    }
    let loader = GgufModelLoader::new(path, device.clone())?;
    let metadata = LoraMetadata::from_loader(&loader)?;
    metadata.validate(model)?;
    let mut adapter = Self {
        layers: HashMap::new(),
        alpha: metadata.alpha,
        rank: metadata.rank,
    };
    adapter.load_component(&loader, Some(&metadata.target_modules))?;
    adapter.validate_non_empty()?;
    Ok(adapter)
}
```

Keep `pub fn load(loader: &GgufModelLoader) -> Result<Self>` only for tests that already pass a loader directly.

- [ ] **Step 5: Update engine LoRA load**

In `engine.rs`, replace:

```rust
self.lora = Some(LoraAdapter::load_from_dir_for_model(path, &self.device, &self.manifest)?);
```

with:

```rust
self.lora = Some(LoraAdapter::load_file_for_model(path, &self.device, &self.manifest)?);
```

- [ ] **Step 6: Remove old directory fallback**

In `lora.rs`, delete the branch that scans directories for `lora_*.gguf` and `lora_manifest.json`. The new runtime accepts `.gguf` paths only.

- [ ] **Step 7: Run LoRA tests**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: first formula test passes. Adapter generation test may skip until Task 9 regenerates models with direct LoRA GGUF files.

- [ ] **Step 8: Commit LoRA runtime**

Run:

```powershell
git add voxui/crates/voxui-inference/src/lora.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/tests/lora_parity.rs
git commit -m "refactor(inference): load LoRA adapters from single GGUF files"
```

Expected: commit succeeds.

---

### Task 7: Desktop And TUI Discovery Migration

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/main.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/config.rs`

- [ ] **Step 1: Update desktop discovery tests**

In `desktop_core.rs`, change the test helper:

```rust
fn create_model_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = root.join(name);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("model.gguf"), b"placeholder").unwrap();
    path
}
```

Replace calls to `create_manifest_dir` with `create_model_dir`.

Replace `scan_lora_entries_includes_none_and_manifest_dirs_sorted` with:

```rust
#[test]
fn scan_lora_entries_includes_none_and_direct_gguf_files_sorted() {
    let tmp = tempdir().unwrap();
    let model = create_model_dir(tmp.path(), "voxcpm2-fp16");
    fs::write(model.join("lora_b.gguf"), b"placeholder").unwrap();
    fs::write(model.join("lora_a.gguf"), b"placeholder").unwrap();
    fs::create_dir_all(model.join("lora_old_dir")).unwrap();
    fs::write(model.join("not_lora.gguf"), b"placeholder").unwrap();

    let entries = super::scan_lora_entries(&model);

    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0], super::LoraEntry::none());
    assert_eq!(entries[1].name, "a");
    assert_eq!(entries[2].name, "b");
    assert!(entries[1].path.as_ref().unwrap().ends_with("lora_a.gguf"));
}
```

- [ ] **Step 2: Run desktop core tests to verify failure**

Run:

```powershell
cd voxui; cargo test -p voxui-desktop --manifest-path crates/voxui-desktop/src-tauri/Cargo.toml desktop_core -- --nocapture
```

Expected: fail because discovery still uses `manifest.json` and LoRA directories.

- [ ] **Step 3: Update model discovery**

In `desktop_core.rs`, change `scan_model_entries` filter:

```rust
if !path.is_dir() || !path.join("model.gguf").is_file() {
    return None;
}
```

- [ ] **Step 4: Update LoRA discovery**

In `desktop_core.rs`, replace `scan_lora_entries` filter with:

```rust
let is_lora_file = path.is_file()
    && path.extension().and_then(|v| v.to_str()) == Some("gguf")
    && name.starts_with("lora_");
if !is_lora_file {
    return None;
}
let display_name = name
    .strip_prefix("lora_")
    .and_then(|v| v.strip_suffix(".gguf"))
    .unwrap_or(&name)
    .to_string();
Some(LoraEntry {
    name: display_name,
    path: Some(display_path(&path)),
})
```

Filename-derived display names are required for this task. Metadata-based display names are covered by the runtime LoRA metadata validation in Task 6 and do not need desktop-side GGUF parsing.

- [ ] **Step 5: Update TUI startup model check**

In `voxui/crates/voxui-app/src/main.rs`, replace:

```rust
let has_model = model_path.join("manifest.json").exists();
```

with:

```rust
let has_model = model_path.join("model.gguf").exists();
```

Run:

```powershell
rg -n "manifest.json|lora_manifest.json|lora_base_lm|base_lm.gguf" voxui/crates/voxui-desktop voxui/crates/voxui-app
```

Update remaining app/discovery references to `model.gguf` or direct `lora_*.gguf`.

- [ ] **Step 6: Run desktop and TUI checks**

Run:

```powershell
cd voxui; cargo test -p voxui-desktop --manifest-path crates/voxui-desktop/src-tauri/Cargo.toml desktop_core -- --nocapture
cd voxui; cargo check -p voxui-app
```

Expected: both pass.

- [ ] **Step 7: Commit discovery migration**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-app/src/main.rs voxui/crates/voxui-app/src/config.rs
git commit -m "refactor(app): discover single GGUF VoxCPM models"
```

Expected: commit succeeds.

---

### Task 8: Test Suite And README Migration

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/inference_suite.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/generate_flow_parity.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/audiovae_parity.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/dit_parity.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/local_encoder_parity.rs`
- Modify: `D:/Sandbox_Share/VoxUI/README.txt`

- [ ] **Step 1: Update inference suite model and LoRA scanning**

In `inference_suite.rs`, replace model checks:

```rust
if !dir.join("model.gguf").exists() {
    eprintln!(
        "  [SKIP] {model_name}: model.gguf not found at {}",
        dir.display()
    );
    return;
}
```

Replace `find_lora_dirs` with:

```rust
fn find_lora_files(model: &Path) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(model)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|v| v.to_str()) == Some("gguf")
                && path.file_stem()
                    .map(|stem| stem.to_string_lossy().starts_with("lora_"))
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}
```

Replace the LoRA loop:

```rust
for lora_file in find_lora_files(&dir) {
    let lora_name = lora_file.file_stem().unwrap().to_string_lossy();
    engine
        .load_lora(&lora_file)
        .unwrap_or_else(|e| panic!("  [FAIL] load_lora({lora_name}): {e}"));
    run_synthesis(
        &mut engine,
        sentence_request(TEXT_EN),
        &format!("{model_name}/{dev_name}/en/{lora_name}"),
    );
    engine.unload_lora();
}
```

Update `full_matrix` filter:

```rust
.filter(|path| path.join("model.gguf").exists())
```

- [ ] **Step 2: Update README commands**

In `README.txt`, replace verify text with:

```text
Verify GGUF exports:
python exporter/verify_gguf.py models/voxcpm05-q4-lm/model.gguf
python exporter/verify_gguf.py models/voxcpm15-q4-lm/model.gguf
python exporter/verify_gguf.py models/voxcpm2-q4-lm/model.gguf
```

Add LoRA export examples:

```text
Export fp16 bundles with LoRA:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-profile fp16 --lora-dir VoxCPM/ft0.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-profile fp16 --lora-dir VoxCPM/ft1.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16 --lora-dir VoxCPM/ft2/latest
```

- [ ] **Step 3: Search for old layout references**

Run:

```powershell
rg -n "manifest.json|base_lm.gguf|residual_lm.gguf|feat_encoder.gguf|feat_decoder.gguf|audio_vae.gguf|projections.gguf|lora_manifest.json|lora_base_lm.gguf" README.txt exporter voxui/crates/voxui-inference voxui/crates/voxui-desktop voxui/crates/voxui-app
```

Expected: only historical docs under `docs/superpowers` may still mention old layout. Update any active code/test/README matches.

- [ ] **Step 4: Run checks**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
cd voxui; cargo check -p voxui-inference
cd voxui; cargo check -p voxui-app
cd voxui; cargo test -p voxui-inference --test native_runtime_purity
```

Expected: all pass.

- [ ] **Step 5: Commit test and README migration**

Run:

```powershell
git add README.txt voxui/crates/voxui-inference/tests
git commit -m "test: migrate VoxCPM checks to single GGUF layout"
```

Expected: commit succeeds.

---

### Task 9: Regenerate Local Model Exports

**Files:**
- Generated files under `D:/Sandbox_Share/VoxUI/models/`

- [ ] **Step 1: Remove old generated model directories safely**

Verify target paths first:

```powershell
$root = (Resolve-Path D:\Sandbox_Share\VoxUI\models).Path
$targets = @(
  "voxcpm05-fp16",
  "voxcpm15-fp16",
  "voxcpm2-fp16",
  "voxcpm05-q4-lm",
  "voxcpm15-q4-lm",
  "voxcpm2-q4-lm"
) | ForEach-Object { Join-Path $root $_ }
$targets | ForEach-Object { if (-not $_.StartsWith($root)) { throw "Refusing to delete outside models root: $_" } }
$targets | ForEach-Object { if (Test-Path $_) { Remove-Item -LiteralPath $_ -Recurse -Force } }
```

Expected: only directories under `D:\Sandbox_Share\VoxUI\models` are removed.

- [ ] **Step 2: Export fp16 bundles with direct LoRA files**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-profile fp16 --lora-dir VoxCPM/ft0.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-profile fp16 --lora-dir VoxCPM/ft1.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16 --lora-dir VoxCPM/ft2/latest
```

Expected: each directory contains `model.gguf`, sidecar tokenizer/config files, and one direct `lora_*.gguf` file.

- [ ] **Step 3: Export q4-lm bundles**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-q4-lm --variant 0.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-q4-lm --variant 1.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-q4-lm --variant 2.0 --quant-profile q4-lm
```

Expected: each q4 directory contains `model.gguf` and sidecars, no old component GGUF files.

- [ ] **Step 4: Verify generated layout**

Run:

```powershell
Get-ChildItem models -Directory | ForEach-Object {
  $ggufs = Get-ChildItem $_.FullName -Filter *.gguf
  [PSCustomObject]@{
    Model = $_.Name
    GgufFiles = ($ggufs.Name -join ", ")
    HasConfig = Test-Path (Join-Path $_.FullName "config.json")
    HasTokenizer = Test-Path (Join-Path $_.FullName "tokenizer.json")
  }
} | Format-Table -AutoSize
```

Expected: each row includes `model.gguf`; fp16 rows also include one `lora_*.gguf`; q4 rows include only `model.gguf`.

- [ ] **Step 5: Verify GGUF metadata**

Run:

```powershell
python exporter/verify_gguf.py models/voxcpm05-fp16/model.gguf
python exporter/verify_gguf.py models/voxcpm15-fp16/model.gguf
python exporter/verify_gguf.py models/voxcpm2-fp16/model.gguf
python exporter/verify_gguf.py models/voxcpm2-fp16/lora_ft2.gguf
```

Expected: metadata includes `voxcpm.schema_version = 2`, `voxcpm.kind = 'base'` for base files, and `voxcpm.kind = 'lora'` for LoRA files.

- [ ] **Step 6: Commit regenerated model pointers only if tracked**

Run:

```powershell
git status --short models
```

If `models/` is untracked or ignored, do not commit generated model files. If tracked changes appear under `models/`, commit them intentionally:

```powershell
git add models
git commit -m "build(models): regenerate VoxCPM single GGUF exports"
```

Expected: no source code changes are mixed into a generated-model commit.

---

### Task 10: CPU Verification

**Files:**
- Source files touched by prior tasks.
- Generated WAVs under `D:/Sandbox_Share/VoxUI/test_output/`

- [ ] **Step 1: Run Python exporter tests**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: `OK`.

- [ ] **Step 2: Run Rust loader and focused parity tests**

Run:

```powershell
cd voxui; cargo test -p voxui-gguf
cd voxui; cargo test -p voxui-inference --test manifest_loader
cd voxui; cargo test -p voxui-inference --test request_validation
cd voxui; cargo test -p voxui-inference --test native_runtime_purity
cd voxui; cargo test -p voxui-inference --test audiovae_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test local_encoder_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test dit_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test generate_flow_parity -- --nocapture
cd voxui; cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: all pass. If numeric parity tolerances need adjustment because export order changed but tensor values did not, investigate first; do not loosen tolerances without confirming the tensor data matches.

- [ ] **Step 3: Run CPU inference suite**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test inference_suite --release -- --nocapture
```

Expected: fp16 and q4 CPU cases produce non-empty finite WAVs under `test_output/`, including direct LoRA files for fp16 model dirs.

- [ ] **Step 4: Check generated WAV files**

Run:

```powershell
Get-ChildItem D:\Sandbox_Share\VoxUI\test_output -Filter *.wav | Select-Object Name, Length | Format-Table -AutoSize
```

Expected: WAV files are non-empty.

- [ ] **Step 5: Commit verification fixes if needed**

If verification required source fixes, run:

```powershell
git status --short
git add exporter voxui README.txt
git commit -m "fix: complete single GGUF VoxCPM verification"
```

Expected: commit includes only source/docs/test fixes, not large generated WAV outputs unless intentionally tracked.

---

### Task 11: CUDA Build Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Set CUDA environment**

Run:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
```

- [ ] **Step 2: Run CUDA compile check**

Run:

```powershell
cd voxui; cargo check -p voxui-inference --features cuda
```

Expected: pass.

- [ ] **Step 3: Run CUDA inference suite when local CUDA is available**

Run:

```powershell
cd voxui; cargo test -p voxui-inference --test inference_suite --release --features cuda -- --nocapture
```

Expected: CUDA cases pass or skip only when CUDA device creation fails cleanly.

- [ ] **Step 4: Commit CUDA fixes if needed**

If CUDA-only source fixes were needed, run:

```powershell
git add voxui/crates/voxui-inference
git commit -m "fix(inference): support CUDA single GGUF loading"
```

Expected: commit succeeds.

---

## Plan Self-Review

Spec coverage:

- Single base `model.gguf` export is covered by Tasks 1 and 9.
- Single direct `lora_<name>.gguf` export is covered by Tasks 2 and 9.
- Config/tokenizer sidecars remain covered by Tasks 1 and 9.
- Shared Rust model store and one coordinated loader are covered by Tasks 4 and 5.
- Removal of component manifest/path contracts is covered by Tasks 3, 5, and 8.
- Direct LoRA runtime loading is covered by Task 6.
- Desktop/TUI discovery changes are covered by Task 7.
- README and active test migration are covered by Task 8.
- CPU verification is covered by Task 10.
- CUDA compile/runtime verification is covered by Task 11.

Placeholder scan:

- The plan contains no `TBD`, `TODO`, or open-ended implementation steps.
- Every code-changing task includes the concrete code shape or command needed for the worker.

Type consistency:

- `ModelConfig`, `ModelVariant`, `AudioVaeManifest`, and `GgufModelLoader::from_model_dir` are introduced before later tasks use them.
- Runtime loaders consistently use `load_from_config` and `from_model_config`.
- LoRA runtime uses file paths and `LoraAdapter::load_file_for_model`.
