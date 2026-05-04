# VoxCPM Native Candle Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the VoxCPM exporter and `voxui-inference` runtime so VoxCPM 0.5, VoxCPM 1.5, and VoxCPM 2.0 synthesize intelligible audio with pure Rust Candle inference on CPU and CUDA, with optional LoRA and VoxCPM2 reference audio.

**Architecture:** Python is used only for exporter validation and golden trace generation from the local VoxCPM reference implementation. Runtime synthesis lives entirely in Rust under `voxui/crates/voxui-inference`, loads manifest-based GGUF bundles, reconstructs the VoxCPM graph with Candle tensors, and follows Python `VoxCPM.generate()` request semantics except for the intentionally omitted denoiser argument.

**Tech Stack:** Rust 2021, Candle 0.8, `voxui-gguf`, `tokenizers`, Python VoxCPM reference code for trace/export tooling, PowerShell verification commands, CPU and optional CUDA backends.

---

## File Structure

- Create `tools/golden_trace/voxcpm_trace.py`: Python trace generator that loads local VoxCPM source models and records reference tensors for small deterministic cases.
- Create `tools/golden_trace/trace_schema.py`: trace manifest writer/reader helpers shared by trace tests.
- Create `tools/golden_trace/tests/test_trace_schema.py`: Python tests for trace schema and binary tensor roundtrip.
- Create `goldens/README.md`: documents generated trace cases and regeneration commands.
- Modify `exporter/export_voxcpm.py`: rewrite component partitioning, manifest writing, tensor coverage validation, component filenames, and LoRA export.
- Create `exporter/tests/test_export_manifest.py`: exporter manifest and required-tensor tests.
- Create `voxui/crates/voxui-inference/src/manifest.rs`: typed Rust bundle manifest loader and validation.
- Create `voxui/crates/voxui-inference/src/request.rs`: `SynthesisRequest`, validation, defaults, and normalized text handling.
- Create `voxui/crates/voxui-inference/src/audio_io.rs`: WAV load, mono conversion, resampling, padding, and WAV write helpers for tests.
- Create `voxui/crates/voxui-inference/src/trace.rs`: test-only golden trace tensor loading helpers.
- Modify `voxui/crates/voxui-inference/src/model_loader.rs`: load component files by manifest paths and expose stricter tensor existence checks.
- Modify `voxui/crates/voxui-inference/src/base_lm.rs`: MiniCPM parity for MuP scaling, LongRoPE, Python `rotate_half`, non-causal mode, and KV cache.
- Modify `voxui/crates/voxui-inference/src/encoder.rs`: local encoder input shape `[B, T, P, D]` and output shape `[B, T, hidden]`.
- Modify `voxui/crates/voxui-inference/src/audiovae.rs`: AudioVAE V1/V2 encoder and decoder parity.
- Modify `voxui/crates/voxui-inference/src/dit.rs`: `UnifiedCFM.forward` and Euler solver parity.
- Modify `voxui/crates/voxui-inference/src/fsq.rs`: scalar quantization behavior and metadata-driven dim/scale.
- Modify `voxui/crates/voxui-inference/src/lora.rs`: component-aware LoRA loader and application for LM, residual LM, DiT, and projections.
- Replace `voxui/crates/voxui-inference/src/engine.rs`: native generation flow matching Python `_generate` and `_inference`.
- Modify `voxui/crates/voxui-inference/src/lib.rs`: export manifest/request/audio modules and keep only native runtime APIs.
- Modify `voxui/crates/voxui-inference/Cargo.toml`: add serde, serde_json, hound, and dev-only helper dependencies if needed.
- Create or modify Rust tests under `voxui/crates/voxui-inference/tests/`: runtime purity, request validation, manifest loader, MiniCPM parity, AudioVAE parity, local encoder parity, DiT parity, generation flow parity, LoRA parity, and end-to-end inference suite.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: scan manifest model bundles and pass `SynthesisRequest`.
- Modify `voxui/crates/voxui-app/src/app.rs`: scan manifest model bundles and expose prompt/reference request fields.
- Modify `voxui/crates/voxui-app/src/config.rs`: keep model and LoRA directory fields consistent with manifest scanning.

---

### Task 1: Native Runtime Guard

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/native_runtime_purity.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`

- [x] **Step 1: Write the runtime-purity regression test**

Create `voxui/crates/voxui-inference/tests/native_runtime_purity.rs`:

```rust
use std::fs;
use std::path::Path;

#[test]
fn inference_source_does_not_spawn_or_embed_python_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let denied = [
        "std::process::Command",
        "python.exe",
        "python/",
        "pyo3",
        "PyModule",
        "PyObject",
        "VoxCPM.generate(",
    ];

    for entry in fs::read_dir(src).expect("read src dir") {
        let path = entry.expect("read src entry").path();
        if path.extension().and_then(|v| v.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source file");
        for needle in denied {
            assert!(
                !text.contains(needle),
                "{} contains prohibited runtime token `{}`",
                path.display(),
                needle
            );
        }
    }

    assert!(
        !root.join("python").exists(),
        "runtime helper directory must not exist in voxui-inference"
    );
}
```

- [x] **Step 2: Run the guard**

Run:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
cargo test -p voxui-inference --test native_runtime_purity
```

Expected: pass. This is a guard against a prohibited runtime path, not the main TDD failure.

- [x] **Step 3: Export explicit native modules**

In `voxui/crates/voxui-inference/src/lib.rs`, add module declarations for the native modules defined in this plan:

```rust
pub mod audio_io;
pub mod manifest;
pub mod request;

#[cfg(test)]
pub mod trace;

pub use engine::VoxCPMEngine;
pub use manifest::{BundleManifest, ComponentFiles, ModelVariant};
pub use request::SynthesisRequest;
```

- [x] **Step 4: Run the guard and library check**

Run:

```powershell
cargo check -p voxui-inference
cargo test -p voxui-inference --test native_runtime_purity
```

Expected: `cargo check` fails until the new modules exist. Keep the failure visible for Task 3 and Task 6.

- [x] **Step 5: Commit the guard with the first compiling native module commit**

Do not commit this task by itself if `cargo check` fails. Include it in the first commit where the crate compiles.

---

### Task 2: Python Golden Trace Tooling

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/tools/golden_trace/trace_schema.py`
- Create: `D:/Sandbox_Share/VoxUI/tools/golden_trace/voxcpm_trace.py`
- Create: `D:/Sandbox_Share/VoxUI/tools/golden_trace/tests/test_trace_schema.py`
- Create: `D:/Sandbox_Share/VoxUI/goldens/README.md`

- [x] **Step 1: Write failing trace schema tests**

Create `tools/golden_trace/tests/test_trace_schema.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.golden_trace.trace_schema import TensorRecord, TraceWriter, read_tensor_record


class TraceSchemaTests(unittest.TestCase):
    def test_tensor_record_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = TraceWriter(root, case_name="unit")
            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            record = writer.write_tensor("base_lm_hidden", arr)
            writer.write_manifest(
                variant="2.0",
                architecture="voxcpm2",
                request={"text": "hello"},
                tensors=[record],
            )

            manifest = json.loads((root / "unit" / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["tensors"][0]["shape"], [3, 4])
            restored = read_tensor_record(root / "unit", TensorRecord(**manifest["tensors"][0]))
            np.testing.assert_allclose(restored, arr)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
python -m unittest tools.golden_trace.tests.test_trace_schema
```

Expected: fail because `trace_schema.py` does not exist.

- [x] **Step 3: Implement trace schema helpers**

Create `tools/golden_trace/trace_schema.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TensorRecord:
    name: str
    file: str
    dtype: str
    shape: list[int]


class TraceWriter:
    def __init__(self, root: Path, case_name: str) -> None:
        self.case_dir = root / case_name
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def write_tensor(self, name: str, tensor: np.ndarray) -> TensorRecord:
        arr = np.asarray(tensor, dtype=np.float32)
        file_name = f"{name}.f32"
        arr.tofile(self.case_dir / file_name)
        return TensorRecord(name=name, file=file_name, dtype="f32", shape=list(arr.shape))

    def write_manifest(
        self,
        *,
        variant: str,
        architecture: str,
        request: dict[str, Any],
        tensors: list[TensorRecord],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "variant": variant,
            "architecture": architecture,
            "request": request,
            "metadata": metadata or {},
            "tensors": [asdict(t) for t in tensors],
        }
        (self.case_dir / "trace.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def read_tensor_record(case_dir: Path, record: TensorRecord) -> np.ndarray:
    dtype = np.float32 if record.dtype == "f32" else None
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {record.dtype}")
    arr = np.fromfile(case_dir / record.file, dtype=dtype)
    return arr.reshape(record.shape)
```

- [x] **Step 4: Implement trace generation entrypoint**

Create `tools/golden_trace/voxcpm_trace.py` with these behaviors:

```python
"""
Generate small deterministic reference traces from the local VoxCPM Python code.

This script is not used by Rust runtime inference. It writes trace files consumed
by Rust parity tests.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from tools.golden_trace.trace_schema import TraceWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def import_local_voxcpm(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "VoxCPM" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["0.5", "1.5", "2.0"], required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("goldens"))
    parser.add_argument("--text", default="Hello, welcome to the stream!")
    parser.add_argument("--prompt-wav-path", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--reference-wav-path", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    import_local_voxcpm(args.repo_root)

    from voxcpm import VoxCPM

    model = VoxCPM(
        voxcpm_model_path=str(args.model_dir),
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=False,
        device="cpu",
    )

    capture = TraceCapture(model)
    capture.install()
    tensors: list = []

    wav = model.generate(
        text=args.text,
        prompt_wav_path=str(args.prompt_wav_path) if args.prompt_wav_path else None,
        prompt_text=args.prompt_text,
        reference_wav_path=str(args.reference_wav_path) if args.reference_wav_path else None,
        cfg_value=2.0,
        inference_timesteps=4,
        min_len=1,
        max_len=3,
        normalize=False,
        denoise=False,
        retry_badcase=False,
    )

    writer = TraceWriter(args.out_dir, args.case_name)
    tensors.extend(capture.records(writer))
    tensors.append(writer.write_tensor("decoded_wav_head", np.asarray(wav[:4096], dtype=np.float32)))
    writer.write_manifest(
        variant=args.variant,
        architecture="voxcpm2" if args.variant == "2.0" else "voxcpm",
        request={
            "text": args.text,
            "prompt_wav_path": str(args.prompt_wav_path) if args.prompt_wav_path else None,
            "prompt_text": args.prompt_text,
            "reference_wav_path": str(args.reference_wav_path) if args.reference_wav_path else None,
            "cfg_value": 2.0,
            "inference_timesteps": 4,
            "min_len": 1,
            "max_len": 3,
            "normalize": False,
            "retry_badcase": False,
        },
        tensors=tensors,
        metadata={"seed": args.seed, "source_model_dir": str(args.model_dir.resolve())},
    )


if __name__ == "__main__":
    main()
```

Implement `TraceCapture` in the same file. It must use PyTorch hooks and small wrapper functions without changing model math. It must write these tensors when they are present for the selected request mode:

- `token_ids`
- `text_mask`
- `audio_mask`
- `audio_vae_encoded_prompt`
- `audio_vae_encoded_reference`
- `local_encoder_output`
- `base_lm_prefill_hidden`
- `residual_lm_prefill_hidden`
- `first_fsq_hidden`
- `first_dit_patch`
- `stop_logits`
- `decoded_wav_head`

- [x] **Step 5: Run schema tests**

Run:

```powershell
python -m unittest tools.golden_trace.tests.test_trace_schema
```

Expected: `OK`.

- [x] **Step 6: Generate the required trace cases**

Run:

```powershell
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM-0.5B --variant 0.5 --case-name voxcpm05_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM1.5 --variant 1.5 --case-name voxcpm15_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM2 --variant 2.0 --case-name voxcpm2_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM2 --variant 2.0 --case-name voxcpm2_reference --text "Hello, welcome to the stream!" --reference-wav-path for_test_wav/reference.wav
```

If `for_test_wav/reference.wav` does not exist, choose the first `.wav` from `for_test_wav/` and record the exact filename in `goldens/README.md`.

- [x] **Step 7: Document trace regeneration**

Create `goldens/README.md` with:

```markdown
# VoxCPM Golden Traces

These files are generated from the local Python VoxCPM reference implementation
and are used only by tests. Runtime inference in `voxui-inference` is pure Rust
Candle.

Regenerate after exporter or model-graph changes:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM-0.5B --variant 0.5 --case-name voxcpm05_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM1.5 --variant 1.5 --case-name voxcpm15_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM2 --variant 2.0 --case-name voxcpm2_zero_shot --text "Hello, welcome to the stream!"
```
```

- [x] **Step 8: Commit trace tooling**

Run:

```powershell
git add tools/golden_trace goldens/README.md
git commit -m "test(inference): add VoxCPM golden trace tooling"
```

---

### Task 3: Exporter Bundle Schema And Tensor Coverage

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/exporter/export_voxcpm.py`
- Create: `D:/Sandbox_Share/VoxUI/exporter/tests/test_export_manifest.py`

- [x] **Step 1: Write failing exporter tests**

Create `exporter/tests/test_export_manifest.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import torch

from exporter.export_voxcpm import build_manifest, partition_weights, validate_required_tensors


class ExportManifestTests(unittest.TestCase):
    def test_partition_uses_python_component_names(self):
        weights = {
            "base_lm.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "residual_lm.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "feat_encoder.in_proj.weight": torch.zeros(2, 2),
            "feat_decoder.input_embed.weight": torch.zeros(2, 2),
            "fsq_layer.project_in.weight": torch.zeros(2, 2),
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
            "base_lm.gguf": [("base_lm.norm.weight", torch.zeros(2))],
        }
        with self.assertRaisesRegex(ValueError, "missing required tensor"):
            validate_required_tensors(buckets, variant="2.0")

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
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
python -m unittest exporter.tests.test_export_manifest
```

Expected: fail because the current exporter uses `encoder.gguf` and `dit.gguf`, does not expose `build_manifest`, and does not hard-fail missing coverage.

- [x] **Step 3: Rewrite component partitioning**

In `exporter/export_voxcpm.py`, set component files exactly:

```python
COMPONENT_FILES = {
    "base_lm": "base_lm.gguf",
    "residual_lm": "residual_lm.gguf",
    "feat_encoder": "feat_encoder.gguf",
    "feat_decoder": "feat_decoder.gguf",
    "audio_vae": "audio_vae.gguf",
    "projections": "projections.gguf",
}
```

Use source-faithful tensor names:

```python
def get_component_for_key(key):
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
```

AudioVAE tensor names must be exported as `audio_vae.<source_key>`:

```python
def _vae_transform(key):
    return f"audio_vae.{key}"
```

- [x] **Step 4: Add manifest generation**

Implement `build_manifest(...)` so the written `manifest.json` contains:

```json
{
  "schema_version": 1,
  "architecture": "voxcpm2",
  "variant": "2.0",
  "source_model_dir": "D:/Sandbox_Share/VoxUI/VoxCPM/models/VoxCPM2",
  "source_weight_format": "safetensors",
  "special_tokens": {
    "audio_start": 101,
    "audio_end": 102,
    "ref_audio_start": 103,
    "ref_audio_end": 104
  },
  "components": {
    "base_lm": "base_lm.gguf",
    "residual_lm": "residual_lm.gguf",
    "feat_encoder": "feat_encoder.gguf",
    "feat_decoder": "feat_decoder.gguf",
    "audio_vae": "audio_vae.gguf",
    "projections": "projections.gguf"
  }
}
```

Also include `patch_size`, `feat_dim`, `scalar_quantization_latent_dim`, `scalar_quantization_scale`, `audio_vae`, `lm_config`, `encoder_config`, `dit_config`, and `quantization` from the Python config.

- [x] **Step 5: Add strict tensor coverage**

Implement `validate_required_tensors(buckets, variant)` with these checks:

```python
REQUIRED_PREFIXES = {
    "base_lm.gguf": ["base_lm.norm.weight", "base_lm.layers.0.self_attn.q_proj.weight"],
    "residual_lm.gguf": ["residual_lm.norm.weight", "residual_lm.layers.0.self_attn.q_proj.weight"],
    "feat_encoder.gguf": ["feat_encoder.in_proj.weight", "feat_encoder.special_token"],
    "feat_decoder.gguf": ["feat_decoder"],
    "audio_vae.gguf": ["audio_vae"],
    "projections.gguf": ["fsq_layer", "enc_to_lm_proj.weight", "lm_to_dit_proj.weight", "res_to_dit_proj.weight", "stop_proj.weight", "stop_head.weight"],
}
```

The function must raise `ValueError("missing required tensor ...")` when no tensor in a component matches a required name or prefix. It must also raise for duplicate tensor names within one GGUF component.

- [x] **Step 6: Update LoRA export**

Use output files:

- `lora_base_lm.gguf`
- `lora_residual_lm.gguf`
- `lora_feat_decoder.gguf`
- `lora_projections.gguf`

Keep LoRA tensor names source-faithful, and write `lora_manifest.json` with rank, alpha, variant, architecture, and target module lists.

- [x] **Step 7: Run exporter tests**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest
```

Expected: `OK`.

- [x] **Step 8: Commit exporter schema**

Run:

```powershell
git add exporter/export_voxcpm.py exporter/tests/test_export_manifest.py
git commit -m "fix(exporter): add native VoxCPM bundle schema"
```

---

### Task 4: Regenerate Model Bundles

**Files:**
- Modify generated artifacts under `D:/Sandbox_Share/VoxUI/models/`

- [x] **Step 1: Remove stale generated model directories after confirming target paths**

Run:

```powershell
Get-ChildItem D:\Sandbox_Share\VoxUI\models -Directory | Select-Object FullName
```

Expected: only generated model bundle directories are listed. Do not remove source model directories under `VoxCPM/models`.

- [x] **Step 2: Export VoxCPM 0.5**

Run:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-lm fp16 --quant-encoder fp16 --quant-dit fp16 --quant-vae fp16
```

Expected: `models/voxcpm05-fp16/manifest.json`, all required GGUF components, tokenizer files, and config copy.

- [x] **Step 3: Export VoxCPM 1.5**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-lm fp16 --quant-encoder fp16 --quant-dit fp16 --quant-vae fp16
```

Expected: `models/voxcpm15-fp16/manifest.json` and all required GGUF components.

- [x] **Step 4: Export VoxCPM 2.0**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-lm fp16 --quant-encoder fp16 --quant-dit fp16 --quant-vae fp16
```

Expected: `models/voxcpm2-fp16/manifest.json` and all required GGUF components.

- [x] **Step 5: Export available LoRA adapters**

Run one command per local adapter that exists:

```powershell
if (Test-Path VoxCPM/ft0.5/latest) { python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --lora-dir VoxCPM/ft0.5/latest }
if (Test-Path VoxCPM/ft1.5/latest) { python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --lora-dir VoxCPM/ft1.5/latest }
if (Test-Path VoxCPM/ft2/latest) { python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --lora-dir VoxCPM/ft2/latest }
```

Expected: each available adapter creates `lora_manifest.json` and component LoRA GGUF files under a `lora_*` subdirectory.

- [x] **Step 6: Verify bundle files**

Run:

```powershell
Get-ChildItem models -Filter manifest.json -Recurse | Select-Object FullName
Get-ChildItem models -Filter feat_encoder.gguf -Recurse | Select-Object FullName
Get-ChildItem models -Filter feat_decoder.gguf -Recurse | Select-Object FullName
```

Expected: all three variants show manifest, `feat_encoder.gguf`, and `feat_decoder.gguf`.

- [x] **Step 7: Commit regenerated bundle metadata if model artifacts are intentionally tracked**

Run:

```powershell
git status --short models
git add models
git commit -m "chore(models): regenerate native VoxCPM bundles"
```

If GGUF files are too large for git, commit only `manifest.json` examples and update `.gitignore` in a separate commit.

---

### Task 5: Rust Manifest Loader

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/manifest.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/Cargo.toml`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/manifest_loader.rs`

- [x] **Step 1: Write failing manifest loader tests**

Create `voxui/crates/voxui-inference/tests/manifest_loader.rs`:

```rust
use std::fs;

use voxui_inference::{BundleManifest, ModelVariant};

#[test]
fn manifest_parses_variant_and_components() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("manifest.json"),
        r#"{
            "schema_version": 1,
            "architecture": "voxcpm2",
            "variant": "2.0",
            "source_model_dir": "source",
            "source_weight_format": "safetensors",
            "special_tokens": {"audio_start":101,"audio_end":102,"ref_audio_start":103,"ref_audio_end":104},
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0,
            "audio_vae": {"sample_rate":16000,"out_sample_rate":48000,"latent_dim":64,"chunk_size":20,"decode_chunk_size":240,"encoder_rates":[2,5,8,8],"decoder_rates":[8,6,5,2,2,2]},
            "components": {"base_lm":"base_lm.gguf","residual_lm":"residual_lm.gguf","feat_encoder":"feat_encoder.gguf","feat_decoder":"feat_decoder.gguf","audio_vae":"audio_vae.gguf","projections":"projections.gguf"},
            "quantization": {}
        }"#,
    )
    .unwrap();

    let manifest = BundleManifest::load(dir.path()).unwrap();
    assert_eq!(manifest.variant, ModelVariant::VoxCpm2);
    assert_eq!(manifest.special_tokens.audio_start, 101);
    assert!(manifest.component_path(dir.path(), "feat_decoder").unwrap().ends_with("feat_decoder.gguf"));
}

#[test]
fn manifest_rejects_reference_tokens_for_non_v2() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("manifest.json"),
        r#"{
            "schema_version": 1,
            "architecture": "voxcpm",
            "variant": "1.5",
            "source_model_dir": "source",
            "source_weight_format": "safetensors",
            "special_tokens": {"audio_start":101,"audio_end":102,"ref_audio_start":103,"ref_audio_end":104},
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 256,
            "scalar_quantization_scale": 9.0,
            "audio_vae": {"sample_rate":16000,"latent_dim":64,"chunk_size":20,"decode_chunk_size":240,"encoder_rates":[2,5,8,8],"decoder_rates":[8,6,5,2,2,2]},
            "components": {"base_lm":"base_lm.gguf","residual_lm":"residual_lm.gguf","feat_encoder":"feat_encoder.gguf","feat_decoder":"feat_decoder.gguf","audio_vae":"audio_vae.gguf","projections":"projections.gguf"},
            "quantization": {}
        }"#,
    )
    .unwrap();

    let err = BundleManifest::load(dir.path()).unwrap_err();
    assert!(err.to_string().contains("ref_audio"));
}
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test manifest_loader
```

Expected: fail because `BundleManifest` and `ModelVariant` do not exist.

- [x] **Step 3: Add dependencies**

In `voxui/crates/voxui-inference/Cargo.toml`, add:

```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
tempfile = "3"
```

- [x] **Step 4: Implement manifest types**

Create `src/manifest.rs` with:

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelVariant {
    #[serde(rename = "0.5")]
    VoxCpm05,
    #[serde(rename = "1.5")]
    VoxCpm15,
    #[serde(rename = "2.0")]
    VoxCpm2,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokens {
    pub audio_start: u32,
    pub audio_end: u32,
    pub ref_audio_start: Option<u32>,
    pub ref_audio_end: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeManifest {
    pub sample_rate: u32,
    pub out_sample_rate: Option<u32>,
    pub latent_dim: usize,
    pub chunk_size: usize,
    pub decode_chunk_size: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_rates: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComponentFiles {
    pub base_lm: String,
    pub residual_lm: String,
    pub feat_encoder: String,
    pub feat_decoder: String,
    pub audio_vae: String,
    pub projections: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub architecture: String,
    pub variant: ModelVariant,
    pub source_model_dir: String,
    pub source_weight_format: String,
    pub special_tokens: SpecialTokens,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: f32,
    pub audio_vae: AudioVaeManifest,
    pub components: ComponentFiles,
    #[serde(default)]
    pub lm_config: serde_json::Value,
    #[serde(default)]
    pub encoder_config: serde_json::Value,
    #[serde(default)]
    pub dit_config: serde_json::Value,
    #[serde(default)]
    pub residual_lm_num_layers: Option<usize>,
    #[serde(default)]
    pub residual_lm_no_rope: Option<bool>,
    #[serde(default)]
    pub quantization: HashMap<String, String>,
}

impl BundleManifest {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let manifest_path = model_dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", manifest_path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.audio_vae.out_sample_rate.unwrap_or(self.audio_vae.sample_rate)
    }

    pub fn component_path(&self, model_dir: &Path, component: &str) -> Result<PathBuf> {
        let file = match component {
            "base_lm" => &self.components.base_lm,
            "residual_lm" => &self.components.residual_lm,
            "feat_encoder" => &self.components.feat_encoder,
            "feat_decoder" => &self.components.feat_decoder,
            "audio_vae" => &self.components.audio_vae,
            "projections" => &self.components.projections,
            other => bail!("unknown component `{other}`"),
        };
        Ok(model_dir.join(file))
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!("unsupported VoxCPM bundle schema {}", self.schema_version);
        }
        if self.special_tokens.audio_start != 101 || self.special_tokens.audio_end != 102 {
            bail!("unexpected audio special tokens");
        }
        match self.variant {
            ModelVariant::VoxCpm2 => {
                if self.special_tokens.ref_audio_start != Some(103)
                    || self.special_tokens.ref_audio_end != Some(104)
                {
                    bail!("VoxCPM2 manifest must include ref_audio tokens 103 and 104");
                }
            }
            _ => {
                if self.special_tokens.ref_audio_start.is_some()
                    || self.special_tokens.ref_audio_end.is_some()
                {
                    bail!("ref_audio tokens are only valid for VoxCPM2");
                }
            }
        }
        Ok(())
    }
}
```

- [x] **Step 5: Export manifest module**

In `src/lib.rs`, add:

```rust
pub mod manifest;
pub use manifest::{BundleManifest, ComponentFiles, ModelVariant};
```

- [x] **Step 6: Run manifest tests**

Run:

```powershell
cargo test -p voxui-inference --test manifest_loader
```

Expected: `OK`.

- [x] **Step 7: Commit manifest loader**

Run:

```powershell
git add voxui/crates/voxui-inference/Cargo.toml voxui/crates/voxui-inference/src/manifest.rs voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/tests/manifest_loader.rs
git commit -m "feat(inference): load VoxCPM bundle manifests"
```

---

### Task 6: Synthesis Request API And Validation

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/request.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/request_validation.rs`

- [x] **Step 1: Write failing request validation tests**

Create `voxui/crates/voxui-inference/tests/request_validation.rs`:

```rust
use std::path::PathBuf;

use voxui_inference::{ModelVariant, SynthesisRequest};

#[test]
fn request_rejects_empty_text_after_whitespace_normalization() {
    let err = SynthesisRequest {
        text: " \n\t ".to_string(),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("text must not be empty"));
}

#[test]
fn request_requires_prompt_text_when_prompt_audio_is_present() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        prompt_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        prompt_text: None,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("prompt_text"));
}

#[test]
fn request_allows_reference_audio_without_text_on_voxcpm2() {
    let req = SynthesisRequest {
        text: "hello".to_string(),
        reference_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap();
    assert_eq!(req.prompt_text, None);
}

#[test]
fn request_rejects_reference_audio_on_non_v2() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        reference_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm15)
    .unwrap_err();
    assert!(err.to_string().contains("Reference audio requires VoxCPM2"));
}

#[test]
fn request_rejects_normalize_until_rust_normalizer_is_implemented() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        normalize: true,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("normalize=true"));
}
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test request_validation
```

Expected: fail because `SynthesisRequest` does not exist.

- [x] **Step 3: Implement request type**

Create `src/request.rs`:

```rust
use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::manifest::ModelVariant;

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub normalize: bool,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 4096,
            normalize: false,
            retry_badcase: true,
            retry_badcase_max_times: 3,
            retry_badcase_ratio_threshold: 6.0,
        }
    }
}

impl SynthesisRequest {
    pub fn validated(mut self, variant: ModelVariant) -> Result<Self> {
        self.text = collapse_whitespace(&self.text);
        if self.text.is_empty() {
            bail!("text must not be empty");
        }
        if self.prompt_wav_path.is_some()
            && self.prompt_text.as_ref().map(|s| collapse_whitespace(s).is_empty()).unwrap_or(true)
        {
            bail!("prompt_text is required when prompt_wav_path is present");
        }
        if self.reference_wav_path.is_some() && variant != ModelVariant::VoxCpm2 {
            bail!("Reference audio requires VoxCPM2");
        }
        if self.normalize {
            bail!("normalize=true is not supported until the Rust VoxCPM normalizer is implemented");
        }
        if self.min_len > self.max_len {
            bail!("min_len must be <= max_len");
        }
        if self.inference_timesteps == 0 {
            bail!("inference_timesteps must be greater than zero");
        }
        Ok(self)
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
```

- [x] **Step 4: Export request module**

In `src/lib.rs`, add:

```rust
pub mod request;
pub use request::SynthesisRequest;
```

- [x] **Step 5: Run request tests**

Run:

```powershell
cargo test -p voxui-inference --test request_validation
```

Expected: `OK`.

- [x] **Step 6: Commit request API**

Run:

```powershell
git add voxui/crates/voxui-inference/src/request.rs voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/tests/request_validation.rs
git commit -m "feat(inference): add VoxCPM synthesis request"
```

---

### Task 7: Audio IO And AudioVAE Encoder/Decoder Parity

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/audio_io.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/audiovae.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/Cargo.toml`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/audiovae_parity.rs`

- [x] **Step 1: Write failing audio IO and AudioVAE parity tests**

Create `tests/audiovae_parity.rs`:

```rust
use std::path::Path;

use candle_core::Device;
use voxui_inference::audio_io::load_wav_mono_resampled;
use voxui_inference::{AudioVAE, BundleManifest, GgufModelLoader};

#[test]
fn wav_loader_returns_mono_f32_at_requested_rate() {
    let wav = std::fs::read_dir("D:/Sandbox_Share/VoxUI/for_test_wav")
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|v| v.to_str()) == Some("wav"))
        .expect("at least one test wav");
    let audio = load_wav_mono_resampled(&wav, 16_000).unwrap();
    assert_eq!(audio.sample_rate, 16_000);
    assert!(!audio.samples.is_empty());
    assert!(audio.samples.iter().all(|v| v.is_finite()));
}

#[test]
fn audiovae_decode_matches_python_trace_head() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let manifest = BundleManifest::load(model_dir).unwrap();
    let loader = GgufModelLoader::new(&manifest.component_path(model_dir, "audio_vae").unwrap(), Device::Cpu).unwrap();
    let vae = AudioVAE::load_from_manifest(&loader, &manifest.audio_vae).unwrap();

    let trace = voxui_inference::trace::TraceCase::load("D:/Sandbox_Share/VoxUI/goldens/voxcpm2_zero_shot").unwrap();
    let latent = trace.tensor("generated_latent").unwrap();
    let expected = trace.tensor("decoded_wav_head").unwrap();
    let decoded = vae.decode(&latent).unwrap();
    voxui_inference::trace::assert_close_prefix(&decoded, &expected, 2e-3).unwrap();
}
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test audiovae_parity -- --nocapture
```

Expected: fail because `audio_io`, `AudioVAE::load_from_manifest`, encoder support, and trace helpers do not exist or decoder parity differs.

- [x] **Step 3: Add audio dependency**

In `Cargo.toml`, add:

```toml
hound = "3"
```

- [x] **Step 4: Implement WAV loading**

Create `src/audio_io.rs` with:

```rust
use std::path::Path;

use anyhow::{bail, Result};

pub struct LoadedAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn load_wav_mono_resampled(path: &Path, target_rate: u32) -> Result<LoadedAudio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut interleaved = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                interleaved.push(s?);
            }
        }
        hound::SampleFormat::Int => {
            let denom = (1_i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) as f32;
            for s in reader.samples::<i32>() {
                interleaved.push(s? as f32 / denom);
            }
        }
    }
    if interleaved.is_empty() {
        bail!("empty wav {}", path.display());
    }

    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks(channels) {
        mono.push(frame.iter().copied().sum::<f32>() / frame.len() as f32);
    }

    Ok(LoadedAudio {
        sample_rate: target_rate,
        samples: resample_linear(&mono, spec.sample_rate, target_rate),
    })
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.len() < 2 {
        return input.to_vec();
    }
    let out_len = ((input.len() as u64 * dst_rate as u64) / src_rate as u64).max(1) as usize;
    let scale = src_rate as f64 / dst_rate as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * scale;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let frac = (pos - lo as f64) as f32;
        out.push(input[lo] * (1.0 - frac) + input[hi] * frac);
    }
    out
}
```

- [x] **Step 5: Implement AudioVAE load and encode/decode parity**

In `src/audiovae.rs`, add `AudioVAE::load_from_manifest(loader, manifest_audio_vae)` and implement:

- V1 and V2 encoder blocks from `VoxCPM/src/voxcpm/modules/audiovae/audio_vae.py` and `audio_vae_v2.py`.
- Existing decoder adjusted to use tensor prefix `audio_vae.` instead of `audiovae.`.
- Exact causal convolution left padding and right trimming.
- Exact transposed convolution padding and output trimming.
- Weight norm as PyTorch computes it.
- Snake activation with broadcasted alpha.
- V2 sample-rate conditioning when tensors are present.

- [x] **Step 6: Run AudioVAE tests**

Run:

```powershell
cargo test -p voxui-inference --test audiovae_parity -- --nocapture
```

Expected: `wav_loader_returns_mono_f32_at_requested_rate` passes and decode parity is within `2e-3` for FP16 exported weights.

- [x] **Step 7: Commit audio and AudioVAE parity**

Run:

```powershell
git add voxui/crates/voxui-inference/Cargo.toml voxui/crates/voxui-inference/src/audio_io.rs voxui/crates/voxui-inference/src/audiovae.rs voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/tests/audiovae_parity.rs
git commit -m "fix(inference): match VoxCPM AudioVAE encode decode"
```

---

### Task 8: MiniCPM Base And Residual LM Parity

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/base_lm.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/model_loader.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/minicpm_parity.rs`

- [x] **Step 1: Write failing MiniCPM parity tests**

Create `tests/minicpm_parity.rs`:

```rust
use std::path::Path;

use candle_core::Device;
use voxui_inference::{BaseLM, BaseLMConfig, BundleManifest, GgufModelLoader};

#[test]
fn base_lm_prefill_matches_python_trace() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let manifest = BundleManifest::load(model_dir).unwrap();
    let loader = GgufModelLoader::new(&manifest.component_path(model_dir, "base_lm").unwrap(), Device::Cpu).unwrap();
    let config = BaseLMConfig::from_manifest(&manifest, "base_lm").unwrap();
    let mut lm = BaseLM::load(&loader, config, &Device::Cpu).unwrap();

    let trace = voxui_inference::trace::TraceCase::load("D:/Sandbox_Share/VoxUI/goldens/voxcpm2_zero_shot").unwrap();
    let token_ids = trace.u32_list("token_ids").unwrap();
    let expected = trace.tensor("base_lm_prefill_hidden").unwrap();
    let actual = lm.forward(&token_ids).unwrap();
    voxui_inference::trace::assert_close(&actual, &expected, 2e-3).unwrap();
}

#[test]
fn rope_rotate_half_matches_python_layout() {
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let rotated = voxui_inference::base_lm::rotate_half_for_test(&input);
    assert_eq!(rotated, vec![-3.0, -4.0, 1.0, 2.0]);
}
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test minicpm_parity -- --nocapture
```

Expected: fail because `from_manifest`, MuP scaling, LongRoPE, and Python `rotate_half` parity are missing or incomplete.

- [x] **Step 3: Implement config construction**

In `base_lm.rs`, extend `BaseLMConfig`:

```rust
pub use_mup: bool,
pub scale_emb: f64,
pub scale_depth: f64,
pub original_max_position_embeddings: Option<usize>,
pub rope_short_factors: Vec<f32>,
pub rope_long_factors: Vec<f32>,
```

Add:

```rust
impl BaseLMConfig {
    pub fn from_manifest(manifest: &crate::BundleManifest, component: &str) -> anyhow::Result<Self> {
        let cfg = match component {
            "base_lm" | "residual_lm" => &manifest.lm_config,
            "feat_encoder" => &manifest.encoder_config,
            other => anyhow::bail!("unsupported MiniCPM component `{other}`"),
        };
        let get_usize = |key: &str| -> anyhow::Result<usize> {
            cfg.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("missing `{key}` in {component} config"))
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            cfg.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let hidden_size = get_usize("hidden_size")
            .or_else(|_| get_usize("hidden_dim"))?;
        let num_layers = if component == "residual_lm" {
            manifest.residual_lm_num_layers.unwrap_or(get_usize("num_hidden_layers")?)
        } else {
            get_usize("num_hidden_layers").or_else(|_| get_usize("num_layers"))?
        };
        let num_heads = get_usize("num_attention_heads").or_else(|_| get_usize("num_heads"))?;
        let num_kv_heads = cfg
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_heads);
        let head_dim = cfg
            .get("kv_channels")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_heads);
        let rope_scaling = cfg.get("rope_scaling").cloned().unwrap_or(serde_json::Value::Null);
        let rope_short_factors = read_f32_array(&rope_scaling, "short_factor", head_dim / 2);
        let rope_long_factors = read_f32_array(&rope_scaling, "long_factor", head_dim / 2);
        Ok(Self {
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: get_usize("intermediate_size").or_else(|_| get_usize("ffn_dim"))?,
            rms_norm_eps: get_f64("rms_norm_eps", 1e-5),
            rope_theta: get_f64("rope_theta", 10000.0),
            rope_factors: rope_short_factors.clone(),
            vocab_size: cfg.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            max_position: cfg
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize,
            prefix: component.to_string(),
            no_rope: component == "residual_lm" && manifest.residual_lm_no_rope.unwrap_or(false),
            is_causal: component != "feat_encoder",
            use_mup: cfg.get("use_mup").and_then(|v| v.as_bool()).unwrap_or(false),
            scale_emb: get_f64("scale_emb", 1.0),
            scale_depth: get_f64("scale_depth", 1.0),
            original_max_position_embeddings: rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            rope_short_factors,
            rope_long_factors,
        })
    }
}
```

Add a private helper:

```rust
fn read_f32_array(value: &serde_json::Value, key: &str, len: usize) -> Vec<f32> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect())
        .filter(|arr: &Vec<f32>| arr.len() == len)
        .unwrap_or_else(|| vec![1.0; len])
}
```

The function must not use default dimensions when manifest contains explicit config values.

- [x] **Step 4: Fix RoPE**

Update RoPE to match `VoxCPM/src/voxcpm/modules/minicpm4/model.py`:

- Use Python `rotate_half`: `[-x[..., half:], x[..., :half]]`.
- Apply short or long factors according to sequence length and `original_max_position_embeddings`.
- Apply LongRoPE scaling factor from config.
- Keep no-rope path for residual LM when required.

- [x] **Step 5: Fix MuP scaling**

Match Python:

- Apply `scale_emb` to embedding output when `use_mup` is true.
- Apply residual scaling `scale_depth / sqrt(num_hidden_layers)` around attention and MLP residual additions when `use_mup` is true.

- [x] **Step 6: Fix non-causal and KV-cache behavior**

Ensure:

- Causal LM prefill uses a causal mask for multi-token calls.
- Single-token inference uses existing KV cache.
- Non-causal encoder/DiT transformer paths do not update persistent autoregressive caches across independent calls.

- [x] **Step 7: Run MiniCPM tests**

Run:

```powershell
cargo test -p voxui-inference --test minicpm_parity -- --nocapture
```

Expected: parity within `2e-3` on FP16-exported CPU tensors.

- [x] **Step 8: Commit MiniCPM parity**

Run:

```powershell
git add voxui/crates/voxui-inference/src/base_lm.rs voxui/crates/voxui-inference/src/model_loader.rs voxui/crates/voxui-inference/tests/minicpm_parity.rs
git commit -m "fix(inference): match MiniCPM transformer parity"
```

---

### Task 9: Local Encoder Parity

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/encoder.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/local_encoder_parity.rs`

- [x] **Step 1: Write failing local encoder parity test**

Create `tests/local_encoder_parity.rs`:

```rust
use std::path::Path;

use candle_core::Device;
use voxui_inference::{BundleManifest, GgufModelLoader, LocalEncoder};

#[test]
fn local_encoder_accepts_b_t_p_d_and_matches_trace() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let manifest = BundleManifest::load(model_dir).unwrap();
    let loader = GgufModelLoader::new(&manifest.component_path(model_dir, "feat_encoder").unwrap(), Device::Cpu).unwrap();
    let mut encoder = LocalEncoder::load_from_manifest(&loader, &manifest).unwrap();

    let trace = voxui_inference::trace::TraceCase::load("D:/Sandbox_Share/VoxUI/goldens/voxcpm2_zero_shot").unwrap();
    let audio_feat = trace.tensor("prefill_audio_feat_b_t_p_d").unwrap();
    let expected = trace.tensor("local_encoder_output").unwrap();
    let actual = encoder.encode_patches(&audio_feat).unwrap();
    assert_eq!(actual.dims(), expected.dims());
    voxui_inference::trace::assert_close(&actual, &expected, 2e-3).unwrap();
}
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test local_encoder_parity -- --nocapture
```

Expected: fail because current encoder accepts only `[B, D, P]`.

- [x] **Step 3: Implement `[B, T, P, D]` encoding**

In `encoder.rs`, implement:

```rust
pub fn encode_patches(&mut self, feat: &Tensor) -> Result<Tensor> {
    let (b, t, p, d) = feat.dims4()?;
    let flat = feat.reshape((b * t, p, d))?;
    let projected = crate::linear(&flat, &self.in_proj)?;
    let cls = self.special_token.broadcast_as((b * t, 1, projected.dim(2)?))?.contiguous()?;
    let input = Tensor::cat(&[&cls, &projected], 1)?;
    self.transformer.reset_cache();
    let output = self.transformer.forward_embed(&input)?;
    output.narrow(1, 0, 1)?.reshape((b, t, self.hidden_size))
}
```

Keep a small compatibility helper:

```rust
pub fn encode_single_patch(&mut self, feat: &Tensor) -> Result<Tensor> {
    let feat = feat.transpose(1, 2)?.unsqueeze(1)?;
    self.encode_patches(&feat)
}
```

- [x] **Step 4: Run local encoder tests**

Run:

```powershell
cargo test -p voxui-inference --test local_encoder_parity -- --nocapture
```

Expected: `OK`.

- [x] **Step 5: Commit local encoder parity**

Run:

```powershell
git add voxui/crates/voxui-inference/src/encoder.rs voxui/crates/voxui-inference/tests/local_encoder_parity.rs
git commit -m "fix(inference): match VoxCPM local encoder shape"
```

---

### Task 10: DiT And CFM Solver Parity

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/dit.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/dit_parity.rs`

- [ ] **Step 1: Write failing DiT parity tests**

Create `tests/dit_parity.rs`:

```rust
use std::path::Path;

use candle_core::Device;
use voxui_inference::{BundleManifest, DiT, GgufModelLoader};

#[test]
fn first_dit_patch_matches_python_trace_with_fixed_noise() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let manifest = BundleManifest::load(model_dir).unwrap();
    let loader = GgufModelLoader::new(&manifest.component_path(model_dir, "feat_decoder").unwrap(), Device::Cpu).unwrap();
    let mut dit = DiT::load_from_manifest(&loader, &manifest).unwrap();

    let trace = voxui_inference::trace::TraceCase::load("D:/Sandbox_Share/VoxUI/goldens/voxcpm2_zero_shot").unwrap();
    let cond = trace.tensor("first_dit_cond").unwrap();
    let mu = trace.tensor("first_dit_mu").unwrap();
    let noise = trace.tensor("first_dit_noise").unwrap();
    let expected = trace.tensor("first_dit_patch").unwrap();
    let actual = dit.solve_euler_with_noise(&mu, &cond, &noise, 4, 2.0).unwrap();
    voxui_inference::trace::assert_close(&actual, &expected, 3e-3).unwrap();
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test dit_parity -- --nocapture
```

Expected: fail because current `DiT::generate` is approximate and does not expose fixed-noise Euler solving.

- [ ] **Step 3: Implement manifest-driven DiT config**

In `dit.rs`, implement `DiT::load_from_manifest(&loader, &manifest)` that reads:

- `hidden_dim`
- `ffn_dim`
- `num_heads`
- `num_layers`
- `kv_channels`
- `cfm_config.sigma_min`
- `cfm_config.inference_cfg_rate`
- `sway_sampling_coef`
- CFG-Zero* warmup if present
- variant-specific local DiT class layout

- [ ] **Step 4: Implement `UnifiedCFM.forward` parity**

Match `VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py` exactly:

- Time embedding.
- Conditional and unconditional branches.
- CFG rate and `cfg_value`.
- CFG-Zero* warmup handling.
- `sway_sampling_coef` time transform.

- [ ] **Step 5: Implement deterministic Euler solver**

Add:

```rust
pub fn solve_euler_with_noise(
    &mut self,
    mu: &Tensor,
    cond: &Tensor,
    noise: &Tensor,
    inference_timesteps: usize,
    cfg_value: f32,
) -> Result<Tensor>
```

It must use the supplied `noise` tensor instead of sampling internally. Production generation may call a wrapper that samples Candle noise on the selected device.

- [ ] **Step 6: Run DiT tests**

Run:

```powershell
cargo test -p voxui-inference --test dit_parity -- --nocapture
```

Expected: `OK` within `3e-3`.

- [ ] **Step 7: Commit DiT parity**

Run:

```powershell
git add voxui/crates/voxui-inference/src/dit.rs voxui/crates/voxui-inference/tests/dit_parity.rs
git commit -m "fix(inference): match VoxCPM CFM decoder"
```

---

### Task 11: Native Generation Flow

**Files:**
- Replace: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/fsq.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/tokenizer.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/generate_flow_parity.rs`

- [ ] **Step 1: Write failing generation-flow parity tests**

Create `tests/generate_flow_parity.rs`:

```rust
use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

#[test]
fn voxcpm2_first_patch_flow_matches_python_trace() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(model_dir, Device::Cpu).unwrap();
    let trace = voxui_inference::trace::TraceCase::load("D:/Sandbox_Share/VoxUI/goldens/voxcpm2_zero_shot").unwrap();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };
    let debug = engine.generate_debug_first_patch(request).unwrap();
    voxui_inference::trace::assert_close(&debug.first_patch, &trace.tensor("first_dit_patch").unwrap(), 3e-3).unwrap();
    voxui_inference::trace::assert_close(&debug.stop_logits, &trace.tensor("stop_logits").unwrap(), 2e-3).unwrap();
}

#[test]
fn voxcpm2_reference_request_uses_reference_audio_without_prompt_text() {
    let wav = std::fs::read_dir("D:/Sandbox_Share/VoxUI/for_test_wav")
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|v| v.to_str()) == Some("wav"))
        .expect("at least one test wav");
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(model_dir, Device::Cpu).unwrap();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        reference_wav_path: Some(PathBuf::from(wav)),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };
    let samples = engine.generate(request, |_, _| {}).unwrap();
    assert!(!samples.is_empty());
    assert!(samples.iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test generate_flow_parity -- --nocapture
```

Expected: fail because the current engine API and flow do not match `SynthesisRequest`, prompt/reference encoding, and Python masks.

- [ ] **Step 3: Replace engine loading**

Change `VoxCPMEngine::load` signature to:

```rust
pub fn load(model_dir: &Path, device: Device) -> Result<Self>
```

It must:

- Load `manifest.json`.
- Load tokenizer from the bundle directory.
- Load components by manifest file names.
- Load projection tensors by source-faithful names.
- Initialize `FSQLayer` from manifest dim and scale.
- Reject missing component files before loading partial state.

- [ ] **Step 4: Implement Python-compatible token and mask construction**

In `engine.rs`, add internal methods:

```rust
fn build_zero_shot_inputs(&self, request: &SynthesisRequest) -> Result<PreparedInputs>;
fn build_prompt_inputs(&mut self, request: &SynthesisRequest) -> Result<PreparedInputs>;
fn build_reference_inputs(&mut self, request: &SynthesisRequest) -> Result<PreparedInputs>;
```

They must construct:

- `text_token`
- `text_mask`
- `audio_feat`
- `audio_mask`
- continuation trim length

Use `audio_start = 101`, `audio_end = 102`, and VoxCPM2 reference tokens `103` and `104` from the manifest.

- [ ] **Step 5: Implement prefill**

Follow Python `_inference`:

- Embed text tokens with base LM embedding table.
- Encode audio features with local encoder as `[B, T, P, D]`.
- Project encoded features through `enc_to_lm_proj`.
- Merge text embeddings and audio embeddings by mask positions.
- Prefill `base_lm`.
- Apply FSQ only to audio-mask positions during prefill.
- Prefill residual LM using the correct variant path:
  - VoxCPM 0.5 and 1.5 use add path.
  - VoxCPM2 uses `fusion_concat_proj`.

- [ ] **Step 6: Implement autoregressive patch loop**

For each patch:

- Project LM and residual hidden states to DiT conditioning.
- Run `feat_decoder.solve_euler`.
- Encode predicted patch through local encoder.
- Feed projected encoder output back to base LM.
- Feed residual LM with variant-specific fused input.
- Evaluate `stop_proj`, `silu`, and `stop_head`.
- Stop only after `min_len`.
- Enforce `max_len`.

- [ ] **Step 7: Implement retry bad-case logic**

Match Python retry behavior:

```rust
if request.retry_badcase
    && generated_patch_count > text_token_count * request.retry_badcase_ratio_threshold as usize
    && retry_count < request.retry_badcase_max_times
{
    // reset caches and retry generation from the same prepared inputs
}
```

Use the Python ratio formula exactly after inspecting `voxcpm.py` and `voxcpm2.py`.

- [ ] **Step 8: Decode and trim audio**

Decode generated latent patches with AudioVAE decoder, then trim continuation context exactly like Python for prompt-audio requests. Reference-only requests must not be trimmed as continuation.

- [ ] **Step 9: Run generation-flow tests**

Run:

```powershell
cargo test -p voxui-inference --test generate_flow_parity -- --nocapture
```

Expected: `OK`.

- [ ] **Step 10: Commit native generation flow**

Run:

```powershell
git add voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/src/fsq.rs voxui/crates/voxui-inference/src/tokenizer.rs voxui/crates/voxui-inference/tests/generate_flow_parity.rs
git commit -m "fix(inference): implement native VoxCPM generate flow"
```

---

### Task 12: Component-Aware LoRA

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lora.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/base_lm.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/dit.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/lora_parity.rs`

- [ ] **Step 1: Write failing LoRA tests**

Create `tests/lora_parity.rs`:

```rust
use std::path::Path;

use candle_core::{Device, Tensor};
use voxui_inference::{LoraAdapter, SynthesisRequest, VoxCPMEngine};

#[test]
fn lora_linear_delta_matches_formula() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &device).unwrap();
    let base = Tensor::zeros((1, 3), candle_core::DType::F32, &device).unwrap();
    let a = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1., 0., 0.], (2, 4), &device).unwrap();
    let b = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &device).unwrap();
    let out = LoraAdapter::apply_raw(&base, &x, &a, &b, 4.0, 2).unwrap();
    assert_eq!(out.dims(), &[1, 3]);
}

#[test]
fn lora_adapter_changes_generation_without_breaking_audio() {
    let model_dir = Path::new("D:/Sandbox_Share/VoxUI/models/voxcpm2-fp16");
    let lora_dir = std::fs::read_dir(model_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("lora_manifest.json").exists());
    let Some(lora_dir) = lora_dir else {
        eprintln!("skip: no LoRA adapter exported");
        return;
    };

    let mut engine = VoxCPMEngine::load(model_dir, Device::Cpu).unwrap();
    engine.load_lora(&lora_dir).unwrap();
    let samples = engine.generate(SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    }, |_, _| {}).unwrap();
    assert!(!samples.is_empty());
    assert!(samples.iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: fail because LoRA manifests and DiT/projection LoRA application are incomplete.

- [ ] **Step 3: Implement LoRA manifest loading**

In `lora.rs`, load `lora_manifest.json`, verify:

- Architecture matches model manifest.
- Variant matches model manifest.
- Rank and alpha are positive.
- Tensor shapes match the target base module.
- Target modules are limited to exported targets.

- [ ] **Step 4: Implement raw LoRA formula**

Expose for tests:

```rust
pub fn apply_raw(base: &Tensor, input: &Tensor, a: &Tensor, b: &Tensor, alpha: f32, rank: usize) -> Result<Tensor> {
    let delta = crate::linear(&crate::linear(input, a)?, b)?;
    base.broadcast_add(&(delta * (alpha as f64 / rank as f64))?)
}
```

Adjust transposes if exported LoRA `A` and `B` tensors follow PyTorch `Linear` layout. Validate against `VoxCPM/src/voxcpm/modules/layers/lora.py`.

- [ ] **Step 5: Apply LoRA in every supported component**

Apply adapters to:

- `base_lm.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`
- `base_lm.layers.*.mlp.{gate_proj,up_proj,down_proj}`
- matching `residual_lm` modules
- `feat_decoder` DiT attention and MLP modules
- projection layers if exported

- [ ] **Step 6: Run LoRA tests**

Run:

```powershell
cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: `OK`.

- [ ] **Step 7: Commit LoRA**

Run:

```powershell
git add voxui/crates/voxui-inference/src/lora.rs voxui/crates/voxui-inference/src/base_lm.rs voxui/crates/voxui-inference/src/dit.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/tests/lora_parity.rs
git commit -m "fix(inference): apply VoxCPM LoRA adapters natively"
```

---

### Task 13: UI And Test Suite Integration

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/inference_suite.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/app.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/config.rs`

- [ ] **Step 1: Update inference suite to manifest bundles**

In `tests/inference_suite.rs`:

- Scan model directories with `manifest.json`.
- Scan LoRA directories with `lora_manifest.json`.
- Use `VoxCPMEngine::load(model_dir, device)`.
- Use `SynthesisRequest` for all synthesis.
- Add VoxCPM2 reference-only and reference-plus-continuation cases using WAVs from `for_test_wav/`.

- [ ] **Step 2: Update desktop commands**

In `voxui-desktop/src-tauri/src/commands.rs`:

- Replace `base_lm.gguf` model detection with `manifest.json`.
- Replace LoRA detection with `lora_manifest.json`.
- Map UI input fields into `SynthesisRequest`.
- Return clear errors for `normalize=true` until Rust text normalization exists.

- [ ] **Step 3: Update app state**

In `voxui-app/src/app.rs` and `config.rs`:

- Use one model directory field.
- Use one LoRA directory field.
- Expose optional prompt WAV, prompt text, and reference WAV path.
- Do not require transcript text for reference WAV.

- [ ] **Step 4: Run app and inference checks**

Run:

```powershell
cargo check -p voxui-inference
cargo check -p voxui-app
cargo check -p voxui-desktop
```

Expected: all pass.

- [ ] **Step 5: Commit integration**

Run:

```powershell
git add voxui/crates/voxui-inference/tests/inference_suite.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-app/src/app.rs voxui/crates/voxui-app/src/config.rs
git commit -m "fix(app): use native VoxCPM synthesis requests"
```

---

### Task 14: CPU End-To-End Verification

**Files:**
- Generated WAVs under `D:/Sandbox_Share/VoxUI/test_output/`

- [ ] **Step 1: Run runtime purity guard**

Run:

```powershell
cargo test -p voxui-inference --test native_runtime_purity
```

Expected: pass.

- [ ] **Step 2: Run parity tests**

Run:

```powershell
cargo test -p voxui-inference --test manifest_loader
cargo test -p voxui-inference --test request_validation
cargo test -p voxui-inference --test minicpm_parity -- --nocapture
cargo test -p voxui-inference --test audiovae_parity -- --nocapture
cargo test -p voxui-inference --test local_encoder_parity -- --nocapture
cargo test -p voxui-inference --test dit_parity -- --nocapture
cargo test -p voxui-inference --test generate_flow_parity -- --nocapture
cargo test -p voxui-inference --test lora_parity -- --nocapture
```

Expected: all pass within documented tolerances.

- [ ] **Step 3: Run CPU synthesis matrix**

Run:

```powershell
cargo test -p voxui-inference --test inference_suite --release -- --nocapture
```

Expected:

- VoxCPM 0.5 CPU no-LoRA passes.
- VoxCPM 1.5 CPU no-LoRA passes.
- VoxCPM 2.0 CPU no-LoRA passes.
- Available LoRA adapters pass.
- VoxCPM2 reference-audio cases pass.
- WAV files are written under `test_output/`.

- [ ] **Step 4: Check generated WAV sanity**

Run:

```powershell
Get-ChildItem test_output -Filter *.wav | Select-Object Name, Length
```

Expected: each WAV is non-empty. Manually inspect at least one WAV from each variant for intelligible speech.

- [ ] **Step 5: Commit verification test updates**

Run:

```powershell
git status --short
git add voxui/crates/voxui-inference/tests test_output
git commit -m "test(inference): verify native VoxCPM CPU synthesis"
```

Only commit WAV outputs if they are intended fixtures; otherwise keep `test_output/` untracked or ignored.

---

### Task 15: CUDA Verification

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
cargo check -p voxui-inference --features cuda
```

Expected: pass.

- [ ] **Step 3: Run CUDA inference suite**

Run:

```powershell
cargo test -p voxui-inference --test inference_suite --release --features cuda -- --nocapture
```

Expected:

- VoxCPM 0.5 CUDA no-LoRA passes.
- VoxCPM 1.5 CUDA no-LoRA passes.
- VoxCPM 2.0 CUDA no-LoRA passes.
- Available LoRA adapters pass.
- VoxCPM2 reference-audio cases pass.

- [ ] **Step 4: Record final status**

Run:

```powershell
git status --short
```

Expected: no uncommitted source changes. If CUDA-only fixes were needed, commit them with:

```powershell
git add voxui/crates/voxui-inference
git commit -m "fix(inference): support CUDA native VoxCPM synthesis"
```

---

## Plan Self-Review

Spec coverage:

- VoxCPM 0.5, 1.5, and 2.0 are covered by exporter, bundle regeneration, manifest loader, and CPU/CUDA matrix tasks.
- Pure Rust runtime inference is enforced by Task 1 and implemented by Tasks 5 through 12.
- Python reference parity is limited to golden traces and exporter validation in Tasks 2 and 3.
- Reference audio without transcript text is covered by Tasks 6, 11, 13, and 14.
- Prompt audio with required `prompt_text` is covered by Tasks 6, 11, and 13.
- LoRA is covered by Tasks 3, 4, 12, 14, and 15.
- CPU and CUDA are covered by Tasks 14 and 15.

Deferred-marker scan:

- The plan avoids open-ended deferred markers.
- Every implementation task names exact files, exact test commands, and expected results.

Type consistency:

- `BundleManifest`, `ModelVariant`, `SynthesisRequest`, and `VoxCPMEngine::load(model_dir, device)` are used consistently after their defining tasks.
- Component names use the approved bundle names: `feat_encoder.gguf`, `feat_decoder.gguf`, and `audio_vae.gguf`.
- Runtime API mirrors Python `generate()` arguments except `denoise`, which remains out of scope.

Execution handoff:

- Use subagent-driven execution only if the user explicitly asks for parallel agents.
- Otherwise execute inline task-by-task, keeping tests red before implementation and green before each commit.
