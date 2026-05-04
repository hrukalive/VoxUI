# VoxCPM Generate Bridge Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken GGUF-native synthesis path with a Python `VoxCPM.generate()`-compatible engine callable from Rust, and regenerate model bundles that support VoxCPM 0.5, 1.5, and 2.0.

**Architecture:** Exported `models/*` entries become manifest-based bundles that point to source VoxCPM model directories and optional LoRA checkpoints. `voxui-inference` preserves the existing Rust `VoxCPMEngine` API, but internally owns a persistent Python bridge process that loads the Python VoxCPM implementation and executes the public `generate()` semantics. This fixes audible synthesis first while leaving room for a later native implementation behind the same Rust API.

**Tech Stack:** Python `unittest`, local VoxCPM package, Rust `std::process`, `serde`, `serde_json`, `base64`, Candle device selection, Cargo tests.

---

## File Structure

- Modify `exporter/export_voxcpm.py`: replace the misleading GGUF conversion flow with a manifest bundle exporter that validates source model/config/tokenizer files and writes `manifest.json`.
- Create `exporter/tests/test_bundle_export.py`: unit tests for manifest generation and argument compatibility.
- Create `voxui/crates/voxui-inference/python/voxcpm_bridge.py`: persistent JSON-lines Python bridge that loads `VoxCPM` and calls `generate()`.
- Modify `voxui/crates/voxui-inference/Cargo.toml`: add serialization/base64 dependencies.
- Replace `voxui/crates/voxui-inference/src/engine.rs`: Python bridge-backed `VoxCPMEngine` with the existing `load`, `load_lora`, `unload_lora`, `synthesize`, `sample_rate`, `patch_size`, and `architecture` methods.
- Modify `voxui/crates/voxui-inference/src/lib.rs`: export new request types if needed while keeping `VoxCPMEngine`.
- Modify `voxui/crates/voxui-inference/tests/inference_suite.rs`: scan manifest bundles, add VoxCPM2 reference-audio tests, and keep CPU/CUDA test matrix.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: scan `manifest.json` model bundles and LoRA manifests instead of `base_lm.gguf`.
- Modify `voxui/crates/voxui-app/src/app.rs`: scan `manifest.json` model bundles and LoRA manifests instead of `base_lm.gguf`.
- Modify `voxui/crates/voxui-app/src/config.rs` if needed to fix `lora_dir` / `lora_path` naming mismatch.

---

### Task 1: Exporter Manifest Tests

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/exporter/tests/test_bundle_export.py`
- Modify: `D:/Sandbox_Share/VoxUI/exporter/export_voxcpm.py`

- [ ] **Step 1: Write failing exporter tests**

Create `exporter/tests/test_bundle_export.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from exporter.export_voxcpm import export_bundle


class BundleExportTests(unittest.TestCase):
    def make_model(self, root: Path, architecture: str = "voxcpm2") -> Path:
        model = root / "VoxCPM2"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({
                "architecture": architecture,
                "patch_size": 4,
                "feat_dim": 64,
                "audio_vae_config": {
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                    "latent_dim": 64,
                    "decoder_rates": [8, 6, 5, 2, 2, 2],
                    "encoder_rates": [2, 5, 8, 8],
                },
            }),
            encoding="utf-8",
        )
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "special_tokens_map.json").write_text("{}", encoding="utf-8")
        return model

    def test_export_bundle_writes_manifest_and_tokenizers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.make_model(root)
            out = root / "bundle"

            manifest = export_bundle(
                model_dir=model,
                output_dir=out,
                variant="2.0",
                python_source_dir=root / "VoxCPM" / "src",
                lora_dir=None,
                quantization={"lm": "fp16", "encoder": "fp16", "dit": "fp16", "vae": "fp16"},
            )

            self.assertEqual(manifest["architecture"], "voxcpm2")
            self.assertEqual(manifest["variant"], "2.0")
            self.assertEqual(manifest["source_model_dir"], str(model.resolve()))
            self.assertEqual(manifest["sample_rate"], 48000)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "tokenizer.json").exists())

    def test_export_bundle_rejects_missing_tokenizer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.make_model(root)
            (model / "tokenizer.json").unlink()

            with self.assertRaises(FileNotFoundError):
                export_bundle(
                    model_dir=model,
                    output_dir=root / "bundle",
                    variant="2.0",
                    python_source_dir=root / "VoxCPM" / "src",
                    lora_dir=None,
                    quantization={"lm": "fp16", "encoder": "fp16", "dit": "fp16", "vae": "fp16"},
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m unittest exporter.tests.test_bundle_export
```

Expected: fail because `export_bundle` is not defined or still writes GGUF-style output.

- [ ] **Step 3: Implement manifest exporter**

In `exporter/export_voxcpm.py`, add `export_bundle(...)` and make the CLI call it. Preserve the existing quantization arguments for command compatibility, but record them in the manifest instead of pretending to emit valid native weights.

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m unittest exporter.tests.test_bundle_export
```

Expected: `OK`.

- [ ] **Step 5: Commit exporter tests and implementation**

Run:

```powershell
git add exporter/export_voxcpm.py exporter/tests/test_bundle_export.py
git commit -m "fix(exporter): write VoxCPM manifest bundles"
```

---

### Task 2: Generate Local Model Bundles

**Files:**
- Modify generated artifacts under `D:/Sandbox_Share/VoxUI/models/`

- [ ] **Step 1: Export VoxCPM 0.5 bundle**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05b --variant 0.5 --lora-dir VoxCPM/ft0.5/latest
```

Expected: `models/voxcpm05b/manifest.json` and `models/voxcpm05b/lora_Akit/manifest.json`.

- [ ] **Step 2: Export VoxCPM 1.5 bundle**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15 --variant 1.5 --lora-dir VoxCPM/ft1.5/latest
```

Expected: `models/voxcpm15/manifest.json` and `models/voxcpm15/lora_Akit/manifest.json`.

- [ ] **Step 3: Export VoxCPM 2.0 bundle**

Run:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2 --variant 2.0 --lora-dir VoxCPM/ft2/latest
```

Expected: `models/voxcpm2/manifest.json` and `models/voxcpm2/lora_Akit/manifest.json`.

- [ ] **Step 4: Inspect manifests**

Run:

```powershell
Get-ChildItem models -Filter manifest.json -Recurse | Select-Object FullName
```

Expected: manifests for all three model bundles and LoRA bundle directories.

- [ ] **Step 5: Commit bundle manifests**

Run:

```powershell
git add models/voxcpm05b models/voxcpm15 models/voxcpm2
git commit -m "chore(models): add VoxCPM manifest bundles"
```

---

### Task 3: Python Bridge Tests And Script

**Files:**
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/python/voxcpm_bridge.py`
- Create: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/bridge_contract.rs`

- [ ] **Step 1: Write Rust contract test for missing bridge path**

Create `voxui/crates/voxui-inference/tests/bridge_contract.rs`:

```rust
use std::path::Path;

#[test]
fn python_bridge_script_is_present() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("python/voxcpm_bridge.py");
    assert!(script.exists(), "missing bridge script at {}", script.display());
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test bridge_contract
```

Expected: fail because `python/voxcpm_bridge.py` does not exist.

- [ ] **Step 3: Create bridge script**

Create `voxcpm_bridge.py` with JSON-lines protocol:

```python
#!/usr/bin/env python
import argparse
import base64
import json
import os
import struct
import sys
import traceback
from pathlib import Path

import numpy as np


def write_msg(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lora_config(lora_config_path):
    from voxcpm.model.voxcpm import LoRAConfig

    with open(lora_config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = raw.get("lora_config", raw)
    return LoRAConfig(**cfg)


class Bridge:
    def __init__(self, manifest_path, device, lora_manifest_path=None):
        self.manifest = load_manifest(manifest_path)
        self.device = device
        self.model = None
        self.lora_manifest_path = lora_manifest_path
        python_source_dir = self.manifest.get("python_source_dir")
        if python_source_dir:
            sys.path.insert(0, python_source_dir)
        self.load_model(lora_manifest_path)

    def load_model(self, lora_manifest_path=None):
        from voxcpm import VoxCPM

        lora_config = None
        lora_weights_path = None
        if lora_manifest_path:
            lora_manifest = load_manifest(lora_manifest_path)
            lora_config_path = lora_manifest["lora_config_path"]
            lora_config = build_lora_config(lora_config_path)
            lora_weights_path = lora_manifest["lora_weights_path"]

        self.model = VoxCPM(
            voxcpm_model_path=self.manifest["source_model_dir"],
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=False,
            device=self.device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
        )
        self.lora_manifest_path = lora_manifest_path

    def synthesize(self, req):
        wav = self.model.generate(
            text=req["text"],
            prompt_wav_path=req.get("prompt_wav_path"),
            prompt_text=req.get("prompt_text"),
            reference_wav_path=req.get("reference_wav_path"),
            cfg_value=float(req.get("cfg_value", 2.0)),
            inference_timesteps=int(req.get("inference_timesteps", 10)),
            min_len=int(req.get("min_len", 2)),
            max_len=int(req.get("max_len", 4096)),
            normalize=bool(req.get("normalize", False)),
            denoise=False,
            retry_badcase=bool(req.get("retry_badcase", True)),
            retry_badcase_max_times=int(req.get("retry_badcase_max_times", 3)),
            retry_badcase_ratio_threshold=float(req.get("retry_badcase_ratio_threshold", 6.0)),
        )
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        raw = arr.tobytes(order="C")
        return {
            "ok": True,
            "sample_rate": int(self.manifest["sample_rate"]),
            "num_samples": int(arr.size),
            "samples_f32_le_b64": base64.b64encode(raw).decode("ascii"),
        }

    def handle(self, msg):
        cmd = msg.get("cmd")
        if cmd == "synthesize":
            return self.synthesize(msg["request"])
        if cmd == "load_lora":
            self.load_model(msg["lora_manifest_path"])
            return {"ok": True}
        if cmd == "unload_lora":
            self.load_model(None)
            return {"ok": True}
        if cmd == "shutdown":
            return {"ok": True, "shutdown": True}
        return {"ok": False, "error": f"unknown command: {cmd}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-manifest")
    args = parser.parse_args()

    bridge = Bridge(args.manifest, args.device, args.lora_manifest)
    write_msg({"ok": True, "ready": True, "architecture": bridge.manifest["architecture"], "sample_rate": bridge.manifest["sample_rate"], "patch_size": bridge.manifest["patch_size"]})

    for line in sys.stdin:
        try:
            msg = json.loads(line)
            resp = bridge.handle(msg)
            write_msg(resp)
            if resp.get("shutdown"):
                break
        except Exception as exc:
            write_msg({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run contract test**

Run:

```powershell
cargo test -p voxui-inference --test bridge_contract
```

Expected: pass.

- [ ] **Step 5: Commit bridge script**

Run:

```powershell
git add voxui/crates/voxui-inference/python/voxcpm_bridge.py voxui/crates/voxui-inference/tests/bridge_contract.rs
git commit -m "feat(inference): add VoxCPM Python bridge"
```

---

### Task 4: Rust Engine Bridge

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/Cargo.toml`
- Replace: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/engine.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/src/lib.rs`

- [ ] **Step 1: Write failing Rust engine test**

Add to `bridge_contract.rs`:

```rust
use candle_core::Device;
use voxui_inference::VoxCPMEngine;

#[test]
fn engine_rejects_directory_without_manifest() {
    let missing = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let err = VoxCPMEngine::load(missing, missing, Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("manifest.json"));
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cargo test -p voxui-inference --test bridge_contract
```

Expected: fail or panic because current engine tries to load `base_lm.gguf`.

- [ ] **Step 3: Add dependencies**

In `voxui/crates/voxui-inference/Cargo.toml`, add:

```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
base64 = "0.22"
```

- [ ] **Step 4: Replace engine internals**

Replace `engine.rs` with a bridge-backed implementation that:

- Reads `manifest.json`.
- Selects Python executable from `VOXCPM_PYTHON` or `%USERPROFILE%\py_env\voxcpm\Scripts\python.exe`.
- Spawns `voxcpm_bridge.py`.
- Sends synthesize/load_lora/unload_lora JSON commands.
- Decodes `samples_f32_le_b64` into `Vec<f32>`.
- Preserves `sample_rate()`, `patch_size()`, `architecture()`, `set_dit_steps()`, `load_lora()`, `unload_lora()`, and `synthesize()`.

- [ ] **Step 5: Run contract test**

Run:

```powershell
cargo test -p voxui-inference --test bridge_contract
```

Expected: pass.

- [ ] **Step 6: Commit Rust bridge engine**

Run:

```powershell
git add voxui/crates/voxui-inference/Cargo.toml voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/tests/bridge_contract.rs
git commit -m "fix(inference): route synthesis through VoxCPM generate bridge"
```

---

### Task 5: App And Test Bundle Scanning

**Files:**
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-inference/tests/inference_suite.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/app.rs`
- Modify: `D:/Sandbox_Share/VoxUI/voxui/crates/voxui-app/src/config.rs` if needed.

- [ ] **Step 1: Write failing scan expectations**

Change model scanning in tests and UI to require `manifest.json`; LoRA directories require `manifest.json`.

- [ ] **Step 2: Run cargo check to observe current failures**

Run:

```powershell
cargo check -p voxui-inference
```

Expected: failures until scanning and config naming are consistent.

- [ ] **Step 3: Update scanners**

Replace `base_lm.gguf` checks with `manifest.json` checks. Replace `lora_base_lm.gguf` checks with LoRA `manifest.json` checks.

- [ ] **Step 4: Fix config naming**

Use the existing `AppConfig.lora_dir` field consistently, or rename all uses to `lora_path` in one commit. The minimal fix is to change app code references from `config.lora_path` to `config.lora_dir` and struct initialization from `lora_path:` to `lora_dir:`.

- [ ] **Step 5: Run cargo check**

Run:

```powershell
cargo check -p voxui-inference
cargo check -p voxui-app
```

Expected: both pass.

- [ ] **Step 6: Commit scan/config fixes**

Run:

```powershell
git add voxui/crates/voxui-inference/tests/inference_suite.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-app/src/app.rs voxui/crates/voxui-app/src/config.rs
git commit -m "fix(app): scan VoxCPM manifest bundles"
```

---

### Task 6: CPU End-To-End Verification

**Files:**
- Modify test outputs under `D:/Sandbox_Share/VoxUI/test_output/`

- [ ] **Step 1: Activate Python environment**

Run:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
```

Expected: prompt uses the VoxCPM Python environment.

- [ ] **Step 2: Run one CPU synthesis test**

Run:

```powershell
cargo test -p voxui-inference --test inference_suite --release -- voxcpm2_cpu --nocapture
```

Expected: WAV written under `test_output/`, finite samples, non-silent RMS.

- [ ] **Step 3: Run full CPU matrix**

Run:

```powershell
cargo test -p voxui-inference --test inference_suite --release -- --nocapture
```

Expected: VoxCPM 0.5, 1.5, and 2.0 CPU no-LoRA/LoRA/reference tests pass. CUDA-specific tests skip without the `cuda` feature.

- [ ] **Step 4: Commit verification-related test updates**

Run:

```powershell
git add voxui/crates/voxui-inference/tests/inference_suite.rs test_output
git commit -m "test(inference): verify VoxCPM generate bridge outputs"
```

---

### Task 7: CUDA End-To-End Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Set CUDA build environment**

Run:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
& ~\py_env\voxcpm\Scripts\activate.ps1
```

- [ ] **Step 2: Run CUDA matrix**

Run:

```powershell
cargo test -p voxui-inference --test inference_suite --release --features cuda -- --nocapture
```

Expected: VoxCPM 0.5, 1.5, and 2.0 CUDA no-LoRA/LoRA/reference tests pass.

- [ ] **Step 3: Commit if test fixture output changed**

Run:

```powershell
git status --short
```

Expected: no source changes. If WAV fixtures are intentionally tracked, add and commit them.

---

## Plan Self-Review

Spec coverage:

- Exporter rewrite is covered by Tasks 1 and 2.
- `generate()` API parity is covered by Tasks 3 and 4.
- VoxCPM 0.5, 1.5, and 2.0 support is covered by Tasks 2, 6, and 7.
- LoRA support is covered by Tasks 2, 3, 4, 6, and 7.
- Reference audio support is covered by Tasks 3, 4, 6, and 7.
- CPU/CUDA support is covered by Tasks 6 and 7.

Placeholder scan:

- The plan contains no `TODO`/`TBD` placeholders.
- Each task has concrete files and commands.

Type consistency:

- Public Rust compatibility remains `VoxCPMEngine::load`, `load_lora`, `unload_lora`, and `synthesize`.
- Python-facing request names match `VoxCPM.generate()` except `denoise`, which is intentionally forced false inside the bridge.
