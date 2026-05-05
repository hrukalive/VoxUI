# VoxUI Q4 Matrix Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mixed q4 GGUF export coverage, include q4 bundles in the inference matrix with sentence-length Chinese and English cases, document concise build/test commands, make source path JSON fields informational, and add debug-mode desktop model-load logging.

**Architecture:** The exporter gains a named `q4-lm` profile that only quantizes lower-sensitivity LM/projection components. Runtime manifest loading treats source provenance fields as optional, while component filenames and model settings remain required. Desktop diagnostics use existing Rust `log`/`env_logger` and browser console logging, gated by debug builds for frontend noise.

**Tech Stack:** Python exporter/unit tests, Rust workspace crates (`voxui-inference`, `voxui-gguf`, `voxui-desktop`), Tauri 2, Leptos CSR, Cargo, PowerShell.

---

## File Structure

- Modify `exporter/export_voxcpm.py`: add `--quant-profile`, `profile_default_quant_args`, and `resolve_quant_args`.
- Modify `exporter/tests/test_export_manifest.py`: test q4-lm profile defaults and CLI override semantics.
- Modify `voxui/crates/voxui-inference/src/manifest.rs`: make provenance fields optional with serde defaults.
- Modify `voxui/crates/voxui-inference/tests/manifest_loader.rs`: prove missing provenance fields parse successfully.
- Modify `voxui/crates/voxui-inference/tests/inference_suite.rs`: replace short/mojibake strings with sentence-length Chinese/English strings, add q4-lm targeted tests, and keep full discovery matrix.
- Modify `voxui/crates/voxui-inference/src/engine.rs`: add component-level debug logs during model load.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: add command-level debug/warn/error logs for model load, LoRA, and synthesis busy/error paths.
- Modify `voxui/crates/voxui-desktop/src/app.rs`: add debug-build browser console logs around startup and model load invokes.
- Modify `README.txt`: replace the one-line build note with concise command lines.
- Generate local artifacts under `models/voxcpm05-q4-lm`, `models/voxcpm15-q4-lm`, `models/voxcpm2-q4-lm`, and `test_output/*.wav`.

---

### Task 1: Add Exporter Q4-LM Profile

**Files:**
- Modify: `exporter/export_voxcpm.py`
- Modify: `exporter/tests/test_export_manifest.py`

- [ ] **Step 1: Write failing exporter profile tests**

Add `resolve_quant_args` to the import list in `exporter/tests/test_export_manifest.py`:

```python
from exporter.export_voxcpm import build_manifest, partition_weights, resolve_quant_args, validate_required_tensors
```

Add these tests inside `ExportManifestTests`:

```python
    def test_q4_lm_profile_quantizes_only_lower_sensitivity_components_for_voxcpm2(self):
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

    def test_q4_lm_profile_keeps_non_v2_vae_fp16(self):
        quant_args = resolve_quant_args(
            variant="1.5",
            profile="q4-lm",
            quant_lm=None,
            quant_encoder=None,
            quant_dit=None,
            quant_vae=None,
        )
        self.assertEqual(quant_args["quant_lm"], "q4")
        self.assertEqual(quant_args["quant_encoder"], "fp16")
        self.assertEqual(quant_args["quant_dit"], "fp16")
        self.assertEqual(quant_args["quant_vae"], "fp16")

    def test_quant_profile_allows_explicit_component_override(self):
        quant_args = resolve_quant_args(
            variant="2.0",
            profile="q4-lm",
            quant_lm=None,
            quant_encoder=None,
            quant_dit="q8",
            quant_vae="fp16",
        )
        self.assertEqual(quant_args["quant_lm"], "q4")
        self.assertEqual(quant_args["quant_dit"], "q8")
        self.assertEqual(quant_args["quant_vae"], "fp16")
```

- [ ] **Step 2: Run exporter tests and verify they fail**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: FAIL with an import error or missing symbol for `resolve_quant_args`.

- [ ] **Step 3: Implement q4-lm profile resolution**

In `exporter/export_voxcpm.py`, add this code after `QUANT_ARG_MAP`:

```python
QUANT_PROFILES = ("manual", "fp16", "q4-lm")


def profile_default_quant_args(profile: str, variant: str) -> dict[str, str]:
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
    if profile == "q4-lm":
        return {
            "quant_lm": "q4",
            "quant_encoder": "fp16",
            "quant_dit": "fp16",
            "quant_vae": "f32" if variant == "2.0" else "fp16",
        }
    raise ValueError(f"Unknown quantization profile {profile!r}; expected one of {list(QUANT_PROFILES)}")


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
```

Update `main()` argument parsing in `exporter/export_voxcpm.py`:

```python
    parser.add_argument("--quant-profile", default="manual", choices=QUANT_PROFILES)
    parser.add_argument("--quant-lm", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-encoder", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-dit", default=None, choices=sorted(QUANT_MAP))
    parser.add_argument("--quant-vae", default=None, choices=sorted(QUANT_MAP))
```

Replace the current `quant_args = { ... }` block in `main()` with:

```python
    quant_args = resolve_quant_args(
        variant=args.variant,
        profile=args.quant_profile,
        quant_lm=args.quant_lm,
        quant_encoder=args.quant_encoder,
        quant_dit=args.quant_dit,
        quant_vae=args.quant_vae,
    )
```

- [ ] **Step 4: Run exporter tests and verify they pass**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
```

Expected: PASS for all exporter manifest tests.

- [ ] **Step 5: Commit exporter profile changes**

Run:

```powershell
git add exporter/export_voxcpm.py exporter/tests/test_export_manifest.py
git commit -m "feat(exporter): add q4 lm quantization profile"
```

---

### Task 2: Make Source Provenance Fields Informational

**Files:**
- Modify: `voxui/crates/voxui-inference/src/manifest.rs`
- Modify: `voxui/crates/voxui-inference/tests/manifest_loader.rs`

- [ ] **Step 1: Write failing manifest loader test**

Add this test to `voxui/crates/voxui-inference/tests/manifest_loader.rs`:

```rust
#[test]
fn manifest_accepts_missing_source_provenance_fields() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("manifest.json"),
        r#"{
            "schema_version": 1,
            "architecture": "voxcpm2",
            "variant": "2.0",
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
    assert!(manifest.source_model_dir.is_none());
    assert!(manifest.source_weight_format.is_none());
    assert_eq!(manifest.variant, ModelVariant::VoxCpm2);
}
```

- [ ] **Step 2: Run manifest test and verify it fails**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test manifest_loader manifest_accepts_missing_source_provenance_fields
```

Expected: FAIL while parsing `manifest.json` because `source_model_dir` and `source_weight_format` are missing.

- [ ] **Step 3: Make provenance fields optional**

In `voxui/crates/voxui-inference/src/manifest.rs`, replace these fields:

```rust
    pub source_model_dir: String,
    pub source_weight_format: String,
```

with:

```rust
    #[serde(default)]
    pub source_model_dir: Option<String>,
    #[serde(default)]
    pub source_weight_format: Option<String>,
```

- [ ] **Step 4: Run manifest loader tests**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test manifest_loader
```

Expected: PASS.

- [ ] **Step 5: Commit manifest loader changes**

Run:

```powershell
git add voxui/crates/voxui-inference/src/manifest.rs voxui/crates/voxui-inference/tests/manifest_loader.rs
git commit -m "fix(inference): treat source manifest fields as provenance"
```

---

### Task 3: Extend Inference Suite Sentence and Q4 Coverage

**Files:**
- Modify: `voxui/crates/voxui-inference/tests/inference_suite.rs`

- [ ] **Step 1: Write a cheap failing assertion for sentence-length matrix inputs**

Add this test near the existing `#[test]` functions in `voxui/crates/voxui-inference/tests/inference_suite.rs`:

```rust
#[test]
fn matrix_text_inputs_are_sentence_length() {
    assert!(TEXT_ZH.chars().count() >= 20);
    assert!(TEXT_EN.split_whitespace().count() >= 10);
}
```

- [ ] **Step 2: Run the new assertion and verify it fails on the current Chinese text**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
```

Expected: FAIL because the current Chinese test text is not a valid sentence-length Unicode string.

- [ ] **Step 3: Replace test strings and request helper**

In `voxui/crates/voxui-inference/tests/inference_suite.rs`, replace the current constants:

```rust
const TEST_DIT_STEPS: usize = 10;
const TEXT_ZH: &str = "ä½ å¥½ï¼Œæ¬¢è¿Žæ¥åˆ°ç›´æ’­é—´ï¼";
const TEXT_EN: &str = "Hello, welcome to the stream!";
```

with:

```rust
const TEST_DIT_STEPS: usize = 10;
const TEST_MAX_LEN: usize = 6;
const TEXT_ZH: &str = "\u{4eca}\u{5929}\u{7684}\u{76f4}\u{64ad}\u{5df2}\u{7ecf}\u{51c6}\u{5907}\u{597d}\u{4e86}\u{ff0c}\u{8bf7}\u{7528}\u{6e29}\u{548c}\u{81ea}\u{7136}\u{7684}\u{8bed}\u{6c14}\u{5411}\u{89c2}\u{4f17}\u{6253}\u{4e2a}\u{62db}\u{547c}\u{3002}";
const TEXT_EN: &str = "The audience is already waiting, so please introduce tonight's stream in a calm and friendly voice.";
```

Replace `short_request` with:

```rust
fn sentence_request(text: &str) -> SynthesisRequest {
    SynthesisRequest {
        text: text.to_string(),
        inference_timesteps: TEST_DIT_STEPS,
        min_len: 1,
        max_len: TEST_MAX_LEN,
        retry_badcase: false,
        ..SynthesisRequest::default()
    }
}
```

Replace every `short_request(` call in this file with `sentence_request(`.

- [ ] **Step 4: Add q4-lm targeted model tests**

Add these tests after the existing fp16 CPU tests:

```rust
#[test]
fn voxcpm05_q4_lm_cpu() {
    test_model_on_device("voxcpm05-q4-lm", get_cpu_device());
}

#[test]
fn voxcpm15_q4_lm_cpu() {
    test_model_on_device("voxcpm15-q4-lm", get_cpu_device());
}

#[test]
fn voxcpm2_q4_lm_cpu() {
    test_model_on_device("voxcpm2-q4-lm", get_cpu_device());
}
```

Add these tests after the existing fp16 CUDA tests:

```rust
#[test]
fn voxcpm05_q4_lm_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm05-q4-lm", device);
}

#[test]
fn voxcpm15_q4_lm_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm15-q4-lm", device);
}

#[test]
fn voxcpm2_q4_lm_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm2-q4-lm", device);
}
```

- [ ] **Step 5: Run the cheap assertion and compile the inference suite**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
cargo test -p voxui-inference --test inference_suite --no-run
```

Expected: PASS for the sentence assertion and successful test binary compilation.

- [ ] **Step 6: Commit inference suite changes**

Run:

```powershell
git add voxui/crates/voxui-inference/tests/inference_suite.rs
git commit -m "test(inference): include q4 lm sentence matrix cases"
```

---

### Task 4: Add Desktop and Engine Debug Logs

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Add backend model-load diagnostics in inference engine**

In `voxui/crates/voxui-inference/src/engine.rs`, change the first import:

```rust
use std::path::Path;
```

to:

```rust
use std::path::Path;
use std::time::Instant;
```

Add this import with the other external imports:

```rust
use log::debug;
```

Inside `VoxCPMEngine::load`, insert this as the first statement:

```rust
        let load_started = Instant::now();
        debug!(
            "VoxCPMEngine::load start model_dir={} device={:?}",
            model_dir.display(),
            device
        );
```

After `let tokenizer = ...?;`, insert:

```rust
        debug!(
            "VoxCPMEngine::load manifest/tokenizer ready in {:.2}s",
            load_started.elapsed().as_secs_f64()
        );
```

Before each component load, insert a start log. After each component is constructed, insert a done log. Use these exact messages:

```rust
        debug!("VoxCPMEngine::load base_lm start");
        debug!("VoxCPMEngine::load base_lm done");
        debug!("VoxCPMEngine::load residual_lm start");
        debug!("VoxCPMEngine::load residual_lm done");
        debug!("VoxCPMEngine::load feat_encoder start");
        debug!("VoxCPMEngine::load feat_encoder done");
        debug!("VoxCPMEngine::load feat_decoder start");
        debug!("VoxCPMEngine::load feat_decoder done");
        debug!("VoxCPMEngine::load audio_vae start");
        debug!("VoxCPMEngine::load audio_vae done");
        debug!("VoxCPMEngine::load projections start");
        debug!("VoxCPMEngine::load projections done");
```

Before `Ok(Self {`, insert:

```rust
        debug!(
            "VoxCPMEngine::load complete arch={} sample_rate={} patch_size={} elapsed={:.2}s",
            config.architecture,
            config.sample_rate,
            config.patch_size,
            load_started.elapsed().as_secs_f64()
        );
```

- [ ] **Step 2: Add Tauri command logs**

In `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`, add:

```rust
use std::time::Instant;
```

Add this import with the other external imports:

```rust
use log::{debug, error, warn};
```

Replace the first lines of `load_model` through device selection with:

```rust
    let started = Instant::now();
    debug!("load_model requested model_dir={model_dir} backend={backend}");
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => {
            let message = engine_busy_message();
            warn!("load_model rejected: {message}");
            return Err(message);
        }
    };
    let model_path = PathBuf::from(&model_dir);
    let (device, actual_backend, warning) = select_device(&backend);
    debug!(
        "load_model selected backend requested={} actual={} warning={:?}",
        backend,
        actual_backend,
        warning
    );
    let engine_slot = Arc::clone(&state.engine);
```

Replace the `let engine = tokio::task::spawn_blocking...` block in `load_model` with:

```rust
    let engine = match tokio::task::spawn_blocking(move || VoxCPMEngine::load(&model_path, device)).await {
        Ok(Ok(engine)) => engine,
        Ok(Err(err)) => {
            let message = format!("model load failed: {err}");
            error!("load_model failed after {:.2}s: {message}", started.elapsed().as_secs_f64());
            return Err(message);
        }
        Err(err) => {
            let message = format!("model load task failed: {err}");
            error!("load_model task failed after {:.2}s: {message}", started.elapsed().as_secs_f64());
            return Err(message);
        }
    };
```

After storing the engine and before `Ok(info)`, insert:

```rust
    debug!(
        "load_model complete arch={} sample_rate={} backend={} elapsed={:.2}s",
        info.architecture,
        info.sample_rate,
        info.backend,
        started.elapsed().as_secs_f64()
    );
```

In `apply_lora`, replace the busy guard line with:

```rust
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => {
            let message = engine_busy_message();
            warn!("apply_lora rejected: {message}");
            return Err(message);
        }
    };
    debug!("apply_lora requested lora_dir={:?}", args.lora_dir);
```

In `synthesize`, add this line after `let index = args.index;`:

```rust
    debug!("synthesize requested index={index}");
```

In the `Err(message)` arm of `try_begin_synthesis()` in `synthesize`, add:

```rust
            warn!("synthesize rejected index={index}: {message}");
```

In `emit_synthesis_error`, insert this as the first statement:

```rust
    error!("synthesis error index={index}: {message}");
```

- [ ] **Step 3: Add frontend debug-console helper**

In `voxui/crates/voxui-desktop/src/app.rs`, add this helper after `non_empty_option`:

```rust
#[cfg(debug_assertions)]
fn debug_log(message: &str) {
    web_sys::console::debug_1(&format!("[VoxUI] {message}").into());
}

#[cfg(not(debug_assertions))]
fn debug_log(_message: &str) {}
```

Add these debug calls in the startup `spawn_local` block:

```rust
            debug_log("startup: get_config start");
```

Before `if let Ok(config) = ...`, and inside the success block after config values are applied:

```rust
                debug_log(&format!("startup: get_config ok model_dir={} backend={}", config.model_dir, config.backend));
```

Before listing models:

```rust
            debug_log("startup: list_models start");
```

After `set_models.set(model_list);`:

```rust
                debug_log(&format!("startup: list_models ok selected={} count={}", selected.name, models.get_untracked().len()));
```

Before listing audio devices:

```rust
            debug_log("startup: list_audio_devices start");
```

After `set_audio_device.set(selected_device);`:

```rust
                debug_log(&format!("startup: list_audio_devices ok host={} device_count={}", audio_host.get_untracked(), devices.get_untracked().len()));
```

Before listing LoRAs:

```rust
            debug_log("startup: list_lora_dirs start");
```

After `set_loras.set(lora_list);`:

```rust
                    debug_log(&format!("startup: list_lora_dirs ok count={}", loras.get_untracked().len()));
```

Before the startup `load_model` invoke:

```rust
            debug_log(&format!("startup: load_model start model_dir={} backend={}", md, be));
```

Inside the startup load success arm:

```rust
                    debug_log(&format!("startup: load_model ok arch={} sample_rate={} backend={}", info.architecture, info.sample_rate, info.backend));
```

Inside the startup load error arm:

```rust
                    debug_log(&format!("startup: load_model error {e}"));
```

In `on_model_selected`, before invoking `load_model`:

```rust
            debug_log(&format!("model_select: load_model start model_dir={} backend={}", path, be));
```

Inside the `on_model_selected` load success and error arms:

```rust
                    debug_log(&format!("model_select: load_model ok arch={} sample_rate={} backend={}", info.architecture, info.sample_rate, info.backend));
```

```rust
                    debug_log(&format!("model_select: load_model error {e}"));
```

In `on_apply_settings`, before the reload invoke:

```rust
                debug_log(&format!("settings: load_model start model_dir={} backend={}", requested_model_dir, requested_backend));
```

Inside the settings reload success and error arms:

```rust
                        debug_log(&format!("settings: load_model ok backend={}", final_backend));
```

```rust
                        debug_log(&format!("settings: load_model error {e}"));
```

- [ ] **Step 4: Run formatting and desktop/inference compile checks**

Run:

```powershell
cd voxui
cargo fmt
cargo check -p voxui-inference
cargo test --manifest-path crates/voxui-desktop/src-tauri/Cargo.toml
```

Expected: PASS.

- [ ] **Step 5: Commit debug logging changes**

Run:

```powershell
git add voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src/app.rs
git commit -m "chore(desktop): add debug logs for model loading"
```

---

### Task 5: Replace README with Concise Commands

**Files:**
- Modify: `README.txt`

- [ ] **Step 1: Replace README content**

Replace `README.txt` with:

```text
VoxUI commands

CUDA/MSVC environment:
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"; $env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"; $env:CUDA_COMPUTE_CAP = "89"; $env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"

Build inference:
cd voxui; cargo build -p voxui-inference --release
cd voxui; cargo build -p voxui-inference --features cuda --release

Build desktop:
cd voxui\crates\voxui-desktop; trunk build --release
cd voxui\crates\voxui-desktop\src-tauri; cargo build --features cuda --release

Run desktop with debug logs:
cd voxui\crates\voxui-desktop\src-tauri; $env:RUST_LOG = "voxui_desktop=debug,voxui_inference=debug"; cargo tauri dev --features cuda

Export fp16 bundles:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-profile fp16
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-profile fp16
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16

Export q4-lm bundles:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-q4-lm --variant 0.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-q4-lm --variant 1.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-q4-lm --variant 2.0 --quant-profile q4-lm

Verify GGUF bundles:
python exporter/verify_gguf.py models/voxcpm05-q4-lm
python exporter/verify_gguf.py models/voxcpm15-q4-lm
python exporter/verify_gguf.py models/voxcpm2-q4-lm

Run tests:
python -m unittest exporter.tests.test_export_manifest -v
cd voxui; cargo test -p voxui-gguf
cd voxui; cargo test -p voxui-inference --test manifest_loader
cd voxui; cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
cd voxui; cargo test -p voxui-inference --features cuda --test inference_suite full_matrix -- --nocapture --test-threads=1
```

- [ ] **Step 2: Verify README is concise and command-oriented**

Run:

```powershell
Get-Content README.txt
```

Expected: the output contains the command sections shown above and no obsolete one-line-only README.

- [ ] **Step 3: Commit README changes**

Run:

```powershell
git add README.txt
git commit -m "docs: add concise build and test commands"
```

---

### Task 6: Generate Q4-LM Bundles and Full Matrix WAVs

**Files and artifacts:**
- Create/modify: `models/voxcpm05-q4-lm/*`
- Create/modify: `models/voxcpm15-q4-lm/*`
- Create/modify: `models/voxcpm2-q4-lm/*`
- Create/modify: `test_output/*.wav`

- [ ] **Step 1: Export q4-lm bundles**

Run from repo root:

```powershell
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-q4-lm --variant 0.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-q4-lm --variant 1.5 --quant-profile q4-lm
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-q4-lm --variant 2.0 --quant-profile q4-lm
```

Expected: each command prints `Writing ... quant=q4` for `base_lm.gguf`, `residual_lm.gguf`, and `projections.gguf`; each command prints `quant=fp16` or `quant=f32` for the audio-sensitive components.

- [ ] **Step 2: Verify q4-lm manifests**

Run:

```powershell
Get-Content -Raw models/voxcpm05-q4-lm/manifest.json | ConvertFrom-Json | Select-Object -ExpandProperty quantization
Get-Content -Raw models/voxcpm15-q4-lm/manifest.json | ConvertFrom-Json | Select-Object -ExpandProperty quantization
Get-Content -Raw models/voxcpm2-q4-lm/manifest.json | ConvertFrom-Json | Select-Object -ExpandProperty quantization
```

Expected:

```text
base_lm.gguf=q4
residual_lm.gguf=q4
projections.gguf=q4
feat_encoder.gguf=fp16
feat_decoder.gguf=fp16
audio_vae.gguf=fp16 for VoxCPM 0.5/1.5, f32 for VoxCPM2
```

- [ ] **Step 3: Verify GGUF files**

Run:

```powershell
python exporter/verify_gguf.py models/voxcpm05-q4-lm
python exporter/verify_gguf.py models/voxcpm15-q4-lm
python exporter/verify_gguf.py models/voxcpm2-q4-lm
```

Expected: each directory verifies without errors and lists Q4_0 tensor data for q4 components.

- [ ] **Step 4: Regenerate full matrix WAV outputs**

Run:

```powershell
Remove-Item -LiteralPath test_output -Recurse -Force -ErrorAction SilentlyContinue
cd voxui
cargo test -p voxui-inference --features cuda --test inference_suite full_matrix -- --nocapture --test-threads=1
```

Expected: PASS. The log includes fp16 and q4-lm model directories. `test_output/*.wav` is recreated with Chinese and English WAVs for each discovered model/device case.

- [ ] **Step 5: Inspect generated WAV list**

Run from repo root:

```powershell
Get-ChildItem test_output -Filter *.wav | Sort-Object Name | Select-Object Name, Length
```

Expected: every listed WAV has non-zero length. Names include `zh` and `en` for fp16 and q4-lm model labels.

- [ ] **Step 6: Commit generated model metadata only if binary artifacts are already tracked**

Run:

```powershell
git status --short models test_output
git ls-files models test_output
```

Expected: decide from actual tracked state. If `models` and `test_output` are untracked binary artifacts, leave them uncommitted and report their local paths. If existing generated artifacts are tracked, stage only the intended q4-lm directories and regenerated WAVs:

```powershell
git add models/voxcpm05-q4-lm models/voxcpm15-q4-lm models/voxcpm2-q4-lm test_output
git commit -m "chore(models): add q4 lm inference matrix artifacts"
```

---

### Task 7: Final Verification

**Files:**
- Verify all modified files and local artifacts.

- [ ] **Step 1: Run Python tests**

Run:

```powershell
python -m unittest exporter.tests.test_export_manifest -v
python exporter/quantize.py
```

Expected: PASS and quantizer self-test prints `All self-tests passed.`

- [ ] **Step 2: Run Rust unit and integration compile checks**

Run:

```powershell
cd voxui
cargo fmt --check
cargo test -p voxui-gguf
cargo test -p voxui-inference --test manifest_loader
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
cargo test --manifest-path crates/voxui-desktop/src-tauri/Cargo.toml
```

Expected: PASS.

- [ ] **Step 3: Run desktop debug smoke command**

Run:

```powershell
cd voxui\crates\voxui-desktop\src-tauri
$env:RUST_LOG = "voxui_desktop=debug,voxui_inference=debug"
cargo tauri dev --features cuda
```

Expected: the debug console/log output includes `load_model requested`, `VoxCPMEngine::load start`, component start/done messages, and either `load_model complete` or a concrete load error.

- [ ] **Step 4: Review final git state**

Run:

```powershell
git status --short
git log --oneline -n 8
```

Expected: only intentional generated artifacts remain untracked if they are local-only. Source and README changes are committed in the task commits above.
