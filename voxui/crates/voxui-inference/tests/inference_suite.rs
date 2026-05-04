//! Integration test suite for VoxCPM inference engine.
//!
//! Tests all model variants under `models/` with and without LoRA,
//! synthesizing both Chinese and English text, on both CPU and CUDA backends.
//! Saves output WAV files to `test_output/` for manual inspection.
//!
//! Run from the workspace directory (D:\Sandbox_Share\VoxUI\voxui):
//!   cargo test -p voxui-inference --test inference_suite --release --features cuda -- --nocapture
//!
//! CPU-only:
//!   cargo test -p voxui-inference --test inference_suite --release -- --nocapture
//!
//! Single test:
//!   cargo test -p voxui-inference --test inference_suite --release --features cuda -- voxcpm2_q4_cuda --nocapture

use std::cell::Cell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use voxui_inference::VoxCPMEngine;

/// Output directory for WAV files.
const OUTPUT_DIR: &str = "D:\\Sandbox_Share\\VoxUI\\test_output";

/// Root directory containing model folders.
/// Tests must be run with CWD = project root (D:\Sandbox_Share\VoxUI).
const MODELS_DIR: &str = "D:\\Sandbox_Share\\VoxUI\\models";

/// Diffusion steps for testing (fewer = faster, 5 is minimum for coherence).
const TEST_DIT_STEPS: usize = 10;

/// Chinese test sentence.
const TEXT_ZH: &str = "你好，欢迎来到直播间！";

/// English test sentence.
const TEXT_EN: &str = "Hello, welcome to the stream!";

// ===========================================================================
// Helpers
// ===========================================================================

/// Write a PCM f32 mono WAV file.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create WAV file {}: {e}", path.display()));

    // RIFF header
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    // fmt chunk
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap();  // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap();  // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();  // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    // data chunk
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();

    // Convert f32 [-1, 1] to i16
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes()).unwrap();
    }
}

fn get_cpu_device() -> Device {
    Device::Cpu
}

fn get_cuda_device() -> Option<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(dev) => Some(dev),
            Err(e) => {
                eprintln!("  [WARN] CUDA device init failed: {e}");
                None
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        _ => "Unknown",
    }
}

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(MODELS_DIR).join(name)
}

fn find_lora_dirs(model: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("lora_") && entry.path().is_dir() {
                if entry.path().join("lora_base_lm.gguf").exists() {
                    dirs.push(entry.path());
                }
            }
        }
    }
    dirs.sort();
    dirs
}

/// Run one synthesis call, assert output validity, save WAV, print results.
fn run_synthesis(engine: &mut VoxCPMEngine, text: &str, label: &str) -> (usize, f64) {
    let t0 = Instant::now();
    let last_step = Cell::new(0usize);
    let last_total = Cell::new(0usize);
    let samples = engine
        .synthesize(text, TEST_DIT_STEPS, |step, total| {
            last_step.set(step);
            last_total.set(total);
        })
        .unwrap_or_else(|e| panic!("  [FAIL] synthesize({label}): {e}"));

    let dur = t0.elapsed().as_secs_f64();
    let n = samples.len();
    let sr = engine.sample_rate();
    let audio_dur = n as f64 / sr as f64;

    println!(
        "    [OK] {label}: {n} samples ({audio_dur:.2}s @ {sr}Hz), \
         steps={}/{}, wall={dur:.1}s",
        last_step.get(),
        last_total.get(),
    );

    // Sanity checks
    assert!(n > 0, "synthesize returned empty audio for {label}");
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "synthesize produced NaN/Inf for {label}"
    );
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 1e-6,
        "synthesize produced near-silence (max_abs={max_abs}) for {label}"
    );

    // Save WAV file
    let wav_name = label.replace('/', "_").replace(' ', "_");
    let wav_path = PathBuf::from(OUTPUT_DIR).join(format!("{wav_name}.wav"));
    write_wav(&wav_path, &samples, sr);
    println!("    [WAV] {}", wav_path.display());

    (n, dur)
}

/// Full test for a model variant on a specific device.
fn test_model_on_device(model_name: &str, device: Device) {
    let dir = model_dir(model_name);
    if !dir.join("base_lm.gguf").exists() {
        eprintln!("  [SKIP] {model_name}: base_lm.gguf not found at {}", dir.display());
        return;
    }

    let dev_name = device_name(&device);
    println!("\n{}", "=".repeat(70));
    println!("  Model: {model_name}  |  Device: {dev_name}");
    println!("{}", "=".repeat(70));

    // Load engine
    let t0 = Instant::now();
    let mut engine = VoxCPMEngine::load(&dir, &dir, device)
        .unwrap_or_else(|e| panic!("[FAIL] load {model_name} on {dev_name}: {e}"));
    let load_dur = t0.elapsed().as_secs_f64();

    println!(
        "  Loaded in {load_dur:.1}s — arch={}, sr={}, patch={}",
        engine.architecture(),
        engine.sample_rate(),
        engine.patch_size(),
    );

    // --- Without LoRA ---
    println!("\n  --- Without LoRA ({dev_name}) ---");
    run_synthesis(&mut engine, TEXT_ZH, &format!("{model_name}/{dev_name}/zh/no-lora"));
    run_synthesis(&mut engine, TEXT_EN, &format!("{model_name}/{dev_name}/en/no-lora"));

    // --- With each LoRA ---
    let lora_dirs = find_lora_dirs(&dir);
    for lora_dir in &lora_dirs {
        let lora_name = lora_dir.file_name().unwrap().to_string_lossy();
        println!("\n  --- LoRA: {lora_name} ({dev_name}) ---");

        engine
            .load_lora(lora_dir)
            .unwrap_or_else(|e| panic!("  [FAIL] load_lora({lora_name}): {e}"));

        run_synthesis(&mut engine, TEXT_ZH, &format!("{model_name}/{dev_name}/zh/{lora_name}"));
        run_synthesis(&mut engine, TEXT_EN, &format!("{model_name}/{dev_name}/en/{lora_name}"));

        engine.unload_lora();
    }

    if lora_dirs.is_empty() {
        println!("  [INFO] No lora_* directories found");
    }

    println!("\n  ✓ {model_name} on {dev_name} PASSED");
}

// ===========================================================================
// CPU Tests — one per model variant
// ===========================================================================

#[test]
fn voxcpm05b_fp16_cpu() {
    test_model_on_device("voxcpm05b-fp16", get_cpu_device());
}

#[test]
fn voxcpm15_fp16_cpu() {
    test_model_on_device("voxcpm15-fp16", get_cpu_device());
}

#[test]
fn voxcpm2_fp16_cpu() {
    test_model_on_device("voxcpm2-fp16", get_cpu_device());
}

#[test]
fn voxcpm2_q4_cpu() {
    test_model_on_device("voxcpm2-q4", get_cpu_device());
}

// ===========================================================================
// CUDA Tests — one per model variant (skipped if CUDA unavailable)
// ===========================================================================

#[test]
fn voxcpm05b_fp16_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm05b-fp16", device);
}

#[test]
fn voxcpm15_fp16_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm15-fp16", device);
}

#[test]
fn voxcpm2_fp16_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm2-fp16", device);
}

#[test]
fn voxcpm2_q4_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm2-q4", device);
}

// ===========================================================================
// Full matrix test — all models × all backends
// ===========================================================================

#[test]
fn full_matrix() {
    let models_path = PathBuf::from(MODELS_DIR);
    println!("\n[INFO] Scanning for models in {}", models_path.display());
    if !models_path.exists() {
        eprintln!("[SKIP] models/ directory not found — run from project root");
        return;
    }

    let mut model_dirs: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&models_path) {
        for entry in entries.flatten() {
            if entry.path().join("base_lm.gguf").exists() {
                model_dirs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    model_dirs.sort();

    let mut devices: Vec<(&str, Device)> = vec![("CPU", get_cpu_device())];
    if let Some(cuda) = get_cuda_device() {
        devices.push(("CUDA", cuda));
    }

    println!("\n{}", "=".repeat(70));
    println!(
        "  FULL MATRIX: {} models × {} backends",
        model_dirs.len(),
        devices.len()
    );
    println!("  Models: {:?}", model_dirs);
    println!("  Backends: {:?}", devices.iter().map(|(n, _)| *n).collect::<Vec<_>>());
    println!("{}", "=".repeat(70));

    let mut pass_count = 0;
    let mut fail_count = 0;

    for model_name in &model_dirs {
        for (dev_name, device) in &devices {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                test_model_on_device(model_name, device.clone());
            }));
            match result {
                Ok(()) => {
                    pass_count += 1;
                }
                Err(_) => {
                    eprintln!("  ✗ {model_name} on {dev_name} FAILED");
                    fail_count += 1;
                }
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!(
        "  RESULTS: {pass_count} passed, {fail_count} failed (total {})",
        pass_count + fail_count
    );
    println!("{}", "=".repeat(70));

    assert_eq!(fail_count, 0, "{fail_count} test(s) failed");
}
