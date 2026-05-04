//! End-to-end native VoxCPM inference checks.

use std::cell::Cell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

const TEST_DIT_STEPS: usize = 10;
const TEXT_ZH: &str = "你好，欢迎来到直播间！";
const TEXT_EN: &str = "Hello, welcome to the stream!";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create WAV file {}: {e}", path.display()));
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &sample in samples {
        let sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&sample.to_le_bytes()).unwrap();
    }
}

fn get_cpu_device() -> Device {
    Device::Cpu
}

fn get_cuda_device() -> Option<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).ok()
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
    repo_root().join("models").join(name)
}

fn output_dir() -> PathBuf {
    repo_root().join("test_output")
}

fn first_test_wav() -> Option<PathBuf> {
    std::fs::read_dir(repo_root().join("for_test_wav"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|v| v.to_str()) == Some("wav"))
}

fn find_lora_dirs(model: &Path) -> Vec<PathBuf> {
    let mut dirs = std::fs::read_dir(model)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("lora_manifest.json").exists())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs
}

fn run_synthesis(engine: &mut VoxCPMEngine, request: SynthesisRequest, label: &str) -> (usize, f64) {
    let t0 = Instant::now();
    let last_step = Cell::new(0usize);
    let last_total = Cell::new(0usize);
    let samples = engine
        .generate(request, |step, total| {
            last_step.set(step);
            last_total.set(total);
        })
        .unwrap_or_else(|e| panic!("  [FAIL] generate({label}): {e}"));

    let dur = t0.elapsed().as_secs_f64();
    let n = samples.len();
    let sr = engine.sample_rate();
    let audio_dur = n as f64 / sr as f64;

    println!(
        "    [OK] {label}: {n} samples ({audio_dur:.2}s @ {sr}Hz), steps={}/{}, wall={dur:.1}s",
        last_step.get(),
        last_total.get(),
    );

    assert!(n > 0, "generate returned empty audio for {label}");
    assert!(samples.iter().all(|s| s.is_finite()), "generate produced NaN/Inf for {label}");
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 1e-6, "generate produced near-silence for {label}");

    let wav_name = label.replace('/', "_").replace(' ', "_");
    let wav_path = output_dir().join(format!("{wav_name}.wav"));
    write_wav(&wav_path, &samples, sr);
    println!("    [WAV] {}", wav_path.display());

    (n, dur)
}

fn short_request(text: &str) -> SynthesisRequest {
    SynthesisRequest {
        text: text.to_string(),
        inference_timesteps: TEST_DIT_STEPS,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    }
}

fn test_model_on_device(model_name: &str, device: Device) {
    let dir = model_dir(model_name);
    if !dir.join("manifest.json").exists() {
        eprintln!("  [SKIP] {model_name}: manifest.json not found at {}", dir.display());
        return;
    }

    let dev_name = device_name(&device);
    println!("\n{}", "=".repeat(70));
    println!("  Model: {model_name}  |  Device: {dev_name}");
    println!("{}", "=".repeat(70));

    let t0 = Instant::now();
    let mut engine = VoxCPMEngine::load(&dir, device)
        .unwrap_or_else(|e| panic!("[FAIL] load {model_name} on {dev_name}: {e}"));
    println!(
        "  Loaded in {:.1}s, arch={}, sr={}, patch={}",
        t0.elapsed().as_secs_f64(),
        engine.architecture(),
        engine.sample_rate(),
        engine.patch_size(),
    );

    run_synthesis(&mut engine, short_request(TEXT_ZH), &format!("{model_name}/{dev_name}/zh/no-lora"));
    run_synthesis(&mut engine, short_request(TEXT_EN), &format!("{model_name}/{dev_name}/en/no-lora"));

    if model_name.contains("voxcpm2") {
        if let Some(wav) = first_test_wav() {
            let mut request = short_request(TEXT_EN);
            request.reference_wav_path = Some(wav.clone());
            run_synthesis(&mut engine, request, &format!("{model_name}/{dev_name}/en/reference"));

            let mut request = short_request(TEXT_EN);
            request.reference_wav_path = Some(wav.clone());
            request.prompt_wav_path = Some(wav);
            request.prompt_text = Some("Hello, welcome to the stream!".to_string());
            run_synthesis(&mut engine, request, &format!("{model_name}/{dev_name}/en/ref-cont"));
        }
    }

    for lora_dir in find_lora_dirs(&dir) {
        let lora_name = lora_dir.file_name().unwrap().to_string_lossy();
        engine
            .load_lora(&lora_dir)
            .unwrap_or_else(|e| panic!("  [FAIL] load_lora({lora_name}): {e}"));
        run_synthesis(&mut engine, short_request(TEXT_EN), &format!("{model_name}/{dev_name}/en/{lora_name}"));
        engine.unload_lora();
    }

    println!("\n  {model_name} on {dev_name} passed");
}

#[test]
fn voxcpm05_fp16_cpu() {
    test_model_on_device("voxcpm05-fp16", get_cpu_device());
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
fn voxcpm05_fp16_cuda() {
    let Some(device) = get_cuda_device() else {
        eprintln!("[SKIP] CUDA not available");
        return;
    };
    test_model_on_device("voxcpm05-fp16", device);
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
fn full_matrix() {
    let models_path = repo_root().join("models");
    let mut model_dirs = std::fs::read_dir(&models_path)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .map(|entry| entry.path())
        .filter(|path| path.join("manifest.json").exists())
        .filter_map(|path| path.file_name().map(|name| name.to_string_lossy().to_string()))
        .collect::<Vec<_>>();
    model_dirs.sort();

    let mut devices = vec![("CPU", get_cpu_device())];
    if let Some(cuda) = get_cuda_device() {
        devices.push(("CUDA", cuda));
    }

    for model_name in &model_dirs {
        for (_, device) in &devices {
            test_model_on_device(model_name, device.clone());
        }
    }
}
