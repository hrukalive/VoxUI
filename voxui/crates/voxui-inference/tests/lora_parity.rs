use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use voxui_inference::{LoraAdapter, SynthesisRequest, VoxCPMEngine};

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

#[test]
fn lora_linear_delta_matches_formula() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &device).unwrap();
    let base = Tensor::zeros((1, 3), candle_core::DType::F32, &device).unwrap();
    let a = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1., 0., 0.], (2, 4), &device)
        .unwrap();
    let b = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &device).unwrap();

    let out = LoraAdapter::apply_raw(&base, &x, &a, &b, 4.0, 2).unwrap();

    assert_eq!(out.dims(), &[1, 3]);
    assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![10.0, 6.0, 16.0]]);
}

#[test]
fn lora_adapter_changes_generation_without_breaking_audio() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let Ok(entries) = std::fs::read_dir(&model_dir) else {
        eprintln!("skip: model directory not found at {}", model_dir.display());
        return;
    };
    let lora_file = entries
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

    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    engine.load_lora(&lora_file).unwrap();
    let samples = engine
        .generate(
            SynthesisRequest {
                text: "Hello, welcome to the stream!".to_string(),
                inference_timesteps: 4,
                min_len: 1,
                max_len: 3,
                retry_badcase: false,
                ..SynthesisRequest::default()
            },
            |_, _| {},
        )
        .unwrap();

    assert!(!samples.is_empty());
    assert!(samples.iter().all(|v| v.is_finite()));
}
