use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

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
fn voxcpm2_first_patch_flow_matches_python_trace() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };

    let debug = engine
        .generate_debug_first_patch_with_noise(request, trace.tensor("first_dit_noise").unwrap())
        .unwrap();

    voxui_inference::trace::assert_close(
        &debug.first_patch,
        &trace.tensor("first_dit_patch").unwrap(),
        8e-3,
    )
    .unwrap();
    voxui_inference::trace::assert_close(
        &debug.stop_logits,
        &trace.tensor("stop_logits").unwrap(),
        2e-3,
    )
    .unwrap();
}

#[test]
fn voxcpm05_first_patch_flow_matches_python_trace() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm05-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm05_zero_shot")).unwrap();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };

    let debug = engine
        .generate_debug_first_patch_with_noise(request, trace.tensor("first_dit_noise").unwrap())
        .unwrap();

    voxui_inference::trace::assert_close(
        &debug.first_patch,
        &trace.tensor("first_dit_patch").unwrap(),
        8e-3,
    )
    .unwrap();
    voxui_inference::trace::assert_close(
        &debug.stop_logits,
        &trace.tensor("stop_logits").unwrap(),
        2e-3,
    )
    .unwrap();
}

#[test]
fn voxcpm2_reference_request_uses_reference_audio_without_prompt_text() {
    let root = repo_root();
    let wav = std::fs::read_dir(root.join("for_test_wav"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|v| v.to_str()) == Some("wav"))
        .expect("at least one test wav");
    let model_dir = root.join("models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
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
