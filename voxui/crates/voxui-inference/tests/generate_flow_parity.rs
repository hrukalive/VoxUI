use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{
    fsq::FSQLayer, GgufModelLoader, ModelConfig, ModelVariant, SynthesisRequest, VoxCPMEngine,
};

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
fn voxcpm2_streaming_yields_finite_ordered_chunks() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    let sample_rate = engine.sample_rate();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 1,
        min_len: 0,
        max_len: 1,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };

    let mut chunks = Vec::new();
    engine
        .generate_streaming(request, |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();

    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|chunk| chunk.sample_rate == sample_rate));
    assert!(chunks.iter().all(|chunk| !chunk.samples.is_empty()));
    assert!(chunks
        .iter()
        .flat_map(|chunk| chunk.samples.iter())
        .all(|v| v.is_finite()));
    assert!(chunks
        .iter()
        .enumerate()
        .all(|(idx, chunk)| chunk.patch_index == idx));
    assert!(chunks.last().unwrap().is_final);
}

#[test]
fn voxcpm2_long_zh_stop_loop_matches_python_trace() {
    let root = repo_root();
    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_long_zh_stop_loop"))
            .unwrap();
    let model_dir = root.join("models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    let request = SynthesisRequest {
        text: "我说什么来着，我不知道你是什么脾气啊，我肯定要邦邦敲一下。".to_string(),
        inference_timesteps: 10,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };

    let debug = engine
        .generate_debug_stop_loop_with_patches(
            request,
            trace.tensor("generated_audio_feat").unwrap(),
        )
        .unwrap();

    let expected_stop_logits = trace.tensor("stop_logits_sequence").unwrap();
    assert_eq!(
        debug.generated_patch_count,
        expected_stop_logits.dim(0).unwrap(),
        "Rust generated patch count should match Python stop-loop length"
    );
    voxui_inference::trace::assert_close(&debug.stop_logits_sequence, &expected_stop_logits, 5e-2)
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
fn voxcpm05_fsq_prefill_matches_python_trace() {
    assert_fsq_prefill_matches_python_trace(
        "voxcpm05-fp16",
        "voxcpm05_zero_shot",
        ModelVariant::VoxCpm05,
    );
}

#[test]
fn voxcpm15_fsq_prefill_matches_python_trace() {
    assert_fsq_prefill_matches_python_trace(
        "voxcpm15-fp16",
        "voxcpm15_zero_shot",
        ModelVariant::VoxCpm15,
    );
}

#[test]
fn voxcpm2_fsq_prefill_matches_python_trace() {
    assert_fsq_prefill_matches_python_trace(
        "voxcpm2-fp16",
        "voxcpm2_zero_shot",
        ModelVariant::VoxCpm2,
    );
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

fn assert_fsq_prefill_matches_python_trace(
    model_name: &str,
    trace_name: &str,
    variant: ModelVariant,
) {
    let root = repo_root();
    let model_dir = root.join("models").join(model_name);
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = ModelConfig::load(&model_dir, variant).unwrap();
    let fsq = FSQLayer::load(
        &loader,
        config.scalar_quantization_latent_dim,
        config.scalar_quantization_scale as f64,
    )
    .unwrap();
    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens").join(trace_name)).unwrap();

    let actual = fsq
        .forward(&trace.tensor("base_lm_prefill_hidden").unwrap())
        .unwrap();

    voxui_inference::trace::assert_close(&actual, &trace.tensor("first_fsq_hidden").unwrap(), 2e-3)
        .unwrap();
}
