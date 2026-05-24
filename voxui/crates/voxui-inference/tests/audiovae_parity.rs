use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::audio_io::load_wav_mono_resampled;
use voxui_inference::{AudioVAE, GgufModelLoader, ModelConfig, ModelVariant};

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
fn wav_loader_returns_mono_f32_at_requested_rate() {
    let wav = std::fs::read_dir(repo_root().join("for_test_wav"))
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
    let root = repo_root();
    let vae = load_voxcpm2_vae(&root);

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let latent = trace.tensor("generated_latent").unwrap();
    let expected = trace.tensor("decoded_wav_head").unwrap();
    let decoded = vae.decode(&latent).unwrap();
    voxui_inference::trace::assert_close_prefix(&decoded, &expected, 2e-3).unwrap();
}

#[test]
fn audiovae_streaming_decode_matches_full_decode_for_patch_chunks() {
    let root = repo_root();
    let vae = load_voxcpm2_vae(&root);

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let latent = trace.tensor("generated_latent").unwrap();
    let full = vae.decode(&latent).unwrap();

    let mut streaming = vae.streaming_decoder();
    let patch_size = 4;
    let latent_len = latent.dim(2).unwrap();
    let mut chunks = Vec::new();
    for start in (0..latent_len).step_by(patch_size) {
        let len = patch_size.min(latent_len - start);
        chunks.push(
            streaming
                .decode_chunk(&latent.narrow(2, start, len).unwrap())
                .unwrap(),
        );
    }
    let refs = chunks.iter().collect::<Vec<_>>();
    let streamed = candle_core::Tensor::cat(&refs, 2).unwrap();

    voxui_inference::trace::assert_close(&streamed, &full, 2e-3).unwrap();
}

#[test]
fn voxcpm05_audiovae_decode_matches_python_trace_head() {
    let root = repo_root();
    let vae = load_vae(&root, "voxcpm05-fp16", ModelVariant::VoxCpm05);

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm05_zero_shot")).unwrap();
    let latent = trace.tensor("generated_latent").unwrap();
    let expected = trace.tensor("decoded_wav_head").unwrap();
    let decoded = vae.decode(&latent).unwrap();
    voxui_inference::trace::assert_close_prefix(&decoded, &expected, 2e-3).unwrap();
}

#[test]
fn audiovae_encode_matches_python_trace() {
    let root = repo_root();
    let vae = load_voxcpm2_vae(&root);
    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_reference")).unwrap();
    let audio = trace.tensor("audio_vae_encode_input").unwrap();
    let expected = trace.tensor("audio_vae_encode_output").unwrap();
    let encoded = vae.encode(&audio).unwrap();
    voxui_inference::trace::assert_close(&encoded, &expected, 2e-3).unwrap();
}

fn load_voxcpm2_vae(root: &Path) -> AudioVAE {
    load_vae(root, "voxcpm2-fp16", ModelVariant::VoxCpm2)
}

fn load_vae(root: &Path, model_name: &str, variant: ModelVariant) -> AudioVAE {
    let model_dir = root.join("models").join(model_name);
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = ModelConfig::load(&model_dir, variant).unwrap();
    AudioVAE::load_from_config(&loader, &config.audio_vae).unwrap()
}
