use std::path::{Path, PathBuf};

use candle_core::Device;
use serde::Deserialize;
use voxui_inference::audio_io::load_wav_mono_resampled;
use voxui_inference::{AudioVAE, AudioVaeManifest, GgufModelLoader};

#[derive(Deserialize)]
struct TestComponents {
    audio_vae: String,
}

#[derive(Deserialize)]
struct TestManifest {
    audio_vae: AudioVaeManifest,
    components: TestComponents,
}

impl TestManifest {
    fn load(model_dir: &Path) -> Self {
        let text = std::fs::read_to_string(model_dir.join("manifest.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn audio_vae_path(&self, model_dir: &Path) -> PathBuf {
        model_dir.join(&self.components.audio_vae)
    }
}

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
    let model_dir = root.join("models/voxcpm2-fp16");
    let manifest = TestManifest::load(&model_dir);
    let loader = GgufModelLoader::new(&manifest.audio_vae_path(&model_dir), Device::Cpu).unwrap();
    AudioVAE::load_from_manifest(&loader, &manifest.audio_vae).unwrap()
}
