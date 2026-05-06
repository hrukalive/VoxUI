use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{GgufModelLoader, LocalEncoder, ModelConfig, ModelVariant};

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
fn local_encoder_accepts_b_t_p_d_and_matches_trace() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
    let mut encoder = LocalEncoder::load_from_config(&loader, &config).unwrap();

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let audio_feat = trace.tensor("prefill_audio_feat_b_t_p_d").unwrap();
    let expected = trace.tensor("local_encoder_output").unwrap();
    let actual = encoder.encode_patches(&audio_feat).unwrap();
    assert_eq!(actual.dims(), expected.dims());
    voxui_inference::trace::assert_close(&actual, &expected, 8e-3).unwrap();
}
