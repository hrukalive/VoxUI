use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{BaseLM, BaseLMConfig, BundleManifest, GgufModelLoader};

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
fn base_lm_prefill_matches_python_trace() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let manifest = BundleManifest::load(&model_dir).unwrap();
    let loader = GgufModelLoader::new(
        &manifest.component_path(&model_dir, "base_lm").unwrap(),
        Device::Cpu,
    )
    .unwrap();
    let config = BaseLMConfig::from_manifest(&manifest, "base_lm").unwrap();
    let mut lm = BaseLM::load(&loader, config, &Device::Cpu).unwrap();

    let trace = voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let token_ids = trace.u32_list("token_ids").unwrap();
    let expected = trace.tensor("base_lm_prefill_hidden").unwrap();
    let actual = lm.forward(&token_ids).unwrap();
    voxui_inference::trace::assert_close(&actual, &expected, 8e-3).unwrap();
}

#[test]
fn rope_rotate_half_matches_python_layout() {
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let rotated = voxui_inference::base_lm::rotate_half_for_test(&input);
    assert_eq!(rotated, vec![-3.0, -4.0, 1.0, 2.0]);
}
