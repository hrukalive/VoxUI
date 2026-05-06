use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::{DiT, GgufModelLoader, ModelConfig, ModelVariant};

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
fn first_dit_patch_matches_python_trace_with_fixed_noise() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
    let dit = DiT::load_from_config(&loader, &config).unwrap();

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let cond = trace.tensor("first_dit_cond").unwrap();
    let mu = trace.tensor("first_dit_mu").unwrap();
    let noise = trace.tensor("first_dit_noise").unwrap();
    let expected = trace.tensor("first_dit_patch").unwrap();
    let actual = dit
        .solve_euler_with_noise(&mu, &cond, &noise, 4, 2.0)
        .unwrap();
    voxui_inference::trace::assert_close(&actual, &expected, 8e-3).unwrap();
}
