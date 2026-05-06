use std::path::{Path, PathBuf};

use candle_core::Device;
use serde::Deserialize;
use voxui_inference::dit::DiTConfig;
use voxui_inference::{DiT, GgufModelLoader};

#[derive(Deserialize)]
struct TestComponents {
    feat_decoder: String,
}

#[derive(Deserialize)]
struct TestManifest {
    components: TestComponents,
    feat_dim: usize,
    #[serde(default)]
    lm_config: serde_json::Value,
    #[serde(default)]
    dit_config: serde_json::Value,
}

impl TestManifest {
    fn load(model_dir: &Path) -> Self {
        let text = std::fs::read_to_string(model_dir.join("manifest.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn feat_decoder_path(&self, model_dir: &Path) -> PathBuf {
        model_dir.join(&self.components.feat_decoder)
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
fn first_dit_patch_matches_python_trace_with_fixed_noise() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let manifest = TestManifest::load(&model_dir);
    let loader =
        GgufModelLoader::new(&manifest.feat_decoder_path(&model_dir), Device::Cpu).unwrap();
    let dit = DiT::load(&loader, dit_config(&manifest), &Device::Cpu).unwrap();

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

fn dit_config(manifest: &TestManifest) -> DiTConfig {
    let dit = &manifest.dit_config;
    let lm = &manifest.lm_config;
    let hidden_dim = get_usize(dit, &["hidden_dim", "hidden_size"], 1024);
    let num_heads = get_usize(dit, &["num_heads", "num_attention_heads"], 16);
    let head_dim = dit
        .get("kv_channels")
        .or_else(|| lm.get("kv_channels"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_dim / num_heads);
    let num_kv_heads = lm
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(num_heads);
    let rope_scaling = lm.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
    let cfm = dit.get("cfm_config").unwrap_or(&serde_json::Value::Null);

    DiTConfig {
        prefix: "feat_decoder.estimator".to_string(),
        hidden_dim,
        num_layers: get_usize(dit, &["num_layers", "num_hidden_layers"], 12),
        num_heads,
        num_kv_heads,
        head_dim,
        ffn_dim: get_usize(dit, &["ffn_dim", "intermediate_size"], 4096),
        rms_norm_eps: get_f64(lm, "rms_norm_eps", 1e-5),
        scale_depth: get_f64(lm, "scale_depth", 1.0),
        use_mup: lm.get("use_mup").and_then(|v| v.as_bool()).unwrap_or(false),
        rope_theta: get_f64(lm, "rope_theta", 10000.0),
        original_max_position_embeddings: rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        rope_short_factors: read_f32_array(rope_scaling, "short_factor", head_dim / 2),
        rope_long_factors: read_f32_array(rope_scaling, "long_factor", head_dim / 2),
        cfg_value: get_f64(cfm, "inference_cfg_rate", 1.0),
        n_steps: 10,
        sway_coef: get_f64(dit, "sway_sampling_coef", 1.0),
        latent_dim: manifest.feat_dim,
    }
}

fn get_usize(cfg: &serde_json::Value, keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|key| cfg.get(*key).and_then(|v| v.as_u64()).map(|v| v as usize))
        .unwrap_or(default)
}

fn get_f64(cfg: &serde_json::Value, key: &str, default: f64) -> f64 {
    cfg.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn read_f32_array(value: &serde_json::Value, key: &str, len: usize) -> Vec<f32> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_f64().unwrap_or(1.0) as f32)
                .collect()
        })
        .filter(|arr: &Vec<f32>| arr.len() == len)
        .unwrap_or_else(|| vec![1.0; len])
}
