use std::path::{Path, PathBuf};

use candle_core::Device;
use voxui_inference::manifest::{ModelConfig, ModelVariant};
use voxui_inference::{BaseLM, BaseLMConfig, GgufModelLoader};

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
    let model = ModelConfig::load(&model_dir, ModelVariant::VoxCpm2).unwrap();
    let loader = GgufModelLoader::from_model_dir(&model_dir, Device::Cpu).unwrap();
    let config = base_lm_config(&model, "base_lm");
    let mut lm = BaseLM::load(&loader, config, &Device::Cpu).unwrap();

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
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

fn base_lm_config(model: &ModelConfig, component: &str) -> BaseLMConfig {
    let cfg = &model.lm_config;
    let hidden_size = get_usize_any(cfg, component, &["hidden_size", "hidden_dim"]);
    let num_heads = get_usize_any(cfg, component, &["num_attention_heads", "num_heads"]);
    let head_dim = cfg
        .get("kv_channels")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_heads);
    let half_dim = head_dim / 2;
    let rope_scaling = cfg.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
    let rope_short_factors = read_f32_array(rope_scaling, "short_factor", half_dim);
    let rope_long_factors = read_f32_array(rope_scaling, "long_factor", half_dim);

    BaseLMConfig {
        hidden_size,
        num_layers: if component == "residual_lm" {
            model.residual_lm_num_layers.unwrap_or(get_usize_any(
                cfg,
                component,
                &["num_hidden_layers", "num_layers"],
            ))
        } else {
            get_usize_any(cfg, component, &["num_hidden_layers", "num_layers"])
        },
        num_heads,
        num_kv_heads: cfg
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_heads),
        head_dim,
        intermediate_size: get_usize_any(cfg, component, &["intermediate_size", "ffn_dim"]),
        rms_norm_eps: get_f64(cfg, "rms_norm_eps", 1e-5),
        rope_theta: get_f64(cfg, "rope_theta", 10000.0),
        rope_factors: rope_short_factors.clone(),
        use_mup: cfg
            .get("use_mup")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        scale_emb: get_f64(cfg, "scale_emb", 1.0),
        scale_depth: get_f64(cfg, "scale_depth", 1.0),
        original_max_position_embeddings: rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        rope_short_factors,
        rope_long_factors,
        vocab_size: cfg.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        max_position: cfg
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        prefix: component.to_string(),
        no_rope: component == "residual_lm" && model.residual_lm_no_rope.unwrap_or(false),
        is_causal: true,
    }
}

fn get_usize_any(cfg: &serde_json::Value, component: &str, keys: &[&str]) -> usize {
    keys.iter()
        .find_map(|key| cfg.get(*key).and_then(|v| v.as_u64()).map(|v| v as usize))
        .unwrap_or_else(|| panic!("missing one of `{keys:?}` in {component} config"))
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
