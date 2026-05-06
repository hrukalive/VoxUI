use std::path::{Path, PathBuf};

use candle_core::Device;
use serde::Deserialize;
use voxui_inference::{BaseLMConfig, GgufModelLoader, LocalEncoder};

#[derive(Deserialize)]
struct TestComponents {
    feat_encoder: String,
}

#[derive(Deserialize)]
struct TestManifest {
    components: TestComponents,
    #[serde(default)]
    lm_config: serde_json::Value,
    #[serde(default)]
    encoder_config: serde_json::Value,
}

impl TestManifest {
    fn load(model_dir: &Path) -> Self {
        let text = std::fs::read_to_string(model_dir.join("manifest.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn feat_encoder_path(&self, model_dir: &Path) -> PathBuf {
        model_dir.join(&self.components.feat_encoder)
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
fn local_encoder_accepts_b_t_p_d_and_matches_trace() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let manifest = TestManifest::load(&model_dir);
    let loader =
        GgufModelLoader::new(&manifest.feat_encoder_path(&model_dir), Device::Cpu).unwrap();
    let config = encoder_config(&manifest);
    let mut encoder = LocalEncoder::load(&loader, config, &Device::Cpu).unwrap();

    let trace =
        voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm2_zero_shot")).unwrap();
    let audio_feat = trace.tensor("prefill_audio_feat_b_t_p_d").unwrap();
    let expected = trace.tensor("local_encoder_output").unwrap();
    let actual = encoder.encode_patches(&audio_feat).unwrap();
    assert_eq!(actual.dims(), expected.dims());
    voxui_inference::trace::assert_close(&actual, &expected, 8e-3).unwrap();
}

fn encoder_config(manifest: &TestManifest) -> BaseLMConfig {
    let cfg = &manifest.encoder_config;
    let fallback_cfg = &manifest.lm_config;
    let hidden_size = get_usize_any(
        cfg,
        fallback_cfg,
        "feat_encoder",
        &["hidden_size", "hidden_dim"],
    );
    let num_heads = get_usize_any(
        cfg,
        fallback_cfg,
        "feat_encoder",
        &["num_attention_heads", "num_heads"],
    );
    let head_dim = cfg
        .get("kv_channels")
        .or_else(|| fallback_cfg.get("kv_channels"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_heads);
    let half_dim = head_dim / 2;
    let rope_scaling = cfg
        .get("rope_scaling")
        .or_else(|| fallback_cfg.get("rope_scaling"))
        .unwrap_or(&serde_json::Value::Null);
    let rope_short_factors = read_f32_array(rope_scaling, "short_factor", half_dim);
    let rope_long_factors = read_f32_array(rope_scaling, "long_factor", half_dim);

    BaseLMConfig {
        hidden_size,
        num_layers: get_usize_any(
            cfg,
            fallback_cfg,
            "feat_encoder",
            &["num_hidden_layers", "num_layers"],
        ),
        num_heads,
        num_kv_heads: cfg
            .get("num_key_value_heads")
            .or_else(|| fallback_cfg.get("num_key_value_heads"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_heads),
        head_dim,
        intermediate_size: get_usize_any(
            cfg,
            fallback_cfg,
            "feat_encoder",
            &["intermediate_size", "ffn_dim"],
        ),
        rms_norm_eps: get_f64(cfg, fallback_cfg, "rms_norm_eps", 1e-5),
        rope_theta: get_f64(cfg, fallback_cfg, "rope_theta", 10000.0),
        rope_factors: rope_short_factors.clone(),
        use_mup: cfg
            .get("use_mup")
            .or_else(|| fallback_cfg.get("use_mup"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        scale_emb: get_f64(cfg, fallback_cfg, "scale_emb", 1.0),
        scale_depth: get_f64(cfg, fallback_cfg, "scale_depth", 1.0),
        original_max_position_embeddings: rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        rope_short_factors,
        rope_long_factors,
        vocab_size: 0,
        max_position: cfg
            .get("max_position_embeddings")
            .or_else(|| fallback_cfg.get("max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        prefix: "feat_encoder.encoder".to_string(),
        no_rope: false,
        is_causal: false,
    }
}

fn get_usize_any(
    cfg: &serde_json::Value,
    fallback_cfg: &serde_json::Value,
    component: &str,
    keys: &[&str],
) -> usize {
    keys.iter()
        .find_map(|key| {
            cfg.get(*key)
                .or_else(|| fallback_cfg.get(*key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or_else(|| panic!("missing one of `{keys:?}` in {component} config"))
}

fn get_f64(
    cfg: &serde_json::Value,
    fallback_cfg: &serde_json::Value,
    key: &str,
    default: f64,
) -> f64 {
    cfg.get(key)
        .or_else(|| fallback_cfg.get(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
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
