use std::fs;

use voxui_inference::{GgufModelLoader, ModelConfig, ModelVariant};

#[test]
fn model_loader_requires_model_gguf_in_directory() {
    let dir = tempfile::tempdir().unwrap();
    let err = GgufModelLoader::from_model_dir(dir.path(), candle_core::Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("model.gguf"));
}

#[test]
fn model_config_parses_variant_and_audio_vae_from_config_json() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
            "architectures": ["voxcpm2"],
            "patch_size": 4,
            "feat_dim": 80,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0,
            "audio_vae_config": {
                "sample_rate": 16000,
                "out_sample_rate": 48000,
                "latent_dim": 96,
                "chunk_size": 24,
                "decode_chunk_size": 320,
                "encoder_rates": [2, 5, 8, 8],
                "decoder_rates": [8, 6, 5, 2, 2, 2]
            },
            "lm_config": {"hidden_size": 2048},
            "encoder_config": {"num_hidden_layers": 12},
            "dit_config": {"num_layers": 16},
            "residual_lm_num_layers": 6,
            "residual_lm_no_rope": true
        }"#,
    )
    .unwrap();

    let config = ModelConfig::load(dir.path(), ModelVariant::VoxCpm2).unwrap();
    assert_eq!(config.schema_version, 2);
    assert_eq!(config.architecture, "voxcpm2");
    assert_eq!(config.variant, ModelVariant::VoxCpm2);
    assert_eq!(config.special_tokens.audio_start, 101);
    assert_eq!(config.special_tokens.audio_end, 102);
    assert_eq!(config.special_tokens.ref_audio_start, Some(103));
    assert_eq!(config.special_tokens.ref_audio_end, Some(104));
    assert_eq!(config.patch_size, 4);
    assert_eq!(config.feat_dim, 80);
    assert_eq!(config.scalar_quantization_latent_dim, 512);
    assert_eq!(config.scalar_quantization_scale, 9.0);
    assert_eq!(config.audio_vae.sample_rate, 16000);
    assert_eq!(config.audio_vae.out_sample_rate, Some(48000));
    assert_eq!(config.output_sample_rate(), 48000);
    assert_eq!(config.audio_vae.latent_dim, 96);
    assert_eq!(config.audio_vae.chunk_size, 24);
    assert_eq!(config.audio_vae.decode_chunk_size, 320);
    assert_eq!(config.audio_vae.encoder_rates, vec![2, 5, 8, 8]);
    assert_eq!(config.audio_vae.decoder_rates, vec![8, 6, 5, 2, 2, 2]);
    assert_eq!(config.lm_config["hidden_size"], 2048);
    assert_eq!(config.encoder_config["num_hidden_layers"], 12);
    assert_eq!(config.dit_config["num_layers"], 16);
    assert_eq!(config.residual_lm_num_layers, Some(6));
    assert_eq!(config.residual_lm_no_rope, Some(true));
}

#[test]
fn model_config_rejects_v2_variant_with_non_v2_architecture() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
            "architectures": ["voxcpm"],
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0
        }"#,
    )
    .unwrap();

    let err = ModelConfig::load(dir.path(), ModelVariant::VoxCpm2).unwrap_err();
    assert!(err
        .to_string()
        .contains("VoxCPM2 variant requires voxcpm2 architecture"));
}

#[test]
fn model_config_rejects_non_v2_variant_with_v2_architecture() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
            "architectures": ["voxcpm2"],
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 256,
            "scalar_quantization_scale": 9.0
        }"#,
    )
    .unwrap();

    let err = ModelConfig::load(dir.path(), ModelVariant::VoxCpm15).unwrap_err();
    assert!(err
        .to_string()
        .contains("voxcpm2 architecture requires VoxCPM2 variant"));
}
