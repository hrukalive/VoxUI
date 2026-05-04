use std::fs;

use voxui_inference::{BundleManifest, ModelVariant};

#[test]
fn manifest_parses_variant_and_components() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("manifest.json"),
        r#"{
            "schema_version": 1,
            "architecture": "voxcpm2",
            "variant": "2.0",
            "source_model_dir": "source",
            "source_weight_format": "safetensors",
            "special_tokens": {"audio_start":101,"audio_end":102,"ref_audio_start":103,"ref_audio_end":104},
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9.0,
            "audio_vae": {"sample_rate":16000,"out_sample_rate":48000,"latent_dim":64,"chunk_size":20,"decode_chunk_size":240,"encoder_rates":[2,5,8,8],"decoder_rates":[8,6,5,2,2,2]},
            "components": {"base_lm":"base_lm.gguf","residual_lm":"residual_lm.gguf","feat_encoder":"feat_encoder.gguf","feat_decoder":"feat_decoder.gguf","audio_vae":"audio_vae.gguf","projections":"projections.gguf"},
            "quantization": {}
        }"#,
    )
    .unwrap();

    let manifest = BundleManifest::load(dir.path()).unwrap();
    assert_eq!(manifest.variant, ModelVariant::VoxCpm2);
    assert_eq!(manifest.special_tokens.audio_start, 101);
    assert!(manifest.component_path(dir.path(), "feat_decoder").unwrap().ends_with("feat_decoder.gguf"));
}

#[test]
fn manifest_rejects_reference_tokens_for_non_v2() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("manifest.json"),
        r#"{
            "schema_version": 1,
            "architecture": "voxcpm",
            "variant": "1.5",
            "source_model_dir": "source",
            "source_weight_format": "safetensors",
            "special_tokens": {"audio_start":101,"audio_end":102,"ref_audio_start":103,"ref_audio_end":104},
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 256,
            "scalar_quantization_scale": 9.0,
            "audio_vae": {"sample_rate":16000,"latent_dim":64,"chunk_size":20,"decode_chunk_size":240,"encoder_rates":[2,5,8,8],"decoder_rates":[8,6,5,2,2,2]},
            "components": {"base_lm":"base_lm.gguf","residual_lm":"residual_lm.gguf","feat_encoder":"feat_encoder.gguf","feat_decoder":"feat_decoder.gguf","audio_vae":"audio_vae.gguf","projections":"projections.gguf"},
            "quantization": {}
        }"#,
    )
    .unwrap();

    let err = BundleManifest::load(dir.path()).unwrap_err();
    assert!(err.to_string().contains("ref_audio"));
}
