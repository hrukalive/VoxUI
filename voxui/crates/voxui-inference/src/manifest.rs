//! Manifest types for native VoxCPM model bundles.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ModelVariant {
    #[serde(rename = "0.5")]
    VoxCpm05,
    #[serde(rename = "1.5")]
    VoxCpm15,
    #[serde(rename = "2.0")]
    VoxCpm2,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokens {
    pub audio_start: u32,
    pub audio_end: u32,
    pub ref_audio_start: Option<u32>,
    pub ref_audio_end: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeManifest {
    #[serde(default)]
    pub encoder_dim: Option<usize>,
    #[serde(default)]
    pub decoder_dim: Option<usize>,
    pub sample_rate: u32,
    pub out_sample_rate: Option<u32>,
    pub latent_dim: usize,
    pub chunk_size: usize,
    pub decode_chunk_size: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_rates: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub schema_version: u32,
    pub architecture: String,
    pub variant: ModelVariant,
    pub special_tokens: SpecialTokens,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: f32,
    pub audio_vae: AudioVaeManifest,
    pub lm_config: serde_json::Value,
    pub encoder_config: serde_json::Value,
    pub dit_config: serde_json::Value,
    pub residual_lm_num_layers: Option<usize>,
    pub residual_lm_no_rope: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawModelConfig {
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    patch_size: usize,
    #[serde(default)]
    feat_dim: Option<usize>,
    scalar_quantization_latent_dim: usize,
    scalar_quantization_scale: f32,
    #[serde(default)]
    audio_vae_config: RawAudioVaeConfig,
    #[serde(default)]
    lm_config: serde_json::Value,
    #[serde(default)]
    encoder_config: serde_json::Value,
    #[serde(default)]
    dit_config: serde_json::Value,
    #[serde(default)]
    residual_lm_num_layers: Option<usize>,
    #[serde(default)]
    residual_lm_no_rope: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct RawAudioVaeConfig {
    #[serde(default)]
    encoder_dim: Option<usize>,
    #[serde(default)]
    decoder_dim: Option<usize>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    out_sample_rate: Option<u32>,
    #[serde(default)]
    latent_dim: Option<usize>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    decode_chunk_size: Option<usize>,
    #[serde(default)]
    encoder_rates: Vec<usize>,
    #[serde(default)]
    decoder_rates: Vec<usize>,
}

impl ModelConfig {
    pub fn load(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let raw: RawModelConfig = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", config_path.display()))?;
        Self::from_raw(raw, variant)
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.audio_vae
            .out_sample_rate
            .unwrap_or(self.audio_vae.sample_rate)
    }

    fn from_raw(raw: RawModelConfig, variant: ModelVariant) -> Result<Self> {
        let architecture = raw
            .architecture
            .or_else(|| raw.architectures.into_iter().next())
            .context("config.json must include architecture or architectures")?;
        validate_config_architecture(&architecture, variant)?;

        let audio_vae = AudioVaeManifest {
            encoder_dim: raw.audio_vae_config.encoder_dim,
            decoder_dim: raw.audio_vae_config.decoder_dim,
            sample_rate: raw
                .audio_vae_config
                .sample_rate
                .unwrap_or_else(|| default_sample_rate(variant)),
            out_sample_rate: raw.audio_vae_config.out_sample_rate,
            latent_dim: raw
                .audio_vae_config
                .latent_dim
                .or(raw.feat_dim)
                .unwrap_or(64),
            chunk_size: raw.audio_vae_config.chunk_size.unwrap_or(20),
            decode_chunk_size: raw.audio_vae_config.decode_chunk_size.unwrap_or(240),
            encoder_rates: raw.audio_vae_config.encoder_rates,
            decoder_rates: raw.audio_vae_config.decoder_rates,
        };

        Ok(Self {
            schema_version: 2,
            architecture,
            variant,
            special_tokens: special_tokens_for_variant(variant),
            patch_size: raw.patch_size,
            feat_dim: raw.feat_dim.unwrap_or(64),
            scalar_quantization_latent_dim: raw.scalar_quantization_latent_dim,
            scalar_quantization_scale: raw.scalar_quantization_scale,
            audio_vae,
            lm_config: raw.lm_config,
            encoder_config: raw.encoder_config,
            dit_config: raw.dit_config,
            residual_lm_num_layers: raw.residual_lm_num_layers,
            residual_lm_no_rope: raw.residual_lm_no_rope,
        })
    }
}

fn special_tokens_for_variant(variant: ModelVariant) -> SpecialTokens {
    let (ref_audio_start, ref_audio_end) = match variant {
        ModelVariant::VoxCpm2 => (Some(103), Some(104)),
        _ => (None, None),
    };
    SpecialTokens {
        audio_start: 101,
        audio_end: 102,
        ref_audio_start,
        ref_audio_end,
    }
}

fn default_sample_rate(variant: ModelVariant) -> u32 {
    match variant {
        ModelVariant::VoxCpm2 => 16_000,
        _ => 44_100,
    }
}

fn validate_config_architecture(architecture: &str, variant: ModelVariant) -> Result<()> {
    match variant {
        ModelVariant::VoxCpm2 => {
            if architecture != "voxcpm2" {
                bail!("VoxCPM2 variant requires voxcpm2 architecture");
            }
        }
        _ => {
            if architecture == "voxcpm2" {
                bail!("voxcpm2 architecture requires VoxCPM2 variant");
            }
        }
    }
    Ok(())
}
