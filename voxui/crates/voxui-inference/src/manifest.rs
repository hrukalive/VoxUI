//! Manifest types for native VoxCPM model bundles.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    pub sample_rate: u32,
    pub out_sample_rate: Option<u32>,
    pub latent_dim: usize,
    pub chunk_size: usize,
    pub decode_chunk_size: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_rates: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComponentFiles {
    pub base_lm: String,
    pub residual_lm: String,
    pub feat_encoder: String,
    pub feat_decoder: String,
    pub audio_vae: String,
    pub projections: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub architecture: String,
    pub variant: ModelVariant,
    pub source_model_dir: String,
    pub source_weight_format: String,
    pub special_tokens: SpecialTokens,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: f32,
    pub audio_vae: AudioVaeManifest,
    pub components: ComponentFiles,
    #[serde(default)]
    pub lm_config: serde_json::Value,
    #[serde(default)]
    pub encoder_config: serde_json::Value,
    #[serde(default)]
    pub dit_config: serde_json::Value,
    #[serde(default)]
    pub residual_lm_num_layers: Option<usize>,
    #[serde(default)]
    pub residual_lm_no_rope: Option<bool>,
    #[serde(default)]
    pub quantization: HashMap<String, String>,
}

impl BundleManifest {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let manifest_path = model_dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", manifest_path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.audio_vae.out_sample_rate.unwrap_or(self.audio_vae.sample_rate)
    }

    pub fn component_path(&self, model_dir: &Path, component: &str) -> Result<PathBuf> {
        let file = match component {
            "base_lm" => &self.components.base_lm,
            "residual_lm" => &self.components.residual_lm,
            "feat_encoder" => &self.components.feat_encoder,
            "feat_decoder" => &self.components.feat_decoder,
            "audio_vae" => &self.components.audio_vae,
            "projections" => &self.components.projections,
            other => bail!("unknown component `{other}`"),
        };
        Ok(model_dir.join(file))
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!("unsupported VoxCPM bundle schema {}", self.schema_version);
        }
        if self.special_tokens.audio_start != 101 || self.special_tokens.audio_end != 102 {
            bail!("unexpected audio special tokens");
        }
        match self.variant {
            ModelVariant::VoxCpm2 => {
                if self.architecture != "voxcpm2" {
                    bail!("VoxCPM2 variant requires voxcpm2 architecture");
                }
                if self.special_tokens.ref_audio_start != Some(103)
                    || self.special_tokens.ref_audio_end != Some(104)
                {
                    bail!("VoxCPM2 manifest must include ref_audio tokens 103 and 104");
                }
            }
            _ => {
                if self.special_tokens.ref_audio_start.is_some()
                    || self.special_tokens.ref_audio_end.is_some()
                {
                    bail!("ref_audio tokens are only valid for VoxCPM2");
                }
            }
        }
        Ok(())
    }
}
