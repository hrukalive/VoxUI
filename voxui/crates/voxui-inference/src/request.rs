//! Synthesis request types matching VoxCPM generate semantics.

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::manifest::ModelVariant;

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub normalize: bool,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub consolidate_n: usize,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            normalize: false,
            retry_badcase: true,
            retry_badcase_max_times: 3,
            retry_badcase_ratio_threshold: 6.0,
            consolidate_n: 1,
        }
    }
}

impl SynthesisRequest {
    pub fn validated(mut self, variant: ModelVariant) -> Result<Self> {
        self.text = collapse_whitespace(&self.text);
        if self.text.is_empty() {
            bail!("text must not be empty");
        }

        if let Some(prompt_text) = self.prompt_text.as_mut() {
            *prompt_text = collapse_whitespace(prompt_text);
        }

        if self.prompt_wav_path.is_some()
            && self
                .prompt_text
                .as_ref()
                .map(|text| text.is_empty())
                .unwrap_or(true)
        {
            bail!("prompt_text is required when prompt_wav_path is present");
        }
        if self.prompt_wav_path.is_none() && self.prompt_text.is_some() {
            bail!("prompt_wav_path is required when prompt_text is present");
        }
        if self.reference_wav_path.is_some() && variant != ModelVariant::VoxCpm2 {
            bail!("Reference audio requires VoxCPM2");
        }
        if self.normalize {
            bail!(
                "normalize=true is not supported until the Rust VoxCPM normalizer is implemented"
            );
        }
        if self.min_len > self.max_len {
            bail!("min_len must be <= max_len");
        }
        if self.inference_timesteps == 0 {
            bail!("inference_timesteps must be greater than zero");
        }
        if self.cfg_value <= 0.0 {
            bail!("cfg_value must be greater than zero");
        }
        if self.retry_badcase_ratio_threshold <= 0.0 {
            bail!("retry_badcase_ratio_threshold must be greater than zero");
        }
        if self.consolidate_n == 0 {
            bail!("consolidate_n must be greater than zero");
        }
        Ok(self)
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
