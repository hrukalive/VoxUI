use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;

use crate::i18n::Language;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: String,
    pub lora_path: Option<String>,
    #[serde(default)]
    pub prompt_wav_path: Option<String>,
    #[serde(default)]
    pub prompt_text: Option<String>,
    #[serde(default)]
    pub reference_wav_path: Option<String>,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    #[serde(default = "default_dit_steps")]
    pub dit_steps: usize,
    #[serde(default)]
    pub language: Language,
}

fn default_dit_steps() -> usize { 10 }

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: "models".into(),
            lora_path: None,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
            backend: "CUDA".into(),
            audio_host: String::new(),
            audio_device: String::new(),
            max_chars: 80,
            dit_steps: 10,
            language: Language::default(),
        }
    }
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        PathBuf::from("voxui_config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(config) = serde_json::from_str(&contents) {
                    return config;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(Self::config_path(), json)?;
        Ok(())
    }
}
