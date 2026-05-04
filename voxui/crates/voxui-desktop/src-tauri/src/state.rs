use std::sync::Mutex;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use voxui_inference::VoxCPMEngine;
use voxui_audio::AudioSystem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
    #[serde(default)]
    pub lora_dir: Option<String>,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub audio_host: String,
    #[serde(default)]
    pub audio_device: String,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default = "default_dit_steps")]
    pub dit_steps: usize,
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_model_dir() -> String { "models".into() }
fn default_backend() -> String { "CUDA".into() }
fn default_max_chars() -> usize { 80 }
fn default_dit_steps() -> usize { 10 }
fn default_language() -> String { "Chinese".into() }

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            lora_dir: None,
            backend: default_backend(),
            audio_host: String::new(),
            audio_device: String::new(),
            max_chars: default_max_chars(),
            dit_steps: default_dit_steps(),
            language: default_language(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let path = PathBuf::from("voxui_config.json");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write("voxui_config.json", json)?;
        Ok(())
    }
}

pub struct AppState {
    pub engine: Mutex<Option<VoxCPMEngine>>,
    pub audio_system: AudioSystem,
    pub config: Mutex<AppConfig>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            audio_system: AudioSystem::new(),
            config: Mutex::new(AppConfig::load()),
        }
    }
}
