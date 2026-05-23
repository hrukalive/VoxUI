use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageMode {
    System,
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationSettings {
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            retry_badcase: true,
            retry_badcase_max_times: 3,
            retry_badcase_ratio_threshold: 6.0,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub model_root: Option<PathBuf>,
    pub selected_model_id: Option<String>,
    pub language: LanguageMode,
    pub backend: BackendKind,
    pub audio_host: Option<String>,
    pub audio_device: Option<String>,
    pub volume: f32,
    pub max_input_chars: usize,
    pub generation: GenerationSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_root: None,
            selected_model_id: None,
            language: LanguageMode::System,
            backend: BackendKind::Cpu,
            audio_host: None,
            audio_device: None,
            volume: 0.8,
            max_input_chars: 280,
            generation: GenerationSettings::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
    pub model_dir: PathBuf,
    pub model_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub model_bytes: u64,
    pub lora_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioHostDto {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDeviceDto {
    pub name: String,
    pub host_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub model_id: String,
    pub backend: BackendKind,
    pub generation: GenerationSettings,
}
