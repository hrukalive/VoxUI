use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use voxui_audio::AudioSystem;
use voxui_inference::VoxCPMEngine;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppConfig {
    #[serde(default = "default_model_root")]
    pub model_root: String,
    #[serde(default)]
    pub selected_model_choice_id: String,
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
    #[serde(default)]
    pub lora_dir: Option<String>,
    #[serde(default)]
    pub prompt_wav_path: Option<String>,
    #[serde(default)]
    pub prompt_text: Option<String>,
    #[serde(default)]
    pub reference_wav_path: Option<String>,
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

fn default_model_dir() -> String {
    "models".into()
}
fn default_model_root() -> String {
    default_program_models_dir()
        .unwrap_or_else(|| PathBuf::from("models"))
        .to_string_lossy()
        .replace('\\', "/")
}
pub fn default_program_models_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("models")))
}
fn default_backend() -> String {
    "CUDA".into()
}
fn default_max_chars() -> usize {
    120
}
fn default_dit_steps() -> usize {
    10
}
fn default_language() -> String {
    "Chinese".into()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_root: default_model_root(),
            selected_model_choice_id: String::new(),
            model_dir: default_model_dir(),
            lora_dir: None,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
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
    pub fn from_save_value_preserving(current: &Self, value: serde_json::Value) -> Result<Self> {
        let has_model_root = value.get("model_root").is_some();
        let has_selected_model_choice_id = value.get("selected_model_choice_id").is_some();
        let mut next: Self = serde_json::from_value(value)?;
        if !has_model_root {
            next.model_root = current.model_root.clone();
        }
        if !has_selected_model_choice_id {
            next.selected_model_choice_id = current.selected_model_choice_id.clone();
        }
        Ok(next)
    }

    pub fn config_path() -> PathBuf {
        PathBuf::from("voxui_config.json")
    }

    pub fn load() -> Self {
        Self::load_from_path(&Self::config_path())
    }

    pub fn load_from_path(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        self.save_to_path(&Self::config_path())
    }

    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

pub struct AppState {
    pub engine: Arc<Mutex<Option<VoxCPMEngine>>>,
    pub audio_system: AudioSystem,
    pub config: Arc<Mutex<AppConfig>>,
    synthesis_busy: Arc<AtomicBool>,
    pub cancel_load: Arc<AtomicBool>,
    pub cancel_synthesis: Arc<AtomicBool>,
}

pub struct SynthesisBusyGuard {
    synthesis_busy: Arc<AtomicBool>,
}

impl Drop for SynthesisBusyGuard {
    fn drop(&mut self) {
        self.synthesis_busy.store(false, Ordering::Release);
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            audio_system: AudioSystem::new(),
            config: Arc::new(Mutex::new(AppConfig::load())),
            synthesis_busy: Arc::new(AtomicBool::new(false)),
            cancel_load: Arc::new(AtomicBool::new(false)),
            cancel_synthesis: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn try_begin_synthesis(&self) -> std::result::Result<SynthesisBusyGuard, String> {
        self.synthesis_busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| SynthesisBusyGuard {
                synthesis_busy: Arc::clone(&self.synthesis_busy),
            })
            .map_err(|_| "Synthesis already in progress".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn config_round_trips_desktop_tts_fields() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("voxui_config.json");
        let config = AppConfig {
            model_root: "models".to_string(),
            selected_model_choice_id: "voxcpm2-fp16".to_string(),
            model_dir: "models/voxcpm2-fp16".to_string(),
            lora_dir: Some("models/voxcpm2-fp16/lora_ft2".to_string()),
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
            backend: "CUDA".to_string(),
            audio_host: "Wasapi".to_string(),
            audio_device: "Speakers".to_string(),
            max_chars: 120,
            dit_steps: 12,
            language: "English".to_string(),
        };

        config.save_to_path(&path).unwrap();
        let loaded = AppConfig::load_from_path(&path);

        assert_eq!(loaded.model_dir, config.model_dir);
        assert_eq!(loaded.lora_dir, config.lora_dir);
        assert_eq!(loaded.prompt_wav_path, config.prompt_wav_path);
        assert_eq!(loaded.prompt_text, config.prompt_text);
        assert_eq!(loaded.reference_wav_path, config.reference_wav_path);
        assert_eq!(loaded.backend, "CUDA");
        assert_eq!(loaded.dit_steps, 12);
    }

    #[test]
    fn config_round_trips_model_root_and_selected_choice() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("voxui_config.json");
        let config = AppConfig {
            model_root: "D:/Models".to_string(),
            selected_model_choice_id: "voxcpm2-q4-lm::lora_ft2.gguf".to_string(),
            model_dir: "models/voxcpm2-q4-lm".to_string(),
            lora_dir: Some("models/voxcpm2-q4-lm/lora_ft2.gguf".to_string()),
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
            backend: "CUDA".to_string(),
            audio_host: "Wasapi".to_string(),
            audio_device: "Speakers".to_string(),
            max_chars: 120,
            dit_steps: 12,
            language: "English".to_string(),
        };

        config.save_to_path(&path).unwrap();
        let loaded = AppConfig::load_from_path(&path);

        assert_eq!(loaded.model_root, "D:/Models");
        assert_eq!(
            loaded.selected_model_choice_id,
            "voxcpm2-q4-lm::lora_ft2.gguf"
        );
    }

    #[test]
    fn missing_model_root_uses_non_empty_default() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("voxui_config.json");
        fs::write(&path, r#"{"backend":"CPU"}"#).unwrap();

        let loaded = AppConfig::load_from_path(&path);

        assert!(!loaded.model_root.trim().is_empty());
        assert_eq!(loaded.selected_model_choice_id, "");
    }

    #[test]
    fn save_value_without_new_selection_fields_preserves_existing_values() {
        let current = AppConfig {
            model_root: "D:/Models".to_string(),
            selected_model_choice_id: "voxcpm2-q4-lm::lora_ft2.gguf".to_string(),
            ..AppConfig::default()
        };
        let value = serde_json::json!({
            "model_dir": "models/legacy",
            "backend": "CPU"
        });

        let next = AppConfig::from_save_value_preserving(&current, value).unwrap();

        assert_eq!(next.model_root, "D:/Models");
        assert_eq!(
            next.selected_model_choice_id,
            "voxcpm2-q4-lm::lora_ft2.gguf"
        );
        assert_eq!(next.model_dir, "models/legacy");
        assert_eq!(next.backend, "CPU");
    }

    #[test]
    fn save_value_with_new_selection_fields_uses_explicit_values() {
        let current = AppConfig {
            model_root: "D:/Models".to_string(),
            selected_model_choice_id: "voxcpm2-q4-lm::lora_ft2.gguf".to_string(),
            ..AppConfig::default()
        };
        let value = serde_json::json!({
            "model_root": "D:/OtherModels",
            "selected_model_choice_id": "",
            "backend": "CPU"
        });

        let next = AppConfig::from_save_value_preserving(&current, value).unwrap();

        assert_eq!(next.model_root, "D:/OtherModels");
        assert_eq!(next.selected_model_choice_id, "");
        assert_eq!(next.backend, "CPU");
    }

    #[test]
    fn default_model_root_uses_program_folder_models_with_normalized_separators() {
        let exe = std::env::current_exe().unwrap();
        let expected_path = exe.parent().unwrap().join("models");

        assert_eq!(default_program_models_dir().unwrap(), expected_path);
        assert_eq!(
            default_model_root(),
            expected_path.to_string_lossy().replace('\\', "/")
        );
    }

    #[test]
    fn busy_guard_rejects_second_synthesis_until_dropped() {
        let state = AppState::new();

        let first = state.try_begin_synthesis();
        assert!(first.is_ok());
        assert!(state.try_begin_synthesis().is_err());

        drop(first);
        assert!(state.try_begin_synthesis().is_ok());
    }
}
