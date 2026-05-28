use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use crate::types::{AppConfig, BackendKind, GenerationSettings, LanguageMode, ThemeMode};

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedConfig {
    pub config: AppConfig,
    pub backend_saved: bool,
}

pub fn detect_system_language() -> LanguageMode {
    detect_language_from_locale(sys_locale::get_locale().as_deref())
}

pub fn detect_language_from_locale(locale: Option<&str>) -> LanguageMode {
    match locale {
        Some(value) if value.to_ascii_lowercase().starts_with("zh") => LanguageMode::Chinese,
        _ => LanguageMode::English,
    }
}

pub fn default_config_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("voxui_config.json")
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    load_config_with_metadata(path).map(|loaded| loaded.config)
}

pub fn load_config_with_metadata(path: &Path) -> Result<LoadedConfig> {
    if !path.exists() {
        return Ok(LoadedConfig {
            config: AppConfig::default(),
            backend_saved: false,
        });
    }

    let text =
        fs::read_to_string(path).with_context(|| format!("read config from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse config from {}", path.display()))?;
    let backend_saved = value
        .as_object()
        .is_some_and(|object| object.contains_key("backend"));
    let config: AppConfig = serde_json::from_value(value)
        .with_context(|| format!("parse config from {}", path.display()))?;
    Ok(LoadedConfig {
        config,
        backend_saved,
    })
}

pub fn save_config(path: &Path, config: &AppConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create config directory {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(config).context("serialize app config")?;
    fs::write(path, text).with_context(|| format!("write config to {}", path.display()))
}
