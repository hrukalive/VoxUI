use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use crate::types::{AppConfig, BackendKind, GenerationSettings, LanguageMode, ThemeMode};

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
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let text =
        fs::read_to_string(path).with_context(|| format!("read config from {}", path.display()))?;
    let mut config: AppConfig =
        serde_json::from_str(&text).with_context(|| format!("parse config from {}", path.display()))?;
    config.normalize_for_build();
    Ok(config)
}

pub fn save_config(path: &Path, config: &AppConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create config directory {}", parent.display()))?;
    }
    let mut config = config.clone();
    config.normalize_for_build();
    let text = serde_json::to_string_pretty(&config).context("serialize app config")?;
    fs::write(path, text).with_context(|| format!("write config to {}", path.display()))
}
