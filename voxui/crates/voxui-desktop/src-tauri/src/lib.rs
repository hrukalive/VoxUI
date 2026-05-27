use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use app_core::AppCore;
use types::AppConfig;

pub mod app_core;
pub mod audio;
pub mod commands;
pub mod config;
pub mod generation_queue;
pub mod inference_sidecar;
pub mod model_discovery;
pub mod playback;
pub mod types;

pub fn run() {
    let _ = tracing_subscriber::fmt().try_init();
    let (config, config_path) = startup_config();
    let mut core = AppCore::from_config(config)
        .expect("persisted app config should initialize desktop app core");
    core.set_config_path(config_path);
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Arc::new(Mutex::new(core)))
        .invoke_handler(tauri::generate_handler![
            commands::get_app_state,
            commands::set_config_patch,
            commands::discover_models,
            commands::get_audio_state,
            commands::browse_model_dir,
            commands::browse_prompt_wav,
            commands::browse_reference_wav,
            commands::test_audio,
            commands::load_model,
            commands::enqueue_generation,
            commands::regenerate,
            commands::cancel_model_load,
            commands::cancel_generation,
            commands::play_audio,
            commands::stop_audio,
        ])
        .run(tauri::generate_context!())
        .expect("failed to run AhanSays desktop app");
}

fn startup_config() -> (AppConfig, PathBuf) {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("AhanSays");
    let config_path = crate::config::default_config_path(&config_dir);
    let config = match crate::config::load_config(&config_path) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(
                "failed to load desktop config from {}: {error}",
                config_path.display()
            );
            AppConfig::default()
        }
    };

    (config, config_path)
}
