use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use app_core::AppCore;
use tauri::Manager;
use types::AppConfig;

pub mod app_core;
pub mod audio;
pub mod commands;
pub mod config;
pub mod generation_queue;
pub mod inference_sidecar;
pub mod live;
pub mod model_discovery;
pub mod openblive;
pub mod playback;
pub mod types;

pub fn run() {
    init_tracing();
    let (config, backend_saved, config_path) = startup_config();
    let mut core = AppCore::from_loaded_config(config, backend_saved)
        .expect("persisted app config should initialize desktop app core");
    core.set_config_path(config_path);
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Arc::new(Mutex::new(core)))
        .setup(|app| {
            let shared = app.state::<Arc<Mutex<AppCore>>>().inner().clone();
            if let Err(error) = commands::initialize_sidecar(app.handle(), shared) {
                tracing::warn!("failed to initialize inference sidecar: {error}");
            }
            Ok(())
        })
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

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "info,voxui_desktop=debug,voxui_inference_sidecar=debug,voxui_inference=info",
        )
    });
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn startup_config() -> (AppConfig, bool, PathBuf) {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("AhanSays");
    let config_path = crate::config::default_config_path(&config_dir);
    let loaded = match crate::config::load_config_with_metadata(&config_path) {
        Ok(loaded) => loaded,
        Err(error) => {
            tracing::warn!(
                "failed to load desktop config from {}: {error}",
                config_path.display()
            );
            crate::config::LoadedConfig {
                config: AppConfig::default(),
                backend_saved: false,
            }
        }
    };

    (loaded.config, loaded.backend_saved, config_path)
}
