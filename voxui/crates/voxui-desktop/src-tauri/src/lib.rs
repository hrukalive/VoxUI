use std::sync::{Arc, Mutex};

use app_core::AppCore;
use types::AppConfig;

pub mod app_core;
pub mod audio;
pub mod commands;
pub mod config;
pub mod generation_queue;
pub mod model_discovery;
pub mod playback;
pub mod types;

pub fn run() {
    let _ = tracing_subscriber::fmt().try_init();
    let core = AppCore::from_config(AppConfig::default())
        .expect("default app config should initialize desktop app core");
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Arc::new(Mutex::new(core)))
        .invoke_handler(tauri::generate_handler![
            commands::get_app_state,
            commands::set_config_patch,
            commands::discover_models,
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
