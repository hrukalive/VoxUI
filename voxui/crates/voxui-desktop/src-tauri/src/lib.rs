use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use app_core::AppCore;
use tauri::{Manager, RunEvent, WindowEvent};
use types::AppConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloseRequestHandling {
    HideWindow,
    ShutdownApp,
}

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
            if let Err(error) = commands::initialize_sidecar(app.handle(), shared.clone()) {
                tracing::warn!("failed to initialize inference sidecar: {error}");
                if let Ok(mut core) = shared.lock() {
                    core.set_sidecar_init_error(error.to_string());
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                match close_request_handling(window.label()) {
                    CloseRequestHandling::HideWindow => {
                        api.prevent_close();
                        if let Err(error) = window.hide() {
                            tracing::warn!("failed to hide live monitor window: {error}");
                        }
                    }
                    CloseRequestHandling::ShutdownApp => {
                        let app = window.app_handle();
                        if let Some(monitor) = app.get_webview_window("live-monitor") {
                            let _ = monitor.destroy();
                        }
                        commands::shutdown_live_worker_for_app_exit();
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_app_state,
            commands::set_config_patch,
            commands::get_live_state,
            commands::set_live_config_patch,
            commands::clear_live_items,
            commands::connect_openblive,
            commands::disconnect_openblive,
            commands::mock_live_message,
            commands::show_live_monitor,
            commands::send_live_suggestion,
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
            commands::exit_app,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build AhanSays desktop app")
        .run(|_app, event| {
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                commands::shutdown_live_worker_for_app_exit();
            }
        });
}

fn close_request_handling(window_label: &str) -> CloseRequestHandling {
    match window_label {
        "main" => CloseRequestHandling::ShutdownApp,
        _ => CloseRequestHandling::HideWindow,
    }
}

fn runtime_config_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| config_path_next_to_exe(&exe))
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| std::env::temp_dir())
                .join("voxui_config.json")
        })
}

fn config_path_next_to_exe(exe_path: &Path) -> Option<PathBuf> {
    exe_path.parent().map(crate::config::default_config_path)
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "info,voxui_desktop=debug,voxui_inference_sidecar=debug,voxui_inference=info",
        )
    });
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{close_request_handling, config_path_next_to_exe, CloseRequestHandling};

    #[test]
    fn live_monitor_close_hides_window_to_keep_webview_reusable() {
        assert_eq!(
            close_request_handling("live-monitor"),
            CloseRequestHandling::HideWindow
        );
    }

    #[test]
    fn main_window_close_shuts_down_app_resources() {
        assert_eq!(
            close_request_handling("main"),
            CloseRequestHandling::ShutdownApp
        );
    }

    #[test]
    fn runtime_config_path_is_next_to_executable() {
        assert_eq!(
            config_path_next_to_exe(Path::new("D:/Apps/AhanSays/AhanSays.exe")),
            Some(PathBuf::from("D:/Apps/AhanSays/voxui_config.json"))
        );
    }
}

fn startup_config() -> (AppConfig, bool, PathBuf) {
    let config_path = runtime_config_path();
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
