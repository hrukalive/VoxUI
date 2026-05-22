mod commands;
mod desktop_core;
mod state;

use state::AppState;

pub fn run() {
    let subscriber_result = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .try_init();
    match subscriber_result {
        Ok(()) => tracing::debug!("tracing subscriber installed"),
        Err(e) => tracing::debug!("tracing subscriber already set: {e}"),
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::list_model_choices,
            commands::list_lora_dirs,
            commands::list_audio_devices,
            commands::load_model,
            commands::load_model_choice,
            commands::apply_lora,
            commands::synthesize,
            commands::get_config,
            commands::save_config,
            commands::browse_model_root,
            commands::test_audio_device,
            commands::cancel_load,
            commands::cancel_synthesis,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VoxUI");
}
