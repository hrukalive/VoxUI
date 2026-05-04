mod commands;
mod state;

use state::AppState;

pub fn run() {
    let _ = env_logger::try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::list_lora_dirs,
            commands::list_audio_devices,
            commands::load_model,
            commands::load_lora,
            commands::unload_lora,
            commands::synthesize,
            commands::get_config,
            commands::save_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VoxUI");
}
