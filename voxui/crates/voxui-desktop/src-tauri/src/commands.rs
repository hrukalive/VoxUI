use std::sync::{Arc, Mutex};

use tauri::{Emitter, State, Window};

use crate::app_core::AppCore;
use crate::generation_queue::HistoryItem;
use crate::types::{AppSnapshot, CommandResult, ConfigPatch, ModelChoice, PlaybackStateEvent};

pub type SharedAppCore = Arc<Mutex<AppCore>>;

pub(crate) fn with_core<T>(
    state: State<'_, SharedAppCore>,
    f: impl FnOnce(&mut AppCore) -> anyhow::Result<T>,
) -> Result<T, String> {
    let mut core = state
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?;
    f(&mut core).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn get_app_state(state: State<'_, SharedAppCore>) -> Result<AppSnapshot, String> {
    with_core(state, |core| Ok(core.snapshot()))
}

#[tauri::command]
pub fn set_config_patch(
    state: State<'_, SharedAppCore>,
    patch: ConfigPatch,
) -> Result<AppSnapshot, String> {
    with_core(state, |core| core.apply_patch(patch))
}

#[tauri::command]
pub fn discover_models(state: State<'_, SharedAppCore>) -> Result<Vec<ModelChoice>, String> {
    with_core(state, |core| core.rescan_models())
}

#[tauri::command]
pub fn enqueue_generation(
    state: State<'_, SharedAppCore>,
    text: String,
) -> Result<HistoryItem, String> {
    with_core(state, |core| core.enqueue_generation(text))
}

#[tauri::command]
pub fn cancel_model_load(state: State<'_, SharedAppCore>) -> Result<CommandResult, String> {
    with_core(state, |core| {
        core.cancel_model_load_state();
        Ok(CommandResult { ok: true })
    })
}

#[tauri::command]
pub fn cancel_generation(
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    with_core(state, |core| {
        Ok(CommandResult {
            ok: core.cancel_generation_item(&item_id),
        })
    })
}

#[tauri::command]
pub fn stop_audio(window: Window) -> Result<CommandResult, String> {
    window
        .emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: None,
                state: "stopped".to_string(),
            },
        )
        .map_err(|error| error.to_string())?;
    Ok(CommandResult { ok: true })
}
