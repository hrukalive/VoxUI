use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use tauri::{Emitter, State, Window};
use tokio::task;
use voxui_inference::VoxCPMEngine;

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
pub fn load_model(
    window: Window,
    state: State<'_, SharedAppCore>,
    choice_id: String,
) -> Result<crate::types::LoadStartResult, String> {
    let (choice, load_id, cancel) = with_core(state.clone(), |core| {
        let choice = core.selected_choice().map_err(|err| anyhow::anyhow!(err.to_string()))?;
        if choice.id != choice_id {
            return Err(anyhow::anyhow!(
                "requested choice does not match selected model"
            ));
        }
        let (load_id, cancel) = core.mark_load_started()?;
        Ok((choice, load_id, cancel))
    })?;

    let shared = state.inner().clone();
    task::spawn_blocking(move || {
        let result = load_engine_for_choice(&window, &choice, &cancel);
        let done = match result {
            Ok(engine) => {
                let loaded_model_id = if let Ok(mut core) = shared.lock() {
                    if !core.mark_load_success(load_id, choice.id.clone(), engine) {
                        return;
                    }
                    core.snapshot().loaded_model_id
                } else {
                    None
                };
                if loaded_model_id.is_none() {
                    crate::types::ModelLoadDoneEvent {
                        status: "error".to_string(),
                        selected_model_id: Some(choice.id.clone()),
                        loaded_model_id: None,
                        error: Some("app state lock poisoned".to_string()),
                    }
                } else {
                    crate::types::ModelLoadDoneEvent {
                        status: "success".to_string(),
                        selected_model_id: Some(choice.id.clone()),
                        loaded_model_id,
                        error: None,
                    }
                }
            }
            Err(error) => {
                if let Ok(mut core) = shared.lock() {
                    if !core.mark_load_finished_without_swap_for_load(load_id) {
                        return;
                    }
                    let loaded = core.snapshot().loaded_model_id;
                    let _ = window.emit(
                        "model_load_done",
                        crate::types::ModelLoadDoneEvent {
                            status: "error".to_string(),
                            selected_model_id: Some(choice.id.clone()),
                            loaded_model_id: loaded,
                            error: Some(error),
                        },
                    );
                    return;
                }
                crate::types::ModelLoadDoneEvent {
                    status: "error".to_string(),
                    selected_model_id: Some(choice.id.clone()),
                    loaded_model_id: None,
                    error: Some("app state lock poisoned".to_string()),
                }
            }
        };
        let _ = window.emit("model_load_done", done);
    });

    Ok(crate::types::LoadStartResult {
        started: true,
        choice_id,
    })
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

fn load_engine_for_choice(
    window: &Window,
    choice: &crate::types::ModelChoice,
    cancel: &AtomicBool,
) -> Result<VoxCPMEngine, String> {
    let total_bytes = choice.model_bytes + choice.lora_bytes;
    window
        .emit(
            "model_load_progress",
            crate::types::ModelLoadProgressEvent {
                phase: "reading".to_string(),
                loaded_bytes: total_bytes,
                total_bytes,
                component: None,
                component_index: 0,
                component_total: 0,
            },
        )
        .map_err(|err| err.to_string())?;

    let mut engine = VoxCPMEngine::load_with_progress(
        &choice.model_dir,
        Device::Cpu,
        |current, total| {
            let _ = window.emit(
                "model_load_progress",
                crate::types::ModelLoadProgressEvent {
                    phase: "device_loading".to_string(),
                    loaded_bytes: total_bytes,
                    total_bytes,
                    component: None,
                    component_index: current,
                    component_total: total,
                },
            );
        },
        Some(cancel),
    )
    .map_err(|err| err.to_string())?;

    if let Some(lora_path) = choice.lora_path.as_ref() {
        engine.load_lora(lora_path).map_err(|err| err.to_string())?;
    }

    Ok(engine)
}
