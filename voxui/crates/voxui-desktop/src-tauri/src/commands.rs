use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use candle_core::Device;
use tauri::{AppHandle, Emitter, State, Window};
use tauri_plugin_dialog::DialogExt;
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_inference::VoxCPMEngine;

use crate::app_core::AppCore;
use crate::generation_queue::HistoryItem;
use crate::types::{
    AppSnapshot, AudioStateDto, BackendKind, CommandResult, ConfigPatch,
    GenerationDoneEvent, GenerationProgressEvent, ModelChoice, PlaybackStateEvent,
};

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
pub fn get_audio_state() -> AudioStateDto {
    let system = AudioSystem::new();
    crate::audio::audio_state(&system)
}

#[tauri::command]
pub fn browse_model_dir(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .blocking_pick_folder()
        .map(|path| path.to_string()))
}

#[tauri::command]
pub fn browse_prompt_wav(app: AppHandle) -> Result<Option<String>, String> {
    browse_wav_file(app)
}

#[tauri::command]
pub fn browse_reference_wav(app: AppHandle) -> Result<Option<String>, String> {
    browse_wav_file(app)
}

#[tauri::command]
pub fn test_audio(state: State<'_, SharedAppCore>) -> Result<CommandResult, String> {
    let config = with_core(state, |core| Ok(core.snapshot().config))?;
    let system = AudioSystem::new();
    let host = config
        .audio_host
        .clone()
        .unwrap_or_else(|| system.default_host_name());
    let devices = crate::audio::list_devices(&system, &host).map_err(|err| err.to_string())?;
    let device = crate::audio::resolve_output_device_name(
        config.audio_device.clone(),
        &devices,
        system.default_device_name(&host),
    )
        .map_err(|err| err.to_string())?;
    let sample_rate = 48_000;
    let samples = crate::audio::sine_with_fades(sample_rate, 48_000, 440.0, config.volume);
    let mut player =
        AudioPlayer::new(&host, &device, sample_rate).map_err(|err| err.to_string())?;
    player
        .play_blocking(samples)
        .map_err(|err| err.to_string())?;

    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn load_model(
    window: Window,
    state: State<'_, SharedAppCore>,
    choice_id: String,
) -> Result<crate::types::LoadStartResult, String> {
    let (choice, backend, load_id, cancel) = with_core(state.clone(), |core| {
        let choice = core
            .selected_choice()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        if choice.id != choice_id {
            return Err(anyhow::anyhow!(
                "requested choice does not match selected model"
            ));
        }
        let (load_id, cancel) = core.mark_load_started()?;
        Ok((choice, core.config().backend, load_id, cancel))
    })?;

    let shared = state.inner().clone();
    let spawn_result = spawn_background("voxui-model-load", move || {
        let result = load_engine_for_choice(&window, &choice, backend, &cancel);
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
    if let Err(error) = spawn_result {
        let _ = with_core(state, |core| {
            core.mark_load_finished_without_swap_for_load(load_id);
            Ok(())
        });
        return Err(error);
    }

    Ok(crate::types::LoadStartResult {
        started: true,
        choice_id,
    })
}

#[tauri::command]
pub fn enqueue_generation(
    window: Window,
    state: State<'_, SharedAppCore>,
    text: String,
) -> Result<HistoryItem, String> {
    let item = with_core(state.clone(), |core| core.enqueue_generation(text))?;
    spawn_generation(window, state.inner().clone(), item.id.clone());
    Ok(item)
}

#[tauri::command]
pub fn regenerate(
    window: Window,
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    with_core(state.clone(), |core| {
        let config = core.config().clone();
        core.regenerate_item(&item_id, &config)?;
        Ok(CommandResult { ok: true })
    })?;
    spawn_generation(window, state.inner().clone(), item_id);
    Ok(CommandResult { ok: true })
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
pub fn play_audio(
    window: Window,
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    let run = with_core(state.clone(), |core| core.begin_playback(&item_id))?;
    let item_id = run.item_id.clone();
    window
        .emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: Some(item_id),
                state: "playing".to_string(),
            },
        )
        .map_err(|err| err.to_string())?;
    spawn_playback(window, state.inner().clone(), run);
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn stop_audio(
    window: Window,
    state: State<'_, SharedAppCore>,
) -> Result<CommandResult, String> {
    let item_id = with_core(state, |core| Ok(core.stop_playback()))?;
    window
        .emit(
            "playback_state",
            PlaybackStateEvent {
                item_id,
                state: "stopped".to_string(),
            },
        )
        .map_err(|error| error.to_string())?;
    Ok(CommandResult { ok: true })
}

fn load_engine_for_choice(
    window: &Window,
    choice: &crate::types::ModelChoice,
    backend: BackendKind,
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

    let device = device_for_backend(backend)?;
    let mut engine = VoxCPMEngine::load_with_progress(
        &choice.model_dir,
        device,
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

fn device_for_backend(backend: BackendKind) -> Result<Device, String> {
    match backend {
        BackendKind::Cpu => Ok(Device::Cpu),
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|error| error.to_string())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA backend requested but voxui-desktop was built without CUDA support".to_string())
            }
        }
    }
}

fn browse_wav_file(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .add_filter("WAV audio", &["wav"])
        .blocking_pick_file()
        .map(|path| path.to_string()))
}

fn spawn_generation(window: Window, shared: SharedAppCore, item_id: String) {
    let event_item_id = item_id.clone();
    let worker_window = window.clone();
    if let Err(error) = spawn_background("voxui-generation", move || {
        let run = match shared.lock() {
            Ok(mut core) => core.begin_generation_run(&item_id),
            Err(_) => Err("app state lock poisoned".to_string()),
        };
        let run = match run {
            Ok(run) => run,
            Err(error) => {
                let _ = worker_window.emit(
                    "generation_done",
                    GenerationDoneEvent {
                        item_id,
                        status: "skipped".to_string(),
                        error: Some(error),
                        sample_rate: None,
                        duration_seconds: None,
                    },
                );
                return;
            }
        };

        let progress_window = worker_window.clone();
        let progress_shared = shared.clone();
        let progress_item_id = run.item_id.clone();
        let result = AppCore::execute_generation_run(run, |current, total| {
            if let Ok(mut core) = progress_shared.lock() {
                core.mark_generation_progress(&progress_item_id, current, total);
            }
            let _ = progress_window.emit(
                "generation_progress",
                GenerationProgressEvent {
                    item_id: progress_item_id.clone(),
                    current,
                    total,
                },
            );
        });

        let done = match result {
            Ok((run, samples, duration_seconds)) => {
                let item_id = run.item_id.clone();
                let sample_rate = run.sample_rate;
                if let Ok(mut core) = shared.lock() {
                    core.finish_generation_success(run, samples, duration_seconds);
                }
                GenerationDoneEvent {
                    item_id,
                    status: "ready".to_string(),
                    error: None,
                    sample_rate: Some(sample_rate),
                    duration_seconds: Some(duration_seconds),
                }
            }
            Err((run, error)) => {
                let item_id = run.item_id.clone();
                let error = match shared.lock() {
                    Ok(mut core) => core.finish_generation_failure(run, error),
                    Err(_) => error,
                };
                GenerationDoneEvent {
                    item_id,
                    status: "failed".to_string(),
                    error: Some(error),
                    sample_rate: None,
                    duration_seconds: None,
                }
            }
        };
        let _ = worker_window.emit("generation_done", done);
    }) {
        let _ = window.emit(
            "generation_done",
            GenerationDoneEvent {
                item_id: event_item_id,
                status: "skipped".to_string(),
                error: Some(error),
                sample_rate: None,
                duration_seconds: None,
            },
        );
    }
}

fn spawn_playback(
    window: Window,
    shared: SharedAppCore,
    run: crate::app_core::PlaybackRun,
) {
    let event_item_id = run.item_id.clone();
    let worker_window = window.clone();
    let worker_shared = shared.clone();
    if let Err(error) = spawn_background("voxui-playback", move || {
        let config = match worker_shared.lock() {
            Ok(core) => core.snapshot().config,
            Err(_) => {
                let _ = worker_window.emit(
                    "playback_state",
                    PlaybackStateEvent {
                        item_id: Some(run.item_id),
                        state: "error".to_string(),
                    },
                );
                return;
            }
        };
        let system = AudioSystem::new();
        let host = config
            .audio_host
            .clone()
            .unwrap_or_else(|| system.default_host_name());
        let device = crate::audio::list_devices(&system, &host)
            .and_then(|devices| {
                crate::audio::resolve_output_device_name(
                    config.audio_device.clone(),
                    &devices,
                    system.default_device_name(&host),
                )
            })
            .map_err(|err| err.to_string());

        let stop_result = match device {
            Ok(device) => {
                match AudioPlayer::new(&host, &device, run.audio.sample_rate)
                    .and_then(|mut player| {
                        let done = player.play(run.audio.samples)?;
                        wait_for_playback(done, run.stop, &mut player);
                        Ok(())
                    }) {
                    Ok(()) => "stopped".to_string(),
                    Err(error) => format!("error:{error}"),
                }
            }
            Err(error) => format!("error:{error}"),
        };

        let item_id = match worker_shared.lock() {
            Ok(mut core) => core.finish_playback(&run.item_id),
            Err(_) => None,
        };
        let _ = worker_window.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id,
                state: stop_result,
            },
        );
    }) {
        let item_id = match shared.lock() {
            Ok(mut core) => core.finish_playback(&event_item_id),
            Err(_) => None,
        };
        let _ = window.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id,
                state: format!("error:{error}"),
            },
        );
    }
}

fn wait_for_playback(
    done: mpsc::Receiver<()>,
    stop: mpsc::Receiver<()>,
    player: &mut AudioPlayer,
) {
    loop {
        if stop.try_recv().is_ok() {
            player.stop();
            break;
        }
        match done.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

fn spawn_background(name: &'static str, task: impl FnOnce() + Send + 'static) -> Result<(), String> {
    thread::Builder::new()
        .name(name.to_string())
        .spawn(task)
        .map(|_| ())
        .map_err(|error| format!("spawn background task {name}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn background_tasks_do_not_require_tokio_runtime() {
        let (sender, receiver) = mpsc::channel();

        super::spawn_background("voxui-test-background", move || {
            sender.send(42).unwrap();
        })
        .unwrap();

        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).unwrap(), 42);
    }
}
