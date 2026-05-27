use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use tauri::{AppHandle, Emitter, State, Window};
use tauri_plugin_dialog::DialogExt;
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_sidecar_protocol::{
    BackendKind as ProtocolBackendKind, Frame, OperationStatus, SidecarCommand, SidecarEvent,
    SynthesisRequestDto,
};

use crate::app_core::AppCore;
use crate::generation_queue::HistoryItem;
use crate::inference_sidecar::{SidecarProcess, SidecarReaderEvent};
use crate::types::{
    AppSnapshot, AudioStateDto, BackendKind, CommandResult, ConfigPatch, GenerationDoneEvent,
    GenerationProgressEvent, ModelChoice, PlaybackStateEvent,
};

pub type SharedAppCore = Arc<Mutex<AppCore>>;

static SIDECAR_PROCESS: OnceLock<Mutex<Option<SidecarProcess>>> = OnceLock::new();

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
    let samples = crate::audio::sine_with_fades(sample_rate, 48_000, 440.0, 1.0);
    let mut player =
        AudioPlayer::new(&host, &device, sample_rate).map_err(|err| err.to_string())?;
    player
        .play_with_volume(samples, config.volume)
        .and_then(|done| {
            done.recv()
                .map_err(|_| anyhow::anyhow!("playback channel closed unexpectedly"))
        })
        .map_err(|err| err.to_string())?;

    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn load_model(
    window: Window,
    state: State<'_, SharedAppCore>,
    choice_id: String,
) -> Result<crate::types::LoadStartResult, String> {
    let (choice, backend, load_id) = with_core(state.clone(), |core| {
        let choice = core
            .selected_choice()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        if choice.id != choice_id {
            return Err(anyhow::anyhow!(
                "requested choice does not match selected model"
            ));
        }
        let (load_id, _) = core.mark_load_started()?;
        Ok((choice, core.config().backend, load_id))
    })?;

    let shared = state.inner().clone();
    let command = SidecarCommand::LoadModel {
        load_id,
        model_dir: choice.model_dir.clone(),
        lora_path: choice.lora_path.clone(),
        backend: protocol_backend(backend),
    };
    if let Err(error) = send_sidecar_command(&window, shared, command) {
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
    kick_generation_queue(&window, state.inner().clone());
    Ok(item)
}

#[tauri::command]
pub fn regenerate(
    window: Window,
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    let stopped_item_id = with_core(state.clone(), |core| {
        let config = core.config().clone();
        core.regenerate_item_stopping_playback(&item_id, &config)
    })?;
    if stopped_item_id.is_some() {
        window
            .emit(
                "playback_state",
                PlaybackStateEvent {
                    item_id: stopped_item_id,
                    state: "stopped".to_string(),
                },
            )
            .map_err(|err| err.to_string())?;
    }
    kick_generation_queue(&window, state.inner().clone());
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn cancel_model_load(
    window: Window,
    state: State<'_, SharedAppCore>,
) -> Result<CommandResult, String> {
    let load_id = with_core(state.clone(), |core| Ok(core.cancel_model_load_state()))?;
    if let Some(load_id) = load_id {
        let _ = send_sidecar_command(
            &window,
            state.inner().clone(),
            SidecarCommand::CancelLoad { load_id },
        );
    }
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn cancel_generation(
    window: Window,
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    let canceled = with_core(state.clone(), |core| {
        Ok(CommandResult {
            ok: core.cancel_generation_item(&item_id),
        })
    })?;
    if canceled.ok {
        let _ = send_sidecar_command(
            &window,
            state.inner().clone(),
            SidecarCommand::CancelSynthesis { item_id },
        );
    }
    Ok(canceled)
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

fn kick_generation_queue(window: &Window, shared: SharedAppCore) {
    let next_run = match shared.lock() {
        Ok(mut core) => core.begin_next_generation_run(),
        Err(_) => return,
    };

    match next_run {
        Ok(Some(run)) => {
            let item_id = run.item_id.clone();
            let command = SidecarCommand::Synthesize {
                item_id: item_id.clone(),
                request: synthesis_request_dto(run.request.clone()),
                streaming: run.streaming,
            };
            if let Err(error) = send_sidecar_command(window, shared.clone(), command) {
                let done = match shared.lock() {
                    Ok(mut core) => {
                        let error = core.finish_generation_failure(run, error);
                        GenerationDoneEvent {
                            item_id,
                            status: "failed".to_string(),
                            error: Some(error),
                            sample_rate: None,
                            duration_seconds: None,
                        }
                    }
                    Err(_) => GenerationDoneEvent {
                        item_id,
                        status: "failed".to_string(),
                        error: Some("app state lock poisoned".to_string()),
                        sample_rate: None,
                        duration_seconds: None,
                    },
                };
                let _ = window.emit("generation_done", done);
                kick_generation_queue(window, shared);
            }
        }
        Ok(None) => {}
        Err(error) => tracing::warn!("failed to start queued generation: {error}"),
    }
}

fn protocol_backend(backend: BackendKind) -> ProtocolBackendKind {
    match backend {
        BackendKind::Cpu => ProtocolBackendKind::Cpu,
        BackendKind::Cuda => ProtocolBackendKind::Cuda,
    }
}

fn sidecar_slot() -> &'static Mutex<Option<SidecarProcess>> {
    SIDECAR_PROCESS.get_or_init(|| Mutex::new(None))
}

fn send_sidecar_command(
    window: &Window,
    shared: SharedAppCore,
    command: SidecarCommand,
) -> Result<(), String> {
    ensure_sidecar(window, shared)?;
    let mut guard = sidecar_slot()
        .lock()
        .map_err(|_| "sidecar process lock poisoned".to_string())?;
    let Some(process) = guard.as_mut() else {
        return Err("sidecar process is unavailable".to_string());
    };
    if let Err(error) = process.send(command) {
        let _ = process.kill();
        *guard = None;
        return Err(error.to_string());
    }
    Ok(())
}

fn ensure_sidecar(window: &Window, shared: SharedAppCore) -> Result<(), String> {
    let mut guard = sidecar_slot()
        .lock()
        .map_err(|_| "sidecar process lock poisoned".to_string())?;
    if guard.is_some() {
        return Ok(());
    }

    let path = resolve_sidecar_path()?;
    let (process, receiver) = SidecarProcess::spawn(&path).map_err(|error| error.to_string())?;
    spawn_sidecar_reader(window.clone(), shared, receiver)?;
    *guard = Some(process);
    Ok(())
}

fn resolve_sidecar_path() -> Result<std::path::PathBuf, String> {
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    let dir = exe
        .parent()
        .ok_or_else(|| format!("executable path has no parent: {}", exe.display()))?;
    let candidates = [
        dir.join("voxui-inference-sidecar.exe"),
        dir.join("voxui-inference-sidecar"),
    ];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| {
            format!(
                "voxui-inference-sidecar was not found next to {}",
                exe.display()
            )
        })
}

fn spawn_sidecar_reader(
    window: Window,
    shared: SharedAppCore,
    receiver: mpsc::Receiver<SidecarReaderEvent>,
) -> Result<(), String> {
    spawn_background("voxui-sidecar-reader", move || {
        for event in receiver {
            match event {
                SidecarReaderEvent::Frame(frame) => {
                    handle_sidecar_event(&window, shared.clone(), frame);
                }
                SidecarReaderEvent::Error(error) => {
                    tracing::error!("sidecar reader error: {error}");
                    let _ = window.emit(
                        "generation_done",
                        GenerationDoneEvent {
                            item_id: String::new(),
                            status: "failed".to_string(),
                            error: Some(error),
                            sample_rate: None,
                            duration_seconds: None,
                        },
                    );
                    break;
                }
                SidecarReaderEvent::Eof => break,
            }
        }
        if let Ok(mut guard) = sidecar_slot().lock() {
            *guard = None;
        }
    })
}

fn handle_sidecar_event(window: &Window, shared: SharedAppCore, frame: Frame<SidecarEvent>) {
    match frame.header {
        SidecarEvent::Ready => {}
        SidecarEvent::ModelLoadProgress {
            phase,
            loaded_bytes,
            total_bytes,
            component,
            component_index,
            component_total,
            ..
        } => {
            let _ = window.emit(
                "model_load_progress",
                crate::types::ModelLoadProgressEvent {
                    phase,
                    loaded_bytes,
                    total_bytes,
                    component,
                    component_index,
                    component_total,
                },
            );
        }
        SidecarEvent::ModelLoadDone {
            load_id,
            status,
            sample_rate,
            error,
        } => {
            let done = match shared.lock() {
                Ok(mut core) => {
                    match status {
                        OperationStatus::Success => {
                            if let (Ok(choice), Some(sample_rate)) =
                                (core.selected_choice(), sample_rate)
                            {
                                core.mark_load_success(load_id, choice.id, sample_rate);
                            }
                        }
                        OperationStatus::Canceled | OperationStatus::Failed => {
                            core.mark_load_finished_without_swap_for_load(load_id);
                        }
                    }
                    let snapshot = core.snapshot();
                    crate::types::ModelLoadDoneEvent {
                        status: operation_status_label(status).to_string(),
                        selected_model_id: snapshot.selected_model_id,
                        loaded_model_id: snapshot.loaded_model_id,
                        error,
                    }
                }
                Err(_) => crate::types::ModelLoadDoneEvent {
                    status: "failed".to_string(),
                    selected_model_id: None,
                    loaded_model_id: None,
                    error: Some("app state lock poisoned".to_string()),
                },
            };
            let _ = window.emit("model_load_done", done);
        }
        SidecarEvent::GenerationProgress {
            item_id,
            current,
            total,
        } => {
            if let Ok(mut core) = shared.lock() {
                core.mark_generation_progress(&item_id, current, total);
            }
            let _ = window.emit(
                "generation_progress",
                GenerationProgressEvent {
                    item_id,
                    current,
                    total,
                },
            );
        }
        SidecarEvent::AudioChunk {
            item_id,
            sample_rate,
            ..
        } => {
            if let Ok(samples) =
                crate::inference_sidecar::sidecar_samples_from_payload(&frame.payload)
            {
                if let Ok(mut core) = shared.lock() {
                    let _ = core.append_generation_audio_chunk(&item_id, samples, sample_rate);
                }
            }
        }
        SidecarEvent::AudioFinal {
            item_id,
            sample_rate,
            ..
        } => {
            if let Ok(samples) =
                crate::inference_sidecar::sidecar_samples_from_payload(&frame.payload)
            {
                if let Ok(mut core) = shared.lock() {
                    let _ = core.append_generation_audio_chunk(&item_id, samples, sample_rate);
                }
            }
        }
        SidecarEvent::GenerationDone {
            item_id,
            status,
            sample_rate,
            duration_seconds,
            error,
        } => {
            let mut playback_run = None;
            let done = match shared.lock() {
                Ok(mut core) => {
                    match status {
                        OperationStatus::Success => {
                            if let (Some(rate), Some(duration)) = (sample_rate, duration_seconds) {
                                let _ = core.finish_generation_success_from_sidecar(
                                    &item_id, rate, duration,
                                );
                                playback_run =
                                    core.begin_or_queue_auto_playback(&item_id).ok().flatten();
                            }
                        }
                        OperationStatus::Canceled => {
                            let _ = core.finish_generation_canceled_from_sidecar(&item_id);
                        }
                        OperationStatus::Failed => {
                            let _ = core.finish_generation_failure_from_sidecar(
                                &item_id,
                                error
                                    .clone()
                                    .unwrap_or_else(|| "generation failed".to_string()),
                            );
                        }
                    }
                    GenerationDoneEvent {
                        item_id: item_id.clone(),
                        status: operation_status_label(status).to_string(),
                        error,
                        sample_rate,
                        duration_seconds,
                    }
                }
                Err(_) => GenerationDoneEvent {
                    item_id: item_id.clone(),
                    status: "failed".to_string(),
                    error: Some("app state lock poisoned".to_string()),
                    sample_rate: None,
                    duration_seconds: None,
                },
            };
            let _ = window.emit("generation_done", done);
            if let Some(playback_run) = playback_run {
                let _ = window.emit(
                    "playback_state",
                    PlaybackStateEvent {
                        item_id: Some(playback_run.item_id.clone()),
                        state: "playing".to_string(),
                    },
                );
                spawn_playback(window.clone(), shared.clone(), playback_run);
            }
            kick_generation_queue(window, shared);
        }
        SidecarEvent::Error { message } => {
            tracing::error!("sidecar error: {message}");
        }
    }
}

fn operation_status_label(status: OperationStatus) -> &'static str {
    match status {
        OperationStatus::Success => "success",
        OperationStatus::Canceled => "canceled",
        OperationStatus::Failed => "failed",
    }
}

fn synthesis_request_dto(request: voxui_inference::SynthesisRequest) -> SynthesisRequestDto {
    SynthesisRequestDto {
        text: request.text,
        prompt_wav_path: request.prompt_wav_path,
        prompt_text: request.prompt_text,
        reference_wav_path: request.reference_wav_path,
        cfg_value: request.cfg_value,
        inference_timesteps: request.inference_timesteps,
        min_len: request.min_len,
        max_len: request.max_len,
        retry_badcase: request.retry_badcase,
        retry_badcase_max_times: request.retry_badcase_max_times,
        retry_badcase_ratio_threshold: request.retry_badcase_ratio_threshold,
        consolidate_n: request.consolidate_n,
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

fn spawn_playback(window: Window, shared: SharedAppCore, run: crate::app_core::PlaybackRun) {
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
                match AudioPlayer::new(&host, &device, run.audio.sample_rate).and_then(
                    |mut player| {
                        let done = player
                            .play_with_volume_handle(run.audio.samples, run.volume.clone())?;
                        wait_for_playback(done, run.stop, &mut player);
                        Ok(())
                    },
                ) {
                    Ok(()) => "stopped".to_string(),
                    Err(error) => format!("error:{error}"),
                }
            }
            Err(error) => format!("error:{error}"),
        };

        let completion = match worker_shared.lock() {
            Ok(mut core) => core.finish_playback_and_next(&run.item_id),
            Err(_) => crate::app_core::PlaybackCompletion {
                stopped_item_id: None,
                next_run: None,
            },
        };
        let _ = worker_window.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: completion.stopped_item_id,
                state: stop_result,
            },
        );
        if let Some(next_run) = completion.next_run {
            let _ = worker_window.emit(
                "playback_state",
                PlaybackStateEvent {
                    item_id: Some(next_run.item_id.clone()),
                    state: "playing".to_string(),
                },
            );
            spawn_playback(worker_window.clone(), worker_shared.clone(), next_run);
        }
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

fn wait_for_playback(done: mpsc::Receiver<()>, stop: mpsc::Receiver<()>, player: &mut AudioPlayer) {
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

fn spawn_background(
    name: &'static str,
    task: impl FnOnce() + Send + 'static,
) -> Result<(), String> {
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
