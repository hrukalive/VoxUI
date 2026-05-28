use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use tauri::{AppHandle, Emitter, State, Window};
use tauri_plugin_dialog::DialogExt;
use voxui_audio::{AudioPlayer, AudioSystem, StreamingPlayer, VolumeHandle};
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
static STREAMING_PLAYERS: OnceLock<Mutex<HashMap<String, mpsc::Sender<StreamingPlaybackCommand>>>> =
    OnceLock::new();

enum StreamingPlaybackCommand {
    Push(Vec<f32>),
    Finish,
    Stop,
}

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
    tracing::info!(
        load_id,
        choice_id = %choice.id,
        model_dir = %choice.model_dir.display(),
        backend = ?backend,
        "requesting sidecar model load"
    );
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
        stop_streaming_playback(&item_id);
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
    tracing::debug!(
        command = sidecar_command_name(&command),
        "queueing sidecar command"
    );
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
    let target_triple = option_env!("TAURI_ENV_TARGET_TRIPLE").unwrap_or("x86_64-pc-windows-msvc");
    let candidates = [
        dir.join("voxui-inference-sidecar.exe"),
        dir.join(format!("voxui-inference-sidecar-{target_triple}.exe")),
        dir.join("voxui-inference-sidecar"),
        dir.join(format!("voxui-inference-sidecar-{target_triple}")),
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
                    tracing::debug!(
                        event = sidecar_event_name(&frame.header),
                        "received sidecar event"
                    );
                    handle_sidecar_event(&window, shared.clone(), frame);
                }
                SidecarReaderEvent::Error(error) => {
                    tracing::error!("sidecar reader error: {error}");
                    handle_sidecar_exit(&window, shared.clone(), error);
                    break;
                }
                SidecarReaderEvent::Eof => {
                    tracing::warn!("sidecar stdout reached eof");
                    handle_sidecar_exit(
                        &window,
                        shared.clone(),
                        "sidecar exited unexpectedly".to_string(),
                    );
                    break;
                }
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
            load_id,
            phase,
            loaded_bytes,
            total_bytes,
            component,
            component_index,
            component_total,
            ..
        } => {
            let should_emit = shared
                .lock()
                .map(|core| core.active_load_id() == Some(load_id))
                .unwrap_or(false);
            if should_emit {
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
        }
        SidecarEvent::ModelLoadDone {
            load_id,
            status,
            sample_rate,
            error,
        } => {
            tracing::info!(
                load_id,
                status = ?status,
                sample_rate,
                error = error.as_deref().unwrap_or(""),
                "received model load completion"
            );
            let done = match shared.lock() {
                Ok(mut core) => {
                    let accepted = match status {
                        OperationStatus::Success => {
                            if let (Ok(choice), Some(sample_rate)) =
                                (core.selected_choice(), sample_rate)
                            {
                                core.mark_load_success(load_id, choice.id, sample_rate)
                            } else {
                                false
                            }
                        }
                        OperationStatus::Canceled | OperationStatus::Failed => {
                            core.mark_load_finished_without_swap_for_load(load_id)
                        }
                    };
                    if !accepted {
                        return;
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
            let accepted = shared
                .lock()
                .map(|mut core| core.mark_generation_progress(&item_id, current, total))
                .unwrap_or(false);
            if !accepted {
                return;
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
                if !accepts_sidecar_generation_event(&shared, &item_id, false) {
                    return;
                }
                handle_streaming_audio_chunk(window, shared.clone(), item_id, samples, sample_rate);
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
                if !accepts_sidecar_generation_event(&shared, &item_id, false) {
                    return;
                }
                let should_kick = if let Ok(mut core) = shared.lock() {
                    if let Err(error) =
                        core.append_generation_audio_chunk(&item_id, samples, sample_rate)
                    {
                        let _ = core
                            .finish_generation_failure_from_sidecar(&item_id, error.to_string());
                        let _ = window.emit(
                            "generation_done",
                            GenerationDoneEvent {
                                item_id,
                                status: "failed".to_string(),
                                error: Some(error.to_string()),
                                sample_rate: None,
                                duration_seconds: None,
                            },
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if should_kick {
                    kick_generation_queue(window, shared);
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
            let mut accepted = false;
            let mut final_status = status;
            let mut final_error = error.clone();
            let mut final_sample_rate = sample_rate;
            let mut final_duration_seconds = duration_seconds;
            let done = match shared.lock() {
                Ok(mut core) => {
                    if !core.accepts_sidecar_generation_event(&item_id, true) {
                        return;
                    }
                    accepted = true;
                    match final_status {
                        OperationStatus::Success => {
                            if let (Some(rate), Some(duration)) = (sample_rate, duration_seconds) {
                                match core.finish_generation_success_from_sidecar(
                                    &item_id, rate, duration,
                                ) {
                                    Ok(crate::app_core::SidecarGenerationOutcome::Ready) => {
                                        if !finish_streaming_playback(window, item_id.clone()) {
                                            playback_run = core
                                                .begin_or_queue_auto_playback(&item_id)
                                                .ok()
                                                .flatten();
                                        }
                                    }
                                    Ok(crate::app_core::SidecarGenerationOutcome::Canceled) => {
                                        stop_streaming_playback(&item_id);
                                        final_status = OperationStatus::Canceled;
                                        final_error = None;
                                        final_sample_rate = None;
                                        final_duration_seconds = None;
                                    }
                                    Err(error) => {
                                        stop_streaming_playback(&item_id);
                                        let _ = core.finish_generation_failure_from_sidecar(
                                            &item_id,
                                            error.to_string(),
                                        );
                                        final_status = OperationStatus::Failed;
                                        final_error = Some(error.to_string());
                                    }
                                }
                            }
                        }
                        OperationStatus::Canceled => {
                            stop_streaming_playback(&item_id);
                            let _ = core.finish_generation_canceled_from_sidecar(&item_id);
                        }
                        OperationStatus::Failed => {
                            stop_streaming_playback(&item_id);
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
                        status: operation_status_label(final_status).to_string(),
                        error: final_error,
                        sample_rate: final_sample_rate,
                        duration_seconds: final_duration_seconds,
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
            if !accepted {
                return;
            }
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

fn accepts_sidecar_generation_event(shared: &SharedAppCore, item_id: &str, terminal: bool) -> bool {
    shared
        .lock()
        .map(|core| core.accepts_sidecar_generation_event(item_id, terminal))
        .unwrap_or(false)
}

fn handle_sidecar_exit(window: &Window, shared: SharedAppCore, error: String) {
    let (recovery, snapshot) = match shared.lock() {
        Ok(mut core) => {
            let recovery = core.handle_sidecar_exit(error.clone());
            let snapshot = core.snapshot();
            (recovery, snapshot)
        }
        Err(_) => return,
    };

    if recovery.failed_load {
        let _ = window.emit(
            "model_load_done",
            crate::types::ModelLoadDoneEvent {
                status: "failed".to_string(),
                selected_model_id: snapshot.selected_model_id,
                loaded_model_id: snapshot.loaded_model_id,
                error: Some(error.clone()),
            },
        );
    }

    if let Some(item_id) = recovery.stopped_generation_item_id.as_ref() {
        stop_streaming_playback(item_id);
    }

    if let Some(item_id) = recovery.failed_generation_item_id {
        let _ = window.emit(
            "generation_done",
            GenerationDoneEvent {
                item_id,
                status: "failed".to_string(),
                error: Some(error),
                sample_rate: None,
                duration_seconds: None,
            },
        );
    }
}

fn operation_status_label(status: OperationStatus) -> &'static str {
    match status {
        OperationStatus::Success => "success",
        OperationStatus::Canceled => "canceled",
        OperationStatus::Failed => "failed",
    }
}

fn sidecar_command_name(command: &SidecarCommand) -> &'static str {
    match command {
        SidecarCommand::LoadModel { .. } => "load_model",
        SidecarCommand::CancelLoad { .. } => "cancel_load",
        SidecarCommand::Synthesize { .. } => "synthesize",
        SidecarCommand::CancelSynthesis { .. } => "cancel_synthesis",
        SidecarCommand::Shutdown => "shutdown",
    }
}

fn sidecar_event_name(event: &SidecarEvent) -> &'static str {
    match event {
        SidecarEvent::Ready => "ready",
        SidecarEvent::ModelLoadProgress { .. } => "model_load_progress",
        SidecarEvent::ModelLoadDone { .. } => "model_load_done",
        SidecarEvent::GenerationProgress { .. } => "generation_progress",
        SidecarEvent::AudioChunk { .. } => "audio_chunk",
        SidecarEvent::AudioFinal { .. } => "audio_final",
        SidecarEvent::GenerationDone { .. } => "generation_done",
        SidecarEvent::Error { .. } => "error",
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

fn streaming_players() -> &'static Mutex<HashMap<String, mpsc::Sender<StreamingPlaybackCommand>>> {
    STREAMING_PLAYERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn handle_streaming_audio_chunk(
    window: &Window,
    shared: SharedAppCore,
    item_id: String,
    samples: Vec<f32>,
    sample_rate: u32,
) {
    match shared.lock() {
        Ok(mut core) => {
            if core
                .append_generation_audio_chunk(&item_id, samples.clone(), sample_rate)
                .is_err()
            {
                return;
            }
        }
        Err(_) => return,
    }

    let mut players = match streaming_players().lock() {
        Ok(players) => players,
        Err(_) => return,
    };
    if !players.contains_key(&item_id) {
        match start_streaming_playback_worker(window.clone(), shared, item_id.clone(), sample_rate)
        {
            Ok(sender) => {
                players.insert(item_id.clone(), sender);
                let _ = window.emit(
                    "playback_state",
                    PlaybackStateEvent {
                        item_id: Some(item_id.clone()),
                        state: "playing".to_string(),
                    },
                );
            }
            Err(error) => {
                tracing::warn!("failed to start streaming playback for {item_id}: {error}");
                return;
            }
        }
    }

    if let Some(sender) = players.get(&item_id) {
        if let Err(error) = sender.send(StreamingPlaybackCommand::Push(samples)) {
            tracing::warn!("failed to push streaming audio for {item_id}: {error}");
        }
    }
}

fn start_streaming_playback_worker(
    window: Window,
    shared: SharedAppCore,
    item_id: String,
    sample_rate: u32,
) -> Result<mpsc::Sender<StreamingPlaybackCommand>, String> {
    let config = shared
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?
        .snapshot()
        .config;
    let (sender, receiver) = mpsc::channel();
    let worker_item_id = item_id.clone();
    spawn_background("voxui-streaming-playback", move || {
        let state = run_streaming_playback_worker(config, sample_rate, receiver);
        let _ = window.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: Some(worker_item_id),
                state,
            },
        );
    })?;
    Ok(sender)
}

fn run_streaming_playback_worker(
    config: crate::types::AppConfig,
    sample_rate: u32,
    receiver: mpsc::Receiver<StreamingPlaybackCommand>,
) -> String {
    let mut player = match create_streaming_player(config, sample_rate) {
        Ok(player) => player,
        Err(error) => return format!("error:{error}"),
    };

    while let Ok(command) = receiver.recv() {
        match command {
            StreamingPlaybackCommand::Push(samples) => {
                if let Err(error) = player.push(&samples) {
                    return format!("error:{error}");
                }
            }
            StreamingPlaybackCommand::Finish => {
                return match player.finish().and_then(|drain| drain.wait()) {
                    Ok(()) => "stopped".to_string(),
                    Err(error) => format!("error:{error}"),
                };
            }
            StreamingPlaybackCommand::Stop => {
                player.stop();
                return "stopped".to_string();
            }
        }
    }

    player.stop();
    "stopped".to_string()
}

fn create_streaming_player(
    config: crate::types::AppConfig,
    sample_rate: u32,
) -> Result<StreamingPlayer, String> {
    let system = AudioSystem::new();
    let host = config
        .audio_host
        .clone()
        .unwrap_or_else(|| system.default_host_name());
    let devices = crate::audio::list_devices(&system, &host).map_err(|error| error.to_string())?;
    let device = crate::audio::resolve_output_device_name(
        config.audio_device.clone(),
        &devices,
        system.default_device_name(&host),
    )
    .map_err(|error| error.to_string())?;
    StreamingPlayer::new(
        &host,
        &device,
        sample_rate,
        0.25,
        VolumeHandle::new(config.volume),
    )
    .map_err(|error| error.to_string())
}

fn finish_streaming_playback(window: &Window, item_id: String) -> bool {
    let sender = match streaming_players().lock() {
        Ok(mut players) => players.remove(&item_id),
        Err(_) => None,
    };
    let Some(sender) = sender else {
        return false;
    };
    if let Err(error) = sender.send(StreamingPlaybackCommand::Finish) {
        tracing::warn!("failed to finish streaming playback for {item_id}: {error}");
        let _ = window.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: Some(item_id),
                state: "error:streaming playback worker stopped".to_string(),
            },
        );
    }
    true
}

fn stop_streaming_playback(item_id: &str) {
    let sender = match streaming_players().lock() {
        Ok(mut players) => players.remove(item_id),
        Err(_) => None,
    };
    if let Some(sender) = sender {
        let _ = sender.send(StreamingPlaybackCommand::Stop);
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
