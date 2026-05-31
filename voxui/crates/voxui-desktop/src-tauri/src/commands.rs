use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager, State, WebviewWindowBuilder, Window};
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
    GenerationProgressEvent, LiveMessageKind, MainInputReplaceEvent, ModelChoice,
    PlaybackStateEvent, SidecarCapabilities,
};

pub type SharedAppCore = Arc<Mutex<AppCore>>;

static SIDECAR_PROCESS: OnceLock<Mutex<Option<SidecarProcess>>> = OnceLock::new();
static LIVE_WORKER: OnceLock<Mutex<Option<LiveWorkerHandle>>> = OnceLock::new();
static STREAMING_PLAYERS: OnceLock<Mutex<HashMap<String, StreamingPlaybackControl>>> =
    OnceLock::new();
const LIVE_WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

struct LiveWorkerHandle {
    stop: Arc<AtomicBool>,
    done: mpsc::Receiver<()>,
    start: mpsc::Sender<()>,
}

impl LiveWorkerHandle {
    fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    fn request_stop_and_wait(self, timeout: Duration) -> bool {
        self.request_stop();
        match self.done.recv_timeout(timeout) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => true,
            Err(mpsc::RecvTimeoutError::Timeout) => false,
        }
    }

    fn start_worker(&self) -> Result<(), String> {
        self.start
            .send(())
            .map_err(|_| "OpenLive worker stopped before start signal".to_string())
    }
}

enum StreamingPlaybackCommand {
    Push(Vec<f32>),
    Finish,
    Stop,
}

struct StreamingPlaybackControl {
    sender: mpsc::Sender<StreamingPlaybackCommand>,
    volume: VolumeHandle,
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
    app: AppHandle,
    state: State<'_, SharedAppCore>,
    patch: ConfigPatch,
) -> Result<AppSnapshot, String> {
    let updates_volume = patch.volume.is_some();
    let (snapshot, live_snapshot) = with_core(state, |core| {
        let snapshot = core.apply_patch(patch)?;
        let live_snapshot = core.live_snapshot_for_current_language();
        Ok((snapshot, live_snapshot))
    })?;
    if updates_volume {
        set_streaming_playback_volume(snapshot.config.volume);
    }
    let _ = app.emit("app_config_changed", snapshot.clone());
    emit_live_snapshot(&app, "live_items_changed", &live_snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn set_runtime_volume(state: State<'_, SharedAppCore>, volume: f32) -> Result<f32, String> {
    let volume = with_core(state, |core| Ok(core.set_runtime_volume(volume)))?;
    set_streaming_playback_volume(volume);
    Ok(volume)
}

#[tauri::command]
pub fn get_live_state(
    state: State<'_, SharedAppCore>,
) -> Result<crate::types::LiveSnapshot, String> {
    with_core(state, |core| Ok(core.live_snapshot_for_current_language()))
}

#[tauri::command]
pub fn set_live_config_patch(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
    patch: crate::types::LiveConfigPatch,
) -> Result<crate::types::LiveSnapshot, String> {
    let snapshot = with_core(state, |core| core.apply_live_patch(patch))?;
    emit_live_snapshot(&app, "live_items_changed", &snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn clear_live_items(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
) -> Result<crate::types::LiveSnapshot, String> {
    let snapshot = with_core(state, |core| {
        core.clear_live_items();
        Ok(core.live_snapshot_for_current_language())
    })?;
    emit_live_snapshot(&app, "live_items_changed", &snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn mock_live_message(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
    kind: String,
) -> Result<crate::types::LiveSnapshot, String> {
    let kind = match kind.as_str() {
        "danmu" => LiveMessageKind::Danmu,
        "gift" => LiveMessageKind::Gift,
        "superchat" => LiveMessageKind::Superchat,
        "guard" => LiveMessageKind::Guard,
        "like" => LiveMessageKind::Like,
        "enter" => LiveMessageKind::Enter,
        other => return Err(format!("unsupported live message kind: {other}")),
    };
    let event = crate::live::create_mock_live_event(kind).map_err(|e| e.to_string())?;
    let snapshot = with_core(state, |core| {
        core.add_live_event(event)
            .map(|_| core.live_snapshot_for_current_language())
    })?;
    emit_live_snapshot(&app, "live_items_changed", &snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn exit_app(app: AppHandle) -> Result<CommandResult, String> {
    if let Some(window) = app.get_webview_window("main") {
        window.close().map_err(|e| e.to_string())?;
    }
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub async fn show_live_monitor(app: AppHandle) -> Result<CommandResult, String> {
    show_live_monitor_impl(&app)?;
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn connect_openblive(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
    identity_code: String,
) -> Result<crate::types::LiveSnapshot, String> {
    let shared = state.inner().clone();
    let mut guard = live_worker_slot()
        .lock()
        .map_err(|_| "OpenLive worker lock poisoned".to_string())?;
    if guard.is_some() {
        return shared
            .lock()
            .map(|core| core.live_snapshot_for_current_language())
            .map_err(|_| "app state lock poisoned".to_string());
    }

    let (identity_code, enable_ceve_server_heartbeat, snapshot) = {
        let mut core = shared
            .lock()
            .map_err(|_| "app state lock poisoned".to_string())?;
        let identity_code = identity_code.trim().to_string();
        if identity_code.is_empty() {
            return Err("OpenLive identity code is empty".to_string());
        }
        let config = core
            .apply_live_patch(crate::types::LiveConfigPatch {
                identity_code: Some(identity_code),
                ..crate::types::LiveConfigPatch::default()
            })
            .map_err(|error| error.to_string())?
            .config;
        let snapshot = core.set_live_status(crate::types::LiveStatus::Connecting, None);
        (
            config.identity_code.clone(),
            config.enable_ceve_server_heartbeat,
            snapshot,
        )
    };
    emit_live_snapshot(&app, "live_status_changed", &snapshot);

    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let (done_sender, done_receiver) = mpsc::channel();
    let (start_sender, start_receiver) = mpsc::channel();
    let worker_app = app.clone();
    let worker_shared = shared.clone();
    let spawn_result = thread::Builder::new()
        .name("voxui-openblive".to_string())
        .spawn(move || {
            if start_receiver.recv().is_ok() {
                crate::openblive::run_openblive_worker(
                    identity_code,
                    enable_ceve_server_heartbeat,
                    {
                        let app = worker_app.clone();
                        let shared = worker_shared.clone();
                        move |raw| handle_openblive_event(&app, shared.clone(), raw)
                    },
                    {
                        let app = worker_app.clone();
                        let shared = worker_shared.clone();
                        move |status, message| {
                            handle_openblive_status(&app, shared.clone(), status, message)
                        }
                    },
                    move || worker_stop.load(Ordering::SeqCst),
                );
            }
            let _ = done_sender.send(());
            clear_live_worker_slot();
        });
    if let Err(error) = spawn_result {
        let snapshot = match shared.lock() {
            Ok(mut core) => core.set_live_status(
                crate::types::LiveStatus::Error,
                Some(format!("spawn OpenLive worker: {error}")),
            ),
            Err(_) => return Err("app state lock poisoned".to_string()),
        };
        emit_live_snapshot(&app, "live_status_changed", &snapshot);
        return Err(format!("spawn OpenLive worker: {error}"));
    }

    *guard = Some(LiveWorkerHandle {
        stop,
        done: done_receiver,
        start: start_sender,
    });
    if let Some(handle) = guard.as_ref() {
        if let Err(error) = handle.start_worker() {
            *guard = None;
            let snapshot = match shared.lock() {
                Ok(mut core) => {
                    core.set_live_status(crate::types::LiveStatus::Error, Some(error.clone()))
                }
                Err(_) => return Err("app state lock poisoned".to_string()),
            };
            emit_live_snapshot(&app, "live_status_changed", &snapshot);
            return Err(error);
        }
    }

    shared
        .lock()
        .map(|core| core.live_snapshot_for_current_language())
        .map_err(|_| "app state lock poisoned".to_string())
}

#[tauri::command]
pub fn disconnect_openblive(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
) -> Result<crate::types::LiveSnapshot, String> {
    let stopping = signal_live_worker_stop()?;
    let snapshot = with_core(state, |core| {
        let status = if stopping {
            crate::types::LiveStatus::Disconnecting
        } else {
            crate::types::LiveStatus::Disconnected
        };
        Ok(core.set_live_status(status, None))
    })?;
    emit_live_snapshot(&app, "live_status_changed", &snapshot);
    Ok(snapshot)
}

#[tauri::command]
pub fn send_live_suggestion(
    app: AppHandle,
    state: State<'_, SharedAppCore>,
    item_id: String,
    mode: String,
    skip_replace: bool,
) -> Result<crate::types::LiveSuggestionResult, String> {
    let mode = match mode.as_str() {
        "normal" => crate::live::SuggestionMode::Normal,
        "switch" => crate::live::SuggestionMode::Switch,
        other => return Err(format!("unsupported live suggestion mode: {other}")),
    };
    let text = with_core(state, |core| {
        core.live_suggestion_for_item_current_language(&item_id, mode)
            .ok_or_else(|| anyhow::anyhow!("live item is unavailable or filtered: {item_id}"))
    })?;
    if !skip_replace {
        let event = MainInputReplaceEvent { text: text.clone() };
        if let Some(main) = app.get_webview_window("main") {
            main.emit("main_input_replace", event)
                .map_err(|error| error.to_string())?;
        } else {
            app.emit("main_input_replace", event)
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(crate::types::LiveSuggestionResult { text })
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
        backend: protocol_backend(backend),
    };
    if let Err(error) = send_sidecar_command(window.app_handle(), shared, command) {
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
    kick_generation_queue(window.app_handle(), state.inner().clone());
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
    kick_generation_queue(window.app_handle(), state.inner().clone());
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
            window.app_handle(),
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
            window.app_handle(),
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
    spawn_playback(window.app_handle().clone(), state.inner().clone(), run);
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

use wreq::Client;

const ONESHOT_FREE_ENDPOINT: &str = "https://oneshot-free.www.deepl.com/v1/translate";
const EXTENSION_ID: &str = "cofdbpoegempjloogbagkncekinflcnj";
const MAX_FREE_TEXT_LENGTH: usize = 1500;

fn get_translation_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .emulation(wreq_util::Emulation::Chrome120)
            .timeout(Duration::from_secs(20))
            .build()
            .expect("Failed to build wreq client")
    })
}

fn map_lang_code(code: &str, is_source: bool) -> String {
    if is_source && code == "auto" {
        return String::new();
    }
    match code {
        "ZH" => "zh-Hans".to_string(),
        "ZH-HANT" => "zh-Hant".to_string(),
        "EN" => "en-US".to_string(),
        _ => code.to_lowercase(),
    }
}

static COOKIE_WARMED: std::sync::Once = std::sync::Once::new();

#[tauri::command]
pub async fn translate_text(
    text: String,
    source_lang: String,
    target_lang: String,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("No text to translate".to_string());
    }
    if text.chars().count() > MAX_FREE_TEXT_LENGTH {
        return Err(format!(
            "Text too long (max {} characters)",
            MAX_FREE_TEXT_LENGTH
        ));
    }

    let client = get_translation_client();

    COOKIE_WARMED.call_once(|| {
        let client = client.clone();
        tauri::async_runtime::spawn(async move {
            let _ = client
                .get("https://www.deepl.com/translator")
                .timeout(Duration::from_secs(5))
                .send()
                .await;
        });
    });

    let api_source = map_lang_code(&source_lang, true);
    let api_target = map_lang_code(&target_lang, false);

    let body = serde_json::json!({
        "text": [text],
        "target_lang": api_target,
        "source_lang": api_source,
        "usage_type": "Translate",
        "app_information": {
            "os": "brex_macOS",
            "os_version": "brex_chrome_120.0.0.0",
            "app_version": "1.86.0",
            "app_build": "chrome_web_store",
            "instance_id": uuid::Uuid::new_v4().to_string(),
        },
    });

    let response = client
        .post(ONESHOT_FREE_ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Accept", "*/*")
        .header("Authorization", "None")
        .header("Origin", format!("chrome-extension://{EXTENSION_ID}"))
        .header("Sec-Fetch-Site", "cross-site")
        .header("Sec-Fetch-Mode", "cors")
        .header("Sec-Fetch-Dest", "empty")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Translation request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let msg = match status.as_u16() {
            429 => "Translation service rate-limited. Please wait and try again.",
            400 => "Invalid language code or request.",
            404 => "No text provided.",
            _ => return Err(format!("Translation service returned status {status}")),
        };
        return Err(msg.to_string());
    }

    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let translated = json["translations"][0]["text"]
        .as_str()
        .ok_or_else(|| "Unexpected translation response format".to_string())?
        .to_string();

    Ok(translated)
}

fn kick_generation_queue(app: &AppHandle, shared: SharedAppCore) {
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
            if let Err(error) = send_sidecar_command(app, shared.clone(), command) {
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
                let _ = app.emit("generation_done", done);
                kick_generation_queue(app, shared);
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

fn desktop_backend(backend: ProtocolBackendKind) -> BackendKind {
    match backend {
        ProtocolBackendKind::Cpu => BackendKind::Cpu,
        ProtocolBackendKind::Cuda => BackendKind::Cuda,
    }
}

fn sidecar_slot() -> &'static Mutex<Option<SidecarProcess>> {
    SIDECAR_PROCESS.get_or_init(|| Mutex::new(None))
}

fn live_worker_slot() -> &'static Mutex<Option<LiveWorkerHandle>> {
    LIVE_WORKER.get_or_init(|| Mutex::new(None))
}

fn signal_live_worker_stop() -> Result<bool, String> {
    let guard = live_worker_slot()
        .lock()
        .map_err(|_| "OpenLive worker lock poisoned".to_string())?;
    let Some(handle) = guard.as_ref() else {
        return Ok(false);
    };
    handle.request_stop();
    Ok(true)
}

fn clear_live_worker_slot() {
    if let Ok(mut guard) = live_worker_slot().lock() {
        *guard = None;
    }
}

pub fn shutdown_live_worker_for_app_exit() {
    let handle = match live_worker_slot().lock() {
        Ok(mut guard) => guard.take(),
        Err(_) => return,
    };
    if let Some(handle) = handle {
        if !handle.request_stop_and_wait(LIVE_WORKER_SHUTDOWN_TIMEOUT) {
            tracing::warn!(
                timeout_ms = LIVE_WORKER_SHUTDOWN_TIMEOUT.as_millis(),
                "OpenLive worker did not stop before app exit timeout"
            );
        }
    }
}

fn handle_openblive_status(
    app: &AppHandle,
    shared: SharedAppCore,
    status: crate::types::LiveStatus,
    status_message: Option<String>,
) {
    let snapshot = match shared.lock() {
        Ok(mut core) => core.set_live_status(status, status_message),
        Err(_) => return,
    };
    emit_live_snapshot(app, "live_status_changed", &snapshot);

    match status {
        crate::types::LiveStatus::Connected => {
            if let Err(error) = show_live_monitor_impl(app) {
                tracing::warn!("failed to show live monitor window: {error}");
            }
        }
        crate::types::LiveStatus::Connecting
        | crate::types::LiveStatus::Disconnecting
        | crate::types::LiveStatus::Disconnected
        | crate::types::LiveStatus::Error => {}
    }
}

fn handle_openblive_event(app: &AppHandle, shared: SharedAppCore, raw: serde_json::Value) {
    let event = match crate::live::parse_live_event(raw) {
        Ok(Some(event)) => event,
        Ok(None) => return,
        Err(error) => {
            tracing::warn!("failed to parse OpenLive event: {error}");
            return;
        }
    };

    let snapshot = match shared.lock() {
        Ok(mut core) => match core.add_live_event(event) {
            Ok(_) => core.live_snapshot_for_current_language(),
            Err(error) => {
                tracing::warn!("failed to add OpenLive event: {error}");
                return;
            }
        },
        Err(_) => return,
    };
    emit_live_snapshot(app, "live_items_changed", &snapshot);
}

fn emit_live_snapshot(app: &AppHandle, event: &str, snapshot: &crate::types::LiveSnapshot) {
    let _ = app.emit(event, snapshot.clone());
}

fn show_live_monitor_impl(app: &AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("live-monitor") {
        window.show().map_err(|error| error.to_string())?;
        window.set_focus().map_err(|error| error.to_string())?;
        return Ok(());
    }

    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|window| window.label == "live-monitor")
        .cloned()
        .ok_or_else(|| "live-monitor window config is missing".to_string())?;
    let window = WebviewWindowBuilder::from_config(app, &config)
        .map_err(|error| error.to_string())?
        .build()
        .map_err(|error| error.to_string())?;
    window.show().map_err(|error| error.to_string())?;
    window.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn send_sidecar_command(
    app: &AppHandle,
    shared: SharedAppCore,
    command: SidecarCommand,
) -> Result<(), String> {
    tracing::debug!(
        command = sidecar_command_name(&command),
        "queueing sidecar command"
    );
    ensure_sidecar(app, shared)?;
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

pub fn initialize_sidecar(app: &AppHandle, shared: SharedAppCore) -> Result<(), String> {
    ensure_sidecar(app, shared)
}

fn ensure_sidecar(app: &AppHandle, shared: SharedAppCore) -> Result<(), String> {
    let mut guard = sidecar_slot()
        .lock()
        .map_err(|_| "sidecar process lock poisoned".to_string())?;
    if guard.is_some() {
        return Ok(());
    }

    let path = resolve_sidecar_path()?;
    let (process, receiver) = SidecarProcess::spawn(&path).map_err(|error| error.to_string())?;
    spawn_sidecar_reader(app.clone(), shared, receiver)?;
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
    app: AppHandle,
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
                    handle_sidecar_event(&app, shared.clone(), frame);
                }
                SidecarReaderEvent::Error(error) => {
                    tracing::error!("sidecar reader error: {error}");
                    handle_sidecar_exit(&app, shared.clone(), error);
                    break;
                }
                SidecarReaderEvent::Eof => {
                    tracing::warn!("sidecar stdout reached eof");
                    handle_sidecar_exit(
                        &app,
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

fn handle_sidecar_event(app: &AppHandle, shared: SharedAppCore, frame: Frame<SidecarEvent>) {
    match frame.header {
        SidecarEvent::Ready {
            cuda_available,
            default_backend,
        } => {
            let capabilities = SidecarCapabilities {
                cuda_available,
                default_backend: desktop_backend(default_backend),
            };
            tracing::debug!(
                cuda_available = cuda_available,
                "received sidecar capabilities"
            );
            if let Ok(mut core) = shared.lock() {
                core.apply_sidecar_capabilities(capabilities);
            }
            let _ = app.emit("sidecar_capabilities", capabilities);
        }
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
                let _ = app.emit(
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
            let _ = app.emit("model_load_done", done);
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
            let _ = app.emit(
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
                handle_streaming_audio_chunk(app, shared.clone(), item_id, samples, sample_rate);
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
                        let _ = app.emit(
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
                    kick_generation_queue(app, shared);
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
                                        if !finish_streaming_playback(app, item_id.clone()) {
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
            let _ = app.emit("generation_done", done);
            if let Some(playback_run) = playback_run {
                let _ = app.emit(
                    "playback_state",
                    PlaybackStateEvent {
                        item_id: Some(playback_run.item_id.clone()),
                        state: "playing".to_string(),
                    },
                );
                spawn_playback(app.clone(), shared.clone(), playback_run);
            }
            kick_generation_queue(app, shared);
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

fn handle_sidecar_exit(app: &AppHandle, shared: SharedAppCore, error: String) {
    let (recovery, snapshot) = match shared.lock() {
        Ok(mut core) => {
            let recovery = core.handle_sidecar_exit(error.clone());
            let snapshot = core.snapshot();
            (recovery, snapshot)
        }
        Err(_) => return,
    };

    if recovery.failed_load {
        let _ = app.emit(
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
        let _ = app.emit(
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
        SidecarEvent::Ready { .. } => "ready",
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

fn streaming_players() -> &'static Mutex<HashMap<String, StreamingPlaybackControl>> {
    STREAMING_PLAYERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn handle_streaming_audio_chunk(
    app: &AppHandle,
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
        match start_streaming_playback_worker(app.clone(), shared, item_id.clone(), sample_rate) {
            Ok(control) => {
                players.insert(item_id.clone(), control);
                let _ = app.emit(
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

    if let Some(control) = players.get(&item_id) {
        if let Err(error) = control.sender.send(StreamingPlaybackCommand::Push(samples)) {
            tracing::warn!("failed to push streaming audio for {item_id}: {error}");
        }
    }
}

fn start_streaming_playback_worker(
    app: AppHandle,
    shared: SharedAppCore,
    item_id: String,
    sample_rate: u32,
) -> Result<StreamingPlaybackControl, String> {
    let config = shared
        .lock()
        .map_err(|_| "app state lock poisoned".to_string())?
        .snapshot()
        .config;
    let (sender, receiver) = mpsc::channel();
    let volume = VolumeHandle::new(config.volume);
    let worker_volume = volume.clone();
    let worker_item_id = item_id.clone();
    spawn_background("voxui-streaming-playback", move || {
        let state = run_streaming_playback_worker(config, sample_rate, worker_volume, receiver);
        let _ = app.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: Some(worker_item_id),
                state,
            },
        );
    })?;
    Ok(StreamingPlaybackControl { sender, volume })
}

fn run_streaming_playback_worker(
    config: crate::types::AppConfig,
    sample_rate: u32,
    volume: VolumeHandle,
    receiver: mpsc::Receiver<StreamingPlaybackCommand>,
) -> String {
    let mut player = match create_streaming_player(config, sample_rate, volume) {
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
    volume: VolumeHandle,
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
    StreamingPlayer::new(&host, &device, sample_rate, 0.25, volume)
        .map_err(|error| error.to_string())
}

fn set_streaming_playback_volume(volume: f32) {
    if let Ok(players) = streaming_players().lock() {
        for control in players.values() {
            control.volume.set(volume);
        }
    }
}

fn finish_streaming_playback(app: &AppHandle, item_id: String) -> bool {
    let control = match streaming_players().lock() {
        Ok(mut players) => players.remove(&item_id),
        Err(_) => None,
    };
    let Some(control) = control else {
        return false;
    };
    if let Err(error) = control.sender.send(StreamingPlaybackCommand::Finish) {
        tracing::warn!("failed to finish streaming playback for {item_id}: {error}");
        let _ = app.emit(
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
    let control = match streaming_players().lock() {
        Ok(mut players) => players.remove(item_id),
        Err(_) => None,
    };
    if let Some(control) = control {
        let _ = control.sender.send(StreamingPlaybackCommand::Stop);
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

fn spawn_playback(app: AppHandle, shared: SharedAppCore, run: crate::app_core::PlaybackRun) {
    let event_item_id = run.item_id.clone();
    let worker_app = app.clone();
    let worker_shared = shared.clone();
    if let Err(error) = spawn_background("voxui-playback", move || {
        let config = match worker_shared.lock() {
            Ok(core) => core.snapshot().config,
            Err(_) => {
                let _ = worker_app.emit(
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
        let _ = worker_app.emit(
            "playback_state",
            PlaybackStateEvent {
                item_id: completion.stopped_item_id,
                state: stop_result,
            },
        );
        if let Some(next_run) = completion.next_run {
            let _ = worker_app.emit(
                "playback_state",
                PlaybackStateEvent {
                    item_id: Some(next_run.item_id.clone()),
                    state: "playing".to_string(),
                },
            );
            spawn_playback(worker_app.clone(), worker_shared.clone(), next_run);
        }
    }) {
        let item_id = match shared.lock() {
            Ok(mut core) => core.finish_playback(&event_item_id),
            Err(_) => None,
        };
        let _ = app.emit(
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
    use super::{spawn_background, LiveWorkerHandle};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn background_tasks_do_not_require_tokio_runtime() {
        let (sender, receiver) = mpsc::channel();

        spawn_background("voxui-test-background", move || {
            sender.send(42).unwrap();
        })
        .unwrap();

        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).unwrap(), 42);
    }

    #[test]
    fn live_worker_start_gate_blocks_until_slot_registration_signal() {
        let (start_sender, start_receiver) = mpsc::channel();
        let (entered_sender, entered_receiver) = mpsc::channel();

        let worker = std::thread::spawn(move || {
            start_receiver.recv().unwrap();
            entered_sender.send(()).unwrap();
        });

        assert!(entered_receiver
            .recv_timeout(Duration::from_millis(20))
            .is_err());
        start_sender.send(()).unwrap();
        assert_eq!(
            entered_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(())
        );
        worker.join().unwrap();
    }

    #[test]
    fn live_worker_shutdown_wait_is_bounded() {
        let stop = Arc::new(AtomicBool::new(false));
        let (_done_sender, done_receiver) = mpsc::channel();
        let (start_sender, _start_receiver) = mpsc::channel();
        let handle = LiveWorkerHandle {
            stop: stop.clone(),
            done: done_receiver,
            start: start_sender,
        };

        assert!(!handle.request_stop_and_wait(Duration::from_millis(1)));
        assert!(stop.load(Ordering::SeqCst));
    }

    #[test]
    fn live_monitor_window_creation_uses_async_configured_window_path() {
        let source = include_str!("commands.rs").replace("\r\n", "\n");
        let implementation = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("commands implementation source");

        assert!(
            implementation.contains("pub async fn show_live_monitor"),
            "show_live_monitor must be async because creating WebView2 windows from a synchronous command can deadlock on Windows"
        );
        assert!(
            implementation.contains("WebviewWindowBuilder::from_config"),
            "live monitor fallback creation should use the configured live-monitor window"
        );
        assert!(
            !implementation.contains("WebviewWindowBuilder::new(app, \"live-monitor\""),
            "live monitor fallback creation should not dynamically create an ad-hoc WebView2 window"
        );
    }

    #[test]
    fn config_patch_emits_app_config_and_live_snapshot_updates() {
        let source = include_str!("commands.rs").replace("\r\n", "\n");
        let implementation = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("commands implementation source");
        let app_config_changed = ["app", "_config", "_changed"].concat();

        assert!(
            implementation.contains(&app_config_changed),
            "set_config_patch should broadcast app config changes to every window"
        );
        assert!(
            implementation.contains("emit_live_snapshot(&app, \"live_items_changed\", &live_snapshot"),
            "set_config_patch should refresh live snapshots because language and auto-period can change rendered live rows"
        );
    }

    #[test]
    fn streaming_playback_stores_updateable_volume_handles() {
        let source = include_str!("commands.rs").replace("\r\n", "\n");
        let implementation = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("commands implementation source");

        assert!(
            implementation.contains("struct StreamingPlaybackControl"),
            "streaming playback map should store both command sender and shared volume handle"
        );
        assert!(
            implementation.contains("fn set_streaming_playback_volume"),
            "runtime and persisted volume changes should update active streaming players"
        );
        assert!(
            implementation.contains("volume.set(volume)"),
            "streaming volume updates should use the shared VolumeHandle"
        );
    }
}
