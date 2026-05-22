use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_inference::VoxCPMEngine;

use tauri_plugin_dialog::DialogExt;

use crate::desktop_core::{
    discover_models_root, scan_lora_entries, scan_model_choices, scan_model_entries, LoraEntry,
    ModelChoice, ModelEntry, SynthesisArgs,
};
use crate::state::{default_program_models_dir, AppConfig, AppState};

#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub architecture: String,
    pub sample_rate: u32,
    pub backend: String,
    pub warning: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct AudioDeviceList {
    pub hosts: Vec<String>,
    pub selected_host: String,
    pub devices: Vec<String>,
    pub selected_device: String,
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[derive(Clone, Serialize)]
struct ErrorPayload {
    index: u32,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApplyLoraArgs {
    pub lora_dir: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadModelChoiceArgs {
    pub choice_id: String,
    pub model_dir: String,
    pub model_path: String,
    pub lora_path: Option<String>,
    pub backend: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LoadProgressPayload {
    pub phase: String,
    pub file_label: Option<String>,
    pub bytes_read: u64,
    pub total_bytes: u64,
    pub backend: Option<String>,
}

#[tauri::command]
pub fn list_models() -> Vec<ModelEntry> {
    let root = discover_models_root();
    tracing::debug!("list_models models_root={}", root.display());
    let entries = scan_model_entries(&root);
    tracing::debug!("list_models found {} model(s)", entries.len());
    for entry in &entries {
        tracing::debug!("list_models entry name={} path={}", entry.name, entry.path);
    }
    entries
}

#[tauri::command(rename_all = "snake_case")]
pub fn list_model_choices(model_root: String) -> Vec<ModelChoice> {
    let root = if model_root.trim().is_empty() {
        default_program_models_dir().unwrap_or_else(|| PathBuf::from("models"))
    } else {
        PathBuf::from(model_root)
    };
    tracing::debug!("list_model_choices model_root={}", root.display());
    scan_model_choices(&root)
}

#[tauri::command(rename_all = "snake_case")]
pub fn list_lora_dirs(model_dir: String) -> Vec<LoraEntry> {
    scan_lora_entries(&PathBuf::from(model_dir))
}

#[tauri::command(rename_all = "snake_case")]
pub fn list_audio_devices(state: State<AppState>, host: Option<String>) -> AudioDeviceList {
    let hosts: Vec<String> = state
        .audio_system
        .hosts()
        .iter()
        .map(|h| h.name.clone())
        .collect();
    let selected_host = host
        .filter(|name| hosts.iter().any(|known| known == name))
        .unwrap_or_else(|| state.audio_system.default_host_name());
    let devices = state
        .audio_system
        .devices(&selected_host)
        .map(|devs| {
            devs.into_iter()
                .map(|device| device.name)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let selected_device = state
        .audio_system
        .default_device_name(&selected_host)
        .ok()
        .filter(|device| devices.iter().any(|known| known == device))
        .or_else(|| devices.first().cloned())
        .unwrap_or_default();

    AudioDeviceList {
        hosts,
        selected_host,
        devices,
        selected_device,
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_dir: String,
    backend: String,
) -> Result<ModelInfo, String> {
    tracing::debug!("load_model requested model_dir={model_dir} backend={backend}");
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => {
            let message = engine_busy_message();
            tracing::warn!(
                "load_model rejected busy model_dir={model_dir} backend={backend} error={message}"
            );
            return Err(message);
        }
    };
    // Reset cancel flag for this load
    state.cancel_load.store(false, Ordering::Release);
    let cancel_token = Arc::clone(&state.cancel_load);

    let started_at = Instant::now();
    let model_path = PathBuf::from(&model_dir);
    let (device, actual_backend, warning) = select_device(&backend);
    if let Some(message) = warning.as_ref() {
        tracing::warn!(
            "load_model backend warning requested_backend={backend} actual_backend={actual_backend} warning={message}"
        );
    } else {
        tracing::debug!(
            "load_model backend selected requested_backend={backend} actual_backend={actual_backend}"
        );
    }
    let engine_slot = Arc::clone(&state.engine);
    let app_for_progress = app.clone();

    let engine = match tokio::task::spawn_blocking(move || {
        VoxCPMEngine::load_with_progress(
            &model_path,
            device,
            |step, total| {
                let _ = app_for_progress.emit(
                    "load-progress",
                    serde_json::json!({
                        "step": step,
                        "total": total,
                    }),
                );
            },
            Some(&cancel_token),
        )
    })
    .await
    {
        Ok(Ok(engine)) => engine,
        Ok(Err(err)) => {
            let msg = format!("{err:#}");
            if msg.contains("cancelled") {
                tracing::info!("load_model cancelled model_dir={model_dir}");
            } else {
                tracing::error!(
                    "load_model failed model_dir={model_dir} backend={actual_backend} elapsed_seconds={:.3} error={err:#}",
                    started_at.elapsed().as_secs_f64()
                );
            }
            return Err(format!("model load failed: {err}"));
        }
        Err(err) => {
            tracing::error!(
                "load_model task failed model_dir={model_dir} backend={actual_backend} elapsed_seconds={:.3} error={err}",
                started_at.elapsed().as_secs_f64()
            );
            return Err(format!("model load task failed: {err}"));
        }
    };

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
        backend: actual_backend,
        warning,
    };
    tracing::debug!(
        "load_model complete architecture={} sample_rate={} backend={} elapsed_seconds={:.3}",
        info.architecture,
        info.sample_rate,
        info.backend,
        started_at.elapsed().as_secs_f64()
    );

    *engine_slot
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())? = Some(engine);
    let _ = app.emit("engine-ready", info.clone());
    drop(_busy);
    Ok(info)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn load_model_choice(
    app: AppHandle,
    state: State<'_, AppState>,
    args: LoadModelChoiceArgs,
) -> Result<ModelInfo, String> {
    tracing::debug!(
        "load_model_choice requested choice_id={} model_dir={} lora_path={:?} backend={}",
        args.choice_id,
        args.model_dir,
        args.lora_path,
        args.backend
    );
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => return Err(engine_busy_message()),
    };
    state.cancel_load.store(false, Ordering::Release);
    let cancel_token = Arc::clone(&state.cancel_load);

    let started_at = Instant::now();
    let model_dir = PathBuf::from(&args.model_dir);
    let (model_path, lora_path) = load_paths_from_args(&args)?;
    let requested_backend = args.backend.clone();
    let choice_id = args.choice_id.clone();
    let (device, actual_backend, warning) = select_device(&requested_backend);
    let actual_backend_for_task = actual_backend.clone();
    let engine_slot = Arc::clone(&state.engine);
    let app_for_task = app.clone();

    let engine = match tokio::task::spawn_blocking(move || {
        read_file_for_progress(&app_for_task, &model_path, "model.gguf", &cancel_token)?;
        if let Some(path) = lora_path.as_ref() {
            let label = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("LoRA GGUF");
            read_file_for_progress(&app_for_task, path, label, &cancel_token)?;
        }
        emit_device_loading(&app_for_task, &actual_backend_for_task);
        if cancel_token.load(Ordering::Relaxed) {
            return Err("model loading cancelled".to_string());
        }

        let mut engine =
            VoxCPMEngine::load_with_progress(&model_dir, device, |_, _| {}, Some(&cancel_token))
                .map_err(|err| format!("{err:#}"))?;
        if let Some(path) = lora_path.as_ref() {
            if cancel_token.load(Ordering::Relaxed) {
                return Err("model loading cancelled".to_string());
            }
            engine
                .load_lora(path)
                .map_err(|err| format!("LoRA load failed: {err:#}"))?;
            if cancel_token.load(Ordering::Relaxed) {
                return Err("model loading cancelled".to_string());
            }
        }
        Ok::<_, String>(engine)
    })
    .await
    {
        Ok(Ok(engine)) => engine,
        Ok(Err(message)) => return Err(format!("model load failed: {message}")),
        Err(err) => return Err(format!("model load task failed: {err}")),
    };

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
        backend: actual_backend,
        warning,
    };

    *engine_slot
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())? = Some(engine);
    tracing::debug!(
        "load_model_choice complete choice_id={} elapsed_seconds={:.3}",
        choice_id,
        started_at.elapsed().as_secs_f64()
    );
    let _ = app.emit("engine-ready", info.clone());
    Ok(info)
}

#[tauri::command(rename_all = "snake_case")]
pub fn apply_lora(state: State<AppState>, args: ApplyLoraArgs) -> Result<(), String> {
    let requested_lora = args.lora_dir.clone();
    tracing::debug!("apply_lora requested lora_dir={requested_lora:?}");
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => {
            let message = engine_busy_message();
            tracing::warn!("apply_lora rejected busy lora_dir={requested_lora:?} error={message}");
            return Err(message);
        }
    };
    let mut guard = match state.engine.lock() {
        Ok(guard) => guard,
        Err(_) => {
            tracing::error!(
                "apply_lora failed lora_dir={requested_lora:?} error=engine lock poisoned"
            );
            return Err("engine lock poisoned".to_string());
        }
    };
    let engine = match guard.as_mut() {
        Some(engine) => engine,
        None => {
            tracing::error!(
                "apply_lora failed lora_dir={requested_lora:?} error=Engine not loaded"
            );
            return Err("Engine not loaded".to_string());
        }
    };

    match args.lora_dir {
        Some(path) if !path.trim().is_empty() => {
            let trimmed = path.trim();
            match engine.load_lora(&PathBuf::from(trimmed)) {
                Ok(()) => {
                    tracing::debug!("apply_lora complete lora_dir={trimmed}");
                    Ok(())
                }
                Err(err) => {
                    tracing::error!("apply_lora failed lora_dir={trimmed} error={err:#}");
                    Err(format!("LoRA load failed: {err}"))
                }
            }
        }
        _ => {
            engine.unload_lora();
            tracing::debug!("apply_lora complete lora_dir=None");
            Ok(())
        }
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn synthesize(
    app: AppHandle,
    state: State<'_, AppState>,
    args: SynthesisArgs,
) -> Result<(), String> {
    let index = args.index;
    tracing::debug!("synthesize requested index={index}");
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(message) => {
            tracing::warn!("synthesize rejected busy index={index} error={message}");
            return emit_synthesis_error(&app, index, message);
        }
    };
    let request = args.into_request();
    let config = match state.config.lock() {
        Ok(config) => config.clone(),
        Err(_) => {
            return emit_synthesis_error(&app, index, "config lock poisoned".to_string());
        }
    };
    // Reset cancel flag for this synthesis
    state.cancel_synthesis.store(false, Ordering::Release);
    let cancel_token = Arc::clone(&state.cancel_synthesis);
    let engine_slot = Arc::clone(&state.engine);
    let app_for_task = app.clone();

    let task = tokio::task::spawn_blocking(move || {
        let (host, device_name) = resolve_audio_output(&config)?;
        let (samples, sample_rate) = {
            let mut guard = engine_slot
                .lock()
                .map_err(|_| "engine lock poisoned".to_string())?;
            let engine = guard.as_mut().ok_or("Engine not loaded")?;
            let sample_rate = engine.sample_rate();
            let generated = engine
                .generate_cancellable(
                    request,
                    |step, total| {
                        if cancel_token.load(Ordering::Relaxed) {
                            return;
                        }
                        let _ = app_for_task.emit(
                            "tts-progress",
                            ProgressPayload {
                                step: step as u32,
                                total: total as u32,
                                index,
                            },
                        );
                    },
                    Some(&cancel_token),
                )
                .map_err(|e| format!("generation failed: {e}"))?;
            if cancel_token.load(Ordering::Relaxed) {
                return Err("synthesis cancelled".to_string());
            }
            (generated, sample_rate)
        };

        let mut player = AudioPlayer::new(&host, &device_name, sample_rate)
            .map_err(|e| format!("audio init failed: {e}"))?;
        player
            .play_blocking(samples)
            .map_err(|e| format!("playback failed: {e}"))?;
        Ok(())
    })
    .await;

    let result: Result<(), String> = match task {
        Ok(result) => result,
        Err(err) => {
            return emit_synthesis_error(&app, index, format!("synthesis task failed: {err}"));
        }
    };

    match result {
        Ok(()) => {
            let _ = app.emit("tts-complete", serde_json::json!({ "index": index }));
            Ok(())
        }
        Err(message) => emit_synthesis_error(&app, index, message),
    }
}

#[tauri::command]
pub fn get_config(state: State<AppState>) -> AppConfig {
    state
        .config
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

#[tauri::command(rename_all = "snake_case")]
pub fn save_config(state: State<AppState>, config: serde_json::Value) -> Result<(), String> {
    let mut guard = state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())?;
    let next = AppConfig::from_save_value_preserving(&guard, config).map_err(|e| format!("{e}"))?;
    next.save().map_err(|e| format!("{e}"))?;
    *guard = next;
    Ok(())
}

#[tauri::command]
pub async fn browse_model_root(app: AppHandle) -> Result<Option<String>, String> {
    tokio::task::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title("Select models folder")
            .blocking_pick_folder()
            .map(|path| {
                path.into_path()
                    .map(|path| path.to_string_lossy().replace('\\', "/"))
                    .map_err(|err| format!("selected folder is not a filesystem path: {err}"))
            })
            .transpose()
    })
    .await
    .map_err(|err| format!("folder browser task failed: {err}"))?
}

#[tauri::command(rename_all = "snake_case")]
pub async fn test_audio_device(host: String, device: String) -> Result<(), String> {
    tracing::debug!("test_audio_device requested host={host} device={device}");
    tokio::task::spawn_blocking(move || {
        let sample_rate = 48000u32;
        let duration_secs = 0.5f32;
        let freq = 440.0f32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Sine wave with fade-in/fade-out to avoid clicks
                let envelope = if t < 0.005 {
                    t / 0.005
                } else if t > duration_secs - 0.005 {
                    (duration_secs - t) / 0.005
                } else {
                    1.0
                };
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3 * envelope
            })
            .collect();

        let mut player = AudioPlayer::new(&host, &device, sample_rate)
            .map_err(|e| format!("audio init failed: {e}"))?;
        player
            .play_blocking(samples)
            .map_err(|e| format!("playback failed: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("test task failed: {e}"))?
}

#[tauri::command]
pub fn cancel_load(state: State<AppState>) {
    state.cancel_load.store(true, Ordering::Release);
    tracing::info!("cancel_load requested");
}

#[tauri::command]
pub fn cancel_synthesis(state: State<AppState>) {
    state.cancel_synthesis.store(true, Ordering::Release);
    tracing::info!("cancel_synthesis requested");
}

fn resolve_audio_output(config: &AppConfig) -> Result<(String, String), String> {
    let audio_system = AudioSystem::new();
    let hosts = audio_system
        .hosts()
        .iter()
        .map(|host| host.name.clone())
        .collect::<Vec<_>>();
    let host = if !config.audio_host.trim().is_empty()
        && hosts.iter().any(|host| host == &config.audio_host)
    {
        config.audio_host.clone()
    } else {
        audio_system.default_host_name()
    };

    let devices = audio_system
        .devices(&host)
        .map_err(|e| format!("audio device lookup failed for host {host}: {e}"))?
        .into_iter()
        .map(|device| device.name)
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(format!("no output devices found for audio host {host}"));
    }

    let device = if !config.audio_device.trim().is_empty()
        && devices.iter().any(|device| device == &config.audio_device)
    {
        config.audio_device.clone()
    } else {
        audio_system
            .default_device_name(&host)
            .ok()
            .filter(|device| devices.iter().any(|known| known == device))
            .or_else(|| devices.first().cloned())
            .ok_or_else(|| format!("no output devices found for audio host {host}"))?
    };
    Ok((host, device))
}

fn emit_synthesis_error(app: &AppHandle, index: u32, message: String) -> Result<(), String> {
    if is_busy_message(&message) {
        tracing::warn!("emit_synthesis_error index={index} message={message}");
    } else {
        tracing::error!("emit_synthesis_error index={index} message={message}");
    }
    let _ = app.emit(
        "tts-error",
        ErrorPayload {
            index,
            message: message.clone(),
        },
    );
    Err(message)
}

fn emit_read_progress(app: &AppHandle, file_label: &str, bytes_read: u64, total_bytes: u64) {
    let _ = app.emit(
        "load-progress",
        LoadProgressPayload {
            phase: "reading".to_string(),
            file_label: Some(file_label.to_string()),
            bytes_read,
            total_bytes,
            backend: None,
        },
    );
}

fn canonical_model_path(model_dir: &Path) -> PathBuf {
    model_dir.join("model.gguf")
}

fn load_paths_from_args(args: &LoadModelChoiceArgs) -> Result<(PathBuf, Option<PathBuf>), String> {
    let model_dir = PathBuf::from(&args.model_dir);
    let _ignored_payload_model_path = &args.model_path;
    let model_path = canonical_model_path(&model_dir);
    let lora_path = validate_lora_path_for_model(&model_dir, args.lora_path.as_deref())?;
    Ok((model_path, lora_path))
}

fn validate_lora_path_for_model(
    model_dir: &Path,
    lora_path: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let Some(lora_path) = lora_path.filter(|value| !value.trim().is_empty()) else {
        return Ok(None);
    };

    let model_dir = model_dir.canonicalize().map_err(|err| {
        format!(
            "failed to canonicalize model dir {}: {err}",
            model_dir.display()
        )
    })?;
    let lora_path = PathBuf::from(lora_path.trim());
    let canonical_lora_path = lora_path.canonicalize().map_err(|err| {
        format!(
            "failed to canonicalize LoRA path {}: {err}",
            lora_path.display()
        )
    })?;
    let parent = canonical_lora_path.parent().ok_or_else(|| {
        format!(
            "LoRA path has no parent directory: {}",
            canonical_lora_path.display()
        )
    })?;
    if parent != model_dir {
        return Err(format!(
            "LoRA file must be a direct file under model dir {}",
            model_dir.display()
        ));
    }
    if !canonical_lora_path.is_file() {
        return Err(format!(
            "LoRA path must be a file: {}",
            canonical_lora_path.display()
        ));
    }
    let file_name = canonical_lora_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            format!(
                "LoRA path has invalid filename: {}",
                canonical_lora_path.display()
            )
        })?;
    if !file_name.starts_with("lora_") {
        return Err(format!("LoRA filename must start with lora_: {file_name}"));
    }
    if canonical_lora_path
        .extension()
        .and_then(|value| value.to_str())
        != Some("gguf")
    {
        return Err(format!("LoRA file must have .gguf extension: {file_name}"));
    }

    Ok(Some(canonical_lora_path))
}

fn read_file_for_progress(
    app: &AppHandle,
    path: &PathBuf,
    file_label: &str,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<(), String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let total = file
        .metadata()
        .map_err(|err| format!("failed to read metadata for {}: {err}", path.display()))?
        .len();
    let mut read = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    emit_read_progress(app, file_label, 0, total);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("model loading cancelled".to_string());
        }
        let count = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        read += count as u64;
        emit_read_progress(app, file_label, read.min(total), total);
    }
    Ok(())
}

fn emit_device_loading(app: &AppHandle, backend: &str) {
    let _ = app.emit(
        "load-progress",
        LoadProgressPayload {
            phase: "device_loading".to_string(),
            file_label: None,
            bytes_read: 0,
            total_bytes: 0,
            backend: Some(backend.to_string()),
        },
    );
}

fn is_busy_message(message: &str) -> bool {
    message == engine_busy_message() || message == "Synthesis already in progress"
}

fn engine_busy_message() -> String {
    "engine is busy; wait for the current synthesis to finish".to_string()
}

fn select_device(requested: &str) -> (candle_core::Device, String, Option<String>) {
    match requested {
        "CUDA" => select_cuda_device(),
        _ => (candle_core::Device::Cpu, "CPU".to_string(), None),
    }
}

#[cfg(feature = "cuda")]
fn select_cuda_device() -> (candle_core::Device, String, Option<String>) {
    match candle_core::Device::new_cuda(0) {
        Ok(device) => (device, "CUDA".to_string(), None),
        Err(err) => (
            candle_core::Device::Cpu,
            "CPU".to_string(),
            Some(format!("CUDA unavailable, using CPU: {err}")),
        ),
    }
}

#[cfg(not(feature = "cuda"))]
fn select_cuda_device() -> (candle_core::Device, String, Option<String>) {
    (
        candle_core::Device::Cpu,
        "CPU".to_string(),
        Some("CUDA was requested, but this build was compiled without CUDA support".to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_model_path, engine_busy_message, is_busy_message, load_paths_from_args,
        validate_lora_path_for_model, LoadModelChoiceArgs,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn classifies_busy_messages() {
        assert!(is_busy_message(&engine_busy_message()));
        assert!(is_busy_message("Synthesis already in progress"));
        assert!(!is_busy_message("generation failed: missing tensor"));
    }

    #[test]
    fn validates_lora_path_is_direct_lora_gguf_under_model_dir() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model-a");
        fs::create_dir_all(&model_dir).unwrap();
        let lora_path = model_dir.join("lora_ft2.gguf");
        fs::write(&lora_path, b"placeholder").unwrap();

        assert_eq!(
            validate_lora_path_for_model(&model_dir, None).unwrap(),
            None
        );
        assert_eq!(
            validate_lora_path_for_model(&model_dir, Some(lora_path.to_str().unwrap())).unwrap(),
            Some(lora_path.canonicalize().unwrap())
        );
    }

    #[test]
    fn rejects_lora_paths_outside_model_dir_or_wrong_shape() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model-a");
        let sibling = tmp.path().join("sibling");
        let subdir = model_dir.join("subdir");
        fs::create_dir_all(&subdir).unwrap();
        fs::create_dir_all(&sibling).unwrap();
        let not_lora = model_dir.join("not_lora.gguf");
        let wrong_extension = model_dir.join("lora_ft2.bin");
        let nested = subdir.join("lora_ft2.gguf");
        let outside = sibling.join("lora_ft2.gguf");
        for path in [&not_lora, &wrong_extension, &nested, &outside] {
            fs::write(path, b"placeholder").unwrap();
        }

        for rejected in [&not_lora, &wrong_extension, &nested, &outside] {
            assert!(
                validate_lora_path_for_model(&model_dir, Some(rejected.to_str().unwrap())).is_err(),
                "expected {} to be rejected",
                rejected.display()
            );
        }
    }

    #[test]
    fn load_paths_are_derived_from_model_dir_not_payload_model_path() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model-a");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"placeholder").unwrap();
        let payload_model_path = tmp.path().join("sibling").join("model.gguf");
        let args = LoadModelChoiceArgs {
            choice_id: "model-a".to_string(),
            model_dir: model_dir.to_string_lossy().into_owned(),
            model_path: payload_model_path.to_string_lossy().into_owned(),
            lora_path: None,
            backend: "CPU".to_string(),
        };

        let (model_path, lora_path) = load_paths_from_args(&args).unwrap();

        assert_eq!(model_path, canonical_model_path(&model_dir));
        assert_ne!(model_path, payload_model_path);
        assert_eq!(lora_path, None);
    }
}
