use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_inference::VoxCPMEngine;

use crate::desktop_core::{
    discover_models_root, scan_lora_entries, scan_model_entries, LoraEntry, ModelEntry,
    SynthesisArgs,
};
use crate::state::{AppConfig, AppState};

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

#[tauri::command]
pub fn list_models() -> Vec<ModelEntry> {
    scan_model_entries(&discover_models_root())
}

#[tauri::command]
pub fn list_lora_dirs(model_dir: String) -> Vec<LoraEntry> {
    scan_lora_entries(&PathBuf::from(model_dir))
}

#[tauri::command]
pub fn list_audio_devices(state: State<AppState>, host: Option<String>) -> AudioDeviceList {
    let hosts: Vec<String> = state.audio_system.hosts().iter().map(|h| h.name.clone()).collect();
    let selected_host = host
        .filter(|name| hosts.iter().any(|known| known == name))
        .unwrap_or_else(|| state.audio_system.default_host_name());
    let devices = state
        .audio_system
        .devices(&selected_host)
        .map(|devs| devs.into_iter().map(|device| device.name).collect::<Vec<_>>())
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

#[tauri::command]
pub async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_dir: String,
    backend: String,
) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&model_dir);
    let (device, actual_backend, warning) = select_device(&backend);
    let engine_slot = Arc::clone(&state.engine);

    let engine = tokio::task::spawn_blocking(move || VoxCPMEngine::load(&model_path, device))
        .await
        .map_err(|e| format!("model load task failed: {e}"))?
        .map_err(|e| format!("model load failed: {e}"))?;

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
        backend: actual_backend,
        warning,
    };

    *engine_slot.lock().map_err(|_| "engine lock poisoned".to_string())? = Some(engine);
    let _ = app.emit("engine-ready", info.clone());
    Ok(info)
}

#[tauri::command]
pub fn apply_lora(state: State<AppState>, args: ApplyLoraArgs) -> Result<(), String> {
    let mut guard = state
        .engine
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())?;
    let engine = guard.as_mut().ok_or("Engine not loaded")?;

    match args.lora_dir {
        Some(path) if !path.trim().is_empty() => engine
            .load_lora(&PathBuf::from(path.trim()))
            .map_err(|e| format!("LoRA load failed: {e}")),
        _ => {
            engine.unload_lora();
            Ok(())
        }
    }
}

#[tauri::command]
pub async fn synthesize(
    app: AppHandle,
    state: State<'_, AppState>,
    args: SynthesisArgs,
) -> Result<(), String> {
    let _busy = state.try_begin_synthesis()?;
    let index = args.index;
    let request = args.into_request();
    let config = state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())?
        .clone();
    let engine_slot = Arc::clone(&state.engine);
    let app_for_task = app.clone();

    let result: Result<(), String> = tokio::task::spawn_blocking(move || {
        let (host, device_name) = resolve_audio_output(&config)?;
        let (samples, sample_rate) = {
            let mut guard = engine_slot
                .lock()
                .map_err(|_| "engine lock poisoned".to_string())?;
            let engine = guard.as_mut().ok_or("Engine not loaded")?;
            let sample_rate = engine.sample_rate();
            let generated = engine
                .generate(request, |step, total| {
                    let _ = app_for_task.emit(
                        "tts-progress",
                        ProgressPayload {
                            step: step as u32,
                            total: total as u32,
                            index,
                        },
                    );
                })
                .map_err(|e| format!("generation failed: {e}"))?;
            (generated, sample_rate)
        };

        let mut player = AudioPlayer::new(&host, &device_name, sample_rate)
            .map_err(|e| format!("audio init failed: {e}"))?;
        player
            .play_blocking(samples)
            .map_err(|e| format!("playback failed: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("synthesis task failed: {e}"))?;

    match result {
        Ok(()) => {
            let _ = app.emit("tts-complete", serde_json::json!({ "index": index }));
            Ok(())
        }
        Err(message) => {
            let _ = app.emit(
                "tts-error",
                ErrorPayload {
                    index,
                    message: message.clone(),
                },
            );
            Err(message)
        }
    }
}

#[tauri::command]
pub fn get_config(state: State<AppState>) -> AppConfig {
    state.config.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

#[tauri::command]
pub fn save_config(state: State<AppState>, config: AppConfig) -> Result<(), String> {
    config.save().map_err(|e| format!("{e}"))?;
    *state.config.lock().map_err(|_| "config lock poisoned".to_string())? = config;
    Ok(())
}

fn resolve_audio_output(config: &AppConfig) -> Result<(String, String), String> {
    let audio_system = AudioSystem::new();
    let host = if config.audio_host.trim().is_empty() {
        audio_system.default_host_name()
    } else {
        config.audio_host.clone()
    };
    let device = if config.audio_device.trim().is_empty() {
        audio_system
            .default_device_name(&host)
            .map_err(|e| format!("default audio device lookup failed: {e}"))?
    } else {
        config.audio_device.clone()
    };
    Ok((host, device))
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
