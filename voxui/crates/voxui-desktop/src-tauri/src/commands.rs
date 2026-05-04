use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};
use serde::Serialize;
use crate::state::{AppConfig, AppState};
use voxui_inference::{SynthesisRequest, VoxCPMEngine};
use voxui_audio::AudioPlayer;

#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub architecture: String,
    pub sample_rate: u32,
}

#[derive(Serialize, Clone)]
pub struct AudioDeviceList {
    pub hosts: Vec<String>,
    pub devices: Vec<String>,
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[tauri::command]
pub fn list_models() -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            if entry.path().join("manifest.json").exists() {
                if let Some(name) = entry.path().to_str() {
                    models.push(name.replace('\\', "/"));
                }
            }
        }
    }
    models.sort();
    models
}

#[tauri::command]
pub fn list_lora_dirs(model_dir: String) -> Vec<String> {
    let mut dirs = vec!["None".to_string()];
    let path = PathBuf::from(&model_dir);
    if let Ok(entries) = std::fs::read_dir(&path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("lora_") && entry.path().is_dir() {
                if entry.path().join("lora_manifest.json").exists() {
                    dirs.push(name);
                }
            }
        }
    }
    dirs
}

#[tauri::command]
pub fn list_audio_devices(state: State<AppState>) -> AudioDeviceList {
    let hosts: Vec<String> = state.audio_system.hosts().iter().map(|h| h.name.clone()).collect();
    let default_host = state.audio_system.default_host_name();
    let devices = state.audio_system.devices(&default_host)
        .map(|devs| devs.into_iter().map(|d| d.name).collect())
        .unwrap_or_default();
    AudioDeviceList { hosts, devices }
}

#[tauri::command]
pub async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_dir: String,
    backend: String,
) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&model_dir);
    let device = select_device(&backend);

    let engine = tokio::task::spawn_blocking(move || {
        VoxCPMEngine::load(&model_path, device)
    })
    .await
    .map_err(|e| format!("spawn error: {e}"))?
    .map_err(|e| format!("load error: {e}"))?;

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
    };

    *state.engine.lock().unwrap() = Some(engine);
    let _ = app.emit("engine-ready", ());
    Ok(info)
}

#[tauri::command]
pub fn load_lora(state: State<AppState>, lora_dir: String) -> Result<(), String> {
    let mut guard = state.engine.lock().unwrap();
    let engine = guard.as_mut().ok_or("Engine not loaded")?;
    let path = PathBuf::from(&lora_dir);
    engine.load_lora(&path).map_err(|e| format!("{e}"))
}

#[tauri::command]
pub fn unload_lora(state: State<AppState>) -> Result<(), String> {
    let mut guard = state.engine.lock().unwrap();
    let engine = guard.as_mut().ok_or("Engine not loaded")?;
    engine.unload_lora();
    Ok(())
}

#[tauri::command]
pub fn synthesize(
    app: AppHandle,
    state: State<AppState>,
    text: String,
    dit_steps: u32,
    index: u32,
    prompt_wav_path: Option<String>,
    prompt_text: Option<String>,
    reference_wav_path: Option<String>,
) -> Result<(), String> {
    let config = state.config.lock().unwrap().clone();

    // Synthesis (blocking the command — Tauri runs commands on a thread pool)
    let (samples, sample_rate) = {
        let mut guard = state.engine.lock().unwrap();
        let engine = guard.as_mut().ok_or("Engine not loaded")?;
        let app_clone = app.clone();
        let sr = engine.sample_rate();
        let request = SynthesisRequest {
            text,
            prompt_wav_path: prompt_wav_path.map(PathBuf::from),
            prompt_text,
            reference_wav_path: reference_wav_path.map(PathBuf::from),
            inference_timesteps: dit_steps as usize,
            ..SynthesisRequest::default()
        };
        let result = engine.generate(request, |step, total| {
            let _ = app_clone.emit("tts-progress", ProgressPayload {
                step: step as u32,
                total: total as u32,
                index,
            });
        }).map_err(|e| format!("{e}"))?;
        (result, sr)
    };

    // Play audio
    let host = if config.audio_host.is_empty() {
        state.audio_system.default_host_name()
    } else {
        config.audio_host.clone()
    };
    let device_name = if config.audio_device.is_empty() {
        state.audio_system.default_device_name(&host).unwrap_or_default()
    } else {
        config.audio_device.clone()
    };

    let mut player = AudioPlayer::new(&host, &device_name, sample_rate)
        .map_err(|e| format!("Audio init error: {e}"))?;
    player.play_blocking(samples).map_err(|e| format!("Playback error: {e}"))?;

    let _ = app.emit("tts-complete", serde_json::json!({ "index": index }));
    Ok(())
}

#[tauri::command]
pub fn get_config(state: State<AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
pub fn save_config(state: State<AppState>, config: AppConfig) -> Result<(), String> {
    config.save().map_err(|e| format!("{e}"))?;
    *state.config.lock().unwrap() = config;
    Ok(())
}

fn select_device(backend: &str) -> candle_core::Device {
    match backend {
        "CUDA" => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(0).unwrap_or_else(|e| {
                    log::warn!("CUDA init failed: {e}, falling back to CPU");
                    candle_core::Device::Cpu
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                log::warn!("CUDA not compiled in, using CPU");
                candle_core::Device::Cpu
            }
        }
        _ => candle_core::Device::Cpu,
    }
}
