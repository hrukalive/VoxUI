use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::components::*;
use crate::i18n::Language;
use crate::tauri_api;

#[derive(Clone, Debug)]
pub struct TtsEntry {
    pub timestamp: String,
    pub text: String,
    pub status: String,
    pub progress: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LoraEntry {
    pub name: String,
    pub path: Option<String>,
}

#[derive(Serialize)]
struct SynthesizeArgs {
    args: SynthesisPayload,
}

#[derive(Serialize)]
struct SynthesisPayload {
    index: u32,
    text: String,
    dit_steps: usize,
    prompt_wav_path: Option<String>,
    prompt_text: Option<String>,
    reference_wav_path: Option<String>,
}

#[derive(Serialize)]
struct LoadModelArgs {
    model_dir: String,
    backend: String,
}

#[derive(Serialize)]
struct ListLoraArgs {
    model_dir: String,
}

#[derive(Serialize)]
struct ListAudioDevicesArgs {
    host: Option<String>,
}

#[derive(Serialize)]
struct ApplyLoraArgs {
    args: ApplyLoraPayload,
}

#[derive(Serialize)]
struct ApplyLoraPayload {
    lora_dir: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub model_dir: String,
    pub lora_dir: Option<String>,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    pub dit_steps: usize,
    pub language: String,
}

#[derive(Deserialize, Debug)]
struct ModelInfo {
    architecture: String,
    sample_rate: u32,
    backend: String,
    warning: Option<String>,
}

#[derive(Deserialize, Debug)]
struct AudioDeviceList {
    hosts: Vec<String>,
    selected_host: String,
    devices: Vec<String>,
    selected_device: String,
}

#[derive(Deserialize, Debug)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[derive(Deserialize, Debug)]
struct CompletePayload {
    index: u32,
}

#[derive(Deserialize, Debug)]
struct ErrorPayload {
    index: u32,
    message: String,
}

fn non_empty_option(value: String) -> Option<String> {
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn lora_selection_option(value: String) -> Option<String> {
    if value.trim() == "None" {
        None
    } else {
        non_empty_option(value)
    }
}

#[cfg(debug_assertions)]
fn debug_log(message: &str) {
    web_sys::console::debug_1(&JsValue::from_str(message));
}

#[cfg(not(debug_assertions))]
fn debug_log(_message: &str) {}

fn truncate_to_chars(value: &str, max_chars: usize) -> String {
    match value.char_indices().nth(max_chars) {
        Some((byte_index, _)) => value[..byte_index].to_string(),
        None => value.to_string(),
    }
}

fn valid_lora_selection(selection: Option<String>, entries: &[LoraEntry]) -> Option<String> {
    selection.filter(|selected| {
        entries
            .iter()
            .any(|entry| entry.path.as_ref() == Some(selected))
    })
}

fn selected_audio_device(
    configured: String,
    devices: &[String],
    backend_selected: String,
) -> String {
    let configured = configured.trim().to_string();
    if !configured.is_empty() && devices.iter().any(|device| device == &configured) {
        configured
    } else if !backend_selected.trim().is_empty() {
        backend_selected
    } else {
        devices.first().cloned().unwrap_or_default()
    }
}

async fn list_loras_for_model(model_dir: String) -> Result<Vec<LoraEntry>, String> {
    tauri_api::invoke::<_, Vec<LoraEntry>>("list_lora_dirs", &ListLoraArgs { model_dir }).await
}

async fn apply_lora_selection(lora_dir: Option<String>) -> Result<(), String> {
    tauri_api::invoke_unit(
        "apply_lora",
        &ApplyLoraArgs {
            args: ApplyLoraPayload { lora_dir },
        },
    )
    .await
}

#[component]
pub fn App() -> impl IntoView {
    let (lang, set_lang) = signal(Language::Chinese);
    let (history, set_history) = signal(Vec::<TtsEntry>::new());
    let (progress, set_progress) = signal(0.0f64);
    let (status, set_status) = signal("loading".to_string());
    let (show_settings, set_show_settings) = signal(false);
    let (engine_ready, set_engine_ready) = signal(false);
    let (next_index, set_next_index) = signal(0u32);
    let (active_index, set_active_index) = signal(None::<u32>);

    // Config state
    let (model_dir, set_model_dir) = signal(String::new());
    let (lora_dir, set_lora_dir) = signal("None".to_string());
    let (backend, set_backend) = signal("CUDA".to_string());
    let (audio_host, set_audio_host) = signal(String::new());
    let (audio_device, set_audio_device) = signal(String::new());
    let (max_chars, set_max_chars) = signal(80usize);
    let (dit_steps, set_dit_steps) = signal(10usize);
    let (model_name, set_model_name) = signal(String::new());
    let (prompt_wav_path, set_prompt_wav_path) = signal(String::new());
    let (prompt_text, set_prompt_text) = signal(String::new());
    let (reference_wav_path, set_reference_wav_path) = signal(String::new());
    let (actual_backend, set_actual_backend) = signal(String::new());
    let (status_message, set_status_message) = signal(String::new());

    // Available options
    let (models, set_models) = signal(Vec::<ModelEntry>::new());
    let (loras, set_loras) = signal(Vec::<LoraEntry>::new());
    let (hosts, set_hosts) = signal(Vec::<String>::new());
    let (devices, set_devices) = signal(Vec::<String>::new());

    let (no_model, set_no_model) = signal(false);

    // Initialize on mount
    {
        let set_status = set_status.clone();
        spawn_local(async move {
            // Load config
            debug_log("startup: config load start");
            match tauri_api::invoke_no_args::<AppConfig>("get_config").await {
                Ok(config) => {
                    debug_log(&format!(
                        "startup: config load success model_dir={} backend={}",
                        config.model_dir, config.backend
                    ));
                    set_model_dir.set(config.model_dir.clone());
                    set_lora_dir.set(config.lora_dir.clone().unwrap_or("None".into()));
                    set_prompt_wav_path.set(config.prompt_wav_path.unwrap_or_default());
                    set_prompt_text.set(config.prompt_text.unwrap_or_default());
                    set_reference_wav_path.set(config.reference_wav_path.unwrap_or_default());
                    set_backend.set(config.backend.clone());
                    set_actual_backend.set(config.backend.clone());
                    set_audio_host.set(config.audio_host.clone());
                    set_audio_device.set(config.audio_device.clone());
                    set_max_chars.set(config.max_chars);
                    set_dit_steps.set(config.dit_steps);
                    if config.language == "English" {
                        set_lang.set(Language::English);
                    }
                }
                Err(e) => {
                    debug_log(&format!("startup: config load error {e}"));
                }
            }

            // List models
            debug_log("startup: model list start");
            match tauri_api::invoke_no_args::<Vec<ModelEntry>>("list_models").await {
                Ok(model_list) => {
                    debug_log(&format!(
                        "startup: model list success count={}",
                        model_list.len()
                    ));
                    if model_list.is_empty() {
                        set_no_model.set(true);
                        set_status.set("idle".into());
                        return;
                    }
                    let configured = model_dir.get_untracked();
                    let selected = model_list
                        .iter()
                        .find(|entry| entry.path == configured)
                        .cloned()
                        .unwrap_or_else(|| model_list[0].clone());
                    set_model_dir.set(selected.path.clone());
                    set_model_name.set(selected.name.clone());
                    set_models.set(model_list);
                }
                Err(e) => {
                    debug_log(&format!("startup: model list error {e}"));
                }
            }

            // List audio devices
            debug_log("startup: audio device list start");
            match tauri_api::invoke::<_, AudioDeviceList>(
                "list_audio_devices",
                &ListAudioDevicesArgs {
                    host: non_empty_option(audio_host.get_untracked()),
                },
            )
            .await
            {
                Ok(audio) => {
                    debug_log(&format!(
                        "startup: audio device list success host={} devices={}",
                        audio.selected_host,
                        audio.devices.len()
                    ));
                    set_hosts.set(audio.hosts);
                    set_audio_host.set(audio.selected_host);
                    let selected_device = selected_audio_device(
                        audio_device.get_untracked(),
                        &audio.devices,
                        audio.selected_device,
                    );
                    set_devices.set(audio.devices);
                    set_audio_device.set(selected_device);
                }
                Err(e) => {
                    debug_log(&format!("startup: audio device list error {e}"));
                }
            }

            // List loras
            let md = model_dir.get_untracked();
            debug_log(&format!("startup: lora list start model_dir={md}"));
            let selected_lora = match list_loras_for_model(md).await {
                Ok(lora_list) => {
                    debug_log(&format!(
                        "startup: lora list success count={}",
                        lora_list.len()
                    ));
                    let selected_lora = valid_lora_selection(
                        lora_selection_option(lora_dir.get_untracked()),
                        &lora_list,
                    );
                    set_lora_dir.set(selected_lora.clone().unwrap_or_else(|| "None".to_string()));
                    set_loras.set(lora_list);
                    selected_lora
                }
                Err(e) => {
                    debug_log(&format!("startup: lora list error {e}"));
                    set_lora_dir.set("None".into());
                    set_loras.set(Vec::new());
                    set_status_message.set(e);
                    None
                }
            };

            // Load model
            set_status.set("loading".into());
            let md = model_dir.get_untracked();
            let be = backend.get_untracked();
            debug_log(&format!(
                "startup: model load start model_dir={md} backend={be}"
            ));
            match tauri_api::invoke::<_, ModelInfo>(
                "load_model",
                &LoadModelArgs {
                    model_dir: md.clone(),
                    backend: be,
                },
            )
            .await
            {
                Ok(info) => {
                    debug_log(&format!(
                        "startup: model load success architecture={} sample_rate={} backend={}",
                        info.architecture, info.sample_rate, info.backend
                    ));
                    set_engine_ready.set(true);
                    set_actual_backend.set(info.backend.clone());
                    set_status.set("ready".into());
                    set_status_message.set(info.warning.unwrap_or_default());
                    match apply_lora_selection(selected_lora).await {
                        Ok(()) => debug_log("startup: lora apply success"),
                        Err(e) => {
                            debug_log(&format!("startup: lora apply error {e}"));
                            set_status.set(format!("Error: {}", e));
                            set_status_message.set(e);
                        }
                    }
                }
                Err(e) => {
                    debug_log(&format!("startup: model load error {e}"));
                    set_engine_ready.set(false);
                    web_sys::console::error_1(&format!("Model load error: {}", e).into());
                    set_status.set(format!("Error: {}", e));
                    set_status_message.set(e);
                }
            }
        });
    }

    // Listen for progress events
    {
        let active_index = active_index.clone();
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let progress_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) =
                    serde_wasm_bindgen::from_value::<ProgressPayload>(payload_value)
                {
                    if payload.total > 0 {
                        let progress_value = payload.step as f64 / payload.total as f64;
                        set_history.update(|history| {
                            if let Some(entry) = history.get_mut(payload.index as usize) {
                                entry.progress = progress_value;
                            }
                        });
                        if active_index.get_untracked() == Some(payload.index) {
                            set_progress.set(progress_value);
                        }
                    }
                }
            });
            let _ = tauri_api::tauri_listen("tts-progress", &progress_cb).await;
            progress_cb.forget();
        });
    }

    // Listen for completion events
    {
        let active_index = active_index.clone();
        let set_active_index = set_active_index.clone();
        let set_status = set_status.clone();
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let complete_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) =
                    serde_wasm_bindgen::from_value::<CompletePayload>(payload_value)
                {
                    set_history.update(|history| {
                        if let Some(entry) = history.get_mut(payload.index as usize) {
                            entry.status = "done".into();
                            entry.progress = 1.0;
                        }
                    });
                    if active_index.get_untracked() == Some(payload.index) {
                        set_active_index.set(None);
                        set_status.set("ready".into());
                        set_progress.set(0.0);
                    }
                } else {
                    // Compatibility with older events that did not include an index.
                    set_history.update(|history| {
                        if let Some(entry) = history
                            .iter_mut()
                            .rev()
                            .find(|entry| entry.status == "generating")
                        {
                            entry.status = "done".into();
                            entry.progress = 1.0;
                        }
                    });
                }
            });
            let _ = tauri_api::tauri_listen("tts-complete", &complete_cb).await;
            complete_cb.forget();
        });
    }

    // Listen for synthesis errors
    {
        let active_index = active_index.clone();
        let set_active_index = set_active_index.clone();
        let set_status = set_status.clone();
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let error_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) = serde_wasm_bindgen::from_value::<ErrorPayload>(payload_value) {
                    set_history.update(|history| {
                        if let Some(entry) = history.get_mut(payload.index as usize) {
                            entry.status = format!("error: {}", payload.message);
                            entry.progress = 0.0;
                        }
                    });
                    if active_index.get_untracked() == Some(payload.index) {
                        set_active_index.set(None);
                        set_status.set("ready".into());
                        set_progress.set(0.0);
                    }
                }
            });
            let _ = tauri_api::tauri_listen("tts-error", &error_cb).await;
            error_cb.forget();
        });
    }

    let on_submit = move |text: String| -> bool {
        if status.get_untracked() == "generating"
            || active_index.get_untracked().is_some()
            || !engine_ready.get_untracked()
        {
            return false;
        }

        let idx = next_index.get_untracked();
        set_next_index.set(idx + 1);
        set_active_index.set(Some(idx));

        let trimmed = {
            let mc = max_chars.get_untracked();
            truncate_to_chars(&text, mc)
        };

        let now = js_sys::Date::new_0();
        let timestamp = format!(
            "{:02}:{:02}:{:02}",
            now.get_hours(),
            now.get_minutes(),
            now.get_seconds()
        );

        set_history.update(|h| {
            h.push(TtsEntry {
                timestamp,
                text: trimmed.clone(),
                status: "generating".into(),
                progress: 0.0,
            });
        });

        set_status.set("generating".into());
        set_progress.set(0.0);

        let steps = dit_steps.get_untracked();
        spawn_local(async move {
            let payload = SynthesisPayload {
                dit_steps: steps,
                index: idx,
                text: trimmed,
                prompt_wav_path: non_empty_option(prompt_wav_path.get_untracked()),
                prompt_text: non_empty_option(prompt_text.get_untracked()),
                reference_wav_path: non_empty_option(reference_wav_path.get_untracked()),
            };

            if let Err(e) =
                tauri_api::invoke_unit("synthesize", &SynthesizeArgs { args: payload }).await
            {
                web_sys::console::error_1(&format!("Synthesize error: {}", e).into());
                set_history.update(|history| {
                    if let Some(entry) = history.get_mut(idx as usize) {
                        entry.status = format!("error: {}", e);
                        entry.progress = 0.0;
                    }
                });
                if active_index.get_untracked() == Some(idx) {
                    set_active_index.set(None);
                    set_status.set("ready".into());
                    set_progress.set(0.0);
                    set_status_message.set(e);
                }
            }
        });
        true
    };

    let on_model_selected = move |path: String| {
        set_model_dir.set(path.clone());
        set_no_model.set(false);
        set_status.set("loading".into());
        spawn_local(async move {
            let be = backend.get_untracked();
            debug_log(&format!(
                "model selection: load start model_dir={} backend={}",
                path, be
            ));
            let selected_name = models
                .get_untracked()
                .into_iter()
                .find(|entry| entry.path == path)
                .map(|entry| entry.name)
                .unwrap_or_else(|| path.clone());
            set_model_name.set(selected_name);
            debug_log(&format!(
                "model selection: lora list start model_dir={path}"
            ));
            let selected_lora = match list_loras_for_model(path.clone()).await {
                Ok(lora_list) => {
                    debug_log(&format!(
                        "model selection: lora list success count={}",
                        lora_list.len()
                    ));
                    let selected_lora = valid_lora_selection(
                        lora_selection_option(lora_dir.get_untracked()),
                        &lora_list,
                    );
                    set_lora_dir.set(selected_lora.clone().unwrap_or_else(|| "None".to_string()));
                    set_loras.set(lora_list);
                    selected_lora
                }
                Err(e) => {
                    debug_log(&format!("model selection: lora list error {e}"));
                    set_lora_dir.set("None".into());
                    set_loras.set(Vec::new());
                    set_status_message.set(e);
                    None
                }
            };
            match tauri_api::invoke::<_, ModelInfo>(
                "load_model",
                &LoadModelArgs {
                    model_dir: path.clone(),
                    backend: be,
                },
            )
            .await
            {
                Ok(info) => {
                    debug_log(&format!(
                        "model selection: load success architecture={} sample_rate={} backend={}",
                        info.architecture, info.sample_rate, info.backend
                    ));
                    set_engine_ready.set(true);
                    set_actual_backend.set(info.backend.clone());
                    set_status.set("ready".into());
                    set_status_message.set(info.warning.unwrap_or_default());
                    match apply_lora_selection(selected_lora).await {
                        Ok(()) => debug_log("model selection: lora apply success"),
                        Err(e) => {
                            debug_log(&format!("model selection: lora apply error {e}"));
                            set_status.set(format!("Error: {}", e));
                            set_status_message.set(e);
                        }
                    }
                }
                Err(e) => {
                    debug_log(&format!("model selection: load error {e}"));
                    set_engine_ready.set(false);
                    set_status.set(format!("Error: {}", e));
                    set_status_message.set(e);
                }
            }
        });
    };

    let on_apply_settings = move |vals: SettingsValues| {
        if active_index.get_untracked().is_some()
            || matches!(status.get_untracked().as_str(), "generating" | "loading")
        {
            set_status_message
                .set("Finish the current operation before changing settings".to_string());
            return;
        }

        set_show_settings.set(false);
        let requested_model_dir = vals.model_dir.clone();
        let requested_lora = lora_selection_option(vals.lora_dir.clone());
        let requested_backend = vals.backend.clone();
        let requested_audio_host = vals.audio_host.clone();
        let requested_audio_device = vals.audio_device.clone();
        let requested_max_chars = vals.max_chars;
        let requested_dit_steps = vals.dit_steps;
        let requested_prompt_wav_path = vals.prompt_wav_path.clone();
        let requested_prompt_text = vals.prompt_text.clone();
        let requested_reference_wav_path = vals.reference_wav_path.clone();
        let requested_language = vals.language.clone();
        let previous_engine_ready = engine_ready.get_untracked();
        let need_reload = requested_model_dir != model_dir.get_untracked()
            || requested_backend != backend.get_untracked();
        let next_language = if requested_language == "English" {
            Language::English
        } else {
            Language::Chinese
        };

        if need_reload {
            debug_log(&format!(
                "settings reload start model_dir={} backend={}",
                requested_model_dir, requested_backend
            ));
            set_engine_ready.set(false);
            set_status.set("loading".into());
            set_status_message.set(String::new());
        }

        spawn_local(async move {
            let restore_after_error = |message: String| {
                if need_reload {
                    debug_log(&format!("settings reload error {message}"));
                } else {
                    debug_log(&format!("settings apply error {message}"));
                }
                if need_reload {
                    set_engine_ready.set(previous_engine_ready);
                    if previous_engine_ready {
                        set_status.set("ready".into());
                    } else {
                        set_status.set(format!("Error: {}", message));
                    }
                }
                set_status_message.set(message);
            };

            let lora_list = match list_loras_for_model(requested_model_dir.clone()).await {
                Ok(lora_list) => lora_list,
                Err(e) => {
                    restore_after_error(e);
                    return;
                }
            };
            let selected_lora = valid_lora_selection(requested_lora, &lora_list);

            let audio = match tauri_api::invoke::<_, AudioDeviceList>(
                "list_audio_devices",
                &ListAudioDevicesArgs {
                    host: non_empty_option(requested_audio_host.clone()),
                },
            )
            .await
            {
                Ok(audio) => audio,
                Err(e) => {
                    restore_after_error(e);
                    return;
                }
            };
            let validated_audio_device = selected_audio_device(
                requested_audio_device.clone(),
                &audio.devices,
                audio.selected_device.clone(),
            );
            let validated_audio_host = audio.selected_host.clone();
            let audio_hosts = audio.hosts;
            let audio_devices = audio.devices;

            let mut final_lora = selected_lora.clone();
            let mut final_backend = actual_backend.get_untracked();
            let mut final_status_message = String::new();

            if need_reload {
                match tauri_api::invoke::<_, ModelInfo>(
                    "load_model",
                    &LoadModelArgs {
                        model_dir: requested_model_dir.clone(),
                        backend: requested_backend.clone(),
                    },
                )
                .await
                {
                    Ok(info) => {
                        debug_log(&format!(
                            "settings reload success architecture={} sample_rate={} backend={}",
                            info.architecture, info.sample_rate, info.backend
                        ));
                        final_backend = info.backend;
                        final_status_message = info.warning.unwrap_or_default();
                    }
                    Err(e) => {
                        restore_after_error(e);
                        return;
                    }
                }
            } else if let Err(e) = apply_lora_selection(selected_lora).await {
                restore_after_error(e);
                return;
            }

            if need_reload {
                if let Err(e) = apply_lora_selection(final_lora.clone()).await {
                    final_lora = None;
                    final_status_message = e;
                }
            }

            let config = serde_json::json!({
                "model_dir": requested_model_dir.clone(),
                "lora_dir": final_lora.clone(),
                "prompt_wav_path": non_empty_option(requested_prompt_wav_path.clone()),
                "prompt_text": non_empty_option(requested_prompt_text.clone()),
                "reference_wav_path": non_empty_option(requested_reference_wav_path.clone()),
                "backend": requested_backend.clone(),
                "audio_host": validated_audio_host.clone(),
                "audio_device": validated_audio_device.clone(),
                "max_chars": requested_max_chars,
                "dit_steps": requested_dit_steps,
                "language": requested_language,
            });
            let _ = tauri_api::invoke_unit("save_config", &serde_json::json!({ "config": config }))
                .await;

            let selected_name = models
                .get_untracked()
                .into_iter()
                .find(|entry| entry.path == requested_model_dir)
                .map(|entry| entry.name)
                .unwrap_or_else(|| requested_model_dir.clone());

            set_model_dir.set(requested_model_dir);
            set_model_name.set(selected_name);
            set_lora_dir.set(final_lora.unwrap_or_else(|| "None".to_string()));
            set_loras.set(lora_list);
            set_backend.set(requested_backend);
            set_actual_backend.set(final_backend);
            set_audio_host.set(validated_audio_host);
            set_audio_device.set(validated_audio_device);
            set_hosts.set(audio_hosts);
            set_devices.set(audio_devices);
            set_max_chars.set(requested_max_chars);
            set_dit_steps.set(requested_dit_steps);
            set_prompt_wav_path.set(requested_prompt_wav_path);
            set_prompt_text.set(requested_prompt_text);
            set_reference_wav_path.set(requested_reference_wav_path);
            set_lang.set(next_language);
            set_engine_ready.set(true);
            set_status.set("ready".into());
            set_status_message.set(final_status_message);
        });
    };

    view! {
        <div class="flex flex-col h-screen">
            <Header lang=lang on_settings=move |_| {
                if active_index.get_untracked().is_some()
                    || matches!(status.get_untracked().as_str(), "generating" | "loading")
                {
                    set_status_message.set("Finish the current operation before changing settings".to_string());
                } else {
                    set_show_settings.set(true);
                }
            } />
            <History lang=lang entries=history />
            <ProgressBar progress=progress status=status lang=lang />
            <InputBox lang=lang engine_ready=engine_ready status=status on_submit=on_submit />
            <StatusBar
                lang=lang
                status=status
                model_name=model_name
                actual_backend=actual_backend
                lora_dir=lora_dir
                audio_host=audio_host
                audio_device=audio_device
                status_message=status_message
            />
            <Show when=move || show_settings.get()>
                <SettingsModal
                    lang=lang
                    model_dir=model_dir
                    lora_dir=lora_dir
                    backend=backend
                    audio_host=audio_host
                    audio_device=audio_device
                    max_chars=max_chars
                    dit_steps=dit_steps
                    prompt_wav_path=prompt_wav_path
                    prompt_text=prompt_text
                    reference_wav_path=reference_wav_path
                    models=models
                    loras=loras
                    hosts=hosts
                    devices=devices
                    on_close=move |_| set_show_settings.set(false)
                    on_apply=on_apply_settings
                />
            </Show>
            <Show when=move || no_model.get()>
                <ModelSelect lang=lang on_select=on_model_selected />
            </Show>
        </div>
    }
}
