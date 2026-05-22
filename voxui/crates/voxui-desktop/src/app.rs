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

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    pub model_dir: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    pub lora_path: Option<String>,
    pub lora_size_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoadProgress {
    Hidden,
    Reading {
        label: String,
        bytes_read: u64,
        total_bytes: u64,
    },
    DeviceLoading {
        backend: String,
    },
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
struct ListModelChoicesArgs {
    model_root: String,
}

#[derive(Serialize)]
struct LoadModelChoiceArgs {
    args: LoadModelChoicePayload,
}

#[derive(Serialize)]
struct LoadModelChoicePayload {
    choice_id: String,
    model_dir: String,
    model_path: String,
    lora_path: Option<String>,
    backend: String,
}

#[derive(Serialize)]
struct ListAudioDevicesArgs {
    host: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub model_root: String,
    pub selected_model_choice_id: String,
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
struct LoadProgressPayload {
    phase: String,
    file_label: Option<String>,
    bytes_read: u64,
    total_bytes: u64,
    backend: Option<String>,
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

fn selected_choice(choices: &[ModelChoice], selected_id: &str) -> Option<ModelChoice> {
    choices
        .iter()
        .find(|choice| choice.id == selected_id)
        .cloned()
        .or_else(|| choices.first().cloned())
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

fn language_from_config(value: &str) -> Language {
    if value == "English" {
        Language::English
    } else {
        Language::Chinese
    }
}

fn choice_config_fields(
    choices: Vec<ModelChoice>,
    selected_id: String,
) -> (String, Option<String>) {
    let selected = selected_choice(&choices, &selected_id);
    (
        selected
            .as_ref()
            .map(|choice| choice.model_dir.clone())
            .unwrap_or_default(),
        selected
            .as_ref()
            .and_then(|choice| choice.lora_path.clone()),
    )
}

fn choice_identity_key(model_root: &str, choice: &ModelChoice) -> String {
    format!(
        "{}::{}::{}",
        model_root.trim(),
        choice.model_dir,
        choice.lora_path.as_deref().unwrap_or_default()
    )
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

    let (model_root, set_model_root) = signal(String::new());
    let (selected_choice_id, set_selected_choice_id) = signal(String::new());
    let (loaded_choice_key, set_loaded_choice_key) = signal(String::new());
    let (loaded_choice_name, set_loaded_choice_name) = signal(String::new());
    let (model_choices, set_model_choices) = signal(Vec::<ModelChoice>::new());
    let (load_progress, set_load_progress) = signal(LoadProgress::Hidden);
    let (load_in_progress, set_load_in_progress) = signal(false);
    let (settings_apply_in_progress, set_settings_apply_in_progress) = signal(false);

    let (backend, set_backend) = signal("CUDA".to_string());
    let (audio_host, set_audio_host) = signal(String::new());
    let (audio_device, set_audio_device) = signal(String::new());
    let (max_chars, set_max_chars) = signal(80usize);
    let (dit_steps, set_dit_steps) = signal(10usize);
    let (prompt_wav_path, set_prompt_wav_path) = signal(String::new());
    let (prompt_text, set_prompt_text) = signal(String::new());
    let (reference_wav_path, set_reference_wav_path) = signal(String::new());
    let (actual_backend, set_actual_backend) = signal(String::new());
    let (status_message, set_status_message) = signal(String::new());

    let (hosts, set_hosts) = signal(Vec::<String>::new());
    let (devices, set_devices) = signal(Vec::<String>::new());
    let (no_model, set_no_model) = signal(false);

    {
        spawn_local(async move {
            let mut legacy_model_dir = String::new();
            let mut legacy_lora_path = None::<String>;
            debug_log("startup: config load start");
            match tauri_api::invoke_no_args::<AppConfig>("get_config").await {
                Ok(config) => {
                    debug_log(&format!(
                        "startup: config load success model_root={} selected_choice={} backend={}",
                        config.model_root, config.selected_model_choice_id, config.backend
                    ));
                    set_model_root.set(config.model_root.clone());
                    set_selected_choice_id.set(config.selected_model_choice_id.clone());
                    legacy_model_dir = config.model_dir.clone();
                    legacy_lora_path = config.lora_dir.clone();
                    set_prompt_wav_path.set(config.prompt_wav_path.unwrap_or_default());
                    set_prompt_text.set(config.prompt_text.unwrap_or_default());
                    set_reference_wav_path.set(config.reference_wav_path.unwrap_or_default());
                    set_backend.set(config.backend.clone());
                    set_actual_backend.set(config.backend.clone());
                    set_audio_host.set(config.audio_host.clone());
                    set_audio_device.set(config.audio_device.clone());
                    set_max_chars.set(config.max_chars);
                    set_dit_steps.set(config.dit_steps);
                    set_lang.set(language_from_config(&config.language));
                    if config.selected_model_choice_id.trim().is_empty() {
                        set_selected_choice_id.set(legacy_model_dir.clone());
                    }
                }
                Err(e) => {
                    debug_log(&format!("startup: config load error {e}"));
                    set_status_message.set(e);
                }
            }

            debug_log("startup: model choices list start");
            match tauri_api::invoke::<_, Vec<ModelChoice>>(
                "list_model_choices",
                &ListModelChoicesArgs {
                    model_root: model_root.get_untracked(),
                },
            )
            .await
            {
                Ok(choices) => {
                    debug_log(&format!("startup: model choices count={}", choices.len()));
                    set_no_model.set(choices.is_empty());
                    let configured_id = selected_choice_id.get_untracked();
                    let selected = choices
                        .iter()
                        .find(|choice| choice.id == configured_id)
                        .cloned()
                        .or_else(|| {
                            choices
                                .iter()
                                .find(|choice| {
                                    choice.model_dir == legacy_model_dir
                                        && choice.lora_path == legacy_lora_path
                                })
                                .cloned()
                        })
                        .or_else(|| choices.first().cloned());
                    set_selected_choice_id
                        .set(selected.map(|choice| choice.id).unwrap_or_default());
                    set_model_choices.set(choices);
                    set_status.set("idle".into());
                }
                Err(e) => {
                    debug_log(&format!("startup: model choices error {e}"));
                    set_status_message.set(e);
                    set_status.set("idle".into());
                }
            }

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
                    set_status_message.set(e);
                }
            }
        });
    }

    {
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

    {
        spawn_local(async move {
            let load_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) =
                    serde_wasm_bindgen::from_value::<LoadProgressPayload>(payload_value)
                {
                    match payload.phase.as_str() {
                        "reading" => set_load_progress.set(LoadProgress::Reading {
                            label: payload.file_label.unwrap_or_else(|| "GGUF".to_string()),
                            bytes_read: payload.bytes_read,
                            total_bytes: payload.total_bytes,
                        }),
                        "device_loading" => set_load_progress.set(LoadProgress::DeviceLoading {
                            backend: payload.backend.unwrap_or_else(|| "device".to_string()),
                        }),
                        _ => {}
                    }
                }
            });
            let _ = tauri_api::tauri_listen("load-progress", &load_cb).await;
            load_cb.forget();
        });
    }

    {
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

    {
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
                        set_status_message.set(payload.message);
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

        let trimmed = truncate_to_chars(&text, max_chars.get_untracked());
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

        spawn_local(async move {
            let payload = SynthesisPayload {
                dit_steps: dit_steps.get_untracked(),
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

    let on_choice_selected = move |choice_id: String| {
        if settings_apply_in_progress.get_untracked() {
            set_status_message
                .set("Finish the current operation before changing settings".to_string());
            return;
        }

        let next_selected_choice_id = choice_id;
        set_selected_choice_id.set(next_selected_choice_id.clone());
        set_no_model.set(model_choices.get_untracked().is_empty());
        set_status_message.set(String::new());

        let next_choices = model_choices.get_untracked();
        let (selected_model_dir, selected_lora_path) = choice_config_fields(
            next_choices.clone(),
            next_selected_choice_id.clone(),
        );
        let config = serde_json::json!({
            "model_root": model_root.get_untracked(),
            "selected_model_choice_id": next_selected_choice_id,
            "model_dir": selected_model_dir,
            "lora_dir": selected_lora_path,
            "prompt_wav_path": non_empty_option(prompt_wav_path.get_untracked()),
            "prompt_text": non_empty_option(prompt_text.get_untracked()),
            "reference_wav_path": non_empty_option(reference_wav_path.get_untracked()),
            "backend": backend.get_untracked(),
            "audio_host": audio_host.get_untracked(),
            "audio_device": audio_device.get_untracked(),
            "max_chars": max_chars.get_untracked(),
            "dit_steps": dit_steps.get_untracked(),
            "language": match lang.get_untracked() {
                Language::English => "English",
                Language::Chinese => "Chinese",
            },
        });

        spawn_local(async move {
            if let Err(e) =
                tauri_api::invoke_unit("save_config", &serde_json::json!({ "config": config }))
                    .await
            {
                debug_log(&format!("dropdown selection save error {e}"));
                set_status_message.set(e);
            }
        });
    };

    let on_load_or_cancel = move |_| {
        if settings_apply_in_progress.get_untracked() {
            set_status_message
                .set("Finish the current operation before changing settings".to_string());
            return;
        }

        if load_in_progress.get_untracked() {
            spawn_local(async move {
                let _ = tauri_api::invoke_no_args::<()>("cancel_load").await;
            });
            return;
        }

        let selected = selected_choice(
            &model_choices.get_untracked(),
            &selected_choice_id.get_untracked(),
        );
        let Some(choice) = selected else {
            set_status_message.set("No model selected".to_string());
            return;
        };

        set_load_in_progress.set(true);
        set_engine_ready.set(false);
        set_status.set("loading".into());
        set_status_message.set(String::new());
        set_load_progress.set(LoadProgress::Reading {
            label: "model.gguf".to_string(),
            bytes_read: 0,
            total_bytes: choice.model_size_bytes,
        });

        spawn_local(async move {
            let result = tauri_api::invoke::<_, ModelInfo>(
                "load_model_choice",
                &LoadModelChoiceArgs {
                    args: LoadModelChoicePayload {
                        choice_id: choice.id.clone(),
                        model_dir: choice.model_dir.clone(),
                        model_path: choice.model_path.clone(),
                        lora_path: choice.lora_path.clone(),
                        backend: backend.get_untracked(),
                    },
                },
            )
            .await;

            set_load_in_progress.set(false);
            set_load_progress.set(LoadProgress::Hidden);
            match result {
                Ok(info) => {
                    debug_log(&format!(
                        "model load success architecture={} sample_rate={} backend={}",
                        info.architecture, info.sample_rate, info.backend
                    ));
                    set_engine_ready.set(true);
                    set_loaded_choice_key.set(choice_identity_key(&model_root.get_untracked(), &choice));
                    set_loaded_choice_name.set(choice.name.clone());
                    set_actual_backend.set(info.backend.clone());
                    set_status.set("ready".into());
                    set_status_message.set(info.warning.unwrap_or_default());
                }
                Err(e) => {
                    debug_log(&format!("model load error {e}"));
                    let had_loaded = !loaded_choice_key.get_untracked().is_empty();
                    set_engine_ready.set(had_loaded);
                    set_status.set(if had_loaded {
                        "ready".into()
                    } else {
                        "idle".into()
                    });
                    set_status_message.set(e);
                }
            }
        });
    };

    let on_apply_settings = move |vals: SettingsValues| {
        if active_index.get_untracked().is_some()
            || status.get_untracked() == "generating"
            || load_in_progress.get_untracked()
            || settings_apply_in_progress.get_untracked()
        {
            set_status_message
                .set("Finish the current operation before changing settings".to_string());
            return;
        }

        let requested_model_root = vals.model_root.clone();
        let requested_backend = vals.backend.clone();
        let requested_audio_host = vals.audio_host.clone();
        let requested_audio_device = vals.audio_device.clone();
        let requested_max_chars = vals.max_chars;
        let requested_dit_steps = vals.dit_steps;
        let requested_prompt_wav_path = vals.prompt_wav_path.clone();
        let requested_prompt_text = vals.prompt_text.clone();
        let requested_reference_wav_path = vals.reference_wav_path.clone();
        let requested_language = vals.language.clone();
        let model_root_changed = requested_model_root != model_root.get_untracked();
        let next_language = language_from_config(&requested_language);
        set_settings_apply_in_progress.set(true);

        spawn_local(async move {
            let restore_after_error = |message: String| {
                debug_log(&format!("settings apply error {message}"));
                set_status_message.set(message);
                set_settings_apply_in_progress.set(false);
            };

            let mut next_choices = model_choices.get_untracked();
            let mut next_selected_choice_id = selected_choice_id.get_untracked();
            if model_root_changed {
                debug_log(&format!(
                    "settings model choices list start model_root={requested_model_root}"
                ));
                next_choices = match tauri_api::invoke::<_, Vec<ModelChoice>>(
                    "list_model_choices",
                    &ListModelChoicesArgs {
                        model_root: requested_model_root.clone(),
                    },
                )
                .await
                {
                    Ok(choices) => choices,
                    Err(e) => {
                        restore_after_error(e);
                        return;
                    }
                };
                next_selected_choice_id = selected_choice(&next_choices, &next_selected_choice_id)
                    .map(|choice| choice.id)
                    .unwrap_or_default();
            }

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

            let (selected_model_dir, selected_lora_path) =
                choice_config_fields(next_choices.clone(), next_selected_choice_id.clone());
            let config = serde_json::json!({
                "model_root": requested_model_root.clone(),
                "selected_model_choice_id": next_selected_choice_id.clone(),
                "model_dir": selected_model_dir,
                "lora_dir": selected_lora_path,
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
            if let Err(e) =
                tauri_api::invoke_unit("save_config", &serde_json::json!({ "config": config }))
                    .await
            {
                restore_after_error(e);
                return;
            }

            if model_root_changed {
                set_model_choices.set(next_choices.clone());
                set_selected_choice_id.set(next_selected_choice_id);
                set_no_model.set(next_choices.is_empty());
            }
            set_show_settings.set(false);
            set_model_root.set(requested_model_root);
            set_backend.set(requested_backend);
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
            set_status_message.set(String::new());
            set_settings_apply_in_progress.set(false);
            if !engine_ready.get_untracked() && status.get_untracked() == "loading" {
                set_status.set("idle".into());
            }
        });
    };

    let selected_choice_name = Signal::derive(move || {
        selected_choice(&model_choices.get(), &selected_choice_id.get())
            .map(|choice| choice.name)
            .unwrap_or_default()
    });
    let selected_choice_key = Signal::derive(move || {
        selected_choice(&model_choices.get(), &selected_choice_id.get())
            .map(|choice| choice_identity_key(&model_root.get(), &choice))
            .unwrap_or_default()
    });
    let generating =
        Signal::derive(move || active_index.get().is_some() || status.get() == "generating");

    view! {
        <div class="flex flex-col h-screen">
            <Header
                lang=lang
                choices=model_choices
                selected_choice_id=selected_choice_id
                selected_choice_key=selected_choice_key
                loaded_choice_key=loaded_choice_key.into()
                load_in_progress=load_in_progress
                generating=generating
                settings_apply_in_progress=settings_apply_in_progress
                on_choice_selected=on_choice_selected
                on_load_or_cancel=on_load_or_cancel
                on_settings=move |_| {
                    if settings_apply_in_progress.get_untracked() {
                        set_status_message
                            .set("Finish the current operation before changing settings".to_string());
                    } else if active_index.get_untracked().is_some()
                        || status.get_untracked() == "generating"
                        || load_in_progress.get_untracked()
                    {
                        set_status_message.set("Finish the current operation before changing settings".to_string());
                    } else {
                        set_show_settings.set(true);
                    }
                }
            />
            <History lang=lang entries=history />
            <ProgressBar progress=progress status=status lang=lang />
            <ModelLoadProgressBar progress=load_progress lang=lang />
            <InputBox lang=lang engine_ready=engine_ready status=status on_submit=on_submit on_cancel=move |_| {
                spawn_local(async move {
                    let _ = tauri_api::invoke_no_args::<()>("cancel_synthesis").await;
                });
            } />
            <StatusBar
                lang=lang
                status=status
                selected_choice_name=selected_choice_name
                loaded_choice_name=loaded_choice_name.into()
                actual_backend=actual_backend
                audio_host=audio_host
                audio_device=audio_device
                status_message=status_message
            />
            <Show when=move || show_settings.get()>
                <SettingsModal
                    lang=lang
                    model_root=model_root
                    backend=backend
                    audio_host=audio_host
                    audio_device=audio_device
                    max_chars=max_chars
                    dit_steps=dit_steps
                    prompt_wav_path=prompt_wav_path
                    prompt_text=prompt_text
                    reference_wav_path=reference_wav_path
                    hosts=hosts
                    devices=devices
                    settings_apply_in_progress=settings_apply_in_progress
                    loading_or_generating=Signal::derive(move || {
                        load_in_progress.get() || generating.get() || settings_apply_in_progress.get()
                    })
                    on_close=move |_| set_show_settings.set(false)
                    on_apply=on_apply_settings
                />
            </Show>
            <Show when=move || no_model.get()>
                <div class="shrink-0 px-4 py-2 bg-gray-900 border-t border-gray-700 text-xs text-gray-400">
                    {move || lang.get().t("no_models_found")}
                </div>
            </Show>
        </div>
    }
}
