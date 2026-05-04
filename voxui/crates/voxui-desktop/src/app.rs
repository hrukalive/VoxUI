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

#[component]
pub fn App() -> impl IntoView {
    let (lang, set_lang) = signal(Language::Chinese);
    let (history, set_history) = signal(Vec::<TtsEntry>::new());
    let (progress, set_progress) = signal(0.0f64);
    let (status, set_status) = signal("loading".to_string());
    let (show_settings, set_show_settings) = signal(false);
    let (engine_ready, set_engine_ready) = signal(false);
    let (next_index, set_next_index) = signal(0u32);

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
    let (_status_message, set_status_message) = signal(String::new());

    // Available options
    let (models, set_models) = signal(Vec::<ModelEntry>::new());
    let (loras, set_loras) = signal(Vec::<LoraEntry>::new());
    let (model_paths, set_model_paths) = signal(Vec::<String>::new());
    let (lora_paths, set_lora_paths) = signal(Vec::<String>::new());
    let (hosts, set_hosts) = signal(Vec::<String>::new());
    let (devices, set_devices) = signal(Vec::<String>::new());

    let (no_model, set_no_model) = signal(false);

    // Initialize on mount
    {
        let set_status = set_status.clone();
        spawn_local(async move {
            // Load config
            if let Ok(config) = tauri_api::invoke_no_args::<AppConfig>("get_config").await {
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

            // List models
            if let Ok(model_list) = tauri_api::invoke_no_args::<Vec<ModelEntry>>("list_models").await {
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
                set_model_paths.set(
                    model_list
                        .iter()
                        .map(|entry| entry.path.clone())
                        .collect::<Vec<_>>(),
                );
                set_models.set(model_list);
            }

            // List audio devices
            if let Ok(audio) = tauri_api::invoke::<_, AudioDeviceList>(
                "list_audio_devices",
                &ListAudioDevicesArgs {
                    host: non_empty_option(audio_host.get_untracked()),
                },
            ).await {
                set_hosts.set(audio.hosts);
                set_audio_host.set(audio.selected_host);
                set_devices.set(audio.devices);
                set_audio_device.set(audio.selected_device);
            }

            // List loras
            let md = model_dir.get_untracked();
            if let Ok(lora_list) = tauri_api::invoke::<_, Vec<LoraEntry>>(
                "list_lora_dirs",
                &ListLoraArgs { model_dir: md.clone() },
            ).await {
                set_lora_paths.set(
                    lora_list
                        .iter()
                        .map(|entry| entry.path.clone().unwrap_or_else(|| "None".to_string()))
                        .collect::<Vec<_>>(),
                );
                set_loras.set(lora_list);
            }

            // Load model
            set_status.set("loading".into());
            let md = model_dir.get_untracked();
            let be = backend.get_untracked();
            match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: md, backend: be }).await {
                Ok(info) => {
                    set_engine_ready.set(true);
                    set_actual_backend.set(info.backend.clone());
                    set_status.set("ready".into());
                    set_status_message.set(info.warning.unwrap_or_default());
                    let lora_path = lora_selection_option(lora_dir.get_untracked());
                    let _ = tauri_api::invoke_unit(
                        "apply_lora",
                        &ApplyLoraArgs {
                            args: ApplyLoraPayload { lora_dir: lora_path },
                        },
                    )
                    .await;
                }
                Err(e) => {
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
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let progress_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) = serde_wasm_bindgen::from_value::<ProgressPayload>(payload_value) {
                    if payload.total > 0 {
                        let progress_value = payload.step as f64 / payload.total as f64;
                        set_progress.set(progress_value);
                        set_history.update(|history| {
                            if let Some(entry) = history.get_mut(payload.index as usize) {
                                entry.progress = progress_value;
                            }
                        });
                    }
                }
            });
            let _ = tauri_api::tauri_listen("tts-progress", &progress_cb).await;
            progress_cb.forget();
        });
    }

    // Listen for completion events
    {
        let set_status = set_status.clone();
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let complete_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                set_status.set("ready".into());
                set_progress.set(0.0);
                if let Ok(payload) = serde_wasm_bindgen::from_value::<CompletePayload>(payload_value) {
                    set_history.update(|history| {
                        if let Some(entry) = history.get_mut(payload.index as usize) {
                            entry.status = "done".into();
                            entry.progress = 1.0;
                        }
                    });
                } else {
                    // Compatibility with older events that did not include an index.
                    set_history.update(|history| {
                        if let Some(entry) = history.iter_mut().rev().find(|entry| entry.status == "generating") {
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
        let set_status = set_status.clone();
        let set_progress = set_progress.clone();
        let set_history = set_history.clone();
        spawn_local(async move {
            let error_cb = Closure::new(move |val: JsValue| {
                let payload_value = tauri_api::event_payload(val);
                if let Ok(payload) = serde_wasm_bindgen::from_value::<ErrorPayload>(payload_value) {
                    set_status.set("ready".into());
                    set_progress.set(0.0);
                    set_history.update(|history| {
                        if let Some(entry) = history.get_mut(payload.index as usize) {
                            entry.status = format!("error: {}", payload.message);
                            entry.progress = 0.0;
                        }
                    });
                }
            });
            let _ = tauri_api::tauri_listen("tts-error", &error_cb).await;
            error_cb.forget();
        });
    }

    let on_submit = move |text: String| {
        let idx = next_index.get_untracked();
        set_next_index.set(idx + 1);

        let trimmed = {
            let mc = max_chars.get_untracked();
            if text.len() > mc { text[..mc].to_string() } else { text.clone() }
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

            if let Err(e) = tauri_api::invoke_unit("synthesize", &SynthesizeArgs { args: payload }).await {
                web_sys::console::error_1(&format!("Synthesize error: {}", e).into());
            }
        });
    };

    let on_model_selected = move |path: String| {
        set_model_dir.set(path.clone());
        set_no_model.set(false);
        set_status.set("loading".into());
        spawn_local(async move {
            let be = backend.get_untracked();
            let selected_name = models
                .get_untracked()
                .into_iter()
                .find(|entry| entry.path == path)
                .map(|entry| entry.name)
                .unwrap_or_else(|| path.clone());
            set_model_name.set(selected_name);
            match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: path, backend: be }).await {
                Ok(info) => {
                    set_engine_ready.set(true);
                    set_actual_backend.set(info.backend.clone());
                    set_status.set("ready".into());
                    set_status_message.set(info.warning.unwrap_or_default());
                    let lora_path = lora_selection_option(lora_dir.get_untracked());
                    let _ = tauri_api::invoke_unit(
                        "apply_lora",
                        &ApplyLoraArgs {
                            args: ApplyLoraPayload { lora_dir: lora_path },
                        },
                    )
                    .await;
                }
                Err(e) => {
                    set_engine_ready.set(false);
                    set_status.set(format!("Error: {}", e));
                    set_status_message.set(e);
                }
            }
        });
    };

    let on_apply_settings = move |vals: SettingsValues| {
        set_show_settings.set(false);
        let need_reload = vals.model_dir != model_dir.get_untracked() || vals.backend != backend.get_untracked();
        let lora_dir_option = lora_selection_option(vals.lora_dir.clone());
        let selected_lora = lora_dir_option.clone().unwrap_or_else(|| "None".to_string());

        set_model_dir.set(vals.model_dir.clone());
        set_lora_dir.set(selected_lora.clone());
        set_backend.set(vals.backend.clone());
        set_audio_host.set(vals.audio_host.clone());
        set_audio_device.set(vals.audio_device.clone());
        set_max_chars.set(vals.max_chars);
        set_dit_steps.set(vals.dit_steps);

        // Save config
        {
            let config = serde_json::json!({
                "model_dir": vals.model_dir,
                "lora_dir": lora_dir_option.clone(),
                "prompt_wav_path": non_empty_option(prompt_wav_path.get_untracked()),
                "prompt_text": non_empty_option(prompt_text.get_untracked()),
                "reference_wav_path": non_empty_option(reference_wav_path.get_untracked()),
                "backend": vals.backend,
                "audio_host": vals.audio_host,
                "audio_device": vals.audio_device,
                "max_chars": vals.max_chars,
                "dit_steps": vals.dit_steps,
                "language": match lang.get_untracked() { Language::Chinese => "Chinese", Language::English => "English" },
            });
            spawn_local(async move {
                let _ = tauri_api::invoke_unit("save_config", &serde_json::json!({ "config": config })).await;
            });
        }

        if need_reload {
            set_engine_ready.set(false);
            set_status.set("loading".into());
            let md = vals.model_dir.clone();
            let be = vals.backend.clone();
            let ld = lora_dir_option.clone();
            let selected_name = models
                .get_untracked()
                .into_iter()
                .find(|entry| entry.path == md)
                .map(|entry| entry.name)
                .unwrap_or_else(|| md.clone());
            set_model_name.set(selected_name);
            spawn_local(async move {
                match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: md, backend: be }).await {
                    Ok(info) => {
                        set_engine_ready.set(true);
                        set_actual_backend.set(info.backend.clone());
                        set_status.set("ready".into());
                        set_status_message.set(info.warning.unwrap_or_default());
                        let _ = tauri_api::invoke_unit(
                            "apply_lora",
                            &ApplyLoraArgs {
                                args: ApplyLoraPayload { lora_dir: ld },
                            },
                        )
                        .await;
                    }
                    Err(e) => {
                        set_engine_ready.set(false);
                        set_status.set(format!("Error: {}", e));
                        set_status_message.set(e);
                    }
                }
            });
        } else {
            spawn_local(async move {
                let _ = tauri_api::invoke_unit(
                    "apply_lora",
                    &ApplyLoraArgs {
                        args: ApplyLoraPayload { lora_dir: lora_dir_option },
                    },
                )
                .await;
            });
        }
    };

    view! {
        <div class="flex flex-col h-screen">
            <Header lang=lang on_settings=move |_| set_show_settings.set(true) />
            <History lang=lang entries=history />
            <ProgressBar progress=progress status=status lang=lang />
            <InputBox lang=lang engine_ready=engine_ready on_submit=on_submit />
            <StatusBar lang=lang status=status model_name=model_name backend=actual_backend />
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
                    models=model_paths
                    loras=lora_paths
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
