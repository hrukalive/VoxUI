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

#[derive(Serialize)]
struct SynthesizeArgs {
    text: String,
    dit_steps: u32,
    index: u32,
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
struct LoadLoraArgs {
    lora_dir: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub model_dir: String,
    pub lora_dir: Option<String>,
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
}

#[derive(Deserialize, Debug)]
struct AudioDeviceList {
    hosts: Vec<String>,
    devices: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
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

    // Available options
    let (models, set_models) = signal(Vec::<String>::new());
    let (loras, set_loras) = signal(Vec::<String>::new());
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
                set_backend.set(config.backend.clone());
                set_audio_host.set(config.audio_host.clone());
                set_audio_device.set(config.audio_device.clone());
                set_max_chars.set(config.max_chars);
                set_dit_steps.set(config.dit_steps);
                if config.language == "English" {
                    set_lang.set(Language::English);
                }
            }

            // List models
            if let Ok(model_list) = tauri_api::invoke_no_args::<Vec<String>>("list_models").await {
                if model_list.is_empty() {
                    set_no_model.set(true);
                    set_status.set("idle".into());
                    return;
                }
                let dir = model_dir.get_untracked();
                if dir.is_empty() || !model_list.contains(&dir) {
                    set_model_dir.set(model_list[0].clone());
                }
                set_models.set(model_list);
            }

            // List audio devices
            if let Ok(audio) = tauri_api::invoke_no_args::<AudioDeviceList>("list_audio_devices").await {
                set_hosts.set(audio.hosts);
                set_devices.set(audio.devices);
            }

            // List loras
            let md = model_dir.get_untracked();
            if let Ok(lora_list) = tauri_api::invoke::<_, Vec<String>>("list_lora_dirs", &ListLoraArgs { model_dir: md.clone() }).await {
                set_loras.set(lora_list);
            }

            // Load model
            set_status.set("loading".into());
            let md = model_dir.get_untracked();
            let be = backend.get_untracked();
            set_model_name.set(md.split('/').last().unwrap_or(&md).to_string());
            match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: md, backend: be }).await {
                Ok(_info) => {
                    set_engine_ready.set(true);
                    set_status.set("ready".into());
                    // Load LoRA if configured
                    let ld = lora_dir.get_untracked();
                    if ld != "None" && !ld.is_empty() {
                        let _ = tauri_api::invoke_unit("load_lora", &LoadLoraArgs { lora_dir: ld }).await;
                    }
                }
                Err(e) => {
                    web_sys::console::error_1(&format!("Model load error: {}", e).into());
                    set_status.set(format!("Error: {}", e));
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
                if let Ok(payload) = serde_wasm_bindgen::from_value::<ProgressPayload>(val) {
                    if payload.total > 0 {
                        set_progress.set(payload.step as f64 / payload.total as f64);
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
            let complete_cb = Closure::new(move |_val: JsValue| {
                set_status.set("ready".into());
                set_progress.set(0.0);
                // Mark latest generating entry as done
                set_history.update(|h| {
                    if let Some(entry) = h.iter_mut().rev().find(|e| e.status == "generating") {
                        entry.status = "done".into();
                        entry.progress = 1.0;
                    }
                });
            });
            let _ = tauri_api::tauri_listen("tts-complete", &complete_cb).await;
            complete_cb.forget();
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

        let steps = dit_steps.get_untracked() as u32;
        spawn_local(async move {
            if let Err(e) = tauri_api::invoke_unit("synthesize", &SynthesizeArgs {
                text: trimmed,
                dit_steps: steps,
                index: idx,
            }).await {
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
            match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: path, backend: be }).await {
                Ok(_) => {
                    set_engine_ready.set(true);
                    set_status.set("ready".into());
                }
                Err(e) => {
                    set_status.set(format!("Error: {}", e));
                }
            }
        });
    };

    let on_apply_settings = move |vals: SettingsValues| {
        set_show_settings.set(false);
        let need_reload = vals.model_dir != model_dir.get_untracked() || vals.backend != backend.get_untracked();

        set_model_dir.set(vals.model_dir.clone());
        set_lora_dir.set(vals.lora_dir.clone());
        set_backend.set(vals.backend.clone());
        set_audio_host.set(vals.audio_host.clone());
        set_audio_device.set(vals.audio_device.clone());
        set_max_chars.set(vals.max_chars);
        set_dit_steps.set(vals.dit_steps);

        // Save config
        {
            let config = serde_json::json!({
                "model_dir": vals.model_dir,
                "lora_dir": if vals.lora_dir == "None" { serde_json::Value::Null } else { serde_json::Value::String(vals.lora_dir.clone()) },
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
            let ld = vals.lora_dir.clone();
            set_model_name.set(md.split('/').last().unwrap_or(&md).to_string());
            spawn_local(async move {
                match tauri_api::invoke::<_, ModelInfo>("load_model", &LoadModelArgs { model_dir: md, backend: be }).await {
                    Ok(_) => {
                        set_engine_ready.set(true);
                        set_status.set("ready".into());
                        if ld != "None" && !ld.is_empty() {
                            let _ = tauri_api::invoke_unit("load_lora", &LoadLoraArgs { lora_dir: ld }).await;
                        }
                    }
                    Err(e) => set_status.set(format!("Error: {}", e)),
                }
            });
        }
    };

    view! {
        <div class="flex flex-col h-screen">
            <Header lang=lang on_settings=move |_| set_show_settings.set(true) />
            <History lang=lang entries=history />
            <ProgressBar progress=progress status=status lang=lang />
            <InputBox lang=lang engine_ready=engine_ready on_submit=on_submit />
            <StatusBar lang=lang status=status model_name=model_name backend=backend />
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
