use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::components::header::Header;
use crate::components::history::HistoryList;
use crate::components::input_box::InputBox;
use crate::components::load_progress_modal::LoadProgressModal;
use crate::components::settings_modal::SettingsModal;
use crate::i18n::{labels, UiLanguage};
use crate::tauri_api::{
    AppConfig, AppSnapshot, AudioState, BackendKind, ConfigPatch, GenerationDoneEvent,
    GenerationProgressEvent, GenerationSettings, HistoryStatus, LanguageMode, LoadUiState,
    ModelLoadDoneEvent, ModelLoadProgressEvent, PlaybackStateEvent,
};

#[component]
pub fn App() -> impl IntoView {
    let (settings_open, set_settings_open) = signal(false);
    let (load_open, set_load_open) = signal(false);
    let (load_percent, set_load_percent) = signal(0.0_f32);
    let (snapshot, set_snapshot) = signal(None::<AppSnapshot>);
    let (audio_state, set_audio_state) = signal(AudioState::default());

    // Root component is mounted once; Tauri event listeners intentionally live for the app lifetime.
    spawn_local(async move {
        if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
            set_snapshot.set(Some(next_snapshot));
        }
    });
    spawn_local(async move {
        if let Ok(next_audio_state) = crate::tauri_api::get_audio_state().await {
            set_audio_state.set(next_audio_state);
        }
    });

    let current_snapshot = move || snapshot.get().unwrap_or_else(fallback_snapshot);
    let current_labels = move || labels(ui_language(current_snapshot().config.language));
    let refresh_snapshot = move || {
        spawn_local(async move {
            if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
                set_snapshot.set(Some(next_snapshot));
            }
        });
    };
    let cancel_load = move || {
        set_load_open.set(false);
        spawn_local(async move {
            let _ = crate::tauri_api::cancel_model_load().await;
            refresh_snapshot();
        });
    };

    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("model_load_progress", move |event| {
            if let Ok(progress) =
                crate::tauri_api::decode_app_event::<ModelLoadProgressEvent>(event)
            {
                set_load_percent.set(progress.percent());
                set_load_open.set(true);
            }
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("model_load_done", move |event| {
            let _ = crate::tauri_api::decode_app_event::<ModelLoadDoneEvent>(event);
            set_load_percent.set(100.0);
            set_load_open.set(false);
            refresh_snapshot();
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("generation_progress", move |event| {
            let _ = crate::tauri_api::decode_app_event::<GenerationProgressEvent>(event);
            refresh_snapshot();
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("generation_done", move |event| {
            let _ = crate::tauri_api::decode_app_event::<GenerationDoneEvent>(event);
            refresh_snapshot();
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("playback_state", move |event| {
            let _ = crate::tauri_api::decode_app_event::<PlaybackStateEvent>(event);
            refresh_snapshot();
        })
        .await;
    });

    view! {
        <div class="app-shell">
            {move || {
                let snapshot = current_snapshot();
                let labels = labels(ui_language(snapshot.config.language));
                let selected_model_id = snapshot.selected_model_id.clone();
                let loaded_model_id = snapshot.loaded_model_id.clone();
                let load_disabled = selected_model_id.is_none()
                    || selected_model_id == loaded_model_id
                    || matches!(snapshot.load_state, LoadUiState::Loading);
                view! {
                    <Header
                        labels=labels
                        models=snapshot.models
                        selected_model_id=snapshot.selected_model_id
                        loaded_model_id=snapshot.loaded_model_id
                        load_disabled=load_disabled
                        on_model_select=move |model_id| {
                            spawn_local(async move {
                                let selected_model_id = if model_id.is_empty() {
                                    None
                                } else {
                                    Some(model_id)
                                };
                                if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(ConfigPatch {
                                    selected_model_id: Some(selected_model_id),
                                    ..ConfigPatch::default()
                                })
                                .await
                                {
                                    set_snapshot.set(Some(next_snapshot));
                                }
                            });
                        }
                        on_load=move || {
                            if let Some(choice_id) = current_snapshot().selected_model_id {
                                set_load_percent.set(0.0);
                                set_load_open.set(true);
                                spawn_local(async move {
                                    if crate::tauri_api::load_model(choice_id).await.is_err() {
                                        set_load_open.set(false);
                                    }
                                    refresh_snapshot();
                                });
                            }
                        }
                        on_open_settings=move || set_settings_open.set(true)
                    />
                }
            }}
            {move || {
                let snapshot = current_snapshot();
                let labels = labels(ui_language(snapshot.config.language));
                view! {
                    <HistoryList
                        labels=labels
                        items=snapshot.history
                        on_play=move |item_id| {
                            spawn_local(async move {
                                let is_playing = current_snapshot()
                                    .history
                                    .iter()
                                    .any(|item| item.id == item_id && matches!(item.status, HistoryStatus::Playing));
                                let result = if is_playing {
                                    crate::tauri_api::stop_audio().await
                                } else {
                                    crate::tauri_api::play_audio(item_id).await
                                };
                                if result.is_ok() {
                                    refresh_snapshot();
                                }
                            });
                        }
                        on_regenerate=move |item_id| {
                            spawn_local(async move {
                                if crate::tauri_api::regenerate(item_id).await.is_ok() {
                                    refresh_snapshot();
                                }
                            });
                        }
                        on_cancel=move |item_id| {
                            spawn_local(async move {
                                let _ = crate::tauri_api::cancel_generation(item_id).await;
                                refresh_snapshot();
                            });
                        }
                    />
                }
            }}
            {move || {
                let snapshot = current_snapshot();
                let labels = labels(ui_language(snapshot.config.language));
                let max_chars = snapshot.config.max_input_chars;
                let generate_disabled =
                    snapshot.loaded_model_id.is_none() || matches!(snapshot.load_state, LoadUiState::Loading);
                view! {
                    <InputBox
                        labels=labels
                        max_chars=max_chars
                        disabled=generate_disabled
                        on_generate=move |text| {
                            spawn_local(async move {
                                if crate::tauri_api::enqueue_generation(text).await.is_ok() {
                                    refresh_snapshot();
                                }
                            });
                        }
                    />
                }
            }}
            {move || {
                view! {
                    <SettingsModal
                        labels=current_labels()
                        config=move || current_snapshot().config
                        audio_state=move || audio_state.get()
                        open=move || settings_open.get()
                        on_close=move || set_settings_open.set(false)
                        on_config_patch=move |patch| {
                            spawn_local(async move {
                                if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(patch).await {
                                    set_snapshot.set(Some(next_snapshot));
                                }
                            });
                        }
                        on_browse_model_dir=move || {
                            spawn_local(async move {
                                if let Ok(Some(path)) = crate::tauri_api::browse_model_dir().await {
                                    if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(ConfigPatch {
                                        model_root: Some(Some(path)),
                                        ..ConfigPatch::default()
                                    })
                                    .await
                                    {
                                        set_snapshot.set(Some(next_snapshot));
                                    }
                                }
                            });
                        }
                        on_browse_prompt_wav=move || {
                            spawn_local(async move {
                                if let Ok(Some(path)) = crate::tauri_api::browse_prompt_wav().await {
                                    let mut generation = current_snapshot().config.generation;
                                    generation.prompt_wav_path = Some(path);
                                    if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(ConfigPatch {
                                        generation: Some(generation),
                                        ..ConfigPatch::default()
                                    })
                                    .await
                                    {
                                        set_snapshot.set(Some(next_snapshot));
                                    }
                                }
                            });
                        }
                        on_browse_reference_wav=move || {
                            spawn_local(async move {
                                if let Ok(Some(path)) = crate::tauri_api::browse_reference_wav().await {
                                    let mut generation = current_snapshot().config.generation;
                                    generation.reference_wav_path = Some(path);
                                    if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(ConfigPatch {
                                        generation: Some(generation),
                                        ..ConfigPatch::default()
                                    })
                                    .await
                                    {
                                        set_snapshot.set(Some(next_snapshot));
                                    }
                                }
                            });
                        }
                        on_test_audio=move || {
                            spawn_local(async move {
                                let _ = crate::tauri_api::test_audio().await;
                            });
                        }
                    />
                    <LoadProgressModal
                        labels=current_labels()
                        open=move || load_open.get() && matches!(current_snapshot().load_state, LoadUiState::Loading)
                        percent=move || load_percent.get()
                        on_close=cancel_load
                    />
                }
            }}
        </div>
    }
}

fn ui_language(language: LanguageMode) -> UiLanguage {
    match language {
        LanguageMode::English => UiLanguage::English,
        LanguageMode::Chinese | LanguageMode::System => UiLanguage::Chinese,
    }
}

fn fallback_snapshot() -> AppSnapshot {
    AppSnapshot {
        config: AppConfig {
            model_root: None,
            selected_model_id: None,
            language: LanguageMode::System,
            backend: BackendKind::Cpu,
            audio_host: None,
            audio_device: None,
            volume: 0.8,
            max_input_chars: 280,
            generation: GenerationSettings {
                cfg_value: 2.0,
                inference_timesteps: 10,
                min_len: 2,
                max_len: 2000,
                retry_badcase: true,
                retry_badcase_max_times: 3,
                retry_badcase_ratio_threshold: 6.0,
                prompt_wav_path: None,
                prompt_text: None,
                reference_wav_path: None,
            },
        },
        models: Vec::new(),
        selected_model_id: None,
        loaded_model_id: None,
        load_state: LoadUiState::Idle,
        history: Vec::new(),
    }
}
