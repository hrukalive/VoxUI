use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::components::error_modal::ErrorModal;
use crate::components::header::Header;
use crate::components::history::HistoryList;
use crate::components::input_box::InputBox;
use crate::components::live_monitor::LiveMonitor;
use crate::components::load_progress_modal::LoadProgressModal;
use crate::components::settings_modal::{SettingsModal, SettingsPage};
use crate::components::translation_bar::TranslationBar;
use crate::i18n::{labels, UiLanguage};
use crate::tauri_api::{
    AppConfig, AppSnapshot, AudioState, AutoGenMode, BackendKind, ConfigPatch, GenerationDoneEvent,
    GenerationProgressEvent, GenerationSettings, HistoryStatus, LanguageMode, LiveConfig,
    LiveConfigPatch, LiveSnapshot, LiveStatus, LoadUiState, ModelLoadDoneEvent,
    ModelLoadProgressEvent, PlaybackStateEvent, ReplacementRule, SendMode, TemplateConfig,
    ThemeMode, TranslationPair, TranslationSettings,
};

#[component]
pub fn App() -> impl IntoView {
    let (settings_open, set_settings_open) = signal(false);
    let (settings_page, set_settings_page) = signal(SettingsPage::General);
    let (load_open, set_load_open) = signal(false);
    let (load_percent, set_load_percent) = signal(0.0_f32);
    let (sidecar_ready, set_sidecar_ready) = signal(false);
    let (sidecar_error, set_sidecar_error) = signal(None::<String>);
    let (load_error, set_load_error) = signal(None::<String>);
    let (live_error, set_live_error) = signal(None::<String>);
    let (snapshot, set_snapshot) = signal(Some(fallback_snapshot()));
    let (live_snapshot, set_live_snapshot) = signal(fallback_live_snapshot());
    let (audio_state, set_audio_state) = signal(AudioState::default());
    let (input_replacement, set_input_replacement) = signal(None::<String>);
    let (input_text, set_input_text) = signal(String::new());
    let (translation_error, set_translation_error) = signal(None::<String>);
    let (volume_preview, set_volume_preview) = signal(fallback_snapshot().config.volume);
    let (volume_adjusting, set_volume_adjusting) = signal(false);

    // Root component is mounted once; Tauri event listeners intentionally live for the app lifetime.
    spawn_local(async move {
        if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
            if let Some(error) = next_snapshot.sidecar_init_error.clone() {
                set_sidecar_error.set(Some(error));
            }
            set_snapshot.set(Some(next_snapshot));
        }
    });
    spawn_local(async move {
        if let Ok(next_audio_state) = crate::tauri_api::get_audio_state().await {
            set_audio_state.set(next_audio_state);
        }
    });
    spawn_local(async move {
        if let Ok(next_live_snapshot) = crate::tauri_api::get_live_state().await {
            set_live_snapshot.set(next_live_snapshot);
        }
    });

    let current_snapshot = move || snapshot.get().unwrap_or_else(fallback_snapshot);
    let current_snapshot_untracked =
        move || snapshot.get_untracked().unwrap_or_else(fallback_snapshot);
    let current_labels = move || {
        let snapshot = current_snapshot();
        labels(ui_language(
            snapshot.config.language,
            snapshot.system_language,
        ))
    };
    Effect::new(move |_| {
        let volume = current_snapshot().config.volume;
        if !volume_adjusting.get_untracked() {
            set_volume_preview.set(volume);
        }
    });
    let refresh_snapshot = move || {
        spawn_local(async move {
            if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
                set_snapshot.set(Some(next_snapshot));
            }
        });
    };
    let refresh_live_snapshot = move || {
        spawn_local(async move {
            if let Ok(next_live_snapshot) = crate::tauri_api::get_live_state().await {
                set_live_snapshot.set(next_live_snapshot);
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
            if let Ok(done) = crate::tauri_api::decode_app_event::<ModelLoadDoneEvent>(event) {
                if done.status != "success" && done.status != "canceled" {
                    set_load_error.set(Some(done.error.unwrap_or_else(|| done.status)));
                }
            }
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
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("sidecar_capabilities", move |_| {
            set_sidecar_ready.set(true);
            refresh_snapshot();
        })
        .await;
        refresh_snapshot();
        set_sidecar_ready.set(true);
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("app_config_changed", move |event| {
            if let Ok(snapshot) = crate::tauri_api::decode_app_event::<AppSnapshot>(event) {
                set_snapshot.set(Some(snapshot));
            } else {
                refresh_snapshot();
            }
            refresh_live_snapshot();
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("main_input_replace", move |event| {
            if let Ok(payload) =
                crate::tauri_api::decode_app_event::<crate::tauri_api::MainInputReplaceEvent>(event)
            {
                set_input_replacement.set(Some(payload.text));
            }
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("live_status_changed", move |event| {
            if let Ok(snapshot) = crate::tauri_api::decode_app_event::<LiveSnapshot>(event) {
                if snapshot.status == LiveStatus::Error {
                    if let Some(ref message) = snapshot.status_message {
                        set_live_error.set(Some(message.clone()));
                    }
                }
                set_live_snapshot.set(snapshot);
            }
        })
        .await;
    });
    spawn_local(async move {
        let _ = crate::tauri_api::listen_app_event("live_items_changed", move |event| {
            if let Ok(snapshot) = crate::tauri_api::decode_app_event::<LiveSnapshot>(event) {
                set_live_snapshot.set(snapshot);
            }
        })
        .await;
    });

    let commit_config_patch = move |patch: ConfigPatch| {
        set_snapshot.update(|snapshot| {
            if let Some(snapshot) = snapshot.as_mut() {
                apply_optimistic_patch(snapshot, &patch);
            }
        });
        spawn_local(async move {
            if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(patch).await {
                set_snapshot.set(Some(next_snapshot));
            }
        });
    };
    let commit_live_patch = move |patch: LiveConfigPatch| {
        set_live_snapshot.update(|snapshot| apply_live_optimistic_patch(snapshot, &patch));
        spawn_local(async move {
            if let Ok(next_snapshot) = crate::tauri_api::set_live_config_patch(patch).await {
                set_live_snapshot.set(next_snapshot);
            }
        });
    };
    let preview_volume = move |volume: f32| {
        let volume = volume.clamp(0.0, 1.0);
        set_volume_adjusting.set(true);
        set_volume_preview.set(volume);
        spawn_local(async move {
            let _ = crate::tauri_api::set_runtime_volume(volume).await;
        });
    };
    let commit_volume = move |volume: f32| {
        let volume = volume.clamp(0.0, 1.0);
        set_volume_adjusting.set(false);
        set_volume_preview.set(volume);
        commit_config_patch(ConfigPatch {
            volume: Some(volume),
            ..ConfigPatch::default()
        });
    };

    let current_ui_language = move || {
        let snapshot = current_snapshot();
        ui_language(snapshot.config.language, snapshot.system_language)
    };

    let is_live_monitor_window = crate::tauri_api::current_window_label() == "live-monitor";
    if is_live_monitor_window {
        return view! {
            <div
                class:theme-light=move || current_snapshot().config.theme == ThemeMode::Light
                class:theme-dark=move || current_snapshot().config.theme == ThemeMode::Dark
            >
                <LiveMonitor
                    labels=current_labels
                    snapshot=move || live_snapshot.get()
                    on_live_patch=commit_live_patch
                    on_send=move |item_id, switch, enqueue_direct| {
                        spawn_local(async move {
                            let mode = if switch { "switch" } else { "normal" }.to_string();
                            if enqueue_direct {
                                match crate::tauri_api::send_live_suggestion(item_id, mode, true).await {
                                    Ok(result) => {
                                        let _ = crate::tauri_api::enqueue_generation(result.text).await;
                                    }
                                    _ => {}
                                }
                            } else {
                                let _ = crate::tauri_api::send_live_suggestion(item_id, mode, false).await;
                            }
                        });
                    }
                    on_clear=move || {
                        spawn_local(async move {
                            if let Ok(next_snapshot) = crate::tauri_api::clear_live_items().await {
                                set_live_snapshot.set(next_snapshot);
                            }
                        });
                    }
                    translation_config=move || current_snapshot().config.translation.clone()
                    on_translation_patch=move |translation| {
                        commit_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </div>
        }
        .into_any();
    }

    view! {
        <div
            class="app-shell"
            class:theme-light=move || current_snapshot().config.theme == ThemeMode::Light
            class:theme-dark=move || current_snapshot().config.theme == ThemeMode::Dark
        >
            <Show when=move || !sidecar_ready.get()>
                <div class="modal-backdrop" role="presentation">
                    <section class="modal startup-modal" role="dialog" aria-modal="true">
                        <header class="modal-header">
                            <h2>{move || current_labels().loading}</h2>
                        </header>
                        <p>{move || current_labels().loading}</p>
                    </section>
                </div>
            </Show>
            {move || {
                let snapshot = current_snapshot();
                let labels = labels(ui_language(
                    snapshot.config.language,
                    snapshot.system_language,
                ));
                let selected_model_id = snapshot.selected_model_id.clone();
                let loaded_model_id = snapshot.loaded_model_id.clone();
                let load_disabled = selected_model_id.is_none()
                    || selected_model_id == loaded_model_id
                    || matches!(snapshot.load_state, LoadUiState::Loading);
                let volume = volume_preview.get();
                view! {
                    <Header
                        labels=labels
                        models=snapshot.models
                        selected_model_id=snapshot.selected_model_id
                        load_disabled=load_disabled
                        volume=volume
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
                            if let Some(choice_id) = current_snapshot_untracked().selected_model_id {
                                set_load_error.set(None);
                                set_load_percent.set(0.0);
                                set_load_open.set(true);
                                spawn_local(async move {
                                    if let Err(error) = crate::tauri_api::load_model(choice_id).await {
                                        set_load_error.set(Some(error));
                                        set_load_open.set(false);
                                    }
                                    refresh_snapshot();
                                });
                            }
                        }
                        on_volume_input=preview_volume
                        on_volume_commit=commit_volume
                        on_open_settings=move || set_settings_open.set(true)
                        on_open_live_monitor=move || {
                            spawn_local(async move {
                                if let Err(error) = crate::tauri_api::show_live_monitor().await {
                                    set_live_error.set(Some(error));
                                }
                            });
                            let live_snapshot = live_snapshot.get_untracked();
                            let identity_code = live_snapshot.config.identity_code.trim().to_string();
                            if identity_code.is_empty() {
                                set_live_error.set(Some("Identity code not provided".to_string()));
                                return;
                            }
                            if matches!(live_snapshot.status, LiveStatus::Connected | LiveStatus::Connecting | LiveStatus::Disconnecting) {
                                return;
                            }
                            spawn_local(async move {
                                match crate::tauri_api::connect_openblive(identity_code).await {
                                    Ok(next_snapshot) => set_live_snapshot.set(next_snapshot),
                                    Err(error) => set_live_error.set(Some(error)),
                                }
                            });
                        }
                    />
                }
            }}
            <HistoryList
                labels=current_labels
                items=move || current_snapshot().history
                on_play=move |item_id| {
                    spawn_local(async move {
                        let is_playing = current_snapshot_untracked()
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
            <InputBox
                labels=current_labels
                language=current_ui_language
                max_chars=move || current_snapshot().config.max_input_chars
                auto_period=move || current_snapshot().config.auto_period
                disabled=move || {
                    let snapshot = current_snapshot();
                    snapshot.loaded_model_id.is_none() || matches!(snapshot.load_state, LoadUiState::Loading)
                }
                replacement_text=move || input_replacement.get()
                on_replacement_consumed=move || set_input_replacement.set(None)
                on_text_change=move |text| set_input_text.set(text)
                translation_bar={
                    view! {
                        <TranslationBar
                            labels=current_labels
                            config=move || current_snapshot().config.clone()
                            input_text=move || input_text.get()
                            disabled=move || {
                                let snapshot = current_snapshot();
                                snapshot.loaded_model_id.is_none() || matches!(snapshot.load_state, LoadUiState::Loading)
                            }
                            on_replace_text=move |text| set_input_replacement.set(Some(text))
                            on_enqueue=move |text| {
                                let snapshot = current_snapshot_untracked();
                                let target_lang = snapshot.config.translation.outbound.target_lang.clone();
                                let final_text = if snapshot.config.auto_period {
                                    ensure_period(&text, &target_lang)
                                } else {
                                    text
                                };
                                set_input_text.set(String::new());
                                set_input_replacement.set(Some(String::new()));
                                spawn_local(async move {
                                    if crate::tauri_api::enqueue_generation(final_text).await.is_ok() {
                                        refresh_snapshot();
                                    }
                                });
                            }
                            on_error=move |err| set_translation_error.set(Some(err))
                            on_config_patch=commit_config_patch
                        />
                    }.into_any()
                }
                on_generate=move |text| {
                    spawn_local(async move {
                        if crate::tauri_api::enqueue_generation(text).await.is_ok() {
                            refresh_snapshot();
                        }
                    });
                }
            />
            <SettingsModal
                labels=current_labels
                language=current_ui_language
                config=move || current_snapshot().config
                volume=move || volume_preview.get()
                live_snapshot=move || live_snapshot.get()
                cuda_available=move || current_snapshot().cuda_available
                audio_state=move || audio_state.get()
                open=move || settings_open.get()
                active_page=move || settings_page.get()
                on_close=move || set_settings_open.set(false)
                on_page_select=move |page| set_settings_page.set(page)
                on_config_patch=commit_config_patch
                on_volume_input=preview_volume
                on_volume_commit=commit_volume
                on_live_patch=commit_live_patch
                on_live_connect=move || {
                    spawn_local(async move {
                        if let Err(error) = crate::tauri_api::show_live_monitor().await {
                            set_live_error.set(Some(error));
                        }
                    });
                    let live_snapshot = live_snapshot.get_untracked();
                    let identity_code = live_snapshot.config.identity_code.trim().to_string();
                    if identity_code.is_empty() {
                        set_live_error.set(Some("Identity code not provided".to_string()));
                        return;
                    }
                    if matches!(live_snapshot.status, LiveStatus::Connected | LiveStatus::Connecting | LiveStatus::Disconnecting) {
                        return;
                    }
                    spawn_local(async move {
                        match crate::tauri_api::connect_openblive(identity_code).await {
                            Ok(next_snapshot) => set_live_snapshot.set(next_snapshot),
                            Err(error) => set_live_error.set(Some(error)),
                        }
                    });
                }
                on_live_disconnect=move || {
                    spawn_local(async move {
                        if let Ok(next_snapshot) = crate::tauri_api::disconnect_openblive().await {
                            set_live_snapshot.set(next_snapshot);
                        }
                    });
                }
                on_live_mock_message=move |kind| {
                    spawn_local(async move {
                        if let Ok(next_snapshot) = crate::tauri_api::mock_live_message(kind).await {
                            set_live_snapshot.set(next_snapshot);
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
                            let mut generation = current_snapshot_untracked().config.generation;
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
                            let mut generation = current_snapshot_untracked().config.generation;
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
            {move || {
                view! {
                    <LoadProgressModal
                        labels=current_labels()
                        open=move || load_open.get() && matches!(current_snapshot().load_state, LoadUiState::Loading)
                        percent=move || load_percent.get()
                        on_close=cancel_load
                    />
                }
            }}
            {move || {
                let labels = current_labels();
                view! {
                    <ErrorModal
                        labels=labels
                        open=move || load_error.get().is_some()
                        title=move || current_labels().model_load_failed.to_string()
                        message=move || load_error.get().unwrap_or_default()
                        on_close=move || set_load_error.set(None)
                    />
                }
            }}
            {move || {
                let labels = current_labels();
                view! {
                    <ErrorModal
                        labels=labels
                        open=move || live_error.get().is_some()
                        title=move || current_labels().connection_failed.to_string()
                        message=move || live_error.get().unwrap_or_default()
                        danger=true
                        on_close=move || set_live_error.set(None)
                    />
                }
            }}
            {move || {
                let labels = current_labels();
                view! {
                    <ErrorModal
                        labels=labels
                        open=move || sidecar_error.get().is_some()
                        title=move || current_labels().sidecar_load_failed.to_string()
                        message=move || sidecar_error.get().unwrap_or_default()
                        danger=true
                        on_close=move || {
                            set_sidecar_error.set(None);
                            spawn_local(async move {
                                let _ = crate::tauri_api::exit_app().await;
                            });
                        }
                    />
                }
            }}
            {move || {
                let labels = current_labels();
                view! {
                    <ErrorModal
                        labels=labels
                        open=move || translation_error.get().is_some()
                        title=move || current_labels().translation_failed.to_string()
                        message=move || translation_error.get().unwrap_or_default()
                        on_close=move || set_translation_error.set(None)
                    />
                }
            }}
        </div>
    }
    .into_any()
}

fn ui_language(language: LanguageMode, system_language: LanguageMode) -> UiLanguage {
    match language {
        LanguageMode::English => UiLanguage::English,
        LanguageMode::Chinese => UiLanguage::Chinese,
        LanguageMode::System => match system_language {
            LanguageMode::Chinese => UiLanguage::Chinese,
            LanguageMode::English | LanguageMode::System => UiLanguage::English,
        },
    }
}

fn fallback_snapshot() -> AppSnapshot {
    AppSnapshot {
        config: AppConfig {
            model_root: None,
            selected_model_id: None,
            language: LanguageMode::System,
            theme: ThemeMode::Dark,
            backend: BackendKind::Cpu,
            audio_host: None,
            audio_device: None,
            volume: 0.8,
            max_input_chars: 280,
            auto_period: true,
            generation: GenerationSettings {
                cfg_value: 2.0,
                inference_timesteps: 10,
                min_len: 2,
                max_len: 2000,
                streaming: false,
                stream_consolidate_n: 10,
                retry_badcase: true,
                retry_badcase_max_times: 3,
                retry_badcase_ratio_threshold: 6.0,
                prompt_wav_path: None,
                prompt_text: None,
                reference_wav_path: None,
            },
            translation: TranslationSettings {
                outbound: TranslationPair {
                    source_lang: "auto".into(),
                    target_lang: "EN".into(),
                },
                inbound: TranslationPair {
                    source_lang: "auto".into(),
                    target_lang: "ZH".into(),
                },
                translate_enqueue: false,
            },
            live: fallback_live_config(),
        },
        system_language: LanguageMode::English,
        cuda_available: false,
        sidecar_init_error: None,
        models: Vec::new(),
        selected_model_id: None,
        loaded_model_id: None,
        load_state: LoadUiState::Idle,
        history: Vec::new(),
    }
}

fn fallback_live_config() -> LiveConfig {
    LiveConfig {
        identity_code: String::new(),
        enable_ceve_server_heartbeat: false,
        show_danmu: true,
        show_gifts: true,
        show_superchats: true,
        show_guards: true,
        show_likes: false,
        show_enters: true,
        send_mode: SendMode::Manual,
        auto_gen_mode: AutoGenMode::None,
        auto_gen_danmu: false,
        auto_gen_gifts: true,
        auto_gen_superchats: true,
        auto_gen_guards: true,
        auto_gen_likes: false,
        auto_gen_enters: true,
        templates: TemplateConfig {
            danmu: "{msg}".to_string(),
            gift_zh: "感谢{mapped_uname}送出的{gift_num}个{gift_name}".to_string(),
            gift_en: "Thank you {mapped_uname} for {gift_num} {gift_name}".to_string(),
            superchat_zh: "感谢{mapped_uname}的SC：{message}".to_string(),
            superchat_en: "Thank you {mapped_uname} for the superchat saying {message}".to_string(),
            guard_zh: "感谢{mapped_uname}开通的{guard_label}".to_string(),
            guard_en: "Thank you {mapped_uname} for joining as {guard_label}".to_string(),
            like_zh: "感谢{mapped_uname}给直播间点赞".to_string(),
            like_en: "Thank you {mapped_uname} for liking the stream".to_string(),
            enter_zh: "欢迎{mapped_uname}进入直播间".to_string(),
            enter_en: "Hi {mapped_uname}, welcome to the stream".to_string(),
        },
        replacement_rules: vec![
            ReplacementRule {
                enabled: true,
                from: "我".to_string(),
                to: "你".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "I".to_string(),
                to: "you".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "me".to_string(),
                to: "you".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "my".to_string(),
                to: "your".to_string(),
            },
        ],
        mapped_unames: Default::default(),
        original_unames: Default::default(),
    }
}

fn fallback_live_snapshot() -> LiveSnapshot {
    LiveSnapshot {
        status: LiveStatus::Disconnected,
        status_message: None,
        config: fallback_live_config(),
        items: Vec::new(),
    }
}

fn apply_optimistic_patch(snapshot: &mut AppSnapshot, patch: &ConfigPatch) {
    if let Some(model_root) = patch.model_root.as_ref() {
        snapshot.config.model_root = model_root.clone();
    }
    if let Some(selected_model_id) = patch.selected_model_id.as_ref() {
        snapshot.selected_model_id = selected_model_id.clone();
        snapshot.config.selected_model_id = selected_model_id.clone();
    }
    if let Some(language) = patch.language {
        snapshot.config.language = language;
    }
    if let Some(theme) = patch.theme {
        snapshot.config.theme = theme;
    }
    if let Some(backend) = patch.backend {
        snapshot.config.backend = if snapshot.cuda_available {
            backend
        } else {
            BackendKind::Cpu
        };
    }
    if let Some(audio_host) = patch.audio_host.as_ref() {
        let audio_host = empty_string_as_none(audio_host.clone());
        if snapshot.config.audio_host != audio_host {
            snapshot.config.audio_device = None;
        }
        snapshot.config.audio_host = audio_host;
    }
    if let Some(audio_device) = patch.audio_device.as_ref() {
        snapshot.config.audio_device = empty_string_as_none(audio_device.clone());
    }
    if let Some(volume) = patch.volume {
        snapshot.config.volume = volume.clamp(0.0, 1.0);
    }
    if let Some(max_input_chars) = patch.max_input_chars {
        snapshot.config.max_input_chars = max_input_chars.max(1);
    }
    if let Some(auto_period) = patch.auto_period {
        snapshot.config.auto_period = auto_period;
    }
    if let Some(generation) = patch.generation.as_ref() {
        snapshot.config.generation = generation.clone();
    }
    if let Some(translation) = patch.translation.as_ref() {
        snapshot.config.translation = translation.clone();
    }
}

fn empty_string_as_none(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

fn ensure_period(text: &str, target_lang: &str) -> String {
    let period = match target_lang {
        "ZH" | "ZH-HANT" | "JA" => '。',
        _ => '.',
    };
    let endings = ['?', '!', '.', '…', '？', '！', '。'];
    if text.is_empty() || text.ends_with(&endings) {
        text.to_string()
    } else {
        format!("{}{}", text, period)
    }
}

fn apply_live_optimistic_patch(snapshot: &mut LiveSnapshot, patch: &LiveConfigPatch) {
    if let Some(identity_code) = patch.identity_code.as_ref() {
        snapshot.config.identity_code = identity_code.clone();
    }
    if let Some(enable_ceve_server_heartbeat) = patch.enable_ceve_server_heartbeat {
        snapshot.config.enable_ceve_server_heartbeat = enable_ceve_server_heartbeat;
    }
    if let Some(show_danmu) = patch.show_danmu {
        snapshot.config.show_danmu = show_danmu;
    }
    if let Some(show_gifts) = patch.show_gifts {
        snapshot.config.show_gifts = show_gifts;
    }
    if let Some(show_superchats) = patch.show_superchats {
        snapshot.config.show_superchats = show_superchats;
    }
    if let Some(show_guards) = patch.show_guards {
        snapshot.config.show_guards = show_guards;
    }
    if let Some(show_likes) = patch.show_likes {
        snapshot.config.show_likes = show_likes;
    }
    if let Some(show_enters) = patch.show_enters {
        snapshot.config.show_enters = show_enters;
    }
    if let Some(send_mode) = patch.send_mode {
        snapshot.config.send_mode = send_mode;
    }
    if let Some(auto_gen_mode) = patch.auto_gen_mode {
        snapshot.config.auto_gen_mode = auto_gen_mode;
    }
    if let Some(auto_gen_danmu) = patch.auto_gen_danmu {
        snapshot.config.auto_gen_danmu = auto_gen_danmu;
    }
    if let Some(auto_gen_gifts) = patch.auto_gen_gifts {
        snapshot.config.auto_gen_gifts = auto_gen_gifts;
    }
    if let Some(auto_gen_superchats) = patch.auto_gen_superchats {
        snapshot.config.auto_gen_superchats = auto_gen_superchats;
    }
    if let Some(auto_gen_guards) = patch.auto_gen_guards {
        snapshot.config.auto_gen_guards = auto_gen_guards;
    }
    if let Some(auto_gen_likes) = patch.auto_gen_likes {
        snapshot.config.auto_gen_likes = auto_gen_likes;
    }
    if let Some(auto_gen_enters) = patch.auto_gen_enters {
        snapshot.config.auto_gen_enters = auto_gen_enters;
    }
    if let Some(templates) = patch.templates.as_ref() {
        snapshot.config.templates = templates.clone();
    }
    if let Some(replacement_rules) = patch.replacement_rules.as_ref() {
        snapshot.config.replacement_rules = replacement_rules.clone();
    }
    let mapped_unames_changed = patch.mapped_unames.is_some();
    if let Some(mapped_unames) = patch.mapped_unames.as_ref() {
        snapshot.config.mapped_unames = mapped_unames.clone();
    }
    if let Some(original_unames) = patch.original_unames.as_ref() {
        snapshot.config.original_unames = original_unames.clone();
    } else if mapped_unames_changed {
        for item in &snapshot.items {
            if snapshot.config.mapped_unames.contains_key(&item.open_id) {
                snapshot
                    .config
                    .original_unames
                    .insert(item.open_id.clone(), item.uname.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    fn closure_body<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source
            .find(start)
            .unwrap_or_else(|| panic!("missing closure start: {start}"));
        let tail = &source[start_index..];
        let end_index = tail
            .find(end)
            .unwrap_or_else(|| panic!("missing closure end after {start}: {end}"));
        &tail[..end_index]
    }

    #[test]
    fn history_list_is_not_remounted_by_snapshot_refreshes() {
        let source = include_str!("app.rs").replace("\r\n", "\n");
        let snapshot_wrapped_history = "let snapshot = current_snapshot();\n                let labels = labels(ui_language(\n                    snapshot.config.language,\n                    snapshot.system_language,\n                ));\n                view! {\n                    <HistoryList";

        assert!(
            !source.contains(snapshot_wrapped_history),
            "HistoryList must stay mounted across snapshot refreshes so progress updates do not reset its scroll state"
        );
    }

    #[test]
    fn monitor_button_shows_window_before_connection_checks() {
        let source = include_str!("app.rs").replace("\r\n", "\n");
        let body = closure_body(
            &source,
            "on_open_live_monitor=move || {",
            "                    />",
        );
        let show_index = body
            .find("crate::tauri_api::show_live_monitor().await")
            .expect("monitor button must show the live monitor window");
        let identity_index = body
            .find("identity_code.is_empty()")
            .expect("monitor button should still validate identity before connecting");
        let connect_index = body
            .find("crate::tauri_api::connect_openblive(identity_code).await")
            .expect("monitor button should connect when needed");

        assert!(
            show_index < identity_index && show_index < connect_index,
            "monitor button must show the window before identity/status connection decisions"
        );
    }

    #[test]
    fn settings_live_connect_shows_monitor_before_connecting() {
        let source = include_str!("app.rs").replace("\r\n", "\n");
        let body = closure_body(
            &source,
            "on_live_connect=move || {",
            "                }\n                on_live_disconnect=",
        );
        let show_index = body
            .find("crate::tauri_api::show_live_monitor().await")
            .expect("settings connect must show the live monitor window");
        let connect_index = body
            .find("crate::tauri_api::connect_openblive(identity_code).await")
            .expect("settings connect should still connect to OpenLive");

        assert!(
            show_index < connect_index,
            "settings connect must show the monitor window before connecting"
        );
    }

    #[test]
    fn live_connect_handlers_do_not_suppress_connecting_snapshots() {
        let source = include_str!("app.rs").replace("\r\n", "\n");
        let monitor_body = closure_body(
            &source,
            "on_open_live_monitor=move || {",
            "                    />",
        );
        let settings_body = closure_body(
            &source,
            "on_live_connect=move || {",
            "                }\n                on_live_disconnect=",
        );

        assert!(
            !monitor_body.contains("LiveStatus::Connecting"),
            "monitor button should still call connect_openblive for a Connecting snapshot; backend worker guard handles true duplicates"
        );
        assert!(
            !settings_body.contains("LiveStatus::Connecting"),
            "settings connect should still call connect_openblive for a Connecting snapshot; backend worker guard handles true duplicates"
        );
    }

    #[test]
    fn app_listens_for_config_and_live_snapshot_refresh_events() {
        let source = include_str!("app.rs").replace("\r\n", "\n");
        let implementation = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("app implementation source");
        let app_config_changed = ["app", "_config", "_changed"].concat();
        let refresh_live_snapshot = ["refresh", "_live", "_snapshot"].concat();

        assert!(
            implementation.contains(&app_config_changed),
            "all windows should listen for app config changes so language/theme updates reach the live monitor"
        );
        assert!(
            implementation.contains(&refresh_live_snapshot),
            "config changes that affect live rendering should refresh live snapshots"
        );
    }
}
