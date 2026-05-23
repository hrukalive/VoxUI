use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::components::header::Header;
use crate::components::history::HistoryList;
use crate::components::input_box::InputBox;
use crate::components::load_progress_modal::LoadProgressModal;
use crate::components::settings_modal::SettingsModal;
use crate::i18n::{labels, UiLanguage};
use crate::tauri_api::{
    AppConfig, AppSnapshot, BackendKind, ConfigPatch, GenerationSettings, LanguageMode,
    LoadUiState,
};

#[component]
pub fn App() -> impl IntoView {
    let labels = labels(UiLanguage::Chinese);
    let (settings_open, set_settings_open) = signal(false);
    let (load_open, set_load_open) = signal(false);
    let (snapshot, set_snapshot) = signal(None::<AppSnapshot>);

    spawn_local(async move {
        if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
            set_snapshot.set(Some(next_snapshot));
        }
    });

    let current_snapshot = move || snapshot.get().unwrap_or_else(fallback_snapshot);
    let refresh_snapshot = move || {
        spawn_local(async move {
            if let Ok(next_snapshot) = crate::tauri_api::get_app_state().await {
                set_snapshot.set(Some(next_snapshot));
            }
        });
    };

    view! {
        <div class="app-shell">
            {move || {
                let snapshot = current_snapshot();
                view! {
                    <Header
                        labels=labels
                        models=snapshot.models
                        selected_model_id=snapshot.selected_model_id
                        loaded_model_id=snapshot.loaded_model_id
                        load_disabled=matches!(snapshot.load_state, LoadUiState::Loading)
                        on_model_select=|_| {}
                        on_load=move || set_load_open.set(true)
                        on_open_settings=move || set_settings_open.set(true)
                    />
                }
            }}
            {move || {
                let snapshot = current_snapshot();
                view! {
                    <HistoryList
                        labels=labels
                        items=snapshot.history
                        on_play=move |item_id| {
                            spawn_local(async move {
                                let _ = crate::tauri_api::play_audio(item_id).await;
                            });
                        }
                        on_regenerate=move |item_id| {
                            spawn_local(async move {
                                if crate::tauri_api::regenerate(item_id).await.is_ok() {
                                    refresh_snapshot();
                                }
                            });
                        }
                        on_cancel=|_| {}
                    />
                }
            }}
            {move || {
                let max_chars = current_snapshot().config.max_input_chars;
                view! {
                    <InputBox
                        labels=labels
                        max_chars=max_chars
                        disabled=false
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
            <SettingsModal
                labels=labels
                config=move || current_snapshot().config
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
                labels=labels
                open=move || load_open.get()
                percent=|| 42.0
                on_close=move || set_load_open.set(false)
            />
        </div>
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
