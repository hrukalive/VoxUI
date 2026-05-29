use leptos::prelude::*;
use std::collections::BTreeMap;

use crate::components::controls::{CustomSelect, NumberCounter, SelectOption};
use crate::i18n::{Labels, UiLanguage};
use crate::tauri_api::{
    AppConfig, AudioDevice, AudioHost, AudioState, BackendKind, ConfigPatch, GenerationSettings,
    LanguageMode, LiveConfigPatch, LiveMessageKind, LiveSnapshot, LiveStatus, ReplacementRule,
    TemplateConfig, ThemeMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsPage {
    General,
    Inference,
    Audio,
    Live,
    About,
}

const DEFAULT_AUDIO_CHOICE: &str = "__voxui_default_audio__";

#[component]
pub fn SettingsModal(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    language: impl Fn() -> UiLanguage + Send + Sync + 'static + Copy,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    live_snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    cuda_available: impl Fn() -> bool + Send + Sync + 'static + Copy,
    audio_state: impl Fn() -> AudioState + Send + Sync + 'static + Copy,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    active_page: impl Fn() -> SettingsPage + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
    on_page_select: impl Fn(SettingsPage) + Send + Sync + 'static + Copy,
    on_config_patch: impl Fn(ConfigPatch) + Send + Sync + 'static + Copy,
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    on_live_connect: impl Fn() + Send + Sync + 'static + Copy,
    on_live_disconnect: impl Fn() + Send + Sync + 'static + Copy,
    on_live_mock_message: impl Fn(LiveMessageKind) + Send + Sync + 'static + Copy,
    on_browse_model_dir: impl Fn() + Send + Sync + 'static + Copy,
    on_browse_prompt_wav: impl Fn() + Send + Sync + 'static + Copy,
    on_browse_reference_wav: impl Fn() + Send + Sync + 'static + Copy,
    on_test_audio: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let patch_generation = move |update: Box<dyn FnOnce(&mut GenerationSettings)>| {
        let mut generation = config().generation;
        update(&mut generation);
        on_config_patch(ConfigPatch {
            generation: Some(generation),
            ..ConfigPatch::default()
        });
    };

    view! {
        <Show when=open>
            <div class="modal-backdrop" role="presentation">
                <section class="modal settings-modal" role="dialog" aria-modal="true" aria-label=move || labels().settings>
                    <header class="modal-header">
                        <h2>{move || labels().settings}</h2>
                        <button class="primary-button" type="button" aria-label=move || labels().save on:click=move |_| { on_close() }>
                            {move || labels().save}
                        </button>
                    </header>

                    <div class="settings-layout">
                        <nav class="settings-tabs" aria-label=move || labels().settings>
                            <button type="button" class:active=move || active_page() == SettingsPage::General on:click=move |_| on_page_select(SettingsPage::General)>{move || labels().settings_general}</button>
                            <button type="button" class:active=move || active_page() == SettingsPage::Inference on:click=move |_| on_page_select(SettingsPage::Inference)>{move || labels().settings_inference}</button>
                            <button type="button" class:active=move || active_page() == SettingsPage::Audio on:click=move |_| on_page_select(SettingsPage::Audio)>{move || labels().settings_audio}</button>
                            <button type="button" class:active=move || active_page() == SettingsPage::Live on:click=move |_| on_page_select(SettingsPage::Live)>{move || labels().live}</button>
                            <button type="button" class:active=move || active_page() == SettingsPage::About on:click=move |_| on_page_select(SettingsPage::About)>{move || labels().settings_about}</button>
                        </nav>

                        <div class="settings-content">
                            <Show when=move || active_page() == SettingsPage::General>
                                <section class="settings-section">
                                    <h3>{move || labels().settings_general}</h3>
                                    <div class="settings-grid">
                                        <div class="settings-field settings-field-with-button settings-span-2">
                                            <label for="settings-model-dir">{move || labels().model_folder}</label>
                                            <input
                                                id="settings-model-dir"
                                                type="text"
                                                prop:value=move || config().model_root.unwrap_or_default()
                                                placeholder=move || labels().model_folder
                                                readonly=true
                                            />
                                            <button class="secondary-button" type="button" on:click=move |_| { on_browse_model_dir() }>
                                                {move || labels().browse}
                                            </button>
                                        </div>
                                        <label class="settings-field" for="settings-language">
                                            <span>{move || labels().language}</span>
                                            <CustomSelect
                                                class="settings-select-control"
                                                aria_label=labels().language
                                                value=move || language_value(config().language).to_string()
                                                options=move || language_options(labels())
                                                disabled=move || false
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        language: Some(parse_language(&value)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <label class="settings-field" for="settings-theme">
                                            <span>{move || labels().theme}</span>
                                            <CustomSelect
                                                class="settings-select-control"
                                                aria_label=labels().theme
                                                value=move || theme_value(config().theme).to_string()
                                                options=move || theme_options(labels())
                                                disabled=move || false
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        theme: Some(parse_theme(&value)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <label class="settings-field" for="settings-max-input">
                                            <span>{move || labels().max_input_characters}</span>
                                            <NumberCounter
                                                aria_label=labels().max_input_characters
                                                value=move || config().max_input_chars.to_string()
                                                disabled=move || false
                                                min="1"
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        max_input_chars: Some(parse_usize(&value, config().max_input_chars)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                    </div>
                                </section>
                            </Show>

                            <Show when=move || active_page() == SettingsPage::Inference>
                                <section class="settings-section">
                                    <h3>{move || labels().settings_inference}</h3>
                                    <div class="settings-grid">
                                        <label class="settings-field settings-span-2" for="settings-backend">
                                            <span>{move || labels().backend}</span>
                                            <CustomSelect
                                                class="settings-select-control"
                                                aria_label=labels().backend
                                                value=move || backend_value(config().backend).to_string()
                                                options=move || backend_options(labels(), cuda_available())
                                                disabled=move || false
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        backend: Some(parse_backend(&value)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <label class="settings-checkbox settings-switch" for="settings-streaming">
                                            <input id="settings-streaming" type="checkbox" prop:checked=move || config().generation.streaming on:change=move |event| {
                                                let checked = event_target_checked(&event);
                                                patch_generation(Box::new(move |generation| generation.streaming = checked));
                                            } />
                                            <span>{move || labels().streaming}</span>
                                        </label>
                                        <label class="settings-field" for="settings-stream-consolidate">
                                            <span>{move || labels().stream_consolidate_n}</span>
                                            <NumberCounter
                                                aria_label=labels().stream_consolidate_n
                                                value=move || config().generation.stream_consolidate_n.to_string()
                                                disabled=move || !config().generation.streaming
                                                min="1"
                                                on_change=move |next_value| {
                                                    let value = parse_usize(&next_value, config().generation.stream_consolidate_n).max(1);
                                                    patch_generation(Box::new(move |generation| generation.stream_consolidate_n = value));
                                                }
                                            />
                                        </label>
                                        <label class="settings-field" for="settings-cfg">
                                            <span>{move || labels().cfg_value}</span>
                                            <NumberCounter aria_label=labels().cfg_value value=move || config().generation.cfg_value.to_string() disabled=move || false min="0" step="0.1" on_change=move |next_value| {
                                                let value = parse_f32(&next_value, config().generation.cfg_value);
                                                patch_generation(Box::new(move |generation| generation.cfg_value = value));
                                            } />
                                        </label>
                                        <label class="settings-field" for="settings-steps">
                                            <span>{move || labels().inference_steps}</span>
                                            <NumberCounter aria_label=labels().inference_steps value=move || config().generation.inference_timesteps.to_string() disabled=move || false min="1" on_change=move |next_value| {
                                                let value = parse_usize(&next_value, config().generation.inference_timesteps);
                                                patch_generation(Box::new(move |generation| generation.inference_timesteps = value));
                                            } />
                                        </label>
                                        <label class="settings-field" for="settings-min-len">
                                            <span>{move || labels().min_length}</span>
                                            <NumberCounter aria_label=labels().min_length value=move || config().generation.min_len.to_string() disabled=move || false min="0" on_change=move |next_value| {
                                                let value = parse_usize(&next_value, config().generation.min_len);
                                                patch_generation(Box::new(move |generation| generation.min_len = value));
                                            } />
                                        </label>
                                        <label class="settings-field" for="settings-max-len">
                                            <span>{move || labels().max_length}</span>
                                            <NumberCounter aria_label=labels().max_length value=move || config().generation.max_len.to_string() disabled=move || false min="1" on_change=move |next_value| {
                                                let value = parse_usize(&next_value, config().generation.max_len);
                                                patch_generation(Box::new(move |generation| generation.max_len = value));
                                            } />
                                        </label>
                                        <label class="settings-checkbox settings-switch" for="settings-retry-badcase">
                                            <input
                                                id="settings-retry-badcase"
                                                type="checkbox"
                                                disabled=move || config().generation.streaming
                                                prop:checked=move || config().generation.retry_badcase
                                                on:change=move |event| {
                                                let checked = event_target_checked(&event);
                                                patch_generation(Box::new(move |generation| generation.retry_badcase = checked));
                                            } />
                                            <span>{move || labels().retry_badcase}</span>
                                        </label>
                                        <label class="settings-field" for="settings-retry-max">
                                            <span>{move || labels().retry_max_times}</span>
                                            <NumberCounter
                                                aria_label=labels().retry_max_times
                                                value=move || config().generation.retry_badcase_max_times.to_string()
                                                disabled=move || !config().generation.retry_badcase || config().generation.streaming
                                                min="0"
                                                on_change=move |next_value| {
                                                let value = parse_usize(&next_value, config().generation.retry_badcase_max_times);
                                                patch_generation(Box::new(move |generation| generation.retry_badcase_max_times = value));
                                            } />
                                        </label>
                                        <label class="settings-field" for="settings-ratio-threshold">
                                            <span>{move || labels().retry_ratio_threshold}</span>
                                            <NumberCounter
                                                aria_label=labels().retry_ratio_threshold
                                                value=move || config().generation.retry_badcase_ratio_threshold.to_string()
                                                disabled=move || !config().generation.retry_badcase || config().generation.streaming
                                                min="0"
                                                step="0.1"
                                                on_change=move |next_value| {
                                                let value = parse_f32(&next_value, config().generation.retry_badcase_ratio_threshold);
                                                patch_generation(Box::new(move |generation| generation.retry_badcase_ratio_threshold = value));
                                            } />
                                        </label>
                                        <div class="settings-field settings-field-with-button settings-span-2">
                                            <label for="settings-prompt-wav">{move || labels().prompt_wav}</label>
                                            <input id="settings-prompt-wav" type="text" prop:value=move || config().generation.prompt_wav_path.unwrap_or_default() placeholder=move || labels().prompt_wav readonly=true />
                                            <button class="secondary-button" type="button" on:click=move |_| { on_browse_prompt_wav() }>
                                                {move || labels().browse}
                                            </button>
                                        </div>
                                        <div class="settings-field settings-field-with-button settings-span-2">
                                            <label for="settings-reference-wav">{move || labels().reference_wav}</label>
                                            <input id="settings-reference-wav" type="text" prop:value=move || config().generation.reference_wav_path.unwrap_or_default() placeholder=move || labels().reference_wav readonly=true />
                                            <button class="secondary-button" type="button" on:click=move |_| { on_browse_reference_wav() }>
                                                {move || labels().browse}
                                            </button>
                                        </div>
                                        <label class="settings-field settings-span-2" for="settings-prompt-text">
                                            <span>{move || labels().prompt_text}</span>
                                            <textarea
                                                id="settings-prompt-text"
                                                rows="3"
                                                prop:value=move || config().generation.prompt_text.unwrap_or_default()
                                                on:change=move |event| {
                                                    let value = event_target_value(&event);
                                                    patch_generation(Box::new(move |generation| {
                                                        generation.prompt_text = if value.is_empty() { None } else { Some(value.clone()) };
                                                    }));
                                                }
                                            ></textarea>
                                        </label>
                                    </div>
                                </section>
                            </Show>

                            <Show when=move || active_page() == SettingsPage::Audio>
                                <section class="settings-section">
                                    <h3>{move || labels().settings_audio}</h3>
                                    <div class="settings-grid">
                                        <label class="settings-field settings-span-2" for="settings-audio-driver">
                                            <span>{move || labels().audio_driver}</span>
                                            <CustomSelect
                                                class="settings-select-control"
                                                aria_label=labels().audio_driver
                                                value=move || config().audio_host.unwrap_or_else(|| DEFAULT_AUDIO_CHOICE.to_string())
                                                options=move || audio_host_options(audio_state(), config().audio_host, labels())
                                                disabled=move || false
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        audio_host: Some(parse_optional_audio_choice(value)),
                                                        audio_device: Some(None),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <label class="settings-field settings-span-2" for="settings-output-device">
                                            <span>{move || labels().output_device}</span>
                                            <CustomSelect
                                                class="settings-select-control"
                                                aria_label=labels().output_device
                                                value=move || config().audio_device.unwrap_or_else(|| DEFAULT_AUDIO_CHOICE.to_string())
                                                options=move || audio_device_options(audio_state(), &config(), labels())
                                                disabled=move || false
                                                on_change=move |value| {
                                                    on_config_patch(ConfigPatch {
                                                        audio_device: Some(parse_optional_audio_choice(value)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <div class="settings-field settings-volume-test-field settings-span-2">
                                            <span>{move || format!("{}: {}%", labels().volume, volume_to_percent(config().volume))}</span>
                                            <input
                                                id="settings-volume"
                                                type="range"
                                                min="0"
                                                max="100"
                                                prop:value=move || volume_to_percent(config().volume)
                                                on:input=move |event| {
                                                    on_config_patch(ConfigPatch {
                                                        volume: Some((parse_f32(&event_target_value(&event), config().volume * 100.0) / 100.0).clamp(0.0, 1.0)),
                                                        ..ConfigPatch::default()
                                                    });
                                                }
                                            />
                                            <button class="primary-button" type="button" on:click=move |_| { on_test_audio() }>
                                                {move || labels().test}
                                            </button>
                                        </div>
                                    </div>
                                </section>
                            </Show>

                            <Show when=move || active_page() == SettingsPage::Live>
                                <section class="settings-section live-settings-section">
                                    <h3>{move || labels().live}</h3>
                                    <div class="settings-grid live-settings-grid">
                                        <label class="settings-field settings-span-2" for="settings-live-identity-code">
                                            <span>{move || labels().identity_code}</span>
                                            <input
                                                id="settings-live-identity-code"
                                                type="text"
                                                prop:value=move || live_snapshot().config.identity_code
                                                on:change=move |event| {
                                                    on_live_patch(LiveConfigPatch {
                                                        identity_code: Some(event_target_value(&event)),
                                                        ..LiveConfigPatch::default()
                                                    });
                                                }
                                            />
                                        </label>
                                        <div class="settings-field live-status-field settings-span-2">
                                            <span>{move || labels().live}</span>
                                            <strong>{move || live_status_label(labels(), live_snapshot().status)}</strong>
                                            <button
                                                class="primary-button"
                                                type="button"
                                                on:click=move |_| {
                                                    if live_snapshot().status == LiveStatus::Connected {
                                                        on_live_disconnect();
                                                    } else {
                                                        on_live_connect();
                                                    }
                                                }
                                            >
                                                {move || if live_snapshot().status == LiveStatus::Connected { labels().disconnect } else { labels().connect }}
                                            </button>
                                        </div>
                                        <label class="settings-checkbox settings-switch live-checkbox" for="settings-live-ceve-heartbeat">
                                            <input id="settings-live-ceve-heartbeat" type="checkbox" prop:checked=move || live_snapshot().config.enable_ceve_server_heartbeat on:change=move |event| {
                                                on_live_patch(LiveConfigPatch {
                                                    enable_ceve_server_heartbeat: Some(event_target_checked(&event)),
                                                    ..LiveConfigPatch::default()
                                                });
                                            } />
                                            <span>{move || labels().ceve_heartbeat}</span>
                                        </label>
                                        <div class="live-checkbox-grid settings-span-2">
                                            {move || {
                                                let labels = labels();
                                                vec![
                                                    (LiveMessageKind::Danmu, labels.danmu),
                                                    (LiveMessageKind::Gift, labels.gift),
                                                    (LiveMessageKind::Superchat, labels.superchat),
                                                    (LiveMessageKind::Guard, labels.guard),
                                                    (LiveMessageKind::Like, labels.like),
                                                    (LiveMessageKind::Enter, labels.enter),
                                                ].into_iter().map(|(kind, label)| {
                                                    let checked = match kind {
                                                        LiveMessageKind::Danmu => live_snapshot().config.show_danmu,
                                                        LiveMessageKind::Gift => live_snapshot().config.show_gifts,
                                                        LiveMessageKind::Superchat => live_snapshot().config.show_superchats,
                                                        LiveMessageKind::Guard => live_snapshot().config.show_guards,
                                                        LiveMessageKind::Like => live_snapshot().config.show_likes,
                                                        LiveMessageKind::Enter => live_snapshot().config.show_enters,
                                                    };
                                                    let kind_for_test = kind;
                                                    let kind_for_checkbox = kind;
                                                    view! {
                                                        <div class="live-checkbox-row">
                                                            <label class="settings-checkbox live-checkbox">
                                                                <input type="checkbox" prop:checked=checked on:change=move |event| {
                                                                    let checked = event_target_checked(&event);
                                                                    match kind_for_checkbox {
                                                                        LiveMessageKind::Danmu => on_live_patch(LiveConfigPatch { show_danmu: Some(checked), ..LiveConfigPatch::default() }),
                                                                        LiveMessageKind::Gift => on_live_patch(LiveConfigPatch { show_gifts: Some(checked), ..LiveConfigPatch::default() }),
                                                                        LiveMessageKind::Superchat => on_live_patch(LiveConfigPatch { show_superchats: Some(checked), ..LiveConfigPatch::default() }),
                                                                        LiveMessageKind::Guard => on_live_patch(LiveConfigPatch { show_guards: Some(checked), ..LiveConfigPatch::default() }),
                                                                        LiveMessageKind::Like => on_live_patch(LiveConfigPatch { show_likes: Some(checked), ..LiveConfigPatch::default() }),
                                                                        LiveMessageKind::Enter => on_live_patch(LiveConfigPatch { show_enters: Some(checked), ..LiveConfigPatch::default() }),
                                                                    };
                                                                } />
                                                                <span>{label}</span>
                                                            </label>
                                                            <button
                                                                class="live-test-button"
                                                                type="button"
                                                                title="Test"
                                                                aria-label="Test"
                                                                on:click=move |_| on_live_mock_message(kind_for_test)
                                                            >
                                                                "Test"
                                                            </button>
                                                        </div>
                                                    }
                                                }).collect_view()
                                            }}
                                        </div>

                                        <div class="live-subsection settings-span-2">
                                            <h4>{move || labels().send}</h4>
                                            <div class="live-template-grid">
                                                {template_textarea("settings-live-template-danmu", move || labels().danmu.to_string(), move || live_snapshot().config.templates.danmu, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.danmu = value))}
                                                <Show when=move || language() != UiLanguage::English>
                                                    {template_textarea("settings-live-template-gift-zh", move || format!("{} zh", labels().gift), move || live_snapshot().config.templates.gift_zh, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.gift_zh = value))}
                                                </Show>
                                                <Show when=move || language() == UiLanguage::English>
                                                    {template_textarea("settings-live-template-gift-en", move || format!("{} en", labels().gift), move || live_snapshot().config.templates.gift_en, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.gift_en = value))}
                                                </Show>
                                                <Show when=move || language() != UiLanguage::English>
                                                    {template_textarea("settings-live-template-superchat-zh", move || format!("{} zh", labels().superchat), move || live_snapshot().config.templates.superchat_zh, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.superchat_zh = value))}
                                                </Show>
                                                <Show when=move || language() == UiLanguage::English>
                                                    {template_textarea("settings-live-template-superchat-en", move || format!("{} en", labels().superchat), move || live_snapshot().config.templates.superchat_en, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.superchat_en = value))}
                                                </Show>
                                                <Show when=move || language() != UiLanguage::English>
                                                    {template_textarea("settings-live-template-guard-zh", move || format!("{} zh", labels().guard), move || live_snapshot().config.templates.guard_zh, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.guard_zh = value))}
                                                </Show>
                                                <Show when=move || language() == UiLanguage::English>
                                                    {template_textarea("settings-live-template-guard-en", move || format!("{} en", labels().guard), move || live_snapshot().config.templates.guard_en, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.guard_en = value))}
                                                </Show>
                                                <Show when=move || language() != UiLanguage::English>
                                                    {template_textarea("settings-live-template-like-zh", move || format!("{} zh", labels().like), move || live_snapshot().config.templates.like_zh, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.like_zh = value))}
                                                </Show>
                                                <Show when=move || language() == UiLanguage::English>
                                                    {template_textarea("settings-live-template-like-en", move || format!("{} en", labels().like), move || live_snapshot().config.templates.like_en, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.like_en = value))}
                                                </Show>
                                                <Show when=move || language() != UiLanguage::English>
                                                    {template_textarea("settings-live-template-enter-zh", move || format!("{} zh", labels().enter), move || live_snapshot().config.templates.enter_zh, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.enter_zh = value))}
                                                </Show>
                                                <Show when=move || language() == UiLanguage::English>
                                                    {template_textarea("settings-live-template-enter-en", move || format!("{} en", labels().enter), move || live_snapshot().config.templates.enter_en, move |value| patch_template(live_snapshot, on_live_patch, move |templates| templates.enter_en = value))}
                                                </Show>
                                            </div>
                                        </div>

                                        <div class="live-subsection settings-span-2">
                                            <div class="live-subsection-header">
                                                <h4>{move || labels().switch_send}</h4>
                                                <button class="secondary-button live-symbol-button" type="button" aria-label=move || labels().switch_send on:click=move |_| {
                                                    let mut rules = live_snapshot().config.replacement_rules;
                                                    rules.push(ReplacementRule { enabled: true, from: String::new(), to: String::new() });
                                                    patch_replacement_rules(on_live_patch, rules);
                                                }>"+"</button>
                                            </div>
                                            <div class="live-list">
                                                {move || {
                                                    replacement_rule_rows(live_snapshot())
                                                        .into_iter()
                                                        .map(|(index, rule)| {
                                                        view! {
                                                            <div class="live-replacement-row">
                                                                <label class="live-inline-checkbox">
                                                                    <input type="checkbox" prop:checked=rule.enabled on:change=move |event| {
                                                                        let mut rules = live_snapshot().config.replacement_rules;
                                                                        if let Some(rule) = rules.get_mut(index) {
                                                                            rule.enabled = event_target_checked(&event);
                                                                        }
                                                                        patch_replacement_rules(on_live_patch, rules);
                                                                    } />
                                                                </label>
                                                                <input type="text" aria-label=move || labels().switch_send prop:value=rule.from.clone() on:change=move |event| {
                                                                    let mut rules = live_snapshot().config.replacement_rules;
                                                                    if let Some(rule) = rules.get_mut(index) {
                                                                        rule.from = event_target_value(&event);
                                                                    }
                                                                    patch_replacement_rules(on_live_patch, rules);
                                                                } />
                                                                <input type="text" aria-label=move || labels().switch_send prop:value=rule.to.clone() on:change=move |event| {
                                                                    let mut rules = live_snapshot().config.replacement_rules;
                                                                    if let Some(rule) = rules.get_mut(index) {
                                                                        rule.to = event_target_value(&event);
                                                                    }
                                                                    patch_replacement_rules(on_live_patch, rules);
                                                                } />
                                                                <button class="secondary-button live-remove-button live-symbol-button" type="button" aria-label=move || labels().clear on:click=move |_| {
                                                                    let mut rules = live_snapshot().config.replacement_rules;
                                                                    if index < rules.len() {
                                                                        rules.remove(index);
                                                                    }
                                                                    patch_replacement_rules(on_live_patch, rules);
                                                                }>"x"</button>
                                                            </div>
                                                        }
                                                        })
                                                        .collect_view()
                                                }}
                                            </div>
                                        </div>

                                        <div class="live-subsection settings-span-2">
                                            <h4>{move || labels().danmu}</h4>
                                            <div class="live-list">
                                                <For
                                                    each=move || mapped_name_rows(live_snapshot())
                                                    key=|(open_id, _, _)| open_id.clone()
                                                    children=move |(open_id, original, mapped)| {
                                                        view! {
                                                            <div class="live-name-row">
                                                                <code class="live-open-id">{open_id.clone()}</code>
                                                                <span>{original}</span>
                                                                <input type="text" aria-label=move || labels().danmu prop:value=mapped on:change=move |event| {
                                                                    let mut mapped_unames = live_snapshot().config.mapped_unames;
                                                                    mapped_unames.insert(open_id.clone(), event_target_value(&event));
                                                                    patch_mapped_unames(on_live_patch, mapped_unames);
                                                                } />
                                                            </div>
                                                        }
                                                    }
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </section>
                            </Show>

                            <Show when=move || active_page() == SettingsPage::About>
                                <section class="settings-section">
                                    <h3>{move || labels().settings_about}</h3>
                                    <div class="about-panel settings-span-2">
                                        <p>{move || labels().about_text}</p>
                                    </div>
                                </section>
                            </Show>
                        </div>
                    </div>
                </section>
            </div>
        </Show>
    }
}

fn audio_hosts_with_current(mut state: AudioState, current_host: Option<String>) -> Vec<AudioHost> {
    if let Some(current_host) = current_host {
        if !state.hosts.iter().any(|host| host.name == current_host) {
            state.hosts.push(AudioHost { name: current_host });
        }
    }

    state.hosts
}

fn audio_devices_for_selected_host(mut state: AudioState, config: &AppConfig) -> Vec<AudioDevice> {
    let selected_host = config
        .audio_host
        .as_deref()
        .or(state.default_host.as_deref());
    let Some(selected_host) = selected_host else {
        return config
            .audio_device
            .clone()
            .map(|name| AudioDevice {
                name,
                host_name: String::new(),
            })
            .into_iter()
            .collect();
    };

    let mut devices = state
        .devices
        .drain(..)
        .filter(|device| device.host_name == selected_host)
        .collect::<Vec<_>>();

    if let Some(current_device) = config.audio_device.as_ref() {
        if !devices.iter().any(|device| device.name == *current_device) {
            devices.push(AudioDevice {
                name: current_device.clone(),
                host_name: selected_host.to_string(),
            });
        }
    }

    devices
}

fn language_options(labels: Labels) -> Vec<SelectOption> {
    vec![
        SelectOption::new("system", labels.system),
        SelectOption::new("chinese", labels.chinese),
        SelectOption::new("english", labels.english),
    ]
}

fn theme_options(labels: Labels) -> Vec<SelectOption> {
    vec![
        SelectOption::new("dark", labels.theme_dark),
        SelectOption::new("light", labels.theme_light),
    ]
}

fn backend_options(labels: Labels, cuda_available: bool) -> Vec<SelectOption> {
    let mut options = vec![SelectOption::new("cpu", labels.cpu)];
    if cuda_available {
        options.push(SelectOption::new("cuda", labels.cuda));
    }
    options
}

fn audio_host_options(
    state: AudioState,
    current_host: Option<String>,
    labels: Labels,
) -> Vec<SelectOption> {
    std::iter::once(SelectOption::new(
        DEFAULT_AUDIO_CHOICE,
        default_label(labels.default_choice, state.default_host.as_deref()),
    ))
    .chain(
        audio_hosts_with_current(state, current_host)
            .into_iter()
            .map(|host| SelectOption::new(host.name.clone(), host.name)),
    )
    .collect()
}

fn audio_device_options(
    state: AudioState,
    config: &AppConfig,
    labels: Labels,
) -> Vec<SelectOption> {
    let default_device = default_device_for_selected_host(&state, config);
    std::iter::once(SelectOption::new(
        DEFAULT_AUDIO_CHOICE,
        default_label(labels.default_choice, default_device.as_deref()),
    ))
    .chain(
        audio_devices_for_selected_host(state, config)
            .into_iter()
            .map(|device| SelectOption::new(device.name.clone(), device.name)),
    )
    .collect()
}

fn default_device_for_selected_host(state: &AudioState, config: &AppConfig) -> Option<String> {
    let selected_host = config
        .audio_host
        .as_deref()
        .or(state.default_host.as_deref())?;

    state
        .default_devices
        .iter()
        .find(|device| device.host_name == selected_host)
        .map(|device| device.name.clone())
}

fn default_label(label: &str, value: Option<&str>) -> String {
    match value {
        Some(value) if !value.is_empty() => format!("{label} ({value})"),
        _ => label.to_string(),
    }
}

fn parse_optional_audio_choice(value: String) -> Option<String> {
    if value.is_empty() || value == DEFAULT_AUDIO_CHOICE {
        Some(String::new())
    } else {
        Some(value)
    }
}

fn theme_value(theme: ThemeMode) -> &'static str {
    match theme {
        ThemeMode::Dark => "dark",
        ThemeMode::Light => "light",
    }
}

fn parse_theme(value: &str) -> ThemeMode {
    match value {
        "light" => ThemeMode::Light,
        _ => ThemeMode::Dark,
    }
}

fn language_value(language: LanguageMode) -> &'static str {
    match language {
        LanguageMode::System => "system",
        LanguageMode::Chinese => "chinese",
        LanguageMode::English => "english",
    }
}

fn parse_language(value: &str) -> LanguageMode {
    match value {
        "chinese" => LanguageMode::Chinese,
        "english" => LanguageMode::English,
        _ => LanguageMode::System,
    }
}

fn backend_value(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Cuda => "cuda",
    }
}

fn parse_backend(value: &str) -> BackendKind {
    match value {
        "cuda" => BackendKind::Cuda,
        _ => BackendKind::Cpu,
    }
}

fn volume_to_percent(volume: f32) -> String {
    ((volume.clamp(0.0, 1.0) * 100.0).round() as usize).to_string()
}

fn parse_f32(value: &str, fallback: f32) -> f32 {
    value.parse().unwrap_or(fallback)
}

fn parse_usize(value: &str, fallback: usize) -> usize {
    value.parse().unwrap_or(fallback)
}

fn live_status_label(labels: Labels, status: LiveStatus) -> &'static str {
    match status {
        LiveStatus::Disconnected => match labels.english {
            "English" => "Disconnected",
            _ => "未连接",
        },
        LiveStatus::Connecting => match labels.english {
            "English" => "Connecting",
            _ => "连接中",
        },
        LiveStatus::Connected => match labels.english {
            "English" => "Connected",
            _ => "已连接",
        },
        LiveStatus::Disconnecting => match labels.english {
            "English" => "Disconnecting",
            _ => "断开中",
        },
        LiveStatus::Error => labels.history_status_failed,
    }
}

fn template_textarea(
    id: &'static str,
    label: impl Fn() -> String + Send + Sync + 'static + Copy,
    value: impl Fn() -> String + Send + Sync + 'static + Copy,
    on_change: impl Fn(String) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    view! {
        <label class="settings-field live-template-field" for=id>
            <span>{label}</span>
            <textarea
                id=id
                rows="2"
                prop:value=value
                on:change=move |event| on_change(event_target_value(&event))
            ></textarea>
        </label>
    }
}

fn patch_template(
    live_snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    update: impl FnOnce(&mut TemplateConfig),
) {
    let mut templates = live_snapshot().config.templates;
    update(&mut templates);
    on_live_patch(LiveConfigPatch {
        templates: Some(templates),
        ..LiveConfigPatch::default()
    });
}

fn patch_replacement_rules(
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    replacement_rules: Vec<ReplacementRule>,
) {
    on_live_patch(LiveConfigPatch {
        replacement_rules: Some(replacement_rules),
        ..LiveConfigPatch::default()
    });
}

fn patch_mapped_unames(
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    mapped_unames: BTreeMap<String, String>,
) {
    on_live_patch(LiveConfigPatch {
        mapped_unames: Some(mapped_unames),
        ..LiveConfigPatch::default()
    });
}

fn replacement_rule_rows(snapshot: LiveSnapshot) -> Vec<(usize, ReplacementRule)> {
    snapshot
        .config
        .replacement_rules
        .into_iter()
        .enumerate()
        .collect()
}

fn mapped_name_rows(snapshot: LiveSnapshot) -> Vec<(String, String, String)> {
    let mut rows = snapshot
        .config
        .original_unames
        .iter()
        .map(|(open_id, original)| {
            let mapped = snapshot
                .config
                .mapped_unames
                .get(open_id)
                .cloned()
                .unwrap_or_else(|| original.clone());
            (open_id.clone(), original.clone(), mapped)
        })
        .collect::<Vec<_>>();

    for (open_id, mapped) in snapshot.config.mapped_unames {
        if !rows
            .iter()
            .any(|(existing_open_id, _, _)| existing_open_id == &open_id)
        {
            rows.push((open_id.clone(), open_id, mapped));
        }
    }

    rows
}
