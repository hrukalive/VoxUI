use leptos::prelude::*;

use crate::components::controls::{CustomSelect, NumberCounter, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::{
    AppConfig, AudioDevice, AudioHost, AudioState, BackendKind, ConfigPatch, GenerationSettings,
    LanguageMode, ThemeMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsPage {
    General,
    Inference,
    Audio,
    About,
}

const DEFAULT_AUDIO_CHOICE: &str = "__voxui_default_audio__";

#[component]
pub fn SettingsModal(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    audio_state: impl Fn() -> AudioState + Send + Sync + 'static + Copy,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    active_page: impl Fn() -> SettingsPage + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
    on_page_select: impl Fn(SettingsPage) + Send + Sync + 'static + Copy,
    on_config_patch: impl Fn(ConfigPatch) + Send + Sync + 'static + Copy,
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
                                                options=move || backend_options(labels())
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

fn backend_options(labels: Labels) -> Vec<SelectOption> {
    vec![
        SelectOption::new("cpu", labels.cpu),
        SelectOption::new("cuda", labels.cuda),
    ]
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
        None
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
