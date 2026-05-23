use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::{AppConfig, BackendKind, ConfigPatch, GenerationSettings, LanguageMode};

#[component]
pub fn SettingsModal(
    labels: Labels,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
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
                <section class="modal settings-modal" role="dialog" aria-modal="true" aria-label={labels.settings}>
                    <header class="modal-header">
                        <h2>{labels.settings}</h2>
                        <button class="icon-button" aria-label={labels.cancel} on:click=move |_| { on_close() }>
                            {"×"}
                        </button>
                    </header>
                    <div class="settings-content">
                        <section class="settings-section">
                            <h3>{labels.model}</h3>
                            <div class="settings-field settings-field-with-button">
                                <label for="settings-model-dir">{labels.model_folder}</label>
                                <input
                                    id="settings-model-dir"
                                    type="text"
                                    prop:value=move || config().model_root.unwrap_or_default()
                                    placeholder={labels.model_folder}
                                    readonly=true
                                />
                                <button class="secondary-button" type="button" on:click=move |_| { on_browse_model_dir() }>
                                    {"Browse"}
                                </button>
                            </div>
                        </section>

                        <section class="settings-section">
                            <h3>{"Interface"}</h3>
                            <label class="settings-field" for="settings-language">
                                <span>{labels.language}</span>
                                <select
                                    id="settings-language"
                                    prop:value=move || language_value(config().language)
                                    on:change=move |event| {
                                        on_config_patch(ConfigPatch {
                                            language: Some(parse_language(&event_target_value(&event))),
                                            ..ConfigPatch::default()
                                        });
                                    }
                                >
                                    <option value="system">{labels.system}</option>
                                    <option value="chinese">{labels.chinese}</option>
                                    <option value="english">{labels.english}</option>
                                </select>
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Inference"}</h3>
                            <label class="settings-field" for="settings-backend">
                                <span>{labels.backend}</span>
                                <select
                                    id="settings-backend"
                                    prop:value=move || backend_value(config().backend)
                                    on:change=move |event| {
                                        on_config_patch(ConfigPatch {
                                            backend: Some(parse_backend(&event_target_value(&event))),
                                            ..ConfigPatch::default()
                                        });
                                    }
                                >
                                    <option value="cpu">{labels.cpu}</option>
                                    <option value="cuda">{labels.cuda}</option>
                                </select>
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Audio"}</h3>
                            <div class="settings-grid">
                                <label class="settings-field" for="settings-audio-driver">
                                    <span>{"Driver"}</span>
                                    <select
                                        id="settings-audio-driver"
                                        prop:value=move || config().audio_host.unwrap_or_default()
                                        on:change=move |event| {
                                            let value = event_target_value(&event);
                                            on_config_patch(ConfigPatch {
                                                audio_host: Some(if value.is_empty() { None } else { Some(value) }),
                                                ..ConfigPatch::default()
                                            });
                                        }
                                    >
                                        <option value="">{"Default"}</option>
                                        {move || config().audio_host.map(|host| view! { <option value={host.clone()}>{host.clone()}</option> })}
                                    </select>
                                </label>
                                <label class="settings-field" for="settings-output-device">
                                    <span>{"Output device"}</span>
                                    <select
                                        id="settings-output-device"
                                        prop:value=move || config().audio_device.unwrap_or_default()
                                        on:change=move |event| {
                                            let value = event_target_value(&event);
                                            on_config_patch(ConfigPatch {
                                                audio_device: Some(if value.is_empty() { None } else { Some(value) }),
                                                ..ConfigPatch::default()
                                            });
                                        }
                                    >
                                        <option value="">{"Default"}</option>
                                        {move || config().audio_device.map(|device| view! { <option value={device.clone()}>{device.clone()}</option> })}
                                    </select>
                                </label>
                                <label class="settings-field" for="settings-volume">
                                    <span>{labels.volume}</span>
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
                                </label>
                                <div class="settings-field settings-action-field">
                                    <span>{"Audio test"}</span>
                                    <button class="secondary-button" type="button" on:click=move |_| { on_test_audio() }>
                                        {"Test"}
                                    </button>
                                </div>
                            </div>
                        </section>

                        <section class="settings-section">
                            <h3>{"VoxCPM generation"}</h3>
                            <div class="settings-grid">
                                <label class="settings-field" for="settings-cfg">
                                    <span>{"CFG value"}</span>
                                    <input id="settings-cfg" type="number" min="0" step="0.1" prop:value=move || config().generation.cfg_value.to_string() on:change=move |event| {
                                        let value = parse_f32(&event_target_value(&event), config().generation.cfg_value);
                                        patch_generation(Box::new(move |generation| generation.cfg_value = value));
                                    } />
                                </label>
                                <label class="settings-field" for="settings-steps">
                                    <span>{"Inference steps"}</span>
                                    <input id="settings-steps" type="number" min="1" step="1" prop:value=move || config().generation.inference_timesteps.to_string() on:change=move |event| {
                                        let value = parse_usize(&event_target_value(&event), config().generation.inference_timesteps);
                                        patch_generation(Box::new(move |generation| generation.inference_timesteps = value));
                                    } />
                                </label>
                                <label class="settings-field" for="settings-min-len">
                                    <span>{"Min length"}</span>
                                    <input id="settings-min-len" type="number" min="0" step="1" prop:value=move || config().generation.min_len.to_string() on:change=move |event| {
                                        let value = parse_usize(&event_target_value(&event), config().generation.min_len);
                                        patch_generation(Box::new(move |generation| generation.min_len = value));
                                    } />
                                </label>
                                <label class="settings-field" for="settings-max-len">
                                    <span>{"Max length"}</span>
                                    <input id="settings-max-len" type="number" min="1" step="1" prop:value=move || config().generation.max_len.to_string() on:change=move |event| {
                                        let value = parse_usize(&event_target_value(&event), config().generation.max_len);
                                        patch_generation(Box::new(move |generation| generation.max_len = value));
                                    } />
                                </label>
                                <label class="settings-checkbox" for="settings-retry-badcase">
                                    <input id="settings-retry-badcase" type="checkbox" prop:checked=move || config().generation.retry_badcase on:change=move |event| {
                                        let checked = event_target_checked(&event);
                                        patch_generation(Box::new(move |generation| generation.retry_badcase = checked));
                                    } />
                                    <span>{"Retry badcase"}</span>
                                </label>
                                <label class="settings-field" for="settings-retry-max">
                                    <span>{"Retry max times"}</span>
                                    <input id="settings-retry-max" type="number" min="0" step="1" prop:value=move || config().generation.retry_badcase_max_times.to_string() on:change=move |event| {
                                        let value = parse_usize(&event_target_value(&event), config().generation.retry_badcase_max_times);
                                        patch_generation(Box::new(move |generation| generation.retry_badcase_max_times = value));
                                    } />
                                </label>
                                <label class="settings-field" for="settings-ratio-threshold">
                                    <span>{"Retry ratio threshold"}</span>
                                    <input id="settings-ratio-threshold" type="number" min="0" step="0.1" prop:value=move || config().generation.retry_badcase_ratio_threshold.to_string() on:change=move |event| {
                                        let value = parse_f32(&event_target_value(&event), config().generation.retry_badcase_ratio_threshold);
                                        patch_generation(Box::new(move |generation| generation.retry_badcase_ratio_threshold = value));
                                    } />
                                </label>
                            </div>
                        </section>

                        <section class="settings-section">
                            <h3>{"Input"}</h3>
                            <label class="settings-field" for="settings-max-input">
                                <span>{"Max input characters"}</span>
                                <input
                                    id="settings-max-input"
                                    type="number"
                                    min="1"
                                    step="1"
                                    prop:value=move || config().max_input_chars.to_string()
                                    on:change=move |event| {
                                        on_config_patch(ConfigPatch {
                                            max_input_chars: Some(parse_usize(&event_target_value(&event), config().max_input_chars)),
                                            ..ConfigPatch::default()
                                        });
                                    }
                                />
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Advanced prompt/reference"}</h3>
                            <div class="settings-field settings-field-with-button">
                                <label for="settings-prompt-wav">{"Prompt WAV"}</label>
                                <input id="settings-prompt-wav" type="text" prop:value=move || config().generation.prompt_wav_path.unwrap_or_default() placeholder="Prompt WAV" readonly=true />
                                <button class="secondary-button" type="button" on:click=move |_| { on_browse_prompt_wav() }>
                                    {"Browse"}
                                </button>
                            </div>
                            <label class="settings-field" for="settings-prompt-text">
                                <span>{"Prompt text"}</span>
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
                            <div class="settings-field settings-field-with-button">
                                <label for="settings-reference-wav">{"Reference WAV"}</label>
                                <input id="settings-reference-wav" type="text" prop:value=move || config().generation.reference_wav_path.unwrap_or_default() placeholder="Reference WAV" readonly=true />
                                <button class="secondary-button" type="button" on:click=move |_| { on_browse_reference_wav() }>
                                    {"Browse"}
                                </button>
                            </div>
                        </section>
                    </div>
                </section>
            </div>
        </Show>
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
