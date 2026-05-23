use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn SettingsModal(
    labels: Labels,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
    on_browse_model_dir: impl Fn() + Send + Sync + 'static + Copy,
    on_browse_prompt_wav: impl Fn() + Send + Sync + 'static + Copy,
    on_browse_reference_wav: impl Fn() + Send + Sync + 'static + Copy,
    on_test_audio: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
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
                                <input id="settings-model-dir" type="text" value="" placeholder={labels.model_folder} readonly=true />
                                <button class="secondary-button" type="button" on:click=move |_| { on_browse_model_dir() }>
                                    {"Browse"}
                                </button>
                            </div>
                        </section>

                        <section class="settings-section">
                            <h3>{"Interface"}</h3>
                            <label class="settings-field" for="settings-language">
                                <span>{labels.language}</span>
                                <select id="settings-language">
                                    <option>{labels.system}</option>
                                    <option>{labels.chinese}</option>
                                    <option>{labels.english}</option>
                                </select>
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Inference"}</h3>
                            <label class="settings-field" for="settings-backend">
                                <span>{labels.backend}</span>
                                <select id="settings-backend">
                                    <option>{labels.cpu}</option>
                                    <option>{labels.cuda}</option>
                                </select>
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Audio"}</h3>
                            <div class="settings-grid">
                                <label class="settings-field" for="settings-audio-driver">
                                    <span>{"Driver"}</span>
                                    <select id="settings-audio-driver">
                                        <option>{"Default"}</option>
                                    </select>
                                </label>
                                <label class="settings-field" for="settings-output-device">
                                    <span>{"Output device"}</span>
                                    <select id="settings-output-device">
                                        <option>{"Default"}</option>
                                    </select>
                                </label>
                                <label class="settings-field" for="settings-volume">
                                    <span>{labels.volume}</span>
                                    <input id="settings-volume" type="range" min="0" max="100" value="80" />
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
                                    <input id="settings-cfg" type="number" min="0" step="0.1" value="2.0" />
                                </label>
                                <label class="settings-field" for="settings-steps">
                                    <span>{"Inference steps"}</span>
                                    <input id="settings-steps" type="number" min="1" step="1" value="10" />
                                </label>
                                <label class="settings-field" for="settings-min-len">
                                    <span>{"Min length"}</span>
                                    <input id="settings-min-len" type="number" min="0" step="1" value="2" />
                                </label>
                                <label class="settings-field" for="settings-max-len">
                                    <span>{"Max length"}</span>
                                    <input id="settings-max-len" type="number" min="1" step="1" value="2000" />
                                </label>
                                <label class="settings-checkbox" for="settings-retry-badcase">
                                    <input id="settings-retry-badcase" type="checkbox" checked=true />
                                    <span>{"Retry badcase"}</span>
                                </label>
                                <label class="settings-field" for="settings-retry-max">
                                    <span>{"Retry max times"}</span>
                                    <input id="settings-retry-max" type="number" min="0" step="1" value="3" />
                                </label>
                                <label class="settings-field" for="settings-ratio-threshold">
                                    <span>{"Retry ratio threshold"}</span>
                                    <input id="settings-ratio-threshold" type="number" min="0" step="0.1" value="6.0" />
                                </label>
                            </div>
                        </section>

                        <section class="settings-section">
                            <h3>{"Input"}</h3>
                            <label class="settings-field" for="settings-max-input">
                                <span>{"Max input characters"}</span>
                                <input id="settings-max-input" type="number" min="1" step="1" value="280" />
                            </label>
                        </section>

                        <section class="settings-section">
                            <h3>{"Advanced prompt/reference"}</h3>
                            <div class="settings-field settings-field-with-button">
                                <label for="settings-prompt-wav">{"Prompt WAV"}</label>
                                <input id="settings-prompt-wav" type="text" value="" placeholder="Prompt WAV" readonly=true />
                                <button class="secondary-button" type="button" on:click=move |_| { on_browse_prompt_wav() }>
                                    {"Browse"}
                                </button>
                            </div>
                            <label class="settings-field" for="settings-prompt-text">
                                <span>{"Prompt text"}</span>
                                <textarea id="settings-prompt-text" rows="3"></textarea>
                            </label>
                            <div class="settings-field settings-field-with-button">
                                <label for="settings-reference-wav">{"Reference WAV"}</label>
                                <input id="settings-reference-wav" type="text" value="" placeholder="Reference WAV" readonly=true />
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
