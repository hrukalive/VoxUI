use leptos::prelude::*;

use crate::components::controls::{translation_lang_options, CustomSelect};
use crate::i18n::Labels;
use crate::tauri_api::{AppConfig, ConfigPatch, TranslationSettings};

#[component]
pub fn TranslationBar(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    input_text: impl Fn() -> String + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_replace_text: impl Fn(String) + 'static + Copy,
    on_enqueue: impl Fn(String) + 'static + Copy,
    on_config_patch: impl Fn(ConfigPatch) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let (translating, set_translating) = signal(false);

    let target_value = move || config().translation.outbound.target_lang.clone();
    let target_disabled = move || disabled() || translating.get();

    let translate_action = move || {
        let text = input_text();
        if text.trim().is_empty() || translating.get() {
            return;
        }
        set_translating.set(true);
        let source_lang = config().translation.outbound.source_lang.clone();
        let target_lang = config().translation.outbound.target_lang.clone();
        let enqueue = config().translation.translate_enqueue;

        spawn_local(async move {
            match crate::tauri_api::translate_text(text, source_lang, target_lang).await {
                Ok(translated) => {
                    if enqueue {
                        on_enqueue(translated);
                    } else {
                        on_replace_text(translated);
                    }
                }
                Err(_) => {}
            }
            set_translating.set(false);
        });
    };

    view! {
        <div class="translation-bar">
            <label class="translation-bar-select">
                <CustomSelect
                    class="translation-target-select"
                    aria_label=move || labels().target_language
                    value=target_value
                    options=move || translation_lang_options(false, &labels())
                    disabled=target_disabled
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.outbound.target_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
            <button
                class="primary-button translation-button"
                type="button"
                disabled=move || disabled() || input_text().trim().is_empty() || translating.get()
                on:click=move |_| translate_action()
            >
                {move || if translating.get() { labels().translating } else { labels().translate }}
            </button>
            <label class="translation-checkbox">
                <input
                    id="translation-enqueue"
                    type="checkbox"
                    prop:checked=move || config().translation.translate_enqueue
                    disabled=target_disabled
                    on:change=move |event| {
                        let mut translation = config().translation.clone();
                        translation.translate_enqueue = event_target_checked(&event);
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
                <span>{move || labels().enqueue_translation}</span>
            </label>
        </div>
    }
}
