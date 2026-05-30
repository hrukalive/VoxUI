use leptos::ev::SubmitEvent;
use leptos::prelude::*;

use crate::i18n::{Labels, UiLanguage};

#[component]
pub fn InputBox(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    language: impl Fn() -> UiLanguage + Send + Sync + 'static + Copy,
    max_chars: impl Fn() -> usize + Send + Sync + 'static + Copy,
    auto_period: impl Fn() -> bool + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    replacement_text: impl Fn() -> Option<String> + Send + Sync + 'static + Copy,
    on_replacement_consumed: impl Fn() + Send + Sync + 'static + Copy,
    on_generate: impl Fn(String) + 'static + Copy,
    #[prop(optional)] translation_bar: Option<AnyView>,
    #[prop(optional)] on_text_change: Option<impl Fn(String) + 'static + Copy>,
) -> impl IntoView {
    let (text, set_text) = signal(String::new());
    let char_count = move || text.get().chars().count();
    let is_over_limit = move || char_count() > max_chars();
    let generate_disabled = move || disabled() || text.get().trim().is_empty() || is_over_limit();

    let submit = move |event: SubmitEvent| {
        event.prevent_default();
        if !generate_disabled() {
            let raw = text.get().trim().to_owned();
            let final_text = if auto_period() {
                ensure_period(&raw, language())
            } else {
                raw
            };
            on_generate(final_text);
            set_text.set(String::new());
            if let Some(ref cb) = on_text_change {
                cb(String::new());
            }
        }

        fn ensure_period(text: &str, language: UiLanguage) -> String {
            let period = match language {
                UiLanguage::Chinese => '。',
                UiLanguage::English => '.',
            };
            let endings = ['?', '!', '.', '…', '？', '！', '。'];
            if text.is_empty() || text.ends_with(&endings) {
                text.to_string()
            } else {
                format!("{}{}", text, period)
            }
        }
    };

    Effect::new(move |_| {
        if let Some(replacement) = replacement_text() {
            set_text.set(replacement.clone());
            if let Some(ref cb) = on_text_change {
                cb(replacement);
            }
            on_replacement_consumed();
        }
    });

    view! {
        <form class="composer-panel" on:submit=submit>
            <div class="composer-row">
                {if let Some(bar) = translation_bar {
                    view! { <div class="composer-translation-column">{bar}</div> }.into_any()
                } else {
                    ().into_any()
                }}
                <div class="composer-field">
                    <textarea
                        class="composer-input"
                        prop:value=move || text.get()
                        placeholder=move || labels().input_placeholder
                        disabled=move || disabled()
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            if let Some(ref cb) = on_text_change {
                                cb(value.clone());
                            }
                            set_text.set(value);
                        }
                    ></textarea>
                    <span class:over-limit=is_over_limit class="char-counter">
                        {move || format!("{}/{}", char_count(), max_chars())}
                    </span>
                </div>
            </div>
            <div class="composer-actions">
                <button class="generate-button" type="submit" disabled=generate_disabled>
                    {move || labels().generate}
                </button>
                <button
                    class="secondary-button composer-clear-button"
                    type="button"
                    disabled=move || text.get().is_empty()
                    on:click=move |_| {
                        set_text.set(String::new());
                        if let Some(ref cb) = on_text_change {
                            cb(String::new());
                        }
                    }
                >
                    {move || labels().clear}
                </button>
            </div>
        </form>
    }
}
