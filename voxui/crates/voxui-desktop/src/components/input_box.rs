use leptos::ev::SubmitEvent;
use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn InputBox(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    max_chars: impl Fn() -> usize + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    replacement_text: impl Fn() -> Option<String> + Send + Sync + 'static + Copy,
    on_replacement_consumed: impl Fn() + Send + Sync + 'static + Copy,
    on_generate: impl Fn(String) + 'static + Copy,
) -> impl IntoView {
    let (text, set_text) = signal(String::new());
    let char_count = move || text.get().chars().count();
    let is_over_limit = move || char_count() > max_chars();
    let generate_disabled = move || disabled() || text.get().trim().is_empty() || is_over_limit();

    let submit = move |event: SubmitEvent| {
        event.prevent_default();
        if !generate_disabled() {
            on_generate(text.get().trim().to_owned());
            set_text.set(String::new());
        }
    };

    Effect::new(move |_| {
        if let Some(replacement) = replacement_text() {
            set_text.set(replacement);
            on_replacement_consumed();
        }
    });

    view! {
        <form class="composer-panel" on:submit=submit>
            <div class="composer-field">
                <textarea
                    class="composer-input"
                    prop:value=move || text.get()
                    placeholder=move || labels().input_placeholder
                    disabled=move || disabled()
                    on:input=move |event| set_text.set(event_target_value(&event))
                ></textarea>
                <span class:over-limit=is_over_limit class="char-counter">
                    {move || format!("{}/{}", char_count(), max_chars())}
                </span>
            </div>
            <div class="composer-actions">
                <button class="generate-button" type="submit" disabled=generate_disabled>
                    {move || labels().generate}
                </button>
                <button
                    class="secondary-button composer-clear-button"
                    type="button"
                    disabled=move || text.get().is_empty()
                    on:click=move |_| set_text.set(String::new())
                >
                    {move || labels().clear}
                </button>
            </div>
        </form>
    }
}
