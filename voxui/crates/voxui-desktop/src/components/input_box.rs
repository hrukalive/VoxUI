use leptos::ev::SubmitEvent;
use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn InputBox(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    max_chars: impl Fn() -> usize + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
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
            <button class="generate-button" type="submit" disabled=generate_disabled>
                {move || labels().generate}
            </button>
        </form>
    }
}
