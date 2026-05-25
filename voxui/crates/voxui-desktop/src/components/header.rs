use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::ModelChoice;

#[component]
pub fn Header(
    labels: Labels,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_disabled: bool,
    on_model_select: impl Fn(String) + 'static + Copy,
    on_load: impl Fn() + 'static + Copy,
    on_open_settings: impl Fn() + 'static + Copy,
) -> impl IntoView {
    let selected_model_id = selected_model_id.unwrap_or_default();

    view! {
        <header class="app-header">
            <div class="brand">
                <strong>{labels.title}</strong>
                <span>{labels.subtitle}</span>
            </div>

            <select
                class="model-select"
                aria-label={labels.model}
                prop:value=selected_model_id.clone()
                on:change=move |event| on_model_select(event_target_value(&event))
            >
                <option value="">{labels.model}</option>
                {models
                    .into_iter()
                    .map(|model| {
                        let selected = model.id == selected_model_id;
                        view! {
                            <option value={model.id.clone()} selected={selected}>
                                {model.display_name}
                            </option>
                        }
                    })
                    .collect_view()}
            </select>

            <button
                class="primary-button"
                disabled=load_disabled
                on:click=move |_| on_load()
            >
                {labels.load}
            </button>

            <button
                class="icon-button"
                title={labels.settings}
                aria-label={labels.settings}
                on:click=move |_| on_open_settings()
            >
                {"⚙"}
            </button>

            {loaded_model_id
                .map(|model_id| {
                    view! { <span class="loaded-pill">{model_id}</span> }
                })}
        </header>
    }
}
