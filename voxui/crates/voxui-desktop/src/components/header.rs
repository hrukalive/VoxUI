use leptos::prelude::*;

use crate::components::controls::{CustomSelect, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::ModelChoice;

#[component]
pub fn Header(
    labels: Labels,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_disabled: bool,
    on_model_select: impl Fn(String) + Send + Sync + 'static + Copy,
    on_load: impl Fn() + Send + Sync + 'static + Copy,
    on_open_settings: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let selected_model_id = selected_model_id.unwrap_or_default();
    let model_options = {
        let models = models.clone();
        move || {
            models
                .clone()
                .into_iter()
                .map(|model| SelectOption::new(model.id, model.display_name))
                .collect::<Vec<_>>()
        }
    };
    let current_model_id = {
        let selected_model_id = selected_model_id.clone();
        move || selected_model_id.clone()
    };

    view! {
        <header class="app-header">
            <div class="brand">
                <strong>{labels.title}</strong>
                <span>{labels.subtitle}</span>
            </div>

            <CustomSelect
                class="model-select"
                aria_label=labels.model
                value=current_model_id
                options=model_options
                disabled=move || false
                on_change=move |model_id| on_model_select(model_id)
            />

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
