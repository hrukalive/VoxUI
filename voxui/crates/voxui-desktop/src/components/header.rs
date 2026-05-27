use leptos::prelude::*;

use crate::components::controls::{CustomSelect, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::ModelChoice;

#[component]
pub fn Header(
    labels: Labels,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    load_disabled: bool,
    volume: f32,
    on_model_select: impl Fn(String) + Send + Sync + 'static + Copy,
    on_load: impl Fn() + Send + Sync + 'static + Copy,
    on_volume_change: impl Fn(f32) + Send + Sync + 'static + Copy,
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

            <label class="header-volume-control" title={labels.volume}>
                <span>{move || format!("{}%", volume_to_percent(volume))}</span>
                <input
                    type="range"
                    min="0"
                    max="100"
                    aria-label={labels.volume}
                    prop:value=move || volume_to_percent(volume).to_string()
                    on:input=move |event| {
                        let value = event_target_value(&event)
                            .parse::<f32>()
                            .unwrap_or(volume * 100.0);
                        on_volume_change((value / 100.0).clamp(0.0, 1.0));
                    }
                />
            </label>

            <button
                class="icon-button"
                title={labels.settings}
                aria-label={labels.settings}
                on:click=move |_| on_open_settings()
            >
                {"⚙"}
            </button>
        </header>
    }
}

fn volume_to_percent(volume: f32) -> usize {
    (volume.clamp(0.0, 1.0) * 100.0).round() as usize
}
