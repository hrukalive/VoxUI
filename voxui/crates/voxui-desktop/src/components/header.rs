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
    on_open_live_monitor: impl Fn() + Send + Sync + 'static + Copy,
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
                aria_label=move || labels.model
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
                title={labels.live}
                aria-label={labels.live}
                on:click=move |_| on_open_live_monitor()
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
            </button>
            <button
                class="icon-button"
                title={labels.settings}
                aria-label={labels.settings}
                on:click=move |_| on_open_settings()
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
            </button>
        </header>
    }
}

fn volume_to_percent(volume: f32) -> usize {
    (volume.clamp(0.0, 1.0) * 100.0).round() as usize
}
