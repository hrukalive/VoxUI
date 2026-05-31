use leptos::prelude::*;

use crate::components::controls::{CustomSelect, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::{LoraEntry, ModelChoice};

#[component]
pub fn Header(
    labels: Labels,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    load_disabled: bool,
    loras: Vec<LoraEntry>,
    selected_lora_id: Option<String>,
    lora_disabled: bool,
    on_lora_select: impl Fn(Option<String>) + Send + Sync + 'static + Copy,
    volume: f32,
    on_model_select: impl Fn(String) + Send + Sync + 'static + Copy,
    on_load: impl Fn() + Send + Sync + 'static + Copy,
    on_volume_input: impl Fn(f32) + Send + Sync + 'static + Copy,
    on_volume_commit: impl Fn(f32) + Send + Sync + 'static + Copy,
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

    let selected_lora_id = selected_lora_id.unwrap_or_default();
    let lora_options = {
        let loras = loras.clone();
        move || {
            let mut opts: Vec<SelectOption> = vec![
                SelectOption::new(String::new(), "None".to_string()),
            ];
            for lora in &loras {
                opts.push(SelectOption::new(lora.id.clone(), lora.display_name.clone()));
            }
            opts
        }
    };
    let current_lora_id = {
        let selected_lora_id = selected_lora_id.clone();
        move || selected_lora_id.clone()
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

            <CustomSelect
                class="lora-select"
                aria_label=move || labels.lora
                value=current_lora_id
                options=lora_options
                disabled=move || lora_disabled
                on_change=move |lora_id| on_lora_select(if lora_id.is_empty() { None } else { Some(lora_id) })
            />

            <label class="header-volume-control" title={labels.volume}>
                <span>{move || format!("{}%", volume_to_percent(volume))}</span>
                <input
                    type="range"
                    min="0"
                    max="100"
                aria-label={labels.volume}
                prop:value=move || volume_to_percent(volume).to_string()
                on:input=move |event| {
                    on_volume_input(volume_from_event(&event, volume));
                }
                on:change=move |event| {
                    on_volume_commit(volume_from_event(&event, volume));
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

fn volume_from_event(event: &web_sys::Event, fallback: f32) -> f32 {
    (event_target_value(event)
        .parse::<f32>()
        .unwrap_or(fallback * 100.0)
        / 100.0)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn volume_slider_previews_on_input_and_commits_on_change() {
        let source = include_str!("header.rs").replace("\r\n", "\n");
        let implementation = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("header implementation source");

        assert!(
            implementation.contains("on_volume_input"),
            "dragging the header volume slider should call a preview callback"
        );
        assert!(
            implementation.contains("on:input=move |event|"),
            "volume preview should run for every drag input event"
        );
        assert!(
            implementation.contains("on_volume_commit"),
            "releasing/changing the header volume slider should call a commit callback"
        );
        assert!(
            implementation.contains("on:change=move |event|"),
            "volume persistence should happen on change, not every input event"
        );
    }
}
