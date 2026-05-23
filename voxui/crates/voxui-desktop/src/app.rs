use leptos::prelude::*;

use crate::components::header::Header;
use crate::components::history::HistoryList;
use crate::components::input_box::InputBox;
use crate::components::load_progress_modal::LoadProgressModal;
use crate::components::settings_modal::SettingsModal;
use crate::i18n::{labels, UiLanguage};
use crate::tauri_api::ModelChoice;

#[component]
pub fn App() -> impl IntoView {
    let labels = labels(UiLanguage::Chinese);
    let (settings_open, set_settings_open) = signal(false);
    let (load_open, set_load_open) = signal(false);
    let models = fallback_models();
    let max_chars = 300;

    view! {
        <div class="app-shell">
            <Header
                labels=labels
                models=models
                selected_model_id=Some("ahan-default".to_owned())
                loaded_model_id=None
                load_disabled=false
                on_model_select=|_| {}
                on_load=move || set_load_open.set(true)
                on_open_settings=move || set_settings_open.set(true)
            />
            <HistoryList
                labels=labels
                items=Vec::new()
                on_play=|_| {}
                on_regenerate=|_| {}
                on_cancel=|_| {}
            />
            <InputBox
                labels=labels
                max_chars=max_chars
                disabled=false
                on_generate=|_| {}
            />
            <SettingsModal
                labels=labels
                open=move || settings_open.get()
                on_close=move || set_settings_open.set(false)
            />
            <LoadProgressModal
                labels=labels
                open=move || load_open.get()
                percent=|| 42.0
                on_close=move || set_load_open.set(false)
            />
        </div>
    }
}

fn fallback_models() -> Vec<ModelChoice> {
    vec![ModelChoice {
        id: "ahan-default".to_owned(),
        display_name: "AhanSays Default".to_owned(),
        model_dir: String::new(),
        model_path: String::new(),
        lora_path: None,
        model_bytes: 0,
        lora_bytes: 0,
    }]
}
