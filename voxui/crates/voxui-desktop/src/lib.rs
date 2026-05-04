use wasm_bindgen::prelude::*;

mod app;
mod tauri_api;
mod i18n;
mod components;

#[wasm_bindgen(start)]
pub fn main() {
    leptos::mount::mount_to_body(app::App);
}
