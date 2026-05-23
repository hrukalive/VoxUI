mod app;
mod i18n;
mod tauri_api;

use wasm_bindgen::JsCast;

fn main() {
    let mount_target = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("app"))
        .and_then(|element| element.dyn_into::<web_sys::HtmlElement>().ok())
        .expect("missing #app mount target");

    leptos::mount::mount_to(mount_target, app::App).forget();
}
