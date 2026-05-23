use leptos::prelude::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <div class="app-shell">
            <header class="app-header">
                <div class="brand">
                    <strong>"焓言焓语"</strong>
                    <span>"AhanSays"</span>
                </div>
            </header>
            <section class="history-panel"></section>
            <footer class="composer-panel"></footer>
        </div>
    }
}
