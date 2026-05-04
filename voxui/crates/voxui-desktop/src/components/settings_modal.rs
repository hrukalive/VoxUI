use leptos::prelude::*;
use crate::i18n::Language;

#[component]
pub fn SettingsModal(
    lang: ReadSignal<Language>,
    /// Current values
    model_dir: ReadSignal<String>,
    lora_dir: ReadSignal<String>,
    backend: ReadSignal<String>,
    audio_host: ReadSignal<String>,
    audio_device: ReadSignal<String>,
    max_chars: ReadSignal<usize>,
    dit_steps: ReadSignal<usize>,
    /// Available options
    models: ReadSignal<Vec<String>>,
    loras: ReadSignal<Vec<String>>,
    hosts: ReadSignal<Vec<String>>,
    devices: ReadSignal<Vec<String>>,
    /// Callbacks
    on_close: impl Fn(()) + 'static + Clone,
    on_apply: impl Fn(SettingsValues) + 'static,
) -> impl IntoView {
    let (sel_model, set_sel_model) = signal(model_dir.get_untracked());
    let (sel_lora, set_sel_lora) = signal(lora_dir.get_untracked());
    let (sel_backend, set_sel_backend) = signal(backend.get_untracked());
    let (sel_host, set_sel_host) = signal(audio_host.get_untracked());
    let (sel_device, set_sel_device) = signal(audio_device.get_untracked());
    let (sel_max_chars, set_sel_max_chars) = signal(max_chars.get_untracked());
    let (sel_dit_steps, set_sel_dit_steps) = signal(dit_steps.get_untracked());

    let on_close_apply = on_close.clone();

    view! {
        <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-lg shadow-xl w-[480px] max-h-[90vh] overflow-y-auto border border-gray-600">
                <div class="flex items-center justify-between px-4 py-3 border-b border-gray-700">
                    <h2 class="text-lg font-semibold">{move || lang.get().t("settings")}</h2>
                    <button class="text-gray-400 hover:text-white" on:click={
                        let on_close = on_close.clone();
                        move |_| on_close(())
                    }>"✕"</button>
                </div>
                <div class="p-4 space-y-4">
                    // Model directory
                    <SettingsField label=move || lang.get().t("model")>
                        <select
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            on:change=move |ev| set_sel_model.set(event_target_value(&ev))
                        >
                            <For
                                each=move || models.get()
                                key=|m| m.clone()
                                children=move |m| {
                                    let selected = m == sel_model.get();
                                    let m2 = m.clone();
                                    view! { <option value={m} selected=selected>{m2}</option> }
                                }
                            />
                        </select>
                    </SettingsField>

                    // LoRA
                    <SettingsField label=move || lang.get().t("lora")>
                        <select
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            on:change=move |ev| set_sel_lora.set(event_target_value(&ev))
                        >
                            <For
                                each=move || loras.get()
                                key=|l| l.clone()
                                children=move |l| {
                                    let selected = l == sel_lora.get();
                                    let l2 = l.clone();
                                    view! { <option value={l} selected=selected>{l2}</option> }
                                }
                            />
                        </select>
                    </SettingsField>

                    // Backend
                    <SettingsField label=move || lang.get().t("backend")>
                        <select
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            on:change=move |ev| set_sel_backend.set(event_target_value(&ev))
                        >
                            <option value="CPU" selected=move || sel_backend.get() == "CPU">"CPU"</option>
                            <option value="CUDA" selected=move || sel_backend.get() == "CUDA">"CUDA"</option>
                        </select>
                    </SettingsField>

                    // Audio host
                    <SettingsField label=move || lang.get().t("audio_host")>
                        <select
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            on:change=move |ev| set_sel_host.set(event_target_value(&ev))
                        >
                            <For
                                each=move || hosts.get()
                                key=|h| h.clone()
                                children=move |h| {
                                    let selected = h == sel_host.get();
                                    let h2 = h.clone();
                                    view! { <option value={h} selected=selected>{h2}</option> }
                                }
                            />
                        </select>
                    </SettingsField>

                    // Audio device
                    <SettingsField label=move || lang.get().t("audio_device")>
                        <select
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            on:change=move |ev| set_sel_device.set(event_target_value(&ev))
                        >
                            <For
                                each=move || devices.get()
                                key=|d| d.clone()
                                children=move |d| {
                                    let selected = d == sel_device.get();
                                    let d2 = d.clone();
                                    view! { <option value={d} selected=selected>{d2}</option> }
                                }
                            />
                        </select>
                    </SettingsField>

                    // Max chars
                    <SettingsField label=move || lang.get().t("max_chars")>
                        <input
                            type="number"
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            prop:value=move || sel_max_chars.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                    set_sel_max_chars.set(v);
                                }
                            }
                        />
                    </SettingsField>

                    // DIT steps
                    <SettingsField label=move || lang.get().t("dit_steps")>
                        <input
                            type="number"
                            class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
                            prop:value=move || sel_dit_steps.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                    set_sel_dit_steps.set(v);
                                }
                            }
                        />
                    </SettingsField>
                </div>
                <div class="flex justify-end gap-2 px-4 py-3 border-t border-gray-700">
                    <button
                        class="px-4 py-1.5 rounded bg-gray-600 hover:bg-gray-500 text-sm"
                        on:click={
                            let on_close = on_close_apply.clone();
                            move |_| on_close(())
                        }
                    >
                        {move || lang.get().t("cancel")}
                    </button>
                    <button
                        class="px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-sm font-medium"
                        on:click=move |_| {
                            on_apply(SettingsValues {
                                model_dir: sel_model.get(),
                                lora_dir: sel_lora.get(),
                                backend: sel_backend.get(),
                                audio_host: sel_host.get(),
                                audio_device: sel_device.get(),
                                max_chars: sel_max_chars.get(),
                                dit_steps: sel_dit_steps.get(),
                            });
                        }
                    >
                        {move || lang.get().t("apply")}
                    </button>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Debug)]
pub struct SettingsValues {
    pub model_dir: String,
    pub lora_dir: String,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    pub dit_steps: usize,
}

#[component]
fn SettingsField(
    label: impl Fn() -> &'static str + Send + 'static,
    children: Children,
) -> impl IntoView {
    view! {
        <div>
            <label class="block text-sm text-gray-400 mb-1">{label}</label>
            {children()}
        </div>
    }
}
