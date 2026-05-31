use leptos::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
}

impl SelectOption {
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
        }
    }
}

#[component]
pub fn CustomSelect(
    aria_label: impl Fn() -> &'static str + Send + Sync + Clone + 'static,
    value: impl Fn() -> String + Send + Sync + Clone + 'static,
    options: impl Fn() -> Vec<SelectOption> + Send + Sync + Clone + 'static,
    disabled: impl Fn() -> bool + Send + Sync + Clone + 'static,
    on_change: impl Fn(String) + Send + Sync + Clone + 'static,
    #[prop(default = "")] class: &'static str,
) -> impl IntoView {
    let (open, set_open) = signal(false);
    let (suppress_button_click, set_suppress_button_click) = signal(false);
    let (suppress_option_click, set_suppress_option_click) = signal(false);
    let (option_pointer_down, set_option_pointer_down) = signal(false);
    let class_name = if class.is_empty() {
        "custom-select".to_string()
    } else {
        format!("custom-select {class}")
    };

    let selected_value = value.clone();
    let selected_options = options.clone();
    let selected_label = move || {
        let current = selected_value();
        selected_options()
            .into_iter()
            .find(|option| option.value == current)
            .map(|option| option.label)
            .unwrap_or_default()
    };
    let button_disabled = disabled.clone();
    let mouse_down_disabled = disabled.clone();
    let click_disabled = disabled.clone();
    let menu_disabled_attr = disabled.clone();
    let menu_disabled_class = disabled.clone();
    let menu_options = options.clone();
    let menu_value = value.clone();
    let menu_on_change = on_change.clone();
    let button_aria_label = aria_label.clone();
    let menu_aria_label = aria_label.clone();

    view! {
        <div class=class_name on:focusout=move |_| {
            if !option_pointer_down.get() {
                set_open.set(false);
            }
        }>
            <button
                class="custom-select-button"
                type="button"
                aria-label=move || button_aria_label()
                aria-expanded=move || open.get().to_string()
                disabled=move || button_disabled()
                on:mousedown=move |event| {
                    if !mouse_down_disabled() && event.button() == 0 {
                        set_option_pointer_down.set(false);
                        set_suppress_option_click.set(false);
                        set_suppress_button_click.set(true);
                        set_open.update(|open| *open = !*open);
                    }
                }
                on:click=move |_| {
                    if suppress_button_click.get() {
                        set_suppress_button_click.set(false);
                        return;
                    }
                    if !click_disabled() {
                        set_suppress_option_click.set(false);
                        set_open.update(|open| *open = !*open);
                    }
                }
            >
                <span>{selected_label}</span>
                <span class="custom-select-arrow" aria-hidden="true"></span>
            </button>
            <div
                class="custom-select-menu"
                class:open=move || open.get() && !menu_disabled_class()
                hidden=move || !open.get() || menu_disabled_attr()
                role="listbox"
                aria-label=move || menu_aria_label()
            >
                {move || {
                    let current_value = menu_value();
                    menu_options()
                        .into_iter()
                        .map(|option| {
                            let selected = current_value == option.value;
                            let mouseup_on_change = menu_on_change.clone();
                            let click_on_change = menu_on_change.clone();
                            let mouseup_option_value = option.value.clone();
                            let click_option_value = option.value.clone();
                            let option_label = option.label;
                            view! {
                                <button
                                    class:selected=selected
                                    class="custom-select-option"
                                    type="button"
                                    role="option"
                                    aria-selected=selected.to_string()
                                    on:mousedown=move |event| {
                                        event.prevent_default();
                                        if event.button() == 0 {
                                            set_option_pointer_down.set(true);
                                        }
                                    }
                                    on:mouseup=move |event| {
                                        if event.button() == 0 {
                                            set_option_pointer_down.set(false);
                                            set_suppress_option_click.set(true);
                                            mouseup_on_change(mouseup_option_value.clone());
                                            set_open.set(false);
                                        }
                                    }
                                    on:click=move |_| {
                                        if suppress_option_click.get() {
                                            set_suppress_option_click.set(false);
                                            return;
                                        }
                                        click_on_change(click_option_value.clone());
                                        set_open.set(false);
                                    }
                                >
                                    {option_label}
                                </button>
                            }
                        })
                        .collect_view()
                }}
            </div>
        </div>
    }
}

#[component]
pub fn NumberCounter(
    aria_label: impl Fn() -> &'static str + Send + Sync + Clone + 'static,
    value: impl Fn() -> String + Send + Sync + Clone + 'static,
    disabled: impl Fn() -> bool + Send + Sync + Clone + 'static,
    on_change: impl Fn(String) + Send + Sync + Clone + 'static,
    #[prop(default = "1")] step: &'static str,
    #[prop(default = "")] min: &'static str,
) -> impl IntoView {
    let container_disabled = disabled.clone();
    let input_disabled = disabled.clone();
    let increment_disabled_attr = disabled.clone();
    let increment_disabled_click = disabled.clone();
    let decrement_disabled_attr = disabled.clone();
    let decrement_disabled_click = disabled.clone();
    let input_value = value.clone();
    let input_on_change = on_change.clone();
    let increment_value = value.clone();
    let increment_on_change = on_change.clone();
    let decrement_value = value.clone();
    let decrement_on_change = on_change.clone();
    let input_aria_label = aria_label.clone();

    view! {
        <div class="number-counter" class:disabled=move || container_disabled()>
            <input
                type="text"
                inputmode="decimal"
                aria-label=move || input_aria_label()
                disabled=move || input_disabled()
                prop:value=input_value
                on:change=move |event| input_on_change(event_target_value(&event))
            />
            <div class="number-counter-buttons" aria-hidden="true">
                <button
                    type="button"
                    tabindex="-1"
                    disabled=move || increment_disabled_attr()
                    on:click=move |_| {
                        if !increment_disabled_click() {
                            increment_on_change(adjust_number(&increment_value(), step, min, 1.0));
                        }
                    }
                ></button>
                <button
                    type="button"
                    tabindex="-1"
                    disabled=move || decrement_disabled_attr()
                    on:click=move |_| {
                        if !decrement_disabled_click() {
                            decrement_on_change(adjust_number(&decrement_value(), step, min, -1.0));
                        }
                    }
                ></button>
            </div>
        </div>
    }
}

fn adjust_number(current: &str, step: &str, min: &str, direction: f64) -> String {
    let current = current.parse::<f64>().unwrap_or(0.0);
    let step_value = step.parse::<f64>().unwrap_or(1.0);
    let min_value = min.parse::<f64>().ok();
    let precision = decimal_places(step);
    let mut next = current + (step_value * direction);

    if let Some(min_value) = min_value {
        next = next.max(min_value);
    }

    format_number(next, precision)
}

fn decimal_places(value: &str) -> usize {
    value
        .split_once('.')
        .map(|(_, fraction)| fraction.len())
        .unwrap_or(0)
}

fn format_number(value: f64, precision: usize) -> String {
    if precision == 0 {
        (value.round() as i64).to_string()
    } else {
        format!("{value:.precision$}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn aria_labels_are_reactive_functions() {
        let source = include_str!("controls.rs").replace("\r\n", "\n");

        assert!(
            source.contains("aria_label: impl Fn() -> &'static str"),
            "shared controls should accept aria-label closures so callers do not evaluate signal-backed labels during component construction"
        );
        assert!(
            source.contains("aria-label=move || button_aria_label()"),
            "CustomSelect button aria-label should be read from a reactive closure"
        );
        assert!(
            source.contains("aria-label=move || menu_aria_label()"),
            "CustomSelect menu aria-label should be read from a reactive closure"
        );
        assert!(
            source.contains("aria-label=move || input_aria_label()"),
            "NumberCounter input aria-label should be read from a reactive closure"
        );
    }
}

pub fn translation_lang_options(
    include_auto: bool,
    labels: &crate::i18n::Labels,
) -> Vec<SelectOption> {
    let mut options = Vec::new();

    if include_auto {
        options.push(SelectOption::new("auto", labels.auto_detect));
    }

    options.push(SelectOption::new("ZH", "中文"));
    options.push(SelectOption::new("EN", "English"));
    options.push(SelectOption::new("JA", "日本語"));

    let rest: &[(&str, &str)] = &[
        ("KO", "한국어"),
        ("DE", "Deutsch"),
        ("FR", "Français"),
        ("RU", "Русский"),
        ("ES", "Español"),
        ("VI", "Tiếng Việt"),
        ("IT", "Italiano"),
        ("EL", "Ελληνικά"),
        ("ID", "Bahasa Indonesia"),
        ("PL", "Polski"),
        ("PT-BR", "Português (Brasil)"),
        ("RO", "Română"),
        ("TR", "Türkçe"),
        ("AR", "العربية"),
        ("CS", "Čeština"),
    ];

    for (code, label) in rest {
        options.push(SelectOption::new(*code, *label));
    }

    options
}
