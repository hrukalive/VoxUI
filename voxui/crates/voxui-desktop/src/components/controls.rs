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
    aria_label: &'static str,
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

    view! {
        <div class=class_name on:focusout=move |_| {
            if !option_pointer_down.get() {
                set_open.set(false);
            }
        }>
            <button
                class="custom-select-button"
                type="button"
                aria-label=aria_label
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
                aria-label=aria_label
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
    aria_label: &'static str,
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

    view! {
        <div class="number-counter" class:disabled=move || container_disabled()>
            <input
                type="text"
                inputmode="decimal"
                aria-label=aria_label
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

pub fn translation_lang_options(include_auto: bool, labels: &crate::i18n::Labels) -> Vec<SelectOption> {
    let mut options = Vec::new();

    if include_auto {
        options.push(SelectOption::new("auto", labels.auto_detect));
    }

    options.push(SelectOption::new("ZH", "中文"));
    options.push(SelectOption::new("EN", "English"));
    options.push(SelectOption::new("JA", "日本語"));

    let rest: &[(&str, &str)] = &[
        ("AR", "العربية"), ("BG", "Български"), ("CS", "Čeština"), ("DA", "Dansk"),
        ("DE", "Deutsch"), ("EL", "Ελληνικά"), ("EN-GB", "English (UK)"), ("EN-US", "English (US)"),
        ("ES", "Español"), ("ES-419", "Español (Latinoamérica)"), ("ET", "Eesti"),
        ("FI", "Suomi"), ("FR", "Français"), ("HE", "עברית"), ("HU", "Magyar"),
        ("ID", "Bahasa Indonesia"), ("IT", "Italiano"), ("KO", "한국어"), ("LT", "Lietuvių"),
        ("LV", "Latviešu"), ("NB", "Norsk"), ("NL", "Nederlands"), ("PL", "Polski"),
        ("PT-BR", "Português (Brasil)"), ("PT-PT", "Português (Portugal)"),
        ("RO", "Română"), ("RU", "Русский"), ("SK", "Slovenčina"), ("SL", "Slovenščina"),
        ("SV", "Svenska"), ("TR", "Türkçe"), ("UK", "Українська"), ("VI", "Tiếng Việt"),
        ("ZH-HANT", "繁體中文"),
    ];

    for (code, label) in rest {
        options.push(SelectOption::new(*code, *label));
    }

    options
}
