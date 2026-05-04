use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph},
    Frame,
};
use unicode_width::UnicodeWidthChar;

use crate::app::{App, AppMode};
use crate::history::TtsStatus;

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Min(5),    // history
        Constraint::Length(1), // progress
        Constraint::Length(3), // input
        Constraint::Length(1), // status
    ])
    .split(f.area());

    render_history(f, app, chunks[0]);
    render_progress(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
    render_status(f, app, chunks[3]);

    if app.mode == AppMode::Settings {
        render_settings_popup(f, app);
    }

    if app.mode == AppMode::ModelSelect {
        render_model_select_popup(f, app);
    }
}

fn render_history(f: &mut Frame, app: &App, area: Rect) {
    let s = app.strings();
    let items: Vec<ListItem> = app
        .history
        .iter()
        .map(|entry| {
            let (icon, color) = entry.status_icon();
            let mut spans = vec![
                Span::styled(
                    format!("[{}] ", entry.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(&entry.text),
                Span::raw(" "),
                Span::styled(icon, Style::default().fg(color)),
            ];
            if let TtsStatus::Error(ref msg) = entry.status {
                spans.push(Span::styled(
                    format!(" — {}", msg),
                    Style::default().fg(Color::Red),
                ));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title(format!(" {} - {} ", s.app_title, s.tts_history))
                .borders(Borders::ALL),
        )
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(list, area);
}

fn render_progress(f: &mut Frame, app: &App, area: Rect) {
    let label = if !app.engine_ready && app.progress == 0.0 {
        app.strings().status_loading.to_string()
    } else if app.progress_msg.is_empty() {
        if app.progress > 0.0 {
            format!("{:.0}%", app.progress * 100.0)
        } else {
            app.strings().status_ready.to_string()
        }
    } else {
        format!("{:.0}% ({})", app.progress * 100.0, app.progress_msg)
    };

    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(app.progress as f64)
        .label(label);

    f.render_widget(gauge, area);
}

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let s = app.strings();
    let display = if app.input.text.is_empty() {
        Span::styled(
            s.input_placeholder,
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw(&app.input.text)
    };

    let input = Paragraph::new(Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Cyan)),
        display,
    ]))
    .block(Block::default().borders(Borders::ALL).title(" Input "));

    f.render_widget(input, area);

    if app.mode == AppMode::Normal {
        let cursor_col = cursor_col_width(&app.input.text, app.input.cursor);
        #[allow(clippy::cast_possible_truncation)]
        f.set_cursor_position((area.x + 3 + cursor_col, area.y + 1));
    }
}

fn cursor_col_width(input: &str, cursor_pos: usize) -> u16 {
    input
        .chars()
        .take(cursor_pos)
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(1) as u16)
        .sum()
}

fn render_status(f: &mut Frame, app: &App, area: Rect) {
    let s = app.strings();
    let hint = format!("  {}", s.input_hint);
    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            &app.status_line,
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(hint, Style::default().fg(Color::DarkGray)),
    ]));
    f.render_widget(status, area);
}

fn render_settings_popup(f: &mut Frame, app: &App) {
    let s = app.strings();
    let area = centered_rect(55, 50, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(format!(" {} ", s.settings_title))
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = vec![Line::from("")];

    for (i, (label, options, selected)) in app.settings_values.iter().enumerate() {
        let is_selected = i == app.settings_field;
        let value = options.get(*selected).map_or("???", |s| s.as_str());

        let label_style = if is_selected {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let value_style = if is_selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };

        let arrow = if is_selected { ">" } else { " " };

        lines.push(Line::from(vec![
            Span::styled(format!(" {} {:<10}", arrow, label), label_style),
            Span::styled(format!("[▾ {:<24}]", value), value_style),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("[Enter: {}]", s.settings_apply), Style::default().fg(Color::Green)),
        Span::raw("  "),
        Span::styled(format!("[Esc: {}]", s.settings_cancel), Style::default().fg(Color::Red)),
        Span::raw("  "),
        Span::styled(format!("[Tab: {}]", s.settings_next), Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(format!("[↑↓: {}]", s.settings_change), Style::default().fg(Color::DarkGray)),
    ]));

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

fn render_model_select_popup(f: &mut Frame, app: &App) {
    let s = app.strings();
    let area = centered_rect(65, 40, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(format!(" {} ", s.model_not_found_title))
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", s.model_not_found_msg),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(""),
    ];

    // Input field
    let input_text = if app.model_select_input.text.is_empty() {
        "."
    } else {
        &app.model_select_input.text
    };
    lines.push(Line::from(vec![
        Span::raw("  ┌"),
        Span::raw("─".repeat((inner.width as usize).saturating_sub(6))),
        Span::raw("┐"),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  │ "),
        Span::styled(input_text, Style::default().fg(Color::White)),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  └"),
        Span::raw("─".repeat((inner.width as usize).saturating_sub(6))),
        Span::raw("┘"),
    ]));

    lines.push(Line::from(""));

    // Hint
    lines.push(Line::from(Span::styled(
        format!("  {}", s.model_not_found_hint),
        Style::default().fg(Color::DarkGray),
    )));

    // Error message if any
    if !app.model_select_error.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {}", app.model_select_error),
            Style::default().fg(Color::Red),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("[Enter: {}]", s.confirm),
            Style::default().fg(Color::Green),
        ),
        Span::raw("  "),
        Span::styled(
            "[Esc: quit]",
            Style::default().fg(Color::Red),
        ),
    ]));

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    // Place cursor in the input field
    // Line index 4 (0-based) in inner = the input text line
    let cursor_col = cursor_col_width(&app.model_select_input.text, app.model_select_input.cursor);
    #[allow(clippy::cast_possible_truncation)]
    f.set_cursor_position((inner.x + 4 + cursor_col, inner.y + 4));
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(r);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}
