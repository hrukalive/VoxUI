use chrono::Local;

#[derive(Debug, Clone)]
pub enum TtsStatus {
    Queued,
    Generating(f32),
    Playing,
    Done,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct TtsEntry {
    pub timestamp: String,
    pub text: String,
    pub status: TtsStatus,
}

impl TtsEntry {
    pub fn new(text: String) -> Self {
        Self {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            text,
            status: TtsStatus::Queued,
        }
    }

    pub fn status_icon(&self) -> (&str, ratatui::style::Color) {
        use ratatui::style::Color;
        match &self.status {
            TtsStatus::Done => ("✓", Color::Green),
            TtsStatus::Playing => ("▶", Color::Yellow),
            TtsStatus::Generating(_) => ("⏳", Color::Yellow),
            TtsStatus::Queued => ("…", Color::DarkGray),
            TtsStatus::Error(_) => ("✗", Color::Red),
        }
    }
}
