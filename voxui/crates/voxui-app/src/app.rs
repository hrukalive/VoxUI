use std::path::Path;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;
use voxui_audio::AudioSystem;

use crate::config::AppConfig;
use crate::history::{TtsEntry, TtsStatus};
use crate::i18n::{self, Language, Strings};
use crate::input::InputState;

#[derive(Debug, PartialEq, Eq)]
pub enum AppMode {
    Normal,
    Settings,
    ModelSelect,
}

pub enum TtsCommand {
    Synthesize { index: usize, text: String, dit_steps: usize },
    ReloadEngine { model_dir: String, backend: String },
    LoadLora(String),
    UnloadLora,
    Cancel,
}

pub enum UiUpdate {
    Progress(usize, usize, usize), // entry_index, step, total
    Completed(usize, Vec<f32>),    // entry_index, PCM samples
    Error(usize, String),          // entry_index, message
    EngineReady,
    EngineError(String),
}

/// (label, options, selected_index)
pub type SettingsField = (String, Vec<String>, usize);

pub struct App {
    pub mode: AppMode,
    pub history: Vec<TtsEntry>,
    pub history_scroll: usize,
    pub input: InputState,
    pub progress: f32,
    pub progress_msg: String,
    pub status_line: String,
    pub should_quit: bool,
    pub engine_ready: bool,
    pub language: Language,

    // Settings
    pub model_dir: String,
    pub audio_host: String,
    pub audio_device: String,
    pub backend: String,

    // Settings popup state
    pub settings_field: usize,
    pub settings_values: Vec<SettingsField>,

    // Model-select modal state
    pub model_select_input: InputState,
    pub model_select_error: String,

    // Async channels
    pub tts_tx: mpsc::Sender<TtsCommand>,
    pub ui_rx: mpsc::Receiver<UiUpdate>,
}

impl App {
    pub fn new(tts_tx: mpsc::Sender<TtsCommand>, ui_rx: mpsc::Receiver<UiUpdate>, config: AppConfig) -> Self {
        let model_options = scan_model_dirs("models");
        let lora_options = scan_lora_dirs(&config.model_dir);

        // Find indices matching saved config values
        let model_idx = model_options.iter().position(|s| s == &config.model_dir).unwrap_or(0);
        let backend_options = backend_options();
        let backend_idx = backend_options.iter().position(|s| s == &config.backend).unwrap_or(0);
        let audio_system = AudioSystem::new();
        let mut audio_options: Vec<String> = audio_system.hosts().iter().map(|h| h.name.clone()).collect();
        if audio_options.is_empty() {
            audio_options.push("No audio host".into());
        }
        let audio_idx = audio_options.iter().position(|s| s == &config.audio_host).unwrap_or(0);
        let default_host = audio_options[audio_idx].clone();
        let mut device_options: Vec<String> = audio_system
            .devices(&default_host)
            .map(|devs| devs.into_iter().map(|d| d.name).collect())
            .unwrap_or_default();
        if device_options.is_empty() {
            device_options.push("No audio device".into());
        }
        let device_idx = device_options.iter().position(|s| s == &config.audio_device).unwrap_or(0);
        let max_chars_options: Vec<String> = vec!["80".into(), "120".into(), "200".into()];
        let max_chars_idx = max_chars_options.iter().position(|s| s == &config.max_chars.to_string()).unwrap_or(0);
        let dit_steps_options: Vec<String> = vec!["5".into(), "10".into(), "15".into(), "20".into(), "30".into()];
        let dit_steps_idx = dit_steps_options.iter().position(|s| s == &config.dit_steps.to_string()).unwrap_or(1);
        let lora_idx = if let Some(ref lp) = config.lora_path {
            lora_options.iter().position(|s| s == lp).unwrap_or(0)
        } else {
            0
        };

        let language = config.language;
        let lang_options: Vec<String> = Language::ALL.iter().map(|l| l.display_name().to_string()).collect();
        let lang_idx = Language::ALL.iter().position(|&l| l == language).unwrap_or(0);

        let s = i18n::get_strings(language);

        let settings_values = vec![
            (s.settings_model.to_string(), model_options, model_idx),
            (s.settings_lora.to_string(), lora_options, lora_idx),
            (s.settings_backend.to_string(), backend_options, backend_idx),
            (s.settings_audio_host.to_string(), audio_options, audio_idx),
            (s.settings_audio_device.to_string(), device_options, device_idx),
            (s.settings_max_chars.to_string(), max_chars_options, max_chars_idx),
            (s.settings_dit_steps.to_string(), dit_steps_options, dit_steps_idx),
            (s.settings_language.to_string(), lang_options, lang_idx),
        ];

        let model_dir = settings_values[0].1.get(model_idx).cloned().unwrap_or_else(|| config.model_dir.clone());
        let max_chars = config.max_chars;

        let mut model_select_input = InputState::new(512);
        model_select_input.text = config.model_dir.clone();
        model_select_input.cursor = model_select_input.text.chars().count();

        Self {
            mode: AppMode::Normal,
            history: Vec::new(),
            history_scroll: 0,
            input: InputState::new(max_chars),
            progress: 0.0,
            progress_msg: String::new(),
            status_line: String::new(),
            should_quit: false,
            engine_ready: false,
            language,
            model_dir,
            audio_host: config.audio_host.clone().into(),
            audio_device: config.audio_device.clone().into(),
            backend: config.backend.clone().into(),
            settings_field: 0,
            settings_values,
            model_select_input,
            model_select_error: String::new(),
            tts_tx,
            ui_rx,
        }
    }

    pub fn strings(&self) -> &'static Strings {
        i18n::get_strings(self.language)
    }

    /// Update settings field labels to match current language.
    fn refresh_settings_labels(&mut self) {
        let s = self.strings();
        let labels = [
            s.settings_model,
            s.settings_lora,
            s.settings_backend,
            s.settings_audio_host,
            s.settings_audio_device,
            s.settings_max_chars,
            s.settings_dit_steps,
            s.settings_language,
        ];
        for (i, label) in labels.iter().enumerate() {
            if i < self.settings_values.len() {
                self.settings_values[i].0 = label.to_string();
            }
        }
    }

    pub fn refresh_status_line(&mut self) {
        let s = self.strings();
        let ready = if self.engine_ready { s.status_ready } else { s.status_loading };
        self.status_line = format!(
            " {} | Model: {} | Backend: {} | Audio: {} / {}",
            ready, self.model_dir, self.backend, self.audio_host, self.audio_device,
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            AppMode::Normal => self.handle_normal_key(key),
            AppMode::Settings => self.handle_settings_key(key),
            AppMode::ModelSelect => self.handle_model_select_key(key),
        }
    }

    fn handle_normal_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }

        match key.code {
            KeyCode::Enter => self.submit_text(),
            KeyCode::F(2) => self.mode = AppMode::Settings,
            KeyCode::Esc => self.should_quit = true,
            KeyCode::Up => {
                self.history_scroll = self.history_scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.history.is_empty() {
                    self.history_scroll =
                        (self.history_scroll + 1).min(self.history.len().saturating_sub(1));
                }
            }
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.move_left(),
            KeyCode::Right => self.input.move_right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Char(c) => self.input.insert(c),
            _ => {}
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
        let field_count = self.settings_values.len();
        match key.code {
            KeyCode::F(2) | KeyCode::Esc => self.mode = AppMode::Normal,
            KeyCode::Enter => {
                self.apply_settings();
                self.mode = AppMode::Normal;
            }
            KeyCode::Tab => {
                self.settings_field = (self.settings_field + 1) % field_count;
            }
            KeyCode::BackTab => {
                self.settings_field = if self.settings_field == 0 {
                    field_count - 1
                } else {
                    self.settings_field - 1
                };
            }
            KeyCode::Up | KeyCode::Left => {
                let field = &mut self.settings_values[self.settings_field];
                if field.2 > 0 {
                    field.2 -= 1;
                }
            }
            KeyCode::Down | KeyCode::Right => {
                let field = &mut self.settings_values[self.settings_field];
                if field.2 + 1 < field.1.len() {
                    field.2 += 1;
                }
            }
            _ => {}
        }
    }

    fn apply_settings(&mut self) {
        let new_model_dir = self.settings_values[0].1[self.settings_values[0].2].clone();
        // index 1 = LoRA (unused for now)
        let new_backend = self.settings_values[2].1[self.settings_values[2].2].clone();
        self.audio_host = self.settings_values[3].1[self.settings_values[3].2].clone();
        self.audio_device = self.settings_values[4].1[self.settings_values[4].2].clone();
        let max_chars_str = &self.settings_values[5].1[self.settings_values[5].2];
        if let Ok(v) = max_chars_str.parse::<usize>() {
            self.input.max_chars = v;
        }

        // Diffusion steps (index 6) — stored in config for engine
        let dit_steps_str = &self.settings_values[6].1[self.settings_values[6].2];
        let dit_steps = dit_steps_str.parse::<usize>().unwrap_or(10);

        // Language (index 7)
        let lang_idx = self.settings_values[7].2;
        self.language = Language::ALL[lang_idx];
        self.refresh_settings_labels();

        // If model dir or backend changed, reload the engine
        let needs_reload = new_model_dir != self.model_dir || new_backend != self.backend;
        self.model_dir = new_model_dir;
        self.backend = new_backend;
        self.refresh_status_line();

        if needs_reload {
            self.engine_ready = false;
            self.refresh_status_line();
            let tx = self.tts_tx.clone();
            let _ = tx.try_send(TtsCommand::ReloadEngine {
                model_dir: self.model_dir.clone(),
                backend: self.backend.clone(),
            });
        }

        // Send LoRA command
        let lora_val = &self.settings_values[1].1[self.settings_values[1].2];
        let s = self.strings();
        {
            let tx = self.tts_tx.clone();
            if lora_val == "None" || lora_val == s.none {
                let _ = tx.try_send(TtsCommand::UnloadLora);
            } else {
                let lora_path = format!("{}/{}", self.model_dir, lora_val);
                let _ = tx.try_send(TtsCommand::LoadLora(lora_path));
            }
        }

        let config = AppConfig {
            model_dir: self.model_dir.clone(),
            lora_path: if lora_val == "None" || lora_val == s.none { None } else { Some(lora_val.clone()) },
            backend: self.backend.clone(),
            audio_host: self.audio_host.clone(),
            audio_device: self.audio_device.clone(),
            max_chars: self.input.max_chars,
            dit_steps,
            language: self.language,
        };
        let _ = config.save();
    }

    fn handle_model_select_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }
        // Ctrl+V paste support
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('v') {
            // crossterm doesn't provide clipboard; paste events come as bracketed paste
            return;
        }
        match key.code {
            KeyCode::Enter => self.confirm_model_select(),
            KeyCode::Esc => self.should_quit = true,
            KeyCode::Backspace => self.model_select_input.backspace(),
            KeyCode::Delete => self.model_select_input.delete(),
            KeyCode::Left => self.model_select_input.move_left(),
            KeyCode::Right => self.model_select_input.move_right(),
            KeyCode::Home => self.model_select_input.home(),
            KeyCode::End => self.model_select_input.end(),
            KeyCode::Char(c) => self.model_select_input.insert(c),
            _ => {}
        }
    }

    fn confirm_model_select(&mut self) {
        let path_str = self.model_select_input.text.trim().to_string();
        // Accept both forward and backslash paths on Windows
        let path = std::path::PathBuf::from(&path_str);
        if path.join("base_lm.gguf").exists() {
            // Normalize path for storage (use OS-native separators)
            let normalized = path.to_string_lossy().to_string();
            self.model_dir = normalized.clone();
            self.model_select_error.clear();
            self.mode = AppMode::Normal;

            // Update config model_dir in settings_values[0] if present
            if !self.settings_values[0].1.contains(&normalized) {
                self.settings_values[0].1.push(normalized.clone());
            }
            let idx = self.settings_values[0].1.iter().position(|s| s == &normalized).unwrap_or(0);
            self.settings_values[0].2 = idx;

            // Save config
            let lora_val = &self.settings_values[1].1[self.settings_values[1].2];
            let saved_lora_path = lora_val.clone();
            let dit_steps_str = &self.settings_values[6].1[self.settings_values[6].2];
            let dit_steps = dit_steps_str.parse::<usize>().unwrap_or(10);
            let s = self.strings();
            let config = AppConfig {
                model_dir: self.model_dir.clone(),
                lora_path: if saved_lora_path == "None" || saved_lora_path == s.none { None } else { Some(saved_lora_path.clone()) },
                backend: self.backend.clone(),
                audio_host: self.audio_host.clone(),
                audio_device: self.audio_device.clone(),
                max_chars: self.input.max_chars,
                dit_steps,
                language: self.language,
            };
            let _ = config.save();

            // Signal engine to load
            self.engine_ready = false;
            self.refresh_status_line();
            let tx = self.tts_tx.clone();
            let _ = tx.try_send(TtsCommand::ReloadEngine {
                model_dir: self.model_dir.clone(),
                backend: self.backend.clone(),
            });

            // Re-apply LoRA if one was configured
            if saved_lora_path != "None" && saved_lora_path != s.none {
                let lora_path = format!("{}/{}", self.model_dir, saved_lora_path);
                let _ = tx.try_send(TtsCommand::LoadLora(lora_path));
            }
        } else {
            self.model_select_error = self.strings().model_not_found_error.to_string();
        }
    }

    fn submit_text(&mut self) {
        let text = self.input.text.trim().to_string();
        if text.is_empty() || !self.engine_ready {
            return;
        }
        let _ = self.input.take();
        self.history.push(TtsEntry::new(text.clone()));
        let index = self.history.len() - 1;
        self.history_scroll = index;

        // Send to inference task
        let tx = self.tts_tx.clone();
        let dit_steps_str = &self.settings_values[6].1[self.settings_values[6].2];
        let dit_steps = dit_steps_str.parse::<usize>().unwrap_or(10);
        let _ = tx.try_send(TtsCommand::Synthesize { index, text, dit_steps });
    }

    pub fn poll_updates(&mut self) {
        while let Ok(update) = self.ui_rx.try_recv() {
            match update {
                UiUpdate::Progress(idx, step, total) => {
                    if let Some(entry) = self.history.get_mut(idx) {
                        let pct = if total > 0 { step as f32 / total as f32 } else { 0.0 };
                        entry.status = TtsStatus::Generating(pct);
                        self.progress = pct;
                        self.progress_msg = format!("{}/{}", step, total);
                    }
                }
                UiUpdate::Completed(idx, _samples) => {
                    if let Some(entry) = self.history.get_mut(idx) {
                        entry.status = TtsStatus::Done;
                    }
                    self.progress = 0.0;
                    self.progress_msg.clear();
                    // Audio playback would go here
                }
                UiUpdate::Error(idx, msg) => {
                    if let Some(entry) = self.history.get_mut(idx) {
                        entry.status = TtsStatus::Error(msg);
                    }
                    self.progress = 0.0;
                    self.progress_msg.clear();
                }
                UiUpdate::EngineReady => {
                    self.engine_ready = true;
                    self.refresh_status_line();
                }
                UiUpdate::EngineError(msg) => {
                    self.engine_ready = false;
                    self.status_line = format!(" Engine error: {}", msg);
                }
            }
        }
    }

    pub fn update_progress(&mut self, step: usize, total: usize) {
        if total > 0 {
            self.progress = step as f32 / total as f32;
        }
    }
}

/// Get available backend options based on compile-time and runtime CUDA availability.
fn backend_options() -> Vec<String> {
    let mut opts = vec!["CPU".to_string()];
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            opts.push("CUDA".to_string());
        } else {
            opts.push("CUDA (unavailable)".to_string());
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        opts.push("CUDA (not compiled)".to_string());
    }
    opts
}

/// Scan `models/` directory for subdirectories containing GGUF files.
fn scan_model_dirs(base: &str) -> Vec<String> {
    let base_path = Path::new(base);
    let mut dirs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(base_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("base_lm.gguf").exists() {
                if let Some(name) = path.to_str() {
                    dirs.push(name.replace('\\', "/"));
                }
            }
        }
    }
    dirs.sort();
    if dirs.is_empty() {
        dirs.push("models".into());
    }
    dirs
}

/// Scan model directory for LoRA subdirectories (names starting with `lora_`).
fn scan_lora_dirs(model_dir: &str) -> Vec<String> {
    let mut options = vec!["None".to_string()];
    let path = std::path::Path::new(model_dir);
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("lora_") && entry.path().is_dir() {
                // Verify it contains at least one lora_*.gguf file
                if entry.path().join("lora_base_lm.gguf").exists() {
                    options.push(name);
                }
            }
        }
    }
    options
}
