mod app;
mod config;
mod history;
mod i18n;
mod input;
mod ui;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use app::{App, TtsCommand, UiUpdate};
use config::AppConfig;
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

/// Process TTS commands in a loop.
fn process_commands(
    engine: &mut VoxCPMEngine,
    tts_rx: &mut mpsc::Receiver<TtsCommand>,
    ui_tx: &mpsc::Sender<UiUpdate>,
) {
    while let Some(cmd) = tts_rx.blocking_recv() {
        match cmd {
            TtsCommand::ReloadEngine { model_dir, backend } => {
                let device = select_device(&backend);
                let model_path = PathBuf::from(&model_dir);
                match VoxCPMEngine::load(&model_path, device) {
                    Ok(new_engine) => {
                        *engine = new_engine;
                        let _ = ui_tx.blocking_send(UiUpdate::EngineReady);
                    }
                    Err(e) => {
                        let _ = ui_tx.blocking_send(UiUpdate::EngineError(e.to_string()));
                    }
                }
            }
            TtsCommand::Synthesize {
                index,
                text,
                dit_steps,
                prompt_wav_path,
                prompt_text,
                reference_wav_path,
            } => {
                let tx = ui_tx.clone();
                let request = SynthesisRequest {
                    text,
                    prompt_wav_path: prompt_wav_path.map(PathBuf::from),
                    prompt_text,
                    reference_wav_path: reference_wav_path.map(PathBuf::from),
                    inference_timesteps: dit_steps,
                    ..SynthesisRequest::default()
                };
                match engine.generate(request, |step, total| {
                    let _ = tx.blocking_send(UiUpdate::Progress(index, step, total));
                }) {
                    Ok(samples) => {
                        let _ = tx.blocking_send(UiUpdate::Completed(index, samples));
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(UiUpdate::Error(index, e.to_string()));
                    }
                }
            }
            TtsCommand::LoadLora(path) => {
                if let Err(e) = engine.load_lora(std::path::Path::new(&path)) {
                    let _ = ui_tx.blocking_send(UiUpdate::EngineError(format!("LoRA load error: {}", e)));
                }
            }
            TtsCommand::UnloadLora => {
                engine.unload_lora();
            }
            TtsCommand::Cancel => { /* TODO */ }
        }
    }
}

/// Select the compute device based on backend string and compile-time features.
fn select_device(backend: &str) -> candle_core::Device {
    match backend {
        "CUDA" => {
            #[cfg(feature = "cuda")]
            {
                match candle_core::Device::new_cuda(0) {
                    Ok(dev) => {
                        eprintln!("Using CUDA device 0");
                        dev
                    }
                    Err(e) => {
                        eprintln!("CUDA requested but failed: {}. Falling back to CPU.", e);
                        candle_core::Device::Cpu
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("CUDA requested but not compiled with --features cuda. Using CPU.");
                candle_core::Device::Cpu
            }
        }
        _ => candle_core::Device::Cpu,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install panic hook that restores terminal before printing panic info
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let saved_config = AppConfig::load();

    let (tts_tx, mut tts_rx) = mpsc::channel::<TtsCommand>(32);
    let (ui_tx, ui_rx) = mpsc::channel::<UiUpdate>(64);

    // Check if model exists before starting engine
    let model_path = PathBuf::from(&saved_config.model_dir);
    let has_model = model_path.join("model.gguf").exists();

    // Load engine in background, then serve TTS requests
    let engine_ui_tx = ui_tx.clone();
    let model_dir_for_engine = saved_config.model_dir.clone();
    let backend_for_engine = saved_config.backend.clone();
    tokio::task::spawn_blocking(move || {
        // If model doesn't exist at startup, wait for a ReloadEngine command first
        if !has_model {
            // Wait for user to pick a valid model path via ModelSelect modal
            if let Some(cmd) = tts_rx.blocking_recv() {
                match cmd {
                    TtsCommand::ReloadEngine { model_dir, backend } => {
                        let device = select_device(&backend);
                        let model_path = PathBuf::from(&model_dir);
                        match VoxCPMEngine::load(&model_path, device) {
                            Ok(mut engine) => {
                                let _ = engine_ui_tx.blocking_send(UiUpdate::EngineReady);
                                // Fall through to command loop
                                process_commands(&mut engine, &mut tts_rx, &engine_ui_tx);
                            }
                            Err(e) => {
                                let _ = engine_ui_tx.blocking_send(UiUpdate::EngineError(e.to_string()));
                            }
                        }
                    }
                    _ => return,
                }
            }
            return;
        }

        let model_dir = PathBuf::from(&model_dir_for_engine);
        let device = select_device(&backend_for_engine);

        let mut engine = match VoxCPMEngine::load(&model_dir, device) {
            Ok(engine) => {
                let _ = engine_ui_tx.blocking_send(UiUpdate::EngineReady);
                engine
            }
            Err(e) => {
                let _ = engine_ui_tx.blocking_send(UiUpdate::EngineError(e.to_string()));
                return;
            }
        };

        process_commands(&mut engine, &mut tts_rx, &engine_ui_tx);
    });

    let mut app = App::new(tts_tx, ui_rx, saved_config);
    if !has_model {
        app.mode = app::AppMode::ModelSelect;
    }
    app.refresh_status_line();

    loop {
        terminal.draw(|f| ui::render(f, &app))?;

        // Poll async updates from inference
        app.poll_updates();

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                // Only handle key press events (ignore release/repeat on Windows)
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key);
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
