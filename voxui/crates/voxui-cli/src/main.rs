mod args;
mod runner;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use candle_core::Device;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use args::Args;
use runner::Runner;

fn main() -> Result<()> {
    let args = Args::parse();
    args.validate()?;

    let device = if args.cuda {
        Device::new_cuda(0).context("CUDA device not available")?
    } else {
        Device::Cpu
    };

    let mut runner = Runner::load(&args.model, args.lora, device)?;
    runner.display_info();

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_handler = cancel.clone();
    ctrlc::set_handler(move || {
        cancel_handler.store(true, Ordering::SeqCst);
    })
    .context("failed to set Ctrl+C handler")?;

    let mut rl = DefaultEditor::new().context("failed to create line editor")?;

    println!("Type text to synthesize. Empty line or /quit to exit.\n");

    loop {
        // Reset cancel flag before each prompt
        cancel.store(false, Ordering::SeqCst);

        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() || line == "/quit" || line == "/exit" {
                    break;
                }
                rl.add_history_entry(&line)
                    .context("failed to save history")?;

                match runner.synthesize_and_play(&line, Some(&cancel)) {
                    Ok(()) => {
                        if cancel.load(Ordering::SeqCst) {
                            println!("  Cancelled.");
                        } else {
                            println!("  Done.");
                        }
                    }
                    Err(e) => {
                        eprintln!("  Error: {:#}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C at the prompt — exit
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("  Error: {err}");
                break;
            }
        }
    }

    println!("Goodbye.");
    Ok(())
}
