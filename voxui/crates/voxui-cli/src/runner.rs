use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

pub struct Runner {
    engine: VoxCPMEngine,
    lora_path: Option<PathBuf>,
    device_kind: &'static str,
}

impl Runner {
    /// Load the model from `model_dir` on the given device. Applies LoRA if provided.
    pub fn load(
        model_dir: &Path,
        lora_path: Option<PathBuf>,
        device: Device,
    ) -> Result<Self> {
        let device_kind = if device.is_cuda() { "CUDA" } else { "CPU" };

        let mut engine = VoxCPMEngine::load(model_dir, device)
            .context("failed to load model")?;

        if let Some(ref lp) = lora_path {
            engine
                .load_lora(lp)
                .context("failed to load LoRA adapter")?;
        }

        Ok(Self {
            engine,
            lora_path,
            device_kind,
        })
    }

    /// Synthesize `text` and play audio. In streaming mode, audio plays as chunks
    /// arrive (retry_badcase is disabled). In batch mode, the full waveform is
    /// generated first, then played at once.
    /// Checks `cancel`; returns early if set.
    pub fn synthesize_and_play(
        &mut self,
        text: &str,
        stream: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        if stream {
            self.synthesize_streaming(text, cancel)
        } else {
            self.synthesize_batch(text, cancel)
        }
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        let sample_rate = self.engine.sample_rate();

        let request = SynthesisRequest {
            text: text.to_string(),
            retry_badcase: false,
            ..Default::default()
        };

        let mut samples = Vec::new();
        let mut patch_count = 0usize;
        let mut max_patches = 0usize;
        let started = Instant::now();

        let result = self.engine.generate_streaming_cancellable(
            request,
            |chunk| {
                patch_count = chunk.generated_patch_count;
                max_patches = chunk.max_patches;
                let bar_width = 20usize;
                let filled = bar_width
                    .saturating_mul(patch_count)
                    .checked_div(max_patches.max(1))
                    .unwrap_or(0);
                let bar: String = std::iter::repeat_n('=', filled)
                    .chain(std::iter::repeat_n('>', if patch_count < max_patches { 1 } else { 0 }))
                    .chain(std::iter::repeat_n(' ', bar_width.saturating_sub(filled).saturating_sub(1)))
                    .collect();
                eprint!(
                    "\r  Synthesizing... [{bar}] {patch_count}/{max_patches} patches",
                );
                samples.extend_from_slice(&chunk.samples);
                Ok(())
            },
            cancel,
        );

        if let Err(e) = result {
            if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                eprintln!();
                return Ok(());
            }
            eprintln!();
            return Err(e).context("synthesis failed");
        }

        let elapsed = started.elapsed().as_secs_f64();
        eprintln!();
        eprintln!("  Synthesis: {:.2}s", elapsed);

        if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
            return Ok(());
        }

        let audio = AudioSystem::new();
        let host_name = audio.default_host_name();
        let device_name = audio.default_device_name(&host_name)
            .context("no default audio output device")?;
        let mut player = AudioPlayer::new(&host_name, &device_name, sample_rate)
            .context("failed to create audio player")?;
        player
            .play_blocking(samples)
            .context("audio playback failed")?;

        Ok(())
    }

    fn synthesize_batch(
        &mut self,
        text: &str,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        let sample_rate = self.engine.sample_rate();
        let request = SynthesisRequest {
            text: text.to_string(),
            ..Default::default()
        };

        let started = Instant::now();
        let result = self.engine.generate_cancellable(
            request,
            |current, max| {
                let bar_width = 20usize;
                let filled = bar_width
                    .saturating_mul(current)
                    .checked_div(max.max(1))
                    .unwrap_or(0);
                let bar: String = std::iter::repeat_n('=', filled)
                    .chain(std::iter::repeat_n('>', if current < max { 1 } else { 0 }))
                    .chain(std::iter::repeat_n(' ', bar_width.saturating_sub(filled).saturating_sub(1)))
                    .collect();
                eprint!(
                    "\r  Synthesizing... [{bar}] {current}/{max} patches",
                );
            },
            cancel,
        );

        let samples = match result {
            Ok(samples) => samples,
            Err(e) => {
                if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                    eprintln!();
                    return Ok(());
                }
                eprintln!();
                return Err(e).context("synthesis failed");
            }
        };

        let elapsed = started.elapsed().as_secs_f64();
        eprintln!();
        eprintln!("  Synthesis: {:.2}s", elapsed);

        if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
            return Ok(());
        }

        let audio = AudioSystem::new();
        let host_name = audio.default_host_name();
        let device_name = audio.default_device_name(&host_name)
            .context("no default audio output device")?;
        let mut player = AudioPlayer::new(&host_name, &device_name, sample_rate)
            .context("failed to create audio player")?;
        player
            .play_blocking(samples)
            .context("audio playback failed")?;

        Ok(())
    }

    /// Display a one-line summary of the loaded configuration.
    pub fn display_info(&self) {
        let arch = self.engine.architecture();
        let device = self.device_kind;
        let lora = self
            .lora_path
            .as_ref()
            .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "unknown".to_string()))
            .unwrap_or_else(|| "none".to_string());
        println!("Model: {arch}  |  Device: {device}  |  LoRA: {lora}");
    }
}
