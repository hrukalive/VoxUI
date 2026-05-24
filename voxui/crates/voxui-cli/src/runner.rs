use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use candle_core::Device;
use voxui_audio::StreamingPlayer;
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

    /// Synthesize `text` and stream audio to the default output device.
    /// Checks `cancel` after each chunk; returns early if set.
    pub fn synthesize_and_play(
        &mut self,
        text: &str,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        let sample_rate = self.engine.sample_rate();
        let mut player = StreamingPlayer::new(sample_rate, 1.0)
            .context("failed to create audio player")?;

        let request = SynthesisRequest {
            text: text.to_string(),
            ..Default::default()
        };

        let mut patch_count = 0usize;
        let mut max_patches = 0usize;

        let result = self.engine.generate_streaming_cancellable(
            request,
            |chunk| {
                patch_count = chunk.generated_patch_count;
                max_patches = chunk.max_patches;
                // Simple progress bar: 20 chars wide
                let bar_width = 20usize;
                let filled = bar_width
                    .saturating_mul(patch_count)
                    .checked_div(max_patches.max(1))
                    .unwrap_or(0);
                let bar: String = std::iter::repeat('=').take(filled)
                    .chain(std::iter::repeat('>').take(if patch_count < max_patches { 1 } else { 0 }))
                    .chain(std::iter::repeat(' ').take(bar_width.saturating_sub(filled).saturating_sub(1)))
                    .collect();
                eprint!(
                    "\r  Synthesizing... [{bar}] {patch_count}/{max_patches} patches",
                );
                player.push(&chunk.samples);
                Ok(())
            },
            cancel,
        );

        if let Err(e) = result {
            // Check if it was a cancellation
            if cancel.map(|c| c.load(Ordering::SeqCst)).unwrap_or(false) {
                return Ok(());
            }
            return Err(e).context("synthesis failed");
        }

        player.flush();

        // Let cancellation show as cancelled, not done
        if cancel.map(|c| c.load(Ordering::SeqCst)).unwrap_or(false) {
            return Ok(());
        }

        Ok(())
    }

    /// Display a one-line summary of the loaded configuration.
    pub fn display_info(&self) {
        let arch = self.engine.architecture();
        let device = self.device_kind;
        let lora = self
            .lora_path
            .as_ref()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .unwrap_or_else(|| "none".to_string());
        println!("Model: {arch}  |  Device: {device}  |  LoRA: {lora}");
    }
}
