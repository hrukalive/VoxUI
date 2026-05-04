//! Audio loading and resampling helpers for native VoxCPM inference.

use std::path::Path;

use anyhow::{bail, Result};

pub struct LoadedAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn load_wav_mono_resampled(path: &Path, target_rate: u32) -> Result<LoadedAudio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut interleaved = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample?);
            }
        }
        hound::SampleFormat::Int => {
            let denom = (1_i64 << spec.bits_per_sample.saturating_sub(1)) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample? as f32 / denom);
            }
        }
    }

    if interleaved.is_empty() {
        bail!("empty wav {}", path.display());
    }

    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks(channels) {
        mono.push(frame.iter().copied().sum::<f32>() / frame.len() as f32);
    }

    Ok(LoadedAudio {
        sample_rate: target_rate,
        samples: resample_linear(&mono, spec.sample_rate, target_rate),
    })
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.len() < 2 {
        return input.to_vec();
    }
    let out_len = ((input.len() as u64 * dst_rate as u64) / src_rate as u64).max(1) as usize;
    let scale = src_rate as f64 / dst_rate as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * scale;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let frac = (pos - lo as f64) as f32;
        out.push(input[lo] * (1.0 - frac) + input[hi] * frac);
    }
    out
}
