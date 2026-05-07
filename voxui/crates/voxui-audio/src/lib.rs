//! Audio capture and playback for VoxUI.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use r8brain_rs::{PrecisionProfile, Resampler};
use std::sync::{mpsc, Arc, Mutex};

#[derive(Debug, Clone)]
pub struct HostInfo {
    pub name: String,
    pub id: cpal::HostId,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub host_name: String,
}

pub struct AudioSystem {
    hosts: Vec<HostInfo>,
}

impl AudioSystem {
    pub fn new() -> Self {
        let hosts = cpal::available_hosts()
            .into_iter()
            .map(|id| HostInfo {
                name: format!("{:?}", id),
                id,
            })
            .collect();
        Self { hosts }
    }

    pub fn hosts(&self) -> &[HostInfo] {
        &self.hosts
    }

    pub fn devices(&self, host_name: &str) -> Result<Vec<DeviceInfo>> {
        let info = self
            .hosts
            .iter()
            .find(|h| h.name == host_name)
            .ok_or_else(|| anyhow!("unknown host: {host_name}"))?;
        let host = cpal::host_from_id(info.id)?;
        let devices = host
            .output_devices()?
            .filter_map(|d| {
                d.name().ok().map(|name| DeviceInfo {
                    name,
                    host_name: host_name.to_string(),
                })
            })
            .collect();
        Ok(devices)
    }

    pub fn default_host_name(&self) -> String {
        let id = cpal::default_host().id();
        format!("{:?}", id)
    }

    pub fn default_device_name(&self, host_name: &str) -> Result<String> {
        let info = self
            .hosts
            .iter()
            .find(|h| h.name == host_name)
            .ok_or_else(|| anyhow!("unknown host: {host_name}"))?;
        let host = cpal::host_from_id(info.id)?;
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device for host {host_name}"))?;
        Ok(device.name()?)
    }
}

pub struct AudioPlayer {
    device: cpal::Device,
    sample_rate: u32,
    stop_flag: Arc<Mutex<bool>>,
    stream: Option<cpal::Stream>,
}

impl AudioPlayer {
    pub fn new(host_name: &str, device_name: &str, sample_rate: u32) -> Result<Self> {
        let host_id = cpal::available_hosts()
            .into_iter()
            .find(|id| format!("{:?}", id) == host_name)
            .ok_or_else(|| anyhow!("unknown host: {host_name}"))?;
        let host = cpal::host_from_id(host_id)?;
        let device = host
            .output_devices()?
            .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
            .ok_or_else(|| anyhow!("device not found: {device_name}"))?;
        Ok(Self {
            device,
            sample_rate,
            stop_flag: Arc::new(Mutex::new(false)),
            stream: None,
        })
    }

    pub fn play(&mut self, samples: Vec<f32>) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel();
        let stop = self.stop_flag.clone();

        // Reset stop flag
        *stop.lock().unwrap() = false;

        // Query device native sample rate
        let default_config = self.device.default_output_config()?;
        let channels = default_config.channels();
        let device_rate = default_config.sample_rate().0;

        // Resample if device rate differs from requested
        let (playback_samples, playback_rate) = if device_rate != self.sample_rate {
            let resampled = resample(&samples, self.sample_rate, device_rate)?;
            (resampled, device_rate)
        } else {
            (samples, self.sample_rate)
        };

        let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(playback_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let buffer = Arc::new(Mutex::new(playback_samples));
        let pos = Arc::new(Mutex::new(0usize));
        let buf = buffer.clone();
        let p = pos.clone();
        let s = stop.clone();
        let done_sent = Arc::new(Mutex::new(false));
        let done_sent2 = done_sent.clone();

        let stream = self.device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let samples = buf.lock().unwrap();
                let mut current = p.lock().unwrap();
                let stopped = *s.lock().unwrap();
                let ch = channels as usize;

                // Write mono samples to all channels
                for frame in data.chunks_mut(ch) {
                    let val = if stopped || *current >= samples.len() {
                        0.0
                    } else {
                        let v = samples[*current];
                        *current += 1;
                        v
                    };
                    for sample in frame.iter_mut() {
                        *sample = val;
                    }
                }

                if (stopped || *current >= samples.len()) && !*done_sent2.lock().unwrap() {
                    *done_sent2.lock().unwrap() = true;
                    let _ = tx.send(());
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )?;

        stream.play()?;

        // Store stream so it lives as long as the player (or until next play/stop)
        self.stream = Some(stream);

        Ok(rx)
    }

    pub fn play_blocking(&mut self, samples: Vec<f32>) -> Result<()> {
        let rx = self.play(samples)?;
        rx.recv()
            .map_err(|_| anyhow!("playback channel closed unexpectedly"))?;
        Ok(())
    }

    pub fn stop(&mut self) {
        *self.stop_flag.lock().unwrap() = true;
        self.stream = None; // dropping Stream stops playback
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Resample audio from `src_rate` to `dst_rate` using r8brain.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(samples.to_vec());
    }
    let max_input_len = samples.len().min(8192);
    let mut resampler = Resampler::new(
        src_rate as f64,
        dst_rate as f64,
        max_input_len,
        2.0,
        PrecisionProfile::Bits24,
    );
    // Estimate output length
    let ratio = dst_rate as f64 / src_rate as f64;
    let mut output = Vec::with_capacity((samples.len() as f64 * ratio * 1.1) as usize);
    let mut buf = vec![0.0f64; max_input_len * 4];

    for chunk in samples.chunks(max_input_len) {
        let input_f64: Vec<f64> = chunk.iter().map(|&s| s as f64).collect();
        let n = resampler.process(&input_f64, &mut buf);
        for &s in &buf[..n] {
            output.push(s as f32);
        }
    }
    Ok(output)
}
