//! Audio capture and playback for VoxUI.

use std::sync::{Arc, Mutex, mpsc};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use anyhow::{Result, anyhow};

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
        let buffer = Arc::new(Mutex::new(samples));
        let pos = Arc::new(Mutex::new(0usize));
        let stop = self.stop_flag.clone();

        // Reset stop flag
        *stop.lock().unwrap() = false;

        // Use default device config, override sample rate if supported
        let default_config = self.device.default_output_config()?;
        let channels = default_config.channels();
        let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

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
        rx.recv().map_err(|_| anyhow!("playback channel closed unexpectedly"))?;
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
