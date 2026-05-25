//! Audio capture and playback for VoxUI.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use r8brain_rs::{PrecisionProfile, Resampler};
use ringbuf::{traits::{Consumer, Observer, Producer, Split}, HeapRb};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

const PLAYBACK_CLEAR_AND_DRAIN_MS: usize = 50;

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
    stop_flag: Option<Arc<AtomicBool>>,
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
            stop_flag: None,
            stream: None,
        })
    }

    pub fn play(&mut self, samples: Vec<f32>) -> Result<mpsc::Receiver<()>> {
        self.stop();

        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));

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
        let playback_samples = prepare_playback_samples(playback_samples, playback_rate);

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
                let stopped = s.load(Ordering::SeqCst);
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
        self.stop_flag = Some(stop);
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
        if let Some(stop_flag) = self.stop_flag.take() {
            stop_flag.store(true, Ordering::SeqCst);
        }
        self.stream = None; // dropping Stream stops playback
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Streaming audio player that resamples and pushes audio chunks to the default
/// output device in real time via a lock-free ring buffer.
#[allow(dead_code)]
pub struct StreamingPlayer {
    stream: cpal::Stream,
    producer: ringbuf::HeapProd<f32>,
    resampler: Resampler,
    resample_ratio: f64,
}

impl StreamingPlayer {
    /// Opens the default output device, creates a resampler from `source_sample_rate`
    /// to the device native rate, and allocates a ring buffer sized for `pre_buffer_secs`
    /// seconds of audio at the source rate.
    pub fn new(source_sample_rate: u32, pre_buffer_secs: f32) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))?;
        let default_config = device.default_output_config()?;
        let channels = default_config.channels();
        if channels == 0 {
            return Err(anyhow!("output device reports 0 channels"));
        }
        let device_rate = default_config.sample_rate().0;

        let buffer_capacity =
            (source_sample_rate as f32 * pre_buffer_secs).ceil() as usize;
        let ring = HeapRb::<f32>::new(buffer_capacity);
        let (producer, mut consumer) = ring.split();

        let resampler = Resampler::new(
            source_sample_rate as f64,
            device_rate as f64,
            8192, // max input samples per process() call
            2.0,
            PrecisionProfile::Bits24,
        );

        let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(device_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let ch = channels as usize;
                let available = consumer.occupied_len();
                let want = data.len() / ch;
                let to_read = want.min(available);
                let mut buf = [0.0f32; 4096];
                let actual = consumer.pop_slice(&mut buf[..to_read]);
                let mut sample_iter = buf[..actual].iter().copied();
                for frame in data.chunks_mut(ch) {
                    let val = sample_iter.next().unwrap_or(0.0);
                    for sample in frame.iter_mut() {
                        *sample = val;
                    }
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )?;

        stream.play()?;

        Ok(Self {
            stream,
            producer,
            resampler,
            resample_ratio: device_rate as f64 / source_sample_rate as f64,
        })
    }

    /// Resample `samples` from the source rate to device rate and push into the
    /// ring buffer. Blocks only if the ring buffer is full (backpressure).
    pub fn push(&mut self, samples: &[f32]) {
        let input_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        let max_out = (input_f64.len() as f64 * self.resample_ratio * 1.1).ceil() as usize;
        let mut buf = vec![0.0f64; max_out];
        let n = self.resampler.process(&input_f64, &mut buf);
        for &s in &buf[..n] {
            // Spin-wait if ring buffer is full (backpressure)
            while self.producer.try_push(s as f32).is_err() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Block until all enqueued audio has been consumed by the output device,
    /// or until the optional cancel flag is set. Returns true if cancelled.
    pub fn flush_until(&self, cancel: Option<&std::sync::atomic::AtomicBool>) -> bool {
        while self.producer.occupied_len() > 0 {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst)) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    /// Block until all enqueued audio has been consumed by the output device.
    pub fn flush(&self) {
        self.flush_until(None);
    }
}

fn prepare_playback_samples(samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return samples;
    }

    let silence_len = ((sample_rate as usize * PLAYBACK_CLEAR_AND_DRAIN_MS) / 1_000).max(1);
    let mut prepared = Vec::with_capacity(samples.len() + silence_len * 2);
    prepared.resize(silence_len, 0.0);
    prepared.extend(samples);
    prepared.resize(prepared.len() + silence_len, 0.0);
    prepared
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

#[cfg(test)]
mod tests {
    #[test]
    fn playback_buffer_starts_and_ends_with_silence() {
        let samples = vec![0.25, -0.5, 0.75];
        let prepared = super::prepare_playback_samples(samples.clone(), 1_000);

        assert!(prepared.len() > samples.len());
        assert_eq!(&prepared[..50], vec![0.0; 50].as_slice());
        assert_eq!(
            &prepared[50..50 + samples.len()],
            samples.as_slice()
        );
        assert_eq!(&prepared[50 + samples.len()..], vec![0.0; 50].as_slice());
    }

    #[test]
    fn playback_buffer_uses_at_least_one_silence_sample_for_low_rates() {
        let prepared = super::prepare_playback_samples(vec![1.0], 1);

        assert_eq!(prepared, vec![0.0, 1.0, 0.0]);
    }
}
