use anyhow::Result;
use voxui_audio::AudioSystem;

use crate::types::{AudioDeviceDto, AudioHostDto};

pub fn list_hosts(system: &AudioSystem) -> Vec<AudioHostDto> {
    system
        .hosts()
        .iter()
        .map(|host| AudioHostDto {
            name: host.name.clone(),
        })
        .collect()
}

pub fn list_devices(system: &AudioSystem, host_name: &str) -> Result<Vec<AudioDeviceDto>> {
    let devices = system
        .devices(host_name)?
        .into_iter()
        .map(|device| AudioDeviceDto {
            name: device.name,
            host_name: device.host_name,
        })
        .collect();

    Ok(devices)
}

pub fn apply_volume(samples: &[f32], volume: f32) -> Vec<f32> {
    let volume = volume.clamp(0.0, 1.0);

    samples.iter().map(|sample| sample * volume).collect()
}

pub fn sine_with_fades(
    sample_rate: u32,
    len_samples: usize,
    frequency_hz: f32,
    volume: f32,
) -> Vec<f32> {
    if len_samples == 0 {
        return Vec::new();
    }
    if sample_rate == 0 {
        return vec![0.0; len_samples];
    }

    let volume = volume.clamp(0.0, 1.0);
    let fade_samples = ((sample_rate / 100) as usize).max(1).min(len_samples / 2);
    let two_pi = std::f32::consts::PI * 2.0;

    (0..len_samples)
        .map(|index| {
            let seconds = index as f32 / sample_rate as f32;
            let fade_in = if fade_samples > 0 && index < fade_samples {
                index as f32 / fade_samples as f32
            } else {
                1.0
            };
            let remaining = len_samples - 1 - index;
            let fade_out = if fade_samples > 0 && remaining < fade_samples {
                remaining as f32 / fade_samples as f32
            } else {
                1.0
            };
            let fade = fade_in.min(fade_out);

            (two_pi * frequency_hz * seconds).sin() * volume * fade
        })
        .collect()
}
