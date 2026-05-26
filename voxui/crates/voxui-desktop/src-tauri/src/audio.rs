use anyhow::Result;
use voxui_audio::AudioSystem;

use crate::types::{AudioDeviceDto, AudioHostDto, AudioStateDto};

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

pub fn audio_state(system: &AudioSystem) -> AudioStateDto {
    let hosts = list_hosts(system);
    let devices = hosts
        .iter()
        .flat_map(|host| list_devices(system, &host.name).unwrap_or_default())
        .collect();
    let default_host = hosts
        .iter()
        .any(|host| host.name == system.default_host_name())
        .then(|| system.default_host_name());

    AudioStateDto {
        hosts,
        devices,
        default_host,
    }
}

pub fn resolve_output_device_name(
    configured_device: Option<String>,
    available_devices: &[AudioDeviceDto],
    default_device: Result<String>,
) -> Result<String> {
    if let Some(device) = configured_device {
        if available_devices
            .iter()
            .any(|available| available.name == device)
        {
            return Ok(device);
        }
    }

    default_device
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioDeviceDto;

    #[test]
    fn resolve_output_device_falls_back_when_configured_device_is_not_available() {
        let devices = vec![AudioDeviceDto {
            name: "Default Speakers".to_string(),
            host_name: "Wasapi".to_string(),
        }];

        let resolved = resolve_output_device_name(
            Some("Missing Speakers".to_string()),
            &devices,
            Ok("Default Speakers".to_string()),
        )
        .unwrap();

        assert_eq!(resolved, "Default Speakers");
    }
}
