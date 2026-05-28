use voxui_audio::{apply_loudness_volume, volume_to_gain};
use voxui_desktop::audio::sine_with_fades;
use voxui_desktop::playback::{GeneratedAudio, GeneratedAudioCache};

#[test]
fn volume_uses_loudness_curve_and_clamps_volume() {
    let samples = [-0.5, 0.25, 1.0];

    assert_eq!(apply_loudness_volume(&samples, 2.0), samples);
    assert_eq!(apply_loudness_volume(&samples, -1.0), vec![0.0, 0.0, 0.0]);
    assert!(volume_to_gain(0.5) < 0.5);
}

#[test]
fn sine_wave_has_faded_edges() {
    let samples = sine_with_fades(1_000, 1_000, 5.0, 0.8);

    assert_eq!(samples.len(), 1_000);
    assert_eq!(samples[0], 0.0);
    assert!(samples[50].abs() > 0.75);
    assert!(samples[999].abs() < 0.05);
    assert!(samples.iter().all(|sample| sample.abs() <= 0.8));
}

#[test]
fn generated_audio_cache_preserves_previous_until_replaced() {
    let mut cache = GeneratedAudioCache::default();
    let first = GeneratedAudio {
        samples: vec![0.1, 0.2],
        sample_rate: 24_000,
    };
    let second = GeneratedAudio {
        samples: vec![0.3, 0.4],
        sample_rate: 48_000,
    };
    let replacement = GeneratedAudio {
        samples: vec![0.5],
        sample_rate: 16_000,
    };

    cache.insert("first".to_string(), first.clone());
    cache.insert("second".to_string(), second.clone());

    assert_eq!(cache.get("first"), Some(&first));
    assert_eq!(cache.get("second"), Some(&second));

    cache.insert("first".to_string(), replacement.clone());

    assert_eq!(cache.get("first"), Some(&replacement));
    assert_eq!(cache.get("second"), Some(&second));
    assert_eq!(cache.remove("first"), Some(replacement));
    assert!(cache.get("first").is_none());
}

#[test]
fn zero_length_sine_output_is_empty() {
    assert!(sine_with_fades(24_000, 0, 440.0, 0.8).is_empty());
}

#[test]
fn zero_sample_rate_sine_output_is_silence() {
    let samples = sine_with_fades(0, 4, 440.0, 0.5);

    assert_eq!(samples, vec![0.0, 0.0, 0.0, 0.0]);
    assert!(samples.iter().all(|sample| sample.is_finite()));
}

#[test]
fn very_short_sine_outputs_are_finite() {
    for len_samples in [1, 2, 3] {
        let samples = sine_with_fades(24_000, len_samples, 440.0, 0.5);

        assert_eq!(samples.len(), len_samples);
        assert!(samples.iter().all(|sample| sample.is_finite()));
    }
}
