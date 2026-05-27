use voxui_desktop::inference_sidecar::{is_active_generation_event, sidecar_samples_from_payload};
use voxui_sidecar_protocol::{f32_samples_to_le_bytes, SidecarEvent};

#[test]
fn stale_generation_event_is_rejected() {
    let event = SidecarEvent::GenerationProgress {
        item_id: "old".to_string(),
        current: 1,
        total: 2,
    };

    assert!(!is_active_generation_event(Some("new"), &event));
    assert!(is_active_generation_event(Some("old"), &event));
}

#[test]
fn audio_payload_decodes_pcm_samples() {
    let samples = vec![0.0, 0.25, -0.5];
    let payload = f32_samples_to_le_bytes(&samples);

    assert_eq!(sidecar_samples_from_payload(&payload).unwrap(), samples);
}
