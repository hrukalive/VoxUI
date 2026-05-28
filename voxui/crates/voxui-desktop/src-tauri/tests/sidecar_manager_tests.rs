use std::sync::mpsc;

use voxui_desktop::inference_sidecar::{
    is_active_generation_event, read_sidecar_frames, sidecar_samples_from_payload,
    SidecarReaderEvent,
};
use voxui_sidecar_protocol::{f32_samples_to_le_bytes, write_frame, Frame, SidecarEvent};

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

#[test]
fn reader_emits_frames_and_clean_eof() {
    let frame = Frame {
        header: SidecarEvent::Ready,
        payload: Vec::new(),
    };
    let mut bytes = Vec::new();
    write_frame(&mut bytes, &frame).unwrap();
    let (sender, receiver) = mpsc::channel();

    read_sidecar_frames(bytes.as_slice(), sender);

    match receiver.recv().unwrap() {
        SidecarReaderEvent::Frame(received) => assert_eq!(received, frame),
        other => panic!("unexpected reader event: {other:?}"),
    }
    assert!(matches!(receiver.recv().unwrap(), SidecarReaderEvent::Eof));
}

#[test]
fn reader_reports_truncated_frame_as_error() {
    let (sender, receiver) = mpsc::channel();

    read_sidecar_frames([1_u8, 2, 3].as_slice(), sender);

    match receiver.recv().unwrap() {
        SidecarReaderEvent::Error(message) => assert!(message.contains("unexpected eof")),
        other => panic!("unexpected reader event: {other:?}"),
    }
}
