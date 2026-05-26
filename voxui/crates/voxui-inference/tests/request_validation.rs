use std::path::PathBuf;

use voxui_inference::{ModelVariant, SynthesisChunk, SynthesisRequest};

#[test]
fn synthesis_chunk_is_public_api() {
    let chunk = SynthesisChunk {
        samples: vec![0.0, 0.25],
        sample_rate: 48_000,
        patch_index: 0,
        max_patches: 8,
        generated_patch_count: 1,
        is_final: false,
    };

    assert_eq!(chunk.samples.len(), 2);
    assert_eq!(chunk.sample_rate, 48_000);
    assert!(!chunk.is_final);
}

#[test]
fn request_rejects_empty_text_after_whitespace_normalization() {
    let err = SynthesisRequest {
        text: " \n\t ".to_string(),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("text must not be empty"));
}

#[test]
fn request_requires_prompt_text_when_prompt_audio_is_present() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        prompt_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        prompt_text: None,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("prompt_text"));
}

#[test]
fn request_allows_reference_audio_without_text_on_voxcpm2() {
    let req = SynthesisRequest {
        text: "hello".to_string(),
        reference_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap();
    assert_eq!(req.prompt_text, None);
}

#[test]
fn request_rejects_reference_audio_on_non_v2() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        reference_wav_path: Some(PathBuf::from("for_test_wav/example.wav")),
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm15)
    .unwrap_err();
    assert!(err.to_string().contains("Reference audio requires VoxCPM2"));
}

#[test]
fn request_rejects_normalize_until_rust_normalizer_is_implemented() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        normalize: true,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();
    assert!(err.to_string().contains("normalize=true"));
}

#[test]
fn request_default_consolidate_n_is_one() {
    assert_eq!(SynthesisRequest::default().consolidate_n, 1);
}

#[test]
fn request_rejects_zero_consolidate_n() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        consolidate_n: 0,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();

    assert!(err.to_string().contains("consolidate_n"));
}
