use std::fs;

use tempfile::TempDir;
use voxui_desktop::app_core::{load_button_enabled, AppCore};
use voxui_desktop::generation_queue::HistoryStatus;
use voxui_desktop::types::{AppConfig, ConfigPatch, LoadUiState};

#[test]
fn load_button_requires_selection_and_difference_from_loaded() {
    assert!(!load_button_enabled(None, None, LoadUiState::Idle, false));
    assert!(load_button_enabled(
        Some("a"),
        None,
        LoadUiState::Idle,
        false
    ));
    assert!(!load_button_enabled(
        Some("a"),
        Some("a"),
        LoadUiState::Idle,
        false
    ));
    assert!(!load_button_enabled(
        Some("b"),
        Some("a"),
        LoadUiState::Loading,
        false
    ));
    assert!(!load_button_enabled(
        Some("b"),
        Some("a"),
        LoadUiState::Idle,
        true
    ));
}

#[test]
fn startup_discovers_models_and_restores_selection() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();
    let config = AppConfig {
        model_root: Some(temp.path().to_path_buf()),
        selected_model_id: Some("voxcpm2-fp16".to_string()),
        ..AppConfig::default()
    };

    let core = AppCore::from_config(config).unwrap();
    let snapshot = core.snapshot();

    assert_eq!(snapshot.models.len(), 1);
    assert_eq!(snapshot.models[0].id, "voxcpm2-fp16");
    assert_eq!(snapshot.selected_model_id.as_deref(), Some("voxcpm2-fp16"));
    assert_eq!(snapshot.loaded_model_id, None);
}

#[test]
fn startup_replaces_stale_saved_selection_with_first_model_in_config_snapshot() {
    let temp = TempDir::new().unwrap();
    let z_model = temp.path().join("z-model");
    let a_model = temp.path().join("a-model");
    fs::create_dir(&z_model).unwrap();
    fs::create_dir(&a_model).unwrap();
    fs::write(z_model.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(a_model.join("model.gguf"), [1u8; 4]).unwrap();
    let config = AppConfig {
        model_root: Some(temp.path().to_path_buf()),
        selected_model_id: Some("missing-model".to_string()),
        ..AppConfig::default()
    };

    let core = AppCore::from_config(config).unwrap();
    let snapshot = core.snapshot();

    assert_eq!(
        snapshot
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a-model", "z-model"]
    );
    assert_eq!(snapshot.selected_model_id.as_deref(), Some("a-model"));
    assert_eq!(snapshot.config.selected_model_id, snapshot.selected_model_id);
}

#[test]
fn enqueue_generation_rejects_when_no_model_is_loaded() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let error = core.enqueue_generation("hello".to_string()).unwrap_err();

    assert!(error.to_string().contains("no model loaded"));
}

#[test]
fn enqueue_generation_creates_queued_item_when_loaded() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("loaded-model-a".to_string());

    let item = core.enqueue_generation("hello".to_string()).unwrap();
    let snapshot = core.snapshot();

    assert_eq!(item.text, "hello");
    assert_eq!(item.status, HistoryStatus::Queued);
    assert_eq!(item.snapshot.model_id, "loaded-model-a");
    assert_eq!(snapshot.history.len(), 1);
    assert_eq!(snapshot.history[0], item);
}

#[test]
fn failed_load_preserves_previous_loaded_model_id() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("old-model".to_string());

    core.finish_model_load_for_test(Err("load failed".to_string()));

    assert_eq!(core.snapshot().loaded_model_id.as_deref(), Some("old-model"));
}

#[test]
fn starting_load_while_active_is_rejected_and_keeps_active_load() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let (active_load_id, active_cancel) = core.begin_model_load_for_test();
    let error = core.mark_load_started().unwrap_err();

    assert!(error.to_string().contains("model load already in progress"));
    assert_eq!(
        core.complete_model_load_success_for_test(active_load_id, "first-model".to_string()),
        true
    );
    assert!(!active_cancel.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(
        core.snapshot().loaded_model_id.as_deref(),
        Some("first-model")
    );
}

#[test]
fn canceling_active_load_prevents_later_stale_success_from_replacing_loaded_model() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("old-model".to_string());

    let (load_id, cancel) = core.begin_model_load_for_test();
    core.cancel_model_load_state();
    let swapped = core.complete_model_load_success_for_test(load_id, "new-model".to_string());

    assert!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!swapped);
    assert_eq!(core.snapshot().load_state, LoadUiState::Idle);
    assert_eq!(core.snapshot().loaded_model_id.as_deref(), Some("old-model"));
}

#[test]
fn stale_completion_does_not_overwrite_newer_completed_load() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let (stale_load_id, _) = core.begin_model_load_for_test();
    core.cancel_model_load_state();
    let (current_load_id, _) = core.begin_model_load_for_test();

    assert!(core.complete_model_load_success_for_test(
        current_load_id,
        "current-model".to_string()
    ));
    assert!(!core.complete_model_load_success_for_test(
        stale_load_id,
        "stale-model".to_string()
    ));
    assert_eq!(
        core.snapshot().loaded_model_id.as_deref(),
        Some("current-model")
    );
}

#[test]
fn selection_change_cancels_active_load_and_rejects_stale_completion() {
    let temp = TempDir::new().unwrap();
    let a_model = temp.path().join("a-model");
    let b_model = temp.path().join("b-model");
    fs::create_dir(&a_model).unwrap();
    fs::create_dir(&b_model).unwrap();
    fs::write(a_model.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(b_model.join("model.gguf"), [1u8; 4]).unwrap();

    let mut core = AppCore::from_config(AppConfig {
        model_root: Some(temp.path().to_path_buf()),
        selected_model_id: Some("a-model".to_string()),
        ..AppConfig::default()
    })
    .unwrap();
    core.set_loaded_model_for_test("old-model".to_string());

    let (load_id, cancel) = core.begin_model_load_for_test();
    core.apply_patch(ConfigPatch {
        model_root: None,
        selected_model_id: Some(Some("b-model".to_string())),
        language: None,
        backend: None,
        audio_host: None,
        audio_device: None,
        volume: None,
        max_input_chars: None,
        generation: None,
    })
    .unwrap();
    let swapped = core.complete_model_load_success_for_test(load_id, "a-model".to_string());

    assert!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!swapped);
    assert_eq!(core.snapshot().load_state, LoadUiState::Idle);
    assert_eq!(core.snapshot().selected_model_id.as_deref(), Some("b-model"));
    assert_eq!(core.snapshot().loaded_model_id.as_deref(), Some("old-model"));
}
