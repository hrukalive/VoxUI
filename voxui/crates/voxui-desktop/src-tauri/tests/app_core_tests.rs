use std::fs;

use tempfile::TempDir;
use voxui_desktop::app_core::{load_button_enabled, AppCore};
use voxui_desktop::generation_queue::HistoryStatus;
use voxui_desktop::types::{AppConfig, LoadUiState};

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
