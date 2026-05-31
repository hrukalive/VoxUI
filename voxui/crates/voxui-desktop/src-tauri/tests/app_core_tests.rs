use std::fs;
use std::path::PathBuf;

use tempfile::TempDir;
use voxui_desktop::app_core::{load_button_enabled, AppCore};
use voxui_desktop::generation_queue::HistoryStatus;
use voxui_desktop::types::{
    AppConfig, BackendKind, ConfigPatch, GenerationSettings, LanguageMode, LoadUiState,
    SidecarCapabilities, ThemeMode,
};

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
fn startup_replaces_stale_saved_selection_with_first_discovered_model() {
    let temp = TempDir::new().unwrap();
    let a_model = temp.path().join("a-model");
    fs::create_dir(&a_model).unwrap();
    fs::write(a_model.join("model.gguf"), [1u8; 4]).unwrap();
    let config = AppConfig {
        model_root: Some(temp.path().to_path_buf()),
        selected_model_id: Some("missing-model".to_string()),
        ..AppConfig::default()
    };

    let core = AppCore::from_config(config).unwrap();
    let snapshot = core.snapshot();

    assert_eq!(snapshot.models.len(), 1);
    assert_eq!(snapshot.selected_model_id.as_deref(), Some("a-model"));
    assert_eq!(
        snapshot.config.selected_model_id,
        snapshot.selected_model_id
    );
}

#[test]
fn changing_audio_host_clears_saved_audio_device() {
    let mut core = AppCore::from_config(AppConfig {
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        ..AppConfig::default()
    })
    .unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: None,
            audio_host: Some(Some("Asio".to_string())),
            audio_device: None,
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.audio_host.as_deref(), Some("Asio"));
    assert_eq!(snapshot.config.audio_device, None);
}

#[test]
fn sidecar_capabilities_choose_cuda_for_missing_saved_backend() {
    let mut core = AppCore::from_loaded_config(AppConfig::default(), false).unwrap();

    core.apply_sidecar_capabilities(SidecarCapabilities {
        cuda_available: true,
        default_backend: BackendKind::Cuda,
    });

    let snapshot = core.snapshot();
    assert!(snapshot.cuda_available);
    assert_eq!(snapshot.config.backend, BackendKind::Cuda);
}

#[test]
fn sidecar_capabilities_keep_saved_cpu_when_cuda_is_supported() {
    let mut core = AppCore::from_loaded_config(
        AppConfig {
            backend: BackendKind::Cpu,
            ..AppConfig::default()
        },
        true,
    )
    .unwrap();

    core.apply_sidecar_capabilities(SidecarCapabilities {
        cuda_available: true,
        default_backend: BackendKind::Cuda,
    });

    let snapshot = core.snapshot();
    assert!(snapshot.cuda_available);
    assert_eq!(snapshot.config.backend, BackendKind::Cpu);
}

#[test]
fn sidecar_capabilities_normalize_unsupported_saved_cuda_to_cpu() {
    let mut core = AppCore::from_loaded_config(
        AppConfig {
            backend: BackendKind::Cuda,
            ..AppConfig::default()
        },
        true,
    )
    .unwrap();

    core.apply_sidecar_capabilities(SidecarCapabilities {
        cuda_available: false,
        default_backend: BackendKind::Cpu,
    });

    let snapshot = core.snapshot();
    assert!(!snapshot.cuda_available);
    assert_eq!(snapshot.config.backend, BackendKind::Cpu);
}

#[test]
fn unsupported_cuda_backend_patch_is_ignored_until_sidecar_reports_cuda() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: Some(BackendKind::Cuda),
            audio_host: None,
            audio_device: None,
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.backend, BackendKind::Cpu);
    assert!(!snapshot.cuda_available);

    core.apply_sidecar_capabilities(SidecarCapabilities {
        cuda_available: true,
        default_backend: BackendKind::Cuda,
    });
    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: Some(BackendKind::Cuda),
            audio_host: None,
            audio_device: None,
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.backend, BackendKind::Cuda);
}

#[test]
fn clearing_audio_host_to_default_clears_host_and_device() {
    let mut core = AppCore::from_config(AppConfig {
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        ..AppConfig::default()
    })
    .unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: None,
            audio_host: Some(None),
            audio_device: Some(None),
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.audio_host, None);
    assert_eq!(snapshot.config.audio_device, None);
}

#[test]
fn empty_audio_host_patch_clears_host_and_device() {
    let mut core = AppCore::from_config(AppConfig {
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        ..AppConfig::default()
    })
    .unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: None,
            audio_host: Some(Some(String::new())),
            audio_device: Some(Some(String::new())),
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.audio_host, None);
    assert_eq!(snapshot.config.audio_device, None);
}

#[test]
fn clearing_audio_device_to_default_keeps_selected_host() {
    let mut core = AppCore::from_config(AppConfig {
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        ..AppConfig::default()
    })
    .unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: None,
            audio_host: None,
            audio_device: Some(None),
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.audio_host.as_deref(), Some("Wasapi"));
    assert_eq!(snapshot.config.audio_device, None);
}

#[test]
fn empty_audio_device_patch_clears_device_and_keeps_host() {
    let mut core = AppCore::from_config(AppConfig {
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        ..AppConfig::default()
    })
    .unwrap();

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: None,
            selected_model_id: None,
            selected_lora_id: None,
            language: None,
            theme: None,
            backend: None,
            audio_host: None,
            audio_device: Some(Some(String::new())),
            volume: None,
            max_input_chars: None,
            generation: None,
            translation: None,
        })
        .unwrap();

    assert_eq!(snapshot.config.audio_host.as_deref(), Some("Wasapi"));
    assert_eq!(snapshot.config.audio_device, None);
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
fn request_snapshot_converts_to_synthesis_request() {
    let mut core = AppCore::from_config(AppConfig {
        generation: GenerationSettings {
            cfg_value: 3.5,
            inference_timesteps: 22,
            min_len: 7,
            max_len: 77,
            streaming: false,
            stream_consolidate_n: 1,
            retry_badcase: false,
            retry_badcase_max_times: 9,
            retry_badcase_ratio_threshold: 4.25,
            prompt_wav_path: Some(PathBuf::from("prompt.wav")),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some(PathBuf::from("reference.wav")),
        },
        ..AppConfig::default()
    })
    .unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core
        .enqueue_generation(" hello world ".to_string())
        .unwrap();

    let request = core.synthesis_request_for_test(&item.id).unwrap();

    assert_eq!(request.text, "hello world");
    assert_eq!(request.prompt_wav_path, Some(PathBuf::from("prompt.wav")));
    assert_eq!(request.prompt_text.as_deref(), Some("prompt text"));
    assert_eq!(
        request.reference_wav_path,
        Some(PathBuf::from("reference.wav"))
    );
    assert_eq!(request.cfg_value, 3.5);
    assert_eq!(request.inference_timesteps, 22);
    assert_eq!(request.min_len, 7);
    assert_eq!(request.max_len, 77);
    assert!(!request.normalize);
    assert!(!request.retry_badcase);
    assert_eq!(request.retry_badcase_max_times, 9);
    assert_eq!(request.retry_badcase_ratio_threshold, 4.25);
}

#[test]
fn run_generation_now_does_not_revive_canceled_item() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();

    assert!(core.cancel_generation_item(&item.id));
    let error = core.run_generation_now(&item.id, |_, _| {}).unwrap_err();
    let snapshot = core.snapshot();

    assert!(error.contains("not queued"));
    assert_eq!(snapshot.history[0].status, HistoryStatus::Canceled);
    assert_eq!(snapshot.history[0].error, None);
    assert!(!snapshot.history[0].has_audio);
}

#[test]
fn playback_state_requires_cached_audio_and_can_stop() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();

    assert!(core.begin_playback(&item.id).is_err());

    core.set_generated_audio_for_test(item.id.clone(), vec![0.0; 8], 16_000);
    let _run = core.begin_playback(&item.id).unwrap();
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Playing);

    assert_eq!(core.stop_playback().as_deref(), Some(item.id.as_str()));
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Ready);
}

#[test]
fn regenerate_stops_playback_when_regenerating_playing_item() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let target = core.enqueue_generation("target".to_string()).unwrap();
    core.set_generated_audio_for_test(target.id.clone(), vec![0.0; 8], 16_000);

    let _run = core.begin_playback(&target.id).unwrap();
    let stopped = core
        .regenerate_item_stopping_playback(&target.id, &AppConfig::default())
        .unwrap();
    let snapshot = core.snapshot();

    assert_eq!(stopped.as_deref(), Some(target.id.as_str()));
    assert_eq!(snapshot.history[0].status, HistoryStatus::Queued);
}

#[test]
fn regenerate_keeps_other_item_playing() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let playing = core.enqueue_generation("playing".to_string()).unwrap();
    let target = core.enqueue_generation("target".to_string()).unwrap();
    core.set_generated_audio_for_test(playing.id.clone(), vec![0.0; 8], 16_000);
    core.set_generated_audio_for_test(target.id.clone(), vec![0.0; 8], 16_000);

    let _run = core.begin_playback(&playing.id).unwrap();
    let stopped = core
        .regenerate_item_stopping_playback(&target.id, &AppConfig::default())
        .unwrap();
    let snapshot = core.snapshot();

    assert_eq!(stopped, None);
    assert_eq!(snapshot.history[0].status, HistoryStatus::Playing);
    assert_eq!(snapshot.history[1].status, HistoryStatus::Queued);
}

#[test]
fn automatic_playback_waits_until_current_playback_finishes() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let playing = core.enqueue_generation("playing".to_string()).unwrap();
    let next = core.enqueue_generation("next".to_string()).unwrap();
    core.set_generated_audio_for_test(playing.id.clone(), vec![0.0; 8], 16_000);
    core.set_generated_audio_for_test(next.id.clone(), vec![1.0; 8], 16_000);

    let _run = core.begin_playback(&playing.id).unwrap();
    assert!(core
        .begin_or_queue_auto_playback(&next.id)
        .unwrap()
        .is_none());
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Playing);
    assert_eq!(core.snapshot().history[1].status, HistoryStatus::Ready);

    let finished = core.finish_playback_and_next(&playing.id);

    assert_eq!(
        finished.stopped_item_id.as_deref(),
        Some(playing.id.as_str())
    );
    assert_eq!(
        finished.next_run.as_ref().map(|run| run.item_id.as_str()),
        Some(next.id.as_str())
    );
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Ready);
    assert_eq!(core.snapshot().history[1].status, HistoryStatus::Playing);
}

#[test]
fn begin_generation_rejects_second_active_generation_without_consuming_queue() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let first = core.enqueue_generation("first".to_string()).unwrap();
    let second = core.enqueue_generation("second".to_string()).unwrap();

    core.begin_generation_for_test(&first.id).unwrap();
    let error = core.begin_generation_for_test(&second.id).unwrap_err();
    let snapshot = core.snapshot();

    assert!(error.contains("generation already in progress"));
    assert_eq!(snapshot.history[0].status, HistoryStatus::Generating);
    assert_eq!(snapshot.history[1].status, HistoryStatus::Queued);
    assert_eq!(snapshot.history[1].error, None);
}

#[test]
fn model_load_cannot_start_while_generation_is_active() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.begin_generation_for_test(&item.id).unwrap();

    let error = core.mark_load_started().unwrap_err();
    let snapshot = core.snapshot();

    assert!(error.to_string().contains("generation already in progress"));
    assert_eq!(snapshot.load_state, LoadUiState::Idle);
    assert_eq!(snapshot.history[0].status, HistoryStatus::Generating);
}

#[test]
fn generation_cannot_start_while_model_load_is_active() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    let (_load_id, _cancel) = core.begin_model_load_for_test();

    let error = match core.begin_generation_run(&item.id) {
        Ok(_) => panic!("generation should not start while model load is active"),
        Err(error) => error,
    };
    let snapshot = core.snapshot();

    assert!(error.contains("model load already in progress"));
    assert_eq!(snapshot.history[0].status, HistoryStatus::Queued);
    assert_eq!(snapshot.history[0].error, None);
}

#[test]
fn failed_load_preserves_previous_loaded_model_id() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("old-model".to_string());

    core.finish_model_load_for_test(Err("load failed".to_string()));

    assert_eq!(
        core.snapshot().loaded_model_id.as_deref(),
        Some("old-model")
    );
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
    assert_eq!(
        core.snapshot().loaded_model_id.as_deref(),
        Some("old-model")
    );
}

#[test]
fn stale_completion_does_not_overwrite_newer_completed_load() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let (stale_load_id, _) = core.begin_model_load_for_test();
    core.cancel_model_load_state();
    let (current_load_id, _) = core.begin_model_load_for_test();

    assert!(core.complete_model_load_success_for_test(current_load_id, "current-model".to_string()));
    assert!(!core.complete_model_load_success_for_test(stale_load_id, "stale-model".to_string()));
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
        auto_period: None,
        model_root: None,
        selected_model_id: Some(Some("b-model".to_string())),
        selected_lora_id: None,
        language: None,
        theme: None,
        backend: None,
        audio_host: None,
        audio_device: None,
        volume: None,
        max_input_chars: None,
        generation: None,
        translation: None,
    })
    .unwrap();
    let swapped = core.complete_model_load_success_for_test(load_id, "a-model".to_string());

    assert!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!swapped);
    assert_eq!(core.snapshot().load_state, LoadUiState::Idle);
    assert_eq!(
        core.snapshot().selected_model_id.as_deref(),
        Some("b-model")
    );
    assert_eq!(
        core.snapshot().loaded_model_id.as_deref(),
        Some("old-model")
    );
}

#[test]
fn applying_config_patch_persists_the_saved_config_to_disk() {
    let temp = TempDir::new().unwrap();
    let model_root = temp.path().join("models");
    fs::create_dir_all(model_root.join("alpha")).unwrap();
    fs::write(model_root.join("alpha").join("model.gguf"), [0u8; 4]).unwrap();

    let config_path = temp.path().join("nested").join("voxui_config.json");
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_config_path(config_path.clone());

    let snapshot = core
        .apply_patch(ConfigPatch {
            auto_period: None,
            model_root: Some(Some(model_root.clone())),
            selected_model_id: Some(Some("alpha".to_string())),
            selected_lora_id: None,
            language: Some(LanguageMode::Chinese),
            theme: Some(ThemeMode::Light),
            backend: Some(BackendKind::Cuda),
            audio_host: Some(Some("Wasapi".to_string())),
            audio_device: Some(Some("Speakers".to_string())),
            volume: Some(0.55),
            max_input_chars: Some(360),
            generation: Some(GenerationSettings {
                cfg_value: 3.5,
                inference_timesteps: 18,
                min_len: 4,
                max_len: 1800,
                streaming: false,
                stream_consolidate_n: 1,
                retry_badcase: false,
                retry_badcase_max_times: 2,
                retry_badcase_ratio_threshold: 4.5,
                prompt_wav_path: Some(PathBuf::from("prompt.wav")),
                prompt_text: Some("prompt".to_string()),
                reference_wav_path: Some(PathBuf::from("reference.wav")),
            }),
            translation: None,
        })
        .unwrap();

    let saved = voxui_desktop::config::load_config(&config_path).unwrap();

    assert_eq!(snapshot.config, saved);
    assert_eq!(saved.theme, ThemeMode::Light);
}

#[test]
fn queued_generation_advance_prefers_the_next_waiting_item_after_completion() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());

    let first = core.enqueue_generation("first".to_string()).unwrap();
    let second = core.enqueue_generation("second".to_string()).unwrap();

    let first_run = core.begin_next_generation_run().unwrap().unwrap();
    assert_eq!(first_run.item_id, first.id);
    core.finish_generation_success(first_run, vec![0.0; 16], 16_000.0);

    let second_run = core.begin_next_generation_run().unwrap().unwrap();
    assert_eq!(second_run.item_id, second.id);
}

#[test]
fn canceling_an_active_generation_marks_it_canceled() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());

    let item = core.enqueue_generation("hello".to_string()).unwrap();
    let run = core.begin_generation_run(&item.id).unwrap();

    assert!(core.cancel_generation_item(&item.id));
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Canceled);

    core.finish_generation_canceled(run);
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Canceled);
}

#[test]
fn volume_patch_updates_active_playback_handle() {
    let mut core = AppCore::from_config(AppConfig {
        volume: 0.25,
        ..AppConfig::default()
    })
    .unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("ready".to_string()).unwrap();
    core.set_generated_audio_for_test(item.id.clone(), vec![0.0; 8], 16_000);

    let playback = core.begin_playback(&item.id).unwrap();
    core.apply_patch(ConfigPatch {
        auto_period: None,
        model_root: None,
        selected_model_id: None,
        selected_lora_id: None,
        language: None,
        theme: None,
        backend: None,
        audio_host: None,
        audio_device: None,
        volume: Some(0.75),
        max_input_chars: None,
        generation: None,
        translation: None,
    })
    .unwrap();

    assert_eq!(playback.volume.get(), 0.75);
}

#[test]
fn runtime_volume_updates_active_playback_without_config_patch() {
    let mut core = AppCore::from_config(AppConfig {
        volume: 0.25,
        ..AppConfig::default()
    })
    .unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("ready".to_string()).unwrap();
    core.set_generated_audio_for_test(item.id.clone(), vec![0.0; 8], 16_000);

    let playback = core.begin_playback(&item.id).unwrap();
    let volume = core.set_runtime_volume(0.75);

    assert_eq!(volume, 0.75);
    assert_eq!(playback.volume.get(), 0.75);
    assert_eq!(core.snapshot().config.volume, 0.75);
}

#[test]
fn canceling_active_regeneration_keeps_previous_audio_playable() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());

    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.set_generated_audio_for_test(item.id.clone(), vec![0.0; 8], 16_000);
    core.regenerate_item(&item.id, &AppConfig::default())
        .unwrap();
    let run = core.begin_generation_run(&item.id).unwrap();

    assert!(core.cancel_generation_item(&item.id));
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Ready);
    assert!(core.has_audio(&item.id));
    assert!(core.begin_playback(&item.id).is_ok());

    core.finish_generation_canceled(run);
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Ready);
}

#[test]
fn canceled_active_generation_rejects_late_audio_chunks_but_accepts_completion() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.begin_generation_for_test(&item.id).unwrap();

    assert!(core.cancel_generation_item(&item.id));

    assert!(!core.accepts_sidecar_generation_event(&item.id, false));
    assert!(core.accepts_sidecar_generation_event(&item.id, true));
    assert!(core
        .append_generation_audio_chunk(&item.id, vec![0.1, 0.2], 16_000)
        .is_err());
    assert!(!core.has_audio(&item.id));
}

#[test]
fn final_audio_failure_restores_previous_audio_and_keeps_item_failed() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.set_generated_audio_for_test(item.id.clone(), vec![0.9; 2], 16_000);
    core.regenerate_item(&item.id, &AppConfig::default())
        .unwrap();
    let _run = core.begin_generation_run(&item.id).unwrap();

    assert!(!core.has_audio(&item.id));
    core.append_generation_audio_chunk(&item.id, vec![0.1, 0.2], 16_000)
        .unwrap();
    let error = core
        .finish_generation_failure_from_sidecar(&item.id, "sample rate changed".to_string())
        .unwrap();

    assert_eq!(error, "sample rate changed");
    assert_eq!(core.snapshot().history[0].status, HistoryStatus::Failed);
    assert!(core.has_audio(&item.id));
}

#[test]
fn sidecar_exit_clears_loaded_model_and_fails_active_generation_without_consuming_queue() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let active = core.enqueue_generation("active".to_string()).unwrap();
    let queued = core.enqueue_generation("queued".to_string()).unwrap();
    core.begin_generation_for_test(&active.id).unwrap();

    let recovery = core.handle_sidecar_exit("sidecar exited unexpectedly".to_string());
    let snapshot = core.snapshot();

    assert_eq!(snapshot.loaded_model_id, None);
    assert_eq!(snapshot.history[0].status, HistoryStatus::Failed);
    assert_eq!(
        snapshot.history[0].error.as_deref(),
        Some("sidecar exited unexpectedly")
    );
    assert_eq!(snapshot.history[1].id, queued.id);
    assert_eq!(snapshot.history[1].status, HistoryStatus::Queued);
    assert_eq!(
        recovery.failed_generation_item_id.as_deref(),
        Some(active.id.as_str())
    );
    assert_eq!(
        recovery.stopped_generation_item_id.as_deref(),
        Some(active.id.as_str())
    );
    assert!(core
        .begin_next_generation_run()
        .unwrap_err()
        .contains("no model loaded"));
}

#[test]
fn sidecar_exit_after_local_cancel_keeps_item_canceled_and_releases_active_generation() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.begin_generation_for_test(&item.id).unwrap();
    assert!(core.cancel_generation_item(&item.id));

    let recovery = core.handle_sidecar_exit("sidecar exited unexpectedly".to_string());
    let snapshot = core.snapshot();

    assert_eq!(snapshot.loaded_model_id, None);
    assert_eq!(snapshot.history[0].status, HistoryStatus::Canceled);
    assert_eq!(recovery.failed_generation_item_id, None);
    assert_eq!(
        recovery.stopped_generation_item_id.as_deref(),
        Some(item.id.as_str())
    );
    assert!(core
        .begin_next_generation_run()
        .unwrap_err()
        .contains("no model loaded"));
}

#[test]
fn sidecar_exit_clears_loaded_model_and_finishes_active_load() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("old-model".to_string());
    let (_load_id, _cancel) = core.begin_model_load_for_test();

    let recovery = core.handle_sidecar_exit("sidecar exited unexpectedly".to_string());
    let snapshot = core.snapshot();

    assert_eq!(snapshot.loaded_model_id, None);
    assert_eq!(snapshot.load_state, LoadUiState::Idle);
    assert!(recovery.failed_load);
}
