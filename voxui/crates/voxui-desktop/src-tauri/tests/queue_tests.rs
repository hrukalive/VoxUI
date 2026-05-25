use voxui_desktop::generation_queue::{GenerationQueue, HistoryStatus};
use voxui_desktop::types::{AppConfig, BackendKind};

fn configured_model(model_id: &str) -> AppConfig {
    AppConfig {
        selected_model_id: Some(model_id.to_string()),
        backend: BackendKind::Cuda,
        ..AppConfig::default()
    }
}

#[test]
fn enqueue_captures_settings_and_preserves_order() {
    let mut config = configured_model("selected-model-a");
    config.generation.cfg_value = 3.25;
    config.generation.inference_timesteps = 12;
    let mut queue = GenerationQueue::default();

    let first_id = queue.enqueue("first text".to_string(), "loaded-model-a", &config);
    config.selected_model_id = Some("selected-model-b".to_string());
    config.backend = BackendKind::Cpu;
    config.generation.cfg_value = 1.5;
    let second_id = queue.enqueue("second text".to_string(), "loaded-model-b", &config);

    let items = queue.items();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].id, first_id);
    assert_eq!(items[0].text, "first text");
    assert_eq!(items[0].status, HistoryStatus::Queued);
    assert_eq!(items[0].snapshot.model_id, "loaded-model-a");
    assert_eq!(items[0].snapshot.backend, BackendKind::Cuda);
    assert_eq!(items[0].snapshot.generation.cfg_value, 3.25);
    assert_eq!(items[0].snapshot.generation.inference_timesteps, 12);
    assert_eq!(items[1].id, second_id);
    assert_eq!(items[1].text, "second text");
    assert_eq!(items[1].snapshot.model_id, "loaded-model-b");
    assert_eq!(items[1].snapshot.backend, BackendKind::Cpu);
    assert_eq!(queue.next_queued_id(), Some(first_id.as_str()));
}

#[test]
fn cancel_queued_item_marks_it_canceled() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let first_id = queue.enqueue("first text".to_string(), "model-a", &config);
    let second_id = queue.enqueue("second text".to_string(), "model-a", &config);

    assert!(queue.cancel_queued(&first_id));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Canceled);
    assert_eq!(items[0].error, None);
    assert_eq!(queue.next_queued_id(), Some(second_id.as_str()));
}

#[test]
fn regeneration_attempt_keeps_existing_audio_flag_until_success() {
    let mut config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let id = queue.enqueue("text".to_string(), "loaded-model-a", &config);
    queue.mark_ready(&id);

    config.selected_model_id = Some("selected-but-not-loaded-model-b".to_string());
    config.backend = BackendKind::Cpu;
    config.generation.cfg_value = 4.0;

    assert!(queue.start_regeneration(&id, "loaded-model-a", &config));

    let item = &queue.items()[0];
    assert_eq!(item.status, HistoryStatus::Queued);
    assert_eq!(item.progress_current, 0);
    assert_eq!(item.progress_total, 0);
    assert_eq!(item.error, None);
    assert!(item.has_audio);
    assert_eq!(item.snapshot.model_id, "loaded-model-a");
    assert_eq!(item.snapshot.backend, BackendKind::Cpu);
    assert_eq!(item.snapshot.generation.cfg_value, 4.0);
}

#[test]
fn canceling_regeneration_with_existing_audio_returns_item_to_ready() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let id = queue.enqueue("text".to_string(), "loaded-model-a", &config);
    queue.mark_ready(&id);

    assert!(queue.start_regeneration(&id, "loaded-model-a", &config));
    assert!(queue.mark_canceled(&id));

    let item = &queue.items()[0];
    assert_eq!(item.status, HistoryStatus::Ready);
    assert!(item.has_audio);
    assert_eq!(item.error, None);
}

#[test]
fn playback_marks_ready_audio_as_playing_and_stops_it() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let ready_id = queue.enqueue("ready text".to_string(), "model-a", &config);
    let queued_id = queue.enqueue("queued text".to_string(), "model-a", &config);

    assert!(!queue.mark_playing(&ready_id));

    queue.mark_ready(&ready_id);
    assert!(queue.mark_playing(&ready_id));
    assert!(!queue.mark_playing(&queued_id));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Playing);
    assert_eq!(items[1].status, HistoryStatus::Queued);

    assert_eq!(queue.mark_all_stopped(), Some(ready_id.clone()));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Ready);
    assert_eq!(items[1].status, HistoryStatus::Queued);
}
