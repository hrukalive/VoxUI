use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::generation_queue::{GenerationQueue, HistoryItem, HistoryStatus};
use crate::model_discovery::discover_models;
use crate::playback::{GeneratedAudio, GeneratedAudioCache};
use crate::types::{AppConfig, AppSnapshot, ConfigPatch, LoadUiState, ModelChoice};
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

struct ActiveModelLoad {
    id: u64,
    cancel: Arc<AtomicBool>,
}

pub struct AppCore {
    config: AppConfig,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_state: LoadUiState,
    engine: Option<voxui_inference::VoxCPMEngine>,
    next_load_id: u64,
    active_load: Option<ActiveModelLoad>,
    queue: GenerationQueue,
    audio_cache: GeneratedAudioCache,
    active_generation_item_id: Option<String>,
}

pub struct GenerationRun {
    pub item_id: String,
    pub request: SynthesisRequest,
    pub engine: VoxCPMEngine,
    pub sample_rate: u32,
}

impl AppCore {
    pub fn from_config(mut config: AppConfig) -> Result<Self> {
        let models = match config.model_root.as_deref() {
            Some(root) => discover_models(root)?,
            None => Vec::new(),
        };
        let selected_model_id = select_existing_model(config.selected_model_id.clone(), &models);
        config.selected_model_id = selected_model_id.clone();

        Ok(Self {
            config,
            models,
            selected_model_id,
            loaded_model_id: None,
            load_state: LoadUiState::Idle,
            engine: None,
            next_load_id: 1,
            active_load: None,
            queue: GenerationQueue::default(),
            audio_cache: GeneratedAudioCache::default(),
            active_generation_item_id: None,
        })
    }

    pub fn snapshot(&self) -> AppSnapshot {
        AppSnapshot {
            config: self.config.clone(),
            models: self.models.clone(),
            selected_model_id: self.selected_model_id.clone(),
            loaded_model_id: self.loaded_model_id.clone(),
            load_state: self.load_state,
            history: self.queue.items().to_vec(),
        }
    }

    pub fn apply_patch(&mut self, patch: ConfigPatch) -> Result<AppSnapshot> {
        if let Some(model_root) = patch.model_root {
            self.cancel_model_load_state();
            self.config.model_root = model_root;
            self.rescan_models()?;
        }

        if let Some(selected_model_id) = patch.selected_model_id {
            self.cancel_model_load_state();
            self.selected_model_id = select_existing_model(selected_model_id, &self.models);
            self.config.selected_model_id = self.selected_model_id.clone();
        }

        if let Some(language) = patch.language {
            self.config.language = language;
        }
        if let Some(backend) = patch.backend {
            self.config.backend = backend;
        }
        if let Some(audio_host) = patch.audio_host {
            if self.config.audio_host != audio_host {
                self.config.audio_device = None;
            }
            self.config.audio_host = audio_host;
        }
        if let Some(audio_device) = patch.audio_device {
            self.config.audio_device = audio_device;
        }
        if let Some(volume) = patch.volume {
            self.config.volume = volume.clamp(0.0, 1.0);
        }
        if let Some(max_input_chars) = patch.max_input_chars {
            self.config.max_input_chars = max_input_chars.max(1);
        }
        if let Some(generation) = patch.generation {
            self.config.generation = generation;
        }

        Ok(self.snapshot())
    }

    pub fn rescan_models(&mut self) -> Result<Vec<ModelChoice>> {
        self.cancel_model_load_state();
        self.models = match self.config.model_root.as_deref() {
            Some(root) => discover_models(root)?,
            None => Vec::new(),
        };
        self.selected_model_id =
            select_existing_model(self.selected_model_id.clone(), &self.models);
        self.config.selected_model_id = self.selected_model_id.clone();
        Ok(self.models.clone())
    }

    pub fn enqueue_generation(&mut self, text: String) -> Result<HistoryItem> {
        let text = text.trim().to_string();
        if text.is_empty() {
            bail!("input text is empty");
        }
        if text.chars().count() > self.config.max_input_chars {
            bail!(
                "input text exceeds maximum length of {} characters",
                self.config.max_input_chars
            );
        }

        let loaded_model_id = self
            .loaded_model_id
            .clone()
            .context("no model loaded for generation")?;
        let id = self.queue.enqueue(text, loaded_model_id, &self.config);

        self.queue
            .items()
            .iter()
            .find(|item| item.id == id)
            .cloned()
            .context("queued item was not found after enqueue")
    }

    pub fn regenerate_item(&mut self, item_id: &str, config: &AppConfig) -> Result<()> {
        let loaded_model_id = self
            .loaded_model_id
            .clone()
            .context("no model loaded for generation")?;
        if !self
            .queue
            .start_regeneration(item_id, loaded_model_id, config)
        {
            bail!("unknown history item: {item_id}");
        }
        Ok(())
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn has_audio(&self, item_id: &str) -> bool {
        self.queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .map(|item| item.has_audio && self.audio_cache.contains(item_id))
            .unwrap_or(false)
    }

    pub fn run_generation_now<F>(
        &mut self,
        item_id: &str,
        progress: F,
    ) -> Result<(u32, f32), String>
    where
        F: Fn(usize, usize),
    {
        let run = self.begin_generation_run(item_id)?;
        let result = Self::execute_generation_run(run, progress);
        match result {
            Ok((run, samples, duration_seconds)) => {
                let sample_rate = run.sample_rate;
                self.finish_generation_success(run, samples, duration_seconds);
                Ok((sample_rate, duration_seconds))
            }
            Err((run, error)) => Err(self.finish_generation_failure(run, error)),
        }
    }

    pub fn begin_generation_run(&mut self, item_id: &str) -> Result<GenerationRun, String> {
        if self.active_load.is_some() || self.load_state == LoadUiState::Loading {
            return Err("model load already in progress".to_string());
        }
        let item = self
            .queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .ok_or_else(|| format!("unknown history item: {item_id}"))?;
        if item.status != HistoryStatus::Queued {
            return Err(format!(
                "history item is not queued for generation: {item_id}"
            ));
        }
        if self.active_generation_item_id.is_some() {
            return Err("generation already in progress".to_string());
        }
        let request = self
            .synthesis_request(item_id)
            .map_err(|error| error.to_string())?;
        let engine = self
            .engine
            .take()
            .ok_or_else(|| "no model engine loaded for generation".to_string())?;
        let sample_rate = engine.sample_rate();

        if let Err(error) = self.begin_generation_state(item_id) {
            self.engine = Some(engine);
            return Err(error);
        }

        Ok(GenerationRun {
            item_id: item_id.to_string(),
            request,
            engine,
            sample_rate,
        })
    }

    pub fn begin_generation_for_test(&mut self, item_id: &str) -> Result<(), String> {
        self.begin_generation_state(item_id)
    }

    fn begin_generation_state(&mut self, item_id: &str) -> Result<(), String> {
        let item = self
            .queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .ok_or_else(|| format!("unknown history item: {item_id}"))?;
        if item.status != HistoryStatus::Queued {
            return Err(format!(
                "history item is not queued for generation: {item_id}"
            ));
        }
        if self.active_generation_item_id.is_some() {
            return Err("generation already in progress".to_string());
        }
        if !self.queue.mark_generating(item_id) {
            return Err(format!("unknown history item: {item_id}"));
        }
        self.active_generation_item_id = Some(item_id.to_string());
        Ok(())
    }

    pub fn mark_generation_progress(&mut self, item_id: &str, current: usize, total: usize) -> bool {
        self.queue.mark_progress(item_id, current, total)
    }

    pub fn finish_generation_success(
        &mut self,
        run: GenerationRun,
        samples: Vec<f32>,
        duration_seconds: f32,
    ) {
        let item_id = run.item_id;
        self.engine = Some(run.engine);
        if self.active_generation_item_id.as_deref() == Some(item_id.as_str()) {
            self.active_generation_item_id = None;
        }
        self.audio_cache.insert(
            item_id.clone(),
            GeneratedAudio {
                samples,
                sample_rate: run.sample_rate,
            },
        );
        self.queue.mark_ready(&item_id);
        let _ = duration_seconds;
    }

    pub fn finish_generation_failure(&mut self, run: GenerationRun, error: String) -> String {
        let item_id = run.item_id;
        self.engine = Some(run.engine);
        if self.active_generation_item_id.as_deref() == Some(item_id.as_str()) {
            self.active_generation_item_id = None;
        }
        self.queue.mark_failed(&item_id, error.clone());
        error
    }

    pub fn execute_generation_run<F>(
        mut run: GenerationRun,
        progress: F,
    ) -> std::result::Result<(GenerationRun, Vec<f32>, f32), (GenerationRun, String)>
    where
        F: Fn(usize, usize),
    {
        match run
            .engine
            .generate_cancellable(run.request.clone(), progress, None)
        {
            Ok(samples) => {
                let duration_seconds = samples.len() as f32 / run.sample_rate as f32;
                Ok((run, samples, duration_seconds))
            }
            Err(error) => Err((run, error.to_string())),
        }
    }

    pub fn synthesis_request_for_test(
        &self,
        item_id: &str,
    ) -> Result<voxui_inference::SynthesisRequest> {
        self.synthesis_request(item_id)
    }

    fn synthesis_request(&self, item_id: &str) -> Result<voxui_inference::SynthesisRequest> {
        let item = self
            .queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .ok_or_else(|| anyhow::anyhow!("unknown history item: {item_id}"))?;
        Ok(voxui_inference::SynthesisRequest {
            text: item.text.clone(),
            prompt_wav_path: item.snapshot.generation.prompt_wav_path.clone(),
            prompt_text: item.snapshot.generation.prompt_text.clone(),
            reference_wav_path: item.snapshot.generation.reference_wav_path.clone(),
            cfg_value: item.snapshot.generation.cfg_value,
            inference_timesteps: item.snapshot.generation.inference_timesteps,
            min_len: item.snapshot.generation.min_len,
            max_len: item.snapshot.generation.max_len,
            normalize: false,
            retry_badcase: item.snapshot.generation.retry_badcase,
            retry_badcase_max_times: item.snapshot.generation.retry_badcase_max_times,
            retry_badcase_ratio_threshold: item.snapshot.generation.retry_badcase_ratio_threshold,
        })
    }

    pub fn cancel_model_load_state(&mut self) {
        if let Some(active_load) = self.active_load.take() {
            active_load.cancel.store(true, Ordering::SeqCst);
        }
        self.load_state = LoadUiState::Idle;
    }

    pub fn selected_choice(&self) -> Result<ModelChoice> {
        let selected = self
            .selected_model_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no model selected"))?;
        self.models
            .iter()
            .find(|choice| &choice.id == selected)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("selected model is no longer available"))
    }

    pub fn mark_load_started(&mut self) -> Result<(u64, Arc<AtomicBool>)> {
        if self.active_generation_item_id.is_some() {
            bail!("generation already in progress");
        }
        if self.active_load.is_some() || self.load_state == LoadUiState::Loading {
            bail!("model load already in progress");
        }

        let load_id = self.next_load_id;
        self.next_load_id = self.next_load_id.saturating_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.active_load = Some(ActiveModelLoad {
            id: load_id,
            cancel: cancel.clone(),
        });
        self.load_state = LoadUiState::Loading;
        Ok((load_id, cancel))
    }

    pub fn mark_load_success(
        &mut self,
        load_id: u64,
        choice_id: String,
        engine: voxui_inference::VoxCPMEngine,
    ) -> bool {
        self.mark_load_success_inner(load_id, choice_id, Some(engine))
    }

    fn mark_load_success_inner(
        &mut self,
        load_id: u64,
        choice_id: String,
        engine: Option<voxui_inference::VoxCPMEngine>,
    ) -> bool {
        if !self.active_load_matches(load_id) {
            return false;
        }

        self.active_load = None;
        if let Some(engine) = engine {
            self.engine = Some(engine);
        }
        self.loaded_model_id = Some(choice_id);
        self.load_state = LoadUiState::Idle;
        true
    }

    pub fn mark_load_finished_without_swap(&mut self) {
        self.load_state = LoadUiState::Idle;
    }

    pub fn mark_load_finished_without_swap_for_load(&mut self, load_id: u64) -> bool {
        if !self.active_load_matches(load_id) {
            return false;
        }

        self.active_load = None;
        self.load_state = LoadUiState::Idle;
        true
    }

    pub fn finish_model_load_for_test(&mut self, result: Result<(), String>) {
        match result {
            Ok(()) => {
                self.loaded_model_id = Some("new-model".to_string());
                self.load_state = LoadUiState::Idle;
            }
            Err(_) => self.mark_load_finished_without_swap(),
        }
    }

    pub fn begin_model_load_for_test(&mut self) -> (u64, Arc<AtomicBool>) {
        self.mark_load_started().unwrap()
    }

    pub fn complete_model_load_success_for_test(
        &mut self,
        load_id: u64,
        choice_id: String,
    ) -> bool {
        self.mark_load_success_inner(load_id, choice_id, None)
    }

    pub fn cancel_generation_item(&mut self, item_id: &str) -> bool {
        self.queue.cancel_queued(item_id)
    }

    pub fn set_loaded_model_for_test(&mut self, model_id: String) {
        self.loaded_model_id = Some(model_id);
    }

    fn active_load_matches(&self, load_id: u64) -> bool {
        self.active_load
            .as_ref()
            .is_some_and(|active_load| active_load.id == load_id)
    }
}

pub fn load_button_enabled(
    selected_model_id: Option<&str>,
    loaded_model_id: Option<&str>,
    load_state: LoadUiState,
    generation_running: bool,
) -> bool {
    let Some(selected_model_id) = selected_model_id else {
        return false;
    };

    load_state == LoadUiState::Idle
        && !generation_running
        && loaded_model_id != Some(selected_model_id)
}

fn select_existing_model(saved: Option<String>, models: &[ModelChoice]) -> Option<String> {
    if let Some(saved) = saved {
        if models.iter().any(|model| model.id == saved) {
            return Some(saved);
        }
    }

    models.first().map(|model| model.id.clone())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    use super::*;
    use crate::types::{BackendKind, ConfigPatch, GenerationSettings, LanguageMode};

    #[test]
    fn apply_patch_rescans_model_root_and_clamps_values() {
        let temp = TempDir::new().unwrap();
        write_model(temp.path(), "alpha");

        let mut core = AppCore::from_config(AppConfig::default()).unwrap();
        let snapshot = core
            .apply_patch(ConfigPatch {
                model_root: Some(Some(temp.path().to_path_buf())),
                selected_model_id: None,
                language: Some(LanguageMode::English),
                backend: Some(BackendKind::Cuda),
                audio_host: Some(Some("host-a".to_string())),
                audio_device: Some(Some("device-a".to_string())),
                volume: Some(1.25),
                max_input_chars: Some(0),
                generation: Some(GenerationSettings {
                    cfg_value: 3.0,
                    ..GenerationSettings::default()
                }),
            })
            .unwrap();

        assert_eq!(snapshot.models.len(), 1);
        assert_eq!(snapshot.selected_model_id.as_deref(), Some("alpha"));
        assert_eq!(snapshot.config.selected_model_id.as_deref(), Some("alpha"));
        assert_eq!(snapshot.config.language, LanguageMode::English);
        assert_eq!(snapshot.config.backend, BackendKind::Cuda);
        assert_eq!(snapshot.config.audio_host.as_deref(), Some("host-a"));
        assert_eq!(snapshot.config.audio_device.as_deref(), Some("device-a"));
        assert_eq!(snapshot.config.volume, 1.0);
        assert_eq!(snapshot.config.max_input_chars, 1);
        assert_eq!(snapshot.config.generation.cfg_value, 3.0);
    }

    #[test]
    fn rescan_models_preserves_valid_selection_and_updates_config() {
        let temp = TempDir::new().unwrap();
        write_model(temp.path(), "alpha");
        write_model(temp.path(), "beta");

        let mut config = AppConfig {
            model_root: Some(temp.path().to_path_buf()),
            selected_model_id: Some("beta".to_string()),
            ..AppConfig::default()
        };
        let mut core = AppCore::from_config(config.clone()).unwrap();

        fs::remove_file(temp.path().join("beta").join("model.gguf")).unwrap();
        let models = core.rescan_models().unwrap();

        config.selected_model_id = Some("alpha".to_string());
        assert_eq!(models.len(), 1);
        assert_eq!(core.snapshot().selected_model_id.as_deref(), Some("alpha"));
        assert_eq!(
            core.snapshot().config.selected_model_id,
            config.selected_model_id
        );
    }

    #[test]
    fn apply_patch_can_clear_nullable_config_fields() {
        let mut core = AppCore::from_config(AppConfig {
            selected_model_id: Some("missing".to_string()),
            audio_host: Some("host-a".to_string()),
            audio_device: Some("device-a".to_string()),
            ..AppConfig::default()
        })
        .unwrap();

        let snapshot = core
            .apply_patch(ConfigPatch {
                model_root: Some(None),
                selected_model_id: Some(None),
                language: None,
                backend: None,
                audio_host: Some(None),
                audio_device: Some(None),
                volume: Some(-0.5),
                max_input_chars: None,
                generation: None,
            })
            .unwrap();

        assert!(snapshot.models.is_empty());
        assert_eq!(snapshot.selected_model_id, None);
        assert_eq!(snapshot.config.selected_model_id, None);
        assert_eq!(snapshot.config.audio_host, None);
        assert_eq!(snapshot.config.audio_device, None);
        assert_eq!(snapshot.config.volume, 0.0);
    }

    #[test]
    fn apply_patch_normalizes_missing_selected_model_id() {
        let temp = TempDir::new().unwrap();
        write_model(temp.path(), "alpha");

        let mut core = AppCore::from_config(AppConfig {
            model_root: Some(temp.path().to_path_buf()),
            ..AppConfig::default()
        })
        .unwrap();

        let snapshot = core
            .apply_patch(ConfigPatch {
                model_root: None,
                selected_model_id: Some(Some("missing".to_string())),
                language: None,
                backend: None,
                audio_host: None,
                audio_device: None,
                volume: None,
                max_input_chars: None,
                generation: None,
            })
            .unwrap();

        assert_eq!(snapshot.selected_model_id.as_deref(), Some("alpha"));
        assert_eq!(snapshot.config.selected_model_id.as_deref(), Some("alpha"));
    }

    #[test]
    fn cancel_generation_item_only_cancels_queued_items() {
        let mut core = AppCore::from_config(AppConfig::default()).unwrap();
        core.set_loaded_model_for_test("alpha".to_string());
        let item = core.enqueue_generation("hello".to_string()).unwrap();

        assert!(core.cancel_generation_item(&item.id));
        assert!(!core.cancel_generation_item(&item.id));
        assert!(!core.cancel_generation_item("missing"));
    }

    fn write_model(root: &Path, name: &str) {
        let model_dir = root.join(name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"model").unwrap();
    }
}
