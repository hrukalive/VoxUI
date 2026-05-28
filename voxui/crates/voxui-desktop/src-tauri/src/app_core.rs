use std::collections::{btree_map::Entry, BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use anyhow::{bail, Context, Result};
use voxui_audio::VolumeHandle;

use crate::generation_queue::{GenerationQueue, HistoryItem, HistoryStatus};
use crate::live::{LiveEvent, LiveLanguage, LiveState, SuggestionMode};
use crate::model_discovery::discover_models;
use crate::playback::{GeneratedAudio, GeneratedAudioCache};
use crate::types::{
    AppConfig, AppSnapshot, ConfigPatch, LiveConfigPatch, LiveSnapshot, LiveStatus, LoadUiState,
    ModelChoice, SidecarCapabilities,
};
use voxui_inference::SynthesisRequest;

#[derive(Debug)]
struct ActiveModelLoad {
    id: u64,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug)]
struct ActiveGeneration {
    item_id: String,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug)]
struct ActivePlayback {
    item_id: String,
    volume: VolumeHandle,
    stop: mpsc::Sender<()>,
}

pub struct AppCore {
    config: AppConfig,
    backend_saved: bool,
    sidecar_capabilities: SidecarCapabilities,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    loaded_sample_rate: Option<u32>,
    load_state: LoadUiState,
    next_load_id: u64,
    active_load: Option<ActiveModelLoad>,
    config_path: Option<PathBuf>,
    queue: GenerationQueue,
    audio_cache: GeneratedAudioCache,
    active_generation: Option<ActiveGeneration>,
    active_generation_item_id: Option<String>,
    active_generation_audio_backup: Option<(String, GeneratedAudio)>,
    active_playback: Option<ActivePlayback>,
    pending_auto_playback: VecDeque<String>,
    live: LiveState,
}

#[derive(Debug)]
pub struct GenerationRun {
    pub item_id: String,
    pub request: SynthesisRequest,
    pub sample_rate: u32,
    pub streaming: bool,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SidecarExitRecovery {
    pub failed_load: bool,
    pub failed_generation_item_id: Option<String>,
    pub stopped_generation_item_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarGenerationOutcome {
    Ready,
    Canceled,
}

pub struct PlaybackRun {
    pub item_id: String,
    pub audio: GeneratedAudio,
    pub volume: VolumeHandle,
    pub stop: mpsc::Receiver<()>,
}

pub struct PlaybackCompletion {
    pub stopped_item_id: Option<String>,
    pub next_run: Option<PlaybackRun>,
}

impl AppCore {
    pub fn from_config(config: AppConfig) -> Result<Self> {
        Self::from_loaded_config(config, true)
    }

    pub fn from_loaded_config(mut config: AppConfig, backend_saved: bool) -> Result<Self> {
        let sidecar_capabilities = SidecarCapabilities::default();
        normalize_backend_for_sidecar(&mut config, backend_saved, sidecar_capabilities);
        normalize_generation_settings(&mut config.generation);
        let models = match config.model_root.as_deref() {
            Some(root) => discover_models(root)?,
            None => Vec::new(),
        };
        let selected_model_id = select_existing_model(config.selected_model_id.clone(), &models);
        config.selected_model_id = selected_model_id.clone();

        Ok(Self {
            config,
            backend_saved,
            sidecar_capabilities,
            models,
            selected_model_id,
            loaded_model_id: None,
            loaded_sample_rate: None,
            load_state: LoadUiState::Idle,
            next_load_id: 1,
            active_load: None,
            config_path: None,
            queue: GenerationQueue::default(),
            audio_cache: GeneratedAudioCache::default(),
            active_generation: None,
            active_generation_item_id: None,
            active_generation_audio_backup: None,
            active_playback: None,
            pending_auto_playback: VecDeque::new(),
            live: LiveState::default(),
        })
    }

    pub fn snapshot(&self) -> AppSnapshot {
        AppSnapshot {
            config: self.config.clone(),
            system_language: crate::config::detect_system_language(),
            cuda_available: self.sidecar_capabilities.cuda_available,
            models: self.models.clone(),
            selected_model_id: self.selected_model_id.clone(),
            loaded_model_id: self.loaded_model_id.clone(),
            load_state: self.load_state,
            history: self.queue.items().to_vec(),
        }
    }

    pub fn apply_sidecar_capabilities(&mut self, capabilities: SidecarCapabilities) -> AppSnapshot {
        self.sidecar_capabilities = capabilities;
        normalize_backend_for_sidecar(&mut self.config, self.backend_saved, capabilities);
        self.snapshot()
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
        if let Some(theme) = patch.theme {
            self.config.theme = theme;
        }
        if let Some(backend) = patch.backend {
            self.config.backend = backend;
            self.backend_saved = true;
            self.config.normalize_for_sidecar(self.sidecar_capabilities);
        }
        if let Some(audio_host) = patch.audio_host {
            let audio_host = empty_string_as_none(audio_host);
            if self.config.audio_host != audio_host {
                self.config.audio_device = None;
            }
            self.config.audio_host = audio_host;
        }
        if let Some(audio_device) = patch.audio_device {
            self.config.audio_device = empty_string_as_none(audio_device);
        }
        if let Some(volume) = patch.volume {
            self.config.volume = volume.clamp(0.0, 1.0);
            if let Some(active_playback) = self.active_playback.as_ref() {
                active_playback.volume.set(self.config.volume);
            }
        }
        if let Some(max_input_chars) = patch.max_input_chars {
            self.config.max_input_chars = max_input_chars.max(1);
        }
        if let Some(mut generation) = patch.generation {
            normalize_generation_settings(&mut generation);
            self.config.generation = generation;
        }

        self.persist_config()?;
        Ok(self.snapshot())
    }

    pub fn add_live_event(&mut self, event: LiveEvent) -> Result<String> {
        let username_mapping_changed = if !event.open_id.is_empty() && !event.uname.is_empty() {
            initialize_uname_mapping(
                &mut self.config.live.original_unames,
                &event.open_id,
                &event.uname,
            ) | initialize_uname_mapping(
                &mut self.config.live.mapped_unames,
                &event.open_id,
                &event.uname,
            )
        } else {
            false
        };

        let item_id = self.live.add_event(event);
        if username_mapping_changed {
            self.persist_config()?;
        }
        Ok(item_id)
    }

    pub fn apply_live_patch(&mut self, patch: LiveConfigPatch) -> Result<LiveSnapshot> {
        if let Some(identity_code) = patch.identity_code {
            self.config.live.identity_code = identity_code;
        }
        if let Some(enable_ceve_server_heartbeat) = patch.enable_ceve_server_heartbeat {
            self.config.live.enable_ceve_server_heartbeat = enable_ceve_server_heartbeat;
        }
        if let Some(show_danmu) = patch.show_danmu {
            self.config.live.show_danmu = show_danmu;
        }
        if let Some(show_gifts) = patch.show_gifts {
            self.config.live.show_gifts = show_gifts;
        }
        if let Some(show_superchats) = patch.show_superchats {
            self.config.live.show_superchats = show_superchats;
        }
        if let Some(show_guards) = patch.show_guards {
            self.config.live.show_guards = show_guards;
        }
        if let Some(show_likes) = patch.show_likes {
            self.config.live.show_likes = show_likes;
        }
        if let Some(show_enters) = patch.show_enters {
            self.config.live.show_enters = show_enters;
        }
        if let Some(templates) = patch.templates {
            self.config.live.templates = templates;
        }
        if let Some(replacement_rules) = patch.replacement_rules {
            self.config.live.replacement_rules = replacement_rules;
        }
        if let Some(mapped_unames) = patch.mapped_unames {
            self.config.live.mapped_unames = mapped_unames;
        }

        self.persist_config()?;
        Ok(self.live_snapshot(LiveLanguage::English))
    }

    pub fn live_snapshot(&self, language: LiveLanguage) -> LiveSnapshot {
        self.live.snapshot(&self.config.live, language)
    }

    pub fn set_live_status(
        &mut self,
        status: LiveStatus,
        status_message: Option<String>,
    ) -> LiveSnapshot {
        self.live.set_status(status, status_message);
        self.live_snapshot(LiveLanguage::English)
    }

    pub fn clear_live_items(&mut self) {
        self.live.clear_items();
    }

    pub fn live_suggestion_for_item(
        &self,
        item_id: &str,
        language: LiveLanguage,
        mode: SuggestionMode,
    ) -> Option<String> {
        self.live
            .suggestion_for_item(item_id, &self.config.live, language, mode)
    }

    pub fn add_live_event_for_test(&mut self, event: LiveEvent) -> Result<String> {
        self.add_live_event(event)
    }

    pub fn live_snapshot_for_test(&self, language: LiveLanguage) -> LiveSnapshot {
        self.live_snapshot(language)
    }

    pub fn live_status_for_test(&self) -> LiveStatus {
        self.live.status()
    }

    pub fn set_config_path(&mut self, path: PathBuf) {
        self.config_path = Some(path);
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

    pub fn regenerate_item_stopping_playback(
        &mut self,
        item_id: &str,
        config: &AppConfig,
    ) -> Result<Option<String>> {
        let stopped_item_id = self.stop_playback_if_active(item_id);
        self.pending_auto_playback
            .retain(|pending_item_id| pending_item_id != item_id);
        self.regenerate_item(item_id, config)?;
        Ok(stopped_item_id)
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

    fn ensure_playable(&self, item_id: &str) -> Result<()> {
        if !self.audio_cache.contains(item_id) {
            bail!("unknown item or no generated audio: {item_id}");
        }
        let item = self
            .queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .ok_or_else(|| anyhow::anyhow!("unknown history item: {item_id}"))?;
        if item.status != HistoryStatus::Ready || !item.has_audio {
            bail!("history item is not ready for playback: {item_id}");
        }
        Ok(())
    }

    pub fn begin_playback(&mut self, item_id: &str) -> Result<PlaybackRun> {
        self.pending_auto_playback
            .retain(|pending_item_id| pending_item_id != item_id);
        let audio = self
            .audio_cache
            .get(item_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown item or no generated audio: {item_id}"))?;
        if let Some(active_playback) = self.active_playback.take() {
            let _ = active_playback.stop.send(());
        }
        self.queue.mark_all_stopped();
        if !self.queue.mark_playing(item_id) {
            bail!("history item is not ready for playback: {item_id}");
        }

        let (stop_sender, stop_receiver) = mpsc::channel();
        let volume = VolumeHandle::new(self.config.volume);
        self.active_playback = Some(ActivePlayback {
            item_id: item_id.to_string(),
            volume: volume.clone(),
            stop: stop_sender,
        });

        Ok(PlaybackRun {
            item_id: item_id.to_string(),
            audio,
            volume,
            stop: stop_receiver,
        })
    }

    pub fn begin_or_queue_auto_playback(&mut self, item_id: &str) -> Result<Option<PlaybackRun>> {
        self.ensure_playable(item_id)?;
        if self.active_playback.is_some() {
            if !self
                .pending_auto_playback
                .iter()
                .any(|pending_item_id| pending_item_id == item_id)
            {
                self.pending_auto_playback.push_back(item_id.to_string());
            }
            return Ok(None);
        }

        self.begin_playback(item_id).map(Some)
    }

    pub fn stop_playback(&mut self) -> Option<String> {
        if let Some(active_playback) = self.active_playback.take() {
            let _ = active_playback.stop.send(());
        }
        self.queue.mark_all_stopped()
    }

    fn stop_playback_if_active(&mut self, item_id: &str) -> Option<String> {
        if !self
            .active_playback
            .as_ref()
            .is_some_and(|active_playback| active_playback.item_id == item_id)
        {
            return None;
        }

        self.stop_playback()
    }

    pub fn finish_playback(&mut self, item_id: &str) -> Option<String> {
        if self
            .active_playback
            .as_ref()
            .is_some_and(|active_playback| active_playback.item_id == item_id)
        {
            self.active_playback = None;
            return self.queue.mark_all_stopped();
        }
        None
    }

    pub fn finish_playback_and_next(&mut self, item_id: &str) -> PlaybackCompletion {
        let stopped_item_id = self.finish_playback(item_id);
        let mut next_run = None;

        while let Some(next_item_id) = self.pending_auto_playback.pop_front() {
            match self.begin_playback(&next_item_id) {
                Ok(run) => {
                    next_run = Some(run);
                    break;
                }
                Err(error) => {
                    tracing::warn!("failed to start pending playback {next_item_id}: {error}");
                }
            }
        }

        PlaybackCompletion {
            stopped_item_id,
            next_run,
        }
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
        progress(0, 0);
        let error = "generation execution is owned by the inference sidecar".to_string();
        Err(self.finish_generation_failure(run, error))
    }

    pub fn begin_generation_run(&mut self, item_id: &str) -> Result<GenerationRun, String> {
        if self.active_load.is_some() || self.load_state == LoadUiState::Loading {
            return Err("model load already in progress".to_string());
        }
        if self.loaded_model_id.is_none() {
            return Err("no model loaded for generation".to_string());
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
        let streaming = item.snapshot.generation.streaming;
        let request = self
            .synthesis_request(item_id)
            .map_err(|error| error.to_string())?;
        let sample_rate = self.loaded_sample_rate.unwrap_or(16_000);
        let cancel = Arc::new(AtomicBool::new(false));

        if let Err(error) = self.begin_generation_state(item_id, cancel.clone()) {
            return Err(error);
        }

        Ok(GenerationRun {
            item_id: item_id.to_string(),
            request,
            sample_rate,
            streaming,
            cancel,
        })
    }

    pub fn begin_generation_for_test(&mut self, item_id: &str) -> Result<(), String> {
        self.begin_generation_state(item_id, Arc::new(AtomicBool::new(false)))
            .map(|_| ())
    }

    pub fn begin_next_generation_run(&mut self) -> Result<Option<GenerationRun>, String> {
        if self.loaded_model_id.is_none() {
            return Err("no model loaded for generation".to_string());
        }
        if self.active_generation_item_id.is_some() {
            return Ok(None);
        }
        let Some(item_id) = self.queue.next_queued_id().map(str::to_string) else {
            return Ok(None);
        };
        self.begin_generation_run(&item_id).map(Some)
    }

    fn begin_generation_state(
        &mut self,
        item_id: &str,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let active_generation_audio_backup = {
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
            if item.has_audio {
                self.audio_cache
                    .get(item_id)
                    .cloned()
                    .map(|audio| (item_id.to_string(), audio))
            } else {
                None
            }
        };
        if !self.queue.mark_generating(item_id) {
            return Err(format!("unknown history item: {item_id}"));
        }
        self.active_generation_audio_backup = active_generation_audio_backup;
        if self.active_generation_audio_backup.is_some() {
            let _ = self.audio_cache.remove(item_id);
        }
        self.active_generation = Some(ActiveGeneration {
            item_id: item_id.to_string(),
            cancel,
        });
        self.active_generation_item_id = Some(item_id.to_string());
        Ok(())
    }

    pub fn mark_generation_progress(
        &mut self,
        item_id: &str,
        current: usize,
        total: usize,
    ) -> bool {
        if !self.accepts_sidecar_generation_event(item_id, false) {
            return false;
        }
        self.queue.mark_progress(item_id, current, total)
    }

    pub fn finish_generation_success(
        &mut self,
        run: GenerationRun,
        samples: Vec<f32>,
        duration_seconds: f32,
    ) {
        let item_id = run.item_id;
        self.clear_active_generation(&item_id);
        self.audio_cache.insert(
            item_id.clone(),
            GeneratedAudio {
                samples,
                sample_rate: run.sample_rate,
            },
        );
        self.queue.mark_ready(&item_id);
        self.active_generation_audio_backup = None;
        let _ = duration_seconds;
    }

    pub fn append_generation_audio_chunk(
        &mut self,
        item_id: &str,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<()> {
        if !self.accepts_sidecar_generation_event(item_id, false) {
            bail!("stale audio chunk for inactive item: {item_id}");
        }
        self.audio_cache
            .append(item_id.to_string(), samples, sample_rate)?;
        Ok(())
    }

    pub fn finish_generation_success_from_sidecar(
        &mut self,
        item_id: &str,
        sample_rate: u32,
        duration_seconds: f32,
    ) -> Result<SidecarGenerationOutcome> {
        if !self.accepts_sidecar_generation_event(item_id, true) {
            bail!("stale generation completion for inactive item: {item_id}");
        }
        if self.active_generation_canceled(item_id) {
            self.clear_active_generation(item_id);
            let _ = self.queue.mark_canceled(item_id);
            self.restore_generation_audio(item_id);
            self.active_generation_audio_backup = None;
            return Ok(SidecarGenerationOutcome::Canceled);
        }
        self.clear_active_generation(item_id);
        if !self.audio_cache.contains(item_id) {
            self.audio_cache.insert(
                item_id.to_string(),
                GeneratedAudio {
                    samples: Vec::new(),
                    sample_rate,
                },
            );
        }
        self.queue.mark_ready(item_id);
        self.active_generation_audio_backup = None;
        let _ = duration_seconds;
        Ok(SidecarGenerationOutcome::Ready)
    }

    pub fn finish_generation_failure(&mut self, run: GenerationRun, error: String) -> String {
        let item_id = run.item_id;
        self.clear_active_generation(&item_id);
        self.queue.mark_failed(&item_id, error.clone());
        self.restore_generation_audio(&item_id);
        self.active_generation_audio_backup = None;
        error
    }

    pub fn finish_generation_failure_from_sidecar(
        &mut self,
        item_id: &str,
        error: String,
    ) -> Result<String> {
        if !self.accepts_sidecar_generation_event(item_id, true) {
            bail!("stale generation failure for inactive item: {item_id}");
        }
        if self.active_generation_canceled(item_id) {
            self.clear_active_generation(item_id);
            let _ = self.queue.mark_canceled(item_id);
            self.restore_generation_audio(item_id);
            self.active_generation_audio_backup = None;
            return Ok(error);
        }
        self.clear_active_generation(item_id);
        self.queue.mark_failed(item_id, error.clone());
        self.restore_generation_audio(item_id);
        self.active_generation_audio_backup = None;
        Ok(error)
    }

    pub fn finish_generation_canceled(&mut self, run: GenerationRun) {
        let item_id = run.item_id;
        self.clear_active_generation(&item_id);
        if !self.queue.mark_canceled(&item_id) {
            let _ = self.queue.mark_canceled(&item_id);
        }
        self.restore_generation_audio(&item_id);
        self.active_generation_audio_backup = None;
    }

    pub fn finish_generation_canceled_from_sidecar(&mut self, item_id: &str) -> Result<()> {
        if !self.accepts_sidecar_generation_event(item_id, true) {
            bail!("stale generation cancellation for inactive item: {item_id}");
        }
        self.clear_active_generation(item_id);
        let _ = self.queue.mark_canceled(item_id);
        self.restore_generation_audio(item_id);
        self.active_generation_audio_backup = None;
        Ok(())
    }

    pub fn accepts_sidecar_generation_event(&self, item_id: &str, terminal: bool) -> bool {
        let Some(active_generation) = self.active_generation.as_ref() else {
            return false;
        };
        if active_generation.item_id != item_id {
            return false;
        }
        terminal || !active_generation.cancel.load(Ordering::SeqCst)
    }

    pub fn handle_sidecar_exit(&mut self, error: String) -> SidecarExitRecovery {
        let failed_load =
            self.active_load.take().is_some() || self.load_state == LoadUiState::Loading;
        self.load_state = LoadUiState::Idle;
        self.loaded_model_id = None;
        self.loaded_sample_rate = None;

        let stopped_generation_item_id = self.active_generation_item_id.clone();
        let mut failed_generation_item_id = None;
        if let Some(active_generation) = self.active_generation.take() {
            self.active_generation_item_id = None;
            let item_id = active_generation.item_id;
            if active_generation.cancel.load(Ordering::SeqCst) {
                self.restore_generation_audio(&item_id);
                let _ = self.queue.mark_canceled(&item_id);
            } else {
                self.restore_generation_audio(&item_id);
                self.queue.mark_failed(&item_id, error);
                failed_generation_item_id = Some(item_id);
            }
        } else {
            self.active_generation_item_id = None;
            self.active_generation_audio_backup = None;
        }
        self.active_generation_audio_backup = None;

        SidecarExitRecovery {
            failed_load,
            failed_generation_item_id,
            stopped_generation_item_id,
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
            retry_badcase: item.snapshot.generation.retry_badcase
                && !item.snapshot.generation.streaming,
            retry_badcase_max_times: item.snapshot.generation.retry_badcase_max_times,
            retry_badcase_ratio_threshold: item.snapshot.generation.retry_badcase_ratio_threshold,
            consolidate_n: item.snapshot.generation.stream_consolidate_n.max(1),
        })
    }

    pub fn cancel_model_load_state(&mut self) -> Option<u64> {
        let mut canceled_load_id = None;
        if let Some(active_load) = self.active_load.take() {
            canceled_load_id = Some(active_load.id);
            active_load.cancel.store(true, Ordering::SeqCst);
        }
        self.load_state = LoadUiState::Idle;
        canceled_load_id
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

    pub fn mark_load_success(&mut self, load_id: u64, choice_id: String, sample_rate: u32) -> bool {
        if !self.active_load_matches(load_id) {
            return false;
        }

        self.active_load = None;
        self.loaded_model_id = Some(choice_id);
        self.loaded_sample_rate = Some(sample_rate);
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
                self.loaded_sample_rate = Some(16_000);
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
        self.mark_load_success(load_id, choice_id, 16_000)
    }

    pub fn cancel_generation_item(&mut self, item_id: &str) -> bool {
        if self.queue.cancel_queued(item_id) {
            return true;
        }

        if self
            .active_generation
            .as_ref()
            .is_some_and(|active| active.item_id == item_id)
        {
            if let Some(active_generation) = self.active_generation.as_ref() {
                active_generation.cancel.store(true, Ordering::SeqCst);
            }
            self.restore_generation_audio(item_id);
            return self.queue.mark_canceled(item_id);
        }

        false
    }

    pub fn set_loaded_model_for_test(&mut self, model_id: String) {
        self.loaded_model_id = Some(model_id);
        self.loaded_sample_rate = Some(16_000);
    }

    pub fn set_generated_audio_for_test(
        &mut self,
        item_id: String,
        samples: Vec<f32>,
        sample_rate: u32,
    ) {
        self.audio_cache.insert(
            item_id.clone(),
            GeneratedAudio {
                samples,
                sample_rate,
            },
        );
        self.queue.mark_ready(&item_id);
    }

    fn active_load_matches(&self, load_id: u64) -> bool {
        self.active_load
            .as_ref()
            .is_some_and(|active_load| active_load.id == load_id)
    }

    pub fn active_load_id(&self) -> Option<u64> {
        self.active_load.as_ref().map(|active_load| active_load.id)
    }

    fn active_generation_canceled(&self, item_id: &str) -> bool {
        self.active_generation
            .as_ref()
            .is_some_and(|active| active.item_id == item_id && active.cancel.load(Ordering::SeqCst))
    }

    fn restore_generation_audio(&mut self, item_id: &str) {
        if let Some((backup_item_id, backup_audio)) = self.active_generation_audio_backup.as_ref() {
            self.audio_cache
                .insert(backup_item_id.clone(), backup_audio.clone());
        } else {
            let _ = self.audio_cache.remove(item_id);
        }
    }

    fn clear_active_generation(&mut self, item_id: &str) {
        if self
            .active_generation
            .as_ref()
            .is_some_and(|active| active.item_id == item_id)
        {
            self.active_generation = None;
        }
        if self.active_generation_item_id.as_deref() == Some(item_id) {
            self.active_generation_item_id = None;
        }
    }

    fn persist_config(&self) -> Result<()> {
        if let Some(path) = self.config_path.as_ref() {
            crate::config::save_config(path, &self.config)?;
        }
        Ok(())
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

fn empty_string_as_none(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

fn initialize_uname_mapping(
    mappings: &mut BTreeMap<String, String>,
    open_id: &str,
    uname: &str,
) -> bool {
    match mappings.entry(open_id.to_string()) {
        Entry::Vacant(entry) => {
            entry.insert(uname.to_string());
            true
        }
        Entry::Occupied(mut entry) if entry.get().is_empty() => {
            entry.insert(uname.to_string());
            true
        }
        Entry::Occupied(_) => false,
    }
}

fn normalize_generation_settings(generation: &mut crate::types::GenerationSettings) {
    generation.stream_consolidate_n = generation.stream_consolidate_n.max(1);
}

fn normalize_backend_for_sidecar(
    config: &mut AppConfig,
    backend_saved: bool,
    capabilities: SidecarCapabilities,
) {
    if backend_saved {
        config.normalize_for_sidecar(capabilities);
    } else {
        config.backend = capabilities.default_backend;
        config.normalize_for_sidecar(capabilities);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    use super::*;
    use crate::types::{BackendKind, ConfigPatch, GenerationSettings, LanguageMode, ThemeMode};

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
                theme: Some(ThemeMode::Light),
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
        assert_eq!(snapshot.config.theme, ThemeMode::Light);
        assert_eq!(snapshot.config.backend, BackendKind::Cpu);
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
    fn synthesis_request_uses_streaming_config() {
        let mut core = AppCore::from_config(AppConfig {
            generation: GenerationSettings {
                streaming: true,
                stream_consolidate_n: 4,
                retry_badcase: true,
                ..GenerationSettings::default()
            },
            ..AppConfig::default()
        })
        .unwrap();
        core.set_loaded_model_for_test("alpha".to_string());

        let item = core.enqueue_generation("hello".to_string()).unwrap();
        let request = core.synthesis_request_for_test(&item.id).unwrap();

        assert!(!request.retry_badcase);
        assert_eq!(request.consolidate_n, 4);
    }

    #[test]
    fn streaming_config_preserves_retry_preference_but_disables_request_retry() {
        let mut core = AppCore::from_config(AppConfig {
            generation: GenerationSettings {
                streaming: true,
                retry_badcase: true,
                ..GenerationSettings::default()
            },
            ..AppConfig::default()
        })
        .unwrap();
        core.set_loaded_model_for_test("alpha".to_string());

        assert!(core.snapshot().config.generation.retry_badcase);

        let item = core.enqueue_generation("hello".to_string()).unwrap();
        let request = core.synthesis_request_for_test(&item.id).unwrap();

        assert!(!request.retry_badcase);
    }

    #[test]
    fn playback_can_overlap_an_active_generation() {
        let mut core = AppCore::from_config(AppConfig::default()).unwrap();
        core.set_loaded_model_for_test("model".to_string());
        let ready = core.enqueue_generation("ready".to_string()).unwrap();
        core.set_generated_audio_for_test(ready.id.clone(), vec![0.0; 8], 16_000);
        let generating = core.enqueue_generation("generating".to_string()).unwrap();

        core.begin_generation_for_test(&generating.id).unwrap();
        let playback = core.begin_playback(&ready.id).unwrap();
        let snapshot = core.snapshot();

        assert_eq!(playback.item_id, ready.id);
        assert_eq!(snapshot.history[0].status, HistoryStatus::Playing);
        assert_eq!(snapshot.history[1].status, HistoryStatus::Generating);
    }

    #[test]
    fn sidecar_generation_run_does_not_take_local_engine() {
        let mut core = AppCore::from_config(AppConfig::default()).unwrap();
        core.set_loaded_model_for_test("model".to_string());
        let item = core.enqueue_generation("hello".to_string()).unwrap();

        let run = core.begin_generation_run(&item.id).unwrap();

        assert_eq!(run.item_id, item.id);
        assert_eq!(run.sample_rate, 16_000);
    }

    #[test]
    fn streaming_audio_chunks_accumulate_until_done() {
        let mut core = AppCore::from_config(AppConfig::default()).unwrap();
        core.set_loaded_model_for_test("model".to_string());
        let item = core.enqueue_generation("hello".to_string()).unwrap();
        core.begin_generation_for_test(&item.id).unwrap();

        core.append_generation_audio_chunk(&item.id, vec![0.1, 0.2], 16_000)
            .unwrap();
        core.append_generation_audio_chunk(&item.id, vec![0.3], 16_000)
            .unwrap();
        core.finish_generation_success_from_sidecar(&item.id, 16_000, 3.0 / 16_000.0)
            .unwrap();

        assert!(core.has_audio(&item.id));
        assert_eq!(core.audio_cache.get(&item.id).unwrap().samples.len(), 3);
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
                theme: None,
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
                theme: None,
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
