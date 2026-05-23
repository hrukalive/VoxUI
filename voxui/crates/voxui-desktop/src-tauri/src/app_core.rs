use anyhow::{bail, Context, Result};

use crate::generation_queue::{GenerationQueue, HistoryItem};
use crate::model_discovery::discover_models;
use crate::types::{AppConfig, AppSnapshot, LoadUiState, ModelChoice};

#[derive(Debug)]
pub struct AppCore {
    config: AppConfig,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_state: LoadUiState,
    queue: GenerationQueue,
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
            queue: GenerationQueue::default(),
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

    pub fn set_loaded_model_for_test(&mut self, model_id: String) {
        self.loaded_model_id = Some(model_id);
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
