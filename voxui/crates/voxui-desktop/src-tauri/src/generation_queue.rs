use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{AppConfig, RequestSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    Queued,
    Generating,
    Canceled,
    Failed,
    Ready,
    Playing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryItem {
    pub id: String,
    pub text: String,
    pub status: HistoryStatus,
    pub progress_current: usize,
    pub progress_total: usize,
    pub error: Option<String>,
    pub has_audio: bool,
    pub snapshot: RequestSnapshot,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationQueue {
    items: Vec<HistoryItem>,
}

impl GenerationQueue {
    pub fn enqueue(
        &mut self,
        text: String,
        loaded_model_id: impl Into<String>,
        config: &AppConfig,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        self.items.push(HistoryItem {
            id: id.clone(),
            text,
            status: HistoryStatus::Queued,
            progress_current: 0,
            progress_total: 0,
            error: None,
            has_audio: false,
            snapshot: Self::snapshot(loaded_model_id, config),
        });
        id
    }

    pub fn items(&self) -> &[HistoryItem] {
        &self.items
    }

    pub fn next_queued_id(&self) -> Option<&str> {
        self.items
            .iter()
            .find(|item| item.status == HistoryStatus::Queued)
            .map(|item| item.id.as_str())
    }

    pub fn cancel_queued(&mut self, id: &str) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        if item.status != HistoryStatus::Queued {
            return false;
        }
        item.status = HistoryStatus::Canceled;
        item.progress_current = 0;
        item.progress_total = 0;
        item.error = None;
        true
    }

    pub fn mark_generating(&mut self, id: &str) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.status = HistoryStatus::Generating;
        item.error = None;
        true
    }

    pub fn mark_progress(&mut self, id: &str, current: usize, total: usize) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.progress_current = current;
        item.progress_total = total;
        true
    }

    pub fn mark_ready(&mut self, id: &str) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.status = HistoryStatus::Ready;
        item.error = None;
        item.has_audio = true;
        true
    }

    pub fn mark_playing(&mut self, id: &str) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        if item.status != HistoryStatus::Ready || !item.has_audio {
            return false;
        }
        item.status = HistoryStatus::Playing;
        true
    }

    pub fn mark_all_stopped(&mut self) -> Option<String> {
        let mut stopped_item_id = None;
        for item in &mut self.items {
            if item.status == HistoryStatus::Playing {
                item.status = HistoryStatus::Ready;
                stopped_item_id = Some(item.id.clone());
            }
        }
        stopped_item_id
    }

    pub fn mark_failed(&mut self, id: &str, error: String) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.status = HistoryStatus::Failed;
        item.error = Some(error);
        true
    }

    pub fn mark_canceled(&mut self, id: &str) -> bool {
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.status = if item.has_audio {
            HistoryStatus::Ready
        } else {
            HistoryStatus::Canceled
        };
        item.progress_current = 0;
        item.progress_total = 0;
        item.error = None;
        true
    }

    pub fn start_regeneration(
        &mut self,
        id: &str,
        loaded_model_id: impl Into<String>,
        config: &AppConfig,
    ) -> bool {
        let snapshot = Self::snapshot(loaded_model_id, config);
        let Some(item) = self.item_mut(id) else {
            return false;
        };
        item.status = HistoryStatus::Queued;
        item.progress_current = 0;
        item.progress_total = 0;
        item.error = None;
        item.snapshot = snapshot;
        true
    }

    fn snapshot(loaded_model_id: impl Into<String>, config: &AppConfig) -> RequestSnapshot {
        RequestSnapshot {
            model_id: loaded_model_id.into(),
            backend: config.backend,
            generation: config.generation.clone(),
        }
    }

    fn item_mut(&mut self, id: &str) -> Option<&mut HistoryItem> {
        self.items.iter_mut().find(|item| item.id == id)
    }
}
