use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"])]
    async fn invoke(cmd: &str, args: JsValue) -> JsValue;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub models: Vec<ModelChoice>,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub history: Vec<HistoryItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub volume: f32,
    pub max_input_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryItem {
    pub id: String,
    pub text: String,
    pub status: String,
    pub progress_current: usize,
    pub progress_total: usize,
    pub has_audio: bool,
}

pub async fn get_app_state() -> Result<AppSnapshot, String> {
    let value = invoke("get_app_state", JsValue::NULL).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}
