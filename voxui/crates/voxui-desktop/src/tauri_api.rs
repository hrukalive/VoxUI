use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_namespace = ["window", "__TAURI__", "core"])]
    async fn invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"])]
    async fn listen(event: &str, handler: &Closure<dyn Fn(JsValue)>) -> JsValue;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub models: Vec<ModelChoice>,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub load_state: LoadUiState,
    pub history: Vec<HistoryItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_root: Option<String>,
    pub selected_model_id: Option<String>,
    pub language: LanguageMode,
    pub backend: BackendKind,
    pub audio_host: Option<String>,
    pub audio_device: Option<String>,
    pub volume: f32,
    pub max_input_chars: usize,
    pub generation: GenerationSettings,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioState {
    pub hosts: Vec<AudioHost>,
    pub devices: Vec<AudioDevice>,
    pub default_host: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioHost {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub host_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationSettings {
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ConfigPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_root: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_model_id: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<LanguageMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_host: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_device: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationSettings>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
    pub model_dir: String,
    pub model_path: String,
    pub lora_path: Option<String>,
    pub model_bytes: u64,
    pub lora_bytes: u64,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub model_id: String,
    pub backend: BackendKind,
    pub generation: GenerationSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageMode {
    System,
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadUiState {
    Idle,
    Loading,
}

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

#[derive(Debug, Serialize)]
struct ItemArgs {
    #[serde(rename = "itemId")]
    item_id: String,
}

pub async fn get_app_state() -> Result<AppSnapshot, String> {
    let value = invoke("get_app_state", JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn get_audio_state() -> Result<AudioState, String> {
    let value = invoke("get_audio_state", JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn browse_model_dir() -> Result<Option<String>, String> {
    browse_path("browse_model_dir").await
}

pub async fn browse_prompt_wav() -> Result<Option<String>, String> {
    browse_path("browse_prompt_wav").await
}

pub async fn browse_reference_wav() -> Result<Option<String>, String> {
    browse_path("browse_reference_wav").await
}

pub async fn set_config_patch(patch: ConfigPatch) -> Result<AppSnapshot, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "patch": patch }))
        .map_err(|err| err.to_string())?;
    let value = invoke("set_config_patch", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn enqueue_generation(text: String) -> Result<HistoryItem, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "text": text }))
        .map_err(|err| err.to_string())?;
    let value = invoke("enqueue_generation", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn play_audio(item_id: String) -> Result<(), String> {
    let args =
        serde_wasm_bindgen::to_value(&ItemArgs { item_id }).map_err(|err| err.to_string())?;
    invoke("play_audio", args)
        .await
        .map_err(stringify_js_error)
        .map(|_| ())
}

pub async fn regenerate(item_id: String) -> Result<(), String> {
    let args =
        serde_wasm_bindgen::to_value(&ItemArgs { item_id }).map_err(|err| err.to_string())?;
    invoke("regenerate", args)
        .await
        .map_err(stringify_js_error)
        .map(|_| ())
}

pub async fn test_audio() -> Result<(), String> {
    invoke("test_audio", JsValue::NULL)
        .await
        .map_err(stringify_js_error)
        .map(|_| ())
}

pub async fn listen_app_event(
    event: &'static str,
    handler: impl Fn(JsValue) + 'static,
) -> Result<(), String> {
    let closure = Closure::wrap(Box::new(handler) as Box<dyn Fn(JsValue)>);
    let value = listen(event, &closure).await;

    if value.is_undefined() || value.is_function() {
        closure.forget();
        Ok(())
    } else {
        Err(stringify_js_error(value))
    }
}

async fn browse_path(command: &str) -> Result<Option<String>, String> {
    let value = invoke(command, JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

fn stringify_js_error(value: JsValue) -> String {
    if let Some(message) = value.as_string() {
        return message;
    }

    js_sys::JSON::stringify(&value)
        .ok()
        .and_then(|json| json.as_string())
        .unwrap_or_else(|| format!("{value:?}"))
}
