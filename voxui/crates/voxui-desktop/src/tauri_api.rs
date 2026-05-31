use std::collections::BTreeMap;

use crate::i18n::Labels;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
export async function tauriInvoke(cmd, args) {
  const invoke = globalThis.__TAURI__?.core?.invoke;
  if (typeof invoke !== "function") {
    throw new Error("Tauri core invoke API is unavailable");
  }
  return await invoke(cmd, args);
}

export async function tauriListen(event, handler) {
  const listen = globalThis.__TAURI__?.event?.listen;
  if (typeof listen !== "function") {
    throw new Error("Tauri event listen API is unavailable");
  }
  return await listen(event, handler);
}

export function currentWindowLabel() {
  const current = globalThis.__TAURI__?.webviewWindow?.getCurrentWebviewWindow?.();
  return typeof current?.label === "string" ? current.label : "main";
}
"#)]
extern "C" {
    #[wasm_bindgen(catch, js_name = tauriInvoke)]
    async fn invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(catch, js_name = tauriListen)]
    async fn listen(event: &str, handler: &Closure<dyn Fn(JsValue)>) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_name = currentWindowLabel)]
    fn current_window_label_js() -> String;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub system_language: LanguageMode,
    pub cuda_available: bool,
    pub sidecar_init_error: Option<String>,
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
    pub theme: ThemeMode,
    pub backend: BackendKind,
    pub audio_host: Option<String>,
    pub audio_device: Option<String>,
    pub volume: f32,
    pub max_input_chars: usize,
    pub auto_period: bool,
    pub generation: GenerationSettings,
    pub translation: TranslationSettings,
    pub live: LiveConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationPair {
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationSettings {
    pub outbound: TranslationPair,
    pub inbound: TranslationPair,
    pub translate_enqueue: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioState {
    pub hosts: Vec<AudioHost>,
    pub devices: Vec<AudioDevice>,
    pub default_host: Option<String>,
    pub default_devices: Vec<AudioDevice>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandResult {
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadStartResult {
    pub started: bool,
    pub choice_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadProgressEvent {
    pub phase: String,
    pub loaded_bytes: u64,
    pub total_bytes: u64,
    pub component: Option<String>,
    pub component_index: usize,
    pub component_total: usize,
}

impl ModelLoadProgressEvent {
    pub fn percent(&self) -> f32 {
        if self.component_total > 0 {
            ((self.component_index as f64 / self.component_total as f64) * 100.0) as f32
        } else if self.total_bytes > 0 {
            ((self.loaded_bytes as f64 / self.total_bytes as f64) * 100.0) as f32
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadDoneEvent {
    pub status: String,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationProgressEvent {
    pub item_id: String,
    pub current: usize,
    pub total: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationDoneEvent {
    pub item_id: String,
    pub status: String,
    pub error: Option<String>,
    pub sample_rate: Option<u32>,
    pub duration_seconds: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaybackStateEvent {
    pub item_id: Option<String>,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MainInputReplaceEvent {
    pub text: String,
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
    pub streaming: bool,
    pub stream_consolidate_n: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveStatus {
    Disconnected,
    Connecting,
    Connected,
    Disconnecting,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveMessageKind {
    Danmu,
    Gift,
    Superchat,
    Guard,
    Like,
    Enter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SendMode {
    Manual,
    AutoEnqueue,
}

impl SendMode {
    pub fn value(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::AutoEnqueue => "auto_enqueue",
        }
    }

    pub fn from_value(value: &str) -> Self {
        match value {
            "auto_enqueue" => Self::AutoEnqueue,
            _ => Self::Manual,
        }
    }

    pub fn label(self, labels: Labels) -> &'static str {
        match self {
            Self::Manual => labels.send_mode_manual,
            Self::AutoEnqueue => labels.send_mode_auto_enqueue,
        }
    }

    pub fn direct_enqueue_on_click(self) -> bool {
        matches!(self, Self::AutoEnqueue)
    }
}

impl Default for SendMode {
    fn default() -> Self {
        Self::Manual
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoGenMode {
    None,
    Normal,
    Replacement,
}

impl AutoGenMode {
    pub fn value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Normal => "auto_gen",
            Self::Replacement => "auto_gen_replacement",
        }
    }

    pub fn from_value(value: &str) -> Self {
        match value {
            "auto_gen" => Self::Normal,
            "auto_gen_replacement" => Self::Replacement,
            _ => Self::None,
        }
    }

    pub fn label(self, labels: Labels) -> &'static str {
        match self {
            Self::None => labels.auto_gen_mode_none,
            Self::Normal => labels.auto_gen_mode_normal,
            Self::Replacement => labels.auto_gen_mode_replacement,
        }
    }
}

impl Default for AutoGenMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementRule {
    pub enabled: bool,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub danmu: String,
    pub gift_zh: String,
    pub gift_en: String,
    pub superchat_zh: String,
    pub superchat_en: String,
    pub guard_zh: String,
    pub guard_en: String,
    pub like_zh: String,
    pub like_en: String,
    pub enter_zh: String,
    pub enter_en: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveConfig {
    pub identity_code: String,
    pub enable_ceve_server_heartbeat: bool,
    pub show_danmu: bool,
    pub show_gifts: bool,
    pub show_superchats: bool,
    pub show_guards: bool,
    pub show_likes: bool,
    pub show_enters: bool,
    pub send_mode: SendMode,
    pub auto_gen_mode: AutoGenMode,
    pub auto_gen_danmu: bool,
    pub auto_gen_gifts: bool,
    pub auto_gen_superchats: bool,
    pub auto_gen_guards: bool,
    pub auto_gen_likes: bool,
    pub auto_gen_enters: bool,
    pub templates: TemplateConfig,
    pub replacement_rules: Vec<ReplacementRule>,
    pub mapped_unames: BTreeMap<String, String>,
    pub original_unames: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveConfigPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_ceve_server_heartbeat: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_danmu: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_gifts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_superchats: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_guards: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_likes: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_enters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_mode: Option<SendMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_mode: Option<AutoGenMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_danmu: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_gifts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_superchats: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_guards: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_likes: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_gen_enters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub templates: Option<TemplateConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_rules: Option<Vec<ReplacementRule>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_unames: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_unames: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveMonitorItem {
    pub id: String,
    pub kind: LiveMessageKind,
    pub paid: bool,
    pub open_id: String,
    pub uname: String,
    pub mapped_uname: String,
    pub suggestion: String,
    pub raw_message: Option<String>,
    pub raw_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSnapshot {
    pub status: LiveStatus,
    pub status_message: Option<String>,
    pub config: LiveConfig,
    pub items: Vec<LiveMonitorItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSuggestionResult {
    pub text: String,
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
    pub theme: Option<ThemeMode>,
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
    pub auto_period: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translation: Option<TranslationSettings>,
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
pub enum ThemeMode {
    Dark,
    Light,
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

#[derive(Debug, Serialize)]
struct ConnectOpenbliveArgs {
    #[serde(rename = "identityCode")]
    identity_code: String,
}

#[derive(Debug, Serialize)]
struct LiveSuggestionArgs {
    #[serde(rename = "itemId")]
    item_id: String,
    mode: String,
    #[serde(rename = "skipReplace")]
    skip_replace: bool,
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

pub async fn set_runtime_volume(volume: f32) -> Result<f32, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "volume": volume }))
        .map_err(|err| err.to_string())?;
    let value = invoke("set_runtime_volume", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn get_live_state() -> Result<LiveSnapshot, String> {
    let value = invoke("get_live_state", JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn set_live_config_patch(patch: LiveConfigPatch) -> Result<LiveSnapshot, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "patch": patch }))
        .map_err(|err| err.to_string())?;
    let value = invoke("set_live_config_patch", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn connect_openblive(identity_code: String) -> Result<LiveSnapshot, String> {
    let args = serde_wasm_bindgen::to_value(&ConnectOpenbliveArgs { identity_code })
        .map_err(|err| err.to_string())?;
    let value = invoke("connect_openblive", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn disconnect_openblive() -> Result<LiveSnapshot, String> {
    let value = invoke("disconnect_openblive", JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn send_live_suggestion(
    item_id: String,
    mode: String,
    skip_replace: bool,
) -> Result<LiveSuggestionResult, String> {
    let args = serde_wasm_bindgen::to_value(&LiveSuggestionArgs {
        item_id,
        mode,
        skip_replace,
    })
    .map_err(|err| err.to_string())?;

    let value = invoke("send_live_suggestion", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn exit_app() -> Result<CommandResult, String> {
    command_result("exit_app", JsValue::NULL).await
}

pub async fn mock_live_message(kind: LiveMessageKind) -> Result<LiveSnapshot, String> {
    let kind_str = match kind {
        LiveMessageKind::Danmu => "danmu",
        LiveMessageKind::Gift => "gift",
        LiveMessageKind::Superchat => "superchat",
        LiveMessageKind::Guard => "guard",
        LiveMessageKind::Like => "like",
        LiveMessageKind::Enter => "enter",
    };
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "kind": kind_str }))
        .map_err(|err| err.to_string())?;
    let value = invoke("mock_live_message", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn show_live_monitor() -> Result<CommandResult, String> {
    command_result("show_live_monitor", JsValue::NULL).await
}

pub async fn clear_live_items() -> Result<LiveSnapshot, String> {
    let value = invoke("clear_live_items", JsValue::NULL)
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

pub async fn translate_text(
    text: String,
    source_lang: String,
    target_lang: String,
) -> Result<String, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({
        "text": text,
        "sourceLang": source_lang,
        "targetLang": target_lang,
    }))
    .map_err(|err| err.to_string())?;
    let value = invoke("translate_text", args)
        .await
        .map_err(stringify_js_error)?;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn load_model(choice_id: String) -> Result<LoadStartResult, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({ "choiceId": choice_id }))
        .map_err(|err| err.to_string())?;
    let value = invoke("load_model", args)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn cancel_model_load() -> Result<CommandResult, String> {
    command_result("cancel_model_load", JsValue::NULL).await
}

pub async fn cancel_generation(item_id: String) -> Result<CommandResult, String> {
    let args =
        serde_wasm_bindgen::to_value(&ItemArgs { item_id }).map_err(|err| err.to_string())?;
    command_result("cancel_generation", args).await
}

pub async fn play_audio(item_id: String) -> Result<(), String> {
    let args =
        serde_wasm_bindgen::to_value(&ItemArgs { item_id }).map_err(|err| err.to_string())?;
    invoke("play_audio", args)
        .await
        .map_err(stringify_js_error)
        .map(|_| ())
}

pub async fn stop_audio() -> Result<(), String> {
    invoke("stop_audio", JsValue::NULL)
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
    let value = listen(event, &closure).await.map_err(stringify_js_error)?;

    if value.is_undefined() || value.is_function() {
        closure.forget();
        Ok(())
    } else {
        Err(stringify_js_error(value))
    }
}

pub fn decode_app_event<T>(value: JsValue) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let payload = js_sys::Reflect::get(&value, &JsValue::from_str("payload"))
        .ok()
        .filter(|payload| !payload.is_undefined())
        .unwrap_or(value);

    serde_wasm_bindgen::from_value(payload).map_err(|err| err.to_string())
}

pub fn current_window_label() -> String {
    current_window_label_js()
}

async fn browse_path(command: &str) -> Result<Option<String>, String> {
    let value = invoke(command, JsValue::NULL)
        .await
        .map_err(stringify_js_error)?;

    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

async fn command_result(command: &str, args: JsValue) -> Result<CommandResult, String> {
    let value = invoke(command, args).await.map_err(stringify_js_error)?;

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
