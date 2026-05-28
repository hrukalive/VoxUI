use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::generation_queue::HistoryItem;

fn deserialize_double_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            streaming: false,
            stream_consolidate_n: 10,
            retry_badcase: true,
            retry_badcase_max_times: 3,
            retry_badcase_ratio_threshold: 6.0,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
        }
    }
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

impl LiveMessageKind {
    pub fn is_paid(self) -> bool {
        matches!(
            self,
            LiveMessageKind::Gift | LiveMessageKind::Superchat | LiveMessageKind::Guard
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplacementRule {
    pub enabled: bool,
    pub from: String,
    pub to: String,
}

impl Default for ReplacementRule {
    fn default() -> Self {
        Self {
            enabled: true,
            from: String::new(),
            to: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
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

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            danmu: "{msg}".to_string(),
            gift_zh: "感谢{mapped_uname}送出的{gift_num}个{gift_name}".to_string(),
            gift_en: "Thank you {mapped_uname} for {gift_num} {gift_name}".to_string(),
            superchat_zh: "感谢{mapped_uname}的醒目留言：{message}".to_string(),
            superchat_en: "Thank you {mapped_uname} for the superchat saying {message}".to_string(),
            guard_zh: "感谢{mapped_uname}开通的{guard_label}".to_string(),
            guard_en: "Thank you {mapped_uname} for joining as {guard_label}".to_string(),
            like_zh: "感谢{mapped_uname}给直播间点赞".to_string(),
            like_en: "Thank you {mapped_uname} for liking the stream".to_string(),
            enter_zh: "欢迎{mapped_uname}进入直播间".to_string(),
            enter_en: "Hi {mapped_uname}, welcome to the stream".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiveConfig {
    pub identity_code: String,
    pub enable_ceve_server_heartbeat: bool,
    pub show_danmu: bool,
    pub show_gifts: bool,
    pub show_superchats: bool,
    pub show_guards: bool,
    pub show_likes: bool,
    pub show_enters: bool,
    pub templates: TemplateConfig,
    pub replacement_rules: Vec<ReplacementRule>,
    pub mapped_unames: BTreeMap<String, String>,
    pub original_unames: BTreeMap<String, String>,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            identity_code: String::new(),
            enable_ceve_server_heartbeat: false,
            show_danmu: true,
            show_gifts: true,
            show_superchats: true,
            show_guards: true,
            show_likes: false,
            show_enters: true,
            templates: TemplateConfig::default(),
            replacement_rules: vec![
                ReplacementRule {
                    enabled: true,
                    from: "我的".to_string(),
                    to: "你的".to_string(),
                },
                ReplacementRule {
                    enabled: true,
                    from: "我".to_string(),
                    to: "你".to_string(),
                },
                ReplacementRule {
                    enabled: true,
                    from: "I".to_string(),
                    to: "you".to_string(),
                },
                ReplacementRule {
                    enabled: true,
                    from: "me".to_string(),
                    to: "you".to_string(),
                },
                ReplacementRule {
                    enabled: true,
                    from: "my".to_string(),
                    to: "your".to_string(),
                },
            ],
            mapped_unames: BTreeMap::new(),
            original_unames: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub model_root: Option<PathBuf>,
    pub selected_model_id: Option<String>,
    pub language: LanguageMode,
    pub theme: ThemeMode,
    pub backend: BackendKind,
    pub audio_host: Option<String>,
    pub audio_device: Option<String>,
    pub volume: f32,
    pub max_input_chars: usize,
    pub generation: GenerationSettings,
    pub live: LiveConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_root: None,
            selected_model_id: None,
            language: LanguageMode::System,
            theme: ThemeMode::Dark,
            backend: default_backend(),
            audio_host: None,
            audio_device: None,
            volume: 0.8,
            max_input_chars: 280,
            generation: GenerationSettings::default(),
            live: LiveConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn normalize_for_sidecar(&mut self, capabilities: SidecarCapabilities) {
        if !capabilities.cuda_available && self.backend == BackendKind::Cuda {
            self.backend = BackendKind::Cpu;
        }
    }
}

fn default_backend() -> BackendKind {
    BackendKind::Cpu
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarCapabilities {
    pub cuda_available: bool,
    pub default_backend: BackendKind,
}

impl Default for SidecarCapabilities {
    fn default() -> Self {
        Self {
            cuda_available: false,
            default_backend: BackendKind::Cpu,
        }
    }
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
pub struct ConfigPatch {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_double_option"
    )]
    pub model_root: Option<Option<PathBuf>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_double_option"
    )]
    pub selected_model_id: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<LanguageMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<ThemeMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendKind>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_double_option"
    )]
    pub audio_host: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_double_option"
    )]
    pub audio_device: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_chars: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationSettings>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadUiState {
    Idle,
    Loading,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
    pub model_dir: PathBuf,
    pub model_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub model_bytes: u64,
    pub lora_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioHostDto {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDeviceDto {
    pub name: String,
    pub host_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStateDto {
    pub hosts: Vec<AudioHostDto>,
    pub devices: Vec<AudioDeviceDto>,
    pub default_host: Option<String>,
    pub default_devices: Vec<AudioDeviceDto>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub model_id: String,
    pub backend: BackendKind,
    pub generation: GenerationSettings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub system_language: LanguageMode,
    pub cuda_available: bool,
    pub models: Vec<ModelChoice>,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub load_state: LoadUiState,
    pub history: Vec<HistoryItem>,
}
