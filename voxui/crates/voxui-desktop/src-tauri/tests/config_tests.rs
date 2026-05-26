use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use voxui_desktop::config::{
    default_config_path, detect_language_from_locale, load_config, save_config, AppConfig,
    BackendKind, LanguageMode,
};

fn unique_temp_dir(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    std::env::temp_dir().join(format!(
        "voxui_desktop_config_tests_{}_{}_{}",
        test_name,
        std::process::id(),
        nanos
    ))
}

#[test]
fn tauri_config_exposes_global_tauri_api() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tauri.conf.json");
    let config_text = fs::read_to_string(config_path).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_text).unwrap();

    assert_eq!(
        config["app"]["withGlobalTauri"],
        serde_json::Value::Bool(true)
    );
}

#[test]
fn config_defaults_to_system_language_and_preferred_backend() {
    let config = AppConfig::default();
    let expected_backend = if cfg!(feature = "cuda") {
        BackendKind::Cuda
    } else {
        BackendKind::Cpu
    };

    assert_eq!(config.language, LanguageMode::System);
    assert_eq!(config.backend, expected_backend);
    assert_eq!(config.volume, 0.8);
    assert_eq!(config.max_input_chars, 280);
    assert_eq!(config.generation.inference_timesteps, 10);
    assert_eq!(config.generation.cfg_value, 2.0);
    assert!(config.generation.streaming);
    assert_eq!(config.generation.stream_consolidate_n, 10);
    assert!(config.generation.retry_badcase);
}

#[test]
fn detects_chinese_for_zh_locale() {
    assert_eq!(
        detect_language_from_locale(Some("zh-CN")),
        LanguageMode::Chinese
    );
    assert_eq!(
        detect_language_from_locale(Some("zh_TW")),
        LanguageMode::Chinese
    );
}

#[test]
fn detects_english_for_non_zh_or_missing_locale() {
    assert_eq!(
        detect_language_from_locale(Some("en-US")),
        LanguageMode::English
    );
    assert_eq!(
        detect_language_from_locale(Some("ja-JP")),
        LanguageMode::English
    );
    assert_eq!(detect_language_from_locale(None), LanguageMode::English);
}

#[test]
fn config_round_trips_as_json() {
    let config = AppConfig {
        model_root: Some(PathBuf::from("D:/Sandbox_Share/VoxUI/models")),
        selected_model_id: Some("voxcpm2-fp16|lora_a1.gguf".to_string()),
        language: LanguageMode::Chinese,
        backend: BackendKind::Cuda,
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        volume: 0.42,
        max_input_chars: 320,
        ..AppConfig::default()
    };

    let encoded = serde_json::to_string_pretty(&config).unwrap();
    let decoded: AppConfig = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.model_root, config.model_root);
    assert_eq!(decoded.selected_model_id, config.selected_model_id);
    assert_eq!(decoded.language, LanguageMode::Chinese);
    assert_eq!(decoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.volume, 0.42);
    assert_eq!(decoded.max_input_chars, 320);
}

#[test]
fn empty_config_json_deserializes_to_defaults() {
    let decoded: AppConfig = serde_json::from_str("{}").unwrap();

    assert_eq!(decoded, AppConfig::default());
}

#[test]
fn partial_generation_json_preserves_values_and_defaults_missing_fields() {
    let decoded: AppConfig =
        serde_json::from_str(r#"{ "generation": { "cfg_value": 3.5 } }"#).unwrap();

    assert_eq!(decoded.generation.cfg_value, 3.5);
    assert_eq!(decoded.generation.inference_timesteps, 10);
    assert!(decoded.generation.streaming);
    assert!(decoded.generation.retry_badcase);
    assert_eq!(decoded.language, LanguageMode::System);
    assert_eq!(decoded.backend, AppConfig::default().backend);
}

#[test]
fn default_config_path_uses_voxui_config_filename() {
    let path = default_config_path(Path::new("D:/Sandbox_Share/VoxUI/config"));

    assert!(path.ends_with("voxui_config.json"));
}

#[test]
fn load_config_returns_defaults_when_file_is_missing() {
    let root = unique_temp_dir("missing");
    let path = root.join("missing").join("voxui_config.json");

    let config = load_config(&path).unwrap();

    assert_eq!(config, AppConfig::default());
}

#[test]
fn save_config_creates_parent_directory_and_load_config_reads_written_json() {
    let root = unique_temp_dir("save_load");
    let path = root.join("nested").join("config").join("voxui_config.json");
    let config = AppConfig {
        model_root: Some(PathBuf::from("D:/Sandbox_Share/VoxUI/models")),
        selected_model_id: Some("voxcpm2-fp16|lora_a1.gguf".to_string()),
        language: LanguageMode::Chinese,
        backend: BackendKind::Cuda,
        volume: 0.55,
        max_input_chars: 360,
        ..AppConfig::default()
    };

    save_config(&path, &config).unwrap();

    assert!(path.exists());
    let written = fs::read_to_string(&path).unwrap();
    assert!(written.contains("\"selected_model_id\""));
    let decoded = load_config(&path).unwrap();

    fs::remove_dir_all(root).unwrap();

    assert_eq!(decoded, config);
}
