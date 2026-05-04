#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use voxui_inference::SynthesisRequest;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoraEntry {
    pub name: String,
    pub path: Option<String>,
}

impl LoraEntry {
    pub fn none() -> Self {
        Self {
            name: "None".to_string(),
            path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisArgs {
    pub index: u32,
    pub text: String,
    pub dit_steps: usize,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
}

impl SynthesisArgs {
    pub fn into_request(self) -> SynthesisRequest {
        SynthesisRequest {
            text: self.text,
            prompt_wav_path: optional_path(self.prompt_wav_path),
            prompt_text: optional_string(self.prompt_text),
            reference_wav_path: optional_path(self.reference_wav_path),
            inference_timesteps: self.dit_steps,
            ..SynthesisRequest::default()
        }
    }
}

pub fn scan_model_entries(models_root: &Path) -> Vec<ModelEntry> {
    let mut entries = fs::read_dir(models_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() || !path.join("manifest.json").is_file() {
                return None;
            }
            Some(ModelEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                path: display_path(&path),
            })
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| left.name.cmp(&right.name));
    entries
}

pub fn scan_lora_entries(model_dir: &Path) -> Vec<LoraEntry> {
    let mut loras = fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("lora_")
                || !path.is_dir()
                || !path.join("lora_manifest.json").is_file()
            {
                return None;
            }
            Some(LoraEntry {
                name,
                path: Some(display_path(&path)),
            })
        })
        .collect::<Vec<_>>();

    loras.sort_by(|left, right| left.name.cmp(&right.name));

    let mut entries = vec![LoraEntry::none()];
    entries.extend(loras);
    entries
}

pub fn discover_models_root() -> PathBuf {
    if let Ok(current_dir) = std::env::current_dir() {
        for ancestor in current_dir.ancestors().take(7) {
            let candidate = ancestor.join("models");
            if candidate.is_dir() {
                return candidate;
            }
        }
    }

    PathBuf::from("models")
}

pub fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn optional_path(value: Option<String>) -> Option<PathBuf> {
    optional_string(value).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use tempfile::tempdir;

    fn create_manifest_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("manifest.json"), "{}").unwrap();
        path
    }

    #[test]
    fn scan_model_entries_returns_only_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        create_manifest_dir(tmp.path(), "voxcpm05-fp16");
        fs::create_dir_all(tmp.path().join("not-a-model")).unwrap();

        let entries = super::scan_model_entries(tmp.path());

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "voxcpm05-fp16");
        assert_eq!(entries[1].name, "voxcpm2-fp16");
        assert!(entries.iter().all(|entry| entry.path.contains("voxcpm")));
    }

    #[test]
    fn scan_lora_entries_includes_none_and_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        let model = create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        let lora_b = model.join("lora_b");
        let lora_a = model.join("lora_a");
        fs::create_dir_all(&lora_b).unwrap();
        fs::create_dir_all(&lora_a).unwrap();
        fs::write(lora_b.join("lora_manifest.json"), "{}").unwrap();
        fs::write(lora_a.join("lora_manifest.json"), "{}").unwrap();
        fs::create_dir_all(model.join("lora_without_manifest")).unwrap();

        let entries = super::scan_lora_entries(&model);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], super::LoraEntry::none());
        assert_eq!(entries[1].name, "lora_a");
        assert_eq!(entries[2].name, "lora_b");
        assert!(entries[1].path.as_ref().unwrap().ends_with("lora_a"));
    }

    #[test]
    fn synthesis_args_builds_native_request_with_prompt_and_reference_paths() {
        let args = super::SynthesisArgs {
            index: 4,
            text: " hello   world ".to_string(),
            dit_steps: 7,
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
        };

        let request = args.into_request();

        assert_eq!(request.text, " hello   world ");
        assert_eq!(request.inference_timesteps, 7);
        assert_eq!(request.prompt_text.as_deref(), Some("prompt text"));
        assert_eq!(
            request.prompt_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/prompt.wav")
        );
        assert_eq!(
            request.reference_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/reference.wav")
        );
    }

    #[test]
    fn empty_optional_strings_do_not_create_paths() {
        let args = super::SynthesisArgs {
            index: 0,
            text: "hello".to_string(),
            dit_steps: 10,
            prompt_wav_path: Some("   ".to_string()),
            prompt_text: Some("   ".to_string()),
            reference_wav_path: Some(String::new()),
        };

        let request = args.into_request();

        assert!(request.prompt_wav_path.is_none());
        assert!(request.prompt_text.is_none());
        assert!(request.reference_wav_path.is_none());
    }
}
