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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    pub model_dir: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    pub lora_path: Option<String>,
    pub lora_size_bytes: Option<u64>,
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
            if !path.is_dir() || !path.join("model.gguf").is_file() {
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

pub fn scan_model_choices(models_root: &Path) -> Vec<ModelChoice> {
    let mut choices = Vec::new();
    let mut model_dirs = fs::read_dir(models_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let model_path = path.join("model.gguf");
            if !model_path.is_file() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            Some((name, path, model_path))
        })
        .collect::<Vec<_>>();

    model_dirs.sort_by(|left, right| left.0.cmp(&right.0));

    for (model_name, model_dir, model_path) in model_dirs {
        let model_size_bytes = file_size(&model_path);
        choices.push(ModelChoice {
            id: model_name.clone(),
            name: model_name.clone(),
            model_dir: display_path(&model_dir),
            model_path: display_path(&model_path),
            model_size_bytes,
            lora_path: None,
            lora_size_bytes: None,
        });

        let mut loras = fs::read_dir(&model_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().into_owned();
                let is_lora_file = path.is_file()
                    && path.extension().and_then(|value| value.to_str()) == Some("gguf")
                    && file_name.starts_with("lora_");
                if !is_lora_file {
                    return None;
                }
                let stem = path.file_stem()?.to_string_lossy().into_owned();
                Some((file_name, stem, path))
            })
            .collect::<Vec<_>>();
        loras.sort_by(|left, right| left.0.cmp(&right.0));

        for (file_name, lora_name, lora_path) in loras {
            choices.push(ModelChoice {
                id: format!("{model_name}::{file_name}"),
                name: format!("{model_name} | {lora_name}"),
                model_dir: display_path(&model_dir),
                model_path: display_path(&model_path),
                model_size_bytes,
                lora_path: Some(display_path(&lora_path)),
                lora_size_bytes: Some(file_size(&lora_path)),
            });
        }
    }

    choices
}

pub fn scan_lora_entries(model_dir: &Path) -> Vec<LoraEntry> {
    let mut loras = fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_lora_file = path.is_file()
                && path.extension().and_then(|v| v.to_str()) == Some("gguf")
                && name.starts_with("lora_");
            if !is_lora_file {
                return None;
            }
            let display_name = name
                .strip_prefix("lora_")
                .and_then(|v| v.strip_suffix(".gguf"))
                .unwrap_or(&name)
                .to_string();
            Some(LoraEntry {
                name: display_name,
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
        for ancestor in current_dir.ancestors().take(6) {
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

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0)
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
    use std::sync::Mutex;
    use tempfile::tempdir;

    static CURRENT_DIR_LOCK: Mutex<()> = Mutex::new(());

    fn create_model_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("model.gguf"), b"placeholder").unwrap();
        path
    }

    #[test]
    fn scan_model_entries_returns_only_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        create_model_dir(tmp.path(), "voxcpm2-fp16");
        create_model_dir(tmp.path(), "voxcpm05-fp16");
        fs::create_dir_all(tmp.path().join("not-a-model")).unwrap();

        let entries = super::scan_model_entries(tmp.path());

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "voxcpm05-fp16");
        assert_eq!(entries[1].name, "voxcpm2-fp16");
        assert!(entries.iter().all(|entry| entry.path.contains("voxcpm")));
    }

    #[test]
    fn scan_lora_entries_includes_none_and_direct_gguf_files_sorted() {
        let tmp = tempdir().unwrap();
        let model = create_model_dir(tmp.path(), "voxcpm2-fp16");
        fs::write(model.join("lora_b.gguf"), b"placeholder").unwrap();
        fs::write(model.join("lora_a.gguf"), b"placeholder").unwrap();
        fs::create_dir_all(model.join("lora_old_dir")).unwrap();
        fs::write(model.join("not_lora.gguf"), b"placeholder").unwrap();

        let entries = super::scan_lora_entries(&model);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], super::LoraEntry::none());
        assert_eq!(entries[1].name, "a");
        assert_eq!(entries[2].name, "b");
        assert!(entries[1].path.as_ref().unwrap().ends_with("lora_a.gguf"));
    }

    #[test]
    fn scan_model_choices_flattens_base_and_lora_files_sorted() {
        let tmp = tempdir().unwrap();
        let model_b = create_model_dir(tmp.path(), "voxcpm2-q4-lm");
        fs::write(model_b.join("lora_ft2.gguf"), b"placeholder").unwrap();
        fs::write(model_b.join("lora_alpha.gguf"), b"placeholder").unwrap();
        create_model_dir(tmp.path(), "voxcpm05-fp16");
        fs::write(model_b.join("not_lora.gguf"), b"placeholder").unwrap();
        fs::create_dir_all(model_b.join("lora_old_dir.gguf")).unwrap();

        let choices = super::scan_model_choices(tmp.path());
        let names = choices
            .iter()
            .map(|choice| choice.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "voxcpm05-fp16",
                "voxcpm2-q4-lm",
                "voxcpm2-q4-lm | lora_alpha",
                "voxcpm2-q4-lm | lora_ft2",
            ]
        );
        assert!(choices[0].lora_path.is_none());
        assert!(choices[2].lora_path.as_ref().unwrap().ends_with("lora_alpha.gguf"));
    }

    #[test]
    fn model_choice_id_is_relative_to_model_root_and_lora_file() {
        let tmp = tempdir().unwrap();
        let model = create_model_dir(tmp.path(), "voxcpm2-q4-lm");
        fs::write(model.join("lora_ft2.gguf"), b"placeholder").unwrap();

        let choices = super::scan_model_choices(tmp.path());

        assert!(choices.iter().any(|choice| choice.id == "voxcpm2-q4-lm"));
        assert!(choices
            .iter()
            .any(|choice| choice.id == "voxcpm2-q4-lm::lora_ft2.gguf"));
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

    #[test]
    fn discover_models_root_stops_after_six_ancestors() {
        let _guard = CURRENT_DIR_LOCK.lock().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        let tmp = tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();

        let current = tmp
            .path()
            .join("one")
            .join("two")
            .join("three")
            .join("four")
            .join("five")
            .join("six");
        fs::create_dir_all(&current).unwrap();
        std::env::set_current_dir(&current).unwrap();

        let discovered = super::discover_models_root();

        std::env::set_current_dir(original_dir).unwrap();
        assert_eq!(discovered, std::path::PathBuf::from("models"));
    }
}
