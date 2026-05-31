use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, Result};

use crate::types::ModelChoice;

pub fn discover_models(root: &Path) -> Result<Vec<ModelChoice>> {
    if !root
        .try_exists()
        .with_context(|| format!("failed to inspect model root {}", root.display()))?
    {
        return Ok(Vec::new());
    }

    let mut model_dirs = Vec::new();
    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read model root {}", root.display()))?
    {
        let entry = entry
            .with_context(|| format!("failed to read entry in model root {}", root.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to read file type for {}", path.display()))?;
        if file_type.is_dir() {
            model_dirs.push(path);
        }
    }
    model_dirs.sort();

    let mut choices = Vec::new();
    for model_dir in model_dirs {
        let model_path = model_dir.join("model.gguf");
        let model_metadata = match model_path.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to read metadata for {}", model_path.display())
                });
            }
        };

        let model_name = model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("model directory name is not UTF-8: {}", model_dir.display()))?
            .to_owned();
        let model_bytes = model_metadata.len();

        choices.push(ModelChoice {
            id: choice_id(root, &model_dir, None)?,
            display_name: model_name.clone(),
            model_dir: model_dir.clone(),
            model_path: model_path.clone(),
            lora_path: None,
            model_bytes,
            lora_bytes: 0,
        });
    }

    Ok(choices)
}

pub fn choice_id(root: &Path, model_dir: &Path, lora_path: Option<&Path>) -> Result<String> {
    let model_id = relative_slash_path(root, model_dir)?;
    if let Some(lora_path) = lora_path {
        let lora_file = lora_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("LoRA path has no UTF-8 file name")?;
        Ok(format!("{model_id}|{lora_file}"))
    } else {
        Ok(model_id)
    }
}

fn is_lora_candidate(path: &Path, file_type: &fs::FileType) -> Result<bool> {
    Ok(file_type.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| {
                format!("LoRA candidate file name is not UTF-8: {}", path.display())
            })?
            != "model.gguf")
}

fn relative_slash_path(root: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(root).with_context(|| {
        format!(
            "failed to make {} relative to {}",
            path.display(),
            root.display()
        )
    })?;
    Ok(relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}
