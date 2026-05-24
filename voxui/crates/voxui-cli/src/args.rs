use std::path::PathBuf;

use anyhow::{bail, Result};

#[derive(clap::Parser)]
#[command(name = "voxui-cli", about = "Interactive VoxCPM TTS CLI")]
pub struct Args {
    /// Path to model directory containing model.gguf, config.json, tokenizer.json
    #[arg(long, value_name = "DIR")]
    pub model: PathBuf,

    /// Path to LoRA adapter .gguf file
    #[arg(long, value_name = "FILE")]
    pub lora: Option<PathBuf>,

    /// Use CUDA device instead of CPU
    #[arg(long)]
    pub cuda: bool,

    /// Stream synthesis and playback (disables badcase retry)
    #[arg(long)]
    pub stream: bool,
}

impl Args {
    /// Validate that model directory exists and contains required files,
    /// and that the optional LoRA file exists.
    pub fn validate(&self) -> Result<()> {
        if !self.model.exists() {
            bail!("model directory not found: {}", self.model.display());
        }
        if !self.model.is_dir() {
            bail!("model path is not a directory: {}", self.model.display());
        }

        let required = ["model.gguf", "config.json", "tokenizer.json"];
        let mut missing = Vec::new();
        for file in &required {
            if !self.model.join(file).exists() {
                missing.push(*file);
            }
        }
        if !missing.is_empty() {
            bail!(
                "model directory {} missing required files: {}",
                self.model.display(),
                missing.join(", ")
            );
        }

        if let Some(ref lora) = self.lora {
            if !lora.exists() {
                bail!("LoRA file not found: {}", lora.display());
            }
            if !lora.extension().and_then(|e| e.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")) {
                bail!("LoRA file must have .gguf extension: {}", lora.display());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_missing_model_dir() {
        let args = Args {
            model: PathBuf::from("__nonexistent_dir__"),
            lora: None,
            cuda: false,
            stream: false,
        };
        let err = args.validate().unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn validate_empty_model_dir_missing_files() {
        let tmp = std::env::temp_dir().join("voxui_cli_test_empty");
        let _ = std::fs::create_dir_all(&tmp);
        let args = Args {
            model: tmp.clone(),
            lora: None,
            cuda: false,
            stream: false,
        };
        let err = args.validate().unwrap_err();
        assert!(err.to_string().contains("missing required files"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_model_dir_with_required_files() {
        let tmp = std::env::temp_dir().join("voxui_cli_test_valid");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("model.gguf"), b"").unwrap();
        std::fs::write(tmp.join("config.json"), b"").unwrap();
        std::fs::write(tmp.join("tokenizer.json"), b"").unwrap();
        let args = Args {
            model: tmp.clone(),
            lora: None,
            cuda: false,
            stream: false,
        };
        assert!(args.validate().is_ok());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_lora_not_found() {
        let tmp = std::env::temp_dir().join("voxui_cli_test_lora");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("model.gguf"), b"").unwrap();
        std::fs::write(tmp.join("config.json"), b"").unwrap();
        std::fs::write(tmp.join("tokenizer.json"), b"").unwrap();
        let args = Args {
            model: tmp.clone(),
            lora: Some(PathBuf::from("__nonexistent_lora__.gguf")),
            cuda: false,
            stream: false,
        };
        let err = args.validate().unwrap_err();
        assert!(err.to_string().contains("not found"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_lora_wrong_extension() {
        let tmp = std::env::temp_dir().join("voxui_cli_test_lora_ext");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("model.gguf"), b"").unwrap();
        std::fs::write(tmp.join("config.json"), b"").unwrap();
        std::fs::write(tmp.join("tokenizer.json"), b"").unwrap();
        let lora_path = tmp.join("adapter.bin");
        std::fs::write(&lora_path, b"").unwrap();
        let args = Args {
            model: tmp.clone(),
            lora: Some(lora_path),
            cuda: false,
            stream: false,
        };
        let err = args.validate().unwrap_err();
        assert!(err.to_string().contains(".gguf extension"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
