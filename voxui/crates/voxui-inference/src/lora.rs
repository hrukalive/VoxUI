use std::collections::HashMap;
use std::path::Path;
use candle_core::{Device, Tensor};
use crate::model_loader::GgufModelLoader;
use anyhow::Result;

pub struct LoraAdapter {
    /// Maps "layers.{i}.self_attn.q_proj" -> (lora_A, lora_B)
    pub layers: HashMap<String, (Tensor, Tensor)>,
    pub alpha: f32,
    pub rank: u32,
}

impl LoraAdapter {
    pub fn load(loader: &GgufModelLoader) -> Result<Self> {
        let metadata = loader.metadata();
        let rank = metadata.get("voxcpm.lora.rank")
            .and_then(|v| v.as_u32())
            .unwrap_or(32);
        let alpha = metadata.get("voxcpm.lora.alpha")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as f32;

        let names = loader.tensor_names();

        let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut b_tensors: HashMap<String, Tensor> = HashMap::new();

        for name in &names {
            if name.ends_with(".lora_A") {
                let base = name.trim_end_matches(".lora_A").to_string();
                a_tensors.insert(base, loader.load_tensor_optimal(name)?);
            } else if name.ends_with(".lora_B") {
                let base = name.trim_end_matches(".lora_B").to_string();
                b_tensors.insert(base, loader.load_tensor_optimal(name)?);
            }
        }

        let mut layers = HashMap::new();
        for (base, a) in a_tensors {
            if let Some(b) = b_tensors.remove(&base) {
                layers.insert(base, (a, b));
            }
        }

        Ok(Self { layers, alpha, rank })
    }

    /// Load all LoRA adapters from a directory containing lora_*.gguf files.
    pub fn load_from_dir(dir: &Path, device: &Device) -> Result<Self> {
        let mut all_layers = HashMap::new();
        let mut rank = 32u32;
        let mut alpha = 32.0f32;

        for entry in std::fs::read_dir(dir)?.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "gguf").unwrap_or(false)
               && path.file_stem().map(|s| s.to_string_lossy().starts_with("lora_")).unwrap_or(false)
            {
                let loader = GgufModelLoader::new(&path, device.clone())?;
                let meta = loader.metadata();

                if let Some(r) = meta.get("voxcpm.lora.rank").and_then(|v| v.as_u32()) {
                    rank = r;
                }
                if let Some(a) = meta.get("voxcpm.lora.alpha").and_then(|v| v.as_u32()) {
                    alpha = a as f32;
                }

                let names = loader.tensor_names();
                for name in &names {
                    if name.ends_with(".lora_A") {
                        let base = name.trim_end_matches(".lora_A").to_string();
                        let b_name = format!("{}.lora_B", base);
                        if loader.has_tensor(&b_name) {
                            let a = loader.load_tensor_optimal(name)?;
                            let b = loader.load_tensor_optimal(&b_name)?;
                            all_layers.insert(base, (a, b));
                        }
                    }
                }
            }
        }

        Ok(Self { layers: all_layers, alpha, rank })
    }

    /// Apply LoRA to a linear layer output.
    /// base_output = x @ weight^T (already computed)
    /// lora_output = base_output + (x @ A^T @ B^T) * (alpha / rank)
    pub fn apply(&self, layer_name: &str, base_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        if let Some((a, b)) = self.layers.get(layer_name) {
            let scaling = (self.alpha / self.rank as f32) as f64;
            // Compute in f32 for precision, cast result to match base_output dtype
            let lora_input = input.to_dtype(candle_core::DType::F32)?;
            let a_f32 = a.to_dtype(candle_core::DType::F32)?;
            let b_f32 = b.to_dtype(candle_core::DType::F32)?;
            let lora_out = crate::linear(&lora_input, &a_f32)?;
            let lora_out = crate::linear(&lora_out, &b_f32)?;
            let lora_out = (lora_out * scaling)?;
            let lora_out = lora_out.to_dtype(base_output.dtype())?;
            let result = (base_output + lora_out)?;
            Ok(result)
        } else {
            Ok(base_output.clone())
        }
    }
}
