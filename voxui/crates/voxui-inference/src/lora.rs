use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::manifest::{ModelConfig, ModelVariant};
use crate::model_loader::GgufModelLoader;

pub struct LoraAdapter {
    /// Fully-qualified module name -> (lora_A, lora_B).
    pub layers: HashMap<String, (Tensor, Tensor)>,
    pub alpha: f32,
    pub rank: usize,
}

#[derive(Debug, Default, Deserialize)]
struct LoraTargetModules {
    #[serde(default)]
    lm: Vec<String>,
    #[serde(default)]
    dit: Vec<String>,
    #[serde(default)]
    projections: Vec<String>,
}

#[derive(Debug)]
struct LoraMetadata {
    architecture: String,
    variant: ModelVariant,
    name: String,
    rank: usize,
    alpha: f32,
    target_modules: LoraTargetModules,
}

impl LoraAdapter {
    pub fn load(loader: &GgufModelLoader) -> Result<Self> {
        let rank = loader
            .metadata()
            .get("voxcpm.lora.rank")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        let alpha = loader
            .metadata()
            .get("voxcpm.lora.alpha")
            .and_then(|v| v.as_f32())
            .unwrap_or(rank as f32);
        let mut adapter = Self {
            layers: HashMap::new(),
            alpha,
            rank,
        };
        adapter.load_component(loader, None)?;
        adapter.validate_non_empty()?;
        Ok(adapter)
    }

    pub fn load_file_for_model(
        path: &Path,
        device: &Device,
        model: &ModelConfig,
    ) -> Result<Self> {
        if path.extension().and_then(|v| v.to_str()) != Some("gguf") {
            bail!("LoRA path must be a .gguf file: {}", path.display());
        }
        let loader = GgufModelLoader::new(path, device.clone())?;
        let metadata = LoraMetadata::from_loader(&loader)?;
        metadata.validate(model)?;
        let mut adapter = Self {
            layers: HashMap::new(),
            alpha: metadata.alpha,
            rank: metadata.rank,
        };
        adapter.load_component(&loader, Some(&metadata.target_modules))?;
        adapter.validate_non_empty()?;
        log::debug!("loaded LoRA adapter `{}` from {}", metadata.name, path.display());
        Ok(adapter)
    }

    pub fn apply_raw(
        base_output: &Tensor,
        input: &Tensor,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        rank: usize,
    ) -> Result<Tensor> {
        if rank == 0 {
            bail!("LoRA rank must be positive");
        }
        let scaling = (alpha / rank as f32) as f64;
        let input = input.to_dtype(DType::F32)?;
        let a = a.to_dtype(DType::F32)?;
        let b = b.to_dtype(DType::F32)?;
        let delta = crate::linear(&crate::linear(&input, &a)?, &b)?;
        let delta = (delta * scaling)?.to_dtype(base_output.dtype())?;
        (base_output + delta).map_err(Into::into)
    }

    pub fn apply(&self, layer_name: &str, base_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        if let Some((a, b)) = self.layers.get(layer_name) {
            Self::apply_raw(base_output, input, a, b, self.alpha, self.rank)
        } else {
            Ok(base_output.clone())
        }
    }

    fn load_component(
        &mut self,
        loader: &GgufModelLoader,
        targets: Option<&LoraTargetModules>,
    ) -> Result<()> {
        let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut b_tensors: HashMap<String, Tensor> = HashMap::new();
        for name in loader.tensor_names() {
            if let Some(base) = name.strip_suffix(".lora_A") {
                a_tensors.insert(base.to_string(), loader.load_tensor_optimal(name)?);
            } else if let Some(base) = name.strip_suffix(".lora_B") {
                b_tensors.insert(base.to_string(), loader.load_tensor_optimal(name)?);
            }
        }

        for (base, a) in a_tensors {
            if let Some(targets) = targets {
                if !targets.allows(&base) {
                    bail!("LoRA tensor target `{base}` is not enabled by LoRA metadata");
                }
            }
            let b = b_tensors
                .remove(&base)
                .ok_or_else(|| anyhow::anyhow!("missing `{base}.lora_B`"))?;
            validate_lora_shapes(&base, &a, &b, self.rank)?;
            if self.layers.insert(base.clone(), (a, b)).is_some() {
                bail!("duplicate LoRA target `{base}`");
            }
        }
        if !b_tensors.is_empty() {
            let mut dangling = b_tensors.keys().cloned().collect::<Vec<_>>();
            dangling.sort();
            bail!("missing lora_A tensors for {:?}", dangling);
        }
        Ok(())
    }

    fn validate_non_empty(&self) -> Result<()> {
        if self.rank == 0 {
            bail!("LoRA rank must be positive");
        }
        if self.alpha <= 0.0 {
            bail!("LoRA alpha must be positive");
        }
        if self.layers.is_empty() {
            bail!("LoRA adapter contains no tensor pairs");
        }
        Ok(())
    }
}

impl LoraMetadata {
    fn from_loader(loader: &GgufModelLoader) -> Result<Self> {
        let metadata = loader.metadata();
        let kind = metadata
            .get("voxcpm.kind")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if kind != "lora" {
            bail!("LoRA GGUF voxcpm.kind must be `lora`, got `{kind}`");
        }
        let architecture = metadata
            .get("voxcpm.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("LoRA GGUF missing voxcpm.architecture"))?
            .to_string();
        let variant = match metadata
            .get("voxcpm.variant")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("LoRA GGUF missing voxcpm.variant"))?
        {
            "0.5" => ModelVariant::VoxCpm05,
            "1.5" => ModelVariant::VoxCpm15,
            "2.0" => ModelVariant::VoxCpm2,
            other => bail!("unsupported LoRA variant `{other}`"),
        };
        let name = metadata
            .get("voxcpm.lora.name")
            .and_then(|v| v.as_str())
            .unwrap_or("adapter")
            .to_string();
        let rank = metadata
            .get("voxcpm.lora.rank")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize;
        let alpha = metadata
            .get("voxcpm.lora.alpha")
            .and_then(|v| v.as_f32())
            .unwrap_or(rank as f32);
        let target_modules = metadata
            .get("voxcpm.lora.target_modules")
            .and_then(|v| v.as_str())
            .and_then(|text| serde_json::from_str::<LoraTargetModules>(text).ok())
            .unwrap_or_default();
        Ok(Self {
            architecture,
            variant,
            name,
            rank,
            alpha,
            target_modules,
        })
    }

    fn validate(&self, model: &ModelConfig) -> Result<()> {
        if self.rank == 0 {
            bail!("LoRA rank must be positive");
        }
        if self.alpha <= 0.0 {
            bail!("LoRA alpha must be positive");
        }
        if self.architecture != model.architecture {
            bail!(
                "LoRA architecture `{}` does not match model `{}`",
                self.architecture,
                model.architecture
            );
        }
        if self.variant != model.variant {
            bail!(
                "LoRA variant {:?} does not match model {:?}",
                self.variant,
                model.variant
            );
        }
        Ok(())
    }
}

impl LoraTargetModules {
    fn allows(&self, base: &str) -> bool {
        if base.starts_with("base_lm.") || base.starts_with("residual_lm.") {
            return target_suffix_allowed(base, &self.lm);
        }
        if base.starts_with("feat_decoder.") {
            return target_suffix_allowed(base, &self.dit);
        }
        if self.projections.is_empty() {
            return true;
        }
        let allowed = self
            .projections
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        allowed.contains(base)
    }
}

fn target_suffix_allowed(base: &str, allowed: &[String]) -> bool {
    if allowed.is_empty() {
        return true;
    }
    let Some(last) = base.rsplit('.').next() else {
        return false;
    };
    allowed.iter().any(|target| target == last)
}

fn validate_lora_shapes(base: &str, a: &Tensor, b: &Tensor, rank: usize) -> Result<()> {
    let (a_rank, _a_in) = a
        .dims2()
        .with_context(|| format!("LoRA A tensor `{base}` must be rank-2"))?;
    let (_b_out, b_rank) = b
        .dims2()
        .with_context(|| format!("LoRA B tensor `{base}` must be rank-2"))?;
    if a_rank != rank || b_rank != rank {
        bail!(
            "LoRA target `{base}` rank mismatch: manifest rank {rank}, A {:?}, B {:?}",
            a.dims(),
            b.dims()
        );
    }
    Ok(())
}
