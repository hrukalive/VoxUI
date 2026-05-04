use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::manifest::{BundleManifest, ModelVariant};
use crate::model_loader::GgufModelLoader;

pub struct LoraAdapter {
    /// Fully-qualified module name -> (lora_A, lora_B).
    pub layers: HashMap<String, (Tensor, Tensor)>,
    pub alpha: f32,
    pub rank: usize,
}

#[derive(Debug, Deserialize)]
struct LoraManifest {
    schema_version: u32,
    architecture: String,
    variant: ModelVariant,
    rank: usize,
    alpha: f32,
    #[serde(default)]
    target_modules: LoraTargetModules,
    components: HashMap<String, String>,
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

    pub fn load_from_dir(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_from_dir_inner(dir, device, None)
    }

    pub fn load_from_dir_for_model(
        dir: &Path,
        device: &Device,
        model: &BundleManifest,
    ) -> Result<Self> {
        Self::load_from_dir_inner(dir, device, Some(model))
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

    fn load_from_dir_inner(
        dir: &Path,
        device: &Device,
        model: Option<&BundleManifest>,
    ) -> Result<Self> {
        let manifest_path = dir.join("lora_manifest.json");
        if manifest_path.exists() {
            let text = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?;
            let manifest: LoraManifest = serde_json::from_str(&text)
                .with_context(|| format!("parse {}", manifest_path.display()))?;
            manifest.validate(model)?;

            let mut adapter = Self {
                layers: HashMap::new(),
                alpha: manifest.alpha,
                rank: manifest.rank,
            };
            for file in manifest.components.values() {
                let path = dir.join(file);
                if !path.exists() {
                    bail!("missing LoRA component {}", path.display());
                }
                let loader = GgufModelLoader::new(&path, device.clone())?;
                adapter.load_component(&loader, Some(&manifest.target_modules))?;
            }
            adapter.validate_non_empty()?;
            return Ok(adapter);
        }

        let mut adapter = Self {
            layers: HashMap::new(),
            alpha: 32.0,
            rank: 32,
        };
        for entry in std::fs::read_dir(dir)?.flatten() {
            let path = entry.path();
            let is_lora_gguf = path.extension().and_then(|v| v.to_str()) == Some("gguf")
                && path
                    .file_stem()
                    .map(|s| s.to_string_lossy().starts_with("lora_"))
                    .unwrap_or(false);
            if is_lora_gguf {
                let loader = GgufModelLoader::new(&path, device.clone())?;
                if let Some(rank) = loader.metadata().get("voxcpm.lora.rank").and_then(|v| v.as_u32()) {
                    adapter.rank = rank as usize;
                }
                if let Some(alpha) = loader.metadata().get("voxcpm.lora.alpha").and_then(|v| v.as_f32()) {
                    adapter.alpha = alpha;
                }
                adapter.load_component(&loader, None)?;
            }
        }
        adapter.validate_non_empty()?;
        Ok(adapter)
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
                    bail!("LoRA tensor target `{base}` is not enabled by lora_manifest.json");
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

impl LoraManifest {
    fn validate(&self, model: Option<&BundleManifest>) -> Result<()> {
        if self.schema_version != 1 {
            bail!("unsupported LoRA manifest schema {}", self.schema_version);
        }
        if self.rank == 0 {
            bail!("LoRA rank must be positive");
        }
        if self.alpha <= 0.0 {
            bail!("LoRA alpha must be positive");
        }
        if let Some(model) = model {
            if self.architecture != model.architecture {
                bail!(
                    "LoRA architecture `{}` does not match model `{}`",
                    self.architecture,
                    model.architecture
                );
            }
            if self.variant != model.variant {
                bail!("LoRA variant {:?} does not match model {:?}", self.variant, model.variant);
            }
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
        let allowed = self.projections.iter().map(String::as_str).collect::<HashSet<_>>();
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
