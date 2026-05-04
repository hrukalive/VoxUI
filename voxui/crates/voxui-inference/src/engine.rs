//! Full VoxCPM2 inference pipeline: text -> audio PCM.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::silu;

use crate::audiovae::{AudioVAE, AudioVAEConfig};
use crate::base_lm::{BaseLM, BaseLMConfig};
use crate::dit::{DiT, DiTConfig};
use crate::encoder::LocalEncoder;
use crate::fsq::FSQLayer;
use crate::lora::LoraAdapter;
use crate::model_loader::GgufModelLoader;
use crate::tokenizer::VoxTokenizer;

/// Configuration for the inference engine.
pub struct EngineConfig {
    pub patch_size: usize,      // 4 (2 for VoxCPM-0.5B)
    pub latent_dim: usize,      // 64
    pub max_steps: usize,       // 200
    pub min_steps: usize,       // 5
    pub dit_steps: usize,       // 10 (CFM inference steps)
    pub cfg_value: f64,         // 2.0
    pub sample_rate: u32,       // 48000 / 44100 / 16000
    pub architecture: String,   // "voxcpm" or "voxcpm2"
    pub scalar_quant_dim: usize, // 512 (VoxCPM2) or 256 (VoxCPM/1.5)
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            patch_size: 4,
            latent_dim: 64,
            max_steps: 200,
            min_steps: 5,
            dit_steps: 10,
            cfg_value: 2.0,
            sample_rate: 48000,
            architecture: "voxcpm2".into(),
            scalar_quant_dim: 512,
        }
    }
}

/// Full VoxCPM inference engine (supports VoxCPM-0.5B, VoxCPM1.5, VoxCPM2).
pub struct VoxCPMEngine {
    tokenizer: VoxTokenizer,
    base_lm: BaseLM,
    residual_lm: BaseLM,
    encoder: LocalEncoder,
    fsq: FSQLayer,
    dit: DiT,
    vae: AudioVAE,
    // Projection weights (from projections.gguf)
    lm_to_dit_proj: Tensor,     // [dit_hidden, lm_hidden]
    res_to_dit_proj: Tensor,    // [dit_hidden, lm_hidden]
    enc_to_lm_proj: Tensor,     // [lm_hidden, enc_hidden]
    fusion_concat_proj: Option<Tensor>, // [lm_hidden, 2*lm_hidden] — v2 only
    stop_proj: Tensor,          // [hidden, hidden]
    stop_head: Tensor,          // [2, hidden]
    // Optional bias tensors (v1 models have these)
    lm_to_dit_proj_bias: Option<Tensor>,
    res_to_dit_proj_bias: Option<Tensor>,
    enc_to_lm_proj_bias: Option<Tensor>,
    stop_proj_bias: Option<Tensor>,
    // Optional LoRA
    lora: Option<LoraAdapter>,
    device: Device,
    config: EngineConfig,
}

// Use shared linear from crate root
use crate::linear;

/// Linear with error context showing dimensions on failure.
fn linear_ctx(x: &Tensor, weight: &Tensor, ctx: &str) -> Result<Tensor> {
    linear(x, weight)
        .with_context(|| format!("{}: lhs {:?}, weight {:?}", ctx, x.dims(), weight.dims()))
}

impl VoxCPMEngine {
    /// Load all model components from a directory of GGUF files.
    ///
    /// Tokenizer is loaded from model_dir (expects tokenizer.json in same folder).
    /// Reads configuration from GGUF metadata to support VoxCPM-0.5B, VoxCPM1.5, and VoxCPM2.
    pub fn load(
        model_dir: &Path,
        tokenizer_dir: &Path,
        device: Device,
    ) -> Result<Self> {
        // 1. Load tokenizer — try model_dir first, then tokenizer_dir as fallback
        let tokenizer = if model_dir.join("tokenizer.json").exists() {
            VoxTokenizer::from_dir(model_dir)?
        } else {
            VoxTokenizer::from_dir(tokenizer_dir)
                .with_context(|| format!(
                    "Tokenizer not found in {:?} or {:?}",
                    model_dir, tokenizer_dir
                ))?
        };

        // 2. Load base_lm.gguf — read architecture/config from its metadata
        let base_lm_path = model_dir.join("base_lm.gguf");
        let base_lm_loader =
            GgufModelLoader::new(&base_lm_path, device.clone())
                .with_context(|| format!("Failed to load GGUF file: {}", base_lm_path.display()))?;

        let meta = base_lm_loader.metadata();
        let architecture = meta
            .get("voxcpm.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("voxcpm2")
            .to_string();
        let residual_lm_no_rope = meta
            .get("voxcpm.residual_lm_no_rope")
            .and_then(|v| match v {
                voxui_gguf::MetadataValue::Bool(b) => Some(*b),
                voxui_gguf::MetadataValue::Uint32(v) => Some(*v != 0),
                _ => None,
            })
            .unwrap_or(architecture == "voxcpm2");

        let base_lm_config = Self::read_lm_config(&base_lm_loader, "base_lm", false)?;
        let base_lm = BaseLM::load(&base_lm_loader, base_lm_config, &device)?;

        // 3. Load residual_lm.gguf
        let res_lm_path = model_dir.join("residual_lm.gguf");
        let res_lm_loader =
            GgufModelLoader::new(&res_lm_path, device.clone())
                .with_context(|| format!("Failed to load GGUF file: {}", res_lm_path.display()))?;
        let res_lm_config =
            Self::read_lm_config(&res_lm_loader, "residual_lm", residual_lm_no_rope)?;
        let residual_lm = BaseLM::load(&res_lm_loader, res_lm_config, &device)?;

        // 3b. Load encoder.gguf (local encoder for predicted features)
        let enc_path = model_dir.join("encoder.gguf");
        let enc_loader = GgufModelLoader::new(&enc_path, device.clone())
            .with_context(|| format!("Failed to load GGUF file: {}", enc_path.display()))?;
        let mut enc_config = Self::read_lm_config(&enc_loader, "encoder", false)?;
        enc_config.is_causal = false; // encoder uses non-causal (full) attention
        enc_config.prefix = "encoder.encoder".to_string(); // tensor prefix is encoder.encoder.layers.{i}
        let encoder = LocalEncoder::load(&enc_loader, enc_config, &device)?;

        // 4. Load projections.gguf (contains FSQ + projection weights)
        let proj_path = model_dir.join("projections.gguf");
        let proj_loader =
            GgufModelLoader::new(&proj_path, device.clone())
                .with_context(|| format!("Failed to load GGUF file: {}", proj_path.display()))?;
        let proj_meta = proj_loader.metadata();

        let patch_size = proj_meta
            .get("voxcpm.patch_size")
            .or_else(|| proj_meta.get("voxcpm.projections.patch_size"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4) as usize;

        let scalar_quant_dim = proj_meta
            .get("voxcpm.scalar_quant_latent_dim")
            .or_else(|| proj_meta.get("voxcpm.projections.scalar_quantization_latent_dim"))
            .and_then(|v| v.as_u32())
            .unwrap_or(if architecture == "voxcpm2" { 512 } else { 256 }) as usize;

        let fsq = FSQLayer::load(&proj_loader, scalar_quant_dim, 9.0)?;

        let lm_to_dit_proj = proj_loader.load_tensor_optimal("lm_to_dit_proj.weight")
            .context("Expected tensor 'lm_to_dit_proj.weight' in projections.gguf")?;
        let res_to_dit_proj = proj_loader.load_tensor_optimal("res_to_dit_proj.weight")
            .context("Expected tensor 'res_to_dit_proj.weight' in projections.gguf")?;
        let enc_to_lm_proj = proj_loader.load_tensor_optimal("enc_to_lm_proj.weight")
            .context("Expected tensor 'enc_to_lm_proj.weight' in projections.gguf")?;
        let fusion_concat_proj = if proj_loader.has_tensor("fusion_concat_proj.weight") {
            Some(proj_loader.load_tensor_optimal("fusion_concat_proj.weight")?)
        } else {
            None // VoxCPM v1 doesn't have this
        };
        let stop_proj = proj_loader.load_tensor_optimal("stop_proj.weight")
            .context("Expected tensor 'stop_proj.weight' in projections.gguf")?;
        let stop_head = proj_loader.load_tensor_optimal("stop_head.weight")
            .context("Expected tensor 'stop_head.weight' in projections.gguf")?;

        // Optional bias tensors (v1 models have these)
        let load_optional_bias = |name: &str| -> Result<Option<Tensor>> {
            if proj_loader.has_tensor(name) {
                Ok(Some(proj_loader.load_tensor_optimal(name)?))
            } else {
                Ok(None)
            }
        };
        let lm_to_dit_proj_bias = load_optional_bias("lm_to_dit_proj.bias")?;
        let res_to_dit_proj_bias = load_optional_bias("res_to_dit_proj.bias")?;
        let enc_to_lm_proj_bias = load_optional_bias("enc_to_lm_proj.bias")?;
        let stop_proj_bias = load_optional_bias("stop_proj.bias")?;

        // 5. Load dit.gguf — read config from metadata
        let dit_path = model_dir.join("dit.gguf");
        let dit_loader = GgufModelLoader::new(&dit_path, device.clone())
            .with_context(|| format!("Failed to load GGUF file: {}", dit_path.display()))?;
        let dit_config = Self::read_dit_config(&dit_loader)?;
        let dit = DiT::load(&dit_loader, dit_config, &device)?;

        // 6. Load audiovae.gguf (required — all VoxCPM variants have AudioVAE)
        let vae_path = model_dir.join("audiovae.gguf");
        let vae_loader = GgufModelLoader::new(&vae_path, device.clone())
            .with_context(|| format!("Failed to load GGUF file: {}", vae_path.display()))?;
        let vae_meta = vae_loader.metadata();
        // Default sample rate depends on architecture
        let default_sr = if architecture == "voxcpm2" { 48000 } else { 16000 };
        let sample_rate = vae_meta
            .get("voxcpm.audiovae.out_sample_rate")
            .or_else(|| vae_meta.get("voxcpm.audiovae.sample_rate"))
            .and_then(|v| v.as_u32())
            .unwrap_or(default_sr);
        // Detect decoder_dim from pointwise conv weight shape (model.1.weight_v: [decoder_dim, latent_dim, 1])
        let decoder_dim = vae_meta
            .get("voxcpm.audiovae.decoder_dim")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                // Infer from model.1.weight_v first dim
                vae_loader.tensor_info("audiovae.decoder.model.1.weight_v")
                    .map(|info| info.shape[0] as usize)
                    .unwrap_or(2048)
            });
        // SR conditioning: only present in V2 (VoxCPM2)
        let has_sr = vae_loader.has_tensor("audiovae.decoder.sr_cond_model.2.scale_embed.weight");
        let sr_idx = if has_sr { Some(3) } else { None };
        let vae_config = AudioVAEConfig {
            sample_rate,
            decoder_dim,
            sr_idx,
            decoder_rates: vec![], // auto-detect from tensor shapes
            ..AudioVAEConfig::default()
        };
        let vae = AudioVAE::load(&vae_loader, vae_config)?;

        let config = EngineConfig {
            patch_size,
            sample_rate,
            architecture,
            scalar_quant_dim,
            ..EngineConfig::default()
        };

        Ok(Self {
            tokenizer,
            base_lm,
            residual_lm,
            encoder,
            fsq,
            dit,
            vae,
            lm_to_dit_proj,
            res_to_dit_proj,
            enc_to_lm_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            lm_to_dit_proj_bias,
            res_to_dit_proj_bias,
            enc_to_lm_proj_bias,
            stop_proj_bias,
            lora: None,
            device,
            config,
        })
    }

    /// Read DiT config from GGUF metadata, falling back to defaults.
    fn read_dit_config(loader: &GgufModelLoader) -> Result<DiTConfig> {
        let meta = loader.metadata();
        let get_u32 = |key: &str, default: u32| -> u32 {
            meta.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            meta.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
        };

        let hidden_dim = get_u32("voxcpm.dit.hidden_dim", 1024) as usize;
        let num_heads = get_u32("voxcpm.dit.num_heads", 16) as usize;
        let num_kv_heads = get_u32("voxcpm.dit.num_kv_heads", 2) as usize;
        // head_dim: try kv_channels first, fallback to hidden_dim / num_heads
        let head_dim = meta.get("voxcpm.dit.kv_channels")
            .or_else(|| meta.get("voxcpm.dit.head_dim"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or_else(|| if num_heads > 0 { hidden_dim / num_heads } else { 128 });

        Ok(DiTConfig {
            hidden_dim,
            num_layers: get_u32("voxcpm.dit.num_layers", 12) as usize,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_dim: get_u32("voxcpm.dit.ffn_dim", 4096) as usize,
            rms_norm_eps: get_f32("voxcpm.dit.rms_norm_eps", 1e-5) as f64,
            scale_depth: get_f32("voxcpm.dit.scale_depth", 1.4) as f64,
            cfg_value: get_f32("voxcpm.dit.cfg_value", 2.0) as f64,
            n_steps: get_u32("voxcpm.dit.n_steps", 10) as usize,
            sway_coef: get_f32("voxcpm.dit.sway_coef", 1.0) as f64,
            latent_dim: get_u32("voxcpm.dit.latent_dim", 64) as usize,
        })
    }

    /// Get the output sample rate for this model.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the patch size for this model.
    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Get the architecture ID ("voxcpm" or "voxcpm2").
    pub fn architecture(&self) -> &str {
        &self.config.architecture
    }

    /// Read LM config from GGUF metadata, falling back to defaults.
    fn read_lm_config(
        loader: &GgufModelLoader,
        prefix: &str,
        no_rope: bool,
    ) -> Result<BaseLMConfig> {
        let meta = loader.metadata();
        let get_u32 = |key: &str, default: u32| -> u32 {
            meta.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            meta.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
        };

        let hidden_size = meta.get(&format!("voxcpm.{prefix}.hidden_size"))
            .or_else(|| meta.get(&format!("voxcpm.{prefix}.hidden_dim")))
            .and_then(|v| v.as_u32())
            .unwrap_or(2048) as usize;
        // Try num_layers first (residual_lm specific), then num_hidden_layers (standard)
        let num_layers = meta.get(&format!("voxcpm.{prefix}.num_layers"))
            .and_then(|v| v.as_u32())
            .or_else(|| meta.get(&format!("voxcpm.{prefix}.num_hidden_layers")).and_then(|v| v.as_u32()))
            .unwrap_or(28) as usize;
        let num_heads = meta.get(&format!("voxcpm.{prefix}.num_attention_heads"))
            .or_else(|| meta.get(&format!("voxcpm.{prefix}.num_heads")))
            .and_then(|v| v.as_u32())
            .unwrap_or(16) as usize;
        let num_kv_heads = get_u32(&format!("voxcpm.{prefix}.num_key_value_heads"), 2) as usize;
        // head_dim: try kv_channels from metadata, fallback to hidden_size / num_heads
        let head_dim = meta.get(&format!("voxcpm.{prefix}.kv_channels"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or_else(|| if num_heads > 0 { hidden_size / num_heads } else { 128 });
        let intermediate_size = meta.get(&format!("voxcpm.{prefix}.intermediate_size"))
            .or_else(|| meta.get(&format!("voxcpm.{prefix}.ffn_dim")))
            .and_then(|v| v.as_u32())
            .unwrap_or(6144) as usize;
        let rms_norm_eps = get_f32(&format!("voxcpm.{prefix}.rms_norm_eps"), 1e-5) as f64;
        let rope_theta = get_f32(&format!("voxcpm.{prefix}.rope_theta"), 10000.0) as f64;
        let vocab_size = get_u32(&format!("voxcpm.{prefix}.vocab_size"), 73448) as usize;

        // LongRope factors (optional)
        let rope_factors = if let Some(voxui_gguf::MetadataValue::ArrayFloat32(factors)) =
            meta.get(&format!("voxcpm.{prefix}.rope_factors"))
        {
            factors.clone()
        } else {
            vec![1.0; head_dim / 2]
        };

        Ok(BaseLMConfig {
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            rope_factors,
            vocab_size,
            max_position: 4096,
            prefix: prefix.to_string(),
            no_rope,
            is_causal: true,
        })
    }

    /// Set the number of DiT inference steps.
    pub fn set_dit_steps(&mut self, steps: usize) {
        self.config.dit_steps = steps;
    }

    /// Load LoRA adapters from a directory containing lora_*.gguf files.
    pub fn load_lora(&mut self, path: &Path) -> Result<()> {
        let adapter = LoraAdapter::load_from_dir(path, &self.device)?;
        self.lora = Some(adapter);
        Ok(())
    }

    /// Unload the current LoRA adapter.
    pub fn unload_lora(&mut self) {
        self.lora = None;
    }

    /// Synthesize text to PCM f32 audio samples.
    ///
    /// `progress` is called with `(current_step, estimated_total_steps)`.
    pub fn synthesize<F: Fn(usize, usize)>(
        &mut self,
        text: &str,
        dit_steps: usize,
        progress: F,
    ) -> Result<Vec<f32>> {
        if text.is_empty() {
            bail!("text must not be empty");
        }

        let max_steps = self.config.max_steps;
        let min_steps = self.config.min_steps;
        let patch_size = self.config.patch_size;
        let latent_dim = self.config.latent_dim;
        let estimated_steps = (text.chars().count() / 2).max(5).min(max_steps);

        // 1. Tokenize
        let token_ids = self.tokenizer.encode(text)?;
        if token_ids.is_empty() {
            bail!("tokenizer produced no tokens");
        }

        // 2. Reset caches
        self.base_lm.reset_cache();
        self.residual_lm.reset_cache();

        // Determine working dtype (f16 on CUDA, f32 on CPU)
        let working_dtype = if self.device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };

        // 3. Prefill base_lm with text tokens
        let text_embed = self.base_lm.embed(&token_ids)?;
        let enc_out = self.base_lm.forward_embed_with_lora(&text_embed, self.lora.as_ref())?; // [1, T, hidden]
        let seq_len = token_ids.len();

        // Extract last hidden state: [1, 1, hidden]
        let mut lm_hidden = enc_out.narrow(1, seq_len - 1, 1)?;

        // 4. Prefill residual_lm with a dummy step (or zeros)
        // For first step, residual_lm has no prior context.
        // Feed zeros as initial input.
        let hidden_size = lm_hidden.dim(2)?;
        let zero_embed =
            Tensor::zeros((1, 1, hidden_size), working_dtype, &self.device)?;
        let res_out = self.residual_lm.forward_embed_with_lora(&zero_embed, self.lora.as_ref())?;
        let mut res_hidden = res_out.narrow(1, 0, 1)?; // [1, 1, hidden]

        // 5. Initial conditioning for DiT: zeros
        let mut cond =
            Tensor::zeros((1, latent_dim, patch_size), working_dtype, &self.device)?;

        // 6. Autoregressive loop
        let mut patches: Vec<Tensor> = Vec::new();

        let is_v2 = self.config.architecture == "voxcpm2";

        for step in 0..max_steps {
            progress(step, estimated_steps);

            // Apply FSQ bottleneck to lm_hidden: [1, 1, hidden] -> [1, 1, hidden]
            let fsq_hidden = self.fsq.forward(&lm_hidden)?;

            // Project to DiT conditioning
            // fsq_hidden: [1, 1, hidden] -> squeeze to [1, hidden]
            let lm_h = fsq_hidden.squeeze(1)?; // [1, hidden]
            let res_h = res_hidden.squeeze(1)?; // [1, hidden]

            let mut dit_mu_lm = linear_ctx(&lm_h, &self.lm_to_dit_proj, "lm_to_dit_proj")?; // [1, dit_hidden]
            if let Some(ref bias) = self.lm_to_dit_proj_bias {
                dit_mu_lm = dit_mu_lm.broadcast_add(bias)?;
            }
            let mut dit_mu_res = linear_ctx(&res_h, &self.res_to_dit_proj, "res_to_dit_proj")?; // [1, dit_hidden]
            if let Some(ref bias) = self.res_to_dit_proj_bias {
                dit_mu_res = dit_mu_res.broadcast_add(bias)?;
            }
            // v2: concat two projections (2 conditioning tokens)
            // v1: sum two projections (1 conditioning token)
            let dit_mu = if is_v2 {
                Tensor::cat(&[&dit_mu_lm, &dit_mu_res], 1)? // [1, 2*dit_hidden]
            } else {
                (dit_mu_lm + dit_mu_res)? // [1, dit_hidden]
            };

            // Generate one patch via DiT
            let pred_feat =
                self.dit.generate(&dit_mu, &cond, patch_size, dit_steps)?; // [1, 64, patch_size]

            // Encode predicted patch for feeding back via local encoder
            let encoded = self.encoder.encode(&pred_feat)?; // [1, 1, enc_hidden=1024]
            let mut curr_embed = linear_ctx(&encoded, &self.enc_to_lm_proj, "enc_to_lm_proj")?; // [1, 1, lm_hidden]
            if let Some(ref bias) = self.enc_to_lm_proj_bias {
                curr_embed = curr_embed.broadcast_add(bias)?;
            }

            // Update cond for next step
            cond = pred_feat.clone();

            // Stop predictor: silu(lm_h @ stop_proj^T) @ stop_head^T -> [1, 2]
            let mut stop_proj_out = linear_ctx(&lm_h, &self.stop_proj, "stop_proj")?;
            if let Some(ref bias) = self.stop_proj_bias {
                stop_proj_out = stop_proj_out.broadcast_add(bias)?;
            }
            let stop_in = silu(&stop_proj_out)?;
            let stop_logits = linear_ctx(&stop_in, &self.stop_head, "stop_head")?; // [1, 2]

            patches.push(pred_feat);

            // Check stop condition
            if step >= min_steps {
                let stop_data = stop_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                if stop_data[0][1] > stop_data[0][0] {
                    break;
                }
            }

            // Advance base_lm one step with encoded features
            let lm_out =
                self.base_lm.forward_embed_with_lora(&curr_embed, self.lora.as_ref())?; // [1, 1, hidden]
            lm_hidden = lm_out;

            // Advance residual_lm: v2 uses fusion_concat_proj(cat(lm, embed)), v1 uses add
            let lm_h_new = lm_hidden.squeeze(1)?; // [1, hidden]
            let curr_e = curr_embed.squeeze(1)?; // [1, hidden]
            let fused = if is_v2 {
                let fused_input =
                    Tensor::cat(&[&lm_h_new, &curr_e], 1)?; // [1, 2*hidden]
                linear_ctx(&fused_input, self.fusion_concat_proj.as_ref().unwrap(), "fusion_concat_proj")? // [1, hidden]
            } else {
                (lm_h_new + curr_e)? // [1, hidden] — elementwise add
            };
            let fused = fused.unsqueeze(1)?; // [1, 1, hidden]
            let res_out =
                self.residual_lm.forward_embed_with_lora(&fused, self.lora.as_ref())?; // [1, 1, hidden]
            res_hidden = res_out;
        }

        progress(patches.len(), patches.len());

        if patches.is_empty() {
            bail!("no patches generated");
        }

        // 7. Concat patches: [1, 64, N*patch_size]
        let latent = Tensor::cat(
            &patches.iter().collect::<Vec<_>>(),
            2,
        )?;

        // 8. VAE decode: [1, 64, N*patch_size] -> [1, 1, N*patch_size*1920]
        // VAE operates in f32 (weight_norm weights loaded as f32)
        let latent = latent.to_dtype(DType::F32)?;
        let audio = self.vae.decode(&latent)?;

        // 9. Extract PCM samples as Vec<f32>
        let audio = audio.squeeze(0)?.squeeze(0)?; // [samples]
        let samples = audio.to_vec1::<f32>()?;

        log::info!("Synthesize produced {} samples ({:.1}s at {}Hz)", 
            samples.len(), samples.len() as f32 / self.config.sample_rate as f32, self.config.sample_rate);

        Ok(samples)
    }
}
