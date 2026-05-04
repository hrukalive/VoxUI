//! DiT (Diffusion Transformer) with CFM Euler solver for latent generation.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::silu;

use crate::lora::LoraAdapter;
use crate::model_loader::GgufModelLoader;

/// Configuration for the DiT transformer and CFM solver.
pub struct DiTConfig {
    pub prefix: String,
    pub hidden_dim: usize,   // 1024
    pub num_layers: usize,   // 12
    pub num_heads: usize,    // 16
    pub num_kv_heads: usize, // 2
    pub head_dim: usize,     // 128
    pub ffn_dim: usize,      // 4096
    pub rms_norm_eps: f64,   // 1e-5
    pub scale_depth: f64,    // 1.4
    pub use_mup: bool,
    pub rope_theta: f64,
    pub original_max_position_embeddings: Option<usize>,
    pub rope_short_factors: Vec<f32>,
    pub rope_long_factors: Vec<f32>,
    pub cfg_value: f64,      // 2.0
    pub n_steps: usize,      // 10
    pub sway_coef: f64,      // 1.0
    pub latent_dim: usize,   // 64
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            prefix: "dit.estimator".to_string(),
            hidden_dim: 1024,
            num_layers: 12,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            ffn_dim: 4096,
            rms_norm_eps: 1e-5,
            scale_depth: 1.4,
            use_mup: false,
            rope_theta: 10000.0,
            original_max_position_embeddings: None,
            rope_short_factors: vec![1.0; 64],
            rope_long_factors: vec![1.0; 64],
            cfg_value: 2.0,
            n_steps: 10,
            sway_coef: 1.0,
            latent_dim: 64,
        }
    }
}

/// Weights for projections with optional bias.
struct LinearWeight {
    weight: Tensor,
    bias: Option<Tensor>,
}

/// Weights for a single DiT transformer layer.
struct DiTLayer {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
}

/// DiT model with CFM solver.
pub struct DiT {
    config: DiTConfig,
    // Projection layers (with bias)
    in_proj: LinearWeight,
    cond_proj: LinearWeight,
    out_proj: LinearWeight,
    // Time embedding MLPs
    time_mlp_1: LinearWeight,
    time_mlp_2: LinearWeight,
    delta_time_mlp_1: LinearWeight,
    delta_time_mlp_2: LinearWeight,
    // Transformer
    layers: Vec<DiTLayer>,
    final_norm: Tensor,
    // Precomputed RoPE
    cos_cache: Tensor,
    sin_cache: Tensor,
    device: Device,
}

/// RMS normalization.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let eps_t =
        mean_sq
            .zeros_like()?
            .broadcast_add(&Tensor::new(&[eps as f32], mean_sq.device())?)?;
    let norm = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let out = x.broadcast_mul(&norm)?;
    let weight = weight.to_dtype(DType::F32)?;
    let out = out.broadcast_mul(&weight)?;
    out.to_dtype(dtype).map_err(Into::into)
}

/// Linear: x @ weight^T (+ bias). Handles 2D and 3D inputs.
fn linear(x: &Tensor, w: &LinearWeight) -> Result<Tensor> {
    let out = crate::linear(x, &w.weight)?;
    if let Some(ref bias) = w.bias {
        Ok(out.broadcast_add(bias)?)
    } else {
        Ok(out)
    }
}

/// Linear without bias: x @ weight^T. Handles 2D and 3D inputs.
fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    crate::linear(x, weight)
}

/// Apply rotary position embeddings.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let (_b, _h, _t, hd) = x.dims4()?;
    let half = hd / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?;
    let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&o1, &o2], 3)?
        .to_dtype(orig_dtype)
        .map_err(Into::into)
}

/// Repeat KV heads for GQA.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, t, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_h, n_rep, t, hd))?
        .reshape((b, kv_h * n_rep, t, hd))
        .map_err(Into::into)
}

/// Sinusoidal positional embedding with scale factor.
fn sinusoidal_embedding(t: &Tensor, dim: usize, scale: f64) -> Result<Tensor> {
    let input_dtype = t.dtype();
    let half = dim / 2;
    // emb[i] = exp(i * -log(10000) / (half-1))
    let log10000 = (10000.0f64).ln();
    let emb: Vec<f32> = (0..half)
        .map(|i| (-(i as f64) * log10000 / (half as f64 - 1.0)).exp() as f32)
        .collect();
    let emb = Tensor::new(emb.as_slice(), t.device())?; // [half]

    // t_scaled = scale * t
    let t_f32 = t.to_dtype(DType::F32)?;
    let t_scaled = (t_f32 * scale)?; // [B]

    // outer product: [B, half]
    let angles = t_scaled.unsqueeze(1)?.broadcast_mul(&emb.unsqueeze(0)?)?;

    let sin_emb = angles.sin()?;
    let cos_emb = angles.cos()?;
    let out = Tensor::cat(&[&sin_emb, &cos_emb], 1)?; // [B, dim]
    out.to_dtype(input_dtype).map_err(Into::into)
}

fn get_usize(cfg: &serde_json::Value, keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|key| cfg.get(*key).and_then(|v| v.as_u64()).map(|v| v as usize))
        .unwrap_or(default)
}

fn get_f64(cfg: &serde_json::Value, key: &str, default: f64) -> f64 {
    cfg.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn read_f32_array(value: &serde_json::Value, key: &str, len: usize) -> Vec<f32> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect())
        .filter(|arr: &Vec<f32>| arr.len() == len)
        .unwrap_or_else(|| vec![1.0; len])
}

impl DiT {
    pub fn load_from_manifest(
        loader: &GgufModelLoader,
        manifest: &crate::BundleManifest,
    ) -> Result<Self> {
        let dit = &manifest.dit_config;
        let lm = &manifest.lm_config;
        let hidden_dim = get_usize(dit, &["hidden_dim", "hidden_size"], 1024);
        let num_heads = get_usize(dit, &["num_heads", "num_attention_heads"], 16);
        let head_dim = dit
            .get("kv_channels")
            .or_else(|| lm.get("kv_channels"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_dim / num_heads);
        let num_kv_heads = lm
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_heads);
        let rope_scaling = lm.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
        let cfm = dit.get("cfm_config").unwrap_or(&serde_json::Value::Null);
        let config = DiTConfig {
            prefix: "feat_decoder.estimator".to_string(),
            hidden_dim,
            num_layers: get_usize(dit, &["num_layers", "num_hidden_layers"], 12),
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_dim: get_usize(dit, &["ffn_dim", "intermediate_size"], 4096),
            rms_norm_eps: get_f64(lm, "rms_norm_eps", 1e-5),
            scale_depth: get_f64(lm, "scale_depth", 1.0),
            use_mup: lm.get("use_mup").and_then(|v| v.as_bool()).unwrap_or(false),
            rope_theta: get_f64(lm, "rope_theta", 10000.0),
            original_max_position_embeddings: rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            rope_short_factors: read_f32_array(rope_scaling, "short_factor", head_dim / 2),
            rope_long_factors: read_f32_array(rope_scaling, "long_factor", head_dim / 2),
            cfg_value: get_f64(cfm, "inference_cfg_rate", 1.0),
            n_steps: 10,
            sway_coef: get_f64(dit, "sway_sampling_coef", 1.0),
            latent_dim: manifest.feat_dim,
        };
        Self::load(loader, config, loader.device())
    }

    /// Load DiT weights from GGUF.
    pub fn load(loader: &GgufModelLoader, config: DiTConfig, device: &Device) -> Result<Self> {
        let p = &config.prefix;

        let load_lw = |name: &str| -> Result<LinearWeight> {
            let weight = loader.load_tensor_optimal(&format!("{p}.{name}.weight"))?;
            let bias_name = format!("{p}.{name}.bias");
            let bias = if loader.has_tensor(&bias_name) {
                Some(loader.load_tensor_optimal(&bias_name)?)
            } else {
                None
            };
            Ok(LinearWeight { weight, bias })
        };

        let in_proj = load_lw("in_proj")?;
        let cond_proj = load_lw("cond_proj")?;
        let out_proj = load_lw("out_proj")?;
        let time_mlp_1 = load_lw("time_mlp.linear_1")?;
        let time_mlp_2 = load_lw("time_mlp.linear_2")?;
        let delta_time_mlp_1 = load_lw("delta_time_mlp.linear_1")?;
        let delta_time_mlp_2 = load_lw("delta_time_mlp.linear_2")?;

        let final_norm = loader.load_tensor(&format!("{p}.decoder.norm.weight"))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lp = format!("{p}.decoder.layers.{i}");
            layers.push(DiTLayer {
                q_proj: loader.load_tensor_optimal(&format!("{lp}.self_attn.q_proj.weight"))?,
                k_proj: loader.load_tensor_optimal(&format!("{lp}.self_attn.k_proj.weight"))?,
                v_proj: loader.load_tensor_optimal(&format!("{lp}.self_attn.v_proj.weight"))?,
                o_proj: loader.load_tensor_optimal(&format!("{lp}.self_attn.o_proj.weight"))?,
                gate_proj: loader.load_tensor_optimal(&format!("{lp}.mlp.gate_proj.weight"))?,
                up_proj: loader.load_tensor_optimal(&format!("{lp}.mlp.up_proj.weight"))?,
                down_proj: loader.load_tensor_optimal(&format!("{lp}.mlp.down_proj.weight"))?,
                input_layernorm: loader.load_tensor(&format!("{lp}.input_layernorm.weight"))?,
                post_attention_layernorm: loader
                    .load_tensor(&format!("{lp}.post_attention_layernorm.weight"))?,
            });
        }

        // Precompute RoPE cache (no LongRope factors, standard theta=10000)
        let (cos_cache, sin_cache) = Self::build_rope_cache(&config, device)?;

        Ok(Self {
            config,
            in_proj,
            cond_proj,
            out_proj,
            time_mlp_1,
            time_mlp_2,
            delta_time_mlp_1,
            delta_time_mlp_2,
            layers,
            final_norm,
            cos_cache,
            sin_cache,
            device: device.clone(),
        })
    }

    fn build_rope_cache(config: &DiTConfig, device: &Device) -> Result<(Tensor, Tensor)> {
        let half_dim = config.head_dim / 2;
        let max_pos = config.original_max_position_embeddings.unwrap_or(8192).max(8192);
        let factors = if let Some(original) = config.original_max_position_embeddings {
            if max_pos > original {
                &config.rope_long_factors
            } else {
                &config.rope_short_factors
            }
        } else {
            &config.rope_short_factors
        };
        let mut freqs = vec![0f32; half_dim];
        for i in 0..half_dim {
            freqs[i] =
                1.0 / (config.rope_theta as f32).powf(2.0 * i as f32 / config.head_dim as f32)
                    / factors[i];
        }
        let freqs = Tensor::new(freqs.as_slice(), device)?;
        let scaling_factor = config
            .original_max_position_embeddings
            .filter(|v| *v > 1)
            .map(|original| {
                let scale = max_pos as f64 / original as f64;
                (1.0 + scale.ln() / (original as f64).ln()).sqrt()
            })
            .unwrap_or(1.0);
        let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let angles = positions.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        Ok(((angles.cos()? * scaling_factor)?, (angles.sin()? * scaling_factor)?))
    }

    /// Run the DiT forward pass (single evaluation).
    /// x: [B, latent_dim, T], mu: [B, N*hidden_dim], t: [B], cond: [B, latent_dim, T'], dt: [B]
    /// mu can be [B, hidden_dim] (N=1) or [B, N*hidden_dim] (N>1, e.g. N=2 for VoxCPM2).
    fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
        lora: Option<&LoraAdapter>,
    ) -> Result<Tensor> {
        // Determine model dtype from weights and cast inputs to match
        let model_dtype = self.in_proj.weight.dtype();
        let x = x.to_dtype(model_dtype)?;
        let mu = mu.to_dtype(model_dtype)?;
        let t = t.to_dtype(model_dtype)?;
        let cond = cond.to_dtype(model_dtype)?;
        let dt = dt.to_dtype(model_dtype)?;

        let (b, _ld, t_len) = x.dims3()?;
        let (_, _ld2, cond_len) = cond.dims3()?;

        // x: [B, 64, T] -> [B, T, 64] -> [B, T, 1024]
        let x_t = x.transpose(1, 2)?;
        let x_proj = linear(&x_t, &self.in_proj)?;

        // cond: [B, 64, T'] -> [B, T', 64] -> [B, T', 1024]
        let cond_t = cond.transpose(1, 2)?;
        let cond_proj = linear(&cond_t, &self.cond_proj)?;

        // Time embeddings
        let t_emb = sinusoidal_embedding(&t, self.config.hidden_dim, 1000.0)?;
        let t_emb = linear(&t_emb, &self.time_mlp_1)?;
        let t_emb = silu(&t_emb)?;
        let t_emb = linear(&t_emb, &self.time_mlp_2)?;

        let dt_emb = sinusoidal_embedding(&dt, self.config.hidden_dim, 1000.0)?;
        let dt_emb = linear(&dt_emb, &self.delta_time_mlp_1)?;
        let dt_emb = silu(&dt_emb)?;
        let dt_emb = linear(&dt_emb, &self.delta_time_mlp_2)?;

        let t_emb = (t_emb + dt_emb)?; // [B, 1024]

        // Reshape mu from [B, N*hidden_dim] to [B, N, hidden_dim] where N = mu.dim(1) / hidden_dim
        let mu_dim = mu.dim(1)?;
        let n_mu_tokens = mu_dim / self.config.hidden_dim;
        let mu_tokens = mu.reshape((b, n_mu_tokens, self.config.hidden_dim))?; // [B, N, 1024]

        // t_emb as a separate token: [B, 1, 1024]
        let t_token = t_emb.unsqueeze(1)?;

        // Concatenate: [mu_tokens, t_token, cond_proj, x_proj] along seq dim
        let mut hidden = Tensor::cat(&[&mu_tokens, &t_token, &cond_proj, &x_proj], 1)?;
        let prefix_len = n_mu_tokens + 1 + cond_len;
        let total_len = prefix_len + t_len;

        // RoPE cos/sin for full sequence
        let cos = self.cos_cache.narrow(0, 0, total_len)?;
        let sin = self.sin_cache.narrow(0, 0, total_len)?;

        let num_kv_groups = self.config.num_heads / self.config.num_kv_heads;
        let scale = 1.0 / (self.config.head_dim as f64).sqrt();
        let residual_scale = if self.config.use_mup {
            self.config.scale_depth / (self.config.num_layers as f64).sqrt()
        } else {
            1.0
        };

        // Transformer layers (non-causal, muP scaling)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            hidden = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // QKV
            let attn_input = hidden.clone();
            let mut q = linear_no_bias(&hidden, &layer.q_proj)?;
            let mut k = linear_no_bias(&hidden, &layer.k_proj)?;
            let mut v = linear_no_bias(&hidden, &layer.v_proj)?;
            if let Some(lora) = lora {
                let name = format!(
                    "{}.decoder.layers.{}.self_attn.q_proj",
                    self.config.prefix, layer_idx
                );
                q = lora.apply(&name, &q, &attn_input)?;
                let name = format!(
                    "{}.decoder.layers.{}.self_attn.k_proj",
                    self.config.prefix, layer_idx
                );
                k = lora.apply(&name, &k, &attn_input)?;
                let name = format!(
                    "{}.decoder.layers.{}.self_attn.v_proj",
                    self.config.prefix, layer_idx
                );
                v = lora.apply(&name, &v, &attn_input)?;
            }

            let q = q
                .reshape((b, total_len, self.config.num_heads, self.config.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b, total_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b, total_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?;

            // RoPE
            let q = apply_rope(&q, &cos, &sin)?;
            let k = apply_rope(&k, &cos, &sin)?;

            // GQA expand
            let q = q.contiguous()?;
            let k = repeat_kv(&k, num_kv_groups)?.contiguous()?;
            let v = repeat_kv(&v, num_kv_groups)?.contiguous()?;

            // Full (non-causal) attention
            let scores = q.matmul(&k.t()?.contiguous()?)?;
            let scores = (scores * scale)?;
            let attn_weights = crate::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v)?;

            let attn_out = attn_out
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, total_len, self.config.num_heads * self.config.head_dim))?;
            let o_input = attn_out.clone();
            let mut attn_out = linear_no_bias(&attn_out, &layer.o_proj)?;
            if let Some(lora) = lora {
                let name = format!(
                    "{}.decoder.layers.{}.self_attn.o_proj",
                    self.config.prefix, layer_idx
                );
                attn_out = lora.apply(&name, &attn_out, &o_input)?;
            }

            hidden = (residual + (attn_out * residual_scale)?)?;

            // Post-attention MLP
            let residual = hidden.clone();
            hidden =
                rms_norm(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;

            let mlp_input = hidden.clone();
            let mut gate = linear_no_bias(&hidden, &layer.gate_proj)?;
            let mut up = linear_no_bias(&hidden, &layer.up_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.decoder.layers.{}.mlp.gate_proj", self.config.prefix, layer_idx);
                gate = lora.apply(&name, &gate, &mlp_input)?;
                let name = format!("{}.decoder.layers.{}.mlp.up_proj", self.config.prefix, layer_idx);
                up = lora.apply(&name, &up, &mlp_input)?;
            }
            let gate = silu(&gate)?;
            let down_input = (gate * up)?;
            let mut mlp_out = linear_no_bias(&down_input, &layer.down_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.decoder.layers.{}.mlp.down_proj", self.config.prefix, layer_idx);
                mlp_out = lora.apply(&name, &mlp_out, &down_input)?;
            }
            hidden = (residual + (mlp_out * residual_scale)?)?;
        }

        // Final norm
        hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;

        // Extract last T tokens (the x portion)
        let output = hidden.narrow(1, prefix_len, t_len)?; // [B, T, 1024]
        let out_proj = LinearWeight {
            weight: self.out_proj.weight.clone(),
            bias: self.out_proj.bias.clone(),
        };
        let output = linear(&output, &out_proj)?; // [B, T, 64]
        output.transpose(1, 2).map_err(Into::into) // [B, 64, T]
    }

    /// Run the CFM Euler solver to generate clean latent from noise.
    /// mu: [B, N*hidden_dim] from LM output (N=1 or N=2 for VoxCPM2)
    /// cond: [B, latent_dim, T'] conditioning latent
    /// patch_size: number of output time frames
    pub fn generate(&self, mu: &Tensor, cond: &Tensor, patch_size: usize, n_steps: usize) -> Result<Tensor> {
        let b = mu.dims()[0];
        let noise = Tensor::randn(0f32, 1f32, (b, self.config.latent_dim, patch_size), &self.device)?;
        self.solve_euler_with_noise(mu, cond, &noise, n_steps, self.config.cfg_value as f32)
    }

    pub fn solve_euler_with_noise(
        &self,
        mu: &Tensor,
        cond: &Tensor,
        noise: &Tensor,
        n_steps: usize,
        cfg_value: f32,
    ) -> Result<Tensor> {
        self.solve_euler_with_noise_inner(mu, cond, noise, n_steps, cfg_value, None)
    }

    pub fn solve_euler_with_noise_lora(
        &self,
        mu: &Tensor,
        cond: &Tensor,
        noise: &Tensor,
        n_steps: usize,
        cfg_value: f32,
        lora: Option<&LoraAdapter>,
    ) -> Result<Tensor> {
        self.solve_euler_with_noise_inner(mu, cond, noise, n_steps, cfg_value, lora)
    }

    fn solve_euler_with_noise_inner(
        &self,
        mu: &Tensor,
        cond: &Tensor,
        noise: &Tensor,
        n_steps: usize,
        cfg_value: f32,
        lora: Option<&LoraAdapter>,
    ) -> Result<Tensor> {
        let b = mu.dims()[0];
        let patch_size = noise.dim(2)?;
        let mut x = noise.to_dtype(DType::F32)?;
        // Time schedule with sway sampling
        let sway = self.config.sway_coef;
        let mut t_span = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let t_lin = 1.0 - (i as f64) / (n_steps as f64); // linspace 1.0 -> 0.0
            let t_sway = t_lin + sway * ((std::f64::consts::FRAC_PI_2 * t_lin).cos() - 1.0 + t_lin);
            t_span.push(t_sway as f32);
        }

        // CFG-Zero* warmup steps
        let zero_init_steps = 1.max(((n_steps + 1) as f64 * 0.04) as usize);

        let mut t_val = t_span[0];

        for step in 1..=n_steps {
            let dt_val = t_val - t_span[step];

            let dphi_dt = if step <= zero_init_steps {
                Tensor::zeros((b, self.config.latent_dim, patch_size), DType::F32, &self.device)?
            } else {
                // Classifier-free guidance: double batch
                let x_doubled = Tensor::cat(&[&x, &x], 0)?; // [2B, 64, T]
                let mu_zeros = Tensor::zeros_like(mu)?;
                let mu_doubled = Tensor::cat(&[mu, &mu_zeros], 0)?; // [2B, N*1024]

                let t_tensor = Tensor::new(&[t_val, t_val], &self.device)?;
                // For B>1 we'd need to tile, but typically B=1 for inference
                let t_doubled = if b == 1 {
                    t_tensor
                } else {
                    let t_single = Tensor::from_vec(vec![t_val; b], (b,), &self.device)?;
                    Tensor::cat(&[&t_single, &t_single], 0)?
                };

                let dt_zero = Tensor::zeros((2 * b,), DType::F32, &self.device)?;
                let cond_doubled = Tensor::cat(&[cond, cond], 0)?; // [2B, 64, T']

                let v = self.forward(
                    &x_doubled,
                    &mu_doubled,
                    &t_doubled,
                    &cond_doubled,
                    &dt_zero,
                    lora,
                )?;
                // Cast back to f32 for ODE solver math
                let v = v.to_dtype(DType::F32)?;

                // Split: v_cond = v[:B], v_uncond = v[B:]
                let v_cond = v.narrow(0, 0, b)?;
                let v_uncond = v.narrow(0, b, b)?;

                // CFG-Zero* rescaling per sample
                // st_star = dot(v_cond, v_uncond) / ||v_uncond||^2
                // Flatten spatial dims for dot product
                let flat_dim = self.config.latent_dim * patch_size;
                let vc_flat = v_cond.reshape((b, flat_dim))?;
                let vu_flat = v_uncond.reshape((b, flat_dim))?;

                // dot product per sample: sum(vc * vu, dim=-1)
                let dot = (&vc_flat * &vu_flat)?.sum(D::Minus1)?; // [B]
                let vu_sq = vu_flat.sqr()?.sum(D::Minus1)?; // [B]
                // Avoid division by zero
                let eps = Tensor::new(&[1e-8f32], &self.device)?;
                let vu_sq_safe = (vu_sq + eps.broadcast_as((b,))?)?;
                let st_star = (dot / vu_sq_safe)?; // [B]

                // Reshape for broadcasting: [B, 1, 1]
                let st_star = st_star.unsqueeze(1)?.unsqueeze(2)?;

                // dphi_dt = v_uncond * st_star + cfg * (v_cond - v_uncond * st_star)
                let vu_scaled = v_uncond.broadcast_mul(&st_star)?;
                let diff = (v_cond - &vu_scaled)?;
                let cfg_t = Tensor::new(&[cfg_value], &self.device)?;
                (vu_scaled + diff.broadcast_mul(&cfg_t)?)?
            };

            // Euler step: x = x - dt * dphi_dt (t goes from 1 to 0)
            let dt_tensor = Tensor::new(&[dt_val], &self.device)?;
            x = (x - dphi_dt.broadcast_mul(&dt_tensor)?)?;
            t_val = t_span[step];
        }

        Ok(x)
    }
}
