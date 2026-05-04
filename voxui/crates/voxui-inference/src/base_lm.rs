//! MiniCPM-4 (VoxCPM2) base language model forward pass.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::silu;

use crate::model_loader::GgufModelLoader;

/// Configuration for the base LM transformer.
pub struct BaseLMConfig {
    pub hidden_size: usize,      // 2048
    pub num_layers: usize,       // 28
    pub num_heads: usize,        // 16 (query heads)
    pub num_kv_heads: usize,     // 2 (GQA)
    pub head_dim: usize,         // 128
    pub intermediate_size: usize, // 6144
    pub rms_norm_eps: f64,       // 1e-5
    pub rope_theta: f64,         // 10000
    pub rope_factors: Vec<f32>,  // LongRope factors (64 elements = head_dim/2)
    pub vocab_size: usize,       // 73448
    pub max_position: usize,     // max sequence length for cache/rope precomputation
    pub prefix: String,          // tensor name prefix: "base_lm" or "residual_lm"
    pub no_rope: bool,           // skip RoPE (used by residual LM)
    pub is_causal: bool,         // true for LMs, false for encoder
}

impl Default for BaseLMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 28,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            intermediate_size: 6144,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_factors: vec![1.0; 64], // default: no scaling
            vocab_size: 73448,
            max_position: 4096,
            prefix: "base_lm".to_string(),
            no_rope: false,
            is_causal: true,
        }
    }
}

/// Weights for a single transformer layer.
struct TransformerLayer {
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

/// The base language model (MiniCPM-4 / VoxCPM2 architecture).
pub struct BaseLM {
    config: BaseLMConfig,
    embed_tokens: Option<Tensor>,
    layers: Vec<TransformerLayer>,
    norm: Tensor,
    // Precomputed RoPE
    cos_cache: Tensor, // [max_position, head_dim/2]
    sin_cache: Tensor,
    // KV cache per layer
    k_cache: Vec<Tensor>, // [1, num_kv_heads, cached_len, head_dim]
    v_cache: Vec<Tensor>,
    cache_len: usize,
    device: Device,
}

/// RMS normalization: x * rsqrt(mean(x^2) + eps) * weight
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    // Compute in f32 for stability
    let x = x.to_dtype(DType::F32)?;
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let eps_t = mean_sq.zeros_like()?.broadcast_add(
        &Tensor::new(&[eps as f32], mean_sq.device())?,
    )?;
    let norm = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let out = x.broadcast_mul(&norm)?;
    let weight = weight.to_dtype(DType::F32)?;
    let out = out.broadcast_mul(&weight)?;
    out.to_dtype(dtype).map_err(Into::into)
}

/// Apply rotary position embeddings to x.
/// x: [batch, heads, seq_len, head_dim]
/// cos, sin: [seq_len, head_dim/2] (sliced for the relevant positions)
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _t, hd) = x.dims4()?;
    let half = hd / 2;
    // Split into first half and second half of head_dim
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    // cos/sin are [seq_len, half], need [1, 1, seq_len, half]
    // Cast to match x's dtype (f16 on CUDA, f32 on CPU)
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(x.dtype())?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?.to_dtype(x.dtype())?;
    // RoPE: (x1*cos - x2*sin, x1*sin + x2*cos)
    let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&o1, &o2], 3).map_err(Into::into)
}

/// Repeat KV heads to match query head count.
/// x: [batch, kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, t, hd) = x.dims4()?;
    // [b, kv_h, 1, t, hd] -> [b, kv_h, n_rep, t, hd] -> [b, kv_h*n_rep, t, hd]
    x.unsqueeze(2)?
        .expand((b, kv_h, n_rep, t, hd))?
        .reshape((b, kv_h * n_rep, t, hd))
        .map_err(Into::into)
}

impl BaseLM {
    /// Load model weights from a GGUF file.
    pub fn load(loader: &GgufModelLoader, config: BaseLMConfig, device: &Device) -> Result<Self> {
        // embed_tokens is optional — residual_lm and encoder don't have it
        let embed_name = format!("{}.embed_tokens.weight", config.prefix);
        let embed_tokens = if loader.has_tensor(&embed_name) {
            Some(loader.load_tensor_optimal(&embed_name)?)
        } else {
            None
        };
        let norm = loader.load_tensor(&format!("{}.norm.weight", config.prefix))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("{}.layers.{i}", config.prefix);
            layers.push(TransformerLayer {
                q_proj: loader.load_tensor_optimal(&format!("{prefix}.self_attn.q_proj.weight"))?,
                k_proj: loader.load_tensor_optimal(&format!("{prefix}.self_attn.k_proj.weight"))?,
                v_proj: loader.load_tensor_optimal(&format!("{prefix}.self_attn.v_proj.weight"))?,
                o_proj: loader.load_tensor_optimal(&format!("{prefix}.self_attn.o_proj.weight"))?,
                gate_proj: loader.load_tensor_optimal(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up_proj: loader.load_tensor_optimal(&format!("{prefix}.mlp.up_proj.weight"))?,
                down_proj: loader.load_tensor_optimal(&format!("{prefix}.mlp.down_proj.weight"))?,
                input_layernorm: loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            });
        }

        // Precompute RoPE cos/sin cache with LongRope factors
        let (cos_cache, sin_cache) = Self::build_rope_cache(&config, device)?;

        // Initialize empty KV caches
        let k_cache = Vec::new();
        let v_cache = Vec::new();

        let mut model = Self {
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
            k_cache,
            v_cache,
            cache_len: 0,
            device: device.clone(),
        };
        model.reset_cache();
        Ok(model)
    }

    /// Build the RoPE cos/sin cache using LongRope frequency scaling.
    fn build_rope_cache(config: &BaseLMConfig, device: &Device) -> Result<(Tensor, Tensor)> {
        let half_dim = config.head_dim / 2;
        assert_eq!(
            config.rope_factors.len(),
            half_dim,
            "rope_factors must have head_dim/2 elements"
        );

        // base_freq[i] = 1 / (theta ^ (2i / head_dim))
        // scaled_freq[i] = base_freq[i] / factor[i]
        let mut freqs = vec![0f32; half_dim];
        for i in 0..half_dim {
            let base = 1.0
                / (config.rope_theta as f32)
                    .powf(2.0 * i as f32 / config.head_dim as f32);
            freqs[i] = base / config.rope_factors[i];
        }
        let freqs = Tensor::new(freqs.as_slice(), device)?; // [half_dim]

        // positions: [0, 1, ..., max_position-1]
        let positions: Vec<f32> = (0..config.max_position).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?; // [max_pos]

        // outer product: [max_pos, half_dim]
        let angles = positions.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;

        let cos_cache = angles.cos()?;
        let sin_cache = angles.sin()?;
        Ok((cos_cache, sin_cache))
    }

    /// Reset KV cache to empty state.
    pub fn reset_cache(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.cache_len = 0;
        // We'll lazily initialize caches on first forward call.
    }

    /// Linear: x @ weight^T. Delegates to shared crate::linear.
    fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        crate::linear(x, weight)
    }

    /// Embed token IDs into hidden states [1, T, hidden_size].
    /// Only works for models with an embedding table (base_lm). Panics otherwise.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let embed = self.embed_tokens.as_ref()
            .ok_or_else(|| anyhow::anyhow!("This model has no embed_tokens (residual_lm/encoder don't support embed)"))?;
        let seq_len = token_ids.len();
        let ids_tensor = Tensor::new(token_ids, &self.device)?;
        let hidden = embed.index_select(&ids_tensor, 0)?;
        hidden.reshape((1, seq_len, self.config.hidden_size)).map_err(Into::into)
    }

    /// Run forward pass for a sequence of tokens. Returns hidden_states [1, seq_len, hidden_size].
    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            bail!("token_ids must not be empty");
        }

        let hidden = self.embed(token_ids)?;
        self.forward_embed(&hidden)
    }

    /// Run forward pass with pre-computed embeddings [1, seq_len, hidden_size].
    /// Returns hidden_states [1, seq_len, hidden_size].
    pub fn forward_embed(&mut self, hidden: &Tensor) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let mut hidden = hidden.clone();

        // Get RoPE cos/sin for positions [cache_len .. cache_len+seq_len)
        let start_pos = self.cache_len;
        let cos = self.cos_cache.narrow(0, start_pos, seq_len)?; // [T, head_dim/2]
        let sin = self.sin_cache.narrow(0, start_pos, seq_len)?;

        let num_kv_groups = self.config.num_heads / self.config.num_kv_heads;
        let scale = 1.0 / (self.config.head_dim as f64).sqrt();

        // 2. Transformer layers
        for layer_idx in 0..self.config.num_layers {
            let layer = &self.layers[layer_idx];

            // Pre-attention norm
            let residual = hidden.clone();
            hidden = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // QKV projections
            let q = Self::linear(&hidden, &layer.q_proj)?; // [1, T, num_heads*head_dim]
            let k = Self::linear(&hidden, &layer.k_proj)?; // [1, T, num_kv_heads*head_dim]
            let v = Self::linear(&hidden, &layer.v_proj)?; // [1, T, num_kv_heads*head_dim]

            // Reshape to [B, heads, T, head_dim]
            let q = q
                .reshape((1, seq_len, self.config.num_heads, self.config.head_dim))?
                .transpose(1, 2)?; // [1, 16, T, 128]
            let k = k
                .reshape((1, seq_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?; // [1, 2, T, 128]
            let v = v
                .reshape((1, seq_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?; // [1, 2, T, 128]

            // Apply RoPE (unless disabled)
            let (q, k) = if self.config.no_rope {
                (q, k)
            } else {
                (apply_rope(&q, &cos, &sin)?, apply_rope(&k, &cos, &sin)?)
            };

            // Update KV cache
            let (k_full, v_full) = if self.k_cache.len() > layer_idx && self.cache_len > 0 {
                let k_full =
                    Tensor::cat(&[&self.k_cache[layer_idx], &k], 2)?; // [1, 2, cache+T, 128]
                let v_full = Tensor::cat(&[&self.v_cache[layer_idx], &v], 2)?;
                (k_full, v_full)
            } else {
                (k.clone(), v.clone())
            };

            // Store updated cache
            if self.k_cache.len() <= layer_idx {
                self.k_cache.push(k_full.clone());
                self.v_cache.push(v_full.clone());
            } else {
                self.k_cache[layer_idx] = k_full.clone();
                self.v_cache[layer_idx] = v_full.clone();
            }

            let total_len = start_pos + seq_len;

            // GQA: expand KV heads
            let k_exp = repeat_kv(&k_full, num_kv_groups)?; // [1, 16, total_len, 128]
            let v_exp = repeat_kv(&v_full, num_kv_groups)?;

            // Attention scores: Q @ K^T / sqrt(d)
            let scores = q.matmul(&k_exp.t()?)?; // [1, 16, T, total_len]
            let scores = (scores * scale)?;

            // Causal mask: position i can attend to positions <= start_pos + i
            let scores = if self.config.is_causal && seq_len > 1 {
                // Build causal mask for prefill
                let mask = Self::causal_mask(seq_len, total_len, start_pos, &self.device)?;
                let mask = mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, T, total_len]
                scores.broadcast_add(&mask)?
            } else {
                // Non-causal (encoder) or single token: full attention
                scores
            };

            let attn_weights = crate::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v_exp)?; // [1, 16, T, 128]

            // Reshape back: [1, T, num_heads * head_dim]
            let attn_out = attn_out
                .transpose(1, 2)?
                .reshape((1, seq_len, self.config.num_heads * self.config.head_dim))?;

            // Output projection
            let attn_out = Self::linear(&attn_out, &layer.o_proj)?;
            hidden = (residual + attn_out)?;

            // Post-attention norm + MLP
            let residual = hidden.clone();
            hidden =
                rms_norm(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;

            let gate = silu(&Self::linear(&hidden, &layer.gate_proj)?)?;
            let up = Self::linear(&hidden, &layer.up_proj)?;
            let mlp_out = Self::linear(&(gate * up)?, &layer.down_proj)?;
            hidden = (residual + mlp_out)?;
        }

        // 3. Final norm
        hidden = rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

        self.cache_len = start_pos + seq_len;
        Ok(hidden)
    }

    /// Run forward pass with pre-computed embeddings and optional LoRA adapter.
    /// Returns hidden_states [1, seq_len, hidden_size].
    pub fn forward_embed_with_lora(&mut self, hidden: &Tensor, lora: Option<&crate::lora::LoraAdapter>) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let mut hidden = hidden.clone();

        let start_pos = self.cache_len;
        let cos = self.cos_cache.narrow(0, start_pos, seq_len)?;
        let sin = self.sin_cache.narrow(0, start_pos, seq_len)?;

        let num_kv_groups = self.config.num_heads / self.config.num_kv_heads;
        let scale = 1.0 / (self.config.head_dim as f64).sqrt();

        for layer_idx in 0..self.config.num_layers {
            let layer = &self.layers[layer_idx];

            let residual = hidden.clone();
            hidden = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // QKV projections with LoRA
            let q_input = hidden.clone();
            let mut q = Self::linear(&hidden, &layer.q_proj)?;
            let mut k = Self::linear(&hidden, &layer.k_proj)?;
            let mut v = Self::linear(&hidden, &layer.v_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.layers.{}.self_attn.q_proj", self.config.prefix, layer_idx);
                q = lora.apply(&name, &q, &q_input)?;
                let name = format!("{}.layers.{}.self_attn.k_proj", self.config.prefix, layer_idx);
                k = lora.apply(&name, &k, &q_input)?;
                let name = format!("{}.layers.{}.self_attn.v_proj", self.config.prefix, layer_idx);
                v = lora.apply(&name, &v, &q_input)?;
            }

            let q = q
                .reshape((1, seq_len, self.config.num_heads, self.config.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((1, seq_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((1, seq_len, self.config.num_kv_heads, self.config.head_dim))?
                .transpose(1, 2)?;

            let (q, k) = if self.config.no_rope {
                (q, k)
            } else {
                (apply_rope(&q, &cos, &sin)?, apply_rope(&k, &cos, &sin)?)
            };

            let (k_full, v_full) = if self.k_cache.len() > layer_idx && self.cache_len > 0 {
                let k_full = Tensor::cat(&[&self.k_cache[layer_idx], &k], 2)?;
                let v_full = Tensor::cat(&[&self.v_cache[layer_idx], &v], 2)?;
                (k_full, v_full)
            } else {
                (k.clone(), v.clone())
            };

            if self.k_cache.len() <= layer_idx {
                self.k_cache.push(k_full.clone());
                self.v_cache.push(v_full.clone());
            } else {
                self.k_cache[layer_idx] = k_full.clone();
                self.v_cache[layer_idx] = v_full.clone();
            }

            let total_len = start_pos + seq_len;

            let k_exp = repeat_kv(&k_full, num_kv_groups)?;
            let v_exp = repeat_kv(&v_full, num_kv_groups)?;

            let scores = q.matmul(&k_exp.t()?)?;
            let scores = (scores * scale)?;

            let scores = if self.config.is_causal && seq_len > 1 {
                let mask = Self::causal_mask(seq_len, total_len, start_pos, &self.device)?;
                let mask = mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };

            let attn_weights = crate::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v_exp)?;

            let attn_out = attn_out
                .transpose(1, 2)?
                .reshape((1, seq_len, self.config.num_heads * self.config.head_dim))?;

            // O projection with LoRA
            let o_input = attn_out.clone();
            let mut attn_out = Self::linear(&attn_out, &layer.o_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.layers.{}.self_attn.o_proj", self.config.prefix, layer_idx);
                attn_out = lora.apply(&name, &attn_out, &o_input)?;
            }
            hidden = (residual + attn_out)?;

            // Post-attention norm + MLP with LoRA
            let residual = hidden.clone();
            hidden = rms_norm(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;

            let mlp_input = hidden.clone();
            let mut gate = Self::linear(&hidden, &layer.gate_proj)?;
            let mut up = Self::linear(&hidden, &layer.up_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.layers.{}.mlp.gate_proj", self.config.prefix, layer_idx);
                gate = lora.apply(&name, &gate, &mlp_input)?;
                let name = format!("{}.layers.{}.mlp.up_proj", self.config.prefix, layer_idx);
                up = lora.apply(&name, &up, &mlp_input)?;
            }
            let gate = silu(&gate)?;
            let down_input = (gate * up)?;
            let mut mlp_out = Self::linear(&down_input, &layer.down_proj)?;
            if let Some(lora) = lora {
                let name = format!("{}.layers.{}.mlp.down_proj", self.config.prefix, layer_idx);
                mlp_out = lora.apply(&name, &mlp_out, &down_input)?;
            }
            hidden = (residual + mlp_out)?;
        }

        hidden = rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

        self.cache_len = start_pos + seq_len;
        Ok(hidden)
    }

    /// Run single-token forward step (autoregressive). Returns hidden_states [1, 1, hidden_size].
    pub fn forward_step(&mut self, token_id: u32) -> Result<Tensor> {
        self.forward(&[token_id])
    }

    /// Run single-step forward with a pre-computed embedding [1, 1, hidden_size].
    pub fn forward_step_embed(&mut self, embed: &Tensor) -> Result<Tensor> {
        self.forward_embed(embed)
    }

    /// Build a causal mask: 0 where allowed, -inf where masked.
    /// For a prefill of `seq_len` tokens starting at `start_pos`, attending to `total_len` KV entries.
    fn causal_mask(
        seq_len: usize,
        total_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // query position i (0..seq_len) has absolute position start_pos + i
        // it can attend to kv position j (0..total_len) if j <= start_pos + i
        let mut mask = vec![0f32; seq_len * total_len];
        for i in 0..seq_len {
            let abs_pos = start_pos + i;
            for j in 0..total_len {
                if j > abs_pos {
                    mask[i * total_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(mask, (seq_len, total_len), device).map_err(Into::into)
    }
}
