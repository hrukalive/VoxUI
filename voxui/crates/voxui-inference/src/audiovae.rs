//! AudioVAE decoder: latent [B, 64, T] -> PCM.
//! Supports both V1 (VoxCPM-0.5B, VoxCPM1.5) and V2 (VoxCPM2) architectures.

use anyhow::Result;
use candle_core::Tensor;

use crate::manifest::AudioVaeManifest;
use crate::GgufModelLoader;

/// Configuration for the AudioVAE decoder.
pub struct AudioVAEConfig {
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sample_rate: u32,
    /// SR conditioning index (V2 only). If None, SR conditioning is skipped.
    pub sr_idx: Option<usize>,
}

impl Default for AudioVAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            decoder_dim: 2048,
            decoder_rates: vec![8, 6, 5, 2, 2, 2],
            sample_rate: 48000,
            sr_idx: Some(3),
        }
    }
}

/// Precomputed weight-normed conv1d weights + bias.
#[derive(Clone)]
struct Conv1dParams {
    weight: Tensor, // [out, in/groups, kernel]
    bias: Tensor,   // [out]
    groups: usize,
}

/// A residual unit: snake1 -> dw_conv -> snake2 -> pw_conv.
#[derive(Clone)]
struct ResidualUnit {
    alpha1: Tensor,
    dw_conv: Conv1dParams,
    alpha2: Tensor,
    pw_conv: Conv1dParams,
    dilation: usize,
}

/// A decoder block: (optional SR cond) -> snake -> transposed conv -> 3 residual units.
#[derive(Clone)]
struct DecoderBlock {
    scale: Option<Tensor>, // [dim] precomputed for sr_idx (V2 only)
    bias: Option<Tensor>,  // [dim] (V2 only)
    alpha: Tensor,         // snake alpha
    trans_conv: Conv1dParams,
    stride: usize,
    res_units: Vec<ResidualUnit>,
}

/// An encoder block: 3 residual units -> snake -> strided causal conv.
#[derive(Clone)]
struct EncoderBlock {
    res_units: Vec<ResidualUnit>,
    alpha: Tensor,
    down_conv: Conv1dParams,
    stride: usize,
}

/// AudioVAE encoder path producing the mean latent.
#[derive(Clone)]
struct CausalEncoder {
    first_conv: Conv1dParams,
    blocks: Vec<EncoderBlock>,
    fc_mu: Conv1dParams,
    hop_length: usize,
}

/// AudioVAE decoder.
#[derive(Clone)]
pub struct AudioVAE {
    encoder: Option<CausalEncoder>,
    dw_conv: Conv1dParams,
    pw_conv: Conv1dParams,
    blocks: Vec<DecoderBlock>,
    final_alpha: Tensor,
    final_conv: Conv1dParams,
}

pub struct StreamingAudioVaeDecoder {
    vae: AudioVAE,
    states: StreamingConvStates,
}

#[derive(Default)]
struct StreamingConvStates {
    states: Vec<Option<Tensor>>,
    cursor: usize,
}

impl StreamingConvStates {
    fn begin_chunk(&mut self) {
        self.cursor = 0;
    }

    fn clear(&mut self) {
        self.states.clear();
        self.cursor = 0;
    }

    fn next(&mut self) -> &mut Option<Tensor> {
        if self.cursor == self.states.len() {
            self.states.push(None);
        }
        let idx = self.cursor;
        self.cursor += 1;
        &mut self.states[idx]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute weight_norm: g * v / ||v|| per output channel.
fn weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    // g: [out, 1, 1], v: [out, in_per_group, kernel]
    // norm over dims 1,2 per output channel
    let v_sq = v.sqr()?;
    let norm = v_sq.sum_keepdim((1, 2))?.sqrt()?; // [out, 1, 1]
    let w = v.broadcast_mul(&g)?.broadcast_div(&norm)?;
    Ok(w)
}

/// Load a weight-normed conv1d from the GGUF file.
fn load_conv(
    loader: &GgufModelLoader,
    prefix: &str,
    requested_groups: usize,
) -> Result<Conv1dParams> {
    let g_name = format!("{prefix}.weight_g");
    let v_name = format!("{prefix}.weight_v");
    let bias_name = format!("{prefix}.bias");
    loader.ensure_dense_supported(&g_name, "audio_vae weight_norm")?;
    loader.ensure_dense_supported(&v_name, "conv1d")?;
    loader.ensure_dense_supported(&bias_name, "conv1d bias")?;
    let g = loader.load_tensor(&g_name)?;
    let v = loader.load_tensor(&v_name)?;
    let bias = loader.load_tensor(&bias_name)?;
    let groups = if requested_groups > 1 && v.dim(1)? == 1 {
        requested_groups
    } else {
        1
    };
    let weight = weight_norm(&g, &v)?;
    Ok(Conv1dParams {
        weight,
        bias,
        groups,
    })
}

/// Snake activation: x + (1/alpha) * sin(alpha * x)^2.
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha)?;
    let s = ax.sin()?.sqr()?;
    let inv_alpha = alpha.recip()?;
    let out = x.broadcast_add(&s.broadcast_mul(&inv_alpha)?)?;
    Ok(out)
}

/// Causal (left-padded) conv1d.
fn causal_conv1d(x: &Tensor, params: &Conv1dParams, dilation: usize) -> Result<Tensor> {
    let kernel_size = params.weight.dim(2)?;
    let padding = (kernel_size - 1) * dilation;
    causal_conv1d_with_padding(x, params, padding, 1, dilation)
}

/// Causal conv1d with explicit left padding and stride.
fn causal_conv1d_with_padding(
    x: &Tensor,
    params: &Conv1dParams,
    left_padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    // Left-pad with zeros: pad the last dimension on the left only.
    let x = if left_padding > 0 {
        x.pad_with_zeros(2, left_padding, 0)?
    } else {
        x.clone()
    };
    let out = x.conv1d(&params.weight, 0, stride, dilation, params.groups)?;
    // Add bias: bias is [out_channels], broadcast to [B, out_channels, T]
    let bias = params.bias.reshape((1, (), 1))?;
    Ok(out.broadcast_add(&bias)?)
}

fn causal_conv1d_streaming(
    x: &Tensor,
    params: &Conv1dParams,
    dilation: usize,
    states: &mut StreamingConvStates,
) -> Result<Tensor> {
    let kernel_size = params.weight.dim(2)?;
    let left_padding = (kernel_size - 1) * dilation;
    if left_padding == 0 {
        return causal_conv1d_with_padding(x, params, 0, 1, dilation);
    }

    let state = states.next();
    let x_pad = match state.as_ref() {
        Some(prev) => Tensor::cat(&[prev, x], 2)?,
        None => x.pad_with_zeros(2, left_padding, 0)?,
    };
    update_left_context(state, x, left_padding)?;

    let out = x_pad.conv1d(&params.weight, 0, 1, dilation, params.groups)?;
    let bias = params.bias.reshape((1, (), 1))?;
    Ok(out.broadcast_add(&bias)?)
}

/// Causal transposed conv1d (upsample).
fn causal_conv_transpose1d(x: &Tensor, params: &Conv1dParams, stride: usize) -> Result<Tensor> {
    // Python CausalTransposeConv1d consumes padding/output_padding in the
    // subclass and does not pass them to nn.ConvTranspose1d. It then trims the
    // right side by exactly `stride` for VoxCPM's kernel_size=2*stride blocks.
    let out = x.conv_transpose1d(&params.weight, 0, 0, stride, 1, params.groups)?;
    let input_len = x.dim(2)?;
    let target_len = input_len * stride;
    let actual_len = out.dim(2)?;
    let out = if actual_len > target_len {
        out.narrow(2, 0, target_len)?
    } else {
        out
    };
    let bias = params.bias.reshape((1, (), 1))?;
    Ok(out.broadcast_add(&bias)?)
}

fn causal_conv_transpose1d_streaming(
    x: &Tensor,
    params: &Conv1dParams,
    stride: usize,
    states: &mut StreamingConvStates,
) -> Result<Tensor> {
    let kernel_size = params.weight.dim(2)?;
    let ctx = (kernel_size - 1) / stride;
    if ctx == 0 {
        return causal_conv_transpose1d(x, params, stride);
    }

    let state = states.next();
    let x_full = match state.as_ref() {
        Some(prev) => Tensor::cat(&[prev, x], 2)?,
        None => x.pad_with_zeros(2, ctx, 0)?,
    };
    update_left_context(state, x, ctx)?;

    let out = x_full.conv_transpose1d(&params.weight, 0, 0, stride, 1, params.groups)?;
    let bias = params.bias.reshape((1, (), 1))?;
    let out = out.broadcast_add(&bias)?;
    let left = ctx * stride;
    let target_len = x.dim(2)? * stride;
    out.narrow(2, left, target_len).map_err(Into::into)
}

fn update_left_context(state: &mut Option<Tensor>, x: &Tensor, context_len: usize) -> Result<()> {
    if context_len == 0 {
        return Ok(());
    }

    let x_len = x.dim(2)?;
    let next = if x_len >= context_len {
        x.narrow(2, x_len - context_len, context_len)?
    } else {
        let prev = match state.as_ref() {
            Some(prev) => prev.clone(),
            None => Tensor::zeros((x.dim(0)?, x.dim(1)?, context_len), x.dtype(), x.device())?,
        };
        let combined = Tensor::cat(&[&prev, x], 2)?;
        let combined_len = combined.dim(2)?;
        combined.narrow(2, combined_len - context_len, context_len)?
    };
    *state = Some(next.contiguous()?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

impl ResidualUnit {
    fn load(loader: &GgufModelLoader, prefix: &str, dilation: usize, dim: usize) -> Result<Self> {
        let alpha1 = loader.load_tensor(&format!("{prefix}.block.0.alpha"))?;
        let dw_conv = load_conv(loader, &format!("{prefix}.block.1"), dim)?; // groups=dim
        let alpha2 = loader.load_tensor(&format!("{prefix}.block.2.alpha"))?;
        let pw_conv = load_conv(loader, &format!("{prefix}.block.3"), 1)?;
        Ok(Self {
            alpha1,
            dw_conv,
            alpha2,
            pw_conv,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = snake(x, &self.alpha1)?;
        let x = causal_conv1d(&x, &self.dw_conv, self.dilation)?;
        let x = snake(&x, &self.alpha2)?;
        let x = causal_conv1d(&x, &self.pw_conv, 1)?;
        Ok(x.add(&residual)?)
    }

    fn forward_streaming(&self, x: &Tensor, states: &mut StreamingConvStates) -> Result<Tensor> {
        let residual = x.clone();
        let x = snake(x, &self.alpha1)?;
        let x = causal_conv1d_streaming(&x, &self.dw_conv, self.dilation, states)?;
        let x = snake(&x, &self.alpha2)?;
        let x = causal_conv1d(&x, &self.pw_conv, 1)?;
        Ok(x.add(&residual)?)
    }
}

impl DecoderBlock {
    fn load(
        loader: &GgufModelLoader,
        block_idx: usize, // 0..N, maps to model.{2..2+N-1}
        _in_dim: usize,
        out_dim: usize,
        stride: usize,
        sr_idx: Option<usize>,
    ) -> Result<Self> {
        let model_idx = block_idx + 2;
        let prefix = format!("audio_vae.decoder.model.{model_idx}");

        // SR conditioning - optional (V2 only)
        let (scale, bias) = if let Some(sr_idx) = sr_idx {
            let sr_prefix = format!("audio_vae.decoder.sr_cond_model.{model_idx}");
            let scale_name = format!("{sr_prefix}.scale_embed.weight");
            if loader.has_tensor(&scale_name) {
                let scale_embed = loader.load_tensor(&scale_name)?;
                let bias_embed = loader.load_tensor(&format!("{sr_prefix}.bias_embed.weight"))?;
                (
                    Some(scale_embed.get(sr_idx)?),
                    Some(bias_embed.get(sr_idx)?),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let alpha = loader.load_tensor(&format!("{prefix}.block.0.alpha"))?;
        let trans_conv = load_conv(loader, &format!("{prefix}.block.1"), 1)?;

        // 3 residual units at block.{2,3,4} with dilations [1, 3, 9]
        let dilations = [1, 3, 9];
        let mut res_units = Vec::new();
        for (i, &dil) in dilations.iter().enumerate() {
            let r_prefix = format!("{prefix}.block.{}", i + 2);
            res_units.push(ResidualUnit::load(loader, &r_prefix, dil, out_dim)?);
        }

        Ok(Self {
            scale,
            bias,
            alpha,
            trans_conv,
            stride,
            res_units,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SR conditioning (V2 only): x * scale + bias
        let x = if let (Some(scale), Some(bias)) = (&self.scale, &self.bias) {
            let scale = scale.reshape((1, (), 1))?;
            let bias = bias.reshape((1, (), 1))?;
            x.broadcast_mul(&scale)?.broadcast_add(&bias)?
        } else {
            x.clone()
        };

        let x = snake(&x, &self.alpha)?;
        let mut x = causal_conv_transpose1d(&x, &self.trans_conv, self.stride)?;

        for ru in &self.res_units {
            x = ru.forward(&x)?;
        }
        Ok(x)
    }

    fn forward_streaming(&self, x: &Tensor, states: &mut StreamingConvStates) -> Result<Tensor> {
        let x = if let (Some(scale), Some(bias)) = (&self.scale, &self.bias) {
            let scale = scale.reshape((1, (), 1))?;
            let bias = bias.reshape((1, (), 1))?;
            x.broadcast_mul(&scale)?.broadcast_add(&bias)?
        } else {
            x.clone()
        };

        let x = snake(&x, &self.alpha)?;
        let mut x = causal_conv_transpose1d_streaming(&x, &self.trans_conv, self.stride, states)?;

        for ru in &self.res_units {
            x = ru.forward_streaming(&x, states)?;
        }
        Ok(x)
    }
}

impl EncoderBlock {
    fn load(
        loader: &GgufModelLoader,
        block_idx: usize,
        input_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let prefix = format!("audio_vae.encoder.block.{block_idx}");
        let dilations = [1, 3, 9];
        let mut res_units = Vec::new();
        for (i, &dilation) in dilations.iter().enumerate() {
            res_units.push(ResidualUnit::load(
                loader,
                &format!("{prefix}.block.{i}"),
                dilation,
                input_dim,
            )?);
        }
        let alpha = loader.load_tensor(&format!("{prefix}.block.3.alpha"))?;
        let down_conv = load_conv(loader, &format!("{prefix}.block.4"), groups)?;
        Ok(Self {
            res_units,
            alpha,
            down_conv,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for unit in &self.res_units {
            x = unit.forward(&x)?;
        }
        let x = snake(&x, &self.alpha)?;
        // Python CausalConv1d consumes padding/output_padding in the subclass
        // and only uses them to left-pad. For encoder downsample blocks this
        // effective left padding is exactly `stride`.
        causal_conv1d_with_padding(&x, &self.down_conv, self.stride, self.stride, 1)
    }
}

impl CausalEncoder {
    fn load(loader: &GgufModelLoader, manifest: &AudioVaeManifest) -> Result<Self> {
        let encoder_dim = manifest.encoder_dim.unwrap_or(128);
        let depthwise = true;
        let first_conv = load_conv(loader, "audio_vae.encoder.block.0", 1)?;
        let mut blocks = Vec::new();
        let mut dim = encoder_dim;
        let mut hop_length = 1usize;
        for (idx, &stride) in manifest.encoder_rates.iter().enumerate() {
            let groups = if depthwise { dim } else { 1 };
            blocks.push(EncoderBlock::load(loader, idx + 1, dim, stride, groups)?);
            dim *= 2;
            hop_length *= stride;
        }
        let fc_mu = load_conv(loader, "audio_vae.encoder.fc_mu", 1)?;
        Ok(Self {
            first_conv,
            blocks,
            fc_mu,
            hop_length,
        })
    }

    fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let mut x = match audio.dims().len() {
            2 => audio.unsqueeze(1)?,
            3 => audio.clone(),
            _ => anyhow::bail!(
                "audio tensor must be [B, T] or [B, 1, T], got {:?}",
                audio.dims()
            ),
        };
        let len = x.dim(2)?;
        let right_pad = len.div_ceil(self.hop_length) * self.hop_length - len;
        if right_pad > 0 {
            x = x.pad_with_zeros(2, 0, right_pad)?;
        }
        x = causal_conv1d(&x, &self.first_conv, 1)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        causal_conv1d(&x, &self.fc_mu, 1)
    }
}

impl AudioVAE {
    /// Load the AudioVAE decoder using model config metadata.
    pub fn load_from_config(loader: &GgufModelLoader, manifest: &AudioVaeManifest) -> Result<Self> {
        let decoder_dim = manifest
            .decoder_dim
            .or_else(|| {
                loader
                    .tensor_info("audio_vae.decoder.model.1.weight_v")
                    .map(|info| info.shape[0] as usize)
            })
            .unwrap_or(2048);
        let encoder = if loader.has_tensor("audio_vae.encoder.block.0.weight_v") {
            Some(CausalEncoder::load(loader, manifest)?)
        } else {
            None
        };
        let mut vae = Self::load(
            loader,
            AudioVAEConfig {
                latent_dim: manifest.latent_dim,
                decoder_dim,
                decoder_rates: manifest.decoder_rates.clone(),
                sample_rate: manifest.out_sample_rate.unwrap_or(manifest.sample_rate),
                sr_idx: if loader.has_tensor("audio_vae.decoder.sr_cond_model.2.scale_embed.weight")
                {
                    Some(3)
                } else {
                    None
                },
            },
        )?;
        vae.encoder = encoder;
        Ok(vae)
    }

    /// Encode waveform audio to latent means.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        self.encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("AudioVAE encoder weights were not loaded"))?
            .encode(audio)
    }

    #[allow(dead_code)]
    fn infer_decoder_dim(loader: &GgufModelLoader) -> usize {
        loader
            .tensor_info("audio_vae.decoder.model.1.weight_v")
            .map(|info| info.shape[0] as usize)
            .unwrap_or(2048)
    }

    /// Detect the number of decoder blocks from GGUF tensor names.
    fn count_decoder_blocks(loader: &GgufModelLoader) -> usize {
        let names = loader.tensor_names();
        let mut max_block: usize = 0;
        for name in &names {
            if let Some(rest) = name.strip_prefix("audio_vae.decoder.model.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(n) = rest[..dot_pos].parse::<usize>() {
                        // Blocks at indices 2..2+N-1; look for snake alpha as marker
                        if n >= 2 && rest.contains("block.0.alpha") {
                            max_block = max_block.max(n);
                        }
                    }
                }
            }
        }
        if max_block < 2 {
            0
        } else {
            max_block - 2 + 1
        }
    }

    /// Infer decoder_rates from transposed conv kernel sizes.
    fn infer_decoder_rates(loader: &GgufModelLoader, num_blocks: usize) -> Result<Vec<usize>> {
        let mut rates = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let model_idx = i + 2;
            // Transposed conv weight_v shape: [in, out, kernel_size]
            // stride = kernel_size / 2
            let name = format!("audio_vae.decoder.model.{model_idx}.block.1.weight_v");
            loader.ensure_dense_supported(&name, "audio_vae decoder rate inference")?;
            let w = loader.load_tensor(&name)?;
            let kernel_size = w.dim(2)?;
            rates.push(kernel_size / 2);
        }
        Ok(rates)
    }

    /// Load the AudioVAE decoder from a GGUF file.
    pub fn load(loader: &GgufModelLoader, config: AudioVAEConfig) -> Result<Self> {
        // model.0: depthwise causal conv (groups=latent_dim)
        let dw_conv = load_conv(loader, "audio_vae.decoder.model.0", config.latent_dim)?;
        // model.1: pointwise conv latent_dim -> decoder_dim
        let pw_conv = load_conv(loader, "audio_vae.decoder.model.1", 1)?;

        // Detect block count and rates dynamically
        let num_blocks = Self::count_decoder_blocks(loader);
        let decoder_rates =
            if !config.decoder_rates.is_empty() && config.decoder_rates.len() == num_blocks {
                config.decoder_rates
            } else {
                Self::infer_decoder_rates(loader, num_blocks)?
            };

        // Decoder blocks
        let mut blocks = Vec::new();
        let mut dim = config.decoder_dim;
        for (i, &rate) in decoder_rates.iter().enumerate() {
            let out_dim = dim / 2;
            blocks.push(DecoderBlock::load(
                loader,
                i,
                dim,
                out_dim,
                rate,
                config.sr_idx,
            )?);
            dim = out_dim;
        }

        // Final snake + conv: indices are 2+num_blocks and 2+num_blocks+1
        let final_snake_idx = 2 + num_blocks;
        let final_conv_idx = final_snake_idx + 1;
        let final_alpha =
            loader.load_tensor(&format!("audio_vae.decoder.model.{final_snake_idx}.alpha"))?;
        let final_conv = load_conv(
            loader,
            &format!("audio_vae.decoder.model.{final_conv_idx}"),
            1,
        )?;

        Ok(Self {
            encoder: None,
            dw_conv,
            pw_conv,
            blocks,
            final_alpha,
            final_conv,
        })
    }

    /// Decode latent to PCM audio.
    /// Input: `[B, 64, T]` — Output: `[B, 1, T * 1920]` (f32 in [-1, 1]).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // DW causal conv
        let x = causal_conv1d(latent, &self.dw_conv, 1)?;
        // PW conv
        let x = causal_conv1d(&x, &self.pw_conv, 1)?;

        // Decoder blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final snake + conv + tanh
        let x = snake(&x, &self.final_alpha)?;
        let x = causal_conv1d(&x, &self.final_conv, 1)?;
        Ok(x.tanh()?)
    }

    pub fn streaming_decoder(&self) -> StreamingAudioVaeDecoder {
        StreamingAudioVaeDecoder {
            vae: self.clone(),
            states: StreamingConvStates::default(),
        }
    }
}

impl StreamingAudioVaeDecoder {
    pub fn decode_chunk(&mut self, latent_chunk: &Tensor) -> Result<Tensor> {
        self.states.begin_chunk();
        let x = causal_conv1d_streaming(latent_chunk, &self.vae.dw_conv, 1, &mut self.states)?;
        let mut x = causal_conv1d(&x, &self.vae.pw_conv, 1)?;

        for block in &self.vae.blocks {
            x = block.forward_streaming(&x, &mut self.states)?;
        }

        let x = snake(&x, &self.vae.final_alpha)?;
        let x = causal_conv1d_streaming(&x, &self.vae.final_conv, 1, &mut self.states)?;
        Ok(x.tanh()?)
    }

    pub fn reset(&mut self) {
        self.states.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use byteorder::{LittleEndian, WriteBytesExt};
    use candle_core::Device;
    use std::io::Write;

    fn write_string(mut out: impl Write, value: &str) -> Result<()> {
        out.write_u64::<LittleEndian>(value.len() as u64)?;
        out.write_all(value.as_bytes())?;
        Ok(())
    }

    fn align_32(len: usize) -> usize {
        (len + 31) & !31
    }

    fn write_quantized_conv_gguf(path: &std::path::Path) -> Result<()> {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(0x46554747)?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "audio_vae.decoder.model.0.weight_g")?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u32::<LittleEndian>(0)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "audio_vae.decoder.model.0.weight_v")?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(32)?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(4)?;

        write_string(&mut bytes, "audio_vae.decoder.model.0.bias")?;
        bytes.write_u32::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u32::<LittleEndian>(0)?;
        bytes.write_u64::<LittleEndian>(22)?;

        bytes.resize(align_32(bytes.len()), 0);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes()[0..2]);
        bytes.extend_from_slice(&[0x88; 16]);
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        std::fs::write(path, bytes)?;
        Ok(())
    }

    #[test]
    fn quantized_conv_weight_is_rejected() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.gguf");
        write_quantized_conv_gguf(&path)?;
        let loader = GgufModelLoader::new(&path, Device::Cpu)?;

        let err = match load_conv(&loader, "audio_vae.decoder.model.0", 1) {
            Ok(_) => anyhow::bail!("expected quantized conv load to fail"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("unsupported quantized tensor audio_vae.decoder.model.0.weight_v"));
        Ok(())
    }
}
