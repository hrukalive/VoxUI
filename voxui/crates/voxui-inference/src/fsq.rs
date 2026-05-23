use anyhow::Result;
use candle_core::Tensor;

use crate::model_loader::GgufModelLoader;
use crate::{LinearWeight, RuntimeTensor};

pub struct FSQLayer {
    in_proj: LinearWeight,
    in_proj_bias: Option<RuntimeTensor>,
    out_proj: LinearWeight,
    out_proj_bias: Option<RuntimeTensor>,
    scale: f64, // 9.0 for VoxCPM2
}

impl FSQLayer {
    pub fn load(loader: &GgufModelLoader, _latent_dim: usize, scale: f64) -> Result<Self> {
        let in_proj = loader.load_linear_weight("fsq_layer.in_proj.weight")?;
        let in_proj_bias = if loader.has_tensor("fsq_layer.in_proj.bias") {
            Some(loader.load_runtime_tensor("fsq_layer.in_proj.bias")?)
        } else {
            None
        };
        let out_proj = loader.load_linear_weight("fsq_layer.out_proj.weight")?;
        let out_proj_bias = if loader.has_tensor("fsq_layer.out_proj.bias") {
            Some(loader.load_runtime_tensor("fsq_layer.out_proj.bias")?)
        } else {
            None
        };
        Ok(Self {
            in_proj,
            in_proj_bias,
            out_proj,
            out_proj_bias,
            scale,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // hidden: [B, T, hidden_size] or [B, hidden_size]
        // Bottleneck: hidden_size -> latent_dim -> hidden_size
        let x = linear_with_optional_bias(hidden, &self.in_proj, self.in_proj_bias.as_ref())?;
        let x = x.tanh()?;
        // Quantize: round(x * scale) / scale
        // Work in f32 for the scalar ops to avoid dtype mismatch on CUDA
        let x = x.to_dtype(candle_core::DType::F32)?;
        let scaled = (x * self.scale)?;
        let rounded = scaled.round()?;
        let quantized = (rounded / self.scale)?;
        // Project back to output dim
        linear_with_optional_bias(&quantized, &self.out_proj, self.out_proj_bias.as_ref())
    }
}

fn linear_with_optional_bias(
    input: &Tensor,
    weight: &LinearWeight,
    bias: Option<&RuntimeTensor>,
) -> Result<Tensor> {
    let out = weight.forward(input)?;
    if let Some(bias) = bias {
        out.broadcast_add(&bias.to_dense_dtype(out.dtype())?)
            .map_err(Into::into)
    } else {
        Ok(out)
    }
}
