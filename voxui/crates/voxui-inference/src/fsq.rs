use candle_core::Tensor;
use crate::model_loader::GgufModelLoader;
use crate::linear;
use anyhow::Result;

pub struct FSQLayer {
    in_proj: Tensor,   // weight [latent_dim, hidden_size]
    out_proj: Tensor,  // weight [out_dim, latent_dim]
    scale: f64,        // 9.0 for VoxCPM2
}

impl FSQLayer {
    pub fn load(loader: &GgufModelLoader, _latent_dim: usize, scale: f64) -> Result<Self> {
        let in_proj = loader.load_tensor_optimal("fsq_layer.in_proj.weight")?;
        let out_proj = loader.load_tensor_optimal("fsq_layer.out_proj.weight")?;
        Ok(Self { in_proj, out_proj, scale })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // hidden: [B, T, hidden_size] or [B, hidden_size]
        // Bottleneck: hidden_size -> latent_dim -> hidden_size
        let x = linear(hidden, &self.in_proj)?;
        let x = x.tanh()?;
        // Quantize: round(x * scale) / scale
        // Work in f32 for the scalar ops to avoid dtype mismatch on CUDA
        let x = x.to_dtype(candle_core::DType::F32)?;
        let scaled = (x * self.scale)?;
        let rounded = scaled.round()?;
        let quantized = (rounded / self.scale)?;
        // Cast back to original dtype for the output projection
        let quantized = quantized.to_dtype(self.out_proj.dtype())?;
        // Project back to output dim
        let out = linear(&quantized, &self.out_proj)?;
        Ok(out)
    }
}
