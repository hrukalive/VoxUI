//! Inference engine for VoxUI.

pub mod audio_io;
pub mod audiovae;
pub mod base_lm;
pub mod dit;
pub mod encoder;
pub mod engine;
pub mod fsq;
pub mod lora;
mod manifest;
pub mod model_loader;
pub mod request;
pub mod residual_lm;
pub mod tokenizer;
pub mod trace;

pub use audiovae::AudioVAE;
pub use base_lm::{BaseLM, BaseLMConfig};
pub use dit::DiT;
pub use encoder::LocalEncoder;
pub use engine::VoxCPMEngine;
pub use lora::LoraAdapter;
pub use manifest::{AudioVaeManifest, ModelConfig, ModelVariant, SpecialTokens};
pub use model_loader::GgufModelLoader;
pub use request::SynthesisRequest;
pub use tokenizer::VoxTokenizer;

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Softmax on the last dimension, implemented with standard ops for CUDA compatibility.
/// candle_nn::ops::softmax_last_dim only has a CPU impl (CustomOp1 without cuda_fwd).
pub(crate) fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    // Compute in f32 for numerical stability
    let x = x.to_dtype(DType::F32)?;
    let max = x.max_keepdim(candle_core::D::Minus1)?;
    let x = x.broadcast_sub(&max)?;
    let exp = x.exp()?;
    let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
    let out = exp.broadcast_div(&sum)?;
    out.to_dtype(dtype).map_err(Into::into)
}

/// Matrix multiply that handles 3D input with 2D weight.
/// x: [B, T, in] or [B*T, in], w: [in, out] -> [..., out]
pub(crate) fn matmul_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() <= 2 {
        x.matmul(w).map_err(Into::into)
    } else if dims.len() == 3 {
        let (b, t, in_dim) = (dims[0], dims[1], dims[2]);
        let out_dim = w.dim(1)?;
        let flat = x.reshape(&[b * t, in_dim])?;
        let result = flat.matmul(w)?;
        result.reshape(&[b, t, out_dim]).map_err(Into::into)
    } else {
        x.matmul(w).map_err(Into::into)
    }
}

/// Linear: x @ weight^T. Handles 2D and 3D inputs.
/// weight: [out, in], x: [..., in] -> [..., out]
pub(crate) fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let w = weight.t()?;
    matmul_2d(x, &w)
}
