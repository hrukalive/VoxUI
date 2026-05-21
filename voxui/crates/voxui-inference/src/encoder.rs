//! Local encoder (VoxCPMLocEnc) — non-causal MiniCPM transformer that encodes
//! predicted latent features into a CLS embedding for the base LM.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::base_lm::{BaseLM, BaseLMConfig};
use crate::model_loader::GgufModelLoader;
use crate::{LinearWeight, RuntimeTensor};

/// The local encoder: in_proj + special_token + non-causal transformer.
pub struct LocalEncoder {
    in_proj: LinearWeight,        // [enc_hidden, 64]
    in_proj_bias: RuntimeTensor,  // [enc_hidden]
    special_token: RuntimeTensor, // [1, 1, enc_hidden]
    transformer: BaseLM,
    hidden_size: usize,
}

impl LocalEncoder {
    pub fn load_from_config(loader: &GgufModelLoader, model: &crate::ModelConfig) -> Result<Self> {
        let config = BaseLMConfig::from_model_config(model, "feat_encoder")?;
        Self::load(loader, config, loader.device())
    }

    /// Load encoder weights from a GGUF file.
    pub fn load(loader: &GgufModelLoader, config: BaseLMConfig, device: &Device) -> Result<Self> {
        let in_proj = loader.load_linear_weight("feat_encoder.in_proj.weight")?;
        let in_proj_bias = loader.load_runtime_tensor("feat_encoder.in_proj.bias")?;
        let special_token = loader.load_runtime_tensor("feat_encoder.special_token")?;
        let hidden = config.hidden_size;
        let transformer = BaseLM::load(loader, config, device)?;
        Ok(Self {
            in_proj,
            in_proj_bias,
            special_token,
            transformer,
            hidden_size: hidden,
        })
    }

    /// Encode latent patches `[B, T, P, D]` into `[B, T, hidden]`.
    pub fn encode_patches(&mut self, feat: &Tensor) -> Result<Tensor> {
        let (b, t, p, d) = feat.dims4()?;
        let feat = feat.to_dtype(DType::F32)?;
        let flat = feat.reshape((b * t, p, d))?;
        let projected = self.in_proj.forward(&flat)?;
        let bias = self
            .in_proj_bias
            .to_dense_dtype(projected.dtype())?
            .reshape((1, 1, self.hidden_size))?;
        let projected = projected.broadcast_add(&bias)?;
        let cls = self
            .special_token
            .to_dense_dtype(projected.dtype())?
            .reshape(&[1, 1, self.hidden_size])?
            .broadcast_as((b * t, 1, self.hidden_size))?;
        let cls = cls.contiguous()?;
        let input = Tensor::cat(&[&cls, &projected], 1)?;
        self.transformer.reset_cache();
        let output = self.transformer.forward_embed(&input)?;
        output
            .narrow(1, 0, 1)?
            .reshape((b, t, self.hidden_size))
            .map_err(Into::into)
    }

    /// Compatibility helper for a single `[B, D, P]` latent patch sequence.
    pub fn encode_single_patch(&mut self, feat: &Tensor) -> Result<Tensor> {
        let feat = feat.transpose(1, 2)?.unsqueeze(1)?;
        self.encode_patches(&feat)
    }

    /// Encode predicted features into a CLS embedding.
    /// feat: [B, 64, P] -> [B, 1, enc_hidden]
    pub fn encode(&mut self, feat: &Tensor) -> Result<Tensor> {
        self.encode_single_patch(feat)
    }
}
