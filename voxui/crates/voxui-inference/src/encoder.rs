//! Local encoder (VoxCPMLocEnc) — non-causal MiniCPM transformer that encodes
//! predicted latent features into a CLS embedding for the base LM.

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::base_lm::{BaseLM, BaseLMConfig};
use crate::model_loader::GgufModelLoader;

/// The local encoder: in_proj + special_token + non-causal transformer.
pub struct LocalEncoder {
    in_proj: Tensor,        // [enc_hidden, 64]
    special_token: Tensor,  // [1, 1, enc_hidden]
    transformer: BaseLM,
}

impl LocalEncoder {
    /// Load encoder weights from a GGUF file.
    pub fn load(loader: &GgufModelLoader, config: BaseLMConfig, device: &Device) -> Result<Self> {
        let in_proj = loader.load_tensor_optimal("encoder.in_proj.weight")?;
        let special_token = loader.load_tensor_optimal("encoder.special_token")?;
        // Reshape from [1,1,1,hidden] to [1,1,hidden]
        let hidden = config.hidden_size;
        let special_token = special_token.reshape(&[1, 1, hidden])?;
        let transformer = BaseLM::load(loader, config, device)?;
        Ok(Self { in_proj, special_token, transformer })
    }

    /// Encode predicted features into a CLS embedding.
    /// feat: [B, 64, P] -> [B, 1, enc_hidden]
    pub fn encode(&mut self, feat: &Tensor) -> Result<Tensor> {
        let (b, _d, _p) = feat.dims3()?;
        // Cast input to match model dtype (f16 on CUDA)
        let feat = feat.to_dtype(self.in_proj.dtype())?;
        let feat_t = feat.transpose(1, 2)?; // [B, P, 64]
        // Project to hidden dim
        let projected = crate::linear(&feat_t, &self.in_proj)?; // [B, P, enc_hidden]
        // Prepend special_token (CLS)
        let cls = self.special_token.broadcast_as((b, 1, projected.dim(2)?))?;
        let cls = cls.contiguous()?;
        let input = Tensor::cat(&[&cls, &projected], 1)?; // [B, P+1, enc_hidden]
        // Run transformer (non-causal: reset cache so seq_len>1 path is used,
        // but we need full attention. The is_causal=false flag disables causal masking.)
        self.transformer.reset_cache();
        let output = self.transformer.forward_embed(&input)?; // [B, P+1, enc_hidden]
        // Take CLS token output (position 0)
        let cls_out = output.narrow(1, 0, 1)?; // [B, 1, enc_hidden]
        Ok(cls_out)
    }
}
