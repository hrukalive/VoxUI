use crate::base_lm::{BaseLM, BaseLMConfig};
use crate::model_loader::GgufModelLoader;
use candle_core::Device;

/// Residual LM — same architecture as BaseLM but with:
/// - "residual_lm." tensor name prefix
/// - Potentially no RoPE
/// - Fewer layers
pub type ResidualLM = BaseLM;

/// Load the residual LM from its GGUF file.
pub fn load_residual_lm(
    loader: &GgufModelLoader,
    config: BaseLMConfig,
    device: &Device,
) -> anyhow::Result<BaseLM> {
    BaseLM::load(loader, config, device)
}
