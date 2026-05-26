use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use voxui_gguf::MetadataValue;
use voxui_inference::base_lm::{BaseLM, BaseLMConfig};
use voxui_inference::model_loader::GgufModelLoader;
use voxui_inference::VoxTokenizer;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load GGUF file
    let gguf_path = Path::new(r"D:\Sandbox_Share\VoxUI\models\voxcpm2-q4-lm\model.gguf");
    println!("Loading GGUF from: {}", gguf_path.display());
    let loader = GgufModelLoader::new(gguf_path, device.clone())?;
    println!("GGUF loaded. Tensor count: {}", loader.tensor_names().len());

    // Print metadata
    let meta = loader.metadata();
    println!("\nMetadata keys:");
    for (k, v) in meta {
        println!("  {k}: {v:?}");
    }

    // 2. Build config from metadata
    let prefix = "base_lm";
    let get_u32 = |key: &str, default: u32| -> u32 {
        meta.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        meta.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    };

    let hidden_size = get_u32(&format!("{prefix}.hidden_size"), 2048) as usize;
    let num_layers = get_u32(&format!("{prefix}.num_layers"), 28) as usize;
    let num_heads = get_u32(&format!("{prefix}.num_heads"), 16) as usize;
    let num_kv_heads = get_u32(&format!("{prefix}.num_kv_heads"), 2) as usize;
    let head_dim = get_u32(&format!("{prefix}.head_dim"), 128) as usize;
    let intermediate_size = get_u32(&format!("{prefix}.intermediate_size"), 6144) as usize;
    let rms_norm_eps = get_f32(&format!("{prefix}.rms_norm_eps"), 1e-5) as f64;
    let rope_theta = get_f32(&format!("{prefix}.rope_theta"), 10000.0) as f64;
    let vocab_size = get_u32(&format!("{prefix}.vocab_size"), 73448) as usize;

    let rope_factors = if let Some(MetadataValue::ArrayFloat32(factors)) =
        meta.get(&format!("{prefix}.rope_factors"))
    {
        factors.clone()
    } else {
        vec![1.0; head_dim / 2]
    };

    let config = BaseLMConfig {
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps,
        rope_theta,
        rope_factors: rope_factors.clone(),
        use_mup: false,
        scale_emb: 1.0,
        scale_depth: 1.0,
        original_max_position_embeddings: None,
        rope_short_factors: rope_factors.clone(),
        rope_long_factors: rope_factors,
        vocab_size,
        max_position: 4096,
        prefix: prefix.to_string(),
        no_rope: false,
        is_causal: true,
    };

    println!("\nBaseLMConfig:");
    println!("  hidden_size: {hidden_size}");
    println!("  num_layers: {num_layers}");
    println!("  num_heads: {num_heads}");
    println!("  num_kv_heads: {num_kv_heads}");
    println!("  head_dim: {head_dim}");
    println!("  intermediate_size: {intermediate_size}");
    println!("  vocab_size: {vocab_size}");

    // 3. Load BaseLM model (dequantizes all tensors to f32)
    println!("\nLoading BaseLM model (dequantizing all tensors to f32)...");
    let start = std::time::Instant::now();
    let mut model = BaseLM::load(&loader, config, &device)?;
    let elapsed = start.elapsed();
    println!(
        "BaseLM loaded successfully! {} layers in {:.2}s",
        num_layers,
        elapsed.as_secs_f64()
    );

    // 4. Load tokenizer
    let tokenizer_dir = Path::new(r"D:\Sandbox_Share\VoxUI\VoxCPM\models\VoxCPM2");
    println!("\nLoading tokenizer from: {}", tokenizer_dir.display());
    let tokenizer = VoxTokenizer::from_dir(tokenizer_dir)?;
    println!("Tokenizer loaded. Vocab size: {}", tokenizer.vocab_size());

    // 5. Encode test string
    let test_text = "你好世界";
    let token_ids = tokenizer.encode(test_text)?;
    println!("\nEncoded \"{test_text}\" -> token IDs: {token_ids:?}");

    // Decode back to verify
    let decoded = tokenizer.decode(&token_ids)?;
    println!("Decoded back: \"{decoded}\"");

    // 6. Test embedding first
    println!("\nTesting embedding...");
    let embed = model.embed(&token_ids)?;
    println!(
        "Embedding shape: {:?}, dtype: {:?}",
        embed.shape(),
        embed.dtype()
    );

    // 7. Forward pass (full model)
    println!("\nRunning forward pass with {} tokens...", token_ids.len());
    let start = std::time::Instant::now();
    match model.forward(&token_ids) {
        Ok(hidden) => {
            let elapsed = start.elapsed();
            println!(
                "Forward pass complete in {:.2}s. Output shape: {:?}",
                elapsed.as_secs_f64(),
                hidden.shape()
            );
            let first_vals: Vec<f32> = hidden
                .flatten_all()?
                .narrow(0, 0, 8.min(hidden.elem_count()))?
                .to_vec1()?;
            println!("First output values: {:?}", first_vals);
        }
        Err(e) => {
            println!("Forward pass failed (expected during development): {e}");
            println!(
                "This likely indicates a tensor shape mismatch that needs fixing in base_lm.rs"
            );
        }
    }

    println!("\nTest complete!");
    Ok(())
}
