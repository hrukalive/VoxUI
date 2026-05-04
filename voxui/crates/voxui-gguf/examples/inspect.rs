use std::path::PathBuf;
use voxui_gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: inspect <path.gguf>");

    println!("Opening: {}", path.display());
    let gguf = GgufFile::open(&path)?;

    println!("\n=== Metadata ({} entries) ===", gguf.metadata.len());
    let mut keys: Vec<_> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let val = &gguf.metadata[key];
        println!("  {}: {:?}", key, val);
    }

    println!("\n=== Tensors ({} total) ===", gguf.tensors.len());
    for (i, t) in gguf.tensors.iter().take(5).enumerate() {
        println!(
            "  [{}] {} shape={:?} dtype={} size={}",
            i, t.name, t.shape, t.dtype, t.data_size
        );
    }
    if gguf.tensors.len() > 5 {
        println!("  ... and {} more", gguf.tensors.len() - 5);
    }

    // Dequantize first tensor
    if let Some(first) = gguf.tensors.first() {
        println!("\n=== Dequantizing '{}' ===", first.name);
        let data = gguf.tensor_f32(&first.name)?;
        let n = data.len().min(10);
        println!("  First {} values: {:?}", n, &data[..n]);
        println!("  Total elements: {}", data.len());
    }

    Ok(())
}
