use std::path::Path;

use anyhow::Result;
use tokenizers::Tokenizer;

pub struct VoxTokenizer {
    inner: Tokenizer,
}

impl VoxTokenizer {
    /// Load tokenizer from a directory containing tokenizer.json
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner: tokenizer })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_load_and_encode() {
        let dir = PathBuf::from(r"D:\Sandbox_Share\VoxUI\VoxCPM\models\VoxCPM2");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
        let tok = VoxTokenizer::from_dir(&dir).expect("failed to load tokenizer");
        assert!(tok.vocab_size() > 0);

        let ids = tok.encode("Hello, world!").expect("encode failed");
        assert!(!ids.is_empty());

        let text = tok.decode(&ids).expect("decode failed");
        assert!(text.contains("Hello"));
    }
}
