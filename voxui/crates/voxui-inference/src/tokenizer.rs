use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub struct VoxTokenizer {
    inner: Tokenizer,
    multichar_chinese_tokens: HashSet<String>,
}

impl VoxTokenizer {
    /// Load tokenizer from a directory containing tokenizer.json
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let multichar_chinese_tokens = tokenizer
            .get_vocab(true)
            .into_keys()
            .filter(|token| is_pure_multichar_chinese(token))
            .collect();
        Ok(Self {
            inner: tokenizer,
            multichar_chinese_tokens,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        let mut ids = Vec::new();
        for (token, original_id) in encoding.get_tokens().iter().zip(encoding.get_ids()) {
            let clean_token = token.replace("\u{2581}", "");
            if self.multichar_chinese_tokens.contains(&clean_token) {
                for ch in clean_token.chars() {
                    let ch_token = ch.to_string();
                    let id = self.inner.token_to_id(&ch_token).with_context(|| {
                        format!(
                            "Chinese character token `{ch_token}` is missing from tokenizer vocab"
                        )
                    })?;
                    ids.push(id);
                }
            } else {
                ids.push(*original_id);
            }
        }
        Ok(ids)
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

fn is_pure_multichar_chinese(token: &str) -> bool {
    let mut count = 0usize;
    for ch in token.chars() {
        count += 1;
        if !is_cjk_unified_ideograph(ch) {
            return false;
        }
    }
    count >= 2
}

fn is_cjk_unified_ideograph(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
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
