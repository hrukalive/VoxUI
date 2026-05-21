# VoxCPM Inference Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Rust native inference match Python reference behavior for VoxCPM 0.5/1.5 DiT conditioning and VoxCPM2 Chinese LoRA tokenization.

**Architecture:** Keep the existing shared `VoxCPMEngine`, but add focused variant branches where the Python references differ. `VoxTokenizer` owns Chinese token splitting parity; `DiTConfig` owns a variant-derived conditioning mode used inside `DiT::forward`.

**Tech Stack:** Rust 2021, Candle, `tokenizers`, existing `voxui-inference` and `voxui-gguf` crates, CUDA feature tests.

---

## File Structure

- Modify `voxui/crates/voxui-inference/src/tokenizer.rs`: store pure multi-character Chinese vocab tokens and encode through token-level parity logic.
- Create `voxui/crates/voxui-inference/tests/tokenizer_parity.rs`: verify Rust tokenizer ids against Python wrapper-derived expected ids and direct-tokenizer non-regression cases.
- Modify `voxui/crates/voxui-inference/src/dit.rs`: add `DiTConditioningMode`, select it from `ModelVariant`, and branch sequence construction in `DiT::forward`.
- No changes to `voxui/crates/voxui-inference/src/engine.rs` are expected because `DiT::load_from_config` already receives `ModelConfig`.
- No changes to `voxui/crates/voxui-gguf`, exporter code, model bundles, or desktop UI.

---

### Task 1: Tokenizer Chinese Parity

**Files:**
- Modify: `voxui/crates/voxui-inference/src/tokenizer.rs`
- Create: `voxui/crates/voxui-inference/tests/tokenizer_parity.rs`

- [ ] **Step 1: Write the failing tokenizer parity tests**

Create `voxui/crates/voxui-inference/tests/tokenizer_parity.rs`:

```rust
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use voxui_inference::VoxTokenizer;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn model_dir() -> PathBuf {
    repo_root().join("models/voxcpm2-fp16")
}

fn direct_encode(text: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(model_dir().join("tokenizer.json")).unwrap();
    tokenizer.encode(text, false).unwrap().get_ids().to_vec()
}

#[test]
fn splits_multichar_chinese_tokens_like_python_reference() {
    let tokenizer = VoxTokenizer::from_dir(&model_dir()).unwrap();
    let text = "第一百五十四条裁定适用于下列范围";

    let actual = tokenizer.encode(text).unwrap();

    assert_eq!(direct_encode(text), vec![59320, 47804]);
    assert_eq!(
        actual,
        vec![
            59320, 59438, 59382, 59635, 59637, 59482, 59614, 59548, 59659, 59421,
            59823, 59415, 59433, 59454, 59913, 59951, 60016,
        ]
    );
}

#[test]
fn keeps_existing_direct_tokenization_when_python_wrapper_would_match() {
    let tokenizer = VoxTokenizer::from_dir(&model_dir()).unwrap();
    let cases = [
        "我说什么来着，我不知道你是什么脾气啊，我肯定要邦邦敲一下。",
        "これはテストですなの！なんでだよ？",
        "This inference matrix sentence exercises q4 language model coverage on every backend.",
    ];

    for text in cases {
        assert_eq!(tokenizer.encode(text).unwrap(), direct_encode(text), "{text}");
    }
}
```

- [ ] **Step 2: Run the tokenizer tests and verify the expected failure**

Run:

```powershell
cargo test -p voxui-inference --test tokenizer_parity
```

Expected before implementation:

```text
test splits_multichar_chinese_tokens_like_python_reference ... FAILED
```

The failure should show Rust returning direct ids `[59320, 47804]` instead of the expanded Python-wrapper ids.

- [ ] **Step 3: Implement tokenizer parity**

Replace the top of `voxui/crates/voxui-inference/src/tokenizer.rs` with these imports and struct fields:

```rust
use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub struct VoxTokenizer {
    inner: Tokenizer,
    multichar_chinese_tokens: HashSet<String>,
}
```

Update `VoxTokenizer::from_dir`:

```rust
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
```

Update `VoxTokenizer::encode`:

```rust
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
                let id = self
                    .inner
                    .token_to_id(&ch_token)
                    .with_context(|| format!("Chinese character token `{ch_token}` is missing from tokenizer vocab"))?;
                ids.push(id);
            }
        } else {
            ids.push(*original_id);
        }
    }
    Ok(ids)
}
```

Add helper functions near the bottom of the non-test module:

```rust
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
```

- [ ] **Step 4: Run tokenizer tests and existing lightweight inference tests**

Run:

```powershell
cargo test -p voxui-inference --test tokenizer_parity
cargo test -p voxui-inference --test manifest_loader
```

Expected:

```text
test result: ok
```

- [ ] **Step 5: Commit tokenizer parity**

Run:

```powershell
git add voxui/crates/voxui-inference/src/tokenizer.rs voxui/crates/voxui-inference/tests/tokenizer_parity.rs
git commit -m "fix(inference): match VoxCPM Chinese tokenizer parity"
```

---

### Task 2: Variant-Aware DiT Conditioning

**Files:**
- Modify: `voxui/crates/voxui-inference/src/dit.rs`

- [ ] **Step 1: Write failing DiT conditioning unit tests**

Add this test module at the bottom of `voxui/crates/voxui-inference/src/dit.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dit_conditioning_prefix_len_uses_v1_single_mu_plus_time_token() {
        let prefix = conditioning_prefix_len(DiTConditioningMode::VoxCpm, 1024, 1024, 3).unwrap();
        assert_eq!(prefix, 4);
    }

    #[test]
    fn dit_conditioning_prefix_len_uses_v2_mu_tokens_plus_time_token() {
        let prefix = conditioning_prefix_len(DiTConditioningMode::VoxCpm2, 2048, 1024, 3).unwrap();
        assert_eq!(prefix, 6);
    }

    #[test]
    fn dit_conditioning_rejects_v1_multi_token_mu() {
        let err = conditioning_prefix_len(DiTConditioningMode::VoxCpm, 2048, 1024, 3).unwrap_err();
        assert!(err.to_string().contains("VoxCPM DiT expects one mu token"));
    }

    #[test]
    fn dit_conditioning_rejects_non_divisible_v2_mu() {
        let err = conditioning_prefix_len(DiTConditioningMode::VoxCpm2, 1536, 1024, 3).unwrap_err();
        assert!(err.to_string().contains("mu dimension"));
    }
}
```

- [ ] **Step 2: Run the DiT unit tests and verify the expected failure**

Run:

```powershell
cargo test -p voxui-inference dit_conditioning
```

Expected before implementation:

```text
error[E0425]: cannot find function `conditioning_prefix_len`
```

- [ ] **Step 3: Add the conditioning mode type and config field**

In `voxui/crates/voxui-inference/src/dit.rs`, add this import near the top:

```rust
use crate::manifest::ModelVariant;
```

Add this enum before `DiTConfig`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiTConditioningMode {
    VoxCpm,
    VoxCpm2,
}

impl DiTConditioningMode {
    fn from_variant(variant: ModelVariant) -> Self {
        match variant {
            ModelVariant::VoxCpm05 | ModelVariant::VoxCpm15 => Self::VoxCpm,
            ModelVariant::VoxCpm2 => Self::VoxCpm2,
        }
    }
}
```

Add this field to `DiTConfig`:

```rust
pub conditioning_mode: DiTConditioningMode,
```

Set it in `Default for DiTConfig`:

```rust
conditioning_mode: DiTConditioningMode::VoxCpm2,
```

Set it in `DiT::load_from_config`:

```rust
conditioning_mode: DiTConditioningMode::from_variant(model.variant),
```

- [ ] **Step 4: Add the prefix helper used by tests and forward**

Add this helper near the other private helper functions in `dit.rs`:

```rust
fn conditioning_prefix_len(
    mode: DiTConditioningMode,
    mu_dim: usize,
    hidden_dim: usize,
    cond_len: usize,
) -> Result<usize> {
    if hidden_dim == 0 || mu_dim == 0 || mu_dim % hidden_dim != 0 {
        anyhow::bail!(
            "DiT mu dimension {mu_dim} must be a positive multiple of hidden dimension {hidden_dim}"
        );
    }
    let n_mu_tokens = mu_dim / hidden_dim;
    match mode {
        DiTConditioningMode::VoxCpm => {
            if n_mu_tokens != 1 {
                anyhow::bail!("VoxCPM DiT expects one mu token, got {n_mu_tokens}");
            }
            Ok(1 + cond_len)
        }
        DiTConditioningMode::VoxCpm2 => Ok(n_mu_tokens + 1 + cond_len),
    }
}
```

- [ ] **Step 5: Branch `DiT::forward` sequence construction**

In `DiT::forward`, replace the existing block that computes `mu_tokens`, `t_token`, `hidden`, and `prefix_len` with:

```rust
let mu_dim = mu.dim(1)?;
let prefix_len = conditioning_prefix_len(
    self.config.conditioning_mode,
    mu_dim,
    self.config.hidden_dim,
    cond_len,
)?;
let conditioning_tokens = match self.config.conditioning_mode {
    DiTConditioningMode::VoxCpm => (mu + &t_emb)?.unsqueeze(1)?,
    DiTConditioningMode::VoxCpm2 => {
        let n_mu_tokens = mu_dim / self.config.hidden_dim;
        let mu_tokens = mu.reshape((b, n_mu_tokens, self.config.hidden_dim))?;
        let t_token = t_emb.unsqueeze(1)?;
        Tensor::cat(&[&mu_tokens, &t_token], 1)?
    }
};

let mut hidden = Tensor::cat(&[&conditioning_tokens, &cond_proj, &x_proj], 1)?;
let total_len = prefix_len + t_len;
```

- [ ] **Step 6: Run DiT and VoxCPM2 parity tests**

Run:

```powershell
cargo test -p voxui-inference dit_conditioning
cargo test -p voxui-inference --test dit_parity
cargo test -p voxui-inference --test generate_flow_parity
```

Expected:

```text
test result: ok
```

- [ ] **Step 7: Commit DiT conditioning parity**

Run:

```powershell
git add voxui/crates/voxui-inference/src/dit.rs
git commit -m "fix(inference): use variant-specific DiT conditioning"
```

---

### Task 3: CUDA Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run full focused Rust test set**

Run:

```powershell
cargo test -p voxui-inference --test tokenizer_parity
cargo test -p voxui-inference --test manifest_loader
cargo test -p voxui-inference --test dit_parity
cargo test -p voxui-inference --test generate_flow_parity
```

Expected:

```text
test result: ok
```

- [ ] **Step 2: Run CUDA q4 inference matrix**

Run:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test inference_suite q4_lm_cuda -- --nocapture --test-threads=1
```

Expected:

```text
test voxcpm05_q4_lm_cuda ... ok
test voxcpm15_q4_lm_cuda ... ok
test voxcpm2_q4_lm_cuda ... ok
```

Inspect the printed generation step counts. VoxCPM2 should still stop early for the current English/Japanese/Chinese LoRA cases. VoxCPM 0.5/1.5 must generate finite non-silent audio; if they still hit the bounded max step count, record that as remaining parity risk rather than hiding it.

- [ ] **Step 3: Run final status check**

Run:

```powershell
git status --short
```

Expected:

```text
 M README.txt
```

Only the pre-existing `README.txt` change should remain unstaged if all implementation commits were made.
