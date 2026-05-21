# VoxCPM Inference Parity Design

## Goal

Fix native Rust inference parity for VoxCPM 0.5, VoxCPM 1.5, and VoxCPM2 Chinese LoRA inference without changing GGUF bundle layout or broadening the public API more than necessary.

## Context

The Rust project lives in `voxui/`. The Python references live in `VoxCPM/src/voxcpm/model/`, with separate model classes for VoxCPM 0.5/1.5 (`voxcpm.py`) and VoxCPM2 (`voxcpm2.py`). GGUF exports for all target variants are already present in `models/`.

Initial investigation showed:

- Rust can load the 0.5 and 1.5 GGUF bundles on CUDA, including their LoRA adapters.
- Current 0.5/1.5 CUDA runs produce finite audio, but both run to the bounded maximum generation length instead of stopping early. That is consistent with loaded weights being driven through a subtly wrong inference graph.
- Python wraps every model tokenizer with `mask_multichar_chinese_tokens`. Rust currently calls `Tokenizer::encode` directly.
- Python VoxCPM 0.5/1.5 DiT and VoxCPM2 DiT differ. V1 uses one conditioning token computed as `mu + t`; V2 reshapes `mu` into one or more tokens and appends `t` as a separate token.

## Approach

Implement two focused parity fixes:

1. Add Chinese tokenizer parity in Rust by reproducing Python's `mask_multichar_chinese_tokens` behavior for pure multi-character CJK tokens in the tokenizer vocabulary.
2. Add variant-aware DiT conditioning so VoxCPM 0.5/1.5 use V1 conditioning and VoxCPM2 keeps the current V2 conditioning.

This keeps existing model discovery, GGUF parsing, tensor names, LoRA loading, and UI model selection unchanged.

## Alternatives Considered

### Option A: Tokenizer Fix Only

This is too narrow. It addresses a clear Chinese mismatch but does not explain 0.5/1.5 max-length behavior or the Python reference split between `voxcpm.py` and `voxcpm2.py`.

### Option B: Full Architecture Split

Create separate Rust model structs for VoxCPM and VoxCPM2. This is architecturally clean, but too broad for the first repair because the current code already branches many V1/V2 differences and the immediate mismatch appears localized to tokenizer and DiT conditioning.

### Option C: Focused Variant Branches

Add explicit, tested branches for the discovered mismatches while leaving the common engine intact. This is the recommended approach because it is small, testable, and matches the current codebase style.

## Components

### Tokenizer

Modify `voxui/crates/voxui-inference/src/tokenizer.rs`.

`VoxTokenizer::from_dir` should precompute the set of vocabulary entries whose visible token text is at least two Unicode scalar values long and whose characters are all in the CJK Unified Ideographs range `U+4E00..=U+9FFF`.

`VoxTokenizer::encode` should follow Python reference behavior:

- Tokenize the input text without adding special tokens.
- For each token, remove a possible SentencePiece leading marker from a copy used only for matching.
- If the cleaned token is a pure multi-character Chinese vocab token, split it into single-character tokens.
- Convert the resulting token sequence to ids.
- Preserve existing direct tokenization behavior for Japanese, English, punctuation, mixed CJK/non-CJK tokens, and single Chinese characters.

Error handling should return an `anyhow::Error` if a split token cannot be converted to an id.

### DiT Conditioning

Modify `voxui/crates/voxui-inference/src/dit.rs` and the call site in `voxui/crates/voxui-inference/src/engine.rs`.

Add an internal DiT conditioning mode:

- `VoxCpm`: V1 mode for variants 0.5 and 1.5.
- `VoxCpm2`: V2 mode for variant 2.0.

For V1 mode, `DiT::forward` should build the decoder sequence as:

```text
[(mu + t), cond_tokens..., x_tokens...]
```

The hidden extraction should skip `1 + cond_len` prefix tokens.

For V2 mode, keep the current behavior:

```text
[mu_tokens..., t, cond_tokens..., x_tokens...]
```

The hidden extraction should skip `mu_token_count + 1 + cond_len` prefix tokens.

The public generation methods should continue to accept the same inputs. The mode is selected at load time from `ModelConfig.variant`.

### Tests

Add focused tests before implementation:

- Tokenizer parity test using a Chinese phrase known to differ between direct tokenizer encode and Python wrapper behavior. The expected Rust ids should match the Python wrapper output.
- Tokenizer non-regression test for the existing Chinese sentence where direct and wrapped tokenization already match, plus Japanese and English examples.
- DiT unit test that exercises sequence construction/prefix sizing for V1 and V2 modes without requiring full CUDA generation.
- Existing VoxCPM2 parity tests must continue to pass.

Run CUDA q4 inference after the focused tests pass:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test inference_suite q4_lm_cuda -- --nocapture --test-threads=1
```

## Acceptance Criteria

- Rust tokenizer output matches the Python wrapper for multi-character Chinese vocabulary tokens.
- Existing English and Japanese tokenizer behavior remains unchanged.
- VoxCPM2 golden parity tests continue to pass.
- VoxCPM 0.5 and 1.5 run through their V1 DiT conditioning path.
- CUDA q4 inference for 0.5, 1.5, and 2.0 loads and generates finite non-silent audio with LoRA.
- No exporter or GGUF layout change is required.

## Out of Scope

- Re-exporting model bundles.
- Adding a full Rust text normalizer.
- Rewriting the engine into separate top-level VoxCPM and VoxCPM2 structs.
- Changing desktop UI model discovery.
- CPU performance optimization.
