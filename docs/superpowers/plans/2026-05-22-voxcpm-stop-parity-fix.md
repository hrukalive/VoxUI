# VoxCPM Stop Parity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Rust VoxCPM 0.5 generation stop at the same step as the Python reference for a traced prompt, then verify VoxCPM 1.5 and VoxCPM2 do not regress.

**Architecture:** Add a dedicated Python stop trace for VoxCPM 0.5 that records per-step stop logits, hidden states, and generated patch summaries. Add an internal Rust debug trace path that records the same semantic points from `VoxCPMEngine`, then compare the two traces in a focused parity test and patch the first confirmed mismatch.

**Tech Stack:** Rust 2021, Candle, existing `voxui-inference` trace helpers, Python VoxCPM reference under `VoxCPM/`, NumPy/Torch via `C:\Users\Reon\py_env\voxcpm`.

---

## File Structure

- Modify `tools/golden_trace/voxcpm_trace.py`: add a `--trace-kind stop` mode for VoxCPM 0.5/1.5 generation-loop traces.
- Modify `voxui/crates/voxui-inference/src/trace.rs`: parse trace `request`, `metadata`, and stop-decision lists from `trace.json`.
- Modify `voxui/crates/voxui-inference/src/engine.rs`: add internal debug structs and a `generate_debug_stop_trace_with_noise` helper that records per-step generation state.
- Add `voxui/crates/voxui-inference/tests/stop_parity.rs`: compare Rust stop trace against `goldens/voxcpm05_stop_parity`.
- Generate `goldens/voxcpm05_stop_parity/*`: Python-owned trace metadata and tensors for the target prompt.
- Modify the first confirmed Rust mismatch only. The likely files are `voxui/crates/voxui-inference/src/engine.rs` or `voxui/crates/voxui-inference/src/base_lm.rs`.

---

### Task 1: Add Python VoxCPM 0.5 Stop Trace

**Files:**
- Modify: `tools/golden_trace/voxcpm_trace.py`
- Generated: `goldens/voxcpm05_stop_parity/trace.json`
- Generated: `goldens/voxcpm05_stop_parity/*.f32`

- [ ] **Step 1: Add stop-trace arguments**

In `tools/golden_trace/voxcpm_trace.py`, add these parser arguments after `--runtime-dtype`:

```python
    parser.add_argument("--trace-kind", choices=["first_patch", "stop"], default="first_patch")
    parser.add_argument("--stop-max-len", type=int, default=120)
    parser.add_argument("--stop-min-len", type=int, default=2)
```

- [ ] **Step 2: Add tensor summary helpers**

Add these helpers below `to_numpy`:

```python
def tensor_stats(tensor: Any) -> np.ndarray:
    arr = to_numpy(tensor).reshape(-1)
    if arr.size == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(
        [arr.mean(), arr.std(), arr.min(), arr.max()],
        dtype=np.float32,
    )


def stack_or_empty(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    return np.stack([np.asarray(row, dtype=np.float32).reshape(width) for row in rows], axis=0)
```

- [ ] **Step 3: Add a VoxCPM v1 stop trace runner**

Add this function above `main()`:

```python
@torch.inference_mode()
def run_v1_stop_trace(
    model: Any,
    *,
    target_text: str,
    min_len: int,
    max_len: int,
    inference_timesteps: int,
    cfg_value: float,
) -> dict[str, Any]:
    text_token = torch.LongTensor(model.text_tokenizer(target_text))
    text_token = torch.cat(
        [
            text_token,
            torch.tensor([model.audio_start_token], dtype=torch.int32, device=text_token.device),
        ],
        dim=-1,
    )
    text_length = text_token.shape[0]
    audio_feat = torch.zeros(
        (text_length, model.patch_size, model.audio_vae.latent_dim),
        dtype=torch.float32,
        device=text_token.device,
    )
    text_mask = torch.ones(text_length).type(torch.int32).to(text_token.device)
    audio_mask = torch.zeros(text_length).type(torch.int32).to(text_token.device)

    text = text_token.unsqueeze(0).to(model.device)
    text_mask = text_mask.unsqueeze(0).to(model.device)
    feat = audio_feat.unsqueeze(0).to(model.device).to(get_dtype(model.config.dtype))
    feat_mask = audio_mask.unsqueeze(0).to(model.device)

    bsz, _seq, _patch, _dim = feat.shape
    feat_embed = model.feat_encoder(feat)
    feat_embed = model.enc_to_lm_proj(feat_embed)
    scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
    text_embed = model.base_lm.embed_tokens(text) * scale_emb
    combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
    prefix_feat_cond = feat[:, -1, ...]

    enc_outputs, kv_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
    model.base_lm.kv_cache.fill_caches(kv_cache_tuple)
    enc_outputs = model.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
    lm_hidden = enc_outputs[:, -1, :]

    residual_inputs = enc_outputs + feat_mask.unsqueeze(-1) * feat_embed
    residual_enc_outputs, residual_kv_cache_tuple = model.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
    model.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    residual_hidden = residual_enc_outputs[:, -1, :]

    stop_logits_rows: list[np.ndarray] = []
    stop_decisions: list[int] = []
    lm_hidden_stats: list[np.ndarray] = []
    residual_hidden_stats: list[np.ndarray] = []
    pred_feat_stats: list[np.ndarray] = []
    first_noise: np.ndarray | None = None
    pred_feat_seq = []

    original_randn = torch.randn

    def traced_randn(*args, **kwargs):
        nonlocal first_noise
        result = original_randn(*args, **kwargs)
        if first_noise is None:
            first_noise = to_numpy(result)
        return result

    torch.randn = traced_randn
    try:
        for step in range(max_len):
            dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
            dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
            dit_hidden = dit_hidden_1 + dit_hidden_2

            pred_feat = model.feat_decoder(
                mu=dit_hidden,
                patch_size=model.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(1, 2)

            curr_embed = model.feat_encoder(pred_feat.unsqueeze(1))
            curr_embed = model.enc_to_lm_proj(curr_embed)
            pred_feat_seq.append(pred_feat.unsqueeze(1))
            prefix_feat_cond = pred_feat

            stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
            stop_flag = int(stop_logits.argmax(dim=-1)[0].cpu().item())
            stop_logits_rows.append(to_numpy(stop_logits).reshape(2))
            stop_decisions.append(stop_flag)
            lm_hidden_stats.append(tensor_stats(lm_hidden))
            residual_hidden_stats.append(tensor_stats(residual_hidden))
            pred_feat_stats.append(tensor_stats(pred_feat))

            if step > min_len and stop_flag == 1:
                break

            lm_hidden = model.base_lm.forward_step(
                curr_embed[:, 0, :],
                torch.tensor([model.base_lm.kv_cache.step()], device=curr_embed.device),
            ).clone()
            lm_hidden = model.fsq_layer(lm_hidden)
            residual_hidden = model.residual_lm.forward_step(
                lm_hidden + curr_embed[:, 0, :],
                torch.tensor([model.residual_lm.kv_cache.step()], device=curr_embed.device),
            ).clone()
    finally:
        torch.randn = original_randn

    generated_feat = torch.cat(pred_feat_seq, dim=1).squeeze(0).cpu()
    latent = rearrange(torch.cat(pred_feat_seq, dim=1), "b t p d -> b d (t p)", b=bsz, p=model.patch_size)
    decoded = model.audio_vae.decode(latent.to(torch.float32)).squeeze(1).cpu()

    return {
        "token_ids": [int(v) for v in text.detach().cpu().reshape(-1).tolist()],
        "first_dit_noise": np.zeros((1, model.feat_dim, model.patch_size), dtype=np.float32)
        if first_noise is None
        else first_noise,
        "stop_logits_by_step": stack_or_empty(stop_logits_rows, 2),
        "stop_decisions": stop_decisions,
        "lm_hidden_stats_by_step": stack_or_empty(lm_hidden_stats, 4),
        "residual_hidden_stats_by_step": stack_or_empty(residual_hidden_stats, 4),
        "pred_feat_stats_by_step": stack_or_empty(pred_feat_stats, 4),
        "generated_audio_feat": to_numpy(generated_feat),
        "decoded_wav_head": to_numpy(decoded[:, :4096]),
        "generated_step_count": len(stop_decisions),
    }
```

- [ ] **Step 4: Branch `main()` for stop traces**

In `main()`, after `force_runtime_dtype(model, args.runtime_dtype)`, insert this branch before constructing `TraceCapture`:

```python
    if args.trace_kind == "stop":
        if args.variant not in {"0.5", "1.5"}:
            raise ValueError("--trace-kind stop currently supports VoxCPM 0.5 and 1.5")
        result = run_v1_stop_trace(
            model.tts_model,
            target_text=args.text,
            min_len=args.stop_min_len,
            max_len=args.stop_max_len,
            inference_timesteps=4,
            cfg_value=2.0,
        )
        writer = TraceWriter(args.out_dir, args.case_name)
        writer.write_u32_list("token_ids", result["token_ids"])
        writer.write_u32_list("stop_decisions", result["stop_decisions"])
        tensors = [
            writer.write_tensor("first_dit_noise", result["first_dit_noise"]),
            writer.write_tensor("stop_logits_by_step", result["stop_logits_by_step"]),
            writer.write_tensor("lm_hidden_stats_by_step", result["lm_hidden_stats_by_step"]),
            writer.write_tensor("residual_hidden_stats_by_step", result["residual_hidden_stats_by_step"]),
            writer.write_tensor("pred_feat_stats_by_step", result["pred_feat_stats_by_step"]),
            writer.write_tensor("generated_audio_feat", result["generated_audio_feat"]),
            writer.write_tensor("decoded_wav_head", result["decoded_wav_head"]),
        ]
        writer.write_manifest(
            variant=args.variant,
            architecture="voxcpm",
            request={
                "text": args.text,
                "prompt_wav_path": None,
                "prompt_text": None,
                "reference_wav_path": None,
                "cfg_value": 2.0,
                "inference_timesteps": 4,
                "min_len": args.stop_min_len,
                "max_len": args.stop_max_len,
                "normalize": False,
                "retry_badcase": False,
            },
            tensors=tensors,
            metadata={
                "seed": args.seed,
                "source_model_dir": str(args.model_dir.resolve()),
                "runtime_dtype": args.runtime_dtype,
                "generated_step_count": int(result["generated_step_count"]),
                "trace_kind": "stop",
            },
        )
        return
```

- [ ] **Step 5: Generate the VoxCPM 0.5 stop trace**

Run:

```powershell
& C:\Users\Reon\py_env\voxcpm\Scripts\activate.ps1
python tools\golden_trace\voxcpm_trace.py --repo-root . --model-dir models\voxcpm05-fp16 --variant 0.5 --case-name voxcpm05_stop_parity --out-dir goldens --text "Hello, welcome to the stream!" --seed 1234 --runtime-dtype float32 --trace-kind stop --stop-min-len 1 --stop-max-len 120
```

Expected:

```text
goldens\voxcpm05_stop_parity\trace.json exists
goldens\voxcpm05_stop_parity\stop_logits_by_step.f32 exists
```

- [ ] **Step 6: Inspect the Python stop point**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("goldens/voxcpm05_stop_parity/trace.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("steps", data["metadata"]["generated_step_count"])
print("decisions", data["lists"]["stop_decisions"])
print("request", data["request"])
'@ | python -
```

Expected:

```text
steps <a number less than 120>
decisions [...]
request {'text': 'Hello, welcome to the stream!', ...}
```

- [ ] **Step 7: Commit the Python stop trace tooling and golden**

Run:

```powershell
git add tools/golden_trace/voxcpm_trace.py goldens/voxcpm05_stop_parity
git commit -m "test(inference): add VoxCPM 0.5 stop trace"
```

---

### Task 2: Teach Rust Trace Helpers About Request Metadata

**Files:**
- Modify: `voxui/crates/voxui-inference/src/trace.rs`

- [ ] **Step 1: Extend trace manifest structs**

In `voxui/crates/voxui-inference/src/trace.rs`, replace `TraceManifest` with:

```rust
#[derive(Debug, Deserialize)]
struct TraceManifest {
    #[serde(default)]
    request: TraceRequest,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    lists: HashMap<String, Vec<u32>>,
    tensors: Vec<TensorRecord>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TraceRequest {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub cfg_value: f32,
    #[serde(default)]
    pub inference_timesteps: usize,
    #[serde(default)]
    pub min_len: usize,
    #[serde(default)]
    pub max_len: usize,
    #[serde(default)]
    pub retry_badcase: bool,
}
```

- [ ] **Step 2: Add request and metadata accessors**

Add these methods inside `impl TraceCase`:

```rust
    pub fn request(&self) -> &TraceRequest {
        &self.manifest.request
    }

    pub fn metadata_usize(&self, name: &str) -> Result<usize> {
        self.manifest
            .metadata
            .get(name)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .ok_or_else(|| anyhow::anyhow!("trace metadata `{name}` not found as usize"))
    }
```

- [ ] **Step 3: Run existing trace users**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test generate_flow_parity voxcpm2_first_patch_flow_matches_python_trace
```

Expected:

```text
test voxcpm2_first_patch_flow_matches_python_trace ... ok
```

- [ ] **Step 4: Commit trace helper changes**

Run:

```powershell
git add voxui/crates/voxui-inference/src/trace.rs
git commit -m "test(inference): expose trace request metadata"
```

---

### Task 3: Add Rust Stop Debug Trace Helper

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`

- [ ] **Step 1: Add debug structs**

In `voxui/crates/voxui-inference/src/engine.rs`, add these public structs after `FirstPatchDebug`:

```rust
#[derive(Debug)]
pub struct StopTraceStep {
    pub stop_logits: Tensor,
    pub stop_decision: u32,
    pub lm_hidden_stats: [f32; 4],
    pub residual_hidden_stats: [f32; 4],
    pub pred_feat_stats: [f32; 4],
}

#[derive(Debug)]
pub struct StopTraceDebug {
    pub steps: Vec<StopTraceStep>,
    pub generated_audio_feat: Tensor,
    pub generated_step_count: usize,
}
```

- [ ] **Step 2: Add the public debug entry point**

Add this method near `generate_debug_first_patch_with_noise`:

```rust
    pub fn generate_debug_stop_trace_with_noise(
        &mut self,
        request: SynthesisRequest,
        first_noise: Tensor,
    ) -> Result<StopTraceDebug> {
        let request = request.validated(self.config.variant)?;
        let prepared = self.build_inputs(&request)?;
        let max_len = bounded_max_len(&request, prepared.target_text_token_count);
        let mut state = self.prefill(&prepared)?;
        let mut steps = Vec::new();

        for step in 0..max_len {
            let fixed_noise = if step == 0 { Some(&first_noise) } else { None };
            let lm_hidden_stats = tensor_stats4(&state.lm_hidden)?;
            let residual_hidden_stats = tensor_stats4(&state.residual_hidden)?;
            let (_latent_patch, stop_logits, pred_feat) =
                self.generate_one_patch(&mut state, &request, fixed_noise)?;
            let logits = stop_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let stop_decision = logits
                .first()
                .and_then(|row| row.get(1).zip(row.first()))
                .map(|(stop, keep)| u32::from(stop > keep))
                .unwrap_or(0);
            steps.push(StopTraceStep {
                stop_logits,
                stop_decision,
                lm_hidden_stats,
                residual_hidden_stats,
                pred_feat_stats: tensor_stats4(&pred_feat)?,
            });
            if step > request.min_len && stop_decision == 1 {
                break;
            }
        }

        let latent = patches_to_latent(
            &state.generated_patches,
            self.config.latent_dim,
            self.config.patch_size,
        )?;
        Ok(StopTraceDebug {
            steps,
            generated_audio_feat: latent.transpose(1, 2)?.contiguous()?,
            generated_step_count: state.generated_patches.len(),
        })
    }
```

- [ ] **Step 3: Add tensor stats helper**

Add this helper near `bounded_max_len`:

```rust
fn tensor_stats4(tensor: &Tensor) -> Result<[f32; 4]> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        return Ok([0.0, 0.0, 0.0, 0.0]);
    }
    let len = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / len;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / len;
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok([mean, std, min, max])
}
```

- [ ] **Step 4: Run a compile check**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test generate_flow_parity voxcpm2_first_patch_flow_matches_python_trace
```

Expected:

```text
test voxcpm2_first_patch_flow_matches_python_trace ... ok
```

- [ ] **Step 5: Commit Rust debug helper**

Run:

```powershell
git add voxui/crates/voxui-inference/src/engine.rs
git commit -m "test(inference): add stop trace debug helper"
```

---

### Task 4: Add the Failing Rust Stop Parity Test

**Files:**
- Add: `voxui/crates/voxui-inference/tests/stop_parity.rs`

- [ ] **Step 1: Create the test file**

Create `voxui/crates/voxui-inference/tests/stop_parity.rs`:

```rust
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

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

fn stats_tensor(rows: &[[f32; 4]]) -> Tensor {
    let flat = rows.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
    Tensor::from_vec(flat, (rows.len(), 4), &Device::Cpu).unwrap()
}

fn logits_tensor(rows: &[candle_core::Tensor]) -> Tensor {
    let mut flat = Vec::new();
    for row in rows {
        flat.extend(row.to_dtype(candle_core::DType::F32).unwrap().to_vec2::<f32>().unwrap()[0].iter());
    }
    Tensor::from_vec(flat, (rows.len(), 2), &Device::Cpu).unwrap()
}

#[test]
fn voxcpm05_stop_steps_match_python_trace() {
    let root = repo_root();
    let trace = voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm05_stop_parity"))
        .unwrap();
    let request_meta = trace.request();
    let request = SynthesisRequest {
        text: request_meta.text.clone(),
        cfg_value: request_meta.cfg_value,
        inference_timesteps: request_meta.inference_timesteps,
        min_len: request_meta.min_len,
        max_len: request_meta.max_len,
        retry_badcase: request_meta.retry_badcase,
        ..SynthesisRequest::default()
    };

    let mut engine = VoxCPMEngine::load(&root.join("models/voxcpm05-fp16"), Device::Cpu).unwrap();
    let debug = engine
        .generate_debug_stop_trace_with_noise(request, trace.tensor("first_dit_noise").unwrap())
        .unwrap();

    let expected_steps = trace.metadata_usize("generated_step_count").unwrap();
    let expected_decisions = trace.u32_list("stop_decisions").unwrap();
    let actual_decisions = debug
        .steps
        .iter()
        .map(|step| step.stop_decision)
        .collect::<Vec<_>>();

    assert_eq!(debug.generated_step_count, expected_steps);
    assert_eq!(actual_decisions, expected_decisions);

    let actual_logits = logits_tensor(
        &debug
            .steps
            .iter()
            .map(|step| step.stop_logits.clone())
            .collect::<Vec<_>>(),
    );
    voxui_inference::trace::assert_close(
        &actual_logits,
        &trace.tensor("stop_logits_by_step").unwrap(),
        3e-3,
    )
    .unwrap();

    let lm_stats = debug
        .steps
        .iter()
        .map(|step| step.lm_hidden_stats)
        .collect::<Vec<_>>();
    voxui_inference::trace::assert_close(
        &stats_tensor(&lm_stats),
        &trace.tensor("lm_hidden_stats_by_step").unwrap(),
        2e-2,
    )
    .unwrap();
}
```

- [ ] **Step 2: Run the new test and capture the failure**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test stop_parity -- --nocapture
```

Expected before the fix:

```text
test voxcpm05_stop_steps_match_python_trace ... FAILED
```

The useful failure is one of:

```text
assertion `left == right` failed
```

or:

```text
max abs diff ... exceeds tolerance
```

- [ ] **Step 3: Do not commit the failing test alone unless it reproduces the reported bug**

If the test unexpectedly passes, run:

```powershell
cd voxui
cargo test -p voxui-inference --features cuda --test inference_suite voxcpm05_q4_lm_cuda -- --nocapture --test-threads=1
```

Expected if the bug is CUDA/q4-only:

```text
VoxCPMEngine::run_generation_once generated_patch_count=<bounded max>
```

Record that output in the next task by adding the q4 path to the parity test or by making the fix target CUDA runtime quantization.

---

### Task 5: Patch the First Confirmed Mismatch

**Files:**
- Modify one of:
  - `voxui/crates/voxui-inference/src/engine.rs`
  - `voxui/crates/voxui-inference/src/base_lm.rs`
  - `voxui/crates/voxui-inference/src/weights.rs`
  - `voxui/crates/voxui-inference/src/lora.rs`

- [ ] **Step 1: Classify the stop parity failure**

Use this decision table from the failed `stop_parity` output:

```text
generated_step_count differs, stop_logits close until Python stop:
  fix stop decision/class ordering in engine.rs.

step 0 lm_hidden_stats differs:
  inspect prefill input construction, tokenizer ids, base_lm prefill, residual_lm prefill.

step 0 lm_hidden_stats close but stop_logits differ:
  inspect stop_proj/stop_head loading and linear_projection.

step 0 matches but later lm_hidden_stats diverge:
  inspect base_lm/residual_lm cache position stepping and generated patch embedding update.

only CUDA/q4 diverges:
  inspect runtime tensor dtype/quantized linear behavior in weights.rs and lora.rs.
```

- [ ] **Step 2: Apply the smallest patch**

If the failure is the most likely stop decision/class-ordering mismatch, patch `run_generation_once` and `generate_debug_stop_trace_with_noise` in `engine.rs` to use an explicit argmax helper. Add this helper near `tensor_stats4`:

```rust
fn stop_decision_from_logits(stop_logits: &Tensor) -> Result<u32> {
    let logits = stop_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let row = logits
        .first()
        .ok_or_else(|| anyhow::anyhow!("stop logits are empty"))?;
    if row.len() != 2 {
        bail!("stop logits must have two classes, got {}", row.len());
    }
    Ok(u32::from(row[1] >= row[0]))
}
```

Then replace the inline stop decision in `run_generation_once` with:

```rust
            let stop_flag = stop_decision_from_logits(&stop_logits)? == 1;
```

And replace the inline debug stop decision in `generate_debug_stop_trace_with_noise` with:

```rust
            let stop_decision = stop_decision_from_logits(&stop_logits)?;
```

If the failure points somewhere else, patch only that specific mismatch and add a short code comment explaining the Python reference behavior being matched.

- [ ] **Step 3: Re-run stop parity**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test stop_parity -- --nocapture
```

Expected:

```text
test voxcpm05_stop_steps_match_python_trace ... ok
```

- [ ] **Step 4: Commit the parity test and fix**

Run:

```powershell
git add voxui/crates/voxui-inference/tests/stop_parity.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/src/base_lm.rs voxui/crates/voxui-inference/src/weights.rs voxui/crates/voxui-inference/src/lora.rs
git commit -m "fix(inference): match VoxCPM stop parity"
```

If only some of those files changed, `git add` will stage only existing modified paths that match.

---

### Task 6: Regression Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run focused parity tests**

Run:

```powershell
cd voxui
cargo test -p voxui-inference --test stop_parity
cargo test -p voxui-inference --test generate_flow_parity
cargo test -p voxui-inference --test tokenizer_parity
cargo test -p voxui-inference --test dit_parity
```

Expected:

```text
test result: ok
```

- [ ] **Step 2: Run CUDA q4 matrix**

Run:

```powershell
cd voxui
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

Also inspect the printed generation step counts. VoxCPM 0.5 must not run to the bounded max for the trace-owned prompt when Python stops early.

- [ ] **Step 3: Check workspace state**

Run:

```powershell
git status --short
```

Expected:

```text
Only pre-existing unrelated dirty files remain, or no changes remain after commits.
```
