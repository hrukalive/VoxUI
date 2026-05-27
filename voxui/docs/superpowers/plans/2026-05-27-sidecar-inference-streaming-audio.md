# Sidecar Inference and Streaming Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move VoxUI inference into a long-lived sidecar process and upgrade `voxui-audio` so streaming and non-streaming synthesis share a robust sample-rate-independent PCM playback path.

**Architecture:** Tauri remains the control plane for queueing, cancellation, history, playback state, and frontend events. A new inference sidecar owns `VoxCPMEngine` and speaks a length-prefixed protocol over stdio. `voxui-audio` gets a production streaming session that keeps one stateful `r8brain` resampler per playback session.

**Tech Stack:** Rust workspace, Tauri 2, Leptos CSR, Candle, CPAL, r8brain-rs, ringbuf, serde/serde_json.

---

## File Structure

- Create `crates/voxui-sidecar-protocol/`: shared protocol crate with command/event enums, frame headers, and read/write helpers.
- Create `crates/voxui-inference-sidecar/`: binary crate that owns `VoxCPMEngine` and services protocol commands over stdio.
- Modify `Cargo.toml`: add the two new crates to the workspace.
- Modify `crates/voxui-audio/src/lib.rs`: split testable streaming resampling helpers from CPAL playback and upgrade streaming playback.
- Create `crates/voxui-desktop/src-tauri/src/inference_sidecar.rs`: sidecar process lifecycle, request/response routing, stale event filtering helpers.
- Modify `crates/voxui-desktop/src-tauri/src/app_core.rs`: replace in-process engine ownership with sidecar-oriented model/generation state.
- Modify `crates/voxui-desktop/src-tauri/src/commands.rs`: route load/generation commands to the sidecar manager and route audio chunks into playback.
- Modify `crates/voxui-desktop/src-tauri/tauri.conf.json`: configure the sidecar as an external binary for packaged builds.
- Modify `scripts/package-windows-cuda.ps1`: ensure the sidecar binary is built and copied for Windows CUDA packaging.
- Add focused tests under the new crates and existing crate test directories.

---

### Task 1: Shared Sidecar Protocol Crate

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/voxui-sidecar-protocol/Cargo.toml`
- Create: `crates/voxui-sidecar-protocol/src/lib.rs`
- Test: `crates/voxui-sidecar-protocol/src/lib.rs`

- [ ] **Step 1: Add failing protocol round-trip tests**

Create `crates/voxui-sidecar-protocol/Cargo.toml`:

```toml
[package]
name = "voxui-sidecar-protocol"
version.workspace = true
edition.workspace = true

[dependencies]
anyhow.workspace = true
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Create `crates/voxui-sidecar-protocol/src/lib.rs` with tests first:

```rust
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_preserves_header_and_payload() {
        let frame = Frame {
            header: SidecarEvent::AudioChunk {
                item_id: "item-1".to_string(),
                sample_rate: 16_000,
                current: 3,
                total: 10,
                is_final: false,
            },
            payload: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame).unwrap();

        let decoded: Frame<SidecarEvent> = read_frame(&mut bytes.as_slice()).unwrap();

        assert_eq!(decoded, frame);
    }

    #[test]
    fn read_frame_rejects_truncated_payload() {
        let frame = Frame {
            header: SidecarCommand::CancelSynthesis {
                item_id: "item-1".to_string(),
            },
            payload: vec![1, 2, 3],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame).unwrap();
        bytes.pop();

        let error = read_frame::<_, SidecarCommand>(&mut bytes.as_slice()).unwrap_err();

        assert!(error.to_string().contains("payload"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test -p voxui-sidecar-protocol
```

Expected: fails because the workspace does not yet include the crate and the protocol types/functions are not implemented.

- [ ] **Step 3: Add workspace member and protocol implementation**

Modify root `Cargo.toml` workspace members:

```toml
members = [
    "crates/voxui-gguf",
    "crates/voxui-inference",
    "crates/voxui-audio",
    "crates/voxui-cli",
    "crates/voxui-desktop",
    "crates/voxui-desktop/src-tauri",
    "crates/voxui-sidecar-protocol",
]
```

Complete `crates/voxui-sidecar-protocol/src/lib.rs`:

```rust
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SidecarCommand {
    LoadModel {
        load_id: u64,
        model_dir: PathBuf,
        lora_path: Option<PathBuf>,
        backend: BackendKind,
    },
    CancelLoad {
        load_id: u64,
    },
    Synthesize {
        item_id: String,
        request: SynthesisRequestDto,
        streaming: bool,
    },
    CancelSynthesis {
        item_id: String,
    },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SidecarEvent {
    Ready,
    ModelLoadProgress {
        load_id: u64,
        phase: String,
        loaded_bytes: u64,
        total_bytes: u64,
        component: Option<String>,
        component_index: usize,
        component_total: usize,
    },
    ModelLoadDone {
        load_id: u64,
        status: OperationStatus,
        sample_rate: Option<u32>,
        error: Option<String>,
    },
    GenerationProgress {
        item_id: String,
        current: usize,
        total: usize,
    },
    AudioChunk {
        item_id: String,
        sample_rate: u32,
        current: usize,
        total: usize,
        is_final: bool,
    },
    AudioFinal {
        item_id: String,
        sample_rate: u32,
        duration_seconds: f32,
    },
    GenerationDone {
        item_id: String,
        status: OperationStatus,
        sample_rate: Option<u32>,
        duration_seconds: Option<f32>,
        error: Option<String>,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    Success,
    Canceled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRequestDto {
    pub text: String,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub consolidate_n: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame<T> {
    pub header: T,
    pub payload: Vec<u8>,
}

pub fn write_frame<W, T>(writer: &mut W, frame: &Frame<T>) -> Result<()>
where
    W: Write,
    T: Serialize,
{
    let header = serde_json::to_vec(&frame.header).context("serialize frame header")?;
    if header.len() > MAX_HEADER_BYTES {
        bail!("frame header exceeds {MAX_HEADER_BYTES} bytes");
    }
    if frame.payload.len() > MAX_PAYLOAD_BYTES {
        bail!("frame payload exceeds {MAX_PAYLOAD_BYTES} bytes");
    }

    writer.write_all(&(header.len() as u32).to_le_bytes())?;
    writer.write_all(&(frame.payload.len() as u32).to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(&frame.payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame<R, T>(reader: &mut R) -> Result<Frame<T>>
where
    R: Read,
    T: for<'de> Deserialize<'de>,
{
    let mut lens = [0_u8; 8];
    reader.read_exact(&mut lens).context("read frame lengths")?;
    let header_len = u32::from_le_bytes(lens[0..4].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(lens[4..8].try_into().unwrap()) as usize;
    if header_len > MAX_HEADER_BYTES {
        bail!("frame header length {header_len} exceeds {MAX_HEADER_BYTES}");
    }
    if payload_len > MAX_PAYLOAD_BYTES {
        bail!("frame payload length {payload_len} exceeds {MAX_PAYLOAD_BYTES}");
    }

    let mut header_bytes = vec![0_u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .context("read frame header")?;
    let mut payload = vec![0_u8; payload_len];
    reader
        .read_exact(&mut payload)
        .context("read frame payload")?;
    let header = serde_json::from_slice(&header_bytes).context("deserialize frame header")?;
    Ok(Frame { header, payload })
}

pub fn f32_samples_to_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

pub fn f32_samples_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("PCM payload length must be a multiple of 4 bytes");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_preserves_header_and_payload() {
        let frame = Frame {
            header: SidecarEvent::AudioChunk {
                item_id: "item-1".to_string(),
                sample_rate: 16_000,
                current: 3,
                total: 10,
                is_final: false,
            },
            payload: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame).unwrap();

        let decoded: Frame<SidecarEvent> = read_frame(&mut bytes.as_slice()).unwrap();

        assert_eq!(decoded, frame);
    }

    #[test]
    fn read_frame_rejects_truncated_payload() {
        let frame = Frame {
            header: SidecarCommand::CancelSynthesis {
                item_id: "item-1".to_string(),
            },
            payload: vec![1, 2, 3],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame).unwrap();
        bytes.pop();

        let error = read_frame::<_, SidecarCommand>(&mut bytes.as_slice()).unwrap_err();

        assert!(error.to_string().contains("payload"));
    }

    #[test]
    fn pcm_bytes_round_trip() {
        let samples = vec![0.0, 0.5, -0.25, 1.0];
        let bytes = f32_samples_to_le_bytes(&samples);
        assert_eq!(f32_samples_from_le_bytes(&bytes).unwrap(), samples);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cargo test -p voxui-sidecar-protocol
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add Cargo.toml crates/voxui-sidecar-protocol
git commit -m "Add sidecar protocol crate"
```

---

### Task 2: Testable Streaming Resampler in voxui-audio

**Files:**
- Modify: `crates/voxui-audio/src/lib.rs`

- [ ] **Step 1: Add failing tests for chunked resampling and flush**

Append tests in `crates/voxui-audio/src/lib.rs` under the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn streaming_resampler_splits_large_input_and_flushes_tail() {
    let mut resampler = super::StreamingResampler::new(16_000, 48_000, 128).unwrap();
    let input = (0..1_000)
        .map(|idx| ((idx as f32) / 20.0).sin())
        .collect::<Vec<_>>();

    let out = resampler.process(&input).unwrap();
    let tail = resampler.finish().unwrap();

    assert!(!out.is_empty());
    assert!(!tail.is_empty());
    assert!(out.len() + tail.len() > input.len());
}

#[test]
fn streaming_resampler_chunked_output_matches_one_shot_output_length() {
    let input = (0..3_000)
        .map(|idx| ((idx as f32) / 30.0).sin() * 0.5)
        .collect::<Vec<_>>();
    let mut whole = super::StreamingResampler::new(24_000, 48_000, 512).unwrap();
    let mut chunked = super::StreamingResampler::new(24_000, 48_000, 512).unwrap();

    let mut whole_out = whole.process(&input).unwrap();
    whole_out.extend(whole.finish().unwrap());

    let mut chunked_out = Vec::new();
    for chunk in input.chunks(137) {
        chunked_out.extend(chunked.process(chunk).unwrap());
    }
    chunked_out.extend(chunked.finish().unwrap());

    let len_delta = whole_out.len().abs_diff(chunked_out.len());
    assert!(len_delta <= 2, "length delta was {len_delta}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cargo test -p voxui-audio streaming_resampler --lib
```

Expected: fails because `StreamingResampler` does not exist.

- [ ] **Step 3: Add `StreamingResampler` helper**

Add near the existing `StreamingPlayer` code:

```rust
pub struct StreamingResampler {
    source_rate: u32,
    device_rate: u32,
    resampler: Resampler,
}

impl StreamingResampler {
    pub fn new(source_rate: u32, device_rate: u32, max_input_len: usize) -> Result<Self> {
        if source_rate == 0 || device_rate == 0 {
            return Err(anyhow!("sample rates must be non-zero"));
        }
        Ok(Self {
            source_rate,
            device_rate,
            resampler: Resampler::new(
                source_rate as f64,
                device_rate as f64,
                max_input_len.max(1),
                2.0,
                PrecisionProfile::Bits24,
            ),
        })
    }

    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    pub fn device_rate(&self) -> u32 {
        self.device_rate
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let ratio = self.device_rate as f64 / self.source_rate as f64;
        let mut output = Vec::with_capacity((samples.len() as f64 * ratio * 1.2).ceil() as usize);
        let max_input_len = self.resampler.max_input_len();

        for chunk in samples.chunks(max_input_len) {
            let input_f64 = chunk.iter().map(|&sample| sample as f64).collect::<Vec<_>>();
            let max_out = ((input_f64.len() as f64 * ratio).ceil() as usize)
                .saturating_add(8192)
                .max(8192);
            let mut buf = vec![0.0_f64; max_out];
            let len = self.resampler.process(&input_f64, &mut buf);
            output.extend(buf[..len].iter().map(|&sample| sample as f32));
        }

        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Vec<f32>> {
        let mut buf = vec![0.0_f64; 16_384];
        let len = self.resampler.flush(&mut buf);
        Ok(buf[..len].iter().map(|&sample| sample as f32).collect())
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
cargo test -p voxui-audio streaming_resampler --lib
```

Expected: the two new tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-audio/src/lib.rs
git commit -m "Add stateful streaming resampler"
```

---

### Task 3: Upgrade voxui-audio StreamingPlayer API

**Files:**
- Modify: `crates/voxui-audio/src/lib.rs`
- Modify: `crates/voxui-audio/examples/play_test.rs` if it references old APIs

- [ ] **Step 1: Add tests for session configuration helpers**

Add tests under `#[cfg(test)] mod tests`:

```rust
#[test]
fn streaming_buffer_capacity_uses_device_rate() {
    assert_eq!(super::streaming_buffer_capacity(16_000, 48_000, 0.25), 12_000);
    assert_eq!(super::streaming_buffer_capacity(48_000, 16_000, 0.25), 12_000);
}

#[test]
fn streaming_buffer_capacity_has_minimum_size() {
    assert_eq!(super::streaming_buffer_capacity(16_000, 48_000, 0.0), 4_800);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cargo test -p voxui-audio streaming_buffer_capacity --lib
```

Expected: fails because `streaming_buffer_capacity` does not exist.

- [ ] **Step 3: Replace `StreamingPlayer` with production session shape**

Modify `StreamingPlayer` so it stores selected host/device, volume, stop/drain state, and a `StreamingResampler`. The API should be:

```rust
const STREAMING_MIN_BUFFER_MS: usize = 100;

pub struct StreamingPlayer {
    stream: cpal::Stream,
    producer: ringbuf::HeapProd<f32>,
    resampler: StreamingResampler,
    stop: Arc<AtomicBool>,
    done: Option<mpsc::Receiver<()>>,
}

impl StreamingPlayer {
    pub fn new(
        host_name: &str,
        device_name: &str,
        source_sample_rate: u32,
        pre_buffer_secs: f32,
        volume: VolumeHandle,
    ) -> Result<Self> {
        let host_id = cpal::available_hosts()
            .into_iter()
            .find(|id| format!("{:?}", id) == host_name)
            .ok_or_else(|| anyhow!("unknown host: {host_name}"))?;
        let host = cpal::host_from_id(host_id)?;
        let device = host
            .output_devices()?
            .find(|device| device.name().map(|name| name == device_name).unwrap_or(false))
            .ok_or_else(|| anyhow!("device not found: {device_name}"))?;
        let default_config = device.default_output_config()?;
        let channels = default_config.channels();
        if channels == 0 {
            return Err(anyhow!("output device reports 0 channels"));
        }
        let device_rate = default_config.sample_rate().0;
        let capacity = streaming_buffer_capacity(source_sample_rate, device_rate, pre_buffer_secs);
        let ring = HeapRb::<f32>::new(capacity);
        let (producer, mut consumer) = ring.split();
        let stop = Arc::new(AtomicBool::new(false));
        let callback_stop = stop.clone();
        let callback_volume = volume;
        let (done_tx, done_rx) = mpsc::channel();
        let done_sent = Arc::new(Mutex::new(false));
        let callback_done_sent = done_sent.clone();

        let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(device_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let ch = channels as usize;
                let gain = callback_volume.gain();
                let stopped = callback_stop.load(Ordering::SeqCst);
                let mut wrote_nonzero = false;
                for frame in data.chunks_mut(ch) {
                    let value = if stopped {
                        0.0
                    } else {
                        consumer.try_pop().unwrap_or(0.0) * gain
                    };
                    wrote_nonzero |= value != 0.0;
                    for sample in frame {
                        *sample = value;
                    }
                }
                if stopped && !wrote_nonzero && !*callback_done_sent.lock().unwrap() {
                    *callback_done_sent.lock().unwrap() = true;
                    let _ = done_tx.send(());
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )?;
        stream.play()?;

        Ok(Self {
            stream,
            producer,
            resampler: StreamingResampler::new(source_sample_rate, device_rate, 8192)?,
            stop,
            done: Some(done_rx),
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        let resampled = self.resampler.process(samples)?;
        self.push_resampled(&resampled);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<mpsc::Receiver<()>> {
        let tail = self.resampler.finish()?;
        self.push_resampled(&tail);
        self.stop.store(true, Ordering::SeqCst);
        self.done
            .take()
            .ok_or_else(|| anyhow!("streaming playback has already been finished"))
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    fn push_resampled(&mut self, samples: &[f32]) {
        for &sample in samples {
            while self.producer.try_push(sample).is_err() {
                if self.stop.load(Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

fn streaming_buffer_capacity(source_rate: u32, device_rate: u32, pre_buffer_secs: f32) -> usize {
    let rate = source_rate.max(device_rate).max(1);
    let min = rate as usize * STREAMING_MIN_BUFFER_MS / 1_000;
    let requested = (rate as f32 * pre_buffer_secs.max(0.0)).ceil() as usize;
    requested.max(min.max(1))
}
```

- [ ] **Step 4: Run audio tests**

Run:

```powershell
cargo test -p voxui-audio --lib
```

Expected: all `voxui-audio` tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-audio/src/lib.rs crates/voxui-audio/examples/play_test.rs
git commit -m "Upgrade streaming audio player"
```

---

### Task 4: Inference Sidecar Binary Skeleton

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/voxui-inference-sidecar/Cargo.toml`
- Create: `crates/voxui-inference-sidecar/src/main.rs`
- Create: `crates/voxui-inference-sidecar/src/lib.rs`
- Test: `crates/voxui-inference-sidecar/src/lib.rs`

- [ ] **Step 1: Add failing command handler tests**

Create `crates/voxui-inference-sidecar/Cargo.toml`:

```toml
[package]
name = "voxui-inference-sidecar"
version.workspace = true
edition.workspace = true

[[bin]]
name = "voxui-inference-sidecar"
path = "src/main.rs"

[dependencies]
anyhow.workspace = true
candle-core.workspace = true
voxui-inference = { path = "../voxui-inference" }
voxui-sidecar-protocol = { path = "../voxui-sidecar-protocol" }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
```

Create `crates/voxui-inference-sidecar/src/lib.rs` with tests:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use voxui_sidecar_protocol::{OperationStatus, SidecarCommand, SidecarEvent};

pub struct SidecarEngine {
    active_load: Option<(u64, Arc<AtomicBool>)>,
    active_generation: Option<(String, Arc<AtomicBool>)>,
}

impl Default for SidecarEngine {
    fn default() -> Self {
        Self {
            active_load: None,
            active_generation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_generation_sets_active_cancel_flag() {
        let mut engine = SidecarEngine::default();
        let cancel = Arc::new(AtomicBool::new(false));
        engine.active_generation = Some(("item-1".to_string(), cancel.clone()));

        let events = engine.handle_control_for_test(SidecarCommand::CancelSynthesis {
            item_id: "item-1".to_string(),
        });

        assert!(cancel.load(Ordering::SeqCst));
        assert_eq!(
            events,
            vec![SidecarEvent::GenerationDone {
                item_id: "item-1".to_string(),
                status: OperationStatus::Canceled,
                sample_rate: None,
                duration_seconds: None,
                error: None,
            }]
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test -p voxui-inference-sidecar
```

Expected: fails because workspace is not updated and `handle_control_for_test` is missing.

- [ ] **Step 3: Add workspace member and sidecar skeleton**

Add to root `Cargo.toml` members:

```toml
"crates/voxui-inference-sidecar",
```

Complete `crates/voxui-inference-sidecar/src/lib.rs`:

```rust
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::Device;
use voxui_inference::{SynthesisRequest, VoxCPMEngine};
use voxui_sidecar_protocol::{
    f32_samples_to_le_bytes, BackendKind, Frame, OperationStatus, SidecarCommand, SidecarEvent,
};

pub struct SidecarEngine {
    engine: Option<VoxCPMEngine>,
    active_load: Option<(u64, Arc<AtomicBool>)>,
    active_generation: Option<(String, Arc<AtomicBool>)>,
}

impl Default for SidecarEngine {
    fn default() -> Self {
        Self {
            engine: None,
            active_load: None,
            active_generation: None,
        }
    }
}

impl SidecarEngine {
    pub fn run<R, W>(&mut self, mut reader: R, mut writer: W) -> Result<()>
    where
        R: Read,
        W: Write,
    {
        voxui_sidecar_protocol::write_frame(
            &mut writer,
            &Frame {
                header: SidecarEvent::Ready,
                payload: Vec::new(),
            },
        )?;
        loop {
            let frame: Frame<SidecarCommand> =
                voxui_sidecar_protocol::read_frame(&mut reader).context("read sidecar command")?;
            if matches!(frame.header, SidecarCommand::Shutdown) {
                break;
            }
            self.handle_command(frame.header, &mut writer)?;
        }
        Ok(())
    }

    pub fn handle_control_for_test(&mut self, command: SidecarCommand) -> Vec<SidecarEvent> {
        match command {
            SidecarCommand::CancelSynthesis { item_id } => {
                if let Some((active_id, cancel)) = self.active_generation.as_ref() {
                    if active_id == &item_id {
                        cancel.store(true, Ordering::SeqCst);
                        return vec![SidecarEvent::GenerationDone {
                            item_id,
                            status: OperationStatus::Canceled,
                            sample_rate: None,
                            duration_seconds: None,
                            error: None,
                        }];
                    }
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn handle_command<W: Write>(&mut self, command: SidecarCommand, writer: &mut W) -> Result<()> {
        match command {
            SidecarCommand::LoadModel {
                load_id,
                model_dir,
                lora_path,
                backend,
            } => self.load_model(load_id, model_dir, lora_path, backend, writer),
            SidecarCommand::CancelLoad { load_id } => {
                if let Some((active_id, cancel)) = self.active_load.as_ref() {
                    if *active_id == load_id {
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
                Ok(())
            }
            SidecarCommand::Synthesize {
                item_id,
                request,
                streaming,
            } => self.synthesize(item_id, request, streaming, writer),
            SidecarCommand::CancelSynthesis { item_id } => {
                for event in self.handle_control_for_test(SidecarCommand::CancelSynthesis {
                    item_id,
                }) {
                    send_event(writer, event, Vec::new())?;
                }
                Ok(())
            }
            SidecarCommand::Shutdown => Ok(()),
        }
    }

    fn load_model<W: Write>(
        &mut self,
        load_id: u64,
        model_dir: std::path::PathBuf,
        lora_path: Option<std::path::PathBuf>,
        backend: BackendKind,
        writer: &mut W,
    ) -> Result<()> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.active_load = Some((load_id, cancel.clone()));
        let device = match backend {
            BackendKind::Cpu => Device::Cpu,
            BackendKind::Cuda => Device::new_cuda(0).context("create CUDA device")?,
        };
        let result = VoxCPMEngine::load_with_progress(
            &model_dir,
            device,
            |component_index, component_total| {
                let _ = send_event(
                    writer,
                    SidecarEvent::ModelLoadProgress {
                        load_id,
                        phase: "device_loading".to_string(),
                        loaded_bytes: 0,
                        total_bytes: 0,
                        component: None,
                        component_index,
                        component_total,
                    },
                    Vec::new(),
                );
            },
            Some(&cancel),
        )
        .and_then(|mut engine| {
            if let Some(path) = lora_path {
                engine.load_lora(&path)?;
            }
            Ok(engine)
        });
        self.active_load = None;

        match result {
            Ok(engine) => {
                let sample_rate = engine.sample_rate();
                self.engine = Some(engine);
                send_event(
                    writer,
                    SidecarEvent::ModelLoadDone {
                        load_id,
                        status: OperationStatus::Success,
                        sample_rate: Some(sample_rate),
                        error: None,
                    },
                    Vec::new(),
                )
            }
            Err(error) => send_event(
                writer,
                SidecarEvent::ModelLoadDone {
                    load_id,
                    status: if cancel.load(Ordering::SeqCst) {
                        OperationStatus::Canceled
                    } else {
                        OperationStatus::Failed
                    },
                    sample_rate: None,
                    error: Some(error.to_string()),
                },
                Vec::new(),
            ),
        }
    }

    fn synthesize<W: Write>(
        &mut self,
        item_id: String,
        request: voxui_sidecar_protocol::SynthesisRequestDto,
        streaming: bool,
        writer: &mut W,
    ) -> Result<()> {
        let Some(engine) = self.engine.as_mut() else {
            return send_event(
                writer,
                SidecarEvent::GenerationDone {
                    item_id,
                    status: OperationStatus::Failed,
                    sample_rate: None,
                    duration_seconds: None,
                    error: Some("no model loaded".to_string()),
                },
                Vec::new(),
            );
        };
        let cancel = Arc::new(AtomicBool::new(false));
        self.active_generation = Some((item_id.clone(), cancel.clone()));
        let sample_rate = engine.sample_rate();
        let request = SynthesisRequest {
            text: request.text,
            prompt_wav_path: request.prompt_wav_path,
            prompt_text: request.prompt_text,
            reference_wav_path: request.reference_wav_path,
            cfg_value: request.cfg_value,
            inference_timesteps: request.inference_timesteps,
            min_len: request.min_len,
            max_len: request.max_len,
            normalize: false,
            retry_badcase: request.retry_badcase && !streaming,
            retry_badcase_max_times: request.retry_badcase_max_times,
            retry_badcase_ratio_threshold: request.retry_badcase_ratio_threshold,
            consolidate_n: request.consolidate_n,
        };

        let result = if streaming {
            let mut all_samples = Vec::new();
            let chunk_result = engine.generate_streaming_cancellable(
                request,
                |chunk| {
                    send_event(
                        writer,
                        SidecarEvent::GenerationProgress {
                            item_id: item_id.clone(),
                            current: chunk.generated_patch_count,
                            total: chunk.max_patches,
                        },
                        Vec::new(),
                    )?;
                    send_event(
                        writer,
                        SidecarEvent::AudioChunk {
                            item_id: item_id.clone(),
                            sample_rate: chunk.sample_rate,
                            current: chunk.generated_patch_count,
                            total: chunk.max_patches,
                            is_final: chunk.is_final,
                        },
                        f32_samples_to_le_bytes(&chunk.samples),
                    )?;
                    all_samples.extend_from_slice(&chunk.samples);
                    Ok(())
                },
                Some(&cancel),
            );
            chunk_result.map(|_| all_samples)
        } else {
            engine.generate_cancellable(
                request,
                |current, total| {
                    let _ = send_event(
                        writer,
                        SidecarEvent::GenerationProgress {
                            item_id: item_id.clone(),
                            current,
                            total,
                        },
                        Vec::new(),
                    );
                },
                Some(&cancel),
            )
        };
        self.active_generation = None;

        match result {
            Ok(samples) => {
                let duration_seconds = samples.len() as f32 / sample_rate as f32;
                if !streaming {
                    send_event(
                        writer,
                        SidecarEvent::AudioFinal {
                            item_id: item_id.clone(),
                            sample_rate,
                            duration_seconds,
                        },
                        f32_samples_to_le_bytes(&samples),
                    )?;
                }
                send_event(
                    writer,
                    SidecarEvent::GenerationDone {
                        item_id,
                        status: if cancel.load(Ordering::SeqCst) {
                            OperationStatus::Canceled
                        } else {
                            OperationStatus::Success
                        },
                        sample_rate: Some(sample_rate),
                        duration_seconds: Some(duration_seconds),
                        error: None,
                    },
                    Vec::new(),
                )
            }
            Err(error) => send_event(
                writer,
                SidecarEvent::GenerationDone {
                    item_id,
                    status: if cancel.load(Ordering::SeqCst) {
                        OperationStatus::Canceled
                    } else {
                        OperationStatus::Failed
                    },
                    sample_rate: None,
                    duration_seconds: None,
                    error: Some(error.to_string()),
                },
                Vec::new(),
            ),
        }
    }
}

fn send_event<W: Write>(writer: &mut W, header: SidecarEvent, payload: Vec<u8>) -> Result<()> {
    voxui_sidecar_protocol::write_frame(writer, &Frame { header, payload })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_generation_sets_active_cancel_flag() {
        let mut engine = SidecarEngine::default();
        let cancel = Arc::new(AtomicBool::new(false));
        engine.active_generation = Some(("item-1".to_string(), cancel.clone()));

        let events = engine.handle_control_for_test(SidecarCommand::CancelSynthesis {
            item_id: "item-1".to_string(),
        });

        assert!(cancel.load(Ordering::SeqCst));
        assert_eq!(
            events,
            vec![SidecarEvent::GenerationDone {
                item_id: "item-1".to_string(),
                status: OperationStatus::Canceled,
                sample_rate: None,
                duration_seconds: None,
                error: None,
            }]
        );
    }
}
```

Create `crates/voxui-inference-sidecar/src/main.rs`:

```rust
fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt().with_writer(std::io::stderr).try_init();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut engine = voxui_inference_sidecar::SidecarEngine::default();
    engine.run(stdin.lock(), stdout.lock())
}
```

- [ ] **Step 4: Run sidecar tests**

Run:

```powershell
cargo test -p voxui-inference-sidecar
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add Cargo.toml crates/voxui-inference-sidecar
git commit -m "Add inference sidecar skeleton"
```

---

### Task 5: Desktop Sidecar Manager

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/Cargo.toml`
- Create: `crates/voxui-desktop/src-tauri/src/inference_sidecar.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/lib.rs`
- Test: `crates/voxui-desktop/src-tauri/tests/sidecar_manager_tests.rs`

- [ ] **Step 1: Add stale event and PCM decoding tests**

Create `crates/voxui-desktop/src-tauri/tests/sidecar_manager_tests.rs`:

```rust
use voxui_desktop::inference_sidecar::{is_active_generation_event, sidecar_samples_from_payload};
use voxui_sidecar_protocol::{f32_samples_to_le_bytes, SidecarEvent};

#[test]
fn stale_generation_event_is_rejected() {
    let event = SidecarEvent::GenerationProgress {
        item_id: "old".to_string(),
        current: 1,
        total: 2,
    };

    assert!(!is_active_generation_event(Some("new"), &event));
    assert!(is_active_generation_event(Some("old"), &event));
}

#[test]
fn audio_payload_decodes_pcm_samples() {
    let samples = vec![0.0, 0.25, -0.5];
    let payload = f32_samples_to_le_bytes(&samples);

    assert_eq!(sidecar_samples_from_payload(&payload).unwrap(), samples);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test -p voxui-desktop sidecar_manager_tests
```

Expected: fails because `inference_sidecar` module and dependency are missing.

- [ ] **Step 3: Add dependency and manager helpers**

Modify `crates/voxui-desktop/src-tauri/Cargo.toml`:

```toml
voxui-sidecar-protocol = { path = "../../voxui-sidecar-protocol" }
```

Modify `crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod inference_sidecar;
```

Create `crates/voxui-desktop/src-tauri/src/inference_sidecar.rs`:

```rust
use std::io::{BufReader, BufWriter};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result};
use voxui_sidecar_protocol::{f32_samples_from_le_bytes, Frame, SidecarCommand, SidecarEvent};

pub struct SidecarProcess {
    child: Child,
    writer: BufWriter<ChildStdin>,
}

impl SidecarProcess {
    pub fn spawn(sidecar_path: impl AsRef<std::path::Path>) -> Result<(Self, mpsc::Receiver<Frame<SidecarEvent>>)> {
        let mut child = Command::new(sidecar_path.as_ref())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn sidecar {}", sidecar_path.as_ref().display()))?;
        let stdin = child.stdin.take().context("sidecar stdin unavailable")?;
        let stdout = child.stdout.take().context("sidecar stdout unavailable")?;
        let (sender, receiver) = mpsc::channel();
        spawn_reader(stdout, sender);
        Ok((
            Self {
                child,
                writer: BufWriter::new(stdin),
            },
            receiver,
        ))
    }

    pub fn send(&mut self, command: SidecarCommand) -> Result<()> {
        voxui_sidecar_protocol::write_frame(
            &mut self.writer,
            &Frame {
                header: command,
                payload: Vec::new(),
            },
        )
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

fn spawn_reader(stdout: ChildStdout, sender: mpsc::Sender<Frame<SidecarEvent>>) {
    thread::Builder::new()
        .name("voxui-sidecar-reader".to_string())
        .spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Ok(frame) = voxui_sidecar_protocol::read_frame(&mut reader) {
                if sender.send(frame).is_err() {
                    break;
                }
            }
        })
        .expect("spawn sidecar reader thread");
}

pub fn is_active_generation_event(active_item_id: Option<&str>, event: &SidecarEvent) -> bool {
    let Some(active_item_id) = active_item_id else {
        return false;
    };
    match event {
        SidecarEvent::GenerationProgress { item_id, .. }
        | SidecarEvent::AudioChunk { item_id, .. }
        | SidecarEvent::AudioFinal { item_id, .. }
        | SidecarEvent::GenerationDone { item_id, .. } => item_id == active_item_id,
        _ => true,
    }
}

pub fn sidecar_samples_from_payload(payload: &[u8]) -> Result<Vec<f32>> {
    f32_samples_from_le_bytes(payload)
}
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cargo test -p voxui-desktop sidecar_manager_tests
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/Cargo.toml crates/voxui-desktop/src-tauri/src/lib.rs crates/voxui-desktop/src-tauri/src/inference_sidecar.rs crates/voxui-desktop/src-tauri/tests/sidecar_manager_tests.rs
git commit -m "Add desktop sidecar manager"
```

---

### Task 6: Refactor AppCore for Sidecar-Owned Engine

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`
- Test: existing `crates/voxui-desktop/src-tauri/tests/queue_tests.rs`
- Test: existing `crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`

- [ ] **Step 1: Add failing test for sidecar generation run without local engine**

Append a test to `app_core.rs` tests:

```rust
#[test]
fn sidecar_generation_run_does_not_take_local_engine() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();

    let run = core.begin_generation_run(&item.id).unwrap();

    assert_eq!(run.item_id, item.id);
    assert_eq!(run.sample_rate, 16_000);
}
```

- [ ] **Step 2: Run test to expose current coupling**

Run:

```powershell
cargo test -p voxui-desktop sidecar_generation_run_does_not_take_local_engine --lib
```

Expected: fails or exposes that `GenerationRun` still contains `Option<VoxCPMEngine>` and execution is in-process.

- [ ] **Step 3: Replace local engine field with sidecar load metadata**

In `AppCore`, replace:

```rust
engine: Option<voxui_inference::VoxCPMEngine>,
```

with:

```rust
loaded_sample_rate: Option<u32>,
```

Change `GenerationRun` to:

```rust
pub struct GenerationRun {
    pub item_id: String,
    pub request: SynthesisRequest,
    pub sample_rate: u32,
    pub streaming: bool,
    pub cancel: Arc<AtomicBool>,
}
```

Change `mark_load_success` to accept sample rate instead of engine:

```rust
pub fn mark_load_success(&mut self, load_id: u64, choice_id: String, sample_rate: u32) -> bool {
    if !self.active_load_matches(load_id) {
        return false;
    }
    self.active_load = None;
    self.loaded_model_id = Some(choice_id);
    self.loaded_sample_rate = Some(sample_rate);
    self.load_state = LoadUiState::Idle;
    true
}
```

Change `begin_generation_run` sample rate selection:

```rust
let sample_rate = self.loaded_sample_rate.unwrap_or(16_000);
```

Remove `execute_generation_run`, direct `VoxCPMEngine` imports, and engine restoration in generation finish methods. Keep `finish_generation_success`, `finish_generation_failure`, and `finish_generation_canceled` state updates, but have success accept `item_id`, `samples`, `sample_rate`, and `duration_seconds`.

- [ ] **Step 4: Run AppCore tests**

Run:

```powershell
cargo test -p voxui-desktop app_core --lib
cargo test -p voxui-desktop queue_tests
```

Expected: tests pass after updating test helpers that previously used local engine assumptions.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/app_core.rs crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/tests
git commit -m "Refactor app core for sidecar inference"
```

---

### Task 7: Wire Sidecar Events into Desktop Commands

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/playback.rs`
- Test: `crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`

- [ ] **Step 1: Add failing test for audio chunk accumulation**

Add an AppCore test:

```rust
#[test]
fn streaming_audio_chunks_accumulate_until_done() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation("hello".to_string()).unwrap();
    core.begin_generation_for_test(&item.id).unwrap();

    core.append_generation_audio_chunk(&item.id, vec![0.1, 0.2], 16_000)
        .unwrap();
    core.append_generation_audio_chunk(&item.id, vec![0.3], 16_000)
        .unwrap();
    core.finish_generation_success_from_sidecar(&item.id, 16_000, 3.0 / 16_000.0)
        .unwrap();

    assert!(core.has_audio(&item.id));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test -p voxui-desktop streaming_audio_chunks_accumulate_until_done --lib
```

Expected: fails because chunk accumulation methods do not exist.

- [ ] **Step 3: Add AppCore sidecar completion methods**

Add to `AppCore`:

```rust
pub fn append_generation_audio_chunk(
    &mut self,
    item_id: &str,
    samples: Vec<f32>,
    sample_rate: u32,
) -> Result<()> {
    if self.active_generation_item_id.as_deref() != Some(item_id) {
        bail!("stale audio chunk for inactive item: {item_id}");
    }
    self.audio_cache
        .append(item_id.to_string(), samples, sample_rate)?;
    Ok(())
}

pub fn finish_generation_success_from_sidecar(
    &mut self,
    item_id: &str,
    sample_rate: u32,
    duration_seconds: f32,
) -> Result<()> {
    if self.active_generation_item_id.as_deref() != Some(item_id) {
        bail!("stale generation completion for inactive item: {item_id}");
    }
    self.clear_active_generation(item_id);
    if !self.audio_cache.contains(item_id) {
        self.audio_cache.insert(
            item_id.to_string(),
            GeneratedAudio {
                samples: Vec::new(),
                sample_rate,
            },
        );
    }
    self.queue.mark_ready(item_id);
    let _ = duration_seconds;
    Ok(())
}
```

Update `GeneratedAudioCache` in `playback.rs` to support append:

```rust
pub fn append(&mut self, item_id: String, samples: Vec<f32>, sample_rate: u32) -> Result<()> {
    match self.items.get_mut(&item_id) {
        Some(audio) => {
            if audio.sample_rate != sample_rate {
                bail!("sample rate changed for generated audio item {item_id}");
            }
            audio.samples.extend(samples);
        }
        None => {
            self.items.insert(item_id, GeneratedAudio { samples, sample_rate });
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Route sidecar events in `commands.rs`**

Replace in-process `load_engine_for_choice` and `spawn_generation` execution with:

```rust
fn handle_sidecar_event(window: &Window, shared: SharedAppCore, frame: Frame<SidecarEvent>) {
    match frame.header {
        SidecarEvent::ModelLoadProgress { load_id: _, phase, loaded_bytes, total_bytes, component, component_index, component_total } => {
            let _ = window.emit("model_load_progress", crate::types::ModelLoadProgressEvent {
                phase,
                loaded_bytes,
                total_bytes,
                component,
                component_index,
                component_total,
            });
        }
        SidecarEvent::ModelLoadDone { load_id, status, sample_rate, error } => {
            let done = match shared.lock() {
                Ok(mut core) => {
                    if status == OperationStatus::Success {
                        if let Some(rate) = sample_rate {
                            let selected = core.selected_choice().ok().map(|choice| choice.id);
                            if let Some(choice_id) = selected {
                                core.mark_load_success(load_id, choice_id, rate);
                            }
                        }
                    } else {
                        core.mark_load_finished_without_swap_for_load(load_id);
                    }
                    crate::types::ModelLoadDoneEvent {
                        status: format!("{status:?}").to_lowercase(),
                        selected_model_id: core.snapshot().selected_model_id,
                        loaded_model_id: core.snapshot().loaded_model_id,
                        error,
                    }
                }
                Err(_) => crate::types::ModelLoadDoneEvent {
                    status: "failed".to_string(),
                    selected_model_id: None,
                    loaded_model_id: None,
                    error: Some("app state lock poisoned".to_string()),
                },
            };
            let _ = window.emit("model_load_done", done);
        }
        SidecarEvent::GenerationProgress { item_id, current, total } => {
            if let Ok(mut core) = shared.lock() {
                core.mark_generation_progress(&item_id, current, total);
            }
            let _ = window.emit("generation_progress", GenerationProgressEvent { item_id, current, total });
        }
        SidecarEvent::AudioChunk { item_id, sample_rate, .. } => {
            if let Ok(samples) = crate::inference_sidecar::sidecar_samples_from_payload(&frame.payload) {
                if let Ok(mut core) = shared.lock() {
                    let _ = core.append_generation_audio_chunk(&item_id, samples, sample_rate);
                }
            }
        }
        SidecarEvent::AudioFinal { item_id, sample_rate, duration_seconds } => {
            if let Ok(samples) = crate::inference_sidecar::sidecar_samples_from_payload(&frame.payload) {
                if let Ok(mut core) = shared.lock() {
                    let _ = core.append_generation_audio_chunk(&item_id, samples, sample_rate);
                    let _ = core.finish_generation_success_from_sidecar(&item_id, sample_rate, duration_seconds);
                }
            }
        }
        SidecarEvent::GenerationDone { item_id, status, sample_rate, duration_seconds, error } => {
            let done = match shared.lock() {
                Ok(mut core) => {
                    match status {
                        OperationStatus::Success => {
                            if let (Some(rate), Some(duration)) = (sample_rate, duration_seconds) {
                                let _ = core.finish_generation_success_from_sidecar(&item_id, rate, duration);
                            }
                        }
                        OperationStatus::Canceled => {
                            if let Ok(run) = core.begin_generation_run(&item_id) {
                                core.finish_generation_canceled(run);
                            }
                        }
                        OperationStatus::Failed => {
                            if let Ok(run) = core.begin_generation_run(&item_id) {
                                core.finish_generation_failure(run, error.clone().unwrap_or_else(|| "generation failed".to_string()));
                            }
                        }
                    }
                    GenerationDoneEvent {
                        item_id,
                        status: format!("{status:?}").to_lowercase(),
                        error,
                        sample_rate,
                        duration_seconds,
                    }
                }
                Err(_) => GenerationDoneEvent {
                    item_id,
                    status: "failed".to_string(),
                    error: Some("app state lock poisoned".to_string()),
                    sample_rate: None,
                    duration_seconds: None,
                },
            };
            let _ = window.emit("generation_done", done);
            kick_generation_queue(window, shared);
        }
        SidecarEvent::Ready | SidecarEvent::Error { .. } => {}
    }
}
```

Before committing this task, run `rg -n "VoxCPMEngine|execute_generation_run|load_engine_for_choice" crates/voxui-desktop/src-tauri/src`. Expected: no matches in `app_core.rs` or `commands.rs` except references in deleted diff context.

- [ ] **Step 5: Run desktop tests**

Run:

```powershell
cargo test -p voxui-desktop
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/commands.rs crates/voxui-desktop/src-tauri/src/app_core.rs crates/voxui-desktop/src-tauri/src/playback.rs crates/voxui-desktop/src-tauri/tests
git commit -m "Wire sidecar events into desktop state"
```

---

### Task 8: Package Sidecar with Tauri

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/tauri.conf.json`
- Modify: `scripts/package-windows-cuda.ps1`
- Modify: `crates/voxui-inference-sidecar/Cargo.toml`

- [ ] **Step 1: Add sidecar binary to Tauri config**

Modify `crates/voxui-desktop/src-tauri/tauri.conf.json` bundle section:

```json
"bundle": {
  "active": false,
  "targets": "all",
  "externalBin": [
    "../../target/release/voxui-inference-sidecar"
  ],
  "icon": [
    "icons/32x32.png",
    "icons/128x128.png",
    "icons/128x128@2x.png",
    "icons/icon.ico"
  ]
}
```

Use the same relative path for development and release builds. The packaging script builds the release sidecar before the Tauri build, so the configured file exists when Tauri resolves external binaries.

- [ ] **Step 2: Update Windows CUDA packaging script**

Add this feature to `crates/voxui-inference-sidecar/Cargo.toml`:

```toml
[features]
default = []
cuda = ["voxui-inference/cuda"]
```

Add before the Tauri build command in `scripts/package-windows-cuda.ps1`:

```powershell
cargo build --release -p voxui-inference-sidecar --features cuda
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build voxui-inference-sidecar"
}
```

- [ ] **Step 3: Run sidecar release build**

Run:

```powershell
cargo build --release -p voxui-inference-sidecar
```

Expected: sidecar binary builds successfully.

- [ ] **Step 4: Run Tauri desktop check build**

Run:

```powershell
cargo check -p voxui-desktop
```

Expected: desktop crate compiles.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/tauri.conf.json scripts/package-windows-cuda.ps1 crates/voxui-inference-sidecar/Cargo.toml
git commit -m "Package inference sidecar"
```

---

### Task 9: End-to-End Verification

**Files:**
- No planned source changes unless verification exposes defects.

- [ ] **Step 1: Run focused crate tests**

Run:

```powershell
cargo test -p voxui-sidecar-protocol
cargo test -p voxui-audio
cargo test -p voxui-inference-sidecar
cargo test -p voxui-desktop
```

Expected: all pass.

- [ ] **Step 2: Run workspace check**

Run:

```powershell
cargo check --workspace
```

Expected: workspace compiles.

- [ ] **Step 3: Run desktop app manually**

Run:

```powershell
cargo tauri dev
```

Expected:

- model loading starts the sidecar and reports progress;
- streaming generation plays audio before generation completes;
- non-streaming generation emits final audio only after completion;
- canceling active generation stops playback and starts the next queued item;
- replay of completed history uses cached audio.

- [ ] **Step 4: Fix defects with narrow commits**

For each defect, write a failing test first when the defect is testable. Use commit messages of the form:

```powershell
git commit -m "Fix sidecar cancellation race"
git commit -m "Fix streaming audio drain completion"
```

- [ ] **Step 5: Final status**

Run:

```powershell
git status --short
```

Expected: clean worktree.

---

## Self-Review Notes

- The plan covers the approved spec sections: sidecar process, Tauri-owned queueing, streaming and non-streaming modes, cancellation, model load progress, audio engine upgrade, packaging, and verification.
- IPC framing is explicitly length-prefixed stdio with JSON headers and optional binary payloads.
- The internal audio path remains PCM; encoded browser/media playback is excluded.
- The first implementation keeps completed audio cache memory-only and uses a 250 ms prebuffer.
