# voxui-cli Design

A CLI crate for end-to-end interactive testing of the VoxCPM TTS system with streaming synthesis and real-time audio playback.

## CLI Interface

```
voxui-cli --model <MODEL_DIR> [--lora <LORA_GGUF>] [--cuda]
```

| Flag | Required | Description |
|---|---|---|
| `--model <PATH>` | Yes | Directory containing `model.gguf`, `config.json`, `tokenizer.json` |
| `--lora <PATH>` | No | Path to a LoRA `.gguf` file to apply after model load |
| `--cuda` | No | Use CUDA device (default: CPU) |

**Validation at startup:**
- Model directory must exist and contain `model.gguf`, `config.json`, `tokenizer.json`
- LoRA path, if provided, must exist and end with `.gguf`
- If `--cuda` is passed but CUDA is unavailable, exit with error

## Crate Structure

```
voxui/crates/voxui-cli/
  Cargo.toml
  src/
    main.rs          -- entry point, device creation, REPL loop glue
    args.rs          -- clap derive Args struct, path validation
    runner.rs        -- Runner: engine lifecycle, synthesis dispatch, playback
```

Add `"crates/voxui-cli"` to `voxui/Cargo.toml` workspace members.

**Dependencies:** `clap` (derive), `rustyline` (REPL with history/line-editing), `voxui-inference`, `voxui-audio`, `anyhow`, `itertools`, `indicatif` (progress bars).

## Module Responsibilities

### `args.rs`
- `Args` struct with clap derive (`--model`, `--lora`, `--cuda`)
- `Args::validate()` — checks model dir contains required files, LoRA file exists
- Returns early `anyhow::Error` on validation failures

### `runner.rs`
- `Runner` struct holding `VoxCPMEngine` + optional `LoraAdapter`
- `Runner::load(args, device) -> Result<Self>` — loads model with progress bar, applies LoRA if provided
- `Runner::synthesize_and_play(text, cancel) -> Result<()>` — creates `SynthesisRequest` with defaults, calls `generate_streaming_cancellable`, pushes audio chunks to `StreamingPlayer`, returns when playback completes or is cancelled
- Displays summary line after load: variant, device type, LoRA path (if any)

### `main.rs`
- Parse args, validate, print fatal errors to stderr, exit 1
- Create device (`Device::Cpu` or `Device::new_cuda(0)`)
- Load `Runner`
- Enter `rustyline` REPL loop
- Install `ctrlc` SIGINT handler that sets an `AtomicBool`; checked in REPL (exit) and synthesis (cancel)

## REPL Flow

```
Model: VoxCPM 1.5  |  Device: CUDA  |  LoRA: my_voice.gguf
Type text to synthesize. Empty line or /quit to exit.

> Hello, world!
  Synthesizing... [====>                    ] 12/200 patches
  Done.

> This is a longer sentence that takes more time.
  Synthesizing... [==========>              ] 28/200 patches
  Done.

>                                           ← empty line exits
Goodbye.
```

- Empty line, `/quit`, `/exit`, or Ctrl+C at the prompt exits
- Progress line prints patch count during streaming synthesis (overwritten each chunk via `\r`)
- `Done.` printed after playback completes
- If a synthesis call fails, print error to stderr and return to prompt

## Streaming Playback

### StreamingPlayer (new type in `voxui-audio`)

A real-time streaming audio player with resampling support:

```rust
pub struct StreamingPlayer {
    stream: cpal::Stream,
    resampler: r8brain::HiresFixedResampler<f64>,
    sender: ringbuf::Producer<f32>,
    device_sample_rate: u32,
}

impl StreamingPlayer {
    pub fn new(source_sample_rate: u32, pre_buffer_secs: f32) -> Result<Self>;
    pub fn push(&mut self, samples: &[f32]);
    pub fn flush(&self);
}
```

- **Constructor:** Opens default output device via CPAL, creates r8brain resampler (source rate to device native rate), creates lock-free SPSC ring buffer sized for `pre_buffer_secs` at source rate, spawns CPAL output stream in callback mode that pulls from ring buffer
- **push():** Resamples chunk from source rate to device rate via r8brain, pushes resampled samples into ring buffer. Non-blocking; ring buffer has fixed capacity and applies backpressure if full
- **flush():** Blocks until ring buffer is empty (all audio consumed by CPAL callback)

### REPL Synthesis Flow

1. Create `StreamingPlayer` with engine's sample rate and 1-second pre-buffer
2. Call `generate_streaming_cancellable()` with cancel token
3. On each `SynthesisChunk`: update progress line, call `player.push(&chunk.samples)`
4. After all chunks emitted: `player.flush()`, print `Done.`

### Buffer Sizing

| Variant | Engine SR | Chunk Duration | Ring Buffer Size |
|---|---|---|---|
| VoxCPM 0.5 | 16kHz | ~40ms | 16k samples (~1s) |
| VoxCPM 1.5 | 44.1kHz | ~40ms | 44.1k samples (~1s) |
| VoxCPM 2 | 48kHz | ~40ms | 48k samples (~1s) |

### New dependency for `voxui-audio`
- `ringbuf` — lock-free SPSC ring buffer

## Device Support

| Flag | Device | Notes |
|---|---|---|
| (none) | `Device::Cpu` | Always available |
| `--cuda` | `Device::new_cuda(0)` | Requires CUDA feature; build with `--features cuda` and set env vars: `CUDA_PATH`, `PATH` (MSVC tools), `CUDA_COMPUTE_CAP`, `NVCC_APPEND_FLAGS` |

The `voxui-cli` Cargo.toml forwards the `cuda` feature to `voxui-inference`.

## Error Handling

| Scenario | Behavior |
|---|---|
| Missing `model.gguf` / `config.json` / `tokenizer.json` | Fatal: exit 1, print missing file |
| LoRA path doesn't exist | Fatal: exit 1 |
| LoRA incompatible with model | Fatal during load: exit 1 |
| CUDA unavailable but `--cuda` passed | Fatal: exit 1 |
| Model load fails (OOM, bad GGUF) | Fatal: exit 1 |
| Synthesis fails (bad input, runtime error) | Print error to stderr, return to prompt |
| Audio device fails | Print error, return to prompt |
| Ctrl+C during synthesis | Cancel generation cleanly, print "Cancelled.", return to prompt |
| Ctrl+C at prompt | Exit program cleanly |

## Testing

Given hardware dependencies (GPU, audio device), integration tests are impractical. Unit tests cover:

- `args.rs`: missing model dir, missing required files, invalid LoRA path, LoRA file doesn't exist
- `runner.rs`: device creation (CPU, optional CUDA)
- `voxui-audio`: `StreamingPlayer` basic lifecycle (constructor, push, flush) when audio device is available

The CLI itself serves as manual integration testing for the full pipeline.
