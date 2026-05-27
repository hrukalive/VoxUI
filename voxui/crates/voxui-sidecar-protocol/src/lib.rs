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
