use std::io::BufWriter;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result};
use voxui_sidecar_protocol::{
    f32_samples_from_le_bytes, read_frame, write_frame, Frame, SidecarCommand, SidecarEvent,
};

pub struct SidecarProcess {
    child: Child,
    writer: BufWriter<ChildStdin>,
}

impl SidecarProcess {
    pub fn spawn(
        sidecar_path: impl AsRef<Path>,
    ) -> Result<(Self, mpsc::Receiver<Frame<SidecarEvent>>)> {
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
        write_frame(
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
    thread::spawn(move || {
        let mut stdout = stdout;
        loop {
            match read_frame(&mut stdout) {
                Ok(frame) => {
                    if sender.send(frame).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

pub fn is_active_generation_event(active_item_id: Option<&str>, event: &SidecarEvent) -> bool {
    match event {
        SidecarEvent::GenerationProgress { item_id, .. }
        | SidecarEvent::AudioChunk { item_id, .. }
        | SidecarEvent::AudioFinal { item_id, .. }
        | SidecarEvent::GenerationDone { item_id, .. } => active_item_id == Some(item_id.as_str()),
        _ => true,
    }
}

pub fn sidecar_samples_from_payload(payload: &[u8]) -> Result<Vec<f32>> {
    f32_samples_from_le_bytes(payload)
}
