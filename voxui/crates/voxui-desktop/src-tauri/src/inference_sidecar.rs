use std::io::{BufWriter, ErrorKind, Read};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use anyhow::{Context, Result};
use voxui_sidecar_protocol::{
    f32_samples_from_le_bytes, read_frame, write_frame, Frame, SidecarCommand, SidecarEvent,
};

#[derive(Debug)]
pub enum SidecarReaderEvent {
    Frame(Frame<SidecarEvent>),
    Error(String),
    Eof,
}

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

pub struct SidecarProcess {
    child: Child,
    writer: Option<BufWriter<ChildStdin>>,
}

impl SidecarProcess {
    pub fn spawn(
        sidecar_path: impl AsRef<Path>,
    ) -> Result<(Self, mpsc::Receiver<SidecarReaderEvent>)> {
        let sidecar_dir = sidecar_path
            .as_ref()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        let sidecar_exe = sidecar_path.as_ref().to_path_buf();
        tracing::info!("spawning inference sidecar: {}", sidecar_exe.display());
        let mut cmd = Command::new(&sidecar_exe);
        cmd.current_dir(&sidecar_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn()
            .with_context(|| format!("spawn sidecar {}", sidecar_exe.display()))?;
        let stdin = child.stdin.take().context("sidecar stdin unavailable")?;
        let stdout = child.stdout.take().context("sidecar stdout unavailable")?;
        let (sender, receiver) = mpsc::channel();

        spawn_reader(stdout, sender);

        Ok((
            Self {
                child,
                writer: Some(BufWriter::new(stdin)),
            },
            receiver,
        ))
    }

    pub fn send(&mut self, command: SidecarCommand) -> Result<()> {
        tracing::debug!(
            command = sidecar_command_name(&command),
            "sending sidecar command"
        );
        let writer = self.writer.as_mut().context("sidecar stdin is closed")?;
        write_frame(
            writer,
            &Frame {
                header: command,
                payload: Vec::new(),
            },
        )
    }

    pub fn shutdown(mut self) -> Result<()> {
        self.send(SidecarCommand::Shutdown)
            .context("send sidecar shutdown")?;
        self.close_stdin();
        if self.wait_for_exit(SHUTDOWN_TIMEOUT)? {
            Ok(())
        } else {
            self.kill()?;
            anyhow::bail!("sidecar did not exit within {:?}", SHUTDOWN_TIMEOUT)
        }
    }

    pub fn kill(&mut self) -> Result<()> {
        self.close_stdin();
        match self.child.try_wait()? {
            Some(_) => Ok(()),
            None => {
                match self.child.kill() {
                    Ok(()) => {}
                    Err(error) if error.kind() == ErrorKind::InvalidInput => {}
                    Err(error) => return Err(error).context("kill sidecar process"),
                }
                self.child.wait().context("wait for killed sidecar")?;
                Ok(())
            }
        }
    }

    fn close_stdin(&mut self) {
        self.writer.take();
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> Result<bool> {
        let start = Instant::now();
        loop {
            if self.child.try_wait()?.is_some() {
                return Ok(true);
            }
            if start.elapsed() >= timeout {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for SidecarProcess {
    fn drop(&mut self) {
        let _ = self.kill();
    }
}

fn spawn_reader(stdout: ChildStdout, sender: mpsc::Sender<SidecarReaderEvent>) {
    thread::spawn(move || {
        read_sidecar_frames(stdout, sender);
    });
}

pub fn read_sidecar_frames<R: Read>(mut reader: R, sender: mpsc::Sender<SidecarReaderEvent>) {
    loop {
        match read_frame(&mut reader) {
            Ok(frame) => {
                if sender.send(SidecarReaderEvent::Frame(frame)).is_err() {
                    break;
                }
            }
            Err(error) if is_clean_eof(&error) => {
                let _ = sender.send(SidecarReaderEvent::Eof);
                break;
            }
            Err(error) => {
                let _ = sender.send(SidecarReaderEvent::Error(error.to_string()));
                break;
            }
        }
    }
}

fn is_clean_eof(error: &anyhow::Error) -> bool {
    error.to_string().contains("sidecar protocol clean eof")
}

fn sidecar_command_name(command: &SidecarCommand) -> &'static str {
    match command {
        SidecarCommand::LoadModel { .. } => "load_model",
        SidecarCommand::CancelLoad { .. } => "cancel_load",
        SidecarCommand::Synthesize { .. } => "synthesize",
        SidecarCommand::CancelSynthesis { .. } => "cancel_synthesis",
        SidecarCommand::Shutdown => "shutdown",
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_reaps_running_child() {
        let mut process = test_process();

        process.kill().unwrap();

        assert!(process.child.try_wait().unwrap().is_some());
        process.kill().unwrap();
    }

    fn test_process() -> SidecarProcess {
        let mut command = long_running_command();
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        SidecarProcess {
            child,
            writer: Some(BufWriter::new(stdin)),
        }
    }

    #[cfg(windows)]
    fn long_running_command() -> Command {
        let mut command = Command::new("powershell");
        command.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 30"]);
        command
    }

    #[cfg(not(windows))]
    fn long_running_command() -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30"]);
        command
    }
}
