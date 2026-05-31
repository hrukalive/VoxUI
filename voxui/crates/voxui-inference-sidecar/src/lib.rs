use std::cell::RefCell;
use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use candle_core::Device;
use voxui_inference::{SynthesisChunk, SynthesisRequest, VoxCPMEngine};
use voxui_sidecar_protocol::{
    f32_samples_to_le_bytes, read_frame, write_frame, BackendKind, Frame, OperationStatus,
    SidecarCommand, SidecarEvent, SynthesisRequestDto,
};

#[derive(Default)]
pub struct SidecarEngine {
    engine: Option<VoxCPMEngine>,
    load_cancel: SharedCancel<u64>,
    generation_cancel: SharedCancel<String>,
}

impl SidecarEngine {
    pub fn run<R, W>(&mut self, reader: R, mut writer: W) -> Result<()>
    where
        R: Read + Send + 'static,
        W: Write,
    {
        emit_frame(&mut writer, ready_frame())?;
        tracing::debug!("sidecar emitted ready event");
        let commands = spawn_command_reader(
            reader,
            self.load_cancel.clone(),
            self.generation_cancel.clone(),
        );

        while let Ok(command) = commands.recv() {
            let command = command?;
            tracing::debug!(
                command = sidecar_command_name(&command),
                "sidecar received command"
            );
            match self.handle_command_write(command, &mut writer) {
                Ok(shutdown) => {
                    if shutdown {
                        break;
                    }
                }
                Err(error) if is_broken_pipe(&error) => {
                    tracing::debug!("parent process closed the pipe, exiting");
                    break;
                }
                Err(error) => return Err(error),
            }
        }

        Ok(())
    }

    fn handle_command_write<W>(&mut self, command: SidecarCommand, writer: &mut W) -> Result<bool>
    where
        W: Write,
    {
        self.handle_command_with_emit(command, |frame| emit_frame(writer, frame))
    }

    #[cfg(test)]
    fn handle_command(
        &mut self,
        command: SidecarCommand,
        events: &mut Vec<Frame<SidecarEvent>>,
    ) -> Result<bool> {
        self.handle_command_with_emit(command, |frame| {
            events.push(frame);
            Ok(())
        })
    }

    fn handle_command_with_emit<F>(&mut self, command: SidecarCommand, mut emit: F) -> Result<bool>
    where
        F: FnMut(Frame<SidecarEvent>) -> Result<()>,
    {
        match command {
            SidecarCommand::LoadModel {
                load_id,
                model_dir,
                backend,
            } => {
                tracing::info!(
                    load_id,
                    model_dir = %model_dir.display(),
                    backend = ?backend,
                    "sidecar starting model load"
                );
                self.cancel_active_generation();
                self.cancel_active_load();
                let cancel = Arc::new(AtomicBool::new(false));
                install_active_cancel(&self.load_cancel, load_id, cancel.clone());
                let device = device_for_backend(backend);

                let load_result = match device {
                    Ok(device) => {
                        let emit_cell = RefCell::new(&mut emit);
                        let progress_error = RefCell::new(None);
                        let cancel_for_progress = cancel.clone();
                        let loaded = VoxCPMEngine::load_with_progress(
                            &model_dir,
                            device,
                            |component_index, component_total| {
                                let event = Frame {
                                    header: SidecarEvent::ModelLoadProgress {
                                        load_id,
                                        phase: "loading".to_string(),
                                        loaded_bytes: component_index as u64,
                                        total_bytes: component_total as u64,
                                        component: None,
                                        component_index,
                                        component_total,
                                    },
                                    payload: Vec::new(),
                                };
                                if let Err(error) = (emit_cell.borrow_mut())(event) {
                                    *progress_error.borrow_mut() = Some(error);
                                    cancel_for_progress.store(true, Ordering::Relaxed);
                                }
                            },
                            Some(cancel.as_ref()),
                        );
                        let progress_error = progress_error.into_inner();
                        match (loaded, progress_error) {
                            (_, Some(error)) => Err(error),
                            (Ok(engine), None) => Ok(engine),
                            (Err(error), None) => Err(error),
                        }
                    }
                    Err(error) => Err(error),
                };

                let canceled = cancel.load(Ordering::Relaxed);
                clear_matching_cancel(&self.load_cancel, &cancel);

                match load_result {
                    Ok(engine) if !canceled => {
                        let sample_rate = engine.sample_rate();
                        tracing::info!(load_id, sample_rate, "sidecar model load succeeded");
                        self.engine = Some(engine);
                        emit(Frame {
                            header: SidecarEvent::ModelLoadDone {
                                load_id,
                                status: OperationStatus::Success,
                                sample_rate: Some(sample_rate),
                                error: None,
                            },
                            payload: Vec::new(),
                        })?;
                    }
                    Ok(_) => {
                        emit(model_load_done_canceled(load_id))?;
                    }
                    Err(error) if canceled || is_cancel_error(&error) => {
                        tracing::info!(load_id, error = %error, "sidecar model load canceled");
                        emit(model_load_done_canceled(load_id))?;
                    }
                    Err(error) => {
                        tracing::error!(load_id, error = %error, "sidecar model load failed");
                        emit(Frame {
                            header: SidecarEvent::ModelLoadDone {
                                load_id,
                                status: OperationStatus::Failed,
                                sample_rate: None,
                                error: Some(error.to_string()),
                            },
                            payload: Vec::new(),
                        })?;
                    }
                }
            }
            SidecarCommand::CancelLoad { load_id } => {
                request_cancel(&self.load_cancel, load_id);
                emit(model_load_done_canceled(load_id))?;
            }
            SidecarCommand::Synthesize {
                item_id,
                request,
                streaming,
            } => {
                self.cancel_active_generation();
                let Some(engine) = self.engine.as_mut() else {
                    emit(Frame {
                        header: SidecarEvent::GenerationDone {
                            item_id,
                            status: OperationStatus::Failed,
                            sample_rate: None,
                            duration_seconds: None,
                            error: Some("model is not loaded".to_string()),
                        },
                        payload: Vec::new(),
                    })?;
                    return Ok(false);
                };

                let cancel = Arc::new(AtomicBool::new(false));
                install_active_cancel(&self.generation_cancel, item_id.clone(), cancel.clone());
                let request = synthesis_request_from_dto(request);

                if let Err(error) = engine.reconcile_lora(request.lora_path.as_deref()) {
                    emit(Frame {
                        header: SidecarEvent::GenerationDone {
                            item_id: item_id.clone(),
                            status: OperationStatus::Failed,
                            sample_rate: None,
                            duration_seconds: None,
                            error: Some(format!("failed to load LoRA: {error}")),
                        },
                        payload: Vec::new(),
                    })?;
                    return Ok(false);
                }

                let result = if streaming {
                    run_streaming_generation(
                        engine,
                        item_id.clone(),
                        request,
                        cancel.clone(),
                        &mut emit,
                    )
                } else {
                    run_non_streaming_generation(
                        engine,
                        item_id.clone(),
                        request,
                        cancel.clone(),
                        &mut emit,
                    )
                };
                let canceled = cancel.load(Ordering::Relaxed);
                clear_matching_cancel(&self.generation_cancel, &cancel);

                for frame in generation_done_for_result(item_id, result, canceled) {
                    emit(frame)?;
                }
            }
            SidecarCommand::CancelSynthesis { item_id } => {
                self.cancel_active_generation();
                emit(generation_done_canceled(item_id))?;
            }
            SidecarCommand::Shutdown => {
                self.cancel_active_load();
                self.cancel_active_generation();
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn cancel_active_load(&mut self) {
        if let Some(cancel) = take_active_cancel(&self.load_cancel) {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    fn cancel_active_generation(&mut self) {
        if let Some(cancel) = active_cancel(&self.generation_cancel) {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

#[derive(Default)]
struct CancelState<I> {
    active: Option<(I, Arc<AtomicBool>)>,
    pending: Option<I>,
}

type SharedCancel<I> = Arc<Mutex<CancelState<I>>>;

fn spawn_command_reader<R>(
    mut reader: R,
    load_cancel: SharedCancel<u64>,
    generation_cancel: SharedCancel<String>,
) -> mpsc::Receiver<Result<SidecarCommand>>
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("voxui-sidecar-command-reader".to_string())
        .spawn(move || loop {
            let frame: Frame<SidecarCommand> = match read_frame(&mut reader) {
                Ok(frame) => frame,
                Err(error) if is_clean_eof(&error) => break,
                Err(error) => {
                    tracing::error!("failed to read sidecar command: {error}");
                    let _ = sender.send(Err(error));
                    break;
                }
            };

            if let Some(command) =
                handle_immediate_command(frame.header, &load_cancel, &generation_cancel)
            {
                if sender.send(Ok(command)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn sidecar command reader");
    receiver
}

fn handle_immediate_command(
    command: SidecarCommand,
    load_cancel: &SharedCancel<u64>,
    generation_cancel: &SharedCancel<String>,
) -> Option<SidecarCommand> {
    match command {
        SidecarCommand::CancelLoad { load_id } => {
            request_cancel(load_cancel, load_id);
            None
        }
        SidecarCommand::CancelSynthesis { item_id } => {
            request_cancel(generation_cancel, item_id);
            None
        }
        SidecarCommand::Shutdown => {
            if let Some(cancel) = active_cancel(load_cancel) {
                cancel.store(true, Ordering::Relaxed);
            }
            if let Some(cancel) = active_cancel(generation_cancel) {
                cancel.store(true, Ordering::Relaxed);
            }
            Some(SidecarCommand::Shutdown)
        }
        other => Some(other),
    }
}

fn install_active_cancel<I>(slot: &SharedCancel<I>, id: I, cancel: Arc<AtomicBool>)
where
    I: Clone + Eq,
{
    let mut state = slot.lock().expect("active cancel lock poisoned");
    if state.pending.as_ref().is_some_and(|pending| pending == &id) {
        state.pending = None;
        cancel.store(true, Ordering::Relaxed);
    }
    state.active = Some((id, cancel));
}

fn request_cancel<I>(slot: &SharedCancel<I>, id: I)
where
    I: Clone + Eq,
{
    let mut state = slot.lock().expect("active cancel lock poisoned");
    if let Some((active_id, cancel)) = state.active.as_ref() {
        if active_id == &id {
            cancel.store(true, Ordering::Relaxed);
            return;
        }
    }
    state.pending = Some(id);
}

fn active_cancel<I>(slot: &SharedCancel<I>) -> Option<Arc<AtomicBool>> {
    slot.lock()
        .expect("active cancel lock poisoned")
        .active
        .as_ref()
        .map(|(_, cancel)| cancel.clone())
}

fn take_active_cancel<I>(slot: &SharedCancel<I>) -> Option<Arc<AtomicBool>> {
    slot.lock()
        .expect("active cancel lock poisoned")
        .active
        .take()
        .map(|(_, cancel)| cancel)
}

fn clear_matching_cancel<I>(slot: &SharedCancel<I>, cancel: &Arc<AtomicBool>) {
    let mut state = slot.lock().expect("active cancel lock poisoned");
    if state
        .active
        .as_ref()
        .is_some_and(|(_, current)| Arc::ptr_eq(current, cancel))
    {
        state.active = None;
    }
}

#[derive(Debug, PartialEq)]
struct GenerationOutcome {
    sample_rate: u32,
    duration_seconds: f32,
}

fn generation_done_for_result(
    item_id: String,
    result: Result<GenerationOutcome>,
    canceled: bool,
) -> Vec<Frame<SidecarEvent>> {
    let frame = match result {
        Ok(outcome) if !canceled => Frame {
            header: SidecarEvent::GenerationDone {
                item_id,
                status: OperationStatus::Success,
                sample_rate: Some(outcome.sample_rate),
                duration_seconds: Some(outcome.duration_seconds),
                error: None,
            },
            payload: Vec::new(),
        },
        Ok(_) => generation_done_canceled(item_id),
        Err(error) if canceled || is_cancel_error(&error) => generation_done_canceled(item_id),
        Err(error) => Frame {
            header: SidecarEvent::GenerationDone {
                item_id,
                status: OperationStatus::Failed,
                sample_rate: None,
                duration_seconds: None,
                error: Some(error.to_string()),
            },
            payload: Vec::new(),
        },
    };

    vec![frame]
}

fn run_non_streaming_generation<F>(
    engine: &mut VoxCPMEngine,
    item_id: String,
    request: SynthesisRequest,
    cancel: Arc<AtomicBool>,
    emit: &mut F,
) -> Result<GenerationOutcome>
where
    F: FnMut(Frame<SidecarEvent>) -> Result<()>,
{
    let sample_rate = engine.sample_rate();
    let progress_error = RefCell::new(None);
    let cancel_for_progress = cancel.clone();
    let samples = {
        let emit_cell = RefCell::new(&mut *emit);
        engine.generate_cancellable(
            request,
            |current, total| {
                let event = Frame {
                    header: SidecarEvent::GenerationProgress {
                        item_id: item_id.clone(),
                        current,
                        total,
                    },
                    payload: Vec::new(),
                };
                if let Err(error) = (emit_cell.borrow_mut())(event) {
                    *progress_error.borrow_mut() = Some(error);
                    cancel_for_progress.store(true, Ordering::Relaxed);
                }
            },
            Some(cancel.as_ref()),
        )?
    };
    if let Some(error) = progress_error.into_inner() {
        return Err(error);
    }

    let duration_seconds = duration_seconds(samples.len(), sample_rate);
    emit(Frame {
        header: SidecarEvent::AudioFinal {
            item_id: item_id.clone(),
            sample_rate,
            duration_seconds,
        },
        payload: f32_samples_to_le_bytes(&samples),
    })?;
    Ok(GenerationOutcome {
        sample_rate,
        duration_seconds,
    })
}

fn run_streaming_generation<F>(
    engine: &mut VoxCPMEngine,
    item_id: String,
    request: SynthesisRequest,
    cancel: Arc<AtomicBool>,
    emit: &mut F,
) -> Result<GenerationOutcome>
where
    F: FnMut(Frame<SidecarEvent>) -> Result<()>,
{
    let mut total_samples = 0usize;
    let mut sample_rate = engine.sample_rate();
    engine.generate_streaming_cancellable(
        request,
        |chunk| emit_streaming_chunk(&item_id, &mut total_samples, &mut sample_rate, chunk, emit),
        Some(cancel.as_ref()),
    )?;
    let duration_seconds = duration_seconds(total_samples, sample_rate);
    Ok(GenerationOutcome {
        sample_rate,
        duration_seconds,
    })
}

fn emit_streaming_chunk<F>(
    item_id: &str,
    total_samples: &mut usize,
    sample_rate: &mut u32,
    chunk: SynthesisChunk,
    emit: &mut F,
) -> Result<()>
where
    F: FnMut(Frame<SidecarEvent>) -> Result<()>,
{
    *sample_rate = chunk.sample_rate;
    *total_samples += chunk.samples.len();
    emit(Frame {
        header: SidecarEvent::GenerationProgress {
            item_id: item_id.to_string(),
            current: chunk.generated_patch_count,
            total: chunk.max_patches,
        },
        payload: Vec::new(),
    })?;
    emit(Frame {
        header: SidecarEvent::AudioChunk {
            item_id: item_id.to_string(),
            sample_rate: chunk.sample_rate,
            current: chunk.patch_index + 1,
            total: chunk.max_patches,
            is_final: chunk.is_final,
        },
        payload: f32_samples_to_le_bytes(&chunk.samples),
    })
}

fn synthesis_request_from_dto(dto: SynthesisRequestDto) -> SynthesisRequest {
    SynthesisRequest {
        text: dto.text,
        prompt_wav_path: dto.prompt_wav_path,
        prompt_text: dto.prompt_text,
        reference_wav_path: dto.reference_wav_path,
        cfg_value: dto.cfg_value,
        inference_timesteps: dto.inference_timesteps,
        min_len: dto.min_len,
        max_len: dto.max_len,
        retry_badcase: dto.retry_badcase,
        retry_badcase_max_times: dto.retry_badcase_max_times,
        retry_badcase_ratio_threshold: dto.retry_badcase_ratio_threshold,
        consolidate_n: dto.consolidate_n,
        lora_path: dto.lora_path,
        ..SynthesisRequest::default()
    }
}

fn device_for_backend(backend: BackendKind) -> Result<Device> {
    match backend {
        BackendKind::Cpu => Ok(Device::Cpu),
        BackendKind::Cuda => Device::new_cuda(0).context("initialize CUDA device 0"),
    }
}

fn emit_frame<W>(writer: &mut W, frame: Frame<SidecarEvent>) -> Result<()>
where
    W: Write,
{
    write_frame(writer, &frame)
}

fn ready_frame() -> Frame<SidecarEvent> {
    let cuda_available = sidecar_cuda_available();
    Frame {
        header: SidecarEvent::Ready {
            cuda_available,
            default_backend: if cuda_available {
                BackendKind::Cuda
            } else {
                BackendKind::Cpu
            },
        },
        payload: Vec::new(),
    }
}

fn sidecar_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

fn model_load_done_canceled(load_id: u64) -> Frame<SidecarEvent> {
    Frame {
        header: SidecarEvent::ModelLoadDone {
            load_id,
            status: OperationStatus::Canceled,
            sample_rate: None,
            error: None,
        },
        payload: Vec::new(),
    }
}

fn generation_done_canceled(item_id: String) -> Frame<SidecarEvent> {
    Frame {
        header: SidecarEvent::GenerationDone {
            item_id,
            status: OperationStatus::Canceled,
            sample_rate: None,
            duration_seconds: None,
            error: None,
        },
        payload: Vec::new(),
    }
}

fn duration_seconds(sample_count: usize, sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        0.0
    } else {
        sample_count as f32 / sample_rate as f32
    }
}

fn is_cancel_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("cancelled") || message.contains("canceled")
}

fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == ErrorKind::BrokenPipe)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn cancel_generation_sets_active_cancel_flag() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut sidecar = SidecarEngine::default();
        install_active_cancel(
            &sidecar.generation_cancel,
            "item-1".to_string(),
            cancel.clone(),
        );
        let mut events = Vec::new();

        sidecar
            .handle_command(
                SidecarCommand::CancelSynthesis {
                    item_id: "item-1".to_string(),
                },
                &mut events,
            )
            .unwrap();

        assert!(cancel.load(Ordering::Relaxed));
        assert_eq!(
            events,
            vec![Frame {
                header: SidecarEvent::GenerationDone {
                    item_id: "item-1".to_string(),
                    status: OperationStatus::Canceled,
                    sample_rate: None,
                    duration_seconds: None,
                    error: None,
                },
                payload: Vec::new(),
            }]
        );
    }

    #[test]
    fn immediate_cancel_synthesis_sets_flag_without_forwarding_command() {
        let cancel = Arc::new(AtomicBool::new(false));
        let load_cancel = SharedCancel::default();
        let generation_cancel = SharedCancel::default();
        install_active_cancel(&generation_cancel, "item-1".to_string(), cancel.clone());

        let forwarded = handle_immediate_command(
            SidecarCommand::CancelSynthesis {
                item_id: "item-1".to_string(),
            },
            &load_cancel,
            &generation_cancel,
        );

        assert!(forwarded.is_none());
        assert!(cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn prequeued_cancel_applies_when_matching_generation_is_installed() {
        let cancel = Arc::new(AtomicBool::new(false));
        let generation_cancel = SharedCancel::default();

        request_cancel(&generation_cancel, "item-1".to_string());
        install_active_cancel(&generation_cancel, "item-1".to_string(), cancel.clone());

        assert!(cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn only_most_recent_pending_cancel_is_retained() {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let generation_cancel = SharedCancel::default();

        request_cancel(&generation_cancel, "item-1".to_string());
        request_cancel(&generation_cancel, "item-2".to_string());
        install_active_cancel(&generation_cancel, "item-1".to_string(), first.clone());
        install_active_cancel(&generation_cancel, "item-2".to_string(), second.clone());

        assert!(!first.load(Ordering::Relaxed));
        assert!(second.load(Ordering::Relaxed));
    }

    #[test]
    fn canceled_success_result_emits_only_canceled_generation_done() {
        let frames = generation_done_for_result(
            "item-1".to_string(),
            Ok(GenerationOutcome {
                sample_rate: 24_000,
                duration_seconds: 1.5,
            }),
            true,
        );

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], generation_done_canceled("item-1".to_string()));
    }
}
