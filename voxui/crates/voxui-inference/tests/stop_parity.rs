use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use voxui_inference::{engine::StopTraceStep, SynthesisRequest, VoxCPMEngine};

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

#[test]
fn voxcpm05_stop_steps_match_python_trace() -> Result<()> {
    let root = repo_root();
    let trace = voxui_inference::trace::TraceCase::load(root.join("goldens/voxcpm05_stop_parity"))?;
    let trace_request = trace.request();
    let request = SynthesisRequest {
        text: trace_request.text.clone(),
        cfg_value: trace_request.cfg_value,
        inference_timesteps: trace_request.inference_timesteps,
        min_len: trace_request.min_len,
        max_len: trace_request.max_len,
        retry_badcase: trace_request.retry_badcase,
        ..SynthesisRequest::default()
    };

    let model_dir = root.join("models/voxcpm05-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu)?;
    let debug =
        engine.generate_debug_stop_trace_with_noise(request, trace.tensor("first_dit_noise")?)?;

    let expected_step_count = trace.metadata_usize("generated_step_count")?;
    assert_eq!(debug.generated_step_count, expected_step_count);

    let expected_stop_decisions = trace.u32_list("stop_decisions")?;
    let actual_stop_decisions = debug
        .steps
        .iter()
        .map(|step| step.stop_decision)
        .collect::<Vec<_>>();
    assert_eq!(actual_stop_decisions, expected_stop_decisions);

    voxui_inference::trace::assert_close(
        &stop_logits_by_step(&debug.steps)?,
        &trace.tensor("stop_logits_by_step")?,
        3e-3,
    )?;
    voxui_inference::trace::assert_close(
        &stats_by_step(debug.steps.iter().map(|step| step.lm_hidden_stats))?,
        &trace.tensor("lm_hidden_stats_by_step")?,
        2e-2,
    )?;

    Ok(())
}

fn stop_logits_by_step(steps: &[StopTraceStep]) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(steps.len() * 2);
    for step in steps {
        let logits = step.stop_logits.to_dtype(DType::F32)?.reshape((2,))?;
        flat.extend(logits.to_vec1::<f32>()?);
    }
    Tensor::from_vec(flat, (steps.len(), 2), &Device::Cpu).map_err(Into::into)
}

fn stats_by_step(stats: impl IntoIterator<Item = [f32; 4]>) -> Result<Tensor> {
    let mut flat = Vec::new();
    let mut step_count = 0usize;
    for stat in stats {
        flat.extend(stat);
        step_count += 1;
    }
    if flat.len() != step_count * 4 {
        bail!("expected four stats per step");
    }
    Tensor::from_vec(flat, (step_count, 4), &Device::Cpu).map_err(Into::into)
}
