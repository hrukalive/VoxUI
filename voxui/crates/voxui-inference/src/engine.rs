//! Native VoxCPM generation pipeline.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::silu;

use crate::audio_io::load_wav_mono_resampled;
use crate::audiovae::AudioVAE;
use crate::base_lm::{BaseLM, BaseLMConfig};
use crate::dit::DiT;
use crate::encoder::LocalEncoder;
use crate::fsq::FSQLayer;
use crate::lora::LoraAdapter;
use crate::manifest::{ModelConfig, ModelVariant};
use crate::model_loader::GgufModelLoader;
use crate::request::SynthesisRequest;
use crate::tokenizer::VoxTokenizer;
use crate::{LinearWeight, RuntimeTensor};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub patch_size: usize,
    pub latent_dim: usize,
    pub sample_rate: u32,
    pub encode_sample_rate: u32,
    pub audio_chunk_size: usize,
    pub decode_chunk_size: usize,
    pub architecture: String,
    pub variant: ModelVariant,
}

#[derive(Debug)]
pub struct FirstPatchDebug {
    pub first_patch: Tensor,
    pub stop_logits: Tensor,
}

#[derive(Debug)]
pub struct StopLoopDebug {
    pub stop_logits_sequence: Tensor,
    pub generated_patch_count: usize,
}

#[derive(Debug, Clone)]
pub struct SynthesisChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub patch_index: usize,
    pub max_patches: usize,
    pub generated_patch_count: usize,
    pub is_final: bool,
}

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

struct LinearProjection {
    weight: LinearWeight,
    bias: Option<RuntimeTensor>,
}

struct PreparedInputs {
    text_tokens: Vec<u32>,
    text_mask: Tensor,
    audio_feat: Tensor,
    audio_mask: Tensor,
    audio_mask_values: Vec<f32>,
    target_text_token_count: usize,
    continuation_context_len: usize,
}

struct GenerationState {
    lm_hidden: Tensor,
    residual_hidden: Tensor,
    prefix_feat_cond: Tensor,
    generated_patches: Vec<Tensor>,
    context_len: usize,
}

struct GenerationOutput {
    latent: Tensor,
    generated_patch_count: usize,
    context_len: usize,
}

pub struct VoxCPMEngine {
    manifest: ModelConfig,
    tokenizer: VoxTokenizer,
    base_lm: BaseLM,
    residual_lm: BaseLM,
    encoder: LocalEncoder,
    fsq: FSQLayer,
    dit: DiT,
    vae: AudioVAE,
    lm_to_dit_proj: LinearProjection,
    res_to_dit_proj: LinearProjection,
    enc_to_lm_proj: LinearProjection,
    fusion_concat_proj: Option<LinearProjection>,
    stop_proj: LinearProjection,
    stop_head: LinearProjection,
    lora: Option<LoraAdapter>,
    device: Device,
    config: EngineConfig,
}

impl VoxCPMEngine {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        Self::load_with_progress(model_dir, device, |_, _| {}, None)
    }

    pub fn load_with_progress<F: Fn(usize, usize)>(
        model_dir: &Path,
        device: Device,
        on_progress: F,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        let started_at = Instant::now();
        tracing::info!(
            "VoxCPMEngine::load start model_dir={} device={device:?}",
            model_dir.display()
        );

        let base_loader = GgufModelLoader::from_model_dir(model_dir, device.clone())?;
        let variant = read_variant_from_loader(&base_loader)?;
        let manifest = ModelConfig::load(model_dir, variant)?;
        validate_base_metadata(&base_loader, &manifest)?;
        let tokenizer = VoxTokenizer::from_dir(model_dir)
            .with_context(|| format!("load tokenizer from {}", model_dir.display()))?;
        tracing::info!(
            "VoxCPMEngine::load manifest/tokenizer ready architecture={} sample_rate={} patch_size={} elapsed_seconds={:.3}",
            manifest.architecture,
            manifest.output_sample_rate(),
            manifest.patch_size,
            started_at.elapsed().as_secs_f64()
        );

        let check_cancel = |step: usize| -> Result<()> {
            on_progress(step, 6);
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                bail!("model loading cancelled");
            }
            Ok(())
        };

        check_cancel(0)?;
        tracing::info!("VoxCPMEngine::load component start name=base_lm");
        let base_lm_config = BaseLMConfig::from_model_config(&manifest, "base_lm")?;
        let base_lm = BaseLM::load(&base_loader, base_lm_config, &device)?;
        tracing::info!(
            "VoxCPMEngine::load component done name=base_lm elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        check_cancel(1)?;
        tracing::info!("VoxCPMEngine::load component start name=residual_lm");
        let residual_lm_config = BaseLMConfig::from_model_config(&manifest, "residual_lm")?;
        let residual_lm = BaseLM::load(&base_loader, residual_lm_config, &device)?;
        tracing::info!(
            "VoxCPMEngine::load component done name=residual_lm elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        check_cancel(2)?;
        tracing::info!("VoxCPMEngine::load component start name=feat_encoder");
        let encoder = LocalEncoder::load_from_config(&base_loader, &manifest)?;
        tracing::info!(
            "VoxCPMEngine::load component done name=feat_encoder elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        check_cancel(3)?;
        tracing::info!("VoxCPMEngine::load component start name=feat_decoder");
        let dit = DiT::load_from_config(&base_loader, &manifest)?;
        tracing::info!(
            "VoxCPMEngine::load component done name=feat_decoder elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        check_cancel(4)?;
        tracing::info!("VoxCPMEngine::load component start name=audio_vae");
        let vae = AudioVAE::load_from_config(&base_loader, &manifest.audio_vae)?;
        tracing::info!(
            "VoxCPMEngine::load component done name=audio_vae elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        check_cancel(5)?;
        tracing::info!("VoxCPMEngine::load component start name=projections");
        let fsq = FSQLayer::load(
            &base_loader,
            manifest.scalar_quantization_latent_dim,
            manifest.scalar_quantization_scale as f64,
        )?;
        let lm_to_dit_proj = load_projection(&base_loader, "lm_to_dit_proj")?;
        let res_to_dit_proj = load_projection(&base_loader, "res_to_dit_proj")?;
        let enc_to_lm_proj = load_projection(&base_loader, "enc_to_lm_proj")?;
        let fusion_concat_proj = if base_loader.has_tensor("fusion_concat_proj.weight") {
            Some(load_projection(&base_loader, "fusion_concat_proj")?)
        } else {
            None
        };
        let stop_proj = load_projection(&base_loader, "stop_proj")?;
        let stop_head = load_projection(&base_loader, "stop_head")?;
        tracing::info!(
            "VoxCPMEngine::load component done name=projections elapsed_seconds={:.3}",
            started_at.elapsed().as_secs_f64()
        );

        let audio_chunk_size = product_or_manifest(
            &manifest.audio_vae.encoder_rates,
            manifest.audio_vae.chunk_size,
        );
        let decode_chunk_size = product_or_manifest(
            &manifest.audio_vae.decoder_rates,
            manifest.audio_vae.decode_chunk_size,
        );
        let config = EngineConfig {
            patch_size: manifest.patch_size,
            latent_dim: manifest.feat_dim,
            sample_rate: manifest.output_sample_rate(),
            encode_sample_rate: manifest.audio_vae.sample_rate,
            audio_chunk_size,
            decode_chunk_size,
            architecture: manifest.architecture.clone(),
            variant: manifest.variant,
        };
        tracing::info!(
            "VoxCPMEngine::load complete architecture={} sample_rate={} patch_size={} elapsed_seconds={:.3}",
            config.architecture,
            config.sample_rate,
            config.patch_size,
            started_at.elapsed().as_secs_f64()
        );

        Ok(Self {
            manifest,
            tokenizer,
            base_lm,
            residual_lm,
            encoder,
            fsq,
            dit,
            vae,
            lm_to_dit_proj,
            res_to_dit_proj,
            enc_to_lm_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            lora: None,
            device,
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    pub fn architecture(&self) -> &str {
        &self.config.architecture
    }

    pub fn load_lora(&mut self, path: &Path) -> Result<()> {
        self.lora = Some(LoraAdapter::load_file_for_model(
            path,
            &self.device,
            &self.manifest,
        )?);
        Ok(())
    }

    pub fn unload_lora(&mut self) {
        self.lora = None;
    }

    pub fn generate_debug_first_patch(
        &mut self,
        request: SynthesisRequest,
    ) -> Result<FirstPatchDebug> {
        self.generate_debug_first_patch_inner(request, None)
    }

    pub fn generate_debug_first_patch_with_noise(
        &mut self,
        request: SynthesisRequest,
        noise: Tensor,
    ) -> Result<FirstPatchDebug> {
        self.generate_debug_first_patch_inner(request, Some(noise))
    }

    pub fn generate_debug_stop_loop_with_noises(
        &mut self,
        request: SynthesisRequest,
        noises: Tensor,
    ) -> Result<StopLoopDebug> {
        let request = request.validated(self.config.variant)?;
        let prepared = self.build_inputs(&request)?;
        let max_len = bounded_max_len(&request, prepared.target_text_token_count);
        let mut state = self.prefill(&prepared)?;
        let noise_count = noises.dim(0)?;
        let mut stop_logits = Vec::new();
        let mut generated_patch_count = 0usize;

        for step in 0..max_len {
            if step >= noise_count {
                bail!(
                    "fixed DIT noise trace ended after {noise_count} steps before Rust stop fired"
                );
            }
            let noise = noises.narrow(0, step, 1)?;
            let (_latent_patch, logits, _pred_feat) =
                self.generate_one_patch(&mut state, &request, Some(&noise))?;
            generated_patch_count += 1;
            let stop_flag = stop_flag_from_logits(&logits)?;
            stop_logits.push(logits);
            if step > request.min_len && stop_flag {
                break;
            }
        }

        if stop_logits.is_empty() {
            bail!("VoxCPM generated no stop logits");
        }
        let stop_refs = stop_logits.iter().collect::<Vec<_>>();
        Ok(StopLoopDebug {
            stop_logits_sequence: Tensor::cat(&stop_refs, 0)?,
            generated_patch_count,
        })
    }

    pub fn generate_debug_stop_loop_with_patches(
        &mut self,
        request: SynthesisRequest,
        generated_patches: Tensor,
    ) -> Result<StopLoopDebug> {
        let request = request.validated(self.config.variant)?;
        let prepared = self.build_inputs(&request)?;
        let max_len = bounded_max_len(&request, prepared.target_text_token_count);
        let mut state = self.prefill(&prepared)?;
        let patch_count = generated_patches.dim(0)?;
        let mut stop_logits = Vec::new();
        let mut generated_patch_count = 0usize;

        for step in 0..max_len {
            if step >= patch_count {
                bail!(
                    "generated patch trace ended after {patch_count} steps before Rust stop fired"
                );
            }

            let pred_feat = generated_patches.narrow(0, step, 1)?;
            let curr_embed = self.encoder.encode_patches(&pred_feat.unsqueeze(1)?)?;
            let curr_embed =
                self.apply_projection(&curr_embed, &self.enc_to_lm_proj, "enc_to_lm_proj")?;
            let logits = self.stop_logits(&state.lm_hidden)?;

            generated_patch_count += 1;
            let stop_flag = stop_flag_from_logits(&logits)?;
            stop_logits.push(logits);
            if step > request.min_len && stop_flag {
                break;
            }

            let lm_out = self
                .base_lm
                .forward_embed_with_lora(&curr_embed, self.lora.as_ref())?;
            let lm_hidden = lm_out.squeeze(1)?;
            state.lm_hidden = self.fsq.forward(&lm_hidden)?;

            let curr_embed_step = curr_embed.squeeze(1)?;
            let residual_input = if self.config.variant == ModelVariant::VoxCpm2 {
                let fusion = self
                    .fusion_concat_proj
                    .as_ref()
                    .context("VoxCPM2 requires fusion_concat_proj")?;
                self.apply_projection(
                    &Tensor::cat(&[&state.lm_hidden, &curr_embed_step], 1)?,
                    fusion,
                    "fusion_concat_proj",
                )?
            } else {
                (state.lm_hidden.clone() + curr_embed_step)?
            };
            let residual_out = self
                .residual_lm
                .forward_embed_with_lora(&residual_input.unsqueeze(1)?, self.lora.as_ref())?;
            state.residual_hidden = residual_out.squeeze(1)?;
        }

        if stop_logits.is_empty() {
            bail!("VoxCPM generated no stop logits");
        }
        let stop_refs = stop_logits.iter().collect::<Vec<_>>();
        Ok(StopLoopDebug {
            stop_logits_sequence: Tensor::cat(&stop_refs, 0)?,
            generated_patch_count,
        })
    }

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

        let generated_patches = state
            .generated_patches
            .get(state.context_len..)
            .context("stop trace context length exceeds generated patch count")?;
        Ok(StopTraceDebug {
            steps,
            generated_audio_feat: generated_patches_to_audio_feat(
                generated_patches,
                self.config.latent_dim,
                self.config.patch_size,
            )?,
            generated_step_count: generated_patches.len(),
        })
    }

    pub fn generate<F: Fn(usize, usize)>(
        &mut self,
        request: SynthesisRequest,
        progress: F,
    ) -> Result<Vec<f32>> {
        self.generate_cancellable(request, progress, None)
    }

    pub fn generate_cancellable<F: Fn(usize, usize)>(
        &mut self,
        request: SynthesisRequest,
        progress: F,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let request = request.validated(self.config.variant)?;
        let prepared = self.build_inputs(&request)?;
        let max_len = bounded_max_len(&request, prepared.target_text_token_count);
        tracing::info!(
            "VoxCPMEngine::generate text_token_count={} bounded_max_len={} request_max_len={} ratio_threshold={}",
            prepared.target_text_token_count,
            max_len,
            request.max_len,
            request.retry_badcase_ratio_threshold
        );
        let attempts = if request.retry_badcase {
            request.retry_badcase_max_times.max(1)
        } else {
            1
        };

        let mut last_output = None;
        for _ in 0..attempts {
            let output =
                self.run_generation_once(&prepared, &request, max_len, &progress, cancel)?;
            let is_badcase = request.retry_badcase
                && output.generated_patch_count as f32
                    >= prepared.target_text_token_count as f32
                        * request.retry_badcase_ratio_threshold;
            last_output = Some(output);
            if !is_badcase {
                break;
            }
        }

        let output = last_output.context("generation produced no output")?;
        self.decode_generation(output)
    }

    pub fn generate_streaming<F>(&mut self, request: SynthesisRequest, on_chunk: F) -> Result<()>
    where
        F: FnMut(SynthesisChunk) -> Result<()>,
    {
        self.generate_streaming_cancellable(request, on_chunk, None)
    }

    pub fn generate_streaming_cancellable<F>(
        &mut self,
        request: SynthesisRequest,
        mut on_chunk: F,
        cancel: Option<&AtomicBool>,
    ) -> Result<()>
    where
        F: FnMut(SynthesisChunk) -> Result<()>,
    {
        let request = validate_streaming_request(request, self.config.variant)?;

        let prepared = self.build_inputs(&request)?;
        let max_len = bounded_max_len(&request, prepared.target_text_token_count);
        tracing::info!(
            "VoxCPMEngine::generate_streaming text_token_count={} bounded_max_len={} request_max_len={}",
            prepared.target_text_token_count,
            max_len,
            request.max_len
        );

        let progress = |_, _| {};
        let mut decoder = self.vae.streaming_decoder();
        let mut emit_chunk = |engine: &VoxCPMEngine,
                              state: &GenerationState,
                              step: usize,
                              max_patches: usize,
                              is_final: bool|
         -> Result<()> {
            let latest_patch = state
                .generated_patches
                .last()
                .context("streaming generation produced no patch")?;
            let latent_chunk = latest_patch.transpose(1, 2)?.contiguous()?;
            let audio = decoder.decode_chunk(&latent_chunk.to_dtype(DType::F32)?)?;
            on_chunk(SynthesisChunk {
                samples: audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?,
                sample_rate: engine.config.sample_rate,
                patch_index: step,
                max_patches,
                generated_patch_count: step + 1,
                is_final,
            })
        };
        self.run_generation_once_with_patches(
            &prepared,
            &request,
            max_len,
            &progress,
            cancel,
            &mut emit_chunk,
        )?;
        Ok(())
    }

    /// Compatibility wrapper for older callers. New code should use `generate`.
    pub fn synthesize<F: Fn(usize, usize)>(
        &mut self,
        text: &str,
        dit_steps: usize,
        progress: F,
    ) -> Result<Vec<f32>> {
        self.generate(
            SynthesisRequest {
                text: text.to_string(),
                inference_timesteps: dit_steps,
                ..SynthesisRequest::default()
            },
            progress,
        )
    }

    fn generate_debug_first_patch_inner(
        &mut self,
        request: SynthesisRequest,
        noise: Option<Tensor>,
    ) -> Result<FirstPatchDebug> {
        let request = request.validated(self.config.variant)?;
        let prepared = self.build_inputs(&request)?;
        let mut state = self.prefill(&prepared)?;
        let (first_patch, stop_logits, _pred_feat) =
            self.generate_one_patch(&mut state, &request, noise.as_ref())?;
        Ok(FirstPatchDebug {
            first_patch,
            stop_logits,
        })
    }

    fn build_inputs(&mut self, request: &SynthesisRequest) -> Result<PreparedInputs> {
        match (
            request.reference_wav_path.as_ref(),
            request.prompt_wav_path.as_ref(),
        ) {
            (Some(_), Some(_)) => self.build_reference_prompt_inputs(request),
            (Some(_), None) => self.build_reference_inputs(request),
            (None, Some(_)) => self.build_prompt_inputs(request),
            (None, None) => self.build_zero_shot_inputs(request),
        }
    }

    fn build_zero_shot_inputs(&self, request: &SynthesisRequest) -> Result<PreparedInputs> {
        let mut text_tokens = self.tokenizer.encode(&request.text)?;
        let target_text_token_count = text_tokens.len();
        text_tokens.push(self.manifest.special_tokens.audio_start);
        let text_len = text_tokens.len();
        let audio_feat = self.zero_feat(text_len)?.unsqueeze(0)?;
        let text_mask_values = vec![1.0; text_len];
        let audio_mask_values = vec![0.0; text_len];
        self.prepared(
            text_tokens,
            text_mask_values,
            audio_feat,
            audio_mask_values,
            target_text_token_count,
        )
    }

    fn build_prompt_inputs(&mut self, request: &SynthesisRequest) -> Result<PreparedInputs> {
        let prompt_text = request
            .prompt_text
            .as_ref()
            .context("prompt_text is required when prompt_wav_path is present")?;
        let mut text_tokens = self
            .tokenizer
            .encode(&format!("{prompt_text}{}", request.text))?;
        let target_text_token_count = self.tokenizer.encode(&request.text)?.len();
        text_tokens.push(self.manifest.special_tokens.audio_start);
        let text_len = text_tokens.len();

        let prompt_feat =
            self.encode_wav_patches(request.prompt_wav_path.as_ref().unwrap(), PaddingMode::Left)?;
        let prompt_len = prompt_feat.dim(0)?;
        text_tokens.extend(std::iter::repeat(0).take(prompt_len));
        let audio_feat =
            Tensor::cat(&[&self.zero_feat(text_len)?, &prompt_feat], 0)?.unsqueeze(0)?;
        let mut text_mask_values = vec![1.0; text_len];
        text_mask_values.extend(std::iter::repeat(0.0).take(prompt_len));
        let mut audio_mask_values = vec![0.0; text_len];
        audio_mask_values.extend(std::iter::repeat(1.0).take(prompt_len));

        self.prepared(
            text_tokens,
            text_mask_values,
            audio_feat,
            audio_mask_values,
            target_text_token_count,
        )
    }

    fn build_reference_inputs(&mut self, request: &SynthesisRequest) -> Result<PreparedInputs> {
        let mut text_tokens = self.tokenizer.encode(&request.text)?;
        let target_text_token_count = text_tokens.len();
        text_tokens.push(self.manifest.special_tokens.audio_start);
        let text_len = text_tokens.len();

        let ref_feat = self.encode_wav_patches(
            request.reference_wav_path.as_ref().unwrap(),
            PaddingMode::Right,
        )?;
        let (ref_tokens, ref_feats, ref_text_mask, ref_audio_mask) =
            self.make_ref_prefix(&ref_feat)?;

        let mut all_tokens = ref_tokens;
        all_tokens.extend(text_tokens);
        let audio_feat = Tensor::cat(&[&ref_feats, &self.zero_feat(text_len)?], 0)?.unsqueeze(0)?;
        let mut text_mask_values = ref_text_mask;
        text_mask_values.extend(std::iter::repeat(1.0).take(text_len));
        let mut audio_mask_values = ref_audio_mask;
        audio_mask_values.extend(std::iter::repeat(0.0).take(text_len));

        self.prepared(
            all_tokens,
            text_mask_values,
            audio_feat,
            audio_mask_values,
            target_text_token_count,
        )
    }

    fn build_reference_prompt_inputs(
        &mut self,
        request: &SynthesisRequest,
    ) -> Result<PreparedInputs> {
        let prompt_text = request
            .prompt_text
            .as_ref()
            .context("prompt_text is required when prompt_wav_path is present")?;
        let mut text_tokens = self
            .tokenizer
            .encode(&format!("{prompt_text}{}", request.text))?;
        let target_text_token_count = self.tokenizer.encode(&request.text)?.len();
        text_tokens.push(self.manifest.special_tokens.audio_start);
        let text_len = text_tokens.len();

        let ref_feat = self.encode_wav_patches(
            request.reference_wav_path.as_ref().unwrap(),
            PaddingMode::Right,
        )?;
        let prompt_feat =
            self.encode_wav_patches(request.prompt_wav_path.as_ref().unwrap(), PaddingMode::Left)?;
        let prompt_len = prompt_feat.dim(0)?;
        let (ref_tokens, ref_feats, ref_text_mask, ref_audio_mask) =
            self.make_ref_prefix(&ref_feat)?;

        let mut all_tokens = ref_tokens;
        all_tokens.extend(text_tokens);
        all_tokens.extend(std::iter::repeat(0).take(prompt_len));
        let text_pad_feat = self.zero_feat(text_len)?;
        let audio_feat =
            Tensor::cat(&[&ref_feats, &text_pad_feat, &prompt_feat], 0)?.unsqueeze(0)?;

        let mut text_mask_values = ref_text_mask;
        text_mask_values.extend(std::iter::repeat(1.0).take(text_len));
        text_mask_values.extend(std::iter::repeat(0.0).take(prompt_len));
        let mut audio_mask_values = ref_audio_mask;
        audio_mask_values.extend(std::iter::repeat(0.0).take(text_len));
        audio_mask_values.extend(std::iter::repeat(1.0).take(prompt_len));

        self.prepared(
            all_tokens,
            text_mask_values,
            audio_feat,
            audio_mask_values,
            target_text_token_count,
        )
    }

    fn prepared(
        &self,
        text_tokens: Vec<u32>,
        text_mask_values: Vec<f32>,
        audio_feat: Tensor,
        audio_mask_values: Vec<f32>,
        target_text_token_count: usize,
    ) -> Result<PreparedInputs> {
        let seq_len = text_tokens.len();
        if text_mask_values.len() != seq_len || audio_mask_values.len() != seq_len {
            bail!("prepared input mask length does not match token length");
        }
        if audio_feat.dims() != [1, seq_len, self.config.patch_size, self.config.latent_dim] {
            bail!(
                "prepared audio feature shape {:?} does not match token length {seq_len}",
                audio_feat.dims()
            );
        }
        let text_mask = Tensor::from_vec(text_mask_values, (1, seq_len), &self.device)?;
        let audio_mask = Tensor::from_vec(audio_mask_values.clone(), (1, seq_len), &self.device)?;
        let continuation_context_len = self.continuation_context_len(&audio_mask_values);
        Ok(PreparedInputs {
            text_tokens,
            text_mask,
            audio_feat,
            audio_mask,
            audio_mask_values,
            target_text_token_count,
            continuation_context_len,
        })
    }

    fn prefill(&mut self, prepared: &PreparedInputs) -> Result<GenerationState> {
        self.base_lm.reset_cache();
        self.residual_lm.reset_cache();

        let seq_len = prepared.text_tokens.len();
        let feat_embed = self.encoder.encode_patches(&prepared.audio_feat)?;
        let feat_embed =
            self.apply_projection(&feat_embed, &self.enc_to_lm_proj, "enc_to_lm_proj")?;
        let text_embed = self.base_lm.embed(&prepared.text_tokens)?;

        let text_mask = prepared
            .text_mask
            .unsqueeze(2)?
            .to_dtype(text_embed.dtype())?;
        let audio_mask = prepared
            .audio_mask
            .unsqueeze(2)?
            .to_dtype(feat_embed.dtype())?;
        let combined_embed =
            (text_embed.broadcast_mul(&text_mask)? + feat_embed.broadcast_mul(&audio_mask)?)?;

        let enc_outputs = self
            .base_lm
            .forward_embed_with_lora(&combined_embed, self.lora.as_ref())?;
        let fsq_outputs = self.fsq.forward(&enc_outputs)?;
        let enc_mask = prepared
            .text_mask
            .unsqueeze(2)?
            .to_dtype(enc_outputs.dtype())?;
        let fsq_mask = prepared
            .audio_mask
            .unsqueeze(2)?
            .to_dtype(fsq_outputs.dtype())?;
        let enc_outputs =
            (fsq_outputs.broadcast_mul(&fsq_mask)? + enc_outputs.broadcast_mul(&enc_mask)?)?;
        let lm_hidden = enc_outputs.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        let residual_input = if self.config.variant == ModelVariant::VoxCpm2 {
            let fusion = self
                .fusion_concat_proj
                .as_ref()
                .context("VoxCPM2 requires fusion_concat_proj")?;
            let masked_feat = feat_embed.broadcast_mul(&audio_mask)?;
            self.apply_projection(
                &Tensor::cat(&[&enc_outputs, &masked_feat], 2)?,
                fusion,
                "fusion_concat_proj",
            )?
        } else {
            (enc_outputs + feat_embed.broadcast_mul(&audio_mask)?)?
        };
        let residual_outputs = self
            .residual_lm
            .forward_embed_with_lora(&residual_input, self.lora.as_ref())?;
        let residual_hidden = residual_outputs.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        let prefix_feat_cond = prepared
            .audio_feat
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .transpose(1, 2)?
            .contiguous()?;
        let generated_patches = self.initial_context_patches(prepared)?;

        Ok(GenerationState {
            lm_hidden,
            residual_hidden,
            prefix_feat_cond,
            generated_patches,
            context_len: prepared.continuation_context_len,
        })
    }

    fn run_generation_once<F: Fn(usize, usize)>(
        &mut self,
        prepared: &PreparedInputs,
        request: &SynthesisRequest,
        max_len: usize,
        progress: &F,
        cancel: Option<&AtomicBool>,
    ) -> Result<GenerationOutput> {
        let mut noop = |_: &VoxCPMEngine, _: &GenerationState, _: usize, _: usize, _: bool| Ok(());
        self.run_generation_once_with_patches(
            prepared, request, max_len, progress, cancel, &mut noop,
        )
    }

    fn run_generation_once_with_patches<F: Fn(usize, usize)>(
        &mut self,
        prepared: &PreparedInputs,
        request: &SynthesisRequest,
        max_len: usize,
        progress: &F,
        cancel: Option<&AtomicBool>,
        on_patch: &mut dyn FnMut(&VoxCPMEngine, &GenerationState, usize, usize, bool) -> Result<()>,
    ) -> Result<GenerationOutput> {
        let mut state = self.prefill(prepared)?;
        let mut generated_patch_count = 0usize;

        for step in 0..max_len {
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                bail!("synthesis cancelled");
            }
            progress(step, max_len);
            let (_latent_patch, stop_logits, _pred_feat) =
                self.generate_one_patch(&mut state, request, None)?;
            generated_patch_count += 1;

            let stop_flag = stop_flag_from_logits(&stop_logits)?;
            let is_final = (step > request.min_len && stop_flag) || step + 1 == max_len;
            on_patch(self, &state, step, max_len, is_final)?;
            if is_final {
                break;
            }
        }
        progress(generated_patch_count, max_len);
        tracing::info!(
            "VoxCPMEngine::run_generation_once generated_patch_count={} max_len={}",
            generated_patch_count,
            max_len
        );

        if generated_patch_count == 0 {
            bail!("VoxCPM generated no latent patches");
        }
        let latent = patches_to_latent(
            &state.generated_patches,
            self.config.latent_dim,
            self.config.patch_size,
        )?;
        Ok(GenerationOutput {
            latent,
            generated_patch_count,
            context_len: state.context_len,
        })
    }

    fn generate_one_patch(
        &mut self,
        state: &mut GenerationState,
        request: &SynthesisRequest,
        fixed_noise: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dit_hidden_1 =
            self.apply_projection(&state.lm_hidden, &self.lm_to_dit_proj, "lm_to_dit_proj")?;
        let dit_hidden_2 = self.apply_projection(
            &state.residual_hidden,
            &self.res_to_dit_proj,
            "res_to_dit_proj",
        )?;
        let dit_hidden = if self.config.variant == ModelVariant::VoxCpm2 {
            Tensor::cat(&[&dit_hidden_1, &dit_hidden_2], 1)?
        } else {
            (dit_hidden_1 + dit_hidden_2)?
        };

        let noise = match fixed_noise {
            Some(noise) => noise.to_device(&self.device)?,
            None => Tensor::randn(
                0f32,
                1f32,
                (1, self.config.latent_dim, self.config.patch_size),
                &self.device,
            )?,
        };
        let latent_patch = self.dit.solve_euler_with_noise_lora(
            &dit_hidden,
            &state.prefix_feat_cond,
            &noise,
            request.inference_timesteps,
            request.cfg_value,
            self.lora.as_ref(),
        )?;
        let pred_feat = latent_patch.transpose(1, 2)?.contiguous()?;
        let pred_feat_for_encoder = pred_feat.unsqueeze(1)?;
        let curr_embed = self.encoder.encode_patches(&pred_feat_for_encoder)?;
        let curr_embed =
            self.apply_projection(&curr_embed, &self.enc_to_lm_proj, "enc_to_lm_proj")?;

        state.generated_patches.push(pred_feat.clone());
        state.prefix_feat_cond = latent_patch.clone();

        let stop_logits = self.stop_logits(&state.lm_hidden)?;

        let lm_out = self
            .base_lm
            .forward_embed_with_lora(&curr_embed, self.lora.as_ref())?;
        let lm_hidden = lm_out.squeeze(1)?;
        state.lm_hidden = self.fsq.forward(&lm_hidden)?;

        let curr_embed_step = curr_embed.squeeze(1)?;
        let residual_input = if self.config.variant == ModelVariant::VoxCpm2 {
            let fusion = self
                .fusion_concat_proj
                .as_ref()
                .context("VoxCPM2 requires fusion_concat_proj")?;
            self.apply_projection(
                &Tensor::cat(&[&state.lm_hidden, &curr_embed_step], 1)?,
                fusion,
                "fusion_concat_proj",
            )?
        } else {
            (state.lm_hidden.clone() + curr_embed_step)?
        };
        let residual_out = self
            .residual_lm
            .forward_embed_with_lora(&residual_input.unsqueeze(1)?, self.lora.as_ref())?;
        state.residual_hidden = residual_out.squeeze(1)?;

        Ok((latent_patch, stop_logits, pred_feat))
    }

    fn stop_logits(&self, lm_hidden: &Tensor) -> Result<Tensor> {
        let stop_in = silu(&linear_projection(lm_hidden, &self.stop_proj)?)?;
        linear_projection(&stop_in, &self.stop_head)
    }

    fn apply_projection(
        &self,
        x: &Tensor,
        projection: &LinearProjection,
        name: &str,
    ) -> Result<Tensor> {
        let base = linear_projection(x, projection)?;
        if let Some(lora) = self.lora.as_ref() {
            lora.apply(name, &base, x)
        } else {
            Ok(base)
        }
    }

    fn decode_generation(&self, output: GenerationOutput) -> Result<Vec<f32>> {
        let audio = self.vae.decode(&output.latent.to_dtype(DType::F32)?)?;
        let mut samples = audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        if output.context_len > 0 {
            let trim_len =
                output.context_len * self.config.patch_size * self.config.decode_chunk_size;
            if trim_len < samples.len() {
                samples.drain(0..trim_len);
            } else {
                samples.clear();
            }
        }
        Ok(samples)
    }

    fn zero_feat(&self, len: usize) -> Result<Tensor> {
        Tensor::zeros(
            (len, self.config.patch_size, self.config.latent_dim),
            DType::F32,
            &self.device,
        )
        .map_err(Into::into)
    }

    fn make_ref_prefix(&self, ref_feat: &Tensor) -> Result<(Vec<u32>, Tensor, Vec<f32>, Vec<f32>)> {
        let ref_len = ref_feat.dim(0)?;
        let ref_audio_start = self
            .manifest
            .special_tokens
            .ref_audio_start
            .context("manifest missing ref_audio_start")?;
        let ref_audio_end = self
            .manifest
            .special_tokens
            .ref_audio_end
            .context("manifest missing ref_audio_end")?;

        let mut tokens = Vec::with_capacity(ref_len + 2);
        tokens.push(ref_audio_start);
        tokens.extend(std::iter::repeat(0).take(ref_len));
        tokens.push(ref_audio_end);

        let z1 = self.zero_feat(1)?;
        let feats = Tensor::cat(&[&z1, ref_feat, &z1], 0)?;
        let mut text_mask = Vec::with_capacity(ref_len + 2);
        text_mask.push(1.0);
        text_mask.extend(std::iter::repeat(0.0).take(ref_len));
        text_mask.push(1.0);
        let mut audio_mask = Vec::with_capacity(ref_len + 2);
        audio_mask.push(0.0);
        audio_mask.extend(std::iter::repeat(1.0).take(ref_len));
        audio_mask.push(0.0);
        Ok((tokens, feats, text_mask, audio_mask))
    }

    fn encode_wav_patches(&self, path: &Path, padding: PaddingMode) -> Result<Tensor> {
        let loaded = load_wav_mono_resampled(path, self.config.encode_sample_rate)?;
        let mut samples = loaded.samples;
        let patch_len = self.config.patch_size * self.config.audio_chunk_size;
        if patch_len == 0 {
            bail!("invalid audio patch length");
        }
        let remainder = samples.len() % patch_len;
        if remainder != 0 {
            let padding_size = patch_len - remainder;
            match padding {
                PaddingMode::Left => {
                    let mut padded = vec![0.0; padding_size];
                    padded.extend(samples);
                    samples = padded;
                }
                PaddingMode::Right => samples.extend(std::iter::repeat(0.0).take(padding_size)),
            }
        }

        let len = samples.len();
        let audio = Tensor::from_vec(samples, (1, len), &self.device)?;
        let latent = self.vae.encode(&audio)?;
        latent_to_patches(&latent, self.config.patch_size)
    }

    fn continuation_context_len(&self, audio_mask_values: &[f32]) -> usize {
        if audio_mask_values.last().copied().unwrap_or(0.0) == 0.0 {
            return 0;
        }
        let audio_count = audio_mask_values
            .iter()
            .filter(|value| **value > 0.5)
            .count();
        let streaming_prefix_len = streaming_prefix_len(self.config.variant);
        audio_count.min(streaming_prefix_len - 1)
    }

    fn initial_context_patches(&self, prepared: &PreparedInputs) -> Result<Vec<Tensor>> {
        let context_len = prepared.continuation_context_len;
        if context_len == 0 {
            return Ok(Vec::new());
        }
        let mut indices = Vec::new();
        for (idx, value) in prepared.audio_mask_values.iter().enumerate() {
            if *value > 0.5 {
                indices.push(idx);
            }
        }
        let start = indices.len().saturating_sub(context_len);
        let mut patches = Vec::with_capacity(context_len);
        for idx in &indices[start..] {
            patches.push(
                prepared
                    .audio_feat
                    .narrow(1, *idx, 1)?
                    .squeeze(1)?
                    .contiguous()?,
            );
        }
        Ok(patches)
    }
}

#[derive(Clone, Copy)]
enum PaddingMode {
    Left,
    Right,
}

fn load_projection(loader: &GgufModelLoader, name: &str) -> Result<LinearProjection> {
    let weight_name = format!("{name}.weight");
    let bias_name = format!("{name}.bias");
    Ok(LinearProjection {
        weight: loader
            .load_linear_weight(&weight_name)
            .with_context(|| format!("load projection tensor {weight_name}"))?,
        bias: if loader.has_tensor(&bias_name) {
            Some(loader.load_runtime_tensor(&bias_name)?)
        } else {
            None
        },
    })
}

fn linear_projection(x: &Tensor, projection: &LinearProjection) -> Result<Tensor> {
    let out = projection.weight.forward(x)?;
    if let Some(bias) = projection.bias.as_ref() {
        out.broadcast_add(&bias.to_dense_dtype(out.dtype())?)
            .map_err(Into::into)
    } else {
        Ok(out)
    }
}

fn product_or_manifest(values: &[usize], manifest_value: usize) -> usize {
    if values.is_empty() {
        manifest_value
    } else {
        values.iter().product()
    }
}

fn bounded_max_len(request: &SynthesisRequest, target_text_token_count: usize) -> usize {
    let ratio_limit =
        (target_text_token_count as f32 * request.retry_badcase_ratio_threshold + 10.0) as usize;
    request.max_len.min(ratio_limit)
}

fn validate_streaming_request(
    request: SynthesisRequest,
    variant: ModelVariant,
) -> Result<SynthesisRequest> {
    let request = request.validated(variant)?;
    if request.retry_badcase {
        bail!("retry_badcase=true is not supported for streaming synthesis");
    }
    Ok(request)
}

fn streaming_prefix_len(variant: ModelVariant) -> usize {
    if variant == ModelVariant::VoxCpm2 {
        4
    } else {
        3
    }
}

fn stop_flag_from_logits(stop_logits: &Tensor) -> Result<bool> {
    Ok(stop_logits
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .first()
        .and_then(|row| row.get(1).zip(row.first()))
        .map(|(stop, keep)| stop > keep)
        .unwrap_or(false))
}

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

fn read_variant_from_loader(loader: &GgufModelLoader) -> Result<ModelVariant> {
    let value = loader
        .metadata()
        .get("voxcpm.variant")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("model.gguf missing voxcpm.variant metadata"))?;
    match value {
        "0.5" => Ok(ModelVariant::VoxCpm05),
        "1.5" => Ok(ModelVariant::VoxCpm15),
        "2.0" => Ok(ModelVariant::VoxCpm2),
        other => anyhow::bail!("unsupported voxcpm.variant `{other}`"),
    }
}

fn validate_base_metadata(loader: &GgufModelLoader, model: &ModelConfig) -> Result<()> {
    let kind = loader
        .metadata()
        .get("voxcpm.kind")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if kind != "base" {
        anyhow::bail!("model.gguf voxcpm.kind must be `base`, got `{kind}`");
    }
    let architecture = loader
        .metadata()
        .get("voxcpm.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if architecture != model.architecture {
        anyhow::bail!(
            "model.gguf architecture `{}` does not match config `{}`",
            architecture,
            model.architecture
        );
    }
    Ok(())
}

fn latent_to_patches(latent: &Tensor, patch_size: usize) -> Result<Tensor> {
    let latent = latent.squeeze(0)?;
    let (latent_dim, latent_len) = latent.dims2()?;
    let patch_count = latent_len / patch_size;
    if patch_count == 0 {
        bail!("encoded audio is shorter than one latent patch");
    }
    let latent = latent.narrow(1, 0, patch_count * patch_size)?;
    latent
        .reshape((latent_dim, patch_count, patch_size))?
        .transpose(0, 1)?
        .transpose(1, 2)?
        .contiguous()
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ModelVariant;

    #[test]
    fn voxcpm05_manifest_rates_give_python_patch_audio_lengths() {
        let model = ModelConfig {
            schema_version: 2,
            architecture: "voxcpm".to_string(),
            variant: ModelVariant::VoxCpm05,
            special_tokens: crate::manifest::SpecialTokens {
                audio_start: 101,
                audio_end: 102,
                ref_audio_start: None,
                ref_audio_end: None,
            },
            patch_size: 2,
            feat_dim: 64,
            scalar_quantization_latent_dim: 256,
            scalar_quantization_scale: 9.0,
            audio_vae: crate::manifest::AudioVaeManifest {
                encoder_dim: Some(128),
                decoder_dim: Some(1536),
                sample_rate: 16_000,
                out_sample_rate: None,
                latent_dim: 64,
                chunk_size: 640,
                decode_chunk_size: 640,
                encoder_rates: vec![2, 5, 8, 8],
                decoder_rates: vec![8, 8, 5, 2],
            },
            lm_config: serde_json::json!({}),
            encoder_config: serde_json::json!({}),
            dit_config: serde_json::json!({}),
            residual_lm_num_layers: Some(6),
            residual_lm_no_rope: Some(false),
        };

        let audio_chunk_size =
            product_or_manifest(&model.audio_vae.encoder_rates, model.audio_vae.chunk_size);
        let decode_chunk_size = product_or_manifest(
            &model.audio_vae.decoder_rates,
            model.audio_vae.decode_chunk_size,
        );

        assert_eq!(audio_chunk_size, 640);
        assert_eq!(decode_chunk_size, 640);
        assert_eq!(model.patch_size * audio_chunk_size, 1280);
        assert_eq!(model.patch_size * decode_chunk_size, 1280);
    }

    #[test]
    fn streaming_prefix_len_matches_python_reference_defaults() {
        assert_eq!(streaming_prefix_len(ModelVariant::VoxCpm05), 3);
        assert_eq!(streaming_prefix_len(ModelVariant::VoxCpm15), 3);
        assert_eq!(streaming_prefix_len(ModelVariant::VoxCpm2), 4);
    }

    #[test]
    fn streaming_request_rejects_retry_badcase() {
        let err = validate_streaming_request(
            SynthesisRequest {
                text: "hello".to_string(),
                retry_badcase: true,
                ..SynthesisRequest::default()
            },
            ModelVariant::VoxCpm2,
        )
        .unwrap_err();

        assert!(err.to_string().contains("retry_badcase=true"));
    }
}

fn patches_to_latent(patches: &[Tensor], latent_dim: usize, patch_size: usize) -> Result<Tensor> {
    if patches.is_empty() {
        bail!("no latent patches to decode");
    }
    let refs = patches.iter().collect::<Vec<_>>();
    let seq = Tensor::cat(&refs, 0)?;
    let patch_count = seq.dim(0)?;
    seq.transpose(1, 2)?
        .transpose(0, 1)?
        .reshape((1, latent_dim, patch_count * patch_size))
        .map_err(Into::into)
}

fn generated_patches_to_audio_feat(
    patches: &[Tensor],
    latent_dim: usize,
    patch_size: usize,
) -> Result<Tensor> {
    if patches.is_empty() {
        bail!("no generated patches to trace");
    }
    for patch in patches {
        let dims = patch.dims3()?;
        if dims != (1, patch_size, latent_dim) {
            bail!(
                "generated patch shape must be [1, patch_size, latent_dim], got {:?}",
                dims
            );
        }
    }
    let refs = patches.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}
