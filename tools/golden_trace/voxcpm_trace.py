"""
Generate small deterministic reference traces from the local VoxCPM Python code.

This script is not used by Rust runtime inference. It writes trace files consumed
by Rust parity tests.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.golden_trace.trace_schema import TensorRecord, TraceWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def import_local_voxcpm(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "VoxCPM" / "src"))


def has_python_checkpoint(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and (
        (model_dir / "audiovae.safetensors").exists() or (model_dir / "audiovae.pth").exists()
    )


def resolve_python_model_dir(repo_root: Path, model_dir: Path, variant: str) -> Path:
    if has_python_checkpoint(model_dir):
        return model_dir

    python_model_names = {
        "0.5": "VoxCPM-0.5B",
        "1.5": "VoxCPM1.5",
        "2.0": "VoxCPM2",
    }
    fallback_name = python_model_names.get(variant)
    if fallback_name:
        fallback_dir = repo_root / "VoxCPM" / "models" / fallback_name
        if has_python_checkpoint(fallback_dir):
            return fallback_dir
    return model_dir


def to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to(torch.float32).cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


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


def force_runtime_dtype(pipeline: Any, runtime_dtype: str) -> None:
    if runtime_dtype == "config":
        return
    model = pipeline.tts_model
    if runtime_dtype == "float32":
        dtype = torch.float32
        config_value = "float32"
    elif runtime_dtype == "bfloat16":
        dtype = torch.bfloat16
        config_value = "bfloat16"
    else:
        raise ValueError(f"unsupported runtime dtype: {runtime_dtype}")
    model.config.dtype = config_value
    model.to(dtype)
    model.audio_vae = model.audio_vae.to(torch.float32)
    max_length = getattr(model.config, "max_length", 8192)
    if hasattr(model.base_lm, "setup_cache"):
        model.base_lm.setup_cache(1, max_length, model.device, dtype)
    if hasattr(model.residual_lm, "setup_cache"):
        model.residual_lm.setup_cache(1, max_length, model.device, dtype)


class TraceCapture:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.model = pipeline.tts_model
        self.values: dict[str, np.ndarray] = {}
        self.lists: dict[str, list[int]] = {}
        self._dit_noises: list[np.ndarray] = []
        self._stop_logits_sequence: list[np.ndarray] = []
        self._orig_inference = None
        self._orig_encode_wav = None
        self._orig_vae_encode = None
        self._orig_decode = None
        self._orig_randn = None
        self._handles = []

    def install(self) -> None:
        self._wrap_inference()
        self._wrap_encode_wav()
        self._wrap_vae_encode()
        self._wrap_decode()
        self._wrap_randn()
        self._install_module_hooks()

    def records(self, writer: TraceWriter) -> list[TensorRecord]:
        self.restore()
        if self._dit_noises:
            self.values["dit_noises"] = np.concatenate(self._dit_noises, axis=0)
        if self._stop_logits_sequence:
            self.values["stop_logits_sequence"] = np.concatenate(self._stop_logits_sequence, axis=0)
        for name, values in self.lists.items():
            writer.write_u32_list(name, values)
        return [writer.write_tensor(name, value) for name, value in self.values.items()]

    def restore(self) -> None:
        if self._orig_inference is not None:
            self.model._inference = self._orig_inference
            self._orig_inference = None
        if self._orig_encode_wav is not None:
            self.model._encode_wav = self._orig_encode_wav
            self._orig_encode_wav = None
        if self._orig_vae_encode is not None:
            self.model.audio_vae.encode = self._orig_vae_encode
            self._orig_vae_encode = None
        if self._orig_decode is not None:
            self.model.audio_vae.decode = self._orig_decode
            self._orig_decode = None
        if self._orig_randn is not None:
            torch.randn = self._orig_randn
            self._orig_randn = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _store_tensor_once(self, name: str, tensor: Any) -> None:
        if name not in self.values:
            self.values[name] = to_numpy(tensor)

    def _wrap_inference(self) -> None:
        self._orig_inference = self.model._inference

        def wrapped_inference(text, text_mask, feat, feat_mask, *args, **kwargs):
            self.lists.setdefault("token_ids", [int(v) for v in text.detach().cpu().reshape(-1).tolist()])
            self._store_tensor_once("text_mask", text_mask)
            self._store_tensor_once("audio_mask", feat_mask)
            self._store_tensor_once("prefill_audio_feat_b_t_p_d", feat)
            generator = self._orig_inference(text, text_mask, feat, feat_mask, *args, **kwargs)

            def traced_generator():
                for item in generator:
                    if isinstance(item, tuple) and item:
                        self._store_tensor_once("generated_latent", item[0])
                        if len(item) > 1:
                            self._store_tensor_once("generated_audio_feat", item[1])
                    yield item

            return traced_generator()

        self.model._inference = wrapped_inference

    def _wrap_encode_wav(self) -> None:
        if not hasattr(self.model, "_encode_wav"):
            return
        self._orig_encode_wav = self.model._encode_wav

        def wrapped_encode_wav(path, *args, **kwargs):
            result = self._orig_encode_wav(path, *args, **kwargs)
            padding_mode = kwargs.get("padding_mode", args[0] if args else "")
            if padding_mode == "right":
                self._store_tensor_once("audio_vae_encoded_reference", result)
            else:
                self._store_tensor_once("audio_vae_encoded_prompt", result)
            return result

        self.model._encode_wav = wrapped_encode_wav

    def _wrap_decode(self) -> None:
        self._orig_decode = self.model.audio_vae.decode

        def wrapped_decode(*args, **kwargs):
            result = self._orig_decode(*args, **kwargs)
            self._store_tensor_once("audio_vae_decoded_raw", result)
            return result

        self.model.audio_vae.decode = wrapped_decode

    def _wrap_vae_encode(self) -> None:
        self._orig_vae_encode = self.model.audio_vae.encode

        def wrapped_encode(audio_data, sample_rate, *args, **kwargs):
            self._store_tensor_once("audio_vae_encode_input", audio_data)
            result = self._orig_vae_encode(audio_data, sample_rate, *args, **kwargs)
            self._store_tensor_once("audio_vae_encode_output", result)
            return result

        self.model.audio_vae.encode = wrapped_encode

    def _wrap_randn(self) -> None:
        self._orig_randn = torch.randn

        def wrapped_randn(*args, **kwargs):
            result = self._orig_randn(*args, **kwargs)
            self._store_tensor_once("first_dit_noise", result)
            self._dit_noises.append(to_numpy(result))
            return result

        torch.randn = wrapped_randn

    def _install_module_hooks(self) -> None:
        self._handles.append(self.model.feat_encoder.register_forward_hook(self._feat_encoder_hook))
        self._handles.append(self.model.base_lm.register_forward_hook(self._base_lm_hook))
        self._handles.append(self.model.residual_lm.register_forward_hook(self._residual_lm_hook))
        self._handles.append(self.model.fsq_layer.register_forward_hook(self._fsq_hook))
        try:
            self._handles.append(
                self.model.feat_decoder.register_forward_hook(self._feat_decoder_hook, with_kwargs=True)
            )
        except TypeError:
            self._handles.append(self.model.feat_decoder.register_forward_hook(self._feat_decoder_hook_no_kwargs))
        self._handles.append(self.model.stop_head.register_forward_hook(self._stop_head_hook))

    def _feat_encoder_hook(self, _module, _inputs, output) -> None:
        self._store_tensor_once("local_encoder_output", output)

    def _base_lm_hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._store_tensor_once("base_lm_prefill_hidden", hidden)

    def _residual_lm_hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._store_tensor_once("residual_lm_prefill_hidden", hidden)

    def _fsq_hook(self, _module, _inputs, output) -> None:
        self._store_tensor_once("first_fsq_hidden", output)

    def _feat_decoder_hook(self, _module, inputs, kwargs, output) -> None:
        mu = kwargs.get("mu") if kwargs else None
        cond = kwargs.get("cond") if kwargs else None
        if mu is None and inputs:
            mu = inputs[0]
        if cond is not None:
            self._store_tensor_once("first_dit_cond", cond)
        if mu is not None:
            self._store_tensor_once("first_dit_mu", mu)
        self._store_tensor_once("first_dit_patch", output)

    def _feat_decoder_hook_no_kwargs(self, _module, inputs, output) -> None:
        if inputs:
            self._store_tensor_once("first_dit_mu", inputs[0])
        self._store_tensor_once("first_dit_patch", output)

    def _stop_head_hook(self, _module, _inputs, output) -> None:
        self._store_tensor_once("stop_logits", output)
        self._stop_logits_sequence.append(to_numpy(output))


@torch.inference_mode()
def run_v1_stop_trace(
    model: Any,
    *,
    text: str,
    min_len: int,
    max_len: int,
    inference_timesteps: int,
    cfg_value: float,
) -> dict[str, Any]:
    from einops import rearrange
    from voxcpm.model.utils import get_dtype

    prompt_audio_feat = torch.empty((0, model.patch_size, model.audio_vae.latent_dim), dtype=torch.float32)
    text_token = torch.LongTensor(model.text_tokenizer(text))
    text_token = torch.cat(
        [
            text_token,
            torch.tensor(
                [model.audio_start_token],
                dtype=torch.int32,
                device=text_token.device,
            ),
        ],
        dim=-1,
    )

    audio_length = prompt_audio_feat.size(0)
    text_length = text_token.shape[0]
    text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
    audio_pad_feat = torch.zeros(
        (text_token.shape[0], model.patch_size, model.audio_vae.latent_dim),
        dtype=torch.float32,
        device=text_token.device,
    )
    text_token = torch.cat([text_token, text_pad_token])
    audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
    text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
    audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)

    text_token = text_token.unsqueeze(0).to(model.device)
    text_mask = text_mask.unsqueeze(0).to(model.device)
    audio_feat = audio_feat.unsqueeze(0).to(model.device).to(get_dtype(model.config.dtype))
    audio_mask = audio_mask.unsqueeze(0).to(model.device)

    first_dit_noise: np.ndarray | None = None
    orig_randn = torch.randn

    def wrapped_randn(*args, **kwargs):
        nonlocal first_dit_noise
        result = orig_randn(*args, **kwargs)
        if first_dit_noise is None:
            first_dit_noise = to_numpy(result)
        return result

    torch.randn = wrapped_randn
    try:
        B, _, _, _ = audio_feat.shape

        prefill_encoder = getattr(model, "_feat_encoder_raw", model.feat_encoder)
        feat_embed = prefill_encoder(audio_feat)
        feat_embed = model.enc_to_lm_proj(feat_embed)

        if model.config.lm_config.use_mup:
            scale_emb = model.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = model.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = audio_feat[:, -1, ...]
        pred_feat_seq = []

        enc_outputs, kv_cache_tuple = model.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = model.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_enc_outputs, residual_kv_cache_tuple = model.residual_lm(
            inputs_embeds=enc_outputs + audio_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        model.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        stop_logits_by_step = []
        stop_decisions = []
        lm_hidden_stats_by_step = []
        residual_hidden_stats_by_step = []
        pred_feat_stats_by_step = []

        for i in range(max_len):
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

            stop_logits_by_step.append(to_numpy(stop_logits).reshape(2))
            stop_decisions.append(stop_flag)
            lm_hidden_stats_by_step.append(tensor_stats(lm_hidden))
            residual_hidden_stats_by_step.append(tensor_stats(residual_hidden))
            pred_feat_stats_by_step.append(tensor_stats(pred_feat))

            if i > min_len and stop_flag == 1:
                break

            lm_hidden = model.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([model.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

            lm_hidden = model.fsq_layer(lm_hidden)
            residual_hidden = model.residual_lm.forward_step(
                lm_hidden + curr_embed[:, 0, :],
                torch.tensor([model.residual_lm.kv_cache.step()], device=curr_embed.device),
            ).clone()

        pred_feat_seq_tensor = torch.cat(pred_feat_seq, dim=1)
        feat_pred = rearrange(pred_feat_seq_tensor, "b t p d -> b d (t p)", b=B, p=model.patch_size)
        decoded_wav = model.audio_vae.decode(feat_pred.to(torch.float32)).squeeze(1).cpu()
        generated_audio_feat = pred_feat_seq_tensor.squeeze(0).cpu()

        return {
            "token_ids": [int(v) for v in text_token.detach().cpu().reshape(-1).tolist()],
            "first_dit_noise": first_dit_noise if first_dit_noise is not None else np.zeros((0,), dtype=np.float32),
            "stop_logits_by_step": stack_or_empty(stop_logits_by_step, 2),
            "stop_decisions": stop_decisions,
            "lm_hidden_stats_by_step": stack_or_empty(lm_hidden_stats_by_step, 4),
            "residual_hidden_stats_by_step": stack_or_empty(residual_hidden_stats_by_step, 4),
            "pred_feat_stats_by_step": stack_or_empty(pred_feat_stats_by_step, 4),
            "generated_audio_feat": to_numpy(generated_audio_feat),
            "decoded_wav_head": to_numpy(decoded_wav.squeeze(0)[:4096]),
            "generated_step_count": len(stop_decisions),
        }
    finally:
        torch.randn = orig_randn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["0.5", "1.5", "2.0"], required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("goldens"))
    parser.add_argument("--text", default="Hello, welcome to the stream!")
    parser.add_argument("--prompt-wav-path", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--reference-wav-path", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime-dtype", choices=["float32", "bfloat16", "config"], default="float32")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--retry-badcase", action="store_true")
    parser.add_argument("--trace-kind", choices=["first_patch", "stop"], default="first_patch")
    parser.add_argument("--stop-max-len", type=int, default=120)
    parser.add_argument("--stop-min-len", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    import_local_voxcpm(args.repo_root)

    from voxcpm import VoxCPM

    if args.trace_kind == "stop":
        if args.variant not in {"0.5", "1.5"}:
            raise ValueError("stop traces only support VoxCPM variants 0.5 and 1.5")
        if args.stop_max_len <= 0:
            parser.error("--stop-max-len must be > 0 for stop traces")
        if args.stop_min_len < 0:
            parser.error("--stop-min-len must be >= 0 for stop traces")
        if args.stop_min_len >= args.stop_max_len:
            parser.error("--stop-min-len must be < --stop-max-len for stop traces")
        if args.prompt_wav_path or args.prompt_text or args.reference_wav_path:
            raise ValueError("stop traces currently support zero-shot generation only")

        model_dir = resolve_python_model_dir(args.repo_root, args.model_dir, args.variant)
        model = VoxCPM(
            voxcpm_model_path=str(model_dir),
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=False,
            device="cpu",
        )
        force_runtime_dtype(model, args.runtime_dtype)

        trace = run_v1_stop_trace(
            model.tts_model,
            text=args.text,
            min_len=args.stop_min_len,
            max_len=args.stop_max_len,
            inference_timesteps=4,
            cfg_value=2.0,
        )

        writer = TraceWriter(args.out_dir, args.case_name)
        writer.write_u32_list("token_ids", trace["token_ids"])
        writer.write_u32_list("stop_decisions", trace["stop_decisions"])
        tensors = [
            writer.write_tensor("first_dit_noise", trace["first_dit_noise"]),
            writer.write_tensor("stop_logits_by_step", trace["stop_logits_by_step"]),
            writer.write_tensor("lm_hidden_stats_by_step", trace["lm_hidden_stats_by_step"]),
            writer.write_tensor("residual_hidden_stats_by_step", trace["residual_hidden_stats_by_step"]),
            writer.write_tensor("pred_feat_stats_by_step", trace["pred_feat_stats_by_step"]),
            writer.write_tensor("generated_audio_feat", trace["generated_audio_feat"]),
            writer.write_tensor("decoded_wav_head", trace["decoded_wav_head"]),
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
                "source_model_dir": str(model_dir.resolve()),
                "runtime_dtype": args.runtime_dtype,
                "generated_step_count": trace["generated_step_count"],
                "trace_kind": "stop",
            },
        )
        return

    model = VoxCPM(
        voxcpm_model_path=str(args.model_dir),
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=False,
        device="cpu",
    )
    force_runtime_dtype(model, args.runtime_dtype)

    capture = TraceCapture(model)
    capture.install()

    wav = model.generate(
        text=args.text,
        prompt_wav_path=str(args.prompt_wav_path) if args.prompt_wav_path else None,
        prompt_text=args.prompt_text,
        reference_wav_path=str(args.reference_wav_path) if args.reference_wav_path else None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        min_len=args.min_len,
        max_len=args.max_len,
        normalize=False,
        denoise=False,
        retry_badcase=args.retry_badcase,
    )

    writer = TraceWriter(args.out_dir, args.case_name)
    tensors = capture.records(writer)
    tensors.append(writer.write_tensor("decoded_wav_head", np.asarray(wav[:4096], dtype=np.float32)))
    writer.write_manifest(
        variant=args.variant,
        architecture="voxcpm2" if args.variant == "2.0" else "voxcpm",
        request={
            "text": args.text,
            "prompt_wav_path": str(args.prompt_wav_path) if args.prompt_wav_path else None,
            "prompt_text": args.prompt_text,
            "reference_wav_path": str(args.reference_wav_path) if args.reference_wav_path else None,
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.inference_timesteps,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "normalize": False,
            "retry_badcase": args.retry_badcase,
        },
        tensors=tensors,
        metadata={
            "seed": args.seed,
            "source_model_dir": str(args.model_dir.resolve()),
            "runtime_dtype": args.runtime_dtype,
        },
    )


if __name__ == "__main__":
    main()
