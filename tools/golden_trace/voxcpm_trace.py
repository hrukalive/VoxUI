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


def to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to(torch.float32).cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


class TraceCapture:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.model = pipeline.tts_model
        self.values: dict[str, np.ndarray] = {}
        self.lists: dict[str, list[int]] = {}
        self._orig_inference = None
        self._orig_encode_wav = None
        self._orig_decode = None
        self._orig_randn = None
        self._handles = []

    def install(self) -> None:
        self._wrap_inference()
        self._wrap_encode_wav()
        self._wrap_decode()
        self._wrap_randn()
        self._install_module_hooks()

    def records(self, writer: TraceWriter) -> list[TensorRecord]:
        self.restore()
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

    def _wrap_randn(self) -> None:
        self._orig_randn = torch.randn

        def wrapped_randn(*args, **kwargs):
            result = self._orig_randn(*args, **kwargs)
            self._store_tensor_once("first_dit_noise", result)
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
    args = parser.parse_args()

    set_seed(args.seed)
    import_local_voxcpm(args.repo_root)

    from voxcpm import VoxCPM

    model = VoxCPM(
        voxcpm_model_path=str(args.model_dir),
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=False,
        device="cpu",
    )

    capture = TraceCapture(model)
    capture.install()

    wav = model.generate(
        text=args.text,
        prompt_wav_path=str(args.prompt_wav_path) if args.prompt_wav_path else None,
        prompt_text=args.prompt_text,
        reference_wav_path=str(args.reference_wav_path) if args.reference_wav_path else None,
        cfg_value=2.0,
        inference_timesteps=4,
        min_len=1,
        max_len=3,
        normalize=False,
        denoise=False,
        retry_badcase=False,
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
            "cfg_value": 2.0,
            "inference_timesteps": 4,
            "min_len": 1,
            "max_len": 3,
            "normalize": False,
            "retry_badcase": False,
        },
        tensors=tensors,
        metadata={"seed": args.seed, "source_model_dir": str(args.model_dir.resolve())},
    )


if __name__ == "__main__":
    main()
