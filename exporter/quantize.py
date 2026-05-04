"""Quantization functions for GGUF export: F32, FP16, Q8_0, Q4_0."""

import numpy as np
import struct


def quantize_f32(tensor: np.ndarray) -> bytes:
    """Keep float32 values and return raw little-endian bytes."""
    return tensor.astype(np.float32).tobytes()


def quantize_fp16(tensor: np.ndarray) -> bytes:
    """Cast float32 to float16, return raw bytes."""
    return tensor.astype(np.float16).tobytes()


def quantize_q8_0(tensor: np.ndarray) -> bytes:
    """Block quantize to Q8_0 format.
    Block size = 32. Per-block: 2 bytes (f16 scale) + 32 bytes (int8) = 34 bytes.
    """
    BLOCK_SIZE = 32
    flat = tensor.flatten().astype(np.float32)
    remainder = len(flat) % BLOCK_SIZE
    if remainder:
        flat = np.concatenate([flat, np.zeros(BLOCK_SIZE - remainder, dtype=np.float32)])

    n_blocks = len(flat) // BLOCK_SIZE
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    # Compute scales: max absolute value per block / 127
    amax = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    scales = np.where(amax != 0, amax / 127.0, 0.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)

    # Quantize: round(block / scale), clip to [-128, 127]
    # Avoid division by zero
    safe_scales = np.where(scales_f32 != 0, scales_f32, 1.0)[:, np.newaxis]
    qi = np.round(blocks / safe_scales).astype(np.int32)
    qi = np.clip(qi, -128, 127).astype(np.int8)
    # Zero out blocks with zero scale
    zero_mask = (scales_f32 == 0)[:, np.newaxis]
    qi = np.where(zero_mask, np.int8(0), qi)

    # Interleave scale (2 bytes) + quants (32 bytes) per block
    scale_bytes = scales.view(np.uint8).reshape(n_blocks, 2)
    qi_bytes = qi.view(np.uint8).reshape(n_blocks, BLOCK_SIZE)
    out = np.empty((n_blocks, 34), dtype=np.uint8)
    out[:, :2] = scale_bytes
    out[:, 2:] = qi_bytes
    return out.tobytes()


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    """Block quantize to Q4_0 format.
    Block size = 32. Per-block: 2 bytes (f16 scale) + 16 bytes (packed nibbles) = 18 bytes.
    """
    BLOCK_SIZE = 32
    flat = tensor.flatten().astype(np.float32)
    remainder = len(flat) % BLOCK_SIZE
    if remainder:
        flat = np.concatenate([flat, np.zeros(BLOCK_SIZE - remainder, dtype=np.float32)])

    n_blocks = len(flat) // BLOCK_SIZE
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    # Compute scales
    amax = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    scales = np.where(amax != 0, amax / 8.0, 0.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)

    # Quantize
    safe_scales = np.where(scales_f32 != 0, scales_f32, 1.0)[:, np.newaxis]
    qi = np.round(blocks / safe_scales).astype(np.int32)
    qi = np.clip(qi, -8, 7)

    # Unsigned offset: qi + 8 maps [-8,7] -> [0,15]
    qu = (qi + 8).astype(np.uint8)

    # For zero-scale blocks, qu should be 8 (the zero point)
    zero_mask = (scales_f32 == 0)[:, np.newaxis]
    qu = np.where(zero_mask, np.uint8(8), qu)

    # Pack pairs of nibbles: low from even indices, high from odd indices
    lo = qu[:, 0::2] & 0xF   # (n_blocks, 16)
    hi = qu[:, 1::2] & 0xF   # (n_blocks, 16)
    packed = (lo | (hi << 4)).astype(np.uint8)  # (n_blocks, 16)

    # Interleave scale + packed
    scale_bytes = scales.view(np.uint8).reshape(n_blocks, 2)
    out = np.empty((n_blocks, 18), dtype=np.uint8)
    out[:, :2] = scale_bytes
    out[:, 2:] = packed
    return out.tobytes()


def dequantize_q8_0(data: bytes, shape: list[int]) -> np.ndarray:
    """Dequantize Q8_0 back to float32."""
    BLOCK_SIZE = 32
    BLOCK_BYTES = 34
    n_blocks = len(data) // BLOCK_BYTES
    n_elements = int(np.prod(shape))

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
    qi = raw[:, 2:].copy().view(np.int8).astype(np.float32)  # (n_blocks, 32)
    out = (qi * scales).reshape(-1)
    return out[:n_elements].reshape(shape)


def dequantize_q4_0(data: bytes, shape: list[int]) -> np.ndarray:
    """Dequantize Q4_0 back to float32."""
    BLOCK_SIZE = 32
    BLOCK_BYTES = 18
    n_blocks = len(data) // BLOCK_BYTES
    n_elements = int(np.prod(shape))

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
    packed = raw[:, 2:]  # (n_blocks, 16)

    # Unpack nibbles
    lo = (packed & 0xF).astype(np.int32) - 8        # (n_blocks, 16)
    hi = ((packed >> 4) & 0xF).astype(np.int32) - 8  # (n_blocks, 16)

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    vals = np.empty((n_blocks, BLOCK_SIZE), dtype=np.float32)
    vals[:, 0::2] = lo * scales
    vals[:, 1::2] = hi * scales

    out = vals.reshape(-1)
    return out[:n_elements].reshape(shape)


if __name__ == "__main__":
    np.random.seed(42)
    tensor = np.random.randn(128).astype(np.float32)

    # FP16
    fp16_data = quantize_fp16(tensor)
    assert len(fp16_data) == 128 * 2, f"FP16 size mismatch: {len(fp16_data)} != 256"
    print(f"FP16: {len(fp16_data)} bytes (expected 256) OK")

    # Q8_0
    q8_data = quantize_q8_0(tensor)
    expected_q8 = (128 // 32) * 34
    assert len(q8_data) == expected_q8, f"Q8_0 size mismatch: {len(q8_data)} != {expected_q8}"
    q8_deq = dequantize_q8_0(q8_data, [128])
    # Check error on non-tiny values
    # Q8_0: max absolute error per block is ~d/127 where d=max(abs(block))/127, so ~max/127^2
    # Use RMSE and absolute error checks instead of relative error on small values
    q8_abs_err = np.max(np.abs(q8_deq - tensor))
    q8_rmse = np.sqrt(np.mean((q8_deq - tensor) ** 2))
    print(f"Q8_0: {len(q8_data)} bytes (expected {expected_q8}), max abs error: {q8_abs_err:.6f}, rmse: {q8_rmse:.6f}")
    assert q8_abs_err < 0.05, f"Q8_0 abs error too high: {q8_abs_err}"
    print("Q8_0 OK")

    # Q4_0
    q4_data = quantize_q4_0(tensor)
    expected_q4 = (128 // 32) * 18
    assert len(q4_data) == expected_q4, f"Q4_0 size mismatch: {len(q4_data)} != {expected_q4}"
    q4_deq = dequantize_q4_0(q4_data, [128])
    q4_abs_err = np.max(np.abs(q4_deq - tensor))
    q4_rmse = np.sqrt(np.mean((q4_deq - tensor) ** 2))
    print(f"Q4_0: {len(q4_data)} bytes (expected {expected_q4}), max abs error: {q4_abs_err:.4f}, rmse: {q4_rmse:.4f}")
    assert q4_abs_err < 0.5, f"Q4_0 abs error too high: {q4_abs_err}"
    print("Q4_0 OK")

    print("\nAll self-tests passed.")
