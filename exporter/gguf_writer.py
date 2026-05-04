"""GGUF v3 binary format writer."""

import struct
import io
import math

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

# GGUF metadata value types
GGUF_METADATA_VALUE_TYPE_UINT8 = 0
GGUF_METADATA_VALUE_TYPE_INT8 = 1
GGUF_METADATA_VALUE_TYPE_UINT16 = 2
GGUF_METADATA_VALUE_TYPE_INT16 = 3
GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_INT32 = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL = 7
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9
GGUF_METADATA_VALUE_TYPE_UINT64 = 10
GGUF_METADATA_VALUE_TYPE_INT64 = 11
GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12

_TYPE_TO_STRUCT = {
    GGUF_METADATA_VALUE_TYPE_UINT8: ("B", 1),
    GGUF_METADATA_VALUE_TYPE_INT8: ("b", 1),
    GGUF_METADATA_VALUE_TYPE_UINT16: ("<H", 2),
    GGUF_METADATA_VALUE_TYPE_INT16: ("<h", 2),
    GGUF_METADATA_VALUE_TYPE_UINT32: ("<I", 4),
    GGUF_METADATA_VALUE_TYPE_INT32: ("<i", 4),
    GGUF_METADATA_VALUE_TYPE_FLOAT32: ("<f", 4),
    GGUF_METADATA_VALUE_TYPE_BOOL: ("?", 1),
    GGUF_METADATA_VALUE_TYPE_UINT64: ("<Q", 8),
    GGUF_METADATA_VALUE_TYPE_INT64: ("<q", 8),
    GGUF_METADATA_VALUE_TYPE_FLOAT64: ("<d", 8),
}

ALIGNMENT = 32

# Expected data sizes per element for each tensor type
def _expected_tensor_size(shape: list[int], dtype: int) -> int:
    elements = 1
    for d in shape:
        elements *= d
    if dtype == GGML_TYPE_F32:
        return elements * 4
    elif dtype == GGML_TYPE_F16:
        return elements * 2
    elif dtype == GGML_TYPE_Q4_0:
        return math.ceil(elements / 32) * 18
    elif dtype == GGML_TYPE_Q8_0:
        return math.ceil(elements / 32) * 34
    else:
        return None  # unknown type, skip validation


def _align(pos: int, alignment: int = ALIGNMENT) -> int:
    return (pos + alignment - 1) // alignment * alignment


class GGUFWriter:
    def __init__(self):
        self._metadata: list[tuple[str, object, int]] = []  # (key, value, type)
        self._tensors: list[tuple[str, bytes, list[int], int]] = []  # (name, data, shape, dtype)

    def add_metadata(self, key: str, value, value_type=None):
        """Add a metadata key-value pair. Auto-detect type if not specified."""
        if any(k == key for k, _, _ in self._metadata):
            raise ValueError(f"Duplicate metadata key: {key!r}")
        if value_type is None:
            if isinstance(value, bool):
                value_type = GGUF_METADATA_VALUE_TYPE_BOOL
            elif isinstance(value, str):
                value_type = GGUF_METADATA_VALUE_TYPE_STRING
            elif isinstance(value, int):
                if -(2**31) <= value <= 2**31 - 1:
                    value_type = GGUF_METADATA_VALUE_TYPE_INT32
                elif value >= 0:
                    value_type = GGUF_METADATA_VALUE_TYPE_UINT64
                else:
                    value_type = GGUF_METADATA_VALUE_TYPE_INT64
            elif isinstance(value, float):
                value_type = GGUF_METADATA_VALUE_TYPE_FLOAT32
            elif isinstance(value, list):
                value_type = GGUF_METADATA_VALUE_TYPE_ARRAY
            else:
                raise ValueError(f"Cannot auto-detect type for {type(value)}")
        self._metadata.append((key, value, value_type))

    def add_tensor(self, name: str, data: bytes, shape: list[int], dtype: int):
        """Add a tensor. data is pre-quantized raw bytes. dtype is GGML type enum."""
        if any(n == name for n, _, _, _ in self._tensors):
            raise ValueError(f"Duplicate tensor name: {name!r}")
        expected = _expected_tensor_size(shape, dtype)
        if expected is not None and len(data) != expected:
            raise ValueError(
                f"Tensor {name!r}: data size {len(data)} does not match "
                f"expected {expected} for shape {shape} and dtype {dtype}"
            )
        self._tensors.append((name, data, shape, dtype))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_value(self, f, value, value_type: int):
        if value_type == GGUF_METADATA_VALUE_TYPE_STRING:
            self._write_string(f, value)
        elif value_type == GGUF_METADATA_VALUE_TYPE_ARRAY:
            # Detect element type from first element
            if len(value) == 0:
                raise ValueError("Empty arrays not supported (cannot determine element type)")
            first = value[0]
            first_type = type(first)
            for i, elem in enumerate(value):
                if type(elem) is not first_type:
                    raise TypeError(
                        f"Array element {i} has type {type(elem).__name__}, "
                        f"expected {first_type.__name__}"
                    )
            if isinstance(first, bool):
                elem_type = GGUF_METADATA_VALUE_TYPE_BOOL
            elif isinstance(first, str):
                elem_type = GGUF_METADATA_VALUE_TYPE_STRING
            elif isinstance(first, int):
                elem_type = GGUF_METADATA_VALUE_TYPE_INT32
            elif isinstance(first, float):
                elem_type = GGUF_METADATA_VALUE_TYPE_FLOAT32
            else:
                raise ValueError(f"Cannot detect array element type for {type(first)}")
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(value)))
            for elem in value:
                self._write_value(f, elem, elem_type)
        else:
            fmt, _ = _TYPE_TO_STRUCT[value_type]
            f.write(struct.pack(fmt, value))

    def _write_metadata_kv(self, f, key: str, value, value_type: int):
        self._write_string(f, key)
        f.write(struct.pack("<I", value_type))
        self._write_value(f, value, value_type)

    def write(self, path: str):
        """Write the complete GGUF file."""
        with open(path, "wb") as f:
            # Header
            f.write(b"GGUF")  # magic
            f.write(struct.pack("<I", 3))  # version
            f.write(struct.pack("<Q", len(self._tensors)))  # tensor_count
            f.write(struct.pack("<Q", len(self._metadata)))  # metadata_kv_count

            # Metadata KV pairs
            for key, value, vtype in self._metadata:
                self._write_metadata_kv(f, key, value, vtype)

            # Compute tensor offsets (relative to start of data section)
            offsets = []
            current_offset = 0
            for _, data, _, _ in self._tensors:
                offsets.append(current_offset)
                current_offset += len(data)
                current_offset = _align(current_offset)

            # Tensor info entries
            for i, (name, data, shape, dtype) in enumerate(self._tensors):
                self._write_string(f, name)
                f.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", offsets[i]))

            # Pad to 32-byte alignment before data section
            pos = f.tell()
            pad = _align(pos) - pos
            if pad > 0:
                f.write(b"\x00" * pad)

            # Write tensor data
            data_section_start = f.tell()
            for i, (_, data, _, _) in enumerate(self._tensors):
                # Seek to correct offset within data section
                target = data_section_start + offsets[i]
                current = f.tell()
                if target < current:
                    raise RuntimeError(
                        f"Tensor data overlap detected: target offset {target} "
                        f"is behind current position {current}"
                    )
                if target > current:
                    f.write(b"\x00" * (target - current))
                f.write(data)


if __name__ == "__main__":
    import tempfile
    import os

    writer = GGUFWriter()

    # Add metadata
    writer.add_metadata("general.architecture", "test")
    writer.add_metadata("general.name", "test-model")
    writer.add_metadata("test.uint32_val", 42, GGUF_METADATA_VALUE_TYPE_UINT32)
    writer.add_metadata("test.float32_val", 3.14, GGUF_METADATA_VALUE_TYPE_FLOAT32)

    # Add tensors
    # F32 tensor: 2x3 = 6 floats = 24 bytes
    f32_data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    writer.add_tensor("test.f32_tensor", f32_data, [2, 3], GGML_TYPE_F32)

    # F16 tensor: 4 elements = 8 bytes
    f16_data = struct.pack("<4H", 0x3C00, 0x4000, 0x4200, 0x4400)  # 1.0, 2.0, 3.0, 4.0 in f16
    writer.add_tensor("test.f16_tensor", f16_data, [4], GGML_TYPE_F16)

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer.write(tmp_path)
        print(f"Wrote GGUF file: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        # Read back and validate
        with open(tmp_path, "rb") as f:
            # Check magic
            magic = f.read(4)
            assert magic == b"GGUF", f"Bad magic: {magic}"

            # Check version
            version = struct.unpack("<I", f.read(4))[0]
            assert version == 3, f"Bad version: {version}"

            # Check counts
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            assert tensor_count == 2, f"Bad tensor count: {tensor_count}"

            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
            assert metadata_kv_count == 4, f"Bad metadata count: {metadata_kv_count}"

            # Read metadata
            def read_string(f):
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8")

            def read_value(f, vtype):
                if vtype == GGUF_METADATA_VALUE_TYPE_STRING:
                    return read_string(f)
                elif vtype == GGUF_METADATA_VALUE_TYPE_ARRAY:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    return [read_value(f, elem_type) for _ in range(count)]
                else:
                    fmt, size = _TYPE_TO_STRUCT[vtype]
                    return struct.unpack(fmt, f.read(size))[0]

            metadata = {}
            for _ in range(metadata_kv_count):
                key = read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                val = read_value(f, vtype)
                metadata[key] = val

            assert metadata["general.architecture"] == "test"
            assert metadata["general.name"] == "test-model"
            assert metadata["test.uint32_val"] == 42
            assert abs(metadata["test.float32_val"] - 3.14) < 0.001

            # Read tensor infos
            tensor_infos = []
            for _ in range(tensor_count):
                name = read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensor_infos.append((name, shape, dtype, offset))

            assert tensor_infos[0][0] == "test.f32_tensor"
            assert tensor_infos[0][1] == [2, 3]
            assert tensor_infos[0][2] == GGML_TYPE_F32
            assert tensor_infos[1][0] == "test.f16_tensor"
            assert tensor_infos[1][1] == [4]
            assert tensor_infos[1][2] == GGML_TYPE_F16

            # Skip to aligned data section
            pos = f.tell()
            aligned_pos = _align(pos)
            f.seek(aligned_pos)
            data_start = f.tell()

            # Read tensor data
            f.seek(data_start + tensor_infos[0][3])
            read_f32 = f.read(len(f32_data))
            assert read_f32 == f32_data, "F32 tensor data mismatch"

            f.seek(data_start + tensor_infos[1][3])
            read_f16 = f.read(len(f16_data))
            assert read_f16 == f16_data, "F16 tensor data mismatch"

        print("All validations passed!")

    finally:
        os.unlink(tmp_path)
