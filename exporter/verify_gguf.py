"""GGUF v3 binary format verification tool.

Usage:
    python exporter/verify_gguf.py <path>

Where <path> is a .gguf file or a directory containing .gguf files.
"""

import struct
import sys
import os
import math
from pathlib import Path

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
}

# bytes per element (or per block for quantized)
def _type_data_size(dtype, elements):
    """Estimate data size in bytes for `elements` of `dtype`."""
    if dtype == 0:  # F32
        return elements * 4
    elif dtype == 1:  # F16
        return elements * 2
    elif dtype == 2:  # Q4_0
        return math.ceil(elements / 32) * 18
    elif dtype == 8:  # Q8_0
        return math.ceil(elements / 32) * 34
    else:
        return None

METADATA_TYPE_STRUCT = {
    0: ("B", 1),   # UINT8
    1: ("b", 1),   # INT8
    2: ("<H", 2),  # UINT16
    3: ("<h", 2),  # INT16
    4: ("<I", 4),  # UINT32
    5: ("<i", 4),  # INT32
    6: ("<f", 4),  # FLOAT32
    7: ("?", 1),   # BOOL
    # 8 = STRING
    # 9 = ARRAY
    10: ("<Q", 8), # UINT64
    11: ("<q", 8), # INT64
    12: ("<d", 8), # FLOAT64
}


def _read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_value(f, vtype):
    if vtype == 8:  # STRING
        return _read_string(f)
    elif vtype == 9:  # ARRAY
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    else:
        fmt, size = METADATA_TYPE_STRUCT[vtype]
        return struct.unpack(fmt, f.read(size))[0]


def _fmt_size(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    elif nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    elif nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def _align(pos, alignment=32):
    return (pos + alignment - 1) // alignment * alignment


def verify_file(path):
    file_size = os.path.getsize(path)
    print(f"\n=== {os.path.basename(path)} ({_fmt_size(file_size)}) ===")

    with open(path, "rb") as f:
        # Magic
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"  ERROR: Bad magic {magic!r}, expected b'GGUF'")
            return False

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_count = struct.unpack("<Q", f.read(8))[0]

        print(f"  Version: {version}")
        print(f"  Tensor count: {tensor_count}")
        print(f"  Metadata KV count: {metadata_count}")

        # Metadata
        print(f"\nMetadata:")
        for _ in range(metadata_count):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _read_value(f, vtype)
            val_str = repr(val) if isinstance(val, str) else str(val)
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            print(f"  {key} = {val_str}")

        # Tensor info
        tensors = []
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, shape, dtype, offset))

        # Data section start
        data_start = _align(f.tell())

        print(f"\nTensors ({len(tensors)} total):")
        total_data = 0
        for name, shape, dtype, offset in tensors:
            dtype_name = GGML_TYPE_NAMES.get(dtype, f"type_{dtype}")
            elements = 1
            for d in shape:
                elements *= d
            est_size = _type_data_size(dtype, elements)
            size_str = _fmt_size(est_size) if est_size else "?"
            if est_size:
                total_data += est_size
            shape_str = str(shape).replace(" ", "")
            print(f"  {name:<55} {shape_str:<20} {dtype_name:<6} {size_str}")

        print(f"\nTotal: {len(tensors)} tensors, {_fmt_size(total_data)} data")
        return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.gguf"))
        if not files:
            print(f"No .gguf files found in {target}")
            sys.exit(1)
        ok = True
        for f in files:
            try:
                if not verify_file(str(f)):
                    ok = False
            except Exception as e:
                print(f"\n  ERROR reading {f.name}: {e}")
                ok = False
        sys.exit(0 if ok else 1)
    elif target.is_file():
        try:
            ok = verify_file(str(target))
            sys.exit(0 if ok else 1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        print(f"Not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
