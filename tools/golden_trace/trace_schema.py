from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TensorRecord:
    name: str
    file: str
    dtype: str
    shape: list[int]


class TraceWriter:
    def __init__(self, root: Path, case_name: str) -> None:
        self.case_dir = root / case_name
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.lists: dict[str, list[int]] = {}

    def write_tensor(self, name: str, tensor: np.ndarray) -> TensorRecord:
        arr = np.asarray(tensor, dtype=np.float32)
        file_name = f"{name}.f32"
        arr.tofile(self.case_dir / file_name)
        return TensorRecord(name=name, file=file_name, dtype="f32", shape=list(arr.shape))

    def write_u32_list(self, name: str, values: list[int] | np.ndarray) -> None:
        self.lists[name] = [int(v) for v in values]

    def write_manifest(
        self,
        *,
        variant: str,
        architecture: str,
        request: dict[str, Any],
        tensors: list[TensorRecord],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "variant": variant,
            "architecture": architecture,
            "request": request,
            "metadata": metadata or {},
            "lists": self.lists,
            "tensors": [asdict(t) for t in tensors],
        }
        (self.case_dir / "trace.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def read_tensor_record(case_dir: Path, record: TensorRecord) -> np.ndarray:
    dtype = np.float32 if record.dtype == "f32" else None
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {record.dtype}")
    arr = np.fromfile(case_dir / record.file, dtype=dtype)
    return arr.reshape(record.shape)
