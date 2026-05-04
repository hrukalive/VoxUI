import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.golden_trace.trace_schema import TensorRecord, TraceWriter, read_tensor_record


class TraceSchemaTests(unittest.TestCase):
    def test_tensor_record_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = TraceWriter(root, case_name="unit")
            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            record = writer.write_tensor("base_lm_hidden", arr)
            writer.write_manifest(
                variant="2.0",
                architecture="voxcpm2",
                request={"text": "hello"},
                tensors=[record],
            )

            manifest = json.loads((root / "unit" / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["tensors"][0]["shape"], [3, 4])
            restored = read_tensor_record(root / "unit", TensorRecord(**manifest["tensors"][0]))
            np.testing.assert_allclose(restored, arr)


if __name__ == "__main__":
    unittest.main()
