import unittest

import numpy as np

from exporter.quantize import dequantize_q4_0, quantize_q4_0


class Q4QuantizeTests(unittest.TestCase):
    def test_q4_0_packs_low_nibbles_as_first_half_of_block(self):
        values = np.array(list(range(-8, 8)) * 2, dtype=np.float32)

        data = quantize_q4_0(values)

        self.assertEqual(data[:2], np.float16(1.0).tobytes())
        self.assertEqual(data[2:], bytes((i | (i << 4)) for i in range(16)))

    def test_q4_0_dequantizes_ggml_half_block_nibble_order(self):
        data = np.float16(1.0).tobytes() + bytes((i | (i << 4)) for i in range(16))

        values = dequantize_q4_0(data, [32])

        np.testing.assert_array_equal(
            values,
            np.array(list(range(-8, 8)) * 2, dtype=np.float32),
        )

    def test_q4_0_uses_signed_scale_for_positive_max_blocks(self):
        values = np.array([-1.0] * 31 + [8.0], dtype=np.float32)

        data = quantize_q4_0(values)

        self.assertEqual(data[:2], np.float16(-1.0).tobytes())
        self.assertEqual(data[2:], bytes([0x99] * 15 + [0x09]))
        np.testing.assert_array_equal(dequantize_q4_0(data, [32]), values)


if __name__ == "__main__":
    unittest.main()
