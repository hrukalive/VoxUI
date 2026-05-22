use half::f16;

use crate::types::GgmlType;

pub fn dequantize(data: &[u8], dtype: GgmlType, n_elements: usize) -> anyhow::Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => dequantize_f32(data, n_elements),
        GgmlType::F16 => dequantize_f16(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
    }
}

fn dequantize_f32(data: &[u8], n_elements: usize) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(data.len() >= n_elements * 4, "insufficient data for F32");
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bytes = [
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

fn dequantize_f16(data: &[u8], n_elements: usize) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(data.len() >= n_elements * 2, "insufficient data for F16");
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn dequantize_q4_0(data: &[u8], n_elements: usize) -> anyhow::Result<Vec<f32>> {
    let n_blocks = (n_elements + 31) / 32;
    anyhow::ensure!(data.len() >= n_blocks * 18, "insufficient data for Q4_0");
    let mut out = Vec::with_capacity(n_blocks * 32);
    let mut offset = 0;
    for _ in 0..n_blocks {
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        offset += 2;
        for j in 0..16 {
            let byte = data[offset + j];
            let lo = (byte & 0xF) as i32 - 8;
            out.push(lo as f32 * scale);
        }
        for j in 0..16 {
            let byte = data[offset + j];
            let hi = (byte >> 4) as i32 - 8;
            out.push(hi as f32 * scale);
        }
        offset += 16;
    }
    out.truncate(n_elements);
    Ok(out)
}

fn dequantize_q8_0(data: &[u8], n_elements: usize) -> anyhow::Result<Vec<f32>> {
    let n_blocks = (n_elements + 31) / 32;
    anyhow::ensure!(data.len() >= n_blocks * 34, "insufficient data for Q8_0");
    let mut out = Vec::with_capacity(n_blocks * 32);
    let mut offset = 0;
    for _ in 0..n_blocks {
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        offset += 2;
        for i in 0..32 {
            let val = data[offset + i] as i8;
            out.push(val as f32 * scale);
        }
        offset += 32;
    }
    out.truncate(n_elements);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_dequantizes_low_nibbles_as_first_half_of_block() -> anyhow::Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        data.extend((0u8..16).map(|i| i | (i << 4)));

        let values = dequantize(&data, GgmlType::Q4_0, 32)?;
        let expected = (-8..8)
            .chain(-8..8)
            .map(|value| value as f32)
            .collect::<Vec<_>>();

        assert_eq!(values, expected);
        Ok(())
    }
}
