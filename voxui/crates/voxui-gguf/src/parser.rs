use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{bail, ensure, Context};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;

use crate::dequant;
use crate::types::*;

const GGUF_MAGIC: u32 = 0x46554747;

fn read_string(cursor: &mut Cursor<&[u8]>) -> anyhow::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_metadata_value(cursor: &mut Cursor<&[u8]>, vtype: u32) -> anyhow::Result<MetadataValue> {
    match vtype {
        0 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        1 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_string(cursor)?)),
        9 => {
            // Array
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            match elem_type {
                4 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(cursor.read_u32::<LittleEndian>()?);
                    }
                    Ok(MetadataValue::ArrayUint32(v))
                }
                5 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(cursor.read_i32::<LittleEndian>()?);
                    }
                    Ok(MetadataValue::ArrayInt32(v))
                }
                6 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(cursor.read_f32::<LittleEndian>()?);
                    }
                    Ok(MetadataValue::ArrayFloat32(v))
                }
                8 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(read_string(cursor)?);
                    }
                    Ok(MetadataValue::ArrayString(v))
                }
                _ => {
                    // Skip unsupported array types by reading element by element
                    for _ in 0..count {
                        let _ = read_metadata_value(cursor, elem_type)?;
                    }
                    // Return as string description
                    Ok(MetadataValue::String(format!(
                        "<array type={} count={}>",
                        elem_type, count
                    )))
                }
            }
        }
        10 => Ok(MetadataValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        other => bail!("unknown metadata value type: {}", other),
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data: &[u8] = &mmap;
        let mut cursor = Cursor::new(data);

        // Header
        let magic = cursor.read_u32::<LittleEndian>()?;
        ensure!(magic == GGUF_MAGIC, "not a GGUF file (bad magic)");

        let version = cursor.read_u32::<LittleEndian>()?;
        ensure!(version == 3, "unsupported GGUF version: {}", version);

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        // Metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let vtype = cursor.read_u32::<LittleEndian>()?;
            let value = read_metadata_value(&mut cursor, vtype)
                .with_context(|| format!("parsing metadata key '{}'", key))?;
            metadata.insert(key, value);
        }

        // Tensor infos
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_map = HashMap::with_capacity(tensor_count);
        for i in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw).with_context(|| {
                format!("tensor '{}' has unsupported dtype {}", name, dtype_raw)
            })?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            let data_size = compute_data_size(&shape, dtype);

            tensor_map.insert(name.clone(), i);
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
                data_size,
            });
        }

        // Align to 32 bytes for data section
        let pos = cursor.position() as usize;
        let data_offset = (pos + 31) & !31;

        Ok(GgufFile {
            metadata,
            tensors,
            tensor_map,
            mmap,
            data_offset,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let info = self
            .tensor_info(name)
            .with_context(|| format!("tensor '{}' not found", name))?;
        let data = self
            .tensor_data(name)
            .with_context(|| format!("tensor '{}' data out of bounds", name))?;
        let n_elements: usize = info.shape.iter().product::<u64>() as usize;
        dequant::dequantize(data, info.dtype, n_elements)
    }

    pub fn tensor_raw(&self, name: &str) -> anyhow::Result<RawTensor<'_>> {
        let info = self
            .tensor_info(name)
            .with_context(|| format!("tensor '{}' not found", name))?;
        let data = self
            .tensor_data(name)
            .with_context(|| format!("tensor '{}' data out of bounds", name))?;
        Ok(RawTensor { info, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_string(mut out: impl Write, value: &str) -> anyhow::Result<()> {
        out.write_u64::<LittleEndian>(value.len() as u64)?;
        out.write_all(value.as_bytes())?;
        Ok(())
    }

    fn align_32(len: usize) -> usize {
        (len + 31) & !31
    }

    fn write_test_gguf(path: &std::path::Path) -> anyhow::Result<()> {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "dense")?;
        bytes.write_u32::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(4)?;
        bytes.write_u32::<LittleEndian>(0)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "q4")?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(32)?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(16)?;

        bytes.resize(align_32(bytes.len()), 0);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&4.0f32.to_le_bytes());
        bytes.extend_from_slice(&[0x00; 18]);
        std::fs::write(path, bytes)?;
        Ok(())
    }

    #[test]
    fn tensor_raw_exposes_dtype_shape_and_bytes_without_dequantizing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.gguf");
        write_test_gguf(&path)?;

        let gguf = GgufFile::open(&path)?;
        let raw = gguf.tensor_raw("q4")?;

        assert_eq!(raw.info.name, "q4");
        assert_eq!(raw.info.shape, vec![1, 32]);
        assert_eq!(raw.info.dtype, GgmlType::Q4_0);
        assert_eq!(raw.info.element_count(), 32);
        assert!(raw.info.is_quantized());
        assert_eq!(raw.data.len(), 18);
        Ok(())
    }

    #[test]
    fn dense_tensor_info_reports_non_quantized_element_count() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.gguf");
        write_test_gguf(&path)?;

        let gguf = GgufFile::open(&path)?;
        let raw = gguf.tensor_raw("dense")?;

        assert_eq!(raw.info.dtype, GgmlType::F32);
        assert_eq!(raw.info.element_count(), 4);
        assert!(!raw.info.is_quantized());
        assert_eq!(raw.data.len(), 16);
        Ok(())
    }
}
