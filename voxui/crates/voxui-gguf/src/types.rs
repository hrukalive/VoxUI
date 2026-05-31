use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    ArrayFloat32(Vec<f32>),
    ArrayString(Vec<String>),
    ArrayUint32(Vec<u32>),
    ArrayInt32(Vec<i32>),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u32),
            MetadataValue::Uint64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            MetadataValue::Int64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            MetadataValue::Uint8(v) => Some(*v as u32),
            MetadataValue::Uint16(v) => Some(*v as u32),
            MetadataValue::Int16(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            MetadataValue::Int32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            MetadataValue::Int32(v) => Some(*v as f32),
            MetadataValue::Uint32(v) => Some(*v as f32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> anyhow::Result<Self> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            8 => Ok(GgmlType::Q8_0),
            other => anyhow::bail!("unsupported GGML type: {}", other),
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, GgmlType::Q4_0 | GgmlType::Q8_0)
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgmlType::F32 => write!(f, "F32"),
            GgmlType::F16 => write!(f, "F16"),
            GgmlType::Q4_0 => write!(f, "Q4_0"),
            GgmlType::Q8_0 => write!(f, "Q8_0"),
        }
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
    pub data_size: usize,
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn is_quantized(&self) -> bool {
        self.dtype.is_quantized()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RawTensor<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

pub struct GgufFile {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub(crate) tensor_map: HashMap<String, usize>,
    pub(crate) mmap: memmap2::Mmap,
    pub(crate) data_offset: usize,
}

impl GgufFile {
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_map.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        let start = self.data_offset + info.offset as usize;
        let end = start + info.data_size;
        Some(&self.mmap[start..end])
    }

    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }
}

pub fn compute_data_size(shape: &[u64], dtype: GgmlType) -> usize {
    let n_elements: usize = shape.iter().product::<u64>() as usize;
    match dtype {
        GgmlType::F32 => n_elements * 4,
        GgmlType::F16 => n_elements * 2,
        GgmlType::Q4_0 => {
            n_elements.div_ceil(32) * 18
        }
        GgmlType::Q8_0 => {
            n_elements.div_ceil(32) * 34
        }
    }
}
