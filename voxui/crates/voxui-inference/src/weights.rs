use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{
    quantized::{self, GgmlDType, QMatMul},
    DType, Device, Module, Tensor,
};
use voxui_gguf::{GgmlType, RawTensor};

#[derive(Clone, Debug)]
pub(crate) enum RuntimeTensor {
    Dense(Tensor),
    Quantized(Arc<QuantizedTensor>),
}

#[derive(Debug)]
pub(crate) struct QuantizedTensor {
    name: String,
    shape: Vec<usize>,
    dtype: GgmlDType,
    raw: Arc<[u8]>,
    device: Device,
}

#[derive(Clone, Debug)]
pub(crate) enum LinearWeight {
    Dense(Tensor),
    Quantized(QMatMul),
}

pub(crate) fn map_ggml_dtype(dtype: GgmlType) -> Result<GgmlDType> {
    match dtype {
        GgmlType::F32 => Ok(GgmlDType::F32),
        GgmlType::F16 => Ok(GgmlDType::F16),
        GgmlType::Q4_0 => Ok(GgmlDType::Q4_0),
        GgmlType::Q8_0 => Ok(GgmlDType::Q8_0),
    }
}

impl RuntimeTensor {
    pub(crate) fn from_raw_quantized(raw: RawTensor<'_>, device: &Device) -> Result<Self> {
        let dtype = map_ggml_dtype(raw.info.dtype)?;
        let shape = raw
            .info
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();
        let raw_bytes: Arc<[u8]> = Arc::from(raw.data.to_vec().into_boxed_slice());
        let tensor = QuantizedTensor {
            name: raw.info.name.clone(),
            shape,
            dtype,
            raw: raw_bytes,
            device: device.clone(),
        };
        Ok(RuntimeTensor::Quantized(Arc::new(tensor)))
    }

    pub(crate) fn to_dense_dtype(&self, dtype: DType) -> Result<Tensor> {
        match self {
            RuntimeTensor::Dense(tensor) => tensor.to_dtype(dtype).map_err(Into::into),
            RuntimeTensor::Quantized(tensor) => {
                tensor.dequantize()?.to_dtype(dtype).map_err(Into::into)
            }
        }
    }

    pub(crate) fn embedding_rows(&self, token_ids: &[u32], dtype: DType) -> Result<Tensor> {
        match self {
            RuntimeTensor::Dense(tensor) => {
                let ids = Tensor::new(token_ids, tensor.device())?;
                tensor.index_select(&ids, 0)?.to_dtype(dtype).map_err(Into::into)
            }
            RuntimeTensor::Quantized(tensor) => tensor.embedding_rows(token_ids, dtype),
        }
    }
}

impl QuantizedTensor {
    fn dequantize(&self) -> Result<Tensor> {
        let qtensor = quantized::ggml_file::qtensor_from_ggml(
            self.dtype,
            self.raw.as_ref(),
            self.shape.clone(),
            &self.device,
        )
        .with_context(|| format!("dequantize quantized tensor {}", self.name))?;
        qtensor.dequantize(&self.device).map_err(Into::into)
    }

    fn embedding_rows(&self, token_ids: &[u32], dtype: DType) -> Result<Tensor> {
        let dense = self.dequantize()?;
        let ids = Tensor::new(token_ids, &self.device)?;
        dense.index_select(&ids, 0)?.to_dtype(dtype).map_err(Into::into)
    }
}

impl LinearWeight {
    pub(crate) fn input_dtype(&self) -> DType {
        match self {
            LinearWeight::Dense(weight) => weight.dtype(),
            LinearWeight::Quantized(_) => DType::F32,
        }
    }

    pub(crate) fn from_raw_quantized(raw: RawTensor<'_>, device: &Device) -> Result<Self> {
        let dtype = map_ggml_dtype(raw.info.dtype)?;
        let shape = raw
            .info
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();
        let qtensor = quantized::ggml_file::qtensor_from_ggml(dtype, raw.data, shape, device)
            .with_context(|| format!("load quantized linear tensor {}", raw.info.name))?;
        Ok(LinearWeight::Quantized(QMatMul::from_qtensor(qtensor)?))
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            LinearWeight::Dense(weight) => {
                if input.dtype() == weight.dtype() {
                    crate::linear(input, weight)
                } else {
                    crate::linear(&input.to_dtype(weight.dtype())?, weight)?
                        .to_dtype(input.dtype())
                        .map_err(Into::into)
                }
            }
            LinearWeight::Quantized(weight) => {
                let input_dtype = input.dtype();
                let out = weight.forward(&input.to_dtype(DType::F32)?)?;
                out.to_dtype(input_dtype).map_err(Into::into)
            }
        }
    }

    #[cfg(test)]
    fn quantize_for_test(weight: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let qtensor = quantized::QTensor::quantize(weight, dtype)?;
        Ok(Self::Quantized(QMatMul::from_qtensor(qtensor)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weight(device: &Device) -> Result<Tensor> {
        let data = (0..64)
            .map(|v| (v as f32 - 31.0) / 16.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (2, 32), device).map_err(Into::into)
    }

    #[test]
    fn q4_linear_forward_matches_dense_shape_and_reasonable_values() -> Result<()> {
        let device = Device::Cpu;
        let weight = test_weight(&device)?;
        let input = Tensor::from_vec(vec![0.25f32; 64], (2, 32), &device)?;

        let dense = LinearWeight::Dense(weight.clone());
        let q4 = LinearWeight::quantize_for_test(&weight, candle_core::quantized::GgmlDType::Q4_0)?;

        let dense_out = dense.forward(&input)?;
        let q4_out = q4.forward(&input)?;

        assert_eq!(dense_out.dims(), &[2, 2]);
        assert_eq!(q4_out.dims(), &[2, 2]);
        let dense_values = dense_out.to_vec2::<f32>()?;
        let q4_values = q4_out.to_vec2::<f32>()?;
        for (dense_row, q4_row) in dense_values.iter().zip(q4_values.iter()) {
            for (dense_value, q4_value) in dense_row.iter().zip(q4_row.iter()) {
                assert!(
                    (dense_value - q4_value).abs() < 0.50,
                    "dense={dense_value} q4={q4_value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn dense_runtime_tensor_can_materialize_matching_dtype() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let runtime = RuntimeTensor::Dense(tensor);

        let materialized = runtime.to_dense_dtype(DType::F32)?;

        assert_eq!(materialized.dims(), &[2, 2]);
        assert_eq!(materialized.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn linear_weight_reports_dense_or_quantized_input_dtype() -> Result<()> {
        let device = Device::Cpu;
        let weight = test_weight(&device)?;
        let dense = LinearWeight::Dense(weight.clone());
        let q4 = LinearWeight::quantize_for_test(&weight, candle_core::quantized::GgmlDType::Q4_0)?;

        assert_eq!(dense.input_dtype(), DType::F32);
        assert_eq!(q4.input_dtype(), DType::F32);
        Ok(())
    }
}
