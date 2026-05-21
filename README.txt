VoxUI commands

CUDA/MSVC environment:
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"; $env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"; $env:CUDA_COMPUTE_CAP = "89"; $env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"

Build inference:
cd voxui; cargo build -p voxui-inference --release
cd voxui; cargo build -p voxui-inference --features cuda --release

Build desktop:
cd voxui\crates\voxui-desktop; trunk build --release
cd voxui\crates\voxui-desktop\src-tauri; cargo build --features cuda --release

Run desktop with debug logs:
cd voxui\crates\voxui-desktop\src-tauri; $env:RUST_LOG = "voxui_desktop=debug,voxui_inference=debug"; cargo tauri dev --features cuda

Export fp16 bundles:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-profile fp16
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-profile fp16
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16

Export fp16 bundles with LoRA:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-fp16 --variant 0.5 --quant-profile fp16 --lora-dir VoxCPM/ft0.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-fp16 --variant 1.5 --quant-profile fp16 --lora-dir VoxCPM/ft1.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-fp16 --variant 2.0 --quant-profile fp16 --lora-dir VoxCPM/ft2/latest

Export q4-lm bundles:
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM-0.5B --output-dir models/voxcpm05-q4-lm --variant 0.5 --quant-profile q4-lm --lora-dir VoxCPM/ft0.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM1.5 --output-dir models/voxcpm15-q4-lm --variant 1.5 --quant-profile q4-lm --lora-dir VoxCPM/ft1.5/latest
python exporter/export_voxcpm.py --model-dir VoxCPM/models/VoxCPM2 --output-dir models/voxcpm2-q4-lm --variant 2.0 --quant-profile q4-lm --lora-dir VoxCPM/ft2/latest

Verify GGUF exports:
python exporter/verify_gguf.py models/voxcpm05-q4-lm/model.gguf
python exporter/verify_gguf.py models/voxcpm15-q4-lm/model.gguf
python exporter/verify_gguf.py models/voxcpm2-q4-lm/model.gguf

Run tests:
python -m unittest exporter.tests.test_export_manifest -v
cd voxui; cargo test -p voxui-gguf
cd voxui; cargo test -p voxui-inference --test manifest_loader --features cuda
cd voxui; cargo test -p voxui-inference --test inference_suite voxcpm2_fp16_cuda --features cuda
cd voxui; cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
cd voxui; cargo test -p voxui-inference --features cuda --test inference_suite full_matrix -- --nocapture --test-threads=1
