# VoxCPM Golden Traces

These files are generated from the local Python VoxCPM reference implementation
and are used only by tests. Runtime inference in `voxui-inference` is pure Rust
Candle.

Regenerate after exporter or model-graph changes:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM-0.5B --variant 0.5 --case-name voxcpm05_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM1.5 --variant 1.5 --case-name voxcpm15_zero_shot --text "Hello, welcome to the stream!"
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM2 --variant 2.0 --case-name voxcpm2_zero_shot --text "Hello, welcome to the stream!"
```

For VoxCPM2 reference-audio traces, choose a local WAV from `for_test_wav/`:

```powershell
python tools/golden_trace/voxcpm_trace.py --model-dir VoxCPM/models/VoxCPM2 --variant 2.0 --case-name voxcpm2_reference --text "Hello, welcome to the stream!" --reference-wav-path "for_test_wav/感谢大家一个月以来的陪伴，也感谢大家一个月以来的支持。.wav"
```
