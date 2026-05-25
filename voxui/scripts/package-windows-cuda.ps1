param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$PackageDir = "target\dist\voxui-cli-windows-cuda",
    [string]$ComputeCap = "89",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CudaPath)) {
    throw "CUDA_PATH is not set. Pass -CudaPath or set CUDA_PATH to a CUDA Toolkit install, for example C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6."
}

$cudaBin = Join-Path $CudaPath "bin"
if (-not (Test-Path -LiteralPath $cudaBin)) {
    throw "CUDA bin directory not found: $cudaBin"
}

$cudaDlls = @(
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "curand64_10.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_126.dll"
)

$missing = @(
    foreach ($dll in $cudaDlls) {
        $path = Join-Path $cudaBin $dll
        if (-not (Test-Path -LiteralPath $path)) {
            $dll
        }
    }
)

if ($missing.Count -gt 0) {
    throw "Missing CUDA DLL(s) in ${cudaBin}: $($missing -join ', ')"
}

if (-not $SkipBuild) {
    $env:CUDA_PATH = $CudaPath
    $env:PATH = "$cudaBin;$env:PATH"
    $env:CUDA_COMPUTE_CAP = $ComputeCap
    $env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"

    cargo build --release -p voxui-cli --features cuda
}

$exePath = "target\release\voxui-cli.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Release executable not found: $exePath"
}

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
Copy-Item -LiteralPath $exePath -Destination $PackageDir -Force

$pdbPath = "target\release\voxui_cli.pdb"
if (Test-Path -LiteralPath $pdbPath) {
    Copy-Item -LiteralPath $pdbPath -Destination $PackageDir -Force
}

foreach ($dll in $cudaDlls) {
    Copy-Item -LiteralPath (Join-Path $cudaBin $dll) -Destination $PackageDir -Force
}

Write-Host "Packaged voxui-cli CUDA release at $PackageDir"
Get-ChildItem -LiteralPath $PackageDir | Select-Object Name, Length
