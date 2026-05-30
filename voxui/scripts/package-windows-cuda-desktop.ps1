param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$ComputeCap = "89",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CudaPath)) {
    throw "CUDA_PATH is not set. Pass -CudaPath or set CUDA_PATH to a CUDA Toolkit install, e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6."
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

    Write-Host "Building sidecar with CUDA..."
    cargo build --release -p voxui-inference-sidecar --features cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build voxui-inference-sidecar"
    }

    Write-Host "Building desktop app..."
    cargo build --release -p voxui-desktop
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build voxui-desktop"
    }
}

$sidecarExePath = "target\release\voxui-inference-sidecar.exe"
if (-not (Test-Path -LiteralPath $sidecarExePath)) {
    throw "Sidecar executable not found: $sidecarExePath"
}

$desktopExePath = "target\release\voxui-desktop-bin.exe"
if (-not (Test-Path -LiteralPath $desktopExePath)) {
    throw "Desktop executable not found: $desktopExePath"
}

$pkgDir = "target\package\AhanSays-cuda"
New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null

Copy-Item -LiteralPath $desktopExePath -Destination $pkgDir -Force
Copy-Item -LiteralPath $sidecarExePath -Destination $pkgDir -Force

$sidecarTauriExePath = Join-Path $pkgDir "voxui-inference-sidecar-x86_64-pc-windows-msvc.exe"
Copy-Item -LiteralPath $sidecarExePath -Destination $sidecarTauriExePath -Force

foreach ($dll in $cudaDlls) {
    Copy-Item -LiteralPath (Join-Path $cudaBin $dll) -Destination $pkgDir -Force
}

Write-Host "Packaged AhanSays CUDA desktop release at $pkgDir"
Get-ChildItem -LiteralPath $pkgDir | Select-Object Name, Length
