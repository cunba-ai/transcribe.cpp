<#
.SYNOPSIS
  本地构建 transcribe.cpp 的 shared library + install tree,产出可用于
  TRANSCRIBE_PREBUILT_PREFIX 的目录(含 lib/transcribe-link.json + bin/*.dll +
  include/),供 sound-server 的 prebuilt 模式消费。

.DESCRIPTION
  参考 audio.cpp/scripts/build_windows.ps1 的 MSVC 环境探测逻辑,但聚焦单一
  目标:Windows + CUDA(本地开发场景)。不做 preset 矩阵,不做 OpenMP/ccache
  等可选特性 —— 产出最小可用的 install tree 即可。

  构建产物在 build/windows-install/(即 CMAKE_INSTALL_PREFIX)。构建完成后
  脚本会打印如何用它构建 sound-server:
    TRANSCRIBE_PREBUILT_PREFIX=<repo>\build\windows-install cargo build

.PARAMETER CudaArchitectures
  CUDA 架构,默认 "auto"(探测本机 GPU)。可显式指定如 "75;80;86;89;90"。

.PARAMETER Clean
  构建前清理 build 目录。

.PARAMETER ConfigureOnly
  只 configure,不 build/install。

.EXAMPLE
  .\scripts\build_windows.ps1
  .\scripts\build_windows.ps1 -CudaArchitectures "86"
  .\scripts\build_windows.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string]$CudaArchitectures = "auto",
    [switch]$Clean,
    [switch]$ConfigureOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# === Helper functions (adapted from audio.cpp/scripts/build_windows.ps1) ===

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )
    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -replace "\\", "/")
}

function Add-PathFront {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ((Test-Path $Path) -and (($env:PATH -split [IO.Path]::PathSeparator) -notcontains $Path)) {
        $env:PATH = ($Path, $env:PATH) -join [IO.Path]::PathSeparator
    }
}

function Find-FirstFile {
    param([Parameter(Mandatory = $true)][string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($null -ne $found) { return $found.FullName }
    }
    return ""
}

function Find-CudaRoot {
    foreach ($root in @($env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        if ($root -and (Test-Path (Join-Path $root "bin\nvcc.exe"))) {
            return (Resolve-Path $root).Path
        }
    }
    $nvcc = Find-FirstFile @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
        "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"
    )
    if ($nvcc -ne "") { return (Resolve-Path (Join-Path (Split-Path $nvcc -Parent) "..")).Path }
    return ""
}

function Test-VsInstall {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -ne "" -and (Test-Path (Join-Path $Path "VC\Tools\MSVC")))
}

function Find-VsInstall {
    $candidates = @()
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    foreach ($root in $roots) {
        foreach ($year in @("2026", "2022", "2019")) {
            foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
                $candidates += (Join-Path $root "Microsoft Visual Studio\$year\$edition")
            }
        }
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $found) { $candidates += $found }
    }
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-VsInstall $candidate) { return (Resolve-Path $candidate).Path }
    }
    return ""
}

function Add-MsvcEnvironment {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    $vcvars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvars) {
        $cmd = "`"$vcvars`" >nul && set"
        foreach ($line in (& cmd.exe /d /s /c $cmd)) {
            if ($line -match "^([^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
    }
}

function Resolve-CudaArchitectures {
    if ($CudaArchitectures -ne "" -and $CudaArchitectures -ne "auto") {
        return $CudaArchitectures
    }
    $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $smi) { return "75;80;86;89;90" }
    $cap = (& $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    if ($cap -notmatch "^(\d+)\.(\d+)") { return "75;80;86;89;90" }
    return "$($Matches[1])$($Matches[2])"
}

# === Main flow =============================================================

# 1. Find CUDA toolkit
$cudaRoot = Find-CudaRoot
if ($cudaRoot -eq "") {
    throw "Official CUDA Toolkit was not found. Install it so nvcc exists under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin, or set CUDA_PATH."
}
Add-PathFront (Join-Path $cudaRoot "bin")
$env:CUDA_PATH = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot

# 2. Find MSVC toolchain
$vsInstall = Find-VsInstall
if ($vsInstall -eq "") {
    throw "Visual Studio Build Tools C++ workload was not found. Install Build Tools 2022+ with MSVC, Windows SDK, CMake/Ninja."
}
Add-MsvcEnvironment $vsInstall

# 3. 找 cmake / ninja(VS 自带或 PATH)
$cmake = (Get-Command "cmake.exe" -ErrorAction SilentlyContinue).Source
$ninja = (Get-Command "ninja.exe" -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    $cmake = Find-FirstFile @("$vsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
}
if (-not $ninja) {
    $ninja = Find-FirstFile @("$vsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
}
if (-not $cmake -or -not $ninja) {
    throw "cmake.exe or ninja.exe not found. Install CMake/Ninja or ensure VS C++ workload includes them."
}

$arch = Resolve-CudaArchitectures
$sourceDir = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path $sourceDir "build\windows-cuda"
$installPrefix = Join-Path $sourceDir "build\windows-install"

Write-Host "CUDA: $cudaRoot (archs: $arch)"
Write-Host "Visual Studio: $vsInstall"
Write-Host "CMake: $cmake"
Write-Host "Ninja: $ninja"
Write-Host "Source: $sourceDir"
Write-Host "Build:  $buildDir"
Write-Host "Install prefix: $installPrefix"

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning $buildDir"
    Remove-Item -Recurse -Force $buildDir
}

# 4. Configure
$configureArgs = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninja)",
    "-DCMAKE_C_FLAGS=/utf-8",
    "-DCMAKE_CXX_FLAGS=/utf-8 /EHsc",
    "-DTRANSCRIBE_SHARED_EMBED=ON",
    "-DTRANSCRIBE_INSTALL=ON",
    "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $installPrefix)",
    "-DTRANSCRIBE_BUILD_TESTS=OFF",
    "-DTRANSCRIBE_BUILD_EXAMPLES=OFF",
    "-DTRANSCRIBE_BUILD_TOOLS=OFF",
    "-DTRANSCRIBE_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$arch",
    "-DCUDAToolkit_ROOT=$(Convert-ToCMakePath $cudaRoot)"
)
Invoke-Checked $cmake $configureArgs

if ($ConfigureOnly) { exit 0 }

# 5. Build + Install
$jobs = [Math]::Max(2, [Environment]::ProcessorCount)
Invoke-Checked $cmake @("--build", $buildDir, "-j", $jobs.ToString(), "--target", "transcribe")
Invoke-Checked $cmake @("--install", $buildDir)

# 6. 验证 install tree 含 manifest(prebuilt 模式的关键依赖)
$manifest = Join-Path $installPrefix "lib\transcribe-link.json"
if (-not (Test-Path $manifest)) {
    throw "transcribe-link.json missing from $installPrefix\lib — prebuilt mode will fail. Check that TRANSCRIBE_INSTALL=ON took effect."
}

# 7. 列出产物 + 打印使用说明
Write-Host ""
Write-Host "=== Install tree contents ==="
Get-ChildItem -Path $installPrefix -Recurse | Where-Object { -not $_.PSIsContainer } | ForEach-Object { Write-Host "  $($_.FullName.Replace($installPrefix, '<install>'))" }
Write-Host ""
Write-Host "=== Manifest (transcribe-link.json) ==="
Get-Content $manifest
Write-Host ""
Write-Host "=== Done. Use it to build sound-server (prebuilt mode): ==="
Write-Host "  TRANSCRIBE_PREBUILT_PREFIX=`"$installPrefix`" cargo build  (in sound-server/)"
Write-Host "  # Runtime: copy build\windows-install\bin\*.dll next to sound-server.exe"
