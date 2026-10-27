<#
.SYNOPSIS
  One-command end-to-end run: build the fork's transcribe.dll (self-contained,
  embedded VAD), build the transcribe-e2e CLI, compose the multitalker
  bundle, run the full check matrix.

.DESCRIPTION
  The e2e suite drives transcribe.dll exactly the way sound-rs does
  (sound-server's models.registry.json posture): the transcribe-cpp safe
  wrapper with the dynload feature, runtime dlopen, no compile-time native
  dependency in the CLI itself.

  Model files are NOT downloaded by this script. Put them in --models-dir
  (default models/e2e); every check whose model file is absent SKIPs, so the
  suite degrades gracefully. The multitalker bundle is composed on demand
  from the ASR half + the Sortformer GGUF when both are present
  (scripts/compose-multitalker-bundle.py, the upstream #219 flow).

  Source set used to develop the matrix (copy from the model store):
    whisper-tiny-Q8_0.gguf, moonshine-tiny-Q8_0.gguf,
    moonshine-streaming-tiny-Q8_0.gguf, SenseVoiceSmall-Q8_0.gguf,
    Fun-ASR-MLT-Nano-2512-Q8_0.gguf, Qwen3-ASR-0.6B-Q4_0.gguf,
    gigaam-v3-rnnt-Q8_0.gguf, parakeet-tdt-0.6b-v3-Q8_0.gguf,
    multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf,
    diar_streaming_sortformer_4spk-v2.1-F32.gguf

.EXAMPLES
  # full flow from a clean tree
  powershell -File scripts/e2e/run-e2e.ps1

  # reuse an existing DLL, only rerun the CLI matrix
  powershell -File scripts/e2e/run-e2e.ps1 -SkipDll

  # only the sortformer checks
  powershell -File scripts/e2e/run-e2e.ps1 -SkipDll -Only sf-
#>
param(
    [string]$ModelsDir = "models/e2e",
    [string]$SamplesDir = "samples",
    [string]$BuildDir = "build/e2e-dll",
    [string]$VadGguf = "build/e2e-assets/silero-vad-v6.2-F32.gguf",
    [switch]$SkipDll,
    [string[]]$Only = @()
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent | Split-Path -Parent
Set-Location $repo
Write-Host "== transcribe-e2e: repo root $repo" -ForegroundColor Cyan

# ── 1. toolchain env (MSVC + Ninja) ─────────────────────────────────────────
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio with VC tools found (vswhere)" }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
& cmd.exe /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($Matches[1])" -Value $Matches[2] }
}
# sccache breaks .rc compiles in the shared-embed config; drop just those
# entries (keep cargo itself if it lives in the same directory).
$env:PATH = ($env:PATH -split ';' | Where-Object {
    -not (Test-Path (Join-Path $_ 'sccache.exe'))
}) -join ';'

# ── 2. the self-contained DLL (sound-rs dynload posture) ───────────────────
$dll = Join-Path $BuildDir 'bin/transcribe.dll'
if (-not $SkipDll) {
    if (-not (Test-Path $VadGguf)) {
        Write-Host "== fetching embedded-VAD asset -> $VadGguf" -ForegroundColor Cyan
        New-Item -ItemType Directory -Force (Split-Path $VadGguf) | Out-Null
        # Single source of truth for the pinned digest: cmake's pin.
        $cmakeText = Get-Content cmake/TranscribeEmbedVad.cmake -Raw
        if ($cmakeText -notmatch 'TRANSCRIBE_EMBED_VAD_MODEL_SHA256\s+"([0-9a-f]{64})"') {
            throw "cannot read the pinned VAD sha256 from cmake/TranscribeEmbedVad.cmake"
        }
        $sha = $Matches[1]
        # hf-mirror first (CN network), huggingface.co fallback
        curl.exe -sfL --retry 3 --retry-delay 5 `
            "https://hf-mirror.com/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf" `
            -o $VadGguf
        if ($LASTEXITCODE -ne 0) {
            curl.exe -sfL --retry 3 `
                "https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf" `
                -o $VadGguf
            if ($LASTEXITCODE -ne 0) { throw "VAD asset download failed (mirror + hf both)" }
        }
        $hash = (Get-FileHash $VadGguf -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $sha) {
            throw "silero VAD gguf sha256 mismatch: $hash (want $sha)"
        }
    }
    Write-Host "== building transcribe.dll ($BuildDir, SHARED_EMBED + embedded VAD)" -ForegroundColor Cyan
    cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
        -DTRANSCRIBE_SHARED_EMBED=ON `
        "-DTRANSCRIBE_EMBED_VAD_MODEL_FILE=$(Resolve-Path $VadGguf)"
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    cmake --build $BuildDir --target transcribe
    if ($LASTEXITCODE -ne 0) { throw "dll build failed" }
} elseif (-not (Test-Path $dll)) {
    throw "dll not found at $dll and -SkipDll given"
}

# ── 3. compose the multitalker bundle when both halves exist ───────────────
$asr = Join-Path $ModelsDir 'multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf'
$sf  = Join-Path $ModelsDir 'diar_streaming_sortformer_4spk-v2.1-F32.gguf'
$bundle = Join-Path $ModelsDir 'multitalker-bundle-v1-Q8_0.gguf'
if ((Test-Path $asr) -and (Test-Path $sf) -and -not (Test-Path $bundle)) {
    Write-Host "== composing multitalker bundle" -ForegroundColor Cyan
    uv run --no-project --with gguf scripts/compose-multitalker-bundle.py `
        -o $bundle $asr $sf
    if ($LASTEXITCODE -ne 0) { throw "bundle compose failed" }
}

# ── 4. build + run the CLI matrix ──────────────────────────────────────────
Write-Host "== building transcribe-e2e (dynload)" -ForegroundColor Cyan
cargo build -p transcribe-e2e
if ($LASTEXITCODE -ne 0) { throw "cli build failed" }

Write-Host "== running the matrix" -ForegroundColor Cyan
$exe = "target/debug/transcribe-e2e.exe"
$cliArgs = @('--dll', $dll, '--models', $ModelsDir, '--samples', $SamplesDir, '-v')
foreach ($o in $Only) { $cliArgs += @('--only', $o) }
& $exe @cliArgs
exit $LASTEXITCODE
