//! Voice Activity Detection (VAD).
//!
//! VAD is an OPTIONAL preprocessing step before the family decoder: it slices
//! the input PCM into speech-bounded windows so long / sparse audio decodes
//! faster and avoids single-decode context exhaustion. It requires the native
//! library built with `-DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON` AND `audiocpp.dll`
//! (or `libaudiocpp.so`) loadable at runtime; otherwise it silently degrades
//! to the full-buffer decode with a `WARN` log.
//!
//! Two surfaces:
//!   - [`VadMode`] / [`VadOptions`] embedded in [`crate::RunOptions`] to gate
//!     the per-run VAD chunk loop.
//!   - [`detect_speech`] — a standalone VAD that returns speech segments
//!     without running ASR (wraps the C `transcribe_vad`).
//!
//! Streaming runs (`Session::stream_*`) never use VAD.

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use crate::error::check;
use crate::sys;

/// VAD algorithm selection. Mirrors the C `transcribe_vad_mode`.
///
/// `Off` is the default (full-buffer decode, no VAD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadMode {
    /// Full-buffer decode, no VAD (the default).
    #[default]
    Off,
    /// Neural VAD (Silero) via audiocpp.dll — needs the embedded model.
    Silero,
    /// Energy/RMS VAD via audiocpp.dll — no model needed.
    Energy,
}

impl VadMode {
    fn to_c(self) -> sys::transcribe_vad_mode {
        match self {
            VadMode::Off => sys::transcribe_vad_mode::TRANSCRIBE_VAD_OFF,
            VadMode::Silero => sys::transcribe_vad_mode::TRANSCRIBE_VAD_SILERO,
            VadMode::Energy => sys::transcribe_vad_mode::TRANSCRIBE_VAD_ENERGY,
        }
    }
}

/// audiocpp compute backend forwarded to the VAD model inside the dll. The dll
/// ships CPU+CUDA/ROCm/SYCL/Vulkan; pick via this field, not by swapping dlls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadBackend {
    /// CPU (always available). The default.
    #[default]
    Cpu,
    /// NVIDIA CUDA (also matches AMD ROCm/HIP and MUSA builds).
    Cuda,
    /// Vulkan (NVIDIA / AMD / Intel / Apple-over-MoltenVK).
    Vulkan,
    /// Apple Metal (macOS/iOS only).
    Metal,
    /// Intel oneAPI SYCL.
    Sycl,
}

impl VadBackend {
    fn to_c(self) -> c_int {
        // AUDIOCPP_BACKEND_* values (ref/cuda-release/audiocpp.h:77-81).
        match self {
            VadBackend::Cpu => 0,
            VadBackend::Cuda => 1,
            VadBackend::Vulkan => 2,
            VadBackend::Metal => 3,
            VadBackend::Sycl => 4,
        }
    }
}

/// VAD configuration. Embedded in [`crate::RunOptions`] (per-run chunk loop)
/// and accepted by [`detect_speech`] (standalone). All fields optional —
/// `Default` gives Silero on CPU with family-default chunk sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct VadOptions {
    /// Algorithm. `Off` disables VAD (the [`crate::RunOptions`] default).
    pub mode: VadMode,
    /// Path to audiocpp.dll/libaudiocpp.so. `None` → discovery order
    /// (`TRANSCRIBE_VAD_DLL` env, exe dir, cwd, system PATH).
    pub dll_path: Option<String>,
    /// Compute backend for the VAD model inside audiocpp.
    pub backend: VadBackend,
    /// GPU device index; ignored on CPU.
    pub device_id: i32,
    /// CPU threads; 0 = auto.
    pub n_threads: i32,
    /// Per-window ceiling in ms. `None` → family `effective_max_audio_ms`
    /// (or 30000 if unbounded). Cap this for encoder-decoder families whose
    /// generation budget is small (e.g. Qwen3-ASR).
    pub max_chunk_ms: Option<i64>,
    /// Merge speech segments separated by less than this (default 500ms).
    pub merge_gap_ms: Option<i64>,
    /// Padding added to each window's sides (default 250ms).
    pub padding_ms: Option<i64>,
    /// Silero speech-probability threshold (default 0.5).
    pub silero_threshold: Option<f32>,
    /// Silero min speech duration (default 250ms).
    pub silero_min_speech_ms: Option<i64>,
    /// Silero min silence to end a segment (default 100ms).
    pub silero_min_silence_ms: Option<i64>,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            mode: VadMode::Silero,
            dll_path: None,
            backend: VadBackend::Cpu,
            device_id: 0,
            n_threads: 0,
            max_chunk_ms: None,
            merge_gap_ms: None,
            padding_ms: None,
            silero_threshold: None,
            silero_min_speech_ms: None,
            silero_min_silence_ms: None,
        }
    }
}

impl VadOptions {
    /// Build the C params struct. The returned `CString`s for `dll_path` must
    /// outlive the C call; the caller keeps them alive (RunOptions stores the
    /// VadOptions, detect_speech scopes them locally).
    pub(crate) fn to_c(&self) -> (sys::transcribe_vad_params, Option<CString>) {
        let dll = self.dll_path.as_ref().and_then(|p| CString::new(p.as_str()).ok());
        let mut p = sys::transcribe_vad_params {
            struct_size: std::mem::size_of::<sys::transcribe_vad_params>() as u64,
            mode: self.mode.to_c(),
            dll_path: dll.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            weight_path: ptr::null(), // reserved; embedded dll ignores it
            backend: self.backend.to_c(),
            device_id: self.device_id,
            n_threads: self.n_threads,
            max_chunk_ms: self.max_chunk_ms.unwrap_or(0),
            merge_gap_ms: self.merge_gap_ms.unwrap_or(0),
            padding_ms: self.padding_ms.unwrap_or(0),
            silero_threshold: self.silero_threshold.unwrap_or(0.0),
            silero_min_speech_ms: self.silero_min_speech_ms.unwrap_or(0),
            silero_min_silence_ms: self.silero_min_silence_ms.unwrap_or(0),
        };
        // 0 in the C struct means "use default" everywhere; the C side reads
        // <=0 / 0.0 as "default". Zero-init matches that for the Option::None
        // fields, but be explicit for clarity on the signed/float defaults.
        if p.merge_gap_ms == 0 {
            p.merge_gap_ms = 0; // C treats 0 as "use default 500ms"
        }
        (p, dll)
    }
}

/// One speech segment, ms-resolution. Returned by [`detect_speech`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSegment {
    /// Segment start, milliseconds from audio start.
    pub start_ms: i64,
    /// Segment end (exclusive), milliseconds from audio start.
    pub end_ms: i64,
    /// Detector confidence in `[0.0, 1.0]`. Energy VAD reports `1.0`.
    pub confidence: f32,
}

impl From<sys::transcribe_vad_segment> for VadSegment {
    fn from(s: sys::transcribe_vad_segment) -> Self {
        VadSegment {
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            confidence: s.confidence,
        }
    }
}

/// Standalone VAD: detect speech segments in `pcm` without running ASR.
///
/// `pcm` is 16 kHz mono f32 in `[-1.0, 1.0]`. `sample_rate` must be 16000.
/// Returns the detected segments on success.
///
/// Requires the native library built with
/// `-DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON` AND audiocpp loadable at runtime;
/// otherwise returns [`crate::Error`] (backend unavailable). For the per-run
/// VAD chunk loop (VAD + ASR in one call), set [`crate::RunOptions::vad`]
/// instead.
///
/// # Errors
///
/// - [`crate::Error`] from the C status (e.g. `Backend` if the dll is missing,
///   `SampleRate` if `sample_rate != 16000`, `InvalidArg` on null/empty input).
pub fn detect_speech(pcm: &[f32], sample_rate: u32, options: &VadOptions) -> crate::Result<Vec<VadSegment>> {
    if pcm.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "transcribe_vad: empty pcm".into(),
        ));
    }
    if sample_rate != 16000 {
        return Err(crate::Error::InvalidArgument(format!(
            "transcribe_vad: sample_rate must be 16000, got {sample_rate}"
        )));
    }
    let (params, _dll) = options.to_c();
    let mut segments_ptr: *mut sys::transcribe_vad_segment = ptr::null_mut();
    let mut n: i64 = 0;
    // SAFETY: pcm outlives the call; params + its CString (_dll) outlive too.
    // The C contract guarantees *out_segments == NULL on every error return.
    let status = unsafe {
        sys::transcribe_vad(
            pcm.as_ptr(),
            pcm.len() as i32,
            sample_rate as i32,
            &params,
            &mut segments_ptr,
            &mut n,
        )
    };
    // Map status first; if non-OK the array is NULL and free is a no-op.
    if let Err(e) = check(status, "transcribe_vad") {
        // Still call free for safety (NULL-safe per contract).
        unsafe { sys::transcribe_free_vad(segments_ptr) };
        return Err(e);
    }
    if segments_ptr.is_null() || n <= 0 {
        return Ok(Vec::new());
    }
    // Copy out into owned VadSegments BEFORE freeing the C array.
    let segments = unsafe {
        let slice = std::slice::from_raw_parts(segments_ptr, n as usize);
        slice.iter().copied().map(VadSegment::from).collect::<Vec<_>>()
    };
    unsafe { sys::transcribe_free_vad(segments_ptr) };
    Ok(segments)
}
