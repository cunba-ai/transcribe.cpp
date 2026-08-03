//! Progress reporting for in-flight runs.
//!
//! Install a [`ProgressCallback`] on a session to receive chunk-level progress
//! during `run` (VAD chunk loop today; per-family internals in a future phase).
//! The callback receives a completion fraction, a stage label, and completed /
//! total chunk counts; returning `ProgressAction::Cancel` aborts the run at the
//! next chunk boundary (the call then returns [`crate::Error::Aborted`] with
//! the partial transcript, mirroring [`crate::CancelToken`]).
//!
//! Semantics match the C `transcribe_progress_callback` (and audio.cpp's
//! `audiocpp_progress_fn`): non-zero return cancels. Streaming runs
//! (`Session::stream_*`) do NOT fire this callback — streaming progress stays
//! pull-based via [`crate::StreamUpdate`].
//!
//! # Threading
//!
//! The callback fires synchronously on the thread running `run`. It must be
//! cheap and thread-safe (the C contract). Install before `run`, not
//! concurrently with one.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex};

/// What a progress callback returns: keep going, or cancel the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// Continue the run.
    Continue,
    /// Abort at the next chunk boundary; partial transcript is preserved and
    /// the run returns [`crate::Error::Aborted`].
    Cancel,
}

impl ProgressAction {
    #[inline]
    fn to_c(self) -> c_int {
        match self {
            ProgressAction::Continue => 0,
            ProgressAction::Cancel => 1,
        }
    }
}

/// One progress snapshot handed to the callback.
#[derive(Debug, Clone)]
pub struct Progress {
    /// Completion fraction in `[0.0, 1.0]`.
    pub fraction: f32,
    /// Short stage label valid for the duration of the call (e.g. `"asr+whisper"`).
    /// Copied to an owned `String` so the callback can outlive the C pointer.
    pub stage: String,
    /// Chunks completed so far (`0..=total`).
    pub completed: i64,
    /// Total chunks (>= 1).
    pub total: i64,
}

/// Type alias for the user callback. Receives a [`Progress`] snapshot; returns
/// [`ProgressAction::Cancel`] to abort.
pub type ProgressFn = dyn Fn(Progress) -> ProgressAction + Send + Sync + 'static;

/// Internal storage type name re-exported for Session's field. The callback
/// box lives behind an `Arc<Mutex<...>>` so the C trampoline can reach it from
/// the run thread.
pub(crate) type ProgressCallbackCb = Arc<Mutex<Box<ProgressFn>>>;

/// A handle wrapping a user progress callback. Install it on a session via
/// [`crate::Session::set_progress_callback`].
///
/// Clones share the same callback slot.
#[derive(Clone)]
pub struct ProgressCallback {
    pub(crate) cb: ProgressCallbackCb,
}

impl std::fmt::Debug for ProgressCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressCallback").finish_non_exhaustive()
    }
}

impl ProgressCallback {
    /// Wrap a closure as a progress callback.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(Progress) -> ProgressAction + Send + Sync + 'static,
    {
        ProgressCallback {
            cb: Arc::new(Mutex::new(Box::new(f))),
        }
    }
}

/// The C progress trampoline. `user_data` is the `*const Mutex<Box<ProgressFn>>`
/// pointee of the `Arc` the session retains (installed via `Arc::as_ptr`, which
/// yields a pointer to the Arc's *inner* data, not to the Arc handle). Invoked
/// on the run thread between chunks.
///
/// # SAFETY contract
///
/// `user_data` must point at a live `Mutex<Box<ProgressFn>>` owned by an
/// `Arc` the session retains until the callback is cleared. The session
/// enforces this by storing the `Arc` clone in `Session::progress`.
pub(crate) extern "C" fn progress_trampoline(
    progress: f32,
    stage: *const c_char,
    completed_units: i64,
    total_units: i64,
    user_data: *mut c_void,
) -> c_int {
    if user_data.is_null() {
        return 0;
    }
    // SAFETY: `user_data` is the `*const Mutex<Box<ProgressFn>>` produced by
    // `Arc::as_ptr` in `Session::set_progress_callback`. The session holds the
    // owning `Arc` clone for as long as the callback is set, keeping this
    // pointee live. `Arc::as_ptr` returns a pointer to the inner data (the
    // `Mutex`), NOT to the `Arc` handle — casting to `*const Mutex<...>` (not
    // `*const Arc<...>`) matches the install site.
    let mutex = unsafe { &*(user_data as *const Mutex<Box<ProgressFn>>) };
    let stage = if stage.is_null() {
        String::new()
    } else {
        // SAFETY: C side documents the pointer as a valid UTF-8 CStr for the
        // duration of the call (the library's own stage strings are literals).
        unsafe { CStr::from_ptr(stage) }.to_string_lossy().into_owned()
    };
    let snapshot = Progress {
        fraction: progress,
        stage,
        completed: completed_units,
        total: total_units,
    };
    let action = match mutex.lock() {
        Ok(guard) => (guard)(snapshot),
        // A poisoned mutex means a prior callback panicked; treat as cancel so
        // the run stops rather than spinning on a broken callback.
        Err(_) => ProgressAction::Cancel,
    };
    action.to_c()
}
