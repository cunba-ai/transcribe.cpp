//! Dynamic-loading entry points (feature `dynload`).
//!
//! Loads the native transcribe library at runtime instead of linking it. The
//! FFI layer (`transcribe-cpp-sys`) resolves every C entry point from the
//! loaded library; this module exposes the one-time init and a readiness
//! probe. Every other API in this crate (`Model`, `Session`, ...) behaves
//! identically in this posture.

use std::path::Path;

use crate::Result;

/// Load the native transcribe library at runtime.
///
/// `path` — optional explicit location (absolute path or bare name). `None`
/// searches the executable's directory and the OS default search path for the
/// platform's conventional name (`transcribe.dll` / `libtranscribe.so` /
/// `libtranscribe.dylib`).
///
/// Resolves every C entry point up front (fail-fast on any missing symbol)
/// and runs the pre-1.0 base-version gate. Call ONCE at startup, before any
/// model is loaded — the first call wins, and an explicit path is only honored
/// if it happens before any lazy auto-init triggered by a model call.
pub fn init_dynamic(path: Option<&Path>) -> Result<()> {
    crate::sys::transcribe_dyn_load(path).map_err(crate::Error::DynamicLoad)?;
    // Fail fast on a 0.x ABI mismatch instead of at the first model load.
    crate::version::ensure_compatible()?;
    Ok(())
}

/// Whether the native library has been loaded successfully.
pub fn is_dynamic_loaded() -> bool {
    crate::sys::transcribe_dyn_loaded()
}
