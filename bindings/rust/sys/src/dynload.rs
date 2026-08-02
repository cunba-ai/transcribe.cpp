//! Runtime loader for the `dynload` posture.
//!
//! Loads the native transcribe library at runtime and resolves every C entry
//! point from [`TRANSCRIBE_DYN_SYMBOLS`](crate::transcribe_dyn::TRANSCRIBE_DYN_SYMBOLS)
//! into a raw function-pointer table. The generated trampolines in
//! `transcribe_dyn.rs` forward through it.
//!
//! Lifecycle:
//! - `transcribe_dyn_load(path)` — explicit one-time init (the safe wrapper's
//!   `init_dynamic`). Resolves all symbols eagerly: any missing symbol is a
//!   hard error at init (fail-fast) instead of a crash mid-call.
//! - Lazy auto-init: the first trampoline call with no explicit init tries the
//!   default search (exe dir + platform names). Naive consumers "just work"
//!   when the library sits next to their binary.
//! - If loading failed, every trampoline returns a type-shaped default
//!   (null / an error status) — never a null-pointer call.
//!
//! Threading: `OnceLock` + atomics. The library is loaded at most once; the
//! resolved pointers are only ever read afterwards.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, OnceLock};

/// The library handle type: on Windows `os::windows::Library` is a distinct
/// type (with `LOAD_WITH_ALTERED_SEARCH_PATH` support); everywhere else the
/// cross-platform `Library` is used directly.
#[cfg(windows)]
type NativeLibrary = libloading::os::windows::Library;
#[cfg(not(windows))]
type NativeLibrary = libloading::Library;

/// Load outcome, cached process-wide. `Err(String)` carries the reason for
/// `transcribe_dyn_load`; trampolines only need `Ok` vs `Err`.
static STATE: OnceLock<Result<Arc<NativeLibrary>, String>> = OnceLock::new();

/// Resolved entry points; index == `TRANSCRIBE_DYN_SYMBOLS` index. Allocated
/// on the first successful load.
static SLOTS: OnceLock<Box<[AtomicPtr<()>]>> = OnceLock::new();

/// Platform conventional library name.
fn default_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["transcribe.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libtranscribe.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        &["libtranscribe.so"]
    }
}

/// Search candidates in order: explicit path, exe dir + platform name, then
/// the OS search path (bare name).
fn candidates(explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        v.push(p.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for n in default_names() {
                v.push(dir.join(n));
            }
        }
    }
    for n in default_names() {
        v.push(PathBuf::from(n));
    }
    v
}

fn open_one(path: &Path) -> Result<NativeLibrary, String> {
    #[cfg(windows)]
    {
        use libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH;
        // LOAD_WITH_ALTERED_SEARCH_PATH: sibling DLLs (ggml.dll, ggml-cuda.dll,
        // ...) resolve from the library's own directory when it lives outside
        // the exe dir. Only meaningful for a qualified path — the flag's
        // behavior with a bare name is unspecified, so those use the default
        // search.
        let qualified = path.parent().map(|p| !p.as_os_str().is_empty()).unwrap_or(false);
        if qualified {
            unsafe { NativeLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }
                .map_err(|e| format!("LoadLibraryExW({}) failed: {e}", path.display()))
        } else {
            unsafe { NativeLibrary::new(path) }
                .map_err(|e| format!("LoadLibraryW({}) failed: {e}", path.display()))
        }
    }
    #[cfg(not(windows))]
    {
        unsafe { NativeLibrary::new(path) }
            .map_err(|e| format!("dlopen({}) failed: {e}", path.display()))
    }
}

/// Load the library (first candidate wins) and resolve every entry point
/// eagerly. Failure carries the last per-candidate error.
fn load(explicit: Option<&Path>) -> Result<Arc<NativeLibrary>, String> {
    let mut last_err: Option<String> = None;
    let mut lib: Option<NativeLibrary> = None;
    for c in candidates(explicit) {
        match open_one(&c) {
            Ok(l) => {
                lib = Some(l);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let lib = match lib {
        Some(l) => Arc::new(l),
        None => {
            return Err(match last_err {
                Some(e) => format!("transcribe library not found ({e})"),
                None => "transcribe library not found".to_string(),
            })
        }
    };

    let symbols = crate::transcribe_dyn::TRANSCRIBE_DYN_SYMBOLS;
    let mut slots: Vec<AtomicPtr<()>> = Vec::with_capacity(symbols.len());
    for (i, name) in symbols.iter().enumerate() {
        let sym = unsafe { lib.get::<*mut std::ffi::c_void>(name) }.map_err(|e| {
            format!(
                "symbol #{} '{}' missing from loaded library: {e}",
                i,
                String::from_utf8_lossy(name)
            )
        })?;
        // Detach the lifetime: the Arc<Library> stays alive in STATE, and the
        // raw pointer is only read through atomics afterwards. (Windows'
        // os::Symbol exposes `as_raw_ptr` directly; the cross-platform one
        // wraps it in `try_as_raw_ptr`.)
        #[cfg(windows)]
        let ptr = sym.as_raw_ptr();
        #[cfg(not(windows))]
        let ptr = sym.try_as_raw_ptr().unwrap_or(std::ptr::null_mut());
        slots.push(AtomicPtr::new(ptr as *mut ()));
    }
    let _ = SLOTS.set(slots.into_boxed_slice());
    Ok(lib)
}

/// Explicit one-time load (called by the safe wrapper's `init_dynamic`).
/// `None` runs the default search. Must be called before the first trampoline
/// use for the explicit path to win over lazy auto-init.
pub fn transcribe_dyn_load(path: Option<&Path>) -> Result<(), String> {
    match STATE.get_or_init(|| load(path)) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

/// Whether the library is currently loaded.
pub fn transcribe_dyn_loaded() -> bool {
    matches!(STATE.get(), Some(Ok(_)))
}

/// Resolve a trampoline index to a raw function pointer, lazily auto-initing
/// the default search on first use. `None` when the library is not loaded.
pub(crate) fn fn_ptr(index: usize) -> Option<*const ()> {
    let _ = STATE.get_or_init(|| load(None));
    let slots = SLOTS.get()?;
    let p = slots.get(index)?.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        Some(p as *const ())
    }
}
