//! Workspace automation for the Rust bindings.
//!
//! `cargo xtask bindgen`          regenerate the committed FFI bindings
//! `cargo xtask bindgen --check`  fail if the committed bindings are stale
//!
//! The committed output lives at `bindings/rust/sys/src/transcribe_sys.rs`.
//! It is generated from `include/transcribe/extensions.h` (the flattened
//! public surface) and pinned to `include/transcribe.abihash`: the generated
//! header embeds the hash, so ANY public-ABI change moves either the bindgen
//! output or the embedded hash, and the `--check` diff goes red. This is the
//! Rust arm of the cross-binding drift gate (notes/bindings-requirements.md §2).
//!
//! The `dynload` variant lives at `bindings/rust/sys/src/transcribe_dyn.rs`:
//! the same types with the C entry points replaced by same-named trampolines
//! that forward through the runtime-loaded library (feature `dynload`). Both
//! files are generated from the SAME bindgen pass, so the two cannot drift.
//!
//! Comments are disabled and a fixed enum style is used so the output is
//! deterministic across libclang versions; rustfmt formats it for review.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const GENERATED: &str = "bindings/rust/sys/src/transcribe_sys.rs";
const GENERATED_DYN: &str = "bindings/rust/sys/src/transcribe_dyn.rs";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.iter().any(|a| a == "--check");
    match args.first().map(String::as_str) {
        Some("bindgen") => run_bindgen(check),
        _ => {
            eprintln!("usage: cargo xtask bindgen [--check]");
            ExitCode::from(2)
        }
    }
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/bindings/rust/xtask
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root above bindings/rust/xtask")
        .to_path_buf()
}

fn run_bindgen(check: bool) -> ExitCode {
    let root = repo_root();
    let (fresh_sys, fresh_dyn) = generate(&root);
    let target_sys = root.join(GENERATED);
    let target_dyn = root.join(GENERATED_DYN);
    let committed_sys = std::fs::read_to_string(&target_sys).unwrap_or_default();

    // The dyn file is derived from a sys text. Use the FRESH text when it
    // matches the committed one (the normal, same-environment case); when they
    // differ — header drift OR a different libclang resolving enum underlying
    // types differently (c_int vs c_uint across machines) — derive from the
    // COMMITTED text so the two committed bindings stay in the same world and
    // the CI drift check passes everywhere. The sys.rs diff itself is
    // reviewed/committed separately; re-run after committing it to refresh
    // the dyn file.
    let sys_for_dyn = if fresh_sys == committed_sys {
        fresh_sys.clone()
    } else {
        eprintln!(
            "xtask: warning — regenerated transcribe_sys.rs differs from the committed file \
             (header drift, or this machine's libclang resolves enum underlying types \
             differently). transcribe_dyn.rs is derived from the COMMITTED file so the pair \
             stays consistent; commit the transcribe_sys.rs diff first, then re-run \
             `cargo xtask bindgen` to refresh transcribe_dyn.rs."
        );
        committed_sys
    };
    let dyn_text = derive_dyn(&sys_for_dyn);
    let targets = [
        (GENERATED, fresh_sys),
        (GENERATED_DYN, dyn_text),
    ];

    let mut ok = true;
    for (name, text) in &targets {
        let target = root.join(name);
        if check {
            let committed = std::fs::read_to_string(&target).unwrap_or_default();
            if committed != *text {
                eprintln!(
                    "xtask: {name} is STALE.\n\
                     The public header or its ABI digest changed. Regenerate with:\n\
                     \n    cargo xtask bindgen\n\n\
                     and review + commit the result."
                );
                ok = false;
            } else {
                println!("xtask: {name} is up to date");
            }
        } else {
            std::fs::write(&target, text).expect("write generated bindings");
            println!("xtask: wrote {name}");
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// One C entry point from the bindgen output.
struct FnSig {
    name: String,
    /// (param name, param type) — types as bindgen spelled them.
    params: Vec<(String, String)>,
    /// Return type as bindgen spelled it ("()" for void).
    ret: String,
}

fn generate(root: &Path) -> (String, String) {
    let include = root.join("include");
    let header = include.join("transcribe/extensions.h");
    let abihash = std::fs::read_to_string(include.join("transcribe.abihash"))
        .expect("read include/transcribe.abihash");
    let abihash = abihash.trim();

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        // Determinism: comment extraction varies by libclang version, so the
        // committed file would drift across machines if we kept doc comments.
        .generate_comments(false)
        // FFI-safe enums: a NewType is a transparent integer wrapper, so an
        // out-of-range value from C is never UB (the safe crate validates at
        // the public boundary — the "never trust transmute" rule).
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .prepend_enum_name(false)
        // Only emit declarations from our own headers (skip stdint/stddef).
        // The path class matches both `/` (Linux/macOS CI) and `\` (Windows)
        // separators so regeneration is deterministic across platforms.
        .allowlist_file(r".*include[/\\]transcribe\.h")
        .allowlist_file(r".*include[/\\]transcribe[/\\].*\.h")
        // Version macros are deliberately NOT emitted: a version-only bump must
        // not churn the committed bindings or the abihash (notes/releasing.md
        // §8 P0 #1). The runtime version comes from CARGO_PKG_VERSION instead
        // (transcribe-cpp/src/version.rs), matching the Python/TS generator,
        // which drops these from both its output and the hashed digest.
        .blocklist_item(r"TRANSCRIBE_VERSION_(MAJOR|MINOR|PATCH|NUMBER)")
        // Compile-time layout assertions (free belt-and-suspenders; the
        // per-field check is otherwise waived for bindgen).
        .layout_tests(true)
        .generate()
        .expect("bindgen generate");

    let banner = format!(
        "// @generated by `cargo xtask bindgen` from include/transcribe/extensions.h\n\
         // DO NOT EDIT BY HAND. Regenerate: `cargo xtask bindgen`.\n\
         // Pinned to include/transcribe.abihash = {abihash}\n\
         \n\
         /// The public-ABI digest these bindings were generated against\n\
         /// (sha256/16 over the normalized FFI surface). The load-time version\n\
         /// gate and the CI drift check both anchor on this value.\n\
         pub const PUBLIC_HEADER_HASH: &str = \"{abihash}\";\n\
         \n"
    );
    let bindings_text = format!("{banner}{bindings}");
    // Normalize to LF: the committed files are LF, and Windows writes can
    // otherwise leak CRLF into the diff.
    let bindings_text = bindings_text.replace("\r\n", "\n");
    let dyn_text = derive_dyn(&bindings_text);
    (bindings_text, dyn_text)
}

/// Derive the `dynload` variant from a `transcribe_sys.rs` text: same banner
/// and types, but every `unsafe extern "C" { ... }` block (one fn each)
/// replaced by a runtime trampoline.
fn derive_dyn(sys_text: &str) -> String {
    let marker = "/* automatically generated by rust-bindgen";
    let idx = sys_text
        .find(marker)
        .expect("bindgen output marker in sys text");
    let banner = &sys_text[..idx];
    let body = &sys_text[idx..];

    let (types_only, fns) = split_functions(body);
    let dyn_section = generate_trampolines(&fns);
    format!(
        "{banner}{types_only}\n\
         // ─── dynload trampolines (feature `dynload`) ───────────────────────────\n\
         // Each C entry point is shadowed by a same-named Rust trampoline that\n\
         // forwards through the runtime-loaded library (crate::dynload). The\n\
         // index in TRANSCRIBE_DYN_SYMBOLS is the trampoline index: the loader\n\
         // resolves every name at init and fails fast on any missing symbol.\n\
         // When the library is not loaded a type-shaped default is returned\n\
         // (null / an error status) instead of calling through a null pointer.\n\
         \n{dyn_section}"
    )
}

/// Split the bindgen text into (types-only text, parsed C entry points).
///
/// The generated file declares every entry point in its own
/// `unsafe extern "C" { pub fn ...; }` block (bindgen's per-item blocks), so
/// each block is self-contained: strip the blocks for the types-only file and
/// parse them for the trampoline signatures.
fn split_functions(text: &str) -> (String, Vec<FnSig>) {
    const BLOCK: &str = "unsafe extern \"C\" {";
    let mut types_only = String::with_capacity(text.len());
    let mut fns = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(BLOCK) {
        types_only.push_str(&rest[..pos]);
        let inner_start = pos + BLOCK.len();
        let brace_end = rest[inner_start..]
            .find('}')
            .map(|i| inner_start + i)
            .expect("extern block has a closing brace");
        fns.push(parse_fn(&rest[inner_start..brace_end]));
        rest = &rest[brace_end + 1..]; // skip the block incl. its closing brace
    }
    types_only.push_str(rest);
    (types_only, fns)
}

/// Parse one `pub fn name(params) -> ret;` declaration body.
fn parse_fn(inner: &str) -> FnSig {
    let t = inner
        .trim()
        .strip_prefix("pub fn")
        .expect("extern block contains a `pub fn`");
    let t = t.trim_start();
    let name_end = t
        .find('(')
        .expect("fn name is followed by '('");
    let name = t[..name_end].trim().to_string();
    let rest = &t[name_end..];

    let (params_text, rest) = take_balanced(rest, '(', ')');
    let params = split_params(params_text)
        .into_iter()
        .map(|p| {
            let sep = first_top_level_colon(&p)
                .unwrap_or_else(|| panic!("param `{p}` has no name: type separator"));
            (p[..sep].trim().to_string(), p[sep + 1..].trim().to_string())
        })
        .collect();

    let rest = rest.trim_start();
    let ret = if let Some(r) = rest.strip_prefix("->") {
        r.trim()
            .strip_suffix(';')
            .expect("return type ends with ';'")
            .trim()
            .to_string()
    } else {
        // No return type declared — the fn returns ().
        assert!(rest.trim_end().strip_suffix(';').unwrap_or("").trim().is_empty(), "unexpected fn tail: {rest:?}");
        "()".to_string()
    };

    FnSig { name, params, ret }
}

/// Split a paren-delimited parameter list on top-level commas (bindgen emits
/// a trailing comma, which would otherwise yield a phantom empty param).
fn split_params(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(text[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        out.push(last.to_string());
    }
    out
}

/// Index of the first `:` at paren depth 0 (the `name: type` separator).
fn first_top_level_colon(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            ':' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// From `text` starting at an opening `open`, return (contents, rest after the
/// matching close).
fn take_balanced(text: &str, open: char, close: char) -> (&str, &str) {
    let rest = text.strip_prefix(open).expect("starts with opening paren");
    let mut depth = 1usize;
    for (i, ch) in rest.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return (&rest[..i], &rest[i + 1..]);
            }
        }
    }
    panic!("unbalanced `{open}`/`{close}` in: {text}")
}

/// The value a trampoline returns when the library is not loaded, by return
/// type shape. Unknown shapes panic loudly so the drift gate catches them.
fn not_loaded_default(ret: &str) -> &'static str {
    match ret {
        "()" => "()",
        "bool" => "false",
        "usize" => "0",
        "::std::os::raw::c_int" => "-1",
        "transcribe_status" => "transcribe_status(8)", // TRANSCRIBE_ERR_BACKEND
        "transcribe_timestamp_kind" => "transcribe_timestamp_kind(0)",
        "transcribe_stream_state" => "transcribe_stream_state(0)",
        s if s.starts_with("*const ") => "::std::ptr::null()",
        s if s.starts_with("*mut ") => "::std::ptr::null_mut()",
        other => panic!(
            "xtask: unhandled return type `{other}` — add a not-loaded default to \
             transcribe_dyn generation"
        ),
    }
}

/// The whole dynload section: the symbol-name table (index == trampoline
/// index) followed by one trampoline per entry point.
fn generate_trampolines(fns: &[FnSig]) -> String {
    let mut out = String::new();
    out.push_str("pub(crate) const TRANSCRIBE_DYN_SYMBOLS: &[&[u8]] = &[\n");
    for f in fns {
        out.push_str(&format!("    b\"{}\",\n", f.name));
    }
    out.push_str("];\n\n");
    for (i, f) in fns.iter().enumerate() {
        out.push_str(&generate_trampoline(i, f));
        out.push('\n');
    }
    out
}

fn generate_trampoline(index: usize, f: &FnSig) -> String {
    // Locals use a reserved-style prefix (double underscore, which C headers
    // cannot use) so they can never shadow a real parameter name; any residual
    // collision would fail the generated code's compile, caught by the CI
    // drift check's build lane.
    let params = f
        .params
        .iter()
        .map(|(n, t)| format!("{n}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arg_names = f
        .params
        .iter()
        .map(|(n, _)| n.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let ptr_ty = format!(
        "unsafe extern \"C\" fn({}) -> {}",
        f.params
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        f.ret
    );
    let default = not_loaded_default(&f.ret);
    format!(
        "/// Runtime trampoline for `{name}` — index {index} in\n\
         /// [`TRANSCRIBE_DYN_SYMBOLS`].\n\
         #[no_mangle]\n\
         pub unsafe extern \"C\" fn {name}({params}) -> {ret} {{\n\
         \x20   match crate::dynload::fn_ptr({index}) {{\n\
         \x20       Some(__ptr) => {{\n\
         \x20           let __f: {ptr_ty} = unsafe {{ ::std::mem::transmute(__ptr) }};\n\
         \x20           unsafe {{ __f({arg_names}) }}\n\
         \x20       }}\n\
         \x20       None => {default},\n\
         \x20   }}\n\
         }}\n",
        name = f.name,
        ret = f.ret,
    )
}
