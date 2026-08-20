# transcribe-cpp-sys

Raw native FFI bindings for
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), a C/C++
speech-to-text library built on ggml.

> **Status: in development (0.2.0).** This crate exposes the unsafe, generated
> FFI surface. Most users want the safe wrapper,
> [`transcribe-cpp`](https://crates.io/crates/transcribe-cpp).

Raw-FFI consumers upgrading from 0.1 should follow the
[0.2 migration guide](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/migrating-to-0.2.md).

## What it does

`build.rs` has two link paths, both feeding the same `emit_link_lines()` so
downstream consumers (the `transcribe-cpp` wrapper, and crates depending on
it) see identical `DEP_TRANSCRIBE_*` metadata either way:

1. **Source build (default).** Compiles the vendored C++ tree via CMake (the
   crate tarball carries the whole tree) and reconstructs the link line from
   the installed `transcribe-link.json` manifest — no hardcoded per-platform
   link lists. Requires a C++ toolchain + CMake (and, for GPU backends, the
   matching CUDA/HIP/OneAPI toolkit).
2. **Prebuilt (set `TRANSCRIBE_PREBUILT_PREFIX`).** Reads
   `transcribe-link.json` straight from a prebuilt CMake install tree and
   emits the link directives, skipping the CMake build entirely. **No GPU
   toolkit is needed** — the backend code is already compiled into the
   prebuilt library. This is how CI-built shared libraries are consumed. See
   [Using a prebuilt library](#using-a-prebuilt-library) below.

The committed bindgen output means **libclang is not needed** to build this
crate in either path.

## Build prerequisites

A C++ toolchain and **CMake**. There is no external compression dependency —
the deflate codec (miniz) is vendored into the library, so no system zlib /
vcpkg setup is required on any platform. The static link is the default; the
`shared` feature links a shared library instead.

## Features

- `metal` (default on Apple), `vulkan`, `cuda`, `rocm`, `openmp` — each forwards
  to the matching `TRANSCRIBE_*` CMake option (`rocm` enables `TRANSCRIBE_HIP`).
- `shared` — link a shared `libtranscribe` (`.so`/`.dylib`/`.dll`) loaded at
  runtime instead of statically baking it in. The default is a self-contained
  static link.
- `dynamic-backends` — additionally ship each compute backend (the per-ISA CPU
  tiers, Vulkan, CUDA, ROCm, …) as a loadable module next to the library, selected at
  runtime by `transcribe_init_backends_default()` when the modules sit next to
  `libtranscribe`, or `transcribe_init_backends(dir)` for a custom provider
  directory. Implies `shared`.

## ROCm builds

Install ROCm 6.1 or newer, then enable the first-class `rocm` feature:

```sh
cargo build --no-default-features --features rocm
```

The build detects the attached AMD GPU. To target a specific architecture, pass
it through CMake, for example
`TRANSCRIBE_CMAKE_ARGS="-DAMDGPU_TARGETS=gfx1201"`.

## Windows Vulkan builds

The `vulkan` feature requires the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) on Windows. Once the
SDK is installed and a new terminal sees `VULKAN_SDK`, build normally:

```powershell
cargo build --features vulkan
```

Windows' legacy path limit can otherwise break ggml's nested Vulkan shader
build. The build script handles this automatically by compiling through a
short, per-build NTFS junction under `%LOCALAPPDATA%\tcs`; installed artifacts
and Cargo metadata still use the durable `OUT_DIR` paths. Junction creation
does not require administrator rights.

If junction creation is blocked by filesystem or corporate policy, the build
prints a warning and falls back to the original `OUT_DIR`. Set a short Cargo
target directory to avoid `MAX_PATH` in that case:

```powershell
$env:CARGO_TARGET_DIR = "C:\tc-target"
cargo build --features vulkan
```

## Build-flag escape hatch

The features above cover the common, tested configurations. Anything else CMake
accepts can be forwarded via the `TRANSCRIBE_CMAKE_ARGS` (or `CMAKE_ARGS`) env
var — e.g. `TRANSCRIBE_CMAKE_ARGS="-DGGML_VULKAN=ON" cargo build`. These are
split on whitespace with simple double-quote handling, applied after the
feature-derived defines (so a user `-D` wins), and unsupported/untested by
design: they exist so a Cargo feature is never a hard ceiling on what you can
configure. The link line is still reconstructed from the generated manifest, so
whatever you turn on links correctly.

## Using a prebuilt library

If you have a prebuilt install tree (e.g. downloaded from CI, or produced by a
local `cmake --install`), you can skip the CMake source build entirely — no
C++ toolchain or GPU toolkit needed at the consumer's `cargo build`. Point
`TRANSCRIBE_PREBUILT_PREFIX` at the install prefix:

```bash
TRANSCRIBE_PREBUILT_PREFIX=/path/to/install-prefix cargo build
```

The prefix must be a CMake install tree produced with
`-DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_INSTALL=ON` — i.e. it contains
`lib/transcribe-link.json`, the shared library (`lib/libtranscribe.so` /
`libtranscribe.dylib` / `bin/transcribe.dll`), and the headers under `include/`.
The build script reads the manifest and emits exactly the same link directives
the source-build path would, so the safe wrapper and any downstream crate see
no difference.

CI builds one install tree per GPU backend × platform (CUDA/ROCm/SYCL ×
Linux/Windows) and ships them as zip artifacts; see
[`.github/workflows/capi-build.yml`](../../.github/workflows/capi-build.yml).
Download the one matching your target, unzip it, and set
`TRANSCRIBE_PREBUILT_PREFIX` to the unzipped prefix.

**Runtime note.** A shared library still needs to be found at run time. On
Linux/macOS the build script emits an rpath to the install tree's `lib` dir;
on Windows (no rpath) the build script stages `bin/*.dll` next to Cargo's
build artifacts, but for a deployed binary you must place `transcribe.dll`
(and any ggml DLLs) next to the executable yourself.

## ABI drift

The generated FFI is committed and CI-checked against `include/transcribe.abihash`
(`cargo xtask bindgen --check`): a public-header ABI change turns the check red
until the bindings are regenerated. Per-field layout checks are waived because
bindgen takes layout from a real compiler at generation time.

- Crate: `transcribe-cpp-sys` (raw FFI; the safe API is `transcribe-cpp`)
- License: MIT
