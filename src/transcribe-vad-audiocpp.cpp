// transcribe-vad-audiocpp.cpp - runtime loader for audiocpp.dll.

#include "transcribe-vad-audiocpp.h"

#include <cstdlib>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace transcribe::vad::audiocpp {

namespace {

void * platform_load(const char * path) {
#if defined(_WIN32)
    return LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void platform_unload(void * h) {
#if defined(_WIN32)
    if (h) {
        FreeLibrary(static_cast<HMODULE>(h));
    }
#else
    if (h) {
        dlclose(h);
    }
#endif
}

void * platform_sym(void * h, const char * name) {
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(h), name));
#else
    return dlsym(h, name);
#endif
}

// Walk the dll discovery order, returning the first loadable path, or the
// bare name "audiocpp.dll"/"libaudiocpp.so" as a last resort (lets the OS
// loader search PATH). Fills candidate; returns true if candidate is set.
bool discover_path(const char * explicit_path, std::string & candidate) {
    if (explicit_path && explicit_path[0] != '\0') {
        candidate = explicit_path;
        return true;
    }
#if defined(_WIN32)
    const char * env = std::getenv("TRANSCRIBE_VAD_DLL");
    if (env && env[0] != '\0') {
        candidate = env;
        return true;
    }
    // Executable directory.
    char exe[MAX_PATH] = {0};
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        std::string dir(exe);
        const auto slash = dir.find_last_of("\\/");
        if (slash != std::string::npos) {
            dir.resize(slash + 1);
            candidate = dir + "audiocpp.dll";
            // Prefer explicit exe-dir copy if present; LoadLibrary will fail
            // cleanly otherwise and we fall through to the bare name.
            if (GetFileAttributesA(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) {
                return true;
            }
        }
    }
    candidate = "audiocpp.dll";  // let the OS loader search cwd + PATH
    return true;
#else
    const char * env = std::getenv("TRANSCRIBE_VAD_DLL");
    if (env && env[0] != '\0') {
        candidate = env;
        return true;
    }
    candidate = "libaudiocpp.so";
    return true;
#endif
}

template <typename Fn>
bool resolve(void * h, const char * name, Fn & out, std::string & err_msg) {
    void * p = platform_sym(h, name);
    if (p == nullptr) {
        err_msg += std::string("missing symbol: ") + name + "; ";
        return false;
    }
    // Reinterpret_cast through void* is the standard pattern for dlsym/
    // GetProcAddress function pointers (a direct cast triggers -Wpedantic).
    out = reinterpret_cast<Fn>(p);
    return true;
}

}  // namespace

loaded_dll load_audiocpp_dll(const char * explicit_path, std::string & err_msg) {
    err_msg.clear();
    loaded_dll result;

    std::string path;
    if (!discover_path(explicit_path, path)) {
        err_msg = "audiocpp discovery produced no candidate path";
        return result;
    }

    void * h = platform_load(path.c_str());
    if (h == nullptr) {
#if defined(_WIN32)
        err_msg = "LoadLibraryA failed for \"" + path + "\" (err=" + std::to_string(GetLastError()) + ")";
#else
        const char * e = dlerror();
        err_msg        = std::string("dlopen failed for \"") + path + "\": " + (e ? e : "unknown");
#endif
        return result;
    }

    bool ok = true;
    ok &= resolve(h, "audiocpp_load_model", result.syms.load_model, err_msg);
    ok &= resolve(h, "audiocpp_free_model", result.syms.free_model, err_msg);
    ok &= resolve(h, "audiocpp_vad", result.syms.vad, err_msg);
    ok &= resolve(h, "audiocpp_vad_energy", result.syms.vad_energy, err_msg);
    ok &= resolve(h, "audiocpp_free_vad", result.syms.free_vad, err_msg);
    ok &= resolve(h, "audiocpp_free_string", result.syms.free_string, err_msg);
    ok &= resolve(h, "audiocpp_clear_error", result.syms.clear_error, err_msg);
    ok &= resolve(h, "audiocpp_set_progress_callback", result.syms.set_progress_callback, err_msg);

    if (!ok) {
        platform_unload(h);
        err_msg = "audiocpp.dll loaded but missing required symbols: " + err_msg;
        return result;  // handle stays nullptr
    }

    result.handle = h;
    return result;
}

}  // namespace transcribe::vad::audiocpp
