// transcribe-vad-audiocpp.cpp - runtime loader for audiocpp.dll.

#include "transcribe-vad-audiocpp.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#else
#    include <dlfcn.h>
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
    char exe[MAX_PATH] = { 0 };
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        std::string dir(exe);
        const auto  slash = dir.find_last_of("\\/");
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

template <typename Fn> bool resolve(void * h, const char * name, Fn & out, std::string & err_msg) {
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

runtime & runtime::instance() {
    static runtime inst;
    return inst;
}

bool runtime::ensure_loaded(const char *        dll_path,
                            transcribe_vad_mode mode,
                            int                 backend,
                            int                 device_id,
                            int                 n_threads,
                            std::string &       err_msg) {
    std::call_once(load_once_, [&] {
        loaded_ok_  = false;
        auto loaded = load_audiocpp_dll(dll_path, load_err_);
        if (loaded.handle == nullptr) {
            return;  // load_err_ already set
        }
        dll_handle_ = loaded.handle;
        syms_       = loaded.syms;

        if (mode == TRANSCRIBE_VAD_SILERO) {
            // Embedded weights: pass NULL model_path. family_hint="silero_vad",
            // task=VAD(0), backend/device/threads forwarded.
            audiocpp_error_t err{};
            model_ = syms_.load_model(/*model_path*/ nullptr, "silero_vad",
                                      /*task VAD*/ 0, backend, device_id, n_threads, &err);
            if (model_ == nullptr) {
                load_err_ = "audiocpp_load_model(silero_vad) returned NULL";
                if (syms_.clear_error) {
                    syms_.clear_error(&err);
                }
                return;
            }
        }
        // ENERGY mode: no model needed.
        loaded_ok_ = true;
    });
    err_msg = load_err_;
    return loaded_ok_;
}

}  // namespace transcribe::vad::audiocpp

namespace transcribe::vad {

// audiocpp's segment struct (from ref/cuda-release/audiocpp.h:364-378):
//   struct { int64_t start_sample; int64_t end_sample; float confidence; }
//   struct audiocpp_vad_t { segment_t * segments; int64_t n_segments; };
// Redeclared locally to avoid including audiocpp.h.
namespace audiocpp {
// Concrete layout for the opaque audiocpp_vad_t the loader hands back. Match
// ref/cuda-release/audiocpp.h:364-378 verbatim. Defined here (not in the
// header) so only this TU sees the layout. audiocpp_error_t is fully defined
// in the header (caller-allocated POD).
struct audiocpp_vad_segment_t {
    int64_t start_sample;
    int64_t end_sample;
    float   confidence;
};

struct audiocpp_vad_t {
    audiocpp_vad_segment_t * segments;
    int64_t                  n_segments;
};
}  // namespace audiocpp

std::vector<time_span> vad_invoke(const float * pcm, int64_t n_samples, const struct transcribe_vad_params & params) {
    using transcribe::vad::audiocpp::audiocpp_error_t;
    using transcribe::vad::audiocpp::audiocpp_vad_t;
    using transcribe::vad::audiocpp::runtime;

    auto & rt = runtime::instance();
    if (!rt.loaded_ok()) {
        throw std::runtime_error("audiocpp runtime not loaded");
    }
    const auto * syms = &rt.syms();

    // Build options_json for SILERO tuning (ENERGY accepts the same shape for
    // threshold/etc. where applicable; extra keys are ignored by audiocpp).
    std::string options = "{";
    bool        first   = true;
    auto        add     = [&](const char * k, const std::string & v) {
        if (!first) {
            options += ",";
        }
        options += "\"";
        options += k;
        options += "\":";
        options += v;
        first = false;
    };
    if (params.silero_threshold > 0.0f) {
        add("threshold", std::to_string(params.silero_threshold));
    }
    if (params.silero_min_speech_ms > 0) {
        add("min_speech_duration_ms", std::to_string(params.silero_min_speech_ms));
    }
    if (params.silero_min_silence_ms > 0) {
        add("min_silence_duration_ms", std::to_string(params.silero_min_silence_ms));
    }
    options += "}";

    audiocpp_vad_t * result = nullptr;
    {
        std::lock_guard<std::mutex> lk(rt.vad_mutex());
        audiocpp_error_t            err{};
        if (params.mode == TRANSCRIBE_VAD_ENERGY) {
            result = syms->vad_energy(pcm, n_samples, 16000, options.c_str(), &err);
        } else {
            // SILERO
            result = syms->vad(rt.model(), pcm, n_samples, 16000, options.c_str(), &err);
        }
    }
    if (result == nullptr) {
        throw std::runtime_error("audiocpp_vad returned NULL");
    }

    std::vector<time_span> out;
    out.reserve(static_cast<size_t>(result->n_segments));
    for (int64_t i = 0; i < result->n_segments; ++i) {
        const auto & s = result->segments[i];
        time_span    ts;
        ts.start_ms   = s.start_sample * 1000 / 16000;
        ts.end_ms     = s.end_sample * 1000 / 16000;
        ts.confidence = s.confidence;
        out.push_back(ts);
    }
    syms->free_vad(result);
    return out;
}

}  // namespace transcribe::vad
