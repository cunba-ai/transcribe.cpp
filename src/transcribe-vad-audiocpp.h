// transcribe-vad-audiocpp.h - dynamic loader for audiocpp.dll.
//
// We do NOT include audiocpp.h (compile-time zero-dependency on audio.cpp).
// Instead we redeclare the handful of symbols we need as opaque types +
// function pointers, and resolve them at runtime via LoadLibrary/dlopen.
// Signatures MUST match ref/cuda-release/audiocpp.h verbatim.

#ifndef TRANSCRIBE_VAD_AUDIOCPP_H
#define TRANSCRIBE_VAD_AUDIOCPP_H

#include "transcribe.h"
#include "transcribe-vad.h"  // time_span

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace transcribe::vad::audiocpp {

// ---- Opaque audiocpp types -----------------------------------------------
//
// audiocpp_model_t / audiocpp_vad_t stay forward-declared opaque (we only
// pass pointers through; the loader hands them back and we hand them to
// free). audiocpp_error_t is a caller-allocated POD ({int code; char *msg})
// that we instantiate before each call, so it gets a full definition here
// matching ref/cuda-release/audiocpp.h:66-70 verbatim.

struct audiocpp_model_t;  // opaque
struct audiocpp_vad_t;    // opaque
struct audiocpp_error_t {
    int   code;
    char *message;
};

// ---- Function pointer typedefs (mirror audiocpp.h prototypes) -------------
//
// AUDIOCPP_TASK_VAD = 0, AUDIOCPP_BACKEND_CPU = 0.

using load_model_fn = audiocpp_model_t * (*)(const char * model_path,
                                             const char * family_hint,
                                             int          task,
                                             int          backend,
                                             int          device_id,
                                             int          n_threads,
                                             audiocpp_error_t * err);

using free_model_fn = void (*)(audiocpp_model_t * model);

// Returns audiocpp_vad_t (caller frees via free_vad). NULL on failure.
using vad_fn = audiocpp_vad_t * (*)(const audiocpp_model_t * model,
                                    const float *            pcm,
                                    int64_t                  n_samples,
                                    int                      sample_rate,
                                    const char *             options_json,
                                    audiocpp_error_t *       err);

// Energy VAD: no model needed.
using vad_energy_fn = audiocpp_vad_t * (*)(const float *      pcm,
                                           int64_t            n_samples,
                                           int                sample_rate,
                                           const char *       options_json,
                                           audiocpp_error_t * err);

using free_vad_fn    = void (*)(audiocpp_vad_t * vad);
using free_string_fn = void (*)(char * s);
using clear_error_fn = void (*)(audiocpp_error_t * err);

using set_progress_callback_fn = void (*)(audiocpp_model_t * model,
                                          int (*)(float, const char *, int64_t, int64_t, void *),
                                          void * user_data);

// ---- The resolved symbol table -------------------------------------------
//
// Every pointer is non-null on a successful load. If load_audiocpp_dll fails,
// handle == nullptr and all pointers are nullptr.
struct symbols {
    load_model_fn            load_model            = nullptr;
    free_model_fn            free_model            = nullptr;
    vad_fn                   vad                   = nullptr;
    vad_energy_fn            vad_energy            = nullptr;
    free_vad_fn              free_vad              = nullptr;
    free_string_fn           free_string           = nullptr;
    clear_error_fn           clear_error           = nullptr;
    set_progress_callback_fn set_progress_callback = nullptr;
};

struct loaded_dll {
    void *  handle = nullptr;  // HMODULE / void* from dlopen
    symbols syms{};
};

// Load the dll and resolve all symbols. explicit_path == nullptr triggers
// the discovery order (see transcribe_vad_params.dll_path docs in the
// public header): env TRANSCRIBE_VAD_DLL, executable dir, cwd, system PATH.
// On failure returns {handle=nullptr, syms all-null}; err_msg holds the
// reason. Never throws.
loaded_dll load_audiocpp_dll(const char * explicit_path, std::string & err_msg);

// ---- Process-level singleton managing the loaded dll + model --------------
//
// audiocpp's single model handle is NOT safe for concurrent runs
// (audiocpp.h:17-19). We serialize all VAD calls through vad_mutex().
// Loading is idempotent (std::call_once): the first ensure_loaded wins;
// later calls return the cached ok/err state without retrying.
class runtime {
public:
    static runtime & instance();

    // Idempotent. On the first call, loads the dll (via load_audiocpp_dll
    // with the given dll_path) and, for SILERO mode, the model via
    // audiocpp_load_model(NULL, "silero_vad", ...). ENERGY mode skips the
    // model load. Returns true on success; on failure returns false and
    // err_msg holds the reason (subsequent calls return the same cached
    // result without retrying — a bad dll won't recover mid-process).
    //
    // Thread-safe. Never throws.
    bool ensure_loaded(const char *         dll_path,
                       transcribe_vad_mode  mode,
                       int                  backend,
                       int                  device_id,
                       int                  n_threads,
                       std::string &        err_msg);

    bool               loaded_ok() const { return loaded_ok_; }
    const symbols &    syms() const { return syms_; }
    audiocpp_model_t * model() const { return model_; }  // may be null (ENERGY)

    // Serializes audiocpp_vad / audiocpp_vad_energy calls (single handle is
    // not concurrent-safe). Callers lock this around their vad invoke.
    std::mutex & vad_mutex() { return vad_mutex_; }

private:
    runtime() = default;
    std::once_flag     load_once_;
    void *             dll_handle_ = nullptr;
    symbols            syms_{};
    audiocpp_model_t * model_      = nullptr;
    std::mutex         vad_mutex_;
    bool               loaded_ok_ = false;
    std::string        load_err_;
};

}  // namespace transcribe::vad::audiocpp

namespace transcribe::vad {

// ---- VAD invocation -------------------------------------------------------
//
// Runs VAD (SILERO or ENERGY) on pcm, returning ms-resolution segments.
// Internally locks audiocpp::runtime::vad_mutex(). Throws std::runtime_error
// on any audiocpp failure (caller in run_with_vad / transcribe_vad catches
// and degrades). mode is read from params.
//
// Requires audiocpp::runtime::instance().ensure_loaded(...) to have returned
// true for this mode before the call.
std::vector<time_span> vad_invoke(const float *                   pcm,
                                  int64_t                         n_samples,
                                  const struct transcribe_vad_params & params);

}  // namespace transcribe::vad

#endif  // TRANSCRIBE_VAD_AUDIOCPP_H
