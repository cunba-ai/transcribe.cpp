// transcribe-vad-audiocpp.h - dynamic loader for audiocpp.dll.
//
// We do NOT include audiocpp.h (compile-time zero-dependency on audio.cpp).
// Instead we redeclare the handful of symbols we need as opaque types +
// function pointers, and resolve them at runtime via LoadLibrary/dlopen.
// Signatures MUST match ref/cuda-release/audiocpp.h verbatim.

#ifndef TRANSCRIBE_VAD_AUDIOCPP_H
#define TRANSCRIBE_VAD_AUDIOCPP_H

#include "transcribe.h"

#include <cstdint>
#include <string>

namespace transcribe::vad::audiocpp {

// ---- Opaque audiocpp types (forward-declared; never defined here) ---------

struct audiocpp_model_t;  // opaque
struct audiocpp_vad_t;    // opaque
struct audiocpp_error_t;  // opaque (we only pass pointers through)

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

}  // namespace transcribe::vad::audiocpp

#endif  // TRANSCRIBE_VAD_AUDIOCPP_H
