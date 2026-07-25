// transcribe-vad.cpp - struct_size negotiation + (Task 6) plan() impl.

#include "transcribe-vad.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace transcribe::vad {

bool params_present(const struct transcribe_run_params * run_params) {
    if (run_params == nullptr) {
        return false;
    }
    // struct_size is the caller's declared size. The vad field exists only
    // if the caller's struct is large enough to hold it. offsetof gives the
    // byte offset of vad within the (fully-defined) struct as this library
    // was compiled; a caller built against an older, shorter header sets a
    // smaller struct_size and is treated as not having the field.
    const size_t vad_off = offsetof(struct transcribe_run_params, vad);
    return run_params->struct_size >= vad_off + sizeof(struct transcribe_vad_params);
}

transcribe_vad_mode effective_mode(const struct transcribe_run_params * run_params) {
    if (!params_present(run_params)) {
        return TRANSCRIBE_VAD_OFF;
    }
    return run_params->vad.mode;
}

// plan() is implemented in Task 6.

}  // namespace transcribe::vad
