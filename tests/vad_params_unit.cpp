// vad_params_unit.cpp - struct_size negotiation for the vad field.
//
// Pins the backward-compatibility contract: a caller whose
// transcribe_run_params is smaller than the vad field's offset is treated
// as VAD-OFF, so existing binaries never accidentally enable VAD. Also
// pins init() defaults for the new vad sub-struct.

#include "transcribe-vad.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using transcribe::vad::effective_mode;
using transcribe::vad::params_present;

void test_init_sets_defaults() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    CHECK(p.struct_size == sizeof(p));
    CHECK(p.vad.struct_size == sizeof(p.vad));
    CHECK(p.vad.mode == TRANSCRIBE_VAD_OFF);
    // A freshly-init'd full struct reports its vad field present and OFF.
    CHECK(params_present(&p));
    CHECK(effective_mode(&p) == TRANSCRIBE_VAD_OFF);
}

void test_null_params_is_off() {
    CHECK(!params_present(nullptr));
    CHECK(effective_mode(nullptr) == TRANSCRIBE_VAD_OFF);
}

// Simulate a caller built against an OLDER header whose struct ended before
// the vad field. We hand-stamp a short struct_size and verify the negotiation
// helper reports the field absent (and effective_mode returns OFF), without
// ever reading the (absent) vad bytes.
void test_old_struct_size_treated_as_off() {
    transcribe_run_params full;
    transcribe_run_params_init(&full);
    const size_t vad_off = offsetof(transcribe_run_params, vad);  // byte offset of vad field

    // A caller whose declared struct_size stops BEFORE the vad field.
    const uint64_t short_size = static_cast<uint64_t>(vad_off);
    full.struct_size          = short_size;  // pretend the caller's struct is this short

    CHECK(!params_present(&full));
    CHECK(effective_mode(&full) == TRANSCRIBE_VAD_OFF);
}

void test_explicit_silero_mode_round_trips() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.vad.mode = TRANSCRIBE_VAD_SILERO;
    CHECK(effective_mode(&p) == TRANSCRIBE_VAD_SILERO);

    p.vad.mode = TRANSCRIBE_VAD_ENERGY;
    CHECK(effective_mode(&p) == TRANSCRIBE_VAD_ENERGY);

    p.vad.mode = TRANSCRIBE_VAD_OFF;
    CHECK(effective_mode(&p) == TRANSCRIBE_VAD_OFF);
}

}  // namespace

int main() {
    test_init_sets_defaults();
    test_null_params_is_off();
    test_old_struct_size_treated_as_off();
    test_explicit_silero_mode_round_trips();

    if (g_failures == 0) {
        std::printf("vad_params_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_params_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
