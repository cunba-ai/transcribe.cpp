// vad_embedded_unit.cpp - the embedded Silero VAD asset through the public
// API, model-file-free.
//
// Covers transcribe_model_load_embedded_vad end to end on a default-ON
// build: load with NULL params and with explicit CPU params, the VAD info
// contract (16 kHz / 512-sample frames), a ~2 s run whose middle second is
// a speech-like harmonic burst and whose flanks are near-silence (frame
// count, a segment plausibly covering the burst, finite in-range
// probabilities), INVALID_ARG on a NULL out-model, BAD_STRUCT_SIZE on a
// zeroed params struct, and the NOT_IMPLEMENTED contract when the library
// was built with TRANSCRIBE_EMBED_VAD_MODEL=OFF (reported as a ctest SKIP,
// return 77).

#include "transcribe.h"
#include "transcribe/vad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_EMBED_VAD
#    error "TRANSCRIBE_TEST_EMBED_VAD must be defined by the build system (1 = library embeds the asset)"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// 2 s at 16 kHz: near-silence (0.5 s), a speech-like burst (1.0 s),
// near-silence (0.5 s). The burst is a 140 Hz harmonic stack under 3 Hz
// syllabic amplitude modulation — the reference Silero model (validated in
// the silero-vad 6.2.3 Python env) classifies pure tones and white noise as
// non-speech, so a "loud sine/noise burst" alone would detect nothing; this
// surrogate reliably yields one segment covering the burst. The burst
// boundaries in samples are what the segment-bounds assertion refers to.
constexpr int  kSilenceHead = 8000;
constexpr int  kBurst       = 16000;
constexpr int  kSilenceTail = 8000;

std::vector<float> signal() {
    std::vector<float> v(kSilenceHead + kBurst + kSilenceTail);
    uint32_t           seed = 98765;
    for (size_t i = 0; i < v.size(); ++i) {
        seed      = seed * 1664525u + 1013904223u;
        const float noise = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.01f;
        const bool  burst = i >= kSilenceHead && i < kSilenceHead + kBurst;
        if (burst) {
            const float t    = static_cast<float>(i - kSilenceHead) / 16000.0f;
            const float env  = 0.5f * (1.0f + std::sin(6.2831853f * 3.0f * t - 1.5707964f));
            const float harm = std::sin(6.2831853f * 140.0f * t) + 0.5f * std::sin(6.2831853f * 280.0f * t) +
                               0.25f * std::sin(6.2831853f * 560.0f * t) + 0.3f * std::sin(6.2831853f * 1820.0f * t) +
                               0.2f * std::sin(6.2831853f * 3360.0f * t);
            v[i] = noise + 0.4f * env * harm;
        } else {
            v[i] = noise;
        }
    }
    return v;
}

void test_embedded_load_and_run() {
    // NULL params = all defaults. The embedded asset is CPU-friendly, so
    // the default AUTO policy resolving to strict CPU is fine here.
    transcribe_model * m = reinterpret_cast<transcribe_model *>(0xdeadbeef);
    CHECK(transcribe_model_load_embedded_vad(nullptr, &m) == TRANSCRIBE_OK);
    CHECK(m != nullptr);
    if (m == nullptr) {
        return;
    }

    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_VAD);
    CHECK(std::string(transcribe_model_arch_string(m)) == "silero_vad");

    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000);
    CHECK(info.frame_samples == 512);

    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, nullptr, &s) == TRANSCRIBE_OK);
    CHECK(s != nullptr);

    const std::vector<float> pcm = signal();
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);

    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    // 32000 samples -> ceil(32000 / 512) frames; run zero-pads the tail.
    CHECK(r.n_probs == (static_cast<int64_t>(pcm.size()) + 511) / 512);
    const float * probs = transcribe_vad_probs(s);
    CHECK(probs != nullptr);
    for (int i = 0; i < r.n_probs; ++i) {
        CHECK(std::isfinite(probs[i]) && probs[i] >= 0.0f && probs[i] <= 1.0f);
    }

    // At least one detected segment must plausibly cover the burst: its
    // bounds (loose — padding, min-speech and min-silence all shift them)
    // must reach into both halves of the burst region.
    const int64_t burst_lo = kSilenceHead + kBurst / 4;
    const int64_t burst_hi = kSilenceHead + 3 * kBurst / 4;
    bool         covers    = false;
    for (int i = 0; i < r.n_segments; ++i) {
        transcribe_vad_segment seg;
        transcribe_vad_segment_init(&seg);
        CHECK(transcribe_vad_get_segment(s, i, &seg) == TRANSCRIBE_OK);
        if (seg.start_sample < burst_lo && seg.end_sample > burst_hi) {
            covers = true;
        }
    }
    CHECK(r.n_segments >= 1);
    CHECK(covers);

    transcribe_vad_session_free(s);
    transcribe_model_free(m);

    // Explicit params take the same validation path as load_file: a
    // well-formed CPU request loads; a zeroed struct ({0}) is rejected
    // with BAD_STRUCT_SIZE and *out_model stays NULL.
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m2 = nullptr;
    CHECK(transcribe_model_load_embedded_vad(&mp, &m2) == TRANSCRIBE_OK);
    CHECK(m2 != nullptr);
    transcribe_model_free(m2);

    transcribe_model_load_params zeroed{};
    transcribe_model *           m3 = reinterpret_cast<transcribe_model *>(0xdeadbeef);
    CHECK(transcribe_model_load_embedded_vad(&zeroed, &m3) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(m3 == nullptr);
}

void test_invalid_args() {
    // NULL out-model: INVALID_ARG (and nothing to write).
    CHECK(transcribe_model_load_embedded_vad(nullptr, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
}

}  // namespace

int main() {
#if TRANSCRIBE_TEST_EMBED_VAD
    test_invalid_args();
    test_embedded_load_and_run();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("vad_embedded_unit: ok\n");
    return EXIT_SUCCESS;
#else
    // Library built with TRANSCRIBE_EMBED_VAD_MODEL=OFF: pin the documented
    // contract (NOT_IMPLEMENTED, *out_model NULL) and skip the rest.
    test_invalid_args();
    transcribe_model * m = reinterpret_cast<transcribe_model *>(0xdeadbeef);
    CHECK(transcribe_model_load_embedded_vad(nullptr, &m) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(m == nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("vad_embedded_unit: skipped (TRANSCRIBE_EMBED_VAD_MODEL=OFF; NOT_IMPLEMENTED verified)\n");
    return 77;
#endif
}
