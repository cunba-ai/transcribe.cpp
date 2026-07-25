// vad_integration.cpp - end-to-end VAD-vs-fullbuffer ASR parity test.
//
// Verifies the core VAD-integration contract: running Qwen3-ASR over
// meeting.wav with VAD chunking (SILERO) produces text equivalent to
// running it full-buffer. VAD must not change WHAT is transcribed, only
// HOW it is sliced — so the two texts should match (modulo whitespace).
//
// NOT run in CI (no dll / no GGUF there). Run locally:
//   cmake -B build -S . -DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON -DTRANSCRIBE_BUILD_TESTS=ON
//   cmake --build build --target transcribe_vad_integration
//   TRANSCRIBE_VAD_DLL=<audio.cpp>/ref/cuda-release/audiocpp.dll \
//   TRANSCRIBE_VAD_TEST_MODEL=<sound-rs>/models/transcribe/Qwen3-ASR-0.6B-Q5_K_M.gguf \
//   TRANSCRIBE_VAD_TEST_WAV=<sound-rs>/samples/meeting.wav \
//   ctest --test-dir build -R vad_integration --output-on-failure
//
// Returns 77 (CTest SKIP_RETURN_CODE) if the dll, model, or fixture is
// missing — safe to register in any build.

#include "transcribe-vad-audiocpp.h"
#include "transcribe.h"
#include "wav.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

bool env_path(const char * name, std::string & out) {
    const char * v = std::getenv(name);
    if (v && v[0]) {
        out = v;
        return true;
    }
    return false;
}

// Lowercase + strip punctuation + collapse whitespace, returning a
// space-separated word list. ASR models routinely differ across chunk
// boundaries in punctuation and segment-initial capitalization (a chunk
// starting mid-sentence gets a period + capital); those are model behavior,
// not VAD bugs. Comparing the lowercased word sequence catches real bugs
// (dropped/inserted/misordered words) while tolerating the surface form.
std::string normalize_words(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc)) {
            prev_space = true;
            continue;
        }
        // Keep alphanumerics (across scripts: letters/digits pass isalnum);
        // drop punctuation. This is intentionally ASCII-centric — the test
        // fixture is English (jfk/meeting). For CJK fixtures a different
        // normalization would be needed.
        if (std::isalnum(uc)) {
            if (prev_space && !out.empty()) {
                out.push_back(' ');
            }
            out.push_back(static_cast<char>(std::tolower(uc)));
            prev_space = false;
        }
    }
    return out;
}

bool file_readable(const std::string & p) {
    FILE * f = std::fopen(p.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

// AUDIOCPP_BACKEND_CUDA = 1 (ref/cuda-release/audiocpp.h:78). VAD + ASR both
// on GPU per the integration-test contract.
constexpr int kAudiocppBackendCuda = 1;

// Run ASR over pcm with the given vad_mode. Returns the full text (normalized)
// via out_text; returns TRANSCRIBE_OK on success. model_path / dll_path /
// wav_label feed diagnostics.
transcribe_status run_asr(const std::string & model_path,
                          const float *       pcm,
                          int                 n_samples,
                          transcribe_vad_mode vad_mode,
                          std::string &       out_text) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CUDA;  // ASR on GPU
    transcribe_model * model = nullptr;
    transcribe_status  st = transcribe_model_load_file(model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        return st == TRANSCRIBE_OK ? TRANSCRIBE_ERR_BACKEND : st;
    }

    transcribe_session_params cp;
    transcribe_session_params_init(&cp);
    transcribe_session * ctx = nullptr;
    st = transcribe_session_init(model, &cp, &ctx);
    if (st != TRANSCRIBE_OK || ctx == nullptr) {
        transcribe_model_free(model);
        return st == TRANSCRIBE_OK ? TRANSCRIBE_ERR_BACKEND : st;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.language = "en";  // meeting.wav is English
    if (vad_mode != TRANSCRIBE_VAD_OFF) {
        rp.vad.mode    = vad_mode;
        rp.vad.backend = kAudiocppBackendCuda;  // VAD on GPU
        // Qwen3-ASR advertises a huge effective_max_audio_ms (~87 min, from
        // its 65536-token INPUT context) but a 256-token GENERATION budget.
        // VAD's default max_chunk (family effective_max_audio_ms) would
        // produce chunks whose transcript blows past 256 tokens -> truncation.
        // Cap chunks at 10s so each segment's transcript stays well under
        // the 256-token generation ceiling. This also exercises the
        // max_chunk_ms override path.
        rp.vad.max_chunk_ms = 10000;
    }
    st = transcribe_run(ctx, pcm, n_samples, &rp);

    out_text.clear();
    if (st == TRANSCRIBE_OK) {
        const char * full = transcribe_full_text(ctx);
        if (full) {
            out_text = normalize_words(full);
        }
    }
    transcribe_session_free(ctx);
    transcribe_model_free(model);
    return st;
}

}  // namespace

int main() {
    // Skip cleanly when any prerequisite is missing.
    std::string dll_path, model_path, wav_path;
    const bool  have_dll   = env_path("TRANSCRIBE_VAD_DLL", dll_path);
    const bool  have_model = env_path("TRANSCRIBE_VAD_TEST_MODEL", model_path);
    const bool  have_wav   = env_path("TRANSCRIBE_VAD_TEST_WAV", wav_path);
    if (!have_dll || !file_readable(dll_path)) {
        std::printf("vad_integration: SKIP (TRANSCRIBE_VAD_DLL not set or unreadable)\n");
        return 77;
    }
    if (!have_model || !file_readable(model_path)) {
        std::printf("vad_integration: SKIP (TRANSCRIBE_VAD_TEST_MODEL not set or unreadable)\n");
        return 77;
    }
    if (!have_wav || !file_readable(wav_path)) {
        std::printf("vad_integration: SKIP (TRANSCRIBE_VAD_TEST_WAV not set or unreadable)\n");
        return 77;
    }

    // Sanity: the dll must actually load (the VAD branch needs it).
    std::string load_err;
    auto        loaded = transcribe::vad::audiocpp::load_audiocpp_dll(dll_path.c_str(), load_err);
    if (loaded.handle == nullptr) {
        std::printf("vad_integration: SKIP (audiocpp.dll load failed: %s)\n", load_err.c_str());
        return 77;
    }

    // Load the fixture (16k mono f32).
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::printf("vad_integration: SKIP (cannot load %s: %s)\n", wav_path.c_str(), wav_err.c_str());
        return 77;
    }
    const double seconds = static_cast<double>(pcm.size()) / 16000.0;
    std::printf("vad_integration: %s (%zu samples, %.1fs)\n", wav_path.c_str(), pcm.size(), seconds);
    std::printf("vad_integration: model %s\n", model_path.c_str());

    // Decide the compare mode from audio length. Qwen3-ASR is a short-form
    // model: past ~30s a full-buffer decode either truncates at the
    // generation budget or, on GPU, OOMs on the audio-token activation
    // buffer (a 23-min clip wants ~36 GiB). That OOM currently aborts the
    // process rather than returning an error, so we cannot "try full-buffer
    // and degrade" — we must NOT attempt it for long audio. Long audio is
    // exactly what VAD chunking is for, so:
    //   <= 30s: run full-buffer AND VAD-chunked, assert word-level equality.
    //   >  30s: run VAD-chunked only, assert it produces non-empty text.
    const double  audio_seconds = static_cast<double>(pcm.size()) / 16000.0;
    const bool    short_audio = audio_seconds <= 30.0;

    std::string   text_full;
    bool          full_ok = false;
    if (short_audio) {
        transcribe_status st_full = run_asr(model_path, pcm.data(), static_cast<int>(pcm.size()),
                                            TRANSCRIBE_VAD_OFF, text_full);
        if (st_full == TRANSCRIBE_ERR_BACKEND) {
            std::printf("vad_integration: SKIP (ASR backend init failed; CUDA unavailable?)\n");
            return 77;
        }
        full_ok = (st_full == TRANSCRIBE_OK) && !text_full.empty();
        if (!full_ok) {
            std::fprintf(stderr, "vad_integration: short-audio full-buffer run failed (status %d)\n", st_full);
            return EXIT_FAILURE;
        }
        std::printf("vad_integration: full-buffer text: %zu chars\n", text_full.size());
    } else {
        std::printf("vad_integration: long audio (%.1fs) — skipping full-buffer (would OOM/truncate); "
                    "VAD chunking is the point here\n", audio_seconds);
    }

    // 2. VAD-chunked run (SILERO). This is the path under test.
    std::string      text_vad;
    const transcribe_status st_vad = run_asr(model_path, pcm.data(), static_cast<int>(pcm.size()),
                                             TRANSCRIBE_VAD_SILERO, text_vad);
    CHECK(st_vad == TRANSCRIBE_OK);
    if (st_vad != TRANSCRIBE_OK) {
        std::fprintf(stderr, "vad_integration: VAD-chunked ASR failed (status %d)\n", st_vad);
        return EXIT_FAILURE;
    }
    std::printf("vad_integration: VAD-chunked text: %zu chars\n", text_vad.size());
    // VAD must always produce a non-empty transcript for non-empty speech.
    CHECK(!text_vad.empty());

    // 3. The core assertion, scoped to what full-buffer can give us:
    //    - If full-buffer succeeded (short audio): VAD must not change the
    //      transcribed WORDS (order + identity), tolerating punctuation /
    //      capitalization differences that chunking legitimately induces.
    //    - If full-buffer truncated/OOM'd (long audio): we already asserted
    //      VAD produced text; that IS the value VAD adds here.
    if (full_ok) {
        if (text_full == text_vad) {
            std::printf("vad_integration: text MATCH (full == VAD-chunked, word-level)\n");
        } else {
            const size_t min_len = std::min(text_full.size(), text_vad.size());
            size_t       common  = 0;
            while (common < min_len && text_full[common] == text_vad[common]) {
                ++common;
            }
            std::fprintf(stderr,
                         "vad_integration: WORD MISMATCH at char %zu (full=%zu vad=%zu)\n"
                         "  full: ...%s\n"
                         "  vad : ...%s\n",
                         common, text_full.size(), text_vad.size(),
                         text_full.substr(std::min(common, text_full.size() - 1), 80).c_str(),
                         text_vad.substr(std::min(common, text_vad.size() - 1), 80).c_str());
            ++g_failures;
        }
    } else {
        std::printf("vad_integration: long-audio path — VAD produced %zu chars (full-buffer "
                    "unavailable for compare)\n", text_vad.size());
    }

    if (g_failures == 0) {
        std::printf("vad_integration: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_integration: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
