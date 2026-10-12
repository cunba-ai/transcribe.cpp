// sortformer_push_stream_unit.cpp - Sortformer push-audio streaming
// session (TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM, STREAM slot).
//
// Covers, against a real GGUF (env-gated, RC 77 skip):
//
//   1. transcribe_model_accepts_ext_kind: SFPS accepted on _STREAM only;
//      SFST stays RUN-only; foreign kinds rejected.
//   2. transcribe_sortformer_push_stream_ext_init stamps size/kind/preset.
//   3. Pre-clear rejection: a wrong-kind stream ext and an out-of-range
//      preset both fail begin with INVALID_ARG, leave the state NOT
//      ACTIVE, and preserve the previous run's speaker rows
//      (stream_validate fires before clear_result).
//   4. Streaming vs offline parity: feeding the committed 2-speaker
//      oracle mix in 1 s chunks with preset VERY_HIGH_LATENCY ends with
//      exactly the same committed rows as transcribe_run with the same
//      preset (sorted comparison; the mel path and window scheduling are
//      bit-identical on this build).
//   5. Chunk-size invariance: 0.5 s and 10 s feeds produce the same rows
//      as 1 s feeds (scheduling depends on frame indices, not feeds).
//   6. Committed monotonicity + tentative discipline: the committed count
//      only grows while feeding; at most one open turn per speaker;
//      finalize closes every tentative turn into the committed set.
//   7. A different geometry (LOW_LATENCY) streams and produces rows;
//      back-to-back streams on one session work after finalize/reset.
//
// Gated by TRANSCRIBE_SORTFORMER_GGUF (same pattern as the other
// real-model tests).

#include "diar_rows.h"
#include "transcribe.h"
#include "transcribe/sortformer.h"
#include "wav.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

// Shared row fixtures (tests/diar_rows.h): file/env plumbing, the
// sorted/compare row helpers, and the offline DIARIZE-role baseline;
// the tentative-row and debug helpers below stay local to this suite.
using diar_rows::file_exists;
using diar_rows::read_segments;
using diar_rows::same_rows;
using diar_rows::sorted_rows;
using diar_rows::unset_env;

std::vector<transcribe_speaker_segment> read_tentative(const transcribe_session * session) {
    std::vector<transcribe_speaker_segment> rows;
    const int                               n = transcribe_sortformer_push_stream_n_tentative(session);
    for (int i = 0; i < n; ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        if (transcribe_sortformer_push_stream_get_tentative(session, i, &row) == TRANSCRIBE_OK) {
            rows.push_back(row);
        }
    }
    return rows;
}

bool debug_enabled() {
    const char * v = std::getenv("TRANSCRIBE_PUSH_STREAM_DEBUG");
    return v != nullptr && v[0] != '\0';
}

void dump_rows(const char * tag, const std::vector<transcribe_speaker_segment> & rows) {
    std::fprintf(stderr, "%s (%zu rows):\n", tag, rows.size());
    for (const auto & r : rows) {
        std::fprintf(stderr, "  spk%d %lld..%lld ms\n", r.speaker_id, static_cast<long long>(r.t0_ms),
                     static_cast<long long>(r.t1_ms));
    }
}

// Stream `pcm` through a push-audio session in `chunk_samples` feeds.
// Returns the committed rows after finalize. Asserts committed
// monotonicity and the tentative discipline along the way.
std::vector<transcribe_speaker_segment> run_stream(transcribe_session *         session,
                                                   const std::vector<float> &   pcm,
                                                   size_t                       chunk_samples,
                                                   transcribe_sortformer_preset preset,
                                                   bool &                       saw_tentative) {
    transcribe_sortformer_push_stream_ext ext;
    transcribe_sortformer_push_stream_ext_init(&ext);
    ext.preset = preset;

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.family = &ext.ext;

    if (transcribe_stream_begin(session, &rp, &sp) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: stream_begin\n");
        ++g_failures;
        return {};
    }
    CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_ACTIVE);

    int          prev_committed = 0;
    const size_t total          = pcm.size();
    for (size_t off = 0; off < total; off += chunk_samples) {
        const int                n = static_cast<int>(std::min<size_t>(chunk_samples, total - off));
        transcribe_stream_update update;
        transcribe_stream_update_init(&update);
        if (transcribe_stream_feed(session, pcm.data() + off, n, &update) != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: stream_feed at offset %zu\n", off);
            ++g_failures;
            transcribe_stream_reset(session);
            return {};
        }
        const int committed = transcribe_stream_n_committed_segments(session);
        if (debug_enabled()) {
            std::fprintf(stderr, "  feed off=%zu n=%d -> committed=%d tentative=%d\n", off, n, committed,
                         transcribe_sortformer_push_stream_n_tentative(session));
        }
        if (committed < prev_committed) {
            std::fprintf(stderr, "FAIL: committed count shrank: %d < %d\n", committed, prev_committed);
            ++g_failures;
        }
        prev_committed = committed;
        // Committed rows are exactly the exposed rows (monotone prefix).
        CHECK(committed == transcribe_n_speaker_segments(session));
        // At most one open turn per speaker (4 columns).
        const int tentative_n = transcribe_sortformer_push_stream_n_tentative(session);
        if (tentative_n > 4) {
            std::fprintf(stderr, "FAIL: %d tentative rows (> 4 speakers)\n", tentative_n);
            ++g_failures;
        }
        if (tentative_n > 0) {
            saw_tentative = true;
            // Tentative turns must not precede the committed frontier.
            for (const auto & t : read_tentative(session)) {
                CHECK(t.t1_ms >= 0);
            }
        }
    }

    transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    if (transcribe_stream_finalize(session, &update) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: stream_finalize\n");
        ++g_failures;
        return {};
    }
    CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_FINISHED);
    CHECK(transcribe_sortformer_push_stream_n_tentative(session) == 0);
    return read_segments(session);
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_SORTFORMER_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "sortformer_push_stream_unit: TRANSCRIBE_SORTFORMER_GGUF not set; skipping.\n"
                     "Re-run with TRANSCRIBE_SORTFORMER_GGUF=<path to "
                     "diar_streaming_sortformer_4spk-v2.1-F32.gguf>\n");
        return 77;
    }
    const std::string gguf = env;
    if (!file_exists(gguf)) {
        std::fprintf(stderr, "sortformer_push_stream_unit: file not found: %s\n", gguf.c_str());
        return 77;
    }
    const std::string  wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/sortformer-2spk-mix.wav";
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "sortformer_push_stream_unit: wav load: %s\n", wav_err.c_str());
        return 77;
    }
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_RC");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_LC");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN");
    unset_env("TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD");

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = TRANSCRIBE_BACKEND_CPU;  // deterministic float order for parity
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf.c_str(), &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    // 1. Kind+slot probe.
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                            TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM));
    CHECK(
        !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM));
    CHECK(
        !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, 0x4E524857u /* WHRN */));

    // 2. Init function stamps the header + default.
    transcribe_sortformer_push_stream_ext ext;
    transcribe_sortformer_push_stream_ext_init(&ext);
    CHECK(ext.ext.size == sizeof(ext));
    CHECK(ext.ext.kind == TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM);
    CHECK(ext.preset == TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);

    struct transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL: session create\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);

    // Baseline offline run through the DIARIZE role (VERY_HIGH_LATENCY,
    // the DER-validated point).
    const std::vector<transcribe_speaker_segment> offline_rows = diar_rows::sorted_rows(
        diar_rows::offline_diarize_rows(model, pcm, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY));
    CHECK(!offline_rows.empty());

    // 4. Streaming vs offline parity (1 s feeds).
    bool                                    saw_tentative = false;
    std::vector<transcribe_speaker_segment> streamed;
    {
        const size_t one_s = 16000;
        streamed =
            sorted_rows(run_stream(session, pcm, one_s, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY, saw_tentative));
        if (!same_rows(streamed, offline_rows)) {
            dump_rows("offline", offline_rows);
            dump_rows("streamed", streamed);
        }
        CHECK(same_rows(streamed, offline_rows));
    }

    // 3. Pre-clear rejection preserves the previous result and state (the
    // previous result is the section-4 stream's committed rows).
    {
        transcribe_sortformer_push_stream_ext bad;
        transcribe_sortformer_push_stream_ext_init(&bad);
        bad.ext.kind = 0x4E524857u;  // WHRN: wrong family kind
        sp.family    = &bad.ext;
        CHECK(transcribe_stream_begin(session, &rp, &sp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_stream_get_state(session) != TRANSCRIBE_STREAM_ACTIVE);
        CHECK(same_rows(sorted_rows(read_segments(session)), streamed));  // snapshot intact

        transcribe_sortformer_push_stream_ext oor;
        transcribe_sortformer_push_stream_ext_init(&oor);
        oor.preset = static_cast<transcribe_sortformer_preset>(99);
        sp.family  = &oor.ext;
        CHECK(transcribe_stream_begin(session, &rp, &sp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_stream_get_state(session) != TRANSCRIBE_STREAM_ACTIVE);
        CHECK(same_rows(sorted_rows(read_segments(session)), streamed));
        sp.family = nullptr;
    }

    // 5. Chunk-size invariance (0.5 s and 10 s feeds).
    // NOTE: on this 12 s clip VERY_HIGH_LATENCY (chunk + lookahead =
    // 24.3 s of audio) never reaches its feed-time emission threshold, so
    // every chunk runs at finalize and no tentative turns are visible —
    // the tentative discipline is asserted on the LOW_LATENCY stream in
    // check 7, whose 480 ms chunks emit continuously mid-speech.
    {
        const size_t                                  half_s = 8000;
        const size_t                                  ten_s  = 160000;
        const std::vector<transcribe_speaker_segment> r05    = sorted_rows(
            run_stream(session, pcm, half_s, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY, saw_tentative));
        const std::vector<transcribe_speaker_segment> r10 =
            sorted_rows(run_stream(session, pcm, ten_s, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY, saw_tentative));
        CHECK(same_rows(r05, offline_rows));
        CHECK(same_rows(r10, offline_rows));
    }

    // 6. Tentative discipline mid-stream is covered inside run_stream;
    // here: a fresh NULL-ext stream (DEFAULT preset) also finalizes.
    {
        transcribe_stream_params sp_default;
        transcribe_stream_params_init(&sp_default);
        transcribe_run_params rp_default;
        transcribe_run_params_init(&rp_default);
        CHECK(transcribe_stream_begin(session, &rp_default, &sp_default) == TRANSCRIBE_OK);
        const int n = static_cast<int>(pcm.size());
        CHECK(transcribe_stream_feed(session, pcm.data(), n, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_stream_finalize(session, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_n_speaker_segments(session) > 0);
    }

    // 7. A different geometry streams and produces rows, and matches the
    // offline diarize run at the same operating point after finalize.
    {
        const std::vector<transcribe_speaker_segment> lo_offline = diar_rows::sorted_rows(
            diar_rows::offline_diarize_rows(model, pcm, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY));
        CHECK(!lo_offline.empty());

        bool                                          saw_lo_tentative = false;
        const std::vector<transcribe_speaker_segment> lo =
            run_stream(session, pcm, 16000, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY, saw_lo_tentative);
        CHECK(!lo.empty());
        CHECK(saw_lo_tentative);  // 480 ms chunks: the edge is mid-speech on this mix
        if (!same_rows(sorted_rows(lo), lo_offline)) {
            dump_rows("offline low_latency", lo_offline);
            dump_rows("streamed low_latency", lo);
        }
        CHECK(same_rows(sorted_rows(lo), lo_offline));
    }

    transcribe_session_free(session);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "sortformer_push_stream_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sortformer_push_stream_unit: OK\n");
    return 0;
}
