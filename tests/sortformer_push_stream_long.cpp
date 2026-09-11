// sortformer_push_stream_long.cpp - long-file push-audio streaming
// soak test (TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM, STREAM slot).
//
// Streams a multi-minute file through the push-audio session in fixed
// feeds and checks the invariants that only show up at length:
//
//   - committed count and rows are append-only while feeding;
//   - speaker ids stay within the model's 4 arrival-order columns;
//   - timestamps never exceed the audio received so far;
//   - after finalize the committed rows equal the offline transcribe_run
//     rows at the same operating point (sorted comparison);
//   - peak process memory stays bounded by the per-chunk graph plus
//     tails (reported; the chunk geometry is length-independent).
//
// Env-gated (RC 77 skip): TRANSCRIBE_SORTFORMER_GGUF (model) and
// TRANSCRIBE_SORTFORMER_LONG_WAV (audio). TRANSCRIBE_SORTFORMER_LONG_FEED_MS
// optionally overrides the feed size (default 1000 ms).

#include "transcribe.h"
#include "transcribe/sortformer.h"
#include "wav.h"

#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
// clang-format off: psapi.h requires windows.h to be included first.
#    include <windows.h>
#    include <psapi.h>
// clang-format on
#else
#    include <sys/resource.h>
#endif

#include <algorithm>
#include <chrono>
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

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<transcribe_speaker_segment> read_segments(const transcribe_session * session) {
    std::vector<transcribe_speaker_segment> rows;
    const int                               n = transcribe_n_speaker_segments(session);
    for (int i = 0; i < n; ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        if (transcribe_get_speaker_segment(session, i, &row) == TRANSCRIBE_OK) {
            rows.push_back(row);
        }
    }
    return rows;
}

std::vector<transcribe_speaker_segment> sorted_rows(std::vector<transcribe_speaker_segment> rows) {
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.t0_ms != b.t0_ms) {
            return a.t0_ms < b.t0_ms;
        }
        return a.speaker_id < b.speaker_id;
    });
    return rows;
}

bool same_rows(const std::vector<transcribe_speaker_segment> & a, const std::vector<transcribe_speaker_segment> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].t0_ms != b[i].t0_ms || a[i].t1_ms != b[i].t1_ms || a[i].speaker_id != b[i].speaker_id) {
            return false;
        }
    }
    return true;
}

int64_t peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<int64_t>(pmc.PeakWorkingSetSize);
    }
    return -1;
#else
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<int64_t>(ru.ru_maxrss) * 1024;
#endif
}

}  // namespace

int main() {
    const char * env_model = std::getenv("TRANSCRIBE_SORTFORMER_GGUF");
    const char * env_wav   = std::getenv("TRANSCRIBE_SORTFORMER_LONG_WAV");
    if (env_model == nullptr || env_model[0] == '\0' || env_wav == nullptr || env_wav[0] == '\0') {
        std::fprintf(stderr,
                     "sortformer_push_stream_long: TRANSCRIBE_SORTFORMER_GGUF / "
                     "TRANSCRIBE_SORTFORMER_LONG_WAV not set; skipping.\n");
        return 77;
    }
    if (!file_exists(env_model) || !file_exists(env_wav)) {
        std::fprintf(stderr, "sortformer_push_stream_long: file not found\n");
        return 77;
    }
    int feed_ms = 1000;
    if (const char * v = std::getenv("TRANSCRIBE_SORTFORMER_LONG_FEED_MS"); v != nullptr && v[0] != '\0') {
        feed_ms = std::atoi(v);
        if (feed_ms <= 0) {
            feed_ms = 1000;
        }
    }

    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(env_wav, pcm, wav_err)) {
        std::fprintf(stderr, "sortformer_push_stream_long: wav load: %s\n", wav_err.c_str());
        return 77;
    }
    const int64_t audio_ms = static_cast<int64_t>(pcm.size()) * 1000 / 16000;
    std::printf("sortformer_push_stream_long: %s (%.1f s), feed %d ms\n", env_wav, audio_ms / 1000.0, feed_ms);

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(env_model, &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }
    struct transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL: session create\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // Offline baseline at the same operating point.
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_sortformer_stream_ext run_ext;
    transcribe_sortformer_stream_ext_init(&run_ext);
    run_ext.preset             = TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY;
    rp.family                  = &run_ext.ext;
    const auto t_offline_start = std::chrono::steady_clock::now();
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const double offline_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_offline_start).count();
    rp.family              = nullptr;
    const std::vector<transcribe_speaker_segment> offline_rows = sorted_rows(read_segments(session));
    std::printf("offline: %zu rows in %.1f s\n", offline_rows.size(), offline_s);

    // Push-audio stream, fixed-size feeds.
    transcribe_sortformer_push_stream_ext ext;
    transcribe_sortformer_push_stream_ext_init(&ext);
    ext.preset = TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY;
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.family = &ext.ext;
    CHECK(transcribe_stream_begin(session, &rp, &sp) == TRANSCRIBE_OK);

    const size_t chunk_samples  = static_cast<size_t>(feed_ms) * 16;
    int          prev_committed = 0;
    size_t       feeds          = 0;
    const auto   t_start        = std::chrono::steady_clock::now();
    for (size_t off = 0; off < pcm.size(); off += chunk_samples) {
        const int n = static_cast<int>(std::min<size_t>(chunk_samples, pcm.size() - off));
        if (transcribe_stream_feed(session, pcm.data() + off, n, nullptr) != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: stream_feed at offset %zu\n", off);
            ++g_failures;
            break;
        }
        ++feeds;
        const int committed = transcribe_stream_n_committed_segments(session);
        CHECK(committed >= prev_committed);
        if (committed < prev_committed) {
            std::fprintf(stderr, "FAIL: committed shrank %d -> %d at feed %zu\n", prev_committed, committed, feeds);
        }
        prev_committed                                     = committed;
        // Rows and committed count agree; ids in range; no future times.
        const std::vector<transcribe_speaker_segment> rows = read_segments(session);
        CHECK(static_cast<int>(rows.size()) == committed);
        const int64_t received_ms = static_cast<int64_t>(off + n) * 1000 / 16000;
        for (const auto & r : rows) {
            CHECK(r.speaker_id >= 1 && r.speaker_id <= 4);
            CHECK(r.t0_ms <= received_ms && r.t1_ms <= received_ms);
        }
    }
    CHECK(transcribe_stream_finalize(session, nullptr) == TRANSCRIBE_OK);
    const double stream_s        = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const double realtime_factor = stream_s > 0.0 ? static_cast<double>(audio_ms) / 1000.0 / stream_s : 0.0;

    const std::vector<transcribe_speaker_segment> streamed = sorted_rows(read_segments(session));
    std::printf("stream: %zu rows (%zu feeds) in %.1f s (%.1fx realtime)\n", streamed.size(), feeds, stream_s,
                realtime_factor);
    std::printf("peak RSS: %.0f MB\n", peak_rss_bytes() / (1024.0 * 1024.0));

    if (!same_rows(streamed, offline_rows)) {
        std::fprintf(stderr, "FAIL: streamed rows != offline rows (%zu vs %zu)\n", streamed.size(),
                     offline_rows.size());
        for (size_t i = 0; i < std::max(streamed.size(), offline_rows.size()); ++i) {
            if (i < streamed.size() && i < offline_rows.size() &&
                (streamed[i].t0_ms != offline_rows[i].t0_ms || streamed[i].t1_ms != offline_rows[i].t1_ms ||
                 streamed[i].speaker_id != offline_rows[i].speaker_id)) {
                std::fprintf(stderr, "  row %zu: stream spk%d %lld..%lld vs offline spk%d %lld..%lld\n", i,
                             streamed[i].speaker_id, static_cast<long long>(streamed[i].t0_ms),
                             static_cast<long long>(streamed[i].t1_ms), offline_rows[i].speaker_id,
                             static_cast<long long>(offline_rows[i].t0_ms),
                             static_cast<long long>(offline_rows[i].t1_ms));
            }
        }
        ++g_failures;
    }

    transcribe_session_free(session);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "sortformer_push_stream_long: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sortformer_push_stream_long: OK\n");
    return 0;
}
