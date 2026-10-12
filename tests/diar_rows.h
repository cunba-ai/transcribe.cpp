// diar_rows.h - speaker-row fixtures shared by the sortformer test
// binaries (diarize_unit, push_stream_unit, push_stream_long).
//
// Everything here is plumbing around transcribe_speaker_segment rows:
// reading them off a session, the canonical sorted order used by the
// parity comparisons, row equality on the (t0, t1, speaker) triple, env
// hygiene for the TRANSCRIBE_SORTFORMER_STREAM_* overrides, and the
// offline DIARIZE-role baseline the push-stream parity checks run
// against. CHECK macros stay per-file (house style); nothing here
// fails a test on its own — callers decide what a bad row means.
//
// Header-only by design: each test binary #includes it next to its own
// sources (include path wired per-target in tests/CMakeLists.txt, the
// same way src/ and examples/common headers are wired).

#pragma once

#include "transcribe.h"
#include "transcribe/diarize.h"
#include "transcribe/sortformer.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace diar_rows {

inline bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

inline void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    ::_putenv_s(key, value);
#else
    ::setenv(key, value, 1);
#endif
}

inline void unset_env(const char * key) {
#if defined(_WIN32)
    ::_putenv_s(key, "");  // MSVC CRT: empty value removes the variable
#else
    ::unsetenv(key);
#endif
}

// Committed rows exposed on an ASR-side session (push-stream surface).
inline std::vector<transcribe_speaker_segment> read_segments(const transcribe_session * session) {
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

// Rows currently held by a DIARIZE-role session (after a run).
inline std::vector<transcribe_speaker_segment> read_diarize_rows(const transcribe_diarize_session * s) {
    std::vector<transcribe_speaker_segment> rows;
    for (int i = 0; i < transcribe_diarize_n_segments(s); ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        transcribe_diarize_get_segment(s, i, &row);
        rows.push_back(row);
    }
    return rows;
}

// Canonical order for comparisons: by start time, then speaker.
inline std::vector<transcribe_speaker_segment> sorted_rows(std::vector<transcribe_speaker_segment> rows) {
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.t0_ms != b.t0_ms) {
            return a.t0_ms < b.t0_ms;
        }
        return a.speaker_id < b.speaker_id;
    });
    return rows;
}

// Row equality on the (t0, t1, speaker) triple; the confidence field is
// not a parity dimension (the offline extractor leaves it NaN).
inline bool same_rows(const std::vector<transcribe_speaker_segment> & a, const std::vector<transcribe_speaker_segment> & b) {
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

// Offline baseline through the DIARIZE role: a private diarize session,
// one run at `preset`, its rows. The push-stream parity checks compare
// their finalized rows against this at the same operating point. An
// empty return means the run failed — callers CHECK on non-empty when
// they expect rows.
inline std::vector<transcribe_speaker_segment> offline_diarize_rows(transcribe_model *             model,
                                                                    const std::vector<float> &     pcm,
                                                                    transcribe_sortformer_preset  preset) {
    transcribe_diarize_session * dia = nullptr;
    if (transcribe_diarize_session_init(model, nullptr, &dia) != TRANSCRIBE_OK) {
        return {};
    }
    transcribe_sortformer_diarize_ext ext;
    transcribe_sortformer_diarize_ext_init(&ext);
    ext.preset = preset;
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &ext.ext;
    const transcribe_status st = transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), &dp);
    std::vector<transcribe_speaker_segment> rows = st == TRANSCRIBE_OK ? read_diarize_rows(dia) : std::vector<transcribe_speaker_segment>{};
    transcribe_diarize_session_free(dia);
    return rows;
}

}  // namespace diar_rows
