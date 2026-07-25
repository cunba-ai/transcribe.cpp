// transcribe-vad-integrate.cpp - VAD chunk loop + result merging.
//
// Task 7 lands the per-chunk result-merge helpers here. Task 8 adds the
// run_with_vad body (co-located with the run_one_inner change so the VAD
// branch and degrade path land together).

#include "transcribe-vad-integrate.h"
#include "transcribe-vad.h"
#include "transcribe-session.h"

#include <algorithm>
#include <string>
#include <vector>

namespace transcribe::vad {

chunk_baseline snapshot(const transcribe_session & s) {
    return chunk_baseline{s.segments.size(), s.words.size(), s.tokens.size()};
}

// Offset the entries added since `base` by chunk.keep_span.start_ms, and
// fix up cross-references so word.seg_index / token.seg_index /
// segment.first_word / segment.first_token point at the GLOBAL arrays.
//
// Invariants maintained (matching how run() drivers populate the vectors
// with chunk-local indices before this fold):
//   - new segment/word/token t0_ms/t1_ms += keep_span.start_ms
//   - new token.seg_index += base.n_segments (was chunk-local, now global)
//   - new word.seg_index  += base.n_segments
//   - new word.first_token += base.n_tokens
//   - new segment.first_word  += base.n_words
//   - new segment.first_token += base.n_tokens
void offset_chunk_results(transcribe_session &   s,
                          const chunk_baseline & base,
                          const chunk_plan &     chunk) {
    const int64_t dt = chunk.keep_span.start_ms;

    // tokens
    for (size_t i = base.n_tokens; i < s.tokens.size(); ++i) {
        s.tokens[i].t0_ms += dt;
        s.tokens[i].t1_ms += dt;
        s.tokens[i].seg_index += static_cast<int>(base.n_segments);
    }
    // words
    for (size_t i = base.n_words; i < s.words.size(); ++i) {
        s.words[i].t0_ms += dt;
        s.words[i].t1_ms += dt;
        s.words[i].seg_index += static_cast<int>(base.n_segments);
        s.words[i].first_token += static_cast<int>(base.n_tokens);
    }
    // segments
    for (size_t i = base.n_segments; i < s.segments.size(); ++i) {
        s.segments[i].t0_ms += dt;
        s.segments[i].t1_ms += dt;
        s.segments[i].first_word += static_cast<int>(base.n_words);
        s.segments[i].first_token += static_cast<int>(base.n_tokens);
    }
}

void rollback_to(transcribe_session & s, const chunk_baseline & base) {
    if (s.segments.size() > base.n_segments) {
        s.segments.resize(base.n_segments);
    }
    if (s.words.size() > base.n_words) {
        s.words.resize(base.n_words);
    }
    if (s.tokens.size() > base.n_tokens) {
        s.tokens.resize(base.n_tokens);
    }
}

void rebuild_full_text(transcribe_session & s) {
    s.full_text.clear();
    for (size_t i = 0; i < s.segments.size(); ++i) {
        if (i > 0 && !s.full_text.empty() && s.full_text.back() != ' ') {
            s.full_text.push_back(' ');
        }
        s.full_text += s.segments[i].text;
    }
    s.raw_text = s.full_text;
}

// run_with_vad body is added in Task 8.

}  // namespace transcribe::vad
