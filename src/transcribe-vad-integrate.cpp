// transcribe-vad-integrate.cpp - VAD chunk loop + result merging.
//
// Task 7 lands the per-chunk result-merge helpers here. Task 8 adds the
// run_with_vad body (co-located with the run_one_inner change so the VAD
// branch and degrade path land together).

// Double-insurance: this TU is only added to the build when
// TRANSCRIBE_VAD_VIA_AUDIOCPP is on (see src/CMakeLists.txt). The #error
// catches a hand-edited build that tries to compile it unconditionally.
#if !defined(TRANSCRIBE_VAD_VIA_AUDIOCPP) || (TRANSCRIBE_VAD_VIA_AUDIOCPP == 0)
#  error "transcribe-vad-integrate.cpp requires -DTRANSCRIBE_VAD_VIA_AUDIOCPP=1"
#endif

#include "transcribe-vad-integrate.h"
#include "transcribe-vad-audiocpp.h"
#include "transcribe-vad.h"
#include "transcribe-arch.h"
#include "transcribe-log.h"
#include "transcribe-model.h"
#include "transcribe-session.h"

#include <algorithm>
#include <stdexcept>
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

transcribe_status run_with_vad(struct transcribe_session *          session,
                               const float *                        pcm,
                               int                                  n_samples,
                               const struct transcribe_run_params * params,
                               bool &                               degraded) {
    degraded = false;

    const transcribe_vad_mode mode = effective_mode(params);
    if (mode == TRANSCRIBE_VAD_OFF) {
        degraded = true;  // caller asked for VAD but field says OFF; degrade safely
        return TRANSCRIBE_OK;
    }

    // 1. Ensure the audiocpp runtime is loaded. On failure -> degrade.
    std::string       err;
    const auto &      vp = params->vad;
    if (!audiocpp::runtime::instance().ensure_loaded(
            vp.dll_path, mode, vp.backend, vp.device_id, vp.n_threads, err)) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                            "VAD: audiocpp load failed (%s); falling back to full-buffer decode",
                            err.c_str());
        degraded = true;
        return TRANSCRIBE_OK;
    }

    // 2. Run VAD -> speech segments.
    std::vector<time_span> speech;
    try {
        speech = vad_invoke(pcm, n_samples, vp);
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                            "VAD: invoke failed (%s); falling back to full-buffer decode", e.what());
        degraded = true;
        return TRANSCRIBE_OK;
    }

    // 3. No speech at all -> empty result, success.
    const int64_t total_ms = static_cast<int64_t>(n_samples) * 1000 / 16000;
    if (speech.empty()) {
        session->has_result = true;
        session->full_text.clear();
        session->raw_text.clear();
        return TRANSCRIBE_OK;
    }

    // 4. Plan decode windows. Resolve max_chunk_ms: explicit >0 wins, else
    //    family effective_max_audio_ms, else 30000.
    int64_t max_chunk = vp.max_chunk_ms;
    if (max_chunk <= 0) {
        transcribe_session_limits lim{};
        transcribe_session_limits_init(&lim);
        if (transcribe_session_get_limits(session, &lim) == TRANSCRIBE_OK && lim.effective_max_audio_ms > 0) {
            max_chunk = lim.effective_max_audio_ms;
        } else {
            max_chunk = 30000;
        }
    }
    const int64_t merge_gap = vp.merge_gap_ms != 0 ? vp.merge_gap_ms : 500;
    const int64_t padding   = vp.padding_ms >= 0 ? vp.padding_ms : 250;
    auto          chunks    = plan(speech, total_ms, max_chunk, merge_gap, padding);
    if (chunks.empty()) {
        degraded = true;
        return TRANSCRIBE_OK;
    }

    // 5. The VAD loop.
    auto *             arch  = session->model->arch;  // const Arch *
    const std::string  stage = std::string("asr+") + (arch->name ? arch->name : "?");
    const int64_t      n_total = static_cast<int64_t>(chunks.size());

    // Initial progress (0/N). Non-0 return = cancel -> abort with nothing yet.
    if (session->emit_progress(stage.c_str(), 0, n_total) != 0) {
        session->was_aborted = true;
        session->has_result  = true;
        return TRANSCRIBE_ERR_ABORTED;
    }

    for (int64_t i = 0; i < n_total; ++i) {
        // abort_cb check (existing mechanism, independent of progress_cb)
        if (session->poll_abort()) {
            rebuild_full_text(*session);
            session->has_result = true;
            return TRANSCRIBE_ERR_ABORTED;
        }

        // Slice PCM for this chunk's source_span (ms -> samples at 16 kHz).
        const int64_t s0 = chunks[static_cast<size_t>(i)].source_span.start_ms;
        const int64_t s1 = chunks[static_cast<size_t>(i)].source_span.end_ms;
        const int     off = static_cast<int>(s0 * 16000 / 1000);
        const int     len = static_cast<int>((s1 - s0) * 16000 / 1000);
        if (off < 0 || len <= 0 || static_cast<int64_t>(off) + len > n_samples) {
            // Defensive: plan() should prevent this, but never overrun.
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                                "VAD: chunk %lld out of bounds (off=%d len=%d n=%d); skipping",
                                static_cast<long long>(i), off, len, n_samples);
            continue;
        }
        const float * sub_pcm = pcm + off;

        // Snapshot before run; offset after. Reset per-chunk timing accumulators
        // so arch->run sees a clean slate for THIS chunk's decode timing.
        chunk_baseline base = snapshot(*session);
        session->t_mel_us    = 0;
        session->t_encode_us = 0;
        session->t_decode_us = 0;

        const transcribe_status st = arch->run(session, sub_pcm, len, params);
        if (st != TRANSCRIBE_OK) {
            // arch->run may have appended partial entries for this chunk;
            // roll them back so the global result only contains completed
            // chunks. Preserve earlier chunks' results.
            rollback_to(*session, base);
            rebuild_full_text(*session);
            session->has_result = !session->segments.empty();
            return st;
        }
        offset_chunk_results(*session, base, chunks[static_cast<size_t>(i)]);

        // Progress after chunk i. Non-0 = cancel -> preserve results so far.
        if (session->emit_progress(stage.c_str(), i + 1, n_total) != 0) {
            rebuild_full_text(*session);
            session->was_aborted = true;
            session->has_result  = true;
            return TRANSCRIBE_ERR_ABORTED;
        }
    }

    rebuild_full_text(*session);
    session->has_result = true;
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::vad
