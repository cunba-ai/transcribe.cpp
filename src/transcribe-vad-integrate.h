// transcribe-vad-integrate.h - run_with_vad: the VAD chunk loop that wraps
// arch->run, plus the per-chunk result-merge helpers.
//
// run_with_vad is the VAD branch of run_one_inner (Task 8 wires it in). It:
//   1. ensures the audiocpp runtime is loaded (returns degraded=true to
//      signal "fall back to full-buffer decode" on load failure)
//   2. runs vad_invoke -> plan -> per-chunk arch->run loop
//   3. between chunks, folds the chunk's segment/word/token entries into the
//      global timeline (timestamp offset + cross-index fixup)
//   4. honors progress_cb (non-0 return) and abort_cb by stopping early
//      with partial results preserved

#ifndef TRANSCRIBE_VAD_INTEGRATE_H
#define TRANSCRIBE_VAD_INTEGRATE_H

#include "transcribe.h"
#include "transcribe-vad.h"  // chunk_plan, chunk_baseline used in merge helpers

#include <cstddef>

struct transcribe_session;
struct transcribe_run_params;

namespace transcribe::vad {

// Run the VAD chunk loop. On success returns TRANSCRIBE_OK and the session's
// result fields hold the merged multi-chunk result. On VAD-load failure
// sets *degraded=true (and the caller should run the original full-buffer
// arch->run). On any other error returns the status; partial results are
// preserved on abort.
//
// pcm/n_samples/params are the original transcribe_run inputs.
transcribe_status run_with_vad(struct transcribe_session *          session,
                               const float *                        pcm,
                               int                                  n_samples,
                               const struct transcribe_run_params * params,
                               bool &                               degraded);

// ---- Testable merge helpers (exposed for vad_merge_unit) -----------------

struct chunk_baseline {
    size_t n_segments = 0;
    size_t n_words    = 0;
    size_t n_tokens   = 0;
};

// Snapshot the session's entry counts (call before a chunk's arch->run).
chunk_baseline snapshot(const struct transcribe_session & s);

// Fold this chunk's newly-added entries into the global timeline: shift
// their timestamps by chunk.keep_span.start_ms and fix cross-references
// (word/token seg_index, segment first_word/first_token) from chunk-local
// to global indices.
void offset_chunk_results(struct transcribe_session &   s,
                          const chunk_baseline &        base,
                          const chunk_plan &            chunk);

// Trim the three result vectors back to baseline (drop a failed chunk's
// partial entries while keeping earlier chunks' results).
void rollback_to(struct transcribe_session & s, const chunk_baseline & base);

// Rebuild full_text / raw_text by space-joining segment texts.
void rebuild_full_text(struct transcribe_session & s);

}  // namespace transcribe::vad

#endif  // TRANSCRIBE_VAD_INTEGRATE_H
