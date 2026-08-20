// vad_merge_unit.cpp - per-chunk result-merge helper coverage.
//
// offset_chunk_results is the trickiest invariant in the VAD loop: it must
// shift per-chunk timestamps to global and fix every cross-reference
// (word->seg, token->seg, seg->first_word, seg->first_token). We drive it
// with a real (stack-constructed, no-model) session: simulate a chunk's
// arch->run appending chunk-local entries, call offset_chunk_results, and
// assert the global-timeline invariants. This is stronger than documenting
// the contract — it exercises the actual helper.

#include "transcribe-session.h"
#include "transcribe-vad-integrate.h"
#include "transcribe-vad.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using transcribe::vad::chunk_baseline;
using transcribe::vad::chunk_plan;
using transcribe::vad::offset_chunk_results;
using transcribe::vad::rebuild_full_text;
using transcribe::vad::rollback_to;
using transcribe::vad::snapshot;
using transcribe::vad::time_span;

// Simulate arch->run for one chunk: append a segment + word + token with
// CHUNK-LOCAL timestamps and indices, exactly as a family run() driver would.
// The merge helper then shifts them to global.
void append_chunk_local(transcribe_session & s, int64_t seg_t0, int64_t seg_t1, const char * text) {
    using Seg  = transcribe_session::SegmentEntry;
    using Word = transcribe_session::WordEntry;
    using Tok  = transcribe_session::TokenEntry;

    const int seg_idx_local = static_cast<int>(s.segments.size());  // grows by 1 below

    Tok tok;
    tok.text       = "t";
    tok.t0_ms      = seg_t0;
    tok.t1_ms      = seg_t1;
    tok.seg_index  = 0;  // chunk-local: this chunk's first segment
    tok.word_index = 0;
    s.tokens.push_back(tok);

    Word w;
    w.text        = text;
    w.t0_ms       = seg_t0;
    w.t1_ms       = seg_t1;
    w.seg_index   = 0;  // chunk-local
    w.first_token = 0;  // chunk-local (relative to this chunk's token range)
    w.n_tokens    = 1;
    s.words.push_back(w);

    Seg seg;
    seg.text        = text;
    seg.t0_ms       = seg_t0;
    seg.t1_ms       = seg_t1;
    seg.first_word  = 0;  // chunk-local
    seg.n_words     = 1;
    seg.first_token = 0;  // chunk-local
    seg.n_tokens    = 1;
    (void) seg_idx_local;
    s.segments.push_back(seg);
}

chunk_plan make_chunk(int64_t keep_start, int64_t keep_end) {
    chunk_plan cp;
    cp.keep_span   = time_span{ keep_start, keep_end, 1.0f };
    cp.source_span = time_span{ keep_start, keep_end, 1.0f };
    return cp;
}

void test_two_chunks_timestamps_global() {
    transcribe_session s;

    // Chunk 1 at offset 1000ms: one segment [0,2000] local -> [1000,3000] global
    chunk_baseline b1 = snapshot(s);
    append_chunk_local(s, 0, 2000, "hello");
    offset_chunk_results(s, b1, make_chunk(1000, 3000));

    // Chunk 2 at offset 5000ms: one segment [0,1500] local -> [5000,6500] global
    chunk_baseline b2 = snapshot(s);
    append_chunk_local(s, 0, 1500, "world");
    offset_chunk_results(s, b2, make_chunk(5000, 6500));

    CHECK(s.segments.size() == 2);
    CHECK(s.segments[0].t0_ms == 1000);
    CHECK(s.segments[0].t1_ms == 3000);
    CHECK(s.segments[1].t0_ms == 5000);
    CHECK(s.segments[1].t1_ms == 6500);

    // word seg_index must be GLOBAL (0 and 1, not both 0)
    CHECK(s.words[0].seg_index == 0);
    CHECK(s.words[1].seg_index == 1);

    // segment.first_word / first_token must be GLOBAL
    CHECK(s.segments[0].first_word == 0);
    CHECK(s.segments[0].first_token == 0);
    CHECK(s.segments[1].first_word == 1);
    CHECK(s.segments[1].first_token == 1);

    // token seg_index global
    CHECK(s.tokens[0].seg_index == 0);
    CHECK(s.tokens[1].seg_index == 1);
}

void test_rebuild_full_text_joins_with_space() {
    transcribe_session s;
    append_chunk_local(s, 0, 1000, "foo");
    append_chunk_local(s, 0, 1000, "bar");
    rebuild_full_text(s);
    // Segments are space-joined so multi-chunk output reads naturally.
    CHECK(s.full_text == "foo bar");
    CHECK(s.raw_text == "foo bar");
}

void test_rebuild_full_text_single_segment_no_leading_space() {
    transcribe_session s;
    append_chunk_local(s, 0, 1000, "solo");
    rebuild_full_text(s);
    CHECK(s.full_text == "solo");  // no leading/trailing space
}

void test_rollback_drops_failed_chunk() {
    transcribe_session s;
    append_chunk_local(s, 0, 1000, "keep");  // chunk 1
    chunk_baseline b = snapshot(s);
    append_chunk_local(s, 0, 1000, "drop");  // chunk 2 (will fail)
    CHECK(s.segments.size() == 2);
    rollback_to(s, b);
    CHECK(s.segments.size() == 1);
    CHECK(s.segments[0].text == "keep");
    CHECK(s.words.size() == 1);
    CHECK(s.tokens.size() == 1);
}

void test_empty_chunk_no_change() {
    transcribe_session s;
    chunk_baseline     b = snapshot(s);
    // No append — offset over an empty delta is a no-op.
    offset_chunk_results(s, b, make_chunk(5000, 6000));
    CHECK(s.segments.empty());
    CHECK(s.words.empty());
    CHECK(s.tokens.empty());
}

}  // namespace

int main() {
    test_two_chunks_timestamps_global();
    test_rebuild_full_text_joins_with_space();
    test_rebuild_full_text_single_segment_no_leading_space();
    test_rollback_drops_failed_chunk();
    test_empty_chunk_no_change();
    if (g_failures == 0) {
        std::printf("vad_merge_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_merge_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
