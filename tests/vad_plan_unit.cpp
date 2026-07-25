// vad_plan_unit.cpp - exhaustive tests for vad::plan() (pure function).
//
// plan() turns speech segments into decode windows. This is the most
// logic-dense part of the VAD integration and has no dll dependency, so we
// test it hard: merging, splitting, padding clamping, empty input, single
// oversized segment, etc.

#include "transcribe-vad.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

using transcribe::vad::chunk_plan;
using transcribe::vad::plan;
using transcribe::vad::time_span;

time_span ms(int64_t a, int64_t b) { return time_span{a, b, 1.0f}; }

void assert_windows_sorted_nonoverlapping(const std::vector<chunk_plan> & w) {
    for (size_t i = 1; i < w.size(); ++i) {
        CHECK(w[i].keep_span.start_ms >= w[i - 1].keep_span.end_ms);
    }
}

void assert_keep_within_source(const std::vector<chunk_plan> & w) {
    for (const auto & c : w) {
        CHECK(c.source_span.start_ms <= c.keep_span.start_ms);
        CHECK(c.source_span.end_ms >= c.keep_span.end_ms);
    }
}

void test_empty_input() {
    auto w = plan({}, 10000, 15000, 500, 250);
    CHECK(w.empty());
}

void test_single_segment_one_window() {
    auto w = plan({ms(1000, 3000)}, 10000, 15000, 500, 250);
    CHECK(w.size() == 1);
    CHECK(w[0].keep_span.start_ms == 1000);
    CHECK(w[0].keep_span.end_ms == 3000);
    // padding 250 clamped within [0,10000]
    CHECK(w[0].source_span.start_ms == 750);
    CHECK(w[0].source_span.end_ms == 3250);
}

void test_merge_small_gap() {
    // two segs 1000ms apart, merge_gap=500 -> NOT merged (gap >= merge_gap)
    auto w = plan({ms(1000, 2000), ms(2500, 3500)}, 10000, 15000, 500, 0);
    CHECK(w.size() == 2);
    // two segs 300ms apart, merge_gap=500 -> merged into one window
    auto w2 = plan({ms(1000, 2000), ms(2300, 3300)}, 10000, 15000, 500, 0);
    CHECK(w2.size() == 1);
    CHECK(w2[0].keep_span.start_ms == 1000);
    CHECK(w2[0].keep_span.end_ms == 3300);
}

void test_padding_clamped_at_boundaries() {
    // seg at very start: padding clamped to 0
    auto w = plan({ms(0, 1000)}, 10000, 15000, 500, 500);
    CHECK(w.size() == 1);
    CHECK(w[0].source_span.start_ms == 0);  // 0-500 clamped
    CHECK(w[0].source_span.end_ms == 1500);
    // seg at very end: padding clamped to total_ms
    auto w2 = plan({ms(9000, 10000)}, 10000, 15000, 500, 500);
    CHECK(w2[0].source_span.start_ms == 8500);
    CHECK(w2[0].source_span.end_ms == 10000);
}

void test_split_oversized_window() {
    // one segment 40s long, ceiling 15s -> split into [0,15000],[15000,30000],[30000,40000]
    auto w = plan({ms(0, 40000)}, 40000, 15000, 500, 0);
    CHECK(w.size() == 3);
    CHECK(w[0].keep_span.start_ms == 0);
    CHECK(w[0].keep_span.end_ms == 15000);
    CHECK(w[1].keep_span.start_ms == 15000);
    CHECK(w[1].keep_span.end_ms == 30000);
    CHECK(w[2].keep_span.start_ms == 30000);
    CHECK(w[2].keep_span.end_ms == 40000);
    assert_windows_sorted_nonoverlapping(w);
}

void test_split_prefers_internal_gap() {
    // Two segs separated by a 1000ms gap. With merge_gap=2000 (> gap) they
    // merge into one 20s window; ceiling 15s forces a split. The planner
    // should split at the speech-segment boundary inside the window (9500,
    // the first seg's end = the start of the internal gap), NOT hard-cut
    // at 15000 — splitting at a boundary keeps each decode window voiced.
    auto w = plan({ms(0, 9500), ms(10500, 20000)}, 20000, 15000, 2000, 0);
    CHECK(w.size() == 2);
    CHECK(w[0].keep_span.start_ms == 0);
    CHECK(w[0].keep_span.end_ms == 9500);  // cut at seg boundary, not 15000
    CHECK(w[1].keep_span.start_ms == 9500);
    CHECK(w[1].keep_span.end_ms == 20000);
}

void test_split_hard_cut_when_no_gap() {
    // A single 20s segment with no internal gap, ceiling 15s: must hard-cut
    // at 15000 (no boundary to prefer).
    auto w = plan({ms(0, 20000)}, 20000, 15000, 500, 0);
    CHECK(w.size() == 2);
    CHECK(w[0].keep_span.start_ms == 0);
    CHECK(w[0].keep_span.end_ms == 15000);
    CHECK(w[1].keep_span.start_ms == 15000);
    CHECK(w[1].keep_span.end_ms == 20000);
}

void test_no_ceiling_one_window() {
    // max_chunk_ms <= 0 -> never split
    auto w = plan({ms(0, 50000)}, 50000, 0, 500, 0);
    CHECK(w.size() == 1);
    CHECK(w[0].keep_span.start_ms == 0);
    CHECK(w[0].keep_span.end_ms == 50000);
}

void test_segments_clipped_to_total() {
    // seg extends past total_ms -> clipped
    auto w = plan({ms(8000, 12000)}, 10000, 15000, 500, 0);
    CHECK(w.size() == 1);
    CHECK(w[0].keep_span.start_ms == 8000);
    CHECK(w[0].keep_span.end_ms == 10000);  // clipped
}

void test_degenerate_segment_dropped() {
    // zero-length and inverted segs dropped
    auto w = plan({ms(1000, 1000), ms(3000, 2000), ms(1000, 2000)}, 10000, 15000, 500, 0);
    CHECK(w.size() == 1);
    CHECK(w[0].keep_span.start_ms == 1000);
    CHECK(w[0].keep_span.end_ms == 2000);
}

void test_many_segments_unsorted() {
    // input unsorted -> output sorted, non-overlapping keep spans
    std::vector<time_span> segs = {ms(5000, 6000), ms(0, 1000), ms(2000, 3000)};
    auto                   w = plan(segs, 10000, 15000, 500, 0);
    assert_windows_sorted_nonoverlapping(w);
    assert_keep_within_source(w);
    CHECK(w.size() == 3);
    CHECK(w[0].keep_span.start_ms == 0);
}

}  // namespace

int main() {
    test_empty_input();
    test_single_segment_one_window();
    test_merge_small_gap();
    test_padding_clamped_at_boundaries();
    test_split_oversized_window();
    test_split_prefers_internal_gap();
    test_split_hard_cut_when_no_gap();
    test_no_ceiling_one_window();
    test_segments_clipped_to_total();
    test_degenerate_segment_dropped();
    test_many_segments_unsorted();

    if (g_failures == 0) {
        std::printf("vad_plan_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_plan_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
