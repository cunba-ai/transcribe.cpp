// transcribe-vad.cpp - struct_size negotiation + (Task 6) plan() impl.

#include "transcribe-vad.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace transcribe::vad {

bool params_present(const struct transcribe_run_params * run_params) {
    if (run_params == nullptr) {
        return false;
    }
    // struct_size is the caller's declared size. The vad field exists only
    // if the caller's struct is large enough to hold it. offsetof gives the
    // byte offset of vad within the (fully-defined) struct as this library
    // was compiled; a caller built against an older, shorter header sets a
    // smaller struct_size and is treated as not having the field.
    const size_t vad_off = offsetof(struct transcribe_run_params, vad);
    return run_params->struct_size >= vad_off + sizeof(struct transcribe_vad_params);
}

transcribe_vad_mode effective_mode(const struct transcribe_run_params * run_params) {
    if (!params_present(run_params)) {
        return TRANSCRIBE_VAD_OFF;
    }
    return run_params->vad.mode;
}

std::vector<chunk_plan> plan(const std::vector<time_span> & speech,
                             int64_t                        total_ms,
                             int64_t                        max_chunk_ms,
                             int64_t                        merge_gap_ms,
                             int64_t                        padding_ms) {
    std::vector<chunk_plan> out;
    if (total_ms <= 0 || speech.empty()) {
        return out;
    }

    // 1. Sort + clip speech to [0, total_ms], drop zero/negative-length.
    std::vector<time_span> segs;
    segs.reserve(speech.size());
    for (auto s : speech) {
        if (s.end_ms <= s.start_ms) {
            continue;
        }
        s.start_ms = std::max<int64_t>(s.start_ms, 0);
        s.end_ms   = std::min<int64_t>(s.end_ms, total_ms);
        if (s.end_ms <= s.start_ms) {
            continue;
        }
        segs.push_back(s);
    }
    if (segs.empty()) {
        return out;
    }
    std::sort(segs.begin(), segs.end(),
              [](const time_span & a, const time_span & b) { return a.start_ms < b.start_ms; });

    // 2. Merge segments separated by < merge_gap_ms into windows. A window
    // is the union [first.start, last.end] of a run of near-adjacent segs.
    // (merge_gap_ms <= 0 means never merge — each seg is its own window
    // candidate, subject to the max_chunk split below.) We also record the
    // internal gap end-offsets so step 3 can prefer them as split points.
    struct window {
        int64_t               start_ms;
        int64_t               end_ms;
        std::vector<int64_t>  gap_offsets;  // end_ms of each merged-in predecessor seg
    };
    std::vector<window> windows;
    window              cur{segs[0].start_ms, segs[0].end_ms, {}};
    for (size_t i = 1; i < segs.size(); ++i) {
        const int64_t gap = segs[i].start_ms - cur.end_ms;
        if (merge_gap_ms > 0 && gap < merge_gap_ms) {
            if (gap > 0) {
                cur.gap_offsets.push_back(cur.end_ms);  // candidate split point
            }
            cur.end_ms = segs[i].end_ms;
        } else {
            windows.push_back(std::move(cur));
            cur = window{segs[i].start_ms, segs[i].end_ms, {}};
        }
    }
    windows.push_back(std::move(cur));

    // 3. Split windows exceeding max_chunk_ms (<=0 means no ceiling). Prefer
    // splitting at the largest internal gap that keeps the left half <=
    // ceiling; if no usable gap exists or a single seg exceeds the ceiling,
    // hard-split at max_chunk_ms.
    const bool has_ceiling = max_chunk_ms > 0;
    std::vector<window> split;
    for (auto & w : windows) {
        if (!has_ceiling || (w.end_ms - w.start_ms) <= max_chunk_ms) {
            split.push_back(std::move(w));
            continue;
        }
        int64_t              win_start = w.start_ms;
        auto                 gaps = w.gap_offsets;  // copy; consumed from front
        size_t               gi = 0;
        while (win_start < w.end_ms && (w.end_ms - win_start) > max_chunk_ms) {
            int64_t cut = win_start + max_chunk_ms;  // hard-cut default
            // Prefer a gap in (win_start, win_start + max_chunk_ms] closest
            // to the ceiling (largest gap keeps windows full).
            for (; gi < gaps.size() && gaps[gi] <= win_start + max_chunk_ms; ++gi) {
                if (gaps[gi] > win_start) {
                    cut = gaps[gi];
                }
            }
            split.push_back(window{win_start, cut, {}});
            win_start = cut;
        }
        if (win_start < w.end_ms) {
            split.push_back(window{win_start, w.end_ms, {}});
        }
    }

    // 4. Build chunk_plan with padding. source_span = [start-pad, end+pad]
    // clamped to [0, total_ms]; keep_span = [start, end] (no padding).
    const int64_t pad = padding_ms > 0 ? padding_ms : 0;
    for (const auto & w : split) {
        chunk_plan cp;
        cp.keep_span.start_ms     = w.start_ms;
        cp.keep_span.end_ms       = w.end_ms;
        cp.keep_span.confidence   = 1.0f;
        cp.source_span.start_ms   = std::max<int64_t>(w.start_ms - pad, 0);
        cp.source_span.end_ms     = std::min<int64_t>(w.end_ms + pad, total_ms);
        cp.source_span.confidence = 1.0f;
        out.push_back(cp);
    }
    return out;
}

}  // namespace transcribe::vad
