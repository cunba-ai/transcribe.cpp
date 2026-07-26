// transcribe-vad.h - internal VAD types and pure planning functions.
//
// The public ABI types (transcribe_vad_mode, transcribe_vad_params,
// transcribe_vad_segment) live in include/transcribe.h. This header holds
// the internal-only pure-function API (vad::plan) and the struct_size
// negotiation helpers used by run_one_inner to detect whether the caller's
// run_params carries the vad field at all.

#ifndef TRANSCRIBE_VAD_H
#define TRANSCRIBE_VAD_H

#include "transcribe.h"

#include <cstdint>
#include <vector>

namespace transcribe::vad {

// Internal ms-resolution span (the public transcribe_vad_segment mirrors
// this but lives in the C ABI header).
struct time_span {
    int64_t start_ms   = 0;
    int64_t end_ms     = 0;
    float   confidence = 0.0f;
};

// A decode window produced by plan(). source_span is the PCM range fed to
// arch->run (includes padding/context); keep_span is the sub-range whose
// timestamps are attributed to this chunk (source_span minus padding).
struct chunk_plan {
    time_span source_span;
    time_span keep_span;
};

// Decide whether the caller's run_params actually carries the vad field.
// Returns false for callers built against an older header whose
// transcribe_run_params was shorter than the vad field's offset, so they
// silently get the pre-VAD full-buffer path. run_params may be NULL
// (means "all defaults" -> VAD off).
bool params_present(const struct transcribe_run_params * run_params);

// Read the effective vad_mode, treating a missing field as OFF.
transcribe_vad_mode effective_mode(const struct transcribe_run_params * run_params);

// Plan decode windows from speech segments.
//
//   speech       VAD-detected speech segments (ms), need not be sorted/merged
//   total_ms     total audio length in ms (windows cannot exceed [0, total_ms])
//   max_chunk_ms per-window ceiling; <=0 treated as "no ceiling" (one window)
//   merge_gap_ms segments separated by less than this are merged into one
//                window; <=0 means never merge
//   padding_ms   padding added to each window's source_span on both sides
//                (clamped to [0, total_ms]); keep_span excludes the padding
//
// Returns >=1 window for non-empty speech. Windows are sorted by start_ms,
// non-overlapping, each within [0, total_ms]. A window longer than
// max_chunk_ms is split at the largest internal gap between its constituent
// speech segments (or, if a single segment alone exceeds max_chunk_ms,
// hard-split at max_chunk_ms).
std::vector<chunk_plan> plan(const std::vector<time_span> & speech,
                             int64_t                        total_ms,
                             int64_t                        max_chunk_ms,
                             int64_t                        merge_gap_ms,
                             int64_t                        padding_ms);

}  // namespace transcribe::vad

#endif  // TRANSCRIBE_VAD_H
