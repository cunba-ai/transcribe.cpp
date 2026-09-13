/*
 * include/transcribe/sortformer.h - Sortformer-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming-operating-point run extension (RUN slot), the push-audio
 * stream extension (STREAM slot), and their init functions plus the
 * push-stream tentative-turn accessors.
 *
 * Sortformer (diar_streaming_sortformer_4spk-v2.1) is a diarization-only model: a run
 * produces no text; the product is the who-spoke-when rows read back via
 * transcribe_n_speaker_segments / transcribe_get_speaker_segment
 * (TRANSCRIBE_FEATURE_DIARIZATION). The compute core is streaming
 * (AOSC speaker cache + FIFO); the batch transcribe_run over a whole
 * recording is exposed through the RUN-slot operating-point extension,
 * and the push-audio live session through the STREAM-slot extension
 * (transcribe_stream_begin / _feed / _finalize). Both take the same
 * preset enum; the two kinds are slot-exclusive.
 *
 * Probe via transcribe_model_accepts_ext_kind(model, slot, kind)
 * before pointing transcribe_run_params::family (RUN slot) or
 * transcribe_stream_params::family (STREAM slot) at the struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_SORTFORMER_H
#define TRANSCRIBE_SORTFORMER_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'SFST' little-endian = 0x54534653 */
#define TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM 0x54534653u

/* 'SFPS' little-endian = 0x53504653 */
#define TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM 0x53504653u

/*
 * Streaming operating point (latency / accuracy trade-off).
 *
 * The model processes audio in fixed chunks and carries speaker identity
 * across chunks in a bounded cache; the operating point sets the chunk
 * geometry. Each named preset is a jointly-tuned bundle published by the
 * upstream model (chunk length, lookahead, FIFO and speaker-cache
 * geometry) - the menu is discrete, not a continuous latency dial, and
 * only these bundles are accuracy-validated (AMI DER, see
 * docs/porting/families/sortformer.md).
 *
 *   DEFAULT             The GGUF-shipped checkpoint configuration.
 *   VERY_HIGH_LATENCY   ~30.4 s algorithmic lookahead (chunk 340 + rc 40
 *                       frames @ 80 ms). Highest accuracy; the published
 *                       operating point for offline file processing.
 *   HIGH_LATENCY        ~10.0 s lookahead (chunk 124 + rc 1).
 *   LOW_LATENCY         ~1.04 s lookahead (chunk 6 + rc 7). The
 *                       real-time operating point. Note: much higher
 *                       compute per audio second than the larger chunks
 *                       (many small windows); see the family doc for
 *                       measured throughput.
 *
 * Values outside the enum range are rejected by transcribe_run with
 * TRANSCRIBE_ERR_INVALID_ARG before the previous result is cleared.
 */
typedef enum {
    TRANSCRIBE_SORTFORMER_PRESET_DEFAULT           = 0,
    TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY = 1,
    TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY      = 2,
    TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY       = 3,
} transcribe_sortformer_preset;

struct transcribe_sortformer_stream_ext {
    struct transcribe_ext        ext;
    transcribe_sortformer_preset preset;
};

/* Fills ext.size/kind and preset = DEFAULT (GGUF-shipped cfg). */
TRANSCRIBE_API void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * ext);

/*
 * Push-audio stream extension (TRANSCRIBE_EXT_SLOT_STREAM; point
 * transcribe_stream_params::family at it). Carries the same operating-
 * point preset as the RUN-slot extension; DEFAULT keeps the GGUF-shipped
 * chunk geometry. The extension is copied out before
 * transcribe_stream_begin returns (caller-owned storage).
 *
 * The stream produces no text. After every feed,
 * transcribe_n_speaker_segments / transcribe_get_speaker_segment expose
 * the committed turns: append-only for the stream's life, timestamps in
 * absolute ms from stream start, speaker ids global and in arrival
 * order. The trailing turns that later audio may still extend are
 * tentative and read through the two accessors below; finalize closes
 * them into the committed set.
 */
struct transcribe_sortformer_push_stream_ext {
    struct transcribe_ext        ext;
    transcribe_sortformer_preset preset;
};

/* Fills ext.size/kind and preset = DEFAULT (GGUF-shipped cfg). */
TRANSCRIBE_API void transcribe_sortformer_push_stream_ext_init(struct transcribe_sortformer_push_stream_ext * ext);

/*
 * Tentative (still-open) speaker turns on a sortformer push-audio
 * stream: at most one open row per speaker, subject to revision or
 * extension by later feeds. 0 on a NULL session, when the session is
 * not a sortformer push-audio stream, or after finalize closed every
 * open turn into the committed set.
 */
TRANSCRIBE_API int transcribe_sortformer_push_stream_n_tentative(const struct transcribe_session * session);

/*
 * Read one tentative turn into caller-owned storage. Same contract as
 * transcribe_get_speaker_segment: INVALID_ARG on NULL out,
 * BAD_STRUCT_SIZE on a zero/short struct_size, otherwise OK with the
 * struct written when i is in range and left zero-initialized when it
 * is not.
 */
TRANSCRIBE_API transcribe_status
transcribe_sortformer_push_stream_get_tentative(const struct transcribe_session *   session,
                                                int                                 i,
                                                struct transcribe_speaker_segment * out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_SORTFORMER_H */
