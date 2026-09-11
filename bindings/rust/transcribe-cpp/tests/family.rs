//! Per-family stream-extension happy-path tests (parakeet cache-aware,
//! parakeet buffered, voxtral realtime). Each proves the extension end to end:
//! the typed options materialize a kind-tagged struct, the model ACCEPTS the
//! kind on its stream slot, `stream_begin` consumes it, and a short feed +
//! finalize emits text. NOT a transcription-accuracy check (that is the family
//! port's C-level / WER job) — a short feed keeps these fast even for the 4B
//! voxtral model, so we assert the stream ran and produced non-empty text.
//!
//! Mirrors Swift's `FamilyStreamTests`. Each gates on its own per-family GGUF
//! and skips cleanly when absent (parakeet runs in CI once the canary repos
//! exist; voxtral is local-only — ~2.5 GB is too heavy for CI). The
//! wrong-family reject is already covered by
//! `streaming::stream_family_extension_accepted_or_rejected`.

mod common;

use transcribe_cpp::sys::{
    TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
    TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM, TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM,
};
use transcribe_cpp::{
    ExtSlot, Model, ParakeetBufferedStreamOptions, ParakeetStreamOptions, RunExtension, RunOptions,
    SortformerPreset, SortformerStreamOptions, SpeakerSegment, Stream, StreamExtension,
    StreamOptions, VoxtralRealtimeStreamOptions,
};

/// Feed the first ~2 s of `pcm` in 100 ms chunks, finalize, and return
/// `(is_final, full_text)` (owned copies). A short feed keeps the test fast and
/// proves consumption without a full transcription.
fn short_feed_text(stream: &mut Stream<'_>, pcm: &[f32]) -> (bool, String) {
    let clip = &pcm[..pcm.len().min(2 * 16_000)];
    for frame in clip.chunks(1600) {
        stream.feed(frame).expect("feed");
    }
    let is_final = stream.finalize().expect("finalize").is_final;
    (is_final, stream.text().full)
}

#[test]
fn parakeet_cache_aware_acceptance_discriminates() {
    let Some(model_path) = common::smoke_parakeet_stream_model() else {
        eprintln!("skip parakeet_cache_aware_acceptance_discriminates: model absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    // The header's documented discrimination: the cache-aware variant accepts
    // PARAKEET_STREAM and rejects PARAKEET_BUFFERED_STREAM.
    assert!(
        model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM),
        "cache-aware should accept PARAKEET_STREAM"
    );
    assert!(
        !model.accepts_ext(
            ExtSlot::Stream,
            TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM
        ),
        "cache-aware should reject PARAKEET_BUFFERED_STREAM"
    );
}

#[test]
fn parakeet_cache_aware_streams_with_extension() {
    let (Some(model_path), Some(pcm)) =
        (common::smoke_parakeet_stream_model(), common::smoke_audio())
    else {
        eprintln!("skip parakeet_cache_aware_streams_with_extension: model/audio absent");
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let opts = StreamOptions {
        // att_context_right = -1 selects the model's default (max-accuracy) menu entry.
        family: Some(StreamExtension::ParakeetStream(ParakeetStreamOptions {
            att_context_right: Some(-1),
        })),
        ..Default::default()
    };
    let mut stream = session.stream(&RunOptions::default(), &opts).unwrap();
    let (is_final, text) = short_feed_text(&mut stream, &pcm);
    assert!(is_final);
    assert!(
        !text.trim().is_empty(),
        "cache-aware stream produced no text"
    );
}

#[test]
fn parakeet_buffered_streams_with_extension() {
    let (Some(model_path), Some(pcm)) = (
        common::smoke_parakeet_buffered_model(),
        common::smoke_audio(),
    ) else {
        eprintln!("skip parakeet_buffered_streams_with_extension: model/audio absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(
        model.accepts_ext(
            ExtSlot::Stream,
            TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM
        ),
        "buffered model should accept PARAKEET_BUFFERED_STREAM"
    );
    // Defaults (left/chunk/right = None -> -1) resolve to the model's menu
    // default (L=5600/C=1040/R=1040). An explicit override must be an 80 ms
    // multiple AND land on a tuple in the training menu, else stream_begin
    // returns INVALID_ARG — so the path-proving choice is the default.
    let mut session = model.session().unwrap();
    let opts = StreamOptions {
        family: Some(StreamExtension::ParakeetBuffered(
            ParakeetBufferedStreamOptions::default(),
        )),
        ..Default::default()
    };
    let mut stream = session.stream(&RunOptions::default(), &opts).unwrap();
    let (is_final, text) = short_feed_text(&mut stream, &pcm);
    assert!(is_final);
    assert!(!text.trim().is_empty(), "buffered stream produced no text");
}

#[test]
fn voxtral_realtime_streams_with_extension() {
    let (Some(model_path), Some(pcm)) = (common::smoke_voxtral_model(), common::smoke_audio())
    else {
        eprintln!("skip voxtral_realtime_streams_with_extension: model/audio absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(
        model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM),
        "voxtral model should accept VOXTRAL_REALTIME_STREAM"
    );
    let mut session = model.session().unwrap();
    let opts = StreamOptions {
        family: Some(StreamExtension::VoxtralRealtime(
            VoxtralRealtimeStreamOptions {
                num_delay_tokens: Some(4),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let mut stream = session.stream(&RunOptions::default(), &opts).unwrap();
    let (is_final, text) = short_feed_text(&mut stream, &pcm);
    assert!(is_final);
    assert!(!text.trim().is_empty(), "voxtral produced no text");
}

// --- sortformer push-audio diarization (SFPS, stream slot) ----------------

#[test]
fn sortformer_acceptance_discriminates() {
    let Some(model_path) = common::smoke_sortformer_model() else {
        eprintln!("skip sortformer_acceptance_discriminates: model absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(
        model.capabilities().supports_streaming,
        "sortformer is natively streaming"
    );
    assert!(
        model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM),
        "sortformer should accept SORTFORMER_PUSH_STREAM on the stream slot"
    );
    assert!(
        !model.accepts_ext(ExtSlot::Run, TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM),
        "SORTFORMER_PUSH_STREAM is stream-slot-only"
    );
}

#[test]
fn sortformer_streams_with_extension() {
    let (Some(model_path), Some(pcm)) =
        (common::smoke_sortformer_model(), common::smoke_diar_audio())
    else {
        eprintln!("skip sortformer_streams_with_extension: model/audio absent");
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();

    // 12 s mix, 480 ms chunks: the LOW_LATENCY geometry emits mid-stream, so
    // tentative turns must appear at some edge and all close at finalize.
    let mut saw_tentative = false;
    let mut stream = session
        .stream(
            &RunOptions::default(),
            &StreamOptions {
                family: Some(StreamExtension::Sortformer(SortformerStreamOptions {
                    preset: Some(SortformerPreset::LowLatency),
                })),
                ..StreamOptions::default()
            },
        )
        .unwrap();
    for frame in pcm.chunks(7_680) {
        stream.feed(frame).expect("feed");
        if !stream.tentative_speaker_segments().is_empty() {
            saw_tentative = true;
        }
    }
    stream.finalize().unwrap();
    let low = stream.speaker_segments();
    drop(stream);
    assert!(!low.is_empty(), "the oracle mix has two speakers");
    assert!(
        saw_tentative,
        "LOW_LATENCY edges are mid-speech on this mix"
    );
    for t in &low {
        assert!((1..=4).contains(&t.speaker_id));
    }

    // The offline run at the SAME operating point must reach the same
    // segmentation (frame-exact here: the incremental mel and window
    // scheduling are bit-identical to the batch path on a given build).
    let offline = session
        .run(
            &pcm,
            &RunOptions {
                family: Some(RunExtension::Sortformer(SortformerStreamOptions {
                    preset: Some(SortformerPreset::LowLatency),
                })),
                ..RunOptions::default()
            },
        )
        .expect("offline run at LOW_LATENCY");
    let key = |r: &SpeakerSegment| (r.t0_ms, r.t1_ms, r.speaker_id);
    let mut low_sorted = low.clone();
    low_sorted.sort_by_key(key);
    let mut offline_sorted = offline.speaker_segments.clone();
    offline_sorted.sort_by_key(key);
    // Compare the semantic fields only: `p` is NaN (not produced), and
    // NaN != NaN would fail a derived PartialEq.
    let rows_eq = low_sorted.len() == offline_sorted.len()
        && low_sorted
            .iter()
            .zip(&offline_sorted)
            .all(|(a, b)| key(a) == key(b));
    assert!(
        rows_eq,
        "streaming == offline segmentation at the same operating point"
    );
}
