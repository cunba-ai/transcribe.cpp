//! Embedded Silero VAD (`transcribe_model_load_embedded_vad`): load with no
//! model file on disk, VAD info, and a run on synthetic audio with a speech
//! burst in the middle. The bytes live in the native library, so — unlike
//! `tests/vad.rs` — this needs no fixture and runs wherever the library is
//! loaded. In `dynload` posture without a loaded library it skips cleanly.

mod common;

use transcribe_cpp::{Backend, Model, ModelOptions, Role, VadOptions};

#[test]
fn embedded_vad_load_info_run() {
    // The trampolines would return type-shaped errors through a null table;
    // the skip keeps that indistinguishable-from-failure mode out of this
    // test (common::library_loaded, shared with the fixture resolvers).
    if !common::library_loaded() {
        eprintln!("skip embedded_vad_load_info_run: native library not loaded (dynload posture)");
        return;
    }
    common::ensure_backends();

    let opts = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    let model = match Model::load_embedded_vad_with(&opts) {
        Ok(model) => model,
        Err(transcribe_cpp::Error::NotImplemented(msg)) => {
            // Library built with TRANSCRIBE_EMBED_VAD_MODEL=OFF.
            eprintln!("skip embedded_vad_load_info_run: {msg}");
            return;
        }
        Err(e) => panic!("load_embedded_vad: {e}"),
    };
    assert!(model.roles().contains(Role::Vad));
    assert_eq!(model.arch(), "silero_vad");
    let info = model.vad_info().unwrap();
    assert_eq!(info.sample_rate, 16000);
    assert_eq!(info.frame_samples, 512);

    // ~2 s: near-silence, 1 s speech-like harmonic burst (140 Hz stack
    // under 3 Hz syllabic AM — the reference model rejects pure tones and
    // white noise), near-silence.
    let pcm: Vec<f32> = (0..32_000usize)
        .map(|i| {
            let noise = (((i.wrapping_mul(1664525).wrapping_add(1013904223)) >> 8) as f32
                / 16_777_216.0
                - 0.5)
                * 0.01;
            if !(8000..24000).contains(&i) {
                return noise;
            }
            let tb = (i - 8000) as f32 / 16000.0;
            let env = 0.5 * (1.0 + (6.283_185_3 * 3.0 * tb - 1.570_796_4).sin());
            let harm = (6.283_185_3 * 140.0 * tb).sin()
                + 0.5 * (6.283_185_3 * 280.0 * tb).sin()
                + 0.25 * (6.283_185_3 * 560.0 * tb).sin()
                + 0.3 * (6.283_185_3 * 1820.0 * tb).sin()
                + 0.2 * (6.283_185_3 * 3360.0 * tb).sin();
            noise + 0.4 * env * harm
        })
        .collect();

    let mut vad = model.vad_session().unwrap();
    let result = vad.run(&pcm, &VadOptions::default()).unwrap();
    assert_eq!(result.probs.len(), (pcm.len() + 511) / 512);
    for &p in &result.probs {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }
    assert!(!result.segments.is_empty());
    let burst_lo = 8000 + 4000;
    let burst_hi = 8000 + 12000;
    assert!(
        result
            .segments
            .iter()
            .any(|s| s.start_sample < burst_lo && s.end_sample > burst_hi),
        "no segment plausibly covers the burst: {:?}",
        result.segments
    );
}
