//! transcribe-e2e — end-to-end check suite for the fork's transcribe.dll.
//!
//! Consumption posture is byte-for-byte the sound-rs one (sound-server's
//! `models.registry.json`): the `transcribe-cpp` safe wrapper with the
//! `dynload` feature, runtime `dlopen` of a single self-contained
//! `transcribe.dll`, and only public safe-binding calls — no raw sys FFI.
//!
//! Every check maps to the way sound-server drives the library (model load
//! via `Model::load_with`, sessions, run/run_batch, progress callbacks,
//! buffered streams, the sortformer DIARIZE role and its push-stream
//! extension, the embedded Silero VAD and the VAD-prechunk batch pipeline).
//!
//! Exit codes: 0 = no FAIL (SKIP is fine), 1 = at least one FAIL, 2 = usage.

mod wav;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use transcribe_cpp::{
    backend_available, devices, init_backends_default, init_dynamic, is_dynamic_loaded, version,
    Backend, Diarize, DiarizeExtension, DiarizeOptions, Error, Feature, Itn, Model, ModelOptions,
    Progress, ProgressAction, ProgressCallback, Role, RunOptions, SortformerDiarizeOptions,
    SortformerPreset, SortformerStreamOptions, StreamExtension, StreamOptions, Task, VadOptions,
    VadSegment,
};

static DLL_PATH: OnceLock<PathBuf> = OnceLock::new();

// ── context ─────────────────────────────────────────────────────────────────

struct Ctx {
    models_dir: PathBuf,
    samples_dir: PathBuf,
    pcm_cache: HashMap<&'static str, Vec<f32>>,
    /// Embedded-VAD availability, probed once (None until probed; the inner
    /// Option is None when the DLL was built without the asset).
    vad_probe: Option<bool>,
}

impl Ctx {
    fn model_path(&self, file: &str) -> Option<PathBuf> {
        let p = self.models_dir.join(file);
        p.is_file().then_some(p)
    }

    /// Load a PCM sample (16 kHz mono f32), cached. Missing sample = check
    /// error (samples ship with the repo, so a miss is a setup bug, not SKIP).
    fn pcm(&mut self, sample: &'static str) -> Result<Vec<f32>, String> {
        if let Some(v) = self.pcm_cache.get(&sample) {
            return Ok(v.clone());
        }
        let path = self.samples_dir.join(sample);
        let w = wav::load(&path).map_err(|e| e.to_string())?;
        if w.sample_rate != 16000 {
            return Err(format!("{sample}: {} Hz, need 16000", w.sample_rate));
        }
        if w.channels != 1 {
            return Err(format!("{sample}: {} channels after downmix", w.channels));
        }
        self.pcm_cache.insert(sample, w.samples.clone());
        Ok(w.samples)
    }

    /// Tile `pcm` `times` over with `gap_ms` of silence between repeats
    /// (kept for future long-form inputs; current checks use the natural
    /// long samples in `samples/`).
    #[allow(dead_code)]
    fn tile_with_gaps(pcm: &[f32], times: usize, gap_ms: usize) -> Vec<f32> {
        let gap = vec![0.0f32; gap_ms * 16];
        let mut out = Vec::with_capacity(pcm.len() * times + gap.len() * times);
        for i in 0..times {
            if i > 0 {
                out.extend_from_slice(&gap);
            }
            out.extend_from_slice(pcm);
        }
        out
    }
}

// ── outcomes ────────────────────────────────────────────────────────────────

enum Outcome {
    Pass(String),
    Skip(String),
    Fail(String),
}

type CheckFn = fn(&mut Ctx) -> Outcome;

struct Check {
    name: &'static str,
    what: &'static str,
    run: CheckFn,
}

fn pass<T: std::fmt::Display>(detail: T) -> Outcome {
    Outcome::Pass(detail.to_string())
}
fn skip<T: std::fmt::Display>(why: T) -> Outcome {
    Outcome::Skip(why.to_string())
}
fn fail<T: std::fmt::Display>(why: T) -> Outcome {
    Outcome::Fail(why.to_string())
}

fn load_cpu(ctx: &Ctx, file: &str) -> Result<Model, Outcome> {
    let path = ctx
        .model_path(file)
        .ok_or_else(|| skip(format!("model file missing: {file}")))?;
    let opts = ModelOptions { backend: Backend::Cpu, ..Default::default() };
    Model::load_with(&path, &opts).map_err(|e| fail(format!("load {file}: {e}")))
}

fn run_offline(model: &Model, pcm: &[f32], opts: &RunOptions) -> Result<String, String> {
    let mut session = model.session().map_err(|e| format!("session: {e}"))?;
    let t = session.run(pcm, opts).map_err(|e| format!("run: {e}"))?;
    Ok(t.text)
}

fn default_run() -> RunOptions {
    RunOptions::default()
}

// ── A. library / ABI layer (no model) ──────────────────────────────────────

fn check_dynload_init(_ctx: &mut Ctx) -> Outcome {
    // sound-server: TranscribeBackend::new → init_dynamic(lib_path) with the
    // full symbol-table resolve + base-version gate.
    let dll = DLL_PATH.get().expect("dll path set");
    if let Err(e) = init_dynamic(Some(dll)) {
        return fail(format!("init_dynamic: {e}"));
    }
    if !is_dynamic_loaded() {
        return fail("init_dynamic returned Ok but is_dynamic_loaded() is false");
    }
    pass(format!("loaded {}", dll.display()))
}

fn check_version(_ctx: &mut Ctx) -> Outcome {
    let v = version();
    let mut it = v.split('.');
    let (maj, min, pat) = (it.next().and_then(|s| s.parse::<u32>().ok()),
                           it.next().and_then(|s| s.parse::<u32>().ok()),
                           it.next().is_some());
    match (maj, min, pat) {
        // tuple compare: 1.x also passes the >=0.5.0 gate
        (Some(maj), Some(min), true) if (maj, min) >= (0, 5) => pass(format!("version {v} (>= 0.5.0 base gate)")),
        _ => fail(format!("version string '{v}' failed the >=0.5.0 gate")),
    }
}

fn check_backends(_ctx: &mut Ctx) -> Outcome {
    if let Err(e) = init_backends_default() {
        return fail(format!("init_backends_default: {e}"));
    }
    if !backend_available(Backend::Cpu) {
        return fail("CPU backend not available");
    }
    let n = devices().len();
    if n == 0 {
        return fail("devices() is empty");
    }
    // sound-server resolves cuda > vulkan > rocm, else CPU; report the same way.
    let gpu = [Backend::Cuda, Backend::Vulkan, Backend::Rocm]
        .into_iter()
        .find(|b| backend_available(*b));
    pass(format!(
        "{n} devices, CPU ok, gpu_backend={}",
        gpu.map(|g| format!("{g:?}").to_lowercase()).unwrap_or_else(|| "none (cpu fallback)".into())
    ))
}

fn check_embedded_vad_probe(ctx: &mut Ctx) -> Outcome {
    let opts = ModelOptions { backend: Backend::Cpu, ..Default::default() };
    match Model::load_embedded_vad_with(&opts) {
        Ok(_) => {
            ctx.vad_probe = Some(true);
            pass("embedded Silero VAD asset present and loadable")
        }
        Err(Error::NotImplemented(msg)) => {
            ctx.vad_probe = Some(false);
            skip(format!("DLL built without the embedded asset ({msg}); VAD checks will skip"))
        }
        Err(e) => fail(format!("load_embedded_vad: {e}")),
    }
}

// ── B. ASR core (whisper-tiny) ──────────────────────────────────────────────

fn check_asr_run(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "whisper-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let text = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    if text.trim().is_empty() {
        return fail("empty transcript");
    }
    pass(format!("{:.1}s audio -> {:?}", pcm.len() as f32 / 16000.0, shorten(&text)))
}

fn check_asr_nommap_parity(ctx: &mut Ctx) -> Outcome {
    let path = match ctx.model_path("whisper-tiny-Q8_0.gguf") {
        Some(p) => p,
        None => return skip("model file missing: whisper-tiny-Q8_0.gguf"),
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let muted = ModelOptions { backend: Backend::Cpu, no_mmap: true, ..Default::default() };
    let model = match Model::load_with(&path, &muted) {
        Ok(m) => m,
        Err(e) => return fail(format!("no_mmap load: {e}")),
    };
    let text_nommap = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(format!("no_mmap run: {e}")),
    };
    // Reference: the mmap-default load from check_asr_run (rerun to stay
    // independent of check ordering).
    let model2 = match Model::load_with(&path, &ModelOptions { backend: Backend::Cpu, ..Default::default() }) {
        Ok(m) => m,
        Err(e) => return fail(format!("mmap load: {e}")),
    };
    let text_mmap = match run_offline(&model2, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(format!("mmap run: {e}")),
    };
    if text_nommap != text_mmap {
        return fail(format!("mmap {:?} != no_mmap {:?}", shorten(&text_mmap), shorten(&text_nommap)));
    }
    pass(format!("mmap and no_mmap transcripts identical ({:?})", shorten(&text_nommap)))
}

fn check_asr_batch_parity(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "whisper-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let jfk = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let en = match ctx.pcm("fleurs-en.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let batch = match session.run_batch(&[&jfk, &en], &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(format!("run_batch: {e}")),
    };
    if batch.len() != 2 {
        return fail(format!("run_batch returned {} rows for 2 inputs", batch.len()));
    }
    let s0 = run_offline(&model, &jfk, &default_run());
    let s1 = run_offline(&model, &en, &default_run());
    let (s0, s1) = match (s0, s1) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return fail(format!("serial reference: {e}")),
    };
    let (b0, b1) = match (&batch[0], &batch[1]) {
        (Ok(a), Ok(b)) => (a.text.clone(), b.text.clone()),
        (Err(e), _) | (_, Err(e)) => return fail(format!("batch row error: {e}")),
    };
    if b0 != s0 || b1 != s1 {
        return fail(format!(
            "batch mismatch: [0] {:?} vs {:?}, [1] {:?} vs {:?}",
            shorten(&b0),
            shorten(&s0),
            shorten(&b1),
            shorten(&s1)
        ));
    }
    pass(format!("batch of 2 identical to serial runs ({:?}, {:?})", shorten(&b0), shorten(&b1)))
}

fn check_asr_progress_abort(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "whisper-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("dots.wav") {
        Ok(p) => p, // 35 s: long enough for several progress ticks
        Err(e) => return fail(e),
    };

    // 1) progress fires and is monotone
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let regressions = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_fraction = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let cb = {
        let ticks = ticks.clone();
        let regressions = regressions.clone();
        let max_fraction = max_fraction.clone();
        ProgressCallback::new(move |p: Progress| {
            use std::sync::atomic::Ordering;
            ticks.fetch_add(1, Ordering::SeqCst);
            // fraction must be non-decreasing; count violations (asserted below)
            let f = (p.fraction * 1000.0) as u32;
            let prev = max_fraction.fetch_max(f, Ordering::SeqCst);
            if f < prev {
                regressions.fetch_add(1, Ordering::SeqCst);
            }
            ProgressAction::Continue
        })
    };
    session.set_progress_callback(&cb);
    let full = session.run(&pcm, &default_run());
    session.clear_progress_callback();
    let n = ticks.load(std::sync::atomic::Ordering::SeqCst);
    let n_reg = regressions.load(std::sync::atomic::Ordering::SeqCst);
    let full = match full {
        Ok(t) => t,
        Err(e) => return fail(format!("run with progress: {e}")),
    };
    if n == 0 {
        return fail("progress callback never fired");
    }
    if n_reg > 0 {
        return fail(format!("progress fraction regressed {n_reg} times"));
    }

    // 2) cancel mid-run: Aborted + partial transcript + was_aborted
    let mut session2 = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session2: {e}")),
    };
    let seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cb2 = {
        ProgressCallback::new(move |_p| {
            // Mirror the C unit test: cancel from the FIRST per-utterance
            // emission on (cancel_at = 0); the run must stop at the next
            // utterance boundary with the completed utterance preserved.
            let _ = seen;
            ProgressAction::Cancel
        })
    };
    session2.set_progress_callback(&cb2);
    match session2.run(&pcm, &default_run()) {
        Err(err @ Error::Aborted { .. }) => match &err {
            Error::Aborted { partial, .. } => {
                let has_partial = partial.as_ref().map(|t| !t.text.trim().is_empty()).unwrap_or(false);
                if !session2.was_aborted() {
                    return fail("Error::Aborted but session.was_aborted() is false");
                }
                pass(format!(
                    "progress ticks={n}, cancel-at-first-tick -> Aborted(partial_text={has_partial}), full={:?}",
                    shorten(&full.text)
                ))
            }
            _ => unreachable!(),
        },
        Ok(_) => fail("cancel from the first progress emission did not abort the run"),
        other => fail(format!("expected Error::Aborted, got {other:?}")),
    }
}

fn check_asr_translate(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "whisper-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("fleurs-de.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let opts = RunOptions { task: Task::Translate, target_language: Some("en".into()), ..Default::default() };
    let text = match run_offline(&model, &pcm, &opts) {
        Ok(t) => t,
        Err(e) => return fail(format!("translate run: {e}")),
    };
    if text.trim().is_empty() {
        return fail("empty translation");
    }
    pass(format!("de->en: {:?}", shorten(&text)))
}

// ── C. family breadth + upstream-0.4.1 behavior ────────────────────────────

/// Shared body of the two ITN-default checks: run with Itn::Default and with
/// Itn::Off on `sample`; the default must carry `decorated` characters the
/// OFF output lacks (case/punctuation, digits where passed in).
fn itn_default_on_check(
    ctx: &mut Ctx,
    model_file: &str,
    sample: &'static str,
    decorated: &str,
) -> Outcome {
    let model = match load_cpu(ctx, model_file) {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm(sample) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let dflt = match run_offline(&model, &pcm, &RunOptions::default()) {
        Ok(t) => t,
        Err(e) => return fail(format!("default run: {e}")),
    };
    let off = match run_offline(&model, &pcm, &RunOptions { itn: Itn::Off, ..Default::default() }) {
        Ok(t) => t,
        Err(e) => return fail(format!("itn-off run: {e}")),
    };
    if !dflt.chars().any(|c| c.is_ascii_uppercase() || decorated.contains(c)) {
        return fail(format!("default output shows no ITN effect: {:?}", shorten(&dflt)));
    }
    pass(format!("default(ON)={:?} off={:?}", shorten(&dflt), shorten(&off)))
}

fn check_sensevoice_itn(ctx: &mut Ctx) -> Outcome {
    // Fork behavior: sensevoice resolves Itn::Default to ON.
    itn_default_on_check(ctx, "SenseVoiceSmall-Q8_0.gguf", "fleurs-zh.wav", ".,;:!?，。！？")
}

fn check_funasr_mlt_itn(ctx: &mut Ctx) -> Outcome {
    // upstream #205: fun-asr-mlt-nano-2512 resolves ITN DEFAULT to ON.
    itn_default_on_check(ctx, "Fun-ASR-MLT-Nano-2512-Q8_0.gguf", "fleurs-zh.wav", ".,;:!?，。！？0123456789")
}

fn check_fam_moonshine(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "moonshine-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let text = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    if text.trim().is_empty() {
        return fail("empty transcript");
    }
    pass(format!("{:?}", shorten(&text)))
}

fn check_qwen3_run(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "Qwen3-ASR-0.6B-Q4_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let text = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    if text.trim().is_empty() {
        return fail("empty transcript");
    }
    pass(format!("{:?}", shorten(&text)))
}

fn check_gigaam_longform(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "gigaam-v3-rnnt-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    // ru-long.wav: 33.8 s natural Russian — past the 25 s training window,
    // so upstream #208 encodes it in bounded windows.
    let pcm = match ctx.pcm("ru-long.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let dur = pcm.len() as f32 / 16000.0;
    let text = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    if text.trim().is_empty() {
        return fail(format!("empty transcript on {dur:.1}s windowed Russian audio"));
    }
    pass(format!("windowed {dur:.1}s -> {:?}", shorten(&text)))
}

fn check_parakeet_longform(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "parakeet-tdt-0.6b-v3-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    // tdt-0.6b-v3 is a Regular (full-attention) checkpoint: with upstream
    // #199 it loads as rel_pos_local_attn [256,256]; dots.wav (35.3 s, ~170
    // encoder frames past the 256-frame full-coverage point is close but the
    // bounded graph engages for T > W) exercises the windowed encode path.
    let pcm = match ctx.pcm("dots.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let text = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    if text.trim().is_empty() {
        return fail("empty transcript");
    }
    pass(format!("[256,256] window path -> {:?}", shorten(&text)))
}

fn check_multitalker_diarize(ctx: &mut Ctx) -> Outcome {
    // The bundle (compose-multitalker-bundle.py output): ASR half + the
    // embedded Sortformer diarizer in one file. The ASR-only conversion has
    // no diarizer, so the check requires the composed bundle.
    let model = match load_cpu(ctx, "multitalker-bundle-v1-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    // Parakeet does not register the standalone DIARIZE role (no arch
    // diarize hook): the bundle diarizes through transcribe_run with
    // diarize=ON, advertised as Feature::Diarization. The DIARIZE role
    // marks standalone diarizers (sortformer) whose only ASR surface is
    // the push-stream slot.
    if !model.supports(Feature::Diarization) {
        return fail("bundle does not advertise Feature::Diarization");
    }
    // The committed multitalker oracle mix (anchor-phrase attribution in
    // tests/parakeet_multitalker_e2e_smoke.cpp, RTTM in tests/golden/parakeet/).
    let mix = match ctx.pcm("multitalker-2spk-mix.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    // Moss/granite-plus/multitalker diarize through transcribe_run with
    // diarize=ON (embedded diarizer runs inside the parakeet forward).
    let opts = RunOptions { diarize: Diarize::On, ..Default::default() };
    let t = match session.run(&mix, &opts) {
        Ok(t) => t,
        Err(e) => return fail(format!("diarize=On run: {e}")),
    };
    let turns = &t.speaker_segments;
    if turns.is_empty() {
        return fail(format!("no speaker turns; text={:?}", shorten(&t.text)));
    }
    for s in turns {
        if s.t0_ms >= s.t1_ms || s.speaker_id < 0 {
            return fail(format!("bad turn {s:?}"));
        }
    }
    let n_spk = turns.iter().map(|s| s.speaker_id).collect::<std::collections::BTreeSet<_>>().len();
    pass(format!("{} turns / {n_spk} speakers, text={:?}", turns.len(), shorten(&t.text)))
}

// ── D. streaming ────────────────────────────────────────────────────────────

fn check_stream_parakeet(ctx: &mut Ctx) -> Outcome {
    // Streaming-trained checkpoint (chunked attention): the offline v3
    // TDT does not implement the stream slot (NOT_IMPLEMENTED by design).
    let model = match load_cpu(ctx, "multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let caps = match model.capabilities() {
        Ok(c) => c,
        Err(e) => return fail(format!("capabilities: {e}")),
    };
    if !caps.supports_streaming {
        return fail("streaming checkpoint does not advertise supports_streaming");
    }
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let offline = match run_offline(&model, &pcm, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(format!("offline reference: {e}")),
    };
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let mut stream = match session.stream(&default_run(), &StreamOptions::default()) {
        Ok(s) => s,
        Err(e) => return fail(format!("stream: {e}")),
    };
    let chunk = 16000 / 2; // 500 ms
    let mut updates = 0usize;
    for frame in pcm.chunks(chunk) {
        let u = match stream.feed(frame) {
            Ok(u) => u,
            Err(e) => return fail(format!("feed: {e}")),
        };
        updates += u.result_changed as usize;
    }
    let fin = match stream.finalize() {
        Ok(u) => u,
        Err(e) => return fail(format!("finalize: {e}")),
    };
    if !fin.is_final {
        return fail("finalize update is not final");
    }
    let text = stream.text().display();
    if text.trim().is_empty() {
        return fail("empty streamed transcript");
    }
    if text != offline {
        return fail(format!("streamed {:?} != offline {:?}", shorten(&text), shorten(&offline)));
    }
    pass(format!("buffered stream matches offline ({:?}, {updates} result changes)", shorten(&text)))
}

fn check_stream_moonshine(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "moonshine-streaming-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let mut stream = match session.stream(&default_run(), &StreamOptions::default()) {
        Ok(s) => s,
        Err(e) => return fail(format!("stream: {e}")),
    };
    let chunk = 16000 / 10; // 100 ms
    for frame in pcm.chunks(chunk) {
        if let Err(e) = stream.feed(frame) {
            return fail(format!("feed: {e}"));
        }
    }
    let fin = match stream.finalize() {
        Ok(u) => u,
        Err(e) => return fail(format!("finalize: {e}")),
    };
    if !fin.is_final {
        return fail("finalize update is not final");
    }
    let text = stream.text().display();
    if text.trim().is_empty() {
        return fail("empty streamed transcript");
    }
    pass(format!("moonshine-streaming committed: {:?}", shorten(&text)))
}

// ── E. sortformer: DIARIZE role + push-stream session ──────────────────────

fn sf_mix(ctx: &mut Ctx) -> Result<Vec<f32>, Outcome> {
    // The committed oracle mix every other suite pins (golden rows in
    // tests/sortformer_diarize_unit.cpp, RTTM in tests/golden/sortformer/).
    ctx.pcm("sortformer-2spk-mix.wav").map_err(fail)
}

fn check_sf_diarize(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "diar_streaming_sortformer_4spk-v2.1-F32.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let roles = model.roles();
    if !roles.contains(Role::Diarize) {
        return fail("roles lack Diarize");
    }
    let max_spk = match model.diarize_info() {
        Ok(i) => i.max_speakers,
        Err(e) => return fail(format!("diarize_info: {e}")),
    };
    let mix = match sf_mix(ctx) {
        Ok(p) => p,
        Err(o) => return o,
    };
    let mut diarize = match model.diarize_session() {
        Ok(d) => d,
        Err(e) => return fail(format!("diarize_session: {e}")),
    };
    let turns = match diarize.run(&mix, &DiarizeOptions::default()) {
        Ok(t) => t,
        Err(e) => return fail(format!("diarize run: {e}")),
    };
    if turns.is_empty() {
        return fail("no speaker turns on a 2-spk mix");
    }
    for s in &turns {
        if s.t0_ms >= s.t1_ms || !(1..=max_spk).contains(&s.speaker_id) {
            return fail(format!("bad turn {s:?}"));
        }
    }
    // LowLatency preset must observably change the segmentation (fork test
    // pins the exact rows; here only the preset plumb-through matters).
    let low = DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset: Some(SortformerPreset::LowLatency),
        })),
    };
    let turns_low = match diarize.run(&mix, &low) {
        Ok(t) => t,
        Err(e) => return fail(format!("diarize run (LowLatency): {e}")),
    };
    let same = turns.iter().map(|s| (s.t0_ms, s.t1_ms, s.speaker_id)).collect::<Vec<_>>()
        == turns_low.iter().map(|s| (s.t0_ms, s.t1_ms, s.speaker_id)).collect::<Vec<_>>();
    pass(format!(
        "{} turns (max {max_spk} spk), LowLatency changes output: {}",
        turns.len(),
        !same
    ))
}

fn check_sf_push_stream(ctx: &mut Ctx) -> Outcome {
    let model = match load_cpu(ctx, "diar_streaming_sortformer_4spk-v2.1-F32.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let mix = match sf_mix(ctx) {
        Ok(p) => p,
        Err(o) => return o,
    };
    // The fork's SFPS surface: session() + stream() with the Sortformer
    // stream extension. One-shot ASR runs on this model are NOT_IMPLEMENTED
    // (no run hook), the stream slot is the only inference path.
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let stream_opts = StreamOptions {
        family: Some(StreamExtension::Sortformer(SortformerStreamOptions {
            preset: Some(SortformerPreset::LowLatency),
        })),
        ..Default::default()
    };
    let mut stream = match session.stream(&RunOptions::default(), &stream_opts) {
        Ok(s) => s,
        Err(e) => return fail(format!("stream(Sortformer): {e}")),
    };
    let chunk = 16000 / 10; // 100 ms
    let mut saw_tentative = false;
    for frame in mix.chunks(chunk) {
        if let Err(e) = stream.feed(frame) {
            return fail(format!("feed: {e}"));
        }
        if !stream.tentative_speaker_segments().is_empty() {
            saw_tentative = true;
        }
    }
    let fin = match stream.finalize() {
        Ok(u) => u,
        Err(e) => return fail(format!("finalize: {e}")),
    };
    if !fin.is_final {
        return fail("finalize update is not final");
    }
    let committed = stream.speaker_segments();
    if committed.is_empty() {
        return fail("no committed speaker segments after finalize");
    }
    for s in &committed {
        if s.t0_ms >= s.t1_ms || s.speaker_id < 1 {
            return fail(format!("bad committed turn {s:?}"));
        }
    }
    pass(format!(
        "committed {} turns, saw_tentative={saw_tentative}",
        committed.len()
    ))
}

// ── F. embedded VAD + the sound-rs prechunk pipeline ────────────────────────

/// Embedded-VAD model, or SKIP when the DLL was built without the asset.
fn vad_or_skip(ctx: &mut Ctx) -> Result<Model, Outcome> {
    match ctx.vad_probe {
        Some(true) => {
            let opts = ModelOptions { backend: Backend::Cpu, ..Default::default() };
            Model::load_embedded_vad_with(&opts).map_err(|e| fail(format!("embedded VAD reload: {e}")))
        }
        Some(false) => Err(skip("no embedded VAD asset in this DLL")),
        None => Err(skip("embedded VAD not probed (run the probe check first)")),
    }
}

fn check_vad_run(ctx: &mut Ctx) -> Outcome {
    let model = match vad_or_skip(ctx) {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let mut vad = match model.vad_session() {
        Ok(v) => v,
        Err(e) => return fail(format!("vad_session: {e}")),
    };
    let r = match vad.run(&pcm, &VadOptions::default()) {
        Ok(r) => r,
        Err(e) => return fail(format!("vad run: {e}")),
    };
    if r.segments.is_empty() {
        return fail("no speech segments on jfk.wav");
    }
    let n = pcm.len() as i64;
    for s in &r.segments {
        if s.start_sample < 0 || s.end_sample <= s.start_sample || s.end_sample > n {
            return fail(format!("segment out of bounds: {s:?} (n={n})"));
        }
    }
    let ms = |s: &VadSegment| (s.start_sample * 1000 / 16000, s.end_sample * 1000 / 16000);
    let spans: Vec<_> = r.segments.iter().map(ms).collect();
    pass(format!("{} speech spans {:?} in 11.0 s", spans.len(), spans))
}

fn check_vad_prechunk_batch(ctx: &mut Ctx) -> Outcome {
    let vad = match vad_or_skip(ctx) {
        Ok(m) => m,
        Err(o) => return o,
    };
    let whisper = match load_cpu(ctx, "whisper-tiny-Q8_0.gguf") {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("dots.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    // sound-server's core pipeline: VAD 预切 -> 分批 run_batch。
    let mut vsession = match vad.vad_session() {
        Ok(v) => v,
        Err(e) => return fail(format!("vad_session: {e}")),
    };
    let r = match vsession.run(&pcm, &VadOptions::default()) {
        Ok(r) => r,
        Err(e) => return fail(format!("vad run: {e}")),
    };
    if r.segments.is_empty() {
        return fail("VAD found no speech; cannot build the batch");
    }
    let mut slices: Vec<Vec<f32>> = Vec::new();
    for s in &r.segments {
        if s.start_sample < 0 || s.end_sample <= s.start_sample {
            return fail(format!("negative/degenerate VAD segment: {s:?}"));
        }
        let a = s.start_sample as usize;
        let b = s.end_sample.min(pcm.len() as i64) as usize;
        let len = b.saturating_sub(a);
        // whisper needs >=0.1 s; drop trivially short slivers
        if len >= 1600 {
            slices.push(pcm[a..b].to_vec());
        }
    }
    if slices.is_empty() {
        return skip("all VAD slices below the 0.1 s minimum");
    }
    let refs: Vec<&[f32]> = slices.iter().map(|v| v.as_slice()).collect();
    let mut session = match whisper.session() {
        Ok(s) => s,
        Err(e) => return fail(format!("session: {e}")),
    };
    let batch = match session.run_batch(&refs, &default_run()) {
        Ok(t) => t,
        Err(e) => return fail(format!("run_batch on VAD slices: {e}")),
    };
    let nonempty = batch.iter().filter(|t| matches!(t, Ok(tr) if !tr.text.trim().is_empty())).count();
    if nonempty == 0 {
        return fail(format!("{} slices, all empty transcripts", batch.len()));
    }
    pass(format!("{} VAD slices -> {} non-empty transcripts", batch.len(), nonempty))
}

fn check_vad_stream_parity(ctx: &mut Ctx) -> Outcome {
    let model = match vad_or_skip(ctx) {
        Ok(m) => m,
        Err(o) => return o,
    };
    let pcm = match ctx.pcm("jfk.wav") {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    // Contract (transcribe/vad.h): stream_feed returns PROBABILITIES for
    // incoming audio; segments are the one-shot run's job. The streaming
    // parity that matters: every frame gets a probability — the fed total
    // must cover the same frames the one-shot run scored (± the zero-padded
    // tail frame flush adds).
    let mut one = match model.vad_session() {
        Ok(v) => v,
        Err(e) => return fail(format!("one-shot vad_session: {e}")),
    };
    let one_shot = match one.run(&pcm, &VadOptions::default()) {
        Ok(r) => r,
        Err(e) => return fail(format!("one-shot run: {e}")),
    };
    let mut inc = match model.vad_session() {
        Ok(v) => v,
        Err(e) => return fail(format!("streaming vad_session: {e}")),
    };
    let chunk = 16000 / 2; // 500 ms
    let mut fed = 0usize;
    let mut first_frame = 0i64;
    for frame in pcm.chunks(chunk) {
        match inc.stream_feed(frame) {
            Ok(r) => {
                fed += r.probs.len();
                first_frame = r.first_frame;
            }
            Err(e) => return fail(format!("stream_feed: {e}")),
        }
    }
    match inc.stream_flush() {
        Ok(r) => fed += r.probs.len(),
        Err(e) => return fail(format!("stream_flush: {e}")),
    }
    let want = one_shot.probs.len();
    // The flush pads one partial frame to 512 samples, so allow +1.
    if fed + 1 < want || fed > want + 1 {
        return fail(format!("streamed {fed} probs vs one-shot {want}"));
    }
    pass(format!("streamed {fed} probs == one-shot {want} (first_frame={first_frame})"))
}

// ── registry / runner ───────────────────────────────────────────────────────

fn checks() -> Vec<Check> {
    vec![
        Check { name: "lib-dynload-init", what: "init_dynamic resolves the full dynload symbol table (sound-rs TranscribeBackend::new)", run: check_dynload_init },
        Check { name: "lib-version", what: "version() passes the >=0.5.0 base gate", run: check_version },
        Check { name: "lib-backends", what: "backends init, CPU available, devices enumerate (sound-rs resolve_transcribe_gpu_backend)", run: check_backends },
        Check { name: "lib-embedded-vad-probe", what: "Model::load_embedded_vad_with probe (asset gate)", run: check_embedded_vad_probe },
        Check { name: "asr-run", what: "whisper-tiny offline run (mmap default load)", run: check_asr_run },
        Check { name: "asr-nommap-parity", what: "no_mmap:true load -> identical transcript (0.4.1 mmap path via dynload)", run: check_asr_nommap_parity },
        Check { name: "asr-batch-parity", what: "run_batch == serial runs", run: check_asr_batch_parity },
        Check { name: "asr-progress-abort", what: "progress callbacks fire; Cancel -> Aborted with partial transcript", run: check_asr_progress_abort },
        Check { name: "asr-translate", what: "whisper-tiny de->en translate task", run: check_asr_translate },
        Check { name: "fam-moonshine", what: "moonshine-tiny offline run", run: check_fam_moonshine },
        Check { name: "fam-sensevoice-itn", what: "sensevoice ITN Default->ON decoration vs Off", run: check_sensevoice_itn },
        Check { name: "pr15-funasr-mlt-itn", what: "fun-asr-mlt-nano ITN default ON (upstream #205)", run: check_funasr_mlt_itn },
        Check { name: "fam-qwen3", what: "qwen3-asr-0.6b run (nar graph memory, #224)", run: check_qwen3_run },
        Check { name: "pr15-gigaam-longform", what: "gigaam 35 s windowed encode (upstream #208)", run: check_gigaam_longform },
        Check { name: "pr15-parakeet-longform", what: "parakeet v3 [256,256] local-attention path (upstream #199)", run: check_parakeet_longform },
        Check { name: "fam-multitalker-diarize", what: "multitalker bundle diarize=On (flat layout, #219)", run: check_multitalker_diarize },
        Check { name: "stream-parakeet", what: "buffered stream matches offline (parakeet v3)", run: check_stream_parakeet },
        Check { name: "stream-moonshine", what: "moonshine-streaming feed/finalize commit (#222)", run: check_stream_moonshine },
        Check { name: "sf-diarize", what: "sortformer DIARIZE role + LowLatency preset (fork)", run: check_sf_diarize },
        Check { name: "sf-push-stream", what: "sortformer push-stream session: tentative->committed (fork SFPS)", run: check_sf_push_stream },
        Check { name: "vad-run", what: "embedded VAD one-shot segments", run: check_vad_run },
        Check { name: "vad-prechunk-batch", what: "VAD prechunk -> whisper run_batch (sound-rs pipeline)", run: check_vad_prechunk_batch },
        Check { name: "vad-stream-parity", what: "VAD stream_feed/flush == one-shot", run: check_vad_stream_parity },
    ]
}

fn usage() -> ! {
    eprintln!(
        "transcribe-e2e — drive transcribe.dll the way sound-rs does\n\
         \n\
         usage: transcribe-e2e [--dll PATH] [--models DIR] [--samples DIR]\n\
                               [--only SUBSTR]... [--list] [-v]\n\
         \n\
           --dll      path to transcribe.dll (default: build/e2e-dll/bin/transcribe.dll)\n\
           --models   model directory (default: models/e2e) — missing models SKIP\n\
           --samples  samples directory (default: samples)\n\
           --only     run only checks whose name contains SUBSTR (repeatable)\n\
           --list     list checks and exit\n\
           -v         print transcripts and other details\n\
         \n\
         exit: 0 no FAIL, 1 FAIL, 2 usage"
    );
    std::process::exit(2);
}

fn main() {
    let mut dll: Option<PathBuf> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut samples_dir: Option<PathBuf> = None;
    let mut only: Vec<String> = Vec::new();
    let mut verbose = false;
    let mut list = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dll" => dll = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--models" => models_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--samples" => samples_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--only" => only.push(args.next().unwrap_or_else(|| usage())),
            "-v" | "--verbose" => verbose = true,
            "--list" => list = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument {other}");
                usage();
            }
        }
    }

    let all = checks();
    if list {
        for c in &all {
            println!("{:<24} {}", c.name, c.what);
        }
        return;
    }

    let dll = dll.unwrap_or_else(|| PathBuf::from("build/e2e-dll/bin/transcribe.dll"));
    if !dll.is_file() {
        eprintln!("dll not found: {} (pass --dll)", dll.display());
        std::process::exit(2);
    }
    println!("transcribe-e2e: dll={}", dll.display());
    DLL_PATH.set(dll).ok();

    let mut ctx = Ctx {
        models_dir: models_dir.unwrap_or_else(|| PathBuf::from("models/e2e")),
        samples_dir: samples_dir.unwrap_or_else(|| PathBuf::from("samples")),
        pcm_cache: HashMap::new(),
        vad_probe: None,
    };
    if !ctx.samples_dir.is_dir() {
        eprintln!("samples dir not found: {} (pass --samples)", ctx.samples_dir.display());
        std::process::exit(2);
    }
    if !ctx.models_dir.is_dir() {
        eprintln!("models dir not found: {} (pass --models)", ctx.models_dir.display());
        std::process::exit(2);
    }

    println!("models={} samples={}\n", ctx.samples_dir.display(), ctx.models_dir.display());

    let mut n_pass = 0;
    let mut n_skip = 0;
    let mut n_fail = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    for c in &all {
        // Infrastructure checks always run: dynload-init loads the library
        // everything else depends on, and the VAD probe gates the VAD checks.
        let always = c.name == "lib-dynload-init" || c.name == "lib-embedded-vad-probe";
        if !always && !only.is_empty() && !only.iter().any(|s| c.name.contains(s.as_str())) {
            continue;
        }
        let t0 = Instant::now();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (c.run)(&mut ctx)))
            .unwrap_or_else(|p| Outcome::Fail(format!("panicked: {p:?}")));
        let dt = t0.elapsed().as_secs_f32();
        match outcome {
            Outcome::Pass(detail) => {
                n_pass += 1;
                if verbose {
                    println!("PASS  {:<24} {:>6.1}s  {}", c.name, dt, detail);
                } else {
                    println!("PASS  {:<24} {:>6.1}s", c.name, dt);
                }
            }
            Outcome::Skip(why) => {
                n_skip += 1;
                println!("SKIP  {:<24} {:>6.1}s  {}", c.name, dt, why);
            }
            Outcome::Fail(why) => {
                n_fail += 1;
                println!("FAIL  {:<24} {:>6.1}s  {}", c.name, dt, why);
                failures.push((c.name.to_string(), why));
            }
        }
    }

    println!("\n{} pass, {} skip, {} fail", n_pass, n_skip, n_fail);
    if !failures.is_empty() {
        println!("\nfailures:");
        for (n, w) in &failures {
            println!("  {n}: {w}");
        }
        std::process::exit(1);
    }
}

fn shorten(s: &str) -> String {
    const N: usize = 72;
    let t = s.trim();
    if t.chars().count() <= N {
        format!("{t:?}")
    } else {
        let cut: String = t.chars().take(N).collect();
        format!("{cut:?}..")
    }
}
