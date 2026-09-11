# Sortformer

Status: port complete (Stages 1-8; ship gate passed 2026-07-22)

Intake signed off 2026-07-19: capability scope approved as drafted;
acceptance set = AMI (DER + JER vs NeMo reference, plus prob-tensor parity).
Stage 4 tensor parity 6/6; compression internals verified bit-exact vs a
neutral arbiter (residual #1). Stage 7 ref-dtype gate PASS: **C++ AMI DER
14.59% / JER 19.51%** (16 mtgs, very_high_latency, forced-alignment RTTMs +
dihard3 PP) vs measured reference 14.83% / 19.89% (gate: reference + 0.01).
Shipped matrix: **F32 + F16 + Q8_0** (k tiers withdrawn; see "Quant policy
(Stage 7)"). Private HF repo:
`handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf`.

Batch posture: **ACCEPTED GAP — no `run_batch()`** (user-approved Stage 4
deferral; single-session `transcribe_run` is the shipped path; revisit with
multitalker interop). Streaming posture: **natively streaming — PASS**; the
chunk/lookahead contract is the preset menu in "Public API (run extension)"
(default = GGUF-shipped checkpoint cfg; very_high_latency ~30.4 s lookahead
... low_latency ~1.04 s), exposed via `transcribe_sortformer_stream_ext`.
Push-audio `transcribe_stream_*` entry point is future work (STREAM-slot
kind reserved by design).

Streaming Sortformer is a **frame-level end-to-end neural speaker diarizer**
(architecture pattern `encoder-diarizer`, new to this repo). It is NOT a
transcription model: it consumes 16 kHz mono audio and emits a `T x 4`
matrix of per-frame per-speaker activity probabilities (one row per 80 ms
frame, up to 4 speakers, columns in speaker arrival order). It runs online
with a stateful Arrival-Order Speaker Cache (AOSC) + FIFO.

Primary reason this family is being ported: it is the named hard
dependency of the Parakeet multitalker speaker-attributed ASR path
(`spk_supervision='diar'`). See
`reports/porting/parakeet/multitalker-parakeet-streaming-0.6b-v1/intake.json`.

## Identity

- Family key: `sortformer`
- Upstream architecture string: `nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel`
- Architecture pattern: `encoder-diarizer` (NEST/FastConformer encoder -> 18L Transformer encoder -> 2 FF -> 4 sigmoid/frame)
- Hugging Face repo: `nvidia/diar_streaming_sortformer_4spk-v2.1`
- Hugging Face revision: `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`
- License: NVIDIA Open Model License (v2.1). Sibling `diar_streaming_sortformer_4spk-v2` is CC-BY-4.0.
- Variants:
  - `diar_streaming_sortformer_4spk-v2.1` — this port (best meeting DER; multitalker's named dependency).
  - `streaming-4spk-v2` — architecturally identical, CC-BY-4.0. Deferred; add by dropping in weights once v2.1 is validated.

### Diarization scope policy

Sortformer is the repo's first diarization-only family. transcribe.cpp
remains a transcription library; diarization-only models are in scope only
when they (a) feed a transcription pipeline in-repo (here: the named
`spk_supervision='diar'` dependency of parakeet multitalker speaker-
attributed ASR), or (b) reuse an encoder family the repo already maintains
(here: the NEST FastConformer is parakeet's ConformerEncoder, reused
verbatim). A diarizer meeting neither clause (e.g. a pyannote
segmentation+clustering port: new architecture, new dependency surface, no
in-repo ASR consumer) is out of scope. Standalone diarization output is
exposed because it falls out of the multitalker dependency for free, via
the pre-existing transcript-independent `transcribe_speaker_segment` ABI —
no diarizer-specific output surface was added.

## Public API (run extension)

`include/transcribe/sortformer.h` — `TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM`
(`SFST`, RUN slot; registered in `docs/extension-kinds.md`). A run produces
no text; results are read via `transcribe_n_speaker_segments` /
`transcribe_get_speaker_segment` (`TRANSCRIBE_FEATURE_DIARIZATION`).

```c
transcribe_sortformer_stream_ext ext;
transcribe_sortformer_stream_ext_init(&ext);          /* preset = DEFAULT (GGUF cfg) */
ext.preset = TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY;
run_params.family = &ext.ext;                          /* probe accepts_ext_kind first */
```

Preset menu (frames are 80 ms; chunk/rc/fifo/update/spkcache geometry per
`k_presets` in `src/arch/sortformer/stream.cpp`, mirroring the upstream
published operating points):

| Preset | Lookahead | Geometry (chunk/rc/fifo/update/cache) | DER (AMI ihm test, FA + dihard3 PP) | Throughput (M4 CPU) |
| --- | --- | --- | --- | --- |
| `VERY_HIGH_LATENCY` | ~30.4 s | 340/40/40/300/188 | 14.59% (full 16 mtgs; REF 14.83%) | ~25-31x realtime |
| `HIGH_LATENCY` | ~10.0 s | 124/1/124/124/188 | not separately gated | — |
| `LOW_LATENCY` | ~1.04 s | 6/7/188/144/188 | 14.80% (3-mtg subset; REF 14.81%) | ~0.85x realtime (many small windows) |
| `DEFAULT` | GGUF cfg | checkpoint values | — | — |

The menu is discrete (jointly-tuned bundles), not a latency dial; only
these bundles are accuracy-validated. Precedence: GGUF cfg < ext preset <
`TRANSCRIBE_SORTFORMER_STREAM_PRESET` env < per-field env (env layers are
validation hooks; `small` is a diagnostic-only env preset, deliberately
not in the public enum). An out-of-range preset or wrong-kind ext is
rejected pre-clear (`run_validate`), preserving the previous result.
Unit test: `tests/sortformer_stream_ext_unit.cpp`
(`TRANSCRIBE_SORTFORMER_GGUF`-gated).

## Push-audio streaming (STREAM slot)

The compute core is natively streaming; this section specifies the
push-audio session surface that exposes it (`transcribe_stream_begin` /
`_feed` / `_finalize`), registered as a separate STREAM-slot kind taking
the same `transcribe_sortformer_preset` enum.

### Extension kind

`TRANSCRIBE_EXT_KIND_SORTFORMER_PUSH_STREAM` — FourCC `SFPS`,
little-endian `0x53504653`, `TRANSCRIBE_EXT_SLOT_STREAM`, struct
`transcribe_sortformer_push_stream_ext { transcribe_ext ext;
transcribe_sortformer_preset preset; }`, init function
`transcribe_sortformer_push_stream_ext_init` (stamps `preset = DEFAULT`).
Registered in `docs/extension-kinds.md`. One kind per schema: the preset
enum is shared with the RUN-slot kind, but the slots are distinct and
each kind stays in exactly one slot. `accepts_ext_kind` accepts `SFST`
on RUN and `SFPS` on STREAM, nothing else. `caps.supports_streaming` is
forced on after `read_capability_kv` (the compute core is streaming by
construction; the GGUF capability KV must not gate the surface).

### Session state layout

One stream per session, single-threaded (matches the dispatcher's
ACTIVE-state gate and the Rust binding's `&mut` lease; no locks inside
the family). State hangs off `SortformerSession` as `PushStream`:

- `active`, resolved `SortformerStreamParams` (preset resolved at begin,
  identical precedence to `run()`) and `ms_per_frame`.
- Raw-PCM tail: `std::vector<float> pcm` plus the absolute stream sample
  index of `pcm[0]`. Samples older than the next uncomputed STFT frame's
  left window edge are dropped; the retained tail is bounded by
  `n_fft/2 + hop` samples plus one feed, independent of stream length.
- Mel tail: `std::vector<float> mel` columns `[mel_start, mel_computed)`
  of the logical `[n_mels, T]` matrix. Columns before the next chunk's
  `win_lo` (`stt - chunk_left_context*sub`, clamped) are dropped;
  retention is bounded by one chunk window (`chunk_len + lc + rc`
  frames), independent of stream length. Peak compute memory is the
  per-chunk graph (as offline); nothing else grows with duration except
  the append-only `total_preds` / committed-turn rows (the product).
- Chunk cursor `stt` (next chunk's middle-start mel frame), and the
  decode bookkeeping (`committed` rows, per-speaker open-run tails).
- The AOSC/FIFO state is the existing persistent `DiarStreamScratch`
  (`sc.stream`); `stream_begin` resets it exactly as the offline core
  does.

### Incremental mel frontend

`run()` computes the whole-file mel once; the push path cannot. The
Sortformer frontend (`pad_mode="constant"`, `normalize="none"`,
`nemo_seq_len_ceil`) is per-frame independent, which makes an exact
incremental form possible:

- STFT frame `t` reads pre-emphasized samples `[t*hop - n_fft/2,
  t*hop + n_fft/2)` (zero outside the recording). Frame `t` is fully
  determined once `t*hop + n_fft/2 <= n_arrived`; since `n_fft/2 > hop`,
  any frame computable mid-stream is a "real" frame (index `< floor(N/hop)`).
- `MelFrontend::compute_frames()` (new, in `transcribe-mel.cpp`) computes
  an arbitrary frame range from the caller's sample tail, sharing the
  constructor's window/filterbank/FFT and reusing the exact scalar fused
  STFT+mel+log inner loop of `compute()` (per-frame, deterministic;
  this build has no system BLAS). Pre-emphasis is applied on the fly per
  sample (`y[i] = x[i] - alpha*x[i-1]`, `y[0] = x[0]`), matching
  `compute()`'s padded-buffer arithmetic bit-for-bit on the same build.
- The trailing partial frame of the finalized stream
  (`floor(N/hop) < t < ceil(N/hop)`, at most one) is the zero mask
  `compute()` applies for `normalize="none"`; `stream_finalize` emits it
  as zeros. Feeds never emit masked frames.

This makes streaming mel bit-identical to the batch path on the same
build (verified by test, see below).

### Window scheduling (parity with the batch core)

`run_diar_streaming_core`'s per-chunk body (Graph A pre-encode over the
mel window, host concat `[spkcache|fifo|chunk]`, Graph B, AOSC/FIFO
`streaming_update_sync`) is factored verbatim into
`run_diar_streaming_window()`; the offline core loop calls it with
identical window arithmetic (no math change — the offline path's outputs
and dumps are untouched). The push session drives the same function:

- Feed time: a chunk is emitted as soon as its **full** window exists:
  `stt + chunk_len*sub + rc*sub <= frames_ready`. Window arithmetic is
  the core's verbatim (`left_offset = min(lc*sub, stt)`, `end = stt +
  chunk_len*sub`, `rc` frames of right context). Any chunk the batch
  loop would run with full right context is emitted at feed time with
  the identical window; `frames_ready <= T_final` always holds, so a
  feed-time emission implies the batch loop also had full right context.
- Finalize time: remaining real + masked frames are computed, then the
  remaining chunks run with the batch loop's clipped formulas
  (`end = min(stt + chunk_len*sub, T_final)`,
  `right_offset = min(rc*sub, T_final - end)`), `while (stt < T_final)`.

Together: chunk boundaries, window contents, and AOSC state updates are
exactly the batch loop's, independent of feed sizes (chunk-size
invariance is asserted by test). Output frame `i` is final once
accumulated (`total_preds` is append-only; NeMo streaming guarantees
frames are never revisited).

### Committed / tentative semantics

Every window re-decodes the accumulated `[T, 4]` probs with the shipped
0.5-threshold run extractor (per-speaker maximal runs). A run whose end
is determined (activity dropped below threshold) can never change; a
run still open at the processed edge may grow.

- `transcribe_n_speaker_segments` / `transcribe_get_speaker_segment`
  expose **committed turns only**: append-only, monotone in count,
  timestamps in absolute ms from stream start, speaker ids global and
  arrival-order (AOSC columns). Rows are appended in turn-close order.
  `transcribe_stream_n_committed_segments` equals the row count.
- Tentative (open) turns are exposed via new family accessors on the
  same session:
  `transcribe_sortformer_push_stream_n_tentative()` /
  `_get_tentative(session, i, out)` returning
  `transcribe_speaker_segment` rows. At most one open row per speaker.
  `feed` returns them until `stream_finalize` closes every open run at
  the final timestamp, after which the tentative set is empty and the
  committed rows are the full offline-equivalent segmentation.
- The stream carries no text: `full_text` stays empty,
  `result_kind = TRANSCRIBE_TIMESTAMPS_NONE`. `update->result_changed`
  is set whenever the feed processed at least one window (new decoded
  frames at the edge) or the committed set grew — it is a cheap "the
  decode input moved" probe, not a per-set diff. **Consumers must pull
  the tentative accessors after every feed** (open turns can extend
  without the committed set growing); the committed set is the only
  append-only view. `audio_committed_ms` reports the processed edge,
  `buffered_ms` the lookahead not yet consumed (drain hint).

Parity contract: push-stream vs offline `transcribe_run` on the same
audio and preset produce the same committed rows after finalize (sorted
comparison). On the no-BLAS scalar CPU builds this repo ships and tests
(Windows/MinGW, no system BLAS) the incremental mel and window
scheduling are bit-identical to the batch path, so the tests assert
EXACT row equality. On BLAS/Accelerate builds `compute()`'s batched
sgemm filterbank matmul may differ from `compute_frames()`'s per-frame
fused accumulation by float rounding, which can move a boundary by a
frame — exact equality is a build property, and consumers on such
builds should compare with a small ms tolerance instead.

### Error paths

- `transcribe_stream_begin`: dispatcher preflight handles state/struct/
  capability; family `stream_validate` runs `transcribe_ext_check` on
  the `SFPS` kind and the preset range check pre-clear (rejection
  preserves the previous snapshot, no FAILED transition), matching the
  RUN-slot `run_validate` gate. A NULL stream ext selects DEFAULT.
- `stream_feed` / `stream_finalize`: abort callback is polled per chunk
  (`TRANSCRIBE_ERR_ABORTED` transitions to FAILED, partial rows
  preserved); ggml failures surface as `TRANSCRIBE_ERR_GGUF`.
- Audio too short to produce >= 2 mel frames yields an empty
  segmentation at finalize (OK, zero rows) — a deliberate divergence
  from `transcribe_run`, which rejects such clips up front.

### Threading

One stream per session, one thread at a time (dispatcher-enforced);
no locks inside the family. The existing per-session `DiarStreamScratch`
and scheduler keep concurrent sessions of one model isolated; the
0.x one-compute-per-model rule applies unchanged (the Rust binding's
per-model lease covers the stream lifetime).

### Tests

`tests/sortformer_push_stream_unit.cpp`
(`TRANSCRIBE_SORTFORMER_GGUF`-gated, RC 77 skip): kind/slot probe +
init stamping; `stream_validate` rejection with snapshot preservation;
feed-vs-offline parity on the committed 2-spk oracle; chunk-size
invariance (1 s vs 10 s feeds identical); tentative->committed
convergence at finalize. `tests/sortformer_push_stream_long.cpp`
(env-gated wav): full-stream run over a multi-minute file — bounded
memory (constant chunk geometry), global speaker identity, committed
monotonicity. Existing tests must stay green unchanged (offline path
byte-identical; golden RTTM untouched).

## Quant policy (Stage 7)

Shipped matrix: **F32 (reference) + F16 + Q8_0** only
(`FAMILY_PRESETS["sortformer"]` in `scripts/lib/quant_policy.py`). The
k-quant tiers are withdrawn: this family's output depends on discrete
AOSC speaker-cache compression decisions, and k-tier weight error can
deterministically flip a near-tie pick and permute speaker labels
mid-stream. Observed at Q5_K_M on AMI TS3003b under the default CPU
operating point (9.47% -> 32.13% DER; deterministic across thread
counts, escaped by Metal float order and by high_latency). The flip is
chaotic in quant error (coarser Q4_K_M passed the same sample), so a
clean 16-meeting result cannot certify a k tier; the tiers also save
little disk (139 -> 92 MB) and are slower than Q8_0 on CPU. Full
evidence: `reports/diar/diar_streaming_sortformer_4spk-v2.1.ami-ihm-test-fa.summary.md`.

A future push-audio entry point (`transcribe_stream_begin/feed`, live
diarization) registers a separate STREAM-slot kind taking the same preset
enum; the multitalker interop path (raw T x 4 supervision into parakeet's
layer-0 speaker kernel) is internal C++ and does not round-trip through
this surface.

## References

- Canonical reference: NeMo `SortformerEncLabelModel.from_pretrained(...)` + `sortformer_modules.SortformerModules` (streaming speaker cache). Same NeMo toolkit already pinned for parakeet/canary.
- Instrumented reference: `scripts/envs/sortformer/` (to be created at Stage 2, mirroring `scripts/envs/parakeet`).
- Cross-check references:
  - Sortformer paper: https://arxiv.org/abs/2409.06656
  - Streaming Sortformer (AOSC): https://arxiv.org/abs/2507.18446
  - NEST encoder: https://arxiv.org/abs/2408.13106
  - NeMo eval script: `examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py`

## Commands

Oracle audio (deterministic 2-speaker mix, committed):

```bash
uv run scripts/gen_sortformer_oracle_audio.py
# -> samples/sortformer-2spk-mix.wav + tests/golden/sortformer/sortformer-2spk-mix.rttm
```

Reference dumps (NeMo, via scripts/envs/sortformer):

```bash
BASE=build/validate/sortformer/diar_streaming_sortformer_4spk-v2.1/sortformer-2spk-mix
uv run --project scripts/envs/sortformer scripts/dump_reference_sortformer_nemo.py encoder \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --audio samples/sortformer-2spk-mix.wav --out $BASE/encoder/ref
uv run --project scripts/envs/sortformer scripts/dump_reference_sortformer_nemo.py diarize \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --audio samples/sortformer-2spk-mix.wav --out $BASE/diarize/ref
```

Conversion:

```bash
scripts/convert-sortformer.py -> models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf (arch `sortformer`).
```

Validation:

```bash
# Tensor parity (offline enc.* + diar.preds_offline + streaming diar.probs):
uv run scripts/validate.py all --family sortformer --backend cpu     # 6/6 green (single-chunk diar.probs)
VALIDATE_SORTFORMER_PRESET=small uv run scripts/validate.py all --family sortformer  # multi-chunk FIFO+compress diagnostic

# Acceptance (DER + JER, official protocol: forced-alignment RTTMs + dihard3 PP).
uv run scripts/diar/ingest_ami.py --config ihm --split test               # audio + manual RTTMs
uv run scripts/diar/fetch_ami_forced_alignment.py --config ihm --split test  # forced-alignment RTTMs + FA manifest
# Reference baseline (one-time):
uv run --project scripts/envs/sortformer scripts/diar/run_reference_sortformer_nemo.py \
  --manifest samples/diar/ami-ihm-test.manifest.jsonl \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --preset very_high_latency \
  --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
  --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-ami-ihm-test-PPdihard \
  --out reports/diar/diar_streaming_sortformer_4spk-v2.1-REF.ami-ihm-test.PPdihard.jsonl
# C++ port (dumps diar.probs per meeting, applies the SAME dihard3 PP):
uv run --project scripts/envs/sortformer scripts/diar/run_cpp_sortformer.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --gguf models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf --preset very_high_latency \
  --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
  --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-cpp-ami-ihm-test \
  --out reports/diar/diar_streaming_sortformer_4spk-v2.1-CPP.ami-ihm-test.jsonl --backend cpu
uv run scripts/diar/score_der.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-cpp-ami-ihm-test \
  --out reports/diar/diar_streaming_sortformer_4spk-v2.1-CPP.ami-ihm-test-fa.PPdihard.score.json
# -> REF DER 14.83% / JER 19.89% ; C++ DER 14.59% / JER 19.51% (within 0.24 pt).
```

Benchmarks (publication scope; on-doc artifacts
`reports/perf/apple-m4/diar-streaming-sortformer-4spk-v2-1-publication_*.json`):

```bash
uv run scripts/bench/run.py \
  --models diar_streaming_sortformer_4spk-v2.1 \
  --quants f16,q8_0 \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name diar_streaming_sortformer_4spk-v2.1-publication
# apple-m4: Metal ~170x (jfk) / ~110x (dots) rtf; CPU Q8_0 100x/51x, F16 80x/44x.
# Note: low_latency preset runs ~1.2x realtime on CPU (per-chunk graph
# rebuild dominates at 0.5 s chunks); the shipped default and
# very_high_latency are unaffected.
```

## Capability Validation

> NOTE (non-standard family): the template's forced `MUST PASS` rows are
> transcription rows (explicit-language / auto transcribe). Sortformer
> emits no transcript, so those rows are `OUT OF SCOPE — not a
> transcription model`. The forced-row spirit is preserved by making the
> **diarization** capability the obligated one: streaming diarization,
> offline (single-chunk) diarization, and the raw activity-tensor output
> are the `MUST PASS` rows for this family. These substitutions require
> user sign-off at intake.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Streaming diarization (<=4 spk) | online, AOSC/FIFO cache | port streaming diarize vs NeMo reference at a fixed preset (e.g. 1.04s latency) | prob-tensor parity vs reference; DER + JER within tolerance on AMI | MUST PASS | PASS — `diar.probs` parity (single-chunk 2.96e-4; multi-chunk `small` preset ~3.9e-4 uniform); AMI DER 14.59% / JER 19.51% vs ref 14.83% / 19.89% |
| Offline diarization | single large chunk | port diarize vs NeMo reference | prob-tensor parity vs reference | MUST PASS | PASS — `diar.preds_offline` 2.96e-4 (validate.py all 6/6) |
| Speaker-activity tensor | `include_tensor_outputs` | T x 4 sigmoid probs, port vs reference on the same audio | max-abs-diff within tolerance, columns arrival-order aligned | MUST PASS | PASS — `diar.probs` [150x4] within tolerance |
| Multitalker interop | feed diar supervision | Sortformer T x 4 drives parakeet multitalker layer-0 speaker kernel | multitalker emits speaker-attributed transcript | OUT OF SCOPE — owned by the multitalker port; unblocked once both land | N/A (deferred to multitalker port) |
| Transcription (text) | n/a | n/a | model produces no text output | OUT OF SCOPE — not a transcription model | N/A |
| Translation | n/a | n/a | n/a | OUT OF SCOPE — not a transcription model | N/A |
| Timestamps (transcription) | n/a | n/a | diarization segment times are intrinsic, not a transcript-timestamp capability | OUT OF SCOPE — not a transcription model | N/A |
| >4 speakers | n/a | n/a | hard cap at 4 speakers | OUT OF SCOPE — architectural 4-speaker cap | N/A |
| Batch (`run_batch`) | n/a | n/a | parallel multi-file fast path | ACCEPTED GAP — user-approved Stage 4 deferral; single-session runs only | N/A (deferred) |

## Reference baselines (Stage 2 Oracle)

- **Tensor parity oracle** (`build/validate/sortformer/diar_streaming_sortformer_4spk-v2.1/sortformer-2spk-mix/`): mel + fastconformer + encoder_proj + transformer + offline preds + streaming `diar.probs` [150×4]. Offline preds == streaming probs on this short clip (single chunk). Tolerances in `tests/tolerances/sortformer.json` (finalized Stage 4; no `_provisional` flags).
- **Measured reference DER/JER** on AMI ihm test (16 meetings, 9.06 h), preset `very_high_latency`, collar 0.0s / overlap included (NVIDIA AMI protocol). The reference pipeline REPRODUCES the published number (15.90 @ very_high_latency AMI-IHM). Full 2x2 decomposition of the two protocol levers:

  | predictions | vs manual RTTMs (diarizers-community/ami) | vs forced-alignment RTTMs (nttcslab-sp) |
  |---|---|---|
  | no post-processing | 29.44% | 15.96% |
  | + dihard3 post-processing | 27.95% | **14.83%** |

  - **Official gate = forced-alignment RTTMs + dihard3 PP: DER 14.83%, JER 19.89%** (missed 5.89%, FA 5.60%, confusion 3.34%). File: `reports/diar/diar_streaming_sortformer_4spk-v2.1-REF.ami-ihm-test-fa.PPdihard.score.json`. Stage 4 / Stage 7 apply the SAME PP to the C++ probs and score vs the SAME forced-alignment RTTMs, gated against this number.
  - The RTTM source is ~13 pts of the published-vs-manual gap; post-processing another ~1 pt. Confusion stays ~3% throughout, so the diarization core is correct; the manual-RTTM inflation was recall against dense annotations (within-utterance pauses), exactly as NVIDIA's forced-alignment `only_words` labels avoid.
  - Ground truth: nttcslab-sp/diar-forced-alignment (ASRU 2025, card ref [7]), `only_words`, full-corpus-ASR partition — the labels NVIDIA used. PP: NeMo v2 `dihard3-dev` config (no AMI-specific or v2.1 PP is published; dihard3 is the meeting-domain proxy).
  - Diagnostic/context files also kept: `...ami-ihm-test.score.json` (no-PP/manual 29.44%), `...ami-ihm-test-fa.noPP.score.json` (no-PP/FA 15.96%).

## Conversion (Stage 3)

`scripts/convert-sortformer.py` -> `models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf` (F32, 471 MB, arch string `sortformer`). 971 tensors emitted; 19 skipped (2 `preprocessor.*` recomputed in the C++ frontend, 17 BN `num_batches_tracked` counters). Gate B all-PASS. The GGUF tensor-name contract Stage 4's `src/arch/sortformer/weights.cpp` must implement:

- `enc.pre_encode.*` — dw_striding subsample (same as Parakeet).
- `enc.blocks.{0..16}.*` — 17 Conformer/NEST blocks; identical suffix map to Parakeet's `ENCODER_BLOCK_TABLE` (macaron FF1/FF2, rel-pos self-attn with `pos_bias_u/v`, conv module with `conv.bn.{weight,bias,running_mean,running_var}`). `use_bias=true` (linear biases present); `conv_norm_type=batch_norm`.
- `diar.encoder_proj.{weight,bias}` — Linear 512 -> 192 between the two encoders.
- `tf.blocks.{0..17}.*` — 18 post-LN Transformer blocks: `norm_1` -> attn(`attn.{q,k,v,out}`) -> `norm_2` -> FFN(`ff.{in,out}`), ReLU, `pre_ln=false`.
- `diar.fc1.*`, `diar.spk_head.*` (4x384 -> 4 sigmoid), `diar.single_spk_head.*` (4x192).

Tensor-name mapping decisions: Conformer names mirror `convert-parakeet.py` verbatim (shared NeMo ConformerEncoder). Transformer norms named `norm_1`/`norm_2` and BN kept as `.bn.` so `reference_dtype_for`/`policy.cpp` route them to F32; representatives registered in `scripts/lib/test_quant_policy_sync.py`. GGUF slug is the variant (`diar_streaming_sortformer_4spk-v2.1`), not the longer repo slug.

## Notes

- Frontend `normalize=NA` (no per-feature normalization); 128 mels, n_fft 512, win 400 / hop 160, hann, dither 1e-5. `preemphasis=0.97` CONFIRMED from the loaded preprocessor at Stage 2 (NeMo default applied, not disabled).
- Reaching the published DER later (optional, not a port gate) needs forced-alignment RTTMs + a tuned post-processing config; the `--collar` / `--skip-overlap` knobs in `scripts/diar/score_der.py` are for diagnostics only.
- Output post-processing (probs -> RTTM segments) is dataset-tuned upstream. Port gate is prob-tensor parity; DER/JER use a fixed/documented post-processing and are compared to the reference model, not to the published numbers.
- Streaming operating-point knobs: `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_update_period`, `spkcache_len` (all in 80 ms frames). Published DER corresponds to specific presets (1.04s vs 30.4s input-buffer latency).
