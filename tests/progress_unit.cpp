// progress_unit.cpp - white-box tests for the progress callback fire-path.
//
// Pins the transcribe_set_progress_callback / emit_progress contract:
//   - setter stores cb + userdata; NULL clears
//   - emit_progress computes progress fraction and forwards args verbatim
//   - non-zero callback return propagates as non-zero emit_progress return
//     (the cancellation handshake, mirroring audiocpp_progress_fn)
//
// And the production emission points re-established after the upstream
// 0.4.0 sync deleted the fork VAD chunk loop (the previous sole emitter):
//   - transcribe::asr_progress_stage builds the "asr+<arch>" stage label
//   - transcribe::run_batch_serial emits per-utterance progress and turns
//     a canceling callback into TRANSCRIBE_ERR_ABORTED with the completed
//     utterances preserved
//   - transcribe::decode_batch_slices does the same for the fast-path
//     slice-decode loop
// (The run_batch dispatcher's 0/n and n/n bookends around these loops are
// exercised end to end by the real-model batch tests; here the loops are
// driven directly, following the run_dispatch_unit.cpp fake-arch pattern.)
//
// Session is stack-constructed (transcribe_session session;) — the base
// struct's member initializers give us progress_cb=nullptr out of the box.

#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

// Captures every callback invocation for assertion.
struct Capture {
    std::vector<float>       progresses;
    std::vector<std::string> stages;
    std::vector<int64_t>     completeds;
    std::vector<int64_t>     totals;
    int                      return_code = 0;   // what the callback returns
    int                      cancel_at   = -1;  // call index that starts cancelling
    int                      call_count  = 0;
};

int capture_cb(float progress, const char * stage, int64_t completed, int64_t total, void * ud) {
    auto * c = static_cast<Capture *>(ud);
    c->progresses.push_back(progress);
    c->stages.emplace_back(stage ? stage : "");
    c->completeds.push_back(completed);
    c->totals.push_back(total);
    c->call_count += 1;
    if (c->cancel_at >= 0 && c->call_count > c->cancel_at) {
        return 1;  // request cancel
    }
    return c->return_code;
}

// Fake arch so asr_progress_stage has a name to read. Everything else is
// null; the loops under test never dispatch through it.
const transcribe::Arch k_arch = {
    /* .name             = */ "fake-asr",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ nullptr,
    /* .vad              = */ nullptr,
};

void test_setter_installs_and_clears() {
    transcribe_session session;  // member init: progress_cb = nullptr

    Capture cap;
    transcribe_set_progress_callback(&session, capture_cb, &cap);
    CHECK(session.progress_cb == capture_cb);
    CHECK(session.progress_userdata == &cap);

    // emit_progress should now route to capture_cb.
    CHECK(session.emit_progress("s", 1, 4) == 0);
    CHECK(cap.call_count == 1);
    CHECK(std::fabs(cap.progresses[0] - 0.25f) < 1e-6f);

    transcribe_set_progress_callback(&session, nullptr, nullptr);
    CHECK(session.progress_cb == nullptr);
    // After clear, emit_progress is a no-op returning 0.
    CHECK(session.emit_progress("s", 2, 4) == 0);
    CHECK(cap.call_count == 1);  // unchanged
}

void test_emit_progress_fraction_and_args() {
    transcribe_session session;

    Capture cap;
    transcribe_set_progress_callback(&session, capture_cb, &cap);

    session.emit_progress("asr+whisper", 0, 10);
    session.emit_progress("asr+whisper", 5, 10);
    session.emit_progress("asr+whisper", 10, 10);

    CHECK(cap.call_count == 3);
    CHECK(std::fabs(cap.progresses[0] - 0.0f) < 1e-6f);
    CHECK(std::fabs(cap.progresses[1] - 0.5f) < 1e-6f);
    CHECK(std::fabs(cap.progresses[2] - 1.0f) < 1e-6f);
    CHECK(cap.stages[0] == "asr+whisper");
    CHECK(cap.completeds[2] == 10);
    CHECK(cap.totals[2] == 10);

    // total <= 0 must not divide by zero; frac clamps to 0.
    cap.call_count = 0;
    session.emit_progress("x", 3, 0);
    CHECK(cap.call_count == 1);
    CHECK(std::fabs(cap.progresses[0] - 0.0f) < 1e-6f);
}

void test_cancel_return_propagates() {
    transcribe_session session;

    Capture cap;
    cap.return_code = 1;  // request cancel
    transcribe_set_progress_callback(&session, capture_cb, &cap);

    CHECK(session.emit_progress("s", 3, 10) == 1);
    CHECK(cap.call_count == 1);

    cap.return_code = 42;
    CHECK(session.emit_progress("s", 4, 10) == 42);
}

void test_null_session_setter_is_safe() {
    // Must not crash on NULL session.
    transcribe_set_progress_callback(nullptr, capture_cb, nullptr);
    transcribe_set_progress_callback(nullptr, nullptr, nullptr);
}

void test_default_session_has_no_progress_cb() {
    // A freshly-constructed session must not fire any callback (the batch
    // drivers rely on emit_progress being a no-op when unset, so existing
    // runs pay zero overhead).
    transcribe_session session;
    CHECK(session.progress_cb == nullptr);
    CHECK(session.progress_userdata == nullptr);
    CHECK(session.emit_progress("s", 1, 1) == 0);
}

// ---------------------------------------------------------------------------
// Production emission points
// ---------------------------------------------------------------------------

void test_stage_label() {
    transcribe_session session;
    CHECK(transcribe::asr_progress_stage(&session) == "asr+?");  // no model

    transcribe_model model;
    model.arch    = &k_arch;
    model.roles   = TRANSCRIBE_ROLE_ASR;
    session.model = &model;
    CHECK(transcribe::asr_progress_stage(&session) == "asr+fake-asr");
}

// A run_one stub that behaves like a family run(): writes a result into the
// scratch slot so run_batch_serial snapshots a real entry.
transcribe_status stub_run_one(transcribe_session * session) {
    session->full_text  = "utt";
    session->has_result = true;
    return TRANSCRIBE_OK;
}

void test_run_batch_serial_emits_per_utterance() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;

    transcribe_session session;
    session.model = &model;

    Capture cap;
    transcribe_set_progress_callback(&session, capture_cb, &cap);

    const float             pcm1[1] = { 0.0f }, pcm2[1] = { 0.0f }, pcm3[1] = { 0.0f };
    const float *           pcm[] = { pcm1, pcm2, pcm3 };
    const int               ns[]  = { 1, 1, 1 };
    const transcribe_status st =
        transcribe::run_batch_serial(&session, pcm, ns, 3, [&](const float *, int) { return stub_run_one(&session); });

    CHECK(st == TRANSCRIBE_OK);
    CHECK(session.was_aborted == false);
    // The dispatcher owns the 0/n and n/n bookends: the serial loop emits
    // only the intermediate boundaries 1/n .. (n-1)/n.
    CHECK(cap.call_count == 2);
    CHECK(cap.stages[0] == "asr+fake-asr" && cap.stages[1] == "asr+fake-asr");
    CHECK(cap.completeds[0] == 1 && cap.totals[0] == 3);
    CHECK(cap.completeds[1] == 2 && cap.totals[1] == 3);
    CHECK(std::fabs(cap.progresses[0] - (1.0f / 3.0f)) < 1e-6f);
    CHECK(std::fabs(cap.progresses[1] - (2.0f / 3.0f)) < 1e-6f);
    CHECK(session.batch_results.size() == 3);
}

void test_run_batch_serial_cancel_preserves_partials() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;

    transcribe_session session;
    session.model = &model;

    Capture cap;
    cap.cancel_at = 0;  // cancel from the first per-utterance emission on
    transcribe_set_progress_callback(&session, capture_cb, &cap);

    const float             pcm1[1] = { 0.0f }, pcm2[1] = { 0.0f }, pcm3[1] = { 0.0f };
    const float *           pcm[] = { pcm1, pcm2, pcm3 };
    const int               ns[]  = { 1, 1, 1 };
    const transcribe_status st =
        transcribe::run_batch_serial(&session, pcm, ns, 3, [&](const float *, int) { return stub_run_one(&session); });

    CHECK(st == TRANSCRIBE_ERR_ABORTED);
    CHECK(session.was_aborted == true);
    CHECK(cap.call_count == 1);  // loop stopped at the first emission
    // The utterance that completed is preserved; the rest were never run.
    CHECK(session.batch_results.size() == 1);
    CHECK(session.batch_results[0].status == TRANSCRIBE_OK);
    CHECK(session.batch_results[0].full_text == "utt");
}

void test_decode_batch_slices_emits_and_cancels() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;

    // Emission pass: 3 slices, all complete.
    {
        transcribe_session session;
        session.model = &model;

        Capture cap;
        transcribe_set_progress_callback(&session, capture_cb, &cap);

        const float             host[6] = { 0.0f };  // 3 slices x 2 elements
        const transcribe_status st =
            transcribe::decode_batch_slices(&session, 3, host, 2, 0, 0, [&](int, const float *) {
                session.full_text  = "slice";
                session.has_result = true;
                return TRANSCRIBE_OK;
            });
        CHECK(st == TRANSCRIBE_OK);
        CHECK(cap.call_count == 2);  // 1/3 and 2/3; dispatcher owns 0/n and 3/3
        CHECK(cap.stages[0] == "asr+fake-asr");
        CHECK(cap.completeds[0] == 1 && cap.totals[0] == 3);
        CHECK(cap.completeds[1] == 2 && cap.totals[1] == 3);
        CHECK(session.batch_results.size() == 3);
    }

    // Cancel pass: cancel at the first per-slice emission; completed slices
    // are preserved and the loop stops.
    {
        transcribe_session session;
        session.model = &model;

        Capture cap;
        cap.cancel_at = 0;
        transcribe_set_progress_callback(&session, capture_cb, &cap);

        const float             host[6] = { 0.0f };
        const transcribe_status st =
            transcribe::decode_batch_slices(&session, 3, host, 2, 0, 0, [&](int, const float *) {
                session.full_text  = "slice";
                session.has_result = true;
                return TRANSCRIBE_OK;
            });
        CHECK(st == TRANSCRIBE_ERR_ABORTED);
        CHECK(session.was_aborted == true);
        CHECK(cap.call_count == 1);
        CHECK(session.batch_results.size() == 1);
    }
}

}  // namespace

int main() {
    test_setter_installs_and_clears();
    test_emit_progress_fraction_and_args();
    test_cancel_return_propagates();
    test_null_session_setter_is_safe();
    test_default_session_has_no_progress_cb();
    test_stage_label();
    test_run_batch_serial_emits_per_utterance();
    test_run_batch_serial_cancel_preserves_partials();
    test_decode_batch_slices_emits_and_cancels();

    if (g_failures == 0) {
        std::printf("progress_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "progress_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
