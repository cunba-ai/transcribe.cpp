// vad_progress_unit.cpp - white-box tests for the progress callback hook.
//
// Pins the transcribe_set_progress_callback / emit_progress contract:
//   - setter stores cb + userdata; NULL clears
//   - emit_progress computes progress fraction and forwards args verbatim
//   - non-zero callback return propagates as non-zero emit_progress return
//     (the cancellation handshake, mirroring audiocpp_progress_fn)
//
// Session is stack-constructed (transcribe_session session;) following the
// run_dispatch_unit.cpp pattern — no model needed, the base struct's member
// initializers give us progress_cb=nullptr out of the box.

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
    int                      return_code = 0;  // what the callback returns
    int                      call_count  = 0;
};

int capture_cb(float progress, const char * stage, int64_t completed, int64_t total, void * ud) {
    auto * c = static_cast<Capture *>(ud);
    c->progresses.push_back(progress);
    c->stages.emplace_back(stage ? stage : "");
    c->completeds.push_back(completed);
    c->totals.push_back(total);
    c->call_count += 1;
    return c->return_code;
}

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
    // A freshly-constructed session must not fire any callback (the VAD
    // loop and run() internals rely on emit_progress being a no-op when
    // unset, so existing runs pay zero overhead).
    transcribe_session session;
    CHECK(session.progress_cb == nullptr);
    CHECK(session.progress_userdata == nullptr);
    CHECK(session.emit_progress("s", 1, 1) == 0);
}

}  // namespace

int main() {
    test_setter_installs_and_clears();
    test_emit_progress_fraction_and_args();
    test_cancel_return_propagates();
    test_null_session_setter_is_safe();
    test_default_session_has_no_progress_cb();

    if (g_failures == 0) {
        std::printf("vad_progress_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_progress_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
