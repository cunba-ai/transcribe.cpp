// vad_loader_unit.cpp - load_audiocpp_dll failure-path coverage.
//
// We do NOT have audiocpp.dll in CI (see spec §6.3), so we only test the
// failure path: a missing/invalid path must return handle=nullptr with a
// non-empty err_msg, and never throw. The success path is covered by the
// local-run integration test (vad_integration.cpp).

#include "transcribe-vad-audiocpp.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void test_missing_path_fails_cleanly() {
    std::string err;
    auto r = transcribe::vad::audiocpp::load_audiocpp_dll(
        "Z:\\nonexistent\\path\\definitely_not_here.dll", err);
    CHECK(r.handle == nullptr);
    CHECK(r.syms.load_model == nullptr);
    CHECK(r.syms.vad == nullptr);
    CHECK(!err.empty());
}

void test_null_explicit_path_does_not_crash() {
    // NULL triggers discovery; on a CI box without the dll, this must also
    // fail cleanly rather than throw. We only assert it doesn't crash and
    // err is meaningful when handle==null. (A dev box with the dll on PATH
    // may legitimately return a non-null handle, so we don't assert null.)
    std::string err;
    auto r = transcribe::vad::audiocpp::load_audiocpp_dll(nullptr, err);
    if (r.handle == nullptr) {
        CHECK(!err.empty());
    }
}

}  // namespace

int main() {
    test_missing_path_fails_cleanly();
    test_null_explicit_path_does_not_crash();
    if (g_failures == 0) {
        std::printf("vad_loader_unit: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_loader_unit: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
