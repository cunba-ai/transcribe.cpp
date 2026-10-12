// role_resolve_unit.cpp - load-time role validation (transcribe::resolve_roles) and the public role API (transcribe_model_roles, UNSUPPORTED_ROLE).

#include "transcribe-arch.h"
#include "transcribe-diarize.h"
#include "transcribe-langid.h"
#include "transcribe-model.h"
#include "transcribe-vad.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

transcribe_status fake_init_context(transcribe_model *, const transcribe_session_params *, transcribe_session **) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

transcribe_status fake_run(transcribe_session *, const float *, int, const transcribe_run_params *) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

transcribe_status fake_stream_finalize(transcribe_session *, transcribe_stream_update *) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

const transcribe::Arch k_asr_arch = {
    /* .name             = */ "fake_asr",
    /* .load             = */ nullptr,
    /* .init_context     = */ fake_init_context,
    /* .run              = */ fake_run,
};

// init_context without run: not a usable ASR arch.
const transcribe::Arch k_half_asr_arch = {
    /* .name             = */ "fake_half_asr",
    /* .load             = */ nullptr,
    /* .init_context     = */ fake_init_context,
};

// Stream-only ASR arch (the sortformer shape): init_context + the full
// streaming trio, no one-shot run hook. Legitimately serves the ASR session
// surface via transcribe_stream_begin; transcribe_run on it is the
// dispatcher's NOT_IMPLEMENTED.
transcribe::Arch make_stream_only_arch() {
    transcribe::Arch a = {};
    a.name             = "fake_stream_only";
    a.init_context     = fake_init_context;
    a.stream_begin     = [](transcribe_session *, const transcribe_run_params *, const transcribe_stream_params *) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    };
    a.stream_feed = [](transcribe_session *, const float *, int, transcribe_stream_update *) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    };
    a.stream_finalize = fake_stream_finalize;
    return a;
}

const transcribe::Arch k_stream_only_arch = make_stream_only_arch();

// init_context + stream_begin but no feed/finalize: an incomplete trio is
// still not a usable ASR arch.
transcribe::Arch make_partial_stream_arch() {
    transcribe::Arch a = make_stream_only_arch();
    a.name             = "fake_partial_stream";
    a.stream_feed      = nullptr;
    return a;
}

const transcribe::Arch k_partial_stream_arch = make_partial_stream_arch();

const transcribe::Arch k_no_role_arch = {
    /* .name             = */ "fake_no_role",
};

// Diarize-only: a diarize ops table and no ASR hooks. resolve_roles only
// checks the table is present, so an empty one suffices.
const transcribe::DiarizeOps k_diarize_ops = {};

transcribe::Arch make_diarize_arch() {
    transcribe::Arch a = {};
    a.name             = "fake_diarize";
    a.diarize          = &k_diarize_ops;
    return a;
}

const transcribe::Arch k_diarize_arch = make_diarize_arch();

const transcribe::LangidOps k_langid_ops = {};

transcribe::Arch make_langid_arch() {
    transcribe::Arch a = {};
    a.name             = "fake_langid";
    a.langid           = &k_langid_ops;
    return a;
}

const transcribe::Arch k_langid_arch = make_langid_arch();

const transcribe::VadOps k_vad_ops = {};

transcribe::Arch make_vad_arch() {
    transcribe::Arch a = {};
    a.name             = "fake_vad";
    a.vad              = &k_vad_ops;
    return a;
}

const transcribe::Arch k_vad_arch = make_vad_arch();

void test_resolve_roles() {
    constexpr uint32_t ASR = TRANSCRIBE_ROLE_ASR;
    constexpr uint32_t DIA = TRANSCRIBE_ROLE_DIARIZE;
    constexpr uint32_t LID = TRANSCRIBE_ROLE_LANGID;
    constexpr uint32_t VAD = TRANSCRIBE_ROLE_VAD;

    struct Case {
        const transcribe::Arch * arch;
        uint32_t                 roles_in;
        transcribe_status        want;
        uint32_t                 roles_out;  // checked on OK only
    };

    const Case cases[] = {
        { &k_asr_arch,            0,                TRANSCRIBE_OK,                  ASR }, // defaults to ASR
        { &k_asr_arch,            ASR,              TRANSCRIBE_OK,                  ASR },
        { &k_diarize_arch,        DIA,              TRANSCRIBE_OK,                  DIA },
        { &k_diarize_arch,        0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // no default for non-ASR roles
        { &k_no_role_arch,        0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_half_asr_arch,       0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_half_asr_arch,       ASR,              TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        // Stream-only ASR: the full trio satisfies the ASR session contract.
        { &k_stream_only_arch,    0,                TRANSCRIBE_OK,                  ASR },
        { &k_stream_only_arch,    ASR,              TRANSCRIBE_OK,                  ASR },
        { &k_stream_only_arch,    ASR | DIA,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // no diarize ops table
        { &k_partial_stream_arch, ASR,              TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // incomplete trio
        { &k_asr_arch,            DIA,              TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_asr_arch,            ASR | DIA,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_diarize_arch,        ASR | DIA,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_langid_arch,         LID,              TRANSCRIBE_OK,                  LID },
        { &k_langid_arch,         0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_asr_arch,            LID,              TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // no langid ops table
        { &k_diarize_arch,        DIA | LID,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_langid_arch,         ASR | LID,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_vad_arch,            VAD,              TRANSCRIBE_OK,                  VAD },
        { &k_vad_arch,            0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_langid_arch,         VAD,              TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // no vad ops table
        { &k_vad_arch,            ASR | VAD,        TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
        { &k_asr_arch,            ASR | (1u << 31), TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   }, // unknown bit
        { nullptr,                0,                TRANSCRIBE_ERR_NOT_IMPLEMENTED, 0   },
    };
    for (const Case & c : cases) {
        transcribe_model model;
        model.arch                 = c.arch;
        model.roles                = c.roles_in;
        const transcribe_status st = transcribe::resolve_roles(&model);
        CHECK(st == c.want);
        if (st == TRANSCRIBE_OK) {
            CHECK(model.roles == c.roles_out);
        }
    }
}

// ASR entry points refuse a model that does not serve ASR, before
// touching params or the arch; capabilities leave the caller's struct alone.
void test_asr_entry_points_check_role() {
    transcribe_model model;
    model.arch  = &k_asr_arch;
    model.roles = TRANSCRIBE_ROLE_DIARIZE;  // not ASR

    transcribe_session * s = reinterpret_cast<transcribe_session *>(0x1);
    CHECK(transcribe_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    caps.max_audio_ms = 1234;  // sentinel: must survive the rejection
    CHECK(transcribe_model_get_capabilities(&model, &caps) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(caps.max_audio_ms == 1234);

    CHECK(transcribe_model_roles(&model) == TRANSCRIBE_ROLE_DIARIZE);

    // Same model with the ASR bit: capabilities succeed; session_init gets
    // past the role check to the fake arch's init_context.
    model.roles = TRANSCRIBE_ROLE_ASR;
    CHECK(transcribe_model_get_capabilities(&model, &caps) == TRANSCRIBE_OK);
    CHECK(transcribe_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(s == nullptr);
    CHECK(transcribe_model_roles(&model) == TRANSCRIBE_ROLE_ASR);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);  // the rejections log at ERROR

    test_resolve_roles();
    test_asr_entry_points_check_role();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("role_resolve_unit: ok\n");
    return EXIT_SUCCESS;
}
