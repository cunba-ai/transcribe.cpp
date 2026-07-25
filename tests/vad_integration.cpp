// vad_integration.cpp - end-to-end VAD test against a real audiocpp.dll.
//
// NOT run in CI (no dll there — see spec §6.3). Run locally:
//   cmake -B build -S . -DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON -DTRANSCRIBE_BUILD_TESTS=ON
//   cmake --build build --target transcribe_vad_integration
//   # put audiocpp.dll next to the exe, or set TRANSCRIBE_VAD_DLL:
//   cp <audio.cpp>/audiocpp.dll build/bin/
//   ctest --test-dir build -R vad_integration --output-on-failure
//
// Returns 77 (CTest SKIP_RETURN_CODE) if the dll can't be loaded, so an
// accidental CI run is a clean SKIP rather than a FAIL.
//
// Audio fixture: pass a path via TRANSCRIBE_VAD_TEST_WAV (default samples/jfk.wav).
// Expects a 16 kHz mono s16le PCM WAV.

#include "transcribe-vad-audiocpp.h"
#include "transcribe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Minimal 16k-mono-s16le WAV loader. Avoids dr_wav (already instantiated in
// examples/common/wav.cpp) so this test stays self-contained.
bool load_wav_mono_16k_s16(const std::string & path, std::vector<float> & out_pcm, std::string & err) {
    out_pcm.clear();
    err.clear();
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    unsigned char hdr[44];
    if (std::fread(hdr, 1, 44, f) != 44) {
        std::fclose(f);
        err = "short header";
        return false;
    }
    // Validate RIFF/WAVE/fmt/data tags + PCM format + 16k mono 16-bit.
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fclose(f);
        err = "not a RIFF/WAVE file";
        return false;
    }
    // Walk chunks from offset 12 to find fmt + data. Read until both found
    // or EOF; no fixed upper bound (chunks like LIST/fact may interleave).
    unsigned short              audio_format = 0, num_channels = 0;
    unsigned int                sample_rate = 0;
    bool                        have_fmt = false;
    bool                        have_data = false;
    std::vector<unsigned char>  data;
    long                        pos = 12;
    for (int guard = 0; guard < 32; ++guard) {
        unsigned char ch[8];
        if (std::fseek(f, pos, SEEK_SET) != 0 || std::fread(ch, 1, 8, f) != 8) {
            break;
        }
        const unsigned int chunk_size = static_cast<unsigned int>(ch[4]) | (static_cast<unsigned int>(ch[5]) << 8) |
                                         (static_cast<unsigned int>(ch[6]) << 16) | (static_cast<unsigned int>(ch[7]) << 24);
        pos += 8;
        if (std::memcmp(ch, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (chunk_size >= 16 && std::fread(fmt, 1, 16, f) == 16) {
                audio_format = static_cast<unsigned short>(fmt[0] | (fmt[1] << 8));
                num_channels = static_cast<unsigned short>(fmt[2] | (fmt[3] << 8));
                sample_rate  = static_cast<unsigned int>(fmt[4]) | (static_cast<unsigned int>(fmt[5]) << 8) |
                               (static_cast<unsigned int>(fmt[6]) << 16) | (static_cast<unsigned int>(fmt[7]) << 24);
                have_fmt = true;
            }
            pos += static_cast<long>(chunk_size + (chunk_size & 1));  // pad to even
        } else if (std::memcmp(ch, "data", 4) == 0) {
            data.resize(chunk_size);
            if (chunk_size > 0 && std::fread(data.data(), 1, chunk_size, f) == chunk_size) {
                have_data = true;
            }
            break;  // data is conventionally last; stop walking
        } else {
            pos += static_cast<long>(chunk_size + (chunk_size & 1));
        }
    }
    std::fclose(f);
    if (!have_fmt) {
        err = "no fmt chunk";
        return false;
    }
    if (audio_format != 1 || num_channels != 1 || sample_rate != 16000) {
        err = "expected PCM 16k mono, got format=" + std::to_string(audio_format) +
              " ch=" + std::to_string(num_channels) + " sr=" + std::to_string(sample_rate);
        return false;
    }
    if (data.size() < 2) {
        err = "empty data chunk";
        return false;
    }
    const size_t n_samples = data.size() / 2;
    out_pcm.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        const short s = static_cast<short>(static_cast<unsigned short>(data[2 * i]) |
                                           (static_cast<unsigned short>(data[2 * i + 1]) << 8));
        out_pcm[i] = static_cast<float>(s) / 32768.0f;
    }
    return true;
}

}  // namespace

int main() {
    // Probe dll loadability first; SKIP cleanly if absent.
    std::string load_err;
    auto        loaded = transcribe::vad::audiocpp::load_audiocpp_dll(nullptr, load_err);
    if (loaded.handle == nullptr) {
        std::printf("vad_integration: SKIP (audiocpp.dll not loadable: %s)\n", load_err.c_str());
        return 77;
    }

    // Locate the test fixture. Default to meeting.wav — a long recording
    // where VAD chunking actually pays off (jfk.wav is ~11s continuous speech,
    // too short to exercise the multi-chunk path meaningfully).
    const char * wav_env = std::getenv("TRANSCRIBE_VAD_TEST_WAV");
    const std::string wav_path = wav_env && wav_env[0]
        ? std::string(wav_env)
        : std::string("E:/AI-Agent-Project/sound-rs/samples/meeting.wav");

    std::vector<float> pcm;
    std::string        wav_err;
    if (!load_wav_mono_16k_s16(wav_path, pcm, wav_err)) {
        std::printf("vad_integration: SKIP (cannot load fixture %s: %s)\n", wav_path.c_str(), wav_err.c_str());
        return 77;
    }
    std::printf("vad_integration: loaded %s (%zu samples, %.1fs)\n",
                wav_path.c_str(), pcm.size(), static_cast<double>(pcm.size()) / 16000.0);

    // Run standalone VAD via the public ABI (SILERO).
    struct transcribe_vad_params vp;
    std::memset(&vp, 0, sizeof(vp));
    vp.struct_size = sizeof(vp);
    vp.mode        = TRANSCRIBE_VAD_SILERO;

    transcribe_vad_segment * segs = nullptr;
    int64_t                  n    = 0;
    const transcribe_status st =
        transcribe_vad(pcm.data(), static_cast<int>(pcm.size()), 16000, &vp, &segs, &n);
    if (st == TRANSCRIBE_ERR_BACKEND) {
        std::printf("vad_integration: SKIP (VAD backend unavailable)\n");
        transcribe_free_vad(segs);
        return 77;
    }
    CHECK(st == TRANSCRIBE_OK);
    CHECK(n >= 1);
    int64_t total_speech_ms = 0;
    int64_t first_start = -1, last_end = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (i == 0) {
            first_start = segs[i].start_ms;
        }
        last_end = segs[i].end_ms;
        total_speech_ms += segs[i].end_ms - segs[i].start_ms;
        // Sanity: every segment within the audio bounds.
        CHECK(segs[i].start_ms >= 0);
        CHECK(segs[i].end_ms <= static_cast<int64_t>(pcm.size()) * 1000 / 16000);
        CHECK(segs[i].end_ms > segs[i].start_ms);
    }
    std::printf("vad_integration: SILERO returned %lld segment(s); first=%lldms last_end=%lldms speech=%lldms\n",
                static_cast<long long>(n), static_cast<long long>(first_start),
                static_cast<long long>(last_end), static_cast<long long>(total_speech_ms));
    // A 23-minute recording must yield a non-trivial amount of speech.
    CHECK(total_speech_ms > 60000);
    transcribe_free_vad(segs);

    // Also exercise ENERGY mode (no model; cheaper).
    vp.mode = TRANSCRIBE_VAD_ENERGY;
    segs    = nullptr;
    n       = 0;
    const transcribe_status st2 =
        transcribe_vad(pcm.data(), static_cast<int>(pcm.size()), 16000, &vp, &segs, &n);
    CHECK(st2 == TRANSCRIBE_OK);
    CHECK(n >= 1);
    std::printf("vad_integration: ENERGY returned %lld segment(s)\n", static_cast<long long>(n));
    transcribe_free_vad(segs);

    if (g_failures == 0) {
        std::printf("vad_integration: all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "vad_integration: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
}
