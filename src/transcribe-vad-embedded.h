// transcribe-vad-embedded.h - symbols of the Silero VAD GGUF embedded in
// the library binary (CMake option TRANSCRIBE_EMBED_VAD_MODEL).
//
// INTERNAL. The bytes live in the generated source
// ${CMAKE_BINARY_DIR}/generated/transcribe-vad-embedded.cpp, emitted at
// configure time by cmake/TranscribeEmbedVad.cmake from the Silero VAD v6.2
// F32 GGUF (see docs/models/silero-vad-v6.2.md). That source is compiled
// into the transcribe target ONLY when the option is ON, so these extern
// declarations are referenced solely from code guarded by
// #ifdef TRANSCRIBE_EMBED_VAD_MODEL; nothing links against them otherwise.

#pragma once

#include <cstddef>

namespace transcribe {

// The whole GGUF file (header, KV metadata, tensor data) as a byte array.
// Static storage; valid for the life of the process. The GGUF buffer reader
// is memcpy-based, so no alignment beyond unsigned char is required.
extern const unsigned char kEmbeddedVadGguf[];

// sizeof(kEmbeddedVadGguf), kept next to the array in the generated source.
extern const size_t kEmbeddedVadGgufSize;

}  // namespace transcribe
