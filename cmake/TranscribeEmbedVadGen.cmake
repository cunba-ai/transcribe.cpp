# TranscribeEmbedVadGen.cmake - emit the C++ source embedding the Silero VAD
# GGUF as a byte array.
#
# Run in script mode by cmake/TranscribeEmbedVad.cmake at configure time:
#
#   cmake -DGEN_INPUT=<model.gguf>
#         -DGEN_OUTPUT=<build>/generated/transcribe-vad-embedded.cpp
#         -DGEN_SHA256=<pinned sha256>
#         -DGEN_SOURCE_URL=<download URL>
#         -DGEN_BYTES_PER_ROW=<row width>
#         -P TranscribeEmbedVadGen.cmake
#
# The output is a build-tree artifact (never committed): an extern-defined
# kEmbeddedVadGguf[] + kEmbeddedVadGgufSize matching
# src/transcribe-vad-embedded.h. The pinned SHA256 is embedded in the header
# comment; when it already matches the existing output the file is left
# untouched, so reconfigures skip the byte-array rewrite.

if(NOT DEFINED GEN_INPUT OR NOT DEFINED GEN_OUTPUT OR NOT DEFINED GEN_SHA256)
    message(FATAL_ERROR "TranscribeEmbedVadGen.cmake: GEN_INPUT, GEN_OUTPUT and GEN_SHA256 are required")
endif()

if(NOT EXISTS "${GEN_INPUT}")
    message(FATAL_ERROR "TranscribeEmbedVadGen.cmake: input '${GEN_INPUT}' does not exist")
endif()

# Fast path: the existing output was generated from the same bytes. Read the
# comment header (first 2 KB) and compare the recorded SHA256.
set(_fresh OFF)
if(EXISTS "${GEN_OUTPUT}")
    file(READ "${GEN_OUTPUT}" _head 2048)
    if(_head MATCHES "SHA256: ${GEN_SHA256}")
        set(_fresh ON)
    endif()
endif()

if(_fresh)
    return()
endif()

file(SHA256 "${GEN_INPUT}" _actual_sha)
if(NOT _actual_sha STREQUAL GEN_SHA256)
    # The module verified the hash before invoking us; this is a guard against
    # the file changing between the check and the generation.
    message(FATAL_ERROR "TranscribeEmbedVadGen.cmake: '${GEN_INPUT}' sha256 ${_actual_sha} != expected ${GEN_SHA256}")
endif()

file(SIZE "${GEN_INPUT}" _size)

if(NOT DEFINED GEN_BYTES_PER_ROW)
    set(GEN_BYTES_PER_ROW 1024)
endif()

file(WRITE "${GEN_OUTPUT}"
"// GENERATED FILE - DO NOT EDIT.
// Produced at CMake configure time by cmake/TranscribeEmbedVadGen.cmake from
// the Silero VAD v6.2 F32 GGUF. Not committed; compiled into libtranscribe
// only when TRANSCRIBE_EMBED_VAD_MODEL is ON.
//
// Source URL: ${GEN_SOURCE_URL}
// SHA256: ${GEN_SHA256}
// Size: ${_size} bytes

#include \"transcribe-vad-embedded.h\"

namespace transcribe {

const unsigned char kEmbeddedVadGguf[] = {
")

file(READ "${GEN_INPUT}" _hex HEX)

set(_pos 0)  # byte offset into the model file
while(_pos LESS _size)
    math(EXPR _take "${GEN_BYTES_PER_ROW}")
    math(EXPR _end "${_pos} + ${_take}")
    if(_end GREATER _size)
        math(EXPR _take "${_size} - ${_pos}")
    endif()
    # The hex string has two chars per byte; convert both offset and length.
    math(EXPR _hex_pos "${_pos} * 2")
    math(EXPR _hex_len "${_take} * 2")
    string(SUBSTRING "${_hex}" ${_hex_pos} ${_hex_len} _row_hex)
    math(EXPR _pos "${_pos} + ${_take}")

    # Case-insensitive classes: file(READ ... HEX) emits lowercase hex on
    # some CMake versions, uppercase on others.
    string(REGEX MATCHALL "[0-9A-Fa-f][0-9A-Fa-f]" _bytes "${_row_hex}")
    list(LENGTH _bytes _nbytes)
    if(NOT _nbytes EQUAL _take)
        # Extraction must consume exactly the bytes of this row; anything
        # else means the hex text held non-hex characters (corrupt read).
        message(FATAL_ERROR "TranscribeEmbedVadGen.cmake: hex pair extraction got ${_nbytes} pairs for ${_take} bytes at offset ${_pos}")
    endif()
    list(TRANSFORM _bytes PREPEND "0x")
    string(JOIN ", " _row ${_bytes})
    file(APPEND "${GEN_OUTPUT}" "    ${_row},\n")
endwhile()

file(APPEND "${GEN_OUTPUT}"
"};
const size_t kEmbeddedVadGgufSize = sizeof(kEmbeddedVadGguf);

}  // namespace transcribe
")

message(STATUS "transcribe: embedded VAD asset generated (${_size} bytes, sha256 ${GEN_SHA256})")
