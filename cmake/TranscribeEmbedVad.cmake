# TranscribeEmbedVad.cmake - embed the Silero VAD model into libtranscribe.
#
# Included once from the top-level CMakeLists.txt. When
# TRANSCRIBE_EMBED_VAD_MODEL is ON, this resolves the Silero VAD v6.2 F32
# GGUF (docs/models/silero-vad-v6.2.md) at CONFIGURE time, in this priority
# order:
#
#   1. TRANSCRIBE_EMBED_VAD_MODEL_FILE (cache, explicit absolute path).
#   2. A configure-time download of TRANSCRIBE_EMBED_VAD_MODEL_URL into
#      ${CMAKE_BINARY_DIR}/model-cache/, skipped when the cached file's
#      SHA256 already matches TRANSCRIBE_EMBED_VAD_MODEL_SHA256 (the cache
#      sidecar <file>.sha256 records the verified digest).
#   3. Neither worked: FATAL_ERROR when the user explicitly requested the
#      option, or the option flips OFF with a WARNING when it was the
#      default (so offline builds still configure).
#
# On success it emits ${CMAKE_BINARY_DIR}/generated/transcribe-vad-embedded.cpp
# (see TranscribeEmbedVadGen.cmake) and exports for src/CMakeLists.txt:
#
#   TRANSCRIBE_VAD_EMBEDDED_CPP    path of the generated source
#   TRANSCRIBE_EMBED_VAD_MODEL_ON  TRUE when the asset is embedded
#
# The bytes ship as the F32 GGUF documented in the model page; the pinned
# SHA256 is over the artifact produced by scripts/convert-silero_vad.py from
# the silero-vad==6.2.3 wheel (the converter is deterministic, so a local
# regeneration yields the identical file).

# Whether the user passed the option explicitly (vs. taking the ON default).
# Recorded on the FIRST configure, then kept in the cache, so reconfigures
# do not misread a default-ON value as an explicit request.
if(NOT DEFINED TRANSCRIBE_EMBED_VAD_MODEL_EXPLICIT)
    if(DEFINED TRANSCRIBE_EMBED_VAD_MODEL)
        set(TRANSCRIBE_EMBED_VAD_MODEL_EXPLICIT ON CACHE INTERNAL "User passed TRANSCRIBE_EMBED_VAD_MODEL explicitly")
    else()
        set(TRANSCRIBE_EMBED_VAD_MODEL_EXPLICIT OFF CACHE INTERNAL "User passed TRANSCRIBE_EMBED_VAD_MODEL explicitly")
    endif()
endif()

option(TRANSCRIBE_EMBED_VAD_MODEL
    "Embed the Silero VAD GGUF into libtranscribe (transcribe_model_load_embedded_vad)" ON)

set(TRANSCRIBE_EMBED_VAD_MODEL_FILE "" CACHE FILEPATH
    "Explicit Silero VAD v6.2 GGUF to embed (skips the configure-time download)")
set(TRANSCRIBE_EMBED_VAD_MODEL_URL
    "https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf"
    CACHE STRING "Download URL of the Silero VAD v6.2 F32 GGUF to embed")
set(TRANSCRIBE_EMBED_VAD_MODEL_SHA256
    "0d871a800c420fdd5d1b17a3aea4296d1c39608f0ff4a9295eb76457eb1c88e6"
    CACHE STRING "Pinned SHA256 of the embedded Silero VAD GGUF")

set(TRANSCRIBE_VAD_EMBEDDED_CPP "")
set(TRANSCRIBE_EMBED_VAD_MODEL_ON FALSE)

if(NOT TRANSCRIBE_EMBED_VAD_MODEL)
    return()
endif()

set(_vad_gguf "")
set(_vad_fail "")

if(TRANSCRIBE_EMBED_VAD_MODEL_FILE)
    # Priority 1: explicit file. Explicit input, explicit errors.
    if(NOT EXISTS "${TRANSCRIBE_EMBED_VAD_MODEL_FILE}")
        message(FATAL_ERROR
            "TRANSCRIBE_EMBED_VAD_MODEL_FILE '${TRANSCRIBE_EMBED_VAD_MODEL_FILE}' does not exist")
    endif()
    file(SHA256 "${TRANSCRIBE_EMBED_VAD_MODEL_FILE}" _vad_sha)
    if(NOT _vad_sha STREQUAL TRANSCRIBE_EMBED_VAD_MODEL_SHA256)
        message(FATAL_ERROR
            "TRANSCRIBE_EMBED_VAD_MODEL_FILE sha256 ${_vad_sha} != pinned "
            "${TRANSCRIBE_EMBED_VAD_MODEL_SHA256}. The embedded asset must be the Silero VAD "
            "v6.2 F32 GGUF (docs/models/silero-vad-v6.2.md). Rebuild it with "
            "`uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py`, or "
            "update TRANSCRIBE_EMBED_VAD_MODEL_SHA256 if the pin is stale.")
    endif()
    set(_vad_gguf "${TRANSCRIBE_EMBED_VAD_MODEL_FILE}")
else()
    # Priority 2: build-dir cache + download.
    set(_vad_cache_dir "${CMAKE_BINARY_DIR}/model-cache")
    set(_vad_cache "${_vad_cache_dir}/silero-vad-v6.2-F32.gguf")
    set(_vad_have OFF)
    if(EXISTS "${_vad_cache}")
        file(SHA256 "${_vad_cache}" _vad_sha)
        if(_vad_sha STREQUAL TRANSCRIBE_EMBED_VAD_MODEL_SHA256)
            set(_vad_have ON)
        endif()
    endif()
    if(NOT _vad_have)
        file(MAKE_DIRECTORY "${_vad_cache_dir}")
        set(_vad_tmp "${_vad_cache}.download")
        file(REMOVE "${_vad_tmp}")
        message(STATUS "transcribe: downloading embedded VAD model from ${TRANSCRIBE_EMBED_VAD_MODEL_URL}")
        file(DOWNLOAD "${TRANSCRIBE_EMBED_VAD_MODEL_URL}" "${_vad_tmp}"
            TLS_VERIFY ON
            TIMEOUT 300
            INACTIVITY_TIMEOUT 30
            STATUS _vad_dl_status)
        list(GET _vad_dl_status 0 _vad_dl_code)
        if(NOT _vad_dl_code EQUAL 0)
            list(GET _vad_dl_status 1 _vad_dl_err)
            file(REMOVE "${_vad_tmp}")
            set(_vad_fail "download failed (code ${_vad_dl_code}: ${_vad_dl_err})")
        else()
            file(SHA256 "${_vad_tmp}" _vad_sha)
            if(NOT _vad_sha STREQUAL TRANSCRIBE_EMBED_VAD_MODEL_SHA256)
                file(REMOVE "${_vad_tmp}")
                set(_vad_fail "downloaded file sha256 ${_vad_sha} != pinned ${TRANSCRIBE_EMBED_VAD_MODEL_SHA256}")
            else()
                file(RENAME "${_vad_tmp}" "${_vad_cache}")
                set(_vad_have ON)
            endif()
        endif()
    endif()
    if(_vad_have)
        # Record the verified digest next to the cache so humans (and the
        # next configure) can see why the download was skipped.
        file(WRITE "${_vad_cache}.sha256" "${TRANSCRIBE_EMBED_VAD_MODEL_SHA256}")
        set(_vad_gguf "${_vad_cache}")
    endif()
endif()

if(_vad_gguf STREQUAL "")
    # Priority 3: nothing resolved.
    if(TRANSCRIBE_EMBED_VAD_MODEL_EXPLICIT)
        message(FATAL_ERROR
            "TRANSCRIBE_EMBED_VAD_MODEL=ON but no Silero VAD GGUF could be resolved: ${_vad_fail}. "
            "Provide one with -DTRANSCRIBE_EMBED_VAD_MODEL_FILE=<path> (see "
            "docs/models/silero-vad-v6.2.md), or build with -DTRANSCRIBE_EMBED_VAD_MODEL=OFF.")
    endif()
    message(WARNING
        "transcribe: TRANSCRIBE_EMBED_VAD_MODEL defaulting OFF - the embedded Silero VAD "
        "asset could not be resolved (${_vad_fail}). transcribe_model_load_embedded_vad() "
        "will return TRANSCRIBE_ERR_NOT_IMPLEMENTED. Pass -DTRANSCRIBE_EMBED_VAD_MODEL_FILE="
        "<path> to embed a local copy.")
    set(TRANSCRIBE_EMBED_VAD_MODEL OFF CACHE BOOL
        "Embed the Silero VAD GGUF into libtranscribe (transcribe_model_load_embedded_vad)" FORCE)
    return()
endif()

set(TRANSCRIBE_VAD_EMBEDDED_CPP "${CMAKE_BINARY_DIR}/generated/transcribe-vad-embedded.cpp")
execute_process(
    # COMMAND list elements are single argv entries; no shell quoting needed.
    COMMAND "${CMAKE_COMMAND}"
            -DGEN_INPUT=${_vad_gguf}
            -DGEN_OUTPUT=${TRANSCRIBE_VAD_EMBEDDED_CPP}
            -DGEN_SHA256=${TRANSCRIBE_EMBED_VAD_MODEL_SHA256}
            -DGEN_SOURCE_URL=${TRANSCRIBE_EMBED_VAD_MODEL_URL}
            -P "${CMAKE_CURRENT_LIST_DIR}/TranscribeEmbedVadGen.cmake"
    RESULT_VARIABLE _vad_gen_result)
if(NOT _vad_gen_result EQUAL 0)
    message(FATAL_ERROR "transcribe: generating the embedded VAD source failed (${_vad_gen_result})")
endif()

set(TRANSCRIBE_EMBED_VAD_MODEL_ON TRUE)
