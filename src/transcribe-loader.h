// transcribe-loader.h - internal GGUF loader (architecture-agnostic).
//
// INTERNAL, C++17. The Loader lets transcribe_model_load_file answer two
// questions before committing to a per-family code path: is this a readable
// GGUF, and what architecture does it claim (general.architecture /
// stt.variant)? It is then handed to the per-arch handler, which may take
// ownership of the gguf_context via release_gguf(); if it does not, the
// Loader's destructor frees the context on stack unwinding.

#pragma once

#include "transcribe-model.h"  // transcribe::MetaMap (must match the model's type)
#include "transcribe.h"

#include <map>
#include <string>

struct gguf_context;

namespace transcribe {

class Loader {
  public:
    Loader() = default;
    ~Loader();

    // Non-copyable; move is not needed (the loader is always either
    // released to the handler or destroyed on the stack).
    Loader(const Loader &)             = delete;
    Loader & operator=(const Loader &) = delete;
    Loader(Loader &&)                  = delete;
    Loader & operator=(Loader &&)      = delete;

    // Open a GGUF file and read its identification fields.
    //
    // Returns:
    //   TRANSCRIBE_OK                  on success.
    //   TRANSCRIBE_ERR_INVALID_ARG     if path is null.
    //   TRANSCRIBE_ERR_FILE_NOT_FOUND  if (and ONLY if) a stat() pre-check
    //                                  proves the path does not exist —
    //                                  i.e. stat() fails with ENOENT or
    //                                  ENOTDIR. Other stat() failures
    //                                  (EACCES, ELOOP, ...) and a
    //                                  successful stat() of a non-regular
    //                                  file (e.g. directory) fall through
    //                                  to gguf_init_from_file and surface
    //                                  as TRANSCRIBE_ERR_GGUF below.
    //   TRANSCRIBE_ERR_GGUF            if gguf_init_from_file rejects the
    //                                  file (corrupt magic, truncated
    //                                  header, unsupported version, IO
    //                                  error, path is a directory, etc.)
    //                                  or if the required
    //                                  general.architecture KV is missing
    //                                  / wrong type.
    transcribe_status open(const char * path);

    // Adopt a gguf_context the caller created over an in-memory GGUF
    // (gguf_init_from_buffer) and read the same identification fields
    // open() does. The Loader takes ownership of `gguf`; on every non-OK
    // return the context is freed and not adopted.
    //
    // `data` / `size` record the backing buffer so a per-family load() can
    // re-open the SAME bytes for tensor data instead of a path; the buffer
    // must outlive the Loader (the embedded-asset case is static storage).
    // Keep them consistent with the context: they are only read together.
    //
    // Returns:
    //   TRANSCRIBE_OK               on success.
    //   TRANSCRIBE_ERR_INVALID_ARG  if gguf or data is null / size is 0.
    //   TRANSCRIBE_ERR_GGUF         if the required general.architecture KV
    //                               is missing / wrong type, or stt.variant
    //                               has the wrong type (same rules as open).
    transcribe_status adopt(gguf_context * gguf, const void * data, size_t size);

    // Identification fields, populated on a successful open() or adopt().
    // The arch string is required by the GGUF schema we accept; the variant
    // string is optional and may be empty for older or partial files. The
    // path stays empty for an adopted (in-memory) loader.
    const std::string & path() const { return path_; }

    const std::string & arch() const { return arch_; }

    const std::string & variant() const { return variant_; }

    // All scalar-string metadata KVs read on open() or adopt(), keyed by
    // GGUF key. Copied onto the model after dispatch and surfaced publicly
    // via transcribe_model_meta_val_str(). Never affects dispatch.
    const MetaMap & meta() const { return meta_; }

    // Borrowed pointer to the underlying gguf_context. Valid until the
    // Loader is destroyed or release_gguf() is called. Returns nullptr
    // before a successful open() / adopt().
    gguf_context * gguf() const { return gguf_; }

    // The in-memory backing recorded by adopt(), or null for a file loader.
    const void * mem_data() const { return mem_data_; }

    size_t mem_size() const { return mem_size_; }

    // Hand the gguf_context off to the per-arch handler. After this call
    // the Loader no longer owns the context and the caller must
    // gguf_free() it. The Loader is still safe to destroy.
    gguf_context * release_gguf();

  private:
    // Read arch / variant / scalar-string metadata from gguf_. Shared tail
    // of open() and adopt(). Returns TRANSCRIBE_ERR_GGUF on a missing or
    // wrong-typed general.architecture (required) or a wrong-typed
    // stt.variant (optional key, BadType still fatal — see open()).
    transcribe_status read_ident();

    std::string    path_;
    std::string    arch_;
    std::string    variant_;
    MetaMap        meta_;
    gguf_context * gguf_     = nullptr;
    const void *   mem_data_ = nullptr;
    size_t         mem_size_ = 0;
};

}  // namespace transcribe
