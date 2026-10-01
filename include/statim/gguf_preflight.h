#pragma once

#include <cstddef>
#include <string>

namespace statim {

// Validate the on-disk structure before ggml deserializes any enums or tensor shapes.
void gguf_preflight(const std::string& path);

// The same check on a file already in memory (a MappedFile): the caller then hands this exact
// mapping to gguf_init_from_buffer, so the preflight and ggml see the same bytes.
void gguf_preflight(const void* data, size_t size);

}  // namespace statim
