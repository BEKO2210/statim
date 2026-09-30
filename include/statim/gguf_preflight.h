#pragma once

#include <string>

namespace statim {

// Validate the on-disk structure before ggml deserializes any enums or tensor shapes.
void gguf_preflight(const std::string& path);

}  // namespace statim
