// Matrix types validate() in src/model.cpp accepts. tools/quantize.cpp uses the same list so it
// cannot write a file the loader refuses (for example iq4_nl).
#pragma once

#include <string>

#include "ggml.h"

namespace statim {

inline bool matrix_type_ok(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// ggml enum order, so the quantize error text stays stable.
inline std::string matrix_type_names() {
    std::string out;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto t = static_cast<ggml_type>(i);
        if (!matrix_type_ok(t)) continue;
        if (!out.empty()) out += ", ";
        out += ggml_type_name(t);
    }
    return out;
}

}  // namespace statim
