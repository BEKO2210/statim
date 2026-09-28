#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.h"
#include "statim/tokenizer.h"

namespace statim {

namespace {
// Types are checked before every read: gguf_get_* abort on a mismatch.
std::vector<std::string> str_array(const gguf_context* g, const char* k) {
    int64_t id = gguf_find_key(g, k);
    if (id < 0) throw std::runtime_error(std::string("statim: model file is missing '") + k + "'");
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING)
        throw std::runtime_error(std::string("statim: model metadata '") + k + "' must be a string array");
    std::vector<std::string> out(gguf_get_arr_n(g, id));
    for (size_t i = 0; i < out.size(); ++i) out[i] = gguf_get_arr_str(g, id, i);
    return out;
}
std::string str(const gguf_context* g, const char* k) {
    int64_t id = gguf_find_key(g, k);
    if (id < 0) throw std::runtime_error(std::string("statim: model file is missing '") + k + "'");
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING)
        throw std::runtime_error(std::string("statim: model metadata '") + k + "' must be a string");
    return gguf_get_val_str(g, id);
}
}  // namespace

TokenizerData tokenizer_data_from_gguf(const gguf_context* g) {
    return Tokenizer::from_parts(str_array(g, "tokenizer.statim.tokens"), str_array(g, "tokenizer.statim.merges"),
                                 str(g, "tokenizer.statim.model"), str(g, "tokenizer.statim.normalizer"),
                                 str(g, "tokenizer.statim.pre_tokenizer"), str(g, "tokenizer.statim.added_tokens"));
}

}  // namespace statim
