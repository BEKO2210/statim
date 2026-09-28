// libFuzzer harness: Tokenizer::encode on arbitrary bytes (valid and invalid UTF-8).
//
// Tokenizers: the tiny Metaspace and ByteLevel models' (always), plus every tokenizer.json in
// $STATIM_FUZZ_TOKENIZERS (colon-separated) or, by default, the real Laya tokenizers under
// models/ when fetched (tools/fetch_models.sh).
//
// The first input byte picks max_tokens (0 = no truncation), the rest is the text.
// Invariants: never throws, every id is inside the vocabulary, and truncation returns exactly
// the prefix of the untruncated encoding (HF truncation=True, max_length=n).
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "statim/model.h"
#include "statim/tokenizer.h"

using namespace statim;

namespace {
std::vector<std::shared_ptr<Model>>* g_hold = nullptr;
std::vector<const Tokenizer*>* g_toks = nullptr;
}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    g_hold = new std::vector<std::shared_ptr<Model>>();
    g_toks = new std::vector<const Tokenizer*>();
    for (const char* name : {"tiny-metaspace.gguf", "tiny-bytelevel.gguf"}) {
        g_hold->push_back(Model::load(statim_fuzz::data_dir() + "/" + name, "cpu"));
        g_toks->push_back(&g_hold->back()->tokenizer());
    }
    std::string paths;
    if (const char* env = std::getenv("STATIM_FUZZ_TOKENIZERS")) paths = env;
    else paths = std::string(STATIM_SOURCE_DIR) + "/models/laya-multilingual/tokenizer/tokenizer.json:" +
                 STATIM_SOURCE_DIR + "/models/laya/tokenizer/tokenizer.json";
    std::stringstream ss(paths);
    for (std::string p; std::getline(ss, p, ':');) {
        if (p.empty() || !std::ifstream(p)) {
            if (!p.empty()) std::fprintf(stderr, "fuzz_tokenizer: skipping missing %s\n", p.c_str());
            continue;
        }
        g_toks->push_back(new Tokenizer(Tokenizer::load_hf_json(p)));
        std::fprintf(stderr, "fuzz_tokenizer: loaded %s\n", p.c_str());
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;
    const size_t max_tokens = data[0] < 128 ? 0 : (data[0] & 63) + 1;
    const std::string_view text(reinterpret_cast<const char*>(data) + 1, size - 1);
    for (const Tokenizer* tok : *g_toks) {
        const std::vector<int32_t> ids = tok->encode(text);
        const auto vocab = static_cast<int32_t>(tok->vocab_size());
        for (int32_t id : ids)
            if (id < 0 || id >= vocab) statim_fuzz::violated("token id outside the vocabulary");
        if (max_tokens) {
            const std::vector<int32_t> cut = tok->encode(text, max_tokens);
            const size_t n = std::min(max_tokens, ids.size());
            if (cut.size() != n || !std::equal(cut.begin(), cut.end(), ids.begin()))
                statim_fuzz::violated("truncated encoding is not a prefix of the full encoding");
        }
    }
    return 0;
}
