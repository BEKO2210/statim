// Statim — BPE tokenizer compatible with HuggingFace `tokenizers`.
//
// Supported pipelines (exactly what tokenizer.json of the shipped models uses):
//   * Gemma / SentencePiece style: Replace(" " -> U+2581) normalizer, Metaspace
//     pre-tokenizer, BPE with byte_fallback (<0xNN> pieces) and fuse_unk.
//   * GPT-2 / ModernBERT style: NFC normalizer, ByteLevel pre-tokenizer
//     (GPT-2 regex), BPE over the byte-to-unicode alphabet.
// Added tokens are matched on the raw text (normalized=false) or on the
// normalized text (normalized=true) before pre-tokenization, leftmost-longest,
// honoring lstrip/rstrip/single_word — as HF's AddedVocabulary does.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace statim {

// One entry of tokenizer.json "added_tokens".
struct AddedToken {
    int32_t id = -1;
    std::string content;
    bool special = false;      // informational only (encode never skips specials)
    bool normalized = false;   // true: matched on normalized text, false: on raw text
    bool lstrip = false;       // match swallows Unicode whitespace to its left
    bool rstrip = false;       // match swallows Unicode whitespace to its right
    bool single_word = false;  // match only if not glued to word characters
};

// One BPE merge rule "left right -> result"; its rank is its index in
// TokenizerData::merges (lower rank merges first). All three are vocab ids.
struct BpeMerge {
    int32_t left = -1;
    int32_t right = -1;
    int32_t result = -1;
};

enum class NormalizerKind : int32_t {
    None = 0,
    Replace = 1,  // replace every occurrence of normalizer_pattern by normalizer_content
    NFC = 2,      // Unicode canonical composition
};

enum class PreTokenizerKind : int32_t {
    Metaspace = 0,  // ' ' -> metaspace, optional prepend, split before each metaspace
    ByteLevel = 1,  // GPT-2 regex split, then bytes mapped to the GPT-2 unicode alphabet
};

enum class PrependScheme : int32_t {
    Always = 0,  // prepend metaspace to every text segment that does not start with it
    First = 1,   // only to the segment starting at offset 0 of the input
    Never = 2,
};

// Everything the tokenizer needs. Filled from tokenizer.json (load_hf_json) or
// from GGUF metadata at runtime; plain data so it can be serialized 1:1.
struct TokenizerData {
    // Vocabulary, id -> piece (UTF-8), model vocab plus added tokens that are
    // not part of it; unused ids are empty strings. Byte-level vocabularies
    // store pieces in the GPT-2 byte-to-unicode alphabet (e.g. "Ġthe").
    std::vector<std::string> vocab;
    std::vector<BpeMerge> merges;          // rank = index
    std::vector<AddedToken> added;         // in tokenizer.json order
    std::string unk_token;                 // empty = no unk token
    bool byte_fallback = false;            // unknown chars -> <0xNN> byte pieces
    bool fuse_unk = false;                 // consecutive unk chars -> one unk
    bool ignore_merges = false;            // whole word in vocab -> emit directly

    NormalizerKind normalizer = NormalizerKind::None;
    std::string normalizer_pattern;        // Replace only
    std::string normalizer_content;        // Replace only

    PreTokenizerKind pre_tokenizer = PreTokenizerKind::Metaspace;
    std::string metaspace = "\xE2\x96\x81";  // Metaspace replacement char (U+2581)
    PrependScheme prepend_scheme = PrependScheme::Always;
    bool split_on_metaspace = true;        // Metaspace "split"
    bool add_prefix_space = false;         // ByteLevel: prepend ' ' to each segment
    bool use_regex = true;                 // ByteLevel: apply the GPT-2 split regex
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerData data);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Parses a HuggingFace tokenizer.json. Throws std::runtime_error on
    // unsupported components.
    static TokenizerData load_hf_json(const std::string& path);

    // Shared construction path for tokenizer.json and GGUF metadata.
    //   vocab:  id -> piece, including added tokens (holes = empty strings)
    //   merges: "left right" strings, rank = index
    //   model_json: tokenizer.json "model" object without vocab/merges
    //   normalizer_json / pre_tokenizer_json: the sub-objects (or "null")
    //   added_tokens_json: the "added_tokens" array
    // load_hf_json is exactly: split tokenizer.json into these parts + from_parts.
    static TokenizerData from_parts(std::vector<std::string> vocab,
                                    const std::vector<std::string>& merges,
                                    std::string_view model_json,
                                    std::string_view normalizer_json,
                                    std::string_view pre_tokenizer_json,
                                    std::string_view added_tokens_json);

    // Equivalent to HF `tok(text, add_special_tokens=False)["input_ids"]`.
    // max_tokens > 0 truncates to the first max_tokens ids (HF truncation=True,
    // max_length=max_tokens). Invalid UTF-8 bytes are replaced by U+FFFD.
    // Thread-safe.
    std::vector<int32_t> encode(std::string_view text, size_t max_tokens = 0) const;

    int32_t token_id(std::string_view piece) const;  // -1 if unknown; added tokens first
    size_t vocab_size() const;                       // max id + 1 incl. added tokens
    const TokenizerData& data() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace statim
