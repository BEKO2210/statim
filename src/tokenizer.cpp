// Statim tokenizer — HuggingFace `tokenizers` compatible BPE.
// Control flow mirrors tokenizers 0.2x: AddedVocabulary::extract_and_normalize
// -> pre-tokenizer -> BPE::merge_word/Word::merge_all.
#include "statim/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

#include "json.hpp"

namespace statim {
namespace {

#include "unicode_tables.inc"

// ---------------------------------------------------------------- UTF-8

inline size_t utf8_seq_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Decodes one code point from valid UTF-8.
inline uint32_t utf8_decode(const char* p, size_t& len) {
    const auto* s = reinterpret_cast<const uint8_t*>(p);
    len = utf8_seq_len(s[0]);
    switch (len) {
        case 1: return s[0];
        case 2: return ((s[0] & 0x1Fu) << 6) | (s[1] & 0x3Fu);
        case 3: return ((s[0] & 0x0Fu) << 12) | ((s[1] & 0x3Fu) << 6) | (s[2] & 0x3Fu);
        default:
            return ((s[0] & 0x07u) << 18) | ((s[1] & 0x3Fu) << 12) | ((s[2] & 0x3Fu) << 6) |
                   (s[3] & 0x3Fu);
    }
}

// Length of a valid UTF-8 sequence at s[i], or 0 if invalid.
inline size_t utf8_valid_len(const uint8_t* s, size_t n, size_t i) {
    uint8_t c = s[i];
    if (c < 0x80) return 1;
    auto cont = [&](size_t k) { return i + k < n && (s[i + k] & 0xC0) == 0x80; };
    if (c >= 0xC2 && c <= 0xDF) return cont(1) ? 2 : 0;
    if (c >= 0xE0 && c <= 0xEF) {
        if (!cont(1) || !cont(2)) return 0;
        uint8_t c1 = s[i + 1];
        if (c == 0xE0 && c1 < 0xA0) return 0;   // overlong
        if (c == 0xED && c1 >= 0xA0) return 0;  // surrogates
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (!cont(1) || !cont(2) || !cont(3)) return 0;
        uint8_t c1 = s[i + 1];
        if (c == 0xF0 && c1 < 0x90) return 0;
        if (c == 0xF4 && c1 >= 0x90) return 0;
        return 4;
    }
    return 0;
}

// Returns true if `in` was valid; otherwise writes a copy with U+FFFD for
// every invalid byte into `out`.
bool sanitize_utf8(std::string_view in, std::string& out) {
    const auto* s = reinterpret_cast<const uint8_t*>(in.data());
    size_t n = in.size(), i = 0;
    while (i < n) {
        if (s[i] < 0x80) { ++i; continue; }
        size_t l = utf8_valid_len(s, n, i);
        if (l == 0) break;
        i += l;
    }
    if (i == n) return true;
    out.assign(in.data(), i);
    while (i < n) {
        size_t l = utf8_valid_len(s, n, i);
        if (l == 0) { out += "\xEF\xBF\xBD"; ++i; continue; }
        out.append(in.data() + i, l);
        i += l;
    }
    return false;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ---------------------------------------------------------------- Unicode classes

// Rust char::is_whitespace == Oniguruma \s (Unicode White_Space).
inline bool is_ws(uint32_t c) {
    if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D);
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

enum CharClass : uint8_t { kLetter = 0, kNumber = 1, kSpace = 2, kOther = 3 };

// Class as seen by HF's Oniguruma GPT-2 regex (\p{L}, \p{N}, \s).
inline CharClass classify(uint32_t c) {
    if (c < 0x80) {
        if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') return kLetter;
        if (c >= '0' && c <= '9') return kNumber;
        if (is_ws(c)) return kSpace;
        return kOther;
    }
    const uint32_t* end = kClassRuns + std::size(kClassRuns);
    const uint32_t* it = std::upper_bound(kClassRuns, end, c,
                                          [](uint32_t v, uint32_t e) { return v < (e >> 2); });
    return static_cast<CharClass>(*(it - 1) & 3);
}

// Approximates Rust char::is_alphanumeric (used only for single_word tokens).
inline bool is_alnum(uint32_t c) {
    CharClass k = classify(c);
    return k == kLetter || k == kNumber;
}

// ---------------------------------------------------------------- NFC (HF-exact tables)

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount,
                   kSCount = kLCount * kNCount;

inline uint32_t ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    const uint32_t* end = kCcc + std::size(kCcc);
    const uint32_t* it = std::lower_bound(kCcc, end, cp, [](uint32_t e, uint32_t v) { return (e >> 8) < v; });
    return (it != end && (*it >> 8) == cp) ? (*it & 0xFF) : 0;
}

inline bool nfc_relevant(uint32_t cp) {
    if (cp < 0x300) return false;
    size_t lo = 0, hi = std::size(kNfcRelevant);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kNfcRelevant[mid][1] < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < std::size(kNfcRelevant) && kNfcRelevant[lo][0] <= cp;
}

inline int64_t compose_pair(uint32_t a, uint32_t b) {
    if (a - kLBase < kLCount && b - kVBase < kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a - kSBase < kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    uint64_t key = static_cast<uint64_t>(a) << 21 | b;
    const ComposeEntry* end = kCompose + std::size(kCompose);
    const ComposeEntry* it = std::lower_bound(kCompose, end, key,
                                              [](const ComposeEntry& e, uint64_t k) { return e.key < k; });
    return (it != end && it->key == key) ? static_cast<int64_t>(it->cp) : -1;
}

void decompose_into(uint32_t cp, std::vector<uint32_t>& out) {
    if (cp - kSBase < kSCount) {
        uint32_t si = cp - kSBase;
        out.push_back(kLBase + si / kNCount);
        out.push_back(kVBase + (si % kNCount) / kTCount);
        if (si % kTCount) out.push_back(kTBase + si % kTCount);
        return;
    }
    const DecompEntry* end = kDecomp + std::size(kDecomp);
    const DecompEntry* it = std::lower_bound(kDecomp, end, cp,
                                             [](const DecompEntry& e, uint32_t v) { return e.cp < v; });
    if (it != end && it->cp == cp) {
        out.insert(out.end(), kDecompData + it->off, kDecompData + it->off + it->len);
    } else {
        out.push_back(cp);
    }
}

// NFC exactly as HF's normalizers.NFC (unicode-normalization-alignments data).
std::string nfc(std::string_view s) {
    bool simple = true;
    for (unsigned char c : s) {
        if (c >= 0xCC) { simple = false; break; }  // lead bytes >= 0xCC encode >= U+0300
    }
    if (simple) return std::string(s);
    bool relevant = false;
    for (size_t i = 0; i < s.size() && !relevant;) {
        size_t len;
        relevant = nfc_relevant(utf8_decode(s.data() + i, len));
        i += len;
    }
    if (!relevant) return std::string(s);
    thread_local std::vector<uint32_t> buf;
    buf.clear();
    for (size_t i = 0; i < s.size();) {
        size_t len;
        decompose_into(utf8_decode(s.data() + i, len), buf);
        i += len;
    }
    // Canonical ordering: stable sort of every run of non-starters.
    for (size_t i = 0; i < buf.size();) {
        if (ccc_of(buf[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < buf.size() && ccc_of(buf[j]) != 0) ++j;
        std::stable_sort(buf.begin() + i, buf.begin() + j,
                         [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
        i = j;
    }
    // Canonical composition.
    std::string out;
    out.reserve(s.size());
    if (!buf.empty()) {
        size_t w = 1;
        int64_t starter = ccc_of(buf[0]) == 0 ? 0 : -1;
        uint32_t last = starter == 0 ? 0 : 256;
        for (size_t r = 1; r < buf.size(); ++r) {
            uint32_t cp = buf[r], cc = ccc_of(cp);
            if (starter >= 0 && (last < cc || last == 0)) {
                int64_t comp = compose_pair(buf[starter], cp);
                if (comp >= 0) { buf[starter] = static_cast<uint32_t>(comp); continue; }
            }
            if (cc == 0) starter = static_cast<int64_t>(w);
            last = cc;
            buf[w++] = cp;
        }
        buf.resize(w);
    }
    for (uint32_t cp : buf) append_utf8(out, cp);
    return out;
}

std::string replace_all(std::string_view s, std::string_view pat, std::string_view rep) {
    std::string r;
    r.reserve(s.size() + s.size() / 4);
    size_t i = 0;
    while (true) {
        size_t j = s.find(pat, i);
        if (j == std::string_view::npos) break;
        r.append(s.data() + i, j - i);
        r.append(rep);
        i = j + pat.size();
    }
    r.append(s.data() + i, s.size() - i);
    return r;
}

// ---------------------------------------------------------------- added-token matcher

// Byte trie with Aho-Corasick "leftmost-longest" semantics.
class Matcher {
public:
    void add(std::string_view pat, int32_t token_index) {
        if (pat.empty()) return;
        if (nodes_.empty()) nodes_.push_back(-1);
        int32_t node = 0;
        for (unsigned char c : pat) {
            uint64_t key = (static_cast<uint64_t>(node) << 8) | c;
            auto it = edges_.find(key);
            if (it == edges_.end()) {
                int32_t nn = static_cast<int32_t>(nodes_.size());
                nodes_.push_back(-1);
                edges_.emplace(key, nn);
                node = nn;
            } else {
                node = it->second;
            }
        }
        if (nodes_[node] < 0) nodes_[node] = token_index;  // first pattern wins on duplicates
        first_[static_cast<unsigned char>(pat[0])] = true;
    }

    bool empty() const { return nodes_.empty(); }

    // Finds the leftmost match starting at or after `from`; longest among those.
    bool next(std::string_view s, size_t from, size_t& mstart, size_t& mend, int32_t& tok) const {
        const size_t n = s.size();
        for (size_t i = from; i < n; ++i) {
            if (!first_[static_cast<unsigned char>(s[i])]) continue;
            int32_t node = 0, best = -1;
            size_t best_end = 0;
            for (size_t k = i; k < n; ++k) {
                uint64_t key = (static_cast<uint64_t>(node) << 8) | static_cast<unsigned char>(s[k]);
                auto it = edges_.find(key);
                if (it == edges_.end()) break;
                node = it->second;
                if (nodes_[node] >= 0) { best = nodes_[node]; best_end = k + 1; }
            }
            if (best >= 0) {
                mstart = i;
                mend = best_end;
                tok = best;
                return true;
            }
        }
        return false;
    }

private:
    std::vector<int32_t> nodes_;  // node -> token index or -1
    std::unordered_map<uint64_t, int32_t> edges_;
    std::array<bool, 256> first_{};
};

struct Segment {
    size_t begin, end;
    int32_t token;  // index into added tokens, -1 = plain text
};

// Byte index of the leftmost char of the trailing whitespace run, or s.size().
size_t space_leftmost_at_end(std::string_view s) {
    size_t i = s.size(), res = s.size();
    while (i > 0) {
        size_t j = i - 1;
        while (j > 0 && (static_cast<unsigned char>(s[j]) & 0xC0) == 0x80) --j;
        size_t len;
        if (!is_ws(utf8_decode(s.data() + j, len))) break;
        res = j;
        i = j;
    }
    return res;
}

size_t space_rightmost_at_start(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t len;
        if (!is_ws(utf8_decode(s.data() + i, len))) break;
        i += len;
    }
    return i;
}

uint32_t last_char(std::string_view s) {
    size_t j = s.size() - 1;
    while (j > 0 && (static_cast<unsigned char>(s[j]) & 0xC0) == 0x80) --j;
    size_t len;
    return utf8_decode(s.data() + j, len);
}

// ---------------------------------------------------------------- GPT-2 byte-level

std::array<std::string, 256> make_byte_alphabet() {
    std::array<std::string, 256> m;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = keep ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
        append_utf8(m[b], cp);
    }
    return m;
}

// Splits s like the GPT-2 regex
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Appends [begin, end) byte ranges.
void gpt2_split(std::string_view s, std::vector<std::pair<size_t, size_t>>& out,
                std::vector<uint32_t>& cps, std::vector<uint32_t>& offs,
                std::vector<uint8_t>& cls) {
    cps.clear(); offs.clear(); cls.clear();
    for (size_t i = 0; i < s.size();) {
        size_t len;
        uint32_t cp = utf8_decode(s.data() + i, len);
        cps.push_back(cp);
        offs.push_back(static_cast<uint32_t>(i));
        cls.push_back(classify(cp));
        i += len;
    }
    const size_t n = cps.size();
    offs.push_back(static_cast<uint32_t>(s.size()));
    size_t i = 0;
    while (i < n) {
        size_t end = i;
        uint32_t c = cps[i];
        if (c == '\'' && i + 1 < n) {
            uint32_t c1 = cps[i + 1];
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                end = i + 2;
            } else if (i + 2 < n) {
                uint32_t c2 = cps[i + 2];
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                    end = i + 3;
            }
        }
        if (end == i) {
            size_t k = (c == ' ' && i + 1 < n && cls[i + 1] != kSpace) ? i + 1 : i;
            uint8_t k_cls = cls[k];
            if (k_cls != kSpace) {
                end = k + 1;
                if (k_cls == kOther) {
                    while (end < n && cls[end] == kOther) ++end;
                } else {
                    while (end < n && cls[end] == k_cls) ++end;
                }
            } else {
                size_t j = i;
                while (j < n && cls[j] == kSpace) ++j;
                end = (j == n || j - i == 1) ? j : j - 1;
            }
        }
        out.emplace_back(offs[i], offs[end]);
        i = end;
    }
}

// Open-addressing index over a vector of strings: piece -> id (first wins).
// Holds a pointer to the vector, which must outlive the index.
class PieceIndex {
public:
    void build(const std::vector<std::string>& keys) {
        keys_ = &keys;
        size_t cap = 16;
        while (cap < keys.size() * 2) cap <<= 1;
        mask_ = cap - 1;
        slots_.assign(cap, Slot{0, 0});
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].empty()) continue;
            uint64_t h = hash(keys[i]);
            uint32_t tag = static_cast<uint32_t>(h >> 32);
            for (size_t p = h & mask_;; p = (p + 1) & mask_) {
                Slot& sl = slots_[p];
                if (sl.id1 == 0) { sl = {static_cast<uint32_t>(i + 1), tag}; break; }
                if (sl.tag == tag && keys[sl.id1 - 1] == keys[i]) break;
            }
        }
    }

    int32_t find(std::string_view k) const {
        if (slots_.empty()) return -1;
        uint64_t h = hash(k);
        uint32_t tag = static_cast<uint32_t>(h >> 32);
        for (size_t p = h & mask_;; p = (p + 1) & mask_) {
            const Slot& sl = slots_[p];
            if (sl.id1 == 0) return -1;
            if (sl.tag == tag && (*keys_)[sl.id1 - 1] == k) return static_cast<int32_t>(sl.id1 - 1);
        }
    }

private:
    struct Slot {
        uint32_t id1;  // id + 1, 0 = empty
        uint32_t tag;
    };
    static uint64_t hash(std::string_view s) { return std::hash<std::string_view>{}(s); }
    const std::vector<std::string>* keys_ = nullptr;
    std::vector<Slot> slots_;
    size_t mask_ = 0;
};

struct MergeInfo {
    uint64_t key;
    int32_t rank;
    int32_t result;
};

// Open-addressing map (left, right) -> (rank, result). Later duplicates
// overwrite earlier ones, as in HF's HashMap collect.
class PairMap {
public:
    void build(const std::vector<BpeMerge>& merges) {
        size_t cap = 16;
        while (cap < merges.size() * 2) cap <<= 1;
        mask_ = cap - 1;
        slots_.assign(cap, MergeInfo{kEmpty, 0, 0});
        for (size_t r = 0; r < merges.size(); ++r) {
            uint64_t key = pair_key(merges[r].left, merges[r].right);
            for (size_t p = mix(key) & mask_;; p = (p + 1) & mask_) {
                MergeInfo& sl = slots_[p];
                if (sl.key == kEmpty || sl.key == key) {
                    sl = {key, static_cast<int32_t>(r), merges[r].result};
                    break;
                }
            }
        }
    }

    const MergeInfo* find(int32_t a, int32_t b) const {
        if (slots_.empty()) return nullptr;
        uint64_t key = pair_key(a, b);
        for (size_t p = mix(key) & mask_;; p = (p + 1) & mask_) {
            const MergeInfo& sl = slots_[p];
            if (sl.key == key) return &sl;
            if (sl.key == kEmpty) return nullptr;
        }
    }

    static uint64_t pair_key(int32_t a, int32_t b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

private:
    static constexpr uint64_t kEmpty = ~0ull;
    static uint64_t mix(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdull;
        k ^= k >> 33;
        return k;
    }
    std::vector<MergeInfo> slots_;
    size_t mask_ = 0;
};

struct Symbol {
    int32_t id, prev, next, len;
};

struct QItem {
    int32_t rank, pos, result;
    bool operator<(const QItem& o) const {  // max-heap pops lowest rank, then lowest pos
        return rank != o.rank ? rank > o.rank : pos > o.pos;
    }
};

}  // namespace

// ---------------------------------------------------------------- Impl

struct Tokenizer::Impl {
    TokenizerData d;
    PieceIndex vocab_map;                                     // piece -> id
    std::unordered_map<std::string_view, int32_t> added_map;  // added content -> id
    PairMap merge_map;
    std::array<int32_t, 256> byte_ids{};
    bool all_bytes = false;
    int32_t unk_id = -1;
    size_t vsize = 0;
    Matcher raw_matcher, norm_matcher;
    std::array<std::string, 256> byte_alphabet;

    static constexpr size_t kCacheCap = 1 << 16;
    static constexpr size_t kCacheMaxWord = 128;
    mutable std::shared_mutex cache_mu;
    struct SvHash {
        using is_transparent = void;
        size_t operator()(std::string_view v) const { return std::hash<std::string_view>{}(v); }
    };
    mutable std::unordered_map<std::string, std::vector<int32_t>, SvHash, std::equal_to<>> cache;

    explicit Impl(TokenizerData data) : d(std::move(data)) {
        vocab_map.build(d.vocab);
        merge_map.build(d.merges);
        all_bytes = true;
        for (int b = 0; b < 256; ++b) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            byte_ids[b] = vocab_map.find(buf);
            if (byte_ids[b] < 0) all_bytes = false;
        }
        if (!d.unk_token.empty()) {
            unk_id = vocab_map.find(d.unk_token);
        }
        vsize = d.vocab.size();
        // HF keeps one entry per content: the last definition wins.
        std::unordered_map<std::string_view, int32_t> last;
        for (size_t i = 0; i < d.added.size(); ++i) {
            const AddedToken& t = d.added[i];
            if (t.content.empty()) continue;
            last[t.content] = static_cast<int32_t>(i);
            added_map[t.content] = t.id;
            vsize = std::max(vsize, static_cast<size_t>(t.id) + 1);
        }
        for (size_t i = 0; i < d.added.size(); ++i) {
            const AddedToken& t = d.added[i];
            if (t.content.empty() || last[t.content] != static_cast<int32_t>(i)) continue;
            if (t.normalized) {
                norm_matcher.add(normalize(t.content), static_cast<int32_t>(i));
            } else {
                raw_matcher.add(t.content, static_cast<int32_t>(i));
            }
        }
        byte_alphabet = make_byte_alphabet();
    }

    std::string normalize(std::string_view s) const {
        switch (d.normalizer) {
            case NormalizerKind::NFC: return nfc(s);
            case NormalizerKind::Replace:
                return replace_all(s, d.normalizer_pattern, d.normalizer_content);
            default: return std::string(s);
        }
    }

    // AddedVocabulary::find_matches
    void find_matches(std::string_view s, const Matcher& m, std::vector<Segment>& out) const {
        out.clear();
        if (s.empty()) return;
        size_t start_offset = 0, pos = 0, ms, me;
        int32_t tok;
        while (!m.empty() && m.next(s, pos, ms, me, tok)) {
            pos = me;
            size_t start = ms, stop = me;
            const AddedToken& t = d.added[tok];
            if (t.single_word) {
                bool start_space = start == 0 || !is_alnum(last_char(s.substr(0, start)));
                bool stop_space = stop == s.size();
                if (!stop_space) {
                    size_t l;
                    stop_space = !is_alnum(utf8_decode(s.data() + stop, l));
                }
                if (!start_space || !stop_space) continue;
            }
            if (t.lstrip) start = std::max(space_leftmost_at_end(s.substr(0, start)), start_offset);
            if (t.rstrip) stop += space_rightmost_at_start(s.substr(stop));
            if (start_offset < start) out.push_back({start_offset, start, -1});
            out.push_back({start, stop, tok});
            start_offset = stop;
        }
        if (start_offset < s.size()) out.push_back({start_offset, s.size(), -1});
    }

    // BPE::merge_word + Word::merge_all
    void bpe(std::string_view w, std::vector<int32_t>& ids) const {
        if (d.ignore_merges) {
            int32_t id = vocab_map.find(w);
            if (id >= 0) { ids.push_back(id); return; }
        }
        const bool cacheable = w.size() <= kCacheMaxWord;
        if (cacheable) {
            std::shared_lock lk(cache_mu);
            auto it = cache.find(w);
            if (it != cache.end()) {
                ids.insert(ids.end(), it->second.begin(), it->second.end());
                return;
            }
        }

        thread_local std::vector<Symbol> sym;
        thread_local std::vector<QItem> heap;
        sym.clear();
        heap.clear();
        auto add = [&](int32_t id, int32_t len) {
            int32_t n = static_cast<int32_t>(sym.size());
            if (n > 0) sym.back().next = n;
            sym.push_back({id, n - 1, -1, len});
        };
        int32_t pend_unk = -1, pend_len = 0;  // fused unk not yet emitted
        for (size_t i = 0; i < w.size();) {
            size_t len = utf8_seq_len(static_cast<unsigned char>(w[i]));
            std::string_view ch = w.substr(i, len);
            i += len;
            int32_t cid = vocab_map.find(ch);
            if (cid >= 0) {
                if (pend_unk >= 0) { add(pend_unk, pend_len); pend_unk = -1; }
                add(cid, static_cast<int32_t>(len));
                continue;
            }
            if (d.byte_fallback) {
                bool ok = all_bytes;
                if (!ok) {
                    ok = true;
                    for (unsigned char b : ch) ok = ok && byte_ids[b] >= 0;
                }
                if (ok) {
                    // HF quirk: a pending unk is not flushed before byte pieces.
                    for (unsigned char b : ch) add(byte_ids[b], 1);
                    continue;
                }
            }
            if (unk_id >= 0) {
                if (pend_unk >= 0 && d.fuse_unk) {
                    pend_len += static_cast<int32_t>(len);
                } else {
                    if (pend_unk >= 0) add(pend_unk, pend_len);
                    pend_unk = unk_id;
                    pend_len = static_cast<int32_t>(len);
                }
            }
        }
        if (pend_unk >= 0) add(pend_unk, pend_len);

        const int32_t n = static_cast<int32_t>(sym.size());
        for (int32_t i = 0; i + 1 < n; ++i) {
            if (const MergeInfo* m = merge_map.find(sym[i].id, sym[i + 1].id))
                heap.push_back({m->rank, i, m->result});
        }
        std::make_heap(heap.begin(), heap.end());
        auto push = [&](QItem q) { heap.push_back(q); std::push_heap(heap.begin(), heap.end()); };
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end());
            QItem top = heap.back();
            heap.pop_back();
            Symbol& cur = sym[top.pos];
            if (cur.len == 0 || cur.next == -1) continue;
            const Symbol right = sym[cur.next];
            const MergeInfo* m = merge_map.find(cur.id, right.id);
            if (!m || m->result != top.result) continue;
            sym[cur.next].len = 0;
            cur.id = top.result;
            cur.len += right.len;
            cur.next = right.next;
            if (right.next >= 0 && right.next < n) sym[right.next].prev = top.pos;
            if (cur.prev >= 0) {
                if (const MergeInfo* p = merge_map.find(sym[cur.prev].id, cur.id))
                    push({p->rank, cur.prev, p->result});
            }
            if (cur.next >= 0 && cur.next < n) {
                if (const MergeInfo* q = merge_map.find(cur.id, sym[cur.next].id))
                    push({q->rank, top.pos, q->result});
            }
        }
        const size_t first = ids.size();
        for (const Symbol& s : sym) {
            if (s.len != 0) ids.push_back(s.id);
        }
        if (cacheable) {
            std::unique_lock lk(cache_mu);
            if (cache.size() < kCacheCap)
                cache.emplace(std::string(w), std::vector<int32_t>(ids.begin() + first, ids.end()));
        }
    }

    // Pre-tokenizes one normalized, token-free segment and runs BPE on each word.
    void encode_segment(std::string_view s, bool at_origin, std::vector<int32_t>& ids) const {
        if (s.empty()) return;
        if (d.pre_tokenizer == PreTokenizerKind::Metaspace) {
            std::string buf = s.find(' ') == std::string_view::npos
                                  ? std::string(s)
                                  : replace_all(s, " ", d.metaspace);
            const std::string_view ms = d.metaspace;
            bool starts = buf.compare(0, ms.size(), ms) == 0;
            if (!starts && (d.prepend_scheme == PrependScheme::Always ||
                            (d.prepend_scheme == PrependScheme::First && at_origin)))
                buf.insert(0, ms);
            std::string_view v = buf;
            if (!d.split_on_metaspace || ms.empty()) { bpe(v, ids); return; }
            // SplitDelimiterBehavior::MergedWithNext
            size_t begin = 0;
            size_t p = v.find(ms, 1);
            while (p != std::string_view::npos) {
                bpe(v.substr(begin, p - begin), ids);
                begin = p;
                p = v.find(ms, p + ms.size());
            }
            if (begin < v.size()) bpe(v.substr(begin), ids);
            return;
        }
        // ByteLevel
        std::string pre;
        if (d.add_prefix_space && s[0] != ' ') {
            pre.reserve(s.size() + 1);
            pre += ' ';
            pre.append(s);
            s = pre;
        }
        thread_local std::vector<std::pair<size_t, size_t>> pieces;
        thread_local std::vector<uint32_t> cps, offs;
        thread_local std::vector<uint8_t> cls;
        pieces.clear();
        if (d.use_regex) {
            gpt2_split(s, pieces, cps, offs, cls);
        } else {
            pieces.emplace_back(0, s.size());
        }
        std::string mapped;
        for (auto [b, e] : pieces) {
            mapped.clear();
            for (size_t k = b; k < e; ++k) mapped += byte_alphabet[static_cast<unsigned char>(s[k])];
            bpe(mapped, ids);
        }
    }

    std::vector<int32_t> encode(std::string_view text, size_t max_tokens) const {
        std::vector<int32_t> ids;
        std::string clean;
        if (!sanitize_utf8(text, clean)) text = clean;
        auto done = [&] { return max_tokens > 0 && ids.size() >= max_tokens; };

        std::vector<Segment> raw_segs, norm_segs;
        find_matches(text, raw_matcher, raw_segs);
        for (const Segment& rs : raw_segs) {
            if (done()) break;
            if (rs.token >= 0) { ids.push_back(d.added[rs.token].id); continue; }
            if (rs.begin == rs.end) continue;
            std::string norm = normalize(text.substr(rs.begin, rs.end - rs.begin));
            find_matches(norm, norm_matcher, norm_segs);
            for (const Segment& ns : norm_segs) {
                if (done()) break;
                if (ns.token >= 0) { ids.push_back(d.added[ns.token].id); continue; }
                if (ns.begin == ns.end) continue;
                encode_segment(std::string_view(norm).substr(ns.begin, ns.end - ns.begin),
                               rs.begin == 0 && ns.begin == 0, ids);
            }
        }
        if (max_tokens > 0 && ids.size() > max_tokens) ids.resize(max_tokens);
        return ids;
    }
};

// ---------------------------------------------------------------- public API

Tokenizer::Tokenizer(TokenizerData data) : impl_(std::make_unique<Impl>(std::move(data))) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text, size_t max_tokens) const {
    return impl_->encode(text, max_tokens);
}

int32_t Tokenizer::token_id(std::string_view piece) const {
    auto a = impl_->added_map.find(piece);
    if (a != impl_->added_map.end()) return a->second;
    return impl_->vocab_map.find(piece);
}

size_t Tokenizer::vocab_size() const { return impl_->vsize; }
const TokenizerData& Tokenizer::data() const { return impl_->d; }

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("tokenizer: " + msg); }

std::string str_or(const json& o, const char* k, const std::string& def) {
    auto it = o.find(k);
    return (it == o.end() || it->is_null()) ? def : it->get<std::string>();
}

bool bool_or(const json& o, const char* k, bool def) {
    auto it = o.find(k);
    return (it == o.end() || it->is_null()) ? def : it->get<bool>();
}

}  // namespace

TokenizerData Tokenizer::from_parts(std::vector<std::string> vocab,
                                    const std::vector<std::string>& merges,
                                    std::string_view model_json,
                                    std::string_view normalizer_json,
                                    std::string_view pre_tokenizer_json,
                                    std::string_view added_tokens_json) {
    TokenizerData d;
    const json m = json::parse(model_json);
    if (str_or(m, "type", "BPE") != "BPE") fail("only BPE models are supported");
    if (!str_or(m, "continuing_subword_prefix", "").empty() ||
        !str_or(m, "end_of_word_suffix", "").empty())
        fail("continuing_subword_prefix/end_of_word_suffix not supported");
    if (m.contains("dropout") && !m["dropout"].is_null() && m["dropout"].get<double>() != 0.0)
        fail("BPE dropout not supported");
    d.unk_token = str_or(m, "unk_token", "");
    d.byte_fallback = bool_or(m, "byte_fallback", false);
    d.fuse_unk = bool_or(m, "fuse_unk", false);
    d.ignore_merges = bool_or(m, "ignore_merges", false);

    d.vocab = std::move(vocab);
    PieceIndex ids;
    ids.build(d.vocab);
    auto vid = [&](std::string_view s) {
        int32_t id = ids.find(s);
        if (id < 0) fail("merge token not in vocab: " + std::string(s));
        return id;
    };
    d.merges.reserve(merges.size());
    std::string cat;
    for (const std::string& line : merges) {
        size_t sp = line.find(' ');
        if (sp == std::string::npos || line.find(' ', sp + 1) != std::string::npos)
            fail("bad merge: " + line);
        std::string_view a(line.data(), sp), b(line.data() + sp + 1, line.size() - sp - 1);
        cat.assign(a);
        cat.append(b);
        d.merges.push_back({vid(a), vid(b), vid(cat)});
    }

    const json added = json::parse(added_tokens_json);
    if (!added.is_null()) {
        for (const json& t : added) {
            AddedToken a;
            a.id = t.at("id").get<int32_t>();
            a.content = t.at("content").get<std::string>();
            a.special = bool_or(t, "special", false);
            a.normalized = bool_or(t, "normalized", !a.special);
            a.lstrip = bool_or(t, "lstrip", false);
            a.rstrip = bool_or(t, "rstrip", false);
            a.single_word = bool_or(t, "single_word", false);
            if (a.id < 0) fail("negative added token id");
            d.added.push_back(std::move(a));
        }
    }

    const json jn = json::parse(normalizer_json);
    if (!jn.is_null()) {
        std::string t = jn.at("type").get<std::string>();
        if (t == "NFC") {
            d.normalizer = NormalizerKind::NFC;
        } else if (t == "Replace") {
            const json& p = jn.at("pattern");
            if (!p.contains("String")) fail("Replace normalizer: only String patterns supported");
            d.normalizer = NormalizerKind::Replace;
            d.normalizer_pattern = p["String"].get<std::string>();
            d.normalizer_content = jn.at("content").get<std::string>();
            if (d.normalizer_pattern.empty()) fail("Replace normalizer: empty pattern");
        } else {
            fail("unsupported normalizer: " + t);
        }
    }

    const json jp = json::parse(pre_tokenizer_json);
    if (jp.is_null()) fail("missing pre_tokenizer");
    std::string pt = jp.at("type").get<std::string>();
    if (pt == "Metaspace") {
        d.pre_tokenizer = PreTokenizerKind::Metaspace;
        d.metaspace = str_or(jp, "replacement", "\xE2\x96\x81");
        d.split_on_metaspace = bool_or(jp, "split", true);
        if (jp.contains("prepend_scheme")) {
            std::string ps = jp["prepend_scheme"].get<std::string>();
            d.prepend_scheme = ps == "always" ? PrependScheme::Always
                             : ps == "first"  ? PrependScheme::First
                                              : PrependScheme::Never;
        } else {
            d.prepend_scheme = bool_or(jp, "add_prefix_space", true) ? PrependScheme::Always
                                                                     : PrependScheme::Never;
        }
    } else if (pt == "ByteLevel") {
        d.pre_tokenizer = PreTokenizerKind::ByteLevel;
        d.add_prefix_space = bool_or(jp, "add_prefix_space", true);
        d.use_regex = bool_or(jp, "use_regex", true);
    } else {
        fail("unsupported pre_tokenizer: " + pt);
    }
    return d;
}

TokenizerData Tokenizer::load_hf_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    json j = json::parse(buf);
    buf.clear();
    buf.shrink_to_fit();

    json& m = j.at("model");
    std::vector<std::string> vocab;
    {
        json& jv = m.at("vocab");
        int32_t max_id = -1;
        for (auto it = jv.begin(); it != jv.end(); ++it) max_id = std::max(max_id, it.value().get<int32_t>());
        if (j.contains("added_tokens")) {
            for (const json& t : j["added_tokens"]) max_id = std::max(max_id, t.at("id").get<int32_t>());
        }
        vocab.resize(static_cast<size_t>(max_id + 1));
        for (auto it = jv.begin(); it != jv.end(); ++it) vocab[it.value().get<int32_t>()] = it.key();
        if (j.contains("added_tokens")) {
            for (const json& t : j["added_tokens"]) {
                std::string& slot = vocab[t.at("id").get<int32_t>()];
                if (slot.empty()) slot = t.at("content").get<std::string>();
            }
        }
    }
    std::vector<std::string> merges;
    {
        json& jm = m.at("merges");
        merges.reserve(jm.size());
        for (json& e : jm) {
            if (e.is_array()) {
                merges.push_back(e.at(0).get<std::string>() + " " + e.at(1).get<std::string>());
            } else {
                merges.push_back(e.get<std::string>());
            }
        }
    }
    m.erase("vocab");
    m.erase("merges");
    auto part = [&](const char* k) { return j.contains(k) ? j[k].dump() : std::string("null"); };
    return from_parts(std::move(vocab), merges, m.dump(), part("normalizer"),
                      part("pre_tokenizer"), part("added_tokens"));
}

}  // namespace statim
