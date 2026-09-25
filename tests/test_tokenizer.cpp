// Parity test: statim::Tokenizer vs HuggingFace `tokenizers` golden ids.
// Usage: test_tokenizer [project_root]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "statim/tokenizer.h"

#ifndef STATIM_SOURCE_DIR
#define STATIM_SOURCE_DIR "."
#endif

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Case {
    std::string text;
    std::vector<int32_t> ids, ids48;
};

static std::string snippet(const std::string& s, size_t max = 80) {
    std::string r;
    for (size_t i = 0; i < s.size() && r.size() < max; ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\n') r += "\\n";
        else if (c == '\t') r += "\\t";
        else if (c == '\r') r += "\\r";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); r += b; }
        else r += static_cast<char>(c);
    }
    return r;
}

static std::string ids_around(const std::vector<int32_t>& v, size_t pos) {
    std::string r = "[";
    size_t b = pos > 3 ? pos - 3 : 0, e = std::min(v.size(), pos + 4);
    if (b > 0) r += "... ";
    for (size_t i = b; i < e; ++i) r += (i == pos ? "*" : "") + std::to_string(v[i]) + (i + 1 < e ? " " : "");
    if (e < v.size()) r += " ...";
    return r + "] len=" + std::to_string(v.size());
}

static size_t first_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) return i;
    return n;
}

// Re-serializes TokenizerData into from_parts() inputs (what the GGUF loader
// stores) and times the parsed-data construction path.
static bool check_from_parts(const statim::TokenizerData& d, double& ms) {
    using nlohmann::json;
    std::vector<std::string> merges;
    merges.reserve(d.merges.size());
    for (const auto& m : d.merges) merges.push_back(d.vocab[m.left] + " " + d.vocab[m.right]);
    json model = {{"type", "BPE"}, {"byte_fallback", d.byte_fallback}, {"fuse_unk", d.fuse_unk},
                  {"ignore_merges", d.ignore_merges}};
    model["unk_token"] = d.unk_token.empty() ? json() : json(d.unk_token);
    json norm;
    if (d.normalizer == statim::NormalizerKind::NFC) norm = {{"type", "NFC"}};
    if (d.normalizer == statim::NormalizerKind::Replace)
        norm = {{"type", "Replace"}, {"pattern", {{"String", d.normalizer_pattern}}},
                {"content", d.normalizer_content}};
    json pre;
    if (d.pre_tokenizer == statim::PreTokenizerKind::Metaspace) {
        const char* ps[] = {"always", "first", "never"};
        pre = {{"type", "Metaspace"}, {"replacement", d.metaspace},
               {"prepend_scheme", ps[static_cast<int>(d.prepend_scheme)]}, {"split", d.split_on_metaspace}};
    } else {
        pre = {{"type", "ByteLevel"}, {"add_prefix_space", d.add_prefix_space}, {"use_regex", d.use_regex}};
    }
    json added = json::array();
    for (const auto& a : d.added)
        added.push_back({{"id", a.id}, {"content", a.content}, {"special", a.special},
                         {"normalized", a.normalized}, {"lstrip", a.lstrip}, {"rstrip", a.rstrip},
                         {"single_word", a.single_word}});
    std::vector<std::string> vocab = d.vocab;
    auto t0 = Clock::now();
    statim::TokenizerData r = statim::Tokenizer::from_parts(std::move(vocab), merges, model.dump(),
                                                            norm.dump(), pre.dump(), added.dump());
    statim::Tokenizer tok(std::move(r));
    ms = ms_since(t0);
    const auto& x = tok.data();
    bool same = x.vocab == d.vocab && x.merges.size() == d.merges.size() && x.added.size() == d.added.size();
    for (size_t i = 0; same && i < x.merges.size(); ++i)
        same = x.merges[i].left == d.merges[i].left && x.merges[i].right == d.merges[i].right &&
               x.merges[i].result == d.merges[i].result;
    return same;
}

static bool run(const std::string& name, const std::string& tok_path, const std::string& golden_path) {
    std::printf("== %s\n", name.c_str());
    auto t0 = Clock::now();
    statim::TokenizerData data = statim::Tokenizer::load_hf_json(tok_path);
    double t_json = ms_since(t0);
    auto t1 = Clock::now();
    statim::Tokenizer tok(std::move(data));
    double t_build = ms_since(t1);
    std::printf("load: json+from_parts %.1f ms, Tokenizer build %.1f ms, vocab_size %zu, merges %zu\n",
                t_json, t_build, tok.vocab_size(), tok.data().merges.size());

    double t_parts = 0;
    bool parts_ok = check_from_parts(tok.data(), t_parts);
    std::printf("load: from_parts + Tokenizer build (GGUF path) %.1f ms, round-trip %s\n", t_parts,
                parts_ok ? "identical" : "DIFFERENT");
    if (!parts_ok) return false;

    std::vector<Case> cases;
    std::ifstream f(golden_path);
    if (!f) { std::printf("FAIL: cannot open %s\n", golden_path.c_str()); return false; }
    std::string line;
    size_t bytes = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line);
        cases.push_back({j["text"].get<std::string>(), j["ids"].get<std::vector<int32_t>>(),
                         j["ids48"].get<std::vector<int32_t>>()});
        bytes += cases.back().text.size();
    }

    size_t ok = 0, ok48 = 0, shown = 0;
    auto t2 = Clock::now();
    std::vector<std::vector<int32_t>> got(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) got[i] = tok.encode(cases[i].text);
    double t_cold = ms_since(t2);
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        if (got[i] == c.ids) {
            ++ok;
        } else if (shown++ < 15) {
            size_t p = first_diff(got[i], c.ids);
            std::printf("MISMATCH #%zu at pos %zu: \"%s\"\n  want %s\n  got  %s\n", i, p,
                        snippet(c.text).c_str(), ids_around(c.ids, p).c_str(),
                        ids_around(got[i], p).c_str());
        }
        auto g48 = tok.encode(c.text, 48);
        if (g48 == c.ids48) {
            ++ok48;
        } else if (shown++ < 15) {
            std::printf("MISMATCH48 #%zu at pos %zu: \"%s\"\n", i, first_diff(g48, c.ids48),
                        snippet(c.text).c_str());
        }
    }

    // Warm pass (word cache populated).
    auto t3 = Clock::now();
    size_t ntok = 0;
    for (const Case& c : cases) ntok += tok.encode(c.text).size();
    double t_warm = ms_since(t3);

    // Concurrent pass: 4 threads must reproduce the golden ids.
    std::atomic<size_t> thread_bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) {
        th.emplace_back([&, t] {
            for (size_t i = t; i < cases.size(); i += 4)
                if (tok.encode(cases[i].text) != cases[i].ids) ++thread_bad;
        });
    }
    for (auto& x : th) x.join();

    double mb = bytes / 1e6;
    std::printf("full ids: %zu/%zu, truncated@48: %zu/%zu, threaded mismatches: %zu\n", ok,
                cases.size(), ok48, cases.size(), thread_bad.load());
    std::printf("throughput: cold %.1f MB/s, warm %.1f MB/s (%.2f MB, %zu tokens, %.0f ktok/s)\n",
                mb / (t_cold / 1e3), mb / (t_warm / 1e3), mb, ntok, ntok / t_warm);
    return ok == cases.size() && ok48 == cases.size() && thread_bad == 0 && !cases.empty();
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : STATIM_SOURCE_DIR;
    bool pass = true;
    pass &= run("laya-multilingual (Gemma BPE, byte_fallback)",
                root + "/models/laya-multilingual/tokenizer/tokenizer.json",
                root + "/tests/data/tokenizer_golden_multilingual.jsonl");
    pass &= run("laya (ModernBERT byte-level BPE)", root + "/models/laya/tokenizer/tokenizer.json",
                root + "/tests/data/tokenizer_golden_english.jsonl");
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
