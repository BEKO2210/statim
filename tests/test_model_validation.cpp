// Model::load must reject malformed checkpoints with an exception (never abort, hang or accept
// a file that crashes later). Each case rewrites one metadata key of the tiny fuzz model.
// Usage: test_model_validation fuzz/data/tiny-metaspace.gguf [case-name-substring]
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "statim/engine.h"
#include "statim/model.h"

using namespace statim;

static int failures = 0;
static const char* only = nullptr;  // run just the cases whose name contains this

// Copy `src`, apply `edit` to its metadata, write it to a temp file and try to load it. Only the
// metadata is re-serialized; the tensor data section is copied byte for byte.
static void expect_rejected(const std::string& src, const char* what, const std::function<void(gguf_context*)>& edit) {
    if (only && !std::strstr(what, only)) return;
    gguf_init_params p{/*no_alloc=*/true, nullptr};
    gguf_context* g = gguf_init_from_file(src.c_str(), p);
    if (!g) throw std::runtime_error("cannot read " + src);
    const size_t data_off = gguf_get_data_offset(g);
    edit(g);
    char path[] = "/tmp/statim-model-validation-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    close(fd);
    const bool written = gguf_write_to_file(g, path, /*only_meta=*/true);
    gguf_free(g);
    if (!written) throw std::runtime_error("cannot write test model");
    {
        std::ifstream in(src, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(data_off));
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << in.rdbuf();
    }
    bool rejected = false;
    try {
        Model::load(path, "cpu");
    } catch (const std::exception& e) {
        rejected = true;
        std::printf("ok   %-44s %s\n", what, e.what());
    }
    if (!rejected) {
        std::printf("FAIL %-44s accepted\n", what);
        ++failures;
    }
    std::remove(path);
}

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s tiny-model.gguf\n", argv[0]);
        return 2;
    }
    const std::string src = argv[1];
    if (argc > 2) only = argv[2];
    // the unmodified model loads and answers
    {
        auto m = Model::load(src, "cpu");
        Engine e(m);
        auto r = e.decide(ojson("the and"), ojson::parse(R"({"q":{"type":"noul","instructions":"ok?"}})"));
        if (!r.contains("answers")) throw std::runtime_error("tiny model did not answer");
    }
    // gguf_get_val_* abort on a type mismatch: every read must check the stored type first
    expect_rejected(src, "u32 hyperparameter stored as string", [](gguf_context* g) {
        gguf_set_val_str(g, "laya.encoder.n_embd", "16");
    });
    expect_rejected(src, "i32 special id stored as u32", [](gguf_context* g) {
        gguf_set_val_u32(g, "tokenizer.statim.cls_id", 2);
    });
    expect_rejected(src, "f32 array stored as string array", [](gguf_context* g) {
        const char* v[] = {"1", "1", "1"};
        gguf_set_arr_str(g, "laya.temperature", v, 3);
    });
    expect_rejected(src, "string array stored as i32 array", [](gguf_context* g) {
        const int32_t v[] = {1, 2, 3};
        gguf_set_arr_data(g, "tokenizer.statim.merges", GGUF_TYPE_INT32, v, 3);
    });
    expect_rejected(src, "tokenizer JSON stored as u32", [](gguf_context* g) {
        gguf_set_val_u32(g, "tokenizer.statim.model", 7);
    });
    // calibration tables used to be parsed only in Engine() (and per request): now at load
    expect_rejected(src, "temperature_by_options not JSON", [](gguf_context* g) {
        gguf_set_val_str(g, "laya.temperature_by_options", "{\"d\"e");
    });
    expect_rejected(src, "temperature_by_options value not a number", [](gguf_context* g) {
        gguf_set_val_str(g, "laya.temperature_by_options", "{\"choice:2\": \"hot\"}");
    });
    expect_rejected(src, "lang_temperatures temperature too short", [](gguf_context* g) {
        gguf_set_val_str(g, "laya.lang_temperatures", "{\"de\": {\"temperature\": [1.0]}}");
    });
    expect_rejected(src, "lang_temperatures entry not an object", [](gguf_context* g) {
        gguf_set_val_str(g, "laya.lang_temperatures", "{\"de\": [1.0, 1.0, 1.0]}");
    });
    // an empty mask token silently disabled scrubbing the mask piece out of caller text
    expect_rejected(src, "empty mask token", [](gguf_context* g) {
        gguf_set_val_str(g, "tokenizer.statim.mask_token", "");
    });
    // general.name is echoed in every response: invalid UTF-8 made every request a 500
    expect_rejected(src, "general.name not UTF-8", [](gguf_context* g) {
        gguf_set_val_str(g, "general.name", "\x82iny");
    });
    // ids index the embedding table in get_rows
    expect_rejected(src, "special id outside the embedding table", [](gguf_context* g) {
        gguf_set_val_i32(g, "tokenizer.statim.sep_id", 1 << 20);
    });
    expect_rejected(src, "negative pad id", [](gguf_context* g) {
        gguf_set_val_i32(g, "tokenizer.statim.pad_id", -1);
    });
    // hyperparameters must match the tensors the graph builder reads
    expect_rejected(src, "n_embd does not match the tensors", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.encoder.n_embd", 32);
    });
    expect_rejected(src, "n_head does not divide n_embd", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.encoder.n_head", 3);
    });
    expect_rejected(src, "n_act does not match the act head", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.head.n_act", 5);
    });
    expect_rejected(src, "n_ff does not match the MLP", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.encoder.n_ff", 8);
    });
    expect_rejected(src, "max_position below the prompt minimum", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.encoder.max_position", 8);
    });
    expect_rejected(src, "local_window zero", [](gguf_context* g) {
        gguf_set_val_u32(g, "laya.encoder.local_window", 0);
    });
    expect_rejected(src, "non-finite norm_eps", [](gguf_context* g) {
        gguf_set_val_f32(g, "laya.encoder.norm_eps", -1.0f);
    });
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all malformed models rejected", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
