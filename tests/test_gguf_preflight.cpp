#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "statim/gguf_preflight.h"
#include "statim/mapped_file.h"
#include "statim/model.h"

namespace fs = std::filesystem;

static int failures = 0;

static void u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

static void u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

static void str(std::vector<uint8_t>& b, const std::string& s) {
    u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

static std::vector<uint8_t> header(int64_t nt = 0, int64_t nk = 0, uint32_t version = 3) {
    std::vector<uint8_t> b{'G', 'G', 'U', 'F'};
    u32(b, version);
    u64(b, static_cast<uint64_t>(nt));
    u64(b, static_cast<uint64_t>(nk));
    return b;
}

static void scalar_kv(std::vector<uint8_t>& b, const std::string& key, uint32_t type, uint64_t value = 0) {
    str(b, key);
    u32(b, type);
    const size_t n = type <= GGUF_TYPE_BOOL ? std::array<size_t, 8>{1, 1, 2, 2, 4, 4, 4, 1}[type] : 8;
    for (size_t i = 0; i < n; ++i) b.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

static void string_kv(std::vector<uint8_t>& b, const std::string& key, const std::string& value) {
    str(b, key); u32(b, GGUF_TYPE_STRING); str(b, value);
}

static void tensor(std::vector<uint8_t>& b, const std::string& name, const std::vector<int64_t>& ne,
                   uint32_t type = GGML_TYPE_F32, uint64_t offset = 0) {
    str(b, name); u32(b, ne.size());
    for (int64_t n : ne) u64(b, static_cast<uint64_t>(n));
    u32(b, type); u64(b, offset);
}

static void pad(std::vector<uint8_t>& b, size_t alignment = 32) {
    b.resize((b.size() + alignment - 1) / alignment * alignment);
}

static std::string write_temp(const std::vector<uint8_t>& bytes) {
    char path[] = "/tmp/statim-gguf-preflight-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    close(fd);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("writing temporary GGUF failed");
    return path;
}

static void expect_rejected(const char* what, std::vector<uint8_t> bytes, const std::string& fragment) {
    const std::string path = write_temp(bytes);
    try {
        statim::gguf_preflight(path);
        std::printf("FAIL %-40s accepted\n", what);
        ++failures;
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find(fragment) == std::string::npos) {
            std::printf("FAIL %-40s %s (wanted '%s')\n", what, e.what(), fragment.c_str());
            ++failures;
        } else {
            std::printf("ok   %-40s %s\n", what, e.what());
        }
    }
    std::remove(path.c_str());
}

static void expect_passes(const char* what, const std::string& path) {
    try {
        statim::gguf_preflight(path);
        std::printf("ok   %-40s accepted\n", what);
    } catch (const std::exception& e) {
        std::printf("FAIL %-40s %s\n", what, e.what());
        ++failures;
    }
}

static std::string writer_file() {
    const std::string path = write_temp({});
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.name", "preflight test");
    ggml_context* c = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    ggml_tensor* t = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
    ggml_set_name(t, "weight");
    float value = 1.0f;
    t->data = &value;
    gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(c);
    if (!ok) throw std::runtime_error("gguf writer failed");
    return path;
}

int main(int argc, char** argv) try {
    const fs::path source = argc > 1 ? argv[1] : ".";

    auto b = header(); b[0] = 'X'; expect_rejected("bad magic", b, "magic is");
    expect_rejected("unsupported version", header(0, 0, 1), "version is 1, limit 2 or 3");
    expect_rejected("negative n_tensors", header(-1, 0), "n_tensors is -1");
    expect_rejected("negative n_kv", header(0, -1), "n_kv is -1");
    expect_rejected("n_tensors exceeds file", header(2, 0), "n_tensors is 2");
    expect_rejected("n_kv exceeds file", header(0, 2), "n_kv is 2");

    b = header(0, 1); u64(b, 100); b.resize(b.size() + 4); expect_rejected("key length exceeds file", b, "key length is 100");
    b = header(0, 2); scalar_kv(b, "x", GGUF_TYPE_UINT8); scalar_kv(b, "x", GGUF_TYPE_UINT8);
    expect_rejected("duplicate key", b, "limit unique keys");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_COUNT); expect_rejected("invalid KV type", b, "GGUF_TYPE_COUNT");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_FLOAT64); b.resize(b.size() + 4);
    expect_rejected("truncated scalar", b, "value length is 8");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_STRING); u64(b, 20); b.resize(b.size() + 4);
    expect_rejected("string length exceeds file", b, "value length is 20");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_ARRAY); u32(b, GGUF_TYPE_COUNT); u64(b, 0);
    expect_rejected("invalid array element type", b, "array element type is");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_ARRAY); u32(b, GGUF_TYPE_ARRAY); u64(b, 0);
    expect_rejected("nested array", b, "nested arrays are not allowed");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_ARRAY); u32(b, GGUF_TYPE_UINT64); u64(b, UINT64_MAX);
    expect_rejected("array multiplication overflow", b, "byte count is");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_ARRAY); u32(b, GGUF_TYPE_UINT64); u64(b, 2); u64(b, 1);
    expect_rejected("array exceeds remaining bytes", b, "byte count is 16");
    b = header(0, 1); str(b, "x"); u32(b, GGUF_TYPE_ARRAY); u32(b, GGUF_TYPE_STRING); u64(b, 1); u64(b, 9); b.push_back('a');
    expect_rejected("string array element exceeds file", b, "array[0] length is 9");
    b = header(0, 1); string_kv(b, "general.alignment", "32"); expect_rejected("alignment wrong type", b, "alignment type");
    b = header(0, 1); scalar_kv(b, "general.alignment", GGUF_TYPE_UINT32, 0); expect_rejected("zero alignment", b, "non-zero power of two");
    b = header(0, 1); scalar_kv(b, "general.alignment", GGUF_TYPE_UINT32, 3); expect_rejected("non-power-of-two alignment", b, "non-zero power of two");

    b = header(1, 0); tensor(b, std::string(GGML_MAX_NAME, 'n'), {1}); pad(b); b.resize(b.size() + 4);
    expect_rejected("long tensor name", b, "name length is 64");
    b = header(2, 0); tensor(b, "x", {1}, GGML_TYPE_F32, 0); tensor(b, "x", {1}, GGML_TYPE_F32, 32); pad(b); b.resize(b.size() + 36);
    expect_rejected("duplicate tensor name", b, "unique tensor names");
    b = header(1, 0); tensor(b, "x", {}); pad(b); expect_rejected("zero tensor dimensions", b, "n_dims is 0");
    b = header(1, 0); str(b, "x"); u32(b, GGML_MAX_DIMS + 1); b.resize(b.size() + 32);
    expect_rejected("too many tensor dimensions", b, "GGML_MAX_DIMS");
    b = header(1, 0); tensor(b, "x", {-1}); pad(b); expect_rejected("negative tensor extent", b, "ne[0] is -1");
    b = header(1, 0); tensor(b, "x", {INT64_MAX, 2}); pad(b); expect_rejected("element count overflow (U1)", b, "element count");
    b = header(1, 0); tensor(b, "x", {1}, GGML_TYPE_COUNT); pad(b); expect_rejected("out-of-range tensor type (U2)", b, "type is");
    // A running product above INT64_MAX followed by a zero extent: the final count is 0, but ggml's
    // guard multiplies the prefix first, so the check must fire inside the loop (U1).
    b = header(1, 0); tensor(b, "x", {int64_t{1} << 62, 2, 0}); pad(b); expect_rejected("element count above INT64_MAX before a zero extent (U1)", b, "limit INT64_MAX");
    // ggml_nbytes multiplies type_size by ne[0] before dividing by the block size
    b = header(1, 0); tensor(b, "x", {int64_t{1} << 62}); pad(b); expect_rejected("ne[0] * type size overflow", b, "ne[0] * type size");
    b = header(1, 0); tensor(b, std::string("a\0b", 3), {1}); pad(b); b.resize(b.size() + 32);
    expect_rejected("tensor name with an embedded NUL", b, "embedded NUL");
    b = header(1, 0); tensor(b, "x", {1}, 4); pad(b); expect_rejected("zero block-size type", b, "block size is 0");
    b = header(1, 0); tensor(b, "x", {1}, GGML_TYPE_Q8_0); pad(b); expect_rejected("row not divisible by block", b, "multiple of block size");
    b = header(1, 0); tensor(b, "x", {1, INT64_MAX}, GGML_TYPE_F64); pad(b); expect_rejected("tensor byte-size overflow", b, "byte size is");
    b = header(1, 0); tensor(b, "x", {1}, GGML_TYPE_F32, 1); pad(b); b.resize(b.size() + 5);
    expect_rejected("misaligned tensor offset", b, "aligned to 32");
    b = header(2, 0); tensor(b, "a", {1}, GGML_TYPE_F32, 0); tensor(b, "b", {1}, GGML_TYPE_F32, 64); pad(b); b.resize(b.size() + 68);
    expect_rejected("non-contiguous tensor layout", b, "limit expected 32");
    b = header(1, 0); tensor(b, "x", {1}); expect_rejected("data start beyond file", b, "data section start");
    b = header(1, 0); tensor(b, "x", {2}); pad(b); b.resize(b.size() + 4);
    expect_rejected("tensor exceeds data section", b, "data end is 8");
    b = header(1, 0); tensor(b, "x", {1}); pad(b); b.resize(b.size() + 4);
    expect_rejected("last tensor's padding beyond the file", b, "padded data section");

    b = header(1, 1); scalar_kv(b, "general.alignment", GGUF_TYPE_UINT32, 64);
    tensor(b, std::string(GGML_MAX_NAME - 1, 'n'), {16}); pad(b, 64); b.resize(b.size() + 64);
    const std::string aligned = write_temp(b); expect_passes("alignment 64 and a 63-byte name", aligned); std::remove(aligned.c_str());
    b = header(0, 0, 2);
    const std::string v2 = write_temp(b); expect_passes("valid empty GGUF v2", v2); std::remove(v2.c_str());
    const std::string written = writer_file(); expect_passes("ggml writer output", written); std::remove(written.c_str());

    // Open once (READINESS #45): the preflight, ggml and the tensors read one mapping of one open
    // file, so a file renamed over the path after the open cannot slip past the preflight.
    {
        auto expect_map_rejected = [&](const char* what, const std::string& path, const char* needle) {
            try {
                statim::MappedFile m(path);
                ++failures;
                std::printf("FAIL %-44s mapped\n", what);
            } catch (const std::runtime_error& e) {
                const bool ok = std::strstr(e.what(), needle) != nullptr;
                failures += !ok;
                std::printf("%s %-44s %s\n", ok ? "ok  " : "FAIL", what, e.what());
            }
        };
        const fs::path dir = fs::temp_directory_path() / ("statim-mapped-" + std::to_string(getpid()));
        fs::create_directories(dir);
        expect_map_rejected("a directory is not a model file", dir.string(), "not a regular file");
        const std::string fifo = (dir / "fifo").string();
        if (mkfifo(fifo.c_str(), 0600) == 0)  // must not block: no writer ever opens it
            expect_map_rejected("a FIFO is refused without blocking", fifo, "not a regular file");
        const std::string empty = (dir / "empty.gguf").string();
        std::ofstream(empty).close();
        expect_map_rejected("an empty file", empty, "is empty");

        const std::string original = writer_file();
        statim::MappedFile mapped(original);
        const std::vector<char> before(static_cast<const char*>(mapped.data()),
                                       static_cast<const char*>(mapped.data()) + mapped.size());
        std::vector<uint8_t> other = header(0, 0);  // a different, smaller valid file
        const std::string swap = write_temp(other);
        fs::rename(swap, original);  // the path now names another file
        const bool same = mapped.size() == before.size() &&
                          std::memcmp(mapped.data(), before.data(), before.size()) == 0;
        bool checked = true;
        try { statim::gguf_preflight(mapped.data(), mapped.size()); } catch (const std::exception&) { checked = false; }
        const bool ok = same && checked && fs::file_size(original) != mapped.size();
        failures += !ok;
        std::printf("%s %-44s mapping unchanged after the path was swapped\n", ok ? "ok  " : "FAIL",
                    "rename over a mapped model");
        std::remove(original.c_str());
        fs::remove_all(dir);
    }

    const fs::path regressions = source / "fuzz/regressions/gguf";
    for (const fs::directory_entry& entry : fs::directory_iterator(regressions)) {
        const std::string name = entry.path().filename().string();
        if (name == "ggml-nelements-overflow-in-guard.gguf") {
            try { statim::gguf_preflight(entry.path()); ++failures; std::printf("FAIL regression %-29s accepted by preflight\n", name.c_str()); }
            catch (const std::runtime_error& e) { std::printf("ok   regression %-29s %s\n", name.c_str(), e.what()); }
        } else {
            expect_passes(("structural regression " + name).c_str(), entry.path());
            try { (void)statim::Model::load(entry.path(), "cpu"); std::printf("FAIL regression %-29s loaded\n", name.c_str()); ++failures; }
            catch (const std::exception& e) { std::printf("ok   later validation %-29s %s\n", name.c_str(), e.what()); }
        }
    }

    const std::array model_paths{source / "models/laya-multilingual-f32.gguf", source / "models/laya-english-f32.gguf"};
    for (const fs::path& model : model_paths) {
        if (!fs::exists(model)) { std::printf("model files absent; model preflight skipped\n"); return failures ? 1 : 77; }
        expect_passes(model.filename().c_str(), model);
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all GGUF preflight checks passed", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
