// statim-quantize: re-encode the matmul weights of a Statim GGUF model.
//
//   statim-quantize in-f32.gguf out.gguf q4_k [--embd q8_0]
//
// Only 2-D projection weights are quantized. Norms, biases, the type embedding and tensors
// whose row length does not fit the block size keep their precision (or fall back to q8_0).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

static ggml_type parse_type(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name(static_cast<ggml_type>(t));
        if (n && s == n) return static_cast<ggml_type>(t);
    }
    std::fprintf(stderr, "unknown type '%s' (e.g. f16, q8_0, q5_0, q4_0, q4_K, q5_K, q6_K)\n", s.c_str());
    std::exit(2);
}

static bool is_matmul_weight(const std::string& name, const ggml_tensor* t) {
    if (ggml_n_dims(t) != 2 || t->ne[1] < 64) return false;
    if (name.find("norm") != std::string::npos || name.rfind("type_emb", 0) == 0) return false;
    if (name.rfind("act_head", 0) == 0 || name.rfind("scorer.3", 0) == 0) return false;  // tiny, precision-critical
    return name.size() > 6 && name.compare(name.size() - 6, 6, "weight") == 0;
}

static std::vector<float> to_f32(const ggml_tensor* t) {
    std::vector<float> out(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), t->data, out.size() * 4);
    else ggml_get_type_traits(t->type)->to_float(t->data, out.data(), out.size());
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s in.gguf out.gguf <type> [--embd <type>]\n", argv[0]);
        return 2;
    }
    const ggml_type qt = parse_type(argv[3]);
    ggml_type et = qt;
    for (int i = 4; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--embd")) et = parse_type(argv[i + 1]);

    ggml_cpu_init();
    ggml_context* ctx = nullptr;
    gguf_context* in = gguf_init_from_file(argv[1], {/*no_alloc=*/false, &ctx});
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    gguf_context* out = gguf_init_empty();
    // Preserve source-checkpoint identity (including statim.checkpoint_sha256) unchanged.
    gguf_set_kv(out, in);

    std::vector<std::vector<uint8_t>> buffers;
    ggml_context* octx = ggml_init({ggml_tensor_overhead() * (gguf_get_n_tensors(in) + 8), nullptr, true});
    size_t bytes_in = 0, bytes_out = 0;
    int n_quant = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) {
        const std::string name = gguf_get_tensor_name(in, i);
        ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
        bytes_in += ggml_nbytes(t);
        ggml_type target = t->type;
        if (name == "encoder.embeddings.tok_embeddings.weight") target = et;
        else if (is_matmul_weight(name, t)) target = qt;
        if (target != t->type && t->ne[0] % ggml_blck_size(target) != 0)
            target = t->ne[0] % 32 == 0 ? GGML_TYPE_Q8_0 : t->type;  // row length does not fit the block

        ggml_tensor* o = ggml_new_tensor(octx, target, ggml_n_dims(t), t->ne);
        ggml_set_name(o, name.c_str());
        buffers.emplace_back(ggml_nbytes(o));
        if (target == t->type) {
            std::memcpy(buffers.back().data(), t->data, ggml_nbytes(t));
        } else {
            auto f = to_f32(t);
            if (target == GGML_TYPE_F32) std::memcpy(buffers.back().data(), f.data(), f.size() * 4);
            else ggml_quantize_chunk(target, f.data(), buffers.back().data(), 0, t->ne[1], t->ne[0], nullptr);
            ++n_quant;
        }
        o->data = buffers.back().data();
        gguf_add_tensor(out, o);
        bytes_out += ggml_nbytes(o);
    }
    gguf_set_val_str(out, "statim.quantization", argv[3]);
    if (!gguf_write_to_file(out, argv[2], false)) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    std::printf("%s: %d tensors re-encoded, %.1f MB -> %.1f MB\n", argv[2], n_quant, bytes_in / 1e6, bytes_out / 1e6);
    gguf_free(out);
    gguf_free(in);
    ggml_free(octx);
    ggml_free(ctx);
    return 0;
}
