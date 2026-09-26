#include "statim/model.h"

#include "kernels.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-impl.h"  // ggml_graph_view for STATIM_PROFILE

namespace statim {

// Implemented in tokenizer_gguf.cpp: rebuilds TokenizerData from GGUF metadata.
TokenizerData tokenizer_data_from_gguf(const gguf_context* g);

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("statim: " + msg); }

int64_t key(const gguf_context* g, const char* k, bool required = true) {
    int64_t id = gguf_find_key(g, k);
    if (id < 0 && required) fail(std::string("model file is missing metadata key '") + k + "'");
    return id;
}
uint32_t get_u32(const gguf_context* g, const char* k) { return gguf_get_val_u32(g, key(g, k)); }
float get_f32(const gguf_context* g, const char* k) { return gguf_get_val_f32(g, key(g, k)); }
std::string get_str(const gguf_context* g, const char* k, const std::string& def = "") {
    int64_t id = key(g, k, false);
    return id < 0 ? def : std::string(gguf_get_val_str(g, id));
}

struct EncLayer {
    ggml_tensor *attn_norm = nullptr, *wqkv, *wo, *mlp_norm, *wi, *wo_mlp;
};
struct HeadLayer {
    ggml_tensor *norm1_w, *norm1_b, *in_w, *in_b, *out_w, *out_b;
    ggml_tensor *norm2_w, *norm2_b, *l1_w, *l1_b, *l2_w, *l2_b;
};

}  // namespace

struct Model::Impl {
    HParams hp;
    std::unique_ptr<Tokenizer> tok;
    gguf_context* gguf = nullptr;
    ggml_context* ctx_w = nullptr;
    void* map = nullptr;
    size_t map_size = 0;
    size_t weight_bytes = 0;
    ggml_backend_buffer_t repack_buf = nullptr;  // weights re-laid-out for the CPU's int GEMM kernels
    int n_repacked = 0;

    ggml_tensor *tok_embd, *embd_norm, *final_norm, *type_emb;
    std::vector<EncLayer> enc;
    std::vector<HeadLayer> head;
    ggml_tensor *sc_norm_w, *sc_norm_b, *sc1_w, *sc1_b, *sc2_w, *sc2_b;
    // act head runs on the host (tiny), weights converted to f32 once
    std::vector<float> act0_w, act0_b, act2_w, act2_b;

    ggml_tensor* t(const std::string& name, bool required = true) {
        ggml_tensor* x = ggml_get_tensor(ctx_w, name.c_str());
        if (!x && required) fail("model file is missing tensor '" + name + "'");
        return x;
    }

    ~Impl() {
        if (repack_buf) ggml_backend_buffer_free(repack_buf);
        if (ctx_w) ggml_free(ctx_w);
        if (gguf) gguf_free(gguf);
        if (map) munmap(map, map_size);
    }
};

static std::vector<float> to_f32(const ggml_tensor* x) {
    std::vector<float> out(ggml_nelements(x));
    if (x->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), x->data, out.size() * sizeof(float));
    } else if (x->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(x->data), out.data(), out.size());
    } else {
        const auto* tr = ggml_get_type_traits(x->type);
        if (!tr->to_float) fail("cannot dequantize tensor " + std::string(x->name));
        tr->to_float(x->data, out.data(), out.size());
    }
    return out;
}

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
const HParams& Model::hparams() const { return impl_->hp; }
const Tokenizer& Model::tokenizer() const { return *impl_->tok; }
size_t Model::weight_bytes() const { return impl_->weight_bytes; }

// Quantized projection weights are copied into ggml's CPU "repack" buffer when the CPU has
// a blocked int8 GEMM for their type (AVX2: q4_0/q4_K 8x8; ARM dotprod/i8mm: q4_0/q8_0).
// Everything else stays zero-copy in the mmap.
static bool repack_eligible(const ggml_tensor* t) {
    if (ggml_n_dims(t) != 2) return false;
    if (ggml_cpu_has_avx2())
        return (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_K) && t->ne[1] % 8 == 0;
    if (ggml_cpu_has_neon() && (ggml_cpu_has_matmul_int8() || ggml_cpu_has_dotprod()))
        return (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q8_0) && t->ne[1] % 4 == 0;
    return false;
}

static void repack_weights(Model::Impl& M) {
    if (std::getenv("STATIM_NO_REPACK")) return;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return;
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    if (!get_extra) return;
    ggml_backend_buffer_type_t buft = nullptr;
    for (ggml_backend_buffer_type_t* b = get_extra(dev); b && *b; ++b)
        if (std::strstr(ggml_backend_buft_name(*b), "REPACK")) buft = *b;
    if (!buft) return;

    std::vector<ggml_tensor*> todo;
    size_t total = 0;
    const size_t align = ggml_backend_buft_get_alignment(buft);
    for (ggml_tensor* t = ggml_get_first_tensor(M.ctx_w); t; t = ggml_get_next_tensor(M.ctx_w, t)) {
        if (std::string(ggml_get_name(t)) == "encoder.embeddings.tok_embeddings.weight") continue;  // get_rows only
        if (!repack_eligible(t)) continue;
        todo.push_back(t);
        total += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    }
    if (todo.empty()) return;
    M.repack_buf = ggml_backend_buft_alloc_buffer(buft, total);
    if (!M.repack_buf) return;
    ggml_tallocr talloc = ggml_tallocr_new(M.repack_buf);
    for (ggml_tensor* t : todo) {
        const void* src = t->data;
        t->data = nullptr;
        ggml_tallocr_alloc(&talloc, t);
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        ++M.n_repacked;
    }
}

std::shared_ptr<Model> Model::load(const std::string& path) {
    std::shared_ptr<Model> m(new Model());
    Impl& M = *m->impl_;

    gguf_init_params gp{/*no_alloc=*/true, &M.ctx_w};
    M.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!M.gguf) fail("cannot read GGUF model '" + path + "'");
    if (get_str(M.gguf, "statim.format") != "statim-decision-v1")
        fail("'" + path + "' is not a Statim decision model (convert it with tools/convert_laya.py)");

    // Map the file read-only; tensors point straight into the page cache (zero-copy load,
    // shared between processes serving the same model).
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail("cannot open '" + path + "'");
    struct stat st{};
    fstat(fd, &st);
    M.map_size = static_cast<size_t>(st.st_size);
    M.map = mmap(nullptr, M.map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (M.map == MAP_FAILED) {
        M.map = nullptr;
        fail("mmap failed for '" + path + "'");
    }
    const size_t data_off = gguf_get_data_offset(M.gguf);
    for (int64_t i = 0; i < gguf_get_n_tensors(M.gguf); ++i) {
        ggml_tensor* x = ggml_get_tensor(M.ctx_w, gguf_get_tensor_name(M.gguf, i));
        size_t off = data_off + gguf_get_tensor_offset(M.gguf, i);
        if (off + ggml_nbytes(x) > M.map_size) fail("truncated model file '" + path + "'");
        x->data = static_cast<char*>(M.map) + off;
        M.weight_bytes += ggml_nbytes(x);
    }

    repack_weights(M);

    HParams& h = M.hp;
    const gguf_context* g = M.gguf;
    h.name = get_str(g, "general.name");
    h.n_embd = get_u32(g, "laya.encoder.n_embd");
    h.n_layer = get_u32(g, "laya.encoder.n_layer");
    h.n_head = get_u32(g, "laya.encoder.n_head");
    h.n_ff = get_u32(g, "laya.encoder.n_ff");
    h.local_window = get_u32(g, "laya.encoder.local_window");
    h.rope_theta_global = get_f32(g, "laya.encoder.rope_theta_global");
    h.rope_theta_local = get_f32(g, "laya.encoder.rope_theta_local");
    h.norm_eps = get_f32(g, "laya.encoder.norm_eps");
    h.max_position = get_u32(g, "laya.encoder.max_position");
    {
        int64_t id = key(g, "laya.encoder.layer_is_global");
        const auto* v = static_cast<const int8_t*>(gguf_get_arr_data(g, id));
        for (size_t i = 0; i < gguf_get_arr_n(g, id); ++i) h.layer_is_global.push_back(v[i] != 0);
        if ((int)h.layer_is_global.size() != h.n_layer) fail("layer_is_global has wrong length");
    }
    h.head_n_layer = get_u32(g, "laya.head.n_layer");
    h.head_n_head = get_u32(g, "laya.head.n_head");
    h.head_n_ff = get_u32(g, "laya.head.n_ff");
    h.n_act = get_u32(g, "laya.head.n_act");
    h.max_len = get_u32(g, "laya.max_len");
    h.head_max_len = get_u32(g, "laya.head_max_len");
    {
        int64_t id = key(g, "laya.temperature");
        const auto* v = static_cast<const float*>(gguf_get_arr_data(g, id));
        h.temperature.assign(v, v + gguf_get_arr_n(g, id));
    }
    h.temperature_by_options_json = get_str(g, "laya.temperature_by_options", "{}");
    h.lang_temperatures_json = get_str(g, "laya.lang_temperatures", "{}");
    h.cls_id = gguf_get_val_i32(g, key(g, "tokenizer.statim.cls_id"));
    h.sep_id = gguf_get_val_i32(g, key(g, "tokenizer.statim.sep_id"));
    h.mask_id = gguf_get_val_i32(g, key(g, "tokenizer.statim.mask_id"));
    h.pad_id = gguf_get_val_i32(g, key(g, "tokenizer.statim.pad_id"));
    h.mask_token = get_str(g, "tokenizer.statim.mask_token", "<mask>");

    M.tok = std::make_unique<Tokenizer>(tokenizer_data_from_gguf(g));

    M.tok_embd = M.t("encoder.embeddings.tok_embeddings.weight");
    M.embd_norm = M.t("encoder.embeddings.norm.weight");
    M.final_norm = M.t("encoder.final_norm.weight");
    M.type_emb = M.t("type_emb.weight");
    h.weight_type = ggml_type_name(M.t("encoder.layers.0.attn.Wqkv.weight")->type);
    for (int l = 0; l < h.n_layer; ++l) {
        std::string p = "encoder.layers." + std::to_string(l) + ".";
        EncLayer L;
        L.attn_norm = M.t(p + "attn_norm.weight", l != 0);  // layer 0 has an identity attn_norm
        L.wqkv = M.t(p + "attn.Wqkv.weight");
        L.wo = M.t(p + "attn.Wo.weight");
        L.mlp_norm = M.t(p + "mlp_norm.weight");
        L.wi = M.t(p + "mlp.Wi.weight");
        L.wo_mlp = M.t(p + "mlp.Wo.weight");
        M.enc.push_back(L);
    }
    for (int l = 0; l < h.head_n_layer; ++l) {
        std::string p = "head.layers." + std::to_string(l) + ".";
        HeadLayer L;
        L.norm1_w = M.t(p + "norm1.weight");
        L.norm1_b = M.t(p + "norm1.bias");
        L.in_w = M.t(p + "self_attn.in_proj_weight");
        L.in_b = M.t(p + "self_attn.in_proj_bias");
        L.out_w = M.t(p + "self_attn.out_proj.weight");
        L.out_b = M.t(p + "self_attn.out_proj.bias");
        L.norm2_w = M.t(p + "norm2.weight");
        L.norm2_b = M.t(p + "norm2.bias");
        L.l1_w = M.t(p + "linear1.weight");
        L.l1_b = M.t(p + "linear1.bias");
        L.l2_w = M.t(p + "linear2.weight");
        L.l2_b = M.t(p + "linear2.bias");
        M.head.push_back(L);
    }
    M.sc_norm_w = M.t("scorer.0.weight");
    M.sc_norm_b = M.t("scorer.0.bias");
    M.sc1_w = M.t("scorer.1.weight");
    M.sc1_b = M.t("scorer.1.bias");
    M.sc2_w = M.t("scorer.3.weight");
    M.sc2_b = M.t("scorer.3.bias");
    M.act0_w = to_f32(M.t("act_head.0.weight"));
    M.act0_b = to_f32(M.t("act_head.0.bias"));
    M.act2_w = to_f32(M.t("act_head.2.weight"));
    M.act2_b = to_f32(M.t("act_head.2.bias"));
    return m;
}

// ---------------------------------------------------------------------------------------------
// Graph construction

namespace {

struct GraphIO {
    ggml_tensor *ids, *pos, *mask_global, *mask_local, *qtype, *gather_rows, *pool_rows, *qrows = nullptr, *mask_q = nullptr;
    ggml_tensor *logits, *pooled, *hidden;
};

ggml_tensor* layer_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    return b ? ggml_add(c, x, b) : x;
}

ggml_tensor* linear(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b = nullptr) {
    x = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, x, b) : x;
}

void geglu_op(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* src = dst->src[0];
    const long ff = dst->ne[0], rows = ggml_nrows(dst);
    const long per = (rows + nth - 1) / nth;
    geglu_rows(static_cast<float*>(dst->data), static_cast<const float*>(src->data), rows, ff, ith * per, (ith + 1) * per);
}

// q, k, v: [hd, n_head, L, B] -> [n_embd, L, B]
ggml_tensor* attention(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
                       int hd, int n_head, int L, int B, bool flash) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    q = ggml_permute(c, q, 0, 2, 1, 3);  // [hd, L, nh, B]
    k = ggml_permute(c, k, 0, 2, 1, 3);
    if (flash) {
        v = ggml_permute(c, v, 0, 2, 1, 3);  // [hd, L, nh, B]
        ggml_tensor* o = ggml_flash_attn_ext(c, q, k, v, mask, scale, 0.0f, 0.0f);  // [hd, nh, L, B]
        return ggml_reshape_3d(c, o, hd * n_head, L, B);
    }
    ggml_tensor* kq = ggml_mul_mat(c, k, q);  // [L_k, L_q, nh, B]
    kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));  // [L, hd, nh, B]
    ggml_tensor* o = ggml_mul_mat(c, vt, kq);                        // [hd, L_q, nh, B]
    o = ggml_permute(c, o, 0, 2, 1, 3);                              // [hd, nh, L, B]
    return ggml_cont_3d(c, o, hd * n_head, L, B);
}

}  // namespace

struct Runner::Impl {
    RunOptions opts;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<uint8_t> meta_buf;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (backend) ggml_backend_free(backend);
    }
};

Runner::Runner(std::shared_ptr<Model> model, RunOptions opts) : model_(std::move(model)), impl_(std::make_unique<Impl>()) {
    impl_->opts = opts;
    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) fail("cannot initialise CPU backend");
    int nt = opts.n_threads > 0 ? opts.n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ggml_backend_cpu_set_n_threads(impl_->backend, nt);
    impl_->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    impl_->meta_buf.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
}

Runner::~Runner() = default;

static void build_masks(const std::vector<const std::vector<int32_t>*>& seqs, int L, int window,
                        std::vector<ggml_fp16_t>& global, std::vector<ggml_fp16_t>& local) {
    const int B = static_cast<int>(seqs.size());
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    global.assign(static_cast<size_t>(L) * L * B, ninf);
    local.assign(static_cast<size_t>(L) * L * B, ninf);
    const int half = window / 2;
    for (int b = 0; b < B; ++b) {
        const int n = static_cast<int>(seqs[b]->size());
        for (int i = 0; i < L; ++i) {
            ggml_fp16_t* g = &global[(static_cast<size_t>(b) * L + i) * L];
            ggml_fp16_t* l = &local[(static_cast<size_t>(b) * L + i) * L];
            for (int j = 0; j < n; ++j) {
                g[j] = zero;
                if (std::abs(i - j) <= half) l[j] = zero;
            }
            // padded query rows would otherwise be all -inf (NaN); let them attend to token 0
            if (i >= n) g[0] = l[0] = zero;
        }
    }
}

std::vector<ItemResult> Runner::run(const std::vector<Item>& items) {
    if (items.empty()) return {};
    const Model::Impl& M = *model_->impl();
    const HParams& h = M.hp;
    const int B = static_cast<int>(items.size());
    int L = 0;
    std::vector<const std::vector<int32_t>*> seqs;
    for (const Item& it : items) {
        if (it.ids.empty()) fail("empty item");
        if (static_cast<int>(it.ids.size()) > h.max_position) fail("sequence longer than max_position");
        L = std::max(L, static_cast<int>(it.ids.size()));
        seqs.push_back(&it.ids);
    }
    const bool flash = impl_->opts.flash_attn;
    // flash attention needs the mask row count padded to GGML_KQ_MASK_PAD
    const int Lpad = L;

    ggml_init_params ip{impl_->meta_buf.size(), impl_->meta_buf.data(), /*no_alloc=*/true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(c, 16384, false);

    const int d = h.n_embd, hd = d / h.n_head;
    GraphIO io{};
    io.ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, static_cast<int64_t>(L) * B);
    ggml_set_input(io.ids);
    io.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, L);
    ggml_set_input(io.pos);
    io.mask_global = ggml_new_tensor_4d(c, GGML_TYPE_F16, L, Lpad, 1, B);
    ggml_set_input(io.mask_global);
    io.mask_local = ggml_new_tensor_4d(c, GGML_TYPE_F16, L, Lpad, 1, B);
    ggml_set_input(io.mask_local);
    io.qtype = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    ggml_set_input(io.qtype);
    size_t n_markers = 0;
    for (const Item& it : items) n_markers += it.markers.size();
    io.gather_rows = ggml_new_tensor_1d(c, GGML_TYPE_I32, static_cast<int64_t>(n_markers));
    ggml_set_input(io.gather_rows);
    io.pool_rows = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    ggml_set_input(io.pool_rows);

    // ---- ModernBERT encoder
    ggml_tensor* x = ggml_get_rows(c, M.tok_embd, io.ids);  // [d, L*B]
    x = ggml_reshape_3d(c, x, d, L, B);
    x = layer_norm(c, x, M.embd_norm, nullptr, h.norm_eps);
    for (int l = 0; l < h.n_layer; ++l) {
        const EncLayer& Ly = M.enc[l];
        const bool glob = h.layer_is_global[l];
        ggml_tensor* a = Ly.attn_norm ? layer_norm(c, x, Ly.attn_norm, nullptr, h.norm_eps) : x;
        ggml_tensor* qkv = linear(c, a, Ly.wqkv);  // [3d, L, B]
        const size_t es = ggml_element_size(qkv);
        auto view = [&](int part) {
            return ggml_view_4d(c, qkv, hd, h.n_head, L, B, hd * es, qkv->nb[1], qkv->nb[2], part * d * es);
        };
        ggml_tensor* q = ggml_cont(c, view(0));
        ggml_tensor* k = ggml_cont(c, view(1));
        ggml_tensor* v = ggml_cont(c, view(2));
        const float theta = glob ? h.rope_theta_global : h.rope_theta_local;
        q = ggml_rope_ext(c, q, io.pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, h.max_position, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, io.pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, h.max_position, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor* o = attention(c, q, k, v, glob ? io.mask_global : io.mask_local, hd, h.n_head, L, B, flash);
        x = ggml_add(c, x, linear(c, o, Ly.wo));

        ggml_tensor* m = layer_norm(c, x, Ly.mlp_norm, nullptr, h.norm_eps);
        m = linear(c, m, Ly.wi);  // [2*ff, L, B]; first half = input, second half = gate
        ggml_tensor* args[] = {m};
        m = ggml_custom_4d(c, GGML_TYPE_F32, h.n_ff, L, B, 1, args, 1, geglu_op, GGML_N_TASKS_MAX, nullptr);
        x = ggml_add(c, x, linear(c, m, Ly.wo_mlp));
    }
    x = layer_norm(c, x, M.final_norm, nullptr, h.norm_eps);
    io.hidden = x;

    // ---- decision head: type embedding + pre-norm transformer (ReLU FFN, biased projections)
    ggml_tensor* te = ggml_get_rows(c, M.type_emb, io.qtype);  // [d, B]
    x = ggml_add(c, x, ggml_reshape_3d(c, te, d, 1, B));
    const int hhd = d / h.head_n_head;
    // Only [CLS] and the [MASK] rows of the last head layer are read, so that layer computes its
    // queries, output projection and FFN for those rows alone (keys/values still span all tokens).
    int Qn = 1;
    for (const Item& it : items) Qn = std::max(Qn, 1 + static_cast<int>(it.markers.size()));
    io.qrows = ggml_new_tensor_1d(c, GGML_TYPE_I32, static_cast<int64_t>(Qn) * B);
    ggml_set_input(io.qrows);
    io.mask_q = ggml_new_tensor_4d(c, GGML_TYPE_F16, L, Qn, 1, B);
    ggml_set_input(io.mask_q);
    const int n_head_layers = static_cast<int>(M.head.size());
    int rows_per_seq = L;  // rows of x per sequence: L until the pruned layer, Qn after it
    for (int li = 0; li < n_head_layers; ++li) {
        const HeadLayer& Ly = M.head[li];
        ggml_tensor* a = layer_norm(c, x, Ly.norm1_w, Ly.norm1_b, 1e-5f);
        if (li + 1 < n_head_layers) {
            ggml_tensor* qkv = linear(c, a, Ly.in_w, Ly.in_b);
            const size_t es = ggml_element_size(qkv);
            auto view = [&](int part) {
                return ggml_view_4d(c, qkv, hhd, h.head_n_head, L, B, hhd * es, qkv->nb[1], qkv->nb[2], part * d * es);
            };
            ggml_tensor* o = attention(c, ggml_cont(c, view(0)), ggml_cont(c, view(1)), ggml_cont(c, view(2)),
                                       io.mask_global, hhd, h.head_n_head, L, B, flash);
            x = ggml_add(c, x, linear(c, o, Ly.out_w, Ly.out_b));
        } else {
            ggml_tensor* wq = ggml_view_2d(c, Ly.in_w, d, d, Ly.in_w->nb[1], 0);
            ggml_tensor* wkv = ggml_view_2d(c, Ly.in_w, d, 2 * d, Ly.in_w->nb[1], d * Ly.in_w->nb[1]);
            ggml_tensor* bq = ggml_view_1d(c, Ly.in_b, d, 0);
            ggml_tensor* bkv = ggml_view_1d(c, Ly.in_b, 2 * d, d * ggml_element_size(Ly.in_b));
            ggml_tensor* kv = linear(c, a, wkv, bkv);  // [2d, L, B]
            const size_t es = ggml_element_size(kv);
            ggml_tensor* k = ggml_cont(c, ggml_view_4d(c, kv, hhd, h.head_n_head, L, B, hhd * es, kv->nb[1], kv->nb[2], 0));
            ggml_tensor* v = ggml_cont(c, ggml_view_4d(c, kv, hhd, h.head_n_head, L, B, hhd * es, kv->nb[1], kv->nb[2], d * es));
            ggml_tensor* aq = ggml_get_rows(c, ggml_reshape_2d(c, a, d, static_cast<int64_t>(L) * B), io.qrows);
            ggml_tensor* q = ggml_reshape_4d(c, linear(c, aq, wq, bq), hhd, h.head_n_head, Qn, B);
            // attention() expects equal query/key lengths; call the kernels directly
            const float scale = 1.0f / std::sqrt(static_cast<float>(hhd));
            ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3), *kp = ggml_permute(c, k, 0, 2, 1, 3);
            ggml_tensor* o;
            if (flash) {
                o = ggml_flash_attn_ext(c, qp, kp, ggml_permute(c, v, 0, 2, 1, 3), io.mask_q, scale, 0.0f, 0.0f);
                o = ggml_reshape_3d(c, o, d, Qn, B);
            } else {
                ggml_tensor* kq = ggml_soft_max_ext(c, ggml_mul_mat(c, kp, qp), io.mask_q, scale, 0.0f);
                ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
                o = ggml_cont_3d(c, ggml_permute(c, ggml_mul_mat(c, vt, kq), 0, 2, 1, 3), d, Qn, B);
            }
            ggml_tensor* xq = ggml_get_rows(c, ggml_reshape_2d(c, x, d, static_cast<int64_t>(L) * B), io.qrows);
            x = ggml_add(c, ggml_reshape_3d(c, xq, d, Qn, B), linear(c, o, Ly.out_w, Ly.out_b));
            rows_per_seq = Qn;
        }
        ggml_tensor* f = layer_norm(c, x, Ly.norm2_w, Ly.norm2_b, 1e-5f);
        f = ggml_relu(c, linear(c, f, Ly.l1_w, Ly.l1_b));
        x = ggml_add(c, x, linear(c, f, Ly.l2_w, Ly.l2_b));
    }

    // ---- scorer on the marker rows, pooled [CLS] rows for the act head
    ggml_tensor* flat = ggml_reshape_2d(c, x, d, static_cast<int64_t>(rows_per_seq) * B);
    ggml_tensor* mk = ggml_get_rows(c, flat, io.gather_rows);  // [d, n_markers]
    mk = layer_norm(c, mk, M.sc_norm_w, M.sc_norm_b, 1e-5f);
    mk = ggml_gelu_erf(c, linear(c, mk, M.sc1_w, M.sc1_b));
    io.logits = linear(c, mk, M.sc2_w, M.sc2_b);  // [1, n_markers]
    io.pooled = ggml_get_rows(c, flat, io.pool_rows);  // [d, B]
    ggml_set_output(io.logits);
    ggml_set_output(io.pooled);
    ggml_build_forward_expand(gf, io.logits);
    ggml_build_forward_expand(gf, io.pooled);

    if (!ggml_gallocr_alloc_graph(impl_->galloc, gf)) {
        ggml_free(c);
        fail("out of memory while allocating the compute graph");
    }

    // ---- inputs
    std::vector<int32_t> ids(static_cast<size_t>(L) * B, h.pad_id);
    for (int b = 0; b < B; ++b) std::copy(items[b].ids.begin(), items[b].ids.end(), ids.begin() + static_cast<size_t>(b) * L);
    ggml_backend_tensor_set(io.ids, ids.data(), 0, ggml_nbytes(io.ids));
    std::vector<int32_t> pos(L);
    for (int i = 0; i < L; ++i) pos[i] = i;
    ggml_backend_tensor_set(io.pos, pos.data(), 0, ggml_nbytes(io.pos));
    std::vector<ggml_fp16_t> mg, ml;
    build_masks(seqs, L, h.local_window, mg, ml);
    ggml_backend_tensor_set(io.mask_global, mg.data(), 0, ggml_nbytes(io.mask_global));
    ggml_backend_tensor_set(io.mask_local, ml.data(), 0, ggml_nbytes(io.mask_local));
    const bool pruned = !M.head.empty();
    std::vector<int32_t> qt(B), rows, pool(B), qrows(static_cast<size_t>(Qn) * B, 0);
    std::vector<ggml_fp16_t> mq(static_cast<size_t>(L) * Qn * B, ggml_fp32_to_fp16(-INFINITY));
    for (int b = 0; b < B; ++b) {
        qt[b] = items[b].qtype;
        const int n = static_cast<int>(items[b].ids.size());
        // compact query rows of the last head layer: [CLS], markers..., padding repeats [CLS]
        for (int r = 0; r < Qn; ++r) {
            const int src = r == 0 || r > static_cast<int>(items[b].markers.size()) ? 0 : items[b].markers[r - 1];
            qrows[static_cast<size_t>(b) * Qn + r] = b * L + src;
            for (int j = 0; j < n; ++j) mq[(static_cast<size_t>(b) * Qn + r) * L + j] = ggml_fp32_to_fp16(0.0f);
        }
        pool[b] = pruned ? b * Qn : b * L;
        for (size_t mi = 0; mi < items[b].markers.size(); ++mi) {
            const int32_t mpos = items[b].markers[mi];
            if (mpos < 0 || mpos >= n) fail("marker outside sequence");
            rows.push_back(pruned ? b * Qn + 1 + static_cast<int>(mi) : b * L + mpos);
        }
    }
    if (pruned) {
        ggml_backend_tensor_set(io.qrows, qrows.data(), 0, ggml_nbytes(io.qrows));
        ggml_backend_tensor_set(io.mask_q, mq.data(), 0, ggml_nbytes(io.mask_q));
    }
    ggml_backend_tensor_set(io.qtype, qt.data(), 0, ggml_nbytes(io.qtype));
    if (!rows.empty()) ggml_backend_tensor_set(io.gather_rows, rows.data(), 0, ggml_nbytes(io.gather_rows));
    ggml_backend_tensor_set(io.pool_rows, pool.data(), 0, ggml_nbytes(io.pool_rows));

    if (std::getenv("STATIM_PROFILE")) {
        // per-op timing: evaluate the graph one node at a time
        std::map<std::string, double> by_op;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_cgraph gv = ggml_graph_view(gf, i, i + 1);
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(impl_->backend, &gv);
            ggml_tensor* n = ggml_graph_node(gf, i);
            std::string k = ggml_op_desc(n);
            if (n->op == GGML_OP_MUL_MAT)
                k += std::string(n->src[0]->buffer == nullptr || std::strchr(ggml_get_name(n->src[0]), '.') ? "(W " : "(A ") +
                     ggml_type_name(n->src[0]->type) + ")";
            by_op[k] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        for (auto& [k, v] : by_op) std::fprintf(stderr, "  %-22s %9.1f ms\n", k.c_str(), v);
    } else if (ggml_backend_graph_compute(impl_->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(c);
        fail("graph compute failed");
    }

    std::vector<float> logits(n_markers), pooled(static_cast<size_t>(d) * B);
    if (n_markers) ggml_backend_tensor_get(io.logits, logits.data(), 0, ggml_nbytes(io.logits));
    ggml_backend_tensor_get(io.pooled, pooled.data(), 0, ggml_nbytes(io.pooled));
    ggml_free(c);

    // ---- per-item outputs + act head on the host
    std::vector<ItemResult> out(B);
    size_t off = 0;
    const int A0 = static_cast<int>(M.act0_b.size()), NA = static_cast<int>(M.act2_b.size());
    std::vector<float> in(d + 4), hid(A0);
    for (int b = 0; b < B; ++b) {
        const size_t k = items[b].markers.size();
        out[b].logits.assign(logits.begin() + off, logits.begin() + off + k);
        off += k;
        // features: top1, top1 - top2, normalized entropy, k / 255 (see laya.common.DecisionModel)
        std::vector<float> p(out[b].logits);
        float mx = p.empty() ? 0.f : *std::max_element(p.begin(), p.end()), s = 0.f;
        for (float& v : p) s += (v = std::exp(v - mx));
        for (float& v : p) v /= s;
        std::vector<float> sorted(p);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        float top1 = sorted.empty() ? 0.f : sorted[0], top2 = sorted.size() > 1 ? sorted[1] : 0.f;
        float kk = static_cast<float>(std::max<size_t>(k, 2)), ent = 0.f;
        for (float v : p) ent -= v * std::log(std::max(v, 1e-9f));
        ent /= std::log(kk);
        std::copy(pooled.begin() + static_cast<size_t>(b) * d, pooled.begin() + static_cast<size_t>(b + 1) * d, in.begin());
        in[d] = top1;
        in[d + 1] = top1 - top2;
        in[d + 2] = ent;
        in[d + 3] = kk / 255.0f;
        for (int j = 0; j < A0; ++j) {
            float acc = M.act0_b[j];
            const float* w = &M.act0_w[static_cast<size_t>(j) * (d + 4)];
            for (int i = 0; i < d + 4; ++i) acc += w[i] * in[i];
            hid[j] = 0.5f * acc * (1.0f + std::erf(acc * static_cast<float>(M_SQRT1_2)));
        }
        std::vector<float> act(NA);
        float amx = -INFINITY;
        for (int j = 0; j < NA; ++j) {
            float acc = M.act2_b[j];
            for (int i = 0; i < A0; ++i) acc += M.act2_w[static_cast<size_t>(j) * A0 + i] * hid[i];
            act[j] = acc;
            amx = std::max(amx, acc);
        }
        float as = 0.f;
        for (float& v : act) as += (v = std::exp(v - amx));
        for (float& v : act) v /= as;
        out[b].act_probs = std::move(act);
    }
    return out;
}

}  // namespace statim
