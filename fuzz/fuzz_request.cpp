// libFuzzer harness: POST /v1/systemone and /v1/systemone/batch bodies.
//
// Runs exactly what the HTTP handler runs on a request body -- parse_decide_request() (JSON
// preflight, field validation, option parsing, per-state limits), check_work(), question
// validation, tokenization, prompt packing, a real forward pass on a tiny two-layer model, answer
// decoding (small requests), the consensus fusion and response serialization. Every input is tried as a single
// and as a batch request against a Metaspace (Gemma-style) and a ByteLevel (GPT-2-style) model.
//
// Contract checked: a caller mistake surfaces as HttpError 400/413/422 or QuestionError (422);
// any other exception, crash, sanitizer report, leak or hang is a bug (the server would answer
// 500 or die).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "statim/engine.h"
#include "statim/security.h"

using namespace statim;

namespace {
struct Loaded {
    std::shared_ptr<Model> model;
    std::unique_ptr<Engine> engine;
};
std::vector<Loaded>* g_models = nullptr;

void run(const std::string& raw, bool batch) {
    DecideRequest r;
    try {
        r = parse_decide_request(raw, batch);
    } catch (const HttpError& e) {
        if (e.status != 400 && e.status != 413 && e.status != 422) statim_fuzz::violated("unexpected HTTP status");
        return;
    } catch (const QuestionError&) {
        return;
    }
    const ojson& questions = r.body["questions"];
    std::vector<ojson> per_model;
    for (auto& m : *g_models) {
        try {
            check_work(questions, r.states.size(), m.model->hparams(), r.opts, SecurityLimits{}, 2);
            // Tokenization and prompt packing for every state (cheap) ...
            size_t rows = 0;
            for (const auto& st : r.states) rows += m.engine->encode(st, questions, r.opts).size();
            // ... a real forward pass only while it stays fast under ASan (each row is one encoder
            // sequence; ensemble and calibration multiply it inside decide_batch).
            if (rows * static_cast<size_t>(r.opts.ensemble) * (r.opts.calibrate ? 4 : 1) > 16) continue;
            DecideOptions opts = r.opts;
            opts.return_logits = true;  // what consensus requests; exercises the logits output path
            opts.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            std::vector<ojson> results = m.engine->decide_batch(r.states, questions, opts);
            if (results.size() != r.states.size()) statim_fuzz::violated("one result per state");
            for (auto& res : results) {
                // mirrors the handler's post-processing
                (void)res.at("usage").at("input_tokens").get<size_t>();
                if (res.contains("answers"))
                    for (auto& [qid, ans] : res["answers"].items())
                        if (ans.contains("answer_confidence")) (void)(ans["answer_confidence"].get<double>() < 0.5);
            }
            (void)ojson{{"results", results}}.dump();
            per_model.push_back(results.front());
        } catch (const HttpError& e) {
            if (e.status != 400 && e.status != 413 && e.status != 422) statim_fuzz::violated("unexpected HTTP status");
            return;
        } catch (const QuestionError&) {
            return;
        }
    }
    if (per_model.size() == 2) {
        ojson fused = fuse_answers(questions, {&per_model[0], &per_model[1]}, {0.5, 0.5});
        (void)fused.dump();
    }
}
}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    g_models = new std::vector<Loaded>();
    RunOptions ro;
    ro.n_threads = 1;
    for (const char* name : {"tiny-metaspace.gguf", "tiny-bytelevel.gguf"}) {
        Loaded l;
        l.model = Model::load(statim_fuzz::data_dir() + "/" + name, "cpu");
        l.engine = std::make_unique<Engine>(l.model, ro);
        g_models->push_back(std::move(l));
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string raw(reinterpret_cast<const char*>(data), size);
    run(raw, false);
    run(raw, true);
    return 0;
}
