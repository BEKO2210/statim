// libFuzzer harness: Model::load on arbitrary files (seeds: the tiny GGUF models).
//
// A malformed or hostile model file must be rejected with an exception at load time. When a file
// is accepted, the model must also be safe to run: the harness tokenizes and runs one small
// request (choice, score and noul questions) through a real Engine. Crashes, aborts (GGML_ASSERT),
// sanitizer reports, leaks and hangs are bugs.
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "common.h"
#include "statim/engine.h"
#include "statim/security.h"

using namespace statim;

namespace {
int g_fd = -1;
std::string g_path;
ojson* g_questions = nullptr;
}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    g_fd = memfd_create("statim-fuzz-gguf", 0);
    if (g_fd < 0) statim_fuzz::violated("memfd_create failed");
    g_path = "/proc/self/fd/" + std::to_string(g_fd);
    g_questions = new ojson(ojson::parse(R"({
        "pick": {"type": "choice", "instructions": "Pick one", "criteria": {"a": "first", "b": null, "c": ""}},
        "rate": {"type": "score", "instructions": "Rate it", "criteria": ["low", "mid", "high"]},
        "ok": {"type": "noul", "instructions": "Is it fine?"}})"));
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (ftruncate(g_fd, 0) != 0 || pwrite(g_fd, data, size, 0) != static_cast<ssize_t>(size))
        statim_fuzz::violated("cannot write the model file");
    std::shared_ptr<Model> model;
    try {
        model = Model::load(g_path, "cpu");
    } catch (const std::exception&) {
        return 0;  // rejected: the expected outcome for almost every input
    }
    RunOptions ro;
    ro.n_threads = 1;
    Engine engine(model, ro);
    DecideOptions opts;
    opts.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    opts.ensemble = 2;
    try {
        (void)engine.decide(ojson("the cat and the <mask> in a hat \xC3\xA9\xE2\x82\xAC"), *g_questions, opts).dump();
        opts.calibrate = true;
        (void)engine.decide(ojson{{"text", "and in"}, {"n", 3}}, *g_questions, opts).dump();
    } catch (const HttpError&) {
    } catch (const QuestionError&) {
    }
    return 0;
}
