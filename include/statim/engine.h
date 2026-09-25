// Statim — typed decisions (choice / score / noul) over any state, Laya/Jev-compatible output.
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "statim/model.h"

namespace statim {

using ojson = nlohmann::ordered_json;

// A caller mistake in a question definition (HTTP 422). The message names the question.
struct QuestionError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct DecideOptions {
    std::optional<int> max_len, head_max_len;  // override the checkpoint's token budgets
    std::optional<std::string> lang;          // selects per-language temperatures if the model has them
    int ensemble = 1;  // >1: average over cyclic option orders (choice/score), reduces position bias
};

// Python json.dumps(ensure_ascii=False) with the given separators; key order preserved.
std::string py_json_dumps(const ojson& v, const char* item_sep = ", ", const char* key_sep = ": ");

class Engine {
public:
    Engine(std::shared_ptr<Model> model, RunOptions run = {});
    ~Engine();

    // One state, many questions: {"model", "answers", "usage"} exactly like laya.Agent.system_one.
    ojson decide(const ojson& state, const ojson& questions, const DecideOptions& opts = {});

    // Many states, same questions, packed into shared forward passes.
    std::vector<ojson> decide_batch(const std::vector<ojson>& states, const ojson& questions,
                                    const DecideOptions& opts = {});

    // Tokenized encoder rows for one state (exposed for parity tests and benchmarks).
    std::vector<Item> encode(const ojson& state, const ojson& questions, const DecideOptions& opts = {}) const;

    const Model& model() const { return *model_; }

private:
    struct Impl;
    std::shared_ptr<Model> model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace statim
