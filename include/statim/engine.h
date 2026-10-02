// Statim — typed decisions (choice / score / noul) over any state, Laya/Jev-compatible output.
#pragma once

#include <chrono>
#include <atomic>
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
    int ensemble = 1;              // >1: average choice answers over K cyclic option orders (position debiasing)
    double ensemble_margin = 1.0;  // only ensemble when top-1 minus top-2 probability is below this (1 = always)
    bool return_logits = false;    // add the raw (pre-temperature) option logits to every answer
    // Contextual calibration (Zhao et al., 2021): divide out the answer distribution the question
    // produces on content-free versions of the state. Label-free; the content-free scores are
    // computed once per question and state shape, then cached.
    bool calibrate = false;
    // Cooperative request deadline; disabled for ordinary in-process inference.
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    // Shared because one HTTP watcher may outlive the stack frame that starts a forward pass.
    // Null keeps ordinary in-process inference free of an atomic load.
    std::shared_ptr<std::atomic<bool>> cancelled;
};

// Python json.dumps(ensure_ascii=False) with the given separators; key order preserved.
std::string py_json_dumps(const ojson& v, const char* item_sep = ", ", const char* key_sep = ": ");

struct Question;  // validated question (engine internal)

// Consensus of several checkpoints: weighted mean of their option log-probabilities (the
// results must have been produced with return_logits). Answers are re-decoded at T = 1.
ojson fuse_answers(const ojson& questions, const std::vector<const ojson*>& results, const std::vector<double>& weights);

// Question families for "adapter": "auto" routing (the decision categories of the training
// mixture). question_family() returns the family of one question, or nothing:
//   1. the question ID is split into lowercase ASCII words (runs of letters and digits); if they
//      contain keywords of exactly one family, that family;
//   2. if they contain none, the same test on the instructions (when a string);
//   3. otherwise (no keyword, or keywords of several families) no family.
// request_family() is the family shared by every question of a request, or nothing.
const std::vector<std::pair<std::string, std::vector<std::string>>>& question_families();
std::optional<std::string> question_family(const std::string& id, const ojson& question);
std::optional<std::string> request_family(const ojson& questions);

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
    std::vector<ItemResult> run_packed(const std::vector<Item>& items);
    std::vector<double> null_logp(const Question& q, const ojson& state, int max_len, int head_max_len);

    struct Impl;
    std::shared_ptr<Model> model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace statim
