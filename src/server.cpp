// Statim HTTP server: Jev/Laya-compatible POST /v1/systemone plus operational endpoints.
#include "statim/server.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "statim/http_security.h"
#include "playground.inc"
#include "statim/engine.h"

namespace statim {

namespace {

// A Prometheus label value: backslash, double quote and newline escaped (text exposition format).
// Model names come from the command line and may contain any of them.
std::string prom_label(const std::string& v) {
    std::string out;
    for (char c : v) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

std::string new_request_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(rng()));
    return buf;
}

// Cheap English detector: nearly all letters ASCII and enough common English function words.
bool looks_english(const std::string& text) {
    size_t ascii_letters = 0, other_letters = 0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            ascii_letters += std::isalpha(c) != 0;
            ++i;
        } else {
            ++other_letters;
            i += c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        }
    }
    if (ascii_letters == 0 || other_letters * 50 > ascii_letters) return false;
    static const char* kWords[] = {"the", "and", "you", "to", "is", "of", "for", "please", "my", "we", "i", "it",
                                   "this", "that", "with", "have", "not", "are", "was", "can", "your", "our", "on"};
    size_t words = 0, hits = 0;
    std::string w;
    auto flush = [&] {
        if (w.empty()) return;
        ++words;
        for (const char* k : kWords) hits += w == k;
        w.clear();
    };
    for (unsigned char c : text) {
        if (std::isalpha(c)) w.push_back(static_cast<char>(std::tolower(c)));
        else flush();
    }
    flush();
    return words < 4 ? hits > 0 || words > 0 : hits * 10 >= words;  // >= 10% function words
}

// Fixed-bucket latency histogram for Prometheus.
struct Histogram {
    static constexpr double kBounds[] = {5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
    std::atomic<uint64_t> buckets[std::size(kBounds) + 1]{};
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum_us{0};
    void observe(double ms) {
        size_t i = 0;
        while (i < std::size(kBounds) && ms > kBounds[i]) ++i;
        buckets[i]++;
        count++;
        sum_us += static_cast<uint64_t>(ms * 1000.0);
    }
};

struct Summary {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum_milli{0};
    void observe(double value) {
        count++;
        sum_milli += static_cast<uint64_t>(std::max(0.0, value) * 1000.0);
    }
};

// A fixed number of compute slots per base model (each engine owns its compute buffers; weights
// are shared). The base model and its adapter views share the slots, so adapters never add
// compute threads, and they share the engines too: the pool never holds more engines than slots.
// An engine serves one view; a view gets one when it first runs, and when every engine is taken,
// an idle engine of another view is dropped for it. Compute buffers only grow, so without the cap
// every adapter would keep its own set of peak-sized buffers. Requests borrow an engine; when all
// slots are busy and the wait queue is full the caller gets 503 instead of piling up.
class EnginePool {
public:
    EnginePool(const std::shared_ptr<Model>& model, int workers, int threads_per_worker) {
        ro_.n_threads = threads_per_worker;
        for (int i = 0; i < workers; ++i) free_[model.get()].push_back(std::make_unique<Engine>(model, ro_));
        size_ = engines_ = workers;
    }
    struct Lease {
        EnginePool* pool;
        const Model* key;
        std::unique_ptr<Engine> engine;
        ~Lease() {
            if (engine) pool->release(key, std::move(engine));
        }
    };
    std::unique_ptr<Lease> acquire(const std::shared_ptr<Model>& model, std::chrono::steady_clock::time_point deadline) {
        auto l = std::make_unique<Lease>();  // allocated before a slot is taken: nothing below may throw
        l->pool = this;                      // until the guarded engine construction
        l->key = model.get();
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_until(lk, deadline, [&] { return busy_ < size_; })) throw HttpError(503, "inference queue deadline exceeded");
        ++busy_;
        if (auto it = free_.find(model.get()); it != free_.end() && !it->second.empty()) {
            l->engine = std::move(it->second.back());
            it->second.pop_back();
            return l;
        }
        // At the cap, some engine is idle: engines_ counts the idle ones and those of the other busy
        // slots, and busy_ (this request included) is at most size_. This view has none idle, so it
        // belongs to another view; take it from the view with the most idle engines.
        std::vector<std::unique_ptr<Engine>>* from = nullptr;
        if (engines_ >= size_)
            for (auto& [key, idle] : free_)
                if (key != model.get() && !idle.empty() && (!from || idle.size() > from->size())) from = &idle;
        std::unique_ptr<Engine> evicted;
        if (from) {
            evicted = std::move(from->back());
            from->pop_back();
        } else {
            ++engines_;
        }
        lk.unlock();
        evicted.reset();  // frees its compute buffers before the new engine allocates its own
        try {
            l->engine = std::make_unique<Engine>(model, ro_);
        } catch (...) {
            l->engine.reset();
            lk.lock();
            --engines_;
            --busy_;
            cv_.notify_one();
            throw;
        }
        return l;
    }
    int busy() {
        std::lock_guard<std::mutex> lk(mu_);
        return busy_;
    }
    int size() const { return size_; }
    int engines() {
        std::lock_guard<std::mutex> lk(mu_);
        return engines_;
    }

private:
    void release(const Model* key, std::unique_ptr<Engine> e) noexcept {
        {
            std::lock_guard<std::mutex> lk(mu_);
            --busy_;
            try {
                free_[key].push_back(std::move(e));
            } catch (...) {  // out of memory: give up the engine (destroyed on return), never the slot
                --engines_;
            }
        }
        cv_.notify_one();
    }
    RunOptions ro_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<const Model*, std::vector<std::unique_ptr<Engine>>> free_;
    int size_ = 0;
    int engines_ = 0;  // idle engines plus one per busy slot (holding or building one); <= size_
    int busy_ = 0;
};

// A LoRA adapter view of a loaded model (Model::with_adapter), selected with "adapter".
struct LoadedAdapter {
    std::string name;
    std::shared_ptr<Model> model;
};

struct LoadedModel {
    std::string name;
    std::shared_ptr<Model> model;
    std::unique_ptr<EnginePool> pool;
    std::vector<LoadedAdapter> adapters;
    const LoadedAdapter* adapter(const std::string& n) const {
        for (auto& a : adapters)
            if (a.name == n) return &a;
        return nullptr;
    }
    // "adapter": "auto": the first adapter (in --adapter order) that serves the family
    const LoadedAdapter* adapter_for(const std::string& family) const {
        for (auto& a : adapters)
            for (auto& c : a.model->adapter()->categories)
                if (c == family) return &a;
        return nullptr;
    }
    std::string adapter_names() const {
        std::string s;
        for (auto& a : adapters) s += (s.empty() ? "" : ", ") + a.name;
        return s.empty() ? "none loaded" : "loaded: " + s;
    }
};

// Validated single requests wait here before any engine is leased.  Each route has its own
// coordinator, so an incompatible group or a busy checkpoint cannot hold another model's window.
class MicroBatcher {
public:
    using Clock = std::chrono::steady_clock;
    using Run = std::function<std::vector<ojson>(const std::vector<ojson>&, const ojson&, DecideOptions)>;

    struct Item {
        ojson state;
        ojson questions;
        DecideOptions opts;
        Clock::time_point deadline;
        Clock::time_point queued_at;
        std::mutex mu;
        std::condition_variable cv;
        bool cancelled = false;
        bool done = false;
        ojson result;
        std::exception_ptr error;
        double infer_ms = 0;
    };

    MicroBatcher(int window_ms, int max_batch, int dispatchers, Run run, Summary& sizes, Summary& waits)
        : window_(window_ms), max_batch_(max_batch), run_(std::move(run)), sizes_(sizes), waits_(waits) {
        collector_ = std::thread([this] { collect(); });
        for (int i = 0; i < dispatchers; ++i) workers_.emplace_back([this] { execute(); });
    }

    ~MicroBatcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        collector_.join();
        for (auto& t : workers_) t.join();
    }

    std::pair<ojson, double> submit(std::string key, ojson state, ojson questions, DecideOptions opts,
                                    Clock::time_point deadline, const std::function<bool()>& disconnected) {
        auto item = std::make_shared<Item>();
        item->state = std::move(state);
        item->questions = std::move(questions);
        item->opts = std::move(opts);
        item->deadline = deadline;
        item->queued_at = Clock::now();
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto& group = groups_[std::move(key)];
            if (group.items.empty()) group.first = item->queued_at;
            group.items.push_back(item);
        }
        cv_.notify_all();

        std::unique_lock<std::mutex> lk(item->mu);
        while (!item->done) {
            const auto poll = std::min(deadline, Clock::now() + std::chrono::milliseconds(10));
            item->cv.wait_until(lk, poll);
            if (!item->done && (Clock::now() >= deadline || disconnected())) {
                item->cancelled = true;
                throw HttpError(422, Clock::now() >= deadline ? "inference deadline exceeded" : "inference cancelled");
            }
        }
        if (Clock::now() >= deadline) throw HttpError(422, "inference deadline exceeded");
        if (item->error) std::rethrow_exception(item->error);
        return {std::move(item->result), item->infer_ms};
    }

private:
    struct Group {
        Clock::time_point first;
        std::deque<std::shared_ptr<Item>> items;
    };

    void collect() {
        for (;;) {
            std::vector<std::shared_ptr<Item>> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                for (;;) {
                    if (stopping_) return;
                    auto ready = groups_.end();
                    auto wake = Clock::time_point::max();
                    const auto now = Clock::now();
                    for (auto it = groups_.begin(); it != groups_.end(); ++it) {
                        const auto due = it->second.first + std::chrono::milliseconds(window_);
                        if (static_cast<int>(it->second.items.size()) >= max_batch_ || now >= due) {
                            ready = it;
                            break;
                        }
                        wake = std::min(wake, due);
                    }
                    if (ready != groups_.end()) {
                        auto& q = ready->second.items;
                        while (!q.empty() && static_cast<int>(batch.size()) < max_batch_) {
                            batch.push_back(std::move(q.front()));
                            q.pop_front();
                        }
                        if (q.empty()) groups_.erase(ready);
                        else ready->second.first = q.front()->queued_at;
                        break;
                    }
                    if (wake == Clock::time_point::max()) cv_.wait(lk);
                    else cv_.wait_until(lk, wake);
                }
            }
            const auto collected = Clock::now();
            for (auto& item : batch)
                waits_.observe(std::chrono::duration<double, std::milli>(collected - item->queued_at).count());
            {
                std::lock_guard<std::mutex> lk(mu_);
                ready_.push_back(std::move(batch));
            }
            cv_.notify_all();
        }
    }

    void execute() {
        for (;;) {
            std::vector<std::shared_ptr<Item>> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stopping_ || !ready_.empty(); });
                if (ready_.empty()) return;
                batch = std::move(ready_.front());
                ready_.pop_front();
            }
            std::vector<std::shared_ptr<Item>> active;
            const auto started = Clock::now();
            for (auto& item : batch) {
                std::lock_guard<std::mutex> lk(item->mu);
                if (!item->cancelled && started < item->deadline) active.push_back(item);
            }
            if (active.empty()) continue;
            sizes_.observe(static_cast<double>(active.size()));

            std::vector<ojson> states;
            states.reserve(active.size());
            auto deadline = active.front()->deadline;
            for (auto& item : active) {
                states.push_back(item->state);
                deadline = std::max(deadline, item->deadline);
            }
            DecideOptions opts = active.front()->opts;
            opts.deadline = deadline;
            std::vector<ojson> results;
            std::exception_ptr error;
            const auto ti = Clock::now();
            try {
                results = run_(states, active.front()->questions, std::move(opts));
            } catch (...) {
                error = std::current_exception();
            }
            const double infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - ti).count();
            for (size_t i = 0; i < active.size(); ++i) {
                auto& item = active[i];
                {
                    std::lock_guard<std::mutex> lk(item->mu);
                    if (!item->cancelled) {
                        item->error = error;
                        if (!error) item->result = std::move(results[i]);
                        item->infer_ms = infer_ms;
                        item->done = true;
                    }
                }
                item->cv.notify_one();
            }
        }
    }

    int window_;
    int max_batch_;
    Run run_;
    Summary& sizes_;
    Summary& waits_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<std::string, Group> groups_;
    std::deque<std::vector<std::shared_ptr<Item>>> ready_;
    bool stopping_ = false;
    std::thread collector_;
    std::vector<std::thread> workers_;
};

httplib::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->stop();
}

}  // namespace

int run_server(const ServerConfig& cfg) {
    if (cfg.max_concurrent < 1 || cfg.max_concurrent > 256 || cfg.workers < 1 || cfg.workers > 64 ||
        cfg.ensemble < 1 || cfg.ensemble > 8 || cfg.max_len < 0 || cfg.head_max_len < 0 || cfg.port < 1 || cfg.port > 65535 ||
        cfg.batch_window_ms < 0 || cfg.max_batch < 1 || cfg.max_batch > static_cast<int>(max_batch_states))
        throw std::runtime_error("invalid server worker, concurrency, ensemble, port or token budget configuration");
    if (cfg.api_keys.empty() && !is_loopback_host(cfg.host) && !cfg.allow_unauthenticated) {
        const std::string host = ojson(cfg.host).dump();
        std::fprintf(stderr, "statim serve: error: host %s is not loopback and no API key is configured; "
                             "set STATIM_API_KEY, pass --api-key-file FILE, or pass --allow-unauthenticated "
                             "(trusted networks only)\n", host.c_str());
        return 2;
    }
    std::vector<LoadedModel> models;
    const int workers = std::max(1, cfg.workers);
    const int threads = cfg.threads > 0 ? cfg.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int per_worker = std::max(1, threads / workers);
    for (const auto& [name, path] : cfg.models) {
        auto t0 = std::chrono::steady_clock::now();
        LoadedModel lm;
        lm.name = name;
        lm.model = Model::load(path, cfg.device);
        DecideOptions defaults;
        if (cfg.max_len) defaults.max_len = cfg.max_len;
        if (cfg.head_max_len) defaults.head_max_len = cfg.head_max_len;
        effective_max_len(lm.model->hparams(), defaults);
        lm.pool = std::make_unique<EnginePool>(lm.model, workers, per_worker);
        std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "info"}, {"event", "model_loaded"},
            {"model", name}, {"path", path}, {"weights", lm.model->hparams().weight_type}, {"device", lm.model->device()},
            {"bytes", lm.model->weight_bytes()}, {"workers", workers}, {"threads_per_worker", per_worker},
            {"ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}}.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
        models.push_back(std::move(lm));
    }
    if (models.empty()) {
        std::fprintf(stderr, "statim serve: no model given (-m path.gguf)\n");
        return 2;
    }
    for (const AdapterSpec& spec : cfg.adapters) {
        LoadedModel* lm = nullptr;
        for (auto& m : models)
            if (m.name == spec.model) lm = &m;
        if (!lm) throw std::runtime_error("--adapter " + spec.model + ":" + spec.name + ": no model named '" + spec.model + "'");
        const bool valid_name = !spec.name.empty() && spec.name.size() <= 64 &&
            std::all_of(spec.name.begin(), spec.name.end(), [](char c) {
                return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
            });
        if (!valid_name || spec.name == "auto" || spec.name == "none")
            throw std::runtime_error("--adapter name '" + spec.name + "' must be 1-64 of [A-Za-z0-9_.-] and not 'auto' or 'none'");
        if (lm->adapter(spec.name)) throw std::runtime_error("--adapter " + spec.model + ":" + spec.name + " given twice");
        LoadedAdapter la;
        la.name = spec.name;
        std::optional<AdapterMode> mode;
        if (!cfg.adapter_mode.empty()) mode = cfg.adapter_mode == "runtime" ? AdapterMode::runtime : AdapterMode::merge;
        la.model = Model::with_adapter(lm->model, spec.path, mode, threads);
        const AdapterInfo& ai = *la.model->adapter();
        std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "info"}, {"event", "adapter_loaded"},
            {"model", lm->name}, {"adapter", la.name}, {"path", spec.path}, {"mode", ai.mode == AdapterMode::merge ? "merge" : "runtime"},
            {"rank", ai.rank}, {"pairs", ai.n_pairs}, {"pairs_applied", ai.n_applied}, {"categories", ai.categories},
            {"bytes", ai.bytes}, {"ms", ai.load_ms}}.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
        lm->adapters.push_back(std::move(la));
    }

    const bool any_adapters = !cfg.adapters.empty();
    std::vector<std::string> api_keys = cfg.api_keys;
    std::atomic<int> in_flight{0};
    std::atomic<uint64_t> req_total{0}, req_errors{0}, tokens_total{0}, rejected_busy{0};
    std::mutex status_mu;
    std::map<int, uint64_t> status_counts;
    Histogram latency;
    Summary batch_size, batch_wait_ms;
    const auto started = std::chrono::steady_clock::now();

    SecureServer srv(cfg.request_timeout);
    g_server = &srv;
    srv.new_task_queue = [&] { return new httplib::ThreadPool(static_cast<size_t>(cfg.max_concurrent + 4), 0, cfg.http_queue); };
    srv.set_payload_max_length(max_body_bytes);
    srv.set_read_timeout(cfg.request_timeout, 0);
    srv.set_write_timeout(30, 0);
    srv.set_keep_alive_max_count(100);
    srv.set_keep_alive_timeout(2);

    auto send_json = [](httplib::Response& res, int status, const ojson& body) {
        res.status = status;
        res.set_content(body.dump(), "application/json");
    };
    auto authorized = [&](const httplib::Request& req) { return bearer_authorized(req, api_keys); };
    configure_http_security(srv, api_keys);
    auto by_name = [&](const std::string& n) -> LoadedModel* {
        for (auto& m : models)
            if (m.name == n) return &m;
        return nullptr;
    };
    // Explicit "model" wins; otherwise, with both an "english" and a "multilingual" checkpoint
    // loaded, English text goes to the English one and everything else to the multilingual one.
    auto find_model = [&](const ojson& body, const std::vector<ojson>& states, std::string& reason) -> LoadedModel& {
        if (body.contains("model") && body["model"].is_string()) {
            const std::string want = body["model"].get<std::string>();
            for (auto& m : models)
                if (m.name == want || "convaiinnovations/laya-" + m.name == want) {
                    reason = "requested";
                    return m;
                }
            // unknown ids (e.g. a Jev model name) fall through to routing, like laya.serve
        }
        // a named adapter that exactly one model carries selects that model
        if (body.contains("adapter") && body["adapter"].is_string()) {
            const std::string want = body["adapter"].get<std::string>();
            LoadedModel* only = nullptr;
            int n = 0;
            for (auto& m : models)
                if (m.adapter(want)) only = &m, ++n;
            if (n == 1) {
                reason = "adapter";
                return *only;
            }
        }
        LoadedModel* en = by_name("english");
        LoadedModel* ml = by_name("multilingual");
        if (en && ml) {
            std::string text;
            for (const auto& st : states) text += st.is_string() ? st.get<std::string>() : py_json_dumps(st);
            bool english = looks_english(text);
            reason = english ? "lang:en" : "lang:other";
            return english ? *en : *ml;
        }
        reason = "default";
        return models.front();
    };

    // weights: the base model, or one of its adapter views
    auto run_one_model = [](LoadedModel& lm, const std::shared_ptr<Model>& weights, const std::vector<ojson>& states,
                            const ojson& questions, DecideOptions opts) {
        auto lease = lm.pool->acquire(weights, opts.deadline);
        return lease->engine->decide_batch(states, questions, opts);
    };
    LoadedModel* consensus_en = by_name("english");
    LoadedModel* consensus_ml = by_name("multilingual");
    auto run_consensus = [&](const std::vector<ojson>& states, const ojson& questions, DecideOptions opts) {
        const bool keep_logits = opts.return_logits;
        opts.return_logits = true;
        std::vector<ojson> re = run_one_model(*consensus_en, consensus_en->model, states, questions, opts);
        std::vector<ojson> rm = run_one_model(*consensus_ml, consensus_ml->model, states, questions, opts);
        std::vector<ojson> results;
        results.reserve(re.size());
        for (size_t i = 0; i < re.size(); ++i) {
            ojson f = fuse_answers(questions, {&re[i], &rm[i]}, {0.5, 0.5});
            f["model"] = "consensus";
            if (keep_logits)
                for (auto it = f["answers"].begin(); it != f["answers"].end(); ++it) {
                    it.value()["logits_by_model"] = {{"english", re[i]["answers"][it.key()]["logits"]},
                                                       {"multilingual", rm[i]["answers"][it.key()]["logits"]}};
                }
            results.push_back(std::move(f));
        }
        return results;
    };

    // one coordinator per weight set: requests for different adapters never share a forward pass
    std::map<const Model*, std::unique_ptr<MicroBatcher>> batchers;
    std::unique_ptr<MicroBatcher> consensus_batcher;
    if (cfg.batch_window_ms > 0) {
        for (auto& m : models) {
            std::vector<std::shared_ptr<Model>> weights{m.model};
            for (auto& a : m.adapters) weights.push_back(a.model);
            for (auto& w : weights)
                batchers[w.get()] = std::make_unique<MicroBatcher>(
                    cfg.batch_window_ms, cfg.max_batch, workers,
                    [&m, w, &run_one_model](const std::vector<ojson>& states, const ojson& questions, DecideOptions opts) {
                        return run_one_model(m, w, states, questions, std::move(opts));
                    }, batch_size, batch_wait_ms);
        }
        if (consensus_en && consensus_ml)
            consensus_batcher = std::make_unique<MicroBatcher>(cfg.batch_window_ms, cfg.max_batch, workers,
                                                               run_consensus, batch_size, batch_wait_ms);
    }

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool batch, const httplib::ContentReader& reader) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto supplied_id = req.get_header_value("X-Request-Id");
        const std::string rid = valid_request_id(supplied_id) ? supplied_id : new_request_id();
        res.set_header("X-Request-Id", rid);
        std::string model_name = "-", adapter_name = "-";
        size_t n_tokens = 0;
        int status = 200;
        struct InFlight {
            std::atomic<int>& n;
            bool held = false;
            ~InFlight() {
                if (held) --n;
            }
        } guard{in_flight};
        try {
            if (!authorized(req)) throw HttpError{401, "invalid or missing bearer token"};
            if (in_flight.fetch_add(1) >= cfg.max_concurrent) {
                --in_flight;
                rejected_busy++;
                res.set_header("Retry-After", "1");
                throw HttpError{503, "server busy, try again later"};
            }
            guard.held = true;
            std::string raw;
            bool oversized = false;
            if (!reader([&](const char* data, size_t n) {
                if (n > max_body_bytes - raw.size()) { oversized = true; return false; }
                raw.append(data, n);
                return true;
            })) throw HttpError(oversized || res.status == 413 ? 413 : 400, "request body incomplete or exceeds limit");
            RequestDefaults defaults;
            defaults.ensemble = cfg.ensemble;
            defaults.calibrate = cfg.calibrate;
            defaults.max_len = cfg.max_len;
            defaults.head_max_len = cfg.head_max_len;
            defaults.min_confidence = cfg.min_confidence;
            DecideRequest parsed = parse_decide_request(raw, batch, defaults, cfg.limits);
            const ojson& body = parsed.body;
            const ojson& questions = body["questions"];
            DecideOptions& opts = parsed.opts;
            const double min_confidence = parsed.min_confidence;
            std::vector<ojson>& states = parsed.states;

            // LoRA adapter: absent, null or "none" = the base weights; "auto" = the adapter that
            // serves the request's question family (request_family); otherwise a loaded adapter name.
            const std::string want_adapter =
                body.contains("adapter") && body["adapter"].is_string() ? body["adapter"].get<std::string>() : "none";
            const bool named_adapter = want_adapter != "none" && want_adapter != "auto";

            // Consensus: both English and multilingual checkpoints answer, their option
            // log-probabilities are averaged. Opt in per request ("model": "consensus") or by default
            // with --consensus; the default does not apply to a request that names an adapter, which
            // selects its model itself.
            LoadedModel* en = by_name("english");
            LoadedModel* ml = by_name("multilingual");
            const bool want_consensus =
                en && ml && ((body.contains("model") && body["model"] == "consensus") ||
                             (cfg.consensus && !body.contains("model") && !named_adapter));
            std::string reason;
            LoadedModel& lm = want_consensus ? *en : find_model(body, states, reason);

            const LoadedAdapter* adapter = nullptr;
            std::string adapter_reason = "none";
            if (want_consensus) {
                if (named_adapter) throw HttpError{422, "adapter '" + want_adapter + "' cannot be combined with consensus"};
                if (want_adapter == "auto") adapter_reason = "auto:consensus";
            } else if (want_adapter == "auto") {
                const auto family = request_family(questions);
                if (!family) adapter_reason = "auto:no-family";
                else if ((adapter = lm.adapter_for(*family))) adapter_reason = "auto:" + *family;
                else adapter_reason = "auto:" + *family + ":no-adapter";
            } else if (named_adapter) {
                adapter = lm.adapter(want_adapter);
                if (!adapter) {
                    std::string others;
                    for (auto& m : models)
                        for (auto& ad : m.adapters)
                            if (&m != &lm) others += (others.empty() ? "" : ", ") + m.name + ":" + ad.name;
                    throw HttpError{422, "unknown adapter '" + want_adapter + "' for model '" + lm.name + "' (" +
                                             lm.adapter_names() + (others.empty() ? "" : "; other models: " + others) + ")"};
                }
                adapter_reason = "requested";
            }
            const std::shared_ptr<Model>& weights = adapter ? adapter->model : lm.model;
            check_work(questions, states.size(), lm.model->hparams(), opts, cfg.limits, want_consensus ? 2 : 1);
            if (want_consensus) check_work(questions, states.size(), ml->model->hparams(), opts, cfg.limits, 2);
            opts.deadline = t0 + std::chrono::seconds(cfg.inference_timeout);
            model_name = want_consensus ? "consensus" : lm.name;
            adapter_name = adapter ? adapter->name : "-";
            std::vector<ojson> results;
            double infer_ms = 0;
            if (!batch && cfg.batch_window_ms > 0) {
                // dump() preserves validated object order.  Include language because multilingual
                // checkpoints may select a different temperature even when every listed option matches.
                ojson budgets = ojson::array({effective_max_len(lm.model->hparams(), opts),
                                              opts.head_max_len.value_or(lm.model->hparams().head_max_len)});
                if (want_consensus)
                    budgets.push_back({effective_max_len(ml->model->hparams(), opts),
                                       opts.head_max_len.value_or(ml->model->hparams().head_max_len)});
                ojson compatible = {questions, budgets, adapter ? ojson(adapter->name) : ojson(nullptr), opts.ensemble, opts.ensemble_margin,
                                    opts.calibrate, opts.return_logits, min_confidence,
                                    opts.lang ? ojson(*opts.lang) : ojson(nullptr)};
                MicroBatcher* batcher = want_consensus ? consensus_batcher.get() : batchers.at(weights.get()).get();
                auto [result, shared_ms] = batcher->submit(compatible.dump(), states.front(), questions, opts,
                                                           opts.deadline, req.is_connection_closed);
                results.push_back(std::move(result));
                infer_ms = shared_ms;
                if (want_consensus) reason = "consensus";
            } else {
                const auto ti = std::chrono::steady_clock::now();
                results = want_consensus ? run_consensus(states, questions, opts)
                                         : run_one_model(lm, weights, states, questions, opts);
                infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
                if (want_consensus) reason = "consensus";
            }
            for (auto& r : results) {
                n_tokens += r["usage"]["input_tokens"].get<size_t>();
                r["routing"] = {{"model", model_name}, {"reason", reason}, {"engine", "statim"},
                                {"weights", lm.model->hparams().weight_type}};
                if (any_adapters || body.contains("adapter")) {  // unchanged responses for servers without adapters
                    r["routing"]["adapter"] = adapter ? ojson(adapter->name) : ojson(nullptr);
                    r["routing"]["adapter_reason"] = adapter_reason;
                }
                if (min_confidence > 0 && r.contains("answers"))
                    for (auto& [qid, ans] : r["answers"].items())
                        if (ans.contains("answer_confidence"))
                            ans["escalate"] = ans["answer_confidence"].get<double>() < min_confidence;
            }
            char timing[64];
            std::snprintf(timing, sizeof timing, "inference;dur=%.2f", infer_ms);
            res.set_header("Server-Timing", timing);
            std::snprintf(timing, sizeof timing, "%.2f", infer_ms);
            res.set_header("X-Inference-Time-Ms", timing);
            const std::string response = (batch ? ojson{{"results", results}} : results.front()).dump();
            if (response.size() > cfg.limits.max_response_bytes) throw HttpError(413, "response exceeds byte budget");
            res.status = 200;
            res.set_content(response, "application/json");
        } catch (const HttpError& e) {
            status = e.status;
            res.set_header("Connection", "close");
            send_json(res, status, {{"detail", e.what()}});
        } catch (const QuestionError& e) {
            status = 422;
            send_json(res, status, {{"detail", e.what()}});
        } catch (const std::exception& e) {
            status = 500;
            std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "error"}, {"event", "inference_failed"},
                {"request_id", rid}, {"error", e.what()}}.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
            send_json(res, status, {{"detail", "inference failed"}});
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        req_total++;
        if (status >= 400) req_errors++;
        tokens_total += n_tokens;
        latency.observe(ms);
        {
            std::lock_guard<std::mutex> lk(status_mu);
            status_counts[status]++;
        }
        if (cfg.access_log)
            std::fprintf(stdout, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "info"}, {"event", "request"},
                {"request_id", rid}, {"path", req.path}, {"status", status}, {"model", model_name}, {"adapter", adapter_name}, {"tokens", n_tokens},
                {"ms", ms}, {"remote", req.remote_addr}}.dump().c_str());
        std::fflush(stdout);
    };

    srv.Post("/v1/systemone", [&](const httplib::Request& q, httplib::Response& r, const httplib::ContentReader& reader) { handle(q, r, false, reader); });
    srv.Post("/v1/systemone/batch", [&](const httplib::Request& q, httplib::Response& r, const httplib::ContentReader& reader) { handle(q, r, true, reader); });

    srv.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        if (cfg.playground) res.set_content(kPlaygroundHtml, "text/html; charset=utf-8");
        else send_json(res, 404, {{"detail", "not found"}});
    });
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, {{"status", "ok"}, {"version", STATIM_VERSION}});
    });
    srv.Get("/ready", [&](const httplib::Request&, httplib::Response& res) {
        bool ready = in_flight.load() < cfg.max_concurrent;
        send_json(res, ready ? 200 : 503, {{"ready", ready}});
    });
    srv.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req)) return send_json(res, 401, {{"detail", "invalid or missing bearer token"}});
        ojson data = ojson::array();
        for (auto& m : models) {
            const HParams& h = m.model->hparams();
            data.push_back({{"id", m.name}, {"object", "model"}, {"owned_by", "statim"}, {"source", h.name},
                            {"weights", h.weight_type}, {"layers", h.n_layer}, {"hidden", h.n_embd}, {"max_len", h.max_len},
                            {"vocab", m.model->tokenizer().vocab_size()}, {"device", m.model->device()}});
            ojson adapters = ojson::array();
            for (auto& a : m.adapters) {
                const AdapterInfo& ai = *a.model->adapter();
                adapters.push_back({{"id", a.name}, {"source", ai.name}, {"mode", ai.mode == AdapterMode::merge ? "merge" : "runtime"},
                                    {"rank", ai.rank}, {"alpha", ai.alpha}, {"pairs", ai.n_pairs}, {"pairs_applied", ai.n_applied},
                                    {"categories", ai.categories}, {"bytes", ai.bytes}});
            }
            if (any_adapters) data.back()["adapters"] = adapters;
        }
        send_json(res, 200, {{"object", "list"}, {"data", data}});
    });
    srv.Get("/metrics", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req)) return send_json(res, 401, {{"detail", "invalid or missing bearer token"}});
        std::ostringstream o;
        o << "# HELP statim_requests_total Inference requests by HTTP status.\n# TYPE statim_requests_total counter\n";
        {
            std::lock_guard<std::mutex> lk(status_mu);
            for (auto& [code, n] : status_counts) o << "statim_requests_total{code=\"" << code << "\"} " << n << "\n";
        }
        o << "# HELP statim_request_duration_ms End-to-end request latency.\n# TYPE statim_request_duration_ms histogram\n";
        uint64_t cum = 0;
        for (size_t i = 0; i < std::size(Histogram::kBounds); ++i) {
            cum += latency.buckets[i];
            o << "statim_request_duration_ms_bucket{le=\"" << Histogram::kBounds[i] << "\"} " << cum << "\n";
        }
        cum += latency.buckets[std::size(Histogram::kBounds)];
        o << "statim_request_duration_ms_bucket{le=\"+Inf\"} " << cum << "\n";
        o << "statim_request_duration_ms_sum " << latency.sum_us / 1000.0 << "\n";
        o << "statim_request_duration_ms_count " << latency.count << "\n";
        o << "# TYPE statim_input_tokens_total counter\nstatim_input_tokens_total " << tokens_total << "\n";
        o << "# TYPE statim_rejected_busy_total counter\nstatim_rejected_busy_total " << rejected_busy << "\n";
        o << "# HELP statim_batch_size States executed in server-created micro-batches.\n"
             "# TYPE statim_batch_size summary\nstatim_batch_size_sum " << batch_size.sum_milli / 1000.0
          << "\nstatim_batch_size_count " << batch_size.count << "\n";
        o << "# HELP statim_batch_wait_ms Time an admitted request waited for compatible peers.\n"
             "# TYPE statim_batch_wait_ms summary\nstatim_batch_wait_ms_sum " << batch_wait_ms.sum_milli / 1000.0
          << "\nstatim_batch_wait_ms_count " << batch_wait_ms.count << "\n";
        o << "# TYPE statim_in_flight gauge\nstatim_in_flight " << in_flight << "\n";
        o << "# TYPE statim_uptime_seconds gauge\nstatim_uptime_seconds "
          << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << "\n";
        // one block per metric family, as text parsers that close a family on a name change expect
        o << "# TYPE statim_workers_busy gauge\n";
        for (auto& m : models) o << "statim_workers_busy{model=\"" << prom_label(m.name) << "\"} " << m.pool->busy() << "\n";
        o << "# TYPE statim_model_info gauge\n";
        for (auto& m : models)
            o << "statim_model_info{model=\"" << prom_label(m.name) << "\",weights=\"" << prom_label(m.model->hparams().weight_type)
              << "\",version=\"" << STATIM_VERSION << "\"} 1\n";
        if (any_adapters) {
            o << "# HELP statim_engines Engines (compute buffers) held or being built for a model and its adapters; at most --workers.\n"
                 "# TYPE statim_engines gauge\n";
            for (auto& m : models) o << "statim_engines{model=\"" << prom_label(m.name) << "\"} " << m.pool->engines() << "\n";
            o << "# TYPE statim_adapter_info gauge\n";
            for (auto& m : models)
                for (auto& a : m.adapters)
                    o << "statim_adapter_info{model=\"" << prom_label(m.name) << "\",adapter=\"" << prom_label(a.name) << "\",mode=\""
                      << (a.model->adapter()->mode == AdapterMode::merge ? "merge" : "runtime") << "\"} 1\n";
            o << "# HELP statim_adapter_bytes Memory an adapter adds on top of the shared base weights.\n"
                 "# TYPE statim_adapter_bytes gauge\n";
            for (auto& m : models)
                for (auto& a : m.adapters)
                    o << "statim_adapter_bytes{model=\"" << prom_label(m.name) << "\",adapter=\"" << prom_label(a.name) << "\"} "
                      << a.model->adapter()->bytes << "\n";
        }
        res.set_content(o.str(), "text/plain; version=0.0.4");
    });

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "info"}, {"event", "listening"},
        {"host", cfg.host}, {"port", cfg.port}, {"auth", !api_keys.empty()}, {"auth_status", api_keys.empty() ? "off" : "on"}}.dump().c_str());
    if (api_keys.empty() && !is_loopback_host(cfg.host))
        std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "warn"}, {"event", "auth_off_on_network"},
            {"host", cfg.host}, {"detail", "listening beyond loopback without API keys: every endpoint is open; "
                                           "set STATIM_API_KEY or --api-key-file"}}.dump().c_str());
    if (!srv.listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "statim serve: cannot listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    std::fprintf(stderr, "{\"ts\":\"%s\",\"level\":\"info\",\"event\":\"shutdown\"}\n", now_iso8601().c_str());
    g_server = nullptr;
    return 0;
}

}  // namespace statim
