// Statim HTTP server: Jev/Laya-compatible POST /v1/systemone plus operational endpoints.
#include "statim/server.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>

#include "statim/http_security.h"
#include "playground.inc"
#include "statim/engine.h"

namespace statim {

namespace {

constexpr size_t kMaxBatchStates = 256;

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

// A fixed set of engines (each owns its compute buffers; weights are shared). Requests borrow
// one; when all are busy and the wait queue is full the caller gets 503 instead of piling up.
class EnginePool {
public:
    EnginePool(const std::shared_ptr<Model>& model, int workers, int threads_per_worker) {
        RunOptions ro;
        ro.n_threads = threads_per_worker;
        for (int i = 0; i < workers; ++i) free_.push_back(std::make_unique<Engine>(model, ro));
        size_ = workers;
    }
    struct Lease {
        EnginePool* pool;
        std::unique_ptr<Engine> engine;
        ~Lease() {
            if (engine) pool->release(std::move(engine));
        }
    };
    std::unique_ptr<Lease> acquire(std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_until(lk, deadline, [&] { return !free_.empty(); })) throw HttpError(503, "inference queue deadline exceeded");
        auto l = std::make_unique<Lease>();
        l->pool = this;
        l->engine = std::move(free_.back());
        free_.pop_back();
        return l;
    }
    int busy() {
        std::lock_guard<std::mutex> lk(mu_);
        return size_ - static_cast<int>(free_.size());
    }
    int size() const { return size_; }

private:
    void release(std::unique_ptr<Engine> e) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            free_.push_back(std::move(e));
        }
        cv_.notify_one();
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<Engine>> free_;
    int size_ = 0;
};

struct LoadedModel {
    std::string name;
    std::shared_ptr<Model> model;
    std::unique_ptr<EnginePool> pool;
};

httplib::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->stop();
}

}  // namespace

int run_server(const ServerConfig& cfg) {
    if (cfg.max_concurrent < 1 || cfg.max_concurrent > 256 || cfg.workers < 1 || cfg.workers > 64 ||
        cfg.ensemble < 1 || cfg.ensemble > 8 || cfg.max_len < 0 || cfg.head_max_len < 0 || cfg.port < 1 || cfg.port > 65535)
        throw std::runtime_error("invalid server worker, concurrency, ensemble, port or token budget configuration");
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
            {"ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}}.dump().c_str());
        models.push_back(std::move(lm));
    }
    if (models.empty()) {
        std::fprintf(stderr, "statim serve: no model given (-m path.gguf)\n");
        return 2;
    }

    std::vector<std::string> api_keys = cfg.api_keys;
    std::atomic<int> in_flight{0};
    std::atomic<uint64_t> req_total{0}, req_errors{0}, tokens_total{0}, rejected_busy{0};
    std::mutex status_mu;
    std::map<int, uint64_t> status_counts;
    Histogram latency;
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

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool batch, const httplib::ContentReader& reader) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto supplied_id = req.get_header_value("X-Request-Id");
        const std::string rid = valid_request_id(supplied_id) ? supplied_id : new_request_id();
        res.set_header("X-Request-Id", rid);
        std::string model_name = "-";
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
            ojson body = parse_request(raw, cfg.limits);
            validate_request_fields(body);
            if (!body.is_object() || !body.contains("questions"))
                throw HttpError{400, "request body must be an object with a 'questions' field"};
            const ojson& questions = body["questions"];
            DecideOptions opts;
            if (body.contains("lang") && body["lang"].is_string()) opts.lang = body["lang"].get<std::string>();
            opts.ensemble = cfg.ensemble;
            if (body.contains("ensemble")) opts.ensemble = bounded_integer(body["ensemble"], 1, 8, "ensemble");
            opts.calibrate = cfg.calibrate;
            if (body.contains("calibrate") && body["calibrate"].is_boolean()) opts.calibrate = body["calibrate"].get<bool>();
            if (body.contains("return_logits") && body["return_logits"].is_boolean())
                opts.return_logits = body["return_logits"].get<bool>();
            // token budgets, as laya's predict_batch(max_len=, head_max_len=). Many-option choices
            // (e.g. 77 intents) need head_max_len ~512 or every option is cut to one subword.
            if (cfg.max_len > 0) opts.max_len = cfg.max_len;
            if (cfg.head_max_len > 0) opts.head_max_len = cfg.head_max_len;
            for (const char* k : {"max_len", "head_max_len"}) {
                if (!body.contains(k)) continue;
                (std::string(k) == "max_len" ? opts.max_len : opts.head_max_len) = bounded_integer(body[k], 32, 8192, k);
            }
            if (body.contains("ensemble_margin") && body["ensemble_margin"].is_number())
                opts.ensemble_margin = std::clamp(body["ensemble_margin"].get<double>(), 0.0, 1.0);
            std::vector<ojson> states;
            if (batch) {
                if (!body.contains("states") || !body["states"].is_array())
                    throw HttpError{400, "request body must contain a 'states' array"};
                if (body["states"].size() > kMaxBatchStates)
                    throw HttpError{413, "too many states (" + std::to_string(body["states"].size()) + " > " +
                                             std::to_string(kMaxBatchStates) + ")"};
                for (const auto& s : body["states"]) {
                    check_limits(s, questions);
                    states.push_back(s);
                }
            } else {
                ojson state = body.contains("state") ? body["state"] : ojson();
                check_limits(state, questions);
                states.push_back(state);
            }

            // Consensus: both English and multilingual checkpoints answer, their option
            // log-probabilities are averaged. Opt in per request ("model": "consensus") or by default
            // with --consensus.
            LoadedModel* en = by_name("english");
            LoadedModel* ml = by_name("multilingual");
            const bool want_consensus =
                en && ml && ((body.contains("model") && body["model"] == "consensus") || (cfg.consensus && !body.contains("model")));
            std::string reason;
            LoadedModel& lm = want_consensus ? *en : find_model(body, states, reason);
            check_work(questions, states.size(), lm.model->hparams(), opts, cfg.limits, want_consensus ? 2 : 1);
            if (want_consensus) check_work(questions, states.size(), ml->model->hparams(), opts, cfg.limits, 2);
            opts.deadline = t0 + std::chrono::seconds(cfg.inference_timeout);
            model_name = want_consensus ? "consensus" : lm.name;
            const auto ti = std::chrono::steady_clock::now();
            std::vector<ojson> results;
            if (want_consensus) {
                const bool keep_logits = opts.return_logits;
                opts.return_logits = true;
                std::vector<ojson> re, rm;
                {
                    auto lease = en->pool->acquire(opts.deadline);
                    re = lease->engine->decide_batch(states, questions, opts);
                }
                {
                    auto lease = ml->pool->acquire(opts.deadline);
                    rm = lease->engine->decide_batch(states, questions, opts);
                }
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
                reason = "consensus";
            } else {
                auto lease = lm.pool->acquire(opts.deadline);
                results = lease->engine->decide_batch(states, questions, opts);
            }
            const double infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
            for (auto& r : results) {
                n_tokens += r["usage"]["input_tokens"].get<size_t>();
                r["routing"] = {{"model", model_name}, {"reason", reason}, {"engine", "statim"},
                                {"weights", lm.model->hparams().weight_type}};
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
                {"request_id", rid}, {"error", e.what()}}.dump().c_str());
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
                {"request_id", rid}, {"path", req.path}, {"status", status}, {"model", model_name}, {"tokens", n_tokens},
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
        o << "# TYPE statim_in_flight gauge\nstatim_in_flight " << in_flight << "\n";
        o << "# TYPE statim_uptime_seconds gauge\nstatim_uptime_seconds "
          << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << "\n";
        for (auto& m : models) {
            o << "statim_workers_busy{model=\"" << m.name << "\"} " << m.pool->busy() << "\n";
            o << "statim_model_info{model=\"" << m.name << "\",weights=\"" << m.model->hparams().weight_type
              << "\",version=\"" << STATIM_VERSION << "\"} 1\n";
        }
        res.set_content(o.str(), "text/plain; version=0.0.4");
    });

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "info"}, {"event", "listening"},
        {"host", cfg.host}, {"port", cfg.port}, {"auth", !api_keys.empty()}, {"auth_status", api_keys.empty() ? "off" : "on"}}.dump().c_str());
    if (!srv.listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "statim serve: cannot listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    std::fprintf(stderr, "{\"ts\":\"%s\",\"level\":\"info\",\"event\":\"shutdown\"}\n", now_iso8601().c_str());
    g_server = nullptr;
    return 0;
}

}  // namespace statim
