// statim — command line entry point.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "statim/engine.h"
#include "statim/server.h"

using statim::ojson;

namespace {

void usage() {
    std::fprintf(stderr,
                 "statim %s — local System-1 decision engine (Laya/Jev compatible)\n\n"
                 "usage:\n"
                 "  statim serve   -m [name=]model.gguf [-m ...] [--host 127.0.0.1] [--port 8080]\n"
                 "                 [--threads N] [--workers W] [--max-concurrent 16] [--ensemble K]\n"
                 "                 [--api-key-file FILE] [--no-access-log]\n"
                 "  statim decide  -m model.gguf [--ensemble K] [--lang xx] < request.json\n"
                 "                 (request: {\"state\": ..., \"questions\": {...}})\n"
                 "  statim bench   -m model.gguf [--threads N] [--runs 5] < request.json\n"
                 "  statim info    -m model.gguf\n"
                 "  statim version\n\n"
                 "env: STATIM_API_KEY (comma-separated keys), STATIM_LOG=debug\n",
                 STATIM_VERSION);
}

std::string read_all(std::istream& in) {
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string model_name_from_path(const std::string& p) {
    std::string base = p.substr(p.find_last_of('/') + 1);
    auto dot = base.rfind(".gguf");
    if (dot != std::string::npos) base = base.substr(0, dot);
    for (const char* pre : {"laya-"}) {
        if (base.rfind(pre, 0) == 0) base = base.substr(std::strlen(pre));
    }
    for (const char* suf : {"-f32", "-f16", "-q8_0", "-q5_K", "-q4_K", "-q4_0"}) {
        size_t n = std::strlen(suf);
        if (base.size() > n && base.compare(base.size() - n, n, suf) == 0) base = base.substr(0, base.size() - n);
    }
    return base;
}

void quiet_ggml_log() {
    if (std::getenv("STATIM_LOG") && std::string(std::getenv("STATIM_LOG")) == "debug") return;
    ggml_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
    }, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    quiet_ggml_log();
    const std::string cmd = argv[1];
    if (cmd == "version" || cmd == "--version") {
        std::printf("statim %s\n", STATIM_VERSION);
        return 0;
    }
    statim::ServerConfig cfg;
    statim::DecideOptions dopts;
    int runs = 5;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            std::string v = next();
            auto eq = v.find('=');
            if (eq != std::string::npos && v.find('/') > eq) cfg.models.emplace_back(v.substr(0, eq), v.substr(eq + 1));
            else cfg.models.emplace_back(model_name_from_path(v), v);
        } else if (a == "--host") cfg.host = next();
        else if (a == "--port") cfg.port = std::atoi(next().c_str());
        else if (a == "--threads" || a == "-t") cfg.threads = std::atoi(next().c_str());
        else if (a == "--workers") cfg.workers = std::atoi(next().c_str());
        else if (a == "--max-concurrent") cfg.max_concurrent = std::atoi(next().c_str());
        else if (a == "--ensemble") cfg.ensemble = dopts.ensemble = std::atoi(next().c_str());
        else if (a == "--lang") dopts.lang = next();
        else if (a == "--runs") runs = std::atoi(next().c_str());
        else if (a == "--no-access-log") cfg.access_log = false;
        else if (a == "--api-key-file") {
            std::ifstream f(next());
            std::string line;
            while (std::getline(f, line))
                if (!line.empty() && line[0] != '#') cfg.api_keys.push_back(line);
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (const char* env = std::getenv("STATIM_API_KEY")) {
        std::stringstream ss(env);
        std::string k;
        while (std::getline(ss, k, ','))
            if (!k.empty()) cfg.api_keys.push_back(k);
    }
    if (cfg.models.empty()) {
        usage();
        return 2;
    }
    try {
        if (cmd == "serve") return statim::run_server(cfg);

        auto model = statim::Model::load(cfg.models.front().second);
        if (cmd == "info") {
            const auto& h = model->hparams();
            ojson info = {{"name", h.name}, {"weights", h.weight_type}, {"weight_bytes", model->weight_bytes()},
                          {"encoder", {{"layers", h.n_layer}, {"hidden", h.n_embd}, {"heads", h.n_head}, {"ff", h.n_ff},
                                       {"local_window", h.local_window}, {"rope_theta_global", h.rope_theta_global},
                                       {"rope_theta_local", h.rope_theta_local}}},
                          {"head", {{"layers", h.head_n_layer}, {"heads", h.head_n_head}, {"ff", h.head_n_ff}}},
                          {"max_len", h.max_len}, {"head_max_len", h.head_max_len}, {"vocab", model->tokenizer().vocab_size()}};
            std::cout << info.dump(2) << "\n";
            return 0;
        }
        statim::RunOptions ro;
        ro.n_threads = cfg.threads;
        statim::Engine engine(model, ro);
        ojson req = ojson::parse(read_all(std::cin));
        const ojson state = req.contains("state") ? req["state"] : ojson();
        const ojson questions = req.contains("questions") ? req["questions"] : ojson::object();
        if (cmd == "decide") {
            std::cout << engine.decide(state, questions, dopts).dump(2) << "\n";
            return 0;
        }
        if (cmd == "bench" && req.contains("states")) {
            // per-state latency over a list of states (same protocol as bench/bench_laya_python.py)
            const ojson& states = req["states"];
            engine.decide(states[0], questions, dopts);  // warm-up
            std::vector<double> ms;
            for (const auto& st : states) {
                auto t0 = std::chrono::steady_clock::now();
                engine.decide(st, questions, dopts);
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            ojson out = {{"engine", "statim"}, {"weights", model->hparams().weight_type}, {"threads", cfg.threads},
                         {"states", ms.size()}};
            double sum = 0;
            for (double v : ms) sum += v;
            out["mean_ms"] = sum / ms.size();
            std::vector<double> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            out["p50_ms"] = sorted[sorted.size() / 2];
            ojson per = ojson::array();
            for (double v : ms) per.push_back(std::round(v * 10) / 10);
            out["per_state_ms"] = per;
            std::cout << out.dump() << "\n";
            return 0;
        }
        if (cmd == "bench") {
            engine.decide(state, questions, dopts);  // warm-up
            std::vector<double> ms;
            for (int r = 0; r < runs; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                engine.decide(state, questions, dopts);
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            double sum = 0;
            for (double v : ms) sum += v;
            std::printf("{\"runs\":%d,\"mean_ms\":%.2f,\"p50_ms\":%.2f,\"min_ms\":%.2f,\"max_ms\":%.2f}\n", runs, sum / runs,
                        ms[ms.size() / 2], ms.front(), ms.back());
            return 0;
        }
        usage();
        return 2;
    } catch (const statim::QuestionError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
