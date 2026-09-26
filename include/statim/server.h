#pragma once

#include <string>
#include <utility>
#include <vector>

#ifndef STATIM_VERSION
#define STATIM_VERSION "0.1.0"
#endif

namespace statim {

struct ServerConfig {
    std::vector<std::pair<std::string, std::string>> models;  // name -> gguf path; first is the default
    std::string host = "127.0.0.1";
    int port = 8080;
    int threads = 0;          // total compute threads (0 = all cores)
    int workers = 1;          // concurrent inference engines sharing the weights
    int max_concurrent = 16;  // requests past auth at once; more get 503
    int ensemble = 1;         // default option-order views per question
    bool calibrate = false;   // default contextual calibration for choice questions
    bool consensus = false;   // default: fuse english + multilingual checkpoints when both are loaded
    std::vector<std::string> api_keys;
    bool access_log = true;
    bool playground = true;   // serve the web playground at "/"
};

int run_server(const ServerConfig& cfg);

}  // namespace statim
