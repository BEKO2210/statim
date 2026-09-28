#pragma once

#include <string>
#include <utility>
#include <vector>

#include "statim/security.h"

#ifndef STATIM_VERSION
#define STATIM_VERSION "0.7.0"
#endif

namespace statim {

struct ServerConfig {
    std::vector<std::pair<std::string, std::string>> models;  // name -> gguf path; first is the default
    std::string host = "127.0.0.1";
    int port = 8080;
    int threads = 0;          // total compute threads (0 = all cores)
    std::string device;       // "cpu", "gpu", "vulkan", "Vulkan0" ... (empty: $STATIM_DEVICE or cpu)
    int workers = 1;          // concurrent inference engines sharing the weights
    int max_concurrent = 16;  // requests past auth at once; more get 503
    int batch_window_ms = 0;  // single-request micro-batch collection window (0 = disabled)
    int max_batch = 16;       // maximum compatible single requests per micro-batch
    int ensemble = 1;         // default option-order views per question
    double min_confidence = 0; // >0: mark answers below this answer_confidence with "escalate": true
    int max_len = 0, head_max_len = 0;  // default token budgets (0 = the checkpoint's)
    bool calibrate = false;   // default contextual calibration for choice questions
    bool consensus = false;   // default: fuse english + multilingual checkpoints when both are loaded
    std::vector<std::string> api_keys;
    SecurityLimits limits;
    int http_queue = 32;
    int request_timeout = 30; // absolute header + body read deadline, seconds
    int inference_timeout = 120; // includes engine queue wait; cooperative compute deadline
    bool access_log = true;
    bool playground = true;   // serve the web playground at "/"
};

int run_server(const ServerConfig& cfg);

}  // namespace statim
