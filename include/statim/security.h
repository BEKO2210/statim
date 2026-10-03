#pragma once

#include "statim/engine.h"
#include <chrono>
#include <list>
#include <unordered_map>

namespace statim {
struct HttpError : std::runtime_error {
    int status;
    HttpError(int status, const std::string& detail) : std::runtime_error(detail), status(status) {}
};
struct ApiKeyConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct FrameAncestorsConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
enum class ApiScope : uint8_t {
    inference = 1,
    metrics = 2,
};
constexpr uint8_t all_api_scopes = static_cast<uint8_t>(ApiScope::inference) |
                                   static_cast<uint8_t>(ApiScope::metrics);
struct ApiKey {
    std::string key;
    std::string id;
    uint8_t scopes = all_api_scopes;
    ApiKey() = default;
    ApiKey(std::string value, uint8_t scope_mask = all_api_scopes);
    bool operator==(const ApiKey& other) const {
        return key == other.key && id == other.id && scopes == other.scopes;
    }
};
bool key_has_scope(const ApiKey& key, ApiScope scope);
struct SecurityLimits {
    size_t max_json_depth = 64;
    size_t max_json_nodes = 100000;
    size_t max_object_members = 1024;
    size_t max_request_work = 4096;
    size_t max_request_tokens = 1048576;
    size_t max_attention_bytes = 1024ULL * 1024 * 1024;
    size_t max_response_bytes = 16 * 1024 * 1024;
};
constexpr size_t max_body_bytes = 2 * 1024 * 1024;
constexpr size_t max_batch_states = 256;
ojson parse_request(const std::string& text, const SecurityLimits& limits = {});
void validate_request_fields(const ojson& body);
void check_limits(const ojson& state, const ojson& questions);
int bounded_integer(const ojson& value, int low, int high, const std::string& name);
int effective_max_len(const HParams& h, const DecideOptions& opts);
void check_work(const ojson& questions, size_t states, const HParams& h, const DecideOptions& opts,
                const SecurityLimits& limits, size_t models = 1);
// Server-side defaults a request may override (ServerConfig flags).
struct RequestDefaults {
    int ensemble = 1;
    bool calibrate = false;
    int max_len = 0, head_max_len = 0;  // 0 = the checkpoint's
    double min_confidence = 0;
};
// A parsed, validated POST /v1/systemone (batch=false) or /v1/systemone/batch body.
struct DecideRequest {
    ojson body;
    std::vector<ojson> states;
    DecideOptions opts;
    double min_confidence = 0;
};
// Everything the HTTP handler does with the raw body before a model is chosen. Throws HttpError
// (400/413/422) or QuestionError (422) for caller mistakes; anything else is a server bug.
DecideRequest parse_decide_request(const std::string& raw, bool batch, const RequestDefaults& defaults = {},
                                   const SecurityLimits& limits = {});
std::vector<ApiKey> load_key_file(const std::string& path);
std::vector<ApiKey> load_key_env(const std::string& value);
void validate_api_key(const std::string& key, const std::string& source);
bool is_loopback_host(const std::string& host);
std::vector<std::string> parse_frame_ancestors(const std::string& value);
bool valid_request_id(const std::string& value);
std::string now_iso8601();
// Content-Security-Policy for the built-in playground page: its one inline <script> and one inline
// <style> block are allowed by SHA-256 hash, everything else stays closed (no 'unsafe-inline' for
// scripts, no foreign origins). Framing is denied unless explicit origins are supplied. Throws if
// the page has not exactly one inline script and style block.
std::string playground_csp(const std::string& html, const std::vector<std::string>& frame_ancestors = {});

// Exact semantic keys avoid hash-collision changes to inference. Both retained key
// bytes and values are charged; an entry ceiling also bounds allocator overhead.
class CalibrationCache {
public:
    explicit CalibrationCache(size_t max_bytes = 4 * 1024 * 1024, size_t max_entries = 4096)
        : max_bytes_(max_bytes), max_entries_(max_entries) {}
    bool get(const std::string& key, std::vector<double>& value);
    void put(std::string key, const std::vector<double>& value);
    size_t bytes() const { return bytes_; }
    size_t size() const { return entries_.size(); }
private:
    struct Entry { std::string key; std::vector<double> value; size_t bytes; };
    std::list<Entry> entries_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    size_t bytes_ = 0, max_bytes_, max_entries_;
};
} // namespace statim
