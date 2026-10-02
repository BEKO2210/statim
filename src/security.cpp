#include "statim/security.h"
#include "sha256.h"
#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace statim {
namespace {
constexpr size_t kMaxQuestions = 64;
constexpr size_t kMaxStateChars = 50000;
constexpr size_t kMaxChoiceOptions = 100;
constexpr size_t kMaxScoreLevels = 32;
constexpr size_t kMaxTotalOptions = 512;

size_t utf8_codepoints(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

}
void check_limits(const ojson& state, const ojson& questions) {
    if (state.is_null()) throw HttpError{400, "'state' is required"};
    if (!questions.is_object()) throw HttpError{400, "'questions' must be an object"};
    if (questions.size() > kMaxQuestions)
        throw HttpError{413, "too many questions (" + std::to_string(questions.size()) + " > " + std::to_string(kMaxQuestions) + ")"};
    size_t total = 0;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const ojson& q = it.value();
        if (!q.is_object() || !q.contains("criteria") || !q.contains("type") || !q["type"].is_string()) continue;
        const std::string t = q["type"].get<std::string>();
        const ojson& c = q["criteria"];
        if (t == "choice" && c.is_structured()) {
            total += c.size();
            if (c.size() > kMaxChoiceOptions)
                throw HttpError{413, "too many choice options for '" + it.key() + "' (" + std::to_string(c.size()) + " > " +
                                         std::to_string(kMaxChoiceOptions) + ")"};
        } else if (t == "score" && c.is_array()) {
            total += c.size();
            if (c.size() > kMaxScoreLevels)
                throw HttpError{413, "too many score levels for '" + it.key() + "' (" + std::to_string(c.size()) + " > " +
                                         std::to_string(kMaxScoreLevels) + ")"};
        }
    }
    if (total > kMaxTotalOptions)
        throw HttpError{413, "too many answer options across questions (" + std::to_string(total) + " > " +
                                 std::to_string(kMaxTotalOptions) + ")"};
    size_t n = state.is_string() ? utf8_codepoints(state.get<std::string>()) : utf8_codepoints(py_json_dumps(state));
    if (n > kMaxStateChars)
        throw HttpError{413, "state too large (" + std::to_string(n) + " > " + std::to_string(kMaxStateChars) + " chars)"};
}

namespace {
struct Preflight : nlohmann::json_sax<ojson> {
    const SecurityLimits& limits;
    struct Frame { size_t members = 0; std::unordered_set<std::string> keys; };
    std::vector<Frame> stack;
    size_t nodes = 0;
    explicit Preflight(const SecurityLimits& l) : limits(l) {}
    bool node() {
        if (++nodes > limits.max_json_nodes) throw HttpError(413, "too many JSON nodes");
        return true;
    }
    bool start() {
        node();
        if (stack.size() >= std::min<size_t>(limits.max_json_depth, 128)) throw HttpError(413, "JSON nesting too deep");
        stack.emplace_back();
        return true;
    }
    bool null() override { return node(); }
    bool boolean(bool) override { return node(); }
    bool number_integer(number_integer_t) override { return node(); }
    bool number_unsigned(number_unsigned_t) override { return node(); }
    bool number_float(number_float_t, const string_t&) override { return node(); }
    bool string(string_t&) override { return node(); }
    bool binary(binary_t&) override { return node(); }
    bool start_object(std::size_t) override { return start(); }
    bool start_array(std::size_t) override { return start(); }
    bool end_object() override { stack.pop_back(); return true; }
    bool end_array() override { stack.pop_back(); return true; }
    bool key(string_t& s) override {
        node();
        auto& f = stack.back();
        if (++f.members > limits.max_object_members) throw HttpError(413, "too many JSON object members");
        if (s.size() > 4096) throw HttpError(413, "JSON key exceeds 4096 bytes");
        if (!f.keys.insert(s).second) throw HttpError(400, "duplicate JSON object key");
        return true;
    }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override { return false; }
};
// Unknown fields are ignored, as by laya.serve (clients such as Jev send extra metadata). They cannot
// amplify memory: bodies, JSON depth and members are bounded by the preflight, and the calibration
// cache keys only on the validated question, never on the raw definition.
void fields(const ojson& obj, std::initializer_list<const char*> /*known*/) {
    if (!obj.is_object()) throw HttpError(400, "expected JSON object");
}
size_t rendered_size(const ojson& v) { return v.is_string() ? v.get_ref<const std::string&>().size() : v.dump().size(); }
std::string trim(std::string s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
void add_key(std::vector<std::string>& keys, std::string key, const std::string& source) {
    key = trim(std::move(key));
    if (key.empty()) return;
    validate_api_key(key, source);
    keys.push_back(std::move(key));
}
}
void validate_api_key(const std::string& key, const std::string& source) {
    if (key.size() < 32)
        throw ApiKeyConfigError("API key from " + source + " has length " + std::to_string(key.size()) +
                                "; minimum is 32 characters; generate a key with: openssl rand -hex 32");
    if (key.size() > 4096 || std::any_of(key.begin(), key.end(), [](unsigned char c) { return c <= 32 || c >= 127; }))
        throw ApiKeyConfigError("API key from " + source + " must be 32-4096 printable ASCII characters without whitespace");
}
ojson parse_request(const std::string& text, const SecurityLimits& limits) {
    if (text.size() > max_body_bytes) throw HttpError(413, "request body exceeds 2 MiB");
    Preflight sax(limits);
    if (!ojson::sax_parse(text, &sax)) throw HttpError(400, "request body must be valid JSON");
    return ojson::parse(text);
}
void validate_request_fields(const ojson& body) {
    fields(body, {"state", "states", "questions", "model", "lang", "ensemble", "ensemble_margin", "calibrate",
                  "return_logits", "max_len", "head_max_len", "min_confidence", "adapter"});
    if (!body.contains("questions") || !body["questions"].is_object()) throw HttpError(400, "'questions' must be an object");
    for (const char* name : {"model", "lang"})
        if (body.contains(name) && (!body[name].is_string() || rendered_size(body[name]) > 256))
            throw HttpError(422, std::string(name) + " must be a string of at most 256 bytes");
    if (body.contains("adapter") && !body["adapter"].is_null() &&
        (!body["adapter"].is_string() || rendered_size(body["adapter"]) > 256))
        throw HttpError(422, "adapter must be null or a string of at most 256 bytes");
    const auto& qs = body["questions"];
    if (qs.size() > 64) throw HttpError(413, "too many questions");
    for (auto it = qs.begin(); it != qs.end(); ++it) {
        if (it.key().size() > 256) throw HttpError(413, "question ID exceeds 256 bytes");
        const auto& q = it.value();
        fields(q, {"type", "instructions", "criteria", "labels"});
        if (!q.contains("type") || !q["type"].is_string() ||
            (q["type"] != "choice" && q["type"] != "score" && q["type"] != "noul" && q["type"] != "yes_no"))
            throw HttpError(422, "unknown question type; use choice, score, noul or yes_no");
        if (!q.contains("instructions")) throw HttpError(422, "question requires instructions");
        if (q.contains("instructions") && rendered_size(q["instructions"]) > 16384)
            throw HttpError(413, "instructions exceed 16384 bytes");
        for (const char* name : {"criteria", "labels"}) {
            if (!q.contains(name)) continue;
            const auto& cs = q[name];
            if (cs.is_structured()) for (auto c = cs.begin(); c != cs.end(); ++c) {
                if (cs.is_object() && c.key().size() > 1024) throw HttpError(413, "label exceeds 1024 bytes");
                if (rendered_size(c.value()) > 4096) throw HttpError(413, "criterion or label exceeds 4096 bytes");
            }
        }
    }
}
DecideRequest parse_decide_request(const std::string& raw, bool batch, const RequestDefaults& defaults,
                                   const SecurityLimits& limits) {
    DecideRequest r;
    r.body = parse_request(raw, limits);
    validate_request_fields(r.body);
    for (auto& q : r.body["questions"])
        if (q["type"] == "yes_no") q["type"] = "noul";
    const ojson& body = r.body;
    if (!body.is_object() || !body.contains("questions"))
        throw HttpError{400, "request body must be an object with a 'questions' field"};
    const ojson& questions = body["questions"];
    DecideOptions& opts = r.opts;
    if (body.contains("lang") && body["lang"].is_string()) opts.lang = body["lang"].get<std::string>();
    opts.ensemble = defaults.ensemble;
    if (body.contains("ensemble")) opts.ensemble = bounded_integer(body["ensemble"], 1, 8, "ensemble");
    opts.calibrate = defaults.calibrate;
    if (body.contains("calibrate") && body["calibrate"].is_boolean()) opts.calibrate = body["calibrate"].get<bool>();
    if (body.contains("return_logits") && body["return_logits"].is_boolean())
        opts.return_logits = body["return_logits"].get<bool>();
    // token budgets, as laya's predict_batch(max_len=, head_max_len=). Many-option choices
    // (e.g. 77 intents) need head_max_len ~512 or every option is cut to one subword.
    if (defaults.max_len > 0) opts.max_len = defaults.max_len;
    if (defaults.head_max_len > 0) opts.head_max_len = defaults.head_max_len;
    for (const char* k : {"max_len", "head_max_len"}) {
        if (!body.contains(k)) continue;
        (std::string(k) == "max_len" ? opts.max_len : opts.head_max_len) = bounded_integer(body[k], 32, 8192, k);
    }
    if (body.contains("ensemble_margin") && body["ensemble_margin"].is_number())
        opts.ensemble_margin = std::clamp(body["ensemble_margin"].get<double>(), 0.0, 1.0);
    // Selective prediction: answers below the threshold get "escalate": true so the caller can hand
    // them to a person or a larger model; without a threshold the response is unchanged.
    r.min_confidence = defaults.min_confidence;
    if (body.contains("min_confidence")) {
        const auto& v = body["min_confidence"];
        if (!v.is_number() || v.get<double>() < 0.0 || v.get<double>() > 1.0)
            throw HttpError{422, "min_confidence must be a number from 0 to 1"};
        r.min_confidence = v.get<double>();
    }
    if (batch) {
        if (!body.contains("states") || !body["states"].is_array())
            throw HttpError{400, "request body must contain a 'states' array"};
        if (body["states"].size() > max_batch_states)
            throw HttpError{413, "too many states (" + std::to_string(body["states"].size()) + " > " +
                                     std::to_string(max_batch_states) + ")"};
        for (const auto& s : body["states"]) {
            check_limits(s, questions);
            r.states.push_back(s);
        }
    } else {
        ojson state = body.contains("state") ? body["state"] : ojson();
        check_limits(state, questions);
        r.states.push_back(std::move(state));
    }
    return r;
}
int bounded_integer(const ojson& v, int low, int high, const std::string& name) {
    bool ok = v.is_number_integer();
    if (ok && v.is_number_unsigned()) ok = v.get<uint64_t>() >= static_cast<uint64_t>(low) && v.get<uint64_t>() <= static_cast<uint64_t>(high);
    else if (ok) ok = v.get<int64_t>() >= low && v.get<int64_t>() <= high;
    if (!ok) throw HttpError(422, name + " must be an integer between " + std::to_string(low) + " and " + std::to_string(high));
    return v.get<int>();
}
int effective_max_len(const HParams& h, const DecideOptions& opts) {
    int head = bounded_integer(opts.head_max_len.value_or(h.head_max_len), 32, 8192, "head_max_len");
    int len = bounded_integer(opts.max_len.value_or(h.max_len), 32, 8192, "max_len");
    // Preserve Laya's automatic state room while rejecting capacity overflow.
    len = std::max(len, head + 128);
    if (len > h.max_position) throw HttpError(422, "effective max_len exceeds model capacity (head_max_len needs 128 state tokens)");
    return len;
}
void check_work(const ojson& qs, size_t states, const HParams& h, const DecideOptions& opts,
                const SecurityLimits& limits, size_t models) {
    if (!qs.is_object() || qs.size() > 64 || states > 256 || models < 1 || models > 2)
        throw HttpError(413, "request exceeds state/question/model limits");
    bounded_integer(opts.ensemble, 1, 8, "ensemble");
    const size_t len = effective_max_len(h, opts);
    size_t views = 0;
    for (const auto& q : qs) {
        bool choice = q.value("type", std::string()) == "choice";
        size_t options = q.contains("criteria") ? q["criteria"].size() : 2;
        views += choice ? std::min<size_t>(std::max(1, opts.ensemble), std::max<size_t>(1, options)) : 1;
        if (opts.calibrate && choice) views += 3; // assume every shape misses cache
    }
    // Inputs have already been bounded (256 states, 64 questions, 8 views, 2 models).
    size_t work = states * views * models;
    if (work > limits.max_request_work) throw HttpError(413, "request exceeds state/question/view work limit");
    if (work > limits.max_request_tokens / len) throw HttpError(413, "request exceeds aggregate token budget");
    // Bound rows * actual_length^2 for every packed graph. Using floor(8192 / len)
    // would underestimate graphs of slightly shorter rows that fit an extra row.
    const size_t padded_tokens = std::min<size_t>(32 * len, 8192);
    size_t attention = padded_tokens * len * std::max(1, std::max(h.n_head, h.head_n_head)) * sizeof(float) * 2;
    if (work && attention > limits.max_attention_bytes) throw HttpError(413, "request exceeds attention memory budget");
    size_t response_per_state = 1024;
    for (auto it = qs.begin(); it != qs.end(); ++it)
        response_per_state += 4096 + 8 * (it.key().size() + it.value().dump().size());
    if (states && response_per_state > limits.max_response_bytes / states)
        throw HttpError(413, "request exceeds response byte budget");
}
std::vector<std::string> load_key_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ApiKeyConfigError("cannot open configured API key file: " + path);
    std::vector<std::string> keys;
    std::string line;
    size_t line_number = 0;
    while (std::getline(f, line)) {
        ++line_number;
        line = trim(std::move(line));
        if (!line.empty() && line.front() != '#')
            add_key(keys, std::move(line), path + ":" + std::to_string(line_number));
    }
    if (f.bad() || keys.empty()) throw ApiKeyConfigError("configured API key file contains no valid keys or cannot be read: " + path);
    return keys;
}
std::vector<std::string> load_key_env(const std::string& value) {
    std::vector<std::string> keys;
    std::istringstream in(value);
    std::string key;
    while (std::getline(in, key, ',')) add_key(keys, std::move(key), "STATIM_API_KEY");
    if (keys.empty()) throw ApiKeyConfigError("configured STATIM_API_KEY contains no valid keys");
    return keys;
}
bool is_loopback_host(const std::string& host) {
    if (host == "localhost" || host == "::1") return true;
    size_t begin = 0;
    int octets[4]{};
    for (int i = 0; i < 4; ++i) {
        const size_t end = host.find('.', begin);
        if ((i < 3 && end == std::string::npos) || (i == 3 && end != std::string::npos)) return false;
        const size_t stop = end == std::string::npos ? host.size() : end;
        if (stop == begin || stop - begin > 3) return false;
        int value = 0;
        for (size_t j = begin; j < stop; ++j) {
            const unsigned char c = static_cast<unsigned char>(host[j]);
            if (c < '0' || c > '9') return false;
            value = value * 10 + (c - '0');
        }
        if (value > 255) return false;
        octets[i] = value;
        begin = stop + 1;
    }
    return octets[0] == 127;
}

namespace {
bool valid_ipv4(const std::string& host) {
    size_t begin = 0;
    for (int i = 0; i < 4; ++i) {
        const size_t end = host.find('.', begin);
        if ((i < 3 && end == std::string::npos) || (i == 3 && end != std::string::npos)) return false;
        const size_t stop = end == std::string::npos ? host.size() : end;
        if (stop == begin || stop - begin > 3) return false;
        int value = 0;
        for (size_t j = begin; j < stop; ++j) {
            if (host[j] < '0' || host[j] > '9') return false;
            value = value * 10 + host[j] - '0';
        }
        if (value > 255 || (stop - begin > 1 && host[begin] == '0')) return false;
        begin = stop + 1;
    }
    return true;
}

bool valid_dns_name(const std::string& host) {
    if (host.empty() || host.size() > 253 || host.front() == '.' || host.back() == '.') return false;
    size_t begin = 0;
    while (begin < host.size()) {
        const size_t end = host.find('.', begin);
        const size_t stop = end == std::string::npos ? host.size() : end;
        if (stop == begin || stop - begin > 63 || host[begin] == '-' || host[stop - 1] == '-') return false;
        for (size_t i = begin; i < stop; ++i) {
            const unsigned char c = static_cast<unsigned char>(host[i]);
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) && c != '-') return false;
        }
        begin = stop + 1;
    }
    return true;
}

bool ipv6_groups(const std::string& part, bool allow_ipv4, size_t& groups) {
    if (part.empty()) return true;
    if (part.front() == ':' || part.back() == ':') return false;
    size_t begin = 0;
    while (begin < part.size()) {
        const size_t end = part.find(':', begin);
        const size_t stop = end == std::string::npos ? part.size() : end;
        if (stop == begin) return false;
        const std::string group = part.substr(begin, stop - begin);
        if (group.find('.') != std::string::npos) {
            if (!allow_ipv4 || end != std::string::npos || !valid_ipv4(group)) return false;
            groups += 2;
        } else {
            if (group.size() > 4) return false;
            for (unsigned char c : group)
                if (!std::isxdigit(c)) return false;
            ++groups;
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return true;
}

bool valid_ipv6(const std::string& host) {
    if (host.empty() || host.find(':') == std::string::npos) return false;
    const size_t compression = host.find("::");
    if (compression != std::string::npos && host.find("::", compression + 2) != std::string::npos) return false;
    size_t groups = 0;
    if (compression == std::string::npos)
        return ipv6_groups(host, true, groups) && groups == 8;
    const std::string left = host.substr(0, compression);
    const std::string right = host.substr(compression + 2);
    return ipv6_groups(left, right.empty(), groups) && ipv6_groups(right, true, groups) && groups < 8;
}

bool valid_origin(const std::string& origin) {
    const size_t scheme_end = origin.find("://");
    if (scheme_end == std::string::npos) return false;
    const std::string scheme = origin.substr(0, scheme_end);
    if (scheme != "http" && scheme != "https") return false;
    const std::string authority = origin.substr(scheme_end + 3);
    if (authority.empty() || authority.find_first_of("/?#@*'\";,\\") != std::string::npos) return false;

    std::string host;
    std::string port;
    if (authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close == std::string::npos) return false;
        host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') return false;
            port = authority.substr(close + 2);
        }
        if (!valid_ipv6(host)) return false;
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            if (authority.find(':') != colon) return false;
            host = authority.substr(0, colon);
            port = authority.substr(colon + 1);
        } else {
            host = authority;
        }
        const bool numeric_dotted = !host.empty() &&
            std::all_of(host.begin(), host.end(), [](unsigned char c) { return (c >= '0' && c <= '9') || c == '.'; });
        if ((numeric_dotted && !valid_ipv4(host)) || (!numeric_dotted && !valid_dns_name(host))) return false;
    }
    if (!port.empty()) {
        if (port.size() > 5) return false;
        unsigned value = 0;
        for (unsigned char c : port) {
            if (c < '0' || c > '9') return false;
            value = value * 10 + c - '0';
        }
        if (value == 0 || value > 65535) return false;
    } else if (!authority.empty() && authority.back() == ':') {
        return false;
    }
    return true;
}
}  // namespace

std::vector<std::string> parse_frame_ancestors(const std::string& value) {
    if (value.empty()) throw FrameAncestorsConfigError("--frame-ancestors requires at least one origin");
    for (unsigned char c : value)
        if (c < 0x20 || c == 0x7f)
            throw FrameAncestorsConfigError("--frame-ancestors contains control characters");

    std::vector<std::string> origins;
    std::unordered_set<std::string> seen;
    size_t begin = 0;
    while (begin < value.size()) {
        while (begin < value.size() && value[begin] == ' ') ++begin;
        if (begin == value.size()) break;
        if (value[begin] == ',') throw FrameAncestorsConfigError("--frame-ancestors contains an empty origin");
        const size_t end = value.find_first_of(" ,", begin);
        const std::string origin = value.substr(begin, end - begin);
        if (origin.size() > 256)
            throw FrameAncestorsConfigError("--frame-ancestors origin exceeds 256 bytes");
        if (!valid_origin(origin))
            throw FrameAncestorsConfigError("invalid --frame-ancestors origin '" + origin +
                                            "' (expected http://host[:port] or https://host[:port])");
        if (seen.insert(origin).second) {
            origins.push_back(origin);
            if (origins.size() > 8) throw FrameAncestorsConfigError("--frame-ancestors accepts at most 8 origins");
        }
        if (end == std::string::npos) break;
        begin = end;
        while (begin < value.size() && value[begin] == ' ') ++begin;
        if (begin < value.size() && value[begin] == ',') {
            ++begin;
            while (begin < value.size() && value[begin] == ' ') ++begin;
            if (begin == value.size() || value[begin] == ',')
                throw FrameAncestorsConfigError("--frame-ancestors contains an empty origin");
        }
    }
    if (origins.empty()) throw FrameAncestorsConfigError("--frame-ancestors requires at least one origin");
    return origins;
}
bool valid_request_id(const std::string& v) {
    return !v.empty() && v.size() <= 128 && std::all_of(v.begin(), v.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    });
}
std::string now_iso8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}
bool CalibrationCache::get(const std::string& key, std::vector<double>& value) {
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    entries_.splice(entries_.begin(), entries_, it->second);
    value = it->second->value;
    return true;
}
void CalibrationCache::put(std::string key, const std::vector<double>& value) {
    // Account for the map's key copy and conservative per-entry bookkeeping.
    const size_t cost = key.capacity() + key.size() + value.size() * sizeof(double) + 256;
    if (cost > max_bytes_ || !max_entries_) return;
    auto old = index_.find(key);
    if (old != index_.end()) {
        bytes_ -= old->second->bytes;
        entries_.erase(old->second);
        index_.erase(old);
    }
    while (!entries_.empty() && (bytes_ > max_bytes_ - cost || entries_.size() >= max_entries_)) {
        auto& e = entries_.back();
        bytes_ -= e.bytes;
        index_.erase(e.key);
        entries_.pop_back();
    }
    entries_.push_front({std::move(key), value, cost});
    try { index_.emplace(entries_.front().key, entries_.begin()); }
    catch (...) { entries_.pop_front(); throw; }
    bytes_ += cost;
}

namespace {
std::string base64(const uint8_t* p, size_t n) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t(p[i]) << 16) | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out += t[(v >> 18) & 63];
        out += t[(v >> 12) & 63];
        out += i + 1 < n ? t[(v >> 6) & 63] : '=';
        out += i + 2 < n ? t[v & 63] : '=';
    }
    return out;
}

// 'sha256-...' of the text between the only <tag> and its </tag>, as CSP hashes it (exact bytes).
std::string inline_hash(const std::string& html, const std::string& tag) {
    const std::string open = "<" + tag + ">", close = "</" + tag + ">";
    const size_t a = html.find(open);
    if (a == std::string::npos || html.find(open, a + 1) != std::string::npos)
        throw std::runtime_error("playground: expected exactly one inline <" + tag + ">");
    const size_t b = html.find(close, a);
    if (b == std::string::npos) throw std::runtime_error("playground: unterminated <" + tag + ">");
    Sha256 h;
    h.update(html.data() + a + open.size(), b - a - open.size());
    const auto d = h.digest();
    return "'sha256-" + base64(d.data(), d.size()) + "'";
}
}  // namespace

std::string playground_csp(const std::string& html, const std::vector<std::string>& frame_ancestors) {
    // style-src-attr allows the inline style="--v:..." attributes the page renders for the
    // probability bars; attributes cannot run code, while scripts and style blocks stay hash-pinned.
    if (frame_ancestors.size() > 8)
        throw FrameAncestorsConfigError("playground frame ancestors exceed the limit of 8 origins");
    for (const auto& origin : frame_ancestors)
        if (origin.size() > 256 || !valid_origin(origin))
            throw FrameAncestorsConfigError("invalid playground frame ancestor");
    std::string ancestors = frame_ancestors.empty() ? "'none'" : frame_ancestors.front();
    for (size_t i = 1; i < frame_ancestors.size(); ++i) ancestors += " " + frame_ancestors[i];
    return "default-src 'none'; script-src " + inline_hash(html, "script") + "; style-src " + inline_hash(html, "style") +
           "; style-src-attr 'unsafe-inline'; font-src data:; img-src data:; connect-src 'self'; base-uri 'none'; "
           "form-action 'none'; frame-ancestors " + ancestors;
}

} // namespace statim
