#include "statim/security.h"
#include <algorithm>
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
void add_key(std::vector<std::string>& keys, std::string key) {
    key = trim(std::move(key));
    if (key.empty()) return;
    if (key.size() > 4096 || std::any_of(key.begin(), key.end(), [](unsigned char c) { return c <= 32 || c >= 127; }))
        throw std::runtime_error("API key must be 1-4096 printable ASCII characters without whitespace");
    keys.push_back(std::move(key));
}
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
            (q["type"] != "choice" && q["type"] != "score" && q["type"] != "noul"))
            throw HttpError(422, "unknown question type; use choice, score or noul");
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
    const ojson& body = r.body;
    validate_request_fields(body);
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
    if (!f) throw std::runtime_error("cannot open configured API key file: " + path);
    std::vector<std::string> keys;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(std::move(line));
        if (!line.empty() && line.front() != '#') add_key(keys, std::move(line));
    }
    if (f.bad() || keys.empty()) throw std::runtime_error("configured API key file contains no valid keys or cannot be read: " + path);
    return keys;
}
std::vector<std::string> load_key_env(const std::string& value) {
    std::vector<std::string> keys;
    std::istringstream in(value);
    std::string key;
    while (std::getline(in, key, ',')) add_key(keys, std::move(key));
    if (keys.empty()) throw std::runtime_error("configured STATIM_API_KEY contains no valid keys");
    return keys;
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
} // namespace statim
