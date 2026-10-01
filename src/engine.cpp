#include "statim/engine.h"
#include "statim/security.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace statim {

namespace {

constexpr const char* kQTypes[] = {"choice", "score", "noul"};
constexpr float kTempMin = 0.5f, kTempMax = 5.0f;

int qtype_of(const std::string& t) {
    for (int i = 0; i < 3; ++i)
        if (t == kQTypes[i]) return i;
    return -1;
}

void append_utf8_escaped(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", ch);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
        }
    }
    out.push_back('"');
}

// Python repr() of a float: shortest round-trip digits, always with a '.' or exponent.
std::string py_float(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    char buf[32];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    std::string s(buf);
    // Python switches to exponent notation for exp < -4 or exp >= 16 and formats it as 1e-05 / 1e+16.
    int exp10 = v == 0 ? 0 : static_cast<int>(std::floor(std::log10(std::fabs(v))));
    if (exp10 >= 16 || exp10 < -4) {
        char e[40];
        std::string mant = s;
        auto epos = mant.find('e');
        std::string digits;
        if (epos != std::string::npos) digits = mant.substr(0, epos);
        else {
            // re-render in exponent form with the same number of significant digits
            std::snprintf(e, sizeof e, "%.16e", v);
            for (int prec = 0; prec <= 16; ++prec) {
                std::snprintf(e, sizeof e, "%.*e", prec, v);
                if (std::strtod(e, nullptr) == v) break;
            }
            std::string t(e);
            epos = t.find('e');
            digits = t.substr(0, epos);
            mant = t;
        }
        epos = mant.find('e');
        int ex = std::atoi(mant.c_str() + epos + 1);
        std::snprintf(e, sizeof e, "%se%c%02d", digits.c_str(), ex < 0 ? '-' : '+', std::abs(ex));
        return e;
    }
    if (s.find_first_of(".e") == std::string::npos) s += ".0";
    return s;
}

void dump(std::string& out, const ojson& v, const char* is, const char* ks) {
    switch (v.type()) {
        case ojson::value_t::null: out += "null"; break;
        case ojson::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
        case ojson::value_t::number_integer: out += std::to_string(v.get<int64_t>()); break;
        case ojson::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
        case ojson::value_t::number_float: out += py_float(v.get<double>()); break;
        case ojson::value_t::string: append_utf8_escaped(out, v.get_ref<const std::string&>()); break;
        case ojson::value_t::array: {
            out.push_back('[');
            bool first = true;
            for (const auto& e : v) {
                if (!first) out += is;
                first = false;
                dump(out, e, is, ks);
            }
            out.push_back(']');
            break;
        }
        case ojson::value_t::object: {
            out.push_back('{');
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += is;
                first = false;
                append_utf8_escaped(out, it.key());
                out += ks;
                dump(out, it.value(), is, ks);
            }
            out.push_back('}');
            break;
        }
        default: out += "null";
    }
}

// Python str() of a scalar label (choice labels given as a list may be numbers, bools, None).
std::string py_str(const ojson& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_null()) return "None";
    if (v.is_number_float()) return py_float(v.get<double>());
    return v.dump();
}

// JSON object key Python's json module would emit for a dict key.
std::string py_key(const ojson& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_null()) return "null";
    if (v.is_number_float()) return py_float(v.get<double>());
    return v.dump();
}

std::string render_criterion(const ojson& v) {
    return v.is_string() ? v.get<std::string>() : py_json_dumps(v, ", ", ": ");
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

double round4(double x) { return std::nearbyint(x * 10000.0) / 10000.0; }

float clamp_temperature(double t) {
    if (!std::isfinite(t)) return 1.0f;
    return static_cast<float>(std::min<double>(kTempMax, std::max<double>(kTempMin, t)));
}

std::string temp_bucket(int qt, size_t k) {
    const char* size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    return std::string(kQTypes[qt]) + ":" + size;
}

}  // namespace

// Internal, validated form of one question (mirrors laya.Agent._to_internal).
struct Question {
    std::string id;
    int qt = 0;
    std::string ins;
    std::vector<ojson> labels;       // choice: original label values, in order
    std::vector<ojson> levels;       // score: original level values
    std::vector<std::string> options;  // rendered option texts, label-index order
};

namespace {


void check_question(const std::string& qid, const ojson& q) {
    auto err = [&](const std::string& m) { throw QuestionError("question '" + qid + "': " + m); };
    if (!q.is_object()) err("definition must be an object");
    if (!q.contains("type") || !q["type"].is_string() || qtype_of(q["type"].get<std::string>()) < 0)
        err("unknown type; use one of ['choice', 'noul', 'score']");
    const std::string t = q["type"].get<std::string>();
    if (!q.contains("instructions")) err("no 'instructions'; add the text the model should answer");
    const ojson crit = q.contains("criteria") ? q["criteria"] : ojson();
    if (t == "choice") {
        if (!crit.is_object() && !crit.is_array())
            err("a choice question takes 'criteria' as a dict of label -> description, or a list of labels");
        if (crit.empty()) err("a choice question needs at least one criterion");
        if (crit.is_array())
            for (size_t i = 0; i < crit.size(); ++i)
                if (crit[i].is_structured())
                    err("choice label " + std::to_string(i) + " must be a scalar (a string, number or null)");
    } else if (t == "score") {
        if (!crit.is_array()) err("a score question takes 'criteria' as a list of level descriptions, index 0 first");
        if (crit.empty()) err("a score question needs at least one level");
        for (size_t i = 0; i < crit.size(); ++i)
            if (crit[i].is_null()) err("score level " + std::to_string(i) + " is null; give every level a description");
    } else if (!crit.is_null()) {
        if (!crit.is_object()) err("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it");
        for (auto it = crit.begin(); it != crit.end(); ++it) {
            std::string k = it.key();
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            if (k != "true" && k != "false") err("a noul question takes 'criteria' keyed only 'true'/'false'");
        }
    }
    if (q.contains("labels")) {
        if (t != "noul") err("'labels' is only supported for noul questions");
    }
}

std::pair<std::string, std::string> noul_labels(const std::string& qid, const ojson& q) {
    if (!q.contains("labels")) return {"false", "true"};
    const ojson& l = q["labels"];
    auto bad = [&] {
        throw QuestionError("question '" + qid + "': noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
    };
    if (!l.is_object() || l.size() != 2 || !l.contains("false") || !l.contains("true")) bad();
    if (!l["false"].is_string() || !l["true"].is_string()) bad();
    auto strip = [](std::string s) {
        const char* ws = " \t\n\r\f\v";
        s.erase(0, s.find_first_not_of(ws));
        s.erase(s.find_last_not_of(ws) + 1);
        return s;
    };
    std::string f = strip(l["false"].get<std::string>()), t = strip(l["true"].get<std::string>());
    if (f.empty() || t.empty() || f == t) bad();
    return {f, t};
}

Question to_internal(const std::string& qid, const ojson& q) {
    check_question(qid, q);
    Question out;
    out.id = qid;
    out.qt = qtype_of(q["type"].get<std::string>());
    out.ins = q["instructions"].is_string() ? q["instructions"].get<std::string>() : py_json_dumps(q["instructions"]);
    const ojson crit = q.contains("criteria") ? q["criteria"] : ojson();
    if (out.qt == 0) {
        if (crit.is_array()) {
            for (const auto& c : crit) {
                // {c: None for c in crit}: duplicate labels collapse, first position wins
                bool dup = false;
                for (const auto& l : out.labels) dup |= (l == c);
                if (dup) continue;
                out.labels.push_back(c);
                out.options.push_back(py_str(c));
            }
        } else {
            for (auto it = crit.begin(); it != crit.end(); ++it) {
                out.labels.emplace_back(it.key());
                const ojson& v = it.value();
                bool plain = v.is_null() || (v.is_string() && v.get<std::string>().empty());
                out.options.push_back(plain ? it.key() : it.key() + ": " + render_criterion(v));
            }
        }
    } else if (out.qt == 1) {
        for (size_t i = 0; i < crit.size(); ++i) {
            out.levels.push_back(crit[i]);
            out.options.push_back("level " + std::to_string(i) + ": " + render_criterion(crit[i]));
        }
    } else {
        auto [fl, tl] = noul_labels(qid, q);
        ojson fc, tc;
        if (crit.is_object())
            for (auto it = crit.begin(); it != crit.end(); ++it) {
                std::string k = it.key();
                std::transform(k.begin(), k.end(), k.begin(), ::tolower);
                (k == "true" ? tc : fc) = it.value();
            }
        auto has = [](const ojson& v) { return !v.is_null() && !(v.is_string() && v.get<std::string>().empty()); };
        out.options.push_back(fl + ": " + (has(fc) ? render_criterion(fc) : "no, the statement does not hold"));
        out.options.push_back(tl + ": " + (has(tc) ? render_criterion(tc) : "yes, the statement holds"));
    }
    return out;
}

}  // namespace

std::string py_json_dumps(const ojson& v, const char* item_sep, const char* key_sep) {
    std::string out;
    dump(out, v, item_sep, key_sep);
    return out;
}

// Temperature-scaled softmax over option logits -> Laya-shaped answer object.
ojson decode_answer(const Question& q, const std::vector<double>& z, const std::vector<float>& act, float t) {
        const size_t k = q.options.size();
        std::vector<double> p(k);
        double mx = -INFINITY, sum = 0;
        for (size_t j = 0; j < k; ++j) mx = std::max(mx, z[j] / t);
        for (size_t j = 0; j < k; ++j) sum += (p[j] = std::exp(z[j] / t - mx));
        for (double& v : p) v /= sum;
        double pmax = *std::max_element(p.begin(), p.end());
        double conf = 1.0;
        if (k >= 2) {
            double ent = 0;
            for (double v : p) ent -= v * std::log(std::clamp(v, 1e-12, 1.0));
            conf = std::clamp(1.0 - ent / std::log(static_cast<double>(k)), 0.0, 1.0);
        }
        ojson action = {{"act_probability", round4(act.empty() ? 0.0 : act[0])}};
        if (q.qt == 0) {
            size_t best = static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
            ojson probs = ojson::object();
            for (size_t j = 0; j < k; ++j) probs[py_key(q.labels[j])] = round4(p[j]);
            return {{"type", "choice"}, {"choice", q.labels[best]}, {"probabilities", probs},
                    {"confidence", round4(conf)}, {"answer_confidence", round4(std::clamp(pmax, 0.0, 1.0))}, {"action", action}};
        }
        if (q.qt == 1) {
            double e = 0;
            ojson legend = ojson::object(), probs = ojson::object();
            for (size_t j = 0; j < k; ++j) {
                e += static_cast<double>(j) * p[j];
                legend[std::to_string(j)] = q.levels[j];
                probs[std::to_string(j)] = round4(p[j]);
            }
            return {{"type", "score"}, {"score", round4(e)}, {"legend", legend}, {"probabilities", probs},
                    {"confidence", round4(conf)}, {"answer_confidence", round4(std::clamp(pmax, 0.0, 1.0))}, {"action", action}};
        }
        return {{"type", "noul"}, {"noul", round4(p[1])}, {"confidence", round4(std::max(p[1], 1.0 - p[1]))},
                {"answer_confidence", round4(std::clamp(pmax, 0.0, 1.0))}, {"action", action}};
    }

struct Engine::Impl {
    std::mutex mu;
    std::mutex cache_mu;
    CalibrationCache null_cache;  // byte-bounded LRU of semantic question+shape keys
    std::unique_ptr<Runner> runner;
    std::vector<float> temperature;
    std::vector<std::pair<std::string, float>> temp_by_options;
    ojson lang_temps;

    ojson answer(const Question& q, const std::vector<double>& z, const std::vector<float>& act, const DecideOptions& opts) const {
        return decode_answer(q, z, act, temperature_for(q.qt, q.options.size(), opts.lang));
    }

    float temperature_for(int qt, size_t k, const std::optional<std::string>& lang) const {
        const std::string bucket = temp_bucket(qt, k);
        if (lang && !lang_temps.empty()) {
            std::string l = lang->substr(0, lang->find('-'));
            std::transform(l.begin(), l.end(), l.begin(), ::tolower);
            if (lang_temps.contains(l)) {
                const ojson& c = lang_temps[l];
                if (c.contains("temperature_by_options") && c["temperature_by_options"].contains(bucket))
                    return clamp_temperature(c["temperature_by_options"][bucket].get<double>());
                if (c.contains("temperature")) return clamp_temperature(c["temperature"][qt].get<double>());
            }
        }
        for (const auto& [b, t] : temp_by_options)
            if (b == bucket) return t;
        return temperature[qt];
    }
};

Engine::Engine(std::shared_ptr<Model> model, RunOptions run) : model_(std::move(model)), impl_(std::make_unique<Impl>()) {
    impl_->runner = std::make_unique<Runner>(model_, run);
    const HParams& h = model_->hparams();
    for (float t : h.temperature) impl_->temperature.push_back(clamp_temperature(t));
    while (impl_->temperature.size() < 3) impl_->temperature.push_back(1.0f);
    ojson tbo = ojson::parse(h.temperature_by_options_json);
    for (auto it = tbo.begin(); it != tbo.end(); ++it)
        impl_->temp_by_options.emplace_back(it.key(), clamp_temperature(it.value().get<double>()));
    impl_->lang_temps = ojson::parse(h.lang_temperatures_json);
}

Engine::~Engine() = default;

static std::vector<Question> parse_questions(const ojson& questions) {
    if (!questions.is_object()) throw QuestionError("'questions' must be an object");
    std::vector<Question> qs;
    for (auto it = questions.begin(); it != questions.end(); ++it) qs.push_back(to_internal(it.key(), it.value()));
    return qs;
}

static std::string serialize_state(const ojson& state) {
    return state.is_string() ? state.get<std::string>() : py_json_dumps(state);
}

namespace {
// build_sequence(): [CLS] <type> question: ins [SEP] ([MASK] opt)* [SEP] state [SEP]
Item build_item(const Tokenizer& tok, const HParams& h, const Question& q, const std::vector<int32_t>& state_ids,
                bool truncate_left, int max_len, int head_max_len, const std::vector<size_t>& order) {
    const std::string& mask = h.mask_token;
    std::vector<int32_t> head = tok.encode(std::string(kQTypes[q.qt]) + " question: " + replace_all(q.ins, mask, " "));
    std::vector<std::vector<int32_t>> opts;
    size_t opt_total = 0;
    for (size_t i : order) {
        std::vector<int32_t> o{h.mask_id};
        auto t = tok.encode(" " + replace_all(q.options[i], mask, " "), 48);
        o.insert(o.end(), t.begin(), t.end());
        opt_total += o.size();
        opts.push_back(std::move(o));
    }
    long budget = static_cast<long>(head_max_len) - static_cast<long>(opt_total);
    if (budget < 16) {
        long per = std::max<long>(4, (head_max_len - 16) / std::max<long>(1, static_cast<long>(opts.size())));
        opt_total = 0;
        for (auto& o : opts) {
            if (static_cast<long>(o.size()) > per) o.resize(per);
            opt_total += o.size();
        }
        budget = static_cast<long>(head_max_len) - static_cast<long>(opt_total);
    }
    long keep = std::max<long>(8, budget);
    if (static_cast<long>(head.size()) > keep) head.resize(keep);

    Item it;
    it.qtype = q.qt;
    it.ids.push_back(h.cls_id);
    it.ids.insert(it.ids.end(), head.begin(), head.end());
    it.ids.push_back(h.sep_id);
    for (auto& o : opts) {
        it.markers.push_back(static_cast<int32_t>(it.ids.size()));
        it.ids.insert(it.ids.end(), o.begin(), o.end());
    }
    it.ids.push_back(h.sep_id);
    long room = std::max<long>(0, max_len - static_cast<long>(it.ids.size()) - 1);
    size_t n = std::min<size_t>(state_ids.size(), static_cast<size_t>(room));
    if (truncate_left) it.ids.insert(it.ids.end(), state_ids.end() - n, state_ids.end());
    else it.ids.insert(it.ids.end(), state_ids.begin(), state_ids.begin() + n);
    it.ids.push_back(h.sep_id);
    if (static_cast<long>(it.ids.size()) > max_len) it.ids.resize(max_len);
    std::vector<int32_t> kept;
    for (int32_t m : it.markers)
        if (m < max_len) kept.push_back(m);
    if (kept.size() != q.options.size())
        throw QuestionError("question '" + q.id + "' options exceed head_max_len=" + std::to_string(head_max_len));
    it.markers = std::move(kept);
    return it;
}

// Same JSON shape, every string replaced by `fill` (numbers/bools are kept: they are structure).
ojson content_free(const ojson& v, const std::string& fill) {
    if (v.is_string()) return fill;
    if (v.is_array()) {
        ojson out = ojson::array();
        for (const auto& e : v) out.push_back(content_free(e, fill));
        return out;
    }
    if (v.is_object()) {
        ojson out = ojson::object();
        for (auto it = v.begin(); it != v.end(); ++it) out[it.key()] = content_free(it.value(), fill);
        return out;
    }
    return v;
}

std::vector<size_t> rotation(size_t k, size_t r) {
    std::vector<size_t> o(k);
    for (size_t i = 0; i < k; ++i) o[i] = (i + r) % k;
    return o;
}
}  // namespace

std::vector<Item> Engine::encode(const ojson& state, const ojson& questions, const DecideOptions& opts) const {
    const HParams& h = model_->hparams();
    auto qs = parse_questions(questions);
    const int head_max_len = opts.head_max_len.value_or(h.head_max_len);
    // a raised option budget must leave room for the state (never binds at the checkpoint defaults)
    const int max_len = effective_max_len(h, opts);
    auto state_ids = model_->tokenizer().encode(replace_all(serialize_state(state), h.mask_token, " "));
    std::vector<Item> items;
    for (const auto& q : qs)
        items.push_back(build_item(model_->tokenizer(), h, q, state_ids, state.is_array(), max_len, head_max_len,
                                   rotation(q.options.size(), 0)));
    return items;
}

std::vector<ojson> Engine::decide_batch(const std::vector<ojson>& states, const ojson& questions, const DecideOptions& opts) {
    if (std::chrono::steady_clock::now() >= opts.deadline) throw HttpError(422, "inference deadline exceeded");
    if (opts.cancelled && opts.cancelled->load(std::memory_order_relaxed)) throw InferenceCancelled();
    impl_->runner->set_deadline(opts.deadline, opts.cancelled);
    const HParams& h = model_->hparams();
    auto qs = parse_questions(questions);
    std::vector<ojson> results;
    if (qs.empty()) {
        for (size_t i = 0; i < states.size(); ++i)
            results.push_back({{"model", h.name}, {"answers", ojson::object()}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}});
        return results;
    }
    const int head_max_len = opts.head_max_len.value_or(h.head_max_len);
    // a raised option budget must leave room for the state (never binds at the checkpoint defaults)
    const int max_len = effective_max_len(h, opts);
    const size_t ens = static_cast<size_t>(std::max(1, opts.ensemble));

    // Phase 1: every (state, question) in the canonical option order, exactly like Laya.
    std::vector<std::vector<int32_t>> state_ids(states.size());
    std::vector<Item> items;
    std::vector<std::pair<size_t, size_t>> where;  // (state, question) per item
    std::vector<size_t> tokens(states.size(), 0);
    for (size_t s = 0; s < states.size(); ++s) {
        if (std::chrono::steady_clock::now() >= opts.deadline) throw HttpError(422, "inference deadline exceeded");
        if (opts.cancelled && opts.cancelled->load(std::memory_order_relaxed)) throw InferenceCancelled();
        if (states[s].is_null()) throw QuestionError("'state' is required");
        state_ids[s] = model_->tokenizer().encode(replace_all(serialize_state(states[s]), h.mask_token, " "));
        for (size_t qi = 0; qi < qs.size(); ++qi) {
            items.push_back(build_item(model_->tokenizer(), h, qs[qi], state_ids[s], states[s].is_array(), max_len,
                                       head_max_len, rotation(qs[qi].options.size(), 0)));
            tokens[s] += items.back().ids.size();
            where.emplace_back(s, qi);
        }
    }
    std::vector<ItemResult> base = run_packed(items);

    // Phase 2 (opt-in): choice questions get extra cyclic option orders; nominal labels only,
    // since rotating ordinal score levels would break their scale. With ensemble_margin < 1 only
    // close calls (top-1 minus top-2 probability below the margin) pay for the extra views.
    struct View {
        std::vector<size_t> order;
        std::vector<float> logits;
    };
    std::vector<std::vector<View>> views(items.size());
    if (ens > 1) {
        std::vector<Item> extra;
        std::vector<std::pair<size_t, std::vector<size_t>>> extra_of;  // base index, order
        for (size_t i = 0; i < items.size(); ++i) {
            const Question& q = qs[where[i].second];
            const size_t k = q.options.size();
            if (q.qt != 0 || k < 2) continue;
            std::vector<float> p(base[i].logits);
            float mx = *std::max_element(p.begin(), p.end()), sum = 0;
            for (float& v : p) sum += (v = std::exp(v - mx));
            std::sort(p.begin(), p.end(), std::greater<>());
            if ((p[0] - p[1]) / sum >= opts.ensemble_margin) continue;
            const size_t n = std::min(ens, k);
            for (size_t v = 1; v < n; ++v) {
                auto order = rotation(k, v * k / n);
                extra.push_back(build_item(model_->tokenizer(), h, q, state_ids[where[i].first],
                                           states[where[i].first].is_array(), max_len, head_max_len, order));
                extra_of.emplace_back(i, std::move(order));
            }
        }
        if (!extra.empty()) {
            auto more = run_packed(extra);
            for (size_t j = 0; j < more.size(); ++j) views[extra_of[j].first].push_back({extra_of[j].second, std::move(more[j].logits)});
        }
    }

    results.resize(states.size());
    for (size_t s = 0; s < states.size(); ++s)
        results[s] = {{"model", h.name}, {"answers", ojson::object()}, {"usage", {{"input_tokens", tokens[s]}, {"output_tokens", 0}}}};
    for (size_t i = 0; i < items.size(); ++i) {
        const auto [s, qi] = where[i];
        const Question& q = qs[qi];
        const size_t k = q.options.size();
        std::vector<double> z(k);
        if (views[i].empty()) {
            for (size_t j = 0; j < k; ++j) z[j] = base[i].logits[j];  // single view: raw logits, exactly Laya
        } else {
            // mean log-softmax over all option orders, mapped back to label index
            views[i].push_back({rotation(k, 0), base[i].logits});
            for (const View& v : views[i]) {
                double mx = *std::max_element(v.logits.begin(), v.logits.end()), sum = 0;
                for (float x : v.logits) sum += std::exp(x - mx);
                const double lse = mx + std::log(sum);
                for (size_t j = 0; j < k; ++j) z[v.order[j]] += (v.logits[j] - lse) / static_cast<double>(views[i].size());
            }
        }
        if (opts.calibrate && q.qt == 0 && k >= 2) {
            const auto& zn = null_logp(qs[qi], states[s], max_len, head_max_len);
            for (size_t j = 0; j < k; ++j) z[j] -= zn[j];
        }
        results[s]["answers"][q.id] = impl_->answer(q, z, base[i].act_probs, opts);
        if (opts.return_logits) {
            ojson lg = ojson::array();
            for (double v : z) lg.push_back(v);
            results[s]["answers"][q.id]["logits"] = std::move(lg);
        }
    }
    return results;
}

std::vector<double> Engine::null_logp(const Question& q, const ojson& state, int max_len, int head_max_len) {
    // Only the rendered inputs affect calibration: ignored metadata and question
    // IDs must never become retained cache material. Exact keys preserve parity.
    const std::string key = ojson::array({q.qt, q.ins, q.options, content_free(state, ""), max_len, head_max_len}).dump();
    {
        std::lock_guard<std::mutex> lk(impl_->cache_mu);
        std::vector<double> cached;
        if (impl_->null_cache.get(key, cached)) return cached;
    }
    const HParams& h = model_->hparams();
    std::vector<Item> items;
    for (const char* fill : {"", "N/A", "[MASK]"}) {
        ojson cf = content_free(state, fill);
        auto ids = model_->tokenizer().encode(replace_all(serialize_state(cf), h.mask_token, " "));
        items.push_back(build_item(model_->tokenizer(), h, q, ids, cf.is_array(), max_len, head_max_len,
                                   rotation(q.options.size(), 0)));
    }
    auto res = run_packed(items);
    const size_t k = q.options.size();
    std::vector<double> mean(k, 0.0);
    for (const auto& r : res) {  // average in probability space, then back to log space
        double mx = *std::max_element(r.logits.begin(), r.logits.end()), sum = 0;
        for (float v : r.logits) sum += std::exp(v - mx);
        for (size_t j = 0; j < k; ++j) mean[j] += std::exp(r.logits[j] - mx) / sum / static_cast<double>(res.size());
    }
    for (double& v : mean) v = std::log(std::max(v, 1e-12));
    std::lock_guard<std::mutex> lk(impl_->cache_mu);
    impl_->null_cache.put(key, mean);
    return mean;
}

std::vector<ItemResult> Engine::run_packed(const std::vector<Item>& items) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    // Sort by length so padding stays small; cap rows and padded tokens per graph. Rows of one
    // state have near-identical lengths, so a single state is usually one graph.
    std::vector<size_t> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return items[a].ids.size() < items[b].ids.size(); });
    std::vector<ItemResult> out(items.size());
    constexpr size_t kMaxRows = 32, kMaxPaddedTokens = 8192;
    size_t start = 0;
    while (start < order.size()) {
        size_t end = start, longest = 0;
        while (end < order.size() && end - start < kMaxRows) {
            size_t len = std::max(longest, items[order[end]].ids.size());
            if (end > start && len * (end - start + 1) > kMaxPaddedTokens) break;
            longest = len;
            ++end;
        }
        std::vector<Item> chunk;
        for (size_t i = start; i < end; ++i) chunk.push_back(items[order[i]]);
        auto r = impl_->runner->run(chunk);
        for (size_t i = start; i < end; ++i) out[order[i]] = std::move(r[i - start]);
        start = end;
    }
    return out;
}

ojson fuse_answers(const ojson& questions, const std::vector<const ojson*>& results, const std::vector<double>& weights) {
    auto qs = parse_questions(questions);
    ojson out = *results.front();
    size_t tokens = 0;
    for (const ojson* r : results) tokens += (*r)["usage"]["input_tokens"].get<size_t>();
    out["usage"]["input_tokens"] = tokens;
    for (const Question& q : qs) {
        const size_t k = q.options.size();
        std::vector<double> z(k, 0.0);
        for (size_t m = 0; m < results.size(); ++m) {
            const ojson& lg = (*results[m])["answers"][q.id]["logits"];
            double mx = -INFINITY, sum = 0;
            for (size_t j = 0; j < k; ++j) mx = std::max(mx, lg[j].get<double>());
            for (size_t j = 0; j < k; ++j) sum += std::exp(lg[j].get<double>() - mx);
            for (size_t j = 0; j < k; ++j) z[j] += weights[m] * (lg[j].get<double>() - mx - std::log(sum));
        }
        std::vector<float> act;
        for (const auto& v : (*results.front())["answers"][q.id]["action"]["act_probability"].is_number()
                                 ? std::vector<float>{(*results.front())["answers"][q.id]["action"]["act_probability"].get<float>()}
                                 : std::vector<float>{})
            act.push_back(v);
        out["answers"][q.id] = decode_answer(q, z, act, 1.0f);
    }
    return out;
}

ojson Engine::decide(const ojson& state, const ojson& questions, const DecideOptions& opts) {
    return decide_batch({state}, questions, opts)[0];
}

const std::vector<std::pair<std::string, std::vector<std::string>>>& question_families() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> kFamilies = {
        {"sentiment", {"sentiment", "polarity"}},
        {"emotion", {"emotion", "emotions", "emotional", "feeling", "feelings", "mood"}},
        {"complaint", {"complaint", "complaints", "complain", "complaining"}},
        {"nli", {"nli", "entailment", "entail", "entails", "contradiction", "contradict", "contradicts"}},
        {"safety", {"safety", "unsafe", "toxic", "toxicity", "harmful", "moderation"}},
        {"reading", {"reading", "comprehension", "passage"}},
        {"similarity", {"similarity", "similar", "paraphrase", "paraphrases"}},
        {"topic", {"topic", "topics"}},
        {"intent", {"intent", "intents", "intention"}},
        {"stance", {"stance"}},
        {"formality", {"formality", "formal", "informal"}},
        {"urgency", {"urgency", "urgent"}},
        {"fact_check", {"fact", "facts", "factual", "claim", "claims"}},
        {"pii", {"pii", "personally"}},
    };
    return kFamilies;
}

namespace {

// Families whose keywords occur as whole lowercase ASCII words in text.
std::vector<std::string> families_in(const std::string& text) {
    std::vector<std::string> words;
    std::string w;
    for (char ch : text + " ") {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c < 0x80 && std::isalnum(c)) w.push_back(static_cast<char>(std::tolower(c)));
        else if (!w.empty()) words.push_back(std::move(w)), w.clear();
    }
    std::vector<std::string> hits;
    for (const auto& [family, keys] : question_families())
        for (const auto& k : keys)
            if (std::find(words.begin(), words.end(), k) != words.end()) {
                hits.push_back(family);
                break;
            }
    return hits;
}

}  // namespace

std::optional<std::string> question_family(const std::string& id, const ojson& question) {
    std::vector<std::string> hits = families_in(id);
    if (hits.empty() && question.is_object() && question.contains("instructions") && question["instructions"].is_string())
        hits = families_in(question["instructions"].get<std::string>());
    if (hits.size() == 1) return hits.front();
    return std::nullopt;
}

std::optional<std::string> request_family(const ojson& questions) {
    std::optional<std::string> family;
    if (!questions.is_object() || questions.empty()) return std::nullopt;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        auto f = question_family(it.key(), it.value());
        if (!f || (family && *family != *f)) return std::nullopt;
        family = f;
    }
    return family;
}

}  // namespace statim
