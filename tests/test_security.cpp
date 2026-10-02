#include "statim/security.h"
#include "statim/http_security.h"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace statim;
static int checks = 0;
void require(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}
template<class F> void rejects(int status, F fn) {
    try { fn(); } catch (const HttpError& e) { require(e.status == status, "wrong rejection status"); return; }
    throw std::runtime_error("expected rejection");
}
template<class F> void startup_rejects(F fn) {
    try { fn(); } catch (const std::runtime_error&) { ++checks; return; }
    throw std::runtime_error("expected startup failure");
}
template<class F> std::string startup_error(F fn) {
    try { fn(); } catch (const std::runtime_error& e) { ++checks; return e.what(); }
    throw std::runtime_error("expected startup failure");
}
// Real cpp-httplib request processing without a listening socket. Separate input
// and output ensures an attempted body drain cannot accidentally read its reply.
class MemoryStream : public httplib::Stream {
public:
    std::string input, output;
    size_t offset = 0, empty_reads = 0;
    explicit MemoryStream(std::string s) : input(std::move(s)) {}
    bool is_readable() const override { return offset < input.size(); }
    bool wait_readable() const override { return is_readable(); }
    bool wait_writable() const override { return true; }
    ssize_t read(char* p, size_t n) override {
        n = std::min(n, input.size() - offset);
        if (!n) ++empty_reads;
        std::memcpy(p, input.data() + offset, n); offset += n; return n;
    }
    ssize_t write(const char* p, size_t n) override { output.append(p, n); return n; }
    void get_remote_ip_and_port(std::string& s, int& p) const override { s = "127.0.0.1"; p = 1; }
    void get_local_ip_and_port(std::string& s, int& p) const override { s = "127.0.0.1"; p = 2; }
    socket_t socket() const override { return INVALID_SOCKET; }
    time_t duration() const override { return 0; }
};
class TestServer : public httplib::Server {
public:
    void process(MemoryStream& stream) {
        bool closed = false;
        process_request(stream, "127.0.0.1", 1, "127.0.0.1", 2, false, closed, nullptr);
    }
};
int main(int argc, char** argv) try {
    const std::string test_key = "statim-test-key-0123456789abcdef";
    SecurityLimits limits;
    auto question = ojson{{"type", "choice"}, {"instructions", "Choose"}, {"criteria", {"yes", "no"}}};
    ojson qs = {{"decision", question}};
    ojson request = {{"state", "hello"}, {"questions", qs}};
    require(parse_request(request.dump()) == request, "valid request changed");
    auto yes_no = request;
    yes_no["questions"]["decision"] = {{"type", "yes_no"}, {"instructions", "Is it so?"},
                                           {"labels", {{"false", "no"}, {"true", "yes"}}}};
    auto normalized = parse_decide_request(yes_no.dump(), false);
    require(normalized.body["questions"]["decision"]["type"] == "noul",
            "yes_no was not normalized to noul");
    auto unknown = request;
    unknown["questions"]["decision"]["type"] = "maybe";
    try {
        (void)parse_decide_request(unknown.dump(), false);
        throw std::runtime_error("unknown question type accepted");
    } catch (const HttpError& e) {
        require(e.status == 422 && std::string(e.what()) == "unknown question type; use choice, score, noul or yes_no",
                "unknown question type error did not list yes_no");
    }
    {  // playground CSP: inline blocks pinned by SHA-256 (values from `printf a | openssl dgst -sha256 -binary | base64`)
        const std::string csp = playground_csp("<style>a</style><p>x</p><script>b</script>");
        require(csp.find("style-src 'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs='") != std::string::npos, "playground CSP: style hash");
        require(csp.find("script-src 'sha256-PiPoFgA5WUoziU9lZOGxNIu9egCI1CxKy3PurtWcAJ0='") != std::string::npos, "playground CSP: script hash");
        require(csp.find("frame-ancestors 'none'") != std::string::npos && csp.find("default-src 'none'") != std::string::npos,
                "playground CSP: closed defaults");
        const std::string framed = playground_csp("<style>a</style><p>x</p><script>b</script>",
                                                  {"https://a.example", "https://b.example:8443"});
        const size_t frame_at = framed.find("frame-ancestors ");
        require(frame_at != std::string::npos &&
                    framed.substr(frame_at) == "frame-ancestors https://a.example https://b.example:8443",
                "playground CSP: allowed frame ancestors");
        require(csp.find("script-src 'unsafe-inline'") == std::string::npos, "playground CSP: no inline scripts");
        bool threw = false;
        try { (void)playground_csp("<style>a</style><script>b</script><script>c</script>"); } catch (const std::exception&) { threw = true; }
        require(threw, "playground CSP: a second inline script is refused, not silently left unhashed");
    }
    require(parse_frame_ancestors("https://a.example, http://127.0.0.1:8080 https://a.example") ==
                std::vector<std::string>({"https://a.example", "http://127.0.0.1:8080"}),
            "frame ancestors were not parsed and de-duplicated in order");
    require(parse_frame_ancestors("https://[2001:db8::1]:8443") ==
                std::vector<std::string>({"https://[2001:db8::1]:8443"}),
            "IPv6 frame ancestor rejected");
    const std::vector<std::string> invalid_ancestors = {
        "*", "'self'", "https://a.example/path", "https://a.example?x", "javascript:alert(1)",
        "https://a.example;script-src *", "https://a.example\r\nX-Test: injected",
        "https://[1:2:3:4:5:6:7:8:]"};
    for (const std::string& value : invalid_ancestors)
        startup_rejects([&] { (void)parse_frame_ancestors(value); });
    startup_rejects([] {
        (void)parse_frame_ancestors("https://a.example https://b.example https://c.example https://d.example "
                                    "https://e.example https://f.example https://g.example https://h.example https://i.example");
    });
    startup_rejects([] { (void)parse_frame_ancestors("https://" + std::string(250, 'a') + ".example"); });
    rejects(413, [&] { parse_request("{\"state\":" + std::string(30000, '[') + "0" + std::string(30000, ']') + ",\"questions\":{}}"); });
    require(parse_request(std::string(64, '[') + "0" + std::string(64, ']')).is_array(), "depth boundary rejected");
    rejects(413, [&] { parse_request(std::string(65, '[') + "0" + std::string(65, ']')); });
    std::string wide = "{";
    for (int i = 0; i < 30000; ++i) wide += (i ? "," : "") + std::string("\"k") + std::to_string(i) + "\":0";
    wide += "}";
    rejects(413, [&] { parse_request(wide); });
    auto small = limits; small.max_json_nodes = 5;
    rejects(413, [&] { parse_request("[0,0,0,0,0]", small); });
    rejects(400, [&] { parse_request("{\"x\":1,\"x\":2}"); });
    rejects(400, [&] { parse_request("{broken"); });
    rejects(413, [&] { parse_request(std::string(max_body_bytes + 1, ' ')); });
    auto extra = request; extra["ignored"] = 0;  // unknown fields are ignored, as by laya.serve
    validate_request_fields(extra); ++checks;
    extra = request; extra["questions"]["decision"]["ignored"] = std::string(1024 * 1024, 'x');
    validate_request_fields(extra); ++checks;
    auto bad = request; bad["questions"]["decision"]["instructions"] = std::string(16385, 'x');
    rejects(413, [&] { validate_request_fields(bad); });
    bad = request; bad["questions"]["decision"]["criteria"][0] = std::string(1000000, 'x');
    rejects(413, [&] { validate_request_fields(bad); });
    rejects(413, [&] { check_limits(std::string(50001, 'x'), qs); });
    for (const auto& n : {ojson(4294967328ULL), ojson(-4294967264LL), ojson(UINT64_MAX), ojson(32.0), ojson("32")})
        rejects(422, [&] { bounded_integer(n, 32, 8192, "max_len"); });
    require(bounded_integer(32, 32, 8192, "max_len") == 32, "valid integer changed");
    HParams h; h.n_head = h.head_n_head = 12;
    DecideOptions opts;
    require(effective_max_len(h, opts) == 512, "default budget changed");
    opts.max_len = 32; opts.head_max_len = 8192;
    rejects(422, [&] { effective_max_len(h, opts); });
    opts.head_max_len = 192;
    require(effective_max_len(h, opts) == 320, "Laya automatic state room changed");
    opts = {};
    check_work(qs, 256, h, opts, limits);
    ojson many = ojson::object();
    for (int i = 0; i < 64; ++i) many[std::to_string(i)] = question;
    rejects(413, [&] { check_work(many, 256, h, opts, limits); });
    small = limits; small.max_request_tokens = 511;
    rejects(413, [&] { check_work(qs, 1, h, opts, small); });
    small = limits; small.max_request_work = 1;
    opts.ensemble = 2;
    rejects(413, [&] { check_work(qs, 1, h, opts, small); });
    opts = {}; opts.calibrate = true;
    rejects(413, [&] { check_work(qs, 1, h, opts, small); });
    opts = {};
    rejects(413, [&] { check_work(qs, 1, h, opts, small, 2); });
    small = limits; small.max_attention_bytes = 1;
    rejects(413, [&] { check_work(qs, 1, h, opts, small); });
    small = limits; small.max_attention_bytes = 1000000000;
    opts.max_len = 1370; // shorter rows of 1365 fit six/graph and exceed this cap
    rejects(413, [&] { check_work(qs, 6, h, opts, small); });
    opts = {};
    small = limits; small.max_response_bytes = 1;
    rejects(413, [&] { check_work(qs, 1, h, opts, small); });

    startup_rejects([] { load_key_file("/nonexistent/statim-security-keys"); });
    for (auto text : {"", " , , ", "\t\r\n", "bad key"}) startup_rejects([&] { load_key_env(text); });
    const std::string key31(31, 'a'), key32(32, 'b');
    const std::string short_error = startup_error([&] { validate_api_key(key31, "STATIM_API_KEY"); });
    require(short_error.find("STATIM_API_KEY") != std::string::npos &&
            short_error.find("length 31") != std::string::npos &&
            short_error.find("minimum is 32") != std::string::npos &&
            short_error.find("openssl rand -hex 32") != std::string::npos,
            "short-key error lacks remediation details");
    require(short_error.find(key31) == std::string::npos, "short-key error exposes key bytes");
    validate_api_key(key32, "STATIM_API_KEY"); ++checks;
    require(load_key_env(" " + test_key + "," + key32 + " ") == std::vector<std::string>({test_key, key32}),
            "key whitespace handling");
    for (auto host : {"127.0.0.1", "127.1.2.3", "127.255.255.255", "::1", "localhost"})
        require(is_loopback_host(host), "loopback host rejected");
    for (auto host : {"0.0.0.0", "::", "192.168.1.2", "example.test", "localhost.", "LOCALHOST",
                      "127.0.0", "127.0.0.1.example", "127.0.0.256"})
        require(!is_loopback_host(host), "non-loopback host accepted");
    auto path = std::filesystem::temp_directory_path() / ("statim-keys-" + std::to_string(getpid()));
    for (auto text : {"", "# comment\n  # comment\r\n"}) {
        { std::ofstream file(path); file << text; }
        startup_rejects([&] { load_key_file(path); });
    }
    { std::ofstream file(path); file << "# comment\n" << test_key << "\n\n  " << key32 << "\r\n"; }
    require(load_key_file(path) == std::vector<std::string>({test_key, key32}), "several valid file keys rejected");
    { std::ofstream file(path); file << "# comment\n" << key31 << "\n"; }
    const std::string file_error = startup_error([&] { load_key_file(path); });
    require(file_error.find(path.string() + ":2") != std::string::npos, "short file-key error lacks file and line");
    require(file_error.find(key31) == std::string::npos, "short file-key error exposes key bytes");
    { std::ofstream file(path); file << test_key << "\n"; }
    std::filesystem::permissions(path, std::filesystem::perms::none);
    if (geteuid() != 0) startup_rejects([&] { load_key_file(path); });
    else std::cerr << "note: running as root, which can read a mode-000 file; unreadable key file check skipped\n";
    std::filesystem::permissions(path, std::filesystem::perms::owner_all);
    std::filesystem::remove(path);
    require(valid_request_id("audit-123_ABC.test"), "valid request id rejected");
    for (auto s : {"audit%0D%0Aforged", "audit\r\nforged", "\" ,\"admin\":true", ""})
        require(!valid_request_id(s), "log injection accepted");
    require(!valid_request_id(std::string(129, 'x')), "long request ID accepted");
    std::atomic<bool> times_ok{true};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) threads.emplace_back([&] {
        for (int j = 0; j < 1000; ++j) if (now_iso8601().size() != 20) times_ok = false;
    });
    for (auto& t : threads) t.join();
    require(times_ok, "timestamp race");

    require(bearer_authorized(httplib::Request{}, {}), "local auth-off default rejected");
    {
        // fixed-length comparison: a maximal key still matches, prefixes/extensions and oversized headers do not
        const std::string big(4096, 'k');
        auto with = [](const std::string& v) { httplib::Request r; r.headers.emplace("Authorization", v); return r; };
        require(bearer_authorized(with("Bearer " + big), {test_key, big}), "4096-byte key rejected");
        require(!bearer_authorized(with("Bearer " + big + "k"), {big}), "longer header accepted");
        require(!bearer_authorized(with("Bearer " + big.substr(1)), {big}), "key prefix accepted");
        require(!bearer_authorized(with("Bearer " + test_key.substr(0, 31)), {test_key}), "key prefix accepted");
        require(!bearer_authorized(with("Bearer " + std::string(100000, 'k')), {big}), "oversized header accepted");
        require(bearer_authorized(with("Bearer " + test_key), {big, test_key}), "second key rejected");
    }
    CalibrationCache cache(1024, 2);
    std::vector<double> found;
    cache.put("a", {1}); cache.put("b", {2});
    require(cache.get("a", found) && found == std::vector<double>{1}, "cache hit changed result");
    cache.put("c", {3});
    require(!cache.get("b", found) && cache.get("a", found), "cache is not LRU");
    cache.put(std::string(1024 * 1024, 'x'), {0});
    require(cache.bytes() <= 1024 && cache.size() <= 2, "oversized cache entry retained");
    for (int i = 0; i < 5000; ++i) cache.put(std::to_string(i), {double(i)});
    require(cache.bytes() <= 1024 && cache.size() <= 2, "cache grows unbounded");
    require(cache.get("4999", found) && found[0] == 4999, "cache final result changed");

    TestServer server;
    configure_http_security(server, {test_key});
    bool reached = false;
    server.Post("/v1/systemone", [&](const httplib::Request&, httplib::Response& r) { reached = true; r.set_content("{}", "application/json"); });
    for (auto endpoint : {"/metrics", "/v1/models", "/health", "/ready"})
        server.Get(endpoint, [](const httplib::Request&, httplib::Response& r) { r.set_content("{}", "application/json"); });
    server.Get("/throw", [](const httplib::Request&, httplib::Response&) { throw std::runtime_error("private-file-secret"); });
    auto replay = [&](std::string wire, int status) {
        MemoryStream stream(std::move(wire));
        server.process(stream);
        require(stream.output.find("HTTP/1.1 " + std::to_string(status)) == 0, "unexpected HTTP status");
        return stream;
    };
    auto no_auth = replay("POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n", 401);
    require(no_auth.empty_reads == 0 && !reached, "unauthorized body read/drained");
    const std::string auth = "Authorization: Bearer " + test_key + "\r\n";
    auto large = replay("POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n" + auth + "Content-Length: 999999999\r\n\r\n", 413);
    require(large.empty_reads == 0 && !reached, "oversized body read/drained");
    for (auto framing : {"Content-Length: 100\r\nTransfer-Encoding: chunked", "Content-Length: 0\r\nTransfer-Encoding: chunked",
                         "Content-Length: 2\r\nContent-Length: 2", "Content-Length: 2\r\nContent-Length: 3", "Content-Length: 2, 2", "Content-Length: -1"}) {
        auto stream = replay(std::string("POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n") + auth + framing + "\r\n\r\n", 400);
        require(!reached && stream.empty_reads == 0, "conflicting framing consumed body");
    }
    auto chunked = replay("POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n" + auth +
                          "Transfer-Encoding: chunked\r\n\r\n300000\r\n" + std::string(3 * 1024 * 1024, 'x') + "\r\n0\r\n\r\n", 413);
    require(!reached && chunked.offset < chunked.input.size(), "oversized chunked body drained or dispatched");
    for (auto endpoint : {"/metrics", "/v1/models"}) {
        replay(std::string("GET ") + endpoint + " HTTP/1.1\r\nHost: localhost\r\n\r\n", 401);
        replay(std::string("GET ") + endpoint + " HTTP/1.1\r\nHost: localhost\r\n" + auth + "\r\n", 200);
    }
    for (auto endpoint : {"/health", "/ready"}) replay(std::string("GET ") + endpoint + " HTTP/1.1\r\nHost: localhost\r\n\r\n", 200);
    auto failure = replay("GET /throw HTTP/1.1\r\nHost: localhost\r\n" + auth + "\r\n", 500);
    require(failure.output.find("private-file-secret") == std::string::npos && failure.output.find("EXCEPTION_WHAT") == std::string::npos,
            "exception disclosed to client");
    require(failure.output.find("{\"detail\":\"internal server error\"}") != std::string::npos, "exception shape changed");
    MemoryStream inner("bytes remain");
    DeadlineStream expired(inner, std::chrono::steady_clock::now());
    char c;
    require(expired.read(&c, 1) < 0 && !expired.wait_readable() && inner.offset == 0, "absolute deadline bypassed by ready data");
    // Deterministically fill the worker and its single pending queue slot.
    httplib::ThreadPool pool(1, 0, 1);
    std::promise<void> started, release;
    auto wait = release.get_future().share();
    require(pool.enqueue([&] { started.set_value(); wait.wait(); }), "worker enqueue failed");
    started.get_future().wait();
    bool queued = pool.enqueue([] {}), rejected = !pool.enqueue([] {});
    release.set_value(); pool.shutdown();
    require(queued && rejected, "HTTP queue is not bounded");
    if (argc > 1) {
        auto model = Model::load(argv[1], "cpu");
        RunOptions ro; ro.n_threads = 2;
        Engine engine(model, ro);
        DecideOptions normal; normal.return_logits = true;
        auto baseline = engine.decide("A simple decision.", qs, normal);
        ojson noul_qs = {{"answer", {{"type", "noul"}, {"instructions", "Is this simple?"},
                                      {"labels", {{"false", "no"}, {"true", "yes"}}}}}};
        auto yes_no_qs = noul_qs;
        yes_no_qs["answer"]["type"] = "yes_no";
        require(engine.decide("A simple decision.", yes_no_qs, normal) ==
                    engine.decide("A simple decision.", noul_qs, normal),
                "yes_no does not behave identically to noul");
        auto timed = normal; timed.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        require(engine.decide("A simple decision.", qs, timed) == baseline, "deadline changes inference bits");
        timed.deadline = std::chrono::steady_clock::now();
        rejects(422, [&] { engine.decide("A simple decision.", qs, timed); });
        auto calibrated = normal; calibrated.calibrate = true;
        auto miss = engine.decide("A simple decision.", qs, calibrated);
        require(engine.decide("A simple decision.", qs, calibrated) == miss, "calibration cache hit changes bits");
        // unknown question fields are ignored and never reach the calibration cache key
        auto ignored = qs; ignored["decision"]["ignored"] = std::string(1024 * 1024, 'x');
        require(engine.decide("A simple decision.", ignored, calibrated) == miss, "ignored field changes the decision");
        Runner runner(model, ro);
        auto items = engine.encode(std::string(10000, 'x'), qs);
        runner.set_deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
        rejects(422, [&] { runner.run(items); });
        runner.set_deadline(std::chrono::steady_clock::time_point::max());
        auto recovered = runner.run(items);
        Runner fresh(model, ro);
        auto expected = fresh.run(items);
        require(recovered.size() == expected.size() && recovered[0].logits == expected[0].logits &&
                recovered[0].act_probs == expected[0].act_probs, "CPU executor corrupted after cancellation");
    }
    std::cout << "security: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "security failure: " << e.what() << '\n';
    return 1;
}
