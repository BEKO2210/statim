#pragma once
#include "httplib.h"
#include "statim/security.h"

namespace statim {
// One absolute header+body deadline per request, including a peer that continually
// sends bytes. Writes retain their own timeout so a completed inference can reply.
class DeadlineStream : public httplib::Stream {
public:
    DeadlineStream(httplib::Stream& inner, std::chrono::steady_clock::time_point deadline)
        : inner_(inner), deadline_(deadline) {}
    bool expired() const { return std::chrono::steady_clock::now() >= deadline_; }
    bool is_readable() const override { return !expired() && inner_.is_readable(); }
    bool wait_readable() const override { return !expired() && inner_.wait_readable(); }
    bool wait_writable() const override { return inner_.wait_writable(); }
    bool is_peer_alive() const override { return inner_.is_peer_alive(); }
    ssize_t read(char* p, size_t n) override {
        if (expired()) { error_ = httplib::Error::Timeout; return -1; }
        auto count = inner_.read(p, n);
        error_ = inner_.get_error();
        return count;
    }
    ssize_t write(const char* p, size_t n) override { return inner_.write(p, n); }
    void get_remote_ip_and_port(std::string& ip, int& port) const override { inner_.get_remote_ip_and_port(ip, port); }
    void get_local_ip_and_port(std::string& ip, int& port) const override { inner_.get_local_ip_and_port(ip, port); }
    socket_t socket() const override { return inner_.socket(); }
    time_t duration() const override { return inner_.duration(); }
    void set_read_timeout(time_t sec, time_t usec) override { inner_.set_read_timeout(sec, usec); }
    const char* buffered_data(size_t& n) const override {
        if (expired()) { n = 0; return nullptr; }
        return inner_.buffered_data(n);
    }
    void consume_buffered(size_t n) override { inner_.consume_buffered(n); }
private:
    httplib::Stream& inner_;
    std::chrono::steady_clock::time_point deadline_;
};
class SecureServer : public httplib::Server {
public:
    explicit SecureServer(int seconds) : seconds_(seconds) {}
private:
    int seconds_;
    bool process_and_close_socket(socket_t sock) override {
        // Application owns only this accepted socket. The bounded upstream pool
        // owns pending sockets; rejected enqueues are closed by cpp-httplib.
        auto result = serve_guarded([&] {
            std::string remote, local;
            int rp = 0, lp = 0;
            httplib::detail::get_remote_ip_and_port(sock, remote, rp);
            httplib::detail::get_local_ip_and_port(sock, local, lp);
            return httplib::detail::process_server_socket_core(svr_sock_, sock, keep_alive_max_count_,
                keep_alive_timeout_sec_, [&](bool close, bool& closed) {
                    const auto started = std::chrono::steady_clock::now();
                    httplib::detail::SocketStream socket_stream(sock, seconds_, 0, 30, 0,
                        static_cast<time_t>(seconds_) * 1000, started);
                    DeadlineStream stream(socket_stream, started + std::chrono::seconds(seconds_));
                    return process_request(stream, remote, rp, local, lp, close, closed, nullptr);
                });
        });
        // No body draining on rejected requests: close immediately.
        httplib::detail::shutdown_socket(sock);
        httplib::detail::close_socket(sock);
        return result;
    }
};
inline void send_error(httplib::Response& res, int status, const std::string& detail) {
    res.status = status;
    res.set_content(ojson{{"detail", detail}}.dump(), "application/json");
    res.set_header("Connection", "close");
}
inline void sanitize_exception(const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string detail = "unknown exception";
    try { if (ep) std::rethrow_exception(ep); }
    catch (const std::exception& e) { detail = e.what(); }
    catch (...) {}
    // error_handler_t::replace: an exception message with invalid UTF-8 must not throw from here.
    std::fprintf(stderr, "%s\n", ojson{{"ts", now_iso8601()}, {"level", "error"},
        {"event", "http_exception"}, {"error", detail}}.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
    res.headers.erase("EXCEPTION_WHAT");
    send_error(res, 500, "internal server error");
}
inline bool reject_framing(const httplib::Request& req, httplib::Response& res) {
    const auto lengths = req.get_header_value_count("Content-Length");
    if (lengths > 1 || (lengths && req.has_header("Transfer-Encoding"))) {
        send_error(res, 400, "ambiguous HTTP body framing");
        return true;
    }
    if (lengths) {
        const auto value = req.get_header_value("Content-Length");
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
            send_error(res, 400, "invalid Content-Length");
            return true;
        }
    }
    if (lengths && req.get_header_value_u64("Content-Length") > max_body_bytes) {
        send_error(res, 413, "request body exceeds 2 MiB");
        return true;
    }
    return false;
}
inline const ApiKey* authenticated_key(const httplib::Request& req, const std::vector<ApiKey>& keys) {
    if (req.get_header_value_count("Authorization") != 1) return nullptr;
    const auto auth = req.get_header_value("Authorization");
    // Every comparison runs over the longest possible header ("Bearer " + a 4096-byte key), so the
    // time depends on neither the configured keys' lengths nor the supplied header's.
    constexpr size_t n = 7 + 4096;
    if (auth.size() > n) return nullptr;
    const ApiKey* matched = nullptr;
    for (const auto& key : keys) {
        const std::string expected = "Bearer " + key.key;
        unsigned char diff = static_cast<unsigned char>(expected.size() != auth.size());
        for (size_t i = 0; i < n; ++i)
            diff |= static_cast<unsigned char>((i < expected.size() ? expected[i] : 0) ^ (i < auth.size() ? auth[i] : 0));
        if (diff == 0) matched = &key;
    }
    return matched;
}
inline bool bearer_authorized(const httplib::Request& req, const std::vector<ApiKey>& keys) {
    return keys.empty() || authenticated_key(req, keys) != nullptr;
}
inline bool bearer_authorized(const httplib::Request& req, const std::vector<ApiKey>& keys, ApiScope scope) {
    if (keys.empty()) return true;
    const ApiKey* key = authenticated_key(req, keys);
    return key && key_has_scope(*key, scope);
}
inline bool cors_origin_allowed(const httplib::Request& req, const std::vector<std::string>& origins) {
    if (origins.empty() || req.get_header_value_count("Origin") != 1) return false;
    const std::string origin = req.get_header_value("Origin");
    return std::find(origins.begin(), origins.end(), origin) != origins.end();
}
inline bool cors_route(const std::string& path) {
    return path == "/" || path == "/health" || path == "/ready" || path == "/metrics" ||
           path == "/v1/models" || path == "/v1/systemone" || path == "/v1/systemone/batch";
}
inline void configure_http_security(httplib::Server& srv, std::vector<ApiKey> keys,
                                    std::vector<std::string> cors_origins = {}) {
    srv.set_payload_max_length(max_body_bytes);
    srv.set_exception_handler(sanitize_exception);
    srv.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.body.empty()) send_error(res, res.status, res.status == 413 ? "request body exceeds limit" : "HTTP request failed");
    });
    srv.set_post_routing_handler([cors_origins](const httplib::Request& req, httplib::Response& res) {
        if (!cors_origin_allowed(req, cors_origins)) return;
        std::string exposed;
        if (res.has_header("Retry-After")) exposed = "Retry-After";
        if (res.has_header("X-Request-Id")) exposed += (exposed.empty() ? "" : ", ") + std::string("X-Request-Id");
        if (!exposed.empty()) res.set_header("Access-Control-Expose-Headers", exposed);
    });
    srv.set_pre_routing_handler([keys = std::move(keys), cors_origins = std::move(cors_origins)]
                                (const httplib::Request& req, httplib::Response& res) {
        // On every response, including errors: no MIME sniffing, no referrer to other sites.
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("Referrer-Policy", "no-referrer");
        const bool cors = cors_origin_allowed(req, cors_origins);
        // With an allowlist the response depends on Origin for every request, so a shared cache
        // must not hand a response without CORS headers to an allowed origin (or the reverse).
        if (!cors_origins.empty()) res.set_header("Vary", "Origin");
        if (cors) res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        if (cors && req.method == "OPTIONS" && cors_route(req.path) &&
            req.get_header_value_count("Access-Control-Request-Method") == 1) {
            const std::string method = req.get_header_value("Access-Control-Request-Method");
            if (method == "GET" || method == "POST") {
                res.status = 204;
                res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
                res.set_header("Access-Control-Max-Age", "600");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        if (reject_framing(req, res)) return httplib::Server::HandlerResponse::Handled;
        const bool public_path = req.path == "/health" || req.path == "/ready" || req.path == "/";
        if (!public_path && !keys.empty()) {
            const ApiKey* key = authenticated_key(req, keys);
            if (!key) {
                send_error(res, 401, "invalid or missing bearer token");
                return httplib::Server::HandlerResponse::Handled;
            }
            const bool inference = (req.method == "POST" &&
                    (req.path == "/v1/systemone" || req.path == "/v1/systemone/batch")) ||
                                   (req.method == "GET" && req.path == "/v1/models");
            const bool metrics = req.method == "GET" && req.path == "/metrics";
            if ((inference && !key_has_scope(*key, ApiScope::inference)) ||
                (metrics && !key_has_scope(*key, ApiScope::metrics))) {
                const char* scope = metrics ? "metrics" : "inference";
                send_error(res, 403, std::string("API key lacks the '") + scope + "' scope");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        if (public_path || req.path == "/metrics" || req.path == "/v1/models") {
            if (req.has_header("Transfer-Encoding") || req.get_header_value_u64("Content-Length") != 0) {
                send_error(res, 400, "this endpoint does not accept a request body");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
}
} // namespace statim
