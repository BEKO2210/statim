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
inline bool bearer_authorized(const httplib::Request& req, const std::vector<std::string>& keys) {
    if (keys.empty()) return true;
    if (req.get_header_value_count("Authorization") != 1) return false;
    const auto auth = req.get_header_value("Authorization");
    // Every comparison runs over the longest possible header ("Bearer " + a 4096-byte key), so the
    // time depends on neither the configured keys' lengths nor the supplied header's.
    constexpr size_t n = 7 + 4096;
    if (auth.size() > n) return false;
    bool ok = false;
    for (const auto& key : keys) {
        const std::string expected = "Bearer " + key;
        unsigned char diff = static_cast<unsigned char>(expected.size() != auth.size());
        for (size_t i = 0; i < n; ++i)
            diff |= static_cast<unsigned char>((i < expected.size() ? expected[i] : 0) ^ (i < auth.size() ? auth[i] : 0));
        ok |= diff == 0;
    }
    return ok;
}
inline void configure_http_security(httplib::Server& srv, std::vector<std::string> keys) {
    srv.set_payload_max_length(max_body_bytes);
    srv.set_exception_handler(sanitize_exception);
    srv.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.body.empty()) send_error(res, res.status, res.status == 413 ? "request body exceeds limit" : "HTTP request failed");
    });
    srv.set_pre_routing_handler([keys = std::move(keys)](const httplib::Request& req, httplib::Response& res) {
        if (reject_framing(req, res)) return httplib::Server::HandlerResponse::Handled;
        const bool public_path = req.path == "/health" || req.path == "/ready" || req.path == "/";
        if (!public_path && !bearer_authorized(req, keys)) {
            send_error(res, 401, "invalid or missing bearer token");
            return httplib::Server::HandlerResponse::Handled;
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
