// tools/docker/healthcheck.cpp
//
// Minimal static HTTP probe for container HEALTHCHECK.
// Connects to 127.0.0.1:8080 (or configurable host/port), sends GET /health,
// and exits 0 if the server responds with HTTP 200.
// Exits 1 on any error, connection failure, timeout, or non-200 status code.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int kTimeoutMs = 3000;

int main(int argc, char* argv[]) {
    const char* host = "127.0.0.1";
    int port = 8080;
    const char* path = "/health";

    if (const char* env_host = std::getenv("STATIM_HEALTHCHECK_HOST")) {
        host = env_host;
    }
    if (const char* env_port = std::getenv("STATIM_HEALTHCHECK_PORT")) {
        port = std::atoi(env_port);
    }
    if (const char* env_path = std::getenv("STATIM_HEALTHCHECK_PATH")) {
        path = env_path;
    }

    if (argc > 1) host = argv[1];
    if (argc > 2) port = std::atoi(argv[2]);
    if (argc > 3) path = argv[3];

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return 1;

    // Set non-blocking to enforce connect timeout
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
        close(sock);
        return 1;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        close(sock);
        return 1;
    }

    int rc = connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
    if (rc < 0) {
        if (errno != EINPROGRESS) {
            close(sock);
            return 1;
        }
        struct pollfd pfd{};
        pfd.fd = sock;
        pfd.events = POLLOUT;
        int poll_rc = poll(&pfd, 1, kTimeoutMs);
        if (poll_rc <= 0) {
            close(sock);
            return 1;
        }
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 || so_error != 0) {
            close(sock);
            return 1;
        }
    }

    // Restore blocking with socket send/recv timeouts
    fcntl(sock, F_SETFL, flags);
    struct timeval tv{};
    tv.tv_sec = kTimeoutMs / 1000;
    tv.tv_usec = (kTimeoutMs % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    char req[512];
    int req_len = std::snprintf(req, sizeof(req),
        "GET %s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n",
        path, host, port);
    if (req_len <= 0 || req_len >= static_cast<int>(sizeof(req))) {
        close(sock);
        return 1;
    }

    ssize_t sent = send(sock, req, static_cast<size_t>(req_len), 0);
    if (sent != req_len) {
        close(sock);
        return 1;
    }

    char buf[512];
    ssize_t recvd = recv(sock, buf, sizeof(buf) - 1, 0);
    close(sock);
    if (recvd <= 0) return 1;
    buf[recvd] = '\0';

    // Must be HTTP response with status code 200
    // Format: "HTTP/1.x 200 ..."
    if (std::strncmp(buf, "HTTP/1.", 7) != 0) return 1;

    const char* p = std::strchr(buf, ' ');
    if (!p) return 1;
    while (*p == ' ') p++;

    if (p[0] == '2' && p[1] == '0' && p[2] == '0' && (p[3] == ' ' || p[3] == '\r' || p[3] == '\n')) {
        return 0;
    }
    return 1;
}
