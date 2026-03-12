#include "progress_ipc.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#endif
#include <unistd.h>
#include <cstring>
#include <mutex>
#include <algorithm>

namespace {
#if defined(_WIN32)
SOCKET g_sock = INVALID_SOCKET;
bool wsa_started = false;
#else
int g_fd = -1;
#endif
std::mutex g_mu;
}

bool progress_ipc_init(const std::string& sock_path) {
    if (sock_path.empty()) return false;

    #if defined(_WIN32)
    WSADATA wsa;
    if (!wsa_started) {
        if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) return false;
        wsa_started = true;
    }

    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    SOCKET s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) return false;

    if (connect(s, res->ai_addr, (int)res->ai_addrlen) != 0) {
        return false;
    }
    #else
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path.c_str());

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return false;
    }
    #endif
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    g_sock = s;
    #else
    g_fd = fd;
    #endif
    return true;
}

static std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

void send_msg(std::string msg){
    #if defined(_WIN32)
    send(g_sock, msg.data(), (int)msg.size(), 0);
    #else
    (void)::write(g_fd, msg.data(), msg.size());
    #endif
}

void progress_ipc_send_output(const std::string& path){
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_sock == INVALID_SOCKET) return;
    #else
    if (g_fd < 0) return;
    #endif
    std::string msg = "{\"kind\":\"output\",\"path\":\"" + json_escape(path) + "\"}\n";
    send_msg(msg);
}

void progress_ipc_send_stage(const std::string& stage, const std::string& status) {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_sock == INVALID_SOCKET) return;
    #else
    if (g_fd < 0) return;
    #endif
    std::string msg = "{\"kind\":\"stage\",\"stage\":\"" + stage + "\",\"status\":\"" + status + "\"}\n";
    send_msg(msg);
}

void progress_ipc_send(const std::string& stage, int progress) {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_sock == INVALID_SOCKET) return;
    #else
    if (g_fd < 0) return;
    #endif
    progress = std::max(0, std::min(100, progress));
    std::string msg = "{\"kind\":\"progress\",\"stage\":\"" + stage + "\",\"progress\":" + std::to_string(progress) + "}\n";
    send_msg(msg);
}

void progress_ipc_close() {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_sock != INVALID_SOCKET) {
        closesocket(g_sock);
        g_sock = INVALID_SOCKET;
    }
    if (wsa_started) {
        WSACleanup();
        wsa_started = false;
    }
    #else
    if (g_fd >= 0) {
        ::close(g_fd);
        g_fd = -1;
    }
    #endif
}