#include "progress_ipc.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <mutex>
#include <algorithm>

namespace {
int g_fd = -1;
std::mutex g_mu;
}

bool progress_ipc_init(const std::string& sock_path) {
    if (sock_path.empty()) return false;

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path.c_str());

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return false;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    g_fd = fd;
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

void progress_ipc_send_output(const std::string& path){
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_fd < 0) return;
    std::string msg = "{\"kind\":\"output\",\"path\":\"" + json_escape(path) + "\"}\n";
    (void)::write(g_fd, msg.data(), msg.size());
}

void progress_ipc_send_stage(const std::string& stage, const std::string& status) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_fd < 0) return;
    std::string msg = "{\"kind\":\"stage\",\"stage\":\"" + stage + "\",\"status\":\"" + status + "\"}\n";
    (void)::write(g_fd, msg.data(), msg.size());
}

void progress_ipc_send(const std::string& stage, int progress) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_fd < 0) return;

    progress = std::max(0, std::min(100, progress));
    std::string msg = "{\"kind\":\"progress\",\"stage\":\"" + stage + "\",\"progress\":" + std::to_string(progress) + "}\n";
    (void)::write(g_fd, msg.data(), msg.size());
}

void progress_ipc_close() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_fd >= 0) {
        ::close(g_fd);
        g_fd = -1;
    }
}