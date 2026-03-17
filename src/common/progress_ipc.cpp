#include "progress_ipc.h"

#if defined(_WIN32)
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif
#include <cstring>
#include <mutex>
#include <algorithm>
#include <cmath>

namespace {
#if defined(_WIN32)
HANDLE g_pipe = INVALID_HANDLE_VALUE;
#else
int g_fd = -1;
#endif
std::mutex g_mu;
}

bool progress_ipc_init(const std::string& sock_path) {
    if (sock_path.empty()) return false;

    #if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;

    h = CreateFileA(
        sock_path.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (h == INVALID_HANDLE_VALUE) return false;

    DWORD err = GetLastError();

    if (err == ERROR_PIPE_BUSY) {
        if (!WaitNamedPipeA(sock_path.c_str(), 200)) {
            Sleep(20);
        } else if (err == ERROR_FILE_NOT_FOUND) {
            Sleep(20);
        } else {
            return false;
        }
    }
    if (h == INVALID_HANDLE_VALUE) return false;
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
    g_pipe = h;
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
    DWORD written = 0;
    (void)WriteFile(g_pipe, msg.data(), (DWORD)msg.size(), &written, nullptr);
    #else
    (void)::write(g_fd, msg.data(), msg.size());
    #endif
}

void progress_ipc_send_output(const std::string& path){
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_pipe == INVALID_HANDLE_VALUE) return;
    #else
    if (g_fd < 0) return;
    #endif
    std::string msg = "{\"kind\":\"output\",\"path\":\"" + json_escape(path) + "\"}\n";
    send_msg(msg);
}

void progress_ipc_send_stage(const std::string& stage, const std::string& status) {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_pipe == INVALID_HANDLE_VALUE) return;
    #else
    if (g_fd < 0) return;
    #endif
    std::string msg = "{\"kind\":\"stage\",\"stage\":\"" + stage + "\",\"status\":\"" + status + "\"}\n";
    send_msg(msg);
}

void progress_ipc_send(const std::string& stage, int progress) {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_pipe == INVALID_HANDLE_VALUE) return;
    #else
    if (g_fd < 0) return;
    #endif
    progress = (std::max)(0, (std::min)(100, progress));
    std::string msg = "{\"kind\":\"progress\",\"stage\":\"" + stage + "\",\"progress\":" + std::to_string(progress) + "}\n";
    send_msg(msg);
}

void progress_ipc_close() {
    std::lock_guard<std::mutex> lk(g_mu);
    #if defined(_WIN32)
    if (g_pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pipe);
        g_pipe = INVALID_HANDLE_VALUE;
    }
    #else
    if (g_fd >= 0) {
        ::close(g_fd);
        g_fd = -1;
    }
    #endif
}