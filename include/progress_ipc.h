#pragma once
#include <string>

bool progress_ipc_init(const std::string& sock_path);
void progress_ipc_send(const std::string& stage, int progress);
void progress_ipc_send_stage(const std::string& stage, const std::string& status);
void progress_ipc_send_output(const std::string& path);
void progress_ipc_close();