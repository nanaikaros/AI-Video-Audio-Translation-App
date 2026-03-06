#include "include/translation.h"
#include "include/video.h"
#include "whisper.h"
#include "whisper_bridge.h"
#include "include/params.h"
#include "include/progress_ipc.h"

#include "include/spdlog/sinks/basic_file_sink.h"
#include "include/spdlog/spdlog.h"

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

static int params_parse(int argc, char ** argv, ai_translation_parmas& atp){
    atp.app_name = argv[0];
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "--video" && i + 1 < argc) atp.video_path = argv[++i];
        else if (arg == "--whisper-model" && i + 1 < argc) atp.whisper_model_path = argv[++i];
        else if ((arg == "--translation-model" || arg == "-m") && i + 1 < argc) atp.translation_model_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) atp.thread_num = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) atp.output_video_path = argv[++i];
        else if (arg == "--progress-sock" && i + 1 < argc) atp.progress_sock_path = argv[++i];
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return -1;
        }
    }

    if (atp.video_path.empty()) return -2;
    if (atp.whisper_model_path.empty()) return -3;
    if (atp.translation_model_path.empty()) return -4;
    return 0;
}

void logfile_init()
{
    try 
    {
        auto logger = spdlog::basic_logger_mt("app_logger", "logs/basic-log.log");
        logger->set_pattern("[%n][%Y-%m-%d %H:%M:%S.%e] [%l] [%t]  %v");
  		logger->set_level(spdlog::level::debug);
  		spdlog::flush_every(std::chrono::seconds(5));

        logger->info("--------------------------------");
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
    }
}

int main(int argc, char ** argv){
    logfile_init();
    auto logger = spdlog::get("app_logger");

    ai_translation_parmas atp;
    int ret = params_parse(argc, argv, atp);
    if (ret != 0) {
        logger->error("params error");
        std::cerr << "params error: " << ret << std::endl;
        return -1;
    }

    progress_ipc_init(atp.progress_sock_path);

    output_params out_params;
    pipeline_buffer buffer;

    progress_ipc_send_stage("prepare", "running");

    // 视频处理
    progress_ipc_send_stage("video", "running");
    ret = video_strat(atp, out_params, buffer);
    if (ret != 0) {
        logger->error("video_strat failed");
        return -1;
    }
    progress_ipc_send_stage("video", "running");
    logger->info("video completed");

    // whisper 语音识别处理
    progress_ipc_send_stage("whisper", "running");
    ret = whisper_start(atp, out_params, buffer);
    if (ret != 0) {
        logger->error("whisper_start failed");
        return -1;
    }
    progress_ipc_send_stage("whisper", "done");
    logger->info("whisper complete");

    // translation翻译处理
    progress_ipc_send_stage("translation", "running");
    ret = translation_start(atp, buffer);
    if (ret != 0) {
        logger->error("translation_start failed");
        return -1;
    }
    progress_ipc_send_stage("translation", "done");

    if(atp.output_video_path.empty()) {
        logger->error("output video path is empty");
        return -1;
    }

    std::filesystem::path file(atp.video_path);
    std::string fileName = file.filename().stem().string();

    // 默认mp4格式
    // todo: 改为支持多种视频格式输出
    std::string output_path = atp.output_video_path + "/" + fileName + "_subtitle.mkv";

    ret = mux_video_with_ass_api(atp.video_path.c_str(), buffer, output_path.c_str());
    if (ret != 0) {
        logger->error("mux_video_with_srt_mp4_api failed");
        return -1;
    }

    progress_ipc_send_stage("done", "done");
    logger->info("subtitle file in: {}", output_path);
    progress_ipc_send_output(output_path);
    progress_ipc_close();
    return 0;
}