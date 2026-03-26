#include "include/translation.h"
#include "include/video.h"
#include "whisper.h"
#include "whisper_bridge.h"
#include "include/params.h"
#include "include/progress_ipc.h"

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

static int params_parse(int argc, char ** argv, ai_translation_parmas& atp){
    atp.app_name = argv[0];
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if ((arg == "--video" || arg == "-v") && i + 1 < argc) atp.video_path = argv[++i];
        else if ((arg == "--whisper-model" || arg == "-w") && i + 1 < argc) atp.whisper_model_path = argv[++i];
        else if ((arg == "--translation-model" || arg == "-m") && i + 1 < argc) atp.translation_model_path = argv[++i];
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) atp.thread_num = std::stoi(argv[++i]);
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) atp.output_video_path = argv[++i];
        else if (arg == "--progress-sock" && i + 1 < argc) atp.progress_sock_path = argv[++i];
        else if (arg == "--ocr") atp.use_ocr = true;
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

int main(int argc, char ** argv){
    ai_translation_parmas atp;
    int ret = params_parse(argc, argv, atp);
    if (ret != 0) {
        // logger->error("params error");
        std::cerr << "params error" << std::endl;
        return -1;
    }

    progress_ipc_init(atp.progress_sock_path);

    output_params out_params;
    pipeline_buffer buffer;

    progress_ipc_send_stage("prepare", "running");

    // video
    progress_ipc_send_stage("video", "running");
    ret = video_strat(atp, out_params, buffer);
    if (ret != 0) {
        progress_ipc_send_stage("video", "error");
        return -1;
    }
    progress_ipc_send_stage("video", "done");

    if(atp.use_ocr){
        // ocr
        
    } else {
        // whisper
        progress_ipc_send_stage("whisper", "running");
        ret = whisper_start(atp, out_params, buffer);
        if (ret != 0) {
            progress_ipc_send_stage("whisper", "error");
            return -1;
        }
        progress_ipc_send_stage("whisper", "done");
    }
    

    // translation
    progress_ipc_send_stage("translation", "running");
    ret = translation_start(atp, buffer);
    if (ret != 0) {
        progress_ipc_send_stage("translation", "error");
        return -1; 
    }
    progress_ipc_send_stage("translation", "done");

    if(atp.output_video_path.empty()) {
        std::cerr << "video path is empty" << std::endl;
        return -1;
    }

    std::filesystem::path file(atp.video_path);
    std::string fileName = file.filename().stem().string();

    // todo: mp4 avi
    std::string output_path = atp.output_video_path + "/" + fileName + "_subtitle.mkv";

    ret = mux_video_with_ass_api(atp.video_path.c_str(), buffer, output_path.c_str());
    if (ret != 0) {
        std::cerr << "ass error" << std::endl;
        return -1;
    }

    progress_ipc_send_stage("done", "done");
    progress_ipc_send_output(output_path);
    progress_ipc_close();
    return 0;
}