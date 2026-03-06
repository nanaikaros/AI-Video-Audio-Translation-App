#pragma once
#include <string>
#include <vector>

/**
 * srt结构体
 * 索引 时间戳 文本
 */
struct SubtitlesEntry {
    int index = 0;
    std::string timecode;
    std::string text;
};

struct ai_translation_parmas {
    std::string app_name;
    std::string video_path;
    std::string whisper_model_path;
    std::string translation_model_path;

    std::string output_video_path;

    size_t thread_num;
    std::string progress_sock_path;
};

struct pipeline_buffer {
    std::vector<float> pcm_mono_16k;   
    std::vector<SubtitlesEntry> asr_entries;
    std::vector<SubtitlesEntry> subtitles_entries;
};

struct output_params {
    std::string video_mkv_path;
    std::string audio_mka_path;

    std::string srt_path;
};