#pragma once
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <opencv2/opencv.hpp>

/**
 * srt结构体
 * 索引 时间戳 文本
 */
struct SubtitlesEntry {
    int index = 0;
    std::string timecode;
    std::string text;
    absl::optional<std::pair<float, float>> p1 = absl::nullopt; // 对角线 {x, y}
    absl::optional<std::pair<float, float>> p2 = absl::nullopt;
    int64_t t0_cs;
    int64_t t1_cs;
};

struct ai_translation_parmas {
    std::string app_name;
    std::string video_path;
    std::string whisper_model_path;
    std::string translation_model_path;

    std::string output_video_path;

    size_t thread_num;
    std::string progress_sock_path;

    bool use_ocr = false;
    bool ocr_all_frames = false;   // full frame
    double sample_time = 1;      // sample frame default 1 s
};

struct OcrFrame {
    int width = 0;
    int height = 0;
    int linesize = 0;         // rgb byte row
    int64_t ts_ms = 0;        
    std::vector<uint8_t> rgb; // linesize * height
    cv::Mat mat;
};

struct pipeline_buffer {
    std::vector<float> pcm_mono_16k;   
    std::vector<SubtitlesEntry> asr_entries; // 待翻译
    std::vector<SubtitlesEntry> subtitles_entries; // 翻译后
    std::vector<OcrFrame> ocr_frames;
};

struct output_params {
    std::string video_mkv_path;
    std::string audio_mka_path;

    std::string srt_path;
};

struct glossary_pair {
    std::string src;
    std::string dst;
};