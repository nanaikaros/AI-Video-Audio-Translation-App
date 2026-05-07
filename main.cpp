#include "include/translation.h"
#include "include/video.h"
#include "whisper.h"
#include "whisper_bridge.h"
#include "include/params.h"
#include "include/progress_ipc.h"
#include "include/ocr_main.h"
#include "src/tool/fileloader.hpp"
#include "nlohmann/json.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

static std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());

    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c <= 0x7F) {
            if (!std::isspace(c)) {
                out.push_back(static_cast<uint32_t>(c));
            }
            ++i;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            uint32_t cp = ((c & 0x1F) << 6) |
                          (static_cast<unsigned char>(s[i + 1]) & 0x3F);
            out.push_back(cp);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(s[i + 2]) & 0x3F);
            out.push_back(cp);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            uint32_t cp = ((c & 0x07) << 18) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
                          ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(s[i + 3]) & 0x3F);
            out.push_back(cp);
            i += 4;
        } else {
            ++i;
        }
    }

    return out;
}

static std::vector<std::string> char_bigrams(const std::string& s) {
    const auto cps = utf8_to_codepoints(s);
    std::vector<std::string> grams;
    if (cps.size() < 2) return grams;
    grams.reserve(cps.size() - 1);

    for (size_t i = 0; i + 1 < cps.size(); ++i) {
        std::string g;
        g.reserve(8);
        uint32_t a = cps[i], b = cps[i + 1];
        g.append(reinterpret_cast<const char*>(&a), sizeof(a));
        g.append(reinterpret_cast<const char*>(&b), sizeof(b));
        grams.push_back(std::move(g));
    }
    return grams;
}

static double jaccard_bigrams(const std::string& a, const std::string& b) {
    auto ag = char_bigrams(a);
    auto bg = char_bigrams(b);
    if (ag.empty() || bg.empty()) return 0.0;

    std::sort(ag.begin(), ag.end());
    std::sort(bg.begin(), bg.end());

    size_t i = 0, j = 0, inter = 0, uni = 0;
    while (i < ag.size() && j < bg.size()) {
        if (ag[i] == bg[j]) { ++inter; ++uni; ++i; ++j; }
        else if (ag[i] < bg[j]) { ++uni; ++i; }
        else { ++uni; ++j; }
    }
    uni += (ag.size() - i) + (bg.size() - j);
    return uni ? (double)inter / (double)uni : 0.0;
}

static double lcs_ratio(const std::string& a, const std::string& b) {
    const auto acp = utf8_to_codepoints(a);
    const auto bcp = utf8_to_codepoints(b);
    const size_t n = acp.size();
    const size_t m = bcp.size();
    if (n == 0 || m == 0) return 0.0;

    std::vector<size_t> prev(m + 1, 0), cur(m + 1, 0);
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            if (acp[i - 1] == bcp[j - 1]) {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = std::max(prev[j], cur[j - 1]);
            }
        }
        std::swap(prev, cur);
        std::fill(cur.begin(), cur.end(), 0);
    }

    const double lcs = static_cast<double>(prev[m]);
    const double denom = static_cast<double>(std::max(n, m));
    return denom > 0 ? (lcs / denom) : 0.0;
}

static int match_subtitle_cn(
    const std::string& ocr_text,
    const std::vector<std::string>& cn_lines,
    size_t start_index,
    size_t window,
    double min_score
) {
    if (cn_lines.empty()) return -1;

    size_t best_idx = cn_lines.size();
    double best_score = 0.0;

    const size_t end = std::min(cn_lines.size(), start_index + window);
    for (size_t i = start_index; i < end; ++i) {
        const double jb = jaccard_bigrams(ocr_text, cn_lines[i]);
        const double lr = lcs_ratio(ocr_text, cn_lines[i]);
        const double score = 0.6 * jb + 0.4 * lr;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_score < min_score) return -1;
    return static_cast<int>(best_idx);
}

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
        else if (arg == "--ocr") { atp.use_ocr = true; atp.ocr_all_frames = false; atp.sample_time = 0.5f; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return -1;
        }
    }

    if (atp.video_path.empty()) return -2;
    if (atp.whisper_model_path.empty() && !atp.use_ocr) return -3;
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
        progress_ipc_send_stage("ocr", "running");
        ret = ocr_start(atp, out_params, buffer);
        if(ret != 0){
            progress_ipc_send_stage("ocr", "error");
            return -1;
        }
        progress_ipc_send_stage("ocr", "done");
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
    
    // 加载翻译好的字幕文件
    // std::vector<std::string>subtilte_doc = tool::load_txt_lines("/Users/wang/code/test/DOOKII从3分03秒cho回答是_开始.txt");
    
    // const size_t ALIGN_WINDOW = subtilte_doc.size();         // 全文搜索
    // const double ALIGN_MIN_SCORE = 0.35;                      // 阈值，根据需要调整
    // size_t doc_idx = 0;

    // for (size_t i = 0; i < buffer.asr_entries.size(); ++i) {
    //     auto &entry = buffer.asr_entries[i];
    //     const std::string &ocr_text = entry.trans_text.value();
    //     if (ocr_text.empty()) {
    //         entry.trans_text = absl::nullopt;
    //         continue;
    //     }

    //     int best = match_subtitle_cn(ocr_text, subtilte_doc, doc_idx, ALIGN_WINDOW, ALIGN_MIN_SCORE);
    //     if (best >= 0) {
    //         entry.trans_text = subtilte_doc[best];
    //         doc_idx = static_cast<size_t>(best) + 1; // 单向推进避免重复匹配
    //     } else {
    //         entry.trans_text = std::string();
    //     }
    // }

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
    std::string output_mp4_path = atp.output_video_path + "/" + fileName + "_subtitle.mp4";

    ret = mux_video_with_ass_api(atp.video_path.c_str(), buffer, output_path.c_str());
    if (ret != 0) {
        std::cerr << "ass error" << std::endl;
        return -1;
    }

    // mp4
    ret = mkv_to_mp4_with_subtitles(output_path, output_mp4_path);
    if (ret != 0) {
        std::cerr << "mp4 error" << std::endl;
        return -1;
    }

    progress_ipc_send_output(output_path);
    progress_ipc_send_stage("done", "done");
    progress_ipc_close();
    return 0;
}