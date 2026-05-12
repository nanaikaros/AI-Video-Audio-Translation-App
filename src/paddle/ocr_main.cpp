#include "../../include/ocr_main.h"

#include <filesystem>
#include <locale>
#include <codecvt>
#include <utility> 
#include <iostream>

#include "korean/korean.hpp"
#include "ocr_cache/ocr_cache.hpp"

#include "./src/api/pipelines/ocr.h"
#include "./src/pipelines/ocr/result.h"
#include "../../include/params.h"
#include "../../include/whisper_bridge.h"

#define GAP_SUBTITLE_POS 35

bool same_subtitle(const absl::optional<std::pair<float, float>>& p1,
                   const absl::optional<std::pair<float, float>>& p2) {
    if (!p1.has_value() || !p2.has_value()) return false;

    float dx = std::abs(p1->first - p2->first);
    float dy = std::abs(p1->second - p2->second);

    return dx <= GAP_SUBTITLE_POS && dy <= GAP_SUBTITLE_POS;
}

static bool text_area_small(SubtitlesEntry& e){
    int area = 6000;
    if(e.p1.has_value() && e.p2.has_value()){
        int x = e.p2.value().first - e.p1.value().first;
        int y = e.p2.value().second - e.p1.value().second;

        if(x * y < area) return true;
    }
    return false;
}

std::string normalize_text(std::string s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isspace(c)) continue;
    out.push_back((char)std::tolower(c));
  }
  return out;
}

static std::unordered_map<std::string, int> build_ngram_hist(const std::string& s, size_t n) {
  std::unordered_map<std::string, int> hist;
  if (s.empty()) return hist;

  if (s.size() < n) {
    hist[s] = 1;
    return hist;
  }

  for (size_t i = 0; i + n <= s.size(); ++i) {
    ++hist[s.substr(i, n)];
  }
  return hist;
}

static double cosine_sim_hist(const std::unordered_map<std::string, int>& a, 
    const std::unordered_map<std::string, int>& b) {
  if (a.empty() || b.empty()) return 0.0;

  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;

  for (const auto& kv : a) {
    const double va = static_cast<double>(kv.second);
    na += va * va;
    auto it = b.find(kv.first);
    if (it != b.end()) {
      dot += va * static_cast<double>(it->second);
    }
  }

  for (const auto& kv : b) {
    const double vb = static_cast<double>(kv.second);
    nb += vb * vb;
  }

  if (na <= 1e-12 || nb <= 1e-12) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

bool same_region(const std::string& a, const std::string& b) {
  std::string na = normalize_text(a);
  std::string nb = normalize_text(b);

  if (na.empty() || nb.empty()) return false;
  if (na == nb) return true;

  // 包含关系
  if(na.size() < nb.size()) std::swap(na, nb);
  if(na.find(nb) != -1) return true;

  const size_t max_len = std::max(na.size(), nb.size());
  const size_t ngram_n = (max_len <= 6) ? 1 : 2;
  const double threshold = (max_len <= 6) ? 0.92 : 0.85;

  const auto ha = build_ngram_hist(na, ngram_n);
  const auto hb = build_ngram_hist(nb, ngram_n);

  const double sim = cosine_sim_hist(ha, hb);
  return sim >= threshold;
};

bool is_target_text(std::string& str){
  if (str.empty()) return false;

  const unsigned char* s = reinterpret_cast<const unsigned char*>(str.data());
  const size_t n = str.size();

  std::string cleaned;
  cleaned.reserve(n);

  bool has_hangul = false;

  for (size_t i = 0; i < n; ) {
    const unsigned char c = s[i];

    if (c <= 0x7F) {
      // ASCII: 只保留字母数字和空格，去掉标点
      if (std::isalnum(c) || std::isspace(c)) {
        cleaned.push_back(static_cast<char>(c));
      }
      ++i;
      continue;
    }

    // 韩文三字节：保留
    if (is_hangul_syllable_utf8(s, n, i)) {
      cleaned.push_back(static_cast<char>(s[i]));
      cleaned.push_back(static_cast<char>(s[i + 1]));
      cleaned.push_back(static_cast<char>(s[i + 2]));
      has_hangul = true;
      i += 3;
      continue;
    }

    // 其他非ascii字符（含各种标点/符号）直接丢弃
    // 简单 UTF-8 跳过，避免卡死在同一字节
    if      ((c & 0xE0) == 0xC0) i += 2;
    else if ((c & 0xF0) == 0xE0) i += 3;
    else if ((c & 0xF8) == 0xF0) i += 4;
    else ++i;
  }

  str.swap(cleaned);
  return has_hangul && !str.empty();
}

static void merge_subtilte(std::vector<SubtitlesEntry>& subtitle){
  if (subtitle.empty()) return;
  std::sort(subtitle.begin(), subtitle.end(),
            [](const SubtitlesEntry& a, const SubtitlesEntry& b) {
              return a.t0_cs < b.t0_cs;
            });

  const int64_t time_gap_cs = 200; // 3s 可调

  std::vector<SubtitlesEntry> out;
  out.reserve(subtitle.size());

  for (auto &e : subtitle) {
    bool merged = false;

    // 从后往前回扫，只在时间窗口内匹配
    for (int i = (int)out.size() - 1; i >= 0; --i) {
      if (out[i].t1_cs + time_gap_cs < e.t0_cs) break;

      if (same_region(out[i].text, e.text)) {
        out[i].t1_cs = std::max(out[i].t1_cs, e.t1_cs);
        out[i].timecode =
          to_timestamp(out[i].t0_cs, true) + " --> " + to_timestamp(out[i].t1_cs, true);
        
        merged = true;
        break;
      }
    }

    if (!merged) {
      out.push_back(e);
    }
  }

  subtitle.swap(out);
}

int ocr_start(ai_translation_parmas& atp, output_params& out_params, 
  pipeline_buffer& buffer){
  PaddleOCRParams params;
  std::vector<SubtitlesEntry>& subtitle = buffer.asr_entries;
  subtitle.clear();
  int n = buffer.ocr_frames.size();

  if (load_ocr_cache(atp, subtitle)) {
    progress_ipc_send("ocr", 100);
    merge_subtilte(subtitle);
    save_ocr_cache(atp, subtitle);
    return 0;
  }

  std::filesystem::path bin = std::filesystem::absolute(atp.app_name);
  std::filesystem::path resource_root = bin.parent_path().parent_path();
  std::filesystem::path model_root = resource_root / "models";

  // text 
  params.text_detection_model_dir = (model_root / "PP-OCRv5_mobile_det_infer").string();
  params.text_recognition_model_dir = (model_root / "korean_PP-OCRv5_mobile_rec_infer").string();

  params.device = "gpu"; // 推理时使用GPU。请确保编译时添加 -DWITH_GPU=ON 选项，否则使用CPU。
  params.use_doc_orientation_classify = false;  // 不使用文档方向分类模型。
  params.use_doc_unwarping = false; // 不使用文本图像矫正模型。
  params.use_textline_orientation = false; // 不使用文本行方向分类模型。
  params.text_detection_model_name = "PP-OCRv5_mobile_det"; // 使用 PP-OCRv5_server_det 模型进行检测。
  params.text_recognition_model_name = "korean_PP-OCRv5_mobile_rec"; // 使用 PP-OCRv5_server_rec 模型进行识别。

  auto infer = My_PaddleOCR(params);

  if (n <= 0) {
    save_ocr_cache(atp, subtitle);
    progress_ipc_send("ocr", 100);
    return 0;
  }

  const int batch_size = 8;
  const int progress_den = (n > 0) ? n : 1;

  for (int batch_start = 0; batch_start < n; batch_start += batch_size) {
    int batch_end = batch_start + batch_size;
    if (batch_end > n) batch_end = n;

    std::vector<cv::Mat> batch_mat;
    batch_mat.reserve(batch_end - batch_start);
    for (int i = batch_start; i < batch_end; ++i) {
      batch_mat.emplace_back(buffer.ocr_frames[i].mat);
    }

    auto outputs = infer.Predict(batch_mat);

    int loop_count = static_cast<int>(outputs.size());
    int frame_count = batch_end - batch_start;
    if (loop_count > frame_count) loop_count = frame_count;
    
    for (int local = 0; local < loop_count; ++local) {
      int j = batch_start + local;
      auto& output = outputs[local];
      OCRResult* result = dynamic_cast<OCRResult*>(output.get());
      if(!result) continue;
      OCRPipelineResult pipeline_result = result->GetPipelineResult();
      if(pipeline_result.rec_texts.empty()) continue;

      int64_t t0 = buffer.ocr_frames[j].ts_ms;
      int64_t t1 = t0 + 33; // 默认兜底约 30fps
      
      if (atp.sample_time > 0.0) {
        t1 = t0 + static_cast<int64_t>(atp.sample_time * 1000.0);
      } else if (j + 1 < n) {
        t1 = buffer.ocr_frames[j + 1].ts_ms;
      }

      int64_t t0_cs = std::max<int64_t>(0, t0 / 10);
      int64_t t1_cs = std::max<int64_t>(t0_cs + 1, (t1 + 9) / 10);
      
      for (int i = 0; i < static_cast<int>(pipeline_result.rec_texts.size()); ++i) {
        // todo：添加多种语言判断
        if(!is_target_text(pipeline_result.rec_texts[i])) continue;

        SubtitlesEntry e;
        e.index = j;
        e.timecode = to_timestamp(t0_cs, true) + " --> " + to_timestamp(t1_cs, true);
        e.text = pipeline_result.rec_texts[i];

        auto& p = pipeline_result.rec_boxes[i];
        auto& score = pipeline_result.rec_scores[i];
        if(score < 0.90) continue;
        //   x1------------------------ ⬇️
        //   |                        |
        //   |                        |
        //   |                        |
        //   ------------------------x2
        const float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];

        const float cy = 0.5f * (y1 + y2);

        const int fh = std::max(1, buffer.ocr_frames[j].mat.rows);

        // 控制翻译区域
        if (cy < fh * 0.3) continue; // 只翻译画面下半部份
        // 控制翻译字体大小
        if(text_area_small(e)) continue;

        e.p1 = std::make_pair(p[0], p[1]);
        e.p2 = std::make_pair(p[2], p[3]);
        e.t0_cs = t0_cs;
        e.t1_cs = t1_cs;
        
        // 字幕合并
        bool merge = false;
        // const std::string e_norm = normalize_text(e.text);  
        // const std::string s_norm = normalize_text(subtitle[k].text);
        for (int k = subtitle.size() - 1; k >= 0; --k) {
          // 时间在2s内 才进行合并
          if (subtitle[k].t1_cs + 200 < e.t0_cs) continue;
          if (same_region(subtitle[k].text, e.text) &&
              same_subtitle(subtitle[k].p1, e.p1) &&
              same_subtitle(subtitle[k].p2, e.p2)) {
            merge = true;
            subtitle[k].t1_cs = std::max(subtitle[k].t1_cs, e.t1_cs);
            subtitle[k].timecode =
              to_timestamp(subtitle[k].t0_cs, true) + " --> " + to_timestamp(subtitle[k].t1_cs, true);
            break;
          }
        }
        if (!merge) subtitle.emplace_back(std::move(e));
      }
      
      progress_ipc_send("ocr", static_cast<int>((j + 1) * 100.0 / progress_den));
    }
  }

  merge_subtilte(subtitle);
  if (!save_ocr_cache(atp, subtitle)) {
    std::cerr << "warning: failed to save OCR cache" << std::endl;
  }

  return 0;
}