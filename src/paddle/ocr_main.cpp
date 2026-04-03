#include "../../include/ocr_main.h"

#include <filesystem>
#include "./src/api/pipelines/ocr.h"
#include "./src/pipelines/ocr/result.h"
#include "../../include/params.h"
#include "../../include/whisper_bridge.h"

int ocr_start(ai_translation_parmas& atp, output_params& out_params, 
    pipeline_buffer& buffer){
    PaddleOCRParams params;
    std::vector<SubtitlesEntry>& subtitle = buffer.asr_entries;
    int n = buffer.ocr_frames.size();

    // namespace fs = std::filesystem;
    // fs::path(__FILE__).parent_path();

    // text 
    params.text_detection_model_dir = "/Users/wang/code/test/src/paddle/models/PP-OCRv5_server_det_infer";
    params.text_recognition_model_dir = "/Users/wang/code/test/src/paddle/models/korean_PP-OCRv5_mobile_rec_infer";

    params.device = "gpu"; // 推理时使用GPU。请确保编译时添加 -DWITH_GPU=ON 选项，否则使用CPU。
    params.use_doc_orientation_classify = false;  // 不使用文档方向分类模型。
    params.use_doc_unwarping = false; // 不使用文本图像矫正模型。
    params.use_textline_orientation = false; // 不使用文本行方向分类模型。
    params.text_detection_model_name = "PP-OCRv5_server_det"; // 使用 PP-OCRv5_server_det 模型进行检测。
    params.text_recognition_model_name = "korean_PP-OCRv5_mobile_rec"; // 使用 PP-OCRv5_server_rec 模型进行识别。
    // params.text_recognition_batch_size = 16;
    
    auto infer = My_PaddleOCR(params);

    if (n <= 0) {
      progress_ipc_send("ocr", 100);
      return 0;
    }

    const int batch_size = 4;
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
      if (result != nullptr) {
        OCRPipelineResult pipeline_result = result->GetPipelineResult();

        if (!pipeline_result.rec_texts.empty()) {
          int64_t t0 = buffer.ocr_frames[j].ts_ms;
          int64_t t1 = (j + 1 < n)
            ? buffer.ocr_frames[j + 1].ts_ms
            : (t0 + atp.sample_time * 1000);

          for (int i = 0; i < static_cast<int>(pipeline_result.rec_texts.size()); ++i) {
            if (i >= static_cast<int>(pipeline_result.rec_boxes.size())) continue;

            SubtitlesEntry e;
            e.index = j;
            e.timecode = to_timestamp(t0, true) + " --> " + to_timestamp(t1, true);
            e.text = pipeline_result.rec_texts[i];

            auto& p = pipeline_result.rec_boxes[i];
            e.p1 = std::make_pair(p[0], p[1]);
            e.p2 = std::make_pair(p[2], p[3]);

            subtitle.emplace_back(std::move(e));
          }
        }
      }
      progress_ipc_send("ocr", static_cast<int>((j + 1) * 100.0 / progress_den));
    }
  }
    return 0;
}