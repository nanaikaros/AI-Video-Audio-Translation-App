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

    // PaddleOCR Predict -> pipeline_infer_ Predict(input) -> infer_ Predict;
    // auto outputs = infer.Predict("/Users/wang/code/test/models/general_ocr_002.png");

    std::vector<cv::Mat> mat;
    // mat.resize(n);

    for(int i = 0; i < n; ++i){
      mat.push_back(buffer.ocr_frames[i].mat);
    }
    auto outputs = infer.Predict(mat);
    
    for (int j = 0; j < outputs.size(); ++j) {
      auto& output = outputs[j];
      OCRResult* result = dynamic_cast<OCRResult*>(output.get());
      if(result == nullptr) continue;
      OCRPipelineResult pipeline_result = result->GetPipelineResult();
      if(pipeline_result.rec_texts.size() == 0) continue;
      
      int64_t t0 = buffer.ocr_frames[j].ts_ms;
      for(int i = 0; i < pipeline_result.rec_texts.size(); ++i){
        SubtitlesEntry e;
        e.index = j;
        int t1 = (j + 1 < n) ? buffer.ocr_frames[j + 1].ts_ms : (t0 + 1000);
        e.timecode = to_timestamp(t0, true) + " --> " + to_timestamp(t1, true);
        e.text = pipeline_result.rec_texts[i];
        if (i >= pipeline_result.rec_boxes.size()) continue;
        auto& p = pipeline_result.rec_boxes[i];
        e.p1 = std::make_pair(p[0], p[1]);
        e.p2 = std::make_pair(p[2], p[3]);
        subtitle.emplace_back(std::move(e));
      }
    }

    // debug
    int a = 0;

    return 0;
}