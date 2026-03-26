#pragma once
#include "../include/params.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include "ass_split.h"
}

#include <iostream>
#include <string>

int video_strat(ai_translation_parmas&, output_params&, pipeline_buffer&);
int mux_video_with_ass_api(const char* video_path, pipeline_buffer&, 
    const char* out_video_path);

int video_extract_picture(const std::string& in_video,
                          std::vector<OcrFrame>& frames_out,
                          int interval_sec = 1);