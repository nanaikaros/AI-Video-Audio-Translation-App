#include "common/common.h"
#include "common/common-whisper.h"
#include "grammar-parser.h"

#include <whisper.h>
#include "../include/whisper_bridge.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <cfloat>
#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// command-line parameters
struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.6f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_confidence= false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = true;
    int32_t gpu_device   = 0;
    bool suppress_nst    = false;
    bool carry_initial_prompt = false;

    std::string language  = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model     = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;

    // Voice Activity Detection (VAD) parameters
    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
};

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

static std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

static void whisper_print_progress_callback(struct whisper_context * /*ctx*/,
                                            struct whisper_state * /*state*/,
                                            int progress,
                                            void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev = &(((whisper_print_user_data *) user_data)->progress_prev);

    if (progress >= *progress_prev + progress_step || progress == 100) {
        *progress_prev = progress;
        progress_ipc_send("whisper", progress);

        if (progress == 100) {}
    }
}

static void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;
    
    if (params.no_prints) {
        return; 
    }
    
    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                const int n_colors = (int) k_colors.size();
                int raw_col = (int) (std::pow(p, 3)*float(n_colors));
                if (raw_col < 0) raw_col = 0;
                if (raw_col > n_colors - 1) raw_col = n_colors - 1;
                const int col = raw_col;

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else if (params.print_confidence) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                int style_idx = 2;     // High confidence - dim
                if (p < 0.33) {
                    style_idx = 0;     // Low confidence - inverse (highlighted)
                } else if (p < 0.66) {
                    style_idx = 1;     // Medium confidence - underlined
                }
                printf("%s%s%s%s", speaker.c_str(), k_styles[style_idx].c_str(), text, "\033[0m");
            }
        } else {
            const char * text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

static void output_srt(struct whisper_context * ctx, std::ofstream & fout, const whisper_params & params, std::vector<std::vector<float>> pcmf32s) {
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        std::string speaker = "";

        if (params.diarize && pcmf32s.size() == 2)
        {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        fout << i + 1 + params.offset_n << "\n";
        fout << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
        fout << speaker << text << "\n\n";
    }
}

static void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

int whisper_start(ai_translation_parmas& atp, output_params& out, pipeline_buffer& buffer) {
    ggml_backend_load_all();

    if (buffer.pcm_mono_16k.empty()) {
        fprintf(stderr, "error: pcm buffer is empty\n");
        return -1;
    }
    if (atp.whisper_model_path.empty()) {
        fprintf(stderr, "error: whisper model path is empty\n");
        return -1;
    }

    whisper_params params;
    params.n_threads      = atp.thread_num > 0 ? (int)atp.thread_num : 8;
    params.n_processors   = 4;
    params.model          = atp.whisper_model_path;
    params.language       = "auto";
    params.no_prints      = true;
    params.print_progress = true;
    params.output_srt     = false;

    if (params.no_prints) {
        whisper_log_set(cb_log_disable, nullptr);
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.gpu_device = params.gpu_device;
    cparams.flash_attn = params.flash_attn;

    whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return -1;
    }

    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!whisper_is_multilingual(ctx)) {
        params.language = "en";
        params.translate = false;
    }

    std::vector<float> pcmf32 = buffer.pcm_mono_16k;
    std::vector<std::vector<float>> pcmf32s = { pcmf32 };

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = params.print_progress;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.print_special    = params.print_special;
    wparams.translate        = params.translate;
    wparams.language         = params.language.c_str();
    wparams.detect_language  = params.detect_language;
    wparams.n_threads        = params.n_threads;
    wparams.no_timestamps    = params.no_timestamps;

    whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

    wparams.new_segment_callback           = whisper_print_segment_callback;
    wparams.new_segment_callback_user_data = &user_data;

    if (wparams.print_progress) {
        wparams.progress_callback           = whisper_print_progress_callback;
        wparams.progress_callback_user_data = &user_data;
    }

    int ret = whisper_full_parallel(ctx, wparams, pcmf32.data(), (int)pcmf32.size(), params.n_processors);
    if (ret != 0) {
        whisper_free(ctx);
        return -1;
    }

    buffer.asr_entries.clear();
    const int n_segments = whisper_full_n_segments(ctx);
    buffer.asr_entries.reserve(n_segments);

    for (int i = 0; i < n_segments; ++i) {
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        SubtitlesEntry e;
        e.index = i + 1;
        e.timecode = to_timestamp(t0, true) + " --> " + to_timestamp(t1, true);
        e.text = whisper_full_get_segment_text(ctx, i) ? whisper_full_get_segment_text(ctx, i) : "";
        buffer.asr_entries.push_back(std::move(e));
    }

    if (!params.no_prints) {
        whisper_print_timings(ctx);
    }

    whisper_free(ctx);
    return 0;
}
