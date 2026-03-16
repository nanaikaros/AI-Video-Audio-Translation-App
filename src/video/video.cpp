#include "video.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/mathematics.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/log.h>
}

static std::string fferr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(buf, AV_ERROR_MAX_STRING_SIZE, errnum);
    return std::string(buf);
}

static std::string ass_time_to_string(int centiseconds) {
    int cs = std::max(0, centiseconds);
    int hours = cs / 360000;
    int minutes = (cs % 360000) / 6000;
    int seconds = (cs % 6000) / 100;
    int frac = cs % 100;

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d:%02d:%02d.%02d", hours, minutes, seconds, frac);
    return std::string(buf);
}

/**
 * Extract audio pcm data
 * 
 * @param in_video input video
 * @param pcm_out pcm data output in vector
 */
int video_extract_audio_pcm_16k(const std::string& in_video, std::vector<float>& pcm_out) {
    pcm_out.clear();

    int ret = 0;
    AVFormatContext* inFmt = nullptr;
    AVCodecContext* decCtx = nullptr;
    SwrContext* swr = nullptr;
    AVFrame* inFrame = nullptr;
    AVPacket* inPkt = nullptr;
    int audioStream = -1;

    if ((ret = avformat_open_input(&inFmt, in_video.c_str(), nullptr, nullptr)) < 0) goto end;
    if ((ret = avformat_find_stream_info(inFmt, nullptr)) < 0) goto end;

    ret = av_find_best_stream(inFmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (ret < 0) goto end;
    audioStream = ret;

    {
        AVStream* st = inFmt->streams[audioStream];
        const AVCodec* dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec) { ret = AVERROR_DECODER_NOT_FOUND; goto end; }

        decCtx = avcodec_alloc_context3(dec);
        if (!decCtx) { ret = AVERROR(ENOMEM); goto end; }

        if ((ret = avcodec_parameters_to_context(decCtx, st->codecpar)) < 0) goto end;
        if ((ret = avcodec_open2(decCtx, dec, nullptr)) < 0) goto end;
    }

    {
        AVChannelLayout dst_ch;
        av_channel_layout_default(&dst_ch, 1); // mono
        // audio resmample
        ret = swr_alloc_set_opts2(
            &swr,
            &dst_ch, AV_SAMPLE_FMT_FLT, 16000,
            &decCtx->ch_layout, decCtx->sample_fmt, decCtx->sample_rate,
            0, nullptr
        );
        av_channel_layout_uninit(&dst_ch);
        if (ret < 0 || !swr) { ret = AVERROR_UNKNOWN; goto end; }
        if ((ret = swr_init(swr)) < 0) goto end;
    }

    inFrame = av_frame_alloc();
    inPkt = av_packet_alloc();
    if (!inFrame || !inPkt) { ret = AVERROR(ENOMEM); goto end; }

    while ((ret = av_read_frame(inFmt, inPkt)) >= 0) {
        if (inPkt->stream_index != audioStream) { av_packet_unref(inPkt); continue; }

        if ((ret = avcodec_send_packet(decCtx, inPkt)) < 0) { av_packet_unref(inPkt); goto end; }
        av_packet_unref(inPkt);

        while (true) {
            ret = avcodec_receive_frame(decCtx, inFrame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) goto end;

            int dst_nb = (int)av_rescale_rnd(
                swr_get_delay(swr, decCtx->sample_rate) + inFrame->nb_samples,
                16000, decCtx->sample_rate, AV_ROUND_UP
            );
            std::vector<float> tmp(dst_nb);
            uint8_t* out_data[1] = { reinterpret_cast<uint8_t*>(tmp.data()) };

            int n = swr_convert(
                swr,
                out_data, dst_nb,
                (const uint8_t**)inFrame->extended_data, inFrame->nb_samples
            );
            if (n < 0) { ret = n; goto end; }

            pcm_out.insert(pcm_out.end(), tmp.begin(), tmp.begin() + n);
            av_frame_unref(inFrame);
        }
    }
    if (ret == AVERROR_EOF) ret = 0;

end:
    if (inPkt) av_packet_free(&inPkt);
    if (inFrame) av_frame_free(&inFrame);
    if (swr) swr_free(&swr);
    if (decCtx) avcodec_free_context(&decCtx);
    if (inFmt) avformat_close_input(&inFmt);
    return ret;
}

int video_strat(ai_translation_parmas& atp, output_params& out, pipeline_buffer& buffer) {
    std::vector<float>& pcm = buffer.pcm_mono_16k;

    if (atp.video_path.empty()) {
        std::cerr << "video path is empty" << std::endl;
        return -1;
    }

    out.video_mkv_path = atp.video_path;
    out.audio_mka_path.clear();

    int ret = video_extract_audio_pcm_16k(atp.video_path, pcm);
    if (ret < 0) {
        std::cerr << "extract pcm failed: " << fferr(ret) << std::endl;
        return -1;
    }

    if (pcm.empty()) {
        std::cerr << "extract pcm empty" << std::endl;
        return -1;
    }

    return 0;
}

/**
 * Timecode(ASS): 0:00:00.00 --> 0:00:00.00
 *            or: 0:00:00.00,0:00:00.00
 * 
 * @param tc timecode
 * @param t0_ms start
 * @param t1_ms end
 * 
 * @return centiseconds
 */
static bool parse_ass_timecode_ms(const std::string& tc, int64_t& t0_cs, int64_t& t1_cs) {
    auto trim = [](const std::string& s) -> std::string {
        size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    };

    auto parse_one = [&](const std::string& in, int64_t& out_cs) -> bool {
        std::string s = trim(in);
        int h = 0, m = 0, sec = 0;

        size_t p1 = s.find(':');
        size_t p2 = (p1 == std::string::npos) ? std::string::npos : s.find(':', p1 + 1);
        if (p1 == std::string::npos || p2 == std::string::npos) return false;

        size_t pf = s.find_first_of(".,", p2 + 1);
        try {
            h   = std::stoi(s.substr(0, p1));
            m   = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));
            sec = std::stoi(s.substr(p2 + 1, (pf == std::string::npos ? s.size() : pf) - (p2 + 1)));
        } catch (...) { return false; }

        int cs = 0;
        if (pf != std::string::npos) {
            std::string fs = s.substr(pf + 1);
            if (!fs.empty() && std::isdigit((unsigned char)fs[0])) cs += (fs[0] - '0') * 10;
            if (fs.size() > 1 && std::isdigit((unsigned char)fs[1])) cs += (fs[1] - '0');
        }

        out_cs = ((int64_t)h * 3600 + (int64_t)m * 60 + sec) * 100 + cs;
        return true;
    };

    std::string s = trim(tc);
    size_t p = s.find("-->");
    if (p == std::string::npos) return false;

    std::string lhs = trim(s.substr(0, p));
    std::string rhs = trim(s.substr(p + 3));
    if (!parse_one(lhs, t0_cs)) return false;
    if (!parse_one(rhs, t1_cs)) return false;

    if (t1_cs <= t0_cs) t1_cs = t0_cs + 100;
    return true;
}

/**
 * write_ass_from_entry
 */
static int write_ass_from_entry(AVFormatContext* ofmt, int s_out_idx, const ASSDialog& e, int64_t& last_pts){
    int start = e.start, end = e.end;
    if(end <= start) {
        end = start + 1;
    }

    AVStream* st = ofmt->streams[s_out_idx];
    int64_t pts = av_rescale_q(start, AVRational{1, 100}, st->time_base);
    int64_t dur = av_rescale_q(std::max<int64_t>(1, end - start), AVRational{1, 100}, st->time_base);

    // add
    if (last_pts != AV_NOPTS_VALUE && pts <= last_pts) {
        pts = last_pts + 1;
    }
    last_pts = pts;

    std::string text = e.text ? e.text : "";
    if(text.empty()) text = " ";

    for (size_t pos = 0; (pos = text.find('\n', pos)) != std::string::npos; ) {
        text.replace(pos, 1, "\\N");
        pos += 2;
    }

    // Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    static int read_order = 0;
    std::string payload = std::to_string(read_order++) + ",0,Default,,0,0,0,," + text;
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return AVERROR(ENOMEM);

    int ret = av_new_packet(pkt, payload.size());
    if (ret < 0) {
        av_packet_free(&pkt);
        return ret;
    }

    memcpy(pkt->data, payload.data(), payload.size());

    pkt->stream_index = s_out_idx;
    pkt->pts = pts;
    pkt->dts = pts;
    pkt->duration = std::max<int64_t>(1, dur);

    ret = av_interleaved_write_frame(ofmt, pkt);
    av_packet_free(&pkt);

    if (ret < 0) {
        std::cerr << "[mux] write subtitle failed: " << fferr(ret) << "\n";
    }
    return ret;
}

std::vector<ASSDialog> srt_to_ass(std::vector<SubtitlesEntry>& entry){
    int n = entry.size();
    std::vector<ASSDialog> ass;
    ass.reserve(n);

    for(int i = 0; i < n; ++i){
        int64_t t0 = 0, t1 = 0;
        if(!parse_ass_timecode_ms(entry[i].timecode, t0, t1)){
            std::cerr << "[ass] skip bad timecode: " << entry[i].timecode << "\n";
            continue;
        }

        ASSDialog d{};

        d.start = t0;
        d.end = t1;
        
        d.text = const_cast<char *>(entry[i].text.c_str());
        ass.push_back(d);
    }

    return ass;
} 

/**
 * defalut header
 */
static int set_default_ass_header(AVStream* s_st, int x, int y) {
    const int base = std::max(1, std::min(x, y));
    const int fontSize = std::clamp((int)std::lround(y * 0.06), 24, 72);
    const int marginV  = std::clamp((int)std::lround(y * 0.03),  12, 80);
    const int marginLR = std::clamp((int)std::lround(x * 0.015), 10, 60);
    const int outline  = std::clamp((int)std::lround(base * 0.0025), 1, 4);
    const int shadow   = std::clamp((int)std::lround(base * 0.0015), 0, 3);

    std::string hdr =
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: " + std::to_string(x) + "\n"
        "PlayResY: " + std::to_string(y) + "\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial," + std::to_string(fontSize) +
        ",&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1," +
        std::to_string(outline) + "," + std::to_string(shadow) + ",2," +
        std::to_string(marginLR) + "," + std::to_string(marginLR) + "," + std::to_string(marginV) + ",1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

    size_t n = hdr.size();
    uint8_t* extra = (uint8_t*)av_mallocz(n + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!extra) return AVERROR(ENOMEM);

    std::memcpy(extra, hdr.data(), n);
    s_st->codecpar->extradata = extra;
    s_st->codecpar->extradata_size = (int)n;
    return 0;
}

/**
 * Embed subtitles into the video
 *
 * @param video_path
 * @param buffer subtitles
 * @param out_video_path output video path
 */
int mux_video_with_ass_api(const char* video_path, pipeline_buffer& buffer, const char* out_video_path) {
    AVFormatContext *vfmt = nullptr, *ofmt = nullptr;
    AVPacket *pkt = nullptr;
    int ret = 0;
    int width = 1920, height = 1080;
    
    std::vector<int> vmap;
    int subtitle_idx = -1;
    int entries_idx = 0;

    int copied_av_streams = 0;
    int64_t written_av_packets = 0;

    int64_t last_pts = AV_NOPTS_VALUE;

    // subtitles gen
    const std::vector<ASSDialog>& entries = srt_to_ass(buffer.subtitles_entries);
    if (entries.empty()) {
        std::cerr << "[mux] no subtitle entries\n";
        return AVERROR(EINVAL);
    }

    if (!video_path || !out_video_path) return AVERROR(EINVAL);

    if ((ret = avformat_open_input(&vfmt, video_path, nullptr, nullptr)) < 0) goto end;
    if ((ret = avformat_find_stream_info(vfmt, nullptr)) < 0) goto end;

    if ((ret = avformat_alloc_output_context2(&ofmt, nullptr, "matroska", out_video_path)) < 0) 
        goto end;
    
    vmap.assign(vfmt->nb_streams, -1);
    for (unsigned i = 0; i < vfmt->nb_streams; ++i) {
        AVStream* in_st = vfmt->streams[i];
        AVMediaType mt = in_st->codecpar->codec_type;
        if (mt == AVMEDIA_TYPE_VIDEO && in_st->codecpar->width > 0 
                && in_st->codecpar->height > 0) {
            width = in_st->codecpar->width;
            height = in_st->codecpar->height;
        }
        if (mt != AVMEDIA_TYPE_VIDEO && mt != AVMEDIA_TYPE_AUDIO) continue;

        AVStream* out_st = avformat_new_stream(ofmt, nullptr);
        if (!out_st) { ret = AVERROR(ENOMEM); goto end; }

        if ((ret = avcodec_parameters_copy(out_st->codecpar, in_st->codecpar)) < 0) goto end;
        out_st->codecpar->codec_tag = 0;
        out_st->time_base = in_st->time_base;
        vmap[i] = out_st->index;
        copied_av_streams++;
    }

    if (copied_av_streams == 0) {
        std::cerr << "[mux] no audio/video stream copied\n";
        ret = AVERROR_STREAM_NOT_FOUND;
        goto end;
    }

    // subtitle stream
    {
        AVStream* subtitle_st = avformat_new_stream(ofmt, nullptr);
        if (!subtitle_st) { ret = AVERROR(ENOMEM); goto end; }
        subtitle_st->time_base = AVRational{1, 100};
        subtitle_st->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;
        //ass
        subtitle_st->codecpar->codec_id = AV_CODEC_ID_ASS;
        subtitle_st->codecpar->codec_tag = 0;
    
        if ((ret = set_default_ass_header(subtitle_st, width, height)) < 0) goto end;
        // mov text
        // s_st->codecpar->codec_id = AV_CODEC_ID_MOV_TEXT;
        // s_st->codecpar->codec_tag = MKTAG('t','x','3','g');
        subtitle_st->disposition |= AV_DISPOSITION_DEFAULT | AV_DISPOSITION_FORCED;
        av_dict_set(&subtitle_st->metadata, "language", "zho", 0);
        subtitle_idx = subtitle_st->index;
    }

    if (!(ofmt->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&ofmt->pb, out_video_path, AVIO_FLAG_WRITE)) < 0) goto end;
    }

    if ((ret = avformat_write_header(ofmt, nullptr)) < 0) goto end;

    pkt = av_packet_alloc();
    if (!pkt) { ret = AVERROR(ENOMEM); goto end; }

    while ((ret = av_read_frame(vfmt, pkt)) >= 0) {
        int in_idx = pkt->stream_index;
        int out_idx = (in_idx >= 0 && in_idx < (int)vmap.size()) ? vmap[in_idx] : -1;
        if(out_idx < 0) {
            av_packet_unref(pkt);
            continue;
        }

        AVStream* in_st = vfmt->streams[in_idx];
        int64_t ts = (pkt->pts != AV_NOPTS_VALUE) ? pkt->pts : pkt->dts;
        int64_t cur_cs = (ts == AV_NOPTS_VALUE)
            ? INT64_MIN
            : av_rescale_q(ts, in_st->time_base, AVRational{1, 100});

        while(entries_idx < entries.size() && 
                cur_cs != INT64_MIN && entries[entries_idx].start <= cur_cs){
            int r = write_ass_from_entry(ofmt, subtitle_idx, entries[entries_idx], last_pts);
            if (r < 0) { ret = r; goto end; }
            entries_idx++;
        }

        
        AVStream* out_st = ofmt->streams[out_idx];
        av_packet_rescale_ts(pkt, in_st->time_base, out_st->time_base);
        pkt->stream_index = out_idx;
        ret = av_interleaved_write_frame(ofmt, pkt);
        if (ret < 0) { av_packet_unref(pkt); goto end; }
        written_av_packets++;
        
        av_packet_unref(pkt);
    }

    for (int i = entries_idx; i < entries.size(); ++i) {
        int r = write_ass_from_entry(ofmt, subtitle_idx, entries[i], last_pts);
        if (r < 0) { ret = r; goto end; }
    }
    
    if (ret == AVERROR_EOF) ret = 0;
    if (ret < 0) goto end;

    if (written_av_packets == 0) {
        std::cerr << "[mux] no av packet written\n";
        ret = AVERROR_INVALIDDATA;
        goto end;
    }

    ret = av_write_trailer(ofmt);

end:
    if (pkt) av_packet_free(&pkt);
    if (vfmt) avformat_close_input(&vfmt);
    if (ofmt) {
        if (!(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb) avio_closep(&ofmt->pb);
        avformat_free_context(ofmt);
    }
    return ret;
}

