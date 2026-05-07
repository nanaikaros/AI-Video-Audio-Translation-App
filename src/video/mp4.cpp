#include "video.h"

#include <cstdlib>
#include <sstream>
#include <string>

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string escape_ffmpeg_filter_path(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char c : s) {
        if (c == '\\' || c == ':' || c == '\'' || c == ',' || c == '[' || c == ']') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

int mkv_to_mp4_with_subtitles(const std::string& input_mkv,
                              const std::string& output_mp4) {
    if (input_mkv.empty() || output_mp4.empty()) {
        return -1;
    }

    const std::string filter =
        "subtitles=filename=" + escape_ffmpeg_filter_path(input_mkv) + ":stream_index=0";

    const std::string cmd_str = 
        "ffmpeg -i " + shell_quote(input_mkv)
        + " -vf " + shell_quote(filter)
        + " -c:v libx264 -crf 18 -preset medium -c:a copy "
        + shell_quote(output_mp4)
        + " -y";

    std::string cmd = "/bin/bash -lc " + shell_quote(cmd_str);

    return std::system(cmd.c_str());
}