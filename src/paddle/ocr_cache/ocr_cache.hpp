#include <filesystem>
#include <fstream>
#include <iostream>

#include "third_party/nlohmann/json.hpp"

#include "../../include/whisper_bridge.h"
#include "../../include/params.h"

using json = nlohmann::json;

// build the cache json
static std::filesystem::path build_ocr_cache_path(const ai_translation_parmas& atp) {
  std::filesystem::path video(atp.video_path);
  std::string file_stem = video.stem().string(); // 不含后缀
  return video.parent_path() / std::filesystem::path(".temp") / file_stem / "ocr.json";
}

static json subtitle_entry_to_json(const SubtitlesEntry& e) {
  json j;
  j["index"] = e.index;
  j["timecode"] = e.timecode;
  j["text"] = e.text;
  j["t0_cs"] = e.t0_cs;
  j["t1_cs"] = e.t1_cs;

  if (e.p1.has_value()) {
    j["p1"] = { e.p1->first, e.p1->second };
  } else {
    j["p1"] = nullptr;
  }

  if (e.p2.has_value()) {
    j["p2"] = { e.p2->first, e.p2->second };
  } else {
    j["p2"] = nullptr;
  }

  if(e.trans_text.has_value()){
    j["trans_text"] = e.trans_text;
  } else {
    j["trans_text"] = nullptr;
  }

  return j;
}

static bool subtitle_entry_from_json(const json& j, SubtitlesEntry& e) {
  try {
    e.index = j.value("index", 0);
    e.timecode = j.value("timecode", "");
    e.text = j.value("text", "");
    e.t0_cs = j.value("t0_cs", int64_t(0));
    e.t1_cs = j.value("t1_cs", int64_t(0));

    if (j.contains("p1") && !j["p1"].is_null() && j["p1"].is_array() && j["p1"].size() == 2) {
      e.p1 = std::make_pair(j["p1"][0].get<float>(), j["p1"][1].get<float>());
    } else {
      e.p1 = absl::nullopt;
    }

    if (j.contains("p2") && !j["p2"].is_null() && j["p2"].is_array() && j["p2"].size() == 2) {
      e.p2 = std::make_pair(j["p2"][0].get<float>(), j["p2"][1].get<float>());
    } else {
      e.p2 = absl::nullopt;
    }

    if (e.timecode.empty()) {
      e.timecode = to_timestamp(e.t0_cs, true) + " --> " + to_timestamp(e.t1_cs, true);
    }

    if(j.contains("trans_text") && !j["trans_text"].is_null()){
      e.trans_text = j.value("trans_text", "");
    } else {
      e.trans_text = absl::nullopt;
    }

    return true;
  } catch (...) {
    return false;
  }
}

static bool load_ocr_cache(const ai_translation_parmas& atp, std::vector<SubtitlesEntry>& subtitle) {
  const auto cache_path = build_ocr_cache_path(atp);
  if (!std::filesystem::exists(cache_path)) return false;

  std::ifstream fin(cache_path);
  if (!fin.is_open()) return false;

  json root;
  try {
    fin >> root;
  } catch (...) {
    return false;
  }

  if (!root.contains("entries") || !root["entries"].is_array()) return false;

  std::vector<SubtitlesEntry> loaded;
  loaded.reserve(root["entries"].size());

  for (const auto& item : root["entries"]) {
    SubtitlesEntry e;
    if (!subtitle_entry_from_json(item, e)) return false;
    loaded.emplace_back(std::move(e));
  }

  subtitle = std::move(loaded);
  return true;
}

static bool save_ocr_cache(const ai_translation_parmas& atp, const std::vector<SubtitlesEntry>& subtitle) {
  const auto cache_path = build_ocr_cache_path(atp);
  std::error_code ec;
  std::filesystem::create_directories(cache_path.parent_path(), ec);
  if (ec) return false;

  json root;
  root["version"] = 1;
  root["video_path"] = atp.video_path;
  root["entries"] = json::array();

  for (const auto& e : subtitle) {
    root["entries"].push_back(subtitle_entry_to_json(e));
  }

  std::ofstream fout(cache_path);
  if (!fout.is_open()) return false;

  // strict 会抛异常；replace 会替换非法 UTF-8
  fout << root.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
  return static_cast<bool>(fout);
}