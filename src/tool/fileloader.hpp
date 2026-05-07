#pragma once

#include <zip.h>

#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <fstream>

namespace tool {

static inline std::string trim_copy(std::string s) {
	auto not_space = [](unsigned char c) { return !std::isspace(c); };
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
	s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
	return s;
}

static inline std::vector<std::string> load_txt_lines(const std::string & path) {
	std::vector<std::string> lines;
	std::ifstream fin(path);
	if (!fin.is_open()) return lines;

	std::string line;
	while (std::getline(fin, line)) {
		line = trim_copy(line);
		if (!line.empty()) {
			lines.push_back(line);
		}
	}

	return lines;
}

} // namespace tool
