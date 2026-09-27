// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cxx/private/path.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <system_error>
#include <vector>

namespace cxx {

struct DirectoryVersion {
  int major{};
  std::optional<int> minor;
  std::optional<int> patch;

  [[nodiscard]] static auto parse(const std::string& text)
      -> std::optional<DirectoryVersion> {
    static const std::regex versionRe(R"(^(\d+)(?:\.(\d+))?(?:\.(\d+))?$)");
    std::smatch match;
    if (!std::regex_match(text, match, versionRe)) return std::nullopt;

    DirectoryVersion version;
    version.major = std::stoi(match[1].str());
    if (match[2].matched) version.minor = std::stoi(match[2].str());
    if (match[3].matched) version.patch = std::stoi(match[3].str());
    return version;
  }

  [[nodiscard]] auto operator<(const DirectoryVersion& other) const -> bool {
    if (major != other.major) return major < other.major;

    const auto maxInt = std::numeric_limits<int>::max();

    if (minor != other.minor) {
      return minor.value_or(maxInt) < other.minor.value_or(maxInt);
    }

    return patch.value_or(maxInt) < other.patch.value_or(maxInt);
  }
};

[[nodiscard]] inline auto versionedSubdirectories(const fs::path& dir)
    -> std::vector<fs::path> {
  struct VersionedPath {
    fs::path path;
    DirectoryVersion version;
  };

  std::vector<VersionedPath> entries;

  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (!entry.is_directory()) continue;
    auto version = DirectoryVersion::parse(entry.path().filename().string());
    if (!version) continue;
    entries.push_back({entry.path(), *version});
  }

  std::ranges::sort(entries, [](const auto& a, const auto& b) {
    return b.version < a.version;
  });

  std::vector<fs::path> paths;
  for (auto& entry : entries) paths.push_back(std::move(entry.path));
  return paths;
}

}  // namespace cxx
