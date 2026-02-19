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

#include <cxx/macos_toolchain.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <algorithm>
#include <format>
#include <ranges>
#include <regex>

namespace cxx {

MacOSToolchain::MacOSToolchain(Preprocessor* preprocessor, std::string arch)
    : Toolchain(preprocessor), arch_(std::move(arch)) {
  std::string xcodeContentsBasePath = "/Applications/Xcode.app/Contents";

  platformPath_ = std::format(
      "{}/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX.sdk",
      xcodeContentsBasePath);

  toolchainPath_ =
      std::format("{}/Developer/Toolchains/XcodeDefault.xctoolchain",
                  xcodeContentsBasePath);

  if (arch_ == "aarch64") {
    memoryLayout()->setSizeOfLongDouble(8);
    memoryLayout()->setTriple("arm64-apple-macosx15.0.0");
  } else if (arch_ == "x86_64") {
    memoryLayout()->setSizeOfLongDouble(16);
    memoryLayout()->setTriple("x86_64-apple-macosx15.0.0");
  } else {
    cxx_runtime_error(std::format("Unsupported architecture: {}", arch_));
  }
}

void MacOSToolchain::setSysroot(std::string sysroot) {
  sysroot_ = std::move(sysroot);
  if (!sysroot_.empty() && sysroot_.back() == '/') sysroot_.pop_back();
}

namespace {
struct Version {
  int major{};
  std::optional<int> minor;
  std::optional<int> patch;

  static auto parse(const std::string& s) -> std::optional<Version> {
    // parse version numbers of the form "major[.minor[.patch]]", don't
    // throw exception, just return nullopt if the format is invalid
    std::regex versionRe(R"(^(\d+)(?:\.(\d+))?(?:\.(\d+))?$)");
    std::smatch match;
    if (!std::regex_match(s, match, versionRe)) return std::nullopt;

    Version version;
    version.major = std::stoi(match[1].str());
    if (match[2].matched) version.minor = std::stoi(match[2].str());
    if (match[3].matched) version.patch = std::stoi(match[3].str());
    return version;
  }

  auto operator<(const Version& other) const {
    if (major != other.major) return major < other.major;

    auto maxInt = std::numeric_limits<int>::max();

    if (minor != other.minor)
      return minor.value_or(maxInt) < other.minor.value_or(maxInt);

    return patch.value_or(maxInt) < other.patch.value_or(maxInt);
  }
};

auto to_string(const Version& version) -> std::string {
  if (version.patch.has_value())
    return std::format("{}.{}.{}", version.major, version.minor.value_or(0),
                       version.patch.value());

  if (version.minor.has_value())
    return std::format("{}.{}", version.major, version.minor.value());

  return std::format("{}", version.major);
}

}  // namespace

void MacOSToolchain::addSystemIncludePaths() {
  auto platform = sysroot_.empty() ? platformPath_ : sysroot_;

  const auto clangLibDir =
      std::filesystem::path{toolchainPath_} / "usr" / "lib" / "clang";

  struct VersionedResourcePath {
    std::filesystem::path path;
    Version version;
  };

  std::vector<VersionedResourcePath> candidates;

  if (fs::exists(clangLibDir) && std::filesystem::is_directory(clangLibDir)) {
    for (const auto& e : std::filesystem::directory_iterator(clangLibDir)) {
      if (!e.is_directory()) continue;

      auto version = Version::parse(e.path().filename().string());
      if (!version) continue;

      if (!is_directory(e.path() / "include")) continue;

      candidates.emplace_back(e.path(), version.value());
    }
  }

  std::ranges::sort(candidates, std::less<>{}, &VersionedResourcePath::version);

  for (const auto& candidate : candidates | std::ranges::views::reverse) {
    addSystemIncludePath((candidate.path / "include").string());
    break;
  }

  addSystemIncludePath(std::format("{}/usr/include", platform));

  addSystemIncludePath(std::format("{}/usr/include", toolchainPath_));

  addSystemIncludePath(std::format("{}/System/Library/Frameworks", platform));

  addSystemIncludePath(
      std::format("{}/System/Library/SubFrameworks", platform));
}

void MacOSToolchain::addSystemCppIncludePaths() {
  auto platform = sysroot_.empty() ? platformPath_ : sysroot_;
  addSystemIncludePath(std::format("{}/usr/include/c++/v1", platform));
}

void MacOSToolchain::addPredefinedMacros() {
  defineMacro("__autoreleasing", "");
  defineMacro("__building_module(a)", "0");
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__signed__", "signed");
  defineMacro("__signed", "signed");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Pragma(x)", "");

  addCommonMacros();
  addCommonMacOSMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addMacOSCxx26Macros();
  } else {
    addCommonC23Macros();
    addMacOSC23Macros();
  }

  if (arch_ == "aarch64") {
    addMacOSAArch64Macros();
  } else if (arch_ == "x86_64") {
    addMacOSX86_64Macros();
  } else {
    cxx_runtime_error(std::format("Unsupported architecture: {}", arch_));
  }
}

}  // namespace cxx
