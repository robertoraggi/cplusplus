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
#include <cxx/private/versioned_directories.h>

#include <cstdio>
#include <cstdlib>
#include <format>
#include <initializer_list>
#include <optional>
#include <utility>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace cxx {
namespace {

constexpr int kFallbackVersionMajor = 26;

auto hostProductVersion() -> std::optional<std::pair<int, int>> {
#ifdef __APPLE__
  char buffer[64] = {};
  std::size_t size = sizeof(buffer) - 1;
  if (sysctlbyname("kern.osproductversion", buffer, &size, nullptr, 0) != 0) {
    return std::nullopt;
  }

  int major = 0;
  int minor = 0;
  if (std::sscanf(buffer, "%d.%d", &major, &minor) < 1) return std::nullopt;

  return std::pair{major, minor};
#else
  return std::nullopt;
#endif
}

[[nodiscard]] auto environmentPath(const char* name) -> std::string {
  const auto value = std::getenv(name);
  if (!value) return {};
  return value;
}

[[nodiscard]] auto selectedDeveloperDir() -> std::string {
  if (auto path = environmentPath("DEVELOPER_DIR"); !path.empty()) return path;

  std::error_code ec;
  auto link = fs::read_symlink("/var/db/xcode_select_link", ec);
  if (!ec) return link.string();

  return "/Applications/Xcode.app/Contents/Developer";
}

[[nodiscard]] auto firstExistingDirectory(
    std::initializer_list<fs::path> candidates) -> std::string {
  for (const auto& candidate : candidates) {
    if (fs::is_directory(candidate)) return candidate.string();
  }
  return {};
}

}  // namespace

MacOSToolchain::MacOSToolchain(Preprocessor* preprocessor, Triple triple)
    : Toolchain(preprocessor, std::move(triple)),
      developerDir_(selectedDeveloperDir()) {
  versionMajor_ = kFallbackVersionMajor;
  versionMinor_ = 0;

  auto osVersion = this->triple().osVersion();
  if (!osVersion) osVersion = hostProductVersion();

  if (osVersion) {
    versionMajor_ = osVersion->first;
    versionMinor_ = osVersion->second;
  }

  switch (this->triple().arch()) {
    case TripleArch::kAArch64:
      memoryLayout()->setSizeOfLongDouble(8, 53);
      memoryLayout()->setTriple(
          std::format("arm64-apple-macosx{}", deploymentTargetTriplePart()));
      break;
    case TripleArch::kX86_64:
      memoryLayout()->setSizeOfLongDouble(16, 64);
      memoryLayout()->setTriple(
          std::format("x86_64-apple-macosx{}", deploymentTargetTriplePart()));
      break;
    default:
      cxx_runtime_error(std::format("Unsupported architecture: {}",
                                    this->triple().archName()));
  }
}

auto MacOSToolchain::deploymentTargetTriplePart() const -> std::string {
  return std::format("{}.{}.0", versionMajor_, versionMinor_);
}

auto MacOSToolchain::deploymentTargetMacroValue() const -> std::string {
  return std::format("{}{:02}{:02}", versionMajor_, versionMinor_, 0);
}

auto MacOSToolchain::defaultSysroot() const -> std::string {
  if (auto path = environmentPath("SDKROOT"); !path.empty()) return path;

  const auto developerDir = fs::path{developerDir_};
  return firstExistingDirectory(
      {developerDir / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
       developerDir / "SDKs/MacOSX.sdk"});
}

auto MacOSToolchain::defaultResourceDir() const -> std::string {
  const auto developerDir = fs::path{developerDir_};
  const auto clangLibDir = fs::path{firstExistingDirectory(
      {developerDir / "Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
       developerDir / "usr/lib/clang"})};

  for (const auto& path : versionedSubdirectories(clangLibDir)) {
    if (fs::is_directory(path / "include")) return path.string();
  }

  return Toolchain::defaultResourceDir();
}

void MacOSToolchain::addSystemIncludePaths() {
  const auto sysroot = headerSysroot();

  addSystemIncludePath(std::format("{}/usr/local/include", sysroot));
  addBuiltinIncludePath();
  addSystemIncludePath(std::format("{}/usr/include", sysroot));
  addSystemIncludePath(std::format("{}/System/Library/Frameworks", sysroot));
  addSystemIncludePath(std::format("{}/System/Library/SubFrameworks", sysroot));
  addSystemIncludePath(std::format("{}/Library/Frameworks", sysroot));
}

void MacOSToolchain::addSystemCppIncludePaths() {
  addSystemIncludePath(std::format("{}/usr/include/c++/v1", headerSysroot()));
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

  addCommonMacros();
  addCommonMacOSMacros();

  defineMacro("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__",
              deploymentTargetMacroValue());
  defineMacro("__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__",
              deploymentTargetMacroValue());

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addMacOSCxx26Macros();
  } else {
    addCommonC23Macros();
    addMacOSC23Macros();
  }

  if (triple().arch() == TripleArch::kAArch64) {
    addMacOSAArch64Macros();
  } else {
    addMacOSX86_64Macros();
  }
}

}  // namespace cxx
