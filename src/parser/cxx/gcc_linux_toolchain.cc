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

#include <cxx/gcc_linux_toolchain.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/private/versioned_directories.h>

#include <filesystem>
#include <format>
#include <optional>
#include <system_error>

namespace cxx {

namespace {

[[nodiscard]] auto namesTargetOf(const std::string& name, const Triple& triple)
    -> bool {
  const Triple candidate{name};
  return candidate.arch() == triple.arch() &&
         candidate.os() == TripleOS::kLinux;
}

[[nodiscard]] auto findTargetSubdirectory(const fs::path& dir,
                                          const Triple& triple)
    -> std::optional<fs::path> {
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (!entry.is_directory()) continue;
    if (namesTargetOf(entry.path().filename().string(), triple)) {
      return entry.path();
    }
  }
  return std::nullopt;
}

}  // namespace

GCCLinuxToolchain::GCCLinuxToolchain(Preprocessor* preprocessor, Triple triple)
    : Toolchain(preprocessor, std::move(triple)) {
  const auto arch = this->triple().arch();

  switch (arch) {
    case TripleArch::kAArch64:
      memoryLayout()->setSizeOfLongDouble(16, 113);
      memoryLayout()->setWideCharUnderlyingType(4, /*isSigned=*/false);
      break;
    case TripleArch::kX86_64:
      memoryLayout()->setSizeOfLongDouble(16, 64);
      break;
    default:
      cxx_runtime_error(std::format("Unsupported architecture: {}",
                                    this->triple().archName()));
  }

  memoryLayout()->setTriple(this->triple().withArchName(to_string(arch)).str());
}

auto GCCLinuxToolchain::multiarchName() const -> std::string {
  return std::format("{}-linux-gnu", to_string(triple().arch()));
}

auto GCCLinuxToolchain::gccInstallDir() const -> std::optional<std::string> {
  const auto sysroot = headerSysroot();

  for (const auto& gccLibDir : {std::format("{}/usr/lib/gcc", sysroot),
                                std::format("{}/usr/lib64/gcc", sysroot),
                                std::format("{}/usr/lib/gcc-cross", sysroot)}) {
    auto targetDir = findTargetSubdirectory(gccLibDir, triple());
    if (!targetDir) continue;

    for (const auto& installDir : versionedSubdirectories(*targetDir)) {
      if (fs::is_directory(installDir / "include")) return installDir.string();
    }
  }

  return std::nullopt;
}

void GCCLinuxToolchain::addSystemIncludePaths() {
  const auto sysroot = headerSysroot();

  addBuiltinIncludePath();

  if (auto installDir = gccInstallDir()) {
    addSystemIncludePath(std::format("{}/include", *installDir));
  }

  addSystemIncludePath(std::format("{}/usr/local/include", sysroot));

  const auto multiarchIncludeDir =
      std::format("{}/usr/include/{}", sysroot, multiarchName());
  if (fs::is_directory(multiarchIncludeDir)) {
    addSystemIncludePath(multiarchIncludeDir);
  }

  addSystemIncludePath(std::format("{}/usr/include", sysroot));
}

void GCCLinuxToolchain::addSystemCppIncludePaths() {
  if (usesLibCxx_) {
    addLibCxxIncludePaths();
    return;
  }

  addLibStdCxxIncludePaths();
}

void GCCLinuxToolchain::addLibStdCxxIncludePaths() {
  const auto includeDir =
      fs::path{std::format("{}/usr/include", headerSysroot())};
  const auto multiarchIncludeDir = includeDir / multiarchName();

  for (const auto& libstdcxxDir : versionedSubdirectories(includeDir / "c++")) {
    const auto version = libstdcxxDir.filename();

    auto targetDir = multiarchIncludeDir / "c++" / version;
    if (!fs::is_directory(targetDir)) {
      auto gccTargetDir = findTargetSubdirectory(libstdcxxDir, triple());
      if (!gccTargetDir) continue;
      targetDir = *gccTargetDir;
    }

    addSystemIncludePath(libstdcxxDir.string());
    addSystemIncludePath(targetDir.string());
    addSystemIncludePath((libstdcxxDir / "backward").string());
    return;
  }
}

void GCCLinuxToolchain::addLibCxxIncludePaths() {
  const auto sysroot = headerSysroot();

  for (const auto& includeDir :
       {fs::path{std::format("{}/usr/local/include", sysroot)},
        fs::path{std::format("{}/usr/include", sysroot)}}) {
    const auto libcxxDir = includeDir / "c++" / "v1";
    if (!fs::is_directory(libcxxDir)) continue;

    const auto targetDir = includeDir / multiarchName() / "c++" / "v1";
    if (fs::is_directory(targetDir)) addSystemIncludePath(targetDir.string());

    addSystemIncludePath(libcxxDir.string());
    return;
  }
}

void GCCLinuxToolchain::addPredefinedMacros() {
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__restrict__", "");
  defineMacro("__restrict", "");
  defineMacro("__signed__", "signed");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");

  addCommonMacros();
  addCommonLinuxMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addLinuxCxx26Macros();
  } else {
    addCommonC23Macros();
    addLinuxC23Macros();
  }

  if (triple().arch() == TripleArch::kAArch64) {
    addLinuxAArch64Macros();
  } else {
    addLinuxX86_64Macros();
  }
}

}  // namespace cxx
