// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <format>

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

void MacOSToolchain::addSystemIncludePaths() {
  addSystemIncludePath(
      std::format("{}/System/Library/Frameworks", platformPath_));

  addSystemIncludePath(std::format("{}/usr/include", toolchainPath_));

  addSystemIncludePath(std::format("{}/usr/include", platformPath_));

  std::vector<std::string_view> versions{
      "17.0.0",
      "16.0.0",
      "15.0.0",
  };

  for (auto version : versions) {
    const std::string path =
        std::format("{}/usr/lib/clang/{}/include", toolchainPath_, version);
    if (fs::exists(path)) {
      addSystemIncludePath(path);
    }
  }
}

void MacOSToolchain::addSystemCppIncludePaths() {
  addSystemIncludePath(std::format("{}/usr/include/c++/v1", platformPath_));
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
