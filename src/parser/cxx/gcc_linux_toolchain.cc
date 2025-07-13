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

#include <cxx/gcc_linux_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <format>

namespace cxx {

GCCLinuxToolchain::GCCLinuxToolchain(Preprocessor* preprocessor,
                                     std::string arch)
    : Toolchain(preprocessor), arch_(std::move(arch)) {
  for (int version : {15, 14, 13, 12, 11, 10, 9}) {
    const auto path = fs::path(
        std::format("/usr/lib/gcc/{}-linux-gnu/{}/include", arch_, version));

    if (exists(path)) {
      version_ = version;
      break;
    }
  }
}

void GCCLinuxToolchain::addSystemIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(
        std::format("/usr/lib/gcc/{}-linux-gnu/{}/include", arch_, version));
  };

  addSystemIncludePath("/usr/include");
  addSystemIncludePath(std::format("/usr/include/{}-linux-gnu", arch_));
  addSystemIncludePath("/usr/local/include");

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addSystemCppIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(std::format("/usr/include/c++/{}/backward", version));

    addSystemIncludePath(
        std::format("/usr/include/{}-linux-gnu/c++/{}", arch_, version));

    addSystemIncludePath(std::format("/usr/include/c++/{}", version));
  };

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addPredefinedMacros() {
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__restrict__", "");
  defineMacro("__restrict", "");
  defineMacro("__signed__", "signed");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Pragma(x)", "");

  addCommonMacros();
  addCommonLinuxMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addLinuxCxx26Macros();
  } else {
    addCommonC23Macros();
    addLinuxC23Macros();
  }

  if (arch_ == "aarch64") {
    addLinuxAArch64Macros();
  } else if (arch_ == "x86_64") {
    addLinuxX86_64Macros();
  } else {
    cxx_runtime_error(std::format("Unsupported architecture: {}", arch_));
  }
}

}  // namespace cxx
