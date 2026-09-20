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

#include <cxx/toolchain.h>

#include <optional>
#include <string>
#include <utility>

namespace cxx {

class MacOSToolchain final : public Toolchain {
 public:
  explicit MacOSToolchain(
      Preprocessor* preprocessor, std::string arch = "aarch64",
      std::optional<std::pair<int, int>> osVersion = std::nullopt);

  [[nodiscard]] auto arch() const -> std::string { return arch_; }

  [[nodiscard]] auto sysroot() const -> const std::string& { return sysroot_; }

  [[nodiscard]] auto deploymentTargetTriplePart() const -> std::string;

  [[nodiscard]] auto deploymentTargetMacroValue() const -> std::string;
  void setSysroot(std::string sysroot);

  void addSystemIncludePaths() override;

 protected:
  [[nodiscard]] auto defaultResourceDir() const -> std::string override;

 public:
  void addSystemCppIncludePaths() override;
  void addPredefinedMacros() override;

 private:
  int versionMajor_ = 0;
  int versionMinor_ = 0;
  std::string platformPath_;
  std::string toolchainPath_;
  std::string arch_;
  std::string sysroot_;
};

}  // namespace cxx
