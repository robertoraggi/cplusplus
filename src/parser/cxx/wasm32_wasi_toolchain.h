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

#pragma once

#include <cxx/toolchain.h>

#include <optional>
#include <string>

namespace cxx {

class Wasm32WasiToolchain final : public Toolchain {
 public:
  explicit Wasm32WasiToolchain(Preprocessor* preprocessor);

  const std::string& appdir() const;
  void setAppdir(std::string appdir);

  const std::string& sysroot() const;
  void setSysroot(std::string sysroot);

  void addSystemIncludePaths() override;
  void addSystemCppIncludePaths() override;
  void addPredefinedMacros() override;

 private:
  std::string appdir_;
  std::string sysroot_;
  std::optional<int> version_;
};

}  // namespace cxx
