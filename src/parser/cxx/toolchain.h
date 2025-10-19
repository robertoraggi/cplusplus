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

#include <cxx/cxx_fwd.h>

#include <memory>
#include <string>

namespace cxx {

class Preprocessor;
class MemoryLayout;

class Toolchain {
 public:
  Toolchain(const Toolchain&) = delete;
  auto operator=(const Toolchain&) -> Toolchain& = delete;

  explicit Toolchain(Preprocessor* preprocessor);
  virtual ~Toolchain();

  [[nodiscard]] auto language() const -> LanguageKind;

  [[nodiscard]] auto memoryLayout() const -> MemoryLayout* {
    return memoryLayout_.get();
  }

  void setMemoryLayout(std::unique_ptr<MemoryLayout> memoryLayout);

  virtual void addSystemIncludePaths() = 0;
  virtual void addSystemCppIncludePaths() = 0;
  virtual void addPredefinedMacros() = 0;

  [[nodiscard]] auto preprocessor() const -> Preprocessor* {
    return preprocessor_;
  }

  void defineMacro(const std::string& name, const std::string& definition);

  void addSystemIncludePath(std::string path);

  void addCommonMacros();
  void addCommonC23Macros();
  void addCommonCxx26Macros();
  void addCommonLinuxMacros();
  void addCommonMacOSMacros();
  void addCommonWindowsMacros();
  void addCommonWASIMacros();
  void addLinuxAArch64Macros();
  void addLinuxX86_64Macros();
  void addMacOSAArch64Macros();
  void addMacOSX86_64Macros();
  void addWindowsAArch64Macros();
  void addWindowsX86_64Macros();
  void addWASIWasm32Macros();
  void addLinuxC23Macros();
  void addMacOSC23Macros();
  void addWindowsC23Macros();
  void addWASIC23Macros();
  void addLinuxCxx26Macros();
  void addMacOSCxx26Macros();
  void addWindowsCxx26Macros();
  void addWASICxx26Macros();

 private:
  Preprocessor* preprocessor_;
  std::unique_ptr<MemoryLayout> memoryLayout_;
};

}  // namespace cxx
