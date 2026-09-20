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

#include <cxx/cxx_fwd.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {
class Preprocessor;
class MemoryLayout;

enum class LinkerFlavor {
  kUnknown,
  kGnu,
  kDarwin,
  kWasm,
  kWinLink,
};

struct LanguageStandard {
  std::string_view name;
  LanguageKind language;
  std::string_view versionMacroValue;
};

[[nodiscard]] auto findLanguageStandard(std::string_view name)
    -> const LanguageStandard*;

class Toolchain {
 public:
  Toolchain(const Toolchain&) = delete;
  auto operator=(const Toolchain&) -> Toolchain& = delete;

  explicit Toolchain(Preprocessor* preprocessor);
  virtual ~Toolchain();

  [[nodiscard]] auto language() const -> LanguageKind { return language_; }

  void setLanguage(LanguageKind language);

  void setLanguageStandard(const LanguageStandard* languageStandard);

  [[nodiscard]] auto exceptionsEnabled() const -> bool {
    return exceptionsEnabled_;
  }

  void setExceptionsEnabled(bool exceptionsEnabled) {
    exceptionsEnabled_ = exceptionsEnabled;
  }

  [[nodiscard]] auto memoryLayout() const -> MemoryLayout* {
    return memoryLayout_.get();
  }

  [[nodiscard]] auto appdir() const -> const std::string& { return appdir_; }

  void setAppdir(std::string appdir);

  [[nodiscard]] auto resourceDir() const -> std::string;

  void setResourceDir(std::string resourceDir);

  void setMemoryLayout(std::unique_ptr<MemoryLayout> memoryLayout);

  virtual void initMemoryLayout();
  virtual void addSystemIncludePaths() = 0;
  virtual void addSystemCppIncludePaths() = 0;
  virtual void addPredefinedMacros() = 0;

  [[nodiscard]] virtual auto linkerFlavor() const -> LinkerFlavor {
    return LinkerFlavor::kUnknown;
  }

  virtual void addLinkerStartArgs(std::vector<std::string>& args) const {}

  virtual void addLinkerEndArgs(std::vector<std::string>& args) const {}

  virtual void applyEntryPointAbi(TranslationUnit* unit) const {}

  [[nodiscard]] auto preprocessor() const -> Preprocessor* {
    return preprocessor_;
  }

  void defineMacro(const std::string& name, const std::string& definition);
  void undefMacro(const std::string& name);

  void addSystemIncludePath(std::string path);

  void addCommonMacros();
  void addCommonC23Macros();
  void addCommonCxx26Macros();
  void addFeatureTestMacros();
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

 protected:
  [[nodiscard]] virtual auto defaultResourceDir() const -> std::string;

 private:
  [[nodiscard]] auto cplusplusMacroValue() const -> std::string_view;
  [[nodiscard]] auto stdcVersionMacroValue() const -> std::string_view;

  Preprocessor* preprocessor_;
  std::string appdir_;
  std::string resourceDir_;
  std::unique_ptr<MemoryLayout> memoryLayout_;
  LanguageKind language_ = LanguageKind::kCXX;
  const LanguageStandard* languageStandard_ = nullptr;
  bool exceptionsEnabled_ = true;
};
}  // namespace cxx
