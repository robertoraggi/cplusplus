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

#include <cxx/windows_toolchain.h>

// cxx
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <format>
#include <utility>

namespace cxx {

WindowsToolchain::WindowsToolchain(Preprocessor* preprocessor,
                                   std::string arch = "x86_64")
    : Toolchain(preprocessor), arch_(std::move(arch)) {
  memoryLayout()->setSizeOfLong(4);
  memoryLayout()->setSizeOfLongLong(8);
  memoryLayout()->setSizeOfLongDouble(8);
  memoryLayout()->setSizeOfPointer(8);

  if (arch_ == "aarch64") {
    memoryLayout()->setTriple("aarch64-windows");
  } else if (arch_ == "x86_64") {
    memoryLayout()->setTriple("x86_64-windows");
  } else {
    cxx_runtime_error(std::format("Unsupported architecture: {}", arch_));
  }
}

void WindowsToolchain::setVctoolsdir(std::string path) {
  vctoolsdir_ = std::move(path);
}

void WindowsToolchain::setWinsdkdir(std::string path) {
  winsdkdir_ = std::move(path);
}

void WindowsToolchain::setWinsdkversion(std::string version) {
  winsdkversion_ = std::move(version);
}

void WindowsToolchain::addSystemIncludePaths() {
  addSystemIncludePath(
      (fs::path(winsdkdir_) /
       std::string(std::format("Include/{}/winrt", winsdkversion_)))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) /
       std::string(std::format("Include/{}/um", winsdkversion_)))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) /
       std::string(std::format("Include/{}/shared", winsdkversion_)))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) /
       std::string(std::format("Include/{}/ucrt", winsdkversion_)))
          .string());

  addSystemIncludePath(
      (fs::path(vctoolsdir_) / std::string("atlmfc/include")).string());

  addSystemIncludePath(
      (fs::path(vctoolsdir_) / std::string("include")).string());
}

void WindowsToolchain::addSystemCppIncludePaths() {}

void WindowsToolchain::addPredefinedMacros() {
  defineMacro("__pragma(a)", "");
  defineMacro("__declspec(a)", "");
  defineMacro("__cdecl", "");
  defineMacro("__fastcall", "");
  defineMacro("__thiscall", "");
  defineMacro("__vectorcall", "");
  defineMacro("__stdcall", "");
  defineMacro("__forceinline", "inline");
  defineMacro("__unaligned", "");
  defineMacro("_Pragma(a)", "");

  addCommonMacros();
  addCommonWindowsMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addWindowsCxx26Macros();
  } else {
    addCommonC23Macros();
    addWindowsC23Macros();
  }

  if (arch_ == "aarch64") {
    addWindowsAArch64Macros();
  } else if (arch_ == "x86_64") {
    addWindowsX86_64Macros();
  } else {
    cxx_runtime_error(std::format("Unsupported architecture: {}", arch_));
  }
}

}  // namespace cxx
