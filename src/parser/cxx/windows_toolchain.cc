// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/preprocessor.h>
#include <cxx/private/format.h>
#include <cxx/private/path.h>

#include <utility>

namespace cxx {

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
      (fs::path(winsdkdir_) / fmt::format("Include/{}/winrt", winsdkversion_))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) / fmt::format("Include/{}/um", winsdkversion_))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) / fmt::format("Include/{}/shared", winsdkversion_))
          .string());

  addSystemIncludePath(
      (fs::path(winsdkdir_) / fmt::format("Include/{}/ucrt", winsdkversion_))
          .string());

  addSystemIncludePath((fs::path(vctoolsdir_) / "atlmfc/include").string());

  addSystemIncludePath((fs::path(vctoolsdir_) / "include").string());
}

void WindowsToolchain::addSystemCppIncludePaths() {}

void WindowsToolchain::addPredefinedMacros() {
  // clang-format off
  defineMacro("__cplusplus", "202101L");
  defineMacro("_WIN32", "1");
  defineMacro("_WIN64", "1");
  defineMacro("_MT", "1");
  defineMacro("_M_AMD64", "100");
  defineMacro("_M_X64", "100");
  defineMacro("_MSC_BUILD", "1");
  defineMacro("_MSC_EXTENSIONS", "1");
  defineMacro("_MSC_FULL_VER", "193000000");
  defineMacro("_MSC_VER", "1930");
  defineMacro("_MSVC_LANG", "201705L");
  defineMacro("_CPPRTTI", "1");
  defineMacro("_CPPUNWIND", "1");
  defineMacro("_WCHAR_T_DEFINED", "1");
  defineMacro("__BOOL_DEFINED", "1");

  defineMacro("__int8", "char");
  defineMacro("__int16", "short");
  defineMacro("__int32", "int");
  defineMacro("__int64", "long long");

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

  // clang-format on
}

}  // namespace cxx
