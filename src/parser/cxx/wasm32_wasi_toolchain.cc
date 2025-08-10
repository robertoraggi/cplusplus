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

#include <cxx/wasm32_wasi_toolchain.h>

// cxx
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <format>

namespace cxx {

Wasm32WasiToolchain::Wasm32WasiToolchain(Preprocessor* preprocessor)
    : Toolchain(preprocessor) {
  setMemoryLayout(std::make_unique<MemoryLayout>(32));
  memoryLayout()->setSizeOfLongDouble(16);
  memoryLayout()->setSizeOfLongLong(8);
  memoryLayout()->setTriple("wasm32");
}

auto Wasm32WasiToolchain::appdir() const -> const std::string& {
  return appdir_;
}

void Wasm32WasiToolchain::setAppdir(std::string appdir) {
  appdir_ = std::move(appdir);

  if (!appdir_.empty() && appdir_.back() == '/') {
    appdir_.pop_back();
  }
}

auto Wasm32WasiToolchain::sysroot() const -> const std::string& {
  return sysroot_;
}

void Wasm32WasiToolchain::setSysroot(std::string sysroot) {
  sysroot_ = std::move(sysroot);

  if (!sysroot_.empty() && sysroot_.back() == '/') {
    sysroot_.pop_back();
  }
}

void Wasm32WasiToolchain::addSystemIncludePaths() {
  addSystemIncludePath(std::format("{}/include", sysroot_));
  addSystemIncludePath(std::format("{}/include/wasm32-wasi", sysroot_));
  addSystemIncludePath(std::format("{}/../lib/cxx/include", appdir_));
}

void Wasm32WasiToolchain::addSystemCppIncludePaths() {
  if (language() != LanguageKind::kCXX) return;

  addSystemIncludePath(std::format("{}/include/c++/v1", sysroot_));
  addSystemIncludePath(std::format("{}/include/wasm32-wasi/c++/v1", sysroot_));
}

void Wasm32WasiToolchain::addPredefinedMacros() {
  defineMacro("__extension__", "");
  defineMacro("__autoreleasing", "");
  defineMacro("__strong", "");
  defineMacro("__unsafe_unretained", "");
  defineMacro("__weak", "");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Pragma(x)", "");
  defineMacro("_Thread_local", "thread_local");

  addCommonMacros();
  addCommonWASIMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addWASICxx26Macros();
  } else {
    addCommonC23Macros();
    addWASIC23Macros();
  }
}

}  // namespace cxx
