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

#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>
#include <cxx/views/symbols.h>
#include <cxx/wasm32_wasi_toolchain.h>

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

void Wasm32WasiToolchain::addLinkerStartArgs(
    std::vector<std::string>& args) const {
  const auto libdir = std::format("{}/lib/wasm32-wasi", sysroot_);

  args.push_back(std::format("{}/crt1.o", libdir));
  args.push_back(std::format("-L{}", libdir));
}

void Wasm32WasiToolchain::applyEntryPointAbi(TranslationUnit* unit) const {
  auto control = unit->control();
  auto main = views::find_function(
      unit->globalScope()->find(control->getIdentifier("main")),
      [](FunctionSymbol* func) { return func->isDefined(); });

  if (!main) return;
  if (main->externalName() || main->aliasName()) return;

  auto functionType = type_cast<FunctionType>(main->type());
  if (!functionType || functionType->isVariadic()) return;
  if (functionType->returnType() != control->getIntType()) return;

  const auto& params = functionType->parameterTypes();

  if (params.empty()) {
    main->setAliasName(control->getIdentifier("__main_void"));
    main->setHiddenVisibility(true);
    return;
  }

  if (params.size() == 2 && params[0] == control->getIntType() &&
      type_cast<PointerType>(params[1])) {
    main->setExternalName(control->getIdentifier("__main_argc_argv"));
    main->setHiddenVisibility(true);
  }
}

void Wasm32WasiToolchain::addLinkerEndArgs(
    std::vector<std::string>& args) const {
  args.push_back("-lc");

  if (language() == LanguageKind::kCXX) {
    args.push_back("-lc++");
    args.push_back("-lc++abi");
  }

  const auto builtins =
      std::format("{}/lib/wasm32-wasi/libclang_rt.builtins-wasm32.a", sysroot_);
  if (fs::exists(builtins)) {
    args.push_back(builtins);
  }
}
}  // namespace cxx
