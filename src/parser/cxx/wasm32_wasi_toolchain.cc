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
namespace {

[[nodiscard]] auto wasiPreviewMacro(TripleOS os) -> std::string_view {
  if (os == TripleOS::kWasiPreview2) return "__wasip2__";
  return "__wasip1__";
}

[[nodiscard]] auto defaultNewAlignment(const Triple& triple) -> std::size_t {
  if (triple.os() == TripleOS::kEmscripten) return 8;
  return 16;
}

}  // namespace

Wasm32WasiToolchain::Wasm32WasiToolchain(Preprocessor* preprocessor,
                                         Triple triple)
    : Toolchain(preprocessor, std::move(triple)) {
  setMemoryLayout(std::make_unique<MemoryLayout>(32));
  memoryLayout()->setSizeOfLongDouble(16, 113);
  memoryLayout()->setSizeOfLongLong(8);
  memoryLayout()->setWideCharUnderlyingType(4, /*isSigned=*/true);
  memoryLayout()->setDefaultNewAlignment(defaultNewAlignment(this->triple()));
  memoryLayout()->setTriple(this->triple().str());
}

auto Wasm32WasiToolchain::defaultSysroot() const -> std::string {
  if (appdir().empty()) return {};
  auto path = fs::path{appdir()}.parent_path() / "lib" / "wasi-sysroot";
  return path.lexically_normal().string();
}

auto Wasm32WasiToolchain::multiarchName() const -> std::optional<std::string> {
  const auto& triple = this->triple();
  if (triple.os() == TripleOS::kUnknown) return std::nullopt;

  auto name = std::format("{}-{}", triple.archName(), triple.osName());
  if (!triple.environmentName().empty()) {
    name += std::format("-{}", triple.environmentName());
  }
  return name;
}

auto Wasm32WasiToolchain::libraryDir() const -> std::string {
  if (auto multiarch = multiarchName()) {
    return std::format("{}/lib/{}", sysroot(), *multiarch);
  }
  return std::format("{}/lib", sysroot());
}

void Wasm32WasiToolchain::addSystemIncludePaths() {
  addBuiltinIncludePath();

  const auto includeDir = std::format("{}/include", headerSysroot());
  if (auto multiarch = multiarchName()) {
    addSystemIncludePath(std::format("{}/{}", includeDir, *multiarch));
  }
  addSystemIncludePath(includeDir);
}

void Wasm32WasiToolchain::addSystemCppIncludePaths() {
  const auto includeDir = std::format("{}/include", headerSysroot());
  if (auto multiarch = multiarchName()) {
    const auto exceptionsDir = exceptionsEnabled() ? "eh" : "noeh";
    addSystemIncludePath(
        std::format("{}/{}/{}/c++/v1", includeDir, *multiarch, exceptionsDir));
    addSystemIncludePath(std::format("{}/{}/c++/v1", includeDir, *multiarch));
  }
  addSystemIncludePath(std::format("{}/c++/v1", includeDir));
}

void Wasm32WasiToolchain::addTargetMacros() {
  const auto& triple = this->triple();

  if (triple.os() == TripleOS::kEmscripten) {
    defineMacro("__EMSCRIPTEN__", "1");
    defineMacro("__unix", "1");
    defineMacro("__unix__", "1");
    defineMacro("unix", "1");
  }

  if (!triple.isWasi()) return;

  defineMacro("__wasi__", "1");
  defineMacro(std::string{wasiPreviewMacro(triple.os())}, "1");

  if (!hasThreads()) return;

  defineMacro("__wasm_atomics__", "1");
  if (language() != LanguageKind::kCXX) return;
  defineMacro("__STDCPP_THREADS__", "1");
}

auto Wasm32WasiToolchain::hasThreads() const -> bool {
  return triple().isWasi() &&
         triple().environment() == TripleEnvironment::kThreads;
}

void Wasm32WasiToolchain::addPredefinedMacros() {
  defineMacro("__extension__", "");
  defineMacro("__autoreleasing", "");
  defineMacro("__strong", "");
  defineMacro("__unsafe_unretained", "");
  defineMacro("__weak", "");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Thread_local", "thread_local");

  addCommonMacros();
  addCommonWASIMacros();

  if (language() == LanguageKind::kCXX) {
    addCommonCxx26Macros();
    addWASICxx26Macros();
    defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__",
                std::format("{}UL", memoryLayout()->defaultNewAlignment()));
  } else {
    addCommonC23Macros();
    addWASIC23Macros();
  }

  addTargetMacros();
}

void Wasm32WasiToolchain::addLinkerStartArgs(
    std::vector<std::string>& args) const {
  const auto libdir = libraryDir();

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
      std::format("{}/libclang_rt.builtins-wasm32.a", libraryDir());
  if (fs::exists(builtins)) {
    args.push_back(builtins);
  }
}
}  // namespace cxx
