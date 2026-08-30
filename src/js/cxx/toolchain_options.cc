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

#include "toolchain_options.h"

#include <cxx/preprocessor.h>
#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>

namespace cxx::js {

using emscripten::val;

namespace {

template <typename F>
void addIncludePaths(const val& options, const char* name, F&& add) {
  for (auto& path : stringArrayOption(options, name)) add(std::move(path));
}

void applyLanguageStandard(Wasm32WasiToolchain* toolchain, const val& options) {
  val standard = options["std"];
  if (!standard.isString()) return;

  auto languageStandard = findLanguageStandard(standard.as<std::string>());
  if (!languageStandard) return;
  if (languageStandard->language != toolchain->language()) return;

  toolchain->setLanguageStandard(languageStandard);
}

void applyMacros(Preprocessor* preprocessor, const val& options) {
  for (const auto& macro : stringArrayOption(options, "undefines")) {
    preprocessor->undefMacro(macro);
  }

  for (const auto& macro : stringArrayOption(options, "defines")) {
    auto sep = macro.find_first_of('=');
    if (sep == std::string::npos) {
      preprocessor->defineMacro(macro, "1");
    } else {
      preprocessor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }
}

}  // namespace

auto stringArrayOption(const val& options, const char* name)
    -> std::vector<std::string> {
  if (options.isUndefined()) return {};
  val value = options[name];
  if (!value.isArray()) return {};
  return emscripten::vecFromJSArray<std::string>(value);
}

auto configureToolchain(TranslationUnit* unit, const val& options)
    -> std::unique_ptr<Wasm32WasiToolchain> {
  auto preprocessor = unit->preprocessor();
  if (!preprocessor) return {};

  auto toolchain = std::make_unique<Wasm32WasiToolchain>(preprocessor);

  if (!options.isUndefined()) {
    if (val appdir = options["appdir"]; appdir.isString()) {
      toolchain->setAppdir(appdir.as<std::string>());
    }

    if (val sysroot = options["sysroot"]; sysroot.isString()) {
      toolchain->setSysroot(sysroot.as<std::string>());
    }

    applyLanguageStandard(toolchain.get(), options);
  }

  toolchain->initMemoryLayout();
  toolchain->addSystemCppIncludePaths();
  toolchain->addSystemIncludePaths();
  toolchain->addPredefinedMacros();

  addIncludePaths(options, "quoteIncludePaths", [&](std::string path) {
    preprocessor->addQuoteIncludePath(std::move(path));
  });

  addIncludePaths(options, "includePaths", [&](std::string path) {
    preprocessor->addUserIncludePath(std::move(path));
  });

  addIncludePaths(options, "systemIncludePaths", [&](std::string path) {
    preprocessor->addSystemIncludePath(std::move(path));
  });

  applyMacros(preprocessor, options);

  preprocessor->setCanResolveFiles(true);

  return toolchain;
}

}  // namespace cxx::js
