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

#include <cxx/cli.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/macos_toolchain.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/toolchain.h>
#include <cxx/toolchain_config.h>
#include <cxx/triple.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

namespace cxx {
namespace {

auto applicationDirectory(const CLI& cli) -> fs::path {
#if __wasi__
  return fs::path("/usr/bin/");
#else
  return std::filesystem::canonical(
      std::filesystem::path(cli.app_name).remove_filename());
#endif
}

auto defaultArch() -> std::string {
#if defined(__wasm32__) || defined(__wasi__) || defined(__EMSCRIPTEN__)
  return "wasm32";
#elif defined(__aarch64__) || defined(__arm64__)
  return "aarch64";
#else
  return "x86_64";
#endif
}

auto toolchainForTriple(const Triple& triple) -> std::string {
  if (triple.isDarwin()) return "darwin";

  switch (triple.os()) {
    case TripleOS::kLinux:
      return "linux";
    case TripleOS::kWindows:
      return "windows";
    default:
      break;
  }

  if (triple.isWebAssembly()) return "wasm32";

  return {};
}

auto makeToolchain(const CLI& cli, Preprocessor* preprocessor)
    -> std::unique_ptr<Toolchain> {
  const auto target = selectTarget(cli);
  if (!target.valid) return {};

  const auto& toolchainId = target.toolchain;

  const auto appDir = applicationDirectory(cli);

  auto configure = [&](std::unique_ptr<Toolchain> toolchain) {
    toolchain->setAppdir(appDir.string());
    if (auto paths = cli.get("-resource-dir"); !paths.empty()) {
      toolchain->setResourceDir(paths.back());
    }
    return toolchain;
  };

  if (toolchainId == "darwin") {
    auto toolchain = std::make_unique<MacOSToolchain>(preprocessor, target.arch,
                                                      target.osVersion);
    if (auto paths = cli.get("-isysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    } else if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    }
    return configure(std::move(toolchain));
  }

  if (toolchainId == "wasm32") {
    auto toolchain = std::make_unique<Wasm32WasiToolchain>(preprocessor);
    if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    } else {
      toolchain->setSysroot(
          (appDir / std::string("../lib/wasi-sysroot")).string());
    }
    return configure(std::move(toolchain));
  }

  if (toolchainId == "linux") {
    return configure(
        std::make_unique<GCCLinuxToolchain>(preprocessor, target.arch));
  }

  if (toolchainId == "windows") {
    auto toolchain =
        std::make_unique<WindowsToolchain>(preprocessor, target.arch);
    if (auto paths = cli.get("-vctoolsdir"); !paths.empty()) {
      toolchain->setVctoolsdir(paths.back());
    }
    if (auto paths = cli.get("-winsdkdir"); !paths.empty()) {
      toolchain->setWinsdkdir(paths.back());
    }
    if (auto versions = cli.get("-winsdkversion"); !versions.empty()) {
      toolchain->setWinsdkversion(versions.back());
    }
    return configure(std::move(toolchain));
  }

  return {};
}

}  // namespace

auto selectTarget(const CLI& cli) -> TargetSelection {
  TargetSelection target;
  target.toolchain = "wasm32";
  target.arch = defaultArch();

  std::optional<std::string> requestedTriple;
  if (auto value = cli.getSingle("--target")) requestedTriple = value;
  if (auto value = cli.getSingle("-target")) requestedTriple = value;

  if (requestedTriple) {
    const Triple triple{*requestedTriple};

    target.triple = triple.str();
    target.osVersion = triple.osVersion();

    if (triple.arch() != TripleArch::kUnknown) {
      target.arch = std::string{to_string(triple.arch())};
    }

    target.toolchain = toolchainForTriple(triple);
    target.valid = !target.toolchain.empty();
  } else if (auto id = cli.getSingle("-toolchain")) {
    target.toolchain = *id;
    if (target.toolchain == "macos") target.toolchain = "darwin";
  }

  if (auto arch = cli.getSingle("-arch")) target.arch = *arch;

  return target;
}

auto targetTripleOf(const CLI& cli) -> std::string {
  const auto target = selectTarget(cli);
  if (!target.triple.empty()) return target.triple;

  auto toolchain = createToolchainForLinking(cli, LanguageKind::kCXX);
  if (!toolchain) return target.triple;

  return toolchain->memoryLayout()->triple();
}

auto describeUnsupportedTarget(const CLI& cli) -> std::string {
  const auto target = selectTarget(cli);
  if (!target.valid) {
    return std::format("cxx: no toolchain for target '{}'", target.triple);
  }
  return std::format("cxx: unknown toolchain '{}'", target.toolchain);
}

auto languageOf(const CLI& cli, const std::string& fileName) -> LanguageKind {
  if (auto lang = cli.getSingle("-x")) {
    return lang == "c" ? LanguageKind::kC : LanguageKind::kCXX;
  }
  return fileName.ends_with(".c") ? LanguageKind::kC : LanguageKind::kCXX;
}

auto createToolchainForLinking(const CLI& cli, LanguageKind language)
    -> std::unique_ptr<Toolchain> {
  auto toolchain = makeToolchain(cli, nullptr);
  if (!toolchain) return {};
  toolchain->setLanguage(language);
  return toolchain;
}

auto createToolchain(const CLI& cli, Preprocessor* preprocessor,
                     LanguageKind language, std::string& error)
    -> std::unique_ptr<Toolchain> {
  auto toolchain = makeToolchain(cli, preprocessor);
  if (!toolchain) return {};

  toolchain->setLanguage(language);
  toolchain->setExceptionsEnabled(!cli.opt_fno_exceptions);
  toolchain->initMemoryLayout();

  if (auto standardName = cli.getSingle("-std")) {
    auto standard = findLanguageStandard(*standardName);
    if (!standard) {
      error = std::format("cxx: invalid value '{}' in '-std={}'", *standardName,
                          *standardName);
      return toolchain;
    }
    if (standard->language != toolchain->language()) {
      auto languageName =
          toolchain->language() == LanguageKind::kCXX ? "C++" : "C";
      error =
          std::format("cxx: invalid argument '-std={}' not allowed with '{}'",
                      *standardName, languageName);
      return toolchain;
    }
    toolchain->setLanguageStandard(standard);
  }

  if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();
  if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();
  toolchain->addPredefinedMacros();

  for (const auto& path : cli.get("-iquote")) {
    preprocessor->addQuoteIncludePath(path);
  }
  for (const auto& path : cli.get("-I")) {
    preprocessor->addUserIncludePath(path);
  }
  for (const auto& path : cli.get("-isystem")) {
    preprocessor->addSystemIncludePath(path);
  }
  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of('=');
    if (sep == std::string::npos) {
      preprocessor->defineMacro(macro, "1");
    } else {
      preprocessor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }
  for (const auto& macro : cli.get("-U")) {
    preprocessor->undefMacro(macro);
  }

  return toolchain;
}
}  // namespace cxx
