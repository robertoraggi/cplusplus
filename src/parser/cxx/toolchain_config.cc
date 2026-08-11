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

#include <cxx/toolchain_config.h>

#include <cxx/cli.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/toolchain.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#include <format>

namespace cxx {
namespace {

auto makeToolchain(const CLI& cli, Preprocessor* preprocessor)
    -> std::unique_ptr<Toolchain> {
  auto toolchainId = cli.getSingle("-toolchain").value_or("wasm32");

  if (toolchainId == "darwin" || toolchainId == "macos") {
    std::string host = "aarch64";
#ifdef __x86_64__
    host = "x86_64";
#endif
    auto toolchain = std::make_unique<MacOSToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));
    if (auto paths = cli.get("-isysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    } else if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    }
    return toolchain;
  }

  if (toolchainId == "wasm32") {
    auto toolchain = std::make_unique<Wasm32WasiToolchain>(preprocessor);
    fs::path appDir;
#if __wasi__
    appDir = fs::path("/usr/bin/");
#elif !defined(CXX_NO_FILESYSTEM)
    appDir = std::filesystem::canonical(
        std::filesystem::path(cli.app_name).remove_filename());
#elif __unix__ || __APPLE__
    char* appName = realpath(cli.app_name.c_str(), nullptr);
    appDir = fs::path(appName).remove_filename().string();
    std::free(appName);
#endif
    toolchain->setAppdir(appDir.string());
    if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      toolchain->setSysroot(paths.back());
    } else {
      toolchain->setSysroot((appDir / std::string("../lib/wasi-sysroot")).string());
    }
    return toolchain;
  }

  std::string host = "x86_64";
#ifdef __aarch64__
  host = "aarch64";
#endif
  if (toolchainId == "linux") {
    return std::make_unique<GCCLinuxToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));
  }

  if (toolchainId == "windows") {
    auto toolchain = std::make_unique<WindowsToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));
    if (auto paths = cli.get("-vctoolsdir"); !paths.empty()) {
      toolchain->setVctoolsdir(paths.back());
    }
    if (auto paths = cli.get("-winsdkdir"); !paths.empty()) {
      toolchain->setWinsdkdir(paths.back());
    }
    if (auto versions = cli.get("-winsdkversion"); !versions.empty()) {
      toolchain->setWinsdkversion(versions.back());
    }
    return toolchain;
  }

  return {};
}

}

auto createToolchain(const CLI& cli, Preprocessor* preprocessor,
                     std::string& error) -> std::unique_ptr<Toolchain> {
  auto toolchain = makeToolchain(cli, preprocessor);
  if (!toolchain) return {};

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
      error = std::format("cxx: invalid argument '-std={}' not allowed with '{}'",
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
}
