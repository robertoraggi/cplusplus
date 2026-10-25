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
#include <cxx/spirv_environment.h>
#include <cxx/spirv_toolchain.h>
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
#include <variant>
#include <vector>

namespace cxx {
namespace {

auto applicationDirectory(const CLI& cli) -> fs::path {
#if __wasi__
  return fs::path("/usr/bin/");
#elif __EMSCRIPTEN__
  return fs::path(cli.app_name).remove_filename();
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

auto defaultArchOf(std::string_view toolchainId) -> std::string {
  if (toolchainId == "wasm32") return "wasm32";
  return defaultArch();
}

auto defaultTripleOf(std::string_view toolchainId) -> Triple {
  if (toolchainId == "spirv32" || toolchainId == "spirv64")
    return Triple{std::format("{}-unknown-unknown", toolchainId)};
  const auto arch = defaultArchOf(toolchainId);
  if (toolchainId == "wasm32") return Triple{"wasm32-wasip1"};
  if (toolchainId == "darwin")
    return Triple{std::format("{}-apple-macosx", arch)};
  if (toolchainId == "linux") {
    return Triple{std::format("{}-unknown-linux-gnu", arch)};
  }
  if (toolchainId == "windows") {
    return Triple{std::format("{}-pc-windows-msvc", arch)};
  }
  return Triple{};
}

[[nodiscard]] auto namesUnknownArch(const Triple& triple) -> bool {
  if (triple.archName().empty()) return false;
  return triple.arch() == TripleArch::kUnknown;
}

auto toolchainForTriple(const Triple& triple) -> std::string {
  if (triple.isSPIRV()) {
    if (!spirvEnvironmentOf(triple)) return {};
    return "spirv";
  }
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

auto toolchainTriple(const TargetSelection& target) -> Triple {
  if (target.triple.arch() != TripleArch::kUnknown) return target.triple;
  return target.triple.withArchName(defaultArchOf(target.toolchain));
}

auto instantiateToolchain(const CLI& cli, Preprocessor* preprocessor,
                          const TargetSelection& target)
    -> std::unique_ptr<Toolchain> {
  auto triple = toolchainTriple(target);

  if (target.toolchain == "spirv")
    return std::make_unique<SPIRVToolchain>(preprocessor, std::move(triple));

  if (target.toolchain == "darwin") {
    return std::make_unique<MacOSToolchain>(preprocessor, std::move(triple));
  }

  if (target.toolchain == "wasm32") {
    return std::make_unique<Wasm32WasiToolchain>(preprocessor,
                                                 std::move(triple));
  }

  if (target.toolchain == "linux") {
    auto toolchain =
        std::make_unique<GCCLinuxToolchain>(preprocessor, std::move(triple));
    toolchain->setUsesLibCxx(cli.getSingle("-stdlib") == "libc++");
    return toolchain;
  }

  if (target.toolchain == "windows") {
    auto toolchain =
        std::make_unique<WindowsToolchain>(preprocessor, std::move(triple));
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

auto makeToolchain(const CLI& cli, Preprocessor* preprocessor)
    -> std::unique_ptr<Toolchain> {
  const auto target = selectTarget(cli);
  if (!target.valid) return {};

  auto toolchain = instantiateToolchain(cli, preprocessor, target);
  if (!toolchain) return {};

  toolchain->setExceptionsEnabled(cli.exceptionsEnabled(
      !toolchain->triple().isWebAssembly() && !toolchain->triple().isSPIRV()));
  toolchain->setAppdir(applicationDirectory(cli).string());

  if (auto paths = cli.get("-resource-dir"); !paths.empty()) {
    toolchain->setResourceDir(paths.back());
  }
  if (auto paths = cli.get("--sysroot"); !paths.empty()) {
    toolchain->setSysroot(paths.back());
  }
  if (auto paths = cli.get("-isysroot"); !paths.empty()) {
    toolchain->setHeaderSysroot(paths.back());
  }

  return toolchain;
}

[[nodiscard]] auto isKnownStandardLibrary(std::string_view name) -> bool {
  return name == "libc++" || name == "libstdc++";
}

[[nodiscard]] auto withHeaderSysroot(const Toolchain* toolchain,
                                     const std::string& path) -> std::string {
  if (!path.starts_with('/')) return path;
  return toolchain->headerSysroot() + path;
}

void addCommandLineSystemIncludePaths(const CLI& cli, Toolchain* toolchain) {
  auto preprocessor = toolchain->preprocessor();

  for (const auto& match : cli) {
    auto option = std::get_if<CLIOption>(&match);
    if (!option) continue;

    const auto& name = std::get<0>(*option);
    const auto& path = std::get<1>(*option);

    if (name == "-isystem") {
      preprocessor->addSystemIncludePath(path);
    } else if (name == "-iwithsysroot") {
      preprocessor->addSystemIncludePath(withHeaderSysroot(toolchain, path));
    }
  }
}

void applyCommandLineMacros(const CLI& cli, Preprocessor* preprocessor) {
  for (const auto& match : cli) {
    auto option = std::get_if<CLIOption>(&match);
    if (!option) continue;

    const auto& name = std::get<0>(*option);
    const auto& macro = std::get<1>(*option);

    if (name == "-U") {
      preprocessor->undefMacro(macro);
      continue;
    }
    if (name != "-D") continue;
    auto sep = macro.find_first_of('=');
    if (sep == std::string::npos) {
      preprocessor->defineMacro(macro, "1");
    } else {
      preprocessor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }
}

void addStandardIncludePaths(const CLI& cli, Toolchain* toolchain) {
  if (cli.opt_nostdinc) return;

  if (toolchain->language() == LanguageKind::kCXX && !cli.opt_nostdincpp) {
    if (auto paths = cli.get("-stdlib++-isystem"); !paths.empty()) {
      for (const auto& path : paths) toolchain->addSystemIncludePath(path);
    } else {
      toolchain->addSystemCppIncludePaths();
    }
  }

  toolchain->addSystemIncludePaths();
}

}  // namespace

auto selectTarget(const CLI& cli) -> TargetSelection {
  TargetSelection target;

  auto requestedTriple = cli.getLast({"--target", "-target"});

  if (requestedTriple) {
    target.triple = Triple{*requestedTriple};
    target.requested = true;
    target.toolchain = toolchainForTriple(target.triple);
    target.valid = !target.toolchain.empty();
    if (namesUnknownArch(target.triple)) target.valid = false;
  } else {
    target.toolchain = cli.getSingle("-toolchain").value_or("wasm32");
    if (target.toolchain == "macos") target.toolchain = "darwin";
    target.triple = defaultTripleOf(target.toolchain);
    if (target.triple.isSPIRV()) target.toolchain = "spirv";
  }

  if (auto arch = cli.getSingle("-arch")) {
    target.triple = target.triple.withArchName(*arch);
  }

  return target;
}

auto targetTripleOf(const CLI& cli) -> std::string {
  const auto target = selectTarget(cli);
  if (target.requested) return target.triple.str();

  auto toolchain = createToolchainForLinking(cli, LanguageKind::kCXX);
  if (!toolchain) return target.triple.str();

  return toolchain->memoryLayout()->triple();
}

auto describeUnsupportedTarget(const CLI& cli) -> std::string {
  const auto target = selectTarget(cli);
  if (!target.valid) {
    return std::format("cxx: no toolchain for target '{}'",
                       target.triple.str());
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

  auto profile = cli.getSingle("--device-profile");
  if (!profile && toolchain->triple().arch() == TripleArch::kSPIRV32)
    profile = "vulkan1.1-f32";

  if (profile && *profile != "none") {
    if (*profile != "vulkan1.1-f32") {
      error = std::format("cxx: unknown device profile '{}'", *profile);
      return toolchain;
    }
    if (toolchain->triple().arch() != TripleArch::kSPIRV32) {
      error = "cxx: device profile 'vulkan1.1-f32' requires a spirv32 target";
      return toolchain;
    }
    if (cli.exceptionsEnabled(false)) {
      error = "cxx: device profile 'vulkan1.1-f32' does not support exceptions";
      return toolchain;
    }
    if (auto option = cli.hostOnlyOutputOption()) {
      error = std::format(
          "cxx: {} needs a host target; device profile 'vulkan1.1-f32' "
          "emits -emit-ir, -emit-mlir, -S, -c, -emit-spirv and -emit-wgsl",
          *option);
      return toolchain;
    }
    toolchain->memoryLayout()->setDeviceProfile(DeviceProfile::kVulkan11F32);
  }

  toolchain->setLanguage(language);
  toolchain->setRttiEnabled(cli.rttiEnabled());
  if (toolchain->supportsPositionIndependence()) {
    toolchain->memoryLayout()->setPositionIndependence(
        cli.positionIndependence(toolchain->defaultPositionIndependence()));
  }
  toolchain->initMemoryLayout();

  if (auto name = cli.getSingle("-stdlib");
      name && !isKnownStandardLibrary(*name)) {
    error = std::format("cxx: invalid library name in argument '-stdlib={}'",
                        *name);
    return toolchain;
  }

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

  addCommandLineSystemIncludePaths(cli, toolchain.get());
  addStandardIncludePaths(cli, toolchain.get());
  toolchain->addPredefinedMacros();

  for (const auto& path : cli.get("-iquote")) {
    preprocessor->addQuoteIncludePath(path);
  }
  for (const auto& path : cli.get("-I")) {
    preprocessor->addUserIncludePath(path);
  }
  applyCommandLineMacros(cli, preprocessor);

  return toolchain;
}
}  // namespace cxx
