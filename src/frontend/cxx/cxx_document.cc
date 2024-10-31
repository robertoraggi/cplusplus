// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "cxx_document.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/lexer.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/requests.h>
#include <cxx/lsp/types.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/scope.h>
#include <cxx/symbol_printer.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

namespace cxx::lsp {

namespace {

struct Diagnostics final : cxx::DiagnosticsClient {
  json messages = json::array();
  Vector<lsp::Diagnostic> diagnostics{messages};

  void report(const cxx::Diagnostic& diag) override {
    std::string_view fileName;
    std::uint32_t line = 0;
    std::uint32_t column = 0;

    preprocessor()->getTokenStartPosition(diag.token(), &line, &column,
                                          &fileName);

    std::uint32_t endLine = 0;
    std::uint32_t endColumn = 0;

    preprocessor()->getTokenEndPosition(diag.token(), &endLine, &endColumn,
                                        nullptr);

    auto tmp = json::object();

    auto d = diagnostics.emplace_back();

    int s = std::max(int(line) - 1, 0);
    int sc = std::max(int(column) - 1, 0);
    int e = std::max(int(endLine) - 1, 0);
    int ec = std::max(int(endColumn) - 1, 0);

    d.message(diag.message());
    d.range().start(lsp::Position(tmp).line(s).character(sc));
    d.range().end(lsp::Position(tmp).line(e).character(ec));
  }
};

}  // namespace

struct CxxDocument::Private {
  const CLI& cli;
  long version;
  Control control;
  Diagnostics diagnosticsClient;
  TranslationUnit unit{&control, &diagnosticsClient};
  std::shared_ptr<Toolchain> toolchain;

  Private(const CLI& cli, long version) : cli(cli), version(version) {}

  void configure();
};

void CxxDocument::Private::configure() {
  auto preprocesor = unit.preprocessor();

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
    toolchainId = "wasm32";
  }

  if (toolchainId == "darwin" || toolchainId == "macos") {
    toolchain = std::make_unique<MacOSToolchain>(preprocesor);
  } else if (toolchainId == "wasm32") {
    auto wasmToolchain = std::make_unique<Wasm32WasiToolchain>(preprocesor);

    fs::path app_dir;

#if __wasi__
    app_dir = fs::path("/usr/bin/");
#elif !defined(CXX_NO_FILESYSTEM)
    app_dir = std::filesystem::canonical(
        std::filesystem::path(cli.app_name).remove_filename());
#elif __unix__ || __APPLE__
    char* app_name = realpath(cli.app_name.c_str(), nullptr);
    app_dir = fs::path(app_name).remove_filename().string();
    std::free(app_name);
#endif

    wasmToolchain->setAppdir(app_dir.string());

    if (auto paths = cli.get("--sysroot"); !paths.empty()) {
      wasmToolchain->setSysroot(paths.back());
    } else {
      auto sysroot_dir = app_dir / std::string("../lib/wasi-sysroot");
      wasmToolchain->setSysroot(sysroot_dir.string());
    }

    toolchain = std::move(wasmToolchain);
  } else if (toolchainId == "linux") {
    std::string host;
#ifdef __aarch64__
    host = "aarch64";
#elif __x86_64__
    host = "x86_64";
#endif

    std::string arch = cli.getSingle("-arch").value_or(host);
    toolchain = std::make_unique<GCCLinuxToolchain>(preprocesor, arch);
  } else if (toolchainId == "windows") {
    auto windowsToolchain = std::make_unique<WindowsToolchain>(preprocesor);

    if (auto paths = cli.get("-vctoolsdir"); !paths.empty()) {
      windowsToolchain->setVctoolsdir(paths.back());
    }

    if (auto paths = cli.get("-winsdkdir"); !paths.empty()) {
      windowsToolchain->setWinsdkdir(paths.back());
    }

    if (auto versions = cli.get("-winsdkversion"); !versions.empty()) {
      windowsToolchain->setWinsdkversion(versions.back());
    }

    toolchain = std::move(windowsToolchain);
  }

  if (toolchain) {
    control.setMemoryLayout(toolchain->memoryLayout());

    if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();

    if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();

    toolchain->addPredefinedMacros();
  }

  for (const auto& path : cli.get("-I")) {
    preprocesor->addSystemIncludePath(path);
  }

  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of("=");

    if (sep == std::string::npos) {
      preprocesor->defineMacro(macro, "1");
    } else {
      preprocesor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }

  for (const auto& macro : cli.get("-U")) {
    preprocesor->undefMacro(macro);
  }
}

CxxDocument::CxxDocument(const CLI& cli, long version)
    : d(std::make_unique<Private>(cli, version)) {}

void CxxDocument::parse(std::string source, std::string fileName) {
  d->configure();

  auto& unit = d->unit;
  auto& cli = d->cli;

  unit.setSource(std::move(source), fileName);

  auto preprocessor = unit.preprocessor();
  preprocessor->squeeze();

  unit.parse(ParserConfiguration{
      .checkTypes = cli.opt_fcheck,
      .fuzzyTemplateResolution = true,
      .staticAssert = cli.opt_fstatic_assert || cli.opt_fcheck,
      .reflect = !cli.opt_fno_reflect,
      .templates = cli.opt_ftemplates,
  });
}

CxxDocument::~CxxDocument() {}

auto CxxDocument::version() const -> long { return d->version; }

auto CxxDocument::diagnostics() const -> Vector<Diagnostic> {
  return Vector<Diagnostic>(d->diagnosticsClient.messages);
}

auto CxxDocument::translationUnit() const -> TranslationUnit* {
  return &d->unit;
}

}  // namespace cxx::lsp