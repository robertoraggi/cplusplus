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

#include "cxx_document.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/types.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#ifndef CXX_NO_THREADS
#include <atomic>
#endif

#include <format>
#include <iostream>

namespace cxx::lsp {

namespace {

struct Diagnostics final : cxx::DiagnosticsClient {
  json messages = json::array();
  Vector<lsp::Diagnostic> diagnostics{messages};

  void report(const cxx::Diagnostic& diag) override {
    auto start = preprocessor()->tokenStartPosition(diag.token());
    auto end = preprocessor()->tokenEndPosition(diag.token());

    auto tmp = json::object();

    auto d = diagnostics.emplace_back();

    int s = std::max(int(start.line) - 1, 0);
    int sc = std::max(int(start.column) - 1, 0);
    int e = std::max(int(end.line) - 1, 0);
    int ec = std::max(int(end.column) - 1, 0);

    d.message(diag.message());
    d.range().start(lsp::Position(tmp).line(s).character(sc));
    d.range().end(lsp::Position(tmp).line(e).character(ec));
  }
};

}  // namespace

struct CxxDocument::Private {
  const CLI& cli;
  std::string fileName;
  long version;
  Diagnostics diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  std::shared_ptr<Toolchain> toolchain;
  Vector<CompletionItem> completionItems;

#ifndef CXX_NO_THREADS
  std::atomic<bool> cancelled{false};
#else
  bool cancelled{false};
#endif

  Private(const CLI& cli, std::string fileName, long version)
      : cli(cli), fileName(std::move(fileName)), version(version) {}

  void configure();
};

void CxxDocument::Private::configure() {
  auto preprocessor = unit.preprocessor();

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
    toolchainId = "wasm32";
  }

  if (toolchainId == "darwin" || toolchainId == "macos") {
    std::string host = "aarch64";
#ifdef __x86_64__
    host = "x86_64";
#endif

    toolchain = std::make_unique<MacOSToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));
  } else if (toolchainId == "wasm32") {
    auto wasmToolchain = std::make_unique<Wasm32WasiToolchain>(preprocessor);

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
    std::string host = "x86_64";
#ifdef __aarch64__
    host = "aarch64";
#endif

    toolchain = std::make_unique<GCCLinuxToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));

  } else if (toolchainId == "windows") {
    std::string host = "x86_64";
#ifdef __aarch64__
    host = "aarch64";
#endif

    auto windowsToolchain = std::make_unique<WindowsToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));

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
    unit.control()->setMemoryLayout(toolchain->memoryLayout());

    if (!cli.opt_nostdinc) toolchain->addSystemIncludePaths();

    if (!cli.opt_nostdincpp) toolchain->addSystemCppIncludePaths();

    toolchain->addPredefinedMacros();
  }

  for (const auto& path : cli.get("-I")) {
    preprocessor->addSystemIncludePath(path);
  }

  for (const auto& macro : cli.get("-D")) {
    auto sep = macro.find_first_of("=");

    if (sep == std::string::npos) {
      preprocessor->defineMacro(macro, "1");
    } else {
      preprocessor->defineMacro(macro.substr(0, sep), macro.substr(sep + 1));
    }
  }

  for (const auto& macro : cli.get("-U")) {
    preprocessor->undefMacro(macro);
  }
}

CxxDocument::CxxDocument(const CLI& cli, std::string fileName, long version)
    : d(std::make_unique<Private>(cli, std::move(fileName), version)) {}

auto CxxDocument::isCancelled() const -> bool {
#ifndef CXX_NO_THREADS
  return d->cancelled.load();
#else
  return d->cancelled;
#endif
}

void CxxDocument::cancel() {
#ifndef CXX_NO_THREADS
  d->cancelled.store(true);
#else
  d->cancelled = true;
#endif
}

auto CxxDocument::fileName() const -> const std::string& { return d->fileName; }

void CxxDocument::codeCompletionAt(std::string source, std::uint32_t line,
                                   std::uint32_t column,
                                   Vector<CompletionItem> completionItems) {
  std::swap(d->completionItems, completionItems);

  auto& unit = d->unit;

  (void)unit.blockErrors(true);

  unit.preprocessor()->requestCodeCompletionAt(line, column);

  parse(std::move(source));

  std::swap(d->completionItems, completionItems);
}

void CxxDocument::parse(std::string source) {
  d->configure();

  auto& unit = d->unit;
  auto& cli = d->cli;

  auto preprocessor = unit.preprocessor();

  DefaultPreprocessorState state{*preprocessor};

  unit.beginPreprocessing(std::move(source), d->fileName);

  while (state) {
    if (isCancelled()) break;
    std::visit(state, unit.continuePreprocessing());
  }

  unit.endPreprocessing();

  auto stopParsingPredicate = [this] { return isCancelled(); };

  auto complete = [this](const CodeCompletionContext& context) {
    if (auto memberCompletionContext =
            std::get_if<MemberCompletionContext>(&context)) {
      // simple member completion
      auto objectType = memberCompletionContext->objectType;

      if (auto pointerType = type_cast<PointerType>(objectType)) {
        objectType = type_cast<ClassType>(pointerType->elementType());
      }

      if (auto classType = type_cast<ClassType>(objectType)) {
        auto classSymbol = classType->symbol();
        for (auto member : views::members(classSymbol)) {
          if (!member->name()) continue;
          auto item = d->completionItems.emplace_back();
          item.label(to_string(member->name()));
        }
      }
    }
  };

  unit.parse(ParserConfiguration{
      .checkTypes = cli.opt_fcheck,
      .fuzzyTemplateResolution = true,
      .reflect = !cli.opt_fno_reflect,
      .stopParsingPredicate = stopParsingPredicate,
      .complete = complete,
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

auto CxxDocument::textOf(AST* ast) -> std::optional<std::string_view> {
  return textInRange(ast->firstSourceLocation(), ast->lastSourceLocation());
}

auto CxxDocument::textInRange(SourceLocation start, SourceLocation end)
    -> std::optional<std::string_view> {
  auto& unit = d->unit;
  auto preprocessor = unit.preprocessor();

  const auto startToken = unit.tokenAt(start);
  const auto endToken = unit.tokenAt(end.previous());

  if (startToken.fileId() != endToken.fileId()) {
    return std::nullopt;
  }

  std::string_view source = preprocessor->source(startToken.fileId());

  const auto offset = startToken.offset();
  const auto length = endToken.offset() + endToken.length() - offset;

  return source.substr(offset, length);
}

}  // namespace cxx::lsp