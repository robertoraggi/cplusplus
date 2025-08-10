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

#include "frontend.h"

#include <cxx/ast.h>
#include <cxx/ast_pretty_printer.h>
#include <cxx/ast_printer.h>
#include <cxx/ast_visitor.h>
#include <cxx/cli.h>
#include <cxx/control.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/lexer.h>
#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#ifdef CXX_WITH_MLIR
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#endif

#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include "check_expression_types.h"
#include "dump_tokens.h"
#include "verify_diagnostics_client.h"

namespace cxx {

Frontend::Frontend(const CLI& cli, std::string fileName)
    : cli(cli), fileName_(std::move(fileName)) {
  diagnosticsClient_ = std::make_unique<VerifyDiagnosticsClient>();
  unit_ = std::make_unique<TranslationUnit>(diagnosticsClient_.get());

  actions_.emplace_back([this]() { showSearchPaths(std::cerr); });
  actions_.emplace_back([this]() { preprocess(); });
  actions_.emplace_back([this]() { printPreprocessedText(); });
  actions_.emplace_back([this]() { dumpMacros(std::cout); });
  actions_.emplace_back([this]() { dumpTokens(std::cout); });
  actions_.emplace_back([this]() { unit_->preprocessor()->squeeze(); });
  actions_.emplace_back([this]() { parse(); });
  actions_.emplace_back([this]() { dumpSymbols(std::cout); });
  actions_.emplace_back([this]() { dumpAst(); });
  actions_.emplace_back([this]() { printAstIfNeeded(); });
  actions_.emplace_back([this]() { serializeAst(); });
  actions_.emplace_back([this]() { emitIR(); });
}

Frontend::~Frontend() {}

auto Frontend::translationUnit() const -> TranslationUnit* {
  return unit_.get();
}

auto Frontend::toolchain() const -> Toolchain* { return toolchain_.get(); }

auto Frontend::fileName() const -> const std::string& { return fileName_; }

void Frontend::addAction(std::function<void()> action) {
  actions_.emplace_back(std::move(action));
}

auto Frontend::operator()() -> bool {
  prepare();
  preparePreprocessor();

  for (const auto& action : actions_) {
    if (shouldExit_) break;
    action();
  }

  diagnosticsClient_->verifyExpectedDiagnostics();

  return !diagnosticsClient_->hasErrors();
}

void Frontend::withOutputStream(
    const std::optional<std::string>& extension,
    const std::function<void(std::ostream&)>& action) {
  auto explicitOutput = cli.getSingle("-o");

  if (explicitOutput == "-" || (!explicitOutput.has_value() &&
                                (!extension.has_value() || fileName_ == "-"))) {
    action(std::cout);
    return;
  }

  auto inputFile = fs::path{fileName_}.filename();
  auto defaultOutputFile = inputFile.replace_extension(*extension);

  auto outputFile = cli.getSingle("-o").value_or(defaultOutputFile.string());

  std::ofstream output(outputFile);
  action(output);
}

#ifdef CXX_WITH_MLIR
void Frontend::withRawOutputStream(
    const std::optional<std::string>& extension,
    const std::function<void(llvm::raw_ostream&)>& action) {
  auto explicitOutput = cli.getSingle("-o");

  if (explicitOutput == "-" || (!explicitOutput.has_value() &&
                                (!extension.has_value() || fileName_ == "-"))) {
    action(llvm::outs());
    return;
  }

  auto inputFile = fs::path{fileName_}.filename();
  auto defaultOutputFile = inputFile.replace_extension(*extension);

  auto outputFile = cli.getSingle("-o").value_or(defaultOutputFile.string());

  std::error_code error_code;
  llvm::raw_fd_ostream output(outputFile, error_code);
  action(output);
}
#endif

void Frontend::printPreprocessedText() {
  if (!cli.opt_E && !cli.opt_Eonly) {
    return;
  }

  if (cli.opt_dM) {
    // If we are only dumping macros, we don't need to output the preprocessed
    // text.
    return;
  }

  shouldExit_ = true;

  if (cli.opt_Eonly) {
    // If we are only preprocessing, we don't need to output the preprocessed
    return;
  }

  withOutputStream(std::nullopt, [&](std::ostream& out) {
    unit_->preprocessor()->getPreprocessedText(unit_->tokens(), out);
  });
}

void Frontend::preprocess() {
  auto source = readAll(fileName_);

  if (!source.has_value()) {
    std::cerr << std::format("cxx: No such file or directory: '{}'\n",
                             fileName_);
    shouldExit_ = true;
    exitStatus_ = EXIT_FAILURE;
    return;
  }

  unit_->setSource(std::move(*source), fileName_);
}

void Frontend::dumpMacros(std::ostream& out) {
  if (!cli.opt_E && !cli.opt_dM) return;

  unit_->preprocessor()->printMacros(out);

  shouldExit_ = true;
}

void Frontend::prepare() {
  auto preprocessor = unit_->preprocessor();

  const auto lang = cli.getSingle("-x");

  if (lang == "c" || (!lang.has_value() && fileName_.ends_with(".c"))) {
    // set the language to C
    preprocessor->setLanguage(LanguageKind::kC);
  }

  if (cli.opt_verify) {
    diagnosticsClient_->setVerify(true);
    preprocessor->setCommentHandler(diagnosticsClient_.get());
  }

  auto toolchainId = cli.getSingle("-toolchain");

  if (!toolchainId) {
    toolchainId = "wasm32";
  }

  if (toolchainId == "darwin" || toolchainId == "macos") {
    // on macOS we default to aarch64, since it is the most common
    std::string host = "aarch64";

#if __x86_64__
    host = "x86_64";
#endif

    toolchain_ = std::make_unique<MacOSToolchain>(
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

    toolchain_ = std::move(wasmToolchain);
  } else if (toolchainId == "linux") {
    // on linux we default to x86_64, unless the host is aarch64
    std::string host = "x86_64";

#ifdef __aarch64__
    host = "aarch64";
#endif

    toolchain_ = std::make_unique<GCCLinuxToolchain>(
        preprocessor, cli.getSingle("-arch").value_or(host));
  } else if (toolchainId == "windows") {
    // on linux we default to x86_64, unless the host is aarch64
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

    toolchain_ = std::move(windowsToolchain);
  }

  unit_->control()->setMemoryLayout(toolchain_->memoryLayout());
}

void Frontend::preparePreprocessor() {
  auto preprocessor = unit_->preprocessor();

  if (cli.opt_P) {
    preprocessor->setOmitLineMarkers(true);
  }

  if (!cli.opt_nostdinc) {
    toolchain_->addSystemIncludePaths();
  }

  if (!cli.opt_nostdincpp) {
    toolchain_->addSystemCppIncludePaths();
  }

  toolchain_->addPredefinedMacros();

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

  if (cli.opt_H && (cli.opt_E || cli.opt_Eonly)) {
    preprocessor->setOnWillIncludeHeader(
        [&](const std::string& header, int level) {
          std::string fill(level, '.');
          std::cout << std::format("{} {}\n", fill, header);
        });
  }
}

void Frontend::parse() {
  unit_->parse(ParserConfiguration{
      .checkTypes = cli.opt_fcheck || unit_->language() == LanguageKind::kC,
      .fuzzyTemplateResolution = true,
      .reflect = !cli.opt_fno_reflect,
  });

  if (cli.opt_freport_missing_types) {
    (void)checkExpressionTypes(*unit_);
  }
}

void Frontend::dumpTokens(std::ostream& out) {
  if (!cli.opt_dump_tokens) return;

  auto dumpTokens = DumpTokens{cli};
  dumpTokens(*unit_, out);

  shouldExit_ = true;
}

void Frontend::dumpSymbols(std::ostream& out) {
  if (!cli.opt_dump_symbols) return;
  auto globalScope = unit_->globalScope();
  auto globalNamespace = globalScope->owner();
  cxx::dump(out, globalNamespace);
}

void Frontend::dumpAst() {
  if (!cli.opt_ast_dump) return;
  auto printAST = ASTPrinter{unit_.get(), std::cout};
  printAST(unit_->ast());
}

void Frontend::printAstIfNeeded() {
  if (!cli.opt_ast_print) return;
  auto prettyPrinter = ASTPrettyPrinter{unit_.get(), std::cout};
  prettyPrinter(unit_->ast());
}

void Frontend::serializeAst() {
  if (!cli.opt_emit_ast) return;
  auto outputFile = fs::path{fileName_}.filename().replace_extension(".ast");
  std::ofstream out(outputFile.string(), std::ios::binary);
  (void)unit_->serialize(out);
}

void Frontend::showSearchPaths(std::ostream& out) {
  if (!cli.opt_v) return;

  out << std::format("#include <...> search starts here:\n");

  const auto& searchPaths = unit_->preprocessor()->systemIncludePaths();

  for (const auto& path : searchPaths | std::views::reverse) {
    out << std::format(" {}\n", path);
  }

  out << std::format("End of search list.\n");
}

void Frontend::emitIR() {
  if (!cli.opt_emit_ir) return;

#ifdef CXX_WITH_MLIR
  mlir::MLIRContext context;
  context.loadDialect<mlir::cxx::CxxDialect>();

  auto codegen = cxx::Codegen{context, unit_.get()};

  auto ir = codegen(unit_->ast());

  if (failed(lowerToMLIR(ir.module))) {
    std::cerr << "cxx: failed to lower C++ AST to MLIR" << std::endl;
    shouldExit_ = true;
    exitStatus_ = EXIT_FAILURE;
    return;
  }

  mlir::OpPrintingFlags flags;
  if (cli.opt_g) {
    flags.enableDebugInfo(true, false);
  }

  withRawOutputStream(std::nullopt, [&](llvm::raw_ostream& out) {
    ir.module->print(out, flags);
  });

#endif
}

auto Frontend::readAll(const std::string& fileName, std::istream& in)
    -> std::optional<std::string> {
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);
  return code;
}

auto Frontend::readAll(const std::string& fileName)
    -> std::optional<std::string> {
  if (fileName == "-" || fileName.empty()) return readAll("<stdin>", std::cin);
  if (std::ifstream stream(fileName); stream) return readAll(fileName, stream);
  return std::nullopt;
}

}  // namespace cxx
