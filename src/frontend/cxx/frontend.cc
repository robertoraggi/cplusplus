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
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/wasm32_wasi_toolchain.h>
#include <cxx/windows_toolchain.h>

#ifdef CXX_WITH_MLIR
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Pass.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#endif

#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include "check_expression_types.h"
#include "dump_tokens.h"
#include "verify_diagnostics_client.h"

namespace cxx {

struct Frontend::Private {
  Frontend& frontend;
  const CLI& cli;
  std::string fileName_;
  std::unique_ptr<TranslationUnit> unit_;
  std::unique_ptr<VerifyDiagnosticsClient> diagnosticsClient_;
  std::unique_ptr<Toolchain> toolchain_;
  std::vector<std::function<void()>> actions_;
#ifdef CXX_WITH_MLIR
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::ModuleOp module_;
  std::unique_ptr<llvm::LLVMContext> llvmContext_;
  std::unique_ptr<llvm::Module> llvmModule_;
#endif
  bool shouldExit_ = false;
  int exitStatus_ = 0;

  Private(Frontend& frontend, const CLI& cli, std::string fileName);
  ~Private();

  [[nodiscard]] auto needsIR() const -> bool {
    return cli.opt_emit_ir || cli.opt_emit_llvm || cli.opt_S || cli.opt_c;
  }

  [[nodiscard]] auto needsLLVMIR() const -> bool {
    return cli.opt_emit_llvm || cli.opt_S || cli.opt_c;
  }

  void prepare();
  void preparePreprocessor();
  void preprocess();
  void parse();
  void showSearchPaths(std::ostream& out);
  void dumpTokens(std::ostream& out);
  void dumpSymbols(std::ostream& out);
  void serializeAst();
  void dumpAst();
  void printAstIfNeeded();
  void generateIR();
  void emitIR();
  void emitLLVMIR();
  void emitCode();
  void emitObjectFile();
  void printPreprocessedText();
  void dumpMacros(std::ostream& out);

  [[nodiscard]] auto readAll(const std::string& fileName, std::istream& in)
      -> std::optional<std::string>;

  [[nodiscard]] auto readAll(const std::string& fileName)
      -> std::optional<std::string>;

  void withOutputStream(const std::optional<std::string>& extension,
                        const std::function<void(std::ostream&)>& action);

#ifdef CXX_WITH_MLIR
  void withRawOutputStream(
      const std::optional<std::string>& extension,
      const std::function<void(llvm::raw_pwrite_stream&)>& action);
#endif
};

Frontend::Frontend(const CLI& cli, std::string fileName) {
  priv = std::make_unique<Private>(*this, cli, std::move(fileName));
}

Frontend::~Frontend() {}

auto Frontend::translationUnit() const -> TranslationUnit* {
  return priv->unit_.get();
}

auto Frontend::toolchain() const -> Toolchain* {
  return priv->toolchain_.get();
}

auto Frontend::fileName() const -> const std::string& {
  return priv->fileName_;
}

void Frontend::addAction(std::function<void()> action) {
  priv->actions_.emplace_back(std::move(action));
}

auto Frontend::operator()() -> bool {
  priv->prepare();
  priv->preparePreprocessor();

  for (const auto& action : priv->actions_) {
    if (priv->shouldExit_) break;
    action();
  }

  priv->diagnosticsClient_->verifyExpectedDiagnostics();

  return !priv->diagnosticsClient_->hasErrors();
}

Frontend::Private::Private(Frontend& frontend, const CLI& cli,
                           std::string fileName)
    : frontend(frontend), cli(cli), fileName_(std::move(fileName)) {
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
  actions_.emplace_back([this]() { generateIR(); });
  actions_.emplace_back([this]() { emitIR(); });
  actions_.emplace_back([this]() { emitLLVMIR(); });
  actions_.emplace_back([this]() { emitCode(); });
}

Frontend::Private::~Private() {}

void Frontend::Private::withOutputStream(
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
void Frontend::Private::withRawOutputStream(
    const std::optional<std::string>& extension,
    const std::function<void(llvm::raw_pwrite_stream&)>& action) {
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

void Frontend::Private::printPreprocessedText() {
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

void Frontend::Private::preprocess() {
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

void Frontend::Private::dumpMacros(std::ostream& out) {
  if (!cli.opt_E && !cli.opt_dM) return;

  unit_->preprocessor()->printMacros(out);

  shouldExit_ = true;
}

void Frontend::Private::prepare() {
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

  toolchain_->initMemoryLayout();
}

void Frontend::Private::preparePreprocessor() {
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

void Frontend::Private::parse() {
  unit_->parse(ParserConfiguration{
      .checkTypes =
          cli.opt_fcheck || needsIR() || unit_->language() == LanguageKind::kC,
      .fuzzyTemplateResolution = true,
  });

  if (cli.opt_freport_missing_types) {
    (void)checkExpressionTypes(*unit_);
  }
}

void Frontend::Private::dumpTokens(std::ostream& out) {
  if (!cli.opt_dump_tokens) return;

  auto dumpTokens = DumpTokens{cli};
  dumpTokens(*unit_, out);

  shouldExit_ = true;
}

void Frontend::Private::dumpSymbols(std::ostream& out) {
  if (!cli.opt_dump_symbols) return;
  auto globalScope = unit_->globalScope();
  auto globalNamespace = globalScope;
  cxx::dump(out, globalNamespace);
}

void Frontend::Private::dumpAst() {
  if (!cli.opt_ast_dump) return;
  auto printAST = ASTPrinter{unit_.get(), std::cout};
  printAST(unit_->ast());
}

void Frontend::Private::printAstIfNeeded() {
  if (!cli.opt_ast_print) return;
  auto prettyPrinter = ASTPrettyPrinter{unit_.get(), std::cout};
  prettyPrinter(unit_->ast());
}

void Frontend::Private::serializeAst() {
  if (!cli.opt_emit_ast) return;
  auto outputFile = fs::path{fileName_}.filename().replace_extension(".ast");
  std::ofstream out(outputFile.string(), std::ios::binary);
  (void)unit_->serialize(out);
}

void Frontend::Private::showSearchPaths(std::ostream& out) {
  if (!cli.opt_v) return;

  out << std::format("#include <...> search starts here:\n");

  const auto& searchPaths = unit_->preprocessor()->systemIncludePaths();

  for (const auto& path : searchPaths | std::views::reverse) {
    out << std::format(" {}\n", path);
  }

  out << std::format("End of search list.\n");
}

void Frontend::Private::generateIR() {
  if (cli.opt_fsyntax_only) return;
  if (!needsIR()) return;

#ifdef CXX_WITH_MLIR
  context_ = std::make_unique<mlir::MLIRContext>();
  context_->loadDialect<mlir::cxx::CxxDialect>();

  auto codegen = cxx::Codegen{*context_, unit_.get()};

  auto ir = codegen(unit_->ast());

  if (succeeded(lowerToMLIR(ir.module))) {
    module_ = ir.module;
    return;
  }

  std::cerr << "cxx: failed to lower C++ AST to MLIR" << std::endl;
  shouldExit_ = true;
  exitStatus_ = EXIT_FAILURE;
#endif
}

void Frontend::Private::emitIR() {
  if (!cli.opt_emit_ir) return;

#ifdef CXX_WITH_MLIR
  if (!module_) return;

  shouldExit_ = true;

  mlir::OpPrintingFlags flags;
  if (cli.opt_g) {
    auto prettyForm = true;
    flags.enableDebugInfo(true, prettyForm);
  }

  withRawOutputStream(std::nullopt, [&](llvm::raw_ostream& out) {
    module_->print(out, flags);
  });

#endif
}

void Frontend::Private::emitLLVMIR() {
  if (!needsLLVMIR()) return;

#ifdef CXX_WITH_MLIR
  if (!module_) return;

  llvmContext_ = std::make_unique<llvm::LLVMContext>();
  llvmModule_ = exportToLLVMIR(module_, *llvmContext_);

  if (!llvmModule_) {
    std::cerr << "cxx: failed to lower MLIR module to LLVM IR" << std::endl;
    shouldExit_ = true;
    exitStatus_ = EXIT_FAILURE;
    return;
  }

  if (!cli.opt_emit_llvm) return;

  shouldExit_ = true;

  withRawOutputStream(
      ".ll", [&](llvm::raw_ostream& out) { llvmModule_->print(out, nullptr); });

#endif
}

void Frontend::Private::emitCode() {
  if (!cli.opt_S && !cli.opt_c) return;
#ifdef CXX_WITH_MLIR
  llvm::InitializeAllAsmPrinters();

  auto triple = llvm::Triple{toolchain_->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    std::cerr << std::format("cxx: cannot find target for triple '{}': {}\n",
                             triple.getTriple(), error);
    shouldExit_ = true;
    exitStatus_ = EXIT_FAILURE;
    return;
  }

  llvm::TargetOptions opt;

  auto RM = std::optional<llvm::Reloc::Model>();

  auto targetMachine =
      target->createTargetMachine(llvm::Triple{triple}, "generic", "", opt, RM);

  if (!targetMachine) {
    std::cerr << std::format("cxx: cannot create target machine for '{}': {}\n",
                             triple.getTriple(), error);
    shouldExit_ = true;
    exitStatus_ = EXIT_FAILURE;
    return;
  }

  std::string extension;
  if (cli.opt_S) {
    extension = ".s";
  } else if (cli.opt_c) {
    extension = ".o";
  }

  withRawOutputStream(extension, [&](llvm::raw_pwrite_stream& out) {
    llvm::legacy::PassManager pm;

    llvm::CodeGenFileType fileType;
    if (cli.opt_S) {
      fileType = llvm::CodeGenFileType::AssemblyFile;
    } else {
      fileType = llvm::CodeGenFileType::ObjectFile;
    }

    if (targetMachine->addPassesToEmitFile(pm, out, nullptr, fileType)) {
      std::cerr << "cxx: target machine cannot emit assembly\n";
      shouldExit_ = true;
      exitStatus_ = EXIT_FAILURE;
      return;
    }

    pm.run(*llvmModule_);
    out.flush();
  });
#endif
}

auto Frontend::Private::readAll(const std::string& fileName, std::istream& in)
    -> std::optional<std::string> {
  std::string code;
  char buffer[4 * 1024];
  do {
    in.read(buffer, sizeof(buffer));
    code.append(buffer, in.gcount());
  } while (in);
  return code;
}

auto Frontend::Private::readAll(const std::string& fileName)
    -> std::optional<std::string> {
  if (fileName == "-" || fileName.empty()) return readAll("<stdin>", std::cin);
  if (std::ifstream stream(fileName); stream) return readAll(fileName, stream);
  return std::nullopt;
}

}  // namespace cxx
