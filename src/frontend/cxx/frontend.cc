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
#include <cxx/lexer.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/symbols.h>
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

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
  std::optional<std::string> objectOutput_;
#ifdef CXX_WITH_MLIR
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::ModuleOp module_;
  std::unique_ptr<llvm::LLVMContext> llvmContext_;
  std::unique_ptr<llvm::Module> llvmModule_;
#endif
  bool shouldExit_ = false;
  bool failed_ = false;

  Private(Frontend& frontend, const CLI& cli, std::string fileName);
  ~Private();

  void fail() {
    shouldExit_ = true;
    failed_ = true;
  }

  void exitIfErrors() {
    if (diagnosticsClient_->hasErrors()) shouldExit_ = true;
  }

  [[nodiscard]] auto needsIR() const -> bool {
    return cli.opt_emit_cxx_ir || cli.opt_emit_mlir || cli.opt_emit_llvm ||
           cli.opt_S || cli.opt_c || objectOutput_.has_value();
  }

  [[nodiscard]] auto needsLLVMIR() const -> bool {
    return cli.opt_emit_llvm || cli.opt_S || cli.opt_c ||
           objectOutput_.has_value();
  }

  void prepare();
  void preparePreprocessor();
  void preprocess();
  void parse();
  void showSearchPaths(std::ostream& out);
  void dumpTokens(std::ostream& out);
  void dumpSymbols(std::ostream& out);
  void dumpRecordLayouts(std::ostream& out);
  void serializeAst();
  void dumpAst();
  void printAstIfNeeded();
  void generateIR();
  void emitCxxIR();
  void lowerIR();
  void emitMLIR();
  void emitLLVMIR();
  void emitCode();
  void emitObjectFile();
  void printPreprocessedText();
  void writeDepFile();
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

void Frontend::setObjectOutput(std::string path) {
  priv->objectOutput_ = std::move(path);
}

auto Frontend::operator()() -> bool {
  priv->prepare();
  priv->preparePreprocessor();

  for (const auto& action : priv->actions_) {
    if (priv->shouldExit_) break;
    action();
  }

  priv->diagnosticsClient_->verifyExpectedDiagnostics();

  return !priv->diagnosticsClient_->hasErrors() && !priv->failed_;
}

Frontend::Private::Private(Frontend& frontend, const CLI& cli,
                           std::string fileName)
    : frontend(frontend), cli(cli), fileName_(std::move(fileName)) {
  diagnosticsClient_ = std::make_unique<VerifyDiagnosticsClient>();
  unit_ = std::make_unique<TranslationUnit>(diagnosticsClient_.get());

  actions_.emplace_back([this]() { showSearchPaths(std::cerr); });
  actions_.emplace_back([this]() { preprocess(); });
  actions_.emplace_back([this]() { writeDepFile(); });
  actions_.emplace_back([this]() { printPreprocessedText(); });
  actions_.emplace_back([this]() { dumpMacros(std::cout); });
  actions_.emplace_back([this]() { dumpTokens(std::cout); });
  actions_.emplace_back([this]() { unit_->preprocessor()->squeeze(); });
  actions_.emplace_back([this]() { parse(); });
  actions_.emplace_back([this]() { dumpSymbols(std::cout); });
  actions_.emplace_back([this]() { dumpRecordLayouts(std::cout); });
  actions_.emplace_back([this]() { dumpAst(); });
  actions_.emplace_back([this]() { printAstIfNeeded(); });
  actions_.emplace_back([this]() { serializeAst(); });
  actions_.emplace_back([this]() { exitIfErrors(); });
  actions_.emplace_back(
      [this]() { toolchain_->applyEntryPointAbi(unit_.get()); });
  actions_.emplace_back([this]() { generateIR(); });
  actions_.emplace_back([this]() { exitIfErrors(); });
  actions_.emplace_back([this]() { emitCxxIR(); });
  actions_.emplace_back([this]() { lowerIR(); });
  actions_.emplace_back([this]() { emitMLIR(); });
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
    return;
  }

  shouldExit_ = true;

  if (cli.opt_Eonly) {
    return;
  }

  withOutputStream(std::nullopt, [&](std::ostream& out) {
    unit_->preprocessor()->getPreprocessedText(unit_->tokens(), out);
  });
}

static auto quoteDepfileTarget(const std::string& target) -> std::string {
  std::string result;
  for (char ch : target) {
    if (ch == '$') result += '$';
    if (ch == '#' || ch == ' ' || ch == '\t') result += '\\';
    result += ch;
  }
  return result;
}

static void formatDepFile(std::ostream& out, const std::string& target,
                          const std::vector<std::string>& deps, bool phony) {
  out << target << ':';
  for (const auto& dep : deps) out << " \\\n  " << dep;
  out << '\n';
  if (!phony) return;
  for (const auto& dep : deps) out << '\n' << dep << ":\n";
}

void Frontend::Private::writeDepFile() {
  bool toStdout = cli.opt_M || cli.opt_MM;
  bool toFile = cli.opt_MD || cli.opt_MMD;
  if (!toStdout && !toFile) return;

  bool skipSystem = cli.opt_MM || cli.opt_MMD;
  auto preprocessor = unit_->preprocessor();

  std::vector<std::string> deps;
  deps.push_back(fileName_);
  for (const auto& [f, isSys] : preprocessor->includedFiles()) {
    if (skipSystem && isSys) continue;
    deps.push_back(f);
  }

  auto mqTarget = cli.getSingle("-MQ");
  auto mtTarget = cli.getSingle("-MT");
  std::string target;
  if (mqTarget) {
    target = quoteDepfileTarget(*mqTarget);
  } else if (mtTarget) {
    target = *mtTarget;
  } else {
    auto inputFile = fs::path{fileName_}.filename();
    target = inputFile.replace_extension(".o").string();
  }

  if (toStdout) {
    formatDepFile(std::cout, target, deps, cli.opt_MP);
    shouldExit_ = true;
    return;
  }

  auto mfPath = cli.getSingle("-MF");
  std::string depFileName;
  if (mfPath) {
    depFileName = *mfPath;
  } else {
    auto inputFile = fs::path{fileName_}.filename();
    depFileName = inputFile.replace_extension(".d").string();
  }

  std::ofstream depOut(depFileName);
  formatDepFile(depOut, target, deps, cli.opt_MP);
}

void Frontend::Private::preprocess() {
  auto source = readAll(fileName_);

  if (!source.has_value()) {
    std::cerr << std::format("cxx: No such file or directory: '{}'\n",
                             fileName_);
    fail();
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
    preprocessor->setLanguage(LanguageKind::kC);
  }

  if (cli.opt_verify) {
    diagnosticsClient_->setVerify(true);
    preprocessor->setCommentHandler(diagnosticsClient_.get());
  }

  std::string error;
  toolchain_ = createToolchain(cli, preprocessor, error);
  if (!error.empty()) {
    std::cerr << error << '\n';
    fail();
    return;
  }
  if (!toolchain_) {
    auto id = cli.getSingle("-toolchain").value_or("wasm32");
    std::cerr << std::format("cxx: unknown toolchain '{}'\n", id);
    fail();
    return;
  }
  unit_->control()->setMemoryLayout(toolchain_->memoryLayout());
}

void Frontend::Private::preparePreprocessor() {
  auto preprocessor = unit_->preprocessor();

  if (cli.opt_P) {
    preprocessor->setOmitLineMarkers(true);
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
  if (auto errorLimitStr = cli.getSingle("-ferror-limit")) {
    int limit = std::atoi(errorLimitStr->c_str());
    if (limit > 0) diagnosticsClient_->setErrorLimit(limit);
  }

  bool checkTypes = cli.opt_fcheck;
  if (cli.opt_fvalidate_ast) checkTypes = true;
  if (needsIR()) checkTypes = true;
  if (unit_->language() == LanguageKind::kC) checkTypes = true;

  unit_->parse(ParserConfiguration{
      .checkTypes = checkTypes,
      .validateAst = cli.opt_fvalidate_ast,
      .allowUnprototypedFunctions = cli.opt_fno_strict_prototypes,
      .stopParsingPredicate = [this]() -> bool {
        return diagnosticsClient_->errorLimitReached();
      },
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
  cxx::dump(out, globalNamespace, unit_.get());
}

void Frontend::Private::dumpRecordLayouts(std::ostream& out) {
  if (!cli.opt_dump_record_layouts) return;

  auto globalScope = unit_->globalScope();

  auto classKeyword = [](ClassSymbol* cls) -> std::string_view {
    return cls->isUnion() ? "union" : "struct";
  };

  std::function<void(ClassSymbol*, const ClassLayout*, int indent,
                     std::uint64_t baseOffset)>
      dumpClassMembers;

  dumpClassMembers = [&](ClassSymbol* classSymbol, const ClassLayout* layout,
                         int indent, std::uint64_t baseOffset) {
    std::string pad(indent * 2, ' ');

    for (auto base : classSymbol->baseClasses()) {
      if (base->isVirtual()) continue;
      auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClassSymbol) continue;

      auto baseInfo = layout->getBaseInfo(baseClassSymbol);
      if (!baseInfo) continue;

      auto absOffset = baseOffset + baseInfo->offset;
      out << std::format("{:>9} |{}{} {} (base)\n", absOffset, pad,
                         classKeyword(baseClassSymbol),
                         to_string(baseClassSymbol->name()));

      auto baseLayout = baseClassSymbol->layout();
      if (baseLayout) {
        dumpClassMembers(baseClassSymbol, baseLayout, indent + 1, absOffset);
      }
    }

    for (auto field :
         cxx::views::members(classSymbol) | cxx::views::non_static_fields) {
      auto fieldInfo = layout->getFieldInfo(field);
      if (!fieldInfo) continue;

      auto absOffset = baseOffset + fieldInfo->offset;

      if (!field->name()) {
        if (auto classType = type_cast<ClassType>(field->type())) {
          auto nestedClass = classType->symbol();
          if (nestedClass && !nestedClass->name()) {
            out << std::format("{:>9} |{}{} (anonymous) \n", absOffset, pad,
                               classKeyword(nestedClass));

            auto nestedLayout = nestedClass->layout();
            if (nestedLayout) {
              dumpClassMembers(nestedClass, nestedLayout, indent + 1,
                               absOffset);
            }
            continue;
          }
        }
      }

      auto typeStr = to_string(field->type());
      auto nameStr = field->name() ? to_string(field->name()) : "";

      if (field->isBitField() && fieldInfo->bitWidth > 0) {
        auto absByte = absOffset + fieldInfo->bitOffset / 8;
        auto startBit = static_cast<int>(fieldInfo->bitOffset % 8);
        auto endBit = startBit + static_cast<int>(fieldInfo->bitWidth) - 1;
        auto offsetStr = std::format("{}:{}-{}", absByte, startBit, endBit);
        out << std::format("{:>9} |{}{} {}\n", offsetStr, pad, typeStr,
                           nameStr);
      } else if (field->isBitField() && fieldInfo->bitWidth == 0) {
        auto absByte = absOffset + fieldInfo->bitOffset / 8;
        auto startBit = static_cast<int>(fieldInfo->bitOffset % 8);
        auto offsetStr = std::format("{}:{}-", absByte, startBit);
        out << std::format("{:>9} |{}{}\n", offsetStr, pad, typeStr);
      } else {
        out << std::format("{:>9} |{}{} {}\n", absOffset, pad, typeStr,
                           nameStr);
      }
    }
  };

  std::function<void(ScopeSymbol*)> visitScope;
  visitScope = [&](ScopeSymbol* scope) {
    for (auto member : scope->members()) {
      if (auto classSymbol = symbol_cast<ClassSymbol>(member)) {
        auto layout = classSymbol->layout();
        if (!layout) continue;

        out << std::format("\n*** Dumping AST Record Layout\n");
        out << std::format("{:>9} | {} {}\n", 0, classKeyword(classSymbol),
                           to_string(classSymbol->name()));

        dumpClassMembers(classSymbol, layout, 1, 0);

        for (auto vbase : layout->virtualBases()) {
          auto baseInfo = layout->getBaseInfo(vbase);
          if (!baseInfo) continue;
          out << std::format("{:>9} | {}{} {} (virtual base)\n",
                             baseInfo->offset, std::string(2, ' '),
                             classKeyword(vbase), to_string(vbase->name()));
          if (auto vbaseLayout = vbase->layout()) {
            dumpClassMembers(vbase, vbaseLayout, 2, baseInfo->offset);
          }
        }

        out << std::format("{:>9} | [sizeof={}, dsize={}, align={},\n", "",
                           layout->size(), layout->size(), layout->alignment());
        if (layout->virtualBases().empty()) {
          out << std::format("{:>9} |  nvsize={}, nvalign={}]\n", "",
                             layout->size(), layout->alignment());
        } else {
          out << std::format("{:>9} |  nvsize={}, nvalign={}]\n", "",
                             layout->nonVirtualSize(),
                             layout->nonVirtualAlignment());
        }
      }

      if (auto nestedScope = symbol_cast<ScopeSymbol>(member)) {
        visitScope(nestedScope);
      }
    }
  };

  visitScope(globalScope);
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

  auto preprocessor = unit_->preprocessor();

  out << std::format("#include \"...\" search starts here:\n");
  for (const auto& path : preprocessor->quoteIncludePaths()) {
    out << std::format(" {}\n", path);
  }
  for (const auto& path : preprocessor->userIncludePaths()) {
    out << std::format(" {}\n", path);
  }

  out << std::format("#include <...> search starts here:\n");
  for (const auto& path : preprocessor->systemIncludePaths()) {
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

  auto codegen = cxx::Codegen{*context_, unit_.get(), cli.opt_g};

  auto ir = codegen(unit_->ast());
  module_ = ir.module;

#endif
}

void Frontend::Private::emitCxxIR() {
  if (!cli.opt_emit_cxx_ir) return;

#ifdef CXX_WITH_MLIR
  if (!module_) return;

  shouldExit_ = true;

  mlir::OpPrintingFlags flags;
  if (cli.opt_g) {
    auto prettyForm = true;
    flags.enableDebugInfo(true, prettyForm);
  }

  withRawOutputStream(
      "mlir", [&](llvm::raw_ostream& out) { module_->print(out, flags); });

#endif
}

void Frontend::Private::lowerIR() {
#ifdef CXX_WITH_MLIR
  if (!module_) return;
  if (cli.opt_fsyntax_only) return;

  auto needsLowering = cli.opt_emit_mlir || needsLLVMIR();

  if (!needsLowering) return;

  if (succeeded(lowerToMLIR(module_))) {
    return;
  }

  std::cerr << "cxx: failed to lower C++ AST to MLIR" << std::endl;
  fail();
  module_ = nullptr;
#endif
}

void Frontend::Private::emitMLIR() {
  if (!cli.opt_emit_mlir) return;

#ifdef CXX_WITH_MLIR
  if (!module_) return;

  shouldExit_ = true;

  mlir::OpPrintingFlags flags;
  if (cli.opt_g) {
    auto prettyForm = true;
    flags.enableDebugInfo(true, prettyForm);
  }

  withRawOutputStream(
      "mlir", [&](llvm::raw_ostream& out) { module_->print(out, flags); });

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
    fail();
    return;
  }

  if (!cli.opt_emit_llvm) return;

  shouldExit_ = true;

  withRawOutputStream(
      ".ll", [&](llvm::raw_ostream& out) { llvmModule_->print(out, nullptr); });

#endif
}

void Frontend::Private::emitCode() {
  if (!cli.opt_S && !cli.opt_c && !objectOutput_.has_value()) return;
#ifdef CXX_WITH_MLIR
  if (!llvmModule_) return;

  llvm::InitializeAllAsmPrinters();

  auto triple = llvm::Triple{toolchain_->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    std::cerr << std::format("cxx: cannot find target for triple '{}': {}\n",
                             triple.getTriple(), error);
    fail();
    return;
  }

  llvm::TargetOptions opt;

  auto RM = std::optional<llvm::Reloc::Model>();

  auto targetMachine =
      target->createTargetMachine(llvm::Triple{triple}, "generic", "", opt, RM);

  if (!targetMachine) {
    std::cerr << std::format("cxx: cannot create target machine for '{}': {}\n",
                             triple.getTriple(), error);
    fail();
    return;
  }

  const bool emitAssembly = cli.opt_S && !objectOutput_.has_value();

  auto emit = [&](llvm::raw_pwrite_stream& out) {
    llvm::legacy::PassManager pm;

    llvm::CodeGenFileType fileType = emitAssembly
                                         ? llvm::CodeGenFileType::AssemblyFile
                                         : llvm::CodeGenFileType::ObjectFile;

    if (targetMachine->addPassesToEmitFile(pm, out, nullptr, fileType)) {
      std::cerr << "cxx: target machine cannot emit assembly\n";
      fail();
      return;
    }

    pm.run(*llvmModule_);
    out.flush();
  };

  if (objectOutput_.has_value()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(*objectOutput_, ec);
    if (ec) {
      std::cerr << std::format("cxx: cannot open '{}': {}\n", *objectOutput_,
                               ec.message());
      fail();
      return;
    }
    emit(out);
    return;
  }

  withRawOutputStream(emitAssembly ? ".s" : ".o", emit);
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
