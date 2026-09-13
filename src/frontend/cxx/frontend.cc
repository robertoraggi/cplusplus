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
#include <cxx/freeze_audit.h>
#include <cxx/lexer.h>
#include <cxx/memory_layout.h>
#include <cxx/pch.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>
#include <cxx/symbols.h>
#include <cxx/time_trace.h>
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#ifdef CXX_WITH_MLIR
#include <cxx/codegen/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/mlir/mlir_emitter.h>
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
  std::unique_ptr<llvm::TargetMachine> targetMachine_;
#endif
  bool shouldExit_ = false;
  std::optional<PreprocessorSnapshot> preprocessorSnapshot_;
  bool failed_ = false;
  bool resumedFromPrecompiledHeader_ = false;

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
  void reportFreezeErrors(const FreezeAudit& audit);
  void capturePreprocessorSnapshot();
  void loadPrecompiledHeader();
  void serializePrecompiledHeader();
  [[nodiscard]] auto compatibilityKeys() const -> PrecompiledHeaderKeys;
  [[nodiscard]] auto optionDigest() const -> std::string;
  [[nodiscard]] auto targetKey() const -> std::string;
  [[nodiscard]] auto languageKey() const -> std::string;
  void dumpAst();
  void printAstIfNeeded();
  void generateIR();
  void emitCxxIR();
  void lowerIR();
  void emitMLIR();
  void emitLLVMIR();
  void emitCode();

#ifdef CXX_WITH_MLIR
  [[nodiscard]] auto llvmOptimizationLevel() const -> llvm::OptimizationLevel;
  [[nodiscard]] auto targetMachine() -> llvm::TargetMachine*;
#endif

  [[nodiscard]] auto debugCompilationDirectory() const -> std::string;
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
  if (priv->cli.getSingle("-ftime-trace")) priv->unit_->enableTimeTrace();
  priv->prepare();
  priv->preparePreprocessor();

  for (const auto& action : priv->actions_) {
    if (priv->shouldExit_) break;
    action();
  }

  if (auto path = priv->cli.getSingle("-ftime-trace")) {
    std::ofstream out{*path};
    priv->unit_->timeTrace()->write(out);
    out.flush();
    if (!out) {
      std::cerr << std::format("cannot write time trace '{}'\n", *path);
      priv->failed_ = true;
    }
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
  actions_.emplace_back([this]() { loadPrecompiledHeader(); });
  actions_.emplace_back([this]() { preprocess(); });
  actions_.emplace_back([this]() { writeDepFile(); });
  actions_.emplace_back([this]() { printPreprocessedText(); });
  actions_.emplace_back([this]() { dumpMacros(std::cout); });
  actions_.emplace_back([this]() { dumpTokens(std::cout); });
  actions_.emplace_back([this]() { capturePreprocessorSnapshot(); });
  actions_.emplace_back([this]() { unit_->preprocessor()->squeeze(); });
  actions_.emplace_back([this]() { parse(); });
  actions_.emplace_back([this]() { dumpSymbols(std::cout); });
  actions_.emplace_back([this]() { dumpRecordLayouts(std::cout); });
  actions_.emplace_back([this]() { dumpAst(); });
  actions_.emplace_back([this]() { printAstIfNeeded(); });
  actions_.emplace_back([this]() { serializePrecompiledHeader(); });
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
    unit_->preprocessor()->getPreprocessedText(
        unit_->tokens(), unit_->packAlignmentChanges(), out);
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
  TimeTrace::Scope trace{unit_->timeTrace(), "Preprocess"};

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

  if (cli.opt_verify) {
    diagnosticsClient_->setVerify(true);
    preprocessor->setCommentHandler(diagnosticsClient_.get());
  }

  std::string error;
  toolchain_ =
      createToolchain(cli, preprocessor, languageOf(cli, fileName_), error);
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
  TimeTrace::Scope trace{unit_->timeTrace(), "Parse"};
  if (auto errorLimitStr = cli.getSingle("-ferror-limit")) {
    int limit = std::atoi(errorLimitStr->c_str());
    if (limit > 0) diagnosticsClient_->setErrorLimit(limit);
  }

  bool checkTypes = !cli.opt_fno_check;
  if (cli.opt_fvalidate_ast) checkTypes = true;
  if (cli.opt_emit_pch) checkTypes = true;
  if (needsIR()) checkTypes = true;
  if (unit_->language() == LanguageKind::kC) checkTypes = true;

  ParserConfiguration config{
      .checkTypes = checkTypes,
      .validateAst = cli.opt_fvalidate_ast,
      .allowUnprototypedFunctions = cli.opt_fno_strict_prototypes,
      .stopParsingPredicate = [this]() -> bool {
        return diagnosticsClient_->errorLimitReached();
      },
  };

  if (resumedFromPrecompiledHeader_) {
    // The end-of-translation-unit pass runs once, over the union of the
    // prefix's restored queues and the suffix's own (9.5).
    unit_->resume(std::move(config));
  } else {
    unit_->parse(std::move(config));
  }

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
                         to_string(baseClassSymbol->type()));

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
                           to_string(classSymbol->type()));

        dumpClassMembers(classSymbol, layout, 1, 0);

        for (auto vbase : layout->virtualBases()) {
          auto baseInfo = layout->getBaseInfo(vbase);
          if (!baseInfo) continue;
          out << std::format("{:>9} | {}{} {} (virtual base)\n",
                             baseInfo->offset, std::string(2, ' '),
                             classKeyword(vbase), to_string(vbase->type()));
          if (auto vbaseLayout = vbase->layout()) {
            dumpClassMembers(vbase, vbaseLayout, 2, baseInfo->offset);
          }
        }

        auto traits = unit_->typeTraits();

        out << std::format(
            "{:>9} | [sizeof={}, dsize={}, align={},\n", "", layout->size(),
            traits.data_size(classSymbol->type()), layout->alignment());
        if (layout->virtualBases().empty()) {
          out << std::format("{:>9} |  nvsize={}, nvalign={}]\n", "",
                             traits.non_virtual_size(classSymbol->type()),
                             layout->alignment());
        } else {
          out << std::format("{:>9} |  nvsize={}, nvalign={}]\n", "",
                             traits.non_virtual_size(classSymbol->type()),
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

void Frontend::Private::reportFreezeErrors(const FreezeAudit& audit) {
  for (const auto& error : audit.errors()) {
    std::cerr << std::format("cxx: cannot write a precompiled header: {}\n",
                             error);
  }
  fail();
}

void Frontend::Private::capturePreprocessorSnapshot() {
  if (!cli.opt_emit_pch) return;

  FreezeAudit audit{unit_.get()};

  if (!audit.checkPreprocessingBoundary()) {
    reportFreezeErrors(audit);
    return;
  }

  preprocessorSnapshot_ = unit_->preprocessor()->snapshot();
}

void Frontend::Private::loadPrecompiledHeader() {
  auto pchFile = cli.getSingle("-include-pch");
  if (!pchFile.has_value()) return;

  TimeTrace::Scope trace{unit_->timeTrace(), "Load precompiled header",
                         *pchFile};

  std::ifstream in(*pchFile, std::ios::binary | std::ios::ate);

  if (!in) {
    std::cerr << std::format("cxx: cannot open precompiled header '{}'\n",
                             *pchFile);
    fail();
    return;
  }

  const auto size = in.tellg();
  in.seekg(0);

  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char*>(data.data()), size);

  if (!in) {
    std::cerr << std::format("cxx: cannot read precompiled header '{}'\n",
                             *pchFile);
    fail();
    return;
  }

  PrecompiledHeaderReader reader{unit_.get(), compatibilityKeys()};

  if (!reader(data)) {
    std::cerr << std::format("cxx: {}: {}\n", *pchFile, reader.error());
    fail();
    return;
  }

  resumedFromPrecompiledHeader_ = true;
}

auto Frontend::Private::compatibilityKeys() const -> PrecompiledHeaderKeys {
  return {precompiledHeaderSerializationAbi(), targetKey(), languageKey(),
          optionDigest()};
}

auto Frontend::Private::optionDigest() const -> std::string {
  // Only the options that change the meaning of the prefix belong here; a
  // mismatch is an ordinary cache miss with a reason, never a silent accept.
  return std::format("reflect={} strict-prototypes={}",
                     cli.opt_fno_reflect ? 0 : 1,
                     cli.opt_fno_strict_prototypes ? 0 : 1);
}

auto Frontend::Private::targetKey() const -> std::string {
  if (!toolchain_) return "none";

  auto memoryLayout = toolchain_->memoryLayout();

  return std::format("{}/{}/{}", memoryLayout->triple(), memoryLayout->arch(),
                     memoryLayout->sizeOfLongDouble());
}

auto Frontend::Private::languageKey() const -> std::string {
  auto language = "c++";
  if (unit_->language() == LanguageKind::kC) language = "c";

  auto version = cli.getSingle("-std").value_or("default");

  return std::format("{}/{}", language, version);
}

void Frontend::Private::serializePrecompiledHeader() {
  if (!cli.opt_emit_pch) return;
  if (!preprocessorSnapshot_.has_value()) return;
  if (diagnosticsClient_->hasErrors()) {
    shouldExit_ = true;
    return;
  }

  shouldExit_ = true;

  PrecompiledHeaderWriter writer{unit_.get(), compatibilityKeys()};
  writer.setPreprocessorState(std::move(*preprocessorSnapshot_));

  for (const auto& [fileName, isSystemHeader] :
       unit_->preprocessor()->includedFiles()) {
    writer.addDependency({fileName, {}, isSystemHeader});
  }

  const auto data = writer();

  if (!writer.errors().empty()) {
    for (const auto& error : writer.errors()) {
      std::cerr << std::format("cxx: cannot write a precompiled header: {}\n",
                               error);
    }
    fail();
    return;
  }

  auto outputFile = cli.getSingle("-o").value_or(
      fs::path{fileName_}.filename().replace_extension(".pch").string());

  std::ofstream out(outputFile, std::ios::binary);

  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));

  out.close();

  if (!out) {
    std::cerr << std::format("cxx: cannot write '{}'\n", outputFile);
    fail();
  }
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

  auto emitter = cxx::ir::MlirEmitter{*context_, unit_.get()};
  auto codegen =
      cxx::Codegen{emitter,
                   unit_.get(),
                   {.debugInfo = cli.opt_g,
                    .debugCompilationDirectory = debugCompilationDirectory()}};

  auto ir = codegen(unit_->ast());
  module_ = emitter.module(ir.module);

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

  if (const auto level = llvmOptimizationLevel();
      level != llvm::OptimizationLevel::O0) {
    optimizeLLVMIR(*llvmModule_, targetMachine(), level);
  }

  if (!cli.opt_emit_llvm) return;

  shouldExit_ = true;

  withRawOutputStream(
      ".ll", [&](llvm::raw_ostream& out) { llvmModule_->print(out, nullptr); });

#endif
}

auto Frontend::Private::debugCompilationDirectory() const -> std::string {
  if (auto dir = cli.getSingle("-fdebug-compilation-dir")) return *dir;
  return fs::working_directory().string();
}

#ifdef CXX_WITH_MLIR
auto Frontend::Private::llvmOptimizationLevel() const
    -> llvm::OptimizationLevel {
  switch (cli.optimizationLevel()) {
    case 1:
      return llvm::OptimizationLevel::O1;
    case 2:
      return llvm::OptimizationLevel::O2;
    case 3:
      return llvm::OptimizationLevel::O3;
    default:
      return llvm::OptimizationLevel::O0;
  }
}

auto Frontend::Private::targetMachine() -> llvm::TargetMachine* {
  if (targetMachine_) return targetMachine_.get();

  llvm::InitializeAllAsmPrinters();

  auto triple = llvm::Triple{toolchain_->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    std::cerr << std::format("cxx: cannot find target for triple '{}': {}\n",
                             triple.getTriple(), error);
    return nullptr;
  }

  const auto codeGenOptLevel = [&] {
    switch (cli.optimizationLevel()) {
      case 1:
        return llvm::CodeGenOptLevel::Less;
      case 2:
        return llvm::CodeGenOptLevel::Default;
      case 3:
        return llvm::CodeGenOptLevel::Aggressive;
      default:
        return llvm::CodeGenOptLevel::None;
    }
  }();

  llvm::TargetOptions opt;

  targetMachine_ =
      std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
          llvm::Triple{triple}, "generic", "", opt,
          std::optional<llvm::Reloc::Model>(), std::nullopt, codeGenOptLevel));

  if (!targetMachine_) {
    std::cerr << std::format("cxx: cannot create target machine for '{}': {}\n",
                             triple.getTriple(), error);
  }

  return targetMachine_.get();
}
#endif

void Frontend::Private::emitCode() {
  if (!cli.opt_S && !cli.opt_c && !objectOutput_.has_value()) return;
#ifdef CXX_WITH_MLIR
  if (!llvmModule_) return;

  auto targetMachine = this->targetMachine();

  if (!targetMachine) {
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
