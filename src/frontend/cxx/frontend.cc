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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/cxxir/cxxir_emitter.h>
#include <cxx/freeze_audit.h>
#include <cxx/lexer.h>
#include <cxx/memory_layout.h>
#include <cxx/pch.h>
#include <cxx/preprocessor.h>
#include <cxx/printing.h>
#include <cxx/private/file.h>
#include <cxx/private/path.h>
#include <cxx/private/printing-priv.h>
#include <cxx/private/read_all.h>
#include <cxx/symbols.h>
#include <cxx/time_trace.h>
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <system_error>

#ifdef CXX_WITH_MLIR
#include <cxx/mlir/backend.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/mlir/mlir_emitter.h>
#include <cxx/mlir/mlir_source_contexts.h>
#include <cxx/mlir/mlir_source_region.h>
#include <cxx/mlir/passes.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <format>
#include <iterator>
#include <print>
#include <string>

#include "check_expression_types.h"
#include "dump_tokens.h"
#include "verify_diagnostics_client.h"

namespace cxx {

namespace {

#ifdef CXX_WITH_MLIR

[[nodiscard]] auto backendRunsAlwaysInliner() -> bool {
  return llvmPipelineRunsAlwaysInliner();
}

[[nodiscard]] auto severityOf(mlir::DiagnosticSeverity severity) -> Severity {
  switch (severity) {
    case mlir::DiagnosticSeverity::Error:
      return Severity::Error;
    case mlir::DiagnosticSeverity::Warning:
      return Severity::Warning;
    case mlir::DiagnosticSeverity::Note:
      return Severity::Note;
    case mlir::DiagnosticSeverity::Remark:
      return Severity::Message;
  }
  std::unreachable();
}

[[nodiscard]] auto startsAt(const SourcePosition& position,
                            mlir::FileLineColLoc location) -> bool {
  if (position.line != location.getLine()) return false;
  if (position.column != location.getColumn()) return false;
  return position.fileName == std::string_view{location.getFilename()};
}

[[nodiscard]] auto tokenAt(const TranslationUnit& unit,
                           mlir::FileLineColLoc location) -> SourceLocation {
  for (unsigned index = 0; index < unit.tokenCount(); ++index) {
    auto source = unit.locationOfIndex(index);
    if (startsAt(unit.tokenStartPosition(source), location)) return source;
  }
  return {};
}

#else

[[nodiscard]] auto backendRunsAlwaysInliner() -> bool { return false; }

#endif

}  // namespace

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
  mlir::OwningOpRef<mlir::ModuleOp> parsedModule_;
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
           cli.opt_emit_spirv || cli.opt_emit_wgsl || cli.opt_S || cli.opt_c ||
           objectOutput_.has_value();
  }

  void prepare();
  void preparePreprocessor();
  void configureDiagnostics();
  void preprocess();
  void parse();
  void showSearchPaths(std::format_context::iterator& out);
  void dumpTokens(std::format_context::iterator& out);
  void dumpSymbols(std::format_context::iterator& out);
  void dumpRecordLayouts(std::format_context::iterator& out);
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
  void generateCxxIR(const std::function<void(const ir::CxxIREmitter&,
                                              ir::ModuleRef)>& action);
  [[nodiscard]] auto codegenOptions() const -> Codegen::Options;
  [[nodiscard]] auto targetsDevice() const -> bool;
  void compileIR();

#ifdef CXX_WITH_MLIR
  [[nodiscard]] auto reportConflictingOutputs() -> bool;
  [[nodiscard]] auto selectBackendOutput() const -> BackendOutput;
  [[nodiscard]] auto isKernelArtifact(BackendOutput output) const -> bool;
  [[nodiscard]] static auto outputExtension(BackendOutput output)
      -> std::string;
  [[nodiscard]] auto kernelOutputPath(BackendOutput output) const
      -> std::string;
  [[nodiscard]] auto hasValidKernelOutputPaths(BackendOutput output) -> bool;
  void writeKernelArtifact(BackendOutput output,
                           const BackendArtifact& artifact);
  void writeObjectFile(const std::string& path, const std::string& code);
  [[nodiscard]] auto reportBackendDiagnostic(mlir::Diagnostic& diagnostic)
      -> bool;
#endif

  [[nodiscard]] auto debugCompilationDirectory() const -> std::string;
  void printPreprocessedText();
  void writeDepFile();
  void dumpMacros(std::format_context::iterator& out);

  [[nodiscard]] auto readAll(const std::string& fileName)
      -> std::optional<std::string>;

  void withOutput(
      std::FILE* file,
      const std::function<void(std::format_context::iterator&)>& action,
      std::string_view name = "<stdout>");

  void withOutputFile(
      const std::optional<std::string>& extension,
      const std::function<void(std::format_context::iterator&)>& action);

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
    detail::File file(std::fopen(path->c_str(), "w"));
    bool written = false;
    if (file) {
#if __cpp_exceptions
      try {
#endif
        std::print(file.get(), "{}", *priv->unit_->timeTrace());
        written = std::fflush(file.get()) == 0;
#if __cpp_exceptions
      } catch (const std::system_error&) {
      }
#endif
    }
    if (!written) {
      std::print(stderr, "cannot write time trace '{}'\n", *path);
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
  configureDiagnostics();
  unit_ = std::make_unique<TranslationUnit>(diagnosticsClient_.get());

  actions_.emplace_back([this]() {
    withOutput(stderr, [&](auto& out) { showSearchPaths(out); });
  });
  actions_.emplace_back([this]() { loadPrecompiledHeader(); });
  actions_.emplace_back([this]() { preprocess(); });
  actions_.emplace_back([this]() { writeDepFile(); });
  actions_.emplace_back([this]() { printPreprocessedText(); });
  actions_.emplace_back(
      [this]() { withOutput(stdout, [&](auto& out) { dumpMacros(out); }); });
  actions_.emplace_back(
      [this]() { withOutput(stdout, [&](auto& out) { dumpTokens(out); }); });
  actions_.emplace_back([this]() { capturePreprocessorSnapshot(); });
  actions_.emplace_back([this]() { unit_->preprocessor()->squeeze(); });
  actions_.emplace_back([this]() { parse(); });
  actions_.emplace_back(
      [this]() { withOutput(stdout, [&](auto& out) { dumpSymbols(out); }); });
  actions_.emplace_back([this]() {
    withOutput(stdout, [&](auto& out) { dumpRecordLayouts(out); });
  });
  actions_.emplace_back([this]() { dumpAst(); });
  actions_.emplace_back([this]() { printAstIfNeeded(); });
  actions_.emplace_back([this]() { serializePrecompiledHeader(); });
  actions_.emplace_back([this]() { exitIfErrors(); });
  actions_.emplace_back(
      [this]() { toolchain_->applyEntryPointAbi(unit_.get()); });
  actions_.emplace_back([this]() { generateIR(); });
  actions_.emplace_back([this]() { exitIfErrors(); });
  actions_.emplace_back([this]() { compileIR(); });
}

Frontend::Private::~Private() {}

void Frontend::Private::withOutput(
    std::FILE* file,
    const std::function<void(std::format_context::iterator&)>& action,
    std::string_view name) {
#if __cpp_exceptions
  try {
#endif
    std::print(file, "{}", detail::FormattingAction{action});
    if (std::fflush(file) == 0) return;
#if __cpp_exceptions
  } catch (const std::system_error&) {
  }
#endif
  std::print(stderr, "cxx: failed to write output file '{}'\n", name);
  fail();
}

void Frontend::Private::withOutputFile(
    const std::optional<std::string>& extension,
    const std::function<void(std::format_context::iterator&)>& action) {
  auto explicitOutput = cli.getSingle("-o");
  if (explicitOutput == "-" ||
      (!explicitOutput && (!extension || fileName_ == "-"))) {
    withOutput(stdout, action);
    return;
  }
  auto outputFile = explicitOutput.value_or("");
  if (!explicitOutput) {
    auto inputFile = fs::path{fileName_}.filename();
    outputFile = inputFile.replace_extension(*extension).string();
  }
  detail::File output(std::fopen(outputFile.c_str(), "w"));
  if (!output) {
    std::print(stderr, "cxx: cannot open output file '{}'\n", outputFile);
    fail();
    return;
  }
  withOutput(output.get(), action, outputFile);
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
  if (error_code) {
    std::print(stderr, "{}{}{}{}{}", "cxx: cannot open output file '",
               outputFile, "': ", error_code.message(), '\n');
    fail();
    return;
  }
  action(output);
  output.flush();
  if (output.has_error()) {
    std::print(stderr, "{}{}{}", "cxx: failed to write output file '",
               outputFile, "'\n");
    output.clear_error();
    fail();
  }
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

  withOutputFile(std::nullopt, [&](std::format_context::iterator& out) {
    out = std::format_to(out, "{}",
                         unit_->preprocessor()->preprocessedText(
                             unit_->tokens(), unit_->packAlignmentChanges()));
  });
}

static void collectExistingDirectories(std::format_context::iterator& out,
                                       const std::vector<std::string>& paths,
                                       std::vector<std::string>& existing) {
  for (const auto& path : paths) {
    if (fs::is_directory(path)) {
      existing.push_back(path);
      continue;
    }
    out = std::format_to(out, "ignoring nonexistent directory \"{}\"\n", path);
  }
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

static void formatDepFile(std::format_context::iterator& out,
                          const std::string& target,
                          const std::vector<std::string>& deps, bool phony) {
  out = std::format_to(out, "{}{}", target, ':');
  for (const auto& dep : deps)
    out = std::format_to(out, "{}{}", " \\\n  ", dep);
  out = std::format_to(out, "{}", '\n');
  if (!phony) return;
  for (const auto& dep : deps)
    out = std::format_to(out, "{}{}{}", '\n', dep, ":\n");
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
    withOutput(stdout, [&](auto& out) {
      formatDepFile(out, target, deps, cli.opt_MP);
    });
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

  detail::File depOut(std::fopen(depFileName.c_str(), "w"));
  if (!depOut) {
    std::print(stderr, "cxx: cannot open output file '{}'\n", depFileName);
    fail();
    return;
  }
  withOutput(
      depOut.get(),
      [&](auto& out) { formatDepFile(out, target, deps, cli.opt_MP); },
      depFileName);
}

void Frontend::Private::preprocess() {
  TimeTrace::Scope trace{unit_->timeTrace(), "Preprocess"};

  auto source = readAll(fileName_);

  if (!source.has_value()) {
    std::print(stderr, "cxx: No such file or directory: '{}'\n", fileName_);
    fail();
    return;
  }

  unit_->setSource(std::move(*source), fileName_);
}

void Frontend::Private::dumpMacros(std::format_context::iterator& out) {
  if (!cli.opt_E && !cli.opt_dM) return;

  out = std::format_to(out, "{}", unit_->preprocessor()->macros());

  shouldExit_ = true;
}

void Frontend::Private::configureDiagnostics() {
  if (auto errorLimitStr = cli.getSingle("-ferror-limit")) {
    int limit = std::atoi(errorLimitStr->c_str());
    if (limit > 0) diagnosticsClient_->setErrorLimit(limit);
  }
  for (const auto& warning : cli.get("-W")) {
    if (warning == "system-headers")
      diagnosticsClient_->setSystemHeaderWarnings(true);
    else if (warning == "no-system-headers")
      diagnosticsClient_->setSystemHeaderWarnings(false);
  }
}

void Frontend::Private::prepare() {
  if (auto kernel = cli.getSingle("--kernel"); kernel && kernel->empty()) {
    std::print(stderr, "{}",
               "cxx: --kernel requires a nonempty entry point name\n");
    fail();
    return;
  }
#ifndef CXX_WITH_SPIRV
  if (cli.opt_emit_spirv || cli.opt_emit_wgsl) {
    std::print(stderr, "{}", "cxx: this build has no device kernel support\n");
    fail();
    return;
  }
#endif
  auto preprocessor = unit_->preprocessor();

  if (cli.opt_verify) {
    diagnosticsClient_->setVerify(true);
    preprocessor->setCommentHandler(diagnosticsClient_.get());
  }

  std::string error;
  toolchain_ =
      createToolchain(cli, preprocessor, languageOf(cli, fileName_), error);
  if (!error.empty()) {
    std::print(stderr, "{}{}", error, '\n');
    fail();
    return;
  }
  if (!toolchain_) {
    std::print(stderr, "{}{}", describeUnsupportedTarget(cli), '\n');
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
          std::print("{} {}\n", fill, header);
        });
  }
}

void Frontend::Private::parse() {
  TimeTrace::Scope trace{unit_->timeTrace(), "Parse"};
  bool checkTypes = !cli.opt_fno_check;
  if (cli.opt_emit_pch) checkTypes = true;
  if (needsIR()) checkTypes = true;
  if (unit_->language() == LanguageKind::kC) checkTypes = true;

  ParserConfiguration config{
      .checkTypes = checkTypes,
      .allowUnprototypedFunctions = cli.opt_fno_strict_prototypes,
      .exceptionsEnabled = toolchain_->exceptionsEnabled(),
      .rttiEnabled = toolchain_->rttiEnabled(),
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

void Frontend::Private::dumpTokens(std::format_context::iterator& out) {
  if (!cli.opt_dump_tokens) return;

  out = std::format_to(out, "{}", DumpTokens{*unit_});

  shouldExit_ = true;
}

void Frontend::Private::dumpSymbols(std::format_context::iterator& out) {
  if (!cli.opt_dump_symbols) return;
  auto globalScope = unit_->globalScope();
  auto globalNamespace = globalScope;
  out =
      std::format_to(out, "{}", cxx::dumpSymbols(globalNamespace, unit_.get()));
}

void Frontend::Private::dumpRecordLayouts(std::format_context::iterator& out) {
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
      out = std::format_to(out, "{:>9} |{}{} {} (base)\n", absOffset, pad,
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
            out = std::format_to(out, "{:>9} |{}{} (anonymous) \n", absOffset,
                                 pad, classKeyword(nestedClass));

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
        out = std::format_to(out, "{:>9} |{}{} {}\n", offsetStr, pad, typeStr,
                             nameStr);
      } else if (field->isBitField() && fieldInfo->bitWidth == 0) {
        auto absByte = absOffset + fieldInfo->bitOffset / 8;
        auto startBit = static_cast<int>(fieldInfo->bitOffset % 8);
        auto offsetStr = std::format("{}:{}-", absByte, startBit);
        out = std::format_to(out, "{:>9} |{}{}\n", offsetStr, pad, typeStr);
      } else {
        out = std::format_to(out, "{:>9} |{}{} {}\n", absOffset, pad, typeStr,
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

        out = std::format_to(out, "\n*** Dumping AST Record Layout\n");
        out =
            std::format_to(out, "{:>9} | {} {}\n", 0, classKeyword(classSymbol),
                           to_string(classSymbol->type()));

        dumpClassMembers(classSymbol, layout, 1, 0);

        for (auto vbase : layout->virtualBases()) {
          auto baseInfo = layout->getVirtualBaseInfo(vbase);
          if (!baseInfo) continue;
          out = std::format_to(out, "{:>9} | {}{} {} (virtual base)\n",
                               baseInfo->offset, std::string(2, ' '),
                               classKeyword(vbase), to_string(vbase->type()));
          if (auto vbaseLayout = vbase->layout()) {
            dumpClassMembers(vbase, vbaseLayout, 2, baseInfo->offset);
          }
        }

        auto traits = unit_->typeTraits();

        out = std::format_to(out, "{:>9} | [sizeof={}, dsize={}, align={},\n",
                             "", layout->size(),
                             traits.data_size(classSymbol->type()),
                             layout->alignment());
        if (layout->virtualBases().empty()) {
          out = std::format_to(out, "{:>9} |  nvsize={}, nvalign={}]\n", "",
                               traits.non_virtual_size(classSymbol->type()),
                               layout->alignment());
        } else {
          out = std::format_to(out, "{:>9} |  nvsize={}, nvalign={}]\n", "",
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
  withOutput(stdout, [&](auto& out) {
    out = std::format_to(out, "{}", cxx::dumpAST(unit_.get(), unit_->ast()));
  });
}

void Frontend::Private::printAstIfNeeded() {
  if (!cli.opt_ast_print) return;
  std::print("{}", formatAST(unit_.get(), unit_->ast()));
}

void Frontend::Private::reportFreezeErrors(const FreezeAudit& audit) {
  for (const auto& error : audit.errors()) {
    std::print(stderr, "cxx: cannot write a precompiled header: {}\n", error);
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

  detail::File in(std::fopen(pchFile->c_str(), "rb"));

  if (!in) {
    std::print(stderr, "cxx: cannot open precompiled header '{}'\n", *pchFile);
    fail();
    return;
  }

  auto data = detail::readAll(in.get());
  if (!data) {
    std::print(stderr, "cxx: cannot read precompiled header '{}'\n", *pchFile);
    fail();
    return;
  }

  PrecompiledHeaderReader reader{unit_.get(), compatibilityKeys()};

  if (!reader(std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(data->data()), data->size()))) {
    std::print(stderr, "cxx: {}: {}\n", *pchFile, reader.error());
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

  return std::format("{}/{}/{}/{}", memoryLayout->triple(),
                     memoryLayout->arch(), memoryLayout->sizeOfLongDouble(),
                     to_string(memoryLayout->deviceProfile()));
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
      std::print(stderr, "cxx: cannot write a precompiled header: {}\n", error);
    }
    fail();
    return;
  }

  auto outputFile = cli.getSingle("-o").value_or(
      fs::path{fileName_}.filename().replace_extension(".pch").string());

  detail::File out(std::fopen(outputFile.c_str(), "wb"));
  bool written = false;
  if (out)
    written =
        std::fwrite(data.data(), 1, data.size(), out.get()) == data.size() &&
        std::fflush(out.get()) == 0;
  if (!written) {
    std::print(stderr, "cxx: cannot write '{}'\n", outputFile);
    fail();
  }
}

void Frontend::Private::showSearchPaths(std::format_context::iterator& out) {
  if (!cli.opt_v) return;

  auto preprocessor = unit_->preprocessor();

  std::vector<std::string> quotePaths;
  std::vector<std::string> anglePaths;

  collectExistingDirectories(out, preprocessor->quoteIncludePaths(),
                             quotePaths);
  collectExistingDirectories(out, preprocessor->userIncludePaths(), anglePaths);
  collectExistingDirectories(out, preprocessor->systemIncludePaths(),
                             anglePaths);

  std::vector<std::string> frameworkPaths;
  collectExistingDirectories(out, preprocessor->systemFrameworkPaths(),
                             frameworkPaths);

  out = std::format_to(out, "#include \"...\" search starts here:\n");
  for (const auto& path : quotePaths) out = std::format_to(out, " {}\n", path);

  out = std::format_to(out, "#include <...> search starts here:\n");
  for (const auto& path : anglePaths) out = std::format_to(out, " {}\n", path);
  for (const auto& path : frameworkPaths)
    out = std::format_to(out, " {} (framework directory)\n", path);

  out = std::format_to(out, "End of search list.\n");
}

void Frontend::Private::generateCxxIR(
    const std::function<void(const ir::CxxIREmitter&, ir::ModuleRef)>& action) {
  TimeTrace::Scope trace(unit_->timeTrace(), "Generate CXXIR");
  ir::CxxIREmitter emitter{unit_.get()};
  Codegen codegen{emitter, unit_.get(), codegenOptions()};
  auto result = codegen(unit_->ast());
  if (diagnosticsClient_->hasErrors()) {
    fail();
    return;
  }
  if (!result.module) return;
  action(emitter, result.module);
}

auto Frontend::Private::codegenOptions() const -> Codegen::Options {
  return {.debugInfo = cli.opt_g,
          .debugCompilationDirectory = debugCompilationDirectory(),
          .runsAlwaysInliner = backendRunsAlwaysInliner()};
}

void Frontend::Private::generateIR() {
  if (cli.opt_fsyntax_only) return;
  if (!needsIR()) return;

  auto useExperimentalCxxir = cli.experimentalCxxIREnabled();

#ifndef CXX_WITH_MLIR
  useExperimentalCxxir = true;
#endif

  if (useExperimentalCxxir && cli.opt_emit_cxx_ir &&
      cli.getSingle("-mlir-pass-pipeline").value_or("").empty()) {
    withOutputFile("mlir", [&](std::format_context::iterator& out) {
      generateCxxIR([&](const ir::CxxIREmitter& emitter, ir::ModuleRef module) {
        out = std::format_to(out, "{}", emitter.module(module, cli.opt_g));
      });
    });
    shouldExit_ = true;
    return;
  }

#ifdef CXX_WITH_MLIR
  TimeTrace::Scope trace(unit_->timeTrace(), "Generate IR");
  context_ = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);
  context_->loadDialect<mlir::cxx::CxxDialect>();

  if (cli.experimentalCxxIREnabled()) {
    std::optional<std::string> text;
    generateCxxIR([&](const ir::CxxIREmitter& emitter, ir::ModuleRef module) {
      text = std::format("{}", emitter.module(module, true));
    });
    if (!text) return;
    TimeTrace::Scope trace(unit_->timeTrace(), "Read CXXIR");
    parsedModule_ = parseCxxIR(*context_, *text, cli.irVerificationEnabled());
    if (!parsedModule_) {
      std::print(stderr, "{}", "cxx: failed to read experimental CXX IR\n");
      fail();
      return;
    }
    module_ = parsedModule_.get();
    return;
  }

  auto emitter = cxx::ir::MlirEmitter{*context_, unit_.get()};
  auto mode = std::getenv("CXX_SOURCE_CONTEXT_MODE");
  ir::SourceContextOptions options;
  options.timeTrace = unit_->timeTrace();
  if (mode && std::string_view(mode) == "scf") {
    options.mode = ir::SourceContextOptions::Mode::Structured;
  } else if (mode && std::string_view(mode) == "if") {
    options.mode = ir::SourceContextOptions::Mode::StructuredIf;
  } else if (mode) {
    llvm::errs() << "cxx: invalid CXX_SOURCE_CONTEXT_MODE: " << mode << "\n";
    fail();
    return;
  }
  bool enabled = mode;
  if (cli.count("-fexperimental-scf") || cli.count("-fno-experimental-scf")) {
    enabled = cli.experimentalSCFEnabled();
    options.mode = ir::SourceContextOptions::Mode::Structured;
  }
  auto observer = enabled ? emitter.enableSourceContexts(options) : nullptr;
  auto codegenOptions = this->codegenOptions();
  codegenOptions.sourceObserver = observer;
  auto codegen = cxx::Codegen{emitter, unit_.get(), std::move(codegenOptions)};

  auto ir = codegen(unit_->ast());
  module_ = emitter.module(ir.module);
  if (observer && options.mode == ir::SourceContextOptions::Mode::Structured) {
    TimeTrace::Scope trace(unit_->timeTrace(), "SCF finish");
    if (failed(ir::finalizeSourceRegions(module_, unit_->timeTrace()))) {
      fail();
      module_ = nullptr;
    }
  }
#endif
}

auto Frontend::Private::targetsDevice() const -> bool {
  return toolchain_ &&
         toolchain_->memoryLayout()->deviceProfile() != DeviceProfile::kNone;
}

auto Frontend::Private::debugCompilationDirectory() const -> std::string {
  if (auto dir = cli.getSingle("-fdebug-compilation-dir")) return *dir;
  return fs::working_directory().string();
}

#ifndef CXX_WITH_MLIR

void Frontend::Private::compileIR() {
  if (!needsIR()) return;
  std::print(stderr, "{}", "cxx: code generation requires a build with MLIR\n");
  fail();
}

#else

void Frontend::Private::compileIR() {
  if (!needsIR()) return;
  if (!module_) return;
  if (reportConflictingOutputs()) return;

  const auto output = selectBackendOutput();
  if (isKernelArtifact(output) && !hasValidKernelOutputPaths(output)) return;

  const auto passPipeline = cli.getSingle("-mlir-pass-pipeline").value_or("");
  const auto kernel = cli.getSingle("--kernel").value_or("");

  registerPipelinePasses();

  auto artifact =
      compileModule(module_,
                    {.output = output,
                     .optimizationLevel = cli.optimizationLevel(),
                     .debugInfo = cli.opt_g,
                     .verifyIR = cli.irVerificationEnabled(),
                     .passPipeline = passPipeline,
                     .kernel = kernel,
                     .timeTrace = unit_->timeTrace(),
                     .diagnosticSink =
                         [this](mlir::Diagnostic& diagnostic) {
                           return reportBackendDiagnostic(diagnostic);
                         }},
                    llvm::errs());

  if (!artifact) {
    std::print(stderr, "cxx: {}\n", artifact.error());
    fail();
    return;
  }

  if (isKernelArtifact(output)) {
    writeKernelArtifact(output, *artifact);
    return;
  }

  if (output == BackendOutput::kObject && objectOutput_) {
    writeObjectFile(*objectOutput_, artifact->code);
    return;
  }

  withRawOutputStream(
      outputExtension(output),
      [&](llvm::raw_pwrite_stream& out) { out << artifact->code; });
}

auto Frontend::Private::reportConflictingOutputs() -> bool {
  std::vector<std::string_view> kernelOutputs;
  if (cli.opt_emit_spirv) kernelOutputs.push_back("-emit-spirv");
  if (cli.opt_emit_wgsl) kernelOutputs.push_back("-emit-wgsl");
  if (cli.opt_c && targetsDevice()) kernelOutputs.push_back("-c");
  if (kernelOutputs.size() > 1) {
    std::print(stderr, "cxx: {} and {} are mutually exclusive\n",
               kernelOutputs[0], kernelOutputs[1]);
    fail();
    return true;
  }
  if (!kernelOutputs.empty()) return false;
  if (!targetsDevice() || !cli.opt_emit_mlir || !cli.opt_S) return false;
  std::print(stderr, "{}", "cxx: -emit-mlir and -S are mutually exclusive\n");
  fail();
  return true;
}

auto Frontend::Private::selectBackendOutput() const -> BackendOutput {
  const bool device = targetsDevice();
  if (cli.opt_emit_spirv) return BackendOutput::kSPIRV;
  if (cli.opt_emit_wgsl) return BackendOutput::kWGSL;
  if (device && cli.opt_c) return BackendOutput::kObject;
  if (device && cli.opt_emit_mlir) return BackendOutput::kMLIR;
  if (device && cli.opt_S) return BackendOutput::kAssembly;
  if (cli.opt_emit_cxx_ir) return BackendOutput::kCxxIR;
  if (cli.opt_emit_mlir) return BackendOutput::kMLIR;
  if (cli.opt_emit_llvm) return BackendOutput::kLLVMIR;
  if (cli.opt_S && !objectOutput_) return BackendOutput::kAssembly;
  return BackendOutput::kObject;
}

auto Frontend::Private::isKernelArtifact(BackendOutput output) const -> bool {
  if (output == BackendOutput::kSPIRV || output == BackendOutput::kWGSL)
    return true;
  return output == BackendOutput::kObject && targetsDevice();
}

auto Frontend::Private::outputExtension(BackendOutput output) -> std::string {
  switch (output) {
    case BackendOutput::kCxxIR:
    case BackendOutput::kMLIR:
      return "mlir";
    case BackendOutput::kLLVMIR:
      return "ll";
    case BackendOutput::kAssembly:
      return "s";
    case BackendOutput::kObject:
      return "o";
    case BackendOutput::kSPIRV:
      return "spv";
    case BackendOutput::kWGSL:
      return "wgsl";
  }
  std::unreachable();
}

auto Frontend::Private::kernelOutputPath(BackendOutput output) const
    -> std::string {
  return cli.getSingle("-o").value_or(
      fs::path(fileName_)
          .filename()
          .replace_extension(outputExtension(output))
          .string());
}

auto Frontend::Private::hasValidKernelOutputPaths(BackendOutput output)
    -> bool {
  auto manifestPath = cli.getSingle("--resource-manifest");
  if (!manifestPath) return true;
  if (manifestPath->empty() || *manifestPath == "-") {
    std::print(stderr, "{}", "cxx: --resource-manifest requires a file path\n");
    fail();
    return false;
  }
  if (fs::file_identity(kernelOutputPath(output)) !=
      fs::file_identity(*manifestPath))
    return true;
  std::print(stderr, "{}",
             "cxx: kernel artifact and resource manifest need distinct "
             "paths\n");
  fail();
  return false;
}

void Frontend::Private::writeKernelArtifact(BackendOutput output,
                                            const BackendArtifact& artifact) {
  const auto outputPath = kernelOutputPath(output);
  std::error_code error;
  llvm::ToolOutputFile file(outputPath, error,
                            output == BackendOutput::kWGSL
                                ? llvm::sys::fs::OF_Text
                                : llvm::sys::fs::OF_None);
  if (error) {
    std::print(stderr, "cxx: {}: {}\n", error.message(), outputPath);
    fail();
    return;
  }
  std::optional<llvm::ToolOutputFile> manifest;
  if (auto manifestPath = cli.getSingle("--resource-manifest")) {
    manifest.emplace(*manifestPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      std::print(stderr, "cxx: {}: {}\n", error.message(), *manifestPath);
      fail();
      return;
    }
    manifest->os() << artifact.manifest;
    manifest->os().flush();
  }
  file.os() << artifact.code;
  file.os().flush();
  if (file.os().has_error() || (manifest && manifest->os().has_error())) {
    std::print(stderr, "{}", "cxx: cannot write kernel outputs\n");
    fail();
    return;
  }
  file.keep();
  if (manifest) manifest->keep();
}

void Frontend::Private::writeObjectFile(const std::string& path,
                                        const std::string& code) {
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error) {
    std::print(stderr, "cxx: cannot open '{}': {}\n", path, error.message());
    fail();
    return;
  }
  out << code;
  out.flush();
  if (!out.has_error()) return;
  std::print(stderr, "cxx: failed to write output file '{}'\n", path);
  out.clear_error();
  fail();
}

auto Frontend::Private::reportBackendDiagnostic(mlir::Diagnostic& diagnostic)
    -> bool {
  auto location =
      diagnostic.getLocation()->findInstanceOf<mlir::FileLineColLoc>();
  if (!location) return false;
  auto source = tokenAt(*unit_, location);
  if (!source) return false;
  unit_->report(unit_->diagnosticsClient(), source,
                severityOf(diagnostic.getSeverity()), diagnostic.str());
  return true;
}

#endif

auto Frontend::Private::readAll(const std::string& fileName)
    -> std::optional<std::string> {
  if (fileName == "-" || fileName.empty()) return detail::readAll(stdin);
  detail::File file(std::fopen(fileName.c_str(), "r"));
  if (!file) return std::nullopt;
  return detail::readAll(file.get());
}
}  // namespace cxx
