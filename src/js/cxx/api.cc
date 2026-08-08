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

#include <cxx/ast.h>
#include <cxx/ast_slot.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>
#include <cxx/wasm32_wasi_toolchain.h>

// emscripten
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <format>
#include <optional>
#include <sstream>

#ifdef CXX_WITH_MLIR
// cxx
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>

// mlir
#include <mlir/IR/MLIRContext.h>

// llvm
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#endif

using namespace emscripten;

namespace {

cxx::ASTSlot getSlot;

EMSCRIPTEN_DECLARE_VAL_TYPE(UnitOptions);
EMSCRIPTEN_DECLARE_VAL_TYPE(DiagnosticList);

auto cplusplusMacroValue(std::string_view standard) -> std::string_view {
  if (standard == "c++14") return "201402L";
  if (standard == "c++17") return "201703L";
  if (standard == "c++20") return "202002L";
  if (standard == "c++23") return "202302L";
  if (standard == "c++26") return "202400L";
  return {};
}

auto severityName(cxx::Severity severity) -> std::string_view {
  switch (severity) {
    case cxx::Severity::Message:
      return "message";
    case cxx::Severity::Note:
      return "note";
    case cxx::Severity::Warning:
      return "warning";
    case cxx::Severity::Error:
      return "error";
    case cxx::Severity::Fatal:
      return "fatal";
  }
}

struct DiagnosticsClient final : cxx::DiagnosticsClient {
  val messages = val::array();
  val currentDiagnostic = val::undefined();
  bool hasErrors = false;

  void report(const cxx::Diagnostic& diag) override {
    if (diag.severity() == cxx::Severity::Error ||
        diag.severity() == cxx::Severity::Fatal) {
      hasErrors = true;
    }

    const auto start = preprocessor()->tokenStartPosition(diag.token());
    const auto end = preprocessor()->tokenEndPosition(diag.token());

    val d = val::object();
    d.set("fileName", val(std::string(start.fileName)));
    d.set("startLine", val(start.line));
    d.set("startColumn", val(start.column));
    d.set("endLine", val(end.line));
    d.set("endColumn", val(end.column));
    d.set("message", val(diag.message()));

    if (diag.severity() == cxx::Severity::Note &&
        !currentDiagnostic.isUndefined()) {
      currentDiagnostic["notes"].call<void>("push", d);
      return;
    }

    d.set("severity", val(std::string(severityName(diag.severity()))));
    d.set("notes", val::array());
    messages.call<void>("push", d);

    if (diag.severity() == cxx::Severity::Warning ||
        diag.severity() == cxx::Severity::Error ||
        diag.severity() == cxx::Severity::Fatal) {
      currentDiagnostic = d;
    } else {
      currentDiagnostic = val::undefined();
    }
  }
};

struct WrappedUnit {
  std::unique_ptr<DiagnosticsClient> diagnosticsClient;
  std::unique_ptr<cxx::TranslationUnit> unit;
  std::unique_ptr<cxx::Wasm32WasiToolchain> toolchain;
  UnitOptions api;
  bool debugInfo = true;

  WrappedUnit(std::string source, std::string filename, UnitOptions api)
      : api(api) {
    diagnosticsClient = std::make_unique<DiagnosticsClient>();

    unit = std::make_unique<cxx::TranslationUnit>(diagnosticsClient.get());

    if (auto preprocessor = unit->preprocessor()) {
      toolchain = std::make_unique<cxx::Wasm32WasiToolchain>(preprocessor);

      if (!api.isUndefined()) {
        if (val appdir = api["appdir"]; appdir.isString()) {
          toolchain->setAppdir(appdir.as<std::string>());
        }

        if (val sysroot = api["sysroot"]; sysroot.isString()) {
          toolchain->setSysroot(sysroot.as<std::string>());
        }

        if (val value = api["debugInfo"]; value.isTrue() || value.isFalse()) {
          debugInfo = value.as<bool>();
        }
      }

      toolchain->initMemoryLayout();
      toolchain->addSystemCppIncludePaths();
      toolchain->addSystemIncludePaths();
      toolchain->addPredefinedMacros();

      if (!api.isUndefined()) {
        if (val standard = api["std"]; standard.isString()) {
          auto value = cplusplusMacroValue(standard.as<std::string>());
          if (!value.empty()) {
            preprocessor->undefMacro("__cplusplus");
            preprocessor->defineMacro("__cplusplus", std::string(value));
          }
        }
      }

      addIncludePaths("quoteIncludePaths", [&](std::string path) {
        preprocessor->addQuoteIncludePath(std::move(path));
      });

      addIncludePaths("includePaths", [&](std::string path) {
        preprocessor->addUserIncludePath(std::move(path));
      });

      addIncludePaths("systemIncludePaths", [&](std::string path) {
        preprocessor->addSystemIncludePath(std::move(path));
      });

      for (const auto& macro : stringArray("undefines")) {
        preprocessor->undefMacro(macro);
      }

      for (const auto& macro : stringArray("defines")) {
        auto sep = macro.find_first_of('=');
        if (sep == std::string::npos) {
          preprocessor->defineMacro(macro, "1");
        } else {
          preprocessor->defineMacro(macro.substr(0, sep),
                                    macro.substr(sep + 1));
        }
      }

      preprocessor->setCanResolveFiles(true);
    }

    unit->beginPreprocessing(std::move(source), std::move(filename));
  }

  auto getUnitHandle() const -> std::intptr_t {
    return (std::intptr_t)unit.get();
  }

  auto getHandle() const -> std::intptr_t { return (std::intptr_t)unit->ast(); }

  auto getDiagnostics() const -> DiagnosticList {
    return DiagnosticList(diagnosticsClient->messages);
  }

  auto stringArray(const char* name) const -> std::vector<std::string> {
    val value = api[name];
    if (!value.isArray()) return {};
    return vecFromJSArray<std::string>(value);
  }

  template <typename F>
  void addIncludePaths(const char* name, F&& add) {
    for (auto& path : stringArray(name)) add(std::move(path));
  }

  auto parse() -> val {
    val exists = val::undefined();
    val readFile = val::undefined();

    if (!api.isUndefined()) {
      exists = api["exists"];
      readFile = api["readFile"];
    }

    auto findCandidate =
        [&exists](const std::vector<cxx::IncludeCandidate>& candidates)
        -> const cxx::IncludeCandidate* {
      if (exists.isUndefined()) return nullptr;

      for (auto& candidate : candidates) {
        if (exists(candidate.fileName).as<bool>()) return &candidate;
      }

      return nullptr;
    };

    while (true) {
      auto state = unit->continuePreprocessing();

      if (std::holds_alternative<cxx::ProcessingComplete>(state)) break;

      if (auto pendingInclude = std::get_if<cxx::PendingInclude>(&state)) {
        auto candidates = pendingInclude->candidates();

        if (auto found = findCandidate(candidates)) {
          pendingInclude->resolveWith(found->fileName, found->isSystemHeader);
        } else {
          pendingInclude->resolveWith(std::nullopt);
        }

      } else if (auto pendingHasIncludes =
                     std::get_if<cxx::PendingHasIncludes>(&state)) {
        for (auto& request : pendingHasIncludes->requests) {
          auto candidates = request.candidates();
          request.setExists(findCandidate(candidates) != nullptr);
        }
      } else if (auto pendingFileContent =
                     std::get_if<cxx::PendingFileContent>(&state)) {
        if (readFile.isUndefined()) {
          pendingFileContent->setContent(std::nullopt);
          continue;
        }

        val content = co_await readFile(pendingFileContent->fileName);

        if (content.isString()) {
          pendingFileContent->setContent(content.as<std::string>());
        } else {
          pendingFileContent->setContent(std::nullopt);
        }
      }
    }

    unit->endPreprocessing();

    unit->parse(cxx::ParserConfiguration{
        .checkTypes = true,
    });

    co_return val{true};
  }

  auto emitCode(const std::string& format) -> std::string {
#ifdef CXX_WITH_MLIR
    if (diagnosticsClient->hasErrors) {
      return {};
    }

    mlir::MLIRContext context{mlir::MLIRContext::Threading::DISABLED};

    context.loadDialect<mlir::cxx::CxxDialect>();

    cxx::Codegen codegen(context, unit.get(), debugInfo);

    auto ir = codegen(unit->ast());

    std::ostringstream out;
    llvm::raw_os_ostream os(out);

    if (format == "cxxir") {
      mlir::OpPrintingFlags flags;
      if (debugInfo) flags.enableDebugInfo(true);
      ir.module->print(os, flags);
      os.flush();
      return out.str();
    }

    if (failed(cxx::lowerToMLIR(ir.module))) {
      return std::format("<error lowering to {}>", format);
    }

    if (format == "mlir") {
      mlir::OpPrintingFlags flags;
      if (debugInfo) flags.enableDebugInfo(true);
      ir.module->print(os, flags);
      os.flush();
      return out.str();
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = cxx::exportToLLVMIR(ir.module, llvmContext);
    llvmModule->setSourceFileName(unit->fileName());

    if (format == "llvm") {
      llvmModule->print(os, nullptr);
      return out.str();
    }

    LLVMInitializeWebAssemblyTargetInfo();
    LLVMInitializeWebAssemblyTarget();
    LLVMInitializeWebAssemblyTargetMC();
    LLVMInitializeWebAssemblyAsmPrinter();

    llvm::TargetOptions opt;

    auto RM = std::optional<llvm::Reloc::Model>();

    auto triple = llvm::Triple{codegen.control()->memoryLayout()->triple()};

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);

    auto targetMachine = target->createTargetMachine(llvm::Triple{triple},
                                                     "generic", "", opt, RM);

    llvm::legacy::PassManager pm;

    llvm::SmallString<0> outputBuffer;
    llvm::raw_svector_ostream outBytes(outputBuffer);

    llvm::CodeGenFileType fileType = llvm::CodeGenFileType::AssemblyFile;
    if (targetMachine->addPassesToEmitFile(pm, outBytes, nullptr, fileType)) {
      return {};
    }

    pm.run(*llvmModule);

    return std::string(outputBuffer.begin(), outputBuffer.size());
#endif

    return {};
  }
};

auto getTokenText(std::intptr_t handle, std::intptr_t unitHandle)
    -> std::string {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);
  auto text = unit->tokenText(cxx::SourceLocation(handle));
  return text;
}

auto getTokenKind(std::intptr_t handle, std::intptr_t unitHandle) -> int {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);
  auto kind = unit->tokenKind(cxx::SourceLocation(handle));
  return static_cast<int>(kind);
}

auto getTokenLocation(std::intptr_t handle, std::intptr_t unitHandle) -> val {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);

  cxx::SourceLocation loc(handle);

  const auto start = unit->tokenStartPosition(loc);
  const auto end = unit->tokenEndPosition(loc);

  val result = val::object();

  result.set("fileName", std::string(start.fileName));
  result.set("startLine", start.line);
  result.set("startColumn", start.column);
  result.set("endLine", end.line);
  result.set("endColumn", end.column);

  return result;
}

auto getStartLocation(std::intptr_t handle, std::intptr_t unitHandle) -> val {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  const auto loc = ast->firstSourceLocation();
  if (!loc) return {};
  return getTokenLocation(loc.index(), unitHandle);
}

auto getEndLocation(std::intptr_t handle, std::intptr_t unitHandle) -> val {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  const auto loc = ast->lastSourceLocation().previous();
  if (!loc) return {};
  return getTokenLocation(loc.index(), unitHandle);
}

auto getIdentifierValue(std::intptr_t handle) -> val {
  auto id = reinterpret_cast<const cxx::Identifier*>(handle);
  if (!id) return {};
  return val(id->value());
}

auto getLiteralValue(std::intptr_t handle) -> val {
  auto id = reinterpret_cast<const cxx::Literal*>(handle);
  if (!id) return {};
  return val(id->value());
}

auto getASTKind(std::intptr_t handle) -> int {
  return static_cast<int>(((cxx::AST*)handle)->kind());
}

auto getListValue(std::intptr_t handle) -> int {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return std::intptr_t(list->value);
}

auto getListNext(std::intptr_t handle) -> std::intptr_t {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return std::intptr_t(list->next);
}

auto getASTSlot(std::intptr_t handle, int slot) -> std::intptr_t {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotNameIndex, slotCount] = getSlot(ast, slot);
  return value;
}

auto getASTSlotKind(std::intptr_t handle, int slot) -> int {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotNameIndex, slotCount] = getSlot(ast, slot);
  return static_cast<int>(slotKind);
}

auto getASTSlotName(std::intptr_t handle, int slot) -> int {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotName, slotCount] = getSlot(ast, slot);
  return static_cast<int>(slotName);
}

auto getASTSlotCount(std::intptr_t handle, int slot) -> int {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotNameIndex, slotCount] = getSlot(ast, slot);
  return static_cast<int>(slotCount);
}

auto createUnit(std::string source, std::string filename, UnitOptions api)
    -> WrappedUnit* {
  auto wrapped = new WrappedUnit(std::move(source), std::move(filename), api);

  return wrapped;
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx) {
  register_type<UnitOptions>(
      "UnitOptions",
      R"({ appdir?: string | undefined; sysroot?: string | undefined; std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined; defines?: string[] | undefined; undefines?: string[] | undefined; quoteIncludePaths?: string[] | undefined; includePaths?: string[] | undefined; systemIncludePaths?: string[] | undefined; debugInfo?: boolean | undefined; exists?: ((path: string) => boolean) | undefined; readFile?: ((path: string) => Promise<string | undefined>) | undefined })");

  register_type<DiagnosticList>(
      "DiagnosticList",
      R"(Array<{ fileName: string; startLine: number; startColumn: number; endLine: number; endColumn: number; message: string; severity: "message" | "note" | "warning" | "error" | "fatal"; notes: Array<{ fileName: string; startLine: number; startColumn: number; endLine: number; endColumn: number; message: string }> }>)");

  class_<WrappedUnit>("Unit")
      .function("parse", &WrappedUnit::parse)
      .function("getHandle", &WrappedUnit::getHandle)
      .function("getUnitHandle", &WrappedUnit::getUnitHandle)
      .function("getDiagnostics", &WrappedUnit::getDiagnostics)
      .function("emitCode", &WrappedUnit::emitCode);

  function("createUnit", &createUnit, allow_raw_pointers());
  function("getASTKind", &getASTKind);
  function("getListValue", &getListValue);
  function("getListNext", &getListNext);
  function("getASTSlot", &getASTSlot);
  function("getASTSlotKind", &getASTSlotKind);
  function("getASTSlotName", &getASTSlotName);
  function("getASTSlotCount", &getASTSlotCount);
  function("getTokenKind", &getTokenKind);
  function("getTokenText", &getTokenText);
  function("getTokenLocation", &getTokenLocation);
  function("getStartLocation", &getStartLocation);
  function("getEndLocation", &getEndLocation);
  function("getIdentifierValue", &getIdentifierValue);
  function("getLiteralValue", &getLiteralValue);
}
