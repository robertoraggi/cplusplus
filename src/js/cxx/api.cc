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
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <optional>

#include "async_parse.h"
#include "emit_code.h"
#include "toolchain_options.h"

using namespace emscripten;

namespace {

cxx::ASTSlot getSlot;

EMSCRIPTEN_DECLARE_VAL_TYPE(UnitOptions);
EMSCRIPTEN_DECLARE_VAL_TYPE(DiagnosticList);

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
  std::string source;
  std::string fileName;
  bool debugInfo = true;

  WrappedUnit(std::string source, std::string fileName, UnitOptions api)
      : api(api), source(std::move(source)), fileName(std::move(fileName)) {
    diagnosticsClient = std::make_unique<DiagnosticsClient>();

    unit = std::make_unique<cxx::TranslationUnit>(diagnosticsClient.get());

    if (!api.isUndefined()) {
      if (val value = api["debugInfo"]; value.isTrue() || value.isFalse()) {
        debugInfo = value.as<bool>();
      }
    }

    toolchain = cxx::js::configureToolchain(unit.get(), api);
  }

  auto getUnitHandle() const -> std::intptr_t {
    return (std::intptr_t)unit.get();
  }

  auto getHandle() const -> std::intptr_t { return (std::intptr_t)unit->ast(); }

  auto getDiagnostics() const -> DiagnosticList {
    return DiagnosticList(diagnosticsClient->messages);
  }

  auto parse() -> val {
    cxx::js::AsyncParseRequest request{
        .unit = unit.get(),
        .source = std::move(source),
        .fileName = fileName,
        .config = {.checkTypes = true},
    };

    if (!api.isUndefined()) {
      request.exists = api["exists"];
      request.readFile = api["readFile"];
      request.shouldContinue = api["shouldContinue"];
    }

    return cxx::js::asyncParse(std::move(request));
  }

  auto emitCode(const std::string& format) -> val {
    const auto objectFile = format == "obj";

    auto emptyOutput = [&] {
      if (objectFile) return val::global("Uint8Array").new_(0);
      return val(std::string{});
    };

    if (diagnosticsClient->hasErrors) return emptyOutput();

    auto generated = cxx::js::generateCode(unit.get(), format, debugInfo);

    if (!generated) return emptyOutput();

    if (!objectFile) return val(std::move(generated->text));

    auto& objectCode = generated->objectCode;

    auto result = val::global("Uint8Array").new_(objectCode.size());

    val memory = val(typed_memory_view(objectCode.size(), objectCode.data()));

    result.call<void>("set", memory);

    return result;
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

auto createUnit(std::string source, std::string fileName, UnitOptions api)
    -> WrappedUnit* {
  return new WrappedUnit(std::move(source), std::move(fileName), api);
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx) {
  register_type<UnitOptions>(
      "UnitOptions",
      R"({ appdir?: string | undefined; sysroot?: string | undefined; std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined; defines?: string[] | undefined; undefines?: string[] | undefined; quoteIncludePaths?: string[] | undefined; includePaths?: string[] | undefined; systemIncludePaths?: string[] | undefined; debugInfo?: boolean | undefined; exists?: ((path: string) => boolean) | undefined; readFile?: ((path: string) => Promise<string | undefined>) | undefined; shouldContinue?: (() => Promise<boolean>) | undefined })");

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
