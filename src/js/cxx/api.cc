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

#include <cxx/ast.h>
#include <cxx/ast_slot.h>
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdlib>
#include <format>
#include <iostream>
#include <optional>
#include <sstream>

using namespace emscripten;

namespace {

cxx::ASTSlot getSlot;

struct DiagnosticsClient final : cxx::DiagnosticsClient {
  val messages = val::array();

  void report(const cxx::Diagnostic& diag) override {
    const auto start = preprocessor()->tokenStartPosition(diag.token());
    const auto end = preprocessor()->tokenEndPosition(diag.token());

    val d = val::object();
    d.set("fileName", val(std::string(start.fileName)));
    d.set("startLine", val(start.line));
    d.set("startColumn", val(start.column));
    d.set("endLine", val(end.line));
    d.set("endColumn", val(end.column));
    d.set("message", val(diag.message()));
    messages.call<void>("push", d);
  }
};

struct WrappedUnit {
  std::unique_ptr<DiagnosticsClient> diagnosticsClient;
  std::unique_ptr<cxx::TranslationUnit> unit;
  val api;

  WrappedUnit(std::string source, std::string filename, val api = {})
      : api(api) {
    diagnosticsClient = std::make_unique<DiagnosticsClient>();

    unit = std::make_unique<cxx::TranslationUnit>(diagnosticsClient.get());

    if (auto preprocessor = unit->preprocessor()) {
      preprocessor->setCanResolveFiles(true);
    }

    unit->beginPreprocessing(std::move(source), std::move(filename));
  }

  auto getUnitHandle() const -> std::intptr_t {
    return (std::intptr_t)unit.get();
  }

  auto getHandle() const -> std::intptr_t { return (std::intptr_t)unit->ast(); }

  auto getDiagnostics() const -> val { return diagnosticsClient->messages; }

  auto parse() -> val {
    val resolve = val::undefined();
    val readFile = val::undefined();

    if (!api.isUndefined()) {
      resolve = api["resolve"];
      readFile = api["readFile"];
      std::cerr << "set resolve and read file\n";
    } else {
      std::cerr << "no resolve and read file\n";
    }

    struct {
      auto operator()(const cxx::SystemInclude& include) -> val {
        return val(include.fileName);
      }
      auto operator()(const cxx::QuoteInclude& include) -> val {
        return val(include.fileName);
      }
    } getHeaderName;

    struct {
      val quoted{"quoted"};
      val angled{"angled"};

      auto operator()(const cxx::SystemInclude& include) -> val {
        return angled;
      }
      auto operator()(const cxx::QuoteInclude& include) -> val {
        return quoted;
      }
    } getIncludeType;

    while (true) {
      auto state = unit->continuePreprocessing();

      if (std::holds_alternative<cxx::ProcessingComplete>(state)) break;

      if (auto pendingInclude = std::get_if<cxx::PendingInclude>(&state)) {
        if (resolve.isUndefined()) {
          pendingInclude->resolveWith(std::nullopt);
          continue;
        }

        auto header = std::visit(getHeaderName, pendingInclude->include);
        auto includeType = std::visit(getIncludeType, pendingInclude->include);

        val resolved = co_await resolve(header, includeType,
                                        pendingInclude->isIncludeNext);

        if (resolved.isString()) {
          pendingInclude->resolveWith(resolved.as<std::string>());
        } else {
          pendingInclude->resolveWith(std::nullopt);
        }

      } else if (auto pendingHasIncludes =
                     std::get_if<cxx::PendingHasIncludes>(&state)) {
        for (auto& request : pendingHasIncludes->requests) {
          if (resolve.isUndefined()) {
            request.setExists(false);
            continue;
          }

          auto header = std::visit(getHeaderName, request.include);
          auto includeType = std::visit(getIncludeType, request.include);

          val resolved =
              co_await resolve(header, includeType, request.isIncludeNext);

          request.setExists(resolved.isString());
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

    unit->parse();

    co_return val{true};
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

auto createUnit(std::string source, std::string filename, val api)
    -> WrappedUnit* {
  auto wrapped = new WrappedUnit(std::move(source), std::move(filename), api);

  return wrapped;
}

auto lexerTokenKind(cxx::Lexer& lexer) -> int {
  return static_cast<int>(lexer.tokenKind());
}

auto lexerTokenText(cxx::Lexer& lexer) -> std::string {
  return std::string(lexer.tokenText());
}

auto lexerNext(cxx::Lexer& lexer) -> int {
  return static_cast<int>(lexer.next());
}

auto preprocesorPreprocess(cxx::Preprocessor& preprocessor, std::string source,
                           std::string filename) -> std::string {
  std::vector<cxx::Token> tokens;
  preprocessor.preprocess(std::move(source), std::move(filename), tokens);
  std::ostringstream out;
  preprocessor.getPreprocessedText(tokens, out);
  return out.str();
}

auto translationUnitGetAST(cxx::TranslationUnit& unit) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(unit.ast());
}

auto translationUnitGetUnitHandle(cxx::TranslationUnit& unit) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(&unit);
}

auto register_control(const char* name = "Control") -> class_<cxx::Control> {
  return class_<cxx::Control>(name).constructor();
}

auto register_diagnostics_client(const char* name = "DiagnosticsClient")
    -> class_<cxx::DiagnosticsClient> {
  return class_<cxx::DiagnosticsClient>(name)
      .constructor()  // ctor
      .function("setPreprocessor", &cxx::DiagnosticsClient::setPreprocessor,
                allow_raw_pointers());
}

auto register_preprocessor(const char* name = "Preprocessor")
    -> class_<cxx::Preprocessor> {
  return class_<cxx::Preprocessor>(name)
      .constructor<cxx::Control*, cxx::DiagnosticsClient*>()
      .function("preprocess", &preprocesorPreprocess)
      .function("addIncludePath", &cxx::Preprocessor::addSystemIncludePath)
      .function("defineMacro", &cxx::Preprocessor::defineMacro)
      .function("undefineMacro", &cxx::Preprocessor::undefMacro)
      .function("canResolveFiles", &cxx::Preprocessor::canResolveFiles)
      .function("setCanResolveFiles", &cxx::Preprocessor::setCanResolveFiles)
      .function("currentPath", &cxx::Preprocessor::currentPath)
      .function("setCurrentPath", &cxx::Preprocessor::setCurrentPath);
}

auto register_lexer(const char* name = "Lexer") -> class_<cxx::Lexer> {
  return class_<cxx::Lexer>(name)
      .constructor<std::string>()

      .property("preprocessing", &cxx::Lexer::preprocessing,
                &cxx::Lexer::setPreprocessing)

      .property("keepComments", &cxx::Lexer::keepComments,
                &cxx::Lexer::setKeepComments)

      .function("tokenKind", &lexerTokenKind)
      .function("tokenAtStartOfLine", &cxx::Lexer::tokenStartOfLine)
      .function("tokenHasLeadingSpace", &cxx::Lexer::tokenLeadingSpace)
      .function("tokenOffset", &cxx::Lexer::tokenPos)
      .function("tokenLength", &cxx::Lexer::tokenLength)
      .function("tokenText", &lexerTokenText)
      .function("next", &lexerNext);
}

auto register_translation_unit(const char* name = "TranslationUnit")
    -> class_<cxx::TranslationUnit> {
  return class_<cxx::TranslationUnit>(name)
      .constructor<cxx::DiagnosticsClient*>()
      .function("setSource", &cxx::TranslationUnit::setSource)
      .function("parse", &cxx::TranslationUnit::parse)
      .function("tokenCount", &cxx::TranslationUnit::tokenCount)
      .function("getAST", &translationUnitGetAST)
      .function("getUnitHandle", &translationUnitGetUnitHandle);
}

}  // namespace

EMSCRIPTEN_BINDINGS(cxx) {
  register_control();
  register_diagnostics_client();
  register_preprocessor();
  register_lexer();
  register_translation_unit();

  class_<WrappedUnit>("Unit")
      .function("parse", &WrappedUnit::parse)
      .function("getHandle", &WrappedUnit::getHandle)
      .function("getUnitHandle", &WrappedUnit::getUnitHandle)
      .function("getDiagnostics", &WrappedUnit::getDiagnostics);

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
