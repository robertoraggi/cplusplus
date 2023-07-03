// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <sstream>

using namespace emscripten;

namespace {

cxx::ASTSlot getSlot;

struct DiagnosticsClient final : cxx::DiagnosticsClient {
  val messages = val::array();

  void report(const cxx::Diagnostic& diag) override {
    std::string_view fileName;
    uint32_t line = 0;
    uint32_t column = 0;

    preprocessor()->getTokenStartPosition(diag.token(), &line, &column,
                                          &fileName);

    uint32_t endLine = 0;
    uint32_t endColumn = 0;

    preprocessor()->getTokenEndPosition(diag.token(), &endLine, &endColumn,
                                        nullptr);

    val d = val::object();
    d.set("fileName", val(std::string(fileName)));
    d.set("startLine", val(line));
    d.set("startColumn", val(column));
    d.set("endLine", val(endLine));
    d.set("endColumn", val(endColumn));
    d.set("message", val(diag.message()));
    messages.call<void>("push", d);
  }
};

struct WrappedUnit {
  cxx::Control control;
  std::unique_ptr<DiagnosticsClient> diagnosticsClient;
  std::unique_ptr<cxx::TranslationUnit> unit;

  WrappedUnit(std::string source, std::string filename) {
    diagnosticsClient = std::make_unique<DiagnosticsClient>();
    unit = std::make_unique<cxx::TranslationUnit>(&control,
                                                  diagnosticsClient.get());
    if (auto preprocessor = unit->preprocessor()) {
      preprocessor->setCanResolveFiles(false);
    }

    unit->setSource(std::move(source), std::move(filename));
  }

  intptr_t getUnitHandle() const { return (intptr_t)unit.get(); }

  intptr_t getHandle() const { return (intptr_t)unit->ast(); }

  val getDiagnostics() const { return diagnosticsClient->messages; }

  bool parse() { return unit->parse(); }
};

std::string getTokenText(intptr_t handle, intptr_t unitHandle) {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);
  auto text = unit->tokenText(cxx::SourceLocation(handle));
  return text;
}

int getTokenKind(intptr_t handle, intptr_t unitHandle) {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);
  auto kind = unit->tokenKind(cxx::SourceLocation(handle));
  return static_cast<int>(kind);
}

val getTokenLocation(intptr_t handle, intptr_t unitHandle) {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);

  cxx::SourceLocation loc(handle);

  unsigned startLine = 0, startColumn = 0;

  unit->getTokenStartPosition(loc, &startLine, &startColumn);

  unsigned endLine = 0, endColumn = 0;

  unit->getTokenEndPosition(loc, &endLine, &endColumn);

  val result = val::object();

  result.set("startLine", startLine);
  result.set("startColumn", startColumn);
  result.set("endLine", endLine);
  result.set("endColumn", endColumn);

  return result;
}

val getStartLocation(intptr_t handle, intptr_t unitHandle) {
  auto ast = (cxx::AST*)handle;
  return getTokenLocation(ast->firstSourceLocation().index(), unitHandle);
}

val getEndLocation(intptr_t handle, intptr_t unitHandle) {
  auto ast = (cxx::AST*)handle;
  return getTokenLocation(ast->lastSourceLocation().previous().index(),
                          unitHandle);
}

int getASTKind(intptr_t handle) {
  return static_cast<int>(((cxx::AST*)handle)->kind());
}

int getListValue(intptr_t handle) {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return intptr_t(list->value);
}

intptr_t getListNext(intptr_t handle) {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return intptr_t(list->next);
}

intptr_t getASTSlot(intptr_t handle, int slot) {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotCount] = getSlot(ast, slot);
  return value;
}

int getASTSlotKind(intptr_t handle, int slot) {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotCount] = getSlot(ast, slot);
  return static_cast<int>(slotKind);
}

int getASTSlotCount(intptr_t handle, int slot) {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto [value, slotKind, slotCount] = getSlot(ast, slot);
  return static_cast<int>(slotCount);
}

WrappedUnit* createUnit(std::string source, std::string filename) {
  auto wrapped = new WrappedUnit(std::move(source), std::move(filename));

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
  std::ostringstream out;
  preprocessor.preprocess(std::move(source), std::move(filename), out);
  return out.str();
}

auto translationUnitGetAST(cxx::TranslationUnit& unit) -> intptr_t {
  return reinterpret_cast<intptr_t>(unit.ast());
}

auto translationUnitGetUnitHandle(cxx::TranslationUnit& unit) -> intptr_t {
  return reinterpret_cast<intptr_t>(&unit);
}

auto register_control(const char* name = "Control") -> class_<cxx::Control> {
  return class_<cxx::Control>(name).constructor();
}

auto register_diagnostics_client(const char* name = "DiagnosticsClient")
    -> class_<cxx::DiagnosticsClient> {
  return class_<cxx::DiagnosticsClient>(name).constructor();
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
      .constructor<cxx::Control*, cxx::DiagnosticsClient*>()
      .function("setSource", &cxx::TranslationUnit::setSource)
      .function("parse", &cxx::TranslationUnit::parse)
      .function("tokenCount", &cxx::TranslationUnit::tokenCount)
      .function("getAST", &translationUnitGetAST)
      .function("getUnitHandle", &translationUnitGetUnitHandle);
}

}  // namespace

EMSCRIPTEN_BINDINGS(my_module) {
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
  function("getASTSlotCount", &getASTSlotCount);
  function("getTokenKind", &getTokenKind);
  function("getTokenText", &getTokenText);
  function("getTokenLocation", &getTokenLocation);
  function("getStartLocation", &getStartLocation);
  function("getEndLocation", &getEndLocation);
}
