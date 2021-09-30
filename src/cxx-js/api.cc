// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;

struct DiagnosticsClient : cxx::DiagnosticsClient {
  val& messages;

  explicit DiagnosticsClient(val& messages) : messages(messages) {}

  void report(const cxx::Diagnostic& diag) override {
    std::string_view fileName;
    uint32_t line = 0;
    uint32_t column = 0;

    preprocessor()->getTokenStartPosition(diag.token(), &line, &column,
                                          &fileName);

    val d = val::object();
    d.set("fileName", val(std::string(fileName)));
    d.set("line", val(line));
    d.set("column", val(column));
    d.set("message", val(diag.message()));
    messages.call<void>("push", d);
  }
};

struct WrappedUnit {
  val messages = val::array();
  DiagnosticsClient diagnosticsClient{messages};

  cxx::Control control;
  cxx::TranslationUnit unit{&control, &diagnosticsClient};

  WrappedUnit() = default;

  intptr_t getUnitHandle() const { return (intptr_t)&unit; }

  intptr_t getHandle() const { return (intptr_t)unit.ast(); }

  val getDiagnostics() const { return messages; }

  bool parse() { return unit.parse(); }
};

static std::string getTokenText(intptr_t handle, intptr_t unitHandle) {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);
  auto text = unit->tokenText(cxx::SourceLocation(handle));
  return text;
}

static val getTokenLocation(intptr_t handle, intptr_t unitHandle) {
  auto unit = reinterpret_cast<cxx::TranslationUnit*>(unitHandle);

  cxx::SourceLocation loc(handle);

  unsigned startLine = 0, startColumn = 0;

  unit->getTokenStartPosition(loc, &startLine, &startColumn);

  unsigned endLine = 0, endColumn = 0;

  unit->getTokenEndPosition(loc, &startLine, &startColumn);

  val result = val::object();

  result.set("startLine", startLine);
  result.set("startColumn", startColumn);
  result.set("endLine", endLine);
  result.set("endColumn", endColumn);

  return result;
}

static int getASTKind(intptr_t handle) {
  return static_cast<int>(((cxx::AST*)handle)->kind());
}

static int getListValue(intptr_t handle) {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return intptr_t(list->value);
}

static intptr_t getListNext(intptr_t handle) {
  auto list = reinterpret_cast<cxx::List<cxx::AST*>*>(handle);
  return intptr_t(list->next);
}

namespace {
cxx::ASTSlot getSlot;
}

static intptr_t getASTSlot(intptr_t handle, int slot) {
  auto ast = reinterpret_cast<cxx::AST*>(handle);
  auto value = getSlot(ast, slot);
  return value;
}

static WrappedUnit* createUnit(std::string source, std::string filename) {
  val messages = val::array();

  auto wrapped = new WrappedUnit();

  DiagnosticsClient diagnosticsClient(messages);
  wrapped->unit.setSource(std::move(source), filename);

  return wrapped;
}

EMSCRIPTEN_BINDINGS(my_module) {
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
  function("getTokenText", &getTokenText);
  function("getTokenLocation", &getTokenLocation);
}
