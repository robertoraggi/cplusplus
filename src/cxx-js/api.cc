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

#include <cxx/control.h>
#include <cxx/translation_unit.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;

struct DiagnosticClient : cxx::DiagnosticClient {
  val& messages;

  explicit DiagnosticClient(val& messages) : messages(messages) {}

  void report(const cxx::Diagnostic& diag) override {
    val d = val::object();
    d.set("fileName", val(diag.fileName()));
    d.set("line", val(diag.line()));
    d.set("column", val(diag.column()));
    d.set("message", val(diag.message()));
    messages.call<void>("push", d);
  }
};

static val parse(std::string source, std::string filename) {
  val messages = val::array();

  cxx::Control control;
  DiagnosticClient diagnosticClient(messages);
  cxx::TranslationUnit unit(&control);
  unit.setDiagnosticClient(&diagnosticClient);
  unit.setFileName(std::move(filename));
  unit.setSource(std::move(source));

  const auto parsed = unit.parse();

  val result = val::object();
  result.set("diagnostics", messages);

  return result;
}

EMSCRIPTEN_BINDINGS(my_module) { function("parse", &parse); }
