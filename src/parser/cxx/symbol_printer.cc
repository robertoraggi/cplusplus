// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbol_printer.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

SymbolPrinter::SymbolPrinter(std::ostream& out) : out(out) {}

void SymbolPrinter::print(Symbol* symbol, int depth) {
  if (!symbol) return;

  std::swap(depth_, depth);
  symbol->accept(this);
  std::swap(depth_, depth);
}

void SymbolPrinter::newline() { out << std::endl; }

void SymbolPrinter::indent() {
  ++depth_;
  indent_ = std::string(depth_ * 2, ' ');
}

void SymbolPrinter::deindent() {
  --depth_;
  indent_ = std::string(depth_ * 2, ' ');
}

auto SymbolPrinter::symbolName(Symbol* symbol) -> std::string {
  if (!symbol->name()) return {};
  return fmt::format("{}", *symbol->name());
}

void SymbolPrinter::printSymbolHead(const std::string_view& kind,
                                    const Name* name) {
  fmt::print(out, "{} - {}", indent_, kind);
  if (name) fmt::print(out, " {}", *name);
}

void SymbolPrinter::printScope(Scope* scope) {
  if (!scope) return;
  for (auto symbol : *scope) {
    symbol->accept(this);
  }
}

void SymbolPrinter::visit(ConceptSymbol* symbol) {
  printSymbolHead("concept:", symbol->name());
  newline();
  indent();
  if (auto params = symbol->templateParameterList()) params->accept(this);
  deindent();
}

void SymbolPrinter::visit(NamespaceSymbol* symbol) {
  printSymbolHead("namespace:", symbol->name());
  newline();
  indent();
  printScope(symbol->scope());
  deindent();
}

void SymbolPrinter::visit(ClassSymbol* symbol) {
  std::string_view templ = symbol->templateParameterList() ? "template " : "";
  auto classKey = to_string_view(symbol->classKey());
  printSymbolHead(fmt::format("{}{}:", templ, classKey), symbol->name());
  newline();
  indent();
  if (auto params = symbol->templateParameterList()) params->accept(this);
  printScope(symbol->scope());
  deindent();
}

void SymbolPrinter::visit(TypedefSymbol* symbol) {
  printSymbolHead("type alias:", symbol->name());
  newline();
  indent();
  if (auto params = symbol->templateParameterList()) params->accept(this);
  deindent();
}

void SymbolPrinter::visit(EnumSymbol* symbol) {
  printSymbolHead("enum:", symbol->name());
  newline();
}

void SymbolPrinter::visit(EnumeratorSymbol* symbol) {
  printSymbolHead("enumerator:", symbol->name());
  newline();
}

void SymbolPrinter::visit(ScopedEnumSymbol* symbol) {
  printSymbolHead("enum class:", symbol->name());
  newline();
  indent();
  printScope(symbol->scope());
  deindent();
}

void SymbolPrinter::visit(TemplateParameterList* symbol) {
  printSymbolHead("template parameters:", symbol->name());
  newline();
  indent();
  printScope(symbol->scope());
  deindent();
}

void SymbolPrinter::visit(TemplateTypeParameterSymbol* symbol) {
  printSymbolHead(
      symbol->isParameterPack() ? "type parameter pack:" : "type parameter:",
      symbol->name());

  if (symbol->defaultType()) {
    out << " = ";
    printType(out, symbol->defaultType());
  }

  newline();
}

void SymbolPrinter::visit(VariableSymbol* symbol) {
  printSymbolHead("variable: ");
  printType(out, symbol->type(), symbolName(symbol));
  newline();
}

void SymbolPrinter::visit(FieldSymbol* symbol) {
  printSymbolHead("field: ");
  printType(out, symbol->type(), symbolName(symbol));
  newline();
}

void SymbolPrinter::visit(FunctionSymbol* symbol) {
  printSymbolHead(symbol->templateParameterList() ? "template function: "
                                                  : "function: ");
  printType(out, symbol->type(), symbolName(symbol));
  newline();
  indent();
  if (auto params = symbol->templateParameterList()) params->accept(this);
  deindent();
}

void SymbolPrinter::visit(ArgumentSymbol* symbol) {
  printSymbolHead("argument:", symbol->name());
  newline();
}

void SymbolPrinter::visit(BlockSymbol* symbol) {
  printSymbolHead("block:", symbol->name());
  newline();
}

}  // namespace cxx
