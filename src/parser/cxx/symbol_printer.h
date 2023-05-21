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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/symbol_visitor.h>
#include <cxx/type_printer.h>

#include <iosfwd>
#include <string_view>

namespace cxx {

class SymbolPrinter final : SymbolVisitor {
 public:
  explicit SymbolPrinter(std::ostream& out);

  void operator()(Symbol* symbol, int depth = 0) { print(symbol, depth); }

  void print(Symbol* symbol, int depth);

 private:
  void printSymbolHead(const std::string_view& kind,
                       const Name* name = nullptr);

  void printScope(Scope* scope);
  void newline();
  void indent();
  void deindent();

  auto symbolName(Symbol* symbol) -> std::string;

  void visit(ConceptSymbol* symbol) override;
  void visit(NamespaceSymbol* symbol) override;
  void visit(ClassSymbol* symbol) override;
  void visit(TypedefSymbol* symbol) override;
  void visit(EnumSymbol* symbol) override;
  void visit(EnumeratorSymbol* symbol) override;
  void visit(ScopedEnumSymbol* symbol) override;
  void visit(TemplateParameterList* symbol) override;
  void visit(TemplateTypeParameterSymbol* symbol) override;
  void visit(VariableSymbol* symbol) override;
  void visit(FieldSymbol* symbol) override;
  void visit(FunctionSymbol* symbol) override;
  void visit(ArgumentSymbol* symbol) override;
  void visit(BlockSymbol* symbol) override;

 private:
  std::ostream& out;
  TypePrinter printType;
  std::string indent_;
  int depth_ = 0;
};

}  // namespace cxx
