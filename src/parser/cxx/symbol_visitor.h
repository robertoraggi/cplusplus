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

#include <cxx/symbols_fwd.h>

namespace cxx {

class SymbolVisitor {
 public:
  virtual ~SymbolVisitor();

  virtual void visit(ClassSymbol*) = 0;
  virtual void visit(ConceptSymbol*) = 0;
  virtual void visit(DependentSymbol*) = 0;
  virtual void visit(EnumeratorSymbol*) = 0;
  virtual void visit(FunctionSymbol*) = 0;
  virtual void visit(GlobalSymbol*) = 0;
  virtual void visit(InjectedClassNameSymbol*) = 0;
  virtual void visit(LocalSymbol*) = 0;
  virtual void visit(MemberSymbol*) = 0;
  virtual void visit(NamespaceSymbol*) = 0;
  virtual void visit(NamespaceAliasSymbol*) = 0;
  virtual void visit(NonTypeTemplateParameterSymbol*) = 0;
  virtual void visit(ParameterSymbol*) = 0;
  virtual void visit(ScopedEnumSymbol*) = 0;
  virtual void visit(TemplateParameterSymbol*) = 0;
  virtual void visit(TemplateParameterPackSymbol*) = 0;
  virtual void visit(TypeAliasSymbol*) = 0;
  virtual void visit(ValueSymbol*) = 0;
};

}  // namespace cxx
