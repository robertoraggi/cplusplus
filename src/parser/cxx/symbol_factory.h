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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>

#include <memory>

namespace cxx {

class SymbolFactory final {
 public:
  SymbolFactory();
  ~SymbolFactory();

  auto newNamespaceSymbol(Scope* enclosingScope, const Name* name)
      -> NamespaceSymbol*;

  auto newClassSymbol(Scope* enclosingScope, const Name* name) -> ClassSymbol*;

  auto newTypedefSymbol(Scope* enclosingScope, const Name* name)
      -> TypedefSymbol*;

  auto newConceptSymbol(Scope* enclosingScope, const Name* name)
      -> ConceptSymbol*;

  auto newEnumSymbol(Scope* enclosingScope, const Name* name) -> EnumSymbol*;

  auto newEnumeratorSymbol(Scope* enclosingScope, const Name* name)
      -> EnumeratorSymbol*;

  auto newScopedEnumSymbol(Scope* enclosingScope, const Name* name)
      -> ScopedEnumSymbol*;

  auto newTemplateParameterList(Scope* enclosingScope)
      -> TemplateParameterList*;

  auto newTemplateTypeParameterSymbol(Scope* enclosingScope, const Name* name)
      -> TemplateTypeParameterSymbol*;

  auto newVariableSymbol(Scope* enclosingScope, const Name* name)
      -> VariableSymbol*;

  auto newFieldSymbol(Scope* enclosingScope, const Name* name) -> FieldSymbol*;

  auto newFunctionSymbol(Scope* enclosingScope, const Name* name)
      -> FunctionSymbol*;

  auto newArgumentSymbol(Scope* enclosingScope, const Name* name)
      -> ArgumentSymbol*;

  auto newBlockSymbol(Scope* enclosingScope, const Name* name) -> BlockSymbol*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx
