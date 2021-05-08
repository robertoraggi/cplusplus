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

#include <cxx/symbol_factory.h>
#include <cxx/symbols.h>

#include <forward_list>

namespace cxx {

struct SymbolFactory::Private {
  std::forward_list<NamespaceSymbol> namespaceSymbols;
  std::forward_list<ClassSymbol> classSymbols;
  std::forward_list<ConceptSymbol> conceptSymbols;
  std::forward_list<TypedefSymbol> typedefSymbols;
  std::forward_list<EnumSymbol> enumSymbols;
  std::forward_list<EnumeratorSymbol> enumeratorSymbols;
  std::forward_list<ScopedEnumSymbol> scopedEnumSymbols;
  std::forward_list<TemplateClassSymbol> templateClassSymbols;
  std::forward_list<TemplateFunctionSymbol> templateFunctionSymbols;
  std::forward_list<TemplateTypeParameterSymbol> templateTypeParameterSymbols;
  std::forward_list<VariableSymbol> variableSymbols;
  std::forward_list<FieldSymbol> fieldSymbols;
  std::forward_list<FunctionSymbol> functionSymbols;
  std::forward_list<ArgumentSymbol> argumentSymbols;
  std::forward_list<BlockSymbol> blockSymbols;
};

SymbolFactory::SymbolFactory() : d(std::make_unique<Private>()) {}

SymbolFactory::~SymbolFactory() {}

NamespaceSymbol* SymbolFactory::newNamespaceSymbol(Scope* enclosingScope,
                                                   const Name* name) {
  auto symbol = &d->namespaceSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

ClassSymbol* SymbolFactory::newClassSymbol(Scope* enclosingScope,
                                           const Name* name) {
  auto symbol = &d->classSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

ConceptSymbol* SymbolFactory::newConceptSymbol(Scope* enclosingScope,
                                               const Name* name) {
  auto symbol = &d->conceptSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

TypedefSymbol* SymbolFactory::newTypedefSymbol(Scope* enclosingScope,
                                               const Name* name) {
  auto symbol = &d->typedefSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

EnumSymbol* SymbolFactory::newEnumSymbol(Scope* enclosingScope,
                                         const Name* name) {
  auto symbol = &d->enumSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

EnumeratorSymbol* SymbolFactory::newEnumeratorSymbol(Scope* enclosingScope,
                                                     const Name* name) {
  auto symbol = &d->enumeratorSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

ScopedEnumSymbol* SymbolFactory::newScopedEnumSymbol(Scope* enclosingScope,
                                                     const Name* name) {
  auto symbol = &d->scopedEnumSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

TemplateClassSymbol* SymbolFactory::newTemplateClassSymbol(
    Scope* enclosingScope, const Name* name) {
  auto symbol = &d->templateClassSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

TemplateFunctionSymbol* SymbolFactory::newTemplateFunctionSymbol(
    Scope* enclosingScope, const Name* name) {
  auto symbol = &d->templateFunctionSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

TemplateTypeParameterSymbol* SymbolFactory::newTemplateTypeParameterSymbol(
    Scope* enclosingScope, const Name* name) {
  auto symbol =
      &d->templateTypeParameterSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

VariableSymbol* SymbolFactory::newVariableSymbol(Scope* enclosingScope,
                                                 const Name* name) {
  auto symbol = &d->variableSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

FieldSymbol* SymbolFactory::newFieldSymbol(Scope* enclosingScope,
                                           const Name* name) {
  auto symbol = &d->fieldSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

FunctionSymbol* SymbolFactory::newFunctionSymbol(Scope* enclosingScope,
                                                 const Name* name) {
  auto symbol = &d->functionSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

ArgumentSymbol* SymbolFactory::newArgumentSymbol(Scope* enclosingScope,
                                                 const Name* name) {
  auto symbol = &d->argumentSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

BlockSymbol* SymbolFactory::newBlockSymbol(Scope* enclosingScope,
                                           const Name* name) {
  auto symbol = &d->blockSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

}  // namespace cxx