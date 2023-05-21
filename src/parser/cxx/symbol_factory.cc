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
  std::forward_list<TemplateParameterList> templateParameterLists;
  std::forward_list<TemplateTypeParameterSymbol> templateTypeParameterSymbols;
  std::forward_list<VariableSymbol> variableSymbols;
  std::forward_list<FieldSymbol> fieldSymbols;
  std::forward_list<FunctionSymbol> functionSymbols;
  std::forward_list<ArgumentSymbol> argumentSymbols;
  std::forward_list<BlockSymbol> blockSymbols;
};

SymbolFactory::SymbolFactory() : d(std::make_unique<Private>()) {}

SymbolFactory::~SymbolFactory() = default;

auto SymbolFactory::newNamespaceSymbol(Scope* enclosingScope, const Name* name)
    -> NamespaceSymbol* {
  auto symbol = &d->namespaceSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newClassSymbol(Scope* enclosingScope, const Name* name)
    -> ClassSymbol* {
  auto symbol = &d->classSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newConceptSymbol(Scope* enclosingScope, const Name* name)
    -> ConceptSymbol* {
  auto symbol = &d->conceptSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newTypedefSymbol(Scope* enclosingScope, const Name* name)
    -> TypedefSymbol* {
  auto symbol = &d->typedefSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newEnumSymbol(Scope* enclosingScope, const Name* name)
    -> EnumSymbol* {
  auto symbol = &d->enumSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newEnumeratorSymbol(Scope* enclosingScope, const Name* name)
    -> EnumeratorSymbol* {
  auto symbol = &d->enumeratorSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newScopedEnumSymbol(Scope* enclosingScope, const Name* name)
    -> ScopedEnumSymbol* {
  auto symbol = &d->scopedEnumSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newTemplateParameterList(Scope* enclosingScope)
    -> TemplateParameterList* {
  auto symbol = &d->templateParameterLists.emplace_front(enclosingScope);
  return symbol;
}

auto SymbolFactory::newTemplateTypeParameterSymbol(Scope* enclosingScope,
                                                   const Name* name)
    -> TemplateTypeParameterSymbol* {
  auto symbol =
      &d->templateTypeParameterSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newVariableSymbol(Scope* enclosingScope, const Name* name)
    -> VariableSymbol* {
  auto symbol = &d->variableSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newFieldSymbol(Scope* enclosingScope, const Name* name)
    -> FieldSymbol* {
  auto symbol = &d->fieldSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newFunctionSymbol(Scope* enclosingScope, const Name* name)
    -> FunctionSymbol* {
  auto symbol = &d->functionSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newArgumentSymbol(Scope* enclosingScope, const Name* name)
    -> ArgumentSymbol* {
  auto symbol = &d->argumentSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

auto SymbolFactory::newBlockSymbol(Scope* enclosingScope, const Name* name)
    -> BlockSymbol* {
  auto symbol = &d->blockSymbols.emplace_front(enclosingScope, name);
  return symbol;
}

}  // namespace cxx