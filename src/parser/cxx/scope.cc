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

#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>

namespace cxx {

namespace {

struct AddTemplateSymbol {
  TemplateParametersSymbol* templateParameters;

  auto addSymbol(auto symbol) {
    auto templateScope = templateParameters->scope();
    auto parentScope = templateScope->enclosingNonTemplateParametersScope();
    symbol->setTemplateParameters(templateParameters);
    parentScope->addSymbol(symbol);
  }

  auto operator()(TypeAliasSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(ConceptSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(ClassSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(FunctionSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(VariableSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(auto) -> bool { return false; }
};

}  // namespace

Scope::Scope(Scope* parent) : parent_(parent) {}

Scope::~Scope() {}

auto Scope::isEnumScope() const -> bool {
  return owner_ && (owner_->isEnum() || owner_->isScopedEnum());
}

auto Scope::isTemplateParametersScope() const -> bool {
  return owner_ && owner_->isTemplateParameters();
}

auto Scope::enclosingNonTemplateParametersScope() const -> Scope* {
  auto scope = parent_;

  while (scope && scope->isTemplateParametersScope()) {
    scope = scope->parent();
  }

  return scope;
}

void Scope::addSymbol(Symbol* symbol) {
  if (symbol->isTemplateParameters()) return;

  if (auto templateParameters = symbol_cast<TemplateParametersSymbol>(owner_)) {
    if (visit(AddTemplateSymbol{templateParameters}, symbol)) return;
  }

  symbol->setEnclosingScope(this);
  symbol->setInsertionPoint(symbols_.size());
  symbols_.emplace(symbol->name(), symbol);
}

void Scope::removeSymbol(Symbol* symbol) {
  auto [first, last] = symbols_.equal_range(symbol->name());
  for (auto it = first; it != last; ++it) {
    if (it->second == symbol) {
      symbols_.erase(it);
      break;
    }
  }
}

auto Scope::usingDirectives() const -> const std::vector<Scope*>& {
  return usingDirectives_;
}

void Scope::addUsingDirective(Scope* scope) {
  usingDirectives_.push_back(scope);
}

}  // namespace cxx
