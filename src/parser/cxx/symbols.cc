// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/symbols.h>

// cxx
#include <cxx/scope.h>
#include <cxx/types.h>

namespace cxx {

auto Symbol::next() const -> Symbol* {
  for (auto sym = link_; sym; sym = sym->link_) {
    if (sym->name_ == name_) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingSymbol() const -> Symbol* {
  if (!enclosingScope_) return nullptr;
  return enclosingScope_->owner();
}

ScopedSymbol::ScopedSymbol(SymbolKind kind, Scope* enclosingScope)
    : Symbol(kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

ScopedSymbol::~ScopedSymbol() {}

auto ScopedSymbol::scope() const -> Scope* { return scope_.get(); }

NamespaceSymbol::NamespaceSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

NamespaceSymbol::~NamespaceSymbol() {}

ConceptSymbol::ConceptSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConceptSymbol::~ConceptSymbol() {}

BaseClassSymbol::BaseClassSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

BaseClassSymbol::~BaseClassSymbol() {}

ClassSymbol::ClassSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

ClassSymbol::~ClassSymbol() {}

auto ClassSymbol::isUnion() const -> bool { return isUnion_; }

void ClassSymbol::setIsUnion(bool isUnion) { isUnion_ = isUnion; }

auto ClassSymbol::isFinal() const -> bool { return isFinal_; }

void ClassSymbol::setFinal(bool isFinal) { isFinal_ = isFinal; }

auto ClassSymbol::baseClasses() const -> const std::vector<BaseClassSymbol*>& {
  return baseClasses_;
}

void ClassSymbol::addBaseClass(BaseClassSymbol* baseClass) {
  baseClasses_.push_back(baseClass);
}

auto ClassSymbol::constructors() const -> const std::vector<FunctionSymbol*>& {
  return constructors_;
}

void ClassSymbol::addConstructor(FunctionSymbol* constructor) {
  constructors_.push_back(constructor);
}

auto ClassSymbol::templateParameters() const -> TemplateParametersSymbol* {
  return templateParameters_;
}

void ClassSymbol::setTemplateParameters(
    TemplateParametersSymbol* templateParameters) {
  templateParameters_ = templateParameters;
}

auto ClassSymbol::isComplete() const -> bool { return isComplete_; }

void ClassSymbol::setComplete(bool isComplete) { isComplete_ = isComplete; }

auto ClassSymbol::sizeInBytes() const -> std::size_t { return sizeInBytes_; }

auto ClassSymbol::hasBaseClass(Symbol* symbol) const -> bool {
  std::unordered_set<const ClassSymbol*> processed;
  return hasBaseClass(symbol, processed);
}

auto ClassSymbol::hasBaseClass(
    Symbol* symbol, std::unordered_set<const ClassSymbol*>& processed) const
    -> bool {
  if (!processed.insert(this).second) {
    return false;
  }

  for (auto baseClass : baseClasses_) {
    auto baseClassSymbol = baseClass->symbol();
    if (baseClassSymbol == symbol) return true;
    if (auto baseClassType = type_cast<ClassType>(baseClassSymbol->type())) {
      if (baseClassType->symbol()->hasBaseClass(symbol, processed)) return true;
    }
  }
  return false;
}

EnumSymbol::EnumSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

EnumSymbol::~EnumSymbol() {}

ScopedEnumSymbol::ScopedEnumSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

ScopedEnumSymbol::~ScopedEnumSymbol() {}

FunctionSymbol::FunctionSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

FunctionSymbol::~FunctionSymbol() {}

OverloadSetSymbol::OverloadSetSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

OverloadSetSymbol::~OverloadSetSymbol() {}

LambdaSymbol::LambdaSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

LambdaSymbol::~LambdaSymbol() {}

FunctionParametersSymbol::FunctionParametersSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

FunctionParametersSymbol::~FunctionParametersSymbol() {}

TemplateParametersSymbol::TemplateParametersSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

TemplateParametersSymbol::~TemplateParametersSymbol() {}

BlockSymbol::BlockSymbol(Scope* enclosingScope)
    : ScopedSymbol(Kind, enclosingScope) {}

BlockSymbol::~BlockSymbol() {}

TypeAliasSymbol::TypeAliasSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeAliasSymbol::~TypeAliasSymbol() {}

VariableSymbol::VariableSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

VariableSymbol::~VariableSymbol() {}

FieldSymbol::FieldSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

FieldSymbol::~FieldSymbol() {}

ParameterSymbol::ParameterSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ParameterSymbol::~ParameterSymbol() {}

TypeParameterSymbol::TypeParameterSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeParameterSymbol::~TypeParameterSymbol() {}

NonTypeParameterSymbol::NonTypeParameterSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

NonTypeParameterSymbol::~NonTypeParameterSymbol() {}

TemplateTypeParameterSymbol::TemplateTypeParameterSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TemplateTypeParameterSymbol::~TemplateTypeParameterSymbol() {}

ConstraintTypeParameterSymbol::ConstraintTypeParameterSymbol(
    Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConstraintTypeParameterSymbol::~ConstraintTypeParameterSymbol() {}

EnumeratorSymbol::EnumeratorSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

EnumeratorSymbol::~EnumeratorSymbol() {}

}  // namespace cxx
