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

#include <cxx/scope.h>
#include <cxx/symbols.h>

namespace cxx {

NamespaceSymbol::NamespaceSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

NamespaceSymbol::~NamespaceSymbol() {}

ConceptSymbol::ConceptSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConceptSymbol::~ConceptSymbol() {}

ClassSymbol::ClassSymbol(Scope* enclosingScope) : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

ClassSymbol::~ClassSymbol() {}

UnionSymbol::UnionSymbol(Scope* enclosingScope) : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

UnionSymbol::~UnionSymbol() {}

EnumSymbol::EnumSymbol(Scope* enclosingScope) : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

EnumSymbol::~EnumSymbol() {}

ScopedEnumSymbol::ScopedEnumSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

ScopedEnumSymbol::~ScopedEnumSymbol() {}

FunctionSymbol::FunctionSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

FunctionSymbol::~FunctionSymbol() {}

LambdaSymbol::LambdaSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

LambdaSymbol::~LambdaSymbol() {}

FunctionParametersSymbol::FunctionParametersSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

FunctionParametersSymbol::~FunctionParametersSymbol() {}

TemplateParametersSymbol::TemplateParametersSymbol(Scope* enclosingScope)
    : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

TemplateParametersSymbol::~TemplateParametersSymbol() {}

BlockSymbol::BlockSymbol(Scope* enclosingScope) : Symbol(Kind, enclosingScope) {
  scope_ = std::make_unique<Scope>(enclosingScope);
  scope_->setOwner(this);
}

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
