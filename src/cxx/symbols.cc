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

#include <cxx/fully_specified_type.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

Symbol::Symbol(Scope* enclosingScope, const Name* name)
    : enclosingScope_(enclosingScope), name_(name) {}

Symbol::~Symbol() {}

const Name* Symbol::name() const { return name_; }

void Symbol::setName(const Name* name) { name_ = name; }

Scope* Symbol::enclosingScope() const { return enclosingScope_; }

void Symbol::setEnclosingScope(Scope* enclosingScope) {
  enclosingScope_ = enclosingScope;
}

const FullySpecifiedType& Symbol::type() const { return type_; }

void Symbol::setType(const FullySpecifiedType& type) { type_ = type; }

NamespaceSymbol::NamespaceSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

NamespaceSymbol::~NamespaceSymbol() {}

ClassSymbol::ClassSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

ClassSymbol::~ClassSymbol() {}

TypedefSymbol::TypedefSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

EnumSymbol::EnumSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

ScopedEnumSymbol::ScopedEnumSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

TemplateSymbol::TemplateSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

TemplateArgumentSymbol::TemplateArgumentSymbol(Scope* enclosingScope,
                                               const Name* name)
    : Symbol(enclosingScope, name) {}

VariableSymbol::VariableSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

FunctionSymbol::FunctionSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

FunctionSymbol::~FunctionSymbol() {}

BlockSymbol* FunctionSymbol::block() const { return block_; }

void FunctionSymbol::setBlock(BlockSymbol* block) { block_ = block; }

ArgumentSymbol::ArgumentSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

BlockSymbol::BlockSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

BlockSymbol::~BlockSymbol() {}

}  // namespace cxx
