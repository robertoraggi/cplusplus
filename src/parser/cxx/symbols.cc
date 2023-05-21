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
#include <cxx/qualified_type.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <list>

namespace cxx {

Symbol::Symbol(Scope* enclosingScope, const Name* name)
    : enclosingScope_(enclosingScope), name_(name) {}

Symbol::~Symbol() = default;

void Symbol::addToEnclosingScope() { enclosingScope_->add(this); }

auto Symbol::unqualifiedId() const -> std::string {
  if (name()) return fmt::format("{}", *name());
  return "__anon__";
}

auto Symbol::qualifiedId() const -> std::string {
  if (enclosingScope_) {
    auto parent = enclosingScope_->owner();

    if (!enclosingScope_->enclosingScope() && !parent->name()) {
      return unqualifiedId();
    }

    return parent->qualifiedId() + "::" + unqualifiedId();
  }

  return unqualifiedId();
}

auto Symbol::isTypeSymbol() const -> bool { return false; }

auto Symbol::index() const -> int {
  if (!enclosingScope_) return -1;

  auto it = std::find(enclosingScope_->begin(), enclosingScope_->end(), this);

  return it != enclosingScope_->end()
             ? int(std::distance(enclosingScope_->begin(), it))
             : -1;
}

TypeSymbol::TypeSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

auto TypeSymbol::isTypeSymbol() const -> bool { return true; }

auto Symbol::enclosingClassOrNamespace() const -> Symbol* {
  for (auto scope = enclosingScope_; scope; scope = scope->enclosingScope()) {
    if (auto sym = dynamic_cast<NamespaceSymbol*>(scope->owner())) return sym;
    if (auto sym = dynamic_cast<ClassSymbol*>(scope->owner())) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingNamespace() const -> NamespaceSymbol* {
  for (auto scope = enclosingScope_; scope; scope = scope->enclosingScope()) {
    if (auto sym = dynamic_cast<NamespaceSymbol*>(scope->owner())) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingClass() const -> ClassSymbol* {
  for (auto scope = enclosingScope_; scope; scope = scope->enclosingScope()) {
    if (auto sym = dynamic_cast<ClassSymbol*>(scope->owner())) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingFunction() const -> FunctionSymbol* {
  for (auto scope = enclosingScope_; scope; scope = scope->enclosingScope()) {
    if (auto sym = dynamic_cast<FunctionSymbol*>(scope->owner())) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingBlock() const -> BlockSymbol* {
  for (auto scope = enclosingScope_; scope; scope = scope->enclosingScope()) {
    if (auto sym = dynamic_cast<BlockSymbol*>(scope->owner())) return sym;
  }
  return nullptr;
}

auto Symbol::type() const -> const QualifiedType& { return type_; }

void Symbol::setType(const QualifiedType& type) { type_ = type; }

auto Symbol::linkage() const -> Linkage { return linkage_; }

void Symbol::setLinkage(Linkage linkage) { linkage_ = linkage; }

auto Symbol::visibility() const -> Visibility { return visibility_; }

void Symbol::setVisibility(Visibility visibility) { visibility_ = visibility; }

auto Symbol::templateParameterList() const -> TemplateParameterList* {
  return templateParameterList_;
}

void Symbol::setTemplateParameterList(
    TemplateParameterList* templateParameterList) {
  templateParameterList_ = templateParameterList;
}

NamespaceSymbol::NamespaceSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

NamespaceSymbol::~NamespaceSymbol() = default;

void NamespaceSymbol::addUsingNamespace(NamespaceSymbol* symbol) {
  usingNamespaces_.push_back(symbol);
}

ClassSymbol::ClassSymbol(Scope* enclosingScope, const Name* name)
    : TypeSymbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

ClassSymbol::~ClassSymbol() = default;

void ClassSymbol::addBaseClass(ClassSymbol* baseClass) {
  baseClasses_.push_back(baseClass);
}

ConceptSymbol::ConceptSymbol(Scope* enclosingScope, const Name* name)
    : TypeSymbol(enclosingScope, name) {}

TypedefSymbol::TypedefSymbol(Scope* enclosingScope, const Name* name)
    : TypeSymbol(enclosingScope, name) {}

EnumSymbol::EnumSymbol(Scope* enclosingScope, const Name* name)
    : TypeSymbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

EnumSymbol::~EnumSymbol() = default;

ScopedEnumSymbol::ScopedEnumSymbol(Scope* enclosingScope, const Name* name)
    : TypeSymbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

ScopedEnumSymbol::~ScopedEnumSymbol() = default;

EnumeratorSymbol::EnumeratorSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

TemplateParameterList::TemplateParameterList(Scope* enclosingScope)
    : Symbol(enclosingScope, nullptr), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

TemplateParameterList::~TemplateParameterList() = default;

TemplateTypeParameterSymbol::TemplateTypeParameterSymbol(Scope* enclosingScope,
                                                         const Name* name)
    : TypeSymbol(enclosingScope, name) {}

VariableSymbol::VariableSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

FieldSymbol::FieldSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

FunctionSymbol::FunctionSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

FunctionSymbol::~FunctionSymbol() = default;

auto FunctionSymbol::block() const -> BlockSymbol* { return block_; }

void FunctionSymbol::setBlock(BlockSymbol* block) { block_ = block; }

ArgumentSymbol::ArgumentSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name) {}

BlockSymbol::BlockSymbol(Scope* enclosingScope, const Name* name)
    : Symbol(enclosingScope, name), scope_(std::make_unique<Scope>()) {
  scope_->setOwner(this);
}

BlockSymbol::~BlockSymbol() = default;

}  // namespace cxx
