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

#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbol_visitor.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/util.h>

#include <cassert>

namespace cxx {

#define PROCESS_SYMBOL_KIND(name) \
  case SymbolKind::k##name:       \
    return #name;

auto symbol_kind_to_string(SymbolKind kind) -> const char* {
  switch (kind) {
    CXX_FOR_EACH_SYMBOL_KIND(PROCESS_SYMBOL_KIND)
    default:
      return nullptr;
  }  // switch
}

#undef PROCESS_SYMBOL_KIND

auto TemplateArgument::make(const Type* type) -> TemplateArgument {
  return TemplateArgument{TemplateArgumentKind::kType, type, 0};
}

auto TemplateArgument::makeLiteral(const Type* type, long value)
    -> TemplateArgument {
  return TemplateArgument{TemplateArgumentKind::kLiteral, type, value};
}

TemplateParameter::TemplateParameter(Control* control,
                                     TemplateParameterKind kind,
                                     const Name* name, const Type* type)
    : kind_(kind), name_(name), type_(type) {}

Symbol::~Symbol() = default;

auto Symbol::findTemplateInstance(
    const std::vector<TemplateArgument>& templ_arguments, Symbol** sym) const
    -> bool {
  for (auto instance : templateInstances_) {
    if (is_same_template_arguments(instance->templateArguments(),
                                   templ_arguments)) {
      *sym = instance;
      return true;
    }
  }

  *sym = nullptr;
  return false;
}

void Symbol::addTemplateInstance(Symbol* instantiatedSymbol) {
  assert(!instantiatedSymbol->templateArguments().empty());

  assert(instantiatedSymbol->primaryTemplate_ == nullptr ||
         instantiatedSymbol->primaryTemplate_ == this);

  instantiatedSymbol->primaryTemplate_ = this;

  templateInstances_.push_back(instantiatedSymbol);
}

auto Symbol::isAnonymous() const -> bool {
  const Identifier* id = name_cast<Identifier>(name_);
  return id && id->isAnonymous();
}

auto Symbol::templateParameters() const
    -> const std::vector<TemplateParameter*>& {
  if (!templateHead_) {
    static std::vector<TemplateParameter*> empty;
    return empty;
  }

  return templateHead_->templateParameters();
}

auto Symbol::isType() const -> bool {
  switch (kind_) {
    case SymbolKind::kTypeAlias:
    case SymbolKind::kClass:
    case SymbolKind::kScopedEnum:
    case SymbolKind::kTemplateParameter:
    case SymbolKind::kTemplateParameterPack:
    case SymbolKind::kInjectedClassName:
      return true;
    default:
      return false;
  }  // switch
}

auto Symbol::isMemberFunction() const -> bool {
  if (!enclosingScope_) {
    return false;
  }

  if (isNot(SymbolKind::kFunction)) {
    return false;
  }

  if (isStatic()) {
    return false;
  }

  auto parentClass = symbol_cast<ClassSymbol>(enclosingScope_->owner());

  if (!parentClass) {
    return false;
  }

  return false;
}

auto Symbol::isGlobalNamespace() const -> bool {
  if (isNot(SymbolKind::kNamespace)) {
    return false;
  }

  return enclosingScope_ == nullptr;
}

auto Symbol::isClassOrNamespace() const -> bool {
  return is(SymbolKind::kClass) || is(SymbolKind::kNamespace);
}

auto Symbol::enclosingClassOrNamespace() const -> Symbol* {
  if (!enclosingScope_) {
    return nullptr;
  }

  if (Scope* scope = enclosingScope_->currentClassOrNamespaceScope()) {
    return scope->owner();
  }

  return nullptr;
}

TypeAliasSymbol::TypeAliasSymbol(Control* control, const Name* name,
                                 const Type* type)
    : SymbolMaker(name, type) {}

ConceptSymbol::ConceptSymbol(Control* control, const Name* name)
    : SymbolMaker(name) {
  setType(control->getConceptType(this));
}

GlobalSymbol::GlobalSymbol(Control* control, const Name* name, const Type* type)
    : SymbolMaker(name, type) {}

FunctionSymbol::FunctionSymbol(Control* control, const Name* name,
                               const Type* type)
    : SymbolMaker(name, type) {
  auto functionType = type_cast<FunctionType>(type);
  assert(functionType->symbol == nullptr);
  const_cast<FunctionType*>(functionType)->symbol = this;
  stackSize_ = 0;
}

auto FunctionSymbol::allocateStack(int size, int alignment) -> int {
  auto offset = align_to(stackSize_, alignment);
  stackSize_ = offset + size;
  return stackSize_;
}

ScopedEnumSymbol::ScopedEnumSymbol(Control* control, const Name* name,
                                   const Type* type)
    : SymbolMaker(name, nullptr) {
  type = control->getScopedEnumType(this, type);
}

EnumeratorSymbol::EnumeratorSymbol(Control* control, const Name* name,
                                   const Type* type, long value)
    : SymbolMaker(name, type), value_(value) {}

ValueSymbol::ValueSymbol(Control* control, const Name* name, const Type* type,
                         long value)
    : SymbolMaker(name, type), value_(value) {}

InjectedClassNameSymbol::InjectedClassNameSymbol(Control* control,
                                                 const Name* name,
                                                 const Type* type)
    : SymbolMaker(name, type) {}

LocalSymbol::LocalSymbol(Control* control, const Name* name, const Type* type)
    : SymbolMaker(name, type) {}

NamespaceSymbol::NamespaceSymbol(Control* control, const Name* name)
    : SymbolMaker(name, nullptr) {
  setType(control->getNamespaceType(this));
}

MemberSymbol::MemberSymbol(Control* control, const Name* name, const Type* type,
                           int offset)
    : SymbolMaker(name, type), offset_(offset) {}

NamespaceAliasSymbol::NamespaceAliasSymbol(Control* control, const Name* name,
                                           Symbol* ns)
    : SymbolMaker(name, ns->type()) {}

ClassSymbol::ClassSymbol(Control* control, const Name* name)
    : SymbolMaker(name, nullptr) {
  setType(control->getClassType(this));
}

auto ClassSymbol::isDerivedFrom(ClassSymbol* symbol) -> bool {
  for (auto baseClass : symbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(baseClass);

    if (!baseClassSymbol) {
      continue;
    }

    if (baseClassSymbol == symbol || baseClassSymbol->isDerivedFrom(symbol)) {
      return true;
    }
  }

  return false;
}

ParameterSymbol::ParameterSymbol(Control* control, const Name* name,
                                 const Type* type, int index)
    : SymbolMaker(name, type), index_(index) {}

TemplateParameterSymbol::TemplateParameterSymbol(Control* control,
                                                 const Name* name, int index)
    : SymbolMaker(name, nullptr), index_(index) {
  setType(control->getGenericType(this));
}

TemplateParameterPackSymbol::TemplateParameterPackSymbol(Control* control,
                                                         const Name* name,
                                                         int index)
    : SymbolMaker(name, nullptr), index_(index) {
  setType(control->getPackType(this));
}

NonTypeTemplateParameterSymbol::NonTypeTemplateParameterSymbol(Control* control,
                                                               const Name* name,
                                                               const Type* type,
                                                               int index)
    : SymbolMaker(name, type), index_(index) {}

void TemplateHead::addTemplateParameter(TemplateParameter* templateParameter) {
  templateParameters_.push_back(templateParameter);
}

DependentSymbol::DependentSymbol(Control* control) : SymbolMaker(nullptr) {
  setType(control->getDependentType(this));
}

#define DECLARE_VISITOR(name) \
  void name##Symbol::accept(SymbolVisitor* visitor) { visitor->visit(this); }

CXX_FOR_EACH_SYMBOL_KIND(DECLARE_VISITOR)

#undef DECLARE_VISITOR

}  // namespace cxx
