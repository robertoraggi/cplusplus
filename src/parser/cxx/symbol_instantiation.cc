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

#include <cxx/symbol_instantiation.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

struct SymbolInstantiation::MakeSymbol {
  SymbolInstantiation& self;

  auto operator()(SymbolKind kind) -> Symbol* {
#define MAKE_SYMBOL(S)   \
  case SymbolKind::k##S: \
    return self.control()->new##S##Symbol(nullptr);

    switch (kind) {
      CXX_FOR_EACH_SYMBOL(MAKE_SYMBOL)
      default:
        return nullptr;
    }  // switch
  }
};

struct SymbolInstantiation::VisitSymbol {
  SymbolInstantiation& self;

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(ConceptSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(ClassSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(EnumSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(ScopedEnumSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(FunctionSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(VariableSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(FieldSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(ParameterSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(EnumeratorSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(FunctionParametersSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(TemplateParametersSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(BlockSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(LambdaSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(TypeParameterSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(NonTypeParameterSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(TemplateTypeParameterSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(ConstraintTypeParameterSymbol* symbol)
      -> Symbol*;
  [[nodiscard]] auto operator()(OverloadSetSymbol* symbol) -> Symbol*;
  [[nodiscard]] auto operator()(BaseClassSymbol* symbol) -> Symbol*;
};

struct SymbolInstantiation::VisitType {
  SymbolInstantiation& self;

  [[nodiscard]] auto substitute(const Type* type) -> const Type* {
    if (!type) return nullptr;
    return visit(*this, type);
  }

  [[nodiscard]] auto operator()(const BuiltinVaListType* type) -> const Type*;
  [[nodiscard]] auto operator()(const VoidType* type) -> const Type*;
  [[nodiscard]] auto operator()(const NullptrType* type) -> const Type*;
  [[nodiscard]] auto operator()(const DecltypeAutoType* type) -> const Type*;
  [[nodiscard]] auto operator()(const AutoType* type) -> const Type*;
  [[nodiscard]] auto operator()(const BoolType* type) -> const Type*;
  [[nodiscard]] auto operator()(const SignedCharType* type) -> const Type*;
  [[nodiscard]] auto operator()(const ShortIntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const IntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const LongIntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const LongLongIntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnsignedCharType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnsignedShortIntType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const UnsignedIntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnsignedLongIntType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnsignedLongLongIntType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const CharType* type) -> const Type*;
  [[nodiscard]] auto operator()(const Char8Type* type) -> const Type*;
  [[nodiscard]] auto operator()(const Char16Type* type) -> const Type*;
  [[nodiscard]] auto operator()(const Char32Type* type) -> const Type*;
  [[nodiscard]] auto operator()(const WideCharType* type) -> const Type*;
  [[nodiscard]] auto operator()(const FloatType* type) -> const Type*;
  [[nodiscard]] auto operator()(const DoubleType* type) -> const Type*;
  [[nodiscard]] auto operator()(const LongDoubleType* type) -> const Type*;
  [[nodiscard]] auto operator()(const QualType* type) -> const Type*;
  [[nodiscard]] auto operator()(const BoundedArrayType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnboundedArrayType* type) -> const Type*;
  [[nodiscard]] auto operator()(const PointerType* type) -> const Type*;
  [[nodiscard]] auto operator()(const LvalueReferenceType* type) -> const Type*;
  [[nodiscard]] auto operator()(const RvalueReferenceType* type) -> const Type*;
  [[nodiscard]] auto operator()(const FunctionType* type) -> const Type*;
  [[nodiscard]] auto operator()(const ClassType* type) -> const Type*;
  [[nodiscard]] auto operator()(const EnumType* type) -> const Type*;
  [[nodiscard]] auto operator()(const ScopedEnumType* type) -> const Type*;
  [[nodiscard]] auto operator()(const MemberObjectPointerType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const MemberFunctionPointerType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const NamespaceType* type) -> const Type*;
  [[nodiscard]] auto operator()(const TypeParameterType* type) -> const Type*;
  [[nodiscard]] auto operator()(const TemplateTypeParameterType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const UnresolvedNameType* type) -> const Type*;
  [[nodiscard]] auto operator()(const UnresolvedBoundedArrayType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const UnresolvedUnderlyingType* type)
      -> const Type*;
  [[nodiscard]] auto operator()(const OverloadSetType* type) -> const Type*;
};

SymbolInstantiation::SymbolInstantiation(
    TranslationUnit* unit, const std::vector<TemplateArgument>& arguments)
    : unit_(unit), arguments_(arguments) {}

SymbolInstantiation::~SymbolInstantiation() {}

auto SymbolInstantiation::control() const -> Control* {
  return unit_->control();
}

auto SymbolInstantiation::operator()(Symbol* symbol) -> Symbol* {
  if (!symbol) return symbol;

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (auto specialization = classSymbol->findSpecialization(arguments_)) {
      return specialization;
    }

    std::swap(current_, symbol);
    auto specialization = replacement(classSymbol);
    std::swap(current_, symbol);

    if (specialization == classSymbol) {
      cxx_runtime_error("cannot specialize itself");
    }

    classSymbol->addSpecialization(arguments_, specialization);
  }

  std::swap(current_, symbol);
  auto instantiation = visit(VisitSymbol{*this}, current_);
  std::swap(current_, symbol);

  return instantiation;
}

auto SymbolInstantiation::findOrCreateReplacement(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;

  if (symbol != current_ && !symbol->hasEnclosingSymbol(current_))
    return symbol;

  auto it = replacements_.find(symbol);
  if (it != replacements_.end()) return it->second;

  auto newSymbol = MakeSymbol{*this}(symbol->kind());
  replacements_[symbol] = newSymbol;

  auto enclosingSymbol = replacement(symbol->enclosingSymbol());
  newSymbol->setEnclosingScope(enclosingSymbol->scope());
  if (symbol->type()) {
    newSymbol->setType(visit(VisitType{*this}, symbol->type()));
  }

  newSymbol->setName(symbol->name());

  return newSymbol;
}

auto SymbolInstantiation::instantiateHelper(Symbol* symbol) -> Symbol* {
  if (!symbol) return symbol;
  return visit(VisitSymbol{*this}, symbol);
}

auto SymbolInstantiation::VisitSymbol::operator()(NamespaceSymbol* symbol)
    -> Symbol* {
  cxx_runtime_error("NamespaceSymbol cannot be instantiated.");
  return nullptr;
}

auto SymbolInstantiation::VisitSymbol::operator()(ConceptSymbol* symbol)
    -> Symbol* {
  cxx_runtime_error("ConceptSymbol cannot be instantiated.");
  return nullptr;
}

auto SymbolInstantiation::VisitSymbol::operator()(ClassSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  newSymbol->setFlags(symbol->flags());

  if (symbol != self.current_) {
    newSymbol->setTemplateParameters(
        self.instantiate(symbol->templateParameters()));
  }

  for (auto baseClass : symbol->baseClasses()) {
    auto newBaseClass = self.instantiate(baseClass);
    newSymbol->addBaseClass(newBaseClass);
  }
  for (auto ctor : symbol->constructors()) {
    auto newCtor = self.instantiate(ctor);
    newSymbol->addConstructor(newCtor);
  }
  for (auto member : symbol->members()) {
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(EnumSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->members()) {
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(ScopedEnumSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->members()) {
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(FunctionSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->members()) {
    if (member->isBlock()) continue;
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(TypeAliasSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(VariableSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(FieldSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(ParameterSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(EnumeratorSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(
    FunctionParametersSymbol* symbol) -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->members()) {
    if (member->isBlock()) continue;
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(
    TemplateParametersSymbol* symbol) -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->members()) {
    auto newMember = self.instantiate(member);
    newSymbol->addMember(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(BlockSymbol* symbol)
    -> Symbol* {
  cxx_runtime_error("BlockSymbol cannot be instantiated.");
  return nullptr;
}

auto SymbolInstantiation::VisitSymbol::operator()(LambdaSymbol* symbol)
    -> Symbol* {
  cxx_runtime_error("LambdaSymbol cannot be instantiated.");
  return nullptr;
}

auto SymbolInstantiation::VisitSymbol::operator()(TypeParameterSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  newSymbol->setIndex(symbol->index());
  newSymbol->setDepth(symbol->depth());
  newSymbol->setParameterPack(symbol->isParameterPack());
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(
    NonTypeParameterSymbol* symbol) -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(
    TemplateTypeParameterSymbol* symbol) -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(
    ConstraintTypeParameterSymbol* symbol) -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(OverloadSetSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  for (auto member : symbol->functions()) {
    auto newMember = self.instantiate(member);
    newSymbol->addFunction(newMember);
  }
  return newSymbol;
}

auto SymbolInstantiation::VisitSymbol::operator()(BaseClassSymbol* symbol)
    -> Symbol* {
  auto newSymbol = self.replacement(symbol);
  return newSymbol;
}

// types

auto SymbolInstantiation::VisitType::operator()(const BuiltinVaListType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const VoidType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const NullptrType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const DecltypeAutoType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const AutoType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const BoolType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const SignedCharType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const ShortIntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const IntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const LongIntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const LongLongIntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const UnsignedCharType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(
    const UnsignedShortIntType* type) -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const UnsignedIntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const UnsignedLongIntType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(
    const UnsignedLongLongIntType* type) -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const CharType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const Char8Type* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const Char16Type* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const Char32Type* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const WideCharType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const FloatType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const DoubleType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const LongDoubleType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const QualType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getQualType(elementType, type->cvQualifiers());
}

auto SymbolInstantiation::VisitType::operator()(const BoundedArrayType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getBoundedArrayType(elementType, type->size());
}

auto SymbolInstantiation::VisitType::operator()(const UnboundedArrayType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getUnboundedArrayType(elementType);
}

auto SymbolInstantiation::VisitType::operator()(const PointerType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getPointerType(elementType);
}

auto SymbolInstantiation::VisitType::operator()(const LvalueReferenceType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getLvalueReferenceType(elementType);
}

auto SymbolInstantiation::VisitType::operator()(const RvalueReferenceType* type)
    -> const Type* {
  auto elementType = substitute(type->elementType());
  return self.control()->getRvalueReferenceType(elementType);
}

auto SymbolInstantiation::VisitType::operator()(const FunctionType* type)
    -> const Type* {
  auto returnType = substitute(type->returnType());

  std::vector<const Type*> parameterTypes;
  for (const auto& parameterType : type->parameterTypes()) {
    parameterTypes.push_back(substitute(parameterType));
  }
  return self.control()->getFunctionType(
      returnType, std::move(parameterTypes), type->isVariadic(),
      type->cvQualifiers(), type->refQualifier(), type->isNoexcept());
}

auto SymbolInstantiation::VisitType::operator()(const ClassType* type)
    -> const Type* {
  return self.control()->getClassType(self.replacement(type->symbol()));
}

auto SymbolInstantiation::VisitType::operator()(const EnumType* type)
    -> const Type* {
  return self.replacement(type->symbol())->type();
}

auto SymbolInstantiation::VisitType::operator()(const ScopedEnumType* type)
    -> const Type* {
  return self.replacement(type->symbol())->type();
}

auto SymbolInstantiation::VisitType::operator()(
    const MemberObjectPointerType* type) -> const Type* {
  cxx_runtime_error("todo: substitute MemberObjectPointerType");
}

auto SymbolInstantiation::VisitType::operator()(
    const MemberFunctionPointerType* type) -> const Type* {
  cxx_runtime_error("todo: substitute MemberFunctionPointerType");
}

auto SymbolInstantiation::VisitType::operator()(const NamespaceType* type)
    -> const Type* {
  return type;
}

auto SymbolInstantiation::VisitType::operator()(const TypeParameterType* type)
    -> const Type* {
  auto templateParameters = getTemplateParameters(self.current_);

  if (type->symbol()->enclosingSymbol() != templateParameters) {
    return self.replacement(type->symbol())->type();
  }

  auto index = type->symbol()->index();

  if (index >= self.arguments_.size()) {
    cxx_runtime_error("type parameter index out of range");
  }

  auto arg = self.arguments_[index];
  return std::get<const Type*>(arg);
}

auto SymbolInstantiation::VisitType::operator()(
    const TemplateTypeParameterType* type) -> const Type* {
  cxx_runtime_error("todo: substitute TemplateTypeParameterType");
}

auto SymbolInstantiation::VisitType::operator()(const UnresolvedNameType* type)
    -> const Type* {
  if (auto templateId = ast_cast<SimpleTemplateIdAST>(type->unqualifiedId())) {
    std::vector<TemplateArgument> args;
    for (auto it = templateId->templateArgumentList; it; it = it->next) {
      if (auto arg = ast_cast<TypeTemplateArgumentAST>(it->value)) {
        args.push_back(substitute(arg->typeId->type));
      }
    }
    auto symbol = self.control()->instantiate(type->translationUnit(),
                                              templateId->primaryTemplateSymbol,
                                              std::move(args));
    return symbol->type();
  }
  cxx_runtime_error("todo: substitute UnresolvedNameType");
}

auto SymbolInstantiation::VisitType::operator()(
    const UnresolvedBoundedArrayType* type) -> const Type* {
  cxx_runtime_error("todo: substitute UnresolvedBoundedArrayType");
}

auto SymbolInstantiation::VisitType::operator()(
    const UnresolvedUnderlyingType* type) -> const Type* {
  cxx_runtime_error("todo: substitute UnresolvedUnderlyingType");
}

auto SymbolInstantiation::VisitType::operator()(const OverloadSetType* type)
    -> const Type* {
  return self.control()->getOverloadSetType(self.replacement(type->symbol()));
}

}  // namespace cxx