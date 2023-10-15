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
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cassert>
#include <cstdlib>
#include <forward_list>
#include <unordered_set>

namespace cxx {

namespace {

template <typename Literal>
struct LiteralHash {
  auto operator()(const Literal& literal) const -> std::size_t {
    return std::hash<std::string_view>{}(literal.value());
  }
};

template <typename Literal>
struct LiteralEqualTo {
  auto operator()(const Literal& lhs, const Literal& rhs) const -> bool {
    return lhs.value() == rhs.value();
  }
};

template <typename Literal>
using LiteralSet =
    std::unordered_set<Literal, LiteralHash<Literal>, LiteralEqualTo<Literal>>;

struct NameHash {
  auto operator()(const Identifier& id) const -> std::size_t {
    return std::hash<std::string_view>{}(id.name());
  }

  auto operator()(const OperatorId& id) const -> std::size_t {
    return std::hash<std::string_view>{}(id.name());
  }

  auto operator()(const DestructorId& id) const -> std::size_t {
    return std::hash<std::string_view>{}(id.name());
  }
};

struct NameEqualTo {
  auto operator()(const Identifier& lhs, const Identifier& rhs) const -> bool {
    return lhs.name() == rhs.name();
  }

  auto operator()(const OperatorId& lhs, const OperatorId& rhs) const -> bool {
    return lhs.name() == rhs.name();
  }

  auto operator()(const DestructorId& lhs, const DestructorId& rhs) const
      -> bool {
    return lhs.name() == rhs.name();
  }
};

template <typename Name>
using NameSet = std::unordered_set<Name, NameHash, NameEqualTo>;

struct TypeHash {
  template <typename T>
  auto operator()(const T& type) const -> std::size_t {
    return 0;
  }
};

struct TypeEqualTo {
  template <typename T>
  auto operator()(const T& left, const T& right) const -> bool {
    return left.equalTo(&right);
  }
};

template <typename T>
using TypeSet = std::unordered_set<T, TypeHash, TypeEqualTo>;

}  // namespace

struct Control::Private {
  MemoryLayout* memoryLayout = nullptr;
  LiteralSet<IntegerLiteral> integerLiterals;
  LiteralSet<FloatLiteral> floatLiterals;
  LiteralSet<StringLiteral> stringLiterals;
  LiteralSet<CharLiteral> charLiterals;
  LiteralSet<WideStringLiteral> wideStringLiterals;
  LiteralSet<Utf8StringLiteral> utf8StringLiterals;
  LiteralSet<Utf16StringLiteral> utf16StringLiterals;
  LiteralSet<Utf32StringLiteral> utf32StringLiterals;
  LiteralSet<CommentLiteral> commentLiterals;

  NameSet<Identifier> identifiers;
  NameSet<OperatorId> operatorIds;
  NameSet<DestructorId> destructorIds;

  std::forward_list<ClassSymbol> classSymbols;
  std::forward_list<ConceptSymbol> conceptSymbols;
  std::forward_list<DependentSymbol> dependentSymbols;
  std::forward_list<EnumeratorSymbol> enumeratorSymbols;
  std::forward_list<FunctionSymbol> functionSymbols;
  std::forward_list<GlobalSymbol> globalSymbols;
  std::forward_list<InjectedClassNameSymbol> injectedClassNameSymbols;
  std::forward_list<LocalSymbol> localSymbols;
  std::forward_list<MemberSymbol> memberSymbols;
  std::forward_list<NamespaceSymbol> namespaceSymbols;
  std::forward_list<NamespaceAliasSymbol> namespaceAliasSymbols;
  std::forward_list<NonTypeTemplateParameterSymbol>
      nonTypeTemplateParameterSymbols;
  std::forward_list<ParameterSymbol> parameterSymbols;
  std::forward_list<ScopedEnumSymbol> scopedEnumSymbols;
  std::forward_list<TemplateParameterSymbol> templateParameterSymbols;
  std::forward_list<TemplateParameterPackSymbol> templateParameterPackSymbols;
  std::forward_list<TypeAliasSymbol> typeAliasSymbols;
  std::forward_list<ValueSymbol> valueSymbols;

  std::forward_list<TemplateParameter> templateParameters;

  InvalidType invalidType;
  NullptrType nullptrType;
  DecltypeAutoType decltypeAutoType;
  AutoType autoType;
  VoidType voidType;
  BoolType boolType;
  CharType charType;
  SignedCharType signedCharType;
  UnsignedCharType unsignedCharType;
  Char8Type char8Type;
  Char16Type char16Type;
  Char32Type char32Type;
  WideCharType wideCharType;
  ShortType shortType;
  UnsignedShortType unsignedShortType;
  IntType intType;
  UnsignedIntType unsignedIntType;
  LongType longType;
  UnsignedLongType unsignedLongType;
  FloatType floatType;
  DoubleType doubleType;

  TypeSet<DependentType> dependentTypes;
  TypeSet<QualType> qualTypes;
  TypeSet<PointerType> pointerTypes;
  TypeSet<LValueReferenceType> lValueReferenceTypes;
  TypeSet<RValueReferenceType> rValueReferenceTypes;
  TypeSet<ArrayType> arrayTypes;
  TypeSet<FunctionType> functionTypes;
  TypeSet<ClassType> classTypes;
  TypeSet<NamespaceType> namespaceTypes;
  TypeSet<MemberPointerType> memberPointerTypes;
  TypeSet<ConceptType> conceptTypes;
  TypeSet<EnumType> enumTypes;
  TypeSet<GenericType> genericTypes;
  TypeSet<PackType> packTypes;
  TypeSet<ScopedEnumType> scopedEnumTypes;

  int anonymousIdCount = 0;
};

Control::Control() : d(std::make_unique<Private>()) {}

Control::~Control() = default;

auto Control::integerLiteral(std::string_view spelling)
    -> const IntegerLiteral* {
  auto [it, inserted] = d->integerLiterals.emplace(std::string(spelling));
  if (inserted) it->initialize();
  return &*it;
}

auto Control::floatLiteral(std::string_view spelling) -> const FloatLiteral* {
  return &*d->floatLiterals.emplace(std::string(spelling)).first;
}

auto Control::stringLiteral(std::string_view spelling) -> const StringLiteral* {
  return &*d->stringLiterals.emplace(std::string(spelling)).first;
}

auto Control::charLiteral(std::string_view spelling) -> const CharLiteral* {
  return &*d->charLiterals.emplace(std::string(spelling)).first;
}

auto Control::wideStringLiteral(std::string_view spelling)
    -> const WideStringLiteral* {
  return &*d->wideStringLiterals.emplace(std::string(spelling)).first;
}

auto Control::utf8StringLiteral(std::string_view spelling)
    -> const Utf8StringLiteral* {
  return &*d->utf8StringLiterals.emplace(std::string(spelling)).first;
}

auto Control::utf16StringLiteral(std::string_view spelling)
    -> const Utf16StringLiteral* {
  return &*d->utf16StringLiterals.emplace(std::string(spelling)).first;
}

auto Control::utf32StringLiteral(std::string_view spelling)
    -> const Utf32StringLiteral* {
  return &*d->utf32StringLiterals.emplace(std::string(spelling)).first;
}

auto Control::commentLiteral(std::string_view spelling)
    -> const CommentLiteral* {
  return &*d->commentLiterals.emplace(std::string(spelling)).first;
}

auto Control::memoryLayout() const -> MemoryLayout* { return d->memoryLayout; }

auto Control::makeAnonymousId(std::string_view base) -> const Identifier* {
  auto id = std::string("$") + std::string(base) +
            std::to_string(++d->anonymousIdCount);
  return getIdentifier(id.c_str());
}

auto Control::getIdentifier(std::string_view name) -> const Identifier* {
  return &*d->identifiers.emplace(std::string(name)).first;
}

auto Control::getOperatorId(std::string_view name) -> const OperatorId* {
  return &*d->operatorIds.emplace(std::string(name)).first;
}

auto Control::getDestructorId(std::string_view name) -> const DestructorId* {
  return &*d->destructorIds.emplace(std::string(name)).first;
}

auto Control::makeTypeParameter(const Name* name) -> TemplateParameter* {
  return &d->templateParameters.emplace_front(
      this, TemplateParameterKind::kType, name, nullptr);
}

auto Control::makeTypeParameterPack(const Name* name) -> TemplateParameter* {
  return &d->templateParameters.emplace_front(
      this, TemplateParameterKind::kPack, name, nullptr);
}

auto Control::makeNonTypeParameter(const Type* type, const Name* name)
    -> TemplateParameter* {
  return &d->templateParameters.emplace_front(
      this, TemplateParameterKind::kNonType, name, type);
}

auto Control::makeParameterSymbol(const Name* name, const Type* type, int index)
    -> ParameterSymbol* {
  return &d->parameterSymbols.emplace_front(this, name, type, index);
}

auto Control::makeClassSymbol(const Name* name) -> ClassSymbol* {
  return &d->classSymbols.emplace_front(this, name);
}

auto Control::makeEnumeratorSymbol(const Name* name, const Type* type, long val)
    -> EnumeratorSymbol* {
  return &d->enumeratorSymbols.emplace_front(this, name, type, val);
}

auto Control::makeFunctionSymbol(const Name* name, const Type* type)
    -> FunctionSymbol* {
  return &d->functionSymbols.emplace_front(this, name, type);
}

auto Control::makeGlobalSymbol(const Name* name, const Type* type)
    -> GlobalSymbol* {
  return &d->globalSymbols.emplace_front(this, name, type);
}

auto Control::makeInjectedClassNameSymbol(const Name* name, const Type* type)
    -> InjectedClassNameSymbol* {
  return &d->injectedClassNameSymbols.emplace_front(this, name, type);
}

auto Control::makeLocalSymbol(const Name* name, const Type* type)
    -> LocalSymbol* {
  return &d->localSymbols.emplace_front(this, name, type);
}

auto Control::makeMemberSymbol(const Name* name, const Type* type, int offset)
    -> MemberSymbol* {
  return &d->memberSymbols.emplace_front(this, name, type, offset);
}

auto Control::makeNamespaceSymbol(const Name* name) -> NamespaceSymbol* {
  return &d->namespaceSymbols.emplace_front(this, name);
}

auto Control::makeNamespaceAliasSymbol(const Name* name, Symbol* ns)
    -> NamespaceAliasSymbol* {
  return &d->namespaceAliasSymbols.emplace_front(this, name, ns);
}

auto Control::makeNonTypeTemplateParameterSymbol(const Name* name,
                                                 const Type* type, int index)
    -> NonTypeTemplateParameterSymbol* {
  return &d->nonTypeTemplateParameterSymbols.emplace_front(this, name, type,
                                                           index);
}

auto Control::makeScopedEnumSymbol(const Name* name, const Type* type)
    -> ScopedEnumSymbol* {
  return &d->scopedEnumSymbols.emplace_front(this, name, type);
}

auto Control::makeTemplateParameterPackSymbol(const Name* name, int index)
    -> TemplateParameterPackSymbol* {
  return &d->templateParameterPackSymbols.emplace_front(this, name, index);
}

auto Control::makeTemplateParameterSymbol(const Name* name, int index)
    -> TemplateParameterSymbol* {
  return &d->templateParameterSymbols.emplace_front(this, name, index);
}

auto Control::makeConceptSymbol(const Name* name) -> ConceptSymbol* {
  return &d->conceptSymbols.emplace_front(this, name);
}

auto Control::makeTypeAliasSymbol(const Name* name, const Type* type)
    -> TypeAliasSymbol* {
  return &d->typeAliasSymbols.emplace_front(this, name, type);
}

auto Control::makeValueSymbol(const Name* name, const Type* type, long val)
    -> ValueSymbol* {
  return &d->valueSymbols.emplace_front(this, name, type, val);
}

auto Control::getInvalidType() -> const InvalidType* { return &d->invalidType; }

auto Control::getNullptrType() -> const NullptrType* { return &d->nullptrType; }

auto Control::getDecltypeAutoType() -> const DecltypeAutoType* {
  return &d->decltypeAutoType;
}

auto Control::getAutoType() -> const AutoType* { return &d->autoType; }

auto Control::getVoidType() -> const VoidType* { return &d->voidType; }

auto Control::getBoolType() -> const BoolType* { return &d->boolType; }

auto Control::getCharType() -> const CharType* { return &d->charType; }

auto Control::getSignedCharType() -> const SignedCharType* {
  return &d->signedCharType;
}

auto Control::getUnsignedCharType() -> const UnsignedCharType* {
  return &d->unsignedCharType;
}

auto Control::getChar8Type() -> const Char8Type* { return &d->char8Type; }

auto Control::getChar16Type() -> const Char16Type* { return &d->char16Type; }

auto Control::getChar32Type() -> const Char32Type* { return &d->char32Type; }

auto Control::getWideCharType() -> const WideCharType* {
  return &d->wideCharType;
}

auto Control::getShortType() -> const ShortType* { return &d->shortType; }

auto Control::getUnsignedShortType() -> const UnsignedShortType* {
  return &d->unsignedShortType;
}

auto Control::getIntType() -> const IntType* { return &d->intType; }

auto Control::getUnsignedIntType() -> const UnsignedIntType* {
  return &d->unsignedIntType;
}

auto Control::getLongType() -> const LongType* { return &d->longType; }

auto Control::getUnsignedLongType() -> const UnsignedLongType* {
  return &d->unsignedLongType;
}

auto Control::getFloatType() -> const FloatType* { return &d->floatType; }

auto Control::getDoubleType() -> const DoubleType* { return &d->doubleType; }

auto Control::getQualType(const Type* elementType, bool isConst,
                          bool isVolatile) -> const QualType* {
  return &*d->qualTypes.emplace(this, elementType, isConst, isVolatile).first;
}

auto Control::getPointerType(const Type* elementType) -> const PointerType* {
  return &*d->pointerTypes.emplace(this, elementType).first;
}

auto Control::getLValueReferenceType(const Type* elementType)
    -> const LValueReferenceType* {
  return &*d->lValueReferenceTypes.emplace(this, elementType).first;
}

auto Control::getRValueReferenceType(const Type* elementType)
    -> const RValueReferenceType* {
  return &*d->rValueReferenceTypes.emplace(this, elementType).first;
}

auto Control::getArrayType(const Type* elementType, int dimension)
    -> const ArrayType* {
  return &*d->arrayTypes.emplace(this, elementType, dimension).first;
}

auto Control::getFunctionType(const Type* returnType,
                              std::vector<Parameter> parameters,
                              bool isVariadic) -> const FunctionType* {
  return &*d->functionTypes
               .emplace(this, nullptr, returnType, std::move(parameters),
                        isVariadic)
               .first;
}

auto Control::getClassType(ClassSymbol* classSymbol) -> const ClassType* {
  return &*d->classTypes.emplace(this, classSymbol).first;
}

auto Control::getNamespaceType(NamespaceSymbol* namespaceSymbol)
    -> const NamespaceType* {
  return &*d->namespaceTypes.emplace(this, namespaceSymbol).first;
}

auto Control::getMemberPointerType(const Type* classType,
                                   const Type* memberType)
    -> const MemberPointerType* {
  return &*d->memberPointerTypes.emplace(this, classType, memberType).first;
}

auto Control::getConceptType(Symbol* symbol) -> const ConceptType* {
  return &*d->conceptTypes.emplace(this, symbol).first;
}

auto Control::getEnumType(Symbol* symbol) -> const EnumType* {
  return &*d->enumTypes.emplace(this, symbol).first;
}

auto Control::getGenericType(Symbol* symbol) -> const GenericType* {
  return &*d->genericTypes.emplace(this, symbol).first;
}

auto Control::getPackType(Symbol* symbol) -> const PackType* {
  return &*d->packTypes.emplace(this, symbol).first;
}

auto Control::getScopedEnumType(ScopedEnumSymbol* symbol,
                                const Type* elementType)
    -> const ScopedEnumType* {
  return &*d->scopedEnumTypes.emplace(this, symbol, elementType).first;
}

auto Control::getConstType(const Type* type) -> const QualType* {
  return getQualType(type, true, false);
}

auto Control::getVolatileType(const Type* type) -> const QualType* {
  return getQualType(type, false, true);
}

auto Control::getConstVolatileType(const Type* type) -> const QualType* {
  return getQualType(type, true, true);
}

auto Control::getDependentType(DependentSymbol* symbol)
    -> const DependentType* {
  return &*d->dependentTypes.emplace(this, symbol).first;
}

}  // namespace cxx