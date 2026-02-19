// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <cstdlib>
#include <forward_list>
#include <set>
#include <unordered_set>

namespace cxx {

namespace {

template <typename Literal>
struct LiteralHash {
  using is_transparent = void;
  auto operator()(const Literal& literal) const -> std::size_t {
    return std::hash<std::string_view>{}(literal.value());
  }
  auto operator()(std::string_view sv) const -> std::size_t {
    return std::hash<std::string_view>{}(sv);
  }
};

template <typename Literal>
struct LiteralEqualTo {
  using is_transparent = void;
  auto operator()(const Literal& lhs, const Literal& rhs) const -> bool {
    return lhs.value() == rhs.value();
  }
  auto operator()(const Literal& lhs, std::string_view rhs) const -> bool {
    return lhs.value() == rhs;
  }
  auto operator()(std::string_view lhs, const Literal& rhs) const -> bool {
    return lhs == rhs.value();
  }
};

template <typename Literal>
using LiteralSet =
    std::unordered_set<Literal, LiteralHash<Literal>, LiteralEqualTo<Literal>>;

}  // namespace

struct Control::Private {
  explicit Private(Control* control) : traits(control) {}

  TypeTraits traits;

  MemoryLayout* memoryLayout = nullptr;
  LiteralSet<IntegerLiteral> integerLiterals;
  LiteralSet<FloatLiteral> floatLiterals;
  LiteralSet<StringLiteral> stringLiterals;
  LiteralSet<CharLiteral> charLiterals;
  LiteralSet<StringLiteral> wideStringLiterals;
  LiteralSet<StringLiteral> utf8StringLiterals;
  LiteralSet<StringLiteral> utf16StringLiterals;
  LiteralSet<StringLiteral> utf32StringLiterals;
  LiteralSet<CommentLiteral> commentLiterals;

  std::unordered_set<Identifier> identifiers;
  std::unordered_set<OperatorId> operatorIds;
  std::unordered_set<DestructorId> destructorIds;
  std::unordered_set<LiteralOperatorId> literalOperatorIds;
  std::unordered_set<ConversionFunctionId> conversionFunctionIds;
  std::unordered_set<TemplateId> templateIds;

  BuiltinVaListType builtinVaListType;
  BuiltinMetaInfoType builtinMetaInfoType;
  VoidType voidType;
  NullptrType nullptrType;
  DecltypeAutoType decltypeAutoType;
  AutoType autoType;
  BoolType boolType;
  SignedCharType signedCharType;
  ShortIntType shortIntType;
  IntType intType;
  LongIntType longIntType;
  LongLongIntType longLongIntType;
  Int128Type int128Type;
  UnsignedCharType unsignedCharType;
  UnsignedShortIntType unsignedShortIntType;
  UnsignedIntType unsignedIntType;
  UnsignedLongIntType unsignedLongIntType;
  UnsignedLongLongIntType unsignedLongLongIntType;
  UnsignedInt128Type unsignedInt128Type;
  CharType charType;
  Char8Type char8Type;
  Char16Type char16Type;
  Char32Type char32Type;
  WideCharType wideCharType;
  FloatType floatType;
  DoubleType doubleType;
  LongDoubleType longDoubleType;
  Float16Type float16Type;

  std::set<QualType> qualTypes;
  std::set<BoundedArrayType> boundedArrayTypes;
  std::set<UnboundedArrayType> unboundedArrayTypes;
  std::set<PointerType> pointerTypes;
  std::set<LvalueReferenceType> lvalueReferenceTypes;
  std::set<RvalueReferenceType> rvalueReferenceTypes;
  std::set<OverloadSetType> overloadSetTypes;
  std::set<FunctionType> functionTypes;
  std::set<MemberObjectPointerType> memberObjectPointerTypes;
  std::set<MemberFunctionPointerType> memberFunctionPointerTypes;
  std::set<TypeParameterType> typeParameterTypes;
  std::set<TemplateTypeParameterType> templateTypeParameterTypes;
  std::set<UnresolvedNameType> unresolvedNameTypes;
  std::set<UnresolvedBoundedArrayType> unresolvedBoundedArrayTypes;
  std::set<UnresolvedUnderlyingType> unresolvedUnderlyingTypes;
  std::set<ClassType> classTypes;
  std::set<NamespaceType> namespaceTypes;
  std::set<EnumType> enumTypes;
  std::set<ScopedEnumType> scopedEnumTypes;

  std::forward_list<NamespaceSymbol> namespaceSymbols;
  std::forward_list<ConceptSymbol> conceptSymbols;
  std::forward_list<BaseClassSymbol> baseClassSymbols;
  std::forward_list<ClassSymbol> classSymbols;
  std::forward_list<EnumSymbol> enumSymbols;
  std::forward_list<ScopedEnumSymbol> scopedEnumSymbols;
  std::forward_list<OverloadSetSymbol> overloadSetSymbols;
  std::forward_list<FunctionSymbol> functionSymbols;
  std::forward_list<LambdaSymbol> lambdaSymbols;
  std::forward_list<FunctionParametersSymbol> functionParametersSymbol;
  std::forward_list<TemplateParametersSymbol> templateParametersSymbol;
  std::forward_list<BlockSymbol> blockSymbols;
  std::forward_list<TypeAliasSymbol> typeAliasSymbols;
  std::forward_list<VariableSymbol> variableSymbols;
  std::forward_list<FieldSymbol> fieldSymbols;
  std::forward_list<ParameterSymbol> parameterSymbols;
  std::forward_list<ParameterPackSymbol> parameterPackSymbols;
  std::forward_list<TypeParameterSymbol> typeParameterSymbols;
  std::forward_list<NonTypeParameterSymbol> nonTypeParameterSymbols;
  std::forward_list<TemplateTypeParameterSymbol> templateTypeParameterSymbols;
  std::forward_list<ConstraintTypeParameterSymbol>
      constraintTypeParameterSymbols;
  std::forward_list<EnumeratorSymbol> enumeratorSymbols;
  std::forward_list<UsingDeclarationSymbol> usingDeclarationSymbols;

  std::forward_list<TypeTraitIdentifierInfo> typeTraitIdentifierInfos;
  std::forward_list<UnaryBuiltinTypeInfo> unaryBuiltinTypeInfos;
  std::forward_list<BuiltinFunctionIdentifierInfo> builtinFunctionInfos;
  std::forward_list<BuiltinTemplateIdentifierInfo> builtinTemplateInfos;

  int anonymousIdCount = 0;

  [[nodiscard]] auto getIdentifier(std::string_view name) -> const Identifier* {
    if (auto it = identifiers.find(name); it != identifiers.end()) return &*it;
    return &*identifiers.emplace(std::string(name)).first;
  }

  void initBuiltinTypeTraits() {
#define PROCESS_BUILTIN(id, name) \
  getIdentifier(name)->setInfo(   \
      &typeTraitIdentifierInfos.emplace_front(BuiltinTypeTraitKind::T_##id));

    FOR_EACH_BUILTIN_TYPE_TRAIT(PROCESS_BUILTIN)

#undef PROCESS_BUILTIN

#define PROCESS_UNARY_BUILTIN(id, name) \
  getIdentifier(name)->setInfo(         \
      &unaryBuiltinTypeInfos.emplace_front(UnaryBuiltinTypeKind::T_##id));
    FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(PROCESS_UNARY_BUILTIN)
#undef PROCESS_UNARY_BUILTIN
  }

  void initBuiltinFunctions() {
#define PROCESS_BUILTIN_FUNCTION(id, name) \
  getIdentifier(name)->setInfo(            \
      &builtinFunctionInfos.emplace_front(BuiltinFunctionKind::T_##id));

    FOR_EACH_BUILTIN_FUNCTION(PROCESS_BUILTIN_FUNCTION)

#undef PROCESS_BUILTIN_FUNCTION
  }

  void initBuiltinTemplates() {
#define PROCESS_BUILTIN_TEMPLATE(id, name) \
  getIdentifier(name)->setInfo(            \
      &builtinTemplateInfos.emplace_front(BuiltinTemplateKind::T_##id));

    FOR_EACH_BUILTIN_TEMPLATE(PROCESS_BUILTIN_TEMPLATE)

#undef PROCESS_BUILTIN_TEMPLATE
  }
};

Control::Control() : d(std::make_unique<Private>(this)) {
  d->initBuiltinTypeTraits();
  d->initBuiltinFunctions();
  d->initBuiltinTemplates();
}

Control::~Control() = default;

auto Control::integerLiteral(std::string_view spelling)
    -> const IntegerLiteral* {
  if (auto it = d->integerLiterals.find(spelling);
      it != d->integerLiterals.end())
    return &*it;
  auto it = d->integerLiterals.emplace(std::string(spelling)).first;
  it->initialize();
  return &*it;
}

auto Control::floatLiteral(std::string_view spelling) -> const FloatLiteral* {
  if (auto it = d->floatLiterals.find(spelling); it != d->floatLiterals.end())
    return &*it;
  auto it = d->floatLiterals.emplace(std::string(spelling)).first;
  it->initialize();
  return &*it;
}

auto Control::stringLiteral(std::string_view spelling) -> const StringLiteral* {
  if (auto it = d->stringLiterals.find(spelling); it != d->stringLiterals.end())
    return &*it;
  auto it = d->stringLiterals.emplace(std::string(spelling)).first;
  it->initialize(StringLiteralEncoding::kNone);
  return &*it;
}

auto Control::charLiteral(std::string_view spelling) -> const CharLiteral* {
  if (auto it = d->charLiterals.find(spelling); it != d->charLiterals.end())
    return &*it;
  auto it = d->charLiterals.emplace(std::string(spelling)).first;
  it->initialize();
  return &*it;
}

auto Control::wideStringLiteral(std::string_view spelling)
    -> const StringLiteral* {
  if (auto it = d->wideStringLiterals.find(spelling);
      it != d->wideStringLiterals.end())
    return &*it;
  auto it = d->wideStringLiterals.emplace(std::string(spelling)).first;
  it->initialize(StringLiteralEncoding::kWide);
  return &*it;
}

auto Control::utf8StringLiteral(std::string_view spelling)
    -> const StringLiteral* {
  if (auto it = d->utf8StringLiterals.find(spelling);
      it != d->utf8StringLiterals.end())
    return &*it;
  auto it = d->utf8StringLiterals.emplace(std::string(spelling)).first;
  it->initialize(StringLiteralEncoding::kUtf8);
  return &*it;
}

auto Control::utf16StringLiteral(std::string_view spelling)
    -> const StringLiteral* {
  if (auto it = d->utf16StringLiterals.find(spelling);
      it != d->utf16StringLiterals.end())
    return &*it;
  auto it = d->utf16StringLiterals.emplace(std::string(spelling)).first;
  it->initialize(StringLiteralEncoding::kUtf16);
  return &*it;
}

auto Control::utf32StringLiteral(std::string_view spelling)
    -> const StringLiteral* {
  if (auto it = d->utf32StringLiterals.find(spelling);
      it != d->utf32StringLiterals.end())
    return &*it;
  auto it = d->utf32StringLiterals.emplace(std::string(spelling)).first;
  it->initialize(StringLiteralEncoding::kUtf32);
  return &*it;
}

auto Control::commentLiteral(std::string_view spelling)
    -> const CommentLiteral* {
  if (auto it = d->commentLiterals.find(spelling);
      it != d->commentLiterals.end())
    return &*it;
  return &*d->commentLiterals.emplace(std::string(spelling)).first;
}

auto Control::memoryLayout() const -> MemoryLayout* { return d->memoryLayout; }

void Control::setMemoryLayout(MemoryLayout* memoryLayout) {
  d->memoryLayout = memoryLayout;
}

auto Control::newAnonymousId(std::string_view base) -> const Identifier* {
  auto id = std::string("$") + std::string(base) +
            std::to_string(++d->anonymousIdCount);
  return getIdentifier(id.c_str());
}

auto Control::getIdentifier(std::string_view name) -> const Identifier* {
  return d->getIdentifier(name);
}

auto Control::getOperatorId(TokenKind op) -> const OperatorId* {
  return &*d->operatorIds.emplace(op).first;
}

auto Control::getDestructorId(const Name* name) -> const DestructorId* {
  return &*d->destructorIds.emplace(name).first;
}

auto Control::getLiteralOperatorId(std::string_view name)
    -> const LiteralOperatorId* {
  return &*d->literalOperatorIds.emplace(std::string(name)).first;
}

auto Control::getConversionFunctionId(const Type* type)
    -> const ConversionFunctionId* {
  return &*d->conversionFunctionIds.emplace(type).first;
}

auto Control::getTemplateId(const Name* name,
                            std::vector<TemplateArgument> arguments)
    -> const TemplateId* {
  return &*d->templateIds.emplace(name, std::move(arguments)).first;
}

auto Control::getSizeType() -> const Type* {
  // TODO: use the correct type
  return getUnsignedLongIntType();
}

auto Control::getBuiltinVaListType() -> const BuiltinVaListType* {
  return &d->builtinVaListType;
}

auto Control::getBuiltinMetaInfoType() -> const BuiltinMetaInfoType* {
  return &d->builtinMetaInfoType;
}

auto Control::getVoidType() -> const VoidType* { return &d->voidType; }

auto Control::getNullptrType() -> const NullptrType* { return &d->nullptrType; }

auto Control::getDecltypeAutoType() -> const DecltypeAutoType* {
  return &d->decltypeAutoType;
}

auto Control::getAutoType() -> const AutoType* { return &d->autoType; }

auto Control::getBoolType() -> const BoolType* { return &d->boolType; }

auto Control::getSignedCharType() -> const SignedCharType* {
  return &d->signedCharType;
}

auto Control::getShortIntType() -> const ShortIntType* {
  return &d->shortIntType;
}

auto Control::getIntType() -> const IntType* { return &d->intType; }

auto Control::getLongIntType() -> const LongIntType* { return &d->longIntType; }

auto Control::getLongLongIntType() -> const LongLongIntType* {
  return &d->longLongIntType;
}

auto Control::getInt128Type() -> const Int128Type* { return &d->int128Type; }

auto Control::getUnsignedCharType() -> const UnsignedCharType* {
  return &d->unsignedCharType;
}

auto Control::getUnsignedShortIntType() -> const UnsignedShortIntType* {
  return &d->unsignedShortIntType;
}

auto Control::getUnsignedIntType() -> const UnsignedIntType* {
  return &d->unsignedIntType;
}

auto Control::getUnsignedLongIntType() -> const UnsignedLongIntType* {
  return &d->unsignedLongIntType;
}

auto Control::getUnsignedLongLongIntType() -> const UnsignedLongLongIntType* {
  return &d->unsignedLongLongIntType;
}

auto Control::getUnsignedInt128Type() -> const UnsignedInt128Type* {
  return &d->unsignedInt128Type;
}

auto Control::getCharType() -> const CharType* { return &d->charType; }

auto Control::getChar8Type() -> const Char8Type* { return &d->char8Type; }

auto Control::getChar16Type() -> const Char16Type* { return &d->char16Type; }

auto Control::getChar32Type() -> const Char32Type* { return &d->char32Type; }

auto Control::getWideCharType() -> const WideCharType* {
  return &d->wideCharType;
}

auto Control::getFloatType() -> const FloatType* { return &d->floatType; }

auto Control::getDoubleType() -> const DoubleType* { return &d->doubleType; }

auto Control::getLongDoubleType() -> const LongDoubleType* {
  return &d->longDoubleType;
}

auto Control::getFloat16Type() -> const Float16Type* { return &d->float16Type; }

auto Control::getQualType(const Type* elementType, CvQualifiers cvQualifiers)
    -> const QualType* {
  if (auto qualType = type_cast<QualType>(elementType)) {
    cvQualifiers = cvQualifiers | qualType->cvQualifiers();
    return &*d->qualTypes
                 .emplace(qualType->elementType(),
                          cvQualifiers | qualType->cvQualifiers())
                 .first;
  }

  return &*d->qualTypes.emplace(elementType, cvQualifiers).first;
}

auto Control::getConstType(const Type* elementType) -> const QualType* {
  return getQualType(elementType, CvQualifiers::kConst);
}

auto Control::getVolatileType(const Type* elementType) -> const QualType* {
  return getQualType(elementType, CvQualifiers::kVolatile);
}

auto Control::getConstVolatileType(const Type* elementType) -> const QualType* {
  return getQualType(elementType, CvQualifiers::kConstVolatile);
}

auto Control::getBoundedArrayType(const Type* elementType, std::size_t size)
    -> const BoundedArrayType* {
  return &*d->boundedArrayTypes.emplace(elementType, size).first;
}

auto Control::getUnboundedArrayType(const Type* elementType)
    -> const UnboundedArrayType* {
  return &*d->unboundedArrayTypes.emplace(elementType).first;
}

auto Control::getPointerType(const Type* elementType) -> const PointerType* {
  return &*d->pointerTypes.emplace(elementType).first;
}

auto Control::getLvalueReferenceType(const Type* elementType)
    -> const LvalueReferenceType* {
  return &*d->lvalueReferenceTypes.emplace(elementType).first;
}

auto Control::getRvalueReferenceType(const Type* elementType)
    -> const RvalueReferenceType* {
  return &*d->rvalueReferenceTypes.emplace(elementType).first;
}

auto Control::getOverloadSetType(OverloadSetSymbol* symbol)
    -> const OverloadSetType* {
  return &*d->overloadSetTypes.emplace(symbol).first;
}

auto Control::getFunctionType(const Type* returnType,
                              std::vector<const Type*> parameterTypes,
                              bool isVariadic, CvQualifiers cvQualifiers,
                              RefQualifier refQualifier, bool isNoexcept)
    -> const FunctionType* {
  return &*d->functionTypes
               .emplace(returnType, std::move(parameterTypes), isVariadic,
                        cvQualifiers, refQualifier, isNoexcept)
               .first;
}

auto Control::getMemberObjectPointerType(const ClassType* classType,
                                         const Type* elementType)
    -> const MemberObjectPointerType* {
  return &*d->memberObjectPointerTypes.emplace(classType, elementType).first;
}

auto Control::getMemberFunctionPointerType(const ClassType* classType,
                                           const FunctionType* functionType)
    -> const MemberFunctionPointerType* {
  return &*d->memberFunctionPointerTypes.emplace(classType, functionType).first;
}

auto Control::getTypeParameterType(int index, int depth, bool isParameterPack)
    -> const TypeParameterType* {
  return &*d->typeParameterTypes.emplace(index, depth, isParameterPack).first;
}

auto Control::getTemplateTypeParameterType(
    int index, int depth, bool isPack,
    std::vector<const Type*> templateParameters)
    -> const TemplateTypeParameterType* {
  return &*d->templateTypeParameterTypes
               .emplace(index, depth, isPack, std::move(templateParameters))
               .first;
}

auto Control::getUnresolvedNameType(TranslationUnit* unit,
                                    NestedNameSpecifierAST* nestedNameSpecifier,
                                    UnqualifiedIdAST* unqualifiedId)
    -> const UnresolvedNameType* {
  return &*d->unresolvedNameTypes
               .emplace(unit, nestedNameSpecifier, unqualifiedId)
               .first;
}

auto Control::getUnresolvedBoundedArrayType(TranslationUnit* unit,
                                            const Type* elementType,
                                            ExpressionAST* sizeExpression)
    -> const UnresolvedBoundedArrayType* {
  return &*d->unresolvedBoundedArrayTypes
               .emplace(unit, elementType, sizeExpression)
               .first;
}

auto Control::getUnresolvedUnderlyingType(TranslationUnit* unit,
                                          TypeIdAST* typeId)
    -> const UnresolvedUnderlyingType* {
  return &*d->unresolvedUnderlyingTypes.emplace(unit, typeId).first;
}

auto Control::getClassType(ClassSymbol* symbol) -> const ClassType* {
  return &*d->classTypes.emplace(symbol).first;
}

auto Control::getNamespaceType(NamespaceSymbol* symbol)
    -> const NamespaceType* {
  return &*d->namespaceTypes.emplace(symbol).first;
}

auto Control::getEnumType(EnumSymbol* symbol) -> const EnumType* {
  return &*d->enumTypes.emplace(symbol).first;
}

auto Control::getScopedEnumType(ScopedEnumSymbol* symbol)
    -> const ScopedEnumType* {
  return &*d->scopedEnumTypes.emplace(symbol).first;
}

auto Control::newNamespaceSymbol(ScopeSymbol* enclosingScope,
                                 SourceLocation loc) -> NamespaceSymbol* {
  auto symbol = &d->namespaceSymbols.emplace_front(enclosingScope);
  symbol->setType(getNamespaceType(symbol));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newConceptSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> ConceptSymbol* {
  auto symbol = &d->conceptSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newBaseClassSymbol(ScopeSymbol* enclosingScope,
                                 SourceLocation loc) -> BaseClassSymbol* {
  auto symbol = &d->baseClassSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newClassSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> ClassSymbol* {
  auto symbol = &d->classSymbols.emplace_front(enclosingScope);
  symbol->setType(getClassType(symbol));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newEnumSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> EnumSymbol* {
  auto symbol = &d->enumSymbols.emplace_front(enclosingScope);
  symbol->setType(getEnumType(symbol));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newScopedEnumSymbol(ScopeSymbol* enclosingScope,
                                  SourceLocation loc) -> ScopedEnumSymbol* {
  auto symbol = &d->scopedEnumSymbols.emplace_front(enclosingScope);
  symbol->setType(getScopedEnumType(symbol));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newOverloadSetSymbol(ScopeSymbol* enclosingScope,
                                   SourceLocation loc) -> OverloadSetSymbol* {
  auto symbol = &d->overloadSetSymbols.emplace_front(enclosingScope);
  symbol->setType(getOverloadSetType(symbol));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newFunctionSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> FunctionSymbol* {
  auto symbol = &d->functionSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newLambdaSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> LambdaSymbol* {
  auto symbol = &d->lambdaSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newFunctionParametersSymbol(ScopeSymbol* enclosingScope,
                                          SourceLocation loc)
    -> FunctionParametersSymbol* {
  auto symbol = &d->functionParametersSymbol.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newTemplateParametersSymbol(ScopeSymbol* enclosingScope,
                                          SourceLocation loc)
    -> TemplateParametersSymbol* {
  auto symbol = &d->templateParametersSymbol.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newBlockSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> BlockSymbol* {
  auto symbol = &d->blockSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newTypeAliasSymbol(ScopeSymbol* enclosingScope,
                                 SourceLocation loc) -> TypeAliasSymbol* {
  auto symbol = &d->typeAliasSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newVariableSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> VariableSymbol* {
  auto symbol = &d->variableSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newFieldSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> FieldSymbol* {
  auto symbol = &d->fieldSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newParameterSymbol(ScopeSymbol* enclosingScope,
                                 SourceLocation loc) -> ParameterSymbol* {
  auto symbol = &d->parameterSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newParameterPackSymbol(ScopeSymbol* enclosingScope,
                                     SourceLocation loc)
    -> ParameterPackSymbol* {
  auto symbol = &d->parameterPackSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newTypeParameterSymbol(ScopeSymbol* enclosingScope,
                                     SourceLocation loc, int index, int depth,
                                     bool isParameterPack)
    -> TypeParameterSymbol* {
  auto symbol = &d->typeParameterSymbols.emplace_front(enclosingScope);
  symbol->setType(getTypeParameterType(index, depth, isParameterPack));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newTemplateTypeParameterSymbol(
    ScopeSymbol* enclosingScope, SourceLocation loc, int index, int depth,
    bool isPack, std::vector<const Type*> parameters)
    -> TemplateTypeParameterSymbol* {
  auto symbol = &d->templateTypeParameterSymbols.emplace_front(enclosingScope);
  symbol->setType(getTemplateTypeParameterType(index, depth, isPack,
                                               std::move(parameters)));
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newNonTypeParameterSymbol(ScopeSymbol* enclosingScope,
                                        SourceLocation loc)
    -> NonTypeParameterSymbol* {
  auto symbol = &d->nonTypeParameterSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newConstraintTypeParameterSymbol(ScopeSymbol* enclosingScope,
                                               SourceLocation loc)
    -> ConstraintTypeParameterSymbol* {
  auto symbol =
      &d->constraintTypeParameterSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newEnumeratorSymbol(ScopeSymbol* enclosingScope,
                                  SourceLocation loc) -> EnumeratorSymbol* {
  auto symbol = &d->enumeratorSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newUsingDeclarationSymbol(ScopeSymbol* enclosingScope,
                                        SourceLocation loc)
    -> UsingDeclarationSymbol* {
  auto symbol = &d->usingDeclarationSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::is_void(const Type* type) -> bool {
  return d->traits.is_void(type);
}

auto Control::is_null_pointer(const Type* type) -> bool {
  return d->traits.is_null_pointer(type);
}

auto Control::is_integral(const Type* type) -> bool {
  return d->traits.is_integral(type);
}

auto Control::is_floating_point(const Type* type) -> bool {
  return d->traits.is_floating_point(type);
}

auto Control::is_array(const Type* type) -> bool {
  return d->traits.is_array(type);
}

auto Control::is_enum(const Type* type) -> bool {
  return d->traits.is_enum(type);
}

auto Control::is_union(const Type* type) -> bool {
  return d->traits.is_union(type);
}

auto Control::is_class(const Type* type) -> bool {
  return d->traits.is_class(type);
}

auto Control::is_function(const Type* type) -> bool {
  return d->traits.is_function(type);
}

auto Control::is_pointer(const Type* type) -> bool {
  return d->traits.is_pointer(type);
}

auto Control::is_lvalue_reference(const Type* type) -> bool {
  return d->traits.is_lvalue_reference(type);
}

auto Control::is_rvalue_reference(const Type* type) -> bool {
  return d->traits.is_rvalue_reference(type);
}

auto Control::is_member_object_pointer(const Type* type) -> bool {
  return d->traits.is_member_object_pointer(type);
}

auto Control::is_member_function_pointer(const Type* type) -> bool {
  return d->traits.is_member_function_pointer(type);
}

auto Control::is_complete(const Type* type) -> bool {
  return d->traits.is_complete(type);
}

auto Control::is_integer(const Type* type) -> bool {
  return d->traits.is_integer(type);
}

auto Control::is_integral_or_unscoped_enum(const Type* type) -> bool {
  return d->traits.is_integral_or_unscoped_enum(type);
}

auto Control::is_arithmetic_or_unscoped_enum(const Type* type) -> bool {
  return is_arithmetic(type) || (is_enum(type) && !is_scoped_enum(type));
}

auto Control::is_fundamental(const Type* type) -> bool {
  return d->traits.is_fundamental(type);
}

auto Control::is_arithmetic(const Type* type) -> bool {
  return d->traits.is_arithmetic(type);
}

auto Control::is_scalar(const Type* type) -> bool {
  return d->traits.is_scalar(type);
}

auto Control::is_object(const Type* type) -> bool {
  return d->traits.is_object(type);
}

auto Control::is_compound(const Type* type) -> bool {
  return d->traits.is_compound(type);
}

auto Control::is_reference(const Type* type) -> bool {
  return d->traits.is_reference(type);
}

auto Control::is_member_pointer(const Type* type) -> bool {
  return d->traits.is_member_pointer(type);
}

auto Control::is_class_or_union(const Type* type) -> bool {
  return is_class(type) || is_union(type);
}

auto Control::is_const(const Type* type) -> bool {
  return d->traits.is_const(type);
}

auto Control::is_volatile(const Type* type) -> bool {
  return d->traits.is_volatile(type);
}

auto Control::is_signed(const Type* type) -> bool {
  return d->traits.is_signed(type);
}

auto Control::is_unsigned(const Type* type) -> bool {
  return d->traits.is_unsigned(type);
}

auto Control::is_bounded_array(const Type* type) -> bool {
  return d->traits.is_bounded_array(type);
}

auto Control::is_unbounded_array(const Type* type) -> bool {
  return d->traits.is_unbounded_array(type);
}

auto Control::is_scoped_enum(const Type* type) -> bool {
  return d->traits.is_scoped_enum(type);
}

auto Control::remove_reference(const Type* type) -> const Type* {
  return d->traits.remove_reference(type);
}

auto Control::add_lvalue_reference(const Type* type) -> const Type* {
  return d->traits.add_lvalue_reference(type);
}

auto Control::add_rvalue_reference(const Type* type) -> const Type* {
  return d->traits.add_rvalue_reference(type);
}

auto Control::remove_extent(const Type* type) -> const Type* {
  return d->traits.remove_extent(type);
}

auto Control::remove_all_extents(const Type* type) -> const Type* {
  while (is_array(type)) {
    type = remove_extent(type);
  }
  return type;
}

auto Control::get_element_type(const Type* type) -> const Type* {
  return d->traits.get_element_type(type);
}

auto Control::remove_cv(const Type* type) -> const Type* {
  return d->traits.remove_cv(type);
}

auto Control::remove_const(const Type* type) -> const Type* {
  if (auto qualType = type_cast<QualType>(type)) {
    if (qualType->isConst()) {
      if (qualType->isVolatile())
        return getQualType(qualType->elementType(), CvQualifiers::kVolatile);
      return qualType->elementType();
    }
  }
  return type;
}

auto Control::remove_volatile(const Type* type) -> const Type* {
  if (auto qualType = type_cast<QualType>(type)) {
    if (qualType->isVolatile()) {
      if (qualType->isConst())
        return getQualType(qualType->elementType(), CvQualifiers::kConst);
      return qualType->elementType();
    }
  }
  return type;
}

auto Control::remove_cvref(const Type* type) -> const Type* {
  return d->traits.remove_cvref(type);
}

auto Control::add_const_ref(const Type* type) -> const Type* {
  return d->traits.add_const_ref(type);
}

auto Control::add_const(const Type* type) -> const Type* {
  return d->traits.add_const(type);
}

auto Control::add_volatile(const Type* type) -> const Type* {
  return d->traits.add_volatile(type);
}

auto Control::add_cv(const Type* type, CvQualifiers cv) -> const Type* {
  if (cxx::is_const(cv)) type = add_const(type);
  if (cxx::is_volatile(cv)) type = add_volatile(type);
  return type;
}

auto Control::get_cv_qualifiers(const Type* type) -> CvQualifiers {
  if (auto qualType = type_cast<QualType>(type))
    return qualType->cvQualifiers();
  return CvQualifiers::kNone;
}

auto Control::remove_pointer(const Type* type) -> const Type* {
  return d->traits.remove_pointer(type);
}

auto Control::add_pointer(const Type* type) -> const Type* {
  return d->traits.add_pointer(type);
}

auto Control::remove_noexcept(const Type* type) -> const Type* {
  const auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return type;
  return getFunctionType(
      functionType->returnType(), functionType->parameterTypes(),
      functionType->isVariadic(), functionType->cvQualifiers(),
      functionType->refQualifier(), false);
}

auto Control::is_base_of(const Type* base, const Type* derived) -> bool {
  auto baseClassType = type_cast<ClassType>(remove_cv(base));
  if (!baseClassType) return false;
  auto derivedClassType = type_cast<ClassType>(remove_cv(derived));
  if (!derivedClassType) {
    // todo: test for closure types
    return false;
  }
  if (derivedClassType->symbol() == baseClassType->symbol()) return true;
  return derivedClassType->symbol()->hasBaseClass(baseClassType->symbol());
}

auto Control::is_same(const Type* a, const Type* b) -> bool {
  return d->traits.is_same(a, b);
}

auto Control::is_convertible(const Type* from, const Type* to) -> bool {
  if (!from || !to) return false;

  auto fromUnqual = remove_cv(from);
  auto toUnqual = remove_cv(to);

  // void -> void is convertible
  if (is_void(fromUnqual) && is_void(toUnqual)) return true;

  // nothing else converts to/from void
  if (is_void(fromUnqual) || is_void(toUnqual)) return false;

  // same type is always convertible
  if (is_same(fromUnqual, toUnqual)) return true;

  // arithmetic conversions
  if (is_arithmetic(fromUnqual) && is_arithmetic(toUnqual)) return true;

  // null pointer to any pointer
  if (is_null_pointer(fromUnqual) && is_pointer(toUnqual)) return true;

  // enum to integral
  if (is_enum(fromUnqual) && is_integral(toUnqual)) return true;

  // pointer conversions: derived* -> base*
  if (is_pointer(fromUnqual) && is_pointer(toUnqual)) {
    auto fromPointee = remove_pointer(fromUnqual);
    auto toPointee = remove_pointer(toUnqual);
    if (is_void(remove_cv(toPointee))) return true;
    if (is_base_of(remove_cv(toPointee), remove_cv(fromPointee))) return true;
  }

  // derived -> base reference binding (class types)
  if (auto fromClass = type_cast<ClassType>(fromUnqual)) {
    if (auto toClass = type_cast<ClassType>(toUnqual)) {
      if (is_base_of(toUnqual, fromUnqual)) return true;
    }
  }

  // bool conversions: arithmetic, pointer, enum -> bool
  if (type_cast<BoolType>(toUnqual)) {
    if (is_arithmetic(fromUnqual) || is_pointer(fromUnqual) ||
        is_enum(fromUnqual) || is_null_pointer(fromUnqual))
      return true;
  }

  return false;
}

auto Control::decay(const Type* type) -> const Type* {
  return d->traits.decay(type);
}

auto Control::is_pod(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (is_void(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    if (cls->hasUserDeclaredConstructors()) return false;
    if (cls->hasVirtualFunctions()) return false;
    if (cls->hasVirtualBaseClasses()) return false;
    return true;
  }
  if (is_array(unqual)) return true;
  return false;
}

namespace {

auto isUserProvided(FunctionSymbol* fn) -> bool {
  return fn && !fn->isDefaulted() && !fn->isDeleted();
}

auto is_trivially_copyable_class(Control* control, ClassSymbol* cls) -> bool {
  if (!cls || !cls->isComplete()) return false;

  auto dtor = cls->destructor();
  if (dtor && dtor->isDeleted()) return false;
  if (isUserProvided(dtor)) return false;
  if (dtor && dtor->isVirtual()) return false;

  if (isUserProvided(cls->copyConstructor())) return false;
  if (isUserProvided(cls->moveConstructor())) return false;
  if (isUserProvided(cls->copyAssignmentOperator())) return false;
  if (isUserProvided(cls->moveAssignmentOperator())) return false;

  if (cls->isPolymorphic()) return false;
  if (cls->hasVirtualBaseClasses()) return false;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!is_trivially_copyable_class(control, baseClass)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType =
        control->remove_all_extents(control->remove_cv(field->type()));
    if (auto ct = type_cast<ClassType>(fieldType)) {
      if (!is_trivially_copyable_class(control, ct->symbol())) return false;
    }
  }

  return true;
}

}  // namespace

auto Control::is_trivial(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    auto defCtor = cls->defaultConstructor();
    if (isUserProvided(defCtor)) return false;
    if (!is_trivially_copyable_class(this, cls)) return false;
    return true;
  }
  if (is_array(unqual)) {
    return is_trivial(remove_all_extents(unqual));
  }
  return false;
}

auto Control::is_standard_layout(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    if (cls->hasVirtualFunctions()) return false;
    if (cls->hasVirtualBaseClasses()) return false;
    return true;
  }
  if (is_array(unqual)) return true;
  return false;
}

auto Control::is_literal_type(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_void(unqual)) return true;
  if (is_scalar(unqual)) return true;
  if (is_reference(unqual)) return true;
  if (is_array(unqual)) {
    // literal if element type is literal (simplified: accept)
    return true;
  }
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    auto dtor = cls->destructor();
    if (dtor && !dtor->isDefaulted() && !dtor->isDeleted()) return false;
    return true;
  }
  return false;
}

auto Control::is_aggregate(const Type* type) -> bool {
  if (is_array(type)) return true;
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls || !cls->isComplete()) return false;
  if (cls->hasUserDeclaredConstructors()) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  return true;
}

auto Control::is_empty(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls || !cls->isComplete()) return false;
  for (auto f : cls->members() | views::non_static_fields) {
    (void)f;
    return false;
  }
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  return true;
}

auto Control::is_polymorphic(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls || !cls->isComplete()) return false;
  return cls->isPolymorphic();
}

auto Control::is_final(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls) return false;
  return cls->isFinal();
}

auto Control::is_constructible(const Type* type,
                               std::span<const Type* const> argTypes) -> bool {
  if (!type) return false;
  auto unqual = remove_cv(type);

  // References require exactly one argument
  if (is_reference(unqual)) {
    if (argTypes.size() != 1) return false;
    // Simplified: allow reference binding
    return true;
  }

  // Scalar types
  if (is_scalar(unqual)) {
    // Default construction (zero args) is valid for scalars
    if (argTypes.empty()) return true;
    // Scalars can be constructed from a single convertible arg
    if (argTypes.size() == 1) return true;
    return false;
  }

  // Array types: constructible if element type is constructible
  if (is_array(unqual)) {
    if (!argTypes.empty()) return false;
    return is_constructible(remove_all_extents(unqual), argTypes);
  }

  // Class types
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;

    // Check default construction (no args)
    if (argTypes.empty()) {
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) return !cls->hasUserDeclaredConstructors();
      return !defCtor->isDeleted();
    }

    // Check copy construction (one arg, same type or const ref)
    if (argTypes.size() == 1) {
      auto argUnqual = remove_cvref(argTypes[0]);
      if (is_same(argUnqual, unqual)) {
        // Copy or move construction
        if (is_rvalue_reference(argTypes[0]) ||
            (!is_reference(argTypes[0]) && !is_const(argTypes[0]))) {
          // Try move constructor first
          auto moveCtor = cls->moveConstructor();
          if (moveCtor && !moveCtor->isDeleted()) return true;
        }
        auto copyCtor = cls->copyConstructor();
        if (copyCtor && !copyCtor->isDeleted()) return true;
        // If no user-declared constructors, implicit copy/move are available
        if (!cls->hasUserDeclaredConstructors()) return true;
      }
    }

    // General case: check if any constructor matches the arity
    for (auto ctor : cls->constructors()) {
      if (ctor->isDeleted()) continue;
      auto ctorType = type_cast<FunctionType>(ctor->type());
      if (!ctorType) continue;
      auto params = ctorType->parameterTypes();
      // Skip the implicit 'this' parameter if present
      if (params.size() == argTypes.size()) return true;
      if (ctorType->isVariadic() && params.size() <= argTypes.size())
        return true;
    }

    return false;
  }

  // void type
  if (is_void(unqual)) return false;

  return false;
}

auto Control::is_nothrow_constructible(const Type* type,
                                       std::span<const Type* const> argTypes)
    -> bool {
  if (!type) return false;
  auto unqual = remove_cv(type);

  // First check if it's constructible at all
  if (!is_constructible(type, argTypes)) return false;

  // References are nothrow
  if (is_reference(unqual)) return true;

  // Scalar types are always nothrow constructible
  if (is_scalar(unqual)) return true;

  // Array types
  if (is_array(unqual))
    return is_nothrow_constructible(remove_all_extents(unqual), argTypes);

  // Class types: check if the matching constructor is noexcept
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;

    if (argTypes.empty()) {
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) {
        if (!cls->hasUserDeclaredConstructors()) return true;
        return false;
      }
      if (defCtor->isDeleted()) return false;
      // Implicitly-declared/defaulted constructors are noexcept unless
      // they need to call a non-noexcept constructor
      if (defCtor->isDefaulted() || !cls->hasUserDeclaredConstructors()) {
        // Check bases and members for non-noexcept default constructors
        for (auto base : cls->baseClasses()) {
          auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
          if (!baseClass) continue;
          std::span<const Type* const> empty;
          if (!is_nothrow_constructible(baseClass->type(), empty)) return false;
        }
        return true;
      }
      auto ctorType = type_cast<FunctionType>(defCtor->type());
      return ctorType && ctorType->isNoexcept();
    }

    if (argTypes.size() == 1) {
      auto argUnqual = remove_cvref(argTypes[0]);
      if (is_same(argUnqual, unqual)) {
        if (is_rvalue_reference(argTypes[0]) ||
            (!is_reference(argTypes[0]) && !is_const(argTypes[0]))) {
          auto moveCtor = cls->moveConstructor();
          if (moveCtor && !moveCtor->isDeleted()) {
            auto ctorType = type_cast<FunctionType>(moveCtor->type());
            return ctorType && ctorType->isNoexcept();
          }
        }
        auto copyCtor = cls->copyConstructor();
        if (copyCtor && !copyCtor->isDeleted()) {
          auto ctorType = type_cast<FunctionType>(copyCtor->type());
          return ctorType && ctorType->isNoexcept();
        }
        if (!cls->hasUserDeclaredConstructors()) return true;
      }
    }

    for (auto ctor : cls->constructors()) {
      if (ctor->isDeleted()) continue;
      auto ctorType = type_cast<FunctionType>(ctor->type());
      if (!ctorType) continue;
      auto params = ctorType->parameterTypes();
      if (params.size() == argTypes.size()) return ctorType->isNoexcept();
      if (ctorType->isVariadic() && params.size() <= argTypes.size())
        return ctorType->isNoexcept();
    }

    return false;
  }

  return false;
}

auto Control::is_trivially_constructible(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    auto defCtor = cls->defaultConstructor();
    if (isUserProvided(defCtor)) return false;
    if (cls->isPolymorphic()) return false;
    if (cls->hasVirtualBaseClasses()) return false;
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (!is_trivially_constructible(baseClass->type())) return false;
    }
    for (auto field : cls->members() | views::non_static_fields) {
      auto fieldType = remove_all_extents(remove_cv(field->type()));
      if (auto ct = type_cast<ClassType>(fieldType)) {
        if (!is_trivially_constructible(ct->symbol()->type())) return false;
      }
    }
    return true;
  }
  if (is_array(unqual)) {
    return is_trivially_constructible(remove_all_extents(unqual));
  }
  return false;
}

auto Control::is_assignable(const Type* to, const Type* from) -> bool {
  if (!to || !from) return false;

  // __is_assignable(T, U) checks whether declval<T>() = declval<U>() is
  // well-formed. T must be an lvalue reference for non-class types (since
  // assignment to a prvalue is not allowed).

  // If T is an lvalue reference, we can assign to its referent.
  if (is_lvalue_reference(to)) {
    auto targetType = remove_reference(to);

    // Can't assign to const.
    if (is_const(targetType)) return false;

    auto target = remove_cv(targetType);
    auto source = remove_cvref(from);

    // Scalar types: any convertible source.
    if (is_scalar(target)) {
      return is_convertible(source, target);
    }

    // Class types: check for assignment operators.
    if (auto classType = type_cast<ClassType>(target)) {
      auto cls = classType->symbol();
      if (!cls || !cls->isComplete()) return false;

      // Check if source is the same class (copy/move assignment).
      if (is_same(source, target)) {
        // For rvalue ref source → move assignment.
        if (is_rvalue_reference(from)) {
          auto moveOp = cls->moveAssignmentOperator();
          if (moveOp && !moveOp->isDeleted()) return true;
        }
        // Copy assignment.
        auto copyOp = cls->copyAssignmentOperator();
        if (copyOp && !copyOp->isDeleted()) return true;
        // Implicit copy/move assignment is available if no user-declared
        // special members suppress it.
        if (!cls->hasUserDeclaredConstructors()) return true;
        return false;
      }

      // General case: check for any operator= that accepts the source type.
      // Conservative: accept for now — full overload resolution is complex.
      return true;
    }

    // Enum, array, etc.
    return false;
  }

  // If T is an rvalue reference to a class type, class rvalue assignment may be
  // valid (e.g., MyClass{} = other;).
  if (is_rvalue_reference(to)) {
    auto targetType = remove_reference(to);
    if (is_const(targetType)) return false;
    auto target = remove_cv(targetType);

    if (auto classType = type_cast<ClassType>(target)) {
      auto cls = classType->symbol();
      if (!cls || !cls->isComplete()) return false;
      // Classes can be assigned to as rvalues (operator= is not
      // ref-qualified by default).
      auto copyOp = cls->copyAssignmentOperator();
      if (copyOp && !copyOp->isDeleted()) return true;
      auto moveOp = cls->moveAssignmentOperator();
      if (moveOp && !moveOp->isDeleted()) return true;
      if (!cls->hasUserDeclaredConstructors()) return true;
      return false;
    }

    return false;
  }

  // T is not a reference.  Assignment to a non-class prvalue is invalid.
  // For class prvalues, operator= on rvalues may still be valid.
  if (auto classType = type_cast<ClassType>(remove_cv(to))) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    auto copyOp = cls->copyAssignmentOperator();
    if (copyOp && !copyOp->isDeleted()) return true;
    auto moveOp = cls->moveAssignmentOperator();
    if (moveOp && !moveOp->isDeleted()) return true;
    if (!cls->hasUserDeclaredConstructors()) return true;
    return false;
  }

  return false;
}

auto Control::is_trivially_assignable(const Type* from, const Type* to)
    -> bool {
  if (!to) return false;
  auto unqual = remove_cvref(from);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;
    if (is_lvalue_reference(to)) {
      auto inner = remove_cv(remove_reference(to));
      if (is_same(inner, unqual)) {
        auto op = cls->copyAssignmentOperator();
        if (!op || isUserProvided(op)) return false;
        return is_trivially_copyable_class(this, cls);
      }
    }
    if (is_rvalue_reference(to)) {
      auto inner = remove_reference(to);
      if (is_same(inner, unqual)) {
        auto op = cls->moveAssignmentOperator();
        if (!op || isUserProvided(op)) return false;
        return is_trivially_copyable_class(this, cls);
      }
    }
    return false;
  }
  return false;
}

auto Control::is_trivially_copyable(const Type* type) -> bool {
  auto ty = remove_cv(remove_all_extents(type));
  if (is_scalar(ty)) return true;
  if (auto classType = type_cast<ClassType>(ty)) {
    return is_trivially_copyable_class(this, classType->symbol());
  }
  return false;
}

auto Control::is_abstract(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls || !cls->isComplete()) return false;
  return cls->isAbstract();
}

auto Control::is_destructible(const Type* type) -> bool {
  if (!type) return false;

  auto unqual = remove_cv(type);

  // References are destructible.
  if (is_reference(unqual)) return true;

  // void is not destructible.
  if (is_void(unqual)) return false;

  // Function types are not destructible.
  if (is_function(unqual)) return false;

  // Unbounded arrays are not destructible.
  if (is_unbounded_array(unqual)) return false;

  // Bounded arrays: destructible if element type is destructible.
  if (is_bounded_array(unqual)) {
    return is_destructible(remove_all_extents(unqual));
  }

  // Scalar types are destructible.
  if (is_scalar(unqual)) return true;

  // Class types: destructible if destructor is accessible and not deleted.
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->symbol();
    if (!cls || !cls->isComplete()) return false;

    auto dtor = cls->destructor();
    if (dtor && dtor->isDeleted()) return false;

    // If there's a destructor and it's not deleted, it's destructible.
    // If there's no explicit destructor, the implicit one is available.
    return true;
  }

  // Enum types are destructible (scalar).
  if (is_enum(unqual)) return true;

  return false;
}

auto Control::has_virtual_destructor(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->symbol();
  if (!cls || !cls->isComplete()) return false;
  return cls->hasVirtualDestructor();
}

}  // namespace cxx
