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

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <format>
#include <forward_list>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cxx {
namespace {
template <typename... Fields>
[[nodiscard]] auto typeFields(const std::tuple<Fields...>& fields)
    -> const std::tuple<Fields...>& {
  return fields;
}

void combineHash(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2);
}

template <typename Field>
[[nodiscard]] auto hashField(const Field& field) -> std::size_t {
  return std::hash<Field>{}(field);
}

template <typename Element>
[[nodiscard]] auto hashField(const std::vector<Element>& elements)
    -> std::size_t {
  std::size_t seed = elements.size();
  for (const auto& element : elements) combineHash(seed, hashField(element));
  return seed;
}

struct TypeFieldsHash {
  template <typename T>
  [[nodiscard]] auto operator()(const T& type) const -> std::size_t {
    return std::apply(
        [](const auto&... fields) {
          std::size_t seed = 0;
          (combineHash(seed, hashField(fields)), ...);
          return seed;
        },
        typeFields(type));
  }
};

struct TypeFieldsEqual {
  template <typename T>
  [[nodiscard]] auto operator()(const T& lhs, const T& rhs) const -> bool {
    return typeFields(lhs) == typeFields(rhs);
  }
};

template <typename T>
using TypeSet = std::unordered_set<T, TypeFieldsHash, TypeFieldsEqual>;

template <typename T, typename... Args>
[[nodiscard]] auto internType(TypeSet<T>& types, Args&&... args) -> const T* {
  T type{std::forward<Args>(args)...};
  if (auto position = types.find(type); position != types.end())
    return &*position;
  return &*types.insert(std::move(type)).first;
}

[[nodiscard]] auto withCompletedTemplateArguments(
    TranslationUnit* unit, NestedNameSpecifierAST* nestedNameSpecifier)
    -> NestedNameSpecifierAST* {
  auto templateQualifier =
      ast_cast<TemplateNestedNameSpecifierAST>(nestedNameSpecifier);
  if (!templateQualifier || !templateQualifier->templateId)
    return nestedNameSpecifier;

  auto templateId = templateQualifier->templateId;

  auto completed = TemplateArguments{unit}.complete(
      templateId->symbol, templateId->templateArgumentList);

  if (completed == templateId->templateArgumentList) return nestedNameSpecifier;

  auto arena = unit->arena();

  auto completedId = SimpleTemplateIdAST::create(arena);
  completedId->identifierLoc = templateId->identifierLoc;
  completedId->lessLoc = templateId->lessLoc;
  completedId->templateArgumentList = completed;
  completedId->greaterLoc = templateId->greaterLoc;
  completedId->identifier = templateId->identifier;
  completedId->symbol = templateId->symbol;

  auto result = TemplateNestedNameSpecifierAST::create(arena);
  result->templateLoc = templateQualifier->templateLoc;
  result->nestedNameSpecifier = templateQualifier->nestedNameSpecifier;
  result->templateId = completedId;
  result->scopeLoc = templateQualifier->scopeLoc;
  result->symbol = templateQualifier->symbol;
  result->isTemplateIntroduced = templateQualifier->isTemplateIntroduced;
  return result;
}

[[nodiscard]] auto appendNestedNameSpecifier(
    TranslationUnit* unit, NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId) -> NestedNameSpecifierAST* {
  if (auto nameId = ast_cast<NameIdAST>(unqualifiedId)) {
    auto result = SimpleNestedNameSpecifierAST::create(unit->arena());
    result->nestedNameSpecifier = nestedNameSpecifier;
    result->identifierLoc = nameId->identifierLoc;
    result->identifier = nameId->identifier;
    return result;
  }

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    auto result = TemplateNestedNameSpecifierAST::create(unit->arena());
    result->nestedNameSpecifier = nestedNameSpecifier;
    result->templateId = templateId;
    return result;
  }

  return nullptr;
}

[[nodiscard]] auto newConstantArgumentSymbol(Control* control, const Type* type,
                                             const ConstValue& value)
    -> VariableSymbol* {
  auto symbol = control->newVariableSymbol(nullptr, {});
  symbol->setType(type);
  symbol->setConstexpr(true);
  symbol->setConstValue(value);
  return symbol;
}

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

#define COUNT_WELL_KNOWN_NAME(id, name) +1
constexpr std::size_t kWellKnownNameCount =
    1 FOR_EACH_WELL_KNOWN_NAME(COUNT_WELL_KNOWN_NAME);
#undef COUNT_WELL_KNOWN_NAME
}  // namespace

struct Control::Private {
  explicit Private(Control*) {}

  std::unordered_set<ClassSymbol*> copyConstructorSelections;

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

  TypeSet<QualType> qualTypes;
  TypeSet<BoundedArrayType> boundedArrayTypes;
  TypeSet<UnboundedArrayType> unboundedArrayTypes;
  TypeSet<PointerType> pointerTypes;
  TypeSet<LvalueReferenceType> lvalueReferenceTypes;
  TypeSet<RvalueReferenceType> rvalueReferenceTypes;
  TypeSet<OverloadSetType> overloadSetTypes;
  TypeSet<FunctionType> functionTypes;
  TypeSet<MemberObjectPointerType> memberObjectPointerTypes;
  TypeSet<MemberFunctionPointerType> memberFunctionPointerTypes;
  TypeSet<TypeParameterType> typeParameterTypes;
  TypeSet<TemplateTypeParameterType> templateTypeParameterTypes;
  std::map<const TemplateTypeParameterType*,
           std::forward_list<TemplateTypeParameterSpecializationType>>
      templateTypeParameterSpecializationTypes;
  TypeSet<PackExpansionType> packExpansionTypes;
  TypeSet<DecltypeType> decltypeTypes;
  TypeSet<UnresolvedNameType> unresolvedNameTypes;
  TypeSet<UnresolvedBoundedArrayType> unresolvedBoundedArrayTypes;
  TypeSet<UnresolvedUnderlyingType> unresolvedUnderlyingTypes;
  TypeSet<UnresolvedBuiltinType> unresolvedBuiltinTypes;
  TypeSet<ClassType> classTypes;
  TypeSet<NamespaceType> namespaceTypes;
  TypeSet<EnumType> enumTypes;
  TypeSet<ScopedEnumType> scopedEnumTypes;
  TypeSet<BitIntType> bitIntTypes;
  TypeSet<SveType> sveTypes;
  TypeSet<UnsignedBitIntType> unsignedBitIntTypes;
  TypeSet<UnresolvedBitIntType> unresolvedBitIntTypes;
  TypeSet<VectorType> vectorTypes;
  TypeSet<UnresolvedVectorType> unresolvedVectorTypes;
  TypeSet<ComplexType> complexTypes;
  TypeSet<AtomicType> atomicTypes;

  std::set<std::vector<const Identifier*>> abiTags;
  std::set<AttributeMap> attributes;
  std::forward_list<NamespaceSymbol> namespaceSymbols;
  std::forward_list<ConceptSymbol> conceptSymbols;
  std::forward_list<DeductionGuideSymbol> deductionGuideSymbols;
  std::forward_list<BaseClassSymbol> baseClassSymbols;
  std::forward_list<InjectedClassNameSymbol> injectedClassNameSymbols;
  std::forward_list<UnresolvedSymbol> unresolvedSymbols;
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
  std::unordered_map<const Type*, TypeAliasSymbol*> typeArgumentSymbols;
  std::map<std::tuple<const Type*, std::uint64_t, std::uint64_t, int, bool>,
           VariableSymbol*>
      constantArgumentSymbols;
  std::map<std::vector<Symbol*>, ParameterPackSymbol*> packArgumentSymbols;
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
  std::forward_list<NamespaceAliasSymbol> namespaceAliasSymbols;

  std::forward_list<TypeTraitIdentifierInfo> typeTraitIdentifierInfos;
  std::forward_list<UnaryBuiltinTypeInfo> unaryBuiltinTypeInfos;
  std::forward_list<BuiltinFunctionIdentifierInfo> builtinFunctionInfos;
  std::forward_list<BuiltinTemplateIdentifierInfo> builtinTemplateInfos;
  std::forward_list<WellKnownNameIdentifierInfo> wellKnownNameInfos;
  std::array<const Identifier*, kWellKnownNameCount> wellKnownIdentifiers{};

  [[nodiscard]] auto allocatedSymbolCounts() const -> SymbolCounts {
    SymbolCounts counts{};
    auto count = [&](const auto& symbols) {
      using SymbolType =
          typename std::remove_cvref_t<decltype(symbols)>::value_type;
      counts[std::to_underlying(SymbolType::Kind)] =
          static_cast<std::size_t>(std::ranges::distance(symbols));
    };
    count(namespaceSymbols);
    count(conceptSymbols);
    count(deductionGuideSymbols);
    count(baseClassSymbols);
    count(injectedClassNameSymbols);
    count(unresolvedSymbols);
    count(classSymbols);
    count(enumSymbols);
    count(scopedEnumSymbols);
    count(overloadSetSymbols);
    count(functionSymbols);
    count(lambdaSymbols);
    count(functionParametersSymbol);
    count(templateParametersSymbol);
    count(blockSymbols);
    count(typeAliasSymbols);
    count(variableSymbols);
    count(fieldSymbols);
    count(parameterSymbols);
    count(parameterPackSymbols);
    count(typeParameterSymbols);
    count(nonTypeParameterSymbols);
    count(templateTypeParameterSymbols);
    count(constraintTypeParameterSymbols);
    count(enumeratorSymbols);
    count(usingDeclarationSymbols);
    count(namespaceAliasSymbols);
    return counts;
  }

  int anonymousIdCount = 0;
  const Type* alignValType = nullptr;
  const Type* nothrowType = nullptr;

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

  void initWellKnownNames() {
#define PROCESS_WELL_KNOWN_NAME(id, name) \
  registerWellKnownName(WellKnownName::T_##id, name);

    FOR_EACH_WELL_KNOWN_NAME(PROCESS_WELL_KNOWN_NAME)

#undef PROCESS_WELL_KNOWN_NAME
  }

  void registerWellKnownName(WellKnownName wellKnownName,
                             std::string_view spelling) {
    auto identifier = getIdentifier(spelling);
    identifier->setInfo(&wellKnownNameInfos.emplace_front(wellKnownName));
    wellKnownIdentifiers[std::to_underlying(wellKnownName)] = identifier;
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
  d->initWellKnownNames();
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

auto Control::stringLiteralFromValue(std::string_view value)
    -> const StringLiteral* {
  return stringLiteral(quoteStringLiteral(value));
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

auto Control::getIdentifier(WellKnownName name) const -> const Identifier* {
  return d->wellKnownIdentifiers[std::to_underlying(name)];
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
  auto layout = memoryLayout();
  if (!layout) return getUnsignedLongIntType();
  if (layout->sizeOfLong() == layout->sizeOfSizeType())
    return getUnsignedLongIntType();
  return getUnsignedLongLongIntType();
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
    cvQualifiers |= qualType->cvQualifiers();
    elementType = qualType->elementType();
  }

  return internType(d->qualTypes, elementType, cvQualifiers);
}

auto Control::getBoundedArrayType(const Type* elementType, std::size_t size)
    -> const BoundedArrayType* {
  return internType(d->boundedArrayTypes, elementType, size);
}

auto Control::getUnboundedArrayType(const Type* elementType)
    -> const UnboundedArrayType* {
  return internType(d->unboundedArrayTypes, elementType);
}

auto Control::getPointerType(const Type* elementType) -> const PointerType* {
  return internType(d->pointerTypes, elementType);
}

auto Control::getLvalueReferenceType(const Type* elementType)
    -> const LvalueReferenceType* {
  return internType(d->lvalueReferenceTypes, elementType);
}

auto Control::getRvalueReferenceType(const Type* elementType)
    -> const RvalueReferenceType* {
  return internType(d->rvalueReferenceTypes, elementType);
}

auto Control::getOverloadSetType(OverloadSetSymbol* symbol)
    -> const OverloadSetType* {
  return internType(d->overloadSetTypes, symbol);
}

auto Control::getFunctionType(const Type* returnType,
                              std::vector<const Type*> parameterTypes,
                              bool isVariadic, CvQualifiers cvQualifiers,
                              RefQualifier refQualifier,
                              ExceptionSpecification exceptionSpecification)
    -> const FunctionType* {
  return internType(d->functionTypes, returnType, std::move(parameterTypes),
                    isVariadic, cvQualifiers, refQualifier,
                    exceptionSpecification);
}

auto Control::getPseudoDestructorType() -> const FunctionType* {
  return getFunctionType(getVoidType(), {}, /*isVariadic=*/false,
                         CvQualifiers::kNone, RefQualifier::kNone,
                         /*exceptionSpecification=*/true);
}

auto Control::getMemberPointerType(const Type* classType,
                                   const Type* memberType) -> const Type* {
  if (auto functionType = type_cast<FunctionType>(memberType))
    return getMemberFunctionPointerType(classType, functionType);
  return getMemberObjectPointerType(classType, memberType);
}

auto Control::getMemberObjectPointerType(const Type* classType,
                                         const Type* elementType)
    -> const MemberObjectPointerType* {
  return internType(d->memberObjectPointerTypes, classType, elementType);
}

auto Control::getMemberFunctionPointerType(const Type* classType,
                                           const FunctionType* functionType)
    -> const MemberFunctionPointerType* {
  return internType(d->memberFunctionPointerTypes, classType, functionType);
}

auto Control::getDependentType() -> const TypeParameterType* {
  return getTypeParameterType(-1, -1, false);
}

auto Control::getTypeParameterType(int index, int depth, bool isParameterPack)
    -> const TypeParameterType* {
  return internType(d->typeParameterTypes, index, depth, isParameterPack);
}

auto Control::getTemplateTypeParameterType(
    int index, int depth, bool isPack,
    std::vector<const Type*> templateParameters)
    -> const TemplateTypeParameterType* {
  return internType(d->templateTypeParameterTypes, index, depth, isPack,
                    std::move(templateParameters));
}

auto Control::getTemplateTypeParameterSpecializationType(
    TranslationUnit* unit, const TemplateTypeParameterType* templateParameter,
    std::vector<TemplateArgument> templateArguments)
    -> const TemplateTypeParameterSpecializationType* {
  auto& types = d->templateTypeParameterSpecializationTypes[templateParameter];
  for (const auto& type : types) {
    if (compare_args(unit, type.templateArguments(), templateArguments))
      return &type;
  }
  return restoreTemplateTypeParameterSpecializationType(
      unit, templateParameter, std::move(templateArguments));
}

auto Control::restoreTemplateTypeParameterSpecializationType(
    TranslationUnit* unit, const TemplateTypeParameterType* templateParameter,
    std::vector<TemplateArgument> templateArguments)
    -> const TemplateTypeParameterSpecializationType* {
  auto& types = d->templateTypeParameterSpecializationTypes[templateParameter];
  return &types.emplace_front(unit, templateParameter,
                              std::move(templateArguments));
}

auto Control::getDecltypeType(TranslationUnit* unit, ExpressionAST* expression)
    -> const DecltypeType* {
  return internType(d->decltypeTypes, unit, expression);
}

auto Control::getPackExpansionType(const Type* pattern)
    -> const PackExpansionType* {
  return internType(d->packExpansionTypes, pattern);
}

auto Control::getUnresolvedNameType(TranslationUnit* unit,
                                    NestedNameSpecifierAST* nestedNameSpecifier,
                                    UnqualifiedIdAST* unqualifiedId)
    -> const UnresolvedNameType* {
  while (nestedNameSpecifier) {
    auto alias = symbol_cast<TypeAliasSymbol>(nestedNameSpecifier->symbol);
    auto expansion =
        alias ? type_cast<UnresolvedNameType>(alias->type()) : nullptr;
    if (!expansion) break;
    auto expandedQualifier = appendNestedNameSpecifier(
        unit, expansion->nestedNameSpecifier(), expansion->unqualifiedId());
    if (!expandedQualifier) break;
    nestedNameSpecifier = expandedQualifier;
  }

  nestedNameSpecifier =
      withCompletedTemplateArguments(unit, nestedNameSpecifier);

  auto type = internType(d->unresolvedNameTypes, unit, nestedNameSpecifier,
                         unqualifiedId);

  unit->captureSnippet(type->sourceLocationRange());

  return type;
}

auto Control::getUnresolvedBoundedArrayType(TranslationUnit* unit,
                                            const Type* elementType,
                                            ExpressionAST* sizeExpression)
    -> const UnresolvedBoundedArrayType* {
  if (sizeExpression)
    unit->captureSnippet(sizeExpression->sourceLocationRange());
  return internType(d->unresolvedBoundedArrayTypes, unit, elementType,
                    sizeExpression);
}

auto Control::getUnresolvedUnderlyingType(TranslationUnit* unit,
                                          TypeIdAST* typeId)
    -> const UnresolvedUnderlyingType* {
  if (typeId) unit->captureSnippet(typeId->sourceLocationRange());
  return internType(d->unresolvedUnderlyingTypes, unit, typeId);
}

auto Control::getUnresolvedBuiltinType(TranslationUnit* unit,
                                       UnaryBuiltinTypeKind builtinKind,
                                       TypeIdAST* typeId)
    -> const UnresolvedBuiltinType* {
  if (typeId) unit->captureSnippet(typeId->sourceLocationRange());
  return internType(d->unresolvedBuiltinTypes, unit, builtinKind, typeId);
}

auto Control::getClassType(ClassSymbol* symbol) -> const ClassType* {
  return internType(d->classTypes, symbol);
}

auto Control::getNamespaceType(NamespaceSymbol* symbol)
    -> const NamespaceType* {
  return internType(d->namespaceTypes, symbol);
}

auto Control::getEnumType(EnumSymbol* symbol) -> const EnumType* {
  return internType(d->enumTypes, symbol);
}

auto Control::getScopedEnumType(ScopedEnumSymbol* symbol)
    -> const ScopedEnumType* {
  return internType(d->scopedEnumTypes, symbol);
}

auto Control::getBitIntType(int numBits) -> const BitIntType* {
  return internType(d->bitIntTypes, numBits);
}

auto Control::getUnsignedBitIntType(int numBits) -> const UnsignedBitIntType* {
  return internType(d->unsignedBitIntTypes, numBits);
}

auto Control::getUnresolvedBitIntType(TranslationUnit* unit,
                                      ExpressionAST* sizeExpression,
                                      bool isUnsigned)
    -> const UnresolvedBitIntType* {
  if (sizeExpression)
    unit->captureSnippet(sizeExpression->sourceLocationRange());
  return internType(d->unresolvedBitIntTypes, unit, sizeExpression, isUnsigned);
}

auto Control::getVectorType(const Type* elementType, std::size_t elementCount,
                            VectorKind vectorKind) -> const VectorType* {
  return internType(d->vectorTypes, elementType, elementCount, vectorKind);
}

auto Control::getUnresolvedVectorType(TranslationUnit* unit,
                                      const Type* elementType,
                                      ExpressionAST* sizeExpression,
                                      VectorKind vectorKind,
                                      VectorSizeKind sizeKind)
    -> const UnresolvedVectorType* {
  if (sizeExpression)
    unit->captureSnippet(sizeExpression->sourceLocationRange());
  return internType(d->unresolvedVectorTypes, unit, elementType, sizeExpression,
                    vectorKind, sizeKind);
}

auto Control::getComplexType(const Type* elementType) -> const ComplexType* {
  return internType(d->complexTypes, elementType);
}

auto Control::getSveType(SveTypeKind sveKind) -> const SveType* {
  return internType(d->sveTypes, sveKind);
}

auto Control::getAtomicType(const Type* elementType) -> const AtomicType* {
  return internType(d->atomicTypes, elementType);
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

auto Control::newDeductionGuideSymbol(ScopeSymbol* enclosingScope,
                                      SourceLocation loc)
    -> DeductionGuideSymbol* {
  auto symbol = &d->deductionGuideSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newBaseClassSymbol(ScopeSymbol* enclosingScope,
                                 SourceLocation loc) -> BaseClassSymbol* {
  auto symbol = &d->baseClassSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newInjectedClassNameSymbol(ScopeSymbol* enclosingScope,
                                         SourceLocation loc)
    -> InjectedClassNameSymbol* {
  auto symbol = &d->injectedClassNameSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newUnresolvedSymbol(ScopeSymbol* enclosingScope,
                                  SourceLocation loc) -> UnresolvedSymbol* {
  auto symbol = &d->unresolvedSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::newClassSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> ClassSymbol* {
  auto symbol = &d->classSymbols.emplace_front(enclosingScope);
  symbol->setType(getClassType(symbol));
  symbol->setLocation(loc);
  symbol->setConstructorOverloadSet(newOverloadSetSymbol(symbol, loc));
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

auto Control::getTypeArgumentSymbol(const Type* type) -> TypeAliasSymbol* {
  auto& symbol = d->typeArgumentSymbols[type];
  if (symbol) return symbol;
  symbol = newTypeAliasSymbol(nullptr, {});
  symbol->setType(type);
  return symbol;
}

auto Control::getPackArgumentSymbol(std::span<Symbol* const> elements)
    -> ParameterPackSymbol* {
  auto& symbol =
      d->packArgumentSymbols[std::vector(elements.begin(), elements.end())];
  if (symbol) return symbol;
  symbol = newParameterPackSymbol(nullptr, {});
  for (auto element : elements) symbol->addElement(element);
  return symbol;
}

auto Control::getConstantArgumentSymbol(const Type* type,
                                        const ConstValue& value)
    -> VariableSymbol* {
  auto integer = std::get_if<ConstInt>(&value);
  if (!integer) return newConstantArgumentSymbol(this, type, value);

  auto& symbol = d->constantArgumentSymbols[std::make_tuple(
      type, integer->lowBits(), integer->highBits(), integer->width(),
      integer->isSigned())];
  if (!symbol) symbol = newConstantArgumentSymbol(this, type, value);
  return symbol;
}

auto Control::newVariableSymbol(ScopeSymbol* enclosingScope, SourceLocation loc)
    -> VariableSymbol* {
  auto symbol = &d->variableSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}

auto Control::getAttributes(AttributeMap attributes) -> const AttributeMap* {
  if (attributes.empty()) return nullptr;
  return &*d->attributes.insert(std::move(attributes)).first;
}

auto Control::getAbiTags(std::vector<const Identifier*> tags)
    -> const std::vector<const Identifier*>* {
  if (tags.empty()) return nullptr;
  return &*d->abiTags.insert(std::move(tags)).first;
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
                                               SourceLocation loc, int index,
                                               int depth, bool isParameterPack)
    -> ConstraintTypeParameterSymbol* {
  auto symbol =
      &d->constraintTypeParameterSymbols.emplace_front(enclosingScope);
  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setParameterPack(isParameterPack);
  symbol->setType(getTypeParameterType(index, depth, isParameterPack));
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

auto Control::newNamespaceAliasSymbol(ScopeSymbol* enclosingScope,
                                      SourceLocation loc)
    -> NamespaceAliasSymbol* {
  auto symbol = &d->namespaceAliasSymbols.emplace_front(enclosingScope);
  symbol->setLocation(loc);
  return symbol;
}
auto Control::beginCopyConstructorSelection(ClassSymbol* classSymbol) -> bool {
  return d->copyConstructorSelections.insert(classSymbol).second;
}

void Control::endCopyConstructorSelection(ClassSymbol* classSymbol) {
  d->copyConstructorSelections.erase(classSymbol);
}

auto Control::allocatedSymbolCounts() const -> SymbolCounts {
  return d->allocatedSymbolCounts();
}

auto Control::anonymousIdCount() const -> int { return d->anonymousIdCount; }

void Control::setAnonymousIdCount(int count) { d->anonymousIdCount = count; }

auto Control::getAlignValType() const -> const Type* { return d->alignValType; }

void Control::setAlignValType(const Type* type) { d->alignValType = type; }

auto Control::getNothrowType() const -> const Type* { return d->nothrowType; }

void Control::setNothrowType(const Type* type) { d->nothrowType = type; }

}  // namespace cxx
