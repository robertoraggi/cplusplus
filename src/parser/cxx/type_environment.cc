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

#include <cxx/type_environment.h>
#include <cxx/types.h>

#include <unordered_set>

namespace cxx {

namespace {

struct Hash {
  template <typename T>
  [[nodiscard]] auto hash_value(const T& value) const -> std::size_t {
    return std::hash<T>()(value);
  }

  [[nodiscard]] auto hash_value(CharacterKind kind) const -> std::size_t {
    using I = std::underlying_type<CharacterKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  [[nodiscard]] auto hash_value(IntegerKind kind) const -> std::size_t {
    using I = std::underlying_type<IntegerKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  [[nodiscard]] auto hash_value(FloatingPointKind kind) const -> std::size_t {
    using I = std::underlying_type<FloatingPointKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  [[nodiscard]] auto hash_value(Qualifiers qualifiers) const -> std::size_t {
    using I = std::underlying_type<Qualifiers>::type;
    return std::hash<I>()(static_cast<I>(qualifiers));
  }

  template <typename T>
  void _hash_combine(std::size_t& seed, const T& val) const {
    seed ^= hash_value(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  template <typename... Types>
  [[nodiscard]] auto hash_combine(const Types&... args) const -> std::size_t {
    std::size_t seed = 0;
    (_hash_combine(seed, args), ...);
    return seed;
  }

  [[nodiscard]] auto hash_value(const QualifiedType& value) const
      -> std::size_t {
    return hash_combine(value.type(), value.qualifiers());
  }

  auto operator()(const UndefinedType& type) const -> std::size_t { return 0; }

  auto operator()(const VoidType& type) const -> std::size_t { return 0; }

  auto operator()(const NullptrType& type) const -> std::size_t { return 0; }

  auto operator()(const BooleanType& type) const -> std::size_t { return 0; }

  auto operator()(const CharacterType& type) const -> std::size_t {
    return hash_value(type.kind());
  }

  auto operator()(const IntegerType& type) const -> std::size_t {
    return hash_combine(type.kind(), type.isUnsigned());
  }

  auto operator()(const FloatingPointType& type) const -> std::size_t {
    return hash_value(type.kind());
  }

  auto operator()(const EnumType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const ScopedEnumType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const PointerType& type) const -> std::size_t {
    return hash_combine(type.elementType(), type.qualifiers());
  }

  auto operator()(const PointerToMemberType& type) const -> std::size_t {
    return hash_combine(type.classType(), type.elementType(),
                        type.qualifiers());
  }

  auto operator()(const ReferenceType& type) const -> std::size_t {
    return hash_value(type.elementType());
  }

  auto operator()(const RValueReferenceType& type) const -> std::size_t {
    return hash_value(type.elementType());
  }

  auto operator()(const ArrayType& type) const -> std::size_t {
    return hash_combine(type.elementType(), type.dimension());
  }

  auto operator()(const UnboundArrayType& type) const -> std::size_t {
    return hash_value(type.elementType());
  }

  auto operator()(const FunctionType& type) const -> std::size_t {
    std::size_t h = 0;
    h = hash_combine(h, type.returnType());
    for (const auto& argTy : type.argumentTypes()) h = hash_combine(h, argTy);
    h = hash_combine(h, type.isVariadic());
    return h;
  }

  auto operator()(const MemberFunctionType& type) const -> std::size_t {
    std::size_t h = 0;
    h = hash_combine(h, type.classType());
    h = hash_combine(h, type.returnType());
    for (const auto& argTy : type.argumentTypes()) h = hash_combine(h, argTy);
    h = hash_combine(h, type.isVariadic());
    return h;
  }

  auto operator()(const NamespaceType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const ClassType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const TemplateType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const TemplateArgumentType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }

  auto operator()(const ConceptType& type) const -> std::size_t {
    return hash_value(type.symbol());
  }
};

struct EqualTo {
  auto operator()(const UndefinedType& type, const UndefinedType& other) const
      -> bool {
    return true;
  }

  auto operator()(const VoidType& type, const VoidType& other) const -> bool {
    return true;
  }

  auto operator()(const NullptrType& type, const NullptrType& other) const
      -> bool {
    return true;
  }

  auto operator()(const BooleanType& type, const BooleanType& other) const
      -> bool {
    return true;
  }

  auto operator()(const CharacterType& type, const CharacterType& other) const
      -> bool {
    return type.kind() == other.kind();
  }

  auto operator()(const IntegerType& type, const IntegerType& other) const
      -> bool {
    return type.kind() == other.kind() &&
           type.isUnsigned() == other.isUnsigned();
  }

  auto operator()(const FloatingPointType& type,
                  const FloatingPointType& other) const -> bool {
    return type.kind() == other.kind();
  }

  auto operator()(const EnumType& type, const EnumType& other) const -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const ScopedEnumType& type, const ScopedEnumType& other) const
      -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const PointerType& type, const PointerType& other) const
      -> bool {
    return type.elementType() == other.elementType() &&
           type.qualifiers() == other.qualifiers();
  }

  auto operator()(const PointerToMemberType& type,
                  const PointerToMemberType& other) const -> bool {
    return type.classType() == other.classType() &&
           type.elementType() == other.elementType() &&
           type.qualifiers() == other.qualifiers();
  }

  auto operator()(const ReferenceType& type, const ReferenceType& other) const
      -> bool {
    return type.elementType() == other.elementType();
  }

  auto operator()(const RValueReferenceType& type,
                  const RValueReferenceType& other) const -> bool {
    return type.elementType() == other.elementType();
  }

  auto operator()(const ArrayType& type, const ArrayType& other) const -> bool {
    return type.elementType() == other.elementType() &&
           type.dimension() == other.dimension();
  }

  auto operator()(const UnboundArrayType& type,
                  const UnboundArrayType& other) const -> bool {
    return type.elementType() == other.elementType();
  }

  auto operator()(const FunctionType& type, const FunctionType& other) const
      -> bool {
    return type.returnType() == other.returnType() &&
           type.argumentTypes() == other.argumentTypes() &&
           type.isVariadic() == other.isVariadic();
  }

  auto operator()(const MemberFunctionType& type,
                  const MemberFunctionType& other) const -> bool {
    return type.classType() == other.classType() &&
           type.returnType() == other.returnType() &&
           type.argumentTypes() == other.argumentTypes() &&
           type.isVariadic() == other.isVariadic();
  }

  auto operator()(const NamespaceType& type, const NamespaceType& other) const
      -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const ClassType& type, const ClassType& other) const -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const TemplateType& type, const TemplateType& other) const
      -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const TemplateArgumentType& type,
                  const TemplateArgumentType& other) const -> bool {
    return type.symbol() == other.symbol();
  }

  auto operator()(const ConceptType& type, const ConceptType& other) const
      -> bool {
    return type.symbol() == other.symbol();
  }
};

template <typename T>
using TypeSet = std::unordered_set<T, Hash, EqualTo>;

}  // namespace

struct TypeEnvironment::Private {
  TypeSet<CharacterType> characterTypes;
  TypeSet<IntegerType> integerTypes;
  TypeSet<FloatingPointType> floatingPointTypes;
  TypeSet<EnumType> enumTypes;
  TypeSet<ScopedEnumType> scopedEnumTypes;
  TypeSet<PointerType> pointerTypes;
  TypeSet<PointerToMemberType> pointerToMemberTypes;
  TypeSet<ReferenceType> referenceTypes;
  TypeSet<RValueReferenceType> rvalueReferenceTypes;
  TypeSet<ArrayType> arrayTypes;
  TypeSet<UnboundArrayType> unboundArrayTypes;
  TypeSet<FunctionType> functionTypes;
  TypeSet<MemberFunctionType> memberFunctionTypes;
  TypeSet<NamespaceType> namespaceTypes;
  TypeSet<ClassType> classTypes;
  TypeSet<TemplateType> templateTypes;
  TypeSet<TemplateArgumentType> templateArgumentTypes;
  TypeSet<ConceptType> conceptTypes;
};

TypeEnvironment::TypeEnvironment() : d(std::make_unique<Private>()) {}

TypeEnvironment::~TypeEnvironment() = default;

auto TypeEnvironment::undefinedType() -> const UndefinedType* {
  return UndefinedType::get();
}

auto TypeEnvironment::errorType() -> const ErrorType* {
  return ErrorType::get();
}

auto TypeEnvironment::autoType() -> const AutoType* { return AutoType::get(); }

auto TypeEnvironment::decltypeAuto() -> const DecltypeAutoType* {
  return DecltypeAutoType::get();
}

auto TypeEnvironment::voidType() -> const VoidType* { return VoidType::get(); }

auto TypeEnvironment::nullptrType() -> const NullptrType* {
  return NullptrType::get();
}

auto TypeEnvironment::booleanType() -> const BooleanType* {
  return BooleanType::get();
}

auto TypeEnvironment::characterType(CharacterKind kind)
    -> const CharacterType* {
  return &*d->characterTypes.emplace(kind).first;
}

auto TypeEnvironment::integerType(IntegerKind kind, bool isUnsigned)
    -> const IntegerType* {
  return &*d->integerTypes.emplace(kind, isUnsigned).first;
}

auto TypeEnvironment::floatingPointType(FloatingPointKind kind)
    -> const FloatingPointType* {
  return &*d->floatingPointTypes.emplace(kind).first;
}

auto TypeEnvironment::enumType(EnumSymbol* symbol) -> const EnumType* {
  return &*d->enumTypes.emplace(symbol).first;
}

auto TypeEnvironment::scopedEnumType(ScopedEnumSymbol* symbol)
    -> const ScopedEnumType* {
  return &*d->scopedEnumTypes.emplace(symbol).first;
}

auto TypeEnvironment::pointerType(const QualifiedType& elementType,
                                  Qualifiers qualifiers) -> const PointerType* {
  return &*d->pointerTypes.emplace(elementType, qualifiers).first;
}

auto TypeEnvironment::pointerToMemberType(const ClassType* classType,
                                          const QualifiedType& elementType,
                                          Qualifiers qualifiers)
    -> const PointerToMemberType* {
  return &*d->pointerToMemberTypes.emplace(classType, elementType, qualifiers)
               .first;
}

auto TypeEnvironment::referenceType(const QualifiedType& elementType)
    -> const ReferenceType* {
  return &*d->referenceTypes.emplace(elementType).first;
}

auto TypeEnvironment::rvalueReferenceType(const QualifiedType& elementType)
    -> const RValueReferenceType* {
  return &*d->rvalueReferenceTypes.emplace(elementType).first;
}

auto TypeEnvironment::arrayType(const QualifiedType& elementType,
                                std::size_t dimension) -> const ArrayType* {
  return &*d->arrayTypes.emplace(elementType, dimension).first;
}

auto TypeEnvironment::unboundArrayType(const QualifiedType& elementType)
    -> const UnboundArrayType* {
  return &*d->unboundArrayTypes.emplace(elementType).first;
}

auto TypeEnvironment::functionType(const QualifiedType& returnType,
                                   std::vector<QualifiedType> argumentTypes,
                                   bool isVariadic) -> const FunctionType* {
  return &*d->functionTypes
               .emplace(returnType, std::move(argumentTypes), isVariadic)
               .first;
}

auto TypeEnvironment::memberFunctionType(
    const ClassType* classType, const QualifiedType& returnType,
    std::vector<QualifiedType> argumentTypes, bool isVariadic)
    -> const MemberFunctionType* {
  return &*d->memberFunctionTypes
               .emplace(classType, returnType, std::move(argumentTypes),
                        isVariadic)
               .first;
}

auto TypeEnvironment::namespaceType(NamespaceSymbol* symbol)
    -> const NamespaceType* {
  return &*d->namespaceTypes.emplace(symbol).first;
}

auto TypeEnvironment::classType(ClassSymbol* symbol) -> const ClassType* {
  return &*d->classTypes.emplace(symbol).first;
}

auto TypeEnvironment::templateType(TemplateParameterList* symbol)
    -> const TemplateType* {
  return &*d->templateTypes.emplace(symbol).first;
}

auto TypeEnvironment::templateArgumentType(TemplateTypeParameterSymbol* symbol)
    -> const TemplateArgumentType* {
  return &*d->templateArgumentTypes.emplace(symbol).first;
}

auto TypeEnvironment::conceptType(ConceptSymbol* symbol) -> const ConceptType* {
  return &*d->conceptTypes.emplace(symbol).first;
}

}  // namespace cxx