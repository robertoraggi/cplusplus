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

#include <cxx/type_environment.h>
#include <cxx/types.h>

#include <unordered_set>

namespace cxx {

namespace {

struct Hash {
  template <typename T>
  std::size_t hash_value(const T& value) const {
    return std::hash<T>()(value);
  }

  std::size_t hash_value(CharacterKind kind) const {
    using I = std::underlying_type<CharacterKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  std::size_t hash_value(IntegerKind kind) const {
    using I = std::underlying_type<IntegerKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  std::size_t hash_value(FloatingPointKind kind) const {
    using I = std::underlying_type<FloatingPointKind>::type;
    return std::hash<I>()(static_cast<I>(kind));
  }

  std::size_t hash_value(Qualifiers qualifiers) const {
    using I = std::underlying_type<Qualifiers>::type;
    return std::hash<I>()(static_cast<I>(qualifiers));
  }

  template <typename T>
  void _hash_combine(std::size_t& seed, const T& val) const {
    seed ^= hash_value(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  template <typename... Types>
  std::size_t hash_combine(const Types&... args) const {
    std::size_t seed = 0;
    (_hash_combine(seed, args), ...);
    return seed;
  }

  std::size_t hash_value(const QualifiedType& value) const {
    return hash_combine(value.type(), value.qualifiers());
  }

  std::size_t operator()(const UndefinedType& type) const { return 0; }

  std::size_t operator()(const UnresolvedType& type) const { return 0; }

  std::size_t operator()(const VoidType& type) const { return 0; }

  std::size_t operator()(const NullptrType& type) const { return 0; }

  std::size_t operator()(const BooleanType& type) const { return 0; }

  std::size_t operator()(const CharacterType& type) const {
    return hash_value(type.kind());
  }

  std::size_t operator()(const IntegerType& type) const {
    return hash_combine(type.kind(), type.isUnsigned());
  }

  std::size_t operator()(const FloatingPointType& type) const {
    return hash_value(type.kind());
  }

  std::size_t operator()(const EnumType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const ScopedEnumType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const PointerType& type) const {
    return hash_value(type.elementType());
  }

  std::size_t operator()(const PointerToMemberType& type) const {
    return hash_combine(type.classType(), type.elementType());
  }

  std::size_t operator()(const ReferenceType& type) const {
    return hash_value(type.elementType());
  }

  std::size_t operator()(const RValueReferenceType& type) const {
    return hash_value(type.elementType());
  }

  std::size_t operator()(const ArrayType& type) const {
    return hash_combine(type.elementType(), type.dimension());
  }

  std::size_t operator()(const UnboundArrayType& type) const {
    return hash_value(type.elementType());
  }

  std::size_t operator()(const FunctionType& type) const {
    std::size_t h = 0;
    hash_combine(h, type.returnType());
    for (const auto& argTy : type.argumentTypes()) hash_combine(h, argTy);
    hash_combine(h, type.isVariadic());
    return h;
  }

  std::size_t operator()(const MemberFunctionType& type) const {
    std::size_t h = 0;
    hash_combine(h, type.classType());
    hash_combine(h, type.returnType());
    for (const auto& argTy : type.argumentTypes()) hash_combine(h, argTy);
    hash_combine(h, type.isVariadic());
    return h;
  }

  std::size_t operator()(const NamespaceType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const ClassType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const TemplateType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const TemplateArgumentType& type) const {
    return hash_value(type.symbol());
  }

  std::size_t operator()(const ConceptType& type) const {
    return hash_value(type.symbol());
  }
};

struct EqualTo {
  bool operator()(const UndefinedType& type, const UndefinedType& other) const {
    return true;
  }

  bool operator()(const UnresolvedType& type,
                  const UnresolvedType& other) const {
    return true;
  }

  bool operator()(const VoidType& type, const VoidType& other) const {
    return true;
  }

  bool operator()(const NullptrType& type, const NullptrType& other) const {
    return true;
  }

  bool operator()(const BooleanType& type, const BooleanType& other) const {
    return true;
  }

  bool operator()(const CharacterType& type, const CharacterType& other) const {
    return type.kind() == other.kind();
  }

  bool operator()(const IntegerType& type, const IntegerType& other) const {
    return type.kind() == other.kind() &&
           type.isUnsigned() == other.isUnsigned();
  }

  bool operator()(const FloatingPointType& type,
                  const FloatingPointType& other) const {
    return type.kind() == other.kind();
  }

  bool operator()(const EnumType& type, const EnumType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const ScopedEnumType& type,
                  const ScopedEnumType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const PointerType& type, const PointerType& other) const {
    return type.elementType() == other.elementType();
  }

  bool operator()(const PointerToMemberType& type,
                  const PointerToMemberType& other) const {
    return type.classType() == other.classType() &&
           type.elementType() == other.elementType();
  }

  bool operator()(const ReferenceType& type, const ReferenceType& other) const {
    return type.elementType() == other.elementType();
  }

  bool operator()(const RValueReferenceType& type,
                  const RValueReferenceType& other) const {
    return type.elementType() == other.elementType();
  }

  bool operator()(const ArrayType& type, const ArrayType& other) const {
    return type.elementType() == other.elementType() &&
           type.dimension() == other.dimension();
  }

  bool operator()(const UnboundArrayType& type,
                  const UnboundArrayType& other) const {
    return type.elementType() == other.elementType();
  }

  bool operator()(const FunctionType& type, const FunctionType& other) const {
    return type.returnType() == other.returnType() &&
           type.argumentTypes() == other.argumentTypes() &&
           type.isVariadic() == other.isVariadic();
  }

  bool operator()(const MemberFunctionType& type,
                  const MemberFunctionType& other) const {
    return type.classType() == other.classType() &&
           type.returnType() == other.returnType() &&
           type.argumentTypes() == other.argumentTypes() &&
           type.isVariadic() == other.isVariadic();
  }

  bool operator()(const NamespaceType& type, const NamespaceType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const ClassType& type, const ClassType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const TemplateType& type, const TemplateType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const TemplateArgumentType& type,
                  const TemplateArgumentType& other) const {
    return type.symbol() == other.symbol();
  }

  bool operator()(const ConceptType& type, const ConceptType& other) const {
    return type.symbol() == other.symbol();
  }
};

template <typename T>
using TypeSet = std::unordered_set<T, Hash, EqualTo>;

}  // namespace

struct TypeEnvironment::Private {
  UnresolvedType unresolvedType;
  VoidType voidType;
  NullptrType nullptrType;
  BooleanType booleanType;
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

TypeEnvironment::~TypeEnvironment() {}

const UndefinedType* TypeEnvironment::undefinedType() {
  return UndefinedType::get();
}

const ErrorType* TypeEnvironment::errorType() { return ErrorType::get(); }

const UnresolvedType* TypeEnvironment::unresolvedType() {
  return &d->unresolvedType;
}

const VoidType* TypeEnvironment::voidType() { return &d->voidType; }

const NullptrType* TypeEnvironment::nullptrType() { return &d->nullptrType; }

const BooleanType* TypeEnvironment::booleanType() { return &d->booleanType; }

const CharacterType* TypeEnvironment::characterType(CharacterKind kind) {
  return &*d->characterTypes.emplace(kind).first;
}

const IntegerType* TypeEnvironment::integerType(IntegerKind kind,
                                                bool isUnsigned) {
  return &*d->integerTypes.emplace(kind, isUnsigned).first;
}

const FloatingPointType* TypeEnvironment::floatingPointType(
    FloatingPointKind kind) {
  return &*d->floatingPointTypes.emplace(kind).first;
}

const EnumType* TypeEnvironment::enumType(EnumSymbol* symbol) {
  return &*d->enumTypes.emplace(symbol).first;
}

const ScopedEnumType* TypeEnvironment::scopedEnumType(
    ScopedEnumSymbol* symbol) {
  return &*d->scopedEnumTypes.emplace(symbol).first;
}

const PointerType* TypeEnvironment::pointerType(
    const QualifiedType& elementType) {
  return &*d->pointerTypes.emplace(elementType).first;
}

const PointerToMemberType* TypeEnvironment::pointerToMemberType(
    const ClassType* classType, const QualifiedType& elementType) {
  return &*d->pointerToMemberTypes.emplace(classType, elementType).first;
}

const ReferenceType* TypeEnvironment::referenceType(
    const QualifiedType& elementType) {
  return &*d->referenceTypes.emplace(elementType).first;
}

const RValueReferenceType* TypeEnvironment::rvalueReferenceType(
    const QualifiedType& elementType) {
  return &*d->rvalueReferenceTypes.emplace(elementType).first;
}

const ArrayType* TypeEnvironment::arrayType(const QualifiedType& elementType,
                                            std::size_t dimension) {
  return &*d->arrayTypes.emplace(elementType, dimension).first;
}

const UnboundArrayType* TypeEnvironment::unboundArrayType(
    const QualifiedType& elementType) {
  return &*d->unboundArrayTypes.emplace(elementType).first;
}

const FunctionType* TypeEnvironment::functionType(
    const QualifiedType& returnType, std::vector<QualifiedType> argumentTypes,
    bool isVariadic) {
  return &*d->functionTypes
               .emplace(returnType, std::move(argumentTypes), isVariadic)
               .first;
}

const MemberFunctionType* TypeEnvironment::memberFunctionType(
    const ClassType* classType, const QualifiedType& returnType,
    std::vector<QualifiedType> argumentTypes, bool isVariadic) {
  return &*d->memberFunctionTypes
               .emplace(classType, returnType, std::move(argumentTypes),
                        isVariadic)
               .first;
}

const NamespaceType* TypeEnvironment::namespaceType(NamespaceSymbol* symbol) {
  return &*d->namespaceTypes.emplace(symbol).first;
}

const ClassType* TypeEnvironment::classType(ClassSymbol* symbol) {
  return &*d->classTypes.emplace(symbol).first;
}

const TemplateType* TypeEnvironment::templateType(TemplateClassSymbol* symbol) {
  return &*d->templateTypes.emplace(symbol).first;
}

const TemplateArgumentType* TypeEnvironment::templateArgumentType(
    TemplateTypeParameterSymbol* symbol) {
  return &*d->templateArgumentTypes.emplace(symbol).first;
}

const ConceptType* TypeEnvironment::conceptType(ConceptSymbol* symbol) {
  return &*d->conceptTypes.emplace(symbol).first;
}

}  // namespace cxx