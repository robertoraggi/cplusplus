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

#include <cxx/codegen/codegen.h>
#include <cxx/codegen/debug_emitter.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
struct Codegen::ConvertDebugType {
  Codegen& gen;

  [[nodiscard]] auto control() const { return gen.control(); }
  [[nodiscard]] auto memoryLayout() const { return control()->memoryLayout(); }

  [[nodiscard]] auto operator()(const VoidType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const NullptrType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const DecltypeAutoType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const AutoType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const BoolType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const SignedCharType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const ShortIntType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const IntType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const LongIntType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const LongLongIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const Int128Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedCharType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedShortIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedLongIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedLongLongIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedInt128Type* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const CharType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const Char8Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const Char16Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const Char32Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const WideCharType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const FloatType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const DoubleType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const LongDoubleType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const Float16Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const QualType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const BoundedArrayType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnboundedArrayType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const PointerType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const LvalueReferenceType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const RvalueReferenceType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const FunctionType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const ClassType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const EnumType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const ScopedEnumType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const MemberObjectPointerType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const MemberFunctionPointerType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const NamespaceType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const TypeParameterType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const TemplateTypeParameterType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(
      const TemplateTypeParameterSpecializationType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const PackExpansionType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const DecltypeType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedNameType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedBoundedArrayType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedUnderlyingType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedBuiltinType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const OverloadSetType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const BuiltinVaListType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const BuiltinMetaInfoType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const BitIntType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnsignedBitIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedBitIntType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const VectorType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const UnresolvedVectorType* type)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const ComplexType* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto operator()(const AtomicType* type) -> ir::DebugTypeRef;

  [[nodiscard]] auto basicType(std::string_view name, const Type* type,
                               ir::DebugEncoding encoding) -> ir::DebugTypeRef;

  [[nodiscard]] auto derivedType(ir::DebugDerivedKind tag, const Type* type,
                                 ir::DebugTypeRef baseType,
                                 uint64_t offsetInBits = 0,
                                 std::string_view name = {},
                                 ir::DebugTypeRef extraData = {})
      -> ir::DebugTypeRef;

  [[nodiscard]] auto classType(const ClassType* type,
                               std::span<const ir::DebugTypeRef> elements)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto enumerationType(const Type* type, Symbol* symbol,
                                     const Type* underlyingType, bool scoped)
      -> ir::DebugTypeRef;
  [[nodiscard]] auto arrayType(const Type* type, const Type* elementType,
                               std::uint64_t count) -> ir::DebugTypeRef;
};

[[nodiscard]] auto Codegen::convertDebugType(const Type* type)
    -> ir::DebugTypeRef {
  if (!type) {
    return {};
  }

  if (auto it = debugTypeCache_.find(type); it != debugTypeCache_.end()) {
    return it->second;
  }

  auto result = visit(ConvertDebugType{*this}, type);
  debugTypeCache_.insert({type, result});
  return result;
}

[[nodiscard]] auto Codegen::ConvertDebugType::basicType(
    std::string_view name, const Type* type, ir::DebugEncoding encoding)
    -> ir::DebugTypeRef {
  return gen.debugEmitter_->basicType(
      {name, memoryLayout()->sizeOf(type).value() * 8, encoding});
}

[[nodiscard]] auto Codegen::ConvertDebugType::derivedType(
    ir::DebugDerivedKind tag, const Type* type, ir::DebugTypeRef baseType,
    uint64_t offsetInBits, std::string_view name, ir::DebugTypeRef extraData)
    -> ir::DebugTypeRef {
  if (!baseType && tag != ir::DebugDerivedKind::Pointer) return {};
  return gen.debugEmitter_->derivedType(
      {.kind = tag,
       .baseType = baseType,
       .sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8,
       .alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8,
       .offsetInBits = offsetInBits,
       .name = name,
       .classType = extraData});
}

[[nodiscard]] auto Codegen::ConvertDebugType::classType(
    const ClassType* type, std::span<const ir::DebugTypeRef> elements)
    -> ir::DebugTypeRef {
  auto symbol = type->symbol();
  auto name = to_string(symbol->name());
  return gen.debugEmitter_->compositeType(
      {.kind = symbol->isUnion() ? ir::DebugCompositeKind::Union
                                 : ir::DebugCompositeKind::Structure,
       .name = name,
       .location = gen.debugLocation(symbol->location()),
       .scope = gen.debugCompileUnit_,
       .sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8,
       .alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8,
       .elements = elements});
}

[[nodiscard]] auto Codegen::ConvertDebugType::enumerationType(
    const Type* type, Symbol* symbol, const Type* underlyingType, bool scoped)
    -> ir::DebugTypeRef {
  auto name = to_string(symbol->name());
  return gen.debugEmitter_->compositeType(
      {.kind = ir::DebugCompositeKind::Enumeration,
       .name = name,
       .location = gen.debugLocation(symbol->location()),
       .scope = gen.debugCompileUnit_,
       .baseType = gen.convertDebugType(underlyingType),
       .sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8,
       .alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8,
       .isScopedEnum = scoped});
}

[[nodiscard]] auto Codegen::ConvertDebugType::arrayType(const Type* type,
                                                        const Type* elementType,
                                                        std::uint64_t count)
    -> ir::DebugTypeRef {
  return gen.debugEmitter_->arrayType(
      {.elementType = gen.convertDebugType(elementType),
       .count = count,
       .countBitWidth =
           static_cast<unsigned>(memoryLayout()->sizeOfSizeType() * 8),
       .sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8,
       .alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8});
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const VoidType* type)
    -> ir::DebugTypeRef {
  return gen.debugEmitter_->basicType({.name = "void"});
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const NullptrType* type) -> ir::DebugTypeRef {
  return gen.debugEmitter_->basicType(
      {.name = "decltype(nullptr)",
       .sizeInBits = memoryLayout()->sizeOfPointer() * 8});
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const DecltypeAutoType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const AutoType* type)
    -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const BoolType* type)
    -> ir::DebugTypeRef {
  return basicType("bool", type, ir::DebugEncoding::Boolean);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const SignedCharType* type) -> ir::DebugTypeRef {
  return basicType("signed char", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const ShortIntType* type) -> ir::DebugTypeRef {
  return basicType("short", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const IntType* type)
    -> ir::DebugTypeRef {
  return basicType("int", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const LongIntType* type) -> ir::DebugTypeRef {
  return basicType("long", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const LongLongIntType* type) -> ir::DebugTypeRef {
  return basicType("long long", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const Int128Type* type)
    -> ir::DebugTypeRef {
  return basicType("int128", type, ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedCharType* type) -> ir::DebugTypeRef {
  return basicType("unsigned char", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedShortIntType* type) -> ir::DebugTypeRef {
  return basicType("unsigned short", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedIntType* type) -> ir::DebugTypeRef {
  return basicType("unsigned int", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedLongIntType* type) -> ir::DebugTypeRef {
  return basicType("unsigned long", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedLongLongIntType* type) -> ir::DebugTypeRef {
  return basicType("unsigned long long", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedInt128Type* type) -> ir::DebugTypeRef {
  return basicType("uint128", type, ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const CharType* type)
    -> ir::DebugTypeRef {
  auto isSigned = gen.traits.is_signed(type);
  return basicType(
      "char", type,
      isSigned ? ir::DebugEncoding::Signed : ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const Char8Type* type)
    -> ir::DebugTypeRef {
  return basicType("char8_t", type, ir::DebugEncoding::Utf);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const Char16Type* type)
    -> ir::DebugTypeRef {
  return basicType("char16_t", type, ir::DebugEncoding::Utf);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const Char32Type* type)
    -> ir::DebugTypeRef {
  return basicType("char32_t", type, ir::DebugEncoding::Utf);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const WideCharType* type) -> ir::DebugTypeRef {
  auto isSigned = gen.traits.is_signed(type);
  return basicType(
      "wchar_t", type,
      isSigned ? ir::DebugEncoding::Signed : ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const FloatType* type)
    -> ir::DebugTypeRef {
  return basicType("float", type, ir::DebugEncoding::Float);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const DoubleType* type)
    -> ir::DebugTypeRef {
  return basicType("double", type, ir::DebugEncoding::Float);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const LongDoubleType* type) -> ir::DebugTypeRef {
  return basicType("long double", type, ir::DebugEncoding::Float);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const Float16Type* type) -> ir::DebugTypeRef {
  return basicType("_Float16", type, ir::DebugEncoding::Float);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const QualType* type)
    -> ir::DebugTypeRef {
  auto resultType = gen.convertDebugType(type->elementType());

  if (type->isVolatile()) {
    resultType = derivedType(ir::DebugDerivedKind::Volatile, type, resultType);
  }

  if (type->isConst()) {
    resultType = derivedType(ir::DebugDerivedKind::Const, type, resultType);
  }

  return resultType;
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const BoundedArrayType* type) -> ir::DebugTypeRef {
  return arrayType(type, type->elementType(), type->size());
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnboundedArrayType* type) -> ir::DebugTypeRef {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(ir::DebugDerivedKind::Pointer, type, elementType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const PointerType* type) -> ir::DebugTypeRef {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(ir::DebugDerivedKind::Pointer, type, elementType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const LvalueReferenceType* type) -> ir::DebugTypeRef {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(ir::DebugDerivedKind::Reference, type, elementType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const RvalueReferenceType* type) -> ir::DebugTypeRef {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(ir::DebugDerivedKind::RvalueReference, type, elementType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const FunctionType* type) -> ir::DebugTypeRef {
  return gen.debugSubroutineType(type);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const ClassType* type)
    -> ir::DebugTypeRef {
  auto symbol = type->symbol();
  if (!symbol) return {};

  auto recSelf = classType(type, {});

  gen.debugTypeCache_[type] = recSelf;

  std::vector<ir::DebugTypeRef> elements;

  auto layout = symbol->layout();

  for (auto base : symbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;
    auto baseTypeAttr = gen.convertDebugType(baseClassSymbol->type());
    if (!baseTypeAttr) continue;

    uint64_t baseOffset = 0;
    if (layout) {
      if (auto baseInfo =
              layout->getBaseInfo(baseClassSymbol, base->isVirtual())) {
        baseOffset = baseInfo->offset;
      }
    }

    auto inheritanceAttr =
        derivedType(ir::DebugDerivedKind::Inheritance, baseClassSymbol->type(),
                    baseTypeAttr, baseOffset * 8);
    if (inheritanceAttr) elements.push_back(inheritanceAttr);
  }

  for (auto field :
       cxx::views::members(symbol) | cxx::views::non_static_fields) {
    auto fieldTypeAttr = gen.convertDebugType(field->type());
    if (!fieldTypeAttr) continue;

    uint64_t fieldOffset = 0;
    if (layout) {
      if (auto fieldInfo = layout->getFieldInfo(field)) {
        fieldOffset = fieldInfo->offset;
      }
    }

    auto memberAttr =
        derivedType(ir::DebugDerivedKind::Member, field->type(), fieldTypeAttr,
                    fieldOffset * 8, to_string(field->name()));
    if (memberAttr) elements.push_back(memberAttr);
  }

  auto fullDef = classType(type, elements);

  gen.debugTypeCache_.insert_or_assign(type, fullDef);

  return fullDef;
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const EnumType* type)
    -> ir::DebugTypeRef {
  return enumerationType(type, type->symbol(), type->underlyingType(), false);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const ScopedEnumType* type) -> ir::DebugTypeRef {
  return enumerationType(type, type->symbol(), type->underlyingType(), true);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const MemberObjectPointerType* type) -> ir::DebugTypeRef {
  auto elementType = gen.convertDebugType(type->elementType());
  auto classType = gen.convertDebugType(type->classType());
  if (!elementType || !classType) return {};
  return derivedType(ir::DebugDerivedKind::MemberPointer, type, elementType, 0,
                     {}, classType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const MemberFunctionPointerType* type) -> ir::DebugTypeRef {
  auto functionType = gen.convertDebugType(type->functionType());
  auto classType = gen.convertDebugType(type->classType());
  if (!functionType || !classType) return {};
  return derivedType(ir::DebugDerivedKind::MemberPointer, type, functionType, 0,
                     {}, classType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const NamespaceType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const TypeParameterType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const TemplateTypeParameterType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const TemplateTypeParameterSpecializationType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const PackExpansionType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const DecltypeType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedNameType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedBoundedArrayType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedUnderlyingType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedBuiltinType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const OverloadSetType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const BuiltinVaListType* type) -> ir::DebugTypeRef {
  auto elementType = gen.debugEmitter_->basicType({.name = "void"});
  return derivedType(ir::DebugDerivedKind::Pointer, type, elementType);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const BuiltinMetaInfoType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const BitIntType* type)
    -> ir::DebugTypeRef {
  return basicType(std::format("_BitInt({})", type->numBits()), type,
                   ir::DebugEncoding::Signed);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnsignedBitIntType* type) -> ir::DebugTypeRef {
  return basicType(std::format("unsigned _BitInt({})", type->numBits()), type,
                   ir::DebugEncoding::Unsigned);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedBitIntType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const VectorType* type)
    -> ir::DebugTypeRef {
  return arrayType(type, type->elementType(), type->elementCount());
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const UnresolvedVectorType* type) -> ir::DebugTypeRef {
  return {};
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(
    const ComplexType* type) -> ir::DebugTypeRef {
  return basicType(to_string(type), type, ir::DebugEncoding::ComplexFloat);
}

[[nodiscard]] auto Codegen::ConvertDebugType::operator()(const AtomicType* type)
    -> ir::DebugTypeRef {
  return derivedType(ir::DebugDerivedKind::Atomic, type,
                     gen.convertDebugType(type->elementType()));
}
}  // namespace cxx
