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
#include <cxx/codegen/emitter.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <limits>

namespace cxx {

struct Codegen::ConvertType {
  Codegen& gen;

  [[nodiscard]] auto control() const { return gen.control(); }
  [[nodiscard]] auto memoryLayout() const { return control()->memoryLayout(); }

  auto getExprType() const -> ir::TypeRef;
  auto getIntType(const Type* type, bool isSigned) -> ir::TypeRef;

  auto operator()(const VoidType* type) -> ir::TypeRef;
  auto operator()(const NullptrType* type) -> ir::TypeRef;
  auto operator()(const DecltypeAutoType* type) -> ir::TypeRef;
  auto operator()(const AutoType* type) -> ir::TypeRef;
  auto operator()(const BoolType* type) -> ir::TypeRef;
  auto operator()(const SignedCharType* type) -> ir::TypeRef;
  auto operator()(const ShortIntType* type) -> ir::TypeRef;
  auto operator()(const IntType* type) -> ir::TypeRef;
  auto operator()(const LongIntType* type) -> ir::TypeRef;
  auto operator()(const LongLongIntType* type) -> ir::TypeRef;
  auto operator()(const Int128Type* type) -> ir::TypeRef;
  auto operator()(const UnsignedCharType* type) -> ir::TypeRef;
  auto operator()(const UnsignedShortIntType* type) -> ir::TypeRef;
  auto operator()(const UnsignedIntType* type) -> ir::TypeRef;
  auto operator()(const UnsignedLongIntType* type) -> ir::TypeRef;
  auto operator()(const UnsignedLongLongIntType* type) -> ir::TypeRef;
  auto operator()(const UnsignedInt128Type* type) -> ir::TypeRef;
  auto operator()(const CharType* type) -> ir::TypeRef;
  auto operator()(const Char8Type* type) -> ir::TypeRef;
  auto operator()(const Char16Type* type) -> ir::TypeRef;
  auto operator()(const Char32Type* type) -> ir::TypeRef;
  auto operator()(const WideCharType* type) -> ir::TypeRef;
  auto operator()(const FloatType* type) -> ir::TypeRef;
  auto operator()(const DoubleType* type) -> ir::TypeRef;
  auto operator()(const LongDoubleType* type) -> ir::TypeRef;
  auto operator()(const Float16Type* type) -> ir::TypeRef;
  auto operator()(const QualType* type) -> ir::TypeRef;
  auto operator()(const BoundedArrayType* type) -> ir::TypeRef;
  auto operator()(const UnboundedArrayType* type) -> ir::TypeRef;
  auto operator()(const PointerType* type) -> ir::TypeRef;
  auto operator()(const LvalueReferenceType* type) -> ir::TypeRef;
  auto operator()(const RvalueReferenceType* type) -> ir::TypeRef;
  auto operator()(const FunctionType* type) -> ir::TypeRef;
  auto operator()(const ClassType* type) -> ir::TypeRef;
  auto operator()(const EnumType* type) -> ir::TypeRef;
  auto operator()(const ScopedEnumType* type) -> ir::TypeRef;
  auto operator()(const MemberObjectPointerType* type) -> ir::TypeRef;
  auto operator()(const MemberFunctionPointerType* type) -> ir::TypeRef;
  auto getMemberPointerIntType() -> ir::TypeRef;
  auto operator()(const NamespaceType* type) -> ir::TypeRef;
  auto operator()(const TypeParameterType* type) -> ir::TypeRef;
  auto operator()(const TemplateTypeParameterType* type) -> ir::TypeRef;
  auto operator()(const UnresolvedNameType* type) -> ir::TypeRef;
  auto operator()(const UnresolvedBoundedArrayType* type) -> ir::TypeRef;
  auto operator()(const UnresolvedUnderlyingType* type) -> ir::TypeRef;
  auto operator()(const UnresolvedBuiltinType* type) -> ir::TypeRef;
  auto operator()(const OverloadSetType* type) -> ir::TypeRef;
  auto operator()(const BuiltinVaListType* type) -> ir::TypeRef;
  auto operator()(const BuiltinMetaInfoType* type) -> ir::TypeRef;
  auto operator()(const BitIntType* type) -> ir::TypeRef;
  auto operator()(const UnsignedBitIntType* type) -> ir::TypeRef;
  auto operator()(const UnresolvedBitIntType* type) -> ir::TypeRef;

  auto operator()(const VectorType* type) -> ir::TypeRef;

  auto operator()(const UnresolvedVectorType* type) -> ir::TypeRef;
  auto operator()(const ComplexType* type) -> ir::TypeRef;
  auto operator()(const AtomicType* type) -> ir::TypeRef;
};

auto Codegen::convertType(const Type* type) -> ir::TypeRef {
  if (!type) return emitter_.unresolvedType();

  return visit(ConvertType{*this}, type);
}

auto Codegen::ConvertType::getExprType() const -> ir::TypeRef {
  return gen.emitter_.unresolvedType();
}

auto Codegen::ConvertType::getIntType(const Type* type, bool isSigned)
    -> ir::TypeRef {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  return gen.emitter_.integerType(static_cast<unsigned>(width));
}

auto Codegen::ConvertType::operator()(const VoidType* type) -> ir::TypeRef {
  return gen.emitter_.voidType();
}

auto Codegen::ConvertType::operator()(const NullptrType* type) -> ir::TypeRef {
  return gen.emitter_.pointerType(gen.emitter_.voidType());
}

auto Codegen::ConvertType::operator()(const DecltypeAutoType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const AutoType* type) -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BoolType* type) -> ir::TypeRef {
  return gen.emitter_.integerType(1);
}

auto Codegen::ConvertType::operator()(const SignedCharType* type)
    -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const ShortIntType* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const IntType* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const LongIntType* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const LongLongIntType* type)
    -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const Int128Type* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const UnsignedCharType* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedShortIntType* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedIntType* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedLongIntType* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedLongLongIntType* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedInt128Type* type)
    -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const CharType* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const Char8Type* type) -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const Char16Type* type) -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const Char32Type* type) -> ir::TypeRef {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const WideCharType* type) -> ir::TypeRef {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const FloatType* type) -> ir::TypeRef {
  return gen.emitter_.floatingType(ir::FloatKind::Single);
}

auto Codegen::ConvertType::operator()(const DoubleType* type) -> ir::TypeRef {
  return gen.emitter_.floatingType(ir::FloatKind::Double);
}

auto Codegen::ConvertType::operator()(const LongDoubleType* type)
    -> ir::TypeRef {
  switch (memoryLayout()->longDoubleMantissaDigits()) {
    case 53:
      return gen.emitter_.floatingType(ir::FloatKind::Double);
    case 64:
      return gen.emitter_.floatingType(ir::FloatKind::X87DoubleExtended);
    default:
      return gen.emitter_.floatingType(ir::FloatKind::Quad);
  }
}

auto Codegen::ConvertType::operator()(const Float16Type* type) -> ir::TypeRef {
  return gen.emitter_.floatingType(ir::FloatKind::Half);
}

auto Codegen::ConvertType::operator()(const QualType* type) -> ir::TypeRef {
  return gen.convertType(type->elementType());
}

auto Codegen::ConvertType::operator()(const BoundedArrayType* type)
    -> ir::TypeRef {
  auto elementType = gen.convertType(type->elementType());
  return gen.arrayType(elementType, type->size());
}

auto Codegen::ConvertType::operator()(const UnboundedArrayType* type)
    -> ir::TypeRef {
  auto elementType = gen.convertType(type->elementType());
  return gen.arrayType(elementType, 0);
}

auto Codegen::ConvertType::operator()(const PointerType* type) -> ir::TypeRef {
  auto elementType = gen.convertType(type->elementType());
  return gen.emitter_.pointerType(elementType);
}

auto Codegen::ConvertType::operator()(const LvalueReferenceType* type)
    -> ir::TypeRef {
  auto elementType = gen.convertType(type->elementType());
  return gen.emitter_.pointerType(elementType);
}

auto Codegen::ConvertType::operator()(const RvalueReferenceType* type)
    -> ir::TypeRef {
  auto elementType = gen.convertType(type->elementType());
  return gen.emitter_.pointerType(elementType);
}

auto Codegen::ConvertType::operator()(const FunctionType* type) -> ir::TypeRef {
  return gen.computeFunctionSignature(type, /*functionSymbol=*/nullptr);
}

auto Codegen::uniqueClassTypeName(std::string name) -> std::string {
  if (classTypeNames_.insert(name).second) return name;

  for (std::size_t counter = 1;; ++counter) {
    auto candidate = std::format("{}.{}", name, counter);
    if (classTypeNames_.insert(candidate).second) return candidate;
  }
}

auto Codegen::ConvertType::operator()(const ClassType* type) -> ir::TypeRef {
  auto classSymbol = type->symbol()->resolvedDefinition();

  if (auto it = gen.classNames_.find(classSymbol);
      it != gen.classNames_.end()) {
    return it->second;
  }

  auto name = to_string(classSymbol->name());
  if (name.empty()) {
    auto enclosingClass = symbol_cast<ClassSymbol>(classSymbol->parent());
    auto enclosingName =
        enclosingClass ? gen.className(gen.convertType(
                             enclosingClass->resolvedDefinition()->type()))
                       : std::string_view{};
    name = std::format("{}.$anon.{}", enclosingName,
                       classSymbol->location().index());
  }

  if (!classSymbol->templateArguments().empty()) {
    ExternalNameEncoder encoder{gen.translationUnit()};
    name = encoder.encode(type);
  }

  if (classSymbol->isUnion()) {
    name = std::format("union.{}", name);
  }

  auto classType = gen.declareClassType(gen.uniqueClassTypeName(name),
                                        classSymbol->isUnion());

  gen.classNames_[classSymbol] = classType;

  if (classSymbol->isTemplatePattern()) {
    return classType;
  }

  std::vector<ir::TypeRef> memberTypes;

  if (classSymbol->isUnion()) {
    ir::TypeRef largestMemberType;
    std::size_t largestMemberSize = 0;

    for (auto field : views::members(classSymbol) | views::non_static_fields) {
      auto fieldSizeOpt = memoryLayout()->sizeOf(field->type());
      if (!fieldSizeOpt) continue;
      auto fieldSize = *fieldSizeOpt;
      if (fieldSize > largestMemberSize) {
        largestMemberSize = fieldSize;
        largestMemberType = gen.convertType(field->type());
      }
    }

    if (largestMemberType) {
      memberTypes.push_back(largestMemberType);
      auto unionSize = static_cast<std::size_t>(classSymbol->sizeInBytes());
      if (largestMemberSize < unionSize) {
        memberTypes.push_back(gen.arrayType(gen.emitter_.integerType(8),
                                            unionSize - largestMemberSize));
      }
    } else {
      memberTypes.push_back(gen.emitter_.integerType(8));
    }

    gen.defineClassType(classType, memberTypes, /*isPacked=*/false);

    return classType;
  }

  auto [members, packed] =
      gen.buildClassMemberTypes(classSymbol, /*includeVirtualBases=*/true);

  gen.defineClassType(classType, members, packed);

  return classType;
}

auto Codegen::arrayType(ir::TypeRef elementType, std::uint64_t size)
    -> ir::TypeRef {
  auto type = emitter_.arrayType(elementType, size);
  arraySizes_[type] = size;
  return type;
}

auto Codegen::arraySize(ir::TypeRef type) const -> std::uint64_t {
  auto it = arraySizes_.find(type);
  return it == arraySizes_.end() ? 0 : it->second;
}

auto Codegen::declareClassType(std::string_view name, bool isUnion)
    -> ir::TypeRef {
  auto classType = emitter_.declareClassType(name);
  auto& info = classTypes_[classType];
  info.name = name;
  info.isUnion = isUnion;
  return classType;
}

void Codegen::defineClassType(ir::TypeRef classType,
                              std::span<const ir::TypeRef> members,
                              bool isPacked) {
  emitter_.defineClassType(classType, members, isPacked);
  auto& info = classTypes_[classType];
  info.members.assign(members.begin(), members.end());
  info.isPacked = isPacked;
  info.isDefined = true;
}

auto Codegen::classMembers(ir::TypeRef classType) const
    -> std::span<const ir::TypeRef> {
  auto it = classTypes_.find(classType);
  if (it == classTypes_.end()) return {};
  return it->second.members;
}

auto Codegen::isClassTypeDefined(ir::TypeRef classType) const -> bool {
  auto it = classTypes_.find(classType);
  return it != classTypes_.end() && it->second.isDefined;
}

auto Codegen::isUnionClassType(ir::TypeRef classType) const -> bool {
  auto it = classTypes_.find(classType);
  return it != classTypes_.end() && it->second.isUnion;
}

auto Codegen::className(ir::TypeRef classType) const -> std::string_view {
  auto it = classTypes_.find(classType);
  return it == classTypes_.end() ? std::string_view{} : it->second.name;
}

auto Codegen::storageAlignment(ir::TypeRef type, std::uint64_t pointerSize)
    -> std::uint64_t {
  const auto roundUpToPowerOfTwo = [](std::uint64_t bytes) {
    std::uint64_t alignment = 1;
    while (alignment < bytes) alignment *= 2;
    return alignment;
  };

  switch (emitter_.typeKind(type)) {
    case ir::TypeKind::Integer:
    case ir::TypeKind::Floating:
      return roundUpToPowerOfTwo((emitter_.scalarWidth(type) + 7) / 8);

    case ir::TypeKind::Pointer:
      return pointerSize;

    case ir::TypeKind::Array:
      return storageAlignment(emitter_.elementType(type), pointerSize);

    case ir::TypeKind::Class: {
      auto it = classTypes_.find(type);
      if (it == classTypes_.end()) return 1;
      if (it->second.isPacked) return 1;

      std::uint64_t alignment = 1;
      for (auto member : it->second.members)
        alignment = std::max(alignment, storageAlignment(member, pointerSize));
      return alignment;
    }

    default:
      return 1;
  }
}

auto Codegen::buildClassMemberTypes(ClassSymbol* classSymbol,
                                    bool includeVirtualBases)
    -> ClassMemberTypes {
  ClassMemberTypes result;

  auto layout = classSymbol->layout();
  if (!layout) return result;

  auto emptyStorageType = this->arrayType(emitter_.integerType(8), 0);

  std::map<std::uint32_t, ir::TypeRef> memberMap;
  std::map<std::uint32_t, std::uint64_t> offsetByIndex;
  std::map<std::uint32_t, ClassSymbol*> pendingBases;

  if (layout->hasDirectVtable()) {
    auto i8Type = emitter_.integerType(8);
    memberMap[layout->vtableIndex()] = emitter_.pointerType(i8Type);
    offsetByIndex[layout->vtableIndex()] = 0;
  }

  for (auto base : classSymbol->baseClasses()) {
    if (!includeVirtualBases && base->isVirtual()) continue;
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) continue;

    auto info = layout->getBaseInfo(baseSym);
    if (!info) continue;
    if (layout->primaryBaseIsVirtual() && layout->primaryBase() == baseSym)
      continue;

    const Type* baseType = base->type();
    if (!baseType) baseType = baseSym->type();

    offsetByIndex[info->index] = info->offset;
    if (baseSym->layout() && baseSym->layout()->isAbiEmpty()) {
      memberMap[info->index] = emptyStorageType;
    } else {
      pendingBases[info->index] = baseSym;
    }
  }

  if (includeVirtualBases) {
    for (auto vbaseSym : layout->virtualBases()) {
      auto info = layout->getBaseInfo(vbaseSym);
      if (!info || memberMap.contains(info->index) ||
          pendingBases.contains(info->index))
        continue;
      if (layout->primaryBaseIsVirtual() && layout->primaryBase() == vbaseSym)
        continue;

      offsetByIndex[info->index] = info->offset;
      if (vbaseSym->layout() && vbaseSym->layout()->isAbiEmpty()) {
        memberMap[info->index] = emptyStorageType;
      } else {
        pendingBases[info->index] = vbaseSym;
      }
    }
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto info = layout->getFieldInfo(field);
    if (!info) continue;
    if (memberMap.contains(info->index) || pendingBases.contains(info->index))
      continue;

    offsetByIndex[info->index] = info->offset;
    if (info->bitWidth > 0 && info->allocUnitSizeBytes > 0) {
      memberMap[info->index] = emitter_.integerType(
          static_cast<unsigned>(info->allocUnitSizeBytes * 8));
    } else if (field->isNoUniqueAddress()) {
      ClassSymbol* fieldSymbol = nullptr;
      if (auto fieldClass = unqualified_cast<ClassType>(field->type())) {
        if (fieldClass->symbol()) {
          fieldSymbol = fieldClass->symbol()->resolvedDefinition();
        }
      }

      const ClassLayout* fieldLayout = nullptr;
      if (fieldSymbol) fieldLayout = fieldSymbol->layout();

      if (fieldLayout && fieldLayout->isAbiEmpty()) {
        memberMap[info->index] = emptyStorageType;
      } else if (fieldLayout && !fieldLayout->virtualBases().empty()) {
        pendingBases[info->index] = fieldSymbol;
      } else {
        memberMap[info->index] = convertType(field->type());
      }
    } else {
      memberMap[info->index] = convertType(field->type());
    }
  }

  for (auto const& [index, baseSym] : pendingBases) {
    auto next = offsetByIndex.upper_bound(index);
    auto available = next != offsetByIndex.end()
                         ? next->second - offsetByIndex[index]
                         : std::numeric_limits<std::uint64_t>::max();
    memberMap[index] = convertBaseEmbedding(baseSym, available);
  }

  const auto recordLimit =
      includeVirtualBases ? layout->size() : layout->nonVirtualSize();

  auto i8Type = emitter_.integerType(8);
  for (auto const& padding : layout->padding()) {
    if (padding.offset >= recordLimit) continue;
    const auto sizeInBytes =
        std::min(padding.sizeInBytes, recordLimit - padding.offset);
    memberMap[padding.index] =
        sizeInBytes == 1
            ? i8Type
            : this->arrayType(i8Type, static_cast<std::uint64_t>(sizeInBytes));
  }

  if (!memberMap.empty()) {
    result.members.resize(memberMap.rbegin()->first + 1);
    for (auto const& [index, type] : memberMap) result.members[index] = type;
  }

  const auto pointerSize = control()->memoryLayout()->sizeOfPointer();

  std::uint64_t recordAlignment = 1;
  for (auto const& [index, type] : memberMap) {
    const auto alignment = this->storageAlignment(type, pointerSize);
    recordAlignment = std::max(recordAlignment, alignment);
    auto offset = offsetByIndex.find(index);
    if (offset == offsetByIndex.end()) continue;
    if (offset->second % alignment != 0) result.packed = true;
  }

  if (recordLimit % recordAlignment != 0) result.packed = true;

  return result;
}

auto Codegen::convertBaseEmbedding(ClassSymbol* baseSymbol,
                                   std::uint64_t availableBytes)
    -> ir::TypeRef {
  auto rep = convertBaseSubobjectType(baseSymbol);

  auto layout = baseSymbol->layout();
  if (!layout || layout->virtualBases().empty()) return rep;

  if (layout->nonVirtualSize() <= availableBytes) return rep;

  auto i8Type = emitter_.integerType(8);
  return this->arrayType(i8Type, availableBytes);
}

auto Codegen::convertBaseSubobjectType(ClassSymbol* classSymbol)
    -> ir::TypeRef {
  classSymbol = classSymbol->resolvedDefinition();

  auto layout = classSymbol->layout();

  if (!layout) return convertType(classSymbol->type());

  if (layout->virtualBases().empty() &&
      layout->nonVirtualSize() >= layout->size()) {
    return convertType(classSymbol->type());
  }

  if (auto it = baseSubobjectTypeNames_.find(classSymbol);
      it != baseSubobjectTypeNames_.end()) {
    return it->second;
  }

  auto name = std::format("{}.base", to_string(classSymbol->name()));

  auto classType =
      declareClassType(uniqueClassTypeName(name), /*isUnion=*/false);

  baseSubobjectTypeNames_[classSymbol] = classType;

  auto [members, packed] =
      buildClassMemberTypes(classSymbol, /*includeVirtualBases=*/false);

  this->defineClassType(classType, members, packed);

  return classType;
}

auto Codegen::ConvertType::operator()(const EnumType* type) -> ir::TypeRef {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return gen.emitter_.integerType(32);
}

auto Codegen::ConvertType::operator()(const ScopedEnumType* type)
    -> ir::TypeRef {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return gen.emitter_.integerType(32);
}

auto Codegen::ConvertType::getMemberPointerIntType() -> ir::TypeRef {
  return gen.pointerSizedIntType();
}

auto Codegen::ConvertType::operator()(const MemberObjectPointerType* type)
    -> ir::TypeRef {
  return getMemberPointerIntType();
}

auto Codegen::ConvertType::operator()(const MemberFunctionPointerType* type)
    -> ir::TypeRef {
  auto classType = gen.declareClassType("$memberfnptr", /*isUnion=*/false);

  if (!gen.isClassTypeDefined(classType)) {
    auto intType = getMemberPointerIntType();
    const ir::TypeRef members[] = {intType, intType};
    gen.defineClassType(classType, members, /*isPacked=*/false);
  }

  return classType;
}

auto Codegen::ConvertType::operator()(const NamespaceType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const TypeParameterType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const TemplateTypeParameterType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedNameType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedBoundedArrayType* type)
    -> ir::TypeRef {
  return gen.convertType(type->elementType());
}

auto Codegen::ConvertType::operator()(const UnresolvedUnderlyingType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedBuiltinType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const OverloadSetType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BuiltinVaListType* type)
    -> ir::TypeRef {
  return gen.emitter_.pointerType(gen.emitter_.voidType());
}

auto Codegen::ConvertType::operator()(const BuiltinMetaInfoType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BitIntType* type) -> ir::TypeRef {
  return gen.emitter_.integerType(type->numBits());
}

auto Codegen::ConvertType::operator()(const UnsignedBitIntType* type)
    -> ir::TypeRef {
  return gen.emitter_.integerType(type->numBits());
}

auto Codegen::ConvertType::operator()(const UnresolvedBitIntType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const VectorType* type) -> ir::TypeRef {
  return gen.emitter_.vectorType(gen.convertType(type->elementType()),
                                 type->elementCount());
}

auto Codegen::ConvertType::operator()(const UnresolvedVectorType* type)
    -> ir::TypeRef {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const ComplexType* type) -> ir::TypeRef {
  auto classType = gen.declareClassType(
      std::format("$complex.{}", to_string(type->elementType())),
      /*isUnion=*/false);

  if (!gen.isClassTypeDefined(classType)) {
    auto elementType = gen.convertType(type->elementType());
    const ir::TypeRef members[] = {elementType, elementType};
    gen.defineClassType(classType, members, /*isPacked=*/false);
  }

  return classType;
}

auto Codegen::ConvertType::operator()(const AtomicType* type) -> ir::TypeRef {
  return gen.convertType(type->elementType());
}
}  // namespace cxx
