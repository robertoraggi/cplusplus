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
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>
#include <limits>

namespace cxx {
namespace {
auto emptyStorageType(mlir::MLIRContext* ctx) -> mlir::Type {
  return mlir::cxx::ArrayType::get(ctx, mlir::IntegerType::get(ctx, 8), 0);
}
}  // namespace

struct Codegen::ConvertType {
  Codegen& gen;

  [[nodiscard]] auto control() const { return gen.control(); }
  [[nodiscard]] auto memoryLayout() const { return control()->memoryLayout(); }

  auto getExprType() const -> mlir::Type;
  auto getIntType(const Type* type, bool isSigned) -> mlir::Type;
  auto getFloatType(const Type* type) -> mlir::Type;

  auto operator()(const VoidType* type) -> mlir::Type;
  auto operator()(const NullptrType* type) -> mlir::Type;
  auto operator()(const DecltypeAutoType* type) -> mlir::Type;
  auto operator()(const AutoType* type) -> mlir::Type;
  auto operator()(const BoolType* type) -> mlir::Type;
  auto operator()(const SignedCharType* type) -> mlir::Type;
  auto operator()(const ShortIntType* type) -> mlir::Type;
  auto operator()(const IntType* type) -> mlir::Type;
  auto operator()(const LongIntType* type) -> mlir::Type;
  auto operator()(const LongLongIntType* type) -> mlir::Type;
  auto operator()(const Int128Type* type) -> mlir::Type;
  auto operator()(const UnsignedCharType* type) -> mlir::Type;
  auto operator()(const UnsignedShortIntType* type) -> mlir::Type;
  auto operator()(const UnsignedIntType* type) -> mlir::Type;
  auto operator()(const UnsignedLongIntType* type) -> mlir::Type;
  auto operator()(const UnsignedLongLongIntType* type) -> mlir::Type;
  auto operator()(const UnsignedInt128Type* type) -> mlir::Type;
  auto operator()(const CharType* type) -> mlir::Type;
  auto operator()(const Char8Type* type) -> mlir::Type;
  auto operator()(const Char16Type* type) -> mlir::Type;
  auto operator()(const Char32Type* type) -> mlir::Type;
  auto operator()(const WideCharType* type) -> mlir::Type;
  auto operator()(const FloatType* type) -> mlir::Type;
  auto operator()(const DoubleType* type) -> mlir::Type;
  auto operator()(const LongDoubleType* type) -> mlir::Type;
  auto operator()(const Float16Type* type) -> mlir::Type;
  auto operator()(const QualType* type) -> mlir::Type;
  auto operator()(const BoundedArrayType* type) -> mlir::Type;
  auto operator()(const UnboundedArrayType* type) -> mlir::Type;
  auto operator()(const PointerType* type) -> mlir::Type;
  auto operator()(const LvalueReferenceType* type) -> mlir::Type;
  auto operator()(const RvalueReferenceType* type) -> mlir::Type;
  auto operator()(const FunctionType* type) -> mlir::Type;
  auto operator()(const ClassType* type) -> mlir::Type;
  auto operator()(const EnumType* type) -> mlir::Type;
  auto operator()(const ScopedEnumType* type) -> mlir::Type;
  auto operator()(const MemberObjectPointerType* type) -> mlir::Type;
  auto operator()(const MemberFunctionPointerType* type) -> mlir::Type;
  auto getMemberPointerIntType() -> mlir::Type;
  auto operator()(const NamespaceType* type) -> mlir::Type;
  auto operator()(const TypeParameterType* type) -> mlir::Type;
  auto operator()(const TemplateTypeParameterType* type) -> mlir::Type;
  auto operator()(const UnresolvedNameType* type) -> mlir::Type;
  auto operator()(const UnresolvedBoundedArrayType* type) -> mlir::Type;
  auto operator()(const UnresolvedUnderlyingType* type) -> mlir::Type;
  auto operator()(const UnresolvedBuiltinType* type) -> mlir::Type;
  auto operator()(const OverloadSetType* type) -> mlir::Type;
  auto operator()(const BuiltinVaListType* type) -> mlir::Type;
  auto operator()(const BuiltinMetaInfoType* type) -> mlir::Type;
  auto operator()(const BitIntType* type) -> mlir::Type;
  auto operator()(const UnsignedBitIntType* type) -> mlir::Type;
  auto operator()(const UnresolvedBitIntType* type) -> mlir::Type;
};

auto Codegen::convertType(const Type* type) -> mlir::Type {
  if (!type) {
    return mlir::cxx::ExprType::get(context_);
  }

  return visit(ConvertType{*this}, type);
}

auto Codegen::ConvertType::getExprType() const -> mlir::Type {
  return mlir::cxx::ExprType::get(gen.context_);
}

auto Codegen::ConvertType::getIntType(const Type* type, bool isSigned)
    -> mlir::Type {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  return mlir::IntegerType::get(gen.context_, width);
}

auto Codegen::ConvertType::getFloatType(const Type* type) -> mlir::Type {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  switch (width) {
    case 16:
      return mlir::Float16Type::get(gen.context_);
    case 32:
      return mlir::Float32Type::get(gen.context_);
    case 64:
      return mlir::Float64Type::get(gen.context_);
    default:
      return mlir::Float64Type::get(gen.context_);
  }
}

auto Codegen::ConvertType::operator()(const VoidType* type) -> mlir::Type {
  return mlir::cxx::VoidType::get(gen.context_);
}

auto Codegen::ConvertType::operator()(const NullptrType* type) -> mlir::Type {
  auto voidType = mlir::cxx::VoidType::get(gen.context_);
  return mlir::cxx::PointerType::get(gen.context_, voidType);
}

auto Codegen::ConvertType::operator()(const DecltypeAutoType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const AutoType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BoolType* type) -> mlir::Type {
  return mlir::IntegerType::get(gen.context_, 1);
}

auto Codegen::ConvertType::operator()(const SignedCharType* type)
    -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const ShortIntType* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const IntType* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const LongIntType* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const LongLongIntType* type)
    -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const Int128Type* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const UnsignedCharType* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedShortIntType* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedIntType* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedLongIntType* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedLongLongIntType* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const UnsignedInt128Type* type)
    -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const CharType* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const Char8Type* type) -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const Char16Type* type) -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const Char32Type* type) -> mlir::Type {
  return getIntType(type, false);
}

auto Codegen::ConvertType::operator()(const WideCharType* type) -> mlir::Type {
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const FloatType* type) -> mlir::Type {
  return getFloatType(type);
}

auto Codegen::ConvertType::operator()(const DoubleType* type) -> mlir::Type {
  return getFloatType(type);
}

auto Codegen::ConvertType::operator()(const LongDoubleType* type)
    -> mlir::Type {
  return getFloatType(type);
}

auto Codegen::ConvertType::operator()(const Float16Type* type) -> mlir::Type {
  return getFloatType(type);
}

auto Codegen::ConvertType::operator()(const QualType* type) -> mlir::Type {
  return gen.convertType(type->elementType());
}

auto Codegen::ConvertType::operator()(const BoundedArrayType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return mlir::cxx::ArrayType::get(gen.context_, elementType, type->size());
}

auto Codegen::ConvertType::operator()(const UnboundedArrayType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return mlir::cxx::ArrayType::get(gen.context_, elementType, 0);
}

auto Codegen::ConvertType::operator()(const PointerType* type) -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return mlir::cxx::PointerType::get(gen.context_, elementType);
}

auto Codegen::ConvertType::operator()(const LvalueReferenceType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return mlir::cxx::PointerType::get(gen.context_, elementType);
}

auto Codegen::ConvertType::operator()(const RvalueReferenceType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return mlir::cxx::PointerType::get(gen.context_, elementType);
}

auto Codegen::ConvertType::operator()(const FunctionType* type) -> mlir::Type {
  return gen.computeFunctionSignature(type, /*functionSymbol=*/nullptr);
}

auto Codegen::ConvertType::operator()(const ClassType* type) -> mlir::Type {
  auto classSymbol = type->symbol();

  auto ctx = gen.context_;

  if (auto it = gen.classNames_.find(classSymbol);
      it != gen.classNames_.end()) {
    return it->second;
  }

  auto name = to_string(classSymbol->name());
  if (name.empty()) {
    auto loc = type->symbol()->location();
    name = std::format("$class_{}", loc.index());
  }

  if (!classSymbol->templateArguments().empty()) {
    ExternalNameEncoder encoder{gen.translationUnit()};
    name = encoder.encode(type);
  }

  if (classSymbol->isUnion()) {
    name = std::format("union.{}", name);
  }

  mlir::cxx::ClassType classType = mlir::cxx::ClassType::getNamed(ctx, name);

  if (!classType.getBody().empty()) {
    auto loc = classSymbol->location();
    name = std::format("{}.$_{}", name, loc.index());
    classType = mlir::cxx::ClassType::getNamed(ctx, name);
  }

  gen.classNames_[classSymbol] = classType;

  if (classSymbol->templateDeclaration()) {
    return classType;
  }

  std::vector<mlir::Type> memberTypes;

  if (classSymbol->isUnion()) {
    mlir::Type largestMemberType;
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
        auto i8Type = mlir::IntegerType::get(gen.context_, 8);
        auto paddingType = mlir::cxx::ArrayType::get(
            gen.context_, i8Type, unionSize - largestMemberSize);
        memberTypes.push_back(paddingType);
      }
    } else {
      memberTypes.push_back(mlir::IntegerType::get(gen.context_, 8));
    }

  } else {
    memberTypes =
        gen.buildClassMemberTypes(classSymbol, /*includeVirtualBases=*/true);
  }

  classType.setBody(memberTypes);

  return classType;
}

auto Codegen::buildClassMemberTypes(ClassSymbol* classSymbol,
                                    bool includeVirtualBases)
    -> std::vector<mlir::Type> {
  std::vector<mlir::Type> memberTypes;

  auto layout = classSymbol->layout();
  if (!layout) return memberTypes;

  std::map<std::uint32_t, mlir::Type> memberMap;
  std::map<std::uint32_t, std::uint64_t> offsetByIndex;
  std::map<std::uint32_t, ClassSymbol*> pendingBases;

  if (layout->hasDirectVtable()) {
    auto i8Type = mlir::IntegerType::get(context_, 8);
    memberMap[layout->vtableIndex()] =
        mlir::cxx::PointerType::get(context_, i8Type);
    offsetByIndex[layout->vtableIndex()] = 0;
  }

  for (auto base : classSymbol->baseClasses()) {
    if (!includeVirtualBases && base->isVirtual()) continue;
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) continue;

    auto info = layout->getBaseInfo(baseSym);
    if (!info) continue;

    const Type* baseType = base->type();
    if (!baseType) baseType = baseSym->type();

    offsetByIndex[info->index] = info->offset;
    if (unit_->typeTraits().is_empty(baseType)) {
      memberMap[info->index] = emptyStorageType(context_);
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

      offsetByIndex[info->index] = info->offset;
      if (unit_->typeTraits().is_empty(vbaseSym->type())) {
        memberMap[info->index] = emptyStorageType(context_);
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
      memberMap[info->index] = mlir::IntegerType::get(
          context_, static_cast<unsigned>(info->allocUnitSizeBytes * 8));
    } else if (field->isNoUniqueAddress() &&
               unit_->typeTraits().is_empty(field->type())) {
      memberMap[info->index] = emptyStorageType(context_);
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

  if (!memberMap.empty()) {
    memberTypes.resize(memberMap.rbegin()->first + 1);
    for (auto const& [index, type] : memberMap) memberTypes[index] = type;
  }

  return memberTypes;
}

auto Codegen::convertBaseEmbedding(ClassSymbol* baseSymbol,
                                   std::uint64_t availableBytes) -> mlir::Type {
  auto rep = convertBaseSubobjectType(baseSymbol);

  auto layout = baseSymbol->layout();
  if (!layout || layout->virtualBases().empty()) return rep;

  const auto reserve = layout->nonVirtualSize();
  const auto align = layout->nonVirtualAlignment();
  const auto natural = align ? (reserve + align - 1) / align * align : reserve;
  if (natural <= availableBytes) return rep;

  auto i8Type = mlir::IntegerType::get(context_, 8);
  return mlir::cxx::ArrayType::get(context_, i8Type, availableBytes);
}

auto Codegen::convertBaseSubobjectType(ClassSymbol* classSymbol) -> mlir::Type {
  auto layout = classSymbol->layout();

  if (!layout || layout->virtualBases().empty()) {
    return convertType(classSymbol->type());
  }

  if (auto it = baseSubobjectTypeNames_.find(classSymbol);
      it != baseSubobjectTypeNames_.end()) {
    return it->second;
  }

  auto name = std::format("{}.base", to_string(classSymbol->name()));
  auto classType = mlir::cxx::ClassType::getNamed(context_, name);
  if (!classType.getBody().empty()) {
    name = std::format("{}.$_{}", name, classSymbol->location().index());
    classType = mlir::cxx::ClassType::getNamed(context_, name);
  }

  baseSubobjectTypeNames_[classSymbol] = classType;

  classType.setBody(
      buildClassMemberTypes(classSymbol, /*includeVirtualBases=*/false));

  return classType;
}

auto Codegen::ConvertType::operator()(const EnumType* type) -> mlir::Type {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return mlir::IntegerType::get(gen.context_, 32);
}

auto Codegen::ConvertType::operator()(const ScopedEnumType* type)
    -> mlir::Type {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return mlir::IntegerType::get(gen.context_, 32);
}

auto Codegen::ConvertType::getMemberPointerIntType() -> mlir::Type {
  return mlir::IntegerType::get(gen.context_,
                                memoryLayout()->sizeOfPointer() * 8);
}

auto Codegen::ConvertType::operator()(const MemberObjectPointerType* type)
    -> mlir::Type {
  return getMemberPointerIntType();
}

auto Codegen::ConvertType::operator()(const MemberFunctionPointerType* type)
    -> mlir::Type {
  auto classType = mlir::cxx::ClassType::getNamed(gen.context_, "$memberfnptr");

  if (classType.getBody().empty()) {
    auto intType = getMemberPointerIntType();
    (void)classType.setBody({intType, intType});
  }

  return classType;
}

auto Codegen::ConvertType::operator()(const NamespaceType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const TypeParameterType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const TemplateTypeParameterType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedNameType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedBoundedArrayType* type)
    -> mlir::Type {
  return gen.convertType(type->elementType());
}

auto Codegen::ConvertType::operator()(const UnresolvedUnderlyingType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedBuiltinType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const OverloadSetType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BuiltinVaListType* type)
    -> mlir::Type {
  auto voidType = mlir::cxx::VoidType::get(gen.context_);
  return mlir::cxx::PointerType::get(gen.context_, voidType);
}

auto Codegen::ConvertType::operator()(const BuiltinMetaInfoType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BitIntType* type) -> mlir::Type {
  return mlir::IntegerType::get(gen.context_, type->numBits());
}

auto Codegen::ConvertType::operator()(const UnsignedBitIntType* type)
    -> mlir::Type {
  return mlir::IntegerType::get(gen.context_, type->numBits());
}

auto Codegen::ConvertType::operator()(const UnresolvedBitIntType* type)
    -> mlir::Type {
  return getExprType();
}
}  // namespace cxx
