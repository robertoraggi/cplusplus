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

#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>

// cxx
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <format>

namespace cxx {

struct Codegen::ConvertDebugType {
  Codegen& gen;

  [[nodiscard]] auto control() const { return gen.control(); }
  [[nodiscard]] auto memoryLayout() const { return control()->memoryLayout(); }

  auto operator()(const VoidType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const NullptrType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const DecltypeAutoType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const AutoType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const BoolType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const SignedCharType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const ShortIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const IntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const LongIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const LongLongIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const Int128Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedCharType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedShortIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedLongIntType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedLongLongIntType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnsignedInt128Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const CharType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const Char8Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const Char16Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const Char32Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const WideCharType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const FloatType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const DoubleType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const LongDoubleType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const Float16Type* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const QualType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const BoundedArrayType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnboundedArrayType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const PointerType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const LvalueReferenceType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const RvalueReferenceType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const FunctionType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const ClassType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const EnumType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const ScopedEnumType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const MemberObjectPointerType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const MemberFunctionPointerType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const NamespaceType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const TypeParameterType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const TemplateTypeParameterType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnresolvedNameType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnresolvedBoundedArrayType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const UnresolvedUnderlyingType* type)
      -> mlir::LLVM::DITypeAttr;
  auto operator()(const OverloadSetType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const BuiltinVaListType* type) -> mlir::LLVM::DITypeAttr;
  auto operator()(const BuiltinMetaInfoType* type) -> mlir::LLVM::DITypeAttr;

  auto basicType(const llvm::Twine& name, const Type* type, unsigned encoding)
      -> mlir::LLVM::DITypeAttr;

  auto derivedType(unsigned tag, const Type* type,
                   mlir::LLVM::DITypeAttr baseType, uint64_t offsetInBits = 0,
                   const llvm::Twine& name = {}) -> mlir::LLVM::DITypeAttr;

  auto compositeType(unsigned tag, const llvm::Twine& name,
                     mlir::LLVM::DITypeAttr baseType,
                     llvm::ArrayRef<mlir::LLVM::DINodeAttr> elements,
                     const Type* type = nullptr) -> mlir::LLVM::DITypeAttr;

  [[nodiscard]] auto context() const -> mlir::MLIRContext* {
    return gen.context_;
  }
};

auto Codegen::convertDebugType(const Type* type) -> mlir::LLVM::DITypeAttr {
  if (!type) {
    return {};
  }

  if (auto it = debugTypeCache_.find(type); it != debugTypeCache_.end()) {
    return it->second;
  }

  // Pre-insert null to handle infinite recursion if not handled by visitors
  // But strictly, visitors for recursive types should handle it.
  // We can't insert null because we need a return value.
  // So we rely on specific visitors (ClassType) to insert FwdDecl.

  auto result = visit(ConvertDebugType{*this}, type);
  debugTypeCache_.insert({type, result});
  return result;
}

auto Codegen::ConvertDebugType::basicType(const llvm::Twine& name,
                                          const Type* type, unsigned encoding)
    -> mlir::LLVM::DITypeAttr {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_base_type, name,
      memoryLayout()->sizeOf(type).value() * 8, encoding);
}

auto Codegen::ConvertDebugType::derivedType(unsigned tag, const Type* type,
                                            mlir::LLVM::DITypeAttr baseType,
                                            uint64_t offsetInBits,
                                            const llvm::Twine& name)
    -> mlir::LLVM::DITypeAttr {
  auto sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8;
  auto alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8;

  if (!baseType && tag != llvm::dwarf::DW_TAG_pointer_type &&
      tag != llvm::dwarf::DW_TAG_structure_type) {
    return {};
  }

  return mlir::LLVM::DIDerivedTypeAttr::get(
      context(), tag,
      name.isTriviallyEmpty() ? mlir::StringAttr::get(context(), "")
                              : mlir::StringAttr::get(context(), name.str()),
      baseType, sizeInBits, alignInBits, offsetInBits, {},
#if LLVM_VERSION_MAJOR < 22
      /*extraData=*/{}
#else
      /*flags=*/{}, /*extraData=*/{}
#endif
  );
}

auto Codegen::ConvertDebugType::compositeType(
    unsigned tag, const llvm::Twine& name, mlir::LLVM::DITypeAttr baseType,
    llvm::ArrayRef<mlir::LLVM::DINodeAttr> elements, const Type* type)
    -> mlir::LLVM::DITypeAttr {
  mlir::LLVM::DIFileAttr file{};
  uint32_t line{};
  mlir::LLVM::DIScopeAttr scope{};
  mlir::LLVM::DIFlags flags{};
  uint64_t sizeInBits{};
  uint64_t alignInBits{};
  if (type) {
    sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8;
    alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8;

    if (auto classType = type_cast<ClassType>(type)) {
      if (auto symbol = classType->symbol()) {
        auto loc = symbol->location();
        auto [filename, l, c] = gen.unit_->tokenStartPosition(loc);
        file = gen.getFileAttr(filename);
        line = l;
        scope = gen.getCompileUnitAttr(filename);
      }
    }
  }
  mlir::LLVM::DIExpressionAttr dataLocation{};
  mlir::LLVM::DIExpressionAttr rank{};
  mlir::LLVM::DIExpressionAttr allocated{};
  mlir::LLVM::DIExpressionAttr associated{};

  return mlir::LLVM::DICompositeTypeAttr::get(
      context(), tag, mlir::StringAttr::get(context(), name.str()), file, line,
      scope, baseType, flags, sizeInBits, alignInBits,
#if LLVM_VERSION_MAJOR < 22
      elements, dataLocation, rank, allocated, associated
#else
      dataLocation, rank, allocated, associated, elements
#endif
  );
}

auto Codegen::ConvertDebugType::operator()(const VoidType* type)
    -> mlir::LLVM::DITypeAttr {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_unspecified_type, "void", 0, 0);
}

auto Codegen::ConvertDebugType::operator()(const NullptrType* type)
    -> mlir::LLVM::DITypeAttr {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_unspecified_type, "decltype(nullptr)",
      memoryLayout()->sizeOfPointer() * 8, 0);
}

auto Codegen::ConvertDebugType::operator()(const DecltypeAutoType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const AutoType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const BoolType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("bool", type, llvm::dwarf::DW_ATE_boolean);
}

auto Codegen::ConvertDebugType::operator()(const SignedCharType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("signed char", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const ShortIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("short", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const IntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("int", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const LongIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("long", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const LongLongIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("long long", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const Int128Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("int128", type, llvm::dwarf::DW_ATE_signed);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedCharType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("unsigned char", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedShortIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("unsigned short", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("unsigned int", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedLongIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("unsigned long", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedLongLongIntType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("unsigned long long", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const UnsignedInt128Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("uint128", type, llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const CharType* type)
    -> mlir::LLVM::DITypeAttr {
  // todo: toolchain specific
  auto isSigned = control()->is_signed(type);
  return basicType(
      "char", type,
      isSigned ? llvm::dwarf::DW_ATE_signed : llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const Char8Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("char8_t", type, llvm::dwarf::DW_ATE_UTF);
}

auto Codegen::ConvertDebugType::operator()(const Char16Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("char16_t", type, llvm::dwarf::DW_ATE_UTF);
}

auto Codegen::ConvertDebugType::operator()(const Char32Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("char32_t", type, llvm::dwarf::DW_ATE_UTF);
}

auto Codegen::ConvertDebugType::operator()(const WideCharType* type)
    -> mlir::LLVM::DITypeAttr {
  auto isSigned = control()->is_signed(type);
  return basicType(
      "wchar_t", type,
      isSigned ? llvm::dwarf::DW_ATE_signed : llvm::dwarf::DW_ATE_unsigned);
}

auto Codegen::ConvertDebugType::operator()(const FloatType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("float", type, llvm::dwarf::DW_ATE_float);
}

auto Codegen::ConvertDebugType::operator()(const DoubleType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("double", type, llvm::dwarf::DW_ATE_float);
}

auto Codegen::ConvertDebugType::operator()(const LongDoubleType* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("long double", type, llvm::dwarf::DW_ATE_float);
}

auto Codegen::ConvertDebugType::operator()(const Float16Type* type)
    -> mlir::LLVM::DITypeAttr {
  return basicType("_Float16", type, llvm::dwarf::DW_ATE_float);
}

auto Codegen::ConvertDebugType::operator()(const QualType* type)
    -> mlir::LLVM::DITypeAttr {
  auto resultType = gen.convertDebugType(type->elementType());

  if (type->isVolatile()) {
    resultType =
        derivedType(llvm::dwarf::DW_TAG_volatile_type, type, resultType);
  }

  if (type->isConst()) {
    resultType = derivedType(llvm::dwarf::DW_TAG_const_type, type, resultType);
  }

  return resultType;
}

auto Codegen::ConvertDebugType::operator()(const BoundedArrayType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = gen.convertDebugType(type->elementType());

  mlir::Attribute count = mlir::IntegerAttr::get(
      mlir::IntegerType::get(context(),
                             control()->memoryLayout()->sizeOfSizeType() * 8),
      type->size());

  mlir::Attribute lowerBound{};
  mlir::Attribute upperBound{};
  mlir::Attribute stride{};

  auto subrange = mlir::LLVM::DISubrangeAttr::get(context(), count, lowerBound,
                                                  upperBound, stride);

  mlir::SmallVector<mlir::LLVM::DINodeAttr> elements{
      subrange,
  };

  return compositeType(llvm::dwarf::DW_TAG_array_type, {}, elementType,
                       elements, type);
}

auto Codegen::ConvertDebugType::operator()(const UnboundedArrayType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(llvm::dwarf::DW_TAG_pointer_type, type, elementType);
}

auto Codegen::ConvertDebugType::operator()(const PointerType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(llvm::dwarf::DW_TAG_pointer_type, type, elementType);
}

auto Codegen::ConvertDebugType::operator()(const LvalueReferenceType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(llvm::dwarf::DW_TAG_reference_type, type, elementType);
}

auto Codegen::ConvertDebugType::operator()(const RvalueReferenceType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = gen.convertDebugType(type->elementType());
  return derivedType(llvm::dwarf::DW_TAG_rvalue_reference_type, type,
                     elementType);
}

auto Codegen::ConvertDebugType::operator()(const FunctionType* type)
    -> mlir::LLVM::DITypeAttr {
  mlir::SmallVector<mlir::LLVM::DITypeAttr> signatureTypes;
  signatureTypes.push_back(gen.convertDebugType(type->returnType()));
  for (auto argType : type->parameterTypes()) {
    signatureTypes.push_back(gen.convertDebugType(argType));
  }
  return mlir::LLVM::DISubroutineTypeAttr::get(context(), signatureTypes);
}

auto Codegen::ConvertDebugType::operator()(const ClassType* type)
    -> mlir::LLVM::DITypeAttr {
  auto symbol = type->symbol();
  if (!symbol) return {};

  // Check cache again because recursive calls might have populated it
  if (auto it = gen.debugTypeCache_.find(type);
      it != gen.debugTypeCache_.end()) {
    return it->second;
  }

  auto recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context()));

  auto tag = symbol->isUnion() ? llvm::dwarf::DW_TAG_union_type
                               : llvm::dwarf::DW_TAG_structure_type;

  // Create Recursive Self Reference
  auto name = to_string(symbol->name());
  auto recSelf = compositeType(tag, name, {}, {}, type);

  // Insert RecSelf into cache
  gen.debugTypeCache_[type] = recSelf;

  mlir::SmallVector<mlir::LLVM::DINodeAttr> elements;

  auto layout = symbol->layout();

  // Add bases
  for (auto* base : symbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;
    auto baseTypeAttr = gen.convertDebugType(baseClassSymbol->type());
    if (!baseTypeAttr) continue;

    uint64_t baseOffset = 0;
    if (layout) {
      if (auto baseInfo = layout->getBaseInfo(baseClassSymbol)) {
        baseOffset = baseInfo->offset;
      }
    }

    auto inheritanceAttr =
        derivedType(llvm::dwarf::DW_TAG_inheritance, baseClassSymbol->type(),
                    baseTypeAttr, baseOffset * 8);
    if (inheritanceAttr) elements.push_back(inheritanceAttr);
  }

  // Add fields
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
        derivedType(llvm::dwarf::DW_TAG_member, field->type(), fieldTypeAttr,
                    fieldOffset * 8, to_string(field->name()));
    if (memberAttr) elements.push_back(memberAttr);
  }

  auto fullDef = compositeType(tag, name, {}, elements, type);

  gen.debugTypeCache_.insert_or_assign(type, fullDef);

  return fullDef;
}

auto Codegen::ConvertDebugType::operator()(const EnumType* type)
    -> mlir::LLVM::DITypeAttr {
  auto symbol = type->symbol();
  auto name = to_string(symbol->name());
  auto underlyingTy = type->underlyingType();
  auto baseType = underlyingTy ? gen.convertDebugType(underlyingTy)
                               : mlir::LLVM::DITypeAttr{};

  auto sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8;
  auto alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8;

  mlir::LLVM::DIFileAttr file{};
  uint32_t line{};
  mlir::LLVM::DIScopeAttr scope{};
  if (symbol->location()) {
    auto [filename, l, c] = gen.unit_->tokenStartPosition(symbol->location());
    file = gen.getFileAttr(filename);
    line = l;
    scope = gen.getCompileUnitAttr(filename);
  }

  return mlir::LLVM::DICompositeTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_enumeration_type,
      mlir::StringAttr::get(context(), name), file, line, scope, baseType,
      mlir::LLVM::DIFlags{}, sizeInBits, alignInBits,
#if LLVM_VERSION_MAJOR < 22
      /*elements=*/{}, /*dataLocation=*/{}, /*rank=*/{}, /*allocated=*/{},
      /*associated=*/{}
#else
      /*dataLocation=*/{}, /*rank=*/{}, /*allocated=*/{}, /*associated=*/{},
      /*elements=*/{}
#endif
  );
}

auto Codegen::ConvertDebugType::operator()(const ScopedEnumType* type)
    -> mlir::LLVM::DITypeAttr {
  auto symbol = type->symbol();
  auto name = to_string(symbol->name());
  auto underlyingTy = type->underlyingType();
  auto baseType = underlyingTy ? gen.convertDebugType(underlyingTy)
                               : mlir::LLVM::DITypeAttr{};

  auto sizeInBits = memoryLayout()->sizeOf(type).value_or(0) * 8;
  auto alignInBits = memoryLayout()->alignmentOf(type).value_or(0) * 8;

  mlir::LLVM::DIFileAttr file{};
  uint32_t line{};
  mlir::LLVM::DIScopeAttr scope{};
  if (symbol->location()) {
    auto [filename, l, c] = gen.unit_->tokenStartPosition(symbol->location());
    file = gen.getFileAttr(filename);
    line = l;
    scope = gen.getCompileUnitAttr(filename);
  }

  return mlir::LLVM::DICompositeTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_enumeration_type,
      mlir::StringAttr::get(context(), name), file, line, scope, baseType,
      mlir::LLVM::DIFlags::EnumClass, sizeInBits, alignInBits,
#if LLVM_VERSION_MAJOR < 22
      /*elements=*/{}, /*dataLocation=*/{}, /*rank=*/{}, /*allocated=*/{},
      /*associated=*/{}
#else
      /*dataLocation=*/{}, /*rank=*/{}, /*allocated=*/{}, /*associated=*/{},
      /*elements=*/{}
#endif
  );
}

auto Codegen::ConvertDebugType::operator()(const MemberObjectPointerType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(
    const MemberFunctionPointerType* type) -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const NamespaceType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const TypeParameterType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(
    const TemplateTypeParameterType* type) -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const UnresolvedNameType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(
    const UnresolvedBoundedArrayType* type) -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const UnresolvedUnderlyingType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const OverloadSetType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

auto Codegen::ConvertDebugType::operator()(const BuiltinVaListType* type)
    -> mlir::LLVM::DITypeAttr {
  auto elementType = mlir::LLVM::DIBasicTypeAttr::get(
      context(), llvm::dwarf::DW_TAG_unspecified_type, "void", 0, 0);
  return derivedType(llvm::dwarf::DW_TAG_pointer_type, type, elementType);
}

auto Codegen::ConvertDebugType::operator()(const BuiltinMetaInfoType* type)
    -> mlir::LLVM::DITypeAttr {
  return {};
}

}  // namespace cxx
