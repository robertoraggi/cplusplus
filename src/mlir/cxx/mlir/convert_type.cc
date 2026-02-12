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
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

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
  auto operator()(const NamespaceType* type) -> mlir::Type;
  auto operator()(const TypeParameterType* type) -> mlir::Type;
  auto operator()(const TemplateTypeParameterType* type) -> mlir::Type;
  auto operator()(const UnresolvedNameType* type) -> mlir::Type;
  auto operator()(const UnresolvedBoundedArrayType* type) -> mlir::Type;
  auto operator()(const UnresolvedUnderlyingType* type) -> mlir::Type;
  auto operator()(const OverloadSetType* type) -> mlir::Type;
  auto operator()(const BuiltinVaListType* type) -> mlir::Type;
  auto operator()(const BuiltinMetaInfoType* type) -> mlir::Type;
};

auto Codegen::convertType(const Type* type) -> mlir::Type {
  if (!type) {
    return builder_.getType<mlir::cxx::ExprType>();
  }

  return visit(ConvertType{*this}, type);
}

auto Codegen::ConvertType::getExprType() const -> mlir::Type {
  return gen.builder_.getType<mlir::cxx::ExprType>();
}

auto Codegen::ConvertType::getIntType(const Type* type, bool isSigned)
    -> mlir::Type {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  return mlir::IntegerType::get(gen.builder_.getContext(), width);
}

auto Codegen::ConvertType::getFloatType(const Type* type) -> mlir::Type {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  switch (width) {
    case 16:
      return mlir::Float16Type::get(gen.builder_.getContext());
    case 32:
      return mlir::Float32Type::get(gen.builder_.getContext());
    case 64:
      return mlir::Float64Type::get(gen.builder_.getContext());
    default:
      return mlir::Float64Type::get(gen.builder_.getContext());
  }
}

auto Codegen::ConvertType::operator()(const VoidType* type) -> mlir::Type {
  return gen.builder_.getType<mlir::cxx::VoidType>();
}

auto Codegen::ConvertType::operator()(const NullptrType* type) -> mlir::Type {
  auto voidType = gen.builder_.getType<mlir::cxx::VoidType>();
  return gen.builder_.getType<mlir::cxx::PointerType>(voidType);
}

auto Codegen::ConvertType::operator()(const DecltypeAutoType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const AutoType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BoolType* type) -> mlir::Type {
  return mlir::IntegerType::get(gen.builder_.getContext(), 1);
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
  // todo: toolchain specific
  return getIntType(type, true);
}

auto Codegen::ConvertType::operator()(const Char8Type* type) -> mlir::Type {
  return getIntType(type, false);  // unsigned 8-bit
}

auto Codegen::ConvertType::operator()(const Char16Type* type) -> mlir::Type {
  return getIntType(type, false);  // unsigned 16-bit
}

auto Codegen::ConvertType::operator()(const Char32Type* type) -> mlir::Type {
  return getIntType(type, false);  // unsigned 32-bit
}

auto Codegen::ConvertType::operator()(const WideCharType* type) -> mlir::Type {
  return getIntType(type, true);  // signed 32-bit on macOS/Unix
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
  return gen.builder_.getType<mlir::cxx::ArrayType>(elementType, type->size());
}

auto Codegen::ConvertType::operator()(const UnboundedArrayType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return gen.builder_.getType<mlir::cxx::PointerType>(elementType);
}

auto Codegen::ConvertType::operator()(const PointerType* type) -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return gen.builder_.getType<mlir::cxx::PointerType>(elementType);
}

auto Codegen::ConvertType::operator()(const LvalueReferenceType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return gen.builder_.getType<mlir::cxx::PointerType>(elementType);
}

auto Codegen::ConvertType::operator()(const RvalueReferenceType* type)
    -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return gen.builder_.getType<mlir::cxx::PointerType>(elementType);
}

auto Codegen::ConvertType::operator()(const FunctionType* type) -> mlir::Type {
  mlir::SmallVector<mlir::Type> inputs;
  for (auto argType : type->parameterTypes()) {
    inputs.push_back(gen.convertType(argType));
  }
  mlir::SmallVector<mlir::Type> results;
  if (!control()->is_void(type->returnType())) {
    results.push_back(gen.convertType(type->returnType()));
  }
  return gen.builder_.getType<mlir::cxx::FunctionType>(inputs, results,
                                                       type->isVariadic());
}

auto Codegen::ConvertType::operator()(const ClassType* type) -> mlir::Type {
  auto classSymbol = type->symbol();

  auto ctx = gen.builder_.getContext();

  if (auto it = gen.classNames_.find(classSymbol);
      it != gen.classNames_.end()) {
    return it->second;
  }

  auto name = to_string(classSymbol->name());
  if (name.empty()) {
    auto loc = type->symbol()->location();
    name = std::format("$class_{}", loc.index());
  }

  // Use mangled type name for template instantiations to avoid collisions
  // between e.g. Test<int> and Test<double> which both have name "Test".
  if (!classSymbol->templateArguments().empty()) {
    ExternalNameEncoder encoder;
    name = encoder.encode(type);
  }

  if (classSymbol->isUnion()) {
    name = std::format("union.{}", name);
  }

  mlir::cxx::ClassType classType = mlir::cxx::ClassType::getNamed(ctx, name);

  gen.classNames_[classSymbol] = classType;

  if (classSymbol->templateDeclaration()) {
    return classType;
  }

  // todo: layout of parent classes, anonymous nested fields, etc.

  std::vector<mlir::Type> memberTypes;

  if (classSymbol->isUnion()) {
    auto size = classSymbol->sizeInBytes();
    auto alignment = classSymbol->alignment();
    if (alignment == 0) alignment = 1;

    auto byteType = mlir::IntegerType::get(gen.builder_.getContext(), 8);
    auto arrayType = gen.builder_.getType<mlir::cxx::ArrayType>(byteType, size);
    memberTypes.push_back(arrayType);

  } else {
    auto layout = classSymbol->layout();
    if (layout && layout->hasDirectVtable()) {
      auto i8Type = mlir::IntegerType::get(gen.builder_.getContext(), 8);
      auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(i8Type);
      memberTypes.push_back(ptrType);
    }

    // Layout of parent classes
    for (auto base : classSymbol->baseClasses()) {
      const Type* baseType = base->type();
      if (!baseType && base->symbol()) {
        baseType = base->symbol()->type();
      }
      memberTypes.push_back(gen.convertType(baseType));
    }

    // Layout of members
    for (auto field : views::members(classSymbol) | views::non_static_fields) {
      auto memberType = gen.convertType(field->type());
      memberTypes.push_back(memberType);
    }
  }

  classType.setBody(memberTypes);

  return classType;
}

auto Codegen::ConvertType::operator()(const EnumType* type) -> mlir::Type {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return mlir::IntegerType::get(gen.builder_.getContext(), 32);
}

auto Codegen::ConvertType::operator()(const ScopedEnumType* type)
    -> mlir::Type {
  if (type->underlyingType()) return gen.convertType(type->underlyingType());
  return mlir::IntegerType::get(gen.builder_.getContext(), 32);
}

auto Codegen::ConvertType::operator()(const MemberObjectPointerType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const MemberFunctionPointerType* type)
    -> mlir::Type {
  return getExprType();
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
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnresolvedUnderlyingType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const OverloadSetType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BuiltinVaListType* type)
    -> mlir::Type {
  // todo: toolchain specific
  auto voidType = gen.builder_.getType<mlir::cxx::VoidType>();
  return gen.builder_.getType<mlir::cxx::PointerType>(voidType);
}

auto Codegen::ConvertType::operator()(const BuiltinMetaInfoType* type)
    -> mlir::Type {
  return getExprType();
}

}  // namespace cxx
