// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

namespace cxx {

struct Codegen::ConvertType {
  Codegen& gen;

  [[nodiscard]] auto control() const { return gen.control(); }
  [[nodiscard]] auto memoryLayout() const { return control()->memoryLayout(); }

  auto getExprType() const -> mlir::Type;
  auto getIntType(const Type* type, bool isSigned) -> mlir::Type;

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
};

auto Codegen::convertType(const Type* type) -> mlir::Type {
  return visit(ConvertType{*this}, type);
}

auto Codegen::ConvertType::getExprType() const -> mlir::Type {
  return gen.builder_.getType<mlir::cxx::ExprType>();
}

auto Codegen::ConvertType::getIntType(const Type* type, bool isSigned)
    -> mlir::Type {
  const auto width = memoryLayout()->sizeOf(type).value() * 8;
  return gen.builder_.getType<mlir::cxx::IntegerType>(width, isSigned);
}

auto Codegen::ConvertType::operator()(const VoidType* type) -> mlir::Type {
  return gen.builder_.getType<mlir::cxx::VoidType>();
}

auto Codegen::ConvertType::operator()(const NullptrType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const DecltypeAutoType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const AutoType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const BoolType* type) -> mlir::Type {
  return gen.builder_.getType<mlir::cxx::BoolType>();
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
  return getExprType();
}

auto Codegen::ConvertType::operator()(const Char8Type* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const Char16Type* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const Char32Type* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const WideCharType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const FloatType* type) -> mlir::Type {
  return gen.builder_.getF32Type();
}

auto Codegen::ConvertType::operator()(const DoubleType* type) -> mlir::Type {
  return gen.builder_.getF64Type();
}

auto Codegen::ConvertType::operator()(const LongDoubleType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const QualType* type) -> mlir::Type {
  return gen.convertType(type->elementType());
}

auto Codegen::ConvertType::operator()(const BoundedArrayType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const UnboundedArrayType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const PointerType* type) -> mlir::Type {
  auto elementType = gen.convertType(type->elementType());
  return gen.builder_.getType<mlir::cxx::PointerType>(elementType);
}

auto Codegen::ConvertType::operator()(const LvalueReferenceType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const RvalueReferenceType* type)
    -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const FunctionType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const ClassType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const EnumType* type) -> mlir::Type {
  return getExprType();
}

auto Codegen::ConvertType::operator()(const ScopedEnumType* type)
    -> mlir::Type {
  return getExprType();
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
  return getExprType();
}

}  // namespace cxx
