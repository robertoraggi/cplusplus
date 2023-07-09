// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>
#include <cxx/type_visitor.h>
#include <cxx/types.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cxx {

Type::~Type() = default;

auto equal_to(const Type* type, const Type* other) -> bool {
  if (type == other) {
    return true;
  }

  if (type == nullptr || other == nullptr) {
    return false;
  }

  return type->equalTo(other);
}

#define PROCESS_TYPE_KIND(kind) \
  case TypeKind::k##kind:       \
    return ((const kind##Type*)this)->equalTo0(((const kind##Type*)other));

auto Type::equalTo(const Type* other) const -> bool {
  if (this == other) {
    return true;
  }
  if (!other || other->kind() != kind_) {
    return false;
  }
  switch (kind_) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE_KIND)
    default:
      return false;
  }  // switch
}

#undef PROCESS_TYPE_KIND

#define PROCESS_TYPE_KIND(kind) \
  case TypeKind::k##kind:       \
    return #kind;

auto to_string(TypeKind kind) -> std::string_view {
  switch (kind) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE_KIND)
    default:
      assert(!"invalid");
      return {};
  }  // switch
}

#undef PROCESS_TYPE_KIND

auto InvalidType::equalTo0(const InvalidType*) const -> bool { return true; }

auto NullptrType::equalTo0(const NullptrType*) const -> bool { return true; }

auto DependentType::equalTo0(const DependentType* other) const -> bool {
  return symbol() == other->symbol();
}

auto AutoType::equalTo0(const AutoType*) const -> bool { return true; }

auto VoidType::equalTo0(const VoidType*) const -> bool { return true; }

auto BoolType::equalTo0(const BoolType*) const -> bool { return true; }

auto CharType::equalTo0(const CharType*) const -> bool { return true; }

auto SignedCharType::equalTo0(const SignedCharType*) const -> bool {
  return true;
}

auto UnsignedCharType::equalTo0(const UnsignedCharType* oher) const -> bool {
  return true;
}

auto ShortType::equalTo0(const ShortType*) const -> bool { return true; }

auto UnsignedShortType::equalTo0(const UnsignedShortType*) const -> bool {
  return true;
}

auto IntType::equalTo0(const IntType*) const -> bool { return true; }

auto UnsignedIntType::equalTo0(const UnsignedIntType*) const -> bool {
  return true;
}

auto LongType::equalTo0(const LongType*) const -> bool { return true; }

auto UnsignedLongType::equalTo0(const UnsignedLongType*) const -> bool {
  return true;
}

auto FloatType::equalTo0(const FloatType*) const -> bool { return true; }

auto DoubleType::equalTo0(const DoubleType*) const -> bool { return true; }

auto QualType::equalTo0(const QualType* other) const -> bool {
  return isConst() == other->isConst() && isVolatile() == other->isVolatile() &&
         elementType()->equalTo(other->elementType());
}

auto PointerType::equalTo0(const PointerType* other) const -> bool {
  return elementType_->equalTo(other->elementType());
}

auto LValueReferenceType::equalTo0(const LValueReferenceType* other) const
    -> bool {
  return elementType()->equalTo(other->elementType());
}

auto RValueReferenceType::equalTo0(const RValueReferenceType* other) const
    -> bool {
  return elementType()->equalTo(other->elementType());
}

auto ArrayType::equalTo0(const ArrayType* other) const -> bool {
  return extent() == other->extent() &&
         elementType()->equalTo(other->elementType());
}

auto FunctionType::equalTo0(const FunctionType* other) const -> bool {
  if (!is_same_type(classType(), other->classType())) {
    return false;
  }
  if (isVariadic() != other->isVariadic()) {
    return false;
  }
  if (!returnType()->equalTo(other->returnType())) {
    return false;
  }
  if (!is_same_parameters(parameters(), other->parameters())) {
    return false;
  }
  return true;
}

auto ConceptType::equalTo0(const ConceptType* other) const -> bool {
  return symbol() == other->symbol();
}

auto ClassType::equalTo0(const ClassType* other) const -> bool {
  return symbol() == other->symbol();
}

auto NamespaceType::equalTo0(const NamespaceType* other) const -> bool {
  return symbol() == other->symbol();
}

auto MemberPointerType::equalTo0(const MemberPointerType* other) const -> bool {
  if (!classType()->equalTo(other->classType())) {
    return false;
  }
  if (!elementType()->equalTo(other->elementType())) {
    return false;
  }
  return true;
}

auto EnumType::equalTo0(const EnumType* other) const -> bool {
  return symbol() == other->symbol();
}

auto GenericType::equalTo0(const GenericType* other) const -> bool {
  return symbol() == other->symbol();
}

auto PackType::equalTo0(const PackType*) const -> bool { return true; }

auto ScopedEnumType::equalTo0(const ScopedEnumType* other) const -> bool {
  return symbol() == other->symbol();
}

DependentType::DependentType(Control* control, DependentSymbol* symbol)
    : symbol_(symbol) {}

auto ClassType::isDerivedFrom(const ClassType* classType) const -> bool {
  auto* classSymbol = symbol_cast<ClassSymbol>(symbol_);
  auto* baseClassSymbol = symbol_cast<ClassSymbol>(classType->symbol_);
  return classSymbol->isDerivedFrom(baseClassSymbol);
}

auto ClassType::isBaseOf(const ClassType* classType) const -> bool {
  return classType->isDerivedFrom(this);
}

PointerType::PointerType(Control* control, const Type* elementType)
    : elementType_(elementType) {}

QualType::QualType(Control* control, const Type* elementType, bool isConst,
                   bool isVolatile)
    : elementType_(elementType), isConst_(isConst), isVolatile_(isVolatile) {}

auto promote_type(Control* control, const Type* type) -> const Type* {
  switch (type->kind()) {
    case TypeKind::kDouble:
    case TypeKind::kFloat:
    case TypeKind::kUnsignedLong:
    case TypeKind::kLong:
    case TypeKind::kUnsignedInt:
    case TypeKind::kInt:
      return type;
    default:
      return control->getIntType();
  }  // switch
}

auto is_same_parameters(const std::vector<Parameter>& params,
                        const std::vector<Parameter>& other) -> bool {
  if (params.size() != other.size()) {
    return false;
  }

  for (std::size_t i = 0; i < params.size(); ++i) {
    if (!is_same_type(params[i].type(), other[i].type())) {
      return false;
    }
  }

  return true;
}

auto is_same_template_arguments(const std::vector<TemplateArgument>& list,
                                const std::vector<TemplateArgument>& other)
    -> bool {
  if (list.size() != other.size()) {
    return false;
  }

  for (std::size_t i = 0; i < list.size(); ++i) {
    const auto& arg = list[i];
    const auto& otherArg = other[i];

    if (arg.kind() != otherArg.kind()) {
      return false;
    }

    if (!is_same_type(arg.type(), otherArg.type())) {
      return false;
    }

    switch (arg.kind()) {
      case TemplateArgumentKind::kType:
        break;

      case TemplateArgumentKind::kLiteral: {
        if (arg.value() != otherArg.value()) {
          return false;
        }
        break;
      }

      default:
        assert(!"invalid template argument type");
    }  // switch
  }

  return true;
}

auto is_same_type(const Type* type, const Type* other) -> bool {
  if (type == other) {
    return true;
  }
  if (!type || !other) {
    return false;
  }
  return type->equalTo(other);
}

auto type_kind(const Type* ty) -> TypeKind { return ty->kind(); }

auto type_element_type(const Type* ty) -> const Type* {
  if (auto qualType = type_cast<QualType>(ty)) {
    return qualType->elementType();
  }
  if (const ReferenceType* refType = type_cast<ReferenceType>(ty)) {
    return refType->elementType();
  }
  if (auto pointerType = type_cast<PointerType>(ty)) {
    return pointerType->elementType();
  }
  if (auto arrayType = type_cast<ArrayType>(ty)) {
    return arrayType->elementType();
  }
  if (auto memberPointerType = type_cast<MemberPointerType>(ty)) {
    return memberPointerType->elementType();
  }
  if (auto scopedEnumType = type_cast<ScopedEnumType>(ty)) {
    return scopedEnumType->elementType();
  }

  if (auto functionType = type_cast<FunctionType>(ty)) {
    return functionType->returnType();
  }

  return nullptr;
}

auto type_extent(const Type* ty) -> int {
  return type_cast<ArrayType>(ty)->extent();
}

auto function_type_parameter_count(const Type* ty) -> std::size_t {
  auto functionType = type_cast<FunctionType>(ty);

  if (!functionType) {
    return 0;
  }

  return functionType->parameters().size();
}

auto is_void_type(const Type* ty) -> bool { return ty->is(TypeKind::kVoid); }

auto is_bool_type(const Type* ty) -> bool { return ty->is(TypeKind::kBool); }

auto is_qual_type(const Type* ty) -> bool { return ty->is(TypeKind::kQual); }

auto is_integral_type(const Type* ty) -> bool {
  switch (ty->kind()) {
    case TypeKind::kBool:
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kShort:
    case TypeKind::kInt:
    case TypeKind::kLong:
    case TypeKind::kUnsignedChar:
    case TypeKind::kUnsignedShort:
    case TypeKind::kUnsignedInt:
    case TypeKind::kUnsignedLong:
      return true;
    default:
      return false;
  }  // switch
}

auto is_floating_point_type(const Type* ty) -> bool {
  switch (ty->kind()) {
    case TypeKind::kFloat:
    case TypeKind::kDouble:
      return true;
    default:
      return false;
  }  // switch
}

auto is_member_object_pointer_type(const Type* ty) -> bool {
  if (auto memberPointerType = type_cast<MemberPointerType>(ty)) {
    return !is_function_type(memberPointerType->elementType());
  }
  return false;
}

auto is_member_function_pointer_type(const Type* ty) -> bool {
  if (auto memberPointerType = type_cast<MemberPointerType>(ty)) {
    return is_function_type(memberPointerType->elementType());
  }
  return false;
}

auto is_pointer_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kPointer);
}

auto is_nullptr_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kNullptr);
}

auto is_lvalue_reference_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kLValueReference);
}

auto is_rvalue_reference_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kRValueReference);
}

auto is_array_type(const Type* ty) -> bool { return ty->is(TypeKind::kArray); }

auto is_function_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kFunction);
}

auto is_namespace_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kNamespace);
}

auto is_class_type(const Type* ty) -> bool { return ty->is(TypeKind::kClass); }

auto is_union_type(const Type* ty) -> bool {
  auto classType = type_cast<ClassType>(ty);
  if (!classType) {
    return false;
  }
  auto* symbol = symbol_cast<ClassSymbol>(classType->symbol());
  return symbol->isUnion();
}

auto is_enum_type(const Type* ty) -> bool { return ty->is(TypeKind::kEnum); }

auto is_unscoped_enum_type(const Type* type) -> bool { return false; }

auto is_scoped_enum_type(const Type* type) -> bool {
  return type->is(TypeKind::kScopedEnum);
}

auto is_signed(const Type* type) -> bool {
  switch (type->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kShort:
    case TypeKind::kInt:
    case TypeKind::kLong:
      return true;
    default:
      return false;
  }  // switch
}

auto is_unsigned(const Type* type) -> bool {
  switch (type->kind()) {
    case TypeKind::kUnsignedChar:
    case TypeKind::kUnsignedShort:
    case TypeKind::kUnsignedInt:
    case TypeKind::kUnsignedLong:
      return true;
    default:
      return false;
  }  // switch
}

auto is_const(const Type* type) -> bool {
  if (auto qualType = type_cast<QualType>(type)) {
    return qualType->isConst();
  }
  return false;
}

auto is_volatile(const Type* type) -> bool {
  if (auto qualType = type_cast<QualType>(type)) {
    return qualType->isVolatile();
  }
  return false;
}

auto make_signed(Control* control, const Type* type) -> const Type* {
  switch (type->kind()) {
    case TypeKind::kUnsignedChar:
      return control->getSignedCharType();
    case TypeKind::kUnsignedShort:
      return control->getShortType();
    case TypeKind::kUnsignedInt:
      return control->getIntType();
    case TypeKind::kUnsignedLong:
      return control->getLongType();
    default:
      return type;
  }  // switch
}

auto make_unsigned(Control* control, const Type* type) -> const Type* {
  switch (type->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
      return control->getUnsignedCharType();
    case TypeKind::kShort:
      return control->getUnsignedShortType();
    case TypeKind::kInt:
      return control->getUnsignedIntType();
    case TypeKind::kLong:
      return control->getUnsignedLongType();
    default:
      return type;
  }  // switch
}

auto remove_cv(const Type* type) -> const Type* {
  if (auto qualType = type_cast<QualType>(type)) {
    return qualType->elementType();
  }
  return type;
}

auto remove_ref(const Type* type) -> const Type* {
  if (auto refType = type_cast<ReferenceType>(type)) {
    return refType->elementType();
  }
  return type;
}

auto remove_cvref(const Type* type) -> const Type* {
  return remove_cv(remove_ref(type));
}

auto is_integral_or_unscoped_enum_type(const Type* ty) -> bool {
  return is_integral_type(ty) || is_unscoped_enum_type(ty);
}

auto is_member_pointer_type(const Type* ty) -> bool {
  return ty->is(TypeKind::kMemberPointer);
}

auto is_reference_type(const Type* ty) -> bool {
  if (is_lvalue_reference_type(ty)) {
    return true;
  }
  if (is_rvalue_reference_type(ty)) {
    return true;
  }
  return false;
}

auto is_compound_type(const Type* type) -> bool {
  if (is_fundamental_type(type)) {
    return false;
  }
  return true;
}

auto is_object_type(const Type* type) -> bool {
  if (is_scalar_type(type)) {
    return true;
  }
  if (is_array_type(type)) {
    return true;
  }
  if (is_union_type(type)) {
    return true;
  }
  if (is_class_type(type)) {
    return true;
  }
  return false;
}

auto is_scalar_type(const Type* type) -> bool {
  if (auto qualType = type_cast<QualType>(type)) {
    return is_scalar_type(qualType->elementType());
  }
  if (is_arithmetic_type(type)) {
    return true;
  }
  if (is_enum_type(type)) {
    return true;
  }
  if (is_pointer_type(type)) {
    return true;
  }
  if (is_member_pointer_type(type)) {
    return true;
  }
  if (is_nullptr_type(type)) {
    return true;
  }
  return false;
}

auto is_arithmetic_type(const Type* ty) -> bool {
  if (is_integral_type(ty)) {
    return true;
  }
  if (is_floating_point_type(ty)) {
    return true;
  }
  return false;
}

auto is_arithmetic_or_unscoped_type(const Type* type) -> bool {
  if (is_arithmetic_type(type)) {
    return true;
  }
  if (is_unscoped_enum_type(type)) {
    return true;
  }
  return false;
}

auto is_fundamental_type(const Type* type) -> bool {
  if (is_void_type(type)) {
    return true;
  }
  if (is_arithmetic_type(type)) {
    return true;
  }
  if (is_nullptr_type(type)) {
    return true;
  }
  return false;
}

auto is_literal_type(const Type* type) -> bool {
  if (is_void_type(remove_cvref(type))) {
    return true;
  }
  if (is_scalar_type(type)) {
    return true;
  }
  if (is_reference_type(type)) {
    return true;
  }

  if (auto arrayType = type_cast<ArrayType>(type)) {
    return is_literal_type(arrayType->elementType());
  }

  if (auto classType = type_cast<ClassType>(remove_cv(type))) {
    auto* symbol = symbol_cast<ClassSymbol>(classType->symbol());

    for (auto baseClass : symbol->baseClasses()) {
      if (auto* baseClassSymbol = symbol_cast<ClassSymbol>(baseClass)) {
        if (!is_literal_type(baseClassSymbol->type())) {
          return false;
        }
      }
    }

    if (symbol->destructor()) {
      return false;
    }

    bool has_constexpr_ctor = true;

    for (Symbol* ctor = symbol->constructors(); ctor; ctor = ctor->next) {
      auto* constructor = symbol_cast<FunctionSymbol>(ctor);
      if (!constructor) {
        continue;
      }
      if (constructor->name() != symbol->name()) {
        continue;
      }
      if (!constructor->isConstexpr()) {
        continue;
      }
      has_constexpr_ctor = true;
      break;
    }

    if (!has_constexpr_ctor) {
      return false;
    }

    if (Scope* members = symbol->members()) {
      for (auto it = members->begin(); it != members->end(); it = it + 1) {
        Symbol* member = *it;
        if (auto* memberSymbol = symbol_cast<MemberSymbol>(member)) {
          if (!is_literal_type(memberSymbol->type())) {
            return false;
          }
        }
      }
    }

    return true;
  }

  return false;
}

auto common_type(Control* control, const Type* type, const Type* other)
    -> const Type* {
  type = remove_cvref(type);
  other = remove_cvref(other);

  if (is_same_type(type, other)) {
    return type;
  }

  if (is_pointer_type(type) || is_pointer_type(other)) {
    return control->getLongType();
  }

  if (type->is(TypeKind::kDouble)) {
    return type;
  }
  if (other->is(TypeKind::kDouble)) {
    return other;
  }

  if (type->is(TypeKind::kFloat)) {
    return type;
  }
  if (other->is(TypeKind::kFloat)) {
    return other;
  }

  if (type->is(TypeKind::kUnsignedLong)) {
    return type;
  }
  if (other->is(TypeKind::kUnsignedLong)) {
    return other;
  }

  if (type->is(TypeKind::kLong)) {
    return type;
  }
  if (other->is(TypeKind::kLong)) {
    return other;
  }

  if (type->is(TypeKind::kUnsignedInt)) {
    return type;
  }
  if (other->is(TypeKind::kUnsignedInt)) {
    return other;
  }

  return control->getIntType();
}

LValueReferenceType::LValueReferenceType(Control* control,
                                         const Type* elementType)
    : ReferenceType(Kind, elementType) {}

RValueReferenceType::RValueReferenceType(Control* control,
                                         const Type* elementType)
    : ReferenceType(Kind, elementType) {}

ArrayType::ArrayType(Control* control, const Type* elementType, int extent)
    : elementType_(elementType), extent_(extent) {}

auto FunctionType::makeTemplate(Control* control, FunctionSymbol* symbol) const
    -> const FunctionType* {
  assert(symbol_ == nullptr || symbol_ == symbol);
  setSymbol(symbol_);
  symbol->setTemplate(true);
  return this;
}

FunctionType::FunctionType(Control* control, const Type* classType,
                           const Type* returnType,
                           std::vector<Parameter> parameters, bool isVariadic)
    : classType_(classType),
      returnType_(returnType),
      parameters_(std::move(parameters)),
      isVariadic_(isVariadic) {}

MemberPointerType::MemberPointerType(Control* control, const Type* classType,
                                     const Type* elementType)
    : classType_(classType), elementType_(elementType) {}

GenericType::GenericType(Control* control, Symbol* symbol) : symbol_(symbol) {}

PackType::PackType(Control* control, Symbol* symbol) : symbol_(symbol) {}

ClassType::ClassType(Control* control, ClassSymbol* symbol) : symbol_(symbol) {}

ConceptType::ConceptType(Control* control, Symbol* symbol) : symbol_(symbol) {}

ScopedEnumType::ScopedEnumType(Control* control, ScopedEnumSymbol* symbol,
                               const Type* elementType)
    : elementType_(elementType), symbol_(symbol) {}

EnumType::EnumType(Control* control, Symbol* symbol) : symbol_(symbol) {}

NamespaceType::NamespaceType(Control* control, NamespaceSymbol* symbol)
    : symbol_(symbol) {}

auto to_string(const Type* type) -> std::string {
  TypePrinter type_printer;

  return type_printer.to_string(type);
}

#define DECLARE_VISITOR(name) \
  void name##Type::accept(TypeVisitor* visitor) const { visitor->visit(this); }

CXX_FOR_EACH_TYPE_KIND(DECLARE_VISITOR)

#undef DECLARE_VISITOR

}  // namespace cxx
