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

auto type_kind_to_string(TypeKind kind) -> const char* {
  switch (kind) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE_KIND)
    default:
      assert(!"invalid");
      return nullptr;
  }  // switch
}

#undef PROCESS_TYPE_KIND

auto InvalidType::equalTo0(const InvalidType*) const -> bool { return true; }

auto NullptrType::equalTo0(const NullptrType*) const -> bool { return true; }

auto DependentType::equalTo0(const DependentType* other) const -> bool {
  return symbol == other->symbol;
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
  return isConst == other->isConst && isVolatile == other->isVolatile &&
         elementType->equalTo(other->elementType);
}

auto PointerType::equalTo0(const PointerType* other) const -> bool {
  return elementType->equalTo(other->elementType);
}

auto LValueReferenceType::equalTo0(const LValueReferenceType* other) const
    -> bool {
  return elementType->equalTo(other->elementType);
}

auto RValueReferenceType::equalTo0(const RValueReferenceType* other) const
    -> bool {
  return elementType->equalTo(other->elementType);
}

auto ArrayType::equalTo0(const ArrayType* other) const -> bool {
  return dim == other->dim && elementType->equalTo(other->elementType);
}

auto FunctionType::equalTo0(const FunctionType* other) const -> bool {
  if (!is_same_type(classType, other->classType)) {
    return false;
  }
  if (isVariadic != other->isVariadic) {
    return false;
  }
  if (!returnType->equalTo(other->returnType)) {
    return false;
  }
  if (!is_same_parameters(parameters, other->parameters)) {
    return false;
  }
  return true;
}

auto ConceptType::equalTo0(const ConceptType* other) const -> bool {
  return symbol == other->symbol;
}

auto ClassType::equalTo0(const ClassType* other) const -> bool {
  return symbol == other->symbol;
}

auto NamespaceType::equalTo0(const NamespaceType* other) const -> bool {
  return symbol == other->symbol;
}

auto MemberPointerType::equalTo0(const MemberPointerType* other) const -> bool {
  if (!classType->equalTo(other->classType)) {
    return false;
  }
  if (!elementType->equalTo(other->elementType)) {
    return false;
  }
  return true;
}

auto EnumType::equalTo0(const EnumType* other) const -> bool {
  return symbol == other->symbol;
}

auto GenericType::equalTo0(const GenericType* other) const -> bool {
  return symbol == other->symbol;
}

auto PackType::equalTo0(const PackType*) const -> bool { return true; }

auto ScopedEnumType::equalTo0(const ScopedEnumType* other) const -> bool {
  return symbol == other->symbol;
}

DependentType::DependentType(Control* control, DependentSymbol* symbol)
    : symbol(symbol) {}

auto ClassType::isDerivedFrom(const ClassType* classType) const -> bool {
  auto* classSymbol = symbol_cast<ClassSymbol>(symbol);
  auto* baseClassSymbol = symbol_cast<ClassSymbol>(classType->symbol);
  return classSymbol->isDerivedFrom(baseClassSymbol);
}

auto ClassType::isBaseOf(const ClassType* classType) const -> bool {
  return classType->isDerivedFrom(this);
}

PointerType::PointerType(Control* control, const Type* elementType)
    : elementType(elementType) {}

QualType::QualType(Control* control, const Type* elementType, bool isConst,
                   bool isVolatile)
    : elementType(elementType), isConst(isConst), isVolatile(isVolatile) {}

auto TemplateArgumentList::make(const Type* type) -> TemplateArgumentList* {
  auto* list = new TemplateArgumentList();
  list->kind = TA_TYPE;
  list->type = type;
  return list;
}

auto TemplateArgumentList::makeLiteral(const Type* type, long value)
    -> TemplateArgumentList* {
  auto* list = new TemplateArgumentList();
  list->kind = TA_LITERAL;
  list->type = type;
  list->value = value;
  return list;
}

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

auto is_same_parameters(ParameterList* params, ParameterList* other) -> bool {
  if (params == other) {
    return true;
  }
  if (!params || !other) {
    return false;
  }
  if (!is_same_type(params->type, other->type)) {
    return false;
  }
  return is_same_parameters(params->next, other->next);
}

auto is_same_template_arguments(const TemplateArgumentList* list,
                                const TemplateArgumentList* other) -> bool {
  if (list == other) {
    return true;
  }
  if (!list || !other) {
    return false;
  }
  if (list->kind != other->kind) {
    return false;
  }
  switch (list->kind) {
    case TA_TYPE:
      if (!is_same_type(list->type, other->type)) {
        return false;
      }
      break;
    case TA_LITERAL:
      if (!is_same_type(list->type, other->type)) {
        return false;
      }
      if (list->value != other->value) {
        return false;
      }
      break;
    default:
      assert(!"invalid template argument type");
  }  // switch
  return is_same_template_arguments(list->next, other->next);
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
  if (const auto* qualType = type_cast<QualType>(ty)) {
    return qualType->elementType;
  }
  if (const ReferenceType* refType = type_cast<ReferenceType>(ty)) {
    return refType->elementType;
  }
  if (const auto* pointerType = type_cast<PointerType>(ty)) {
    return pointerType->elementType;
  }
  if (const auto* arrayType = type_cast<ArrayType>(ty)) {
    return arrayType->elementType;
  }
  if (const auto* memberPointerType = type_cast<MemberPointerType>(ty)) {
    return memberPointerType->elementType;
  }
  if (const auto* scopedEnumType = type_cast<ScopedEnumType>(ty)) {
    return scopedEnumType->elementType;
  }

  if (const auto* functionType = type_cast<FunctionType>(ty)) {
    assert(!"deprecated");
    return functionType->returnType;
  }

  return nullptr;
}

auto type_extent(const Type* ty) -> int {
  return type_cast<ArrayType>(ty)->dim;
}

auto function_type_parameter_count(const Type* ty) -> int {
  int i = 0;
  for (ParameterList* it = type_cast<FunctionType>(ty)->parameters; it;
       it = it->next) {
    ++i;
  }
  return i;
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
  if (const auto* memberPointerType = type_cast<MemberPointerType>(ty)) {
    return !is_function_type(memberPointerType->elementType);
  }
  return false;
}

auto is_member_function_pointer_type(const Type* ty) -> bool {
  if (const auto* memberPointerType = type_cast<MemberPointerType>(ty)) {
    return is_function_type(memberPointerType->elementType);
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
  const auto* classType = type_cast<ClassType>(ty);
  if (!classType) {
    return false;
  }
  auto* symbol = symbol_cast<ClassSymbol>(classType->symbol);
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
  if (const auto* qualType = type_cast<QualType>(type)) {
    return qualType->isConst;
  }
  return false;
}

auto is_volatile(const Type* type) -> bool {
  if (const auto* qualType = type_cast<QualType>(type)) {
    return qualType->isVolatile;
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
  if (const auto* qualType = type_cast<QualType>(type)) {
    return qualType->elementType;
  }
  return type;
}

auto remove_cvref(const Type* type) -> const Type* {
  return remove_cv(type->remove_ref());
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
  if (const auto* qualType = type_cast<QualType>(type)) {
    return is_scalar_type(qualType->elementType);
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

  if (const auto* arrayType = type_cast<ArrayType>(type)) {
    return is_literal_type(arrayType->elementType);
  }

  if (const auto* classType = type_cast<ClassType>(remove_cv(type))) {
    auto* symbol = symbol_cast<ClassSymbol>(classType->symbol);

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

auto Type::remove_ref() const -> const Type* {
  if (const ReferenceType* refType = type_cast<ReferenceType>(this)) {
    return refType->elementType;
  }
  return this;
}

LValueReferenceType::LValueReferenceType(Control* control,
                                         const Type* elementType) {
  this->elementType = elementType;
}

RValueReferenceType::RValueReferenceType(Control* control,
                                         const Type* elementType) {
  this->elementType = elementType;
}

ArrayType::ArrayType(Control* control, const Type* elementType, int dim)
    : elementType(elementType), dim(dim) {}

auto FunctionType::makeTemplate(Control* control, FunctionSymbol* symbol) const
    -> const FunctionType* {
  auto* type = const_cast<FunctionType*>(this);
  assert(type->symbol == nullptr || type->symbol == symbol);
  type->symbol = symbol;
  symbol->setTemplate(true);
  return type;
}

FunctionType::FunctionType(Control* control, const Type* classType,
                           const Type* returnType, ParameterList* parameters,
                           bool isVariadic)
    : classType(classType),
      returnType(returnType),
      parameters(parameters),
      isVariadic(isVariadic) {}

MemberPointerType::MemberPointerType(Control* control, const Type* classType,
                                     const Type* elementType)
    : classType(classType), elementType(elementType) {}

GenericType::GenericType(Control* control, Symbol* symbol) : symbol(symbol) {}

PackType::PackType(Control* control, Symbol* symbol) : symbol(symbol) {}

ClassType::ClassType(Control* control, ClassSymbol* symbol) : symbol(symbol) {}

ConceptType::ConceptType(Control* control, Symbol* symbol) : symbol(symbol) {}

ScopedEnumType::ScopedEnumType(Control* control, ScopedEnumSymbol* symbol,
                               const Type* elementType)
    : elementType(elementType), symbol(symbol) {}

EnumType::EnumType(Control* control, Symbol* symbol) : symbol(symbol) {}

NamespaceType::NamespaceType(Control* control, NamespaceSymbol* symbol)
    : symbol(symbol) {}

auto ParameterList::make(const Name* name, const Type* type) -> ParameterList* {
  auto* param = new ParameterList();
  param->name = name;
  param->type = type;
  return param;
}

auto type_to_string(const Type* type, char* out, std::size_t size)
    -> std::string {
  TypePrinter type_printer;

  return type_printer.to_string(type);
}

#define DECLARE_VISITOR(name) \
  void name##Type::accept(TypeVisitor* visitor) const { visitor->visit(this); }

CXX_FOR_EACH_TYPE_KIND(DECLARE_VISITOR)

#undef DECLARE_VISITOR

}  // namespace cxx
