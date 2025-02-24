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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FRnewOM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <cxx/decl_specs.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

struct DeclSpecs::Visitor {
  DeclSpecs& specs;

  void operator()(GeneratedTypeSpecifierAST* ast);
  void operator()(TypedefSpecifierAST* ast);
  void operator()(FriendSpecifierAST* ast);
  void operator()(ConstevalSpecifierAST* ast);
  void operator()(ConstinitSpecifierAST* ast);
  void operator()(ConstexprSpecifierAST* ast);
  void operator()(InlineSpecifierAST* ast);
  void operator()(StaticSpecifierAST* ast);
  void operator()(ExternSpecifierAST* ast);
  void operator()(ThreadLocalSpecifierAST* ast);
  void operator()(ThreadSpecifierAST* ast);
  void operator()(MutableSpecifierAST* ast);
  void operator()(VirtualSpecifierAST* ast);
  void operator()(ExplicitSpecifierAST* ast);
  void operator()(AutoTypeSpecifierAST* ast);
  void operator()(VoidTypeSpecifierAST* ast);
  void operator()(SizeTypeSpecifierAST* ast);
  void operator()(SignTypeSpecifierAST* ast);
  void operator()(VaListTypeSpecifierAST* ast);
  void operator()(IntegralTypeSpecifierAST* ast);
  void operator()(FloatingPointTypeSpecifierAST* ast);
  void operator()(ComplexTypeSpecifierAST* ast);
  void operator()(NamedTypeSpecifierAST* ast);
  void operator()(AtomicTypeSpecifierAST* ast);
  void operator()(UnderlyingTypeSpecifierAST* ast);
  void operator()(ElaboratedTypeSpecifierAST* ast);
  void operator()(DecltypeAutoSpecifierAST* ast);
  void operator()(DecltypeSpecifierAST* ast);
  void operator()(PlaceholderTypeSpecifierAST* ast);
  void operator()(ConstQualifierAST* ast);
  void operator()(VolatileQualifierAST* ast);
  void operator()(RestrictQualifierAST* ast);
  void operator()(EnumSpecifierAST* ast);
  void operator()(ClassSpecifierAST* ast);
  void operator()(TypenameSpecifierAST* ast);
  void operator()(SplicerTypeSpecifierAST* ast);
};

void DeclSpecs::Visitor::operator()(GeneratedTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(TypedefSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(FriendSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ConstevalSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ConstinitSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ConstexprSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(InlineSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(StaticSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ExternSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ThreadLocalSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ThreadSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(MutableSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(VirtualSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ExplicitSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(AutoTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(VoidTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(SizeTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(SignTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(VaListTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(IntegralTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(FloatingPointTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ComplexTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(NamedTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(AtomicTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(UnderlyingTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ElaboratedTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(DecltypeAutoSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(DecltypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(PlaceholderTypeSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ConstQualifierAST* ast) {}

void DeclSpecs::Visitor::operator()(VolatileQualifierAST* ast) {}

void DeclSpecs::Visitor::operator()(RestrictQualifierAST* ast) {}

void DeclSpecs::Visitor::operator()(EnumSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(ClassSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(TypenameSpecifierAST* ast) {}

void DeclSpecs::Visitor::operator()(SplicerTypeSpecifierAST* ast) {}

DeclSpecs::DeclSpecs(TranslationUnit* unit) : unit(unit) {}

auto DeclSpecs::control() const -> Control* { return unit->control(); }

void DeclSpecs::accept(SpecifierAST* specifier) {
  if (!specifier) return;
  visit(Visitor{*this}, specifier);
}

auto DeclSpecs::getType() const -> const Type* {
  auto type = this->type;

  if (!type || type == control()->getIntType()) {
    if (isLongLong && isUnsigned)
      type = control()->getUnsignedLongLongIntType();
    else if (isLongLong)
      type = control()->getLongLongIntType();
    else if (isLong && isUnsigned)
      type = control()->getUnsignedLongIntType();
    else if (isLong)
      type = control()->getLongIntType();
    else if (isShort && isUnsigned)
      type = control()->getUnsignedShortIntType();
    else if (isShort)
      type = control()->getShortIntType();
    else if (isUnsigned)
      type = control()->getUnsignedIntType();
    else if (isSigned)
      type = control()->getIntType();
  }

  if (!type) return nullptr;

  if (type == control()->getDoubleType() && isLong)
    type = control()->getLongDoubleType();

  if (isSigned && type == control()->getCharType())
    type = control()->getSignedCharType();

  if (isUnsigned) {
    switch (type->kind()) {
      case TypeKind::kChar:
        type = control()->getUnsignedCharType();
        break;
      case TypeKind::kShortInt:
        type = control()->getUnsignedShortIntType();
        break;
      case TypeKind::kInt:
        type = control()->getUnsignedIntType();
        break;
      case TypeKind::kLongInt:
        type = control()->getUnsignedLongIntType();
        break;
      case TypeKind::kLongLongInt:
        type = control()->getUnsignedLongLongIntType();
        break;
      case TypeKind::kChar8:
        type = control()->getUnsignedCharType();
        break;
      case TypeKind::kChar16:
        type = control()->getUnsignedShortIntType();
        break;
      case TypeKind::kChar32:
        type = control()->getUnsignedIntType();
        break;
      case TypeKind::kWideChar:
        type = control()->getUnsignedIntType();
        break;
      default:
        break;
    }  // switch
  }

  if (isConst) type = control()->add_const(type);
  if (isVolatile) type = control()->add_volatile(type);

  return type;
}

auto DeclSpecs::hasTypeSpecifier() const -> bool {
  if (typeSpecifier) return true;
  if (isShort || isLong) return true;
  if (isSigned || isUnsigned) return true;
  return false;
}

void DeclSpecs::setTypeSpecifier(SpecifierAST* specifier) {
  typeSpecifier = specifier;
}

auto DeclSpecs::hasClassOrEnumSpecifier() const -> bool {
  if (!typeSpecifier) return false;
  switch (typeSpecifier->kind()) {
    case ASTKind::ClassSpecifier:
    case ASTKind::EnumSpecifier:
    case ASTKind::ElaboratedTypeSpecifier:
    case ASTKind::TypenameSpecifier:
      return true;
    default:
      return false;
  }  // switch
}

auto DeclSpecs::hasPlaceholderTypeSpecifier() const -> bool {
  if (!typeSpecifier) return false;
  switch (typeSpecifier->kind()) {
    case ASTKind::AutoTypeSpecifier:
    case ASTKind::DecltypeAutoSpecifier:
    case ASTKind::PlaceholderTypeSpecifier:
    case ASTKind::DecltypeSpecifier:
      return true;
    default:
      return false;
  }  // switch
}

}  // namespace cxx