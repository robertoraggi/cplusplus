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

#include <cxx/decl_specs.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

struct DeclSpecs::Visitor {
  DeclSpecs& specs;

  [[nodiscard]] auto control() const -> Control* { return specs.control(); }

  void operator()(TypedefSpecifierAST* ast);
  void operator()(FriendSpecifierAST* ast);
  void operator()(ConstevalSpecifierAST* ast);
  void operator()(ConstinitSpecifierAST* ast);
  void operator()(ConstexprSpecifierAST* ast);
  void operator()(InlineSpecifierAST* ast);
  void operator()(NoreturnSpecifierAST* ast);
  void operator()(StaticSpecifierAST* ast);
  void operator()(ExternSpecifierAST* ast);
  void operator()(RegisterSpecifierAST* ast);
  void operator()(ThreadLocalSpecifierAST* ast);
  void operator()(ThreadSpecifierAST* ast);
  void operator()(MutableSpecifierAST* ast);
  void operator()(VirtualSpecifierAST* ast);
  void operator()(ExplicitSpecifierAST* ast);
  void operator()(AutoTypeSpecifierAST* ast);
  void operator()(VoidTypeSpecifierAST* ast);
  void operator()(SizeTypeSpecifierAST* ast);
  void operator()(SignTypeSpecifierAST* ast);
  void operator()(BuiltinTypeSpecifierAST* ast);
  void operator()(UnaryBuiltinTypeSpecifierAST* ast);
  void operator()(BinaryBuiltinTypeSpecifierAST* ast);
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
  void operator()(AtomicQualifierAST* ast);
  void operator()(EnumSpecifierAST* ast);
  void operator()(ClassSpecifierAST* ast);
  void operator()(TypenameSpecifierAST* ast);
  void operator()(SplicerTypeSpecifierAST* ast);
};

void DeclSpecs::Visitor::operator()(TypedefSpecifierAST* ast) {
  specs.isTypedef = true;
}

void DeclSpecs::Visitor::operator()(FriendSpecifierAST* ast) {
  specs.isFriend = true;
}

void DeclSpecs::Visitor::operator()(ConstevalSpecifierAST* ast) {
  specs.isConsteval = true;
}

void DeclSpecs::Visitor::operator()(ConstinitSpecifierAST* ast) {
  specs.isConstinit = true;
}

void DeclSpecs::Visitor::operator()(ConstexprSpecifierAST* ast) {
  specs.isConstexpr = true;
}

void DeclSpecs::Visitor::operator()(NoreturnSpecifierAST* ast) {
  specs.isNoreturn = true;
}

void DeclSpecs::Visitor::operator()(InlineSpecifierAST* ast) {
  specs.isInline = true;
}

void DeclSpecs::Visitor::operator()(StaticSpecifierAST* ast) {
  specs.isStatic = true;
}

void DeclSpecs::Visitor::operator()(ExternSpecifierAST* ast) {
  specs.isExtern = true;
}

void DeclSpecs::Visitor::operator()(RegisterSpecifierAST* ast) {
  specs.isRegister = true;
}

void DeclSpecs::Visitor::operator()(ThreadLocalSpecifierAST* ast) {
  specs.isThreadLocal = true;
}

void DeclSpecs::Visitor::operator()(ThreadSpecifierAST* ast) {
  specs.isThread = true;
}

void DeclSpecs::Visitor::operator()(MutableSpecifierAST* ast) {
  specs.isMutable = true;
}

void DeclSpecs::Visitor::operator()(VirtualSpecifierAST* ast) {
  specs.isVirtual = true;
}

void DeclSpecs::Visitor::operator()(ExplicitSpecifierAST* ast) {
  specs.isExplicit = true;
}

void DeclSpecs::Visitor::operator()(AutoTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  specs.type_ = control()->getAutoType();
  specs.isAuto = true;
}

void DeclSpecs::Visitor::operator()(VoidTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  specs.type_ = control()->getVoidType();
}

void DeclSpecs::Visitor::operator()(SizeTypeSpecifierAST* ast) {
  switch (ast->specifier) {
    case TokenKind::T_SHORT:
      specs.isShort = true;
      break;

    case TokenKind::T_LONG:
      if (specs.isLong)
        specs.isLongLong = true;
      else
        specs.isLong = true;
      break;

    default:
      break;
  }  // switch
}

void DeclSpecs::Visitor::operator()(SignTypeSpecifierAST* ast) {
  switch (ast->specifier) {
    case TokenKind::T_SIGNED:
      specs.isSigned = true;
      break;

    case TokenKind::T_UNSIGNED:
      specs.isUnsigned = true;
      break;

    default:
      break;
  }  // switch
}

void DeclSpecs::Visitor::operator()(BuiltinTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;

  switch (ast->specifier) {
    case TokenKind::T___BUILTIN_VA_LIST:
      specs.type_ = control()->getBuiltinVaListType();
      break;

    case TokenKind::T___BUILTIN_META_INFO:
      specs.type_ = control()->getBuiltinMetaInfoType();
      break;

    default:
      break;
  }  // switch
}

void DeclSpecs::Visitor::operator()(UnaryBuiltinTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  if (!ast->typeId || !ast->typeId->type) return;
  if (isDependent(specs.translationUnit(), ast->typeId)) return;
  auto inputType = ast->typeId->type;
  switch (ast->builtinKind) {
    case UnaryBuiltinTypeKind::T___REMOVE_CV:
      specs.type_ = control()->remove_cv(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_CONST:
      specs.type_ = control()->remove_const(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_VOLATILE:
      specs.type_ = control()->remove_volatile(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_CVREF:
      specs.type_ = control()->remove_cvref(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_REFERENCE_T:
      specs.type_ = control()->remove_reference(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_POINTER:
      specs.type_ = control()->remove_pointer(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_EXTENT:
      specs.type_ = control()->remove_extent(inputType);
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_ALL_EXTENTS:
      specs.type_ = control()->remove_all_extents(inputType);
      break;
    case UnaryBuiltinTypeKind::T___ADD_LVALUE_REFERENCE:
      specs.type_ = control()->add_lvalue_reference(inputType);
      break;
    case UnaryBuiltinTypeKind::T___ADD_RVALUE_REFERENCE:
      specs.type_ = control()->add_rvalue_reference(inputType);
      break;
    case UnaryBuiltinTypeKind::T___ADD_POINTER:
      specs.type_ = control()->add_pointer(inputType);
      break;
    case UnaryBuiltinTypeKind::T___DECAY:
      specs.type_ = control()->decay(inputType);
      break;
    case UnaryBuiltinTypeKind::T___MAKE_SIGNED:
      // TODO: implement make_signed
      specs.type_ = inputType;
      break;
    case UnaryBuiltinTypeKind::T___MAKE_UNSIGNED:
      // TODO: implement make_unsigned
      specs.type_ = inputType;
      break;
    case UnaryBuiltinTypeKind::T___REMOVE_RESTRICT:
      // restrict is not modeled, so this is a no-op
      specs.type_ = inputType;
      break;
    default:
      break;
  }
}

void DeclSpecs::Visitor::operator()(BinaryBuiltinTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  // ### todo
}

void DeclSpecs::Visitor::operator()(IntegralTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  switch (ast->specifier) {
    case TokenKind::T_CHAR:
      specs.type_ = control()->getCharType();
      break;

    case TokenKind::T_CHAR8_T:
      specs.type_ = control()->getChar8Type();
      break;

    case TokenKind::T_CHAR16_T:
      specs.type_ = control()->getChar16Type();
      break;

    case TokenKind::T_CHAR32_T:
      specs.type_ = control()->getChar32Type();
      break;

    case TokenKind::T_WCHAR_T:
      specs.type_ = control()->getWideCharType();
      break;

    case TokenKind::T_BOOL:
      specs.type_ = control()->getBoolType();
      break;

    case TokenKind::T_INT:
      specs.type_ = control()->getIntType();
      break;

    case TokenKind::T___INT64:
      if (control()->memoryLayout()->sizeOfLong() == 8)
        specs.type_ = control()->getLongIntType();
      else if (control()->memoryLayout()->sizeOfLongLong() == 8)
        specs.type_ = control()->getLongLongIntType();
      break;

    case TokenKind::T___INT128_T:
    case TokenKind::T___INT128:
      specs.type_ = control()->getInt128Type();
      break;

    case TokenKind::T___UINT128_T:
      specs.type_ = control()->getUnsignedInt128Type();
      break;

    default:
      break;
  }  // switch
}

void DeclSpecs::Visitor::operator()(FloatingPointTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  switch (ast->specifier) {
    case TokenKind::T_FLOAT:
      specs.type_ = control()->getFloatType();
      break;

    case TokenKind::T_DOUBLE:
      specs.type_ = control()->getDoubleType();
      break;

    case TokenKind::T_LONG:
      specs.type_ = control()->getLongDoubleType();
      break;

    case TokenKind::T__FLOAT16:
      specs.type_ = control()->getFloat16Type();
      break;

    case TokenKind::T___FLOAT80:
      // ### todo
      break;

    case TokenKind::T___FLOAT128:
      // ### todo
      break;

    default:
      break;
  }  // switch
}

void DeclSpecs::Visitor::operator()(ComplexTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  specs.isComplex = true;
}

void DeclSpecs::Visitor::operator()(NamedTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;

  if (ast->symbol) {
    specs.type_ = ast->symbol->type();
  } else {
    specs.type_ = control()->getUnresolvedNameType(
        specs.translationUnit(), ast->nestedNameSpecifier, ast->unqualifiedId);
  }
}

void DeclSpecs::Visitor::operator()(AtomicTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  // ### todo
}

void DeclSpecs::Visitor::operator()(UnderlyingTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;

  if (ast->typeId) {
    if (auto enumType = type_cast<EnumType>(ast->typeId->type)) {
      specs.type_ = enumType->underlyingType();
    } else if (auto scopedEnumType =
                   type_cast<ScopedEnumType>(ast->typeId->type)) {
      specs.type_ = scopedEnumType->underlyingType();
    } else {
      specs.type_ = control()->getUnresolvedUnderlyingType(
          specs.translationUnit(), ast->typeId);
    }
  }
}

void DeclSpecs::Visitor::operator()(ElaboratedTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  if (ast->symbol) specs.type_ = ast->symbol->type();
}

void DeclSpecs::Visitor::operator()(DecltypeAutoSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  specs.type_ = control()->getDecltypeAutoType();
  specs.isDecltypeAuto = true;
}

void DeclSpecs::Visitor::operator()(DecltypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  specs.type_ = ast->type;
}

void DeclSpecs::Visitor::operator()(PlaceholderTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
}

void DeclSpecs::Visitor::operator()(ConstQualifierAST* ast) {
  specs.isConst = true;
}

void DeclSpecs::Visitor::operator()(VolatileQualifierAST* ast) {
  specs.isVolatile = true;
}

void DeclSpecs::Visitor::operator()(RestrictQualifierAST* ast) {
  specs.isRestrict = true;
}

void DeclSpecs::Visitor::operator()(AtomicQualifierAST* ast) {
  specs.isAtomic = true;
}

void DeclSpecs::Visitor::operator()(EnumSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  if (ast->symbol) specs.type_ = ast->symbol->type();
}

void DeclSpecs::Visitor::operator()(ClassSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;
  if (ast->symbol) specs.type_ = ast->symbol->type();
}

void DeclSpecs::Visitor::operator()(TypenameSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;

  // Resolve the typename when the NNS scope is known.
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol) {
    if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
      auto symbol = lookupType(ast->nestedNameSpecifier->symbol,
                               ast->nestedNameSpecifier, nameId->identifier);
      if (symbol) {
        specs.type_ = symbol->type();
        return;
      }
    }
  }

  specs.type_ = control()->getUnresolvedNameType(
      specs.translationUnit(), ast->nestedNameSpecifier, ast->unqualifiedId);
}

void DeclSpecs::Visitor::operator()(SplicerTypeSpecifierAST* ast) {
  specs.typeSpecifier_ = ast;

  auto interp = ASTInterpreter{specs.translationUnit()};
  auto value = interp.evaluate(ast->splicer->expression);
  if (!value.has_value()) return;

  auto metaPtr = std::get_if<std::shared_ptr<Meta>>(&*value);
  if (!metaPtr) return;

  auto meta = *metaPtr;

  auto typePtr = std::get_if<const Type*>(&meta->value);
  if (!typePtr) return;

  specs.type_ = *typePtr;
}

DeclSpecs::DeclSpecs(TranslationUnit* unit) : unit_(unit) {}

auto DeclSpecs::translationUnit() const -> TranslationUnit* { return unit_; }

auto DeclSpecs::control() const -> Control* {
  return translationUnit()->control();
}

void DeclSpecs::accept(SpecifierAST* specifier) {
  if (!specifier) return;
  visit(Visitor{*this}, specifier);
}

auto DeclSpecs::type() const -> const Type* {
  if (!finished_) {
    cxx_runtime_error("DeclSpecs::finish() not called");
  }

  return type_;
}

void DeclSpecs::setType(const Type* type) { type_ = type; }

void DeclSpecs::finish() {
  if (finished_) {
    cxx_runtime_error("DeclSpecs::finish() called twice");
  }

  finished_ = true;
  if (!type_ || type_ == control()->getIntType()) {
    if (isLongLong && isUnsigned)
      type_ = control()->getUnsignedLongLongIntType();
    else if (isLongLong)
      type_ = control()->getLongLongIntType();
    else if (isLong && isUnsigned)
      type_ = control()->getUnsignedLongIntType();
    else if (isLong)
      type_ = control()->getLongIntType();
    else if (isShort && isUnsigned)
      type_ = control()->getUnsignedShortIntType();
    else if (isShort)
      type_ = control()->getShortIntType();
    else if (isUnsigned)
      type_ = control()->getUnsignedIntType();
    else if (isSigned)
      type_ = control()->getIntType();
  }

  if (type_ == control()->getInt128Type() && isUnsigned) {
    type_ = control()->getUnsignedInt128Type();
  }

  if (!type_) {
    return;
  }

  if (type_ == control()->getDoubleType() && isLong)
    type_ = control()->getLongDoubleType();

  if (isSigned && type_ == control()->getCharType())
    type_ = control()->getSignedCharType();

  if (isUnsigned) {
    switch (type_->kind()) {
      case TypeKind::kChar:
        type_ = control()->getUnsignedCharType();
        break;
      case TypeKind::kShortInt:
        type_ = control()->getUnsignedShortIntType();
        break;
      case TypeKind::kInt:
        type_ = control()->getUnsignedIntType();
        break;
      case TypeKind::kLongInt:
        type_ = control()->getUnsignedLongIntType();
        break;
      case TypeKind::kLongLongInt:
        type_ = control()->getUnsignedLongLongIntType();
        break;
      case TypeKind::kChar8:
        type_ = control()->getUnsignedCharType();
        break;
      case TypeKind::kChar16:
        type_ = control()->getUnsignedShortIntType();
        break;
      case TypeKind::kChar32:
        type_ = control()->getUnsignedIntType();
        break;
      case TypeKind::kWideChar:
        type_ = control()->getUnsignedIntType();
        break;
      default:
        break;
    }  // switch
  }

  if (isConst) type_ = control()->add_const(type_);
  if (isVolatile) type_ = control()->add_volatile(type_);
}

auto DeclSpecs::hasTypeOrSizeSpecifier() const -> bool {
  if (hasTypeSpecifier()) return true;
  if (isShort || isLong) return true;
  if (isSigned || isUnsigned) return true;
  return false;
}

auto DeclSpecs::hasTypeSpecifier() const -> bool {
  if (typeSpecifier_) return true;
  return false;
}

auto DeclSpecs::typeSpecifier() const -> SpecifierAST* {
  return typeSpecifier_;
}

void DeclSpecs::setTypeSpecifier(SpecifierAST* specifier) {
  typeSpecifier_ = specifier;
}

auto DeclSpecs::hasClassOrElaboratedTypeSpecifier() const -> bool {
  if (!typeSpecifier_) return false;
  switch (typeSpecifier_->kind()) {
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
  if (!typeSpecifier_) return false;
  switch (typeSpecifier_->kind()) {
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