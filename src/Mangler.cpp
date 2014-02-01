// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Mangler.h"
#include "Types.h"
#include "Names.h"
#include "Symbols.h"
#include <cassert>

std::string Mangler::encode(FunctionSymbol* symbol) {
  auto funTy = symbol->type()->asFunctionType();
  assert(funTy);
  return "_Z" + mangleName(symbol->name(), symbol->type()) + mangleBareFunctionType(funTy);
}

//
// names
//
std::string Mangler::mangleName(const Name* name, const QualType& type) {
  switch (name->kind()) {
  default: assert(!"unreachable"); return "!";
#define VISIT_NAME(T) case NameKind::k##T: return mangle##T(name->as##T(), type);
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME
  } // switch
}

std::string Mangler::mangleIdentifier(const Identifier* name, const QualType& type) {
  return std::to_string(name->size()) + name->toString();
}

std::string Mangler::mangleDestructorName(const DestructorName* name, const QualType& type) {
  return "@destructor-name@";
}

std::string Mangler::mangleOperatorName(const OperatorName* name, const QualType& type) {
  bool unary = false;
  if (auto funTy = type->asFunctionType())
    unary = funTy->argumentTypes().size() == 1;
  switch (name->op()) {
  default: assert(!"unreachable"); return "!";
  case T_NEW: return "nw";
  case T_NEW_ARRAY: return "na";
  case T_DELETE: return "dl";
  case T_DELETE_ARRAY: return "da";
  case T_PLUS: return unary ? "ps" : "pl";
  case T_MINUS: return unary ? "ng" : "mi";
  case T_AMP: return unary ? "ad" : "an";
  case T_STAR: return unary ? "de" : "ml";
  case T_TILDE: return "co";
  case T_SLASH: return "dv";
  case T_PERCENT: return "rm";
  case T_BAR: return "or";
  case T_CARET: return "eo";
  case T_EQUAL: return "aS";
  case T_PLUS_EQUAL: return "pL";
  case T_MINUS_EQUAL: return "mI";
  case T_STAR_EQUAL: return "mL";
  case T_SLASH_EQUAL: return "dV";
  case T_PERCENT_EQUAL: return "rM";
  case T_AMP_EQUAL: return "aN";
  case T_BAR_EQUAL: return "oR";
  case T_CARET_EQUAL: return "eO";
  case T_LESS_LESS: return "ls";
  case T_GREATER_GREATER: return "rs";
  case T_LESS_LESS_EQUAL: return "lS";
  case T_GREATER_GREATER_EQUAL: return "rS";
  case T_EQUAL_EQUAL: return "eq";
  case T_EXCLAIM_EQUAL: return "ne";
  case T_LESS: return "lt";
  case T_GREATER: return "gt";
  case T_LESS_EQUAL: return "le";
  case T_GREATER_EQUAL: return "ge";
  case T_EXCLAIM: return "nt";
  case T_AMP_AMP: return "aa";
  case T_BAR_BAR: return "oo";
  case T_PLUS_PLUS: return "pp";
  case T_MINUS_MINUS: return "mm";
  case T_COMMA: return "cm";
  case T_MINUS_GREATER_STAR: return "pm";
  case T_MINUS_GREATER: return "pt";
  case T_LPAREN: return "cl";
  case T_LBRACKET: return "ix";
  case T_QUESTION: return "qu";
  } // switch

  return "!";
}

std::string Mangler::mangleQualifiedName(const QualifiedName* name, const QualType& type) {
  return "@qualified-name@";
}

std::string Mangler::mangleTemplateName(const TemplateName* name, const QualType& type) {
  return "@template-name@";
}

//
// types
//
std::string Mangler::mangleType(const QualType& type) {
  switch (type->kind()) {
  default: assert(!"unreachable"); return "!";
#define VISIT_TYPE(T) case TypeKind::k##T: return mangle##T##Type(type->as##T##Type());
FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
  } // switch
}

std::string Mangler::mangleUndefinedType(const UndefinedType* type) {
  return "@undefined-type@";
}

std::string Mangler::mangleAutoType(const AutoType* type) {
  return "@auto-type@";
}

std::string Mangler::mangleVoidType(const VoidType* type) {
  return "v";
}

std::string Mangler::mangleNullptrType(const NullptrType* type) {
  return "@nullptr-type@";
}

std::string Mangler::mangleIntegerType(const IntegerType* type) {
  switch (type->integerKind()) {
  case IntegerKind::kSignedChar: return "a";
  case IntegerKind::kShortInt: return "s";
  case IntegerKind::kInt: return "i";
  case IntegerKind::kLongInt: return "l";
  case IntegerKind::kLongLongInt: return "x";
  case IntegerKind::kUnsignedChar: return "h";
  case IntegerKind::kUnsignedShortInt: return "t";
  case IntegerKind::kUnsignedInt: return "j";
  case IntegerKind::kUnsignedLongInt: return "m";
  case IntegerKind::kUnsignedLongLongInt: return "y";
  case IntegerKind::kWCharT: return "w";
  case IntegerKind::kChar: return "c";
  case IntegerKind::kChar16T: return "Ds";
  case IntegerKind::kChar32T: return "Di";
  case IntegerKind::kBool: return "b";
  case IntegerKind::kInt128: return "n";
  case IntegerKind::kUnsignedInt128: return "o";
  default: assert(!"unreachable"); return "!";
  } // switch
}

std::string Mangler::mangleFloatType(const FloatType* type) {
  switch (type->floatKind()) {
  case FloatKind::kFloat: return "f";
  case FloatKind::kDouble: return "d";
  case FloatKind::kLongDouble: return "e";
  case FloatKind::kFloat128: return "g";
  default: assert(!"unreachable"); return "!";
  } // switch
}

std::string Mangler::manglePointerType(const PointerType* type) {
  return "P" + mangleType(type->elementType());
}

std::string Mangler::mangleLValueReferenceType(const LValueReferenceType* type) {
  return "R" + mangleType(type->elementType());
}

std::string Mangler::mangleRValueReferenceType(const RValueReferenceType* type) {
  return "O" + mangleType(type->elementType());
}

std::string Mangler::mangleBoundedArrayType(const BoundedArrayType* type) {
  return "@array@";
}

std::string Mangler::mangleUnboundedArrayType(const UnboundedArrayType* type) {
  return "@unbounded-array@";
}

std::string Mangler::mangleBareFunctionType(const FunctionType* funTy) {
  std::string sig;
  for (auto&& argTy: funTy->argumentTypes()) {
    sig += mangleType(argTy);
  }
  if (funTy->isVariadic())
    sig += 'z';
  return sig;
}

std::string Mangler::mangleFunctionType(const FunctionType* type) {
  std::string sig;
  sig += 'F';
  sig += mangleBareFunctionType(type);
  sig += 'E';
  return sig;
}

std::string Mangler::mangleClassType(const ClassType* type) {
  return "@class-type@";
}

std::string Mangler::mangleNamedType(const NamedType* type) {
  return "@named-type@";
}

std::string Mangler::mangleElaboratedType(const ElaboratedType* type) {
  return "@elaborated-type@";
}
