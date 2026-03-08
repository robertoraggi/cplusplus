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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {

namespace {

[[nodiscard]] auto is_within_std_namespace(Symbol* symbol) -> bool {
  if (!symbol) return false;
  auto parent = symbol->parent();
  while (parent) {
    if (auto ns = symbol_cast<NamespaceSymbol>(parent)) {
      if (auto id = name_cast<Identifier>(ns->name())) {
        if (id->name() == "std" || id->name() == "__1" ||
            id->name() == "__cxx11")
          return true;
      }
    }
    parent = parent->parent();
  }
  return false;
}

}  // namespace

StandardConversion::StandardConversion(TranslationUnit* unit, bool isC)
    : unit_(unit),
      control_(unit->control()),
      arena_(unit->arena()),
      isC_(isC) {}

auto StandardConversion::checkCvQualifiers(CvQualifiers target,
                                           CvQualifiers source) const -> bool {
  return cv_is_subset_of(source, target);
}

auto StandardConversion::stripCv(const Type*& type) -> CvQualifiers {
  if (auto qualType = type_cast<QualType>(type)) {
    auto cv = qualType->cvQualifiers();
    type = qualType->elementType();
    return cv;
  }
  return {};
}

auto StandardConversion::mergeCv(CvQualifiers cv1, CvQualifiers cv2) const
    -> CvQualifiers {
  return cv1 | cv2;
}

auto StandardConversion::isReferenceCompatible(const Type* targetType,
                                               const Type* sourceType) const
    -> bool {
  auto targetUnqual = unit_->typeTraits().remove_cv(targetType);
  auto sourceUnqual = unit_->typeTraits().remove_cv(sourceType);
  if (!unit_->typeTraits().is_same(targetUnqual, sourceUnqual) &&
      !unit_->typeTraits().is_base_of(targetUnqual, sourceUnqual))
    return false;
  return checkCvQualifiers(unit_->typeTraits().get_cv_qualifiers(targetType),
                           unit_->typeTraits().get_cv_qualifiers(sourceType));
}

auto StandardConversion::isNullPointerConstant(ExpressionAST* expr) const
    -> bool {
  if (!expr) return false;

  for (;;) {
    if (unit_->typeTraits().is_null_pointer(expr->type)) return true;

    if (auto nestedExpr = ast_cast<NestedExpressionAST>(expr)) {
      expr = nestedExpr->expression;
      if (!expr) return false;
      continue;
    }

    if (auto equal = ast_cast<EqualInitializerAST>(expr)) {
      expr = equal->expression;
      if (!expr) return false;
      continue;
    }

    if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
      if (!paren->expressionList || paren->expressionList->next) return false;
      expr = paren->expressionList->value;
      if (!expr) return false;
      continue;
    }

    break;
  }

  if (auto integerLiteral = ast_cast<IntLiteralExpressionAST>(expr))
    return integerLiteral->literal->value() == "0";

  return false;
}

auto StandardConversion::initializerListElementType(
    const Type* targetType) const -> const Type* {
  if (!targetType) return nullptr;

  auto unrefTarget = unit_->typeTraits().remove_reference(targetType);
  auto unqualTarget = unit_->typeTraits().remove_cv(unrefTarget);
  auto classType = type_cast<ClassType>(unqualTarget);
  if (!classType || !classType->symbol()) return nullptr;

  auto classSymbol = classType->symbol();
  auto className = name_cast<Identifier>(classSymbol->name());
  if (!className || className->name() != "initializer_list") return nullptr;
  if (!is_within_std_namespace(classSymbol)) return nullptr;
  if (!classSymbol->isSpecialization()) return nullptr;

  auto args = classSymbol->templateArguments();
  if (args.size() != 1) return nullptr;

  if (auto typeArg = std::get_if<const Type*>(&args[0])) return *typeArg;
  if (auto symbolArg = std::get_if<Symbol*>(&args[0])) {
    auto sym = *symbolArg;
    if (!sym) return nullptr;
    return sym->type();
  }

  return nullptr;
}

auto StandardConversion::lvalueToRvalue(ExpressionAST*& expr) -> bool {
  if (!is_glvalue(expr)) return false;
  if (unit_->typeTraits().is_function(expr->type)) return false;
  if (unit_->typeTraits().is_array(expr->type)) return false;
  if (!unit_->typeTraits().is_complete(expr->type)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
  cast->expression = expr;
  cast->type = unit_->typeTraits().remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::arrayToPointer(ExpressionAST*& expr) -> bool {
  auto unref = unit_->typeTraits().remove_reference(expr->type);
  if (!unit_->typeTraits().is_array(unref)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kArrayToPointerConversion;
  cast->expression = expr;
  cast->type =
      unit_->typeTraits().add_pointer(unit_->typeTraits().remove_extent(unref));
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::functionToPointer(ExpressionAST*& expr) -> bool {
  auto unref = unit_->typeTraits().remove_reference(expr->type);
  if (!unit_->typeTraits().is_function(unref)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
  cast->expression = expr;
  cast->type = unit_->typeTraits().add_pointer(unref);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::integralPromotion(ExpressionAST*& expr,
                                           const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;
  if (!unit_->typeTraits().is_integral(expr->type) &&
      !unit_->typeTraits().is_enum(expr->type))
    return false;

  auto make = [&](const Type* type) {
    auto cast = ImplicitCastExpressionAST::create(arena_);
    cast->castKind = ImplicitCastKind::kIntegralPromotion;
    cast->expression = expr;
    cast->type = type;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  switch (expr->type->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt:
    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar: {
      if (!destinationType) destinationType = control_->getIntType();
      if (destinationType->kind() == TypeKind::kInt ||
          destinationType->kind() == TypeKind::kUnsignedInt) {
        make(destinationType);
        return true;
      }
      return false;
    }

    case TypeKind::kBool: {
      if (!destinationType) destinationType = control_->getIntType();
      if (destinationType->kind() == TypeKind::kInt) {
        make(destinationType);
        return true;
      }
      return false;
    }

    default:
      break;
  }

  if (auto enumType = type_cast<EnumType>(expr->type)) {
    auto type = enumType->underlyingType();
    if (!type) type = control_->getIntType();
    make(type);
    return true;
  }

  return false;
}

auto StandardConversion::floatingPointPromotion(ExpressionAST*& expr,
                                                const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;
  if (!unit_->typeTraits().is_floating_point(expr->type)) return false;
  if (!destinationType) destinationType = control_->getDoubleType();
  if (!unit_->typeTraits().is_floating_point(destinationType)) return false;
  if (expr->type->kind() != TypeKind::kFloat) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kFloatingPointPromotion;
  cast->expression = expr;
  cast->type = control_->getDoubleType();
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::integralConversion(ExpressionAST*& expr,
                                            const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;
  if (!unit_->typeTraits().is_integral_or_unscoped_enum(expr->type))
    return false;
  if (!unit_->typeTraits().is_integer(destinationType)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kIntegralConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::floatingPointConversion(ExpressionAST*& expr,
                                                 const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;
  if (unit_->typeTraits().is_same(expr->type, destinationType)) return true;
  if (!unit_->typeTraits().is_floating_point(expr->type)) return false;
  if (!unit_->typeTraits().is_floating_point(destinationType)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kFloatingPointConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::floatingIntegralConversion(ExpressionAST*& expr,
                                                    const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  auto make = [&] {
    auto cast = ImplicitCastExpressionAST::create(arena_);
    cast->castKind = ImplicitCastKind::kFloatingIntegralConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  if (unit_->typeTraits().is_integral_or_unscoped_enum(expr->type) &&
      unit_->typeTraits().is_floating_point(destinationType)) {
    make();
    return true;
  }

  if (!unit_->typeTraits().is_floating_point(expr->type)) return false;
  if (!unit_->typeTraits().is_integer(destinationType)) return false;
  make();
  return true;
}

auto StandardConversion::ensurePrvalue(ExpressionAST*& expr) -> bool {
  if (lvalueToRvalue(expr)) return true;
  if (arrayToPointer(expr)) return true;
  if (functionToPointer(expr)) return true;
  return false;
}

void StandardConversion::adjustCv(ExpressionAST* expr) {
  if (!is_prvalue(expr)) return;

  auto qualType = type_cast<QualType>(expr->type);
  if (!qualType) return;

  if (unit_->typeTraits().is_class(expr->type) ||
      unit_->typeTraits().is_array(expr->type))
    return;

  expr->type = qualType->elementType();
}

auto StandardConversion::temporaryMaterialization(ExpressionAST*& expr)
    -> bool {
  if (!is_prvalue(expr)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kTemporaryMaterializationConversion;
  cast->expression = expr;
  cast->type = unit_->typeTraits().remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kXValue;
  expr = cast;
  return true;
}

auto StandardConversion::convertImplicitly(ExpressionAST*& expr,
                                           const Type* destinationType)
    -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  auto seq = computeConversionSequence(expr, destinationType);
  if (!seq) return false;

  applyConversionSequence(seq, expr);
  adjustCv(expr);
  return true;
}

auto StandardConversion::usualArithmeticConversion(ExpressionAST*& expr,
                                                   ExpressionAST*& other)
    -> const Type* {
  if (!unit_->typeTraits().is_arithmetic(expr->type) &&
      !unit_->typeTraits().is_enum(expr->type))
    return nullptr;
  if (!unit_->typeTraits().is_arithmetic(other->type) &&
      !unit_->typeTraits().is_enum(other->type))
    return nullptr;

  (void)lvalueToRvalue(expr);
  adjustCv(expr);
  (void)lvalueToRvalue(other);
  adjustCv(other);

  ExpressionAST* savedExpr = expr;
  ExpressionAST* savedOther = other;

  auto unmodified = [&]() -> const Type* {
    expr = savedExpr;
    other = savedOther;
    return nullptr;
  };

  if (unit_->typeTraits().is_scoped_enum(expr->type) ||
      unit_->typeTraits().is_scoped_enum(other->type))
    return unmodified();

  if (unit_->typeTraits().is_floating_point(expr->type) ||
      unit_->typeTraits().is_floating_point(other->type)) {
    if (unit_->typeTraits().is_same(expr->type, other->type)) return expr->type;

    if (!unit_->typeTraits().is_floating_point(expr->type)) {
      if (floatingIntegralConversion(expr, other->type)) return other->type;
      return unmodified();
    }

    if (!unit_->typeTraits().is_floating_point(other->type)) {
      if (floatingIntegralConversion(other, expr->type)) return expr->type;
      return unmodified();
    }

    if (expr->type->kind() == TypeKind::kLongDouble ||
        other->type->kind() == TypeKind::kLongDouble) {
      (void)floatingPointConversion(expr, control_->getLongDoubleType());
      (void)floatingPointConversion(other, control_->getLongDoubleType());
      return control_->getLongDoubleType();
    }

    if (expr->type->kind() == TypeKind::kDouble ||
        other->type->kind() == TypeKind::kDouble) {
      (void)floatingPointConversion(expr, control_->getDoubleType());
      (void)floatingPointConversion(other, control_->getDoubleType());
      return control_->getDoubleType();
    }

    return unmodified();
  }

  (void)integralPromotion(expr);
  (void)integralPromotion(other);

  if (unit_->typeTraits().is_same(expr->type, other->type)) return expr->type;

  auto matchType = [&](const Type* type) -> bool {
    if (expr->type->kind() == type->kind() ||
        other->type->kind() == type->kind()) {
      (void)integralConversion(expr, type);
      (void)integralConversion(other, type);
      return true;
    }
    return false;
  };

  if (unit_->typeTraits().is_signed(expr->type) &&
      unit_->typeTraits().is_signed(other->type)) {
    if (matchType(control_->getLongLongIntType()))
      return control_->getLongLongIntType();
    if (matchType(control_->getLongIntType()))
      return control_->getLongIntType();
    (void)integralConversion(expr, control_->getIntType());
    (void)integralConversion(other, control_->getIntType());
    return control_->getIntType();
  }

  if (unit_->typeTraits().is_unsigned(expr->type) &&
      unit_->typeTraits().is_unsigned(other->type)) {
    if (matchType(control_->getUnsignedLongLongIntType()))
      return control_->getUnsignedLongLongIntType();
    if (matchType(control_->getUnsignedLongIntType()))
      return control_->getUnsignedLongIntType();
    (void)integralConversion(expr, control_->getUnsignedIntType());
    return control_->getUnsignedIntType();
  }

  if (matchType(control_->getUnsignedLongLongIntType()))
    return control_->getUnsignedLongLongIntType();
  if (matchType(control_->getUnsignedLongIntType()))
    return control_->getUnsignedLongIntType();
  if (matchType(control_->getUnsignedIntType()))
    return control_->getUnsignedIntType();
  if (matchType(control_->getUnsignedShortIntType()))
    return control_->getUnsignedShortIntType();
  if (matchType(control_->getUnsignedCharType()))
    return control_->getUnsignedCharType();
  if (matchType(control_->getLongLongIntType()))
    return control_->getLongLongIntType();
  if (matchType(control_->getLongIntType())) return control_->getLongIntType();

  (void)integralConversion(expr, control_->getIntType());
  (void)integralConversion(other, control_->getIntType());
  return control_->getIntType();
}

auto StandardConversion::getQualificationCombinedType(const Type* left,
                                                      const Type* right)
    -> const Type* {
  bool didChange = false;
  return getQualificationCombinedType(left, right, didChange);
}

auto StandardConversion::getQualificationCombinedType(
    const Type* left, const Type* right, bool& didChangeTypeOrQualifiers)
    -> const Type* {
  auto cv1 = stripCv(left);
  auto cv2 = stripCv(right);

  auto bothPointerOrArray = [&] {
    if (unit_->typeTraits().is_pointer(left) &&
        unit_->typeTraits().is_pointer(right))
      return true;
    if (unit_->typeTraits().is_array(left) &&
        unit_->typeTraits().is_array(right))
      return true;
    return false;
  };

  if (!bothPointerOrArray()) {
    const auto cv3 = mergeCv(cv1, cv2);

    if (unit_->typeTraits().is_same(left, right))
      return unit_->typeTraits().add_cv(left, cv3);
    if (unit_->typeTraits().is_base_of(left, right))
      return unit_->typeTraits().add_cv(left, cv1);
    if (unit_->typeTraits().is_base_of(right, left))
      return unit_->typeTraits().add_cv(right, cv2);
    return nullptr;
  }

  auto leftElem = unit_->typeTraits().get_element_type(left);
  if (unit_->typeTraits().is_array(leftElem))
    cv1 = mergeCv(cv1, unit_->typeTraits().get_cv_qualifiers(leftElem));

  auto rightElem = unit_->typeTraits().get_element_type(right);
  if (unit_->typeTraits().is_array(rightElem))
    cv2 = mergeCv(cv2, unit_->typeTraits().get_cv_qualifiers(rightElem));

  auto elemType = getQualificationCombinedType(leftElem, rightElem,
                                               didChangeTypeOrQualifiers);
  if (!elemType) return nullptr;

  auto cv3 = mergeCv(cv1, cv2);
  if (didChangeTypeOrQualifiers) cv3 = cv3 | CvQualifiers::kConst;
  if (cv1 != cv3 || cv2 != cv3) didChangeTypeOrQualifiers = true;
  elemType = unit_->typeTraits().add_cv(elemType, cv3);

  if (unit_->typeTraits().is_array(left) &&
      unit_->typeTraits().is_array(right)) {
    auto leftArr = type_cast<BoundedArrayType>(left);
    auto rightArr = type_cast<BoundedArrayType>(right);

    if (leftArr && rightArr) {
      if (leftArr->size() != rightArr->size()) return nullptr;
      return control_->getBoundedArrayType(elemType, leftArr->size());
    }

    if (leftArr || rightArr) didChangeTypeOrQualifiers = true;
    return control_->getUnboundedArrayType(elemType);
  }

  return control_->getPointerType(elemType);
}

auto StandardConversion::compositePointerType(ExpressionAST*& expr,
                                              ExpressionAST*& other)
    -> const Type* {
  if (unit_->typeTraits().is_null_pointer(expr->type) &&
      unit_->typeTraits().is_null_pointer(other->type))
    return control_->getNullptrType();

  if (isNullPointerConstant(expr)) return other->type;
  if (isNullPointerConstant(other)) return expr->type;

  if (unit_->typeTraits().is_pointer(expr->type) &&
      unit_->typeTraits().is_pointer(other->type)) {
    auto t1 = unit_->typeTraits().get_element_type(expr->type);
    const auto cv1 = stripCv(t1);
    auto t2 = unit_->typeTraits().get_element_type(other->type);
    const auto cv2 = stripCv(t2);

    if (unit_->typeTraits().is_void(t1))
      return control_->getPointerType(unit_->typeTraits().add_cv(t1, cv2));
    if (unit_->typeTraits().is_void(t2))
      return control_->getPointerType(unit_->typeTraits().add_cv(t2, cv1));

    if (auto type = getQualificationCombinedType(expr->type, other->type))
      return type;
  }

  return nullptr;
}

auto StandardConversion::isIntegralPromotion(const Type* sourceType,
                                             const Type* targetType) const
    -> bool {
  if (!sourceType || !targetType) return false;
  auto src = unit_->typeTraits().remove_cv(sourceType);
  auto dst = unit_->typeTraits().remove_cv(targetType);

  switch (src->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt:
    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar:
      return dst->kind() == TypeKind::kInt ||
             dst->kind() == TypeKind::kUnsignedInt;
    case TypeKind::kBool:
      return dst->kind() == TypeKind::kInt;
    default:
      break;
  }

  if (auto enumType = type_cast<EnumType>(src)) {
    if (unit_->typeTraits().is_scoped_enum(src)) return false;
    return unit_->typeTraits().is_integral(dst);
  }

  return false;
}

auto StandardConversion::isFloatingPointPromotion(const Type* sourceType,
                                                  const Type* targetType) const
    -> bool {
  if (!sourceType || !targetType) return false;
  auto src = unit_->typeTraits().remove_cv(sourceType);
  auto dst = unit_->typeTraits().remove_cv(targetType);
  return src->kind() == TypeKind::kFloat && dst->kind() == TypeKind::kDouble;
}

auto StandardConversion::computeConversionSequence(ExpressionAST* expr,
                                                   const Type* targetType)
    -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  if (!expr || !targetType) return seq;

  const Type* currentType = expr->type;
  ValueCategory currentValCat = expr->valueCategory;

  auto addStep = [&](ImplicitCastKind kind, const Type* type) {
    seq.steps.push_back({kind, type});
  };

  if (auto bracedInitList = ast_cast<BracedInitListAST>(expr)) {
    auto elemType = initializerListElementType(targetType);
    if (!elemType) return seq;

    for (auto it = bracedInitList->expressionList; it; it = it->next) {
      if (!it->value || !it->value->type) return seq;
      auto elemSeq = computeConversionSequence(it->value, elemType);
      if (!elemSeq) return seq;
    }

    seq.rank = ConversionRank::kExactMatch;
    addStep(ImplicitCastKind::kIdentity, targetType);
    return seq;
  }

  if (unit_->typeTraits().is_reference(targetType)) {
    if (auto rvalRef = type_cast<RvalueReferenceType>(targetType)) {
      if (currentValCat == ValueCategory::kLValue) {
        auto sourceRefRemoved =
            unit_->typeTraits().remove_reference(currentType);
        auto targetElem = rvalRef->elementType();

        if (!unit_->typeTraits().is_function(
                unit_->typeTraits().remove_reference(targetElem)))
          return seq;

        if (!isReferenceCompatible(targetElem, sourceRefRemoved)) return seq;

        auto sameUnqual = unit_->typeTraits().is_same(
            unit_->typeTraits().remove_cv(sourceRefRemoved),
            unit_->typeTraits().remove_cv(targetElem));
        auto sourceCv = unit_->typeTraits().get_cv_qualifiers(sourceRefRemoved);
        auto targetCv = unit_->typeTraits().get_cv_qualifiers(targetElem);

        seq.bindsToRvalueRef = true;
        seq.bindsToReference = true;
        seq.referenceCv = targetCv;
        seq.rank = sameUnqual ? ConversionRank::kExactMatch
                              : ConversionRank::kConversion;
        addStep((sameUnqual && sourceCv != targetCv)
                    ? ImplicitCastKind::kQualificationConversion
                    : ImplicitCastKind::kIdentity,
                targetElem);
        return seq;
      }
      seq.bindsToRvalueRef = true;
    }

    if (auto lvalRef = type_cast<LvalueReferenceType>(targetType)) {
      auto inner = lvalRef->elementType();
      bool isConst = false;
      if (auto qual = type_cast<QualType>(inner)) isConst = qual->isConst();

      auto sourceRefRemoved = unit_->typeTraits().remove_reference(currentType);

      if (!isConst) {
        if (currentValCat != ValueCategory::kLValue) return seq;
        if (!isReferenceCompatible(inner, sourceRefRemoved)) return seq;

        auto sameUnqual = unit_->typeTraits().is_same(
            unit_->typeTraits().remove_cv(sourceRefRemoved),
            unit_->typeTraits().remove_cv(inner));
        auto sourceCv = unit_->typeTraits().get_cv_qualifiers(sourceRefRemoved);
        auto targetCv = unit_->typeTraits().get_cv_qualifiers(inner);

        seq.bindsToReference = true;
        seq.referenceCv = targetCv;
        seq.rank = sameUnqual ? ConversionRank::kExactMatch
                              : ConversionRank::kConversion;
        addStep((sameUnqual && sourceCv != targetCv)
                    ? ImplicitCastKind::kQualificationConversion
                    : ImplicitCastKind::kIdentity,
                inner);
        return seq;
      }

      if (currentValCat == ValueCategory::kLValue &&
          isReferenceCompatible(inner, sourceRefRemoved)) {
        auto sameUnqual = unit_->typeTraits().is_same(
            unit_->typeTraits().remove_cv(sourceRefRemoved),
            unit_->typeTraits().remove_cv(inner));
        auto sourceCv = unit_->typeTraits().get_cv_qualifiers(sourceRefRemoved);
        auto targetCv = unit_->typeTraits().get_cv_qualifiers(inner);

        seq.bindsToReference = true;
        seq.referenceCv = targetCv;
        seq.rank = sameUnqual ? ConversionRank::kExactMatch
                              : ConversionRank::kConversion;
        addStep((sameUnqual && sourceCv != targetCv)
                    ? ImplicitCastKind::kQualificationConversion
                    : ImplicitCastKind::kIdentity,
                inner);
        return seq;
      }

      if (!isConst && currentValCat != ValueCategory::kLValue) return seq;
    }
  }

  if (unit_->typeTraits().is_array(
          unit_->typeTraits().remove_reference(currentType))) {
    auto unref = unit_->typeTraits().remove_reference(currentType);
    currentType = unit_->typeTraits().add_pointer(
        unit_->typeTraits().remove_extent(unref));
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitCastKind::kArrayToPointerConversion, currentType);
  } else if (unit_->typeTraits().is_function(
                 unit_->typeTraits().remove_reference(currentType))) {
    auto unref = unit_->typeTraits().remove_reference(currentType);
    currentType = unit_->typeTraits().add_pointer(unref);
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitCastKind::kFunctionToPointerConversion, currentType);
  } else if (currentValCat != ValueCategory::kPrValue &&
             !unit_->typeTraits().is_reference(targetType)) {
    currentType = unit_->typeTraits().remove_reference(currentType);
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitCastKind::kLValueToRValueConversion, currentType);
  }

  auto comparisonTargetType = unit_->typeTraits().remove_reference(targetType);

  auto unqualFrom = unit_->typeTraits().remove_cv(currentType);
  auto unqualTo = unit_->typeTraits().remove_cv(comparisonTargetType);

  if (unit_->typeTraits().is_same(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kExactMatch;
    addStep(ImplicitCastKind::kIdentity, comparisonTargetType);
    return seq;
  }

  if (unit_->typeTraits().is_null_pointer(unqualFrom) &&
      unit_->typeTraits().is_pointer(unqualTo)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerConversion, targetType);
    return seq;
  }

  if (unit_->typeTraits().is_integral(unqualFrom) &&
      unit_->typeTraits().is_pointer(unqualTo) && isNullPointerConstant(expr)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerConversion, targetType);
    return seq;
  }

  if (unit_->typeTraits().is_pointer(unqualFrom) &&
      unit_->typeTraits().is_pointer(unqualTo)) {
    auto fromPtr =
        type_cast<PointerType>(unit_->typeTraits().remove_cv(unqualFrom));
    auto toPtr =
        type_cast<PointerType>(unit_->typeTraits().remove_cv(unqualTo));

    if (fromPtr && toPtr) {
      auto fromPointee = fromPtr->elementType();
      auto toPointee = toPtr->elementType();

      auto fromCv = unit_->typeTraits().get_cv_qualifiers(fromPointee);
      auto toCv = unit_->typeTraits().get_cv_qualifiers(toPointee);

      if (cv_is_subset_of(fromCv, toCv)) {
        auto fromUnqual = unit_->typeTraits().remove_cv(fromPointee);
        auto toUnqual = unit_->typeTraits().remove_cv(toPointee);

        if (unit_->typeTraits().is_same(fromUnqual, toUnqual)) {
          seq.rank = ConversionRank::kExactMatch;
          addStep(ImplicitCastKind::kQualificationConversion, targetType);
          return seq;
        }

        if (unit_->typeTraits().is_void(toUnqual)) {
          seq.rank = ConversionRank::kConversion;
          addStep(ImplicitCastKind::kPointerConversion, targetType);
          return seq;
        }

        if (unit_->typeTraits().is_class(fromUnqual) &&
            unit_->typeTraits().is_class(toUnqual)) {
          if (unit_->typeTraits().is_base_of(toUnqual, fromUnqual)) {
            seq.rank = ConversionRank::kConversion;
            addStep(ImplicitCastKind::kPointerConversion, targetType);
            return seq;
          }
        }

        // C mode: void* -> T*
        if (isC_ && unit_->typeTraits().is_void(fromUnqual)) {
          seq.rank = ConversionRank::kConversion;
          addStep(ImplicitCastKind::kPointerConversion, targetType);
          return seq;
        }
      }
    }
  }

  if (isIntegralPromotion(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kPromotion;
    addStep(ImplicitCastKind::kIntegralPromotion, targetType);
    return seq;
  }

  if (isFloatingPointPromotion(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kPromotion;
    addStep(ImplicitCastKind::kFloatingPointPromotion, targetType);
    return seq;
  }

  if ((unit_->typeTraits().is_arithmetic(unqualFrom) ||
       (unit_->typeTraits().is_enum(unqualFrom) &&
        !unit_->typeTraits().is_scoped_enum(unqualFrom))) &&
      unit_->typeTraits().is_arithmetic(unqualTo)) {
    seq.rank = ConversionRank::kConversion;

    if (unit_->typeTraits().is_integral_or_unscoped_enum(unqualFrom) &&
        unit_->typeTraits().is_integral(unqualTo)) {
      addStep(ImplicitCastKind::kIntegralConversion, targetType);
      return seq;
    }

    if (unit_->typeTraits().is_floating_point(unqualFrom) &&
        unit_->typeTraits().is_floating_point(unqualTo)) {
      addStep(ImplicitCastKind::kFloatingPointConversion, targetType);
      return seq;
    }

    addStep(ImplicitCastKind::kFloatingIntegralConversion, targetType);
    return seq;
  }

  if (unit_->typeTraits().is_member_pointer(unqualFrom) &&
      unit_->typeTraits().is_member_pointer(unqualTo)) {
    if (auto srcMop = type_cast<MemberObjectPointerType>(unqualFrom)) {
      if (auto dstMop = type_cast<MemberObjectPointerType>(unqualTo)) {
        if (unit_->typeTraits().is_same(
                unit_->typeTraits().remove_cv(srcMop->elementType()),
                unit_->typeTraits().remove_cv(dstMop->elementType())) &&
            unit_->typeTraits().is_base_of(dstMop->classType(),
                                           srcMop->classType())) {
          seq.rank = ConversionRank::kConversion;
          addStep(ImplicitCastKind::kPointerToMemberConversion, targetType);
          return seq;
        }
      }
    }
  }

  // null pointer constant -> pointer-to-member
  if (unit_->typeTraits().is_member_pointer(unqualTo) &&
      isNullPointerConstant(expr)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerToMemberConversion, targetType);
    return seq;
  }

  if (unit_->typeTraits().is_pointer(unqualFrom) &&
      unit_->typeTraits().is_pointer(unqualTo)) {
    auto srcPtr = type_cast<PointerType>(unqualFrom);
    auto dstPtr = type_cast<PointerType>(unqualTo);
    if (srcPtr && dstPtr) {
      auto srcFunc = type_cast<FunctionType>(srcPtr->elementType());
      auto dstFunc = type_cast<FunctionType>(dstPtr->elementType());
      if (srcFunc && dstFunc && srcFunc->isNoexcept() &&
          !dstFunc->isNoexcept() &&
          unit_->typeTraits().is_same(
              unit_->typeTraits().remove_noexcept(srcFunc), dstFunc)) {
        seq.rank = ConversionRank::kExactMatch;
        addStep(ImplicitCastKind::kFunctionPointerConversion, targetType);
        return seq;
      }
    }
  }

  if (unit_->typeTraits().is_same(unqualTo, control_->getBoolType())) {
    if (unit_->typeTraits().is_arithmetic_or_unscoped_enum(unqualFrom) ||
        unit_->typeTraits().is_pointer(unqualFrom) ||
        unit_->typeTraits().is_member_pointer(unqualFrom)) {
      seq.rank = ConversionRank::kConversion;
      addStep(ImplicitCastKind::kBooleanConversion, targetType);
      return seq;
    }
  }

  if (isC_ && unit_->typeTraits().is_integral_or_unscoped_enum(unqualFrom) &&
      unit_->typeTraits().is_enum(unqualTo) &&
      !unit_->typeTraits().is_scoped_enum(unqualTo)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kIntegralConversion, targetType);
    return seq;
  }

  auto makeUserDefinedSeq =
      [&](FunctionSymbol* func,
          ConversionRank s2Rank) -> ImplicitConversionSequence {
    ImplicitConversionSequence uds;
    uds.kind = ConversionSequenceKind::kUserDefined;
    uds.rank = ConversionRank::kConversion;
    uds.userDefinedConversionFunction = func;
    uds.secondStandardConversionRank = s2Rank;
    uds.steps.push_back(
        {ImplicitCastKind::kUserDefinedConversion, comparisonTargetType});
    return uds;
  };

  ImplicitConversionSequence bestUserDefined;

  auto checkViability = [&](const Type* from,
                            const Type* to) -> std::pair<bool, ConversionRank> {
    if (unit_->typeTraits().is_same(from, to))
      return {true, ConversionRank::kExactMatch};
    if (unit_->typeTraits().is_arithmetic(from) &&
        unit_->typeTraits().is_arithmetic(to))
      return {true, ConversionRank::kConversion};
    if (unit_->typeTraits().is_pointer(from) &&
        unit_->typeTraits().is_pointer(to))
      return {true, ConversionRank::kConversion};
    if (unit_->typeTraits().is_null_pointer(from) &&
        unit_->typeTraits().is_pointer(to))
      return {true, ConversionRank::kConversion};
    return {false, ConversionRank::kNone};
  };

  auto updateBest = [&](FunctionSymbol* func, ConversionRank s2Rank) {
    auto uds = makeUserDefinedSeq(func, s2Rank);
    if (!bestUserDefined || uds.isBetterThan(bestUserDefined))
      bestUserDefined = uds;
  };

  if (auto destClassType = type_cast<ClassType>(unqualTo)) {
    if (auto destClass = destClassType->symbol()) {
      for (auto ctor : destClass->convertingConstructors()) {
        auto funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (params.size() != 1) continue;

        auto paramUnqual = unit_->typeTraits().remove_cv(
            unit_->typeTraits().remove_reference(params[0]));

        auto [viable, s2Rank] = checkViability(unqualFrom, paramUnqual);
        if (viable) updateBest(ctor, s2Rank);
      }
    }
  }

  if (auto srcClassType = type_cast<ClassType>(unqualFrom)) {
    if (auto srcClass = srcClassType->symbol()) {
      for (auto convFunc : srcClass->conversionFunctions()) {
        auto convFuncType = type_cast<FunctionType>(convFunc->type());
        if (!convFuncType) continue;

        auto returnType = convFuncType->returnType();
        if (!returnType) continue;

        auto retUnqual = unit_->typeTraits().remove_cv(returnType);

        auto [viable, s2Rank] = checkViability(retUnqual, unqualTo);
        if (!viable &&
            unit_->typeTraits().is_same(unqualTo, control_->getBoolType())) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        }
        if (viable) updateBest(convFunc, s2Rank);
      }
    }
  }

  if (bestUserDefined) return bestUserDefined;

  return seq;
}

void StandardConversion::applyConversionSequence(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  if (sequence.rank == ConversionRank::kNone) return;

  for (const auto& step : sequence.steps) {
    if (step.kind != ImplicitCastKind::kIdentity)
      wrapWithImplicitCast(step.kind, step.type, expr);
  }
}

void StandardConversion::wrapWithImplicitCast(ImplicitCastKind castKind,
                                              const Type* type,
                                              ExpressionAST*& expr) {
  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = castKind;
  cast->expression = expr;
  cast->type = type;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
}

auto StandardConversion::isNarrowingConversion(const Type* from, const Type* to)
    -> bool {
  if (isC_) return false;

  from = unit_->typeTraits().remove_cv(from);
  to = unit_->typeTraits().remove_cv(to);

  if (unit_->typeTraits().is_same(from, to)) return false;

  if (unit_->typeTraits().is_floating_point(from) &&
      unit_->typeTraits().is_integral(to))
    return true;

  if (unit_->typeTraits().is_floating_point(from) &&
      unit_->typeTraits().is_floating_point(to)) {
    auto fromSize = control_->memoryLayout()->sizeOf(from);
    auto toSize = control_->memoryLayout()->sizeOf(to);
    if (fromSize && toSize && *fromSize > *toSize) return true;
  }

  if (unit_->typeTraits().is_integral_or_unscoped_enum(from) &&
      unit_->typeTraits().is_floating_point(to))
    return true;

  if (unit_->typeTraits().is_integral_or_unscoped_enum(from) &&
      unit_->typeTraits().is_integral(to)) {
    auto fromSize = control_->memoryLayout()->sizeOf(from);
    auto toSize = control_->memoryLayout()->sizeOf(to);
    if (fromSize && toSize) {
      if (*fromSize > *toSize) return true;
      if (*fromSize == *toSize && unit_->typeTraits().is_signed(from) !=
                                      unit_->typeTraits().is_signed(to))
        return true;
    }
  }

  return false;
}

}  // namespace cxx
