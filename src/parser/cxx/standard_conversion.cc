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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {
namespace {
void bindResultToReference(ImplicitConversionSequence& seq,
                           const Type* targetType) {
  const Type* referencedType = nullptr;
  if (auto rvalueRef = type_cast<RvalueReferenceType>(targetType)) {
    seq.bindsToRvalueRef = true;
    referencedType = rvalueRef->elementType();
  } else if (auto lvalueRef = type_cast<LvalueReferenceType>(targetType)) {
    referencedType = lvalueRef->elementType();
  }

  if (!referencedType) return;

  seq.bindsToReference = true;
  if (auto qual = type_cast<QualType>(referencedType))
    seq.referenceCv = qual->cvQualifiers();
}

[[nodiscard]] auto resolveOverloadSetAgainstFunctionType(
    TranslationUnit* unit, OverloadSetSymbol* ovl,
    const FunctionType* targetFunctionType, SourceLocation loc)
    -> FunctionSymbol* {
  FunctionSymbol* match = nullptr;

  for (auto func : ovl->functions()) {
    if (func->canonical() != func) continue;

    FunctionSymbol* candidate = func;

    if (func->templateDeclaration() && !func->isSpecialization()) {
      TemplateArgumentDeduction deduction(unit);
      auto deducedArgs =
          deduction.deduceFromTargetType(func, targetFunctionType);
      if (!deducedArgs.has_value()) continue;

      candidate = ASTRewriter::instantiateForArgs(unit, *deducedArgs, func, loc,
                                                  /*argsComplete=*/true);
      if (!candidate) continue;
    } else if (func->isSpecialization()) {
      continue;
    }

    auto candidateType = type_cast<FunctionType>(candidate->type());
    if (!candidateType) continue;
    if (!unit->typeTraits().is_same(candidateType, targetFunctionType))
      continue;

    if (match && match != candidate) return nullptr;
    match = candidate;
  }

  return match;
}

struct MemberPointerParts {
  const ClassType* classType = nullptr;
  const Type* pointeeType = nullptr;

  [[nodiscard]] explicit operator bool() const { return classType; }
};

struct DecomposeMemberPointer {
  auto operator()(const MemberObjectPointerType* type) const
      -> MemberPointerParts {
    return {type->classType(), type->elementType()};
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> MemberPointerParts {
    return {type->classType(), type->functionType()};
  }

  auto operator()(const Type*) const -> MemberPointerParts { return {}; }
};

[[nodiscard]] auto decomposeMemberPointer(const Type* type)
    -> MemberPointerParts {
  if (!type) return {};
  return visit(DecomposeMemberPointer{}, type);
}
}  // namespace

StandardConversion::StandardConversion(TranslationUnit* unit, bool isC)
    : unit_(unit),
      traits(unit),
      control_(unit->control()),
      arena_(unit->arena()),
      isC_(isC) {}

auto StandardConversion::checkCvQualifiers(CvQualifiers target,
                                           CvQualifiers source) const -> bool {
  return cv_is_subset_of(source, target);
}

auto StandardConversion::instantiateConversionFunctionTemplate(
    FunctionSymbol* convFunc, const Type* targetType, ExpressionAST* expr)
    -> FunctionSymbol* {
  TemplateArgumentDeduction deduction(unit_);
  auto deducedArgs = deduction.deduceFromConversionTarget(convFunc, targetType);
  if (!deducedArgs.has_value()) return nullptr;

  return ASTRewriter::instantiateForArgs(unit_, *deducedArgs, convFunc,
                                         expr->firstSourceLocation(),
                                         /*argsComplete=*/true);
}

auto StandardConversion::hasUniqueNonVirtualBase(const ClassType* derived,
                                                 const ClassType* base)
    -> bool {
  if (!derived || !base) return false;

  auto derivedClass = derived->symbol();
  auto baseClass = base->symbol();
  if (!derivedClass || !baseClass) return false;

  auto derivedDefinition = derivedClass->resolvedDefinition();
  traits.requireCompleteClass(derivedDefinition);

  return derivedDefinition->baseClassOffset(baseClass).has_value();
}

auto StandardConversion::decompositionCvQualifiers(const Type* type) const
    -> CvQualifiers {
  if (traits.is_array(type))
    return traits.get_cv_qualifiers(traits.get_element_type(type));
  return traits.get_cv_qualifiers(type);
}

auto StandardConversion::hasSimilarArrayBound(const Type* source,
                                              const Type* target) const
    -> bool {
  if (!traits.is_array(source) || !traits.is_array(target)) return false;

  auto targetBounded = type_cast<BoundedArrayType>(target);
  if (!targetBounded) return true;

  auto sourceBounded = type_cast<BoundedArrayType>(source);
  if (!sourceBounded) return false;

  return sourceBounded->size() == targetBounded->size();
}

auto StandardConversion::isQualificationConversion(const Type* source,
                                                   const Type* target,
                                                   int level) const -> bool {
  std::vector<CvQualifiers> targetQualifiers;
  int deepestDifference = -1;

  while (true) {
    auto sourceCv = decompositionCvQualifiers(source);
    auto targetCv = decompositionCvQualifiers(target);

    if (!cv_is_subset_of(sourceCv, targetCv)) return false;
    if (sourceCv != targetCv)
      deepestDifference = level + static_cast<int>(targetQualifiers.size());
    targetQualifiers.push_back(targetCv);

    auto sourceUnqual = traits.remove_cv(source);
    auto targetUnqual = traits.remove_cv(target);
    if (traits.is_same(sourceUnqual, targetUnqual)) break;

    auto sourcePointer = type_cast<PointerType>(sourceUnqual);
    auto targetPointer = type_cast<PointerType>(targetUnqual);
    if (sourcePointer && targetPointer) {
      source = sourcePointer->elementType();
      target = targetPointer->elementType();
      continue;
    }

    auto sourceMember = decomposeMemberPointer(sourceUnqual);
    auto targetMember = decomposeMemberPointer(targetUnqual);
    if (sourceMember && targetMember &&
        traits.is_same(sourceMember.classType, targetMember.classType)) {
      source = sourceMember.pointeeType;
      target = targetMember.pointeeType;
      continue;
    }

    if (hasSimilarArrayBound(sourceUnqual, targetUnqual)) {
      source = traits.get_element_type(sourceUnqual);
      target = traits.get_element_type(targetUnqual);
      continue;
    }

    return false;
  }

  for (int k = level < 1 ? 1 : level; k < deepestDifference; ++k) {
    if (!cv_is_subset_of(CvQualifiers::kConst, targetQualifiers[k - level]))
      return false;
  }

  return true;
}

auto StandardConversion::isMemberPointeeConvertible(const Type* source,
                                                    const Type* target) const
    -> bool {
  if (auto sourceFunction = type_cast<FunctionType>(source)) {
    auto targetFunction = type_cast<FunctionType>(target);
    if (!targetFunction) return false;
    if (traits.is_same(sourceFunction, targetFunction)) return true;
    if (!sourceFunction->isNoexcept() || targetFunction->isNoexcept())
      return false;
    return traits.is_same(traits.remove_noexcept(sourceFunction),
                          targetFunction);
  }

  return isQualificationConversion(source, target, /*level=*/1);
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
  auto targetUnqual = traits.remove_cv(targetType);
  auto sourceUnqual = traits.remove_cv(sourceType);
  if (!traits.is_same(targetUnqual, sourceUnqual) &&
      !traits.is_base_of(targetUnqual, sourceUnqual))
    return false;
  return checkCvQualifiers(traits.get_cv_qualifiers(targetType),
                           traits.get_cv_qualifiers(sourceType));
}

auto StandardConversion::isNullPointerConstant(ExpressionAST* expr) const
    -> bool {
  if (!expr) return false;

  for (;;) {
    if (traits.is_null_pointer(expr->type)) return true;

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
    return integerLiteral->literal->integerValue() == 0;

  return false;
}

auto StandardConversion::initializerListElementType(const Type* targetType)
    -> const Type* {
  return traits.initializer_list_element_type(targetType);
}

auto StandardConversion::lvalueToRvalue(ExpressionAST*& expr) -> bool {
  if (!is_glvalue(expr)) return false;
  if (traits.is_function(expr->type)) return false;
  if (traits.is_array(expr->type)) return false;
  if (!traits.is_complete(expr->type)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
  cast->expression = expr;
  cast->type = traits.remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kPrValue;
  adjustCv(cast);
  foldConstantRead(cast);
  expr = cast;
  return true;
}

void StandardConversion::foldConstantRead(ImplicitCastExpressionAST* cast) {
  if (cast->castKind != ImplicitCastKind::kLValueToRValueConversion) return;
  if (cast->constValue) return;

  auto operand = cast->expression;
  while (auto nested = ast_cast<NestedExpressionAST>(operand))
    operand = nested->expression;

  FieldSymbol* field = nullptr;
  bool throughObject = false;
  if (auto id = ast_cast<IdExpressionAST>(operand)) {
    field = symbol_cast<FieldSymbol>(id->symbol);
  } else if (auto member = ast_cast<MemberExpressionAST>(operand)) {
    field = symbol_cast<FieldSymbol>(member->symbol);
    throughObject = true;
  }

  if (!field || !field->isStatic()) return;
  if (!traits.is_scalar(field->type())) return;

  if (!throughObject && field->definition()) return;

  auto interp = ASTInterpreter{unit_};
  if (auto value = interp.evaluate(operand)) {
    cast->constValue = unit_->arena()->make<ConstValue>(std::move(*value));
  }
}

auto StandardConversion::arrayToPointer(ExpressionAST*& expr) -> bool {
  auto unref = traits.remove_reference(expr->type);
  if (!traits.is_array(unref)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->expression = expr;
  cast->type = traits.add_pointer(traits.remove_extent(unref));
  cast->valueCategory = ValueCategory::kPrValue;

  if (type_cast<UnresolvedBoundedArrayType>(unref)) {
    cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
  } else {
    cast->castKind = ImplicitCastKind::kArrayToPointerConversion;
  }

  expr = cast;
  return true;
}

auto StandardConversion::functionToPointer(ExpressionAST*& expr) -> bool {
  auto unref = traits.remove_reference(expr->type);
  if (!traits.is_function(unref)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
  cast->expression = expr;
  cast->type = traits.add_pointer(unref);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto StandardConversion::integralPromotion(ExpressionAST*& expr,
                                           const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;
  if (!traits.is_integral(expr->type) && !traits.is_enum(expr->type))
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
  if (!traits.is_floating_point(expr->type)) return false;
  if (!destinationType) destinationType = control_->getDoubleType();
  if (!traits.is_floating_point(destinationType)) return false;
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
  if (!traits.is_integral_or_unscoped_enum(expr->type)) return false;
  if (!traits.is_integer(destinationType)) return false;

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
  if (traits.is_same(expr->type, destinationType)) return true;
  if (!traits.is_floating_point(expr->type)) return false;
  if (!traits.is_floating_point(destinationType)) return false;

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

  if (traits.is_integral_or_unscoped_enum(expr->type) &&
      traits.is_floating_point(destinationType)) {
    make();
    return true;
  }

  if (!traits.is_floating_point(expr->type)) return false;
  if (!traits.is_integer(destinationType)) return false;
  make();
  return true;
}

auto StandardConversion::ensurePrvalue(ExpressionAST*& expr) -> bool {
  if (lvalueToRvalue(expr)) return true;
  if (arrayToPointer(expr)) return true;
  if (functionToPointer(expr)) return true;
  return false;
}

auto StandardConversion::adjustedCvType(const Type* type) const -> const Type* {
  auto qualType = type_cast<QualType>(type);
  if (!qualType) return type;

  if (traits.is_class(type) || traits.is_array(type)) return type;

  return qualType->elementType();
}

void StandardConversion::adjustCv(ExpressionAST* expr) {
  if (!is_prvalue(expr)) return;
  expr->type = adjustedCvType(expr->type);
}

auto StandardConversion::temporaryMaterialization(ExpressionAST*& expr)
    -> bool {
  if (!is_prvalue(expr)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kTemporaryMaterializationConversion;
  cast->expression = expr;
  cast->type = traits.remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kXValue;
  expr = cast;
  return true;
}

auto StandardConversion::convertImplicitly(
    ExpressionAST*& expr, const Type* destinationType,
    InitializationKind initializationKind) -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  auto seq =
      computeConversionSequence(expr, destinationType, initializationKind);
  if (!seq) return false;

  applyConversionSequence(seq, expr);
  adjustCv(expr);
  return true;
}

auto StandardConversion::convertClassOperandForBuiltinOperator(
    ExpressionAST*& expr) -> bool {
  if (!expr || !expr->type) return false;

  auto classType = type_cast<ClassType>(traits.remove_cvref(expr->type));
  if (!classType) return false;

  auto classSymbol = classType->symbol();
  if (!classSymbol) return false;

  traits.requireCompleteClass(classSymbol);

  const Type* target = nullptr;
  std::vector<ClassSymbol*> pending{classSymbol->resolvedDefinition()};
  std::vector<ClassSymbol*> seen;

  while (!pending.empty()) {
    auto currentClass = pending.back();
    pending.pop_back();
    if (!currentClass) continue;
    if (std::ranges::find(seen, currentClass) != seen.end()) continue;
    seen.push_back(currentClass);

    for (auto base : currentClass->baseClasses()) {
      if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol()))
        pending.push_back(baseClass->resolvedDefinition());
    }

    for (auto convFunc : currentClass->implicitConversionFunctions()) {
      auto convFuncType = type_cast<FunctionType>(convFunc->type());
      if (!convFuncType) continue;

      auto returnType = traits.remove_cvref(convFuncType->returnType());
      if (!returnType) continue;

      if (!traits.is_arithmetic_or_unscoped_enum(returnType) &&
          !traits.is_pointer(returnType))
        continue;

      if (target && !traits.is_same(target, returnType)) return false;

      target = returnType;
    }
  }

  if (!target) return false;

  return convertImplicitly(expr, target);
}

auto StandardConversion::usualArithmeticConversion(ExpressionAST*& expr,
                                                   ExpressionAST*& other)
    -> const Type* {
  if (!traits.is_arithmetic(expr->type) && !traits.is_enum(expr->type))
    return nullptr;
  if (!traits.is_arithmetic(other->type) && !traits.is_enum(other->type))
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

  auto common = commonArithmeticType(expr->type, other->type);
  if (!common) return unmodified();

  if (!convertArithmetic(expr, common) || !convertArithmetic(other, common))
    return unmodified();

  return common;
}

auto StandardConversion::commonArithmeticType(const Type* a, const Type* b)
    -> const Type* {
  auto isArith = [&](const Type* t) {
    return traits.is_arithmetic(t) ||
           (traits.is_enum(t) && !traits.is_scoped_enum(t));
  };
  if (!isArith(a) || !isArith(b)) return nullptr;

  auto fpRank = [](const Type* t) -> int {
    switch (t->kind()) {
      case TypeKind::kLongDouble:
        return 4;
      case TypeKind::kDouble:
        return 3;
      case TypeKind::kFloat:
        return 2;
      case TypeKind::kFloat16:
        return 1;
      default:
        return -1;
    }
  };
  if (traits.is_floating_point(a) || traits.is_floating_point(b))
    return fpRank(a) >= fpRank(b) ? a : b;

  auto isBitInt = [](const Type* t) {
    return t->kind() == TypeKind::kBitInt ||
           t->kind() == TypeKind::kUnsignedBitInt;
  };

  if (isBitInt(a) || isBitInt(b)) {
    if (traits.is_same(a, b)) return a;
    auto numBitsOf = [&](const Type* t) -> int {
      if (t->kind() == TypeKind::kBitInt)
        return type_cast<BitIntType>(t)->numBits();
      if (t->kind() == TypeKind::kUnsignedBitInt)
        return type_cast<UnsignedBitIntType>(t)->numBits();
      return 0;
    };
    int bitsA = numBitsOf(a);
    int bitsB = numBitsOf(b);
    if (bitsA <= 0 || bitsB <= 0) return nullptr;
    int bits = std::max(bitsA, bitsB);
    bool anyUnsigned = traits.is_unsigned(a) || traits.is_unsigned(b);
    if (anyUnsigned) return control_->getUnsignedBitIntType(bits);
    return control_->getBitIntType(bits);
  }

  auto layout = control_->memoryLayout();
  auto sizeOf = [&](const Type* t) -> std::size_t {
    if (layout)
      if (auto s = layout->sizeOf(t)) return *s;
    return 0;
  };

  auto rank = [&](const Type* t) -> int {
    switch (t->kind()) {
      case TypeKind::kBool:
        return 1;
      case TypeKind::kSignedChar:
      case TypeKind::kUnsignedChar:
      case TypeKind::kChar:
        return 2;
      case TypeKind::kShortInt:
      case TypeKind::kUnsignedShortInt:
        return 3;
      case TypeKind::kInt:
      case TypeKind::kUnsignedInt:
        return 4;
      case TypeKind::kLongInt:
      case TypeKind::kUnsignedLongInt:
        return 5;
      case TypeKind::kLongLongInt:
      case TypeKind::kUnsignedLongLongInt:
        return 6;
      case TypeKind::kInt128:
      case TypeKind::kUnsignedInt128:
        return 7;
      default:
        return 0;
    }
  };

  auto promote = [&](const Type* t) -> const Type* {
    if (auto enumType = type_cast<EnumType>(t)) {
      t = enumType->underlyingType();
      if (!t) return control_->getIntType();
    }
    switch (t->kind()) {
      case TypeKind::kBool:
      case TypeKind::kSignedChar:
      case TypeKind::kUnsignedChar:
      case TypeKind::kChar:
      case TypeKind::kChar8:
      case TypeKind::kChar16:
      case TypeKind::kChar32:
      case TypeKind::kWideChar:
      case TypeKind::kShortInt:
      case TypeKind::kUnsignedShortInt: {
        const bool fitsInInt =
            sizeOf(t) < sizeOf(control_->getIntType()) || traits.is_signed(t);
        if (fitsInInt) return control_->getIntType();
        return control_->getUnsignedIntType();
      }
      default:
        return t;
    }
  };
  auto pa = promote(a);
  auto pb = promote(b);

  if (traits.is_same(pa, pb)) return pa;

  const bool ua = traits.is_unsigned(pa);
  const bool ub = traits.is_unsigned(pb);
  const auto ra = rank(pa);
  const auto rb = rank(pb);

  if (ua == ub) return ra >= rb ? pa : pb;

  const Type* u = ua ? pa : pb;
  const Type* s = ua ? pb : pa;
  const auto ru = ua ? ra : rb;
  const auto rs = ua ? rb : ra;

  if (ru >= rs) return u;
  if (sizeOf(s) > sizeOf(u)) return s;

  switch (s->kind()) {
    case TypeKind::kShortInt:
      return control_->getUnsignedShortIntType();
    case TypeKind::kInt:
      return control_->getUnsignedIntType();
    case TypeKind::kLongInt:
      return control_->getUnsignedLongIntType();
    case TypeKind::kLongLongInt:
      return control_->getUnsignedLongLongIntType();
    case TypeKind::kInt128:
      return control_->getUnsignedInt128Type();
    default:
      return u;
  }
}

auto StandardConversion::convertArithmetic(ExpressionAST*& expr,
                                           const Type* destinationType)
    -> bool {
  if (traits.is_same(expr->type, destinationType)) return true;

  if (traits.is_floating_point(destinationType)) {
    if (traits.is_floating_point(expr->type))
      return floatingPointConversion(expr, destinationType);
    return floatingIntegralConversion(expr, destinationType);
  }

  if (integralPromotion(expr, destinationType) &&
      traits.is_same(expr->type, destinationType))
    return true;
  (void)integralPromotion(expr);
  if (traits.is_same(expr->type, destinationType)) return true;
  return integralConversion(expr, destinationType);
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
    if (traits.is_pointer(left) && traits.is_pointer(right)) return true;
    if (traits.is_array(left) && traits.is_array(right)) return true;
    return false;
  };

  if (!bothPointerOrArray()) {
    const auto cv3 = mergeCv(cv1, cv2);

    if (traits.is_same(left, right)) return traits.add_cv(left, cv3);
    if (traits.is_base_of(left, right)) return traits.add_cv(left, cv1);
    if (traits.is_base_of(right, left)) return traits.add_cv(right, cv2);
    return nullptr;
  }

  auto leftElem = traits.get_element_type(left);
  if (traits.is_array(leftElem))
    cv1 = mergeCv(cv1, traits.get_cv_qualifiers(leftElem));

  auto rightElem = traits.get_element_type(right);
  if (traits.is_array(rightElem))
    cv2 = mergeCv(cv2, traits.get_cv_qualifiers(rightElem));

  auto elemType = getQualificationCombinedType(leftElem, rightElem,
                                               didChangeTypeOrQualifiers);
  if (!elemType) return nullptr;

  auto cv3 = mergeCv(cv1, cv2);
  if (didChangeTypeOrQualifiers) cv3 = cv3 | CvQualifiers::kConst;
  if (cv1 != cv3 || cv2 != cv3) didChangeTypeOrQualifiers = true;
  elemType = traits.add_cv(elemType, cv3);

  if (traits.is_array(left) && traits.is_array(right)) {
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
  if (traits.is_null_pointer(expr->type) && traits.is_null_pointer(other->type))
    return control_->getNullptrType();

  if (isNullPointerConstant(expr)) return other->type;
  if (isNullPointerConstant(other)) return expr->type;

  if (traits.is_pointer(expr->type) && traits.is_pointer(other->type)) {
    auto t1 = traits.get_element_type(expr->type);
    const auto cv1 = stripCv(t1);
    auto t2 = traits.get_element_type(other->type);
    const auto cv2 = stripCv(t2);

    if (traits.is_void(t1))
      return control_->getPointerType(traits.add_cv(t1, cv2));
    if (traits.is_void(t2))
      return control_->getPointerType(traits.add_cv(t2, cv1));

    if (auto type = getQualificationCombinedType(expr->type, other->type))
      return type;
  }

  return nullptr;
}

auto StandardConversion::isIntegralPromotion(const Type* sourceType,
                                             const Type* targetType) const
    -> bool {
  if (!sourceType || !targetType) return false;
  auto src = traits.remove_cv(sourceType);
  auto dst = traits.remove_cv(targetType);

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
    if (traits.is_scoped_enum(src)) return false;
    return traits.is_integral(dst);
  }

  return false;
}

auto StandardConversion::isFloatingPointPromotion(const Type* sourceType,
                                                  const Type* targetType) const
    -> bool {
  if (!sourceType || !targetType) return false;
  auto src = traits.remove_cv(sourceType);
  auto dst = traits.remove_cv(targetType);
  return src->kind() == TypeKind::kFloat && dst->kind() == TypeKind::kDouble;
}

auto StandardConversion::computeConversionSequence(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind) -> ImplicitConversionSequence {
  return computeConversionSequenceImpl(expr, targetType, initializationKind,
                                       ConversionScope::kImplicit);
}

auto StandardConversion::computeStandardConversionSequence(
    ExpressionAST* expr, const Type* targetType) -> ImplicitConversionSequence {
  return computeConversionSequenceImpl(expr, targetType,
                                       InitializationKind::kCopyInitialization,
                                       ConversionScope::kStandardOnly);
}

auto StandardConversion::computeConversionSequenceImpl(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind, ConversionScope conversionScope)
    -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  if (!expr || !targetType) return seq;

  seq.destinationType = targetType;

  const Type* currentType = expr->type;
  ValueCategory currentValCat = expr->valueCategory;

  auto addStep = [&](ImplicitCastKind kind, const Type* type) {
    seq.steps.push_back({kind, type});
  };

  if (auto ovlType =
          type_cast<OverloadSetType>(traits.remove_reference(currentType))) {
    if (auto ptrTarget = type_cast<PointerType>(targetType)) {
      if (auto targetFuncType =
              type_cast<FunctionType>(ptrTarget->elementType())) {
        if (resolveOverloadSetAgainstFunctionType(
                unit_, ovlType->symbol(), targetFuncType,
                expr->firstSourceLocation())) {
          seq.rank = ConversionRank::kExactMatch;
          addStep(ImplicitCastKind::kFunctionToPointerConversion, targetType);
          return seq;
        }
      }
    }
    return seq;
  }

  if (auto bracedInitList = ast_cast<BracedInitListAST>(expr)) {
    if (auto elemType = initializerListElementType(targetType)) {
      for (auto it = bracedInitList->expressionList; it; it = it->next) {
        if (!it->value || !it->value->type) return seq;
        auto elemSeq = computeConversionSequence(it->value, elemType);
        if (!elemSeq) return seq;
      }

      seq.rank = ConversionRank::kExactMatch;
      addStep(ImplicitCastKind::kIdentity, targetType);
      return seq;
    }

    auto listTarget = traits.remove_cv(traits.remove_reference(targetType));

    if (!traits.is_class(listTarget)) {
      auto elements = bracedInitList->expressionList;
      if (!elements || elements->next || !elements->value) return seq;
      seq = computeConversionSequence(elements->value, listTarget);
      seq.fromSingleElementList = bool(seq);
      return seq;
    }

    if (listInitializes(bracedInitList, listTarget)) {
      seq.rank = ConversionRank::kConversion;
      seq.kind = ConversionSequenceKind::kUserDefined;
      seq.aggregateInitializedClass = listTarget;
      bindResultToReference(seq, targetType);
      addStep(ImplicitCastKind::kIdentity, listTarget);
    }
    return seq;
  }

  if (traits.is_reference(targetType)) {
    if (auto rvalRef = type_cast<RvalueReferenceType>(targetType)) {
      seq.bindsToRvalueRef = true;

      if (currentValCat == ValueCategory::kLValue) {
        auto sourceRefRemoved = traits.remove_reference(currentType);
        auto targetElem = rvalRef->elementType();

        const bool bindsFunctionLvalue =
            traits.is_function(traits.remove_reference(targetElem)) &&
            isReferenceCompatible(targetElem, sourceRefRemoved);

        if (bindsFunctionLvalue) {
          auto sameUnqual = traits.is_same(traits.remove_cv(sourceRefRemoved),
                                           traits.remove_cv(targetElem));
          auto sourceCv = traits.get_cv_qualifiers(sourceRefRemoved);
          auto targetCv = traits.get_cv_qualifiers(targetElem);

          seq.bindsToReference = true;
          seq.referenceCv = targetCv;
          seq.rank = sameUnqual ? ConversionRank::kExactMatch
                                : ConversionRank::kConversion;
          addStep(!sameUnqual ? ImplicitCastKind::kDerivedToBaseConversion
                  : sourceCv != targetCv
                      ? ImplicitCastKind::kQualificationConversion
                      : ImplicitCastKind::kIdentity,
                  targetElem);
          return seq;
        }

        auto sourceClass = traits.remove_cv(sourceRefRemoved);
        auto targetClass = traits.remove_cv(targetElem);
        if (!traits.is_class(sourceClass)) return seq;
        if (traits.is_same(sourceClass, targetClass)) return seq;
        if (traits.is_base_of(targetClass, sourceClass)) return seq;
      }
    }

    if (auto lvalRef = type_cast<LvalueReferenceType>(targetType)) {
      auto inner = lvalRef->elementType();
      bool isConst = false;
      if (auto qual = type_cast<QualType>(inner)) isConst = qual->isConst();

      auto sourceRefRemoved = traits.remove_reference(currentType);

      const bool bindsDirectly = currentValCat == ValueCategory::kLValue &&
                                 isReferenceCompatible(inner, sourceRefRemoved);

      if (!isConst && !bindsDirectly) {
        auto sourceClass = traits.remove_cv(sourceRefRemoved);
        auto targetClass = traits.remove_cv(inner);
        if (!traits.is_class(sourceClass)) return seq;
        if (traits.is_same(sourceClass, targetClass)) return seq;
        if (traits.is_base_of(targetClass, sourceClass)) return seq;
      }

      if (bindsDirectly) {
        auto sameUnqual = traits.is_same(traits.remove_cv(sourceRefRemoved),
                                         traits.remove_cv(inner));
        auto sourceCv = traits.get_cv_qualifiers(sourceRefRemoved);
        auto targetCv = traits.get_cv_qualifiers(inner);

        seq.bindsToReference = true;
        seq.referenceCv = targetCv;
        seq.rank = sameUnqual ? ConversionRank::kExactMatch
                              : ConversionRank::kConversion;
        addStep(!sameUnqual ? ImplicitCastKind::kDerivedToBaseConversion
                : sourceCv != targetCv
                    ? ImplicitCastKind::kQualificationConversion
                    : ImplicitCastKind::kIdentity,
                inner);
        return seq;
      }
    }
  }

  if (traits.is_array(traits.remove_reference(currentType))) {
    auto unref = traits.remove_reference(currentType);
    currentType = traits.add_pointer(traits.remove_extent(unref));
    currentValCat = ValueCategory::kPrValue;
    if (type_cast<UnresolvedBoundedArrayType>(unref)) {
      addStep(ImplicitCastKind::kLValueToRValueConversion, currentType);
    } else {
      addStep(ImplicitCastKind::kArrayToPointerConversion, currentType);
    }
  } else if (traits.is_function(traits.remove_reference(currentType))) {
    auto unref = traits.remove_reference(currentType);
    currentType = traits.add_pointer(unref);
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitCastKind::kFunctionToPointerConversion, currentType);
  } else if (currentValCat != ValueCategory::kPrValue &&
             !traits.is_reference(targetType)) {
    currentType = adjustedCvType(traits.remove_reference(currentType));
    if (isC_ || !traits.is_class(currentType)) {
      currentValCat = ValueCategory::kPrValue;
      addStep(ImplicitCastKind::kLValueToRValueConversion, currentType);
    }
  }

  auto comparisonTargetType = traits.remove_reference(targetType);

  auto unqualFrom = traits.remove_cv(currentType);
  auto unqualTo = traits.remove_cv(comparisonTargetType);

  if (traits.is_same(unqualFrom, unqualTo)) {
    if (requiresCopyConstruction(expr, targetType) &&
        !selectCopyConstructor(expr, targetType)) {
      return seq;
    }

    seq.rank = ConversionRank::kExactMatch;
    addStep(ImplicitCastKind::kIdentity, comparisonTargetType);
    return seq;
  }

  if (!isC_ && traits.is_class(unqualFrom) && traits.is_class(unqualTo) &&
      traits.is_base_of(unqualTo, unqualFrom)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kDerivedToBaseConversion, comparisonTargetType);
    return seq;
  }

  if (traits.is_null_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerConversion, comparisonTargetType);
    return seq;
  }

  if (traits.is_integral(unqualFrom) && traits.is_pointer(unqualTo) &&
      isNullPointerConstant(expr)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerConversion, comparisonTargetType);
    return seq;
  }

  if (traits.is_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    auto fromPtr = type_cast<PointerType>(traits.remove_cv(unqualFrom));
    auto toPtr = type_cast<PointerType>(traits.remove_cv(unqualTo));

    if (fromPtr && toPtr) {
      auto fromPointee = fromPtr->elementType();
      auto toPointee = toPtr->elementType();

      auto fromCv = traits.get_cv_qualifiers(fromPointee);
      auto toCv = traits.get_cv_qualifiers(toPointee);

      if (cv_is_subset_of(fromCv, toCv)) {
        auto fromUnqual = traits.remove_cv(fromPointee);
        auto toUnqual = traits.remove_cv(toPointee);

        if (isQualificationConversion(fromPointee, toPointee, /*level=*/1)) {
          seq.rank = ConversionRank::kExactMatch;
          seq.hasQualificationConversion = true;
          seq.pointeeUnqual = toUnqual;
          seq.pointeeCv = toCv;
          addStep(ImplicitCastKind::kQualificationConversion,
                  comparisonTargetType);
          return seq;
        }

        if (traits.is_void(toUnqual)) {
          seq.rank = ConversionRank::kConversion;
          seq.pointeeUnqual = toUnqual;
          seq.pointeeCv = toCv;
          addStep(ImplicitCastKind::kPointerConversion, comparisonTargetType);
          return seq;
        }

        if (traits.is_class(fromUnqual) && traits.is_class(toUnqual)) {
          if (traits.is_base_of(toUnqual, fromUnqual)) {
            seq.rank = ConversionRank::kConversion;
            seq.pointeeUnqual = toUnqual;
            seq.pointeeCv = toCv;
            addStep(ImplicitCastKind::kDerivedToBaseConversion,
                    comparisonTargetType);
            return seq;
          }
        }

        if (isC_ && traits.is_void(fromUnqual)) {
          seq.rank = ConversionRank::kConversion;
          addStep(ImplicitCastKind::kPointerConversion, comparisonTargetType);
          return seq;
        }

        if (isC_) {
          auto areVlaCompatible = [&](auto& self, const Type* a,
                                      const Type* b) -> bool {
            a = traits.remove_cv(a);
            b = traits.remove_cv(b);
            if (traits.is_same(a, b)) return true;
            auto va = type_cast<UnresolvedBoundedArrayType>(a);
            auto vb = type_cast<UnresolvedBoundedArrayType>(b);
            if (va && vb)
              return self(self, va->elementType(), vb->elementType());
            return false;
          };
          if (type_cast<UnresolvedBoundedArrayType>(fromUnqual) &&
              type_cast<UnresolvedBoundedArrayType>(toUnqual) &&
              areVlaCompatible(areVlaCompatible, fromUnqual, toUnqual)) {
            seq.rank = ConversionRank::kConversion;
            addStep(ImplicitCastKind::kPointerConversion, comparisonTargetType);
            return seq;
          }
        }
      }
    }
  }

  if (isIntegralPromotion(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kPromotion;
    addStep(ImplicitCastKind::kIntegralPromotion, comparisonTargetType);
    return seq;
  }

  if (isFloatingPointPromotion(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kPromotion;
    addStep(ImplicitCastKind::kFloatingPointPromotion, comparisonTargetType);
    return seq;
  }

  if ((traits.is_arithmetic(unqualFrom) ||
       (traits.is_enum(unqualFrom) && !traits.is_scoped_enum(unqualFrom))) &&
      traits.is_arithmetic(unqualTo)) {
    seq.rank = ConversionRank::kConversion;

    if (traits.is_integral_or_unscoped_enum(unqualFrom) &&
        traits.is_integral(unqualTo)) {
      addStep(ImplicitCastKind::kIntegralConversion, comparisonTargetType);
      return seq;
    }

    if (traits.is_floating_point(unqualFrom) &&
        traits.is_floating_point(unqualTo)) {
      addStep(ImplicitCastKind::kFloatingPointConversion, comparisonTargetType);
      return seq;
    }

    addStep(ImplicitCastKind::kFloatingIntegralConversion,
            comparisonTargetType);
    return seq;
  }

  {
    auto source = decomposeMemberPointer(unqualFrom);
    auto target = decomposeMemberPointer(unqualTo);

    if (source && target &&
        isMemberPointeeConvertible(source.pointeeType, target.pointeeType)) {
      if (traits.is_same(source.classType, target.classType)) {
        seq.rank = ConversionRank::kExactMatch;
        addStep(ImplicitCastKind::kQualificationConversion,
                comparisonTargetType);
        return seq;
      }

      if (hasUniqueNonVirtualBase(target.classType, source.classType)) {
        seq.rank = ConversionRank::kConversion;
        addStep(ImplicitCastKind::kPointerToMemberConversion,
                comparisonTargetType);
        return seq;
      }
    }
  }

  if (traits.is_member_pointer(unqualTo) && isNullPointerConstant(expr)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kPointerToMemberConversion, comparisonTargetType);
    return seq;
  }

  if (traits.is_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    auto srcPtr = type_cast<PointerType>(unqualFrom);
    auto dstPtr = type_cast<PointerType>(unqualTo);
    if (srcPtr && dstPtr) {
      auto srcFunc = type_cast<FunctionType>(srcPtr->elementType());
      auto dstFunc = type_cast<FunctionType>(dstPtr->elementType());
      if (srcFunc && dstFunc && srcFunc->isNoexcept() &&
          !dstFunc->isNoexcept() &&
          traits.is_same(traits.remove_noexcept(srcFunc), dstFunc)) {
        seq.rank = ConversionRank::kExactMatch;
        addStep(ImplicitCastKind::kFunctionPointerConversion,
                comparisonTargetType);
        return seq;
      }
    }
  }

  if (traits.is_same(unqualTo, control_->getBoolType())) {
    if (traits.is_arithmetic_or_unscoped_enum(unqualFrom) ||
        traits.is_pointer(unqualFrom) || traits.is_member_pointer(unqualFrom)) {
      seq.rank = ConversionRank::kConversion;
      addStep(ImplicitCastKind::kBooleanConversion, comparisonTargetType);
      return seq;
    }
  }

  if (isC_ && traits.is_integral_or_unscoped_enum(unqualFrom) &&
      traits.is_enum(unqualTo) && !traits.is_scoped_enum(unqualTo)) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitCastKind::kIntegralConversion, comparisonTargetType);
    return seq;
  }

  if (conversionScope == ConversionScope::kStandardOnly) return seq;

  auto candidateConversionFunctions =
      [&](ClassSymbol* classSymbol) -> std::vector<FunctionSymbol*> {
    if (isDirectInitialization(initializationKind))
      return classSymbol->conversionFunctions();
    return classSymbol->implicitConversionFunctions();
  };

  const bool bindsNonConstLvalueReference = [&] {
    auto lvalueRef = type_cast<LvalueReferenceType>(targetType);
    if (!lvalueRef) return false;
    auto qual = type_cast<QualType>(lvalueRef->elementType());
    return !qual || !qual->isConst();
  }();

  auto referenceBindsConversionResult = [&](const Type* reference,
                                            const Type* resultType) {
    const bool resultIsLvalue =
        type_cast<LvalueReferenceType>(resultType) != nullptr;

    if (auto lvalueRef = type_cast<LvalueReferenceType>(reference)) {
      auto inner = lvalueRef->elementType();
      auto qual = type_cast<QualType>(inner);
      if (qual && qual->isConst()) return true;
      if (!resultIsLvalue) return false;
      return isReferenceCompatible(inner, traits.remove_reference(resultType));
    }

    if (type_cast<RvalueReferenceType>(reference)) return !resultIsLvalue;

    return true;
  };

  auto conversionResultType = [&](FunctionSymbol* func) -> const Type* {
    auto funcType = type_cast<FunctionType>(func->type());
    if (!funcType) return comparisonTargetType;
    if (func->isConstructor()) return comparisonTargetType;
    return funcType->returnType();
  };

  auto makeUserDefinedSeq =
      [&](FunctionSymbol* func,
          ConversionRank s2Rank) -> ImplicitConversionSequence {
    ImplicitConversionSequence uds;
    uds.kind = ConversionSequenceKind::kUserDefined;
    uds.rank = ConversionRank::kConversion;
    uds.userDefinedConversionFunction = func;
    uds.secondStandardConversionRank = s2Rank;
    bindResultToReference(uds, targetType);

    auto resultType = conversionResultType(func);
    uds.steps.push_back({ImplicitCastKind::kUserDefinedConversion, resultType});

    if (!traits.is_same(traits.remove_cv(resultType), comparisonTargetType)) {
      uds.secondStandardConversionTarget = comparisonTargetType;
    }

    return uds;
  };

  ImplicitConversionSequence bestUserDefined;
  FunctionSymbol* bestConversionFunction = nullptr;

  auto standardConversionRank = [&](ExpressionAST* source,
                                    const Type* to) -> ConversionRank {
    return computeStandardConversionSequence(source, to).rank;
  };

  auto updateBest = [&](FunctionSymbol* func, ConversionRank s2Rank) {
    if (bestUserDefined) {
      if (s2Rank < bestUserDefined.secondStandardConversionRank) return;
      if (s2Rank == bestUserDefined.secondStandardConversionRank) {
        if (func->isSpecialization() !=
            bestConversionFunction->isSpecialization()) {
          if (func->isSpecialization()) return;
        } else if (!func->isSpecialization()) {
          return;
        } else {
          auto order = compareFunctionTemplateSpecializations(
              unit_, func, bestConversionFunction);
          if (order < 0) return;
          if (order == 0) return;
        }
      }
    }
    bestUserDefined = makeUserDefinedSeq(func, s2Rank);
    bestConversionFunction = func;
  };

  if (auto destClassType = bindsNonConstLvalueReference
                               ? nullptr
                               : type_cast<ClassType>(unqualTo)) {
    if (auto destClass = destClassType->symbol()) {
      traits.requireCompleteClass(destClass);

      for (auto ctor : destClass->convertingConstructors()) {
        if (isExcludedInheritedConstructor(traits, ctor, destClass,
                                           /*argCount=*/1))
          continue;

        if (ctor->templateDeclaration() && !ctor->isSpecialization()) {
          auto args = make_list_node<ExpressionAST>(unit_->arena(), expr);

          TemplateArgumentDeduction deduction(unit_);
          auto deducedArgs = deduction.deduce(
              ctor, args, /*explicitTemplateArguments=*/nullptr);
          if (!deducedArgs.has_value()) continue;

          auto instCtor = ASTRewriter::instantiateForArgs(
              unit_, *deducedArgs, ctor, expr->firstSourceLocation(),
              /*argsComplete=*/false);
          if (!instCtor) continue;

          ctor = instCtor;
        }

        auto funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (!isCallableWithOneArgument(ctor)) continue;

        auto rank = standardConversionRank(expr, params[0]);
        if (rank != ConversionRank::kNone) updateBest(ctor, rank);
      }
    }
  }

  if (auto srcClassType = type_cast<ClassType>(unqualFrom)) {
    if (auto srcClass = srcClassType->symbol()) {
      std::vector<ClassSymbol*> pending{srcClass->resolvedDefinition()};
      std::vector<ClassSymbol*> seen;
      while (!pending.empty()) {
        auto currentClass = pending.back();
        pending.pop_back();
        if (std::ranges::find(seen, currentClass) != seen.end()) continue;
        seen.push_back(currentClass);

        for (auto base : currentClass->baseClasses()) {
          if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol()))
            pending.push_back(baseClass->resolvedDefinition());
        }

        for (auto convFunc : candidateConversionFunctions(currentClass)) {
          if (convFunc->templateDeclaration() &&
              !convFunc->isSpecialization()) {
            convFunc =
                instantiateConversionFunctionTemplate(convFunc, unqualTo, expr);
            if (!convFunc) continue;
          }

          auto convFuncType = type_cast<FunctionType>(convFunc->type());
          if (!convFuncType) continue;

          auto returnType = convFuncType->returnType();
          if (!returnType) continue;

          if (!referenceBindsConversionResult(targetType, returnType)) continue;

          auto retUnqual =
              traits.remove_cv(traits.remove_reference(returnType));

          if (convFunc->isExplicit()) {
            if (isQualificationConversion(retUnqual, unqualTo, /*level=*/0))
              updateBest(convFunc, ConversionRank::kExactMatch);
            continue;
          }

          auto result = IdExpressionAST::create(arena_);
          result->type = retUnqual;
          result->valueCategory = ValueCategory::kPrValue;
          if (type_cast<LvalueReferenceType>(returnType))
            result->valueCategory = ValueCategory::kLValue;
          else if (type_cast<RvalueReferenceType>(returnType))
            result->valueCategory = ValueCategory::kXValue;
          auto rank = standardConversionRank(result, unqualTo);
          if (rank != ConversionRank::kNone) updateBest(convFunc, rank);
        }
      }
    }
  }

  if (bestUserDefined) return bestUserDefined;

  return seq;
}

void StandardConversion::applyConversionSequence(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  if (sequence.rank == ConversionRank::kNone) return;

  if (sequence.fromSingleElementList) {
    if (auto braced = ast_cast<BracedInitListAST>(expr);
        braced && braced->expressionList && !braced->expressionList->next) {
      expr = braced->expressionList->value;
    }
  }

  for (const auto& step : sequence.steps) {
    if (step.kind == ImplicitCastKind::kIdentity) {
      if (auto braced = ast_cast<BracedInitListAST>(expr);
          braced && !braced->type && step.type && traits.is_class(step.type)) {
        braced->type = step.type;
        braced->valueCategory = ValueCategory::kPrValue;
      }
      continue;
    }
    wrapWithImplicitCast(step.kind, step.type, expr);
    if (step.kind == ImplicitCastKind::kUserDefinedConversion) {
      if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
        recordUserDefinedConversion(cast,
                                    sequence.userDefinedConversionFunction);
      }
    }
  }

  if (sequence.secondStandardConversionTarget) {
    auto second = computeConversionSequence(
        expr, sequence.secondStandardConversionTarget);

    if (second.kind != ConversionSequenceKind::kStandard) return;

    applyConversionSequence(second, expr);
    return;
  }

  if (traits.is_reference(sequence.destinationType)) {
    if (is_prvalue(expr) && !traits.is_class(expr->type) &&
        !traits.is_array(expr->type)) {
      (void)temporaryMaterialization(expr);
    }
    return;
  }

  if (sequence.bindsToReference || sequence.bindsToRvalueRef) return;

  (void)copyInitializeClassPrvalue(expr, sequence.destinationType);
}

auto StandardConversion::requiresCopyConstruction(
    ExpressionAST* expr, const Type* destinationType) const -> bool {
  if (isC_ || !expr || !expr->type || !destinationType) return false;
  if (!is_glvalue(expr)) return false;
  if (traits.is_reference(destinationType)) return false;

  auto classType = type_cast<ClassType>(traits.remove_cv(destinationType));
  if (!classType || !classType->symbol()) return false;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  if (!classSymbol->isComplete()) return false;

  if (!classSymbol->copyConstructor() && !classSymbol->moveConstructor())
    return false;

  auto sourceType = traits.remove_cvref(expr->type);
  return traits.is_same(sourceType, classType);
}

auto StandardConversion::selectCopyConstructor(ExpressionAST* expr,
                                               const Type* destinationType)
    -> FunctionSymbol* {
  if (!requiresCopyConstruction(expr, destinationType)) return nullptr;

  auto classType = type_cast<ClassType>(traits.remove_cv(destinationType));
  auto classSymbol = classType->symbol()->resolvedDefinition();

  if (!control_->beginCopyConstructorSelection(classSymbol)) return nullptr;

  OverloadResolution resolution(unit_);
  auto resolved = resolution.resolveConstructor(classSymbol, {expr});

  control_->endCopyConstructorSelection(classSymbol);

  if (!resolved.best || resolved.ambiguous) return nullptr;

  return resolved.best->symbol;
}

auto StandardConversion::recordClassCopyConstructor(
    ImplicitCastExpressionAST* cast) -> bool {
  auto constructor = selectCopyConstructor(cast->expression, cast->type);
  if (!constructor) return false;

  recordUserDefinedConversion(cast, constructor);
  return true;
}

auto StandardConversion::copyInitializeClassPrvalue(ExpressionAST*& expr,
                                                    const Type* destinationType)
    -> bool {
  auto constructor = selectCopyConstructor(expr, destinationType);
  if (!constructor) return false;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kUserDefinedConversion;
  cast->expression = expr;
  cast->type = traits.remove_cv(destinationType);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  recordUserDefinedConversion(cast, constructor);
  return true;
}

void StandardConversion::materializeConstructorArguments(
    ImplicitCastExpressionAST* cast, FunctionSymbol* constructor) {
  auto params = parameters(constructor);
  if (params.empty()) return;
  if (ast_cast<ParenInitializerAST>(cast->expression)) return;

  auto arguments = make_list_node<ExpressionAST>(arena_, cast->expression);

  auto sequence =
      computeConversionSequence(arguments->value, params[0]->type());
  applyConversionSequence(sequence, arguments->value);

  appendDefaultArguments(constructor, &arguments);

  auto paren = ParenInitializerAST::create(
      arena_, cast->firstSourceLocation(), arguments,
      cast->firstSourceLocation(), ValueCategory::kPrValue, cast->type);
  cast->expression = paren;
}

auto StandardConversion::listInitializes(BracedInitListAST* bracedInitList,
                                         const Type* targetType) -> bool {
  auto classType = type_cast<ClassType>(traits.remove_cv(targetType));
  if (!classType || !classType->symbol()) return false;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  if (!classSymbol->isComplete()) return false;

  std::vector<ExpressionAST*> elements;
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    if (!it->value) return false;
    if (!it->value->type && !ast_cast<BracedInitListAST>(it->value))
      return false;
    elements.push_back(it->value);
  }

  if (!traits.is_aggregate(classType)) {
    OverloadResolution resolution(unit_);
    auto result = resolution.resolveConstructor(classSymbol, elements);
    return result.best && !result.ambiguous;
  }

  std::vector<const Type*> slotTypes;
  for (auto base : classSymbol->baseClasses()) {
    if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol()))
      slotTypes.push_back(baseClass->type());
  }
  for (auto member : views::members(classSymbol)) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (field && !field->isStatic()) slotTypes.push_back(field->type());
  }

  if (elements.size() > slotTypes.size()) return false;

  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (traits.is_array(slotTypes[i])) continue;
    if (!computeConversionSequence(elements[i], slotTypes[i])) return false;
  }

  return true;
}

auto StandardConversion::parameters(FunctionSymbol* function)
    -> std::vector<ParameterSymbol*> {
  std::vector<ParameterSymbol*> result;
  if (!function) return result;
  auto scope = function->functionParameters();
  if (!scope) return result;
  for (auto member : scope->members()) {
    if (auto param = symbol_cast<ParameterSymbol>(member))
      result.push_back(param);
  }
  return result;
}

auto StandardConversion::isCallableWithOneArgument(FunctionSymbol* ctor)
    -> bool {
  auto funcType = type_cast<FunctionType>(ctor->type());
  if (!funcType || funcType->parameterTypes().empty()) return false;
  if (funcType->parameterTypes().size() == 1) return true;

  auto params = parameters(ctor);
  if (params.size() != funcType->parameterTypes().size()) return false;

  for (std::size_t i = 1; i < params.size(); ++i)
    if (!params[i]->defaultArgument()) return false;

  return true;
}

void StandardConversion::appendDefaultArguments(FunctionSymbol* function,
                                                List<ExpressionAST*>** list) {
  auto params = parameters(function);
  if (params.empty() || !list) return;

  auto tail = list;
  std::size_t argCount = 0;
  while (*tail) {
    tail = &(*tail)->next;
    ++argCount;
  }

  for (auto i = argCount; i < params.size(); ++i) {
    auto defaultArgument = params[i]->defaultArgument();
    if (!defaultArgument) break;
    *tail =
        make_list_node<ExpressionAST>(arena_, defaultArgument->clone(arena_));
    auto sequence =
        computeConversionSequence((*tail)->value, params[i]->type());
    applyConversionSequence(sequence, (*tail)->value);
    tail = &(*tail)->next;
  }
}

void StandardConversion::recordUserDefinedConversion(
    ImplicitCastExpressionAST* cast, FunctionSymbol* function) {
  if (!function) return;

  cast->conversionFunction = function;

  if (function->isConstructor()) {
    materializeConstructorArguments(cast, function);
    return;
  }

  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  if (!classSymbol) return;

  auto& objectExpression = cast->expression;
  if (objectExpression && objectExpression->type &&
      is_glvalue(objectExpression)) {
    auto sourceType =
        traits.remove_cv(traits.remove_reference(objectExpression->type));
    if (!traits.is_same(sourceType, classSymbol->type()) &&
        traits.is_base_of(classSymbol->type(), sourceType)) {
      wrapWithImplicitCast(ImplicitCastKind::kDerivedToBaseConversion,
                           classSymbol->type(), objectExpression);
    }
  }

  cast->isVirtualDispatch = isVirtualMemberDispatch(function, objectExpression);
}

auto StandardConversion::isKnownCompleteObject(ExpressionAST* expression)
    -> bool {
  while (expression) {
    if (auto nested = ast_cast<NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(expression);
        cast && cast->castKind == ImplicitCastKind::kDerivedToBaseConversion) {
      expression = cast->expression;
      continue;
    }
    break;
  }

  Symbol* symbol = nullptr;
  if (auto id = ast_cast<IdExpressionAST>(expression)) {
    symbol = id->symbol;
  } else if (auto member = ast_cast<MemberExpressionAST>(expression)) {
    symbol = member->symbol;
  }

  if (!symbol) return false;

  if (!symbol_cast<VariableSymbol>(symbol) &&
      !symbol_cast<ParameterSymbol>(symbol) &&
      !symbol_cast<FieldSymbol>(symbol))
    return false;

  return !traits.is_reference(symbol->type());
}

auto StandardConversion::isVirtualMemberDispatch(
    FunctionSymbol* function, ExpressionAST* objectExpression) -> bool {
  if (!function || !function->isVirtual()) return false;
  if (!function->isImplicitObjectMemberFunction()) return false;
  if (!objectExpression || !is_glvalue(objectExpression)) return false;
  return !isKnownCompleteObject(objectExpression);
}

void StandardConversion::wrapWithImplicitCast(ImplicitCastKind castKind,
                                              const Type* type,
                                              ExpressionAST*& expr) {
  if (castKind == ImplicitCastKind::kFunctionToPointerConversion) {
    if (auto ovlType =
            type_cast<OverloadSetType>(traits.remove_reference(expr->type))) {
      if (auto ptrTarget = type_cast<PointerType>(type)) {
        if (auto targetFuncType =
                type_cast<FunctionType>(ptrTarget->elementType())) {
          auto stripped = expr;
          while (auto nested = ast_cast<NestedExpressionAST>(stripped))
            stripped = nested->expression;

          if (auto resolved = resolveOverloadSetAgainstFunctionType(
                  unit_, ovlType->symbol(), targetFuncType,
                  expr->firstSourceLocation())) {
            if (auto idExpr = ast_cast<IdExpressionAST>(stripped)) {
              idExpr->symbol = resolved;
              idExpr->type = resolved->type();
            } else if (auto memberExpr =
                           ast_cast<MemberExpressionAST>(stripped)) {
              memberExpr->symbol = resolved;
              memberExpr->type = resolved->type();
            }
          }
        }
      }
    }
  }

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = castKind;
  cast->expression = expr;
  cast->type = type;
  cast->valueCategory = castKind == ImplicitCastKind::kQualificationConversion
                            ? expr->valueCategory
                            : ValueCategory::kPrValue;

  if (castKind == ImplicitCastKind::kDerivedToBaseConversion &&
      !traits.is_pointer(type) && is_glvalue(expr)) {
    cast->valueCategory = expr->valueCategory;
  }

  foldConstantRead(cast);

  expr = cast;
}

auto StandardConversion::isNarrowingConversion(const Type* from, const Type* to)
    -> bool {
  if (isC_) return false;

  from = traits.remove_cv(from);
  to = traits.remove_cv(to);

  if (traits.is_same(from, to)) return false;

  if (traits.is_floating_point(from) && traits.is_integral(to)) return true;

  if (traits.is_floating_point(from) && traits.is_floating_point(to)) {
    auto fromSize = control_->memoryLayout()->sizeOf(from);
    auto toSize = control_->memoryLayout()->sizeOf(to);
    if (fromSize && toSize && *fromSize > *toSize) return true;
  }

  if (traits.is_integral_or_unscoped_enum(from) && traits.is_floating_point(to))
    return true;

  if (traits.is_integral_or_unscoped_enum(from) && traits.is_integral(to)) {
    auto fromSize = control_->memoryLayout()->sizeOf(from);
    auto toSize = control_->memoryLayout()->sizeOf(to);
    if (fromSize && toSize) {
      if (*fromSize > *toSize) return true;
      if (*fromSize == *toSize &&
          traits.is_signed(from) != traits.is_signed(to))
        return true;
    }
  }

  return false;
}
}  // namespace cxx
