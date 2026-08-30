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

#include <algorithm>
#include <optional>

namespace cxx {
namespace {
using ReferenceBinding = ImplicitConversionSequence::ReferenceBinding;

[[nodiscard]] auto referenceBindingKind(ValueCategory sourceValueCategory)
    -> ReferenceBinding::Kind {
  switch (sourceValueCategory) {
    case ValueCategory::kLValue:
      return ReferenceBinding::Kind::kDirectToLvalue;
    case ValueCategory::kPrValue:
      return ReferenceBinding::Kind::kToTemporary;
    default:
      return ReferenceBinding::Kind::kDirectToXvalue;
  }
}

void bindResultToReference(ImplicitConversionSequence& seq,
                           const Type* targetType,
                           ValueCategory sourceValueCategory) {
  const Type* referencedType = nullptr;
  if (auto rvalueRef = type_cast<RvalueReferenceType>(targetType)) {
    seq.binding.isRvalueRef = true;
    referencedType = rvalueRef->elementType();
  } else if (auto lvalueRef = type_cast<LvalueReferenceType>(targetType)) {
    referencedType = lvalueRef->elementType();
  }

  if (!referencedType) return;

  seq.binding.kind = referenceBindingKind(sourceValueCategory);
  seq.binding.referencedType = referencedType;
  seq.binding.cv = cv_qualifiers(referencedType);
  seq.binding.referencesFunctionType =
      unqualified_cast<FunctionType>(referencedType) != nullptr;
}

[[nodiscard]] auto resolveOverloadSetAgainstFunctionType(
    TranslationUnit* unit, OverloadSetSymbol* ovl,
    const FunctionType* targetFunctionType, SourceLocation loc)
    -> FunctionSymbol* {
  FunctionSymbol* match = nullptr;
  List<TemplateArgumentAST*>* matchDeducedArguments = nullptr;

  for (auto func : ovl->functions()) {
    if (func->canonical() != func) continue;

    FunctionSymbol* candidate = func;
    List<TemplateArgumentAST*>* candidateDeducedArguments = nullptr;

    if (func->templateDeclaration() && !func->isSpecialization()) {
      TemplateArgumentDeduction deduction(unit);
      auto deducedArgs =
          deduction.deduceFromTargetType(func, targetFunctionType);
      if (!deducedArgs.has_value()) continue;

      candidate = ASTRewriter::instantiateOverloadCandidate(
          unit, *deducedArgs, func, loc, /*argsComplete=*/true);
      if (!candidate) continue;
      candidateDeducedArguments = *deducedArgs;
    } else if (func->isSpecialization()) {
      continue;
    }

    auto candidateType = type_cast<FunctionType>(candidate->type());
    if (!candidateType) continue;
    if (!unit->typeTraits().is_same(candidateType, targetFunctionType))
      continue;

    if (match && match != candidate) return nullptr;
    match = candidate;
    matchDeducedArguments = candidateDeducedArguments;
  }

  ASTRewriter::instantiateSelectedSpecializationDefinition(
      unit, match, matchDeducedArguments);

  return match;
}

[[nodiscard]] auto stripNestedExpressions(ExpressionAST* expr)
    -> ExpressionAST* {
  while (auto nested = ast_cast<NestedExpressionAST>(expr))
    expr = nested->expression;
  return expr;
}

struct MemberPointerParts {
  const Type* classType = nullptr;
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

auto StandardConversion::instantiateConversionFunctionTemplate(
    FunctionSymbol* convFunc, const Type* targetType, ExpressionAST* expr)
    -> FunctionSymbol* {
  TemplateArgumentDeduction deduction(unit_);
  auto deducedArgs = deduction.deduceFromConversionTarget(convFunc, targetType);
  if (!deducedArgs.has_value()) return nullptr;

  return ASTRewriter::instantiateOverloadCandidate(
      unit_, *deducedArgs, convFunc, expr->firstSourceLocation(),
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

  return traits.is_qualification_convertible(control_->getPointerType(source),
                                             control_->getPointerType(target));
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

    if (ast_cast<EqualInitializerAST>(expr) ||
        ast_cast<ParenInitializerAST>(expr)) {
      expr = Initializer{expr}.singleExpression();
      if (!expr) return false;
      continue;
    }

    break;
  }

  if (auto integerLiteral = ast_cast<IntLiteralExpressionAST>(expr))
    return integerLiteral->literal->integerValue() == 0;

  return false;
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
  expr = cast;
  foldConstantRead(expr);
  return true;
}

void StandardConversion::foldConstantRead(ExpressionAST*& expression) {
  auto cast = ast_cast<ImplicitCastExpressionAST>(expression);
  if (!cast) return;
  if (cast->castKind != ImplicitCastKind::kLValueToRValueConversion) return;

  auto operand = cast->expression;
  while (auto nested = ast_cast<NestedExpressionAST>(operand))
    operand = nested->expression;

  FieldSymbol* field = nullptr;
  MemberExpressionAST* member = nullptr;
  if (auto id = ast_cast<IdExpressionAST>(operand)) {
    field = symbol_cast<FieldSymbol>(id->symbol);
  } else {
    member = ast_cast<MemberExpressionAST>(operand);
    if (member) field = symbol_cast<FieldSymbol>(member->symbol);
  }

  if (!field) return;
  if (!field->isStatic()) return;
  if (!traits.is_scalar(field->type())) return;

  if (!member && field->definition()) return;

  auto interp = ASTInterpreter{unit_};
  if (auto value = interp.evaluate(operand)) {
    auto constExpression = ConstExpressionAST::create(arena_);
    constExpression->expression = cast;
    constExpression->constValue =
        unit_->arena()->make<ConstValue>(std::move(*value));
    constExpression->type = cast->type;
    constExpression->valueCategory = cast->valueCategory;
    expression = constExpression;

    if (!member) return;

    auto comma = BinaryExpressionAST::create(arena_);
    comma->leftExpression = member->baseExpression;
    comma->rightExpression = constExpression;
    comma->op = TokenKind::T_COMMA;
    comma->type = constExpression->type;
    comma->valueCategory = constExpression->valueCategory;
    expression = comma;
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

  if (destinationType) {
    if (!traits.is_integral_promotion(expr->type, destinationType))
      return false;
    make(destinationType);
    return true;
  }

  auto promotedType = traits.promoted_integer_type(expr->type);
  if (traits.is_same(promotedType, expr->type)) return false;

  make(promotedType);
  return true;
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

void StandardConversion::adjustCv(ExpressionAST* expr) {
  if (!is_prvalue(expr)) return;
  expr->type = traits.adjusted_cv_type(expr->type);
}

void StandardConversion::prepareOperand(ExpressionAST*& expr) {
  (void)ensurePrvalue(expr);
  adjustCv(expr);
}

void StandardConversion::promoteOperand(ExpressionAST*& expr) {
  prepareOperand(expr);
  if (integralPromotion(expr)) return;
  (void)floatingPointPromotion(expr);
}

void StandardConversion::decayOperand(ExpressionAST*& expr) {
  if (arrayToPointer(expr)) return;
  (void)functionToPointer(expr);
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
  if (seq.form == ConversionSequenceForm::kAmbiguous) return false;
  if (seq.requiresCopyConstruction && !seq.copyConstructor) return false;

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
  prepareOperand(expr);
  prepareOperand(other);

  auto common = commonArithmeticType(expr->type, other->type);
  if (!common) return nullptr;

  if (!convertArithmetic(expr, common) || !convertArithmetic(other, common))
    return nullptr;

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

  auto pa = traits.promoted_integer_type(a);
  auto pb = traits.promoted_integer_type(b);

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

auto StandardConversion::compositeVoidPointerType(const Type* left,
                                                  const Type* right)
    -> const Type* {
  auto leftPointer = type_cast<PointerType>(left);
  auto rightPointer = type_cast<PointerType>(right);
  if (!leftPointer || !rightPointer) return nullptr;

  auto leftElement = leftPointer->elementType();
  auto rightElement = rightPointer->elementType();

  auto leftIsVoid = traits.is_void(leftElement);
  auto rightIsVoid = traits.is_void(rightElement);
  if (!leftIsVoid && !rightIsVoid) return nullptr;

  auto otherElement = leftIsVoid ? rightElement : leftElement;
  if (!traits.is_void(otherElement) && !traits.is_object(otherElement))
    return nullptr;

  auto cv = cv_qualifiers(leftElement) | cv_qualifiers(rightElement);
  return control_->getPointerType(traits.add_cv(control_->getVoidType(), cv));
}

auto StandardConversion::compositeFunctionPointerType(const Type* left,
                                                      const Type* right)
    -> const Type* {
  auto leftPointer = type_cast<PointerType>(left);
  auto rightPointer = type_cast<PointerType>(right);
  if (leftPointer && rightPointer) {
    auto leftFunction = type_cast<FunctionType>(leftPointer->elementType());
    auto rightFunction = type_cast<FunctionType>(rightPointer->elementType());
    if (!leftFunction || !rightFunction) return nullptr;
    auto functionType = traits.remove_noexcept(leftFunction);
    if (!traits.is_same(functionType, traits.remove_noexcept(rightFunction)))
      return nullptr;
    return control_->getPointerType(functionType);
  }

  auto leftMember = decomposeMemberPointer(traits.remove_cv(left));
  auto rightMember = decomposeMemberPointer(traits.remove_cv(right));
  if (!leftMember || !rightMember) return nullptr;

  auto leftFunction = type_cast<FunctionType>(leftMember.pointeeType);
  auto rightFunction = type_cast<FunctionType>(rightMember.pointeeType);
  if (!leftFunction || !rightFunction) return nullptr;
  if (!traits.is_same(leftMember.classType, rightMember.classType))
    return nullptr;

  auto functionType =
      type_cast<FunctionType>(traits.remove_noexcept(leftFunction));
  if (!traits.is_same(functionType, traits.remove_noexcept(rightFunction)))
    return nullptr;

  return control_->getMemberFunctionPointerType(leftMember.classType,
                                                functionType);
}

auto StandardConversion::compositeClassAdjustedType(const Type* type,
                                                    const Type* classType)
    -> const Type* {
  if (auto pointerType = type_cast<PointerType>(type)) {
    auto element = pointerType->elementType();
    return control_->getPointerType(
        traits.add_cv(classType, cv_qualifiers(element)));
  }

  auto member = decomposeMemberPointer(traits.remove_cv(type));
  if (!member) return type;

  if (auto functionType = type_cast<FunctionType>(member.pointeeType)) {
    return control_->getMemberFunctionPointerType(classType, functionType);
  }

  return control_->getMemberObjectPointerType(classType, member.pointeeType);
}

auto StandardConversion::compositePointerClassType(const Type* left,
                                                   const Type* right,
                                                   bool contravariant)
    -> const Type* {
  auto leftUnqualified = traits.remove_cv(left);
  auto rightUnqualified = traits.remove_cv(right);

  if (traits.is_same(leftUnqualified, rightUnqualified)) return nullptr;

  auto base = leftUnqualified;
  auto derived = rightUnqualified;
  if (traits.is_base_of(rightUnqualified, leftUnqualified)) {
    base = rightUnqualified;
    derived = leftUnqualified;
  } else if (!traits.is_base_of(leftUnqualified, rightUnqualified)) {
    return nullptr;
  }

  return contravariant ? derived : base;
}

void StandardConversion::normalizeCompositePointerClass(const Type*& left,
                                                        const Type*& right) {
  auto leftPointer = type_cast<PointerType>(left);
  auto rightPointer = type_cast<PointerType>(right);

  if (leftPointer && rightPointer) {
    auto classType = compositePointerClassType(leftPointer->elementType(),
                                               rightPointer->elementType(),
                                               /*contravariant=*/false);
    if (!classType) return;
    left = compositeClassAdjustedType(left, classType);
    right = compositeClassAdjustedType(right, classType);
    return;
  }

  auto leftMember = decomposeMemberPointer(traits.remove_cv(left));
  auto rightMember = decomposeMemberPointer(traits.remove_cv(right));
  if (!leftMember || !rightMember) return;

  auto classType =
      compositePointerClassType(leftMember.classType, rightMember.classType,
                                /*contravariant=*/true);
  if (!classType) return;
  left = compositeClassAdjustedType(left, classType);
  right = compositeClassAdjustedType(right, classType);
}

auto StandardConversion::compositePointerType(ExpressionAST*& expr,
                                              ExpressionAST*& other)
    -> const Type* {
  if (traits.is_null_pointer(expr->type) && traits.is_null_pointer(other->type))
    return control_->getNullptrType();

  if (isNullPointerConstant(expr)) return other->type;
  if (isNullPointerConstant(other)) return expr->type;

  auto left = expr->type;
  auto right = other->type;

  if (auto type = compositeVoidPointerType(left, right)) return type;

  normalizeCompositePointerClass(left, right);

  if (auto type = compositeFunctionPointerType(left, right)) return type;

  if (!traits.is_similar(left, right)) return nullptr;

  return traits.qualification_combined_type(left, right);
}

auto StandardConversion::computeConversionSequence(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind, ConversionContext context)
    -> ImplicitConversionSequence {
  auto sequence = computeConversionSequenceSteps(expr, targetType,
                                                 initializationKind, context);

  if (!sequence) return sequence;

  if (sequence.form == ConversionSequenceForm::kStandard &&
      !sequence.binding.binds())
    bindResultToReference(sequence, targetType, ValueCategory::kPrValue);

  appendTemporaryMaterialization(sequence);

  return sequence;
}

void StandardConversion::appendTemporaryMaterialization(
    ImplicitConversionSequence& sequence) {
  if (!sequence.binding.bindsToTemporary()) return;
  if (sequence.udc.secondTarget) return;

  auto referencedType = sequence.binding.referencedType;
  if (!referencedType) return;
  if (traits.is_class(referencedType) || traits.is_array(referencedType))
    return;

  sequence.steps.push_back(
      {ImplicitCastKind::kTemporaryMaterializationConversion, referencedType});
}

auto StandardConversion::directReferenceBindingCastKind(
    const Type* referencedType, const Type* sourceType) const
    -> ImplicitCastKind {
  auto target = traits.remove_cv(referencedType);
  auto source = traits.remove_cv(sourceType);

  if (traits.is_same(source, target)) {
    if (cv_qualifiers(referencedType) == cv_qualifiers(sourceType))
      return ImplicitCastKind::kIdentity;
    return ImplicitCastKind::kQualificationConversion;
  }

  if (traits.is_base_of(target, source))
    return ImplicitCastKind::kDerivedToBaseConversion;

  return ImplicitCastKind::kIdentity;
}

auto StandardConversion::referenceBindingSequence(ExpressionAST* expr,
                                                  const Type* targetType)
    -> std::optional<ImplicitConversionSequence> {
  const Type* referencedType = nullptr;
  bool isRvalueReference = false;

  if (auto rvalueRef = type_cast<RvalueReferenceType>(targetType)) {
    referencedType = rvalueRef->elementType();
    isRvalueReference = true;
  } else if (auto lvalueRef = type_cast<LvalueReferenceType>(targetType)) {
    referencedType = lvalueRef->elementType();
  } else {
    return std::nullopt;
  }

  auto sourceType = traits.remove_reference(expr->type);
  const bool sourceIsLvalue = expr->valueCategory == ValueCategory::kLValue;
  const bool referenceCompatible =
      traits.is_reference_compatible(referencedType, sourceType);

  ImplicitConversionSequence seq;
  seq.sourceType = expr->type;
  seq.destinationType = targetType;
  seq.binding.isRvalueRef = isRvalueReference;

  auto bindDirectly =
      [&](ValueCategory category) -> ImplicitConversionSequence {
    bindResultToReference(seq, targetType, category);
    seq.binding.isDirect = true;
    seq.form = ConversionSequenceForm::kStandard;
    auto castKind = is_prvalue(expr) ? ImplicitCastKind::kIdentity
                                     : directReferenceBindingCastKind(
                                           referencedType, sourceType);
    seq.steps.push_back({castKind, referencedType});
    return seq;
  };

  if (!isRvalueReference && sourceIsLvalue && referenceCompatible)
    return bindDirectly(ValueCategory::kLValue);

  const bool convertsThroughConversionFunction =
      traits.is_class(traits.remove_cv(sourceType)) &&
      !traits.is_reference_related(referencedType, sourceType);

  if (convertsThroughConversionFunction) return std::nullopt;

  const auto referencedCv = cv_qualifiers(referencedType);

  if (!isRvalueReference &&
      (!has_const(referencedCv) || has_volatile(referencedCv)))
    return seq;

  if (referenceCompatible &&
      (!sourceIsLvalue || traits.is_function(sourceType)))
    return bindDirectly(expr->valueCategory);

  if (traits.is_reference_related(referencedType, sourceType)) return seq;

  return std::nullopt;
}

auto StandardConversion::isDesignatedInitializerList(
    BracedInitListAST* bracedInitList) const -> bool {
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    if (ast_cast<DesignatedInitializerClauseAST>(it->value)) return true;
  }
  return false;
}

auto StandardConversion::singleListElement(
    BracedInitListAST* bracedInitList) const -> ExpressionAST* {
  auto elements = bracedInitList->expressionList;
  if (!elements || elements->next) return nullptr;
  return elements->value;
}

auto StandardConversion::initializesCharacterArrayFromStringLiteral(
    BracedInitListAST* bracedInitList, const Type* arrayType) const -> bool {
  if (!traits.is_array(arrayType)) return false;
  if (!traits.is_char_type(
          traits.remove_cv(traits.get_element_type(arrayType))))
    return false;

  auto element = singleListElement(bracedInitList);
  if (!element || !element->type) return false;
  if (!ast_cast<StringLiteralExpressionAST>(element)) return false;

  auto literalType = traits.remove_cv(traits.remove_reference(element->type));
  if (!traits.is_array(literalType)) return false;

  return traits.is_same(traits.remove_cv(traits.get_element_type(arrayType)),
                        traits.remove_cv(traits.get_element_type(literalType)));
}

auto StandardConversion::listInitializationSequence(
    BracedInitListAST* bracedInitList, const Type* targetType,
    InitializationKind initializationKind) -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  seq.destinationType = targetType;
  seq.list.isListInitialization = true;

  auto listTarget = traits.remove_cv(traits.remove_reference(targetType));

  auto complete = [&](ImplicitCastKind kind,
                      const Type* type) -> ImplicitConversionSequence {
    if (seq.form == ConversionSequenceForm::kNone)
      seq.form = ConversionSequenceForm::kStandard;
    seq.steps.push_back({kind, type});
    return seq;
  };

  auto aggregateInitialization = [&]() -> ImplicitConversionSequence {
    seq.form = ConversionSequenceForm::kUserDefined;
    seq.list.narrowsElement =
        narrowsAggregateElement(bracedInitList, listTarget);
    seq.udc.aggregateInitializedClass = listTarget;
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return complete(ImplicitCastKind::kIdentity, listTarget);
  };

  auto convertSingleElement =
      [&](ExpressionAST* element,
          const Type* elementTarget) -> ImplicitConversionSequence {
    const bool narrows = traits.is_narrowing_list_element(element, listTarget);
    auto elementSeq =
        computeConversionSequence(element, elementTarget, initializationKind);
    elementSeq.list.isListInitialization = true;
    elementSeq.list.fromSingleElement = bool(elementSeq);
    elementSeq.list.elementCount = 1;
    elementSeq.list.narrowsElement = narrows;
    return elementSeq;
  };

  if (!traits.is_reference(targetType) &&
      isDesignatedInitializerList(bracedInitList)) {
    if (!traits.is_aggregate(listTarget)) return seq;
    if (!listInitializes(bracedInitList, listTarget, initializationKind))
      return seq;
    return aggregateInitialization();
  }

  if (traits.is_class(listTarget) && traits.is_aggregate(listTarget)) {
    if (auto element = singleListElement(bracedInitList);
        element && element->type) {
      auto elementType =
          traits.remove_cv(traits.remove_reference(element->type));
      if (traits.is_same(elementType, listTarget) ||
          traits.is_base_of(listTarget, elementType))
        return convertSingleElement(element, targetType);
    }
  }

  if (initializesCharacterArrayFromStringLiteral(bracedInitList, listTarget)) {
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return complete(ImplicitCastKind::kIdentity, listTarget);
  }

  if (auto elemType = traits.initializer_list_element_type(targetType)) {
    auto worstRank = ConversionRank::kExactMatch;
    std::size_t elementCount = 0;
    for (auto it = bracedInitList->expressionList; it; it = it->next) {
      if (!it->value) return seq;
      auto elemSeq = computeConversionSequence(it->value, elemType);
      if (!elemSeq) return seq;
      if (traits.is_narrowing_list_element(it->value, elemType))
        seq.list.narrowsElement = true;
      worstRank = std::min(worstRank, elemSeq.rank());
      ++elementCount;
    }

    seq.list.initializerListElementType = elemType;
    seq.list.elementCount = elementCount;
    seq.list.elementRank = worstRank;
    return complete(ImplicitCastKind::kIdentity, targetType);
  }

  if (traits.is_array(listTarget)) {
    if (traits.is_lvalue_reference(targetType) &&
        !traits.is_const(traits.remove_reference(targetType)))
      return seq;

    auto elementType = traits.remove_cv(traits.get_element_type(listTarget));
    auto worstRank = ConversionRank::kExactMatch;
    std::size_t elementCount = 0;

    for (auto it = bracedInitList->expressionList; it; it = it->next) {
      if (!it->value || !it->value->type) return seq;
      auto elementSeq = computeConversionSequence(it->value, elementType);
      if (!elementSeq) return seq;
      if (traits.is_narrowing_list_element(it->value, elementType))
        seq.list.narrowsElement = true;
      worstRank = std::min(worstRank, elementSeq.rank());
      ++elementCount;
    }

    if (auto bounded = type_cast<BoundedArrayType>(listTarget);
        bounded && elementCount > bounded->size())
      return seq;

    seq.list.elementCount = elementCount;
    seq.list.elementRank = worstRank;
    seq.list.targetIsUnboundedArray = traits.is_unbounded_array(listTarget);
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return complete(ImplicitCastKind::kIdentity, listTarget);
  }

  if (traits.is_class(listTarget)) {
    if (!listInitializes(bracedInitList, listTarget, initializationKind))
      return seq;
    return aggregateInitialization();
  }

  if (auto element = singleListElement(bracedInitList)) {
    if (ast_cast<BracedInitListAST>(element)) return seq;
    return convertSingleElement(element, listTarget);
  }

  if (!bracedInitList->expressionList) {
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return complete(ImplicitCastKind::kIdentity, listTarget);
  }

  return seq;
}

auto StandardConversion::computeConversionSequenceSteps(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind, ConversionContext context)
    -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  if (!expr || !targetType) return seq;

  seq.sourceType = expr->type;
  seq.destinationType = targetType;

  const Type* currentType = expr->type;
  ValueCategory currentValCat = expr->valueCategory;

  auto addStep = [&](ImplicitCastKind kind, const Type* type) {
    seq.steps.push_back({kind, type});
  };

  auto complete = [&](ImplicitCastKind kind,
                      const Type* type) -> ImplicitConversionSequence {
    if (seq.form == ConversionSequenceForm::kNone)
      seq.form = ConversionSequenceForm::kStandard;
    seq.steps.push_back({kind, type});
    return seq;
  };

  auto unreferencedSourceType = traits.remove_reference(currentType);
  auto overloadSetType = type_cast<OverloadSetType>(unreferencedSourceType);
  auto sourceIsAddressOfOverloadSet = false;

  if (!overloadSetType) {
    if (auto sourcePointer = type_cast<PointerType>(unreferencedSourceType)) {
      overloadSetType =
          type_cast<OverloadSetType>(sourcePointer->elementType());
      sourceIsAddressOfOverloadSet = overloadSetType != nullptr;
    }
  }

  if (overloadSetType) {
    if (auto ptrTarget = type_cast<PointerType>(targetType)) {
      if (auto targetFuncType =
              type_cast<FunctionType>(ptrTarget->elementType())) {
        if (resolveOverloadSetAgainstFunctionType(
                unit_, overloadSetType->symbol(), targetFuncType,
                expr->firstSourceLocation())) {
          return complete(sourceIsAddressOfOverloadSet
                              ? ImplicitCastKind::kIdentity
                              : ImplicitCastKind::kFunctionToPointerConversion,
                          targetType);
        }
      }
    }
    return seq;
  }

  if (auto bracedInitList = ast_cast<BracedInitListAST>(expr)) {
    return listInitializationSequence(bracedInitList, targetType,
                                      initializationKind);
  }

  if (auto referenceSequence = referenceBindingSequence(expr, targetType))
    return *referenceSequence;

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
    currentType = traits.adjusted_cv_type(traits.remove_reference(currentType));
    if (isC_ || !traits.is_class(currentType)) {
      currentValCat = ValueCategory::kPrValue;
      addStep(ImplicitCastKind::kLValueToRValueConversion, currentType);
    }
  }

  auto comparisonTargetType = traits.remove_reference(targetType);

  auto unqualFrom = traits.remove_cv(currentType);
  auto unqualTo = traits.remove_cv(comparisonTargetType);

  if (isDirectInitialization(initializationKind) &&
      traits.is_null_pointer(unqualFrom) &&
      traits.is_same(unqualTo, control_->getBoolType())) {
    return complete(ImplicitCastKind::kBooleanConversion, comparisonTargetType);
  }

  if (traits.is_same(unqualFrom, unqualTo)) {
    seq.requiresCopyConstruction = requiresCopyConstruction(expr, targetType);
    seq.copyConstructor = selectCopyConstructor(expr, targetType);
    bindResultToReference(seq, targetType, currentValCat);
    return complete(ImplicitCastKind::kIdentity, comparisonTargetType);
  }

  if (!isC_ && classAdjustment(unqualFrom, unqualTo) ==
                   ClassAdjustment::kDerivedToBase) {
    bindResultToReference(seq, targetType, currentValCat);
    return complete(ImplicitCastKind::kDerivedToBaseConversion,
                    comparisonTargetType);
  }

  if (traits.is_null_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    return complete(ImplicitCastKind::kPointerConversion, comparisonTargetType);
  }

  if (traits.is_integral(unqualFrom) && traits.is_pointer(unqualTo) &&
      isNullPointerConstant(expr)) {
    return complete(ImplicitCastKind::kPointerConversion, comparisonTargetType);
  }

  if (traits.is_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    auto fromPtr = unqualified_cast<PointerType>(unqualFrom);
    auto toPtr = unqualified_cast<PointerType>(unqualTo);

    if (fromPtr && toPtr) {
      auto fromPointee = fromPtr->elementType();
      auto toPointee = toPtr->elementType();

      auto fromCv = cv_qualifiers(fromPointee);
      auto toCv = cv_qualifiers(toPointee);

      if (is_at_least_as_cv_qualified(toCv, fromCv)) {
        auto fromUnqual = traits.remove_cv(fromPointee);
        auto toUnqual = traits.remove_cv(toPointee);

        if (traits.is_qualification_convertible(
                control_->getPointerType(fromPointee),
                control_->getPointerType(toPointee))) {
          seq.pointeeUnqual = toUnqual;
          seq.pointeeCv = toCv;
          return complete(ImplicitCastKind::kQualificationConversion,
                          comparisonTargetType);
        }

        if (traits.is_void(toUnqual)) {
          seq.pointeeUnqual = toUnqual;
          seq.pointeeCv = toCv;
          return complete(ImplicitCastKind::kPointerConversion,
                          comparisonTargetType);
        }

        if (pointeeClassAdjustment(unqualFrom, unqualTo) ==
            ClassAdjustment::kDerivedToBase) {
          seq.pointeeUnqual = toUnqual;
          seq.pointeeCv = toCv;
          return complete(ImplicitCastKind::kDerivedToBaseConversion,
                          comparisonTargetType);
        }

        if (isC_ && traits.is_void(fromUnqual)) {
          return complete(ImplicitCastKind::kPointerConversion,
                          comparisonTargetType);
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
            return complete(ImplicitCastKind::kPointerConversion,
                            comparisonTargetType);
          }
        }
      }
    }
  }

  if (traits.is_integral_promotion(unqualFrom, unqualTo)) {
    return complete(ImplicitCastKind::kIntegralPromotion, comparisonTargetType);
  }

  if (traits.is_floating_point_promotion(unqualFrom, unqualTo)) {
    return complete(ImplicitCastKind::kFloatingPointPromotion,
                    comparisonTargetType);
  }

  if ((traits.is_arithmetic(unqualFrom) ||
       (traits.is_enum(unqualFrom) && !traits.is_scoped_enum(unqualFrom))) &&
      traits.is_arithmetic(unqualTo)) {
    if (traits.is_integral_or_unscoped_enum(unqualFrom) &&
        traits.is_integral(unqualTo)) {
      return complete(ImplicitCastKind::kIntegralConversion,
                      comparisonTargetType);
    }

    if (traits.is_floating_point(unqualFrom) &&
        traits.is_floating_point(unqualTo)) {
      return complete(ImplicitCastKind::kFloatingPointConversion,
                      comparisonTargetType);
    }

    return complete(ImplicitCastKind::kFloatingIntegralConversion,
                    comparisonTargetType);
  }

  {
    auto source = decomposeMemberPointer(unqualFrom);
    auto target = decomposeMemberPointer(unqualTo);

    if (source && target &&
        isMemberPointeeConvertible(source.pointeeType, target.pointeeType)) {
      if (traits.is_same(source.classType, target.classType)) {
        return complete(ImplicitCastKind::kQualificationConversion,
                        comparisonTargetType);
      }

      auto targetClass = type_cast<ClassType>(target.classType);
      auto sourceClass = type_cast<ClassType>(source.classType);
      if (targetClass && sourceClass &&
          hasUniqueNonVirtualBase(targetClass, sourceClass)) {
        return complete(ImplicitCastKind::kPointerToMemberConversion,
                        comparisonTargetType);
      }
    }
  }

  if (traits.is_member_pointer(unqualTo) && isNullPointerConstant(expr)) {
    return complete(ImplicitCastKind::kPointerToMemberConversion,
                    comparisonTargetType);
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
        return complete(ImplicitCastKind::kFunctionPointerConversion,
                        comparisonTargetType);
      }
    }
  }

  if (traits.is_same(unqualTo, control_->getBoolType())) {
    if (traits.is_arithmetic_or_unscoped_enum(unqualFrom) ||
        traits.is_pointer(unqualFrom) || traits.is_member_pointer(unqualFrom)) {
      return complete(ImplicitCastKind::kBooleanConversion,
                      comparisonTargetType);
    }
  }

  if (isC_ && traits.is_integral_or_unscoped_enum(unqualFrom) &&
      traits.is_enum(unqualTo) && !traits.is_scoped_enum(unqualTo)) {
    return complete(ImplicitCastKind::kIntegralConversion,
                    comparisonTargetType);
  }

  if (context == ConversionContext::kStandardOnly) return seq;

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
      return traits.is_reference_compatible(
          inner, traits.remove_reference(resultType));
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
      [&](FunctionSymbol* func, ConversionRank s2Rank,
          const std::vector<ImplicitConversionSequence::Step>& secondSteps)
      -> ImplicitConversionSequence {
    ImplicitConversionSequence uds;
    uds.form = ConversionSequenceForm::kUserDefined;
    uds.sourceType = expr->type;
    uds.destinationType = targetType;
    uds.udc.function = func;
    uds.udc.secondRank = s2Rank;
    uds.udc.secondSteps = secondSteps;

    auto resultType = conversionResultType(func);
    auto resultValueCategory = ValueCategory::kPrValue;
    if (type_cast<LvalueReferenceType>(resultType))
      resultValueCategory = ValueCategory::kLValue;
    else if (type_cast<RvalueReferenceType>(resultType))
      resultValueCategory = ValueCategory::kXValue;
    bindResultToReference(uds, targetType, resultValueCategory);
    uds.steps.push_back({ImplicitCastKind::kUserDefinedConversion, resultType});

    if (!traits.is_same(traits.remove_cv(resultType), comparisonTargetType)) {
      uds.udc.secondTarget = comparisonTargetType;
    } else {
      uds.udc.secondSteps.clear();
    }

    return uds;
  };

  ImplicitConversionSequence bestUserDefined;
  FunctionSymbol* bestConversionFunction = nullptr;
  bool userDefinedIsAmbiguous = false;

  auto standardConversionSequence =
      [&](ExpressionAST* source, const Type* to) -> ImplicitConversionSequence {
    return computeConversionSequence(source, to,
                                     InitializationKind::kCopyInitialization,
                                     ConversionContext::kStandardOnly);
  };

  auto updateBest = [&](FunctionSymbol* func, ConversionRank s2Rank,
                        const std::vector<ImplicitConversionSequence::Step>&
                            secondSteps = {}) {
    if (bestUserDefined) {
      if (func == bestConversionFunction) return;
      if (s2Rank < bestUserDefined.udc.secondRank) return;
      if (s2Rank == bestUserDefined.udc.secondRank) {
        if (func->isSpecialization() !=
            bestConversionFunction->isSpecialization()) {
          if (func->isSpecialization()) return;
        } else if (!func->isSpecialization()) {
          userDefinedIsAmbiguous = true;
          return;
        } else {
          auto order = compareFunctionTemplateSpecializations(
              unit_, func, bestConversionFunction);
          if (order < 0) return;
          if (order == 0) {
            userDefinedIsAmbiguous = true;
            return;
          }
        }
      }
    }
    bestUserDefined = makeUserDefinedSeq(func, s2Rank, secondSteps);
    bestConversionFunction = func;
    userDefinedIsAmbiguous = false;
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

          auto instCtor = ASTRewriter::instantiateOverloadCandidate(
              unit_, *deducedArgs, ctor, expr->firstSourceLocation(),
              /*argsComplete=*/false);
          if (!instCtor) continue;

          ctor = instCtor;
        }

        auto funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (!isCallableWithOneArgument(ctor)) continue;

        auto argumentSequence = standardConversionSequence(expr, params[0]);
        if (argumentSequence) updateBest(ctor, argumentSequence.rank());
      }
    }
  }

  if (auto srcClassType = type_cast<ClassType>(unqualFrom)) {
    if (auto srcClass = srcClassType->symbol()) {
      traits.requireCompleteClass(srcClass);

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
            if (traits.is_qualification_convertible(retUnqual, unqualTo))
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
          auto secondSequence = standardConversionSequence(result, unqualTo);
          if (secondSequence)
            updateBest(convFunc, secondSequence.rank(), secondSequence.steps);
        }
      }
    }
  }

  if (bestUserDefined) {
    if (userDefinedIsAmbiguous)
      bestUserDefined.form = ConversionSequenceForm::kAmbiguous;
    return bestUserDefined;
  }

  return seq;
}

void StandardConversion::applyStep(const ImplicitConversionSequence& sequence,
                                   const ImplicitConversionSequence::Step& step,
                                   ExpressionAST*& expr) {
  if (step.kind == ImplicitCastKind::kIdentity) {
    if (auto braced = ast_cast<BracedInitListAST>(expr);
        braced && !braced->type && step.type) {
      braced->type = step.type;
      braced->valueCategory = ValueCategory::kPrValue;
    }
    return;
  }

  if (step.kind == ImplicitCastKind::kTemporaryMaterializationConversion) {
    (void)temporaryMaterialization(expr);
    return;
  }

  wrapWithImplicitCast(step.kind, step.type, expr);

  if (step.kind != ImplicitCastKind::kUserDefinedConversion) return;
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr))
    recordUserDefinedConversion(cast, sequence.udc.function);
}

void StandardConversion::applyConversionSequence(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  if (!sequence) return;

  resolveOverloadSet(expr, sequence.destinationType);

  if (sequence.list.fromSingleElement) {
    if (auto braced = ast_cast<BracedInitListAST>(expr);
        braced && braced->expressionList && !braced->expressionList->next) {
      expr = braced->expressionList->value;
    }
  }

  for (const auto& step : sequence.steps) applyStep(sequence, step, expr);
  for (const auto& step : sequence.udc.secondSteps)
    applyStep(sequence, step, expr);

  if (sequence.copyConstructor) applyCopyConstruction(sequence, expr);

  requireDefinitionOfDesignatedField(expr);
}

void StandardConversion::applyCopyConstruction(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kUserDefinedConversion;
  cast->expression = expr;
  cast->type = traits.remove_cv(sequence.destinationType);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  recordUserDefinedConversion(cast, sequence.copyConstructor);
}

void StandardConversion::recordConversionFunction(
    ImplicitCastExpressionAST* cast,
    const ImplicitConversionSequence& sequence) {
  auto function = sequence.form == ConversionSequenceForm::kUserDefined
                      ? sequence.udc.function
                      : sequence.copyConstructor;
  if (!function) return;
  recordUserDefinedConversion(cast, function);
}

auto StandardConversion::requiresCopyConstruction(
    ExpressionAST* expr, const Type* destinationType) const -> bool {
  if (isC_ || !expr || !expr->type || !destinationType) return false;
  if (!is_glvalue(expr)) return false;
  if (traits.is_reference(destinationType)) return false;

  auto classType = unqualified_cast<ClassType>(destinationType);
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

  auto classType = unqualified_cast<ClassType>(destinationType);
  auto classSymbol = classType->symbol()->resolvedDefinition();

  if (!control_->beginCopyConstructorSelection(classSymbol)) return nullptr;

  OverloadResolution resolution(unit_);
  auto resolved = resolution.resolveConstructor(classSymbol, {expr});

  control_->endCopyConstructorSelection(classSymbol);

  if (!resolved.best || resolved.ambiguous) return nullptr;

  return resolved.best->symbol;
}

void StandardConversion::materializeConstructorArguments(
    ImplicitCastExpressionAST* cast, FunctionSymbol* constructor) {
  if (ast_cast<ParenInitializerAST>(cast->expression)) return;

  auto functionType = type_cast<FunctionType>(constructor->type());
  if (!functionType) return;
  auto parameterTypes = functionType->parameterTypes();
  if (parameterTypes.empty()) return;

  auto arguments = make_list_node<ExpressionAST>(arena_, cast->expression);

  auto sequence =
      computeConversionSequence(arguments->value, parameterTypes[0]);
  applyConversionSequence(sequence, arguments->value);

  appendDefaultArguments(constructor, &arguments);

  auto paren = ParenInitializerAST::create(
      arena_, cast->firstSourceLocation(), arguments,
      cast->firstSourceLocation(), ValueCategory::kPrValue, cast->type);
  cast->expression = paren;
}

auto StandardConversion::narrowsAggregateElement(
    BracedInitListAST* bracedInitList, const Type* targetType) -> bool {
  auto classType = unqualified_cast<ClassType>(targetType);
  if (!classType || !classType->symbol()) return false;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  if (!traits.is_aggregate(classType)) return false;

  auto slots = aggregateInitializerSlots(bracedInitList, classSymbol);
  if (!slots) return false;

  for (const auto& slot : *slots) {
    if (!slot.initializer || !slot.elementType) continue;
    if (traits.is_narrowing_list_element(slot.initializer, slot.elementType))
      return true;
  }

  return false;
}

auto StandardConversion::designatedAggregateSlot(
    const std::vector<Symbol*>& elements,
    DesignatedInitializerClauseAST* designated) const
    -> std::optional<std::size_t> {
  auto designatorList = designated->designatorList;
  if (!designatorList || designatorList->next) return std::nullopt;

  auto dot = ast_cast<DotDesignatorAST>(designatorList->value);
  if (!dot || !dot->identifier) return std::nullopt;

  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->name() == dot->identifier) return i;
  }

  return std::nullopt;
}

auto StandardConversion::aggregateInitializerSlots(
    BracedInitListAST* bracedInitList, ClassSymbol* classSymbol)
    -> std::optional<std::vector<AggregateInitializerSlot>> {
  auto elements = traits.aggregate_elements(classSymbol);

  std::vector<AggregateInitializerSlot> slots;
  std::size_t nextIndex = 0;

  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    auto initializer = it->value;
    auto index = nextIndex;

    if (auto designated =
            ast_cast<DesignatedInitializerClauseAST>(initializer)) {
      auto designatedIndex = designatedAggregateSlot(elements, designated);
      if (!designatedIndex) return std::nullopt;
      index = *designatedIndex;
      initializer = designated->initializer;
    }

    if (index >= elements.size()) return std::nullopt;
    nextIndex = index + 1;

    slots.push_back(
        {initializer, traits.aggregate_element_type(elements[index])});
  }

  return slots;
}

auto StandardConversion::listInitializes(BracedInitListAST* bracedInitList,
                                         const Type* targetType,
                                         InitializationKind initializationKind)
    -> bool {
  auto classType = unqualified_cast<ClassType>(targetType);
  if (!classType || !classType->symbol()) return false;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  traits.requireCompleteClass(classSymbol);
  if (!classSymbol->isComplete()) return false;

  std::vector<ExpressionAST*> elements;
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    if (!it->value) return false;
    if (!it->value->type && !ast_cast<BracedInitListAST>(it->value) &&
        !ast_cast<DesignatedInitializerClauseAST>(it->value))
      return false;
    elements.push_back(it->value);
  }

  if (!traits.is_aggregate(classType)) {
    OverloadResolution resolution(unit_);
    auto listInitializationKind = InitializationKind::kCopyListInitialization;
    if (isDirectInitialization(initializationKind)) {
      listInitializationKind = InitializationKind::kDirectListInitialization;
    }

    auto accepts = [&](const ConstructorResult& result) {
      if (!result.best || result.ambiguous) return false;
      if (listInitializationKind ==
              InitializationKind::kCopyListInitialization &&
          result.best->symbol->isExplicit()) {
        return false;
      }
      return true;
    };

    const bool emptyListSelectsDefaultConstructor =
        elements.empty() && classSymbol->defaultConstructor();
    if (!emptyListSelectsDefaultConstructor) {
      auto result = resolution.resolveInitializerListConstructor(
          classSymbol, bracedInitList, listInitializationKind);
      if (accepts(result)) return true;
      if (result.best) return false;
    }
    auto result = resolution.resolveConstructor(classSymbol, elements,
                                                listInitializationKind);
    return accepts(result);
  }

  auto slots = aggregateInitializerSlots(bracedInitList, classSymbol);
  if (!slots) return false;

  for (const auto& slot : *slots) {
    if (!slot.elementType || !slot.initializer) return false;
    if (traits.is_array(slot.elementType)) continue;
    if (!computeConversionSequence(slot.initializer, slot.elementType))
      return false;
  }

  return true;
}

auto StandardConversion::isCallableWithOneArgument(FunctionSymbol* ctor)
    -> bool {
  auto funcType = type_cast<FunctionType>(ctor->type());
  if (!funcType || funcType->parameterTypes().empty()) return false;
  if (funcType->parameterTypes().size() == 1) return true;

  auto params = ctor->parameters();
  if (params.size() != funcType->parameterTypes().size()) return false;

  for (std::size_t i = 1; i < params.size(); ++i)
    if (!params[i]->defaultArgument()) return false;

  return true;
}

void StandardConversion::appendDefaultArguments(FunctionSymbol* function,
                                                List<ExpressionAST*>** list) {
  auto params = function->parameters();
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

void StandardConversion::requireDefinitionOfDesignatedField(
    ExpressionAST* expr) {
  while (expr) {
    if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
      expr = nested->expression;
      continue;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
      expr = cast->expression;
      continue;
    }
    break;
  }

  auto id = ast_cast<IdExpressionAST>(expr);
  if (!id) return;
  ASTRewriter::requireFieldDefinition(unit_,
                                      symbol_cast<FieldSymbol>(id->symbol));
}

void StandardConversion::recordUserDefinedConversion(
    ImplicitCastExpressionAST* cast, FunctionSymbol* function) {
  if (!function) return;

  cast->conversionFunction = function;
  ASTRewriter::requireFunctionDefinition(unit_, function);

  if (function->isConstructor()) {
    materializeConstructorArguments(cast, function);
    return;
  }

  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  if (!classSymbol) return;

  auto& objectExpression = cast->expression;
  if (objectExpression && is_glvalue(objectExpression))
    (void)convertToBaseClass(objectExpression, classSymbol->type());

  cast->isVirtualDispatch =
      traits.is_virtual_member_dispatch(function, objectExpression);
}

auto StandardConversion::classAdjustment(const Type* sourceType,
                                         const Type* targetType) const
    -> ClassAdjustment {
  auto source = traits.remove_cv(traits.remove_reference(sourceType));
  auto target = traits.remove_cv(traits.remove_reference(targetType));

  if (!traits.is_class(source) || !traits.is_class(target))
    return ClassAdjustment::kNone;
  if (traits.is_same(source, target)) return ClassAdjustment::kNone;
  if (traits.is_base_of(target, source)) return ClassAdjustment::kDerivedToBase;
  if (traits.is_base_of(source, target)) return ClassAdjustment::kBaseToDerived;

  return ClassAdjustment::kNone;
}

auto StandardConversion::pointeeClassAdjustment(const Type* sourceType,
                                                const Type* targetType) const
    -> ClassAdjustment {
  auto sourcePointer = unqualified_cast<PointerType>(sourceType);
  auto targetPointer = unqualified_cast<PointerType>(targetType);
  if (!sourcePointer || !targetPointer) return ClassAdjustment::kNone;

  return classAdjustment(sourcePointer->elementType(),
                         targetPointer->elementType());
}

auto StandardConversion::convertToBaseClass(ExpressionAST*& expr,
                                            const Type* baseType) -> bool {
  if (!expr || !expr->type) return false;
  if (classAdjustment(expr->type, baseType) != ClassAdjustment::kDerivedToBase)
    return false;

  wrapWithImplicitCast(ImplicitCastKind::kDerivedToBaseConversion, baseType,
                       expr);
  return true;
}

auto StandardConversion::convertToDerivedClass(ExpressionAST*& expr,
                                               const Type* derivedType)
    -> bool {
  if (!expr || !expr->type) return false;
  if (classAdjustment(expr->type, derivedType) !=
      ClassAdjustment::kBaseToDerived)
    return false;

  wrapWithImplicitCast(ImplicitCastKind::kBaseToDerivedConversion, derivedType,
                       expr);
  return true;
}

auto StandardConversion::pointerConversionCastKind(const Type* sourceType,
                                                   const Type* targetType) const
    -> ImplicitCastKind {
  switch (pointeeClassAdjustment(sourceType, targetType)) {
    case ClassAdjustment::kDerivedToBase:
      return ImplicitCastKind::kDerivedToBaseConversion;
    case ClassAdjustment::kBaseToDerived:
      return ImplicitCastKind::kBaseToDerivedConversion;
    case ClassAdjustment::kNone:
      return ImplicitCastKind::kPointerConversion;
    default:
      return ImplicitCastKind::kPointerConversion;
  }
}

void StandardConversion::convertPointer(ExpressionAST*& expr,
                                        const Type* targetType) {
  wrapWithImplicitCast(pointerConversionCastKind(expr->type, targetType),
                       targetType, expr);
}

void StandardConversion::setResolvedFunction(ExpressionAST* expr,
                                             FunctionSymbol* function) {
  if (auto idExpr = ast_cast<IdExpressionAST>(expr)) {
    idExpr->symbol = function;
    idExpr->type = function->type();
  } else if (auto memberExpr = ast_cast<MemberExpressionAST>(expr)) {
    memberExpr->symbol = function;
    memberExpr->type = function->type();
  }
}

void StandardConversion::resolveOverloadSet(ExpressionAST* expr,
                                            const Type* targetType) {
  auto targetPointer =
      type_cast<PointerType>(traits.remove_reference(targetType));
  if (!targetPointer) return;

  auto targetFunctionType =
      type_cast<FunctionType>(targetPointer->elementType());
  if (!targetFunctionType) return;

  auto stripped = stripNestedExpressions(expr);
  auto addressOf = ast_cast<UnaryExpressionAST>(stripped);
  const bool takesAddress = addressOf && addressOf->op == TokenKind::T_AMP;
  auto designator =
      takesAddress ? stripNestedExpressions(addressOf->expression) : stripped;
  if (!designator || !designator->type) return;

  auto overloadSetType =
      type_cast<OverloadSetType>(traits.remove_reference(designator->type));
  if (!overloadSetType) return;

  auto resolved = resolveOverloadSetAgainstFunctionType(
      unit_, overloadSetType->symbol(), targetFunctionType,
      expr->firstSourceLocation());
  if (!resolved) return;

  setResolvedFunction(designator, resolved);
  if (takesAddress)
    addressOf->type = control_->getPointerType(resolved->type());
}

void StandardConversion::wrapWithImplicitCast(ImplicitCastKind castKind,
                                              const Type* type,
                                              ExpressionAST*& expr) {
  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = castKind;
  cast->expression = expr;
  cast->type = type;
  cast->valueCategory = ValueCategory::kPrValue;
  if (castKind == ImplicitCastKind::kQualificationConversion)
    cast->valueCategory = expr->valueCategory;

  if (castKind == ImplicitCastKind::kDerivedToBaseConversion) {
    const auto convertsPointer = traits.is_pointer(type);
    if (!convertsPointer) {
      if (is_glvalue(expr)) cast->valueCategory = expr->valueCategory;
    }
  }

  expr = cast;
  foldConstantRead(expr);
}

}  // namespace cxx
