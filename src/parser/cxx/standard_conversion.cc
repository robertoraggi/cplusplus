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

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
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

[[nodiscard]] auto isClassAdjustment(ImplicitCastKind castKind) -> bool {
  return castKind == ImplicitCastKind::kDerivedToBaseConversion ||
         castKind == ImplicitCastKind::kBaseToDerivedConversion;
}

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

[[nodiscard]] auto completedSequence(ImplicitConversionSequence sequence,
                                     ImplicitCastKind kind, const Type* type)
    -> ImplicitConversionSequence {
  if (sequence.form == ConversionSequenceForm::kNone)
    sequence.form = ConversionSequenceForm::kStandard;
  sequence.steps.push_back({kind, type});
  return sequence;
}

[[nodiscard]] auto conversionResultValueCategory(const Type* type)
    -> ValueCategory {
  if (auto reference = type_cast<RvalueReferenceType>(type)) {
    if (unqualified_cast<FunctionType>(reference->elementType()))
      return ValueCategory::kLValue;
    return ValueCategory::kXValue;
  }
  if (type_cast<LvalueReferenceType>(type)) return ValueCategory::kLValue;
  return ValueCategory::kPrValue;
}

struct OverloadSetTarget {
  const FunctionType* functionType = nullptr;
  bool isMemberPointer = false;
};

[[nodiscard]] auto overloadSetTargetOf(TranslationUnit* unit,
                                       const Type* targetType)
    -> std::optional<OverloadSetTarget> {
  auto traits = unit->typeTraits();
  auto target = traits.remove_cv(traits.remove_reference(targetType));
  if (auto function = type_cast<FunctionType>(target))
    return OverloadSetTarget{.functionType = function};
  if (auto pointer = type_cast<PointerType>(target)) {
    auto function = type_cast<FunctionType>(pointer->elementType());
    if (function) return OverloadSetTarget{.functionType = function};
  }
  if (auto pointer = type_cast<MemberFunctionPointerType>(target))
    return OverloadSetTarget{.functionType = pointer->functionType(),
                             .isMemberPointer = true};
  return std::nullopt;
}

struct SelectedFunction {
  FunctionSymbol* function = nullptr;
  List<TemplateArgumentAST*>* deducedArguments = nullptr;
  bool isTemplateSpecialization = false;
};

[[nodiscard]] auto matchesTargetFunctionType(TranslationUnit* unit,
                                             const FunctionType* type,
                                             const FunctionType* target)
    -> bool {
  return unit->typeTraits().is_reference_compatible(target, type);
}

[[nodiscard]] auto selectedFunctionFor(TranslationUnit* unit,
                                       FunctionSymbol* function,
                                       const OverloadSetTarget& target,
                                       SourceLocation loc)
    -> std::optional<SelectedFunction> {
  if (function->canonical() != function) return std::nullopt;
  if (function->isSpecialization()) return std::nullopt;
  if (function->isImplicitObjectMemberFunction() != target.isMemberPointer)
    return std::nullopt;

  SelectedFunction selected{.function = function};

  if (function->templateDeclaration()) {
    TemplateArgumentDeduction deduction(unit);
    auto deducedArguments =
        deduction.deduceFromTargetType(function, target.functionType);
    if (!deducedArguments.has_value()) return std::nullopt;
    selected.function = ASTRewriter::instantiateOverloadCandidate(
        unit, *deducedArguments, function, loc, /*argsComplete=*/true);
    if (!selected.function) return std::nullopt;
    selected.deducedArguments = *deducedArguments;
    selected.isTemplateSpecialization = true;
  }

  auto type = type_cast<FunctionType>(selected.function->type());
  if (!type || !matchesTargetFunctionType(unit, type, target.functionType))
    return std::nullopt;

  if (ASTRewriter::evaluateAssociatedConstraints(unit, selected.function) ==
      false)
    return std::nullopt;

  return selected;
}

[[nodiscard]] auto isEliminatedBy(TranslationUnit* unit,
                                  const SelectedFunction& function,
                                  const SelectedFunction& other) -> bool {
  if (function.function == other.function) return false;
  if (function.isTemplateSpecialization != other.isTemplateSpecialization)
    return function.isTemplateSpecialization;
  if (function.isTemplateSpecialization)
    return compareFunctionTemplateSpecializations(unit, other.function,
                                                  function.function) > 0;
  return compareNonTemplateConstraints(unit, other.function,
                                       function.function) > 0;
}

[[nodiscard]] auto isEliminated(TranslationUnit* unit,
                                const SelectedFunction& function,
                                const std::vector<SelectedFunction>& selected)
    -> bool {
  for (const auto& other : selected) {
    if (isEliminatedBy(unit, function, other)) return true;
  }
  return false;
}

[[nodiscard]] auto resolveOverloadSetAgainstTarget(
    TranslationUnit* unit, OverloadSetSymbol* ovl,
    const OverloadSetTarget& target, SourceLocation loc) -> FunctionSymbol* {
  std::vector<SelectedFunction> selected;
  for (auto function : ovl->functions()) {
    auto candidate = selectedFunctionFor(unit, function, target, loc);
    if (!candidate) continue;
    if (std::ranges::contains(selected, candidate->function,
                              &SelectedFunction::function))
      continue;
    selected.push_back(*candidate);
  }

  auto survivors = selected;
  std::erase_if(survivors, [&](const SelectedFunction& function) {
    return isEliminated(unit, function, selected);
  });

  if (survivors.size() != 1) return nullptr;

  auto match = survivors.front();
  ASTRewriter::instantiateSelectedSpecializationDefinition(
      unit, match.function, match.deducedArguments);

  return match.function;
}

[[nodiscard]] auto stripNestedExpressions(ExpressionAST* expr)
    -> ExpressionAST* {
  while (auto nested = ast_cast<NestedExpressionAST>(expr))
    expr = nested->expression;
  return expr;
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

  auto baseDefinition = baseClass->resolvedDefinition();
  if (!derivedDefinition->baseSubobjectInfo(baseDefinition).isUniqueSubobject())
    return false;
  return !derivedDefinition->hasVirtualBasePath(baseDefinition);
}

auto StandardConversion::isMemberPointeeConvertible(const Type* source,
                                                    const Type* target)
    -> bool {
  if (traits.is_function(source))
    return traits.is_reference_compatible(target, source);

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

auto StandardConversion::readsValueDirectly(const Type* type) const -> bool {
  if (isC_) return true;
  return !traits.is_class(traits.remove_cv(traits.remove_reference(type)));
}

auto StandardConversion::valueTransformation(ExpressionAST* expr)
    -> std::optional<ImplicitConversionSequence::Step> {
  auto type = traits.remove_reference(expr->type);
  if (traits.is_array(type)) {
    auto kind = type_cast<UnresolvedBoundedArrayType>(type)
                    ? ImplicitCastKind::kLValueToRValueConversion
                    : ImplicitCastKind::kArrayToPointerConversion;
    return ImplicitConversionSequence::Step{
        kind, traits.add_pointer(traits.remove_extent(type))};
  }
  if (traits.is_function(type)) {
    if (is_prvalue(expr)) return std::nullopt;
    return ImplicitConversionSequence::Step{
        ImplicitCastKind::kFunctionToPointerConversion,
        traits.add_pointer(type)};
  }
  if (!is_glvalue(expr)) return std::nullopt;
  if (!traits.is_complete(type)) return std::nullopt;
  if (!readsValueDirectly(type)) return std::nullopt;
  return ImplicitConversionSequence::Step{
      ImplicitCastKind::kLValueToRValueConversion,
      traits.adjusted_cv_type(type)};
}

void StandardConversion::atomicToNonAtomic(ExpressionAST*& expr) {
  if (!traits.is_atomic(expr->type)) return;

  auto cast = ImplicitCastExpressionAST::create(arena_);
  cast->castKind = ImplicitCastKind::kAtomicToNonAtomic;
  cast->expression = expr;
  cast->type = traits.remove_atomic(expr->type);
  cast->valueCategory = ValueCategory::kPrValue;
  adjustCv(cast);
  expr = cast;
}

void StandardConversion::foldConstantRead(ExpressionAST*& expression) {
  auto cast = ast_cast<ImplicitCastExpressionAST>(expression);
  if (!cast) return;
  if (cast->castKind != ImplicitCastKind::kLValueToRValueConversion) return;
  if (isDependent(unit_, cast)) return;

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
  if (auto value = interp.evaluateStaticDataMember(field)) {
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

auto StandardConversion::integralPromotion(ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;
  if (!traits.is_integral(expr->type) && !traits.is_enum(expr->type))
    return false;

  auto promotedType = traits.promoted_integer_type(expr->type);
  if (traits.is_same(promotedType, expr->type)) return false;

  wrapWithImplicitCast(ImplicitCastKind::kIntegralPromotion, promotedType,
                       expr);
  return true;
}

auto StandardConversion::floatingPointPromotion(ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;
  if (!traits.is_floating_point(expr->type)) return false;
  auto destinationType = control_->getDoubleType();
  if (!traits.is_floating_point_promotion(expr->type, destinationType))
    return false;
  wrapWithImplicitCast(ImplicitCastKind::kFloatingPointPromotion,
                       destinationType, expr);
  return true;
}

auto StandardConversion::arithmeticConversionKind(const Type* source,
                                                  const Type* target) const
    -> std::optional<ImplicitCastKind> {
  if (!traits.is_arithmetic_or_unscoped_enum(source)) return std::nullopt;
  if (!traits.is_arithmetic(target)) return std::nullopt;
  if (traits.is_same(source, target)) return ImplicitCastKind::kIdentity;
  if (traits.is_same(target, control_->getBoolType()))
    return ImplicitCastKind::kBooleanConversion;
  if (traits.is_integral_promotion(source, target))
    return ImplicitCastKind::kIntegralPromotion;
  if (traits.is_floating_point_promotion(source, target))
    return ImplicitCastKind::kFloatingPointPromotion;
  if (traits.is_integral_or_unscoped_enum(source) && traits.is_integral(target))
    return ImplicitCastKind::kIntegralConversion;
  if (traits.is_floating_point(source) && traits.is_floating_point(target))
    return ImplicitCastKind::kFloatingPointConversion;
  return ImplicitCastKind::kFloatingIntegralConversion;
}

void StandardConversion::adjustCv(ExpressionAST* expr) {
  if (!is_prvalue(expr)) return;
  expr->type = traits.adjusted_cv_type(expr->type);
}

void StandardConversion::prepareOperand(ExpressionAST*& expr) {
  if (auto step = valueTransformation(expr)) {
    if (step->kind == ImplicitCastKind::kFunctionToPointerConversion)
      requireNamedFunction(expr);
    wrapWithImplicitCast(step->kind, step->type, expr);
    atomicToNonAtomic(expr);
  }
  adjustCv(expr);
}

void StandardConversion::promoteOperand(ExpressionAST*& expr) {
  prepareOperand(expr);
  if (integralPromotion(expr)) return;
  (void)floatingPointPromotion(expr);
}

void StandardConversion::decayOperand(ExpressionAST*& expr) {
  auto type = traits.remove_reference(expr->type);
  if (!traits.is_array(type) && !traits.is_function(type)) return;
  prepareOperand(expr);
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

  ASTRewriter::requireDestructorOfType(unit_, cast->type);

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

  for (auto convFunc : classSymbol->visibleConversionFunctions()) {
    if (convFunc->isExplicit()) continue;

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

  if (!target) return false;

  return convertImplicitly(expr, target);
}

auto StandardConversion::canSplatIntoVector(ExpressionAST* expr,
                                            const VectorType* vectorType)
    -> bool {
  auto scalarType = traits.remove_cv(expr->type);
  if (!traits.is_arithmetic_or_unscoped_enum(scalarType)) return false;

  auto elementType = vectorType->elementType();

  if (vectorType->vectorKind() == VectorKind::kExt) {
    return !(traits.is_floating_point(scalarType) &&
             traits.is_integral(elementType));
  }

  const bool scalarIsFloating = traits.is_floating_point(scalarType);
  const bool elementIsFloating = traits.is_floating_point(elementType);

  if (scalarIsFloating && !elementIsFloating) {
    auto scalarSize = control_->memoryLayout()->sizeOf(scalarType);
    auto elementSize = control_->memoryLayout()->sizeOf(elementType);
    return scalarSize && elementSize && *scalarSize == *elementSize;
  }

  if (!scalarIsFloating && elementIsFloating) return false;

  return !traits.is_narrowing_list_element(expr, elementType);
}

auto StandardConversion::vectorOperandConversion(ExpressionAST*& expr,
                                                 ExpressionAST*& other)
    -> std::optional<const Type*> {
  auto leftVector = unqualified_cast<VectorType>(expr->type);
  auto rightVector = unqualified_cast<VectorType>(other->type);

  if (!leftVector && !rightVector) return std::nullopt;

  if (leftVector && rightVector) {
    if (traits.is_same(unqualified_type(expr->type),
                       unqualified_type(other->type)))
      return static_cast<const Type*>(leftVector);

    const auto width = traits.vector_width_in_bytes(leftVector);
    if (!width || width != traits.vector_width_in_bytes(rightVector))
      return nullptr;

    const bool leftIsExt = leftVector->vectorKind() == VectorKind::kExt;
    const bool rightIsExt = rightVector->vectorKind() == VectorKind::kExt;

    const VectorType* common =
        rightIsExt && !leftIsExt ? rightVector : leftVector;

    wrapWithImplicitCast(ImplicitCastKind::kVectorConversion, common, expr);
    wrapWithImplicitCast(ImplicitCastKind::kVectorConversion, common, other);

    return static_cast<const Type*>(common);
  }

  auto vectorType = leftVector ? leftVector : rightVector;
  auto& scalar = leftVector ? other : expr;

  if (!canSplatIntoVector(scalar, vectorType)) return nullptr;

  if (!convertArithmetic(scalar, vectorType->elementType())) return nullptr;

  wrapWithImplicitCast(ImplicitCastKind::kVectorSplat, vectorType, scalar);

  return static_cast<const Type*>(vectorType);
}

auto StandardConversion::usualArithmeticConversion(ExpressionAST*& expr,
                                                   ExpressionAST*& other)
    -> const Type* {
  prepareOperand(expr);
  prepareOperand(other);

  if (auto vectorType = vectorOperandConversion(expr, other))
    return *vectorType;

  auto common = commonArithmeticType(expr->type, other->type);
  if (!common) return nullptr;

  if (!convertArithmetic(expr, common) || !convertArithmetic(other, common))
    return nullptr;

  return common;
}

auto StandardConversion::commonArithmeticType(const Type* a, const Type* b)
    -> const Type* {
  a = traits.remove_cv(a);
  b = traits.remove_cv(b);
  if (!traits.is_arithmetic_or_unscoped_enum(a)) return nullptr;
  if (!traits.is_arithmetic_or_unscoped_enum(b)) return nullptr;

  if (!isC_ && traits.is_enum(a) && !traits.is_same(a, b)) {
    if (traits.is_enum(b) || traits.is_floating_point(b)) return nullptr;
  }
  if (!isC_ && traits.is_enum(b) && traits.is_floating_point(a)) return nullptr;

  if (traits.is_complex(a) || traits.is_complex(b)) {
    auto commonReal = commonArithmeticType(traits.complex_element_type(a),
                                           traits.complex_element_type(b));
    if (!commonReal) return nullptr;
    return control_->getComplexType(commonReal);
  }
  if (traits.is_floating_point(a) || traits.is_floating_point(b)) {
    return traits.floating_point_conversion_rank(a) >=
                   traits.floating_point_conversion_rank(b)
               ? a
               : b;
  }
  a = traits.promoted_integer_type(a);
  b = traits.promoted_integer_type(b);
  if (traits.is_same(a, b)) return a;
  auto rankA = traits.integer_conversion_rank(a);
  auto rankB = traits.integer_conversion_rank(b);
  if (traits.is_unsigned(a) == traits.is_unsigned(b))
    return rankA >= rankB ? a : b;
  auto unsignedType = traits.is_unsigned(a) ? a : b;
  auto signedType = traits.is_unsigned(a) ? b : a;
  if (traits.integer_conversion_rank(unsignedType) >=
      traits.integer_conversion_rank(signedType))
    return unsignedType;
  if (traits.representsAllValuesOf(signedType, unsignedType)) return signedType;
  return traits.make_unsigned(signedType);
}

auto StandardConversion::convertArithmetic(ExpressionAST*& expr,
                                           const Type* destinationType)
    -> bool {
  if (traits.is_same(expr->type, destinationType)) return true;

  if (traits.is_complex(destinationType)) {
    if (traits.is_complex(expr->type)) {
      wrapWithImplicitCast(ImplicitCastKind::kComplexConversion,
                           destinationType, expr);
      return true;
    }
    wrapWithImplicitCast(ImplicitCastKind::kRealToComplexConversion,
                         destinationType, expr);
    return true;
  }

  if (traits.is_complex(expr->type)) {
    wrapWithImplicitCast(ImplicitCastKind::kComplexToRealConversion,
                         destinationType, expr);
    return true;
  }

  auto kind = arithmeticConversionKind(expr->type, destinationType);
  if (!kind || !is_prvalue(expr)) return false;
  wrapWithImplicitCast(*kind, destinationType, expr);
  return true;
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

  return control_->getMemberPointerType(classType, member.pointeeType);
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
    const Type* referencedType, const Type* sourceType) -> ImplicitCastKind {
  auto target = traits.remove_cv(referencedType);
  auto source = traits.remove_cv(sourceType);

  if (traits.is_same(source, target)) {
    if (cv_qualifiers(referencedType) == cv_qualifiers(sourceType))
      return ImplicitCastKind::kIdentity;
    return ImplicitCastKind::kQualificationConversion;
  }

  if (traits.is_base_of(target, source))
    return ImplicitCastKind::kDerivedToBaseConversion;

  if (traits.is_function(target))
    return ImplicitCastKind::kFunctionPointerConversion;
  return ImplicitCastKind::kQualificationConversion;
}

void StandardConversion::appendDirectBindingSteps(
    ImplicitConversionSequence& seq, const Type* referencedType,
    const Type* sourceType, ValueCategory sourceValueCategory) {
  auto castKind = directReferenceBindingCastKind(referencedType, sourceType);

  if (sourceValueCategory != ValueCategory::kPrValue) {
    seq.steps.push_back({castKind, referencedType});
    return;
  }

  if (castKind != ImplicitCastKind::kDerivedToBaseConversion) {
    seq.steps.push_back({ImplicitCastKind::kIdentity, referencedType});
    return;
  }

  seq.steps.push_back(
      {ImplicitCastKind::kTemporaryMaterializationConversion, sourceType});
  seq.steps.push_back({castKind, referencedType});
}

auto StandardConversion::directReferenceBinding(
    ImplicitConversionSequence sequence, ValueCategory category)
    -> ImplicitConversionSequence {
  bindResultToReference(sequence, sequence.destinationType, category);
  sequence.binding.isDirect = true;
  sequence.form = ConversionSequenceForm::kStandard;
  appendDirectBindingSteps(sequence, sequence.binding.referencedType,
                           traits.remove_reference(sequence.sourceType),
                           category);
  return sequence;
}

auto StandardConversion::referenceBinding(const Type* targetType,
                                          const Type* sourceUnadjustedType,
                                          ValueCategory valueCategory)
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

  auto sourceType = traits.remove_reference(sourceUnadjustedType);
  const bool sourceIsLvalue = valueCategory == ValueCategory::kLValue;
  const bool referenceCompatible =
      traits.is_reference_compatible(referencedType, sourceType);

  ImplicitConversionSequence seq;
  seq.sourceType = sourceUnadjustedType;
  seq.destinationType = targetType;
  seq.binding.isRvalueRef = isRvalueReference;

  if (!isRvalueReference && sourceIsLvalue && referenceCompatible)
    return directReferenceBinding(seq, ValueCategory::kLValue);

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
    return directReferenceBinding(seq, valueCategory);

  if (traits.is_reference_related(referencedType, sourceType)) {
    if (isRvalueReference && sourceIsLvalue) return seq;
    if (!is_at_least_as_cv_qualified(referencedCv, cv_qualifiers(sourceType)))
      return seq;
  }

  return std::nullopt;
}

auto StandardConversion::isDesignatedInitializerList(
    BracedInitListAST* bracedInitList) const -> bool {
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    if (ast_cast<DesignatedInitializerClauseAST>(it->value)) return true;
  }
  return false;
}

auto StandardConversion::classListInitializationSequence(
    ImplicitConversionSequence sequence, const Type* targetType)
    -> ImplicitConversionSequence {
  auto type = traits.remove_cvref(targetType);
  sequence.form = ConversionSequenceForm::kUserDefined;
  sequence.udc.aggregateInitializedClass = type;
  bindResultToReference(sequence, targetType, ValueCategory::kPrValue);
  return completedSequence(sequence, ImplicitCastKind::kIdentity, type);
}

auto StandardConversion::aggregateListInitializationSequence(
    ImplicitConversionSequence sequence, BracedInitListAST* initializer,
    const Type* targetType) -> ImplicitConversionSequence {
  auto conversion =
      aggregateListConversion(initializer, traits.remove_cvref(targetType));
  if (!conversion.viable) return sequence;
  sequence.list.narrowsElement = conversion.narrows;
  return classListInitializationSequence(sequence, targetType);
}

auto StandardConversion::listElementSequence(
    ExpressionAST* element, const Type* targetType,
    InitializationKind initializationKind) -> ImplicitConversionSequence {
  auto sequence =
      computeConversionSequence(element, targetType, initializationKind);
  sequence.list.isListInitialization = true;
  sequence.list.fromSingleElement = bool(sequence);
  sequence.list.elementCount = 1;
  sequence.list.narrowsElement = traits.is_narrowing_list_element(
      element, traits.remove_cvref(targetType));
  return sequence;
}

auto StandardConversion::listInitializationSequence(
    BracedInitListAST* bracedInitList, const Type* targetType,
    InitializationKind initializationKind) -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  seq.destinationType = targetType;
  seq.list.isListInitialization = true;

  auto listTarget = traits.remove_cv(traits.remove_reference(targetType));

  if (!traits.is_reference(targetType) &&
      isDesignatedInitializerList(bracedInitList)) {
    if (!traits.is_aggregate(listTarget)) return seq;
    return aggregateListInitializationSequence(seq, bracedInitList, targetType);
  }

  if (traits.is_class(listTarget) && traits.is_aggregate(listTarget)) {
    if (auto element = singleInitializerClause(bracedInitList);
        element && element->type) {
      auto elementType =
          traits.remove_cv(traits.remove_reference(element->type));
      if (traits.is_same(elementType, listTarget) ||
          traits.is_base_of(listTarget, elementType))
        return listElementSequence(element, targetType, initializationKind);
    }
  }

  if (auto stringInit = stringLiteralInitialization(
          traits, !isC_, listTarget, singleInitializerClause(bracedInitList));
      stringInit && stringInit->compatible) {
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return completedSequence(seq, ImplicitCastKind::kIdentity, listTarget);
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
    return completedSequence(seq, ImplicitCastKind::kIdentity, targetType);
  }

  if (traits.is_array(listTarget)) {
    if (traits.is_lvalue_reference(targetType) &&
        !traits.is_const(traits.remove_reference(targetType)))
      return seq;

    auto conversion = aggregateListConversion(bracedInitList, listTarget);
    if (!conversion.viable) return seq;

    seq.list.narrowsElement = conversion.narrows;
    seq.list.elementCount = conversion.elementCount;
    seq.list.elementRank = conversion.elementRank;
    seq.list.targetIsUnboundedArray = traits.is_unbounded_array(listTarget);
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return completedSequence(seq, ImplicitCastKind::kIdentity, listTarget);
  }

  if (traits.is_class(listTarget)) {
    if (traits.is_aggregate(listTarget))
      return aggregateListInitializationSequence(seq, bracedInitList,
                                                 targetType);
    if (!listInitializes(bracedInitList, listTarget, initializationKind))
      return seq;
    return classListInitializationSequence(seq, targetType);
  }

  if (auto element = singleInitializerClause(bracedInitList)) {
    if (ast_cast<BracedInitListAST>(element)) return seq;
    return listElementSequence(element, targetType, initializationKind);
  }

  if (!bracedInitList->expressionList) {
    bindResultToReference(seq, targetType, ValueCategory::kPrValue);
    return completedSequence(seq, ImplicitCastKind::kIdentity, listTarget);
  }

  return seq;
}

auto StandardConversion::overloadSetConversionSequence(ExpressionAST* expr,
                                                       const Type* targetType)
    -> std::optional<ImplicitConversionSequence> {
  ImplicitConversionSequence seq;
  seq.sourceType = expr->type;
  seq.destinationType = targetType;
  auto unreferencedSourceType = traits.remove_reference(expr->type);
  auto overloadSetType = type_cast<OverloadSetType>(unreferencedSourceType);
  auto sourceIsAddressOfOverloadSet = false;

  if (!overloadSetType) {
    if (auto sourcePointer = type_cast<PointerType>(unreferencedSourceType)) {
      overloadSetType =
          type_cast<OverloadSetType>(sourcePointer->elementType());
      sourceIsAddressOfOverloadSet = overloadSetType != nullptr;
    }
  }

  OverloadSetSymbol* overloadSet =
      overloadSetType ? overloadSetType->symbol() : nullptr;
  if (!overloadSet) {
    auto designator = stripNestedExpressions(expr);
    if (auto address = ast_cast<UnaryExpressionAST>(designator);
        address && address->op == TokenKind::T_AMP) {
      sourceIsAddressOfOverloadSet = true;
      designator = stripNestedExpressions(address->expression);
    }
    if (auto id = ast_cast<IdExpressionAST>(designator))
      overloadSet = symbol_cast<OverloadSetSymbol>(id->symbol);
    else if (auto member = ast_cast<MemberExpressionAST>(designator))
      overloadSet = symbol_cast<OverloadSetSymbol>(member->symbol);
  }
  if (!overloadSet) return std::nullopt;

  auto target = overloadSetTargetOf(unit_, targetType);
  if (!target) return std::nullopt;
  if (target->isMemberPointer && !sourceIsAddressOfOverloadSet) return seq;
  if (traits.is_reference(targetType) && sourceIsAddressOfOverloadSet)
    return seq;
  auto resolved = resolveOverloadSetAgainstTarget(unit_, overloadSet, *target,
                                                  expr->firstSourceLocation());
  if (!resolved) return seq;
  auto source = IdExpressionAST::create(arena_);
  source->type = resolved->type();
  source->valueCategory = ValueCategory::kLValue;
  if (sourceIsAddressOfOverloadSet) {
    source->type = traits.address_of_function(resolved);
    source->valueCategory = ValueCategory::kPrValue;
  }
  seq = computeConversionSequence(source, targetType,
                                  InitializationKind::kCopyInitialization,
                                  ConversionContext::kStandardOnly);
  seq.sourceType = expr->type;
  seq.resolvedFunction = resolved;
  return seq;
}

auto StandardConversion::pointerConversionSequence(
    ImplicitConversionSequence sequence, const Type* sourceType,
    const Type* targetType) -> std::optional<ImplicitConversionSequence> {
  auto source = type_cast<PointerType>(sourceType);
  auto target = unqualified_cast<PointerType>(targetType);
  if (!source || !target) return std::nullopt;
  auto sourceElement = source->elementType();
  auto targetElement = target->elementType();
  auto sourceCv = cv_qualifiers(sourceElement);
  auto targetCv = cv_qualifiers(targetElement);
  if (!is_at_least_as_cv_qualified(targetCv, sourceCv)) return sequence;
  auto sourceUnqualified = traits.remove_cv(sourceElement);
  auto targetUnqualified = traits.remove_cv(targetElement);
  sequence.pointeeUnqual = targetUnqualified;
  sequence.pointeeCv = targetCv;
  if (traits.is_qualification_convertible(sourceType,
                                          traits.remove_cv(targetType)))
    return completedSequence(
        sequence, ImplicitCastKind::kQualificationConversion, targetType);
  if (traits.is_void(targetUnqualified) && traits.is_object(sourceUnqualified))
    return completedSequence(sequence, ImplicitCastKind::kPointerConversion,
                             targetType);
  if (pointeeClassAdjustment(sourceType, targetType) ==
      ClassAdjustment::kDerivedToBase)
    return completedSequence(
        sequence, ImplicitCastKind::kDerivedToBaseConversion, targetType);
  if (traits.is_function(sourceElement) && traits.is_function(targetElement) &&
      traits.is_reference_compatible(targetElement, sourceElement))
    return completedSequence(
        sequence, ImplicitCastKind::kFunctionPointerConversion, targetType);
  if (!isC_) return sequence;
  if (traits.is_void(sourceUnqualified))
    return completedSequence(sequence, ImplicitCastKind::kPointerConversion,
                             targetType);
  if (traits.is_vla_compatible(sourceUnqualified, targetUnqualified))
    return completedSequence(sequence, ImplicitCastKind::kPointerConversion,
                             targetType);
  return sequence;
}

auto StandardConversion::memberPointerConversionSequence(
    ImplicitConversionSequence sequence, const Type* sourceType,
    const Type* targetType) -> std::optional<ImplicitConversionSequence> {
  auto source = decomposeMemberPointer(sourceType);
  auto target = decomposeMemberPointer(traits.remove_cv(targetType));
  if (!source || !target) return std::nullopt;
  if (!isMemberPointeeConvertible(source.pointeeType, target.pointeeType))
    return sequence;
  if (!traits.is_same(source.classType, target.classType)) {
    auto targetClass = type_cast<ClassType>(target.classType);
    auto sourceClass = type_cast<ClassType>(source.classType);
    if (!hasUniqueNonVirtualBase(targetClass, sourceClass)) return sequence;
    auto adjusted =
        control_->getMemberPointerType(target.classType, source.pointeeType);
    sequence.steps.push_back(
        {ImplicitCastKind::kPointerToMemberConversion, adjusted});
    if (traits.is_same(adjusted, traits.remove_cv(targetType))) {
      sequence.form = ConversionSequenceForm::kStandard;
      return sequence;
    }
  }
  auto kind = traits.is_function(source.pointeeType)
                  ? ImplicitCastKind::kFunctionPointerConversion
                  : ImplicitCastKind::kQualificationConversion;
  return completedSequence(sequence, kind, targetType);
}

auto StandardConversion::computeConversionSequenceSteps(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind, ConversionContext context)
    -> ImplicitConversionSequence {
  ImplicitConversionSequence seq;
  if (!expr || !targetType) return seq;

  seq.sourceType = expr->type;
  seq.destinationType = targetType;

  auto sourceObjectType = traits.remove_cvref(expr->type);
  auto targetObjectType = traits.remove_cvref(targetType);

  if (!traits.is_reference(targetType) ||
      !traits.is_same(sourceObjectType, targetObjectType)) {
    traits.requireCompleteClass(sourceObjectType);
    traits.requireCompleteClass(targetObjectType);
  }

  const Type* currentType = expr->type;
  ValueCategory currentValCat = expr->valueCategory;

  if (auto overload = overloadSetConversionSequence(expr, targetType))
    return *overload;

  if (auto bracedInitList = ast_cast<BracedInitListAST>(expr)) {
    return listInitializationSequence(bracedInitList, targetType,
                                      initializationKind);
  }

  if (auto referenceSequence =
          referenceBinding(targetType, expr->type, expr->valueCategory))
    return *referenceSequence;

  if (auto step = valueTransformation(expr)) {
    currentType = step->type;
    currentValCat = ValueCategory::kPrValue;
    seq.steps.push_back(*step);
  }

  auto comparisonTargetType = traits.remove_reference(targetType);

  if (currentValCat == ValueCategory::kPrValue &&
      traits.is_atomic(currentType) &&
      !traits.is_atomic(comparisonTargetType)) {
    currentType = traits.adjusted_cv_type(traits.remove_atomic(currentType));
    seq.steps.push_back({ImplicitCastKind::kAtomicToNonAtomic, currentType});
  }

  auto unqualFrom = traits.remove_cv(currentType);
  auto unqualTo = traits.remove_cv(comparisonTargetType);

  if (isDirectInitialization(initializationKind) &&
      traits.is_null_pointer(unqualFrom) &&
      traits.is_same(unqualTo, control_->getBoolType())) {
    return completedSequence(seq, ImplicitCastKind::kBooleanConversion,
                             comparisonTargetType);
  }

  if (traits.is_same(unqualFrom, unqualTo)) {
    seq.requiresCopyConstruction = requiresCopyConstruction(expr, targetType);
    seq.copyConstructor =
        selectCopyConstructor(expr, targetType, initializationKind);
    bindResultToReference(seq, targetType, currentValCat);
    return completedSequence(seq, ImplicitCastKind::kIdentity,
                             comparisonTargetType);
  }

  if (auto targetAtomic = type_cast<AtomicType>(unqualTo)) {
    auto valueSequence = computeConversionSequence(
        expr, targetAtomic->elementType(), initializationKind, context);
    if (valueSequence.form != ConversionSequenceForm::kStandard) return seq;
    valueSequence.destinationType = targetType;
    valueSequence.steps.push_back(
        {ImplicitCastKind::kNonAtomicToAtomic, comparisonTargetType});
    return valueSequence;
  }

  if (!traits.is_class_or_union(unqualFrom) &&
      !traits.is_class_or_union(unqualTo) &&
      (traits.is_complex(unqualFrom) || traits.is_complex(unqualTo))) {
    if (traits.is_complex(unqualFrom) && traits.is_complex(unqualTo)) {
      return completedSequence(seq, ImplicitCastKind::kComplexConversion,
                               comparisonTargetType);
    }

    if (traits.is_complex(unqualTo)) {
      if (!traits.is_arithmetic(unqualFrom)) return seq;
      return completedSequence(seq, ImplicitCastKind::kRealToComplexConversion,
                               comparisonTargetType);
    }

    if (traits.is_same(unqualTo, control_->getBoolType())) {
      return completedSequence(seq, ImplicitCastKind::kBooleanConversion,
                               comparisonTargetType);
    }

    if (!isC_ && context != ConversionContext::kStandardOnly) return seq;
    if (!traits.is_arithmetic(unqualTo)) return seq;
    return completedSequence(seq, ImplicitCastKind::kComplexToRealConversion,
                             comparisonTargetType);
  }

  if (auto targetVector = unqualified_cast<VectorType>(unqualTo)) {
    if (auto sourceVector = unqualified_cast<VectorType>(unqualFrom)) {
      const bool bothExt = sourceVector->vectorKind() == VectorKind::kExt &&
                           targetVector->vectorKind() == VectorKind::kExt;
      const auto width = traits.vector_width_in_bytes(sourceVector);
      if (!bothExt && width &&
          width == traits.vector_width_in_bytes(targetVector)) {
        return completedSequence(seq, ImplicitCastKind::kVectorConversion,
                                 comparisonTargetType);
      }
      return seq;
    }

    if (targetVector->vectorKind() == VectorKind::kExt &&
        canSplatIntoVector(expr, targetVector)) {
      return completedSequence(seq, ImplicitCastKind::kVectorSplat,
                               comparisonTargetType);
    }

    return seq;
  }

  if (unqualified_cast<VectorType>(unqualFrom)) return seq;

  if (!isC_ && classAdjustment(unqualFrom, unqualTo) ==
                   ClassAdjustment::kDerivedToBase) {
    bindResultToReference(seq, targetType, currentValCat);
    return completedSequence(seq, ImplicitCastKind::kDerivedToBaseConversion,
                             comparisonTargetType);
  }

  if (isNullPointerConstant(expr) && traits.is_null_pointer(unqualTo)) {
    return completedSequence(seq, ImplicitCastKind::kPointerConversion,
                             comparisonTargetType);
  }

  if (traits.is_null_pointer(unqualFrom) && traits.is_pointer(unqualTo)) {
    return completedSequence(seq, ImplicitCastKind::kPointerConversion,
                             comparisonTargetType);
  }

  if (traits.is_integral(unqualFrom) && traits.is_pointer(unqualTo) &&
      isNullPointerConstant(expr)) {
    return completedSequence(seq, ImplicitCastKind::kPointerConversion,
                             comparisonTargetType);
  }

  if (auto pointer =
          pointerConversionSequence(seq, unqualFrom, comparisonTargetType))
    return *pointer;

  if (traits.is_same(unqualTo, control_->getBoolType())) {
    if (traits.is_arithmetic_or_unscoped_enum(unqualFrom) ||
        traits.is_pointer(unqualFrom) || traits.is_member_pointer(unqualFrom)) {
      return completedSequence(seq, ImplicitCastKind::kBooleanConversion,
                               comparisonTargetType);
    }
  }

  if (auto kind = arithmeticConversionKind(unqualFrom, unqualTo))
    return completedSequence(seq, *kind, comparisonTargetType);

  if (auto member = memberPointerConversionSequence(seq, unqualFrom,
                                                    comparisonTargetType))
    return *member;

  if (traits.is_member_pointer(unqualTo) && isNullPointerConstant(expr)) {
    return completedSequence(seq, ImplicitCastKind::kPointerToMemberConversion,
                             comparisonTargetType);
  }

  if (isC_ && traits.is_integral_or_unscoped_enum(unqualFrom) &&
      traits.is_enum(unqualTo) && !traits.is_scoped_enum(unqualTo)) {
    return completedSequence(seq, ImplicitCastKind::kIntegralConversion,
                             comparisonTargetType);
  }

  if (context == ConversionContext::kStandardOnly) return seq;

  return userDefinedConversionSequence(expr, targetType, initializationKind);
}

auto StandardConversion::conversionFunctionResultType(
    FunctionSymbol* function, const Type* targetType) const -> const Type* {
  if (function->isConstructor()) return traits.remove_reference(targetType);
  auto type = type_cast<FunctionType>(function->type());
  return type ? type->returnType() : nullptr;
}

auto StandardConversion::userDefinedSequence(const Candidate& candidate,
                                             ExpressionAST* expr,
                                             const Type* targetType)
    -> ImplicitConversionSequence {
  ImplicitConversionSequence sequence;
  sequence.form = ConversionSequenceForm::kUserDefined;
  sequence.sourceType = expr->type;
  sequence.destinationType = targetType;
  sequence.udc.function = candidate.symbol;
  sequence.udc.secondRank = candidate.resultConversion->rank();
  sequence.udc.secondSteps = candidate.resultConversion->steps;
  sequence.copyConstructor = candidate.resultConversion->copyConstructor;
  sequence.requiresCopyConstruction =
      candidate.resultConversion->requiresCopyConstruction;
  if (!candidate.conversions.empty()) {
    const auto& first = candidate.conversions.front();
    sequence.udc.firstSteps = first.steps;
    sequence.resolvedFunction = first.resolvedFunction;
  }
  sequence.binding = candidate.resultConversion->binding;
  sequence.steps.push_back(
      {ImplicitCastKind::kUserDefinedConversion,
       conversionFunctionResultType(candidate.symbol, targetType)});
  if (!sequence.udc.secondSteps.empty())
    sequence.udc.secondTarget = traits.remove_reference(targetType);
  return sequence;
}

auto StandardConversion::objectArgumentConversion(FunctionSymbol* function,
                                                  ExpressionAST* expr)
    -> std::optional<ImplicitConversionSequence> {
  if (!function->isImplicitObjectMemberFunction()) return std::nullopt;
  OverloadResolution resolution{unit_};
  auto conversion = resolution.implicitObjectArgumentConversion(
      function,
      {.type = expr->type,
       .cv = cv_qualifiers(traits.remove_reference(expr->type)),
       .valueCategory = expr->valueCategory},
      nullptr);
  if (!conversion) return std::nullopt;
  return *conversion;
}

void StandardConversion::addUserDefinedCandidate(
    std::vector<Candidate>& candidates, FunctionSymbol* function,
    std::optional<ImplicitConversionSequence> argumentConversion,
    ImplicitConversionSequence resultConversion,
    std::optional<ImplicitConversionSequence> objectConversion) {
  if (std::ranges::contains(candidates, function, &Candidate::symbol)) return;
  Candidate candidate;
  candidate.symbol = function;
  candidate.objectConversion = std::move(objectConversion);
  if (argumentConversion)
    candidate.conversions.push_back(std::move(*argumentConversion));
  candidate.resultConversion = std::move(resultConversion);
  candidate.viable = true;
  candidate.fromTemplate = function->isSpecialization();
  candidates.push_back(std::move(candidate));
}

void StandardConversion::addConstructorCandidates(
    std::vector<Candidate>& candidates, ExpressionAST* expr,
    const Type* targetType) {
  auto lvalueRef = type_cast<LvalueReferenceType>(targetType);
  if (lvalueRef && !traits.is_const(lvalueRef->elementType())) return;
  auto classType = type_cast<ClassType>(traits.remove_cvref(targetType));
  if (!classType || !classType->symbol()) return;
  auto classSymbol = classType->symbol()->resolvedDefinition();
  traits.requireCompleteClass(classSymbol);
  for (auto constructor : classSymbol->convertingConstructors())
    addConstructorCandidate(candidates, constructor, classSymbol, expr,
                            targetType);
}

void StandardConversion::addConstructorCandidate(
    std::vector<Candidate>& candidates, FunctionSymbol* constructor,
    ClassSymbol* classSymbol, ExpressionAST* expr, const Type* targetType) {
  if (isExcludedInheritedConstructor(traits, constructor, classSymbol, 1))
    return;
  if (constructor->templateDeclaration() && !constructor->isSpecialization()) {
    auto args = make_list_node<ExpressionAST>(arena_, expr);
    TemplateArgumentDeduction deduction(unit_);
    auto deducedArgs = deduction.deduce(constructor, args, nullptr);
    if (!deducedArgs) return;
    constructor = ASTRewriter::instantiateOverloadCandidate(
        unit_, *deducedArgs, constructor, expr->firstSourceLocation(), false);
    if (!constructor) return;
  }
  auto functionType = type_cast<FunctionType>(constructor->type());
  if (!functionType || !is_callable_with_one_argument(constructor)) return;
  if (ASTRewriter::evaluateAssociatedConstraints(unit_, constructor) == false)
    return;
  auto argument =
      computeConversionSequence(expr, functionType->parameterTypes().front(),
                                InitializationKind::kCopyInitialization,
                                ConversionContext::kStandardOnly);
  if (!argument) return;
  auto resultType = traits.remove_reference(targetType);
  auto result = IdExpressionAST::create(arena_);
  result->type = resultType;
  result->valueCategory = ValueCategory::kPrValue;
  auto resultSequence = computeConversionSequence(
      result, targetType, InitializationKind::kCopyInitialization,
      ConversionContext::kStandardOnly);
  if (!resultSequence) return;
  addUserDefinedCandidate(candidates, constructor, std::move(argument),
                          std::move(resultSequence), std::nullopt);
}

void StandardConversion::addConversionFunctionCandidates(
    std::vector<Candidate>& candidates, ExpressionAST* expr,
    const Type* targetType, InitializationKind initializationKind) {
  auto classType = type_cast<ClassType>(traits.remove_cvref(expr->type));
  if (!classType || !classType->symbol()) return;
  auto classSymbol = classType->symbol()->resolvedDefinition();
  traits.requireCompleteClass(classSymbol);
  materializeClosureConversion(classSymbol, traits.remove_cvref(targetType));
  for (auto function : classSymbol->visibleConversionFunctions())
    addConversionFunctionCandidate(candidates, function, expr, targetType,
                                   initializationKind);
}

void StandardConversion::addConversionFunctionCandidate(
    std::vector<Candidate>& candidates, FunctionSymbol* function,
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind) {
  if (function->isExplicit() && !isDirectInitialization(initializationKind))
    return;
  if (function->templateDeclaration() && !function->isSpecialization()) {
    function = instantiateConversionFunctionTemplate(
        function, traits.remove_cvref(targetType), expr);
    if (!function) return;
  }
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return;
  if (ASTRewriter::evaluateAssociatedConstraints(unit_, function) == false)
    return;
  auto objectConversion = objectArgumentConversion(function, expr);
  if (function->isImplicitObjectMemberFunction() && !objectConversion) return;
  auto returnType = functionType->returnType();
  if (!returnType) return;
  auto resultType = traits.remove_reference(returnType);
  if (traits.is_rvalue_reference(targetType) &&
      traits.is_lvalue_reference(returnType) && !traits.is_function(resultType))
    return;
  if (function->isExplicit() &&
      !traits.is_qualification_convertible(traits.remove_cv(resultType),
                                           traits.remove_cvref(targetType)))
    return;
  auto result = IdExpressionAST::create(arena_);
  result->type = resultType;
  result->valueCategory = conversionResultValueCategory(returnType);
  auto second = computeConversionSequence(
      result, targetType, InitializationKind::kCopyInitialization,
      ConversionContext::kStandardOnly);
  if (!second) return;
  addUserDefinedCandidate(candidates, function, std::nullopt, std::move(second),
                          std::move(objectConversion));
}

auto StandardConversion::userDefinedConversionSequence(
    ExpressionAST* expr, const Type* targetType,
    InitializationKind initializationKind) -> ImplicitConversionSequence {
  std::vector<Candidate> candidates;
  addConstructorCandidates(candidates, expr, targetType);
  addConversionFunctionCandidates(candidates, expr, targetType,
                                  initializationKind);
  OverloadResolution resolution{unit_};
  auto best = resolution.selectBestViableFunction(candidates, true);
  if (!best.best) return {};
  auto sequence = userDefinedSequence(*best.best, expr, targetType);
  if (best.ambiguous) sequence.form = ConversionSequenceForm::kAmbiguous;
  return sequence;
}

void StandardConversion::materializeClosureConversion(ClassSymbol* srcClass,
                                                      const Type* targetType) {
  auto pointerType = unqualified_cast<PointerType>(targetType);
  if (!pointerType) return;

  auto functionType =
      type_cast<FunctionType>(traits.remove_cv(pointerType->elementType()));
  if (!functionType) return;

  Binder binder{unit_};
  (void)binder.materializeClosureFunctionPointerConversion(srcClass,
                                                           functionType);
}

auto StandardConversion::convertedClassSymbol(const Type* type) const
    -> ClassSymbol* {
  if (!type) return nullptr;
  type = traits.remove_cvref(type);
  if (auto pointerType = unqualified_cast<PointerType>(type))
    type = traits.remove_cv(pointerType->elementType());
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return nullptr;
  return classType->symbol();
}

auto StandardConversion::accessingScope() const -> ScopeSymbol* {
  if (accessingScope_) return accessingScope_;
  return unit_->globalScope();
}

auto checkBaseClassConversion(TranslationUnit* unit,
                              ScopeSymbol* accessingScope, ClassSymbol* derived,
                              ClassSymbol* base, SourceLocation loc) -> bool {
  if (derived == base) return true;
  if (isDependent(unit, derived->type())) return true;

  if (!derived->baseSubobjectInfo(base).isUniqueSubobject()) {
    unit->error(
        loc, std::format("'{}' is an ambiguous base class of '{}'",
                         to_string(base->type()), to_string(derived->type())));
    return false;
  }

  AccessContext accessContext{unit, accessingScope};
  if (accessContext.isAccessibleBaseClass(derived, base)) return true;

  unit->error(loc,
              std::format("'{}' is an inaccessible base class of '{}'",
                          to_string(base->type()), to_string(derived->type())));
  return false;
}

auto StandardConversion::baseConversionClasses(const Type* sourceType,
                                               const Type* targetType,
                                               ImplicitCastKind kind) const
    -> std::pair<ClassSymbol*, ClassSymbol*> {
  if (kind == ImplicitCastKind::kDerivedToBaseConversion)
    return {convertedClassSymbol(sourceType), convertedClassSymbol(targetType)};
  if (kind != ImplicitCastKind::kPointerToMemberConversion) return {};
  auto source = decomposeMemberPointer(traits.remove_cv(sourceType));
  auto target = decomposeMemberPointer(traits.remove_cv(targetType));
  if (!source || !target) return {};
  if (traits.is_same(source.classType, target.classType)) return {};
  return {convertedClassSymbol(target.classType),
          convertedClassSymbol(source.classType)};
}

auto StandardConversion::isAccessibleBaseConversion(const Type* sourceType,
                                                    const Type* targetType,
                                                    ImplicitCastKind kind) const
    -> bool {
  auto [derived, base] = baseConversionClasses(sourceType, targetType, kind);
  if (!derived || !base) return true;
  derived = derived->resolvedDefinition();
  base = base->resolvedDefinition();
  if (!derived->baseSubobjectInfo(base).isUniqueSubobject()) return false;
  return AccessContext{unit_, accessingScope()}.isAccessibleBaseClass(derived,
                                                                      base);
}

auto StandardConversion::isAccessible(
    const ImplicitConversionSequence& sequence) -> bool {
  if (!sequence || sequence.form == ConversionSequenceForm::kAmbiguous)
    return false;
  if (sequence.requiresCopyConstruction && !sequence.copyConstructor)
    return false;
  AccessContext access{unit_, accessingScope()};
  for (auto function : {sequence.udc.function, sequence.copyConstructor}) {
    if (!function) continue;
    if (function->isDeleted()) return false;
    auto declaringClass = declaringClassOf(function);
    if (!declaringClass) continue;
    auto sourceClass = convertedClassSymbol(sequence.sourceType);
    auto designatingClass = declaringClass;
    if (!function->isConstructor() && sourceClass)
      designatingClass = sourceClass;
    if (!access.isAccessible(function, designatingClass, sourceClass))
      return false;
  }
  auto sourceType = sequence.sourceType;
  if (sequence.resolvedFunction)
    sourceType = traits.address_of_function(sequence.resolvedFunction);
  for (const auto* steps :
       {&sequence.udc.firstSteps, &sequence.steps, &sequence.udc.secondSteps}) {
    for (const auto& step : *steps) {
      if (!isAccessibleBaseConversion(sourceType, step.type, step.kind))
        return false;
      sourceType = step.type;
    }
  }
  return true;
}

void StandardConversion::checkBaseConversion(const Type* sourceType,
                                             const Type* targetType,
                                             ImplicitCastKind kind,
                                             SourceLocation loc) {
  auto [derived, base] = baseConversionClasses(sourceType, targetType, kind);
  if (!derived || !base) return;
  (void)checkBaseClassConversion(unit_, accessingScope(),
                                 derived->resolvedDefinition(),
                                 base->resolvedDefinition(), loc);
}

void StandardConversion::checkUserDefinedConversionAccess(
    const ImplicitConversionSequence& sequence, ExpressionAST* expr) {
  auto function = sequence.udc.function;
  if (!function) return;
  if (!function->name()) return;
  if (function->name()->kind() != NameKind::kConversionFunctionId) return;

  auto declaringClass = declaringClassOf(function);
  if (!declaringClass) return;

  auto sourceClass = convertedClassSymbol(expr->type);
  auto designatingClass = sourceClass ? sourceClass : declaringClass;

  (void)checkMemberAccess(unit_, accessingScope(), function, designatingClass,
                          sourceClass, expr->firstSourceLocation());
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

  checkBaseConversion(expr->type, step.type, step.kind,
                      expr->firstSourceLocation());
  if (step.kind == ImplicitCastKind::kUserDefinedConversion) {
    checkUserDefinedConversionAccess(sequence, expr);
  }

  wrapWithImplicitCast(step.kind, step.type, expr);

  if (step.kind != ImplicitCastKind::kUserDefinedConversion) return;
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr))
    recordUserDefinedConversion(cast, sequence.udc.function,
                                !sequence.udc.firstSteps.empty());
}

void StandardConversion::applyConversionSequence(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  if (!sequence) return;

  if (sequence.resolvedFunction)
    applyResolvedFunction(expr, sequence.resolvedFunction);

  if (sequence.list.fromSingleElement) {
    if (auto braced = ast_cast<BracedInitListAST>(expr);
        braced && braced->expressionList && !braced->expressionList->next) {
      expr = braced->expressionList->value;
    }
  }

  for (const auto& step : sequence.udc.firstSteps)
    applyStep(sequence, step, expr);

  for (const auto& step : sequence.steps) applyStep(sequence, step, expr);
  for (const auto& step : sequence.udc.secondSteps)
    applyStep(sequence, step, expr);

  if (sequence.copyConstructor) applyCopyConstruction(sequence, expr);

  if (sequence.binding.bindsToTemporary())
    ASTRewriter::requireDestructorOfType(unit_,
                                         sequence.binding.referencedType);

  ASTRewriter::requireDestructorOfType(unit_, sequence.destinationType);

  requireDefinitionOfDesignatedField(expr);
  requireNamedFunction(expr);
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
  if (sequence.resolvedFunction)
    applyResolvedFunction(cast->expression, sequence.resolvedFunction);
  for (const auto& step : sequence.udc.firstSteps)
    applyStep(sequence, step, cast->expression);
  recordUserDefinedConversion(cast, function, !sequence.udc.firstSteps.empty());
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

auto StandardConversion::selectCopyConstructor(
    ExpressionAST* expr, const Type* destinationType,
    InitializationKind initializationKind) -> FunctionSymbol* {
  if (!requiresCopyConstruction(expr, destinationType)) return nullptr;

  auto classType = unqualified_cast<ClassType>(destinationType);
  auto classSymbol = classType->symbol()->resolvedDefinition();

  if (!control_->beginCopyConstructorSelection(classSymbol)) return nullptr;

  OverloadResolution resolution(unit_);
  auto resolved =
      resolution.resolveConstructor(classSymbol, {expr}, initializationKind);

  control_->endCopyConstructorSelection(classSymbol);

  return resolved.selected();
}

void StandardConversion::materializeConstructorArguments(
    ImplicitCastExpressionAST* cast, FunctionSymbol* constructor,
    bool argumentConverted) {
  if (ast_cast<ParenInitializerAST>(cast->expression)) return;

  auto functionType = type_cast<FunctionType>(constructor->type());
  if (!functionType) return;
  auto parameterTypes = functionType->parameterTypes();
  if (parameterTypes.empty()) return;

  auto arguments = make_list_node<ExpressionAST>(arena_, cast->expression);

  if (!argumentConverted) {
    auto sequence =
        computeConversionSequence(arguments->value, parameterTypes[0]);
    applyConversionSequence(sequence, arguments->value);
  }

  appendDefaultArguments(constructor, &arguments, cast->firstSourceLocation());

  auto paren = ParenInitializerAST::create(
      arena_, cast->firstSourceLocation(), arguments,
      cast->firstSourceLocation(), ValueCategory::kPrValue, cast->type);
  cast->expression = paren;
}

auto StandardConversion::aggregateListConversion(
    BracedInitListAST* bracedInitList, const Type* aggregateType)
    -> AggregateListConversion {
  rankedAggregates_.push_back(traits.remove_cv(aggregateType));
  auto result = rankAggregateElements(bracedInitList, aggregateType);
  rankedAggregates_.pop_back();
  return result;
}

auto StandardConversion::rankAggregateElements(
    BracedInitListAST* bracedInitList, const Type* aggregateType)
    -> AggregateListConversion {
  AggregateListConversion result;

  auto plan = planAggregateInitialization(unit_, aggregateType, bracedInitList);
  if (!plan || !plan->valid) return result;

  for (const auto& initialized : plan->initializedElements) {
    if (!initialized.type || !initialized.initializer) return result;

    if (initialized.elided &&
        std::ranges::contains(rankedAggregates_,
                              traits.remove_cv(initialized.type)))
      return result;

    if (traits.is_array(initialized.type) &&
        ast_cast<StringLiteralExpressionAST>(initialized.initializer))
      continue;

    auto elementSeq =
        computeConversionSequence(initialized.initializer, initialized.type);
    if (!elementSeq) return result;

    if (elementSeq.list.narrowsElement ||
        traits.is_narrowing_list_element(initialized.initializer,
                                         initialized.type))
      result.narrows = true;

    result.elementRank = std::min(result.elementRank, elementSeq.rank());
  }

  result.elementCount = plan->initializedElements.size();
  result.viable = true;
  return result;
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

  OverloadResolution resolution(unit_);
  return bool(resolution.selectListConstructor(classSymbol, bracedInitList,
                                               elements, initializationKind));
}

void StandardConversion::appendDefaultArguments(FunctionSymbol* function,
                                                List<ExpressionAST*>** list,
                                                SourceLocation location) {
  auto params = function->parameters();
  if (params.empty() || !list) return;

  auto tail = list;
  std::size_t argCount = 0;
  while (*tail) {
    tail = &(*tail)->next;
    ++argCount;
  }

  for (auto i = argCount; i < params.size(); ++i) {
    auto defaultArgument =
        ASTRewriter::requireDefaultArgument(unit_, params[i]);
    if (!defaultArgument) break;
    *tail =
        make_list_node<ExpressionAST>(arena_, defaultArgument->clone(arena_));
    auto sequence =
        computeConversionSequence((*tail)->value, params[i]->type());
    applyConversionSequence(sequence, (*tail)->value);
    (*tail)->value = makeDefaultInitializer(unit_, (*tail)->value, location,
                                            accessingScope_);
    tail = &(*tail)->next;
  }
}

auto StandardConversion::namedEntity(ExpressionAST* expr) -> Symbol* {
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

  if (auto id = ast_cast<IdExpressionAST>(expr)) return id->symbol;
  return nullptr;
}

void StandardConversion::requireDefinitionOfDesignatedField(
    ExpressionAST* expr) {
  ASTRewriter::requireFieldDefinition(
      unit_, symbol_cast<FieldSymbol>(namedEntity(expr)));
}

void StandardConversion::requireNamedFunction(ExpressionAST* expr) {
  ASTRewriter::requireFunctionDefinition(
      unit_, symbol_cast<FunctionSymbol>(namedEntity(expr)));
}

void StandardConversion::recordUserDefinedConversion(
    ImplicitCastExpressionAST* cast, FunctionSymbol* function,
    bool argumentConverted) {
  if (!function) return;

  cast->conversionFunction = function;
  ASTRewriter::requireFunctionDefinition(unit_, function);

  if (function->isConstructor()) {
    materializeConstructorArguments(cast, function, argumentConverted);
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
                                         const Type* targetType)
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
                                                const Type* targetType)
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
                                                   const Type* targetType)
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
  if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
    setResolvedFunction(nested->expression, function);
    nested->type = nested->expression->type;
    nested->valueCategory = nested->expression->valueCategory;
    return;
  }
  if (auto idExpr = ast_cast<IdExpressionAST>(expr)) {
    idExpr->symbol = function;
    idExpr->type = function->type();
  } else if (auto memberExpr = ast_cast<MemberExpressionAST>(expr)) {
    memberExpr->symbol = function;
    memberExpr->type = function->type();
  }
}

void StandardConversion::applyResolvedFunction(ExpressionAST* expr,
                                               FunctionSymbol* function) {
  if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
    applyResolvedFunction(nested->expression, function);
    nested->type = nested->expression->type;
    nested->valueCategory = nested->expression->valueCategory;
    return;
  }
  auto stripped = stripNestedExpressions(expr);
  auto addressOf = ast_cast<UnaryExpressionAST>(stripped);
  if (addressOf && addressOf->op == TokenKind::T_AMP) {
    setResolvedFunction(addressOf->expression, function);
    addressOf->type = traits.address_of_function(function);
    return;
  }
  setResolvedFunction(stripped, function);
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
  if (castKind == ImplicitCastKind::kFunctionPointerConversion &&
      traits.is_function(type))
    cast->valueCategory = expr->valueCategory;

  if (castKind == ImplicitCastKind::kUserDefinedConversion) {
    if (auto lvalueRef = type_cast<LvalueReferenceType>(type)) {
      cast->type = lvalueRef->elementType();
      cast->valueCategory = ValueCategory::kLValue;
    } else if (auto rvalueRef = type_cast<RvalueReferenceType>(type)) {
      cast->type = rvalueRef->elementType();
      cast->valueCategory = conversionResultValueCategory(type);
    }
  }

  if (isClassAdjustment(castKind) && !traits.is_pointer(type) &&
      is_glvalue(expr)) {
    cast->valueCategory = expr->valueCategory;
  }

  expr = cast;
  foldConstantRead(expr);
}

}  // namespace cxx
