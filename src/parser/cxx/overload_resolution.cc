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
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <span>

namespace cxx {
namespace {
struct CandidateRejectionPrinter {
  [[nodiscard]] auto operator()(const std::string& reason) const
      -> std::string {
    return reason;
  }

  [[nodiscard]] auto operator()(const ArgumentCountMismatch& reason) const
      -> std::string {
    return std::format(
        "requires {} argument{}, but {} {} provided", reason.parameterCount,
        reason.parameterCount == 1 ? "" : "s", reason.argumentCount,
        reason.argumentCount == 1 ? "was" : "were");
  }

  [[nodiscard]] auto operator()(const FailedArgumentConversion& reason) const
      -> std::string {
    return std::format("no known conversion from '{}' to '{}' for argument {}",
                       to_string(reason.source), to_string(reason.destination),
                       reason.argumentIndex + 1);
  }
};

[[nodiscard]] auto callArgumentCount(const Candidate& candidate)
    -> std::size_t {
  auto count = candidate.conversions.size();
  if (candidate.symbol->hasImplicitObjectParameter()) ++count;
  return count;
}

[[nodiscard]] auto functionTemplateHasPackParameter(FunctionSymbol* pattern)
    -> bool {
  auto type = type_cast<FunctionType>(pattern->type());
  if (!type) return false;
  return std::ranges::any_of(type->parameterTypes(), [](const Type* param) {
    return type_cast<PackExpansionType>(param) != nullptr;
  });
}

[[nodiscard]] auto hasClassOrEnumerationOperand(
    const TypeTraits& traits, std::span<const Type* const> operandTypes)
    -> bool {
  return std::ranges::any_of(operandTypes, [&](const Type* type) {
    auto operandType = traits.remove_cvref(type);
    return traits.is_class(operandType) || traits.is_enum(operandType);
  });
}

[[nodiscard]] auto isOverloadableOnlyByMember(TokenKind op) -> bool {
  switch (op) {
    case TokenKind::T_EQUAL:
    case TokenKind::T_LBRACKET:
    case TokenKind::T_MINUS_GREATER:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto isParameterForEnumerationOperand(const TypeTraits& traits,
                                                    const Type* parameterType,
                                                    const Type* operandType)
    -> bool {
  if (!operandType) return false;
  auto enumeration = traits.remove_cvref(operandType);
  if (!traits.is_enum(enumeration)) return false;
  return traits.is_same(traits.remove_cvref(parameterType), enumeration);
}

[[nodiscard]] auto acceptsEnumerationOperands(
    const TypeTraits& traits, std::span<const Type* const> parameterTypes,
    const Type* leftType, const Type* rightType) -> bool {
  if (!parameterTypes.empty() &&
      isParameterForEnumerationOperand(traits, parameterTypes[0], leftType))
    return true;
  if (parameterTypes.size() > 1 &&
      isParameterForEnumerationOperand(traits, parameterTypes[1], rightType))
    return true;
  return false;
}

[[nodiscard]] auto hasClassOperand(const TypeTraits& traits,
                                   const Type* leftType, const Type* rightType)
    -> bool {
  if (traits.is_class(traits.remove_cvref(leftType))) return true;
  return rightType && traits.is_class(traits.remove_cvref(rightType));
}
}  // namespace

auto to_string(const CandidateRejection& rejection) -> std::string {
  return std::visit(CandidateRejectionPrinter{}, rejection);
}

auto compareNonTemplateConstraints(TranslationUnit* unit,
                                   FunctionSymbol* candidate,
                                   FunctionSymbol* other) -> int {
  if (ASTRewriter::isMorePartialOrderingConstrained(unit, candidate, other))
    return 1;
  if (ASTRewriter::isMorePartialOrderingConstrained(unit, other, candidate))
    return -1;
  return 0;
}

namespace {
[[nodiscard]] auto orderedFunction(FunctionSymbol* function)
    -> FunctionSymbol* {
  if (auto origin = function->inheritedConstructorOrigin()) return origin;
  return function;
}
}  // namespace

auto compareCandidateOrdering(TranslationUnit* unit, FunctionSymbol* candidate,
                              bool candidateFromTemplate, FunctionSymbol* other,
                              bool otherFromTemplate, bool preferNonTemplate,
                              const PartialOrderingContext& context) -> int {
  candidate = orderedFunction(candidate);
  other = orderedFunction(other);

  if (preferNonTemplate && candidateFromTemplate != otherFromTemplate)
    return candidateFromTemplate ? -1 : 1;

  if (candidateFromTemplate && otherFromTemplate)
    return compareFunctionTemplateSpecializations(unit, candidate, other,
                                                  context);

  if (!candidateFromTemplate && !otherFromTemplate)
    return compareNonTemplateConstraints(unit, candidate, other);

  return 0;
}

auto templateCandidateArityRejects(FunctionSymbol* pattern, int argCount)
    -> bool {
  auto type = type_cast<FunctionType>(pattern->type());
  if (!type) return false;
  if (type->isVariadic()) return false;
  if (functionTemplateHasPackParameter(pattern)) return false;

  auto params = type->parameterTypes();
  auto paramCount = static_cast<int>(params.size());
  if (argCount > paramCount) return true;
  if (argCount < paramCount &&
      argCount < required_parameter_count(pattern, paramCount)) {
    return true;
  }
  return false;
}

OverloadResolution::OverloadResolution(TranslationUnit* unit)
    : unit_(unit),
      traits(unit),
      control_(unit->control()),
      arena_(unit->arena()),
      stdconv_(unit) {}

using ReferenceBinding = ImplicitConversionSequence::ReferenceBinding;

auto OverloadResolution::implicitObjectParameterClass(
    FunctionSymbol* function, const ImplicitObjectArgument& object,
    ScopeSymbol* lookupScope) -> const Type* {
  if (name_cast<ConversionFunctionId>(function->name()))
    return traits.remove_cvref(object.type);
  if (auto nominatingClass = symbol_cast<ClassSymbol>(lookupScope))
    return nominatingClass->type();
  if (auto classSymbol = symbol_cast<ClassSymbol>(function->parent()))
    return classSymbol->type();
  return traits.remove_cvref(object.type);
}

auto OverloadResolution::implicitObjectArgumentConversion(
    FunctionSymbol* function, const ImplicitObjectArgument& object,
    ScopeSymbol* lookupScope)
    -> std::expected<ImplicitConversionSequence, std::string> {
  ImplicitConversionSequence conversion;
  conversion.form = ConversionSequenceForm::kStandard;
  conversion.sourceType = object.type;
  conversion.destinationType = object.type;
  conversion.steps.push_back({ImplicitCastKind::kIdentity, object.type});

  if (!function->isImplicitObjectMemberFunction()) {
    conversion.isStaticMemberObjectParameter = function->isStatic() &&
                                               function->parent() &&
                                               function->parent()->isClass();
    return conversion;
  }

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return conversion;

  const auto functionCv = functionType->cvQualifiers();
  const auto functionRef = functionType->refQualifier();
  if (!is_at_least_as_cv_qualified(functionCv, object.cv)) {
    return std::unexpected(std::format(
        "'this' argument has type '{}', but function is not "
        "marked {}",
        to_string(object.type), has_const(object.cv) ? "const" : "volatile"));
  }

  const bool objectIsLvalue = object.valueCategory == ValueCategory::kLValue;

  if (functionRef == RefQualifier::kRvalue && objectIsLvalue) {
    return std::unexpected(
        "expects an rvalue for the implicit object argument");
  }

  if (functionRef == RefQualifier::kLvalue && !objectIsLvalue &&
      !(has_const(functionCv) && !has_volatile(functionCv))) {
    return std::unexpected(
        "expects an lvalue for the implicit object argument");
  }

  auto implicitObjectClass =
      implicitObjectParameterClass(function, object, lookupScope);

  conversion.binding.kind = objectIsLvalue
                                ? ReferenceBinding::Kind::kDirectToLvalue
                                : ReferenceBinding::Kind::kDirectToXvalue;
  conversion.binding.isDirect = true;
  conversion.binding.referencedType =
      traits.add_cv(implicitObjectClass, functionCv);
  conversion.binding.cv = functionCv;
  conversion.destinationType =
      control_->getLvalueReferenceType(conversion.binding.referencedType);
  conversion.binding.isRvalueRef = functionRef == RefQualifier::kRvalue;
  conversion.binding.isUnqualifiedImplicitObjectParameter =
      functionRef == RefQualifier::kNone;

  if (stdconv_.classAdjustment(object.type, implicitObjectClass) ==
      ClassAdjustment::kDerivedToBase) {
    conversion.steps.push_back({ImplicitCastKind::kDerivedToBaseConversion,
                                conversion.binding.referencedType});
  }

  return conversion;
}

auto nonObjectParameterTypes(FunctionSymbol* function)
    -> std::span<const Type* const> {
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return {};
  std::span<const Type* const> parameterTypes{functionType->parameterTypes()};
  if (function->hasExplicitObjectParameter() && !parameterTypes.empty())
    parameterTypes = parameterTypes.subspan(1);
  return parameterTypes;
}

auto haveSameParameterTypes(FunctionSymbol* lhs, FunctionSymbol* rhs) -> bool {
  auto lhsType = type_cast<FunctionType>(lhs->type());
  auto rhsType = type_cast<FunctionType>(rhs->type());
  if (!lhsType || !rhsType) return false;
  if (lhsType->isVariadic() != rhsType->isVariadic()) return false;
  return std::ranges::equal(nonObjectParameterTypes(lhs),
                            nonObjectParameterTypes(rhs));
}

namespace {

[[nodiscard]] auto isInheritedConstructorCandidate(const Candidate& candidate)
    -> bool {
  if (candidate.deduction.fromInheritedConstructor) return true;
  return candidate.symbol->inheritedConstructorOrigin() != nullptr;
}

}  // namespace

auto OverloadResolution::haveSameParametersForArguments(const Candidate& lhs,
                                                        const Candidate& rhs)
    -> bool {
  auto lhsType = type_cast<FunctionType>(lhs.symbol->type());
  auto rhsType = type_cast<FunctionType>(rhs.symbol->type());
  if (!lhsType || !rhsType) return false;
  auto lhsParameters = nonObjectParameterTypes(lhs.symbol);
  auto rhsParameters = nonObjectParameterTypes(rhs.symbol);
  for (std::size_t index = 0; index < lhs.conversions.size(); ++index) {
    const auto lhsEllipsis = index >= lhsParameters.size();
    const auto rhsEllipsis = index >= rhsParameters.size();
    if (lhsEllipsis != rhsEllipsis) return false;
    if (lhsEllipsis) continue;
    if (!traits.is_same(lhsParameters[index], rhsParameters[index]))
      return false;
  }
  return true;
}

auto OverloadResolution::compareInheritedConstructors(const Candidate& lhs,
                                                      const Candidate& rhs)
    -> int {
  const auto lhsInherited = isInheritedConstructorCandidate(lhs);
  if (lhsInherited == isInheritedConstructorCandidate(rhs)) return 0;
  if (!haveSameParametersForArguments(lhs, rhs)) return 0;
  return lhsInherited ? -1 : 1;
}

auto compareDeductionCandidates(const DeductionCandidateInfo& lhs,
                                const DeductionCandidateInfo& rhs) -> int {
  if (lhs.fromDeductionGuide != rhs.fromDeductionGuide)
    return lhs.fromDeductionGuide ? 1 : -1;

  if (lhs.isCopyDeductionCandidate != rhs.isCopyDeductionCandidate)
    return lhs.isCopyDeductionCandidate ? 1 : -1;

  if (lhs.fromConstructorTemplate != rhs.fromConstructorTemplate)
    return lhs.fromConstructorTemplate ? -1 : 1;

  return 0;
}

auto OverloadResolution::selectBestViableFunction(
    std::vector<Candidate>& candidates, bool preferNonTemplate)
    -> OverloadResult {
  if (candidates.empty()) return {};

  std::vector<Candidate*> best;
  best.push_back(&candidates[0]);

  for (size_t i = 1; i < candidates.size(); ++i) {
    auto& curr = candidates[i];
    auto& ref = *best[0];

    bool currBetter = false;
    bool refBetter = false;

    if (curr.objectConversion && ref.objectConversion) {
      if (curr.objectConversion->isBetterThan(*ref.objectConversion, traits))
        currBetter = true;
      if (ref.objectConversion->isBetterThan(*curr.objectConversion, traits))
        refBetter = true;
    }

    auto n = std::min(curr.conversions.size(), ref.conversions.size());
    for (size_t j = 0; j < n; ++j) {
      if (curr.conversions[j].isBetterThan(ref.conversions[j], traits))
        currBetter = true;
      if (ref.conversions[j].isBetterThan(curr.conversions[j], traits))
        refBetter = true;
    }

    if (currBetter == refBetter && curr.resultConversion &&
        ref.resultConversion) {
      currBetter =
          curr.resultConversion->isBetterThan(*ref.resultConversion, traits);
      refBetter =
          ref.resultConversion->isBetterThan(*curr.resultConversion, traits);
    }

    if (currBetter && !refBetter) {
      best.clear();
      best.push_back(&curr);
    } else if (refBetter && !currBetter) {
    } else if (int order = compareCandidateOrdering(
                   unit_, curr.symbol, curr.fromTemplate, ref.symbol,
                   ref.fromTemplate, preferNonTemplate,
                   PartialOrderingContext::call(callArgumentCount(curr)));
               order != 0) {
      if (order > 0) {
        best.clear();
        best.push_back(&curr);
      }
    } else if (int order = compareInheritedConstructors(curr, ref);
               order != 0) {
      if (order > 0) {
        best.clear();
        best.push_back(&curr);
      }
    } else if (int order =
                   compareDeductionCandidates(curr.deduction, ref.deduction);
               order != 0) {
      if (order > 0) {
        best.clear();
        best.push_back(&curr);
      }
    } else {
      best.push_back(&curr);
    }
  }

  if (best.empty()) return {};
  if (best.size() > 1) return {best[0], true};

  ASTRewriter::instantiateSelectedSpecializationDefinition(
      unit_, best[0]->symbol, best[0]->deducedTemplateArgs);
  return {best[0], false};
}

auto isExcludedInheritedConstructor(TypeTraits& traits,
                                    FunctionSymbol* constructor,
                                    ClassSymbol* classSymbol, int argCount)
    -> bool {
  if (argCount != 1) return false;

  auto inherited = constructor->inheritedConstructorOrigin();
  if (!inherited) return false;

  auto base = symbol_cast<ClassSymbol>(inherited->parent());
  if (!base) return false;

  auto type = type_cast<FunctionType>(constructor->type());
  if (!type || type->parameterTypes().empty()) return false;

  auto firstParameter = type->parameterTypes().front();
  if (!traits.is_reference(firstParameter)) return false;

  auto referenced = traits.remove_reference(firstParameter);

  return traits.is_reference_related(base->type(), referenced) &&
         traits.is_reference_related(referenced, classSymbol->type());
}

auto OverloadResolution::resolveConstructor(
    ClassSymbol* classSymbol, const std::vector<ExpressionAST*>& args,
    InitializationKind initializationKind) -> ConstructorResult {
  return resolveConstructor(classSymbol, args, initializationKind, false);
}

auto OverloadResolution::hasDefaultConstructor(ClassSymbol* classSymbol)
    -> bool {
  return bool(resolveConstructor(classSymbol, {},
                                 InitializationKind::kDirectInitialization));
}

auto OverloadResolution::selectListConstructor(
    ClassSymbol* classSymbol, BracedInitListAST* bracedInitList,
    const std::vector<ExpressionAST*>& elements,
    InitializationKind initializationKind) -> ConstructorResult {
  const auto listInitializationKind = asListInitialization(initializationKind);

  const bool emptyListSelectsDefaultConstructor =
      elements.empty() && hasDefaultConstructor(classSymbol);

  if (!emptyListSelectsDefaultConstructor) {
    std::vector<ExpressionAST*> wholeList = {bracedInitList};
    auto result = resolveConstructor(classSymbol, wholeList,
                                     listInitializationKind, true);
    if (result.best) {
      result.fromInitializerListConstructor = true;
      return result;
    }
  }

  return resolveConstructor(classSymbol, elements, listInitializationKind,
                            false);
}

auto OverloadResolution::resolveConstructor(
    ClassSymbol* classSymbol, const std::vector<ExpressionAST*>& args,
    InitializationKind initializationKind, bool initializerListConstructorsOnly)
    -> ConstructorResult {
  ConstructorResult result;

  auto argCount = static_cast<int>(args.size());

  const bool excludesExplicitConstructors =
      initializationKind == InitializationKind::kCopyInitialization;

  const auto constructors = classSymbol->constructors();

  auto reject = [&](FunctionSymbol* ctor, CandidateRejection reason) {
    result.rejected.push_back({ctor, std::move(reason)});
  };

  auto rejectArity = [&](FunctionSymbol* ctor, int paramCount) {
    reject(ctor, ArgumentCountMismatch{paramCount, argCount});
  };

  auto bindsReferenceToInitializedClass = [&](const Type* parameterType) {
    auto referencedType = traits.remove_reference(parameterType);
    if (referencedType == parameterType) return false;
    return traits.is_same(traits.remove_cv(referencedType),
                          classSymbol->type());
  };

  auto isInitializerListConstructor = [&](FunctionSymbol* ctor) {
    if (!is_callable_with_one_argument(ctor)) return false;
    auto type = type_cast<FunctionType>(ctor->type());
    return traits.initializer_list_element_type(
               type->parameterTypes().front()) != nullptr;
  };

  for (auto ctor : constructors) {
    if (ctor->canonical() != ctor) continue;
    if (ctor->isSpecialization()) continue;
    if (excludesExplicitConstructors && ctor->isExplicit()) continue;
    if (initializerListConstructorsOnly &&
        !isInitializerListConstructor(ctor)) {
      continue;
    }

    const bool templateCandidate =
        ctor->templateDeclaration() != nullptr && !ctor->isSpecialization();
    List<TemplateArgumentAST*>* deducedArgsForCandidate = nullptr;

    if (templateCandidate) {
      if (templateCandidateArityRejects(ctor, argCount)) {
        auto templateType = type_cast<FunctionType>(ctor->type());
        rejectArity(
            ctor, templateType
                      ? static_cast<int>(templateType->parameterTypes().size())
                      : argCount);
        continue;
      }

      const auto loc = args.empty() ? classSymbol->location()
                                    : args.front()->firstSourceLocation();

      auto deduced = deduceTemplateCandidate(ctor, args, loc);
      if (!deduced) {
        reject(ctor, std::move(deduced.error()));
        continue;
      }

      ctor = deduced->specialization;
      deducedArgsForCandidate = deduced->templateArguments;

      if (excludesExplicitConstructors) {
        if (ctor->isExplicit()) continue;
      }
    }

    auto type = type_cast<FunctionType>(ctor->type());
    if (!type) continue;

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) {
      rejectArity(ctor, paramCount);
      continue;
    }
    if (argCount < paramCount) {
      if (argCount < required_parameter_count(ctor, paramCount)) {
        rejectArity(ctor, paramCount);
        continue;
      }
    }

    if (ASTRewriter::evaluateAssociatedConstraints(unit_, ctor) == false) {
      reject(ctor, "constraints not satisfied");
      continue;
    }

    if (auto owner = symbol_cast<ClassSymbol>(ctor->parent());
        owner && owner->resolvedDefinition() != classSymbol) {
      Binder binder{unit_};
      binder.setReportErrors(!unit_->diagnosticsClient()->isSfinae());
      auto thunk = binder.inheritedConstructorFor(classSymbol, ctor);
      if (!thunk) continue;
      ctor = thunk;
      type = type_cast<FunctionType>(ctor->type());
      if (!type) continue;
    }

    if (isExcludedInheritedConstructor(traits, ctor, classSymbol, argCount)) {
      reject(ctor, "inherited constructor is excluded by a derived signature");
      continue;
    }

    Candidate cand{ctor};
    cand.viable = true;
    cand.fromTemplate = templateCandidate;
    cand.deducedTemplateArgs = deducedArgsForCandidate;

    auto paramIt = type->parameterTypes().begin();
    auto paramEnd = type->parameterTypes().end();
    for (size_t i = 0; i < args.size() && paramIt != paramEnd; ++i, ++paramIt) {
      const auto convertsFirstCopyInitializationArgument =
          excludesExplicitConstructors && i == 0;

      const auto admitsExplicitConversionFunctions =
          i == 0 && args.size() == 1 &&
          isDirectInitialization(initializationKind) &&
          bindsReferenceToInitializedClass(*paramIt);

      auto conv = stdconv_.computeConversionSequence(
          args[i], *paramIt,
          admitsExplicitConversionFunctions
              ? InitializationKind::kDirectInitialization
              : InitializationKind::kCopyInitialization,
          convertsFirstCopyInitializationArgument
              ? ConversionContext::kStandardOnly
              : ConversionContext::kImplicit);
      if (!conv) {
        cand.viable = false;
        reject(ctor, FailedArgumentConversion{args[i]->type, *paramIt, i});
        break;
      }
      cand.conversions.push_back(conv);
    }

    if (cand.viable && type->isVariadic()) {
      for (int i = paramCount; i < argCount; ++i) {
        ImplicitConversionSequence ellipsisConv;
        ellipsisConv.form = ConversionSequenceForm::kEllipsis;
        cand.conversions.push_back(ellipsisConv);
      }
    }

    if (cand.viable) result.candidates.push_back(std::move(cand));
  }

  auto [bestPtr, ambiguous] =
      selectBestViableFunction(result.candidates, /*preferNonTemplate=*/true);
  result.best = bestPtr;
  result.ambiguous = ambiguous;

  if (!result.best) {
    result.failure = ConstructorSelectionFailure::kNoViableConstructor;
  } else if (ambiguous) {
    result.failure = ConstructorSelectionFailure::kAmbiguous;
  } else if (initializationKind ==
                 InitializationKind::kCopyListInitialization &&
             result.best->symbol->isExplicit()) {
    result.failure = ConstructorSelectionFailure::kExplicitInCopyInitialization;
  }

  return result;
}

auto OverloadResolution::computeImplicitConversionSequence(
    ExpressionAST* expr, const Type* targetType) -> ImplicitConversionSequence {
  return stdconv_.computeConversionSequence(expr, targetType);
}

void OverloadResolution::applyImplicitConversion(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  stdconv_.applyConversionSequence(sequence, expr);
}

auto OverloadResolution::findCandidates(ScopeSymbol* scope,
                                        const Name* name) const
    -> FoundCandidates {
  FoundCandidates result;

  if (!scope || !name) return result;

  auto symbol = qualifiedLookup(scope, name);
  if (!symbol) return result;

  result.lookupScope = symbol->parent();
  addLookupCandidates(result.functions, symbol);
  return result;
}

auto OverloadResolution::buildCallCandidate(
    FunctionSymbol* function, const FunctionType* type,
    std::span<ExpressionAST* const> args,
    std::vector<RejectedCandidate>* rejected) -> std::optional<Candidate> {
  const auto argCount = static_cast<int>(args.size());
  const auto paramCount = static_cast<int>(type->parameterTypes().size());

  auto reject = [&](CandidateRejection reason) {
    if (rejected) rejected->push_back({function, std::move(reason)});
  };

  auto rejectArity = [&] {
    reject(ArgumentCountMismatch{paramCount, argCount});
  };

  if (argCount > paramCount && !type->isVariadic()) {
    rejectArity();
    return std::nullopt;
  }

  if (argCount < paramCount &&
      argCount < required_parameter_count(function, paramCount)) {
    rejectArity();
    return std::nullopt;
  }

  if (ASTRewriter::evaluateAssociatedConstraints(unit_, function) == false) {
    reject("constraints not satisfied");
    return std::nullopt;
  }

  Candidate cand{function};
  cand.viable = true;

  auto paramIt = type->parameterTypes().begin();
  auto paramEnd = type->parameterTypes().end();
  for (int i = 0; i < argCount && paramIt != paramEnd; ++i, ++paramIt) {
    auto conv = computeImplicitConversionSequence(args[i], *paramIt);
    if (!conv) {
      reject(FailedArgumentConversion{args[i]->type, *paramIt,
                                      static_cast<std::size_t>(i)});
      return std::nullopt;
    }
    cand.conversions.push_back(conv);
  }

  if (type->isVariadic()) {
    for (int i = paramCount; i < argCount; ++i) {
      ImplicitConversionSequence ellipsisConv;
      ellipsisConv.form = ConversionSequenceForm::kEllipsis;
      cand.conversions.push_back(ellipsisConv);
    }
  }

  return cand;
}

auto OverloadResolution::deduceTemplateCandidate(
    FunctionSymbol* pattern, std::span<ExpressionAST* const> args,
    SourceLocation location) -> std::expected<DeducedCandidate, std::string> {
  List<ExpressionAST*>* expressionList = nullptr;
  auto tail = &expressionList;
  for (auto arg : args) {
    *tail = make_list_node(arena_, arg);
    tail = &(*tail)->next;
  }

  TemplateArgumentDeduction deduction(unit_);
  auto deducedArgs = deduction.deduce(pattern, expressionList,
                                      /*explicitTemplateArguments=*/{});
  if (!deducedArgs.has_value())
    return std::unexpected("template argument deduction failed");

  auto specialization = ASTRewriter::instantiateOverloadCandidate(
      unit_, *deducedArgs, pattern, location, /*argsComplete=*/true);
  if (!specialization)
    return std::unexpected("substitution failed for the deduced arguments");

  return DeducedCandidate{specialization, *deducedArgs};
}

auto OverloadResolution::collectCandidates(Symbol* symbol) const
    -> std::vector<FunctionSymbol*> {
  auto functions = views::each_function(symbol);
  return {functions.begin(), functions.end()};
}

auto OverloadResolution::builtinBinaryOperatorParameterType(
    TokenKind op, const Type* leftType, const Type* rightType) -> const Type* {
  if (!leftType || !rightType) return nullptr;

  const auto isComparison = [op] {
    switch (op) {
      case TokenKind::T_EQUAL_EQUAL:
      case TokenKind::T_EXCLAIM_EQUAL:
      case TokenKind::T_LESS:
      case TokenKind::T_LESS_EQUAL:
      case TokenKind::T_GREATER:
      case TokenKind::T_GREATER_EQUAL:
      case TokenKind::T_LESS_EQUAL_GREATER:
        return true;
      default:
        return false;
    }
  }();

  const auto isArithmetic = [op] {
    switch (op) {
      case TokenKind::T_PLUS:
      case TokenKind::T_MINUS:
      case TokenKind::T_STAR:
      case TokenKind::T_SLASH:
      case TokenKind::T_PERCENT:
      case TokenKind::T_AMP:
      case TokenKind::T_BAR:
      case TokenKind::T_CARET:
        return true;
      default:
        return false;
    }
  }();

  if (!isComparison && !isArithmetic) return nullptr;

  auto left = traits.remove_cvref(leftType);
  auto right = traits.remove_cvref(rightType);

  if (isComparison && traits.is_enum(left) && traits.is_same(left, right))
    return left;

  const auto isArithmeticOperand = [&](const Type* type) {
    return traits.is_arithmetic(type) ||
           (traits.is_enum(type) && !traits.is_scoped_enum(type));
  };

  if (!isArithmeticOperand(left) || !isArithmeticOperand(right)) return nullptr;

  return stdconv_.commonArithmeticType(left, right);
}

auto OverloadResolution::builtinUnaryOperatorParameterTypes(
    TokenKind op, const Type* operandType) -> std::vector<const Type*> {
  switch (op) {
    case TokenKind::T_EXCLAIM:
      return {control_->getBoolType()};
    case TokenKind::T_STAR:
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_TILDE:
    case TokenKind::T_PLUS_PLUS:
    case TokenKind::T_MINUS_MINUS:
      break;
    default:
      return {};
  }

  std::vector<const Type*> types;
  auto classType = type_cast<ClassType>(traits.remove_cvref(operandType));
  if (!classType) return types;
  auto classSymbol = classType->definition();
  traits.requireCompleteClass(classSymbol);
  const auto modifies =
      op == TokenKind::T_PLUS_PLUS || op == TokenKind::T_MINUS_MINUS;

  for (auto function : classSymbol->visibleConversionFunctions()) {
    if (function->isExplicit()) continue;
    auto functionType = type_cast<FunctionType>(function->type());
    if (!functionType) continue;
    auto result = functionType->returnType();
    auto type = traits.remove_cvref(result);
    if (!traits.is_pointer(type)) continue;
    if (op == TokenKind::T_MINUS || op == TokenKind::T_TILDE) continue;
    auto pointee = traits.remove_pointer(type);
    if (op == TokenKind::T_STAR && traits.is_void(pointee)) continue;
    if (modifies) {
      if (!traits.is_object(pointee)) continue;
      if (!traits.is_lvalue_reference(result)) continue;
      if (traits.is_const(traits.remove_reference(result))) continue;
      type = result;
    }
    if (std::ranges::find(types, type) == types.end()) types.push_back(type);
  }

  if (op == TokenKind::T_STAR) return types;

  for (auto type : traits.arithmetic_types()) {
    if (traits.is_same(type, control_->getBoolType())) continue;
    if (modifies) {
      types.push_back(traits.add_lvalue_reference(type));
      types.push_back(traits.add_lvalue_reference(traits.add_volatile(type)));
      continue;
    }
    if (op == TokenKind::T_TILDE && !traits.is_integral(type)) continue;
    if (traits.is_integral(type) && traits.promoted_integer_type(type) != type)
      continue;
    types.push_back(type);
  }
  return types;
}

auto OverloadResolution::resolveBinaryOperator(
    TokenKind op, const std::vector<BinaryOperatorCandidate>& candidates,
    const Type* leftType, const Type* rightType, bool* ambiguous,
    ExpressionAST* leftExpr, ExpressionAST* rightExpr,
    ImplicitConversionSequence* builtinConversion) -> FunctionSymbol* {
  if (builtinConversion) *builtinConversion = {};
  if (ambiguous) *ambiguous = false;

  struct ViableCandidate {
    FunctionSymbol* symbol;
    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;
    List<TemplateArgumentAST*>* deducedTemplateArgs = nullptr;
    bool rewritten = false;
    bool reversed = false;
  };

  auto remove_cvref = [&](const Type* type) {
    if (!type) return type;
    return traits.remove_cvref(type);
  };

  auto candidateBetterThan = [&](const ViableCandidate& lhs,
                                 const ViableCandidate& rhs) -> bool {
    bool lhsBetter = false;

    if (lhs.left.isBetterThan(rhs.left, traits)) {
      lhsBetter = true;
    } else if (rhs.left.isBetterThan(lhs.left, traits)) {
      return false;
    }

    if (lhs.right.has_value() != rhs.right.has_value()) return false;

    if (lhs.right) {
      if (lhs.right->isBetterThan(*rhs.right, traits)) {
        lhsBetter = true;
      } else if (rhs.right->isBetterThan(*lhs.right, traits)) {
        return false;
      }
    }

    if (lhsBetter) return true;
    if (lhs.rewritten != rhs.rewritten) return !lhs.rewritten;
    if (lhs.rewritten && lhs.reversed != rhs.reversed) return !lhs.reversed;
    return false;
  };

  std::vector<ViableCandidate> viable;

  const auto operandsHaveClassType =
      hasClassOperand(traits, leftType, rightType);

  for (auto operatorCandidate : candidates) {
    auto candidate = operatorCandidate.symbol;
    auto candidateLeftType = operatorCandidate.reversed ? rightType : leftType;
    auto candidateRightType = operatorCandidate.reversed ? leftType : rightType;
    auto candidateLeftExpr = operatorCandidate.reversed ? rightExpr : leftExpr;
    auto candidateRightExpr = operatorCandidate.reversed ? leftExpr : rightExpr;

    if (!candidateLeftExpr) continue;
    if (candidateRightType && !candidateRightExpr) continue;

    bool isMember = candidate->isImplicitObjectMemberFunction();
    List<TemplateArgumentAST*>* deducedArgsForCandidate = nullptr;

    if (candidate->templateDeclaration() && !candidate->isSpecialization()) {
      int operandCount = rightExpr ? (isMember ? 1 : 2) : (isMember ? 0 : 1);
      if (templateCandidateArityRejects(candidate, operandCount)) continue;

      std::vector<ExpressionAST*> arguments;
      if (!isMember) arguments.push_back(candidateLeftExpr);
      if (candidateRightExpr) arguments.push_back(candidateRightExpr);

      auto deduced = deduceTemplateCandidate(
          candidate, arguments, candidateLeftExpr->firstSourceLocation());
      if (!deduced) continue;

      candidate = deduced->specialization;
      deducedArgsForCandidate = deduced->templateArguments;
    }

    bool alreadyViable = false;
    for (const auto& v : viable) {
      if (v.symbol == candidate && v.rewritten == operatorCandidate.rewritten &&
          v.reversed == operatorCandidate.reversed) {
        alreadyViable = true;
        break;
      }
    }
    if (alreadyViable) continue;

    auto funcType = type_cast<FunctionType>(candidate->type());
    if (!funcType) continue;

    if (ASTRewriter::evaluateAssociatedConstraints(unit_, candidate) == false)
      continue;

    auto params = funcType->parameterTypes();

    const bool isFilteredByEnumerationOperands =
        !isMember && !operandsHaveClassType;
    if (isFilteredByEnumerationOperands &&
        !acceptsEnumerationOperands(traits, params, candidateLeftType,
                                    candidateRightType))
      continue;

    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;

    if (candidateRightExpr) {
      if (isMember) {
        if (params.size() != 1) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !traits.is_base_of(classType, remove_cvref(candidateLeftType))) {
          continue;
        }
        auto objectConversion = implicitObjectArgumentConversion(
            candidate,
            {.type = candidateLeftType,
             .cv = cv_qualifiers(traits.remove_reference(candidateLeftType)),
             .valueCategory = candidateLeftExpr->valueCategory},
            operatorCandidate.lookupScope);
        if (!objectConversion) continue;
        left = *objectConversion;
        right =
            stdconv_.computeConversionSequence(candidateRightExpr, params[0]);
      } else {
        if (params.size() != 2) continue;
        left = stdconv_.computeConversionSequence(candidateLeftExpr, params[0]);
        if (!left) continue;
        right =
            stdconv_.computeConversionSequence(candidateRightExpr, params[1]);
      }
    } else {
      if (isMember) {
        if (!params.empty()) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !traits.is_base_of(classType, remove_cvref(candidateLeftType))) {
          continue;
        }
        auto objectConversion = implicitObjectArgumentConversion(
            candidate,
            {.type = candidateLeftType,
             .cv = cv_qualifiers(traits.remove_reference(candidateLeftType)),
             .valueCategory = candidateLeftExpr->valueCategory},
            operatorCandidate.lookupScope);
        if (!objectConversion) continue;
        left = *objectConversion;
      } else {
        if (params.size() != 1) continue;
        left = stdconv_.computeConversionSequence(candidateLeftExpr, params[0]);
      }
    }

    if (!left) continue;
    if (candidateRightExpr && (!right || !*right)) continue;

    if (operatorCandidate.reversed && right) std::swap(left, *right);
    viable.push_back({candidate, left, right, deducedArgsForCandidate,
                      operatorCandidate.rewritten, operatorCandidate.reversed});
  }

  if (!rightExpr && leftExpr) {
    for (auto parameterType :
         builtinUnaryOperatorParameterTypes(op, leftType)) {
      auto initializationKind = InitializationKind::kCopyInitialization;
      if (op == TokenKind::T_EXCLAIM)
        initializationKind = InitializationKind::kDirectInitialization;
      auto conversion = stdconv_.computeConversionSequence(
          leftExpr, parameterType, initializationKind);
      if (conversion) viable.push_back({nullptr, conversion, std::nullopt});
    }
  }

  if (viable.empty()) return nullptr;

  if (rightExpr) {
    if (auto parameterType =
            builtinBinaryOperatorParameterType(op, leftType, rightType)) {
      auto left = stdconv_.computeConversionSequence(leftExpr, parameterType);
      auto right = stdconv_.computeConversionSequence(rightExpr, parameterType);
      if (left && right)
        viable.push_back({nullptr, left, right, nullptr, false, false});
    }
  }

  auto best = &viable[0];
  bool foundEquivalent = false;

  for (size_t i = 1; i < viable.size(); ++i) {
    if (candidateBetterThan(viable[i], *best)) {
      best = &viable[i];
      foundEquivalent = false;
      continue;
    }

    if (candidateBetterThan(*best, viable[i])) {
      continue;
    }

    if (!viable[i].symbol || !best->symbol) {
      foundEquivalent = true;
      continue;
    }

    auto order = compareCandidateOrdering(
        unit_, viable[i].symbol, viable[i].symbol->isSpecialization(),
        best->symbol, best->symbol->isSpecialization(),
        /*preferNonTemplate=*/true,
        PartialOrderingContext::call(rightExpr ? 2 : 1, viable[i].reversed,
                                     best->reversed));
    if (order > 0) {
      best = &viable[i];
      foundEquivalent = false;
      continue;
    }
    if (order < 0) continue;

    foundEquivalent = true;
  }

  if (foundEquivalent) {
    if (ambiguous) *ambiguous = true;
    return nullptr;
  }

  if (!best->symbol) {
    if (best->left.form == ConversionSequenceForm::kAmbiguous) {
      if (ambiguous) *ambiguous = true;
      return nullptr;
    }
    if (builtinConversion) *builtinConversion = best->left;
    return nullptr;
  }

  ASTRewriter::instantiateSelectedSpecializationDefinition(
      unit_, best->symbol, best->deducedTemplateArgs);
  lastOperatorRewritten_ = best->rewritten;
  lastOperatorReversed_ = best->reversed;
  return best->symbol;
}

auto OverloadResolution::isRewriteTarget(FunctionSymbol* equalityOperator,
                                         const Type* firstOperandType) -> bool {
  if (!equalityOperator) return false;

  auto equalityType = type_cast<FunctionType>(equalityOperator->type());
  if (!equalityType) return false;

  ScopeSymbol* searchScope = nullptr;

  if (equalityOperator->parent() && equalityOperator->parent()->isClass()) {
    auto classType =
        type_cast<ClassType>(traits.remove_cvref(firstOperandType));
    searchScope = classType ? classType->symbol() : nullptr;
  } else {
    searchScope = equalityOperator->enclosingNamespace();
  }

  if (!searchScope) return true;

  auto notEqualName = control_->getOperatorId(TokenKind::T_EXCLAIM_EQUAL);
  if (!notEqualName) return true;

  for (auto candidate : findCandidates(searchScope, notEqualName).functions) {
    if (isExcludedCandidate(candidate)) continue;
    auto candidateType = type_cast<FunctionType>(candidate->type());
    if (!candidateType) continue;

    if (candidateType->parameterTypes() != equalityType->parameterTypes())
      continue;
    if (candidateType->cvQualifiers() != equalityType->cvQualifiers()) continue;
    if (candidateType->refQualifier() != equalityType->refQualifier()) continue;
    if (candidate->parent() != equalityOperator->parent()) continue;
    if (!TemplateEquivalence{unit_}.same(
            candidate->trailingRequiresClause(),
            equalityOperator->trailingRequiresClause()))
      continue;

    return false;
  }

  return true;
}

auto OverloadResolution::nonMemberOperatorCandidates(
    ScopeSymbol* scope, const Name* name,
    std::span<const Type* const> operandTypes) const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> candidates;

  addLookupCandidates(candidates,
                      unqualifiedNonMemberLookup(control_, scope, name));

  for (auto function : argumentDependentLookup(unit_, name, operandTypes))
    addOverloadCandidate(candidates, function);

  return candidates;
}

auto OverloadResolution::isExcludedCandidate(FunctionSymbol* function) const
    -> bool {
  if (!excludedCandidate_) return false;
  return function->canonical() == excludedCandidate_->canonical();
}

auto OverloadResolution::lookupOperator(
    ScopeSymbol* scope, const Type* type, TokenKind op, const Type* rightType,
    ExpressionAST* leftExpr, ExpressionAST* rightExpr,
    ImplicitConversionSequence* builtinConversion) -> FunctionSymbol* {
  lastLookupAmbiguous_ = false;
  lastOperatorRewritten_ = false;
  lastOperatorReversed_ = false;

  std::vector<const Type*> operandTypes{type};
  if (rightType) operandTypes.push_back(rightType);

  if (!hasClassOrEnumerationOperand(traits, operandTypes)) return nullptr;

  auto name = control_->getOperatorId(op);
  if (!name) return nullptr;

  std::vector<BinaryOperatorCandidate> candidates;

  auto addCandidate = [&](FunctionSymbol* function, ScopeSymbol* lookupScope,
                          bool rewritten, bool reversed) {
    if (isExcludedCandidate(function)) return;
    BinaryOperatorCandidate candidate{function, lookupScope, rewritten,
                                      reversed};
    for (const auto& existing : candidates) {
      if (existing.symbol == function && existing.rewritten == rewritten &&
          existing.reversed == reversed)
        return;
    }
    candidates.push_back(candidate);
  };

  auto addMemberCandidates = [&](const Type* operandType,
                                 const Name* operatorName, bool rewritten,
                                 bool reversed) {
    auto classType = type_cast<ClassType>(traits.remove_cvref(operandType));
    if (!classType) return;
    if (auto classSymbol = classType->symbol()) {
      traits.requireCompleteClass(classSymbol);
      auto found = findCandidates(classSymbol, operatorName);
      for (auto function : found.functions)
        addCandidate(function, found.lookupScope, rewritten, reversed);
    }
  };

  addMemberCandidates(type, name, false, false);

  if (!isOverloadableOnlyByMember(op)) {
    for (auto function : nonMemberOperatorCandidates(scope, name, operandTypes))
      addCandidate(function, nullptr, false, false);
  }

  auto addRewrittenCandidates = [&](TokenKind rewrittenOp,
                                    const Type* firstOperandType,
                                    const Type* secondOperandType,
                                    bool reversed) {
    auto rewrittenName = control_->getOperatorId(rewrittenOp);
    if (!rewrittenName) return;

    const bool requiresRewriteTarget = rewrittenOp == TokenKind::T_EQUAL_EQUAL;

    auto accept = [&](FunctionSymbol* function, ScopeSymbol* lookupScope) {
      if (requiresRewriteTarget && !isRewriteTarget(function, firstOperandType))
        return;
      addCandidate(function, lookupScope, true, reversed);
    };

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(firstOperandType))) {
      if (auto classSymbol = classType->symbol()) {
        traits.requireCompleteClass(classSymbol);
        auto found = findCandidates(classSymbol, rewrittenName);
        for (auto function : found.functions)
          accept(function, found.lookupScope);
      }
    }

    const Type* rewrittenOperandTypes[] = {firstOperandType, secondOperandType};
    for (auto function : nonMemberOperatorCandidates(scope, rewrittenName,
                                                     rewrittenOperandTypes))
      accept(function, nullptr);
  };

  const bool isRelational =
      op == TokenKind::T_LESS || op == TokenKind::T_LESS_EQUAL ||
      op == TokenKind::T_GREATER || op == TokenKind::T_GREATER_EQUAL;
  const bool isThreeWay = op == TokenKind::T_LESS_EQUAL_GREATER;
  const bool isEquality =
      op == TokenKind::T_EQUAL_EQUAL || op == TokenKind::T_EXCLAIM_EQUAL;

  if (rightType) {
    if (isRelational) {
      addRewrittenCandidates(TokenKind::T_LESS_EQUAL_GREATER, type, rightType,
                             false);
    }

    if (isRelational || isThreeWay) {
      addRewrittenCandidates(TokenKind::T_LESS_EQUAL_GREATER, rightType, type,
                             true);
    }

    if (op == TokenKind::T_EXCLAIM_EQUAL) {
      addRewrittenCandidates(TokenKind::T_EQUAL_EQUAL, type, rightType, false);
    }

    if (isEquality) {
      addRewrittenCandidates(TokenKind::T_EQUAL_EQUAL, rightType, type, true);
    }
  }

  bool ambiguous = false;
  auto selected =
      resolveBinaryOperator(op, candidates, type, rightType, &ambiguous,
                            leftExpr, rightExpr, builtinConversion);
  lastLookupAmbiguous_ = ambiguous;
  return selected;
}
}  // namespace cxx
