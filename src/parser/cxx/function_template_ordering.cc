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
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_deduction.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>

namespace cxx {

namespace {

[[nodiscard]] auto primaryTemplateOf(FunctionSymbol* function)
    -> FunctionSymbol* {
  if (!function->isSpecialization()) return function;
  auto primary = function->primaryTemplateSymbol();
  return primary ? primary : function;
}

[[nodiscard]] auto isConversionFunction(FunctionSymbol* function) -> bool {
  return name_cast<ConversionFunctionId>(function->name()) != nullptr;
}

[[nodiscard]] auto hasEquivalentTemplateSignature(TranslationUnit* unit,
                                                  FunctionSymbol* a,
                                                  FunctionSymbol* b) -> bool {
  auto primaryA = primaryTemplateOf(a);
  auto primaryB = primaryTemplateOf(b);
  auto templateA = primaryA->templateDeclaration();
  auto templateB = primaryB->templateDeclaration();
  if (!templateA || !templateB) return false;
  if (!TemplateEquivalence{unit}.sameForOrdering(
          templateA->templateParameterList, templateB->templateParameterList))
    return false;

  auto functionTypeA = type_cast<FunctionType>(primaryA->type());
  auto functionTypeB = type_cast<FunctionType>(primaryB->type());
  if (!functionTypeA || !functionTypeB) return false;
  if (functionTypeA->isVariadic() != functionTypeB->isVariadic()) return false;

  auto paramsA = functionTypeA->parameterTypes();
  auto paramsB = functionTypeB->parameterTypes();
  if (paramsA.size() != paramsB.size()) return false;

  for (std::size_t i = 0; i < paramsA.size(); ++i) {
    if (!TemplateEquivalence{unit}.sameForOrdering(paramsA[i], paramsB[i],
                                                   templateA, templateB))
      return false;
  }

  const bool conversionA = isConversionFunction(primaryA);
  const bool conversionB = isConversionFunction(primaryB);
  if (conversionA != conversionB) return false;
  if (conversionA) {
    auto specializationTypeA = type_cast<FunctionType>(a->type());
    auto specializationTypeB = type_cast<FunctionType>(b->type());
    if (!specializationTypeA || !specializationTypeB ||
        !unit->typeTraits().is_same(specializationTypeA->returnType(),
                                    specializationTypeB->returnType()))
      return false;
  }

  return true;
}

[[nodiscard]] auto compareByConstraints(TranslationUnit* unit,
                                        FunctionSymbol* a, FunctionSymbol* b)
    -> int {
  auto primaryA = primaryTemplateOf(a);
  auto primaryB = primaryTemplateOf(b);
  if (!hasEquivalentTemplateSignature(unit, a, b)) return 0;

  if (ASTRewriter::isMoreConstrained(unit, primaryA, primaryB)) return 1;
  if (ASTRewriter::isMoreConstrained(unit, primaryB, primaryA)) return -1;

  return 0;
}

struct OrderingType {
  const Type* type = nullptr;
  bool isPack = false;
  bool matchesAnyObject = false;
};

[[nodiscard]] auto orderingType(const Type* parameterType) -> OrderingType {
  if (auto expansion = type_cast<PackExpansionType>(parameterType))
    return {.type = expansion->pattern(), .isPack = true};
  return {.type = parameterType};
}

struct TransformedTemplate {
  FunctionSymbol* primary = nullptr;
  TemplateDeclarationAST* templateDeclaration = nullptr;
  std::vector<OrderingType> types;
  std::size_t parameterCount = 0;
  bool hasTrailingPack = false;
  std::size_t trailingPackIndex = 0;
};

struct ReferenceAdjustment {
  const Type* type = nullptr;
  bool wasReference = false;
  bool wasLvalueReference = false;
  CvQualifiers referencedCv = CvQualifiers::kNone;
};

class FunctionTemplateOrdering {
 public:
  FunctionTemplateOrdering(TranslationUnit* unit,
                           const PartialOrderingContext& context)
      : unit_(unit), traits_(unit), context_(context) {}

  [[nodiscard]] auto compare(FunctionSymbol* candidate, FunctionSymbol* other)
      -> int;

 private:
  [[nodiscard]] auto objectParameterType(FunctionSymbol* member,
                                         const Type* otherFirstParameter) const
      -> const Type*;

  [[nodiscard]] auto firstExplicitParameter(FunctionSymbol* primary) const
      -> const Type*;

  [[nodiscard]] auto transform(FunctionSymbol* function, bool reversed,
                               FunctionSymbol* other) const
      -> std::optional<TransformedTemplate>;

  [[nodiscard]] auto orderingKind(FunctionSymbol* candidate,
                                  FunctionSymbol* other) const
      -> PartialOrderingContext::Kind;

  [[nodiscard]] auto usedTypes(std::vector<OrderingType> types) const
      -> std::vector<OrderingType>;

  [[nodiscard]] auto adjust(const Type* type) const -> ReferenceAdjustment;

  [[nodiscard]] auto isAtLeastAsSpecialized(
      const TransformedTemplate& argument,
      const TransformedTemplate& parameter) const -> bool;

  [[nodiscard]] auto deducesIdentically(const TransformedTemplate& argument,
                                        const TransformedTemplate& parameter,
                                        const Type* argumentType,
                                        const Type* parameterType) const
      -> bool;

  [[nodiscard]] auto losesByReferenceBinding(
      const TransformedTemplate& argument, const TransformedTemplate& parameter,
      const ReferenceAdjustment& A, const ReferenceAdjustment& P) const -> bool;

  [[nodiscard]] auto deducesUsedParameters(
      const TypeDeduction& deduction,
      const std::vector<OrderingType>& parameterTypes) const -> bool;

  [[nodiscard]] static auto winsByTrailingPack(const TransformedTemplate& lhs,
                                               const TransformedTemplate& rhs)
      -> bool;

  TranslationUnit* unit_ = nullptr;
  TypeTraits traits_;
  PartialOrderingContext context_;
};

auto FunctionTemplateOrdering::firstExplicitParameter(
    FunctionSymbol* primary) const -> const Type* {
  if (primary->hasImplicitObjectParameter()) return nullptr;
  auto functionType = type_cast<FunctionType>(primary->type());
  if (!functionType || functionType->parameterTypes().empty()) return nullptr;
  return functionType->parameterTypes().front();
}

auto FunctionTemplateOrdering::objectParameterType(
    FunctionSymbol* member, const Type* otherFirstParameter) const
    -> const Type* {
  auto functionType = type_cast<FunctionType>(member->type());
  auto classSymbol = symbol_cast<ClassSymbol>(member->parent());
  auto objectType =
      traits_.add_cv(classSymbol->type(), functionType->cvQualifiers());
  auto control = unit_->control();

  switch (functionType->refQualifier()) {
    case RefQualifier::kRvalue:
      return control->getRvalueReferenceType(objectType);
    case RefQualifier::kLvalue:
      return control->getLvalueReferenceType(objectType);
    case RefQualifier::kNone:
      break;
  }

  if (type_cast<RvalueReferenceType>(otherFirstParameter))
    return control->getRvalueReferenceType(objectType);
  return control->getLvalueReferenceType(objectType);
}

auto FunctionTemplateOrdering::transform(FunctionSymbol* function,
                                         bool reversed,
                                         FunctionSymbol* other) const
    -> std::optional<TransformedTemplate> {
  auto primary = primaryTemplateOf(function);
  auto templateDeclaration = primary->templateDeclaration();
  auto functionType = type_cast<FunctionType>(primary->type());
  if (!templateDeclaration || !functionType) return std::nullopt;

  TransformedTemplate transformed{.primary = primary,
                                  .templateDeclaration = templateDeclaration};

  if (context_.kind == PartialOrderingContext::Kind::kConversion) {
    transformed.types.push_back({.type = functionType->returnType()});
    return transformed;
  }

  if (primary->isStatic() && primary->hasImplicitObjectParameter()) {
    transformed.types.push_back({.matchesAnyObject = true});
  } else if (context_.kind == PartialOrderingContext::Kind::kCall &&
             primary->hasImplicitObjectParameter() &&
             primaryTemplateOf(other)->hasImplicitObjectParameter() &&
             context_.candidateReversed == context_.otherReversed) {
    transformed.types.push_back({.matchesAnyObject = true});
  } else if (primary->hasImplicitObjectParameter()) {
    auto otherFirst = firstExplicitParameter(primaryTemplateOf(other));
    transformed.types.push_back(
        {.type = objectParameterType(primary, otherFirst)});
  }

  for (auto parameterType : functionType->parameterTypes())
    transformed.types.push_back(orderingType(parameterType));

  transformed.parameterCount = transformed.types.size();
  if (!transformed.types.empty() && transformed.types.back().isPack) {
    transformed.hasTrailingPack = true;
    transformed.trailingPackIndex = transformed.types.size() - 1;
  }

  if (reversed) std::ranges::reverse(transformed.types);

  if (context_.kind == PartialOrderingContext::Kind::kFunctionType) {
    transformed.types.push_back({.type = functionType->returnType()});
    return transformed;
  }

  transformed.types = usedTypes(std::move(transformed.types));
  return transformed;
}

auto FunctionTemplateOrdering::usedTypes(std::vector<OrderingType> types) const
    -> std::vector<OrderingType> {
  std::vector<OrderingType> used;
  for (const auto& type : types) {
    if (used.size() == context_.callArgumentCount) break;
    used.push_back(type);
    if (type.isPack) break;
  }
  return used;
}

auto FunctionTemplateOrdering::adjust(const Type* type) const
    -> ReferenceAdjustment {
  ReferenceAdjustment adjusted;
  adjusted.wasLvalueReference = type_cast<LvalueReferenceType>(type) != nullptr;
  adjusted.wasReference = traits_.is_reference(type);
  auto referenced = traits_.remove_reference(type);
  adjusted.referencedCv = cv_qualifiers(referenced);
  adjusted.type = traits_.remove_cv(referenced);
  return adjusted;
}

auto FunctionTemplateOrdering::deducesIdentically(
    const TransformedTemplate& argument, const TransformedTemplate& parameter,
    const Type* argumentType, const Type* parameterType) const -> bool {
  TypeDeduction forward{unit_, parameter.templateDeclaration};
  if (!forward.deduce(parameterType, argumentType)) return false;
  TypeDeduction backward{unit_, argument.templateDeclaration};
  return backward.deduce(argumentType, parameterType);
}

auto FunctionTemplateOrdering::losesByReferenceBinding(
    const TransformedTemplate& argument, const TransformedTemplate& parameter,
    const ReferenceAdjustment& A, const ReferenceAdjustment& P) const -> bool {
  if (!A.wasReference || !P.wasReference) return false;
  if (!deducesIdentically(argument, parameter, A.type, P.type)) return false;
  if (A.wasLvalueReference && !P.wasLvalueReference) return true;
  return is_more_cv_qualified(A.referencedCv, P.referencedCv);
}

auto FunctionTemplateOrdering::deducesUsedParameters(
    const TypeDeduction& deduction,
    const std::vector<OrderingType>& parameterTypes) const -> bool {
  for (const auto& parameterType : parameterTypes) {
    for (auto slot : deduction.mentionedSlots(parameterType.type)) {
      if (!deduction.deduced(slot)) return false;
    }
  }
  return true;
}

auto FunctionTemplateOrdering::isAtLeastAsSpecialized(
    const TransformedTemplate& argument,
    const TransformedTemplate& parameter) const -> bool {
  TypeDeduction deduction{unit_, parameter.templateDeclaration};

  const auto& argumentTypes = argument.types;
  const auto& parameterTypes = parameter.types;

  std::size_t index = 0;
  for (const auto& P : parameterTypes) {
    if (P.matchesAnyObject) {
      if (index == argumentTypes.size()) return false;
      ++index;
      continue;
    }

    auto adjustedP = adjust(P.type);

    if (P.isPack) {
      auto expansion = deduction.beginExpansion(adjustedP.type);
      for (std::size_t element = 0; index < argumentTypes.size();
           ++index, ++element) {
        auto adjustedA = adjust(argumentTypes[index].type);
        if (deduction.beginElement(expansion, element) &&
            !deduction.deduce(adjustedP.type, adjustedA.type))
          return false;
        if (!deduction.endElement(expansion)) return false;
      }
      if (!deduction.endExpansion(expansion)) return false;
      break;
    }

    if (index == argumentTypes.size()) return false;

    const auto& A = argumentTypes[index++];
    if (A.isPack) return false;
    if (A.matchesAnyObject) continue;

    auto adjustedA = adjust(A.type);
    if (!deduction.deduce(adjustedP.type, adjustedA.type)) return false;
    if (losesByReferenceBinding(parameter, argument, adjustedP, adjustedA))
      return false;
  }

  if (index != argumentTypes.size()) return false;

  return deducesUsedParameters(deduction, parameterTypes);
}

auto FunctionTemplateOrdering::winsByTrailingPack(
    const TransformedTemplate& lhs, const TransformedTemplate& rhs) -> bool {
  if (lhs.hasTrailingPack || !rhs.hasTrailingPack) return false;
  return lhs.parameterCount <= rhs.trailingPackIndex;
}

auto FunctionTemplateOrdering::orderingKind(FunctionSymbol* candidate,
                                            FunctionSymbol* other) const
    -> PartialOrderingContext::Kind {
  if (context_.kind != PartialOrderingContext::Kind::kCall)
    return context_.kind;
  if (!isConversionFunction(primaryTemplateOf(candidate))) return context_.kind;
  if (!isConversionFunction(primaryTemplateOf(other))) return context_.kind;
  return PartialOrderingContext::Kind::kConversion;
}

auto FunctionTemplateOrdering::compare(FunctionSymbol* candidate,
                                       FunctionSymbol* other) -> int {
  context_.kind = orderingKind(candidate, other);

  const bool reverses = context_.candidateReversed != context_.otherReversed;

  auto transformedCandidate =
      transform(candidate, reverses && context_.candidateReversed, other);
  auto transformedOther =
      transform(other, reverses && context_.otherReversed, candidate);
  if (!transformedCandidate || !transformedOther) return 0;

  const bool candidateAtLeast =
      isAtLeastAsSpecialized(*transformedCandidate, *transformedOther);
  const bool otherAtLeast =
      isAtLeastAsSpecialized(*transformedOther, *transformedCandidate);

  if (candidateAtLeast && !otherAtLeast) return 1;
  if (otherAtLeast && !candidateAtLeast) return -1;
  if (!candidateAtLeast) return 0;

  if (winsByTrailingPack(*transformedCandidate, *transformedOther)) return 1;
  if (winsByTrailingPack(*transformedOther, *transformedCandidate)) return -1;

  return compareByConstraints(unit_, candidate, other);
}

}  // namespace

auto compareFunctionTemplateSpecializations(
    TranslationUnit* unit, FunctionSymbol* candidate, FunctionSymbol* other,
    const PartialOrderingContext& context) -> int {
  return FunctionTemplateOrdering{unit, context}.compare(candidate, other);
}

}  // namespace cxx
