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

namespace cxx {
namespace {
auto countConcreteTypeNodes(const Type* type) -> int {
  if (!type) return 0;
  if (getTypeParamInfo(type)) return 0;

  if (auto qual = type_cast<QualType>(type))
    return countConcreteTypeNodes(qual->elementType());
  if (auto pointer = type_cast<PointerType>(type))
    return 1 + countConcreteTypeNodes(pointer->elementType());
  if (auto ref = type_cast<LvalueReferenceType>(type))
    return countConcreteTypeNodes(ref->elementType());
  if (auto ref = type_cast<RvalueReferenceType>(type))
    return countConcreteTypeNodes(ref->elementType());
  if (auto array = type_cast<BoundedArrayType>(type))
    return 1 + countConcreteTypeNodes(array->elementType());
  if (auto array = type_cast<UnboundedArrayType>(type))
    return 1 + countConcreteTypeNodes(array->elementType());
  if (auto function = type_cast<FunctionType>(type)) {
    int score = 1 + countConcreteTypeNodes(function->returnType());
    for (auto param : function->parameterTypes())
      score += countConcreteTypeNodes(param);
    return score;
  }

  return 1;
}

auto templateSpecializationRank(FunctionSymbol* function) -> int {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto functionType = type_cast<FunctionType>(primary->type());
  if (!functionType) return 0;

  int rank = 0;
  for (auto param : functionType->parameterTypes())
    rank += countConcreteTypeNodes(param);
  return rank;
}

void collectReferencedTypeParams(const Type* type,
                                 std::vector<std::pair<int, int>>& out) {
  if (!type) return;
  if (auto info = getTypeParamInfo(type)) {
    auto key = std::pair{info->depth, info->index};
    if (std::ranges::find(out, key) == out.end()) out.push_back(key);
    return;
  }
  if (auto qual = type_cast<QualType>(type))
    return collectReferencedTypeParams(qual->elementType(), out);
  if (auto pointer = type_cast<PointerType>(type))
    return collectReferencedTypeParams(pointer->elementType(), out);
  if (auto ref = type_cast<LvalueReferenceType>(type))
    return collectReferencedTypeParams(ref->elementType(), out);
  if (auto ref = type_cast<RvalueReferenceType>(type))
    return collectReferencedTypeParams(ref->elementType(), out);
  if (auto array = type_cast<BoundedArrayType>(type))
    return collectReferencedTypeParams(array->elementType(), out);
  if (auto array = type_cast<UnboundedArrayType>(type))
    return collectReferencedTypeParams(array->elementType(), out);
  if (auto function = type_cast<FunctionType>(type)) {
    collectReferencedTypeParams(function->returnType(), out);
    for (auto param : function->parameterTypes())
      collectReferencedTypeParams(param, out);
    return;
  }
  if (auto classType = type_cast<ClassType>(type)) {
    auto classSymbol = classType->symbol();
    if (!classSymbol || !classSymbol->isSpecialization()) return;
    for (const auto& arg : classSymbol->templateArguments()) {
      if (auto argType = template_argument_as_type(arg))
        collectReferencedTypeParams(argType, out);
    }
  }
}

auto countDistinctTemplateParams(FunctionSymbol* function) -> int {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto functionType = type_cast<FunctionType>(primary->type());
  if (!functionType) return 0;

  std::vector<std::pair<int, int>> distinct;
  for (auto param : functionType->parameterTypes())
    collectReferencedTypeParams(param, distinct);
  return static_cast<int>(distinct.size());
}

auto countDeclaredTemplateParams(FunctionSymbol* function) -> int {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto templateParams = primary->templateParameters();
  if (!templateParams) return 0;

  return static_cast<int>(templateParams->members().size());
}

using TemplateParamKey = std::pair<int, int>;

[[nodiscard]] auto templateParamKeyOf(Symbol* symbol)
    -> std::optional<TemplateParamKey> {
  if (auto info = getTypeParamInfo(symbol->type()))
    return TemplateParamKey{info->depth, info->index};
  if (auto param = symbol_cast<NonTypeParameterSymbol>(symbol))
    return TemplateParamKey{param->depth(), param->index()};
  return std::nullopt;
}

[[nodiscard]] auto templateParamKeyOf(const TemplateArgument& argument)
    -> std::optional<TemplateParamKey> {
  if (auto type = std::get_if<const Type*>(&argument)) {
    if (auto info = getTypeParamInfo(*type))
      return TemplateParamKey{info->depth, info->index};
    return std::nullopt;
  }
  if (auto symbol = std::get_if<Symbol*>(&argument))
    return templateParamKeyOf(*symbol);
  if (auto expression = std::get_if<ExpressionAST*>(&argument)) {
    if (auto id = ast_cast<IdExpressionAST>(*expression);
        id && id->symbol && !id->nestedNameSpecifier)
      return templateParamKeyOf(id->symbol);
  }
  return std::nullopt;
}

[[nodiscard]] auto ownTemplateParamKeys(FunctionSymbol* function)
    -> std::vector<TemplateParamKey> {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto templateParams = primary->templateParameters();
  if (!templateParams) return {};

  std::vector<TemplateParamKey> keys;
  for (auto member : templateParams->members()) {
    if (auto key = templateParamKeyOf(member)) keys.push_back(*key);
  }
  return keys;
}

[[nodiscard]] auto sameDependentShape(TranslationUnit* unit, const Type* x,
                                      const Type* y) -> std::optional<bool> {
  auto tt = unit->typeTraits();
  x = tt.remove_cvref(x);
  y = tt.remove_cvref(y);

  if (auto infoX = getTypeParamInfo(x)) {
    auto infoY = getTypeParamInfo(y);
    if (!infoY) return false;
    return infoX->depth == infoY->depth && infoX->index == infoY->index;
  }
  if (getTypeParamInfo(y)) return false;

  if (auto ptrX = type_cast<PointerType>(x)) {
    auto ptrY = type_cast<PointerType>(y);
    if (!ptrY) return false;
    return sameDependentShape(unit, ptrX->elementType(), ptrY->elementType());
  }
  if (type_cast<PointerType>(y)) return false;

  auto elementIfArray = [](const Type* t) -> const Type* {
    if (auto a = type_cast<BoundedArrayType>(t)) return a->elementType();
    if (auto a = type_cast<UnboundedArrayType>(t)) return a->elementType();
    return nullptr;
  };
  if (auto elemX = elementIfArray(x)) {
    auto elemY = elementIfArray(y);
    if (!elemY) return false;
    return sameDependentShape(unit, elemX, elemY);
  }
  if (elementIfArray(y)) return false;

  if (auto fnX = type_cast<FunctionType>(x)) {
    auto fnY = type_cast<FunctionType>(y);
    if (!fnY) return false;
    if (fnX->isVariadic() != fnY->isVariadic()) return false;
    auto paramsX = fnX->parameterTypes();
    auto paramsY = fnY->parameterTypes();
    if (paramsX.size() != paramsY.size()) return false;
    auto eq = sameDependentShape(unit, fnX->returnType(), fnY->returnType());
    if (!eq || !*eq) return eq;
    for (std::size_t i = 0; i < paramsX.size(); ++i) {
      eq = sameDependentShape(unit, paramsX[i], paramsY[i]);
      if (!eq || !*eq) return eq;
    }
    return true;
  }
  if (type_cast<FunctionType>(y)) return false;

  if (auto classX = type_cast<ClassType>(x)) {
    auto classY = type_cast<ClassType>(y);
    if (!classY) return false;
    auto symX = classX->symbol();
    auto symY = classY->symbol();
    if (!symX || !symY) return false;
    auto identityX =
        symX->isSpecialization() ? symX->primaryTemplateSymbol() : symX;
    auto identityY =
        symY->isSpecialization() ? symY->primaryTemplateSymbol() : symY;
    if (identityX != identityY) return false;
    if (!symX->isSpecialization()) {
      if (symX->templateDeclaration()) return std::nullopt;
      return true;
    }

    auto argsX = symX->templateArguments();
    auto argsY = symY->templateArguments();
    if (argsX.size() != argsY.size()) return false;
    for (std::size_t i = 0; i < argsX.size(); ++i) {
      auto typeX = template_argument_as_type(argsX[i]);
      auto typeY = template_argument_as_type(argsY[i]);
      if (typeX || typeY) {
        if (!typeX || !typeY) return false;
        auto eq = sameDependentShape(unit, typeX, typeY);
        if (!eq || !*eq) return eq;
        continue;
      }
      auto keyX = templateParamKeyOf(argsX[i]);
      auto keyY = templateParamKeyOf(argsY[i]);
      if (keyX || keyY) {
        if (keyX != keyY) return false;
        continue;
      }
      if (!compare_single_arg(unit, argsX[i], argsY[i])) return false;
    }
    return true;
  }
  if (type_cast<ClassType>(y)) return false;

  return tt.is_same(x, y);
}

using DeducedArguments = std::vector<std::optional<TemplateArgument>>;

[[nodiscard]] auto unifyForPartialOrdering(
    TranslationUnit* unit, const std::vector<TemplateParamKey>& bKeys,
    DeducedArguments& deduced, const Type* p, const Type* raw)
    -> std::optional<bool>;

[[nodiscard]] auto unifyArgumentForPartialOrdering(
    TranslationUnit* unit, const std::vector<TemplateParamKey>& bKeys,
    DeducedArguments& deduced, const TemplateArgument& p,
    const TemplateArgument& a) -> std::optional<bool> {
  auto typeP = template_argument_as_type(p);
  auto typeA = template_argument_as_type(a);
  if (typeP || typeA) {
    if (!typeP || !typeA) return false;
    return unifyForPartialOrdering(unit, bKeys, deduced, typeP, typeA);
  }

  if (auto key = templateParamKeyOf(p)) {
    auto it = std::ranges::find(bKeys, *key);
    if (it != bKeys.end()) {
      auto slot = static_cast<std::size_t>(it - bKeys.begin());
      if (!deduced[slot]) {
        deduced[slot] = a;
        return true;
      }
      return compare_single_arg(unit, *deduced[slot], a);
    }
  }

  if (templateParamKeyOf(a)) return false;
  return compare_single_arg(unit, p, a);
}

auto unifyForPartialOrdering(TranslationUnit* unit,
                             const std::vector<TemplateParamKey>& bKeys,
                             DeducedArguments& deduced, const Type* p,
                             const Type* raw) -> std::optional<bool> {
  auto tt = unit->typeTraits();

  const auto cvP = cv_qualifiers(p);
  const auto cvA = cv_qualifiers(raw);
  if (!is_at_least_as_cv_qualified(cvA, cvP)) return false;

  auto unqualifiedP = tt.remove_cv(p);

  if (auto info = getTypeParamInfo(unqualifiedP)) {
    auto it =
        std::ranges::find(bKeys, TemplateParamKey{info->depth, info->index});
    if (it != bKeys.end()) {
      auto residual = unit->typeTraits().add_cv(
          unqualified_type(raw), residual_cv_qualifiers(cvA, cvP));
      auto slot = static_cast<std::size_t>(it - bKeys.begin());
      if (!deduced[slot]) {
        deduced[slot] = TemplateArgument{residual};
        return true;
      }
      auto previous = std::get_if<const Type*>(&*deduced[slot]);
      if (!previous) return false;
      return sameDependentShape(unit, *previous, residual);
    }
  }

  if (cvP != cvA) return false;

  p = unqualifiedP;
  auto rawStripped = tt.remove_cv(raw);

  if (auto refP = type_cast<LvalueReferenceType>(p)) {
    auto refA = type_cast<LvalueReferenceType>(rawStripped);
    if (!refA) return false;
    return unifyForPartialOrdering(unit, bKeys, deduced, refP->elementType(),
                                   refA->elementType());
  }
  if (type_cast<LvalueReferenceType>(rawStripped)) return false;

  if (auto refP = type_cast<RvalueReferenceType>(p)) {
    auto refA = type_cast<RvalueReferenceType>(rawStripped);
    if (!refA) return false;
    return unifyForPartialOrdering(unit, bKeys, deduced, refP->elementType(),
                                   refA->elementType());
  }
  if (type_cast<RvalueReferenceType>(rawStripped)) return false;

  if (auto ptrP = type_cast<PointerType>(p)) {
    auto ptrA = type_cast<PointerType>(rawStripped);
    if (!ptrA) return false;
    return unifyForPartialOrdering(unit, bKeys, deduced, ptrP->elementType(),
                                   ptrA->elementType());
  }
  if (type_cast<PointerType>(rawStripped)) return false;

  auto elementIfArray = [](const Type* t) -> const Type* {
    if (auto a = type_cast<BoundedArrayType>(t)) return a->elementType();
    if (auto a = type_cast<UnboundedArrayType>(t)) return a->elementType();
    return nullptr;
  };
  if (auto elemP = elementIfArray(p)) {
    auto elemA = elementIfArray(rawStripped);
    if (!elemA) return false;
    return unifyForPartialOrdering(unit, bKeys, deduced, elemP, elemA);
  }
  if (elementIfArray(rawStripped)) return false;

  if (auto fnP = type_cast<FunctionType>(p)) {
    auto fnA = type_cast<FunctionType>(rawStripped);
    if (!fnA) return false;
    if (fnP->isVariadic() != fnA->isVariadic()) return false;
    auto paramsP = fnP->parameterTypes();
    auto paramsA = fnA->parameterTypes();
    if (paramsP.size() != paramsA.size()) return false;
    auto ok = unifyForPartialOrdering(unit, bKeys, deduced, fnP->returnType(),
                                      fnA->returnType());
    if (!ok || !*ok) return ok;
    for (std::size_t i = 0; i < paramsP.size(); ++i) {
      ok =
          unifyForPartialOrdering(unit, bKeys, deduced, paramsP[i], paramsA[i]);
      if (!ok || !*ok) return ok;
    }
    return true;
  }
  if (type_cast<FunctionType>(rawStripped)) return false;

  if (auto classP = type_cast<ClassType>(p)) {
    auto classA = type_cast<ClassType>(rawStripped);
    if (!classA) return false;
    auto symP = classP->symbol();
    auto symA = classA->symbol();
    if (!symP || !symA) return false;
    auto identityP =
        symP->isSpecialization() ? symP->primaryTemplateSymbol() : symP;
    auto identityA =
        symA->isSpecialization() ? symA->primaryTemplateSymbol() : symA;
    if (identityP != identityA) return false;
    if (!symP->isSpecialization()) {
      if (symP->templateDeclaration()) return std::nullopt;
      return true;
    }

    auto argsP = symP->templateArguments();
    auto argsA = symA->templateArguments();
    if (argsP.size() != argsA.size()) return false;
    for (std::size_t i = 0; i < argsP.size(); ++i) {
      auto ok = unifyArgumentForPartialOrdering(unit, bKeys, deduced, argsP[i],
                                                argsA[i]);
      if (!ok || !*ok) return ok;
    }
    return true;
  }
  if (type_cast<ClassType>(rawStripped)) return false;

  if (getTypeParamInfo(rawStripped)) return false;
  return tt.is_same(p, rawStripped);
}

[[nodiscard]] auto primaryTemplateOf(FunctionSymbol* function)
    -> FunctionSymbol* {
  if (!function->isSpecialization()) return function;
  auto primary = function->primaryTemplateSymbol();
  return primary ? primary : function;
}

[[nodiscard]] auto hasEquivalentTemplateSignature(TranslationUnit* unit,
                                                  FunctionSymbol* a,
                                                  FunctionSymbol* b) -> bool {
  auto primaryA = primaryTemplateOf(a);
  auto primaryB = primaryTemplateOf(b);
  auto templateA = primaryA->templateDeclaration();
  auto templateB = primaryB->templateDeclaration();
  if (!templateA || !templateB) return false;
  if (!areTemplateParameterListsEquivalentForPartialOrdering(
          unit, templateA->templateParameterList,
          templateB->templateParameterList))
    return false;

  auto functionTypeA = type_cast<FunctionType>(primaryA->type());
  auto functionTypeB = type_cast<FunctionType>(primaryB->type());
  if (!functionTypeA || !functionTypeB) return false;
  if (functionTypeA->isVariadic() != functionTypeB->isVariadic()) return false;

  auto paramsA = functionTypeA->parameterTypes();
  auto paramsB = functionTypeB->parameterTypes();
  if (paramsA.size() != paramsB.size()) return false;

  for (std::size_t i = 0; i < paramsA.size(); ++i) {
    if (!areTypesEquivalentForPartialOrdering(unit, paramsA[i], paramsB[i],
                                              templateA, templateB))
      return false;
  }

  const bool conversionA = name_cast<ConversionFunctionId>(primaryA->name());
  const bool conversionB = name_cast<ConversionFunctionId>(primaryB->name());
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

struct AdjustedPartialOrderingType {
  const Type* type = nullptr;
  bool wasReference = false;
  bool wasLvalueReference = false;
  CvQualifiers referencedCvQualifiers = CvQualifiers::kNone;
};

[[nodiscard]] auto adjustForPartialOrdering(TranslationUnit* unit,
                                            const Type* type)
    -> AdjustedPartialOrderingType {
  auto tt = unit->typeTraits();

  AdjustedPartialOrderingType adjusted;
  adjusted.wasLvalueReference = type_cast<LvalueReferenceType>(type) != nullptr;
  adjusted.wasReference = adjusted.wasLvalueReference ||
                          type_cast<RvalueReferenceType>(type) != nullptr;

  auto referenced = tt.remove_reference(type);
  adjusted.referencedCvQualifiers = cv_qualifiers(referenced);
  adjusted.type = tt.remove_cv(referenced);

  return adjusted;
}

[[nodiscard]] auto losesToReferenceBinding(
    const AdjustedPartialOrderingType& argument,
    const AdjustedPartialOrderingType& parameter) -> bool {
  if (!argument.wasReference || !parameter.wasReference) return false;
  if (argument.wasLvalueReference && !parameter.wasLvalueReference) return true;
  return is_more_cv_qualified(argument.referencedCvQualifiers,
                              parameter.referencedCvQualifiers);
}

[[nodiscard]] auto isLessSpecializedByReferenceBinding(TranslationUnit* unit,
                                                       FunctionSymbol* a,
                                                       FunctionSymbol* b)
    -> bool {
  auto functionTypeA = type_cast<FunctionType>(primaryTemplateOf(a)->type());
  auto functionTypeB = type_cast<FunctionType>(primaryTemplateOf(b)->type());
  if (!functionTypeA || !functionTypeB) return false;

  auto paramsA = functionTypeA->parameterTypes();
  auto paramsB = functionTypeB->parameterTypes();
  if (paramsA.size() != paramsB.size()) return false;

  for (std::size_t i = 0; i < paramsA.size(); ++i) {
    auto argument = adjustForPartialOrdering(unit, paramsB[i]);
    auto parameter = adjustForPartialOrdering(unit, paramsA[i]);
    if (losesToReferenceBinding(argument, parameter)) return true;
  }

  return false;
}

[[nodiscard]] auto isAtLeastAsSpecializedAs(TranslationUnit* unit,
                                            FunctionSymbol* a,
                                            FunctionSymbol* b)
    -> std::optional<bool> {
  auto primaryA = primaryTemplateOf(a);
  auto primaryB = primaryTemplateOf(b);

  auto functionTypeA = type_cast<FunctionType>(primaryA->type());
  auto functionTypeB = type_cast<FunctionType>(primaryB->type());
  if (!functionTypeA || !functionTypeB) return std::nullopt;

  auto paramsA = functionTypeA->parameterTypes();
  auto paramsB = functionTypeB->parameterTypes();
  if (paramsA.size() != paramsB.size()) return std::nullopt;
  if (functionTypeA->isVariadic() != functionTypeB->isVariadic())
    return std::nullopt;

  auto bKeys = ownTemplateParamKeys(b);
  DeducedArguments deduced(bKeys.size());

  for (std::size_t i = 0; i < paramsA.size(); ++i) {
    auto parameter = adjustForPartialOrdering(unit, paramsB[i]);
    auto argument = adjustForPartialOrdering(unit, paramsA[i]);
    auto ok = unifyForPartialOrdering(unit, bKeys, deduced, parameter.type,
                                      argument.type);
    if (!ok) return std::nullopt;
    if (!*ok) return false;
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

[[nodiscard]] auto comparePartialOrderingReal(TranslationUnit* unit,
                                              FunctionSymbol* a,
                                              FunctionSymbol* b)
    -> std::optional<int> {
  auto aAtLeastB = isAtLeastAsSpecializedAs(unit, a, b);
  auto bAtLeastA = isAtLeastAsSpecializedAs(unit, b, a);
  if (!aAtLeastB || !bAtLeastA) return std::nullopt;

  if (*aAtLeastB && *bAtLeastA) {
    if (isLessSpecializedByReferenceBinding(unit, a, b)) aAtLeastB = false;
    if (isLessSpecializedByReferenceBinding(unit, b, a)) bAtLeastA = false;
  }

  if (*aAtLeastB && !*bAtLeastA) return 1;
  if (*bAtLeastA && !*aAtLeastB) return -1;

  return compareByConstraints(unit, a, b);
}

auto compareTemplateSpecialization(TranslationUnit* unit,
                                   FunctionSymbol* candidate,
                                   FunctionSymbol* other) -> int {
  if (auto real = comparePartialOrderingReal(unit, candidate, other))
    return *real;

  auto rankA = templateSpecializationRank(candidate);
  auto rankB = templateSpecializationRank(other);
  if (rankA != rankB) return rankA > rankB ? 1 : -1;

  auto distinctA = countDistinctTemplateParams(candidate);
  auto distinctB = countDistinctTemplateParams(other);
  if (distinctA != distinctB) return distinctA < distinctB ? 1 : -1;

  auto declaredA = countDeclaredTemplateParams(candidate);
  auto declaredB = countDeclaredTemplateParams(other);
  if (declaredA != declaredB) return declaredA < declaredB ? 1 : -1;

  return 0;
}

auto getMinRequiredArgs(FunctionSymbol* func, int totalParams) -> int {
  auto fpScope = func->functionParameters();
  if (!fpScope) return totalParams;

  std::vector<ParameterSymbol*> params;
  for (auto member : fpScope->members()) {
    if (auto param = symbol_cast<ParameterSymbol>(member))
      params.push_back(param);
  }
  if (params.empty()) return totalParams;

  int defaultCount = 0;
  for (int i = static_cast<int>(params.size()) - 1; i >= 0; --i) {
    if (params[i]->defaultArgument())
      ++defaultCount;
    else
      break;
  }
  return totalParams - defaultCount;
}

auto isPackExpansionParameterType(const Type* type) -> bool {
  if (!type) return false;
  if (auto info = getTypeParamInfo(type)) return info->isPack;
  if (auto qual = type_cast<QualType>(type))
    return isPackExpansionParameterType(qual->elementType());
  if (auto ref = type_cast<LvalueReferenceType>(type))
    return isPackExpansionParameterType(ref->elementType());
  if (auto ref = type_cast<RvalueReferenceType>(type))
    return isPackExpansionParameterType(ref->elementType());
  if (auto ptr = type_cast<PointerType>(type))
    return isPackExpansionParameterType(ptr->elementType());
  return false;
}

[[nodiscard]] auto functionTemplateHasPackParameter(FunctionSymbol* pattern)
    -> bool {
  auto type = type_cast<FunctionType>(pattern->type());
  if (!type) return false;
  for (auto param : type->parameterTypes()) {
    if (isPackExpansionParameterType(param)) return true;
  }
  return false;
}
}  // namespace

auto compareFunctionTemplateSpecializations(TranslationUnit* unit,
                                            FunctionSymbol* candidate,
                                            FunctionSymbol* other) -> int {
  return compareTemplateSpecialization(unit, candidate, other);
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
      argCount < getMinRequiredArgs(pattern, paramCount)) {
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

auto OverloadResolution::implicitObjectArgumentConversion(
    FunctionSymbol* function, const ImplicitObjectArgument& object)
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

  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  auto implicitObjectClass =
      classSymbol ? classSymbol->type() : traits.remove_cvref(object.type);

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

  return conversion;
}

auto haveSameParameterTypes(FunctionSymbol* lhs, FunctionSymbol* rhs) -> bool {
  auto lhsType = type_cast<FunctionType>(lhs->type());
  auto rhsType = type_cast<FunctionType>(rhs->type());
  if (!lhsType || !rhsType) return false;
  if (lhsType->isVariadic() != rhsType->isVariadic()) return false;
  return std::ranges::equal(lhsType->parameterTypes(),
                            rhsType->parameterTypes());
}

auto compareDeductionCandidates(const DeductionCandidateInfo& lhs,
                                const DeductionCandidateInfo& rhs,
                                bool parameterTypesMatch) -> int {
  if (parameterTypesMatch &&
      lhs.fromInheritedConstructor != rhs.fromInheritedConstructor)
    return lhs.fromInheritedConstructor ? -1 : 1;

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

    if (currBetter && !refBetter) {
      best.clear();
      best.push_back(&curr);
    } else if (refBetter && !currBetter) {
    } else if (preferNonTemplate && curr.fromTemplate != ref.fromTemplate) {
      if (!curr.fromTemplate) {
        best.clear();
        best.push_back(&curr);
      }
    } else if (int order = curr.fromTemplate && ref.fromTemplate
                               ? compareTemplateSpecialization(
                                     unit_, curr.symbol, ref.symbol)
                               : 0;
               order != 0) {
      if (order > 0) {
        best.clear();
        best.push_back(&curr);
      }
    } else if (int order = compareDeductionCandidates(
                   curr.deduction, ref.deduction,
                   haveSameParameterTypes(curr.symbol, ref.symbol));
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

auto isExcludedInheritedConstructor(const TypeTraits& traits,
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

auto OverloadResolution::resolveInitializerListConstructor(
    ClassSymbol* classSymbol, BracedInitListAST* bracedInitList,
    InitializationKind initializationKind) -> ConstructorResult {
  std::vector<ExpressionAST*> args = {bracedInitList};
  return resolveConstructor(classSymbol, args, initializationKind, true);
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

  auto reject = [&](FunctionSymbol* ctor, std::string reason) {
    result.rejected.push_back({ctor, std::move(reason)});
  };

  auto rejectArity = [&](FunctionSymbol* ctor, int paramCount) {
    reject(ctor, std::format("requires {} argument{}, but {} {} provided",
                             paramCount, paramCount == 1 ? "" : "s", argCount,
                             argCount == 1 ? "was" : "were"));
  };

  auto isInitializerListConstructor = [&](FunctionSymbol* ctor) {
    auto type = type_cast<FunctionType>(ctor->type());
    if (!type || type->parameterTypes().empty()) return false;
    auto firstParameter = type->parameterTypes().front();
    if (!traits.initializer_list_element_type(firstParameter)) return false;
    auto parameterCount = static_cast<int>(type->parameterTypes().size());
    return getMinRequiredArgs(ctor, parameterCount) <= 1;
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

      List<ExpressionAST*>* expressionList = nullptr;
      auto tail = &expressionList;
      for (auto arg : args) {
        *tail = make_list_node(arena_, arg);
        tail = &(*tail)->next;
      }

      TemplateArgumentDeduction deduction(unit_);
      auto deducedArgs = deduction.deduce(ctor, expressionList,
                                          /*explicitTemplateArguments=*/{});
      if (!deducedArgs.has_value()) {
        reject(ctor, "template argument deduction failed");
        continue;
      }

      const auto loc = args.empty() ? classSymbol->location()
                                    : args.front()->firstSourceLocation();

      auto instCtor = ASTRewriter::instantiateOverloadCandidate(
          unit_, *deducedArgs, ctor, loc, /*argsComplete=*/true);
      if (!instCtor) {
        reject(ctor, "substitution failed for the deduced arguments");
        continue;
      }

      ctor = instCtor;
      deducedArgsForCandidate = *deducedArgs;

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
      if (argCount < getMinRequiredArgs(ctor, paramCount)) {
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

      auto conv = stdconv_.computeConversionSequence(
          args[i], *paramIt, InitializationKind::kCopyInitialization,
          convertsFirstCopyInitializationArgument
              ? ConversionContext::kStandardOnly
              : ConversionContext::kImplicit);
      if (!conv) {
        cand.viable = false;
        reject(ctor, std::format(
                         "no known conversion from '{}' to '{}' for argument "
                         "{}",
                         to_string(args[i]->type), to_string(*paramIt), i + 1));
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
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;

  if (!scope || !name) return result;

  auto symbol = qualifiedLookup(scope, name);
  if (!symbol) return result;

  if (auto funcSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    addOverloadCandidate(result, funcSymbol);
    return result;
  }

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    for (auto func : overloadSet->functions()) {
      if (isPureFriend(func)) continue;
      addOverloadCandidate(result, func);
    }
  }

  return result;
}

auto OverloadResolution::buildCallCandidate(
    FunctionSymbol* function, const FunctionType* type,
    std::span<ExpressionAST* const> args,
    std::vector<RejectedCandidate>* rejected) -> std::optional<Candidate> {
  const auto argCount = static_cast<int>(args.size());
  const auto paramCount = static_cast<int>(type->parameterTypes().size());

  auto reject = [&](std::string reason) {
    if (rejected) rejected->push_back({function, std::move(reason)});
  };

  auto rejectArity = [&] {
    reject(std::format("requires {} argument{}, but {} {} provided", paramCount,
                       paramCount == 1 ? "" : "s", argCount,
                       argCount == 1 ? "was" : "were"));
  };

  if (argCount > paramCount && !type->isVariadic()) {
    rejectArity();
    return std::nullopt;
  }

  if (argCount < paramCount &&
      argCount < getMinRequiredArgs(function, paramCount)) {
    rejectArity();
    return std::nullopt;
  }

  Candidate cand{function};
  cand.viable = true;

  auto paramIt = type->parameterTypes().begin();
  auto paramEnd = type->parameterTypes().end();
  for (int i = 0; i < argCount && paramIt != paramEnd; ++i, ++paramIt) {
    auto conv = computeImplicitConversionSequence(args[i], *paramIt);
    if (!conv) {
      reject(
          std::format("no known conversion from '{}' to '{}' for argument {}",
                      to_string(args[i]->type), to_string(*paramIt), i + 1));
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

auto OverloadResolution::resolveCall(
    const std::vector<FunctionSymbol*>& candidates,
    std::span<ExpressionAST* const> args, bool* ambiguous) -> FunctionSymbol* {
  if (ambiguous) *ambiguous = false;

  std::vector<Candidate> viableCandidates;
  for (auto function : candidates) {
    auto type = type_cast<FunctionType>(function->type());
    if (!type) continue;
    if (auto cand = buildCallCandidate(function, type, args))
      viableCandidates.push_back(std::move(*cand));
  }

  auto [bestPtr, isAmbiguous] =
      selectBestViableFunction(viableCandidates, /*preferNonTemplate=*/true);

  if (isAmbiguous) {
    if (ambiguous) *ambiguous = true;
    return nullptr;
  }

  return bestPtr ? bestPtr->symbol : nullptr;
}

auto OverloadResolution::collectCandidates(Symbol* symbol) const
    -> std::vector<FunctionSymbol*> {
  auto functions = views::each_function(symbol);
  return {functions.begin(), functions.end()};
}

auto OverloadResolution::resolveBinaryOperator(
    const std::vector<FunctionSymbol*>& candidates, const Type* leftType,
    const Type* rightType, bool* ambiguous, ExpressionAST* leftExpr,
    ExpressionAST* rightExpr) -> FunctionSymbol* {
  std::vector<BinaryOperatorCandidate> operatorCandidates;
  operatorCandidates.reserve(candidates.size());
  for (auto candidate : candidates)
    operatorCandidates.push_back({.symbol = candidate});
  return resolveBinaryOperator(operatorCandidates, leftType, rightType,
                               ambiguous, leftExpr, rightExpr);
}

auto OverloadResolution::resolveBinaryOperator(
    const std::vector<BinaryOperatorCandidate>& candidates,
    const Type* leftType, const Type* rightType, bool* ambiguous,
    ExpressionAST* leftExpr, ExpressionAST* rightExpr) -> FunctionSymbol* {
  if (ambiguous) *ambiguous = false;

  if (candidates.empty()) return nullptr;

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

      List<ExpressionAST*>* argList = nullptr;
      auto tail = &argList;
      if (!isMember) {
        *tail = make_list_node(arena_, candidateLeftExpr);
        tail = &(*tail)->next;
      }
      if (candidateRightExpr) {
        *tail = make_list_node(arena_, candidateRightExpr);
      }

      TemplateArgumentDeduction deduction(unit_);
      auto deducedArgs = deduction.deduce(candidate, argList,
                                          /*explicitTemplateArguments=*/{});
      if (!deducedArgs.has_value()) continue;

      auto instFunc = ASTRewriter::instantiateOverloadCandidate(
          unit_, *deducedArgs, candidate,
          candidateLeftExpr->firstSourceLocation(),
          /*argsComplete=*/true);
      if (!instFunc) continue;

      candidate = instFunc;
      deducedArgsForCandidate = *deducedArgs;
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

    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;

    if (candidateRightType) {
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
             .valueCategory = candidateLeftExpr->valueCategory});
        if (!objectConversion) continue;
        left = *objectConversion;
        right =
            stdconv_.computeConversionSequence(candidateRightExpr, params[0]);
      } else {
        if (params.size() != 2) continue;
        left = stdconv_.computeConversionSequence(candidateLeftExpr, params[0]);
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
             .valueCategory = candidateLeftExpr->valueCategory});
        if (!objectConversion) continue;
        left = *objectConversion;
      } else {
        if (params.size() != 1) continue;
        left = stdconv_.computeConversionSequence(candidateLeftExpr, params[0]);
      }
    }

    if (!left) continue;
    if (candidateRightType && (!right || !*right)) continue;

    if (operatorCandidate.reversed && right) std::swap(left, *right);
    viable.push_back({candidate, left, right, deducedArgsForCandidate,
                      operatorCandidate.rewritten, operatorCandidate.reversed});
  }

  if (viable.empty()) return nullptr;

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

    if (viable[i].symbol->isSpecialization() !=
        best->symbol->isSpecialization()) {
      if (!viable[i].symbol->isSpecialization()) {
        best = &viable[i];
        foundEquivalent = false;
      }
      continue;
    }

    if (viable[i].symbol->isSpecialization() &&
        best->symbol->isSpecialization()) {
      auto order =
          compareTemplateSpecialization(unit_, viable[i].symbol, best->symbol);
      if (order > 0) {
        best = &viable[i];
        foundEquivalent = false;
        continue;
      }
      if (order < 0) continue;
    }

    foundEquivalent = true;
  }

  if (foundEquivalent) {
    if (ambiguous) *ambiguous = true;
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

  for (auto candidate : findCandidates(searchScope, notEqualName)) {
    auto candidateType = type_cast<FunctionType>(candidate->type());
    if (!candidateType) continue;

    if (candidateType->parameterTypes() != equalityType->parameterTypes())
      continue;
    if (candidateType->cvQualifiers() != equalityType->cvQualifiers()) continue;
    if (candidateType->refQualifier() != equalityType->refQualifier()) continue;
    if (candidate->parent() != equalityOperator->parent()) continue;
    if (!trailingRequiresClausesEquivalent(
            unit_, candidate->trailingRequiresClause(),
            equalityOperator->trailingRequiresClause()))
      continue;

    return false;
  }

  return true;
}

auto OverloadResolution::lookupOperator(const Type* type, TokenKind op,
                                        const Type* rightType,
                                        ExpressionAST* leftExpr,
                                        ExpressionAST* rightExpr)
    -> FunctionSymbol* {
  lastLookupAmbiguous_ = false;
  lastOperatorRewritten_ = false;
  lastOperatorReversed_ = false;

  auto name = control_->getOperatorId(op);
  if (!name) return nullptr;

  std::vector<BinaryOperatorCandidate> candidates;

  auto addCandidate = [&](FunctionSymbol* function, bool rewritten,
                          bool reversed) {
    BinaryOperatorCandidate candidate{function, rewritten, reversed};
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
      for (auto function : findCandidates(classSymbol, operatorName))
        addCandidate(function, rewritten, reversed);
    }
  };

  addMemberCandidates(type, name, false, false);

  auto isClassOrEnum = [&](const Type* t) {
    auto stripped = traits.remove_cvref(t);
    return traits.is_class(stripped) || traits.is_enum(stripped);
  };
  bool operandNeedsAdl =
      isClassOrEnum(type) || (rightType && isClassOrEnum(rightType));

  if (operandNeedsAdl) {
    std::vector<const Type*> argTypes{type};
    if (rightType) argTypes.push_back(rightType);
    for (auto func : argumentDependentLookup(unit_, name, argTypes)) {
      addCandidate(func, false, false);
    }
  }

  auto addRewrittenCandidates = [&](TokenKind rewrittenOp,
                                    const Type* firstOperandType,
                                    const Type* secondOperandType,
                                    bool reversed) {
    auto rewrittenName = control_->getOperatorId(rewrittenOp);
    if (!rewrittenName) return;

    const bool requiresRewriteTarget = rewrittenOp == TokenKind::T_EQUAL_EQUAL;

    auto accept = [&](FunctionSymbol* function) {
      if (requiresRewriteTarget && !isRewriteTarget(function, firstOperandType))
        return;
      addCandidate(function, true, reversed);
    };

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(firstOperandType))) {
      if (auto classSymbol = classType->symbol()) {
        traits.requireCompleteClass(classSymbol);
        for (auto function : findCandidates(classSymbol, rewrittenName))
          accept(function);
      }
    }

    if (operandNeedsAdl) {
      std::vector<const Type*> argTypes{firstOperandType, secondOperandType};
      for (auto function :
           argumentDependentLookup(unit_, rewrittenName, argTypes))
        accept(function);
    }
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
  auto selected = resolveBinaryOperator(candidates, type, rightType, &ambiguous,
                                        leftExpr, rightExpr);
  lastLookupAmbiguous_ = ambiguous;
  return selected;
}
}  // namespace cxx
