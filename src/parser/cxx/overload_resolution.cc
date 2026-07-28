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
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>

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
      if (auto argType = std::get_if<const Type*>(&arg))
        collectReferencedTypeParams(*argType, out);
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

[[nodiscard]] auto ownTemplateParamKeys(FunctionSymbol* function)
    -> std::vector<std::pair<int, int>> {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto templateParams = primary->templateParameters();
  if (!templateParams) return {};

  std::vector<std::pair<int, int>> keys;
  for (auto member : templateParams->members()) {
    if (auto info = getTypeParamInfo(member->type()))
      keys.emplace_back(info->depth, info->index);
  }
  return keys;
}

[[nodiscard]] auto hasTemplateParamPack(FunctionSymbol* function) -> bool {
  auto primary = function->isSpecialization()
                     ? function->primaryTemplateSymbol()
                     : function;
  if (!primary) primary = function;

  auto templateParams = primary->templateParameters();
  if (!templateParams) return false;

  for (auto member : templateParams->members()) {
    if (auto info = getTypeParamInfo(member->type())) {
      if (info->isPack) return true;
    }
  }
  return false;
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
      auto typeX = std::get_if<const Type*>(&argsX[i]);
      auto typeY = std::get_if<const Type*>(&argsY[i]);
      if (!typeX || !typeY) continue;
      auto eq = sameDependentShape(unit, *typeX, *typeY);
      if (!eq || !*eq) return eq;
    }
    return true;
  }
  if (type_cast<ClassType>(y)) return false;

  return tt.is_same(x, y);
}

[[nodiscard]] auto unifyForPartialOrdering(
    TranslationUnit* unit, const std::vector<std::pair<int, int>>& bKeys,
    std::vector<const Type*>& deduced, const Type* p, const Type* raw)
    -> std::optional<bool> {
  auto tt = unit->typeTraits();
  p = tt.remove_cvref(p);
  auto rawStripped = tt.remove_cvref(raw);

  if (auto info = getTypeParamInfo(p)) {
    auto it = std::ranges::find(bKeys, std::pair{info->depth, info->index});
    if (it != bKeys.end()) {
      auto slot = static_cast<std::size_t>(it - bKeys.begin());
      if (!deduced[slot]) {
        deduced[slot] = rawStripped;
        return true;
      }
      return sameDependentShape(unit, deduced[slot], rawStripped);
    }
  }

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
      auto typeP = std::get_if<const Type*>(&argsP[i]);
      auto typeA = std::get_if<const Type*>(&argsA[i]);
      if (!typeP || !typeA) continue;
      auto ok = unifyForPartialOrdering(unit, bKeys, deduced, *typeP, *typeA);
      if (!ok || !*ok) return ok;
    }
    return true;
  }
  if (type_cast<ClassType>(rawStripped)) return false;

  if (getTypeParamInfo(rawStripped)) return false;
  return tt.is_same(p, rawStripped);
}

[[nodiscard]] auto isAtLeastAsSpecializedAs(TranslationUnit* unit,
                                            FunctionSymbol* a,
                                            FunctionSymbol* b)
    -> std::optional<bool> {
  if (hasTemplateParamPack(a) || hasTemplateParamPack(b)) return std::nullopt;

  auto primaryA = a->isSpecialization() ? a->primaryTemplateSymbol() : a;
  if (!primaryA) primaryA = a;
  auto primaryB = b->isSpecialization() ? b->primaryTemplateSymbol() : b;
  if (!primaryB) primaryB = b;

  auto functionTypeA = type_cast<FunctionType>(primaryA->type());
  auto functionTypeB = type_cast<FunctionType>(primaryB->type());
  if (!functionTypeA || !functionTypeB) return std::nullopt;

  auto paramsA = functionTypeA->parameterTypes();
  auto paramsB = functionTypeB->parameterTypes();
  if (paramsA.size() != paramsB.size()) return std::nullopt;
  if (functionTypeA->isVariadic() != functionTypeB->isVariadic())
    return std::nullopt;

  auto bKeys = ownTemplateParamKeys(b);
  std::vector<const Type*> deduced(bKeys.size(), nullptr);

  for (std::size_t i = 0; i < paramsA.size(); ++i) {
    auto ok =
        unifyForPartialOrdering(unit, bKeys, deduced, paramsB[i], paramsA[i]);
    if (!ok) return std::nullopt;
    if (!*ok) return false;
  }
  return true;
}

[[nodiscard]] auto comparePartialOrderingReal(TranslationUnit* unit,
                                              FunctionSymbol* a,
                                              FunctionSymbol* b)
    -> std::optional<int> {
  auto aAtLeastB = isAtLeastAsSpecializedAs(unit, a, b);
  auto bAtLeastA = isAtLeastAsSpecializedAs(unit, b, a);
  if (!aAtLeastB || !bAtLeastA) return std::nullopt;

  if (*aAtLeastB && !*bAtLeastA) return 1;
  if (*bAtLeastA && !*aAtLeastB) return -1;
  return 0;
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
}  // namespace

auto functionTemplateHasPackParameter(FunctionSymbol* pattern) -> bool {
  auto type = type_cast<FunctionType>(pattern->type());
  if (!type) return false;
  for (auto param : type->parameterTypes()) {
    if (isPackExpansionParameterType(param)) return true;
  }
  return false;
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

auto OverloadResolution::initializerListElementType(const Type* targetType)
    -> const Type* {
  return stdconv_.initializerListElementType(targetType);
}

auto OverloadResolution::selectBestViableFunction(
    std::vector<Candidate>& candidates, bool useCvTiebreaker,
    bool preferNonTemplate) -> OverloadResult {
  if (candidates.empty()) return {};

  std::vector<Candidate*> best;
  best.push_back(&candidates[0]);

  for (size_t i = 1; i < candidates.size(); ++i) {
    auto& curr = candidates[i];
    auto& ref = *best[0];

    bool currBetter = false;
    bool refBetter = false;

    auto n = std::min(curr.conversions.size(), ref.conversions.size());
    for (size_t j = 0; j < n; ++j) {
      if (curr.conversions[j].isBetterThan(ref.conversions[j]))
        currBetter = true;
      if (ref.conversions[j].isBetterThan(curr.conversions[j]))
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
    } else if (useCvTiebreaker && curr.exactCvMatch != ref.exactCvMatch) {
      if (curr.exactCvMatch) {
        best.clear();
        best.push_back(&curr);
      }
    } else {
      best.push_back(&curr);
    }
  }

  if (best.empty()) return {};
  if (best.size() > 1) return {best[0], true};

  completeDeferredWinnerBody(best[0]->symbol, best[0]->deducedTemplateArgs);
  return {best[0], false};
}

void OverloadResolution::completeDeferredWinnerBody(
    FunctionSymbol* winner, List<TemplateArgumentAST*>* deducedTemplateArgs) {
  if (!winner || !deducedTemplateArgs) return;
  if (!winner->isSpecialization()) return;
  auto primary = winner->primaryTemplateSymbol();
  if (!primary) return;
  ASTRewriter::instantiateForArgs(unit_, deducedTemplateArgs, primary,
                                  winner->location(),
                                  /*argsComplete=*/true);
}

auto OverloadResolution::resolveConstructor(
    ClassSymbol* classSymbol, const std::vector<ExpressionAST*>& args)
    -> ConstructorResult {
  ConstructorResult result;

  auto argCount = static_cast<int>(args.size());

  const auto constructors = classSymbol->constructors();

  for (auto ctor : constructors) {
    if (ctor->canonical() != ctor) continue;

    const bool templateCandidate =
        ctor->templateDeclaration() != nullptr && !ctor->isSpecialization();
    List<TemplateArgumentAST*>* deducedArgsForCandidate = nullptr;

    if (templateCandidate) {
      if (templateCandidateArityRejects(ctor, argCount)) continue;

      List<ExpressionAST*>* expressionList = nullptr;
      auto tail = &expressionList;
      for (auto arg : args) {
        *tail = make_list_node(arena_, arg);
        tail = &(*tail)->next;
      }

      TemplateArgumentDeduction deduction(unit_);
      auto deducedArgs = deduction.deduce(ctor, expressionList,
                                          /*explicitTemplateArguments=*/{});
      if (!deducedArgs.has_value()) continue;

      const auto loc = args.empty() ? classSymbol->location()
                                    : args.front()->firstSourceLocation();

      auto instCtor = ASTRewriter::instantiateForArgs(
          unit_, *deducedArgs, ctor, loc, /*argsComplete=*/true,
          /*declarationOnly=*/!functionTemplateHasPackParameter(ctor));
      if (!instCtor) continue;

      ctor = instCtor;
      deducedArgsForCandidate = *deducedArgs;
    }

    auto type = type_cast<FunctionType>(ctor->type());
    if (!type) continue;

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) continue;
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(ctor, paramCount)) continue;
    }

    Candidate cand{ctor};
    cand.viable = true;
    cand.fromTemplate = templateCandidate;
    cand.deducedTemplateArgs = deducedArgsForCandidate;

    auto paramIt = type->parameterTypes().begin();
    auto paramEnd = type->parameterTypes().end();
    for (size_t i = 0; i < args.size() && paramIt != paramEnd; ++i, ++paramIt) {
      auto conv = computeImplicitConversionSequence(args[i], *paramIt);
      if (conv.rank == ConversionRank::kNone) {
        cand.viable = false;
        break;
      }
      cand.conversions.push_back(conv);
    }

    if (cand.viable && type->isVariadic()) {
      for (int i = paramCount; i < argCount; ++i) {
        ImplicitConversionSequence ellipsisConv;
        ellipsisConv.kind = ConversionSequenceKind::kEllipsis;
        ellipsisConv.rank = ConversionRank::kConversion;
        cand.conversions.push_back(ellipsisConv);
      }
    }

    if (cand.viable) result.candidates.push_back(std::move(cand));
  }

  auto [bestPtr, ambiguous] = selectBestViableFunction(
      result.candidates, /*useCvTiebreaker=*/false, /*preferNonTemplate=*/true);
  result.best = bestPtr;
  result.ambiguous = ambiguous;
  return result;
}

auto OverloadResolution::computeImplicitConversionSequence(
    ExpressionAST* expr, const Type* targetType) -> ImplicitConversionSequence {
  return stdconv_.computeConversionSequence(expr, targetType);
}

void OverloadResolution::wrapWithImplicitCast(ImplicitCastKind castKind,
                                              const Type* type,
                                              ExpressionAST*& expr) {
  stdconv_.wrapWithImplicitCast(castKind, type, expr);
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
    result.push_back(funcSymbol);
    return result;
  }

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    for (auto func : overloadSet->functions()) {
      if (func->canonical() == func) {
        result.push_back(func);
      }
    }
  }

  return result;
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
  if (ambiguous) *ambiguous = false;

  if (candidates.empty()) return nullptr;

  struct ViableCandidate {
    FunctionSymbol* symbol;
    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;
    List<TemplateArgumentAST*>* deducedTemplateArgs = nullptr;
  };

  auto remove_cvref = [&](const Type* type) {
    if (!type) return type;
    return traits.remove_cvref(type);
  };

  auto makeExactMatch = [&](const Type* type) -> ImplicitConversionSequence {
    ImplicitConversionSequence seq;
    seq.rank = ConversionRank::kExactMatch;
    seq.steps.push_back({ImplicitCastKind::kIdentity, type});
    return seq;
  };

  auto rankImplicitObject = [&](const Type* objectType,
                                const FunctionType* memberFn,
                                bool* viable) -> ImplicitConversionSequence {
    *viable = true;
    auto memberCv = memberFn->cvQualifiers();
    auto objectCv =
        traits.get_cv_qualifiers(traits.remove_reference(objectType));
    if (!cv_is_subset_of(objectCv, memberCv)) {
      *viable = false;
      return {};
    }
    auto seq = makeExactMatch(objectType);
    seq.bindsToReference = true;
    seq.referenceCv = memberCv;
    return seq;
  };

  auto rankConversion = [&](const Type* source,
                            const Type* target) -> ImplicitConversionSequence {
    ImplicitConversionSequence seq;
    if (!source || !target) return seq;

    if (traits.is_rvalue_reference(target) &&
        !traits.is_rvalue_reference(source)) {
      return seq;
    }

    auto s = remove_cvref(source);
    auto t = remove_cvref(target);

    if (traits.is_same(s, t)) return makeExactMatch(target);

    auto decayedSource = traits.decay(source);
    if (traits.is_same(decayedSource, t)) return makeExactMatch(target);

    if (stdconv_.isIntegralPromotion(s, t)) {
      seq.rank = ConversionRank::kPromotion;
      seq.steps.push_back({ImplicitCastKind::kIntegralPromotion, target});
      return seq;
    }

    if (stdconv_.isFloatingPointPromotion(s, t)) {
      seq.rank = ConversionRank::kPromotion;
      seq.steps.push_back({ImplicitCastKind::kFloatingPointPromotion, target});
      return seq;
    }

    if (traits.is_null_pointer(s) && traits.is_pointer(t)) {
      seq.rank = ConversionRank::kConversion;
      seq.steps.push_back({ImplicitCastKind::kPointerConversion, target});
      return seq;
    }

    if (traits.is_pointer(s) && traits.is_pointer(t)) {
      auto fromElem = traits.get_element_type(s);
      auto toElem = traits.get_element_type(t);

      if (fromElem && toElem) {
        auto fromCv = traits.get_cv_qualifiers(fromElem);
        auto toCv = traits.get_cv_qualifiers(toElem);

        if (cv_is_subset_of(fromCv, toCv)) {
          auto fromUnqual = traits.remove_cv(fromElem);
          auto toUnqual = traits.remove_cv(toElem);

          if (traits.is_same(fromUnqual, toUnqual)) {
            seq.rank = ConversionRank::kExactMatch;
            seq.steps.push_back(
                {ImplicitCastKind::kQualificationConversion, target});
            return seq;
          }

          if (traits.is_void(toUnqual)) {
            seq.rank = ConversionRank::kConversion;
            seq.steps.push_back({ImplicitCastKind::kPointerConversion, target});
            return seq;
          }

          if (traits.is_class(fromUnqual) && traits.is_class(toUnqual) &&
              traits.is_base_of(toUnqual, fromUnqual)) {
            seq.rank = ConversionRank::kConversion;
            seq.steps.push_back(
                {ImplicitCastKind::kDerivedToBaseConversion, target});
            return seq;
          }
        }
      }
    }

    if ((traits.is_arithmetic(s) ||
         (traits.is_enum(s) && !traits.is_scoped_enum(s))) &&
        traits.is_arithmetic(t)) {
      seq.rank = ConversionRank::kConversion;
      if (traits.is_integral_or_unscoped_enum(s) && traits.is_integral(t)) {
        seq.steps.push_back({ImplicitCastKind::kIntegralConversion, target});
      } else if (traits.is_floating_point(s) && traits.is_floating_point(t)) {
        seq.steps.push_back(
            {ImplicitCastKind::kFloatingPointConversion, target});
      } else {
        seq.steps.push_back(
            {ImplicitCastKind::kFloatingIntegralConversion, target});
      }
      return seq;
    }

    if (traits.is_same(t, control_->getBoolType()) &&
        (traits.is_pointer(s) || traits.is_null_pointer(s) ||
         traits.is_member_pointer(s))) {
      seq.rank = ConversionRank::kConversion;
      seq.steps.push_back({ImplicitCastKind::kBooleanConversion, target});
      return seq;
    }

    return seq;
  };

  auto candidateBetterThan = [](const ViableCandidate& lhs,
                                const ViableCandidate& rhs) -> bool {
    bool lhsBetter = false;

    if (lhs.left.isBetterThan(rhs.left)) {
      lhsBetter = true;
    } else if (rhs.left.isBetterThan(lhs.left)) {
      return false;
    }

    if (lhs.right.has_value() != rhs.right.has_value()) return false;

    if (lhs.right) {
      if (lhs.right->isBetterThan(*rhs.right)) {
        lhsBetter = true;
      } else if (rhs.right->isBetterThan(*lhs.right)) {
        return false;
      }
    }

    return lhsBetter;
  };

  std::vector<ViableCandidate> viable;

  for (auto candidate : candidates) {
    bool isMember = candidate->isImplicitObjectMemberFunction();
    List<TemplateArgumentAST*>* deducedArgsForCandidate = nullptr;

    if (candidate->templateDeclaration() && !candidate->isSpecialization()) {
      if (!leftExpr) continue;

      int operandCount = rightExpr ? (isMember ? 1 : 2) : (isMember ? 0 : 1);
      if (templateCandidateArityRejects(candidate, operandCount)) continue;

      List<ExpressionAST*>* argList = nullptr;
      auto tail = &argList;
      if (!isMember) {
        *tail = make_list_node(arena_, leftExpr);
        tail = &(*tail)->next;
      }
      if (rightExpr) {
        *tail = make_list_node(arena_, rightExpr);
      }

      TemplateArgumentDeduction deduction(unit_);
      auto deducedArgs = deduction.deduce(candidate, argList,
                                          /*explicitTemplateArguments=*/{});
      if (!deducedArgs.has_value()) continue;

      auto instFunc = ASTRewriter::instantiateForArgs(
          unit_, *deducedArgs, candidate, leftExpr->firstSourceLocation(),
          /*argsComplete=*/true,
          /*declarationOnly=*/!functionTemplateHasPackParameter(candidate));
      if (!instFunc) continue;

      candidate = instFunc;
      deducedArgsForCandidate = *deducedArgs;
    }

    bool alreadyViable = false;
    for (const auto& v : viable) {
      if (v.symbol == candidate) {
        alreadyViable = true;
        break;
      }
    }
    if (alreadyViable) continue;

    auto funcType = type_cast<FunctionType>(candidate->type());
    if (!funcType) continue;

    auto params = funcType->parameterTypes();

    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;

    if (rightType) {
      if (isMember) {
        if (params.size() != 1) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !traits.is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        bool objectViable = false;
        left = rankImplicitObject(leftType, funcType, &objectViable);
        if (!objectViable) continue;
        right = rankConversion(rightType, params[0]);
        if (!*right && rightExpr)
          right = stdconv_.computeConversionSequence(rightExpr, params[0]);
      } else {
        if (params.size() != 2) continue;
        left = rankConversion(leftType, params[0]);
        if (!left && leftExpr)
          left = stdconv_.computeConversionSequence(leftExpr, params[0]);
        right = rankConversion(rightType, params[1]);
        if (!*right && rightExpr)
          right = stdconv_.computeConversionSequence(rightExpr, params[1]);
      }
    } else {
      if (isMember) {
        if (!params.empty()) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !traits.is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        bool objectViable = false;
        left = rankImplicitObject(leftType, funcType, &objectViable);
        if (!objectViable) continue;
      } else {
        if (params.size() != 1) continue;
        left = rankConversion(leftType, params[0]);
      }
    }

    if (!left) continue;
    if (rightType && (!right || !*right)) continue;

    viable.push_back({candidate, left, right, deducedArgsForCandidate});
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

  completeDeferredWinnerBody(best->symbol, best->deducedTemplateArgs);
  return best->symbol;
}

auto OverloadResolution::trySelectOperator(
    const std::vector<FunctionSymbol*>& candidates, const Type* type,
    const Type* rightType, ExpressionAST* leftExpr, ExpressionAST* rightExpr)
    -> FunctionSymbol* {
  if (candidates.empty()) return nullptr;
  bool ambiguous = false;
  auto selected = resolveBinaryOperator(candidates, type, rightType, &ambiguous,
                                        leftExpr, rightExpr);
  lastLookupAmbiguous_ = ambiguous;
  return selected;
}

auto OverloadResolution::lookupOperator(const Type* type, TokenKind op,
                                        const Type* rightType,
                                        ExpressionAST* leftExpr,
                                        ExpressionAST* rightExpr)
    -> FunctionSymbol* {
  lastLookupAmbiguous_ = false;

  auto name = control_->getOperatorId(op);
  if (!name) return nullptr;

  std::vector<FunctionSymbol*> candidates;

  if (auto classType = type_cast<ClassType>(traits.remove_cvref(type))) {
    if (auto classSymbol = classType->symbol()) {
      candidates = findCandidates(classSymbol, name);
    }
  }

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
      if (std::find(candidates.begin(), candidates.end(), func) ==
          candidates.end()) {
        candidates.push_back(func);
      }
    }
  }

  return trySelectOperator(candidates, type, rightType, leftExpr, rightExpr);
}
}  // namespace cxx
