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
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

namespace {

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

}  // namespace

OverloadResolution::OverloadResolution(TranslationUnit* unit)
    : unit_(unit),
      control_(unit->control()),
      arena_(unit->arena()),
      stdconv_(unit) {}

auto OverloadResolution::initializerListElementType(
    const Type* targetType) const -> const Type* {
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
  return {best[0], false};
}

auto OverloadResolution::resolveConstructor(
    ClassSymbol* classSymbol, const std::vector<ExpressionAST*>& args)
    -> ConstructorResult {
  ConstructorResult result;

  auto argCount = static_cast<int>(args.size());

  for (auto ctor : classSymbol->constructors()) {
    if (ctor->canonical() != ctor) continue;

    auto type = type_cast<FunctionType>(ctor->type());
    if (!type) continue;

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) continue;
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(ctor, paramCount)) continue;
    }

    Candidate cand{ctor};
    cand.viable = true;

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

  auto [bestPtr, ambiguous] = selectBestViableFunction(result.candidates);
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
  std::vector<FunctionSymbol*> result;
  if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
    result.push_back(func);
  } else if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    for (auto func : overloadSet->functions()) result.push_back(func);
  }
  return result;
}

auto OverloadResolution::resolveBinaryOperator(
    const std::vector<FunctionSymbol*>& candidates, const Type* leftType,
    const Type* rightType, bool* ambiguous) const -> FunctionSymbol* {
  if (ambiguous) *ambiguous = false;

  if (candidates.empty()) return nullptr;

  struct ViableCandidate {
    FunctionSymbol* symbol;
    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;
  };

  auto remove_cvref = [&](const Type* type) {
    if (!type) return type;
    return control_->remove_cvref(type);
  };

  auto makeExactMatch = [&](const Type* type) -> ImplicitConversionSequence {
    ImplicitConversionSequence seq;
    seq.rank = ConversionRank::kExactMatch;
    seq.steps.push_back({ImplicitCastKind::kIdentity, type});
    return seq;
  };

  auto rankConversion = [&](const Type* source,
                            const Type* target) -> ImplicitConversionSequence {
    ImplicitConversionSequence seq;
    if (!source || !target) return seq;

    auto s = remove_cvref(source);
    auto t = remove_cvref(target);

    if (control_->is_same(s, t)) return makeExactMatch(target);

    auto decayedSource = control_->decay(source);
    if (control_->is_same(decayedSource, t)) return makeExactMatch(target);

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

    if (control_->is_null_pointer(s) && control_->is_pointer(t)) {
      seq.rank = ConversionRank::kConversion;
      seq.steps.push_back({ImplicitCastKind::kPointerConversion, target});
      return seq;
    }

    if (control_->is_pointer(s) && control_->is_pointer(t)) {
      auto fromElem = control_->get_element_type(s);
      auto toElem = control_->get_element_type(t);

      if (fromElem && toElem) {
        auto fromCv = control_->get_cv_qualifiers(fromElem);
        auto toCv = control_->get_cv_qualifiers(toElem);

        if (cv_is_subset_of(fromCv, toCv)) {
          auto fromUnqual = control_->remove_cv(fromElem);
          auto toUnqual = control_->remove_cv(toElem);

          if (control_->is_same(fromUnqual, toUnqual)) {
            seq.rank = ConversionRank::kExactMatch;
            seq.steps.push_back(
                {ImplicitCastKind::kQualificationConversion, target});
            return seq;
          }

          if (control_->is_void(toUnqual) ||
              (control_->is_class(fromUnqual) && control_->is_class(toUnqual) &&
               control_->is_base_of(toUnqual, fromUnqual))) {
            seq.rank = ConversionRank::kConversion;
            seq.steps.push_back({ImplicitCastKind::kPointerConversion, target});
            return seq;
          }
        }
      }
    }

    if ((control_->is_arithmetic(s) ||
         (control_->is_enum(s) && !control_->is_scoped_enum(s))) &&
        control_->is_arithmetic(t)) {
      seq.rank = ConversionRank::kConversion;
      if (control_->is_integral_or_unscoped_enum(s) &&
          control_->is_integral(t)) {
        seq.steps.push_back({ImplicitCastKind::kIntegralConversion, target});
      } else if (control_->is_floating_point(s) &&
                 control_->is_floating_point(t)) {
        seq.steps.push_back(
            {ImplicitCastKind::kFloatingPointConversion, target});
      } else {
        seq.steps.push_back(
            {ImplicitCastKind::kFloatingIntegralConversion, target});
      }
      return seq;
    }

    if (control_->is_same(t, control_->getBoolType())) {
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
    auto funcType = type_cast<FunctionType>(candidate->type());
    if (!funcType) continue;

    auto params = funcType->parameterTypes();
    bool isMember = candidate->parent() && candidate->parent()->isClass();

    ImplicitConversionSequence left;
    std::optional<ImplicitConversionSequence> right;

    if (rightType) {
      if (isMember) {
        if (params.size() != 1) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !control_->is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        left = makeExactMatch(leftType);
        right = rankConversion(rightType, params[0]);
      } else {
        if (params.size() != 2) continue;
        left = rankConversion(leftType, params[0]);
        right = rankConversion(rightType, params[1]);
      }
    } else {
      if (isMember) {
        if (!params.empty()) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !control_->is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        left = makeExactMatch(leftType);
      } else {
        if (params.size() != 1) continue;
        left = rankConversion(leftType, params[0]);
      }
    }

    if (!left) continue;
    if (rightType && (!right || !*right)) continue;

    viable.push_back({candidate, left, right});
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

    foundEquivalent = true;
  }

  if (foundEquivalent) {
    if (ambiguous) *ambiguous = true;
    return nullptr;
  }

  return best->symbol;
}

auto OverloadResolution::trySelectOperator(
    const std::vector<FunctionSymbol*>& candidates, const Type* type,
    const Type* rightType) -> FunctionSymbol* {
  if (candidates.empty()) return nullptr;
  bool ambiguous = false;
  auto selected =
      resolveBinaryOperator(candidates, type, rightType, &ambiguous);
  lastLookupAmbiguous_ = ambiguous;
  return selected;
}

auto OverloadResolution::lookupOperator(const Type* type, TokenKind op,
                                        const Type* rightType)
    -> FunctionSymbol* {
  lastLookupAmbiguous_ = false;

  auto name = control_->getOperatorId(op);
  if (!name) return nullptr;

  if (auto classType = type_cast<ClassType>(control_->remove_cvref(type))) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return nullptr;

    auto candidates = findCandidates(classSymbol, name);
    if (!candidates.empty())
      return trySelectOperator(candidates, type, rightType);
  }

  {
    std::vector<const Type*> argTypes{type};
    if (rightType) argTypes.push_back(rightType);
    if (auto result = trySelectOperator(argumentDependentLookup(name, argTypes),
                                        type, rightType))
      return result;
  }

  return nullptr;
}

}  // namespace cxx
