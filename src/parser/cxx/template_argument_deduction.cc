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
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {

TemplateArgumentDeduction::TemplateArgumentDeduction(TranslationUnit* unit)
    : unit_(unit), control_(unit->control()), arena_(unit->arena()) {}

auto TemplateArgumentDeduction::deduce(
    FunctionSymbol* func, List<ExpressionAST*>* args,
    List<TemplateArgumentAST*>* explicitTemplateArgs)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  collectTemplateParameters(templateDecl);

  if (!substituteExplicitTemplateArguments(explicitTemplateArgs))
    return std::nullopt;

  if (!deduceFromCall(functionType, args)) return std::nullopt;

  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

void TemplateArgumentDeduction::collectTemplateParameters(
    TemplateDeclarationAST* templateDecl) {
  templateParams_.clear();

  for (auto p : ListView{templateDecl->templateParameterList}) {
    TemplateParameterInfo info;

    if (auto sym = p->symbol) {
      info.typeParameterType = type_cast<TypeParameterType>(sym->type());

      if (auto typeParam = symbol_cast<TypeParameterSymbol>(sym)) {
        (void)typeParam;
        info.kind = TemplateParameterInfo::Kind::kType;
        if (info.typeParameterType)
          info.isPack = info.typeParameterType->isParameterPack();
      } else if (auto nonTypeParam = symbol_cast<NonTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kNonType;
        info.isPack = nonTypeParam->isParameterPack();
      } else if (auto constraintParam =
                     symbol_cast<ConstraintTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kConstraint;
        info.isPack = constraintParam->isParameterPack();
      } else if (symbol_cast<TemplateTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kTemplate;
      }
    }

    if (auto t = ast_cast<TypenameTypeParameterAST>(p)) {
      info.isPack = info.isPack || t->isPack;
      info.hasDefault = t->typeId && t->typeId->type;
    } else if (auto n = ast_cast<NonTypeTemplateParameterAST>(p)) {
      info.isPack = info.isPack || (n->declaration && n->declaration->isPack);
      info.hasDefault = n->declaration && n->declaration->expression;
    } else if (auto c = ast_cast<ConstraintTypeParameterAST>(p)) {
      info.isPack = info.isPack || static_cast<bool>(c->ellipsisLoc);
      info.hasDefault = c->typeId && c->typeId->type;
    } else if (auto tt = ast_cast<TemplateTypeParameterAST>(p)) {
      info.isPack = info.isPack || tt->isPack;
      info.hasDefault = tt->idExpression != nullptr;
    }

    templateParams_.push_back(info);
  }

  auto n = templateParams_.size();
  explicitParamArg_.assign(n, nullptr);
  explicitPackArgs_.assign(n, {});
  deducedTypes_.assign(n, nullptr);
  deducedPacks_.assign(n, {});
}

auto TemplateArgumentDeduction::substituteExplicitTemplateArguments(
    List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool {
  std::vector<TemplateArgumentAST*> explicitArgs;
  for (auto arg : ListView{explicitTemplateArgs}) {
    explicitArgs.push_back(arg);
  }

  int explicitIndex = 0;
  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (explicitIndex >= static_cast<int>(explicitArgs.size())) break;

    if (templateParams_[i].isPack) {
      while (explicitIndex < static_cast<int>(explicitArgs.size())) {
        explicitPackArgs_[i].push_back(explicitArgs[explicitIndex]);
        ++explicitIndex;
      }
      break;
    }

    explicitParamArg_[i] = explicitArgs[explicitIndex];
    ++explicitIndex;
  }

  return explicitIndex == static_cast<int>(explicitArgs.size());
}

auto TemplateArgumentDeduction::isExplicitArgumentCompatible(
    const TemplateParameterInfo& info, TemplateArgumentAST* arg) -> bool {
  if (!arg) return false;

  switch (info.kind) {
    case TemplateParameterInfo::Kind::kType:
    case TemplateParameterInfo::Kind::kTemplate:
    case TemplateParameterInfo::Kind::kConstraint: {
      auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
      return typeArg && typeArg->typeId && typeArg->typeId->type;
    }

    case TemplateParameterInfo::Kind::kNonType: {
      auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg);
      return exprArg && exprArg->expression;
    }

    case TemplateParameterInfo::Kind::kUnknown:
      return false;
  }

  return false;
}

auto TemplateArgumentDeduction::isForwardingReference(const Type* paramType)
    -> bool {
  auto rrefParam = type_cast<RvalueReferenceType>(paramType);
  if (!rrefParam) return false;

  auto rrefElem = unit_->typeTraits().remove_cv(rrefParam->elementType());
  auto paramTpt = type_cast<TypeParameterType>(rrefElem);
  if (!paramTpt) return false;

  return !paramTpt->isParameterPack();
}

auto TemplateArgumentDeduction::deduceTypeFromType(const Type* P, const Type* A)
    -> bool {
  auto bareParam = unit_->typeTraits().remove_cvref(P);
  auto tpt = type_cast<TypeParameterType>(bareParam);

  if (!tpt) {
    if (auto ptrParam =
            type_cast<PointerType>(unit_->typeTraits().remove_cv(bareParam))) {
      CvQualifiers cvP = CvQualifiers::kNone;
      const Type* paramElemBase = ptrParam->elementType();
      if (auto qual = type_cast<QualType>(paramElemBase)) {
        cvP = qual->cvQualifiers();
        paramElemBase = qual->elementType();
      }

      if (auto elemTpt = type_cast<TypeParameterType>(paramElemBase)) {
        const Type* argElemType = nullptr;
        if (auto ptrArg = type_cast<PointerType>(unit_->typeTraits().remove_cv(
                unit_->typeTraits().remove_reference(A)))) {
          argElemType = ptrArg->elementType();
        }

        if (argElemType) {
          CvQualifiers cvA = CvQualifiers::kNone;
          const Type* argElemBase = argElemType;
          if (auto qual = type_cast<QualType>(argElemType)) {
            cvA = qual->cvQualifiers();
            argElemBase = qual->elementType();
          }

          CvQualifiers cvT = CvQualifiers::kNone;
          if (is_const(cvA) && !is_const(cvP)) cvT = cvT | CvQualifiers::kConst;
          if (is_volatile(cvA) && !is_volatile(cvP))
            cvT = cvT | CvQualifiers::kVolatile;

          const Type* deducedT = cvT != CvQualifiers::kNone
                                     ? control_->getQualType(argElemBase, cvT)
                                     : argElemBase;

          auto idx = elemTpt->index();
          if (idx >= 0 && idx < static_cast<int>(templateParams_.size())) {
            if (!deducedTypes_[idx]) {
              deducedTypes_[idx] = deducedT;
            } else if (!unit_->typeTraits().is_same(deducedTypes_[idx],
                                                    deducedT)) {
              return false;
            }
            return true;
          }
        }
      }
    }
    return true;
  }

  auto idx = tpt->index();
  if (idx < 0 || idx >= static_cast<int>(templateParams_.size())) return false;

  const Type* deducedArg = A;

  if (templateParams_[idx].isPack) {
    if (!explicitPackArgs_[idx].empty()) {
      auto explicitPackIndex = static_cast<int>(deducedPacks_[idx].size());
      if (explicitPackIndex >=
          static_cast<int>(explicitPackArgs_[idx].size())) {
        return false;
      }
      if (!isExplicitArgumentCompatible(
              templateParams_[idx],
              explicitPackArgs_[idx][explicitPackIndex])) {
        return false;
      }

      auto explicitTypeArg = ast_cast<TypeTemplateArgumentAST>(
          explicitPackArgs_[idx][explicitPackIndex]);
      if (!explicitTypeArg || !explicitTypeArg->typeId ||
          !explicitTypeArg->typeId->type) {
        return false;
      }

      if (!unit_->typeTraits().is_same(explicitTypeArg->typeId->type,
                                       deducedArg)) {
        return false;
      }
    }

    deducedPacks_[idx].push_back(deducedArg);
    return true;
  }

  if (auto explicitArg = explicitParamArg_[idx]) {
    if (!isExplicitArgumentCompatible(templateParams_[idx], explicitArg))
      return false;

    auto explicitTypeArg = ast_cast<TypeTemplateArgumentAST>(explicitArg);
    if (!explicitTypeArg || !explicitTypeArg->typeId ||
        !explicitTypeArg->typeId->type) {
      return false;
    }

    if (!unit_->typeTraits().is_same(explicitTypeArg->typeId->type, deducedArg))
      return false;

    deducedTypes_[idx] = explicitTypeArg->typeId->type;
    return true;
  }

  if (!deducedTypes_[idx]) {
    deducedTypes_[idx] = deducedArg;
  } else if (!unit_->typeTraits().is_same(deducedTypes_[idx], deducedArg)) {
    return false;
  }

  return true;
}

auto TemplateArgumentDeduction::deduceFromCallArgument(const Type* P,
                                                       const Type* A,
                                                       ExpressionAST* argExpr)
    -> bool {
  if (isForwardingReference(P) && argExpr &&
      argExpr->valueCategory == ValueCategory::kLValue) {
    return deduceTypeFromType(P, unit_->typeTraits().add_lvalue_reference(
                                     unit_->typeTraits().remove_reference(A)));
  }

  return deduceTypeFromType(P, unit_->typeTraits().remove_cvref(A));
}

auto TemplateArgumentDeduction::deduceFromCall(const FunctionType* functionType,
                                               List<ExpressionAST*>* args)
    -> bool {
  auto paramIt = functionType->parameterTypes().begin();
  auto paramEnd = functionType->parameterTypes().end();

  for (auto argIt = args; argIt; argIt = argIt->next) {
    auto argType = argIt->value ? argIt->value->type : nullptr;
    if (!argType) return false;

    if (paramIt == paramEnd) return false;

    auto P = *paramIt;

    if (!deduceFromCallArgument(P, argType, argIt->value)) return false;

    auto bareParam = unit_->typeTraits().remove_cvref(P);
    auto tpt = type_cast<TypeParameterType>(bareParam);
    if (!tpt || !templateParams_[tpt->index()].isPack) {
      ++paramIt;
    }
  }

  return true;
}

auto TemplateArgumentDeduction::checkDeducedArguments() -> bool {
  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (templateParams_[i].isPack) continue;
    if (templateParams_[i].hasDefault) continue;
    if (explicitParamArg_[i]) continue;
    if (!deducedTypes_[i]) return false;
  }
  return true;
}

auto TemplateArgumentDeduction::buildTemplateArgumentList()
    -> std::optional<List<TemplateArgumentAST*>*> {
  List<TemplateArgumentAST*>* templArgList = nullptr;
  auto argListIt = &templArgList;

  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (templateParams_[i].isPack) {
      if (!explicitPackArgs_[i].empty()) {
        if (!deducedPacks_[i].empty() &&
            deducedPacks_[i].size() != explicitPackArgs_[i].size()) {
          return std::nullopt;
        }

        for (auto explicitArg : explicitPackArgs_[i]) {
          if (!isExplicitArgumentCompatible(templateParams_[i], explicitArg))
            return std::nullopt;
          *argListIt = make_list_node<TemplateArgumentAST>(arena_, explicitArg);
          argListIt = &(*argListIt)->next;
        }
        continue;
      }

      for (auto& packType : deducedPacks_[i]) {
        auto typeId = TypeIdAST::create(arena_);
        typeId->type = packType;
        auto typeArg = TypeTemplateArgumentAST::create(arena_);
        typeArg->typeId = typeId;
        *argListIt = make_list_node<TemplateArgumentAST>(arena_, typeArg);
        argListIt = &(*argListIt)->next;
      }
      continue;
    }

    if (auto explicitArg = explicitParamArg_[i]) {
      if (!isExplicitArgumentCompatible(templateParams_[i], explicitArg))
        return std::nullopt;
      *argListIt = make_list_node<TemplateArgumentAST>(arena_, explicitArg);
      argListIt = &(*argListIt)->next;
      continue;
    }

    if (!deducedTypes_[i]) {
      if (templateParams_[i].hasDefault) break;
      return std::nullopt;
    }

    auto typeId = TypeIdAST::create(arena_);
    typeId->type = deducedTypes_[i];
    auto typeArg = TypeTemplateArgumentAST::create(arena_);
    typeArg->typeId = typeId;
    *argListIt = make_list_node<TemplateArgumentAST>(arena_, typeArg);
    argListIt = &(*argListIt)->next;
  }

  return templArgList;
}

}  // namespace cxx
