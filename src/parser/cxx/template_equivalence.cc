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
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {

namespace {
[[nodiscard]] auto unqualifiedIdsStructurallyEquivalent(TranslationUnit* unit,
                                                        UnqualifiedIdAST* a,
                                                        UnqualifiedIdAST* b)
    -> bool;

[[nodiscard]] auto typeIdsStructurallyEquivalent(TranslationUnit* unit,
                                                 TypeIdAST* a, TypeIdAST* b)
    -> bool;

[[nodiscard]] auto typesStructurallyEquivalent(TranslationUnit* unit,
                                               const Type* a, const Type* b)
    -> bool {
  if (!a || !b) return false;
  if (a == b) return true;

  auto aParam = type_cast<TypeParameterType>(a);
  auto bParam = type_cast<TypeParameterType>(b);
  if (aParam && bParam) {
    return aParam->depth() == bParam->depth() &&
           aParam->index() == bParam->index();
  }
  if (aParam || bParam) return false;

  return unit->typeTraits().is_same(a, b);
}

[[nodiscard]] auto soleTypeSpecifier(ParameterDeclarationAST* decl)
    -> SpecifierAST* {
  if (!decl) return nullptr;
  SpecifierAST* found = nullptr;
  for (auto spec : ListView{decl->typeSpecifierList}) {
    if (found) return nullptr;
    found = spec;
  }
  return found;
}

[[nodiscard]] auto templateQualifiedNameStructurallyEquivalent(
    TranslationUnit* unit, NestedNameSpecifierAST* aNns,
    UnqualifiedIdAST* aName, NestedNameSpecifierAST* bNns,
    UnqualifiedIdAST* bName) -> bool {
  auto aTns = ast_cast<TemplateNestedNameSpecifierAST>(aNns);
  auto bTns = ast_cast<TemplateNestedNameSpecifierAST>(bNns);
  if (!aTns || !bTns) return false;
  if (aTns->nestedNameSpecifier || bTns->nestedNameSpecifier) return false;

  auto aTemplateId = aTns->templateId;
  auto bTemplateId = bTns->templateId;
  if (!aTemplateId || !bTemplateId) return false;
  if (!aTemplateId->symbol || aTemplateId->symbol != bTemplateId->symbol)
    return false;
  if (!areTemplateArgumentListsSyntacticallyEquivalent(
          unit, aTemplateId->templateArgumentList,
          bTemplateId->templateArgumentList))
    return false;

  auto aNameId = ast_cast<NameIdAST>(aName);
  auto bNameId = ast_cast<NameIdAST>(bName);
  return aNameId && bNameId && aNameId->identifier == bNameId->identifier;
}

[[nodiscard]] auto namedTypeSpecifiersStructurallyEquivalent(
    TranslationUnit* unit, NamedTypeSpecifierAST* a, NamedTypeSpecifierAST* b)
    -> bool {
  if (!a || !b) return false;
  if (a->nestedNameSpecifier || b->nestedNameSpecifier) return false;

  if (ast_cast<NameIdAST>(a->unqualifiedId) &&
      ast_cast<NameIdAST>(b->unqualifiedId)) {
    return typesStructurallyEquivalent(unit,
                                       a->symbol ? a->symbol->type() : nullptr,
                                       b->symbol ? b->symbol->type() : nullptr);
  }

  auto aTid = ast_cast<SimpleTemplateIdAST>(a->unqualifiedId);
  auto bTid = ast_cast<SimpleTemplateIdAST>(b->unqualifiedId);
  if (!aTid || !bTid) return false;
  if (!aTid->symbol || aTid->symbol != bTid->symbol) return false;

  return areTemplateArgumentListsEquivalent(unit, aTid->templateArgumentList,
                                            bTid->templateArgumentList);
}

[[nodiscard]] auto namedTypeSpecifiersSyntacticallyEquivalent(
    TranslationUnit* unit, NamedTypeSpecifierAST* a, NamedTypeSpecifierAST* b)
    -> bool {
  if (!a || !b) return false;
  auto aScope =
      a->nestedNameSpecifier ? a->nestedNameSpecifier->symbol : nullptr;
  auto bScope =
      b->nestedNameSpecifier ? b->nestedNameSpecifier->symbol : nullptr;
  if (aScope != bScope) return false;

  auto aName = ast_cast<NameIdAST>(a->unqualifiedId);
  auto bName = ast_cast<NameIdAST>(b->unqualifiedId);
  if (aName || bName) {
    if (!aName || !bName || aName->identifier != bName->identifier)
      return false;
    return typesStructurallyEquivalent(unit,
                                       a->symbol ? a->symbol->type() : nullptr,
                                       b->symbol ? b->symbol->type() : nullptr);
  }

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a->unqualifiedId);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b->unqualifiedId);
  if (!aTemplateId || !bTemplateId ||
      aTemplateId->identifier != bTemplateId->identifier)
    return false;
  return areTemplateArgumentListsSyntacticallyEquivalent(
      unit, aTemplateId->templateArgumentList,
      bTemplateId->templateArgumentList);
}

auto unqualifiedIdsStructurallyEquivalent(TranslationUnit* unit,
                                          UnqualifiedIdAST* a,
                                          UnqualifiedIdAST* b) -> bool {
  if (a == b) return true;
  auto aName = ast_cast<NameIdAST>(a);
  auto bName = ast_cast<NameIdAST>(b);
  if (aName || bName)
    return aName && bName && aName->identifier == bName->identifier;

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b);
  if (!aTemplateId || !bTemplateId ||
      aTemplateId->identifier != bTemplateId->identifier)
    return false;
  if (aTemplateId->symbol && bTemplateId->symbol &&
      aTemplateId->symbol != bTemplateId->symbol)
    return false;
  return areTemplateArgumentListsEquivalent(unit,
                                            aTemplateId->templateArgumentList,
                                            bTemplateId->templateArgumentList);
}

[[nodiscard]] auto typenameSpecifiersStructurallyEquivalent(
    TranslationUnit* unit, TypenameSpecifierAST* a, TypenameSpecifierAST* b)
    -> bool {
  if (!a || !b) return false;
  return templateQualifiedNameStructurallyEquivalent(
      unit, a->nestedNameSpecifier, a->unqualifiedId, b->nestedNameSpecifier,
      b->unqualifiedId);
}

auto typeIdsStructurallyEquivalent(TranslationUnit* unit, TypeIdAST* a,
                                   TypeIdAST* b) -> bool {
  if (!a || !b) return false;
  auto aSpec = a->typeSpecifierList;
  auto bSpec = b->typeSpecifierList;
  for (; aSpec && bSpec; aSpec = aSpec->next, bSpec = bSpec->next) {
    if (aSpec->value->kind() != bSpec->value->kind()) return false;
    if (auto aNamed = ast_cast<NamedTypeSpecifierAST>(aSpec->value)) {
      if (!namedTypeSpecifiersSyntacticallyEquivalent(
              unit, aNamed, ast_cast<NamedTypeSpecifierAST>(bSpec->value)))
        return false;
      continue;
    }
    if (auto aTypename = ast_cast<TypenameSpecifierAST>(aSpec->value)) {
      if (!typenameSpecifiersStructurallyEquivalent(
              unit, aTypename, ast_cast<TypenameSpecifierAST>(bSpec->value)))
        return false;
      continue;
    }
    if (auto aIntegral = ast_cast<IntegralTypeSpecifierAST>(aSpec->value)) {
      auto bIntegral = ast_cast<IntegralTypeSpecifierAST>(bSpec->value);
      if (!bIntegral || aIntegral->specifier != bIntegral->specifier)
        return false;
      continue;
    }
    if (auto aDecltype = ast_cast<DecltypeSpecifierAST>(aSpec->value)) {
      auto bDecltype = ast_cast<DecltypeSpecifierAST>(bSpec->value);
      if (!bDecltype || !areExpressionsEquivalent(unit, aDecltype->expression,
                                                  bDecltype->expression))
        return false;
    }
  }
  if (aSpec || bSpec) return false;
  return typesStructurallyEquivalent(unit, a->type, b->type);
}

}  // namespace

auto areExpressionsEquivalent(TranslationUnit* unit, ExpressionAST* a,
                              ExpressionAST* b) -> bool {
  if (a == b) return true;
  if (!a || !b) return false;

  if (auto nested = ast_cast<NestedExpressionAST>(a)) {
    return areExpressionsEquivalent(unit, nested->expression, b);
  }
  if (auto nested = ast_cast<NestedExpressionAST>(b)) {
    return areExpressionsEquivalent(unit, a, nested->expression);
  }
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(a)) {
    return areExpressionsEquivalent(unit, cast->expression, b);
  }
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(b)) {
    return areExpressionsEquivalent(unit, a, cast->expression);
  }

  if (auto aLit = ast_cast<IntLiteralExpressionAST>(a)) {
    auto bLit = ast_cast<IntLiteralExpressionAST>(b);
    if (!bLit || !aLit->literal || !bLit->literal) return false;
    return aLit->literal->integerValue() == bLit->literal->integerValue();
  }

  if (auto aLit = ast_cast<BoolLiteralExpressionAST>(a)) {
    auto bLit = ast_cast<BoolLiteralExpressionAST>(b);
    return bLit && aLit->isTrue == bLit->isTrue;
  }

  if (auto aSizeofType = ast_cast<SizeofTypeExpressionAST>(a)) {
    auto bSizeofType = ast_cast<SizeofTypeExpressionAST>(b);
    return bSizeofType &&
           typesStructurallyEquivalent(
               unit, aSizeofType->typeId ? aSizeofType->typeId->type : nullptr,
               bSizeofType->typeId ? bSizeofType->typeId->type : nullptr);
  }

  if (auto aSizeof = ast_cast<SizeofExpressionAST>(a)) {
    auto bSizeof = ast_cast<SizeofExpressionAST>(b);
    return bSizeof && areExpressionsEquivalent(unit, aSizeof->expression,
                                               bSizeof->expression);
  }

  if (auto aSizeofPack = ast_cast<SizeofPackExpressionAST>(a)) {
    auto bSizeofPack = ast_cast<SizeofPackExpressionAST>(b);
    if (!bSizeofPack) return false;
    auto aPack = template_parameter_info(aSizeofPack->symbol);
    auto bPack = template_parameter_info(bSizeofPack->symbol);
    if (aPack || bPack) {
      return aPack && bPack && aPack->depth == bPack->depth &&
             aPack->index == bPack->index;
    }
    return aSizeofPack->symbol == bSizeofPack->symbol;
  }

  if (auto aTrait = ast_cast<TypeTraitExpressionAST>(a)) {
    auto bTrait = ast_cast<TypeTraitExpressionAST>(b);
    if (!bTrait || aTrait->typeTrait != bTrait->typeTrait) return false;
    auto aTypeId = aTrait->typeIdList;
    auto bTypeId = bTrait->typeIdList;
    for (; aTypeId && bTypeId;
         aTypeId = aTypeId->next, bTypeId = bTypeId->next) {
      if (!typeIdsStructurallyEquivalent(unit, aTypeId->value, bTypeId->value))
        return false;
    }
    return !aTypeId && !bTypeId;
  }

  if (auto aUnary = ast_cast<UnaryExpressionAST>(a)) {
    auto bUnary = ast_cast<UnaryExpressionAST>(b);
    return bUnary && aUnary->op == bUnary->op &&
           areExpressionsEquivalent(unit, aUnary->expression,
                                    bUnary->expression);
  }

  if (auto aBinary = ast_cast<BinaryExpressionAST>(a)) {
    auto bBinary = ast_cast<BinaryExpressionAST>(b);
    return bBinary && aBinary->op == bBinary->op &&
           areExpressionsEquivalent(unit, aBinary->leftExpression,
                                    bBinary->leftExpression) &&
           areExpressionsEquivalent(unit, aBinary->rightExpression,
                                    bBinary->rightExpression);
  }

  if (auto aId = ast_cast<IdExpressionAST>(a)) {
    auto bId = ast_cast<IdExpressionAST>(b);
    if (!bId) return false;

    auto aNttp = symbol_cast<NonTypeParameterSymbol>(aId->symbol);
    auto bNttp = symbol_cast<NonTypeParameterSymbol>(bId->symbol);
    if (aNttp || bNttp) {
      return aNttp && bNttp && aNttp->depth() == bNttp->depth() &&
             aNttp->index() == bNttp->index();
    }

    if (ast_cast<TemplateNestedNameSpecifierAST>(aId->nestedNameSpecifier)) {
      return templateQualifiedNameStructurallyEquivalent(
          unit, aId->nestedNameSpecifier, aId->unqualifiedId,
          bId->nestedNameSpecifier, bId->unqualifiedId);
    }

    if (aId->nestedNameSpecifier || bId->nestedNameSpecifier) return false;
    auto aTid = ast_cast<SimpleTemplateIdAST>(aId->unqualifiedId);
    auto bTid = ast_cast<SimpleTemplateIdAST>(bId->unqualifiedId);
    if (!aTid || !bTid) return false;
    if (!aTid->symbol || aTid->symbol != bTid->symbol) return false;

    return areTemplateArgumentListsSyntacticallyEquivalent(
        unit, aTid->templateArgumentList, bTid->templateArgumentList);
  }

  return false;
}

namespace {

[[nodiscard]] auto nonTypeParameterTypesEquivalent(
    TranslationUnit* unit, NonTypeTemplateParameterAST* a,
    NonTypeTemplateParameterAST* b) -> bool {
  if (!a || !b || !a->declaration || !b->declaration) return false;

  if (a->declaration->type && b->declaration->type &&
      !isDependent(unit, a->declaration->type)) {
    return unit->typeTraits().is_same(a->declaration->type,
                                      b->declaration->type);
  }

  auto aSpec = soleTypeSpecifier(a->declaration);
  auto bSpec = soleTypeSpecifier(b->declaration);

  if (auto aNamed = ast_cast<NamedTypeSpecifierAST>(aSpec)) {
    return namedTypeSpecifiersStructurallyEquivalent(
        unit, aNamed, ast_cast<NamedTypeSpecifierAST>(bSpec));
  }

  if (auto aTypename = ast_cast<TypenameSpecifierAST>(aSpec)) {
    return typenameSpecifiersStructurallyEquivalent(
        unit, aTypename, ast_cast<TypenameSpecifierAST>(bSpec));
  }

  return false;
}

}  // namespace

auto typesEquivalentModuloOwnHeadDepth(TranslationUnit* unit, const Type* lhs,
                                       const Type* rhs, int lhsDepth,
                                       int rhsDepth, int ownParamCount)
    -> bool {
  if (!lhs || !rhs) return lhs == rhs;

  auto lhsInfo = getTypeParamInfo(lhs);
  auto rhsInfo = getTypeParamInfo(rhs);
  if (lhsInfo || rhsInfo) {
    if (!lhsInfo || !rhsInfo) return false;
    if (lhsInfo->depth == lhsDepth && lhsInfo->index < ownParamCount) {
      return rhsInfo->depth == rhsDepth && rhsInfo->index == lhsInfo->index &&
             rhsInfo->isPack == lhsInfo->isPack;
    }
    return lhsInfo->depth == rhsInfo->depth &&
           lhsInfo->index == rhsInfo->index &&
           lhsInfo->isPack == rhsInfo->isPack;
  }

  if (auto lhsQual = type_cast<QualType>(lhs)) {
    auto rhsQual = type_cast<QualType>(rhs);
    if (!rhsQual || lhsQual->cvQualifiers() != rhsQual->cvQualifiers())
      return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsQual->elementType(),
                                             rhsQual->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsPtr = type_cast<PointerType>(lhs)) {
    auto rhsPtr = type_cast<PointerType>(rhs);
    if (!rhsPtr) return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsPtr->elementType(),
                                             rhsPtr->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsRef = type_cast<LvalueReferenceType>(lhs)) {
    auto rhsRef = type_cast<LvalueReferenceType>(rhs);
    if (!rhsRef) return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsRef->elementType(),
                                             rhsRef->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsRef = type_cast<RvalueReferenceType>(lhs)) {
    auto rhsRef = type_cast<RvalueReferenceType>(rhs);
    if (!rhsRef) return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsRef->elementType(),
                                             rhsRef->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsArr = type_cast<BoundedArrayType>(lhs)) {
    auto rhsArr = type_cast<BoundedArrayType>(rhs);
    if (!rhsArr || lhsArr->size() != rhsArr->size()) return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsArr->elementType(),
                                             rhsArr->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsArr = type_cast<UnboundedArrayType>(lhs)) {
    auto rhsArr = type_cast<UnboundedArrayType>(rhs);
    if (!rhsArr) return false;
    return typesEquivalentModuloOwnHeadDepth(unit, lhsArr->elementType(),
                                             rhsArr->elementType(), lhsDepth,
                                             rhsDepth, ownParamCount);
  }
  if (auto lhsFn = type_cast<FunctionType>(lhs)) {
    auto rhsFn = type_cast<FunctionType>(rhs);
    if (!rhsFn) return false;
    if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;
    if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
    if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;

    const auto& lhsParams = lhsFn->parameterTypes();
    const auto& rhsParams = rhsFn->parameterTypes();
    if (lhsParams.size() != rhsParams.size()) return false;

    if (!typesEquivalentModuloOwnHeadDepth(unit, lhsFn->returnType(),
                                           rhsFn->returnType(), lhsDepth,
                                           rhsDepth, ownParamCount))
      return false;

    for (std::size_t i = 0; i < lhsParams.size(); ++i) {
      if (!typesEquivalentModuloOwnHeadDepth(unit, lhsParams[i], rhsParams[i],
                                             lhsDepth, rhsDepth, ownParamCount))
        return false;
    }
    return true;
  }
  if (auto lhsClass = type_cast<ClassType>(lhs)) {
    auto rhsClass = type_cast<ClassType>(rhs);
    if (!rhsClass) return false;
    auto lhsSym = lhsClass->symbol();
    auto rhsSym = rhsClass->symbol();
    if (!lhsSym || !rhsSym) return false;
    if (lhsSym == rhsSym) return true;

    if (lhsSym->isSpecialization() && rhsSym->isSpecialization() &&
        lhsSym->primaryTemplateSymbol() == rhsSym->primaryTemplateSymbol()) {
      auto lhsArgs = lhsSym->templateArguments();
      auto rhsArgs = rhsSym->templateArguments();
      if (lhsArgs.size() != rhsArgs.size()) return false;

      for (std::size_t i = 0; i < lhsArgs.size(); ++i) {
        auto lhsArgType = std::get_if<const Type*>(&lhsArgs[i]);
        auto rhsArgType = std::get_if<const Type*>(&rhsArgs[i]);
        if (lhsArgType && rhsArgType) {
          if (!typesEquivalentModuloOwnHeadDepth(unit, *lhsArgType, *rhsArgType,
                                                 lhsDepth, rhsDepth,
                                                 ownParamCount))
            return false;
          continue;
        }
        if (lhsArgs[i] == rhsArgs[i]) continue;
        return false;
      }
      return true;
    }
    return false;
  }

  return unit->typeTraits().is_same(lhs, rhs);
}

namespace {

using TypeIdEquivalence = auto (*)(TranslationUnit*, TypeIdAST*, TypeIdAST*)
    -> bool;

auto walkTemplateArgumentLists(TranslationUnit* unit,
                               List<TemplateArgumentAST*>* a,
                               List<TemplateArgumentAST*>* b,
                               TypeIdEquivalence equivalentTypeIds) -> bool {
  for (; a && b; a = a->next, b = b->next) {
    auto typeA = ast_cast<TypeTemplateArgumentAST>(a->value);
    auto typeB = ast_cast<TypeTemplateArgumentAST>(b->value);
    if (typeA || typeB) {
      if (!typeA || !typeB ||
          !equivalentTypeIds(unit, typeA->typeId, typeB->typeId))
        return false;
      continue;
    }

    auto expressionA = ast_cast<ExpressionTemplateArgumentAST>(a->value);
    auto expressionB = ast_cast<ExpressionTemplateArgumentAST>(b->value);
    if (!expressionA || !expressionB ||
        !areExpressionsEquivalent(unit, expressionA->expression,
                                  expressionB->expression))
      return false;
  }

  return !a && !b;
}

}  // namespace

auto areTemplateArgumentListsEquivalent(TranslationUnit* unit,
                                        List<TemplateArgumentAST*>* a,
                                        List<TemplateArgumentAST*>* b) -> bool {
  auto equivalentTypeIds = [](TranslationUnit* unit, TypeIdAST* a,
                              TypeIdAST* b) {
    return a && b && typesStructurallyEquivalent(unit, a->type, b->type);
  };
  return walkTemplateArgumentLists(unit, a, b, equivalentTypeIds);
}

auto areTemplateArgumentListsSyntacticallyEquivalent(
    TranslationUnit* unit, List<TemplateArgumentAST*>* a,
    List<TemplateArgumentAST*>* b) -> bool {
  return walkTemplateArgumentLists(unit, a, b, typeIdsStructurallyEquivalent);
}

auto areTemplateParameterListsEquivalent(TranslationUnit* unit,
                                         List<TemplateParameterAST*>* aIt,
                                         List<TemplateParameterAST*>* bIt)
    -> bool {
  for (; aIt && bIt; aIt = aIt->next, bIt = bIt->next) {
    auto aParam = aIt->value;
    auto bParam = bIt->value;
    if (aParam->kind() != bParam->kind()) return false;

    auto aTypename = ast_cast<TypenameTypeParameterAST>(aParam);
    auto bTypename = ast_cast<TypenameTypeParameterAST>(bParam);
    if (aTypename && bTypename && aTypename->isPack != bTypename->isPack)
      return false;

    auto aConstraint = ast_cast<ConstraintTypeParameterAST>(aParam);
    auto bConstraint = ast_cast<ConstraintTypeParameterAST>(bParam);
    if (aConstraint || bConstraint) {
      if (!aConstraint || !bConstraint) return false;
      auto aTypeConstraint = aConstraint->typeConstraint;
      auto bTypeConstraint = bConstraint->typeConstraint;
      if (!aTypeConstraint || !bTypeConstraint) return false;
      auto aSymbol = symbol_cast<TypeParameterSymbol>(aConstraint->symbol);
      auto bSymbol = symbol_cast<TypeParameterSymbol>(bConstraint->symbol);
      if ((aSymbol && aSymbol->isParameterPack()) !=
          (bSymbol && bSymbol->isParameterPack()))
        return false;
      if (aTypeConstraint->identifier != bTypeConstraint->identifier)
        return false;
      auto aScope = aTypeConstraint->nestedNameSpecifier
                        ? aTypeConstraint->nestedNameSpecifier->symbol
                        : nullptr;
      auto bScope = bTypeConstraint->nestedNameSpecifier
                        ? bTypeConstraint->nestedNameSpecifier->symbol
                        : nullptr;
      if (aScope != bScope) return false;
      if (!areTemplateArgumentListsSyntacticallyEquivalent(
              unit, aTypeConstraint->templateArgumentList,
              bTypeConstraint->templateArgumentList))
        return false;
    }

    auto aNonType = ast_cast<NonTypeTemplateParameterAST>(aParam);
    auto bNonType = ast_cast<NonTypeTemplateParameterAST>(bParam);
    if (aNonType && bNonType) {
      auto aSymbol = symbol_cast<NonTypeParameterSymbol>(aNonType->symbol);
      auto bSymbol = symbol_cast<NonTypeParameterSymbol>(bNonType->symbol);
      if ((aSymbol && aSymbol->isParameterPack()) !=
          (bSymbol && bSymbol->isParameterPack()))
        return false;
      if (!nonTypeParameterTypesEquivalent(unit, aNonType, bNonType))
        return false;
    }

    auto aTemplate = ast_cast<TemplateTypeParameterAST>(aParam);
    auto bTemplate = ast_cast<TemplateTypeParameterAST>(bParam);
    if (aTemplate && bTemplate) {
      if (aTemplate->isPack != bTemplate->isPack) return false;
      if (!areTemplateParameterListsEquivalent(
              unit, aTemplate->templateParameterList,
              bTemplate->templateParameterList))
        return false;
      if (!aTemplate->requiresClause || !bTemplate->requiresClause) {
        if (aTemplate->requiresClause != bTemplate->requiresClause)
          return false;
      } else if (!areExpressionsEquivalent(
                     unit, aTemplate->requiresClause->expression,
                     bTemplate->requiresClause->expression)) {
        return false;
      }
    }
  }

  return !aIt && !bIt;
}

auto areTemplateParameterListsEquivalentForPartialOrdering(
    TranslationUnit* unit, List<TemplateParameterAST*>* aIt,
    List<TemplateParameterAST*>* bIt) -> bool {
  for (; aIt && bIt; aIt = aIt->next, bIt = bIt->next) {
    auto a = aIt->value;
    auto b = bIt->value;

    const bool aType = ast_cast<TypenameTypeParameterAST>(a) ||
                       ast_cast<ConstraintTypeParameterAST>(a);
    const bool bType = ast_cast<TypenameTypeParameterAST>(b) ||
                       ast_cast<ConstraintTypeParameterAST>(b);
    if (aType || bType) {
      if (!aType || !bType) return false;
      auto aInfo = template_parameter_info(a->symbol);
      auto bInfo = template_parameter_info(b->symbol);
      if (!aInfo || !bInfo || aInfo->isPack != bInfo->isPack) return false;
      continue;
    }

    auto aNonType = ast_cast<NonTypeTemplateParameterAST>(a);
    auto bNonType = ast_cast<NonTypeTemplateParameterAST>(b);
    if (aNonType || bNonType) {
      if (!aNonType || !bNonType ||
          !nonTypeParameterTypesEquivalent(unit, aNonType, bNonType))
        return false;
      auto aInfo = template_parameter_info(a->symbol);
      auto bInfo = template_parameter_info(b->symbol);
      if (!aInfo || !bInfo || aInfo->isPack != bInfo->isPack) return false;
      continue;
    }

    auto aTemplate = ast_cast<TemplateTypeParameterAST>(a);
    auto bTemplate = ast_cast<TemplateTypeParameterAST>(b);
    if (!aTemplate || !bTemplate || aTemplate->isPack != bTemplate->isPack)
      return false;
    if (!areTemplateParameterListsEquivalentForPartialOrdering(
            unit, aTemplate->templateParameterList,
            bTemplate->templateParameterList))
      return false;
  }

  return !aIt && !bIt;
}

auto areTypesEquivalentForPartialOrdering(TranslationUnit* unit, const Type* a,
                                          const Type* b,
                                          TemplateDeclarationAST* aTemplate,
                                          TemplateDeclarationAST* bTemplate)
    -> bool {
  if (!aTemplate || !bTemplate) return false;

  int parameterCount = 0;
  for (auto parameter : ListView{aTemplate->templateParameterList}) {
    (void)parameter;
    ++parameterCount;
  }

  return typesEquivalentModuloOwnHeadDepth(unit, a, b, aTemplate->depth,
                                           bTemplate->depth, parameterCount);
}

auto trailingRequiresClausesEquivalent(TranslationUnit* unit,
                                       RequiresClauseAST* a,
                                       RequiresClauseAST* b) -> bool {
  if (!a || !b) return a == b;
  return areExpressionsEquivalent(unit, a->expression, b->expression);
}

auto areTemplateHeadsEquivalentForRedeclaration(TranslationUnit* unit,
                                                TemplateDeclarationAST* a,
                                                TemplateDeclarationAST* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!areTemplateParameterListsEquivalent(unit, a->templateParameterList,
                                           b->templateParameterList))
    return false;
  if (!a->requiresClause || !b->requiresClause)
    return a->requiresClause == b->requiresClause;
  return areExpressionsEquivalent(unit, a->requiresClause->expression,
                                  b->requiresClause->expression);
}

}  // namespace cxx
