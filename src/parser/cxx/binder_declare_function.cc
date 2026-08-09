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
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
[[nodiscard]] auto areTemplateArgumentListsEquivalent(
    TranslationUnit* unit, List<TemplateArgumentAST*>* a,
    List<TemplateArgumentAST*>* b) -> bool;

namespace {
namespace {
[[nodiscard]] auto expressionsStructurallyEquivalent(TranslationUnit* unit,
                                                     ExpressionAST* a,
                                                     ExpressionAST* b) -> bool;

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
      if (!bDecltype || !expressionsStructurallyEquivalent(
                            unit, aDecltype->expression, bDecltype->expression))
        return false;
    }
  }
  if (aSpec || bSpec) return false;
  return typesStructurallyEquivalent(unit, a->type, b->type);
}

auto expressionsStructurallyEquivalent(TranslationUnit* unit, ExpressionAST* a,
                                       ExpressionAST* b) -> bool {
  if (!a || !b) return false;

  if (auto nested = ast_cast<NestedExpressionAST>(a)) {
    return expressionsStructurallyEquivalent(unit, nested->expression, b);
  }
  if (auto nested = ast_cast<NestedExpressionAST>(b)) {
    return expressionsStructurallyEquivalent(unit, a, nested->expression);
  }
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(a)) {
    return expressionsStructurallyEquivalent(unit, cast->expression, b);
  }
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(b)) {
    return expressionsStructurallyEquivalent(unit, a, cast->expression);
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
    return bSizeof && expressionsStructurallyEquivalent(
                          unit, aSizeof->expression, bSizeof->expression);
  }

  if (auto aUnary = ast_cast<UnaryExpressionAST>(a)) {
    auto bUnary = ast_cast<UnaryExpressionAST>(b);
    return bUnary && aUnary->op == bUnary->op &&
           expressionsStructurallyEquivalent(unit, aUnary->expression,
                                             bUnary->expression);
  }

  if (auto aBinary = ast_cast<BinaryExpressionAST>(a)) {
    auto bBinary = ast_cast<BinaryExpressionAST>(b);
    return bBinary && aBinary->op == bBinary->op &&
           expressionsStructurallyEquivalent(unit, aBinary->leftExpression,
                                             bBinary->leftExpression) &&
           expressionsStructurallyEquivalent(unit, aBinary->rightExpression,
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
}  // namespace

auto areFunctionSignaturesEquivalentForRedeclaration(TranslationUnit* unit,
                                                     const Type* lhs,
                                                     const Type* rhs) -> bool {
  if (!unit || !lhs || !rhs) return false;
  if (unit->typeTraits().is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  if (!unit->typeTraits().is_same(lhsFn->returnType(), rhsFn->returnType())) {
    auto lhsCore = unit->typeTraits().remove_cvref(lhsFn->returnType());
    auto rhsCore = unit->typeTraits().remove_cvref(rhsFn->returnType());
    auto lhsUnresolved = type_cast<UnresolvedNameType>(lhsCore);
    auto rhsUnresolved = type_cast<UnresolvedNameType>(rhsCore);
    if (!lhsUnresolved || !rhsUnresolved) return false;
    if (to_string(lhsUnresolved) != to_string(rhsUnresolved)) return false;
  }
  if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
  if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;
  if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;

  const auto& lhsParams = lhsFn->parameterTypes();
  const auto& rhsParams = rhsFn->parameterTypes();
  if (lhsParams.size() != rhsParams.size()) return false;

  for (std::size_t i = 0; i < lhsParams.size(); ++i) {
    if (!areRedeclarationTypesCompatible(unit, lhsParams[i], rhsParams[i])) {
      return false;
    }
  }

  return true;
}
}  // namespace

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
        !expressionsStructurallyEquivalent(unit, expressionA->expression,
                                           expressionB->expression))
      return false;
  }

  return !a && !b;
}

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
      } else if (!expressionsStructurallyEquivalent(
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
  return expressionsStructurallyEquivalent(unit, a->expression, b->expression);
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
  return expressionsStructurallyEquivalent(unit, a->requiresClause->expression,
                                           b->requiresClause->expression);
}

struct [[nodiscard]] Binder::DeclareFunction {
  Binder& binder;
  DeclaratorAST* declarator = nullptr;
  const Decl& decl;
  FunctionDeclaratorChunkAST* functionDeclarator = nullptr;
  FunctionSymbol* functionSymbol = nullptr;
  FunctionSymbol* shadowedFunction = nullptr;

  auto control() const -> Control* { return binder.control(); }
  auto scope() const -> ScopeSymbol* { return binder.scope(); }

  auto isTemplateFunction() const -> bool {
    return scope()->isTemplateParameters();
  }

  auto isDestructor() const -> bool {
    return name_cast<DestructorId>(decl.getName()) != nullptr;
  }

  auto declaringScopeForFunction() const -> ScopeSymbol*;
  void mergeAsCRedeclaration(FunctionSymbol* otherFunction);
  auto mergeWithMatchingOverload(OverloadSetSymbol* overloadSet) -> bool;
  void checkCRedeclaration(ScopeSymbol* declaringScope);
  [[nodiscard]] auto isLexicallyInsideClass() const -> bool;
  void reportDifferentKindOfSymbol(ScopeSymbol* declaringScope);
  void reportMemberRedeclaration(FunctionSymbol* previous);

  void applyVirtualFlagsFromDeclarator();
  auto enclosingClass() const -> ClassSymbol*;
  void checkVirtualSpecifierOutsideClass();
  void checkOverrideAndFinalSpecifiers(FunctionSymbol* overridden);
  void checkCovariantReturnType(FunctionSymbol* overridden);

  auto declare() -> FunctionSymbol*;

  void checkRedeclaration();
  void checkConstructor();

  void inheritAbiTags(FunctionSymbol* canonical);
  void checkDeclSpecifiers();
  void checkExternalLinkageSpec();

  void checkVirtualSpecifier();
  void mergeRedeclaration();
};

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  return DeclareFunction{*this, declarator, decl}.declare();
}

auto Binder::DeclareFunction::declare() -> FunctionSymbol* {
  functionDeclarator = getFunctionPrototype(declarator);

  auto name = decl.getName();
  auto returnType = decl.getReturnType(scope());
  auto type =
      type_cast<FunctionType>(binder.resolveMemberOfCurrentInstantiation(
          getDeclaratorType(binder.unit_, declarator, returnType),
          binder.currentInstantiationOf(binder.declaringScope())));

  auto originalScope = binder.declaringScope();
  auto targetScope = !decl.specs.isFriend
                         ? binder.scopeForBlockDecl(originalScope)
                         : originalScope;

  functionSymbol = control()->newFunctionSymbol(targetScope, decl.location());
  functionSymbol->setName(name);
  functionSymbol->setType(type);

  functionSymbol->setTrailingRequiresClause(decl.trailingRequiresClause);

  if (functionDeclarator && functionDeclarator->exceptionSpecifier)
    functionSymbol->setExceptionSpecifier(true);

  binder.applyImplicitExceptionSpecification(functionSymbol);

  if (binder.isC() && binder.unit_->config().allowUnprototypedFunctions &&
      functionDeclarator && !functionDeclarator->parameterDeclarationClause) {
    functionSymbol->setNoPrototype(true);
  }

  checkDeclSpecifiers();
  checkExternalLinkageSpec();
  checkVirtualSpecifier();

  if (functionSymbol->isConstructor()) {
    checkConstructor();
    return functionSymbol;
  }

  checkRedeclaration();

  if (targetScope != originalScope) {
    if (functionSymbol->canonical() == functionSymbol)
      functionSymbol->setHidden(true);
    binder.injectUsing(originalScope, name, functionSymbol->canonical(),
                       functionSymbol->location());
  }

  return functionSymbol;
}

auto Binder::DeclareFunction::declaringScopeForFunction() const
    -> ScopeSymbol* {
  auto declaringScope = binder.declaringScope();

  if (!functionSymbol->isFriend()) {
    if (!declaringScope->isClassOrNamespace()) {
      if (auto ns = declaringScope->enclosingNamespace()) return ns;
    }
    return declaringScope;
  }

  if (declaringScope->isNamespace()) return declaringScope;

  auto enclosingNamespace = declaringScope->enclosingNamespace();
  if (enclosingNamespace) return enclosingNamespace;

  return declaringScope;
}

void Binder::DeclareFunction::mergeAsCRedeclaration(
    FunctionSymbol* otherFunction) {
  auto canonical = otherFunction->canonical();
  canonical->addRedeclaration(functionSymbol);
  if (canonical->hasNoPrototype() && !functionSymbol->hasNoPrototype()) {
    canonical->setType(functionSymbol->type());
    canonical->setNoPrototype(false);
  }
  mergeRedeclaration();
}

auto Binder::DeclareFunction::mergeWithMatchingOverload(
    OverloadSetSymbol* overloadSet) -> bool {
  for (auto existingFunction : overloadSet->declaredFunctions()) {
    if (existingFunction->isSpecialization()) continue;

    auto existingTemplateDecl = existingFunction->templateDeclaration();
    auto newTemplateHead = decl.specs.templateHead;
    auto headsEquivalent = areFunctionTemplateHeadsEquivalentForRedeclaration(
        binder.unit_, symbol_cast<ClassSymbol>(declaringScopeForFunction()),
        existingTemplateDecl, newTemplateHead);
    if (!headsEquivalent) {
      auto instantiatingFunction =
          symbol_cast<FunctionSymbol>(binder.instantiatingSymbol());
      headsEquivalent =
          instantiatingFunction &&
          instantiatingFunction->canonical() == existingFunction->canonical() &&
          (existingTemplateDecl != nullptr) != (newTemplateHead != nullptr);
    }
    if (!headsEquivalent) {
      continue;
    }

    bool sigEq = areFunctionSignaturesEquivalentForRedeclaration(
        binder.unit_, existingFunction->type(), functionSymbol->type());

    if (!sigEq && existingTemplateDecl && newTemplateHead &&
        existingTemplateDecl->depth != newTemplateHead->depth) {
      int ownParamCount = 0;
      for ([[maybe_unused]] auto p :
           ListView{newTemplateHead->templateParameterList}) {
        ++ownParamCount;
      }
      sigEq = typesEquivalentModuloOwnHeadDepth(
          binder.unit_, existingFunction->type(), functionSymbol->type(),
          existingTemplateDecl->depth, newTemplateHead->depth, ownParamCount);
    }

    if (!sigEq) continue;

    if (!trailingRequiresClausesEquivalent(
            binder.unit_, existingFunction->trailingRequiresClause(),
            functionSymbol->trailingRequiresClause()))
      continue;

    reportMemberRedeclaration(existingFunction);

    auto canonical = existingFunction->canonical();
    canonical->addRedeclaration(functionSymbol);
    mergeRedeclaration();
    return true;
  }

  return false;
}

void Binder::DeclareFunction::reportDifferentKindOfSymbol(
    ScopeSymbol* declaringScope) {
  if (!isLexicallyInsideClass()) return;

  for (auto candidate : declaringScope->find(functionSymbol->name())) {
    if (symbol_cast<FunctionSymbol>(candidate)) continue;
    if (symbol_cast<OverloadSetSymbol>(candidate)) continue;
    if (symbol_cast<UsingDeclarationSymbol>(candidate)) continue;
    if (symbol_cast<InjectedClassNameSymbol>(candidate)) continue;
    if (is_type(candidate)) continue;

    binder.error(functionSymbol->location(),
                 std::format("redefinition of '{}' as different kind of symbol",
                             to_string(functionSymbol->name())));
    binder.note(candidate->location(), "previous definition is here");
    return;
  }
}

auto Binder::DeclareFunction::isLexicallyInsideClass() const -> bool {
  if (functionSymbol->isFriend()) return false;

  auto enclosingClass = symbol_cast<ClassSymbol>(binder.declaringScope());
  if (!enclosingClass) return false;
  if (enclosingClass->isSpecialization()) return false;

  auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator);
  return id && !id->nestedNameSpecifier;
}

void Binder::DeclareFunction::reportMemberRedeclaration(
    FunctionSymbol* previous) {
  if (!isLexicallyInsideClass()) return;

  const bool previousIsTemplate = previous->templateDeclaration() != nullptr;
  const bool redeclarationIsTemplate = decl.specs.templateHead != nullptr;
  if (previousIsTemplate != redeclarationIsTemplate) return;

  if (isDependent(binder.unit_, previous->type())) return;
  if (isDependent(binder.unit_, functionSymbol->type())) return;

  binder.error(functionSymbol->location(),
               functionSymbol->isConstructor()
                   ? "constructor cannot be redeclared"
                   : "class member cannot be redeclared");
  binder.note(previous->canonical()->location(),
              "previous declaration is here");
}

void Binder::DeclareFunction::checkRedeclaration() {
  if (auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator)) {
    if (auto nns = id->nestedNameSpecifier) {
      if (!nns->symbol && !binder.declaringScope()->isClass()) return;
    }
  }

  auto declaringScope = declaringScopeForFunction();

  if (binder.isC()) {
    checkCRedeclaration(declaringScope);
    return;
  }

  if (functionSymbol->isFriend() && !declaringScope->isClass()) {
    functionSymbol->setHidden(true);
  }

  reportDifferentKindOfSymbol(declaringScope);

  auto overloadSet = binder.overloadSetFor(
      declaringScope, functionSymbol->name(), functionSymbol->location());

  if (!mergeWithMatchingOverload(overloadSet)) {
    overloadSet->addFunction(functionSymbol);
  }

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkCRedeclaration(ScopeSymbol* declaringScope) {
  for (Symbol* candidate : declaringScope->find(functionSymbol->name())) {
    auto otherFunction = symbol_cast<FunctionSymbol>(candidate);
    if (!otherFunction) continue;

    auto canonical = otherFunction->canonical();
    const bool canMerge =
        (binder.unit_->config().allowUnprototypedFunctions &&
         canonical->hasNoPrototype()) ||
        areFunctionSignaturesEquivalentForRedeclaration(
            binder.unit_, canonical->type(), functionSymbol->type());
    if (canMerge) {
      mergeAsCRedeclaration(otherFunction);
    } else {
      binder.error(functionSymbol->location(),
                   std::format("conflicting types for '{}'",
                               to_string(functionSymbol->name())));
      binder.note(canonical->location(),
                  std::format("previous declaration of '{}' is here",
                              to_string(canonical->name())));
    }
    return;
  }

  declaringScope->addSymbol(functionSymbol);

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkConstructor() {
  auto classScope = binder.scope();
  if (classScope && classScope->isTemplateParameters()) {
    classScope = classScope->enclosingNonTemplateParametersScope();
  }
  auto enclosingClass = symbol_cast<ClassSymbol>(classScope);

  if (!enclosingClass) {
    cxx_runtime_error("constructor must be declared inside a class");
  }

  if (!mergeWithMatchingOverload(enclosingClass->constructorOverloadSet())) {
    enclosingClass->addConstructor(functionSymbol);
  }

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkDeclSpecifiers() {
  binder.applySpecifiers(functionSymbol, decl.specs);
}

void Binder::DeclareFunction::checkExternalLinkageSpec() {
  if (binder.isC()) {
    functionSymbol->setLanguageLinkage(LanguageKind::kC);
    return;
  }

  if (scope()->isClass()) {
    functionSymbol->setLanguageLinkage(LanguageKind::kCXX);
    return;
  }

  functionSymbol->setLanguageLinkage(binder.languageLinkage_);
}

void Binder::DeclareFunction::applyVirtualFlagsFromDeclarator() {
  if (!functionDeclarator) return;
  if (functionDeclarator->isOverride) functionSymbol->setOverride(true);
  if (functionDeclarator->isFinal) functionSymbol->setFinal(true);

  if (!functionDeclarator->isPure) return;

  functionSymbol->setPure(true);
  functionSymbol->setVirtual(true);
}

auto Binder::DeclareFunction::enclosingClass() const -> ClassSymbol* {
  return symbol_cast<ClassSymbol>(scope());
}

void Binder::DeclareFunction::checkVirtualSpecifierOutsideClass() {
  if (!functionSymbol->isVirtual() && !functionSymbol->isOverride() &&
      !functionSymbol->isFinal()) {
    return;
  }

  if (functionSymbol->isVirtual()) {
    binder.error(functionSymbol->location(),
                 "'virtual' can only appear on non-static member "
                 "functions");
    functionSymbol->setVirtual(false);
  }

  if (functionSymbol->isOverride()) {
    binder.error(functionSymbol->location(),
                 "'override' can only appear on non-static member "
                 "functions");
  }

  if (functionSymbol->isFinal()) {
    binder.error(functionSymbol->location(),
                 "'final' can only appear on non-static member functions");
  }
}

void Binder::DeclareFunction::checkCovariantReturnType(
    FunctionSymbol* overridden) {
  auto overriderType = type_cast<FunctionType>(functionSymbol->type());
  auto overriddenType = type_cast<FunctionType>(overridden->type());
  if (!overriderType || !overriddenType) return;

  auto overriderReturnType = overriderType->returnType();
  auto overriddenReturnType = overriddenType->returnType();
  if (!overriderReturnType || !overriddenReturnType) return;

  if (isDependent(binder.unit_, overriderReturnType) ||
      isDependent(binder.unit_, overriddenReturnType)) {
    return;
  }

  if (binder.traits.is_covariant_return_type(overriddenReturnType,
                                             overriderReturnType)) {
    return;
  }

  binder.error(functionSymbol->location(),
               std::format("return type of virtual function '{}' is not "
                           "covariant with the return type of the function it "
                           "overrides",
                           to_string(functionSymbol->name())));
  binder.note(overridden->location(), "overridden virtual function is here");
}

void Binder::DeclareFunction::checkOverrideAndFinalSpecifiers(
    FunctionSymbol* overridden) {
  if (functionSymbol->isOverride() && !overridden) {
    binder.error(functionSymbol->location(),
                 std::format("'{}' marked 'override' but does not override "
                             "any member function",
                             to_string(functionSymbol->name())));
  }

  if (!functionSymbol->isFinal() || functionSymbol->isVirtual()) return;

  binder.error(functionSymbol->location(),
               std::format("'{}' marked 'final' but is not virtual",
                           to_string(functionSymbol->name())));
}

auto Binder::findOverriddenFunction(ClassSymbol* cls, FunctionSymbol* fn)
    -> FunctionSymbol* {
  std::unordered_set<ClassSymbol*> visited;
  return findOverriddenFunctionImpl(cls, fn, visited);
}

auto Binder::findOverriddenFunctionImpl(
    ClassSymbol* cls, FunctionSymbol* fn,
    std::unordered_set<ClassSymbol*>& visited) -> FunctionSymbol* {
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass || !visited.insert(baseClass).second) continue;
    baseClass = baseClass->resolvedDefinition();

    auto checkMember = [&](FunctionSymbol* member) -> FunctionSymbol* {
      if (!member->isVirtual()) return nullptr;
      if (!traits.is_corresponding_overrider(fn, member)) return nullptr;
      return member;
    };

    for (auto symbol : baseClass->members()) {
      if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
        if (auto result = checkMember(func)) return result;
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(symbol)) {
        for (auto func : ovl->declaredFunctions()) {
          if (auto result = checkMember(func)) return result;
        }
      }
    }

    if (auto result = findOverriddenFunctionImpl(baseClass, fn, visited))
      return result;
  }
  return nullptr;
}

void Binder::DeclareFunction::checkVirtualSpecifier() {
  applyVirtualFlagsFromDeclarator();

  auto cls = enclosingClass();
  if (!cls) {
    checkVirtualSpecifierOutsideClass();
    return;
  }

  if (functionSymbol->isConstructor()) return;

  auto overridden = binder.findOverriddenFunction(cls, functionSymbol);

  if (overridden) {
    functionSymbol->setVirtual(true);

    if (overridden->isFinal()) {
      binder.error(
          functionSymbol->location(),
          std::format("declaration of '{}' overrides a 'final' function",
                      to_string(functionSymbol->name())));
    }

    checkCovariantReturnType(overridden);
  }

  if (!overridden) {
    for (auto base : cls->baseClasses()) {
      auto baseSymbol = base->symbol();
      if (!baseSymbol) return;
      if (auto baseType = baseSymbol->type();
          baseType && isDependent(binder.unit_, baseType)) {
        return;
      }
    }
  }

  checkOverrideAndFinalSpecifiers(overridden);
}

void Binder::DeclareFunction::inheritAbiTags(FunctionSymbol* canonical) {
  functionSymbol->setAbiTags(canonical->abiTagList());
}

void Binder::DeclareFunction::mergeRedeclaration() {
  auto canonical = functionSymbol->canonical();
  if (!canonical || canonical == functionSymbol) return;

  if (!functionSymbol->isFriend() && canonical->isHidden()) {
    canonical->setHidden(false);
  }

  if (canonical->isStatic()) functionSymbol->setStatic(true);
  if (canonical->isExtern()) functionSymbol->setExtern(true);
  if (canonical->isFriend()) functionSymbol->setFriend(true);
  if (canonical->isConstexpr()) functionSymbol->setConstexpr(true);
  if (canonical->isConsteval()) functionSymbol->setConsteval(true);
  if (canonical->isInline()) functionSymbol->setInline(true);
  if (canonical->isVirtual()) functionSymbol->setVirtual(true);
  if (canonical->isExplicit()) functionSymbol->setExplicit(true);
  if (canonical->isOverride()) functionSymbol->setOverride(true);
  if (canonical->isFinal()) functionSymbol->setFinal(true);
  if (canonical->isPure()) functionSymbol->setPure(true);
  if (canonical->hasCLinkage())
    functionSymbol->setLanguageLinkage(LanguageKind::kC);

  inheritAbiTags(canonical);

  if (functionSymbol->isInline()) canonical->setInline(true);
  if (functionSymbol->isConstexpr()) canonical->setConstexpr(true);
  if (functionSymbol->isConsteval()) canonical->setConsteval(true);
  if (functionSymbol->hasCLinkage())
    canonical->setLanguageLinkage(LanguageKind::kC);

  auto canonParams = canonical->functionParameters();
  auto redeclParams = functionSymbol->functionParameters();
  if (!canonParams || !redeclParams) return;

  auto canonIt = canonParams->members().begin();
  auto canonEnd = canonParams->members().end();
  auto redeclIt = redeclParams->members().begin();
  auto redeclEnd = redeclParams->members().end();

  for (; canonIt != canonEnd && redeclIt != redeclEnd; ++canonIt, ++redeclIt) {
    auto cp = symbol_cast<ParameterSymbol>(*canonIt);
    auto rp = symbol_cast<ParameterSymbol>(*redeclIt);
    if (!cp || !rp) continue;

    if (cp->defaultArgument() && rp->defaultArgument()) {
      binder.error(rp->location(), "redefinition of default argument");
      continue;
    }

    if (!cp->defaultArgument() && rp->defaultArgument()) {
      cp->setDefaultArgument(rp->defaultArgument());
      continue;
    }

    if (cp->defaultArgument() && !rp->defaultArgument()) {
      rp->setDefaultArgument(cp->defaultArgument());
    }
  }
}
}  // namespace cxx
