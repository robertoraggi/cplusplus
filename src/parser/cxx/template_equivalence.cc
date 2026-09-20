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

[[nodiscard]] auto parameterCountOf(TemplateDeclarationAST* templateDecl)
    -> int {
  int count = 0;
  for ([[maybe_unused]] auto parameter :
       ListView{templateDecl->templateParameterList})
    ++count;
  return count;
}

}  // namespace

auto TemplateEquivalence::same(const ExceptionSpecification& a,
                               const ExceptionSpecification& b) const -> bool {
  if (a.index() != b.index()) return false;
  if (auto value = std::get_if<bool>(&a)) return *value == std::get<bool>(b);
  return same(std::get<ExpressionAST*>(a), std::get<ExpressionAST*>(b));
}

auto TemplateEquivalence::same(const Type* a, const Type* b) const -> bool {
  if (!a || !b) return false;
  if (a == b) return true;

  if (auto lhs = type_cast<UnresolvedNameType>(a)) {
    auto rhs = type_cast<UnresolvedNameType>(b);
    return rhs &&
           same(lhs->nestedNameSpecifier(), rhs->nestedNameSpecifier()) &&
           same(lhs->unqualifiedId(), rhs->unqualifiedId());
  }

  if (correspondence_.applies()) return corresponds(a, b, correspondence_);

  auto aParam = type_cast<TypeParameterType>(a);
  auto bParam = type_cast<TypeParameterType>(b);
  if (aParam && bParam) {
    return aParam->depth() == bParam->depth() &&
           aParam->index() == bParam->index();
  }
  if (aParam || bParam) return false;

  return unit_->typeTraits().is_same(a, b);
}

auto TemplateEquivalence::sameQualifiedName(NestedNameSpecifierAST* aQualifier,
                                            UnqualifiedIdAST* aName,
                                            NestedNameSpecifierAST* bQualifier,
                                            UnqualifiedIdAST* bName) const
    -> bool {
  if (!ast_cast<TemplateNestedNameSpecifierAST>(aQualifier)) return false;
  if (!same(aQualifier, bQualifier)) return false;

  auto aNameId = ast_cast<NameIdAST>(aName);
  auto bNameId = ast_cast<NameIdAST>(bName);
  return aNameId && bNameId && aNameId->identifier == bNameId->identifier;
}

auto TemplateEquivalence::same(NamedTypeSpecifierAST* a,
                               NamedTypeSpecifierAST* b) const -> bool {
  if (!a || !b) return false;

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a->unqualifiedId);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b->unqualifiedId);

  if (aTemplateId || bTemplateId) {
    if (!aTemplateId || !bTemplateId) return false;
    if (!aTemplateId->symbol || aTemplateId->symbol != bTemplateId->symbol)
      return false;
    return same(aTemplateId->templateArgumentList,
                bTemplateId->templateArgumentList);
  }

  if (!same(a->nestedNameSpecifier, b->nestedNameSpecifier)) return false;

  if (!ast_cast<NameIdAST>(a->unqualifiedId) ||
      !ast_cast<NameIdAST>(b->unqualifiedId))
    return false;

  return same(a->symbol ? a->symbol->type() : nullptr,
              b->symbol ? b->symbol->type() : nullptr);
}

auto TemplateEquivalence::sameWritten(NamedTypeSpecifierAST* a,
                                      NamedTypeSpecifierAST* b) const -> bool {
  if (!a || !b) return false;

  auto aName = ast_cast<NameIdAST>(a->unqualifiedId);
  auto bName = ast_cast<NameIdAST>(b->unqualifiedId);
  if (aName || bName) {
    if (!aName || !bName) return false;
    auto aParameter = template_parameter_info(a->symbol);
    auto bParameter = template_parameter_info(b->symbol);
    if (!aParameter && !bParameter) {
      if (aName->identifier != bName->identifier) return false;
    }
    return same(a->symbol ? a->symbol->type() : nullptr,
                b->symbol ? b->symbol->type() : nullptr);
  }

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a->unqualifiedId);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b->unqualifiedId);
  if (!aTemplateId || !bTemplateId ||
      aTemplateId->identifier != bTemplateId->identifier)
    return false;
  return sameWritten(aTemplateId->templateArgumentList,
                     bTemplateId->templateArgumentList);
}

auto TemplateEquivalence::same(UnqualifiedIdAST* a, UnqualifiedIdAST* b) const
    -> bool {
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
  return same(aTemplateId->templateArgumentList,
              bTemplateId->templateArgumentList);
}

auto TemplateEquivalence::same(TypenameSpecifierAST* a,
                               TypenameSpecifierAST* b) const -> bool {
  if (!a || !b) return false;
  return sameQualifiedName(a->nestedNameSpecifier, a->unqualifiedId,
                           b->nestedNameSpecifier, b->unqualifiedId);
}

auto TemplateEquivalence::same(TypeIdAST* a, TypeIdAST* b) const -> bool {
  if (!a || !b) return false;
  auto aSpec = a->typeSpecifierList;
  auto bSpec = b->typeSpecifierList;
  for (; aSpec && bSpec; aSpec = aSpec->next, bSpec = bSpec->next) {
    if (aSpec->value->kind() != bSpec->value->kind()) return false;
    if (auto aNamed = ast_cast<NamedTypeSpecifierAST>(aSpec->value)) {
      if (!sameWritten(aNamed, ast_cast<NamedTypeSpecifierAST>(bSpec->value)))
        return false;
      continue;
    }
    if (auto aTypename = ast_cast<TypenameSpecifierAST>(aSpec->value)) {
      if (!same(aTypename, ast_cast<TypenameSpecifierAST>(bSpec->value)))
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
      if (!bDecltype || !same(aDecltype->expression, bDecltype->expression))
        return false;
    }
  }
  if (aSpec || bSpec) return false;
  return same(a->type, b->type);
}

auto TemplateEquivalence::same(ExpressionAST* a, ExpressionAST* b) const
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;

  if (auto nested = ast_cast<NestedExpressionAST>(a))
    return same(nested->expression, b);
  if (auto nested = ast_cast<NestedExpressionAST>(b))
    return same(a, nested->expression);
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(a))
    return same(cast->expression, b);
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(b))
    return same(a, cast->expression);
  if (auto constant = ast_cast<ConstExpressionAST>(a))
    return same(constant->expression, b);
  if (auto constant = ast_cast<ConstExpressionAST>(b))
    return same(a, constant->expression);

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
           same(aSizeofType->typeId ? aSizeofType->typeId->type : nullptr,
                bSizeofType->typeId ? bSizeofType->typeId->type : nullptr);
  }

  if (auto aSizeof = ast_cast<SizeofExpressionAST>(a)) {
    auto bSizeof = ast_cast<SizeofExpressionAST>(b);
    return bSizeof && same(aSizeof->expression, bSizeof->expression);
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
      if (!same(aTypeId->value, bTypeId->value)) return false;
    }
    return !aTypeId && !bTypeId;
  }

  if (auto aUnary = ast_cast<UnaryExpressionAST>(a)) {
    auto bUnary = ast_cast<UnaryExpressionAST>(b);
    return bUnary && aUnary->op == bUnary->op &&
           same(aUnary->expression, bUnary->expression);
  }

  if (auto aBinary = ast_cast<BinaryExpressionAST>(a)) {
    auto bBinary = ast_cast<BinaryExpressionAST>(b);
    return bBinary && aBinary->op == bBinary->op &&
           same(aBinary->leftExpression, bBinary->leftExpression) &&
           same(aBinary->rightExpression, bBinary->rightExpression);
  }

  if (auto aId = ast_cast<IdExpressionAST>(a)) {
    auto bId = ast_cast<IdExpressionAST>(b);
    if (!bId) return false;

    auto aNttp = symbol_cast<NonTypeParameterSymbol>(aId->symbol);
    auto bNttp = symbol_cast<NonTypeParameterSymbol>(bId->symbol);
    if (aNttp || bNttp) {
      if (!aNttp || !bNttp) return false;
      if (correspondence_.applies() &&
          aNttp->depth() == correspondence_.lhsDepth &&
          aNttp->index() < correspondence_.count) {
        return bNttp->depth() == correspondence_.rhsDepth &&
               bNttp->index() == aNttp->index();
      }
      return aNttp->depth() == bNttp->depth() &&
             aNttp->index() == bNttp->index();
    }

    auto aTid = ast_cast<SimpleTemplateIdAST>(aId->unqualifiedId);
    auto bTid = ast_cast<SimpleTemplateIdAST>(bId->unqualifiedId);
    if (aTid || bTid) {
      if (!aTid || !bTid) return false;
      if (!aTid->symbol || aTid->symbol != bTid->symbol) return false;
      return sameWritten(aTid->templateArgumentList,
                         bTid->templateArgumentList);
    }

    auto aNameId = ast_cast<NameIdAST>(aId->unqualifiedId);
    auto bNameId = ast_cast<NameIdAST>(bId->unqualifiedId);
    if (!aNameId || !bNameId) return false;
    if (aNameId->identifier != bNameId->identifier) return false;
    if (aId->symbol && bId->symbol) return aId->symbol == bId->symbol;
    return same(aId->nestedNameSpecifier, bId->nestedNameSpecifier);
  }

  return false;
}

auto TemplateEquivalence::same(NonTypeTemplateParameterAST* a,
                               NonTypeTemplateParameterAST* b) const -> bool {
  if (!a || !b || !a->declaration || !b->declaration) return false;

  if (a->declaration->type && b->declaration->type &&
      !isDependent(unit_, a->declaration->type)) {
    return unit_->typeTraits().is_same(a->declaration->type,
                                       b->declaration->type);
  }

  auto aSpec = soleTypeSpecifier(a->declaration);
  auto bSpec = soleTypeSpecifier(b->declaration);

  if (auto aNamed = ast_cast<NamedTypeSpecifierAST>(aSpec))
    return same(aNamed, ast_cast<NamedTypeSpecifierAST>(bSpec));

  if (auto aTypename = ast_cast<TypenameSpecifierAST>(aSpec))
    return same(aTypename, ast_cast<TypenameSpecifierAST>(bSpec));

  return false;
}

auto TemplateEquivalence::corresponds(
    const TemplateArgument& lhs, const TemplateArgument& rhs,
    ParameterCorrespondence correspondence) const -> bool {
  auto lhsType = template_argument_as_type(lhs);
  auto rhsType = template_argument_as_type(rhs);
  if (lhsType || rhsType) {
    if (!lhsType || !rhsType) return false;
    return corresponds(lhsType, rhsType, correspondence);
  }

  auto lhsInfo = template_argument_parameter_info(lhs);
  auto rhsInfo = template_argument_parameter_info(rhs);
  if (lhsInfo || rhsInfo) {
    if (!lhsInfo || !rhsInfo) return false;
    if (lhsInfo->isPack != rhsInfo->isPack) return false;
    if (lhsInfo->depth == correspondence.lhsDepth &&
        lhsInfo->index < correspondence.count) {
      return rhsInfo->depth == correspondence.rhsDepth &&
             rhsInfo->index == lhsInfo->index;
    }
    return lhsInfo->depth == rhsInfo->depth && lhsInfo->index == rhsInfo->index;
  }

  return lhs == rhs;
}

auto TemplateEquivalence::corresponds(
    const std::vector<TemplateArgument>& lhs,
    const std::vector<TemplateArgument>& rhs,
    ParameterCorrespondence correspondence) const -> bool {
  if (lhs.size() != rhs.size()) return false;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!corresponds(lhs[i], rhs[i], correspondence)) return false;
  }
  return true;
}

auto TemplateEquivalence::corresponds(
    const Type* lhs, const Type* rhs,
    ParameterCorrespondence correspondence) const -> bool {
  if (!lhs || !rhs) return lhs == rhs;

  auto recurse = [&](const Type* a, const Type* b) {
    return corresponds(a, b, correspondence);
  };

  if (auto name = type_cast<UnresolvedNameType>(lhs)) {
    auto other = type_cast<UnresolvedNameType>(rhs);
    TemplateEquivalence equivalence{unit_, correspondence};
    return other &&
           equivalence.same(name->nestedNameSpecifier(),
                            other->nestedNameSpecifier()) &&
           equivalence.same(name->unqualifiedId(), other->unqualifiedId());
  }

  auto lhsInfo = getTypeParamInfo(lhs);
  auto rhsInfo = getTypeParamInfo(rhs);
  if (lhsInfo || rhsInfo) {
    if (!lhsInfo || !rhsInfo) return false;
    if (lhsInfo->depth == correspondence.lhsDepth &&
        lhsInfo->index < correspondence.count) {
      return rhsInfo->depth == correspondence.rhsDepth &&
             rhsInfo->index == lhsInfo->index &&
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
    return recurse(lhsQual->elementType(), rhsQual->elementType());
  }
  if (auto lhsPtr = type_cast<PointerType>(lhs)) {
    auto rhsPtr = type_cast<PointerType>(rhs);
    if (!rhsPtr) return false;
    return recurse(lhsPtr->elementType(), rhsPtr->elementType());
  }
  if (auto lhsRef = type_cast<LvalueReferenceType>(lhs)) {
    auto rhsRef = type_cast<LvalueReferenceType>(rhs);
    if (!rhsRef) return false;
    return recurse(lhsRef->elementType(), rhsRef->elementType());
  }
  if (auto lhsRef = type_cast<RvalueReferenceType>(lhs)) {
    auto rhsRef = type_cast<RvalueReferenceType>(rhs);
    if (!rhsRef) return false;
    return recurse(lhsRef->elementType(), rhsRef->elementType());
  }
  if (auto lhsArr = type_cast<BoundedArrayType>(lhs)) {
    auto rhsArr = type_cast<BoundedArrayType>(rhs);
    if (!rhsArr || lhsArr->size() != rhsArr->size()) return false;
    return recurse(lhsArr->elementType(), rhsArr->elementType());
  }
  if (auto lhsArr = type_cast<UnboundedArrayType>(lhs)) {
    auto rhsArr = type_cast<UnboundedArrayType>(rhs);
    if (!rhsArr) return false;
    return recurse(lhsArr->elementType(), rhsArr->elementType());
  }
  if (auto lhsFn = type_cast<FunctionType>(lhs)) {
    auto rhsFn = type_cast<FunctionType>(rhs);
    if (!rhsFn) return false;
    if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;
    if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
    if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;
    if (!TemplateEquivalence{unit_, correspondence}.same(
            lhsFn->exceptionSpecification(), rhsFn->exceptionSpecification()))
      return false;

    const auto& lhsParams = lhsFn->parameterTypes();
    const auto& rhsParams = rhsFn->parameterTypes();
    if (lhsParams.size() != rhsParams.size()) return false;

    if (!recurse(lhsFn->returnType(), rhsFn->returnType())) return false;

    for (std::size_t i = 0; i < lhsParams.size(); ++i) {
      if (!recurse(lhsParams[i], rhsParams[i])) return false;
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

    auto lhsTemplate = class_template_of(lhsSym);
    if (!lhsTemplate) return false;
    if (lhsTemplate != class_template_of(rhsSym)) return false;

    return corresponds(
        expand_template_arguments(class_template_arguments(lhsSym)),
        expand_template_arguments(class_template_arguments(rhsSym)),
        correspondence);
  }

  return unit_->typeTraits().is_same(lhs, rhs);
}

auto TemplateEquivalence::walkArguments(List<TemplateArgumentAST*>* a,
                                        List<TemplateArgumentAST*>* b,
                                        ArgumentMatch match) const -> bool {
  for (; a && b; a = a->next, b = b->next) {
    auto typeA = ast_cast<TypeTemplateArgumentAST>(a->value);
    auto typeB = ast_cast<TypeTemplateArgumentAST>(b->value);
    if (typeA || typeB) {
      if (!typeA || !typeB) return false;
      const bool equal =
          match == ArgumentMatch::kByWrittenTypeId
              ? same(typeA->typeId, typeB->typeId)
              : typeA->typeId && typeB->typeId &&
                    same(typeA->typeId->type, typeB->typeId->type);
      if (!equal) return false;
      continue;
    }

    auto expressionA = ast_cast<ExpressionTemplateArgumentAST>(a->value);
    auto expressionB = ast_cast<ExpressionTemplateArgumentAST>(b->value);
    if (!expressionA || !expressionB ||
        !same(expressionA->expression, expressionB->expression))
      return false;
  }

  return !a && !b;
}

auto TemplateEquivalence::same(List<TemplateArgumentAST*>* a,
                               List<TemplateArgumentAST*>* b) const -> bool {
  return walkArguments(a, b, ArgumentMatch::kByType);
}

auto TemplateEquivalence::sameWritten(List<TemplateArgumentAST*>* a,
                                      List<TemplateArgumentAST*>* b) const
    -> bool {
  return walkArguments(a, b, ArgumentMatch::kByWrittenTypeId);
}

auto TemplateEquivalence::same(NestedNameSpecifierAST* a,
                               NestedNameSpecifierAST* b) const -> bool {
  if (a == b) return true;
  if (!a || !b) return false;

  if (auto aTemplate = ast_cast<TemplateNestedNameSpecifierAST>(a)) {
    auto bTemplate = ast_cast<TemplateNestedNameSpecifierAST>(b);
    if (!bTemplate) return false;
    if (!same(aTemplate->nestedNameSpecifier, bTemplate->nestedNameSpecifier))
      return false;
    auto aTemplateId = aTemplate->templateId;
    auto bTemplateId = bTemplate->templateId;
    if (!aTemplateId || !bTemplateId) return false;
    if (!aTemplateId->symbol || aTemplateId->symbol != bTemplateId->symbol)
      return false;
    return same(aTemplateId->templateArgumentList,
                bTemplateId->templateArgumentList);
  }

  if (auto aSimple = ast_cast<SimpleNestedNameSpecifierAST>(a)) {
    auto bSimple = ast_cast<SimpleNestedNameSpecifierAST>(b);
    if (!bSimple) return false;
    if (aSimple->symbol || bSimple->symbol)
      return aSimple->symbol == bSimple->symbol;
    if (aSimple->identifier != bSimple->identifier) return false;
    return same(aSimple->nestedNameSpecifier, bSimple->nestedNameSpecifier);
  }

  if (ast_cast<GlobalNestedNameSpecifierAST>(a))
    return ast_cast<GlobalNestedNameSpecifierAST>(b) != nullptr;

  if (auto aDecltype = ast_cast<DecltypeNestedNameSpecifierAST>(a)) {
    auto bDecltype = ast_cast<DecltypeNestedNameSpecifierAST>(b);
    if (!bDecltype) return false;
    if (!aDecltype->decltypeSpecifier || !bDecltype->decltypeSpecifier)
      return false;
    return same(aDecltype->decltypeSpecifier->expression,
                bDecltype->decltypeSpecifier->expression);
  }

  return false;
}

auto TemplateEquivalence::same(List<TemplateParameterAST*>* aIt,
                               List<TemplateParameterAST*>* bIt) const -> bool {
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
      if (!TemplateEquivalence{unit_}.sameWritten(
              aTypeConstraint->templateArgumentList,
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
      if (!same(aNonType, bNonType)) return false;
    }

    auto aTemplate = ast_cast<TemplateTypeParameterAST>(aParam);
    auto bTemplate = ast_cast<TemplateTypeParameterAST>(bParam);
    if (aTemplate && bTemplate) {
      if (aTemplate->isPack != bTemplate->isPack) return false;
      if (!same(aTemplate->templateParameterList,
                bTemplate->templateParameterList))
        return false;
      if (!aTemplate->requiresClause || !bTemplate->requiresClause) {
        if (aTemplate->requiresClause != bTemplate->requiresClause)
          return false;
      } else if (!TemplateEquivalence{unit_}.same(
                     aTemplate->requiresClause->expression,
                     bTemplate->requiresClause->expression)) {
        return false;
      }
    }
  }

  return !aIt && !bIt;
}

auto TemplateEquivalence::sameForOrdering(
    List<TemplateParameterAST*>* aIt, List<TemplateParameterAST*>* bIt) const
    -> bool {
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
      if (!aNonType || !bNonType || !same(aNonType, bNonType)) return false;
      auto aInfo = template_parameter_info(a->symbol);
      auto bInfo = template_parameter_info(b->symbol);
      if (!aInfo || !bInfo || aInfo->isPack != bInfo->isPack) return false;
      continue;
    }

    auto aTemplate = ast_cast<TemplateTypeParameterAST>(a);
    auto bTemplate = ast_cast<TemplateTypeParameterAST>(b);
    if (!aTemplate || !bTemplate || aTemplate->isPack != bTemplate->isPack)
      return false;
    if (!sameForOrdering(aTemplate->templateParameterList,
                         bTemplate->templateParameterList))
      return false;
  }

  return !aIt && !bIt;
}

auto TemplateEquivalence::sameForOrdering(
    const Type* a, const Type* b, TemplateDeclarationAST* aTemplate,
    TemplateDeclarationAST* bTemplate) const -> bool {
  if (!aTemplate || !bTemplate) return false;

  return corresponds(
      a, b, {aTemplate->depth, bTemplate->depth, parameterCountOf(aTemplate)});
}

auto TemplateEquivalence::same(RequiresClauseAST* a, RequiresClauseAST* b) const
    -> bool {
  if (!a || !b) return a == b;
  return TemplateEquivalence{unit_}.same(a->expression, b->expression);
}

auto TemplateEquivalence::same(TemplateDeclarationAST* a,
                               TemplateDeclarationAST* b) const -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!same(a->templateParameterList, b->templateParameterList)) return false;
  if (!a->requiresClause || !b->requiresClause)
    return a->requiresClause == b->requiresClause;

  return TemplateEquivalence{unit_, {a->depth, b->depth, parameterCountOf(a)}}
      .same(a->requiresClause->expression, b->requiresClause->expression);
}

auto TemplateEquivalence::ownFunctionTemplateHead(
    ClassSymbol* enclosingClass, TemplateDeclarationAST* templateHead) const
    -> TemplateDeclarationAST* {
  if (!templateHead) return nullptr;

  const bool isExplicitSpecializationHead =
      templateHead->symbol &&
      templateHead->symbol->isExplicitTemplateSpecialization();

  for (auto current = enclosingClass; current;
       current = symbol_cast<ClassSymbol>(current->parent())) {
    auto enclosingHead = current->templateDeclaration();
    const bool isClassSpecialization = current->isSpecialization();
    if (!enclosingHead && isClassSpecialization) {
      auto primary = current->primaryTemplateSymbol();
      if (primary) enclosingHead = primary->templateDeclaration();
    }
    if (!enclosingHead || enclosingHead->depth != templateHead->depth) {
      continue;
    }
    if (isExplicitSpecializationHead && isClassSpecialization) {
      return nullptr;
    }
    if (same(enclosingHead, templateHead)) {
      return nullptr;
    }
  }

  return templateHead;
}

}  // namespace cxx
