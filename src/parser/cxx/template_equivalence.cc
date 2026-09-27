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
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
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

struct WrittenTemplateArguments {
  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) const
      -> List<TemplateArgumentAST*>* {
    return ast->templateArgumentList;
  }

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast) const
      -> List<TemplateArgumentAST*>* {
    return ast->templateArgumentList;
  }

  [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast) const
      -> List<TemplateArgumentAST*>* {
    return ast->templateArgumentList;
  }

  [[nodiscard]] auto operator()(UnqualifiedIdAST*) const
      -> List<TemplateArgumentAST*>* {
    return nullptr;
  }
};

[[nodiscard]] auto writtenExpression(ExpressionAST* expression)
    -> ExpressionAST* {
  while (expression) {
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(expression)) {
      expression = cast->expression;
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(expression)) {
      expression = constant->expression;
      continue;
    }
    return expression;
  }
  return nullptr;
}

[[nodiscard]] auto namesParameter(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (template_parameter_info(symbol)) return true;
  return symbol_cast<ParameterSymbol>(symbol) != nullptr;
}

[[nodiscard]] auto correspondingParameters(
    const TypeParamInfo& lhs, const TypeParamInfo& rhs,
    const TemplateEquivalence::ParameterCorrespondence& correspondence)
    -> bool {
  if (lhs.isPack != rhs.isPack) return false;
  if (lhs.depth == correspondence.lhsDepth &&
      lhs.index < correspondence.count) {
    return rhs.depth == correspondence.rhsDepth && rhs.index == lhs.index;
  }
  return lhs.depth == rhs.depth && lhs.index == rhs.index;
}

[[nodiscard]] auto parameterPosition(ParameterSymbol* parameter)
    -> std::ptrdiff_t {
  auto scope = symbol_cast<FunctionParametersSymbol>(parameter->parent());
  if (!scope) return -1;
  auto parameters = views::members(scope) | views::parameters;
  return std::ranges::distance(parameters.begin(),
                               std::ranges::find(parameters, parameter));
}

[[nodiscard]] auto parameterListNesting(ParameterSymbol* parameter) -> int {
  int nesting = 0;
  for (Symbol* scope = parameter->parent(); scope; scope = scope->parent()) {
    if (symbol_cast<FunctionParametersSymbol>(scope)) ++nesting;
  }
  return nesting;
}

[[nodiscard]] auto correspondingFunctionParameters(ParameterSymbol* a,
                                                   ParameterSymbol* b) -> bool {
  if (a->isParameterPack() != b->isParameterPack()) return false;
  if (parameterPosition(a) != parameterPosition(b)) return false;
  return parameterListNesting(a) == parameterListNesting(b);
}

[[nodiscard]] auto callsDependentName(TranslationUnit* unit,
                                      CallExpressionAST* call) -> bool {
  auto callee = ast_cast<IdExpressionAST>(call->baseExpression);
  if (!callee || callee->nestedNameSpecifier) return false;
  if (!names_functions(callee->symbol)) return false;
  return isDependent(unit, call);
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
    if (!a->symbol || !b->symbol) {
      if (a->symbol != b->symbol) return false;
      return same(a->nestedNameSpecifier, b->nestedNameSpecifier);
    }
    return same(a->symbol->type(), b->symbol->type());
  }

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a->unqualifiedId);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b->unqualifiedId);
  if (!aTemplateId || !bTemplateId ||
      aTemplateId->identifier != bTemplateId->identifier)
    return false;
  return sameWritten(aTemplateId->templateArgumentList,
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

struct TemplateEquivalence::SameUnqualifiedId {
  const TemplateEquivalence& equivalence;
  UnqualifiedIdAST* other;

  template <typename Node>
  [[nodiscard]] auto counterpart(Node*) const -> Node* {
    return static_cast<Node*>(other);
  }

  [[nodiscard]] auto operator()(NameIdAST* ast) const -> bool {
    return ast->identifier == counterpart(ast)->identifier;
  }

  [[nodiscard]] auto operator()(DestructorIdAST* ast) const -> bool {
    return equivalence.same(ast->id, counterpart(ast)->id);
  }

  [[nodiscard]] auto operator()(DecltypeIdAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!ast->decltypeSpecifier || !b->decltypeSpecifier) return false;
    return equivalence.same(ast->decltypeSpecifier->expression,
                            b->decltypeSpecifier->expression);
  }

  [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast) const -> bool {
    return ast->op == counterpart(ast)->op;
  }

  [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast) const -> bool {
    return ast->identifier == counterpart(ast)->identifier;
  }

  [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->identifier != b->identifier) return false;
    return equivalence.same(ast->templateArgumentList, b->templateArgumentList);
  }

  [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->literalOperatorId, b->literalOperatorId))
      return false;
    return equivalence.same(ast->templateArgumentList, b->templateArgumentList);
  }

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->operatorFunctionId, b->operatorFunctionId))
      return false;
    return equivalence.same(ast->templateArgumentList, b->templateArgumentList);
  }
};

struct TemplateEquivalence::SameRequirement {
  const TemplateEquivalence& equivalence;
  RequirementAST* other;

  template <typename Node>
  [[nodiscard]] auto counterpart(Node*) const -> Node* {
    return static_cast<Node*>(other);
  }

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (bool(ast->noexceptLoc) != bool(b->noexceptLoc)) return false;
    if (!equivalence.same(ast->expression, b->expression)) return false;
    if (!ast->typeConstraint || !b->typeConstraint)
      return ast->typeConstraint == b->typeConstraint;
    return equivalence.same(ast->typeConstraint, b->typeConstraint);
  }

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }
};

struct TemplateEquivalence::SameExpression {
  const TemplateEquivalence& equivalence;
  ExpressionAST* other;

  template <typename Node>
  [[nodiscard]] auto counterpart(Node*) const -> Node* {
    return static_cast<Node*>(other);
  }

  [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!ast->literal || !b->literal) return false;
    if (!equivalence.same(ast->type, b->type)) return false;
    return ast->literal->charValue() == b->literal->charValue();
  }

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast) const -> bool {
    return ast->isTrue == counterpart(ast)->isTrue;
  }

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!ast->literal || !b->literal) return false;
    if (!equivalence.same(ast->type, b->type)) return false;
    return ast->literal->integerValue() == b->literal->integerValue();
  }

  [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!ast->literal || !b->literal) return false;
    if (!equivalence.same(ast->type, b->type)) return false;
    return ast->literal->floatValue() == b->literal->floatValue();
  }

  [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast) const
      -> bool {
    return ast->literal == counterpart(ast)->literal;
  }

  [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    return ast->literal == b->literal && ast->encoding == b->encoding;
  }

  [[nodiscard]] auto operator()(
      UserDefinedStringLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    return ast->literal == b->literal && ast->encoding == b->encoding;
  }

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return equivalence.same(ast->bracedInitList, b->bracedInitList);
  }

  [[nodiscard]] auto operator()(ThisExpressionAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(PackIndexExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->packExpression, b->packExpression)) return false;
    return equivalence.same(ast->indexExpression, b->indexExpression);
  }

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->expression, b->expression)) return false;
    return sameAssociations(ast->genericAssociationList,
                            b->genericAssociationList);
  }

  [[nodiscard]] auto operator()(NestedStatementExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(DefaultInitializerExpressionAST* ast) const
      -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(IdExpressionAST* ast) const -> bool {
    return equivalence.same(ast, counterpart(ast));
  }

  [[nodiscard]] auto operator()(LambdaExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op || ast->foldOp != b->foldOp) return false;
    if (!equivalence.same(ast->leftExpression, b->leftExpression)) return false;
    return equivalence.same(ast->rightExpression, b->rightExpression);
  }

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) const -> bool {
    return equivalence.same(ast, counterpart(ast));
  }

  [[nodiscard]] auto operator()(VaArgExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(SubscriptExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->baseExpression, b->baseExpression)) return false;
    return equivalence.same(ast->indexExpression, b->indexExpression);
  }

  [[nodiscard]] auto operator()(CallExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.sameCallee(ast, b)) return false;
    return equivalence.same(ast->expressionList, b->expressionList);
  }

  [[nodiscard]] auto operator()(TypeConstructionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->type, b->type)) return false;
    return equivalence.same(ast->expressionList, b->expressionList);
  }

  [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->type, b->type)) return false;
    return equivalence.same(ast->bracedInitList, b->bracedInitList);
  }

  [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->accessOp != b->accessOp) return false;
    if (!equivalence.same(ast->baseExpression, b->baseExpression)) return false;
    return sameSplicer(ast->splicer, b->splicer);
  }

  [[nodiscard]] auto operator()(MemberExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->accessOp != b->accessOp) return false;
    if (!equivalence.same(ast->baseExpression, b->baseExpression)) return false;
    if (!equivalence.same(ast->nestedNameSpecifier, b->nestedNameSpecifier))
      return false;
    if (!equivalence.same(ast->unqualifiedId, b->unqualifiedId)) return false;
    return equivalence.sameEntity(ast->symbol, b->symbol);
  }

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    return equivalence.same(ast->baseExpression, b->baseExpression);
  }

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->castOp != b->castOp) return false;
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (ast->identifier != b->identifier) return false;
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return sameDesignators(ast->designatorList, b->designatorList);
  }

  [[nodiscard]] auto operator()(TypeidExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(SpliceExpressionAST* ast) const -> bool {
    return sameSplicer(ast->splicer, counterpart(ast)->splicer);
  }

  [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST*) const
      -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast) const
      -> bool {
    return ast->symbol == counterpart(ast)->symbol;
  }

  [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(ReflectExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(LabelAddressExpressionAST* ast) const -> bool {
    return ast->identifier == counterpart(ast)->identifier;
  }

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(AwaitExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast) const -> bool {
    return equivalence.sameEntity(ast->symbol, counterpart(ast)->symbol);
  }

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast) const -> bool {
    return equivalence.same(ast->typeId, counterpart(ast)->typeId);
  }

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(NewExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (bool(ast->scopeLoc) != bool(b->scopeLoc)) return false;
    if (!sameNewPlacement(ast->newPlacement, b->newPlacement)) return false;
    if (!equivalence.same(ast->objectType, b->objectType)) return false;
    return sameNewInitializer(ast->newInitalizer, b->newInitalizer);
  }

  [[nodiscard]] auto operator()(DeleteExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (bool(ast->scopeLoc) != bool(b->scopeLoc)) return false;
    if (bool(ast->lbracketLoc) != bool(b->lbracketLoc)) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(CastExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->typeId, b->typeId)) return false;
    return equivalence.same(ast->expression, b->expression);
  }

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    if (!equivalence.same(ast->leftExpression, b->leftExpression)) return false;
    return equivalence.same(ast->rightExpression, b->rightExpression);
  }

  [[nodiscard]] auto operator()(ConditionalExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (!equivalence.same(ast->condition, b->condition)) return false;
    if (!equivalence.same(ast->iftrueExpression, b->iftrueExpression))
      return false;
    return equivalence.same(ast->iffalseExpression, b->iffalseExpression);
  }

  [[nodiscard]] auto operator()(YieldExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(ThrowExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    if (!equivalence.same(ast->leftExpression, b->leftExpression)) return false;
    return equivalence.same(ast->rightExpression, b->rightExpression);
  }

  [[nodiscard]] auto operator()(TargetExpressionAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(RightExpressionAST*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(CompoundAssignmentExpressionAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (ast->op != b->op) return false;
    if (!equivalence.same(ast->targetExpression, b->targetExpression))
      return false;
    return equivalence.same(ast->rightExpression, b->rightExpression);
  }

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast) const -> bool {
    auto b = counterpart(ast);
    if (ast->typeTrait != b->typeTrait) return false;
    auto aTypeId = ast->typeIdList;
    auto bTypeId = b->typeIdList;
    for (; aTypeId && bTypeId;
         aTypeId = aTypeId->next, bTypeId = bTypeId->next) {
      if (!equivalence.same(aTypeId->value, bTypeId->value)) return false;
    }
    return !aTypeId && !bTypeId;
  }

  [[nodiscard]] auto operator()(ConditionExpressionAST*) const -> bool {
    return false;
  }

  [[nodiscard]] auto operator()(EqualInitializerAST* ast) const -> bool {
    return equivalence.same(ast->expression, counterpart(ast)->expression);
  }

  [[nodiscard]] auto operator()(BracedInitListAST* ast) const -> bool {
    return equivalence.same(ast->expressionList,
                            counterpart(ast)->expressionList);
  }

  [[nodiscard]] auto operator()(ParenInitializerAST* ast) const -> bool {
    return equivalence.same(ast->expressionList,
                            counterpart(ast)->expressionList);
  }

  [[nodiscard]] auto operator()(ThreeWayComparisonExpressionAST* ast) const
      -> bool {
    return equivalence.same(ast->comparison, counterpart(ast)->comparison);
  }

  [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast) const
      -> bool {
    auto b = counterpart(ast);
    if (!sameDesignators(ast->designatorList, b->designatorList)) return false;
    return equivalence.same(ast->initializer, b->initializer);
  }

 private:
  [[nodiscard]] auto sameSplicer(SplicerAST* a, SplicerAST* b) const -> bool {
    if (!a || !b) return a == b;
    return equivalence.same(a->expression, b->expression);
  }

  [[nodiscard]] auto sameNewPlacement(NewPlacementAST* a,
                                      NewPlacementAST* b) const -> bool {
    if (!a || !b) return a == b;
    return equivalence.same(a->expressionList, b->expressionList);
  }

  [[nodiscard]] auto sameNewInitializer(NewInitializerAST* a,
                                        NewInitializerAST* b) const -> bool {
    if (!a || !b) return a == b;
    if (a->kind() != b->kind()) return false;
    if (auto paren = ast_cast<NewParenInitializerAST>(a)) {
      return equivalence.same(
          paren->expressionList,
          ast_cast<NewParenInitializerAST>(b)->expressionList);
    }
    return equivalence.same(
        ast_cast<NewBracedInitializerAST>(a)->bracedInitList,
        ast_cast<NewBracedInitializerAST>(b)->bracedInitList);
  }

  [[nodiscard]] auto sameDesignator(DesignatorAST* a, DesignatorAST* b) const
      -> bool {
    if (a->kind() != b->kind()) return false;
    if (auto dot = ast_cast<DotDesignatorAST>(a))
      return dot->identifier == ast_cast<DotDesignatorAST>(b)->identifier;
    return equivalence.same(ast_cast<SubscriptDesignatorAST>(a)->expression,
                            ast_cast<SubscriptDesignatorAST>(b)->expression);
  }

  [[nodiscard]] auto sameDesignators(List<DesignatorAST*>* a,
                                     List<DesignatorAST*>* b) const -> bool {
    for (; a && b; a = a->next, b = b->next) {
      if (!sameDesignator(a->value, b->value)) return false;
    }
    return !a && !b;
  }

  [[nodiscard]] auto sameAssociation(GenericAssociationAST* a,
                                     GenericAssociationAST* b) const -> bool {
    if (a->kind() != b->kind()) return false;
    if (auto typed = ast_cast<TypeGenericAssociationAST>(a)) {
      auto other = ast_cast<TypeGenericAssociationAST>(b);
      if (!equivalence.same(typed->typeId, other->typeId)) return false;
      return equivalence.same(typed->expression, other->expression);
    }
    return equivalence.same(
        ast_cast<DefaultGenericAssociationAST>(a)->expression,
        ast_cast<DefaultGenericAssociationAST>(b)->expression);
  }

  [[nodiscard]] auto sameAssociations(List<GenericAssociationAST*>* a,
                                      List<GenericAssociationAST*>* b) const
      -> bool {
    for (; a && b; a = a->next, b = b->next) {
      if (!sameAssociation(a->value, b->value)) return false;
    }
    return !a && !b;
  }
};

auto TemplateEquivalence::same(UnqualifiedIdAST* a, UnqualifiedIdAST* b) const
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (a->kind() != b->kind()) return false;
  return visit(SameUnqualifiedId{*this, b}, a);
}

auto TemplateEquivalence::same(ExpressionAST* a, ExpressionAST* b) const
    -> bool {
  a = writtenExpression(a);
  b = writtenExpression(b);
  if (a == b) return true;
  if (!a || !b) return false;
  if (a->kind() != b->kind()) return false;
  return visit(SameExpression{*this, b}, a);
}

auto TemplateEquivalence::same(List<ExpressionAST*>* a,
                               List<ExpressionAST*>* b) const -> bool {
  for (; a && b; a = a->next, b = b->next) {
    if (!same(a->value, b->value)) return false;
  }
  return !a && !b;
}

auto TemplateEquivalence::same(IdExpressionAST* a, IdExpressionAST* b) const
    -> bool {
  if (!a || !b) return a == b;
  if (namesParameter(a->symbol) || namesParameter(b->symbol))
    return sameEntity(a->symbol, b->symbol);
  if (!a->symbol || !b->symbol) {
    if (!same(a->nestedNameSpecifier, b->nestedNameSpecifier)) return false;
    return same(a->unqualifiedId, b->unqualifiedId);
  }
  if (!sameEntity(a->symbol, b->symbol)) return false;
  return same(visit(WrittenTemplateArguments{}, a->unqualifiedId),
              visit(WrittenTemplateArguments{}, b->unqualifiedId));
}

auto TemplateEquivalence::sameCallee(CallExpressionAST* a,
                                     CallExpressionAST* b) const -> bool {
  if (!callsDependentName(unit_, a) || !callsDependentName(unit_, b))
    return same(a->baseExpression, b->baseExpression);
  auto aCallee = ast_cast<IdExpressionAST>(a->baseExpression);
  auto bCallee = ast_cast<IdExpressionAST>(b->baseExpression);
  return same(aCallee->unqualifiedId, bCallee->unqualifiedId);
}

auto TemplateEquivalence::sameEntity(Symbol* a, Symbol* b) const -> bool {
  if (!a || !b) return !namesParameter(a) && !namesParameter(b);

  auto aTemplateParameter = template_parameter_info(a);
  auto bTemplateParameter = template_parameter_info(b);
  if (aTemplateParameter || bTemplateParameter) {
    if (!aTemplateParameter || !bTemplateParameter) return false;
    return correspondingParameters(*aTemplateParameter, *bTemplateParameter,
                                   correspondence_);
  }

  auto aParameter = symbol_cast<ParameterSymbol>(a);
  auto bParameter = symbol_cast<ParameterSymbol>(b);
  if (aParameter || bParameter) {
    if (!aParameter || !bParameter) return false;
    return correspondingFunctionParameters(aParameter, bParameter);
  }

  return resolve_using_declaration(a)->canonical() ==
         resolve_using_declaration(b)->canonical();
}

auto TemplateEquivalence::same(RequiresExpressionAST* a,
                               RequiresExpressionAST* b) const -> bool {
  auto aParameters = getParameterTypes(unit_, a->parameterDeclarationClause);
  auto bParameters = getParameterTypes(unit_, b->parameterDeclarationClause);
  if (aParameters.size() != bParameters.size()) return false;
  for (std::size_t i = 0; i < aParameters.size(); ++i) {
    if (!same(aParameters[i], bParameters[i])) return false;
  }

  auto aRequirement = a->requirementList;
  auto bRequirement = b->requirementList;
  for (; aRequirement && bRequirement;
       aRequirement = aRequirement->next, bRequirement = bRequirement->next) {
    if (aRequirement->value->kind() != bRequirement->value->kind())
      return false;
    if (!visit(SameRequirement{*this, bRequirement->value},
               aRequirement->value))
      return false;
  }
  return !aRequirement && !bRequirement;
}

auto TemplateEquivalence::same(TypeConstraintAST* a, TypeConstraintAST* b) const
    -> bool {
  if (a->identifier != b->identifier) return false;
  auto aScope =
      a->nestedNameSpecifier ? a->nestedNameSpecifier->symbol : nullptr;
  auto bScope =
      b->nestedNameSpecifier ? b->nestedNameSpecifier->symbol : nullptr;
  if (aScope != bScope) return false;
  return sameWritten(a->templateArgumentList, b->templateArgumentList);
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
    return correspondingParameters(*lhsInfo, *rhsInfo, correspondence);
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

  if (auto lhsDecltype = type_cast<DecltypeType>(lhs)) {
    auto rhsDecltype = type_cast<DecltypeType>(rhs);
    TemplateEquivalence equivalence{unit_, correspondence};
    return rhsDecltype && equivalence.same(lhsDecltype->expression(),
                                           rhsDecltype->expression());
  }

  auto lhsInfo = getTypeParamInfo(lhs);
  auto rhsInfo = getTypeParamInfo(rhs);
  if (lhsInfo || rhsInfo) {
    if (!lhsInfo || !rhsInfo) return false;
    return correspondingParameters(*lhsInfo, *rhsInfo, correspondence);
  }

  if (auto lhsExpansion = type_cast<PackExpansionType>(lhs)) {
    auto rhsExpansion = type_cast<PackExpansionType>(rhs);
    if (!rhsExpansion) return false;
    return recurse(lhsExpansion->pattern(), rhsExpansion->pattern());
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
        expand_template_arguments(class_template_arguments(unit_, lhsSym)),
        expand_template_arguments(class_template_arguments(unit_, rhsSym)),
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
      if (isPackExpansion(typeA->typeId) != isPackExpansion(typeB->typeId))
        return false;
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

auto TemplateEquivalence::sameQualifyingEntity(Symbol* a, Symbol* b) const
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!template_parameter_info(a) && !template_parameter_info(b)) return false;
  return same(a->type(), b->type());
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
      return sameQualifyingEntity(aSimple->symbol, bSimple->symbol);
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
      if (!TemplateEquivalence{unit_}.same(aTypeConstraint, bTypeConstraint))
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
