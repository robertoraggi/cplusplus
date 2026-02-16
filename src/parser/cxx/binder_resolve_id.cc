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

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/name_lookup.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbols.h>

namespace cxx {

namespace {

auto containsDependentType(const Type* type) -> bool;

struct ContainsDependentExpr {
  auto check(ExpressionAST* expr) const -> bool {
    if (!expr) return false;
    return visit(*this, expr);
  }

  auto operator()(SizeofPackExpressionAST*) const -> bool { return true; }
  auto operator()(PackExpansionExpressionAST*) const -> bool { return true; }
  auto operator()(FoldExpressionAST*) const -> bool { return true; }
  auto operator()(RightFoldExpressionAST*) const -> bool { return true; }
  auto operator()(LeftFoldExpressionAST*) const -> bool { return true; }

  auto operator()(IdExpressionAST* ast) const -> bool {
    if (symbol_cast<NonTypeParameterSymbol>(ast->symbol)) return true;
    if (symbol_cast<TypeParameterSymbol>(ast->symbol)) return true;
    if (symbol_cast<TemplateTypeParameterSymbol>(ast->symbol)) return true;
    return false;
  }

  auto operator()(BinaryExpressionAST* ast) const -> bool {
    return check(ast->leftExpression) || check(ast->rightExpression);
  }

  auto operator()(UnaryExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(CastExpressionAST* ast) const -> bool {
    if (ast->typeId && containsDependentType(ast->typeId->type)) return true;
    return check(ast->expression);
  }

  auto operator()(CppCastExpressionAST* ast) const -> bool {
    if (ast->typeId && containsDependentType(ast->typeId->type)) return true;
    return check(ast->expression);
  }

  auto operator()(ImplicitCastExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(NestedExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(ConditionalExpressionAST* ast) const -> bool {
    return check(ast->condition) || check(ast->iftrueExpression) ||
           check(ast->iffalseExpression);
  }

  auto operator()(MemberExpressionAST* ast) const -> bool {
    return check(ast->baseExpression);
  }

  auto operator()(CallExpressionAST* ast) const -> bool {
    if (check(ast->baseExpression)) return true;
    for (auto arg : ListView{ast->expressionList}) {
      if (check(arg)) return true;
    }
    return false;
  }

  auto operator()(SubscriptExpressionAST* ast) const -> bool {
    return check(ast->baseExpression) || check(ast->indexExpression);
  }

  auto operator()(PostIncrExpressionAST* ast) const -> bool {
    return check(ast->baseExpression);
  }

  auto operator()(TypeConstructionAST* ast) const -> bool {
    for (auto arg : ListView{ast->expressionList}) {
      if (check(arg)) return true;
    }
    return false;
  }

  auto operator()(SizeofExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(SizeofTypeExpressionAST* ast) const -> bool {
    return ast->typeId && containsDependentType(ast->typeId->type);
  }

  auto operator()(AlignofTypeExpressionAST* ast) const -> bool {
    return ast->typeId && containsDependentType(ast->typeId->type);
  }

  auto operator()(AlignofExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(TypeTraitExpressionAST* ast) const -> bool {
    for (auto typeId : ListView{ast->typeIdList}) {
      if (typeId && containsDependentType(typeId->type)) return true;
    }
    return false;
  }

  auto operator()(AssignmentExpressionAST* ast) const -> bool {
    return check(ast->leftExpression) || check(ast->rightExpression);
  }

  auto operator()(CompoundAssignmentExpressionAST* ast) const -> bool {
    return check(ast->leftExpression) || check(ast->rightExpression) ||
           check(ast->targetExpression);
  }

  auto operator()(NoexceptExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(AwaitExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(YieldExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(ThrowExpressionAST* ast) const -> bool {
    return check(ast->expression);
  }

  auto operator()(ExpressionAST* ast) const -> bool {
    if (ast && ast->type && containsDependentType(ast->type)) return true;
    return false;
  }
};

auto containsDependentType(const Type* type) -> bool {
  if (!type) return false;
  return visit(
      [](auto t) -> bool {
        using T = std::remove_cvref_t<decltype(*t)>;
        if constexpr (std::is_same_v<T, TypeParameterType> ||
                      std::is_same_v<T, TemplateTypeParameterType> ||
                      std::is_same_v<T, UnresolvedNameType> ||
                      std::is_same_v<T, UnresolvedBoundedArrayType> ||
                      std::is_same_v<T, UnresolvedUnderlyingType>) {
          return true;
        } else if constexpr (std::is_same_v<T, QualType> ||
                             std::is_same_v<T, PointerType> ||
                             std::is_same_v<T, LvalueReferenceType> ||
                             std::is_same_v<T, RvalueReferenceType> ||
                             std::is_same_v<T, BoundedArrayType> ||
                             std::is_same_v<T, UnboundedArrayType>) {
          return containsDependentType(t->elementType());
        } else if constexpr (std::is_same_v<T, FunctionType>) {
          if (containsDependentType(t->returnType())) return true;
          for (auto* param : t->parameterTypes()) {
            if (containsDependentType(param)) return true;
          }
          return false;
        } else if constexpr (std::is_same_v<T, MemberObjectPointerType>) {
          return containsDependentType(t->classType()) ||
                 containsDependentType(t->elementType());
        } else if constexpr (std::is_same_v<T, MemberFunctionPointerType>) {
          return containsDependentType(t->classType()) ||
                 containsDependentType(t->functionType());
        } else {
          return false;
        }
      },
      type);
}

}  // namespace

struct [[nodiscard]] Binder::ResolveUnqualifiedId {
  Binder& binder;
  NestedNameSpecifierAST* nestedNameSpecifier;
  UnqualifiedIdAST* unqualifiedId;
  bool checkTemplates;

  auto control() const -> Control* { return binder.control(); }
  auto inTemplate() const -> bool { return binder.inTemplate_; }

  auto resolveTemplateId(SimpleTemplateIdAST* templateId) -> Symbol*;
  auto shouldKeepTemplateIdAsDependent(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto hasDependentTemplateArguments(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto isDependentTypeArgument(TypeTemplateArgumentAST* typeArg) const -> bool;
  auto isDependentExpressionArgument(
      ExpressionTemplateArgumentAST* expressionArg) const -> bool;
  auto resolveClassTemplateId(SimpleTemplateIdAST* templateId,
                              ClassSymbol* classSymbol) -> Symbol*;
  auto resolveTypeAliasTemplateId(SimpleTemplateIdAST* templateId,
                                  TypeAliasSymbol* typeAliasSymbol) -> Symbol*;
  auto resolveNameId() -> Symbol*;

  auto resolve() -> Symbol*;
};

auto Binder::resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId, bool checkTemplates)
    -> Symbol* {
  return ResolveUnqualifiedId{*this, nestedNameSpecifier, unqualifiedId,
                              checkTemplates}
      .resolve();
}

auto Binder::ResolveUnqualifiedId::isDependentTypeArgument(
    TypeTemplateArgumentAST* typeArg) const -> bool {
  if (!typeArg || !typeArg->typeId) return false;
  return containsDependentType(typeArg->typeId->type);
}

auto Binder::ResolveUnqualifiedId::isDependentExpressionArgument(
    ExpressionTemplateArgumentAST* expressionArg) const -> bool {
  if (!expressionArg) return false;
  return ContainsDependentExpr{}.check(expressionArg->expression);
}

auto Binder::ResolveUnqualifiedId::hasDependentTemplateArguments(
    SimpleTemplateIdAST* templateId) const -> bool {
  for (auto arg : ListView{templateId->templateArgumentList}) {
    if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      if (isDependentTypeArgument(typeArg)) return true;
      continue;
    }

    auto expressionArg = ast_cast<ExpressionTemplateArgumentAST>(arg);
    if (isDependentExpressionArgument(expressionArg)) return true;
  }

  return false;
}

auto Binder::ResolveUnqualifiedId::shouldKeepTemplateIdAsDependent(
    SimpleTemplateIdAST* templateId) const -> bool {
  if (!inTemplate()) return false;
  if (symbol_cast<TemplateTypeParameterSymbol>(templateId->symbol)) {
    return true;
  }
  return hasDependentTemplateArguments(templateId);
}

auto Binder::ResolveUnqualifiedId::resolveClassTemplateId(
    SimpleTemplateIdAST* templateId, ClassSymbol* classSymbol) -> Symbol* {
  auto instance = ASTRewriter::instantiate(
      binder.unit_, templateId->templateArgumentList, classSymbol);

  if (instance) return instance;
  if (!classSymbol->templateDeclaration()) return instance;

  auto templateArgs = ASTRewriter::make_substitution(
      binder.unit_, classSymbol->templateDeclaration(),
      templateId->templateArgumentList);

  if (templateArgs.empty()) return instance;

  if (auto cached = classSymbol->findSpecialization(templateArgs)) {
    return cached;
  }

  auto parentScope = classSymbol->enclosingNonTemplateParametersScope();
  auto spec = control()->newClassSymbol(parentScope, {});
  spec->setName(classSymbol->name());
  classSymbol->addSpecialization(std::move(templateArgs), spec);
  return spec;
}

auto Binder::ResolveUnqualifiedId::resolveTypeAliasTemplateId(
    SimpleTemplateIdAST* templateId, TypeAliasSymbol* typeAliasSymbol)
    -> Symbol* {
  return ASTRewriter::instantiate(
      binder.unit_, templateId->templateArgumentList, typeAliasSymbol);
}

auto Binder::ResolveUnqualifiedId::resolveTemplateId(
    SimpleTemplateIdAST* templateId) -> Symbol* {
  if (!checkTemplates) return templateId->symbol;
  if (shouldKeepTemplateIdAsDependent(templateId)) return templateId->symbol;

  if (auto classSymbol = symbol_cast<ClassSymbol>(templateId->symbol)) {
    return resolveClassTemplateId(templateId, classSymbol);
  }

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(templateId->symbol)) {
    return resolveTypeAliasTemplateId(templateId, typeAliasSymbol);
  }

  return templateId->symbol;
}

auto Binder::ResolveUnqualifiedId::resolveNameId() -> Symbol* {
  auto name = ast_cast<NameIdAST>(unqualifiedId);
  if (!name) return nullptr;

  auto symbol =
      Lookup{binder.scope()}.lookupType(nestedNameSpecifier, name->identifier);
  if (!is_type(symbol)) return nullptr;
  return symbol;
}

auto Binder::ResolveUnqualifiedId::resolve() -> Symbol* {
  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    return resolveTemplateId(templateId);
  }

  return resolveNameId();
}

}  // namespace cxx