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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/binder.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <format>

namespace cxx {

struct ASTRewriter::ExpressionVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(UserDefinedStringLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(ThisExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NestedStatementExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(VaArgExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SubscriptExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(CallExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeConstructionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(MemberExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeidExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(SpliceExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(ReflectExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(LabelAddressExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(AwaitExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(NewExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(DeleteExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(CastExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ConditionalExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(YieldExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ThrowExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(TargetExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(RightExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(CompoundAssignmentExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast)
      -> ExpressionAST*;

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ConditionExpressionAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(EqualInitializerAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(BracedInitListAST* ast) -> ExpressionAST*;

  [[nodiscard]] auto operator()(ParenInitializerAST* ast) -> ExpressionAST*;
};

struct ASTRewriter::NewInitializerVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerAST*;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerAST*;
};

struct ASTRewriter::GenericAssociationVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(DefaultGenericAssociationAST* ast)
      -> GenericAssociationAST*;

  [[nodiscard]] auto operator()(TypeGenericAssociationAST* ast)
      -> GenericAssociationAST*;
};

struct ASTRewriter::LambdaCaptureVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(ThisLambdaCaptureAST* ast) -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(DerefThisLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(SimpleLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(RefLambdaCaptureAST* ast) -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(RefInitLambdaCaptureAST* ast)
      -> LambdaCaptureAST*;

  [[nodiscard]] auto operator()(InitLambdaCaptureAST* ast) -> LambdaCaptureAST*;
};

auto ASTRewriter::expression(ExpressionAST* ast) -> ExpressionAST* {
  if (!ast) return {};
  auto expr = visit(ExpressionVisitor{*this}, ast);
  if (expr) check(expr);
  return expr;
}

auto ASTRewriter::newInitializer(NewInitializerAST* ast) -> NewInitializerAST* {
  if (!ast) return {};
  return visit(NewInitializerVisitor{*this}, ast);
}

auto ASTRewriter::genericAssociation(GenericAssociationAST* ast)
    -> GenericAssociationAST* {
  if (!ast) return {};
  return visit(GenericAssociationVisitor{*this}, ast);
}

auto ASTRewriter::lambdaCapture(LambdaCaptureAST* ast) -> LambdaCaptureAST* {
  if (!ast) return {};
  return visit(LambdaCaptureVisitor{*this}, ast);
}

auto ASTRewriter::newPlacement(NewPlacementAST* ast) -> NewPlacementAST* {
  if (!ast) return {};

  auto copy = NewPlacementAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::lambdaSpecifier(LambdaSpecifierAST* ast)
    -> LambdaSpecifierAST* {
  if (!ast) return {};

  auto copy = LambdaSpecifierAST::create(arena());

  copy->specifierLoc = ast->specifierLoc;
  copy->specifier = ast->specifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CharLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = BoolLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->isTrue = ast->isTrue;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = IntLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = FloatLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = NullptrLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = StringLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionAST* {
  auto copy = UserDefinedStringLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ObjectLiteralExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ThisExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GenericSelectionExpressionAST* ast) -> ExpressionAST* {
  auto copy = GenericSelectionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->genericLoc = ast->genericLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;

  for (auto genericAssociationList = &copy->genericAssociationList;
       auto node : ListView{ast->genericAssociationList}) {
    auto value = rewrite.genericAssociation(node);
    *genericAssociationList = make_list_node(arena(), value);
    genericAssociationList = &(*genericAssociationList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->matchedAssocIndex = ast->matchedAssocIndex;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionAST* {
  auto copy = NestedStatementExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NestedExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionAST* {
  if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol);
      param && param->depth() == rewrite.depth_ &&
      param->index() < rewrite.templateArguments_.size()) {
    auto symbolPtr =
        std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

    if (!symbolPtr) {
      cxx_runtime_error("expected initializer for non-type template parameter");
    }

    auto parameterPack = symbol_cast<ParameterPackSymbol>(*symbolPtr);

    if (parameterPack && parameterPack == rewrite.parameterPack_ &&
        rewrite.elementIndex_.has_value()) {
      auto idx = rewrite.elementIndex_.value();
      auto element = parameterPack->elements()[idx];
      if (auto var = symbol_cast<VariableSymbol>(element)) {
        return rewrite.expression(var->initializer());
      }
    }
  }

  auto copy = IdExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol);
      param && param->depth() == rewrite.depth_ &&
      param->index() < rewrite.templateArguments_.size()) {
    auto symbolPtr =
        std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

    if (!symbolPtr) {
      cxx_runtime_error("expected initializer for non-type template parameter");
    }

    copy->symbol = *symbolPtr;

  } else {
    binder()->bind(copy);
  }

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = LambdaExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->captureDefaultLoc = ast->captureDefaultLoc;

  for (auto captureList = &copy->captureList;
       auto node : ListView{ast->captureList}) {
    auto value = rewrite.lambdaCapture(node);
    *captureList = make_list_node(arena(), value);
    captureList = &(*captureList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite.templateParameter(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->templateRequiresClause =
      rewrite.requiresClause(ast->templateRequiresClause);

  for (auto expressionAttributeList = &copy->expressionAttributeList;
       auto node : ListView{ast->expressionAttributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *expressionAttributeList = make_list_node(arena(), value);
    expressionAttributeList = &(*expressionAttributeList)->next;
  }

  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  for (auto gnuAtributeList = &copy->gnuAtributeList;
       auto node : ListView{ast->gnuAtributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *gnuAtributeList = make_list_node(arena(), value);
    gnuAtributeList = &(*gnuAtributeList)->next;
  }

  for (auto lambdaSpecifierList = &copy->lambdaSpecifierList;
       auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = rewrite.lambdaSpecifier(node);
    *lambdaSpecifierList = make_list_node(arena(), value);
    lambdaSpecifierList = &(*lambdaSpecifierList)->next;
  }

  {
    auto _ = Binder::ScopeGuard(binder());

    if (copy->parameterDeclarationClause) {
      binder()->setScope(
          copy->parameterDeclarationClause->functionParametersSymbol);
    }

    copy->exceptionSpecifier =
        rewrite.exceptionSpecifier(ast->exceptionSpecifier);

    for (auto attributeList = &copy->attributeList;
         auto node : ListView{ast->attributeList}) {
      auto value = rewrite.attributeSpecifier(node);
      *attributeList = make_list_node(arena(), value);
      attributeList = &(*attributeList)->next;
    }

    copy->trailingReturnType =
        rewrite.trailingReturnType(ast->trailingReturnType);
    copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  }

  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));
  copy->captureDefault = ast->captureDefault;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = FoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->foldOpLoc = ast->foldOpLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;
  copy->foldOp = ast->foldOp;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = RightFoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->opLoc = ast->opLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionAST* {
  if (auto parameterPack = rewrite.getParameterPack(ast->expression)) {
    auto savedParameterPack = rewrite.parameterPack_;
    std::swap(rewrite.parameterPack_, parameterPack);

    std::vector<ExpressionAST*> instantiations;
    ExpressionAST* current = nullptr;

    int n = 0;
    for (auto element : rewrite.parameterPack_->elements()) {
      std::optional<int> index{n};
      std::swap(rewrite.elementIndex_, index);

      auto expression = rewrite.expression(ast->expression);
      if (!current) {
        current = expression;
      } else {
        auto binop = BinaryExpressionAST::create(arena());
        binop->valueCategory = current->valueCategory;
        binop->type = current->type;
        binop->leftExpression = current;
        binop->op = ast->op;
        binop->opLoc = ast->opLoc;
        binop->rightExpression = expression;
        current = binop;
      }

      std::swap(rewrite.elementIndex_, index);
      ++n;
    }

    std::swap(rewrite.parameterPack_, parameterPack);

    return current;
  }

  auto copy = LeftFoldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = RequiresExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->requiresLoc = ast->requiresLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto requirementList = &copy->requirementList;
       auto node : ListView{ast->requirementList}) {
    auto value = rewrite.requirement(node);
    *requirementList = make_list_node(arena(), value);
    requirementList = &(*requirementList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = VaArgExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->vaArgLoc = ast->vaArgLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SubscriptExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->lbracketLoc = ast->lbracketLoc;
  copy->indexExpression = rewrite.expression(ast->indexExpression);
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CallExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeConstructionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite.specifier(ast->typeSpecifier);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionAST* {
  auto copy = BracedTypeConstructionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeSpecifier = rewrite.specifier(ast->typeSpecifier);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SpliceMemberExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->templateLoc = ast->templateLoc;
  copy->splicer = rewrite.splicer(ast->splicer);
  copy->symbol = ast->symbol;
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = MemberExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->accessLoc = ast->accessLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->symbol = ast->symbol;
  copy->accessOp = ast->accessOp;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = PostIncrExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->baseExpression = rewrite.expression(ast->baseExpression);
  copy->opLoc = ast->opLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CppCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->castLoc = ast->castLoc;
  copy->lessLoc = ast->lessLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->greaterLoc = ast->greaterLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionAST* {
  auto copy = BuiltinBitCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->castLoc = ast->castLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionAST* {
  auto copy = BuiltinOffsetofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->offsetofLoc = ast->offsetofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->commaLoc = ast->commaLoc;
  copy->identifierLoc = ast->identifierLoc;

  for (auto designatorList = &copy->designatorList;
       auto node : ListView{ast->designatorList}) {
    auto value = rewrite.designator(node);
    *designatorList = make_list_node(arena(), value);
    designatorList = &(*designatorList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeidExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeidOfTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeidLoc = ast->typeidLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SpliceExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->splicer = rewrite.splicer(ast->splicer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = GlobalScopeReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionAST* {
  auto copy = NamespaceReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeIdReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeIdReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ReflectExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->caretCaretLoc = ast->caretCaretLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = LabelAddressExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->ampAmpLoc = ast->ampAmpLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = UnaryExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->opLoc = ast->opLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AwaitExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->awaitLoc = ast->awaitLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SizeofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SizeofTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = SizeofPackExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->sizeofLoc = ast->sizeofLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->rparenLoc = ast->rparenLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofTypeExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AlignofTypeExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AlignofExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->alignofLoc = ast->alignofLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NoexceptExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = NewExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->newLoc = ast->newLoc;
  copy->newPlacement = rewrite.newPlacement(ast->newPlacement);
  copy->lparenLoc = ast->lparenLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite.specifier(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->rparenLoc = ast->rparenLoc;
  copy->newInitalizer = rewrite.newInitializer(ast->newInitalizer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = DeleteExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->scopeLoc = ast->scopeLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->lbracketLoc = ast->lbracketLoc;
  copy->rbracketLoc = ast->rbracketLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = CastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->rparenLoc = ast->rparenLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ImplicitCastExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite.expression(ast->expression);
  copy->castKind = ast->castKind;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = BinaryExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ConditionalExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->condition = rewrite.expression(ast->condition);
  copy->questionLoc = ast->questionLoc;
  copy->iftrueExpression = rewrite.expression(ast->iftrueExpression);
  copy->colonLoc = ast->colonLoc;
  copy->iffalseExpression = rewrite.expression(ast->iffalseExpression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = YieldExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->yieldLoc = ast->yieldLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ThrowExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->throwLoc = ast->throwLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = AssignmentExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->opLoc = ast->opLoc;
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TargetExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TargetExpressionAST::create(arena());
  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(RightExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = RightExpressionAST::create(arena());
  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionAST* {
  auto copy = CompoundAssignmentExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->targetExpression = rewrite.expression(ast->targetExpression);
  copy->opLoc = ast->opLoc;
  copy->leftExpression = rewrite.expression(ast->leftExpression);
  copy->rightExpression = rewrite.expression(ast->rightExpression);
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = PackExpansionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->expression = rewrite.expression(ast->expression);
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionAST* {
  auto copy = DesignatedInitializerClauseAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  for (auto designatorList = &copy->designatorList;
       auto node : ListView{ast->designatorList}) {
    auto value = rewrite.designator(node);
    *designatorList = make_list_node(arena(), value);
    designatorList = &(*designatorList)->next;
  }

  copy->initializer = rewrite.expression(ast->initializer);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = TypeTraitExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->typeTraitLoc = ast->typeTraitLoc;
  copy->lparenLoc = ast->lparenLoc;

  for (auto typeIdList = &copy->typeIdList;
       auto node : ListView{ast->typeIdList}) {
    auto value = rewrite.typeId(node);
    *typeIdList = make_list_node(arena(), value);
    typeIdList = &(*typeIdList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->typeTrait = ast->typeTrait;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionAST* {
  auto copy = ConditionExpressionAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite.specifier(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());
  copy->initializer = rewrite.expression(ast->initializer);
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = EqualInitializerAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionAST* {
  auto copy = BracedInitListAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->commaLoc = ast->commaLoc;
  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionAST* {
  auto copy = ParenInitializerAST::create(arena());

  copy->valueCategory = ast->valueCategory;
  copy->type = ast->type;
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerAST* {
  auto copy = NewParenInitializerAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerAST* {
  auto copy = NewBracedInitializerAST::create(arena());

  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));

  return copy;
}

auto ASTRewriter::GenericAssociationVisitor::operator()(
    DefaultGenericAssociationAST* ast) -> GenericAssociationAST* {
  auto copy = DefaultGenericAssociationAST::create(arena());

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::GenericAssociationVisitor::operator()(
    TypeGenericAssociationAST* ast) -> GenericAssociationAST* {
  auto copy = TypeGenericAssociationAST::create(arena());

  copy->typeId = rewrite.typeId(ast->typeId);
  copy->colonLoc = ast->colonLoc;
  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = ThisLambdaCaptureAST::create(arena());

  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureAST* {
  auto copy = DerefThisLambdaCaptureAST::create(arena());

  copy->starLoc = ast->starLoc;
  copy->thisLoc = ast->thisLoc;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = SimpleLambdaCaptureAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = RefLambdaCaptureAST::create(arena());

  copy->ampLoc = ast->ampLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = RefInitLambdaCaptureAST::create(arena());

  copy->ampLoc = ast->ampLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureAST* {
  auto copy = InitLambdaCaptureAST::create(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->identifier = ast->identifier;

  return copy;
}

}  // namespace cxx
