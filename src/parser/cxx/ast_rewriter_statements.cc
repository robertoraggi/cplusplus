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
#include <cxx/binder.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/symbols.h>

namespace cxx {

struct ASTRewriter::StatementVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(SwitchStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(WhileStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(DoStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ForRangeStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ForStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(BreakStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ContinueStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(ReturnStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(CoroutineReturnStatementAST* ast)
      -> StatementAST*;

  [[nodiscard]] auto operator()(GotoStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(DeclarationStatementAST* ast) -> StatementAST*;

  [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementAST*;
};

struct ASTRewriter::MemInitializerVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerAST*;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerAST*;
};

struct ASTRewriter::ExceptionDeclarationVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
};

auto ASTRewriter::statement(StatementAST* ast) -> StatementAST* {
  if (!ast) return {};
  return visit(StatementVisitor{*this}, ast);
}

auto ASTRewriter::memInitializer(MemInitializerAST* ast) -> MemInitializerAST* {
  if (!ast) return {};
  return visit(MemInitializerVisitor{*this}, ast);
}

auto ASTRewriter::exceptionDeclaration(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationAST* {
  if (!ast) return {};
  return visit(ExceptionDeclarationVisitor{*this}, ast);
}

auto ASTRewriter::asmOperand(AsmOperandAST* ast) -> AsmOperandAST* {
  auto copy = make_node<AsmOperandAST>(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->symbolicNameLoc = ast->symbolicNameLoc;
  copy->rbracketLoc = ast->rbracketLoc;
  copy->constraintLiteralLoc = ast->constraintLiteralLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->symbolicName = ast->symbolicName;
  copy->constraintLiteral = ast->constraintLiteral;

  return copy;
}

auto ASTRewriter::asmQualifier(AsmQualifierAST* ast) -> AsmQualifierAST* {
  auto copy = make_node<AsmQualifierAST>(arena());

  copy->qualifierLoc = ast->qualifierLoc;
  copy->qualifier = ast->qualifier;

  return copy;
}

auto ASTRewriter::asmClobber(AsmClobberAST* ast) -> AsmClobberAST* {
  auto copy = make_node<AsmClobberAST>(arena());

  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::asmGotoLabel(AsmGotoLabelAST* ast) -> AsmGotoLabelAST* {
  auto copy = make_node<AsmGotoLabelAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::handler(HandlerAST* ast) -> HandlerAST* {
  if (!ast) return {};

  auto copy = make_node<HandlerAST>(arena());

  copy->catchLoc = ast->catchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->exceptionDeclaration = exceptionDeclaration(ast->exceptionDeclaration);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(statement(ast->statement));

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<LabeledStatementAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->colonLoc = ast->colonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CaseStatementAST>(arena());

  copy->caseLoc = ast->caseLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DefaultStatementAST>(arena());

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ExpressionStatementAST>(arena());

  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CompoundStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto statementList = &copy->statementList;
       auto node : ListView{ast->statementList}) {
    auto value = rewrite.statement(node);
    *statementList = make_list_node(arena(), value);
    statementList = &(*statementList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<IfStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->ifLoc = ast->ifLoc;
  copy->constexprLoc = ast->constexprLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);
  copy->elseLoc = ast->elseLoc;
  copy->elseStatement = rewrite.statement(ast->elseStatement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ConstevalIfStatementAST>(arena());

  copy->ifLoc = ast->ifLoc;
  copy->exclaimLoc = ast->exclaimLoc;
  copy->constvalLoc = ast->constvalLoc;
  copy->statement = rewrite.statement(ast->statement);
  copy->elseLoc = ast->elseLoc;
  copy->elseStatement = rewrite.statement(ast->elseStatement);
  copy->isNot = ast->isNot;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<SwitchStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->switchLoc = ast->switchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<WhileStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->condition = rewrite.expression(ast->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DoStatementAST>(arena());

  copy->doLoc = ast->doLoc;
  copy->statement = rewrite.statement(ast->statement);
  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ForRangeStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->rangeDeclaration = rewrite.declaration(ast->rangeDeclaration);
  copy->colonLoc = ast->colonLoc;
  copy->rangeInitializer = rewrite.expression(ast->rangeInitializer);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ForStatementAST>(arena());

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<BreakStatementAST>(arena());

  copy->breakLoc = ast->breakLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ContinueStatementAST>(arena());

  copy->continueLoc = ast->continueLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<ReturnStatementAST>(arena());

  copy->returnLoc = ast->returnLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CoroutineReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<CoroutineReturnStatementAST>(arena());

  copy->coreturnLoc = ast->coreturnLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<GotoStatementAST>(arena());

  copy->gotoLoc = ast->gotoLoc;
  copy->starLoc = ast->starLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->isIndirect = ast->isIndirect;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<DeclarationStatementAST>(arena());

  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementAST* {
  auto copy = make_node<TryBlockStatementAST>(arena());

  copy->tryLoc = ast->tryLoc;
  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  for (auto handlerList = &copy->handlerList;
       auto node : ListView{ast->handlerList}) {
    auto value = rewrite.handler(node);
    *handlerList = make_list_node(arena(), value);
    handlerList = &(*handlerList)->next;
  }

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerAST* {
  auto copy = make_node<ParenMemInitializerAST>(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->lparenLoc = ast->lparenLoc;

  for (auto expressionList = &copy->expressionList;
       auto node : ListView{ast->expressionList}) {
    auto value = rewrite.expression(node);
    *expressionList = make_list_node(arena(), value);
    expressionList = &(*expressionList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerAST* {
  auto copy = make_node<BracedMemInitializerAST>(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));
  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = make_node<EllipsisExceptionDeclarationAST>(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = make_node<TypeExceptionDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

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

  return copy;
}

}  // namespace cxx
