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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

namespace cxx {
struct ASTRewriter::StatementVisitor : VisitorBase {
  [[nodiscard]] auto typeChecker() -> TypeChecker {
    return rewrite.typeChecker();
  }

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

struct ASTRewriter::MemInitializerVisitor : VisitorBase {
  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerAST*;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerAST*;

  void resolveBase(MemInitializerAST* ast, MemInitializerAST* copy,
                   NestedNameSpecifierAST* nestedNameSpecifier,
                   UnqualifiedIdAST* unqualifiedId);
};

struct ASTRewriter::ExceptionDeclarationVisitor : VisitorBase {
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

namespace {
[[nodiscard]] auto isPackExpansion(MemInitializerAST* ast) -> bool {
  if (auto paren = ast_cast<ParenMemInitializerAST>(ast))
    return bool(paren->ellipsisLoc);
  if (auto braced = ast_cast<BracedMemInitializerAST>(ast))
    return bool(braced->ellipsisLoc);
  return false;
}

void clearPackExpansion(MemInitializerAST* ast) {
  if (auto paren = ast_cast<ParenMemInitializerAST>(ast))
    paren->ellipsisLoc = {};
  else if (auto braced = ast_cast<BracedMemInitializerAST>(ast))
    braced->ellipsisLoc = {};
}
}  // namespace

auto ASTRewriter::rewriteMemInitializerList(List<MemInitializerAST*>* source)
    -> List<MemInitializerAST*>* {
  List<MemInitializerAST*>* result = nullptr;
  ListAppender<MemInitializerAST> append{arena(), result};

  for (auto node : ListView{source}) {
    if (isPackExpansion(node) && !expandsAnActivePack(node)) {
      if (hasUnresolvedParameterPack(node)) {
        append(memInitializer(node));
        continue;
      }

      auto pack = findReferencedParameterPack(memInitializerId(node));
      if (auto base = symbol_cast<BaseClassSymbol>(node->symbol); !pack && base)
        pack = parameterPackFor(base->symbol());

      if (pack) {
        forEachPackElement(
            node, node->firstSourceLocation(),
            [&] {
              auto value = memInitializer(node);
              clearPackExpansion(value);
              append(value);
            },
            pack);
        continue;
      }
    }

    append(memInitializer(node));
  }

  return result;
}

void ASTRewriter::MemInitializerVisitor::resolveBase(
    MemInitializerAST* ast, MemInitializerAST* copy,
    NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId) {
  Symbol* baseClass = nullptr;

  if (auto base = symbol_cast<BaseClassSymbol>(ast->symbol)) {
    baseClass = rewrite.substitutedTemplateParameterClass(base->symbol());
    if (!baseClass && symbol_cast<TypeParameterSymbol>(base->symbol()))
      copy->symbol = ast->symbol;
    if (!baseClass && symbol_cast<ClassSymbol>(base->symbol()))
      baseClass = base->symbol();
  }

  if (!baseClass && symbol_cast<TypeParameterSymbol>(ast->symbol)) {
    baseClass = rewrite.substitutedTemplateParameterClass(ast->symbol);
    if (!baseClass) copy->symbol = ast->symbol;
  }

  if (!baseClass && ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    auto resolved = binder()->resolve(nestedNameSpecifier, unqualifiedId,
                                      translationUnit()->config().checkTypes);
    if (resolved && resolved->isClass()) baseClass = resolved;
  }

  if (!baseClass) return;

  ClassSymbol* enclosingClass = nullptr;
  for (auto scope = binder()->scope(); scope; scope = scope->parent()) {
    enclosingClass = symbol_cast<ClassSymbol>(scope);
    if (enclosingClass) break;
  }
  if (!enclosingClass) return;

  auto target = symbol_cast<ClassSymbol>(baseClass)->resolvedDefinition();

  for (auto base : enclosingClass->resolvedDefinition()->baseClasses()) {
    auto candidate = symbol_cast<ClassSymbol>(base->symbol());
    if (candidate && candidate->resolvedDefinition() == target) {
      copy->symbol = base;
      return;
    }
  }
}

auto ASTRewriter::exceptionDeclaration(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationAST* {
  if (!ast) return {};
  return visit(ExceptionDeclarationVisitor{*this}, ast);
}

auto ASTRewriter::asmOperand(AsmOperandAST* ast) -> AsmOperandAST* {
  auto copy = AsmOperandAST::create(arena());

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
  auto copy = AsmQualifierAST::create(arena());

  copy->qualifierLoc = ast->qualifierLoc;
  copy->qualifier = ast->qualifier;

  return copy;
}

auto ASTRewriter::asmClobber(AsmClobberAST* ast) -> AsmClobberAST* {
  auto copy = AsmClobberAST::create(arena());

  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::asmGotoLabel(AsmGotoLabelAST* ast) -> AsmGotoLabelAST* {
  auto copy = AsmGotoLabelAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::handler(HandlerAST* ast) -> HandlerAST* {
  if (!ast) return {};

  auto copy = HandlerAST::create(arena());

  auto _ = Binder::ScopeGuard(&binder_);

  if (ast->symbol) {
    copy->symbol = binder_.enterBlock(ast->symbol->location());
  }

  copy->catchLoc = ast->catchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->exceptionDeclaration = exceptionDeclaration(ast->exceptionDeclaration);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = ast_cast<CompoundStatementAST>(statement(ast->statement));

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementAST* {
  auto copy = LabeledStatementAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->colonLoc = ast->colonLoc;
  copy->identifier = ast->identifier;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementAST* {
  auto copy = CaseStatementAST::create(arena());

  copy->caseLoc = ast->caseLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->colonLoc = ast->colonLoc;

  copy->caseValue = ast->caseValue;
  if (copy->expression) {
    auto interp = ASTInterpreter{translationUnit()};
    if (auto value = interp.evaluate(copy->expression)) {
      if (translationUnit()->typeTraits().is_unsigned(copy->expression->type)) {
        if (auto v = interp.toUInt(*value)) copy->caseValue = *v;
      } else if (auto v = interp.toInt(*value)) {
        copy->caseValue = *v;
      }
    }
  }

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementAST* {
  auto copy = DefaultStatementAST::create(arena());

  copy->defaultLoc = ast->defaultLoc;
  copy->colonLoc = ast->colonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementAST* {
  auto copy = ExpressionStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementAST* {
  auto copy = CompoundStatementAST::create(arena());

  auto _ = Binder::ScopeGuard(binder());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->lbraceLoc = ast->lbraceLoc;

  copy->statementList =
      rewrite.rewriteList(ast->statementList, &ASTRewriter::statement);

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementAST* {
  auto copy = IfStatementAST::create(arena());

  auto _ = Binder::ScopeGuard(binder());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->ifLoc = ast->ifLoc;
  copy->constexprLoc = ast->constexprLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  typeChecker().check_bool_condition(copy->condition);
  copy->rparenLoc = ast->rparenLoc;

  std::optional<bool> constexprValue;
  if (ast->constexprLoc && copy->condition &&
      !isDependent(rewrite.unit_, copy->condition)) {
    auto interp = ASTInterpreter{rewrite.unit_};
    if (auto val = interp.evaluate(copy->condition)) {
      constexprValue = interp.toBool(*val);
    }
  }

  copy->elseLoc = ast->elseLoc;
  if (constexprValue.has_value()) {
    if (*constexprValue) {
      copy->statement = rewrite.statement(ast->statement);
    } else {
      copy->elseStatement = rewrite.statement(ast->elseStatement);
    }
  } else {
    copy->statement = rewrite.statement(ast->statement);
    copy->elseStatement = rewrite.statement(ast->elseStatement);
  }

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementAST* {
  auto copy = ConstevalIfStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->ifLoc = ast->ifLoc;
  copy->exclaimLoc = ast->exclaimLoc;
  copy->constvalLoc = ast->constvalLoc;

  {
    TranslationUnit::ImmediateFunctionContextScope immediate{translationUnit(),
                                                             !ast->isNot};
    copy->statement = rewrite.statement(ast->statement);
  }

  copy->elseLoc = ast->elseLoc;

  {
    TranslationUnit::ImmediateFunctionContextScope immediate{translationUnit(),
                                                             ast->isNot};
    copy->elseStatement = rewrite.statement(ast->elseStatement);
  }

  copy->isNot = ast->isNot;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementAST* {
  auto copy = SwitchStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->switchLoc = ast->switchLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  typeChecker().check_integral_condition(copy->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementAST* {
  auto copy = WhileStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->condition = rewrite.expression(ast->condition);
  typeChecker().check_bool_condition(copy->condition);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementAST* {
  auto copy = DoStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->doLoc = ast->doLoc;
  copy->statement = rewrite.statement(ast->statement);
  copy->whileLoc = ast->whileLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  typeChecker().check_bool_condition(copy->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementAST* {
  auto copy = ForRangeStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->rangeInitializer = rewrite.expression(ast->rangeInitializer);
  {
    auto restore = std::exchange(rewrite.rewritingForRangeDeclaration_, true);
    copy->rangeDeclaration = rewrite.declaration(ast->rangeDeclaration);
    rewrite.rewritingForRangeDeclaration_ = restore;
  }
  copy->colonLoc = ast->colonLoc;

  auto rangeDeclSpecifierList = [&]() -> List<SpecifierAST*>* {
    if (auto simpleDecl =
            ast_cast<SimpleDeclarationAST>(copy->rangeDeclaration))
      return simpleDecl->declSpecifierList;
    if (auto structuredBinding =
            ast_cast<StructuredBindingDeclarationAST>(copy->rangeDeclaration))
      return structuredBinding->declSpecifierList;
    return nullptr;
  }();

  auto rangeSpecs = DeclSpecs{rewrite.unit_};
  for (auto node : ListView{rangeDeclSpecifierList}) rangeSpecs.accept(node);
  rangeSpecs.finish();

  binder()->finishForRangeDeclaration(copy, rangeSpecs);

  rewrite.remapStructuredBindingSymbols(
      ast_cast<StructuredBindingDeclarationAST>(ast->rangeDeclaration),
      ast_cast<StructuredBindingDeclarationAST>(copy->rangeDeclaration));

  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementAST* {
  auto copy = ForStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto _ = Binder::ScopeGuard(binder());

  if (ast->symbol) {
    copy->symbol = binder()->enterBlock(ast->symbol->location());
  }

  copy->forLoc = ast->forLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->initializer = rewrite.statement(ast->initializer);
  copy->condition = rewrite.expression(ast->condition);
  typeChecker().check_bool_condition(copy->condition);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;
  copy->statement = rewrite.statement(ast->statement);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementAST* {
  auto copy = BreakStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->breakLoc = ast->breakLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementAST* {
  auto copy = ContinueStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->continueLoc = ast->continueLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = ReturnStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->returnLoc = ast->returnLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  typeChecker().check_return_statement(copy);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(CoroutineReturnStatementAST* ast)
    -> StatementAST* {
  auto copy = CoroutineReturnStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->coreturnLoc = ast->coreturnLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementAST* {
  auto copy = GotoStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->expression = rewrite.expression(ast->expression);
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
  auto copy = DeclarationStatementAST::create(arena());

  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementAST* {
  auto copy = TryBlockStatementAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->tryLoc = ast->tryLoc;
  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  copy->handlerList =
      rewrite.rewriteList(ast->handlerList, &ASTRewriter::handler);

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerAST* {
  auto copy = ParenMemInitializerAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->lparenLoc = ast->lparenLoc;

  copy->expressionList = rewrite.rewriteExpressionList(ast->expressionList);

  copy->rparenLoc = ast->rparenLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;

  resolveBase(ast, copy, copy->nestedNameSpecifier, copy->unqualifiedId);

  return copy;
}

auto ASTRewriter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerAST* {
  auto copy = BracedMemInitializerAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->bracedInitList =
      ast_cast<BracedInitListAST>(rewrite.expression(ast->bracedInitList));
  copy->ellipsisLoc = ast->ellipsisLoc;

  resolveBase(ast, copy, copy->nestedNameSpecifier, copy->unqualifiedId);

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = EllipsisExceptionDeclarationAST::create(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;

  return copy;
}

auto ASTRewriter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationAST* {
  auto copy = TypeExceptionDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto typeSpecifierListCtx = DeclSpecs{rewrite.unit_};
  copy->typeSpecifierList = rewrite.rewriteSpecifierList(ast->typeSpecifierList,
                                                         typeSpecifierListCtx);

  copy->declarator = rewrite.declarator(ast->declarator);

  binder()->bind(copy, Decl{typeSpecifierListCtx, copy->declarator});

  if (ast->symbol && copy->symbol) {
    rewrite.addSymbolRemap(ast->symbol, copy->symbol);
  }

  return copy;
}
}  // namespace cxx
