// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Codegen.h"
#include "AST.h"
#include "TranslationUnit.h"
#include "IR.h"
#include "Control.h"
#include <iostream>
#include <cassert>

Codegen::Codegen(TranslationUnit* unit): unit(unit) {
  _module = new IR::Module();
}

Codegen::~Codegen() {
  delete _module;
}

Control* Codegen::control() const {
  return unit->control();
}

void Codegen::operator()(FunctionDefinitionAST* ast) {
  accept(ast);
}

Codegen::Result Codegen::reduce(const Result& expr) {
  if (expr->isTemp())
    return expr;
  auto t = newTemp();
  _block->emitMove(t, *expr);
  return Result{t};
}

Codegen::Result Codegen::expression(ExpressionAST* ast) {
  Result r{ex};
  if (ast) {
    std::swap(_result, r);
    accept(ast);
    std::swap(_result, r);
  }
  if (! r.is(ex)) {
    return Result{_function->getConst("@@expr@@")};
  }
  return r;
}

void Codegen::condition(ExpressionAST* ast,
                        IR::BasicBlock* iftrue,
                        IR::BasicBlock* iffalse) {
  Result r{iftrue, iffalse};
  if (ast) {
    std::swap(_result, r);
    accept(ast);
    std::swap(_result, r);
  }
  if (r.is(ex)) {
    _block->emitCJump(*r, iftrue, iffalse);
    return;
  }
  if (r.is(nx)) {
    _block->emitCJump(_function->getConst("@@condi@@"), iftrue, iffalse);
  }
}

void Codegen::statement(ExpressionAST* ast) {
  Result r{nx};
  if (ast) {
    std::swap(_result, r);
    accept(ast);
    std::swap(_result, r);
  }
  if (r.is(ex)) {
    _block->emitExp(*r);
    return;
  }

  assert(r.is(nx));
}

void Codegen::statement(StatementAST* ast) {
  accept(ast);
}

void Codegen::declaration(DeclarationAST* ast) {
  accept(ast);
}

void Codegen::place(IR::BasicBlock* block) {
  if (_block && ! _block->isTerminated())
    _block->emitJump(block);
  _function->placeBasicBlock(block);
  _block = block;
}

void Codegen::accept(AST* ast) {
  if (! ast)
    return;
  switch (ast->kind()) {
#define VISIT_AST(x) case ASTKind::k##x: visit(reinterpret_cast<x##AST*>(ast)); break;
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
  } // switch
}

const IR::Temp* Codegen::newTemp() {
  return _function->getTemp(_tempCount++);
}

// ASTs
void Codegen::visit(AttributeAST* ast) {
}

void Codegen::visit(BaseClassAST* ast) {
}

void Codegen::visit(DeclaratorAST* ast) {
}

void Codegen::visit(EnumeratorAST* ast) {
}

void Codegen::visit(LambdaCaptureAST* ast) {
}

void Codegen::visit(LambdaDeclaratorAST* ast) {
}

void Codegen::visit(MemInitializerAST* ast) {
}

void Codegen::visit(ParametersAndQualifiersAST* ast) {
}

void Codegen::visit(PtrOperatorAST* ast) {
}

void Codegen::visit(TranslationUnitAST* ast) {
}

// core declarators
void Codegen::visit(DeclaratorIdAST* ast) {
}

void Codegen::visit(NestedDeclaratorAST* ast) {
}

// declarations
void Codegen::visit(AccessDeclarationAST* ast) {
}

void Codegen::visit(AliasDeclarationAST* ast) {
}

void Codegen::visit(AsmDefinitionAST* ast) {
}

void Codegen::visit(FunctionDefinitionAST* ast) {
  auto function = _module->newFunction(ast->symbol);
  auto entryBlock = function->newBasicBlock();
  auto exitBlock = function->newBasicBlock();
  int tempCount = 0;
  std::map<const Name*, IR::BasicBlock*> labels;
  Loop loop{nullptr, nullptr};

  std::swap(_tempCount, tempCount);
  std::swap(_function, function);
  std::swap(_exitBlock, exitBlock);
  std::swap(_labels, labels);
  std::swap(_loop, loop);

  const IR::Expr* exitValue = newTemp();
  std::swap(_exitValue, exitValue);

  place(entryBlock);
  _block->emitMove(_exitValue, _function->getConst("0"));
  statement(ast->statement);

  _block->emitJump(_exitBlock);
  place(_exitBlock);
  _block->emitRet(_exitValue);

  _function->dump(std::cout);

  std::swap(_function, function);
  std::swap(_exitBlock, exitBlock);
  std::swap(_tempCount, tempCount);
  std::swap(_exitValue, exitValue);
  std::swap(_labels, labels);
  std::swap(_loop, loop);
}

void Codegen::visit(LinkageSpecificationAST* ast) {
}

void Codegen::visit(NamespaceAliasDefinitionAST* ast) {
}

void Codegen::visit(NamespaceDefinitionAST* ast) {
}

void Codegen::visit(OpaqueEnumDeclarationAST* ast) {
}

void Codegen::visit(ParameterDeclarationAST *ast) {
}

void Codegen::visit(SimpleDeclarationAST* ast) {
}

void Codegen::visit(StaticAssertDeclarationAST* ast) {
}

void Codegen::visit(TemplateDeclarationAST* ast) {
}

void Codegen::visit(TemplateTypeParameterAST* ast) {
}

void Codegen::visit(TypeParameterAST* ast) {
}

void Codegen::visit(UsingDeclarationAST* ast) {
}

void Codegen::visit(UsingDirectiveAST* ast) {
}

// expressions
void Codegen::visit(AlignofExpressionAST* ast) {
  _result = Result{_function->getConst("@align-expr@")};
}

void Codegen::visit(BinaryExpressionAST* ast) {
  if (ast->op == T_COMMA) {
    if (_result.accept(nx)) {
      statement(ast->left_expression);
      statement(ast->right_expression);
      return;
    }

    if (_result.accept(ex)) {
      statement(ast->left_expression);
      _result = expression(ast->right_expression);
      return;
    }

    if (_result.accept(cx)) {
      statement(ast->left_expression);
      condition(ast->right_expression, _result.iftrue, _result.iffalse);
      return;
    }

    assert(!"unreachable");
  }

  switch (ast->op) {
  case T_EQUAL:
  case T_PLUS_EQUAL:
  case T_MINUS_EQUAL:
  case T_STAR_EQUAL:
  case T_SLASH_EQUAL:
  case T_PERCENT_EQUAL:
  case T_AMP_EQUAL:
  case T_CARET_EQUAL:
  case T_BAR_EQUAL:
  case T_LESS_LESS_EQUAL:
  case T_GREATER_GREATER_EQUAL: {
    if (_result.accept(nx)) {
      auto target = expression(ast->left_expression);
      auto source = expression(ast->right_expression);
      _block->emitMove(*target, *source, ast->op);
      return;
    }
    // ### TODO
    break;
  }

  case T_AMP_AMP: {
    if (_result.accept(cx)) {
      auto iftrue = _function->newBasicBlock();
      condition(ast->left_expression, iftrue, _result.iffalse);
      place(iftrue);
      condition(ast->right_expression, _result.iftrue, _result.iffalse);
      return;
    }
    // ### TODO
    break;
  }

  case T_BAR_BAR: {
    if (_result.accept(cx)) {
      auto iffalse = _function->newBasicBlock();
      condition(ast->left_expression, _result.iftrue, iffalse);
      place(iffalse);
      condition(ast->right_expression, _result.iftrue, _result.iffalse);
      return;
    }
    // ### TODO
    break;
  }

  default:
    break;
  } // switch

  auto left = expression(ast->left_expression);
  auto right = expression(ast->right_expression);
  _result = Result{_function->getBinop(ast->op, *left, *right)};
}

void Codegen::visit(BracedInitializerAST* ast) {
  _result = Result{_function->getConst("@braced-init@")};
}

void Codegen::visit(BracedTypeCallExpressionAST* ast) {
  _result = Result{_function->getConst("@braced-type-call@")};
}

void Codegen::visit(CallExpressionAST* ast) {
  auto base = expression(ast->base_expression);
  std::vector<const IR::Expr*> args;
  for (auto it = ast->expression_list; it; it = it->next) {
    auto arg = expression(it->value);
    args.push_back(*arg);
  }
  auto call = _function->getCall(*base, std::move(args));
  if (_result.accept(nx)) {
    _block->emitExp(call);
    return;
  }
  _result = Result{call};
}

void Codegen::visit(CastExpressionAST* ast) {
  auto r = expression(ast->expression);
  _result = Result{_function->getCast(ast->targetTy, *r)};
}

void Codegen::visit(ConditionAST* ast) {
  _result = Result{_function->getConst("@condition@")};
}

void Codegen::visit(ConditionalExpressionAST* ast) {
  auto iftrue = _function->newBasicBlock();
  auto iffalse = _function->newBasicBlock();
  auto endif = _function->newBasicBlock();
  auto t = newTemp();
  condition(ast->expression, iftrue, iffalse);
  place(iftrue);
  auto ok = expression(ast->iftrue_expression);
  _block->emitMove(t, *ok);
  _block->emitJump(endif);
  place(iffalse);
  auto ko = expression(ast->iffalse_expression);
  _block->emitMove(t, *ko);
  _block->emitJump(endif);
  place(endif);
  _result = Result{t};
}

void Codegen::visit(CppCastExpressionAST* ast) {
  auto r = expression(ast->expression);
  _result = Result{_function->getCast(ast->targetTy, *r)}; // ### TODO: dynamic_cast, static_cast, ...
}

void Codegen::visit(DeleteExpressionAST* ast) {
  _result = Result{_function->getConst("@delete@")};
}

void Codegen::visit(IdExpressionAST* ast) {
  _result = Result{_function->getSym(ast->id)};
}

void Codegen::visit(IncrExpressionAST* ast) {
  _result = Result{_function->getConst("@incr/decr@")};
}

void Codegen::visit(LambdaExpressionAST* ast) {
  _result = Result{_function->getConst("@lambda@")};
}

void Codegen::visit(LiteralExpressionAST* ast) {
  auto value = unit->tokenText(ast->literal_token);
  _result = Result{_function->getConst(value)};
}

void Codegen::visit(MemberExpressionAST* ast) {
  auto op = unit->tokenKind(ast->access_token);
  auto base = expression(ast->base_expression);
  _result = Result{_function->getMember(op, *base, ast->id)};
}

void Codegen::visit(NestedExpressionAST* ast) {
  accept(ast->expression);
}

void Codegen::visit(NewExpressionAST* ast) {
  _result = Result{_function->getConst("@new@")};
}

void Codegen::visit(NoexceptExpressionAST* ast) {
  _result = Result{_function->getConst("@noexcept@")};
}

void Codegen::visit(PackedExpressionAST* ast) {
  _result = Result{_function->getConst("@packed-expr@")};
}

void Codegen::visit(SimpleInitializerAST* ast) {
  _result = Result{_function->getConst("@simple-init@")};
}

void Codegen::visit(SizeofExpressionAST* ast) {
  _result = Result{_function->getConst("@sizeof@")};
}

void Codegen::visit(SizeofPackedArgsExpressionAST* ast) {
  _result = Result{_function->getConst("@sizeof...@")};
}

void Codegen::visit(SizeofTypeExpressionAST* ast) {
  _result = Result{_function->getConst("@sizeof-type@")};
}

void Codegen::visit(SubscriptExpressionAST* ast) {
  auto base = expression(ast->base_expression);
  auto index = expression(ast->index_expression);
  _result = Result{_function->getSubscript(*base, *index)};
}

void Codegen::visit(TemplateArgumentAST* ast) {
  _result = Result{_function->getConst("@templ-arg@")};
}

void Codegen::visit(ThisExpressionAST* ast) {
  _result = Result{_function->getSym(unit->control()->getIdentifier("this"))};
}

void Codegen::visit(TypeCallExpressionAST* ast) {
  _result = Result{_function->getConst("@type-call@")};
}

void Codegen::visit(TypeIdAST* ast) {
  _result = Result{_function->getConst("@type-id@")};
}

void Codegen::visit(TypeidExpressionAST* ast) {
  _result = Result{_function->getConst("@typeid-expr@")};
}

void Codegen::visit(UnaryExpressionAST* ast) {
  auto expr = expression(ast->expression);
  _result = Result{_function->getUnop(ast->op, *expr)};
}

// names
void Codegen::visit(DecltypeAutoNameAST* ast) {
}

void Codegen::visit(DecltypeNameAST* ast) {
}

void Codegen::visit(DestructorNameAST* ast) {
}

void Codegen::visit(OperatorNameAST* ast) {
}

void Codegen::visit(ConversionFunctionIdAST* ast) {
}

void Codegen::visit(PackedNameAST* ast) {
}

void Codegen::visit(QualifiedNameAST* ast) {
}

void Codegen::visit(SimpleNameAST* ast) {
}

void Codegen::visit(TemplateIdAST* ast) {
}

// postfix declarations
void Codegen::visit(ArrayDeclaratorAST* ast) {
}

void Codegen::visit(FunctionDeclaratorAST* ast) {
}


// specifiers
void Codegen::visit(AlignasAttributeSpecifierAST* ast) {
}

void Codegen::visit(AlignasTypeAttributeSpecifierAST* ast) {
}

void Codegen::visit(AttributeSpecifierAST* ast) {
}

void Codegen::visit(ClassSpecifierAST* ast) {
}

void Codegen::visit(ElaboratedTypeSpecifierAST* ast) {
}

void Codegen::visit(EnumSpecifierAST* ast) {
}

void Codegen::visit(ExceptionSpecificationAST* ast) {
}

void Codegen::visit(NamedSpecifierAST* ast) {
}

void Codegen::visit(SimpleSpecifierAST* ast) {
}

void Codegen::visit(TypenameSpecifierAST* ast) {
}


// statements
void Codegen::visit(BreakStatementAST* ast) {
  if (auto breakLabel = _loop.breakLabel)
    _block->emitJump(breakLabel);
  else
    _block->emitExp(_function->getConst("@break-stmt"));
}

void Codegen::visit(CaseStatementAST* ast) {
  _block->emitExp(_function->getConst("@case-stmt"));
  statement(ast->statement);
}

void Codegen::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statement_list; it; it = it->next) {
    statement(it->value);
  }
}

void Codegen::visit(ContinueStatementAST* ast) {
  if (auto continueLabel = _loop.continueLabel)
    _block->emitJump(continueLabel);
  else
    _block->emitExp(_function->getConst("@continue-stmt"));
}

void Codegen::visit(DeclarationStatementAST* ast) {
  _block->emitExp(_function->getConst("@decl-stmt"));
}

void Codegen::visit(DefaultStatementAST* ast) {
  _block->emitExp(_function->getConst("@default-stmt"));
}

void Codegen::visit(DoStatementAST* ast) {
  auto toploop = _function->newBasicBlock();
  auto continueLoop = _function->newBasicBlock();
  auto endloop = _function->newBasicBlock();
  Loop loop{endloop, continueLoop};
  std::swap(_loop, loop);
  place(toploop);
  statement(ast->statement);
  place(continueLoop);
  condition(ast->expression, toploop, endloop);
  place(endloop);
  std::swap(_loop, loop);
}

void Codegen::visit(ExpressionStatementAST* ast) {
  statement(ast->expression);
}

void Codegen::visit(ForRangeStatementAST* ast) {
  _block->emitExp(_function->getConst("@for-range-stmt"));
}

void Codegen::visit(ForStatementAST* ast) {
  auto topfor = _function->newBasicBlock();
  auto bodyfor = _function->newBasicBlock();
  auto stepfor = _function->newBasicBlock();
  auto endfor = _function->newBasicBlock();
  Loop loop{endfor, stepfor};
  statement(ast->initializer);
  std::swap(_loop, loop);
  place(topfor);
  condition(ast->condition, bodyfor, endfor);
  place(bodyfor);
  statement(ast->statement);
  place(stepfor);
  statement(ast->expression);
  _block->emitJump(topfor);
  place(endfor);
  std::swap(_loop, loop);
}

void Codegen::visit(GotoStatementAST* ast) {
  auto& target = _labels[ast->id];
  if (! target)
    target = _function->newBasicBlock();
  _block->emitJump(target);
}

void Codegen::visit(IfStatementAST* ast) {
  if (! ast->else_statement) {
    auto iftrue = _function->newBasicBlock();
    auto endif = _function->newBasicBlock();
    condition(ast->condition, iftrue, endif);
    place(iftrue);
    statement(ast->statement);
    place(endif);
    return;
  }

  auto iftrue = _function->newBasicBlock();
  auto iffalse = _function->newBasicBlock();
  auto endif = _function->newBasicBlock();
  condition(ast->condition, iftrue, iffalse);
  place(iftrue);
  statement(ast->statement);
  _block->emitJump(endif);
  place(iffalse);
  statement(ast->else_statement);
  place(endif);
}

void Codegen::visit(LabeledStatementAST* ast) {
  auto& target = _labels[ast->id];
  if (! target)
    target = _function->newBasicBlock();
  place(target);
  statement(ast->statement);
}

void Codegen::visit(ReturnStatementAST* ast) {
  if (ast->expression) {
    auto r = expression(ast->expression);
    _block->emitMove(_exitValue, *r);
  }
  _block->emitJump(_exitBlock);
}

void Codegen::visit(SwitchStatementAST* ast) {
  _block->emitExp(_function->getConst("@switch-stmt"));
}

void Codegen::visit(TryBlockStatementAST* ast) {
  _block->emitExp(_function->getConst("@try-block-stmt"));
}

void Codegen::visit(WhileStatementAST* ast) {
  auto topwhile = _function->newBasicBlock();
  auto bodywhile = _function->newBasicBlock();
  auto endwhile = _function->newBasicBlock();
  Loop loop{endwhile, topwhile};
  std::swap(_loop, loop);
  place(topwhile);
  condition(ast->condition, bodywhile, endwhile);
  place(bodywhile);
  statement(ast->statement);
  _block->emitJump(topwhile);
  place(endwhile);
  std::swap(_loop, loop);
}
