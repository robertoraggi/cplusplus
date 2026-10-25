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
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
struct ASTInterpreter::StatementVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(SwitchStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(WhileStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DoStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ForRangeStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ForStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(BreakStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ContinueStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ReturnStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CoroutineReturnStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(GotoStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DeclarationStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto evaluateCondition(ExpressionAST* condition)
      -> std::optional<bool>;

  [[nodiscard]] auto evaluateSwitchCondition(ExpressionAST* condition)
      -> std::optional<std::intmax_t>;

  [[nodiscard]] auto loopIteration(ExpressionAST* condition,
                                   StatementAST* statement,
                                   ExpressionAST* expression)
      -> StatementResult;

  [[nodiscard]] auto runLoopIteration(ExpressionAST* condition,
                                      StatementAST* statement,
                                      ExpressionAST* expression)
      -> StatementResult;
};

struct ASTInterpreter::ExceptionDeclarationVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
};

auto ASTInterpreter::statement(StatementAST* ast) -> StatementResult {
  if (!ast) return {};
  if (!tick()) return {};
  return visit(StatementVisitor{*this}, ast);
}

auto ASTInterpreter::handler(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult =
      exceptionDeclaration(ast->exceptionDeclaration);
  auto statementResult = statement(ast->statement);

  return {};
}

auto ASTInterpreter::exceptionDeclaration(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementResult {
  return interp.statement(ast->statement);
}

auto ASTInterpreter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementResult {
  FullExpressionScope fullExpression{interp};
  if (ast->expression && !interp.discardedValue(ast->expression)) {
    interp.aborted_ = true;
  }
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto result = StatementResult{};
  for (auto node : ListView{ast->statementList}) {
    result = interp.statement(node);
    if (interp.aborted()) break;
    if (result.flow != ControlFlow::kNormal) break;
  }

  if (!interp.endAutomaticScope(mark)) return {};
  return result;
}

auto ASTInterpreter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto result = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);
    auto condition = evaluateCondition(ast->condition);
    if (!condition.has_value()) return {};
    if (*condition) return interp.statement(ast->statement);
    return interp.statement(ast->elseStatement);
  }();

  if (!interp.endAutomaticScope(mark)) return {};
  return result;
}

auto ASTInterpreter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementResult {
  if (!ast->isNot) {
    if (ast->statement) return interp.statement(ast->statement);
  } else {
    if (ast->elseStatement) return interp.statement(ast->elseStatement);
  }
  return {};
}

namespace {
void appendSwitchLabels(std::vector<StatementAST*>& statements,
                        StatementAST* statement) {
  while (statement) {
    statements.push_back(statement);
    if (auto label = ast_cast<CaseStatementAST>(statement)) {
      statement = label->statement;
      continue;
    }
    if (auto label = ast_cast<DefaultStatementAST>(statement)) {
      statement = label->statement;
      continue;
    }
    return;
  }
}
}  // namespace

auto ASTInterpreter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto switchResult = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);

    auto condValue = evaluateSwitchCondition(ast->condition);
    if (!condValue.has_value()) return {};

    std::vector<StatementAST*> stmts;
    if (auto body = ast_cast<CompoundStatementAST>(ast->statement)) {
      for (auto node : ListView{body->statementList})
        appendSwitchLabels(stmts, node);
    } else {
      appendSwitchLabels(stmts, ast->statement);
    }

    int start = -1;
    int defaultIdx = -1;
    for (std::size_t i = 0; i < stmts.size(); ++i) {
      if (auto caseStmt = ast_cast<CaseStatementAST>(stmts[i])) {
        auto matches = caseStmt->caseValue == *condValue;
        if (caseStmt->upperExpression) {
          auto upperValue = interp.evaluate(caseStmt->upperExpression);
          if (!upperValue) return {};
          auto lower = interp.toArithmeticType(
              ConstValue{ConstInt{caseStmt->caseValue}}, ast->condition->type);
          auto upper =
              interp.toArithmeticType(*upperValue, ast->condition->type);
          auto condition = interp.toArithmeticType(
              ConstValue{ConstInt{*condValue}}, ast->condition->type);
          if (!lower || !upper || !condition) return {};
          auto lowerInt = std::get_if<ConstInt>(&*lower);
          auto upperInt = std::get_if<ConstInt>(&*upper);
          auto conditionInt = std::get_if<ConstInt>(&*condition);
          if (!lowerInt || !upperInt || !conditionInt) return {};
          matches = *conditionInt >= *lowerInt && *conditionInt <= *upperInt;
        }
        if (matches) {
          start = static_cast<int>(i);
          break;
        }
      } else if (ast_cast<DefaultStatementAST>(stmts[i])) {
        defaultIdx = static_cast<int>(i);
      }
    }
    if (start < 0) start = defaultIdx;
    if (start < 0) return {};

    for (std::size_t i = start; i < stmts.size(); ++i) {
      auto result = interp.statement(stmts[i]);
      if (interp.aborted()) return {};
      if (result.flow == ControlFlow::kBreak) break;
      if (result.flow == ControlFlow::kContinue ||
          result.flow == ControlFlow::kReturn) {
        return result;
      }
    }
    return {};
  }();

  if (!interp.endAutomaticScope(mark)) return {};
  return switchResult;
}

auto ASTInterpreter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementResult {
  for (;;) {
    if (!interp.tick()) return {};

    auto result = loopIteration(ast->condition, ast->statement, nullptr);
    if (interp.aborted()) return {};
    if (result.flow == ControlFlow::kBreak) break;
    if (result.flow == ControlFlow::kReturn) return result;
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementResult {
  for (;;) {
    if (!interp.tick()) return {};

    auto result = interp.statement(ast->statement);
    if (interp.aborted()) return {};
    if (result.flow == ControlFlow::kBreak) break;
    if (result.flow == ControlFlow::kReturn) return result;

    auto condition = evaluateCondition(ast->expression);
    if (!condition.has_value()) return {};
    if (!*condition) break;
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::evaluateCondition(
    ExpressionAST* condition) -> std::optional<bool> {
  FullExpressionScope fullExpression{interp};
  auto value = interp.expression(condition);
  if (value.has_value()) {
    if (auto truth = interp.toBool(*value)) return truth;
  }
  interp.aborted_ = true;
  return std::nullopt;
}

auto ASTInterpreter::StatementVisitor::evaluateSwitchCondition(
    ExpressionAST* condition) -> std::optional<std::intmax_t> {
  auto value = interp.expression(condition);
  if (value.has_value()) {
    if (auto integer = interp.toInt(*value)) return integer;
  }
  interp.aborted_ = true;
  return std::nullopt;
}

auto ASTInterpreter::StatementVisitor::loopIteration(ExpressionAST* condition,
                                                     StatementAST* statement,
                                                     ExpressionAST* expression)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto result = runLoopIteration(condition, statement, expression);
  if (!interp.endAutomaticScope(mark)) return {};
  return result;
}

auto ASTInterpreter::StatementVisitor::runLoopIteration(
    ExpressionAST* condition, StatementAST* statement,
    ExpressionAST* expression) -> StatementResult {
  if (condition) {
    auto truth = evaluateCondition(condition);
    if (!truth.has_value()) return {};
    if (!*truth) return {ControlFlow::kBreak};
  }

  auto result = interp.statement(statement);
  if (interp.aborted()) return {};
  if (result.flow == ControlFlow::kBreak) return result;
  if (result.flow == ControlFlow::kReturn) return result;

  if (expression && !interp.discardedValue(expression)) {
    interp.aborted_ = true;
    return {};
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementResult {
  return interp.statement(ast->range.statement);
}

auto ASTInterpreter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto loopResult = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);

    for (;;) {
      if (!interp.tick()) return {};

      auto result =
          loopIteration(ast->condition, ast->statement, ast->expression);
      if (interp.aborted()) return {};
      if (result.flow == ControlFlow::kBreak) break;
      if (result.flow == ControlFlow::kReturn) return result;
    }
    return {};
  }();

  if (!interp.endAutomaticScope(mark)) return {};
  return loopResult;
}

auto ASTInterpreter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementResult {
  return {ControlFlow::kBreak};
}

auto ASTInterpreter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementResult {
  return {ControlFlow::kContinue};
}

auto ASTInterpreter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementResult {
  FullExpressionScope fullExpression{interp};
  if (interp.captureReturnAddress_) {
    interp.returnAddress_ = interp.addressOfLvalue(ast->expression);
    if (interp.returnAddress_ || !interp.returnAddressFallsBackToValue_)
      return {ControlFlow::kReturn};
  }

  auto returnType =
      interp.currentFunction_
          ? type_cast<FunctionType>(interp.currentFunction_->type())
          : nullptr;
  auto expressionResult =
      returnType && interp.traits.is_class(returnType->returnType())
          ? interp.initialValue(returnType->returnType(), ast->expression,
                                interp.resultAddress_)
          : interp.expression(ast->expression);
  if (expressionResult.has_value()) {
    interp.setReturnValue(*expressionResult);
  } else if (ast->expression) {
    interp.aborted_ = true;
  }

  return {ControlFlow::kReturn};
}

auto ASTInterpreter::StatementVisitor::operator()(
    CoroutineReturnStatementAST* ast) -> StatementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::initializeAutomaticVariable(Symbol* symbol,
                                                 ExpressionAST* initializer)
    -> bool {
  FullExpressionScope fullExpression{*this};
  if (!symbol) return false;

  auto var = symbol_cast<VariableSymbol>(symbol);
  if (var && var->isStructuredBindingAlias()) return true;

  if (var && traits.is_reference(var->type())) {
    if (frames_.empty()) frames_.push_back({});
    if (!bindReferenceTo(frames_.back(), var,
                         Initializer{initializer}.clause()))
      return false;
    registerAutomaticObject(var);
    return true;
  }

  ExpressionResult initVal;
  if (var) {
    setLocal(symbol, IndeterminateValue{});
    auto address = std::make_shared<ConstAddress>(symbol);
    auto storage = lookupLocalStorage(symbol);
    address->setStorage(storage);
    initVal = initializationValue(var->type(), var->constructor(), initializer,
                                  std::move(address));
    if (!initVal) return false;
    storage->setValue(std::move(*initVal));
    registerAutomaticObject(var);
    return true;
  }

  initVal = expression(initializer);
  if (!initVal.has_value()) return false;

  setLocal(symbol, *initVal);
  return true;
}

auto ASTInterpreter::initializeDecisionVariable(
    ConditionExpressionAST* condition) -> VariableSymbol* {
  if (!initializeAutomaticVariable(condition->symbol, condition->initializer))
    return nullptr;
  return condition->symbol;
}

void ASTInterpreter::interpretStructuredBinding(
    StructuredBindingDeclarationAST* ast) {
  if (auto hidden = ast->binding.hiddenVariable) {
    auto variable = symbol_cast<VariableSymbol>(hidden->symbol);
    if (variable)
      initializeAutomaticVariable(variable, variable->initializer());
  }
  for (auto node : ListView{ast->binding.bindingDeclaratorList}) {
    auto variable = symbol_cast<VariableSymbol>(node->symbol);
    if (variable)
      initializeAutomaticVariable(variable, variable->initializer());
  }
}

auto ASTInterpreter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementResult {
  auto declarationResult = interp.declaration(ast->declaration);

  if (auto simpleDecl = ast_cast<SimpleDeclarationAST>(ast->declaration)) {
    for (auto initDecl : ListView{simpleDecl->initDeclaratorList}) {
      auto variable = symbol_cast<VariableSymbol>(initDecl->symbol);
      if (!variable) continue;
      if (!interp.initializeAutomaticVariable(variable,
                                              variable->initializer())) {
        interp.aborted_ = true;
        break;
      }
    }
  } else if (auto structuredBinding =
                 ast_cast<StructuredBindingDeclarationAST>(ast->declaration)) {
    interp.interpretStructuredBinding(structuredBinding);
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementResult {
  auto statementResult = interp.statement(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = interp.handler(node);
  }

  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);

  return {};
}
}  // namespace cxx
