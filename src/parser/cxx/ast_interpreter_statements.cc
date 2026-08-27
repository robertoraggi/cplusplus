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

  void bindRangeElementBindings(ForRangeStatementAST* ast);

  [[nodiscard]] auto forRangeOverList(ForRangeStatementAST* ast,
                                      VariableSymbol* var,
                                      const ConstValue& rangeVal)
      -> StatementResult;
  [[nodiscard]] auto forRangeOverPointerIterator(ForRangeStatementAST* ast,
                                                 VariableSymbol* var,
                                                 const ConstValue& rangeVal)
      -> StatementResult;
  [[nodiscard]] auto forRangeOverClassIterator(ForRangeStatementAST* ast,
                                               VariableSymbol* var,
                                               const ConstValue& rangeVal)
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
  auto expressionResult = interp.expression(ast->expression);

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
    auto conditionResult = interp.expression(ast->condition);

    if (conditionResult.has_value()) {
      auto boolVal = interp.toBool(*conditionResult);
      if (boolVal.has_value()) {
        if (*boolVal) return interp.statement(ast->statement);
        return interp.statement(ast->elseStatement);
      }
    }

    (void)interp.statement(ast->statement);
    (void)interp.statement(ast->elseStatement);
    return {};
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

auto ASTInterpreter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto switchResult = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);

    auto conditionResult = interp.expression(ast->condition);
    if (!conditionResult.has_value()) return {};
    auto condValue = interp.toInt(*conditionResult);
    if (!condValue.has_value()) return {};

    auto body = ast_cast<CompoundStatementAST>(ast->statement);
    if (!body) return {};

    std::vector<StatementAST*> stmts;
    for (auto node : ListView{body->statementList}) stmts.push_back(node);

    int start = -1;
    int defaultIdx = -1;
    for (std::size_t i = 0; i < stmts.size(); ++i) {
      if (auto caseStmt = ast_cast<CaseStatementAST>(stmts[i])) {
        if (caseStmt->caseValue == *condValue) {
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

    auto conditionResult = interp.expression(ast->condition);
    if (!conditionResult.has_value()) return {};
    auto boolVal = interp.toBool(*conditionResult);
    if (!boolVal.has_value()) return {};
    if (!*boolVal) break;

    auto result = interp.statement(ast->statement);
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

    auto conditionResult = interp.expression(ast->expression);
    if (!conditionResult.has_value()) return {};
    auto boolVal = interp.toBool(*conditionResult);
    if (!boolVal.has_value()) return {};
    if (!*boolVal) break;
  }

  return {};
}

static auto rangeForVariable(DeclarationAST* rangeDeclaration)
    -> VariableSymbol* {
  if (auto simpleDecl = ast_cast<SimpleDeclarationAST>(rangeDeclaration)) {
    auto initDecl = simpleDecl->initDeclaratorList
                        ? simpleDecl->initDeclaratorList->value
                        : nullptr;
    return initDecl ? symbol_cast<VariableSymbol>(initDecl->symbol) : nullptr;
  }

  if (auto structuredBinding =
          ast_cast<StructuredBindingDeclarationAST>(rangeDeclaration)) {
    if (auto hidden = structuredBinding->hiddenVariable)
      return symbol_cast<VariableSymbol>(hidden->symbol);
  }

  return nullptr;
}

void ASTInterpreter::StatementVisitor::bindRangeElementBindings(
    ForRangeStatementAST* ast) {
  auto structuredBinding =
      ast_cast<StructuredBindingDeclarationAST>(ast->rangeDeclaration);
  if (!structuredBinding) return;
  for (auto initDecl : ListView{structuredBinding->bindingDeclaratorList})
    interp.interpretInitDeclarator(initDecl);
}

auto ASTInterpreter::StatementVisitor::forRangeOverList(
    ForRangeStatementAST* ast, VariableSymbol* var, const ConstValue& rangeVal)
    -> StatementResult {
  auto listPtr = std::get_if<std::shared_ptr<InitializerList>>(&rangeVal);
  if (!listPtr || !*listPtr) return {};
  auto list = *listPtr;

  const bool bindByRef = interp.traits.is_reference(var->type());

  for (std::size_t i = 0; i < list->elements.size(); ++i) {
    if (!interp.tick()) return {};

    auto& element = std::get<0>(list->elements[i]);
    if (bindByRef) {
      interp.bindReference(var, &element);
    } else {
      interp.setLocal(var, element);
    }
    bindRangeElementBindings(ast);

    auto result = interp.statement(ast->statement);
    if (interp.aborted()) return {};
    if (result.flow == ControlFlow::kBreak) break;
    if (result.flow == ControlFlow::kReturn) return result;
  }
  return {};
}

auto ASTInterpreter::StatementVisitor::forRangeOverPointerIterator(
    ForRangeStatementAST* ast, VariableSymbol* var, const ConstValue& rangeVal)
    -> StatementResult {
  auto callOnRange = [&](FunctionSymbol* f) -> std::optional<ConstValue> {
    if (ast->usesMemberBeginEnd) {
      auto objPtr = std::get_if<std::shared_ptr<ConstObject>>(&rangeVal);
      if (!objPtr) return std::nullopt;
      auto savedThis = interp.thisObject();
      interp.setThisObject(*objPtr);
      auto result = interp.evaluateCall(f, {});
      interp.setThisObject(savedThis);
      return result;
    }
    return interp.evaluateCall(f, {rangeVal});
  };

  auto beginVal = callOnRange(ast->beginFunction);
  auto endVal = callOnRange(ast->endFunction);
  if (!beginVal.has_value() || !endVal.has_value()) return {};

  auto beginAddr = std::get_if<std::shared_ptr<ConstAddress>>(&*beginVal);
  auto endAddr = std::get_if<std::shared_ptr<ConstAddress>>(&*endVal);
  if (!beginAddr || !*beginAddr || !endAddr || !*endAddr) return {};
  if ((*beginAddr)->symbol() != (*endAddr)->symbol()) return {};

  const bool bindByRef = interp.traits.is_reference(var->type());
  const auto base = (*beginAddr)->offset();

  for (auto off = base; off < (*endAddr)->offset(); ++off) {
    if (!interp.tick()) return {};

    if (bindByRef) {
      auto slot = interp.addressSlot(**beginAddr, off - base);
      if (!slot) return {};
      interp.bindReference(var, slot);
    } else {
      auto elemVal = interp.loadAddress(**beginAddr, off - base);
      if (!elemVal.has_value()) return {};
      interp.setLocal(var, *elemVal);
    }
    bindRangeElementBindings(ast);

    auto result = interp.statement(ast->statement);
    if (interp.aborted()) return {};
    if (result.flow == ControlFlow::kBreak) break;
    if (result.flow == ControlFlow::kReturn) return result;
  }
  return {};
}

auto ASTInterpreter::StatementVisitor::forRangeOverClassIterator(
    ForRangeStatementAST* ast, VariableSymbol* var, const ConstValue& rangeVal)
    -> StatementResult {
  auto callOnRange = [&](FunctionSymbol* f) -> std::optional<ConstValue> {
    if (ast->usesMemberBeginEnd) {
      auto objPtr = std::get_if<std::shared_ptr<ConstObject>>(&rangeVal);
      if (!objPtr) return std::nullopt;
      auto savedThis = interp.thisObject();
      interp.setThisObject(*objPtr);
      auto result = interp.evaluateCall(f, {});
      interp.setThisObject(savedThis);
      return result;
    }
    return interp.evaluateCall(f, {rangeVal});
  };

  auto beginVal = callOnRange(ast->beginFunction);
  auto endVal = callOnRange(ast->endFunction);
  if (!beginVal.has_value() || !endVal.has_value()) return {};

  auto beginObj = std::get_if<std::shared_ptr<ConstObject>>(&*beginVal);
  auto endObj = std::get_if<std::shared_ptr<ConstObject>>(&*endVal);
  if (!beginObj || !*beginObj || !endObj || !*endObj) return {};

  auto callUnary =
      [&](FunctionSymbol* f,
          const std::shared_ptr<ConstObject>& it) -> std::optional<ConstValue> {
    if (!f) return std::nullopt;
    if (f->isImplicitObjectMemberFunction()) {
      auto savedThis = interp.thisObject();
      interp.setThisObject(it);
      auto result = interp.evaluateCall(f, {});
      interp.setThisObject(savedThis);
      return result;
    }
    return interp.evaluateCall(f, {ConstValue{it}});
  };
  auto callBinary =
      [&](FunctionSymbol* f, const std::shared_ptr<ConstObject>& a,
          const std::shared_ptr<ConstObject>& b) -> std::optional<ConstValue> {
    if (!f) return std::nullopt;
    if (f->isImplicitObjectMemberFunction()) {
      auto savedThis = interp.thisObject();
      interp.setThisObject(a);
      auto result = interp.evaluateCall(f, {ConstValue{b}});
      interp.setThisObject(savedThis);
      return result;
    }
    return interp.evaluateCall(f, {ConstValue{a}, ConstValue{b}});
  };

  const bool bindByRef = interp.traits.is_reference(var->type());

  for (;;) {
    if (!interp.tick()) return {};

    auto neq = ast->notEqualReversed
                   ? callBinary(ast->notEqualFunction, *endObj, *beginObj)
                   : callBinary(ast->notEqualFunction, *beginObj, *endObj);
    if (!neq.has_value()) return {};
    auto keepGoing = interp.toBool(*neq);
    if (!keepGoing.has_value()) break;
    if (ast->notEqualRewritten) keepGoing = !*keepGoing;
    if (!*keepGoing) break;

    if (bindByRef) {
      ConstValue* slot = nullptr;
      if (ast->derefFunction->isImplicitObjectMemberFunction()) {
        auto savedThis = interp.thisObject();
        interp.setThisObject(*beginObj);
        slot = interp.evaluateCallLValue(ast->derefFunction, {});
        interp.setThisObject(savedThis);
      } else {
        slot = interp.evaluateCallLValue(ast->derefFunction,
                                         {ConstValue{*beginObj}});
      }
      if (!slot) return {};
      interp.bindReference(var, slot);
    } else {
      auto elemVal = callUnary(ast->derefFunction, *beginObj);
      if (!elemVal.has_value()) return {};
      interp.setLocal(var, *elemVal);
    }
    bindRangeElementBindings(ast);

    auto result = interp.statement(ast->statement);
    if (interp.aborted()) return {};
    if (result.flow == ControlFlow::kBreak) break;
    if (result.flow != ControlFlow::kReturn) {
      if (!callUnary(ast->incrementFunction, *beginObj).has_value()) return {};
      continue;
    }
    return result;
  }
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto result = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);
    (void)interp.declaration(ast->rangeDeclaration);

    auto var = rangeForVariable(ast->rangeDeclaration);

    if (var) {
      auto rangeVal = interp.expression(ast->rangeInitializer);

      if (rangeVal.has_value()) {
        if (std::get_if<std::shared_ptr<InitializerList>>(&*rangeVal)) {
          return forRangeOverList(ast, var, *rangeVal);
        }
        if (ast->beginFunction && ast->isPointerIterator) {
          return forRangeOverPointerIterator(ast, var, *rangeVal);
        }
        if (ast->beginFunction && ast->derefFunction &&
            ast->incrementFunction && ast->notEqualFunction) {
          return forRangeOverClassIterator(ast, var, *rangeVal);
        }
      }
    } else {
      (void)interp.expression(ast->rangeInitializer);
    }

    (void)interp.statement(ast->statement);
    return {};
  }();

  if (!interp.endAutomaticScope(mark)) return {};
  return result;
}

auto ASTInterpreter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementResult {
  auto mark = interp.beginAutomaticScope();
  auto loopResult = [&]() -> StatementResult {
    (void)interp.statement(ast->initializer);

    for (;;) {
      if (!interp.tick()) return {};

      if (ast->condition) {
        auto conditionResult = interp.expression(ast->condition);
        if (!conditionResult.has_value()) return {};
        auto boolVal = interp.toBool(*conditionResult);
        if (!boolVal.has_value()) return {};
        if (!*boolVal) break;
      }

      auto result = interp.statement(ast->statement);
      if (interp.aborted()) return {};
      if (result.flow == ControlFlow::kBreak) break;
      if (result.flow == ControlFlow::kReturn) return result;

      if (ast->expression) {
        auto expressionResult = interp.expression(ast->expression);
        if (!expressionResult.has_value()) return {};
      }
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
  if (interp.captureReturnAddress_) {
    interp.returnAddress_ = interp.addressOfLvalue(ast->expression);
    return {ControlFlow::kReturn};
  }

  if (interp.captureReturnLValue_) {
    interp.returnLValue_ = interp.lvalue(ast->expression);
    return {ControlFlow::kReturn};
  }

  auto expressionResult = interp.expression(ast->expression);
  if (expressionResult.has_value()) {
    interp.setReturnValue(*expressionResult);
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

void ASTInterpreter::interpretInitDeclarator(InitDeclaratorAST* initDecl) {
  if (!initDecl || !initDecl->symbol) return;

  auto var = symbol_cast<VariableSymbol>(initDecl->symbol);

  if (var && traits.is_reference(var->type())) {
    auto initExpr = initDecl->initializer;
    if (auto eq = ast_cast<EqualInitializerAST>(initExpr))
      initExpr = eq->expression;
    if (auto slot = lvalue(initExpr)) {
      bindReference(initDecl->symbol, slot);
    }
    return;
  }

  auto initVal = expression(initDecl->initializer);

  if (!initVal.has_value()) {
    if (auto parenInit = ast_cast<ParenInitializerAST>(initDecl->initializer)) {
      if (var) {
        auto varType = traits.remove_cv(var->type());
        if (auto classType = type_cast<ClassType>(varType)) {
          std::vector<ConstValue> args;
          bool argsOk = true;
          for (auto node : ListView{parenInit->expressionList}) {
            auto val = evaluate(node);
            if (!val) {
              argsOk = false;
              break;
            }
            args.push_back(std::move(*val));
          }
          if (argsOk) {
            if (var->constructor() && var->constructor()->isConstexpr()) {
              initVal = evaluateConstructor(var->constructor(), varType,
                                            std::move(args));
            } else if (auto classSym = classType->symbol()) {
              for (auto ctor : classSym->constructors()) {
                if (ctor->isConstexpr()) {
                  initVal = evaluateConstructor(ctor, varType, std::move(args));
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  if (!initVal.has_value() && !initDecl->initializer) {
    if (var) initVal = defaultConstruct(var->type());
  }

  if (initVal.has_value()) {
    if (var && !traits.is_reference(var->type()) &&
        traits.is_class(traits.remove_cv(var->type())))
      initVal = cloneValue(*initVal);
    setLocal(initDecl->symbol, *initVal);
    if (var) registerAutomaticObject(var);
  }
}

void ASTInterpreter::interpretStructuredBinding(
    StructuredBindingDeclarationAST* ast) {
  interpretInitDeclarator(ast->hiddenVariable);
  for (auto initDecl : ListView{ast->bindingDeclaratorList})
    interpretInitDeclarator(initDecl);
}

auto ASTInterpreter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementResult {
  auto declarationResult = interp.declaration(ast->declaration);

  if (auto simpleDecl = ast_cast<SimpleDeclarationAST>(ast->declaration)) {
    for (auto initDecl : ListView{simpleDecl->initDeclaratorList})
      interp.interpretInitDeclarator(initDecl);
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
