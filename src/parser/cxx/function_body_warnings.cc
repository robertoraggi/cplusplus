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
#include <cxx/ast_visitor.h>
#include <cxx/attributes.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/function_body_warnings.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

[[nodiscard]] auto stripParentheses(ExpressionAST* expr) -> ExpressionAST* {
  while (auto nested = ast_cast<NestedExpressionAST>(expr))
    expr = nested->expression;
  return expr;
}

[[nodiscard]] auto foldCondition(TranslationUnit* unit, ExpressionAST* expr)
    -> std::optional<bool> {
  if (!expr) return std::nullopt;
  if (isDependent(unit, expr)) return std::nullopt;

  ASTInterpreter interp{unit};
  auto value = interp.evaluate(expr);
  if (!value.has_value()) return std::nullopt;
  return interp.toBool(*value);
}

class ReachabilityAnalysis {
 public:
  explicit ReachabilityAnalysis(TranslationUnit* unit) : unit_(unit) {}

  [[nodiscard]] auto completesNormally(StatementAST* stmt) -> bool;
  [[nodiscard]] auto completesNormally(CompoundStatementAST* stmt) -> bool;

 private:
  [[nodiscard]] auto isDivergentExpression(ExpressionAST* expr) -> bool;
  [[nodiscard]] auto foldsToTrue(ExpressionAST* expr) -> bool;
  [[nodiscard]] auto conditionValue(ExpressionAST* expr) -> std::optional<bool>;

  TranslationUnit* unit_;
};

[[nodiscard]] auto isLabel(StatementAST* stmt) -> bool {
  return ast_cast<LabeledStatementAST>(stmt) ||
         ast_cast<CaseStatementAST>(stmt) ||
         ast_cast<DefaultStatementAST>(stmt);
}

class JumpFinder final : ASTVisitor {
 public:
  [[nodiscard]] static auto hasBreak(TranslationUnit* unit, StatementAST* stmt)
      -> bool {
    JumpFinder finder{unit, /*lookingForBreak=*/true};
    finder.accept(stmt);
    return finder.found_;
  }

  [[nodiscard]] static auto hasContinue(TranslationUnit* unit,
                                        StatementAST* stmt) -> bool {
    JumpFinder finder{unit, /*lookingForBreak=*/false};
    finder.accept(stmt);
    return finder.found_;
  }

 private:
  JumpFinder(TranslationUnit* unit, bool lookingForBreak)
      : unit_(unit), lookingForBreak_(lookingForBreak) {}

  auto preVisit(AST*) -> bool override { return !found_; }

  void visit(IfStatementAST* ast) override {
    if (auto taken = foldCondition(unit_, ast->condition)) {
      accept(*taken ? ast->statement : ast->elseStatement);
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(BreakStatementAST*) override {
    if (lookingForBreak_) found_ = true;
  }

  void visit(ContinueStatementAST*) override {
    if (!lookingForBreak_) found_ = true;
  }

  void visit(WhileStatementAST*) override {}
  void visit(DoStatementAST*) override {}
  void visit(ForStatementAST*) override {}
  void visit(ForRangeStatementAST*) override {}
  void visit(LambdaExpressionAST*) override {}

  void visit(SwitchStatementAST* ast) override {
    if (lookingForBreak_) return;
    ASTVisitor::visit(ast);
  }

  TranslationUnit* unit_;
  bool lookingForBreak_;
  bool found_ = false;
};

class DefaultLabelFinder final : ASTVisitor {
 public:
  [[nodiscard]] static auto hasDefaultLabel(StatementAST* stmt) -> bool {
    DefaultLabelFinder finder;
    finder.accept(stmt);
    return finder.found_;
  }

 private:
  auto preVisit(AST*) -> bool override { return !found_; }

  void visit(DefaultStatementAST*) override { found_ = true; }

  void visit(SwitchStatementAST*) override {}
  void visit(LambdaExpressionAST*) override {}

  bool found_ = false;
};

auto ReachabilityAnalysis::conditionValue(ExpressionAST* expr)
    -> std::optional<bool> {
  return foldCondition(unit_, expr);
}

auto ReachabilityAnalysis::foldsToTrue(ExpressionAST* expr) -> bool {
  return conditionValue(expr).value_or(false);
}

auto ReachabilityAnalysis::isDivergentExpression(ExpressionAST* expr) -> bool {
  expr = stripParentheses(expr);
  if (!expr) return false;

  if (ast_cast<ThrowExpressionAST>(expr)) return true;

  auto [callee, function] = calledFunction(expr);
  return function && function->isNoReturn();
}

auto ReachabilityAnalysis::completesNormally(CompoundStatementAST* ast)
    -> bool {
  auto reachable = true;

  for (auto statement : ListView{ast->statementList}) {
    if (isLabel(statement)) {
      reachable = true;
      if (auto labeled = ast_cast<LabeledStatementAST>(statement))
        reachable = completesNormally(labeled->statement);
      continue;
    }

    if (!reachable) continue;

    reachable = completesNormally(statement);
  }

  return reachable;
}

auto ReachabilityAnalysis::completesNormally(StatementAST* stmt) -> bool {
  if (!stmt) return true;

  switch (stmt->kind()) {
    case ASTKind::ReturnStatement:
    case ASTKind::CoroutineReturnStatement:
    case ASTKind::GotoStatement:
    case ASTKind::BreakStatement:
    case ASTKind::ContinueStatement:
      return false;

    case ASTKind::ExpressionStatement:
      return !isDivergentExpression(
          static_cast<ExpressionStatementAST*>(stmt)->expression);

    case ASTKind::CompoundStatement:
      return completesNormally(static_cast<CompoundStatementAST*>(stmt));

    case ASTKind::IfStatement: {
      auto ast = static_cast<IfStatementAST*>(stmt);
      if (auto value = conditionValue(ast->condition)) {
        if (*value) return completesNormally(ast->statement);
        return completesNormally(ast->elseStatement);
      }
      if (!ast->elseStatement) return true;
      return completesNormally(ast->statement) ||
             completesNormally(ast->elseStatement);
    }

    case ASTKind::ConstevalIfStatement: {
      auto ast = static_cast<ConstevalIfStatementAST*>(stmt);
      if (!ast->elseStatement) return true;
      return completesNormally(ast->statement) ||
             completesNormally(ast->elseStatement);
    }

    case ASTKind::WhileStatement: {
      auto ast = static_cast<WhileStatementAST*>(stmt);
      if (!foldsToTrue(ast->condition)) return true;
      return JumpFinder::hasBreak(unit_, ast->statement);
    }

    case ASTKind::DoStatement: {
      auto ast = static_cast<DoStatementAST*>(stmt);
      if (JumpFinder::hasBreak(unit_, ast->statement)) return true;
      if (foldsToTrue(ast->expression)) return false;
      return completesNormally(ast->statement) ||
             JumpFinder::hasContinue(unit_, ast->statement);
    }

    case ASTKind::ForStatement: {
      auto ast = static_cast<ForStatementAST*>(stmt);
      if (ast->condition && !foldsToTrue(ast->condition)) return true;
      return JumpFinder::hasBreak(unit_, ast->statement);
    }

    case ASTKind::SwitchStatement: {
      auto ast = static_cast<SwitchStatementAST*>(stmt);
      if (!DefaultLabelFinder::hasDefaultLabel(ast->statement)) return true;
      if (JumpFinder::hasBreak(unit_, ast->statement)) return true;
      return completesNormally(ast->statement);
    }

    case ASTKind::TryBlockStatement: {
      auto ast = static_cast<TryBlockStatementAST*>(stmt);
      if (completesNormally(ast->statement)) return true;
      for (auto handler : ListView{ast->handlerList}) {
        if (completesNormally(handler->statement)) return true;
      }
      return false;
    }

    default:
      return true;
  }
}

class CoroutineFinder final : ASTVisitor {
 public:
  [[nodiscard]] static auto isCoroutine(AST* body) -> bool {
    CoroutineFinder finder;
    finder.accept(body);
    return finder.found_;
  }

 private:
  auto preVisit(AST*) -> bool override { return !found_; }

  void visit(CoroutineReturnStatementAST*) override { found_ = true; }
  void visit(AwaitExpressionAST*) override { found_ = true; }
  void visit(YieldExpressionAST*) override { found_ = true; }

  void visit(LambdaExpressionAST*) override {}

  bool found_ = false;
};

class DiscardedValueChecker final : ASTVisitor {
 public:
  explicit DiscardedValueChecker(TranslationUnit* unit) : unit_(unit) {}

  void check(AST* body) { accept(body); }

 private:
  void visit(ExpressionStatementAST* ast) override {
    checkDiscarded(ast->expression);
    ASTVisitor::visit(ast);
  }

  void visit(BinaryExpressionAST* ast) override {
    if (ast->op == TokenKind::T_COMMA && !ast->symbol)
      checkDiscarded(ast->leftExpression);
    ASTVisitor::visit(ast);
  }

  void visit(ForStatementAST* ast) override {
    checkDiscarded(ast->expression);
    ASTVisitor::visit(ast);
  }

  void visit(LambdaExpressionAST*) override {}

  [[nodiscard]] auto nodiscardTypeSymbol(const Type* type) -> Symbol* {
    auto traits = unit_->typeTraits();
    auto unqualified = traits.remove_cv(type);

    if (auto classType = type_cast<ClassType>(unqualified))
      return classType->symbol();
    if (auto enumType = type_cast<EnumType>(unqualified))
      return enumType->symbol();
    if (auto scopedEnumType = type_cast<ScopedEnumType>(unqualified))
      return scopedEnumType->symbol();

    return nullptr;
  }

  [[nodiscard]] static auto explain(Symbol* symbol) -> std::string {
    auto reason = attributeArgument(symbol->attributes(), "nodiscard");
    if (!reason)
      reason = attributeArgument(symbol->attributes(), "warn_unused_result");
    if (!reason) return {};
    return std::format(": {}", reason->name());
  }

  void checkDiscarded(ExpressionAST* expr) {
    expr = stripParentheses(expr);
    if (!expr || !expr->type) return;
    if (isDependent(unit_, expr->type)) return;

    if (auto comma = ast_cast<BinaryExpressionAST>(expr);
        comma && comma->op == TokenKind::T_COMMA && !comma->symbol) {
      checkDiscarded(comma->rightExpression);
      return;
    }

    if (auto [callee, function] = calledFunction(expr);
        function && function->isNodiscard()) {
      unit_->warning(
          expr->firstSourceLocation(),
          std::format("ignoring return value of function declared with "
                      "'nodiscard' attribute{}",
                      explain(function)));
      return;
    }

    if (auto constructor = temporaryConstructor(expr);
        constructor && constructor->isNodiscard()) {
      unit_->warning(
          expr->firstSourceLocation(),
          std::format("ignoring temporary created by a constructor declared "
                      "with 'nodiscard' attribute{}",
                      explain(constructor)));
      return;
    }

    if (expr->valueCategory != ValueCategory::kPrValue) return;

    auto symbol = nodiscardTypeSymbol(expr->type);
    if (!symbol || !symbol->isNodiscard()) return;

    unit_->warning(
        expr->firstSourceLocation(),
        std::format("ignoring return value of type '{}' declared with "
                    "'nodiscard' attribute{}",
                    to_string(expr->type), explain(symbol)));
  }

  [[nodiscard]] static auto temporaryConstructor(ExpressionAST* expr)
      -> FunctionSymbol* {
    if (auto braced = ast_cast<BracedTypeConstructionAST>(expr))
      return braced->constructorSymbol;
    if (auto construction = ast_cast<TypeConstructionAST>(expr))
      return construction->constructorSymbol;
    if (auto call = ast_cast<CallExpressionAST>(expr))
      return call->constructorSymbol;
    return nullptr;
  }

  TranslationUnit* unit_;
};

[[nodiscard]] auto bodyStatement(FunctionBodyAST* functionBody)
    -> CompoundStatementAST* {
  if (auto compound = ast_cast<CompoundStatementFunctionBodyAST>(functionBody))
    return compound->statement;
  return nullptr;
}

void checkFallingOffTheEnd(TranslationUnit* unit, const Type* returnType,
                           AST* body, CompoundStatementAST* statement,
                           SourceLocation location) {
  if (!statement || !returnType) return;
  if (containsPlaceholderType(returnType)) return;
  if (isDependent(unit, returnType)) return;
  if (unit->typeTraits().is_void(returnType)) return;
  if (CoroutineFinder::isCoroutine(body)) return;

  ReachabilityAnalysis analysis{unit};
  if (!analysis.completesNormally(statement)) return;

  unit->warning(location,
                std::format("non-void function does not return a value of "
                            "type '{}' on every path",
                            to_string(returnType)));
}

[[nodiscard]] auto reportsBodyWarnings(TranslationUnit* unit) -> bool {
  return unit && unit->config().checkTypes;
}

}  // namespace

void checkDiscardedValueWarnings(TranslationUnit* unit,
                                 FunctionBodyAST* functionBody) {
  if (!reportsBodyWarnings(unit) || !functionBody) return;

  auto statement = bodyStatement(functionBody);
  if (!statement) return;

  DiscardedValueChecker{unit}.check(statement);
}

void checkReturnPathWarnings(TranslationUnit* unit, FunctionSymbol* function,
                             FunctionBodyAST* functionBody) {
  if (!reportsBodyWarnings(unit) || !function || !functionBody) return;

  auto statement = bodyStatement(functionBody);
  if (!statement) return;

  if (function->isConstructor() || function->isDestructor()) return;
  if (function->name() == unit->control()->getIdentifier("main") &&
      function->enclosingNamespace() && !function->enclosingNamespace()->name())
    return;

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return;

  checkFallingOffTheEnd(unit, functionType->returnType(), functionBody,
                        statement, statement->rbraceLoc);
}

void checkLambdaBodyWarnings(TranslationUnit* unit, LambdaExpressionAST* ast) {
  if (!reportsBodyWarnings(unit) || !ast || !ast->statement || !ast->symbol)
    return;

  DiscardedValueChecker{unit}.check(ast->statement);

  auto functionType = type_cast<FunctionType>(ast->symbol->type());
  if (!functionType) return;

  checkFallingOffTheEnd(unit, functionType->returnType(), ast->statement,
                        ast->statement, ast->statement->rbraceLoc);
}

}  // namespace cxx
