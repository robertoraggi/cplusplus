// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

// mlir
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

namespace cxx {

struct Codegen::ExpressionVisitor {
  Codegen& gen;
  ExpressionFormat format = ExpressionFormat::kValue;

  [[nodiscard]] auto control() const -> Control* { return gen.control(); }

  [[nodiscard]] auto is_bool(const Type* type) const -> bool {
    return type_cast<BoolType>(control()->remove_cv(type));
  }

  auto operator()(GeneratedLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(CharLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(BoolLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(IntLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(FloatLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(NullptrLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(StringLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(UserDefinedStringLiteralExpressionAST* ast)
      -> ExpressionResult;
  auto operator()(ObjectLiteralExpressionAST* ast) -> ExpressionResult;
  auto operator()(GenericSelectionExpressionAST* ast) -> ExpressionResult;
  auto operator()(ThisExpressionAST* ast) -> ExpressionResult;
  auto operator()(NestedStatementExpressionAST* ast) -> ExpressionResult;
  auto operator()(NestedExpressionAST* ast) -> ExpressionResult;
  auto operator()(IdExpressionAST* ast) -> ExpressionResult;
  auto operator()(LambdaExpressionAST* ast) -> ExpressionResult;
  auto operator()(FoldExpressionAST* ast) -> ExpressionResult;
  auto operator()(RightFoldExpressionAST* ast) -> ExpressionResult;
  auto operator()(LeftFoldExpressionAST* ast) -> ExpressionResult;
  auto operator()(RequiresExpressionAST* ast) -> ExpressionResult;
  auto operator()(VaArgExpressionAST* ast) -> ExpressionResult;
  auto operator()(SubscriptExpressionAST* ast) -> ExpressionResult;
  auto operator()(CallExpressionAST* ast) -> ExpressionResult;
  auto operator()(TypeConstructionAST* ast) -> ExpressionResult;
  auto operator()(BracedTypeConstructionAST* ast) -> ExpressionResult;
  auto operator()(SpliceMemberExpressionAST* ast) -> ExpressionResult;
  auto operator()(MemberExpressionAST* ast) -> ExpressionResult;
  auto operator()(PostIncrExpressionAST* ast) -> ExpressionResult;
  auto operator()(CppCastExpressionAST* ast) -> ExpressionResult;
  auto operator()(BuiltinBitCastExpressionAST* ast) -> ExpressionResult;
  auto operator()(BuiltinOffsetofExpressionAST* ast) -> ExpressionResult;
  auto operator()(TypeidExpressionAST* ast) -> ExpressionResult;
  auto operator()(TypeidOfTypeExpressionAST* ast) -> ExpressionResult;
  auto operator()(SpliceExpressionAST* ast) -> ExpressionResult;
  auto operator()(GlobalScopeReflectExpressionAST* ast) -> ExpressionResult;
  auto operator()(NamespaceReflectExpressionAST* ast) -> ExpressionResult;
  auto operator()(TypeIdReflectExpressionAST* ast) -> ExpressionResult;
  auto operator()(ReflectExpressionAST* ast) -> ExpressionResult;
  auto operator()(LabelAddressExpressionAST* ast) -> ExpressionResult;
  auto operator()(UnaryExpressionAST* ast) -> ExpressionResult;
  auto operator()(AwaitExpressionAST* ast) -> ExpressionResult;
  auto operator()(SizeofExpressionAST* ast) -> ExpressionResult;
  auto operator()(SizeofTypeExpressionAST* ast) -> ExpressionResult;
  auto operator()(SizeofPackExpressionAST* ast) -> ExpressionResult;
  auto operator()(AlignofTypeExpressionAST* ast) -> ExpressionResult;
  auto operator()(AlignofExpressionAST* ast) -> ExpressionResult;
  auto operator()(NoexceptExpressionAST* ast) -> ExpressionResult;
  auto operator()(NewExpressionAST* ast) -> ExpressionResult;
  auto operator()(DeleteExpressionAST* ast) -> ExpressionResult;
  auto operator()(CastExpressionAST* ast) -> ExpressionResult;
  auto operator()(ImplicitCastExpressionAST* ast) -> ExpressionResult;
  auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;
  auto operator()(ConditionalExpressionAST* ast) -> ExpressionResult;
  auto operator()(YieldExpressionAST* ast) -> ExpressionResult;
  auto operator()(ThrowExpressionAST* ast) -> ExpressionResult;
  auto operator()(AssignmentExpressionAST* ast) -> ExpressionResult;
  auto operator()(CompoundAssignmentExpressionAST* ast) -> ExpressionResult;
  auto operator()(PackExpansionExpressionAST* ast) -> ExpressionResult;
  auto operator()(DesignatedInitializerClauseAST* ast) -> ExpressionResult;
  auto operator()(TypeTraitExpressionAST* ast) -> ExpressionResult;
  auto operator()(ConditionExpressionAST* ast) -> ExpressionResult;
  auto operator()(EqualInitializerAST* ast) -> ExpressionResult;
  auto operator()(BracedInitListAST* ast) -> ExpressionResult;
  auto operator()(ParenInitializerAST* ast) -> ExpressionResult;
};

struct Codegen::NewInitializerVisitor {
  Codegen& gen;

  auto operator()(NewParenInitializerAST* ast) -> NewInitializerResult;
  auto operator()(NewBracedInitializerAST* ast) -> NewInitializerResult;
};

auto Codegen::expression(ExpressionAST* ast, ExpressionFormat format)
    -> ExpressionResult {
  if (ast) return visit(ExpressionVisitor{*this, format}, ast);
  return {};
}

void Codegen::condition(ExpressionAST* ast, mlir::Block* trueBlock,
                        mlir::Block* falseBlock) {
  if (!ast) return;

  if (auto nested = ast_cast<NestedExpressionAST>(ast)) {
    condition(nested->expression, trueBlock, falseBlock);
    return;
  }

  if (auto binop = ast_cast<BinaryExpressionAST>(ast)) {
    if (binop->op == TokenKind::T_AMP_AMP) {
      auto nextBlock = newBlock();
      condition(binop->leftExpression, nextBlock, falseBlock);
      builder_.setInsertionPointToEnd(nextBlock);
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }

    if (binop->op == TokenKind::T_BAR_BAR) {
      auto nextBlock = newBlock();
      condition(binop->leftExpression, trueBlock, nextBlock);
      builder_.setInsertionPointToEnd(nextBlock);
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }
  }

  const auto loc = getLocation(ast->firstSourceLocation());
  auto value = expression(ast);
  builder_.create<mlir::cxx::CondBranchOp>(loc, value.value, mlir::ValueRange{},
                                           mlir::ValueRange{}, trueBlock,
                                           falseBlock);
}

auto Codegen::newInitializer(NewInitializerAST* ast) -> NewInitializerResult {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto Codegen::newPlacement(NewPlacementAST* ast) -> NewPlacementResult {
  if (!ast) return {};

  for (auto node : ListView{ast->expressionList}) {
    auto value = expression(node);
  }

  return {};
}

auto Codegen::ExpressionVisitor::operator()(GeneratedLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);
  auto value = std::int64_t(ast->literal->charValue());
  auto op = gen.builder_.create<mlir::cxx::IntConstantOp>(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);

  auto op =
      gen.builder_.create<mlir::cxx::BoolConstantOp>(loc, type, ast->isTrue);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);
  auto value = ast->literal->integerValue();

  auto op = gen.builder_.create<mlir::cxx::IntConstantOp>(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);

  mlir::TypedAttr value;

  switch (ast->type->kind()) {
    case TypeKind::kFloat:
      value = gen.builder_.getF32FloatAttr(ast->literal->floatValue());
      break;
    case TypeKind::kDouble:
      value = gen.builder_.getF64FloatAttr(ast->literal->floatValue());
      break;
    case TypeKind::kLongDouble:
      value = gen.builder_.getF64FloatAttr(ast->literal->floatValue());
      break;
    default:
      // Handle other float types if necessary
      auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "unsupported float type");
      return {op};
  }

  auto op = gen.builder_.create<mlir::cxx::FloatConstantOp>(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NullptrLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(GenericSelectionExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NestedStatementExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  return gen.expression(ast->expression, format);
}

auto Codegen::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  if (auto local = gen.findOrCreateLocal(ast->symbol)) {
    return {local.value()};
  }

  if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    auto value = enumerator->value().and_then([&](const ConstValue& value) {
      ASTInterpreter interp{gen.unit_};
      return interp.toInt(value);
    });

    if (value.has_value()) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto type = gen.convertType(enumerator->type());
      auto op =
          gen.builder_.create<mlir::cxx::IntConstantOp>(loc, type, *value);
      return {op};
    }
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto nestedNameSpecifierResult = gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  if (auto id = ast_cast<NameIdAST>(ast->unqualifiedId);
      id && !ast->nestedNameSpecifier) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto name = id->identifier->name();
    auto op = gen.builder_.create<mlir::cxx::IdOp>(loc, name);
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  for (auto node : ListView{ast->captureList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto templateRequiresClauseResult = gen(ast->templateRequiresClause);
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = gen(node);
  }

  auto exceptionSpecifierResult = gen(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto trailingReturnTypeResult = gen(ast->trailingReturnType);
  auto requiresClauseResult = gen(ast->requiresClause);
  gen.statement(ast->statement);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->requirementList}) {
    auto value = gen(node);
  }
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto indexExpressionResult = gen.expression(ast->indexExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto check_direct_call = [&]() -> std::optional<ExpressionResult> {
    auto func = ast->baseExpression;

    while (auto nested = ast_cast<NestedExpressionAST>(func)) {
      func = nested->expression;
    }

    auto id = ast_cast<IdExpressionAST>(func);
    if (!id) return {};

    auto functionSymbol = symbol_cast<FunctionSymbol>(id->symbol);

    if (!functionSymbol) return {};

    auto funcOp = gen.findOrCreateFunction(functionSymbol);

    mlir::SmallVector<mlir::Value> arguments;
    for (auto node : ListView{ast->expressionList}) {
      auto value = gen.expression(node);
      arguments.push_back(value.value);
    }

    auto loc = gen.getLocation(ast->lparenLoc);

    auto functionType = type_cast<FunctionType>(functionSymbol->type());
    mlir::SmallVector<mlir::Type> resultTypes;
    if (!control()->is_void(functionType->returnType())) {
      resultTypes.push_back(gen.convertType(functionType->returnType()));
    }

    auto op = gen.builder_.create<mlir::cxx::CallOp>(
        loc, resultTypes, funcOp.getSymName(), arguments, mlir::ArrayAttr{},
        mlir::ArrayAttr{});

    return ExpressionResult{op.getResult()};
  };

  if (auto op = check_direct_call(); op.has_value()) {
    return *op;
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto baseExpressionResult = gen.expression(ast->baseExpression);

  std::vector<mlir::Value> arguments;

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  auto loc = gen.getLocation(ast->lparenLoc);

  auto op = gen.builder_.create<mlir::cxx::CallOp>(
      loc, baseExpressionResult.value, arguments);
#endif
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeSpecifierResult = gen(ast->typeSpecifier);

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeSpecifierResult = gen(ast->typeSpecifier);
  auto bracedInitListResult = gen.expression(ast->bracedInitList);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SpliceMemberExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto splicerResult = gen(ast->splicer);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto nestedNameSpecifierResult = gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto baseExpressionResult = gen.expression(ast->baseExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BuiltinBitCastExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BuiltinOffsetofExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto splicerResult = gen(ast->splicer);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NamespaceReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeIdReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  auto typeIdResult = gen.typeId(ast->typeId);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  switch (ast->op) {
    case TokenKind::T_EXCLAIM: {
      if (type_cast<BoolType>(control()->remove_cv(ast->type))) {
        auto loc = gen.getLocation(ast->opLoc);
        auto expressionResult = gen.expression(ast->expression);
        auto resultType = gen.convertType(ast->type);
        auto op = gen.builder_.create<mlir::cxx::NotOp>(loc, resultType,
                                                        expressionResult.value);
        return {op};
      }
      break;
    }

    case TokenKind::T_PLUS: {
      // unary plus, no-op
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

    case TokenKind::T_MINUS: {
      // unary minus
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      auto loc = gen.getLocation(ast->opLoc);
      auto zero =
          gen.builder_.create<mlir::cxx::IntConstantOp>(loc, resultType, 0);
      auto op = gen.builder_.create<mlir::cxx::SubIOp>(loc, resultType, zero,
                                                       expressionResult.value);

      return {op};
    }

    case TokenKind::T_TILDE: {
      // unary bitwise not
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      auto loc = gen.getLocation(ast->opLoc);
      auto op = gen.builder_.create<mlir::cxx::NotOp>(loc, resultType,
                                                      expressionResult.value);

      return {op};
    }

    default:
      break;
  }  // switch

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto newPlacementResult = gen(ast->newPlacement);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen.declarator(ast->declarator);
  auto newInitalizerResult = gen(ast->newInitalizer);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);

  return expressionResult;
}

auto Codegen::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  switch (ast->castKind) {
    case ImplicitCastKind::kLValueToRValueConversion: {
      // generate a load
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      auto op = gen.builder_.create<mlir::cxx::LoadOp>(loc, resultType,
                                                       expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kIntegralPromotion: {
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      if (is_bool(ast->type)) {
        // If the result type is a boolean, we can use a specialized cast
        auto op = gen.builder_.create<mlir::cxx::IntToBoolOp>(
            loc, resultType, expressionResult.value);
        return {op};
      }

      if (is_bool(ast->expression->type)) {
        // If the expression type is a boolean, we can use a specialized cast
        auto op = gen.builder_.create<mlir::cxx::BoolToIntOp>(
            loc, resultType, expressionResult.value);
        return {op};
      }

      // generate an integral cast
      auto op = gen.builder_.create<mlir::cxx::IntegralCastOp>(
          loc, resultType, expressionResult.value);

      return {op};
    }

    default:
      break;

  }  // switch

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto op = gen.builder_.create<mlir::cxx::ImplicitCastOp>(
      loc, to_string(ast->castKind), expressionResult.value);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  if (ast->op == TokenKind::T_COMMA) {
    // For the comma operator, we evaluate the left expression for its side
    // effects and then return the right expression as the result.
    (void)gen.expression(ast->leftExpression, ExpressionFormat::kSideEffect);
    return gen.expression(ast->rightExpression, format);
  }

  auto loc = gen.getLocation(ast->opLoc);
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->op) {
    case TokenKind::T_PLUS: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::AddIOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_MINUS: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::SubIOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_STAR: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::MulIOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_SLASH: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::DivIOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_PERCENT: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::ModIOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_LESS: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::ShiftLeftOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_GREATER: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::ShiftRightOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EQUAL_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::EqualOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EXCLAIM_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::NotEqualOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::LessThanOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::LessEqualOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::GreaterThanOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = gen.builder_.create<mlir::cxx::GreaterEqualOp>(
            loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    default:
      break;
  }  // switch

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto conditionResult = gen.expression(ast->condition);
  auto iftrueExpressionResult = gen.expression(ast->iftrueExpression);
  auto iffalseExpressionResult = gen.expression(ast->iffalseExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  if (ast->op == TokenKind::T_EQUAL) {
    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);

    // Generate a store operation
    const auto loc = gen.getLocation(ast->opLoc);

    gen.builder_.create<mlir::cxx::StoreOp>(loc, rightExpressionResult.value,
                                            leftExpressionResult.value);

    if (format == ExpressionFormat::kSideEffect) {
      return {};
    }

    if (gen.unit_->language() == LanguageKind::kC) {
      // in C mode the result of the assignment is an rvalue
      auto resultLoc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = gen.convertType(ast->leftExpression->type);

      // generate a load
      auto op = gen.builder_.create<mlir::cxx::LoadOp>(
          resultLoc, resultType, leftExpressionResult.value);

      return {op};
    }

    return leftExpressionResult;
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(DesignatedInitializerClauseAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
auto initializerResult = gen.expression(ast->initializer);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  for (auto node : ListView{ast->typeIdList}) {
    auto value = gen(node);
  }
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen.declarator(ast->declarator);
  auto initializerResult = gen.expression(ast->initializer);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  // auto op =
  //     gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  auto expressionResult = gen.expression(ast->expression);

  return expressionResult;
}

auto Codegen::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }
#endif

  return {op};
}

auto Codegen::NewInitializerVisitor::operator()(NewParenInitializerAST* ast)
    -> NewInitializerResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {};
}

auto Codegen::NewInitializerVisitor::operator()(NewBracedInitializerAST* ast)
    -> NewInitializerResult {
  auto bracedInitListResult = gen.expression(ast->bracedInitList);

  return {};
}

}  // namespace cxx