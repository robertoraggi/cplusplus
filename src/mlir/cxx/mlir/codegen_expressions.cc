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
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

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
  mlir::cxx::CondBranchOp::create(builder_, loc, value.value,
                                  mlir::ValueRange{}, mlir::ValueRange{},
                                  trueBlock, falseBlock);
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

auto Codegen::ExpressionVisitor::operator()(CharLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);
  auto value = std::int64_t(ast->literal->charValue());
  auto op = mlir::cxx::IntConstantOp::create(gen.builder_, loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);

  auto op =
      mlir::cxx::BoolConstantOp::create(gen.builder_, loc, type, ast->isTrue);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);
  auto value = ast->literal->integerValue();

  auto op = mlir::cxx::IntConstantOp::create(gen.builder_, loc, type, value);

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

  auto op = mlir::cxx::FloatConstantOp::create(gen.builder_, loc, type, value);

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
  auto loc = gen.getLocation(ast->literalLoc);
  auto type = gen.convertType(ast->type);
  auto resultType = mlir::cxx::PointerType::get(type.getContext(), type);

  auto it = gen.stringLiterals_.find(ast->literal);
  if (it == gen.stringLiterals_.end()) {
    // todo: clean up
    std::string str(ast->literal->stringValue());
    str.push_back('\0');

    auto initializer = gen.builder_.getStringAttr(str);

    // todo: generate unique name for the global
    auto name = gen.builder_.getStringAttr(gen.newUniqueSymbolName(".str"));

    auto x = mlir::OpBuilder(gen.module_->getContext());
    x.setInsertionPointToEnd(gen.module_.getBody());
    mlir::cxx::GlobalOp::create(x, loc, type, true, name, initializer);

    it = gen.stringLiterals_.insert_or_assign(ast->literal, name).first;
  }

  auto op =
      mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType, it->second);

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
  auto type = gen.convertType(ast->type);
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto loadOp =
      mlir::cxx::LoadOp::create(gen.builder_, loc, type, gen.thisValue_);

  return {loadOp};
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
          mlir::cxx::IntConstantOp::create(gen.builder_, loc, type, *value);
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
    auto op = mlir::cxx::IdOp::create(gen.builder_, loc, name);
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
  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto indexExpressionResult = gen.expression(ast->indexExpression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto resultType = gen.convertType(control()->add_pointer(ast->type));

  if (control()->is_pointer(ast->baseExpression->type)) {
    auto op = mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType,
                                          baseExpressionResult.value,
                                          indexExpressionResult.value);

    return {op};
  }

  auto op = mlir::cxx::SubscriptOp::create(gen.builder_, loc, resultType,
                                           baseExpressionResult.value,
                                           indexExpressionResult.value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto check_direct_call = [&]() -> std::optional<ExpressionResult> {
    auto func = ast->baseExpression;

    while (auto nested = ast_cast<NestedExpressionAST>(func)) {
      func = nested->expression;
    }

    if (auto member = ast_cast<MemberExpressionAST>(func)) {
      auto thisValue = gen.expression(member->baseExpression);
      auto functionSymbol = symbol_cast<FunctionSymbol>(member->symbol);

      auto funcOp = gen.findOrCreateFunction(functionSymbol);

      mlir::SmallVector<mlir::Value> arguments;
      arguments.push_back(thisValue.value);
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

      auto op = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                          funcOp.getSymName(), arguments,
                                          mlir::TypeAttr{});

      if (functionType->isVariadic()) {
        op.setVarCalleeType(
            cast<mlir::cxx::FunctionType>(gen.convertType(functionType)));
      }

      return ExpressionResult{op.getResult()};
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

    auto op = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                        funcOp.getSymName(), arguments,
                                        mlir::TypeAttr{});

    if (functionType->isVariadic()) {
      op.setVarCalleeType(
          cast<mlir::cxx::FunctionType>(gen.convertType(functionType)));
    }

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

  auto op = mlir::cxx::CallOp::create(gen.builder_,
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
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && !field->isStatic()) {
    // todo: introduce ClassLayout to avoid linear searches and support c++
    // class layout
    int fieldIndex = 0;
    auto classSymbol = symbol_cast<ClassSymbol>(field->parent());
    for (auto member : cxx::views::members(classSymbol)) {
      auto f = symbol_cast<FieldSymbol>(member);
      if (!f) continue;
      if (f->isStatic()) continue;
      if (member == field) break;
      ++fieldIndex;
    }

    auto baseExpressionResult = gen.expression(ast->baseExpression);

    auto loc = gen.getLocation(ast->unqualifiedId->firstSourceLocation());

    auto resultType = gen.convertType(control()->add_pointer(ast->type));

    auto op = mlir::cxx::MemberOp::create(
        gen.builder_, loc, resultType, baseExpressionResult.value, fieldIndex);

    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->baseExpression);

  if (control()->is_integral_or_unscoped_enum(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrTy =
        mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
    auto elementTy = ptrTy.getElementType();
    auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                            expressionResult.value);
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto oneOp = mlir::cxx::IntConstantOp::create(
        gen.builder_, loc, resultTy,
        ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1);
    auto addOp =
        mlir::cxx::AddIOp::create(gen.builder_, loc, resultTy, loadOp, oneOp);
    mlir::cxx::StoreOp::create(gen.builder_, loc, addOp,
                               expressionResult.value);
    return {loadOp};
  }
  if (control()->is_floating_point(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrTy =
        mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
    auto elementTy = ptrTy.getElementType();
    auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                            expressionResult.value);
    auto resultTy = gen.convertType(ast->baseExpression->type);

    mlir::Value one;
    double v = ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1;

    switch (control()->remove_cvref(ast->baseExpression->type)->kind()) {
      case TypeKind::kFloat:
        one = mlir::cxx::FloatConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF32FloatAttr(v));
        break;

      case TypeKind::kDouble:
        one = mlir::cxx::FloatConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF64FloatAttr(v));
        break;

      case TypeKind::kLongDouble:
        one = mlir::cxx::FloatConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF64FloatAttr(v));
        break;

      default:
        // Handle other float types if necessary
        auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                   "unsupported float type");
        return {op};
    }

    auto addOp =
        mlir::cxx::AddFOp::create(gen.builder_, loc, resultTy, loadOp, one);
    mlir::cxx::StoreOp::create(gen.builder_, loc, addOp,
                               expressionResult.value);
    return {loadOp};
  }

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
        auto op = mlir::cxx::NotOp::create(gen.builder_, loc, resultType,
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

      if (control()->is_integral_or_unscoped_enum(ast->type)) {
        auto zero =
            mlir::cxx::IntConstantOp::create(gen.builder_, loc, resultType, 0);
        auto op = mlir::cxx::SubIOp::create(gen.builder_, loc, resultType, zero,
                                            expressionResult.value);

        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        resultType.dump();

        mlir::FloatAttr value;
        switch (ast->type->kind()) {
          case TypeKind::kFloat:
            value = gen.builder_.getF32FloatAttr(0);
            break;
          case TypeKind::kDouble:
            value = gen.builder_.getF64FloatAttr(0);
            break;
          case TypeKind::kLongDouble:
            value = gen.builder_.getF64FloatAttr(0);
            break;
          default:
            // Handle other float types if necessary
            auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                       "unsupported float type");
            return {op};
        }

        auto zero = mlir::cxx::FloatConstantOp::create(gen.builder_, loc,
                                                       resultType, value);
        auto op = mlir::cxx::SubFOp::create(gen.builder_, loc, resultType, zero,
                                            expressionResult.value);

        return {op};
      }

      break;
    }

    case TokenKind::T_TILDE: {
      // unary bitwise not
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      auto loc = gen.getLocation(ast->opLoc);
      auto op = mlir::cxx::NotOp::create(gen.builder_, loc, resultType,
                                         expressionResult.value);

      return {op};
    }

    case TokenKind::T_AMP: {
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

    case TokenKind::T_STAR: {
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

    case TokenKind::T_MINUS_MINUS:
    case TokenKind::T_PLUS_PLUS: {
      auto expressionResult = gen.expression(ast->expression);

      if (control()->is_floating_point(ast->expression->type)) {
        mlir::Value one;

        switch (control()->remove_cvref(ast->expression->type)->kind()) {
          case TypeKind::kFloat:
            one = mlir::cxx::FloatConstantOp::create(
                gen.builder_, gen.getLocation(ast->opLoc),
                gen.convertType(ast->expression->type),
                gen.builder_.getF32FloatAttr(1.0));
            break;

          case TypeKind::kDouble:
            one = mlir::cxx::FloatConstantOp::create(
                gen.builder_, gen.getLocation(ast->opLoc),
                gen.convertType(ast->expression->type),
                gen.builder_.getF64FloatAttr(1.0));
            break;

          case TypeKind::kLongDouble:
            one = mlir::cxx::FloatConstantOp::create(
                gen.builder_, gen.getLocation(ast->opLoc),
                gen.convertType(ast->expression->type),
                gen.builder_.getF64FloatAttr(1.0));
            break;

          default:
            // Handle other float types if necessary
            auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                       "unsupported float type");
            return {op};
        }

        auto loc = gen.getLocation(ast->opLoc);

        auto resultType = gen.convertType(ast->type);

        auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                                expressionResult.value);

        mlir::Value addOp;

        if (ast->op == TokenKind::T_MINUS_MINUS)
          addOp = mlir::cxx::SubFOp::create(gen.builder_, loc, resultType,
                                            loadOp, one);
        else
          addOp = mlir::cxx::AddFOp::create(gen.builder_, loc, resultType,
                                            loadOp, one);

        auto storeOp = mlir::cxx::StoreOp::create(gen.builder_, loc, addOp,
                                                  expressionResult.value);

        if (is_glvalue(ast)) {
          return expressionResult;
        }

        auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                            expressionResult.value);

        return {op};
      } else if (control()->is_arithmetic(ast->expression->type)) {
        auto loc = gen.getLocation(ast->opLoc);

        auto oneOp = mlir::cxx::IntConstantOp::create(
            gen.builder_, loc, gen.convertType(control()->getIntType()), 1);

        auto castOneOp = mlir::cxx::IntegralCastOp::create(
            gen.builder_, loc, gen.convertType(ast->expression->type), oneOp);

        auto resultType = gen.convertType(ast->type);

        auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                                expressionResult.value);

        mlir::Value addOp;

        if (ast->op == TokenKind::T_MINUS_MINUS)
          addOp = mlir::cxx::SubIOp::create(gen.builder_, loc, resultType,
                                            loadOp, castOneOp);
        else
          addOp = mlir::cxx::AddIOp::create(gen.builder_, loc, resultType,
                                            loadOp, castOneOp);

        auto storeOp = mlir::cxx::StoreOp::create(gen.builder_, loc, addOp,
                                                  expressionResult.value);

        if (is_glvalue(ast)) {
          return expressionResult;
        }

        auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                            expressionResult.value);

        return {op};
      } else if (control()->is_pointer(ast->expression->type)) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto intTy =
            mlir::cxx::IntegerType::get(gen.builder_.getContext(), 32, true);
        auto one = mlir::cxx::IntConstantOp::create(
            gen.builder_, loc, intTy,
            ast->op == TokenKind::T_MINUS_MINUS ? -1 : 1);
        auto ptrTy = mlir::cast<mlir::cxx::PointerType>(
            expressionResult.value.getType());
        auto elementTy = ptrTy.getElementType();
        auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                                expressionResult.value);
        auto addOp = mlir::cxx::PtrAddOp::create(gen.builder_, loc, elementTy,
                                                 loadOp, one);
        mlir::cxx::StoreOp::create(gen.builder_, loc, addOp,
                                   expressionResult.value);

        if (is_glvalue(ast)) {
          return expressionResult;
        }

        auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                            expressionResult.value);
        return {op};
      }

      auto op =
          gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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

      auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                          expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kIntegralPromotion: {
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      if (is_bool(ast->type)) {
        // If the result type is a boolean, we can use a specialized cast
        auto op = mlir::cxx::IntToBoolOp::create(gen.builder_, loc, resultType,
                                                 expressionResult.value);
        return {op};
      }

      if (is_bool(ast->expression->type)) {
        // If the expression type is a boolean, we can use a specialized cast
        auto op = mlir::cxx::BoolToIntOp::create(gen.builder_, loc, resultType,
                                                 expressionResult.value);
        return {op};
      }

      // generate an integral cast
      auto op = mlir::cxx::IntegralCastOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kFloatingPointPromotion:
    case ImplicitCastKind::kFloatingPointConversion: {
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      // generate a floating point cast
      auto op = mlir::cxx::FloatingPointCastOp::create(
          gen.builder_, loc, resultType, expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kFloatingIntegralConversion: {
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      if (control()->is_floating_point(ast->type)) {
        // If the result type is a floating point, we can use a specialized
        // cast
        auto op = mlir::cxx::IntToFloatOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
        return {op};
      }

      if (control()->is_integral(ast->type)) {
        // If the expression type is an integral, we can use a specialized
        // cast
        auto op = mlir::cxx::FloatToIntOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
        return {op};
      }

      break;
    }

    case ImplicitCastKind::kArrayToPointerConversion: {
      // generate an array to pointer conversion
      auto expressionResult = gen.expression(ast->expression);
      auto resultType = gen.convertType(ast->type);

      auto op = mlir::cxx::ArrayToPointerOp::create(
          gen.builder_, loc, resultType, expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kQualificationConversion: {
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

    default:
      break;

  }  // switch

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto op = mlir::cxx::ImplicitCastOp::create(gen.builder_,
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

  if (ast->op == TokenKind::T_BAR_BAR) {
    auto t = gen.newTemp(control()->getBoolType(), ast->opLoc);

    auto trueBlock = gen.newBlock();
    auto continueBlock = gen.newBlock();
    auto falseBlock = gen.newBlock();
    auto endBlock = gen.newBlock();

    gen.condition(ast->leftExpression, trueBlock, continueBlock);

    gen.builder_.setInsertionPointToEnd(continueBlock);
    gen.condition(ast->rightExpression, trueBlock, falseBlock);

    // build the true block
    gen.builder_.setInsertionPointToEnd(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = mlir::cxx::BoolConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type, true);

    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               trueValue, t);

    auto endLoc = gen.getLocation(ast->lastSourceLocation());
    gen.branch(endLoc, endBlock);

    // build the false block
    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::cxx::BoolConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type, false);
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t);
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    // place the end block
    gen.builder_.setInsertionPointToEnd(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), resultType, t);
    return {loadOp};
  }

  if (ast->op == TokenKind::T_AMP_AMP) {
    auto t = gen.newTemp(control()->getBoolType(), ast->opLoc);

    auto trueBlock = gen.newBlock();
    auto continueBlock = gen.newBlock();
    auto falseBlock = gen.newBlock();
    auto endBlock = gen.newBlock();

    gen.condition(ast->leftExpression, continueBlock, falseBlock);

    gen.builder_.setInsertionPointToEnd(continueBlock);
    gen.condition(ast->rightExpression, trueBlock, falseBlock);

    // build the true block
    gen.builder_.setInsertionPointToEnd(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = mlir::cxx::BoolConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type, true);

    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               trueValue, t);

    auto endLoc = gen.getLocation(ast->lastSourceLocation());
    gen.branch(endLoc, endBlock);

    // build the false block
    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::cxx::BoolConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type, false);
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t);
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    // place the end block
    gen.builder_.setInsertionPointToEnd(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), resultType, t);
    return {loadOp};
  }

  auto loc = gen.getLocation(ast->opLoc);
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->op) {
    case TokenKind::T_PLUS: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::AddIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::AddFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_pointer(ast->leftExpression->type) &&
          control()->is_integer(ast->rightExpression->type)) {
        auto op = mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType,
                                              leftExpressionResult.value,
                                              rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_MINUS: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::SubIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::SubFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_STAR: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::MulIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::MulFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_SLASH: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::DivIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::DivFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_PERCENT: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::ModIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_LESS: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::ShiftLeftOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_GREATER: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::ShiftRightOp::create(gen.builder_, loc, resultType,
                                                  leftExpressionResult.value,
                                                  rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EQUAL_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::EqualOp::create(gen.builder_, loc, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::EqualFOp::create(gen.builder_, loc, resultType,
                                              leftExpressionResult.value,
                                              rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EXCLAIM_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::NotEqualOp::create(gen.builder_, loc, resultType,
                                                leftExpressionResult.value,
                                                rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::NotEqualFOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::LessThanOp::create(gen.builder_, loc, resultType,
                                                leftExpressionResult.value,
                                                rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::LessThanFOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::LessEqualOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::LessEqualFOp::create(gen.builder_, loc, resultType,
                                                  leftExpressionResult.value,
                                                  rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::GreaterThanOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::GreaterThanFOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_EQUAL: {
      if (control()->is_integral(ast->type)) {
        auto op = mlir::cxx::GreaterEqualOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(ast->type)) {
        auto op = mlir::cxx::GreaterEqualFOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
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
  auto trueBlock = gen.newBlock();
  auto falseBlock = gen.newBlock();
  auto endBlock = gen.newBlock();

  auto t = gen.newTemp(ast->type, ast->questionLoc);

  gen.condition(ast->condition, trueBlock, falseBlock);

  auto endLoc = gen.getLocation(ast->lastSourceLocation());

  // place the true block
  gen.builder_.setInsertionPointToEnd(trueBlock);
  auto trueExpressionResult = gen.expression(ast->iftrueExpression);
  auto trueValue = mlir::cxx::StoreOp::create(gen.builder_,
                                              gen.getLocation(ast->questionLoc),
                                              trueExpressionResult.value, t);
  gen.branch(endLoc, endBlock);

  // place the false block
  gen.builder_.setInsertionPointToEnd(falseBlock);
  auto falseExpressionResult = gen.expression(ast->iffalseExpression);
  auto falseValue =
      mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->colonLoc),
                                 falseExpressionResult.value, t);
  gen.branch(endLoc, endBlock);

  // place the end block
  gen.builder_.setInsertionPointToEnd(endBlock);

  if (format == ExpressionFormat::kSideEffect) return {};

  auto resultType = gen.convertType(ast->type);
  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, gen.getLocation(ast->colonLoc), resultType, t);
  return {loadOp};
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

    mlir::cxx::StoreOp::create(gen.builder_, loc, rightExpressionResult.value,
                               leftExpressionResult.value);

    if (format == ExpressionFormat::kSideEffect) {
      return {};
    }

    if (gen.unit_->language() == LanguageKind::kC) {
      // in C mode the result of the assignment is an rvalue
      auto resultLoc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = gen.convertType(ast->leftExpression->type);

      // generate a load
      auto op = mlir::cxx::LoadOp::create(gen.builder_, resultLoc, resultType,
                                          leftExpressionResult.value);

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
