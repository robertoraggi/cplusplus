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
#include <cxx/mlir/codegen.h>
#include <cxx/names.h>
#include <cxx/source_location.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <format>

namespace cxx {
struct [[nodiscard]] Codegen::ExpressionVisitor {
  Codegen& gen;
  ExpressionFormat format = ExpressionFormat::kValue;

  [[nodiscard]] auto control() const -> Control* { return gen.control(); }

  [[nodiscard]] auto is_bool(const Type* type) const -> bool {
    return type_cast<BoolType>(gen.unit_->typeTraits().remove_cv(type));
  }

  [[nodiscard]] auto emitMemberAccess(MemberExpressionAST* ast)
      -> std::optional<std::pair<mlir::Value, ClassLayout::MemberInfo>>;

  [[nodiscard]] auto emitThisFieldAddress(FieldSymbol* field,
                                          SourceLocation loc)
      -> std::optional<std::pair<mlir::Value, ClassLayout::MemberInfo>>;

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
  auto operator()(PackIndexExpressionAST* ast) -> ExpressionResult;
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
  auto operator()(TargetExpressionAST* ast) -> ExpressionResult;
  auto operator()(RightExpressionAST* ast) -> ExpressionResult;
  auto operator()(CompoundAssignmentExpressionAST* ast) -> ExpressionResult;
  auto operator()(PackExpansionExpressionAST* ast) -> ExpressionResult;
  auto operator()(DesignatedInitializerClauseAST* ast) -> ExpressionResult;
  auto operator()(TypeTraitExpressionAST* ast) -> ExpressionResult;
  auto operator()(ConditionExpressionAST* ast) -> ExpressionResult;
  auto operator()(EqualInitializerAST* ast) -> ExpressionResult;
  auto operator()(BracedInitListAST* ast) -> ExpressionResult;
  auto operator()(ParenInitializerAST* ast) -> ExpressionResult;

  auto emitUnaryOpNot(UnaryExpressionAST* ast) -> ExpressionResult;
  auto emitUnaryOpMinus(UnaryExpressionAST* ast) -> ExpressionResult;
  auto emitUnaryOpTilde(UnaryExpressionAST* ast) -> ExpressionResult;
  auto emitUnaryOpIncrDecr(UnaryExpressionAST* ast) -> ExpressionResult;
  auto emitUnaryOpIncrDecrFloat(UnaryExpressionAST* ast, ExpressionResult expr)
      -> ExpressionResult;
  auto emitUnaryOpIncrDecrIntegral(UnaryExpressionAST* ast,
                                   ExpressionResult expr) -> ExpressionResult;
  auto emitUnaryOpIncrDecrPointer(UnaryExpressionAST* ast,
                                  ExpressionResult expr) -> ExpressionResult;

  auto binaryExpression(SourceLocation opLoc, TokenKind op,
                        mlir::Type resultType, ExpressionAST* leftExpression,
                        ExpressionAST* rightExpression,
                        ExpressionResult leftExpressionResult,
                        ExpressionResult rightExpressionResult)
      -> ExpressionResult;

  auto emitBinaryArithmeticOp(SourceLocation loc, TokenKind op,
                              mlir::Type resultType, const Type* leftType,
                              mlir::Value left, mlir::Value right)
      -> ExpressionResult;
  auto emitBinaryArithmeticOpFloat(SourceLocation loc, TokenKind op,
                                   mlir::Type resultType, mlir::Value left,
                                   mlir::Value right) -> ExpressionResult;
  auto emitBinaryArithmeticOpIntegral(SourceLocation loc, TokenKind op,
                                      mlir::Type resultType,
                                      const Type* leftType, mlir::Value left,
                                      mlir::Value right) -> ExpressionResult;
  auto emitBinaryArithmeticOpPointer(SourceLocation loc, TokenKind op,
                                     mlir::Type resultType, mlir::Value left,
                                     mlir::Value right) -> ExpressionResult;

  auto emitBinaryShiftOp(SourceLocation loc, TokenKind op,
                         mlir::Type resultType, const Type* leftType,
                         mlir::Value left, mlir::Value right)
      -> ExpressionResult;

  auto emitBinaryComparisonOp(SourceLocation loc, TokenKind op,
                              mlir::Type resultType, const Type* leftType,
                              mlir::Value left, mlir::Value right)
      -> ExpressionResult;
  auto emitBinaryComparisonOpFloat(SourceLocation loc, TokenKind op,
                                   mlir::Type resultType, mlir::Value left,
                                   mlir::Value right) -> ExpressionResult;
  auto emitBinaryComparisonOpIntegral(SourceLocation loc, TokenKind op,
                                      mlir::Type resultType,
                                      const Type* leftType, mlir::Value left,
                                      mlir::Value right) -> ExpressionResult;
  auto emitBinaryComparisonOpPointer(SourceLocation loc, TokenKind op,
                                     mlir::Type resultType,
                                     const Type* leftType, mlir::Value left,
                                     mlir::Value right) -> ExpressionResult;
  auto emitBinaryBitwiseOp(SourceLocation loc, TokenKind op,
                           mlir::Type resultType, mlir::Value left,
                           mlir::Value right) -> ExpressionResult;

  auto emitImplicitCast(ImplicitCastExpressionAST* ast) -> ExpressionResult;
  auto emitLValueToRValueConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitNumericConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitPointerConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitDerivedToBaseConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitUserDefinedConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitBuiltinCall(CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
      -> ExpressionResult;

  auto codegenBuiltinDispatch(CallExpressionAST* ast,
                              BuiltinFunctionKind builtinKind)
      -> std::optional<ExpressionResult>;

  auto codegenBuiltinLine(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinFile(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinFunction(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeVal(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeValf(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeVall(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNans(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNansf(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNansl(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinAlloca(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinBzero(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinCtz(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinAddressof(CallExpressionAST* ast) -> ExpressionResult;
  auto emitMemberPointerFormation(UnaryExpressionAST* ast)
      -> std::optional<ExpressionResult>;
  auto emitMemberPointerAccess(BinaryExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinAssumeAligned(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinCountZerosGeneric(CallExpressionAST* ast)
      -> ExpressionResult;

  auto emitClassConstruction(ExpressionAST* ast, SourceLocation loc,
                             const Type* classType,
                             List<ExpressionAST*>* argList,
                             FunctionSymbol* constructorSymbol = nullptr)
      -> ExpressionResult;
};

struct Codegen::NewInitializerVisitor {
  Codegen& gen;

  auto operator()(NewParenInitializerAST* ast) -> NewInitializerResult;
  auto operator()(NewBracedInitializerAST* ast) -> NewInitializerResult;
};

auto Codegen::expression(ExpressionAST* ast, ExpressionFormat format)
    -> ExpressionResult {
  if (!ast) return {};

  if (format == ExpressionFormat::kSideEffect) {
    switch (ast->kind()) {
      case ASTKind::IdExpression:
      case ASTKind::ThisExpression:
      case ASTKind::BoolLiteralExpression:
      case ASTKind::CharLiteralExpression:
      case ASTKind::IntLiteralExpression:
      case ASTKind::FloatLiteralExpression:
      case ASTKind::NullptrLiteralExpression:
      case ASTKind::StringLiteralExpression:
        return {};
      default:
        break;
    }
  }

  auto result = visit(ExpressionVisitor{*this, format}, ast);
  result.category = ast->valueCategory;
  return result;
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
      auto conditionalEvaluation = ConditionalEvaluation{*this};
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }

    if (binop->op == TokenKind::T_BAR_BAR) {
      auto nextBlock = newBlock();
      condition(binop->leftExpression, trueBlock, nextBlock);
      builder_.setInsertionPointToEnd(nextBlock);
      auto conditionalEvaluation = ConditionalEvaluation{*this};
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }
  }

  const auto loc = getLocation(ast->firstSourceLocation());
  auto value = expression(ast);

  mlir::cf::CondBranchOp::create(builder_, loc, value.value, trueBlock,
                                 falseBlock);
}

void Codegen::conditionWithCleanups(ExpressionAST* ast, mlir::Block* trueBlock,
                                    mlir::Block* falseBlock) {
  if (!ast) return;

  const auto outerDepth = cleanupStack_.size();
  const auto endLoc = ast->lastSourceLocation();

  auto trueCleanupBlock = newBlock();
  auto falseCleanupBlock = newBlock();

  pushFullExpressionCleanup();
  condition(ast, trueCleanupBlock, falseCleanupBlock);

  builder_.setInsertionPointToEnd(trueCleanupBlock);
  emitBranchWithCleanups(endLoc, trueBlock, outerDepth);

  builder_.setInsertionPointToEnd(falseCleanupBlock);
  emitBranchWithCleanups(endLoc, falseBlock, outerDepth);

  popCleanup(endLoc);
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
  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type, gen.builder_.getIntegerAttr(type, value));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);

  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type,
      gen.builder_.getIntegerAttr(type, ast->isTrue ? 1 : 0));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);

  auto type = gen.convertType(ast->type);
  auto value = ast->literal->integerValue();

  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type, gen.builder_.getIntegerAttr(type, value));

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
      auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "unsupported float type");
      return {op};
  }

  auto op = mlir::arith::ConstantOp::create(gen.builder_, loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NullptrLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);
  auto context = gen.context_;
  auto resultType =
      mlir::cxx::PointerType::get(context, mlir::cxx::VoidType::get(context));
  auto op = mlir::cxx::NullPtrConstantOp::create(gen.builder_, loc, resultType);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->literalLoc);
  auto type = gen.convertType(ast->type);
  auto resultType = mlir::cxx::PointerType::get(type.getContext(), type);

  auto it = gen.stringLiterals_.find(ast->literal);
  if (it == gen.stringLiterals_.end()) {
    std::string str(ast->literal->stringValue());

    switch (ast->literal->encoding()) {
      case StringLiteralEncoding::kUtf16:
        str.push_back('\0');
        str.push_back('\0');
        break;
      case StringLiteralEncoding::kUtf32:
      case StringLiteralEncoding::kWide:
        str.push_back('\0');
        str.push_back('\0');
        str.push_back('\0');
        str.push_back('\0');
        break;
      default:
        str.push_back('\0');
        break;
    }

    auto initializer =
        gen.builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));

    auto name = gen.builder_.getStringAttr(gen.newUniqueSymbolName(".str"));

    auto x = mlir::OpBuilder(gen.module_->getContext());
    x.setInsertionPointToEnd(gen.module_.getBody());
    auto linkage = mlir::cxx::LinkageKindAttr::get(
        gen.context_, mlir::cxx::LinkageKind::Internal);
    mlir::cxx::GlobalOp::create(x, loc, mlir::TypeRange(), type, true,
                                name.getValue(), initializer, linkage,
                                mlir::IntegerAttr{});

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
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = ast->type;
  auto mlirType = gen.convertType(type);
  auto ptrType = mlir::cxx::PointerType::get(gen.context_, mlirType);
  auto allocaOp = mlir::cxx::AllocaOp::create(gen.builder_, loc, ptrType,
                                              gen.getAlignment(type));

  if (gen.unit_->typeTraits().is_array(type)) {
    gen.arrayInit(allocaOp, type, ast->bracedInitList);
  } else if (gen.unit_->typeTraits().is_class(type) && ast->bracedInitList) {
    ast->bracedInitList->type = type;
    gen.emitAggregateInit(allocaOp, type, ast->bracedInitList);
  } else if (ast->bracedInitList) {
    ExpressionAST* initExpr = nullptr;
    if (ast->bracedInitList->expressionList) {
      initExpr = ast->bracedInitList->expressionList->value;
    }
    if (initExpr) {
      auto initResult = gen.expression(initExpr);
      if (initResult.value) {
        mlir::cxx::StoreOp::create(gen.builder_, loc, initResult.value,
                                   allocaOp, gen.getAlignment(type));
      }
    } else {
      auto zero = mlir::arith::ConstantOp::create(
          gen.builder_, loc, mlirType, gen.builder_.getZeroAttr(mlirType));
      mlir::cxx::StoreOp::create(gen.builder_, loc, zero, allocaOp,
                                 gen.getAlignment(type));
    }
  }

  return {allocaOp};
}

auto Codegen::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  if (auto classSymbol = symbol_cast<ClassSymbol>(
          gen.currentFunctionSymbol_ ? gen.currentFunctionSymbol_->parent()
                                     : nullptr)) {
    if (auto capturedThisField = classSymbol->capturedThisField()) {
      auto access =
          emitThisFieldAddress(capturedThisField, ast->firstSourceLocation());
      if (access) {
        auto [fieldAddr, info] = *access;
        auto ptrType = gen.convertType(ast->type);
        auto loadOp = mlir::cxx::LoadOp::create(
            gen.builder_, loc, ptrType, fieldAddr,
            gen.getAlignment(capturedThisField->type()));
        return {loadOp};
      }
    }
  }

  auto ptrType = gen.convertType(ast->type);

  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, ptrType, gen.thisValue_, gen.getAlignment(ast->type));

  return {loadOp};
}

auto Codegen::ExpressionVisitor::operator()(PackIndexExpressionAST* ast)
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
  if (!ast->statement) {
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  std::vector<StatementAST*> stmts;
  for (auto node : ListView{ast->statement->statementList})
    stmts.push_back(node);

  ExpressionStatementAST* lastExprStmt = nullptr;
  if (!stmts.empty() && !gen.unit_->typeTraits().is_void(ast->type)) {
    if (auto last = ast_cast<ExpressionStatementAST>(stmts.back()))
      if (last->expression && last->expression->type) lastExprStmt = last;
  }

  gen.pushCleanup();

  auto count = lastExprStmt ? stmts.size() - 1 : stmts.size();
  for (std::size_t i = 0; i < count; ++i) gen.statement(stmts[i]);

  mlir::Value result;
  if (lastExprStmt)
    result = gen.expression(lastExprStmt->expression, ExpressionFormat::kValue)
                 .value;

  gen.popCleanup(ast->statement->rbraceLoc);

  if (result) return {result};
  return {};
}

auto Codegen::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  if (auto object = gen.takeResultObject(ast)) {
    (void)gen.emitPrvalueInto(object, ast->type, ast->expression,
                              ast->firstSourceLocation());
    return {object};
  }

  return gen.expression(ast->expression, format);
}

auto Codegen::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
    mlir::Value val;
    bool found = false;

    if (auto local = gen.findOrCreateLocal(var)) {
      val = local.value();
      found = true;
    } else if (auto global = gen.findOrCreateGlobal(var)) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = mlir::cxx::PointerType::get(
          gen.context_, gen.convertType(var->type()));
      val = mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType,
                                           global->getSymName());
      found = true;
    }

    if (found) {
      if (gen.unit_->typeTraits().is_reference(var->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(var->type());
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val,
                                        gen.getAlignment(var->type()));
      }
      return {val};
    }
  } else if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    if (auto local = gen.findOrCreateLocal(ast->symbol)) {
      auto val = local.value();
      if (gen.unit_->typeTraits().is_reference(param->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(param->type());
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val,
                                        gen.getAlignment(param->type()));
      }
      return {val};
    }
  } else if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (field->isStatic()) {
      if (auto def = field->definition()) {
        if (auto global = gen.findOrCreateGlobal(def)) {
          auto loc = gen.getLocation(ast->firstSourceLocation());
          auto resultType = mlir::cxx::PointerType::get(
              gen.context_, gen.convertType(def->type()));
          return {mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType,
                                                 global->getSymName())};
        }
      }

      if (!field->definition()) {
        auto global = gen.findOrCreateExternField(field);
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto resultType = mlir::cxx::PointerType::get(
            gen.context_, gen.convertType(field->type()));
        return {mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType,
                                               global.getSymName())};
      }
    }

    if (!field->isStatic()) {
      if (!gen.thisValue_) {
        auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                   "implicit use of 'this' but 'this' is null");
        return {op};
      }

      auto access = emitThisFieldAddress(field, ast->firstSourceLocation());
      if (!access) {
        return {gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "could not access member through 'this'")};
      }
      auto [op, info] = *access;

      if (info.bitWidth > 0) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        bool isSigned = gen.unit_->typeTraits().is_signed(field->type());
        auto loadOp = mlir::cxx::BitfieldLoadOp::create(
            gen.builder_, loc, gen.convertType(ast->type), op,
            gen.builder_.getI32IntegerAttr(info.bitOffset),
            gen.builder_.getI32IntegerAttr(info.bitWidth),
            gen.builder_.getI64IntegerAttr(info.allocUnitSizeBytes),
            gen.builder_.getBoolAttr(isSigned));
        return {loadOp, ValueCategory::kPrValue,
                /*isRValueMaterialized=*/true};
      }

      if (gen.unit_->typeTraits().is_reference(field->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(field->type());
        op = mlir::cxx::LoadOp::create(gen.builder_, loc, type, op,
                                       gen.getAlignment(field->type()));
      }
      return {op};
    }
  } else if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    if (enumerator->value().has_value()) {
      if (auto val = std::get_if<std::intmax_t>(&enumerator->value().value())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(enumerator->type());
        auto op = mlir::arith::ConstantOp::create(
            gen.builder_, loc, type, gen.builder_.getIntegerAttr(type, *val));
        return {op};
      }
    }
  }

  if (ast->symbol) {
    if (auto funcSymbol = symbol_cast<FunctionSymbol>(ast->symbol)) {
      auto funcOp = gen.findOrCreateFunction(funcSymbol);
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto type =
          gen.convertType(gen.control()->getPointerType(funcSymbol->type()));
      auto name = llvm::cast<mlir::StringAttr>(funcOp.getSymNameAttr());
      auto op = mlir::cxx::AddressOfOp::create(gen.builder_, loc, type, name);
      return {op};
    }

    auto op = gen.emitTodoExpr(
        ast->firstSourceLocation(),
        std::format("{}: did fail to generate MLIR code for symbol '{}'",
                    to_string(ast->kind()),
                    to_string(ast->symbol->type(), ast->symbol->name())));
    return {op};
  }

  auto name = get_name(control(), ast->unqualifiedId);

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(),
                       std::format("{}: did fail to resolve name '{}'",
                                   to_string(ast->kind()), to_string(name)));

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
  if (auto classType = type_cast<ClassType>(ast->type)) {
    auto classSymbol = classType->symbol();

    {
      auto savedIP = gen.builder_.saveInsertionPoint();

      gen.builder_.setInsertionPointToEnd(gen.module_.getBody());

      for (auto ctor : classSymbol->constructors()) {
        if (auto funcDecl = ctor->declaration()) {
          (void)gen.declaration(funcDecl);
        }
      }

      for (auto member : classSymbol->members()) {
        for (auto func : views::each_function(member)) {
          if (auto funcDecl = func->declaration()) {
            (void)gen.declaration(funcDecl);
          }
        }
      }

      gen.builder_.restoreInsertionPoint(savedIP);
    }

    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto mlirType = gen.convertType(classType);
    auto ptrType = mlir::cxx::PointerType::get(gen.context_, mlirType);
    auto closureAlloca = mlir::cxx::AllocaOp::create(
        gen.builder_, loc, ptrType, gen.getAlignment(classType));

    std::vector<ExpressionResult> captureArgs;
    for (auto captureNode : ListView{ast->captureList}) {
      ExpressionAST* initExpr = nullptr;
      if (auto simple = ast_cast<SimpleLambdaCaptureAST>(captureNode)) {
        initExpr = simple->initializer;
      } else if (auto ref = ast_cast<RefLambdaCaptureAST>(captureNode)) {
        initExpr = ref->initializer;
      } else if (auto th = ast_cast<ThisLambdaCaptureAST>(captureNode)) {
        initExpr = th->initializer;
      } else if (auto initCap = ast_cast<InitLambdaCaptureAST>(captureNode)) {
        initExpr = initCap->initializer;
      } else if (auto refInitCap =
                     ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
        initExpr = refInitCap->initializer;
      }

      if (!initExpr) continue;
      captureArgs.push_back(gen.expression(initExpr));
    }

    if (ast->constructorSymbol) {
      (void)gen.emitCtorCall(ast->firstSourceLocation(), ast->constructorSymbol,
                             closureAlloca, captureArgs,
                             /*completeObject=*/true);
    }

    return {closureAlloca};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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
  auto loc = gen.getLocation(ast->vaArgLoc);

  auto expressionResult = gen.expression(ast->expression);

  mlir::SmallVector<mlir::Value> arguments;
  arguments.push_back(expressionResult.value);

  mlir::SmallVector<mlir::Type> resultTypes;
  if (ast->type && !gen.unit_->typeTraits().is_void(ast->type)) {
    resultTypes.push_back(gen.convertType(ast->type));
  }

  auto op = mlir::cxx::BuiltinCallOp::create(gen.builder_, loc, resultTypes,
                                             "__builtin_va_arg", arguments);

  return {op.getResult()};
}

auto Codegen::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  if (ast->symbol) {
    auto baseExpressionResult = gen.expression(ast->baseExpression);
    auto indexExpressionResult = gen.expression(ast->indexExpression);
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->lbracketLoc, ast->symbol, baseExpressionResult,
                          {indexExpressionResult}, ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->lbracketLoc, ast->symbol, {},
                          {baseExpressionResult, indexExpressionResult});
    }
  }

  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto indexExpressionResult = gen.expression(ast->indexExpression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto resultType =
      gen.convertType(gen.unit_->typeTraits().add_pointer(ast->type));

  auto baseType = ast->baseExpression->type;
  mlir::Value index = indexExpressionResult.value;

  const Type* strideBaseType = nullptr;
  if (gen.unit_->typeTraits().is_pointer(baseType))
    strideBaseType = gen.unit_->typeTraits().get_element_type(baseType);
  else if (auto vla = type_cast<UnresolvedBoundedArrayType>(baseType))
    strideBaseType = vla->elementType();

  if (strideBaseType) {
    mlir::Value stride;
    const Type* cur = strideBaseType;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      auto countVal = countResult.value;
      if (mlir::isa<mlir::cxx::PointerType>(countVal.getType())) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal =
            mlir::cxx::LoadOp::create(gen.builder_, loc, valueType, countVal,
                                      gen.getAlignment(vla->size()->type));
      }
      if (countVal.getType() != index.getType())
        countVal = mlir::arith::ExtSIOp::create(gen.builder_, loc,
                                                index.getType(), countVal);
      stride = stride
                   ? mlir::arith::MulIOp::create(
                         gen.builder_, loc, index.getType(), stride, countVal)
                   : countVal;
      cur = vla->elementType();
    }
    if (stride)
      index = mlir::arith::MulIOp::create(gen.builder_, loc, index.getType(),
                                          index, stride);
  }

  if (gen.unit_->typeTraits().is_pointer(baseType) ||
      type_cast<UnresolvedBoundedArrayType>(baseType)) {
    auto op = mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType,
                                          baseExpressionResult.value, index);
    return {op};
  }

  auto indexType = ast->indexExpression->type;
  if (gen.unit_->typeTraits().is_pointer(indexType)) {
    auto op = mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType,
                                          indexExpressionResult.value,
                                          baseExpressionResult.value);
    return {op};
  }

  auto op = mlir::cxx::SubscriptOp::create(gen.builder_, loc, resultType,
                                           baseExpressionResult.value,
                                           indexExpressionResult.value);

  return {op};
}

auto Codegen::ExpressionVisitor::emitBuiltinCall(
    CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
    -> ExpressionResult {
  if (auto result = codegenBuiltinDispatch(ast, builtinKind)) {
    return *result;
  }

  auto loc = gen.getLocation(ast->lparenLoc);

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_UNREACHABLE) {
    mlir::cxx::UnreachableOp::create(gen.builder_, loc);
    return {};
  }

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_IS_CONSTANT_EVALUATED) {
    auto boolType = gen.convertType(control()->getBoolType());
    auto falseVal = mlir::arith::ConstantOp::create(
        gen.builder_, loc, boolType, gen.builder_.getIntegerAttr(boolType, 0));
    return {falseVal};
  }

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_CONSTANT_P) {
    auto intType = gen.convertType(control()->getIntType());
    int result = 0;
    auto args = ListView{ast->expressionList};
    auto it = args.begin();
    if (it != args.end()) {
      auto interp = ASTInterpreter{gen.unit_};
      result = interp.evaluate(*it).has_value() ? 1 : 0;
    }
    auto val = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intType,
        gen.builder_.getIntegerAttr(intType, result));
    return {val};
  }

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_EXPECT) {
    auto args = ListView{ast->expressionList};
    auto it = args.begin();
    if (it != args.end()) {
      return gen.expression(*it);
    }
    return {};
  }

  if (ast->constructorSymbol) {
    std::vector<ExpressionResult> callArgs;
    for (auto node : ListView{ast->expressionList})
      callArgs.push_back(gen.expression(node));
    return gen.emitCall(ast->lparenLoc, ast->constructorSymbol, {},
                        std::move(callArgs));
  }

  const auto& name = Token::spell(builtinKind);

  mlir::SmallVector<mlir::Value> arguments;
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  mlir::SmallVector<mlir::Type> resultTypes;
  if (ast->type && !gen.unit_->typeTraits().is_void(ast->type)) {
    resultTypes.push_back(gen.convertType(ast->type));
  }

  auto op = mlir::cxx::BuiltinCallOp::create(gen.builder_, loc, resultTypes,
                                             name, arguments);

  return {op.getResult()};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto func = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(func)) {
    func = nested->expression;
  }

  if (auto id = ast_cast<IdExpressionAST>(func)) {
    if (auto nameId = ast_cast<NameIdAST>(id->unqualifiedId)) {
      if (nameId->identifier) {
        auto builtinKind = nameId->identifier->builtinFunction();
        if (builtinKind != BuiltinFunctionKind::T_NONE) {
          return emitBuiltinCall(ast, builtinKind);
        }
      }
    }
  }

  auto id = ast_cast<IdExpressionAST>(func);
  auto member = ast_cast<MemberExpressionAST>(func);
  ExpressionResult thisValue;

  if (member && ast_cast<DestructorIdAST>(member->unqualifiedId) &&
      !symbol_cast<FunctionSymbol>(member->symbol)) {
    (void)gen.expression(member->baseExpression);
    return {};
  }

  FunctionSymbol* functionSymbol = nullptr;
  if (id) {
    if (auto classSym = symbol_cast<ClassSymbol>(id->symbol)) {
      return emitClassConstruction(ast, ast->lparenLoc, classSym->type(),
                                   ast->expressionList, ast->constructorSymbol);
    }
    if (functionSymbol = symbol_cast<FunctionSymbol>(id->symbol)) {
      if (functionSymbol->isImplicitObjectMemberFunction()) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto classSymbol = symbol_cast<ClassSymbol>(
            functionSymbol->enclosingNonTemplateParametersScope());

        auto currentClass = classSymbol;
        if (gen.currentFunctionSymbol_) {
          if (auto enclosing = symbol_cast<ClassSymbol>(
                  gen.currentFunctionSymbol_
                      ->enclosingNonTemplateParametersScope())) {
            currentClass = enclosing;
          }
        }

        auto loadedThis = gen.loadThisPointer(loc, currentClass);

        auto adjustedThis = gen.emitBaseClassAddress(loc, loadedThis,
                                                     currentClass, classSymbol);
        thisValue = {adjustedThis};
      }
    }

  } else if (member) {
    functionSymbol = symbol_cast<FunctionSymbol>(member->symbol);

    if (functionSymbol) {
      auto baseResult = gen.expression(member->baseExpression);

      if (!functionSymbol->isStatic()) {
        thisValue = baseResult;

        if (auto targetClass =
                symbol_cast<ClassSymbol>(functionSymbol->parent())) {
          auto baseType =
              gen.unit_->typeTraits().remove_cv(member->baseExpression->type);
          if (member->accessOp == TokenKind::T_MINUS_GREATER) {
            baseType = gen.unit_->typeTraits().remove_cv(
                gen.unit_->typeTraits().get_element_type(baseType));
          }
          if (auto fromClassType = type_cast<ClassType>(baseType)) {
            thisValue = {gen.emitBaseClassAddress(
                gen.getLocation(member->firstSourceLocation()), thisValue.value,
                fromClassType->symbol(), targetClass)};
          }
        }
      }
    }
  }

  const FunctionType* functionType = nullptr;

  if (functionSymbol) {
    functionType = type_cast<FunctionType>(functionSymbol->type());
  } else if (gen.unit_->typeTraits().is_pointer(ast->baseExpression->type)) {
    thisValue = gen.expression(ast->baseExpression);

    auto elementType =
        gen.unit_->typeTraits().get_element_type(ast->baseExpression->type);
    functionType = type_cast<cxx::FunctionType>(elementType);
  }

  if (!functionType) {
    auto op =
        gen.emitTodoExpr(ast->firstSourceLocation(), "invalid function call");

    return {op};
  }

  std::vector<ExpressionResult> callArguments;
  for (auto node : ListView{ast->expressionList}) {
    callArguments.push_back(gen.expression(node));
  }

  const bool isVirtualCall =
      ast->isVirtualDispatch && functionSymbol && thisValue.value;

  const bool returnsIndirectly =
      gen.classifyClassValueAbi(functionType->returnType()).kind ==
      ClassValueAbi::Kind::Indirect;

  return gen.emitCall(
      ast->lparenLoc, functionType, functionSymbol, isVirtualCall, thisValue,
      std::move(callArguments),
      returnsIndirectly ? gen.takeResultObject(ast) : mlir::Value{});
}

auto Codegen::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  const Type* targetType = ast->type;

  if (!targetType) {
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  auto loc = gen.getLocation(ast->firstSourceLocation());

  if (gen.unit_->typeTraits().is_class(targetType)) {
    return emitClassConstruction(ast, ast->firstSourceLocation(), targetType,
                                 ast->expressionList, ast->constructorSymbol);
  }

  auto resultType = gen.convertType(targetType);

  if (!ast->expressionList) {
    if (mlir::isa<mlir::IntegerType>(resultType)) {
      auto op = mlir::arith::ConstantOp::create(
          gen.builder_, loc, resultType,
          gen.builder_.getIntegerAttr(resultType, 0));
      return {op};
    }
    if (mlir::isa<mlir::FloatType>(resultType)) {
      auto op = mlir::arith::ConstantOp::create(
          gen.builder_, loc, resultType, gen.builder_.getZeroAttr(resultType));
      return {op};
    }
    if (auto ptrType = mlir::dyn_cast<mlir::cxx::PointerType>(resultType)) {
      return {mlir::cxx::NullPtrConstantOp::create(gen.builder_, loc, ptrType)};
    }
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  auto argResult = gen.expression(ast->expressionList->value);
  auto argType = argResult.value.getType();

  if (argType == resultType) {
    return argResult;
  }

  if (mlir::isa<mlir::IntegerType>(argType) &&
      mlir::isa<mlir::FloatType>(resultType)) {
    return {mlir::arith::SIToFPOp::create(gen.builder_, loc, resultType,
                                          argResult.value)};
  }

  if (mlir::isa<mlir::FloatType>(argType) &&
      mlir::isa<mlir::IntegerType>(resultType)) {
    return {mlir::arith::FPToSIOp::create(gen.builder_, loc, resultType,
                                          argResult.value)};
  }

  if (mlir::isa<mlir::FloatType>(argType) &&
      mlir::isa<mlir::FloatType>(resultType)) {
    if (argType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth())
      return {mlir::arith::ExtFOp::create(gen.builder_, loc, resultType,
                                          argResult.value)};
    return {mlir::arith::TruncFOp::create(gen.builder_, loc, resultType,
                                          argResult.value)};
  }

  if (mlir::isa<mlir::IntegerType>(argType) &&
      mlir::isa<mlir::IntegerType>(resultType)) {
    if (argType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth())
      return {mlir::arith::ExtSIOp::create(gen.builder_, loc, resultType,
                                           argResult.value)};
    return {mlir::arith::TruncIOp::create(gen.builder_, loc, resultType,
                                          argResult.value)};
  }

  return argResult;
}

auto Codegen::ExpressionVisitor::operator()(BracedTypeConstructionAST* ast)
    -> ExpressionResult {
  const Type* targetType = ast->type;

  if (!targetType) {
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  if (gen.unit_->typeTraits().is_class(targetType)) {
    if (ast->constructorSymbol) {
      return emitClassConstruction(
          ast, ast->firstSourceLocation(), targetType,
          ast->bracedInitList ? ast->bracedInitList->expressionList : nullptr,
          ast->constructorSymbol);
    }

    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto temp = gen.newTemp(targetType, ast->firstSourceLocation());
    if (ast->bracedInitList) {
      ast->bracedInitList->type = targetType;
      gen.emitAggregateInit(temp, targetType, ast->bracedInitList);
    }
    return {temp.getResult()};
  }

  if (ast->bracedInitList && !ast->bracedInitList->type) {
    ast->bracedInitList->type = targetType;
  }

  auto bracedInitListResult = gen.expression(ast->bracedInitList);
  return bracedInitListResult;
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

static auto isDerivedFrom(ClassSymbol* derived, ClassSymbol* base) -> bool {
  if (derived == base) return true;
  for (auto b : derived->baseClasses()) {
    auto bs = symbol_cast<ClassSymbol>(b->symbol());
    if (bs && isDerivedFrom(bs, base)) return true;
  }
  return false;
}

static auto isReachableViaAnonymous(ClassSymbol* from, ClassSymbol* target)
    -> bool {
  if (from == target) return true;
  for (auto member : from->members()) {
    auto nested = symbol_cast<ClassSymbol>(member);
    if (!nested || nested->name()) continue;
    if (isReachableViaAnonymous(nested, target)) return true;
  }
  return false;
}

static auto isReachableFrom(ClassSymbol* from, ClassSymbol* target) -> bool {
  if (from == target) return true;
  if (isReachableViaAnonymous(from, target)) return true;
  for (auto b : from->baseClasses()) {
    auto bs = symbol_cast<ClassSymbol>(b->symbol());
    if (bs && isReachableFrom(bs, target)) return true;
  }
  return false;
}

auto Codegen::navigateToClass(mlir::Location loc, mlir::Value value,
                              ClassSymbol* from, ClassSymbol* to)
    -> mlir::Value {
  if (from == to) return value;

  auto fromLayout = from->layout();

  if (fromLayout) {
    for (auto member : from->members()) {
      auto nested = symbol_cast<ClassSymbol>(member);
      if (!nested || nested->name()) continue;
      if (!isReachableViaAnonymous(nested, to)) continue;

      FieldSymbol* anonField = nullptr;
      for (auto m : from->members()) {
        auto f = symbol_cast<FieldSymbol>(m);
        if (!f) continue;
        if (auto ct = type_cast<ClassType>(f->type())) {
          if (ct->symbol() == nested) {
            anonField = f;
            break;
          }
        }
      }
      if (!anonField) continue;

      auto anonInfo = fromLayout->getFieldInfo(anonField);
      if (!anonInfo) continue;

      auto op = memberAddress(loc, value, nested->type(), anonInfo->index);
      return navigateToClass(loc, op, nested, to);
    }
  }

  for (auto base : from->baseClasses()) {
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) continue;

    if (!isReachableFrom(baseSym, to)) continue;

    if (base->isVirtual()) {
      auto op = emitVirtualBaseAddress(loc, value, from, baseSym);
      return navigateToClass(loc, op, baseSym, to);
    }

    std::uint32_t baseIndex = 0;
    if (fromLayout) {
      if (auto bi = fromLayout->getBaseInfo(baseSym)) {
        baseIndex = bi->index;
      }
    }

    auto op = memberAddress(loc, value, baseSym->type(), baseIndex);
    return navigateToClass(loc, op, baseSym, to);
  }

  return value;
}

auto Codegen::ExpressionVisitor::emitThisFieldAddress(FieldSymbol* field,
                                                      SourceLocation srcLoc)
    -> std::optional<std::pair<mlir::Value, ClassLayout::MemberInfo>> {
  if (!field || field->isStatic() || !gen.thisValue_) return std::nullopt;

  auto classSymbol = symbol_cast<ClassSymbol>(field->parent());
  if (!classSymbol) return std::nullopt;

  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;

  auto fieldInfo = layout->getFieldInfo(field);
  if (!fieldInfo) return std::nullopt;

  auto loc = gen.getLocation(srcLoc);

  auto currentClass = classSymbol;
  if (gen.currentFunctionSymbol_) {
    if (auto enclosing = symbol_cast<ClassSymbol>(
            gen.currentFunctionSymbol_
                ->enclosingNonTemplateParametersScope())) {
      currentClass = enclosing;
    }
  }

  auto thisPtr = gen.loadThisPointer(loc, currentClass);

  auto adjustedThis =
      gen.navigateToClass(loc, thisPtr, currentClass, classSymbol);

  auto op =
      gen.memberAddress(loc, adjustedThis, field->type(), fieldInfo->index);
  return std::pair{op, *fieldInfo};
}

auto Codegen::ExpressionVisitor::emitMemberAccess(MemberExpressionAST* ast)
    -> std::optional<std::pair<mlir::Value, ClassLayout::MemberInfo>> {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && !field->isStatic()) {
    auto baseExpressionResult = gen.expression(ast->baseExpression);

    auto baseType =
        gen.unit_->typeTraits().remove_cv(ast->baseExpression->type);

    if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
      baseType = gen.unit_->typeTraits().remove_cv(
          gen.unit_->typeTraits().get_element_type(baseType));
    }

    if (!mlir::isa<mlir::cxx::PointerType>(
            baseExpressionResult.value.getType())) {
      auto tempLoc =
          gen.getLocation(ast->baseExpression->firstSourceLocation());
      auto temp =
          gen.newTemp(baseType, ast->baseExpression->firstSourceLocation());
      mlir::cxx::StoreOp::create(gen.builder_, tempLoc,
                                 baseExpressionResult.value, temp,
                                 gen.getAlignment(baseType));
      baseExpressionResult = {temp};
    }

    auto classType = type_cast<ClassType>(baseType);

    if (!classType) {
      gen.emitTodoExpr(
          ast->firstSourceLocation(),
          std::format("base not class type '{}'", to_string(baseType)));
      return std::nullopt;
    }

    auto startClass = classType->symbol();
    auto fieldClass = symbol_cast<ClassSymbol>(field->parent());

    if (startClass != fieldClass) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      baseExpressionResult.value = gen.navigateToClass(
          loc, baseExpressionResult.value, startClass, fieldClass);
    }

    auto layout = fieldClass->layout();
    if (!layout) {
      gen.emitTodoExpr(ast->firstSourceLocation(), "class layout not computed");
      return std::nullopt;
    }

    auto fieldInfo = layout->getFieldInfo(field);
    if (!fieldInfo) {
      gen.emitTodoExpr(ast->firstSourceLocation(), "field not found in layout");
      return std::nullopt;
    }

    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = gen.memberAddress(loc, baseExpressionResult.value, field->type(),
                                fieldInfo->index);
    return std::pair{op, *fieldInfo};
  }
  return std::nullopt;
}

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  if (format == ExpressionFormat::kSideEffect) {
    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && field->isStatic()) {
      (void)gen.expression(ast->baseExpression, ExpressionFormat::kSideEffect);
      return {};
    }
  }

  if (auto access = emitMemberAccess(ast)) {
    auto [op, info] = *access;

    if (info.bitWidth > 0) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto fieldSym = symbol_cast<FieldSymbol>(ast->symbol);
      bool isSigned =
          fieldSym && gen.unit_->typeTraits().is_signed(fieldSym->type());
      auto loadOp = mlir::cxx::BitfieldLoadOp::create(
          gen.builder_, loc, gen.convertType(ast->type), op,
          gen.builder_.getI32IntegerAttr(info.bitOffset),
          gen.builder_.getI32IntegerAttr(info.bitWidth),
          gen.builder_.getI64IntegerAttr(info.allocUnitSizeBytes),
          gen.builder_.getBoolAttr(isSigned));
      return {loadOp, ValueCategory::kPrValue,
              /*isRValueMaterialized=*/true};
    }

    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && gen.unit_->typeTraits().is_reference(field->type())) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto type = gen.convertType(field->type());
      op = mlir::cxx::LoadOp::create(gen.builder_, loc, type, op,
                                     gen.getAlignment(field->type()));
    }

    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->baseExpression);

  if (ast->symbol) {
    auto loc = gen.getLocation(ast->opLoc);
    auto intTy = mlir::IntegerType::get(gen.context_, 32);
    auto zeroOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intTy, gen.builder_.getIntegerAttr(intTy, 0));
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {{zeroOp}},
                          ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {expressionResult, {zeroOp}});
    }
  }

  if (gen.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto elementTy = gen.convertType(ast->baseExpression->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, loc, elementTy, expressionResult.value,
        gen.getAlignment(ast->baseExpression->type));
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto oneOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultTy,
        gen.builder_.getIntegerAttr(
            resultTy, ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1));
    auto addOp =
        mlir::arith::AddIOp::create(gen.builder_, loc, resultTy, loadOp, oneOp);
    mlir::cxx::StoreOp::create(gen.builder_, loc, addOp, expressionResult.value,
                               gen.getAlignment(ast->baseExpression->type));
    return {loadOp};
  }
  if (gen.unit_->typeTraits().is_floating_point(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrTy =
        mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
    auto elementTy = ptrTy.getElementType();
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, loc, elementTy, expressionResult.value,
        gen.getAlignment(ast->baseExpression->type));
    auto resultTy = gen.convertType(ast->baseExpression->type);

    mlir::Value one;
    double v = ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1;

    switch (gen.unit_->typeTraits()
                .remove_cvref(ast->baseExpression->type)
                ->kind()) {
      case TypeKind::kFloat:
        one = mlir::arith::ConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF32FloatAttr(v));
        break;

      case TypeKind::kDouble:
        one = mlir::arith::ConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF64FloatAttr(v));
        break;

      case TypeKind::kLongDouble:
        one = mlir::arith::ConstantOp::create(
            gen.builder_, gen.getLocation(ast->opLoc),
            gen.convertType(ast->baseExpression->type),
            gen.builder_.getF64FloatAttr(v));
        break;

      default:
        auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                   "unsupported float type");
        return {op};
    }

    auto addOp =
        mlir::arith::AddFOp::create(gen.builder_, loc, resultTy, loadOp, one);
    mlir::cxx::StoreOp::create(gen.builder_, loc, addOp, expressionResult.value,
                               gen.getAlignment(ast->baseExpression->type));
    return {loadOp};
  }
  if (gen.unit_->typeTraits().is_pointer(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, loc, resultTy, expressionResult.value,
        gen.getAlignment(ast->baseExpression->type));
    auto intTy = mlir::IntegerType::get(gen.context_, 32);
    auto oneOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intTy,
        gen.builder_.getIntegerAttr(
            intTy, ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1));
    auto addOp =
        mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultTy, loadOp, oneOp);
    mlir::cxx::StoreOp::create(gen.builder_, loc, addOp, expressionResult.value,
                               gen.getAlignment(ast->baseExpression->type));
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
  auto expressionResult = gen.expression(ast->expression);
  return expressionResult;
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
  if (ast->symbol) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto resultType = gen.convertType(ast->type);

    auto classType = type_cast<ClassType>(
        gen.unit_->typeTraits().remove_cv(ast->typeId->type));
    if (!classType) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "__builtin_offsetof requires a class type")};
    }

    auto classSymbol = classType->symbol();
    auto layout = classSymbol->layout();
    if (!layout) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "class layout not computed")};
    }

    auto fieldInfo = layout->getFieldInfo(ast->symbol);
    if (!fieldInfo) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "field not found in layout")};
    }

    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultType,
        gen.builder_.getIntegerAttr(resultType, fieldInfo->offset));

    return {op};
  }

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

auto Codegen::ExpressionVisitor::emitUnaryOpNot(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (type_cast<BoolType>(gen.unit_->typeTraits().remove_cv(ast->type))) {
    auto loc = gen.getLocation(ast->opLoc);
    auto expressionResult = gen.expression(ast->expression);
    auto resultType = gen.convertType(ast->type);
    auto c1 = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultType,
        gen.builder_.getIntegerAttr(resultType, 1));
    auto op = mlir::arith::XOrIOp::create(gen.builder_, loc, resultType,
                                          expressionResult.value, c1);
    return {op};
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitUnaryOpMinus(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);
  auto loc = gen.getLocation(ast->opLoc);

  if (gen.unit_->typeTraits().is_floating_point(ast->type)) {
    auto op = mlir::arith::NegFOp::create(gen.builder_, loc, resultType,
                                          expressionResult.value);

    return {op};
  }

  if (gen.unit_->typeTraits().is_integral_or_unscoped_enum(ast->type)) {
    auto zero = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultType,
        gen.builder_.getIntegerAttr(resultType, 0));
    auto op = mlir::arith::SubIOp::create(gen.builder_, loc, resultType, zero,
                                          expressionResult.value);

    return {op};
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitUnaryOpTilde(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  auto loc = gen.getLocation(ast->opLoc);
  auto allOnes = mlir::arith::ConstantOp::create(
      gen.builder_, loc, resultType,
      gen.builder_.getIntegerAttr(resultType, -1));
  auto op = mlir::arith::XOrIOp::create(gen.builder_, loc, resultType,
                                        expressionResult.value, allOnes);

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrFloat(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  mlir::Value one;

  switch (gen.unit_->typeTraits().remove_cvref(ast->expression->type)->kind()) {
    case TypeKind::kFloat:
      one = mlir::arith::ConstantOp::create(
          gen.builder_, gen.getLocation(ast->opLoc),
          gen.convertType(ast->expression->type),
          gen.builder_.getF32FloatAttr(1.0));
      break;

    case TypeKind::kDouble:
      one = mlir::arith::ConstantOp::create(
          gen.builder_, gen.getLocation(ast->opLoc),
          gen.convertType(ast->expression->type),
          gen.builder_.getF64FloatAttr(1.0));
      break;

    case TypeKind::kLongDouble:
      one = mlir::arith::ConstantOp::create(
          gen.builder_, gen.getLocation(ast->opLoc),
          gen.convertType(ast->expression->type),
          gen.builder_.getF64FloatAttr(1.0));
      break;

    default:
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "unsupported float type")};
  }

  auto loc = gen.getLocation(ast->opLoc);
  auto resultType = gen.convertType(ast->type);

  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, resultType, expressionResult.value,
      gen.getAlignment(ast->expression->type));

  mlir::Value addOp;

  if (ast->op == TokenKind::T_MINUS_MINUS)
    addOp =
        mlir::arith::SubFOp::create(gen.builder_, loc, resultType, loadOp, one);
  else
    addOp =
        mlir::arith::AddFOp::create(gen.builder_, loc, resultType, loadOp, one);

  auto storeOp = mlir::cxx::StoreOp::create(
      gen.builder_, loc, addOp, expressionResult.value,
      gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                      expressionResult.value,
                                      gen.getAlignment(ast->expression->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrIntegral(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->opLoc);

  auto targetType = gen.convertType(ast->expression->type);
  auto oneOp = mlir::arith::ConstantOp::create(
      gen.builder_, loc, targetType,
      gen.builder_.getIntegerAttr(targetType, 1));

  auto resultType = gen.convertType(ast->type);

  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, resultType, expressionResult.value,
      gen.getAlignment(ast->expression->type));

  mlir::Value addOp;

  if (ast->op == TokenKind::T_MINUS_MINUS)
    addOp = mlir::arith::SubIOp::create(gen.builder_, loc, resultType, loadOp,
                                        oneOp);
  else
    addOp = mlir::arith::AddIOp::create(gen.builder_, loc, resultType, loadOp,
                                        oneOp);

  auto storeOp = mlir::cxx::StoreOp::create(
      gen.builder_, loc, addOp, expressionResult.value,
      gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                      expressionResult.value,
                                      gen.getAlignment(ast->expression->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrPointer(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto intTy = gen.builder_.getIntegerType(32);
  auto one = mlir::arith::ConstantOp::create(
      gen.builder_, loc, intTy,
      gen.builder_.getIntegerAttr(
          intTy, ast->op == TokenKind::T_MINUS_MINUS ? -1 : 1));
  auto resultType = gen.convertType(ast->expression->type);
  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, resultType, expressionResult.value,
      gen.getAlignment(ast->expression->type));
  auto addOp =
      mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType, loadOp, one);
  mlir::cxx::StoreOp::create(gen.builder_, loc, addOp, expressionResult.value,
                             gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                      expressionResult.value,
                                      gen.getAlignment(ast->expression->type));
  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecr(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);

  if (ast->symbol) {
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {},
                          ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult});
    }
  }

  if (gen.unit_->typeTraits().is_floating_point(ast->expression->type)) {
    return emitUnaryOpIncrDecrFloat(ast, expressionResult);
  }

  if (gen.unit_->typeTraits().is_arithmetic(ast->expression->type)) {
    return emitUnaryOpIncrDecrIntegral(ast, expressionResult);
  }

  if (gen.unit_->typeTraits().is_pointer(ast->expression->type)) {
    return emitUnaryOpIncrDecrPointer(ast, expressionResult);
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto ptrType = mlir::cast<mlir::cxx::PointerType>(gen.convertType(ast->type));
  auto funcNameAttr =
      gen.function_
          ? mlir::StringAttr::get(gen.context_, gen.function_.getSymName())
          : mlir::StringAttr{};
  auto op = mlir::cxx::LabelAddressOp::create(
      gen.builder_, loc, ptrType, ast->identifier->name(), mlir::IntegerAttr{},
      funcNameAttr);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (ast->op == TokenKind::T_MINUS_MINUS ||
      ast->op == TokenKind::T_PLUS_PLUS) {
    return emitUnaryOpIncrDecr(ast);
  }

  if (ast->symbol) {
    auto expressionResult = gen.expression(ast->expression);
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {},
                          ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult});
    }
  }

  switch (ast->op) {
    case TokenKind::T_EXCLAIM:
      return emitUnaryOpNot(ast);

    case TokenKind::T_PLUS:
      return {gen.expression(ast->expression).value};

    case TokenKind::T_MINUS:
      return emitUnaryOpMinus(ast);

    case TokenKind::T_TILDE:
      return emitUnaryOpTilde(ast);

    case TokenKind::T_AMP:
      if (auto memberPointer = emitMemberPointerFormation(ast)) {
        return *memberPointer;
      }
      [[fallthrough]];

    case TokenKind::T_STAR:
      return {gen.expression(ast->expression).value};

    default:
      break;
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
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
  if (auto size = ast->value) {
    auto resultlType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultlType,
        gen.builder_.getIntegerAttr(resultlType, size.value()));
    return {op};
  }

  if (ast->expression && ast->expression->type) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto resultType = gen.convertType(ast->type);
    mlir::Value totalElements;
    const Type* cur = ast->expression->type;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      if (!countResult.value) break;
      auto countVal = countResult.value;
      if (mlir::isa<mlir::cxx::PointerType>(countVal.getType())) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal =
            mlir::cxx::LoadOp::create(gen.builder_, loc, valueType, countVal,
                                      gen.getAlignment(vla->size()->type));
      }
      if (countVal.getType() != resultType)
        countVal = mlir::arith::ExtSIOp::create(gen.builder_, loc, resultType,
                                                countVal);
      totalElements = totalElements ? mlir::arith::MulIOp::create(
                                          gen.builder_, loc, resultType,
                                          totalElements, countVal)
                                    : countVal;
      cur = vla->elementType();
    }
    if (totalElements) {
      auto leafSize = static_cast<int64_t>(
          gen.control()->memoryLayout()->sizeOf(cur).value_or(1));
      if (leafSize > 1) {
        auto leafConst = mlir::arith::ConstantOp::create(
            gen.builder_, loc, resultType,
            gen.builder_.getIntegerAttr(resultType, leafSize));
        totalElements = mlir::arith::MulIOp::create(
            gen.builder_, loc, resultType, totalElements, leafConst);
      }
      return {totalElements};
    }
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  if (auto size = ast->value) {
    auto resultlType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultlType,
        gen.builder_.getIntegerAttr(resultlType, size.value()));
    return {op};
  }

  auto typeIdType = ast->typeId ? ast->typeId->type : nullptr;
  if (typeIdType) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto resultType = gen.convertType(ast->type);
    mlir::Value totalElements;
    const Type* cur = typeIdType;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      if (!countResult.value) break;
      auto countVal = countResult.value;
      if (mlir::isa<mlir::cxx::PointerType>(countVal.getType())) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal =
            mlir::cxx::LoadOp::create(gen.builder_, loc, valueType, countVal,
                                      gen.getAlignment(vla->size()->type));
      }
      if (countVal.getType() != resultType)
        countVal = mlir::arith::ExtSIOp::create(gen.builder_, loc, resultType,
                                                countVal);
      totalElements = totalElements ? mlir::arith::MulIOp::create(
                                          gen.builder_, loc, resultType,
                                          totalElements, countVal)
                                    : countVal;
      cur = vla->elementType();
    }
    if (totalElements) {
      auto leafSize = static_cast<int64_t>(
          gen.control()->memoryLayout()->sizeOf(cur).value_or(1));
      if (leafSize > 1) {
        auto leafConst = mlir::arith::ConstantOp::create(
            gen.builder_, loc, resultType,
            gen.builder_.getIntegerAttr(resultType, leafSize));
        totalElements = mlir::arith::MulIOp::create(
            gen.builder_, loc, resultType, totalElements, leafConst);
      }
      return {totalElements};
    }
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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
  if (ast->typeId && ast->typeId->type) {
    auto memoryLayout = control()->memoryLayout();
    auto alignment = memoryLayout->alignmentOf(ast->typeId->type).value();

    auto resultlType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultlType,
        gen.builder_.getIntegerAttr(resultlType, alignment));
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  if (ast->expression && ast->expression->type) {
    auto memoryLayout = control()->memoryLayout();
    auto alignment = memoryLayout->alignmentOf(ast->expression->type).value();
    auto resultlType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultlType,
        gen.builder_.getIntegerAttr(resultlType, alignment));
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  if (ast->value.has_value()) {
    auto resultType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultType,
        gen.builder_.getIntegerAttr(resultType, *ast->value ? 1 : 0));
    return {op};
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto objectType = ast->objectType;
  if (!objectType) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "new: missing objectType")};
  }

  auto memoryLayout = control()->memoryLayout();
  auto sizeOpt = memoryLayout->sizeOf(objectType);
  uint64_t objectSize = sizeOpt.value_or(1);

  auto sizeTy = gen.convertType(control()->getSizeType());

  auto sizeVal = mlir::arith::ConstantOp::create(
      gen.builder_, loc, sizeTy,
      gen.builder_.getIntegerAttr(sizeTy, objectSize));

  auto objectMlirType = gen.convertType(objectType);
  auto ptrType = mlir::cxx::PointerType::get(gen.context_, objectMlirType);

  mlir::Value rawPtr;

  if (ast->newPlacement) {
    mlir::SmallVector<mlir::Value> placementArgs;
    for (auto it = ast->newPlacement->expressionList; it; it = it->next) {
      auto arg = gen.expression(it->value);
      placementArgs.push_back(arg.value);
    }

    if (!placementArgs.empty()) {
      rawPtr = placementArgs[0];
    } else {
      rawPtr = sizeVal;
    }
  } else {
    auto operatorNewName = std::string("_Znwm");

    auto existingFunc =
        gen.module_.lookupSymbol<mlir::cxx::FuncOp>(operatorNewName);
    if (!existingFunc) {
      auto guard = mlir::OpBuilder::InsertionGuard(gen.builder_);
      gen.builder_.setInsertionPointToStart(gen.module_.getBody());

      mlir::SmallVector<mlir::Type> paramTypes{sizeTy};
      mlir::SmallVector<mlir::Type> resultTypes{ptrType};
      auto funcType =
          mlir::cxx::FunctionType::get(gen.context_, paramTypes, resultTypes,
                                       /*isVariadic=*/false);
      auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
          gen.context_, mlir::cxx::LinkageKind::External);
      auto inlineAttr = mlir::cxx::InlineKindAttr::get(
          gen.context_, mlir::cxx::InlineKind::NoInline);
      existingFunc = mlir::cxx::FuncOp::create(
          gen.builder_, loc, operatorNewName, funcType, linkageAttr, inlineAttr,
          mlir::cxx::VisibilityAttr{}, mlir::StringAttr{}, mlir::ArrayAttr{},
          mlir::ArrayAttr{});
    }

    mlir::SmallVector<mlir::Value> args{sizeVal};
    mlir::SmallVector<mlir::Type> resultTypes{ptrType};
    auto callOp = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                            existingFunc.getSymName(), args,
                                            mlir::TypeAttr{});
    rawPtr = callOp.getResult();
  }

  if (ast->constructorSymbol) {
    std::vector<ExpressionResult> ctorArgs;
    if (ast->newInitalizer) {
      if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
        for (auto it = paren->expressionList; it; it = it->next) {
          ctorArgs.push_back(gen.expression(it->value));
        }
      } else if (auto braced =
                     ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
        if (braced->bracedInitList) {
          auto bracedList = ast_cast<BracedInitListAST>(braced->bracedInitList);
          if (bracedList) {
            for (auto it = bracedList->expressionList; it; it = it->next) {
              ctorArgs.push_back(gen.expression(it->value));
            }
          }
        }
      }
    }
    (void)gen.emitCtorCall(ast->newLoc, ast->constructorSymbol, rawPtr,
                           ctorArgs, /*completeObject=*/true);
  } else if (ast->newInitalizer) {
    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      if (paren->expressionList) {
        auto initExpr = paren->expressionList->value;
        auto initVal = gen.expression(initExpr);
        auto val = initVal.value;
        if (initExpr->valueCategory == ValueCategory::kLValue) {
          auto loadedType = gen.convertType(initExpr->type);
          val = mlir::cxx::LoadOp::create(gen.builder_, loc, loadedType, val,
                                          gen.getAlignment(initExpr->type));
        }
        mlir::cxx::StoreOp::create(gen.builder_, loc, val, rawPtr,
                                   gen.getAlignment(objectType));
      }
    } else if (auto braced =
                   ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
      if (braced->bracedInitList) {
        auto initVal = gen.expression(braced->bracedInitList);
        mlir::cxx::StoreOp::create(gen.builder_, loc, initVal.value, rawPtr,
                                   gen.getAlignment(objectType));
      }
    }
  }

  return {rawPtr};
}

auto Codegen::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto ptrResult = gen.expression(ast->expression);
  if (!ptrResult.value) return {};

  auto ptrValue = ptrResult.value;

  if (ast->expression->valueCategory == ValueCategory::kLValue) {
    auto loadedType = gen.convertType(ast->expression->type);
    ptrValue =
        mlir::cxx::LoadOp::create(gen.builder_, loc, loadedType, ptrValue,
                                  gen.getAlignment(ast->expression->type));
  }

  const Type* exprType = ast->expression->type;
  const Type* pointeeType = nullptr;
  if (auto ptrTy =
          type_cast<PointerType>(gen.unit_->typeTraits().remove_cv(exprType))) {
    pointeeType = ptrTy->elementType();
  }

  if (pointeeType) {
    if (auto classType = type_cast<ClassType>(
            gen.unit_->typeTraits().remove_cv(pointeeType))) {
      auto classSymbol = classType->symbol();
      if (auto dtorSymbol = classSymbol->destructor()) {
        if (dtorSymbol->isVirtual()) {
          auto i8Type = gen.builder_.getI8Type();
          auto i8PtrType = mlir::cxx::PointerType::get(gen.context_, i8Type);
          auto i8PtrPtrType =
              mlir::cxx::PointerType::get(gen.context_, i8PtrType);

          auto vptrFieldPtr = gen.memberAddress(loc, ptrValue, i8PtrPtrType, 0);
          auto vtablePtr = mlir::cxx::LoadOp::create(
              gen.builder_, loc, i8PtrPtrType, vptrFieldPtr, 8);

          int slotIndex = gen.vtableSlotIndex(dtorSymbol) + 1;

          auto intTy = gen.convertType(control()->getIntType());
          auto offsetOp = mlir::arith::ConstantOp::create(
              gen.builder_, loc, intTy,
              gen.builder_.getIntegerAttr(intTy, slotIndex));
          auto funcPtrAddr = mlir::cxx::PtrAddOp::create(
              gen.builder_, loc, i8PtrPtrType, vtablePtr, offsetOp);
          auto funcPtr = mlir::cxx::LoadOp::create(gen.builder_, loc, i8PtrType,
                                                   funcPtrAddr, 8);

          mlir::SmallVector<mlir::Value> indirectArgs;
          indirectArgs.push_back(funcPtr);
          indirectArgs.push_back(ptrValue);
          mlir::SmallVector<mlir::Type> dtorResultTypes;
          mlir::cxx::CallOp::create(gen.builder_, loc, dtorResultTypes,
                                    mlir::FlatSymbolRefAttr{}, indirectArgs,
                                    mlir::TypeAttr{});

          return {};
        } else {
          (void)gen.emitCall(ast->deleteLoc,
                             Codegen::completeObjectDtor(dtorSymbol),
                             {ptrValue}, {});
        }
      }
    }
  }

  if (ast->symbol) {
    (void)gen.emitCall(ast->deleteLoc, ast->symbol, {}, {{ptrValue}});
  }

  return {};
}

auto Codegen::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  if (!expressionResult.value) return expressionResult;

  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto resultType = gen.convertType(ast->type);
  if (!resultType) return expressionResult;

  auto srcType = expressionResult.value.getType();

  if (mlir::isa<mlir::cxx::PointerType>(resultType) &&
      mlir::isa<mlir::IntegerType>(srcType)) {
    auto i64Type = gen.builder_.getI64Type();
    mlir::Value intVal = expressionResult.value;
    auto srcWidth = mlir::cast<mlir::IntegerType>(srcType).getWidth();
    if (srcWidth < 64) {
      if (gen.unit_->typeTraits().is_signed(ast->expression->type))
        intVal =
            mlir::arith::ExtSIOp::create(gen.builder_, loc, i64Type, intVal);
      else
        intVal =
            mlir::arith::ExtUIOp::create(gen.builder_, loc, i64Type, intVal);
    } else if (srcWidth > 64) {
      intVal =
          mlir::arith::TruncIOp::create(gen.builder_, loc, i64Type, intVal);
    }
    return {
        mlir::cxx::IntToPtrOp::create(gen.builder_, loc, resultType, intVal)};
  }

  if (mlir::isa<mlir::IntegerType>(resultType) &&
      mlir::isa<mlir::cxx::PointerType>(srcType)) {
    auto i64Type = gen.builder_.getI64Type();
    mlir::Value ptrInt = mlir::cxx::PtrToIntOp::create(
        gen.builder_, loc, i64Type, expressionResult.value);
    auto dstWidth = mlir::cast<mlir::IntegerType>(resultType).getWidth();
    if (dstWidth < 64)
      return {
          mlir::arith::TruncIOp::create(gen.builder_, loc, resultType, ptrInt)};
    if (dstWidth > 64)
      return {
          mlir::arith::ExtUIOp::create(gen.builder_, loc, resultType, ptrInt)};
    return {ptrInt};
  }

  if (mlir::isa<mlir::cxx::PointerType>(resultType) &&
      mlir::isa<mlir::cxx::PointerType>(srcType) && resultType != srcType) {
    return {mlir::cxx::BitcastOp::create(gen.builder_, loc, resultType,
                                         expressionResult.value)};
  }

  return expressionResult;
}

auto Codegen::ExpressionVisitor::emitLValueToRValueConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto expressionResult = gen.expression(ast->expression);

  if (gen.unit_->typeTraits().is_reference(ast->expression->type)) {
    return {expressionResult.value};
  }

  if (expressionResult.isRValueMaterialized) {
    return {expressionResult.value};
  }

  if (expressionResult.category != ValueCategory::kLValue &&
      expressionResult.category != ValueCategory::kXValue) {
    return {expressionResult.value};
  }

  if (!mlir::isa<mlir::cxx::PointerType>(expressionResult.value.getType())) {
    return {expressionResult.value};
  }

  auto resultType = gen.convertType(ast->type);

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                      expressionResult.value,
                                      gen.getAlignment(ast->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitNumericConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->castKind) {
    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kIntegralPromotion: {
      if (mlir::isa<mlir::cxx::PointerType>(expressionResult.value.getType())) {
        auto intVal = mlir::cxx::PtrToIntOp::create(
            gen.builder_, loc, resultType, expressionResult.value);
        return {intVal};
      }

      if (is_bool(ast->type)) {
        auto zero = mlir::arith::ConstantOp::create(
            gen.builder_, loc, expressionResult.value.getType(),
            gen.builder_.getIntegerAttr(expressionResult.value.getType(), 0));
        return {mlir::arith::CmpIOp::create(gen.builder_, loc,
                                            mlir::arith::CmpIPredicate::ne,
                                            expressionResult.value, zero)};
      }

      if (is_bool(ast->expression->type)) {
        return {mlir::arith::ExtUIOp::create(gen.builder_, loc, resultType,
                                             expressionResult.value)};
      }

      auto srcType =
          mlir::cast<mlir::IntegerType>(expressionResult.value.getType());

      auto dstType = mlir::cast<mlir::IntegerType>(resultType);

      if (srcType.getWidth() == dstType.getWidth()) {
        return expressionResult;
      }

      if (dstType.getWidth() < srcType.getWidth()) {
        return {mlir::arith::TruncIOp::create(gen.builder_, loc, resultType,
                                              expressionResult.value)};
      }

      if (gen.unit_->typeTraits().is_signed(ast->expression->type)) {
        return {mlir::arith::ExtSIOp::create(gen.builder_, loc, resultType,
                                             expressionResult.value)};
      }

      return {mlir::arith::ExtUIOp::create(gen.builder_, loc, resultType,
                                           expressionResult.value)};
    }

    case ImplicitCastKind::kFloatingPointPromotion:
    case ImplicitCastKind::kFloatingPointConversion: {
      auto srcWidth = expressionResult.value.getType().getIntOrFloatBitWidth();
      auto dstWidth = resultType.getIntOrFloatBitWidth();

      if (srcWidth == dstWidth) {
        return expressionResult;
      }

      if (srcWidth < dstWidth) {
        auto op = mlir::arith::ExtFOp::create(gen.builder_, loc, resultType,
                                              expressionResult.value);
        return {op};
      }

      auto op = mlir::arith::TruncFOp::create(gen.builder_, loc, resultType,
                                              expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kFloatingIntegralConversion:
      if (is_bool(ast->type)) {
        auto zero = mlir::arith::ConstantOp::create(
            gen.builder_, loc, expressionResult.value.getType(),
            gen.builder_.getZeroAttr(expressionResult.value.getType()));

        auto op = mlir::arith::CmpFOp::create(gen.builder_, loc,
                                              mlir::arith::CmpFPredicate::UNE,
                                              expressionResult.value, zero);

        return {op};
      }

      if (gen.unit_->typeTraits().is_floating_point(ast->type)) {
        if (gen.unit_->typeTraits().is_signed(ast->expression->type)) {
          auto op = mlir::arith::SIToFPOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
          return {op};
        }

        auto op = mlir::arith::UIToFPOp::create(gen.builder_, loc, resultType,
                                                expressionResult.value);

        return {op};
      }

      if (gen.unit_->typeTraits().is_integral(ast->type)) {
        if (gen.unit_->typeTraits().is_signed(ast->type)) {
          auto op = mlir::arith::FPToSIOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
          return {op};
        }

        auto op = mlir::arith::FPToUIOp::create(gen.builder_, loc, resultType,
                                                expressionResult.value);

        return {op};
      }
      break;

    default:
      break;
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitPointerConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->castKind) {
    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kFunctionPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
      return expressionResult;

    case ImplicitCastKind::kPointerConversion: {
      if (expressionResult.value &&
          mlir::isa<mlir::IntegerType>(expressionResult.value.getType())) {
        auto op =
            mlir::cxx::NullPtrConstantOp::create(gen.builder_, loc, resultType);

        return {op};
      }

      if (expressionResult.value &&
          expressionResult.value.getType() != resultType) {
        auto value = mlir::cxx::BitcastOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
        return {value};
      }

      return expressionResult;
    }

    case ImplicitCastKind::kArrayToPointerConversion: {
      auto op = mlir::cxx::ArrayToPointerOp::create(
          gen.builder_, loc, resultType, expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kBooleanConversion: {
      if (!gen.unit_->typeTraits().is_pointer(ast->expression->type)) break;
      auto ptrIntTy = gen.builder_.getI64Type();
      auto ptrInt = mlir::cxx::PtrToIntOp::create(gen.builder_, loc, ptrIntTy,
                                                  expressionResult.value);
      auto zero = mlir::arith::ConstantOp::create(
          gen.builder_, loc, ptrIntTy,
          gen.builder_.getIntegerAttr(ptrIntTy, 0));

      auto op = mlir::arith::CmpIOp::create(
          gen.builder_, loc, mlir::arith::CmpIPredicate::ne, ptrInt, zero);

      return {op};
    }

    default:
      break;
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::emitDerivedToBaseConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto expressionResult = gen.expression(ast->expression);
  if (!expressionResult.value) return expressionResult;

  auto traits = gen.unit_->typeTraits();
  auto sourceClass = type_cast<ClassType>(traits.remove_cv(
      traits.remove_pointer(traits.remove_reference(ast->expression->type))));
  auto targetClass = type_cast<ClassType>(traits.remove_cv(
      traits.remove_pointer(traits.remove_reference(ast->type))));

  mlir::Value value = expressionResult.value;
  if (sourceClass && targetClass &&
      sourceClass->symbol() != targetClass->symbol()) {
    value = gen.emitBaseClassAddress(loc, value, sourceClass->symbol(),
                                     targetClass->symbol());
  }

  if (traits.is_pointer(ast->type)) {
    auto resultType = gen.convertType(ast->type);
    if (value.getType() != resultType) {
      value =
          mlir::cxx::BitcastOp::create(gen.builder_, loc, resultType, value);
    }
  }

  return {value};
}

auto Codegen::ExpressionVisitor::emitUserDefinedConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto function = ast->conversionFunction;
  if (!function) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "unresolved user-defined conversion")};
  }

  if (function->isConstructor()) {
    auto paren = ast_cast<ParenInitializerAST>(ast->expression);
    if (!paren) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "user-defined conversion: no argument list")};
    }
    return emitClassConstruction(ast, ast->firstSourceLocation(), ast->type,
                                 paren->expressionList, function);
  }

  auto exprResult = gen.expression(ast->expression);
  auto objectValue = exprResult.value;
  if (!objectValue) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "invalid object of user-defined conversion")};
  }

  if (!mlir::isa<mlir::cxx::PointerType>(objectValue.getType())) {
    auto sourceType = ast->expression->type;
    auto temp = gen.newTemp(sourceType, ast->firstSourceLocation());
    mlir::cxx::StoreOp::create(gen.builder_, loc, objectValue, temp.getResult(),
                               gen.getAlignment(sourceType));
    objectValue = temp.getResult();
  }

  return gen.emitCall(ast->firstSourceLocation(), function, {objectValue}, {},
                      ast->isVirtualDispatch);
}

auto Codegen::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  if (ast->constValue) {
    (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
    if (auto cst = gen.emitConstInitValue(gen.builder_, loc, ast->type,
                                          *ast->constValue)) {
      return {cst};
    }
  }

  switch (ast->castKind) {
    case ImplicitCastKind::kIdentity:
      return gen.expression(ast->expression);

    case ImplicitCastKind::kLValueToRValueConversion:
      return emitLValueToRValueConversion(ast);

    case ImplicitCastKind::kIntegralPromotion:
    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kFloatingPointPromotion:
    case ImplicitCastKind::kFloatingPointConversion:
    case ImplicitCastKind::kFloatingIntegralConversion:
      return emitNumericConversion(ast);

    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kFunctionPointerConversion:
    case ImplicitCastKind::kArrayToPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
    case ImplicitCastKind::kPointerConversion:
    case ImplicitCastKind::kBooleanConversion:
      return emitPointerConversion(ast);

    case ImplicitCastKind::kDerivedToBaseConversion:
      return emitDerivedToBaseConversion(ast);

    case ImplicitCastKind::kUserDefinedConversion:
      return emitUserDefinedConversion(ast);

    case ImplicitCastKind::kTemporaryMaterializationConversion: {
      auto inner = gen.expression(ast->expression);
      if (!inner.value) break;
      if (mlir::isa<mlir::cxx::PointerType>(inner.value.getType())) {
        return inner;
      }
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto temp = gen.newTemp(ast->type, ast->firstSourceLocation());
      mlir::cxx::StoreOp::create(gen.builder_, loc, inner.value, temp,
                                 gen.getAlignment(ast->type));
      return {temp.getResult()};
    }

    default:
      break;
  }

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
    auto object = gen.takeResultObject(ast);
    (void)gen.expression(ast->leftExpression, ExpressionFormat::kSideEffect);
    if (object) {
      (void)gen.emitPrvalueInto(object, ast->type, ast->rightExpression,
                                ast->opLoc);
      return {object};
    }
    return gen.expression(ast->rightExpression, format);
  }

  if (ast->op == TokenKind::T_DOT_STAR ||
      ast->op == TokenKind::T_MINUS_GREATER_STAR) {
    return emitMemberPointerAccess(ast);
  }

  if (ast->op == TokenKind::T_BAR_BAR) {
    auto t = gen.newTemp(control()->getBoolType(), ast->opLoc);

    auto trueBlock = gen.newBlock();
    auto continueBlock = gen.newBlock();
    auto falseBlock = gen.newBlock();
    auto endBlock = gen.newBlock();

    gen.condition(ast->leftExpression, trueBlock, continueBlock);

    gen.builder_.setInsertionPointToEnd(continueBlock);
    {
      auto conditionalEvaluation = ConditionalEvaluation{gen};
      gen.condition(ast->rightExpression, trueBlock, falseBlock);
    }

    gen.builder_.setInsertionPointToEnd(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 1));

    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               trueValue, t,
                               gen.getAlignment(control()->getBoolType()));

    auto endLoc = gen.getLocation(ast->lastSourceLocation());
    gen.branch(endLoc, endBlock);

    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 0));
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t,
                               gen.getAlignment(control()->getBoolType()));
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    gen.builder_.setInsertionPointToEnd(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), resultType, t,
        gen.getAlignment(control()->getBoolType()));
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
    {
      auto conditionalEvaluation = ConditionalEvaluation{gen};
      gen.condition(ast->rightExpression, trueBlock, falseBlock);
    }

    gen.builder_.setInsertionPointToEnd(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 1));

    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               trueValue, t,
                               gen.getAlignment(control()->getBoolType()));

    auto endLoc = gen.getLocation(ast->lastSourceLocation());
    gen.branch(endLoc, endBlock);

    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 0));
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t,
                               gen.getAlignment(control()->getBoolType()));
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    gen.builder_.setInsertionPointToEnd(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), resultType, t,
        gen.getAlignment(control()->getBoolType()));
    return {loadOp};
  }

  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);

  if (ast->symbol) {
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, leftExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {leftExpressionResult, rightExpressionResult});
    }
  }

  auto resultType = gen.convertType(ast->type);

  return binaryExpression(ast->opLoc, ast->op, resultType, ast->leftExpression,
                          ast->rightExpression, leftExpressionResult,
                          rightExpressionResult);
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpFloat(
    SourceLocation loc, TokenKind binop, mlir::Type resultType,
    mlir::Value left, mlir::Value right) -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  switch (binop) {
    case TokenKind::T_PLUS: {
      auto op = mlir::arith::AddFOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_MINUS: {
      auto op = mlir::arith::SubFOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_STAR: {
      auto op = mlir::arith::MulFOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_SLASH: {
      auto op = mlir::arith::DivFOp::create(gen.builder_, mlirLoc, resultType,

                                            left, right);
      return {op};
    }

    default:
      break;
  }

  auto op = gen.emitTodoExpr(loc, "float arithmetic operator");

  return {op};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpIntegral(
    SourceLocation loc, TokenKind binop, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  bool isSigned = gen.unit_->typeTraits().is_signed(leftType);
  switch (binop) {
    case TokenKind::T_PLUS: {
      auto op = mlir::arith::AddIOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_MINUS: {
      auto op = mlir::arith::SubIOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_STAR: {
      auto op = mlir::arith::MulIOp::create(gen.builder_, mlirLoc, resultType,
                                            left, right);
      return {op};
    }

    case TokenKind::T_SLASH: {
      if (isSigned) {
        auto op = mlir::arith::DivSIOp::create(gen.builder_, mlirLoc,
                                               resultType, left, right);
        return {op};
      }

      auto op = mlir::arith::DivUIOp::create(gen.builder_, mlirLoc, resultType,
                                             left, right);
      return {op};
    }

    case TokenKind::T_PERCENT: {
      if (isSigned) {
        auto op = mlir::arith::RemSIOp::create(gen.builder_, mlirLoc,
                                               resultType, left, right);
        return {op};
      }

      auto op = mlir::arith::RemUIOp::create(gen.builder_, mlirLoc, resultType,
                                             left, right);
      return {op};
    }

    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "integral arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpPointer(
    SourceLocation loc, TokenKind op, mlir::Type resultType, mlir::Value left,
    mlir::Value right) -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  switch (op) {
    case TokenKind::T_PLUS: {
      auto base = left;
      auto offset = right;
      if (!mlir::isa<mlir::cxx::PointerType>(left.getType())) {
        std::swap(base, offset);
      }
      return {mlir::cxx::PtrAddOp::create(gen.builder_, mlirLoc, resultType,
                                          base, offset)};
    }
    case TokenKind::T_MINUS: {
      if (mlir::isa<mlir::cxx::PointerType>(right.getType())) {
        return {mlir::cxx::PtrDiffOp::create(
            gen.builder_, mlirLoc, gen.convertType(control()->getLongIntType()),
            left, right)};
      }
      auto offsetType = right.getType();
      auto zero = mlir::arith::ConstantOp::create(
          gen.builder_, mlirLoc, offsetType,
          gen.builder_.getIntegerAttr(offsetType, 0));
      auto offset = mlir::arith::SubIOp::create(gen.builder_, mlirLoc,
                                                offsetType, zero, right);
      return {mlir::cxx::PtrAddOp::create(gen.builder_, mlirLoc, resultType,
                                          left, offset)};
    }
    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "pointer arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOp(
    SourceLocation loc, TokenKind op, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  if (gen.unit_->typeTraits().is_floating_point(leftType)) {
    return emitBinaryArithmeticOpFloat(loc, op, resultType, left, right);
  }

  if (gen.unit_->typeTraits().is_integral(leftType)) {
    return emitBinaryArithmeticOpIntegral(loc, op, resultType, leftType, left,
                                          right);
  }

  return {gen.emitTodoExpr(loc, "arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryShiftOp(
    SourceLocation opLoc, TokenKind binOp, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  auto loc = gen.getLocation(opLoc);

  if (right.getType() != resultType) {
    if (auto rightInt = mlir::dyn_cast<mlir::IntegerType>(right.getType())) {
      if (auto resInt = mlir::dyn_cast<mlir::IntegerType>(resultType)) {
        if (rightInt.getWidth() > resInt.getWidth())
          right = mlir::arith::TruncIOp::create(gen.builder_, loc, resultType,
                                                right);
        else
          right = mlir::arith::ExtUIOp::create(gen.builder_, loc, resultType,
                                               right);
      }
    }
  }

  if (binOp == TokenKind::T_LESS_LESS) {
    return {mlir::arith::ShLIOp::create(gen.builder_, loc, resultType, left,
                                        right)};
  }

  if (gen.unit_->typeTraits().is_signed(leftType)) {
    return {mlir::arith::ShRSIOp::create(gen.builder_, loc, resultType, left,
                                         right)};
  }
  return {
      mlir::arith::ShRUIOp::create(gen.builder_, loc, resultType, left, right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpFloat(
    SourceLocation loc, TokenKind op, mlir::Type resultType, mlir::Value left,
    mlir::Value right) -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  mlir::arith::CmpFPredicate pred;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      pred = mlir::arith::CmpFPredicate::OEQ;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      pred = mlir::arith::CmpFPredicate::ONE;
      break;
    case TokenKind::T_LESS:
      pred = mlir::arith::CmpFPredicate::OLT;
      break;
    case TokenKind::T_LESS_EQUAL:
      pred = mlir::arith::CmpFPredicate::OLE;
      break;
    case TokenKind::T_GREATER:
      pred = mlir::arith::CmpFPredicate::OGT;
      break;
    case TokenKind::T_GREATER_EQUAL:
      pred = mlir::arith::CmpFPredicate::OGE;
      break;
    default:
      return {gen.emitTodoExpr(loc, "float comparison operator")};
  }
  return {
      mlir::arith::CmpFOp::create(gen.builder_, mlirLoc, pred, left, right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpIntegral(
    SourceLocation loc, TokenKind op, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  bool isSigned = gen.unit_->typeTraits().is_signed(leftType);
  mlir::arith::CmpIPredicate pred;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      pred = mlir::arith::CmpIPredicate::eq;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      pred = mlir::arith::CmpIPredicate::ne;
      break;
    case TokenKind::T_LESS:
      pred = isSigned ? mlir::arith::CmpIPredicate::slt
                      : mlir::arith::CmpIPredicate::ult;
      break;
    case TokenKind::T_LESS_EQUAL:
      pred = isSigned ? mlir::arith::CmpIPredicate::sle
                      : mlir::arith::CmpIPredicate::ule;
      break;
    case TokenKind::T_GREATER:
      pred = isSigned ? mlir::arith::CmpIPredicate::sgt
                      : mlir::arith::CmpIPredicate::ugt;
      break;
    case TokenKind::T_GREATER_EQUAL:
      pred = isSigned ? mlir::arith::CmpIPredicate::sge
                      : mlir::arith::CmpIPredicate::uge;
      break;
    default:
      return {gen.emitTodoExpr(loc, "integral comparison operator")};
  }
  return {
      mlir::arith::CmpIOp::create(gen.builder_, mlirLoc, pred, left, right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpPointer(
    SourceLocation loc, TokenKind op, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  auto intPtrType = gen.builder_.getIntegerType(64);
  auto leftInt =
      mlir::cxx::PtrToIntOp::create(gen.builder_, mlirLoc, intPtrType, left);
  auto rightInt =
      mlir::cxx::PtrToIntOp::create(gen.builder_, mlirLoc, intPtrType, right);
  mlir::arith::CmpIPredicate pred;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      pred = mlir::arith::CmpIPredicate::eq;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      pred = mlir::arith::CmpIPredicate::ne;
      break;
    case TokenKind::T_LESS:
      pred = mlir::arith::CmpIPredicate::ult;
      break;
    case TokenKind::T_LESS_EQUAL:
      pred = mlir::arith::CmpIPredicate::ule;
      break;
    case TokenKind::T_GREATER:
      pred = mlir::arith::CmpIPredicate::ugt;
      break;
    case TokenKind::T_GREATER_EQUAL:
      pred = mlir::arith::CmpIPredicate::uge;
      break;
    default:
      return {gen.emitTodoExpr(loc, "pointer comparison operator")};
  }
  return {mlir::arith::CmpIOp::create(gen.builder_, mlirLoc, pred, leftInt,
                                      rightInt)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOp(
    SourceLocation loc, TokenKind op, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  if (gen.unit_->typeTraits().is_floating_point(leftType)) {
    return emitBinaryComparisonOpFloat(loc, op, resultType, left, right);
  }

  if (gen.unit_->typeTraits().is_integral_or_unscoped_enum(leftType) ||
      gen.unit_->typeTraits().is_pointer(leftType) ||
      gen.unit_->typeTraits().is_null_pointer(leftType)) {
    if (gen.unit_->typeTraits().is_pointer(leftType)) {
      return emitBinaryComparisonOpPointer(loc, op, resultType, leftType, left,
                                           right);
    }
    return emitBinaryComparisonOpIntegral(loc, op, resultType, leftType, left,
                                          right);
  }

  return {gen.emitTodoExpr(loc, "comparison operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryBitwiseOp(
    SourceLocation loc, TokenKind op, mlir::Type resultType, mlir::Value left,
    mlir::Value right) -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  switch (op) {
    case TokenKind::T_CARET:
      return {mlir::arith::XOrIOp::create(gen.builder_, mlirLoc, resultType,
                                          left, right)};
    case TokenKind::T_AMP:
      return {mlir::arith::AndIOp::create(gen.builder_, mlirLoc, resultType,
                                          left, right)};
    case TokenKind::T_BAR:
      return {mlir::arith::OrIOp::create(gen.builder_, mlirLoc, resultType,
                                         left, right)};
    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "bitwise operator")};
}

auto Codegen::ExpressionVisitor::binaryExpression(
    SourceLocation opLoc, TokenKind op, mlir::Type resultType,
    ExpressionAST* leftExpression, ExpressionAST* rightExpression,
    ExpressionResult leftExpressionResult,
    ExpressionResult rightExpressionResult) -> ExpressionResult {
  switch (op) {
    case TokenKind::T_PLUS:
      if (gen.unit_->typeTraits().is_pointer(leftExpression->type) ||
          gen.unit_->typeTraits().is_pointer(rightExpression->type)) {
        return emitBinaryArithmeticOpPointer(opLoc, op, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
      }
      return emitBinaryArithmeticOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_MINUS:
      if (gen.unit_->typeTraits().is_pointer(leftExpression->type)) {
        return emitBinaryArithmeticOpPointer(opLoc, op, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
      }

      return emitBinaryArithmeticOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PERCENT:
      return emitBinaryArithmeticOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_LESS_LESS:
    case TokenKind::T_GREATER_GREATER:
      return emitBinaryShiftOp(opLoc, op, resultType, leftExpression->type,
                               leftExpressionResult.value,
                               rightExpressionResult.value);

    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER:
    case TokenKind::T_GREATER_EQUAL:
      return emitBinaryComparisonOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_CARET:
    case TokenKind::T_AMP:
    case TokenKind::T_BAR:
      return emitBinaryBitwiseOp(opLoc, op, resultType,
                                 leftExpressionResult.value,
                                 rightExpressionResult.value);

    default:
      break;
  }

  return {gen.emitTodoExpr(opLoc, to_string(BinaryExpressionAST::Kind))};
}

auto Codegen::ExpressionVisitor::operator()(ConditionalExpressionAST* ast)
    -> ExpressionResult {
  auto trueBlock = gen.newBlock();
  auto falseBlock = gen.newBlock();
  auto endBlock = gen.newBlock();

  const bool isVoid = gen.unit_->typeTraits().is_void(ast->type);

  const bool sharesResultObject =
      !isVoid && ast->valueCategory == ValueCategory::kPrValue &&
      gen.unit_->typeTraits().is_class(ast->type);

  auto object = sharesResultObject ? gen.takeResultObject(ast) : mlir::Value{};

  mlir::Value t;
  const Type* type = nullptr;
  if (!isVoid) {
    type = ast->type;
    if (ast->valueCategory != ValueCategory::kPrValue) {
      type = control()->getPointerType(type);
    }
    t = object ? object : gen.newTemp(type, ast->questionLoc).getResult();
  }

  gen.condition(ast->condition, trueBlock, falseBlock);

  auto endLoc = gen.getLocation(ast->lastSourceLocation());

  if (isVoid) {
    gen.builder_.setInsertionPointToEnd(trueBlock);
    gen.expression(ast->iftrueExpression);
    gen.branch(endLoc, endBlock);

    gen.builder_.setInsertionPointToEnd(falseBlock);
    gen.expression(ast->iffalseExpression);
    gen.branch(endLoc, endBlock);

    gen.builder_.setInsertionPointToEnd(endBlock);
    return {};
  }

  if (sharesResultObject && !object) gen.addTemporaryCleanup(t, type);

  auto emitArm = [&](ExpressionAST* arm, mlir::Block* armBlock,
                     SourceLocation storeLoc) {
    gen.builder_.setInsertionPointToEnd(armBlock);
    auto conditionalEvaluation = ConditionalEvaluation{gen};

    if (sharesResultObject) {
      (void)gen.emitPrvalueInto(t, type, arm, storeLoc);
    } else {
      auto armResult = gen.expression(arm);
      mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(storeLoc),
                                 armResult.value, t, gen.getAlignment(type));
    }

    gen.branch(endLoc, endBlock);
  };

  emitArm(ast->iftrueExpression, trueBlock, ast->questionLoc);
  emitArm(ast->iffalseExpression, falseBlock, ast->colonLoc);

  gen.builder_.setInsertionPointToEnd(endBlock);

  if (format == ExpressionFormat::kSideEffect) return {};

  if (sharesResultObject) return {t};

  auto resultType = gen.convertType(type);
  auto loadOp =
      mlir::cxx::LoadOp::create(gen.builder_, gen.getLocation(ast->colonLoc),
                                resultType, t, gen.getAlignment(type));
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
  if (ast->symbol) {
    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, leftExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {leftExpressionResult, rightExpressionResult});
    }
  }

  if (ast->op == TokenKind::T_EQUAL) {
    if (auto idExpr = ast_cast<IdExpressionAST>(ast->leftExpression)) {
      if (auto field = symbol_cast<FieldSymbol>(idExpr->symbol);
          field && field->isBitField() && !field->isStatic()) {
        if (auto access =
                emitThisFieldAddress(field, idExpr->firstSourceLocation())) {
          auto [addr, info] = *access;
          auto rhs = gen.expression(ast->rightExpression);
          auto loc = gen.getLocation(ast->firstSourceLocation());

          mlir::cxx::BitfieldStoreOp::create(
              gen.builder_, loc, rhs.value, addr,
              gen.builder_.getI32IntegerAttr(info.bitOffset),
              gen.builder_.getI32IntegerAttr(info.bitWidth),
              gen.builder_.getI64IntegerAttr(info.allocUnitSizeBytes));
          return rhs;
        }
      }
    }

    if (auto member = ast_cast<MemberExpressionAST>(ast->leftExpression)) {
      if (auto field = symbol_cast<FieldSymbol>(member->symbol);
          field && field->isBitField()) {
        if (auto access = emitMemberAccess(member)) {
          auto [addr, info] = *access;
          auto rhs = gen.expression(ast->rightExpression);
          auto loc = gen.getLocation(ast->firstSourceLocation());

          mlir::cxx::BitfieldStoreOp::create(
              gen.builder_, loc, rhs.value, addr,
              gen.builder_.getI32IntegerAttr(info.bitOffset),
              gen.builder_.getI32IntegerAttr(info.bitWidth),
              gen.builder_.getI64IntegerAttr(info.allocUnitSizeBytes));
          return rhs;
        }
      }
    }

    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);

    const auto loc = gen.getLocation(ast->opLoc);

    mlir::cxx::StoreOp::create(gen.builder_, loc, rightExpressionResult.value,
                               leftExpressionResult.value,
                               gen.getAlignment(ast->leftExpression->type));

    if (format == ExpressionFormat::kSideEffect) {
      return {};
    }

    if (gen.unit_->language() == LanguageKind::kC) {
      auto resultLoc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = gen.convertType(ast->leftExpression->type);

      auto op = mlir::cxx::LoadOp::create(
          gen.builder_, resultLoc, resultType, leftExpressionResult.value,
          gen.getAlignment(ast->leftExpression->type));

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

auto Codegen::ExpressionVisitor::operator()(TargetExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.targetValue_;
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RightExpressionAST* ast)
    -> ExpressionResult {
  auto op = gen.targetValue_;
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  if (ast->symbol) {
    auto targetExpressionResult = gen.expression(ast->targetExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, targetExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {targetExpressionResult, rightExpressionResult});
    }
  }

  auto targetExpressionResult = gen.expression(ast->targetExpression);

  auto targetValue = targetExpressionResult.value;

  std::swap(gen.targetValue_, targetValue);
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  std::swap(gen.targetValue_, targetValue);

  auto rightExpressionResult = gen.expression(ast->rightExpression);

  auto resultType = leftExpressionResult.value.getType();

  TokenKind binaryOp = TokenKind::T_EOF_SYMBOL;

  switch (ast->op) {
    case TokenKind::T_PLUS_EQUAL:
      binaryOp = TokenKind::T_PLUS;
      break;

    case TokenKind::T_MINUS_EQUAL:
      binaryOp = TokenKind::T_MINUS;
      break;

    case TokenKind::T_STAR_EQUAL:
      binaryOp = TokenKind::T_STAR;
      break;

    case TokenKind::T_SLASH_EQUAL:
      binaryOp = TokenKind::T_SLASH;
      break;

    case TokenKind::T_PERCENT_EQUAL:
      binaryOp = TokenKind::T_PERCENT;
      break;

    case TokenKind::T_AMP_EQUAL:
      binaryOp = TokenKind::T_AMP;
      break;

    case TokenKind::T_BAR_EQUAL:
      binaryOp = TokenKind::T_BAR;
      break;

    case TokenKind::T_CARET_EQUAL:
      binaryOp = TokenKind::T_CARET;
      break;

    case TokenKind::T_LESS_LESS_EQUAL:
      binaryOp = TokenKind::T_LESS_LESS;
      break;

    case TokenKind::T_GREATER_GREATER_EQUAL:
      binaryOp = TokenKind::T_GREATER_GREATER;
      break;

    default:
      break;
  }

  if (binaryOp == TokenKind::T_EOF_SYMBOL) {
    auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                               "unsupported compound assignment operator");
    return {op};
  }

  auto loc = gen.getLocation(ast->opLoc);

  auto compoundAssignmentOp = binaryExpression(
      ast->opLoc, binaryOp, resultType, ast->leftExpression,
      ast->rightExpression, leftExpressionResult, rightExpressionResult);

  targetValue = compoundAssignmentOp.value;
  std::swap(gen.targetValue_, targetValue);
  auto sourceExpressionResult = gen.expression(ast->adjustExpression);
  std::swap(gen.targetValue_, targetValue);

  if (auto member = ast_cast<MemberExpressionAST>(ast->targetExpression)) {
    if (auto field = symbol_cast<FieldSymbol>(member->symbol);
        field && field->isBitField()) {
      if (auto access = emitMemberAccess(member)) {
        auto [addr, info] = *access;
        mlir::cxx::BitfieldStoreOp::create(
            gen.builder_, loc, sourceExpressionResult.value, addr,
            gen.builder_.getI32IntegerAttr(info.bitOffset),
            gen.builder_.getI32IntegerAttr(info.bitWidth),
            gen.builder_.getI64IntegerAttr(info.allocUnitSizeBytes));

        if (format == ExpressionFormat::kSideEffect) {
          return {};
        }
        return {sourceExpressionResult.value};
      }
    }
  }

  mlir::cxx::StoreOp::create(gen.builder_, loc, sourceExpressionResult.value,
                             targetExpressionResult.value,
                             gen.getAlignment(ast->type));

  if (format == ExpressionFormat::kSideEffect) {
    return {};
  }

  if (gen.unit_->language() == LanguageKind::kC) {
    auto loadType = sourceExpressionResult.value.getType();
    auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, loadType,
                                        targetExpressionResult.value,
                                        gen.getAlignment(ast->type));
    return {op};
  }

  return targetExpressionResult;
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
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
  if (ast->value.has_value()) {
    auto resultType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::arith::ConstantOp::create(
        gen.builder_, loc, resultType,
        gen.builder_.getIntegerAttr(resultType, ast->value.value() ? 1 : 0));
    return {op};
  }

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
  auto var = ast->symbol;
  if (!var) return {};

  gen.emitLocalVariableInit(var, ast->initializer);

  auto local = gen.findOrCreateLocal(var);

  if (!local.has_value()) {
    gen.unit_->error(
        ast->firstSourceLocation(),
        std::format("cannot find local variable '{}'", to_string(var->name())));
    return {};
  }

  return {local.value()};
}

auto Codegen::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  return gen.expression(ast->expression, format);
}

auto Codegen::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  if (!ast->type) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "braced-init-list without type")};
  }

  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(ast->type);
  auto ptrType = mlir::cxx::PointerType::get(gen.context_, type);
  auto temp = mlir::cxx::AllocaOp::create(gen.builder_, loc, ptrType,
                                          gen.getAlignment(ast->type));

  gen.emitAggregateInit(temp, ast->type, ast);

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, type, temp,
                                      gen.getAlignment(ast->type));
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

void Codegen::arrayInit(mlir::Value address, const Type* type,
                        ExpressionAST* init) {
  if (!init) return;

  if (auto equal = ast_cast<EqualInitializerAST>(init)) {
    return arrayInit(address, type, equal->expression);
  }

  if (auto strLit = ast_cast<StringLiteralExpressionAST>(init)) {
    auto arr = type_cast<BoundedArrayType>(type);
    if (!arr && !type_cast<UnboundedArrayType>(type)) return;

    auto loc = getLocation(init->firstSourceLocation());

    std::string str(strLit->literal->stringValue());
    str.push_back('\0');

    auto copyLen = (arr && str.size() > arr->size()) ? arr->size() : str.size();

    if (auto size = control()->memoryLayout()->sizeOf(type)) {
      mlir::cxx::MemSetZeroOp::create(builder_, loc, address, *size);
    }

    auto i8Ty = builder_.getIntegerType(8);
    auto arrTy = mlir::cxx::ArrayType::get(context_, i8Ty, str.size());
    auto it = stringLiterals_.find(strLit->literal);
    if (it == stringLiterals_.end()) {
      auto initializer =
          builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
      auto name = builder_.getStringAttr(newUniqueSymbolName(".str"));
      auto x = mlir::OpBuilder(module_->getContext());
      x.setInsertionPointToEnd(module_.getBody());
      auto linkage = mlir::cxx::LinkageKindAttr::get(
          context_, mlir::cxx::LinkageKind::Internal);
      mlir::cxx::GlobalOp::create(x, loc, mlir::TypeRange(), arrTy, true,
                                  name.getValue(), initializer, linkage,
                                  mlir::IntegerAttr{});
      it = stringLiterals_.insert_or_assign(strLit->literal, name).first;
    }

    auto srcPtrType = mlir::cxx::PointerType::get(context_, arrTy);
    auto srcAddr =
        mlir::cxx::AddressOfOp::create(builder_, loc, srcPtrType, it->second);

    mlir::cxx::MemCpyOp::create(builder_, loc, address, srcAddr, copyLen);
    return;
  }

  auto braced = ast_cast<BracedInitListAST>(init);
  if (!braced) return;

  auto loc = getLocation(braced->firstSourceLocation());

  bool hasDesignated = false;
  for (auto node : ListView{braced->expressionList}) {
    if (ast_cast<DesignatedInitializerClauseAST>(node)) {
      hasDesignated = true;
      break;
    }
  }

  if (hasDesignated) {
    if (auto size = control()->memoryLayout()->sizeOf(type)) {
      mlir::cxx::MemSetZeroOp::create(builder_, loc, address, *size);
    }
    for (auto node : ListView{braced->expressionList}) {
      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
        emitDesignatedInit(address, type, desig);
      }
    }
    return;
  }

  if (auto size = control()->memoryLayout()->sizeOf(type)) {
    mlir::cxx::MemSetZeroOp::create(builder_, loc, address, *size);
  }

  auto elementType = unit_->typeTraits().get_element_type(type);
  auto elementMlirType = convertType(elementType);
  auto resultType = mlir::cxx::PointerType::get(context_, elementMlirType);
  auto intType = builder_.getIntegerType(32);

  int index = 0;

  for (auto node : ListView{braced->expressionList}) {
    auto nodeLoc = getLocation(node->firstSourceLocation());

    auto indexOp = mlir::arith::ConstantOp::create(
        builder_, nodeLoc, intType, builder_.getIntegerAttr(intType, index++));

    auto elementAddress = mlir::cxx::PtrAddOp::create(
        builder_, nodeLoc, resultType, address, indexOp.getResult());

    if (unit_->typeTraits().is_array(elementType)) {
      arrayInit(elementAddress, elementType, node);
    } else {
      auto value = expression(node);
      mlir::cxx::StoreOp::create(builder_, nodeLoc, value.value, elementAddress,
                                 getAlignment(elementType));
    }
  }
}

auto Codegen::emitInPlaceConstruction(mlir::Value address, ExpressionAST* ast)
    -> bool {
  auto construction = ast_cast<BracedTypeConstructionAST>(ast);
  if (!construction || !construction->constructorSymbol) return false;

  std::vector<ExpressionResult> args;
  if (construction->bracedInitList) {
    for (auto node : ListView{construction->bracedInitList->expressionList}) {
      args.push_back(expression(node));
    }
  }

  (void)emitCtorCall(ast->firstSourceLocation(),
                     construction->constructorSymbol, address, std::move(args),
                     /*completeObject=*/false);

  return true;
}

void Codegen::emitAggregateInit(mlir::Value address, const Type* type,
                                BracedInitListAST* ast) {
  auto loc = getLocation(ast->firstSourceLocation());

  if (auto size = control()->memoryLayout()->sizeOf(type)) {
    mlir::cxx::MemSetZeroOp::create(builder_, loc, address, *size);
  }

  if (unit_->typeTraits().is_array(type)) {
    auto elementType = unit_->typeTraits().get_element_type(type);
    auto elementMlirType = convertType(elementType);
    auto resultType = mlir::cxx::PointerType::get(context_, elementMlirType);
    auto intType = builder_.getIntegerType(32);

    int index = 0;
    for (auto node : ListView{ast->expressionList}) {
      auto elemLoc = getLocation(node->firstSourceLocation());

      auto indexOp = mlir::arith::ConstantOp::create(
          builder_, elemLoc, intType, builder_.getIntegerAttr(intType, index));
      auto elementAddress = mlir::cxx::PtrAddOp::create(
          builder_, elemLoc, resultType, address, indexOp.getResult());

      if (auto nested = ast_cast<BracedInitListAST>(node)) {
        emitAggregateInit(elementAddress, elementType, nested);
      } else if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
        emitDesignatedInit(address, type, desig);
      } else if (unit_->typeTraits().is_array(elementType)) {
        arrayInit(elementAddress, elementType, node);
      } else if (unit_->typeTraits().is_class_or_union(
                     unit_->typeTraits().remove_cv(elementType))) {
        (void)emitPrvalueInto(elementAddress, elementType, node,
                              node->firstSourceLocation());
      } else {
        auto val = expression(node);
        mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, elementAddress,
                                   getAlignment(elementType));
      }
      ++index;
    }
  } else if (unit_->typeTraits().is_class_or_union(type)) {
    auto classType = type_cast<ClassType>(unit_->typeTraits().remove_cv(type));
    if (!classType || !classType->symbol()) return;
    auto classSymbol = classType->symbol();

    if (auto elemType =
            unit_->typeTraits().initializer_list_element_type(type)) {
      std::uint64_t count = 0;
      for (auto it = ast->expressionList; it; it = it->next) ++count;

      std::vector<FieldSymbol*> fields;
      for (auto field :
           views::members(classSymbol) | views::non_static_fields) {
        fields.push_back(field);
      }
      if (fields.size() != 2) return;

      auto elemMlirType = convertType(elemType);
      auto elemPtrType = mlir::cxx::PointerType::get(context_, elemMlirType);
      auto intType = builder_.getIntegerType(32);

      mlir::Value beginPtr;
      if (count > 0) {
        auto arrayMlirType =
            mlir::cxx::ArrayType::get(context_, elemMlirType, count);
        auto arrayPtrType =
            mlir::cxx::PointerType::get(context_, arrayMlirType);
        auto arrayAlloca = mlir::cxx::AllocaOp::create(
            builder_, loc, arrayPtrType, getAlignment(elemType));

        int index = 0;
        for (auto node : ListView{ast->expressionList}) {
          auto elemLoc = getLocation(node->firstSourceLocation());
          auto indexOp = mlir::arith::ConstantOp::create(
              builder_, elemLoc, intType,
              builder_.getIntegerAttr(intType, index));
          auto elementAddress = mlir::cxx::PtrAddOp::create(
              builder_, elemLoc, elemPtrType, arrayAlloca, indexOp.getResult());
          auto val = expression(node);
          mlir::cxx::StoreOp::create(builder_, elemLoc, val.value,
                                     elementAddress, getAlignment(elemType));
          ++index;
        }

        auto zeroIdx = mlir::arith::ConstantOp::create(
            builder_, loc, intType, builder_.getIntegerAttr(intType, 0));
        beginPtr = mlir::cxx::PtrAddOp::create(
            builder_, loc, elemPtrType, arrayAlloca, zeroIdx.getResult());
      } else {
        beginPtr =
            mlir::cxx::NullPtrConstantOp::create(builder_, loc, elemPtrType);
      }

      auto layout = classSymbol->layout();
      auto storeField = [&](size_t idx, mlir::Value value) {
        auto field = fields[idx];
        std::uint32_t memberIndex = static_cast<std::uint32_t>(idx);
        if (layout) {
          if (auto fi = layout->getFieldInfo(field)) memberIndex = fi->index;
        }
        auto memberAddr =
            memberAddress(loc, address, field->type(), memberIndex);
        mlir::cxx::StoreOp::create(builder_, loc, value, memberAddr,
                                   getAlignment(field->type()));
      };

      storeField(0, beginPtr);

      auto sizeMlirType = convertType(fields[1]->type());
      auto sizeConst = mlir::arith::ConstantOp::create(
          builder_, loc, sizeMlirType,
          builder_.getIntegerAttr(sizeMlirType, count));
      storeField(1, sizeConst.getResult());
      return;
    }

    if (classType->isUnion()) {
      auto it = ast->expressionList;
      if (!it) return;

      auto& expr = it->value;

      FieldSymbol* targetField = nullptr;

      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
        emitDesignatedInit(address, type, desig);
        return;
      }

      for (auto field :
           views::members(classSymbol) | views::non_static_fields) {
        targetField = field;
        break;
      }

      if (!targetField) return;

      auto layout = classSymbol->layout();
      std::uint32_t memberIndex = 0;
      if (layout) {
        if (auto fi = layout->getFieldInfo(targetField)) {
          memberIndex = fi->index;
        }
      }

      auto elemLoc = getLocation(expr->firstSourceLocation());

      auto memberAddr =
          memberAddress(elemLoc, address, targetField->type(), memberIndex);

      if (auto nested = ast_cast<BracedInitListAST>(expr)) {
        emitAggregateInit(memberAddr, targetField->type(), nested);
      } else if (unit_->typeTraits().is_array(targetField->type())) {
        arrayInit(memberAddr, targetField->type(), expr);
      } else {
        auto val = expression(expr);
        mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, memberAddr,
                                   getAlignment(targetField->type()));
      }
    } else {
      std::vector<FieldSymbol*> fields;
      for (auto field :
           views::members(classSymbol) | views::non_static_fields) {
        fields.push_back(field);
      }

      auto layout = classSymbol->layout();
      size_t fieldIndex = 0;

      for (auto node : ListView{ast->expressionList}) {
        if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
          emitDesignatedInit(address, type, desig);

          if (desig->designatorList) {
            if (auto dot =
                    ast_cast<DotDesignatorAST>(desig->designatorList->value);
                dot && dot->symbol) {
              for (size_t i = 0; i < fields.size(); ++i) {
                if (fields[i] == dot->symbol) {
                  fieldIndex = i + 1;
                  break;
                }
              }
            }
          }
          continue;
        }

        if (fieldIndex >= fields.size()) break;

        auto field = fields[fieldIndex];
        std::uint32_t memberIndex = static_cast<std::uint32_t>(fieldIndex);
        std::optional<ClassLayout::MemberInfo> fi;
        if (layout) {
          fi = layout->getFieldInfo(field);
          if (fi) memberIndex = fi->index;
        }

        auto elemLoc = getLocation(node->firstSourceLocation());

        auto memberAddr =
            memberAddress(elemLoc, address, field->type(), memberIndex);

        if (auto nested = ast_cast<BracedInitListAST>(node)) {
          emitAggregateInit(memberAddr, field->type(), nested);
        } else if (emitInPlaceConstruction(memberAddr, node)) {
          ++fieldIndex;
          continue;
        } else if (unit_->typeTraits().is_array(field->type())) {
          arrayInit(memberAddr, field->type(), node);
        } else if (fi && fi->bitWidth > 0) {
          auto val = expression(node);
          mlir::cxx::BitfieldStoreOp::create(
              builder_, elemLoc, val.value, memberAddr,
              builder_.getI32IntegerAttr(fi->bitOffset),
              builder_.getI32IntegerAttr(fi->bitWidth),
              builder_.getI64IntegerAttr(fi->allocUnitSizeBytes));
        } else {
          auto val = expression(node);
          mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, memberAddr,
                                     getAlignment(field->type()));
        }
        ++fieldIndex;
      }
    }
  } else {
    auto it = ast->expressionList;
    if (!it) return;

    auto val = expression(it->value);
    mlir::cxx::StoreOp::create(builder_, loc, val.value, address,
                               getAlignment(type));
  }
}

void Codegen::emitDesignatedInit(mlir::Value address, const Type* type,
                                 DesignatedInitializerClauseAST* ast) {
  mlir::Value currentAddr = address;
  const Type* currentType = type;
  std::optional<ClassLayout::MemberInfo> currentFieldInfo;

  for (auto desigIt = ast->designatorList; desigIt; desigIt = desigIt->next) {
    auto designator = desigIt->value;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto field = dot->symbol;
      if (!field) return;

      auto classType =
          type_cast<ClassType>(unit_->typeTraits().remove_cv(currentType));
      if (!classType || !classType->symbol()) return;

      auto classSymbol = classType->symbol();
      auto fieldClass = symbol_cast<ClassSymbol>(field->parent());

      if (fieldClass && classSymbol != fieldClass) {
        auto loc = currentAddr.getLoc();
        currentAddr =
            navigateToClass(loc, currentAddr, classSymbol, fieldClass);
        classSymbol = fieldClass;
      }

      auto layout = classSymbol->layout();

      std::uint32_t memberIndex = 0;
      currentFieldInfo = std::nullopt;
      if (layout) {
        if (auto fi = layout->getFieldInfo(field)) {
          memberIndex = fi->index;
          currentFieldInfo = fi;
        }
      }

      auto elemLoc = getLocation(dot->firstSourceLocation());

      currentAddr =
          memberAddress(elemLoc, currentAddr, field->type(), memberIndex);
      currentType = unit_->typeTraits().remove_cv(field->type());

    } else if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      currentFieldInfo = std::nullopt;
      auto elementType = unit_->typeTraits().get_element_type(currentType);
      auto elementMlirType = convertType(elementType);
      auto resultType = mlir::cxx::PointerType::get(context_, elementMlirType);
      auto elemLoc = getLocation(subscript->firstSourceLocation());

      auto indexVal = expression(subscript->expression);
      currentAddr = mlir::cxx::PtrAddOp::create(builder_, elemLoc, resultType,
                                                currentAddr, indexVal.value);
      currentType = unit_->typeTraits().remove_cv(elementType);
    }
  }

  ExpressionAST* initExpr = nullptr;
  if (ast->initializer) {
    if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
      initExpr = equal->expression;
    } else {
      initExpr = ast->initializer;
    }
  }

  if (!initExpr) return;

  auto elemLoc = getLocation(initExpr->firstSourceLocation());

  if (auto nested = ast_cast<BracedInitListAST>(initExpr)) {
    emitAggregateInit(currentAddr, currentType, nested);
  } else {
    auto val = expression(initExpr);
    if (currentFieldInfo && currentFieldInfo->bitWidth > 0) {
      mlir::cxx::BitfieldStoreOp::create(
          builder_, elemLoc, val.value, currentAddr,
          builder_.getI32IntegerAttr(currentFieldInfo->bitOffset),
          builder_.getI32IntegerAttr(currentFieldInfo->bitWidth),
          builder_.getI64IntegerAttr(currentFieldInfo->allocUnitSizeBytes));
    } else {
      mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, currentAddr,
                                 getAlignment(currentType));
    }
  }
}

auto Codegen::ExpressionVisitor::emitClassConstruction(
    ExpressionAST* ast, SourceLocation loc, const Type* classType,
    List<ExpressionAST*>* argList, FunctionSymbol* constructorSymbol)
    -> ExpressionResult {
  classType = gen.unit_->typeTraits().remove_cv(classType);

  auto classT = type_cast<ClassType>(classType);
  if (!classT || !classT->symbol())
    return {gen.emitTodoExpr(loc, "class construction: no class symbol")};

  auto object = gen.takeResultObject(ast);
  const bool ownsTemporary = !object;
  if (ownsTemporary) object = gen.newTemp(classType, loc).getResult();

  std::vector<ExpressionResult> args;
  for (auto it = argList; it; it = it->next)
    args.push_back(gen.expression(it->value));

  int argCount = static_cast<int>(args.size());

  if (!constructorSymbol && argCount != 0)
    return {gen.emitTodoExpr(loc, "class construction: no recorded ctor")};

  if (constructorSymbol) {
    (void)gen.emitCtorCall(loc, constructorSymbol, object, args,
                           /*completeObject=*/true);
  }

  if (ownsTemporary) gen.addTemporaryCleanup(object, classType);

  return {object};
}

auto Codegen::emitCall(SourceLocation loc, FunctionSymbol* symbol,
                       ExpressionResult thisValue,
                       std::vector<ExpressionResult> arguments,
                       bool isVirtualDispatch) -> ExpressionResult {
  return emitCall(loc, type_cast<FunctionType>(symbol->type()), symbol,
                  isVirtualDispatch, thisValue, std::move(arguments));
}

auto Codegen::emitCall(SourceLocation loc, const FunctionType* functionType,
                       FunctionSymbol* symbol, bool isVirtualDispatch,
                       ExpressionResult thisValue,
                       std::vector<ExpressionResult> arguments,
                       mlir::Value resultObject) -> ExpressionResult {
  if (!functionType) return {};

  auto mlirLoc = getLocation(loc);

  const auto& paramTypes = functionType->parameterTypes();

  for (size_t i = 0; i < arguments.size() && i < paramTypes.size(); ++i) {
    auto val = arguments[i].value;
    if (!val) continue;

    if (unit_->typeTraits().is_reference(paramTypes[i])) {
      if (mlir::isa<mlir::cxx::PointerType>(val.getType())) continue;
      auto elemType = unit_->typeTraits().remove_reference(paramTypes[i]);
      auto temp = newTemp(elemType, loc);
      mlir::cxx::StoreOp::create(builder_, mlirLoc, val, temp,
                                 getAlignment(elemType));
      arguments[i] = {temp.getResult()};
      continue;
    }

    if (unit_->typeTraits().is_class(paramTypes[i])) {
      arguments[i] = {abiLowerClassArgument(loc, paramTypes[i], val)};
    }
  }

  mlir::SmallVector<mlir::Value> args;
  if (thisValue.value) {
    args.push_back(thisValue.value);
  }

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (!arguments[i].value) continue;
    args.push_back(arguments[i].value);
  }

  const auto returnsThis = structorReturnsThis(symbol);

  mlir::SmallVector<mlir::Type> resultTypes;
  mlir::Value sretTemp;
  if (returnsThis) {
    if (symbol && !isVirtualDispatch) {
      auto funcOp = findOrCreateFunction(symbol);
      auto results = funcOp.getFunctionType().getResults();
      resultTypes.append(results.begin(), results.end());
    } else if (!args.empty()) {
      resultTypes.push_back(args[0].getType());
    }
  } else {
    sretTemp = abiPrepareResult(loc, functionType->returnType(), resultTypes,
                                resultObject);
  }

  if (sretTemp) {
    args.insert(args.begin(), sretTemp);
  }

  mlir::cxx::CallOp callOp;

  if (isVirtualDispatch) {
    int slotIndex = vtableSlotIndex(symbol);

    auto objectPtr = thisValue.value;

    auto objectPtrType = objectPtr.getType();
    if (auto ptrPtrType = dyn_cast<mlir::cxx::PointerType>(objectPtrType)) {
      if (auto ptrType =
              dyn_cast<mlir::cxx::PointerType>(ptrPtrType.getElementType())) {
        objectPtr = mlir::cxx::LoadOp::create(
            builder_, mlirLoc, ptrPtrType.getElementType(), objectPtr, 4);
      }
    }

    auto i8Type = builder_.getI8Type();
    auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
    auto i8PtrPtrType = mlir::cxx::PointerType::get(context_, i8PtrType);

    auto vptrFieldPtr = memberAddress(mlirLoc, objectPtr, i8PtrPtrType, 0);

    auto vtablePtr = mlir::cxx::LoadOp::create(builder_, mlirLoc, i8PtrPtrType,
                                               vptrFieldPtr, 8);

    auto offsetType = convertType(control()->getIntType());
    auto offsetOp = mlir::arith::ConstantOp::create(
        builder_, mlirLoc, offsetType,
        builder_.getIntegerAttr(offsetType, slotIndex));

    auto funcPtrAddr = mlir::cxx::PtrAddOp::create(
        builder_, mlirLoc, i8PtrPtrType, vtablePtr, offsetOp);

    auto funcPtr =
        mlir::cxx::LoadOp::create(builder_, mlirLoc, i8PtrType, funcPtrAddr, 8);

    mlir::SmallVector<mlir::Value> indirectCallArgs;
    indirectCallArgs.push_back(funcPtr);
    indirectCallArgs.append(args.begin(), args.end());

    callOp = mlir::cxx::CallOp::create(builder_, mlirLoc, resultTypes,
                                       mlir::FlatSymbolRefAttr{},
                                       indirectCallArgs, mlir::TypeAttr{});
  } else if (!symbol) {
    callOp = mlir::cxx::CallOp::create(builder_, mlirLoc, resultTypes,
                                       mlir::FlatSymbolRefAttr{}, args,
                                       mlir::TypeAttr{});
  } else {
    auto funcOp = findOrCreateFunction(symbol);
    callOp =
        mlir::cxx::CallOp::create(builder_, mlirLoc, resultTypes,
                                  funcOp.getSymName(), args, mlir::TypeAttr{});
  }

  if (functionType->isVariadic()) {
    callOp.setVarCalleeType(
        mlir::cast<mlir::cxx::FunctionType>(convertType(functionType)));
  }

  if (returnsThis) return {};

  auto result =
      abiFinishResult(loc, functionType->returnType(), callOp, sretTemp);

  if (sretTemp && !resultObject)
    addTemporaryCleanup(sretTemp, functionType->returnType());

  return result;
}

auto Codegen::emitCtorCall(SourceLocation loc, FunctionSymbol* ctor,
                           mlir::Value thisPtr,
                           std::vector<ExpressionResult> args,
                           bool completeObject) -> ExpressionResult {
  auto target = ctor;
  if (completeObject) {
    if (auto variant = ctor->completeObjectVariant()) target = variant;
  }
  return emitCall(loc, target, {thisPtr}, std::move(args));
}

auto Codegen::ExpressionVisitor::codegenBuiltinLine(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto pos = gen.unit_->tokenStartPosition(ast->firstSourceLocation());
  auto intType = gen.convertType(control()->getIntType());
  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, intType,
      gen.builder_.getIntegerAttr(intType, static_cast<int64_t>(pos.line)));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinFile(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto pos = gen.unit_->tokenStartPosition(ast->firstSourceLocation());
  auto lit = control()->stringLiteral(pos.fileName);

  auto it = gen.stringLiterals_.find(lit);
  if (it == gen.stringLiterals_.end()) {
    std::string str(pos.fileName);
    str.push_back('\0');
    auto i8Type = mlir::IntegerType::get(gen.context_, 8);
    auto arrayType =
        mlir::cxx::ArrayType::get(gen.context_, i8Type, str.size());
    auto initializer =
        gen.builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
    auto name = gen.builder_.getStringAttr(gen.newUniqueSymbolName(".str"));
    auto x = mlir::OpBuilder(gen.module_->getContext());
    x.setInsertionPointToEnd(gen.module_.getBody());
    auto linkage = mlir::cxx::LinkageKindAttr::get(
        gen.context_, mlir::cxx::LinkageKind::Internal);
    mlir::cxx::GlobalOp::create(x, loc, mlir::TypeRange(), arrayType, true,
                                name.getValue(), initializer, linkage,
                                mlir::IntegerAttr{});
    it = gen.stringLiterals_.insert_or_assign(lit, name).first;
  }

  auto i8Type = mlir::IntegerType::get(gen.context_, 8);
  auto resultType = mlir::cxx::PointerType::get(gen.context_, i8Type);
  auto op =
      mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType, it->second);
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinFunction(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  std::string funcName;
  if (auto sym = gen.currentFunctionSymbol_) {
    if (auto id = name_cast<Identifier>(sym->name())) {
      funcName = id->value();
    }
  }

  auto litKey = control()->stringLiteral(funcName);
  auto it = gen.stringLiterals_.find(litKey);
  if (it == gen.stringLiterals_.end()) {
    std::string str = funcName;
    str.push_back('\0');
    auto i8Type = mlir::IntegerType::get(gen.context_, 8);
    auto arrayType =
        mlir::cxx::ArrayType::get(gen.context_, i8Type, str.size());
    auto initializer =
        gen.builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
    auto name = gen.builder_.getStringAttr(gen.newUniqueSymbolName(".str"));
    auto x = mlir::OpBuilder(gen.module_->getContext());
    x.setInsertionPointToEnd(gen.module_.getBody());
    auto linkage = mlir::cxx::LinkageKindAttr::get(
        gen.context_, mlir::cxx::LinkageKind::Internal);
    mlir::cxx::GlobalOp::create(x, loc, mlir::TypeRange(), arrayType, true,
                                name.getValue(), initializer, linkage,
                                mlir::IntegerAttr{});
    it = gen.stringLiterals_.insert_or_assign(litKey, name).first;
  }

  auto i8Type = mlir::IntegerType::get(gen.context_, 8);
  auto resultType = mlir::cxx::PointerType::get(gen.context_, i8Type);
  auto op =
      mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType, it->second);
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeVal(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getDoubleType());
  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type,
      gen.builder_.getF64FloatAttr(std::numeric_limits<double>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeValf(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getFloatType());
  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type,
      gen.builder_.getF32FloatAttr(std::numeric_limits<float>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeVall(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getLongDoubleType());
  auto op = mlir::arith::ConstantOp::create(
      gen.builder_, loc, type,
      gen.builder_.getF64FloatAttr(std::numeric_limits<double>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNans(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getDoubleType());
  auto snan = llvm::APFloat::getSNaN(llvm::APFloat::IEEEdouble());
  auto op = mlir::arith::ConstantOp::create(gen.builder_, loc, type,
                                            mlir::FloatAttr::get(type, snan));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNansf(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getFloatType());
  auto snan = llvm::APFloat::getSNaN(llvm::APFloat::IEEEsingle());
  auto op = mlir::arith::ConstantOp::create(gen.builder_, loc, type,
                                            mlir::FloatAttr::get(type, snan));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNansl(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(control()->getLongDoubleType());
  auto snan = llvm::APFloat::getSNaN(llvm::APFloat::IEEEdouble());
  auto op = mlir::arith::ConstantOp::create(gen.builder_, loc, type,
                                            mlir::FloatAttr::get(type, snan));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinAlloca(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto sizeVal = gen.expression(*it);
  if (!sizeVal.value) return {};
  auto i8Type = mlir::IntegerType::get(gen.context_, 8);
  auto ptrType = mlir::cxx::PointerType::get(gen.context_, i8Type);
  return {mlir::cxx::DynAllocaOp::create(gen.builder_, loc, ptrType,
                                         sizeVal.value, /*alignment=*/1)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinBzero(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto destVal = gen.expression(*it);
  if (!destVal.value) return {};
  ++it;
  if (it == args.end()) return {};
  auto sizeVal = gen.expression(*it);
  if (!sizeVal.value) return {};
  auto memoryLayout = gen.control()->memoryLayout();
  auto sizeType =
      mlir::IntegerType::get(gen.context_, memoryLayout->sizeOfSizeType() * 8);
  auto zero = mlir::arith::ConstantOp::create(
      gen.builder_, loc, gen.builder_.getIntegerType(8),
      gen.builder_.getIntegerAttr(sizeType, 0));
  std::vector<mlir::Value> inputs{destVal.value, zero, sizeVal.value};
  auto op = mlir::cxx::BuiltinCallOp::create(
      gen.builder_, loc, mlir::TypeRange{}, mlir::StringRef("__builtin_memset"),
      inputs);
  return {op.getResult()};
}

auto Codegen::ExpressionVisitor::codegenBuiltinCtz(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto value = gen.expression(*it);
  if (!value.value) return {};

  auto inputType = value.value.getType();

  auto i1Type = mlir::IntegerType::get(gen.context_, 1);

  mlir::SmallVector<mlir::Value> inputs{value.value};

  inputs.push_back(mlir::arith::ConstantOp::create(
      gen.builder_, loc, i1Type, gen.builder_.getIntegerAttr(i1Type, 0)));

  auto resultType = gen.convertType(ast->type);

  auto ctzOp = mlir::cxx::BuiltinCallOp::create(
      gen.builder_, loc, mlir::TypeRange{inputType},
      mlir::StringRef("__builtin_ctz"), inputs);

  auto i32Type = mlir::IntegerType::get(gen.context_, 32);

  if (resultType != i32Type) {
    auto truncOp = mlir::arith::TruncIOp::create(gen.builder_, loc, i32Type,
                                                 ctzOp.getResult());

    return {truncOp.getResult()};
  }

  return {ctzOp.getResult()};
}

auto Codegen::ExpressionVisitor::emitMemberPointerAccess(
    BinaryExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto object = gen.expression(ast->leftExpression);
  auto memberPointer = gen.expression(ast->rightExpression);
  if (!object.value || !memberPointer.value) return {};

  auto charPointerType =
      gen.convertType(control()->getPointerType(control()->getCharType()));

  auto base = mlir::cxx::BitcastOp::create(gen.builder_, loc, charPointerType,
                                           object.value);

  auto address = mlir::cxx::PtrAddOp::create(gen.builder_, loc, charPointerType,
                                             base, memberPointer.value);

  auto resultType = gen.convertType(control()->getPointerType(ast->type));

  return {mlir::cxx::BitcastOp::create(gen.builder_, loc, resultType, address)};
}

auto Codegen::ExpressionVisitor::emitMemberPointerFormation(
    UnaryExpressionAST* ast) -> std::optional<ExpressionResult> {
  auto dataPointerType = type_cast<MemberObjectPointerType>(ast->type);
  if (!dataPointerType) return std::nullopt;

  auto id = ast_cast<IdExpressionAST>(ast->expression);
  if (!id) return std::nullopt;

  auto field = symbol_cast<FieldSymbol>(id->symbol);
  if (!field || field->isStatic()) return std::nullopt;

  auto classSymbol = symbol_cast<ClassSymbol>(field->parent());
  if (!classSymbol) return std::nullopt;

  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;

  auto fieldInfo = layout->getFieldInfo(field);
  if (!fieldInfo) return std::nullopt;

  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto resultType = gen.convertType(ast->type);

  return ExpressionResult{mlir::arith::ConstantOp::create(
      gen.builder_, loc, resultType,
      gen.builder_.getIntegerAttr(
          resultType, static_cast<std::int64_t>(fieldInfo->offset)))};
}

auto Codegen::ExpressionVisitor::codegenBuiltinAddressof(CallExpressionAST* ast)
    -> ExpressionResult {
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};

  return gen.expression(*it);
}

auto Codegen::ExpressionVisitor::codegenBuiltinAssumeAligned(
    CallExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};

  auto pointer = gen.expression(*it);
  if (!pointer.value) return {};

  ++it;
  if (it == args.end()) return pointer;

  auto alignment = gen.expression(*it);
  if (!alignment.value) return pointer;

  mlir::SmallVector<mlir::Value> inputs{pointer.value, alignment.value};

  ++it;
  if (it != args.end()) {
    auto misalignment = gen.expression(*it);
    if (!misalignment.value) return pointer;
    inputs.push_back(misalignment.value);
  }

  auto assumeOp = mlir::cxx::BuiltinCallOp::create(
      gen.builder_, loc, mlir::TypeRange{gen.convertType(ast->type)},
      "__builtin_assume_aligned", inputs);

  return {assumeOp.getResult()};
}

auto Codegen::ExpressionVisitor::codegenBuiltinCountZerosGeneric(
    CallExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  mlir::StringRef builtinName =
      gen.unit_->tokenText(ast->firstSourceLocation()) == "__builtin_clzg"
          ? "__builtin_clz"
          : "__builtin_ctz";
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto value = gen.expression(*it);
  if (!value.value) return {};

  auto inputType = mlir::cast<mlir::IntegerType>(value.value.getType());
  auto i1Type = mlir::IntegerType::get(gen.context_, 1);
  auto i32Type = mlir::IntegerType::get(gen.context_, 32);

  ++it;
  const bool hasFallback = it != args.end();

  auto isZeroUndefBit = mlir::arith::ConstantOp::create(
      gen.builder_, loc, i1Type, gen.builder_.getIntegerAttr(i1Type, 0));

  mlir::SmallVector<mlir::Value> inputs{value.value, isZeroUndefBit};

  auto countOp = mlir::cxx::BuiltinCallOp::create(
      gen.builder_, loc, mlir::TypeRange{inputType}, builtinName, inputs);

  auto adjustWidth = [&](mlir::Value v) -> mlir::Value {
    auto width = mlir::cast<mlir::IntegerType>(v.getType()).getWidth();
    if (width > 32) {
      return mlir::arith::TruncIOp::create(gen.builder_, loc, i32Type, v);
    }
    if (width < 32) {
      return mlir::arith::ExtUIOp::create(gen.builder_, loc, i32Type, v);
    }
    return v;
  };

  mlir::Value raw = adjustWidth(countOp.getResult());

  if (!hasFallback) return {raw};

  auto fallback = gen.expression(*it);
  if (!fallback.value) return {raw};

  auto zero = mlir::arith::ConstantOp::create(
      gen.builder_, loc, inputType, gen.builder_.getIntegerAttr(inputType, 0));
  auto isZero = mlir::arith::CmpIOp::create(
      gen.builder_, loc, mlir::arith::CmpIPredicate::eq, value.value, zero);

  auto fallbackI32 = adjustWidth(fallback.value);

  auto selected = mlir::arith::SelectOp::create(
      gen.builder_, loc, isZero.getResult(), fallbackI32, raw);

  return {selected.getResult()};
}
}  // namespace cxx

#include "builtins_codegen-priv.h"
