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
#include <cxx/ast_printer.h>
#include <cxx/builtin_bit_operations.h>
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/initialization.h>
#include <cxx/lambda_captures.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/source_location.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
struct [[nodiscard]] Codegen::ExpressionVisitor {
  Codegen& gen;
  ExpressionFormat format = ExpressionFormat::kValue;

  [[nodiscard]] auto control() const -> Control* { return gen.control(); }

  [[nodiscard]] auto is_bool(const Type* type) const -> bool {
    return unqualified_cast<BoolType>(type);
  }

  [[nodiscard]] auto emitMemberAccess(MemberExpressionAST* ast)
      -> std::optional<std::pair<ir::ValueRef, ClassLayout::MemberInfo>>;

  [[nodiscard]] auto emitThisFieldAddress(FieldSymbol* field,
                                          SourceLocation loc)
      -> std::optional<std::pair<ir::ValueRef, ClassLayout::MemberInfo>>;

  struct BitFieldAccess {
    ir::ValueRef address;
    ClassLayout::MemberInfo info;
    FieldSymbol* field = nullptr;
  };

  [[nodiscard]] auto emitBitFieldAccess(ExpressionAST* ast)
      -> std::optional<BitFieldAccess>;

  [[nodiscard]] auto emitBitFieldLoad(SourceLocation loc,
                                      const BitFieldAccess& access,
                                      ir::TypeRef resultType) -> ir::ValueRef;

  auto emitBitFieldStore(SourceLocation loc, const BitFieldAccess& access,
                         ir::ValueRef value) -> ir::ValueRef;

  [[nodiscard]] auto emitBitFieldIncrDecr(SourceLocation loc,
                                          ExpressionAST* operand, TokenKind op,
                                          bool postfix)
      -> std::optional<ExpressionResult>;

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
  auto operator()(DefaultInitializerExpressionAST* ast) -> ExpressionResult;
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
  auto operator()(ConstExpressionAST* ast) -> ExpressionResult;
  auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;
  auto operator()(ThreeWayComparisonExpressionAST* ast) -> ExpressionResult;
  auto operator()(ConditionalExpressionAST* ast) -> ExpressionResult;
  auto operator()(YieldExpressionAST* ast) -> ExpressionResult;
  auto operator()(ThrowExpressionAST* ast) -> ExpressionResult;
  auto operator()(AssignmentExpressionAST* ast) -> ExpressionResult;
  auto operator()(TargetExpressionAST* ast) -> ExpressionResult;
  auto operator()(RightExpressionAST* ast) -> ExpressionResult;
  auto operator()(CompoundAssignmentExpressionAST* ast) -> ExpressionResult;
  auto emitAtomicCompoundAssignment(CompoundAssignmentExpressionAST* ast)
      -> ExpressionResult;
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
                        ir::TypeRef resultType, ExpressionAST* leftExpression,
                        ExpressionAST* rightExpression,
                        ExpressionResult leftExpressionResult,
                        ExpressionResult rightExpressionResult)
      -> ExpressionResult;

  auto emitBinaryArithmeticOp(SourceLocation loc, TokenKind op,
                              ir::TypeRef resultType, const Type* leftType,
                              ir::ValueRef left, ir::ValueRef right)
      -> ExpressionResult;
  auto emitBinaryArithmeticOpFloat(SourceLocation loc, TokenKind op,
                                   ir::TypeRef resultType, ir::ValueRef left,
                                   ir::ValueRef right) -> ExpressionResult;
  auto emitBinaryArithmeticOpIntegral(SourceLocation loc, TokenKind op,
                                      ir::TypeRef resultType,
                                      const Type* leftType, ir::ValueRef left,
                                      ir::ValueRef right) -> ExpressionResult;
  auto emitBinaryArithmeticOpPointer(SourceLocation loc, TokenKind op,
                                     ir::TypeRef resultType, ir::ValueRef left,
                                     ir::ValueRef right) -> ExpressionResult;

  auto emitBinaryShiftOp(SourceLocation loc, TokenKind op,
                         ir::TypeRef resultType, const Type* leftType,
                         ir::ValueRef left, ir::ValueRef right)
      -> ExpressionResult;

  auto emitBinaryComparisonOp(SourceLocation loc, TokenKind op,
                              ir::TypeRef resultType, const Type* leftType,
                              ir::ValueRef left, ir::ValueRef right)
      -> ExpressionResult;
  auto emitBinaryComparisonOpFloat(SourceLocation loc, TokenKind op,
                                   ir::TypeRef resultType, ir::ValueRef left,
                                   ir::ValueRef right) -> ExpressionResult;
  auto emitBinaryComparisonOpIntegral(SourceLocation loc, TokenKind op,
                                      ir::TypeRef resultType,
                                      const Type* leftType, ir::ValueRef left,
                                      ir::ValueRef right) -> ExpressionResult;
  auto emitBinaryComparisonOpPointer(SourceLocation loc, TokenKind op,
                                     ir::TypeRef resultType,
                                     const Type* leftType, ir::ValueRef left,
                                     ir::ValueRef right) -> ExpressionResult;
  auto emitThreeWayComparison(ThreeWayComparisonExpressionAST* ast,
                              ExpressionResult left, ExpressionResult right)
      -> ExpressionResult;
  [[nodiscard]] auto comparisonCategoryAddress(SourceLocation loc,
                                               Symbol* symbol) -> ir::ValueRef;
  auto emitBinaryBitwiseOp(SourceLocation loc, TokenKind op,
                           ir::TypeRef resultType, ir::ValueRef left,
                           ir::ValueRef right) -> ExpressionResult;

  auto emitImplicitCast(ImplicitCastExpressionAST* ast) -> ExpressionResult;
  auto emitLValueToRValueConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitNumericConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitPointerConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitVectorSplat(ImplicitCastExpressionAST* ast) -> ExpressionResult;

  auto emitVectorConversion(ImplicitCastExpressionAST* ast) -> ExpressionResult;
  auto emitBaseToDerivedConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitDerivedToBaseConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;
  auto emitUserDefinedConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitPointerToMemberConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitBuiltinCall(CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
      -> ExpressionResult;

  auto codegenBuiltinDispatch(CallExpressionAST* ast,
                              BuiltinFunctionKind builtinKind)
      -> std::optional<ExpressionResult>;

  auto emitArithmeticConversion(SourceLocation loc, ir::ValueRef value,
                                const Type* sourceType, const Type* targetType)
      -> ir::ValueRef;

  auto emitComplexPart(SourceLocation loc, ir::ValueRef value,
                       const ComplexType* complexType, std::int64_t position)
      -> ir::ValueRef;

  auto makeComplexValue(SourceLocation loc, const ComplexType* complexType,
                        ir::ValueRef real, ir::ValueRef imag) -> ir::ValueRef;

  auto emitComplexOperand(SourceLocation loc, ir::ValueRef value,
                          const Type* sourceType, const ComplexType* targetType)
      -> ir::ValueRef;

  auto emitComplexConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitComplexToBoolean(ImplicitCastExpressionAST* ast) -> ExpressionResult;

  auto sequentiallyConsistentOrder(SourceLocation loc) -> ir::ValueRef;

  auto emitAtomicLoad(SourceLocation loc, const Type* valueType,
                      ir::ValueRef address) -> ir::ValueRef;

  void emitAtomicStore(SourceLocation loc, ir::ValueRef address,
                       ir::ValueRef value);

  auto emitAtomicReadModifyWrite(SourceLocation loc, TokenKind binaryOp,
                                 const Type* valueType, ir::ValueRef address,
                                 ir::ValueRef operand) -> ir::ValueRef;

  auto emitAtomicIncrDecr(SourceLocation loc, TokenKind op,
                          const Type* atomicType, ir::ValueRef address,
                          bool postfix) -> ExpressionResult;

  auto emitRealImag(UnaryExpressionAST* ast) -> ExpressionResult;

  auto emitComplexArithmeticOp(SourceLocation loc, TokenKind op,
                               const ComplexType* complexType,
                               ir::ValueRef left, ir::ValueRef right)
      -> ExpressionResult;

  auto emitComplexComparisonOp(SourceLocation loc, TokenKind op,
                               const ComplexType* complexType,
                               ir::ValueRef left, ir::ValueRef right)
      -> ExpressionResult;

  auto codegenBuiltinComplex(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeVal(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeValf(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinHugeVall(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNans(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNansf(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinNansl(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinAlloca(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinBzero(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinArithmeticOverflow(CallExpressionAST* ast)
      -> ExpressionResult;
  auto codegenBuiltinFloatComparison(CallExpressionAST* ast)
      -> ExpressionResult;

  auto codegenBuiltinAddressof(CallExpressionAST* ast) -> ExpressionResult;
  auto emitMemberPointerFormation(UnaryExpressionAST* ast)
      -> std::optional<ExpressionResult>;
  auto emitMemberFunctionPointerFormation(
      UnaryExpressionAST* ast, const MemberFunctionPointerType* pointerType)
      -> std::optional<ExpressionResult>;
  auto emitMemberPointerAccess(BinaryExpressionAST* ast) -> ExpressionResult;
  auto emitMemberFunctionPointerCall(CallExpressionAST* ast,
                                     BinaryExpressionAST* access)
      -> ExpressionResult;
  auto codegenBuiltinAssumeAligned(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinIsNan(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinIsInf(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinIsFinite(CallExpressionAST* ast) -> ExpressionResult;
  auto codegenBuiltinSignbit(CallExpressionAST* ast) -> ExpressionResult;
  [[nodiscard]] auto floatClassificationOperand(CallExpressionAST* ast)
      -> std::optional<std::pair<SourceLocation, ir::ValueRef>>;
  auto codegenBuiltinBitCount(CallExpressionAST* ast) -> ExpressionResult;

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

void Codegen::condition(ExpressionAST* ast, ir::BlockRef trueBlock,
                        ir::BlockRef falseBlock) {
  if (!ast) return;

  if (auto nested = ast_cast<NestedExpressionAST>(ast)) {
    condition(nested->expression, trueBlock, falseBlock);
    return;
  }

  if (auto binop = ast_cast<BinaryExpressionAST>(ast)) {
    if (binop->op == TokenKind::T_AMP_AMP) {
      auto nextBlock = newBlock();
      condition(binop->leftExpression, nextBlock, falseBlock);
      emitter_.setInsertionBlock(nextBlock);
      auto conditionalEvaluation = ConditionalEvaluation{*this};
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }

    if (binop->op == TokenKind::T_BAR_BAR) {
      auto nextBlock = newBlock();
      condition(binop->leftExpression, trueBlock, nextBlock);
      emitter_.setInsertionBlock(nextBlock);
      auto conditionalEvaluation = ConditionalEvaluation{*this};
      condition(binop->rightExpression, trueBlock, falseBlock);
      return;
    }
  }

  auto value = expression(ast);

  emitter_.condBranch(ast->firstSourceLocation(), value.value, trueBlock,
                      falseBlock);
}

void Codegen::conditionWithCleanups(ExpressionAST* ast, ir::BlockRef trueBlock,
                                    ir::BlockRef falseBlock) {
  if (!ast) return;

  const auto outerDepth = cleanupStack_.size();
  const auto endLoc = lastTokenLocation(ast);

  auto trueCleanupBlock = newBlock();
  auto falseCleanupBlock = newBlock();

  pushFullExpressionCleanup();
  condition(ast, trueCleanupBlock, falseCleanupBlock);

  emitter_.setInsertionBlock(trueCleanupBlock);
  emitBranchWithCleanups(endLoc, trueBlock, outerDepth);

  emitter_.setInsertionBlock(falseCleanupBlock);
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
  if (ast->literalOperatorCall) return gen.expression(ast->literalOperatorCall);

  auto loc = ast->literalLoc;

  auto type = gen.convertType(ast->type);
  auto value = std::int64_t(ast->literal->charValue());
  auto op = gen.emitter_.constantInt(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(BoolLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->literalLoc;

  auto type = gen.convertType(ast->type);

  auto op = gen.emitter_.constantInt(loc, type, ast->isTrue ? 1 : 0);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  if (ast->literalOperatorCall) return gen.expression(ast->literalOperatorCall);

  auto loc = ast->literalLoc;

  auto type = gen.convertType(ast->type);
  auto value = ast->literal->integerValue();

  auto op = gen.emitter_.constantInt(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FloatLiteralExpressionAST* ast)
    -> ExpressionResult {
  if (ast->literalOperatorCall) return gen.expression(ast->literalOperatorCall);

  auto loc = ast->literalLoc;

  auto type = gen.convertType(ast->type);

  if (!gen.traits.is_floating_point(ast->type)) {
    auto op =
        gen.emitTodoExpr(ast->firstSourceLocation(), "unsupported float type");
    return {op};
  }

  ir::Initializer value =
      gen.makeFloatInitializer(ast->type, ast->literal->floatValue());

  auto op = gen.emitter_.constantLiteral(loc, type, value);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NullptrLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->literalLoc;
  auto resultType = gen.emitter_.pointerType(gen.emitter_.voidType());
  auto op = gen.emitter_.nullPointer(loc, resultType);
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(StringLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->literalLoc;
  auto type = gen.convertType(ast->type);
  auto resultType = gen.emitter_.pointerType(type);

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
        ir::Initializer::byteString(std::string_view(str.data(), str.size()));

    auto name = gen.newUniqueSymbolName(".str");

    auto insertionGuard = ir::InsertionGuard(gen.emitter_);
    gen.emitter_.setModuleInsertionPoint(false);
    auto linkage = ir::Linkage::Internal;
    (void)gen.declareGlobal(loc, {.name = name,
                                  .type = type,
                                  .linkage = linkage,
                                  .isConstant = true,
                                  .alignment = static_cast<std::uint64_t>(0),
                                  .initializer = initializer,
                                  .unknownLocation = false});

    it = gen.stringLiterals_.insert_or_assign(ast->literal, name).first;
  }

  auto op = gen.emitter_.addressOfSymbol(loc, resultType, it->second);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  if (ast->literalOperatorCall) return gen.expression(ast->literalOperatorCall);

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ObjectLiteralExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = ast->type;
  auto irType = gen.convertType(type);
  auto ptrType = gen.emitter_.pointerType(irType);
  auto allocaOp = gen.emitter_.allocate(loc, ptrType, gen.getAlignment(type));

  if (gen.traits.is_array(type) || gen.traits.is_vector(type)) {
    gen.arrayInit(allocaOp, type, ast->bracedInitList);
  } else if (gen.traits.is_class(type) && ast->bracedInitList) {
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
        gen.emitter_.store(loc, initResult.value, allocaOp,
                           gen.getAlignment(type));
      }
    } else {
      auto zero = gen.emitter_.constantZero(loc, irType);
      gen.emitter_.store(loc, zero, allocaOp, gen.getAlignment(type));
    }
  }

  return {allocaOp};
}

auto Codegen::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  if (auto thisClassType = unqualified_cast<ClassType>(
          gen.traits.get_element_type(gen.traits.remove_cv(ast->type)));
      thisClassType && thisClassType->symbol()) {
    auto thisClass = thisClassType->symbol()->resolvedDefinition();

    ClassSymbol* objectClass = nullptr;
    if (auto object = gen.loadEnclosingObject(loc, thisClass, objectClass);
        object) {
      return {gen.navigateToClass(loc, object, objectClass, thisClass)};
    }
  }

  auto ptrType = gen.convertType(ast->type);

  auto loadOp = gen.emitter_.load(loc, ptrType, gen.thisValue_,
                                  gen.getAlignment(ast->type));

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
  auto selected = getGenericSelectionExpression(ast);

  if (!selected) {
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  return gen.expression(selected);
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
  if (!stmts.empty() && !gen.traits.is_void(ast->type)) {
    if (auto last = ast_cast<ExpressionStatementAST>(stmts.back()))
      if (last->expression && last->expression->type) lastExprStmt = last;
  }

  gen.pushCleanup();

  auto count = lastExprStmt ? stmts.size() - 1 : stmts.size();
  for (std::size_t i = 0; i < count; ++i) gen.statement(stmts[i]);

  ir::ValueRef result;
  if (lastExprStmt)
    result = gen.expression(lastExprStmt->expression, ExpressionFormat::kValue)
                 .value;

  gen.popCleanup(ast->statement->rbraceLoc);

  if (result) return {result};
  return {};
}

auto Codegen::ExpressionVisitor::operator()(
    DefaultInitializerExpressionAST* ast) -> ExpressionResult {
  std::optional<ThisValueGuard> thisValue;
  if (gen.defaultInitializerObject_)
    thisValue.emplace(gen, gen.defaultInitializerObject_);

  if (auto object = gen.takeResultObject(ast)) {
    (void)gen.emitPrvalueInto(object, ast->type, ast->expression,
                              ast->firstSourceLocation());
    return {object};
  }

  return gen.expression(ast->expression, format);
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
    ir::ValueRef val;
    bool found = false;

    if (auto local = gen.findOrCreateLocal(var)) {
      val = local.value();
      found = true;
    } else if (auto global = gen.findOrCreateGlobal(var)) {
      auto loc = ast->firstSourceLocation();
      auto resultType = gen.emitter_.pointerType(gen.convertType(var->type()));
      val = gen.emitter_.addressOfSymbol(loc, resultType,
                                         gen.globalName(*global));
      found = true;
    }

    if (found) {
      return {gen.loadReferenceBinding(ast->firstSourceLocation(), var->type(),
                                       val)};
    }
  } else if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    if (auto local = gen.findOrCreateLocal(ast->symbol)) {
      return {gen.loadReferenceBinding(ast->firstSourceLocation(),
                                       param->type(), local.value())};
    }
  } else if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (field->isStatic()) {
      if (auto def = field->definition()) {
        if (auto global = gen.findOrCreateGlobal(def)) {
          auto loc = ast->firstSourceLocation();
          auto resultType =
              gen.emitter_.pointerType(gen.convertType(def->type()));
          return {gen.emitter_.addressOfSymbol(loc, resultType,
                                               gen.globalName(*global))};
        }
      }

      if (!field->definition()) {
        auto global = gen.findOrCreateStaticField(field);
        auto loc = ast->firstSourceLocation();
        auto resultType =
            gen.emitter_.pointerType(gen.convertType(field->type()));
        return {gen.emitter_.addressOfSymbol(loc, resultType,
                                             gen.globalName(global))};
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
        auto loc = ast->firstSourceLocation();
        bool isSigned = gen.traits.is_signed(field->type());
        auto loadOp = gen.emitter_.loadBitfield(
            loc, gen.convertType(ast->type), op,
            {info.bitOffset, info.bitWidth, info.allocUnitSizeBytes}, isSigned);
        return {loadOp, ValueCategory::kPrValue,
                /*isRValueMaterialized=*/true};
      }

      if (gen.traits.is_reference(field->type())) {
        auto loc = ast->firstSourceLocation();
        auto type = gen.convertType(field->type());
        op = gen.emitter_.load(loc, type, op, gen.getAlignment(field->type()));
      }
      return {op};
    }
  } else if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    if (enumerator->value().has_value()) {
      if (auto val = std::get_if<ConstInt>(&enumerator->value().value())) {
        auto loc = ast->firstSourceLocation();
        auto type = gen.convertType(enumerator->type());
        auto op = gen.emitter_.constantInt(loc, type, val->toIntMax());
        return {op};
      }
    }
  }

  if (ast->symbol) {
    if (auto funcSymbol = symbol_cast<FunctionSymbol>(ast->symbol)) {
      auto funcOp = gen.findOrCreateFunction(funcSymbol);
      auto loc = ast->firstSourceLocation();
      auto type =
          gen.convertType(gen.control()->getPointerType(funcSymbol->type()));
      auto name = gen.functionName(funcOp);
      auto op = gen.emitter_.addressOfSymbol(loc, type, name);
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

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  if (auto classType = type_cast<ClassType>(ast->type)) {
    auto classSymbol = classType->symbol();

    {
      auto savedIP = gen.emitter_.saveInsertionPoint();

      gen.emitter_.setModuleInsertionPoint(false);

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

      gen.emitter_.restoreInsertionPoint(savedIP);
    }

    auto closure = gen.takeResultObject(ast);
    if (!closure) closure = gen.newTemp(classType, ast->firstSourceLocation());

    if (ast->constructorSymbol) {
      (void)gen.emitCtorCall(ast->firstSourceLocation(), ast->constructorSymbol,
                             closure, /*args=*/{},
                             /*completeObject=*/true);
    }

    for (auto captureNode : ListView{ast->captureList}) {
      auto initExpr = capture_initializer(captureNode);
      if (!initExpr) continue;
      gen.emitCaptureInit(classSymbol, closure, captureNode, initExpr);
    }

    return {closure};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  auto interp = ASTInterpreter{gen.unit_};
  if (auto value = interp.evaluate(ast)) {
    if (auto cst = gen.emitConstInitValue(loc, ast->type, *value)) return {cst};
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->vaArgLoc;

  auto expressionResult = gen.expression(ast->expression);

  std::vector<ir::ValueRef> arguments;
  arguments.push_back(expressionResult.value);

  std::vector<ir::TypeRef> resultTypes;
  if (ast->type && !gen.traits.is_void(ast->type)) {
    resultTypes.push_back(gen.convertType(ast->type));
  }

  auto op =
      gen.emitter_.builtinCall(loc, resultTypes, "__builtin_va_arg", arguments);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  if (ast->symbol) {
    auto baseExpressionResult = gen.expression(ast->baseExpression);
    auto indexExpressionResult = gen.expression(ast->indexExpression);
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->lbracketLoc, ast->symbol, baseExpressionResult,
                          {indexExpressionResult}, ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->lbracketLoc, ast->symbol, {},
                          {baseExpressionResult, indexExpressionResult}, false,
                          ast);
    }
  }

  auto baseExpressionResult = gen.expression(ast->baseExpression);
  auto indexExpressionResult = gen.expression(ast->indexExpression);

  auto loc = ast->firstSourceLocation();

  auto resultType = gen.convertType(gen.traits.add_pointer(ast->type));

  auto baseType = ast->baseExpression->type;
  ir::ValueRef index = indexExpressionResult.value;

  const Type* strideBaseType = nullptr;
  if (gen.traits.is_pointer(baseType))
    strideBaseType = gen.traits.get_element_type(baseType);
  else if (auto vla = type_cast<UnresolvedBoundedArrayType>(baseType))
    strideBaseType = vla->elementType();

  if (strideBaseType) {
    ir::ValueRef stride;
    const Type* cur = strideBaseType;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      auto countVal = countResult.value;
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(countVal)) ==
           ir::TypeKind::Pointer)) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal = gen.emitter_.load(loc, valueType, countVal,
                                     gen.getAlignment(vla->size()->type));
      }
      if (gen.emitter_.typeOf(countVal) != gen.emitter_.typeOf(index))
        countVal =
            gen.emitter_.signExtend(loc, countVal, gen.emitter_.typeOf(index));
      stride = stride ? gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, stride,
                                              countVal)
                      : countVal;
      cur = vla->elementType();
    }
    if (stride)
      index = gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, index, stride);
  }

  if (gen.traits.is_pointer(baseType) ||
      type_cast<UnresolvedBoundedArrayType>(baseType)) {
    auto op = gen.emitter_.pointerAdd(loc, resultType,
                                      baseExpressionResult.value, index);
    return {op};
  }

  auto indexType = ast->indexExpression->type;
  if (gen.traits.is_pointer(indexType)) {
    auto op =
        gen.emitter_.pointerAdd(loc, resultType, indexExpressionResult.value,
                                baseExpressionResult.value);
    return {op};
  }

  auto op = gen.emitter_.subscript(loc, resultType, baseExpressionResult.value,
                                   indexExpressionResult.value);

  return {op};
}

auto Codegen::ExpressionVisitor::emitBuiltinCall(
    CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
    -> ExpressionResult {
  if (auto result = codegenBuiltinDispatch(ast, builtinKind)) {
    return *result;
  }

  auto loc = ast->lparenLoc;

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_UNREACHABLE) {
    gen.emitter_.unreachable(loc);
    return {};
  }

  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_IS_CONSTANT_EVALUATED) {
    auto boolType = gen.convertType(control()->getBoolType());
    auto falseVal = gen.emitter_.constantInt(loc, boolType, 0);
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
    auto val = gen.emitter_.constantInt(loc, intType, result);
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

  std::vector<ir::ValueRef> arguments;
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  std::vector<ir::TypeRef> resultTypes;
  if (ast->type && !gen.traits.is_void(ast->type)) {
    resultTypes.push_back(gen.convertType(ast->type));
  }

  auto op = gen.emitter_.builtinCall(loc, resultTypes, name, arguments);

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto func = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(func)) {
    func = nested->expression;
  }

  if (auto id = ast_cast<IdExpressionAST>(func)) {
    auto builtinKind = resolveBuiltinFunctionKind(id);
    if (builtinKind != BuiltinFunctionKind::T_NONE) {
      return emitBuiltinCall(ast, builtinKind);
    }
  }

  if (auto access = ast_cast<BinaryExpressionAST>(func);
      access && (access->op == TokenKind::T_DOT_STAR ||
                 access->op == TokenKind::T_MINUS_GREATER_STAR)) {
    if (type_cast<MemberFunctionPointerType>(
            gen.traits.remove_cvref(access->rightExpression->type)))
      return emitMemberFunctionPointerCall(ast, access);
    return gen.expression(access, format);
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
    if ((functionSymbol = symbol_cast<FunctionSymbol>(id->symbol))) {
      if (functionSymbol->isImplicitObjectMemberFunction()) {
        auto loc = ast->firstSourceLocation();
        auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

        ClassSymbol* currentClass = nullptr;
        auto loadedThis =
            gen.loadEnclosingObject(loc, classSymbol, currentClass);

        auto adjustedThis = gen.emitBaseClassAddress(loc, loadedThis,
                                                     currentClass, classSymbol);
        thisValue = {adjustedThis};
      }
    }

  } else if (member) {
    functionSymbol = symbol_cast<FunctionSymbol>(member->symbol);

    if (functionSymbol && !functionSymbol->hasExplicitObjectParameter()) {
      auto baseResult = gen.expression(member->baseExpression);

      if (functionSymbol->isImplicitObjectMemberFunction()) {
        thisValue = baseResult;

        if (auto targetClass =
                symbol_cast<ClassSymbol>(functionSymbol->parent())) {
          auto baseType = gen.traits.remove_cv(member->baseExpression->type);
          if (member->accessOp == TokenKind::T_MINUS_GREATER) {
            baseType =
                gen.traits.remove_cv(gen.traits.get_element_type(baseType));
          }
          if (auto fromClassType = type_cast<ClassType>(baseType)) {
            thisValue = {gen.emitBaseClassAddress(
                member->firstSourceLocation(), thisValue.value,
                fromClassType->symbol(), targetClass)};
          }
        }
      }
    }
  }

  const FunctionType* functionType = nullptr;
  ir::ValueRef calleeValue;

  if (functionSymbol) {
    functionType = type_cast<FunctionType>(functionSymbol->type());
  } else if (gen.traits.is_pointer(ast->baseExpression->type)) {
    calleeValue = gen.expression(ast->baseExpression).value;

    auto elementType = gen.traits.get_element_type(ast->baseExpression->type);
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

  return gen.emitCall(ast->lparenLoc, functionType, functionSymbol,
                      isVirtualCall, thisValue, std::move(callArguments),
                      gen.takeIndirectResultObject(ast, functionType),
                      calleeValue);
}

auto Codegen::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  const Type* targetType = ast->type;

  if (!targetType) {
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
  }

  auto loc = ast->firstSourceLocation();

  if (gen.traits.is_void(targetType)) {
    for (auto node : ListView{ast->expressionList}) {
      (void)gen.expression(node);
    }
    return {};
  }

  if (gen.traits.is_class(targetType)) {
    if (ast->constructorSymbol) {
      return emitClassConstruction(ast, ast->firstSourceLocation(), targetType,
                                   ast->expressionList, ast->constructorSymbol);
    }

    auto temp = gen.newTemp(targetType, ast->firstSourceLocation());
    gen.emitAggregateInit(temp, targetType, ast->expressionList,
                          ast->firstSourceLocation());
    return {temp};
  }

  auto resultType = gen.convertType(targetType);

  const auto sourceLoc = ast->firstSourceLocation();
  const auto resultKind = gen.emitter_.typeKind(resultType);

  if (!ast->expressionList) {
    switch (resultKind) {
      case ir::TypeKind::Integer: {
        auto op = gen.emitter_.constantInt(loc, resultType, 0);
        return {op};
      }

      case ir::TypeKind::Floating: {
        auto op = gen.emitter_.constantZero(loc, resultType);
        return {op};
      }

      case ir::TypeKind::Pointer:
        return {gen.emitter_.nullPointer(loc, resultType)};

      default:
        break;
    }
    return {gen.emitTodoExpr(sourceLoc, to_string(ast->kind()))};
  }

  auto argResult = gen.expression(ast->expressionList->value);
  auto argType = gen.emitter_.typeOf(argResult.value);

  if (argType == resultType) return argResult;

  const auto argKind = gen.emitter_.typeKind(argType);

  if (argKind == ir::TypeKind::Integer &&
      resultKind == ir::TypeKind::Floating) {
    return {
        gen.emitter_.signedIntToFloat(sourceLoc, argResult.value, resultType)};
  }

  if (argKind == ir::TypeKind::Floating &&
      resultKind == ir::TypeKind::Integer) {
    return {
        gen.emitter_.floatToSignedInt(sourceLoc, argResult.value, resultType)};
  }

  if (argKind == ir::TypeKind::Floating &&
      resultKind == ir::TypeKind::Floating) {
    if (gen.emitter_.scalarWidth(argType) <
        gen.emitter_.scalarWidth(resultType))
      return {gen.emitter_.floatExtend(sourceLoc, argResult.value, resultType)};
    return {gen.emitter_.floatTruncate(sourceLoc, argResult.value, resultType)};
  }

  if (argKind == ir::TypeKind::Integer && resultKind == ir::TypeKind::Integer) {
    if (gen.emitter_.scalarWidth(argType) <
        gen.emitter_.scalarWidth(resultType))
      return {gen.emitter_.signExtend(sourceLoc, argResult.value, resultType)};
    return {gen.emitter_.truncate(sourceLoc, argResult.value, resultType)};
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

  if (gen.traits.is_void(targetType)) {
    if (ast->bracedInitList) {
      for (auto node : ListView{ast->bracedInitList->expressionList}) {
        (void)gen.expression(node);
      }
    }
    return {};
  }

  if (gen.traits.is_class(targetType)) {
    if (ast->constructorSymbol) {
      return emitClassConstruction(
          ast, ast->firstSourceLocation(), targetType,
          gen.constructorArgumentList(ast->bracedInitList),
          ast->constructorSymbol);
    }

    auto object = gen.takeResultObject(ast);
    const bool ownsTemporary = !object;
    if (ownsTemporary)
      object = gen.newTemp(targetType, ast->firstSourceLocation());

    if (ast->bracedInitList) {
      ast->bracedInitList->type = targetType;
      gen.emitAggregateInit(object, targetType, ast->bracedInitList);
    }

    if (ownsTemporary) gen.addTemporaryCleanup(object, targetType);

    return {object};
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

auto Codegen::navigateToClass(SourceLocation loc, ir::ValueRef value,
                              ClassSymbol* from, ClassSymbol* to)
    -> ir::ValueRef {
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
    -> std::optional<std::pair<ir::ValueRef, ClassLayout::MemberInfo>> {
  if (!field || field->isStatic() || !gen.thisValue_) return std::nullopt;

  auto classSymbol = symbol_cast<ClassSymbol>(field->parent());
  if (!classSymbol) return std::nullopt;

  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;

  auto fieldInfo = layout->getFieldInfo(field);
  if (!fieldInfo) return std::nullopt;

  auto loc = srcLoc;

  ClassSymbol* currentClass = nullptr;
  auto thisPtr = gen.loadEnclosingObject(loc, classSymbol, currentClass);

  auto adjustedThis =
      gen.navigateToClass(loc, thisPtr, currentClass, classSymbol);

  auto op =
      gen.memberAddress(loc, adjustedThis, field->type(), fieldInfo->index);
  return std::pair{op, *fieldInfo};
}

auto Codegen::ExpressionVisitor::emitMemberAccess(MemberExpressionAST* ast)
    -> std::optional<std::pair<ir::ValueRef, ClassLayout::MemberInfo>> {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && !field->isStatic()) {
    auto baseExpressionResult = gen.expression(ast->baseExpression);

    auto baseType = gen.traits.remove_cv(ast->baseExpression->type);

    if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
      baseType = gen.traits.remove_cv(gen.traits.get_element_type(baseType));
    }

    if (!(gen.emitter_.typeKind(gen.emitter_.typeOf(
              baseExpressionResult.value)) == ir::TypeKind::Pointer)) {
      auto tempLoc = ast->baseExpression->firstSourceLocation();
      auto temp =
          gen.newTemp(baseType, ast->baseExpression->firstSourceLocation());
      gen.emitter_.store(tempLoc, baseExpressionResult.value, temp,
                         gen.getAlignment(baseType));
      baseExpressionResult = {temp};
    }

    auto classType = type_cast<ClassType>(baseType);

    if (!classType) {
      (void)gen.emitTodoExpr(
          ast->firstSourceLocation(),
          std::format("base not class type '{}'", to_string(baseType)));
      return std::nullopt;
    }

    auto startClass = classType->symbol();
    auto fieldClass = symbol_cast<ClassSymbol>(field->parent());

    if (startClass != fieldClass) {
      auto loc = ast->firstSourceLocation();
      baseExpressionResult.value = gen.navigateToClass(
          loc, baseExpressionResult.value, startClass, fieldClass);
    }

    auto layout = fieldClass->layout();
    if (!layout) {
      (void)gen.emitTodoExpr(ast->firstSourceLocation(),
                             "class layout not computed");
      return std::nullopt;
    }

    auto fieldInfo = layout->getFieldInfo(field);
    if (!fieldInfo) {
      (void)gen.emitTodoExpr(ast->firstSourceLocation(),
                             "field not found in layout");
      return std::nullopt;
    }

    auto loc = ast->firstSourceLocation();
    auto op = gen.memberAddress(loc, baseExpressionResult.value, field->type(),
                                fieldInfo->index);
    return std::pair{op, *fieldInfo};
  }
  return std::nullopt;
}

auto Codegen::ExpressionVisitor::emitBitFieldAccess(ExpressionAST* ast)
    -> std::optional<BitFieldAccess> {
  auto idExpr = ast_cast<IdExpressionAST>(ast);
  auto member = ast_cast<MemberExpressionAST>(ast);
  if (!idExpr && !member) return std::nullopt;

  auto field =
      symbol_cast<FieldSymbol>(idExpr ? idExpr->symbol : member->symbol);
  if (!field || field->isStatic() || !field->isBitField()) return std::nullopt;

  auto access = idExpr
                    ? emitThisFieldAddress(field, idExpr->firstSourceLocation())
                    : emitMemberAccess(member);
  if (!access) return std::nullopt;

  return BitFieldAccess{access->first, access->second, field};
}

auto Codegen::ExpressionVisitor::emitBitFieldIncrDecr(SourceLocation loc,
                                                      ExpressionAST* operand,
                                                      TokenKind op,
                                                      bool postfix)
    -> std::optional<ExpressionResult> {
  auto access = emitBitFieldAccess(operand);
  if (!access) return std::nullopt;

  auto resultType = gen.convertType(operand->type);
  auto oldValue = emitBitFieldLoad(loc, *access, resultType);
  auto step = gen.emitter_.constantInt(loc, resultType,
                                       op == TokenKind::T_PLUS_PLUS ? 1 : -1);
  auto stored = emitBitFieldStore(
      loc, *access,
      gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, oldValue, step));

  return ExpressionResult{postfix ? oldValue : stored};
}

auto Codegen::ExpressionVisitor::emitBitFieldLoad(SourceLocation loc,
                                                  const BitFieldAccess& access,
                                                  ir::TypeRef resultType)
    -> ir::ValueRef {
  const bool isSigned = gen.traits.is_signed(access.field->type());
  return gen.emitter_.loadBitfield(loc, resultType, access.address,
                                   {access.info.bitOffset, access.info.bitWidth,
                                    access.info.allocUnitSizeBytes},
                                   isSigned);
}

auto Codegen::ExpressionVisitor::emitBitFieldStore(SourceLocation loc,
                                                   const BitFieldAccess& access,
                                                   ir::ValueRef value)
    -> ir::ValueRef {
  gen.emitter_.storeBitfield(loc, value, access.address,
                             {access.info.bitOffset, access.info.bitWidth,
                              access.info.allocUnitSizeBytes});

  if (format == ExpressionFormat::kSideEffect) return {};

  return emitBitFieldLoad(loc, access, gen.emitter_.typeOf(value));
}

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto symbol = resolve_using_declaration(ast->symbol);

  if (auto enumerator = symbol_cast<EnumeratorSymbol>(symbol)) {
    (void)gen.expression(ast->baseExpression, ExpressionFormat::kSideEffect);
    if (enumerator->value().has_value()) {
      if (auto val = std::get_if<ConstInt>(&enumerator->value().value())) {
        auto loc = ast->firstSourceLocation();
        auto type = gen.convertType(enumerator->type());
        auto op = gen.emitter_.constantInt(loc, type, val->toIntMax());
        return {op};
      }
    }
  }

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
      auto loc = ast->firstSourceLocation();
      auto fieldSym = symbol_cast<FieldSymbol>(ast->symbol);
      bool isSigned = fieldSym && gen.traits.is_signed(fieldSym->type());
      auto loadOp = gen.emitter_.loadBitfield(
          loc, gen.convertType(ast->type), op,
          {info.bitOffset, info.bitWidth, info.allocUnitSizeBytes}, isSigned);
      return {loadOp, ValueCategory::kPrValue,
              /*isRValueMaterialized=*/true};
    }

    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && gen.traits.is_reference(field->type())) {
      auto loc = ast->firstSourceLocation();
      auto type = gen.convertType(field->type());
      op = gen.emitter_.load(loc, type, op, gen.getAlignment(field->type()));
    }

    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->symbol) {
    if (auto result = emitBitFieldIncrDecr(ast->firstSourceLocation(),
                                           ast->baseExpression, ast->op,
                                           /*postfix=*/true))
      return *result;
  }

  auto expressionResult = gen.expression(ast->baseExpression);

  if (ast->symbol) {
    auto loc = ast->opLoc;
    auto intTy = gen.emitter_.integerType(32);
    auto zeroOp = gen.emitter_.constantInt(loc, intTy, 0);
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {{zeroOp}},
                          ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {expressionResult, {zeroOp}}, false, ast);
    }
  }

  if (gen.traits.is_atomic(ast->baseExpression->type)) {
    return emitAtomicIncrDecr(ast->opLoc, ast->op, ast->baseExpression->type,
                              expressionResult.value, /*postfix=*/true);
  }

  if (gen.traits.is_integral_or_unscoped_enum(ast->baseExpression->type)) {
    auto loc = ast->firstSourceLocation();
    auto elementTy = gen.convertType(ast->baseExpression->type);
    auto loadOp =
        gen.emitter_.load(loc, elementTy, expressionResult.value,
                          gen.getAlignment(ast->baseExpression->type));
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto oneOp = gen.emitter_.constantInt(
        loc, resultTy, ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1);
    auto addOp =
        gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, loadOp, oneOp);
    gen.emitter_.store(loc, addOp, expressionResult.value,
                       gen.getAlignment(ast->baseExpression->type));
    return {loadOp};
  }
  if (gen.traits.is_floating_point(ast->baseExpression->type)) {
    auto loc = ast->firstSourceLocation();
    auto ptrTy = gen.emitter_.typeOf(expressionResult.value);
    auto elementTy = gen.emitter_.elementType(ptrTy);
    auto loadOp =
        gen.emitter_.load(loc, elementTy, expressionResult.value,
                          gen.getAlignment(ast->baseExpression->type));
    auto resultTy = gen.convertType(ast->baseExpression->type);

    ir::ValueRef one;
    double v = ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1;

    switch (gen.traits.remove_cvref(ast->baseExpression->type)->kind()) {
      case TypeKind::kFloat:
        one = gen.emitter_.constantLiteral(
            ast->opLoc, gen.convertType(ast->baseExpression->type),
            ir::Initializer::floatingValue(
                gen.emitter_.floatingType(ir::FloatKind::Single), v));
        break;

      case TypeKind::kDouble:
        one = gen.emitter_.constantLiteral(
            ast->opLoc, gen.convertType(ast->baseExpression->type),
            ir::Initializer::floatingValue(
                gen.emitter_.floatingType(ir::FloatKind::Double), v));
        break;

      case TypeKind::kLongDouble:
        one = gen.emitter_.constantLiteral(
            ast->opLoc, gen.convertType(ast->baseExpression->type),
            ir::Initializer::floatingValue(
                gen.emitter_.floatingType(ir::FloatKind::Double), v));
        break;

      default:
        auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                   "unsupported float type");
        return {op};
    }

    auto addOp =
        gen.emitter_.binaryOp(loc, ir::BinaryOp::AddFloat, loadOp, one);
    gen.emitter_.store(loc, addOp, expressionResult.value,
                       gen.getAlignment(ast->baseExpression->type));
    return {loadOp};
  }
  if (gen.traits.is_pointer(ast->baseExpression->type)) {
    auto loc = ast->firstSourceLocation();
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto loadOp =
        gen.emitter_.load(loc, resultTy, expressionResult.value,
                          gen.getAlignment(ast->baseExpression->type));
    auto intTy = gen.emitter_.integerType(32);
    auto oneOp = gen.emitter_.constantInt(
        loc, intTy, ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1);
    auto addOp = gen.emitter_.pointerAdd(loc, resultTy, loadOp, oneOp);
    gen.emitter_.store(loc, addOp, expressionResult.value,
                       gen.getAlignment(ast->baseExpression->type));
    return {loadOp};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  if (gen.dynamicCastNeedsRuntimeCheck(ast)) {
    return {gen.emitDynamicCast(ast), ast->valueCategory};
  }
  auto expressionResult = gen.expression(ast->expression);
  return expressionResult;
}

auto Codegen::ExpressionVisitor::operator()(BuiltinBitCastExpressionAST* ast)
    -> ExpressionResult {
  auto source = gen.expression(ast->expression);
  if (!source.value) return source;
  auto loc = ast->firstSourceLocation();
  auto sourceType = gen.convertType(ast->expression->type);
  auto targetType = gen.convertType(ast->type);
  auto alignment = std::max(gen.getAlignment(ast->expression->type),
                            gen.getAlignment(ast->type));
  auto storage = gen.emitter_.allocate(
      loc, gen.emitter_.pointerType(sourceType), alignment);
  if (is_glvalue(ast->expression) ||
      (gen.traits.is_class_or_union(ast->expression->type) &&
       gen.emitter_.typeKind(gen.emitter_.typeOf(source.value)) ==
           ir::TypeKind::Pointer))
    source.value = gen.emitter_.load(loc, sourceType, source.value,
                                     gen.getAlignment(ast->expression->type));
  gen.emitter_.store(loc, source.value, storage, alignment);
  auto target =
      gen.emitter_.bitcast(loc, gen.emitter_.pointerType(targetType), storage);
  return {gen.emitter_.load(loc, targetType, target, alignment)};
}

auto Codegen::ExpressionVisitor::operator()(BuiltinOffsetofExpressionAST* ast)
    -> ExpressionResult {
  if (ast->symbol) {
    auto loc = ast->firstSourceLocation();
    auto resultType = gen.convertType(ast->type);

    auto classType = unqualified_cast<ClassType>(ast->typeId->type);
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

    auto op = gen.emitter_.constantInt(loc, resultType, fieldInfo->offset);

    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  return {gen.emitTypeid(ast), ValueCategory::kLValue};
}

auto Codegen::ExpressionVisitor::operator()(TypeidOfTypeExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.traits.remove_reference(ast->typeId->type);
  return {gen.typeInfoAddress(loc, type), ValueCategory::kLValue};
}

auto Codegen::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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

  return {op};
}

auto Codegen::ExpressionVisitor::emitRealImag(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->opLoc;
  const std::uint32_t index = ast->op == TokenKind::T___REAL__ ? 0 : 1;

  auto complexType = unqualified_cast<ComplexType>(ast->expression->type);

  if (!complexType) {
    if (index == 0) return gen.expression(ast->expression);
    (void)gen.expression(ast->expression);
    return {gen.emitter_.constantZero(loc, gen.convertType(ast->type))};
  }

  auto operand = gen.expression(ast->expression);
  if (!operand.value) return {};

  if (operand.category == ValueCategory::kLValue ||
      operand.category == ValueCategory::kXValue) {
    auto address = gen.memberAddress(loc, operand.value,
                                     complexType->elementType(), index);
    return {address, operand.category};
  }

  return {emitComplexPart(loc, operand.value, complexType, index)};
}

auto Codegen::ExpressionVisitor::emitUnaryOpNot(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (unqualified_cast<BoolType>(ast->type)) {
    auto loc = ast->opLoc;
    auto expressionResult = gen.expression(ast->expression);
    auto resultType = gen.convertType(ast->type);
    auto c1 = gen.emitter_.constantInt(loc, resultType, 1);
    auto op = gen.emitter_.binaryOp(loc, ir::BinaryOp::XorInt,
                                    expressionResult.value, c1);
    return {op};
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitUnaryOpMinus(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);
  auto loc = ast->opLoc;

  if (auto complexType = unqualified_cast<ComplexType>(ast->type)) {
    auto elementType = complexType->elementType();
    auto zero = gen.emitter_.constantZero(loc, gen.convertType(elementType));
    auto real = emitBinaryArithmeticOp(
        loc, TokenKind::T_MINUS, gen.convertType(elementType), elementType,
        zero, emitComplexPart(loc, expressionResult.value, complexType, 0));
    auto imag = emitBinaryArithmeticOp(
        loc, TokenKind::T_MINUS, gen.convertType(elementType), elementType,
        zero, emitComplexPart(loc, expressionResult.value, complexType, 1));
    if (!real.value || !imag.value) return {};
    return {makeComplexValue(loc, complexType, real.value, imag.value)};
  }

  if (gen.traits.is_floating_point(ast->type)) {
    auto op = gen.emitter_.negateFloat(loc, resultType, expressionResult.value);

    return {op};
  }

  if (gen.traits.is_integral_or_unscoped_enum(ast->type)) {
    auto zero = gen.emitter_.constantInt(loc, resultType, 0);
    auto op = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, zero,
                                    expressionResult.value);

    return {op};
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitUnaryOpTilde(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  auto loc = ast->opLoc;

  if (auto complexType = unqualified_cast<ComplexType>(ast->type)) {
    auto elementType = complexType->elementType();
    auto zero = gen.emitter_.constantZero(loc, gen.convertType(elementType));
    auto imag = emitBinaryArithmeticOp(
        loc, TokenKind::T_MINUS, gen.convertType(elementType), elementType,
        zero, emitComplexPart(loc, expressionResult.value, complexType, 1));
    if (!imag.value) return {};
    return {makeComplexValue(
        loc, complexType,
        emitComplexPart(loc, expressionResult.value, complexType, 0),
        imag.value)};
  }
  auto allOnes = gen.emitter_.constantInt(loc, resultType, -1);
  auto op = gen.emitter_.binaryOp(loc, ir::BinaryOp::XorInt,
                                  expressionResult.value, allOnes);

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrFloat(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  ir::ValueRef one;

  switch (gen.traits.remove_cvref(ast->expression->type)->kind()) {
    case TypeKind::kFloat:
      one = gen.emitter_.constantLiteral(
          ast->opLoc, gen.convertType(ast->expression->type),
          ir::Initializer::floatingValue(
              gen.emitter_.floatingType(ir::FloatKind::Single), 1.0));
      break;

    case TypeKind::kDouble:
      one = gen.emitter_.constantLiteral(
          ast->opLoc, gen.convertType(ast->expression->type),
          ir::Initializer::floatingValue(
              gen.emitter_.floatingType(ir::FloatKind::Double), 1.0));
      break;

    case TypeKind::kLongDouble:
      one = gen.emitter_.constantLiteral(
          ast->opLoc, gen.convertType(ast->expression->type),
          ir::Initializer::floatingValue(
              gen.emitter_.floatingType(ir::FloatKind::Double), 1.0));
      break;

    default:
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "unsupported float type")};
  }

  auto loc = ast->opLoc;
  auto resultType = gen.convertType(ast->type);

  auto loadOp = gen.emitter_.load(loc, resultType, expressionResult.value,
                                  gen.getAlignment(ast->expression->type));

  ir::ValueRef addOp;

  if (ast->op == TokenKind::T_MINUS_MINUS)
    addOp = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubFloat, loadOp, one);
  else
    addOp = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddFloat, loadOp, one);

  gen.emitter_.store(loc, addOp, expressionResult.value,
                     gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = gen.emitter_.load(loc, resultType, expressionResult.value,
                              gen.getAlignment(ast->expression->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrIntegral(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  auto loc = ast->opLoc;

  auto targetType = gen.convertType(ast->expression->type);
  auto oneOp = gen.emitter_.constantInt(loc, targetType, 1);

  auto resultType = gen.convertType(ast->type);

  auto loadOp = gen.emitter_.load(loc, resultType, expressionResult.value,
                                  gen.getAlignment(ast->expression->type));

  ir::ValueRef addOp;

  if (ast->op == TokenKind::T_MINUS_MINUS)
    addOp = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, loadOp, oneOp);
  else
    addOp = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, loadOp, oneOp);

  gen.emitter_.store(loc, addOp, expressionResult.value,
                     gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = gen.emitter_.load(loc, resultType, expressionResult.value,
                              gen.getAlignment(ast->expression->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecrPointer(
    UnaryExpressionAST* ast, ExpressionResult expressionResult)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto intTy = gen.emitter_.integerType(32);
  auto one = gen.emitter_.constantInt(
      loc, intTy, ast->op == TokenKind::T_MINUS_MINUS ? -1 : 1);
  auto resultType = gen.convertType(ast->expression->type);
  auto loadOp = gen.emitter_.load(loc, resultType, expressionResult.value,
                                  gen.getAlignment(ast->expression->type));
  auto addOp = gen.emitter_.pointerAdd(loc, resultType, loadOp, one);
  gen.emitter_.store(loc, addOp, expressionResult.value,
                     gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = gen.emitter_.load(loc, resultType, expressionResult.value,
                              gen.getAlignment(ast->expression->type));
  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecr(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->symbol) {
    if (auto result = emitBitFieldIncrDecr(ast->opLoc, ast->expression, ast->op,
                                           /*postfix=*/false))
      return *result;
  }

  auto expressionResult = gen.expression(ast->expression);

  if (ast->symbol) {
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {},
                          ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult},
                          false, ast);
    }
  }

  if (gen.traits.is_atomic(ast->expression->type)) {
    return emitAtomicIncrDecr(ast->opLoc, ast->op, ast->expression->type,
                              expressionResult.value, /*postfix=*/false);
  }

  if (gen.traits.is_floating_point(ast->expression->type)) {
    return emitUnaryOpIncrDecrFloat(ast, expressionResult);
  }

  if (gen.traits.is_arithmetic(ast->expression->type)) {
    return emitUnaryOpIncrDecrIntegral(ast, expressionResult);
  }

  if (gen.traits.is_pointer(ast->expression->type)) {
    return emitUnaryOpIncrDecrPointer(ast, expressionResult);
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  return {gen.emitter_.labelAddress(loc, gen.convertType(ast->type),
                                    ast->identifier->name(), gen.function_)};
}

auto Codegen::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (ast->op == TokenKind::T_MINUS_MINUS ||
      ast->op == TokenKind::T_PLUS_PLUS) {
    return emitUnaryOpIncrDecr(ast);
  }

  if (ast->symbol) {
    auto expressionResult = gen.expression(ast->expression);
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {},
                          ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult},
                          false, ast);
    }
  }

  switch (ast->op) {
    case TokenKind::T_EXCLAIM:
      return emitUnaryOpNot(ast);

    case TokenKind::T___REAL__:
    case TokenKind::T___IMAG__:
      return emitRealImag(ast);

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

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  if (auto size = ast->value) {
    auto resultlType = gen.convertType(ast->type);
    auto loc = ast->firstSourceLocation();
    auto op = gen.emitter_.constantInt(loc, resultlType, size.value());
    return {op};
  }

  if (ast->expression && ast->expression->type) {
    auto loc = ast->firstSourceLocation();
    auto resultType = gen.convertType(ast->type);
    ir::ValueRef totalElements;
    const Type* cur = ast->expression->type;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      if (!countResult.value) break;
      auto countVal = countResult.value;
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(countVal)) ==
           ir::TypeKind::Pointer)) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal = gen.emitter_.load(loc, valueType, countVal,
                                     gen.getAlignment(vla->size()->type));
      }
      if (gen.emitter_.typeOf(countVal) != resultType)
        countVal = gen.emitter_.signExtend(vla->size()->firstSourceLocation(),
                                           countVal, resultType);
      totalElements = totalElements
                          ? gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt,
                                                  totalElements, countVal)
                          : countVal;
      cur = vla->elementType();
    }
    if (totalElements) {
      auto leafSize = static_cast<int64_t>(
          gen.control()->memoryLayout()->sizeOf(cur).value_or(1));
      if (leafSize > 1) {
        auto leafConst = gen.emitter_.constantInt(loc, resultType, leafSize);
        totalElements = gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt,
                                              totalElements, leafConst);
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
    auto loc = ast->firstSourceLocation();
    auto op = gen.emitter_.constantInt(loc, resultlType, size.value());
    return {op};
  }

  auto typeIdType = ast->typeId ? ast->typeId->type : nullptr;
  if (typeIdType) {
    auto loc = ast->firstSourceLocation();
    auto resultType = gen.convertType(ast->type);
    ir::ValueRef totalElements;
    const Type* cur = typeIdType;
    while (auto vla = type_cast<UnresolvedBoundedArrayType>(cur)) {
      auto countResult = gen.expression(vla->size());
      if (!countResult.value) break;
      auto countVal = countResult.value;
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(countVal)) ==
           ir::TypeKind::Pointer)) {
        auto valueType = gen.convertType(vla->size()->type);
        countVal = gen.emitter_.load(loc, valueType, countVal,
                                     gen.getAlignment(vla->size()->type));
      }
      if (gen.emitter_.typeOf(countVal) != resultType)
        countVal = gen.emitter_.signExtend(vla->size()->firstSourceLocation(),
                                           countVal, resultType);
      totalElements = totalElements
                          ? gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt,
                                                  totalElements, countVal)
                          : countVal;
      cur = vla->elementType();
    }
    if (totalElements) {
      auto leafSize = static_cast<int64_t>(
          gen.control()->memoryLayout()->sizeOf(cur).value_or(1));
      if (leafSize > 1) {
        auto leafConst = gen.emitter_.constantInt(loc, resultType, leafSize);
        totalElements = gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt,
                                              totalElements, leafConst);
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
    auto loc = ast->firstSourceLocation();
    auto op = gen.emitter_.constantInt(loc, resultlType, alignment);
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  if (ast->expression && ast->expression->type) {
    auto memoryLayout = control()->memoryLayout();
    auto alignment = memoryLayout->alignmentOf(ast->expression->type).value();
    auto resultlType = gen.convertType(ast->type);
    auto loc = ast->firstSourceLocation();
    auto op = gen.emitter_.constantInt(loc, resultlType, alignment);
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  if (ast->value.has_value()) {
    auto resultType = gen.convertType(ast->type);
    auto loc = ast->firstSourceLocation();
    auto op = gen.emitter_.constantInt(loc, resultType, *ast->value ? 1 : 0);
    return {op};
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::loadReferenceBinding(SourceLocation loc, const Type* type,
                                   ir::ValueRef value) -> ir::ValueRef {
  if (!traits.is_reference(type)) return value;
  return emitter_.load(loc, convertType(type), value, getAlignment(type));
}

auto Codegen::requiresZeroInitialization(const Type* type,
                                         FunctionSymbol* constructor) -> bool {
  return traits.requires_zero_initialization(type, constructor);
}

void Codegen::emitZeroInitialization(SourceLocation loc, ir::ValueRef address,
                                     const Type* type) {
  auto size = control()->memoryLayout()->sizeOf(type);
  if (!size) return;
  emitter_.memsetZero(loc, address, *size);
}

auto Codegen::isReservedPlacementAllocation(FunctionSymbol* symbol) -> bool {
  if (!symbol || symbol_cast<ClassSymbol>(symbol->parent())) return false;

  auto functionType = type_cast<FunctionType>(symbol->type());
  if (!functionType || functionType->parameterTypes().size() != 2) return false;

  auto pointerType = type_cast<PointerType>(functionType->parameterTypes()[1]);
  return pointerType && traits.is_void(pointerType->elementType());
}

auto Codegen::arrayCookieSize(const Type* elementType) -> std::uint64_t {
  if (traits.has_trivial_destructor(elementType)) return 0;
  return getSize(control()->getSizeType());
}

auto Codegen::arrayCookiePrefixSize(const Type* elementType) -> std::uint64_t {
  auto cookieSize = arrayCookieSize(elementType);
  if (!cookieSize) return 0;
  return std::max(cookieSize, getAlignment(elementType));
}

auto Codegen::arrayElementCount(SourceLocation loc, const Type* allocatedType,
                                ir::TypeRef countType) -> ir::ValueRef {
  std::uint64_t constantExtents = 1;
  ir::ValueRef runtimeExtent;

  for (auto type = traits.remove_cv(allocatedType); traits.is_array(type);) {
    if (auto bounded = type_cast<BoundedArrayType>(type)) {
      constantExtents *= bounded->size();
      type = traits.remove_cv(bounded->elementType());
      continue;
    }

    if (auto unresolved = type_cast<UnresolvedBoundedArrayType>(type)) {
      auto sizeResult = expression(unresolved->size());
      auto value = sizeResult.value;

      if (value &&
          emitter_.typeKind(emitter_.typeOf(value)) == ir::TypeKind::Pointer) {
        value = emitter_.load(loc, convertType(unresolved->size()->type), value,
                              getAlignment(unresolved->size()->type));
      }

      if (value && emitter_.typeOf(value) != countType) {
        const auto fromWidth = emitter_.scalarWidth(emitter_.typeOf(value));
        const auto toWidth = emitter_.scalarWidth(countType);
        value = fromWidth < toWidth ? emitter_.zeroExtend(loc, value, countType)
                                    : emitter_.truncate(loc, value, countType);
      }

      runtimeExtent = value;
      type = traits.remove_cv(unresolved->elementType());
      continue;
    }

    if (auto unbounded = type_cast<UnboundedArrayType>(type)) {
      type = traits.remove_cv(unbounded->elementType());
      continue;
    }

    break;
  }

  auto constantValue = emitter_.constantInt(
      loc, countType, static_cast<std::int64_t>(constantExtents));

  if (!runtimeExtent) return constantValue;
  if (constantExtents == 1) return runtimeExtent;

  return emitter_.binaryOp(loc, ir::BinaryOp::MulInt, runtimeExtent,
                           constantValue);
}

void Codegen::emitArrayLoop(SourceLocation loc, ir::ValueRef base,
                            const Type* elementType, ir::ValueRef count,
                            bool reverse,
                            const std::function<void(ir::ValueRef)>& body) {
  if (!base || !count) return;

  auto countType = emitter_.typeOf(count);
  auto elementPtrType = emitter_.pointerType(convertType(elementType));
  auto countAlignment = getAlignment(control()->getSizeType());

  auto index =
      emitter_.allocate(loc, emitter_.pointerType(countType), countAlignment);

  emitter_.store(loc, reverse ? count : emitter_.constantZero(loc, countType),
                 index, countAlignment);

  auto conditionBlock = newBlock();
  auto bodyBlock = newBlock();
  auto endBlock = newBlock();

  branch(loc, conditionBlock);

  emitter_.setInsertionBlock(conditionBlock);
  auto position = emitter_.load(loc, countType, index, countAlignment);
  auto more =
      reverse ? emitter_.compareInt(loc, ir::IntPredicate::NotEqual, position,
                                    emitter_.constantZero(loc, countType))
              : emitter_.compareInt(loc, ir::IntPredicate::UnsignedLess,
                                    position, count);
  emitter_.condBranch(loc, more, bodyBlock, endBlock);

  emitter_.setInsertionBlock(bodyBlock);
  auto one = emitter_.constantInt(loc, countType, 1);
  auto elementIndex =
      reverse ? emitter_.binaryOp(loc, ir::BinaryOp::SubInt, position, one)
              : position;
  emitter_.store(
      loc,
      reverse ? elementIndex
              : emitter_.binaryOp(loc, ir::BinaryOp::AddInt, position, one),
      index, countAlignment);
  body(emitter_.pointerAdd(loc, elementPtrType, base, elementIndex));
  branch(loc, conditionBlock);

  emitter_.setInsertionBlock(endBlock);
}

auto Codegen::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto allocatedType = ast->objectType;
  if (!allocatedType) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "new: missing objectType")};
  }

  const bool isArrayNew = gen.traits.is_array(allocatedType);

  auto objectType =
      isArrayNew ? gen.traits.remove_all_extents(allocatedType) : allocatedType;

  auto sizeTy = gen.convertType(control()->getSizeType());
  auto objectIrType = gen.convertType(objectType);
  auto ptrType = gen.emitter_.pointerType(objectIrType);

  const auto objectSize = gen.getSize(objectType);
  const bool hasCookie =
      isArrayNew && !gen.isReservedPlacementAllocation(ast->symbol);
  const auto prefixSize = hasCookie ? gen.arrayCookiePrefixSize(objectType) : 0;
  const auto cookieSize = hasCookie ? gen.arrayCookieSize(objectType) : 0;

  ir::ValueRef elementCount;
  ir::ValueRef sizeVal;

  if (isArrayNew) {
    elementCount = gen.arrayElementCount(loc, allocatedType, sizeTy);
    auto elementSize = gen.emitter_.constantInt(
        loc, sizeTy, static_cast<std::int64_t>(objectSize));
    sizeVal = gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, elementCount,
                                    elementSize);
    if (prefixSize) {
      auto prefix = gen.emitter_.constantInt(
          loc, sizeTy, static_cast<std::int64_t>(prefixSize));
      sizeVal =
          gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, sizeVal, prefix);
    }
  } else {
    sizeVal = gen.emitter_.constantInt(
        loc, sizeTy, static_cast<std::int64_t>(objectSize ? objectSize : 1));
  }

  std::vector<ExpressionResult> allocationArguments;
  allocationArguments.push_back({sizeVal});

  for (auto node : ListView{
           ast->newPlacement ? ast->newPlacement->expressionList : nullptr})
    allocationArguments.push_back(gen.expression(node));

  if (!ast->symbol) {
    return {gen.emitTodoExpr(loc, "new: missing allocation function")};
  }

  auto allocation =
      gen.emitCall(ast->newLoc, ast->symbol, {}, std::move(allocationArguments))
          .value;

  if (!allocation) return {};

  auto rawPtr = allocation;

  if (prefixSize) {
    auto byteType = gen.emitter_.integerType(8);
    auto bytePtrType = gen.emitter_.pointerType(byteType);
    auto base = gen.emitter_.bitcast(loc, bytePtrType, allocation);

    auto cookieOffset = gen.emitter_.constantInt(
        loc, sizeTy, static_cast<std::int64_t>(prefixSize - cookieSize));
    auto cookieAddress = gen.emitter_.bitcast(
        loc, gen.emitter_.pointerType(sizeTy),
        gen.emitter_.pointerAdd(loc, bytePtrType, base, cookieOffset));

    gen.emitter_.store(loc, elementCount, cookieAddress,
                       gen.getAlignment(control()->getSizeType()));

    auto prefix = gen.emitter_.constantInt(
        loc, sizeTy, static_cast<std::int64_t>(prefixSize));
    rawPtr = gen.emitter_.pointerAdd(loc, bytePtrType, base, prefix);
  }

  rawPtr = gen.emitter_.bitcast(loc, ptrType, rawPtr);

  std::vector<ExpressionResult> ctorArgs;
  if (ast->constructorSymbol) {
    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      for (auto node : ListView{paren->expressionList})
        ctorArgs.push_back(gen.expression(node));
    } else if (auto braced =
                   ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
      if (auto bracedList =
              ast_cast<BracedInitListAST>(braced->bracedInitList)) {
        for (auto node : ListView{bracedList->expressionList})
          ctorArgs.push_back(gen.expression(node));
      }
    }
  }

  const bool valueInitialized = ast->newInitalizer && ctorArgs.empty();

  if (isArrayNew) {
    if (ast->constructorSymbol) {
      const bool zeroFirst =
          valueInitialized &&
          gen.requiresZeroInitialization(objectType, ast->constructorSymbol);

      gen.emitArrayLoop(loc, rawPtr, objectType, elementCount,
                        /*reverse=*/false, [&](ir::ValueRef element) {
                          if (zeroFirst)
                            gen.emitZeroInitialization(loc, element,
                                                       objectType);
                          (void)gen.emitCtorCall(loc, ast->constructorSymbol,
                                                 element, ctorArgs,
                                                 /*completeObject=*/true);
                        });
      return {rawPtr};
    }

    if (auto braced = ast_cast<NewBracedInitializerAST>(ast->newInitalizer);
        braced && braced->bracedInitList) {
      gen.arrayInit(rawPtr, allocatedType, braced->bracedInitList);
      return {rawPtr};
    }

    if (ast->newInitalizer && objectSize) {
      gen.emitArrayLoop(loc, rawPtr, objectType, elementCount,
                        /*reverse=*/false, [&](ir::ValueRef element) {
                          gen.emitter_.memsetZero(loc, element, objectSize);
                        });
    }

    return {rawPtr};
  }

  if (ast->constructorSymbol) {
    if (valueInitialized &&
        gen.requiresZeroInitialization(objectType, ast->constructorSymbol)) {
      gen.emitZeroInitialization(loc, rawPtr, objectType);
    }

    (void)gen.emitCtorCall(ast->newLoc, ast->constructorSymbol, rawPtr,
                           std::move(ctorArgs), /*completeObject=*/true);
  } else if (gen.traits.is_class_or_union(gen.traits.remove_cv(objectType))) {
    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      gen.emitAggregateInit(rawPtr, objectType, paren->expressionList, loc);
    } else if (auto braced =
                   ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
      if (auto bracedList =
              ast_cast<BracedInitListAST>(braced->bracedInitList)) {
        gen.emitAggregateInit(rawPtr, objectType, bracedList);
      }
    } else if (objectSize) {
      gen.emitter_.memsetZero(loc, rawPtr, objectSize);
    }
  } else if (ast->newInitalizer) {
    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      if (paren->expressionList) {
        auto initExpr = paren->expressionList->value;
        auto initVal = gen.expression(initExpr);
        auto val = initVal.value;
        if (initExpr->valueCategory == ValueCategory::kLValue) {
          auto loadedType = gen.convertType(initExpr->type);
          val = gen.emitter_.load(loc, loadedType, val,
                                  gen.getAlignment(initExpr->type));
        }
        gen.emitter_.store(loc, val, rawPtr, gen.getAlignment(objectType));
      } else if (objectSize) {
        gen.emitter_.memsetZero(loc, rawPtr, objectSize);
      }
    } else if (auto braced =
                   ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
      if (braced->bracedInitList) {
        auto initVal = gen.expression(braced->bracedInitList);
        gen.emitter_.store(loc, initVal.value, rawPtr,
                           gen.getAlignment(objectType));
      }
    }
  }

  return {rawPtr};
}

auto Codegen::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  auto ptrResult = gen.expression(ast->expression);
  if (!ptrResult.value) return {};

  auto ptrValue = ptrResult.value;

  if (ast->expression->valueCategory == ValueCategory::kLValue) {
    auto loadedType = gen.convertType(ast->expression->type);
    ptrValue = gen.emitter_.load(loc, loadedType, ptrValue,
                                 gen.getAlignment(ast->expression->type));
  }

  const Type* pointeeType = nullptr;
  if (auto ptrTy = unqualified_cast<PointerType>(ast->expression->type)) {
    pointeeType = gen.traits.remove_cv(ptrTy->elementType());
  }

  auto notNullBlock = gen.newBlock();
  auto endBlock = gen.newBlock();

  auto isNull = gen.emitter_.compareInt(
      loc, ir::IntPredicate::Equal,
      gen.emitter_.pointerToInt(loc, gen.convertType(control()->getSizeType()),
                                ptrValue),
      gen.emitter_.constantZero(loc,
                                gen.convertType(control()->getSizeType())));

  gen.emitter_.condBranch(loc, isNull, endBlock, notNullBlock);
  gen.emitter_.setInsertionBlock(notNullBlock);

  auto classType =
      pointeeType ? unqualified_cast<ClassType>(pointeeType) : nullptr;
  auto classSymbol = classType ? classType->symbol() : nullptr;
  auto destructor = classSymbol ? classSymbol->destructor() : nullptr;

  const bool isArrayDelete = static_cast<bool>(ast->lbracketLoc);

  auto deallocationPtr = ptrValue;
  ir::ValueRef deallocationSize;

  if (isArrayDelete) {
    const auto cookieSize = gen.arrayCookieSize(pointeeType);
    const auto prefixSize = gen.arrayCookiePrefixSize(pointeeType);

    if (cookieSize) {
      auto sizeTy = gen.convertType(control()->getSizeType());
      auto byteType = gen.emitter_.integerType(8);
      auto bytePtrType = gen.emitter_.pointerType(byteType);
      auto base = gen.emitter_.bitcast(loc, bytePtrType, ptrValue);

      auto cookieOffset = gen.emitter_.constantInt(
          loc, sizeTy, -static_cast<std::int64_t>(cookieSize));
      auto cookieAddress = gen.emitter_.bitcast(
          loc, gen.emitter_.pointerType(sizeTy),
          gen.emitter_.pointerAdd(loc, bytePtrType, base, cookieOffset));

      auto count =
          gen.emitter_.load(loc, sizeTy, cookieAddress,
                            gen.getAlignment(control()->getSizeType()));

      if (destructor) {
        gen.emitArrayLoop(loc, ptrValue, pointeeType, count, /*reverse=*/true,
                          [&](ir::ValueRef element) {
                            (void)gen.emitCall(
                                ast->deleteLoc,
                                Codegen::completeObjectDtor(destructor),
                                {element}, {});
                          });
      }

      auto prefixOffset = gen.emitter_.constantInt(
          loc, sizeTy, -static_cast<std::int64_t>(prefixSize));
      deallocationPtr =
          gen.emitter_.pointerAdd(loc, bytePtrType, base, prefixOffset);

      auto elementSize = gen.emitter_.constantInt(
          loc, sizeTy,
          static_cast<std::int64_t>(
              control()->memoryLayout()->sizeOf(pointeeType).value_or(0)));
      deallocationSize = gen.emitter_.binaryOp(
          loc, ir::BinaryOp::AddInt,
          gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, count, elementSize),
          gen.emitter_.constantInt(loc, sizeTy,
                                   static_cast<std::int64_t>(prefixSize)));
    }
  } else if (destructor && destructor->isVirtual()) {
    auto i8Type = gen.emitter_.integerType(8);
    auto i8PtrType = gen.emitter_.pointerType(i8Type);
    auto i8PtrPtrType = gen.emitter_.pointerType(i8PtrType);

    auto vptrFieldPtr = gen.memberAddress(loc, ptrValue, i8PtrPtrType, 0);
    auto vtablePtr = gen.emitter_.load(loc, i8PtrPtrType, vptrFieldPtr, 8);

    int slotIndex = gen.vtableSlotIndex(destructor) + 1;

    auto intTy = gen.convertType(control()->getIntType());
    auto offsetOp = gen.emitter_.constantInt(loc, intTy, slotIndex);
    auto funcPtrAddr =
        gen.emitter_.pointerAdd(loc, i8PtrPtrType, vtablePtr, offsetOp);
    auto funcPtr = gen.emitter_.load(loc, i8PtrType, funcPtrAddr, 8);

    const ir::ValueRef dtorArguments[] = {ptrValue};
    ir::CallInfo dtorCall;
    dtorCall.indirectCallee = funcPtr;
    dtorCall.arguments = dtorArguments;
    (void)gen.emitter_.call(gen.implicitLocation(loc), dtorCall);

    gen.branch(loc, endBlock);
    gen.emitter_.setInsertionBlock(endBlock);
    return {};
  } else if (destructor) {
    (void)gen.emitCall(ast->deleteLoc, Codegen::completeObjectDtor(destructor),
                       {ptrValue}, {});
  }

  if (ast->symbol) {
    std::vector<ExpressionResult> arguments{{deallocationPtr}};

    auto signature = deallocationSignatureOf(gen.unit_, ast->symbol);
    const auto& parameterTypes =
        type_cast<FunctionType>(ast->symbol->type())->parameterTypes();

    auto appendArgument = [&](std::uint64_t value) {
      auto type = gen.convertType(parameterTypes[arguments.size()]);
      arguments.push_back({gen.emitter_.constantInt(
          loc, type, static_cast<std::int64_t>(value))});
    };

    auto memoryLayout = control()->memoryLayout();

    if (signature->hasSize) {
      if (deallocationSize)
        arguments.push_back({deallocationSize});
      else
        appendArgument(memoryLayout->sizeOf(pointeeType).value_or(0));
    }

    if (signature->hasAlignment)
      appendArgument(memoryLayout->alignmentOf(pointeeType).value_or(1));

    (void)gen.emitCall(ast->deleteLoc, ast->symbol, {}, std::move(arguments));
  }

  gen.branch(loc, endBlock);
  gen.emitter_.setInsertionBlock(endBlock);

  return {};
}

auto Codegen::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);
  if (!expressionResult.value) return expressionResult;

  auto loc = ast->firstSourceLocation();
  auto resultType = gen.convertType(ast->type);
  if (!resultType) return expressionResult;

  const auto sourceLoc = ast->firstSourceLocation();
  const auto srcType = gen.emitter_.typeOf(expressionResult.value);
  const auto resultKind = gen.emitter_.typeKind(resultType);
  const auto srcKind = gen.emitter_.typeKind(srcType);

  if (resultKind == ir::TypeKind::Pointer && srcKind == ir::TypeKind::Integer) {
    auto wordType = gen.emitter_.integerType(64);
    auto intVal = expressionResult.value;
    const auto srcWidth = gen.emitter_.scalarWidth(srcType);
    if (srcWidth < 64) {
      intVal = gen.traits.is_signed(ast->expression->type)
                   ? gen.emitter_.signExtend(sourceLoc, intVal, wordType)
                   : gen.emitter_.zeroExtend(sourceLoc, intVal, wordType);
    } else if (srcWidth > 64) {
      intVal = gen.emitter_.truncate(sourceLoc, intVal, wordType);
    }
    return {gen.emitter_.intToPointer(loc, resultType, intVal)};
  }

  if (resultKind == ir::TypeKind::Integer && srcKind == ir::TypeKind::Pointer) {
    auto wordType = gen.emitter_.integerType(64);
    auto ptrInt =
        gen.emitter_.pointerToInt(loc, wordType, expressionResult.value);
    const auto dstWidth = gen.emitter_.scalarWidth(resultType);
    if (dstWidth < 64)
      return {gen.emitter_.truncate(sourceLoc, ptrInt, resultType)};
    if (dstWidth > 64)
      return {gen.emitter_.zeroExtend(sourceLoc, ptrInt, resultType)};
    return {ptrInt};
  }

  if (resultKind == ir::TypeKind::Pointer && srcKind == ir::TypeKind::Pointer &&
      resultType != srcType) {
    return {gen.emitter_.bitcast(loc, resultType, expressionResult.value)};
  }

  return expressionResult;
}

auto Codegen::ExpressionVisitor::emitLValueToRValueConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  auto expressionResult = gen.expression(ast->expression);

  if (gen.traits.is_reference(ast->expression->type)) {
    return {expressionResult.value};
  }

  if (expressionResult.isRValueMaterialized) {
    return {expressionResult.value};
  }

  if (expressionResult.category != ValueCategory::kLValue &&
      expressionResult.category != ValueCategory::kXValue) {
    return {expressionResult.value};
  }

  if (!(gen.emitter_.typeKind(gen.emitter_.typeOf(expressionResult.value)) ==
        ir::TypeKind::Pointer)) {
    return {expressionResult.value};
  }

  if (gen.traits.is_atomic(ast->type)) {
    return {emitAtomicLoad(loc, ast->type, expressionResult.value)};
  }

  auto resultType = gen.convertType(ast->type);

  auto op = gen.emitter_.load(loc, resultType, expressionResult.value,
                              gen.getAlignment(ast->type));

  return {op};
}

auto Codegen::ExpressionVisitor::emitNumericConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->castKind) {
    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kIntegralPromotion: {
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(expressionResult.value)) ==
           ir::TypeKind::Pointer)) {
        auto intVal =
            gen.emitter_.pointerToInt(loc, resultType, expressionResult.value);
        return {intVal};
      }

      if (is_bool(ast->type)) {
        auto zero = gen.emitter_.constantInt(
            loc, gen.emitter_.typeOf(expressionResult.value), 0);
        return {gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual,
                                        expressionResult.value, zero)};
      }

      if (is_bool(ast->expression->type)) {
        return {
            gen.emitter_.zeroExtend(loc, expressionResult.value, resultType)};
      }

      auto srcType = gen.emitter_.typeOf(expressionResult.value);

      auto dstType = resultType;

      if (gen.emitter_.scalarWidth(srcType) ==
          gen.emitter_.scalarWidth(dstType)) {
        return expressionResult;
      }

      if (gen.emitter_.scalarWidth(dstType) <
          gen.emitter_.scalarWidth(srcType)) {
        return {gen.emitter_.truncate(loc, expressionResult.value, resultType)};
      }

      if (gen.traits.is_signed(ast->expression->type)) {
        return {
            gen.emitter_.signExtend(loc, expressionResult.value, resultType)};
      }

      return {gen.emitter_.zeroExtend(loc, expressionResult.value, resultType)};
    }

    case ImplicitCastKind::kFloatingPointPromotion:
    case ImplicitCastKind::kFloatingPointConversion: {
      auto srcWidth =
          gen.emitter_.scalarWidth(gen.emitter_.typeOf(expressionResult.value));
      auto dstWidth = gen.emitter_.scalarWidth(resultType);

      if (srcWidth == dstWidth) {
        return expressionResult;
      }

      if (srcWidth < dstWidth) {
        auto op =
            gen.emitter_.floatExtend(loc, expressionResult.value, resultType);
        return {op};
      }

      auto op =
          gen.emitter_.floatTruncate(loc, expressionResult.value, resultType);

      return {op};
    }

    case ImplicitCastKind::kFloatingIntegralConversion:
      if (is_bool(ast->type)) {
        auto zero = gen.emitter_.constantZero(
            loc, gen.emitter_.typeOf(expressionResult.value));

        auto op = gen.emitter_.compareFloat(
            loc, ir::FloatPredicate::UnorderedNotEqual, expressionResult.value,
            zero);

        return {op};
      }

      if (gen.traits.is_floating_point(ast->type)) {
        if (gen.traits.is_signed(ast->expression->type)) {
          auto op = gen.emitter_.signedIntToFloat(loc, expressionResult.value,
                                                  resultType);
          return {op};
        }

        auto op = gen.emitter_.unsignedIntToFloat(loc, expressionResult.value,
                                                  resultType);

        return {op};
      }

      if (gen.traits.is_integral(ast->type)) {
        if (gen.traits.is_signed(ast->type)) {
          auto op = gen.emitter_.floatToSignedInt(loc, expressionResult.value,
                                                  resultType);
          return {op};
        }

        auto op = gen.emitter_.floatToUnsignedInt(loc, expressionResult.value,
                                                  resultType);

        return {op};
      }
      break;

    default:
      break;
  }
  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::emitVectorSplat(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto scalar = gen.expression(ast->expression);
  if (!scalar.value) return scalar;
  auto resultType = gen.convertType(ast->type);
  return {gen.emitter_.vectorSplat(loc, resultType, scalar.value)};
}

auto Codegen::ExpressionVisitor::emitVectorConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto operand = gen.expression(ast->expression);
  if (!operand.value) return operand;
  auto resultType = gen.convertType(ast->type);
  if (gen.emitter_.typeOf(operand.value) == resultType) return operand;
  return {gen.emitter_.bitcast(loc, resultType, operand.value)};
}

auto Codegen::ExpressionVisitor::emitPointerConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto expressionResult = gen.expression(ast->expression);
  auto resultType = gen.convertType(ast->type);

  switch (ast->castKind) {
    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kFunctionPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
      return expressionResult;

    case ImplicitCastKind::kPointerConversion: {
      if (expressionResult.value &&
          (gen.emitter_.typeKind(gen.emitter_.typeOf(expressionResult.value)) ==
           ir::TypeKind::Integer)) {
        auto op = gen.emitter_.nullPointer(loc, resultType);

        return {op};
      }

      if (expressionResult.value &&
          gen.emitter_.typeOf(expressionResult.value) != resultType) {
        auto value =
            gen.emitter_.bitcast(loc, resultType, expressionResult.value);
        return {value};
      }

      return expressionResult;
    }

    case ImplicitCastKind::kArrayToPointerConversion: {
      auto op =
          gen.emitter_.arrayToPointer(loc, resultType, expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kBooleanConversion: {
      if (gen.traits.is_member_pointer(ast->expression->type)) {
        auto value = expressionResult.value;
        auto nullValue = gen.nullMemberObjectPointer();
        if (auto pointerType = unqualified_cast<MemberFunctionPointerType>(
                ast->expression->type)) {
          value =
              gen.memberFunctionPointerFields(loc, pointerType, value).first;
          nullValue = 0;
        }
        auto null = gen.emitter_.constantInt(loc, gen.emitter_.typeOf(value),
                                             nullValue);
        return {gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual, value,
                                        null)};
      }
      if (!gen.traits.is_pointer(ast->expression->type)) break;
      auto ptrIntTy = gen.emitter_.integerType(64);
      auto ptrInt =
          gen.emitter_.pointerToInt(loc, ptrIntTy, expressionResult.value);
      auto zero = gen.emitter_.constantInt(loc, ptrIntTy, 0);

      auto op = gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual, ptrInt,
                                        zero);

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
  auto loc = ast->firstSourceLocation();
  auto expressionResult = gen.expression(ast->expression);
  if (!expressionResult.value) return expressionResult;

  auto traits = gen.traits;
  auto sourceClass = unqualified_cast<ClassType>(
      traits.remove_pointer(traits.remove_reference(ast->expression->type)));
  auto targetClass = unqualified_cast<ClassType>(
      traits.remove_pointer(traits.remove_reference(ast->type)));

  ir::ValueRef value = expressionResult.value;
  if (sourceClass && targetClass &&
      sourceClass->symbol() != targetClass->symbol()) {
    value = gen.emitBaseClassAddress(loc, value, sourceClass->symbol(),
                                     targetClass->symbol());
  }

  if (traits.is_pointer(ast->type)) {
    auto resultType = gen.convertType(ast->type);
    if (gen.emitter_.typeOf(value) != resultType) {
      value = gen.emitter_.bitcast(loc, resultType, value);
    }
  }

  return {value};
}

auto Codegen::ExpressionVisitor::emitBaseToDerivedConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto expressionResult = gen.expression(ast->expression);
  if (!expressionResult.value) return expressionResult;

  auto traits = gen.traits;
  auto sourceClass = unqualified_cast<ClassType>(
      traits.remove_pointer(traits.remove_reference(ast->expression->type)));
  auto targetClass = unqualified_cast<ClassType>(
      traits.remove_pointer(traits.remove_reference(ast->type)));

  if (!sourceClass || !targetClass) return expressionResult;

  return {gen.emitDerivedClassAddress(loc, expressionResult.value,
                                      sourceClass->symbol(),
                                      targetClass->symbol())};
}

auto Codegen::ExpressionVisitor::emitPointerToMemberConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  if (auto targetType = type_cast<MemberFunctionPointerType>(ast->type)) {
    auto sourceType =
        ast->expression
            ? type_cast<MemberFunctionPointerType>(ast->expression->type)
            : nullptr;

    if (!sourceType) {
      (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
      return {gen.emitMemberFunctionPointerValue(loc, targetType, nullptr, 0)};
    }

    auto operand = gen.expression(ast->expression);
    if (!operand.value) return operand;

    auto adjustment = memberPointerBaseAdjustment(sourceType, targetType);
    if (!adjustment.has_value() || *adjustment == 0) return operand;

    auto [pointerField, adjustmentField] =
        gen.memberFunctionPointerFields(loc, sourceType, operand.value);

    auto wordType = gen.pointerSizedIntType();
    auto delta = gen.emitter_.constantInt(loc, wordType, *adjustment * 2);

    auto adjusted = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt,
                                          adjustmentField, delta);

    return {
        gen.makeMemberFunctionPointer(loc, targetType, pointerField, adjusted)};
  }

  auto resultType = gen.convertType(ast->type);

  auto nullValue =
      gen.emitter_.constantInt(loc, resultType, gen.nullMemberObjectPointer());

  auto sourceType =
      ast->expression
          ? type_cast<MemberObjectPointerType>(ast->expression->type)
          : nullptr;

  if (!sourceType) {
    (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
    return {nullValue};
  }

  auto operand = gen.expression(ast->expression);
  if (!operand.value) return operand;

  auto adjustment = memberPointerBaseAdjustment(
      sourceType, type_cast<MemberObjectPointerType>(ast->type));
  if (!adjustment.has_value() || *adjustment == 0) return operand;

  auto adjustmentValue = gen.emitter_.constantInt(loc, resultType, *adjustment);

  auto adjusted = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt,
                                        operand.value, adjustmentValue);

  auto isNull = gen.emitter_.compareInt(loc, ir::IntPredicate::Equal,
                                        operand.value, nullValue);

  return {gen.emitter_.select(loc, isNull, nullValue, adjusted)};
}

auto Codegen::ExpressionVisitor::emitUserDefinedConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

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

  if (!(gen.emitter_.typeKind(gen.emitter_.typeOf(objectValue)) ==
        ir::TypeKind::Pointer)) {
    auto sourceType = ast->expression->type;
    auto temp = gen.newTemp(sourceType, ast->firstSourceLocation());
    gen.emitter_.store(loc, objectValue, temp, gen.getAlignment(sourceType));
    objectValue = temp;
  }

  return gen.emitCall(ast->firstSourceLocation(), function, {objectValue}, {},
                      ast->isVirtualDispatch, ast);
}

auto Codegen::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
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

    case ImplicitCastKind::kBooleanConversion:
      if (gen.traits.is_complex(ast->expression->type))
        return emitComplexToBoolean(ast);
      [[fallthrough]];

    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kFunctionPointerConversion:
    case ImplicitCastKind::kArrayToPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
    case ImplicitCastKind::kPointerConversion:
      return emitPointerConversion(ast);

    case ImplicitCastKind::kAtomicToNonAtomic:
    case ImplicitCastKind::kNonAtomicToAtomic:
      return gen.expression(ast->expression);

    case ImplicitCastKind::kRealToComplexConversion:
    case ImplicitCastKind::kComplexToRealConversion:
    case ImplicitCastKind::kComplexConversion:
      return emitComplexConversion(ast);

    case ImplicitCastKind::kVectorSplat:
      return emitVectorSplat(ast);

    case ImplicitCastKind::kVectorConversion:
      return emitVectorConversion(ast);

    case ImplicitCastKind::kPointerToMemberConversion:
      return emitPointerToMemberConversion(ast);

    case ImplicitCastKind::kDerivedToBaseConversion:
      return emitDerivedToBaseConversion(ast);

    case ImplicitCastKind::kBaseToDerivedConversion:
      return emitBaseToDerivedConversion(ast);

    case ImplicitCastKind::kUserDefinedConversion:
      return emitUserDefinedConversion(ast);

    case ImplicitCastKind::kTemporaryMaterializationConversion: {
      auto inner = gen.expression(ast->expression);
      if (!inner.value) break;
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(inner.value)) ==
           ir::TypeKind::Pointer) &&
          !gen.traits.is_scalar(ast->type)) {
        return inner;
      }
      auto loc = ast->firstSourceLocation();
      auto temp = gen.newTemp(ast->type, ast->firstSourceLocation());
      gen.emitter_.store(loc, inner.value, temp, gen.getAlignment(ast->type));
      return {temp};
    }

    default:
      break;
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ConstExpressionAST* ast)
    -> ExpressionResult {
  if (format == ExpressionFormat::kSideEffect) return {};
  if (!ast->constValue) return gen.expression(ast->expression, format);

  auto loc = ast->firstSourceLocation();
  if (is_glvalue(ast)) {
    auto pointerType = gen.traits.add_pointer(ast->type);
    auto address = gen.emitConstInitValue(loc, pointerType, *ast->constValue);
    return {address};
  }

  auto value = gen.emitConstInitValue(loc, ast->type, *ast->constValue);
  return {value};
}

auto Codegen::ExpressionVisitor::operator()(
    ThreeWayComparisonExpressionAST* ast) -> ExpressionResult {
  if (!ast->comparison)
    return {
        gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};

  if (ast->comparison->symbol) return gen.expression(ast->comparison, format);

  auto left = gen.expression(ast->comparison->leftExpression);
  auto right = gen.expression(ast->comparison->rightExpression);
  return emitThreeWayComparison(ast, left, right);
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

    gen.emitter_.setInsertionBlock(continueBlock);
    {
      auto conditionalEvaluation = ConditionalEvaluation{gen};
      gen.condition(ast->rightExpression, trueBlock, falseBlock);
    }

    gen.emitter_.setInsertionBlock(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = gen.emitter_.constantInt(ast->opLoc, i1type, 1);

    gen.emitter_.store(ast->opLoc, trueValue, t,
                       gen.getAlignment(control()->getBoolType()));

    auto endLoc = lastTokenLocation(ast);
    gen.branch(endLoc, endBlock);

    gen.emitter_.setInsertionBlock(falseBlock);
    auto falseValue = gen.emitter_.constantInt(ast->opLoc, i1type, 0);
    gen.emitter_.store(ast->opLoc, falseValue, t,
                       gen.getAlignment(control()->getBoolType()));
    gen.branch(lastTokenLocation(ast), endBlock);

    gen.emitter_.setInsertionBlock(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = gen.emitter_.load(ast->opLoc, resultType, t,
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

    gen.emitter_.setInsertionBlock(continueBlock);
    {
      auto conditionalEvaluation = ConditionalEvaluation{gen};
      gen.condition(ast->rightExpression, trueBlock, falseBlock);
    }

    gen.emitter_.setInsertionBlock(trueBlock);

    auto i1type = gen.convertType(control()->getBoolType());

    auto trueValue = gen.emitter_.constantInt(ast->opLoc, i1type, 1);

    gen.emitter_.store(ast->opLoc, trueValue, t,
                       gen.getAlignment(control()->getBoolType()));

    auto endLoc = lastTokenLocation(ast);
    gen.branch(endLoc, endBlock);

    gen.emitter_.setInsertionBlock(falseBlock);
    auto falseValue = gen.emitter_.constantInt(ast->opLoc, i1type, 0);
    gen.emitter_.store(ast->opLoc, falseValue, t,
                       gen.getAlignment(control()->getBoolType()));
    gen.branch(lastTokenLocation(ast), endBlock);

    gen.emitter_.setInsertionBlock(endBlock);

    if (format == ExpressionFormat::kSideEffect) return {};

    auto resultType = gen.convertType(ast->type);
    auto loadOp = gen.emitter_.load(ast->opLoc, resultType, t,
                                    gen.getAlignment(control()->getBoolType()));
    return {loadOp};
  }

  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);

  if (ast->symbol) {
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, leftExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {leftExpressionResult, rightExpressionResult}, false,
                          ast);
    }
  }

  auto resultType = gen.convertType(ast->type);

  return binaryExpression(ast->opLoc, ast->op, resultType, ast->leftExpression,
                          ast->rightExpression, leftExpressionResult,
                          rightExpressionResult);
}

auto Codegen::ExpressionVisitor::emitComplexArithmeticOp(
    SourceLocation loc, TokenKind op, const ComplexType* complexType,
    ir::ValueRef left, ir::ValueRef right) -> ExpressionResult {
  auto elementType = complexType->elementType();
  auto irElementType = gen.convertType(elementType);

  auto a = emitComplexPart(loc, left, complexType, 0);
  auto b = emitComplexPart(loc, left, complexType, 1);
  auto c = emitComplexPart(loc, right, complexType, 0);
  auto d = emitComplexPart(loc, right, complexType, 1);

  auto element = [&](TokenKind elementOp, ir::ValueRef x,
                     ir::ValueRef y) -> ir::ValueRef {
    return emitBinaryArithmeticOp(loc, elementOp, irElementType, elementType, x,
                                  y)
        .value;
  };

  switch (op) {
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
      return {makeComplexValue(loc, complexType, element(op, a, c),
                               element(op, b, d))};

    case TokenKind::T_STAR: {
      auto ac = element(TokenKind::T_STAR, a, c);
      auto bd = element(TokenKind::T_STAR, b, d);
      auto ad = element(TokenKind::T_STAR, a, d);
      auto bc = element(TokenKind::T_STAR, b, c);
      return {makeComplexValue(loc, complexType,
                               element(TokenKind::T_MINUS, ac, bd),
                               element(TokenKind::T_PLUS, ad, bc))};
    }

    case TokenKind::T_SLASH: {
      auto cc = element(TokenKind::T_STAR, c, c);
      auto dd = element(TokenKind::T_STAR, d, d);
      auto denominator = element(TokenKind::T_PLUS, cc, dd);
      auto ac = element(TokenKind::T_STAR, a, c);
      auto bd = element(TokenKind::T_STAR, b, d);
      auto bc = element(TokenKind::T_STAR, b, c);
      auto ad = element(TokenKind::T_STAR, a, d);
      auto real = element(TokenKind::T_SLASH,
                          element(TokenKind::T_PLUS, ac, bd), denominator);
      auto imag = element(TokenKind::T_SLASH,
                          element(TokenKind::T_MINUS, bc, ad), denominator);
      return {makeComplexValue(loc, complexType, real, imag)};
    }

    default:
      break;
  }

  return {gen.emitTodoExpr(loc, "complex arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitComplexComparisonOp(
    SourceLocation loc, TokenKind op, const ComplexType* complexType,
    ir::ValueRef left, ir::ValueRef right) -> ExpressionResult {
  auto elementType = complexType->elementType();
  auto boolType = gen.convertType(control()->getBoolType());

  auto a = emitComplexPart(loc, left, complexType, 0);
  auto b = emitComplexPart(loc, left, complexType, 1);
  auto c = emitComplexPart(loc, right, complexType, 0);
  auto d = emitComplexPart(loc, right, complexType, 1);

  auto realEqual =
      emitBinaryComparisonOp(loc, op, boolType, elementType, a, c).value;
  auto imagEqual =
      emitBinaryComparisonOp(loc, op, boolType, elementType, b, d).value;

  if (!realEqual || !imagEqual)
    return {gen.emitTodoExpr(loc, "complex comparison operator")};

  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt, realEqual,
                                    imagEqual)};

    case TokenKind::T_EXCLAIM_EQUAL:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::OrInt, realEqual,
                                    imagEqual)};

    default:
      break;
  }

  return {gen.emitTodoExpr(loc, "complex comparison operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpFloat(
    SourceLocation loc, TokenKind binop, ir::TypeRef resultType,
    ir::ValueRef left, ir::ValueRef right) -> ExpressionResult {
  switch (binop) {
    case TokenKind::T_PLUS:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AddFloat, left, right)};

    case TokenKind::T_MINUS:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::SubFloat, left, right)};

    case TokenKind::T_STAR:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::MulFloat, left, right)};

    case TokenKind::T_SLASH:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::DivFloat, left, right)};

    default:
      break;
  }

  return {gen.emitTodoExpr(loc, "float arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpIntegral(
    SourceLocation loc, TokenKind binop, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  const bool isSigned = gen.traits.is_signed(leftType);
  switch (binop) {
    case TokenKind::T_PLUS:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, left, right)};

    case TokenKind::T_MINUS:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, left, right)};

    case TokenKind::T_STAR:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, left, right)};

    case TokenKind::T_SLASH:
      return {isSigned ? gen.emitter_.binaryOp(loc, ir::BinaryOp::SignedDiv,
                                               left, right)
                       : gen.emitter_.binaryOp(loc, ir::BinaryOp::UnsignedDiv,
                                               left, right)};

    case TokenKind::T_PERCENT:
      return {isSigned ? gen.emitter_.binaryOp(loc, ir::BinaryOp::SignedRem,
                                               left, right)
                       : gen.emitter_.binaryOp(loc, ir::BinaryOp::UnsignedRem,
                                               left, right)};

    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "integral arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpPointer(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType, ir::ValueRef left,
    ir::ValueRef right) -> ExpressionResult {
  switch (op) {
    case TokenKind::T_PLUS: {
      auto base = left;
      auto offset = right;
      if (!(gen.emitter_.typeKind(gen.emitter_.typeOf(left)) ==
            ir::TypeKind::Pointer)) {
        std::swap(base, offset);
      }
      return {gen.emitter_.pointerAdd(loc, resultType, base, offset)};
    }
    case TokenKind::T_MINUS: {
      if ((gen.emitter_.typeKind(gen.emitter_.typeOf(right)) ==
           ir::TypeKind::Pointer)) {
        return {gen.emitter_.pointerDiff(
            loc, gen.convertType(control()->getLongIntType()), left, right)};
      }
      auto offsetType = gen.emitter_.typeOf(right);
      auto zero = gen.emitter_.constantInt(loc, offsetType, 0);
      auto offset =
          gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, zero, right);
      return {gen.emitter_.pointerAdd(loc, resultType, left, offset)};
    }
    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "pointer arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOp(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  if (auto complexType = unqualified_cast<ComplexType>(leftType)) {
    return emitComplexArithmeticOp(loc, op, complexType, left, right);
  }

  if (gen.traits.is_vector(leftType))
    leftType = gen.traits.get_element_type(leftType);

  if (gen.traits.is_floating_point(leftType)) {
    return emitBinaryArithmeticOpFloat(loc, op, resultType, left, right);
  }

  if (gen.traits.is_integral(leftType)) {
    return emitBinaryArithmeticOpIntegral(loc, op, resultType, leftType, left,
                                          right);
  }

  return {gen.emitTodoExpr(loc, "arithmetic operator")};
}

auto Codegen::ExpressionVisitor::emitBinaryShiftOp(
    SourceLocation opLoc, TokenKind binOp, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  if (gen.traits.is_vector(leftType))
    leftType = gen.traits.get_element_type(leftType);

  const auto rightType = gen.emitter_.typeOf(right);

  if (rightType != resultType &&
      gen.emitter_.typeKind(rightType) == ir::TypeKind::Integer &&
      gen.emitter_.typeKind(resultType) == ir::TypeKind::Integer) {
    right = gen.emitter_.scalarWidth(rightType) >
                    gen.emitter_.scalarWidth(resultType)
                ? gen.emitter_.truncate(opLoc, right, resultType)
                : gen.emitter_.zeroExtend(opLoc, right, resultType);
  }

  if (binOp == TokenKind::T_LESS_LESS) {
    return {gen.emitter_.binaryOp(opLoc, ir::BinaryOp::ShiftLeft, left, right)};
  }

  if (gen.traits.is_signed(leftType)) {
    return {gen.emitter_.binaryOp(opLoc, ir::BinaryOp::ArithmeticShiftRight,
                                  left, right)};
  }

  return {gen.emitter_.binaryOp(opLoc, ir::BinaryOp::LogicalShiftRight, left,
                                right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpFloat(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType, ir::ValueRef left,
    ir::ValueRef right) -> ExpressionResult {
  ir::FloatPredicate predicate;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      predicate = ir::FloatPredicate::OrderedEqual;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      predicate = ir::FloatPredicate::OrderedNotEqual;
      break;
    case TokenKind::T_LESS:
      predicate = ir::FloatPredicate::OrderedLess;
      break;
    case TokenKind::T_LESS_EQUAL:
      predicate = ir::FloatPredicate::OrderedLessEqual;
      break;
    case TokenKind::T_GREATER:
      predicate = ir::FloatPredicate::OrderedGreater;
      break;
    case TokenKind::T_GREATER_EQUAL:
      predicate = ir::FloatPredicate::OrderedGreaterEqual;
      break;
    default:
      return {gen.emitTodoExpr(loc, "float comparison operator")};
  }
  return {gen.emitter_.compareFloat(loc, predicate, left, right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpIntegral(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  const bool isSigned = gen.traits.is_signed(leftType);
  ir::IntPredicate predicate;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      predicate = ir::IntPredicate::Equal;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      predicate = ir::IntPredicate::NotEqual;
      break;
    case TokenKind::T_LESS:
      predicate = isSigned ? ir::IntPredicate::SignedLess
                           : ir::IntPredicate::UnsignedLess;
      break;
    case TokenKind::T_LESS_EQUAL:
      predicate = isSigned ? ir::IntPredicate::SignedLessEqual
                           : ir::IntPredicate::UnsignedLessEqual;
      break;
    case TokenKind::T_GREATER:
      predicate = isSigned ? ir::IntPredicate::SignedGreater
                           : ir::IntPredicate::UnsignedGreater;
      break;
    case TokenKind::T_GREATER_EQUAL:
      predicate = isSigned ? ir::IntPredicate::SignedGreaterEqual
                           : ir::IntPredicate::UnsignedGreaterEqual;
      break;
    default:
      return {gen.emitTodoExpr(loc, "integral comparison operator")};
  }
  return {gen.emitter_.compareInt(loc, predicate, left, right)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOpPointer(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  auto intPtrType = gen.emitter_.integerType(64);
  auto leftInt = gen.emitter_.pointerToInt(loc, intPtrType, left);
  auto rightInt = gen.emitter_.pointerToInt(loc, intPtrType, right);
  ir::IntPredicate predicate;
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
      predicate = ir::IntPredicate::Equal;
      break;
    case TokenKind::T_EXCLAIM_EQUAL:
      predicate = ir::IntPredicate::NotEqual;
      break;
    case TokenKind::T_LESS:
      predicate = ir::IntPredicate::UnsignedLess;
      break;
    case TokenKind::T_LESS_EQUAL:
      predicate = ir::IntPredicate::UnsignedLessEqual;
      break;
    case TokenKind::T_GREATER:
      predicate = ir::IntPredicate::UnsignedGreater;
      break;
    case TokenKind::T_GREATER_EQUAL:
      predicate = ir::IntPredicate::UnsignedGreaterEqual;
      break;
    default:
      return {gen.emitTodoExpr(loc, "pointer comparison operator")};
  }
  return {gen.emitter_.compareInt(loc, predicate, leftInt, rightInt)};
}

auto Codegen::ExpressionVisitor::emitBinaryComparisonOp(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType,
    const Type* leftType, ir::ValueRef left, ir::ValueRef right)
    -> ExpressionResult {
  if (auto complexType = unqualified_cast<ComplexType>(leftType)) {
    return emitComplexComparisonOp(loc, op, complexType, left, right);
  }

  if (gen.traits.is_vector(leftType)) {
    auto elementType = gen.traits.get_element_type(leftType);
    auto lanes =
        emitBinaryComparisonOp(loc, op, resultType, elementType, left, right);
    if (!lanes.value) return lanes;
    return {gen.emitter_.signExtend(loc, lanes.value, resultType)};
  }

  if (gen.traits.is_floating_point(leftType)) {
    return emitBinaryComparisonOpFloat(loc, op, resultType, left, right);
  }

  if (gen.traits.is_pointer(leftType)) {
    return emitBinaryComparisonOpPointer(loc, op, resultType, leftType, left,
                                         right);
  }

  auto comparisonType = gen.traits.underlying_type(leftType);

  if (gen.traits.is_integral(comparisonType) ||
      gen.traits.is_null_pointer(comparisonType)) {
    return emitBinaryComparisonOpIntegral(loc, op, resultType, comparisonType,
                                          left, right);
  }

  return {gen.emitTodoExpr(loc, "comparison operator")};
}

auto Codegen::ExpressionVisitor::comparisonCategoryAddress(SourceLocation loc,
                                                           Symbol* symbol)
    -> ir::ValueRef {
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    if (auto definition = field->definition()) {
      auto global = gen.findOrCreateGlobal(definition);
      if (!global) return {};
      auto pointerType =
          gen.emitter_.pointerType(gen.convertType(definition->type()));
      return gen.emitter_.addressOfSymbol(loc, pointerType,
                                          gen.globalName(*global));
    }

    auto global = gen.findOrCreateStaticField(field);
    auto pointerType = gen.emitter_.pointerType(gen.convertType(field->type()));
    return gen.emitter_.addressOfSymbol(loc, pointerType,
                                        gen.globalName(global));
  }

  if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
    auto global = gen.findOrCreateGlobal(variable);
    if (!global) return {};
    auto pointerType =
        gen.emitter_.pointerType(gen.convertType(variable->type()));
    return gen.emitter_.addressOfSymbol(loc, pointerType,
                                        gen.globalName(*global));
  }

  return {};
}

auto Codegen::ExpressionVisitor::emitThreeWayComparison(
    ThreeWayComparisonExpressionAST* ast, ExpressionResult left,
    ExpressionResult right) -> ExpressionResult {
  auto comparison = ast->comparison;
  auto object = gen.takeResultObject(ast);
  const bool ownsTemporary = !object;
  if (ownsTemporary) object = gen.newTemp(ast->type, comparison->opLoc);

  auto lessBlock = gen.newBlock();
  auto equalBlock = gen.newBlock();
  auto greaterBlock = gen.newBlock();
  ir::BlockRef unorderedBlock;
  if (ast->unorderedResult) unorderedBlock = gen.newBlock();
  auto endBlock = gen.newBlock();

  auto boolType = gen.convertType(control()->getBoolType());
  auto compare = [&](TokenKind op) {
    return emitBinaryComparisonOp(comparison->opLoc, op, boolType,
                                  comparison->leftExpression->type, left.value,
                                  right.value)
        .value;
  };

  auto loc = comparison->opLoc;
  if (ast->unorderedResult) {
    auto less = compare(TokenKind::T_LESS);
    auto continueAfterLess = gen.newBlock();
    gen.emitter_.condBranch(loc, less, lessBlock, continueAfterLess);

    gen.emitter_.setInsertionBlock(continueAfterLess);
    auto greater = compare(TokenKind::T_GREATER);
    auto continueAfterGreater = gen.newBlock();
    gen.emitter_.condBranch(loc, greater, greaterBlock, continueAfterGreater);

    gen.emitter_.setInsertionBlock(continueAfterGreater);
    auto equal = compare(TokenKind::T_EQUAL_EQUAL);
    gen.emitter_.condBranch(loc, equal, equalBlock, unorderedBlock);
  } else {
    auto equal = compare(TokenKind::T_EQUAL_EQUAL);
    auto continueAfterEqual = gen.newBlock();
    gen.emitter_.condBranch(loc, equal, equalBlock, continueAfterEqual);

    gen.emitter_.setInsertionBlock(continueAfterEqual);
    auto less = compare(TokenKind::T_LESS);
    gen.emitter_.condBranch(loc, less, lessBlock, greaterBlock);
  }

  auto emitCategory = [&](ir::BlockRef block, Symbol* symbol) {
    gen.emitter_.setInsertionBlock(block);
    auto address = comparisonCategoryAddress(comparison->opLoc, symbol);
    auto categoryType = gen.convertType(ast->type);
    auto value = gen.emitter_.load(loc, categoryType, address,
                                   gen.getAlignment(ast->type));
    gen.emitter_.store(loc, value, object, gen.getAlignment(ast->type));
    gen.branch(loc, endBlock);
  };

  emitCategory(lessBlock, ast->lessResult);
  emitCategory(equalBlock, ast->equalResult);
  emitCategory(greaterBlock, ast->greaterResult);
  if (ast->unorderedResult) emitCategory(unorderedBlock, ast->unorderedResult);

  gen.emitter_.setInsertionBlock(endBlock);
  if (ownsTemporary) gen.addTemporaryCleanup(object, ast->type);
  return {object};
}

auto Codegen::ExpressionVisitor::emitBinaryBitwiseOp(
    SourceLocation loc, TokenKind op, ir::TypeRef resultType, ir::ValueRef left,
    ir::ValueRef right) -> ExpressionResult {
  switch (op) {
    case TokenKind::T_CARET:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::XorInt, left, right)};
    case TokenKind::T_AMP:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt, left, right)};
    case TokenKind::T_BAR:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::OrInt, left, right)};
    default:
      break;
  }
  return {gen.emitTodoExpr(loc, "bitwise operator")};
}

auto Codegen::ExpressionVisitor::binaryExpression(
    SourceLocation opLoc, TokenKind op, ir::TypeRef resultType,
    ExpressionAST* leftExpression, ExpressionAST* rightExpression,
    ExpressionResult leftExpressionResult,
    ExpressionResult rightExpressionResult) -> ExpressionResult {
  switch (op) {
    case TokenKind::T_PLUS:
      if (gen.traits.is_pointer(leftExpression->type) ||
          gen.traits.is_pointer(rightExpression->type)) {
        return emitBinaryArithmeticOpPointer(opLoc, op, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
      }
      return emitBinaryArithmeticOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_MINUS:
      if (gen.traits.is_pointer(leftExpression->type)) {
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

  const bool isVoid = gen.traits.is_void(ast->type);

  const bool sharesResultObject =
      !isVoid && ast->valueCategory == ValueCategory::kPrValue &&
      gen.traits.is_class(ast->type);

  auto object = sharesResultObject ? gen.takeResultObject(ast) : ir::ValueRef{};

  ir::ValueRef t;
  const Type* type = nullptr;
  if (!isVoid) {
    type = ast->type;
    if (ast->valueCategory != ValueCategory::kPrValue) {
      type = control()->getPointerType(type);
    }
    t = object ? object : gen.newTemp(type, ast->questionLoc);
  }

  gen.condition(ast->condition, trueBlock, falseBlock);

  auto endLoc = lastTokenLocation(ast);

  if (isVoid) {
    gen.emitter_.setInsertionBlock(trueBlock);
    (void)gen.expression(ast->iftrueExpression);
    gen.branch(endLoc, endBlock);

    gen.emitter_.setInsertionBlock(falseBlock);
    (void)gen.expression(ast->iffalseExpression);
    gen.branch(endLoc, endBlock);

    gen.emitter_.setInsertionBlock(endBlock);
    return {};
  }

  if (sharesResultObject && !object) gen.addTemporaryCleanup(t, type);

  auto emitArm = [&](ExpressionAST* arm, ir::BlockRef armBlock,
                     SourceLocation storeLoc) {
    gen.emitter_.setInsertionBlock(armBlock);
    auto conditionalEvaluation = ConditionalEvaluation{gen};

    if (sharesResultObject) {
      (void)gen.emitPrvalueInto(t, type, arm, storeLoc);
    } else {
      auto armResult = gen.expression(arm);
      gen.emitter_.store(storeLoc, armResult.value, t, gen.getAlignment(type));
    }

    gen.branch(endLoc, endBlock);
  };

  emitArm(ast->iftrueExpression, trueBlock, ast->questionLoc);
  emitArm(ast->iffalseExpression, falseBlock, ast->colonLoc);

  gen.emitter_.setInsertionBlock(endBlock);

  if (format == ExpressionFormat::kSideEffect) return {};

  if (sharesResultObject) return {t};

  auto resultType = gen.convertType(type);
  auto loadOp =
      gen.emitter_.load(ast->colonLoc, resultType, t, gen.getAlignment(type));
  return {loadOp};
}

auto Codegen::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  if (ast->symbol) {
    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, leftExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {leftExpressionResult, rightExpressionResult}, false,
                          ast);
    }
  }

  if (ast->op == TokenKind::T_EQUAL) {
    if (auto access = emitBitFieldAccess(ast->leftExpression)) {
      auto rhs = gen.expression(ast->rightExpression);
      auto stored =
          emitBitFieldStore(ast->firstSourceLocation(), *access, rhs.value);
      return {stored};
    }

    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);

    const auto loc = ast->opLoc;

    auto assignedType = gen.traits.remove_cv(ast->leftExpression->type);
    if (gen.traits.is_class_or_union(assignedType)) {
      rightExpressionResult.value =
          gen.classValueLoad(loc, assignedType, rightExpressionResult.value);
    }

    if (gen.traits.is_atomic(assignedType)) {
      emitAtomicStore(loc, leftExpressionResult.value,
                      rightExpressionResult.value);
    } else {
      gen.emitter_.store(loc, rightExpressionResult.value,
                         leftExpressionResult.value,
                         gen.getAlignment(ast->leftExpression->type));
    }

    if (format == ExpressionFormat::kSideEffect) {
      return {};
    }

    if (gen.unit_->language() == LanguageKind::kC) {
      auto resultLoc = ast->firstSourceLocation();
      auto resultType = gen.convertType(ast->leftExpression->type);

      auto op =
          gen.emitter_.load(resultLoc, resultType, leftExpressionResult.value,
                            gen.getAlignment(ast->leftExpression->type));

      return {op};
    }

    return leftExpressionResult;
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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

auto Codegen::ExpressionVisitor::emitAtomicCompoundAssignment(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  TokenKind binaryOp = TokenKind::T_EOF_SYMBOL;

  switch (ast->op) {
    case TokenKind::T_PLUS_EQUAL:
      binaryOp = TokenKind::T_PLUS;
      break;
    case TokenKind::T_MINUS_EQUAL:
      binaryOp = TokenKind::T_MINUS;
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
    default:
      break;
  }

  auto loc = ast->opLoc;
  auto address = gen.expression(ast->targetExpression);
  auto operand = gen.expression(ast->rightExpression);

  if (binaryOp == TokenKind::T_EOF_SYMBOL || !address.value || !operand.value) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "atomic compound assignment")};
  }

  auto valueType = gen.traits.remove_atomic(ast->targetExpression->type);

  auto old = emitAtomicReadModifyWrite(loc, binaryOp, valueType, address.value,
                                       operand.value);

  if (!old) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "atomic compound assignment")};
  }

  if (format == ExpressionFormat::kSideEffect) return {};

  return address;
}

auto Codegen::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  if (ast->symbol) {
    auto targetExpressionResult = gen.expression(ast->targetExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      return gen.emitCall(ast->opLoc, ast->symbol, targetExpressionResult,
                          {rightExpressionResult}, ast->isVirtualDispatch, ast);
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {targetExpressionResult, rightExpressionResult},
                          false, ast);
    }
  }

  if (gen.traits.is_atomic(ast->targetExpression->type)) {
    return emitAtomicCompoundAssignment(ast);
  }

  auto bitField = emitBitFieldAccess(ast->targetExpression);

  auto targetExpressionResult =
      bitField ? ExpressionResult{emitBitFieldLoad(
                     ast->targetExpression->firstSourceLocation(), *bitField,
                     gen.convertType(ast->targetExpression->type))}
               : gen.expression(ast->targetExpression);

  auto targetValue = targetExpressionResult.value;

  std::swap(gen.targetValue_, targetValue);
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  std::swap(gen.targetValue_, targetValue);

  auto rightExpressionResult = gen.expression(ast->rightExpression);

  auto resultType = gen.emitter_.typeOf(leftExpressionResult.value);

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

  auto loc = ast->opLoc;

  auto compoundAssignmentOp = binaryExpression(
      ast->opLoc, binaryOp, resultType, ast->leftExpression,
      ast->rightExpression, leftExpressionResult, rightExpressionResult);

  targetValue = compoundAssignmentOp.value;
  std::swap(gen.targetValue_, targetValue);
  auto sourceExpressionResult = gen.expression(ast->adjustExpression);
  std::swap(gen.targetValue_, targetValue);

  if (bitField) {
    return {emitBitFieldStore(loc, *bitField, sourceExpressionResult.value)};
  }

  gen.emitter_.store(loc, sourceExpressionResult.value,
                     targetExpressionResult.value, gen.getAlignment(ast->type));

  if (format == ExpressionFormat::kSideEffect) {
    return {};
  }

  if (gen.unit_->language() == LanguageKind::kC) {
    auto loadType = gen.emitter_.typeOf(sourceExpressionResult.value);
    auto op = gen.emitter_.load(loc, loadType, targetExpressionResult.value,
                                gen.getAlignment(ast->type));
    return {op};
  }

  return targetExpressionResult;
}

auto Codegen::ExpressionVisitor::operator()(PackExpansionExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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
    auto loc = ast->firstSourceLocation();
    auto op =
        gen.emitter_.constantInt(loc, resultType, ast->value.value() ? 1 : 0);
    return {op};
  }

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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

  return {gen.loadReferenceBinding(ast->firstSourceLocation(), var->type(),
                                   local.value())};
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

  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(ast->type);
  auto ptrType = gen.emitter_.pointerType(type);
  auto temp = gen.emitter_.allocate(loc, ptrType, gen.getAlignment(ast->type));

  gen.emitAggregateInit(temp, ast->type, ast);

  auto op = gen.emitter_.load(loc, type, temp, gen.getAlignment(ast->type));
  return {op};
}

auto Codegen::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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

auto Codegen::arrayElementAddress(SourceLocation loc, ir::ValueRef address,
                                  const Type* elementType, std::size_t index)
    -> ir::ValueRef {
  auto indexType = emitter_.integerType(32);
  auto indexOp =
      emitter_.constantInt(loc, indexType, static_cast<std::int64_t>(index));
  auto elementPtrType = emitter_.pointerType(convertType(elementType));
  return emitter_.pointerAdd(loc, elementPtrType, address, indexOp);
}

void Codegen::emitArrayCopy(SourceLocation loc, ir::ValueRef destination,
                            ir::ValueRef source, const Type* arrayType) {
  auto bounded = type_cast<BoundedArrayType>(traits.remove_cv(arrayType));

  if (bounded && !traits.is_trivially_copyable(bounded->elementType())) {
    auto elementType = bounded->elementType();

    for (std::size_t index = 0; index < bounded->size(); ++index) {
      auto destinationElement =
          arrayElementAddress(loc, destination, elementType, index);
      auto sourceElement = arrayElementAddress(loc, source, elementType, index);

      if (traits.is_array(elementType)) {
        emitArrayCopy(loc, destinationElement, sourceElement, elementType);
        continue;
      }

      auto elementClass =
          unqualified_cast<ClassType>(traits.remove_cv(elementType));
      auto copyConstructor =
          elementClass && elementClass->symbol()
              ? elementClass->symbol()->resolvedDefinition()->copyConstructor()
              : nullptr;
      if (!copyConstructor) return;

      (void)emitCtorCall({}, copyConstructor, destinationElement,
                         {{sourceElement}}, /*completeObject=*/true);
    }
    return;
  }

  auto size = control()->memoryLayout()->sizeOf(arrayType);
  if (!size) return;

  emitter_.memcpy(loc, destination, source, *size);
}

void Codegen::arrayInit(ir::ValueRef address, const Type* type,
                        ExpressionAST* init) {
  if (!init) return;

  if (auto initializer = ast_cast<DefaultInitializerExpressionAST>(init))
    return arrayInit(address, type, initializer->expression);

  if (ast_cast<EqualInitializerAST>(init))
    return arrayInit(address, type, Initializer{init}.clause());

  if (auto strLit = ast_cast<StringLiteralExpressionAST>(init)) {
    auto arr = type_cast<BoundedArrayType>(type);
    if (!arr && !type_cast<UnboundedArrayType>(type)) return;

    auto loc = init->firstSourceLocation();

    std::string str(strLit->literal->stringValue());
    str.push_back('\0');

    auto copyLen = (arr && str.size() > arr->size()) ? arr->size() : str.size();

    if (auto size = control()->memoryLayout()->sizeOf(type)) {
      emitter_.memsetZero(loc, address, *size);
    }

    auto i8Ty = emitter_.integerType(8);
    auto arrTy = this->arrayType(i8Ty, str.size());
    auto it = stringLiterals_.find(strLit->literal);
    if (it == stringLiterals_.end()) {
      auto initializer =
          ir::Initializer::byteString(std::string_view(str.data(), str.size()));
      auto name = newUniqueSymbolName(".str");
      auto insertionGuard = ir::InsertionGuard(emitter_);
      emitter_.setModuleInsertionPoint(false);
      auto linkage = ir::Linkage::Internal;
      (void)this->declareGlobal(loc,
                                {.name = name,
                                 .type = arrTy,
                                 .linkage = linkage,
                                 .isConstant = true,
                                 .alignment = static_cast<std::uint64_t>(0),
                                 .initializer = initializer,
                                 .unknownLocation = false});
      it = stringLiterals_.insert_or_assign(strLit->literal, name).first;
    }

    auto srcPtrType = emitter_.pointerType(arrTy);
    auto srcAddr = emitter_.addressOfSymbol(loc, srcPtrType, it->second);

    emitter_.memcpy(loc, address, srcAddr, copyLen);
    return;
  }

  if (isWholeArrayCopy(traits, init, type)) {
    auto source = init;
    while (auto cast = ast_cast<ImplicitCastExpressionAST>(source))
      source = cast->expression;

    auto sourceAddress = expression(source).value;
    if (!sourceAddress) return;

    emitArrayCopy(init->firstSourceLocation(), address, sourceAddress, type);
    return;
  }

  auto braced = ast_cast<BracedInitListAST>(init);
  if (!braced) return;

  auto loc = braced->firstSourceLocation();

  bool hasDesignated = false;
  for (auto node : ListView{braced->expressionList}) {
    if (ast_cast<DesignatedInitializerClauseAST>(node)) {
      hasDesignated = true;
      break;
    }
  }

  if (hasDesignated) {
    if (auto size = control()->memoryLayout()->sizeOf(type)) {
      emitter_.memsetZero(loc, address, *size);
    }
    for (auto node : ListView{braced->expressionList}) {
      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
        emitDesignatedInit(address, type, desig);
      }
    }
    return;
  }

  if (auto size = control()->memoryLayout()->sizeOf(type)) {
    emitter_.memsetZero(loc, address, *size);
  }

  auto elementType = traits.get_element_type(type);

  std::size_t index = 0;

  for (auto node : ListView{braced->expressionList}) {
    auto nodeLoc = node->firstSourceLocation();

    auto elementAddress =
        arrayElementAddress(nodeLoc, address, elementType, index++);

    if (traits.is_array(elementType)) {
      arrayInit(elementAddress, elementType, node);
    } else if (traits.is_class_or_union(traits.remove_cv(elementType))) {
      (void)emitPrvalueInto(elementAddress, elementType, node,
                            node->firstSourceLocation());
    } else {
      auto value = expression(node);
      emitter_.store(nodeLoc, value.value, elementAddress,
                     getAlignment(elementType));
    }
  }
}

auto Codegen::emitInPlaceConstruction(ir::ValueRef address, ExpressionAST* ast)
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
                     /*completeObject=*/true);

  return true;
}

void Codegen::emitAggregateInit(ir::ValueRef address, const Type* type,
                                BracedInitListAST* ast) {
  emitAggregateInit(address, type, ast->expressionList,
                    ast->firstSourceLocation());
}

void Codegen::emitAggregateInit(ir::ValueRef address, const Type* type,
                                List<ExpressionAST*>* initializerList,
                                SourceLocation location) {
  auto loc = location;

  std::optional<DefaultInitializerObjectGuard> defaultInitializerObject;
  if (traits.is_class_or_union(type)) {
    const auto hasDefaultInitializer =
        std::ranges::any_of(ListView{initializerList}, [](ExpressionAST* node) {
          return ast_cast<DefaultInitializerExpressionAST>(node) != nullptr;
        });

    if (hasDefaultInitializer) {
      auto objectPointerType = control()->getPointerType(type);
      auto slotType = emitter_.pointerType(convertType(objectPointerType));
      auto slot =
          emitter_.allocate(loc, slotType, getAlignment(objectPointerType));
      emitter_.store(loc, address, slot, getAlignment(objectPointerType));
      defaultInitializerObject.emplace(*this, slot);
    }
  }

  if (auto size = control()->memoryLayout()->sizeOf(type)) {
    emitter_.memsetZero(loc, address, *size);
  }

  if (traits.is_array(type) || traits.is_vector(type) ||
      traits.is_complex(type)) {
    auto elementType = traits.get_element_type(type);

    std::size_t index = 0;
    for (auto node : ListView{initializerList}) {
      auto elemLoc = node->firstSourceLocation();

      auto elementAddress =
          arrayElementAddress(elemLoc, address, elementType, index);

      if (auto nested = ast_cast<BracedInitListAST>(node)) {
        emitAggregateInit(elementAddress, elementType, nested);
      } else if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
        emitDesignatedInit(address, type, desig);
      } else if (traits.is_array(elementType)) {
        arrayInit(elementAddress, elementType, node);
      } else if (traits.is_class_or_union(traits.remove_cv(elementType))) {
        (void)emitPrvalueInto(elementAddress, elementType, node,
                              node->firstSourceLocation());
      } else {
        auto val = expression(node);
        emitter_.store(elemLoc, val.value, elementAddress,
                       getAlignment(elementType));
      }
      ++index;
    }
  } else if (traits.is_class_or_union(type)) {
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType || !classType->symbol()) return;
    auto classSymbol = classType->symbol();

    if (auto elemType = traits.initializer_list_element_type(type)) {
      std::uint64_t count = 0;
      for (auto it = initializerList; it; it = it->next) ++count;

      std::vector<FieldSymbol*> fields;
      for (auto field :
           views::members(classSymbol) | views::non_static_fields) {
        fields.push_back(field);
      }
      if (fields.size() != 2) return;

      auto elemIrType = convertType(elemType);
      auto elemPtrType = emitter_.pointerType(elemIrType);
      auto intType = emitter_.integerType(32);

      ir::ValueRef beginPtr;
      if (count > 0) {
        auto arrayIrType = this->arrayType(elemIrType, count);
        auto arrayPtrType = emitter_.pointerType(arrayIrType);
        auto arrayAlloca =
            emitter_.allocate(loc, arrayPtrType, getAlignment(elemType));

        std::size_t index = 0;
        for (auto node : ListView{initializerList}) {
          auto elemLoc = node->firstSourceLocation();
          auto elementAddress =
              arrayElementAddress(elemLoc, arrayAlloca, elemType, index);
          if (traits.is_class_or_union(traits.remove_cv(elemType))) {
            (void)emitPrvalueInto(elementAddress, elemType, node,
                                  node->firstSourceLocation());
          } else {
            auto val = expression(node);
            emitter_.store(elemLoc, val.value, elementAddress,
                           getAlignment(elemType));
          }
          ++index;
        }

        auto zeroIdx = emitter_.constantInt(loc, intType, 0);
        beginPtr = emitter_.pointerAdd(loc, elemPtrType, arrayAlloca, zeroIdx);
      } else {
        beginPtr = emitter_.nullPointer(loc, elemPtrType);
      }

      auto layout = classSymbol->layout();
      auto storeField = [&](size_t idx, ir::ValueRef value) {
        auto field = fields[idx];
        std::uint32_t memberIndex = static_cast<std::uint32_t>(idx);
        if (layout) {
          if (auto fi = layout->getFieldInfo(field)) memberIndex = fi->index;
        }
        auto memberAddr =
            memberAddress(loc, address, field->type(), memberIndex);
        emitter_.store(loc, value, memberAddr, getAlignment(field->type()));
      };

      storeField(0, beginPtr);

      auto sizeIrType = convertType(fields[1]->type());
      auto sizeConst = emitter_.constantInt(loc, sizeIrType, count);
      storeField(1, sizeConst);
      return;
    }

    if (classType->isUnion()) {
      auto it = initializerList;
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

      auto elemLoc = expr->firstSourceLocation();

      auto memberAddr =
          memberAddress(elemLoc, address, targetField->type(), memberIndex);

      if (auto nested = ast_cast<BracedInitListAST>(expr)) {
        emitAggregateInit(memberAddr, targetField->type(), nested);
      } else if (traits.is_array(targetField->type())) {
        arrayInit(memberAddr, targetField->type(), expr);
      } else if (traits.is_class_or_union(
                     traits.remove_cv(targetField->type()))) {
        (void)emitPrvalueInto(memberAddr, targetField->type(), expr,
                              expr->firstSourceLocation());
      } else {
        auto val = expression(expr);
        emitter_.store(elemLoc, val.value, memberAddr,
                       getAlignment(targetField->type()));
      }
    } else {
      auto aggregateElements = traits.aggregate_elements(classSymbol);
      auto layout = classSymbol->layout();
      size_t elementIndex = 0;

      for (auto node : ListView{initializerList}) {
        if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
          emitDesignatedInit(address, type, desig);

          if (desig->designatorList) {
            if (auto dot =
                    ast_cast<DotDesignatorAST>(desig->designatorList->value);
                dot && dot->symbol) {
              for (size_t i = 0; i < aggregateElements.size(); ++i) {
                if (aggregateElements[i] == dot->symbol) {
                  elementIndex = i + 1;
                  break;
                }
              }
            }
          }
          continue;
        }

        if (elementIndex >= aggregateElements.size()) break;

        auto element = aggregateElements[elementIndex++];
        auto elementType = subobjectType(element);
        if (!elementType) continue;

        auto elemLoc = node->firstSourceLocation();

        auto elementAddress =
            subobjectAddress(elemLoc, address, classSymbol, element);
        if (!elementAddress) continue;

        std::optional<ClassLayout::MemberInfo> fi;
        if (auto field = symbol_cast<FieldSymbol>(element); field && layout)
          fi = layout->getFieldInfo(field);

        if (fi && fi->bitWidth > 0) {
          emitBitFieldInit(elemLoc, elementAddress, elementType, *fi, node);
        } else if (auto nested = ast_cast<BracedInitListAST>(node)) {
          emitAggregateInit(elementAddress, elementType, nested);
        } else if (emitInPlaceConstruction(elementAddress, node)) {
          continue;
        } else if (traits.is_array(elementType)) {
          arrayInit(elementAddress, elementType, node);
        } else if (traits.is_class_or_union(traits.remove_cv(elementType))) {
          (void)emitPrvalueInto(elementAddress, elementType, node,
                                node->firstSourceLocation());
        } else {
          auto val = expression(node);
          emitter_.store(elemLoc, val.value, elementAddress,
                         getAlignment(elementType));
        }
      }
    }
  } else {
    auto it = initializerList;
    if (!it) return;

    auto val = expression(it->value);
    emitter_.store(loc, val.value, address, getAlignment(type));
  }
}

void Codegen::emitDesignatedInit(ir::ValueRef address, const Type* type,
                                 DesignatedInitializerClauseAST* ast) {
  ir::ValueRef currentAddr = address;
  const Type* currentType = type;
  std::optional<ClassLayout::MemberInfo> currentFieldInfo;

  for (auto desigIt = ast->designatorList; desigIt; desigIt = desigIt->next) {
    auto designator = desigIt->value;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto field = dot->symbol;
      if (!field) return;

      auto classType = unqualified_cast<ClassType>(currentType);
      if (!classType || !classType->symbol()) return;

      auto classSymbol = classType->symbol();
      auto fieldClass = symbol_cast<ClassSymbol>(field->parent());

      if (fieldClass && classSymbol != fieldClass) {
        currentAddr = navigateToClass(ast->firstSourceLocation(), currentAddr,
                                      classSymbol, fieldClass);
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

      auto elemLoc = dot->firstSourceLocation();

      currentAddr =
          memberAddress(elemLoc, currentAddr, field->type(), memberIndex);
      currentType = traits.remove_cv(field->type());

    } else if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      currentFieldInfo = std::nullopt;
      auto elementType = traits.get_element_type(currentType);
      auto elementIrType = convertType(elementType);
      auto resultType = emitter_.pointerType(elementIrType);
      auto elemLoc = subscript->firstSourceLocation();

      auto indexVal = expression(subscript->expression);
      currentAddr =
          emitter_.pointerAdd(elemLoc, resultType, currentAddr, indexVal.value);
      currentType = traits.remove_cv(elementType);
    }
  }

  auto initExpr = Initializer{ast->initializer}.clause();

  if (!initExpr) return;

  auto elemLoc = initExpr->firstSourceLocation();

  if (ast->constructorSymbol) {
    List<ExpressionAST*>* argumentList = nullptr;
    if (auto braced = ast_cast<BracedInitListAST>(initExpr)) {
      argumentList = constructorArgumentList(braced);
    } else if (auto paren = ast_cast<ParenInitializerAST>(initExpr)) {
      argumentList = paren->expressionList;
    } else {
      argumentList = make_list_node<ExpressionAST>(unit_->arena(), initExpr);
    }

    std::vector<ExpressionResult> arguments;
    for (auto it = argumentList; it; it = it->next)
      arguments.push_back(expression(it->value));

    (void)emitCtorCall(initExpr->firstSourceLocation(), ast->constructorSymbol,
                       currentAddr, arguments, /*completeObject=*/true);
  } else if (currentFieldInfo && currentFieldInfo->bitWidth > 0) {
    emitBitFieldInit(elemLoc, currentAddr, currentType, *currentFieldInfo,
                     initExpr);
  } else if (auto nested = ast_cast<BracedInitListAST>(initExpr)) {
    emitAggregateInit(currentAddr, currentType, nested);
  } else if (traits.is_class_or_union(traits.remove_cv(currentType))) {
    (void)emitPrvalueInto(currentAddr, currentType, initExpr,
                          initExpr->firstSourceLocation());
  } else {
    auto val = expression(initExpr);
    emitter_.store(elemLoc, val.value, currentAddr, getAlignment(currentType));
  }
}

void Codegen::emitBitFieldInit(SourceLocation loc, ir::ValueRef address,
                               const Type* type,
                               const ClassLayout::MemberInfo& info,
                               ExpressionAST* init) {
  if (auto equalInit = ast_cast<EqualInitializerAST>(init))
    init = equalInit->expression;

  if (auto braced = ast_cast<BracedInitListAST>(init))
    init = braced->expressionList ? braced->expressionList->value : nullptr;

  auto value = init ? expression(init).value
                    : emitter_.constantInt(loc, convertType(type), 0);

  emitter_.storeBitfield(
      loc, value, address,
      {info.bitOffset, info.bitWidth, info.allocUnitSizeBytes});
}

auto Codegen::ExpressionVisitor::emitClassConstruction(
    ExpressionAST* ast, SourceLocation loc, const Type* classType,
    List<ExpressionAST*>* argList, FunctionSymbol* constructorSymbol)
    -> ExpressionResult {
  classType = gen.traits.remove_cv(classType);

  auto classT = type_cast<ClassType>(classType);
  if (!classT || !classT->symbol())
    return {gen.emitTodoExpr(loc, "class construction: no class symbol")};

  auto object = gen.takeResultObject(ast);
  const bool ownsTemporary = !object;
  if (ownsTemporary) object = gen.newTemp(classType, loc);

  std::vector<ExpressionResult> args;
  for (auto it = argList; it; it = it->next)
    args.push_back(gen.expression(it->value));

  int argCount = static_cast<int>(args.size());

  if (!constructorSymbol && argCount != 0)
    return {gen.emitTodoExpr(loc, "class construction: no recorded ctor")};

  if (!argList && gen.requiresZeroInitialization(classType, constructorSymbol))
    gen.emitZeroInitialization(loc, object, classType);

  if (constructorSymbol) {
    (void)gen.emitCtorCall(loc, constructorSymbol, object, args,
                           /*completeObject=*/true);
  }

  if (ownsTemporary) gen.addTemporaryCleanup(object, classType);

  return {object};
}

auto Codegen::baseStructorVTTArgument(SourceLocation loc,
                                      ClassSymbol* targetClass)
    -> ir::ValueRef {
  if (!currentFunctionSymbol_ || !targetClass) return {};
  targetClass = targetClass->resolvedDefinition();
  auto currentClass =
      symbol_cast<ClassSymbol>(currentFunctionSymbol_->parent());
  if (!currentClass) return {};
  currentClass = currentClass->resolvedDefinition();

  if (targetClass == currentClass && !structorVTTValue_ && entryBlock_ &&
      emitter_.blockParameterCount(entryBlock_) == 1)
    return vttAddress(loc, currentClass, 0);

  if (auto principal = currentFunctionSymbol_->structorPrincipal(); principal) {
    auto layout = buildVTT(currentClass);
    std::size_t index = 0;
    if (targetClass != currentClass) {
      auto found = layout.virtualBaseStarts.find(targetClass);
      if (found == layout.virtualBaseStarts.end()) return {};
      index = found->second;
    }
    return vttAddress(loc, currentClass, index);
  }

  auto currentVTT = structorVTTValue_;
  const auto entryArgumentCount = emitter_.blockParameterCount(entryBlock_);
  if (!currentVTT && entryBlock_ && entryArgumentCount > 1)
    currentVTT = emitter_.blockParameter(entryBlock_, entryArgumentCount - 1);
  if (!currentVTT) return {};

  if (targetClass == currentClass) return currentVTT;

  auto layout = buildVTT(currentClass);
  auto found = layout.directBaseStarts.find(targetClass);
  if (found == layout.directBaseStarts.end()) return {};
  auto indexType = convertType(control()->getIntType());
  auto offset = emitter_.constantInt(loc, indexType, found->second);
  return emitter_.pointerAdd(loc, emitter_.typeOf(currentVTT), currentVTT,
                             offset);
}

auto Codegen::emitCall(SourceLocation loc, FunctionSymbol* symbol,
                       ExpressionResult thisValue,
                       std::vector<ExpressionResult> arguments,
                       bool isVirtualDispatch, ExpressionAST* resultOwner,
                       bool baseObjectStructor) -> ExpressionResult {
  auto functionType = type_cast<FunctionType>(symbol->type());

  return emitCall(loc, functionType, symbol, isVirtualDispatch, thisValue,
                  std::move(arguments),
                  takeIndirectResultObject(resultOwner, functionType), {},
                  baseObjectStructor);
}

auto Codegen::emitCall(SourceLocation loc, const FunctionType* functionType,
                       FunctionSymbol* symbol, bool isVirtualDispatch,
                       ExpressionResult thisValue,
                       std::vector<ExpressionResult> arguments,
                       ir::ValueRef resultObject, ir::ValueRef calleeValue,
                       bool baseObjectStructor) -> ExpressionResult {
  if (!functionType) return {};

  loc = implicitLocation(loc);

  const auto calleeOf = [&](FunctionSymbol* callee) {
    return baseObjectStructor ? findOrCreateBaseObjectStructor(callee)
                              : findOrCreateFunction(callee);
  };

  if (symbol && thisValue.value) {
    auto targetClass = symbol_cast<ClassSymbol>(symbol->parent());
    auto function = calleeOf(symbol);
    const auto suppliedCount = arguments.size() + 1;
    if (requiresVTT(targetClass) &&
        emitter_.functionParameterTypes(function).size() > suppliedCount) {
      auto vtt = baseStructorVTTArgument(loc, targetClass);
      if (vtt) arguments.push_back({vtt});
    }
  }

  const auto& paramTypes = functionType->parameterTypes();

  std::vector<ir::ValueRef> args;
  if (thisValue.value) {
    args.push_back(thisValue.value);
  }

  for (size_t i = 0; i < arguments.size(); ++i) {
    auto val = arguments[i].value;
    if (!val) continue;

    if (i >= paramTypes.size()) {
      args.push_back(val);
      continue;
    }

    if (traits.is_reference(paramTypes[i])) {
      if ((emitter_.typeKind(emitter_.typeOf(val)) == ir::TypeKind::Pointer)) {
        args.push_back(val);
        continue;
      }
      auto elemType = traits.remove_reference(paramTypes[i]);
      auto temp = newTemp(elemType, loc);
      emitter_.store(loc, val, temp, getAlignment(elemType));
      args.push_back(temp);
      continue;
    }

    if (!usesClassValueAbi(paramTypes[i])) {
      args.push_back(val);
      continue;
    }

    if (isClassValueDestroyedInCallee(paramTypes[i])) cancelCleanup(val);

    abiLowerClassArgument(loc, paramTypes[i], val, args);
  }

  const auto returnsThis = structorReturnsThis(symbol);

  std::vector<ir::TypeRef> resultTypes;
  ir::ValueRef sretTemp;
  if (returnsThis) {
    if (symbol && !isVirtualDispatch) {
      auto funcOp = calleeOf(symbol);
      auto results = emitter_.functionResultTypes(funcOp);
      resultTypes.insert(resultTypes.end(), results.begin(), results.end());
    } else if (!args.empty()) {
      resultTypes.push_back(emitter_.typeOf(args[0]));
    }
  } else {
    sretTemp = abiPrepareResult(loc, functionType->returnType(), resultTypes,
                                resultObject);
  }

  if (sretTemp) {
    args.insert(args.begin(), sretTemp);
  }

  ir::CallInfo callInfo;

  if (isVirtualDispatch) {
    int slotIndex = vtableSlotIndex(symbol);

    auto objectPtr = thisValue.value;

    auto objectPtrType = emitter_.typeOf(objectPtr);
    if (auto ptrPtrType =
            emitter_.asType(ir::TypeKind::Pointer, objectPtrType)) {
      if (auto ptrType = emitter_.asType(ir::TypeKind::Pointer,
                                         emitter_.elementType(ptrPtrType))) {
        objectPtr =
            emitter_.load(loc, emitter_.elementType(ptrPtrType), objectPtr, 4);
      }
    }

    auto i8Type = emitter_.integerType(8);
    auto i8PtrType = emitter_.pointerType(i8Type);
    auto i8PtrPtrType = emitter_.pointerType(i8PtrType);

    auto vptrFieldPtr = memberAddress(loc, objectPtr, i8PtrPtrType, 0);

    auto vtablePtr = emitter_.load(loc, i8PtrPtrType, vptrFieldPtr, 8);

    auto offsetType = convertType(control()->getIntType());
    auto offsetOp = emitter_.constantInt(loc, offsetType, slotIndex);

    auto funcPtrAddr =
        emitter_.pointerAdd(loc, i8PtrPtrType, vtablePtr, offsetOp);

    auto funcPtr = emitter_.load(loc, i8PtrType, funcPtrAddr, 8);

    callInfo.indirectCallee = funcPtr;
  } else if (!symbol) {
    callInfo.indirectCallee = calleeValue;
  } else {
    auto funcOp = calleeOf(symbol);
    if (emitter_.functionParameterTypes(funcOp).size() > args.size()) {
      auto targetClass = symbol_cast<ClassSymbol>(symbol->parent());
      if (requiresVTT(targetClass)) {
        auto vtt = baseStructorVTTArgument(loc, targetClass);
        if (vtt) args.push_back(vtt);
      }
    }
    callInfo.callee = this->functionName(funcOp);
  }

  if (functionType->isVariadic()) {
    callInfo.variadicCalleeType = convertType(functionType);
  }

  std::vector<ir::ValueRef> argumentRefs;
  for (auto argument : args) argumentRefs.push_back(argument);

  std::vector<ir::TypeRef> resultTypeRefs;
  for (auto resultType : resultTypes) resultTypeRefs.push_back(resultType);

  auto parameterAbi = computeParameterAbi(functionType, symbol);

  callInfo.arguments = argumentRefs;
  callInfo.results = resultTypeRefs;
  callInfo.parameters = parameterAbi;

  auto callResults = emitter_.call(loc, callInfo);

  if (returnsThis) return {};

  auto result =
      abiFinishResult(loc, functionType->returnType(), callResults, sretTemp);

  if (sretTemp && !resultObject)
    addTemporaryCleanup(sretTemp, functionType->returnType());

  return result;
}

void Codegen::emitCaptureInit(ClassSymbol* classSymbol, ir::ValueRef closure,
                              LambdaCaptureAST* capture,
                              ExpressionAST* initializer) {
  auto field = capture_field(capture);
  if (!field) return;

  auto layout = classSymbol->layout();
  if (!layout) return;

  auto fieldInfo = layout->getFieldInfo(field);
  if (!fieldInfo) return;

  auto loc = capture->firstSourceLocation();
  auto fieldPtr = memberAddress(loc, closure, field->type(), fieldInfo->index);

  emitFieldInitializer(loc, field, fieldPtr, initializer);
}

auto Codegen::emitCtorCall(SourceLocation loc, FunctionSymbol* ctor,
                           ir::ValueRef thisPtr,
                           std::vector<ExpressionResult> args,
                           bool completeObject, ir::ValueRef vtt)
    -> ExpressionResult {
  auto target = ctor;
  if (completeObject) {
    if (auto variant = ctor->completeObjectVariant()) target = variant;
  } else if (auto principal = ctor->structorPrincipal()) {
    target = principal;
  }

  auto targetClass = symbol_cast<ClassSymbol>(target->parent());
  if (targetClass) targetClass = targetClass->resolvedDefinition();
  if (!completeObject && target->isConstructor() && requiresVTT(targetClass)) {
    if (!vtt) vtt = baseStructorVTTArgument(loc, targetClass);
    if (vtt) args.push_back({vtt});
  }
  return emitCall(loc, target, {thisPtr}, std::move(args),
                  /*isVirtualDispatch=*/false, /*resultOwner=*/nullptr,
                  /*baseObjectStructor=*/!completeObject);
}

auto Codegen::ExpressionVisitor::emitArithmeticConversion(
    SourceLocation loc, ir::ValueRef value, const Type* sourceType,
    const Type* targetType) -> ir::ValueRef {
  if (gen.traits.is_same(sourceType, targetType)) return value;

  auto resultType = gen.convertType(targetType);

  const auto sourceIsFloating = gen.traits.is_floating_point(sourceType);
  const auto targetIsFloating = gen.traits.is_floating_point(targetType);

  if (sourceIsFloating && targetIsFloating) {
    auto sourceWidth = gen.emitter_.scalarWidth(gen.emitter_.typeOf(value));
    auto targetWidth = gen.emitter_.scalarWidth(resultType);
    if (sourceWidth == targetWidth) return value;
    if (sourceWidth < targetWidth)
      return gen.emitter_.floatExtend(loc, value, resultType);
    return gen.emitter_.floatTruncate(loc, value, resultType);
  }

  if (!sourceIsFloating && targetIsFloating) {
    if (gen.traits.is_signed(sourceType))
      return gen.emitter_.signedIntToFloat(loc, value, resultType);
    return gen.emitter_.unsignedIntToFloat(loc, value, resultType);
  }

  if (sourceIsFloating && !targetIsFloating) {
    if (gen.traits.is_signed(targetType))
      return gen.emitter_.floatToSignedInt(loc, value, resultType);
    return gen.emitter_.floatToUnsignedInt(loc, value, resultType);
  }

  auto sourceWidth = gen.emitter_.scalarWidth(gen.emitter_.typeOf(value));
  auto targetWidth = gen.emitter_.scalarWidth(resultType);

  if (sourceWidth == targetWidth) return value;
  if (targetWidth < sourceWidth)
    return gen.emitter_.truncate(loc, value, resultType);
  if (gen.traits.is_signed(sourceType))
    return gen.emitter_.signExtend(loc, value, resultType);
  return gen.emitter_.zeroExtend(loc, value, resultType);
}

auto Codegen::ExpressionVisitor::emitComplexPart(SourceLocation loc,
                                                 ir::ValueRef value,
                                                 const ComplexType* complexType,
                                                 std::int64_t position)
    -> ir::ValueRef {
  auto elementType = gen.convertType(complexType->elementType());
  return gen.emitter_.extractValue(loc, elementType, value, position);
}

auto Codegen::ExpressionVisitor::makeComplexValue(
    SourceLocation loc, const ComplexType* complexType, ir::ValueRef real,
    ir::ValueRef imag) -> ir::ValueRef {
  auto resultType = gen.convertType(complexType);
  auto value = gen.emitter_.undef(loc, resultType);
  value = gen.emitter_.insertValue(loc, resultType, value, real, 0);
  return gen.emitter_.insertValue(loc, resultType, value, imag, 1);
}

auto Codegen::ExpressionVisitor::emitComplexOperand(
    SourceLocation loc, ir::ValueRef value, const Type* sourceType,
    const ComplexType* targetType) -> ir::ValueRef {
  auto elementType = targetType->elementType();

  if (auto sourceComplex = unqualified_cast<ComplexType>(sourceType)) {
    auto real = emitArithmeticConversion(
        loc, emitComplexPart(loc, value, sourceComplex, 0),
        sourceComplex->elementType(), elementType);
    auto imag = emitArithmeticConversion(
        loc, emitComplexPart(loc, value, sourceComplex, 1),
        sourceComplex->elementType(), elementType);
    return makeComplexValue(loc, targetType, real, imag);
  }

  auto real = emitArithmeticConversion(loc, value, sourceType, elementType);
  auto imag = gen.emitter_.constantZero(loc, gen.convertType(elementType));
  return makeComplexValue(loc, targetType, real, imag);
}

auto Codegen::ExpressionVisitor::emitComplexConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto operand = gen.expression(ast->expression);
  if (!operand.value) return {};

  auto sourceType = ast->expression->type;

  if (auto targetComplex = unqualified_cast<ComplexType>(ast->type)) {
    return {emitComplexOperand(loc, operand.value, sourceType, targetComplex)};
  }

  auto sourceComplex = unqualified_cast<ComplexType>(sourceType);
  if (!sourceComplex) return {};

  auto real = emitComplexPart(loc, operand.value, sourceComplex, 0);
  return {emitArithmeticConversion(loc, real, sourceComplex->elementType(),
                                   ast->type)};
}

auto Codegen::ExpressionVisitor::sequentiallyConsistentOrder(SourceLocation loc)
    -> ir::ValueRef {
  return gen.emitter_.constantInt(loc, gen.convertType(control()->getIntType()),
                                  5);
}

auto Codegen::ExpressionVisitor::emitAtomicLoad(SourceLocation loc,
                                                const Type* valueType,
                                                ir::ValueRef address)
    -> ir::ValueRef {
  const ir::TypeRef results[] = {gen.convertType(valueType)};
  const ir::ValueRef arguments[] = {address, sequentiallyConsistentOrder(loc)};
  return gen.emitter_.builtinCall(
      loc, results, Token::spell(BuiltinFunctionKind::T___C11_ATOMIC_LOAD),
      arguments);
}

void Codegen::ExpressionVisitor::emitAtomicStore(SourceLocation loc,
                                                 ir::ValueRef address,
                                                 ir::ValueRef value) {
  const ir::ValueRef arguments[] = {address, value,
                                    sequentiallyConsistentOrder(loc)};
  (void)gen.emitter_.builtinCall(
      loc, std::vector<ir::TypeRef>{},
      Token::spell(BuiltinFunctionKind::T___C11_ATOMIC_STORE), arguments);
}

auto Codegen::ExpressionVisitor::emitAtomicReadModifyWrite(
    SourceLocation loc, TokenKind binaryOp, const Type* valueType,
    ir::ValueRef address, ir::ValueRef operand) -> ir::ValueRef {
  BuiltinFunctionKind kind;

  switch (binaryOp) {
    case TokenKind::T_PLUS:
      kind = BuiltinFunctionKind::T___C11_ATOMIC_FETCH_ADD;
      break;
    case TokenKind::T_MINUS:
      kind = BuiltinFunctionKind::T___C11_ATOMIC_FETCH_SUB;
      break;
    case TokenKind::T_AMP:
      kind = BuiltinFunctionKind::T___C11_ATOMIC_FETCH_AND;
      break;
    case TokenKind::T_BAR:
      kind = BuiltinFunctionKind::T___C11_ATOMIC_FETCH_OR;
      break;
    case TokenKind::T_CARET:
      kind = BuiltinFunctionKind::T___C11_ATOMIC_FETCH_XOR;
      break;
    default:
      return {};
  }

  const ir::TypeRef results[] = {gen.convertType(valueType)};
  const ir::ValueRef arguments[] = {address, operand,
                                    sequentiallyConsistentOrder(loc)};
  return gen.emitter_.builtinCall(loc, results, Token::spell(kind), arguments);
}

auto Codegen::ExpressionVisitor::emitAtomicIncrDecr(
    SourceLocation loc, TokenKind op, const Type* atomicType,
    ir::ValueRef address, bool postfix) -> ExpressionResult {
  auto valueType = gen.traits.remove_atomic(atomicType);
  auto irValueType = gen.convertType(valueType);

  auto one = gen.emitter_.constantInt(loc, irValueType, 1);

  const auto binaryOp =
      op == TokenKind::T_PLUS_PLUS ? TokenKind::T_PLUS : TokenKind::T_MINUS;

  auto old = emitAtomicReadModifyWrite(loc, binaryOp, valueType, address, one);

  if (!old) {
    return {gen.emitTodoExpr(loc, "atomic increment")};
  }

  if (postfix) return {old};

  return emitBinaryArithmeticOp(loc, binaryOp, irValueType, valueType, old,
                                one);
}

auto Codegen::ExpressionVisitor::emitComplexToBoolean(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto operand = gen.expression(ast->expression);
  if (!operand.value) return {};

  auto complexType = unqualified_cast<ComplexType>(ast->expression->type);
  if (!complexType) return {};

  auto elementType = complexType->elementType();
  auto boolType = gen.convertType(ast->type);
  auto zero = gen.emitter_.constantZero(loc, gen.convertType(elementType));

  auto realNotZero =
      emitBinaryComparisonOp(
          loc, TokenKind::T_EXCLAIM_EQUAL, boolType, elementType,
          emitComplexPart(loc, operand.value, complexType, 0), zero)
          .value;

  auto imagNotZero =
      emitBinaryComparisonOp(
          loc, TokenKind::T_EXCLAIM_EQUAL, boolType, elementType,
          emitComplexPart(loc, operand.value, complexType, 1), zero)
          .value;

  if (!realNotZero || !imagNotZero) return {};

  return {gen.emitter_.binaryOp(loc, ir::BinaryOp::OrInt, realNotZero,
                                imagNotZero)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinComplex(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto resultType = gen.convertType(ast->type);

  auto value = gen.emitter_.undef(loc, resultType);
  std::int64_t position = 0;

  for (auto argument : ListView{ast->expressionList}) {
    if (position >= 2) break;
    auto part = gen.expression(argument);
    if (!part.value) return {};
    value =
        gen.emitter_.insertValue(loc, resultType, value, part.value, position);
    ++position;
  }

  return {value};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeVal(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getDoubleType());
  auto op = gen.emitter_.constantLiteral(
      loc, type,
      ir::Initializer::floatingValue(
          gen.emitter_.floatingType(ir::FloatKind::Double),
          std::numeric_limits<double>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeValf(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getFloatType());
  auto op = gen.emitter_.constantLiteral(
      loc, type,
      ir::Initializer::floatingValue(
          gen.emitter_.floatingType(ir::FloatKind::Single),
          std::numeric_limits<float>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinHugeVall(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getLongDoubleType());
  auto op = gen.emitter_.constantLiteral(
      loc, type,
      ir::Initializer::floatingValue(type,
                                     std::numeric_limits<double>::infinity()));
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNans(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getDoubleType());
  auto op = gen.emitter_.signalingNaN(loc, type);
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNansf(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getFloatType());
  auto op = gen.emitter_.signalingNaN(loc, type);
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinNansl(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto type = gen.convertType(control()->getLongDoubleType());
  auto op = gen.emitter_.signalingNaN(loc, type);
  return {op};
}

auto Codegen::ExpressionVisitor::codegenBuiltinAlloca(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto sizeVal = gen.expression(*it);
  if (!sizeVal.value) return {};
  auto i8Type = gen.emitter_.integerType(8);
  auto ptrType = gen.emitter_.pointerType(i8Type);
  return {gen.emitter_.dynamicAllocate(loc, ptrType, sizeVal.value,
                                       /*alignment=*/1)};
}

auto Codegen::ExpressionVisitor::floatClassificationOperand(
    CallExpressionAST* ast)
    -> std::optional<std::pair<SourceLocation, ir::ValueRef>> {
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return std::nullopt;

  auto operand = gen.expression(*it);
  if (!operand.value) return std::nullopt;
  if (!(gen.emitter_.typeKind(gen.emitter_.typeOf(operand.value)) ==
        ir::TypeKind::Floating))
    return std::nullopt;

  return std::pair{ast->firstSourceLocation(), operand.value};
}

auto Codegen::ExpressionVisitor::codegenBuiltinIsNan(CallExpressionAST* ast)
    -> ExpressionResult {
  auto operand = floatClassificationOperand(ast);
  if (!operand) return {};
  auto [loc, value] = *operand;

  return {gen.emitter_.compareFloat(loc, ir::FloatPredicate::Unordered, value,
                                    value)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinIsInf(CallExpressionAST* ast)
    -> ExpressionResult {
  auto operand = floatClassificationOperand(ast);
  if (!operand) return {};
  auto [loc, value] = *operand;

  auto floatType = gen.emitter_.typeOf(value);
  auto infinity = std::numeric_limits<double>::infinity();

  auto positiveInfinity = gen.emitter_.constantLiteral(
      loc, floatType, ir::Initializer::floatingValue(floatType, infinity));

  auto negativeInfinity = gen.emitter_.constantLiteral(
      loc, floatType, ir::Initializer::floatingValue(floatType, -infinity));

  auto isPositiveInfinity = gen.emitter_.compareFloat(
      loc, ir::FloatPredicate::OrderedEqual, value, positiveInfinity);

  auto isNegativeInfinity = gen.emitter_.compareFloat(
      loc, ir::FloatPredicate::OrderedEqual, value, negativeInfinity);

  return {gen.emitter_.binaryOp(loc, ir::BinaryOp::OrInt, isPositiveInfinity,
                                isNegativeInfinity)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinIsFinite(CallExpressionAST* ast)
    -> ExpressionResult {
  auto operand = floatClassificationOperand(ast);
  if (!operand) return {};
  auto [loc, value] = *operand;

  auto floatType = gen.emitter_.typeOf(value);
  auto infinity = std::numeric_limits<double>::infinity();

  auto positiveInfinity = gen.emitter_.constantLiteral(
      loc, floatType, ir::Initializer::floatingValue(floatType, infinity));

  auto negativeInfinity = gen.emitter_.constantLiteral(
      loc, floatType, ir::Initializer::floatingValue(floatType, -infinity));

  auto belowInfinity = gen.emitter_.compareFloat(
      loc, ir::FloatPredicate::OrderedLess, value, positiveInfinity);

  auto aboveNegativeInfinity = gen.emitter_.compareFloat(
      loc, ir::FloatPredicate::OrderedGreater, value, negativeInfinity);

  return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt, belowInfinity,
                                aboveNegativeInfinity)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinSignbit(CallExpressionAST* ast)
    -> ExpressionResult {
  auto operand = floatClassificationOperand(ast);
  if (!operand) return {};
  auto [loc, value] = *operand;

  auto floatType = gen.emitter_.typeOf(value);
  auto bitsType = gen.emitter_.integerType(gen.emitter_.scalarWidth(floatType));

  auto bits = gen.emitter_.reinterpretBits(loc, value, bitsType);

  auto zero = gen.emitter_.constantInt(loc, bitsType, 0);

  return {
      gen.emitter_.compareInt(loc, ir::IntPredicate::SignedLess, bits, zero)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinBzero(CallExpressionAST* ast)
    -> ExpressionResult {
  auto loc = ast->firstSourceLocation();
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
  auto sizeType = gen.emitter_.integerType(memoryLayout->sizeOfSizeType() * 8);
  auto zero =
      gen.emitter_.constantLiteral(loc, gen.emitter_.integerType(8),
                                   ir::Initializer::integerValue(sizeType, 0));
  std::vector<ir::ValueRef> inputs{destVal.value, zero, sizeVal.value};
  auto op =
      gen.emitter_.builtinCall(loc, std::vector<ir::TypeRef>{},
                               std::string_view("__builtin_memset"), inputs);
  return {op};
}

auto Codegen::ExpressionVisitor::emitMemberPointerAccess(
    BinaryExpressionAST* ast) -> ExpressionResult {
  auto loc = ast->firstSourceLocation();

  auto object = gen.expression(ast->leftExpression);
  auto memberPointer = gen.expression(ast->rightExpression);
  if (!object.value || !memberPointer.value) return {};

  auto charPointerType =
      gen.convertType(control()->getPointerType(control()->getCharType()));

  auto base = gen.emitter_.bitcast(loc, charPointerType, object.value);

  auto address =
      gen.emitter_.pointerAdd(loc, charPointerType, base, memberPointer.value);

  auto resultType = gen.convertType(control()->getPointerType(ast->type));

  return {gen.emitter_.bitcast(loc, resultType, address)};
}

auto Codegen::ExpressionVisitor::emitMemberFunctionPointerCall(
    CallExpressionAST* ast, BinaryExpressionAST* access) -> ExpressionResult {
  auto pointerType = type_cast<MemberFunctionPointerType>(
      gen.traits.remove_cvref(access->rightExpression->type));
  if (!pointerType) return {};

  auto functionType = type_cast<FunctionType>(pointerType->functionType());
  if (!functionType) return {};

  auto loc = ast->firstSourceLocation();

  auto object = gen.expression(access->leftExpression);
  auto memberPointer = gen.expression(access->rightExpression);
  if (!object.value || !memberPointer.value) return {};

  auto wordType = gen.pointerSizedIntType();
  const auto wordSize =
      static_cast<std::int64_t>(control()->memoryLayout()->sizeOfPointer());
  auto i8PtrType = gen.emitter_.pointerType(gen.emitter_.integerType(8));
  auto i8PtrPtrType = gen.emitter_.pointerType(i8PtrType);

  auto constantWord = [&](std::int64_t value) -> ir::ValueRef {
    return gen.emitter_.constantInt(loc, wordType, value);
  };

  auto [pointerField, adjustmentField] =
      gen.memberFunctionPointerFields(loc, pointerType, memberPointer.value);

  auto objectAddress = gen.emitter_.bitcast(loc, i8PtrType, object.value);

  auto thisAdjustment =
      gen.emitter_.binaryOp(loc, ir::BinaryOp::ArithmeticShiftRight,
                            adjustmentField, constantWord(1));

  auto thisValue =
      gen.emitter_.pointerAdd(loc, i8PtrType, objectAddress, thisAdjustment);

  auto virtualBit = gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt,
                                          adjustmentField, constantWord(1));
  auto isVirtual = gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual,
                                           virtualBit, constantWord(0));

  auto virtualBlock = gen.newBlock();
  auto directBlock = gen.newBlock();
  auto callBlock = gen.newBlock();
  auto calleeArgument =
      gen.emitter_.addBlockParameter(callBlock, i8PtrType, loc);

  gen.emitter_.condBranch(loc, isVirtual, virtualBlock, directBlock);

  gen.emitter_.setInsertionBlock(virtualBlock);
  {
    auto vtable = gen.emitter_.load(
        loc, i8PtrPtrType, gen.emitter_.bitcast(loc, i8PtrPtrType, thisValue),
        static_cast<int>(wordSize));
    auto slotAddress = gen.emitter_.pointerAdd(
        loc, i8PtrType, gen.emitter_.bitcast(loc, i8PtrType, vtable),
        pointerField);
    auto callee = gen.emitter_.load(
        loc, i8PtrType, gen.emitter_.bitcast(loc, i8PtrPtrType, slotAddress),
        static_cast<int>(wordSize));
    gen.branch(loc, callBlock, std::vector<ir::ValueRef>{callee});
  }

  gen.emitter_.setInsertionBlock(directBlock);
  {
    auto callee = gen.emitter_.intToPointer(loc, i8PtrType, pointerField);
    gen.branch(loc, callBlock, std::vector<ir::ValueRef>{callee});
  }

  gen.emitter_.setInsertionBlock(callBlock);

  std::vector<ExpressionResult> callArguments;
  for (auto node : ListView{ast->expressionList}) {
    callArguments.push_back(gen.expression(node));
  }

  return gen.emitCall(ast->lparenLoc, functionType, nullptr, false, {thisValue},
                      std::move(callArguments),
                      gen.takeIndirectResultObject(ast, functionType),
                      calleeArgument);
}

auto Codegen::ExpressionVisitor::emitMemberFunctionPointerFormation(
    UnaryExpressionAST* ast, const MemberFunctionPointerType* pointerType)
    -> std::optional<ExpressionResult> {
  auto id = ast_cast<IdExpressionAST>(ast->expression);
  if (!id) return std::nullopt;

  auto function = symbol_cast<FunctionSymbol>(id->symbol);
  if (!function) return std::nullopt;

  auto pointerClass = type_cast<ClassType>(pointerType->classType());
  auto declaringClass = symbol_cast<ClassSymbol>(function->parent());
  if (!pointerClass || !declaringClass) return std::nullopt;

  auto adjustment = classSubobjectOffset(pointerClass, declaringClass->type());
  if (!adjustment) return std::nullopt;

  return ExpressionResult{gen.emitMemberFunctionPointerValue(
      ast->firstSourceLocation(), pointerType, function, *adjustment)};
}

auto Codegen::ExpressionVisitor::emitMemberPointerFormation(
    UnaryExpressionAST* ast) -> std::optional<ExpressionResult> {
  if (auto functionPointerType =
          type_cast<MemberFunctionPointerType>(ast->type))
    return emitMemberFunctionPointerFormation(ast, functionPointerType);

  auto dataPointerType = type_cast<MemberObjectPointerType>(ast->type);
  if (!dataPointerType) return std::nullopt;

  auto id = ast_cast<IdExpressionAST>(ast->expression);
  if (!id) return std::nullopt;

  auto field = symbol_cast<FieldSymbol>(id->symbol);
  if (!field) return std::nullopt;

  auto offset = field->offsetInClass();
  if (!offset) return std::nullopt;

  auto loc = ast->firstSourceLocation();
  auto resultType = gen.convertType(ast->type);

  return ExpressionResult{gen.emitter_.constantInt(
      loc, resultType, static_cast<std::int64_t>(*offset))};
}

auto Codegen::ExpressionVisitor::codegenBuiltinFloatComparison(
    CallExpressionAST* ast) -> ExpressionResult {
  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{ast->expressionList})
    arguments.push_back(argument);
  if (arguments.size() != 2) return {};

  auto left = gen.expression(arguments[0]);
  auto right = gen.expression(arguments[1]);
  if (!left.value || !right.value) return {};

  auto predicate = ir::FloatPredicate::Unordered;
  switch (resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression))) {
    case BuiltinFunctionKind::T___BUILTIN_ISGREATER:
      predicate = ir::FloatPredicate::OrderedGreater;
      break;
    case BuiltinFunctionKind::T___BUILTIN_ISGREATEREQUAL:
      predicate = ir::FloatPredicate::OrderedGreaterEqual;
      break;
    case BuiltinFunctionKind::T___BUILTIN_ISLESS:
      predicate = ir::FloatPredicate::OrderedLess;
      break;
    case BuiltinFunctionKind::T___BUILTIN_ISLESSEQUAL:
      predicate = ir::FloatPredicate::OrderedLessEqual;
      break;
    case BuiltinFunctionKind::T___BUILTIN_ISLESSGREATER:
      predicate = ir::FloatPredicate::OrderedNotEqual;
      break;
    case BuiltinFunctionKind::T___BUILTIN_ISUNORDERED:
      predicate = ir::FloatPredicate::Unordered;
      break;
    default:
      return {};
  }

  return {gen.emitter_.compareFloat(ast->firstSourceLocation(), predicate,
                                    left.value, right.value)};
}

auto Codegen::ExpressionVisitor::codegenBuiltinArithmeticOverflow(
    CallExpressionAST* ast) -> ExpressionResult {
  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{ast->expressionList})
    arguments.push_back(argument);
  if (arguments.size() != 3) return {};
  auto left = gen.expression(arguments[0]);
  auto right = gen.expression(arguments[1]);
  auto output = gen.expression(arguments[2]);
  if (!left.value || !right.value || !output.value) return {};
  auto pointer = type_cast<PointerType>(arguments[2]->type);
  if (!pointer) return {};
  auto outputType = pointer->elementType();
  auto resultType =
      gen.emitter_.asType(ir::TypeKind::Integer, gen.convertType(outputType));
  auto leftType = gen.emitter_.asType(ir::TypeKind::Integer,
                                      gen.emitter_.typeOf(left.value));
  auto rightType = gen.emitter_.asType(ir::TypeKind::Integer,
                                       gen.emitter_.typeOf(right.value));
  if (!resultType || !leftType || !rightType) return {};
  auto width = std::max(gen.emitter_.scalarWidth(leftType) +
                            gen.emitter_.scalarWidth(rightType),
                        gen.emitter_.scalarWidth(resultType)) +
               1;
  auto wideType = gen.emitter_.integerType(width);
  auto loc = ast->firstSourceLocation();
  auto extend = [&](ir::ValueRef value, const Type* type) -> ir::ValueRef {
    if (gen.unit_->typeTraits().is_signed(type))
      return gen.emitter_.signExtend(loc, value, wideType);
    return gen.emitter_.zeroExtend(loc, value, wideType);
  };
  auto lhs = extend(left.value, arguments[0]->type);
  auto rhs = extend(right.value, arguments[1]->type);

  ir::ValueRef product;
  switch (resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression))) {
    case BuiltinFunctionKind::T___BUILTIN_ADD_OVERFLOW:
      product = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt, lhs, rhs);
      break;
    case BuiltinFunctionKind::T___BUILTIN_SUB_OVERFLOW:
      product = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, lhs, rhs);
      break;
    case BuiltinFunctionKind::T___BUILTIN_MUL_OVERFLOW:
      product = gen.emitter_.binaryOp(loc, ir::BinaryOp::MulInt, lhs, rhs);
      break;
    default:
      return {};
  }

  auto result = gen.emitter_.truncate(loc, product, resultType);
  auto restored = extend(result, outputType);
  auto overflow = gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual,
                                          product, restored);
  gen.emitter_.store(loc, result, output.value, gen.getAlignment(outputType));
  return {overflow};
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
  auto loc = ast->firstSourceLocation();
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};

  auto pointer = gen.expression(*it);
  if (!pointer.value) return {};

  ++it;
  if (it == args.end()) return pointer;

  auto alignment = gen.expression(*it);
  if (!alignment.value) return pointer;

  std::vector<ir::ValueRef> inputs{pointer.value, alignment.value};

  ++it;
  if (it != args.end()) {
    auto misalignment = gen.expression(*it);
    if (!misalignment.value) return pointer;
    inputs.push_back(misalignment.value);
  }

  auto assumeOp = gen.emitter_.builtinCall(
      loc, std::vector<ir::TypeRef>{gen.convertType(ast->type)},
      "__builtin_assume_aligned", inputs);

  return {assumeOp};
}

auto Codegen::ExpressionVisitor::codegenBuiltinBitCount(CallExpressionAST* ast)
    -> ExpressionResult {
  auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
  auto operation = bitCountOperation(resolveBuiltinFunctionKind(idExpr));
  if (!operation) return {};

  auto loc = ast->firstSourceLocation();
  auto args = ListView{ast->expressionList};
  auto it = args.begin();
  if (it == args.end()) return {};
  auto operand = gen.expression(*it);
  if (!operand.value) return {};

  auto operandType = gen.emitter_.typeOf(operand.value);
  auto i1Type = gen.emitter_.integerType(1);
  auto i32Type = gen.emitter_.integerType(32);

  const auto constant = [&](ir::TypeRef type,
                            std::int64_t value) -> ir::ValueRef {
    return gen.emitter_.constantInt(loc, type, value);
  };

  const auto intrinsic = [&](std::string_view name,
                             std::vector<ir::ValueRef> inputs) -> ir::ValueRef {
    return gen.emitter_.builtinCall(loc, std::vector<ir::TypeRef>{operandType},
                                    name, inputs);
  };

  const auto toResultWidth = [&](ir::ValueRef value) -> ir::ValueRef {
    auto width = gen.emitter_.scalarWidth(gen.emitter_.typeOf(value));
    if (width > 32) return gen.emitter_.truncate(loc, value, i32Type);
    if (width < 32) return gen.emitter_.zeroExtend(loc, value, i32Type);
    return value;
  };

  const auto countLeadingZeros = [&](ir::ValueRef value) {
    return toResultWidth(
        intrinsic("__builtin_clz",
                  std::vector<ir::ValueRef>{value, constant(i1Type, 0)}));
  };

  const auto countTrailingZeros = [&](ir::ValueRef value) {
    return toResultWidth(
        intrinsic("__builtin_ctz",
                  std::vector<ir::ValueRef>{value, constant(i1Type, 0)}));
  };

  const auto populationCount = [&](ir::ValueRef value) {
    return toResultWidth(
        intrinsic("__builtin_popcount", std::vector<ir::ValueRef>{value}));
  };

  ir::ValueRef fallback{};
  if (++it != args.end()) fallback = gen.expression(*it).value;

  const auto withFallback = [&](ir::ValueRef value) -> ir::ValueRef {
    if (!fallback) return value;
    auto isZero = gen.emitter_.compareInt(
        loc, ir::IntPredicate::Equal, operand.value, constant(operandType, 0));
    return gen.emitter_.select(loc, isZero, fallback, value);
  };

  switch (*operation) {
    case BitCountOperation::kCountLeadingZeros:
      return {withFallback(countLeadingZeros(operand.value))};

    case BitCountOperation::kCountTrailingZeros:
      return {withFallback(countTrailingZeros(operand.value))};

    case BitCountOperation::kPopulationCount:
      return {populationCount(operand.value)};

    case BitCountOperation::kParity:
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::AndInt,
                                    populationCount(operand.value),
                                    constant(i32Type, 1))};

    case BitCountOperation::kFindFirstSet: {
      auto index = gen.emitter_.binaryOp(loc, ir::BinaryOp::AddInt,
                                         countTrailingZeros(operand.value),
                                         constant(i32Type, 1));
      auto isZero =
          gen.emitter_.compareInt(loc, ir::IntPredicate::Equal, operand.value,
                                  constant(operandType, 0));
      return {gen.emitter_.select(loc, isZero, constant(i32Type, 0), index)};
    }

    case BitCountOperation::kCountLeadingRedundantSignBits: {
      auto complement = gen.emitter_.binaryOp(
          loc, ir::BinaryOp::XorInt, operand.value, constant(operandType, -1));
      auto isNegative =
          gen.emitter_.compareInt(loc, ir::IntPredicate::SignedLess,
                                  operand.value, constant(operandType, 0));
      auto magnitude =
          gen.emitter_.select(loc, isNegative, complement, operand.value);
      return {gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt,
                                    countLeadingZeros(magnitude),
                                    constant(i32Type, 1))};
    }
  }

  return {};
}
}  // namespace cxx

#include "builtins_codegen-priv.h"
