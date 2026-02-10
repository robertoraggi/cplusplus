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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// mlir
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <format>

namespace cxx {

struct [[nodiscard]] Codegen::ExpressionVisitor {
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
  auto emitUserDefinedConversion(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  auto emitBuiltinCall(CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
      -> ExpressionResult;
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
  auto val = value.value;

  if (auto ptrType = mlir::dyn_cast<mlir::cxx::PointerType>(val.getType())) {
    auto nullOp = mlir::cxx::NullPtrConstantOp::create(builder_, loc, ptrType);
    auto i1Type = mlir::IntegerType::get(builder_.getContext(), 1);
    auto ptrIntTy = builder_.getI64Type();
    auto lhsInt = mlir::cxx::PtrToIntOp::create(builder_, loc, ptrIntTy, val);
    auto rhsInt = mlir::cxx::PtrToIntOp::create(builder_, loc, ptrIntTy,
                                                nullOp.getResult());
    val = mlir::arith::CmpIOp::create(
        builder_, loc, mlir::arith::CmpIPredicate::ne, lhsInt, rhsInt);
  }

  mlir::cf::CondBranchOp::create(builder_, loc, val, trueBlock, falseBlock);
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
  auto context = gen.builder_.getContext();
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

    // null terminator
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
        gen.builder_.getContext(), mlir::cxx::LinkageKind::Internal);
    mlir::cxx::GlobalOp::create(x, loc, mlir::TypeRange(), type, true,
                                name.getValue(), initializer, linkage);

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
  auto ptrType = gen.convertType(ast->type);
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, ptrType, gen.thisValue_, gen.getAlignment(ast->type));

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
  gen.statement(ast->statement);

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
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
          gen.builder_.getContext(), gen.convertType(var->type()));
      val = mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType,
                                           global->getSymName());
      found = true;
    }

    if (found) {
      if (gen.control()->is_reference(var->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(var->type());  // pointer type
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val,
                                        gen.getAlignment(var->type()));
      }
      return {val};
    }
  } else if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    if (auto local = gen.findOrCreateLocal(ast->symbol)) {
      auto val = local.value();
      if (gen.control()->is_reference(param->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(param->type());  // pointer type
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val,
                                        gen.getAlignment(param->type()));
      }
      return {val};
    }
  } else if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (!field->isStatic()) {
      if (!gen.thisValue_) {
        // Should not happen in valid non-static member function
        auto op = gen.emitTodoExpr(ast->firstSourceLocation(),
                                   "implicit use of 'this' but 'this' is null");
        return {op};
      }

      auto classSymbol = symbol_cast<ClassSymbol>(field->parent());

      auto layout = classSymbol->layout();
      if (!layout) {
        return {gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "class layout not computed")};
      }

      auto fieldInfo = layout->getFieldInfo(field);
      if (!fieldInfo) {
        return {gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "field not found in layout")};
      }

      auto loc = gen.getLocation(ast->firstSourceLocation());

      auto thisPtrType =
          gen.convertType(gen.control()->getPointerType(classSymbol->type()));

      auto thisPtr = mlir::cxx::LoadOp::create(
          gen.builder_, loc, thisPtrType, gen.thisValue_,
          gen.getAlignment(gen.control()->getPointerType(classSymbol->type())));

      auto resultType =
          gen.convertType(gen.control()->add_pointer(field->type()));

      auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, resultType,
                                            thisPtr, fieldInfo->index);
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

      // Move to module level to emit the function definitions
      gen.builder_.setInsertionPointToEnd(gen.module_.getBody());

      for (auto ctor : classSymbol->constructors()) {
        if (auto funcDecl = ctor->declaration()) {
          (void)gen.declaration(funcDecl);
        }
      }

      for (auto member : classSymbol->members()) {
        if (auto funcSym = symbol_cast<FunctionSymbol>(member)) {
          if (auto funcDecl = funcSym->declaration()) {
            (void)gen.declaration(funcDecl);
          }
        } else if (auto ovl = symbol_cast<OverloadSetSymbol>(member)) {
          for (auto func : ovl->functions()) {
            if (auto funcDecl = func->declaration()) {
              (void)gen.declaration(funcDecl);
            }
          }
        }
      }

      gen.builder_.restoreInsertionPoint(savedIP);
    }

    // Allocate storage for the closure object
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto mlirType = gen.convertType(classType);
    auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(mlirType);
    auto closureAlloca = mlir::cxx::AllocaOp::create(
        gen.builder_, loc, ptrType, gen.getAlignment(classType));

    // Call the default constructor
    for (auto ctor : classSymbol->constructors()) {
      auto ctorFunc = gen.findOrCreateFunction(ctor);
      mlir::SmallVector<mlir::Value> args;
      args.push_back(closureAlloca);
      mlir::SmallVector<mlir::Type> resultTypes;  // void return
      mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                ctorFunc.getSymName(), args, mlir::TypeAttr{});
      break;
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
  if (ast->type && !control()->is_void(ast->type)) {
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
                          {indexExpressionResult});
    } else {
      return gen.emitCall(ast->lbracketLoc, ast->symbol, {},
                          {baseExpressionResult, indexExpressionResult});
    }
  }

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

static auto builtinFunctionName(BuiltinFunctionKind kind) -> llvm::StringRef {
  switch (kind) {
#define BUILTIN_FUNCTION_NAME(tk, str) \
  case BuiltinFunctionKind::T_##tk:    \
    return str;
    FOR_EACH_BUILTIN_FUNCTION(BUILTIN_FUNCTION_NAME)
#undef BUILTIN_FUNCTION_NAME
    default:
      return "";
  }
}

auto Codegen::ExpressionVisitor::emitBuiltinCall(
    CallExpressionAST* ast, BuiltinFunctionKind builtinKind)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->lparenLoc);

  // __builtin_is_constant_evaluated() always returns false at runtime.
  if (builtinKind == BuiltinFunctionKind::T___BUILTIN_IS_CONSTANT_EVALUATED) {
    auto boolType = gen.convertType(control()->getBoolType());
    auto falseVal = mlir::arith::ConstantOp::create(
        gen.builder_, loc, boolType, gen.builder_.getIntegerAttr(boolType, 0));
    return {falseVal};
  }

  auto name = builtinFunctionName(builtinKind);

  mlir::SmallVector<mlir::Value> arguments;
  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  // determine result type
  mlir::SmallVector<mlir::Type> resultTypes;
  if (ast->type && !control()->is_void(ast->type)) {
    resultTypes.push_back(gen.convertType(ast->type));
  }

  auto op = mlir::cxx::BuiltinCallOp::create(gen.builder_, loc, resultTypes,
                                             name, arguments);

  return {op.getResult()};
}

auto Codegen::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  // strip nested expressions
  auto func = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(func)) {
    func = nested->expression;
  }

  // check for builtin function calls
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

  // check for direct calls
  auto id = ast_cast<IdExpressionAST>(func);
  auto member = ast_cast<MemberExpressionAST>(func);
  ExpressionResult thisValue;

  FunctionSymbol* functionSymbol = nullptr;
  if (id) {
    if (functionSymbol = symbol_cast<FunctionSymbol>(id->symbol)) {
      if (functionSymbol->parent()->isClass() && !functionSymbol->isStatic()) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
        auto thisPtrType =
            gen.convertType(gen.control()->getPointerType(classSymbol->type()));
        auto loadedThis = mlir::cxx::LoadOp::create(
            gen.builder_, loc, thisPtrType, gen.thisValue_,
            gen.getAlignment(
                gen.control()->getPointerType(classSymbol->type())));
        thisValue = {loadedThis};
      }
    }

  } else if (member) {
    functionSymbol = symbol_cast<FunctionSymbol>(member->symbol);

    if (functionSymbol) {
      thisValue = gen.expression(member->baseExpression);
    }
  }

  const FunctionType* functionType = nullptr;
  bool isIndirectCall = false;

  if (functionSymbol) {
    // direct call.
    functionType = type_cast<FunctionType>(functionSymbol->type());
  } else if (control()->is_pointer(ast->baseExpression->type)) {
    // indirect call
    isIndirectCall = true;

    thisValue = gen.expression(ast->baseExpression);

    auto elementType = control()->get_element_type(ast->baseExpression->type);
    functionType = type_cast<cxx::FunctionType>(elementType);
  }

  if (!functionType) {
    auto op =
        gen.emitTodoExpr(ast->firstSourceLocation(), "invalid function call");

    return {op};
  }

  mlir::SmallVector<mlir::Value> arguments;

  if (thisValue.value) {
    arguments.push_back(thisValue.value);
  }

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
    arguments.push_back(value.value);
  }

  mlir::SmallVector<mlir::Type> resultTypes;
  if (!control()->is_void(functionType->returnType())) {
    resultTypes.push_back(gen.convertType(functionType->returnType()));
  }

  auto loc = gen.getLocation(ast->lparenLoc);

  mlir::cxx::CallOp callOp;

  bool isVirtualCall = false;
  if (functionSymbol && functionSymbol->isVirtual() && thisValue.value) {
    if (member) {
      auto baseExprType = member->baseExpression->type;
      if (control()->is_pointer(baseExprType) ||
          control()->is_lvalue_reference(baseExprType)) {
        isVirtualCall = true;
      }
    }
  }

  if (isVirtualCall) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

    auto vtableSlots = gen.computeVtableSlots(classSymbol);
    int slotIndex = 0;
    for (size_t i = 0; i < vtableSlots.size(); ++i) {
      if (vtableSlots[i]->name() == functionSymbol->name()) {
        slotIndex = static_cast<int>(i);
        break;
      }
    }

    auto objectPtr = arguments[0];

    auto objectPtrType = objectPtr.getType();
    if (auto ptrPtrType = dyn_cast<mlir::cxx::PointerType>(objectPtrType)) {
      if (auto ptrType =
              dyn_cast<mlir::cxx::PointerType>(ptrPtrType.getElementType())) {
        objectPtr = mlir::cxx::LoadOp::create(
            gen.builder_, loc, ptrPtrType.getElementType(), objectPtr, 4);
      }
    }

    auto i8Type = gen.builder_.getI8Type();
    auto i8PtrType = gen.builder_.getType<mlir::cxx::PointerType>(i8Type);
    auto i8PtrPtrType = gen.builder_.getType<mlir::cxx::PointerType>(i8PtrType);
    auto i8PtrPtrPtrType =
        gen.builder_.getType<mlir::cxx::PointerType>(i8PtrPtrType);

    auto vptrFieldPtr = mlir::cxx::MemberOp::create(
        gen.builder_, loc, i8PtrPtrPtrType, objectPtr, 0);

    auto vtablePtr = mlir::cxx::LoadOp::create(gen.builder_, loc, i8PtrPtrType,
                                               vptrFieldPtr, 8);

    auto offsetType = gen.convertType(gen.control()->getIntType());
    auto offsetOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, offsetType,
        gen.builder_.getIntegerAttr(offsetType, slotIndex));

    auto funcPtrAddr = mlir::cxx::PtrAddOp::create(
        gen.builder_, loc, i8PtrPtrType, vtablePtr, offsetOp);

    auto funcPtr =
        mlir::cxx::LoadOp::create(gen.builder_, loc, i8PtrType, funcPtrAddr, 8);

    mlir::SmallVector<mlir::Value> indirectCallArgs;
    indirectCallArgs.push_back(funcPtr);
    indirectCallArgs.append(arguments.begin(), arguments.end());

    callOp = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                       mlir::FlatSymbolRefAttr{},
                                       indirectCallArgs, mlir::TypeAttr{});
  } else if (isIndirectCall) {
    callOp = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                       mlir::FlatSymbolRefAttr{}, arguments,
                                       mlir::TypeAttr{});
  } else {
    auto funcOp = gen.findOrCreateFunction(functionSymbol);
    callOp = mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                                       funcOp.getSymName(), arguments,
                                       mlir::TypeAttr{});
  }

  if (functionType->isVariadic()) {
    callOp.setVarCalleeType(
        cast<mlir::cxx::FunctionType>(gen.convertType(functionType)));
  }

  return {callOp.getResult()};
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

static auto isDerivedFrom(ClassSymbol* derived, ClassSymbol* base) -> bool {
  if (derived == base) return true;
  for (auto b : derived->baseClasses()) {
    auto bs = symbol_cast<ClassSymbol>(b->symbol());
    if (bs && isDerivedFrom(bs, base)) return true;
  }
  return false;
}

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && !field->isStatic()) {
    auto baseExpressionResult = gen.expression(ast->baseExpression);

    auto baseType = gen.control()->remove_cv(ast->baseExpression->type);

    if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
      baseType =
          control()->remove_cv(gen.control()->get_element_type(baseType));
    }

    auto classType = type_cast<ClassType>(baseType);

    if (!classType) {
      return {gen.emitTodoExpr(
          ast->firstSourceLocation(),
          std::format("base not class type '{}'", to_string(baseType)))};
    }

    auto startClass = classType->symbol();
    auto fieldClass = symbol_cast<ClassSymbol>(field->parent());

    if (startClass != fieldClass) {
      std::function<mlir::Value(mlir::Value, ClassSymbol*, ClassSymbol*)>
          castToStruct;
      castToStruct = [&](mlir::Value value, ClassSymbol* from,
                         ClassSymbol* to) -> mlir::Value {
        if (from == to) return value;

        auto fromLayout = from->layout();

        for (auto base : from->baseClasses()) {
          auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
          if (!baseSym) continue;

          if (isDerivedFrom(baseSym, to)) {
            std::uint32_t baseIndex = 0;
            if (fromLayout) {
              if (auto bi = fromLayout->getBaseInfo(baseSym)) {
                baseIndex = bi->index;
              }
            }

            auto loc = value.getLoc();
            auto ptrType =
                gen.convertType(gen.control()->getPointerType(baseSym->type()));

            auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, ptrType,
                                                  value, baseIndex);

            return castToStruct(op, baseSym, to);
          }
        }
        return value;
      };

      baseExpressionResult.value =
          castToStruct(baseExpressionResult.value, startClass, fieldClass);
    }

    auto layout = fieldClass->layout();
    if (!layout) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "class layout not computed")};
    }

    auto fieldInfo = layout->getFieldInfo(field);
    if (!fieldInfo) {
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "field not found in layout")};
    }

    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto resultType =
        gen.convertType(gen.control()->add_pointer(field->type()));

    auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, resultType,
                                          baseExpressionResult.value,
                                          fieldInfo->index);
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
  if (control()->is_floating_point(ast->baseExpression->type)) {
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

    switch (control()->remove_cvref(ast->baseExpression->type)->kind()) {
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
        // Handle other float types if necessary
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
  if (control()->is_pointer(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrTy =
        mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
    auto elementTy = ptrTy.getElementType();
    auto loadOp = mlir::cxx::LoadOp::create(
        gen.builder_, loc, elementTy, expressionResult.value,
        gen.getAlignment(ast->baseExpression->type));
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto intTy = mlir::IntegerType::get(gen.builder_.getContext(), 32);
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
  if (ast->castOp == TokenKind::T_STATIC_CAST &&
      ast->valueCategory == ValueCategory::kLValue) {
    auto* innerExpr = ast->expression;

    while (auto* implicitCast =
               ast_cast<ImplicitCastExpressionAST>(innerExpr)) {
      innerExpr = implicitCast->expression;
    }

    auto baseResult = gen.expression(innerExpr);

    auto sourceType = control()->remove_cv(innerExpr->type);
    auto targetType = control()->remove_cv(ast->type);

    auto sourceClass = type_cast<ClassType>(sourceType);
    auto targetClass = type_cast<ClassType>(targetType);

    if (sourceClass && targetClass && sourceClass != targetClass) {
      auto derivedSymbol = sourceClass->symbol();
      auto baseSymbol = targetClass->symbol();

      int baseIndex = 0;
      for (auto base : derivedSymbol->baseClasses()) {
        auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
        if (!baseSym) continue;

        if (baseSym == baseSymbol) {
          auto loc = gen.getLocation(ast->firstSourceLocation());
          auto resultType =
              gen.convertType(gen.control()->add_pointer(targetType));

          auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, resultType,
                                                baseResult.value, baseIndex);
          return {op};
        }
        ++baseIndex;
      }
    }

    return baseResult;
  }

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

    // Get the class type from typeId
    auto classType = type_cast<ClassType>(ast->typeId->type);
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
  if (type_cast<BoolType>(control()->remove_cv(ast->type))) {
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

  if (control()->is_floating_point(ast->type)) {
    auto op = mlir::arith::NegFOp::create(gen.builder_, loc, resultType,
                                          expressionResult.value);

    return {op};
  }

  if (control()->is_integral_or_unscoped_enum(ast->type)) {
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

  switch (control()->remove_cvref(ast->expression->type)->kind()) {
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
  auto ptrTy =
      mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
  auto elementTy = ptrTy.getElementType();
  auto loadOp = mlir::cxx::LoadOp::create(
      gen.builder_, loc, elementTy, expressionResult.value,
      gen.getAlignment(ast->expression->type));
  auto addOp =
      mlir::cxx::PtrAddOp::create(gen.builder_, loc, elementTy, loadOp, one);
  mlir::cxx::StoreOp::create(gen.builder_, loc, addOp, expressionResult.value,
                             gen.getAlignment(ast->expression->type));

  if (is_glvalue(ast)) {
    return expressionResult;
  }

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                      expressionResult.value,
                                      gen.getAlignment(ast->expression->type));
  return {op};
}

auto Codegen::ExpressionVisitor::emitUnaryOpIncrDecr(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);

  if (ast->symbol) {
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {});
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult});
    }
  }

  if (control()->is_floating_point(ast->expression->type)) {
    return emitUnaryOpIncrDecrFloat(ast, expressionResult);
  }

  if (control()->is_arithmetic(ast->expression->type)) {
    return emitUnaryOpIncrDecrIntegral(ast, expressionResult);
  }

  if (control()->is_pointer(ast->expression->type)) {
    return emitUnaryOpIncrDecrPointer(ast, expressionResult);
  }

  return {gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()))};
}

auto Codegen::ExpressionVisitor::operator()(LabelAddressExpressionAST* ast)
    -> ExpressionResult {
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

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
      return gen.emitCall(ast->opLoc, ast->symbol, expressionResult, {});
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {}, {expressionResult});
    }
  }

  switch (ast->op) {
    case TokenKind::T_EXCLAIM:
      return emitUnaryOpNot(ast);

    case TokenKind::T_PLUS:
      return gen.expression(ast->expression);

    case TokenKind::T_MINUS:
      return emitUnaryOpMinus(ast);

    case TokenKind::T_TILDE:
      return emitUnaryOpTilde(ast);

    case TokenKind::T_AMP:
    case TokenKind::T_STAR:
      return gen.expression(ast->expression);

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
  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
auto expressionResult = gen.expression(ast->expression);
#endif

  return {op};
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
  auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(objectMlirType);

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
      auto funcType = mlir::cxx::FunctionType::get(gen.builder_.getContext(),
                                                   paramTypes, resultTypes,
                                                   /*isVariadic=*/false);
      auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
          gen.builder_.getContext(), mlir::cxx::LinkageKind::External);
      auto inlineAttr = mlir::cxx::InlineKindAttr::get(
          gen.builder_.getContext(), mlir::cxx::InlineKind::NoInline);
      existingFunc = mlir::cxx::FuncOp::create(
          gen.builder_, loc, operatorNewName, funcType, linkageAttr, inlineAttr,
          mlir::ArrayAttr{}, mlir::ArrayAttr{});
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
    (void)gen.emitCall(ast->newLoc, ast->constructorSymbol, {rawPtr}, ctorArgs);
  } else if (ast->newInitalizer) {
    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      if (paren->expressionList) {
        auto* initExpr = paren->expressionList->value;
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
  if (auto ptrTy = type_cast<PointerType>(control()->remove_cv(exprType))) {
    pointeeType = ptrTy->elementType();
  }

  if (pointeeType) {
    if (auto classType =
            type_cast<ClassType>(control()->remove_cv(pointeeType))) {
      auto classSymbol = classType->symbol();
      if (auto dtorSymbol = classSymbol->destructor()) {
        if (dtorSymbol->isVirtual()) {
          auto i8Type = gen.builder_.getI8Type();
          auto i8PtrType = gen.builder_.getType<mlir::cxx::PointerType>(i8Type);
          auto i8PtrPtrType =
              gen.builder_.getType<mlir::cxx::PointerType>(i8PtrType);
          auto i8PtrPtrPtrType =
              gen.builder_.getType<mlir::cxx::PointerType>(i8PtrPtrType);

          auto vptrFieldPtr = mlir::cxx::MemberOp::create(
              gen.builder_, loc, i8PtrPtrPtrType, ptrValue, 0);
          auto vtablePtr = mlir::cxx::LoadOp::create(
              gen.builder_, loc, i8PtrPtrType, vptrFieldPtr, 8);

          auto vtableSlots = gen.computeVtableSlots(classSymbol);
          int slotIndex = 0;
          for (size_t i = 0; i < vtableSlots.size(); ++i) {
            if (vtableSlots[i]->name() == dtorSymbol->name()) {
              slotIndex = static_cast<int>(i);
              break;
            }
          }

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
        } else {
          (void)gen.emitCall(ast->deleteLoc, dtorSymbol, {ptrValue}, {});
        }
      }
    }
  }

  bool isArrayDelete = static_cast<bool>(ast->lbracketLoc);
  auto operatorDeleteName = std::string(isArrayDelete ? "_ZdaPv" : "_ZdlPv");

  auto i8Type = gen.builder_.getI8Type();
  auto i8PtrType = gen.builder_.getType<mlir::cxx::PointerType>(i8Type);

  auto existingFunc =
      gen.module_.lookupSymbol<mlir::cxx::FuncOp>(operatorDeleteName);
  if (!existingFunc) {
    auto guard = mlir::OpBuilder::InsertionGuard(gen.builder_);
    gen.builder_.setInsertionPointToStart(gen.module_.getBody());

    mlir::SmallVector<mlir::Type> paramTypes{i8PtrType};
    mlir::SmallVector<mlir::Type> resultTypes;
    auto funcType = mlir::cxx::FunctionType::get(gen.builder_.getContext(),
                                                 paramTypes, resultTypes,
                                                 /*isVariadic=*/false);
    auto linkageAttr = mlir::cxx::LinkageKindAttr::get(
        gen.builder_.getContext(), mlir::cxx::LinkageKind::External);
    auto inlineAttr = mlir::cxx::InlineKindAttr::get(
        gen.builder_.getContext(), mlir::cxx::InlineKind::NoInline);
    existingFunc = mlir::cxx::FuncOp::create(
        gen.builder_, loc, operatorDeleteName, funcType, linkageAttr,
        inlineAttr, mlir::ArrayAttr{}, mlir::ArrayAttr{});
  }

  mlir::SmallVector<mlir::Value> args{ptrValue};
  mlir::SmallVector<mlir::Type> resultTypes;
  mlir::cxx::CallOp::create(gen.builder_, loc, resultTypes,
                            existingFunc.getSymName(), args, mlir::TypeAttr{});

  return {};
}

auto Codegen::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = gen.expression(ast->expression);

  return expressionResult;
}

auto Codegen::ExpressionVisitor::emitLValueToRValueConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto expressionResult = gen.expression(ast->expression);

  if (control()->is_reference(ast->expression->type)) {
    return expressionResult;
  }

  auto resultType = gen.convertType(ast->type);

  if (expressionResult.value.getType() == resultType) {
    return expressionResult;
  }

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

      if (control()->is_signed(ast->expression->type)) {
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

      if (control()->is_floating_point(ast->type)) {
        // int to float
        if (control()->is_signed(ast->expression->type)) {
          auto op = mlir::arith::SIToFPOp::create(gen.builder_, loc, resultType,
                                                  expressionResult.value);
          return {op};
        }

        auto op = mlir::arith::UIToFPOp::create(gen.builder_, loc, resultType,
                                                expressionResult.value);

        return {op};
      }

      if (control()->is_integral(ast->type)) {
        // float to int
        if (control()->is_signed(ast->type)) {
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
    case ImplicitCastKind::kQualificationConversion:
      return expressionResult;

    case ImplicitCastKind::kPointerConversion: {
      if (expressionResult.value &&
          mlir::isa<mlir::IntegerType>(expressionResult.value.getType())) {
        auto op =
            mlir::cxx::NullPtrConstantOp::create(gen.builder_, loc, resultType);

        return {op};
      }

      return expressionResult;
    }

    case ImplicitCastKind::kArrayToPointerConversion: {
      auto op = mlir::cxx::ArrayToPointerOp::create(
          gen.builder_, loc, resultType, expressionResult.value);

      return {op};
    }

    case ImplicitCastKind::kBooleanConversion: {
      if (!control()->is_pointer(ast->expression->type)) break;
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
  }  // switch

  auto op =
      gen.emitTodoExpr(ast->firstSourceLocation(), to_string(ast->kind()));

  return {op};
}

auto Codegen::ExpressionVisitor::emitUserDefinedConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto resultType = gen.convertType(ast->type);

  auto* innerExpr = ast->expression;
  auto* sourceType = innerExpr ? innerExpr->type : nullptr;
  auto* sourceUnqual =
      sourceType ? control()->remove_cv(control()->remove_reference(sourceType))
                 : nullptr;
  auto* destType = ast->type;
  auto* destUnqual =
      destType ? control()->remove_cv(control()->remove_reference(destType))
               : nullptr;

  if (auto* srcClassType = type_cast<ClassType>(sourceUnqual)) {
    if (auto* srcClass = srcClassType->symbol()) {
      for (auto* convFunc : srcClass->conversionFunctions()) {
        auto* convFuncType = type_cast<FunctionType>(convFunc->type());
        if (!convFuncType) continue;
        auto* retType = convFuncType->returnType();
        if (!retType) continue;
        auto* retUnqual = control()->remove_cv(retType);
        if (control()->is_same(retUnqual, destUnqual) ||
            (control()->is_arithmetic(retUnqual) &&
             control()->is_arithmetic(destUnqual))) {
          auto exprResult = gen.expression(innerExpr);

          auto objectValue = exprResult.value;
          if (!objectValue) break;

          auto ptrType = objectValue.getType();

          if (!mlir::isa<mlir::cxx::PointerType>(ptrType)) {
            auto temp = gen.newTemp(sourceType, ast->firstSourceLocation());
            mlir::cxx::StoreOp::create(gen.builder_, loc, objectValue,
                                       temp.getResult(),
                                       gen.getAlignment(sourceType));
            objectValue = temp.getResult();
          }

          return gen.emitCall(ast->firstSourceLocation(), convFunc,
                              {objectValue}, {});
        }
      }
    }
  }

  if (auto* dstClassType = type_cast<ClassType>(destUnqual)) {
    if (auto* dstClass = dstClassType->symbol()) {
      for (auto* ctor : dstClass->convertingConstructors()) {
        auto* funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (params.size() != 1) continue;
        auto* paramUnqual =
            control()->remove_cv(control()->remove_reference(params[0]));
        if (control()->is_same(sourceUnqual, paramUnqual) ||
            (control()->is_arithmetic(sourceUnqual) &&
             control()->is_arithmetic(paramUnqual))) {
          auto temp = gen.newTemp(destType, ast->firstSourceLocation());

          auto argResult = gen.expression(innerExpr);

          (void)gen.emitCall(ast->firstSourceLocation(), ctor,
                             {temp.getResult()}, {argResult});
          return ExpressionResult{temp.getResult()};
        }
      }
    }
  }

  return gen.expression(ast->expression);
}

auto Codegen::ExpressionVisitor::operator()(ImplicitCastExpressionAST* ast)
    -> ExpressionResult {
  auto loc = gen.getLocation(ast->firstSourceLocation());

  switch (ast->castKind) {
    case ImplicitCastKind::kLValueToRValueConversion:
      return emitLValueToRValueConversion(ast);

    case ImplicitCastKind::kIntegralPromotion:
    case ImplicitCastKind::kIntegralConversion:
    case ImplicitCastKind::kFloatingPointPromotion:
    case ImplicitCastKind::kFloatingPointConversion:
    case ImplicitCastKind::kFloatingIntegralConversion:
      return emitNumericConversion(ast);

    case ImplicitCastKind::kFunctionToPointerConversion:
    case ImplicitCastKind::kArrayToPointerConversion:
    case ImplicitCastKind::kQualificationConversion:
    case ImplicitCastKind::kPointerConversion:
    case ImplicitCastKind::kBooleanConversion:
      return emitPointerConversion(ast);

    case ImplicitCastKind::kUserDefinedConversion:
      return emitUserDefinedConversion(ast);

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

    auto trueValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 1));

    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               trueValue, t,
                               gen.getAlignment(control()->getBoolType()));

    auto endLoc = gen.getLocation(ast->lastSourceLocation());
    gen.branch(endLoc, endBlock);

    // build the false block
    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 0));
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t,
                               gen.getAlignment(control()->getBoolType()));
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    // place the end block
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
    gen.condition(ast->rightExpression, trueBlock, falseBlock);

    // build the true block
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

    // build the false block
    gen.builder_.setInsertionPointToEnd(falseBlock);
    auto falseValue = mlir::arith::ConstantOp::create(
        gen.builder_, gen.getLocation(ast->opLoc), i1type,
        gen.builder_.getIntegerAttr(i1type, 0));
    mlir::cxx::StoreOp::create(gen.builder_, gen.getLocation(ast->opLoc),
                               falseValue, t,
                               gen.getAlignment(control()->getBoolType()));
    gen.branch(gen.getLocation(ast->lastSourceLocation()), endBlock);

    // place the end block
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
    if (ast->symbol->parent()->isClass() && !ast->symbol->isStatic()) {
      return gen.emitCall(ast->opLoc, ast->symbol, leftExpressionResult,
                          {rightExpressionResult});
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
  }  // switch

  auto op = gen.emitTodoExpr(loc, "float arithmetic operator");

  return {op};
}

auto Codegen::ExpressionVisitor::emitBinaryArithmeticOpIntegral(
    SourceLocation loc, TokenKind binop, mlir::Type resultType,
    const Type* leftType, mlir::Value left, mlir::Value right)
    -> ExpressionResult {
  auto mlirLoc = gen.getLocation(loc);
  bool isSigned = control()->is_signed(leftType);
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
    case TokenKind::T_PLUS:
      return {mlir::cxx::PtrAddOp::create(gen.builder_, mlirLoc, resultType,
                                          left, right)};
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
  if (control()->is_floating_point(leftType)) {
    return emitBinaryArithmeticOpFloat(loc, op, resultType, left, right);
  }

  if (control()->is_integral(leftType)) {
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

  if (binOp == TokenKind::T_LESS_LESS) {
    return {mlir::arith::ShLIOp::create(gen.builder_, loc, resultType, left,
                                        right)};
  }

  if (control()->is_signed(leftType)) {
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
  bool isSigned = control()->is_signed(leftType);
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
  // Convert pointers to integers for comparison
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
  if (control()->is_floating_point(leftType)) {
    return emitBinaryComparisonOpFloat(loc, op, resultType, left, right);
  }

  if (control()->is_integral_or_unscoped_enum(leftType) ||
      control()->is_pointer(leftType) || control()->is_null_pointer(leftType)) {
    if (control()->is_pointer(leftType)) {
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
      if (control()->is_pointer(leftExpression->type) ||
          control()->is_pointer(rightExpression->type)) {
        return emitBinaryArithmeticOpPointer(opLoc, op, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
      }
      return emitBinaryArithmeticOp(opLoc, op, resultType, leftExpression->type,
                                    leftExpressionResult.value,
                                    rightExpressionResult.value);

    case TokenKind::T_MINUS:
      if (control()->is_pointer(leftExpression->type)) {
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

  auto type = ast->type;
  if (ast->valueCategory != ValueCategory::kPrValue) {
    type = control()->getPointerType(type);
  }

  auto t = gen.newTemp(type, ast->questionLoc);

  gen.condition(ast->condition, trueBlock, falseBlock);

  auto endLoc = gen.getLocation(ast->lastSourceLocation());

  // place the true block
  gen.builder_.setInsertionPointToEnd(trueBlock);
  auto trueExpressionResult = gen.expression(ast->iftrueExpression);
  auto trueValue = mlir::cxx::StoreOp::create(
      gen.builder_, gen.getLocation(ast->questionLoc),
      trueExpressionResult.value, t, gen.getAlignment(type));
  gen.branch(endLoc, endBlock);

  // place the false block
  gen.builder_.setInsertionPointToEnd(falseBlock);
  auto falseExpressionResult = gen.expression(ast->iffalseExpression);
  auto falseValue = mlir::cxx::StoreOp::create(
      gen.builder_, gen.getLocation(ast->colonLoc), falseExpressionResult.value,
      t, gen.getAlignment(type));
  gen.branch(endLoc, endBlock);

  // place the end block
  gen.builder_.setInsertionPointToEnd(endBlock);

  if (format == ExpressionFormat::kSideEffect) return {};

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
                          {rightExpressionResult});
    } else {
      return gen.emitCall(ast->opLoc, ast->symbol, {},
                          {leftExpressionResult, rightExpressionResult});
    }
  }

  if (ast->op == TokenKind::T_EQUAL) {
    auto leftExpressionResult = gen.expression(ast->leftExpression);
    auto rightExpressionResult = gen.expression(ast->rightExpression);

    // Generate a store operation
    const auto loc = gen.getLocation(ast->opLoc);

    mlir::cxx::StoreOp::create(gen.builder_, loc, rightExpressionResult.value,
                               leftExpressionResult.value,
                               gen.getAlignment(ast->leftExpression->type));

    if (format == ExpressionFormat::kSideEffect) {
      return {};
    }

    if (gen.unit_->language() == LanguageKind::kC) {
      // in C mode the result of the assignment is an rvalue
      auto resultLoc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = gen.convertType(ast->leftExpression->type);

      // generate a load
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
                          {rightExpressionResult});
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

  mlir::cxx::StoreOp::create(gen.builder_, loc, sourceExpressionResult.value,
                             targetExpressionResult.value,
                             gen.getAlignment(ast->type));

  if (format == ExpressionFormat::kSideEffect) {
    return {};
  }

  if (gen.unit_->language() == LanguageKind::kC) {
    auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
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

  auto local = gen.findOrCreateLocal(var);
  const auto loc = gen.getLocation(ast->firstSourceLocation());

  if (!local.has_value()) {
    gen.unit_->error(
        ast->firstSourceLocation(),
        std::format("cannot find local variable '{}'", to_string(var->name())));
    return {};
  }

  if (gen.control()->is_array(var->type())) {
    gen.arrayInit(local.value(), var->type(), ast->initializer);
  } else if (gen.control()->is_class(var->type())) {
    if (auto ctor = var->constructor()) {
      std::vector<ExpressionResult> args;
      if (ast->initializer) {
        if (auto paren = ast_cast<ParenInitializerAST>(ast->initializer)) {
          for (auto it = paren->expressionList; it; it = it->next) {
            args.push_back(gen.expression(it->value));
          }
        } else if (auto braced =
                       ast_cast<BracedInitListAST>(ast->initializer)) {
          for (auto it = braced->expressionList; it; it = it->next) {
            args.push_back(gen.expression(it->value));
          }
        } else if (auto equal =
                       ast_cast<EqualInitializerAST>(ast->initializer)) {
          args.push_back(gen.expression(equal->expression));
        }
      }
      (void)gen.emitCall(ast->initializer
                             ? ast->initializer->firstSourceLocation()
                             : var->location(),
                         ctor, {local.value()}, args);
    } else {
      if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
        if (auto braced = ast_cast<BracedInitListAST>(equal->expression)) {
          braced->type = var->type();
        }
      }
    }
  } else {
    if (ast->initializer) {
      if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
        auto expressionResult = gen.expression(equal->expression);
        mlir::cxx::StoreOp::create(gen.builder_, loc, expressionResult.value,
                                   local.value(),
                                   gen.getAlignment(var->type()));
      } else {
        auto expressionResult = gen.expression(ast->initializer);
        mlir::cxx::StoreOp::create(gen.builder_, loc, expressionResult.value,
                                   local.value(),
                                   gen.getAlignment(var->type()));
      }
    }
  }

  auto type = gen.convertType(var->type());
  auto val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, local.value(),
                                       gen.getAlignment(var->type()));

  return {val};
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
  if (!ast->type) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "braced-init-list without type")};
  }

  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(ast->type);
  auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(type);
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

  auto braced = ast_cast<BracedInitListAST>(init);
  if (!braced) return;

  auto elementType = control()->get_element_type(type);
  auto elementMlirType = convertType(elementType);
  auto resultType = builder_.getType<mlir::cxx::PointerType>(elementMlirType);
  auto intType = builder_.getIntegerType(32);

  int index = 0;

  for (auto node : ListView{braced->expressionList}) {
    auto loc = getLocation(node->firstSourceLocation());

    auto indexOp = mlir::arith::ConstantOp::create(
        builder_, loc, intType, builder_.getIntegerAttr(intType, index++));

    auto elementAddress = mlir::cxx::PtrAddOp::create(
        builder_, loc, resultType, address, indexOp.getResult());

    if (control()->is_array(elementType)) {
      arrayInit(elementAddress, elementType, node);
    } else {
      auto value = expression(node);
      mlir::cxx::StoreOp::create(builder_, loc, value.value, elementAddress,
                                 getAlignment(elementType));
    }
  }
}

void Codegen::emitAggregateInit(mlir::Value address, const Type* type,
                                BracedInitListAST* ast) {
  auto loc = getLocation(ast->firstSourceLocation());

  if (auto size = control()->memoryLayout()->sizeOf(type)) {
    mlir::cxx::MemSetZeroOp::create(builder_, loc, address, *size);
  }

  if (control()->is_array(type)) {
    auto elementType = control()->get_element_type(type);
    auto elementMlirType = convertType(elementType);
    auto resultType = builder_.getType<mlir::cxx::PointerType>(elementMlirType);
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
      } else {
        auto val = expression(node);
        mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, elementAddress,
                                   getAlignment(elementType));
      }
      ++index;
    }
  } else if (control()->is_class_or_union(type)) {
    auto classType = type_cast<ClassType>(control()->remove_cv(type));
    if (!classType || !classType->symbol()) return;
    auto classSymbol = classType->symbol();

    if (classType->isUnion()) {
      auto it = ast->expressionList;
      if (!it) return;

      auto& expr = it->value;

      FieldSymbol* targetField = nullptr;

      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
        emitDesignatedInit(address, type, desig);
        return;
      }

      // No designator: initialize first non-static field
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

      auto memberMlirType = convertType(targetField->type());
      auto memberPtrType =
          builder_.getType<mlir::cxx::PointerType>(memberMlirType);
      auto elemLoc = getLocation(expr->firstSourceLocation());

      auto memberAddr = mlir::cxx::MemberOp::create(
          builder_, elemLoc, memberPtrType, address, memberIndex);

      if (auto nested = ast_cast<BracedInitListAST>(expr)) {
        emitAggregateInit(memberAddr, targetField->type(), nested);
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
        if (layout) {
          if (auto fi = layout->getFieldInfo(field)) {
            memberIndex = fi->index;
          }
        }

        auto memberMlirType = convertType(field->type());
        auto memberPtrType =
            builder_.getType<mlir::cxx::PointerType>(memberMlirType);
        auto elemLoc = getLocation(node->firstSourceLocation());

        auto memberAddr = mlir::cxx::MemberOp::create(
            builder_, elemLoc, memberPtrType, address, memberIndex);

        if (auto nested = ast_cast<BracedInitListAST>(node)) {
          emitAggregateInit(memberAddr, field->type(), nested);
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

  for (auto desigIt = ast->designatorList; desigIt; desigIt = desigIt->next) {
    auto designator = desigIt->value;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto field = dot->symbol;
      if (!field) return;

      auto classType = type_cast<ClassType>(control()->remove_cv(currentType));
      if (!classType || !classType->symbol()) return;

      auto classSymbol = classType->symbol();
      auto layout = classSymbol->layout();

      std::uint32_t memberIndex = 0;
      if (layout) {
        if (auto fi = layout->getFieldInfo(field)) {
          memberIndex = fi->index;
        }
      }

      auto memberMlirType = convertType(field->type());
      auto memberPtrType =
          builder_.getType<mlir::cxx::PointerType>(memberMlirType);
      auto elemLoc = getLocation(dot->firstSourceLocation());

      currentAddr = mlir::cxx::MemberOp::create(
          builder_, elemLoc, memberPtrType, currentAddr, memberIndex);
      currentType = control()->remove_cv(field->type());

    } else if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      auto elementType = control()->get_element_type(currentType);
      auto elementMlirType = convertType(elementType);
      auto resultType =
          builder_.getType<mlir::cxx::PointerType>(elementMlirType);
      auto elemLoc = getLocation(subscript->firstSourceLocation());

      auto indexVal = expression(subscript->expression);
      currentAddr = mlir::cxx::PtrAddOp::create(builder_, elemLoc, resultType,
                                                currentAddr, indexVal.value);
      currentType = control()->remove_cv(elementType);
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
    mlir::cxx::StoreOp::create(builder_, elemLoc, val.value, currentAddr,
                               getAlignment(currentType));
  }
}

auto Codegen::emitCall(SourceLocation loc, FunctionSymbol* symbol,
                       ExpressionResult thisValue,
                       std::vector<ExpressionResult> arguments)
    -> ExpressionResult {
  auto functionType = type_cast<FunctionType>(symbol->type());
  if (!functionType) return {};

  auto mlirLoc = getLocation(loc);

  mlir::SmallVector<mlir::Value> args;
  if (thisValue.value) {
    args.push_back(thisValue.value);
  }

  for (size_t i = 0; i < arguments.size(); ++i) {
    args.push_back(arguments[i].value);
  }

  mlir::SmallVector<mlir::Type> resultTypes;
  if (!control()->is_void(functionType->returnType())) {
    resultTypes.push_back(convertType(functionType->returnType()));
  }
  auto funcOp = findOrCreateFunction(symbol);
  auto callOp =
      mlir::cxx::CallOp::create(builder_, mlirLoc, resultTypes,
                                funcOp.getSymName(), args, mlir::TypeAttr{});

  if (functionType->isVariadic()) {
    callOp.setVarCalleeType(
        mlir::cast<mlir::cxx::FunctionType>(convertType(functionType)));
  }

  if (resultTypes.empty()) return {};

  return {callOp.getResult()};
}

}  // namespace cxx
