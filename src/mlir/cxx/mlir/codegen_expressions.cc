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
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// mlir
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

  auto binaryExpression(SourceLocation opLoc, TokenKind op,
                        mlir::Type resultType, ExpressionAST* leftExpression,
                        ExpressionAST* rightExpression,
                        ExpressionResult leftExpressionResult,
                        ExpressionResult rightExpressionResult)
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
  if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
    mlir::Value val;
    bool found = false;

    if (auto local = gen.findOrCreateLocal(ast->symbol)) {
      val = local.value();
      found = true;
    } else if (auto it = gen.globalOps_.find(var); it != gen.globalOps_.end()) {
      auto loc = gen.getLocation(ast->firstSourceLocation());
      auto resultType = mlir::cxx::PointerType::get(
          gen.builder_.getContext(), gen.convertType(var->type()));
      val = mlir::cxx::AddressOfOp::create(gen.builder_, loc, resultType,
                                           it->second.getSymName());
      found = true;
    }

    if (found) {
      if (gen.control()->is_reference(var->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(var->type());  // pointer type
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val);
      }
      return {val};
    }
  } else if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    if (auto local = gen.findOrCreateLocal(ast->symbol)) {
      auto val = local.value();
      if (gen.control()->is_reference(param->type())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(param->type());  // pointer type
        val = mlir::cxx::LoadOp::create(gen.builder_, loc, type, val);
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

      int fieldIndex = 0;
      auto classSymbol = symbol_cast<ClassSymbol>(field->parent());
      for (auto member : cxx::views::members(classSymbol)) {
        auto f = symbol_cast<FieldSymbol>(member);
        if (!f) continue;
        if (f->isStatic()) continue;
        if (member == field) break;
        ++fieldIndex;
      }

      auto loc = gen.getLocation(ast->firstSourceLocation());

      auto thisPtrType =
          gen.convertType(gen.control()->getPointerType(classSymbol->type()));

      auto thisPtr = mlir::cxx::LoadOp::create(gen.builder_, loc, thisPtrType,
                                               gen.thisValue_);

      auto resultType =
          gen.convertType(gen.control()->add_pointer(field->type()));

      auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, resultType,
                                            thisPtr, fieldIndex);
      return {op};
    }
  } else if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    if (enumerator->value().has_value()) {
      if (auto val = std::get_if<std::intmax_t>(&enumerator->value().value())) {
        auto loc = gen.getLocation(ast->firstSourceLocation());
        auto type = gen.convertType(enumerator->type());
        auto op =
            mlir::cxx::IntConstantOp::create(gen.builder_, loc, type, *val);
        return {op};
      }
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
  // strip nested expressions
  auto func = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(func)) {
    func = nested->expression;
  }

  // check for direct calls
  auto id = ast_cast<IdExpressionAST>(func);
  auto member = ast_cast<MemberExpressionAST>(func);
  ExpressionResult thisValue;

  FunctionSymbol* functionSymbol = nullptr;
  if (id) {
    functionSymbol = symbol_cast<FunctionSymbol>(id->symbol);
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

  if (isIndirectCall) {
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

namespace {

[[nodiscard]] auto findPath(ClassSymbol* current, ClassSymbol* target,
                            std::vector<int>& path) -> bool {
  if (!current) return false;
  if (current == target) return true;
  int baseIndex = 0;
  for (auto base : current->baseClasses()) {
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) {
      if (const auto* baseType = type_cast<ClassType>(base->type())) {
        baseSym = symbol_cast<ClassSymbol>(baseType->symbol());
      }
    }

    if (baseSym) {
      path.push_back(baseIndex);
      if (findPath(baseSym, target, path)) return true;
      path.pop_back();
    }
    baseIndex++;
  }
  return false;
}

}  // namespace

auto Codegen::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && !field->isStatic()) {
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

    mlir::Value currentPtr = baseExpressionResult.value;

    if (startClass != classSymbol) {
      std::vector<int> path;

      if (findPath(startClass, classSymbol, path)) {
        // Apply path
        auto current = startClass;
        for (int baseIdx : path) {
          auto base = current->baseClasses()[baseIdx];
          const Type* baseType = base->type();
          if (!baseType && base->symbol()) baseType = base->symbol()->type();

          if (!baseType) break;

          auto ptrType = gen.convertType(gen.control()->add_pointer(baseType));
          auto loc = gen.getLocation(ast->firstSourceLocation());
          currentPtr = mlir::cxx::MemberOp::create(gen.builder_, loc, ptrType,
                                                   currentPtr, baseIdx);

          if (base->symbol()) {
            current = symbol_cast<ClassSymbol>(base->symbol());
          } else {
            current = symbol_cast<ClassSymbol>(
                type_cast<ClassType>(baseType)->symbol());
          }
        }
      } else {
        return {gen.emitTodoExpr(ast->firstSourceLocation(),
                                 "cannot find base path")};
      }
    }

    // Adjust fieldIndex to account for bases in the target class
    fieldIndex += classSymbol->baseClasses().size();

    auto loc = gen.getLocation(ast->unqualifiedId->firstSourceLocation());

    auto resultType = gen.convertType(control()->add_pointer(ast->type));

    auto op = mlir::cxx::MemberOp::create(gen.builder_, loc, resultType,
                                          currentPtr, fieldIndex);

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
  if (control()->is_pointer(ast->baseExpression->type)) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrTy =
        mlir::cast<mlir::cxx::PointerType>(expressionResult.value.getType());
    auto elementTy = ptrTy.getElementType();
    auto loadOp = mlir::cxx::LoadOp::create(gen.builder_, loc, elementTy,
                                            expressionResult.value);
    auto resultTy = gen.convertType(ast->baseExpression->type);
    auto intTy =
        mlir::cxx::IntegerType::get(gen.builder_.getContext(), 32, true);
    auto oneOp = mlir::cxx::IntConstantOp::create(
        gen.builder_, loc, intTy, ast->op == TokenKind::T_PLUS_PLUS ? 1 : -1);
    auto addOp =
        mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultTy, loadOp, oneOp);
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
  if (auto size = ast->value) {
    auto resultlType = gen.convertType(ast->type);
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto op = mlir::cxx::IntConstantOp::create(gen.builder_, loc, resultlType,
                                               size.value());
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
    auto op = mlir::cxx::IntConstantOp::create(gen.builder_, loc, resultlType,
                                               size.value());
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
    case ImplicitCastKind::kFunctionToPointerConversion: {
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

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

      if (is_bool(ast->type)) {
        auto op = mlir::cxx::FloatToBoolOp::create(
            gen.builder_, loc, resultType, expressionResult.value);
        return {op};
      }

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

    case ImplicitCastKind::kPointerConversion: {
      auto expressionResult = gen.expression(ast->expression);
      return expressionResult;
    }

    case ImplicitCastKind::kBooleanConversion: {
      if (control()->is_pointer(ast->expression->type)) {
        // generate a pointer to bool cast
        auto expressionResult = gen.expression(ast->expression);
        auto resultType = gen.convertType(ast->type);

        auto op = mlir::cxx::PtrToBoolOp::create(gen.builder_, loc, resultType,
                                                 expressionResult.value);

        return {op};
      }
      break;
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

  auto leftExpressionResult = gen.expression(ast->leftExpression);
  auto rightExpressionResult = gen.expression(ast->rightExpression);
  auto resultType = gen.convertType(ast->type);

  return binaryExpression(ast->opLoc, ast->op, resultType, ast->leftExpression,
                          ast->rightExpression, leftExpressionResult,
                          rightExpressionResult);
}

auto Codegen::ExpressionVisitor::binaryExpression(
    SourceLocation opLoc, TokenKind op, mlir::Type resultType,
    ExpressionAST* leftExpression, ExpressionAST* rightExpression,
    ExpressionResult leftExpressionResult,
    ExpressionResult rightExpressionResult) -> ExpressionResult {
  auto loc = gen.getLocation(opLoc);

  switch (op) {
    case TokenKind::T_PLUS: {
      if (control()->is_integral(leftExpression->type)) {
        auto op = mlir::cxx::AddIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::AddFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_pointer(leftExpression->type) &&
          control()->is_integer(rightExpression->type)) {
        auto op = mlir::cxx::PtrAddOp::create(gen.builder_, loc, resultType,
                                              leftExpressionResult.value,
                                              rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_MINUS: {
      if (control()->is_pointer(leftExpression->type) &&
          control()->is_integer(rightExpression->type)) {
        auto offsetType = gen.convertType(rightExpression->type);

        auto zero =
            mlir::cxx::IntConstantOp::create(gen.builder_, loc, offsetType, 0);

        auto offset = mlir::cxx::SubIOp::create(
            gen.builder_, loc, offsetType, zero, rightExpressionResult.value);

        auto op = mlir::cxx::PtrAddOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value, offset);

        return {op};
      }

      if (control()->is_integral(leftExpression->type)) {
        auto op = mlir::cxx::SubIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::SubFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_pointer(leftExpression->type) &&
          control()->is_pointer(rightExpression->type)) {
        auto op = mlir::cxx::PtrDiffOp::create(
            gen.builder_, loc, gen.convertType(control()->getLongIntType()),
            leftExpressionResult.value, rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_STAR: {
      if (control()->is_integral(leftExpression->type)) {
        auto op = mlir::cxx::MulIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::MulFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_SLASH: {
      if (control()->is_integral(leftExpression->type)) {
        auto op = mlir::cxx::DivIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::DivFOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_PERCENT: {
      if (control()->is_integral(leftExpression->type)) {
        auto op = mlir::cxx::ModIOp::create(gen.builder_, loc, resultType,
                                            leftExpressionResult.value,
                                            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_LESS: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type)) {
        auto op = mlir::cxx::ShiftLeftOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_GREATER: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type)) {
        auto op = mlir::cxx::ShiftRightOp::create(gen.builder_, loc, resultType,
                                                  leftExpressionResult.value,
                                                  rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EQUAL_EQUAL: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type) ||
          control()->is_null_pointer(leftExpression->type)) {
        auto op = mlir::cxx::EqualOp::create(gen.builder_, loc, resultType,
                                             leftExpressionResult.value,
                                             rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::EqualFOp::create(gen.builder_, loc, resultType,
                                              leftExpressionResult.value,
                                              rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_EXCLAIM_EQUAL: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type) ||
          control()->is_null_pointer(leftExpression->type)) {
        auto op = mlir::cxx::NotEqualOp::create(gen.builder_, loc, resultType,
                                                leftExpressionResult.value,
                                                rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::NotEqualFOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type)) {
        auto op = mlir::cxx::LessThanOp::create(gen.builder_, loc, resultType,
                                                leftExpressionResult.value,
                                                rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::LessThanFOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_LESS_EQUAL: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type)) {
        auto op = mlir::cxx::LessEqualOp::create(gen.builder_, loc, resultType,
                                                 leftExpressionResult.value,
                                                 rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::LessEqualFOp::create(gen.builder_, loc, resultType,
                                                  leftExpressionResult.value,
                                                  rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type)) {
        auto op = mlir::cxx::GreaterThanOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::GreaterThanFOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_GREATER_EQUAL: {
      if (control()->is_integral_or_unscoped_enum(leftExpression->type) ||
          control()->is_pointer(leftExpression->type)) {
        auto op = mlir::cxx::GreaterEqualOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      if (control()->is_floating_point(leftExpression->type)) {
        auto op = mlir::cxx::GreaterEqualFOp::create(
            gen.builder_, loc, resultType, leftExpressionResult.value,
            rightExpressionResult.value);
        return {op};
      }

      break;
    }

    case TokenKind::T_CARET: {
      auto op = mlir::cxx::XorOp::create(gen.builder_, loc, resultType,
                                         leftExpressionResult.value,
                                         rightExpressionResult.value);
      return {op};
    }

    case TokenKind::T_AMP: {
      auto op = mlir::cxx::AndOp::create(gen.builder_, loc, resultType,
                                         leftExpressionResult.value,
                                         rightExpressionResult.value);
      return {op};
    }

    case TokenKind::T_BAR: {
      auto op = mlir::cxx::OrOp::create(gen.builder_, loc, resultType,
                                        leftExpressionResult.value,
                                        rightExpressionResult.value);
      return {op};
    }

    default:
      break;
  }  // switch

  auto resultOp = gen.emitTodoExpr(opLoc, to_string(BinaryExpressionAST::Kind));

  return {resultOp};
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
  auto targetExpressionResult = gen.expression(ast->targetExpression);

  auto targetValue = targetExpressionResult.value;

  std::swap(gen.targetValue_, targetValue);
  auto leftExpressionResult = gen.expression(ast->leftExpression);
  std::swap(gen.targetValue_, targetValue);

  auto rightExpressionResult = gen.expression(ast->rightExpression);

  auto resultType = gen.convertType(ast->type);

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
                             targetExpressionResult.value);

  if (format == ExpressionFormat::kSideEffect) {
    return {};
  }

  if (gen.unit_->language() == LanguageKind::kC) {
    auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, resultType,
                                        targetExpressionResult.value);
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
  if (!ast->type) {
    return {gen.emitTodoExpr(ast->firstSourceLocation(),
                             "braced-init-list without type")};
  }

  auto loc = gen.getLocation(ast->firstSourceLocation());
  auto type = gen.convertType(ast->type);
  auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(type);
  auto temp = mlir::cxx::AllocaOp::create(gen.builder_, loc, ptrType);

  int index = 0;
  for (auto node : ListView{ast->expressionList}) {
    mlir::Value memberAddr;

    // Handle Class Type
    if (gen.control()->is_class(ast->type)) {
      // member_index attribute is expected to be explicit I32 attr or similar
      // in create? MemberOp::create(Builder, Loc, ResultType, Base, Index)
      // ResultType must be pointer to member type.
      // We need to resolve member type?
      // MemberOp lowering handled GEP.
      // Cxx dialect MemberOp definition?
      // We need result type.
      // We can iterate fields of the class to get the type of the ith field.

      auto classSymbol = symbol_cast<ClassSymbol>(
          type_cast<ClassType>(gen.control()->remove_cv(ast->type))->symbol());

      // Just find the ith field.
      int fieldCount = 0;
      Symbol* fieldSym = nullptr;
      for (auto member : views::members(classSymbol)) {
        if (auto f = symbol_cast<FieldSymbol>(member)) {
          if (f->isStatic()) continue;
          if (fieldCount == index) {
            fieldSym = f;
            break;
          }
          fieldCount++;
        }
      }

      if (!fieldSym) {
        // Error or mismatch?
        continue;  // or break
      }

      auto memberType = gen.convertType(fieldSym->type());
      auto memberPtrType =
          gen.builder_.getType<mlir::cxx::PointerType>(memberType);

      memberAddr = mlir::cxx::MemberOp::create(gen.builder_, loc, memberPtrType,
                                               temp, index);
    } else if (gen.control()->is_array(ast->type)) {
      // Array handling
      auto elementType = gen.control()->get_element_type(ast->type);
      auto memberType = gen.convertType(elementType);
      auto memberPtrType =
          gen.builder_.getType<mlir::cxx::PointerType>(memberType);
      auto intType = gen.builder_.getType<mlir::cxx::IntegerType>(32, true);
      auto idxVal =
          mlir::cxx::IntConstantOp::create(gen.builder_, loc, intType, index);
      memberAddr = mlir::cxx::PtrAddOp::create(gen.builder_, loc, memberPtrType,
                                               temp, idxVal);
    } else {
      // fallback or scalar braced init? {1}
      // Treat as scalar?
      // For now valid for array/class.
      return {gen.emitTodoExpr(ast->firstSourceLocation(),
                               "braced-init-list non-aggregate")};
    }

    auto val = gen.expression(node).value;
    mlir::cxx::StoreOp::create(gen.builder_, loc, val, memberAddr);

    index++;
  }

  auto op = mlir::cxx::LoadOp::create(gen.builder_, loc, type, temp);
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
  auto intType = builder_.getType<mlir::cxx::IntegerType>(32, true);

  int index = 0;

  for (auto node : ListView{braced->expressionList}) {
    auto loc = getLocation(node->firstSourceLocation());

    auto indexOp =
        mlir::cxx::IntConstantOp::create(builder_, loc, intType, index++);

    auto elementAddress = mlir::cxx::PtrAddOp::create(
        builder_, loc, resultType, address, indexOp.getResult());

    if (control()->is_array(elementType)) {
      arrayInit(elementAddress, elementType, node);
    } else {
      auto value = expression(node);
      mlir::cxx::StoreOp::create(builder_, loc, value.value, elementAddress);
    }
  }
}

}  // namespace cxx
