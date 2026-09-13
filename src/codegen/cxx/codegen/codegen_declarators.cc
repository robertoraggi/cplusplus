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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {
struct Codegen::PtrOperatorVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
      -> PtrOperatorResult;
};

struct Codegen::CoreDeclaratorVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
      -> CoreDeclaratorResult;
};

struct Codegen::DeclaratorChunkVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
};

struct Codegen::MemInitializerVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto emitSubobjectInit(MemInitializerAST* ast,
                                       BracedInitListAST* bracedInitList,
                                       List<ExpressionAST*>* expressionList,
                                       bool isWritten) -> MemInitializerResult;

  [[nodiscard]] auto emitDelegationOrVirtualBaseInit(
      MemInitializerAST* ast, ClassSymbol* targetClass,
      const std::vector<ExpressionResult>& args) -> MemInitializerResult;
};

struct Codegen::LambdaCaptureVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ThisLambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto operator()(DerefThisLambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto operator()(SimpleLambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto operator()(RefLambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto operator()(RefInitLambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto operator()(InitLambdaCaptureAST* ast)
      -> LambdaCaptureResult;
};

struct Codegen::ExceptionSpecifierVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierResult;
};

struct Codegen::RequirementVisitor {
  Codegen& gen;

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
      -> RequirementResult;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementResult;
};

auto Codegen::initDeclarator(InitDeclaratorAST* ast) -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = declarator(ast->declarator);
  auto requiresClauseResult = requiresClause(ast->requiresClause);
  auto initializerResult = expression(ast->initializer);

  return {};
}

auto Codegen::declarator(DeclaratorAST* ast) -> DeclaratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->ptrOpList}) {
    auto value = ptrOperator(node);
  }

  auto coreDeclaratorResult = coreDeclarator(ast->coreDeclarator);

  for (auto node : ListView{ast->declaratorChunkList}) {
    auto value = declaratorChunk(node);
  }

  return {};
}

auto Codegen::ptrOperator(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto Codegen::coreDeclarator(CoreDeclaratorAST* ast) -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto Codegen::declaratorChunk(DeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto Codegen::exceptionSpecifier(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::requirement(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto Codegen::memInitializer(MemInitializerAST* ast) -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto Codegen::lambdaCapture(LambdaCaptureAST* ast) -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto Codegen::requiresClause(RequiresClauseAST* ast) -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::parameterDeclarationClause(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto Codegen::trailingReturnType(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = typeId(ast->typeId);

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen.specifier(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen.specifier(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);
  auto sizeExpressionResult = gen.expression(ast->sizeExpression);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = gen.coreDeclarator(ast->coreDeclarator);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = gen.declarator(ast->declarator);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult =
      gen.parameterDeclarationClause(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen.specifier(node);
  }

  auto exceptionSpecifierResult =
      gen.exceptionSpecifier(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto trailingReturnTypeResult =
      gen.trailingReturnType(ast->trailingReturnType);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(ArrayDeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  auto expressionResult = gen.expression(ast->expression);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierResult {
  return {};
}

auto Codegen::ExceptionSpecifierVisitor::operator()(NoexceptSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  auto typeConstraintResult = gen.typeConstraint(ast->typeConstraint);

  return {};
}

auto Codegen::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerResult {
  return emitSubobjectInit(ast, /*bracedInitList=*/nullptr, ast->expressionList,
                           /*isWritten=*/static_cast<bool>(ast->lparenLoc));
}

auto Codegen::MemInitializerVisitor::operator()(BracedMemInitializerAST* ast)
    -> MemInitializerResult {
  return emitSubobjectInit(
      ast, ast->bracedInitList,
      ast->bracedInitList ? ast->bracedInitList->expressionList : nullptr,
      /*isWritten=*/ast->bracedInitList != nullptr);
}

auto Codegen::MemInitializerVisitor::emitDelegationOrVirtualBaseInit(
    MemInitializerAST* ast, ClassSymbol* targetClass,
    const std::vector<ExpressionResult>& args) -> MemInitializerResult {
  auto loc = ast->firstSourceLocation();

  auto currentClass = symbol_cast<ClassSymbol>(
      gen.currentFunctionSymbol_ ? gen.currentFunctionSymbol_->parent()
                                 : nullptr);
  if (!currentClass || !ast->constructor) return {};

  auto thisPtr = gen.loadThisPointer(loc, currentClass);

  const bool initializesVirtualBase =
      targetClass != currentClass && targetClass != currentClass->definition();

  const bool inCompleteObjectVariant =
      gen.currentFunctionSymbol_->isStructorVariant();

  ir::ValueRef targetPtr = thisPtr;
  if (initializesVirtualBase) {
    if (!inCompleteObjectVariant) return {};
    auto layout = currentClass->layout();
    std::optional<ClassLayout::MemberInfo> baseInfo;
    if (layout) baseInfo = layout->getBaseInfo(targetClass);
    if (!baseInfo) return {};
    targetPtr =
        gen.memberAddress(loc, thisPtr, targetClass->type(), baseInfo->index);
  }

  const bool delegatesToOwnBaseObjectVariant =
      ast->constructor == gen.currentFunctionSymbol_->structorPrincipal();

  const bool delegatesToTargetConstructor =
      !initializesVirtualBase && !delegatesToOwnBaseObjectVariant;

  (void)gen.emitCtorCall(ast->firstSourceLocation(), ast->constructor,
                         targetPtr, args,
                         /*completeObject=*/delegatesToTargetConstructor &&
                             inCompleteObjectVariant);
  return {};
}

auto Codegen::MemInitializerVisitor::emitSubobjectInit(
    MemInitializerAST* ast, BracedInitListAST* bracedInitList,
    List<ExpressionAST*>* expressionList, bool isWritten)
    -> MemInitializerResult {
  auto symbol = ast->symbol;
  if (!symbol) return {};

  if (auto base = symbol_cast<BaseClassSymbol>(symbol);
      base && base->isVirtual()) {
    return {};
  }

  auto loc = ast->firstSourceLocation();

  const Type* targetType = nullptr;
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    targetType = field->type();
  } else if (auto base = symbol_cast<BaseClassSymbol>(symbol)) {
    targetType = base->symbol() ? base->symbol()->type() : nullptr;
  }

  if (!bracedInitList && expressionList && !expressionList->next)
    bracedInitList = Initializer{expressionList->value}.bracedInitList();

  auto wholeObject =
      expressionList && !expressionList->next && targetType &&
      !ast->constructor && is_prvalue(expressionList->value) &&
      expressionList->value->type &&
      gen.traits.is_class_or_union(gen.traits.remove_cv(targetType)) &&
      gen.traits.is_same(gen.traits.remove_cv(expressionList->value->type),
                         gen.traits.remove_cv(targetType));
  const auto aggregateInit =
      bracedInitList && !wholeObject && !ast->constructor && targetType &&
      (gen.traits.is_class_or_union(gen.traits.remove_cv(targetType)) ||
       gen.traits.is_array(targetType));

  std::vector<ExpressionResult> args;
  if (!aggregateInit) {
    for (auto node : ListView{expressionList}) {
      args.push_back(gen.expression(node));
    }
  }

  if (auto targetClass = symbol_cast<ClassSymbol>(symbol)) {
    if (!ast->constructor && args.size() == 1 && args[0].value) {
      auto address = gen.loadThisPointer(loc, targetClass);
      gen.emitter_.store(loc, args[0].value, address,
                         gen.getAlignment(targetClass->type()));
      return {};
    }
    return emitDelegationOrVirtualBaseInit(ast, targetClass, args);
  }

  auto declaringClass = symbol_cast<ClassSymbol>(symbol->parent());
  if (!declaringClass) return {};
  if (!targetType) return {};

  auto classSymbol = symbol_cast<ClassSymbol>(
      gen.currentFunctionSymbol_ ? gen.currentFunctionSymbol_->parent()
                                 : nullptr);
  if (!classSymbol) classSymbol = declaringClass;

  auto thisPtr = gen.loadThisPointer(loc, classSymbol);

  auto fieldPtr = gen.subobjectAddress(loc, thisPtr, classSymbol, symbol);
  if (!fieldPtr) return {};

  if (ast->constructor) {
    const auto completeObject = symbol_cast<FieldSymbol>(symbol) != nullptr;
    auto shape = gen.classSubobjectShape(targetType);
    if (!shape) return {};
    if (isWritten && args.empty() &&
        gen.requiresZeroInitialization(shape->elementType, ast->constructor)) {
      gen.emitZeroInitialization(loc, fieldPtr, targetType);
    }
    for (auto address : gen.subobjectElementAddresses(loc, fieldPtr, *shape)) {
      (void)gen.emitCtorCall(loc, ast->constructor, address, args,
                             completeObject);
    }
    return {};
  }

  if (aggregateInit) {
    gen.emitAggregateInit(fieldPtr, targetType, bracedInitList);
    return {};
  }

  if (gen.hasNoValueRepresentation(targetType)) return {};

  auto layout = declaringClass->layout();
  auto bitField = symbol_cast<FieldSymbol>(symbol);
  if (bitField && !bitField->isBitField()) bitField = nullptr;

  if (args.empty()) {
    if (bitField && layout) {
      if (auto fi = layout->getFieldInfo(bitField)) {
        auto zero = gen.emitter_.constantZero(loc, gen.convertType(targetType));
        gen.emitter_.storeBitfield(
            loc, zero, fieldPtr,
            {fi->bitOffset, fi->bitWidth, fi->allocUnitSizeBytes});
      }
      return {};
    }

    if (auto size = gen.control()->memoryLayout()->sizeOf(targetType))
      gen.emitter_.memsetZero(loc, fieldPtr, *size);

    return {};
  }

  if (args.size() != 1) return {};

  auto value = args[0].value;
  if (!value) {
    (void)gen.emitTodoExpr(ast->firstSourceLocation(),
                           "mem-initializer without a value");
    return {};
  }

  if (bitField && layout) {
    if (auto fi = layout->getFieldInfo(bitField)) {
      gen.emitter_.storeBitfield(
          loc, value, fieldPtr,
          {fi->bitOffset, fi->bitWidth, fi->allocUnitSizeBytes});
      return {};
    }
  }

  if (gen.traits.is_class(gen.traits.remove_cv(targetType)) &&
      (gen.emitter_.typeKind(gen.emitter_.typeOf(value)) ==
       ir::TypeKind::Pointer)) {
    value = gen.emitter_.load(loc, gen.convertType(targetType), value,
                              gen.getAlignment(targetType));
  }

  gen.emitter_.store(loc, value, fieldPtr, 1);

  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(DerefThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(SimpleLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(RefInitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = gen.expression(ast->initializer);

  return {};
}
}  // namespace cxx
