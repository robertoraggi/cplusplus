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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ASTInterpreter::PtrOperatorVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast)
      -> PtrOperatorResult;
};

struct ASTInterpreter::CoreDeclaratorVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorResult;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast)
      -> CoreDeclaratorResult;
};

struct ASTInterpreter::DeclaratorChunkVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
};

struct ASTInterpreter::ExceptionSpecifierVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierResult;
};

struct ASTInterpreter::RequirementVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast)
      -> RequirementResult;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementResult;
};

struct ASTInterpreter::MemInitializerVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(ParenMemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto operator()(BracedMemInitializerAST* ast)
      -> MemInitializerResult;
};

struct ASTInterpreter::LambdaCaptureVisitor {
  ASTInterpreter& interp;

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

auto ASTInterpreter::ptrOperator(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::coreDeclarator(CoreDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::declaratorChunk(DeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::exceptionSpecifier(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::requirement(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::memInitializer(MemInitializerAST* ast)
    -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::lambdaCapture(LambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::initDeclarator(InitDeclaratorAST* ast)
    -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = declarator(ast->declarator);
  auto requiresClauseResult = requiresClause(ast->requiresClause);
  auto initializerResult = expression(ast->initializer);

  return {};
}

auto ASTInterpreter::declarator(DeclaratorAST* ast) -> DeclaratorResult {
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

auto ASTInterpreter::requiresClause(RequiresClauseAST* ast)
    -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto ASTInterpreter::parameterDeclarationClause(
    ParameterDeclarationClauseAST* ast) -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto ASTInterpreter::trailingReturnType(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = interp.specifier(node);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = interp.specifier(node);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(
    BitfieldDeclaratorAST* ast) -> CoreDeclaratorResult {
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);
  auto sizeExpressionResult = interp.expression(ast->sizeExpression);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = interp.coreDeclarator(ast->coreDeclarator);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = interp.declarator(ast->declarator);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult =
      interp.parameterDeclarationClause(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = interp.specifier(node);
  }

  auto exceptionSpecifierResult =
      interp.exceptionSpecifier(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto trailingReturnTypeResult =
      interp.trailingReturnType(ast->trailingReturnType);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto expressionResult = interp.expression(ast->expression);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierResult {
  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = interp.expression(ast->expression);
  auto typeConstraintResult = interp.typeConstraint(ast->typeConstraint);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    ParenMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (value.has_value() && ast->symbol && interp.thisObject()) {
      interp.thisObject()->setField(ast->symbol, std::move(*value));
    }
  }

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);
  auto bracedInitListResult = interp.expression(ast->bracedInitList);

  if (bracedInitListResult.has_value() && ast->symbol && interp.thisObject()) {
    interp.thisObject()->setField(ast->symbol,
                                  std::move(*bracedInitListResult));
  }

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    SimpleLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    RefInitLambdaCaptureAST* ast) -> LambdaCaptureResult {
  auto initializerResult = interp.expression(ast->initializer);

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = interp.expression(ast->initializer);

  return {};
}

}  // namespace cxx
