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

auto Codegen::operator()(InitDeclaratorAST* ast) -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = operator()(ast->declarator);
  auto requiresClauseResult = operator()(ast->requiresClause);
  auto initializerResult = expression(ast->initializer);

  return {};
}

auto Codegen::operator()(DeclaratorAST* ast) -> DeclaratorResult {
  if (!ast) return {};

  for (auto node : ListView{ast->ptrOpList}) {
    auto value = operator()(node);
  }

  auto coreDeclaratorResult = operator()(ast->coreDeclarator);

  for (auto node : ListView{ast->declaratorChunkList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(MemInitializerAST* ast) -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(RequiresClauseAST* ast) -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = expression(ast->expression);

  return {};
}

auto Codegen::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = operator()(ast->typeId);

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto unqualifiedIdResult = gen(ast->unqualifiedId);
  auto sizeExpressionResult = gen.expression(ast->sizeExpression);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = gen(ast->coreDeclarator);

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = gen(ast->declarator);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->cvQualifierList}) {
    auto value = gen(node);
  }

  auto exceptionSpecifierResult = gen(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto trailingReturnTypeResult = gen(ast->trailingReturnType);

  return {};
}

auto Codegen::DeclaratorChunkVisitor::operator()(ArrayDeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  auto expressionResult = gen.expression(ast->expression);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
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
  auto typeConstraintResult = gen(ast->typeConstraint);

  return {};
}

auto Codegen::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::MemInitializerVisitor::operator()(ParenMemInitializerAST* ast)
    -> MemInitializerResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->expressionList}) {
    auto value = gen.expression(node);
  }

  return {};
}

auto Codegen::MemInitializerVisitor::operator()(BracedMemInitializerAST* ast)
    -> MemInitializerResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);
  auto bracedInitListResult = gen.expression(ast->bracedInitList);

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