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

struct Codegen::UnqualifiedIdVisitor {
  Codegen& gen;

  auto operator()(NameIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(DestructorIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(OperatorFunctionIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(LiteralOperatorIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(ConversionFunctionIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(SimpleTemplateIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult;
  auto operator()(OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult;
};

struct Codegen::NestedNameSpecifierVisitor {
  Codegen& gen;

  auto operator()(GlobalNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  auto operator()(SimpleNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  auto operator()(DecltypeNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  auto operator()(TemplateNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
};

struct Codegen::TemplateArgumentVisitor {
  Codegen& gen;

  auto operator()(TypeTemplateArgumentAST* ast) -> TemplateArgumentResult;
  auto operator()(ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult;
};

auto Codegen::unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdResult {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto Codegen::nestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierResult {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto Codegen::templateArgument(TemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdResult {
  auto idResult = gen.unqualifiedId(ast->id);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdResult {
  auto decltypeSpecifierResult = gen.specifier(ast->decltypeSpecifier);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdResult {
  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen.templateArgument(node);
  }

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto literalOperatorIdResult = gen.unqualifiedId(ast->literalOperatorId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen.templateArgument(node);
  }

  return {};
}

auto Codegen::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto operatorFunctionIdResult = gen.unqualifiedId(ast->operatorFunctionId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = gen.templateArgument(node);
  }

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto decltypeSpecifierResult = gen.specifier(ast->decltypeSpecifier);

  return {};
}

auto Codegen::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto templateIdResult = gen.unqualifiedId(ast->templateId);

  return {};
}

auto Codegen::TemplateArgumentVisitor::operator()(TypeTemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

}  // namespace cxx
