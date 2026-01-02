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

struct ASTInterpreter::UnqualifiedIdVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(NameIdAST* ast) -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(DestructorIdAST* ast) -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast)
      -> UnqualifiedIdResult;
};

struct ASTInterpreter::NestedNameSpecifierVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
};

struct ASTInterpreter::TemplateArgumentVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentResult;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentResult;
};

auto ASTInterpreter::unqualifiedId(UnqualifiedIdAST* ast)
    -> UnqualifiedIdResult {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::nestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierResult {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::templateArgument(TemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdResult {
  auto idResult = interp.unqualifiedId(ast->id);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdResult {
  auto decltypeSpecifierResult = interp.specifier(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionIdAST* ast) -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    ConversionFunctionIdAST* ast) -> UnqualifiedIdResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdResult {
  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = interp.templateArgument(node);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto literalOperatorIdResult = interp.unqualifiedId(ast->literalOperatorId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = interp.templateArgument(node);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto operatorFunctionIdResult = interp.unqualifiedId(ast->operatorFunctionId);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = interp.templateArgument(node);
  }

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto decltypeSpecifierResult = interp.specifier(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto templateIdResult = interp.unqualifiedId(ast->templateId);

  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

}  // namespace cxx
