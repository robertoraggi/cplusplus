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

struct ASTInterpreter::DeclarationVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(SimpleDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AsmDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(NamespaceAliasDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingEnumDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(UsingDirectiveAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(StaticAssertDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(AliasDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(OpaqueEnumDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(FunctionDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(TemplateDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ConceptDefinitionAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ExportDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(EmptyDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(AccessDeclarationAST* ast) -> DeclarationResult;

  [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast)
      -> DeclarationResult;

  [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
      -> DeclarationResult;
};

struct ASTInterpreter::TemplateParameterVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
      -> TemplateParameterResult;
};

struct ASTInterpreter::FunctionBodyVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
      -> FunctionBodyResult;

  [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast)
      -> FunctionBodyResult;
};

auto ASTInterpreter::declaration(DeclarationAST* ast) -> DeclarationResult {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::templateParameter(TemplateParameterAST* ast)
    -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::functionBody(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::usingDeclarator(UsingDeclaratorAST* ast)
    -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::asmOperand(AsmOperandAST* ast) -> DeclarationResult {
  auto expressionResult = expression(ast->expression);

  return {};
}

auto ASTInterpreter::asmQualifier(AsmQualifierAST* ast) -> DeclarationResult {
  return {};
}

auto ASTInterpreter::asmClobber(AsmClobberAST* ast) -> DeclarationResult {
  return {};
}

auto ASTInterpreter::asmGotoLabel(AsmGotoLabelAST* ast) -> DeclarationResult {
  return {};
}

auto ASTInterpreter::typeConstraint(TypeConstraintAST* ast)
    -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = templateArgument(node);
  }

  return {};
}

auto ASTInterpreter::lambdaSpecifier(LambdaSpecifierAST* ast)
    -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = interp.specifier(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = interp.initDeclarator(node);
  }

  auto requiresClauseResult = interp.requiresClause(ast->requiresClause);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    auto value = interp.asmQualifier(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    auto value = interp.asmOperand(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    auto value = interp.asmOperand(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    auto value = interp.asmClobber(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    auto value = interp.asmGotoLabel(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = interp.usingDeclarator(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    UsingEnumDeclarationAST* ast) -> DeclarationResult {
  auto enumTypeSpecifierResult = interp.specifier(ast->enumTypeSpecifier);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) -> DeclarationResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    OpaqueEnumDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);
  auto requiresClauseResult = interp.requiresClause(ast->requiresClause);
  auto functionBodyResult = interp.functionBody(ast->functionBody);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = interp.templateParameter(node);
  }

  auto requiresClauseResult = interp.requiresClause(ast->requiresClause);
  auto declarationResult = interp.declaration(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  auto explicitSpecifierResult = interp.specifier(ast->explicitSpecifier);
  auto parameterDeclarationClauseResult =
      interp.parameterDeclarationClause(ast->parameterDeclarationClause);
  auto templateIdResult = interp.unqualifiedId(ast->templateId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExplicitInstantiationAST* ast) -> DeclarationResult {
  auto declarationResult = interp.declaration(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = interp.declaration(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    LinkageSpecificationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = interp.nestedNamespaceSpecifier(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    AttributeDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationResult {
  auto importNameResult = interp.importName(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ParameterDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = interp.specifier(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = interp.unqualifiedId(node);
  }

  auto initializerResult = interp.expression(ast->initializer);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = interp.templateParameter(node);
  }

  auto requiresClauseResult = interp.requiresClause(ast->requiresClause);
  auto idExpressionResult = interp.expression(ast->idExpression);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = interp.declaration(ast->declaration);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = interp.typeConstraint(ast->typeConstraint);
  auto typeIdResult = interp.typeId(ast->typeId);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    DefaultFunctionBodyAST* ast) -> FunctionBodyResult {
  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = interp.memInitializer(node);
  }

  auto statementResult = interp.statement(ast->statement);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = interp.memInitializer(node);
  }

  auto statementResult = interp.statement(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = interp.handler(node);
  }

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

}  // namespace cxx
