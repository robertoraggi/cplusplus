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
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct Codegen::DeclarationVisitor {
  Codegen& gen;

  auto operator()(SimpleDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AsmDeclarationAST* ast) -> DeclarationResult;
  auto operator()(NamespaceAliasDefinitionAST* ast) -> DeclarationResult;
  auto operator()(UsingDeclarationAST* ast) -> DeclarationResult;
  auto operator()(UsingEnumDeclarationAST* ast) -> DeclarationResult;
  auto operator()(UsingDirectiveAST* ast) -> DeclarationResult;
  auto operator()(StaticAssertDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AliasDeclarationAST* ast) -> DeclarationResult;
  auto operator()(OpaqueEnumDeclarationAST* ast) -> DeclarationResult;
  auto operator()(FunctionDefinitionAST* ast) -> DeclarationResult;
  auto operator()(TemplateDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ConceptDefinitionAST* ast) -> DeclarationResult;
  auto operator()(DeductionGuideAST* ast) -> DeclarationResult;
  auto operator()(ExplicitInstantiationAST* ast) -> DeclarationResult;
  auto operator()(ExportDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ExportCompoundDeclarationAST* ast) -> DeclarationResult;
  auto operator()(LinkageSpecificationAST* ast) -> DeclarationResult;
  auto operator()(NamespaceDefinitionAST* ast) -> DeclarationResult;
  auto operator()(EmptyDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AttributeDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ModuleImportDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ParameterDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AccessDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ForRangeDeclarationAST* ast) -> DeclarationResult;
  auto operator()(StructuredBindingDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AsmOperandAST* ast) -> DeclarationResult;
  auto operator()(AsmQualifierAST* ast) -> DeclarationResult;
  auto operator()(AsmClobberAST* ast) -> DeclarationResult;
  auto operator()(AsmGotoLabelAST* ast) -> DeclarationResult;
};

struct Codegen::FunctionBodyVisitor {
  Codegen& gen;

  auto operator()(DefaultFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(TryStatementFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(DeleteFunctionBodyAST* ast) -> FunctionBodyResult;
};

struct Codegen::TemplateParameterVisitor {
  Codegen& gen;

  auto operator()(TemplateTypeParameterAST* ast) -> TemplateParameterResult;
  auto operator()(NonTypeTemplateParameterAST* ast) -> TemplateParameterResult;
  auto operator()(TypenameTypeParameterAST* ast) -> TemplateParameterResult;
  auto operator()(ConstraintTypeParameterAST* ast) -> TemplateParameterResult;
};

auto Codegen::operator()(DeclarationAST* ast) -> DeclarationResult {
  // if (ast) return visit(DeclarationVisitor{*this}, ast);

  // restrict for now to declarations that are not definitions
  if (ast_cast<FunctionDefinitionAST>(ast) ||
      ast_cast<LinkageSpecificationAST>(ast) ||
      ast_cast<SimpleDeclarationAST>(ast) ||
      ast_cast<NamespaceDefinitionAST>(ast)) {
    return visit(DeclarationVisitor{*this}, ast);
  }

  return {};
}

auto Codegen::operator()(TemplateParameterAST* ast) -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto Codegen::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::operator()(TypeConstraintAST* ast) -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto Codegen::operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceAliasDefinitionAST* ast)
    -> DeclarationResult {
  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationResult {
  auto enumTypeSpecifierResult = gen(ast->enumTypeSpecifier);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(StaticAssertDeclarationAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = gen(node);
  }

  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  auto nestedNameSpecifierResult = gen(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = gen(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  auto functionSymbol = ast->symbol;
  auto functionType = type_cast<FunctionType>(functionSymbol->type());

  auto exprType = gen.builder_.getType<mlir::cxx::ExprType>();

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  for (auto paramTy : functionType->parameterTypes()) {
    inputTypes.push_back(gen.convertType(paramTy));
  }

  if (!gen.control()->is_void(functionType->returnType())) {
    resultTypes.push_back(exprType);
  }

  auto funcType = gen.builder_.getFunctionType(inputTypes, resultTypes);
  auto loc = gen.builder_.getUnknownLoc();

  std::vector<std::string> path;
  for (Symbol* symbol = ast->symbol; symbol;
       symbol = symbol->enclosingSymbol()) {
    if (!symbol->name()) continue;
    path.push_back(to_string(symbol->name()));
  }

  std::string name;

  if (ast->symbol->hasCLinkage()) {
    name = to_string(ast->symbol->name());
  } else {
    // todo: external name mangling

    std::ranges::for_each(path | std::views::reverse, [&](auto& part) {
      name += "::";
      name += part;
    });

    // generate unique names until we have proper name mangling
    name += std::format("_{}", ++gen.count_);
  }

  auto savedInsertionPoint = gen.builder_.saveInsertionPoint();

  auto func = gen.builder_.create<mlir::cxx::FuncOp>(loc, name, funcType);

  auto entryBlock = &func.front();

  gen.builder_.setInsertionPointToEnd(entryBlock);

  std::swap(gen.function_, func);

#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto requiresClauseResult = gen(ast->requiresClause);
#endif

  auto functionBodyResult = gen(ast->functionBody);

  std::swap(gen.function_, func);

  auto endLoc = gen.getLocation(ast->lastSourceLocation());

  if (gen.control()->is_void(functionType->returnType())) {
    // If the function returns void, we don't need to return anything.
    gen.builder_.create<mlir::cxx::ReturnOp>(endLoc);
  } else {
    // Otherwise, we need to return a value of the correct type.
    auto r = gen.emitTodoExpr(ast->lastSourceLocation(), "result value");
    auto result =
        gen.builder_.create<mlir::cxx::ReturnOp>(endLoc, r->getResults());
  }

  gen.builder_.restoreInsertionPoint(savedInsertionPoint);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  auto explicitSpecifierResult = gen(ast->explicitSpecifier);
  auto parameterDeclarationClauseResult = gen(ast->parameterDeclarationClause);
  auto templateIdResult = gen(ast->templateId);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportCompoundDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ModuleImportDeclarationAST* ast)
    -> DeclarationResult {
  auto importNameResult = gen(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen(node);
  }

  auto declaratorResult = gen(ast->declarator);
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = gen(node);
  }

  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationResult {
  auto expressionResult = gen.expression(ast->expression);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmQualifierAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
#if false
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen(node);
  }
#endif

  gen.statement(ast->statement);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(TryStatementFunctionBodyAST* ast)
    -> FunctionBodyResult {
#if false
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen(node);
  }

#endif

  gen.statement(ast->statement);

#if false
  for (auto node : ListView{ast->handlerList}) {
    auto value = gen(node);
  }
#endif

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen(node);
  }

  auto requiresClauseResult = gen(ast->requiresClause);
  auto idExpressionResult = gen.expression(ast->idExpression);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = gen(ast->declaration);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = gen(ast->typeId);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = gen(ast->typeConstraint);
  auto typeIdResult = gen(ast->typeId);

  return {};
}

}  // namespace cxx