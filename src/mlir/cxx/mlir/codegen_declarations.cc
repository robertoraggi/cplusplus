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
#include <cxx/external_name_encoder.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// mlir
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Block.h>

#include <format>

namespace cxx {

namespace {

[[nodiscard]] auto is_global_namespace(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (!symbol->isNamespace()) return false;
  if (symbol->parent()) return false;
  return true;
}

}  // namespace

struct Codegen::DeclarationVisitor {
  Codegen& gen;

  void allocateLocals(ScopeSymbol* block);

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

auto Codegen::declaration(DeclarationAST* ast) -> DeclarationResult {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto Codegen::templateParameter(TemplateParameterAST* ast)
    -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto Codegen::functionBody(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto Codegen::nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::typeConstraint(TypeConstraintAST* ast) -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = templateArgument(node);
  }

  return {};
}

auto Codegen::usingDeclarator(UsingDeclaratorAST* ast)
    -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::lambdaSpecifier(LambdaSpecifierAST* ast)
    -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

void Codegen::DeclarationVisitor::allocateLocals(ScopeSymbol* block) {
  for (auto symbol : views::members(block)) {
    if (auto nestedBlock = symbol_cast<BlockSymbol>(symbol)) {
      allocateLocals(nestedBlock);
      continue;
    }

    if (auto var = symbol_cast<VariableSymbol>(symbol)) {
      if (var->isStatic()) continue;

      auto local = gen.findOrCreateLocal(var);
      if (!local.has_value()) {
        gen.unit_->error(var->location(),
                         std::format("cannot allocate local variable '{}'",
                                     to_string(var->name())));
      }
    }
  }
}

auto Codegen::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
  if (!gen.function_) {
    // skip for now, as we only look for local variable declarations
    return {};
  }

#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen.specifier(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = gen.initDeclarator(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);
#endif

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    if (!node->initializer) continue;

    const auto loc = gen.getLocation(var->location());

    auto local = gen.findOrCreateLocal(var);

    if (!local.has_value()) {
      gen.unit_->error(node->initializer->firstSourceLocation(),
                       std::format("cannot find local variable '{}'",
                                   to_string(var->name())));
      continue;
    }

    auto expressionResult = gen.expression(node->initializer);

    const auto elementType = gen.convertType(var->type());

    mlir::cxx::StoreOp::create(gen.builder_, loc, expressionResult.value,
                               local.value());
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    gen.asmQualifier(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    gen.asmOperand(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    gen.asmOperand(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    gen.asmClobber(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    gen.asmGotoLabel(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceAliasDefinitionAST* ast)
    -> DeclarationResult {
#if false
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = gen.usingDeclarator(node);
  }
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationResult {
#if false
  auto enumTypeSpecifierResult = gen.specifier(ast->enumTypeSpecifier);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(StaticAssertDeclarationAST* ast)
    -> DeclarationResult {
#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  auto functionSymbol = ast->symbol;

  auto func = gen.findOrCreateFunction(functionSymbol);
  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  const auto returnType = functionType->returnType();
  const auto needsExitValue = !gen.control()->is_void(returnType);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  // Add the function body.
  auto entryBlock = gen.builder_.createBlock(&func.getBody());
  auto inputs = func.getFunctionType().getInputs();

  for (const auto& input : inputs) {
    entryBlock->addArgument(input, loc);
  }

  auto exitBlock = gen.builder_.createBlock(&func.getBody());
  mlir::cxx::AllocaOp exitValue;

  // set the insertion point to the entry block
  gen.builder_.setInsertionPointToEnd(entryBlock);

  if (needsExitValue) {
    auto exitValueLoc =
        gen.getLocation(ast->functionBody->firstSourceLocation());
    auto exitValueType = gen.convertType(returnType);
    auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(exitValueType);
    exitValue =
        mlir::cxx::AllocaOp::create(gen.builder_, exitValueLoc, ptrType);

    auto id = name_cast<Identifier>(functionSymbol->name());
    if (id && id->name() == "main" &&
        is_global_namespace(functionSymbol->parent())) {
      auto zeroOp = mlir::cxx::IntConstantOp::create(
          gen.builder_, loc, gen.convertType(gen.control()->getIntType()), 0);

      mlir::cxx::StoreOp::create(gen.builder_, exitValueLoc, zeroOp, exitValue);
    }
  }

  std::unordered_map<Symbol*, mlir::Value> locals;

  // function state
  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);

  mlir::Value thisValue;

  // if this is a non static member function, we need to allocate the `this`
  if (!functionSymbol->isStatic() && functionSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

    auto thisType = gen.convertType(classSymbol->type());
    auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(thisType);

    thisValue = gen.newTemp(classSymbol->type(), ast->firstSourceLocation());

    // store the `this` pointer in the entry block
    mlir::cxx::StoreOp::create(gen.builder_, loc,
                               gen.entryBlock_->getArgument(0), thisValue);
  }

  FunctionParametersSymbol* params = nullptr;
  for (auto member : views::members(ast->symbol)) {
    params = symbol_cast<FunctionParametersSymbol>(member);
    if (!params) continue;

    int argc = 0;
    auto args = gen.entryBlock_->getArguments();
    for (auto param : views::members(params)) {
      auto arg = symbol_cast<ParameterSymbol>(param);
      if (!arg) continue;

      auto type = gen.convertType(arg->type());
      auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(type);

      auto loc = gen.getLocation(arg->location());
      auto allocaOp = mlir::cxx::AllocaOp::create(gen.builder_, loc, ptrType);

      auto value = args[argc];
      ++argc;
      mlir::cxx::StoreOp::create(gen.builder_, loc, value, allocaOp);

      gen.locals_.emplace(arg, allocaOp);
    }
  }

  std::swap(gen.thisValue_, thisValue);

  allocateLocals(functionSymbol);

  // generate code for the function body
  auto functionBodyResult = gen.functionBody(ast->functionBody);

  // terminate the function body

  const auto endLoc = gen.getLocation(ast->lastSourceLocation());

  if (!gen.builder_.getBlock()->mightHaveTerminator()) {
    mlir::cf::BranchOp::create(gen.builder_, endLoc, gen.exitBlock_);
  }

  gen.builder_.setInsertionPointToEnd(gen.exitBlock_);

  if (gen.exitValue_) {
    // We need to return a value of the correct type.
    auto elementType = gen.exitValue_.getType().getElementType();

    auto value = mlir::cxx::LoadOp::create(gen.builder_, endLoc, elementType,
                                           gen.exitValue_);

    mlir::cxx::ReturnOp::create(gen.builder_, endLoc, value->getResults());
  } else {
    // If the function returns void, we don't need to return anything.
    mlir::cxx::ReturnOp::create(gen.builder_, endLoc);
  }

  // restore the state
  std::swap(gen.thisValue_, thisValue);

  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen.templateParameter(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);

  auto declarationResult = gen.declaration(ast->declaration);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
#if false
  auto explicitSpecifierResult = gen.specifier(ast->explicitSpecifier);

  auto parameterDeclarationClauseResult =
      gen.parameterDeclarationClause(ast->parameterDeclarationClause);

  auto templateIdResult = gen.unqualifiedId(ast->templateId);
#endif
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportCompoundDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = gen.nestedNamespaceSpecifier(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
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
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ModuleImportDeclarationAST* ast)
    -> DeclarationResult {
  auto importNameResult = gen.importName(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }

  auto declaratorResult = gen.declarator(ast->declarator);
  auto expressionResult = gen.expression(ast->expression);
#endif
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
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen.specifier(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = gen.unqualifiedId(node);
  }

  auto initializerResult = gen.expression(ast->initializer);

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
    auto value = gen.templateParameter(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);

  auto idExpressionResult = gen.expression(ast->idExpression);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = gen.typeConstraint(ast->typeConstraint);
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

void Codegen::asmOperand(AsmOperandAST* ast) {
  auto expressionResult = expression(ast->expression);
}

void Codegen::asmQualifier(AsmQualifierAST* ast) {}

void Codegen::asmClobber(AsmClobberAST* ast) {}

void Codegen::asmGotoLabel(AsmGotoLabelAST* ast) {}
}  // namespace cxx