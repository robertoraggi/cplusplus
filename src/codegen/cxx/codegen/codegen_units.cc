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
#include <cxx/ast_visitor.h>
#include <cxx/attributes.h>
#include <cxx/binder.h>
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbols.h>

namespace cxx {
namespace {
struct ForEachExternalDefinition final : ASTVisitor {
  std::function<void(FunctionDefinitionAST*)> functionCallback;
  std::function<void(AsmDeclarationAST*)> asmCallback;
  std::size_t functionDepth = 0;

  void visit(TemplateDeclarationAST* ast) override {
    if (ast->templateParameterList) return;
    ASTVisitor::visit(ast);
  }

  void visit(AsmDeclarationAST* ast) override {
    if (!functionDepth && asmCallback) asmCallback(ast);
  }

  void visit(FunctionDefinitionAST* ast) override {
    if (ast->symbol && ast->symbol->isTemplatePattern()) return;
    if (functionCallback) functionCallback(ast);
    ++functionDepth;
    ASTVisitor::visit(ast);
    --functionDepth;
  }
};
}  // namespace

struct Codegen::UnitVisitor {
  Codegen& gen;

  auto operator()(TranslationUnitAST* ast) -> UnitResult;
  auto operator()(ModuleUnitAST* ast) -> UnitResult;

  void visitStaticField(FieldSymbol* field) {
    if (auto definition = field->definition()) {
      if (definition->instantiationPattern()) return;
      if (auto global = gen.findOrCreateGlobal(definition)) {
        gen.emitGlobalVarInit(definition, *global);
        if (gen.isNonLocalThreadLocal(definition))
          (void)gen.findOrCreateThreadLocalWrapper(definition, *global);
      }
      return;
    }
    if (!field->isInline() || field->instantiationPattern()) return;
    if (field->constValue()) return;
    auto global = gen.findOrCreateStaticField(field);
    if (gen.isNonLocalThreadLocal(field))
      (void)gen.findOrCreateThreadLocalWrapper(field, global);
  }

  void visitClassStatics(ClassSymbol* classSymbol) {
    if (classSymbol->isTemplatePattern()) {
      for (const auto& specialization : classSymbol->specializations()) {
        auto instance = symbol_cast<ClassSymbol>(specialization.symbol);
        visitClassStatics(instance->resolvedDefinition());
      }
      return;
    }
    visitFunctionAliases(classSymbol);
    for (auto member : classSymbol->members()) {
      if (auto field = symbol_cast<FieldSymbol>(member)) {
        visitStaticField(field);
        continue;
      }
      if (auto nestedClass = symbol_cast<ClassSymbol>(member))
        visitClassStatics(nestedClass);
    }
  }

  void visitVariable(VariableSymbol* var) {
    if (var->isTemplatePattern()) {
      for (const auto& specialization : var->specializations())
        visitVariable(symbol_cast<VariableSymbol>(specialization.symbol));
      return;
    }
    if (var->instantiationPattern()) return;
    if (var->isInline() && var->resolvedDefinition()->constValue()) return;
    if (auto glo = gen.findOrCreateGlobal(var)) {
      gen.emitGlobalVarInit(var, *glo);
      if (gen.isNonLocalThreadLocal(var) &&
          Binder::declaresVariableDefinition(var->resolvedDefinition()))
        (void)gen.findOrCreateThreadLocalWrapper(var, *glo);
    }
  }

  void visitGlobals(ScopeSymbol* scope) {
    auto ns = symbol_cast<NamespaceSymbol>(scope);
    if (!ns) return;

    visitFunctionAliases(ns);

    for (auto member : views::members(ns)) {
      if (auto var = symbol_cast<VariableSymbol>(member)) {
        visitVariable(var);
        continue;
      }

      if (auto nestedNs = symbol_cast<NamespaceSymbol>(member)) {
        visitGlobals(nestedNs);
        continue;
      }

      if (auto classSymbol = symbol_cast<ClassSymbol>(member))
        visitClassStatics(classSymbol);
    }
  }

  void visitFunctionAliases(ScopeSymbol* scope) {
    for (auto member : views::members(scope)) {
      for (auto function : views::declared_functions{member}) {
        if (attributeArgument(function->canonical()->attributes(), "alias"))
          (void)gen.findOrCreateFunction(function);
      }
    }
  }
};

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (!ast) return {};
  return visit(UnitVisitor{*this}, ast);
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  auto module = gen.emitter_.beginModule({
      .name = gen.unit_->fileName(),
      .sourceFile = gen.unit_->fileName(),
      .targetTriple = gen.control()->memoryLayout()->triple(),
      .debugCompilationDirectory = gen.debugCompilationDirectory(),
      .framePointer =
          to_string(gen.control()->memoryLayout()->framePointerKind()),
      .picLevel =
          gen.control()->memoryLayout()->positionIndependence().picLevel,
      .pieLevel =
          gen.control()->memoryLayout()->positionIndependence().pieLevel,
  });

  visitGlobals(gen.unit_->globalScope());

  ForEachExternalDefinition forEachExternalDefinition;

  forEachExternalDefinition.asmCallback = [&](AsmDeclarationAST* assembly) {
    (void)gen.declaration(assembly);
  };

  forEachExternalDefinition.functionCallback =
      [&](FunctionDefinitionAST* function) {
        if (!function->symbol->canonical()->isUsed() &&
            gen.hasVagueFunctionEmission(function->symbol))
          return;
        (void)gen.declaration(function);
      };

  for (auto node : ListView{ast->declarationList}) {
    forEachExternalDefinition.accept(node);
  }

  for (auto function : gen.unit_->explicitInstantiationDefinitions()) {
    (void)gen.findOrCreateFunction(function);
  }

  gen.processPendingFunctions();

  gen.emitThreadLocalInitializers();
  gen.emitter_.endModule();

  UnitResult result{module};
  return result;
}

auto Codegen::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto module = gen.emitter_.beginModule({.name = gen.unit_->fileName()});

  auto globalModuleFragmentResult =
      gen.globalModuleFragment(ast->globalModuleFragment);

  auto moduleDeclarationResult = gen.moduleDeclaration(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  auto privateModuleFragmentResult =
      gen.privateModuleFragment(ast->privateModuleFragment);

  gen.emitThreadLocalInitializers();
  gen.emitter_.endModule();

  UnitResult result{module};
  return result;
}

auto Codegen::globalModuleFragment(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto Codegen::privateModuleFragment(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto Codegen::moduleDeclaration(ModuleDeclarationAST* ast)
    -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);

  auto modulePartitionResult = modulePartition(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  return {};
}

auto Codegen::moduleName(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto Codegen::moduleQualifier(ModuleQualifierAST* ast)
    -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto Codegen::modulePartition(ModulePartitionAST* ast)
    -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}

auto Codegen::importName(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = modulePartition(ast->modulePartition);

  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}
}  // namespace cxx
