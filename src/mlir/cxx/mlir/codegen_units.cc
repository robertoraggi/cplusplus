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
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/IR/DataLayout.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Target/LLVMIR/Import.h>

#include <format>

namespace cxx {

namespace {
struct ForEachExternalDefinition final : ASTVisitor {
  std::function<void(FunctionDefinitionAST*)> functionCallback;

  void visit(TemplateDeclarationAST*) override {
    // Skip template declarations, we only want to visit function definitions.
  }

  void visit(FunctionDefinitionAST* ast) override {
    if (functionCallback) functionCallback(ast);

    ASTVisitor::visit(ast);
  }
};

}  // namespace

struct Codegen::UnitVisitor {
  Codegen& gen;

  auto operator()(TranslationUnitAST* ast) -> UnitResult;
  auto operator()(ModuleUnitAST* ast) -> UnitResult;

  struct VisitSymbols {
    UnitVisitor& p;

    void operator()(NamespaceSymbol* symbol) {
      for (auto member : views::members(symbol)) {
        if (auto var = symbol_cast<VariableSymbol>(member)) {
          p.gen.findOrCreateGlobal(var);
          continue;
        }

        visit(*this, member);
      }
    }

    void operator()(OverloadSetSymbol* symbol) {
      for (auto member : symbol->functions()) {
        visit(*this, member);
      }
    }

    void operator()(FunctionSymbol* symbol) {
      if (symbol->templateDeclaration()) {
        for (auto specialization : symbol->specializations()) {
          visit(*this, specialization.symbol);
        }
        return;
      }

      if (auto funcDecl = symbol->declaration()) {
        p.gen.declaration(funcDecl);
      }
    }

    void operator()(ClassSymbol* symbol) {
      if (symbol->templateDeclaration()) {
        for (auto specialization : symbol->specializations()) {
          visit(*this, specialization.symbol);
        }
        return;
      }

      for (auto specialization : symbol->specializations()) {
        visit(*this, specialization.symbol);
      }

      for (auto ctor : symbol->constructors()) {
        visit(*this, ctor);
      }

      for (auto member : views::members(symbol)) {
        visit(*this, member);
      }
    }

    void operator()(Symbol*) {
      // Do nothing for other symbols.
    }
  } visitor{*this};
};

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();

  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = mlir::ModuleOp::create(gen.builder_, loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  auto memoryLayout = gen.control()->memoryLayout();

  auto triple = llvm::Triple{gen.control()->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    cxx_runtime_error(std::format("failed to find target for triple '{}': {}",
                                  triple.getTriple(), error));
  }

  llvm::TargetOptions opt;

  auto RM = std::optional<llvm::Reloc::Model>();

  auto targetMachine =
      target->createTargetMachine(triple, "generic", "", opt, RM);

  auto llvmDataLayout = targetMachine->createDataLayout();

  auto dataLayout =
      mlir::translateDataLayout(llvmDataLayout, module->getContext());

  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dataLayout);

  module->setAttr("cxx.triple", mlir::StringAttr::get(module->getContext(),
                                                      memoryLayout->triple()));

  module->setAttr(
      "cxx.data-layout",
      mlir::StringAttr::get(module->getContext(),
                            llvmDataLayout.getStringRepresentation()));

  std::swap(gen.module_, module);

  visit(visitor, gen.unit_->globalScope());

#if false
  ForEachExternalDefinition forEachExternalDefinition;

  forEachExternalDefinition.functionCallback =
      [&](FunctionDefinitionAST* function) {
        auto functionResult = gen.declaration(function);
      };

  for (auto node : ListView{ast->declarationList}) {
    forEachExternalDefinition.accept(node);
  }
#endif

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = mlir::ModuleOp::create(gen.builder_, loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  auto globalModuleFragmentResult =
      gen.globalModuleFragment(ast->globalModuleFragment);

  auto moduleDeclarationResult = gen.moduleDeclaration(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  auto privateModuleFragmentResult =
      gen.privateModuleFragment(ast->privateModuleFragment);

  std::swap(gen.module_, module);

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
