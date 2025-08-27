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

struct ASTInterpreter::UnitVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitResult;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitResult;
};

auto ASTInterpreter::unit(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::globalModuleFragment(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto ASTInterpreter::privateModuleFragment(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto ASTInterpreter::moduleDeclaration(ModuleDeclarationAST* ast)
    -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);
  auto modulePartitionResult = modulePartition(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  return {};
}

auto ASTInterpreter::moduleName(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::moduleQualifier(ModuleQualifierAST* ast)
    -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::modulePartition(ModulePartitionAST* ast)
    -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}

auto ASTInterpreter::importName(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = modulePartition(ast->modulePartition);
  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(TranslationUnitAST* ast)
    -> UnitResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto globalModuleFragmentResult =
      interp.globalModuleFragment(ast->globalModuleFragment);
  auto moduleDeclarationResult =
      interp.moduleDeclaration(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = interp.declaration(node);
  }

  auto privateModuleFragmentResult =
      interp.privateModuleFragment(ast->privateModuleFragment);

  return {};
}

}  // namespace cxx
