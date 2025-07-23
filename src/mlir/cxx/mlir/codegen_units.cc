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
#include <cxx/translation_unit.h>

namespace cxx {

struct Codegen::UnitVisitor {
  Codegen& gen;

  auto operator()(TranslationUnitAST* ast) -> UnitResult;
  auto operator()(ModuleUnitAST* ast) -> UnitResult;
};

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = gen.builder_.create<mlir::ModuleOp>(loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = gen.builder_.create<mlir::ModuleOp>(loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  auto globalModuleFragmentResult = gen(ast->globalModuleFragment);
  auto moduleDeclarationResult = gen(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen(node);
  }

  auto privateModuleFragmentResult = gen(ast->privateModuleFragment);

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::operator()(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(ModuleDeclarationAST* ast) -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);
  auto modulePartitionResult = operator()(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = operator()(node);
  }

  return {};
}

auto Codegen::operator()(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto Codegen::operator()(ModuleQualifierAST* ast) -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto Codegen::operator()(ModulePartitionAST* ast) -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto Codegen::operator()(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = operator()(ast->modulePartition);
  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

}  // namespace cxx