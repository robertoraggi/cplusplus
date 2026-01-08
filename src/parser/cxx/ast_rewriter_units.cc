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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>

namespace cxx {

struct ASTRewriter::UnitVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(TranslationUnitAST* ast) -> UnitAST*;

  [[nodiscard]] auto operator()(ModuleUnitAST* ast) -> UnitAST*;
};

auto ASTRewriter::unit(UnitAST* ast) -> UnitAST* {
  if (!ast) return {};
  return visit(UnitVisitor{*this}, ast);
}

auto ASTRewriter::globalModuleFragment(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentAST* {
  if (!ast) return {};

  auto copy = GlobalModuleFragmentAST::create(arena());

  copy->moduleLoc = ast->moduleLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::privateModuleFragment(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentAST* {
  if (!ast) return {};

  auto copy = PrivateModuleFragmentAST::create(arena());

  copy->moduleLoc = ast->moduleLoc;
  copy->colonLoc = ast->colonLoc;
  copy->privateLoc = ast->privateLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::moduleDeclaration(ModuleDeclarationAST* ast)
    -> ModuleDeclarationAST* {
  if (!ast) return {};

  auto copy = ModuleDeclarationAST::create(arena());

  copy->exportLoc = ast->exportLoc;
  copy->moduleLoc = ast->moduleLoc;
  copy->moduleName = moduleName(ast->moduleName);
  copy->modulePartition = modulePartition(ast->modulePartition);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::moduleName(ModuleNameAST* ast) -> ModuleNameAST* {
  if (!ast) return {};

  auto copy = ModuleNameAST::create(arena());

  copy->moduleQualifier = moduleQualifier(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::moduleQualifier(ModuleQualifierAST* ast)
    -> ModuleQualifierAST* {
  if (!ast) return {};

  auto copy = ModuleQualifierAST::create(arena());

  copy->moduleQualifier = moduleQualifier(ast->moduleQualifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->dotLoc = ast->dotLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::modulePartition(ModulePartitionAST* ast)
    -> ModulePartitionAST* {
  if (!ast) return {};

  auto copy = ModulePartitionAST::create(arena());

  copy->colonLoc = ast->colonLoc;
  copy->moduleName = moduleName(ast->moduleName);

  return copy;
}

auto ASTRewriter::importName(ImportNameAST* ast) -> ImportNameAST* {
  if (!ast) return {};

  auto copy = ImportNameAST::create(arena());

  copy->headerLoc = ast->headerLoc;
  copy->modulePartition = modulePartition(ast->modulePartition);
  copy->moduleName = moduleName(ast->moduleName);

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitAST* {
  auto copy = TranslationUnitAST::create(arena());

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  return copy;
}

auto ASTRewriter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitAST* {
  auto copy = ModuleUnitAST::create(arena());

  copy->globalModuleFragment =
      rewrite.globalModuleFragment(ast->globalModuleFragment);
  copy->moduleDeclaration = rewrite.moduleDeclaration(ast->moduleDeclaration);

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->privateModuleFragment =
      rewrite.privateModuleFragment(ast->privateModuleFragment);

  return copy;
}

}  // namespace cxx