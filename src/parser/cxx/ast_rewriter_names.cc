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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <format>
#include <iostream>

namespace cxx {

struct ASTRewriter::UnqualifiedIdVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(NameIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(DestructorIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(DecltypeIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(OperatorFunctionIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(LiteralOperatorIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(ConversionFunctionIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(SimpleTemplateIdAST* ast) -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(LiteralOperatorTemplateIdAST* ast)
      -> UnqualifiedIdAST*;

  [[nodiscard]] auto operator()(OperatorFunctionTemplateIdAST* ast)
      -> UnqualifiedIdAST*;
};

struct ASTRewriter::NestedNameSpecifierVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(GlobalNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(SimpleNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(DecltypeNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;

  [[nodiscard]] auto operator()(TemplateNestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;
};

struct ASTRewriter::TemplateArgumentVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(TypeTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto operator()(ExpressionTemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
};

auto ASTRewriter::unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdAST* {
  if (!ast) return {};
  return visit(UnqualifiedIdVisitor{*this}, ast);
}

auto ASTRewriter::nestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierAST* {
  if (!ast) return {};
  return visit(NestedNameSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::templateArgument(TemplateArgumentAST* ast)
    -> TemplateArgumentAST* {
  if (!ast) return {};
  return visit(TemplateArgumentVisitor{*this}, ast);
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<NameIdAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<DestructorIdAST>(arena());

  copy->tildeLoc = ast->tildeLoc;
  copy->id = rewrite.unqualifiedId(ast->id);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<DecltypeIdAST>(arena());

  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<OperatorFunctionIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->opLoc = ast->opLoc;
  copy->openLoc = ast->openLoc;
  copy->closeLoc = ast->closeLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<LiteralOperatorIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->literalLoc = ast->literalLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->literal = ast->literal;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<ConversionFunctionIdAST>(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = make_node<SimpleTemplateIdAST>(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = make_node<LiteralOperatorTemplateIdAST>(arena());

  copy->literalOperatorId = ast_cast<LiteralOperatorIdAST>(
      rewrite.unqualifiedId(ast->literalOperatorId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = make_node<OperatorFunctionTemplateIdAST>(arena());

  copy->operatorFunctionId = ast_cast<OperatorFunctionIdAST>(
      rewrite.unqualifiedId(ast->operatorFunctionId));
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<GlobalNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<SimpleNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<DecltypeNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = make_node<TemplateNestedNameSpecifierAST>(arena());

  copy->symbol = ast->symbol;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->templateId =
      ast_cast<SimpleTemplateIdAST>(rewrite.unqualifiedId(ast->templateId));
  copy->scopeLoc = ast->scopeLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  auto classSymbol = symbol_cast<ClassSymbol>(copy->symbol);

  auto instance = ASTRewriter::instantiate(
      translationUnit(), copy->templateId->templateArgumentList, classSymbol);

  copy->symbol = symbol_cast<ClassSymbol>(instance);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = make_node<TypeTemplateArgumentAST>(arena());

  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = make_node<ExpressionTemplateArgumentAST>(arena());

  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

}  // namespace cxx
