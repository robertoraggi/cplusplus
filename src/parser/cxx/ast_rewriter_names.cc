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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

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

 private:
  enum class PackResult { kExpanded, kEmpty, kNotPack };

  [[nodiscard]] auto expandTypePackArgument(
      TypeTemplateArgumentAST* typeArg,
      List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult;

  [[nodiscard]] auto expandExprPackArgument(
      ExpressionTemplateArgumentAST* exprArg,
      List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult;

  void substituteTemplateTemplateParameter(SimpleTemplateIdAST* copy);
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
  auto copy = NameIdAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = DestructorIdAST::create(arena());

  copy->tildeLoc = ast->tildeLoc;
  copy->id = rewrite.unqualifiedId(ast->id);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = DecltypeIdAST::create(arena());

  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(OperatorFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = OperatorFunctionIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->opLoc = ast->opLoc;
  copy->openLoc = ast->openLoc;
  copy->closeLoc = ast->closeLoc;
  copy->op = ast->op;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = LiteralOperatorIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->literalLoc = ast->literalLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->literal = ast->literal;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(ConversionFunctionIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = ConversionFunctionIdAST::create(arena());

  copy->operatorLoc = ast->operatorLoc;
  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::expandTypePackArgument(
    TypeTemplateArgumentAST* typeArg,
    List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult {
  if (!typeArg || !typeArg->typeId || !typeArg->typeId->declarator)
    return PackResult::kNotPack;

  auto packDecl =
      ast_cast<ParameterPackAST>(typeArg->typeId->declarator->coreDeclarator);
  if (!packDecl) return PackResult::kNotPack;

  ParameterPackSymbol* pack = nullptr;
  for (auto spec : ListView{typeArg->typeId->typeSpecifierList}) {
    pack = rewrite.getTypeParameterPack(spec);
    if (pack) break;
  }

  if (!pack) return PackResult::kNotPack;
  if (pack->elements().empty()) return PackResult::kEmpty;

  auto savedParameterPack = rewrite.parameterPack_;
  std::swap(rewrite.parameterPack_, pack);

  int n = static_cast<int>(rewrite.parameterPack_->elements().size());
  for (int i = 0; i < n; ++i) {
    std::optional<int> index{i};
    std::swap(rewrite.elementIndex_, index);

    auto expandedTypeId = rewrite.typeId(typeArg->typeId);
    auto expandedArg = TypeTemplateArgumentAST::create(arena());
    expandedArg->typeId = expandedTypeId;

    *templateArgumentList =
        make_list_node(arena(), static_cast<TemplateArgumentAST*>(expandedArg));
    templateArgumentList = &(*templateArgumentList)->next;

    std::swap(rewrite.elementIndex_, index);
  }

  std::swap(rewrite.parameterPack_, pack);
  return PackResult::kExpanded;
}

auto ASTRewriter::UnqualifiedIdVisitor::expandExprPackArgument(
    ExpressionTemplateArgumentAST* exprArg,
    List<TemplateArgumentAST*>**& templateArgumentList) -> PackResult {
  if (!exprArg) return PackResult::kNotPack;

  auto packExpr = ast_cast<PackExpansionExpressionAST>(exprArg->expression);
  if (!packExpr) return PackResult::kNotPack;

  auto parameterPack = rewrite.getParameterPack(packExpr->expression);
  if (!parameterPack) return PackResult::kNotPack;
  if (parameterPack->elements().empty()) return PackResult::kEmpty;

  auto savedParameterPack = rewrite.parameterPack_;
  std::swap(rewrite.parameterPack_, parameterPack);

  int n = static_cast<int>(rewrite.parameterPack_->elements().size());
  for (int i = 0; i < n; ++i) {
    std::optional<int> index{i};
    std::swap(rewrite.elementIndex_, index);

    auto expandedExpr = rewrite.expression(packExpr->expression);
    auto expandedArg = ExpressionTemplateArgumentAST::create(arena());
    expandedArg->expression = expandedExpr;

    *templateArgumentList =
        make_list_node(arena(), static_cast<TemplateArgumentAST*>(expandedArg));
    templateArgumentList = &(*templateArgumentList)->next;

    std::swap(rewrite.elementIndex_, index);
  }

  std::swap(rewrite.parameterPack_, parameterPack);
  return PackResult::kExpanded;
}

void ASTRewriter::UnqualifiedIdVisitor::substituteTemplateTemplateParameter(
    SimpleTemplateIdAST* copy) {
  auto ttpSymbol = symbol_cast<TemplateTypeParameterSymbol>(copy->symbol);
  if (!ttpSymbol) return;

  auto paramType = type_cast<TemplateTypeParameterType>(ttpSymbol->type());
  const auto& args = rewrite.templateArguments_;
  if (!paramType || paramType->depth() != rewrite.depth_ ||
      paramType->index() >= static_cast<int>(args.size()))
    return;

  auto index = paramType->index();
  auto sym = std::get_if<Symbol*>(&args[index]);
  if (!sym) return;

  if (auto pack = symbol_cast<ParameterPackSymbol>(*sym)) {
    if (!rewrite.elementIndex_.has_value()) return;
    auto elemIdx = *rewrite.elementIndex_;
    if (elemIdx >= static_cast<int>(pack->elements().size())) return;
    auto elemSym = pack->elements()[elemIdx];
    if (auto alias = symbol_cast<TypeAliasSymbol>(elemSym)) {
      if (auto classType = type_cast<ClassType>(alias->type())) {
        copy->symbol = classType->symbol();
      }
    }
  } else if (auto alias = symbol_cast<TypeAliasSymbol>(*sym)) {
    if (auto classType = type_cast<ClassType>(alias->type())) {
      copy->symbol = classType->symbol();
    }
  }
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdAST* {
  auto copy = SimpleTemplateIdAST::create(arena());

  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto typeArg = ast_cast<TypeTemplateArgumentAST>(node);
    auto typeResult = expandTypePackArgument(typeArg, templateArgumentList);
    if (typeResult != PackResult::kNotPack) {
      if (typeResult == PackResult::kExpanded) continue;
      if (typeResult == PackResult::kEmpty) continue;
    }

    auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(node);
    auto exprResult = expandExprPackArgument(exprArg, templateArgumentList);
    if (exprResult != PackResult::kNotPack) {
      if (exprResult == PackResult::kExpanded) continue;
      if (exprResult == PackResult::kEmpty) continue;
    }

    auto value = rewrite.templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  substituteTemplateTemplateParameter(copy);

  return copy;
}

auto ASTRewriter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdAST* {
  auto copy = LiteralOperatorTemplateIdAST::create(arena());

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
  auto copy = OperatorFunctionTemplateIdAST::create(arena());

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
  auto copy = GlobalNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = SimpleNestedNameSpecifierAST::create(arena());

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
  auto copy = DecltypeNestedNameSpecifierAST::create(arena());

  copy->symbol = ast->symbol;
  copy->decltypeSpecifier =
      ast_cast<DecltypeSpecifierAST>(rewrite.specifier(ast->decltypeSpecifier));
  copy->scopeLoc = ast->scopeLoc;

  return copy;
}

auto ASTRewriter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierAST* {
  auto copy = TemplateNestedNameSpecifierAST::create(arena());

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
  auto copy = TypeTemplateArgumentAST::create(arena());

  copy->typeId = rewrite.typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentAST* {
  auto copy = ExpressionTemplateArgumentAST::create(arena());

  copy->expression = rewrite.expression(ast->expression);

  return copy;
}

}  // namespace cxx
