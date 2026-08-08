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
#include <cxx/ast_rewriter.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {
struct FindReferencedParameterPack final : ASTVisitor {
  const ASTRewriter& rewriter;
  ParameterPackSymbol* exprPack = nullptr;
  ParameterPackSymbol* typePack = nullptr;

  explicit FindReferencedParameterPack(const ASTRewriter& r) : rewriter(r) {}

  auto preVisit(AST*) -> bool override { return !exprPack; }

  void visit(IdExpressionAST* ast) override {
    if (exprPack) return;

    if (auto pack = rewriter.functionParameterPackFor(ast->symbol)) {
      exprPack = pack;
      return;
    }

    if (auto pack = rewriter.parameterPackFor(ast->symbol)) {
      exprPack = pack;
      return;
    }

    if (ast->unqualifiedId) accept(ast->unqualifiedId);
    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    if (typePack) return;

    if (auto pack = rewriter.parameterPackFor(ast->symbol)) {
      typePack = pack;
      return;
    }

    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
    if (ast->unqualifiedId) accept(ast->unqualifiedId);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    if (typePack) return;

    if (auto pack = rewriter.parameterPackFor(ast->symbol)) {
      typePack = pack;
      return;
    }

    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }

  void visit(TemplateNestedNameSpecifierAST* ast) override {
    if (typePack) return;
    if (ast->templateId) accept(ast->templateId);
    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }
};

struct FindUnresolvedParameterPack final : ASTVisitor {
  const ASTRewriter& rewriter;
  bool found = false;

  explicit FindUnresolvedParameterPack(const ASTRewriter& r) : rewriter(r) {}

  auto preVisit(AST*) -> bool override { return !found; }

  void checkParameter(Symbol* symbol) {
    if (!symbol || found) return;
    if (rewriter.functionParameterPackFor(symbol)) return;
    auto info = template_parameter_info(symbol);
    if (!info || !info->isPack) return;
    if (!rewriter.templateArgumentAt(info->depth, info->index)) found = true;
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    checkParameter(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    checkParameter(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(IdExpressionAST* ast) override {
    checkParameter(ast->symbol);
    ASTVisitor::visit(ast);
  }
};

auto ASTRewriter::hasUnresolvedParameterPack(AST* ast) const -> bool {
  if (!ast) return false;
  FindUnresolvedParameterPack scan{*this};
  scan.accept(ast);
  return scan.found;
}

auto ASTRewriter::templateArgumentAt(int depth, int index) const
    -> const TemplateArgument* {
  auto arguments = &templateArguments_;

  if (depth != depth_) {
    auto it = enclosingTemplateArguments_.find(depth);
    if (it == enclosingTemplateArguments_.end()) return nullptr;
    arguments = &it->second;
  }

  if (index < 0 || index >= static_cast<int>(arguments->size())) return nullptr;
  return &(*arguments)[index];
}

void ASTRewriter::addEnclosingTemplateArguments(
    int depth, std::vector<TemplateArgument> arguments) {
  if (depth == depth_ || arguments.empty()) return;
  enclosingTemplateArguments_.insert_or_assign(depth, std::move(arguments));
}

void ASTRewriter::inheritEnclosingTemplateArguments(Symbol* symbol) {
  for (auto scope = symbol; scope; scope = scope->parent()) {
    auto classSymbol = symbol_cast<ClassSymbol>(scope);
    if (!classSymbol) continue;

    const auto depth = classSymbol->instantiationSubstitutionDepth();
    if (depth < 0) continue;

    addEnclosingTemplateArguments(
        depth, classSymbol->instantiationSubstitutionArguments());
  }
}

auto ASTRewriter::templateArgumentFor(Symbol* templateParameter) const
    -> const TemplateArgument* {
  auto info = template_parameter_info(templateParameter);
  if (!info) return nullptr;
  return templateArgumentAt(info->depth, info->index);
}

auto ASTRewriter::writtenArgumentForAliasedParameter(
    TypeIdAST* patternTypeId) const -> TypeIdAST* {
  if (!writtenTemplateArgumentList_) return nullptr;
  if (!patternTypeId) return nullptr;
  if (!patternTypeId->typeSpecifierList ||
      patternTypeId->typeSpecifierList->next) {
    return nullptr;
  }

  auto named =
      ast_cast<NamedTypeSpecifierAST>(patternTypeId->typeSpecifierList->value);
  if (!named || !symbol_cast<TypeParameterSymbol>(named->symbol)) {
    return nullptr;
  }

  auto info = template_parameter_info(named->symbol);
  if (!info || info->isPack || info->depth != depth_ || info->index < 0) {
    return nullptr;
  }

  int index = 0;
  for (auto argument : ListView{writtenTemplateArgumentList_}) {
    if (index++ != info->index) continue;
    auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
    if (!typeArgument || !typeArgument->typeId) return nullptr;
    return typeArgument->typeId->clone(arena());
  }

  return nullptr;
}

auto ASTRewriter::writtenTypeArgumentSpecifierFor(
    Symbol* templateParameter) const -> NamedTypeSpecifierAST* {
  if (!retainsEnclosingTemplateLevels()) return nullptr;

  auto info = template_parameter_info(templateParameter);
  if (!info || info->isPack || info->depth != depth_ || info->index < 0) {
    return nullptr;
  }

  int index = 0;
  for (auto argument : ListView{writtenTemplateArgumentList_}) {
    if (index++ != info->index) continue;

    auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
    if (!typeArgument || !typeArgument->typeId) return nullptr;

    auto typeId = typeArgument->typeId;
    if (typeId->declarator) return nullptr;
    if (!typeId->typeSpecifierList || typeId->typeSpecifierList->next)
      return nullptr;

    auto named =
        ast_cast<NamedTypeSpecifierAST>(typeId->typeSpecifierList->value);
    if (!named || !ast_cast<SimpleTemplateIdAST>(named->unqualifiedId))
      return nullptr;
    return named->clone(arena());
  }

  return nullptr;
}

auto ASTRewriter::substitutedSymbol(Symbol* templateParameter) const
    -> Symbol* {
  auto argument = templateArgumentFor(templateParameter);
  if (!argument) return nullptr;

  auto symbol = std::get_if<Symbol*>(argument);
  if (!symbol) return nullptr;

  if (auto pack = symbol_cast<ParameterPackSymbol>(*symbol))
    return packElementAt(pack);

  return *symbol;
}

auto ASTRewriter::parameterPackAt(int depth, int index, bool isPack) const
    -> ParameterPackSymbol* {
  if (!isPack) return nullptr;
  auto argument = templateArgumentAt(depth, index);
  if (!argument) return nullptr;
  auto sym = std::get_if<Symbol*>(argument);
  if (!sym) return nullptr;
  return symbol_cast<ParameterPackSymbol>(*sym);
}

auto ASTRewriter::parameterPackFor(Symbol* symbol) const
    -> ParameterPackSymbol* {
  auto info = template_parameter_info(symbol);
  if (!info) return nullptr;
  return parameterPackAt(info->depth, info->index, info->isPack);
}

auto ASTRewriter::functionParameterPackFor(Symbol* symbol) const
    -> ParameterPackSymbol* {
  auto param = symbol_cast<ParameterSymbol>(symbol);
  if (!param) return nullptr;
  auto it = functionParamPacks_.find(param);
  if (it == functionParamPacks_.end()) return nullptr;
  return it->second;
}

auto ASTRewriter::packElementAt(ParameterPackSymbol* pack) const -> Symbol* {
  if (!pack || !elementIndex_.has_value()) return nullptr;
  if (*elementIndex_ >= static_cast<int>(pack->elements().size()))
    return nullptr;
  return pack->elements()[*elementIndex_];
}

auto ASTRewriter::packElementCount(ParameterPackSymbol* pack) const -> int {
  return static_cast<int>(pack->elements().size());
}

auto ASTRewriter::substitutedTemplateParameterClass(Symbol* symbol) const
    -> Symbol* {
  auto typeParam = symbol_cast<TypeParameterSymbol>(symbol);
  if (!typeParam) return nullptr;

  auto resolved = substitutedSymbol(typeParam);
  if (!resolved) return nullptr;

  if (auto alias = symbol_cast<TypeAliasSymbol>(resolved)) {
    if (auto classType = type_cast<ClassType>(
            translationUnit()->typeTraits().remove_cv(alias->type()))) {
      resolved = classType->symbol();
    }
  }

  if (resolved->isClass()) return resolved;

  return nullptr;
}

auto ASTRewriter::findReferencedParameterPack(AST* ast) const
    -> ParameterPackSymbol* {
  FindReferencedParameterPack finder{*this};
  finder.accept(ast);
  return finder.exprPack ? finder.exprPack : finder.typePack;
}

auto ASTRewriter::emptyFoldIdentity(TokenKind op) -> ExpressionAST* {
  if (op == TokenKind::T_AMP_AMP) {
    return BoolLiteralExpressionAST::create(
        arena(), true, ValueCategory::kPrValue, control()->getBoolType());
  }

  if (op == TokenKind::T_BAR_BAR) {
    return BoolLiteralExpressionAST::create(
        arena(), false, ValueCategory::kPrValue, control()->getBoolType());
  }

  if (op == TokenKind::T_COMMA) return nullptr;

  return nullptr;
}
}  // namespace cxx
