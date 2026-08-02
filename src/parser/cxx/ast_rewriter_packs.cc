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
  ASTRewriter& rewriter;
  ParameterPackSymbol* exprPack = nullptr;
  ParameterPackSymbol* typePack = nullptr;

  explicit FindReferencedParameterPack(ASTRewriter& r) : rewriter(r) {}

  auto preVisit(AST*) -> bool override { return !exprPack; }

  auto packAt(int depth, int index, bool isPack) -> ParameterPackSymbol* {
    return rewriter.parameterPackAt(depth, index, isPack);
  }

  void visit(IdExpressionAST* ast) override {
    if (exprPack) return;

    if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol)) {
      if (param->depth() == 0) {
        auto arg = rewriter.templateArguments_[param->index()];
        if (auto pack =
                symbol_cast<ParameterPackSymbol>(std::get<Symbol*>(arg))) {
          exprPack = pack;
          return;
        }
      }
    }

    if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
      auto it = rewriter.functionParamPacks_.find(param);
      if (it != rewriter.functionParamPacks_.end()) {
        exprPack = it->second;
        return;
      }
    }

    if (ast->unqualifiedId) accept(ast->unqualifiedId);
    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    if (typePack) return;

    Symbol* paramSym = symbol_cast<TypeParameterSymbol>(ast->symbol);
    if (!paramSym)
      paramSym = symbol_cast<TemplateTypeParameterSymbol>(ast->symbol);
    if (paramSym) {
      if (auto paramInfo = getTypeParamInfo(paramSym->type())) {
        if (auto pack =
                packAt(paramInfo->depth, paramInfo->index, paramInfo->isPack)) {
          typePack = pack;
          return;
        }
      }
    }

    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
    if (ast->unqualifiedId) accept(ast->unqualifiedId);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    if (typePack) return;

    Symbol* paramSym = symbol_cast<TypeParameterSymbol>(ast->symbol);
    if (!paramSym)
      paramSym = symbol_cast<TemplateTypeParameterSymbol>(ast->symbol);
    if (paramSym) {
      if (auto paramInfo = getTypeParamInfo(paramSym->type())) {
        if (auto pack =
                packAt(paramInfo->depth, paramInfo->index, paramInfo->isPack)) {
          typePack = pack;
          return;
        }
      }
    }

    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }

  void visit(TemplateNestedNameSpecifierAST* ast) override {
    if (typePack) return;
    if (ast->templateId) accept(ast->templateId);
    if (ast->nestedNameSpecifier) accept(ast->nestedNameSpecifier);
  }
};

auto ASTRewriter::parameterPackAt(int depth, int index, bool isPack)
    -> ParameterPackSymbol* {
  if (!isPack) return nullptr;
  if (depth != depth_) return nullptr;
  if (index < 0 || index >= static_cast<int>(templateArguments_.size()))
    return nullptr;
  auto sym = std::get_if<Symbol*>(&templateArguments_[index]);
  if (!sym) return nullptr;
  return symbol_cast<ParameterPackSymbol>(*sym);
}

auto ASTRewriter::parameterPackFor(Symbol* symbol) -> ParameterPackSymbol* {
  auto info = template_parameter_info(symbol);
  if (!info) return nullptr;
  return parameterPackAt(info->depth, info->index, info->isPack);
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

auto ASTRewriter::substitutedTemplateParameterClass(Symbol* symbol) -> Symbol* {
  auto typeParam = symbol_cast<TypeParameterSymbol>(symbol);
  if (!typeParam) return nullptr;

  auto paramType = type_cast<TypeParameterType>(typeParam->type());
  if (!paramType) return nullptr;
  if (paramType->depth() != depth_) return nullptr;
  if (paramType->index() >= static_cast<int>(templateArguments_.size()))
    return nullptr;

  auto argument = std::get_if<Symbol*>(&templateArguments_[paramType->index()]);
  if (!argument) return nullptr;

  Symbol* resolved = *argument;

  if (auto pack = symbol_cast<ParameterPackSymbol>(resolved)) {
    if (!elementIndex_.has_value()) return nullptr;
    if (*elementIndex_ >= static_cast<int>(pack->elements().size()))
      return nullptr;
    resolved = pack->elements()[*elementIndex_];
  }

  if (auto alias = symbol_cast<TypeAliasSymbol>(resolved)) {
    if (auto classType = type_cast<ClassType>(
            translationUnit()->typeTraits().remove_cv(alias->type()))) {
      resolved = classType->symbol();
    }
  }

  if (resolved && resolved->isClass()) return resolved;

  return nullptr;
}

auto ASTRewriter::findReferencedParameterPack(AST* ast)
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
