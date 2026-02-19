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
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

auto ASTRewriter::getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol* {
  struct FindPack final : ASTVisitor {
    ASTRewriter& rewriter;
    ParameterPackSymbol* result = nullptr;

    explicit FindPack(ASTRewriter& r) : rewriter(r) {}

    auto preVisit(AST*) -> bool override { return !result; }

    void visit(IdExpressionAST* ast) override {
      if (result) return;

      if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol)) {
        if (param->depth() != 0) return;
        auto arg = rewriter.templateArguments_[param->index()];
        auto argSymbol = std::get<Symbol*>(arg);
        if (auto pack = symbol_cast<ParameterPackSymbol>(argSymbol)) {
          result = pack;
          return;
        }
      }

      if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
        auto it = rewriter.functionParamPacks_.find(param);
        if (it != rewriter.functionParamPacks_.end()) {
          result = it->second;
          return;
        }
      }
    }

    void visit(NamedTypeSpecifierAST* ast) override {
      if (result) return;

      Symbol* paramSym = symbol_cast<TypeParameterSymbol>(ast->symbol);
      if (!paramSym)
        paramSym = symbol_cast<TemplateTypeParameterSymbol>(ast->symbol);
      if (!paramSym) return;

      auto paramInfo = getTypeParamInfo(paramSym->type());
      if (!paramInfo || !paramInfo->isPack) return;
      if (paramInfo->depth != rewriter.depth_) return;

      auto index = paramInfo->index;
      if (index >= static_cast<int>(rewriter.templateArguments_.size())) return;

      if (auto sym =
              std::get_if<Symbol*>(&rewriter.templateArguments_[index])) {
        if (auto pack = symbol_cast<ParameterPackSymbol>(*sym)) {
          result = pack;
        }
      }
    }
  };

  FindPack finder{*this};
  finder.accept(ast);
  return finder.result;
}

auto ASTRewriter::getTypeParameterPack(SpecifierAST* ast)
    -> ParameterPackSymbol* {
  if (auto named = ast_cast<NamedTypeSpecifierAST>(ast)) {
    Symbol* paramSym = symbol_cast<TypeParameterSymbol>(named->symbol);
    if (!paramSym)
      paramSym = symbol_cast<TemplateTypeParameterSymbol>(named->symbol);
    if (!paramSym) return nullptr;

    auto paramInfo = getTypeParamInfo(paramSym->type());
    if (!paramInfo || !paramInfo->isPack) return nullptr;
    if (paramInfo->depth != depth_) return nullptr;

    auto index = paramInfo->index;
    if (index >= static_cast<int>(templateArguments_.size())) return nullptr;

    if (auto sym = std::get_if<Symbol*>(&templateArguments_[index])) {
      return symbol_cast<ParameterPackSymbol>(*sym);
    }
  }
  return nullptr;
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
