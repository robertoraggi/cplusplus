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
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

// todo remove
#include <cxx/ast_cursor.h>

namespace cxx {

auto ASTRewriter::make_substitution(
    TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> std::vector<TemplateArgument> {
  auto control = unit->control();
  auto interp = ASTInterpreter{unit};

  std::vector<TemplateArgument> templateArguments;

  for (auto arg : ListView{templateArgumentList}) {
    if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
      auto expr = exprArg->expression;
      // ### need to set scope and location
      auto templArg = control->newVariableSymbol(nullptr, {});
      templArg->setInitializer(expr);
      templArg->setType(control->add_const(expr->type));
      templArg->setConstValue(interp.evaluate(expr));
      if (!templArg->constValue().has_value())
        cxx_runtime_error("template argument is not a constant expression");
      templateArguments.push_back(templArg);
    } else if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      auto type = typeArg->typeId->type;
      // ### need to set scope and location
      auto templArg = control->newTypeAliasSymbol(nullptr, {});
      templArg->setType(type);
      templateArguments.push_back(templArg);
    }
  }

  return templateArguments;
}

ASTRewriter::ASTRewriter(TranslationUnit* unit, Scope* scope,
                         const std::vector<TemplateArgument>& templateArguments)
    : unit_(unit), templateArguments_(templateArguments), binder_(unit_) {
  binder_.setScope(scope);
}

ASTRewriter::~ASTRewriter() {}

auto ASTRewriter::control() const -> Control* { return unit_->control(); }

auto ASTRewriter::arena() const -> Arena* { return unit_->arena(); }

auto ASTRewriter::restrictedToDeclarations() const -> bool {
  return restrictedToDeclarations_;
}

void ASTRewriter::setRestrictedToDeclarations(bool restrictedToDeclarations) {
  restrictedToDeclarations_ = restrictedToDeclarations;
}

void ASTRewriter::check(ExpressionAST* ast) {
  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.check(ast);
}

auto ASTRewriter::getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol* {
  for (auto cursor = ASTCursor{ast, {}}; cursor; ++cursor) {
    const auto& current = *cursor;
    if (!std::holds_alternative<AST*>(current.node)) continue;

    auto id = ast_cast<IdExpressionAST>(std::get<AST*>(current.node));
    if (!id) continue;

    auto param = symbol_cast<NonTypeParameterSymbol>(id->symbol);
    if (!param) continue;

    if (param->depth() != 0) continue;

    auto arg = templateArguments_[param->index()];
    auto argSymbol = std::get<Symbol*>(arg);

    auto parameterPack = symbol_cast<ParameterPackSymbol>(argSymbol);
    if (parameterPack) return parameterPack;
  }

  return nullptr;
}

}  // namespace cxx
