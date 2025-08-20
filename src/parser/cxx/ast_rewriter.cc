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
#include <cxx/types.h>

// todo remove
#include <cxx/ast_cursor.h>

namespace cxx {

ASTRewriter::ASTRewriter(TranslationUnit* unit, ScopeSymbol* scope,
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

void ASTRewriter::warning(SourceLocation loc, std::string message) {
  binder_.warning(loc, std::move(message));
}

void ASTRewriter::error(SourceLocation loc, std::string message) {
  binder_.error(loc, std::move(message));
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

auto ASTRewriter::instantiateClassTemplate(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    ClassSymbol* classSymbol) -> ClassSymbol* {
  auto templateDecl = classSymbol->templateDeclaration();

  if (!classSymbol->declaration()) return nullptr;

  auto templateArguments =
      make_substitution(unit, templateDecl, templateArgumentList);

  auto is_primary_template = [&]() -> bool {
    int expected = 0;
    for (const auto& arg : templateArguments) {
      if (!std::holds_alternative<Symbol*>(arg)) return false;

      auto ty = type_cast<TypeParameterType>(std::get<Symbol*>(arg)->type());
      if (!ty) return false;

      if (ty->index() != expected) return false;
      ++expected;
    }
    return true;
  };

  if (is_primary_template()) {
    // if this is a primary template, we can just return the class symbol
    return classSymbol;
  }

  auto subst = classSymbol->findSpecialization(templateArguments);
  if (subst) {
    return subst;
  }

  auto classSpecifier = ast_cast<ClassSpecifierAST>(classSymbol->declaration());
  if (!classSpecifier) return nullptr;

  auto parentScope = classSymbol->enclosingNonTemplateParametersScope();

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};

  rewriter.binder().setInstantiatingSymbol(classSymbol);

  auto instance =
      ast_cast<ClassSpecifierAST>(rewriter.specifier(classSpecifier));

  if (!instance) return nullptr;

  auto classInstance = instance->symbol;

  return classInstance;
}

auto ASTRewriter::instantiateTypeAliasTemplate(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    TypeAliasSymbol* typeAliasSymbol) -> TypeAliasSymbol* {
  auto templateDecl = typeAliasSymbol->templateDeclaration();

  auto aliasDeclaration =
      ast_cast<AliasDeclarationAST>(templateDecl->declaration);

  if (!aliasDeclaration) return nullptr;

  auto templateArguments =
      make_substitution(unit, templateDecl, templateArgumentList);

  auto is_primary_template = [&]() -> bool {
    int expected = 0;
    for (const auto& arg : templateArguments) {
      if (!std::holds_alternative<Symbol*>(arg)) return false;

      auto ty = type_cast<TypeParameterType>(std::get<Symbol*>(arg)->type());
      if (!ty) return false;

      if (ty->index() != expected) return false;
      ++expected;
    }
    return true;
  };

  if (is_primary_template()) {
    // if this is a primary template, we can just return the class symbol
    return typeAliasSymbol;
  }

#if false
  auto subst = typeAliasSymbol->findSpecialization(templateArguments);
  if (subst) {
    return subst;
  }
#endif

  auto parentScope = typeAliasSymbol->parent();
  while (parentScope->isTemplateParameters()) {
    parentScope = parentScope->parent();
  }

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};

  rewriter.binder().setInstantiatingSymbol(typeAliasSymbol);

  auto instance =
      ast_cast<AliasDeclarationAST>(rewriter.declaration(aliasDeclaration));

  if (!instance) return nullptr;

  return instance->symbol;
}

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
      auto value = interp.evaluate(expr);
      if (!value.has_value()) {
#if false
        unit->error(arg->firstSourceLocation(),
                    "template argument is not a constant expression");
#endif
        continue;
      }

      // ### need to set scope and location
      auto templArg = control->newVariableSymbol(nullptr, {});
      templArg->setInitializer(expr);
      templArg->setType(control->add_const(expr->type));
      templArg->setConstValue(value);
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

}  // namespace cxx
