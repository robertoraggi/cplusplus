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
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

namespace {

struct GetTemplateDeclaration {
  auto operator()(ClassSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(VariableSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(TypeAliasSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(FunctionSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(Symbol*) -> TemplateDeclarationAST* { return nullptr; }
};

struct GetDeclaration {
  auto operator()(ClassSymbol* symbol) -> AST* { return symbol->declaration(); }

  auto operator()(VariableSymbol* symbol) -> AST* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;
    return templateDecl->declaration;
  }

  auto operator()(TypeAliasSymbol* symbol) -> AST* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;
    return templateDecl->declaration;
  }

  auto operator()(FunctionSymbol* symbol) -> AST* {
    if (auto declaration = symbol->declaration()) return declaration;

    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    return templateDecl->declaration;
  }

  auto operator()(Symbol*) -> AST* { return nullptr; }
};

struct GetSpecialization {
  const std::vector<TemplateArgument>& templateArguments;

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

struct Instantiate {
  ASTRewriter& rewriter;

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    auto classSpecifier = ast_cast<ClassSpecifierAST>(symbol->declaration());
    if (!classSpecifier) return nullptr;

    auto instance =
        ast_cast<ClassSpecifierAST>(rewriter.specifier(classSpecifier));
    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration = templateDecl->declaration;
    auto simpleDecl = ast_cast<SimpleDeclarationAST>(declaration);
    if (!simpleDecl) return nullptr;

    auto instance =
        ast_cast<SimpleDeclarationAST>(rewriter.declaration(simpleDecl));
    if (!instance || !instance->initDeclaratorList ||
        !instance->initDeclaratorList->value) {
      return nullptr;
    }

    auto instantiatedSymbol = instance->initDeclaratorList->value->symbol;
    if (!instantiatedSymbol) return nullptr;
    return symbol_cast<VariableSymbol>(instantiatedSymbol);
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration = ast_cast<AliasDeclarationAST>(templateDecl->declaration);
    if (!declaration) return nullptr;

    auto instance =
        ast_cast<AliasDeclarationAST>(rewriter.declaration(declaration));
    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    auto functionDef = symbol->declaration();
    if (functionDef) {
      auto instance =
          ast_cast<FunctionDefinitionAST>(rewriter.declaration(functionDef));
      if (!instance) return nullptr;
      return instance->symbol;
    }

    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration =
        ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declaration) return nullptr;

    auto instance =
        ast_cast<SimpleDeclarationAST>(rewriter.declaration(declaration));
    if (!instance || !instance->initDeclaratorList ||
        !instance->initDeclaratorList->value) {
      return nullptr;
    }

    return instance->initDeclaratorList->value->symbol;
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

auto isPrimaryTemplate(const std::vector<TemplateArgument>& templateArguments)
    -> bool {
  int expected = 0;
  for (const auto& arg : templateArguments) {
    if (!std::holds_alternative<Symbol*>(arg)) return false;

    auto sym = std::get<Symbol*>(arg);
    if (!sym) return false;

    if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
      if (pack->elements().size() != 1) return false;
      auto element = pack->elements()[0];
      if (!element) return false;

      auto elementType = element->type();
      if (!elementType) return false;

      auto ty = getTypeParamInfo(elementType);
      if (!ty) return false;
      if (ty->index != expected) return false;
      if (!ty->isPack) return false;
      ++expected;
      continue;
    }

    auto symType = sym->type();
    if (!symType) return false;

    auto ty = getTypeParamInfo(symType);
    if (!ty) return false;
    if (ty->index != expected) return false;
    ++expected;
  }
  return true;
}

auto templateParameterCount(TemplateDeclarationAST* templateDecl) -> int {
  if (!templateDecl) return 0;
  int count = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    (void)parameter;
    ++count;
  }
  return count;
}

}  // namespace

auto ASTRewriter::paste(TranslationUnit* unit, ScopeSymbol* scope,
                        StatementAST* ast) -> StatementAST* {
  auto rewriter = ASTRewriter{unit, scope, {}};
  auto result = rewriter.statement(ast);
  return result;
}

auto ASTRewriter::instantiate(TranslationUnit* unit,
                              List<TemplateArgumentAST*>* templateArgumentList,
                              Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return nullptr;

  auto declaration = visit(GetDeclaration{}, symbol);
  if (!declaration) return nullptr;

  auto templateArguments =
      Substitution(unit, templateDecl, templateArgumentList)
          .templateArguments();

  if (symbol_cast<FunctionSymbol>(symbol) &&
      static_cast<int>(templateArguments.size()) <
          templateParameterCount(templateDecl)) {
    return symbol;
  }

  if (isPrimaryTemplate(templateArguments)) return symbol;

  if (auto cached = visit(GetSpecialization{templateArguments}, symbol)) {
    auto cachedClass = symbol_cast<ClassSymbol>(cached);
    if (!cachedClass) return cached;
    if (cachedClass->declaration()) return cached;
  }

  if (!checkRequiresClause(unit, symbol, templateDecl->requiresClause,
                           templateArguments, templateDecl->depth)) {
    return nullptr;
  }

  if (auto functionDef = ast_cast<FunctionDefinitionAST>(declaration)) {
    if (!checkRequiresClause(unit, symbol, functionDef->requiresClause,
                             templateArguments, templateDecl->depth)) {
      return nullptr;
    }
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, classSymbol, templateArguments);
    if (partial) return partial;
  }

  if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, variableSymbol, templateArguments);
    if (partial) return partial;
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;
  rewriter.binder().setInstantiatingSymbol(symbol);

  auto instantiatedSymbol = visit(Instantiate{rewriter}, symbol);

  return instantiatedSymbol;
}

}  // namespace cxx
