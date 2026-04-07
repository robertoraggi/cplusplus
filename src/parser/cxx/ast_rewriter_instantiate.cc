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
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

struct SfinaeDiagnosticsClient final : DiagnosticsClient {
  bool hadError = false;

  void report(const Diagnostic& diagnostic) override {
    if (diagnostic.severity() == Severity::Error) {
      hadError = true;
    }
  }
};

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

    if (!functionDef) {
      for (auto redecl : symbol->redeclarations()) {
        if (auto def = redecl->declaration()) {
          functionDef = def;
          break;
        }
      }
    }

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

[[nodiscard]] auto isPrimaryTemplate(
    const std::vector<TemplateArgument>& templateArguments) -> bool {
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

[[nodiscard]] auto templateParameterCount(TemplateDeclarationAST* templateDecl)
    -> int {
  if (!templateDecl) return 0;
  int count = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    (void)parameter;
    ++count;
  }
  return count;
}

[[nodiscard]] auto computeInstantiationClassName(
    TranslationUnit* unit, Symbol* primaryTemplate,
    const std::vector<TemplateArgument>& templateArguments) -> std::string {
  if (!primaryTemplate) return "template";
  return to_string(unit->control()->getTemplateId(primaryTemplate->name(),
                                                  templateArguments));
}

[[nodiscard]] auto instantiationLabel(Symbol* symbol) -> std::string_view {
  return symbol_cast<FunctionSymbol>(symbol)
             ? "function template specialization"
             : "template class";
}

[[nodiscard]] auto findMutableSpecialization(Symbol* primary, Symbol* spec)
    -> TemplateSpecialization* {
  if (!primary || !spec) return nullptr;
  auto search = [spec](auto sym) -> TemplateSpecialization* {
    for (auto& s : sym->mutableSpecializations())
      if (s.symbol == spec) return &s;
    return nullptr;
  };
  if (auto cs = symbol_cast<ClassSymbol>(primary)) return search(cs);
  if (auto as = symbol_cast<TypeAliasSymbol>(primary)) return search(as);
  if (auto vs = symbol_cast<VariableSymbol>(primary)) return search(vs);
  if (auto fs = symbol_cast<FunctionSymbol>(primary)) return search(fs);
  return nullptr;
}

}  // namespace

auto ASTRewriter::paste(TranslationUnit* unit, ScopeSymbol* scope,
                        StatementAST* ast) -> StatementAST* {
  auto rewriter = ASTRewriter{unit, scope, {}};
  auto result = rewriter.statement(ast);
  return result;
}

auto ASTRewriter::substituteDefaultTypeId(
    TranslationUnit* unit, TypeIdAST* typeId,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    TemplateParametersSymbol* templateParams) -> TypeIdAST* {
  if (!typeId) return nullptr;
  auto rewriter = ASTRewriter{unit, nullptr,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;

  if (templateParams) {
    rewriter.binder().setScope(templateParams);
  }

  return rewriter.typeId(typeId);
}

void ASTRewriter::reportPendingInstantiationErrors(
    TranslationUnit* unit, Symbol* primaryTemplate, Symbol* instantiated,
    SourceLocation instantiationLoc) {
  if (!primaryTemplate || !instantiated || !instantiationLoc) return;
  if (auto spec = findMutableSpecialization(primaryTemplate, instantiated)) {
    if (!spec->instantiationErrors.empty()) {
      for (auto& diag : spec->instantiationErrors)
        unit->diagnosticsClient()->report(diag);
      spec->instantiationErrors.clear();
      auto name =
          computeInstantiationClassName(unit, primaryTemplate, spec->arguments);
      auto label = instantiationLabel(primaryTemplate);
      unit->note(instantiationLoc,
                 std::format("in instantiation of {} '{}' requested here",
                             label, name));
    }
  }
}

auto ASTRewriter::instantiate(TranslationUnit* unit,
                              List<TemplateArgumentAST*>* templateArgumentList,
                              Symbol* symbol, SourceLocation instantiationLoc,
                              bool sfinaeContext) -> Symbol* {
  if (!symbol) return nullptr;

  if (!unit->config().checkTypes) return nullptr;

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return nullptr;

  auto declaration = visit(GetDeclaration{}, symbol);
  if (!declaration) return nullptr;

  std::optional<SfinaeDiagnosticsClient> sfinaeClient;
  DiagnosticsClient* savedDiagClient = nullptr;
  if (sfinaeContext) {
    sfinaeClient.emplace();
    savedDiagClient = unit->changeDiagnosticsClient(&*sfinaeClient);
  }

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList);

  if (!subst) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return nullptr;
  }

  auto templateArguments = std::move(*subst).templateArguments();

  if (symbol_cast<FunctionSymbol>(symbol) &&
      static_cast<int>(templateArguments.size()) <
          templateParameterCount(templateDecl)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return symbol;
  }

  if (isPrimaryTemplate(templateArguments)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return symbol;
  }

  if (auto cached = visit(GetSpecialization{templateArguments}, symbol)) {
    auto cachedClass = symbol_cast<ClassSymbol>(cached);
    if (!cachedClass) {
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return cached;
    }
    if (cachedClass->declaration()) {
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return cached;
    }
  }

  if (!checkRequiresClause(unit, symbol, templateDecl->requiresClause,
                           templateArguments, templateDecl->depth)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return nullptr;
  }

  if (auto functionDef = ast_cast<FunctionDefinitionAST>(declaration)) {
    if (!checkRequiresClause(unit, symbol, functionDef->requiresClause,
                             templateArguments, templateDecl->depth)) {
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return nullptr;
    }
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, classSymbol, templateArguments);
    if (partial) {
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return partial;
    }
  }

  if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, variableSymbol, templateArguments);
    if (partial) {
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return partial;
    }
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;
  rewriter.binder().setInstantiatingSymbol(symbol);
  rewriter.binder().setInstantiationLoc(instantiationLoc);

  if (sfinaeContext) {
    auto instance = visit(Instantiate{rewriter}, symbol);
    (void)unit->changeDiagnosticsClient(savedDiagClient);
    if (sfinaeClient->hadError) return nullptr;
    auto bodyErrors = rewriter.takeBodyErrors();
    if (!bodyErrors.empty() && instance) {
      if (auto spec = findMutableSpecialization(symbol, instance)) {
        spec->instantiationErrors = std::move(bodyErrors);
      }
    }
    return instance;
  }

  CapturingDiagnosticsClient capturing{unit->diagnosticsClient()};
  (void)unit->changeDiagnosticsClient(&capturing);

  auto instantiatedSymbol = visit(Instantiate{rewriter}, symbol);

  (void)unit->changeDiagnosticsClient(capturing.parent);

  auto bodyErrors = rewriter.takeBodyErrors();
  capturing.diagnostics.insert(capturing.diagnostics.end(),
                               std::make_move_iterator(bodyErrors.begin()),
                               std::make_move_iterator(bodyErrors.end()));

  if (!capturing.diagnostics.empty()) {
    if (auto spec = findMutableSpecialization(symbol, instantiatedSymbol)) {
      spec->instantiationErrors = std::move(capturing.diagnostics);
    }
    if (instantiationLoc) {
      auto name =
          computeInstantiationClassName(unit, symbol, templateArguments);
      auto label = instantiationLabel(symbol);
      unit->note(instantiationLoc,
                 std::format("in instantiation of {} '{}' requested here",
                             label, name));
    }
  }

  return instantiatedSymbol;
}

auto ASTRewriter::ensureCompleteClass(TranslationUnit* unit,
                                      ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->isComplete()) return true;
  if (auto def = classSymbol->definition())
    if (def->isComplete()) return true;
  if (!classSymbol->isSpecialization()) return false;

  auto primaryTemplate = classSymbol->primaryTemplateSymbol();
  if (!primaryTemplate) return false;

  TemplateSpecialization* spec = nullptr;
  for (auto& s : primaryTemplate->mutableSpecializations()) {
    if (s.symbol == classSymbol) {
      spec = &s;
      break;
    }
  }

  if (!spec || !spec->isPendingInstantiation) return false;

  auto pendingArgList = spec->pendingArgumentList;
  auto pendingLoc = spec->pendingInstantiationLoc;
  spec->isPendingInstantiation = false;
  spec->pendingArgumentList = nullptr;

  auto result =
      instantiate(unit, pendingArgList, primaryTemplate, pendingLoc, false);

  if (!result) return false;

  auto resultClass = symbol_cast<ClassSymbol>(result);
  if (!resultClass || !resultClass->isComplete()) return false;

  if (resultClass != classSymbol) {
    classSymbol->addRedeclaration(resultClass);
    classSymbol->setDefinition(resultClass);
    resultClass->setType(classSymbol->type());
  }

  return resultClass->isComplete();
}

}  // namespace cxx
