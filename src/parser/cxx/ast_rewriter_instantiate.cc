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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>
#include <cxx/views/symbols.h>

#include <format>
#include <unordered_set>

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
  ScopeSymbol* parentScope = nullptr;
  bool declarationOnly = false;

  void attachDeferredBody(FunctionDefinitionAST* instance,
                          FunctionDefinitionAST* pattern) {
    if (instance->functionBody) return;
    auto fn = symbol_cast<FunctionSymbol>(instance->symbol);
    if (!fn || fn->hasPendingBody()) return;
    auto pending = std::make_unique<PendingBodyInstantiation>();
    pending->originalDefinition = pattern;
    pending->templateArguments = rewriter.templateArguments();
    pending->parentScope = parentScope;
    pending->depth = rewriter.depth();
    fn->setPendingBody(std::move(pending));
  }

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
    rewriter.retryPendingMemberTemplateAttachment(symbol);

    if (symbol->hasPendingBody()) {
      rewriter.completePendingBody(symbol);
    }

    auto functionDef = symbol->declaration();
    auto definingSymbol = symbol;

    if (!functionDef) {
      for (auto redecl : symbol->redeclarations()) {
        if (auto def = redecl->declaration()) {
          functionDef = def;
          definingSymbol = redecl;
          break;
        }
      }
    }

    if (definingSymbol != symbol) {
      if (auto definingTemplateDecl = definingSymbol->templateDeclaration()) {
        rewriter.setDepth(definingTemplateDecl->depth);
      }
    }

    rewriter.setInstantiatingFunctionTemplateSpecialization(
        symbol->templateDeclaration() != nullptr);

    if (functionDef) {
      auto instance =
          ast_cast<FunctionDefinitionAST>(rewriter.declaration(functionDef));
      if (!instance) return nullptr;
      if (declarationOnly) attachDeferredBody(instance, functionDef);
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
  if (templateArguments.empty()) return false;

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
    ScopeSymbol* scope) -> TypeIdAST* {
  if (!typeId) return nullptr;
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  return rewriter.typeId(typeId);
}

auto ASTRewriter::substituteDefaultExpression(
    TranslationUnit* unit, ExpressionAST* expression,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> ExpressionAST* {
  if (!expression) return nullptr;
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  return rewriter.expression(expression);
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

auto ASTRewriter::instantiateForArgs(
    TranslationUnit* unit, List<TemplateArgumentAST*>* deducedArguments,
    FunctionSymbol* function, SourceLocation instantiationLoc,
    bool argsComplete, bool declarationOnly) -> FunctionSymbol* {
  return symbol_cast<FunctionSymbol>(
      instantiate(unit, deducedArguments, function, instantiationLoc,
                  /*sfinaeContext=*/true, argsComplete, declarationOnly));
}

auto ASTRewriter::instantiate(TranslationUnit* unit,
                              List<TemplateArgumentAST*>* templateArgumentList,
                              Symbol* symbol, SourceLocation instantiationLoc,
                              bool sfinaeContext, bool argsComplete,
                              bool declarationOnly) -> Symbol* {
  if (!symbol) return nullptr;

  if (!unit->config().checkTypes) return nullptr;

  const auto activeClientIsSfinae =
      unit->diagnosticsClient() && unit->diagnosticsClient()->isSfinae();

  if (!sfinaeContext && activeClientIsSfinae) {
    sfinaeContext = true;
  }

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return nullptr;

  auto declaration = visit(GetDeclaration{}, symbol);
  if (!declaration) return nullptr;

  const bool ownsSfinaeClient = sfinaeContext && !activeClientIsSfinae;

  std::optional<SilentDiagnosticsClient> sfinaeClient;
  DiagnosticsClient* savedDiagClient = nullptr;
  if (ownsSfinaeClient) {
    sfinaeClient.emplace();
    savedDiagClient = unit->changeDiagnosticsClient(&*sfinaeClient);
  }

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  argsComplete);

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
      if (!declarationOnly) {
        if (auto cachedFn = symbol_cast<FunctionSymbol>(cached);
            cachedFn && cachedFn->hasPendingBody()) {
          auto bodyErrors = ASTRewriter::completePendingBodyFor(
              unit, cachedFn, /*captureBodyErrors=*/true);
          if (!bodyErrors.empty()) {
            if (auto spec = findMutableSpecialization(symbol, cached)) {
              spec->instantiationErrors = std::move(bodyErrors);
            }
          }
        }
      }
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
  if (declarationOnly) rewriter.setRestrictedToDeclarations(true);

  auto registerFunctionSpecialization = [&](Symbol* result) {
    if (!result || result == symbol) return;
    auto fnTemplate = symbol_cast<FunctionSymbol>(symbol);
    if (!fnTemplate) return;
    auto instance = symbol_cast<FunctionSymbol>(result);
    if (!instance || instance->isSpecialization()) return;

    if (fnTemplate->isFriend()) instance->setFriend(true);

    if (fnTemplate->findSpecialization(templateArguments)) return;
    fnTemplate->addSpecialization(templateArguments, instance);
  };

  if (sfinaeContext) {
    auto instance =
        visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);
    if (ownsSfinaeClient) {
      (void)unit->changeDiagnosticsClient(savedDiagClient);
      if (sfinaeClient->hadError()) return nullptr;
    }
    if (rewriter.substitutionFailed()) return nullptr;

    registerFunctionSpecialization(instance);

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

  auto instantiatedSymbol =
      visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);

  (void)unit->changeDiagnosticsClient(capturing.parent);

  registerFunctionSpecialization(instantiatedSymbol);

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

void ASTRewriter::markExplicitInstantiationDeclared(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    Symbol* symbol) {
  if (!symbol) return;
  if (!unit->config().checkTypes) return;

  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol) return;

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return;

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  /*argsComplete=*/true);
  if (!subst) return;

  auto templateArguments = std::move(*subst).templateArguments();
  if (isPrimaryTemplate(templateArguments)) return;

  classSymbol->addExternInstantiationDeclaration(std::move(templateArguments));
}

auto ASTRewriter::ensureCompleteClass(TranslationUnit* unit,
                                      ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->resolvedDefinition()->isComplete()) return true;
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

void ASTRewriter::instantiateOutOfClassMemberDefinitions(ClassSymbol* pattern) {
  if (!pattern) return;
  if (templateArguments_.empty()) return;

  auto instantiateDefinition = [&](Symbol* def, DeclarationAST* declaration,
                                   TemplateDeclarationAST* templateDecl) {
    auto lexicalScope = def->enclosingNonTemplateParametersScope();
    while (lexicalScope && lexicalScope->isClass()) {
      lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
    }
    auto rewriter = ASTRewriter{
        unit_, lexicalScope, std::vector<TemplateArgument>(templateArguments_)};
    rewriter.depth_ = templateDecl->depth;
    rewriter.binder_.setInstantiatingSymbol(def);
    if (auto instance = symbol_cast<ClassSymbol>(
            pattern->findSpecialization(templateArguments_))) {
      rewriter.remapScopeMembers(pattern, instance);
    }
    (void)rewriter.declaration(declaration);
  };

  auto instanceClass =
      symbol_cast<ClassSymbol>(pattern->findSpecialization(templateArguments_));
  if (instanceClass) {
    if (auto def = symbol_cast<ClassSymbol>(instanceClass->definition());
        def && def != instanceClass) {
      instanceClass = def;
    }
  }

  if (instanceClass) unit_->addPendingMemberInstantiation(instanceClass);

  auto attachPendingBody = [&](FunctionSymbol* target, FunctionSymbol* def,
                               FunctionDefinitionAST* defAst, int depth) {
    auto lexicalScope = def->enclosingNonTemplateParametersScope();
    while (lexicalScope && lexicalScope->isClass()) {
      lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
    }

    auto pending = std::make_unique<PendingBodyInstantiation>();
    pending->originalDefinition = defAst;
    pending->templateArguments = templateArguments_;
    pending->parentScope = lexicalScope;
    pending->depth = depth;
    target->setPendingBody(std::move(pending));
  };

  auto attachPendingDefinition = [&](FunctionSymbol* member) {
    if (!instanceClass) return;
    if (member->isFriend()) return;
    auto def = symbol_cast<FunctionSymbol>(member->definition());
    if (!def || def == member) return;
    auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
    if (!defAst) return;
    auto classTemplateDecl = pattern->templateDeclaration();
    if (!classTemplateDecl) return;

    FunctionSymbol* instanceMember = nullptr;
    for (auto cand : instanceClass->find(member->name())) {
      for (auto fn : views::each_function(cand)) {
        if (!fn->templateDeclaration()) continue;
        if (fn->declaration()) continue;
        if (!areTemplateHeadsEquivalentForRedeclaration(
                unit_, fn->templateDeclaration(),
                member->templateDeclaration())) {
          continue;
        }
        instanceMember = fn;
        break;
      }
      if (instanceMember) break;
    }
    if (!instanceMember) return;
    if (instanceMember->hasPendingBody()) return;

    attachPendingBody(instanceMember, def, defAst, classTemplateDecl->depth);
  };

  auto instantiateFunctionDefinition = [&](FunctionSymbol* member) {
    if (member->templateDeclaration()) return;
    auto def = symbol_cast<FunctionSymbol>(member->definition());
    if (!def || def == member) return;
    auto templateDecl = def->templateDeclaration();
    if (!templateDecl) return;
    auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
    if (!defAst) return;
    if (def->findSpecialization(templateArguments_)) return;
    instantiateDefinition(def, defAst, templateDecl);
  };

  for (auto member : pattern->members()) {
    for (auto function : views::each_function(member)) {
      if (function->templateDeclaration()) attachPendingDefinition(function);
    }
  }

  if (instanceClass) {
    auto classTemplateDecl = pattern->templateDeclaration();
    if (classTemplateDecl) {
      for (auto instanceCtor : instanceClass->constructors()) {
        if (!instanceCtor->templateDeclaration()) continue;
        if (instanceCtor->declaration()) continue;
        if (instanceCtor->hasPendingBody()) continue;

        FunctionSymbol* patternCtor = nullptr;
        for (auto candidate : pattern->constructors()) {
          if (!candidate->templateDeclaration()) continue;
          if (candidate->isFriend()) continue;
          if (!areTemplateHeadsEquivalentForRedeclaration(
                  unit_, instanceCtor->templateDeclaration(),
                  candidate->templateDeclaration())) {
            continue;
          }
          patternCtor = candidate;
          break;
        }
        if (!patternCtor) continue;

        auto def = symbol_cast<FunctionSymbol>(patternCtor->definition());
        if (!def || def == patternCtor) continue;
        auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
        if (!defAst) continue;

        attachPendingBody(instanceCtor, def, defAst, classTemplateDecl->depth);
      }
    }
  }

  for (auto member : pattern->members()) {
    auto memberClass = symbol_cast<ClassSymbol>(member);
    if (!memberClass) continue;
    auto def = symbol_cast<ClassSymbol>(memberClass->definition());
    if (!def || def == memberClass) continue;
    auto templateDecl = def->templateDeclaration();
    if (!templateDecl) continue;
    auto declAst = ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declAst) continue;
    if (instanceClass) {
      bool alreadyComplete = false;
      for (auto cand : instanceClass->find(memberClass->name())) {
        if (auto cls = symbol_cast<ClassSymbol>(cand);
            cls && cls->isComplete()) {
          alreadyComplete = true;
          break;
        }
      }
      if (alreadyComplete) continue;
    }
    instantiateDefinition(def, declAst, templateDecl);
  }

  for (auto member : pattern->members()) {
    for (auto function : views::each_function(member)) {
      instantiateFunctionDefinition(function);
    }
    if (auto field = symbol_cast<FieldSymbol>(member)) {
      if (!field->isStatic()) continue;
      auto def = field->definition();
      if (!def) continue;
      auto templateDecl = def->templateDeclaration();
      if (!templateDecl) continue;
      auto declAst = ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
      if (!declAst) continue;
      if (def->findSpecialization(templateArguments_)) continue;
      instantiateDefinition(def, declAst, templateDecl);
    }
  }

  if (instanceClass) {
    std::vector<FunctionSymbol*> patternCtors;
    for (auto ctor : pattern->constructors()) {
      if (ctor->canonical() == ctor && !ctor->isDefaulted())
        patternCtors.push_back(ctor);
    }
    std::vector<FunctionSymbol*> instanceCtors;
    for (auto ctor : instanceClass->constructors()) {
      if (ctor->canonical() == ctor && !ctor->isDefaulted())
        instanceCtors.push_back(ctor);
    }
    auto classTemplateDecl = pattern->templateDeclaration();
    for (std::size_t i = 0; classTemplateDecl && i < patternCtors.size() &&
                            i < instanceCtors.size();
         ++i) {
      auto ctor = patternCtors[i];
      if (ctor->templateDeclaration()) continue;
      auto def = symbol_cast<FunctionSymbol>(ctor->definition());
      if (!def || def == ctor) continue;
      auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
      if (!defAst) continue;

      auto instanceCtor = instanceCtors[i];
      const bool alreadyHandled = instanceCtor->hasPendingBody() ||
                                  (instanceCtor->declaration() &&
                                   instanceCtor->declaration()->functionBody);
      if (alreadyHandled) continue;

      auto ctorType = type_cast<FunctionType>(ctor->type());
      auto instanceCtorType = type_cast<FunctionType>(instanceCtor->type());
      if (!ctorType || !instanceCtorType ||
          ctorType->parameterTypes().size() !=
              instanceCtorType->parameterTypes().size()) {
        continue;
      }

      attachPendingBody(instanceCtor, def, defAst, classTemplateDecl->depth);
      unit_->addPendingBodyCompletion(instanceCtor);
    }
  }
}

void ASTRewriter::retryPendingMemberTemplateAttachment(FunctionSymbol* member) {
  if (!member || member->hasPendingBody() || member->declaration()) return;
  if (!member->templateDeclaration() || member->isFriend()) return;

  auto instanceClass =
      symbol_cast<ClassSymbol>(member->enclosingNonTemplateParametersScope());
  if (!instanceClass) return;

  auto pattern = instanceClass->primaryTemplateSymbol();
  if (!pattern) return;

  auto classTemplateDecl = pattern->templateDeclaration();
  if (!classTemplateDecl) return;

  FunctionSymbol* patternDef = nullptr;
  FunctionDefinitionAST* patternDefAst = nullptr;
  for (auto candidate : pattern->find(member->name())) {
    for (auto function : views::each_function(candidate)) {
      if (!function->templateDeclaration()) continue;
      if (function->isFriend()) continue;
      auto def = symbol_cast<FunctionSymbol>(function->definition());
      if (!def || def == function) continue;
      auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
      if (!defAst) continue;
      if (!areTemplateHeadsEquivalentForRedeclaration(
              unit_, member->templateDeclaration(),
              function->templateDeclaration())) {
        continue;
      }
      patternDef = def;
      patternDefAst = defAst;
      break;
    }
    if (patternDef) break;
  }
  if (!patternDef) return;

  auto lexicalScope = patternDef->enclosingNonTemplateParametersScope();
  while (lexicalScope && lexicalScope->isClass()) {
    lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
  }

  auto classArgs = instanceClass->templateArguments();
  auto pending = std::make_unique<PendingBodyInstantiation>();
  pending->originalDefinition = patternDefAst;
  pending->templateArguments =
      std::vector<TemplateArgument>(classArgs.begin(), classArgs.end());
  pending->parentScope = lexicalScope;
  pending->depth = classTemplateDecl->depth;
  member->setPendingBody(std::move(pending));
}

void ASTRewriter::completePendingMemberInstantiations(TranslationUnit* unit) {
  if (!unit || !unit->config().checkTypes) return;

  std::unordered_set<ClassSymbol*> processed;
  for (int round = 0; round < 32; ++round) {
    bool any = false;

    SilentDiagnosticsClient silent;
    auto pendingBodies = unit->takePendingBodyCompletions();
    for (auto function : pendingBodies) {
      if (!function || !function->hasPendingBody()) continue;
      any = true;
      auto saved = unit->changeDiagnosticsClient(&silent);
      auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
      rewriter.completePendingBody(function);
      (void)unit->changeDiagnosticsClient(saved);
    }

    auto pending = unit->takePendingMemberInstantiations();
    for (auto instance : pending) {
      if (!instance) continue;
      if (!processed.insert(instance).second) continue;
      auto pattern = instance->primaryTemplateSymbol();
      if (!pattern) continue;
      auto args = instance->templateArguments();
      if (args.empty()) continue;
      any = true;
      auto rewriter =
          ASTRewriter{unit, unit->globalScope(),
                      std::vector<TemplateArgument>(args.begin(), args.end())};
      rewriter.instantiateOutOfClassMemberDefinitions(pattern);
    }
    if (!any) break;
  }
}
}  // namespace cxx
