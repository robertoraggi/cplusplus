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
#include <cxx/ast_validator.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/function_body.h>
#include <cxx/function_body_warnings.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <iterator>
#include <optional>

namespace cxx {
namespace {
[[nodiscard]] auto parametersOf(FunctionParametersSymbol* parameters)
    -> std::vector<ParameterSymbol*> {
  std::vector<ParameterSymbol*> result;
  for (auto member : parameters->members()) {
    if (auto parameter = symbol_cast<ParameterSymbol>(member))
      result.push_back(parameter);
  }
  return result;
}

}  // namespace

void ASTRewriter::remapScopeMembers(ScopeSymbol* oldScope,
                                    ScopeSymbol* newScope) {
  if (!oldScope || !newScope || oldScope == newScope) return;

  if (auto oldClass = symbol_cast<ClassSymbol>(oldScope)) {
    oldScope = oldClass->resolvedDefinition();
  }
  if (auto newClass = symbol_cast<ClassSymbol>(newScope)) {
    newScope = newClass->resolvedDefinition();
  }
  if (oldScope == newScope) return;

  addSymbolRemap(oldScope, newScope);

  for (auto member : newScope->members()) remapInstantiatedMember(member);
}

void ASTRewriter::remapInstantiationPatterns(Symbol* instance) {
  for (auto pattern = instance->instantiationPattern(); pattern;
       pattern = pattern->instantiationPattern()) {
    addSymbolRemap(pattern, instance);
  }
}

void ASTRewriter::remapInstantiatedMember(Symbol* member) {
  for (auto function : views::declared_functions(member))
    remapInstantiationPatterns(function);

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(member)) {
    for (auto usingDeclaration : overloadSet->usingDeclarations())
      remapInstantiatedMember(usingDeclaration);
  }

  auto pattern = member->instantiationPattern();
  if (!pattern) return;

  remapInstantiationPatterns(member);

  if (auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(member)) {
    if (auto patternUsing = symbol_cast<UsingDeclarationSymbol>(pattern))
      addSymbolRemap(patternUsing->target(), usingDeclaration->target());
    return;
  }

  if (!member->isClass() && !member->isEnumOrScopedEnum()) return;

  remapScopeMembers(pattern->asScopeSymbol(), member->asScopeSymbol());
}

void ASTRewriter::remapInitCaptures(LambdaSymbol* pattern,
                                    LambdaSymbol* instance) {
  addSymbolRemap(pattern, instance);
  for (auto variable : instance->members() | views::variables) {
    for (auto patternVariable :
         pattern->find(variable->name()) | views::variables) {
      addSymbolRemap(patternVariable, variable);
    }
  }
}

void ASTRewriter::remapEnclosingClassPatterns(ScopeSymbol* scope) {
  for (auto current = scope; current; current = current->parent()) {
    auto instanceClass = symbol_cast<ClassSymbol>(current);
    if (!instanceClass) continue;
    auto patternClass = instanceClass->instantiationTemplate();
    if (!patternClass) continue;
    remapScopeMembers(patternClass, instanceClass);
  }
}

template <typename S>
auto ASTRewriter::remappedMemberTemplate(ClassSymbol* instanceClass, S* pattern,
                                         const Identifier* name) -> S* {
  auto patternTemplate = pattern;
  if (pattern->isSpecialization()) {
    patternTemplate = pattern->primaryTemplateSymbol();
  }

  if (patternTemplate) {
    auto remapped = symbol_cast<S>(remapSymbol(patternTemplate));
    if (remapped && remapped != patternTemplate &&
        remapped->parent() == instanceClass && remapped->isTemplatePattern()) {
      return remapped;
    }
  }

  if constexpr (std::is_same_v<S, FunctionSymbol>) {
    return views::find_function(instanceClass->find(name),
                                &FunctionSymbol::isTemplatePattern);
  } else {
    for (auto variable : instanceClass->find(name) | views::variables) {
      if (variable->isTemplatePattern()) return variable;
    }
    return nullptr;
  }
}

struct ASTRewriter::MemberTemplateInstantiation {
  ASTRewriter& rewrite;
  SimpleTemplateIdAST* templateId;
  CallExpressionAST* call;
  SourceLocation location;

  [[nodiscard]] auto explicitTemplateArguments() const
      -> List<TemplateArgumentAST*>* {
    if (!templateId) return nullptr;
    return templateId->templateArgumentList;
  }

  [[nodiscard]] auto calledSpecialization(FunctionSymbol* instanceTemplate)
      -> FunctionSymbol* {
    TemplateArgumentDeduction deduction{rewrite.unit_};
    auto deduced = deduction.deduce(instanceTemplate, call->expressionList,
                                    explicitTemplateArguments());
    if (!deduced.has_value()) return nullptr;

    return symbol_cast<FunctionSymbol>(ASTRewriter::instantiate(
        rewrite.unit_, *deduced, instanceTemplate, location,
        /*sfinaeContext=*/false, /*argsComplete=*/true,
        /*declarationOnly=*/true));
  }

  template <typename S>
  [[nodiscard]] auto writtenSpecialization(S* instanceTemplate) -> S* {
    if (!templateId) return nullptr;
    return symbol_cast<S>(ASTRewriter::instantiate(
        rewrite.unit_, templateId->templateArgumentList, instanceTemplate,
        location, /*sfinaeContext=*/false, /*argsComplete=*/false,
        /*declarationOnly=*/std::is_same_v<S, FunctionSymbol>));
  }

  template <typename S>
  [[nodiscard]] auto specializationOf(S* instanceTemplate) -> S* {
    if constexpr (std::is_same_v<S, FunctionSymbol>) {
      if (call) return calledSpecialization(instanceTemplate);
    }
    return writtenSpecialization(instanceTemplate);
  }

  template <typename S>
  [[nodiscard]] auto instantiated(S* pattern) -> Symbol* {
    if (!pattern->isTemplatePattern() && !pattern->isSpecialization())
      return nullptr;

    auto patternClass = symbol_cast<ClassSymbol>(pattern->parent());
    if (!patternClass) return nullptr;

    auto instanceClass =
        symbol_cast<ClassSymbol>(rewrite.remapSymbol(patternClass));
    if (!instanceClass || instanceClass == patternClass) return nullptr;

    auto instanceTemplate = rewrite.remappedMemberTemplate(
        instanceClass, pattern, name_cast<Identifier>(pattern->name()));
    if (!instanceTemplate) return nullptr;
    if (!pattern->isSpecialization()) return instanceTemplate;

    if (auto specialization = specializationOf(instanceTemplate))
      return specialization;
    return instanceTemplate;
  }

  [[nodiscard]] auto operator()(FunctionSymbol* pattern) -> Symbol* {
    return instantiated(pattern);
  }

  [[nodiscard]] auto operator()(VariableSymbol* pattern) -> Symbol* {
    return instantiated(pattern);
  }

  [[nodiscard]] auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

auto ASTRewriter::instantiatedMemberTemplateFor(Symbol* pattern,
                                                SimpleTemplateIdAST* templateId,
                                                CallExpressionAST* call,
                                                SourceLocation location)
    -> Symbol* {
  if (!pattern) return nullptr;
  if (!templateId && !call) return nullptr;
  return visit(MemberTemplateInstantiation{*this, templateId, call, location},
               pattern);
}

void ASTRewriter::remapFunctionParameters(
    FunctionParametersSymbol* patternParameters,
    FunctionParametersSymbol* instanceParameters) {
  const auto patternMembers = parametersOf(patternParameters);
  const auto instanceMembers = parametersOf(instanceParameters);

  std::size_t instanceIndex = 0;

  for (std::size_t i = 0; i < patternMembers.size(); ++i) {
    std::size_t reservedForTrailingParameters = 0;
    for (auto j = i + 1; j < patternMembers.size(); ++j)
      if (!patternMembers[j]->isParameterPack())
        ++reservedForTrailingParameters;

    const auto available = instanceMembers.size() - instanceIndex;

    const auto stillPacked = instanceIndex < instanceMembers.size() &&
                             instanceMembers[instanceIndex]->isParameterPack();

    if (!patternMembers[i]->isParameterPack() || stillPacked) {
      if (available <= reservedForTrailingParameters) break;
      addSymbolRemap(patternMembers[i], instanceMembers[instanceIndex++]);
      continue;
    }

    auto pack =
        control()->newParameterPackSymbol(instanceParameters, SourceLocation{});

    for (auto last = instanceIndex + available - reservedForTrailingParameters;
         instanceIndex < last; ++instanceIndex) {
      pack->addElement(instanceMembers[instanceIndex]);
    }

    functionParamPacks_[patternMembers[i]] = pack;
  }
}

void ASTRewriter::checkMemInitializers(FunctionSymbol* function,
                                       FunctionBodyAST* body) {
  OutsideImmediateContextScope outsideImmediateContext{unit_};

  TypeChecker check{unit_};
  check.setScope(function);
  check.setReportErrors(unit_->config().checkTypes);
  auto hasDependentInitializer = [&] {
    for (auto memInit : ListView{ctor_initializers(body)}) {
      if (auto paren = ast_cast<ParenMemInitializerAST>(memInit)) {
        for (auto expression : ListView{paren->expressionList})
          if (isDependent(unit_, expression)) return true;
      } else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit)) {
        if (braced->bracedInitList &&
            isDependent(unit_, braced->bracedInitList))
          return true;
      }
    }
    return false;
  };
  if (isEnclosedInDependentTemplate(unit_, function, true) ||
      hasDependentInitializer()) {
    check.bind_template_parameter_base_initializers(body);
  } else {
    check.check_mem_initializers(body);
  }
}

auto ASTRewriter::completePendingBodyFor(TranslationUnit* unit,
                                         FunctionSymbol* function,
                                         bool captureBodyErrors)
    -> std::vector<Diagnostic> {
  if (!unit || !function) return {};
  attachPatternDefinition(unit, function);
  if (!function->hasPendingBody()) return {};
  auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
  return rewriter.completePendingBody(function, captureBodyErrors);
}

void ASTRewriter::requirePotentiallyInvokedDestructors(
    TranslationUnit* unit, FunctionSymbol* destructor) {
  if (!unit || !destructor || !destructor->isDestructor()) return;
  if (!unit->requiresDefinitions()) return;

  auto classSymbol = symbol_cast<ClassSymbol>(destructor->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  auto requireDestructorOf = [&](const Type* type) {
    requireDestructorOfType(unit, type);
  };

  for (auto baseClass : classSymbol->baseClasses()) {
    if (auto base = baseClass->symbol()) requireDestructorOf(base->type());
  }

  if (auto layout = classSymbol->layout()) {
    for (auto virtualBase : layout->virtualBases())
      requireDestructorOf(virtualBase->type());
  }

  for (auto field : classSymbol->members() | views::non_static_fields) {
    requireDestructorOf(field->type());
  }
}

void ASTRewriter::requireDestructorOfType(TranslationUnit* unit,
                                          const Type* type) {
  if (!unit || !type) return;
  if (!unit->requiresDefinitions()) return;

  TypeTraits traits{unit};
  auto objectType = traits.remove_cv(traits.remove_all_extents(type));
  auto classType = unqualified_cast<ClassType>(objectType);
  if (!classType || !classType->symbol()) return;

  if (!ensureCompleteClass(unit, classType->symbol())) return;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  requireFunctionDefinition(unit, classSymbol->destructor());
}

void ASTRewriter::requireConstructorInitializers(TranslationUnit* unit,
                                                 FunctionSymbol* constructor) {
  if (constructor->hasPendingBody()) return;
  auto declaration = constructor->declaration();
  if (!declaration) return;
  for (auto initializer :
       ListView{ctor_initializers(declaration->functionBody)})
    requireDefinitionsNamedBy(unit, initializer);
}

void ASTRewriter::requireExplicitInstantiationMembers(TranslationUnit* unit,
                                                      ClassSymbol* instance) {
  if (!unit || !instance) return;

  for (auto function : views::members(instance->resolvedDefinition()) |
                           views::member_functions) {
    if (function->isExcludedFromExplicitInstantiation()) continue;
    if (!isUserProvided(function)) continue;
    auto definition = function->resolvedDefinition();
    if (!definition->isDefined() && !definition->hasPendingBody()) continue;
    requireFunctionDefinition(unit, function);
    unit->addExplicitInstantiationDefinition(function);
  }
}

void ASTRewriter::requireFunctionDefinition(TranslationUnit* unit,
                                            FunctionSymbol* function,
                                            SourceLocation location,
                                            FunctionSymbol* caller) {
  if (!unit || !function) return;
  if (!unit->requiresDefinitions()) return;
  const auto alreadyRequired = function->isDefinitionRequired();
  function->setDefinitionRequired(true);
  unit->addPendingBodyCompletion(function, location, caller);
  if (alreadyRequired) return;
  if (!function->hasPendingBody()) {
    auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
    rewriter.binder_.synthesizeDefaultedMemberBody(function);
  }
  requireFunctionDefinition(unit, function->inheritedConstructor());
  auto definition = function->resolvedDefinition();
  const bool definesBody =
      definition->isDefined() || definition->hasPendingBody();

  if (definition->isDefaulted())
    if (auto declaration = definition->declaration())
      requireDefinitionsNamedBy(unit, declaration->functionBody);

  if (function->isDestructor() && definesBody)
    requirePotentiallyInvokedDestructors(unit, definition);

  if (function->isConstructor())
    requireConstructorInitializers(unit, definition);
}

namespace {

struct RequireNamedDefinitions final : ASTVisitor {
  TranslationUnit* unit;

  explicit RequireNamedDefinitions(TranslationUnit* unit) : unit(unit) {}

  void requireEntity(Symbol* symbol) {
    ASTRewriter::requireFunctionDefinition(unit,
                                           symbol_cast<FunctionSymbol>(symbol));
    ASTRewriter::requireFieldDefinition(unit, symbol_cast<FieldSymbol>(symbol));
  }

  void visit(ParenMemInitializerAST* ast) override {
    requireEntity(ast->constructor);
    ASTVisitor::visit(ast);
  }

  void visit(BracedMemInitializerAST* ast) override {
    requireEntity(ast->constructor);
    ASTVisitor::visit(ast);
  }

  void visit(IdExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(MemberExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(SpliceMemberExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(CallExpressionAST* ast) override {
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(TypeConstructionAST* ast) override {
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(BracedTypeConstructionAST* ast) override {
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(DesignatedInitializerClauseAST* ast) override {
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(LambdaExpressionAST* ast) override {
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(NewExpressionAST* ast) override {
    requireEntity(ast->symbol);
    requireEntity(ast->constructorSymbol);
    ASTVisitor::visit(ast);
  }

  void visit(DeleteExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(SubscriptExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(UnaryExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(PostIncrExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(BinaryExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(AssignmentExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(CompoundAssignmentExpressionAST* ast) override {
    requireEntity(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(ImplicitCastExpressionAST* ast) override {
    requireEntity(ast->conversionFunction);
    if (ast->castKind ==
        ImplicitCastKind::kTemporaryMaterializationConversion) {
      ASTRewriter::requireDestructorOfType(unit, ast->type);
    }
    ASTVisitor::visit(ast);
  }
};

}  // namespace

void ASTRewriter::requireDefinitionsNamedBy(TranslationUnit* unit, AST* ast) {
  if (!unit || !ast) return;
  if (!unit->requiresDefinitions()) return;
  RequireNamedDefinitions{unit}.accept(ast);
}

void ASTRewriter::requireFieldDefinition(TranslationUnit* unit,
                                         FieldSymbol* field) {
  if (!unit || !field || !field->isStatic()) return;
  if (!unit->requiresDefinitions()) return;
  requireFieldInitializer(unit, field);
  if (field->isDefinitionRequired()) return;
  field->setDefinitionRequired(true);

  auto enclosingClass = symbol_cast<ClassSymbol>(field->parent());
  if (!enclosingClass) return;
  unit->reopenMemberInstantiation(enclosingClass->resolvedDefinition());
}

void ASTRewriter::requireFieldInitializer(TranslationUnit* unit,
                                          FieldSymbol* field) {
  if (!unit || !field || !field->hasPendingInitializer()) return;

  auto pending = field->pendingInitializer();
  auto pattern = ast_cast<InitDeclaratorAST>(pending->pattern);
  auto instance = ast_cast<InitDeclaratorAST>(pending->instance);
  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  field->clearPendingInitializer();

  if (!pattern || !instance || !pattern->initializer) return;

  OutsideImmediateContextScope outsideImmediateContext{unit};

  auto rewriter = ASTRewriter{unit, parentScope, std::move(templateArguments)};
  rewriter.depth_ = depth;
  rewriter.inheritEnclosingTemplateArguments(field->parent());

  if (pattern->symbol) {
    auto patternClass = symbol_cast<ClassSymbol>(pattern->symbol->parent());
    auto instanceClass = symbol_cast<ClassSymbol>(field->parent());
    if (patternClass && instanceClass) {
      rewriter.remapScopeMembers(patternClass, instanceClass);
    }
    rewriter.addSymbolRemap(pattern->symbol, field);
  }

  instance->initializer = rewriter.instantiateSeparately(pattern->initializer,
                                                         field, "initializer");
  if (!instance->initializer) return;

  field->setInitializer(instance->initializer);

  if (field->isStatic()) {
    rewriter.typeChecker().check_init_declarator(instance, nullptr);
  } else {
    rewriter.typeChecker().check_field_initializer(field);
  }
}

auto ASTRewriter::patternDefaultArgument(TranslationUnit* unit,
                                         ParameterDeclarationAST* pattern)
    -> ExpressionAST* {
  if (!pattern) return nullptr;
  if (pattern->expression) return pattern->expression;
  return requireDefaultArgument(unit,
                                symbol_cast<ParameterSymbol>(pattern->symbol));
}

auto ASTRewriter::requireDefaultArgument(TranslationUnit* unit,
                                         ParameterSymbol* parameter)
    -> ExpressionAST* {
  if (!unit || !parameter) return nullptr;
  if (auto expression = parameter->defaultArgument()) return expression;

  if (auto source = parameter->defaultArgumentSource()) {
    parameter->setDefaultArgument(requireDefaultArgument(unit, source));
    return parameter->defaultArgument();
  }

  auto pending = parameter->pendingDefaultArgument();
  if (!pending) return nullptr;

  auto pattern = ast_cast<ParameterDeclarationAST>(pending->pattern);
  auto instance = ast_cast<ParameterDeclarationAST>(pending->instance);
  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  parameter->clearPendingDefaultArgument();

  auto defaultArgument = patternDefaultArgument(unit, pattern);
  if (!defaultArgument || !instance) return nullptr;

  OutsideImmediateContextScope outsideImmediateContext{unit};

  auto rewriter = ASTRewriter{unit, parentScope, std::move(templateArguments)};
  rewriter.depth_ = depth;
  rewriter.inheritEnclosingTemplateArguments(parameter->parent());

  if (pattern->symbol) {
    rewriter.remapScopeMembers(pattern->symbol->enclosingClass(),
                               parameter->enclosingClass());
  }

  instance->expression = rewriter.instantiateSeparately(
      defaultArgument, parameter, "default argument");
  if (!instance->expression) return nullptr;

  parameter->setDefaultArgument(instance->expression);
  return instance->expression;
}

auto ASTRewriter::instantiateSeparately(ExpressionAST* pattern, Symbol* owner,
                                        std::string_view construct)
    -> ExpressionAST* {
  auto diagnosticsClient = unit_->diagnosticsClient();
  const auto errorsBefore = diagnosticsClient->errorCount();

  ExpressionAST* instance = nullptr;
  {
    TranslationUnit::DeferredInitializerScope deferredInitializer{unit_, true};
    instance = expression(pattern);
  }

  if (instance) return instance;
  if (diagnosticsClient->errorCount() != errorsBefore) return nullptr;

  unit_->error(owner->location(),
               std::format("cannot instantiate the {} of '{}'", construct,
                           to_string(owner->name())));
  return nullptr;
}

void ASTRewriter::completeDeducedReturnType(TranslationUnit* unit,
                                            Symbol* symbol) {
  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function || !function->hasPendingBody()) return;

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return;
  if (!containsPlaceholderType(functionType->returnType())) return;

  (void)completePendingBodyFor(unit, function);
}

auto ASTRewriter::completedSymbolType(TranslationUnit* unit, Symbol* symbol)
    -> const Type* {
  if (!symbol) return nullptr;
  completeDeducedReturnType(unit, symbol);
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    completePendingExceptionSpecification(unit, function);
  return symbol->type();
}

auto ASTRewriter::completePendingBody(FunctionSymbol* func,
                                      bool captureBodyErrors)
    -> std::vector<Diagnostic> {
  if (!func || !func->hasPendingBody()) return {};

  auto pending = func->pendingBody();

  if (unit_->isFunctionBodyUnparsed(
          ast_cast<FunctionDefinitionAST>(pending->pattern))) {
    unit_->addPendingBodyCompletion(func);
    return {};
  }

  TimeTrace::Scope trace{unit_->timeTrace(), "Function body", func};
  if (auto trace = unit_->timeTrace()) trace->count(TimeTrace::kFunctionBodies);
  TranslationUnit::TemplateInstantiationScope instantiationScope{unit_};
  const auto errorsBefore = unit_->diagnosticsClient()->errorCount();

  const bool deferDiagnostics =
      !captureBodyErrors && unit_->diagnosticsClient()->isSfinae();

  std::optional<CapturingDiagnosticsScope> capture;
  if (captureBodyErrors || deferDiagnostics) capture.emplace(unit_);

  auto finish =
      [&](std::vector<Diagnostic> bodyErrors = {}) -> std::vector<Diagnostic> {
    if (capture.has_value()) {
      capture->finish();
      auto captured = capture->takeDiagnostics();
      bodyErrors.insert(bodyErrors.end(),
                        std::make_move_iterator(captured.begin()),
                        std::make_move_iterator(captured.end()));
    }

    if (captureBodyErrors) return bodyErrors;

    const auto reported = reportOutsideImmediateContext(unit_, bodyErrors);
    if (!func->primaryTemplateSymbol() &&
        (reported || unit_->diagnosticsClient()->errorCount() != errorsBefore))
      notePendingBodyInstantiation(unit_, func);
    return {};
  };

  auto newAst = func->declaration();
  if (!newAst) {
    auto originalDef = ast_cast<FunctionDefinitionAST>(pending->pattern);
    auto classArguments = std::move(pending->templateArguments);
    auto parentScope = pending->parentScope;
    auto depth = pending->depth;
    func->clearPendingBody();

    if (!originalDef || !originalDef->symbol) return finish();

    auto rewriter = ASTRewriter{unit_, parentScope, std::move(classArguments)};
    rewriter.depth_ = depth;
    rewriter.inheritEnclosingTemplateArguments(func->parent());
    rewriter.binder_.setInstantiatingSymbol(originalDef->symbol);

    auto patternClass = symbol_cast<ClassSymbol>(originalDef->symbol->parent());
    auto instanceClass = symbol_cast<ClassSymbol>(func->parent());
    if (patternClass && instanceClass) {
      rewriter.remapScopeMembers(patternClass, instanceClass);
    }

    auto patternTemplateDecl = originalDef->symbol->templateDeclaration();
    rewriter.setInstantiatingFunctionTemplateSpecialization(
        func->isSpecialization());
    rewriter.functionInstanceToDefine_ = func;
    auto rewrittenDecl = patternTemplateDecl
                             ? rewriter.declaration(patternTemplateDecl)
                             : rewriter.declaration(originalDef);

    auto copy = ast_cast<FunctionDefinitionAST>(rewrittenDecl);
    if (!copy) {
      if (auto rewrittenTemplateDecl =
              ast_cast<TemplateDeclarationAST>(rewrittenDecl)) {
        copy =
            ast_cast<FunctionDefinitionAST>(rewrittenTemplateDecl->declaration);
      }
    }

    if (!func->declaration() && copy) func->setDeclaration(copy);

    const auto instantiatesATemplate =
        TemplateEquivalence{unit_}.ownFunctionTemplateHead(
            symbol_cast<ClassSymbol>(originalDef->symbol->parent()),
            patternTemplateDecl);

    if (instantiatesATemplate) return finish();

    return finish(rewriter.takeBodyErrors());
  }

  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  auto originalDef = ast_cast<FunctionDefinitionAST>(pending->pattern);
  func->clearPendingBody();
  if (newAst->symbol) func = newAst->symbol;

  auto rewriter = ASTRewriter{unit_, parentScope, templateArguments};
  rewriter.depth_ = depth;
  rewriter.inheritEnclosingTemplateArguments(func->parent());
  rewriter.binder_.setInstantiatingSymbol(func);

  if (auto oldFunc = symbol_cast<FunctionSymbol>(originalDef->symbol)) {
    auto oldClass = symbol_cast<ClassSymbol>(oldFunc->parent());
    auto newClass = symbol_cast<ClassSymbol>(func->parent());

    std::vector<std::pair<ClassSymbol*, ClassSymbol*>> enclosingClasses;
    while (oldClass && newClass && oldClass != newClass) {
      enclosingClasses.emplace_back(oldClass, newClass);
      oldClass = symbol_cast<ClassSymbol>(oldClass->parent());
      newClass = symbol_cast<ClassSymbol>(newClass->parent());
    }
    std::ranges::reverse(enclosingClasses);
    for (auto [patternClass, instanceClass] : enclosingClasses)
      rewriter.remapScopeMembers(patternClass, instanceClass);

    if (auto oldParams = oldFunc->functionParameters()) {
      if (auto newParams = func->functionParameters()) {
        rewriter.remapFunctionParameters(oldParams, newParams);
      }
    }
  }

  auto functionDeclarator = getFunctionPrototype(newAst->declarator);
  if (!functionDeclarator) {
    rewriter.binder_.setScope(func);
  } else if (auto params = functionDeclarator->parameterDeclarationClause) {
    rewriter.binder_.setScope(params->functionParametersSymbol);
  } else {
    rewriter.binder_.setScope(func);
  }

  newAst->functionBody = rewriter.functionBody(originalDef->functionBody);

  auto bodyErrors = rewriter.takeBodyErrors();

  rewriter.binder_.synthesizeCompleteObjectCtor(func);

  if (ast_cast<DefaultFunctionBodyAST>(newAst->functionBody)) {
    rewriter.binder_.synthesizeDefaultedMemberBody(func);
  } else if (body_statement(newAst->functionBody)) {
    rewriter.checkMemInitializers(func, newAst->functionBody);
    rewriter.binder_.finishAutoReturnType(func);
  }

  if (bodyErrors.empty() && !deferDiagnostics) {
    if (!rewriter.binder_.inTemplate())
      checkReturnPathWarnings(unit_, func, newAst->functionBody);

    validateCompletedInstantiation(unit_, func, newAst);
  }

  return finish(std::move(bodyErrors));
}
}  // namespace cxx
