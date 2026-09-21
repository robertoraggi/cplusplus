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
#include <cxx/function_body_warnings.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
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
auto memberFunctionKey(FunctionSymbol* fn) -> std::pair<bool, std::size_t> {
  bool isConst = false;
  std::size_t arity = 0;
  if (auto ft = type_cast<FunctionType>(fn->type())) {
    isConst = has_const(ft->cvQualifiers());
    arity = ft->parameterTypes().size();
  }
  return {isConst, arity};
}

void collectFunctions(Symbol* member, std::vector<FunctionSymbol*>& out) {
  std::ranges::copy(views::each_function(member), std::back_inserter(out));
}

[[nodiscard]] auto packParameterFlags(FunctionDeclaratorChunkAST* prototype,
                                      std::size_t parameterCount)
    -> std::vector<bool> {
  std::vector<bool> flags(parameterCount, false);
  if (!prototype || !prototype->parameterDeclarationClause) return flags;

  std::size_t i = 0;
  for (auto node : ListView{
           prototype->parameterDeclarationClause->parameterDeclarationList}) {
    if (i == parameterCount) break;
    auto paramDecl = ast_cast<ParameterDeclarationAST>(node);
    flags[i++] = paramDecl && paramDecl->isPack;
  }
  return flags;
}

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
  auto& oldMembers = oldScope->members();
  auto& newMembers = newScope->members();

  std::unordered_map<const Name*, std::vector<Symbol*>> newByName;
  for (auto newMember : newMembers) {
    newByName[newMember->name()].push_back(newMember);
  }

  std::unordered_map<const Name*, std::size_t> nextIndex;
  for (auto oldMember : oldMembers) {
    auto it = newByName.find(oldMember->name());
    if (it == newByName.end()) continue;
    auto& candidates = it->second;
    auto& index = nextIndex[oldMember->name()];
    if (index >= candidates.size()) continue;
    auto newMember = candidates[index++];
    addSymbolRemap(oldMember, newMember);

    if (auto oldMemberClass = symbol_cast<ClassSymbol>(oldMember)) {
      if (auto newMemberClass = symbol_cast<ClassSymbol>(newMember)) {
        if (!newMemberClass->instantiationPattern())
          newMemberClass->setInstantiationPattern(oldMemberClass);
      }
    }
    if (symbol_cast<OverloadSetSymbol>(oldMember) ||
        symbol_cast<OverloadSetSymbol>(newMember)) {
      std::vector<FunctionSymbol*> oldFns, newFns;
      collectFunctions(oldMember, oldFns);
      collectFunctions(newMember, newFns);
      std::vector<bool> used(newFns.size(), false);
      for (auto oldFn : oldFns) {
        auto key = memberFunctionKey(oldFn);
        FunctionSymbol* fallback = nullptr;
        FunctionSymbol* chosen = nullptr;
        for (std::size_t i = 0; i < newFns.size(); ++i) {
          if (memberFunctionKey(newFns[i]) != key) continue;
          if (!fallback) fallback = newFns[i];
          if (!used[i]) {
            used[i] = true;
            chosen = newFns[i];
            break;
          }
        }
        if (!chosen) chosen = fallback;
        if (chosen) addSymbolRemap(oldFn, chosen);
      }
    }
    if (auto oldUsing = symbol_cast<UsingDeclarationSymbol>(oldMember)) {
      if (auto newUsing = symbol_cast<UsingDeclarationSymbol>(newMember);
          newUsing && oldUsing->target() && newUsing->target()) {
        addSymbolRemap(oldUsing->target(), newUsing->target());
      }
    }
    if (auto oldNested = symbol_cast<ClassSymbol>(oldMember)) {
      if (auto newNested = symbol_cast<ClassSymbol>(newMember)) {
        remapScopeMembers(oldNested, newNested);
      }
    } else if (auto oldEnum = symbol_cast<EnumSymbol>(oldMember)) {
      if (auto newEnum = symbol_cast<EnumSymbol>(newMember)) {
        remapScopeMembers(oldEnum, newEnum);
      }
    } else if (auto oldScopedEnum = symbol_cast<ScopedEnumSymbol>(oldMember)) {
      if (auto newScopedEnum = symbol_cast<ScopedEnumSymbol>(newMember)) {
        remapScopeMembers(oldScopedEnum, newScopedEnum);
      }
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

void ASTRewriter::remapFunctionParameters(
    FunctionDeclaratorChunkAST* patternPrototype,
    FunctionDeclaratorChunkAST* instancePrototype,
    FunctionParametersSymbol* patternParameters,
    FunctionParametersSymbol* instanceParameters) {
  const auto patternMembers = parametersOf(patternParameters);
  const auto instanceMembers = parametersOf(instanceParameters);

  const auto patternIsPack =
      packParameterFlags(patternPrototype, patternMembers.size());
  const auto instanceIsPack =
      packParameterFlags(instancePrototype, instanceMembers.size());

  std::size_t instanceIndex = 0;

  for (std::size_t i = 0; i < patternMembers.size(); ++i) {
    std::size_t reservedForTrailingParameters = 0;
    for (auto j = i + 1; j < patternMembers.size(); ++j)
      if (!patternIsPack[j]) ++reservedForTrailingParameters;

    const auto available = instanceMembers.size() - instanceIndex;

    const auto stillPacked =
        instanceIndex < instanceMembers.size() && instanceIsPack[instanceIndex];

    if (!patternIsPack[i] || stillPacked) {
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
                                       CompoundStatementFunctionBodyAST* body) {
  std::optional<CapturingDiagnosticsScope> capture;
  if (unit_->diagnosticsClient()->isSfinae()) capture.emplace(unit_);

  TypeChecker check{unit_};
  check.setScope(function);
  check.setReportErrors(unit_->config().checkTypes);
  auto hasDependentInitializer = [&] {
    for (auto memInit : ListView{body->memInitializerList}) {
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

  if (!capture.has_value()) return;

  capture->finish();
  reportOutsideImmediateContext(unit_, capture->diagnostics());
}

auto ASTRewriter::completePendingBodyFor(TranslationUnit* unit,
                                         FunctionSymbol* function,
                                         bool captureBodyErrors)
    -> std::vector<Diagnostic> {
  if (!unit || !function || !function->hasPendingBody()) return {};
  auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
  return rewriter.completePendingBody(function, captureBodyErrors);
}

void ASTRewriter::requirePotentiallyInvokedDestructors(
    TranslationUnit* unit, FunctionSymbol* destructor) {
  if (!unit || !destructor || !destructor->isDestructor()) return;
  if (!unit->isPotentiallyEvaluated()) return;

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
  if (!unit->isPotentiallyEvaluated()) return;

  TypeTraits traits{unit};
  auto objectType = traits.remove_cv(traits.remove_all_extents(type));
  auto classType = unqualified_cast<ClassType>(objectType);
  if (!classType || !classType->symbol()) return;

  if (!ensureCompleteClass(unit, classType->symbol())) return;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  requireFunctionDefinition(unit, classSymbol->destructor());
}

void ASTRewriter::requireSubobjectDefaultConstructors(
    TranslationUnit* unit, FunctionSymbol* constructor) {
  auto classSymbol = symbol_cast<ClassSymbol>(constructor->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (classSymbol->isUnion()) return;

  TypeTraits traits{unit};

  auto requireDefaultConstructorOf = [&](const Type* type) {
    if (!type) return;
    auto subobjectType = traits.remove_cv(traits.remove_all_extents(type));
    auto classType = unqualified_cast<ClassType>(subobjectType);
    if (!classType || !classType->symbol()) return;
    auto subobjectClass = classType->symbol()->resolvedDefinition();
    if (subobjectClass == classSymbol) return;
    OverloadResolution overloadResolution{unit};
    requireFunctionDefinition(
        unit,
        overloadResolution.resolveConstructor(subobjectClass, {}).selected());
  };

  for (auto baseClass : classSymbol->baseClasses()) {
    if (auto base = baseClass->symbol())
      requireDefaultConstructorOf(base->type());
  }

  if (auto layout = classSymbol->layout()) {
    for (auto virtualBase : layout->virtualBases())
      requireDefaultConstructorOf(virtualBase->type());
  }

  for (auto field : classSymbol->members() | views::non_static_fields) {
    if (field->initializer()) continue;
    requireDefaultConstructorOf(field->type());
  }
}

void ASTRewriter::requireFunctionDefinition(TranslationUnit* unit,
                                            FunctionSymbol* function) {
  if (!unit || !function) return;
  if (!unit->isPotentiallyEvaluated()) return;
  const auto alreadyRequired = function->isDefinitionRequired();
  function->setDefinitionRequired(true);
  unit->addPendingBodyCompletion(function);
  if (alreadyRequired) return;
  if (!function->hasPendingBody()) {
    auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
    rewriter.binder_.synthesizeDefaultedMemberBody(function);
  }
  requireFunctionDefinition(unit, function->inheritedConstructor());
  auto definition = function->resolvedDefinition();
  const bool definesBody =
      definition->isDefined() || definition->hasPendingBody();

  if (function->isDestructor() && definesBody)
    requirePotentiallyInvokedDestructors(unit, definition);

  if (function->isConstructor() && function->isDefaulted()) {
    auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
    if (classSymbol &&
        classSymbol->resolvedDefinition()->defaultConstructor() == function) {
      requireSubobjectDefaultConstructors(unit, function);
    }
  }
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
  if (!unit->isPotentiallyEvaluated()) return;
  RequireNamedDefinitions{unit}.accept(ast);
}

void ASTRewriter::requireFieldDefinition(TranslationUnit* unit,
                                         FieldSymbol* field) {
  if (!unit || !field || !field->isStatic()) return;
  if (!unit->isPotentiallyEvaluated()) return;
  completePendingFieldInitializer(unit, field);
  if (field->isDefinitionRequired()) return;
  field->setDefinitionRequired(true);

  auto enclosingClass = symbol_cast<ClassSymbol>(field->parent());
  if (!enclosingClass) return;
  unit->reopenMemberInstantiation(enclosingClass->resolvedDefinition());
}

void ASTRewriter::completePendingFieldInitializer(TranslationUnit* unit,
                                                  FieldSymbol* field) {
  if (!unit || !field || !field->hasPendingInitializer()) return;

  auto pending = field->pendingInitializer();
  auto pattern = pending->pattern;
  auto instance = pending->instance;
  auto typeSpecifier = pending->typeSpecifier;
  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  field->clearPendingInitializer();

  if (!pattern || !instance || !pattern->initializer) return;

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

  auto diagnosticsClient = unit->diagnosticsClient();
  const auto errorsBefore = diagnosticsClient->errorCount();

  instance->initializer = rewriter.expression(pattern->initializer);

  if (!instance->initializer) {
    if (diagnosticsClient->errorCount() == errorsBefore) {
      unit->error(field->location(),
                  std::format("cannot instantiate the initializer of '{}'",
                              to_string(field->name())));
    }
    return;
  }

  field->setInitializer(instance->initializer);

  if (field->isStatic()) {
    rewriter.typeChecker().check_init_declarator(instance, typeSpecifier);
  } else {
    rewriter.typeChecker().check_field_initializer(field);
  }
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

  if (unit_->isFunctionBodyUnparsed(pending->originalDefinition)) {
    unit_->addPendingBodyCompletion(func);
    return {};
  }

  if (auto trace = unit_->timeTrace()) trace->count(TimeTrace::kFunctionBodies);
  TranslationUnit::TemplateInstantiationScope instantiationScope{unit_};

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

    reportOutsideImmediateContext(unit_, bodyErrors);
    return {};
  };

  auto newAst = func->declaration();
  if (!newAst) {
    auto originalDef = pending->originalDefinition;
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
  auto originalDef = pending->originalDefinition;
  func->clearPendingBody();
  if (newAst->symbol) func = newAst->symbol;

  auto rewriter = ASTRewriter{unit_, parentScope, templateArguments};
  rewriter.depth_ = depth;
  rewriter.inheritEnclosingTemplateArguments(func->parent());
  rewriter.binder_.setInstantiatingSymbol(func);

  if (auto oldFunc = symbol_cast<FunctionSymbol>(originalDef->symbol)) {
    auto oldClass = symbol_cast<ClassSymbol>(oldFunc->parent());
    auto newClass = symbol_cast<ClassSymbol>(func->parent());

    while (oldClass && newClass && oldClass != newClass) {
      rewriter.remapScopeMembers(oldClass, newClass);
      oldClass = symbol_cast<ClassSymbol>(oldClass->parent());
      newClass = symbol_cast<ClassSymbol>(newClass->parent());
    }

    if (auto oldParams = oldFunc->functionParameters()) {
      if (auto newParams = func->functionParameters()) {
        rewriter.remapFunctionParameters(
            getFunctionPrototype(originalDef->declarator),
            getFunctionPrototype(newAst->declarator), oldParams, newParams);
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
  } else if (auto compoundBody = ast_cast<CompoundStatementFunctionBodyAST>(
                 newAst->functionBody)) {
    rewriter.checkMemInitializers(func, compoundBody);
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
