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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/class_template_deduction.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
namespace {
[[nodiscard]] auto initializerCompletesDeclaredType(FieldSymbol* field,
                                                    SpecifierAST* typeSpecifier,
                                                    ScopeSymbol* scope)
    -> bool {
  auto type = field->type();
  if (containsPlaceholderType(type)) return true;
  if (type_cast<UnboundedArrayType>(type)) return true;
  return ClassTemplateArgumentDeduction::placeholderClassTemplate(
             typeSpecifier, scope) != nullptr;
}

[[nodiscard]] auto redeclaresStaticDataMemberTemplate(VariableSymbol* variable)
    -> bool {
  return variable->canonical() != variable;
}

[[nodiscard]] auto instantiatesVariableTemplate(VariableSymbol* pattern,
                                                const Decl& decl) -> bool {
  auto declaredScope = decl.getScope();
  if (!declaredScope || !declaredScope->isClass()) return true;
  return redeclaresStaticDataMemberTemplate(pattern);
}

void applyConstexprConstness(const TypeTraits& traits, FieldSymbol* field) {
  if (!field->isConstexpr()) return;
  field->setType(traits.add_const(field->type()));
}
}  // namespace

struct ASTRewriter::CoreDeclaratorVisitor : VisitorBase {
  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast) -> CoreDeclaratorAST*;
};

struct ASTRewriter::DeclaratorChunkVisitor : VisitorBase {
  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;
};

struct ASTRewriter::PtrOperatorVisitor : VisitorBase {
  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast) -> PtrOperatorAST*;
};

struct ASTRewriter::DesignatorVisitor : VisitorBase {
  [[nodiscard]] auto operator()(DotDesignatorAST* ast) -> DesignatorAST*;

  [[nodiscard]] auto operator()(SubscriptDesignatorAST* ast) -> DesignatorAST*;
};

struct ASTRewriter::ExceptionSpecifierVisitor : VisitorBase {
  [[nodiscard]] auto operator()(ThrowExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;

  [[nodiscard]] auto operator()(NoexceptSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
};

auto ASTRewriter::ptrOperator(PtrOperatorAST* ast) -> PtrOperatorAST* {
  if (!ast) return {};
  return visit(PtrOperatorVisitor{*this}, ast);
}

auto ASTRewriter::coreDeclarator(CoreDeclaratorAST* ast) -> CoreDeclaratorAST* {
  if (!ast) return {};
  return visit(CoreDeclaratorVisitor{*this}, ast);
}

auto ASTRewriter::declaratorChunk(DeclaratorChunkAST* ast)
    -> DeclaratorChunkAST* {
  if (!ast) return {};
  return visit(DeclaratorChunkVisitor{*this}, ast);
}

auto ASTRewriter::designator(DesignatorAST* ast) -> DesignatorAST* {
  if (!ast) return {};
  return visit(DesignatorVisitor{*this}, ast);
}

auto ASTRewriter::exceptionSpecifier(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierAST* {
  if (!ast) return {};
  return visit(ExceptionSpecifierVisitor{*this}, ast);
}

auto ASTRewriter::pendingExceptionSpecifierMark() const -> std::size_t {
  return pendingExceptionSpecifiers_.size();
}

void ASTRewriter::associatePendingExceptionSpecifiers(
    std::size_t mark, FunctionSymbol* function,
    ExceptionSpecifierAST* functionExceptionSpecifier,
    std::function<void()> refreshType) {
  for (auto index = mark; index < pendingExceptionSpecifiers_.size(); ++index) {
    auto& pending = pendingExceptionSpecifiers_[index];
    pending.typeRefreshers.push_back(refreshType);
    if (function && pending.record->instance == functionExceptionSpecifier)
      pendingFunctionExceptionSpecifiers_[function] = index;
  }
}

void ASTRewriter::resolvePendingExceptionSpecifier(std::size_t index) {
  auto& pending = pendingExceptionSpecifiers_[index];
  auto record = pending.record.get();
  if (!record) return;

  auto pattern = ast_cast<NoexceptSpecifierAST>(record->pattern);
  auto instance = ast_cast<NoexceptSpecifierAST>(record->instance);

  if (record->state == PendingInstantiationState::kResolving) {
    error(instance->noexceptLoc,
          "recursive exception specification instantiation");
    record->state = PendingInstantiationState::kRecursionDiagnosed;
    return;
  }

  if (record->state != PendingInstantiationState::kUnresolved) return;

  record->state = PendingInstantiationState::kResolving;

  auto _ = Binder::ScopeGuard{&binder_};
  binder_.setScope(record->parentScope);
  instance->expression = expression(pattern->expression);

  for (auto& refreshType : pending.typeRefreshers) refreshType();

  record->state = PendingInstantiationState::kResolved;
}

void ASTRewriter::completePendingExceptionSpecifiers(std::size_t mark) {
  for (const auto& [function, index] : pendingFunctionExceptionSpecifiers_) {
    if (index < mark) continue;

    auto& pending = pendingExceptionSpecifiers_[index];
    if (pending.record->state != PendingInstantiationState::kUnresolved)
      continue;

    function->setPendingExceptionSpecification(std::move(pending.record));
  }

  for (auto index = mark; index < pendingExceptionSpecifiers_.size(); ++index) {
    resolvePendingExceptionSpecifier(index);
  }
}

auto ASTRewriter::hasPendingExceptionSpecifier(ClassSymbol* classSymbol) const
    -> bool {
  for (const auto& [function, index] : pendingFunctionExceptionSpecifiers_) {
    if (function->parent() != classSymbol) continue;
    auto record = pendingExceptionSpecifiers_[index].record.get();
    if (!record) continue;
    if (record->state == PendingInstantiationState::kUnresolved ||
        record->state == PendingInstantiationState::kResolving)
      return true;
  }
  return false;
}

void ASTRewriter::completePendingExceptionSpecification(
    TranslationUnit* unit, FunctionSymbol* function) {
  if (!function) return;
  Binder{unit}.completeDeferredImplicitExceptionSpecification(function);
  auto pending = function->pendingExceptionSpecification();
  if (!pending) return;

  auto pattern = ast_cast<NoexceptSpecifierAST>(pending->pattern);
  auto instance = ast_cast<NoexceptSpecifierAST>(pending->instance);

  if (pending->state == PendingInstantiationState::kResolving) {
    unit->error(instance->noexceptLoc,
                "recursive exception specification instantiation");
    pending->state = PendingInstantiationState::kRecursionDiagnosed;
    return;
  }

  if (pending->state != PendingInstantiationState::kUnresolved) return;

  pending->state = PendingInstantiationState::kResolving;

  auto patternFunction =
      symbol_cast<FunctionSymbol>(function->instantiationPattern());

  auto rewriter =
      ASTRewriter{unit, pending->parentScope, pending->templateArguments};
  rewriter.depth_ = pending->depth;
  rewriter.inheritEnclosingTemplateArguments(pending->parentScope);
  rewriter.binder_.setInstantiatingSymbol(function);

  auto oldClass = symbol_cast<ClassSymbol>(patternFunction->parent());
  auto newClass = symbol_cast<ClassSymbol>(function->parent());

  while (oldClass && newClass) {
    auto oldParent = symbol_cast<ClassSymbol>(oldClass->parent());
    auto newParent = symbol_cast<ClassSymbol>(newClass->parent());
    if (!oldParent || !newParent) break;
    oldClass = oldParent;
    newClass = newParent;
  }

  if (oldClass && newClass && oldClass != newClass)
    rewriter.remapScopeMembers(oldClass, newClass);

  auto oldParameters = patternFunction->functionParameters();
  auto newParameters = function->functionParameters();
  if (oldParameters && newParameters)
    rewriter.remapFunctionParameters(oldParameters, newParameters);

  instance->expression = rewriter.expression(pattern->expression);
  const bool isNoexcept = exceptionSpecifierIsNoexcept(unit, instance);
  setFunctionNoexcept(unit->control(), function, isNoexcept);

  pending->state = PendingInstantiationState::kResolved;
}

auto ASTRewriter::requiresClause(RequiresClauseAST* ast) -> RequiresClauseAST* {
  if (!ast) return {};

  auto copy = RequiresClauseAST::create(arena());

  copy->requiresLoc = ast->requiresLoc;

  copy->expression = ast->expression;

  return copy;
}

auto ASTRewriter::parameterDeclarationClause(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseAST* {
  if (!ast) return {};

  auto copy = ParameterDeclarationClauseAST::create(arena());

  binder().bind(copy);

  if (ast->functionParametersSymbol)
    copy->functionParametersSymbol->setCvQualifiers(
        ast->functionParametersSymbol->cvQualifiers());

  auto _ = Binder::ScopeGuard(&binder_);

  binder().setScope(copy->functionParametersSymbol);

  ListAppender<ParameterDeclarationAST> append{arena(),
                                               copy->parameterDeclarationList};

  for (auto node : ListView{ast->parameterDeclarationList}) {
    if (auto pack = expandedFunctionParameterPack(node)) {
      auto elements =
          control()->newParameterPackSymbol(binder().scope(), SourceLocation{});
      forEachPackElement(
          node, node->firstSourceLocation(),
          [&] {
            auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
            if (!value) return;
            append(value);
            if (value->symbol) elements->addElement(value->symbol);
          },
          pack);
      if (node->symbol) functionParamPacks_[node->symbol] = elements;
      continue;
    }

    auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
    append(value);
    if (value) addSymbolRemap(node->symbol, value->symbol);
  }

  copy->commaLoc = ast->commaLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->isVariadic = ast->isVariadic;

  return copy;
}

auto ASTRewriter::trailingReturnType(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeAST* {
  if (!ast) return {};

  auto copy = TrailingReturnTypeAST::create(arena());

  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeId = typeId(ast->typeId);

  return copy;
}

auto ASTRewriter::initDeclarator(InitDeclaratorAST* ast,
                                 const DeclSpecs& declSpecs)
    -> InitDeclaratorAST* {
  if (!ast) return {};

  auto copy = InitDeclaratorAST::create(arena());

  auto patternFunction = symbol_cast<FunctionSymbol>(ast->symbol);
  auto functionTemplateHead = rewriteMemberTemplateHead(patternFunction);

  const auto pendingExceptionSpecifierMark =
      this->pendingExceptionSpecifierMark();
  copy->declarator = declarator(ast->declarator);

  auto decl = Decl{declSpecs, copy->declarator};
  if (functionTemplateHead) {
    decl.specs.templateHead = functionTemplateHead;
  } else if (!decl.specs.templateHead && currentTemplateHead_) {
    decl.specs.templateHead = currentTemplateHead_;
  }

  copy->requiresClause = requiresClause(ast->requiresClause);
  decl.trailingRequiresClause = copy->requiresClause;

  auto type =
      getDeclaratorType(translationUnit(), copy->declarator, declSpecs.type());

  const auto addSymbolToParentScope =
      binder().instantiatingSymbol() != ast->symbol;

  const auto declaresClassMember =
      binder_.scope()->isClass() && !symbol_cast<VariableSymbol>(ast->symbol);

  if (declaresClassMember) {
    auto symbol = binder_.declareMemberSymbol(copy->declarator, decl,
                                              addSymbolToParentScope);
    copy->symbol = symbol;

    if (auto newField = symbol_cast<FieldSymbol>(symbol)) {
      if (auto oldField = symbol_cast<FieldSymbol>(ast->symbol);
          oldField && oldField->isNoUniqueAddress()) {
        newField->setNoUniqueAddress(true);
      }
    }
  } else {
    if (auto declId = decl.declaratorId; declId) {
      if (decl.specs.isTypedef) {
        auto typedefSymbol = binder_.declareTypedef(copy->declarator, decl);
        copy->symbol = typedefSymbol;
      } else if (getFunctionPrototype(copy->declarator)) {
        auto functionSymbol = binder_.declareFunction(copy->declarator, decl,
                                                      addSymbolToParentScope);
        if (auto templateHead = decl.specs.templateHead)
          binder_.setTemplateHead(functionSymbol, templateHead);
        copy->symbol = functionSymbol;
      } else {
        auto variableSymbol = binder_.declareVariable(copy->declarator, decl,
                                                      addSymbolToParentScope);
        binder_.declareVariableTemplate(variableSymbol, decl.declaratorId,
                                        decl.specs.templateHead);
        copy->symbol = variableSymbol;

        auto templateVariable = symbol_cast<VariableSymbol>(ast->symbol);
        if (!addSymbolToParentScope &&
            instantiatesVariableTemplate(templateVariable, decl)) {
          templateVariable->canonical()->addSpecialization(
              unit_, templateArguments(), variableSymbol);
        }
      }
    }
  }

  auto function = symbol_cast<FunctionSymbol>(copy->symbol);
  if (function && functionTemplateHead)
    binder_.setTemplateHead(function, functionTemplateHead);
  auto functionExceptionSpecifier =
      static_cast<ExceptionSpecifierAST*>(nullptr);
  if (auto prototype = getFunctionPrototype(copy->declarator))
    functionExceptionSpecifier = prototype->exceptionSpecifier;

  associatePendingExceptionSpecifiers(
      pendingExceptionSpecifierMark, function, functionExceptionSpecifier,
      [this, copy, baseType = declSpecs.type()] {
        auto type = getDeclaratorType(unit_, copy->declarator, baseType);
        if (copy->symbol) copy->symbol->setType(type);
      });

  if (auto fieldSymbol = symbol_cast<FieldSymbol>(copy->symbol);
      fieldSymbol && classBodyDepth_ > 0) {
    if (!fieldSymbol->isStatic()) {
      addSymbolRemap(ast->symbol, copy->symbol);

      if (ast->initializer) {
        if (initializerCompletesDeclaredType(
                fieldSymbol, declSpecs.typeSpecifier(), binder_.scope())) {
          pendingFieldInitializers_.push_back({ast, copy, binder_.scope()});
        } else {
          fieldSymbol->setPendingInitializer(
              pendingInstantiationOf(ast, copy, binder_.scope()));
        }
      }

      return copy;
    }

    const auto canDeferInitializer =
        ast->initializer &&
        !initializerCompletesDeclaredType(
            fieldSymbol, declSpecs.typeSpecifier(), binder_.scope()) &&
        !isEnclosedInDependentTemplate(unit_, binder_.scope(),
                                       /*stopAtConcreteSpecialization=*/true);

    if (canDeferInitializer) {
      addSymbolRemap(ast->symbol, copy->symbol);
      applyConstexprConstness(unit_->typeTraits(), fieldSymbol);

      fieldSymbol->setPendingInitializer(
          pendingInstantiationOf(ast, copy, binder_.scope()));
      return copy;
    }
  }

  copy->initializer = expression(ast->initializer);

  addSymbolRemap(ast->symbol, copy->symbol);

  if (auto fieldSymbol = symbol_cast<FieldSymbol>(copy->symbol)) {
    if (copy->initializer) {
      fieldSymbol->setInitializer(copy->initializer);

      if (fieldSymbol->isStatic())
        typeChecker().check_init_declarator(copy, declSpecs.typeSpecifier());
      else
        typeChecker().check_field_initializer(fieldSymbol);
    }
  } else if (auto variableSymbol = symbol_cast<VariableSymbol>(copy->symbol)) {
    if (!rewritingForRangeDeclaration_)
      typeChecker().check_init_declarator(copy, declSpecs.typeSpecifier());
    binder_.recordVariableDefinition(variableSymbol);
  }

  return copy;
}

void ASTRewriter::completePendingFieldInitializers(std::size_t mark) {
  if (pendingFieldInitializers_.size() <= mark) return;

  std::vector<PendingFieldInitializer> pending{
      pendingFieldInitializers_.begin() + mark,
      pendingFieldInitializers_.end()};

  pendingFieldInitializers_.resize(mark);

  auto savedScope = binder_.scope();

  for (const auto& entry : pending) {
    binder_.setScope(entry.scope);
    entry.instance->initializer = expression(entry.pattern->initializer);

    auto fieldSymbol = symbol_cast<FieldSymbol>(entry.instance->symbol);
    if (!fieldSymbol || !entry.instance->initializer) continue;

    fieldSymbol->setInitializer(entry.instance->initializer);

    auto typeChecker = TypeChecker{unit_};
    typeChecker.setScope(entry.scope);
    typeChecker.check_field_initializer(fieldSymbol);
  }

  binder_.setScope(savedScope);
}

auto ASTRewriter::declarator(DeclaratorAST* ast) -> DeclaratorAST* {
  if (!ast) return {};

  auto copy = DeclaratorAST::create(arena());

  copy->ptrOpList = rewriteList(ast->ptrOpList, &ASTRewriter::ptrOperator);

  copy->coreDeclarator = coreDeclarator(ast->coreDeclarator);

  copy->declaratorChunkList =
      rewriteList(ast->declaratorChunkList, &ASTRewriter::declaratorChunk);

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = PointerOperatorAST::create(arena());

  copy->starLoc = ast->starLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->cvQualifierList =
      rewrite.rewriteList(ast->cvQualifierList, &ASTRewriter::specifier);

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = ReferenceOperatorAST::create(arena());

  copy->refLoc = ast->refLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->refOp = ast->refOp;

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = PtrToMemberOperatorAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->starLoc = ast->starLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->cvQualifierList =
      rewrite.rewriteList(ast->cvQualifierList, &ASTRewriter::specifier);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = BitfieldDeclaratorAST::create(arena());

  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);
  copy->colonLoc = ast->colonLoc;
  copy->sizeExpression = rewrite.expression(ast->sizeExpression);
  copy->trailingAttributeList = rewrite.rewriteList(
      ast->trailingAttributeList, &ASTRewriter::attributeSpecifier);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = ParameterPackAST::create(arena());

  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->coreDeclarator = rewrite.coreDeclarator(ast->coreDeclarator);

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = IdDeclaratorAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = NestedDeclaratorAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->declarator = rewrite.declarator(ast->declarator);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = FunctionDeclaratorChunkAST::create(arena());

  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;

  auto _ = Binder::ScopeGuard{binder()};

  if (copy->parameterDeclarationClause) {
    binder()->setScope(
        copy->parameterDeclarationClause->functionParametersSymbol);
  }

  copy->cvQualifierList =
      rewrite.rewriteList(ast->cvQualifierList, &ASTRewriter::specifier);

  copy->refLoc = ast->refLoc;
  copy->exceptionSpecifier =
      rewrite.exceptionSpecifier(ast->exceptionSpecifier);
  copy->refOp = ast->refOp;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->trailingReturnType =
      rewrite.trailingReturnType(ast->trailingReturnType);
  copy->isFinal = ast->isFinal;
  copy->isOverride = ast->isOverride;
  copy->isPure = ast->isPure;

  return copy;
}

auto ASTRewriter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkAST* {
  auto copy = ArrayDeclaratorChunkAST::create(arena());

  copy->lbracketLoc = ast->lbracketLoc;

  copy->typeQualifierList =
      rewrite.rewriteList(ast->typeQualifierList, &ASTRewriter::specifier);

  copy->expression = rewrite.expression(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  return copy;
}

auto ASTRewriter::DesignatorVisitor::operator()(DotDesignatorAST* ast)
    -> DesignatorAST* {
  auto copy = DotDesignatorAST::create(arena());

  copy->dotLoc = ast->dotLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DesignatorVisitor::operator()(SubscriptDesignatorAST* ast)
    -> DesignatorAST* {
  auto copy = SubscriptDesignatorAST::create(arena());

  copy->lbracketLoc = ast->lbracketLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = ThrowExceptionSpecifierAST::create(arena());

  copy->throwLoc = ast->throwLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

auto ASTRewriter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierAST* {
  auto copy = NoexceptSpecifierAST::create(arena());

  copy->noexceptLoc = ast->noexceptLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->rparenLoc = ast->rparenLoc;

  if (ast->expression && rewrite.classBodyDepth_ > 0 &&
      rewrite.restrictedToDeclarations_) {
    rewrite.pendingExceptionSpecifiers_.push_back(
        {rewrite.pendingInstantiationOf(ast, copy, binder()->scope())});
  } else {
    copy->expression = rewrite.expression(ast->expression);
  }

  return copy;
}
}  // namespace cxx
