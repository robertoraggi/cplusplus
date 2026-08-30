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
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
namespace {
[[nodiscard]] auto initializerCompletesDeclaredType(FieldSymbol* field)
    -> bool {
  auto type = field->type();
  if (containsPlaceholderType(type)) return true;
  return type_cast<UnboundedArrayType>(type) != nullptr;
}

void applyConstexprConstness(const TypeTraits& traits, FieldSymbol* field) {
  if (!field->isConstexpr()) return;
  field->setType(traits.add_const(field->type()));
}
}  // namespace

struct ASTRewriter::CoreDeclaratorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(BitfieldDeclaratorAST* ast)
      -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(ParameterPackAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(IdDeclaratorAST* ast) -> CoreDeclaratorAST*;

  [[nodiscard]] auto operator()(NestedDeclaratorAST* ast) -> CoreDeclaratorAST*;
};

struct ASTRewriter::DeclaratorChunkVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(FunctionDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;

  [[nodiscard]] auto operator()(ArrayDeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;
};

struct ASTRewriter::PtrOperatorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(PointerOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(ReferenceOperatorAST* ast) -> PtrOperatorAST*;

  [[nodiscard]] auto operator()(PtrToMemberOperatorAST* ast) -> PtrOperatorAST*;
};

struct ASTRewriter::DesignatorVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(DotDesignatorAST* ast) -> DesignatorAST*;

  [[nodiscard]] auto operator()(SubscriptDesignatorAST* ast) -> DesignatorAST*;
};

struct ASTRewriter::ExceptionSpecifierVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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
    FunctionSymbol* originalFunction,
    ExceptionSpecifierAST* functionExceptionSpecifier,
    std::function<void()> refreshType) {
  for (auto index = mark; index < pendingExceptionSpecifiers_.size(); ++index) {
    auto& pending = pendingExceptionSpecifiers_[index];
    pending.typeRefreshers.push_back(refreshType);
    if (function && pending.instance == functionExceptionSpecifier) {
      pendingFunctionExceptionSpecifiers_[function] = index;
      pending.originalFunction = originalFunction;
    }
  }
}

void ASTRewriter::resolvePendingExceptionSpecifier(std::size_t index) {
  auto& pending = pendingExceptionSpecifiers_[index];
  if (pending.state == PendingExceptionSpecifierState::kResolved) return;
  if (pending.state == PendingExceptionSpecifierState::kDeferred) return;
  if (pending.state == PendingExceptionSpecifierState::kResolving) {
    error(pending.instance->noexceptLoc,
          "recursive exception specification instantiation");
    return;
  }

  pending.state = PendingExceptionSpecifierState::kResolving;

  auto _ = Binder::ScopeGuard{&binder_};
  binder_.setScope(pending.scope);
  pending.instance->expression = expression(pending.pattern->expression);

  for (auto& refreshType : pending.typeRefreshers) refreshType();

  pending.state = PendingExceptionSpecifierState::kResolved;
}

void ASTRewriter::completePendingExceptionSpecifiers(std::size_t mark) {
  for (const auto& [function, index] : pendingFunctionExceptionSpecifiers_) {
    if (index < mark) continue;

    auto& pending = pendingExceptionSpecifiers_[index];
    if (pending.state != PendingExceptionSpecifierState::kUnresolved) continue;

    auto specification = std::make_unique<PendingExceptionSpecification>();
    specification->original = pending.pattern;
    specification->instance = pending.instance;
    specification->originalFunction = pending.originalFunction;
    specification->templateArguments = templateArguments_;
    specification->parentScope = pending.scope;
    specification->depth = depth_;
    function->setPendingExceptionSpecification(std::move(specification));
    pending.state = PendingExceptionSpecifierState::kDeferred;
  }

  for (auto index = mark; index < pendingExceptionSpecifiers_.size(); ++index) {
    resolvePendingExceptionSpecifier(index);
  }
}

auto ASTRewriter::hasPendingExceptionSpecifier(ClassSymbol* classSymbol) const
    -> bool {
  for (const auto& [function, index] : pendingFunctionExceptionSpecifiers_) {
    if (function->parent() != classSymbol) continue;
    const auto state = pendingExceptionSpecifiers_[index].state;
    if (state == PendingExceptionSpecifierState::kUnresolved ||
        state == PendingExceptionSpecifierState::kResolving)
      return true;
  }
  return false;
}

void ASTRewriter::completePendingExceptionSpecification(
    TranslationUnit* unit, FunctionSymbol* function) {
  if (!function) return;
  auto pending = function->pendingExceptionSpecification();
  if (!pending) return;
  if (pending->state == PendingExceptionSpecificationState::kResolved) return;
  if (pending->state == PendingExceptionSpecificationState::kResolving) {
    if (!pending->recursionDiagnosed) {
      unit->error(pending->instance->noexceptLoc,
                  "recursive exception specification instantiation");
      pending->recursionDiagnosed = true;
    }
    return;
  }

  pending->state = PendingExceptionSpecificationState::kResolving;

  auto rewriter =
      ASTRewriter{unit, pending->parentScope, pending->templateArguments};
  rewriter.depth_ = pending->depth;
  rewriter.inheritEnclosingTemplateArguments(pending->parentScope);
  rewriter.binder_.setInstantiatingSymbol(function);

  auto oldClass = symbol_cast<ClassSymbol>(pending->originalFunction->parent());
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

  auto oldParameters = pending->originalFunction->functionParameters();
  auto newParameters = function->functionParameters();
  if (oldParameters && newParameters)
    rewriter.remapScopeMembers(oldParameters, newParameters);

  pending->instance->expression =
      rewriter.expression(pending->original->expression);
  const bool isNoexcept = exceptionSpecifierIsNoexcept(unit, pending->instance);
  setFunctionNoexcept(unit->control(), function, isNoexcept);

  pending->state = PendingExceptionSpecificationState::kResolved;
}

auto ASTRewriter::requiresClause(RequiresClauseAST* ast) -> RequiresClauseAST* {
  if (!ast) return {};

  auto copy = RequiresClauseAST::create(arena());

  copy->requiresLoc = ast->requiresLoc;
  const auto saved = std::exchange(rewritingConstraintExpression_, true);
  copy->expression = unevaluatedExpression(ast->expression);
  rewritingConstraintExpression_ = saved;

  return copy;
}

auto ASTRewriter::parameterDeclarationClause(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseAST* {
  if (!ast) return {};

  auto copy = ParameterDeclarationClauseAST::create(arena());

  binder().bind(copy);

  auto _ = Binder::ScopeGuard(&binder_);

  binder().setScope(copy->functionParametersSymbol);

  auto originalParameter = [&](const Identifier* identifier) {
    if (!identifier || !ast->functionParametersSymbol)
      return static_cast<ParameterSymbol*>(nullptr);
    for (auto member : ast->functionParametersSymbol->members()) {
      if (auto parameter = symbol_cast<ParameterSymbol>(member);
          parameter && name_cast<Identifier>(parameter->name()) == identifier)
        return parameter;
    }
    return static_cast<ParameterSymbol*>(nullptr);
  };

  for (auto parameterDeclarationList = &copy->parameterDeclarationList;
       auto node : ListView{ast->parameterDeclarationList}) {
    auto paramDecl = ast_cast<ParameterDeclarationAST>(node);

    if (paramDecl && paramDecl->isPack) {
      ParameterPackSymbol* pack = nullptr;
      for (auto specNode : ListView{paramDecl->typeSpecifierList}) {
        pack = findReferencedParameterPack(specNode);
        if (pack) break;
      }

      if (pack) {
        auto originalParam = originalParameter(paramDecl->identifier);

        auto funcParamPack = control()->newParameterPackSymbol(
            binder().scope(), SourceLocation{});

        forEachPackElement(
            paramDecl, paramDecl->firstSourceLocation(),
            [&] {
              auto membersBefore = binder().scope()->members().size();

              auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
              if (value) value->isPack = false;
              *parameterDeclarationList = make_list_node(arena(), value);
              parameterDeclarationList = &(*parameterDeclarationList)->next;

              const auto& members = binder().scope()->members();
              if (members.size() > membersBefore) {
                funcParamPack->addElement(members.back());
              }
            },
            pack);

        if (originalParam) {
          functionParamPacks_[originalParam] = funcParamPack;
        }

        continue;
      }
    }

    auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
    *parameterDeclarationList = make_list_node(arena(), value);
    parameterDeclarationList = &(*parameterDeclarationList)->next;

    if (auto oldParameter =
            originalParameter(paramDecl ? paramDecl->identifier : nullptr)) {
      const auto& members = copy->functionParametersSymbol->members();
      if (!members.empty()) addSymbolRemap(oldParameter, members.back());
    }
  }

  copy->commaLoc = ast->commaLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->isVariadic = ast->isVariadic;

  if (ast->functionParametersSymbol && copy->functionParametersSymbol) {
    auto& oldParams = ast->functionParametersSymbol->members();
    auto& newParams = copy->functionParametersSymbol->members();
    auto n = std::min(oldParams.size(), newParams.size());
    for (std::size_t i = 0; i < n; ++i) {
      addSymbolRemap(oldParams[i], newParams[i]);
    }
  }

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

  if (binder_.scope()->isClass()) {
    auto symbol = binder_.declareMemberSymbol(copy->declarator, decl,
                                              addSymbolToParentScope);
    copy->symbol = symbol;

    if (auto funcSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      if (auto functionDeclarator = getFunctionPrototype(copy->declarator)) {
        if (auto params = functionDeclarator->parameterDeclarationClause) {
          funcSymbol->addSymbol(params->functionParametersSymbol);
        }
      }
    }

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
        if (currentTemplateHead_) {
          functionSymbol->setTemplateDeclaration(currentTemplateHead_);
          functionSymbol->setTemplateParameters(currentTemplateHead_->symbol);
        }
        copy->symbol = functionSymbol;
      } else {
        auto variableSymbol = binder_.declareVariable(copy->declarator, decl,
                                                      addSymbolToParentScope);
        copy->symbol = variableSymbol;

        auto declScope = decl.getScope();
        const auto isOutOfClassMemberDef = declScope && declScope->isClass();

        if (!addSymbolToParentScope && !isOutOfClassMemberDef) {
          auto templateVariable = symbol_cast<VariableSymbol>(ast->symbol);
          templateVariable->addSpecialization(unit_, templateArguments(),
                                              variableSymbol);
        }
      }
    }
  }

  auto function = symbol_cast<FunctionSymbol>(copy->symbol);
  if (function && functionTemplateHead) {
    function->setTemplateDeclaration(functionTemplateHead);
    function->setTemplateParameters(functionTemplateHead->symbol);
  }
  auto functionExceptionSpecifier =
      static_cast<ExceptionSpecifierAST*>(nullptr);
  if (auto prototype = getFunctionPrototype(copy->declarator))
    functionExceptionSpecifier = prototype->exceptionSpecifier;

  associatePendingExceptionSpecifiers(
      pendingExceptionSpecifierMark, function,
      symbol_cast<FunctionSymbol>(ast->symbol), functionExceptionSpecifier,
      [this, copy, baseType = declSpecs.type()] {
        auto type = getDeclaratorType(unit_, copy->declarator, baseType);
        if (copy->symbol) copy->symbol->setType(type);
      });

  if (auto fieldSymbol = symbol_cast<FieldSymbol>(copy->symbol);
      fieldSymbol && classBodyDepth_ > 0) {
    if (!fieldSymbol->isStatic()) {
      addSymbolRemap(ast->symbol, copy->symbol);
      if (ast->initializer)
        pendingFieldInitializers_.push_back({ast, copy, binder_.scope()});
      return copy;
    }

    const auto canDeferInitializer =
        ast->initializer && !initializerCompletesDeclaredType(fieldSymbol) &&
        !isEnclosedInDependentTemplate(unit_, binder_.scope(),
                                       /*stopAtConcreteSpecialization=*/true);

    if (canDeferInitializer) {
      addSymbolRemap(ast->symbol, copy->symbol);
      applyConstexprConstness(unit_->typeTraits(), fieldSymbol);

      auto pending = std::make_unique<PendingFieldInitializerInstantiation>();
      pending->pattern = ast;
      pending->instance = copy;
      pending->typeSpecifier = declSpecs.typeSpecifier();
      pending->templateArguments = templateArguments();
      pending->parentScope = binder_.scope();
      pending->depth = depth_;
      fieldSymbol->setPendingInitializer(std::move(pending));
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

  for (auto ptrOpList = &copy->ptrOpList;
       auto node : ListView{ast->ptrOpList}) {
    auto value = ptrOperator(node);
    *ptrOpList = make_list_node(arena(), value);
    ptrOpList = &(*ptrOpList)->next;
  }

  copy->coreDeclarator = coreDeclarator(ast->coreDeclarator);

  for (auto declaratorChunkList = &copy->declaratorChunkList;
       auto node : ListView{ast->declaratorChunkList}) {
    auto value = declaratorChunk(node);
    *declaratorChunkList = make_list_node(arena(), value);
    declaratorChunkList = &(*declaratorChunkList)->next;
  }

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = PointerOperatorAST::create(arena());

  copy->starLoc = ast->starLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto cvQualifierListCtx = DeclSpecs{rewrite.unit_};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite.specifier(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = ReferenceOperatorAST::create(arena());

  copy->refLoc = ast->refLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->refOp = ast->refOp;

  return copy;
}

auto ASTRewriter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorAST* {
  auto copy = PtrToMemberOperatorAST::create(arena());

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->starLoc = ast->starLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto cvQualifierListCtx = DeclSpecs{rewrite.unit_};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite.specifier(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  return copy;
}

auto ASTRewriter::CoreDeclaratorVisitor::operator()(BitfieldDeclaratorAST* ast)
    -> CoreDeclaratorAST* {
  auto copy = BitfieldDeclaratorAST::create(arena());

  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;
  copy->sizeExpression = rewrite.expression(ast->sizeExpression);

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

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

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

  auto cvQualifierListCtx = DeclSpecs{rewrite.unit_};
  for (auto cvQualifierList = &copy->cvQualifierList;
       auto node : ListView{ast->cvQualifierList}) {
    auto value = rewrite.specifier(node);
    *cvQualifierList = make_list_node(arena(), value);
    cvQualifierList = &(*cvQualifierList)->next;
    cvQualifierListCtx.accept(value);
  }

  copy->refLoc = ast->refLoc;
  copy->exceptionSpecifier =
      rewrite.exceptionSpecifier(ast->exceptionSpecifier);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

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

  auto typeQualifierListCtx = DeclSpecs{rewrite.unit_};
  for (auto typeQualifierList = &copy->typeQualifierList;
       auto node : ListView{ast->typeQualifierList}) {
    auto value = rewrite.specifier(node);
    *typeQualifierList = make_list_node(arena(), value);
    typeQualifierList = &(*typeQualifierList)->next;
    typeQualifierListCtx.accept(value);
  }

  copy->expression = rewrite.expression(ast->expression);
  copy->rbracketLoc = ast->rbracketLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

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
        {ast, copy, nullptr, binder()->scope()});
  } else {
    copy->expression = rewrite.expression(ast->expression);
  }

  return copy;
}
}  // namespace cxx
