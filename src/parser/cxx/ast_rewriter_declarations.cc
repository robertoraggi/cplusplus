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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/function_body.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

namespace cxx {
struct ASTRewriter::DeclarationVisitor : VisitorBase {
  TemplateDeclarationAST* templateHead = nullptr;

  DeclarationVisitor(ASTRewriter& rewrite, TemplateDeclarationAST* templateHead)
      : VisitorBase{rewrite}, templateHead(templateHead) {}

  [[nodiscard]] auto operator()(SimpleDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AsmDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(NamespaceAliasDefinitionAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(UsingDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(UsingEnumDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(UsingDirectiveAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(StaticAssertDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(AliasDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(OpaqueEnumDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(FunctionDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(TemplateDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ConceptDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(DeductionGuideAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExplicitInstantiationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExportDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ExportCompoundDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(LinkageSpecificationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(NamespaceDefinitionAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(EmptyDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(AttributeDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ModuleImportDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(ParameterDeclarationAST* ast)
      -> DeclarationAST*;

  [[nodiscard]] auto operator()(AccessDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(ForRangeDeclarationAST* ast) -> DeclarationAST*;

  [[nodiscard]] auto operator()(StructuredBindingDeclarationAST* ast)
      -> DeclarationAST*;
};

struct ASTRewriter::TemplateParameterVisitor : VisitorBase {
  [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
      -> TemplateParameterAST*;
};

struct ASTRewriter::FunctionBodyVisitor : VisitorBase {
  [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast) -> FunctionBodyAST*;
};

struct ASTRewriter::RequirementVisitor : VisitorBase {
  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementAST*;
};

auto ASTRewriter::declaration(DeclarationAST* ast) -> DeclarationAST* {
  if (!ast) return {};
  return visit(DeclarationVisitor{*this, nullptr}, ast);
}

auto ASTRewriter::templateParameter(TemplateParameterAST* ast)
    -> TemplateParameterAST* {
  if (!ast) return {};
  return visit(TemplateParameterVisitor{*this}, ast);
}

auto ASTRewriter::rewriteTemplateHead(TemplateDeclarationAST* ast)
    -> TemplateDeclarationAST* {
  if (!ast) return nullptr;

  auto copy = TemplateDeclarationAST::create(arena());
  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;
  copy->symbol = control()->newTemplateParametersSymbol(
      binder_.scope(), ast->symbol->location());
  copy->symbol->setExplicitTemplateSpecialization(
      ast->symbol->isExplicitTemplateSpecialization());
  copy->depth = ast->depth;

  binder_.setScope(copy->symbol);

  copy->templateParameterList =
      rewriteList(ast->templateParameterList, &ASTRewriter::templateParameter);

  copy->greaterLoc = ast->greaterLoc;
  copy->requiresClause = requiresClause(ast->requiresClause);
  return copy;
}

auto ASTRewriter::rewriteMemberTemplateHead(Symbol* patternSymbol)
    -> TemplateDeclarationAST* {
  if (!patternSymbol || binder_.instantiatingSymbol() == patternSymbol) {
    return nullptr;
  }

  auto enclosingClass = symbol_cast<ClassSymbol>(patternSymbol->parent());
  auto patternTemplateHead = TemplateEquivalence{unit_}.ownFunctionTemplateHead(
      enclosingClass, template_declaration_of(patternSymbol));
  if (patternTemplateHead == currentTemplatePatternHead_) {
    return currentTemplateHead_;
  }
  if (!patternTemplateHead) return nullptr;

  auto scopeGuard = Binder::ScopeGuard{&binder_};
  return rewriteTemplateHead(patternTemplateHead);
}

auto ASTRewriter::functionBody(FunctionBodyAST* ast) -> FunctionBodyAST* {
  if (!ast) return {};
  BodyErrorScope bodyErrors{*this};
  Binder::FunctionBodyGuard functionBodyScope{&binder_, binder_.scope()};
  return visit(FunctionBodyVisitor{*this}, ast);
}

auto ASTRewriter::lambdaBody(StatementAST* ast) -> CompoundStatementAST* {
  if (!ast) return {};
  BodyErrorScope bodyErrors{*this};
  return ast_cast<CompoundStatementAST>(statement(ast));
}

auto ASTRewriter::requirement(RequirementAST* ast) -> RequirementAST* {
  if (!ast) return {};
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  return visit(RequirementVisitor{*this}, ast);
}

auto ASTRewriter::typeConstraint(TypeConstraintAST* ast) -> TypeConstraintAST* {
  if (!ast) return {};

  auto copy = TypeConstraintAST::create(arena());

  copy->nestedNameSpecifier = nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  copy->templateArgumentList =
      rewriteTemplateArgumentList(ast->templateArgumentList);

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierAST* {
  if (!ast) return {};

  auto copy = NestedNamespaceSpecifierAST::create(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::usingDeclarator(UsingDeclaratorAST* ast)
    -> UsingDeclaratorAST* {
  if (!ast) return {};

  auto copy = UsingDeclaratorAST::create(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = unqualifiedId(ast->unqualifiedId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->symbol = ast->symbol;
  copy->isPack = ast->isPack;

  if (auto nns =
          ast_cast<SimpleNestedNameSpecifierAST>(copy->nestedNameSpecifier)) {
    if (nns->symbol) {
      if (auto remapped = remapSymbol(nns->symbol); remapped != nns->symbol) {
        if (auto scope = binder_.resolveNestedNameSpecifier(remapped)) {
          nns->symbol = scope;
        }
      }
    } else if (nns->identifier && !nns->nestedNameSpecifier) {
      auto resolved = qualifiedLookup(binder_.scope(), nns->identifier,
                                      [](Symbol* s) { return is_type(s); });
      nns->symbol = binder_.resolveNestedNameSpecifier(resolved);
    }
  }

  if (copy->nestedNameSpecifier && copy->nestedNameSpecifier->symbol) {
    auto name = get_name(control(), copy->unqualifiedId);
    auto target = qualifiedLookup(copy->nestedNameSpecifier->symbol, name);
    binder_.bind(copy, target);
  }

  if (ast->symbol && copy->symbol) addSymbolRemap(ast->symbol, copy->symbol);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = SimpleDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto declSpecifierListCtx = DeclSpecs{rewrite.unit_};
  declSpecifierListCtx.templateHead = templateHead;
  declSpecifierListCtx.attributeList = copy->attributeList;
  copy->declSpecifierList = rewrite.rewriteSpecifierList(
      ast->declSpecifierList, declSpecifierListCtx, templateHead);

  if (rewrite.shouldStopSubstitution()) return nullptr;

  if (declSpecifierListCtx.isFriend) {
    auto friendType =
        rewrite.unit_->typeTraits().remove_cv(declSpecifierListCtx.type());
    auto classType = type_cast<ClassType>(friendType);
    auto befriendingClass =
        symbol_cast<ClassSymbol>(rewrite.binder().declaringScope());
    if (classType && befriendingClass)
      classType->definition()->addBefriendingClass(befriendingClass);
  }

  if (!ast->initDeclaratorList) {
    auto patternSpecifier = ast->declSpecifierList;
    for (auto specifier = copy->declSpecifierList;
         specifier && patternSpecifier; specifier = specifier->next,
              patternSpecifier = patternSpecifier->next) {
      auto elab = ast_cast<ElaboratedTypeSpecifierAST>(specifier->value);
      if (!elab || elab->nestedNameSpecifier) continue;
      if (elab->classKey != TokenKind::T_CLASS &&
          elab->classKey != TokenKind::T_STRUCT &&
          elab->classKey != TokenKind::T_UNION) {
        continue;
      }
      auto patternElab =
          ast_cast<ElaboratedTypeSpecifierAST>(patternSpecifier->value);
      rewrite.binder().bind(elab, declSpecifierListCtx,
                            /*isDeclaration=*/true, elab->symbol);
      if (patternElab && patternElab->symbol && elab->symbol) {
        rewrite.addSymbolRemap(patternElab->symbol, elab->symbol);
      }
    }

    if (auto classSpec =
            ast_cast<ClassSpecifierAST>(declSpecifierListCtx.typeSpecifier())) {
      if (classSpec->symbol && !classSpec->symbol->name()) {
        rewrite.binder().declareAnonymousField(classSpec);
      }
    }
  }

  ListAppender<InitDeclaratorAST> appendInitDeclarator{
      arena(), copy->initDeclaratorList};
  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = rewrite.initDeclarator(node, declSpecifierListCtx);
    if (!value) return nullptr;
    appendInitDeclarator(value);
  }

  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  copy->semicolonLoc = ast->semicolonLoc;

  for (auto initDeclarator : ListView{copy->initDeclaratorList}) {
    binder()->applyDeclarationAttributes(initDeclarator->symbol,
                                         copy->attributeList,
                                         initDeclarator->declarator);

    auto function = symbol_cast<FunctionSymbol>(initDeclarator->symbol);
    if (!function) continue;

    auto functionTemplateHead = function->templateDeclaration();
    if (!functionTemplateHead || functionTemplateHead->declaration) continue;

    functionTemplateHead->declaration = SimpleDeclarationAST::create(
        arena(), copy->attributeList, copy->declSpecifierList,
        make_list_node(arena(), initDeclarator), copy->requiresClause,
        copy->semicolonLoc);
  }

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = AsmDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->asmQualifierList =
      rewrite.rewriteList(ast->asmQualifierList, &ASTRewriter::asmQualifier);

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;

  copy->outputOperandList =
      rewrite.rewriteList(ast->outputOperandList, &ASTRewriter::asmOperand);

  copy->inputOperandList =
      rewrite.rewriteList(ast->inputOperandList, &ASTRewriter::asmOperand);

  copy->clobberList =
      rewrite.rewriteList(ast->clobberList, &ASTRewriter::asmClobber);

  copy->gotoLabelList =
      rewrite.rewriteList(ast->gotoLabelList, &ASTRewriter::asmGotoLabel);

  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationAST* {
  auto copy = NamespaceAliasDefinitionAST::create(arena());

  copy->namespaceLoc = ast->namespaceLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = UsingDeclarationAST::create(arena());

  copy->usingLoc = ast->usingLoc;

  ListAppender<UsingDeclaratorAST> append{arena(), copy->usingDeclaratorList};

  for (auto node : ListView{ast->usingDeclaratorList}) {
    if (node->isPack) {
      auto pack =
          rewrite.findReferencedParameterPack(node->nestedNameSpecifier);
      if (!pack)
        pack = rewrite.findReferencedParameterPack(node->unqualifiedId);

      if (pack) {
        rewrite.forEachPackElement(
            node, node->ellipsisLoc,
            [&] {
              auto value = rewrite.usingDeclarator(node);
              value->ellipsisLoc = {};
              value->isPack = false;
              append(value);
            },
            pack);

        continue;
      }
    }

    append(rewrite.usingDeclarator(node));
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = UsingEnumDeclarationAST::create(arena());

  copy->usingLoc = ast->usingLoc;
  copy->enumTypeSpecifier = ast_cast<ElaboratedTypeSpecifierAST>(
      rewrite.specifier(ast->enumTypeSpecifier));
  if (copy->enumTypeSpecifier && ast->enumTypeSpecifier)
    copy->enumTypeSpecifier->symbol =
        rewrite.remapSymbol(ast->enumTypeSpecifier->symbol);
  copy->semicolonLoc = ast->semicolonLoc;

  binder()->bind(copy);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationAST* {
  auto copy = UsingDirectiveAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->usingLoc = ast->usingLoc;
  copy->namespaceLoc = ast->namespaceLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) -> DeclarationAST* {
  auto copy = StaticAssertDeclarationAST::create(arena());

  copy->staticAssertLoc = ast->staticAssertLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  if (binder()->instantiatingSymbol()) {
    auto checker = rewrite.typeChecker();
    checker.check(copy);
  }

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationAST* {
  const auto errorsBefore =
      translationUnit()->diagnosticsClient()->errorCount();
  auto copy = AliasDeclarationAST::create(arena());
  const auto pendingExceptionSpecifierMark =
      rewrite.pendingExceptionSpecifierMark();

  copy->usingLoc = ast->usingLoc;
  copy->identifierLoc = ast->identifierLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->equalLoc = ast->equalLoc;

  copy->gnuAttributeList = rewrite.rewriteList(
      ast->gnuAttributeList, &ASTRewriter::attributeSpecifier);

  copy->typeId = rewrite.typeId(ast->typeId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  if (rewrite.shouldStopSubstitution()) return nullptr;

  const auto addSymbolToParentScope =
      rewrite.binder().instantiatingSymbol() != ast->symbol;

  auto symbol = binder()->declareTypeAlias(
      copy->identifierLoc, copy->identifier, copy->typeId,
      addSymbolToParentScope, templateHead);
  const auto declarationIsValid =
      translationUnit()->diagnosticsClient()->errorCount() == errorsBefore;
  if (declarationIsValid && !addSymbolToParentScope &&
      !rewrite.substitutionFailed() &&
      !rewrite.retainsEnclosingTemplateLevels()) {
    ast->symbol->addSpecialization(translationUnit(),
                                   rewrite.templateArguments(), symbol);
  }

  if (templateHead && addSymbolToParentScope) {
    symbol->setTemplateDeclaration(templateHead);
    symbol->setTemplateParameters(templateHead->symbol);
  }

  copy->symbol = symbol;
  symbol->setDeclaration(copy);
  rewrite.addSymbolRemap(ast->symbol, symbol);

  rewrite.associatePendingExceptionSpecifiers(
      pendingExceptionSpecifierMark, nullptr, nullptr,
      [copy, symbol] { symbol->setType(copy->typeId->type); });

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = OpaqueEnumDeclarationAST::create(arena());

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewrite.unit_};
  copy->typeSpecifierList = rewrite.rewriteSpecifierList(ast->typeSpecifierList,
                                                         typeSpecifierListCtx);

  copy->semicolonLoc = ast->semicolonLoc;
  copy->symbol = rewrite.remapSymbol(ast->symbol);

  return copy;
}

namespace {

[[nodiscard]] auto definesDefaultedFunction(FunctionDefinitionAST* ast)
    -> bool {
  if (ast->symbol) return ast->symbol->isDefaulted();
  return ast_cast<DefaultFunctionBodyAST>(ast->functionBody) != nullptr;
}

[[nodiscard]] auto definesDeletedFunction(FunctionDefinitionAST* ast) -> bool {
  if (ast->symbol) return ast->symbol->isDeleted();
  return ast_cast<DeleteFunctionBodyAST>(ast->functionBody) != nullptr;
}

}  // namespace

auto ASTRewriter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationAST* {
  const auto errorsBefore =
      translationUnit()->diagnosticsClient()->errorCount();
  auto copy = FunctionDefinitionAST::create(arena());
  auto functionTemplateHead = templateHead;
  if (!functionTemplateHead) {
    auto patternFunction = symbol_cast<FunctionSymbol>(ast->symbol);
    functionTemplateHead = rewrite.rewriteMemberTemplateHead(patternFunction);
  }

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto declSpecifierListCtx = DeclSpecs{rewrite.unit_};
  declSpecifierListCtx.templateHead = functionTemplateHead;
  declSpecifierListCtx.attributeList = copy->attributeList;
  copy->declSpecifierList = rewrite.rewriteSpecifierList(ast->declSpecifierList,
                                                         declSpecifierListCtx);

  if (rewrite.shouldStopSubstitution()) return nullptr;

  const auto pendingExceptionSpecifierMark =
      rewrite.pendingExceptionSpecifierMark();
  copy->declarator = rewrite.declarator(ast->declarator);

  if (rewrite.shouldStopSubstitution()) return nullptr;

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());

  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  declaratorDecl.trailingRequiresClause = copy->requiresClause;

  const bool isTemplateInstantiation =
      ast->symbol && ast->symbol->templateDeclaration() &&
      rewrite.binder().instantiatingSymbol() == ast->symbol;

  auto _ = Binder::ScopeGuard{binder()};

  auto declaratorScope = declaratorDecl.getScope();
  if (auto remappedScope =
          symbol_cast<ScopeSymbol>(rewrite.remapSymbol(declaratorScope))) {
    declaratorScope = remappedScope;
  }
  const auto isOutOfClassMemberDef =
      declaratorScope && declaratorScope->isClass();
  if (declaratorScope) binder()->setScope(declaratorScope);

  const bool isFunctionTemplateSpecialization =
      rewrite.instantiatingFunctionTemplateSpecialization_;
  rewrite.instantiatingFunctionTemplateSpecialization_ = false;

  FunctionSymbol* functionSymbol =
      std::exchange(rewrite.functionInstanceToDefine_, nullptr);
  const bool definesExistingInstance = functionSymbol != nullptr;
  if (!functionSymbol && (!isTemplateInstantiation || isOutOfClassMemberDef) &&
      !isFunctionTemplateSpecialization &&
      !(ast->symbol && ast->symbol->isFriend())) {
    functionSymbol = binder()->getFunction(
        binder()->scope(), declaratorDecl.getName(), declaratorType,
        functionTemplateHead, copy->requiresClause);
  }
  if (!functionSymbol) {
    const bool addSymbolToParentScope =
        !isFunctionTemplateSpecialization && !isTemplateInstantiation;
    declaratorDecl.isFunctionDefinition = true;
    functionSymbol = binder()->declareFunction(copy->declarator, declaratorDecl,
                                               addSymbolToParentScope);
  }

  binder()->inheritDeclarationAttributes(functionSymbol, ast->symbol);

  if (ast->symbol && ast->symbol->isFriend()) functionSymbol->setFriend(true);

  if (ast->symbol && ast->symbol->isInline()) functionSymbol->setInline(true);

  if (ast->symbol && ast->symbol->isStatic()) functionSymbol->setStatic(true);

  if (ast->symbol && ast->symbol->isConsteval())
    functionSymbol->setConsteval(true);

  if (ast->symbol && ast->symbol->isConstexpr())
    functionSymbol->setConstexpr(true);

  if (functionSymbol->isFriend() && !definesExistingInstance)
    binder()->recordFunctionDefinition(functionSymbol);

  if (isOutOfClassMemberDef) {
    functionSymbol->setDefined(true);
    if (auto canon = functionSymbol->canonical(); canon != functionSymbol) {
      canon->setDefinition(functionSymbol);
    }
  }

  auto functionDeclarator = getFunctionPrototype(copy->declarator);

  if (auto params = functionDeclarator->parameterDeclarationClause) {
    auto newParams = params->functionParametersSymbol;
    if (auto oldParams = functionSymbol->functionParameters()) {
      binder()->copyDefaultArguments(oldParams, newParams);

      functionSymbol->replaceSymbol(oldParams, newParams);
      newParams->setParent(functionSymbol);
    } else {
      functionSymbol->addSymbol(newParams);
    }
    binder()->setScope(newParams);
  } else {
    binder()->setScope(functionSymbol);
  }

  copy->symbol = functionSymbol;
  copy->symbol->setDeclaration(copy);
  if (functionTemplateHead && !functionTemplateHead->declaration) {
    functionTemplateHead->declaration = copy;
  }

  rewrite.associatePendingExceptionSpecifiers(
      pendingExceptionSpecifierMark, functionSymbol,
      functionDeclarator->exceptionSpecifier,
      [this, copy, functionSymbol, baseType = declSpecifierListCtx.type()] {
        auto type = getDeclaratorType(rewrite.translationUnit(),
                                      copy->declarator, baseType);
        functionSymbol->setType(type);
      });

  if (definesDefaultedFunction(ast)) functionSymbol->setDefaulted(true);
  if (definesDeletedFunction(ast)) functionSymbol->setDeleted(true);

  if (ast->symbol) functionSymbol->setAbiTags(ast->symbol->abiTagList());

  if (!definesExistingInstance && ast->symbol &&
      ast->symbol->templateDeclaration() &&
      (!isOutOfClassMemberDef || isFunctionTemplateSpecialization)) {
    auto instSym =
        symbol_cast<FunctionSymbol>(rewrite.binder().instantiatingSymbol());
    auto primaryForThis = ast->symbol->canonical();
    const auto declarationIsValid =
        translationUnit()->diagnosticsClient()->errorCount() == errorsBefore;
    if (declarationIsValid && !rewrite.substitutionFailed() && instSym &&
        (instSym == ast->symbol || instSym == primaryForThis ||
         instSym->canonical() == primaryForThis ||
         (isFunctionTemplateSpecialization &&
          instSym->templateDeclaration()))) {
      instSym->addSpecialization(translationUnit(), rewrite.templateArguments(),
                                 functionSymbol);
    }
  }

  if (!isTemplateInstantiation && !isFunctionTemplateSpecialization) {
    if (functionTemplateHead) {
      functionSymbol->setTemplateDeclaration(functionTemplateHead);
      functionSymbol->setTemplateParameters(functionTemplateHead->symbol);
    } else if (ast->symbol && ast->symbol->templateParameters()) {
      functionSymbol->setTemplateDeclaration(
          ast->symbol->templateDeclaration());
      functionSymbol->setTemplateParameters(ast->symbol->templateParameters());
    }
  }

  if (!rewrite.restrictedToDeclarations()) {
    if (auto oldFunc = symbol_cast<FunctionSymbol>(ast->symbol)) {
      auto oldClass = symbol_cast<ClassSymbol>(oldFunc->parent());
      auto newClass = symbol_cast<ClassSymbol>(functionSymbol->parent());
      if (oldClass && newClass && oldClass != newClass) {
        rewrite.remapScopeMembers(oldClass, newClass);
      }

      if (auto oldParams = oldFunc->functionParameters()) {
        if (auto newParams = functionSymbol->functionParameters()) {
          rewrite.remapFunctionParameters(oldParams, newParams);
        }
      }
    }

    copy->functionBody = rewrite.functionBody(ast->functionBody);

    binder()->synthesizeCompleteObjectCtor(functionSymbol);

    if (body_statement(copy->functionBody)) {
      rewrite.checkMemInitializers(functionSymbol, copy->functionBody);
      binder()->finishAutoReturnType(functionSymbol);
    }

    binder()->synthesizeDefaultedMemberBody(functionSymbol);
    validateCompletedInstantiation(rewrite.unit_, functionSymbol, copy);
  }

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationAST* {
  auto _ = Binder::ScopeGuard{binder()};
  auto copy = rewrite.rewriteTemplateHead(ast);
  auto savedPatternHead =
      std::exchange(rewrite.currentTemplatePatternHead_, ast);
  auto savedTemplateHead = std::exchange(rewrite.currentTemplateHead_, copy);
  if (ast->declaration)
    copy->declaration =
        visit(DeclarationVisitor{rewrite, copy}, ast->declaration);
  rewrite.currentTemplateHead_ = savedTemplateHead;
  rewrite.currentTemplatePatternHead_ = savedPatternHead;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = ConceptDefinitionAST::create(arena());

  copy->conceptLoc = ast->conceptLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationAST* {
  auto copy = DeductionGuideAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);
  copy->explicitSpecifier = rewrite.specifier(ast->explicitSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->parameterDeclarationClause =
      rewrite.parameterDeclarationClause(ast->parameterDeclarationClause);
  copy->rparenLoc = ast->rparenLoc;
  copy->arrowLoc = ast->arrowLoc;
  copy->templateId =
      ast_cast<SimpleTemplateIdAST>(rewrite.unqualifiedId(ast->templateId));
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationAST* {
  auto copy = ExplicitInstantiationAST::create(arena());

  copy->externLoc = ast->externLoc;
  copy->templateLoc = ast->templateLoc;
  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = ExportDeclarationAST::create(arena());

  copy->exportLoc = ast->exportLoc;
  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationAST* {
  auto copy = ExportCompoundDeclarationAST::create(arena());

  copy->exportLoc = ast->exportLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  copy->declarationList =
      rewrite.rewriteList(ast->declarationList, &ASTRewriter::declaration);

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationAST* {
  auto copy = LinkageSpecificationAST::create(arena());

  copy->externLoc = ast->externLoc;
  copy->stringliteralLoc = ast->stringliteralLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  copy->declarationList =
      rewrite.rewriteList(ast->declarationList, &ASTRewriter::declaration);

  copy->rbraceLoc = ast->rbraceLoc;
  copy->stringLiteral = ast->stringLiteral;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = NamespaceDefinitionAST::create(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->namespaceLoc = ast->namespaceLoc;

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->nestedNamespaceSpecifierList =
      rewrite.rewriteList(ast->nestedNamespaceSpecifierList,
                          &ASTRewriter::nestedNamespaceSpecifier);

  copy->identifierLoc = ast->identifierLoc;

  copy->extraAttributeList = rewrite.rewriteList(
      ast->extraAttributeList, &ASTRewriter::attributeSpecifier);

  copy->lbraceLoc = ast->lbraceLoc;

  copy->declarationList =
      rewrite.rewriteList(ast->declarationList, &ASTRewriter::declaration);

  copy->rbraceLoc = ast->rbraceLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = EmptyDeclarationAST::create(arena());

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = AttributeDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationAST* {
  auto copy = ModuleImportDeclarationAST::create(arena());

  copy->importLoc = ast->importLoc;
  copy->importName = rewrite.importName(ast->importName);

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = ParameterDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  copy->thisLoc = ast->thisLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewrite.unit_};
  typeSpecifierListCtx.attributeList = copy->attributeList;
  copy->typeSpecifierList = rewrite.rewriteSpecifierList(ast->typeSpecifierList,
                                                         typeSpecifierListCtx);

  const auto pendingExceptionSpecifierMark =
      rewrite.pendingExceptionSpecifierMark();
  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->type = declaratorType;
  copy->equalLoc = ast->equalLoc;
  copy->identifier = ast->identifier;
  copy->isThisIntroduced = ast->isThisIntroduced;
  copy->isPack = ast->isPack && !rewrite.expandsAnActivePack(ast);

  const bool inTemplateParameters =
      binder()->scope()->isTemplateParameters() ||
      rewrite.rewritingTemplateParameterDeclaration();

  auto defaultArgument =
      ASTRewriter::patternDefaultArgument(translationUnit(), ast);
  auto defaultArgumentScope = binder()->scope();

  if (defaultArgument && !ast->expression) {
    auto parameter = symbol_cast<ParameterSymbol>(ast->symbol);
    if (auto patternClass = parameter->enclosingClass()) {
      if (auto instanceClass =
              symbol_cast<ClassSymbol>(rewrite.remapSymbol(patternClass)))
        defaultArgumentScope = instanceClass;
    }
  }

  const auto defersDefaultArgument = defaultArgument && !inTemplateParameters;

  if (defaultArgument && !defersDefaultArgument) {
    auto _ = Binder::ScopeGuard{binder()};
    binder()->setScope(defaultArgumentScope);
    TranslationUnit::DeferredInitializerScope deferredInitializer{
        translationUnit(), true};
    copy->expression = rewrite.expression(defaultArgument);
  }

  binder()->bind(copy, declaratorDecl, inTemplateParameters);

  auto parameter = copy->symbol;

  if (defersDefaultArgument && parameter) {
    parameter->setPendingDefaultArgument(
        rewrite.pendingInstantiationOf(ast, copy, defaultArgumentScope));
  }

  rewrite.associatePendingExceptionSpecifiers(
      pendingExceptionSpecifierMark, nullptr, nullptr,
      [this, copy, parameter, baseType = typeSpecifierListCtx.type()] {
        copy->type = getDeclaratorType(rewrite.translationUnit(),
                                       copy->declarator, baseType);
        if (parameter) parameter->setType(copy->type);
      });

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = AccessDeclarationAST::create(arena());

  copy->accessLoc = ast->accessLoc;
  copy->colonLoc = ast->colonLoc;
  copy->accessSpecifier = ast->accessSpecifier;

  binder()->setCurrentAccessSpecifier(toAccessSpecifier(
      copy->accessSpecifier, binder()->defaultAccessSpecifier()));

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = ForRangeDeclarationAST::create(arena());

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationAST* {
  auto copy = StructuredBindingDeclarationAST::create(arena());

  copy->attributeList =
      rewrite.rewriteList(ast->attributeList, &ASTRewriter::attributeSpecifier);

  auto declSpecifierListCtx = DeclSpecs{rewrite.unit_};
  copy->declSpecifierList = rewrite.rewriteSpecifierList(ast->declSpecifierList,
                                                         declSpecifierListCtx);

  copy->refQualifierLoc = ast->refQualifierLoc;
  copy->lbracketLoc = ast->lbracketLoc;

  copy->bindingList =
      rewrite.rewriteList(ast->bindingList, &ASTRewriter::unqualifiedId);

  copy->rbracketLoc = ast->rbracketLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->semicolonLoc = ast->semicolonLoc;

  rewrite.binder().bindStructuredBindings(copy, declSpecifierListCtx);

  rewrite.remapStructuredBindingSymbols(ast, copy);

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = TemplateTypeParameterAST::create(arena());

  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  {
    auto _ = Binder::ScopeGuard{binder()};
    auto parameters = control()->newTemplateParametersSymbol(binder()->scope(),
                                                             ast->templateLoc);
    binder()->setScope(parameters);

    copy->templateParameterList = rewrite.rewriteList(
        ast->templateParameterList, &ASTRewriter::templateParameter);

    copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;

  copy->idExpression =
      ast_cast<IdExpressionAST>(rewrite.expression(ast->idExpression));

  binder()->bind(copy, copy->index, copy->depth);
  rewrite.addSymbolRemap(ast->symbol, copy->symbol);

  recordDefaultTemplateArgument(copy, ast);

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterAST* {
  auto copy = NonTypeTemplateParameterAST::create(arena());

  copy->depth = ast->depth;
  copy->index = ast->index;

  {
    ASTRewriter::TemplateParameterDeclarationGuard guard{rewrite};
    copy->declaration = ast_cast<ParameterDeclarationAST>(
        rewrite.declaration(ast->declaration));
  }

  binder()->bind(copy, copy->index, copy->depth);
  recordDefaultTemplateArgument(copy, ast);
  rewrite.addSymbolRemap(ast->symbol, copy->symbol);

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = TypenameTypeParameterAST::create(arena());

  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;
  copy->isSynthesized = ast->isSynthesized;

  binder()->bind(copy, copy->index, copy->depth);
  recordDefaultTemplateArgument(copy, ast);
  rewrite.addSymbolRemap(ast->symbol, copy->symbol);

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = ConstraintTypeParameterAST::create(arena());

  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->typeConstraint = rewrite.typeConstraint(ast->typeConstraint);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->identifier = ast->identifier;
  copy->isSynthesized = ast->isSynthesized;

  binder()->bind(copy, copy->index, copy->depth);
  recordDefaultTemplateArgument(copy, ast);
  rewrite.addSymbolRemap(ast->symbol, copy->symbol);

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = DefaultFunctionBodyAST::create(arena());

  copy->equalLoc = ast->equalLoc;
  copy->defaultLoc = ast->defaultLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = CompoundStatementFunctionBodyAST::create(arena());

  copy->colonLoc = ast->colonLoc;

  copy->memInitializerList =
      rewrite.rewriteMemInitializerList(ast->memInitializerList);

  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = TryStatementFunctionBodyAST::create(arena());

  copy->tryLoc = ast->tryLoc;
  copy->colonLoc = ast->colonLoc;

  copy->memInitializerList =
      rewrite.rewriteMemInitializerList(ast->memInitializerList);

  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  copy->handlerList =
      rewrite.rewriteList(ast->handlerList, &ASTRewriter::handler);

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = DeleteFunctionBodyAST::create(arena());

  copy->equalLoc = ast->equalLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementAST* {
  auto copy = SimpleRequirementAST::create(arena());

  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementAST* {
  auto copy = CompoundRequirementAST::create(arena());

  copy->lbraceLoc = ast->lbraceLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  copy->rbraceLoc = ast->rbraceLoc;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeConstraint = rewrite.typeConstraint(ast->typeConstraint);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementAST* {
  auto copy = TypeRequirementAST::create(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementAST* {
  auto copy = NestedRequirementAST::create(arena());

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = rewrite.unevaluatedExpression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}
}  // namespace cxx
