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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/symbols.h>

namespace cxx {

struct ASTRewriter::DeclarationVisitor {
  ASTRewriter& rewrite;
  TemplateDeclarationAST* templateHead = nullptr;

  DeclarationVisitor(ASTRewriter& rewrite, TemplateDeclarationAST* templateHead)
      : rewrite(rewrite), templateHead(templateHead) {}

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

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

struct ASTRewriter::TemplateParameterVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(TemplateTypeParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(TypenameTypeParameterAST* ast)
      -> TemplateParameterAST*;

  [[nodiscard]] auto operator()(ConstraintTypeParameterAST* ast)
      -> TemplateParameterAST*;
};

struct ASTRewriter::FunctionBodyVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(DefaultFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(CompoundStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(TryStatementFunctionBodyAST* ast)
      -> FunctionBodyAST*;

  [[nodiscard]] auto operator()(DeleteFunctionBodyAST* ast) -> FunctionBodyAST*;
};

struct ASTRewriter::RequirementVisitor {
  ASTRewriter& rewrite;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return rewrite.unit_;
  }

  [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }
  [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }
  [[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }
  [[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }

  [[nodiscard]] auto operator()(SimpleRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(CompoundRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(TypeRequirementAST* ast) -> RequirementAST*;

  [[nodiscard]] auto operator()(NestedRequirementAST* ast) -> RequirementAST*;
};

auto ASTRewriter::declaration(DeclarationAST* ast,
                              TemplateDeclarationAST* templateHead)
    -> DeclarationAST* {
  if (!ast) return {};
  return visit(DeclarationVisitor{*this, templateHead}, ast);
}

auto ASTRewriter::templateParameter(TemplateParameterAST* ast)
    -> TemplateParameterAST* {
  if (!ast) return {};
  return visit(TemplateParameterVisitor{*this}, ast);
}

auto ASTRewriter::functionBody(FunctionBodyAST* ast) -> FunctionBodyAST* {
  if (!ast) return {};
  return visit(FunctionBodyVisitor{*this}, ast);
}

auto ASTRewriter::requirement(RequirementAST* ast) -> RequirementAST* {
  if (!ast) return {};
  return visit(RequirementVisitor{*this}, ast);
}

auto ASTRewriter::typeConstraint(TypeConstraintAST* ast) -> TypeConstraintAST* {
  if (!ast) return {};

  auto copy = make_node<TypeConstraintAST>(arena());

  copy->nestedNameSpecifier = nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->identifierLoc = ast->identifierLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateArgumentList = &copy->templateArgumentList;
       auto node : ListView{ast->templateArgumentList}) {
    auto value = templateArgument(node);
    *templateArgumentList = make_list_node(arena(), value);
    templateArgumentList = &(*templateArgumentList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierAST* {
  if (!ast) return {};

  auto copy = make_node<NestedNamespaceSpecifierAST>(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->scopeLoc = ast->scopeLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::usingDeclarator(UsingDeclaratorAST* ast)
    -> UsingDeclaratorAST* {
  if (!ast) return {};

  auto copy = make_node<UsingDeclaratorAST>(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier = nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId = unqualifiedId(ast->unqualifiedId);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->symbol = ast->symbol;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<SimpleDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite.specifier(node, templateHead);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  for (auto initDeclaratorList = &copy->initDeclaratorList;
       auto node : ListView{ast->initDeclaratorList}) {
    auto value = rewrite.initDeclarator(node, declSpecifierListCtx);
    *initDeclaratorList = make_list_node(arena(), value);
    initDeclaratorList = &(*initDeclaratorList)->next;
  }

  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AsmDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  for (auto asmQualifierList = &copy->asmQualifierList;
       auto node : ListView{ast->asmQualifierList}) {
    auto value = rewrite.asmQualifier(node);
    *asmQualifierList =
        make_list_node(arena(), ast_cast<AsmQualifierAST>(value));
    asmQualifierList = &(*asmQualifierList)->next;
  }

  copy->asmLoc = ast->asmLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->literalLoc = ast->literalLoc;

  for (auto outputOperandList = &copy->outputOperandList;
       auto node : ListView{ast->outputOperandList}) {
    auto value = rewrite.asmOperand(node);
    *outputOperandList =
        make_list_node(arena(), ast_cast<AsmOperandAST>(value));
    outputOperandList = &(*outputOperandList)->next;
  }

  for (auto inputOperandList = &copy->inputOperandList;
       auto node : ListView{ast->inputOperandList}) {
    auto value = rewrite.asmOperand(node);
    *inputOperandList = make_list_node(arena(), ast_cast<AsmOperandAST>(value));
    inputOperandList = &(*inputOperandList)->next;
  }

  for (auto clobberList = &copy->clobberList;
       auto node : ListView{ast->clobberList}) {
    auto value = rewrite.asmClobber(node);
    *clobberList = make_list_node(arena(), ast_cast<AsmClobberAST>(value));
    clobberList = &(*clobberList)->next;
  }

  for (auto gotoLabelList = &copy->gotoLabelList;
       auto node : ListView{ast->gotoLabelList}) {
    auto value = rewrite.asmGotoLabel(node);
    *gotoLabelList = make_list_node(arena(), ast_cast<AsmGotoLabelAST>(value));
    gotoLabelList = &(*gotoLabelList)->next;
  }

  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;
  copy->literal = ast->literal;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationAST* {
  auto copy = make_node<NamespaceAliasDefinitionAST>(arena());

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
  auto copy = make_node<UsingDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;

  for (auto usingDeclaratorList = &copy->usingDeclaratorList;
       auto node : ListView{ast->usingDeclaratorList}) {
    auto value = rewrite.usingDeclarator(node);
    *usingDeclaratorList = make_list_node(arena(), value);
    usingDeclaratorList = &(*usingDeclaratorList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<UsingEnumDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;
  copy->enumTypeSpecifier = ast_cast<ElaboratedTypeSpecifierAST>(
      rewrite.specifier(ast->enumTypeSpecifier));
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<UsingDirectiveAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

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
  auto copy = make_node<StaticAssertDeclarationAST>(arena());

  copy->staticAssertLoc = ast->staticAssertLoc;
  copy->lparenLoc = ast->lparenLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->commaLoc = ast->commaLoc;
  copy->literalLoc = ast->literalLoc;
  copy->literal = ast->literal;
  copy->rparenLoc = ast->rparenLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AliasDeclarationAST>(arena());

  copy->usingLoc = ast->usingLoc;
  copy->identifierLoc = ast->identifierLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->equalLoc = ast->equalLoc;

  for (auto gnuAttributeList = &copy->gnuAttributeList;
       auto node : ListView{ast->gnuAttributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *gnuAttributeList = make_list_node(arena(), value);
    gnuAttributeList = &(*gnuAttributeList)->next;
  }

  copy->typeId = rewrite.typeId(ast->typeId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;

  const auto addSymbolToParentScope =
      rewrite.binder().instantiatingSymbol() != ast->symbol;

  auto symbol = binder()->declareTypeAlias(copy->identifierLoc, copy->typeId,
                                           addSymbolToParentScope);
  // symbol->setTemplateDeclaration(templateHead);

  copy->symbol = symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<OpaqueEnumDeclarationAST>(arena());

  copy->enumLoc = ast->enumLoc;
  copy->classLoc = ast->classLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->unqualifiedId =
      ast_cast<NameIdAST>(rewrite.unqualifiedId(ast->unqualifiedId));
  copy->colonLoc = ast->colonLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite.specifier(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->emicolonLoc = ast->emicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<FunctionDefinitionAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite.specifier(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{declSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          declSpecifierListCtx.type());

  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);

  auto functionSymbol = binder()->getFunction(
      binder()->scope(), declaratorDecl.getName(), declaratorType);

  if (!functionSymbol) {
    functionSymbol =
        binder()->declareFunction(copy->declarator, declaratorDecl);
  }

  auto _ = Binder::ScopeGuard{binder()};

  binder()->setScope(functionSymbol);

  copy->symbol = functionSymbol;
  copy->symbol->setDeclaration(copy);
  copy->functionBody = rewrite.functionBody(ast->functionBody);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<TemplateDeclarationAST>(arena());

  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  auto _ = Binder::ScopeGuard{binder()};

  auto templateParametersSymbol = control()->newTemplateParametersSymbol(
      binder()->scope(), ast->symbol->location());

  copy->symbol = templateParametersSymbol;

  binder()->setScope(templateParametersSymbol);

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite.templateParameter(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;

  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  copy->declaration = rewrite.declaration(ast->declaration, copy);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ConceptDefinitionAST>(arena());

  copy->conceptLoc = ast->conceptLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->identifier = ast->identifier;
  copy->symbol = ast->symbol;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<DeductionGuideAST>(arena());

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
  auto copy = make_node<ExplicitInstantiationAST>(arena());

  copy->externLoc = ast->externLoc;
  copy->templateLoc = ast->templateLoc;
  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ExportDeclarationAST>(arena());

  copy->exportLoc = ast->exportLoc;
  copy->declaration = rewrite.declaration(ast->declaration);

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<ExportCompoundDeclarationAST>(arena());

  copy->exportLoc = ast->exportLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<LinkageSpecificationAST>(arena());

  copy->externLoc = ast->externLoc;
  copy->stringliteralLoc = ast->stringliteralLoc;
  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->stringLiteral = ast->stringLiteral;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<NamespaceDefinitionAST>(arena());

  copy->inlineLoc = ast->inlineLoc;
  copy->namespaceLoc = ast->namespaceLoc;

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  for (auto nestedNamespaceSpecifierList = &copy->nestedNamespaceSpecifierList;
       auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = rewrite.nestedNamespaceSpecifier(node);
    *nestedNamespaceSpecifierList = make_list_node(arena(), value);
    nestedNamespaceSpecifierList = &(*nestedNamespaceSpecifierList)->next;
  }

  copy->identifierLoc = ast->identifierLoc;

  for (auto extraAttributeList = &copy->extraAttributeList;
       auto node : ListView{ast->extraAttributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *extraAttributeList = make_list_node(arena(), value);
    extraAttributeList = &(*extraAttributeList)->next;
  }

  copy->lbraceLoc = ast->lbraceLoc;

  for (auto declarationList = &copy->declarationList;
       auto node : ListView{ast->declarationList}) {
    auto value = rewrite.declaration(node);
    *declarationList = make_list_node(arena(), value);
    declarationList = &(*declarationList)->next;
  }

  copy->rbraceLoc = ast->rbraceLoc;
  copy->identifier = ast->identifier;
  copy->isInline = ast->isInline;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<EmptyDeclarationAST>(arena());

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AttributeDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<ModuleImportDeclarationAST>(arena());

  copy->importLoc = ast->importLoc;
  copy->importName = rewrite.importName(ast->importName);

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ParameterDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  copy->thisLoc = ast->thisLoc;

  auto typeSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto typeSpecifierList = &copy->typeSpecifierList;
       auto node : ListView{ast->typeSpecifierList}) {
    auto value = rewrite.specifier(node);
    *typeSpecifierList = make_list_node(arena(), value);
    typeSpecifierList = &(*typeSpecifierList)->next;
    typeSpecifierListCtx.accept(value);
  }
  typeSpecifierListCtx.finish();

  copy->declarator = rewrite.declarator(ast->declarator);

  auto declaratorDecl = Decl{typeSpecifierListCtx, copy->declarator};
  auto declaratorType = getDeclaratorType(translationUnit(), copy->declarator,
                                          typeSpecifierListCtx.type());
  copy->type = declaratorType;
  copy->equalLoc = ast->equalLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->identifier = ast->identifier;
  copy->isThisIntroduced = ast->isThisIntroduced;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<AccessDeclarationAST>(arena());

  copy->accessLoc = ast->accessLoc;
  copy->colonLoc = ast->colonLoc;
  copy->accessSpecifier = ast->accessSpecifier;

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationAST* {
  auto copy = make_node<ForRangeDeclarationAST>(arena());

  return copy;
}

auto ASTRewriter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationAST* {
  auto copy = make_node<StructuredBindingDeclarationAST>(arena());

  for (auto attributeList = &copy->attributeList;
       auto node : ListView{ast->attributeList}) {
    auto value = rewrite.attributeSpecifier(node);
    *attributeList = make_list_node(arena(), value);
    attributeList = &(*attributeList)->next;
  }

  auto declSpecifierListCtx = DeclSpecs{rewriter()};
  for (auto declSpecifierList = &copy->declSpecifierList;
       auto node : ListView{ast->declSpecifierList}) {
    auto value = rewrite.specifier(node);
    *declSpecifierList = make_list_node(arena(), value);
    declSpecifierList = &(*declSpecifierList)->next;
    declSpecifierListCtx.accept(value);
  }
  declSpecifierListCtx.finish();

  copy->refQualifierLoc = ast->refQualifierLoc;
  copy->lbracketLoc = ast->lbracketLoc;

  for (auto bindingList = &copy->bindingList;
       auto node : ListView{ast->bindingList}) {
    auto value = rewrite.unqualifiedId(node);
    *bindingList = make_list_node(arena(), ast_cast<NameIdAST>(value));
    bindingList = &(*bindingList)->next;
  }

  copy->rbracketLoc = ast->rbracketLoc;
  copy->initializer = rewrite.expression(ast->initializer);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<TemplateTypeParameterAST>(arena());

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->templateLoc = ast->templateLoc;
  copy->lessLoc = ast->lessLoc;

  for (auto templateParameterList = &copy->templateParameterList;
       auto node : ListView{ast->templateParameterList}) {
    auto value = rewrite.templateParameter(node);
    *templateParameterList = make_list_node(arena(), value);
    templateParameterList = &(*templateParameterList)->next;
  }

  copy->greaterLoc = ast->greaterLoc;
  copy->requiresClause = rewrite.requiresClause(ast->requiresClause);
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->idExpression =
      ast_cast<IdExpressionAST>(rewrite.expression(ast->idExpression));
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<NonTypeTemplateParameterAST>(arena());

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->declaration =
      ast_cast<ParameterDeclarationAST>(rewrite.declaration(ast->declaration));

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<TypenameTypeParameterAST>(arena());

  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->classKeyLoc = ast->classKeyLoc;
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->identifier = ast->identifier;
  copy->isPack = ast->isPack;

  binder()->bind(copy, copy->index, copy->depth);

  return copy;
}

auto ASTRewriter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterAST* {
  auto copy = make_node<ConstraintTypeParameterAST>(arena());

  copy->symbol = ast->symbol;
  copy->depth = ast->depth;
  copy->index = ast->index;
  copy->typeConstraint = rewrite.typeConstraint(ast->typeConstraint);
  copy->ellipsisLoc = ast->ellipsisLoc;
  copy->identifierLoc = ast->identifierLoc;
  copy->equalLoc = ast->equalLoc;
  copy->typeId = rewrite.typeId(ast->typeId);
  copy->identifier = ast->identifier;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = make_node<DefaultFunctionBodyAST>(arena());

  copy->equalLoc = ast->equalLoc;
  copy->defaultLoc = ast->defaultLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = make_node<CompoundStatementFunctionBodyAST>(arena());

  copy->colonLoc = ast->colonLoc;

  for (auto memInitializerList = &copy->memInitializerList;
       auto node : ListView{ast->memInitializerList}) {
    auto value = rewrite.memInitializer(node);
    *memInitializerList = make_list_node(arena(), value);
    memInitializerList = &(*memInitializerList)->next;
  }

  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyAST* {
  auto copy = make_node<TryStatementFunctionBodyAST>(arena());

  copy->tryLoc = ast->tryLoc;
  copy->colonLoc = ast->colonLoc;

  for (auto memInitializerList = &copy->memInitializerList;
       auto node : ListView{ast->memInitializerList}) {
    auto value = rewrite.memInitializer(node);
    *memInitializerList = make_list_node(arena(), value);
    memInitializerList = &(*memInitializerList)->next;
  }

  copy->statement =
      ast_cast<CompoundStatementAST>(rewrite.statement(ast->statement));

  for (auto handlerList = &copy->handlerList;
       auto node : ListView{ast->handlerList}) {
    auto value = rewrite.handler(node);
    *handlerList = make_list_node(arena(), value);
    handlerList = &(*handlerList)->next;
  }

  return copy;
}

auto ASTRewriter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyAST* {
  auto copy = make_node<DeleteFunctionBodyAST>(arena());

  copy->equalLoc = ast->equalLoc;
  copy->deleteLoc = ast->deleteLoc;
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<SimpleRequirementAST>(arena());

  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<CompoundRequirementAST>(arena());

  copy->lbraceLoc = ast->lbraceLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->rbraceLoc = ast->rbraceLoc;
  copy->noexceptLoc = ast->noexceptLoc;
  copy->minusGreaterLoc = ast->minusGreaterLoc;
  copy->typeConstraint = rewrite.typeConstraint(ast->typeConstraint);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<TypeRequirementAST>(arena());

  copy->typenameLoc = ast->typenameLoc;
  copy->nestedNameSpecifier =
      rewrite.nestedNameSpecifier(ast->nestedNameSpecifier);
  copy->templateLoc = ast->templateLoc;
  copy->unqualifiedId = rewrite.unqualifiedId(ast->unqualifiedId);
  copy->semicolonLoc = ast->semicolonLoc;
  copy->isTemplateIntroduced = ast->isTemplateIntroduced;

  return copy;
}

auto ASTRewriter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementAST* {
  auto copy = make_node<NestedRequirementAST>(arena());

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = rewrite.expression(ast->expression);
  copy->semicolonLoc = ast->semicolonLoc;

  return copy;
}

}  // namespace cxx
