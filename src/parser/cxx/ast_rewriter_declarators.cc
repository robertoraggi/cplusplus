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
#include <cxx/ast_interpreter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

#include <format>

namespace cxx {

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

auto ASTRewriter::requiresClause(RequiresClauseAST* ast) -> RequiresClauseAST* {
  if (!ast) return {};

  auto copy = RequiresClauseAST::create(arena());

  copy->requiresLoc = ast->requiresLoc;
  copy->expression = expression(ast->expression);

  return copy;
}

auto ASTRewriter::parameterDeclarationClause(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseAST* {
  if (!ast) return {};

  auto copy = ParameterDeclarationClauseAST::create(arena());

  binder().bind(copy);

  auto _ = Binder::ScopeGuard(&binder_);

  binder().setScope(copy->functionParametersSymbol);

  for (auto parameterDeclarationList = &copy->parameterDeclarationList;
       auto node : ListView{ast->parameterDeclarationList}) {
    auto paramDecl = ast_cast<ParameterDeclarationAST>(node);

    if (paramDecl && paramDecl->isPack) {
      ParameterPackSymbol* pack = nullptr;
      for (auto specNode : ListView{paramDecl->typeSpecifierList}) {
        pack = getTypeParameterPack(specNode);
        if (pack) break;
      }

      if (pack) {
        ParameterSymbol* originalParam = nullptr;
        if (ast->functionParametersSymbol) {
          for (auto member : ast->functionParametersSymbol->members()) {
            if (auto ps = symbol_cast<ParameterSymbol>(member)) {
              if (name_cast<Identifier>(ps->name()) == paramDecl->identifier) {
                originalParam = ps;
                break;
              }
            }
          }
        }

        auto savedParameterPack = parameterPack_;
        std::swap(parameterPack_, pack);

        auto funcParamPack = control()->newParameterPackSymbol(
            binder().scope(), SourceLocation{});

        int n = static_cast<int>(parameterPack_->elements().size());
        for (int i = 0; i < n; ++i) {
          std::optional<int> index{i};
          std::swap(elementIndex_, index);

          auto membersBefore = binder().scope()->members().size();

          auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
          if (value) value->isPack = false;
          *parameterDeclarationList = make_list_node(arena(), value);
          parameterDeclarationList = &(*parameterDeclarationList)->next;

          const auto& members = binder().scope()->members();
          if (members.size() > membersBefore) {
            funcParamPack->addElement(members.back());
          }

          std::swap(elementIndex_, index);
        }

        if (originalParam) {
          functionParamPacks_[originalParam] = funcParamPack;
        }

        std::swap(parameterPack_, pack);
        continue;
      }
    }

    auto value = ast_cast<ParameterDeclarationAST>(declaration(node));
    *parameterDeclarationList = make_list_node(arena(), value);
    parameterDeclarationList = &(*parameterDeclarationList)->next;
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

  copy->declarator = declarator(ast->declarator);

  auto decl = Decl{declSpecs, copy->declarator};

  auto type =
      getDeclaratorType(translationUnit(), copy->declarator, declSpecs.type());

  const auto addSymbolToParentScope =
      binder().instantiatingSymbol() != ast->symbol;

  // ### fix scope
  if (binder_.scope()->isClass()) {
    auto symbol = binder_.declareMemberSymbol(copy->declarator, decl);
    copy->symbol = symbol;
  } else {
    // todo: move to Binder
    if (auto declId = decl.declaratorId; declId) {
      if (decl.specs.isTypedef) {
        auto typedefSymbol = binder_.declareTypedef(copy->declarator, decl);
        copy->symbol = typedefSymbol;
      } else if (getFunctionPrototype(copy->declarator)) {
        auto functionSymbol = binder_.declareFunction(copy->declarator, decl);
        copy->symbol = functionSymbol;
      } else {
        auto variableSymbol = binder_.declareVariable(copy->declarator, decl,
                                                      addSymbolToParentScope);
        // variableSymbol->setTemplateDeclaration(templateHead);
        copy->symbol = variableSymbol;

        if (!addSymbolToParentScope) {
          auto templateVariable = symbol_cast<VariableSymbol>(ast->symbol);
          templateVariable->addSpecialization(templateArguments(),
                                              variableSymbol);
        }
      }
    }
  }

  copy->requiresClause = requiresClause(ast->requiresClause);
  copy->initializer = expression(ast->initializer);

  if (auto fieldSymbol = symbol_cast<FieldSymbol>(copy->symbol)) {
    if (copy->initializer) {
      fieldSymbol->setInitializer(copy->initializer);
    }
  } else if (auto variableSymbol = symbol_cast<VariableSymbol>(copy->symbol)) {
    auto typeChecker = TypeChecker{unit_};
    typeChecker.setScope(binder_.scope());
    typeChecker.check_init_declarator(copy);
  }

  return copy;
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
  copy->expression = rewrite.expression(ast->expression);
  copy->rparenLoc = ast->rparenLoc;

  return copy;
}

}  // namespace cxx