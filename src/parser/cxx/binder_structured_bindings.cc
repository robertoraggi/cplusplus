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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <format>
#include <vector>

namespace cxx {
namespace {
auto asExpression(NameIdAST* nameId) -> const Identifier* {
  return nameId ? nameId->identifier : nullptr;
}
}  // namespace

auto Binder::declareStructuredBindingEntity(
    SourceLocation loc, const Identifier* name, const DeclSpecs& specs,
    TokenKind refOp, ExpressionAST* initializer) -> InitDeclaratorAST* {
  if (!name || !initializer) return nullptr;

  auto ar = unit_->arena();

  auto nameId = NameIdAST::create(ar, name);
  nameId->identifierLoc = loc;

  auto idDeclarator = IdDeclaratorAST::create(ar);
  idDeclarator->unqualifiedId = nameId;

  auto declarator = DeclaratorAST::create(ar);
  declarator->coreDeclarator = idDeclarator;

  if (refOp != TokenKind::T_EOF_SYMBOL) {
    auto refOpAst = ReferenceOperatorAST::create(ar);
    refOpAst->refLoc = loc;
    refOpAst->refOp = refOp;
    declarator->ptrOpList = make_list_node<PtrOperatorAST>(ar, refOpAst);
  }

  Decl decl{specs, declarator};

  auto symbol = declareVariable(declarator, decl,
                                /*addSymbolToParentScope=*/true);
  if (!symbol) return nullptr;

  auto initDeclarator = InitDeclaratorAST::create(ar);
  initDeclarator->declarator = declarator;
  initDeclarator->initializer = initializer;
  initDeclarator->symbol = symbol;

  TypeChecker check{unit_};
  check.setScope(scope());
  check.setReportErrors(unit_->config().checkTypes);
  check.check_init_declarator(initDeclarator);

  return initDeclarator;
}

void Binder::bindStructuredBindings(StructuredBindingDeclarationAST* ast,
                                    const DeclSpecs& specs) {
  if (!ast || !ast->initializer) return;

  int count = 0;
  for (auto it = ast->bindingList; it; it = it->next) ++count;
  if (count == 0) return;

  auto ar = unit_->arena();

  const auto refOp = ast->refQualifierLoc
                         ? unit_->tokenKind(ast->refQualifierLoc)
                         : TokenKind::T_EOF_SYMBOL;

  auto eIdent =
      control()->getIdentifier(std::format("$e{}", ast->lbracketLoc.index()));

  auto initializerRefOp = refOp;
  if (initializerRefOp == TokenKind::T_EOF_SYMBOL &&
      type_cast<BoundedArrayType>(
          traits.remove_cv(traits.remove_reference(ast->initializer->type)))) {
    initializerRefOp = TokenKind::T_AMP;
  }

  auto eInitDeclarator = declareStructuredBindingEntity(
      ast->initializer->firstSourceLocation(), eIdent, specs, initializerRefOp,
      ast->initializer);
  if (!eInitDeclarator) return;
  ast->hiddenVariable = eInitDeclarator;

  auto eSymbol = symbol_cast<VariableSymbol>(eInitDeclarator->symbol);
  if (!eSymbol) return;

  if (isDependent(unit_, eSymbol->type())) {
    auto dependentType = control()->getTypeParameterType(0, 0, false);
    auto placeholderTail = &ast->bindingDeclaratorList;
    for (auto it = ast->bindingList; it; it = it->next) {
      auto name = asExpression(it->value);
      if (!name) continue;

      auto placeholder =
          control()->newVariableSymbol(scope(), it->value->identifierLoc);
      placeholder->setName(name);
      placeholder->setType(dependentType);
      scope()->addSymbol(placeholder);

      auto placeholderDeclarator = InitDeclaratorAST::create(ar);
      placeholderDeclarator->symbol = placeholder;

      *placeholderTail =
          make_list_node<InitDeclaratorAST>(ar, placeholderDeclarator);
      placeholderTail = &(*placeholderTail)->next;
    }
    return;
  }

  auto elementBaseType = traits.remove_reference(eSymbol->type());
  auto unqualifiedBaseType = traits.remove_cv(elementBaseType);
  auto baseCv = traits.get_cv_qualifiers(elementBaseType);

  auto buildEIdExpr = [&]() -> IdExpressionAST* {
    auto idExpr = IdExpressionAST::create(ar);
    idExpr->unqualifiedId = NameIdAST::create(ar, eIdent);
    idExpr->symbol = eSymbol;
    idExpr->type = elementBaseType;
    idExpr->valueCategory = ValueCategory::kLValue;
    return idExpr;
  };

  auto bindingDeclaratorListTail = &ast->bindingDeclaratorList;

  auto declareBinding = [&](NameIdAST* nameId, ExpressionAST* accessExpr) {
    auto name = asExpression(nameId);
    if (!name) return;

    if (!accessExpr || !accessExpr->type) {
      error(
          nameId->identifierLoc,
          std::format("cannot decompose initializer into '{}'", name->name()));
      return;
    }

    auto equalInit = EqualInitializerAST::create(ar);
    equalInit->expression = accessExpr;
    equalInit->valueCategory = accessExpr->valueCategory;
    equalInit->type = accessExpr->type;

    DeclSpecs bindingSpecs{unit_};
    bindingSpecs.setType(accessExpr->type);
    bindingSpecs.finish();

    const auto bindingRefOp =
        accessExpr->valueCategory == ValueCategory::kLValue
            ? TokenKind::T_AMP
            : TokenKind::T_AMP_AMP;

    auto bindingInitDeclarator = declareStructuredBindingEntity(
        nameId->identifierLoc, name, bindingSpecs, bindingRefOp, equalInit);
    if (!bindingInitDeclarator) return;

    *bindingDeclaratorListTail =
        make_list_node<InitDeclaratorAST>(ar, bindingInitDeclarator);
    bindingDeclaratorListTail = &(*bindingDeclaratorListTail)->next;
  };

  if (auto arrayType = type_cast<BoundedArrayType>(unqualifiedBaseType)) {
    if (static_cast<std::size_t>(count) != arrayType->size()) {
      error(ast->lbracketLoc,
            std::format("{} names provided for structured binding of array "
                        "with {} elements",
                        count, arrayType->size()));
      return;
    }

    auto elementType = traits.add_cv(arrayType->elementType(), baseCv);

    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      auto idxLiteral = IntLiteralExpressionAST::create(ar);
      idxLiteral->literal = control()->integerLiteral(std::to_string(index));
      idxLiteral->valueCategory = ValueCategory::kPrValue;
      idxLiteral->type = control()->getSizeType();

      auto subscript = SubscriptExpressionAST::create(ar);
      subscript->baseExpression = buildEIdExpr();
      subscript->indexExpression = idxLiteral;
      subscript->valueCategory = ValueCategory::kLValue;
      subscript->type = elementType;

      declareBinding(it->value, subscript);
    }
    return;
  }

  auto classType = type_cast<ClassType>(unqualifiedBaseType);
  if (!classType || !classType->symbol()) {
    error(ast->lbracketLoc,
          "cannot decompose a non-class, non-array structured binding "
          "initializer");
    return;
  }

  auto classSymbol = classType->symbol();
  (void)traits.requireCompleteClass(classSymbol);

  auto getIdent = control()->getIdentifier("get");

  Symbol* getCandidate = nullptr;
  for (auto current = scope(); current && !getCandidate;
       current = current->parent()) {
    for (auto candidate : current->find(getIdent)) {
      getCandidate = candidate;
      break;
    }
  }

  auto tryTupleGet = [&](int index) -> ExpressionAST* {
    auto idxLiteral = IntLiteralExpressionAST::create(ar);
    idxLiteral->literal = control()->integerLiteral(std::to_string(index));
    idxLiteral->valueCategory = ValueCategory::kPrValue;
    idxLiteral->type = control()->getSizeType();

    auto exprArg = ExpressionTemplateArgumentAST::create(ar);
    exprArg->expression = idxLiteral;

    auto templateId = SimpleTemplateIdAST::create(ar);
    templateId->identifier = getIdent;
    templateId->identifierLoc = ast->initializer->firstSourceLocation();
    templateId->templateArgumentList =
        make_list_node<TemplateArgumentAST>(ar, exprArg);

    auto calleeIdExpr = IdExpressionAST::create(ar);
    calleeIdExpr->unqualifiedId = templateId;
    calleeIdExpr->symbol = getCandidate;

    bind(calleeIdExpr);

    TypeChecker check{unit_};
    check.setScope(scope());
    check.setReportErrors(false);
    check.check(calleeIdExpr);

    auto callExpr = CallExpressionAST::create(ar);
    callExpr->baseExpression = calleeIdExpr;
    callExpr->expressionList = make_list_node<ExpressionAST>(
        ar, static_cast<ExpressionAST*>(buildEIdExpr()));
    check.check(callExpr);

    if (!callExpr->type) return nullptr;
    return callExpr;
  };

  bool tupleLikeOk = false;
  std::vector<ExpressionAST*> tupleAccess;

  if (getCandidate) {
    auto diagnosticsClient = unit_->diagnosticsClient();
    const auto savedBlockErrors =
        diagnosticsClient ? diagnosticsClient->blockErrors(true) : false;

    tupleLikeOk = true;
    for (int index = 0; index < count; ++index) {
      auto access = tryTupleGet(index);
      if (!access) {
        tupleLikeOk = false;
        break;
      }
      tupleAccess.push_back(access);
    }

    if (diagnosticsClient) diagnosticsClient->blockErrors(savedBlockErrors);
  }

  if (tupleLikeOk) {
    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      declareBinding(it->value, tupleAccess[static_cast<std::size_t>(index)]);
    }
    return;
  }

  std::vector<FieldSymbol*> fields;
  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (field && !field->isStatic()) fields.push_back(field);
  }

  if (static_cast<int>(fields.size()) != count) {
    error(ast->lbracketLoc,
          std::format("{} names provided for structured binding of type "
                      "with {} non-static data members",
                      count, fields.size()));
    return;
  }

  int index = 0;
  for (auto it = ast->bindingList; it; it = it->next, ++index) {
    auto field = fields[static_cast<std::size_t>(index)];
    auto memberType = traits.add_cv(field->type(), baseCv);

    auto memberExpr = MemberExpressionAST::create(ar);
    memberExpr->baseExpression = buildEIdExpr();
    memberExpr->accessOp = TokenKind::T_DOT;
    memberExpr->unqualifiedId =
        NameIdAST::create(ar, name_cast<Identifier>(field->name()));
    memberExpr->symbol = field;
    memberExpr->valueCategory = ValueCategory::kLValue;
    memberExpr->type = memberType;

    declareBinding(it->value, memberExpr);
  }
}
}  // namespace cxx
