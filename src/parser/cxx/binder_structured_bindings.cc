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
#include <cxx/diagnostics_client.h>
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <format>
#include <optional>
#include <vector>

namespace cxx {
namespace {
auto asExpression(NameIdAST* nameId) -> const Identifier* {
  return nameId ? nameId->identifier : nullptr;
}

[[nodiscard]] auto enclosingGlobalScope(ScopeSymbol* scope) -> ScopeSymbol* {
  auto root = scope;
  while (root && root->parent()) root = root->parent();
  return root;
}
}  // namespace

auto Binder::declareStructuredBindingEntity(
    SourceLocation loc, const Identifier* name, const DeclSpecs& specs,
    TokenKind refOp, ExpressionAST* initializer, bool addSymbolToParentScope)
    -> InitDeclaratorAST* {
  if (!name) return nullptr;

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

  auto symbol = declareVariable(declarator, decl, addSymbolToParentScope);
  if (!symbol) return nullptr;

  auto initDeclarator = InitDeclaratorAST::create(ar);
  initDeclarator->declarator = declarator;
  initDeclarator->initializer = initializer;
  initDeclarator->symbol = symbol;

  if (initializer) {
    TypeChecker check{unit_};
    check.setScope(scope());
    check.setReportErrors(unit_->config().checkTypes);
    check.check_init_declarator(initDeclarator, nullptr);
  }

  return initDeclarator;
}

auto Binder::structuredBindingEntityName() -> const Identifier* {
  return control()->getIdentifier("$e");
}

void Binder::bindStructuredBindings(StructuredBindingDeclarationAST* ast,
                                    const DeclSpecs& specs) {
  if (!ast || !ast->initializer) return;

  int count = 0;
  for (auto it = ast->bindingList; it; it = it->next) ++count;
  if (count == 0) return;

  const auto refOp = ast->refQualifierLoc
                         ? unit_->tokenKind(ast->refQualifierLoc)
                         : TokenKind::T_EOF_SYMBOL;

  auto eIdent = structuredBindingEntityName();

  auto initializerRefOp = refOp;
  if (initializerRefOp == TokenKind::T_EOF_SYMBOL &&
      unqualified_cast<BoundedArrayType>(
          traits.remove_reference(ast->initializer->type))) {
    initializerRefOp = TokenKind::T_AMP;
  }

  auto eInitDeclarator = declareStructuredBindingEntity(
      ast->initializer->firstSourceLocation(), eIdent, specs, initializerRefOp,
      ast->initializer, false);
  if (!eInitDeclarator) return;
  ast->hiddenVariable = eInitDeclarator;

  auto eSymbol = symbol_cast<VariableSymbol>(eInitDeclarator->symbol);
  if (!eSymbol) return;

  decomposeStructuredBinding(ast, eSymbol);
}

void Binder::decomposeStructuredBinding(StructuredBindingDeclarationAST* ast,
                                        VariableSymbol* eSymbol) {
  int count = 0;
  for (auto it = ast->bindingList; it; it = it->next) ++count;
  if (count == 0) return;

  auto ar = unit_->arena();
  auto eIdent = name_cast<Identifier>(eSymbol->name());

  if (isDependent(unit_, eSymbol->type())) {
    auto dependentType = control()->getDependentType();
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
  auto baseCv = cv_qualifiers(elementBaseType);

  auto buildEIdExpr = [&](ValueCategory valueCategory) -> IdExpressionAST* {
    auto idExpr = IdExpressionAST::create(ar);
    idExpr->unqualifiedId = NameIdAST::create(ar, eIdent);
    idExpr->symbol = eSymbol;
    idExpr->type = elementBaseType;
    idExpr->valueCategory = valueCategory;
    return idExpr;
  };

  auto bindingDeclaratorListTail = &ast->bindingDeclaratorList;

  auto declareBinding = [&](NameIdAST* nameId, ExpressionAST* accessExpr,
                            const Type* declaredType = nullptr) {
    auto name = asExpression(nameId);
    if (!name) return;

    if (!accessExpr || !accessExpr->type) {
      error(
          nameId->identifierLoc,
          std::format("cannot decompose initializer into '{}'", name->name()));
      return;
    }

    if (!declaredType) declaredType = accessExpr->type;

    auto equalInit = EqualInitializerAST::create(ar);
    equalInit->expression = accessExpr;
    equalInit->valueCategory = accessExpr->valueCategory;
    equalInit->type = accessExpr->type;

    DeclSpecs bindingSpecs{unit_};
    bindingSpecs.setType(declaredType);
    bindingSpecs.finish();

    const auto bindingRefOp =
        accessExpr->valueCategory == ValueCategory::kLValue
            ? TokenKind::T_AMP
            : TokenKind::T_AMP_AMP;

    auto bindingInitDeclarator = declareStructuredBindingEntity(
        nameId->identifierLoc, name, bindingSpecs, bindingRefOp, equalInit,
        true);
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
      subscript->baseExpression = buildEIdExpr(ValueCategory::kLValue);
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

  auto indexLiteral = [&](int index) -> IntLiteralExpressionAST* {
    auto literal = IntLiteralExpressionAST::create(ar);
    literal->literal = control()->integerLiteral(std::to_string(index));
    literal->valueCategory = ValueCategory::kPrValue;
    literal->type = control()->getSizeType();
    return literal;
  };

  auto typeArgument = [&](const Type* type) -> TemplateArgumentAST* {
    auto typeId = TypeIdAST::create(ar);
    typeId->type = type;
    return TypeTemplateArgumentAST::create(ar, typeId);
  };

  auto valueArgument = [&](int index) -> TemplateArgumentAST* {
    auto argument = ExpressionTemplateArgumentAST::create(ar);
    argument->expression = indexLiteral(index);
    return argument;
  };

  auto standardLibraryClassTemplate = [&](std::string_view name) -> Symbol* {
    auto stdNamespace = symbol_cast<NamespaceSymbol>(qualifiedLookup(
        enclosingGlobalScope(scope()), control()->getIdentifier("std")));
    if (!stdNamespace) return nullptr;
    return qualifiedLookup(stdNamespace, control()->getIdentifier(name),
                           [](Symbol* s) { return is_type(s); });
  };

  auto instantiateStandardLibraryClass =
      [&](std::string_view name,
          List<TemplateArgumentAST*>* arguments) -> ClassSymbol* {
    auto primary = standardLibraryClassTemplate(name);
    if (!primary) return nullptr;
    auto instance =
        ASTRewriter::instantiate(unit_, arguments, primary, ast->lbracketLoc);
    auto instanceClass = symbol_cast<ClassSymbol>(instance);
    if (!instanceClass) return nullptr;
    (void)traits.requireCompleteClass(instanceClass);
    return symbol_cast<ClassSymbol>(instanceClass->resolvedDefinition());
  };

  auto tupleSizeClass = [&]() -> ClassSymbol* {
    auto arguments =
        make_list_node<TemplateArgumentAST>(ar, typeArgument(elementBaseType));
    auto sizeClass = instantiateStandardLibraryClass("tuple_size", arguments);
    if (!sizeClass || !sizeClass->isComplete()) return nullptr;
    if (!qualifiedLookup(sizeClass, control()->getIdentifier("value")))
      return nullptr;
    return sizeClass;
  };

  auto structuredBindingSize =
      [&](ClassSymbol* sizeClass) -> std::optional<std::intmax_t> {
    auto valueSymbol =
        qualifiedLookup(sizeClass, control()->getIdentifier("value"));
    if (!valueSymbol || !valueSymbol->type()) return std::nullopt;

    if (auto valueField = symbol_cast<FieldSymbol>(valueSymbol))
      ASTRewriter::completePendingFieldInitializer(unit_, valueField);

    auto valueExpr = IdExpressionAST::create(ar);
    valueExpr->unqualifiedId =
        NameIdAST::create(ar, control()->getIdentifier("value"));
    valueExpr->symbol = valueSymbol;
    valueExpr->type = valueSymbol->type();
    valueExpr->valueCategory = ValueCategory::kLValue;

    auto value = ASTInterpreter{unit_}.evaluate(valueExpr);
    if (!value) return std::nullopt;
    auto integer = std::get_if<std::intmax_t>(&*value);
    if (!integer || *integer < 0) return std::nullopt;
    return *integer;
  };

  auto structuredBindingElementType = [&](int index) -> const Type* {
    auto arguments =
        make_list_node<TemplateArgumentAST>(ar, valueArgument(index));
    arguments->next =
        make_list_node<TemplateArgumentAST>(ar, typeArgument(elementBaseType));
    auto elementClass =
        instantiateStandardLibraryClass("tuple_element", arguments);
    if (!elementClass) return nullptr;
    auto typeSymbol =
        qualifiedLookup(elementClass, control()->getIdentifier("type"),
                        [](Symbol* s) { return is_type(s); });
    if (!typeSymbol) return nullptr;
    return typeSymbol->type();
  };

  auto namesConstantIndexedTemplate = [](FunctionSymbol* function) {
    if (!function) return false;
    if (!function->templateDeclaration()) return false;
    auto parameters = template_parameters_of(function);
    if (!parameters) return false;
    const auto& members = parameters->members();
    if (members.empty()) return false;
    return symbol_cast<NonTypeParameterSymbol>(members.front()) != nullptr;
  };

  auto declaresConstantIndexedGet = [&](Symbol* candidate) {
    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(candidate)) {
      for (auto function : overloadSet->functions()) {
        if (namesConstantIndexedTemplate(function)) return true;
      }
      return false;
    }
    return namesConstantIndexedTemplate(symbol_cast<FunctionSymbol>(candidate));
  };

  const bool hasMemberGetTemplate =
      qualifiedLookup(classSymbol, getIdent, declaresConstantIndexedGet) !=
      nullptr;

  auto buildGetTemplateId = [&]() -> SimpleTemplateIdAST* {
    auto templateId = SimpleTemplateIdAST::create(ar);
    templateId->identifier = getIdent;
    templateId->identifierLoc = ast->lbracketLoc;
    return templateId;
  };

  auto tupleEntityValueCategory = ValueCategory::kXValue;
  if (type_cast<LvalueReferenceType>(eSymbol->type()))
    tupleEntityValueCategory = ValueCategory::kLValue;

  auto tupleGet = [&](int index, TypeChecker& check) -> ExpressionAST* {
    auto templateId = buildGetTemplateId();
    templateId->templateArgumentList =
        make_list_node<TemplateArgumentAST>(ar, valueArgument(index));

    auto callExpr = CallExpressionAST::create(ar);

    if (hasMemberGetTemplate) {
      auto memberExpr = MemberExpressionAST::create(ar);
      memberExpr->baseExpression = buildEIdExpr(tupleEntityValueCategory);
      memberExpr->accessOp = TokenKind::T_DOT;
      memberExpr->unqualifiedId = templateId;
      memberExpr->isTemplateIntroduced = true;
      check.check(memberExpr);
      callExpr->baseExpression = memberExpr;
    } else {
      auto calleeIdExpr = IdExpressionAST::create(ar);
      calleeIdExpr->unqualifiedId = templateId;
      bind(calleeIdExpr, true);
      check.check(calleeIdExpr);
      callExpr->baseExpression = calleeIdExpr;
      callExpr->expressionList = make_list_node<ExpressionAST>(
          ar,
          static_cast<ExpressionAST*>(buildEIdExpr(tupleEntityValueCategory)));
    }

    check.check(callExpr);

    if (!callExpr->type) return nullptr;
    return callExpr;
  };

  if (auto sizeClass = tupleSizeClass()) {
    auto tupleSize = structuredBindingSize(sizeClass);

    if (!tupleSize) {
      error(ast->lbracketLoc,
            std::format("'std::tuple_size<{}>::value' is not a non-negative "
                        "integral constant expression",
                        to_string(elementBaseType)));
      return;
    }

    if (count != *tupleSize) {
      error(ast->lbracketLoc,
            std::format("{} names provided for structured binding of type "
                        "with a structured binding size of {}",
                        count, *tupleSize));
      return;
    }

    TypeChecker check{unit_};
    check.setScope(scope());

    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      auto elementType = structuredBindingElementType(index);
      if (!elementType) {
        error(ast->lbracketLoc, std::format("no type named 'type' in "
                                            "'std::tuple_element<{}, {}>'",
                                            index, to_string(elementBaseType)));
        return;
      }
      declareBinding(it->value, tupleGet(index, check), elementType);
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
    memberExpr->baseExpression = buildEIdExpr(ValueCategory::kLValue);
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
