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
#include <cxx/views/symbols.h>

#include <format>
#include <optional>
#include <vector>

namespace cxx {
namespace {
auto asExpression(NameIdAST* nameId) -> const Identifier* {
  return nameId ? nameId->identifier : nullptr;
}

[[nodiscard]] auto declaresNonStaticDataMembers(ClassSymbol* classSymbol)
    -> bool {
  return !std::ranges::empty(views::members(classSymbol) |
                             views::non_static_fields);
}

[[nodiscard]] auto classesDeclaringNonStaticDataMembers(
    ClassSymbol* classSymbol) -> std::vector<ClassSymbol*> {
  std::vector<ClassSymbol*> owners;
  std::vector<ClassSymbol*> visited;
  std::vector<ClassSymbol*> pending{classSymbol->resolvedDefinition()};

  while (!pending.empty()) {
    auto current = pending.back();
    pending.pop_back();
    if (std::ranges::contains(visited, current)) continue;
    visited.push_back(current);

    if (declaresNonStaticDataMembers(current)) owners.push_back(current);

    for (auto base : current->baseClasses()) {
      if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol()))
        pending.push_back(baseClass->resolvedDefinition());
    }
  }

  return owners;
}
}  // namespace

auto Binder::declareStructuredBindingEntity(
    SourceLocation loc, const Identifier* name, const DeclSpecs& specs,
    TokenKind refOp, ExpressionAST* initializer, bool addSymbolToParentScope,
    const Type* declaredType) -> InitDeclaratorAST* {
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

  auto symbol =
      declareVariable(declarator, decl, addSymbolToParentScope, declaredType);
  if (!symbol) return nullptr;

  auto initDeclarator = InitDeclaratorAST::create(ar);
  initDeclarator->declarator = declarator;
  initDeclarator->initializer = initializer;
  initDeclarator->symbol = symbol;

  if (initializer) {
    TypeChecker check{unit_};
    check.setScope(scope());
    check.setReportErrors(unit_->config().checkTypes);
    check.check_init_declarator(initDeclarator, nullptr,
                                ArrayCopyPolicy::kElementwiseCopyAllowed);
  }

  return initDeclarator;
}

auto Binder::structuredBindingArrayCopyType(
    StructuredBindingDeclarationAST* ast, const Type* declaredType,
    const Type* initializerType) const -> const Type* {
  if (ast->refQualifierLoc) return nullptr;
  auto arrayType = traits.remove_reference(initializerType);
  if (!traits.is_array(arrayType)) return nullptr;
  return traits.add_cv(arrayType, cv_qualifiers(declaredType));
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

  auto eInitDeclarator = declareStructuredBindingEntity(
      ast->initializer->firstSourceLocation(), structuredBindingEntityName(),
      specs, refOp, ast->initializer, false,
      structuredBindingArrayCopyType(ast, specs.type(),
                                     ast->initializer->type));
  if (!eInitDeclarator) return;
  ast->hiddenVariable = eInitDeclarator;

  auto eSymbol = symbol_cast<VariableSymbol>(eInitDeclarator->symbol);
  if (!eSymbol) return;

  decomposeStructuredBinding(ast, eSymbol);
}

namespace {
[[nodiscard]] auto namesConstantIndexedTemplate(FunctionSymbol* function)
    -> bool {
  if (!function->templateDeclaration()) return false;
  auto parameters = template_parameters_of(function);
  if (!parameters) return false;
  const auto& members = parameters->members();
  if (members.empty()) return false;
  return symbol_cast<NonTypeParameterSymbol>(members.front()) != nullptr;
}

[[nodiscard]] auto declaresConstantIndexedGet(Symbol* candidate) -> bool {
  for (auto function : views::each_function(candidate)) {
    if (namesConstantIndexedTemplate(function)) return true;
  }
  return false;
}
}  // namespace

struct Binder::DecomposeStructuredBinding {
  Binder& binder;
  StructuredBindingDeclarationAST* ast;
  VariableSymbol* entity;
  Arena* arena;
  int count = 0;
  const Type* entityType = nullptr;
  List<InitDeclaratorAST*>** bindingTail = nullptr;

  DecomposeStructuredBinding(Binder& b, StructuredBindingDeclarationAST* a,
                             VariableSymbol* e)
      : binder(b),
        ast(a),
        entity(e),
        arena(b.unit_->arena()),
        entityType(b.traits.remove_reference(e->type())),
        bindingTail(&a->bindingDeclaratorList) {
    for (auto it = ast->bindingList; it; it = it->next) ++count;
  }

  [[nodiscard]] auto control() const -> Control* { return binder.control(); }

  void operator()() {
    if (count == 0) return;

    if (isDependent(binder.unit_, entity->type())) {
      declareDependentBindings();
      return;
    }

    auto unqualifiedType = binder.traits.remove_cv(entityType);

    if (auto arrayType = type_cast<BoundedArrayType>(unqualifiedType)) {
      decomposeArray(arrayType);
      return;
    }

    auto classType = type_cast<ClassType>(unqualifiedType);
    if (!classType || !classType->symbol()) {
      binder.error(ast->lbracketLoc,
                   "cannot decompose a non-class, non-array structured "
                   "binding initializer");
      return;
    }

    auto classSymbol = classType->symbol();
    (void)binder.traits.requireCompleteClass(classSymbol);

    if (auto sizeClass = tupleSizeClass()) {
      decomposeTupleLike(classSymbol, sizeClass);
      return;
    }

    decomposeDataMembers(classSymbol);
  }

  void declareDependentBindings() {
    auto dependentType = control()->getDependentType();
    for (auto it = ast->bindingList; it; it = it->next) {
      auto name = asExpression(it->value);
      if (!name) continue;

      auto placeholder = control()->newVariableSymbol(binder.scope(),
                                                      it->value->identifierLoc);
      placeholder->setName(name);
      placeholder->setType(dependentType);
      binder.scope()->addSymbol(placeholder);

      auto placeholderDeclarator = InitDeclaratorAST::create(arena);
      placeholderDeclarator->symbol = placeholder;
      appendBinding(placeholderDeclarator);
    }
  }

  void appendBinding(InitDeclaratorAST* declarator) {
    *bindingTail = make_list_node<InitDeclaratorAST>(arena, declarator);
    bindingTail = &(*bindingTail)->next;
  }

  [[nodiscard]] auto entityReference(ValueCategory valueCategory) const
      -> IdExpressionAST* {
    auto reference = IdExpressionAST::create(arena);
    reference->unqualifiedId =
        NameIdAST::create(arena, name_cast<Identifier>(entity->name()));
    reference->symbol = entity;
    reference->type = entityType;
    reference->valueCategory = valueCategory;
    return reference;
  }

  [[nodiscard]] auto indexLiteral(int index) const -> IntLiteralExpressionAST* {
    auto literal = IntLiteralExpressionAST::create(arena);
    literal->literal = control()->integerLiteral(std::to_string(index));
    literal->valueCategory = ValueCategory::kPrValue;
    literal->type = control()->getSizeType();
    return literal;
  }

  [[nodiscard]] auto typeArgument(const Type* type) const
      -> TemplateArgumentAST* {
    auto typeId = TypeIdAST::create(arena);
    typeId->type = type;
    return TypeTemplateArgumentAST::create(arena, typeId);
  }

  [[nodiscard]] auto valueArgument(int index) const -> TemplateArgumentAST* {
    auto argument = ExpressionTemplateArgumentAST::create(arena);
    argument->expression = indexLiteral(index);
    return argument;
  }

  void declareBinding(NameIdAST* nameId, ExpressionAST* access,
                      const Type* declaredType = nullptr) {
    auto name = asExpression(nameId);
    if (!name) return;

    if (!access || !access->type) {
      binder.error(
          nameId->identifierLoc,
          std::format("cannot decompose initializer into '{}'", name->name()));
      return;
    }

    if (!declaredType) declaredType = access->type;

    auto equalInit = EqualInitializerAST::create(arena);
    equalInit->expression = access;
    equalInit->valueCategory = access->valueCategory;
    equalInit->type = access->type;

    DeclSpecs bindingSpecs{binder.unit_};
    bindingSpecs.setType(declaredType);
    bindingSpecs.finish();

    const auto bindingRefOp = access->valueCategory == ValueCategory::kLValue
                                  ? TokenKind::T_AMP
                                  : TokenKind::T_AMP_AMP;

    auto bindingInitDeclarator = binder.declareStructuredBindingEntity(
        nameId->identifierLoc, name, bindingSpecs, bindingRefOp, equalInit,
        true);
    if (!bindingInitDeclarator) return;

    appendBinding(bindingInitDeclarator);
  }

  void decomposeArray(const BoundedArrayType* arrayType) {
    if (static_cast<std::size_t>(count) != arrayType->size()) {
      binder.error(ast->lbracketLoc,
                   std::format("{} names provided for structured binding of "
                               "array with {} elements",
                               count, arrayType->size()));
      return;
    }

    auto elementType = binder.traits.add_cv(arrayType->elementType(),
                                            cv_qualifiers(entityType));

    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      auto subscript = SubscriptExpressionAST::create(arena);
      subscript->baseExpression = entityReference(ValueCategory::kLValue);
      subscript->indexExpression = indexLiteral(index);
      subscript->valueCategory = ValueCategory::kLValue;
      subscript->type = elementType;

      declareBinding(it->value, subscript);
    }
  }

  [[nodiscard]] auto instantiateStandardLibraryClass(
      WellKnownName name, List<TemplateArgumentAST*>* arguments) const
      -> ClassSymbol* {
    auto primary = lookupStandardLibraryType(binder.unit_, name);
    if (!primary) return nullptr;
    auto instance = ASTRewriter::instantiate(binder.unit_, arguments, primary,
                                             ast->lbracketLoc);
    auto instanceClass = symbol_cast<ClassSymbol>(instance);
    if (!instanceClass) return nullptr;
    (void)binder.traits.requireCompleteClass(instanceClass);
    return symbol_cast<ClassSymbol>(instanceClass->resolvedDefinition());
  }

  [[nodiscard]] auto tupleSizeClass() const -> ClassSymbol* {
    auto arguments =
        make_list_node<TemplateArgumentAST>(arena, typeArgument(entityType));
    auto sizeClass =
        instantiateStandardLibraryClass(WellKnownName::T_TUPLE_SIZE, arguments);
    if (!sizeClass || !sizeClass->isComplete()) return nullptr;
    if (!qualifiedLookup(sizeClass, control()->getIdentifier("value")))
      return nullptr;
    return sizeClass;
  }

  [[nodiscard]] auto tupleSize(ClassSymbol* sizeClass) const
      -> std::optional<std::intmax_t> {
    auto valueName = control()->getIdentifier("value");
    auto valueSymbol = qualifiedLookup(sizeClass, valueName);
    if (!valueSymbol || !valueSymbol->type()) return std::nullopt;

    if (auto valueField = symbol_cast<FieldSymbol>(valueSymbol))
      ASTRewriter::requireFieldInitializer(binder.unit_, valueField);

    auto valueExpr = IdExpressionAST::create(arena);
    valueExpr->unqualifiedId = NameIdAST::create(arena, valueName);
    valueExpr->symbol = valueSymbol;
    valueExpr->type = valueSymbol->type();
    valueExpr->valueCategory = ValueCategory::kLValue;

    auto value = ASTInterpreter{binder.unit_}.evaluate(valueExpr);
    if (!value) return std::nullopt;
    auto integer = std::get_if<ConstInt>(&*value);
    if (!integer || integer->isNegative()) return std::nullopt;
    return integer->toIntMax();
  }

  [[nodiscard]] auto tupleElementType(int index) const -> const Type* {
    auto arguments =
        make_list_node<TemplateArgumentAST>(arena, valueArgument(index));
    arguments->next =
        make_list_node<TemplateArgumentAST>(arena, typeArgument(entityType));
    auto elementClass = instantiateStandardLibraryClass(
        WellKnownName::T_TUPLE_ELEMENT, arguments);
    if (!elementClass) return nullptr;
    auto typeSymbol =
        qualifiedLookup(elementClass, control()->getIdentifier("type"),
                        [](Symbol* s) { return is_type(s); });
    if (!typeSymbol) return nullptr;
    return typeSymbol->type();
  }

  [[nodiscard]] auto tupleEntityValueCategory() const -> ValueCategory {
    if (type_cast<LvalueReferenceType>(entity->type()))
      return ValueCategory::kLValue;
    return ValueCategory::kXValue;
  }

  [[nodiscard]] auto tupleGet(int index, bool hasMemberGetTemplate,
                              TypeChecker& check) -> ExpressionAST* {
    auto templateId = SimpleTemplateIdAST::create(arena);
    templateId->identifier = control()->getIdentifier("get");
    templateId->identifierLoc = ast->lbracketLoc;
    templateId->templateArgumentList =
        make_list_node<TemplateArgumentAST>(arena, valueArgument(index));

    auto callExpr = CallExpressionAST::create(arena);
    callExpr->lparenLoc = ast->lbracketLoc;
    callExpr->rparenLoc = ast->rbracketLoc;

    if (hasMemberGetTemplate) {
      auto memberExpr = MemberExpressionAST::create(arena);
      memberExpr->baseExpression = entityReference(tupleEntityValueCategory());
      memberExpr->accessOp = TokenKind::T_DOT;
      memberExpr->unqualifiedId = templateId;
      memberExpr->isTemplateIntroduced = true;
      callExpr->baseExpression = memberExpr;
      check.check(&callExpr->baseExpression);
    } else {
      auto calleeIdExpr = IdExpressionAST::create(arena);
      calleeIdExpr->unqualifiedId = templateId;
      binder.bind(calleeIdExpr, true);
      callExpr->baseExpression = calleeIdExpr;
      check.check(&callExpr->baseExpression);
      callExpr->expressionList = make_list_node<ExpressionAST>(
          arena, entityReference(tupleEntityValueCategory()));
    }

    ExpressionAST* result = callExpr;
    check.check(&result);

    if (!result->type) return nullptr;
    return result;
  }

  void decomposeTupleLike(ClassSymbol* classSymbol, ClassSymbol* sizeClass) {
    auto size = tupleSize(sizeClass);

    if (!size) {
      binder.error(ast->lbracketLoc,
                   std::format("'std::tuple_size<{}>::value' is not a "
                               "non-negative integral constant expression",
                               to_string(entityType)));
      return;
    }

    if (count != *size) {
      binder.error(ast->lbracketLoc,
                   std::format("{} names provided for structured binding of "
                               "type with a structured binding size of {}",
                               count, *size));
      return;
    }

    const bool hasMemberGetTemplate =
        qualifiedLookup(classSymbol, control()->getIdentifier("get"),
                        declaresConstantIndexedGet) != nullptr;

    TypeChecker check{binder.unit_};
    check.setScope(binder.scope());

    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      auto elementType = tupleElementType(index);
      if (!elementType) {
        binder.error(ast->lbracketLoc,
                     std::format("no type named 'type' in "
                                 "'std::tuple_element<{}, {}>'",
                                 index, to_string(entityType)));
        return;
      }
      declareBinding(it->value, tupleGet(index, hasMemberGetTemplate, check),
                     elementType);
    }
  }

  void decomposeDataMembers(ClassSymbol* classSymbol) {
    auto unqualifiedType = binder.traits.remove_cv(entityType);
    auto owners = classesDeclaringNonStaticDataMembers(classSymbol);
    if (owners.size() > 1) {
      binder.error(ast->lbracketLoc,
                   std::format("cannot decompose '{}': its non-static data "
                               "members are not all direct members of the "
                               "same class",
                               to_string(unqualifiedType)));
      return;
    }

    std::vector<FieldSymbol*> fields;
    if (!owners.empty()) {
      std::ranges::copy(
          views::members(owners.front()) | views::non_static_fields,
          std::back_inserter(fields));
    }

    if (std::ranges::any_of(fields, is_anonymous_union_member)) {
      binder.error(ast->lbracketLoc,
                   std::format("cannot decompose '{}': it has an anonymous "
                               "union member",
                               to_string(unqualifiedType)));
      return;
    }

    if (static_cast<int>(fields.size()) != count) {
      binder.error(ast->lbracketLoc,
                   std::format("{} names provided for structured binding of "
                               "type with {} non-static data members",
                               count, fields.size()));
      return;
    }

    TypeChecker check{binder.unit_};
    check.setScope(binder.scope());

    int index = 0;
    for (auto it = ast->bindingList; it; it = it->next, ++index) {
      auto field = fields[static_cast<std::size_t>(index)];

      auto memberExpr = MemberExpressionAST::create(arena);
      memberExpr->baseExpression = entityReference(ValueCategory::kLValue);
      memberExpr->accessOp = TokenKind::T_DOT;
      memberExpr->accessLoc = it->value->identifierLoc;
      auto memberName =
          NameIdAST::create(arena, name_cast<Identifier>(field->name()));
      memberName->identifierLoc = it->value->identifierLoc;
      memberExpr->unqualifiedId = memberName;

      ExpressionAST* access = memberExpr;
      check.check(&access);

      if (memberExpr->symbol != field) access = nullptr;

      declareBinding(it->value, access);
    }
  }
};

void Binder::decomposeStructuredBinding(StructuredBindingDeclarationAST* ast,
                                        VariableSymbol* entity) {
  DecomposeStructuredBinding{*this, ast, entity}();
}
}  // namespace cxx
