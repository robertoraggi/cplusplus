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

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <limits>
#include <map>

namespace cxx {
auto makeDefaultInitializer(TranslationUnit* unit, ExpressionAST* expression,
                            SourceLocation location, ScopeSymbol* scope)
    -> ExpressionAST* {
  if (!expression) return nullptr;
  auto result = DefaultInitializerExpressionAST::create(unit->arena());
  result->expression = expression;
  result->context.location = location;
  result->context.scope = scope;
  result->type = expression->type;
  result->valueCategory = expression->valueCategory;
  return result;
}

namespace {
auto makeClassConstruction(TranslationUnit* unit, const Type* type,
                           FunctionSymbol* constructor,
                           BracedInitListAST* arguments)
    -> BracedTypeConstructionAST* {
  auto alias = unit->control()->newTypeAliasSymbol(nullptr, {});
  alias->setType(type);
  auto specifier = NamedTypeSpecifierAST::create(unit->arena());
  specifier->symbol = alias;
  auto construction = BracedTypeConstructionAST::create(unit->arena());
  construction->typeSpecifier = specifier;
  construction->bracedInitList = arguments;
  construction->constructorSymbol = constructor;
  construction->type = type;
  construction->valueCategory = ValueCategory::kPrValue;
  return construction;
}
}  // namespace

auto InitializedEntity::variable(const Type* type, Symbol* symbol,
                                 SourceLocation location) -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kVariable;
  entity.type_ = type;
  entity.symbol_ = symbol;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::member(const Type* type, Symbol* symbol,
                               SourceLocation location) -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kMember;
  entity.type_ = type;
  entity.symbol_ = symbol;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::base(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kBase;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::arrayElement(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kArrayElement;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::parameter(const Type* type, Symbol* symbol,
                                  SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kParameter;
  entity.type_ = type;
  entity.symbol_ = symbol;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::returnObject(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kReturnObject;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::exceptionObject(const Type* type,
                                        SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kExceptionObject;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::temporary(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kTemporary;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::newObject(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kNewObject;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::delegating(const Type* type, SourceLocation location)
    -> InitializedEntity {
  InitializedEntity entity;
  entity.kind_ = InitializedEntityKind::kDelegating;
  entity.type_ = type;
  entity.location_ = location;
  return entity;
}

auto InitializedEntity::description() const -> std::string {
  switch (kind_) {
    case InitializedEntityKind::kVariable:
      if (symbol_ && symbol_->name())
        return std::format("variable '{}'", to_string(symbol_->name()));
      return "variable";
    case InitializedEntityKind::kMember:
      if (symbol_ && symbol_->name())
        return std::format("member '{}'", to_string(symbol_->name()));
      return std::format("anonymous member of type '{}'", to_string(type_));
    case InitializedEntityKind::kBase:
      return std::format("base class '{}'", to_string(type_));
    case InitializedEntityKind::kArrayElement:
      return std::format("array element of type '{}'", to_string(type_));
    case InitializedEntityKind::kParameter:
      if (symbol_ && symbol_->name())
        return std::format("parameter '{}'", to_string(symbol_->name()));
      return "parameter";
    case InitializedEntityKind::kReturnObject:
      return "return value";
    case InitializedEntityKind::kExceptionObject:
      return "exception object";
    case InitializedEntityKind::kNewObject:
      return std::format("allocated object of type '{}'", to_string(type_));
    case InitializedEntityKind::kDelegating:
      return std::format("delegating constructor of '{}'", to_string(type_));
    case InitializedEntityKind::kTemporary:
      break;
  }
  return std::format("temporary of type '{}'", to_string(type_));
}

auto Initializer::withArgumentList(ExpressionAST* node,
                                   List<ExpressionAST*>** arguments)
    -> Initializer {
  Initializer initializer{node};
  initializer.argumentList_ = arguments;
  return initializer;
}

auto Initializer::stripImplicitCasts(ExpressionAST* expr) -> ExpressionAST* {
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr))
    expr = cast->expression;
  return expr;
}

auto Initializer::stripped() const -> ExpressionAST* {
  auto expr = stripImplicitCasts(node_);
  while (auto initializer = ast_cast<DefaultInitializerExpressionAST>(expr))
    expr = stripImplicitCasts(initializer->expression);
  return expr;
}

auto Initializer::unwrapEqual() const -> ExpressionAST* {
  auto expr = stripped();
  if (auto equal = ast_cast<EqualInitializerAST>(expr))
    return stripImplicitCasts(equal->expression);
  return expr;
}

auto Initializer::form() const -> InitializerForm {
  if (!node_)
    return argumentList_ ? InitializerForm::kParen : InitializerForm::kNone;
  auto expr = stripped();
  if (ast_cast<ParenInitializerAST>(expr)) return InitializerForm::kParen;
  if (ast_cast<BracedInitListAST>(expr)) return InitializerForm::kList;
  if (auto equal = ast_cast<EqualInitializerAST>(expr)) {
    if (ast_cast<BracedInitListAST>(stripImplicitCasts(equal->expression)))
      return InitializerForm::kList;
    return InitializerForm::kEqual;
  }
  return InitializerForm::kExpression;
}

auto Initializer::clause() const -> ExpressionAST* {
  if (auto equal = ast_cast<EqualInitializerAST>(node_))
    return equal->expression;
  return node_;
}

auto Initializer::bracedInitList() const -> BracedInitListAST* {
  if (!node_) return nullptr;
  return ast_cast<BracedInitListAST>(unwrapEqual());
}

auto Initializer::initializationKind() const -> InitializationKind {
  switch (form()) {
    case InitializerForm::kParen:
      return InitializationKind::kDirectInitialization;
    case InitializerForm::kList:
      if (ast_cast<BracedInitListAST>(stripped()))
        return InitializationKind::kDirectListInitialization;
      return InitializationKind::kCopyListInitialization;
    case InitializerForm::kNone:
    case InitializerForm::kEqual:
    case InitializerForm::kExpression:
      break;
  }
  return InitializationKind::kCopyInitialization;
}

auto Initializer::singleExpression() const -> ExpressionAST* {
  if (!node_) return nullptr;
  auto expr = unwrapEqual();
  if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
    if (paren->expressionList && !paren->expressionList->next)
      return paren->expressionList->value;
    return nullptr;
  }
  if (ast_cast<BracedInitListAST>(expr)) return nullptr;
  return expr;
}

auto Initializer::arguments() const -> std::vector<ExpressionAST*> {
  std::vector<ExpressionAST*> args;
  if (!node_) {
    if (argumentList_)
      for (auto it = *argumentList_; it; it = it->next)
        args.push_back(it->value);
    return args;
  }
  auto expr = unwrapEqual();
  if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
    for (auto it = paren->expressionList; it; it = it->next)
      args.push_back(it->value);
  } else if (auto braced = ast_cast<BracedInitListAST>(expr)) {
    for (auto it = braced->expressionList; it; it = it->next)
      args.push_back(it->value);
  } else if (expr) {
    args.push_back(expr);
  }
  return args;
}

auto Initializer::expressionListSlot() const -> List<ExpressionAST*>** {
  if (!node_) return argumentList_;
  auto expr = unwrapEqual();
  if (auto paren = ast_cast<ParenInitializerAST>(expr))
    return &paren->expressionList;
  if (auto braced = ast_cast<BracedInitListAST>(expr))
    return &braced->expressionList;
  return nullptr;
}

class InitializerOperand {
 public:
  InitializerOperand(ExpressionAST* initializer, ExpressionAST* expression)
      : initializer_(initializer),
        expression_(expression),
        slot_(Initializer{initializer}.conversionTarget()) {}

  [[nodiscard]] auto operand() -> ExpressionAST*& {
    return slot_ ? *slot_ : expression_;
  }

  [[nodiscard]] auto result() const -> ExpressionAST* {
    return slot_ ? initializer_ : expression_;
  }

 private:
  ExpressionAST* initializer_;
  ExpressionAST* expression_;
  ExpressionAST** slot_;
};

auto Initializer::conversionTarget() const -> ExpressionAST** {
  if (!node_) return nullptr;
  auto expr = stripped();
  if (auto equal = ast_cast<EqualInitializerAST>(expr))
    return &equal->expression;
  if (auto paren = ast_cast<ParenInitializerAST>(node_)) {
    if (paren->expressionList && !paren->expressionList->next)
      return &paren->expressionList->value;
  }
  return nullptr;
}

auto memInitializerClause(Arena* arena, MemInitializerAST* memInitializer)
    -> ExpressionAST* {
  if (auto braced = ast_cast<BracedMemInitializerAST>(memInitializer))
    return braced->bracedInitList;
  auto paren = ast_cast<ParenMemInitializerAST>(memInitializer);
  if (!paren) return nullptr;
  return ParenInitializerAST::create(arena, paren->lparenLoc,
                                     paren->expressionList, paren->rparenLoc,
                                     ValueCategory::kPrValue, nullptr);
}

auto memInitializerListSlot(MemInitializerAST* memInitializer)
    -> List<ExpressionAST*>** {
  if (auto paren = ast_cast<ParenMemInitializerAST>(memInitializer))
    return &paren->expressionList;
  if (auto braced = ast_cast<BracedMemInitializerAST>(memInitializer);
      braced && braced->bracedInitList)
    return &braced->bracedInitList->expressionList;
  return nullptr;
}

auto memInitializerArgumentSlots(MemInitializerAST* memInitializer)
    -> std::vector<ExpressionAST**> {
  std::vector<ExpressionAST**> args;
  auto slot = memInitializerListSlot(memInitializer);
  if (!slot) return args;
  for (auto it = *slot; it; it = it->next) args.push_back(&it->value);
  return args;
}

auto memInitializerId(MemInitializerAST* memInitializer) -> UnqualifiedIdAST* {
  if (auto paren = ast_cast<ParenMemInitializerAST>(memInitializer))
    return paren->unqualifiedId;
  if (auto braced = ast_cast<BracedMemInitializerAST>(memInitializer))
    return braced->unqualifiedId;
  return nullptr;
}

auto constantExpressionTarget(ExpressionAST*& initializer) -> ExpressionAST** {
  if (!initializer) return nullptr;
  if (auto equal = ast_cast<EqualInitializerAST>(initializer))
    return &equal->expression;
  if (ast_cast<ParenInitializerAST>(initializer)) return nullptr;
  if (ast_cast<BracedInitListAST>(initializer)) return nullptr;
  return &initializer;
}

void Initializer::propagateType() const {
  auto expr = stripped();
  ExpressionAST* wrapped = nullptr;
  if (auto equal = ast_cast<EqualInitializerAST>(expr))
    wrapped = equal->expression;
  else if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
    if (paren->expressionList && !paren->expressionList->next)
      wrapped = paren->expressionList->value;
  }
  if (!wrapped || !expr) return;
  expr->type = wrapped->type;
  expr->valueCategory = wrapped->valueCategory;
}

auto makeParenInitializer(Arena* arena, SourceLocation location,
                          List<ExpressionAST*>* arguments)
    -> ParenInitializerAST* {
  return ParenInitializerAST::create(arena, location, arguments, location,
                                     ValueCategory::kPrValue, nullptr);
}

InitContext::InitContext(TypeChecker& checker)
    : checker(checker),
      unit(checker.translationUnit()),
      control(checker.translationUnit()->control()),
      traits(checker.translationUnit()->typeTraits()) {}

auto InitContext::isCxx() const -> bool {
  return unit->language() == LanguageKind::kCXX;
}

void InitContext::error(SourceLocation loc, std::string message) {
  checker.error(loc, std::move(message));
}

void InitContext::warning(SourceLocation loc, std::string message) {
  checker.warning(loc, std::move(message));
}

auto InitContext::initializesFromSameTypePrvalue(ExpressionAST* expr,
                                                 const Type* targetType) const
    -> bool {
  if (!expr || !expr->type || !is_prvalue(expr)) return false;
  if (!traits.is_class(targetType)) return false;
  return traits.is_same(traits.remove_cv(expr->type),
                        traits.remove_cv(targetType));
}

auto InitContext::isTargetTypeUnresolved(const Type* type) const -> bool {
  if (!type) return true;
  if (isDependent(unit, type)) return true;
  return containsPlaceholderType(type);
}

auto isWholeArrayCopy(const TypeTraits& traits, ExpressionAST* expression,
                      const Type* arrayType) -> bool {
  if (!expression || !expression->type) return false;
  if (!traits.is_array(arrayType)) return false;
  if (ast_cast<BracedInitListAST>(Initializer{expression}.clause()))
    return false;
  if (ast_cast<StringLiteralExpressionAST>(Initializer{expression}.clause()))
    return false;
  return traits.remove_cv(expression->type) == traits.remove_cv(arrayType);
}

namespace {
void applyInitializerConversions(
    TypeChecker& checker, const Initializer& initializer,
    const std::vector<ImplicitConversionSequence>& conversions) {
  if (auto slot = initializer.expressionListSlot()) {
    std::size_t index = 0;
    for (auto it = *slot; it && index < conversions.size();
         it = it->next, ++index)
      checker.applyImplicitConversion(conversions[index], it->value);
    return;
  }
  if (conversions.empty()) return;
  if (auto target = initializer.conversionTarget()) {
    checker.applyImplicitConversion(conversions[0], *target);
    return;
  }
  if (auto node = initializer.node())
    checker.applyImplicitConversion(conversions[0], node);
}

struct AggregateInitGuard {
  AggregateInitGuard(const AggregateInitGuard&) = delete;
  auto operator=(const AggregateInitGuard&) -> AggregateInitGuard& = delete;

  TypeChecker& checker;
  ClassSymbol* classSymbol;
  bool entered;

  AggregateInitGuard(TypeChecker& checker, ClassSymbol* classSymbol)
      : checker(checker),
        classSymbol(classSymbol),
        entered(checker.enterAggregateInitialization(classSymbol)) {}

  ~AggregateInitGuard() {
    if (entered) checker.leaveAggregateInitialization(classSymbol);
  }

  [[nodiscard]] explicit operator bool() const { return entered; }
};

[[nodiscard]] auto checkStringLiteralInitialization(InitContext& ctx,
                                                    const Type* destinationType,
                                                    ExpressionAST* source)
    -> bool {
  auto init = stringLiteralInitialization(ctx.traits, ctx.isCxx(),
                                          destinationType, source);
  if (!init) return false;

  if (!init->compatible) {
    ctx.error(source->firstSourceLocation(),
              std::format("cannot initialize an array of '{}' with a string "
                          "literal of element type '{}'",
                          to_string(init->destinationElementType),
                          to_string(init->sourceElementType)));
    return true;
  }

  if (init->tooLong())
    ctx.error(source->firstSourceLocation(),
              "initializer-string for char array is too long");

  return true;
}

struct ElementInitChecker {
  InitContext& ctx;

  explicit ElementInitChecker(InitContext& ctx) : ctx(ctx) {}

  void check(ExpressionAST*& expr, const Type* targetType,
             std::string errorMessage,
             InitializationKind initializationKind =
                 InitializationKind::kCopyListInitialization) {
    if (isUntypedAfterError(expr)) return;

    if (ctx.traits.is_array(targetType)) {
      checkArrayElementInit(expr, targetType, std::move(errorMessage),
                            initializationKind);
      return;
    }

    if (ctx.traits.is_lvalue_reference(targetType))
      stripLvalueConversions(expr);

    auto source = expr;
    if (!ctx.checker.implicit_conversion(expr, targetType,
                                         initializationKind)) {
      ctx.error(expr->firstSourceLocation(), std::move(errorMessage));
    } else if (isListInitialization(initializationKind)) {
      diagnoseNarrowingListElement(ctx, source, targetType);
    }
  }

  [[nodiscard]] auto checkClassElementInit(ExpressionAST*& expr,
                                           const Type* targetType) -> bool;

 private:
  void checkArrayElementInit(ExpressionAST*& expr, const Type* targetType,
                             std::string errorMessage,
                             InitializationKind initializationKind) {
    if (checkStringLiteralInitialization(ctx, targetType, expr)) return;

    auto elemType =
        ctx.traits.remove_cv(ctx.traits.get_element_type(targetType));
    check(expr, elemType, std::move(errorMessage), initializationKind);
  }

  void stripLvalueConversions(ExpressionAST*& expr) {
    while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
      if (cast->castKind != ImplicitCastKind::kIdentity &&
          cast->castKind != ImplicitCastKind::kLValueToRValueConversion)
        break;
      if (!cast->expression) break;
      expr = cast->expression;
    }
  }
};

auto ElementInitChecker::checkClassElementInit(ExpressionAST*& expr,
                                               const Type* targetType) -> bool {
  if (!ctx.isCxx()) return false;
  if (!expr) return false;
  if (!expr->type) {
    if (!ast_cast<BracedInitListAST>(expr)) return false;
  }
  if (!ctx.traits.is_class(targetType)) return false;
  if (ctx.initializesFromSameTypePrvalue(expr, targetType)) return true;
  if (isDependent(ctx.unit, targetType)) return false;
  if (expr->type) {
    if (isDependent(ctx.unit, expr->type)) return false;
  }

  auto classType = unqualified_cast<ClassType>(targetType);
  if (!classType || !classType->symbol()) return false;
  if (!classType->symbol()->resolvedDefinition()->isComplete()) return false;

  auto arena = ctx.unit->arena();

  auto equal = EqualInitializerAST::create(arena);
  equal->expression = expr;

  ExpressionAST* initializer = equal;
  auto constructor = ctx.checker.check_class_initializer(
      targetType, initializer, expr->firstSourceLocation());

  if (!constructor) return false;

  auto arguments = BracedInitListAST::create(arena);
  if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
    arguments->expressionList = paren->expressionList;
  } else if (auto braced = Initializer{initializer}.bracedInitList()) {
    arguments->expressionList = braced->expressionList;
  } else {
    arguments->expressionList =
        make_list_node<ExpressionAST>(arena, equal->expression);
  }

  auto construction =
      makeClassConstruction(ctx.unit, targetType, constructor, arguments);

  expr = construction;
  return true;
}

[[nodiscard]] auto describeAggregateElement(const TypeTraits& traits,
                                            Symbol* element) -> std::string {
  if (!element) return "array element";
  if (auto field = symbol_cast<FieldSymbol>(element)) {
    if (field->name())
      return std::format("member '{}'", to_string(field->name()));
    return std::format("anonymous member of type '{}'",
                       to_string(traits.aggregate_element_type(element)));
  }
  return std::format("base class '{}'",
                     to_string(traits.aggregate_element_type(element)));
}

class AggregateInitializerBuilder {
 public:
  AggregateInitializerBuilder(TranslationUnit* unit, InitContext* ctx)
      : unit_(unit),
        traits_(unit->typeTraits()),
        conversions_(unit, unit->language() != LanguageKind::kCXX),
        ctx_(ctx) {}

  [[nodiscard]] auto build(const Type* aggregateType, BracedInitListAST* ast)
      -> std::optional<AggregateInitializerPlan>;

 private:
  [[nodiscard]] auto isCxx() const -> bool {
    return unit_->language() == LanguageKind::kCXX;
  }

  [[nodiscard]] auto resolving() const -> bool { return ctx_ != nullptr; }

  void error(SourceLocation location, std::string message) {
    if (ctx_) ctx_->error(location, std::move(message));
  }

  void warning(SourceLocation location, std::string message) {
    if (ctx_) ctx_->warning(location, std::move(message));
  }

  [[nodiscard]] auto shapeOf(const Type* type, AggregateInitializerPlan& plan)
      -> bool;

  [[nodiscard]] auto elementTypeAt(const AggregateInitializerPlan& plan,
                                   std::size_t index) -> const Type*;

  [[nodiscard]] static auto elementsToInitialize(
      const AggregateInitializerPlan& plan) -> std::size_t;

  [[nodiscard]] static auto excessElementsMessage(
      const AggregateInitializerPlan& plan) -> const char*;

  [[nodiscard]] auto appertains(ExpressionAST* clause, const Type* elementType)
      -> bool;

  [[nodiscard]] auto elideBraces(const Type* elementType,
                                 List<ExpressionAST*>*& it)
      -> BracedInitListAST*;

  struct DesignatedTarget {
    std::size_t index = 0;
    Symbol* member = nullptr;
  };

  [[nodiscard]] auto resolveDesignator(
      const AggregateInitializerPlan& plan, const Type* aggregateType,
      DesignatedInitializerClauseAST* designated)
      -> std::optional<DesignatedTarget>;

  void collectDeclaredMembers(ClassSymbol* classSymbol,
                              const Identifier* identifier,
                              std::vector<FieldSymbol*>& found) const;

  [[nodiscard]] auto lookupDesignatedMembers(ClassSymbol* classSymbol,
                                             const Identifier* identifier)
      -> std::vector<FieldSymbol*>;

  [[nodiscard]] auto associatedElementIndex(
      const std::vector<Symbol*>& elements, Symbol* member) const
      -> std::optional<std::size_t>;

  [[nodiscard]] auto declaresMemberOf(ClassSymbol* classSymbol,
                                      ClassSymbol* owner) const -> bool;

  struct DesignatedClause {
    DesignatedInitializerClauseAST* clause = nullptr;
    Symbol* member = nullptr;
  };

  struct PlacedClause {
    std::size_t index = 0;
    ExpressionAST* clause = nullptr;
    DesignatedInitializerClauseAST* designated = nullptr;
    Symbol* member = nullptr;
    bool elided = false;
  };

  [[nodiscard]] static auto remainingDesignators(Symbol* element,
                                                 const DesignatedClause& clause)
      -> List<DesignatorAST*>*;

  [[nodiscard]] auto mergeDesignatedClauses(
      Symbol* element, const std::vector<DesignatedClause>& clauses)
      -> ExpressionAST*;

  [[nodiscard]] auto selectElementInitializer(
      Symbol* element, const std::vector<PlacedClause>& placements)
      -> ExpressionAST*;

  [[nodiscard]] auto spliceSubobjectOverrides(
      Symbol* element, ExpressionAST* initializer,
      const std::vector<DesignatedClause>& overrides) -> ExpressionAST*;

  TranslationUnit* unit_;
  TypeTraits traits_;
  StandardConversion conversions_;
  InitContext* ctx_;
  std::vector<const Type*> elidedAggregates_;
};

auto AggregateInitializerBuilder::shapeOf(const Type* type,
                                          AggregateInitializerPlan& plan)
    -> bool {
  if (!type) return false;
  type = traits_.remove_cv(type);

  if (traits_.is_array(type)) {
    plan.arrayElementType = traits_.remove_cv(traits_.get_element_type(type));
    if (auto bounded = type_cast<BoundedArrayType>(type))
      plan.elementCount = bounded->size();
    else
      plan.elementCount = std::numeric_limits<std::size_t>::max();
    return plan.arrayElementType != nullptr;
  }

  if (auto vectorType = type_cast<VectorType>(type)) {
    plan.arrayElementType = traits_.remove_cv(vectorType->elementType());
    plan.elementCount = vectorType->elementCount();
    plan.isVector = true;
    return plan.arrayElementType != nullptr;
  }

  if (auto complexType = type_cast<ComplexType>(type)) {
    plan.arrayElementType = traits_.remove_cv(complexType->elementType());
    plan.elementCount = 2;
    return plan.arrayElementType != nullptr;
  }

  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return false;

  auto classSymbol = classType->symbol()->resolvedDefinition();
  traits_.requireCompleteClass(classSymbol);
  if (!classSymbol || !classSymbol->isComplete()) return false;

  plan.isUnion = classSymbol->isUnion();
  plan.elements = traits_.aggregate_elements(classSymbol);
  plan.elementCount = plan.elements.size();
  return true;
}

auto AggregateInitializerBuilder::elementTypeAt(
    const AggregateInitializerPlan& plan, std::size_t index) -> const Type* {
  if (plan.arrayElementType) return plan.arrayElementType;
  if (index >= plan.elements.size()) return nullptr;
  return traits_.aggregate_element_type(plan.elements[index]);
}

auto AggregateInitializerBuilder::elementsToInitialize(
    const AggregateInitializerPlan& plan) -> std::size_t {
  if (!plan.isUnion) return plan.elementCount;
  if (plan.elementCount == 0) return 0;
  return 1;
}

auto hasUnexpandedPackExpansion(BracedInitListAST* ast) -> bool {
  if (!ast) return false;
  for (auto clause : ListView{ast->expressionList}) {
    if (ast_cast<PackExpansionExpressionAST>(clause)) return true;
  }
  return false;
}

auto hasTypeDependentInitializerClause(TranslationUnit* unit,
                                       BracedInitListAST* ast) -> bool {
  if (!ast) return false;
  for (auto clause : ListView{ast->expressionList}) {
    if (!clause) continue;
    if (ast_cast<BracedInitListAST>(clause)) continue;
    if (ast_cast<DesignatedInitializerClauseAST>(clause)) continue;
    if (!clause->type) continue;
    if (isDependent(unit, clause->type)) return true;
  }
  return false;
}

auto AggregateInitializerBuilder::excessElementsMessage(
    const AggregateInitializerPlan& plan) -> const char* {
  if (plan.isVector) return "excess elements in vector initializer";
  if (plan.arrayElementType) return "excess elements in array initializer";
  if (plan.isUnion) return "excess elements in union initializer";
  return "excess elements in struct initializer";
}

auto AggregateInitializerBuilder::appertains(ExpressionAST* clause,
                                             const Type* elementType) -> bool {
  if (!clause || !elementType) return true;
  if (ast_cast<BracedInitListAST>(clause)) return true;
  if (ast_cast<DesignatedInitializerClauseAST>(clause)) return true;
  if (ast_cast<PackExpansionExpressionAST>(clause)) return true;
  if (!traits_.is_aggregate(elementType)) return true;
  if (std::ranges::contains(elidedAggregates_, traits_.remove_cv(elementType)))
    return true;

  AggregateInitializerPlan shape;
  if (!shapeOf(elementType, shape)) return true;
  if (shape.elementCount == 0) return true;

  if (!clause->type) return true;
  if (auto stringInit =
          stringLiteralInitialization(traits_, isCxx(), elementType, clause);
      stringInit && stringInit->compatible)
    return true;
  if (traits_.is_compatible(clause->type, elementType)) return true;

  return bool(conversions_.computeConversionSequence(clause, elementType));
}

auto AggregateInitializerBuilder::elideBraces(const Type* elementType,
                                              List<ExpressionAST*>*& it)
    -> BracedInitListAST* {
  auto pool = unit_->arena();
  auto synthetic = BracedInitListAST::create(pool);
  synthetic->lbraceLoc = it->value->firstSourceLocation();
  synthetic->rbraceLoc = it->value->lastSourceLocation();

  auto tail = &synthetic->expressionList;
  auto append = [&](ExpressionAST* clause) {
    *tail = make_list_node<ExpressionAST>(pool, clause);
    tail = &(*tail)->next;
  };

  AggregateInitializerPlan shape;
  if (!shapeOf(elementType, shape)) return synthetic;

  const auto count = elementsToInitialize(shape);

  elidedAggregates_.push_back(traits_.remove_cv(elementType));

  for (std::size_t index = 0; index < count && it; ++index) {
    auto subElementType = elementTypeAt(shape, index);
    if (appertains(it->value, subElementType)) {
      synthetic->rbraceLoc = it->value->lastSourceLocation();
      append(it->value);
      it = it->next;
      continue;
    }
    auto nested = elideBraces(subElementType, it);
    synthetic->rbraceLoc = nested->rbraceLoc;
    append(nested);
  }

  elidedAggregates_.pop_back();

  return synthetic;
}

void AggregateInitializerBuilder::collectDeclaredMembers(
    ClassSymbol* classSymbol, const Identifier* identifier,
    std::vector<FieldSymbol*>& found) const {
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->name() == identifier) {
      found.push_back(field);
      continue;
    }
    if (field->name()) continue;
    auto classType = unqualified_cast<ClassType>(field->type());
    if (!classType || !classType->symbol()) continue;
    collectDeclaredMembers(classType->symbol()->resolvedDefinition(),
                           identifier, found);
  }
}

auto AggregateInitializerBuilder::lookupDesignatedMembers(
    ClassSymbol* classSymbol, const Identifier* identifier)
    -> std::vector<FieldSymbol*> {
  std::vector<FieldSymbol*> found;
  collectDeclaredMembers(classSymbol, identifier, found);
  if (!found.empty()) return found;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (!traits_.is_aggregate(baseClass->type())) continue;

    for (auto member : lookupDesignatedMembers(baseClass, identifier)) {
      if (std::ranges::contains(found, member)) continue;
      found.push_back(member);
    }
  }

  return found;
}

auto AggregateInitializerBuilder::declaresMemberOf(ClassSymbol* classSymbol,
                                                   ClassSymbol* owner) const
    -> bool {
  if (!classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();
  if (classSymbol == owner) return true;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (declaresMemberOf(baseClass, owner)) return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->name()) continue;
    auto classType = unqualified_cast<ClassType>(field->type());
    if (!classType) continue;
    if (declaresMemberOf(classType->symbol(), owner)) return true;
  }

  return false;
}

auto AggregateInitializerBuilder::associatedElementIndex(
    const std::vector<Symbol*>& elements, Symbol* member) const
    -> std::optional<std::size_t> {
  for (std::size_t i = 0; i < elements.size(); ++i)
    if (elements[i] == member) return i;

  auto owner = symbol_cast<ClassSymbol>(member->parent());
  if (!owner) return std::nullopt;
  owner = owner->resolvedDefinition();

  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto classType =
        type_cast<ClassType>(traits_.aggregate_element_type(elements[i]));
    if (!classType) continue;
    if (declaresMemberOf(classType->symbol(), owner)) return i;
  }

  return std::nullopt;
}

auto AggregateInitializerBuilder::resolveDesignator(
    const AggregateInitializerPlan& plan, const Type* aggregateType,
    DesignatedInitializerClauseAST* designated)
    -> std::optional<DesignatedTarget> {
  if (!designated->designatorList) return std::nullopt;
  auto first = designated->designatorList->value;

  if (plan.arrayElementType) {
    auto subscript = ast_cast<SubscriptDesignatorAST>(first);
    if (!subscript) {
      error(first->firstSourceLocation(),
            std::format("member designator on array type '{}'",
                        to_string(aggregateType)));
      return std::nullopt;
    }
    if (ctx_) ctx_->checker.check(&subscript->expression);
    ASTInterpreter interp{unit_};
    auto value = interp.evaluate(subscript->expression);
    if (!value) return std::nullopt;
    auto index = interp.toUInt(*value);
    if (!index) return std::nullopt;
    if (*index >= plan.elementCount) {
      error(subscript->firstSourceLocation(),
            "array designator index exceeds array bounds");
      return std::nullopt;
    }
    return DesignatedTarget{std::size_t(*index), nullptr};
  }

  auto dot = ast_cast<DotDesignatorAST>(first);
  if (!dot) {
    error(first->firstSourceLocation(),
          std::format("array designator on non-array type '{}'",
                      to_string(aggregateType)));
    return std::nullopt;
  }

  auto classType =
      unqualified_cast<ClassType>(traits_.remove_cv(aggregateType));
  if (!classType || !classType->symbol()) return std::nullopt;

  auto designatedName = std::string{"<anonymous>"};
  if (dot->identifier) designatedName = dot->identifier->name();

  auto members = lookupDesignatedMembers(
      classType->symbol()->resolvedDefinition(), dot->identifier);

  if (members.empty()) {
    error(get_name_location(dot),
          std::format("field designator '{}' does not refer to a "
                      "non-static data member",
                      designatedName));
    return std::nullopt;
  }

  if (members.size() > 1) {
    error(get_name_location(dot),
          std::format("field designator '{}' is ambiguous", designatedName));
    return std::nullopt;
  }

  auto field = members.front();

  if (resolving()) dot->symbol = field;

  auto index = associatedElementIndex(plan.elements, field);
  if (!index) {
    error(get_name_location(dot),
          std::format("field designator '{}' does not designate an "
                      "element of '{}'",
                      to_string(field->name()), to_string(aggregateType)));
    return std::nullopt;
  }
  return DesignatedTarget{*index, field};
}

auto AggregateInitializerBuilder::remainingDesignators(
    Symbol* element, const DesignatedClause& clause) -> List<DesignatorAST*>* {
  auto designators = clause.clause->designatorList;
  if (clause.member && clause.member != element) return designators;
  return designators->next;
}

auto AggregateInitializerBuilder::mergeDesignatedClauses(
    Symbol* element, const std::vector<DesignatedClause>& clauses)
    -> ExpressionAST* {
  if (clauses.size() == 1 && !remainingDesignators(element, clauses.front())) {
    auto initializer = clauses.front().clause->initializer;
    if (auto equal = ast_cast<EqualInitializerAST>(initializer))
      return equal->expression;
    return initializer;
  }

  auto pool = unit_->arena();
  auto nested = BracedInitListAST::create(pool);
  nested->lbraceLoc = clauses.front().clause->firstSourceLocation();
  nested->rbraceLoc = clauses.back().clause->lastSourceLocation();

  auto tail = &nested->expressionList;
  for (const auto& designated : clauses) {
    auto rewritten = DesignatedInitializerClauseAST::create(pool);
    rewritten->designatorList = remainingDesignators(element, designated);
    rewritten->initializer = designated.clause->initializer;

    *tail = make_list_node<ExpressionAST>(pool, rewritten);
    tail = &(*tail)->next;
  }

  return nested;
}

auto AggregateInitializerBuilder::spliceSubobjectOverrides(
    Symbol* element, ExpressionAST* initializer,
    const std::vector<DesignatedClause>& overrides) -> ExpressionAST* {
  auto braced = ast_cast<BracedInitListAST>(initializer);
  if (!braced) {
    error(overrides.front().clause->firstSourceLocation(),
          std::format("cannot initialize a subobject of {} that is already "
                      "initialized by an expression",
                      describeAggregateElement(traits_, element)));
    return nullptr;
  }

  auto pool = unit_->arena();
  auto spliced = BracedInitListAST::create(pool);
  spliced->lbraceLoc = braced->lbraceLoc;
  spliced->rbraceLoc = overrides.back().clause->lastSourceLocation();

  auto tail = &spliced->expressionList;
  for (auto it = braced->expressionList; it; it = it->next) {
    *tail = make_list_node<ExpressionAST>(pool, it->value);
    tail = &(*tail)->next;
  }

  for (const auto& override : overrides) {
    auto rewritten = DesignatedInitializerClauseAST::create(pool);
    rewritten->designatorList = remainingDesignators(element, override);
    rewritten->initializer = override.clause->initializer;

    *tail = make_list_node<ExpressionAST>(pool, rewritten);
    tail = &(*tail)->next;
  }

  return spliced;
}

auto AggregateInitializerBuilder::selectElementInitializer(
    Symbol* element, const std::vector<PlacedClause>& placements)
    -> ExpressionAST* {
  auto supersede = [&](SourceLocation location) {
    warning(location, std::format("initialization of {} is overridden",
                                  describeAggregateElement(traits_, element)));
  };

  ExpressionAST* initializer = nullptr;
  SourceLocation initializerLocation;
  std::vector<DesignatedClause> overrides;

  for (const auto& placement : placements) {
    DesignatedClause designated{placement.designated, placement.member};

    if (placement.designated && remainingDesignators(element, designated)) {
      overrides.push_back(designated);
      continue;
    }

    if (initializer) supersede(initializerLocation);
    for (const auto& dropped : overrides)
      supersede(dropped.clause->firstSourceLocation());

    initializer = placement.designated
                      ? mergeDesignatedClauses(element, {designated})
                      : placement.clause;
    initializerLocation = placement.clause->firstSourceLocation();
    overrides.clear();
  }

  if (overrides.empty()) return initializer;
  if (!initializer) return mergeDesignatedClauses(element, overrides);
  return spliceSubobjectOverrides(element, initializer, overrides);
}

auto AggregateInitializerBuilder::build(const Type* aggregateType,
                                        BracedInitListAST* ast)
    -> std::optional<AggregateInitializerPlan> {
  AggregateInitializerPlan plan;
  if (!shapeOf(aggregateType, plan)) return std::nullopt;

  if (hasTypeDependentInitializerClause(unit_, ast)) return std::nullopt;

  const auto unbounded =
      plan.elementCount == std::numeric_limits<std::size_t>::max();

  std::vector<PlacedClause> placed;

  std::size_t next = 0;
  std::size_t bound = 0;
  std::optional<std::size_t> previousDesignatedIndex;

  auto describeAt = [&](std::size_t index) {
    return describeAggregateElement(
        traits_, index < plan.elements.size() ? plan.elements[index] : nullptr);
  };

  for (auto it = ast->expressionList; it;) {
    auto clause = it->value;
    if (!clause) {
      plan.valid = false;
      break;
    }

    if (auto clauseDesignated =
            ast_cast<DesignatedInitializerClauseAST>(clause)) {
      auto target = resolveDesignator(plan, aggregateType, clauseDesignated);
      if (!target) {
        plan.valid = false;
        it = it->next;
        continue;
      }

      const auto index = target->index;

      if (isCxx() && previousDesignatedIndex &&
          index < *previousDesignatedIndex) {
        error(clauseDesignated->firstSourceLocation(),
              std::format("designator for {} is out of declaration order",
                          describeAt(index)));
      }

      previousDesignatedIndex = index;
      placed.push_back(
          {index, clause, clauseDesignated, target->member, false});
      next = index + 1;
      bound = std::max(bound, next);
      it = it->next;
      continue;
    }

    if (next >= plan.elementCount) {
      if (plan.isUnion && plan.elementCount == 0)
        error(clause->firstSourceLocation(), "union has no named members");
      else
        error(clause->firstSourceLocation(), excessElementsMessage(plan));
      plan.valid = false;
      break;
    }

    auto elementType = elementTypeAt(plan, next);

    if (appertains(clause, elementType)) {
      placed.push_back({next, clause, nullptr, nullptr, false});
      it = it->next;
    } else {
      placed.push_back(
          {next, elideBraces(elementType, it), nullptr, nullptr, true});
    }

    ++next;
    bound = std::max(bound, next);

    if (plan.isUnion && it) {
      error(it->value->firstSourceLocation(),
            "excess elements in union initializer");
      plan.valid = false;
      break;
    }
  }

  if (unbounded) plan.elementCount = bound;

  if (plan.arrayElementType) {
    for (const auto& placement : placed) {
      plan.initializedElements.push_back({placement.index, nullptr,
                                          plan.arrayElementType,
                                          placement.clause, placement.elided});
    }
    return plan;
  }

  auto elementAt = [&](std::size_t index) -> Symbol* {
    return index < plan.elements.size() ? plan.elements[index] : nullptr;
  };

  std::map<std::size_t, std::vector<PlacedClause>> byElement;
  for (const auto& placement : placed)
    byElement[placement.index].push_back(placement);

  for (auto& [index, placements] : byElement) {
    auto initializer = selectElementInitializer(elementAt(index), placements);
    if (!initializer) {
      plan.valid = false;
      continue;
    }
    const auto elided = std::ranges::any_of(
        placements,
        [](const PlacedClause& placement) { return placement.elided; });

    plan.initializedElements.push_back({index, elementAt(index),
                                        elementTypeAt(plan, index), initializer,
                                        elided});
  }

  if (plan.isUnion && plan.initializedElements.size() > 1) {
    error(ast->lbraceLoc, "initializing multiple members of a union");
    plan.valid = false;
    plan.initializedElements.resize(1);
  }

  return plan;
}

struct DesignatedInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;

  void check(const Type* currentType, DesignatedInitializerClauseAST* ast);

  auto resolveDotDesignator(const Type* type, DotDesignatorAST* dot)
      -> const Type*;

 private:
  auto resolveDesignators(const Type* type,
                          List<DesignatorAST*>* designatorList) -> const Type*;
  auto resolveSubscriptDesignator(const Type* type,
                                  SubscriptDesignatorAST* subscript)
      -> const Type*;
};

auto DesignatedInitChecker::resolveDesignators(
    const Type* type, List<DesignatorAST*>* designatorList) -> const Type* {
  for (auto it = designatorList; it; it = it->next) {
    if (auto dot = ast_cast<DotDesignatorAST>(it->value))
      type = resolveDotDesignator(type, dot);
    else if (auto subscript = ast_cast<SubscriptDesignatorAST>(it->value))
      type = resolveSubscriptDesignator(type, subscript);
    if (!type) return nullptr;
  }
  return type;
}

auto DesignatedInitChecker::resolveDotDesignator(const Type* type,
                                                 DotDesignatorAST* dot)
    -> const Type* {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) {
    ctx.error(dot->firstSourceLocation(),
              std::format("member designator on non-aggregate type '{}'",
                          to_string(type)));
    return nullptr;
  }

  auto member = qualifiedLookup(classType->symbol(), dot->identifier);
  auto field = symbol_cast<FieldSymbol>(member);
  if (!field) {
    auto designatedName = std::string{"<anonymous>"};
    if (dot->identifier) designatedName = dot->identifier->name();

    ctx.error(get_name_location(dot),
              std::format("field designator '{}' does not refer to a "
                          "non-static data member",
                          designatedName));
    return nullptr;
  }

  dot->symbol = field;
  return ctx.traits.remove_cv(field->type());
}

auto DesignatedInitChecker::resolveSubscriptDesignator(
    const Type* type, SubscriptDesignatorAST* subscript) -> const Type* {
  ctx.checker.check(&subscript->expression);
  if (!ctx.traits.is_array(type)) {
    ctx.error(subscript->firstSourceLocation(),
              std::format("array designator on non-array type '{}'",
                          to_string(type)));
    return nullptr;
  }
  return ctx.traits.remove_cv(ctx.traits.get_element_type(type));
}

void DesignatedInitChecker::check(const Type* currentType,
                                  DesignatedInitializerClauseAST* ast) {
  auto targetType = resolveDesignators(currentType, ast->designatorList);
  if (!targetType) return;
  if (!ast->initializer) return;
  if (isDependent(ctx.unit, targetType)) return;

  if (ctx.traits.is_class(targetType)) {
    auto location = ast->initializer->firstSourceLocation();
    ast->constructorSymbol = ctx.checker.check_class_initializer(
        targetType, ast->initializer, location);
    ast->type = targetType;
    return;
  }

  if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
    if (auto nested = ast_cast<BracedInitListAST>(equal->expression)) {
      ctx.checker.check_braced_init_list(
          targetType, nested, InitializationKind::kCopyListInitialization);
    } else if (equal->expression) {
      elemChecker.check(
          equal->expression, targetType,
          std::format("cannot initialize type '{}' with expression of "
                      "type '{}'",
                      to_string(targetType),
                      to_string(equal->expression->type)));
    }
  } else if (auto braced = ast_cast<BracedInitListAST>(ast->initializer)) {
    ctx.checker.check_braced_init_list(
        targetType, braced, InitializationKind::kCopyListInitialization);
  }

  ast->type = targetType;
}

struct AggregateInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;
  DesignatedInitChecker& desigChecker;

  void checkUnion(ClassSymbol* classSymbol, BracedInitListAST* ast);
  void checkStruct(ClassSymbol* classSymbol, BracedInitListAST* ast);

  [[nodiscard]] auto makeValueInitializer(const Type* type,
                                          SourceLocation location)
      -> ExpressionAST*;

  void checkElementInit(ExpressionAST*& expr, const Type* type,
                        const std::string& description,
                        InitializationKind initializationKind =
                            InitializationKind::kCopyListInitialization);

  [[nodiscard]] auto checkParenthesizedAggregate(
      ClassSymbol* classSymbol, const Type* classType,
      List<ExpressionAST*>* expressionList, SourceLocation location)
      -> BracedInitListAST*;

 private:
  static auto firstNonStaticField(ClassSymbol* symbol) -> FieldSymbol* {
    for (auto field : views::members(symbol) | views::non_static_fields)
      return field;
    return nullptr;
  }

  [[nodiscard]] auto elementDescription(Symbol* element) const -> std::string;
  [[nodiscard]] auto defaultMemberInitializer(FieldSymbol* field,
                                              SourceLocation location)
      -> ExpressionAST*;
  [[nodiscard]] auto implicitElementInitializer(Symbol* element,
                                                SourceLocation location)
      -> ExpressionAST*;

  void checkElementInit(ExpressionAST*& expr, Symbol* element);
  void checkAnonUnionFieldInit(ExpressionAST*& expr, const Type* fieldType);
  void initializeUnionByDefault(ClassSymbol* classSymbol,
                                BracedInitListAST* ast);

  [[nodiscard]] auto makeEmptyInitializerList(SourceLocation location)
      -> BracedInitListAST*;
};

auto AggregateInitChecker::elementDescription(Symbol* element) const
    -> std::string {
  return describeAggregateElement(ctx.traits, element);
}

auto AggregateInitChecker::defaultMemberInitializer(FieldSymbol* field,
                                                    SourceLocation location)
    -> ExpressionAST* {
  auto initializer = field->initializer();
  if (!initializer) return nullptr;

  if (auto equal = ast_cast<EqualInitializerAST>(initializer))
    initializer = equal->expression;

  auto pool = ctx.unit->arena();
  initializer = initializer->clone(pool);
  auto fieldType = ctx.traits.remove_cv(field->type());

  if (auto constructor = field->constructor()) {
    auto arguments = BracedInitListAST::create(pool);
    if (auto paren = ast_cast<ParenInitializerAST>(initializer))
      arguments->expressionList = paren->expressionList;
    else if (auto braced = ast_cast<BracedInitListAST>(initializer))
      arguments->expressionList = braced->expressionList;
    else
      arguments->expressionList =
          make_list_node<ExpressionAST>(pool, initializer);

    initializer =
        makeClassConstruction(ctx.unit, fieldType, constructor, arguments);
  } else if (auto braced = ast_cast<BracedInitListAST>(initializer);
             braced && !ctx.traits.is_class_or_union(fieldType) &&
             !ctx.traits.is_array(fieldType)) {
    if (!braced->expressionList)
      return makeValueInitializer(fieldType, braced->lbraceLoc);
    initializer = braced->expressionList->value;
  }

  auto result = makeDefaultInitializer(ctx.unit, initializer, location,
                                       ctx.checker.scope());
  ctx.checker.evaluateImmediateInvocation(&result);
  return result;
}

auto AggregateInitChecker::makeEmptyInitializerList(SourceLocation location)
    -> BracedInitListAST* {
  auto braced = BracedInitListAST::create(ctx.unit->arena());
  braced->lbraceLoc = location;
  braced->rbraceLoc = location;
  return braced;
}

auto AggregateInitChecker::makeValueInitializer(const Type* type,
                                                SourceLocation location)
    -> ExpressionAST* {
  auto pool = ctx.unit->arena();
  auto braced = makeEmptyInitializerList(location);

  if (!type) return braced;

  if (ctx.traits.is_class_or_union(type)) {
    ExpressionAST* initializer = braced;
    ctx.checker.check_list_initialization(
        type, initializer, InitializationKind::kCopyListInitialization);
    return initializer;
  }

  ctx.checker.check_braced_init_list(
      type, braced, InitializationKind::kCopyListInitialization);
  return braced;
}

auto AggregateInitChecker::implicitElementInitializer(Symbol* element,
                                                      SourceLocation location)
    -> ExpressionAST* {
  if (auto field = symbol_cast<FieldSymbol>(element)) {
    if (auto initializer = defaultMemberInitializer(field, location))
      return initializer;
  }

  auto type = ctx.traits.aggregate_element_type(element);

  if (type && ctx.traits.is_reference(type)) {
    ctx.error(location, std::format("reference {} is not initialized",
                                    elementDescription(element)));
    return makeEmptyInitializerList(location);
  }

  return makeValueInitializer(type, location);
}

void AggregateInitChecker::checkElementInit(ExpressionAST*& expr,
                                            Symbol* element) {
  auto type = ctx.traits.aggregate_element_type(element);
  if (!type) return;

  auto field = symbol_cast<FieldSymbol>(element);

  if (field && !field->name() && ctx.traits.is_union(type)) {
    checkAnonUnionFieldInit(expr, type);
    return;
  }

  checkElementInit(expr, type, elementDescription(element));
}

void AggregateInitChecker::checkElementInit(
    ExpressionAST*& expr, const Type* type, const std::string& description,
    InitializationKind initializationKind) {
  if (ctx.traits.is_class_or_union(type)) {
    if (ast_cast<BracedInitListAST>(expr)) {
      ctx.checker.check_list_initialization(type, expr, initializationKind);
      return;
    }

    if (elemChecker.checkClassElementInit(expr, type)) return;
  } else if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    ctx.checker.check_braced_init_list(
        type, nested, InitializationKind::kCopyListInitialization);
    return;
  }

  elemChecker.check(
      expr, type,
      std::format("cannot initialize {} of type '{}' with "
                  "expression of type '{}'",
                  description, to_string(type), to_string(expr->type)),
      initializationKind);
}

auto AggregateInitChecker::checkParenthesizedAggregate(
    ClassSymbol* classSymbol, const Type* classType,
    List<ExpressionAST*>* expressionList, SourceLocation location)
    -> BracedInitListAST* {
  AggregateInitGuard guard{ctx.checker, classSymbol};
  if (!guard) return nullptr;

  auto elements = ctx.traits.aggregate_elements(classSymbol);

  auto pool = ctx.unit->arena();
  auto normalized = BracedInitListAST::create(pool);
  normalized->lbraceLoc = location;
  normalized->rbraceLoc = location;
  normalized->type = classType;
  normalized->valueCategory = ValueCategory::kPrValue;

  auto tail = &normalized->expressionList;
  std::size_t elementIndex = 0;

  for (auto it = expressionList; it; it = it->next) {
    if (ast_cast<DesignatedInitializerClauseAST>(it->value)) {
      ctx.error(it->value->firstSourceLocation(),
                "designators are not permitted in a parenthesized "
                "initializer of an aggregate");
      return nullptr;
    }

    if (elementIndex >= elements.size()) {
      ctx.error(it->value->firstSourceLocation(),
                "excess elements in struct initializer");
      return nullptr;
    }

    auto element = elements[elementIndex];
    checkElementInit(it->value, ctx.traits.aggregate_element_type(element),
                     elementDescription(element),
                     InitializationKind::kCopyInitialization);

    *tail = make_list_node<ExpressionAST>(pool, it->value);
    tail = &(*tail)->next;
    ++elementIndex;
  }

  for (; elementIndex < elements.size(); ++elementIndex) {
    *tail = make_list_node<ExpressionAST>(
        pool, implicitElementInitializer(elements[elementIndex], location));
    tail = &(*tail)->next;
  }

  return normalized;
}

void AggregateInitChecker::checkAnonUnionFieldInit(ExpressionAST*& expr,
                                                   const Type* fieldType) {
  auto classType = type_cast<ClassType>(fieldType);
  if (!classType || !classType->symbol() || !classType->symbol()->isUnion()) {
    return;
  }

  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    ctx.checker.check_braced_init_list(
        fieldType, nested, InitializationKind::kCopyListInitialization);
    return;
  }

  auto first = firstNonStaticField(classType->symbol());
  if (!first) {
    ctx.error(expr->firstSourceLocation(), "union has no named members");
    return;
  }
  auto firstType = ctx.traits.remove_cv(first->type());
  elemChecker.check(
      expr, firstType,
      std::format("cannot initialize anonymous union member '{}' of "
                  "type '{}' with expression of type '{}'",
                  to_string(first->name()), to_string(firstType),
                  to_string(expr->type)));
}

void AggregateInitChecker::initializeUnionByDefault(ClassSymbol* classSymbol,
                                                    BracedInitListAST* ast) {
  auto pool = ctx.unit->arena();
  auto firstField = firstNonStaticField(classSymbol);
  if (!firstField) return;

  FieldSymbol* variantMember = nullptr;
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->initializer()) continue;
    variantMember = field;
    break;
  }

  if (!variantMember) {
    auto initializer = makeValueInitializer(
        ctx.traits.remove_cv(firstField->type()), ast->lbraceLoc);
    ast->expressionList = make_list_node<ExpressionAST>(pool, initializer);
    return;
  }

  auto initializer = defaultMemberInitializer(variantMember, ast->lbraceLoc);
  if (!initializer) return;

  if (variantMember == firstField) {
    ast->expressionList = make_list_node<ExpressionAST>(pool, initializer);
    return;
  }

  auto dot = DotDesignatorAST::create(pool);
  dot->identifier = name_cast<Identifier>(variantMember->name());
  dot->symbol = variantMember;

  auto clause = DesignatedInitializerClauseAST::create(pool);
  clause->designatorList = make_list_node<DesignatorAST>(pool, dot);
  clause->initializer = initializer;
  clause->type = ctx.traits.remove_cv(variantMember->type());

  ast->expressionList = make_list_node<ExpressionAST>(pool, clause);
}

void AggregateInitChecker::checkUnion(ClassSymbol* classSymbol,
                                      BracedInitListAST* ast) {
  AggregateInitGuard guard{ctx.checker, classSymbol};
  if (!guard) return;

  if (!ast->expressionList) {
    initializeUnionByDefault(classSymbol, ast);
    return;
  }

  auto classType = ctx.control->getClassType(classSymbol);
  auto plan = resolveAggregateInitialization(ctx, classType, ast);
  if (!plan || plan->initializedElements.empty()) return;

  const auto& initialized = plan->initializedElements.front();
  auto initializer = initialized.initializer;
  if (!initializer) return;

  checkElementInit(initializer, initialized.element);

  auto pool = ctx.unit->arena();

  if (initialized.index == 0) {
    ast->expressionList = make_list_node<ExpressionAST>(pool, initializer);
    return;
  }

  auto field = symbol_cast<FieldSymbol>(initialized.element);

  auto dot = DotDesignatorAST::create(pool);
  dot->identifier = name_cast<Identifier>(field->name());
  dot->symbol = field;

  auto clause = DesignatedInitializerClauseAST::create(pool);
  clause->designatorList = make_list_node<DesignatorAST>(pool, dot);
  clause->initializer = initializer;
  clause->type = initialized.type;

  ast->expressionList = make_list_node<ExpressionAST>(pool, clause);
}

void AggregateInitChecker::checkStruct(ClassSymbol* classSymbol,
                                       BracedInitListAST* ast) {
  AggregateInitGuard guard{ctx.checker, classSymbol};
  if (!guard) return;

  if (hasUnexpandedPackExpansion(ast)) return;

  auto classType = ctx.control->getClassType(classSymbol);
  auto plan = resolveAggregateInitialization(ctx, classType, ast);
  if (!plan) return;

  std::vector<ExpressionAST*> initializers(plan->elements.size(), nullptr);

  for (const auto& initialized : plan->initializedElements) {
    auto initializer = initialized.initializer;
    if (!initializer) continue;
    checkElementInit(initializer, initialized.element);
    initializers[initialized.index] = initializer;
  }

  auto pool = ctx.unit->arena();
  List<ExpressionAST*>* normalized = nullptr;
  auto tail = &normalized;

  for (std::size_t i = 0; i < plan->elements.size(); ++i) {
    auto initializer = initializers[i];
    if (!initializer)
      initializer =
          implicitElementInitializer(plan->elements[i], ast->lbraceLoc);

    *tail = make_list_node<ExpressionAST>(pool, initializer);
    tail = &(*tail)->next;
  }

  ast->expressionList = normalized;
}

struct ListInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;
  DesignatedInitChecker& desigChecker;
  AggregateInitChecker& aggregateChecker;

  [[nodiscard]] auto computeBullet(const Type* targetType,
                                   BracedInitListAST* ast,
                                   InitializationKind initializationKind) const
      -> ListInitializationBullet;

  void aggregateInit(const Type* type, BracedInitListAST* ast);

  void characterArrayFromStringLiteral(const Type* type,
                                       BracedInitListAST* ast);

  void singleElementInit(const Type* type, BracedInitListAST* ast,
                         InitializationKind initializationKind);

  void enumerationFromScalar(const Type* type, BracedInitListAST* ast);

  void referenceFromPrvalue(const Type* type, BracedInitListAST* ast,
                            InitializationKind initializationKind);

  void diagnoseIllFormed(const Type* type, BracedInitListAST* ast);

  [[nodiscard]] static auto hasDesignators(BracedInitListAST* ast) -> bool;

  void checkArrayElements(const Type* type, const Type* elementType,
                          BracedInitListAST* ast);

 private:
  [[nodiscard]] auto hasDefaultConstructor(const Type* type) const -> bool;

  [[nodiscard]] auto initializesFixedUnderlyingTypeEnumeration(
      const Type* type, ExpressionAST* element) const -> bool;
};

auto ListInitChecker::hasDesignators(BracedInitListAST* ast) -> bool {
  for (auto it = ast->expressionList; it; it = it->next)
    if (ast_cast<DesignatedInitializerClauseAST>(it->value)) return true;
  return false;
}

auto ListInitChecker::hasDefaultConstructor(const Type* type) const -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return false;
  return OverloadResolution{ctx.unit}.hasDefaultConstructor(
      classType->symbol()->resolvedDefinition());
}

auto ListInitChecker::initializesFixedUnderlyingTypeEnumeration(
    const Type* type, ExpressionAST* element) const -> bool {
  auto enumType = type_cast<ScopedEnumType>(type);
  if (!enumType) return false;
  auto underlyingType = enumType->underlyingType();
  if (!underlyingType) return false;
  if (!element || !element->type) return false;
  if (!ctx.traits.is_scalar(element->type)) return false;
  return bool(ctx.checker.checkImplicitConversion(element, underlyingType));
}

auto ListInitChecker::computeBullet(const Type* targetType,
                                    BracedInitListAST* ast,
                                    InitializationKind initializationKind) const
    -> ListInitializationBullet {
  const auto isReference = ctx.traits.is_reference(targetType);
  const auto type =
      ctx.traits.remove_cv(ctx.traits.remove_reference(targetType));

  const auto designated = hasDesignators(ast);
  auto element = singleInitializerClause(ast);

  if (designated && !isReference) {
    if (!ctx.traits.is_aggregate(type)) return ListInitializationBullet::kNone;
    return ListInitializationBullet::kDesignatedAggregate;
  }

  if (!isReference && ctx.traits.is_class(type) &&
      ctx.traits.is_aggregate(type) && element && element->type) {
    auto elementType = ctx.traits.remove_cvref(element->type);
    if (ctx.traits.is_same(elementType, type) ||
        ctx.traits.is_base_of(type, elementType))
      return ListInitializationBullet::kAggregateFromSameOrDerivedElement;
  }

  if (!isReference &&
      stringLiteralInitialization(ctx.traits, ctx.isCxx(), type, element))
    return ListInitializationBullet::kCharacterArrayFromStringLiteral;

  if (!isReference && ctx.traits.is_aggregate(type))
    return ListInitializationBullet::kAggregate;

  if (!isReference && !ast->expressionList &&
      ctx.traits.is_class_or_union(type) && hasDefaultConstructor(type))
    return ListInitializationBullet::kEmptyListDefaultConstructor;

  if (!isReference && ctx.traits.initializer_list_element_type(type))
    return ListInitializationBullet::kInitializerList;

  if (!isReference && ctx.traits.is_class_or_union(type))
    return ListInitializationBullet::kConstructor;

  if (!isReference && isDirectInitialization(initializationKind) &&
      initializesFixedUnderlyingTypeEnumeration(type, element))
    return ListInitializationBullet::kEnumerationWithFixedUnderlyingType;

  if (!designated && element) {
    if (!isReference ||
        ctx.traits.is_reference_related(ctx.traits.remove_reference(targetType),
                                        element->type))
      return ListInitializationBullet::kSingleElement;
  }

  if (isReference) return ListInitializationBullet::kReferenceToPrvalue;

  if (!ast->expressionList)
    return ListInitializationBullet::kEmptyListValueInitialization;

  return ListInitializationBullet::kNone;
}

void ListInitChecker::aggregateInit(const Type* type, BracedInitListAST* ast) {
  if (auto classType = unqualified_cast<ClassType>(type)) {
    if (!classType->symbol()) return;
    if (classType->isUnion())
      aggregateChecker.checkUnion(classType->symbol(), ast);
    else
      aggregateChecker.checkStruct(classType->symbol(), ast);
    return;
  }

  auto elementType = ctx.traits.remove_cv(ctx.traits.get_element_type(type));
  checkArrayElements(type, elementType, ast);
}

void ListInitChecker::characterArrayFromStringLiteral(const Type* type,
                                                      BracedInitListAST* ast) {
  (void)checkStringLiteralInitialization(ctx, type,
                                         singleInitializerClause(ast));
}

void ListInitChecker::checkArrayElements(const Type* type,
                                         const Type* elementType,
                                         BracedInitListAST* ast) {
  auto plan = resolveAggregateInitialization(ctx, type, ast);
  if (!plan) return;

  std::vector<std::pair<std::size_t, ExpressionAST*>> placements;

  for (const auto& initialized : plan->initializedElements) {
    auto initializer = initialized.initializer;
    if (!initializer) continue;

    if (ast_cast<PackExpansionExpressionAST>(initializer)) {
      placements.emplace_back(initialized.index, initializer);
      continue;
    }

    if (auto designated =
            ast_cast<DesignatedInitializerClauseAST>(initializer)) {
      desigChecker.check(type, designated);
    } else if (auto nested = ast_cast<BracedInitListAST>(initializer)) {
      ctx.checker.check_braced_init_list(
          elementType, nested, InitializationKind::kCopyListInitialization);
    } else {
      elemChecker.check(
          initializer, elementType,
          std::format("cannot initialize array element of type '{}' with "
                      "expression of type '{}'",
                      to_string(elementType), to_string(initializer->type)));
    }

    placements.emplace_back(initialized.index, initializer);
  }

  auto pool = ctx.unit->arena();
  List<ExpressionAST*>* rebuilt = nullptr;
  auto rebuiltTail = &rebuilt;
  for (auto& [index, initializer] : placements) {
    *rebuiltTail = make_list_node<ExpressionAST>(pool, initializer);
    rebuiltTail = &(*rebuiltTail)->next;
  }
  ast->expressionList = rebuilt;

  auto bounded = type_cast<BoundedArrayType>(type);
  if (!bounded) return;
  if (placements.size() == bounded->size()) return;
  if (ctx.traits.is_trivially_constructible(elementType)) return;

  std::vector<ExpressionAST*> slots(bounded->size(), nullptr);
  for (auto& [slot, initializer] : placements) slots[slot] = initializer;

  List<ExpressionAST*>* normalized = nullptr;
  auto tail = &normalized;

  for (auto& initializer : slots) {
    if (!initializer)
      initializer =
          aggregateChecker.makeValueInitializer(elementType, ast->lbraceLoc);

    *tail = make_list_node<ExpressionAST>(pool, initializer);
    tail = &(*tail)->next;
  }

  ast->expressionList = normalized;
}

void ListInitChecker::singleElementInit(const Type* type,
                                        BracedInitListAST* ast,
                                        InitializationKind initializationKind) {
  auto& expr = ast->expressionList->value;

  elemChecker.check(expr, ctx.traits.remove_cv(type),
                    std::format("cannot initialize type '{}' with "
                                "expression of type '{}'",
                                to_string(type), to_string(expr->type)),
                    initializationKind);
}

void ListInitChecker::enumerationFromScalar(const Type* type,
                                            BracedInitListAST* ast) {
  auto enumType = type_cast<ScopedEnumType>(type);
  auto underlyingType = enumType->underlyingType();
  auto& expr = ast->expressionList->value;

  auto source = expr;

  if (!ctx.checker.implicit_conversion(
          expr, underlyingType, InitializationKind::kDirectInitialization)) {
    ctx.error(expr->firstSourceLocation(),
              std::format("cannot initialize type '{}' with expression of "
                          "type '{}'",
                          to_string(type), to_string(source->type)));
    return;
  }

  diagnoseNarrowingListElement(ctx, source, underlyingType);

  auto cast = ImplicitCastExpressionAST::create(ctx.unit->arena());
  cast->expression = expr;
  cast->castKind = ImplicitCastKind::kIntegralConversion;
  cast->type = type;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
}

void ListInitChecker::referenceFromPrvalue(
    const Type* type, BracedInitListAST* ast,
    InitializationKind initializationKind) {
  auto referencedType = ctx.traits.remove_reference(type);

  if (auto unbounded =
          type_cast<UnboundedArrayType>(ctx.traits.remove_cv(referencedType))) {
    std::size_t elementCount = 0;
    for (auto it = ast->expressionList; it; it = it->next) ++elementCount;
    referencedType =
        ctx.traits.add_cv(ctx.control->getBoundedArrayType(
                              unbounded->elementType(), elementCount),
                          cv_qualifiers(referencedType));
  }

  StandardConversion conversions{ctx.unit, !ctx.isCxx()};
  auto binding = conversions.referenceBinding(type, referencedType,
                                              ValueCategory::kPrValue);

  if (!binding || !*binding) {
    ctx.error(ast->firstSourceLocation(),
              std::format("non-const lvalue reference of type '{}' cannot "
                          "bind to an initializer list temporary",
                          to_string(type)));
    return;
  }

  ctx.checker.check_braced_init_list(
      referencedType, ast, InitializationKind::kCopyListInitialization);

  ast->type = referencedType;
  ast->valueCategory = ValueCategory::kPrValue;
}

void ListInitChecker::diagnoseIllFormed(const Type* type,
                                        BracedInitListAST* ast) {
  auto it = ast->expressionList;
  if (!it) return;

  if (ast_cast<DesignatedInitializerClauseAST>(it->value)) {
    ctx.error(it->value->firstSourceLocation(),
              "designator in initializer for scalar type");
    return;
  }

  ctx.error(it->next ? it->next->value->firstSourceLocation()
                     : it->value->firstSourceLocation(),
            "excess elements in scalar initializer");
}

struct ClassInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;
  AggregateInitChecker& aggregateChecker;

  struct Target {
    const Type* type = nullptr;
    ExpressionAST* initializer = nullptr;
    SourceLocation location;
    FunctionSymbol* constructor = nullptr;
    List<ExpressionAST*>** argumentList = nullptr;
    bool diagnoseUnresolved = false;
    std::optional<InitializationKind> initializationKind;
    InitializationBullet bullet = InitializationBullet::kNone;
    ListInitializationBullet listBullet = ListInitializationBullet::kNone;
  };

  void checkClassInit(Target& target);

 private:
  void checkListInit(Target& target, ClassSymbol* classSymbol);

  void checkParenthesizedAggregateInit(Target& target,
                                       ClassSymbol* classSymbol);

  void checkAggregateInit(Target& target, ClassSymbol* classSymbol);
  void checkUserDefinedConversionInit(Target& target, ClassSymbol* classSymbol);
  void checkConstructorInit(Target& target, ClassSymbol* classSymbol,
                            bool diagnoseUnresolved);

  void reportRejectedConstructors(const ConstructorResult& resolution);
  void checkNarrowingArguments(const std::vector<ExpressionAST*>& args,
                               FunctionSymbol* constructor);

  void appendDefaultArguments(Target& target, FunctionSymbol* constructor);

  [[nodiscard]] auto arguments(Target& target) -> std::vector<ExpressionAST*>;

  void applyArgumentConversions(
      Target& target,
      const std::vector<ImplicitConversionSequence>& conversions);

  [[nodiscard]] auto argumentListSlot(Target& target, Arena* arena)
      -> List<ExpressionAST*>**;

  void checkInitializerListElements(Target& target,
                                    BracedInitListAST* bracedInitList,
                                    FunctionSymbol* constructor);

  void diagnoseConstructorSelection(const ConstructorResult& resolution,
                                    ClassSymbol* classSymbol,
                                    SourceLocation location,
                                    bool diagnoseUnresolved);
};

void ClassInitChecker::checkClassInit(Target& target) {
  if (!ctx.unit->config().checkTypes) return;

  auto targetType = ctx.traits.remove_cv(target.type);
  auto classType = type_cast<ClassType>(targetType);
  if (!classType || !classType->symbol()) return;
  auto classSymbol = classType->definition();

  const auto isAggregate = ctx.traits.is_aggregate(classType);
  const auto diagnoseUnresolved = !isAggregate && target.diagnoseUnresolved;

  switch (target.bullet) {
    case InitializationBullet::kSameTypePrvalue:
      return;

    case InitializationBullet::kListInitialization:
      checkListInit(target, classSymbol);
      return;

    case InitializationBullet::kParenthesizedAggregate:
      checkParenthesizedAggregateInit(target, classSymbol);
      return;

    case InitializationBullet::kDefaultInitialization:
    case InitializationBullet::kValueInitializationFromParens:
      checkConstructorInit(target, classSymbol, diagnoseUnresolved);
      return;

    case InitializationBullet::kConstructor:
      checkConstructorInit(target, classSymbol, diagnoseUnresolved);
      if (target.constructor || !isAggregate) return;
      if (Initializer::withArgumentList(target.initializer, target.argumentList)
              .form() == InitializerForm::kParen)
        checkParenthesizedAggregateInit(target, classSymbol);
      else
        checkAggregateInit(target, classSymbol);
      return;

    case InitializationBullet::kUserDefinedConversion:
      checkUserDefinedConversionInit(target, classSymbol);
      return;

    default:
      return;
  }
}

void ClassInitChecker::checkListInit(Target& target, ClassSymbol* classSymbol) {
  const auto diagnoseUnresolved =
      target.listBullet !=
          ListInitializationBullet::kAggregateFromSameOrDerivedElement &&
      target.diagnoseUnresolved;

  checkConstructorInit(target, classSymbol, diagnoseUnresolved);
}

void ClassInitChecker::checkParenthesizedAggregateInit(
    Target& target, ClassSymbol* classSymbol) {
  auto initializer =
      Initializer::withArgumentList(target.initializer, target.argumentList);

  auto slot = initializer.expressionListSlot();
  if (!slot) return;

  auto normalized = aggregateChecker.checkParenthesizedAggregate(
      classSymbol, ctx.traits.remove_cv(target.type), *slot, target.location);
  if (!normalized) return;

  if (target.argumentList && !target.initializer) {
    *target.argumentList = normalized->expressionList;
    return;
  }

  target.initializer = normalized;
}

void ClassInitChecker::checkAggregateInit(Target& target,
                                          ClassSymbol* classSymbol) {
  if (!ctx.unit->config().checkTypes) return;

  auto targetType = ctx.traits.remove_cv(target.type);
  auto bracedInitList = Initializer{target.initializer}.bracedInitList();

  if (bracedInitList) {
    ctx.checker.check_braced_init_list(
        targetType, bracedInitList,
        Initializer{target.initializer}.initializationKind());
    return;
  }

  if (auto equal = ast_cast<EqualInitializerAST>(target.initializer);
      equal && equal->expression) {
    elemChecker.check(
        equal->expression, targetType,
        std::format("cannot initialize type '{}' with expression of type '{}'",
                    to_string(targetType), to_string(equal->expression->type)),
        InitializationKind::kCopyInitialization);
  }
}

void ClassInitChecker::checkUserDefinedConversionInit(
    Target& target, ClassSymbol* classSymbol) {
  if (!ctx.unit->config().checkTypes) return;

  auto initializer = Initializer{target.initializer};
  auto source = initializer.singleExpression();
  if (!source || !source->type) return;
  if (isDependent(ctx.unit, source->type)) return;
  if (isDependent(ctx.unit, target.type)) return;

  InitializerOperand operand{target.initializer, source};

  auto targetType = ctx.traits.remove_cv(target.type);
  auto sequence =
      ctx.checker.checkImplicitConversion(operand.operand(), targetType);

  if (!sequence) {
    if (target.diagnoseUnresolved) {
      auto location = target.location;
      if (!location) location = source->firstSourceLocation();
      diagnoseConversionFailure(
          ctx, InitializedEntity::temporary(targetType, location), source);
    }
    return;
  }

  if (target.diagnoseUnresolved)
    ctx.checker.diagnoseAmbiguousConversion(sequence, operand.operand());

  ctx.checker.applyImplicitConversion(sequence, operand.operand());

  target.initializer = operand.result();

  Initializer{target.initializer}.propagateType();
}

void ClassInitChecker::checkConstructorInit(Target& target,
                                            ClassSymbol* classSymbol,
                                            bool diagnoseUnresolved) {
  if (!ctx.unit->config().checkTypes) return;

  auto args = arguments(target);

  const auto inTemplate = isEnclosedInDependentTemplate(
      ctx.unit, ctx.checker.scope(), /*stopAtConcreteSpecialization=*/true);

  if (inTemplate) {
    if (isEnclosedInDependentTemplate(ctx.unit, classSymbol,
                                      /*stopAtConcreteSpecialization=*/true))
      return;
    if (target.type && isDependent(ctx.unit, target.type)) return;
  }

  for (auto arg : args) {
    if (!arg) continue;
    if (!arg->type) {
      if (inTemplate) return;
      continue;
    }
    if (isDependent(ctx.unit, arg->type)) return;
  }

  OverloadResolution overloadRes(ctx.unit);

  auto initializationKindOf = [&] {
    if (target.initializationKind.has_value())
      return *target.initializationKind;
    if (target.initializer)
      return Initializer{target.initializer}.initializationKind();
    return InitializationKind::kDirectInitialization;
  };

  auto location = target.location;
  if (!location && target.initializer)
    location = target.initializer->firstSourceLocation();
  if (!location) location = classSymbol->location();

  auto bracedInitList = Initializer{target.initializer}.bracedInitList();

  auto resolution =
      bracedInitList
          ? overloadRes.selectListConstructor(classSymbol, bracedInitList, args,
                                              initializationKindOf())
          : overloadRes.resolveConstructor(classSymbol, args,
                                           initializationKindOf());

  if (!resolution) {
    diagnoseConstructorSelection(resolution, classSymbol, location,
                                 diagnoseUnresolved);
    return;
  }

  target.constructor = resolution.selected();
  ctx.checker.checkConstructorAccess(target.constructor, target.location);
  ctx.checker.useFunction(target.constructor, target.location);

  if (resolution.fromInitializerListConstructor) {
    checkInitializerListElements(target, bracedInitList, target.constructor);
    appendDefaultArguments(target, target.constructor);
    return;
  }

  if (bracedInitList) checkNarrowingArguments(args, target.constructor);
  applyArgumentConversions(target, resolution.best->conversions);
  appendDefaultArguments(target, target.constructor);
}

void ClassInitChecker::diagnoseConstructorSelection(
    const ConstructorResult& resolution, ClassSymbol* classSymbol,
    SourceLocation location, bool diagnoseUnresolved) {
  switch (resolution.failure) {
    case ConstructorSelectionFailure::kNoViableConstructor:
      if (!diagnoseUnresolved) return;
      ctx.error(
          location,
          std::format("no matching constructor for initialization of '{}'",
                      to_string(classSymbol->type())));
      reportRejectedConstructors(resolution);
      return;

    case ConstructorSelectionFailure::kAmbiguous:
      ctx.error(location,
                std::format("call to constructor of '{}' is ambiguous",
                            to_string(classSymbol->type())));
      for (const auto& candidate : resolution.candidates) {
        if (!candidate.viable || !candidate.symbol) continue;
        ctx.checker.note(candidate.symbol->location(),
                         std::format("candidate constructor '{}'",
                                     to_string(candidate.symbol->type())));
      }
      return;

    case ConstructorSelectionFailure::kExplicitInCopyInitialization:
      ctx.error(location,
                "chosen constructor is explicit in copy-initialization");
      ctx.checker.note(resolution.best->symbol->location(),
                       "explicit constructor declared here");
      return;

    case ConstructorSelectionFailure::kNone:
      return;
  }
}

void ClassInitChecker::checkNarrowingArguments(
    const std::vector<ExpressionAST*>& args, FunctionSymbol* constructor) {
  auto parameters = constructor->parameters();

  std::size_t index = 0;
  for (auto argument : args) {
    if (index >= parameters.size()) break;
    diagnoseNarrowingListElement(ctx, argument, parameters[index]->type());
    ++index;
  }
}

void ClassInitChecker::reportRejectedConstructors(
    const ConstructorResult& resolution) {
  cxx::reportRejectedConstructors(ctx, resolution);
}

auto ClassInitChecker::arguments(Target& target)
    -> std::vector<ExpressionAST*> {
  if (target.initializer) return Initializer{target.initializer}.arguments();

  std::vector<ExpressionAST*> args;
  if (!target.argumentList) return args;
  for (auto it = *target.argumentList; it; it = it->next)
    args.push_back(it->value);
  return args;
}

void ClassInitChecker::applyArgumentConversions(
    Target& target,
    const std::vector<ImplicitConversionSequence>& conversions) {
  if (target.initializer) {
    applyInitializerConversions(ctx.checker, Initializer{target.initializer},
                                conversions);
    return;
  }

  if (!target.argumentList) return;

  std::size_t index = 0;
  for (auto it = *target.argumentList; it && index < conversions.size();
       it = it->next, ++index) {
    ctx.checker.applyImplicitConversion(conversions[index], it->value);
  }
}

void ClassInitChecker::appendDefaultArguments(Target& target,
                                              FunctionSymbol* constructor) {
  auto params = constructor->parameters();
  if (params.empty()) return;

  const auto argCount = static_cast<int>(arguments(target).size());
  const auto parameterCount = static_cast<int>(params.size());
  if (argCount >= parameterCount) return;
  if (required_parameter_count(constructor, parameterCount) > argCount) return;

  auto tail = argumentListSlot(target, ctx.unit->arena());
  if (!tail) return;

  ctx.checker.append_default_arguments(constructor, tail, target.location);
}

auto ClassInitChecker::argumentListSlot(Target& target, Arena* arena)
    -> List<ExpressionAST*>** {
  if (target.argumentList) return target.argumentList;

  auto initializer = Initializer::stripImplicitCasts(target.initializer);

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    if (!equal->expression) return nullptr;
    auto unwrapped = Initializer::stripImplicitCasts(equal->expression);
    if (auto braced = ast_cast<BracedInitListAST>(unwrapped))
      return &braced->expressionList;
    target.initializer = makeParenInitializer(
        arena, target.location,
        make_list_node<ExpressionAST>(arena, equal->expression));
    return &ast_cast<ParenInitializerAST>(target.initializer)->expressionList;
  }

  if (auto paren = ast_cast<ParenInitializerAST>(initializer))
    return &paren->expressionList;

  if (auto braced = ast_cast<BracedInitListAST>(initializer))
    return &braced->expressionList;

  if (initializer) return nullptr;

  target.initializer = makeParenInitializer(arena, target.location, nullptr);
  return &ast_cast<ParenInitializerAST>(target.initializer)->expressionList;
}

void ClassInitChecker::checkInitializerListElements(
    Target& target, BracedInitListAST* bracedInitList,
    FunctionSymbol* constructor) {
  auto ctorParamType = constructor->parameters().front()->type();
  auto elemType = ctx.traits.initializer_list_element_type(ctorParamType);

  bracedInitList->type = ctorParamType;
  bracedInitList->valueCategory = ValueCategory::kPrValue;
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    if (elemChecker.checkClassElementInit(it->value, elemType)) continue;
    elemChecker.check(
        it->value, elemType,
        std::format("cannot initialize initializer_list element "
                    "of type '{}' with expression of type '{}'",
                    to_string(elemType), to_string(it->value->type)));
  }

  target.initializer = makeParenInitializer(
      ctx.unit->arena(), target.location,
      make_list_node<ExpressionAST>(ctx.unit->arena(), bracedInitList));
}

struct ScalarInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;

  [[nodiscard]] auto check(ExpressionAST* initializer, const Type* declaredType)
      -> ExpressionAST*;
};

auto ScalarInitChecker::check(ExpressionAST* initializer,
                              const Type* declaredType) -> ExpressionAST* {
  if (!initializer) return initializer;

  if (auto bracedInitList = Initializer{initializer}.bracedInitList()) {
    ctx.checker.check_braced_init_list(
        declaredType, bracedInitList,
        Initializer{initializer}.initializationKind());
    return initializer;
  }

  auto initExpr = Initializer{initializer}.singleExpression();
  if (!initExpr) return initializer;

  InitializerOperand operand{initializer, initExpr};

  auto conversionTargetType = ctx.traits.remove_cv(declaredType);

  elemChecker.check(
      operand.operand(), conversionTargetType,
      std::format("cannot initialize type '{}' with expression of type '{}'",
                  to_string(conversionTargetType),
                  to_string(operand.operand()->type)),
      Initializer{initializer}.initializationKind());

  Initializer{initializer}.propagateType();

  return operand.result();
}

struct ReferenceInitChecker {
  InitContext& ctx;

  [[nodiscard]] auto check(const Type* targetType, ExpressionAST* initializer,
                           SourceLocation location) -> ExpressionAST*;
};

auto ReferenceInitChecker::check(const Type* targetType,
                                 ExpressionAST* initializer,
                                 SourceLocation location) -> ExpressionAST* {
  if (isDependent(ctx.unit, targetType)) return initializer;

  if (!initializer) {
    ctx.error(location,
              std::format("reference variable of type '{}' must be initialized",
                          to_string(targetType)));
    return initializer;
  }

  auto initExpr = Initializer{initializer}.singleExpression();
  if (!initExpr) {
    ctx.error(initializer->firstSourceLocation(),
              "reference initializer must be a single expression");
    return initializer;
  }

  InitializerOperand operand{initializer, initExpr};

  auto seq = ctx.checker.checkImplicitConversion(operand.operand(), targetType);
  if (!seq) {
    if (initExpr->type && isDependent(ctx.unit, initExpr->type))
      return initializer;

    ctx.error(
        initExpr->firstSourceLocation(),
        std::format("invalid initialization of reference of type '{}' from "
                    "expression of type '{}'",
                    to_string(targetType), to_string(initExpr->type)));
    return initializer;
  }

  ctx.checker.applyImplicitConversion(seq, operand.operand());
  return operand.result();
}

struct InitializationEngine {
  InitContext& ctx;
  ElementInitChecker elemChecker;
  DesignatedInitChecker desigChecker;
  AggregateInitChecker aggregateChecker;
  ClassInitChecker classChecker;
  ListInitChecker listChecker;
  ScalarInitChecker scalarChecker;
  ReferenceInitChecker refChecker;

  explicit InitializationEngine(InitContext& ctx)
      : ctx(ctx),
        elemChecker(ctx),
        desigChecker{ctx, elemChecker},
        aggregateChecker{ctx, elemChecker, desigChecker},
        classChecker{ctx, elemChecker, aggregateChecker},
        listChecker{ctx, elemChecker, desigChecker, aggregateChecker},
        scalarChecker{ctx, elemChecker},
        refChecker{ctx} {}

  [[nodiscard]] auto compute(const InitializedEntity& entity,
                             InitializationKind kind,
                             const Initializer& initializer)
      -> InitializationSequence;

  [[nodiscard]] auto apply(InitializationSequence& sequence,
                           const InitializedEntity& entity,
                           Initializer& initializer) -> ExpressionAST*;

  void diagnose(const InitializationSequence& sequence,
                const InitializedEntity& entity,
                const Initializer& initializer);

  void listInitialize(ClassInitChecker::Target& target,
                      BracedInitListAST* list);

 private:
  [[nodiscard]] auto considersConstructors(const Type* destinationType,
                                           InitializationKind kind,
                                           const Initializer& initializer) const
      -> bool;

  [[nodiscard]] auto applyClassInitialization(InitializationSequence& sequence,
                                              const InitializedEntity& entity,
                                              Initializer& initializer)
      -> ExpressionAST*;

  [[nodiscard]] auto applyArrayFromExpressionList(
      const InitializedEntity& entity, Initializer& initializer)
      -> ExpressionAST*;
};

auto InitializationEngine::considersConstructors(
    const Type* destinationType, InitializationKind kind,
    const Initializer& initializer) const -> bool {
  if (isDirectInitialization(kind)) return true;
  if (initializer.form() == InitializerForm::kParen) return true;

  auto source = initializer.singleExpression();
  if (!source || !source->type) return false;

  auto sourceType = ctx.traits.remove_cvref(source->type);
  return ctx.traits.is_same(sourceType, destinationType) ||
         ctx.traits.is_base_of(destinationType, sourceType);
}

auto InitializationEngine::compute(const InitializedEntity& entity,
                                   InitializationKind kind,
                                   const Initializer& initializer)
    -> InitializationSequence {
  InitializationSequence sequence;
  sequence.kind = kind;
  sequence.destinationType = entity.type();

  if (ctx.isTargetTypeUnresolved(entity.type())) {
    sequence.failure = InitializationFailure::kUnresolvedDestinationType;
    return sequence;
  }

  if (auto source = initializer.singleExpression();
      source && source->type && isDependent(ctx.unit, source->type)) {
    sequence.failure = InitializationFailure::kDependent;
    return sequence;
  }

  if (!initializer) {
    if (ctx.traits.is_reference(entity.type())) {
      sequence.failure = InitializationFailure::kReferenceWithoutInitializer;
      return sequence;
    }
    sequence.bullet = InitializationBullet::kDefaultInitialization;
    sequence.kind = InitializationKind::kDirectInitialization;
    return sequence;
  }

  if (initializer.bracedInitList()) {
    sequence.bullet = InitializationBullet::kListInitialization;
    sequence.kind = asListInitialization(kind);
    return sequence;
  }

  if (ctx.traits.is_reference(entity.type())) {
    sequence.bullet = InitializationBullet::kReferenceBinding;
    return sequence;
  }

  const auto destinationType = ctx.traits.remove_cv(entity.type());

  if (stringLiteralInitialization(ctx.traits, ctx.isCxx(), destinationType,
                                  initializer.singleExpression())) {
    sequence.bullet = InitializationBullet::kCharacterArrayFromStringLiteral;
    return sequence;
  }

  if (initializer.form() == InitializerForm::kParen &&
      initializer.arguments().empty()) {
    sequence.bullet = InitializationBullet::kValueInitializationFromParens;
    sequence.kind = InitializationKind::kDirectInitialization;
    return sequence;
  }

  if (ctx.traits.is_array(destinationType)) {
    sequence.bullet = InitializationBullet::kArrayFromExpressionList;
    return sequence;
  }

  if (ctx.traits.is_class(destinationType)) {
    if (ctx.initializesFromSameTypePrvalue(initializer.singleExpression(),
                                           destinationType)) {
      sequence.bullet = InitializationBullet::kSameTypePrvalue;
      return sequence;
    }

    if (considersConstructors(destinationType, kind, initializer)) {
      sequence.bullet = InitializationBullet::kConstructor;
      return sequence;
    }

    sequence.bullet = InitializationBullet::kUserDefinedConversion;
    return sequence;
  }

  sequence.bullet = InitializationBullet::kStandardConversion;
  return sequence;
}

void InitializationEngine::listInitialize(ClassInitChecker::Target& target,
                                          BracedInitListAST* list) {
  auto type = target.type;
  list->type = type;
  if (type && isDependent(ctx.unit, type)) return;

  const auto initializationKind = target.initializationKind.value_or(
      InitializationKind::kCopyListInitialization);

  target.listBullet = listChecker.computeBullet(type, list, initializationKind);

  const auto objectType =
      ctx.traits.remove_cv(ctx.traits.remove_reference(type));

  switch (target.listBullet) {
    case ListInitializationBullet::kDesignatedAggregate:
    case ListInitializationBullet::kAggregate:
      listChecker.aggregateInit(objectType, list);
      return;

    case ListInitializationBullet::kCharacterArrayFromStringLiteral:
      listChecker.characterArrayFromStringLiteral(objectType, list);
      return;

    case ListInitializationBullet::kAggregateFromSameOrDerivedElement:
      classChecker.checkClassInit(target);
      if (!target.constructor) listChecker.aggregateInit(objectType, list);
      return;

    case ListInitializationBullet::kEmptyListDefaultConstructor:
    case ListInitializationBullet::kInitializerList:
    case ListInitializationBullet::kConstructor:
      classChecker.checkClassInit(target);
      return;

    case ListInitializationBullet::kEnumerationWithFixedUnderlyingType:
      listChecker.enumerationFromScalar(objectType, list);
      return;

    case ListInitializationBullet::kSingleElement:
      listChecker.singleElementInit(objectType, list, initializationKind);
      return;

    case ListInitializationBullet::kReferenceToPrvalue:
      listChecker.referenceFromPrvalue(type, list, initializationKind);
      return;

    case ListInitializationBullet::kEmptyListValueInitialization:
      return;

    case ListInitializationBullet::kNone:
      listChecker.diagnoseIllFormed(type, list);
      return;
  }
}

auto InitializationEngine::applyClassInitialization(
    InitializationSequence& sequence, const InitializedEntity& entity,
    Initializer& initializer) -> ExpressionAST* {
  ClassInitChecker::Target target{.type = entity.type(),
                                  .initializer = initializer.node(),
                                  .location = entity.location(),
                                  .argumentList = initializer.argumentList(),
                                  .diagnoseUnresolved = true,
                                  .initializationKind = sequence.kind,
                                  .bullet = sequence.bullet};
  classChecker.checkClassInit(target);
  sequence.constructor = target.constructor;
  initializer.setNode(target.initializer);
  return target.initializer;
}

auto InitializationEngine::applyArrayFromExpressionList(
    const InitializedEntity& entity, Initializer& initializer)
    -> ExpressionAST* {
  auto node = initializer.node();
  auto arrayType = ctx.traits.remove_cv(entity.type());
  auto elementType =
      ctx.traits.remove_cv(ctx.traits.get_element_type(arrayType));

  if (initializer.form() != InitializerForm::kParen) {
    auto source = initializer.singleExpression();
    if (entity.arrayCopyPolicy() == ArrayCopyPolicy::kElementwiseCopyAllowed &&
        isWholeArrayCopy(ctx.traits, source, arrayType)) {
      InitializerOperand operand{node, source};
      ctx.checker.applyImplicitConversion(
          ctx.checker.checkImplicitConversion(operand.operand(), arrayType),
          operand.operand());
      return operand.result();
    }

    ctx.error(node->firstSourceLocation(),
              "array initializer must be an initializer list");
    return node;
  }

  auto slot = initializer.expressionListSlot();
  if (!slot) return node;

  std::size_t elementCount = 0;
  for (auto it = *slot; it; it = it->next) ++elementCount;

  auto bounded = type_cast<BoundedArrayType>(arrayType);
  if (bounded && elementCount > bounded->size()) {
    ctx.error(node->firstSourceLocation(),
              "excess elements in array initializer");
    return node;
  }

  auto pool = ctx.unit->arena();
  auto normalized = BracedInitListAST::create(pool);
  normalized->lbraceLoc = node->firstSourceLocation();
  normalized->rbraceLoc = node->lastSourceLocation();
  normalized->type = entity.type();
  normalized->valueCategory = ValueCategory::kPrValue;

  auto tail = &normalized->expressionList;

  auto element =
      InitializedEntity::arrayElement(elementType, entity.location());

  for (auto it = *slot; it; it = it->next) {
    aggregateChecker.checkElementInit(it->value, elementType,
                                      element.description(),
                                      InitializationKind::kCopyInitialization);

    *tail = make_list_node<ExpressionAST>(pool, it->value);
    tail = &(*tail)->next;
  }

  const auto arraySize = bounded ? bounded->size() : elementCount;

  for (auto index = elementCount; index < arraySize; ++index) {
    *tail = make_list_node<ExpressionAST>(
        pool, aggregateChecker.makeValueInitializer(elementType,
                                                    normalized->lbraceLoc));
    tail = &(*tail)->next;
  }

  initializer.setNode(normalized);
  return normalized;
}

auto InitializationEngine::apply(InitializationSequence& sequence,
                                 const InitializedEntity& entity,
                                 Initializer& initializer) -> ExpressionAST* {
  auto node = initializer.node();
  if (!sequence) return node;

  const auto destinationType = ctx.traits.remove_cv(entity.type());
  const auto initializesClass = ctx.traits.is_class(destinationType);

  switch (sequence.bullet) {
    case InitializationBullet::kNone:
      return node;

    case InitializationBullet::kReferenceBinding:
      return refChecker.check(entity.type(), node, entity.location());

    case InitializationBullet::kCharacterArrayFromStringLiteral:
      (void)checkStringLiteralInitialization(ctx, entity.type(),
                                             initializer.singleExpression());
      initializer.propagateType();
      return node;

    case InitializationBullet::kArrayFromExpressionList:
      return applyArrayFromExpressionList(entity, initializer);

    case InitializationBullet::kListInitialization: {
      ClassInitChecker::Target target{
          .type = initializesClass ? destinationType : entity.type(),
          .initializer = node,
          .location = entity.location(),
          .argumentList = initializer.argumentList(),
          .diagnoseUnresolved = true,
          .initializationKind = sequence.kind,
          .bullet = sequence.bullet};
      listInitialize(target, initializer.bracedInitList());
      sequence.listBullet = target.listBullet;
      sequence.constructor = target.constructor;
      initializer.setNode(target.initializer);
      return target.initializer;
    }

    case InitializationBullet::kSameTypePrvalue:
    case InitializationBullet::kConstructor:
    case InitializationBullet::kParenthesizedAggregate:
    case InitializationBullet::kUserDefinedConversion:
    case InitializationBullet::kDefaultInitialization:
    case InitializationBullet::kValueInitializationFromParens:
      if (initializesClass)
        return applyClassInitialization(sequence, entity, initializer);
      return node;

    case InitializationBullet::kStandardConversion:
      return scalarChecker.check(node, entity.type());

    case InitializationBullet::kValueInitialization:
    case InitializationBullet::kZeroInitialization:
      return node;
  }

  return node;
}

void InitializationEngine::diagnose(const InitializationSequence& sequence,
                                    const InitializedEntity& entity,
                                    const Initializer& initializer) {
  switch (sequence.failure) {
    case InitializationFailure::kReferenceWithoutInitializer:
      ctx.error(entity.location(),
                std::format("reference variable of type '{}' must be "
                            "initialized",
                            to_string(entity.type())));
      return;
    default:
      return;
  }
}

}  // namespace

auto materializedTemporary(const TypeTraits& traits, ExpressionAST* expr)
    -> MaterializedTemporary {
  if (!expr || !expr->type) return {};

  if (is_prvalue(expr)) return {expr};

  if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
    if (cast->castKind == ImplicitCastKind::kUserDefinedConversion) return {};
    return materializedTemporary(traits, cast->expression);
  }

  if (auto nested = ast_cast<NestedExpressionAST>(expr))
    return materializedTemporary(traits, nested->expression);

  if (auto subscript = ast_cast<SubscriptExpressionAST>(expr)) {
    if (subscript->symbol) return {};
    auto base = subscript->baseExpression;
    if (!base || !traits.is_array(traits.remove_reference(base->type)))
      return {};
    return materializedTemporary(traits, base);
  }

  if (auto member = ast_cast<MemberExpressionAST>(expr)) {
    if (member->accessOp != TokenKind::T_DOT) return {};
    if (traits.is_reference(member->type)) return {};
    if (!symbol_cast<FieldSymbol>(member->symbol)) return {};
    return materializedTemporary(traits, member->baseExpression);
  }

  if (auto cast = ast_cast<CppCastExpressionAST>(expr))
    return materializedTemporary(traits, cast->expression);

  if (auto conditional = ast_cast<ConditionalExpressionAST>(expr)) {
    auto found = materializedTemporary(traits, conditional->iftrueExpression);
    if (!found)
      found = materializedTemporary(traits, conditional->iffalseExpression);
    found.conditional = bool(found);
    return found;
  }

  if (auto binary = ast_cast<BinaryExpressionAST>(expr)) {
    if (binary->op == TokenKind::T_COMMA)
      return materializedTemporary(traits, binary->rightExpression);

    if (binary->op == TokenKind::T_DOT_STAR && !binary->symbol &&
        !traits.is_reference(binary->type))
      return materializedTemporary(traits, binary->leftExpression);
  }

  return {};
}

auto singleInitializerClause(BracedInitListAST* bracedInitList)
    -> ExpressionAST* {
  if (!bracedInitList) return nullptr;
  auto elements = bracedInitList->expressionList;
  if (!elements || elements->next) return nullptr;
  return elements->value;
}

auto stringLiteralInitialization(const TypeTraits& traits, bool isCxx,
                                 const Type* destinationType,
                                 ExpressionAST* source)
    -> std::optional<StringLiteralInitialization> {
  if (!traits.is_array(destinationType)) return std::nullopt;

  auto literal = ast_cast<StringLiteralExpressionAST>(source);
  if (!literal || !literal->type) return std::nullopt;

  auto literalType = traits.remove_cv(traits.remove_reference(literal->type));
  if (!traits.is_array(literalType)) return std::nullopt;

  StringLiteralInitialization init;
  init.destinationElementType =
      traits.remove_cv(traits.get_element_type(destinationType));
  init.sourceElementType =
      traits.remove_cv(traits.get_element_type(literalType));

  if (!traits.is_char_type(init.destinationElementType)) return std::nullopt;

  const auto ordinaryLiteral =
      traits.is_narrow_char_type(init.sourceElementType);
  const auto utf8Literal = type_cast<Char8Type>(init.sourceElementType);

  init.compatible =
      traits.is_same(init.destinationElementType, init.sourceElementType) ||
      (ordinaryLiteral &&
       traits.is_narrow_char_type(init.destinationElementType)) ||
      (utf8Literal &&
       (type_cast<CharType>(init.destinationElementType) ||
        type_cast<UnsignedCharType>(init.destinationElementType)));

  if (auto sourceArray = type_cast<BoundedArrayType>(literalType)) {
    init.elementCount = sourceArray->size();
    init.minimumElements = isCxx ? init.elementCount : init.elementCount - 1;
  }

  if (auto destinationArray = type_cast<BoundedArrayType>(destinationType)) {
    init.bounded = true;
    init.availableElements = destinationArray->size();
  }

  return init;
}

auto planAggregateInitialization(TranslationUnit* unit,
                                 const Type* aggregateType,
                                 BracedInitListAST* bracedInitList)
    -> std::optional<AggregateInitializerPlan> {
  AggregateInitializerBuilder builder{unit, nullptr};
  return builder.build(aggregateType, bracedInitList);
}

auto resolveAggregateInitialization(InitContext& ctx, const Type* aggregateType,
                                    BracedInitListAST* bracedInitList)
    -> std::optional<AggregateInitializerPlan> {
  AggregateInitializerBuilder builder{ctx.unit, &ctx};
  return builder.build(aggregateType, bracedInitList);
}

void diagnoseNarrowingListElement(InitContext& ctx, ExpressionAST* element,
                                  const Type* targetType) {
  if (!ctx.isCxx()) return;
  if (!element || !element->type) return;
  if (!ctx.traits.is_narrowing_list_element(element, targetType)) return;

  auto source = Initializer::stripImplicitCasts(element);
  ctx.error(element->firstSourceLocation(),
            std::format("narrowing conversion from '{}' to '{}' in "
                        "braced-init-list",
                        to_string(source->type), to_string(targetType)));
}

auto computeInitializationSequence(InitContext& ctx,
                                   const InitializedEntity& entity,
                                   InitializationKind kind,
                                   const Initializer& initializer)
    -> InitializationSequence {
  return InitializationEngine{ctx}.compute(entity, kind, initializer);
}

auto applyInitializationSequence(InitContext& ctx,
                                 InitializationSequence& sequence,
                                 const InitializedEntity& entity,
                                 Initializer& initializer) -> ExpressionAST* {
  return InitializationEngine{ctx}.apply(sequence, entity, initializer);
}

void reportRejectedConstructors(InitContext& ctx,
                                const ConstructorResult& resolution) {
  std::vector<std::pair<SourceLocation, std::string>> reported;

  for (const auto& [symbol, reason] : resolution.rejected) {
    if (!symbol) continue;

    std::pair entry{symbol->location(), reason};
    if (std::ranges::contains(reported, entry)) continue;
    reported.push_back(entry);

    ctx.checker.note(
        symbol->location(),
        std::format("candidate constructor not viable: {}", reason));
  }
}

void diagnoseConversionFailure(InitContext& ctx,
                               const InitializedEntity& entity,
                               ExpressionAST* source) {
  if (!source || !source->type) return;
  if (isUntypedAfterError(source)) return;
  if (isDependent(ctx.unit, source->type)) return;
  if (isDependent(ctx.unit, entity.type())) return;
  if (containsPlaceholderType(source->type)) return;

  auto location = entity.location();
  if (!location) location = source->firstSourceLocation();

  auto targetType = ctx.traits.remove_cv(entity.type());

  if (auto classType = unqualified_cast<ClassType>(targetType);
      classType && classType->symbol()) {
    ctx.error(location,
              std::format("no viable conversion from '{}' to '{}'",
                          to_string(source->type), to_string(targetType)));

    OverloadResolution overloadRes(ctx.unit);
    reportRejectedConstructors(
        ctx, overloadRes.resolveConstructor(
                 classType->symbol()->resolvedDefinition(), {source},
                 InitializationKind::kCopyInitialization));
    return;
  }

  ctx.error(location,
            std::format("cannot initialize {} of type '{}' with an "
                        "expression of type '{}'",
                        entity.description(), to_string(entity.type()),
                        to_string(source->type)));
}

void diagnoseInitializationFailure(InitContext& ctx,
                                   const InitializationSequence& sequence,
                                   const InitializedEntity& entity,
                                   const Initializer& initializer) {
  InitializationEngine{ctx}.diagnose(sequence, entity, initializer);
}

void TypeChecker::check_braced_init_list(
    const Type* type, BracedInitListAST* ast,
    InitializationKind initializationKind) {
  InitContext ctx{*this};
  InitializationEngine engine{ctx};

  ClassInitChecker::Target target{
      .type = type,
      .initializer = ast,
      .location = ast->firstSourceLocation(),
      .diagnoseUnresolved = true,
      .initializationKind = initializationKind,
      .bullet = InitializationBullet::kListInitialization};

  engine.listInitialize(target, ast);

  auto classType = type_cast<ClassType>(ctx.traits.remove_cv(type));
  if (classType && !ctx.traits.is_aggregate(classType)) {
    ast->type = type;
    ast->valueCategory = ValueCategory::kPrValue;
  }
}

void TypeChecker::check_list_initialization(
    const Type* type, ExpressionAST*& expression,
    InitializationKind initializationKind) {
  auto braced = ast_cast<BracedInitListAST>(expression);
  if (!braced) return;
  InitContext ctx{*this};
  InitializationEngine engine{ctx};
  ClassInitChecker::Target target{
      .type = type,
      .initializer = braced,
      .location = braced->firstSourceLocation(),
      .diagnoseUnresolved = true,
      .initializationKind = initializationKind,
      .bullet = InitializationBullet::kListInitialization};
  engine.listInitialize(target, braced);
  if (!target.constructor) return;
  auto arguments = BracedInitListAST::create(unit_->arena());
  arguments->lbraceLoc = braced->lbraceLoc;
  arguments->rbraceLoc = braced->rbraceLoc;
  auto tail = &arguments->expressionList;
  for (auto argument : Initializer{target.initializer}.arguments()) {
    *tail = make_list_node<ExpressionAST>(unit_->arena(), argument);
    tail = &(*tail)->next;
  }
  expression =
      makeClassConstruction(unit_, type, target.constructor, arguments);
}

auto TypeChecker::check_class_initializer(const Type* targetType,
                                          ExpressionAST*& initializer,
                                          SourceLocation location,
                                          List<ExpressionAST*>** argumentList)
    -> FunctionSymbol* {
  InitContext ctx{*this};

  auto entity = InitializedEntity::temporary(targetType, location);
  auto init = Initializer::withArgumentList(initializer, argumentList);

  auto sequence = computeInitializationSequence(
      ctx, entity, init.initializationKind(), init);

  if (!sequence) return nullptr;

  initializer = applyInitializationSequence(ctx, sequence, entity, init);
  return sequence.constructor;
}

}  // namespace cxx
