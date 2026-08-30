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
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>
#include <unordered_set>

namespace cxx {

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
  return stripImplicitCasts(node_);
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

struct StringInitChecker {
  InitContext& ctx;

  [[nodiscard]] auto isStringToCharArrayInit(ExpressionAST* expr,
                                             const Type* targetType) -> bool {
    if (!ctx.traits.is_array(targetType)) return false;
    auto strLit = ast_cast<StringLiteralExpressionAST>(expr);
    if (!strLit) return false;
    auto destElem = elementType(targetType);
    auto srcElem = elementType(strLit->type);
    return ctx.traits.is_same(destElem, srcElem) ||
           (ctx.traits.is_narrow_char_type(destElem) &&
            ctx.traits.is_narrow_char_type(srcElem));
  }

  void checkStringLength(SourceLocation loc, const Type* destArrayType,
                         const Type* srcArrayType) {
    auto destArray = type_cast<BoundedArrayType>(destArrayType);
    auto srcArray = type_cast<BoundedArrayType>(srcArrayType);
    if (!destArray || !srcArray) return;
    auto requiredElements =
        ctx.isCxx() ? srcArray->size() : srcArray->size() - 1;
    if (requiredElements > destArray->size())
      ctx.error(loc, "initializer-string for char array is too long");
  }

  [[nodiscard]] auto elementType(const Type* type) -> const Type* {
    return ctx.traits.remove_cv(ctx.traits.get_element_type(type));
  }
};

struct ElementInitChecker {
  InitContext& ctx;
  StringInitChecker stringInit;

  explicit ElementInitChecker(InitContext& ctx) : ctx(ctx), stringInit{ctx} {}

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

    if (ctx.traits.is_lvalue_reference(targetType) && is_lvalue(expr)) {
      if (checkDirectLvalueBinding(expr, targetType)) return;
    }

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
    if (stringInit.isStringToCharArrayInit(expr, targetType)) {
      stringInit.checkStringLength(expr->firstSourceLocation(), targetType,
                                   expr->type);
      return;
    }

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

  [[nodiscard]] auto checkDirectLvalueBinding(ExpressionAST* expr,
                                              const Type* targetType) -> bool {
    return ctx.traits.is_reference_compatible(
        ctx.traits.remove_reference(targetType),
        ctx.traits.remove_reference(expr->type));
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

  auto construction = BracedTypeConstructionAST::create(arena);
  construction->bracedInitList = arguments;
  construction->constructorSymbol = constructor;
  construction->type = targetType;
  construction->valueCategory = ValueCategory::kPrValue;

  expr = construction;
  return true;
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
  ctx.checker.check(subscript->expression);
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

  [[nodiscard]] auto tryBraceElision(List<ExpressionAST*>*& it,
                                     const Type* targetType) -> bool;

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
  [[nodiscard]] auto declaresMemberOf(ClassSymbol* classSymbol,
                                      ClassSymbol* owner) const -> bool;
  [[nodiscard]] auto elementIndexOf(const std::vector<Symbol*>& elements,
                                    Symbol* member) const
      -> std::optional<std::size_t>;
  [[nodiscard]] auto designatedElementIndex(
      const Type* classType, const std::vector<Symbol*>& elements,
      DesignatedInitializerClauseAST* desig) -> std::optional<std::size_t>;
  [[nodiscard]] auto checkDesignatedElementInit(
      Symbol* element, const Type* type,
      const std::vector<DesignatedInitializerClauseAST*>& clauses)
      -> ExpressionAST*;
  [[nodiscard]] auto defaultMemberInitializer(FieldSymbol* field)
      -> ExpressionAST*;
  [[nodiscard]] auto implicitElementInitializer(Symbol* element,
                                                SourceLocation location)
      -> ExpressionAST*;

  void checkElementInit(ExpressionAST*& expr, Symbol* element);
  void checkAnonUnionFieldInit(ExpressionAST*& expr, const Type* fieldType);
  void initializeUnionByDefault(ClassSymbol* classSymbol,
                                BracedInitListAST* ast);

  [[nodiscard]] auto isSubAggregate(const Type* type) const -> bool;

  [[nodiscard]] auto countScalarInitSlots(const Type* type) const -> size_t;

  [[nodiscard]] auto countScalarInitSlots(
      const Type* type,
      std::unordered_set<const ClassSymbol*>& enclosingClasses) const -> size_t;

  [[nodiscard]] auto buildSyntheticBracedList(List<ExpressionAST*>*& it,
                                              size_t maxCount)
      -> BracedInitListAST*;
};

auto AggregateInitChecker::elementDescription(Symbol* element) const
    -> std::string {
  if (auto field = symbol_cast<FieldSymbol>(element)) {
    if (field->name())
      return std::format("member '{}'", to_string(field->name()));
    return std::format("anonymous member of type '{}'",
                       to_string(ctx.traits.aggregate_element_type(element)));
  }
  return std::format("base class '{}'",
                     to_string(ctx.traits.aggregate_element_type(element)));
}

auto AggregateInitChecker::declaresMemberOf(ClassSymbol* classSymbol,
                                            ClassSymbol* owner) const -> bool {
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

auto AggregateInitChecker::elementIndexOf(const std::vector<Symbol*>& elements,
                                          Symbol* member) const
    -> std::optional<std::size_t> {
  for (std::size_t i = 0; i < elements.size(); ++i)
    if (elements[i] == member) return i;

  auto owner = symbol_cast<ClassSymbol>(member->parent());
  if (!owner) return std::nullopt;
  owner = owner->resolvedDefinition();

  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto classType =
        type_cast<ClassType>(ctx.traits.aggregate_element_type(elements[i]));
    if (!classType) continue;
    if (declaresMemberOf(classType->symbol(), owner)) return i;
  }

  return std::nullopt;
}

auto AggregateInitChecker::designatedElementIndex(
    const Type* classType, const std::vector<Symbol*>& elements,
    DesignatedInitializerClauseAST* desig) -> std::optional<std::size_t> {
  if (!desig->designatorList) return std::nullopt;

  auto dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
  if (!dot) {
    desigChecker.check(classType, desig);
    return std::nullopt;
  }

  if (!desigChecker.resolveDotDesignator(classType, dot)) return std::nullopt;

  auto index = elementIndexOf(elements, dot->symbol);
  if (!index) {
    ctx.error(
        get_name_location(dot),
        std::format("field designator '{}' does not designate an "
                    "element of '{}'",
                    to_string(dot->symbol->name()), to_string(classType)));
  }
  return index;
}

auto AggregateInitChecker::checkDesignatedElementInit(
    Symbol* element, const Type* type,
    const std::vector<DesignatedInitializerClauseAST*>& clauses)
    -> ExpressionAST* {
  auto pool = ctx.unit->arena();

  auto designatesWholeElement = [&] {
    if (clauses.size() != 1) return false;
    auto designators = clauses.front()->designatorList;
    if (!designators || designators->next) return false;
    auto dot = ast_cast<DotDesignatorAST>(designators->value);
    if (!dot) return false;
    return dot->symbol == element;
  };

  if (designatesWholeElement()) {
    auto initializer = clauses.front()->initializer;
    if (auto equal = ast_cast<EqualInitializerAST>(initializer))
      initializer = equal->expression;
    if (!initializer) return nullptr;
    checkElementInit(initializer, element);
    return initializer;
  }

  auto nested = BracedInitListAST::create(pool);
  nested->lbraceLoc = clauses.front()->firstSourceLocation();
  nested->rbraceLoc = clauses.back()->lastSourceLocation();

  auto tail = &nested->expressionList;
  for (auto clause : clauses) {
    auto designators = clause->designatorList;
    auto dot = ast_cast<DotDesignatorAST>(designators->value);
    if (dot->symbol == element) designators = designators->next;

    auto rewritten = DesignatedInitializerClauseAST::create(pool);
    rewritten->designatorList = designators;
    rewritten->initializer = clause->initializer;

    *tail = make_list_node<ExpressionAST>(pool, rewritten);
    tail = &(*tail)->next;
  }

  ExpressionAST* initializer = nested;
  checkElementInit(initializer, element);
  return initializer;
}

auto AggregateInitChecker::defaultMemberInitializer(FieldSymbol* field)
    -> ExpressionAST* {
  auto initializer = field->initializer();
  if (!initializer) return nullptr;

  if (auto equal = ast_cast<EqualInitializerAST>(initializer))
    initializer = equal->expression;

  auto fieldType = ctx.traits.remove_cv(field->type());
  auto pool = ctx.unit->arena();

  if (auto constructor = field->constructor()) {
    auto arguments = BracedInitListAST::create(pool);
    if (auto paren = ast_cast<ParenInitializerAST>(initializer))
      arguments->expressionList = paren->expressionList;
    else if (auto braced = ast_cast<BracedInitListAST>(initializer))
      arguments->expressionList = braced->expressionList;
    else
      arguments->expressionList =
          make_list_node<ExpressionAST>(pool, initializer);

    auto construction = BracedTypeConstructionAST::create(pool);
    construction->bracedInitList = arguments;
    construction->constructorSymbol = constructor;
    construction->type = fieldType;
    construction->valueCategory = ValueCategory::kPrValue;
    return construction;
  }

  const auto initializesScalar = !ctx.traits.is_class_or_union(fieldType) &&
                                 !ctx.traits.is_array(fieldType);

  if (auto braced = ast_cast<BracedInitListAST>(initializer);
      braced && initializesScalar) {
    if (!braced->expressionList)
      return makeValueInitializer(fieldType, braced->lbraceLoc);
    return braced->expressionList->value;
  }

  return initializer;
}

auto AggregateInitChecker::makeValueInitializer(const Type* type,
                                                SourceLocation location)
    -> ExpressionAST* {
  auto pool = ctx.unit->arena();
  auto braced = BracedInitListAST::create(pool);
  braced->lbraceLoc = location;
  braced->rbraceLoc = location;

  if (!type) return braced;

  if (ctx.traits.is_class_or_union(type)) {
    ExpressionAST* initializer = braced;
    auto constructor =
        ctx.checker.check_class_initializer(type, initializer, location);
    if (!constructor) return initializer;

    auto construction = BracedTypeConstructionAST::create(pool);
    construction->bracedInitList = braced;
    construction->constructorSymbol = constructor;
    construction->type = type;
    construction->valueCategory = ValueCategory::kPrValue;
    return construction;
  }

  ctx.checker.check_braced_init_list(
      type, braced, InitializationKind::kCopyListInitialization);
  return braced;
}

auto AggregateInitChecker::implicitElementInitializer(Symbol* element,
                                                      SourceLocation location)
    -> ExpressionAST* {
  if (auto field = symbol_cast<FieldSymbol>(element)) {
    if (auto initializer = defaultMemberInitializer(field)) return initializer;
  }

  auto type = ctx.traits.aggregate_element_type(element);

  if (type && ctx.traits.is_reference(type)) {
    ctx.error(location, std::format("reference {} is not initialized",
                                    elementDescription(element)));
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
      ExpressionAST* initializer = expr;
      auto constructor = ctx.checker.check_class_initializer(
          type, initializer, expr->firstSourceLocation());
      if (!constructor) {
        expr = initializer;
        return;
      }

      auto pool = ctx.unit->arena();
      auto construction = BracedTypeConstructionAST::create(pool);
      construction->bracedInitList = ast_cast<BracedInitListAST>(initializer);
      construction->constructorSymbol = constructor;
      construction->type = type;
      construction->valueCategory = ValueCategory::kPrValue;
      expr = construction;
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

auto AggregateInitChecker::isSubAggregate(const Type* type) const -> bool {
  type = ctx.traits.remove_cv(type);
  if (type_cast<BoundedArrayType>(type)) return true;
  if (auto ct = type_cast<ClassType>(type)) {
    auto cls = ct->symbol();
    if (!cls) return false;
    if (!ctx.isCxx()) return true;
    for (auto ctor : cls->constructors())
      if (!ctor->isDefaulted() && !ctor->isDeleted()) return false;
    return true;
  }
  return false;
}

auto AggregateInitChecker::countScalarInitSlots(const Type* type) const
    -> size_t {
  std::unordered_set<const ClassSymbol*> enclosingClasses;
  return countScalarInitSlots(type, enclosingClasses);
}

auto AggregateInitChecker::countScalarInitSlots(
    const Type* type,
    std::unordered_set<const ClassSymbol*>& enclosingClasses) const -> size_t {
  type = ctx.traits.remove_cv(type);

  if (auto bt = type_cast<BoundedArrayType>(type))
    return bt->size() *
           countScalarInitSlots(bt->elementType(), enclosingClasses);

  if (auto ct = type_cast<ClassType>(type)) {
    auto cls = ct->symbol();
    if (!cls) return 1;
    if (!enclosingClasses.insert(cls).second) return 1;
    size_t total = 0;
    for (auto m : cls->members()) {
      auto f = symbol_cast<FieldSymbol>(m);
      if (!f || f->isStatic()) continue;
      if (cls->isUnion()) {
        total = countScalarInitSlots(f->type(), enclosingClasses);
        break;
      }
      total += countScalarInitSlots(f->type(), enclosingClasses);
    }
    enclosingClasses.erase(cls);
    return total > 0 ? total : 1;
  }

  return 1;
}

auto AggregateInitChecker::buildSyntheticBracedList(List<ExpressionAST*>*& it,
                                                    size_t maxCount)
    -> BracedInitListAST* {
  auto arena = ctx.unit->arena();
  auto syntheticList = BracedInitListAST::create(arena);
  List<ExpressionAST*>* head = nullptr;
  List<ExpressionAST*>* tail = nullptr;
  size_t consumed = 0;
  auto prev = it;
  while (it && consumed < maxCount) {
    auto node = make_list_node(arena, it->value);
    if (!head)
      head = tail = node;
    else {
      tail->next = node;
      tail = node;
    }
    ++consumed;
    prev = it;
    it = it->next;
  }
  it = prev;
  syntheticList->expressionList = head;
  return syntheticList;
}

auto AggregateInitChecker::tryBraceElision(List<ExpressionAST*>*& it,
                                           const Type* targetType) -> bool {
  auto& expr = it->value;

  if (ast_cast<BracedInitListAST>(expr)) return false;

  if (ast_cast<StringLiteralExpressionAST>(expr) &&
      ctx.traits.is_array(targetType) &&
      ctx.traits.is_narrow_char_type(
          ctx.traits.remove_cv(ctx.traits.get_element_type(targetType))))
    return false;

  if (ctx.traits.is_compatible(expr->type, targetType)) return false;

  if (!isSubAggregate(targetType)) return false;

  size_t slots = countScalarInitSlots(targetType);
  auto firstNode = it;
  auto synthetic = buildSyntheticBracedList(it, slots);
  ctx.checker.check_braced_init_list(
      targetType, synthetic, InitializationKind::kCopyListInitialization);

  auto afterRun = it->next;
  firstNode->value = synthetic;
  firstNode->next = afterRun;
  it = firstNode;
  return true;
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

  auto initializer = defaultMemberInitializer(variantMember);
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

  std::vector<Symbol*> variantMembers;
  for (auto field : views::members(classSymbol) | views::non_static_fields)
    variantMembers.push_back(field);

  auto classType = ctx.control->getClassType(classSymbol);

  std::vector<DesignatedInitializerClauseAST*> clauses;
  ExpressionAST* positionalInitializer = nullptr;
  std::optional<std::size_t> activeIndex;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value)) {
      auto index = designatedElementIndex(classType, variantMembers, desig);
      if (!index) continue;

      if (activeIndex && *activeIndex != *index) {
        ctx.error(desig->firstSourceLocation(),
                  "initializing multiple members of a union");
        continue;
      }

      activeIndex = index;
      clauses.push_back(desig);
      continue;
    }

    if (activeIndex) {
      ctx.error(it->value->firstSourceLocation(),
                "excess elements in union initializer");
      break;
    }

    if (variantMembers.empty()) {
      ctx.error(it->value->firstSourceLocation(), "union has no named members");
      return;
    }

    auto element = variantMembers.front();
    if (!tryBraceElision(it, ctx.traits.aggregate_element_type(element)))
      checkElementInit(it->value, element);

    positionalInitializer = it->value;
    activeIndex = 0;
  }

  if (!activeIndex) return;

  auto element = variantMembers[*activeIndex];

  auto initializer = positionalInitializer;
  if (!initializer)
    initializer = checkDesignatedElementInit(
        element, ctx.traits.aggregate_element_type(element), clauses);
  if (!initializer) return;

  auto pool = ctx.unit->arena();

  if (*activeIndex == 0) {
    ast->expressionList = make_list_node<ExpressionAST>(pool, initializer);
    return;
  }

  auto field = symbol_cast<FieldSymbol>(element);

  auto dot = DotDesignatorAST::create(pool);
  dot->identifier = name_cast<Identifier>(field->name());
  dot->symbol = field;

  auto clause = DesignatedInitializerClauseAST::create(pool);
  clause->designatorList = make_list_node<DesignatorAST>(pool, dot);
  clause->initializer = initializer;
  clause->type = ctx.traits.aggregate_element_type(element);

  ast->expressionList = make_list_node<ExpressionAST>(pool, clause);
}

void AggregateInitChecker::checkStruct(ClassSymbol* classSymbol,
                                       BracedInitListAST* ast) {
  AggregateInitGuard guard{ctx.checker, classSymbol};
  if (!guard) return;

  auto elements = ctx.traits.aggregate_elements(classSymbol);
  auto classType = ctx.control->getClassType(classSymbol);

  std::vector<ExpressionAST*> initializers(elements.size(), nullptr);
  std::vector<std::vector<DesignatedInitializerClauseAST*>> designated(
      elements.size());

  std::size_t elementIndex = 0;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value)) {
      auto index = designatedElementIndex(classType, elements, desig);
      if (!index) continue;
      designated[*index].push_back(desig);
      elementIndex = *index + 1;
      continue;
    }

    if (elementIndex >= elements.size()) {
      ctx.error(it->value->firstSourceLocation(),
                "excess elements in struct initializer");
      break;
    }

    auto element = elements[elementIndex];

    if (!tryBraceElision(it, ctx.traits.aggregate_element_type(element)))
      checkElementInit(it->value, element);

    initializers[elementIndex] = it->value;
    ++elementIndex;
  }

  auto pool = ctx.unit->arena();
  List<ExpressionAST*>* normalized = nullptr;
  auto tail = &normalized;

  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto element = elements[i];
    auto initializer = initializers[i];

    if (!designated[i].empty()) {
      if (initializer) {
        ctx.error(designated[i].front()->firstSourceLocation(),
                  std::format("multiple initializations of {}",
                              elementDescription(element)));
      }
      initializer = checkDesignatedElementInit(
          element, ctx.traits.aggregate_element_type(element), designated[i]);
    }

    if (!initializer)
      initializer = implicitElementInitializer(element, ast->lbraceLoc);

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
  StringInitChecker& stringInit;

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
  [[nodiscard]] static auto singleElement(BracedInitListAST* ast)
      -> ExpressionAST*;

  void checkArrayElements(const Type* type, const Type* elementType,
                          BracedInitListAST* ast);

 private:
  [[nodiscard]] auto initializesCharacterArrayFromStringLiteral(
      const Type* type, BracedInitListAST* ast) const -> bool;

  [[nodiscard]] auto hasDefaultConstructor(const Type* type) const -> bool;

  [[nodiscard]] auto initializesFixedUnderlyingTypeEnumeration(
      const Type* type, ExpressionAST* element) const -> bool;

  void checkArrayStringElement(ExpressionAST*& expr, const Type* elementType);
};

auto ListInitChecker::hasDesignators(BracedInitListAST* ast) -> bool {
  for (auto it = ast->expressionList; it; it = it->next)
    if (ast_cast<DesignatedInitializerClauseAST>(it->value)) return true;
  return false;
}

auto ListInitChecker::singleElement(BracedInitListAST* ast) -> ExpressionAST* {
  if (!ast->expressionList || ast->expressionList->next) return nullptr;
  return ast->expressionList->value;
}

auto ListInitChecker::initializesCharacterArrayFromStringLiteral(
    const Type* type, BracedInitListAST* ast) const -> bool {
  if (!ctx.traits.is_array(type)) return false;
  auto element = singleElement(ast);
  auto literal = ast_cast<StringLiteralExpressionAST>(element);
  if (!literal || !literal->type) return false;
  auto destinationElement =
      ctx.traits.remove_cv(ctx.traits.get_element_type(type));
  if (!ctx.traits.is_char_type(destinationElement)) return false;
  auto sourceElement =
      ctx.traits.remove_cv(ctx.traits.get_element_type(literal->type));
  return ctx.traits.is_same(destinationElement, sourceElement) ||
         (ctx.traits.is_narrow_char_type(destinationElement) &&
          ctx.traits.is_narrow_char_type(sourceElement));
}

auto ListInitChecker::hasDefaultConstructor(const Type* type) const -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return false;
  return classType->symbol()->resolvedDefinition()->defaultConstructor();
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
  auto element = singleElement(ast);

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

  if (!isReference && initializesCharacterArrayFromStringLiteral(type, ast))
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
  auto literal = ast_cast<StringLiteralExpressionAST>(singleElement(ast));
  stringInit.checkStringLength(literal->firstSourceLocation(), type,
                               literal->type);
}

void ListInitChecker::checkArrayElements(const Type* type,
                                         const Type* elementType,
                                         BracedInitListAST* ast) {
  auto interp = ASTInterpreter{ctx.unit};
  size_t index = 0;

  std::vector<std::pair<size_t, ExpressionAST*>> placements;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (ast_cast<PackExpansionExpressionAST>(it->value)) {
      ++index;
      continue;
    }

    auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value);

    if (desig && desig->designatorList) {
      if (auto subscript =
              ast_cast<SubscriptDesignatorAST>(desig->designatorList->value)) {
        if (auto val = interp.evaluate(subscript->expression))
          if (auto idx = interp.toUInt(*val)) index = *idx;
      }
    }

    if (auto bounded = type_cast<BoundedArrayType>(type)) {
      if (index >= bounded->size()) {
        ctx.error(it->value->firstSourceLocation(),
                  "excess elements in array initializer");
        break;
      }
    }

    if (auto nested = ast_cast<BracedInitListAST>(it->value)) {
      ctx.checker.check_braced_init_list(
          elementType, nested, InitializationKind::kCopyListInitialization);
    } else if (desig) {
      desigChecker.check(type, desig);
    } else if (auto strLit = ast_cast<StringLiteralExpressionAST>(it->value);
               strLit && ctx.traits.is_array(elementType)) {
      checkArrayStringElement(it->value, elementType);
    } else if (!aggregateChecker.tryBraceElision(it, elementType)) {
      elemChecker.check(
          it->value, elementType,
          std::format("cannot initialize array element of type '{}' with "
                      "expression of type '{}'",
                      to_string(elementType), to_string(it->value->type)));
    }

    placements.emplace_back(index, it->value);
    ++index;
  }

  auto bounded = type_cast<BoundedArrayType>(type);
  if (!bounded) return;
  if (placements.size() == bounded->size()) return;
  if (ctx.traits.is_trivially_constructible(elementType)) return;

  std::vector<ExpressionAST*> slots(bounded->size(), nullptr);
  for (auto& [slot, initializer] : placements) slots[slot] = initializer;

  auto pool = ctx.unit->arena();
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

void ListInitChecker::checkArrayStringElement(ExpressionAST*& expr,
                                              const Type* elementType) {
  auto strLit = ast_cast<StringLiteralExpressionAST>(expr);
  auto destElem =
      ctx.traits.remove_cv(ctx.traits.get_element_type(elementType));
  auto srcElem =
      ctx.traits.remove_cv(ctx.traits.get_element_type(strLit->type));
  bool compatible = ctx.traits.is_same(destElem, srcElem) ||
                    (ctx.traits.is_narrow_char_type(destElem) &&
                     ctx.traits.is_narrow_char_type(srcElem));
  if (!compatible) {
    ctx.error(expr->firstSourceLocation(),
              std::format("cannot initialize array element of type '{}' with "
                          "expression of type '{}'",
                          to_string(elementType), to_string(strLit->type)));
  } else {
    stringInit.checkStringLength(expr->firstSourceLocation(), elementType,
                                 strLit->type);
  }
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

  const auto referencedCv = cv_qualifiers(referencedType);
  const auto bindsPrvalue =
      type_cast<RvalueReferenceType>(type) ||
      (has_const(referencedCv) && !has_volatile(referencedCv));

  if (!bindsPrvalue) {
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

  [[nodiscard]] auto tryInitializerListConstructor(
      Target& target, BracedInitListAST* bracedInitList,
      ClassSymbol* classSymbol, OverloadResolution& overloadRes,
      ConstructorResult& resolution) -> bool;
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
    case InitializationBullet::kUserDefinedConversion:
      checkConstructorInit(target, classSymbol, diagnoseUnresolved);
      if (target.constructor || !isAggregate) return;
      if (Initializer::withArgumentList(target.initializer, target.argumentList)
              .form() == InitializerForm::kParen)
        checkParenthesizedAggregateInit(target, classSymbol);
      else
        checkAggregateInit(target, classSymbol);
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

  ConstructorResult resolution;

  bool selectedInitializerListConstructor = false;
  if (bracedInitList) {
    selectedInitializerListConstructor = tryInitializerListConstructor(
        target, bracedInitList, classSymbol, overloadRes, resolution);
  }

  if (!selectedInitializerListConstructor) {
    resolution = overloadRes.resolveConstructor(classSymbol, args,
                                                initializationKindOf());
  }

  if (!resolution.best) {
    if (diagnoseUnresolved) {
      ctx.error(
          location,
          std::format("no matching constructor for initialization of '{}'",
                      to_string(classSymbol->type())));
      reportRejectedConstructors(resolution);
    }
    return;
  }

  if (resolution.ambiguous) {
    ctx.error(location, std::format("call to constructor of '{}' is ambiguous",
                                    to_string(classSymbol->type())));
    for (const auto& candidate : resolution.candidates) {
      if (!candidate.viable || !candidate.symbol) continue;
      ctx.checker.note(candidate.symbol->location(),
                       std::format("candidate constructor '{}'",
                                   to_string(candidate.symbol->type())));
    }
    return;
  }

  if (initializationKindOf() == InitializationKind::kCopyListInitialization &&
      resolution.best->symbol->isExplicit()) {
    ctx.error(location,
              "chosen constructor is explicit in copy-initialization");
    ctx.checker.note(resolution.best->symbol->location(),
                     "explicit constructor declared here");
    return;
  }

  target.constructor = resolution.best->symbol;
  ctx.checker.checkConstructorAccess(target.constructor, target.location);
  ctx.checker.useFunction(target.constructor, target.location);
  if (selectedInitializerListConstructor) {
    appendDefaultArguments(target, target.constructor);
    return;
  }
  if (bracedInitList) checkNarrowingArguments(args, target.constructor);
  applyArgumentConversions(target, resolution.best->conversions);
  appendDefaultArguments(target, target.constructor);
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

  auto argCount = arguments(target).size();
  if (argCount >= params.size()) return;
  if (!params[argCount]->defaultArgument()) return;

  auto tail = argumentListSlot(target, ctx.unit->arena());
  if (!tail) return;

  ctx.checker.append_default_arguments(constructor, tail);
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

auto ClassInitChecker::tryInitializerListConstructor(
    Target& target, BracedInitListAST* bracedInitList, ClassSymbol* classSymbol,
    OverloadResolution& overloadRes, ConstructorResult& resolution) -> bool {
  const auto emptyListSelectsDefaultConstructor =
      !bracedInitList->expressionList && classSymbol->defaultConstructor();
  if (emptyListSelectsDefaultConstructor) return false;

  auto listInitResolution = overloadRes.resolveInitializerListConstructor(
      classSymbol, bracedInitList,
      target.initializationKind.value_or(
          Initializer{target.initializer}.initializationKind()));
  if (!listInitResolution.best) return false;

  auto parameters = listInitResolution.best->symbol->parameters();
  if (parameters.empty()) return false;

  auto ctorParamType = parameters.front()->type();
  auto elemType = ctx.traits.initializer_list_element_type(ctorParamType);
  if (!elemType) return false;

  resolution = std::move(listInitResolution);

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

  return true;
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

  auto convTarget = Initializer{initializer}.conversionTarget();
  ExpressionAST*& target = convTarget ? *convTarget : initExpr;

  auto conversionTargetType = ctx.traits.remove_cv(declaredType);

  elemChecker.check(
      target, conversionTargetType,
      std::format("cannot initialize type '{}' with expression of type '{}'",
                  to_string(conversionTargetType), to_string(target->type)),
      Initializer{initializer}.initializationKind());

  Initializer{initializer}.propagateType();
  return initializer;
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

  if (auto bracedInitList = Initializer{initializer}.bracedInitList()) {
    if (!bracedInitList->expressionList ||
        bracedInitList->expressionList->next) {
      ctx.error(initializer->firstSourceLocation(),
                "reference initializer must be a single expression");
      return initializer;
    }
  }

  auto initExpr = Initializer{initializer}.singleExpression();
  if (!initExpr) {
    ctx.error(initializer->firstSourceLocation(),
              "reference initializer must be a single expression");
    return initializer;
  }

  auto strippedInitializer = Initializer::stripImplicitCasts(initializer);
  ExpressionAST*& conversionTarget =
      ast_cast<EqualInitializerAST>(strippedInitializer) ? initializer
                                                         : initExpr;

  auto seq = ctx.checker.checkImplicitConversion(conversionTarget, targetType);
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

  ctx.checker.applyImplicitConversion(seq, conversionTarget);
  return initializer;
}

struct InitializationEngine {
  InitContext& ctx;
  ElementInitChecker elemChecker;
  DesignatedInitChecker desigChecker;
  AggregateInitChecker aggregateChecker;
  StringInitChecker stringInitChecker;
  ClassInitChecker classChecker;
  ListInitChecker listChecker;
  ScalarInitChecker scalarChecker;
  ReferenceInitChecker refChecker;

  explicit InitializationEngine(InitContext& ctx)
      : ctx(ctx),
        elemChecker(ctx),
        desigChecker{ctx, elemChecker},
        aggregateChecker{ctx, elemChecker, desigChecker},
        stringInitChecker{ctx},
        classChecker{ctx, elemChecker, aggregateChecker},
        listChecker{ctx, elemChecker, desigChecker, aggregateChecker,
                    stringInitChecker},
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
  [[nodiscard]] auto initializesCharacterArrayFromStringLiteral(
      const Type* destinationType, const Initializer& initializer) const
      -> bool;

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

auto InitializationEngine::initializesCharacterArrayFromStringLiteral(
    const Type* destinationType, const Initializer& initializer) const -> bool {
  if (!ctx.traits.is_array(destinationType)) return false;
  auto source = initializer.singleExpression();
  if (!ast_cast<StringLiteralExpressionAST>(source)) return false;
  auto destinationElement =
      ctx.traits.remove_cv(ctx.traits.get_element_type(destinationType));
  if (!ctx.traits.is_char_type(destinationElement)) return false;
  auto sourceElement =
      ctx.traits.remove_cv(ctx.traits.get_element_type(source->type));
  return ctx.traits.is_same(destinationElement, sourceElement) ||
         (ctx.traits.is_narrow_char_type(destinationElement) &&
          ctx.traits.is_narrow_char_type(sourceElement));
}

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

  if (initializesCharacterArrayFromStringLiteral(destinationType,
                                                 initializer)) {
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

    case InitializationBullet::kCharacterArrayFromStringLiteral: {
      auto source = initializer.singleExpression();
      stringInitChecker.checkStringLength(source->firstSourceLocation(),
                                          entity.type(), source->type);
      initializer.propagateType();
      return node;
    }

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

void diagnoseNarrowingListElement(InitContext& ctx, ExpressionAST* element,
                                  const Type* targetType) {
  if (!ctx.isCxx()) return;
  if (!element || !element->type) return;
  if (!ctx.traits.is_narrowing_list_element(element, targetType)) return;

  ctx.error(element->firstSourceLocation(),
            std::format("narrowing conversion from '{}' to '{}' in "
                        "braced-init-list",
                        to_string(element->type), to_string(targetType)));
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
