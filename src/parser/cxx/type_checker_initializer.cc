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
#include <cxx/class_template_deduction.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <cmath>
#include <format>
#include <unordered_set>

namespace cxx {
namespace {
struct InitContext {
  TypeChecker& checker;
  TranslationUnit* unit;
  Control* control;
  TypeTraits traits;

  explicit InitContext(TypeChecker& checker)
      : checker(checker),
        unit(checker.translationUnit()),
        control(checker.translationUnit()->control()),
        traits(checker.translationUnit()->typeTraits()) {}

  [[nodiscard]] auto isCxx() const -> bool {
    return unit->language() == LanguageKind::kCXX;
  }

  void error(SourceLocation loc, std::string message) {
    checker.error(loc, std::move(message));
  }

  void warning(SourceLocation loc, std::string message) {
    checker.warning(loc, std::move(message));
  }

  [[nodiscard]] auto initializesFromSameTypePrvalue(
      ExpressionAST* expr, const Type* targetType) const -> bool {
    if (!expr || !expr->type || !is_prvalue(expr)) return false;
    if (!traits.is_class(targetType)) return false;
    return traits.is_same(traits.remove_cv(expr->type),
                          traits.remove_cv(targetType));
  }

  [[nodiscard]] auto isTargetTypeUnresolved(const Type* type) const -> bool {
    if (!type) return true;
    if (isDependent(unit, type)) return true;
    return containsPlaceholderType(type);
  }
};

struct InitUnwrapper {
  static auto stripImplicitCasts(ExpressionAST* expr) -> ExpressionAST* {
    while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr))
      expr = cast->expression;
    return expr;
  }

  static auto getBracedInitList(ExpressionAST* initializer)
      -> BracedInitListAST* {
    initializer = stripImplicitCasts(initializer);
    if (auto braced = ast_cast<BracedInitListAST>(initializer)) return braced;
    if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
      auto expr = stripImplicitCasts(equal->expression);
      return ast_cast<BracedInitListAST>(expr);
    }
    return nullptr;
  }

  [[nodiscard]] static auto initializationKind(ExpressionAST* initializer)
      -> InitializationKind {
    initializer = stripImplicitCasts(initializer);
    if (ast_cast<ParenInitializerAST>(initializer))
      return InitializationKind::kDirectInitialization;
    if (ast_cast<BracedInitListAST>(initializer))
      return InitializationKind::kDirectListInitialization;
    if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
      if (ast_cast<BracedInitListAST>(stripImplicitCasts(equal->expression)))
        return InitializationKind::kCopyListInitialization;
    }
    return InitializationKind::kCopyInitialization;
  }

  static auto unwrapSingleExpr(ExpressionAST* initializer) -> ExpressionAST* {
    initializer = stripImplicitCasts(initializer);
    if (auto equal = ast_cast<EqualInitializerAST>(initializer))
      initializer = stripImplicitCasts(equal->expression);
    if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
      if (paren->expressionList && !paren->expressionList->next)
        return paren->expressionList->value;
      return nullptr;
    }
    if (ast_cast<BracedInitListAST>(initializer)) return nullptr;
    return initializer;
  }

  static auto collectArgs(ExpressionAST* initializer)
      -> std::vector<ExpressionAST*> {
    std::vector<ExpressionAST*> args;
    if (!initializer) return args;
    initializer = stripImplicitCasts(initializer);
    if (auto equal = ast_cast<EqualInitializerAST>(initializer))
      initializer = stripImplicitCasts(equal->expression);
    if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
      for (auto it = paren->expressionList; it; it = it->next)
        args.push_back(it->value);
    } else if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
      for (auto it = braced->expressionList; it; it = it->next)
        args.push_back(it->value);
    } else {
      args.push_back(initializer);
    }
    return args;
  }

  static void applyConversions(
      TypeChecker& checker, ExpressionAST* initializer,
      const std::vector<ImplicitConversionSequence>& conversions) {
    if (!initializer) return;
    initializer = stripImplicitCasts(initializer);
    if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
      if (auto braced = ast_cast<BracedInitListAST>(
              stripImplicitCasts(equal->expression))) {
        size_t i = 0;
        for (auto it = braced->expressionList; it; it = it->next, ++i)
          checker.applyImplicitConversion(conversions[i], it->value);
        return;
      }
      checker.applyImplicitConversion(conversions[0], equal->expression);
      return;
    }
    if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
      size_t i = 0;
      for (auto it = paren->expressionList; it; it = it->next, ++i)
        checker.applyImplicitConversion(conversions[i], it->value);
    } else if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
      size_t i = 0;
      for (auto it = braced->expressionList; it; it = it->next, ++i)
        checker.applyImplicitConversion(conversions[i], it->value);
    } else {
      checker.applyImplicitConversion(conversions[0], initializer);
    }
  }

  static auto getConversionTarget(ExpressionAST*& initializer)
      -> ExpressionAST** {
    auto stripped = stripImplicitCasts(initializer);
    if (auto equal = ast_cast<EqualInitializerAST>(stripped))
      return &equal->expression;
    if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
      if (paren->expressionList && !paren->expressionList->next)
        return &paren->expressionList->value;
    }
    return nullptr;
  }

  static auto getConstExpressionTarget(ExpressionAST*& initializer)
      -> ExpressionAST** {
    if (auto equal = ast_cast<EqualInitializerAST>(initializer))
      return &equal->expression;
    if (ast_cast<ParenInitializerAST>(initializer)) return nullptr;
    if (ast_cast<BracedInitListAST>(initializer)) return nullptr;
    return &initializer;
  }

  static void propagateInitializerType(ExpressionAST* initializer) {
    auto stripped = stripImplicitCasts(initializer);
    ExpressionAST* wrapped = nullptr;
    if (auto equal = ast_cast<EqualInitializerAST>(stripped))
      wrapped = equal->expression;
    else if (auto paren = ast_cast<ParenInitializerAST>(stripped)) {
      if (paren->expressionList && !paren->expressionList->next)
        wrapped = paren->expressionList->value;
    }
    if (!wrapped) return;
    stripped->type = wrapped->type;
    stripped->valueCategory = wrapped->valueCategory;
  }
};

struct NarrowingChecker {
  InitContext& ctx;

  void checkNarrowing(SourceLocation loc, const Type* sourceType,
                      ExpressionAST* expr, const Type* targetType) {
    if (!ctx.isCxx()) return;
    if (!sourceType) return;
    if (!ctx.traits.is_narrowing_list_element(expr, targetType)) return;
    ctx.error(loc, std::format("narrowing conversion from '{}' to '{}' in "
                               "braced-init-list",
                               to_string(sourceType), to_string(targetType)));
  }
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
  NarrowingChecker narrowing;
  StringInitChecker stringInit;

  explicit ElementInitChecker(InitContext& ctx)
      : ctx(ctx), narrowing{ctx}, stringInit{ctx} {}

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

    auto sourceType = expr->type;
    if (!ctx.checker.implicit_conversion(expr, targetType,
                                         initializationKind)) {
      ctx.error(expr->firstSourceLocation(), std::move(errorMessage));
    } else if (isListInitialization(initializationKind)) {
      narrowing.checkNarrowing(expr->firstSourceLocation(), sourceType, expr,
                               targetType);
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
    auto sourceType = ctx.traits.remove_reference(expr->type);
    auto referredType = ctx.traits.remove_reference(targetType);

    if (!ctx.traits.is_same(ctx.traits.remove_cv(sourceType),
                            ctx.traits.remove_cv(referredType)))
      return false;

    auto sourceCv = ctx.traits.get_cv_qualifiers(sourceType);
    auto targetCv = ctx.traits.get_cv_qualifiers(referredType);
    return sourceCv == targetCv || sourceCv == CvQualifiers::kNone ||
           targetCv == CvQualifiers::kConstVolatile;
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

  auto classType = type_cast<ClassType>(targetType);
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
  } else if (auto braced = InitUnwrapper::getBracedInitList(initializer)) {
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
  auto classType = type_cast<ClassType>(ctx.traits.remove_cv(type));
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

 private:
  static auto firstNonStaticField(ClassSymbol* symbol) -> FieldSymbol* {
    for (auto field : views::members(symbol) | views::non_static_fields)
      return field;
    return nullptr;
  }

  [[nodiscard]] auto elementType(Symbol* element) const -> const Type*;
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

auto AggregateInitChecker::elementType(Symbol* element) const -> const Type* {
  if (auto field = symbol_cast<FieldSymbol>(element))
    return ctx.traits.remove_cv(field->type());

  auto base = symbol_cast<BaseClassSymbol>(element);
  if (!base) return nullptr;

  auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
  if (!baseClass) return nullptr;

  return baseClass->type();
}

auto AggregateInitChecker::elementDescription(Symbol* element) const
    -> std::string {
  if (auto field = symbol_cast<FieldSymbol>(element)) {
    if (field->name())
      return std::format("member '{}'", to_string(field->name()));
    return std::format("anonymous member of type '{}'",
                       to_string(elementType(element)));
  }
  return std::format("base class '{}'", to_string(elementType(element)));
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
    auto classType = type_cast<ClassType>(ctx.traits.remove_cv(field->type()));
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
    auto classType = type_cast<ClassType>(elementType(elements[i]));
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

  auto type = elementType(element);

  if (type && ctx.traits.is_reference(type)) {
    ctx.error(location, std::format("reference {} is not initialized",
                                    elementDescription(element)));
  }

  return makeValueInitializer(type, location);
}

void AggregateInitChecker::checkElementInit(ExpressionAST*& expr,
                                            Symbol* element) {
  auto type = elementType(element);
  if (!type) return;

  auto field = symbol_cast<FieldSymbol>(element);

  if (field && !field->name() && ctx.traits.is_union(type)) {
    checkAnonUnionFieldInit(expr, type);
    return;
  }

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

  elemChecker.check(expr, type,
                    std::format("cannot initialize {} of type '{}' with "
                                "expression of type '{}'",
                                elementDescription(element), to_string(type),
                                to_string(expr->type)));
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
  TypeChecker::AggregateInitGuard guard{ctx.checker, classSymbol};
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
    if (!tryBraceElision(it, elementType(element)))
      checkElementInit(it->value, element);

    positionalInitializer = it->value;
    activeIndex = 0;
  }

  if (!activeIndex) return;

  auto element = variantMembers[*activeIndex];

  auto initializer = positionalInitializer;
  if (!initializer)
    initializer =
        checkDesignatedElementInit(element, elementType(element), clauses);
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
  clause->type = elementType(element);

  ast->expressionList = make_list_node<ExpressionAST>(pool, clause);
}

void AggregateInitChecker::checkStruct(ClassSymbol* classSymbol,
                                       BracedInitListAST* ast) {
  TypeChecker::AggregateInitGuard guard{ctx.checker, classSymbol};
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

    if (!tryBraceElision(it, elementType(element)))
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
      initializer = checkDesignatedElementInit(element, elementType(element),
                                               designated[i]);
    }

    if (!initializer)
      initializer = implicitElementInitializer(element, ast->lbraceLoc);

    *tail = make_list_node<ExpressionAST>(pool, initializer);
    tail = &(*tail)->next;
  }

  ast->expressionList = normalized;
}

struct BracedInitListChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;
  DesignatedInitChecker& desigChecker;
  AggregateInitChecker& aggregateChecker;
  StringInitChecker& stringInit;

  void check(const Type* type, BracedInitListAST* ast,
             InitializationKind initializationKind);

 private:
  void checkArrayInit(const Type* type, BracedInitListAST* ast);
  void checkClassOrUnionInit(const ClassType* classType,
                             BracedInitListAST* ast);
  void checkScalarInit(const Type* type, BracedInitListAST* ast,
                       InitializationKind initializationKind);
  void checkCharArrayStringInit(const Type* elementType,
                                BracedInitListAST* ast);
  void checkArrayElements(const Type* type, const Type* elementType,
                          BracedInitListAST* ast);
  void checkArrayStringElement(ExpressionAST*& expr, const Type* elementType);
};

void BracedInitListChecker::check(const Type* type, BracedInitListAST* ast,
                                  InitializationKind initializationKind) {
  ast->type = type;
  if (type && isDependent(ctx.unit, type)) return;

  if (ctx.traits.is_array(type))
    checkArrayInit(type, ast);
  else if (auto classType = type_cast<ClassType>(ctx.traits.remove_cv(type)))
    checkClassOrUnionInit(classType, ast);
  else
    checkScalarInit(type, ast, initializationKind);
}

void BracedInitListChecker::checkArrayInit(const Type* type,
                                           BracedInitListAST* ast) {
  auto elementType = ctx.traits.remove_cv(ctx.traits.get_element_type(type));

  if (ctx.traits.is_narrow_char_type(elementType) && ast->expressionList &&
      !ast->expressionList->next) {
    if (ast_cast<StringLiteralExpressionAST>(ast->expressionList->value)) {
      checkCharArrayStringInit(type, ast);
      return;
    }
  }

  checkArrayElements(type, elementType, ast);
}

void BracedInitListChecker::checkCharArrayStringInit(const Type* type,
                                                     BracedInitListAST* ast) {
  auto strLit =
      ast_cast<StringLiteralExpressionAST>(ast->expressionList->value);
  stringInit.checkStringLength(strLit->firstSourceLocation(), type,
                               strLit->type);
}

void BracedInitListChecker::checkArrayElements(const Type* type,
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

void BracedInitListChecker::checkArrayStringElement(ExpressionAST*& expr,
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

void BracedInitListChecker::checkClassOrUnionInit(const ClassType* classType,
                                                  BracedInitListAST* ast) {
  if (!classType->symbol()) return;
  if (classType->isUnion())
    aggregateChecker.checkUnion(classType->symbol(), ast);
  else
    aggregateChecker.checkStruct(classType->symbol(), ast);
}

void BracedInitListChecker::checkScalarInit(
    const Type* type, BracedInitListAST* ast,
    InitializationKind initializationKind) {
  auto it = ast->expressionList;
  if (!it) return;

  if (it->next)
    ctx.error(it->next->value->firstSourceLocation(),
              "excess elements in scalar initializer");

  auto& expr = it->value;
  if (ast_cast<DesignatedInitializerClauseAST>(expr)) {
    ctx.error(expr->firstSourceLocation(),
              "designator in initializer for scalar type");
    return;
  }

  elemChecker.check(expr, type,
                    std::format("cannot initialize type '{}' with "
                                "expression of type '{}'",
                                to_string(type), to_string(expr->type)),
                    initializationKind);
}

struct ClassInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;

  struct Target {
    const Type* type = nullptr;
    ExpressionAST* initializer = nullptr;
    SourceLocation location;
    FunctionSymbol* constructor = nullptr;
    List<ExpressionAST*>** argumentList = nullptr;
    bool diagnoseUnresolved = false;
    std::optional<InitializationKind> initializationKind;
  };

  void checkClassInit(Target& target);

 private:
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

  [[nodiscard]] auto makeParenInitializer(Target& target, Arena* arena,
                                          ExpressionAST* firstArgument)
      -> ParenInitializerAST*;

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

  if (ctx.initializesFromSameTypePrvalue(
          InitUnwrapper::unwrapSingleExpr(target.initializer), targetType))
    return;

  if (ctx.traits.is_aggregate(classType)) {
    if (!target.initializer && !target.argumentList) {
      OverloadResolution overloadRes(ctx.unit);
      auto resolution = overloadRes.resolveConstructor(classSymbol, {});
      if (resolution.best && !resolution.ambiguous) {
        target.constructor = resolution.best->symbol;
        ctx.checker.checkConstructorAccess(target.constructor, target.location);
        ctx.checker.useFunction(target.constructor, target.location);
        appendDefaultArguments(target, target.constructor);
      }
      return;
    }
    if (!target.initializer ||
        ast_cast<ParenInitializerAST>(target.initializer)) {
      checkConstructorInit(target, classSymbol, /*diagnoseUnresolved=*/false);
      if (target.constructor) return;
    } else if (auto braced =
                   InitUnwrapper::getBracedInitList(target.initializer)) {
      if (braced->expressionList && !braced->expressionList->next) {
        auto elemType =
            ctx.traits.remove_cv(braced->expressionList->value->type);
        if (ctx.traits.is_same(elemType, classType) ||
            ctx.traits.is_base_of(classType, elemType)) {
          checkConstructorInit(target, classSymbol,
                               /*diagnoseUnresolved=*/false);
          if (target.constructor) return;
        }
      }
    } else if (ast_cast<EqualInitializerAST>(target.initializer)) {
      checkConstructorInit(target, classSymbol, /*diagnoseUnresolved=*/false);
      if (target.constructor) return;
    }
    checkAggregateInit(target, classSymbol);
  } else {
    checkConstructorInit(target, classSymbol, target.diagnoseUnresolved);
  }
}

void ClassInitChecker::checkAggregateInit(Target& target,
                                          ClassSymbol* classSymbol) {
  if (!ctx.unit->config().checkTypes) return;

  auto targetType = ctx.traits.remove_cv(target.type);
  auto bracedInitList = InitUnwrapper::getBracedInitList(target.initializer);

  if (bracedInitList) {
    ctx.checker.check_braced_init_list(
        targetType, bracedInitList,
        InitUnwrapper::initializationKind(target.initializer));
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
  auto resolution = overloadRes.resolveConstructor(
      classSymbol, args,
      target.initializationKind.value_or(
          target.initializer
              ? InitUnwrapper::initializationKind(target.initializer)
              : InitializationKind::kDirectInitialization));

  auto location = target.location;
  if (!location && target.initializer)
    location = target.initializer->firstSourceLocation();
  if (!location) location = classSymbol->location();

  auto bracedInitList = InitUnwrapper::getBracedInitList(target.initializer);
  const bool selectedInitializerListConstructor =
      bracedInitList &&
      tryInitializerListConstructor(target, bracedInitList, classSymbol,
                                    overloadRes, resolution);

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

  const auto initializationKind = target.initializationKind.value_or(
      target.initializer ? InitUnwrapper::initializationKind(target.initializer)
                         : InitializationKind::kDirectInitialization);
  if (initializationKind == InitializationKind::kCopyListInitialization &&
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
  auto parameters = StandardConversion::parameters(constructor);

  std::size_t index = 0;
  for (auto argument : args) {
    if (index >= parameters.size()) break;
    auto parameterType = parameters[index]->type();
    ++index;

    if (!argument) continue;
    if (!ctx.traits.is_narrowing_list_element(argument, parameterType))
      continue;

    ctx.error(argument->firstSourceLocation(),
              std::format("narrowing conversion from '{}' to '{}' in "
                          "braced-init-list",
                          to_string(argument->type), to_string(parameterType)));
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
  if (target.initializer) return InitUnwrapper::collectArgs(target.initializer);

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
    InitUnwrapper::applyConversions(ctx.checker, target.initializer,
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
  auto params = StandardConversion::parameters(constructor);
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

  auto initializer = InitUnwrapper::stripImplicitCasts(target.initializer);

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    if (!equal->expression) return nullptr;
    auto unwrapped = InitUnwrapper::stripImplicitCasts(equal->expression);
    if (auto braced = ast_cast<BracedInitListAST>(unwrapped))
      return &braced->expressionList;
    target.initializer = makeParenInitializer(target, arena, equal->expression);
    return &ast_cast<ParenInitializerAST>(target.initializer)->expressionList;
  }

  if (auto paren = ast_cast<ParenInitializerAST>(initializer))
    return &paren->expressionList;

  if (auto braced = ast_cast<BracedInitListAST>(initializer))
    return &braced->expressionList;

  if (initializer) return nullptr;

  target.initializer = makeParenInitializer(target, arena, nullptr);
  return &ast_cast<ParenInitializerAST>(target.initializer)->expressionList;
}

auto ClassInitChecker::makeParenInitializer(Target& target, Arena* arena,
                                            ExpressionAST* firstArgument)
    -> ParenInitializerAST* {
  auto arguments = firstArgument
                       ? make_list_node<ExpressionAST>(arena, firstArgument)
                       : nullptr;
  return ParenInitializerAST::create(arena, target.location, arguments,
                                     target.location, ValueCategory::kPrValue,
                                     nullptr);
}

auto ClassInitChecker::tryInitializerListConstructor(
    Target& target, BracedInitListAST* bracedInitList, ClassSymbol* classSymbol,
    OverloadResolution& overloadRes, ConstructorResult& resolution) -> bool {
  const auto emptyListSelectsDefaultConstructor =
      !bracedInitList->expressionList && classSymbol->defaultConstructor();
  if (emptyListSelectsDefaultConstructor) return false;

  std::vector<ExpressionAST*> listInitArgs = {bracedInitList};
  auto listInitResolution = overloadRes.resolveConstructor(
      classSymbol, listInitArgs,
      target.initializationKind.value_or(
          InitUnwrapper::initializationKind(target.initializer)));
  if (!listInitResolution.best) return false;

  auto parameters =
      StandardConversion::parameters(listInitResolution.best->symbol);
  if (parameters.empty()) return false;
  for (std::size_t i = 1; i < parameters.size(); ++i) {
    if (!parameters[i]->defaultArgument()) return false;
  }

  auto ctorParamType = parameters.front()->type();
  auto elemType = overloadRes.initializerListElementType(ctorParamType);
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

  target.initializer =
      makeParenInitializer(target, ctx.unit->arena(), bracedInitList);

  return true;
}

struct ScalarInitChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;

  template <typename S>
  void checkScalarInit(S* var, ExpressionAST*& initializer,
                       const Type* declaredType);
};

template <typename S>
void ScalarInitChecker::checkScalarInit(S* var, ExpressionAST*& initializer,
                                        const Type* declaredType) {
  if (!initializer) return;

  auto bracedInitList = InitUnwrapper::getBracedInitList(initializer);
  if (bracedInitList) {
    ctx.checker.check_braced_init_list(
        declaredType, bracedInitList,
        InitUnwrapper::initializationKind(initializer));
    return;
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(initializer);
  if (!initExpr) return;

  auto convTarget = InitUnwrapper::getConversionTarget(initializer);
  ExpressionAST*& target = convTarget ? *convTarget : initExpr;

  auto conversionTargetType = ctx.traits.remove_cv(declaredType);

  elemChecker.check(
      target, conversionTargetType,
      std::format("cannot initialize type '{}' with expression of type '{}'",
                  to_string(conversionTargetType), to_string(target->type)),
      InitUnwrapper::initializationKind(initializer));

  InitUnwrapper::propagateInitializerType(initializer);
  var->setInitializer(initializer);
}

struct ReferenceInitChecker {
  InitContext& ctx;

  template <typename S>
  void check(S* var, ExpressionAST*& initializer, SourceLocation location);
};

template <typename S>
void ReferenceInitChecker::check(S* var, ExpressionAST*& initializer,
                                 SourceLocation location) {
  auto targetType = var->type();

  if (isDependent(ctx.unit, targetType)) return;

  if (!initializer) {
    ctx.error(location,
              std::format("reference variable of type '{}' must be initialized",
                          to_string(targetType)));
    return;
  }

  if (auto bracedInitList = InitUnwrapper::getBracedInitList(initializer)) {
    if (!bracedInitList->expressionList ||
        bracedInitList->expressionList->next) {
      ctx.error(initializer->firstSourceLocation(),
                "reference initializer must be a single expression");
      return;
    }
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(initializer);
  if (!initExpr) {
    ctx.error(initializer->firstSourceLocation(),
              "reference initializer must be a single expression");
    return;
  }

  auto strippedInitializer = InitUnwrapper::stripImplicitCasts(initializer);
  ExpressionAST*& conversionTarget =
      ast_cast<EqualInitializerAST>(strippedInitializer) ? initializer
                                                         : initExpr;

  auto seq = ctx.checker.checkImplicitConversion(conversionTarget, targetType);
  if (seq.rank == ConversionRank::kNone) {
    if (initExpr->type && isDependent(ctx.unit, initExpr->type)) return;

    ctx.error(
        initExpr->firstSourceLocation(),
        std::format("invalid initialization of reference of type '{}' from "
                    "expression of type '{}'",
                    to_string(targetType), to_string(initExpr->type)));
    return;
  }

  ctx.checker.applyImplicitConversion(seq, conversionTarget);
  var->setInitializer(initializer);
}

struct TypeDeducer {
  InitContext& ctx;

  template <typename S>
  void deduceArraySize(S* var);

  template <typename S>
  void deduceAutoType(S* symbol);

  template <typename S>
  void deduceClassTemplateArguments(S* var, SpecifierAST* typeSpecifier);

 private:
  template <typename S>
  void deduceArraySizeFromBraced(S* var, const UnboundedArrayType* ty,
                                 BracedInitListAST* braced);
  template <typename S>
  void deduceArraySizeFromExpr(S* var, const UnboundedArrayType* ty,
                               ExpressionAST* initExpr);
};

template <typename S>
void TypeDeducer::deduceArraySize(S* var) {
  auto ty = type_cast<UnboundedArrayType>(var->type());
  if (!ty) return;

  auto initializer = var->initializer();
  if (!initializer) return;

  auto bracedInitList = InitUnwrapper::getBracedInitList(initializer);
  if (bracedInitList) {
    deduceArraySizeFromBraced(var, ty, bracedInitList);
    return;
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(initializer);
  if (initExpr) deduceArraySizeFromExpr(var, ty, initExpr);
}

template <typename S>
void TypeDeducer::deduceArraySizeFromBraced(S* var,
                                            const UnboundedArrayType* ty,
                                            BracedInitListAST* braced) {
  if (ctx.traits.is_narrow_char_type(ty->elementType()) &&
      braced->expressionList && !braced->expressionList->next) {
    if (auto strLit = ast_cast<StringLiteralExpressionAST>(
            braced->expressionList->value)) {
      if (auto srcArray = type_cast<BoundedArrayType>(strLit->type)) {
        var->setType(ctx.control->getBoundedArrayType(ty->elementType(),
                                                      srcArray->size()));
        return;
      }
    }
  }

  auto interp = ASTInterpreter{ctx.unit};
  size_t currentIndex = 0;
  size_t maxIndex = 0;
  bool hasElements = false;

  for (auto it = braced->expressionList; it; it = it->next) {
    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value)) {
      if (desig->designatorList) {
        if (auto subscript = ast_cast<SubscriptDesignatorAST>(
                desig->designatorList->value)) {
          if (auto value = interp.evaluate(subscript->expression))
            if (auto idx = interp.toUInt(*value)) currentIndex = *idx;
        }
      }
    }
    if (!hasElements || currentIndex > maxIndex) maxIndex = currentIndex;
    hasElements = true;
    ++currentIndex;
  }

  if (hasElements)
    var->setType(
        ctx.control->getBoundedArrayType(ty->elementType(), maxIndex + 1));
}

template <typename S>
void TypeDeducer::deduceArraySizeFromExpr(S* var, const UnboundedArrayType* ty,
                                          ExpressionAST* initExpr) {
  if (auto bounded = type_cast<BoundedArrayType>(initExpr->type))
    var->setType(
        ctx.control->getBoundedArrayType(ty->elementType(), bounded->size()));
}

static auto deduceAutoReplacement(InitContext& ctx, const Type* P,
                                  const Type* A) -> const Type* {
  if (!A) return nullptr;

  if (type_cast<AutoType>(P)) return A;

  if (auto qual = type_cast<QualType>(P)) {
    auto pCv = qual->cvQualifiers();
    auto aCv = ctx.traits.get_cv_qualifiers(A);
    auto remaining =
        CvQualifiers(std::to_underlying(aCv) & ~std::to_underlying(pCv));
    auto strippedA = ctx.traits.add_cv(ctx.traits.remove_cv(A), remaining);
    return deduceAutoReplacement(ctx, qual->elementType(), strippedA);
  }

  if (auto array = type_cast<BoundedArrayType>(P)) {
    auto aArray = type_cast<BoundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto array = type_cast<UnboundedArrayType>(P)) {
    auto aArray = type_cast<UnboundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto array = type_cast<UnresolvedBoundedArrayType>(P)) {
    auto aArray = type_cast<BoundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto ptr = type_cast<PointerType>(P)) {
    auto aPtr = type_cast<PointerType>(A);
    if (!aPtr) return nullptr;
    return deduceAutoReplacement(ctx, ptr->elementType(), aPtr->elementType());
  }

  if (auto ref = type_cast<LvalueReferenceType>(P)) {
    return deduceAutoReplacement(ctx, ref->elementType(),
                                 ctx.traits.remove_reference(A));
  }

  if (auto ref = type_cast<RvalueReferenceType>(P)) {
    if (type_cast<LvalueReferenceType>(A)) {
      return deduceAutoReplacement(ctx, ref->elementType(), A);
    }
    return deduceAutoReplacement(ctx, ref->elementType(),
                                 ctx.traits.remove_reference(A));
  }

  if (auto function = type_cast<FunctionType>(P)) {
    auto aFunction = type_cast<FunctionType>(A);
    if (!aFunction) return nullptr;

    if (containsPlaceholderType(function->returnType())) {
      return deduceAutoReplacement(ctx, function->returnType(),
                                   aFunction->returnType());
    }

    if (function->parameterTypes().size() != aFunction->parameterTypes().size())
      return nullptr;

    for (std::size_t i = 0; i < function->parameterTypes().size(); ++i) {
      auto parameterType = function->parameterTypes()[i];
      if (!containsPlaceholderType(parameterType)) continue;
      return deduceAutoReplacement(ctx, parameterType,
                                   aFunction->parameterTypes()[i]);
    }
    return nullptr;
  }

  if (auto pointer = type_cast<MemberObjectPointerType>(P)) {
    auto aPointer = type_cast<MemberObjectPointerType>(A);
    if (!aPointer) return nullptr;
    return deduceAutoReplacement(ctx, pointer->elementType(),
                                 aPointer->elementType());
  }

  if (auto pointer = type_cast<MemberFunctionPointerType>(P)) {
    auto aPointer = type_cast<MemberFunctionPointerType>(A);
    if (!aPointer) return nullptr;
    return deduceAutoReplacement(ctx, pointer->functionType(),
                                 aPointer->functionType());
  }

  return nullptr;
}

template <typename S>
void TypeDeducer::deduceClassTemplateArguments(S* var,
                                               SpecifierAST* typeSpecifier) {
  if (!ClassTemplateArgumentDeduction::placeholderClassTemplate(
          typeSpecifier, ctx.checker.scope()))
    return;

  auto initializer = var->initializer();

  const auto initializationKind =
      InitUnwrapper::initializationKind(initializer);

  auto deduced = ctx.checker.deduceClassTemplateSpecialization(
      typeSpecifier, InitUnwrapper::collectArgs(initializer),
      InitUnwrapper::getBracedInitList(initializer) != nullptr,
      initializationKind == InitializationKind::kCopyInitialization ||
          initializationKind == InitializationKind::kCopyListInitialization,
      var->location());

  if (!deduced) return;

  const auto cvQualifiers = ctx.traits.get_cv_qualifiers(var->type());
  var->setType(cvQualifiers != CvQualifiers::kNone
                   ? ctx.control->getQualType(deduced, cvQualifiers)
                   : deduced);
}

template <typename S>
void TypeDeducer::deduceAutoType(S* var) {
  auto declType = var->type();
  if (!containsPlaceholderType(declType)) return;

  if (!var->initializer()) {
    ctx.error(var->location(), "variable with 'auto' type must be initialized");
    return;
  }

  auto deducedExpr = InitUnwrapper::unwrapSingleExpr(var->initializer());

  const bool inTemplate = isEnclosedInDependentTemplate(
      ctx.unit, ctx.checker.scope(), /*stopAtConcreteSpecialization=*/true);
  if (inTemplate && (!deducedExpr || !deducedExpr->type ||
                     isDependent(ctx.unit, deducedExpr) ||
                     isDependent(ctx.unit, deducedExpr->type))) {
    auto dependentType = ctx.control->getDependentType();
    var->setType(ctx.traits.replace_placeholder_types(declType, dependentType));
    return;
  }

  if (!deducedExpr || !deducedExpr->type) return;

  auto deduced = ctx.checker.deducePlaceholderType(declType, deducedExpr);
  if (!deduced) return;
  var->setType(deduced);

  if (auto classType =
          type_cast<ClassType>(ctx.traits.remove_cvref(var->type()))) {
    ctx.traits.requireCompleteClass(classType->symbol());
  }
}

struct ConstexprEvaluator {
  InitContext& ctx;

  template <typename S>
  auto tryEvaluateConstructor(S* var, ASTInterpreter& interp)
      -> std::optional<ConstValue>;
};

template <typename S>
auto ConstexprEvaluator::tryEvaluateConstructor(S* var, ASTInterpreter& interp)
    -> std::optional<ConstValue> {
  auto classType = type_cast<ClassType>(ctx.traits.remove_cv(var->type()));
  if (!classType) return std::nullopt;

  auto classSym = classType->symbol();
  if (!classSym) return std::nullopt;

  auto initArgs = InitUnwrapper::collectArgs(var->initializer());

  if (initArgs.size() == 1) {
    if (auto typeConstruction = ast_cast<TypeConstructionAST>(initArgs[0])) {
      if (typeConstruction->type == classType ||
          typeConstruction->type == var->type()) {
        initArgs.clear();
        for (auto it = typeConstruction->expressionList; it; it = it->next)
          initArgs.push_back(it->value);
      }
    }
  }

  std::vector<ConstValue> args;
  for (auto argExpr : initArgs) {
    auto argVal = interp.evaluate(argExpr);
    if (!argVal) return std::nullopt;
    args.push_back(std::move(*argVal));
  }

  auto constructor = var->constructor();
  if (!constructor) constructor = classSym->defaultConstructor();
  if (!constructor) return std::nullopt;
  if (!constructor->isConstexpr() && !constructor->isDefaulted())
    return std::nullopt;
  auto value =
      interp.evaluateConstructor(constructor, classType, std::move(args));
  if (!value || !interp.isFullyInitialized(*value)) return std::nullopt;
  return value;
}

struct InitDeclaratorChecker {
  InitContext ctx;
  ElementInitChecker elemChecker;
  DesignatedInitChecker desigChecker;
  AggregateInitChecker aggregateChecker;
  StringInitChecker stringInitChecker;
  BracedInitListChecker bracedChecker;
  ClassInitChecker classChecker;
  ScalarInitChecker scalarChecker;
  ReferenceInitChecker refChecker;
  TypeDeducer typeDeducer;
  ConstexprEvaluator constexprEval;

  explicit InitDeclaratorChecker(TypeChecker& checker)
      : ctx(checker),
        elemChecker(ctx),
        desigChecker{ctx, elemChecker},
        aggregateChecker{ctx, elemChecker, desigChecker},
        stringInitChecker{ctx},
        bracedChecker{ctx, elemChecker, desigChecker, aggregateChecker,
                      stringInitChecker},
        classChecker{ctx, elemChecker},
        scalarChecker{ctx, elemChecker},
        refChecker{ctx},
        typeDeducer{ctx},
        constexprEval{ctx} {}

  void checkInitDeclarator(InitDeclaratorAST* ast, SpecifierAST* typeSpecifier);
  template <typename S>
  void checkVariable(S* var, ExpressionAST*& initializer,
                     SourceLocation location,
                     SpecifierAST* typeSpecifier = nullptr);
  void checkBracedInitList(const Type* type, BracedInitListAST* ast,
                           InitializationKind initializationKind);
  void checkFieldInitializer(FieldSymbol* field);
  void evaluateFieldConstValue(FieldSymbol* field);
  [[nodiscard]] auto checkClassInitializer(const Type* targetType,
                                           ExpressionAST*& initializer,
                                           SourceLocation location,
                                           List<ExpressionAST*>** argumentList)
      -> FunctionSymbol*;

 private:
  template <typename S>
  void checkInitialization(S* var, ExpressionAST*& initializer,
                           SourceLocation location);
  void evaluateConstValue(VariableSymbol* var, ExpressionAST*& initializer);
};

void InitDeclaratorChecker::checkInitDeclarator(InitDeclaratorAST* ast,
                                                SpecifierAST* typeSpecifier) {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && field->isStatic()) {
    checkVariable(field, ast->initializer, field->location(), typeSpecifier);
    return;
  }

  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  checkVariable(var, ast->initializer,
                ctx.checker.getInitDeclaratorLocation(ast, var), typeSpecifier);
}

template <typename S>
void InitDeclaratorChecker::checkVariable(S* var, ExpressionAST*& initializer,
                                          SourceLocation location,
                                          SpecifierAST* typeSpecifier) {
  if (initializer) var->setInitializer(initializer);

  typeDeducer.deduceArraySize(var);
  typeDeducer.deduceClassTemplateArguments(var, typeSpecifier);
  typeDeducer.deduceAutoType(var);

  if (var->isConstexpr()) var->setType(ctx.traits.add_const(var->type()));

  if (var->isExtern() && !initializer) return;

  auto objectType =
      ctx.traits.remove_cv(ctx.traits.remove_all_extents(var->type()));
  if (auto classType = type_cast<ClassType>(objectType)) {
    auto classSymbol = classType->symbol()->resolvedDefinition();
    auto destructor = classSymbol->destructor();
    ASTRewriter::requireFunctionDefinition(ctx.unit, destructor);
    if (destructor && destructor->isDeleted()) {
      ctx.checker.error(location, "attempt to use a deleted destructor");
    } else if (destructor) {
      AccessContext accessContext{ctx.unit, ctx.checker.scope()};
      if (!accessContext.isAccessible(destructor, classSymbol, nullptr)) {
        auto accessKind = std::string_view{"private"};
        if (destructor->accessSpecifier() == AccessSpecifier::kProtected)
          accessKind = "protected";
        ctx.checker.error(
            location, std::format("calling a {} destructor of class '{}'",
                                  accessKind, to_string(classSymbol->type())));
      }
    }
  }

  checkInitialization(var, initializer, location);

  if constexpr (std::is_same_v<S, VariableSymbol>)
    evaluateConstValue(var, initializer);
}

void InitDeclaratorChecker::checkBracedInitList(
    const Type* type, BracedInitListAST* ast,
    InitializationKind initializationKind) {
  auto targetType = ctx.traits.remove_cv(type);
  if (auto classType = type_cast<ClassType>(targetType);
      classType && !ctx.traits.is_aggregate(classType)) {
    ExpressionAST* initializer = ast;
    ClassInitChecker::Target target{.type = type,
                                    .initializer = initializer,
                                    .location = ast->firstSourceLocation(),
                                    .diagnoseUnresolved = true,
                                    .initializationKind = initializationKind};
    classChecker.checkClassInit(target);
    ast->type = type;
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }
  bracedChecker.check(type, ast, initializationKind);
}

auto InitDeclaratorChecker::checkClassInitializer(
    const Type* targetType, ExpressionAST*& initializer,
    SourceLocation location, List<ExpressionAST*>** argumentList)
    -> FunctionSymbol* {
  ClassInitChecker::Target target{.type = targetType,
                                  .initializer = initializer,
                                  .location = location,
                                  .argumentList = argumentList,
                                  .diagnoseUnresolved = true};
  classChecker.checkClassInit(target);
  initializer = target.initializer;
  return target.constructor;
}

template <typename S>
void InitDeclaratorChecker::checkInitialization(S* var,
                                                ExpressionAST*& initializer,
                                                SourceLocation location) {
  if (ctx.isTargetTypeUnresolved(var->type())) return;

  if (auto initExpr = InitUnwrapper::unwrapSingleExpr(initializer);
      initExpr && initExpr->type && isDependent(ctx.unit, initExpr->type))
    return;

  if (ctx.traits.is_reference(var->type())) {
    refChecker.check(var, initializer, location);
    return;
  }

  auto objectType = ctx.traits.remove_cv(var->type());

  if (ctx.traits.is_class(objectType)) {
    ClassInitChecker::Target target{.type = var->type(),
                                    .initializer = initializer,
                                    .location = var->location(),
                                    .diagnoseUnresolved = true};
    classChecker.checkClassInit(target);
    if (target.constructor) var->setConstructor(target.constructor);
    if (target.initializer != initializer) {
      initializer = target.initializer;
      var->setInitializer(target.initializer);
    }
    return;
  }

  scalarChecker.checkScalarInit(var, initializer, var->type());
}

void InitDeclaratorChecker::checkFieldInitializer(FieldSymbol* field) {
  auto targetType = ctx.traits.remove_cv(field->type());
  if (ctx.isTargetTypeUnresolved(targetType)) return;

  if (ctx.traits.is_class(targetType)) {
    auto classType = type_cast<ClassType>(targetType);
    if (!classType || !classType->symbol()) return;
    if (!classType->symbol()->resolvedDefinition()->isComplete()) return;

    ClassInitChecker::Target target{.type = field->type(),
                                    .initializer = field->initializer(),
                                    .location = field->location(),
                                    .diagnoseUnresolved = true};
    classChecker.checkClassInit(target);
    if (target.constructor) field->setConstructor(target.constructor);
    if (target.initializer != field->initializer())
      field->setInitializer(target.initializer);
    evaluateFieldConstValue(field);
    return;
  }

  auto equal = ast_cast<EqualInitializerAST>(field->initializer());
  auto init = equal ? equal->expression : field->initializer();

  if (auto braced = ast_cast<BracedInitListAST>(init)) {
    if (!braced->type)
      bracedChecker.check(
          field->type(), braced,
          InitUnwrapper::initializationKind(field->initializer()));
    return;
  }

  if (!equal || !equal->expression) return;
  if (ctx.traits.is_reference(field->type())) return;
  if (isDependent(ctx.unit, equal->expression)) return;

  (void)ctx.checker.implicit_conversion(equal->expression, field->type());
}

void InitDeclaratorChecker::evaluateFieldConstValue(FieldSymbol* field) {
  if (!field->isStatic()) return;
  if (!field->isConstexpr() && !field->isConstinit()) return;
  if (isDependent(ctx.unit, field->type())) return;

  auto interp = ASTInterpreter{ctx.unit};

  std::optional<ConstValue> value;
  if (field->initializer()) value = interp.evaluate(field->initializer());

  if (!value.has_value() || field->constructor()) {
    if (auto ctorValue = constexprEval.tryEvaluateConstructor(field, interp))
      value = std::move(ctorValue);
  }

  field->setConstValue(std::move(value));
}

void InitDeclaratorChecker::evaluateConstValue(VariableSymbol* var,
                                               ExpressionAST*& initializer) {
  auto dependent = var->templateParameters() != nullptr;
  if (!dependent) dependent = isDependent(ctx.unit, var->type());
  if (!dependent && var->initializer())
    dependent = isDependent(ctx.unit, var->initializer());

  if (var->initializer()) {
    auto interp = ASTInterpreter{ctx.unit};
    auto value = interp.evaluate(var->initializer());

    const auto needsConstructor = !value.has_value() || var->constructor();

    if (needsConstructor && var->isConstexpr()) {
      if (auto ctorValue = constexprEval.tryEvaluateConstructor(var, interp))
        value = std::move(ctorValue);
    }

    var->setConstValue(value);
  } else if (var->isConstexpr() && var->constructor()) {
    auto interp = ASTInterpreter{ctx.unit};
    var->setConstValue(constexprEval.tryEvaluateConstructor(var, interp));
  }

  if (var->isConstexpr() && var->constValue().has_value() && !dependent) {
    auto target = InitUnwrapper::getConstExpressionTarget(initializer);
    auto needsConstantNode = target != nullptr;
    if (ctx.traits.is_array(var->type())) needsConstantNode = false;
    if (needsConstantNode) needsConstantNode = *target != nullptr;
    if (needsConstantNode)
      needsConstantNode = !ast_cast<ConstExpressionAST>(*target);

    std::optional<ConstValue> constantValue;
    if (needsConstantNode) {
      constantValue = *var->constValue();
      if (is_glvalue(*target)) {
        auto interp = ASTInterpreter{ctx.unit};
        auto address = interp.evaluateAddress(*target);
        if (address.has_value()) {
          constantValue = std::move(address);
          if (ctx.traits.is_reference(var->type())) {
            var->setConstValue(constantValue);
          }
        } else {
          needsConstantNode = false;
          if (ctx.traits.is_reference(var->type())) {
            var->setConstValue(std::nullopt);
          }
        }
      }
    }

    if (needsConstantNode) {
      auto constant = ConstExpressionAST::create(ctx.unit->arena());
      constant->expression = *target;
      constant->constValue =
          ctx.unit->arena()->make<ConstValue>(std::move(*constantValue));
      constant->type = (*target)->type;
      constant->valueCategory = (*target)->valueCategory;
      *target = constant;
      var->setInitializer(initializer);
    }
  }

  if (var->isConstexpr() && !var->constValue().has_value()) {
    if (!dependent) {
      ctx.error(var->location(), "constexpr variable must be initialized");
    }
  }
}
}  // namespace

void TypeChecker::check_init_declarator(InitDeclaratorAST* ast,
                                        SpecifierAST* typeSpecifier) {
  InitDeclaratorChecker{*this}.checkInitDeclarator(ast, typeSpecifier);
}

void TypeChecker::check_condition_declaration(ConditionExpressionAST* ast) {
  auto var = ast->symbol;
  if (!var) return;
  InitDeclaratorChecker{*this}.checkVariable(var, ast->initializer,
                                             var->location());

  ast->type = var->type();
  ast->valueCategory = ValueCategory::kLValue;
}

void TypeChecker::check_field_initializer(FieldSymbol* field) {
  if (!field || !field->initializer()) return;
  InitDeclaratorChecker{*this}.checkFieldInitializer(field);
}

auto TypeChecker::deducePlaceholderType(const Type* declaredType,
                                        ExpressionAST* initializer)
    -> const Type* {
  if (!initializer) return nullptr;
  if (type_cast<DecltypeAutoType>(declaredType))
    return unit_->typeTraits().decltype_of(initializer);
  auto initializerType = initializer->type;
  if (type_cast<RvalueReferenceType>(declaredType) &&
      initializer->valueCategory == ValueCategory::kLValue) {
    initializerType = unit_->typeTraits().add_lvalue_reference(initializerType);
  }
  return deduceAutoType(declaredType, initializerType);
}

auto TypeChecker::deduceAutoType(const Type* declaredType,
                                 const Type* initializerType) -> const Type* {
  if (!initializerType) return nullptr;

  InitContext ctx{*this};

  if (type_cast<AutoType>(declaredType))
    return ctx.traits.decay(initializerType);

  auto replacement = deduceAutoReplacement(ctx, declaredType, initializerType);
  if (!replacement) return nullptr;
  return ctx.traits.replace_placeholder_types(declaredType, replacement);
}

void TypeChecker::check_braced_init_list(
    const Type* type, BracedInitListAST* ast,
    InitializationKind initializationKind) {
  InitDeclaratorChecker{*this}.checkBracedInitList(type, ast,
                                                   initializationKind);
}

auto TypeChecker::check_class_initializer(const Type* targetType,
                                          ExpressionAST*& initializer,
                                          SourceLocation location,
                                          List<ExpressionAST*>** argumentList)
    -> FunctionSymbol* {
  return InitDeclaratorChecker{*this}.checkClassInitializer(
      targetType, initializer, location, argumentList);
}

void TypeChecker::checkConstructorAccess(FunctionSymbol* constructor,
                                         SourceLocation location) {
  if (!constructor) return;

  auto declaringClass = declaringClassOf(constructor);
  if (!declaringClass) return;

  AccessContext accessContext{unit_, scope_};
  if (accessContext.isAccessible(constructor, declaringClass, nullptr)) return;

  auto accessKind = std::string_view{"private"};
  if (constructor->accessSpecifier() == AccessSpecifier::kProtected)
    accessKind = "protected";

  error(location, std::format("calling a {} constructor of class '{}'",
                              accessKind, to_string(declaringClass->type())));
}

auto TypeChecker::deduceClassTemplateSpecialization(
    SpecifierAST* typeSpecifier, const std::vector<ExpressionAST*>& arguments,
    bool isListInitialization, bool isCopyInitialization,
    SourceLocation location) -> const Type* {
  auto primaryTemplate =
      ClassTemplateArgumentDeduction::placeholderClassTemplate(typeSpecifier,
                                                               scope_);
  if (!primaryTemplate) return nullptr;

  for (auto argument : arguments) {
    if (!argument || !argument->type) return nullptr;
    if (isDependent(unit_, argument->type)) return nullptr;
  }

  ClassTemplateArgumentDeduction::Initializer initializer{
      .arguments = arguments,
      .isListInitialization = isListInitialization,
      .isCopyInitialization = isCopyInitialization};

  ClassTemplateArgumentDeduction deduction{unit_};
  auto deduced =
      deduction.deduce(primaryTemplate, initializer, location, scope_);

  if (!deduced) {
    if (deduction.selectedExplicitOnly()) {
      error(
          location,
          std::format("class template argument deduction for '{}' selected an "
                      "explicit deduction guide for copy-list-initialization",
                      to_string(primaryTemplate->name())));
    } else {
      error(location,
            std::format("no viable constructor or deduction guide for "
                        "deduction of template arguments of '{}'",
                        to_string(primaryTemplate->name())));
    }
    return nullptr;
  }

  return deduced->type();
}
}  // namespace cxx
