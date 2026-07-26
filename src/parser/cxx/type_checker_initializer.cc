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
    if (ast_cast<EqualInitializerAST>(stripped)) return &initializer;
    if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
      if (paren->expressionList && !paren->expressionList->next)
        return &paren->expressionList->value;
    }
    return nullptr;
  }
};

struct NarrowingChecker {
  InitContext& ctx;

  void warnIfNarrowing(SourceLocation loc, const Type* sourceType,
                       ExpressionAST* expr, const Type* targetType) {
    if (!ctx.isCxx()) return;
    if (!sourceType) return;
    if (!ctx.traits.is_narrowing_conversion(sourceType, targetType)) return;
    if (isConstantExprNonNarrowing(expr, targetType)) return;
    ctx.warning(loc, std::format("narrowing conversion from '{}' to '{}' in "
                                 "braced-init-list",
                                 to_string(sourceType), to_string(targetType)));
  }

  [[nodiscard]] auto isConstantExprNonNarrowing(ExpressionAST* expr,
                                                const Type* targetType)
      -> bool {
    if (!expr || !targetType) return false;
    targetType = ctx.traits.remove_cv(targetType);
    expr = InitUnwrapper::stripImplicitCasts(expr);

    if (auto nested = ast_cast<NestedExpressionAST>(expr))
      return isConstantExprNonNarrowing(nested->expression, targetType);

    if (auto equal = ast_cast<EqualInitializerAST>(expr))
      return isConstantExprNonNarrowing(equal->expression, targetType);

    if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
      if (!paren->expressionList || paren->expressionList->next) return false;
      return isConstantExprNonNarrowing(paren->expressionList->value,
                                        targetType);
    }

    if (auto intLit = ast_cast<IntLiteralExpressionAST>(expr))
      return checkIntLitNonNarrowing(intLit, targetType);

    if (auto floatLit = ast_cast<FloatLiteralExpressionAST>(expr))
      return checkFloatLitNonNarrowing(floatLit, targetType);

    auto interp = ASTInterpreter{ctx.unit};
    auto value = interp.evaluate(expr);
    if (!value) return false;

    if (auto intVal = std::get_if<std::intmax_t>(&*value))
      return checkIntValueNonNarrowing(*intVal, targetType);
    if (auto floatVal = std::get_if<float>(&*value))
      return checkFloatValueNonNarrowing(*floatVal, targetType);
    if (auto doubleVal = std::get_if<double>(&*value))
      return checkFloatValueNonNarrowing(*doubleVal, targetType);
    if (auto ldVal = std::get_if<long double>(&*value))
      return checkFloatValueNonNarrowing(static_cast<double>(*ldVal),
                                         targetType);

    return false;
  }

  [[nodiscard]] auto checkIntLitNonNarrowing(IntLiteralExpressionAST* intLit,
                                             const Type* targetType) -> bool {
    if (!intLit->literal) return false;
    return checkIntValueNonNarrowing(
        static_cast<std::intmax_t>(intLit->literal->integerValue()),
        targetType);
  }

  [[nodiscard]] auto checkIntValueNonNarrowing(std::intmax_t value,
                                               const Type* targetType) -> bool {
    if (ctx.traits.is_integral(targetType)) {
      if (value >= 0) {
        return ctx.traits.integer_constant_fits_in_type(
            static_cast<std::uint64_t>(value), targetType);
      }
      if (!ctx.traits.is_signed(targetType)) return false;
      auto targetSize = ctx.control->memoryLayout()->sizeOf(targetType);
      if (!targetSize) return false;
      auto minVal = -(std::intmax_t{1} << (*targetSize * 8 - 1));
      return value >= minVal;
    }

    auto valueLD = static_cast<long double>(value);

    if (type_cast<FloatType>(targetType)) {
      auto conv = static_cast<float>(value);
      return std::isfinite(conv) && static_cast<long double>(conv) == valueLD;
    }
    if (type_cast<DoubleType>(targetType)) {
      auto conv = static_cast<double>(value);
      return std::isfinite(conv) && static_cast<long double>(conv) == valueLD;
    }
    if (type_cast<LongDoubleType>(targetType)) {
      auto conv = static_cast<long double>(value);
      return std::isfinite(static_cast<double>(conv)) && conv == valueLD;
    }
    return false;
  }

  [[nodiscard]] auto checkFloatLitNonNarrowing(
      FloatLiteralExpressionAST* floatLit, const Type* targetType) -> bool {
    if (!floatLit->literal) return false;
    return checkFloatValueNonNarrowing(floatLit->literal->floatValue(),
                                       targetType);
  }

  [[nodiscard]] auto checkFloatValueNonNarrowing(double value,
                                                 const Type* targetType)
      -> bool {
    if (!ctx.traits.is_floating_point(targetType)) return false;
    if (!std::isfinite(value)) return false;

    if (type_cast<FloatType>(targetType)) {
      auto conv = static_cast<float>(value);
      return std::isfinite(conv) && static_cast<double>(conv) == value;
    }
    if (type_cast<DoubleType>(targetType)) return true;
    if (type_cast<LongDoubleType>(targetType)) {
      auto conv = static_cast<long double>(value);
      return std::isfinite(static_cast<double>(conv)) &&
             static_cast<double>(conv) == value;
    }
    return false;
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
    auto maxChars = ctx.isCxx() ? destArray->size() - 1 : destArray->size();
    if (srcArray->size() > maxChars)
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
             std::string errorMessage) {
    if (ctx.traits.is_array(targetType)) {
      checkArrayElementInit(expr, targetType, std::move(errorMessage));
      return;
    }

    if (ctx.traits.is_lvalue_reference(targetType))
      stripLvalueConversions(expr);

    if (ctx.traits.is_lvalue_reference(targetType) && is_lvalue(expr)) {
      if (checkDirectLvalueBinding(expr, targetType)) return;
    }

    auto sourceType = expr->type;
    if (!ctx.checker.implicit_conversion(expr, targetType)) {
      ctx.error(expr->firstSourceLocation(), std::move(errorMessage));
    } else {
      narrowing.warnIfNarrowing(expr->firstSourceLocation(), sourceType, expr,
                                targetType);
    }
  }

  [[nodiscard]] auto checkClassElementInit(ExpressionAST*& expr,
                                           const Type* targetType) -> bool;

 private:
  void checkArrayElementInit(ExpressionAST*& expr, const Type* targetType,
                             std::string errorMessage) {
    if (stringInit.isStringToCharArrayInit(expr, targetType)) {
      stringInit.checkStringLength(expr->firstSourceLocation(), targetType,
                                   expr->type);
      return;
    }

    auto elemType =
        ctx.traits.remove_cv(ctx.traits.get_element_type(targetType));
    check(expr, elemType, std::move(errorMessage));
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
  if (!expr || !expr->type) return false;
  if (!ctx.traits.is_class(targetType)) return false;
  if (isDependent(ctx.unit, targetType)) return false;
  if (isDependent(ctx.unit, expr->type)) return false;

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

 private:
  auto resolveDesignators(const Type* type,
                          List<DesignatorAST*>* designatorList) -> const Type*;
  auto resolveDotDesignator(const Type* type, DotDesignatorAST* dot)
      -> const Type*;
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
    ctx.error(
        dot->firstSourceLocation(),
        std::format("field designator '{}' does not refer to a "
                    "non-static data member",
                    dot->identifier ? dot->identifier->name() : "<anonymous>"));
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

  if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
    if (auto nested = ast_cast<BracedInitListAST>(equal->expression)) {
      ctx.checker.check_braced_init_list(targetType, nested);
    } else if (equal->expression) {
      elemChecker.check(
          equal->expression, targetType,
          std::format("cannot initialize type '{}' with expression of "
                      "type '{}'",
                      to_string(targetType),
                      to_string(equal->expression->type)));
    }
  } else if (auto braced = ast_cast<BracedInitListAST>(ast->initializer)) {
    ctx.checker.check_braced_init_list(targetType, braced);
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

 private:
  static auto firstNonStaticField(ClassSymbol* symbol) -> FieldSymbol* {
    for (auto field : views::members(symbol) | views::non_static_fields)
      return field;
    return nullptr;
  }

  void collectEffectiveFields(ClassSymbol* classSymbol,
                              std::vector<FieldSymbol*>& fields);

  void checkFieldInit(ExpressionAST*& expr, FieldSymbol* field);
  void checkAnonUnionFieldInit(ExpressionAST*& expr, const Type* fieldType);

  [[nodiscard]] auto isSubAggregate(const Type* type) const -> bool;

  [[nodiscard]] auto countScalarInitSlots(const Type* type) const -> size_t;

  [[nodiscard]] auto buildSyntheticBracedList(List<ExpressionAST*>*& it,
                                              size_t maxCount)
      -> BracedInitListAST*;
};

void AggregateInitChecker::collectEffectiveFields(
    ClassSymbol* classSymbol, std::vector<FieldSymbol*>& fields) {
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name()) {
      auto classType =
          type_cast<ClassType>(ctx.traits.remove_cv(field->type()));
      if (classType && classType->symbol()) {
        if (classType->symbol()->isUnion())
          fields.push_back(field);
        else
          collectEffectiveFields(classType->symbol(), fields);
      }
    } else {
      fields.push_back(field);
    }
  }
}

void AggregateInitChecker::checkFieldInit(ExpressionAST*& expr,
                                          FieldSymbol* field) {
  auto fieldType = ctx.traits.remove_cv(field->type());
  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    ctx.checker.check_braced_init_list(fieldType, nested);
    return;
  }

  if (elemChecker.checkClassElementInit(expr, fieldType)) return;

  elemChecker.check(
      expr, fieldType,
      std::format("cannot initialize member '{}' of type '{}' with "
                  "expression of type '{}'",
                  to_string(field->name()), to_string(fieldType),
                  to_string(expr->type)));
}

void AggregateInitChecker::checkAnonUnionFieldInit(ExpressionAST*& expr,
                                                   const Type* fieldType) {
  auto classType = type_cast<ClassType>(fieldType);
  if (!classType || !classType->symbol() || !classType->symbol()->isUnion()) {
    return;
  }

  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    ctx.checker.check_braced_init_list(fieldType, nested);
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
  type = ctx.traits.remove_cv(type);

  if (auto bt = type_cast<BoundedArrayType>(type))
    return bt->size() * countScalarInitSlots(bt->elementType());

  if (auto ct = type_cast<ClassType>(type)) {
    auto cls = ct->symbol();
    if (!cls) return 1;
    if (cls->isUnion()) {
      for (auto m : cls->members())
        if (auto f = symbol_cast<FieldSymbol>(m))
          if (!f->isStatic()) return countScalarInitSlots(f->type());
      return 1;
    }
    size_t total = 0;
    for (auto m : cls->members())
      if (auto f = symbol_cast<FieldSymbol>(m))
        if (!f->isStatic()) total += countScalarInitSlots(f->type());
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
  ctx.checker.check_braced_init_list(targetType, synthetic);

  auto afterRun = it->next;
  firstNode->value = synthetic;
  firstNode->next = afterRun;
  it = firstNode;
  return true;
}

void AggregateInitChecker::checkUnion(ClassSymbol* classSymbol,
                                      BracedInitListAST* ast) {
  auto it = ast->expressionList;
  if (!it) return;

  auto& expr = it->value;

  if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
    desigChecker.check(ctx.control->getClassType(classSymbol), desig);
    if (it->next)
      ctx.error(it->next->value->firstSourceLocation(),
                "excess elements in union initializer");
    return;
  }

  auto field = firstNonStaticField(classSymbol);
  if (!field) {
    ctx.error(expr->firstSourceLocation(), "union has no named members");
    return;
  }

  auto fieldType = ctx.traits.remove_cv(field->type());
  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    ctx.checker.check_braced_init_list(fieldType, nested);
  } else if (!tryBraceElision(it, fieldType)) {
    elemChecker.check(
        expr, fieldType,
        std::format("cannot initialize member '{}' of type '{}' with "
                    "expression of type '{}'",
                    to_string(field->name()), to_string(fieldType),
                    to_string(expr->type)));
  }

  if (it->next)
    ctx.error(it->next->value->firstSourceLocation(),
              "excess elements in union initializer");
}

void AggregateInitChecker::checkStruct(ClassSymbol* classSymbol,
                                       BracedInitListAST* ast) {
  std::vector<FieldSymbol*> fields;
  collectEffectiveFields(classSymbol, fields);

  size_t fieldIndex = 0;

  for (auto it = ast->expressionList; it; it = it->next) {
    auto& expr = it->value;

    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
      desigChecker.check(ctx.control->getClassType(classSymbol), desig);
      if (desig->designatorList) {
        if (auto dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
            dot && dot->symbol) {
          for (size_t i = 0; i < fields.size(); ++i) {
            if (fields[i] == dot->symbol) {
              fieldIndex = i + 1;
              break;
            }
          }
        }
      }
      continue;
    }

    if (fieldIndex >= fields.size()) {
      ctx.error(expr->firstSourceLocation(),
                "excess elements in struct initializer");
      break;
    }

    auto fieldType = ctx.traits.remove_cv(fields[fieldIndex]->type());

    if (!fields[fieldIndex]->name()) {
      checkAnonUnionFieldInit(expr, fieldType);
    } else if (!tryBraceElision(it, fieldType)) {
      checkFieldInit(expr, fields[fieldIndex]);
    }

    ++fieldIndex;
  }
}

struct BracedInitListChecker {
  InitContext& ctx;
  ElementInitChecker& elemChecker;
  DesignatedInitChecker& desigChecker;
  AggregateInitChecker& aggregateChecker;
  StringInitChecker& stringInit;

  void check(const Type* type, BracedInitListAST* ast);

 private:
  void checkArrayInit(const Type* type, BracedInitListAST* ast);
  void checkClassOrUnionInit(const ClassType* classType,
                             BracedInitListAST* ast);
  void checkScalarInit(const Type* type, BracedInitListAST* ast);
  void checkCharArrayStringInit(const Type* elementType,
                                BracedInitListAST* ast);
  void checkArrayElements(const Type* type, const Type* elementType,
                          BracedInitListAST* ast);
  void checkArrayStringElement(ExpressionAST*& expr, const Type* elementType);
};

void BracedInitListChecker::check(const Type* type, BracedInitListAST* ast) {
  ast->type = type;
  if (type && isDependent(ctx.unit, type)) return;

  if (ctx.traits.is_array(type))
    checkArrayInit(type, ast);
  else if (auto classType = type_cast<ClassType>(ctx.traits.remove_cv(type)))
    checkClassOrUnionInit(classType, ast);
  else
    checkScalarInit(type, ast);
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
      ctx.checker.check_braced_init_list(elementType, nested);
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
    ++index;
  }
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

void BracedInitListChecker::checkScalarInit(const Type* type,
                                            BracedInitListAST* ast) {
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
                                to_string(type), to_string(expr->type)));
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
  };

  void checkClassInit(Target& target);

 private:
  void checkAggregateInit(Target& target, ClassSymbol* classSymbol);
  void checkConstructorInit(Target& target, ClassSymbol* classSymbol,
                            bool diagnoseUnresolved);

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
  auto classSymbol = classType->symbol();

  if (ctx.traits.is_aggregate(classType)) {
    if (!target.initializer && !target.argumentList) {
      OverloadResolution overloadRes(ctx.unit);
      auto resolution = overloadRes.resolveConstructor(classSymbol, {});
      if (resolution.best && !resolution.ambiguous) {
        target.constructor = resolution.best->symbol;
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
    ctx.checker.check_braced_init_list(targetType, bracedInitList);
    return;
  }

  if (auto equal = ast_cast<EqualInitializerAST>(target.initializer);
      equal && equal->expression) {
    elemChecker.check(
        equal->expression, targetType,
        std::format("cannot initialize type '{}' with expression of type '{}'",
                    to_string(targetType), to_string(equal->expression->type)));
  }
}

void ClassInitChecker::checkConstructorInit(Target& target,
                                            ClassSymbol* classSymbol,
                                            bool diagnoseUnresolved) {
  if (!ctx.unit->config().checkTypes) return;

  auto args = arguments(target);

  const auto inTemplate = isEnclosedInTemplate(ctx.checker.scope());

  if (inTemplate) {
    if (isEnclosedInTemplate(classSymbol)) return;
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
  auto resolution = overloadRes.resolveConstructor(classSymbol, args);

  auto bracedInitList = InitUnwrapper::getBracedInitList(target.initializer);
  if (bracedInitList &&
      tryInitializerListConstructor(target, bracedInitList, classSymbol,
                                    overloadRes, resolution))
    return;

  if (!resolution.best) {
    if (diagnoseUnresolved)
      ctx.error(target.location, "no matching constructor");
    return;
  }

  if (resolution.ambiguous) {
    ctx.error(target.location, "constructor call is ambiguous");
    return;
  }

  target.constructor = resolution.best->symbol;
  ctx.checker.reportDeletedFunction(target.constructor, target.location);
  applyArgumentConversions(target, resolution.best->conversions);
  appendDefaultArguments(target, target.constructor);
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
  std::vector<ExpressionAST*> listInitArgs = {bracedInitList};
  auto listInitResolution =
      overloadRes.resolveConstructor(classSymbol, listInitArgs);
  if (!listInitResolution.best) return false;

  auto ctorType =
      type_cast<FunctionType>(listInitResolution.best->symbol->type());
  if (!ctorType || ctorType->parameterTypes().size() != 1) return false;

  auto ctorParamType = ctorType->parameterTypes().front();
  auto elemType = overloadRes.initializerListElementType(ctorParamType);
  if (!elemType) return false;

  resolution = std::move(listInitResolution);

  bracedInitList->type = ctorParamType;
  bracedInitList->valueCategory = ValueCategory::kPrValue;
  for (auto it = bracedInitList->expressionList; it; it = it->next) {
    elemChecker.check(
        it->value, elemType,
        std::format("cannot initialize initializer_list element "
                    "of type '{}' with expression of type '{}'",
                    to_string(elemType), to_string(it->value->type)));
  }

  if (!resolution.ambiguous)
    target.constructor = resolution.best->symbol;
  else
    ctx.error(target.location, "constructor call is ambiguous");

  return true;
}

struct ScalarInitChecker {
  InitContext& ctx;

  void checkScalarInit(VariableSymbol* var, ExpressionAST*& initializer,
                       const Type* targetType);
};

void ScalarInitChecker::checkScalarInit(VariableSymbol* var,
                                        ExpressionAST*& initializer,
                                        const Type* targetType) {
  if (!initializer) return;

  auto bracedInitList = InitUnwrapper::getBracedInitList(initializer);
  if (bracedInitList) {
    ctx.checker.check_braced_init_list(targetType, bracedInitList);
    return;
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(initializer);
  if (!initExpr) return;

  auto convTarget = InitUnwrapper::getConversionTarget(initializer);
  ExpressionAST*& target = convTarget ? *convTarget : initExpr;

  if (!ctx.checker.implicit_conversion(target, targetType)) {
    auto seq = ctx.checker.checkImplicitConversion(target, targetType);
    ctx.checker.applyImplicitConversion(seq, target);
  }

  var->setInitializer(initializer);
}

struct ReferenceInitChecker {
  InitContext& ctx;

  void check(VariableSymbol* var, ExpressionAST*& initializer,
             SourceLocation location);
};

void ReferenceInitChecker::check(VariableSymbol* var,
                                 ExpressionAST*& initializer,
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

  void deduceArraySize(VariableSymbol* var);
  void deduceAutoType(VariableSymbol* var);

 private:
  void deduceArraySizeFromBraced(VariableSymbol* var,
                                 const UnboundedArrayType* ty,
                                 BracedInitListAST* braced);
  void deduceArraySizeFromExpr(VariableSymbol* var,
                               const UnboundedArrayType* ty,
                               ExpressionAST* initExpr);
};

void TypeDeducer::deduceArraySize(VariableSymbol* var) {
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

void TypeDeducer::deduceArraySizeFromBraced(VariableSymbol* var,
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

void TypeDeducer::deduceArraySizeFromExpr(VariableSymbol* var,
                                          const UnboundedArrayType* ty,
                                          ExpressionAST* initExpr) {
  if (auto bounded = type_cast<BoundedArrayType>(initExpr->type))
    var->setType(
        ctx.control->getBoundedArrayType(ty->elementType(), bounded->size()));
}

static auto containsAutoPlaceholder(const Type* type) -> bool {
  if (!type) return false;
  if (type_cast<AutoType>(type)) return true;
  if (auto qual = type_cast<QualType>(type))
    return containsAutoPlaceholder(qual->elementType());
  if (auto ptr = type_cast<PointerType>(type))
    return containsAutoPlaceholder(ptr->elementType());
  if (auto ref = type_cast<LvalueReferenceType>(type))
    return containsAutoPlaceholder(ref->elementType());
  if (auto ref = type_cast<RvalueReferenceType>(type))
    return containsAutoPlaceholder(ref->elementType());
  return false;
}

static auto deduceCompoundAuto(InitContext& ctx, const Type* P, const Type* A)
    -> const Type* {
  if (!A) return nullptr;

  if (type_cast<AutoType>(P)) return A;

  if (auto qual = type_cast<QualType>(P)) {
    auto pCv = qual->cvQualifiers();
    auto aCv = ctx.traits.get_cv_qualifiers(A);
    auto remaining =
        CvQualifiers(std::to_underlying(aCv) & ~std::to_underlying(pCv));
    auto strippedA = ctx.traits.add_cv(ctx.traits.remove_cv(A), remaining);
    auto inner = deduceCompoundAuto(ctx, qual->elementType(), strippedA);
    if (!inner) return nullptr;
    return ctx.traits.add_cv(inner, pCv);
  }

  if (auto ptr = type_cast<PointerType>(P)) {
    auto aPtr = type_cast<PointerType>(A);
    if (!aPtr) return nullptr;
    auto inner =
        deduceCompoundAuto(ctx, ptr->elementType(), aPtr->elementType());
    if (!inner) return nullptr;
    return ctx.control->getPointerType(inner);
  }

  if (auto ref = type_cast<LvalueReferenceType>(P)) {
    auto inner = deduceCompoundAuto(ctx, ref->elementType(),
                                    ctx.traits.remove_reference(A));
    if (!inner) return nullptr;
    return ctx.control->getLvalueReferenceType(inner);
  }

  if (auto ref = type_cast<RvalueReferenceType>(P)) {
    auto inner = deduceCompoundAuto(ctx, ref->elementType(),
                                    ctx.traits.remove_reference(A));
    if (!inner) return nullptr;
    return ctx.control->getRvalueReferenceType(inner);
  }

  return nullptr;
}

void TypeDeducer::deduceAutoType(VariableSymbol* var) {
  auto declType = var->type();
  if (!containsAutoPlaceholder(declType)) return;

  if (!var->initializer()) {
    ctx.error(var->location(), "variable with 'auto' type must be initialized");
    return;
  }

  auto deducedExpr = InitUnwrapper::unwrapSingleExpr(var->initializer());

  if ((!deducedExpr || !deducedExpr->type) &&
      isEnclosedInTemplate(ctx.checker.scope())) {
    var->setType(ctx.control->getTypeParameterType(0, 0, false));
    return;
  }

  if (!deducedExpr || !deducedExpr->type) return;

  if (type_cast<AutoType>(declType)) {
    var->setType(ctx.traits.remove_cvref(deducedExpr->type));
  } else {
    auto deduced = deduceCompoundAuto(ctx, declType, deducedExpr->type);
    if (!deduced) return;
    var->setType(deduced);
  }

  if (auto classType =
          type_cast<ClassType>(ctx.traits.remove_cvref(var->type()))) {
    ctx.traits.requireCompleteClass(classType->symbol());
  }
}

struct ConstexprEvaluator {
  InitContext& ctx;

  auto tryEvaluateConstructor(VariableSymbol* var, ASTInterpreter& interp)
      -> std::optional<ConstValue>;
};

auto ConstexprEvaluator::tryEvaluateConstructor(VariableSymbol* var,
                                                ASTInterpreter& interp)
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

  for (auto ctor : classSym->constructors()) {
    if (!ctor->isConstexpr()) {
      if (ctor->isDefaulted() && args.empty()) {
        auto obj = std::make_shared<ConstObject>(classType);
        return ConstValue{std::move(obj)};
      }
      continue;
    }
    return interp.evaluateConstructor(ctor, classType, std::move(args));
  }

  return std::nullopt;
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
        scalarChecker{ctx},
        refChecker{ctx},
        typeDeducer{ctx},
        constexprEval{ctx} {}

  void checkInitDeclarator(InitDeclaratorAST* ast);
  void checkVariable(VariableSymbol* var, ExpressionAST*& initializer,
                     SourceLocation location);
  void checkBracedInitList(const Type* type, BracedInitListAST* ast);
  void checkFieldInitializer(FieldSymbol* field);
  [[nodiscard]] auto checkClassInitializer(const Type* targetType,
                                           ExpressionAST*& initializer,
                                           SourceLocation location,
                                           List<ExpressionAST*>** argumentList)
      -> FunctionSymbol*;

 private:
  void checkInitialization(VariableSymbol* var, ExpressionAST*& initializer,
                           SourceLocation location);
  void evaluateConstValue(VariableSymbol* var);
};

void InitDeclaratorChecker::checkInitDeclarator(InitDeclaratorAST* ast) {
  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  checkVariable(var, ast->initializer,
                ctx.checker.getInitDeclaratorLocation(ast, var));
}

void InitDeclaratorChecker::checkVariable(VariableSymbol* var,
                                          ExpressionAST*& initializer,
                                          SourceLocation location) {
  var->setInitializer(initializer);

  typeDeducer.deduceArraySize(var);
  typeDeducer.deduceAutoType(var);

  if (var->isConstexpr()) var->setType(ctx.traits.add_const(var->type()));

  checkInitialization(var, initializer, location);
  evaluateConstValue(var);
}

void InitDeclaratorChecker::checkBracedInitList(const Type* type,
                                                BracedInitListAST* ast) {
  bracedChecker.check(type, ast);
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

void InitDeclaratorChecker::checkInitialization(VariableSymbol* var,
                                                ExpressionAST*& initializer,
                                                SourceLocation location) {
  if (ctx.traits.is_reference(var->type())) {
    refChecker.check(var, initializer, location);
    return;
  }

  auto targetType = ctx.traits.remove_cv(var->type());

  if (ctx.traits.is_class(targetType)) {
    ClassInitChecker::Target target{
        .type = var->type(),
        .initializer = initializer,
        .location = var->location(),
        .diagnoseUnresolved = initializer || !var->isExtern()};
    classChecker.checkClassInit(target);
    if (target.constructor) var->setConstructor(target.constructor);
    if (target.initializer != initializer) {
      initializer = target.initializer;
      var->setInitializer(target.initializer);
    }
    return;
  }

  scalarChecker.checkScalarInit(var, initializer, targetType);
}

void InitDeclaratorChecker::checkFieldInitializer(FieldSymbol* field) {
  auto targetType = ctx.traits.remove_cv(field->type());
  if (isDependent(ctx.unit, targetType)) return;

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
    return;
  }

  auto equal = ast_cast<EqualInitializerAST>(field->initializer());
  auto init = equal ? equal->expression : field->initializer();

  if (auto braced = ast_cast<BracedInitListAST>(init)) {
    if (!braced->type) bracedChecker.check(field->type(), braced);
    return;
  }

  if (!equal || !equal->expression) return;
  if (ctx.traits.is_reference(field->type())) return;
  if (isDependent(ctx.unit, equal->expression)) return;

  (void)ctx.checker.implicit_conversion(equal->expression, field->type());
}

void InitDeclaratorChecker::evaluateConstValue(VariableSymbol* var) {
  if (var->initializer()) {
    auto interp = ASTInterpreter{ctx.unit};
    auto value = interp.evaluate(var->initializer());

    const auto needsConstructor =
        !value.has_value() ||
        (var->constructor() &&
         std::holds_alternative<std::shared_ptr<InitializerList>>(*value));

    if (needsConstructor && var->isConstexpr()) {
      if (auto ctorValue = constexprEval.tryEvaluateConstructor(var, interp))
        value = std::move(ctorValue);
    }

    var->setConstValue(value);
  } else if (var->isConstexpr() && var->constructor()) {
    auto interp = ASTInterpreter{ctx.unit};
    var->setConstValue(constexprEval.tryEvaluateConstructor(var, interp));
  }

  if (var->isConstexpr() && !var->constValue().has_value()) {
    auto dep = isDependent(ctx.unit, var->type());
    if (!dep && var->initializer())
      dep = isDependent(ctx.unit, var->initializer());
    if (!dep)
      ctx.error(var->location(), "constexpr variable must be initialized");
  }
}
}  // namespace

void TypeChecker::check_init_declarator(InitDeclaratorAST* ast) {
  InitDeclaratorChecker{*this}.checkInitDeclarator(ast);
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
  if (!field || field->isStatic() || !field->initializer()) return;
  InitDeclaratorChecker{*this}.checkFieldInitializer(field);
}

auto TypeChecker::hasAutoPlaceholder(const Type* type) const -> bool {
  return containsAutoPlaceholder(type);
}

auto TypeChecker::deduceAutoType(const Type* declaredType,
                                 const Type* initializerType) -> const Type* {
  if (!initializerType) return nullptr;

  InitContext ctx{*this};

  if (type_cast<AutoType>(declaredType))
    return ctx.traits.remove_cvref(initializerType);

  return deduceCompoundAuto(ctx, declaredType, initializerType);
}

void TypeChecker::check_braced_init_list(const Type* type,
                                         BracedInitListAST* ast) {
  InitDeclaratorChecker{*this}.checkBracedInitList(type, ast);
}

auto TypeChecker::check_class_initializer(const Type* targetType,
                                          ExpressionAST*& initializer,
                                          SourceLocation location,
                                          List<ExpressionAST*>** argumentList)
    -> FunctionSymbol* {
  return InitDeclaratorChecker{*this}.checkClassInitializer(
      targetType, initializer, location, argumentList);
}
}  // namespace cxx
