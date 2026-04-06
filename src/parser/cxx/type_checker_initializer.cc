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

    return false;
  }

  [[nodiscard]] auto checkIntLitNonNarrowing(IntLiteralExpressionAST* intLit,
                                             const Type* targetType) -> bool {
    if (!intLit->literal) return false;
    auto value = intLit->literal->integerValue();

    if (ctx.traits.is_integral(targetType))
      return ctx.traits.integer_constant_fits_in_type(value, targetType);

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
    if (!ctx.traits.is_floating_point(targetType)) return false;
    auto value = floatLit->literal->floatValue();
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
  } else {
    elemChecker.check(
        expr, fieldType,
        std::format("cannot initialize member '{}' of type '{}' with "
                    "expression of type '{}'",
                    to_string(field->name()), to_string(fieldType),
                    to_string(expr->type)));
  }
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
  auto synthetic = buildSyntheticBracedList(it, slots);
  ctx.checker.check_braced_init_list(targetType, synthetic);
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

  void checkClassInit(VariableSymbol* var, InitDeclaratorAST* ast,
                      const Type* targetType);

 private:
  void checkAggregateInit(VariableSymbol* var, InitDeclaratorAST* ast,
                          const Type* targetType, ClassSymbol* classSymbol);
  void checkConstructorInit(VariableSymbol* var, InitDeclaratorAST* ast,
                            ClassSymbol* classSymbol);

  [[nodiscard]] auto tryInitializerListConstructor(
      VariableSymbol* var, BracedInitListAST* bracedInitList,
      ClassSymbol* classSymbol, OverloadResolution& overloadRes,
      ConstructorResult& resolution) -> bool;
};

void ClassInitChecker::checkClassInit(VariableSymbol* var,
                                      InitDeclaratorAST* ast,
                                      const Type* targetType) {
  auto classType = type_cast<ClassType>(targetType);
  if (!classType || !classType->symbol()) return;
  auto classSymbol = classType->symbol();

  bool isAggregate = true;
  for (auto ctor : classSymbol->constructors()) {
    if (!ctor->isDefaulted() && !ctor->isDeleted()) {
      isAggregate = false;
      break;
    }
  }

  if (isAggregate)
    checkAggregateInit(var, ast, targetType, classSymbol);
  else
    checkConstructorInit(var, ast, classSymbol);
}

void ClassInitChecker::checkAggregateInit(VariableSymbol* var,
                                          InitDeclaratorAST* ast,
                                          const Type* targetType,
                                          ClassSymbol* classSymbol) {
  auto bracedInitList = InitUnwrapper::getBracedInitList(ast->initializer);

  if (bracedInitList) {
    ctx.checker.check_braced_init_list(targetType, bracedInitList);
    return;
  }

  if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer);
      equal && equal->expression) {
    elemChecker.check(
        equal->expression, targetType,
        std::format("cannot initialize type '{}' with expression of type '{}'",
                    to_string(targetType), to_string(equal->expression->type)));
  }
}

void ClassInitChecker::checkConstructorInit(VariableSymbol* var,
                                            InitDeclaratorAST* ast,
                                            ClassSymbol* classSymbol) {
  auto args = InitUnwrapper::collectArgs(ast->initializer);
  OverloadResolution overloadRes(ctx.unit);
  auto resolution = overloadRes.resolveConstructor(classSymbol, args);

  auto bracedInitList = InitUnwrapper::getBracedInitList(ast->initializer);
  if (bracedInitList &&
      tryInitializerListConstructor(var, bracedInitList, classSymbol,
                                    overloadRes, resolution))
    return;

  if (!resolution.best) return;

  if (resolution.ambiguous) {
    ctx.error(var->location(), "constructor call is ambiguous");
    return;
  }

  var->setConstructor(resolution.best->symbol);
  InitUnwrapper::applyConversions(ctx.checker, ast->initializer,
                                  resolution.best->conversions);
}

auto ClassInitChecker::tryInitializerListConstructor(
    VariableSymbol* var, BracedInitListAST* bracedInitList,
    ClassSymbol* classSymbol, OverloadResolution& overloadRes,
    ConstructorResult& resolution) -> bool {
  std::vector<ExpressionAST*> listInitArgs = {bracedInitList};
  auto listInitResolution =
      overloadRes.resolveConstructor(classSymbol, listInitArgs);
  if (!listInitResolution.best) return false;

  resolution = std::move(listInitResolution);
  if (auto ctorType = type_cast<FunctionType>(resolution.best->symbol->type());
      ctorType && ctorType->parameterTypes().size() == 1) {
    auto ctorParamType = ctorType->parameterTypes().front();
    if (overloadRes.initializerListElementType(ctorParamType)) {
      bracedInitList->type = ctorParamType;
      bracedInitList->valueCategory = ValueCategory::kPrValue;
    }
  }

  if (!resolution.ambiguous)
    var->setConstructor(resolution.best->symbol);
  else
    ctx.error(var->location(), "constructor call is ambiguous");

  return true;
}

struct ScalarInitChecker {
  InitContext& ctx;

  void checkScalarInit(VariableSymbol* var, InitDeclaratorAST* ast,
                       const Type* targetType);
};

void ScalarInitChecker::checkScalarInit(VariableSymbol* var,
                                        InitDeclaratorAST* ast,
                                        const Type* targetType) {
  if (!ast->initializer) return;

  auto bracedInitList = InitUnwrapper::getBracedInitList(ast->initializer);
  if (bracedInitList) {
    ctx.checker.check_braced_init_list(targetType, bracedInitList);
    return;
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(ast->initializer);
  if (!initExpr) return;

  auto convTarget = InitUnwrapper::getConversionTarget(ast->initializer);
  ExpressionAST*& target = convTarget ? *convTarget : initExpr;

  if (!ctx.checker.implicit_conversion(target, targetType)) {
    auto seq = ctx.checker.checkImplicitConversion(target, targetType);
    ctx.checker.applyImplicitConversion(seq, target);
  }

  var->setInitializer(ast->initializer);
}

struct ReferenceInitChecker {
  InitContext& ctx;

  void check(VariableSymbol* var, InitDeclaratorAST* ast);
};

void ReferenceInitChecker::check(VariableSymbol* var, InitDeclaratorAST* ast) {
  auto targetType = var->type();

  if (!ast->initializer) {
    auto loc = ctx.checker.getInitDeclaratorLocation(ast, var);
    ctx.error(loc,
              std::format("reference variable of type '{}' must be initialized",
                          to_string(targetType)));
    return;
  }

  if (auto bracedInitList =
          InitUnwrapper::getBracedInitList(ast->initializer)) {
    if (!bracedInitList->expressionList ||
        bracedInitList->expressionList->next) {
      ctx.error(ast->initializer->firstSourceLocation(),
                "reference initializer must be a single expression");
      return;
    }
  }

  auto initExpr = InitUnwrapper::unwrapSingleExpr(ast->initializer);
  if (!initExpr) {
    ctx.error(ast->initializer->firstSourceLocation(),
              "reference initializer must be a single expression");
    return;
  }

  auto strippedInitializer =
      InitUnwrapper::stripImplicitCasts(ast->initializer);
  ExpressionAST*& conversionTarget =
      ast_cast<EqualInitializerAST>(strippedInitializer) ? ast->initializer
                                                         : initExpr;

  auto seq = ctx.checker.checkImplicitConversion(conversionTarget, targetType);
  if (seq.rank == ConversionRank::kNone) {
    ctx.error(
        initExpr->firstSourceLocation(),
        std::format("invalid initialization of reference of type '{}' from "
                    "expression of type '{}'",
                    to_string(targetType), to_string(initExpr->type)));
    return;
  }

  ctx.checker.applyImplicitConversion(seq, conversionTarget);
  var->setInitializer(ast->initializer);
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

void TypeDeducer::deduceAutoType(VariableSymbol* var) {
  if (!type_cast<AutoType>(var->type())) return;

  if (!var->initializer()) {
    ctx.error(var->location(), "variable with 'auto' type must be initialized");
  } else {
    auto deducedExpr = InitUnwrapper::unwrapSingleExpr(var->initializer());
    if (deducedExpr && deducedExpr->type)
      var->setType(ctx.traits.remove_cvref(deducedExpr->type));
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
  void checkBracedInitList(const Type* type, BracedInitListAST* ast);

 private:
  void checkInitialization(VariableSymbol* var, InitDeclaratorAST* ast);
  void evaluateConstValue(VariableSymbol* var);
};

void InitDeclaratorChecker::checkInitDeclarator(InitDeclaratorAST* ast) {
  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  var->setInitializer(ast->initializer);

  typeDeducer.deduceArraySize(var);
  typeDeducer.deduceAutoType(var);

  if (var->isConstexpr()) var->setType(ctx.traits.add_const(var->type()));

  checkInitialization(var, ast);
  evaluateConstValue(var);
}

void InitDeclaratorChecker::checkBracedInitList(const Type* type,
                                                BracedInitListAST* ast) {
  bracedChecker.check(type, ast);
}

void InitDeclaratorChecker::checkInitialization(VariableSymbol* var,
                                                InitDeclaratorAST* ast) {
  if (ctx.traits.is_reference(var->type())) {
    refChecker.check(var, ast);
    return;
  }

  auto targetType = ctx.traits.remove_cv(var->type());

  if (ctx.traits.is_class(targetType)) {
    classChecker.checkClassInit(var, ast, targetType);
    return;
  }

  scalarChecker.checkScalarInit(var, ast, targetType);
}

void InitDeclaratorChecker::evaluateConstValue(VariableSymbol* var) {
  if (var->initializer()) {
    auto interp = ASTInterpreter{ctx.unit};
    auto value = interp.evaluate(var->initializer());

    if (!value.has_value() && var->isConstexpr())
      value = constexprEval.tryEvaluateConstructor(var, interp);

    var->setConstValue(value);
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

void TypeChecker::check_braced_init_list(const Type* type,
                                         BracedInitListAST* ast) {
  InitDeclaratorChecker{*this}.checkBracedInitList(type, ast);
}

}  // namespace cxx