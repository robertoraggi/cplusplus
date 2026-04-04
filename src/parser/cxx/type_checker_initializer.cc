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

struct CheckInitDeclarator {
  TypeChecker& check;

  void check_initialization(VariableSymbol* var, InitDeclaratorAST* ast);
  void check_braced_init_list(const Type* type, BracedInitListAST* ast);

  void error(SourceLocation loc, std::string message) {
    check.error(loc, std::move(message));
  }

  void warning(SourceLocation loc, std::string message) {
    check.warning(loc, std::move(message));
  }

  [[nodiscard]] auto integer_constant_fits_in_type(ExpressionAST* expr,
                                                   const Type* targetType)
      -> bool;

  [[nodiscard]] auto constant_expression_non_narrowing(ExpressionAST* expr,
                                                       const Type* targetType)
      -> bool;

  [[nodiscard]] auto collect_init_args(ExpressionAST* initializer)
      -> std::vector<ExpressionAST*>;

  void apply_init_conversions(
      ExpressionAST* initializer,
      const std::vector<ImplicitConversionSequence>& conversions);

  void deduce_array_size(VariableSymbol* var);

  void deduce_auto_type(VariableSymbol* var);

  [[nodiscard]] auto single_paren_initializer_expr(ExpressionAST* expr)
      -> ExpressionAST*;

  [[nodiscard]] auto strip_implicit_initializer_casts(ExpressionAST* expr)
      -> ExpressionAST*;

  [[nodiscard]] auto unwrap_single_initializer_expr(ExpressionAST* initializer)
      -> ExpressionAST*;

  void check_init_declarator(InitDeclaratorAST* ast);

  [[nodiscard]] auto try_evaluate_constexpr_ctor(VariableSymbol* var,
                                                 ASTInterpreter& interp)
      -> std::optional<ConstValue>;

  [[nodiscard]] auto get_braced_initializer(ExpressionAST* initializer)
      -> BracedInitListAST*;

  [[nodiscard]] auto elem_type(const Type* t) -> const Type* {
    auto unit_ = check.translationUnit();
    return unit_->typeTraits().remove_cv(
        unit_->typeTraits().get_element_type(t));
  }

  [[nodiscard]] auto isAnyCharType(const Type* t) -> bool;

  void check_element_init(ExpressionAST*& expr, const Type* targetType,
                          std::string errorMessage);

  [[nodiscard]] auto is_narrowing_conversion(const Type* from, const Type* to)
      -> bool;

  void warn_narrowing(SourceLocation loc, const Type* from, const Type* to);

  void check_designated_initializer(const Type* currentType,
                                    DesignatedInitializerClauseAST* ast);

  [[nodiscard]] auto firstNonStaticField(ClassSymbol* symbol) -> FieldSymbol*;

  void check_union_init(ClassSymbol* classSymbol, BracedInitListAST* ast);
  void collectEffectiveFields(ClassSymbol* classSymbol,
                              std::vector<FieldSymbol*>& fields);

  void check_struct_init(ClassSymbol* classSymbol, BracedInitListAST* ast);

  void check_reference_initialization(VariableSymbol* var,
                                      InitDeclaratorAST* ast);
};

auto CheckInitDeclarator::integer_constant_fits_in_type(ExpressionAST* expr,
                                                        const Type* targetType)
    -> bool {
  // Strip implicit casts
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
    expr = cast->expression;
  }

  auto intLit = ast_cast<IntLiteralExpressionAST>(expr);
  if (!intLit || !intLit->literal) return false;

  auto value = intLit->literal->integerValue();

  auto unit = check.translationUnit();
  auto control = unit->control();

  if (!unit->typeTraits().is_integral(targetType)) return false;

  auto targetSize = control->memoryLayout()->sizeOf(targetType);
  if (!targetSize) return false;

  bool targetSigned = unit->typeTraits().is_signed(targetType);

  if (targetSigned) {
    auto maxVal = (std::uint64_t{1} << (*targetSize * 8 - 1)) - 1;
    return value <= maxVal;
  } else {
    if (*targetSize >= 8) return true;
    auto maxVal = (std::uint64_t{1} << (*targetSize * 8)) - 1;
    return value <= maxVal;
  }
}

auto CheckInitDeclarator::constant_expression_non_narrowing(
    ExpressionAST* expr, const Type* targetType) -> bool {
  if (!expr || !targetType) return false;

  auto unit = check.translationUnit();

  targetType = unit->typeTraits().remove_cv(targetType);

  while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
    expr = cast->expression;
  }

  if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
    return constant_expression_non_narrowing(nested->expression, targetType);
  }

  if (auto equal = ast_cast<EqualInitializerAST>(expr)) {
    return constant_expression_non_narrowing(equal->expression, targetType);
  }

  if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
    if (!paren->expressionList || paren->expressionList->next) return false;
    return constant_expression_non_narrowing(paren->expressionList->value,
                                             targetType);
  }

  if (auto intLit = ast_cast<IntLiteralExpressionAST>(expr)) {
    if (unit->typeTraits().is_integral(targetType)) {
      return integer_constant_fits_in_type(expr, targetType);
    }

    auto value = intLit->literal->integerValue();
    auto valueLD = static_cast<long double>(value);

    if (type_cast<FloatType>(targetType)) {
      auto converted = static_cast<float>(value);
      return std::isfinite(converted) &&
             static_cast<long double>(converted) == valueLD;
    }

    if (type_cast<DoubleType>(targetType)) {
      auto converted = static_cast<double>(value);
      return std::isfinite(converted) &&
             static_cast<long double>(converted) == valueLD;
    }

    if (type_cast<LongDoubleType>(targetType)) {
      auto converted = static_cast<long double>(value);
      return std::isfinite(static_cast<double>(converted)) &&
             converted == valueLD;
    }

    return false;
  }

  auto floatLit = ast_cast<FloatLiteralExpressionAST>(expr);
  if (!floatLit) return false;

  if (!unit->typeTraits().is_floating_point(targetType)) return false;

  auto value = floatLit->literal->floatValue();
  if (!std::isfinite(value)) return false;

  if (type_cast<FloatType>(targetType)) {
    auto converted = static_cast<float>(value);
    return std::isfinite(converted) && static_cast<double>(converted) == value;
  }

  if (type_cast<DoubleType>(targetType)) {
    return true;
  }

  if (type_cast<LongDoubleType>(targetType)) {
    auto converted = static_cast<long double>(value);
    return std::isfinite(static_cast<double>(converted)) &&
           static_cast<double>(converted) == value;
  }

  return false;
}

auto CheckInitDeclarator::collect_init_args(ExpressionAST* initializer)
    -> std::vector<ExpressionAST*> {
  std::vector<ExpressionAST*> args;
  if (!initializer) return args;

  initializer = strip_implicit_initializer_casts(initializer);

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    initializer = strip_implicit_initializer_casts(equal->expression);
  }

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

void CheckInitDeclarator::apply_init_conversions(
    ExpressionAST* initializer,
    const std::vector<ImplicitConversionSequence>& conversions) {
  if (!initializer) return;

  initializer = strip_implicit_initializer_casts(initializer);

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    check.applyImplicitConversion(conversions[0], equal->expression);
    return;
  }

  if (auto paren = ast_cast<ParenInitializerAST>(initializer)) {
    size_t i = 0;
    for (auto it = paren->expressionList; it; it = it->next, ++i)
      check.applyImplicitConversion(conversions[i], it->value);
  } else if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
    size_t i = 0;
    for (auto it = braced->expressionList; it; it = it->next, ++i)
      check.applyImplicitConversion(conversions[i], it->value);
  } else {
    check.applyImplicitConversion(conversions[0], initializer);
  }
}

void CheckInitDeclarator::deduce_array_size(VariableSymbol* var) {
  auto unit_ = check.translationUnit();

  auto ty = type_cast<UnboundedArrayType>(var->type());
  if (!ty) return;

  auto initializer = var->initializer();
  if (!initializer) return;

  auto bracedInitList = get_braced_initializer(initializer);

  if (bracedInitList) {
    auto interp = ASTInterpreter{unit_};
    size_t currentIndex = 0;
    size_t maxIndex = 0;
    bool hasElements = false;

    for (auto it = bracedInitList->expressionList; it; it = it->next) {
      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value)) {
        if (desig->designatorList) {
          if (auto subscript = ast_cast<SubscriptDesignatorAST>(
                  desig->designatorList->value)) {
            if (auto value = interp.evaluate(subscript->expression)) {
              if (auto idx = interp.toUInt(*value)) {
                currentIndex = *idx;
              }
            }
          }
        }
      }
      if (!hasElements || currentIndex > maxIndex) maxIndex = currentIndex;
      hasElements = true;
      ++currentIndex;
    }

    if (hasElements) {
      const auto arrayType = unit_->control()->getBoundedArrayType(
          ty->elementType(), maxIndex + 1);

      var->setType(arrayType);
    }

    return;
  }

  auto initExpr = unwrap_single_initializer_expr(initializer);

  if (initExpr) {
    if (auto boundedArray = type_cast<BoundedArrayType>(initExpr->type)) {
      const auto arrayType = unit_->control()->getBoundedArrayType(
          ty->elementType(), boundedArray->size());

      var->setType(arrayType);
    }
  }
}

void CheckInitDeclarator::deduce_auto_type(VariableSymbol* var) {
  auto unit_ = check.translationUnit();

  if (!type_cast<AutoType>(var->type())) return;

  if (!var->initializer()) {
    error(var->location(), "variable with 'auto' type must be initialized");
  } else {
    auto deducedExpr = unwrap_single_initializer_expr(var->initializer());

    if (deducedExpr && deducedExpr->type)
      var->setType(unit_->typeTraits().remove_cvref(deducedExpr->type));
  }
}

auto CheckInitDeclarator::single_paren_initializer_expr(ExpressionAST* expr)
    -> ExpressionAST* {
  auto paren = ast_cast<ParenInitializerAST>(expr);
  if (!paren) return nullptr;
  if (!paren->expressionList || paren->expressionList->next) return nullptr;
  return paren->expressionList->value;
}

auto CheckInitDeclarator::strip_implicit_initializer_casts(ExpressionAST* expr)
    -> ExpressionAST* {
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
    expr = cast->expression;
  }
  return expr;
}

auto CheckInitDeclarator::unwrap_single_initializer_expr(
    ExpressionAST* initializer) -> ExpressionAST* {
  initializer = strip_implicit_initializer_casts(initializer);

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    initializer = strip_implicit_initializer_casts(equal->expression);
  }

  if (auto expr = single_paren_initializer_expr(initializer)) {
    return expr;
  }

  if (ast_cast<BracedInitListAST>(initializer)) {
    return nullptr;
  }

  return initializer;
}

void CheckInitDeclarator::check_init_declarator(InitDeclaratorAST* ast) {
  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  auto unit_ = check.translationUnit();

  var->setInitializer(ast->initializer);

  deduce_array_size(var);
  deduce_auto_type(var);

  if (var->isConstexpr())
    var->setType(unit_->typeTraits().add_const(var->type()));

  check_initialization(var, ast);

  if (var->initializer()) {
    auto interp = ASTInterpreter{unit_};
    auto value = interp.evaluate(var->initializer());

    if (!value.has_value() && var->isConstexpr())
      value = try_evaluate_constexpr_ctor(var, interp);

    var->setConstValue(value);
  }

  if (var->isConstexpr() && !var->constValue().has_value()) {
    auto dep = isDependent(unit_, var->type());

    if (!dep && var->initializer())
      dep = isDependent(unit_, var->initializer());

    if (!dep) error(var->location(), "constexpr variable must be initialized");
  }
}

auto CheckInitDeclarator::try_evaluate_constexpr_ctor(VariableSymbol* var,
                                                      ASTInterpreter& interp)
    -> std::optional<ConstValue> {
  auto classType = type_cast<ClassType>(
      check.translationUnit()->typeTraits().remove_cv(var->type()));
  if (!classType) return std::nullopt;

  auto classSym = classType->symbol();
  if (!classSym) return std::nullopt;

  auto initArgs = collect_init_args(var->initializer());

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

auto CheckInitDeclarator::get_braced_initializer(ExpressionAST* initializer)
    -> BracedInitListAST* {
  initializer = strip_implicit_initializer_casts(initializer);

  if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
    return braced;
  }

  if (auto equal = ast_cast<EqualInitializerAST>(initializer)) {
    auto expr = strip_implicit_initializer_casts(equal->expression);
    return ast_cast<BracedInitListAST>(expr);
  }

  return nullptr;
}

auto CheckInitDeclarator::isAnyCharType(const Type* t) -> bool {
  return type_cast<CharType>(t) || type_cast<SignedCharType>(t) ||
         type_cast<UnsignedCharType>(t);
}

void CheckInitDeclarator::check_braced_init_list(const Type* type,
                                                 BracedInitListAST* ast) {
  auto unit_ = check.translationUnit();
  auto control = unit_->control();

  ast->type = type;

  // Skip validation when the target type is dependent.
  if (type && isDependent(unit_, type)) return;

  if (unit_->typeTraits().is_array(type)) {
    // Array initialization
    auto elementType = elem_type(type);
    auto interp = ASTInterpreter{unit_};
    size_t index = 0;
    for (auto it = ast->expressionList; it; it = it->next) {
      auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value);

      if (desig) {
        if (auto firstDesigNode = desig->designatorList) {
          if (auto subscript =
                  ast_cast<SubscriptDesignatorAST>(firstDesigNode->value)) {
            if (auto val = interp.evaluate(subscript->expression)) {
              if (auto idx = interp.toUInt(*val)) index = *idx;
            }
          }
        }
      }

      if (auto boundedArrayType = type_cast<BoundedArrayType>(type)) {
        if (index >= boundedArrayType->size()) {
          error(it->value->firstSourceLocation(),
                "excess elements in array initializer");
          break;
        }
      }

      if (auto nested = ast_cast<BracedInitListAST>(it->value)) {
        check_braced_init_list(elementType, nested);
      } else if (desig) {
        check_designated_initializer(type, desig);
      } else if (auto strLit = ast_cast<StringLiteralExpressionAST>(it->value);
                 strLit && unit_->typeTraits().is_array(elementType)) {
        auto destElemType = elem_type(elementType);
        auto srcElemType = elem_type(strLit->type);
        bool compatible =
            unit_->typeTraits().is_same(destElemType, srcElemType) ||
            (isAnyCharType(destElemType) && isAnyCharType(srcElemType));
        if (!compatible) {
          error(it->value->firstSourceLocation(),
                std::format("cannot initialize array element of type '{}' with "
                            "expression of type '{}'",
                            to_string(elementType), to_string(strLit->type)));
        } else if (auto destArray = type_cast<BoundedArrayType>(elementType)) {
          if (auto srcArray = type_cast<BoundedArrayType>(strLit->type)) {
            const auto isC = unit_->language() == LanguageKind::kC;
            const auto maxCharacters =
                isC ? destArray->size() : destArray->size() - 1;

            if (srcArray->size() > maxCharacters) {
              error(
                  it->value->firstSourceLocation(),
                  std::format("initializer-string for char array is too long"));
            }
          }
        }
      } else {
        check_element_init(
            it->value, elementType,
            std::format("cannot initialize array element of type '{}' with "
                        "expression of type '{}'",
                        to_string(elementType), to_string(it->value->type)));
      }
      ++index;
    }
  } else if (unit_->typeTraits().is_class_or_union(type)) {
    auto classType = type_cast<ClassType>(type);
    if (!classType || !classType->symbol()) return;
    auto classSymbol = classType->symbol();

    if (classType->isUnion()) {
      check_union_init(classSymbol, ast);
    } else {
      check_struct_init(classSymbol, ast);
    }
  } else {
    auto it = ast->expressionList;
    if (!it) {
      // Empty braces
      return;
    }

    // Scalar initializer may contain exactly one element
    if (it->next) {
      error(it->next->value->firstSourceLocation(),
            "excess elements in scalar initializer");
    }

    auto& expr = it->value;

    // Unwrap a designated init clause if present (error for scalars)
    if (ast_cast<DesignatedInitializerClauseAST>(expr)) {
      error(expr->firstSourceLocation(),
            "designator in initializer for scalar type");
      return;
    }

    check_element_init(expr, type,
                       std::format("cannot initialize type '{}' with "
                                   "expression of type '{}'",
                                   to_string(type), to_string(expr->type)));
  }
}

void CheckInitDeclarator::check_element_init(ExpressionAST*& expr,
                                             const Type* targetType,
                                             std::string errorMessage) {
  auto unit_ = check.translationUnit();
  auto control = unit_->control();

  const auto isC = unit_->language() == LanguageKind::kC;

  if (unit_->typeTraits().is_array(targetType)) {
    if (auto strLit = ast_cast<StringLiteralExpressionAST>(expr)) {
      auto destElemType = elem_type(targetType);
      auto srcElemType = elem_type(strLit->type);

      const auto compatible =
          unit_->typeTraits().is_same(destElemType, srcElemType) ||
          (isAnyCharType(destElemType) && isAnyCharType(srcElemType));

      if (!compatible) {
        error(expr->firstSourceLocation(), std::move(errorMessage));
        return;
      }

      if (auto destArray = type_cast<BoundedArrayType>(targetType)) {
        if (auto srcArray = type_cast<BoundedArrayType>(strLit->type)) {
          const auto maxCharacters =
              isC ? destArray->size() : destArray->size() - 1;

          if (srcArray->size() > maxCharacters) {
            error(expr->firstSourceLocation(),
                  "initializer-string for char array is too long");
            return;
          }
        }
      }

      return;
    }

    // brace elision
    auto elementType = elem_type(targetType);
    check_element_init(expr, elementType, std::move(errorMessage));
    return;
  }

  if (unit_->typeTraits().is_lvalue_reference(targetType)) {
    while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
      if (cast->castKind != ImplicitCastKind::kIdentity &&
          cast->castKind != ImplicitCastKind::kLValueToRValueConversion) {
        break;
      }
      if (!cast->expression) break;
      expr = cast->expression;
    }
  }

  if (unit_->typeTraits().is_lvalue_reference(targetType) && is_lvalue(expr)) {
    auto sourceType = unit_->typeTraits().remove_reference(expr->type);
    auto referredType = unit_->typeTraits().remove_reference(targetType);

    if (unit_->typeTraits().is_same(
            unit_->typeTraits().remove_cv(sourceType),
            unit_->typeTraits().remove_cv(referredType))) {
      auto sourceCv = unit_->typeTraits().get_cv_qualifiers(sourceType);
      auto targetCv = unit_->typeTraits().get_cv_qualifiers(referredType);

      if (sourceCv == targetCv || sourceCv == CvQualifiers::kNone ||
          targetCv == CvQualifiers::kConstVolatile) {
        return;
      }
    }
  }

  auto sourceType = expr->type;
  if (!check.implicit_conversion(expr, targetType)) {
    error(expr->firstSourceLocation(), std::move(errorMessage));
  } else {
    if (sourceType && is_narrowing_conversion(sourceType, targetType) &&
        !constant_expression_non_narrowing(expr, targetType)) {
      warn_narrowing(expr->firstSourceLocation(), sourceType, targetType);
    }
  }
}

auto CheckInitDeclarator::is_narrowing_conversion(const Type* from,
                                                  const Type* to) -> bool {
  auto unit_ = check.translationUnit();

  if (unit_->language() != LanguageKind::kCXX) return false;

  auto control = unit_->control();

  from = unit_->typeTraits().remove_cv(from);
  to = unit_->typeTraits().remove_cv(to);

  if (unit_->typeTraits().is_same(from, to)) return false;

  if (unit_->typeTraits().is_floating_point(from) &&
      unit_->typeTraits().is_integral(to))
    return true;

  if (unit_->typeTraits().is_floating_point(from) &&
      unit_->typeTraits().is_floating_point(to)) {
    auto fromSize = control->memoryLayout()->sizeOf(from);
    auto toSize = control->memoryLayout()->sizeOf(to);
    if (fromSize && toSize && *fromSize > *toSize) return true;
  }

  if (unit_->typeTraits().is_integral_or_unscoped_enum(from) &&
      unit_->typeTraits().is_floating_point(to))
    return true;

  if (unit_->typeTraits().is_integral_or_unscoped_enum(from) &&
      unit_->typeTraits().is_integral(to)) {
    auto fromSize = control->memoryLayout()->sizeOf(from);
    auto toSize = control->memoryLayout()->sizeOf(to);
    if (fromSize && toSize) {
      if (*fromSize > *toSize) return true;
      if (*fromSize == *toSize && unit_->typeTraits().is_signed(from) !=
                                      unit_->typeTraits().is_signed(to))
        return true;
    }
  }

  return false;
}

void CheckInitDeclarator::warn_narrowing(SourceLocation loc, const Type* from,
                                         const Type* to) {
  warning(loc, std::format("narrowing conversion from '{}' to '{}' in "
                           "braced-init-list",
                           to_string(from), to_string(to)));
}

void CheckInitDeclarator::check_designated_initializer(
    const Type* currentType, DesignatedInitializerClauseAST* ast) {
  auto unit_ = check.translationUnit();
  auto control = unit_->control();

  const Type* targetType = currentType;

  for (auto desigIt = ast->designatorList; desigIt; desigIt = desigIt->next) {
    auto designator = desigIt->value;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto classType =
          type_cast<ClassType>(unit_->typeTraits().remove_cv(targetType));
      if (!classType || !classType->symbol()) {
        error(dot->firstSourceLocation(),
              std::format("member designator on non-aggregate type '{}'",
                          to_string(targetType)));
        return;
      }

      auto member = qualifiedLookup(classType->symbol(), dot->identifier);
      auto field = symbol_cast<FieldSymbol>(member);

      if (!field) {
        error(dot->firstSourceLocation(),
              std::format(
                  "field designator '{}' does not refer to a "
                  "non-static data member",
                  dot->identifier ? dot->identifier->name() : "<anonymous>"));
        return;
      }

      dot->symbol = field;
      targetType = unit_->typeTraits().remove_cv(field->type());
    } else if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      check(subscript->expression);
      if (!unit_->typeTraits().is_array(targetType)) {
        error(subscript->firstSourceLocation(),
              std::format("array designator on non-array type '{}'",
                          to_string(targetType)));
        return;
      }
      targetType = elem_type(targetType);
    }
  }

  if (!ast->initializer) return;

  // Skip initialization check when the target type is dependent.
  if (targetType && isDependent(unit_, targetType)) return;

  if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
    if (auto nested = ast_cast<BracedInitListAST>(equal->expression)) {
      check_braced_init_list(targetType, nested);
    } else if (equal->expression) {
      check_element_init(
          equal->expression, targetType,
          std::format("cannot initialize type '{}' with expression of "
                      "type '{}'",
                      to_string(targetType),
                      to_string(equal->expression->type)));
    }
  } else if (auto initExpr = ast_cast<BracedInitListAST>(ast->initializer)) {
    check_braced_init_list(targetType, initExpr);
  }

  ast->type = targetType;
}

auto CheckInitDeclarator::firstNonStaticField(ClassSymbol* symbol)
    -> FieldSymbol* {
  for (auto field : views::members(symbol) | views::non_static_fields)
    return field;
  return nullptr;
}

void CheckInitDeclarator::check_union_init(ClassSymbol* classSymbol,
                                           BracedInitListAST* ast) {
  auto unit_ = check.translationUnit();
  auto control = unit_->control();

  auto it = ast->expressionList;
  if (!it) return;

  auto& expr = it->value;

  if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
    check_designated_initializer(control->getClassType(classSymbol), desig);
    if (it->next) {
      error(it->next->value->firstSourceLocation(),
            "excess elements in union initializer");
    }
    return;
  }

  auto* firstField = firstNonStaticField(classSymbol);

  if (!firstField) {
    error(expr->firstSourceLocation(), "union has no named members");
    return;
  }

  auto fieldType = unit_->typeTraits().remove_cv(firstField->type());

  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    check_braced_init_list(fieldType, nested);
  } else {
    check_element_init(
        expr, fieldType,
        std::format("cannot initialize member '{}' of type '{}' with "
                    "expression of type '{}'",
                    to_string(firstField->name()), to_string(fieldType),
                    to_string(expr->type)));
  }

  if (it->next) {
    error(it->next->value->firstSourceLocation(),
          "excess elements in union initializer");
  }
}

void CheckInitDeclarator::collectEffectiveFields(
    ClassSymbol* classSymbol, std::vector<FieldSymbol*>& fields) {
  auto unit = check.translationUnit();

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name()) {
      auto classType =
          type_cast<ClassType>(unit->typeTraits().remove_cv(field->type()));
      if (classType && classType->symbol()) {
        auto nestedClass = classType->symbol();
        if (nestedClass->isUnion()) {
          fields.push_back(field);
        } else {
          collectEffectiveFields(nestedClass, fields);
        }
      }
    } else {
      fields.push_back(field);
    }
  }
}

void CheckInitDeclarator::check_struct_init(ClassSymbol* classSymbol,
                                            BracedInitListAST* ast) {
  auto unit_ = check.translationUnit();

  std::vector<FieldSymbol*> fields;
  collectEffectiveFields(classSymbol, fields);

  size_t fieldIndex = 0;

  for (auto it = ast->expressionList; it; it = it->next) {
    auto& expr = it->value;

    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
      check_designated_initializer(unit_->control()->getClassType(classSymbol),
                                   desig);
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
      error(expr->firstSourceLocation(),
            "excess elements in struct initializer");
      break;
    }

    auto fieldType = unit_->typeTraits().remove_cv(fields[fieldIndex]->type());

    if (auto nested = ast_cast<BracedInitListAST>(expr)) {
      check_braced_init_list(fieldType, nested);
    } else if (!fields[fieldIndex]->name()) {
      auto classType = type_cast<ClassType>(fieldType);
      auto anonUnionSymbol =
          (classType && classType->symbol() && classType->symbol()->isUnion())
              ? classType->symbol()
              : nullptr;
      if (!anonUnionSymbol) {
      } else if (auto* firstField = firstNonStaticField(anonUnionSymbol)) {
        auto firstType = unit_->typeTraits().remove_cv(firstField->type());
        check_element_init(
            expr, firstType,
            std::format("cannot initialize anonymous union member '{}' of "
                        "type '{}' with expression of type '{}'",
                        to_string(firstField->name()), to_string(firstType),
                        to_string(expr->type)));
      } else {
        error(expr->firstSourceLocation(), "union has no named members");
      }
    } else {
      check_element_init(
          expr, fieldType,
          std::format("cannot initialize member '{}' of type '{}' with "
                      "expression of type '{}'",
                      to_string(fields[fieldIndex]->name()),
                      to_string(fieldType), to_string(expr->type)));
    }

    ++fieldIndex;
  }
}

void CheckInitDeclarator::check_initialization(VariableSymbol* var,
                                               InitDeclaratorAST* ast) {
  auto unit_ = check.translationUnit();
  auto control = unit_->control();

  if (unit_->typeTraits().is_reference(var->type())) {
    check_reference_initialization(var, ast);
    return;
  }

  auto targetType = unit_->typeTraits().remove_cv(var->type());

  if (unit_->typeTraits().is_class(targetType)) {
    auto classType = type_cast<ClassType>(targetType);
    auto classSymbol = classType->symbol();
    if (!classSymbol) return;

    bool isAggregate = true;
    for (auto ctor : classSymbol->constructors()) {
      if (!ctor->isDefaulted() && !ctor->isDeleted()) {
        isAggregate = false;
        break;
      }
    }

    auto bracedInitList = get_braced_initializer(ast->initializer);

    if (isAggregate && bracedInitList) {
      check_braced_init_list(targetType, bracedInitList);
      return;
    }

    if (isAggregate && !bracedInitList) {
      if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer);
          equal && equal->expression) {
        check_element_init(
            equal->expression, targetType,
            std::format(
                "cannot initialize type '{}' with expression of type '{}'",
                to_string(targetType), to_string(equal->expression->type)));
      }
      return;
    }

    auto args = collect_init_args(ast->initializer);
    OverloadResolution overloadRes(unit_);
    auto resolution = overloadRes.resolveConstructor(classSymbol, args);

    if (bracedInitList) {
      std::vector<ExpressionAST*> listInitArgs = {bracedInitList};
      auto listInitResolution =
          overloadRes.resolveConstructor(classSymbol, listInitArgs);
      if (listInitResolution.best) {
        resolution = std::move(listInitResolution);
        if (auto ctorType =
                type_cast<FunctionType>(resolution.best->symbol->type());
            ctorType && ctorType->parameterTypes().size() == 1) {
          auto ctorParamType = ctorType->parameterTypes().front();
          if (overloadRes.initializerListElementType(ctorParamType)) {
            bracedInitList->type = ctorParamType;
            bracedInitList->valueCategory = ValueCategory::kPrValue;
          }
        }

        if (!resolution.ambiguous) {
          var->setConstructor(resolution.best->symbol);
        } else {
          error(var->location(), "constructor call is ambiguous");
        }
        return;
      }
    }

    if (!resolution.best) return;

    if (resolution.ambiguous) {
      error(var->location(), "constructor call is ambiguous");
      return;
    }

    var->setConstructor(resolution.best->symbol);
    apply_init_conversions(ast->initializer, resolution.best->conversions);
    return;
  }

  if (!ast->initializer) return;

  auto bracedInitList = get_braced_initializer(ast->initializer);

  if (bracedInitList) {
    check_braced_init_list(targetType, bracedInitList);
  } else {
    auto initExpr = unwrap_single_initializer_expr(ast->initializer);

    if (!initExpr) return;

    auto strippedInitializer =
        strip_implicit_initializer_casts(ast->initializer);

    ExpressionAST** conversionTargetPtr = nullptr;
    if (ast_cast<EqualInitializerAST>(strippedInitializer)) {
      conversionTargetPtr = &ast->initializer;
    } else if (auto paren = ast_cast<ParenInitializerAST>(ast->initializer)) {
      if (paren->expressionList && !paren->expressionList->next)
        conversionTargetPtr = &paren->expressionList->value;
    }
    if (!conversionTargetPtr) conversionTargetPtr = &initExpr;

    if (!check.implicit_conversion(*conversionTargetPtr, targetType)) {
      auto seq =
          check.checkImplicitConversion(*conversionTargetPtr, targetType);

      check.applyImplicitConversion(seq, *conversionTargetPtr);
    }

    var->setInitializer(ast->initializer);
  }
}

void CheckInitDeclarator::check_reference_initialization(
    VariableSymbol* var, InitDeclaratorAST* ast) {
  auto targetType = var->type();

  if (!ast->initializer) {
    auto loc = check.getInitDeclaratorLocation(ast, var);

    error(loc,
          std::format("reference variable of type '{}' must be initialized",
                      to_string(targetType)));
    return;
  }

  if (auto bracedInitList = get_braced_initializer(ast->initializer)) {
    if (!bracedInitList->expressionList ||
        bracedInitList->expressionList->next) {
      error(ast->initializer->firstSourceLocation(),
            "reference initializer must be a single expression");
      return;
    }
  }

  auto initExpr = unwrap_single_initializer_expr(ast->initializer);
  if (!initExpr) {
    error(ast->initializer->firstSourceLocation(),
          "reference initializer must be a single expression");
    return;
  }

  auto strippedInitializer = strip_implicit_initializer_casts(ast->initializer);
  ExpressionAST*& conversionTarget =
      ast_cast<EqualInitializerAST>(strippedInitializer) ? ast->initializer
                                                         : initExpr;

  auto seq = check.checkImplicitConversion(conversionTarget, targetType);
  if (seq.rank == ConversionRank::kNone) {
    error(initExpr->firstSourceLocation(),
          std::format("invalid initialization of reference of type '{}' from "
                      "expression of type '{}'",
                      to_string(targetType), to_string(initExpr->type)));
    return;
  }

  check.applyImplicitConversion(seq, conversionTarget);
  var->setInitializer(ast->initializer);
}

}  // namespace

void TypeChecker::check_init_declarator(InitDeclaratorAST* ast) {
  CheckInitDeclarator{*this}.check_init_declarator(ast);
}

void TypeChecker::check_braced_init_list(const Type* type,
                                         BracedInitListAST* ast) {
  CheckInitDeclarator{*this}.check_braced_init_list(type, ast);
}

}  // namespace cxx