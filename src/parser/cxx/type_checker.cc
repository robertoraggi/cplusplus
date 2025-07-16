// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/type_checker.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct TypeChecker::Visitor {
  TypeChecker& check;

  [[nodiscard]] auto arena() const -> Arena* { return check.unit_->arena(); }

  [[nodiscard]] auto globalScope() const -> Scope* {
    return check.unit_->globalScope();
  }

  [[nodiscard]] auto scope() const -> Scope* { return check.scope_; }

  [[nodiscard]] auto control() const -> Control* {
    return check.unit_->control();
  }

  void error(SourceLocation loc, std::string message) {
    if (!check.reportErrors_) return;
    check.unit_->error(loc, std::move(message));
  }

  void warning(SourceLocation loc, std::string message) {
    if (!check.reportErrors_) return;
    check.unit_->warning(loc, std::move(message));
  }

  [[nodiscard]] auto strip_parentheses(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto strip_cv(const Type*& type) -> CvQualifiers;
  [[nodiscard]] auto merge_cv(CvQualifiers cv1, CvQualifiers cv2) const
      -> CvQualifiers;
  [[nodiscard]] auto is_const(CvQualifiers cv) const -> bool;
  [[nodiscard]] auto is_volatile(CvQualifiers cv) const -> bool;

  // standard conversions
  [[nodiscard]] auto lvalue_to_rvalue_conversion(ExpressionAST*& expr) -> bool;
  [[nodiscard]] auto array_to_pointer_conversion(ExpressionAST*& expr) -> bool;
  [[nodiscard]] auto function_to_pointer_conversion(ExpressionAST*& expr)
      -> bool;
  [[nodiscard]] auto integral_promotion(ExpressionAST*& expr) -> bool;
  [[nodiscard]] auto floating_point_promotion(ExpressionAST*& expr) -> bool;
  [[nodiscard]] auto integral_conversion(ExpressionAST*& expr,
                                         const Type* destinationType) -> bool;
  [[nodiscard]] auto floating_point_conversion(ExpressionAST*& expr,
                                               const Type* destinationType)
      -> bool;
  [[nodiscard]] auto floating_integral_conversion(ExpressionAST*& expr,
                                                  const Type* destinationType)
      -> bool;
  [[nodiscard]] auto pointer_conversion(ExpressionAST*& expr,
                                        const Type* destinationType) -> bool;
  [[nodiscard]] auto pointer_to_member_conversion(ExpressionAST*& expr,
                                                  const Type* destinationType)
      -> bool;
  [[nodiscard]] auto function_pointer_conversion(ExpressionAST*& expr,
                                                 const Type* destinationType)
      -> bool;
  [[nodiscard]] auto boolean_conversion(ExpressionAST*& expr,
                                        const Type* destinationType) -> bool;
  [[nodiscard]] auto temporary_materialization_conversion(ExpressionAST*& expr)
      -> bool;
  [[nodiscard]] auto qualification_conversion(ExpressionAST*& expr,
                                              const Type* destinationType)
      -> bool;

  [[nodiscard]] auto ensure_prvalue(ExpressionAST*& expr) -> bool;

  void adjust_cv(ExpressionAST* expr);

  [[nodiscard]] auto implicit_conversion(ExpressionAST*& expr,
                                         const Type* destinationType) -> bool;

  [[nodiscard]] auto usual_arithmetic_conversion(ExpressionAST*& expr,
                                                 ExpressionAST*& other)
      -> const Type*;

  [[nodiscard]] auto get_qualification_combined_type(const Type* left,
                                                     const Type* right)
      -> const Type*;

  [[nodiscard]] auto get_qualification_combined_type(
      const Type* left, const Type* right, bool& didChangeTypeOrQualifiers)
      -> const Type*;

  [[nodiscard]] auto composite_pointer_type(ExpressionAST*& expr,
                                            ExpressionAST*& other)
      -> const Type*;

  [[nodiscard]] auto is_null_pointer_constant(ExpressionAST* expr) const
      -> bool;

  [[nodiscard]] auto check_cv_qualifiers(CvQualifiers target,
                                         CvQualifiers source) const -> bool;

  void check_cpp_cast_expression(CppCastExpressionAST* ast);
  [[nodiscard]] auto check_static_cast(CppCastExpressionAST* ast) -> bool;
  [[nodiscard]] auto check_cast_to_derived(const Type* targetType,
                                           ExpressionAST* expression) -> bool;

  void check_addition(BinaryExpressionAST* ast);
  void check_subtraction(BinaryExpressionAST* ast);

  [[nodiscard]] auto check_member_access(MemberExpressionAST* ast) -> bool;
  [[nodiscard]] auto check_pseudo_destructor_access(MemberExpressionAST* ast)
      -> bool;

  void operator()(GeneratedLiteralExpressionAST* ast);
  void operator()(CharLiteralExpressionAST* ast);
  void operator()(BoolLiteralExpressionAST* ast);
  void operator()(IntLiteralExpressionAST* ast);
  void operator()(FloatLiteralExpressionAST* ast);
  void operator()(NullptrLiteralExpressionAST* ast);
  void operator()(StringLiteralExpressionAST* ast);
  void operator()(UserDefinedStringLiteralExpressionAST* ast);
  void operator()(ObjectLiteralExpressionAST* ast);
  void operator()(ThisExpressionAST* ast);
  void operator()(GenericSelectionExpressionAST* ast);
  void operator()(NestedStatementExpressionAST* ast);
  void operator()(NestedExpressionAST* ast);
  void operator()(IdExpressionAST* ast);
  void operator()(LambdaExpressionAST* ast);
  void operator()(FoldExpressionAST* ast);
  void operator()(RightFoldExpressionAST* ast);
  void operator()(LeftFoldExpressionAST* ast);
  void operator()(RequiresExpressionAST* ast);
  void operator()(VaArgExpressionAST* ast);
  void operator()(SubscriptExpressionAST* ast);
  void operator()(CallExpressionAST* ast);
  void operator()(TypeConstructionAST* ast);
  void operator()(BracedTypeConstructionAST* ast);
  void operator()(SpliceMemberExpressionAST* ast);
  void operator()(MemberExpressionAST* ast);
  void operator()(PostIncrExpressionAST* ast);
  void operator()(CppCastExpressionAST* ast);
  void operator()(BuiltinBitCastExpressionAST* ast);
  void operator()(BuiltinOffsetofExpressionAST* ast);
  void operator()(TypeidExpressionAST* ast);
  void operator()(TypeidOfTypeExpressionAST* ast);
  void operator()(SpliceExpressionAST* ast);
  void operator()(GlobalScopeReflectExpressionAST* ast);
  void operator()(NamespaceReflectExpressionAST* ast);
  void operator()(TypeIdReflectExpressionAST* ast);
  void operator()(ReflectExpressionAST* ast);
  void operator()(LabelAddressExpressionAST* ast);
  void operator()(UnaryExpressionAST* ast);
  void operator()(AwaitExpressionAST* ast);
  void operator()(SizeofExpressionAST* ast);
  void operator()(SizeofTypeExpressionAST* ast);
  void operator()(SizeofPackExpressionAST* ast);
  void operator()(AlignofTypeExpressionAST* ast);
  void operator()(AlignofExpressionAST* ast);
  void operator()(NoexceptExpressionAST* ast);
  void operator()(NewExpressionAST* ast);
  void operator()(DeleteExpressionAST* ast);
  void operator()(CastExpressionAST* ast);
  void operator()(ImplicitCastExpressionAST* ast);
  void operator()(BinaryExpressionAST* ast);
  void operator()(ConditionalExpressionAST* ast);
  void operator()(YieldExpressionAST* ast);
  void operator()(ThrowExpressionAST* ast);
  void operator()(AssignmentExpressionAST* ast);
  void operator()(PackExpansionExpressionAST* ast);
  void operator()(DesignatedInitializerClauseAST* ast);
  void operator()(TypeTraitExpressionAST* ast);
  void operator()(ConditionExpressionAST* ast);
  void operator()(EqualInitializerAST* ast);
  void operator()(BracedInitListAST* ast);
  void operator()(ParenInitializerAST* ast);
};

void TypeChecker::Visitor::operator()(GeneratedLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CharLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BoolLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(IntLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(FloatLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NullptrLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(StringLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ObjectLiteralExpressionAST* ast) {
  if (ast->typeId) {
    ast->type = ast->typeId->type;
  }
  ast->valueCategory = ValueCategory::kLValue;
}

void TypeChecker::Visitor::operator()(ThisExpressionAST* ast) {
  auto scope_ = check.scope_;

  for (auto current = scope_; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current->owner())) {
      // maybe a this expression in a field initializer
      ast->type = control()->getPointerType(classSymbol->type());
      break;
    }

    if (auto functionSymbol = symbol_cast<FunctionSymbol>(current->owner())) {
      if (auto classSymbol =
              symbol_cast<ClassSymbol>(functionSymbol->enclosingSymbol())) {
        auto functionType = type_cast<FunctionType>(functionSymbol->type());
        const auto cv = functionType->cvQualifiers();
        if (cv != CvQualifiers::kNone) {
          auto elementType = control()->getQualType(classSymbol->type(), cv);
          ast->type = control()->getPointerType(elementType);
        } else {
          ast->type = control()->getPointerType(classSymbol->type());
        }
      }

      break;
    }
  }
}

void TypeChecker::Visitor::operator()(GenericSelectionExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NestedStatementExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NestedExpressionAST* ast) {
  if (!ast->expression) return;
  ast->type = ast->expression->type;
  ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::operator()(IdExpressionAST* ast) {
  if (ast->symbol) {
    if (auto conceptSymbol = symbol_cast<ConceptSymbol>(ast->symbol)) {
      ast->type = control()->getBoolType();
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      ast->type = control()->remove_reference(ast->symbol->type());

      if (ast->symbol->isEnumerator() || ast->symbol->isNonTypeParameter()) {
        ast->valueCategory = ValueCategory::kPrValue;
        adjust_cv(ast);
      } else {
        ast->valueCategory = ValueCategory::kLValue;
      }
    }
  }
}

void TypeChecker::Visitor::operator()(LambdaExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(FoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(RightFoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(LeftFoldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(RequiresExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(VaArgExpressionAST* ast) {
  if (ast->typeId) {
    ast->type = ast->typeId->type;
  }
}

void TypeChecker::Visitor::operator()(SubscriptExpressionAST* ast) {
  if (!ast->baseExpression || control()->is_class(ast->baseExpression->type))
    return;

  if (!ast->indexExpression || control()->is_class(ast->indexExpression->type))
    return;

  // builtin subscript operator

  auto array_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                ExpressionAST*& index) {
    if (!control()->is_array(base->type)) return false;
    if (!control()->is_arithmetic_or_unscoped_enum(index->type)) return false;

    (void)temporary_materialization_conversion(base);
    (void)ensure_prvalue(index);
    adjust_cv(index);
    (void)integral_promotion(base);

    ast->type = control()->get_element_type(base->type);
    ast->valueCategory = base->valueCategory;
    return true;
  };

  auto pointer_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                  ExpressionAST*& index) {
    if (!control()->is_pointer(base->type)) return false;
    if (!control()->is_arithmetic_or_unscoped_enum(index->type)) return false;

    (void)ensure_prvalue(base);
    adjust_cv(base);

    (void)ensure_prvalue(index);
    adjust_cv(index);
    (void)integral_promotion(index);

    ast->type = control()->get_element_type(base->type);
    ast->valueCategory = ValueCategory::kLValue;
    return true;
  };

  if (array_subscript(ast, ast->baseExpression, ast->indexExpression)) return;
  if (array_subscript(ast, ast->indexExpression, ast->baseExpression)) return;
  if (pointer_subscript(ast, ast->baseExpression, ast->indexExpression)) return;
  if (pointer_subscript(ast, ast->indexExpression, ast->baseExpression)) return;

  error(ast->firstSourceLocation(),
        std::format("invalid subscript of type '{}' with index type '{}'",
                    to_string(ast->baseExpression->type),
                    to_string(ast->indexExpression->type)));
}

void TypeChecker::Visitor::operator()(CallExpressionAST* ast) {
  if (!ast->baseExpression) return;

  std::vector<const Type*> argumentTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    const Type* argumentType = nullptr;
    if (it->value) argumentType = it->value->type;

    argumentTypes.push_back(argumentType);
  }

  if (auto access = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    if (ast_cast<DestructorIdAST>(access->unqualifiedId)) {
      ast->type = control()->getVoidType();
      return;
    }
  }

  if (auto ovl = type_cast<OverloadSetType>(ast->baseExpression->type)) {
    // TODO: check overload set
    return;
  }

  auto functionType = type_cast<FunctionType>(ast->baseExpression->type);
  if (!functionType) {
    return;
  }

  ast->type = functionType->returnType();

  if (control()->is_lvalue_reference(ast->type)) {
    ast->type = control()->remove_reference(ast->type);
    ast->valueCategory = ValueCategory::kLValue;
  } else if (control()->is_rvalue_reference(ast->type)) {
    ast->type = control()->remove_reference(ast->type);
    ast->valueCategory = ValueCategory::kXValue;
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }

  if (ast->valueCategory == ValueCategory::kPrValue) {
    adjust_cv(ast);
  }
}

void TypeChecker::Visitor::operator()(TypeConstructionAST* ast) {}

void TypeChecker::Visitor::operator()(BracedTypeConstructionAST* ast) {}

void TypeChecker::Visitor::operator()(SpliceMemberExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(MemberExpressionAST* ast) {
  if (check_pseudo_destructor_access(ast)) return;
  if (check_member_access(ast)) return;
}

void TypeChecker::Visitor::operator()(PostIncrExpressionAST* ast) {
  if (control()->is_class(ast->baseExpression->type)) return;

  const std::string_view op =
      ast->op == TokenKind::T_PLUS_PLUS ? "increment" : "decrement";

  // builtin postfix increment operator
  if (!is_glvalue(ast->baseExpression)) {
    error(ast->opLoc, std::format("cannot {} an rvalue of type '{}'", op,
                                  to_string(ast->baseExpression->type)));
    return;
  }

  auto incr_arithmetic = [&]() {
    if (!control()->is_arithmetic(ast->baseExpression->type)) return false;
    auto ty = control()->remove_cv(ast->baseExpression->type);
    if (type_cast<BoolType>(ty)) return false;

    ast->type = ty;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto incr_pointer = [&]() {
    if (!control()->is_pointer(ast->baseExpression->type)) return false;
    auto ty = control()->remove_cv(ast->baseExpression->type);
    ast->type = ty;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  if (incr_arithmetic()) return;
  if (incr_pointer()) return;

  error(ast->opLoc, std::format("cannot {} a value of type '{}'", op,
                                to_string(ast->baseExpression->type)));
}

void TypeChecker::Visitor::operator()(CppCastExpressionAST* ast) {
  check_cpp_cast_expression(ast);

  switch (check.unit_->tokenKind(ast->castLoc)) {
    case TokenKind::T_STATIC_CAST:
      if (check_static_cast(ast)) break;
      error(
          ast->firstSourceLocation(),
          std::format("invalid static_cast of '{}' to '{}'",
                      to_string(ast->expression->type), to_string(ast->type)));
      break;

    default:
      break;
  }  // switch

  if (ast->valueCategory == ValueCategory::kPrValue) {
    adjust_cv(ast);
  }
}

void TypeChecker::Visitor::check_cpp_cast_expression(
    CppCastExpressionAST* ast) {
  if (!ast->typeId) {
    return;
  }

  ast->type = ast->typeId->type;

  if (auto refType = type_cast<LvalueReferenceType>(ast->type)) {
    ast->type = refType->elementType();
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  if (auto rvalueRefType = type_cast<RvalueReferenceType>(ast->type)) {
    ast->type = rvalueRefType->elementType();

    if (type_cast<FunctionType>(ast->type)) {
      ast->valueCategory = ValueCategory::kLValue;
    } else {
      ast->valueCategory = ValueCategory::kXValue;
    }
  }
}

auto TypeChecker::Visitor::check_static_cast(CppCastExpressionAST* ast)
    -> bool {
  if (!ast->typeId) return false;
  auto targetType = ast->typeId->type;

  if (control()->is_void(targetType)) return true;

  if (check_cast_to_derived(targetType, ast->expression)) return true;

  const auto cv1 = control()->get_cv_qualifiers(ast->expression->type);
  const auto cv2 = control()->get_cv_qualifiers(targetType);

  if (!check_cv_qualifiers(cv2, cv1)) return false;

  if (implicit_conversion(ast->expression, ast->type)) return true;

  auto source = ast->expression;
  (void)ensure_prvalue(source);
  adjust_cv(source);

  auto sourcePtr = type_cast<PointerType>(source->type);
  if (!sourcePtr) return false;

  if (!control()->is_void(sourcePtr->elementType())) return false;

  auto targetPtr = type_cast<PointerType>(targetType);
  if (!targetPtr) return false;

  if (!control()->is_object(targetPtr->elementType())) return false;

  ast->expression = source;

  return true;
}

auto TypeChecker::Visitor::check_cast_to_derived(const Type* targetType,
                                                 ExpressionAST* expression)
    -> bool {
  if (!is_lvalue(expression)) return false;

  auto sourceType = expression->type;

  CvQualifiers cv1 = CvQualifiers::kNone;
  if (auto qualType = type_cast<QualType>(sourceType)) {
    cv1 = qualType->cvQualifiers();
    sourceType = qualType->elementType();
  }

  auto targetRefType = type_cast<LvalueReferenceType>(targetType);
  if (!targetRefType) return false;

  targetType = targetRefType->elementType();

  CvQualifiers cv2 = CvQualifiers::kNone;
  if (auto qualType = type_cast<QualType>(targetType)) {
    cv2 = qualType->cvQualifiers();
    targetType = qualType->elementType();
  }

  if (!check_cv_qualifiers(cv2, cv1)) return false;

  if (!control()->is_base_of(sourceType, targetType)) return false;

  return true;
}

void TypeChecker::Visitor::operator()(BuiltinBitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BuiltinOffsetofExpressionAST* ast) {
  auto classType =
      ast->typeId ? type_cast<ClassType>(ast->typeId->type) : nullptr;
  auto id = ast_cast<IdExpressionAST>(ast->expression);

  if (classType && id && !id->nestedNameSpecifier) {
    auto symbol = classType->symbol();
    auto name = get_name(control(), id->unqualifiedId);
    auto member = Lookup{scope()}.qualifiedLookup(symbol->scope(), name);
    auto field = symbol_cast<FieldSymbol>(member);
    ast->symbol = field;
  }

  ast->type = control()->getSizeType();
}

void TypeChecker::Visitor::operator()(TypeidExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeidOfTypeExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SpliceExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(GlobalScopeReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(NamespaceReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeIdReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ReflectExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(LabelAddressExpressionAST* ast) {
  ast->type = control()->getPointerType(control()->getVoidType());
}

void TypeChecker::Visitor::operator()(UnaryExpressionAST* ast) {
  switch (ast->op) {
    case TokenKind::T_STAR: {
      auto pointerType = type_cast<PointerType>(ast->expression->type);
      if (pointerType) {
        (void)ensure_prvalue(ast->expression);
        adjust_cv(ast->expression);
        ast->type = pointerType->elementType();
        ast->valueCategory = ValueCategory::kLValue;
      }
      break;
    }

    case TokenKind::T_AMP_AMP: {
      cxx_runtime_error("address of label");
      ast->type = control()->getPointerType(control()->getVoidType());
      ast->valueCategory = ValueCategory::kPrValue;
      break;
    }

    case TokenKind::T_AMP: {
      if (!ast->expression->type) {
        break;
      }

      if (!is_glvalue(ast->expression)) {
        error(ast->opLoc,
              std::format("cannot take the address of an rvalue of type '{}'",
                          to_string(ast->expression->type)));
        break;
      }

      // TODO xvalue to lvalue

      if (auto idExpr = ast_cast<IdExpressionAST>(ast->expression);
          idExpr && idExpr->nestedNameSpecifier) {
        auto symbol = idExpr->symbol;
        if (auto field = symbol_cast<FieldSymbol>(symbol);
            field && !field->isStatic()) {
          auto parentClass = field->enclosingSymbol();
          auto classType = type_cast<ClassType>(parentClass->type());

          ast->type =
              control()->getMemberObjectPointerType(classType, field->type());

          ast->valueCategory = ValueCategory::kPrValue;

          break;
        }

        if (auto function = symbol_cast<FunctionSymbol>(symbol);
            function && !function->isStatic()) {
          auto functionType = type_cast<FunctionType>(function->type());
          auto parentClass = function->enclosingSymbol();
          auto classType = type_cast<ClassType>(parentClass->type());

          ast->type =
              control()->getMemberFunctionPointerType(classType, functionType);

          ast->valueCategory = ValueCategory::kPrValue;

          break;
        }
      }  // id expression

      ast->type = control()->getPointerType(ast->expression->type);
      ast->valueCategory = ValueCategory::kPrValue;
      break;
    }

    case TokenKind::T_PLUS: {
      ExpressionAST* expr = ast->expression;
      (void)ensure_prvalue(expr);
      adjust_cv(expr);
      if (control()->is_arithmetic_or_unscoped_enum(expr->type) ||
          control()->is_pointer(expr->type)) {
        if (control()->is_integral_or_unscoped_enum(expr->type)) {
          (void)integral_promotion(expr);
        }
        ast->expression = expr;
        ast->type = expr->type;
        ast->valueCategory = ValueCategory::kPrValue;
      }
      break;
    }

    case TokenKind::T_MINUS: {
      ExpressionAST* expr = ast->expression;
      (void)ensure_prvalue(expr);
      adjust_cv(expr);
      if (control()->is_arithmetic_or_unscoped_enum(expr->type)) {
        if (control()->is_integral_or_unscoped_enum(expr->type)) {
          (void)integral_promotion(expr);
        }
        ast->expression = expr;
        ast->type = expr->type;
        ast->valueCategory = ValueCategory::kPrValue;
      }
      break;
    }

    case TokenKind::T_EXCLAIM: {
      (void)implicit_conversion(ast->expression, control()->getBoolType());
      ast->type = control()->getBoolType();
      ast->valueCategory = ValueCategory::kPrValue;
      break;
    }

    case TokenKind::T_TILDE: {
      ExpressionAST* expr = ast->expression;
      (void)ensure_prvalue(expr);
      adjust_cv(expr);
      if (control()->is_integral_or_unscoped_enum(expr->type)) {
        (void)integral_promotion(expr);
        ast->expression = expr;
        ast->type = expr->type;
        ast->valueCategory = ValueCategory::kPrValue;
      }
      break;
    }

    case TokenKind::T_PLUS_PLUS: {
      if (!is_glvalue(ast->expression)) {
        error(ast->opLoc, std::format("cannot increment an rvalue of type '{}'",
                                      to_string(ast->expression->type)));
        break;
      }

      auto ty = ast->expression->type;

      if (control()->is_arithmetic(ty) && !control()->is_const(ty)) {
        ast->type = ty;
        ast->valueCategory = ValueCategory::kLValue;
        break;
      }

      if (auto ptrTy = type_cast<PointerType>(ty)) {
        if (!control()->is_void(ptrTy->elementType())) {
          ast->type = ptrTy;
          ast->valueCategory = ValueCategory::kLValue;
          break;
        }
      }

      error(ast->opLoc, std::format("cannot increment a value of type '{}'",
                                    to_string(ast->expression->type)));
      break;
    }

    case TokenKind::T_MINUS_MINUS: {
      if (!is_glvalue(ast->expression)) {
        error(ast->opLoc, std::format("cannot decrement an rvalue of type '{}'",
                                      to_string(ast->expression->type)));
        break;
      }

      auto ty = ast->expression->type;

      if (control()->is_arithmetic(ty) && !control()->is_const(ty)) {
        ast->type = ty;
        ast->valueCategory = ValueCategory::kLValue;
        break;
      }

      if (auto ptrTy = type_cast<PointerType>(ty)) {
        if (ptrTy && !control()->is_void(ptrTy->elementType())) {
          ast->type = ptrTy;
          ast->valueCategory = ValueCategory::kLValue;
          break;
        }
      }

      error(ast->opLoc, std::format("cannot decrement a value of type '{}'",
                                    to_string(ast->expression->type)));
      break;
    }

    default:
      break;
  }  // switch
}

void TypeChecker::Visitor::operator()(AwaitExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SizeofExpressionAST* ast) {
  ast->type = control()->getSizeType();
}

void TypeChecker::Visitor::operator()(SizeofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();
}

void TypeChecker::Visitor::operator()(SizeofPackExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(AlignofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();
}

void TypeChecker::Visitor::operator()(AlignofExpressionAST* ast) {
  ast->type = control()->getSizeType();
}

void TypeChecker::Visitor::operator()(NoexceptExpressionAST* ast) {
  ast->type = control()->getBoolType();
}

void TypeChecker::Visitor::operator()(NewExpressionAST* ast) {
  // TODO: decay
  auto objectType = control()->remove_reference(ast->objectType);

  if (auto arrayType = type_cast<BoundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(arrayType->elementType());
  } else if (auto unboundedType =
                 type_cast<UnboundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(unboundedType->elementType());
  } else {
    ast->type = control()->getPointerType(ast->objectType);
  }

  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(DeleteExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(CastExpressionAST* ast) {
  if (ast->typeId) {
    ast->type = control()->remove_reference(ast->typeId->type);
    if (control()->is_lvalue_reference(ast->typeId->type))
      ast->valueCategory = ValueCategory::kLValue;
    else if (control()->is_rvalue_reference(ast->typeId->type))
      ast->valueCategory = ValueCategory::kXValue;
    else {
      ast->valueCategory = ValueCategory::kPrValue;
      adjust_cv(ast);
    }
  }
}

void TypeChecker::Visitor::operator()(ImplicitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BinaryExpressionAST* ast) {
  if (!ast->leftExpression) return;
  if (!ast->rightExpression) return;

  switch (ast->op) {
    case TokenKind::T_DOT_STAR:
      break;

    case TokenKind::T_MINUS_GREATER_STAR:
      break;

    case TokenKind::T_STAR:
      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);
      break;

    case TokenKind::T_SLASH:
      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);
      break;

    case TokenKind::T_PLUS:
      check_addition(ast);
      break;

    case TokenKind::T_MINUS:
      check_subtraction(ast);
      break;

    case TokenKind::T_PERCENT:
      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);

      break;

    case TokenKind::T_LESS_LESS:
    case TokenKind::T_GREATER_GREATER:
      (void)usual_arithmetic_conversion(ast->leftExpression,
                                        ast->rightExpression);
      ast->type = ast->leftExpression->type;
      break;

    case TokenKind::T_LESS_EQUAL_GREATER:
      (void)usual_arithmetic_conversion(ast->leftExpression,
                                        ast->rightExpression);
      ast->type = control()->getIntType();
      break;

    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_GREATER:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
      (void)usual_arithmetic_conversion(ast->leftExpression,
                                        ast->rightExpression);
      ast->type = control()->getBoolType();
      break;

    case TokenKind::T_AMP:
    case TokenKind::T_CARET:
    case TokenKind::T_BAR:
      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);

      break;

    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
      (void)implicit_conversion(ast->leftExpression, control()->getBoolType());

      (void)implicit_conversion(ast->rightExpression, control()->getBoolType());

      ast->type = control()->getBoolType();
      break;

    case TokenKind::T_COMMA:
      if (ast->rightExpression) {
        ast->type = ast->rightExpression->type;
        ast->valueCategory = ast->rightExpression->valueCategory;
      }
      break;

    default:
      cxx_runtime_error(
          std::format("invalid operator '{}'", Token::spell(ast->op)));
  }  // switch
}

void TypeChecker::Visitor::operator()(ConditionalExpressionAST* ast) {
  auto check_void_type = [&] {
    if (!control()->is_void(ast->iftrueExpression->type) &&
        !control()->is_void(ast->iffalseExpression->type))
      return false;

    // one of the two expressions is void
    if (ast_cast<ThrowExpressionAST>(
            strip_parentheses(ast->iftrueExpression))) {
      ast->type = ast->iffalseExpression->type;
      ast->valueCategory = ast->iffalseExpression->valueCategory;
      return true;
    }

    if (ast_cast<ThrowExpressionAST>(
            strip_parentheses(ast->iffalseExpression))) {
      ast->type = ast->iftrueExpression->type;
      ast->valueCategory = ast->iftrueExpression->valueCategory;
      return true;
    }

    if (!control()->is_same(ast->iftrueExpression->type,
                            ast->iffalseExpression->type)) {
      error(ast->questionLoc,
            std::format(
                "left operand to ? is '{}', but right operand is of type '{}'",
                to_string(ast->iftrueExpression->type),
                to_string(ast->iffalseExpression->type)));
    }

    ast->type = control()->getVoidType();
    ast->valueCategory = ValueCategory::kPrValue;

    return true;
  };

  auto check_same_type_and_value_category = [&] {
    if (ast->iftrueExpression->valueCategory !=
        ast->iffalseExpression->valueCategory) {
      return false;
    }

    if (!control()->is_same(control()->remove_cv(ast->iftrueExpression->type),
                            control()->remove_cv(ast->iffalseExpression->type)))
      return false;

    ast->valueCategory = ast->iftrueExpression->valueCategory;
    ast->type = ast->iftrueExpression->type;

    return true;
  };

  auto check_arith_types = [&] {
    if (!control()->is_arithmetic_or_unscoped_enum(ast->iftrueExpression->type))
      return false;
    if (!control()->is_arithmetic_or_unscoped_enum(
            ast->iffalseExpression->type))
      return false;

    ast->type = usual_arithmetic_conversion(ast->iftrueExpression,
                                            ast->iffalseExpression);

    if (!ast->type) return false;

    ast->valueCategory = ValueCategory::kPrValue;

    return true;
  };

  auto check_same_types = [&] {
    if (!control()->is_same(ast->iftrueExpression->type,
                            ast->iffalseExpression->type))
      return false;

    ast->type = ast->iftrueExpression->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto check_compatible_pointers = [&] {
    if (!control()->is_pointer(ast->iftrueExpression->type) &&
        !control()->is_pointer(ast->iffalseExpression->type))
      return false;

    ast->type =
        composite_pointer_type(ast->iftrueExpression, ast->iffalseExpression);

    ast->valueCategory = ValueCategory::kPrValue;

    if (!ast->type) return false;

    return true;
  };

  if (ast->iftrueExpression && ast->iffalseExpression) {
    if (check_void_type()) return;
    if (check_same_type_and_value_category()) return;

    (void)array_to_pointer_conversion(ast->iftrueExpression);
    (void)function_to_pointer_conversion(ast->iftrueExpression);

    (void)array_to_pointer_conversion(ast->iffalseExpression);
    (void)function_to_pointer_conversion(ast->iffalseExpression);

    if (check_arith_types()) return;
    if (check_same_types()) return;
    if (check_compatible_pointers()) return;
  }

  if (!ast->type) {
    auto iftrueType =
        ast->iftrueExpression ? ast->iftrueExpression->type : nullptr;

    auto iffalseType =
        ast->iffalseExpression ? ast->iffalseExpression->type : nullptr;

    error(ast->questionLoc,
          std::format(
              "left operand to ? is '{}', but right operand is of type '{}'",
              to_string(iftrueType), to_string(iffalseType)));
  }
}

void TypeChecker::Visitor::operator()(YieldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ThrowExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(AssignmentExpressionAST* ast) {
  if (!ast->leftExpression) return;
  if (!ast->rightExpression) return;

  ast->type = ast->leftExpression->type;
  ast->valueCategory = ast->leftExpression->valueCategory;

  (void)implicit_conversion(ast->rightExpression, ast->type);
}

void TypeChecker::Visitor::operator()(PackExpansionExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(DesignatedInitializerClauseAST* ast) {}

void TypeChecker::Visitor::operator()(TypeTraitExpressionAST* ast) {
  ast->type = control()->getBoolType();
}

void TypeChecker::Visitor::operator()(ConditionExpressionAST* ast) {
  ast->type = control()->getBoolType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(EqualInitializerAST* ast) {
  if (!ast->expression) return;
  ast->type = ast->expression->type;
  ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::operator()(BracedInitListAST* ast) {}

void TypeChecker::Visitor::operator()(ParenInitializerAST* ast) {}

auto TypeChecker::Visitor::strip_parentheses(ExpressionAST* ast)
    -> ExpressionAST* {
  while (auto paren = ast_cast<NestedExpressionAST>(ast)) {
    ast = paren->expression;
  }
  return ast;
}

auto TypeChecker::Visitor::strip_cv(const Type*& type) -> CvQualifiers {
  if (auto qualType = type_cast<QualType>(type)) {
    auto cv = qualType->cvQualifiers();
    type = qualType->elementType();
    return cv;
  }
  return {};
}

auto TypeChecker::Visitor::is_const(CvQualifiers cv) const -> bool {
  return cv == CvQualifiers::kConst || cv == CvQualifiers::kConstVolatile;
}

auto TypeChecker::Visitor::is_volatile(CvQualifiers cv) const -> bool {
  return cv == CvQualifiers::kVolatile || cv == CvQualifiers::kConstVolatile;
}

auto TypeChecker::Visitor::merge_cv(CvQualifiers cv1, CvQualifiers cv2) const
    -> CvQualifiers {
  CvQualifiers cv = CvQualifiers::kNone;
  if (is_const(cv1) || is_const(cv2)) cv = CvQualifiers::kConst;
  if (is_volatile(cv1) || is_volatile(cv2)) {
    if (cv == CvQualifiers::kConst)
      cv = CvQualifiers::kConstVolatile;
    else
      cv = CvQualifiers::kVolatile;
  }
  return cv;
}

auto TypeChecker::Visitor::lvalue_to_rvalue_conversion(ExpressionAST*& expr)
    -> bool {
  if (!is_glvalue(expr)) return false;

  if (control()->is_function(expr->type)) return false;
  if (control()->is_array(expr->type)) return false;
  if (!control()->is_complete(expr->type)) return false;
  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
  cast->expression = expr;
  cast->type = control()->remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto TypeChecker::Visitor::array_to_pointer_conversion(ExpressionAST*& expr)
    -> bool {
  auto unref = control()->remove_reference(expr->type);
  if (!control()->is_array(unref)) return false;
  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kArrayToPointerConversion;
  cast->expression = expr;
  cast->type = control()->add_pointer(control()->remove_extent(unref));
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto TypeChecker::Visitor::function_to_pointer_conversion(ExpressionAST*& expr)
    -> bool {
  auto unref = control()->remove_reference(expr->type);
  if (!control()->is_function(unref)) return false;
  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
  cast->expression = expr;
  cast->type = control()->add_pointer(unref);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto TypeChecker::Visitor::integral_promotion(ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_integral(expr->type) && !control()->is_enum(expr->type))
    return false;

  auto make_implicit_cast = [&](const Type* type) {
    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kIntegralPromotion;
    cast->expression = expr;
    cast->type = type;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  // TODO: bit-fields

  switch (expr->type->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt: {
      make_implicit_cast(control()->getIntType());
      return true;
    }

    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar: {
      make_implicit_cast(control()->getIntType());
      return true;
    }

    case TypeKind::kBool: {
      make_implicit_cast(control()->getIntType());
      return true;
    }

    default:
      break;
  }  // switch

  if (auto enumType = type_cast<EnumType>(expr->type)) {
    auto type = enumType->underlyingType();

    if (!type) {
      // TODO: compute the from the value of the enumeration
      type = control()->getIntType();
    }

    make_implicit_cast(type);

    return true;
  }

  return false;
}

auto TypeChecker::Visitor::floating_point_promotion(ExpressionAST*& expr)
    -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_floating_point(expr->type)) return false;

  if (expr->type->kind() != TypeKind::kFloat) return false;

  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kFloatingPointPromotion;
  cast->expression = expr;
  cast->type = control()->getDoubleType();
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::integral_conversion(ExpressionAST*& expr,
                                               const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_integral_or_unscoped_enum(expr->type)) return false;
  if (!control()->is_integer(destinationType)) return false;

  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kIntegralConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::floating_point_conversion(
    ExpressionAST*& expr, const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_floating_point(expr->type)) return false;
  if (!control()->is_floating_point(destinationType)) return false;

  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kFloatingPointConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::floating_integral_conversion(
    ExpressionAST*& expr, const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto make_integral_conversion = [&] {
    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kFloatingIntegralConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  if (control()->is_integral_or_unscoped_enum(expr->type) &&
      control()->is_floating_point(destinationType)) {
    make_integral_conversion();
    return true;
  }

  if (!control()->is_floating_point(expr->type)) return false;
  if (!control()->is_integer(destinationType)) return false;

  make_integral_conversion();

  return true;
}

auto TypeChecker::Visitor::pointer_to_member_conversion(
    ExpressionAST*& expr, const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_member_pointer(destinationType)) return false;

  auto make_implicit_cast = [&] {
    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kPointerToMemberConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  auto can_convert_null_pointer_constant = [&] {
    if (!is_null_pointer_constant(expr)) return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_member_object_pointer = [&] {
    auto memberObjectPointerType =
        type_cast<MemberObjectPointerType>(expr->type);

    if (!memberObjectPointerType) return false;

    auto destinationMemberObjectPointerType =
        type_cast<MemberObjectPointerType>(destinationType);

    if (!destinationMemberObjectPointerType) return false;

    if (control()->get_cv_qualifiers(memberObjectPointerType->elementType()) !=
        control()->get_cv_qualifiers(
            destinationMemberObjectPointerType->elementType()))
      return false;

    if (!control()->is_base_of(destinationMemberObjectPointerType->classType(),
                               memberObjectPointerType->classType()))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_member_function_pointer = [&] {
    auto memberFunctionPointerType =
        type_cast<MemberFunctionPointerType>(expr->type);

    if (!memberFunctionPointerType) return false;

    auto destinationMemberFunctionPointerType =
        type_cast<MemberFunctionPointerType>(destinationType);

    if (!destinationMemberFunctionPointerType) return false;

    if (control()->get_cv_qualifiers(
            memberFunctionPointerType->functionType()) !=
        control()->get_cv_qualifiers(
            destinationMemberFunctionPointerType->functionType()))
      return false;

    if (!control()->is_base_of(
            destinationMemberFunctionPointerType->classType(),
            memberFunctionPointerType->classType()))
      return false;

    make_implicit_cast();

    return true;
  };

  if (can_convert_null_pointer_constant()) return true;
  if (can_convert_member_object_pointer()) return true;

  return false;
}

auto TypeChecker::Visitor::pointer_conversion(ExpressionAST*& expr,
                                              const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  auto make_implicit_cast = [&] {
    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  auto can_convert_null_pointer_literal = [&] {
    if (!is_null_pointer_constant(expr)) return false;

    if (!control()->is_pointer(destinationType) &&
        !control()->is_null_pointer(destinationType))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_to_void_pointer = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    if (control()->get_cv_qualifiers(pointerType->elementType()) !=
        control()->get_cv_qualifiers(destinationPointerType->elementType()))
      return false;

    if (!control()->is_void(destinationPointerType->elementType()))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_to_base_class_pointer = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    if (control()->get_cv_qualifiers(pointerType->elementType()) !=
        control()->get_cv_qualifiers(destinationPointerType->elementType()))
      return false;

    if (!control()->is_base_of(
            control()->remove_cv(destinationPointerType->elementType()),
            control()->remove_cv(pointerType->elementType())))
      return false;

    make_implicit_cast();

    return true;
  };

  if (can_convert_null_pointer_literal()) return true;
  if (can_convert_to_void_pointer()) return true;
  if (can_convert_to_base_class_pointer()) return true;

  return false;
}

auto TypeChecker::Visitor::function_pointer_conversion(
    ExpressionAST*& expr, const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto can_remove_noexcept_from_function = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto functionType =
        type_cast<FunctionType>(pointerType->elementType());

    if (!functionType) return false;

    if (functionType->isNoexcept()) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    const auto destinationFunctionType =
        type_cast<FunctionType>(destinationPointerType->elementType());

    if (!destinationFunctionType) return false;

    if (!control()->is_same(control()->remove_noexcept(functionType),
                            destinationFunctionType))
      return false;

    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kFunctionPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;

    return true;
  };

  auto can_remove_noexcept_from_member_function__pointer = [&] {
    const auto memberFunctionPointer =
        type_cast<MemberFunctionPointerType>(expr->type);

    if (!memberFunctionPointer) return false;

    if (!memberFunctionPointer->functionType()->isNoexcept()) return false;

    const auto destinationMemberFunctionPointer =
        type_cast<MemberFunctionPointerType>(destinationType);

    if (!destinationMemberFunctionPointer) return false;

    if (destinationMemberFunctionPointer->functionType()->isNoexcept())
      return false;

    if (!control()->is_same(
            control()->remove_noexcept(memberFunctionPointer->functionType()),
            destinationMemberFunctionPointer->functionType()))
      return false;

    auto cast = make_node<ImplicitCastExpressionAST>(arena());
    cast->castKind = ImplicitCastKind::kFunctionPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;

    return true;
  };

  if (can_remove_noexcept_from_function()) return true;
  if (can_remove_noexcept_from_member_function__pointer()) return true;

  return false;
}

auto TypeChecker::Visitor::boolean_conversion(ExpressionAST*& expr,
                                              const Type* destinationType)
    -> bool {
  if (!type_cast<BoolType>(destinationType)) return false;

  if (!is_prvalue(expr)) return false;

  if (!control()->is_arithmetic_or_unscoped_enum(expr->type) &&
      !control()->is_pointer(expr->type) &&
      !control()->is_member_pointer(expr->type))
    return false;

  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kBooleanConversion;
  cast->expression = expr;
  cast->type = control()->getBoolType();
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::temporary_materialization_conversion(
    ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;

  auto cast = make_node<ImplicitCastExpressionAST>(arena());
  cast->castKind = ImplicitCastKind::kTemporaryMaterializationConversion;
  cast->expression = expr;
  cast->type = control()->remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kXValue;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::qualification_conversion(ExpressionAST*& expr,
                                                    const Type* destinationType)
    -> bool {
  return false;
}

auto TypeChecker::Visitor::ensure_prvalue(ExpressionAST*& expr) -> bool {
  if (lvalue_to_rvalue_conversion(expr)) return true;
  if (array_to_pointer_conversion(expr)) return true;
  if (function_to_pointer_conversion(expr)) return true;

  return false;
}

void TypeChecker::Visitor::adjust_cv(ExpressionAST* expr) {
  if (!is_prvalue(expr)) return;

  auto qualType = type_cast<QualType>(expr->type);
  if (!qualType) return;

  if (control()->is_class(expr->type) || control()->is_array(expr->type))
    return;

  expr->type = qualType->elementType();
}

auto TypeChecker::Visitor::implicit_conversion(ExpressionAST*& expr,
                                               const Type* destinationType)
    -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  if (control()->is_same(expr->type, destinationType)) return true;

  auto savedValueCategory = expr->valueCategory;
  auto savedExpr = expr;
  auto didConvert = ensure_prvalue(expr);

  adjust_cv(expr);

  if (integral_promotion(expr)) return true;
  if (floating_point_promotion(expr)) return true;
  if (integral_conversion(expr, destinationType)) return true;
  if (floating_point_conversion(expr, destinationType)) return true;
  if (floating_integral_conversion(expr, destinationType)) return true;
  if (pointer_conversion(expr, destinationType)) return true;
  if (pointer_to_member_conversion(expr, destinationType)) return true;
  if (boolean_conversion(expr, destinationType)) return true;
  if (function_pointer_conversion(expr, destinationType)) return true;
  if (qualification_conversion(expr, destinationType)) return true;

  if (didConvert) return true;

  expr = savedExpr;
  expr->valueCategory = savedValueCategory;

  return false;
}

auto TypeChecker::Visitor::usual_arithmetic_conversion(ExpressionAST*& expr,
                                                       ExpressionAST*& other)
    -> const Type* {
  if (!control()->is_arithmetic(expr->type) && !control()->is_enum(expr->type))
    return nullptr;

  if (!control()->is_arithmetic(other->type) &&
      !control()->is_enum(other->type))
    return nullptr;

  (void)lvalue_to_rvalue_conversion(expr);
  adjust_cv(expr);

  (void)lvalue_to_rvalue_conversion(other);
  adjust_cv(other);

  ExpressionAST* savedExpr = expr;
  ExpressionAST* savedOther = other;

  auto unmodifiedExpressions = [&]() -> const Type* {
    expr = savedExpr;
    other = savedOther;
    return nullptr;
  };

  if (control()->is_scoped_enum(expr->type) ||
      control()->is_scoped_enum(other->type))
    return unmodifiedExpressions();

  if (control()->is_floating_point(expr->type) ||
      control()->is_floating_point(other->type)) {
    if (control()->is_same(expr->type, other->type)) return expr->type;

    if (!control()->is_floating_point(expr->type)) {
      if (floating_integral_conversion(expr, other->type)) return other->type;
      return unmodifiedExpressions();
    }

    if (!control()->is_floating_point(other->type)) {
      if (floating_integral_conversion(other, expr->type)) return expr->type;
      return unmodifiedExpressions();
    }

    if (expr->type->kind() == TypeKind::kLongDouble ||
        other->type->kind() == TypeKind::kLongDouble) {
      (void)floating_point_conversion(expr, control()->getLongDoubleType());
      return control()->getLongDoubleType();
    }

    if (expr->type->kind() == TypeKind::kDouble ||
        other->type->kind() == TypeKind::kDouble) {
      (void)floating_point_conversion(expr, control()->getDoubleType());
      return control()->getDoubleType();
    }

    return unmodifiedExpressions();
  }

  (void)integral_promotion(expr);
  (void)integral_promotion(other);

  if (control()->is_same(expr->type, other->type)) return expr->type;

  auto match_integral_type = [&](const Type* type) -> bool {
    if (expr->type->kind() == type->kind() ||
        other->type->kind() == type->kind()) {
      (void)integral_conversion(expr, type);
      (void)integral_conversion(other, type);
      return true;
    }
    return false;
  };

  if (control()->is_signed(expr->type) && control()->is_signed(other->type)) {
    if (match_integral_type(control()->getLongLongIntType())) {
      return control()->getLongLongIntType();
    }

    if (match_integral_type(control()->getLongIntType())) {
      return control()->getLongIntType();
    }

    (void)integral_conversion(expr, control()->getIntType());
    (void)integral_conversion(other, control()->getIntType());
    return control()->getIntType();
  }

  if (control()->is_unsigned(expr->type) &&
      control()->is_unsigned(other->type)) {
    if (match_integral_type(control()->getUnsignedLongLongIntType())) {
      return control()->getUnsignedLongLongIntType();
    }

    if (match_integral_type(control()->getUnsignedLongIntType())) {
      return control()->getUnsignedLongIntType();
    }

    (void)integral_conversion(expr, control()->getUnsignedIntType());
    return control()->getUnsignedIntType();
  }

  if (match_integral_type(control()->getUnsignedLongLongIntType())) {
    return control()->getUnsignedLongLongIntType();
  }

  if (match_integral_type(control()->getUnsignedLongIntType())) {
    return control()->getUnsignedLongIntType();
  }

  if (match_integral_type(control()->getUnsignedIntType())) {
    return control()->getUnsignedIntType();
  }

  if (match_integral_type(control()->getUnsignedShortIntType())) {
    return control()->getUnsignedShortIntType();
  }

  if (match_integral_type(control()->getUnsignedCharType())) {
    return control()->getUnsignedCharType();
  }

  if (match_integral_type(control()->getLongLongIntType())) {
    return control()->getLongLongIntType();
  }

  if (match_integral_type(control()->getLongIntType())) {
    return control()->getLongIntType();
  }

  (void)integral_conversion(expr, control()->getIntType());
  (void)integral_conversion(other, control()->getIntType());
  return control()->getIntType();
}

auto TypeChecker::Visitor::get_qualification_combined_type(const Type* left,
                                                           const Type* right)
    -> const Type* {
  bool didChangeTypeOrQualifiers = false;

  auto type =
      get_qualification_combined_type(left, right, didChangeTypeOrQualifiers);

  return type;
}

auto TypeChecker::Visitor::get_qualification_combined_type(
    const Type* left, const Type* right, bool& didChangeTypeOrQualifiers)
    -> const Type* {
  auto check_inputs = [&] {
    if (control()->is_pointer(left) && control()->is_pointer(right)) {
      return true;
    }

    if (control()->is_array(left) && control()->is_array(right)) {
      return true;
    }

    return false;
  };

  auto cv1 = strip_cv(left);
  auto cv2 = strip_cv(right);

  if (!check_inputs()) {
    const auto cv3 = merge_cv(cv1, cv2);

    if (control()->is_same(left, right)) {
      return control()->add_cv(left, cv3);
    }

    if (control()->is_base_of(left, right)) {
      return control()->add_cv(left, cv1);
    }

    if (control()->is_base_of(right, left)) {
      return control()->add_cv(right, cv2);
    }

    return nullptr;
  }

  auto leftElementType = control()->get_element_type(left);
  if (control()->is_array(leftElementType)) {
    cv1 = merge_cv(cv1, control()->get_cv_qualifiers(leftElementType));
  }

  auto rightElementType = control()->get_element_type(right);
  if (control()->is_array(rightElementType)) {
    cv2 = merge_cv(cv2, control()->get_cv_qualifiers(rightElementType));
  }

  auto elementType = get_qualification_combined_type(
      leftElementType, rightElementType, didChangeTypeOrQualifiers);

  if (!elementType) {
    return nullptr;
  }

  auto cv3 = merge_cv(cv1, cv2);

  if (didChangeTypeOrQualifiers) cv3 = cv3 | CvQualifiers::kConst;

  if (cv1 != cv3 || cv2 != cv3) didChangeTypeOrQualifiers = true;

  elementType = control()->add_cv(elementType, cv3);

  if (control()->is_array(left) && control()->is_array(right)) {
    auto leftArrayType = type_cast<BoundedArrayType>(left);
    auto rightArrayType = type_cast<BoundedArrayType>(right);

    if (leftArrayType && rightArrayType) {
      if (leftArrayType->size() != rightArrayType->size()) return nullptr;
      return control()->getBoundedArrayType(elementType, leftArrayType->size());
    }

    if (leftArrayType || rightArrayType) {
      // one of arrays is unbounded
      didChangeTypeOrQualifiers = true;
    }

    return control()->getUnboundedArrayType(elementType);
  }

  return control()->getPointerType(elementType);
}

auto TypeChecker::Visitor::composite_pointer_type(ExpressionAST*& expr,
                                                  ExpressionAST*& other)
    -> const Type* {
  if (control()->is_null_pointer(expr->type) &&
      control()->is_null_pointer(other->type))
    return control()->getNullptrType();

  if (is_null_pointer_constant(expr)) return other->type;
  if (is_null_pointer_constant(other)) return expr->type;

  auto is_pointer_to_cv_void = [this](const Type* type) {
    if (!control()->is_pointer(type)) return false;
    if (!control()->is_void(control()->get_element_type(type))) return false;
    return true;
  };

  if (control()->is_pointer(expr->type) && control()->is_pointer(other->type)) {
    auto t1 = control()->get_element_type(expr->type);
    const auto cv1 = strip_cv(t1);

    auto t2 = control()->get_element_type(other->type);
    const auto cv2 = strip_cv(t2);

    if (control()->is_void(t1)) {
      return control()->getPointerType(control()->add_cv(t1, cv2));
    }

    if (control()->is_void(t2)) {
      return control()->getPointerType(control()->add_cv(t2, cv1));
    }

    if (auto type = get_qualification_combined_type(expr->type, other->type)) {
      return type;
    }

    // TODO: check for noexcept function pointers
  }

  return nullptr;
}

auto TypeChecker::Visitor::is_null_pointer_constant(ExpressionAST* expr) const
    -> bool {
  if (control()->is_null_pointer(expr->type)) return true;
  if (auto integerLiteral = ast_cast<IntLiteralExpressionAST>(expr)) {
    return integerLiteral->literal->value() == "0";
  }
  return false;
}

auto TypeChecker::Visitor::check_cv_qualifiers(CvQualifiers target,
                                               CvQualifiers source) const
    -> bool {
  if (source == target) return true;
  if (source == CvQualifiers::kNone) return true;
  if (target == CvQualifiers::kConstVolatile) return true;
  return false;
}

TypeChecker::TypeChecker(TranslationUnit* unit) : unit_(unit) {}

void TypeChecker::operator()(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

void TypeChecker::check(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

void TypeChecker::Visitor::check_addition(BinaryExpressionAST* ast) {
  // ### TODO: check for user-defined conversion operators
  if (control()->is_class(ast->leftExpression->type)) return;
  if (control()->is_class(ast->rightExpression->type)) return;

  if (auto ty = usual_arithmetic_conversion(ast->leftExpression,
                                            ast->rightExpression)) {
    ast->type = ty;
    return;
  }

  (void)ensure_prvalue(ast->leftExpression);
  adjust_cv(ast->leftExpression);

  (void)ensure_prvalue(ast->rightExpression);
  adjust_cv(ast->rightExpression);

  const auto left_is_pointer = control()->is_pointer(ast->leftExpression->type);

  const auto right_is_pointer =
      control()->is_pointer(ast->rightExpression->type);

  const auto left_is_integral =
      control()->is_integral_or_unscoped_enum(ast->leftExpression->type);

  const auto right_is_integral =
      control()->is_integral_or_unscoped_enum(ast->rightExpression->type);

  if (left_is_pointer && right_is_integral) {
    (void)integral_promotion(ast->rightExpression);
    ast->type = ast->leftExpression->type;
    return;
  }

  if (right_is_pointer && left_is_integral) {
    (void)integral_promotion(ast->leftExpression);
    ast->type = ast->rightExpression->type;
    return;
  }

  error(ast->opLoc,
        std::format(
            "invalid operands of types '{}' and '{}' to binary operator '+'",
            to_string(ast->leftExpression->type),
            to_string(ast->rightExpression->type)));
}

void TypeChecker::Visitor::check_subtraction(BinaryExpressionAST* ast) {
  // ### TODO: check for user-defined conversion operators
  if (control()->is_class(ast->leftExpression->type)) return;
  if (control()->is_class(ast->rightExpression->type)) return;

  if (auto ty = usual_arithmetic_conversion(ast->leftExpression,
                                            ast->rightExpression)) {
    ast->type = ty;
    return;
  }

  (void)ensure_prvalue(ast->leftExpression);
  adjust_cv(ast->leftExpression);

  (void)ensure_prvalue(ast->rightExpression);
  adjust_cv(ast->rightExpression);

  auto check_operand_types = [&]() {
    if (!control()->is_pointer(ast->leftExpression->type)) return false;

    if (!control()->is_arithmetic_or_unscoped_enum(
            ast->rightExpression->type) &&
        !control()->is_pointer(ast->rightExpression->type))
      return false;

    return true;
  };

  if (!check_operand_types()) {
    error(ast->opLoc,
          std::format("invalid operands to binary expression '{}' and '{}'",
                      to_string(ast->leftExpression->type),
                      to_string(ast->rightExpression->type)));
    return;
  }

  if (control()->is_pointer(ast->rightExpression->type)) {
    auto leftElementType =
        control()->get_element_type(ast->leftExpression->type);
    (void)strip_cv(leftElementType);

    auto rightElementType =
        control()->get_element_type(ast->rightExpression->type);
    (void)strip_cv(rightElementType);

    if (control()->is_same(leftElementType, rightElementType)) {
      ast->type = control()->getLongIntType();  // TODO: ptrdiff_t
    } else {
      error(ast->opLoc,
            std::format("'{}' and '{}' are not pointers to compatible types",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));
    }

    return;
  }

  (void)integral_promotion(ast->rightExpression);
  ast->type = ast->leftExpression->type;
}

auto TypeChecker::Visitor::check_member_access(MemberExpressionAST* ast)
    -> bool {
  const Type* objectType = ast->baseExpression->type;
  auto cv1 = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointerType = type_cast<PointerType>(objectType);
    if (!pointerType) return false;

    objectType = pointerType->elementType();
    cv1 = strip_cv(objectType);
  }

  auto classType = type_cast<ClassType>(objectType);
  if (!classType) return false;

  auto memberName = get_name(control(), ast->unqualifiedId);

  auto classSymbol = classType->symbol();

  auto symbol =
      Lookup{scope()}.qualifiedLookup(classSymbol->scope(), memberName);

  ast->symbol = symbol;

  if (symbol) {
    ast->type = symbol->type();

    if (symbol->isEnumerator()) {
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      if (is_lvalue(ast->baseExpression)) {
        ast->valueCategory = ValueCategory::kLValue;
      } else {
        ast->valueCategory = ValueCategory::kXValue;
      }

      if (auto field = symbol_cast<FieldSymbol>(symbol);
          field && !field->isStatic()) {
        auto cv2 = strip_cv(ast->type);

        if (is_volatile(cv1) || is_volatile(cv2))
          ast->type = control()->add_volatile(ast->type);

        if (!field->isMutable() && (is_const(cv1) || is_const(cv2)))
          ast->type = control()->add_const(ast->type);
      }
    }
  }

  return true;
}

auto TypeChecker::Visitor::check_pseudo_destructor_access(
    MemberExpressionAST* ast) -> bool {
  auto objectType = ast->baseExpression->type;
  auto cv = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointerType = type_cast<PointerType>(objectType);
    if (!pointerType) return false;
    objectType = pointerType->elementType();
    cv = strip_cv(objectType);
  }

  if (!control()->is_scalar(objectType)) {
    // return false if the object type is not a scalar type
    return false;
  }

  // from this point on we are going to assume that we want a pseudo destructor
  // to be called on a scalar type.

  auto dtor = ast_cast<DestructorIdAST>(ast->unqualifiedId);
  if (!dtor) return true;

  auto name = ast_cast<NameIdAST>(dtor->id);
  if (!name) return true;

  auto symbol =
      Lookup{scope()}.lookupType(ast->nestedNameSpecifier, name->identifier);
  if (!symbol) return true;

  if (!control()->is_same(symbol->type(), objectType)) {
    error(ast->unqualifiedId->firstSourceLocation(),
          "the type of object expression does not match the type "
          "being destroyed");
    return true;
  }

  ast->symbol = symbol;
  ast->type = control()->getFunctionType(control()->getVoidType(), {});

  return true;
}

}  // namespace cxx
