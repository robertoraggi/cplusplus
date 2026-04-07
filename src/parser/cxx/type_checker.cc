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

#include <cxx/type_checker.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/implicit_conversion_sequence.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/preprocessor.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <cmath>
#include <format>
#include <string_view>
#include <unordered_set>

namespace cxx {

namespace {

struct IsPotentiallyThrowing {
  auto operator()(ExpressionAST*) -> bool { return false; }

  auto operator()(ThrowExpressionAST*) -> bool { return true; }

  auto operator()(NoexceptExpressionAST*) -> bool { return false; }

  auto operator()(CallExpressionAST* ast) -> bool {
    auto base = ast->baseExpression;
    if (!base) return true;
    const FunctionType* ft = type_cast<FunctionType>(base->type);
    if (!ft)
      if (auto pt = type_cast<PointerType>(base->type))
        ft = type_cast<FunctionType>(pt->elementType());
    if (!ft || !ft->isNoexcept()) return true;
    for (auto it = ast->expressionList; it; it = it->next)
      if (apply(it->value)) return true;
    return apply(ast->baseExpression);
  }

  auto operator()(TypeConstructionAST* ast) -> bool {
    if (auto classType = type_cast<ClassType>(ast->type)) {
      auto cls = classType->definition();
      if (!cls || !cls->isComplete()) return true;
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) return cls->hasUserDeclaredConstructors();
      auto ctorFuncType = type_cast<FunctionType>(defCtor->type());
      return !ctorFuncType || !ctorFuncType->isNoexcept();
    }
    return false;
  }

  auto operator()(BinaryExpressionAST* ast) -> bool {
    if (ast->symbol) {
      auto ft = type_cast<FunctionType>(ast->symbol->type());
      if (!ft || !ft->isNoexcept()) return true;
    }
    return apply(ast->leftExpression) || apply(ast->rightExpression);
  }

  auto operator()(UnaryExpressionAST* ast) -> bool {
    if (ast->symbol) {
      auto ft = type_cast<FunctionType>(ast->symbol->type());
      if (!ft || !ft->isNoexcept()) return true;
    }
    return apply(ast->expression);
  }

  auto operator()(AssignmentExpressionAST* ast) -> bool {
    if (ast->symbol) {
      auto ft = type_cast<FunctionType>(ast->symbol->type());
      if (!ft || !ft->isNoexcept()) return true;
    }
    return apply(ast->leftExpression) || apply(ast->rightExpression);
  }

  auto operator()(CompoundAssignmentExpressionAST* ast) -> bool {
    if (ast->symbol) {
      auto ft = type_cast<FunctionType>(ast->symbol->type());
      if (!ft || !ft->isNoexcept()) return true;
    }
    return apply(ast->targetExpression) || apply(ast->rightExpression);
  }

  auto operator()(SubscriptExpressionAST* ast) -> bool {
    if (ast->symbol) {
      auto ft = type_cast<FunctionType>(ast->symbol->type());
      if (!ft || !ft->isNoexcept()) return true;
    }
    return apply(ast->baseExpression) || apply(ast->indexExpression);
  }

  auto operator()(DeleteExpressionAST*) -> bool { return true; }

  auto operator()(CppCastExpressionAST* ast) -> bool {
    if (ast->castOp == TokenKind::T_DYNAMIC_CAST) {
      if (!ast->type || !type_cast<PointerType>(ast->type)) return true;
    }
    return apply(ast->expression);
  }

  auto operator()(CastExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(PostIncrExpressionAST* ast) -> bool {
    if (!ast->baseExpression) return false;
    auto baseType = ast->baseExpression->type;
    if (baseType && type_cast<ClassType>(baseType)) return true;
    return false;
  }

  auto operator()(BracedTypeConstructionAST* ast) -> bool {
    if (auto classType = type_cast<ClassType>(ast->type)) {
      auto cls = classType->definition();
      if (!cls || !cls->isComplete()) return true;
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) return cls->hasUserDeclaredConstructors();
      auto ctorFuncType = type_cast<FunctionType>(defCtor->type());
      return !ctorFuncType || !ctorFuncType->isNoexcept();
    }
    return false;
  }

  auto operator()(NewExpressionAST*) -> bool { return true; }

  auto operator()(ImplicitCastExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(NestedExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(ConditionalExpressionAST* ast) -> bool {
    return apply(ast->condition) || apply(ast->iftrueExpression) ||
           apply(ast->iffalseExpression);
  }

  auto apply(ExpressionAST* expr) -> bool {
    if (!expr) return false;
    return visit(*this, expr);
  }
};

[[nodiscard]] auto isPotentiallyThrowing(ExpressionAST* expr) -> bool {
  return IsPotentiallyThrowing{}.apply(expr);
}

}  // namespace

struct TypeChecker::Visitor {
  TypeChecker& check;

  [[nodiscard]] auto arena() const -> Arena* { return check.unit_->arena(); }

  [[nodiscard]] auto globalScope() const -> ScopeSymbol* {
    return check.unit_->globalScope();
  }

  [[nodiscard]] auto scope() const -> ScopeSymbol* { return check.scope_; }

  [[nodiscard]] auto control() const -> Control* {
    return check.unit_->control();
  }

  [[nodiscard]] auto isC() const {
    return check.unit_->language() == LanguageKind::kC;
  }

  [[nodiscard]] auto isCxx() const {
    return check.unit_->language() == LanguageKind::kCXX;
  }

  void error(SourceLocation loc, std::string message) {
    check.error(loc, std::move(message));
  }

  [[nodiscard]] auto is_dependent_type(const Type* type) const -> bool {
    return type && isDependent(check.unit_, type);
  }

  [[nodiscard]] auto dependent_type() const -> const Type* {
    return control()->getTypeParameterType(0, 0, false);
  }

  [[nodiscard]] auto in_template() const -> bool {
    return isEnclosedInTemplate(scope());
  }

  void warning(SourceLocation loc, std::string message) {
    check.warning(loc, std::move(message));
  }

  [[nodiscard]] auto report_unresolved_id(ExpressionAST* expr) -> bool;

  [[nodiscard]] auto strip_parentheses(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto strip_cv(const Type*& type) -> CvQualifiers;

  // standard conversions

  StandardConversion stdconv_{check.unit_,
                              check.unit_->language() == LanguageKind::kC};

  void setResultTypeAndValueCategory(ExpressionAST* ast, const Type* type);

  [[nodiscard]] auto implicit_conversion(ExpressionAST*& expr,
                                         const Type* destinationType) -> bool;

  [[nodiscard]] auto as_pointer(const Type* type) const -> const PointerType* {
    return check.as_pointer(type);
  }

  [[nodiscard]] auto as_array(const Type* type) const -> const Type* {
    if (check.unit_->typeTraits().is_array(type))
      return check.unit_->typeTraits().remove_cv(type);
    return nullptr;
  }

  [[nodiscard]] auto as_class(const Type* type) const -> const ClassType* {
    return check.as_class(type);
  }

  void emit_implicit_cast(ExpressionAST*& outer, ExpressionAST* inner,
                          const Type* type, ImplicitCastKind kind) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = kind;
    cast->expression = inner;
    cast->type = type;
    cast->valueCategory = ValueCategory::kPrValue;
    outer = cast;
  }

  void check_cpp_cast_expression(CppCastExpressionAST* ast);

  [[nodiscard]] auto check_static_cast(ExpressionAST*& expression,
                                       const Type* targetType,
                                       ValueCategory targetVC) -> bool;

  [[nodiscard]] auto check_static_cast_to_derived_ref(
      ExpressionAST*& expression, const Type* targetType,
      ValueCategory targetVC) -> bool;

  [[nodiscard]] auto is_reference_compatible(const Type* targetType,
                                             const Type* sourceType) -> bool;

  [[nodiscard]] auto check_const_cast(ExpressionAST*& expression,
                                      const Type* targetType,
                                      ValueCategory valueCategory) -> bool;

  [[nodiscard]] auto are_similar_types(const Type* T1, const Type* T2) -> bool;

  [[nodiscard]] auto check_reinterpret_cast(ExpressionAST*& expression,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool;

  [[nodiscard]] auto check_reinterpret_cast_permissive(
      ExpressionAST*& expression, const Type* targetType,
      ValueCategory targetVC) -> bool;

  [[nodiscard]] auto casts_away_constness(const Type* sourceType,
                                          const Type* targetType) -> bool;

  [[nodiscard]] auto check_cast_to_derived(ExpressionAST* expression,
                                           const Type* targetType) -> bool;

  void classify_reference_type(ExpressionAST* ast, const Type* fullType);
  void check_address_of(UnaryExpressionAST* ast);
  void check_unary_promote(UnaryExpressionAST* ast);
  void check_shift(BinaryExpressionAST* ast);
  void check_relational(BinaryExpressionAST* ast);
  void check_equality(BinaryExpressionAST* ast);
  void prepare_comparison_operands(BinaryExpressionAST* ast);
  void set_base_symbol(ExpressionAST* base, Symbol* sym);
  void resolve_call_overload(CallExpressionAST* ast,
                             const std::vector<const Type*>& argTypes);
  [[nodiscard]] auto resolve_function_type(CallExpressionAST* ast)
      -> const FunctionType*;
  [[nodiscard]] auto resolve_call_operator(CallExpressionAST* ast)
      -> const FunctionType*;
  [[nodiscard]] auto resolve_arrow_operator(MemberExpressionAST* ast)
      -> FunctionSymbol*;
  void check_call_arguments(CallExpressionAST* ast,
                            const FunctionType* functionType);
  [[nodiscard]] auto try_c_style_cast(CastExpressionAST* ast,
                                      ExpressionAST*& expr,
                                      const Type* targetType,
                                      ValueCategory targetVC) -> bool;

  void check_addition(BinaryExpressionAST* ast);
  void check_subtraction(BinaryExpressionAST* ast);
  void check_prefix_increment_decrement(UnaryExpressionAST* ast,
                                        std::string_view action,
                                        std::string_view opWord);

  [[nodiscard]] auto resolve_operator_overload(
      const Type* leftType, TokenKind op, SourceLocation opLoc,
      const Type* rightType, FunctionSymbol*& symbolOut) -> bool;

  [[nodiscard]] auto resolve_unary_overload(UnaryExpressionAST* ast) -> bool;

  [[nodiscard]] auto resolve_binary_overload(BinaryExpressionAST* ast,
                                             bool setValueCategory = true)
      -> bool;

  [[nodiscard]] auto resolve_assignment_overload(AssignmentExpressionAST* ast)
      -> bool;

  [[nodiscard]] auto resolve_compound_assignment_overload(
      CompoundAssignmentExpressionAST* ast) -> bool;

  [[nodiscard]] auto check_member_access(MemberExpressionAST* ast) -> bool;
  [[nodiscard]] auto check_pseudo_destructor_access(MemberExpressionAST* ast)
      -> bool;

  void check_static_assert(StaticAssertDeclarationAST* ast);

  [[nodiscard]] static auto getParameterSymbols(FunctionSymbol* func)
      -> std::vector<ParameterSymbol*> {
    std::vector<ParameterSymbol*> params;
    if (auto fpScope = func->functionParameters()) {
      for (auto member : fpScope->members()) {
        if (auto param = symbol_cast<ParameterSymbol>(member))
          params.push_back(param);
      }
    }
    return params;
  }

  [[nodiscard]] static auto getMinRequiredArgs(FunctionSymbol* func,
                                               int totalParams) -> int {
    auto params = getParameterSymbols(func);
    if (params.empty()) return totalParams;

    // Count trailing defaults from the end
    int defaultCount = 0;
    for (int i = static_cast<int>(params.size()) - 1; i >= 0; --i) {
      if (params[i]->defaultArgument())
        ++defaultCount;
      else
        break;
    }
    return totalParams - defaultCount;
  }

  void appendDefaultArguments(CallExpressionAST* ast, FunctionSymbol* func,
                              int argCount, int totalParams) {
    if (argCount >= totalParams) return;

    auto params = getParameterSymbols(func);
    if (params.empty()) return;

    auto ar = arena();

    List<ExpressionAST*>** tail = &ast->expressionList;
    while (*tail) tail = &(*tail)->next;

    for (int idx = argCount; idx < totalParams && idx < (int)params.size();
         ++idx) {
      if (auto defaultExpr = params[idx]->defaultArgument()) {
        auto cloned = defaultExpr->clone(ar);
        *tail = make_list_node<ExpressionAST>(ar, cloned);
        tail = &(*tail)->next;
      }
    }
  }

  auto deduceTemplateArguments(
      FunctionSymbol* func, List<ExpressionAST*>* expressionList,
      List<TemplateArgumentAST*>* explicitTemplateArgumentList = nullptr)
      -> std::optional<List<TemplateArgumentAST*>*> {
    TemplateArgumentDeduction deduction(check.unit_);
    return deduction.deduce(func, expressionList, explicitTemplateArgumentList);
  }

  void operator()(CharLiteralExpressionAST* ast);
  void operator()(BoolLiteralExpressionAST* ast);
  void operator()(IntLiteralExpressionAST* ast);
  void operator()(FloatLiteralExpressionAST* ast);
  void operator()(NullptrLiteralExpressionAST* ast);
  void operator()(StringLiteralExpressionAST* ast);
  void operator()(UserDefinedStringLiteralExpressionAST* ast);
  void operator()(ObjectLiteralExpressionAST* ast);
  void operator()(ThisExpressionAST* ast);
  void operator()(PackIndexExpressionAST* ast);
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
  void operator()(TargetExpressionAST* ast);
  void operator()(RightExpressionAST* ast);
  void operator()(CompoundAssignmentExpressionAST* ast);
  void operator()(PackExpansionExpressionAST* ast);
  void operator()(DesignatedInitializerClauseAST* ast);
  void operator()(TypeTraitExpressionAST* ast);
  void operator()(ConditionExpressionAST* ast);
  void operator()(EqualInitializerAST* ast);
  void operator()(BracedInitListAST* ast);
  void operator()(ParenInitializerAST* ast);
};

void TypeChecker::Visitor::operator()(CharLiteralExpressionAST* ast) {
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(BoolLiteralExpressionAST* ast) {
  if (!ast->type) ast->type = control()->getBoolType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(IntLiteralExpressionAST* ast) {
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(FloatLiteralExpressionAST* ast) {
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(NullptrLiteralExpressionAST* ast) {
  if (!ast->type) ast->type = control()->getNullptrType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(StringLiteralExpressionAST* ast) {
  ast->valueCategory = ValueCategory::kLValue;
}

void TypeChecker::Visitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ObjectLiteralExpressionAST* ast) {
  if (ast->typeId) {
    ast->type = ast->typeId->type;
  }
  ast->valueCategory = ValueCategory::kLValue;

  if (ast->type && ast->bracedInitList) {
    check.check_braced_init_list(ast->type, ast->bracedInitList);
  }

  // Compute array bounds
  if (auto unbounded = type_cast<UnboundedArrayType>(ast->type)) {
    if (ast->bracedInitList) {
      auto elementType = unbounded->elementType();
      const auto isCharElem = type_cast<CharType>(elementType) ||
                              type_cast<SignedCharType>(elementType) ||
                              type_cast<UnsignedCharType>(elementType);

      // char array from string literal in braces.
      if (isCharElem && ast->bracedInitList->expressionList &&
          !ast->bracedInitList->expressionList->next) {
        if (auto strLit = ast_cast<StringLiteralExpressionAST>(
                ast->bracedInitList->expressionList->value)) {
          if (auto srcArray = type_cast<BoundedArrayType>(strLit->type)) {
            ast->type =
                control()->getBoundedArrayType(elementType, srcArray->size());
          }
        }
      }

      // count elements
      if (type_cast<UnboundedArrayType>(ast->type)) {
        size_t count = 0;
        for (auto it = ast->bracedInitList->expressionList; it; it = it->next)
          ++count;
        if (count > 0) {
          ast->type = control()->getBoundedArrayType(elementType, count);
        }
      }

      // Update the braced init list type to the completed type.
      ast->bracedInitList->type = ast->type;
    }
  }

  // create an anonymous VariableSymbol for the compound literal
  if (ast->type) {
    auto symbol =
        control()->newVariableSymbol(scope(), ast->firstSourceLocation());
    symbol->setType(ast->type);
    symbol->setStatic(true);
    ast->symbol = symbol;

    if (ast->bracedInitList) {
      auto interp = ASTInterpreter{check.unit_};
      auto value = interp.evaluate(ast->bracedInitList);
      symbol->setConstValue(value);
    }
  }
}

void TypeChecker::Visitor::operator()(ThisExpressionAST* ast) {
  auto scope_ = check.scope_;

  for (auto current = scope_; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current)) {
      // maybe a this expression in a field initializer
      ast->type = control()->getPointerType(classSymbol->type());
      break;
    }

    if (auto functionSymbol = symbol_cast<FunctionSymbol>(current)) {
      if (auto classSymbol =
              symbol_cast<ClassSymbol>(functionSymbol->parent())) {
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

void TypeChecker::Visitor::operator()(PackIndexExpressionAST* ast) {
  if (!ast->indexExpression) {
    error(ast->firstSourceLocation(), "missing index expression in pack index");
    return;
  }

  if (in_template()) {
    ast->valueCategory = ValueCategory::kLValue;
    ast->type = dependent_type();
  }
}

void TypeChecker::Visitor::operator()(GenericSelectionExpressionAST* ast) {
  struct {
    Visitor& self;
    GenericSelectionExpressionAST* ast;
    DefaultGenericAssociationAST* defaultAssoc = nullptr;
    const Type* selectorType = nullptr;
    int index = 0;
    int defaultAssocIndex = -1;

    [[nodiscard]] auto control() -> Control* { return self.control(); }

    void operator()(DefaultGenericAssociationAST* assoc) {
      if (defaultAssoc) {
        self.error(ast->firstSourceLocation(),
                   "multiple default associations in _Generic selection");
        return;
      }

      defaultAssoc = assoc;
      defaultAssocIndex = index;
    }

    void operator()(TypeGenericAssociationAST* assoc) {
      if (!self.check.unit_->typeTraits().is_same(selectorType,
                                                  assoc->typeId->type)) {
        return;
      }

      if (ast->matchedAssocIndex != -1) {
        self.error(ast->firstSourceLocation(),
                   std::format("multiple matching types for _Generic selector "
                               "of type '{}'",
                               to_string(selectorType)));
        return;
      }

      ast->type = assoc->expression->type;
      ast->valueCategory = assoc->expression->valueCategory;
      ast->matchedAssocIndex = index;
    }

    void check() {
      if (!ast->expression) {
        self.error(ast->firstSourceLocation(),
                   "generic selection expression without selector");
        return;
      }

      selectorType =
          self.check.unit_->typeTraits().decay(ast->expression->type);

      if (!selectorType) {
        self.error(ast->firstSourceLocation(),
                   "generic selection expression with invalid selector type");
        return;
      }

      for (auto assoc : ListView{ast->genericAssociationList}) {
        visit(*this, assoc);
        ++index;
      }

      if (ast->matchedAssocIndex == -1 && defaultAssoc) {
        ast->type = defaultAssoc->expression->type;
        ast->valueCategory = defaultAssoc->expression->valueCategory;
        ast->matchedAssocIndex = defaultAssocIndex;
      }

      if (ast->matchedAssocIndex == -1) {
        self.error(
            ast->firstSourceLocation(),
            std::format("no matching type for _Generic selector of type '{}'",
                        to_string(selectorType)));
      }
    }
  } v{*this, ast};

  v.check();
}

void TypeChecker::Visitor::operator()(NestedStatementExpressionAST* ast) {
  if (!ast->statement) {
    error(ast->firstSourceLocation(), "expected a compound statement");
    return;
  }

  // The type is the type of the last expression statement (GNU extension).
  if (!ast->type) {
    StatementAST* lastStmt = nullptr;
    for (auto node : ListView{ast->statement->statementList}) lastStmt = node;
    if (auto exprStmt = ast_cast<ExpressionStatementAST>(lastStmt)) {
      if (exprStmt->expression && exprStmt->expression->type) {
        ast->type = exprStmt->expression->type;
        ast->valueCategory = exprStmt->expression->valueCategory;
      }
    }
    if (!ast->type) ast->type = control()->getVoidType();
  }
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(NestedExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (report_unresolved_id(ast->expression)) return;
  if (!ast->expression->type) return;

  ast->type = ast->expression->type;
  ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::operator()(IdExpressionAST* ast) {
  if (ast->symbol) {
    if (auto conceptSymbol = symbol_cast<ConceptSymbol>(ast->symbol)) {
      ast->type = control()->getBoolType();
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      ast->type =
          check.unit_->typeTraits().remove_reference(ast->symbol->type());

      if (ast->symbol->isEnumerator() || ast->symbol->isNonTypeParameter()) {
        ast->valueCategory = ValueCategory::kPrValue;
        stdconv_.adjustCv(ast);
      } else {
        ast->valueCategory = ValueCategory::kLValue;
      }
    }
  } else {
    // maybe unresolved name
    if (in_template() && ast->nestedNameSpecifier && !ast->symbol) {
      if (isDependent(check.unit_, ast->nestedNameSpecifier)) {
        ast->type = dependent_type();
        ast->valueCategory = ValueCategory::kPrValue;
      }
    }
  }
}

void TypeChecker::Visitor::operator()(LambdaExpressionAST* ast) {
  if (ast->symbol && !ast->type) ast->type = ast->symbol->type();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(FoldExpressionAST* ast) {
  if (!ast->leftExpression && !ast->rightExpression) {
    error(ast->firstSourceLocation(), "expected a fold operand");
    return;
  }

  if (ast->op == TokenKind::T_EOF_SYMBOL ||
      ast->foldOp == TokenKind::T_EOF_SYMBOL) {
    error(ast->firstSourceLocation(), "expected a fold operator");
    return;
  }

  if (ast->op != ast->foldOp) {
    error(ast->firstSourceLocation(),
          std::format("mismatched fold operators '{}' and '{}'",
                      Token::spell(ast->op), Token::spell(ast->foldOp)));
    return;
  }

  if (report_unresolved_id(ast->leftExpression) |
      report_unresolved_id(ast->rightExpression)) {
    return;
  }

  if (!ast->type) {
    if (ast->leftExpression && ast->leftExpression->type) {
      ast->type = ast->leftExpression->type;
    } else if (ast->rightExpression && ast->rightExpression->type) {
      ast->type = ast->rightExpression->type;
    }
  }

  if (!ast->type && (ast->leftExpression || ast->rightExpression)) {
    error(ast->firstSourceLocation(), "invalid fold expression operand");
    return;
  }

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(RightFoldExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected a fold operand");
    return;
  }

  if (ast->op == TokenKind::T_EOF_SYMBOL) {
    error(ast->firstSourceLocation(), "expected a fold operator");
    return;
  }

  if (report_unresolved_id(ast->expression)) return;

  if (!ast->type && ast->expression) ast->type = ast->expression->type;

  if (!ast->type) {
    error(ast->firstSourceLocation(), "invalid fold expression operand");
    return;
  }

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(LeftFoldExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected a fold operand");
    return;
  }

  if (ast->op == TokenKind::T_EOF_SYMBOL) {
    error(ast->firstSourceLocation(), "expected a fold operator");
    return;
  }

  if (report_unresolved_id(ast->expression)) return;

  if (!ast->type && ast->expression) ast->type = ast->expression->type;

  if (!ast->type) {
    error(ast->firstSourceLocation(), "invalid fold expression operand");
    return;
  }

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(RequiresExpressionAST* ast) {
  ast->type = control()->getBoolType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(VaArgExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->typeId || !ast->typeId->type) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  ast->type = ast->typeId->type;
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(SubscriptExpressionAST* ast) {
  if (!ast->baseExpression) {
    error(ast->firstSourceLocation(), "expected an expression before '['");
    return;
  }

  if (!ast->indexExpression) {
    error(ast->firstSourceLocation(), "expected an index expression");
    return;
  }

  if (report_unresolved_id(ast->baseExpression) |
      report_unresolved_id(ast->indexExpression)) {
    return;
  }

  if (!ast->baseExpression->type || !ast->indexExpression->type) return;

  if (in_template() && (is_dependent_type(ast->baseExpression->type) ||
                        is_dependent_type(ast->indexExpression->type))) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  if (auto operatorFunc =
          check.lookupOperator(ast->baseExpression->type, TokenKind::T_LBRACKET,
                               ast->indexExpression->type)) {
    ast->symbol = operatorFunc;
    setResultTypeAndValueCategory(ast, operatorFunc->type());
    return;
  }

  if (check.wasLastOperatorLookupAmbiguous()) {
    error(ast->firstSourceLocation(),
          "call to overloaded operator '[]' is ambiguous");
    return;
  }

  if (check.unit_->typeTraits().is_class(ast->baseExpression->type)) return;
  if (check.unit_->typeTraits().is_class(ast->indexExpression->type)) return;

  auto array_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                ExpressionAST*& index) {
    if (!check.unit_->typeTraits().is_array(base->type)) return false;
    if (!check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(index->type))
      return false;

    (void)stdconv_.temporaryMaterialization(base);
    (void)stdconv_.ensurePrvalue(index);
    stdconv_.adjustCv(index);
    (void)stdconv_.integralPromotion(index);

    ast->type = check.unit_->typeTraits().get_element_type(base->type);
    ast->valueCategory = base->valueCategory;
    return true;
  };

  auto pointer_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                  ExpressionAST*& index) {
    if (!check.unit_->typeTraits().is_pointer(base->type)) return false;
    if (!check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(index->type))
      return false;

    (void)stdconv_.ensurePrvalue(base);
    stdconv_.adjustCv(base);

    (void)stdconv_.ensurePrvalue(index);
    stdconv_.adjustCv(index);
    (void)stdconv_.integralPromotion(index);

    ast->type = check.unit_->typeTraits().get_element_type(base->type);
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

void TypeChecker::Visitor::set_base_symbol(ExpressionAST* base, Symbol* sym) {
  if (auto id = ast_cast<IdExpressionAST>(base))
    id->symbol = sym;
  else if (auto member = ast_cast<MemberExpressionAST>(base))
    member->symbol = sym;
}

void TypeChecker::Visitor::resolve_call_overload(
    CallExpressionAST* ast, const std::vector<const Type*>& argumentTypes) {
  auto ovl = type_cast<OverloadSetType>(ast->baseExpression->type);
  if (!ovl) return;

  OverloadResolution resolution(check.unit_);

  int argCount = 0;
  for (auto it = ast->expressionList; it; it = it->next) ++argCount;

  bool isMemberCall = false;
  CvQualifiers objectCv{};
  ValueCategory objectValueCategory = ValueCategory::kPrValue;
  if (auto access = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    isMemberCall = true;
    const Type* objectType = access->baseExpression->type;
    objectValueCategory = access->baseExpression->valueCategory;

    if (access->accessOp == TokenKind::T_MINUS_GREATER) {
      if (auto ptrType = as_pointer(objectType))
        objectType = ptrType->elementType();
    }

    objectCv = strip_cv(objectType);
  }

  std::vector<FunctionSymbol*> allFunctions;
  for (auto func : ovl->symbol()->functions()) {
    if (func->canonical() != func) continue;
    if (func->isSpecialization()) continue;
    allFunctions.push_back(func);
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (!idExpr->nestedNameSpecifier) {
      auto adlCandidates =
          argumentDependentLookup(ovl->symbol()->name(), argumentTypes);
      for (auto f : adlCandidates)
        if (std::find(allFunctions.begin(), allFunctions.end(), f) ==
            allFunctions.end())
          allFunctions.push_back(f);
    }
  }

  if (!isMemberCall && !allFunctions.empty() &&
      !allFunctions.front()->isStatic()) {
    for (auto current = check.scope_; current; current = current->parent()) {
      if (auto funcSym = symbol_cast<FunctionSymbol>(current)) {
        if (symbol_cast<ClassSymbol>(funcSym->parent())) {
          if (auto funcType = type_cast<FunctionType>(funcSym->type())) {
            isMemberCall = true;
            objectCv = funcType->cvQualifiers();
            objectValueCategory = ValueCategory::kLValue;
          }
        }
        break;
      }
    }
  }

  std::vector<Candidate> candidates;

  auto explicitTemplateArguments = [&]() -> List<TemplateArgumentAST*>* {
    if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
      if (auto templateId =
              ast_cast<SimpleTemplateIdAST>(idExpr->unqualifiedId)) {
        return templateId->templateArgumentList;
      }
      return nullptr;
    }

    auto memberExpr = ast_cast<MemberExpressionAST>(ast->baseExpression);
    if (!memberExpr) return nullptr;
    auto templateId = ast_cast<SimpleTemplateIdAST>(memberExpr->unqualifiedId);
    if (!templateId) return nullptr;
    return templateId->templateArgumentList;
  }();

  for (auto func : allFunctions) {
    auto type = type_cast<FunctionType>(func->type());
    if (!type) continue;

    const bool templateCandidate =
        func->templateDeclaration() != nullptr && !func->isSpecialization();

    if (func->templateDeclaration() && !func->isSpecialization()) {
      auto deducedArgs = deduceTemplateArguments(func, ast->expressionList,
                                                 explicitTemplateArguments);
      if (!deducedArgs.has_value()) continue;
      auto instantiated =
          ASTRewriter::instantiate(check.unit_, *deducedArgs, func,
                                   ast->baseExpression->firstSourceLocation(),
                                   /*sfinaeContext=*/true);
      if (!instantiated) continue;
      auto instFunc = symbol_cast<FunctionSymbol>(instantiated);
      if (!instFunc) continue;
      func = instFunc;
      type = type_cast<FunctionType>(func->type());
      if (!type) continue;
    }

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) continue;
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(func, paramCount)) continue;
    }

    Candidate cand{func};
    cand.viable = true;
    cand.fromTemplate = templateCandidate;

    if (isMemberCall && !func->isStatic()) {
      auto funcCv = type->cvQualifiers();
      auto funcRef = type->refQualifier();

      if (!cv_is_subset_of(objectCv, funcCv)) continue;

      if (funcRef == RefQualifier::kLvalue &&
          objectValueCategory == ValueCategory::kPrValue)
        continue;
      if (funcRef == RefQualifier::kRvalue &&
          objectValueCategory == ValueCategory::kLValue)
        continue;

      cand.exactCvMatch = (objectCv == funcCv);
    }

    auto paramIt = type->parameterTypes().begin();
    auto paramEnd = type->parameterTypes().end();
    for (auto argIt = ast->expressionList; argIt && paramIt != paramEnd;
         argIt = argIt->next, ++paramIt) {
      auto conv =
          resolution.computeImplicitConversionSequence(argIt->value, *paramIt);
      if (conv.rank == ConversionRank::kNone) {
        cand.viable = false;
        break;
      }
      cand.conversions.push_back(conv);
    }

    if (cand.viable && type->isVariadic()) {
      int coveredArgs = static_cast<int>(type->parameterTypes().size());
      int idx = 0;
      for (auto argIt = ast->expressionList; argIt;
           argIt = argIt->next, ++idx) {
        if (idx >= coveredArgs) {
          ImplicitConversionSequence ellipsisConv;
          ellipsisConv.kind = ConversionSequenceKind::kEllipsis;
          ellipsisConv.rank = ConversionRank::kConversion;
          cand.conversions.push_back(ellipsisConv);
        }
      }
    }

    if (cand.viable) candidates.push_back(cand);
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(candidates, isMemberCall, true);

  if (!bestPtr) {
    error(ast->firstSourceLocation(), "no matching function for call");
    return;
  }
  if (ambiguous) {
    error(ast->firstSourceLocation(), "call to function is ambiguous");
    return;
  }

  auto function = bestPtr->symbol;
  if (function->isSpecialization()) {
    ASTRewriter::reportPendingInstantiationErrors(
        check.unit_, function->primaryTemplateSymbol(), function,
        ast->baseExpression->firstSourceLocation());
  }
  ast->baseExpression->type = function->type();
  set_base_symbol(ast->baseExpression, function);

  auto selectedType = type_cast<FunctionType>(function->type());
  if (selectedType) {
    auto totalParams = static_cast<int>(selectedType->parameterTypes().size());
    appendDefaultArguments(ast, function, argCount, totalParams);
  }

  int argIdx = 0;
  for (auto it = ast->expressionList;
       it && argIdx < static_cast<int>(bestPtr->conversions.size());
       it = it->next, ++argIdx) {
    if (bestPtr->conversions[argIdx].kind != ConversionSequenceKind::kEllipsis)
      resolution.applyImplicitConversion(bestPtr->conversions[argIdx],
                                         it->value);
  }
}

auto TypeChecker::Visitor::resolve_function_type(CallExpressionAST* ast)
    -> const FunctionType* {
  auto functionType = type_cast<FunctionType>(ast->baseExpression->type);

  if (functionType) {
    auto stripped = ast->baseExpression;

    while (auto nested = ast_cast<NestedExpressionAST>(stripped))
      stripped = nested->expression;

    bool isDirectCall = false;

    if (auto idExpr = ast_cast<IdExpressionAST>(stripped))
      isDirectCall = symbol_cast<FunctionSymbol>(idExpr->symbol) != nullptr;
    else if (auto memberExpr = ast_cast<MemberExpressionAST>(stripped))
      isDirectCall = symbol_cast<FunctionSymbol>(memberExpr->symbol) != nullptr;

    if (!isDirectCall) {
      (void)stdconv_.functionToPointer(ast->baseExpression);
    }
  }

  if (!functionType &&
      check.unit_->typeTraits().is_pointer(ast->baseExpression->type)) {
    functionType = type_cast<FunctionType>(
        check.unit_->typeTraits().get_element_type(ast->baseExpression->type));
    if (functionType) (void)stdconv_.ensurePrvalue(ast->baseExpression);
  }

  if (!functionType) functionType = resolve_call_operator(ast);

  return functionType;
}

auto TypeChecker::Visitor::resolve_call_operator(CallExpressionAST* ast)
    -> const FunctionType* {
  auto baseType =
      check.unit_->typeTraits().remove_cvref(ast->baseExpression->type);
  auto classType = type_cast<ClassType>(baseType);
  if (!classType) return nullptr;
  auto classSymbol = classType->symbol();
  if (!classSymbol) return nullptr;

  auto operatorName = control()->getOperatorId(TokenKind::T_LPAREN);
  OverloadResolution resolution(check.unit_);
  auto allFunctions = resolution.findCandidates(classSymbol, operatorName);
  if (allFunctions.empty()) return nullptr;

  int argCount = 0;
  for (auto it = ast->expressionList; it; it = it->next) ++argCount;

  auto objectCv = strip_cv(baseType);
  auto objectValueCategory = ast->baseExpression->valueCategory;

  std::vector<Candidate> viableCandidates;
  for (auto func : allFunctions) {
    if (func->templateDeclaration() && !func->isSpecialization()) continue;

    auto type = type_cast<FunctionType>(func->type());
    if (!type) continue;

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) continue;
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(func, paramCount)) continue;
    }

    Candidate cand{func};
    cand.viable = true;

    if (!func->isStatic()) {
      auto funcCv = type->cvQualifiers();
      auto funcRef = type->refQualifier();

      if (!cv_is_subset_of(objectCv, funcCv)) continue;
      if (funcRef == RefQualifier::kLvalue &&
          objectValueCategory == ValueCategory::kPrValue)
        continue;
      if (funcRef == RefQualifier::kRvalue &&
          objectValueCategory == ValueCategory::kLValue)
        continue;

      cand.exactCvMatch = (objectCv == funcCv);
    }

    auto paramIt = type->parameterTypes().begin();
    auto paramEnd = type->parameterTypes().end();
    for (auto argIt = ast->expressionList; argIt && paramIt != paramEnd;
         argIt = argIt->next, ++paramIt) {
      auto conv =
          resolution.computeImplicitConversionSequence(argIt->value, *paramIt);
      if (conv.rank == ConversionRank::kNone) {
        cand.viable = false;
        break;
      }
      cand.conversions.push_back(conv);
    }

    if (cand.viable && type->isVariadic()) {
      int idx = 0;
      for (auto argIt = ast->expressionList; argIt;
           argIt = argIt->next, ++idx) {
        if (idx >= paramCount) {
          ImplicitConversionSequence ellipsisConv;
          ellipsisConv.kind = ConversionSequenceKind::kEllipsis;
          ellipsisConv.rank = ConversionRank::kConversion;
          cand.conversions.push_back(ellipsisConv);
        }
      }
    }

    if (cand.viable) viableCandidates.push_back(std::move(cand));
  }

  if (viableCandidates.empty()) {
    bool anyDeductionSucceeded = false;
    for (auto func : allFunctions) {
      if (!func->templateDeclaration() || func->isSpecialization()) continue;
      auto deducedArgs =
          deduceTemplateArguments(func, ast->expressionList, nullptr);
      if (!deducedArgs.has_value()) continue;
      anyDeductionSucceeded = true;
      auto instantiated = ASTRewriter::instantiate(
          check.unit_, *deducedArgs, func,
          ast->baseExpression->firstSourceLocation(), /*sfinaeContext=*/true);
      if (!instantiated) continue;
      auto instFunc = symbol_cast<FunctionSymbol>(instantiated);
      if (!instFunc) continue;
      Candidate cand{instFunc};
      cand.viable = true;
      viableCandidates.push_back(std::move(cand));
    }
    if (anyDeductionSucceeded && viableCandidates.empty()) {
      for (auto func : allFunctions) {
        if (!func->templateDeclaration() || func->isSpecialization()) continue;
        auto deducedArgs =
            deduceTemplateArguments(func, ast->expressionList, nullptr);
        if (!deducedArgs.has_value()) continue;
        (void)ASTRewriter::instantiate(
            check.unit_, *deducedArgs, func,
            ast->baseExpression->firstSourceLocation(),
            /*sfinaeContext=*/false);
      }
      return nullptr;
    }
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(viableCandidates, true, false);

  if (!bestPtr) return nullptr;
  if (ambiguous) {
    error(ast->firstSourceLocation(),
          "call to overloaded operator() is ambiguous");
    return nullptr;
  }

  auto operatorFunc = bestPtr->symbol;
  if (operatorFunc->templateDeclaration() &&
      !operatorFunc->isSpecialization()) {
    auto deducedArgs =
        deduceTemplateArguments(operatorFunc, ast->expressionList, nullptr);
    if (deducedArgs.has_value()) {
      auto instantiated = ASTRewriter::instantiate(
          check.unit_, *deducedArgs, operatorFunc,
          ast->baseExpression->firstSourceLocation(), /*sfinaeContext=*/true);
      if (auto instFunc = symbol_cast<FunctionSymbol>(instantiated))
        operatorFunc = instFunc;
    }
  }
  if (operatorFunc->isSpecialization()) {
    ASTRewriter::reportPendingInstantiationErrors(
        check.unit_, operatorFunc->primaryTemplateSymbol(), operatorFunc,
        ast->baseExpression->firstSourceLocation());
  }
  auto functionType = type_cast<FunctionType>(operatorFunc->type());
  if (!functionType) return nullptr;

  auto totalParams = static_cast<int>(functionType->parameterTypes().size());
  appendDefaultArguments(ast, operatorFunc, argCount, totalParams);

  auto ar = arena();
  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);
  ast->baseExpression = MemberExpressionAST::create(
      ar, ast->baseExpression, nullptr, opId, operatorFunc, TokenKind::T_DOT,
      false, ValueCategory::kLValue, operatorFunc->type());

  return functionType;
}

auto TypeChecker::Visitor::resolve_arrow_operator(MemberExpressionAST* ast)
    -> FunctionSymbol* {
  auto classType = type_cast<ClassType>(
      check.unit_->typeTraits().remove_cv(ast->baseExpression->type));
  if (!classType) return nullptr;
  auto classSymbol = classType->symbol();
  if (!classSymbol) return nullptr;

  auto operatorName = control()->getOperatorId(TokenKind::T_MINUS_GREATER);
  OverloadResolution resolution(check.unit_);
  auto allFunctions = resolution.findCandidates(classSymbol, operatorName);
  if (allFunctions.empty()) return nullptr;

  auto objectType = ast->baseExpression->type;
  auto objectCv = strip_cv(objectType);
  auto objectVC = ast->baseExpression->valueCategory;

  std::vector<Candidate> viableCandidates;
  for (auto func : allFunctions) {
    auto type = type_cast<FunctionType>(func->type());
    if (!type) continue;
    if (!type->parameterTypes().empty()) continue;

    Candidate cand{func};
    cand.viable = true;

    if (!func->isStatic()) {
      auto funcCv = type->cvQualifiers();
      if (!cv_is_subset_of(objectCv, funcCv)) continue;
      cand.exactCvMatch = (objectCv == funcCv);
    }

    viableCandidates.push_back(std::move(cand));
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(viableCandidates, true, false);
  if (!bestPtr || ambiguous) return nullptr;

  auto operatorFunc = bestPtr->symbol;
  auto functionType = type_cast<FunctionType>(operatorFunc->type());
  if (!functionType) return nullptr;

  auto ar = arena();
  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_MINUS_GREATER);

  auto memberAccess = MemberExpressionAST::create(
      ar, ast->baseExpression, nullptr, opId, operatorFunc, TokenKind::T_DOT,
      false, ValueCategory::kLValue, operatorFunc->type());

  auto callExpr = CallExpressionAST::create(
      ar, memberAccess, /*expressionList=*/nullptr, ValueCategory::kPrValue,
      functionType->returnType());

  ast->baseExpression = callExpr;
  return operatorFunc;
}

void TypeChecker::Visitor::check_call_arguments(
    CallExpressionAST* ast, const FunctionType* functionType) {
  const auto& parameterTypes = functionType->parameterTypes();

  OverloadResolution resolution(check.unit_);

  auto isVaBuiltinCall = [&]() -> bool {
    auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
    if (!idExpr) return false;
    auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId);
    if (!nameId || !nameId->identifier) return false;
    auto kind = nameId->identifier->builtinFunction();
    return kind == BuiltinFunctionKind::T___BUILTIN_VA_START ||
           kind == BuiltinFunctionKind::T___BUILTIN_VA_END ||
           kind == BuiltinFunctionKind::T___BUILTIN_VA_COPY ||
           kind == BuiltinFunctionKind::T___BUILTIN_C23_VA_START;
  }();

  int argc = 0;
  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) {
      error(ast->firstSourceLocation(),
            "invalid call with null argument expression");
      continue;
    }

    if (argc >= static_cast<int>(parameterTypes.size())) {
      if (functionType->isVariadic()) {
        (void)stdconv_.ensurePrvalue(it->value);
        stdconv_.adjustCv(it->value);
        if (stdconv_.integralPromotion(it->value)) continue;
        if (stdconv_.floatingPointPromotion(it->value)) continue;
        continue;
      }
      error(it->value->firstSourceLocation(),
            std::format("too many arguments for function of type '{}'",
                        to_string(functionType)));
      break;
    }

    auto targetType = parameterTypes[argc];
    ++argc;

    if (in_template() && !targetType) continue;

    if (isVaBuiltinCall && type_cast<BuiltinVaListType>(targetType)) {
      continue;
    }

    if (ast_cast<BracedInitListAST>(it->value)) {
      auto seq =
          resolution.computeImplicitConversionSequence(it->value, targetType);
      if (seq.rank == ConversionRank::kNone) {
        error(it->value->firstSourceLocation(),
              std::format("invalid argument of type '{}' for parameter of "
                          "type '{}'",
                          to_string(it->value->type), to_string(targetType)));
      } else {
        auto elemType = resolution.initializerListElementType(targetType);
        if (elemType) {
          it->value->type = targetType;
          it->value->valueCategory = ValueCategory::kPrValue;
        }
        resolution.applyImplicitConversion(seq, it->value);
      }
      continue;
    }

    if (isCxx() && check.unit_->typeTraits().is_reference(targetType)) {
      auto seq =
          resolution.computeImplicitConversionSequence(it->value, targetType);
      if (seq.rank == ConversionRank::kNone) {
        error(it->value->firstSourceLocation(),
              std::format("invalid argument of type '{}' for parameter of "
                          "type '{}'",
                          to_string(it->value->type), to_string(targetType)));
      }
      continue;
    }

    if (!implicit_conversion(it->value, targetType)) {
      error(it->value->firstSourceLocation(),
            std::format("invalid argument of type '{}' for parameter of type "
                        "'{}'",
                        to_string(it->value->type), to_string(targetType)));
    }
  }
}

void TypeChecker::Visitor::operator()(CallExpressionAST* ast) {
  if (!ast->baseExpression) return;

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto classSym = symbol_cast<ClassSymbol>(idExpr->symbol)) {
      ast->type = classSym->type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
  }

  if (in_template() && (is_dependent_type(ast->baseExpression->type) ||
                        type_cast<AutoType>(ast->baseExpression->type))) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto templateId =
            ast_cast<SimpleTemplateIdAST>(idExpr->unqualifiedId)) {
      for (auto arg : ListView{templateId->templateArgumentList}) {
        if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
          if (isDependent(check.unit_, typeArg->typeId)) {
            ast->type = dependent_type();
            ast->valueCategory = ValueCategory::kPrValue;
            return;
          }
        } else if (auto exprArg =
                       ast_cast<ExpressionTemplateArgumentAST>(arg)) {
          if (isDependent(check.unit_, exprArg->expression)) {
            ast->type = dependent_type();
            ast->valueCategory = ValueCategory::kPrValue;
            return;
          }
        }
      }
    }
  }

  std::vector<const Type*> argumentTypes;
  for (auto it = ast->expressionList; it; it = it->next)
    argumentTypes.push_back(it->value ? it->value->type : nullptr);

  if (in_template()) {
    bool anyDependent = false;
    for (const auto argType : argumentTypes) {
      if (!argType || is_dependent_type(argType)) {
        anyDependent = true;
        break;
      }
    }
    if (anyDependent) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
  } else {
    bool anyDependent = false;
    for (const auto argType : argumentTypes) {
      if (is_dependent_type(argType)) {
        anyDependent = true;
        break;
      }
    }
    if (anyDependent) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
  }

  if (auto access = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    if (ast_cast<DestructorIdAST>(access->unqualifiedId)) {
      ast->type = control()->getVoidType();
      return;
    }
  }

  resolve_call_overload(ast, argumentTypes);

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (!idExpr->symbol && !idExpr->nestedNameSpecifier) {
      if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
        auto name = nameId->identifier;
        if (name && check.scope_) {
          auto adlCandidates = argumentDependentLookup(name, argumentTypes);
          if (!adlCandidates.empty()) {
            idExpr->symbol = adlCandidates.front();
            ast->baseExpression->type = adlCandidates.front()->type();
          }
        }
      }
    }
  }

  auto functionType = resolve_function_type(ast);
  if (!functionType) {
    if (type_cast<OverloadSetType>(ast->baseExpression->type)) return;

    if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
        idExpr && !idExpr->symbol) {
      if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
        auto identifier = nameId->identifier;
        auto name = identifier ? identifier->value() : std::string{};

        if (std::string_view{name}.starts_with("__builtin_")) {
          bool report = true;
          if (auto preprocessor = check.unit_->preprocessor()) {
            const auto& token =
                check.unit_->tokenAt(idExpr->firstSourceLocation());
            if (token) report = !preprocessor->isSystemHeader(token.fileId());
          }

          if (!report) return;

          error(idExpr->firstSourceLocation(),
                std::format("unknown builtin function '{}'", name));
        } else {
          error(idExpr->firstSourceLocation(),
                std::format("use of undeclared identifier '{}'", name));
        }
      } else {
        error(idExpr->firstSourceLocation(), "call to unresolved identifier");
      }
      return;
    }

    if (ast->baseExpression->type) {
      error(ast->baseExpression->firstSourceLocation(),
            std::format("called object of type '{}' is not a function or "
                        "function pointer",
                        to_string(ast->baseExpression->type)));
    }

    return;
  }

  auto tryInstantiate = [&](FunctionSymbol* funcSym) {
    if (!funcSym || !funcSym->templateDeclaration() ||
        funcSym->isSpecialization())
      return;

    auto explicitTemplateArguments = [&]() -> List<TemplateArgumentAST*>* {
      if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
        if (auto templateId =
                ast_cast<SimpleTemplateIdAST>(idExpr->unqualifiedId)) {
          return templateId->templateArgumentList;
        }
        return nullptr;
      }

      auto memberExpr = ast_cast<MemberExpressionAST>(ast->baseExpression);
      if (!memberExpr) return nullptr;
      auto templateId =
          ast_cast<SimpleTemplateIdAST>(memberExpr->unqualifiedId);
      if (!templateId) return nullptr;
      return templateId->templateArgumentList;
    }();

    auto deducedArgs = deduceTemplateArguments(funcSym, ast->expressionList,
                                               explicitTemplateArguments);
    if (!deducedArgs.has_value()) return;
    auto instantiated =
        ASTRewriter::instantiate(check.unit_, *deducedArgs, funcSym,
                                 ast->baseExpression->firstSourceLocation(),
                                 /*sfinaeContext=*/true);
    if (!instantiated) return;
    auto instFunc = symbol_cast<FunctionSymbol>(instantiated);
    if (!instFunc) return;
    auto instType = type_cast<FunctionType>(instFunc->type());
    if (!instType) return;
    set_base_symbol(ast->baseExpression, instFunc);
    ast->baseExpression->type = instFunc->type();
    functionType = instType;
    if (instFunc->isSpecialization()) {
      ASTRewriter::reportPendingInstantiationErrors(
          check.unit_, instFunc->primaryTemplateSymbol(), instFunc,
          ast->baseExpression->firstSourceLocation());
    }
  };

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression))
    tryInstantiate(symbol_cast<FunctionSymbol>(idExpr->symbol));

  if (auto memberExpr = ast_cast<MemberExpressionAST>(ast->baseExpression))
    tryInstantiate(symbol_cast<FunctionSymbol>(memberExpr->symbol));

  check_call_arguments(ast, functionType);

  setResultTypeAndValueCategory(ast, functionType);

  if (ast->valueCategory == ValueCategory::kPrValue) stdconv_.adjustCv(ast);
}

void TypeChecker::Visitor::setResultTypeAndValueCategory(ExpressionAST* ast,
                                                         const Type* type) {
  auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return;
  ast->type = functionType->returnType();

  if (check.unit_->typeTraits().is_lvalue_reference(ast->type)) {
    ast->type = check.unit_->typeTraits().remove_reference(ast->type);
    ast->valueCategory = ValueCategory::kLValue;
  } else if (check.unit_->typeTraits().is_rvalue_reference(ast->type)) {
    ast->type = check.unit_->typeTraits().remove_reference(ast->type);
    ast->valueCategory = ValueCategory::kXValue;
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }
}

void TypeChecker::Visitor::operator()(TypeConstructionAST* ast) {
  if (!ast->typeSpecifier) {
    error(ast->firstSourceLocation(), "expected a type specifier");
    return;
  }

  DeclSpecs specs(check.unit_);
  specs.accept(ast->typeSpecifier);
  specs.finish();

  if (!specs.type()) {
    if (in_template()) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
    error(ast->firstSourceLocation(), "invalid type construction");
    return;
  }

  ast->type = specs.type();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(BracedTypeConstructionAST* ast) {
  if (!ast->typeSpecifier) {
    error(ast->firstSourceLocation(), "expected a type specifier");
    return;
  }

  if (!ast->bracedInitList) {
    error(ast->firstSourceLocation(), "expected a braced initializer");
    return;
  }

  DeclSpecs specs(check.unit_);
  specs.accept(ast->typeSpecifier);
  specs.finish();

  if (!specs.type()) {
    if (in_template()) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
    error(ast->firstSourceLocation(), "invalid braced type construction");
    return;
  }

  ast->type = specs.type();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(SpliceMemberExpressionAST* ast) {
  if (!ast->baseExpression) {
    error(ast->firstSourceLocation(), "expected a base expression");
    return;
  }

  if (!ast->splicer) {
    error(ast->firstSourceLocation(), "expected a splicer");
    return;
  }

  if (ast->symbol) {
    ast->type = check.unit_->typeTraits().remove_reference(ast->symbol->type());
    if (ast->symbol->isEnumerator() || ast->symbol->isNonTypeParameter()) {
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      ast->valueCategory = ValueCategory::kLValue;
    }
    return;
  }

  if (!ast->type) ast->type = ast->baseExpression->type;
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(MemberExpressionAST* ast) {
  if (!ast->baseExpression) {
    error(ast->firstSourceLocation(), "expected a base expression");
    return;
  }

  if (report_unresolved_id(ast->baseExpression)) return;
  if (!ast->baseExpression->type) return;

  if (is_dependent_type(ast->baseExpression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    if (auto pointerType = type_cast<PointerType>(ast->baseExpression->type)) {
      if (is_dependent_type(pointerType->elementType())) {
        ast->type = dependent_type();
        ast->valueCategory = ValueCategory::kLValue;
        return;
      }
    }
  }

  if (ast->symbol && ast->type && !is_dependent_type(ast->type)) {
    if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
      (void)stdconv_.ensurePrvalue(ast->baseExpression);
    }
    return;
  }

  if (check_pseudo_destructor_access(ast)) return;
  if (check_member_access(ast)) return;

  error(ast->firstSourceLocation(),
        std::format("invalid member access into expression of type '{}'",
                    to_string(ast->baseExpression->type)));
}

void TypeChecker::Visitor::operator()(PostIncrExpressionAST* ast) {
  if (check.unit_->typeTraits().is_class(ast->baseExpression->type)) {
    if (auto operatorFunc = check.lookupOperator(
            ast->baseExpression->type, ast->op, control()->getIntType())) {
      ast->symbol = operatorFunc;
      setResultTypeAndValueCategory(ast, operatorFunc->type());
      return;
    }

    if (check.wasLastOperatorLookupAmbiguous()) {
      error(ast->opLoc,
            std::format("call to overloaded operator '{}' is ambiguous",
                        Token::spell(ast->op)));
      return;
    }
  }

  const std::string_view op =
      ast->op == TokenKind::T_PLUS_PLUS ? "increment" : "decrement";

  // builtin postfix increment operator
  if (!is_glvalue(ast->baseExpression)) {
    error(ast->opLoc, std::format("cannot {} an rvalue of type '{}'", op,
                                  to_string(ast->baseExpression->type)));
    return;
  }

  auto incr_arithmetic = [&]() {
    if (check.unit_->typeTraits().is_const(ast->baseExpression->type))
      return false;

    if (isCxx() &&
        !check.unit_->typeTraits().is_arithmetic(ast->baseExpression->type))
      return false;

    if (isC() && !check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(
                     ast->baseExpression->type))
      return false;

    auto ty = check.unit_->typeTraits().remove_cv(ast->baseExpression->type);
    if (type_cast<BoolType>(ty)) return false;

    ast->type = ty;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto incr_pointer = [&]() {
    if (!check.unit_->typeTraits().is_pointer(ast->baseExpression->type))
      return false;
    auto ty = check.unit_->typeTraits().remove_cv(ast->baseExpression->type);
    ast->type = ty;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  if (incr_arithmetic()) return;
  if (incr_pointer()) return;

  error(ast->opLoc, std::format("cannot {} a value of type '{}'", op,
                                to_string(ast->baseExpression->type)));
}

void TypeChecker::Visitor::classify_reference_type(ExpressionAST* ast,
                                                   const Type* fullType) {
  if (auto refType = type_cast<LvalueReferenceType>(fullType)) {
    ast->type = refType->elementType();
    ast->valueCategory = ValueCategory::kLValue;
  } else if (auto rrefType = type_cast<RvalueReferenceType>(fullType)) {
    ast->type = rrefType->elementType();
    ast->valueCategory = type_cast<FunctionType>(ast->type)
                             ? ValueCategory::kLValue
                             : ValueCategory::kXValue;
  } else {
    ast->type = fullType;
    ast->valueCategory = ValueCategory::kPrValue;
  }
}

void TypeChecker::Visitor::operator()(CppCastExpressionAST* ast) {
  if (!ast->typeId) return;

  auto fullTargetType = ast->typeId->type;
  classify_reference_type(ast, fullTargetType);

  if (is_dependent_type(ast->type) ||
      (ast->expression && is_dependent_type(ast->expression->type))) {
    return;
  }

  if (in_template() &&
      (!ast->type || !ast->expression || !ast->expression->type)) {
    return;
  }

  using CastCheck =
      bool (Visitor::*)(ExpressionAST*&, const Type*, ValueCategory);
  struct CastInfo {
    TokenKind op;
    CastCheck fn;
    const char* name;
  };

  CastInfo casts[] = {
      {TokenKind::T_STATIC_CAST, &Visitor::check_static_cast, "static_cast"},
      {TokenKind::T_CONST_CAST, &Visitor::check_const_cast, "const_cast"},
      {TokenKind::T_REINTERPRET_CAST, &Visitor::check_reinterpret_cast,
       "reinterpret_cast"},
  };

  for (auto& [op, fn, name] : casts) {
    if (ast->castOp != op) continue;
    if ((this->*fn)(ast->expression, ast->type, ast->valueCategory)) break;
    error(ast->firstSourceLocation(),
          std::format("invalid {} of '{}' to '{}'", name,
                      to_string(ast->expression->type),
                      to_string(fullTargetType)));
    break;
  }

  if (ast->castOp == TokenKind::T_DYNAMIC_CAST)
    warning(ast->firstSourceLocation(), "dynamic_cast is not supported yet");

  if (ast->valueCategory == ValueCategory::kPrValue) stdconv_.adjustCv(ast);
}

auto TypeChecker::Visitor::check_static_cast(ExpressionAST*& expression,
                                             const Type* targetType,
                                             ValueCategory targetVC) -> bool {
  if (!expression || !expression->type) return false;

  if (check.unit_->typeTraits().is_void(targetType)) return true;

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (check_static_cast_to_derived_ref(expression, targetType, targetVC))
      return true;
  }

  if (targetVC == ValueCategory::kXValue && is_lvalue(expression)) {
    if (is_reference_compatible(targetType, expression->type)) return true;
  }

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (is_glvalue(expression) &&
        is_reference_compatible(targetType, expression->type))
      return true;
  }

  if (targetVC == ValueCategory::kPrValue) {
    if (implicit_conversion(expression, targetType)) return true;
  }

  auto source = expression;
  (void)stdconv_.ensurePrvalue(source);
  stdconv_.adjustCv(source);

  auto sourceType = source->type;

  if (check.unit_->typeTraits().is_scoped_enum(sourceType) &&
      (check.unit_->typeTraits().is_integral(targetType) ||
       check.unit_->typeTraits().is_floating_point(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       check.unit_->typeTraits().is_integral(targetType)
                           ? ImplicitCastKind::kIntegralConversion
                           : ImplicitCastKind::kFloatingIntegralConversion);
    return true;
  }

  if ((check.unit_->typeTraits().is_integral(sourceType) ||
       check.unit_->typeTraits().is_enum(sourceType) ||
       check.unit_->typeTraits().is_scoped_enum(sourceType)) &&
      (check.unit_->typeTraits().is_enum(targetType) ||
       check.unit_->typeTraits().is_scoped_enum(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kIntegralConversion);
    return true;
  }

  if (check.unit_->typeTraits().is_floating_point(sourceType) &&
      (check.unit_->typeTraits().is_enum(targetType) ||
       check.unit_->typeTraits().is_scoped_enum(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kFloatingIntegralConversion);
    return true;
  }

  if (check.unit_->typeTraits().is_floating_point(sourceType) &&
      check.unit_->typeTraits().is_floating_point(targetType)) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kFloatingPointConversion);
    return true;
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (auto targetPtr = as_pointer(targetType)) {
      auto srcElem =
          check.unit_->typeTraits().remove_cv(sourcePtr->elementType());
      auto tgtElem =
          check.unit_->typeTraits().remove_cv(targetPtr->elementType());
      auto srcCV =
          check.unit_->typeTraits().get_cv_qualifiers(sourcePtr->elementType());
      auto tgtCV =
          check.unit_->typeTraits().get_cv_qualifiers(targetPtr->elementType());
      if (check.unit_->typeTraits().is_base_of(srcElem, tgtElem) &&
          stdconv_.checkCvQualifiers(tgtCV, srcCV)) {
        emit_implicit_cast(expression, source, targetType,
                           ImplicitCastKind::kPointerConversion);
        return true;
      }
    }
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (check.unit_->typeTraits().is_void(
            check.unit_->typeTraits().remove_cv(sourcePtr->elementType()))) {
      if (auto targetPtr = as_pointer(targetType)) {
        if (check.unit_->typeTraits().is_object(
                check.unit_->typeTraits().remove_cv(
                    targetPtr->elementType()))) {
          auto srcCV = check.unit_->typeTraits().get_cv_qualifiers(
              sourcePtr->elementType());
          auto tgtCV = check.unit_->typeTraits().get_cv_qualifiers(
              targetPtr->elementType());
          if (stdconv_.checkCvQualifiers(tgtCV, srcCV)) {
            expression = source;
            return true;
          }
        }
      }
    }
  }

  if (auto srcMem = type_cast<MemberObjectPointerType>(
          check.unit_->typeTraits().remove_cv(sourceType))) {
    if (auto tgtMem = type_cast<MemberObjectPointerType>(
            check.unit_->typeTraits().remove_cv(targetType))) {
      auto srcClass = srcMem->classType();
      auto tgtClass = tgtMem->classType();
      if (check.unit_->typeTraits().is_base_of(tgtClass, srcClass)) {
        auto srcElemCV =
            check.unit_->typeTraits().get_cv_qualifiers(srcMem->elementType());
        auto tgtElemCV =
            check.unit_->typeTraits().get_cv_qualifiers(tgtMem->elementType());
        if (stdconv_.checkCvQualifiers(tgtElemCV, srcElemCV) &&
            check.unit_->typeTraits().is_same(
                check.unit_->typeTraits().remove_cv(srcMem->elementType()),
                check.unit_->typeTraits().remove_cv(tgtMem->elementType()))) {
          emit_implicit_cast(expression, source, targetType,
                             ImplicitCastKind::kPointerToMemberConversion);
          return true;
        }
      }
    }
  }

  return false;
}

auto TypeChecker::Visitor::check_static_cast_to_derived_ref(
    ExpressionAST*& expression, const Type* targetType, ValueCategory targetVC)
    -> bool {
  if (!is_glvalue(expression)) return false;

  if (targetVC == ValueCategory::kLValue && !is_lvalue(expression))
    return false;

  auto sourceType = expression->type;
  auto srcCV = check.unit_->typeTraits().get_cv_qualifiers(sourceType);
  sourceType = check.unit_->typeTraits().remove_cv(sourceType);

  auto tgtCV = check.unit_->typeTraits().get_cv_qualifiers(targetType);
  auto tgtBase = check.unit_->typeTraits().remove_cv(targetType);

  if (!stdconv_.checkCvQualifiers(tgtCV, srcCV)) return false;

  if (!check.unit_->typeTraits().is_base_of(sourceType, tgtBase)) return false;

  return true;
}

auto TypeChecker::Visitor::is_reference_compatible(const Type* targetType,
                                                   const Type* sourceType)
    -> bool {
  auto t1 = check.unit_->typeTraits().remove_cv(targetType);
  auto t2 = check.unit_->typeTraits().remove_cv(sourceType);
  if (!check.unit_->typeTraits().is_same(t1, t2)) {
    if (!check.unit_->typeTraits().is_base_of(t1, t2)) return false;
  }
  auto cvTarget = check.unit_->typeTraits().get_cv_qualifiers(targetType);
  auto cvSource = check.unit_->typeTraits().get_cv_qualifiers(sourceType);
  return stdconv_.checkCvQualifiers(cvTarget, cvSource);
}

auto TypeChecker::Visitor::check_const_cast(ExpressionAST*& expression,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool {
  if (!targetType) return false;

  auto sourceType = expression->type;
  const Type* T1 = nullptr;
  const Type* T2 = nullptr;

  if (auto targetPtr = type_cast<PointerType>(
          check.unit_->typeTraits().remove_cv(targetType))) {
    auto sourcePtr =
        type_cast<PointerType>(check.unit_->typeTraits().remove_cv(sourceType));
    if (!sourcePtr) return false;

    (void)stdconv_.ensurePrvalue(expression);
    stdconv_.adjustCv(expression);

    T1 = sourcePtr->elementType();
    T2 = targetPtr->elementType();
  } else if (auto targetPtrm = type_cast<MemberObjectPointerType>(
                 check.unit_->typeTraits().remove_cv(targetType))) {
    auto sourcePtrm = type_cast<MemberObjectPointerType>(
        check.unit_->typeTraits().remove_cv(sourceType));
    if (!sourcePtrm) return false;

    if (!check.unit_->typeTraits().is_same(sourcePtrm->classType(),
                                           targetPtrm->classType()))
      return false;

    (void)stdconv_.ensurePrvalue(expression);
    stdconv_.adjustCv(expression);

    T1 = sourcePtrm->elementType();
    T2 = targetPtrm->elementType();
  } else if (targetVC == ValueCategory::kLValue) {
    if (!is_lvalue(expression)) return false;
    T1 = sourceType;
    T2 = targetType;
  } else if (targetVC == ValueCategory::kXValue) {
    if (is_glvalue(expression)) {
      T1 = sourceType;
      T2 = targetType;
    } else if (is_prvalue(expression) &&
               (check.unit_->typeTraits().is_class(sourceType) ||
                check.unit_->typeTraits().is_array(sourceType))) {
      (void)stdconv_.temporaryMaterialization(expression);
      T1 = expression->type;
      T2 = targetType;
    } else {
      return false;
    }
  } else {
    return false;
  }

  if (!T1 || !T2) return false;

  return are_similar_types(T1, T2);
}

auto TypeChecker::Visitor::are_similar_types(const Type* T1, const Type* T2)
    -> bool {
  const Type* curr1 = T1;
  const Type* curr2 = T2;

  while (true) {
    if (check.unit_->typeTraits().is_same(
            check.unit_->typeTraits().remove_cv(curr1),
            check.unit_->typeTraits().remove_cv(curr2))) {
      return true;
    }

    auto u1 = check.unit_->typeTraits().remove_cv(curr1);
    auto u2 = check.unit_->typeTraits().remove_cv(curr2);

    if (auto p1 = as_pointer(u1)) {
      if (auto p2 = as_pointer(u2)) {
        curr1 = p1->elementType();
        curr2 = p2->elementType();
        continue;
      }
    }

    if (auto m1 = type_cast<MemberObjectPointerType>(u1)) {
      if (auto m2 = type_cast<MemberObjectPointerType>(u2)) {
        if (!check.unit_->typeTraits().is_same(m1->classType(),
                                               m2->classType()))
          return false;
        curr1 = m1->elementType();
        curr2 = m2->elementType();
        continue;
      }
    }

    return false;
  }
}

auto TypeChecker::Visitor::check_reinterpret_cast(ExpressionAST*& expression,
                                                  const Type* targetType,
                                                  ValueCategory targetVC)
    -> bool {
  if (!expression || !expression->type) return false;

  auto sourceType = expression->type;

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (!is_glvalue(expression)) return false;
    auto ptrToSource = check.unit_->typeTraits().add_pointer(sourceType);
    auto ptrToTarget = check.unit_->typeTraits().add_pointer(targetType);
    (void)ptrToSource;
    (void)ptrToTarget;
    if ((check.unit_->typeTraits().is_object(
             check.unit_->typeTraits().remove_cv(sourceType)) &&
         check.unit_->typeTraits().is_object(
             check.unit_->typeTraits().remove_cv(targetType))) ||
        (check.unit_->typeTraits().is_function(sourceType) &&
         check.unit_->typeTraits().is_function(targetType))) {
      if (casts_away_constness(sourceType, targetType)) return false;
      return true;
    }
    return false;
  }

  (void)stdconv_.ensurePrvalue(expression);
  stdconv_.adjustCv(expression);
  sourceType = expression->type;

  if (check.unit_->typeTraits().is_same(
          check.unit_->typeTraits().remove_cv(sourceType),
          check.unit_->typeTraits().remove_cv(targetType)))
    return true;

  if (check.unit_->typeTraits().is_pointer(sourceType) &&
      check.unit_->typeTraits().is_integral(targetType)) {
    emit_implicit_cast(expression, expression, targetType,
                       ImplicitCastKind::kIntegralConversion);
    return true;
  }

  if ((check.unit_->typeTraits().is_integral(sourceType) ||
       check.unit_->typeTraits().is_enum(sourceType) ||
       check.unit_->typeTraits().is_scoped_enum(sourceType)) &&
      check.unit_->typeTraits().is_pointer(targetType)) {
    // No implicit cast wrapper - CastExpressionAST codegen emits IntToPtrOp.
    return true;
  }

  if (check.unit_->typeTraits().is_pointer(sourceType) &&
      check.unit_->typeTraits().is_pointer(targetType)) {
    auto srcPtr = as_pointer(sourceType);
    auto tgtPtr = as_pointer(targetType);
    if (srcPtr && tgtPtr &&
        casts_away_constness(srcPtr->elementType(), tgtPtr->elementType()))
      return false;
    if (!check.unit_->typeTraits().is_same(
            check.unit_->typeTraits().remove_cv(sourceType),
            check.unit_->typeTraits().remove_cv(targetType)))
      emit_implicit_cast(expression, expression, targetType,
                         ImplicitCastKind::kPointerConversion);
    return true;
  }

  if (check.unit_->typeTraits().is_member_pointer(sourceType) &&
      check.unit_->typeTraits().is_member_pointer(targetType)) {
    return true;
  }

  if (check.unit_->typeTraits().is_null_pointer(sourceType) &&
      check.unit_->typeTraits().is_integral(targetType))
    return true;

  return false;
}

auto TypeChecker::Visitor::casts_away_constness(const Type* sourceType,
                                                const Type* targetType)
    -> bool {
  auto srcCV = check.unit_->typeTraits().get_cv_qualifiers(sourceType);
  auto tgtCV = check.unit_->typeTraits().get_cv_qualifiers(targetType);

  if (!stdconv_.checkCvQualifiers(tgtCV, srcCV)) return true;

  auto srcPtr = as_pointer(check.unit_->typeTraits().remove_cv(sourceType));
  auto tgtPtr = as_pointer(check.unit_->typeTraits().remove_cv(targetType));
  if (srcPtr && tgtPtr) {
    return casts_away_constness(srcPtr->elementType(), tgtPtr->elementType());
  }

  return false;
}

auto TypeChecker::Visitor::check_cast_to_derived(ExpressionAST* expression,
                                                 const Type* targetType)
    -> bool {
  return check_static_cast_to_derived_ref(expression, targetType,
                                          ValueCategory::kLValue);
}

void TypeChecker::Visitor::operator()(BuiltinOffsetofExpressionAST* ast) {
  ast->type = control()->getSizeType();

  auto classType =
      ast->typeId ? type_cast<ClassType>(
                        check.unit_->typeTraits().remove_cv(ast->typeId->type))
                  : nullptr;

  if (!classType) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  if (!ast->identifier) {
    return;
  }

  auto symbol = classType->symbol();
  auto member = qualifiedLookup(symbol, ast->identifier);

  auto field = symbol_cast<FieldSymbol>(member);
  if (!field) {
    error(ast->firstSourceLocation(),
          std::format("no member named '{}'", ast->identifier->name()));
    return;
  }

  for (auto designator : ListView{ast->designatorList}) {
    if (auto dot = ast_cast<DotDesignatorAST>(designator);
        dot && dot->identifier) {
      // resolve the field in the current class scope
      auto currentClass = type_cast<ClassType>(
          check.unit_->typeTraits().remove_cvref(field->type()));

      if (!currentClass) {
        error(designator->firstSourceLocation(),
              std::format("expected a class or union type, but got '{}'",
                          to_string(field->type())));
        break;
      }

      auto member = qualifiedLookup(currentClass->symbol(), dot->identifier);

      auto field = symbol_cast<FieldSymbol>(member);

      if (!field) {
        error(dot->firstSourceLocation(),
              std::format("no member named '{}' in class '{}'",
                          dot->identifier->name(),
                          to_string(currentClass->symbol()->name())));
      }

      break;
    }

    if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      if (!check.unit_->typeTraits().is_array(field->type()) &&
          !check.unit_->typeTraits().is_pointer(field->type())) {
        error(subscript->firstSourceLocation(),
              std::format("cannot subscript a member of type '{}'",
                          to_string(field->type())));
        break;
      }

      // todo update offset

      continue;
    }
  }

  ast->symbol = field;
}

void TypeChecker::Visitor::operator()(BuiltinBitCastExpressionAST* ast) {
  if (!ast->typeId || !ast->typeId->type) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  if (!ast->expression || !ast->expression->type) return;

  auto targetType = check.unit_->typeTraits().remove_cv(ast->typeId->type);
  auto sourceType = check.unit_->typeTraits().remove_cv(ast->expression->type);

  if (is_dependent_type(targetType) || is_dependent_type(sourceType)) {
    ast->type = ast->typeId->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (check.unit_->typeTraits().is_reference(targetType) ||
      check.unit_->typeTraits().is_reference(sourceType)) {
    error(ast->firstSourceLocation(),
          "__builtin_bit_cast does not support reference types");
    return;
  }

  auto sourceSize = control()->memoryLayout()->sizeOf(sourceType);
  auto targetSize = control()->memoryLayout()->sizeOf(targetType);
  if (!sourceSize || !targetSize || *sourceSize != *targetSize) {
    error(ast->firstSourceLocation(),
          "__builtin_bit_cast requires source and destination to have the same "
          "size");
    return;
  }

  ast->type = ast->typeId->type;
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(TypeidExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->expression->type) {
    if (auto idExpr = ast_cast<IdExpressionAST>(ast->expression);
        idExpr && !idExpr->symbol && !idExpr->nestedNameSpecifier) {
      if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
        auto identifier = nameId->identifier;
        auto name = identifier ? identifier->value() : std::string{};
        error(idExpr->firstSourceLocation(),
              std::format("use of undeclared identifier '{}'", name));
      } else {
        error(idExpr->firstSourceLocation(), "invalid operand to typeid");
      }
    } else {
      error(ast->expression->firstSourceLocation(),
            "invalid operand to typeid");
    }
    return;
  }

  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(TypeidOfTypeExpressionAST* ast) {
  if (!ast->typeId || !ast->typeId->type) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(SpliceExpressionAST* ast) {
  if (!ast->splicer) {
    error(ast->firstSourceLocation(), "expected a splicer");
    return;
  }

  if (!ast->splicer->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->splicer->expression->type) {
    error(ast->splicer->firstSourceLocation(), "invalid splicer expression");
    return;
  }

  ast->type = ast->splicer->expression->type;
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(GlobalScopeReflectExpressionAST* ast) {
  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(NamespaceReflectExpressionAST* ast) {
  if (!ast->identifier || !ast->symbol) {
    error(ast->firstSourceLocation(), "expected a namespace name");
    return;
  }

  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(TypeIdReflectExpressionAST* ast) {
  if (!ast->typeId || !ast->typeId->type) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(ReflectExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->expression->type) {
    if (auto idExpr = ast_cast<IdExpressionAST>(ast->expression);
        idExpr && !idExpr->symbol && !idExpr->nestedNameSpecifier) {
      if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
        auto identifier = nameId->identifier;
        auto name = identifier ? identifier->value() : std::string{};
        error(idExpr->firstSourceLocation(),
              std::format("use of undeclared identifier '{}'", name));
      } else {
        error(idExpr->firstSourceLocation(), "invalid operand to reflection");
      }
    } else {
      error(ast->expression->firstSourceLocation(),
            "invalid operand to reflection");
    }
    return;
  }

  ast->type = control()->getBuiltinMetaInfoType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(LabelAddressExpressionAST* ast) {
  if (!ast->identifier) {
    error(ast->firstSourceLocation(), "expected a label identifier");
    return;
  }

  ast->type = control()->getPointerType(control()->getVoidType());
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::check_address_of(UnaryExpressionAST* ast) {
  if (!ast->expression->type) return;

  if (!is_glvalue(ast->expression)) {
    error(ast->opLoc,
          std::format("cannot take the address of an rvalue of type '{}'",
                      to_string(ast->expression->type)));
    return;
  }

  auto idExpr = ast_cast<IdExpressionAST>(ast->expression);
  if (idExpr && idExpr->nestedNameSpecifier) {
    auto symbol = idExpr->symbol;

    if (auto field = symbol_cast<FieldSymbol>(symbol);
        field && !field->isStatic()) {
      auto classType = type_cast<ClassType>(field->parent()->type());
      ast->type =
          control()->getMemberObjectPointerType(classType, field->type());
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }

    if (auto function = symbol_cast<FunctionSymbol>(symbol);
        function && !function->isStatic()) {
      auto functionType = type_cast<FunctionType>(function->type());
      auto classType = type_cast<ClassType>(function->parent()->type());
      ast->type =
          control()->getMemberFunctionPointerType(classType, functionType);
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }
  }

  ast->type = control()->getPointerType(ast->expression->type);
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::check_unary_promote(UnaryExpressionAST* ast) {
  ExpressionAST* expr = ast->expression;
  (void)stdconv_.ensurePrvalue(expr);
  stdconv_.adjustCv(expr);

  bool valid = false;
  switch (ast->op) {
    case TokenKind::T_PLUS:
      valid = check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(
                  expr->type) ||
              check.unit_->typeTraits().is_pointer(expr->type);
      break;
    case TokenKind::T_MINUS:
      valid =
          check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(expr->type);
      break;
    case TokenKind::T_TILDE:
      valid =
          check.unit_->typeTraits().is_integral_or_unscoped_enum(expr->type);
      break;
    default:
      return;
  }

  if (!valid) return;

  if (check.unit_->typeTraits().is_integral_or_unscoped_enum(expr->type))
    (void)stdconv_.integralPromotion(expr);

  ast->expression = expr;
  ast->type = expr->type;
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(UnaryExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (report_unresolved_id(ast->expression)) return;
  if (!ast->expression->type) return;

  if (is_dependent_type(ast->expression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (resolve_unary_overload(ast)) return;

  switch (ast->op) {
    case TokenKind::T_STAR:
      (void)stdconv_.ensurePrvalue(ast->expression);
      if (auto pointerType = as_pointer(ast->expression->type)) {
        stdconv_.adjustCv(ast->expression);
        ast->type = pointerType->elementType();
        ast->valueCategory = ValueCategory::kLValue;
      }
      break;

    case TokenKind::T_AMP_AMP:
      cxx_runtime_error("address of label");
      ast->type = control()->getPointerType(control()->getVoidType());
      ast->valueCategory = ValueCategory::kPrValue;
      break;

    case TokenKind::T_AMP:
      check_address_of(ast);
      break;

    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_TILDE:
      check_unary_promote(ast);
      break;

    case TokenKind::T_EXCLAIM:
      (void)implicit_conversion(ast->expression, control()->getBoolType());
      ast->type = control()->getBoolType();
      ast->valueCategory = ValueCategory::kPrValue;
      break;

    case TokenKind::T_PLUS_PLUS:
      check_prefix_increment_decrement(ast, "increment", "increment");
      break;

    case TokenKind::T_MINUS_MINUS:
      check_prefix_increment_decrement(ast, "decrement", "decrement");
      break;

    default:
      break;
  }
}

void TypeChecker::Visitor::operator()(AwaitExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->expression->type) {
    error(ast->firstSourceLocation(), "invalid operand to co_await");
    return;
  }

  if (!ast->type) ast->type = ast->expression->type;
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(SizeofExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;

  if (ast->expression) {
    ast->value = control()->memoryLayout()->sizeOf(ast->expression->type);
  }
}

void TypeChecker::Visitor::operator()(SizeofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;

  if (ast->typeId) {
    ast->value = control()->memoryLayout()->sizeOf(ast->typeId->type);
  }
}

void TypeChecker::Visitor::operator()(SizeofPackExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(AlignofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(AlignofExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(NoexceptExpressionAST* ast) {
  ast->type = control()->getBoolType();
  ast->valueCategory = ValueCategory::kPrValue;

  if (ast->expression && ast->expression->type) {
    ast->value = !isPotentiallyThrowing(ast->expression);
  }
}

void TypeChecker::Visitor::operator()(NewExpressionAST* ast) {
  // TODO: decay
  auto objectType = check.unit_->typeTraits().remove_reference(ast->objectType);

  if (auto arrayType = type_cast<BoundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(arrayType->elementType());
  } else if (auto unboundedType =
                 type_cast<UnboundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(unboundedType->elementType());
  } else {
    ast->type = control()->getPointerType(ast->objectType);
  }

  ast->valueCategory = ValueCategory::kPrValue;

  if (auto classType = type_cast<ClassType>(objectType)) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return;

    std::vector<ExpressionAST*> args;
    if (ast->newInitalizer) {
      if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
        for (auto it = paren->expressionList; it; it = it->next)
          args.push_back(it->value);
      } else if (auto braced =
                     ast_cast<NewBracedInitializerAST>(ast->newInitalizer)) {
        if (braced->bracedInitList) {
          for (auto it = braced->bracedInitList->expressionList; it;
               it = it->next)
            args.push_back(it->value);
        }
      }
    }

    std::vector<Candidate> candidates;

    OverloadResolution resolution(check.unit_);

    for (auto ctor : classSymbol->constructors()) {
      if (ctor->canonical() != ctor) continue;

      auto type = type_cast<FunctionType>(ctor->type());
      if (!type) continue;

      if (type->parameterTypes().size() != args.size()) continue;

      Candidate cand{ctor};
      cand.viable = true;

      auto paramIt = type->parameterTypes().begin();
      for (auto arg : args) {
        auto paramType = *paramIt++;
        auto conv =
            resolution.computeImplicitConversionSequence(arg, paramType);
        if (conv.rank == ConversionRank::kNone) {
          cand.viable = false;
          break;
        }
        cand.conversions.push_back(conv);
      }

      if (cand.viable) candidates.push_back(cand);
    }

    if (auto result = resolution.selectBestViableFunction(candidates);
        result.best) {
      ast->constructorSymbol = result.best->symbol;

      for (size_t i = 0; i < args.size(); ++i) {
        resolution.applyImplicitConversion(result.best->conversions[i],
                                           args[i]);
      }
    }
  }
}

void TypeChecker::Visitor::operator()(DeleteExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

auto TypeChecker::Visitor::try_c_style_cast(CastExpressionAST* ast,
                                            ExpressionAST*& expr,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool {
  if (check_const_cast(expr, targetType, targetVC)) return true;
  if (check_static_cast(expr, targetType, targetVC)) return true;
  if (check.unit_->typeTraits().is_pointer(targetType) &&
      check.unit_->typeTraits().is_pointer(expr->type))
    if (check_static_cast(expr, targetType, targetVC)) return true;
  if (check_reinterpret_cast(expr, targetType, targetVC)) return true;
  if (check_reinterpret_cast_permissive(expr, targetType, targetVC))
    return true;
  return false;
}

void TypeChecker::Visitor::operator()(CastExpressionAST* ast) {
  if (!ast->typeId) return;

  classify_reference_type(ast, ast->typeId->type);

  auto expr = ast->expression;
  if (try_c_style_cast(ast, expr, ast->type, ast->valueCategory))
    ast->expression = expr;

  if (ast->valueCategory == ValueCategory::kPrValue) stdconv_.adjustCv(ast);
}

auto TypeChecker::Visitor::check_reinterpret_cast_permissive(
    ExpressionAST*& expression, const Type* targetType, ValueCategory targetVC)
    -> bool {
  if (!expression || !expression->type) return false;

  auto sourceType = expression->type;

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (!is_glvalue(expression)) return false;
    if ((check.unit_->typeTraits().is_object(
             check.unit_->typeTraits().remove_cv(sourceType)) &&
         check.unit_->typeTraits().is_object(
             check.unit_->typeTraits().remove_cv(targetType))) ||
        (check.unit_->typeTraits().is_function(sourceType) &&
         check.unit_->typeTraits().is_function(targetType))) {
      return true;
    }
    return false;
  }

  (void)stdconv_.ensurePrvalue(expression);
  stdconv_.adjustCv(expression);
  sourceType = expression->type;

  if (check.unit_->typeTraits().is_same(
          check.unit_->typeTraits().remove_cv(sourceType),
          check.unit_->typeTraits().remove_cv(targetType)))
    return true;

  if (check.unit_->typeTraits().is_pointer(sourceType) &&
      check.unit_->typeTraits().is_integral(targetType))
    return true;

  if ((check.unit_->typeTraits().is_integral(sourceType) ||
       check.unit_->typeTraits().is_enum(sourceType) ||
       check.unit_->typeTraits().is_scoped_enum(sourceType)) &&
      check.unit_->typeTraits().is_pointer(targetType))
    return true;

  if (check.unit_->typeTraits().is_pointer(sourceType) &&
      check.unit_->typeTraits().is_pointer(targetType)) {
    if (!check.unit_->typeTraits().is_same(
            check.unit_->typeTraits().remove_cv(sourceType),
            check.unit_->typeTraits().remove_cv(targetType)))
      emit_implicit_cast(expression, expression, targetType,
                         ImplicitCastKind::kPointerConversion);
    return true;
  }

  if (check.unit_->typeTraits().is_member_pointer(sourceType) &&
      check.unit_->typeTraits().is_member_pointer(targetType))
    return true;

  if (check.unit_->typeTraits().is_null_pointer(sourceType) &&
      check.unit_->typeTraits().is_integral(targetType))
    return true;

  return false;
}

void TypeChecker::Visitor::operator()(ImplicitCastExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (ast->castKind == ImplicitCastKind::kLValueToRValueConversion) {
    ast->type =
        check.unit_->typeTraits().remove_reference(ast->expression->type);
  } else if (!ast->type || is_dependent_type(ast->type)) {
    ast->type = ast->expression->type;
  }

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::prepare_comparison_operands(
    BinaryExpressionAST* ast) {
  (void)stdconv_.lvalueToRvalue(ast->leftExpression);
  (void)stdconv_.functionToPointer(ast->leftExpression);
  (void)stdconv_.lvalueToRvalue(ast->rightExpression);
  (void)stdconv_.functionToPointer(ast->rightExpression);
}

void TypeChecker::Visitor::check_shift(BinaryExpressionAST* ast) {
  if (check.unit_->typeTraits().is_class_or_union(ast->leftExpression->type) ||
      check.unit_->typeTraits().is_class_or_union(ast->rightExpression->type)) {
    if (resolve_binary_overload(ast)) return;
    error(
        ast->opLoc,
        std::format("'operator {}' is not defined for types {} and {}",
                    Token::spell(ast->op), to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
    return;
  }

  (void)stdconv_.ensurePrvalue(ast->leftExpression);
  stdconv_.adjustCv(ast->leftExpression);
  (void)stdconv_.ensurePrvalue(ast->rightExpression);
  stdconv_.adjustCv(ast->rightExpression);
  (void)stdconv_.integralPromotion(ast->leftExpression);
  (void)stdconv_.integralPromotion(ast->rightExpression);

  if (!check.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->leftExpression->type) ||
      !check.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->rightExpression->type)) {
    error(ast->firstSourceLocation(),
          std::format("invalid operands to binary expression ('{}' and '{}')",
                      to_string(ast->leftExpression->type),
                      to_string(ast->rightExpression->type)));
    return;
  }

  ast->type = ast->leftExpression->type;
}

void TypeChecker::Visitor::check_relational(BinaryExpressionAST* ast) {
  ast->type = control()->getBoolType();

  if (resolve_binary_overload(ast)) return;

  prepare_comparison_operands(ast);

  if (isC()) {
    (void)stdconv_.arrayToPointer(ast->leftExpression);
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  }

  if (check.unit_->typeTraits().is_pointer(ast->leftExpression->type))
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  else if (check.unit_->typeTraits().is_pointer(ast->rightExpression->type))
    (void)stdconv_.arrayToPointer(ast->leftExpression);

  if (stdconv_.usualArithmeticConversion(ast->leftExpression,
                                         ast->rightExpression)) {
    ast->type = control()->getBoolType();
    return;
  }

  if (check.unit_->typeTraits().is_scoped_enum(ast->leftExpression->type)) {
    if (check.unit_->typeTraits().is_same(
            check.unit_->typeTraits().remove_cv(ast->leftExpression->type),
            check.unit_->typeTraits().remove_cv(ast->rightExpression->type))) {
      return;
    }
  }

  if (check.unit_->typeTraits().is_pointer(ast->leftExpression->type) &&
      check.unit_->typeTraits().is_pointer(ast->rightExpression->type)) {
    auto compositeType = stdconv_.compositePointerType(ast->leftExpression,
                                                       ast->rightExpression);
    (void)implicit_conversion(ast->leftExpression, compositeType);
    (void)implicit_conversion(ast->rightExpression, compositeType);
    return;
  }

  error(ast->firstSourceLocation(),
        std::format("invalid operands to binary expression ('{}' and '{}')",
                    to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
}

void TypeChecker::Visitor::check_equality(BinaryExpressionAST* ast) {
  ast->type = control()->getBoolType();

  if (resolve_binary_overload(ast, false)) return;

  prepare_comparison_operands(ast);

  if (isC()) {
    (void)stdconv_.arrayToPointer(ast->leftExpression);
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  }

  if (check.unit_->typeTraits().is_pointer(ast->leftExpression->type) ||
      stdconv_.isNullPointerConstant(ast->leftExpression))
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  else if (check.unit_->typeTraits().is_pointer(ast->rightExpression->type) ||
           stdconv_.isNullPointerConstant(ast->rightExpression))
    (void)stdconv_.arrayToPointer(ast->leftExpression);

  if (stdconv_.usualArithmeticConversion(ast->leftExpression,
                                         ast->rightExpression)) {
    ast->type = control()->getBoolType();
    return;
  }

  // Scoped enum equality: both sides must be the same scoped enum type.
  {
    auto leftBase =
        check.unit_->typeTraits().remove_cv(ast->leftExpression->type);
    auto rightBase =
        check.unit_->typeTraits().remove_cv(ast->rightExpression->type);
    if (check.unit_->typeTraits().is_scoped_enum(leftBase) &&
        check.unit_->typeTraits().is_same(leftBase, rightBase))
      return;
  }

  if ((check.unit_->typeTraits().is_pointer(ast->leftExpression->type) ||
       stdconv_.isNullPointerConstant(ast->leftExpression)) &&
      (check.unit_->typeTraits().is_pointer(ast->rightExpression->type) ||
       stdconv_.isNullPointerConstant(ast->rightExpression))) {
    auto compositeType = stdconv_.compositePointerType(ast->leftExpression,
                                                       ast->rightExpression);
    (void)implicit_conversion(ast->leftExpression, compositeType);
    (void)implicit_conversion(ast->rightExpression, compositeType);
    return;
  }

  error(ast->firstSourceLocation(),
        std::format("invalid operands to binary expression ('{}' and '{}')",
                    to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
}

void TypeChecker::Visitor::operator()(BinaryExpressionAST* ast) {
  if (!ast->leftExpression) {
    error(ast->firstSourceLocation(), "expected a left operand");
    return;
  }

  if (!ast->rightExpression) {
    error(ast->firstSourceLocation(), "expected a right operand");
    return;
  }

  if (report_unresolved_id(ast->leftExpression) |
      report_unresolved_id(ast->rightExpression)) {
    return;
  }

  auto leftType = ast->leftExpression->type;
  auto rightType = ast->rightExpression->type;
  if (!leftType || !rightType) return;

  if (type_cast<AutoType>(check.unit_->typeTraits().remove_cvref(leftType)) ||
      type_cast<AutoType>(check.unit_->typeTraits().remove_cvref(rightType)))
    return;

  if (is_dependent_type(leftType) || is_dependent_type(rightType)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  switch (ast->op) {
    case TokenKind::T_DOT_STAR:
    case TokenKind::T_MINUS_GREATER_STAR:
      break;

    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PERCENT:
      if (resolve_binary_overload(ast)) break;
      ast->type = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                     ast->rightExpression);
      if (!ast->type) {
        error(
            ast->firstSourceLocation(),
            std::format("invalid operands to binary expression ('{}' and '{}')",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));
      }
      break;

    case TokenKind::T_PLUS:
      if (resolve_binary_overload(ast)) break;
      check_addition(ast);
      break;

    case TokenKind::T_MINUS:
      if (resolve_binary_overload(ast)) break;
      check_subtraction(ast);
      break;

    case TokenKind::T_LESS_LESS:
    case TokenKind::T_GREATER_GREATER:
      check_shift(ast);
      break;

    case TokenKind::T_LESS_EQUAL_GREATER:
      (void)stdconv_.usualArithmeticConversion(ast->leftExpression,
                                               ast->rightExpression);
      ast->type = control()->getIntType();
      break;

    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_GREATER:
      check_relational(ast);
      break;

    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
      check_equality(ast);
      break;

    case TokenKind::T_AMP:
    case TokenKind::T_CARET:
    case TokenKind::T_BAR:
      if (resolve_binary_overload(ast)) break;
      ast->type = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                     ast->rightExpression);
      if (!ast->type) {
        error(
            ast->firstSourceLocation(),
            std::format("invalid operands to binary expression ('{}' and '{}')",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));
      }
      break;

    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
      if (!implicit_conversion(ast->leftExpression, control()->getBoolType()) ||
          !implicit_conversion(ast->rightExpression,
                               control()->getBoolType())) {
        error(
            ast->firstSourceLocation(),
            std::format("invalid operands to binary expression ('{}' and '{}')",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));
        break;
      }

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
  }
}

void TypeChecker::Visitor::operator()(ConditionalExpressionAST* ast) {
  if (!ast->condition) {
    error(ast->firstSourceLocation(), "expected a condition expression");
    return;
  }

  if (!ast->iftrueExpression) {
    error(ast->firstSourceLocation(), "expected an expression after '?'");
    return;
  }

  if (!ast->iffalseExpression) {
    error(ast->firstSourceLocation(), "expected an expression after ':'");
    return;
  }

  if (report_unresolved_id(ast->condition) |
      report_unresolved_id(ast->iftrueExpression) |
      report_unresolved_id(ast->iffalseExpression)) {
    return;
  }

  if (!ast->condition->type) return;

  if (is_dependent_type(ast->condition->type) ||
      (ast->iftrueExpression->type &&
       is_dependent_type(ast->iftrueExpression->type)) ||
      (ast->iffalseExpression->type &&
       is_dependent_type(ast->iffalseExpression->type))) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (!implicit_conversion(ast->condition, control()->getBoolType())) {
    error(ast->condition->firstSourceLocation(),
          std::format("invalid condition expression of type '{}'",
                      to_string(ast->condition->type)));
    return;
  }

  auto check_void_type = [&] {
    if (!check.unit_->typeTraits().is_void(ast->iftrueExpression->type) &&
        !check.unit_->typeTraits().is_void(ast->iffalseExpression->type))
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

    if (!check.unit_->typeTraits().is_same(ast->iftrueExpression->type,
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

    if (!check.unit_->typeTraits().is_same(
            check.unit_->typeTraits().remove_cv(ast->iftrueExpression->type),
            check.unit_->typeTraits().remove_cv(ast->iffalseExpression->type)))
      return false;

    ast->type = ast->iftrueExpression->type;

    ast->valueCategory = ast->iftrueExpression->valueCategory;

    return true;
  };

  auto check_arith_types = [&] {
    if (!check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(
            ast->iftrueExpression->type))
      return false;
    if (!check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(
            ast->iffalseExpression->type))
      return false;

    ast->type = stdconv_.usualArithmeticConversion(ast->iftrueExpression,
                                                   ast->iffalseExpression);

    if (!ast->type) return false;

    ast->valueCategory = ValueCategory::kPrValue;

    return true;
  };

  auto check_same_types = [&] {
    if (!check.unit_->typeTraits().is_same(ast->iftrueExpression->type,
                                           ast->iffalseExpression->type))
      return false;

    (void)stdconv_.ensurePrvalue(ast->iftrueExpression);
    (void)stdconv_.ensurePrvalue(ast->iffalseExpression);

    ast->type = ast->iftrueExpression->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto check_compatible_pointers = [&] {
    if (!check.unit_->typeTraits().is_pointer(ast->iftrueExpression->type) &&
        !check.unit_->typeTraits().is_pointer(ast->iffalseExpression->type))
      return false;

    (void)stdconv_.ensurePrvalue(ast->iftrueExpression);
    (void)stdconv_.ensurePrvalue(ast->iffalseExpression);

    ast->type = stdconv_.compositePointerType(ast->iftrueExpression,
                                              ast->iffalseExpression);

    ast->valueCategory = ValueCategory::kPrValue;

    if (!ast->type) return false;

    auto insert_pointer_cast = [&](ExpressionAST*& expr) {
      if (!check.unit_->typeTraits().is_same(expr->type, ast->type)) {
        auto cast = ImplicitCastExpressionAST::create(arena());
        cast->castKind = ImplicitCastKind::kPointerConversion;
        cast->expression = expr;
        cast->type = ast->type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
      }
    };

    insert_pointer_cast(ast->iftrueExpression);
    insert_pointer_cast(ast->iffalseExpression);

    return true;
  };

  if (!ast->iftrueExpression) {
    error(ast->questionLoc,
          "left operand to ? is null, but right operand is not null");
    return;
  }

  if (!ast->iffalseExpression) {
    error(ast->colonLoc,
          "right operand to ? is null, but left operand is not null");
    return;
  }

  if (!ast->iftrueExpression->type || !ast->iffalseExpression->type) return;

  if (isC()) {
    // in C, both expressions must be prvalues
    (void)stdconv_.ensurePrvalue(ast->iftrueExpression);
    (void)stdconv_.ensurePrvalue(ast->iffalseExpression);
  }

  if (check_void_type()) return;
  if (check_same_type_and_value_category()) return;

  (void)stdconv_.arrayToPointer(ast->iftrueExpression);
  (void)stdconv_.functionToPointer(ast->iftrueExpression);

  (void)stdconv_.arrayToPointer(ast->iffalseExpression);
  (void)stdconv_.functionToPointer(ast->iffalseExpression);

  if (check_arith_types()) return;
  if (check_same_types()) return;
  if (check_compatible_pointers()) return;

  auto iftrueType =
      ast->iftrueExpression ? ast->iftrueExpression->type : nullptr;

  auto iffalseType =
      ast->iffalseExpression ? ast->iffalseExpression->type : nullptr;

  error(ast->questionLoc,
        std::format(
            "left operand to ? is '{}', but right operand is of type '{}'",
            to_string(iftrueType), to_string(iffalseType)));
}

void TypeChecker::Visitor::operator()(YieldExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->type) ast->type = ast->expression->type;

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::operator()(ThrowExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(AssignmentExpressionAST* ast) {
  if (!ast->leftExpression) {
    error(ast->firstSourceLocation(), "expected a left operand");
    return;
  }

  if (!ast->rightExpression) {
    error(ast->firstSourceLocation(), "expected a right operand");
    return;
  }

  if (report_unresolved_id(ast->leftExpression) |
      report_unresolved_id(ast->rightExpression)) {
    return;
  }

  if (!ast->leftExpression->type || !ast->rightExpression->type) return;

  if (is_dependent_type(ast->leftExpression->type) ||
      is_dependent_type(ast->rightExpression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (resolve_assignment_overload(ast)) return;

  if (!is_lvalue(ast->leftExpression)) {
    error(ast->opLoc, std::format("cannot assign to an rvalue of type '{}'",
                                  to_string(ast->leftExpression->type)));
    return;
  }

  ast->type = ast->leftExpression->type;

  if (isC()) {
    ast->valueCategory = ValueCategory::kPrValue;
  } else {
    ast->valueCategory = ast->leftExpression->valueCategory;
  }

  if (!implicit_conversion(ast->rightExpression, ast->type)) {
    if (!ast->rightExpression->type) return;

    if (check.unit_->typeTraits().is_class_or_union(
            check.unit_->typeTraits().remove_reference(ast->type))) {
      return;
    }

    error(ast->opLoc,
          std::format("cannot assign expression of type '{}' to '{}'",
                      to_string(ast->rightExpression->type),
                      to_string(ast->type)));
  }
}

void TypeChecker::Visitor::operator()(TargetExpressionAST* ast) {
  if (!ast->type) ast->type = control()->getVoidType();
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(RightExpressionAST* ast) {
  if (!ast->type) ast->type = control()->getVoidType();
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(CompoundAssignmentExpressionAST* ast) {
  if (!ast->targetExpression) {
    error(ast->firstSourceLocation(), "expected a target expression");
    return;
  }

  if (!ast->rightExpression) {
    error(ast->firstSourceLocation(), "expected a right operand");
    return;
  }

  if (!ast->leftExpression) {
    error(ast->firstSourceLocation(), "expected a synthesized left operand");
    return;
  }

  if (report_unresolved_id(ast->targetExpression) |
      report_unresolved_id(ast->rightExpression)) {
    return;
  }

  if (!ast->targetExpression->type || !ast->rightExpression->type) return;

  if (is_dependent_type(ast->targetExpression->type) ||
      is_dependent_type(ast->rightExpression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (resolve_compound_assignment_overload(ast)) return;

  if (!is_lvalue(ast->targetExpression)) {
    error(ast->opLoc, std::format("cannot assign to an rvalue of type '{}'",
                                  to_string(ast->targetExpression->type)));
    return;
  }

  ast->leftExpression->type = ast->targetExpression->type;
  ast->leftExpression->valueCategory = ast->targetExpression->valueCategory;
  ast->type = ast->targetExpression->type;

  if (isCxx()) {
    ast->valueCategory = ValueCategory::kLValue;
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }

  if ((ast->op == TokenKind::T_PLUS_EQUAL ||
       ast->op == TokenKind::T_MINUS_EQUAL) &&
      check.unit_->typeTraits().is_pointer(ast->targetExpression->type) &&
      check.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->rightExpression->type)) {
    // pointer addition/subtraction

    (void)stdconv_.ensurePrvalue(ast->leftExpression);
    stdconv_.adjustCv(ast->leftExpression);

    (void)stdconv_.ensurePrvalue(ast->rightExpression);
    stdconv_.adjustCv(ast->rightExpression);

    (void)stdconv_.integralPromotion(ast->rightExpression);

    if (ast->adjustExpression) {
      ast->adjustExpression->type = ast->leftExpression->type;

      (void)implicit_conversion(ast->adjustExpression, ast->type);
    }

    return;
  }

  if (ast->op == TokenKind::T_LESS_LESS_EQUAL ||
      ast->op == TokenKind::T_GREATER_GREATER_EQUAL) {
    (void)stdconv_.ensurePrvalue(ast->leftExpression);
    stdconv_.adjustCv(ast->leftExpression);
    (void)stdconv_.ensurePrvalue(ast->rightExpression);
    stdconv_.adjustCv(ast->rightExpression);
    (void)stdconv_.integralPromotion(ast->leftExpression);
    (void)stdconv_.integralPromotion(ast->rightExpression);
    if (ast->adjustExpression) {
      ast->adjustExpression->type = ast->leftExpression->type;
      (void)implicit_conversion(ast->adjustExpression, ast->type);
    }
    return;
  }

  auto commonType = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                       ast->rightExpression);

  if (!commonType) {
    error(
        ast->opLoc,
        std::format("invalid compound assignment operator '{}' for types '{}' "
                    "and '{}'",
                    Token::spell(ast->op), to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
    return;
  }

  if (ast->adjustExpression) {
    ast->adjustExpression->type = commonType;

    (void)implicit_conversion(ast->adjustExpression, ast->type);
  }
}

void TypeChecker::Visitor::operator()(PackExpansionExpressionAST* ast) {
  check(ast->expression);
  if (ast->expression) {
    ast->type = ast->expression->type;
    ast->valueCategory = ast->expression->valueCategory;
  }
}

void TypeChecker::Visitor::operator()(DesignatedInitializerClauseAST* ast) {
  if (!ast->initializer) {
    error(ast->firstSourceLocation(), "expected an initializer");
    return;
  }

  if (!ast->type) ast->type = ast->initializer->type;

  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ast->initializer->valueCategory;
}

void TypeChecker::Visitor::operator()(TypeTraitExpressionAST* ast) {
  ast->type = control()->getBoolType();
  auto interp = ASTInterpreter{check.unit_};
  auto value = interp.evaluate(ast);
  if (value.has_value()) {
    ast->value = interp.toBool(*value);
  }
}

void TypeChecker::Visitor::operator()(ConditionExpressionAST* ast) {
  if (!ast->initializer) {
    error(ast->firstSourceLocation(), "expected an initializer expression");
    return;
  }

  if (report_unresolved_id(ast->initializer)) return;
  if (!ast->initializer->type) return;

  if (is_dependent_type(ast->initializer->type)) {
    ast->type = control()->getBoolType();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  auto condition = ast->initializer;
  if (!implicit_conversion(condition, control()->getBoolType())) {
    error(ast->initializer->firstSourceLocation(),
          std::format("invalid condition expression of type '{}'",
                      to_string(ast->initializer->type)));
    return;
  }

  ast->type = control()->getBoolType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(EqualInitializerAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an initializer expression");
    return;
  }

  if (report_unresolved_id(ast->expression)) return;

  ast->type = ast->expression->type;
  ast->valueCategory = ast->expression->valueCategory;
}

void TypeChecker::Visitor::operator()(BracedInitListAST* ast) {
  if (ast->valueCategory == ValueCategory::kNone)
    ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(ParenInitializerAST* ast) {
  for (auto expr : ListView{ast->expressionList}) {
    check(expr);
  }

  if (!ast->expressionList || ast->expressionList->next) return;
  if (!ast->expressionList->value) return;

  ast->type = ast->expressionList->value->type;
  ast->valueCategory = ast->expressionList->value->valueCategory;
}

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

auto TypeChecker::Visitor::implicit_conversion(ExpressionAST*& expr,
                                               const Type* destinationType)
    -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  if (in_template()) {
    if (is_dependent_type(expr->type) || is_dependent_type(destinationType))
      return true;
  }

  return stdconv_.convertImplicitly(expr, destinationType);
}

auto TypeChecker::Visitor::report_unresolved_id(ExpressionAST* expr) -> bool {
  if (!expr || expr->type) return false;

  auto idExpr = ast_cast<IdExpressionAST>(expr);
  if (!idExpr || idExpr->symbol || idExpr->nestedNameSpecifier) return false;

  if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
    auto identifier = nameId->identifier;
    auto name = identifier ? identifier->value() : std::string{};
    error(idExpr->firstSourceLocation(),
          std::format("use of undeclared identifier '{}'", name));
  } else {
    error(idExpr->firstSourceLocation(), "use of unresolved identifier");
  }

  return true;
}

TypeChecker::TypeChecker(TranslationUnit* unit) : unit_(unit) {}

auto TypeChecker::translationUnit() const -> TranslationUnit* { return unit_; }

void TypeChecker::operator()(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

void TypeChecker::check(ExpressionAST* ast) {
  if (!ast) return;
  visit(Visitor{*this}, ast);
}

void TypeChecker::check(DeclarationAST* ast) {
  if (!ast) return;

  if (auto staticAssert = ast_cast<StaticAssertDeclarationAST>(ast)) {
    Visitor{*this}.check_static_assert(staticAssert);
    return;
  }

  auto control = translationUnit()->control();

  auto simpleDeclaration = ast_cast<SimpleDeclarationAST>(ast);
  if (!simpleDeclaration) return;

  for (auto initDeclarator : ListView{simpleDeclaration->initDeclaratorList}) {
    if (!initDeclarator) continue;

    auto var = symbol_cast<VariableSymbol>(initDeclarator->symbol);
    if (!var) continue;
    if (!unit_->typeTraits().is_reference(var->type())) continue;
    if (initDeclarator->initializer) continue;

    auto loc = getInitDeclaratorLocation(initDeclarator, var);
    error(loc,
          std::format("reference variable of type '{}' must be initialized",
                      to_string(var->type())));
  }
}

void TypeChecker::check_mem_initializers(
    CompoundStatementFunctionBodyAST* ast) {
  auto functionSymbol = symbol_cast<FunctionSymbol>(scope_);
  if (!functionSymbol) return;

  if (!functionSymbol->isConstructor()) return;

  auto classSymbol = symbol_cast<ClassSymbol>(
      functionSymbol->enclosingNonTemplateParametersScope());
  if (!classSymbol) return;

  auto control = unit_->control();

  std::unordered_set<Symbol*> explicitlyInitialized;

  for (auto memInit : ListView{ast->memInitializerList}) {
    UnqualifiedIdAST* unqualifiedId = nullptr;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit))
      unqualifiedId = paren->unqualifiedId;
    else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit))
      unqualifiedId = braced->unqualifiedId;

    auto name = get_name(control, unqualifiedId);
    if (!name) continue;

    Symbol* member = nullptr;
    for (auto s : classSymbol->find(name)) {
      if (s->isField() || s->kind() == SymbolKind::kBaseClass) {
        member = s;
        break;
      }
    }

    // Search through anonymous struct/union members recursively.
    if (!member) {
      std::function<FieldSymbol*(ClassSymbol*)> findInAnonymous =
          [&](ClassSymbol* cls) -> FieldSymbol* {
        for (auto m : cls->members()) {
          auto nested = symbol_cast<ClassSymbol>(m);
          if (!nested || nested->name()) continue;
          for (auto s : nested->find(name)) {
            if (auto field = symbol_cast<FieldSymbol>(s)) {
              return field;
            }
          }
          if (auto found = findInAnonymous(nested)) return found;
        }
        return nullptr;
      };
      member = findInAnonymous(classSymbol);
    }

    if (!member) {
      for (auto base : classSymbol->baseClasses()) {
        if (!base->symbol()) continue;
        auto baseType = base->symbol()->type();
        if (!baseType) continue;
        baseType = unit_->typeTraits().remove_cv(baseType);
        if (auto classType = type_cast<ClassType>(baseType)) {
          if (classType->symbol() && classType->symbol()->name() == name) {
            member = base;
            break;
          }
        }
      }
    }

    if (!member) {
      error(memInit->firstSourceLocation(),
            std::format("'{}' is not a member or base class of '{}'",
                        to_string(name), to_string(classSymbol->name())));
      continue;
    }

    memInit->symbol = member;
    explicitlyInitialized.insert(member);

    const Type* targetType = nullptr;
    if (auto field = symbol_cast<FieldSymbol>(member)) {
      targetType = field->type();
    } else if (auto base = symbol_cast<BaseClassSymbol>(member)) {
      targetType = base->symbol() ? base->symbol()->type() : nullptr;
    }

    if (!targetType) continue;

    std::vector<ExpressionAST**> args;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit)) {
      for (auto it = paren->expressionList; it; it = it->next) {
        visit(Visitor{*this}, it->value);
        args.push_back(&it->value);
      }
    } else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit)) {
      if (braced->bracedInitList) {
        for (auto it = braced->bracedInitList->expressionList; it;
             it = it->next) {
          visit(Visitor{*this}, it->value);
          args.push_back(&it->value);
        }
      }
    }

    if (unit_->typeTraits().is_class(targetType)) {
      auto classType =
          type_cast<ClassType>(unit_->typeTraits().remove_cv(targetType));
      if (!classType) continue;
      auto targetClassSymbol = classType->symbol();
      if (!targetClassSymbol) continue;

      std::vector<ExpressionAST*> argValues;
      argValues.reserve(args.size());
      for (auto arg : args) argValues.push_back(*arg);

      auto resolution = OverloadResolution(unit_).resolveConstructor(
          targetClassSymbol, argValues);

      if (!resolution.best) {
        error(memInit->firstSourceLocation(), "no matching constructor");
        continue;
      }

      if (resolution.ambiguous) {
        error(memInit->firstSourceLocation(), "constructor call is ambiguous");
        continue;
      }

      memInit->constructor = resolution.best->symbol;

      for (size_t i = 0; i < args.size(); ++i)
        applyImplicitConversion(resolution.best->conversions[i], *args[i]);
    } else {
      if (args.size() == 1) {
        (void)implicit_conversion(*args[0], targetType);
      } else if (args.size() > 1) {
        error(memInit->firstSourceLocation(),
              "too many initializers for scalar member");
      }
    }
  }

  auto pool = unit_->arena();
  List<MemInitializerAST*>* syntheticList = ast->memInitializerList;

  for (auto base : classSymbol->baseClasses()) {
    if (explicitlyInitialized.count(base)) continue;

    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    // Find the default constructor of the base class
    FunctionSymbol* defaultCtor = nullptr;
    for (auto ctor : baseClassSymbol->constructors()) {
      if (ctor->canonical() != ctor) continue;

      auto funcType = type_cast<FunctionType>(ctor->type());
      if (funcType && funcType->parameterTypes().empty()) {
        defaultCtor = ctor;
        break;
      }
    }

    if (!defaultCtor) continue;

    auto syntheticInit = ParenMemInitializerAST::create(
        pool, /*nestedNameSpecifier=*/nullptr, /*unqualifiedId=*/nullptr,
        /*expressionList=*/nullptr, /*symbol=*/base,
        /*constructor=*/defaultCtor);

    auto node = make_list_node<MemInitializerAST>(pool, syntheticInit);
    node->next = syntheticList;
    syntheticList = node;
  }

  ast->memInitializerList = syntheticList;
}

void TypeChecker::Visitor::check_static_assert(
    StaticAssertDeclarationAST* ast) {
  auto loc = ast->firstSourceLocation();

  if (in_template()) return;

  auto interp = ASTInterpreter{check.unit_};

  auto value = interp.evaluate(ast->expression);

  if (value.has_value()) {
    ast->value = interp.toBool(*value);
  }

  if (ast->value.has_value() && ast->value.value()) {
    return;
  }

  if (!ast->value.has_value()) {
    error(loc,
          "static assertion expression is not an integral constant "
          "expression");
    return;
  }

  if (ast->literalLoc)
    loc = ast->literalLoc;
  else if (ast->expression)
    loc = ast->expression->firstSourceLocation();

  error(loc, ast->literal ? ast->literal->value()
                          : std::string("static assert failed"));
}

void TypeChecker::Visitor::check_addition(BinaryExpressionAST* ast) {
  // ### TODO: check for user-defined conversion operators
  if (check.unit_->typeTraits().is_class(ast->leftExpression->type)) return;
  if (check.unit_->typeTraits().is_class(ast->rightExpression->type)) return;

  if (auto ty = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                   ast->rightExpression)) {
    ast->type = ty;
    return;
  }

  (void)stdconv_.ensurePrvalue(ast->leftExpression);
  stdconv_.adjustCv(ast->leftExpression);

  (void)stdconv_.ensurePrvalue(ast->rightExpression);
  stdconv_.adjustCv(ast->rightExpression);

  const auto left_is_pointer =
      check.unit_->typeTraits().is_pointer(ast->leftExpression->type);

  const auto right_is_pointer =
      check.unit_->typeTraits().is_pointer(ast->rightExpression->type);

  const auto left_is_integral =
      check.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->leftExpression->type);

  const auto right_is_integral =
      check.unit_->typeTraits().is_integral_or_unscoped_enum(
          ast->rightExpression->type);

  if (left_is_pointer && right_is_integral) {
    (void)stdconv_.integralPromotion(ast->rightExpression);
    ast->type = ast->leftExpression->type;
    return;
  }

  if (right_is_pointer && left_is_integral) {
    (void)stdconv_.integralPromotion(ast->leftExpression);
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
  if (check.unit_->typeTraits().is_class(ast->leftExpression->type)) return;
  if (check.unit_->typeTraits().is_class(ast->rightExpression->type)) return;

  if (auto ty = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                   ast->rightExpression)) {
    ast->type = ty;
    return;
  }

  (void)stdconv_.ensurePrvalue(ast->leftExpression);
  stdconv_.adjustCv(ast->leftExpression);

  (void)stdconv_.ensurePrvalue(ast->rightExpression);
  stdconv_.adjustCv(ast->rightExpression);

  auto check_operand_types = [&]() {
    if (!check.unit_->typeTraits().is_pointer(ast->leftExpression->type))
      return false;

    if (!check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(
            ast->rightExpression->type) &&
        !check.unit_->typeTraits().is_pointer(ast->rightExpression->type))
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

  if (check.unit_->typeTraits().is_pointer(ast->rightExpression->type)) {
    auto leftElementType =
        check.unit_->typeTraits().get_element_type(ast->leftExpression->type);
    (void)strip_cv(leftElementType);

    auto rightElementType =
        check.unit_->typeTraits().get_element_type(ast->rightExpression->type);
    (void)strip_cv(rightElementType);

    if (check.unit_->typeTraits().is_same(leftElementType, rightElementType)) {
      ast->type = control()->getLongIntType();  // TODO: ptrdiff_t
    } else {
      error(ast->opLoc,
            std::format("'{}' and '{}' are not pointers to compatible types",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));
    }

    return;
  }

  (void)stdconv_.integralPromotion(ast->rightExpression);
  ast->type = ast->leftExpression->type;
}

void TypeChecker::Visitor::check_prefix_increment_decrement(
    UnaryExpressionAST* ast, std::string_view action, std::string_view opWord) {
  if (!is_glvalue(ast->expression)) {
    error(ast->opLoc, std::format("cannot {} an rvalue of type '{}'", action,
                                  to_string(ast->expression->type)));
    return;
  }

  if (!check.unit_->typeTraits().is_const(ast->expression->type)) {
    auto ty = ast->expression->type;

    if (isCxx()
            ? check.unit_->typeTraits().is_arithmetic(ty)
            : check.unit_->typeTraits().is_arithmetic_or_unscoped_enum(ty)) {
      ast->type = ty;
      ast->valueCategory = ValueCategory::kLValue;
      return;
    }

    if (auto ptrTy = as_pointer(ty)) {
      if (!check.unit_->typeTraits().is_void(ptrTy->elementType())) {
        ast->type = ptrTy;
        ast->valueCategory = ValueCategory::kLValue;
        return;
      }
    }
  }

  error(ast->opLoc, std::format("cannot {} a value of type '{}'", opWord,
                                to_string(ast->expression->type)));
}

auto TypeChecker::Visitor::resolve_operator_overload(
    const Type* leftType, TokenKind op, SourceLocation opLoc,
    const Type* rightType, FunctionSymbol*& symbolOut) -> bool {
  symbolOut = nullptr;

  if (auto symbol = check.lookupOperator(leftType, op, rightType)) {
    symbolOut = symbol;
    return true;
  }

  if (check.wasLastOperatorLookupAmbiguous()) {
    error(opLoc, std::format("call to overloaded operator '{}' is ambiguous",
                             Token::spell(op)));
    return true;
  }

  return false;
}

auto TypeChecker::Visitor::resolve_unary_overload(UnaryExpressionAST* ast)
    -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->expression->type, ast->op, ast->opLoc,
                                 nullptr, operatorFunc)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  setResultTypeAndValueCategory(ast, operatorFunc->type());
  return true;
}

auto TypeChecker::Visitor::resolve_binary_overload(BinaryExpressionAST* ast,
                                                   bool setValueCategory)
    -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->leftExpression->type, ast->op, ast->opLoc,
                                 ast->rightExpression->type, operatorFunc)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  if (setValueCategory) {
    setResultTypeAndValueCategory(ast, operatorFunc->type());
  } else if (auto functionType =
                 type_cast<FunctionType>(operatorFunc->type())) {
    ast->type = functionType->returnType();
  } else {
    ast->type = operatorFunc->type();
  }

  return true;
}

auto TypeChecker::Visitor::resolve_assignment_overload(
    AssignmentExpressionAST* ast) -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->leftExpression->type, ast->op, ast->opLoc,
                                 ast->rightExpression->type, operatorFunc)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  setResultTypeAndValueCategory(ast, operatorFunc->type());
  return true;
}

auto TypeChecker::Visitor::resolve_compound_assignment_overload(
    CompoundAssignmentExpressionAST* ast) -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->targetExpression->type, ast->op,
                                 ast->opLoc, ast->rightExpression->type,
                                 operatorFunc)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  setResultTypeAndValueCategory(ast, operatorFunc->type());
  return true;
}

auto TypeChecker::Visitor::check_member_access(MemberExpressionAST* ast)
    -> bool {
  const Type* objectType = ast->baseExpression->type;
  auto cv1 = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    if (check.unit_->typeTraits().is_class_or_union(
            ast->baseExpression->type)) {
      auto operatorFunc = resolve_arrow_operator(ast);
      if (!operatorFunc) return false;
      auto functionType = type_cast<FunctionType>(operatorFunc->type());
      if (!functionType) return false;
      objectType = functionType->returnType();
      cv1 = strip_cv(objectType);
    } else {
      (void)stdconv_.ensurePrvalue(ast->baseExpression);
      objectType = ast->baseExpression->type;
      cv1 = strip_cv(objectType);
    }

    auto pointerType = as_pointer(objectType);
    if (!pointerType) return false;

    objectType = pointerType->elementType();
    cv1 = strip_cv(objectType);
  }

  auto classType = as_class(objectType);
  if (!classType) return false;

  auto memberName = get_name(control(), ast->unqualifiedId);

  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
  if (templateId && templateId->identifier) memberName = templateId->identifier;

  auto classSymbol = classType->symbol();

  check.unit_->typeTraits().requireCompleteClass(classSymbol);

  auto symbol = qualifiedLookup(classSymbol, memberName);

  ast->symbol = symbol;

  if (!symbol) {
    if (in_template() && classSymbol->templateDeclaration() &&
        !classSymbol->isComplete()) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kLValue;
      return true;
    }

    auto member = std::string{"<unknown>"};
    if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
      if (auto identifier = nameId->identifier) member = identifier->value();
    } else if (templateId && templateId->identifier) {
      member = templateId->identifier->value();
    }

    error(ast->firstSourceLocation(),
          std::format("no member named '{}' in type '{}'", member,
                      to_string(classSymbol->name())));
    return true;
  }

  if (symbol) {
    ast->type = symbol->type();

    if (symbol->isEnumerator()) {
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      if (is_lvalue(ast->baseExpression) ||
          ast->accessOp == TokenKind::T_MINUS_GREATER) {
        ast->valueCategory = ValueCategory::kLValue;
      } else {
        ast->valueCategory = ValueCategory::kXValue;
      }

      if (auto field = symbol_cast<FieldSymbol>(symbol);
          field && !field->isStatic()) {
        auto cv2 = strip_cv(ast->type);

        if (is_volatile(cv1) || is_volatile(cv2))
          ast->type = check.unit_->typeTraits().add_volatile(ast->type);

        if (!field->isMutable() && (is_const(cv1) || is_const(cv2)))
          ast->type = check.unit_->typeTraits().add_const(ast->type);
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
    auto pointerType = as_pointer(objectType);
    if (!pointerType) return false;
    objectType = pointerType->elementType();
    cv = strip_cv(objectType);
  }

  if (!check.unit_->typeTraits().is_scalar(objectType)) {
    // return false if the object type is not a scalar type
    return false;
  }

  auto dtor = ast_cast<DestructorIdAST>(ast->unqualifiedId);
  if (!dtor) return false;

  auto name = ast_cast<NameIdAST>(dtor->id);
  if (!name) return true;

  Symbol* symbol = nullptr;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol)
    symbol =
        qualifiedLookupType(ast->nestedNameSpecifier->symbol, name->identifier);
  else {
    for (auto s = check.scope_; s && !symbol; s = s->parent()) {
      for (auto found : s->find(name->identifier)) {
        if (found->isHidden()) continue;
        if (is_type(found)) {
          symbol = found;
          break;
        }
      }
    }
  }
  if (!symbol) return true;

  if (!check.unit_->typeTraits().is_same(symbol->type(), objectType)) {
    error(ast->unqualifiedId->firstSourceLocation(),
          "the type of object expression does not match the type "
          "being destroyed");
    return true;
  }

  ast->symbol = symbol;
  ast->type = control()->getFunctionType(control()->getVoidType(), {});

  return true;
}

void TypeChecker::check_return_statement(ReturnStatementAST* ast) {
  const Type* targetType = nullptr;
  ScopeSymbol* functionScope = nullptr;
  for (auto current = scope_; current; current = current->parent()) {
    if (!current) continue;
    if (current->isFunction() || current->isLambda()) {
      if (auto functionType = type_cast<FunctionType>(current->type())) {
        targetType = functionType->returnType();
        functionScope = current;
      }
    }
  }

  if (!targetType) return;

  if (type_cast<AutoType>(targetType) && ast->expression &&
      ast->expression->type && !isDependent(unit_, ast->expression->type)) {
    auto deducedType = unit_->typeTraits().remove_cvref(ast->expression->type);
    if (auto funcSym = symbol_cast<FunctionSymbol>(functionScope)) {
      if (auto funcType = type_cast<FunctionType>(funcSym->type())) {
        auto newFuncType = unit_->control()->getFunctionType(
            deducedType,
            std::vector<const Type*>(funcType->parameterTypes().begin(),
                                     funcType->parameterTypes().end()),
            funcType->isVariadic(), funcType->cvQualifiers(),
            funcType->refQualifier(), funcType->isNoexcept());
        funcSym->setType(newFuncType);
        targetType = deducedType;
      }
    }
  }

  if (isDependent(unit_, targetType)) return;
  if (ast->expression && isDependent(unit_, ast->expression)) return;
  if (ast->expression && ast->expression->type &&
      isDependent(unit_, ast->expression->type))
    return;

  auto seq = checkImplicitConversion(ast->expression, targetType);
  applyImplicitConversion(seq, ast->expression);
}

auto TypeChecker::implicit_conversion(ExpressionAST*& yyast,
                                      const Type* targetType) -> bool {
  Visitor visitor{*this};
  return visitor.implicit_conversion(yyast, targetType);
}

void TypeChecker::check_bool_condition(ExpressionAST*& expr) {
  if (expr && expr->type && isDependent(unit_, expr->type)) return;
  Visitor visitor{*this};
  (void)visitor.implicit_conversion(expr, unit_->control()->getBoolType());
}

void TypeChecker::check_integral_condition(ExpressionAST*& expr) {
  if (expr && expr->type && isDependent(unit_, expr->type)) return;
  auto control = unit_->control();
  if (!unit_->typeTraits().is_integral(expr->type) &&
      !unit_->typeTraits().is_enum(expr->type))
    return;
  Visitor visitor{*this};
  (void)visitor.stdconv_.lvalueToRvalue(expr);
  visitor.stdconv_.adjustCv(expr);
  (void)visitor.stdconv_.integralPromotion(expr);
}

void TypeChecker::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->error(loc, std::move(message));
}

void TypeChecker::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->warning(loc, std::move(message));
}

auto TypeChecker::checkImplicitConversion(ExpressionAST* expr,
                                          const Type* targetType)
    -> ImplicitConversionSequence {
  StandardConversion stdconv(unit_, unit_->language() == LanguageKind::kC);
  return stdconv.computeConversionSequence(expr, targetType);
}

void TypeChecker::wrapWithImplicitCast(ImplicitCastKind castKind,
                                       const Type* type, ExpressionAST*& expr) {
  StandardConversion stdconv(unit_, unit_->language() == LanguageKind::kC);
  stdconv.wrapWithImplicitCast(castKind, type, expr);
}

void TypeChecker::applyImplicitConversion(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  StandardConversion stdconv(unit_, unit_->language() == LanguageKind::kC);
  stdconv.applyConversionSequence(sequence, expr);
}

auto TypeChecker::findOverloads(ScopeSymbol* scope, const Name* name) const
    -> std::vector<FunctionSymbol*> {
  OverloadResolution resolution(unit_);
  return resolution.findCandidates(scope, name);
}

auto TypeChecker::selectBestOverload(
    const std::vector<FunctionSymbol*>& candidates, const Type* leftType,
    const Type* rightType, bool* ambiguous) const -> FunctionSymbol* {
  OverloadResolution resolution(unit_);
  return resolution.resolveBinaryOperator(candidates, leftType, rightType,
                                          ambiguous);
}

auto TypeChecker::trySelectOperator(
    const std::vector<FunctionSymbol*>& candidates, const Type* type,
    const Type* rightType) -> FunctionSymbol* {
  if (candidates.empty()) return nullptr;
  bool ambiguous = false;
  auto selected = selectBestOverload(candidates, type, rightType, &ambiguous);
  lastOperatorLookupAmbiguous_ = ambiguous;
  return selected;
}

auto TypeChecker::collectOverloads(Symbol* symbol) const
    -> std::vector<FunctionSymbol*> {
  OverloadResolution resolution(unit_);
  return resolution.collectCandidates(symbol);
}

auto TypeChecker::lookupOperator(const Type* type, TokenKind op,
                                 const Type* rightType) -> FunctionSymbol* {
  OverloadResolution resolution(unit_);
  auto result = resolution.lookupOperator(type, op, rightType);
  lastOperatorLookupAmbiguous_ = resolution.wasLastLookupAmbiguous();
  return result;
}

auto TypeChecker::as_pointer(const Type* type) const -> const PointerType* {
  return type_cast<PointerType>(unit_->typeTraits().remove_cv(type));
}

auto TypeChecker::as_class(const Type* type) const -> const ClassType* {
  return type_cast<ClassType>(unit_->typeTraits().remove_cv(type));
}

auto TypeChecker::getInitDeclaratorLocation(InitDeclaratorAST* ast,
                                            VariableSymbol* var) const
    -> SourceLocation {
  if (!ast) return var ? var->location() : SourceLocation{};

  auto loc = ast->firstSourceLocation();
  if (loc) return loc;

  if (auto declarator = ast->declarator) {
    if (auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator)) {
      if (auto nameId = ast_cast<NameIdAST>(id->unqualifiedId)) {
        if (nameId->identifierLoc) return nameId->identifierLoc;
      }
      loc = id->firstSourceLocation();
      if (loc) return loc;
    }

    loc = declarator->firstSourceLocation();
    if (loc) return loc;
  }

  return var ? var->location() : SourceLocation{};
}

}  // namespace cxx
