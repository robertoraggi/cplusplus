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
#include <cxx/class_template_deduction.h>
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
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cxx {
namespace {
constexpr std::uintmax_t kMaximumAlignment = 1ULL << 32;

[[nodiscard]] auto describe_scope(Symbol* symbol) -> std::string {
  const auto name = to_string(symbol->name());

  if (!symbol_cast<NamespaceSymbol>(symbol)) return std::format("'{}'", name);
  if (name.empty()) return "the unnamed namespace";

  return std::format("namespace '{}'", name);
}

[[nodiscard]] auto delegationReaches(FunctionSymbol* target,
                                     FunctionSymbol* origin) -> bool {
  std::unordered_set<FunctionSymbol*> visited;
  for (auto current = target; current;
       current = current->delegatingConstructor()) {
    if (current == origin) return true;
    if (!visited.insert(current).second) return false;
  }
  return false;
}

[[nodiscard]] auto is_unresolved_id(ExpressionAST* expr) -> bool {
  auto idExpr = ast_cast<IdExpressionAST>(expr);
  return idExpr && !idExpr->symbol;
}

struct IsPotentiallyThrowing {
  TranslationUnit* unit = nullptr;

  auto operator()(ExpressionAST*) -> bool { return false; }

  auto operator()(ThrowExpressionAST*) -> bool { return true; }

  auto operator()(NoexceptExpressionAST*) -> bool { return false; }

  auto operator()(CallExpressionAST* ast) -> bool {
    auto base = ast->baseExpression;
    if (!base) return true;

    FunctionSymbol* function = nullptr;
    if (auto id = ast_cast<IdExpressionAST>(base))
      function = symbol_cast<FunctionSymbol>(id->symbol);
    else if (auto member = ast_cast<MemberExpressionAST>(base))
      function = symbol_cast<FunctionSymbol>(member->symbol);

    const FunctionType* ft = nullptr;
    if (function) {
      if (unit)
        ASTRewriter::completePendingExceptionSpecification(unit, function);
      ft = type_cast<FunctionType>(function->type());
    }
    if (!ft) ft = type_cast<FunctionType>(base->type);
    if (!ft)
      if (auto pt = type_cast<PointerType>(base->type))
        ft = type_cast<FunctionType>(pt->elementType());
    if (!ft || !ft->isNoexcept()) return true;
    for (auto it = ast->expressionList; it; it = it->next)
      if (apply(it->value)) return true;
    return apply(ast->baseExpression);
  }

  auto operator()(TypeConstructionAST* ast) -> bool {
    if (initializationIsPotentiallyThrowing(ast->type, ast->constructorSymbol))
      return true;
    return applyAll(ast->expressionList);
  }

  auto operator()(BinaryExpressionAST* ast) -> bool {
    if (ast->symbol && functionIsPotentiallyThrowing(ast->symbol)) return true;
    return apply(ast->leftExpression) || apply(ast->rightExpression);
  }

  auto operator()(UnaryExpressionAST* ast) -> bool {
    if (ast->symbol && functionIsPotentiallyThrowing(ast->symbol)) return true;
    return apply(ast->expression);
  }

  auto operator()(AssignmentExpressionAST* ast) -> bool {
    if (ast->symbol && functionIsPotentiallyThrowing(ast->symbol)) return true;
    return apply(ast->leftExpression) || apply(ast->rightExpression);
  }

  auto operator()(CompoundAssignmentExpressionAST* ast) -> bool {
    if (ast->symbol && functionIsPotentiallyThrowing(ast->symbol)) return true;
    return apply(ast->targetExpression) || apply(ast->rightExpression);
  }

  auto operator()(SubscriptExpressionAST* ast) -> bool {
    if (ast->symbol && functionIsPotentiallyThrowing(ast->symbol)) return true;
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
    if (initializationIsPotentiallyThrowing(ast->type, ast->constructorSymbol))
      return true;
    return apply(ast->bracedInitList);
  }

  auto operator()(EqualInitializerAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(BracedInitListAST* ast) -> bool {
    return applyAll(ast->expressionList);
  }

  auto operator()(ParenInitializerAST* ast) -> bool {
    return applyAll(ast->expressionList);
  }

  auto operator()(DesignatedInitializerClauseAST* ast) -> bool {
    return apply(ast->initializer);
  }

  auto operator()(ConditionExpressionAST* ast) -> bool {
    return apply(ast->initializer);
  }

  auto operator()(PackExpansionExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(MemberExpressionAST* ast) -> bool {
    return apply(ast->baseExpression);
  }

  auto operator()(GenericSelectionExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(BuiltinBitCastExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(VaArgExpressionAST* ast) -> bool {
    return apply(ast->expression);
  }

  auto operator()(NestedStatementExpressionAST*) -> bool { return true; }

  auto operator()(AwaitExpressionAST*) -> bool { return true; }

  auto operator()(YieldExpressionAST*) -> bool { return true; }

  auto operator()(TypeidExpressionAST* ast) -> bool {
    auto expr = ast->expression;
    while (auto nested = ast_cast<NestedExpressionAST>(expr))
      expr = nested->expression;

    auto unary = ast_cast<UnaryExpressionAST>(expr);
    if (!unary || unary->op != TokenKind::T_STAR) return false;
    if (unary->symbol) return true;

    auto operand = unary->expression ? unary->expression->type : nullptr;
    auto pointerType = type_cast<PointerType>(operand);
    if (!pointerType) return false;

    auto elementType = pointerType->elementType();
    if (auto qualType = type_cast<QualType>(elementType))
      elementType = qualType->elementType();

    auto classType = type_cast<ClassType>(elementType);
    if (!classType) return false;

    auto cls = classType->definition();
    return !cls || cls->isPolymorphic();
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

  auto applyAll(List<ExpressionAST*>* expressions) -> bool {
    for (auto expr : ListView{expressions})
      if (apply(expr)) return true;
    return false;
  }

  auto functionIsPotentiallyThrowing(FunctionSymbol* symbol) -> bool {
    if (unit) ASTRewriter::completePendingExceptionSpecification(unit, symbol);
    auto funcType = type_cast<FunctionType>(symbol->type());
    return !funcType || !funcType->isNoexcept();
  }

  auto initializationIsPotentiallyThrowing(const Type* type,
                                           FunctionSymbol* constructorSymbol)
      -> bool {
    if (constructorSymbol)
      return functionIsPotentiallyThrowing(constructorSymbol);

    auto classType = type_cast<ClassType>(type);
    if (!classType) return false;

    auto cls = classType->definition();
    if (!cls || !cls->isComplete()) return true;

    auto defaultConstructor = cls->defaultConstructor();
    if (!defaultConstructor) return cls->hasUserDeclaredConstructors();

    return functionIsPotentiallyThrowing(defaultConstructor);
  }
};

}  // namespace

auto TypeChecker::isPotentiallyThrowing(ExpressionAST* expr) -> bool {
  return IsPotentiallyThrowing{}.apply(expr);
}

struct TypeChecker::Visitor {
  TypeChecker& check;
  TypeTraits traits;

  explicit Visitor(TypeChecker& check) : check(check), traits(check.unit_) {}

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
    return isEnclosedInDependentTemplate(check.unit_, scope());
  }

  [[nodiscard]] auto require_complete_for_sizeof(SourceLocation loc,
                                                 const Type* type) -> bool {
    if (!type) return false;
    if (is_dependent_type(type)) return false;
    auto stripped = traits.remove_cv(type);
    while (traits.is_array(stripped)) {
      stripped = traits.remove_cv(traits.get_element_type(stripped));
    }
    if (auto classType = type_cast<ClassType>(stripped)) {
      traits.requireCompleteClass(classType->symbol());
      if (!classType->definition()->isComplete()) {
        error(loc, std::format("invalid application of 'sizeof' to an "
                               "incomplete type '{}'",
                               to_string(type)));
        return false;
      }
    }
    return true;
  }

  void warning(SourceLocation loc, std::string message) {
    check.warning(loc, std::move(message));
  }

  void report_unresolved_id(IdExpressionAST* ast);

  void report_unresolved_qualified_id(IdExpressionAST* ast);

  [[nodiscard]] auto add_implicit_object_cv(const Type* fieldType,
                                            FieldSymbol* field) -> const Type*;

  [[nodiscard]] auto base_symbol(ExpressionAST* base) -> Symbol*;

  [[nodiscard]] auto overload_set_of(ExpressionAST* base) -> OverloadSetSymbol*;

  [[nodiscard]] auto designated_function_type(OverloadSetSymbol* overloadSet)
      -> const Type*;

  [[nodiscard]] auto named_symbol_type(Symbol* symbol) -> const Type*;

  [[nodiscard]] auto explicit_template_arguments(ExpressionAST* base)
      -> List<TemplateArgumentAST*>*;

  [[nodiscard]] auto enclosing_function() -> FunctionSymbol*;

  [[nodiscard]] auto strip_parentheses(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto strip_cv(const Type*& type) -> CvQualifiers;

  [[nodiscard]] auto type_info_type(SourceLocation loc) -> const Type*;

  [[nodiscard]] auto comparison_category_type(SourceLocation loc,
                                              std::string_view name)
      -> const Type*;

  StandardConversion stdconv_{check.unit_,
                              check.unit_->language() == LanguageKind::kC};

  void setResultTypeAndValueCategory(ExpressionAST* ast, const Type* type);
  void setResultTypeAndValueCategory(ExpressionAST* ast,
                                     FunctionSymbol* function);

  [[nodiscard]] auto implicit_conversion(
      ExpressionAST*& expr, const Type* destinationType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization) -> bool;

  [[nodiscard]] auto contextual_conversion_to_bool(ExpressionAST*& expr)
      -> bool;

  [[nodiscard]] auto as_pointer(const Type* type) const -> const PointerType* {
    return check.as_pointer(type);
  }

  [[nodiscard]] auto as_array(const Type* type) const -> const Type* {
    if (traits.is_array(type)) return traits.remove_cv(type);
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
  void check_three_way_comparison(BinaryExpressionAST* ast);
  [[nodiscard]] auto rewrite_not_equal_as_negated_equal(
      BinaryExpressionAST* ast) -> bool;
  void check_equality(BinaryExpressionAST* ast);
  void prepare_comparison_operands(BinaryExpressionAST* ast);
  void set_base_symbol(ExpressionAST* base, Symbol* sym);
  enum class CallResolution { kNoOverloadSet, kResolved, kFailed };

  [[nodiscard]] auto resolve_call_overload(
      CallExpressionAST* ast, const std::vector<const Type*>& argTypes)
      -> CallResolution;
  [[nodiscard]] auto resolve_function_type(CallExpressionAST* ast)
      -> const FunctionType*;
  [[nodiscard]] auto resolve_call_operator(CallExpressionAST* ast)
      -> const FunctionType*;
  [[nodiscard]] auto resolve_arrow_operator(MemberExpressionAST* ast)
      -> FunctionSymbol*;
  void check_function_arguments(List<ExpressionAST*>* arguments,
                                SourceLocation callLoc,
                                const FunctionType* functionType);
  [[nodiscard]] auto typeCheckBuiltinDispatch(CallExpressionAST* ast,
                                              BuiltinFunctionKind kind) -> bool;
  [[nodiscard]] auto checkBuiltinInvoke(CallExpressionAST* ast) -> bool;
  [[nodiscard]] auto checkBuiltinAddressof(CallExpressionAST* ast) -> bool;
  [[nodiscard]] auto checkBuiltinAssumeAligned(CallExpressionAST* ast) -> bool;
  [[nodiscard]] auto checkBuiltinVaListAccess(CallExpressionAST* ast) -> bool;
  void check_member_pointer_access(BinaryExpressionAST* ast);
  [[nodiscard]] auto checkBuiltinCountZerosGeneric(CallExpressionAST* ast)
      -> bool;
  void resolveBuiltinLibcall(CallExpressionAST* ast);
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
      const Type* rightType, FunctionSymbol*& symbolOut,
      ExpressionAST* leftExpr = nullptr, ExpressionAST* rightExpr = nullptr)
      -> bool;

  [[nodiscard]] auto resolve_unary_overload(UnaryExpressionAST* ast) -> bool;

  void apply_operator_argument_conversions(FunctionSymbol* operatorFunc,
                                           ExpressionAST*& leftExpression,
                                           ExpressionAST*& rightExpression);

  void adjust_member_operator_object_argument(FunctionSymbol* operatorFunc,
                                              ExpressionAST*& objectExpression);

  [[nodiscard]] auto is_known_complete_object(ExpressionAST* expression)
      -> bool;

  [[nodiscard]] auto is_virtual_member_operator_dispatch(
      FunctionSymbol* operatorFunc, ExpressionAST* objectExpression) -> bool;

  void mark_virtual_dispatch(CallExpressionAST* ast);

  [[nodiscard]] auto has_initializer_list_constructor(ClassSymbol* classSymbol)
      -> bool;

  [[nodiscard]] auto convert_class_operands_for_builtin(
      BinaryExpressionAST* ast) -> bool;

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

  [[nodiscard]] auto resolveDestructorIdType(MemberExpressionAST* ast,
                                             DestructorIdAST* dtor) -> Symbol*;

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
    stdconv_.appendDefaultArguments(func, &ast->expressionList);
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
  [[nodiscard]] auto splicedExpression(SplicerAST* splicer) -> ExpressionAST*;
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
    check.check_braced_init_list(ast->type, ast->bracedInitList,
                                 InitializationKind::kDirectListInitialization);
  }

  if (auto unbounded = type_cast<UnboundedArrayType>(ast->type)) {
    if (ast->bracedInitList) {
      auto elementType = unbounded->elementType();
      const auto isCharElem = type_cast<CharType>(elementType) ||
                              type_cast<SignedCharType>(elementType) ||
                              type_cast<UnsignedCharType>(elementType);

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

      if (type_cast<UnboundedArrayType>(ast->type)) {
        size_t count = 0;
        for (auto it = ast->bracedInitList->expressionList; it; it = it->next)
          ++count;
        if (count > 0) {
          ast->type = control()->getBoundedArrayType(elementType, count);
        }
      }

      ast->bracedInitList->type = ast->type;
    }
  }

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
      if (classSymbol->isClosureType()) {
        if (auto capturedThisField = classSymbol->capturedThisField()) {
          ast->type = capturedThisField->type();
          break;
        }
        continue;
      }
      ast->type = control()->getPointerType(classSymbol->type());
      break;
    }

    if (auto functionSymbol = symbol_cast<FunctionSymbol>(current)) {
      if (auto classSymbol =
              symbol_cast<ClassSymbol>(functionSymbol->parent())) {
        if (auto capturedThisField = classSymbol->capturedThisField()) {
          ast->type = capturedThisField->type();
        } else {
          auto functionType = type_cast<FunctionType>(functionSymbol->type());
          const auto cv = functionType->cvQualifiers();
          if (cv != CvQualifiers::kNone) {
            auto elementType = control()->getQualType(classSymbol->type(), cv);
            ast->type = control()->getPointerType(elementType);
          } else {
            ast->type = control()->getPointerType(classSymbol->type());
          }
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
      if (!self.traits.is_same(selectorType, assoc->typeId->type)) {
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

      selectorType = self.traits.decay(ast->expression->type);

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

  if (!ast->expression->type) return;

  ast->type = ast->expression->type;
  ast->valueCategory = ast->expression->valueCategory;
}

auto TypeChecker::Visitor::add_implicit_object_cv(const Type* fieldType,
                                                  FieldSymbol* field)
    -> const Type* {
  auto fieldClass = symbol_cast<ClassSymbol>(field->parent());
  if (!fieldClass) return fieldType;

  auto func = enclosing_function();
  if (!func) return fieldType;

  auto funcClass = symbol_cast<ClassSymbol>(func->parent());

  if (!funcClass) return fieldType;
  if (!traits.is_base_of(fieldClass->type(), funcClass->type()))
    return fieldType;

  auto funcType = type_cast<FunctionType>(func->type());
  if (!funcType) return fieldType;

  const auto objectCv = funcType->cvQualifiers();

  if (is_volatile(objectCv)) fieldType = traits.add_volatile(fieldType);
  if (!field->isMutable() && is_const(objectCv))
    fieldType = traits.add_const(fieldType);

  return fieldType;
}

void TypeChecker::Visitor::operator()(IdExpressionAST* ast) {
  if (!ast->symbol) {
    if (!ast->nestedNameSpecifier) {
      report_unresolved_id(ast);
      return;
    }

    if (!isDependent(check.unit_, ast->nestedNameSpecifier)) {
      report_unresolved_qualified_id(ast);
    } else if (in_template()) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
    }

    return;
  }

  if (auto usingDecl = symbol_cast<UsingDeclarationSymbol>(ast->symbol);
      usingDecl && !usingDecl->target() && in_template()) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  if (symbol_cast<ConceptSymbol>(ast->symbol)) {
    ast->type = control()->getBoolType();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(ast->symbol)) {
    if (in_template() &&
        hasDependentTemplateArguments(
            check.unit_, ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId))) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return;
    }

    ast->type = designated_function_type(overloadSet);
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  ast->type = traits.remove_reference(named_symbol_type(ast->symbol));

  if (ast->symbol->isEnumerator() || ast->symbol->isNonTypeParameter()) {
    ast->valueCategory = ValueCategory::kPrValue;
    stdconv_.adjustCv(ast);
    return;
  }

  ast->valueCategory = ValueCategory::kLValue;

  if (ast->nestedNameSpecifier) return;

  auto field = symbol_cast<FieldSymbol>(ast->symbol);
  if (!field || field->isStatic()) return;

  ast->type = add_implicit_object_cv(ast->type, field);
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
    apply_operator_argument_conversions(operatorFunc, ast->baseExpression,
                                        ast->indexExpression);
    ast->isVirtualDispatch =
        is_virtual_member_operator_dispatch(operatorFunc, ast->baseExpression);
    setResultTypeAndValueCategory(ast, operatorFunc);
    return;
  }

  if (check.wasLastOperatorLookupAmbiguous()) {
    error(ast->firstSourceLocation(),
          "call to overloaded operator '[]' is ambiguous");
    return;
  }

  if (traits.is_class(ast->baseExpression->type)) return;
  if (traits.is_class(ast->indexExpression->type)) return;

  auto array_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                ExpressionAST*& index) {
    if (!traits.is_array(base->type)) return false;
    if (!traits.is_arithmetic_or_unscoped_enum(index->type)) return false;

    (void)stdconv_.temporaryMaterialization(base);
    (void)stdconv_.ensurePrvalue(index);
    stdconv_.adjustCv(index);
    (void)stdconv_.integralPromotion(index);

    ast->type = traits.get_element_type(base->type);
    ast->valueCategory = base->valueCategory;
    return true;
  };

  auto pointer_subscript = [this](ExpressionAST* ast, ExpressionAST*& base,
                                  ExpressionAST*& index) {
    if (!traits.is_pointer(base->type)) return false;
    if (!traits.is_arithmetic_or_unscoped_enum(index->type)) return false;

    (void)stdconv_.ensurePrvalue(base);
    stdconv_.adjustCv(base);

    (void)stdconv_.ensurePrvalue(index);
    stdconv_.adjustCv(index);
    (void)stdconv_.integralPromotion(index);

    ast->type = traits.get_element_type(base->type);
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
  base = strip_parentheses(base);
  if (auto id = ast_cast<IdExpressionAST>(base))
    id->symbol = sym;
  else if (auto member = ast_cast<MemberExpressionAST>(base))
    member->symbol = sym;
}

auto TypeChecker::Visitor::designated_function_type(
    OverloadSetSymbol* overloadSet) -> const Type* {
  if (auto function = designatedFunction(overloadSet))
    return named_symbol_type(function);
  return overloadSet->type();
}

auto TypeChecker::Visitor::named_symbol_type(Symbol* symbol) -> const Type* {
  ASTRewriter::completeDeducedReturnType(check.unit_, symbol);
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    ASTRewriter::completePendingExceptionSpecification(check.unit_, function);
  return symbol->type();
}

auto TypeChecker::Visitor::overload_set_of(ExpressionAST* base)
    -> OverloadSetSymbol* {
  if (auto ovl = type_cast<OverloadSetType>(base->type)) return ovl->symbol();
  return symbol_cast<OverloadSetSymbol>(base_symbol(base));
}

auto TypeChecker::Visitor::base_symbol(ExpressionAST* base) -> Symbol* {
  base = strip_parentheses(base);
  if (auto id = ast_cast<IdExpressionAST>(base)) return id->symbol;
  if (auto member = ast_cast<MemberExpressionAST>(base)) return member->symbol;
  return nullptr;
}

auto TypeChecker::Visitor::explicit_template_arguments(ExpressionAST* base)
    -> List<TemplateArgumentAST*>* {
  auto unqualifiedId = [&]() -> UnqualifiedIdAST* {
    if (auto id = ast_cast<IdExpressionAST>(base)) return id->unqualifiedId;
    if (auto member = ast_cast<MemberExpressionAST>(base))
      return member->unqualifiedId;
    return nullptr;
  }();

  auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId);
  if (!templateId) return nullptr;

  return templateId->templateArgumentList;
}

auto TypeChecker::Visitor::enclosing_function() -> FunctionSymbol* {
  if (!check.scope_) return nullptr;
  if (auto func = symbol_cast<FunctionSymbol>(check.scope_)) return func;
  return check.scope_->enclosingFunction();
}

auto TypeChecker::Visitor::resolve_call_overload(
    CallExpressionAST* ast, const std::vector<const Type*>& argumentTypes)
    -> CallResolution {
  auto overloadSet = overload_set_of(ast->baseExpression);
  if (!overloadSet) return CallResolution::kNoOverloadSet;

  const auto overloadFunctions = overloadSet->functions();

  OverloadResolution resolution(check.unit_);

  int argCount = 0;
  for (auto it = ast->expressionList; it; it = it->next) ++argCount;

  bool isMemberCall = false;
  CvQualifiers objectCv{};
  const Type* objectType = nullptr;
  ValueCategory objectValueCategory = ValueCategory::kPrValue;
  if (auto access = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    isMemberCall = true;
    objectType = access->baseExpression->type;
    objectValueCategory = access->baseExpression->valueCategory;

    if (access->accessOp == TokenKind::T_MINUS_GREATER) {
      if (auto ptrType = as_pointer(objectType))
        objectType = ptrType->elementType();
    }

    auto unqualifiedObjectType = objectType;
    objectCv = strip_cv(unqualifiedObjectType);
  }

  std::vector<FunctionSymbol*> allFunctions;
  for (auto func : overloadFunctions) {
    if (func->isSpecialization()) continue;
    if (isPureFriend(func)) continue;
    auto canonical = func->canonical();
    if (std::ranges::contains(allFunctions, canonical)) continue;
    allFunctions.push_back(canonical);
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
      idExpr && !idExpr->nestedNameSpecifier) {
    auto isClassMember = [](FunctionSymbol* func) {
      return !func->isFriend() && func->parent() && func->parent()->isClass();
    };

    if (std::ranges::none_of(allFunctions, isClassMember)) {
      auto adlCandidates = argumentDependentLookup(
          check.unit_, overloadSet->name(), argumentTypes);
      for (auto f : adlCandidates)
        if (std::find(allFunctions.begin(), allFunctions.end(), f) ==
            allFunctions.end())
          allFunctions.push_back(f);
    }

    if (allFunctions.empty() && isArgumentDependentCallee(overloadSet)) {
      report_unresolved_id(idExpr);
      return CallResolution::kFailed;
    }
  }

  if (!isMemberCall && !allFunctions.empty() &&
      !allFunctions.front()->isStatic()) {
    if (auto funcSym = enclosing_function();
        funcSym && symbol_cast<ClassSymbol>(funcSym->parent())) {
      if (auto funcType = type_cast<FunctionType>(funcSym->type())) {
        isMemberCall = true;
        objectCv = funcType->cvQualifiers();
        objectValueCategory = ValueCategory::kLValue;
      }
    }
  }

  std::vector<Candidate> candidates;

  auto explicitTemplateArguments =
      explicit_template_arguments(ast->baseExpression);

  bool hasDependentTemplateCandidate = false;

  std::vector<std::pair<FunctionSymbol*, std::string>> rejected;
  auto reject = [&](FunctionSymbol* func, std::string reason) {
    rejected.emplace_back(func, std::move(reason));
  };
  auto rejectArity = [&](FunctionSymbol* func, int paramCount) {
    reject(func, std::format("requires {} argument{}, but {} {} provided",
                             paramCount, paramCount == 1 ? "" : "s", argCount,
                             argCount == 1 ? "was" : "were"));
  };

  for (auto func : allFunctions) {
    auto type = type_cast<FunctionType>(func->type());
    if (!type) continue;

    const bool templateCandidate =
        func->templateDeclaration() != nullptr && !func->isSpecialization();
    List<TemplateArgumentAST*>* deducedArgsForCandidate = nullptr;

    if (func->templateDeclaration() && !func->isSpecialization()) {
      if (templateCandidateArityRejects(func, argCount)) {
        rejectArity(func, static_cast<int>(type->parameterTypes().size()));
        continue;
      }

      auto deducedArgs = deduceTemplateArguments(func, ast->expressionList,
                                                 explicitTemplateArguments);
      if (!deducedArgs.has_value()) {
        if (in_template()) {
          hasDependentTemplateCandidate = true;
        } else {
          reject(func, "template argument deduction failed");
        }
        continue;
      }

      if (in_template() &&
          std::ranges::any_of(
              ListView{*deducedArgs}, [&](TemplateArgumentAST* arg) {
                return isDependentTemplateArgument(check.unit_, arg);
              })) {
        hasDependentTemplateCandidate = true;
        continue;
      }
      auto instFunc = ASTRewriter::instantiateForArgs(
          check.unit_, *deducedArgs, func,
          ast->baseExpression->firstSourceLocation(), /*argsComplete=*/true,
          /*declarationOnly=*/!functionTemplateHasPackParameter(func));
      if (!instFunc) {
        if (in_template()) {
          hasDependentTemplateCandidate = true;
        } else {
          reject(func, "substitution failure");
        }
        continue;
      }
      func = instFunc;
      type = type_cast<FunctionType>(func->type());
      if (!type) continue;
      deducedArgsForCandidate = *deducedArgs;
    }

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) {
      rejectArity(func, paramCount);
      continue;
    }
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(func, paramCount)) {
        rejectArity(func, paramCount);
        continue;
      }
    }

    auto constraintsSatisfied =
        ASTRewriter::evaluateAssociatedConstraints(check.unit_, func);
    if (!constraintsSatisfied.has_value()) {
      hasDependentTemplateCandidate = true;
      continue;
    }
    if (!*constraintsSatisfied) {
      reject(func, "constraints not satisfied");
      continue;
    }

    Candidate cand{func};
    cand.viable = true;
    cand.fromTemplate = templateCandidate;
    cand.deducedTemplateArgs = deducedArgsForCandidate;

    if (isMemberCall) {
      auto objectConversion = resolution.implicitObjectArgumentConversion(
          func, {.type = objectType,
                 .cv = objectCv,
                 .valueCategory = objectValueCategory});
      if (!objectConversion) {
        reject(func, objectConversion.error());
        continue;
      }
      cand.objectConversion = *objectConversion;
    }

    auto paramIt = type->parameterTypes().begin();
    auto paramEnd = type->parameterTypes().end();
    for (auto argIt = ast->expressionList; argIt && paramIt != paramEnd;
         argIt = argIt->next, ++paramIt) {
      if (in_template() && is_dependent_type(*paramIt)) {
        hasDependentTemplateCandidate = true;
        cand.viable = false;
        break;
      }
      auto conv =
          resolution.computeImplicitConversionSequence(argIt->value, *paramIt);
      if (conv.rank == ConversionRank::kNone) {
        cand.viable = false;
        reject(func,
               std::format(
                   "no known conversion from '{}' to '{}' for "
                   "argument {}",
                   to_string(argIt->value->type), to_string(*paramIt),
                   std::distance(type->parameterTypes().begin(), paramIt) + 1));
        break;
      }
      cand.conversions.push_back(conv);
    }

    if (cand.viable && type->isVariadic()) {
      ImplicitConversionSequence ellipsisConv;
      ellipsisConv.kind = ConversionSequenceKind::kEllipsis;
      ellipsisConv.rank = ConversionRank::kConversion;

      for (auto i = paramCount; i < argCount; ++i)
        cand.conversions.push_back(ellipsisConv);
    }

    if (cand.viable) candidates.push_back(std::move(cand));
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(candidates, true);

  if (!bestPtr) {
    if (hasDependentTemplateCandidate) {
      ast->type = dependent_type();
      ast->valueCategory = ValueCategory::kPrValue;
      return CallResolution::kFailed;
    }
    error(ast->firstSourceLocation(),
          std::format("no matching function for call to '{}'",
                      to_string(overloadSet->name())));
    for (auto& [func, reason] : rejected) {
      check.note(func->location(),
                 std::format("candidate function not viable: {}", reason));
    }
    return CallResolution::kFailed;
  }
  if (ambiguous) {
    error(ast->firstSourceLocation(), "call to function is ambiguous");
    return CallResolution::kFailed;
  }

  auto function = bestPtr->symbol;
  if (function->isSpecialization()) {
    ASTRewriter::reportPendingInstantiationErrors(
        check.unit_, function->primaryTemplateSymbol(), function,
        ast->baseExpression->firstSourceLocation());
  }
  auto selectedFunctionType = named_symbol_type(function);
  ast->baseExpression->type = selectedFunctionType;
  set_base_symbol(ast->baseExpression, function);

  auto selectedType = type_cast<FunctionType>(selectedFunctionType);
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

  return CallResolution::kResolved;
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

  if (!functionType && traits.is_pointer(ast->baseExpression->type)) {
    functionType = type_cast<FunctionType>(
        traits.get_element_type(ast->baseExpression->type));
    if (functionType) (void)stdconv_.ensurePrvalue(ast->baseExpression);
  }

  if (!functionType) functionType = resolve_call_operator(ast);

  return functionType;
}

auto TypeChecker::Visitor::resolve_call_operator(CallExpressionAST* ast)
    -> const FunctionType* {
  auto baseType = traits.remove_cvref(ast->baseExpression->type);
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

  bool anyDeductionSucceeded = false;

  std::vector<Candidate> viableCandidates;
  for (auto pattern : allFunctions) {
    auto func = pattern;
    const bool fromTemplate =
        pattern->templateDeclaration() && !pattern->isSpecialization();

    if (fromTemplate) {
      auto deducedArgs =
          deduceTemplateArguments(pattern, ast->expressionList, nullptr);
      if (!deducedArgs.has_value()) continue;
      anyDeductionSucceeded = true;
      func = ASTRewriter::instantiateForArgs(
          check.unit_, *deducedArgs, pattern,
          ast->baseExpression->firstSourceLocation(), false);
      if (!func) continue;
    }

    auto type = type_cast<FunctionType>(func->type());
    if (!type) continue;

    auto paramCount = static_cast<int>(type->parameterTypes().size());
    if (argCount > paramCount && !type->isVariadic()) continue;
    if (argCount < paramCount) {
      if (argCount < getMinRequiredArgs(func, paramCount)) continue;
    }

    Candidate cand{func};
    cand.viable = true;
    cand.fromTemplate = fromTemplate;

    auto objectConversion = resolution.implicitObjectArgumentConversion(
        func, {.type = baseType,
               .cv = objectCv,
               .valueCategory = objectValueCategory});
    if (!objectConversion) continue;
    cand.objectConversion = *objectConversion;

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

  if (viableCandidates.empty() && anyDeductionSucceeded) {
    for (auto func : allFunctions) {
      if (!func->templateDeclaration() || func->isSpecialization()) continue;
      auto deducedArgs =
          deduceTemplateArguments(func, ast->expressionList, nullptr);
      if (!deducedArgs.has_value()) continue;
      (void)ASTRewriter::instantiate(check.unit_, *deducedArgs, func,
                                     ast->baseExpression->firstSourceLocation(),
                                     false);
    }
    return nullptr;
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(viableCandidates, true);

  if (!bestPtr) return nullptr;
  if (ambiguous) {
    error(ast->firstSourceLocation(),
          "call to overloaded operator() is ambiguous");
    return nullptr;
  }

  auto operatorFunc = bestPtr->symbol;
  if (operatorFunc->isSpecialization()) {
    ASTRewriter::reportPendingInstantiationErrors(
        check.unit_, operatorFunc->primaryTemplateSymbol(), operatorFunc,
        ast->baseExpression->firstSourceLocation());
  }
  auto functionType = type_cast<FunctionType>(named_symbol_type(operatorFunc));
  if (!functionType) return nullptr;

  auto totalParams = static_cast<int>(functionType->parameterTypes().size());
  appendDefaultArguments(ast, operatorFunc, argCount, totalParams);

  auto ar = arena();
  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);
  ast->baseExpression = MemberExpressionAST::create(
      ar, ast->baseExpression, nullptr, opId, operatorFunc, TokenKind::T_DOT,
      false, ValueCategory::kLValue, functionType);

  return functionType;
}

auto TypeChecker::Visitor::resolve_arrow_operator(MemberExpressionAST* ast)
    -> FunctionSymbol* {
  auto classType =
      type_cast<ClassType>(traits.remove_cv(ast->baseExpression->type));
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

    auto objectConversion = resolution.implicitObjectArgumentConversion(
        func, {.type = objectType, .cv = objectCv, .valueCategory = objectVC});
    if (!objectConversion) continue;
    cand.objectConversion = *objectConversion;

    viableCandidates.push_back(std::move(cand));
  }

  auto [bestPtr, ambiguous] =
      resolution.selectBestViableFunction(viableCandidates);
  if (!bestPtr || ambiguous) return nullptr;

  auto operatorFunc = bestPtr->symbol;
  auto functionType = type_cast<FunctionType>(named_symbol_type(operatorFunc));
  if (!functionType) return nullptr;

  auto ar = arena();
  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_MINUS_GREATER);

  auto memberAccess = MemberExpressionAST::create(
      ar, ast->baseExpression, nullptr, opId, operatorFunc, TokenKind::T_DOT,
      false, ValueCategory::kLValue, functionType);

  auto callExpr = CallExpressionAST::create(
      ar, memberAccess, /*expressionList=*/nullptr,
      is_virtual_member_operator_dispatch(operatorFunc, ast->baseExpression),
      /*constructorSymbol=*/nullptr, ValueCategory::kPrValue,
      functionType->returnType());

  ast->baseExpression = callExpr;
  return operatorFunc;
}

void TypeChecker::Visitor::check_function_arguments(
    List<ExpressionAST*>* arguments, SourceLocation callLoc,
    const FunctionType* functionType) {
  const auto& parameterTypes = functionType->parameterTypes();

  OverloadResolution resolution(check.unit_);

  int argc = 0;
  for (auto it = arguments; it; it = it->next) {
    if (!it->value) {
      error(callLoc, "invalid call with null argument expression");
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

    if (auto bracedInitList = ast_cast<BracedInitListAST>(it->value)) {
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
          for (auto elemIt = bracedInitList->expressionList; elemIt;
               elemIt = elemIt->next) {
            (void)check.implicit_conversion(elemIt->value, elemType);
          }
        }
        resolution.applyImplicitConversion(seq, it->value);
      }
      continue;
    }

    if (isCxx() && traits.is_reference(targetType)) {
      auto seq =
          resolution.computeImplicitConversionSequence(it->value, targetType);
      if (seq.rank == ConversionRank::kNone) {
        error(it->value->firstSourceLocation(),
              std::format("invalid argument of type '{}' for parameter of "
                          "type '{}'",
                          to_string(it->value->type), to_string(targetType)));
      } else {
        resolution.applyImplicitConversion(seq, it->value);
      }
      continue;
    }

    if (!implicit_conversion(it->value, targetType)) {
      error(it->value->firstSourceLocation(),
            std::format("invalid argument of type '{}' for parameter of type "
                        "'{}'",
                        to_string(it->value->type), to_string(targetType)));
      continue;
    }

    check.reportDeletedConversion(it->value);
  }
}

auto TypeChecker::Visitor::checkBuiltinInvoke(CallExpressionAST* ast) -> bool {
  std::vector<ExpressionAST*> args;
  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value || !it->value->type) return true;
    args.push_back(it->value);
  }

  if (args.empty()) {
    error(ast->firstSourceLocation(),
          "too few arguments to '__builtin_invoke'");
    return true;
  }

  auto restArgs = [&](std::size_t from) -> List<ExpressionAST*>* {
    List<ExpressionAST*>* result = nullptr;
    auto tail = &result;
    for (auto i = from; i < args.size(); ++i) {
      *tail = make_list_node<ExpressionAST>(arena(), args[i]);
      tail = &(*tail)->next;
    }
    return result;
  };

  auto fExpr = args[0];
  auto decayedF = traits.decay(fExpr->type);

  auto isDirectOrBase = [&](const Type* classType, const Type* candidate) {
    return traits.is_same(classType, candidate) ||
           traits.is_base_of(classType, candidate);
  };

  if (auto mfp = type_cast<MemberFunctionPointerType>(decayedF)) {
    if (args.size() < 2) {
      error(ast->firstSourceLocation(),
            "too few arguments to '__builtin_invoke'");
      return true;
    }

    auto decayedA0 = traits.decay(args[1]->type);
    auto classType = mfp->classType();

    bool ok = isDirectOrBase(classType, decayedA0);
    if (!ok) {
      if (auto ptr = type_cast<PointerType>(decayedA0)) {
        ok = isDirectOrBase(classType, traits.decay(ptr->elementType()));
      }
    }

    if (!ok) {
      error(ast->firstSourceLocation(),
            "no matching function for call to '__builtin_invoke'");
      return true;
    }

    ast->expressionList = restArgs(2);
    check_function_arguments(ast->expressionList, ast->firstSourceLocation(),
                             mfp->functionType());
    setResultTypeAndValueCategory(ast, mfp->functionType());
    return true;
  }

  if (auto mop = type_cast<MemberObjectPointerType>(decayedF)) {
    if (args.size() != 2) {
      error(ast->firstSourceLocation(),
            "'__builtin_invoke' on a pointer to member object takes exactly "
            "one object argument");
      return true;
    }

    auto decayedA0 = traits.decay(args[1]->type);
    auto classType = mop->classType();

    if (isDirectOrBase(classType, decayedA0)) {
      ast->type = mop->elementType();
      ast->valueCategory = args[1]->valueCategory == ValueCategory::kLValue
                               ? ValueCategory::kLValue
                               : ValueCategory::kXValue;
      return true;
    }

    if (auto ptr = type_cast<PointerType>(decayedA0);
        ptr && isDirectOrBase(classType, traits.decay(ptr->elementType()))) {
      ast->type = mop->elementType();
      ast->valueCategory = ValueCategory::kLValue;
      return true;
    }

    error(ast->firstSourceLocation(),
          "no matching function for call to '__builtin_invoke'");
    return true;
  }

  ast->baseExpression = fExpr;
  ast->expressionList = restArgs(1);
  ast->type = nullptr;
  ast->valueCategory = ValueCategory::kNone;
  (*this)(ast);
  return true;
}

auto TypeChecker::Visitor::checkBuiltinAddressof(CallExpressionAST* ast)
    -> bool {
  auto args = ListView{ast->expressionList};
  auto it = args.begin();

  if (it == args.end() || !(*it)->type) {
    error(ast->firstSourceLocation(),
          "too few arguments to '__builtin_addressof'");
    return true;
  }

  auto arg = *it;

  if (arg->valueCategory == ValueCategory::kPrValue) {
    error(arg->firstSourceLocation(),
          std::format("cannot take the address of an rvalue of type '{}'",
                      to_string(arg->type)));
  }

  ast->type = control()->getPointerType(arg->type);
  ast->valueCategory = ValueCategory::kPrValue;
  return true;
}

void TypeChecker::Visitor::check_member_pointer_access(
    BinaryExpressionAST* ast) {
  auto& object = ast->leftExpression;
  auto& memberPointer = ast->rightExpression;

  (void)stdconv_.ensurePrvalue(memberPointer);

  const auto throughPointer = ast->op == TokenKind::T_MINUS_GREATER_STAR;
  if (throughPointer) (void)stdconv_.ensurePrvalue(object);

  auto objectType = traits.remove_reference(object->type);
  if (throughPointer) {
    auto pointerType = type_cast<PointerType>(objectType);
    if (!pointerType) {
      error(ast->firstSourceLocation(),
            std::format("left operand of '->*' must be a pointer to class "
                        "(was '{}')",
                        to_string(object->type)));
      return;
    }
    objectType = pointerType->elementType();
  }

  const auto objectQualifiers = traits.get_cv_qualifiers(objectType);

  const auto qualified = [&](const Type* memberType) {
    const auto memberQualifiers = traits.get_cv_qualifiers(memberType);
    const auto combined =
        static_cast<CvQualifiers>(static_cast<int>(objectQualifiers) |
                                  static_cast<int>(memberQualifiers));
    if (combined == memberQualifiers) return memberType;
    return static_cast<const Type*>(
        control()->getQualType(traits.remove_cv(memberType), combined));
  };

  if (auto dataPointer =
          type_cast<MemberObjectPointerType>(memberPointer->type)) {
    ast->type = qualified(dataPointer->elementType());
    ast->valueCategory = throughPointer || is_lvalue(object)
                             ? ValueCategory::kLValue
                             : ValueCategory::kXValue;
    return;
  }

  if (auto functionPointer =
          type_cast<MemberFunctionPointerType>(memberPointer->type)) {
    ast->type = functionPointer->functionType();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  error(ast->firstSourceLocation(),
        std::format("right operand of '{}' must be a pointer to member "
                    "(was '{}')",
                    Token::spell(ast->op), to_string(memberPointer->type)));
}

auto TypeChecker::Visitor::checkBuiltinVaListAccess(CallExpressionAST* ast)
    -> bool {
  auto vaListArg = ast->expressionList;

  if (!vaListArg || !vaListArg->value || !vaListArg->value->type) return true;

  if (!type_cast<BuiltinVaListType>(
          traits.remove_cvref(vaListArg->value->type))) {
    error(vaListArg->value->firstSourceLocation(),
          std::format("expected an object of type '__builtin_va_list', not "
                      "'{}'",
                      to_string(vaListArg->value->type)));
    return true;
  }

  if (!is_lvalue(vaListArg->value)) {
    error(vaListArg->value->firstSourceLocation(),
          "expected an lvalue of type '__builtin_va_list'");
    return true;
  }

  for (auto it = vaListArg->next; it; it = it->next) {
    if (!it->value || !it->value->type) continue;
    if (type_cast<BuiltinVaListType>(traits.remove_cvref(it->value->type)))
      continue;
    (void)stdconv_.ensurePrvalue(it->value);
    stdconv_.adjustCv(it->value);
  }

  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
  return true;
}

auto TypeChecker::Visitor::checkBuiltinAssumeAligned(CallExpressionAST* ast)
    -> bool {
  auto pointerArg = ast->expressionList;
  auto alignmentArg = pointerArg ? pointerArg->next : nullptr;
  auto misalignmentArg = alignmentArg ? alignmentArg->next : nullptr;

  if (!alignmentArg) {
    error(ast->firstSourceLocation(),
          "too few arguments to '__builtin_assume_aligned', expected 2");
    return true;
  }

  if (misalignmentArg && misalignmentArg->next) {
    error(ast->firstSourceLocation(),
          "too many arguments to '__builtin_assume_aligned', expected at "
          "most 3");
    return true;
  }

  if (!alignmentArg->value) return false;

  if (misalignmentArg && misalignmentArg->value) {
    auto sizeType = control()->getSizeType();
    if (!implicit_conversion(misalignmentArg->value, sizeType)) {
      error(misalignmentArg->value->firstSourceLocation(),
            std::format("invalid argument of type '{}' for parameter of type "
                        "'{}'",
                        to_string(misalignmentArg->value->type),
                        to_string(sizeType)));
    }
  }

  auto interp = ASTInterpreter{check.unit_};
  auto alignmentLoc = alignmentArg->value->firstSourceLocation();
  auto value = interp.evaluate(alignmentArg->value);
  auto alignment = value.has_value() ? interp.toUInt(*value) : std::nullopt;

  if (!alignment.has_value()) {
    if (isDependent(check.unit_, alignmentArg->value)) return false;
    error(alignmentLoc,
          "argument to '__builtin_assume_aligned' must be a constant integer");
    return true;
  }

  if (!std::has_single_bit(*alignment)) {
    error(alignmentLoc, "requested alignment is not a power of 2");
    return true;
  }

  if (*alignment > kMaximumAlignment) {
    warning(alignmentLoc,
            std::format("requested alignment must be {} bytes or smaller; "
                        "maximum alignment assumed",
                        kMaximumAlignment));
    alignment = kMaximumAlignment;
  }

  auto sizeType = control()->getSizeType();
  (void)implicit_conversion(alignmentArg->value, sizeType);

  auto folded = ImplicitCastExpressionAST::create(check.unit_->arena());
  folded->castKind = ImplicitCastKind::kIdentity;
  folded->expression = alignmentArg->value;
  folded->type = alignmentArg->value->type;
  folded->valueCategory = ValueCategory::kPrValue;
  folded->constValue = check.unit_->arena()->make<ConstValue>(
      static_cast<std::intmax_t>(*alignment));
  alignmentArg->value = folded;

  return false;
}

auto TypeChecker::Visitor::checkBuiltinCountZerosGeneric(CallExpressionAST* ast)
    -> bool {
  auto first = ast->expressionList;

  if (!first || !first->value || !first->value->type) {
    error(ast->firstSourceLocation(),
          std::format("too few arguments to '{}'",
                      check.unit_->tokenText(ast->firstSourceLocation())));
    return true;
  }

  auto builtinName = check.unit_->tokenText(ast->firstSourceLocation());
  auto fallback = first->next;

  if (fallback && fallback->next) {
    error(ast->firstSourceLocation(),
          std::format("too many arguments to '{}', expected at most 2",
                      builtinName));
    return true;
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    if (it->value) (void)stdconv_.ensurePrvalue(it->value);
  }

  if (auto type = first->value->type;
      !traits.is_integral(type) || !traits.is_unsigned(type)) {
    error(first->value->firstSourceLocation(),
          std::format("1st argument to '{}' must be a scalar unsigned integer "
                      "type (was '{}')",
                      builtinName, to_string(type)));
  }

  if (fallback && fallback->value) {
    auto intType = control()->getIntType();
    if (fallback->value->type != intType) {
      error(fallback->value->firstSourceLocation(),
            std::format("2nd argument to '{}' must be a scalar 'int' type "
                        "(was '{}')",
                        builtinName, to_string(fallback->value->type)));
    }
  }

  ast->type = control()->getIntType();
  ast->valueCategory = ValueCategory::kPrValue;
  return true;
}

void TypeChecker::Visitor::mark_virtual_dispatch(CallExpressionAST* ast) {
  if (auto member = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    auto function = symbol_cast<FunctionSymbol>(member->symbol);
    if (!function || !function->isVirtual()) return;
    if (!function->isImplicitObjectMemberFunction()) return;
    if (member->nestedNameSpecifier) return;

    if (member->accessOp == TokenKind::T_MINUS_GREATER) {
      ast->isVirtualDispatch = true;
      return;
    }

    if (!is_glvalue(member->baseExpression)) return;
    ast->isVirtualDispatch = !is_known_complete_object(member->baseExpression);
    return;
  }

  if (auto id = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    auto function = symbol_cast<FunctionSymbol>(id->symbol);
    if (!function || !function->isVirtual()) return;
    if (!function->isImplicitObjectMemberFunction()) return;
    if (id->nestedNameSpecifier) return;
    ast->isVirtualDispatch = true;
  }
}

void TypeChecker::Visitor::operator()(CallExpressionAST* ast) {
  if (!ast->baseExpression) return;

  if (isUntypedAfterError(ast->baseExpression)) {
    markUntypedAfterError(ast);
    return;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto classSym = symbol_cast<ClassSymbol>(idExpr->symbol)) {
      ast->type = classSym->type();
      ast->valueCategory = ValueCategory::kPrValue;

      traits.requireCompleteClass(classSym);

      ExpressionAST* initializer = nullptr;
      ast->constructorSymbol = check.check_class_initializer(
          ast->type, initializer, ast->lparenLoc, &ast->expressionList);
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

  const auto hasUntypedArgument = std::ranges::any_of(
      argumentTypes, [](const Type* argType) { return !argType; });

  auto isUndeducedType = [&](const Type* argType) {
    return is_dependent_type(argType) ||
           (in_template() && type_cast<AutoType>(traits.remove_cv(argType)));
  };

  if (std::ranges::any_of(argumentTypes, isUndeducedType) ||
      (in_template() && hasUntypedArgument)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (std::ranges::any_of(ListView{ast->expressionList}, isUntypedAfterError)) {
    markUntypedAfterError(ast);
    return;
  }

  if (auto access = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    if (ast_cast<DestructorIdAST>(access->unqualifiedId)) {
      ast->type = control()->getVoidType();
      return;
    }
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
      if (auto identifier = nameId->identifier) {
        auto builtinKind = identifier->builtinFunction();
        if (builtinKind != BuiltinFunctionKind::T_NONE) {
          if (typeCheckBuiltinDispatch(ast, builtinKind)) return;
        }
      }
    }
  }

  if (resolve_call_overload(ast, argumentTypes) == CallResolution::kFailed)
    return;

  auto functionType = resolve_function_type(ast);
  if (!functionType) {
    if (type_cast<OverloadSetType>(ast->baseExpression->type)) return;

    if (ast->baseExpression->type) {
      error(ast->baseExpression->firstSourceLocation(),
            std::format("called object of type '{}' is not a function or "
                        "function pointer",
                        to_string(ast->baseExpression->type)));
    }

    return;
  }

  if (auto funcSym =
          symbol_cast<FunctionSymbol>(base_symbol(ast->baseExpression))) {
    check.reportDeletedFunction(funcSym, ast->firstSourceLocation());
    check.requireFunctionDefinition(funcSym);

    int argCount = 0;
    for (auto it = ast->expressionList; it; it = it->next) ++argCount;
    int totalParams = static_cast<int>(functionType->parameterTypes().size());
    appendDefaultArguments(ast, funcSym, argCount, totalParams);
  }

  check_function_arguments(ast->expressionList, ast->firstSourceLocation(),
                           functionType);

  mark_virtual_dispatch(ast);

  setResultTypeAndValueCategory(ast, functionType);

  if (ast->valueCategory == ValueCategory::kPrValue) stdconv_.adjustCv(ast);

  resolveBuiltinLibcall(ast);
}

void TypeChecker::Visitor::resolveBuiltinLibcall(CallExpressionAST* ast) {
  auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
  if (!idExpr) return;
  auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId);
  if (!nameId || !nameId->identifier) return;

  auto kind = nameId->identifier->builtinFunction();
  if (kind == BuiltinFunctionKind::T_NONE) return;

  const auto isOperatorNew =
      kind == BuiltinFunctionKind::T___BUILTIN_OPERATOR_NEW;

  if (isOperatorNew ||
      kind == BuiltinFunctionKind::T___BUILTIN_OPERATOR_DELETE) {
    std::vector<const Type*> argumentTypes;
    for (auto it = ast->expressionList; it; it = it->next)
      argumentTypes.push_back(it->value ? it->value->type : nullptr);
    ast->constructorSymbol =
        isOperatorNew
            ? resolveBuiltinOperatorNew(check.unit_, argumentTypes)
            : resolveBuiltinOperatorDelete(check.unit_, argumentTypes);
    return;
  }

  if (!isBuiltinLibcall(kind)) return;

  auto funcType = type_cast<FunctionType>(idExpr->type);
  if (!funcType && idExpr->symbol)
    funcType = type_cast<FunctionType>(idExpr->symbol->type());
  if (!funcType) return;

  const std::string_view spelling = Token::spell(kind);
  const std::string_view prefix = "__builtin_";
  const auto libcName = std::string(spelling.substr(prefix.size()));

  ast->constructorSymbol =
      resolveBuiltinLibcallSymbol(check.unit_, libcName.c_str(), funcType);
}

void TypeChecker::Visitor::setResultTypeAndValueCategory(
    ExpressionAST* ast, FunctionSymbol* function) {
  setResultTypeAndValueCategory(ast, named_symbol_type(function));
}

void TypeChecker::Visitor::setResultTypeAndValueCategory(ExpressionAST* ast,
                                                         const Type* type) {
  auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return;
  ast->type = functionType->returnType();

  if (traits.is_lvalue_reference(ast->type)) {
    ast->type = traits.remove_reference(ast->type);
    ast->valueCategory = ValueCategory::kLValue;
  } else if (traits.is_rvalue_reference(ast->type)) {
    ast->type = traits.remove_reference(ast->type);
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

  if (traits.is_void(traits.remove_cv(ast->type))) {
    if (ast->expressionList && ast->expressionList->next) {
      error(ast->expressionList->next->value->firstSourceLocation(),
            "excess elements in 'void' initializer");
    }
    return;
  }

  if (ClassTemplateArgumentDeduction::placeholderClassTemplate(
          ast->typeSpecifier, check.scope())) {
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->expressionList})
      arguments.push_back(argument);

    auto deduced = check.deduceClassTemplateSpecialization(
        ast->typeSpecifier, arguments, /*isListInitialization=*/false,
        /*isCopyInitialization=*/false, ast->lparenLoc);

    if (!deduced) return;
    ast->type = deduced;
  }

  if (auto classType = type_cast<ClassType>(traits.remove_cv(ast->type))) {
    traits.requireCompleteClass(classType->symbol());

    if (!classType->symbol()) return;

    ExpressionAST* initializer = nullptr;
    ast->constructorSymbol = check.check_class_initializer(
        ast->type, initializer, ast->lparenLoc, &ast->expressionList);
  } else if (ast->expressionList && !ast->expressionList->next) {
    (void)check_static_cast(ast->expressionList->value, ast->type,
                            ValueCategory::kPrValue);
  }
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

  if (traits.is_void(traits.remove_cv(ast->type))) {
    if (auto elements = ast->bracedInitList->expressionList) {
      error(elements->value->firstSourceLocation(),
            "excess elements in 'void' initializer");
    }
    return;
  }

  if (ClassTemplateArgumentDeduction::placeholderClassTemplate(
          ast->typeSpecifier, check.scope())) {
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->bracedInitList->expressionList})
      arguments.push_back(argument);

    auto deduced = check.deduceClassTemplateSpecialization(
        ast->typeSpecifier, arguments, /*isListInitialization=*/true,
        /*isCopyInitialization=*/false, ast->bracedInitList->lbraceLoc);

    if (!deduced) return;
    ast->type = deduced;
  }

  if (auto classType = type_cast<ClassType>(traits.remove_cv(ast->type))) {
    traits.requireCompleteClass(classType->symbol());

    auto classSymbol = classType->symbol();
    if (!classSymbol) return;
    if (has_initializer_list_constructor(classSymbol)) return;

    ExpressionAST* initializer = ast->bracedInitList;
    ast->constructorSymbol = check.check_class_initializer(
        ast->type, initializer, ast->bracedInitList->lbraceLoc);
  }
}

auto TypeChecker::Visitor::has_initializer_list_constructor(
    ClassSymbol* classSymbol) -> bool {
  for (auto ctor : classSymbol->constructors()) {
    auto functionType = type_cast<FunctionType>(ctor->type());
    if (!functionType) continue;
    const auto& params = functionType->parameterTypes();
    if (params.empty()) continue;
    auto paramType = traits.remove_cv(traits.remove_reference(params[0]));
    if (traits.initializer_list_element_type(paramType)) return true;
  }
  return false;
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
    ast->type = traits.remove_reference(named_symbol_type(ast->symbol));
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

  if (ast->accessOp == TokenKind::T_DOT &&
      traits.is_class(ast->baseExpression->type)) {
    (void)stdconv_.temporaryMaterialization(ast->baseExpression);
  }

  if (ast->symbol && ast->type && !is_dependent_type(ast->type) &&
      !symbol_cast<OverloadSetSymbol>(ast->symbol)) {
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
  if (in_template() && is_dependent_type(ast->baseExpression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (traits.is_class(ast->baseExpression->type)) {
    if (auto operatorFunc = check.lookupOperator(
            ast->baseExpression->type, ast->op, control()->getIntType())) {
      ast->symbol = operatorFunc;
      adjust_member_operator_object_argument(operatorFunc, ast->baseExpression);
      ast->isVirtualDispatch = is_virtual_member_operator_dispatch(
          operatorFunc, ast->baseExpression);
      setResultTypeAndValueCategory(ast, operatorFunc);
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

  if (!is_glvalue(ast->baseExpression)) {
    error(ast->opLoc, std::format("cannot {} an rvalue of type '{}'", op,
                                  to_string(ast->baseExpression->type)));
    return;
  }

  auto incr_arithmetic = [&]() {
    if (traits.is_const(ast->baseExpression->type)) return false;

    if (isCxx() && !traits.is_arithmetic(ast->baseExpression->type))
      return false;

    if (isC() &&
        !traits.is_arithmetic_or_unscoped_enum(ast->baseExpression->type))
      return false;

    auto ty = traits.remove_cv(ast->baseExpression->type);
    if (type_cast<BoolType>(ty)) return false;

    ast->type = ty;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto incr_pointer = [&]() {
    if (!traits.is_pointer(ast->baseExpression->type)) return false;
    auto ty = traits.remove_cv(ast->baseExpression->type);
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

  if (!ast->type || !ast->expression || !ast->expression->type) return;

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

  if (traits.is_void(targetType)) return true;

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (check_static_cast_to_derived_ref(expression, targetType, targetVC))
      return true;
  }

  auto bind_reference = [&] {
    auto t1 = traits.remove_cv(targetType);
    auto t2 = traits.remove_cv(expression->type);
    if (!traits.is_same(t1, t2) && traits.is_class(t1) && traits.is_class(t2)) {
      check.wrapWithImplicitCast(ImplicitCastKind::kDerivedToBaseConversion,
                                 targetType, expression);
    }
    return true;
  };

  if (targetVC == ValueCategory::kXValue && is_lvalue(expression)) {
    if (is_reference_compatible(targetType, expression->type))
      return bind_reference();
  }

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (is_glvalue(expression) &&
        is_reference_compatible(targetType, expression->type))
      return bind_reference();
  }

  if (targetVC == ValueCategory::kPrValue) {
    if (implicit_conversion(expression, targetType,
                            InitializationKind::kDirectInitialization))
      return true;
  }

  auto source = expression;
  (void)stdconv_.ensurePrvalue(source);
  stdconv_.adjustCv(source);

  auto sourceType = source->type;

  if (traits.is_scoped_enum(sourceType) &&
      (traits.is_integral(targetType) ||
       traits.is_floating_point(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       traits.is_integral(targetType)
                           ? ImplicitCastKind::kIntegralConversion
                           : ImplicitCastKind::kFloatingIntegralConversion);
    return true;
  }

  if ((traits.is_integral(sourceType) || traits.is_enum(sourceType) ||
       traits.is_scoped_enum(sourceType)) &&
      (traits.is_enum(targetType) || traits.is_scoped_enum(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kIntegralConversion);
    return true;
  }

  if (traits.is_floating_point(sourceType) &&
      (traits.is_enum(targetType) || traits.is_scoped_enum(targetType))) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kFloatingIntegralConversion);
    return true;
  }

  if (traits.is_floating_point(sourceType) &&
      traits.is_floating_point(targetType)) {
    emit_implicit_cast(expression, source, targetType,
                       ImplicitCastKind::kFloatingPointConversion);
    return true;
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (auto targetPtr = as_pointer(targetType)) {
      auto srcElem = traits.remove_cv(sourcePtr->elementType());
      auto tgtElem = traits.remove_cv(targetPtr->elementType());
      auto srcCV = traits.get_cv_qualifiers(sourcePtr->elementType());
      auto tgtCV = traits.get_cv_qualifiers(targetPtr->elementType());
      if (traits.is_base_of(srcElem, tgtElem) &&
          !traits.is_virtual_base_of(srcElem, tgtElem) &&
          stdconv_.checkCvQualifiers(tgtCV, srcCV)) {
        emit_implicit_cast(expression, source, targetType,
                           ImplicitCastKind::kBaseToDerivedConversion);
        return true;
      }
    }
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (traits.is_void(traits.remove_cv(sourcePtr->elementType()))) {
      if (auto targetPtr = as_pointer(targetType)) {
        if (traits.is_object(traits.remove_cv(targetPtr->elementType()))) {
          auto srcCV = traits.get_cv_qualifiers(sourcePtr->elementType());
          auto tgtCV = traits.get_cv_qualifiers(targetPtr->elementType());
          if (stdconv_.checkCvQualifiers(tgtCV, srcCV)) {
            expression = source;
            return true;
          }
        }
      }
    }
  }

  if (auto srcMem =
          type_cast<MemberObjectPointerType>(traits.remove_cv(sourceType))) {
    if (auto tgtMem =
            type_cast<MemberObjectPointerType>(traits.remove_cv(targetType))) {
      auto srcClass = srcMem->classType();
      auto tgtClass = tgtMem->classType();
      if (traits.is_base_of(tgtClass, srcClass)) {
        auto srcElemCV = traits.get_cv_qualifiers(srcMem->elementType());
        auto tgtElemCV = traits.get_cv_qualifiers(tgtMem->elementType());
        if (stdconv_.checkCvQualifiers(tgtElemCV, srcElemCV) &&
            traits.is_same(traits.remove_cv(srcMem->elementType()),
                           traits.remove_cv(tgtMem->elementType()))) {
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
  auto srcCV = traits.get_cv_qualifiers(sourceType);
  sourceType = traits.remove_cv(sourceType);

  auto tgtCV = traits.get_cv_qualifiers(targetType);
  auto tgtBase = traits.remove_cv(targetType);

  if (!stdconv_.checkCvQualifiers(tgtCV, srcCV)) return false;

  if (traits.is_same(sourceType, tgtBase)) return false;
  if (!traits.is_base_of(sourceType, tgtBase)) return false;
  if (traits.is_virtual_base_of(sourceType, tgtBase)) return false;

  check.wrapWithImplicitCast(ImplicitCastKind::kBaseToDerivedConversion,
                             targetType, expression);

  return true;
}

auto TypeChecker::Visitor::is_reference_compatible(const Type* targetType,
                                                   const Type* sourceType)
    -> bool {
  auto t1 = traits.remove_cv(targetType);
  auto t2 = traits.remove_cv(sourceType);
  if (!traits.is_same(t1, t2)) {
    if (!traits.is_base_of(t1, t2)) return false;
  }
  auto cvTarget = traits.get_cv_qualifiers(targetType);
  auto cvSource = traits.get_cv_qualifiers(sourceType);
  return stdconv_.checkCvQualifiers(cvTarget, cvSource);
}

auto TypeChecker::Visitor::check_const_cast(ExpressionAST*& expression,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool {
  if (!targetType) return false;

  auto sourceType = expression->type;
  const Type* T1 = nullptr;
  const Type* T2 = nullptr;

  if (auto targetPtr = type_cast<PointerType>(traits.remove_cv(targetType))) {
    auto sourcePtr = type_cast<PointerType>(traits.remove_cv(sourceType));
    if (!sourcePtr) return false;

    (void)stdconv_.ensurePrvalue(expression);
    stdconv_.adjustCv(expression);

    T1 = sourcePtr->elementType();
    T2 = targetPtr->elementType();
  } else if (auto targetPtrm = type_cast<MemberObjectPointerType>(
                 traits.remove_cv(targetType))) {
    auto sourcePtrm =
        type_cast<MemberObjectPointerType>(traits.remove_cv(sourceType));
    if (!sourcePtrm) return false;

    if (!traits.is_same(sourcePtrm->classType(), targetPtrm->classType()))
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
               (traits.is_class(sourceType) || traits.is_array(sourceType))) {
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
    if (traits.is_same(traits.remove_cv(curr1), traits.remove_cv(curr2))) {
      return true;
    }

    auto u1 = traits.remove_cv(curr1);
    auto u2 = traits.remove_cv(curr2);

    if (auto p1 = as_pointer(u1)) {
      if (auto p2 = as_pointer(u2)) {
        curr1 = p1->elementType();
        curr2 = p2->elementType();
        continue;
      }
    }

    if (auto m1 = type_cast<MemberObjectPointerType>(u1)) {
      if (auto m2 = type_cast<MemberObjectPointerType>(u2)) {
        if (!traits.is_same(m1->classType(), m2->classType())) return false;
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
    auto ptrToSource = traits.add_pointer(sourceType);
    auto ptrToTarget = traits.add_pointer(targetType);
    (void)ptrToSource;
    (void)ptrToTarget;
    if ((traits.is_object(traits.remove_cv(sourceType)) &&
         traits.is_object(traits.remove_cv(targetType))) ||
        (traits.is_function(sourceType) && traits.is_function(targetType))) {
      if (casts_away_constness(sourceType, targetType)) return false;
      return true;
    }
    return false;
  }

  (void)stdconv_.ensurePrvalue(expression);
  stdconv_.adjustCv(expression);
  sourceType = expression->type;

  if (traits.is_same(traits.remove_cv(sourceType),
                     traits.remove_cv(targetType)))
    return true;

  if (traits.is_pointer(sourceType) && traits.is_integral(targetType)) {
    emit_implicit_cast(expression, expression, targetType,
                       ImplicitCastKind::kIntegralConversion);
    return true;
  }

  if ((traits.is_integral(sourceType) || traits.is_enum(sourceType) ||
       traits.is_scoped_enum(sourceType)) &&
      traits.is_pointer(targetType)) {
    return true;
  }

  if (traits.is_pointer(sourceType) && traits.is_pointer(targetType)) {
    auto srcPtr = as_pointer(sourceType);
    auto tgtPtr = as_pointer(targetType);
    if (srcPtr && tgtPtr &&
        casts_away_constness(srcPtr->elementType(), tgtPtr->elementType()))
      return false;
    if (!traits.is_same(traits.remove_cv(sourceType),
                        traits.remove_cv(targetType)))
      emit_implicit_cast(expression, expression, targetType,
                         ImplicitCastKind::kPointerConversion);
    return true;
  }

  if (traits.is_member_pointer(sourceType) &&
      traits.is_member_pointer(targetType)) {
    return true;
  }

  if (traits.is_null_pointer(sourceType) && traits.is_integral(targetType))
    return true;

  return false;
}

auto TypeChecker::Visitor::casts_away_constness(const Type* sourceType,
                                                const Type* targetType)
    -> bool {
  auto srcCV = traits.get_cv_qualifiers(sourceType);
  auto tgtCV = traits.get_cv_qualifiers(targetType);

  if (!stdconv_.checkCvQualifiers(tgtCV, srcCV)) return true;

  auto srcPtr = as_pointer(traits.remove_cv(sourceType));
  auto tgtPtr = as_pointer(traits.remove_cv(targetType));
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
      ast->typeId ? type_cast<ClassType>(traits.remove_cv(ast->typeId->type))
                  : nullptr;

  if (!classType) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  if (!ast->identifier) {
    return;
  }

  auto symbol = classType->symbol();
  traits.requireCompleteClass(symbol);
  symbol = symbol->resolvedDefinition();
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
      auto currentClass =
          type_cast<ClassType>(traits.remove_cvref(field->type()));

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
      if (!traits.is_array(field->type()) &&
          !traits.is_pointer(field->type())) {
        error(subscript->firstSourceLocation(),
              std::format("cannot subscript a member of type '{}'",
                          to_string(field->type())));
        break;
      }

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

  auto targetType = traits.remove_cv(ast->typeId->type);
  auto sourceType = traits.remove_cv(ast->expression->type);

  if (is_dependent_type(targetType) || is_dependent_type(sourceType)) {
    ast->type = ast->typeId->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (traits.is_reference(targetType) || traits.is_reference(sourceType)) {
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

auto TypeChecker::Visitor::comparison_category_type(SourceLocation loc,
                                                    std::string_view name)
    -> const Type* {
  Symbol* categorySymbol = nullptr;

  auto stdId = control()->getIdentifier("std");
  if (auto stdNamespace =
          symbol_cast<NamespaceSymbol>(qualifiedLookup(globalScope(), stdId))) {
    categorySymbol =
        qualifiedLookup(stdNamespace, control()->getIdentifier(name),
                        [](Symbol* s) { return is_type(s); });
  }

  const Type* categoryType = nullptr;
  if (categorySymbol) {
    categoryType =
        type_cast<ClassType>(traits.remove_cv(categorySymbol->type()));
  }

  if (!categoryType) {
    error(loc, "you need to include <compare> before using the '<=>' operator");
    return nullptr;
  }

  return categoryType;
}

void TypeChecker::Visitor::check_three_way_comparison(
    BinaryExpressionAST* ast) {
  ast->valueCategory = ValueCategory::kPrValue;

  auto leftType = traits.remove_cvref(ast->leftExpression->type);
  auto rightType = traits.remove_cvref(ast->rightExpression->type);

  if (traits.is_pointer(leftType) && traits.is_pointer(rightType)) {
    ast->type = comparison_category_type(ast->opLoc, "strong_ordering");
    return;
  }

  auto commonType = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                       ast->rightExpression);

  if (!commonType) {
    error(ast->firstSourceLocation(),
          std::format("invalid operands to binary expression ('{}' and '{}')",
                      to_string(ast->leftExpression->type),
                      to_string(ast->rightExpression->type)));
    return;
  }

  const auto category = traits.is_floating_point(commonType)
                            ? std::string_view{"partial_ordering"}
                            : std::string_view{"strong_ordering"};

  ast->type = comparison_category_type(ast->opLoc, category);
}

auto TypeChecker::Visitor::type_info_type(SourceLocation loc) -> const Type* {
  Symbol* typeInfoSymbol = nullptr;

  auto stdId = control()->getIdentifier("std");
  if (auto stdNamespace =
          symbol_cast<NamespaceSymbol>(qualifiedLookup(globalScope(), stdId))) {
    typeInfoSymbol =
        qualifiedLookup(stdNamespace, control()->getIdentifier("type_info"),
                        [](Symbol* s) { return is_type(s); });
  }

  const Type* typeInfoType = nullptr;
  if (typeInfoSymbol) {
    typeInfoType =
        type_cast<ClassType>(traits.remove_cv(typeInfoSymbol->type()));
  }

  if (!typeInfoType) {
    error(loc,
          "you need to include <typeinfo> before using the 'typeid' operator");
    return nullptr;
  }

  return traits.add_const(typeInfoType);
}

void TypeChecker::Visitor::operator()(TypeidExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (!ast->expression->type) {
    if (!is_unresolved_id(ast->expression)) {
      error(ast->expression->firstSourceLocation(),
            "invalid operand to typeid");
    }
    return;
  }

  if (auto typeInfoType = type_info_type(ast->firstSourceLocation())) {
    ast->type = typeInfoType;
    ast->valueCategory = ValueCategory::kLValue;
  }
}

void TypeChecker::Visitor::operator()(TypeidOfTypeExpressionAST* ast) {
  if (!ast->typeId || !ast->typeId->type) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  if (auto typeInfoType = type_info_type(ast->firstSourceLocation())) {
    ast->type = typeInfoType;
    ast->valueCategory = ValueCategory::kLValue;
  }
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
    if (!is_unresolved_id(ast->splicer->expression)) {
      error(ast->splicer->firstSourceLocation(), "invalid splicer expression");
    }
    return;
  }

  ast->type = ast->splicer->expression->type;
  ast->valueCategory = ValueCategory::kPrValue;

  if (auto reflected = splicedExpression(ast->splicer)) {
    ast->type = reflected->type;
    ast->valueCategory = reflected->valueCategory;
  }
}

auto TypeChecker::Visitor::splicedExpression(SplicerAST* splicer)
    -> ExpressionAST* {
  if (is_dependent_type(splicer->expression->type)) return nullptr;

  auto interp = ASTInterpreter{check.unit_};
  auto value = interp.evaluate(splicer->expression);
  if (!value.has_value()) return nullptr;

  auto metaPtr = std::get_if<std::shared_ptr<Meta>>(&*value);
  if (!metaPtr) return nullptr;

  auto constExpr = std::get_if<Meta::ConstExpr>(&(*metaPtr)->value);
  if (!constExpr) return nullptr;

  return constExpr->expression;
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
    if (!is_unresolved_id(ast->expression)) {
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
  if (idExpr) {
    if (auto function = designatedFunction(idExpr->symbol)) {
      idExpr->symbol = function;
      check.requireFunctionDefinition(function);
    }
  }
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

    if (auto function = designatedFunction(symbol);
        function && !function->isStatic()) {
      auto functionType = type_cast<FunctionType>(named_symbol_type(function));
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
      valid = traits.is_arithmetic_or_unscoped_enum(expr->type) ||
              traits.is_pointer(expr->type);
      break;
    case TokenKind::T_MINUS:
      valid = traits.is_arithmetic_or_unscoped_enum(expr->type);
      break;
    case TokenKind::T_TILDE:
      valid = traits.is_integral_or_unscoped_enum(expr->type);
      break;
    default:
      return;
  }

  if (!valid) return;

  if (traits.is_integral_or_unscoped_enum(expr->type))
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
      (void)contextual_conversion_to_bool(ast->expression);
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

  if (ast->expression && require_complete_for_sizeof(ast->firstSourceLocation(),
                                                     ast->expression->type)) {
    ast->value = control()->memoryLayout()->sizeOf(ast->expression->type);
  }
}

void TypeChecker::Visitor::operator()(SizeofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();
  ast->valueCategory = ValueCategory::kPrValue;

  if (ast->typeId && require_complete_for_sizeof(ast->firstSourceLocation(),
                                                 ast->typeId->type)) {
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
    ast->value = !IsPotentiallyThrowing{check.unit_}.apply(ast->expression);
  }
}

void TypeChecker::Visitor::operator()(NewExpressionAST* ast) {
  if (ClassTemplateArgumentDeduction::placeholderClassTemplate(
          ast->typeSpecifierList ? ast->typeSpecifierList->value : nullptr,
          check.scope())) {
    std::vector<ExpressionAST*> arguments;
    bool isListInitialization = false;

    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      for (auto argument : ListView{paren->expressionList})
        arguments.push_back(argument);
    } else if (auto braced =
                   ast_cast<NewBracedInitializerAST>(ast->newInitalizer);
               braced && braced->bracedInitList) {
      isListInitialization = true;
      for (auto argument : ListView{braced->bracedInitList->expressionList})
        arguments.push_back(argument);
    }

    auto deduced = check.deduceClassTemplateSpecialization(
        ast->typeSpecifierList->value, arguments, isListInitialization,
        /*isCopyInitialization=*/false, ast->newLoc);

    if (!deduced) return;
    ast->objectType = deduced;
  }

  auto objectType = traits.remove_reference(ast->objectType);

  if (auto arrayType = type_cast<BoundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(arrayType->elementType());
  } else if (auto unboundedType =
                 type_cast<UnboundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(unboundedType->elementType());
  } else if (auto unresolvedArrayType =
                 type_cast<UnresolvedBoundedArrayType>(ast->objectType)) {
    ast->type = control()->getPointerType(unresolvedArrayType->elementType());
  } else {
    ast->type = control()->getPointerType(ast->objectType);
  }

  ast->valueCategory = ValueCategory::kPrValue;

  if (auto classType = type_cast<ClassType>(objectType)) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return;

    traits.requireCompleteClass(classSymbol);

    if (auto braced = ast_cast<NewBracedInitializerAST>(ast->newInitalizer);
        braced && braced->bracedInitList) {
      ExpressionAST* initializer = braced->bracedInitList;
      ast->constructorSymbol =
          check.check_class_initializer(objectType, initializer, ast->newLoc);
      return;
    }

    if (auto paren = ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
      ExpressionAST* initializer = nullptr;
      ast->constructorSymbol = check.check_class_initializer(
          objectType, initializer, ast->newLoc, &paren->expressionList);
      return;
    }

    if (ast->newInitalizer) return;

    List<ExpressionAST*>* defaultArguments = nullptr;
    ExpressionAST* initializer = nullptr;
    ast->constructorSymbol = check.check_class_initializer(
        objectType, initializer, ast->newLoc, &defaultArguments);

    if (defaultArguments) {
      ast->newInitalizer = NewParenInitializerAST::create(
          arena(), ast->newLoc, defaultArguments, ast->newLoc);
    }
  } else if (auto paren =
                 ast_cast<NewParenInitializerAST>(ast->newInitalizer)) {
    if (paren->expressionList && !paren->expressionList->next) {
      (void)implicit_conversion(paren->expressionList->value, objectType,
                                InitializationKind::kDirectInitialization);
    }
  } else if (auto braced =
                 ast_cast<NewBracedInitializerAST>(ast->newInitalizer);
             braced && braced->bracedInitList) {
    check.check_braced_init_list(objectType, braced->bracedInitList,
                                 InitializationKind::kDirectListInitialization);
  }
}

void TypeChecker::Visitor::operator()(DeleteExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;

  if (!ast->expression) return;

  auto operandType = traits.remove_cv(ast->expression->type);
  auto pointerType = type_cast<PointerType>(operandType);
  if (!pointerType) return;

  auto pointeeType = traits.remove_cv(pointerType->elementType());
  if (is_dependent_type(pointeeType)) return;

  ClassSymbol* classSymbol = nullptr;
  if (auto classType = type_cast<ClassType>(pointeeType))
    classSymbol = classType->symbol();

  const bool isArrayDelete = static_cast<bool>(ast->lbracketLoc);
  ast->symbol =
      resolveUsualOperatorDelete(check.unit_, classSymbol, isArrayDelete);
}

auto TypeChecker::Visitor::try_c_style_cast(CastExpressionAST* ast,
                                            ExpressionAST*& expr,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool {
  if (check_const_cast(expr, targetType, targetVC)) return true;
  if (check_static_cast(expr, targetType, targetVC)) return true;
  if (traits.is_pointer(targetType) && traits.is_pointer(expr->type))
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
    if ((traits.is_object(traits.remove_cv(sourceType)) &&
         traits.is_object(traits.remove_cv(targetType))) ||
        (traits.is_function(sourceType) && traits.is_function(targetType))) {
      return true;
    }
    return false;
  }

  (void)stdconv_.ensurePrvalue(expression);
  stdconv_.adjustCv(expression);
  sourceType = expression->type;

  if (traits.is_same(traits.remove_cv(sourceType),
                     traits.remove_cv(targetType)))
    return true;

  if (traits.is_pointer(sourceType) && traits.is_integral(targetType))
    return true;

  if ((traits.is_integral(sourceType) || traits.is_enum(sourceType) ||
       traits.is_scoped_enum(sourceType)) &&
      traits.is_pointer(targetType))
    return true;

  if (traits.is_pointer(sourceType) && traits.is_pointer(targetType)) {
    if (!traits.is_same(traits.remove_cv(sourceType),
                        traits.remove_cv(targetType)))
      emit_implicit_cast(expression, expression, targetType,
                         ImplicitCastKind::kPointerConversion);
    return true;
  }

  if (traits.is_member_pointer(sourceType) &&
      traits.is_member_pointer(targetType))
    return true;

  if (traits.is_null_pointer(sourceType) && traits.is_integral(targetType))
    return true;

  return false;
}

void TypeChecker::Visitor::operator()(ImplicitCastExpressionAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an expression");
    return;
  }

  if (ast->castKind == ImplicitCastKind::kLValueToRValueConversion) {
    ast->type = traits.remove_reference(ast->expression->type);
    stdconv_.adjustCv(ast);
    stdconv_.foldConstantRead(ast);
  } else if (ast->castKind == ImplicitCastKind::kUserDefinedConversion &&
             !ast->conversionFunction && ast->type &&
             !is_dependent_type(ast->type)) {
    if (auto paren = ast_cast<ParenInitializerAST>(ast->expression);
        paren && paren->expressionList) {
      ast->expression = paren->expressionList->value;
    }

    auto sequence =
        stdconv_.computeConversionSequence(ast->expression, ast->type);
    if (sequence.kind == ConversionSequenceKind::kUserDefined) {
      stdconv_.recordUserDefinedConversion(
          ast, sequence.userDefinedConversionFunction);
    } else {
      (void)stdconv_.recordClassCopyConstructor(ast);
    }
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
  if (traits.is_class_or_union(ast->leftExpression->type) ||
      traits.is_class_or_union(ast->rightExpression->type)) {
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

  if (!traits.is_integral_or_unscoped_enum(ast->leftExpression->type) ||
      !traits.is_integral_or_unscoped_enum(ast->rightExpression->type)) {
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

  if (traits.is_pointer(ast->leftExpression->type))
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  else if (traits.is_pointer(ast->rightExpression->type))
    (void)stdconv_.arrayToPointer(ast->leftExpression);

  if (stdconv_.usualArithmeticConversion(ast->leftExpression,
                                         ast->rightExpression)) {
    ast->type = control()->getBoolType();
    return;
  }

  if (traits.is_scoped_enum(ast->leftExpression->type)) {
    if (traits.is_same(traits.remove_cv(ast->leftExpression->type),
                       traits.remove_cv(ast->rightExpression->type))) {
      return;
    }
  }

  if (traits.is_pointer(ast->leftExpression->type) &&
      traits.is_pointer(ast->rightExpression->type)) {
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

auto TypeChecker::Visitor::rewrite_not_equal_as_negated_equal(
    BinaryExpressionAST* ast) -> bool {
  if (isC()) return false;
  if (ast->op != TokenKind::T_EXCLAIM_EQUAL) return false;

  auto boolType = control()->getBoolType();

  auto equalExpression = BinaryExpressionAST::create(check.unit_->arena());
  equalExpression->leftExpression = ast->leftExpression;
  equalExpression->opLoc = ast->opLoc;
  equalExpression->rightExpression = ast->rightExpression;
  equalExpression->op = TokenKind::T_EQUAL_EQUAL;
  equalExpression->type = boolType;
  equalExpression->valueCategory = ValueCategory::kPrValue;

  if (!resolve_binary_overload(equalExpression, false)) return false;
  if (!equalExpression->symbol) return false;

  ExpressionAST* comparison = equalExpression;
  if (!implicit_conversion(comparison, boolType)) return false;

  auto negated = BoolLiteralExpressionAST::create(check.unit_->arena());
  negated->literalLoc = ast->opLoc;
  negated->isTrue = false;
  negated->type = boolType;
  negated->valueCategory = ValueCategory::kPrValue;

  ast->leftExpression = comparison;
  ast->rightExpression = negated;
  ast->op = TokenKind::T_EQUAL_EQUAL;
  ast->symbol = nullptr;
  ast->isVirtualDispatch = false;
  ast->type = boolType;
  ast->valueCategory = ValueCategory::kPrValue;

  return true;
}

void TypeChecker::Visitor::check_equality(BinaryExpressionAST* ast) {
  ast->type = control()->getBoolType();

  if (resolve_binary_overload(ast, false)) return;

  if (rewrite_not_equal_as_negated_equal(ast)) return;

  prepare_comparison_operands(ast);

  if (isC()) {
    (void)stdconv_.arrayToPointer(ast->leftExpression);
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  }

  if (traits.is_pointer(ast->leftExpression->type) ||
      stdconv_.isNullPointerConstant(ast->leftExpression))
    (void)stdconv_.arrayToPointer(ast->rightExpression);
  else if (traits.is_pointer(ast->rightExpression->type) ||
           stdconv_.isNullPointerConstant(ast->rightExpression))
    (void)stdconv_.arrayToPointer(ast->leftExpression);

  if (stdconv_.usualArithmeticConversion(ast->leftExpression,
                                         ast->rightExpression)) {
    ast->type = control()->getBoolType();
    return;
  }

  {
    auto leftBase = traits.remove_cv(ast->leftExpression->type);
    auto rightBase = traits.remove_cv(ast->rightExpression->type);
    if (traits.is_scoped_enum(leftBase) && traits.is_same(leftBase, rightBase))
      return;
  }

  if ((traits.is_pointer(ast->leftExpression->type) ||
       stdconv_.isNullPointerConstant(ast->leftExpression)) &&
      (traits.is_pointer(ast->rightExpression->type) ||
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

  auto leftType = ast->leftExpression->type;
  auto rightType = ast->rightExpression->type;
  if (!leftType || !rightType) return;

  if (type_cast<AutoType>(traits.remove_cvref(leftType)) ||
      type_cast<AutoType>(traits.remove_cvref(rightType)))
    return;

  if (is_dependent_type(leftType) || is_dependent_type(rightType)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kPrValue;
    return;
  }

  switch (ast->op) {
    case TokenKind::T_DOT_STAR:
    case TokenKind::T_MINUS_GREATER_STAR:
      check_member_pointer_access(ast);
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
      if (resolve_binary_overload(ast)) break;
      check_three_way_comparison(ast);
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
      if (!contextual_conversion_to_bool(ast->leftExpression) ||
          !contextual_conversion_to_bool(ast->rightExpression)) {
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

  if (!check.check_bool_condition(ast->condition)) return;

  auto check_void_type = [&] {
    if (!traits.is_void(ast->iftrueExpression->type) &&
        !traits.is_void(ast->iffalseExpression->type))
      return false;

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

    if (!traits.is_same(ast->iftrueExpression->type,
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

    if (!traits.is_same(traits.remove_cv(ast->iftrueExpression->type),
                        traits.remove_cv(ast->iffalseExpression->type)))
      return false;

    ast->type = ast->iftrueExpression->type;

    ast->valueCategory = ast->iftrueExpression->valueCategory;

    return true;
  };

  auto check_arith_types = [&] {
    if (!traits.is_arithmetic_or_unscoped_enum(ast->iftrueExpression->type))
      return false;
    if (!traits.is_arithmetic_or_unscoped_enum(ast->iffalseExpression->type))
      return false;

    ast->type = stdconv_.usualArithmeticConversion(ast->iftrueExpression,
                                                   ast->iffalseExpression);

    if (!ast->type) return false;

    ast->valueCategory = ValueCategory::kPrValue;

    return true;
  };

  auto check_same_types = [&] {
    if (!traits.is_same(ast->iftrueExpression->type,
                        ast->iffalseExpression->type))
      return false;

    (void)stdconv_.ensurePrvalue(ast->iftrueExpression);
    (void)stdconv_.ensurePrvalue(ast->iffalseExpression);

    ast->type = ast->iftrueExpression->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto check_compatible_pointers = [&] {
    if (!traits.is_pointer(ast->iftrueExpression->type) &&
        !traits.is_pointer(ast->iffalseExpression->type))
      return false;

    (void)stdconv_.ensurePrvalue(ast->iftrueExpression);
    (void)stdconv_.ensurePrvalue(ast->iffalseExpression);

    ast->type = stdconv_.compositePointerType(ast->iftrueExpression,
                                              ast->iffalseExpression);

    ast->valueCategory = ValueCategory::kPrValue;

    if (!ast->type) return false;

    auto insert_pointer_cast = [&](ExpressionAST*& expr) {
      if (traits.is_same(expr->type, ast->type)) return;

      auto castKind = ImplicitCastKind::kPointerConversion;
      auto srcPtr = type_cast<PointerType>(traits.remove_cv(expr->type));
      auto tgtPtr = type_cast<PointerType>(traits.remove_cv(ast->type));
      if (srcPtr && tgtPtr) {
        auto srcElem = traits.remove_cv(srcPtr->elementType());
        auto tgtElem = traits.remove_cv(tgtPtr->elementType());
        if (!traits.is_same(srcElem, tgtElem) && traits.is_class(srcElem) &&
            traits.is_class(tgtElem) && traits.is_base_of(tgtElem, srcElem)) {
          castKind = ImplicitCastKind::kDerivedToBaseConversion;
        }
      }

      auto cast = ImplicitCastExpressionAST::create(arena());
      cast->castKind = castKind;
      cast->expression = expr;
      cast->type = ast->type;
      cast->valueCategory = ValueCategory::kPrValue;
      expr = cast;
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

    if (traits.is_class_or_union(traits.remove_reference(ast->type))) {
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
      traits.is_pointer(ast->targetExpression->type) &&
      traits.is_integral_or_unscoped_enum(ast->rightExpression->type)) {
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

  check.check_condition_declaration(ast);
}

void TypeChecker::Visitor::operator()(EqualInitializerAST* ast) {
  if (!ast->expression) {
    error(ast->firstSourceLocation(), "expected an initializer expression");
    return;
  }

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

auto TypeChecker::Visitor::implicit_conversion(
    ExpressionAST*& expr, const Type* destinationType,
    InitializationKind initializationKind) -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  if (in_template()) {
    if (is_dependent_type(expr->type) || is_dependent_type(destinationType))
      return true;
  }

  return stdconv_.convertImplicitly(expr, destinationType, initializationKind);
}

auto TypeChecker::Visitor::contextual_conversion_to_bool(ExpressionAST*& expr)
    -> bool {
  return implicit_conversion(expr, control()->getBoolType(),
                             InitializationKind::kDirectInitialization);
}

void TypeChecker::Visitor::report_unresolved_qualified_id(
    IdExpressionAST* ast) {
  const auto name = to_string(get_name(control(), ast->unqualifiedId));

  error(ast->unqualifiedId->firstSourceLocation(),
        std::format("no member named '{}' in {}", name,
                    describe_scope(ast->nestedNameSpecifier->symbol)));
}

void TypeChecker::Visitor::report_unresolved_id(IdExpressionAST* ast) {
  auto name = get_name(control(), ast->unqualifiedId);
  if (auto templateId = name_cast<TemplateId>(name)) name = templateId->name();

  const auto spelling = to_string(name);

  if (spelling.starts_with("__builtin_")) {
    error(ast->firstSourceLocation(),
          std::format("unknown builtin function '{}'", spelling));
  } else {
    error(ast->firstSourceLocation(),
          std::format("use of undeclared identifier '{}'", spelling));
  }
}

auto isUntypedAfterError(ExpressionAST* expr) -> bool {
  if (!expr) return false;
  if (expr->type) return false;
  return !ast_cast<BracedInitListAST>(expr);
}

void markUntypedAfterError(ExpressionAST* expr) {
  if (expr) expr->type = nullptr;
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

namespace {
[[nodiscard]] auto lookupBaseClassType(ClassSymbol* classSymbol,
                                       const Identifier* name,
                                       const TypeTraits& traits)
    -> const ClassType* {
  for (auto scope = static_cast<ScopeSymbol*>(classSymbol); scope;
       scope = scope->parent()) {
    auto found = qualifiedLookupType(scope, name);
    if (!found || !found->type()) continue;
    if (auto classType = type_cast<ClassType>(traits.remove_cv(found->type())))
      return classType;
  }
  return nullptr;
}

[[nodiscard]] auto isElementwiseArrayCopy(
    const std::vector<ExpressionAST**>& args, const Type* arrayType,
    const TypeTraits& traits) -> bool {
  if (args.size() != 1) return false;
  auto cast = ast_cast<ImplicitCastExpressionAST>(*args[0]);
  if (!cast) return false;
  if (cast->castKind != ImplicitCastKind::kLValueToRValueConversion)
    return false;
  return traits.remove_cv(cast->type) == traits.remove_cv(arrayType);
}
}  // namespace

void TypeChecker::bind_template_parameter_base_initializers(
    CompoundStatementFunctionBodyAST* ast) {
  auto functionSymbol = symbol_cast<FunctionSymbol>(scope_);
  if (!functionSymbol || !functionSymbol->isConstructor()) return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return;

  auto control = unit_->control();

  for (auto memInit : ListView{ast->memInitializerList}) {
    if (memInit->symbol) continue;

    UnqualifiedIdAST* unqualifiedId = nullptr;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit))
      unqualifiedId = paren->unqualifiedId;
    else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit))
      unqualifiedId = braced->unqualifiedId;

    auto name = get_name(control, ast_cast<NameIdAST>(unqualifiedId));
    if (!name) continue;

    for (auto base : classSymbol->baseClasses()) {
      if (base->name() != name) continue;
      if (!symbol_cast<TypeParameterSymbol>(base->symbol())) continue;
      memInit->symbol = base;
      break;
    }
  }
}

void TypeChecker::check_mem_initializers(
    CompoundStatementFunctionBodyAST* ast) {
  if (!unit_->config().checkTypes) return;

  auto functionSymbol = symbol_cast<FunctionSymbol>(scope_);
  if (!functionSymbol) return;

  if (!functionSymbol->isConstructor()) return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return;

  auto control = unit_->control();

  std::unordered_set<Symbol*> explicitlyInitialized;
  MemInitializerAST* delegatingInit = nullptr;

  auto collectArgs = [&](MemInitializerAST* memInit) {
    std::vector<ExpressionAST**> args;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit)) {
      for (auto it = paren->expressionList; it; it = it->next)
        args.push_back(&it->value);
    } else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit)) {
      if (braced->bracedInitList) {
        for (auto it = braced->bracedInitList->expressionList; it;
             it = it->next)
          args.push_back(&it->value);
      }
    }
    return args;
  };

  auto memInitializerClause =
      [&](MemInitializerAST* memInit) -> ExpressionAST* {
    if (auto braced = ast_cast<BracedMemInitializerAST>(memInit))
      return braced->bracedInitList;
    auto paren = ast_cast<ParenMemInitializerAST>(memInit);
    if (!paren) return nullptr;
    return ParenInitializerAST::create(unit_->arena(), paren->lparenLoc,
                                       paren->expressionList, paren->rparenLoc,
                                       ValueCategory::kPrValue, nullptr);
  };

  auto completeResolvedConstructorCall = [&](MemInitializerAST* memInit) {
    auto parameters = StandardConversion::parameters(memInit->constructor);
    auto args = collectArgs(memInit);

    for (size_t i = 0; i < args.size() && i < parameters.size(); ++i)
      (void)implicit_conversion(*args[i], parameters[i]->type());

    if (args.size() >= parameters.size()) return;

    List<ExpressionAST*>** tail = nullptr;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit))
      tail = &paren->expressionList;
    else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit);
             braced && braced->bracedInitList)
      tail = &braced->bracedInitList->expressionList;
    if (!tail) return;

    while (*tail) tail = &(*tail)->next;
    append_default_arguments(memInit->constructor, tail);
  };

  for (auto memInit : ListView{ast->memInitializerList}) {
    UnqualifiedIdAST* unqualifiedId = nullptr;
    if (auto paren = ast_cast<ParenMemInitializerAST>(memInit))
      unqualifiedId = paren->unqualifiedId;
    else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit))
      unqualifiedId = braced->unqualifiedId;

    if (!unqualifiedId) {
      if (memInit->symbol) explicitlyInitialized.insert(memInit->symbol);
      if (memInit->constructor) completeResolvedConstructorCall(memInit);
      continue;
    }

    auto name = get_name(control, unqualifiedId);

    if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId);
        templateId && templateId->identifier) {
      name = templateId->identifier;
    }

    if (!name) {
      if (memInit->symbol) explicitlyInitialized.insert(memInit->symbol);
      continue;
    }

    if (!memInit->symbol && name == classSymbol->name()) {
      auto args = collectArgs(memInit);

      std::vector<ExpressionAST*> argValues;
      argValues.reserve(args.size());
      for (auto arg : args) argValues.push_back(*arg);

      auto resolution =
          OverloadResolution(unit_).resolveConstructor(classSymbol, argValues);

      if (!resolution.best) {
        error(memInit->firstSourceLocation(),
              "no matching constructor for delegation");
        continue;
      }

      if (resolution.ambiguous) {
        error(memInit->firstSourceLocation(),
              "delegating constructor call is ambiguous");
        continue;
      }

      if (delegationReaches(resolution.best->symbol, functionSymbol)) {
        error(memInit->firstSourceLocation(),
              "constructor delegates to itself");
        continue;
      }

      functionSymbol->setDelegatingConstructor(resolution.best->symbol);

      memInit->symbol = classSymbol;
      memInit->constructor = resolution.best->symbol;

      for (size_t i = 0; i < args.size(); ++i)
        applyImplicitConversion(resolution.best->conversions[i], *args[i]);

      delegatingInit = memInit;
      continue;
    }

    Symbol* member = memInit->symbol;

    if (!member) {
      for (auto s : classSymbol->find(name)) {
        if (s->isField() || s->kind() == SymbolKind::kBaseClass) {
          member = s;
          break;
        }
      }
    }

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
      auto traits = unit_->typeTraits();
      auto denoted =
          lookupBaseClassType(classSymbol, name_cast<Identifier>(name), traits);

      for (auto base : classSymbol->baseClasses()) {
        if (!base->symbol()) continue;
        auto baseType = base->symbol()->type();
        if (!baseType) continue;
        auto classType = type_cast<ClassType>(traits.remove_cv(baseType));
        if (!classType || !classType->symbol()) continue;
        if (classType->symbol()->name() != name && classType != denoted)
          continue;
        member = base;
        break;
      }
    }

    if (!member) {
      if (auto layout = classSymbol->layout()) {
        for (auto vbase : layout->virtualBases()) {
          if (vbase->name() == name) {
            member = vbase;
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

    if (!explicitlyInitialized.insert(member).second) {
      error(memInit->firstSourceLocation(),
            std::format("multiple initializations of '{}'", to_string(name)));
      continue;
    }

    memInit->symbol = member;

    const Type* targetType = nullptr;
    if (auto field = symbol_cast<FieldSymbol>(member)) {
      targetType = field->type();
    } else if (auto base = symbol_cast<BaseClassSymbol>(member)) {
      targetType = base->symbol() ? base->symbol()->type() : nullptr;
    } else if (auto vbase = symbol_cast<ClassSymbol>(member)) {
      targetType = vbase->type();
    }

    if (!targetType) continue;

    auto args = collectArgs(memInit);

    if (unit_->typeTraits().is_class(targetType)) {
      List<ExpressionAST*>** memInitArguments = nullptr;
      if (auto paren = ast_cast<ParenMemInitializerAST>(memInit))
        memInitArguments = &paren->expressionList;
      else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit);
               braced && braced->bracedInitList)
        memInitArguments = &braced->bracedInitList->expressionList;

      if (memInit->constructor) {
        completeResolvedConstructorCall(memInit);
      } else {
        ExpressionAST* initializer = memInitializerClause(memInit);
        memInit->constructor = check_class_initializer(
            targetType, initializer, memInit->firstSourceLocation(),
            memInitArguments);
      }
    } else if (unit_->typeTraits().is_array(targetType)) {
      auto braced = ast_cast<BracedMemInitializerAST>(memInit);
      auto paren = ast_cast<ParenMemInitializerAST>(memInit);

      const auto valueInitialized = paren && !paren->expressionList;

      if (braced && braced->bracedInitList) {
        check_braced_init_list(targetType, braced->bracedInitList,
                               InitializationKind::kDirectListInitialization);
      } else if (!valueInitialized &&
                 !isElementwiseArrayCopy(args, targetType,
                                         unit_->typeTraits())) {
        error(memInit->firstSourceLocation(),
              "an array member must be initialized with a braced "
              "initializer list");
      }
    } else {
      if (args.size() == 1) {
        (void)implicit_conversion(*args[0], targetType);
      } else if (args.size() > 1) {
        error(memInit->firstSourceLocation(),
              "too many initializers for scalar member");
      }
    }
  }

  if (delegatingInit) {
    if (ast->memInitializerList->next) {
      error(delegatingInit->firstSourceLocation(),
            "an initializer for a delegating constructor must appear alone");
    }
    return;
  }

  auto pool = unit_->arena();
  auto traits = unit_->typeTraits();

  auto makeDefaultArgumentList =
      [&](FunctionSymbol* ctor) -> List<ExpressionAST*>* {
    List<ExpressionAST*>* list = nullptr;
    append_default_arguments(ctor, &list);
    return list;
  };

  auto resolveDefaultConstructor = [&](ClassSymbol* cls) -> FunctionSymbol* {
    if (!cls) return nullptr;
    cls = cls->resolvedDefinition();
    OverloadResolution overloadRes(unit_);
    auto resolution = overloadRes.resolveConstructor(cls, {});
    if (resolution.ambiguous) return nullptr;
    return resolution.best ? resolution.best->symbol : nullptr;
  };

  auto makeClassNsdmiInit =
      [&](FieldSymbol* field,
          ExpressionAST* initializer) -> ParenMemInitializerAST* {
    auto makeInit = [&](FunctionSymbol* ctor, List<ExpressionAST*>* args) {
      return ParenMemInitializerAST::create(
          pool, /*nestedNameSpecifier=*/nullptr, /*unqualifiedId=*/nullptr,
          args, /*symbol=*/field, /*constructor=*/ctor);
    };

    ExpressionAST* expr = initializer;
    if (auto equal = ast_cast<EqualInitializerAST>(expr))
      expr = equal->expression;

    if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
      auto ctor = field->constructor();
      if (!ctor) return nullptr;
      return makeInit(ctor, paren->expressionList);
    }

    auto fieldType = traits.remove_cv(field->type());

    if (auto braced = ast_cast<BracedInitListAST>(expr)) {
      auto ctor = field->constructor();
      const auto passListWhole =
          braced->type &&
          !traits.is_same(traits.remove_cv(braced->type), fieldType);
      if (ctor && !passListWhole) {
        List<ExpressionAST*>* argList = nullptr;
        auto tail = &argList;
        for (auto it = braced->expressionList; it; it = it->next) {
          *tail = make_list_node<ExpressionAST>(pool, it->value);
          tail = &(*tail)->next;
        }
        return makeInit(ctor, argList);
      }
      if (ctor) {
        return makeInit(ctor, make_list_node<ExpressionAST>(pool, braced));
      }
      if (!braced->type) braced->type = field->type();
      return makeInit(nullptr, make_list_node<ExpressionAST>(pool, braced));
    }

    while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
      if (cast->castKind !=
          ImplicitCastKind::kTemporaryMaterializationConversion)
        break;
      expr = cast->expression;
    }

    if (expr->valueCategory == ValueCategory::kPrValue && expr->type &&
        traits.is_same(traits.remove_cv(expr->type), fieldType)) {
      return makeInit(nullptr, make_list_node<ExpressionAST>(pool, expr));
    }

    if (auto ctor = field->constructor())
      return makeInit(ctor, make_list_node<ExpressionAST>(pool, expr));

    return nullptr;
  };

  auto makeNsdmiInit =
      [&](FieldSymbol* field,
          ExpressionAST* initializer) -> ParenMemInitializerAST* {
    if (type_cast<ClassType>(traits.remove_cv(field->type())))
      return makeClassNsdmiInit(field, initializer);

    if (auto braced = ast_cast<BracedInitListAST>(initializer);
        braced && !braced->type) {
      braced->type = field->type();
    }

    return ParenMemInitializerAST::create(
        pool, /*nestedNameSpecifier=*/nullptr, /*unqualifiedId=*/nullptr,
        make_list_node<ExpressionAST>(pool, initializer),
        /*symbol=*/field, /*constructor=*/nullptr);
  };

  auto makeAnonymousUnionInit =
      [&](FieldSymbol* field) -> ParenMemInitializerAST* {
    auto unionType = type_cast<ClassType>(traits.remove_cv(field->type()));
    if (!unionType || !unionType->symbol()) return nullptr;

    auto unionSymbol = unionType->symbol()->resolvedDefinition();
    if (!unionSymbol->isUnion() || unionSymbol->name()) return nullptr;

    for (auto member :
         cxx::views::members(unionSymbol) | cxx::views::non_static_fields) {
      if (auto initializer = member->initializer())
        return makeNsdmiInit(member, initializer);
    }

    return nullptr;
  };

  std::vector<MemInitializerAST*> written;
  for (auto memInit : ListView{ast->memInitializerList})
    if (memInit->symbol) written.push_back(memInit);

  const auto writtenOrder = written;

  List<MemInitializerAST*>* newList = nullptr;
  auto newTail = &newList;
  std::unordered_map<MemInitializerAST*, int> canonicalPos;
  int position = 0;

  auto append = [&](MemInitializerAST* node) {
    canonicalPos[node] = position++;
    *newTail = make_list_node<MemInitializerAST>(pool, node);
    newTail = &(*newTail)->next;
  };

  auto take = [&](Symbol* member) -> MemInitializerAST* {
    for (auto& node : written) {
      if (node && node->symbol == member) return std::exchange(node, nullptr);
    }
    return nullptr;
  };

  auto isVirtualBaseInit = [&](MemInitializerAST* node) {
    if (auto base = symbol_cast<BaseClassSymbol>(node->symbol))
      return base->isVirtual();
    if (auto cls = symbol_cast<ClassSymbol>(node->symbol))
      return cls != classSymbol && cls != classSymbol->definition();
    return false;
  };

  for (auto& node : written) {
    if (node && isVirtualBaseInit(node)) append(std::exchange(node, nullptr));
  }

  for (auto base : classSymbol->baseClasses()) {
    if (auto node = take(base)) {
      append(node);
      continue;
    }

    if (base->isVirtual()) continue;

    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    auto defaultCtor = resolveDefaultConstructor(baseClassSymbol);
    if (!defaultCtor) continue;

    append(ParenMemInitializerAST::create(
        pool, /*nestedNameSpecifier=*/nullptr, /*unqualifiedId=*/nullptr,
        makeDefaultArgumentList(defaultCtor), /*symbol=*/base,
        /*constructor=*/defaultCtor));
  }

  auto ownsAnonymousMember = [&](FieldSymbol* field, Symbol* member) {
    auto fieldClass = type_cast<ClassType>(traits.remove_cv(field->type()));
    if (!fieldClass) return false;
    for (auto parent = member->parent(); parent; parent = parent->parent()) {
      auto parentClass = symbol_cast<ClassSymbol>(parent);
      if (!parentClass) return false;
      if (parentClass == fieldClass->symbol()) return true;
      if (parentClass->name()) return false;
    }
    return false;
  };

  for (auto field :
       cxx::views::members(classSymbol) | cxx::views::non_static_fields) {
    if (auto node = take(field)) {
      append(node);
      continue;
    }

    if (!field->name()) {
      bool placedAnonymousMember = false;
      for (auto& node : written) {
        if (node && ownsAnonymousMember(field, node->symbol)) {
          append(std::exchange(node, nullptr));
          placedAnonymousMember = true;
        }
      }
      if (placedAnonymousMember) continue;

      if (auto anonymousInit = makeAnonymousUnionInit(field)) {
        append(anonymousInit);
        continue;
      }
    }

    if (classSymbol->isUnion()) continue;

    auto fieldType = traits.remove_cv(field->type());
    if (isDependent(unit_, fieldType)) continue;

    ParenMemInitializerAST* syntheticInit = nullptr;

    if (auto initializer = field->initializer()) {
      syntheticInit = makeNsdmiInit(field, initializer);
    } else if (auto classType = type_cast<ClassType>(fieldType)) {
      auto fieldClassSymbol = classType->symbol();
      if (!fieldClassSymbol || !fieldClassSymbol->name()) continue;

      auto defaultCtor = resolveDefaultConstructor(fieldClassSymbol);
      if (!defaultCtor) continue;

      syntheticInit = ParenMemInitializerAST::create(
          pool, /*nestedNameSpecifier=*/nullptr, /*unqualifiedId=*/nullptr,
          makeDefaultArgumentList(defaultCtor), /*symbol=*/field,
          /*constructor=*/defaultCtor);
    }

    if (!syntheticInit) continue;

    append(syntheticInit);
  }

  for (auto& node : written) {
    if (node) append(std::exchange(node, nullptr));
  }

  ast->memInitializerList = newList;

  int lastPosition = -1;
  for (auto node : writtenOrder) {
    auto it = canonicalPos.find(node);
    if (it == canonicalPos.end()) continue;
    if (it->second < lastPosition) {
      warning(node->firstSourceLocation(),
              "initializer order does not match the declaration order");
      break;
    }
    lastPosition = it->second;
  }

  auto variant = functionSymbol->completeObjectVariant();
  if (!variant && functionSymbol->canonical())
    variant = functionSymbol->canonical()->completeObjectVariant();
  if (!variant || !variant->declaration()) return;

  auto variantBody = ast_cast<CompoundStatementFunctionBodyAST>(
      variant->declaration()->functionBody);
  if (!variantBody) return;

  for (auto memInit : ListView{ast->memInitializerList}) {
    ClassSymbol* vbase = nullptr;
    if (auto base = symbol_cast<BaseClassSymbol>(memInit->symbol);
        base && base->isVirtual()) {
      vbase = symbol_cast<ClassSymbol>(base->symbol());
    } else if (auto cls = symbol_cast<ClassSymbol>(memInit->symbol);
               cls && cls != classSymbol && cls != classSymbol->definition()) {
      vbase = cls;
    }
    if (!vbase) continue;
    vbase = vbase->resolvedDefinition();

    for (auto node : ListView{variantBody->memInitializerList}) {
      auto placeholder = ast_cast<ParenMemInitializerAST>(node);
      if (!placeholder || placeholder->symbol != vbase) continue;

      placeholder->constructor = memInit->constructor;
      if (auto paren = ast_cast<ParenMemInitializerAST>(memInit)) {
        placeholder->expressionList = paren->expressionList;
      } else if (auto braced = ast_cast<BracedMemInitializerAST>(memInit);
                 braced && braced->bracedInitList) {
        placeholder->expressionList = braced->bracedInitList->expressionList;
      }
      break;
    }
  }
}

void TypeChecker::Visitor::check_static_assert(
    StaticAssertDeclarationAST* ast) {
  auto loc = ast->firstSourceLocation();

  if (in_template()) return;
  if (isUntypedAfterError(ast->expression)) return;

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

auto TypeChecker::Visitor::convert_class_operands_for_builtin(
    BinaryExpressionAST* ast) -> bool {
  const auto leftIsClass = traits.is_class_or_union(ast->leftExpression->type);
  const auto rightIsClass =
      traits.is_class_or_union(ast->rightExpression->type);

  if (!leftIsClass && !rightIsClass) return true;

  if (leftIsClass &&
      !stdconv_.convertClassOperandForBuiltinOperator(ast->leftExpression)) {
    error(
        ast->opLoc,
        std::format("'operator {}' is not defined for types {} and {}",
                    Token::spell(ast->op), to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
    return false;
  }

  if (rightIsClass &&
      !stdconv_.convertClassOperandForBuiltinOperator(ast->rightExpression)) {
    error(
        ast->opLoc,
        std::format("'operator {}' is not defined for types {} and {}",
                    Token::spell(ast->op), to_string(ast->leftExpression->type),
                    to_string(ast->rightExpression->type)));
    return false;
  }

  return true;
}

void TypeChecker::Visitor::check_addition(BinaryExpressionAST* ast) {
  if (!convert_class_operands_for_builtin(ast)) return;

  if (auto ty = stdconv_.usualArithmeticConversion(ast->leftExpression,
                                                   ast->rightExpression)) {
    ast->type = ty;
    return;
  }

  (void)stdconv_.ensurePrvalue(ast->leftExpression);
  stdconv_.adjustCv(ast->leftExpression);

  (void)stdconv_.ensurePrvalue(ast->rightExpression);
  stdconv_.adjustCv(ast->rightExpression);

  const auto left_is_pointer = traits.is_pointer(ast->leftExpression->type);

  const auto right_is_pointer = traits.is_pointer(ast->rightExpression->type);

  const auto left_is_integral =
      traits.is_integral_or_unscoped_enum(ast->leftExpression->type);

  const auto right_is_integral =
      traits.is_integral_or_unscoped_enum(ast->rightExpression->type);

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
  if (!convert_class_operands_for_builtin(ast)) return;

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
    if (!traits.is_pointer(ast->leftExpression->type)) return false;

    if (!traits.is_arithmetic_or_unscoped_enum(ast->rightExpression->type) &&
        !traits.is_pointer(ast->rightExpression->type))
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

  if (traits.is_pointer(ast->rightExpression->type)) {
    auto leftElementType = traits.get_element_type(ast->leftExpression->type);
    (void)strip_cv(leftElementType);

    auto rightElementType = traits.get_element_type(ast->rightExpression->type);
    (void)strip_cv(rightElementType);

    if (traits.is_same(leftElementType, rightElementType)) {
      ast->type = control()->getLongIntType();
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
  if (in_template() && is_dependent_type(ast->expression->type)) {
    ast->type = dependent_type();
    ast->valueCategory = ValueCategory::kLValue;
    return;
  }

  if (!is_glvalue(ast->expression)) {
    error(ast->opLoc, std::format("cannot {} an rvalue of type '{}'", action,
                                  to_string(ast->expression->type)));
    return;
  }

  if (!traits.is_const(ast->expression->type)) {
    auto ty = ast->expression->type;

    if (isCxx() ? traits.is_arithmetic(ty)
                : traits.is_arithmetic_or_unscoped_enum(ty)) {
      ast->type = ty;
      ast->valueCategory = ValueCategory::kLValue;
      return;
    }

    if (auto ptrTy = as_pointer(ty)) {
      if (!traits.is_void(ptrTy->elementType())) {
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
    const Type* rightType, FunctionSymbol*& symbolOut, ExpressionAST* leftExpr,
    ExpressionAST* rightExpr) -> bool {
  symbolOut = nullptr;

  if (auto symbol =
          check.lookupOperator(leftType, op, rightType, leftExpr, rightExpr)) {
    symbolOut = symbol;
    check.reportDeletedFunction(symbol, opLoc);
    check.requireFunctionDefinition(symbol);
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
  adjust_member_operator_object_argument(operatorFunc, ast->expression);
  ast->isVirtualDispatch =
      is_virtual_member_operator_dispatch(operatorFunc, ast->expression);
  setResultTypeAndValueCategory(ast, operatorFunc);
  return true;
}

void TypeChecker::Visitor::apply_operator_argument_conversions(
    FunctionSymbol* operatorFunc, ExpressionAST*& leftExpression,
    ExpressionAST*& rightExpression) {
  auto functionType = type_cast<FunctionType>(operatorFunc->type());
  if (!functionType) return;

  const bool isMember = operatorFunc->isImplicitObjectMemberFunction();
  if (isMember) {
    adjust_member_operator_object_argument(operatorFunc, leftExpression);
    auto arguments =
        make_list_node<ExpressionAST>(check.unit_->arena(), rightExpression);
    check_function_arguments(arguments, rightExpression->firstSourceLocation(),
                             functionType);
    rightExpression = arguments->value;
    return;
  }

  auto arguments =
      make_list_node<ExpressionAST>(check.unit_->arena(), leftExpression);
  arguments->next =
      make_list_node<ExpressionAST>(check.unit_->arena(), rightExpression);
  check_function_arguments(arguments, leftExpression->firstSourceLocation(),
                           functionType);
  leftExpression = arguments->value;
  rightExpression = arguments->next->value;
}

void TypeChecker::Visitor::adjust_member_operator_object_argument(
    FunctionSymbol* operatorFunc, ExpressionAST*& objectExpression) {
  if (!operatorFunc || !objectExpression || !objectExpression->type) return;
  if (!operatorFunc->isImplicitObjectMemberFunction()) return;
  if (!is_glvalue(objectExpression)) return;

  auto classSymbol = symbol_cast<ClassSymbol>(operatorFunc->parent());
  if (!classSymbol) return;

  auto sourceType =
      traits.remove_cv(traits.remove_reference(objectExpression->type));
  if (traits.is_same(sourceType, classSymbol->type())) return;
  if (!traits.is_base_of(classSymbol->type(), sourceType)) return;

  check.wrapWithImplicitCast(ImplicitCastKind::kDerivedToBaseConversion,
                             classSymbol->type(), objectExpression);
}

auto TypeChecker::Visitor::is_known_complete_object(ExpressionAST* expression)
    -> bool {
  return stdconv_.isKnownCompleteObject(expression);
}

auto TypeChecker::Visitor::is_virtual_member_operator_dispatch(
    FunctionSymbol* operatorFunc, ExpressionAST* objectExpression) -> bool {
  return stdconv_.isVirtualMemberDispatch(operatorFunc, objectExpression);
}

auto TypeChecker::Visitor::resolve_binary_overload(BinaryExpressionAST* ast,
                                                   bool setValueCategory)
    -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->leftExpression->type, ast->op, ast->opLoc,
                                 ast->rightExpression->type, operatorFunc,
                                 ast->leftExpression, ast->rightExpression)) {
    return false;
  }

  if (!operatorFunc) return true;

  if (check.wasLastOperatorRewritten()) {
    auto operatorId = name_cast<OperatorId>(operatorFunc->name());
    const auto rewrittenOp =
        operatorId ? operatorId->op() : TokenKind::T_LESS_EQUAL_GREATER;
    const bool reversed = check.wasLastOperatorReversed();

    auto comparison = BinaryExpressionAST::create(check.unit_->arena());
    comparison->leftExpression = ast->leftExpression;
    comparison->opLoc = ast->opLoc;
    comparison->rightExpression = ast->rightExpression;
    comparison->op = rewrittenOp;
    if (reversed)
      std::swap(comparison->leftExpression, comparison->rightExpression);
    comparison->symbol = operatorFunc;
    apply_operator_argument_conversions(
        operatorFunc, comparison->leftExpression, comparison->rightExpression);
    comparison->isVirtualDispatch = is_virtual_member_operator_dispatch(
        operatorFunc, comparison->leftExpression);
    setResultTypeAndValueCategory(comparison, operatorFunc);

    if (rewrittenOp == TokenKind::T_EQUAL_EQUAL) {
      if (ast->op == TokenKind::T_EQUAL_EQUAL) {
        ast->leftExpression = comparison->leftExpression;
        ast->rightExpression = comparison->rightExpression;
        ast->symbol = operatorFunc;
        ast->isVirtualDispatch = comparison->isVirtualDispatch;
        setResultTypeAndValueCategory(ast, named_symbol_type(operatorFunc));
        return true;
      }

      auto falseLiteral = BoolLiteralExpressionAST::create(
          check.unit_->arena(), ast->opLoc, false, ValueCategory::kPrValue,
          control()->getBoolType());

      ast->leftExpression = comparison;
      ast->rightExpression = falseLiteral;
      ast->op = TokenKind::T_EQUAL_EQUAL;
      ast->symbol = nullptr;
      ast->isVirtualDispatch = false;
      check_equality(ast);
      return true;
    }

    auto zero = IntLiteralExpressionAST::create(
        check.unit_->arena(), control()->integerLiteral("0"),
        ValueCategory::kPrValue, control()->getIntType());
    if (reversed) {
      ast->leftExpression = zero;
      ast->rightExpression = comparison;
    } else {
      ast->leftExpression = comparison;
      ast->rightExpression = zero;
    }
    ast->symbol = nullptr;
    ast->isVirtualDispatch = false;
    check_relational(ast);
    return true;
  }

  ast->symbol = operatorFunc;

  apply_operator_argument_conversions(operatorFunc, ast->leftExpression,
                                      ast->rightExpression);
  ast->isVirtualDispatch =
      is_virtual_member_operator_dispatch(operatorFunc, ast->leftExpression);

  auto operatorType = named_symbol_type(operatorFunc);

  if (setValueCategory) {
    setResultTypeAndValueCategory(ast, operatorType);
  } else if (auto functionType = type_cast<FunctionType>(operatorType)) {
    ast->type = functionType->returnType();
  } else {
    ast->type = operatorType;
  }

  return true;
}

auto TypeChecker::Visitor::resolve_assignment_overload(
    AssignmentExpressionAST* ast) -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->leftExpression->type, ast->op, ast->opLoc,
                                 ast->rightExpression->type, operatorFunc,
                                 ast->leftExpression, ast->rightExpression)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  apply_operator_argument_conversions(operatorFunc, ast->leftExpression,
                                      ast->rightExpression);
  ast->isVirtualDispatch =
      is_virtual_member_operator_dispatch(operatorFunc, ast->leftExpression);
  setResultTypeAndValueCategory(ast, operatorFunc);
  return true;
}

auto TypeChecker::Visitor::resolve_compound_assignment_overload(
    CompoundAssignmentExpressionAST* ast) -> bool {
  FunctionSymbol* operatorFunc = nullptr;
  if (!resolve_operator_overload(ast->targetExpression->type, ast->op,
                                 ast->opLoc, ast->rightExpression->type,
                                 operatorFunc, ast->targetExpression,
                                 ast->rightExpression)) {
    return false;
  }

  if (!operatorFunc) return true;

  ast->symbol = operatorFunc;
  apply_operator_argument_conversions(operatorFunc, ast->targetExpression,
                                      ast->rightExpression);
  ast->isVirtualDispatch =
      is_virtual_member_operator_dispatch(operatorFunc, ast->targetExpression);
  setResultTypeAndValueCategory(ast, operatorFunc);
  return true;
}

auto TypeChecker::Visitor::check_member_access(MemberExpressionAST* ast)
    -> bool {
  const Type* objectType = ast->baseExpression->type;
  auto cv1 = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    if (traits.is_class_or_union(ast->baseExpression->type)) {
      std::vector<const Type*> visited;
      while (traits.is_class_or_union(ast->baseExpression->type)) {
        auto current = traits.remove_cv(ast->baseExpression->type);
        if (std::ranges::find(visited, current) != visited.end()) return false;
        visited.push_back(current);
        if (!resolve_arrow_operator(ast)) return false;
      }
    } else {
      (void)stdconv_.ensurePrvalue(ast->baseExpression);
    }

    objectType = ast->baseExpression->type;
    cv1 = strip_cv(objectType);

    auto pointerType = as_pointer(objectType);
    if (!pointerType) return false;

    objectType = pointerType->elementType();
    cv1 = strip_cv(objectType);
  }

  auto classType = as_class(objectType);
  if (!classType) return false;

  auto classSymbol = classType->symbol();

  traits.requireCompleteClass(classSymbol);

  if (auto dtor = ast_cast<DestructorIdAST>(ast->unqualifiedId)) {
    auto typeSymbol = resolveDestructorIdType(ast, dtor);
    Symbol* symbol = nullptr;
    if (typeSymbol && traits.is_same(typeSymbol->type(), objectType)) {
      symbol = classSymbol->destructor();
    }

    ast->symbol = symbol;

    if (!symbol) {
      error(dtor->firstSourceLocation(),
            "the type of object expression does not match the type "
            "being destroyed");
      return true;
    }

    ast->type = symbol->type();
    ast->valueCategory = (is_lvalue(ast->baseExpression) ||
                          ast->accessOp == TokenKind::T_MINUS_GREATER)
                             ? ValueCategory::kLValue
                             : ValueCategory::kXValue;
    return true;
  }

  auto memberName = get_name(control(), ast->unqualifiedId);

  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
  if (templateId && templateId->identifier) memberName = templateId->identifier;

  Symbol* lookupScope = classSymbol;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol) {
    lookupScope = ast->nestedNameSpecifier->symbol;

    if (lookupScope != classSymbol && !in_template() &&
        !traits.is_base_of(lookupScope->type(), objectType)) {
      error(ast->firstSourceLocation(),
            std::format("'{}::{}' is not a member of a base class of '{}'",
                        to_string(lookupScope->name()), to_string(memberName),
                        to_string(classSymbol->name())));
      return true;
    }
  }

  auto symbol = qualifiedLookup(lookupScope, memberName);

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
                      to_string(lookupScope->name())));
    return true;
  }

  if (symbol) {
    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
      ast->type = designated_function_type(overloadSet);
      ast->valueCategory = ValueCategory::kLValue;
      return true;
    }

    auto symbolType = named_symbol_type(symbol);
    ast->type = symbolType;

    if (symbol->isEnumerator()) {
      ast->valueCategory = ValueCategory::kPrValue;
    } else if (traits.is_reference(symbolType)) {
      ast->type = traits.remove_reference(symbolType);
      ast->valueCategory = ValueCategory::kLValue;
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
          ast->type = traits.add_volatile(ast->type);

        if (!field->isMutable() && (is_const(cv1) || is_const(cv2)))
          ast->type = traits.add_const(ast->type);
      }
    }
  }

  return true;
}

auto TypeChecker::Visitor::resolveDestructorIdType(MemberExpressionAST* ast,
                                                   DestructorIdAST* dtor)
    -> Symbol* {
  auto name = ast_cast<NameIdAST>(dtor->id);
  if (!name) return nullptr;

  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol) {
    return qualifiedLookupType(ast->nestedNameSpecifier->symbol,
                               name->identifier);
  }

  for (auto s = check.scope_; s; s = s->parent()) {
    for (auto found : s->find(name->identifier)) {
      if (found->isHidden()) continue;
      if (is_type(found)) return found;
    }
  }

  return nullptr;
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

  if (!traits.is_scalar(objectType)) {
    return false;
  }

  auto dtor = ast_cast<DestructorIdAST>(ast->unqualifiedId);
  if (!dtor) return false;

  auto symbol = resolveDestructorIdType(ast, dtor);
  if (!symbol) return true;

  if (!traits.is_same(symbol->type(), objectType)) {
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
      break;
    }
  }

  if (!targetType) return;

  if (auto braced = ast_cast<BracedInitListAST>(ast->expression)) {
    if (!type_cast<AutoType>(targetType) && !isDependent(unit_, targetType)) {
      check_braced_init_list(targetType, braced,
                             InitializationKind::kCopyListInitialization);
    }
    return;
  }

  if (containsPlaceholderType(targetType) && ast->expression &&
      ast->expression->type && !isDependent(unit_, ast->expression->type)) {
    auto deducedType = deducePlaceholderType(targetType, ast->expression);
    auto funcType = type_cast<FunctionType>(functionScope->type());
    if (deducedType && funcType) {
      auto newFuncType = unit_->control()->getFunctionType(
          deducedType,
          std::vector<const Type*>(funcType->parameterTypes().begin(),
                                   funcType->parameterTypes().end()),
          funcType->isVariadic(), funcType->cvQualifiers(),
          funcType->refQualifier(), funcType->isNoexcept());
      functionScope->setType(newFuncType);
      targetType = deducedType;
    }
  }

  if (isDependent(unit_, targetType)) return;
  if (ast->expression && isDependent(unit_, ast->expression)) return;
  if (ast->expression && ast->expression->type &&
      isDependent(unit_, ast->expression->type))
    return;

  treatMoveEligibleOperandAsRvalue(ast->expression, functionScope, targetType);

  auto seq = checkImplicitConversion(ast->expression, targetType);
  applyImplicitConversion(seq, ast->expression);
  reportDeletedConversion(ast->expression);
}

auto TypeChecker::isMoveEligibleOperand(ExpressionAST* expr,
                                        ScopeSymbol* functionScope) const
    -> bool {
  if (!expr || !functionScope) return false;
  if (expr->valueCategory != ValueCategory::kLValue) return false;
  if (!unit_->typeTraits().is_class(expr->type)) return false;

  auto idExpression = ast_cast<IdExpressionAST>(expr);
  if (!idExpression || idExpression->nestedNameSpecifier) return false;

  auto symbol = idExpression->symbol;
  if (auto var = symbol_cast<VariableSymbol>(symbol)) {
    if (var->isStatic() || var->isExtern() || var->isThreadLocal())
      return false;
  } else if (!symbol_cast<ParameterSymbol>(symbol)) {
    return false;
  }

  if (unit_->typeTraits().is_reference(symbol->type())) return false;
  if (unit_->typeTraits().is_volatile(symbol->type())) return false;

  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (scope == functionScope) return true;
    if (scope->isFunction() || scope->isLambda()) return false;
    if (scope->isClass() || scope->isNamespace()) return false;
  }

  return false;
}

void TypeChecker::treatMoveEligibleOperandAsRvalue(ExpressionAST*& expr,
                                                   ScopeSymbol* functionScope,
                                                   const Type* targetType) {
  if (!isMoveEligibleOperand(expr, functionScope)) return;

  auto cast = ImplicitCastExpressionAST::create(unit_->arena());
  cast->castKind = ImplicitCastKind::kIdentity;
  cast->expression = expr;
  cast->type = expr->type;
  cast->valueCategory = ValueCategory::kXValue;

  Visitor visitor{*this};
  auto constructor = visitor.stdconv_.selectCopyConstructor(cast, targetType);
  if (!constructor) return;

  auto params = StandardConversion::parameters(constructor);
  if (params.empty()) return;
  if (!type_cast<RvalueReferenceType>(params[0]->type())) return;

  expr = cast;
}

auto TypeChecker::implicit_conversion(ExpressionAST*& yyast,
                                      const Type* targetType,
                                      InitializationKind initializationKind)
    -> bool {
  Visitor visitor{*this};
  return visitor.implicit_conversion(yyast, targetType, initializationKind);
}

auto TypeChecker::check_bool_condition(ExpressionAST*& expr) -> bool {
  if (!expr || !expr->type) return false;
  if (isDependent(unit_, expr->type)) return true;

  auto conditionType = expr->type;

  Visitor visitor{*this};
  if (visitor.contextual_conversion_to_bool(expr)) return true;

  error(expr->firstSourceLocation(),
        std::format("invalid condition expression of type '{}'",
                    to_string(conditionType)));
  return false;
}

void TypeChecker::check_integral_condition(ExpressionAST*& expr) {
  if (!expr || !expr->type) return;
  if (isDependent(unit_, expr->type)) return;

  auto traits = unit_->typeTraits();
  Visitor visitor{*this};

  auto conditionType = expr->type;

  if (traits.is_class(traits.remove_cv(conditionType)))
    (void)visitor.stdconv_.convertClassOperandForBuiltinOperator(expr);

  if (!traits.is_integral(expr->type) && !traits.is_enum(expr->type)) {
    error(expr->firstSourceLocation(),
          std::format("condition of type '{}' is not of integral or "
                      "enumeration type",
                      to_string(conditionType)));
    return;
  }

  (void)visitor.stdconv_.lvalueToRvalue(expr);
  visitor.stdconv_.adjustCv(expr);
  (void)visitor.stdconv_.integralPromotion(expr);
}

void TypeChecker::reportDeletedConversion(ExpressionAST* expr) {
  auto cast = ast_cast<ImplicitCastExpressionAST>(expr);
  if (!cast) return;
  if (cast->castKind != ImplicitCastKind::kUserDefinedConversion) return;
  reportDeletedFunction(cast->conversionFunction, expr->firstSourceLocation());
}

void TypeChecker::reportDeletedFunction(FunctionSymbol* function,
                                        SourceLocation loc) {
  if (!function || !function->isDeleted()) return;

  if (function->isDefaulted() && function->isConstructor()) {
    auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
    auto functionType = type_cast<FunctionType>(function->type());
    const bool isDefaultConstructor =
        functionType && functionType->parameterTypes().empty();

    if (classSymbol && isDefaultConstructor) {
      error(loc,
            std::format("call to implicitly-deleted default constructor of "
                        "'{}'",
                        to_string(classSymbol->name())));
      return;
    }
  }

  error(loc, std::format("use of deleted function '{}'",
                         to_string(function->name())));
}

void TypeChecker::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->error(loc, std::move(message));
}

void TypeChecker::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->warning(loc, std::move(message));
}

void TypeChecker::note(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->note(loc, std::move(message));
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

void TypeChecker::append_default_arguments(FunctionSymbol* function,
                                           List<ExpressionAST*>** list) {
  StandardConversion stdconv(unit_, unit_->language() == LanguageKind::kC);
  stdconv.appendDefaultArguments(function, list);
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
                                 const Type* rightType, ExpressionAST* leftExpr,
                                 ExpressionAST* rightExpr) -> FunctionSymbol* {
  OverloadResolution resolution(unit_);
  auto result =
      resolution.lookupOperator(type, op, rightType, leftExpr, rightExpr);
  lastOperatorLookupAmbiguous_ = resolution.wasLastLookupAmbiguous();
  lastOperatorRewritten_ = resolution.wasLastOperatorRewritten();
  lastOperatorReversed_ = resolution.wasLastOperatorReversed();
  return result;
}

void TypeChecker::requireFunctionDefinition(FunctionSymbol* function) {
  if (!potentiallyEvaluated_) return;
  ASTRewriter::requireFunctionDefinition(unit_, function);
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

#include "private/builtins_typechecker-priv.h"
