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
#include <cxx/implicit_conversion_sequence.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>
#include <set>
#include <unordered_set>

namespace cxx {

struct OverloadCandidate {
  FunctionSymbol* symbol = nullptr;
  std::vector<ImplicitConversionSequence> conversions;
  bool viable = false;
  bool exactCvMatch = true;
};

struct OverloadResult {
  OverloadCandidate* best = nullptr;
  bool ambiguous = false;
};

inline auto selectBestCandidate(std::vector<OverloadCandidate>& candidates,
                                bool useCvTiebreaker = false)
    -> OverloadResult {
  if (candidates.empty()) return {};

  std::vector<OverloadCandidate*> best;
  best.push_back(&candidates[0]);

  for (size_t i = 1; i < candidates.size(); ++i) {
    auto& curr = candidates[i];
    auto& ref = *best[0];

    bool currBetter = false;
    bool refBetter = false;

    auto n = std::min(curr.conversions.size(), ref.conversions.size());
    for (size_t j = 0; j < n; ++j) {
      if (curr.conversions[j].isBetterThan(ref.conversions[j]))
        currBetter = true;
      if (ref.conversions[j].isBetterThan(curr.conversions[j]))
        refBetter = true;
    }

    if (currBetter && !refBetter) {
      best.clear();
      best.push_back(&curr);
    } else if (refBetter && !currBetter) {
      // ref remains best
    } else if (useCvTiebreaker && curr.exactCvMatch != ref.exactCvMatch) {
      if (curr.exactCvMatch) {
        best.clear();
        best.push_back(&curr);
      }
      // else ref remains best
    } else {
      best.push_back(&curr);
    }
  }

  if (best.empty()) return {};
  if (best.size() > 1) return {best[0], /*ambiguous=*/true};
  return {best[0], /*ambiguous=*/false};
}

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

  [[nodiscard]] auto is_parsing_c() const {
    return check.unit_->language() == LanguageKind::kC;
  }

  [[nodiscard]] auto is_parsing_cxx() const {
    return check.unit_->language() == LanguageKind::kCXX;
  }

  void error(SourceLocation loc, std::string message) {
    check.error(loc, std::move(message));
  }

  void warning(SourceLocation loc, std::string message) {
    check.warning(loc, std::move(message));
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
  [[nodiscard]] auto integral_promotion(ExpressionAST*& expr,
                                        const Type* destinationType = nullptr)
      -> bool;
  [[nodiscard]] auto floating_point_promotion(
      ExpressionAST*& expr, const Type* destinationType = nullptr) -> bool;
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

  [[nodiscard]] auto user_defined_conversion(ExpressionAST*& expr,
                                             const Type* destinationType)
      -> bool;

  [[nodiscard]] auto ensure_prvalue(ExpressionAST*& expr) -> bool;

  void adjust_cv(ExpressionAST* expr);
  void setResultTypeAndValueCategory(ExpressionAST* ast, const Type* type);

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

  [[nodiscard]] auto as_pointer(const Type* type) const -> const PointerType* {
    return check.as_pointer(type);
  }

  [[nodiscard]] auto as_array(const Type* type) const -> const Type* {
    if (control()->is_array(type)) return control()->remove_cv(type);
    return nullptr;
  }

  [[nodiscard]] auto as_class(const Type* type) const -> const ClassType* {
    return check.as_class(type);
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

  void check_addition(BinaryExpressionAST* ast);
  void check_subtraction(BinaryExpressionAST* ast);

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
      if (!self.control()->is_same(selectorType, assoc->typeId->type)) {
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

      selectorType = control()->decay(ast->expression->type);

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
  } else {
    // maybe unresolved name
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
  if (!ast->baseExpression || !ast->indexExpression) return;

  if (auto operatorFunc =
          check.lookupOperator(ast->baseExpression->type, TokenKind::T_LBRACKET,
                               ast->indexExpression->type)) {
    ast->symbol = operatorFunc;
    setResultTypeAndValueCategory(ast, operatorFunc->type());
    return;
  }

  if (control()->is_class(ast->baseExpression->type)) return;
  if (control()->is_class(ast->indexExpression->type)) return;

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

  // overload resolution
  if (auto ovl = type_cast<OverloadSetType>(ast->baseExpression->type)) {
    std::vector<OverloadCandidate> candidates;

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

    // Collect all candidate functions from the overload set
    std::vector<FunctionSymbol*> allFunctions;
    for (auto func : ovl->symbol()->functions()) {
      if (func->canonical() != func) continue;
      if (func->isSpecialization()) continue;
      allFunctions.push_back(func);
    }

    // adl
    if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
      if (!idExpr->nestedNameSpecifier) {
        auto adlCandidates = Lookup{check.scope_}.argumentDependentLookup(
            ovl->symbol()->name(), argumentTypes);

        std::set<FunctionSymbol*> seen;
        for (auto f : allFunctions) seen.insert(f);
        for (auto f : adlCandidates) {
          if (seen.insert(f).second) {
            allFunctions.push_back(f);
          }
        }
      }
    }

    for (auto func : allFunctions) {
      auto type = type_cast<FunctionType>(func->type());
      if (!type) continue;

      if (func->templateDeclaration()) {
        auto templateDecl = func->templateDeclaration();

        std::vector<const TypeParameterType*> templateParamTypes;
        for (auto p : ListView{templateDecl->templateParameterList}) {
          if (auto sym = p->symbol) {
            templateParamTypes.push_back(
                type_cast<TypeParameterType>(sym->type()));
          } else {
            templateParamTypes.push_back(nullptr);
          }
        }

        std::vector<const Type*> deducedTypes(templateParamTypes.size(),
                                              nullptr);

        bool deductionFailed = false;
        auto paramIt2 = type->parameterTypes().begin();
        for (auto argIt2 = ast->expressionList;
             argIt2 && paramIt2 != type->parameterTypes().end();
             argIt2 = argIt2->next, ++paramIt2) {
          auto paramType = *paramIt2;
          auto argType = argIt2->value ? argIt2->value->type : nullptr;
          if (!argType) {
            deductionFailed = true;
            break;
          }

          auto bareParam = control()->remove_cvref(paramType);

          if (auto tpt = type_cast<TypeParameterType>(bareParam)) {
            auto idx = tpt->index();
            if (idx >= 0 && idx < static_cast<int>(deducedTypes.size())) {
              auto bareArg = control()->remove_cvref(argType);
              if (!deducedTypes[idx]) {
                deducedTypes[idx] = bareArg;
              } else if (deducedTypes[idx] != bareArg) {
                deductionFailed = true;
                break;
              }
            }
          }
        }

        if (!deductionFailed) {
          for (auto& dt : deducedTypes) {
            if (!dt) {
              deductionFailed = true;
              break;
            }
          }
        }

        if (deductionFailed) continue;

        // Build template argument AST list
        List<TemplateArgumentAST*>* templArgList = nullptr;
        auto argListIt = &templArgList;

        for (auto& deducedType : deducedTypes) {
          auto typeId = TypeIdAST::create(arena());
          typeId->type = deducedType;

          auto typeArg = TypeTemplateArgumentAST::create(arena());
          typeArg->typeId = typeId;

          *argListIt = make_list_node<TemplateArgumentAST>(arena(), typeArg);
          argListIt = &(*argListIt)->next;
        }

        // Instantiate the function template
        auto instantiated =
            ASTRewriter::instantiate(check.unit_, templArgList, func);

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
        auto minRequired = getMinRequiredArgs(func, paramCount);
        if (argCount < minRequired) continue;
      }

      OverloadCandidate cand{func};
      cand.viable = true;

      // Check implicit object parameter for non-static member functions
      if (isMemberCall && !func->isStatic()) {
        auto funcCv = type->cvQualifiers();
        auto funcRef = type->refQualifier();

        auto objCvInt = static_cast<int>(objectCv);
        auto funcCvInt = static_cast<int>(funcCv);
        if ((objCvInt & ~funcCvInt) != 0) {
          continue;
        }

        // ref-qualifier check
        if (funcRef == RefQualifier::kLvalue) {
          if (objectValueCategory == ValueCategory::kPrValue) continue;
        } else if (funcRef == RefQualifier::kRvalue) {
          if (objectValueCategory == ValueCategory::kLValue) continue;
        }

        cand.exactCvMatch = (objectCv == funcCv);
      }

      auto paramIt = type->parameterTypes().begin();
      auto paramEnd = type->parameterTypes().end();
      for (auto argIt = ast->expressionList; argIt && paramIt != paramEnd;
           argIt = argIt->next, ++paramIt) {
        auto paramType = *paramIt;

        auto conv = check.checkImplicitConversion(argIt->value, paramType);
        if (conv.rank == ConversionRank::kNone) {
          cand.viable = false;
          break;
        }
        cand.conversions.push_back(conv);
      }

      if (cand.viable) candidates.push_back(cand);
    }

    auto [bestPtr, ambiguous] =
        selectBestCandidate(candidates, /*useCvTiebreaker=*/isMemberCall);

    if (!bestPtr) {
      error(ast->firstSourceLocation(), "no matching function for call");
      return;
    } else if (ambiguous) {
      error(ast->firstSourceLocation(), "call to function is ambiguous");
      return;
    }

    auto function = bestPtr->symbol;
    auto& selectedCandidate = *bestPtr;
    ast->baseExpression->type = function->type();

    if (auto id = ast_cast<IdExpressionAST>(ast->baseExpression)) {
      id->symbol = function;
    } else if (auto member =
                   ast_cast<MemberExpressionAST>(ast->baseExpression)) {
      member->symbol = function;
    }

    auto selectedType = type_cast<FunctionType>(function->type());
    if (selectedType) {
      auto totalParams =
          static_cast<int>(selectedType->parameterTypes().size());
      appendDefaultArguments(ast, function, argCount, totalParams);
    }

    int argIdx = 0;
    for (auto it = ast->expressionList;
         it && argIdx < selectedCandidate.conversions.size();
         it = it->next, ++argIdx) {
      check.applyImplicitConversion(selectedCandidate.conversions[argIdx],
                                    it->value);
    }
  }

  // adl
  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (!idExpr->symbol && !idExpr->nestedNameSpecifier) {
      if (auto nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
        auto name = nameId->identifier;
        if (name && check.scope_) {
          auto adlCandidates =
              Lookup{check.scope_}.argumentDependentLookup(name, argumentTypes);
          if (!adlCandidates.empty()) {
            auto func = adlCandidates.front();
            idExpr->symbol = func;
            ast->baseExpression->type = func->type();
          }
        }
      }
    }
  }

  auto functionType = type_cast<FunctionType>(ast->baseExpression->type);

  if (!functionType) {
    if (control()->is_pointer(ast->baseExpression->type)) {
      // resolve pointer to function type
      functionType = type_cast<FunctionType>(
          control()->get_element_type(ast->baseExpression->type));

      if (functionType) {
        (void)ensure_prvalue(ast->baseExpression);
      }
    }
  }

  if (!functionType) {
    if (auto classType = type_cast<ClassType>(
            control()->remove_cvref(ast->baseExpression->type))) {
      auto classSymbol = classType->symbol();
      if (classSymbol) {
        auto operatorName = control()->getOperatorId(TokenKind::T_LPAREN);
        auto candidates = check.findOverloads(classSymbol, operatorName);
        if (!candidates.empty()) {
          auto operatorFunc = candidates.front();
          functionType = type_cast<FunctionType>(operatorFunc->type());
          if (functionType) {
            auto ar = arena();
            auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);
            auto memberExpr = MemberExpressionAST::create(
                ar, ast->baseExpression, /*nestedNameSpecifier=*/nullptr,
                /*unqualifiedId=*/opId, /*symbol=*/operatorFunc,
                /*accessOp=*/TokenKind::T_DOT,
                /*isTemplateIntroduced=*/false,
                /*valueCategory=*/ValueCategory::kLValue,
                /*type=*/operatorFunc->type());
            ast->baseExpression = memberExpr;
          }
        }
      }
    }
  }

  if (!functionType) {
    // todo: enable when support for the __builtin_<op> functions is added

#if false
    error(ast->firstSourceLocation(),
          std::format("invalid call of type '{}'",
                      to_string(ast->baseExpression->type)));
#endif
    return;
  }

  auto deduceAndInstantiateTemplate = [&](FunctionSymbol* funcSym) -> bool {
    if (!funcSym || !funcSym->templateDeclaration()) return false;

    auto& paramTypes = functionType->parameterTypes();
    auto templateDecl = funcSym->templateDeclaration();

    int templateParamCount = 0;
    for (auto p : ListView{templateDecl->templateParameterList}) {
      (void)p;
      ++templateParamCount;
    }

    std::vector<const Type*> deducedTypes(templateParamCount, nullptr);
    bool deductionFailed = false;

    int pi = 0;
    for (auto argIt = ast->expressionList;
         argIt && pi < static_cast<int>(paramTypes.size());
         argIt = argIt->next, ++pi) {
      auto argType = argIt->value ? argIt->value->type : nullptr;
      if (!argType) {
        deductionFailed = true;
        break;
      }

      auto bareParam = control()->remove_cvref(paramTypes[pi]);
      if (auto tpt = type_cast<TypeParameterType>(bareParam)) {
        auto idx = tpt->index();
        if (idx >= 0 && idx < templateParamCount) {
          auto bareArg = control()->remove_cvref(argType);
          if (!deducedTypes[idx]) {
            deducedTypes[idx] = bareArg;
          } else if (deducedTypes[idx] != bareArg) {
            deductionFailed = true;
            break;
          }
        }
      }
    }

    if (!deductionFailed) {
      for (auto& dt : deducedTypes) {
        if (!dt) {
          deductionFailed = true;
          break;
        }
      }
    }

    if (deductionFailed) return false;

    // Build template argument AST list
    List<TemplateArgumentAST*>* templArgList = nullptr;
    auto templArgIt = &templArgList;

    for (auto& deducedType : deducedTypes) {
      auto typeId = TypeIdAST::create(arena());
      typeId->type = deducedType;

      auto typeArg = TypeTemplateArgumentAST::create(arena());
      typeArg->typeId = typeId;

      *templArgIt = make_list_node<TemplateArgumentAST>(arena(), typeArg);
      templArgIt = &(*templArgIt)->next;
    }

    auto instantiated =
        ASTRewriter::instantiate(check.unit_, templArgList, funcSym);

    if (!instantiated) return false;

    auto instFunc = symbol_cast<FunctionSymbol>(instantiated);
    if (!instFunc) return false;

    auto instType = type_cast<FunctionType>(instFunc->type());
    if (!instType) return false;

    if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
      idExpr->symbol = instFunc;
    } else if (auto memberExpr =
                   ast_cast<MemberExpressionAST>(ast->baseExpression)) {
      memberExpr->symbol = instFunc;
    }
    ast->baseExpression->type = instFunc->type();
    functionType = instType;
    return true;
  };

  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    deduceAndInstantiateTemplate(symbol_cast<FunctionSymbol>(idExpr->symbol));
  }

  if (auto memberExpr = ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    deduceAndInstantiateTemplate(
        symbol_cast<FunctionSymbol>(memberExpr->symbol));
  }

  // Check the arguments
  const auto& parameterTypes = functionType->parameterTypes();

  int argc = 0;
  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) {
      error(ast->firstSourceLocation(),
            "invalid call with null argument expression");
      continue;
    }

    if (argc >= parameterTypes.size()) {
      if (functionType->isVariadic()) {
        // do the promotion for the variadic arguments
        (void)ensure_prvalue(it->value);
        adjust_cv(it->value);

        if (integral_promotion(it->value)) continue;
        if (floating_point_promotion(it->value)) continue;

        continue;
      }

      error(it->value->firstSourceLocation(),
            std::format("too many arguments for function of type '{}'",
                        to_string(functionType)));
      break;
    }

    auto targetType = parameterTypes[argc];
    ++argc;

    if (is_parsing_cxx()) {
      if (control()->is_reference(targetType)) {
        // TODO: check reference binding
        continue;
      }
    }

    if (!implicit_conversion(it->value, targetType)) {
      error(it->value->firstSourceLocation(),
            std::format("invalid argument of type '{}' for parameter of type "
                        "'{}'",
                        to_string(it->value->type), to_string(targetType)));
    }
  }

  setResultTypeAndValueCategory(ast, functionType);

  if (ast->valueCategory == ValueCategory::kPrValue) {
    adjust_cv(ast);
  }
}

void TypeChecker::Visitor::setResultTypeAndValueCategory(ExpressionAST* ast,
                                                         const Type* type) {
  auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return;
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
    if (control()->is_const(ast->baseExpression->type)) return false;

    if (is_parsing_cxx() &&
        !control()->is_arithmetic(ast->baseExpression->type))
      return false;

    if (is_parsing_c() &&
        !control()->is_arithmetic_or_unscoped_enum(ast->baseExpression->type))
      return false;

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
  if (!ast->typeId) return;

  auto fullTargetType = ast->typeId->type;
  ast->type = fullTargetType;

  if (auto refType = type_cast<LvalueReferenceType>(fullTargetType)) {
    ast->type = refType->elementType();
    ast->valueCategory = ValueCategory::kLValue;
  } else if (auto rrefType = type_cast<RvalueReferenceType>(fullTargetType)) {
    ast->type = rrefType->elementType();
    if (type_cast<FunctionType>(ast->type)) {
      ast->valueCategory = ValueCategory::kLValue;
    } else {
      ast->valueCategory = ValueCategory::kXValue;
    }
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }

  switch (ast->castOp) {
    case TokenKind::T_STATIC_CAST: {
      if (check_static_cast(ast->expression, ast->type, ast->valueCategory))
        break;
      error(ast->firstSourceLocation(),
            std::format("invalid static_cast of '{}' to '{}'",
                        to_string(ast->expression->type),
                        to_string(fullTargetType)));
      break;
    }

    case TokenKind::T_CONST_CAST: {
      if (check_const_cast(ast->expression, ast->type, ast->valueCategory))
        break;
      error(ast->firstSourceLocation(),
            std::format("invalid const_cast of '{}' to '{}'",
                        to_string(ast->expression->type),
                        to_string(fullTargetType)));
      break;
    }

    case TokenKind::T_REINTERPRET_CAST: {
      if (check_reinterpret_cast(ast->expression, ast->type,
                                 ast->valueCategory))
        break;
      error(ast->firstSourceLocation(),
            std::format("invalid reinterpret_cast of '{}' to '{}'",
                        to_string(ast->expression->type),
                        to_string(fullTargetType)));
      break;
    }

    case TokenKind::T_DYNAMIC_CAST: {
      // no rtti yet
      warning(ast->firstSourceLocation(), "dynamic_cast is not supported yet");
      break;
    }

    default:
      break;
  }  // switch

  if (ast->valueCategory == ValueCategory::kPrValue) {
    adjust_cv(ast);
  }
}

auto TypeChecker::Visitor::check_static_cast(ExpressionAST*& expression,
                                             const Type* targetType,
                                             ValueCategory targetVC) -> bool {
  if (!expression || !expression->type) return false;

  if (control()->is_void(targetType)) return true;

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
  (void)ensure_prvalue(source);
  adjust_cv(source);

  auto sourceType = source->type;

  if (control()->is_scoped_enum(sourceType) &&
      (control()->is_integral(targetType) ||
       control()->is_floating_point(targetType))) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = control()->is_integral(targetType)
                         ? ImplicitCastKind::kIntegralConversion
                         : ImplicitCastKind::kFloatingIntegralConversion;
    cast->expression = source;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if ((control()->is_integral(sourceType) || control()->is_enum(sourceType) ||
       control()->is_scoped_enum(sourceType)) &&
      (control()->is_enum(targetType) ||
       control()->is_scoped_enum(targetType))) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = ImplicitCastKind::kIntegralConversion;
    cast->expression = source;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if (control()->is_floating_point(sourceType) &&
      (control()->is_enum(targetType) ||
       control()->is_scoped_enum(targetType))) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = ImplicitCastKind::kFloatingIntegralConversion;
    cast->expression = source;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if (control()->is_floating_point(sourceType) &&
      control()->is_floating_point(targetType)) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = ImplicitCastKind::kFloatingPointConversion;
    cast->expression = source;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (auto targetPtr = as_pointer(targetType)) {
      auto srcElem = control()->remove_cv(sourcePtr->elementType());
      auto tgtElem = control()->remove_cv(targetPtr->elementType());
      auto srcCV = control()->get_cv_qualifiers(sourcePtr->elementType());
      auto tgtCV = control()->get_cv_qualifiers(targetPtr->elementType());
      if (control()->is_base_of(srcElem, tgtElem) &&
          check_cv_qualifiers(tgtCV, srcCV)) {
        auto cast = ImplicitCastExpressionAST::create(arena());
        cast->castKind = ImplicitCastKind::kPointerConversion;
        cast->expression = source;
        cast->type = targetType;
        cast->valueCategory = ValueCategory::kPrValue;
        expression = cast;
        return true;
      }
    }
  }

  if (auto sourcePtr = as_pointer(sourceType)) {
    if (control()->is_void(control()->remove_cv(sourcePtr->elementType()))) {
      if (auto targetPtr = as_pointer(targetType)) {
        if (control()->is_object(
                control()->remove_cv(targetPtr->elementType()))) {
          auto srcCV = control()->get_cv_qualifiers(sourcePtr->elementType());
          auto tgtCV = control()->get_cv_qualifiers(targetPtr->elementType());
          if (check_cv_qualifiers(tgtCV, srcCV)) {
            expression = source;
            return true;
          }
        }
      }
    }
  }

  if (auto srcMem = type_cast<MemberObjectPointerType>(
          control()->remove_cv(sourceType))) {
    if (auto tgtMem = type_cast<MemberObjectPointerType>(
            control()->remove_cv(targetType))) {
      auto srcClass = srcMem->classType();
      auto tgtClass = tgtMem->classType();
      if (control()->is_base_of(tgtClass, srcClass)) {
        auto srcElemCV = control()->get_cv_qualifiers(srcMem->elementType());
        auto tgtElemCV = control()->get_cv_qualifiers(tgtMem->elementType());
        if (check_cv_qualifiers(tgtElemCV, srcElemCV) &&
            control()->is_same(control()->remove_cv(srcMem->elementType()),
                               control()->remove_cv(tgtMem->elementType()))) {
          auto cast = ImplicitCastExpressionAST::create(arena());
          cast->castKind = ImplicitCastKind::kPointerToMemberConversion;
          cast->expression = source;
          cast->type = targetType;
          cast->valueCategory = ValueCategory::kPrValue;
          expression = cast;
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
  auto srcCV = control()->get_cv_qualifiers(sourceType);
  sourceType = control()->remove_cv(sourceType);

  auto tgtCV = control()->get_cv_qualifiers(targetType);
  auto tgtBase = control()->remove_cv(targetType);

  if (!check_cv_qualifiers(tgtCV, srcCV)) return false;

  if (!control()->is_base_of(sourceType, tgtBase)) return false;

  return true;
}

auto TypeChecker::Visitor::is_reference_compatible(const Type* targetType,
                                                   const Type* sourceType)
    -> bool {
  auto t1 = control()->remove_cv(targetType);
  auto t2 = control()->remove_cv(sourceType);
  if (!control()->is_same(t1, t2)) {
    if (!control()->is_base_of(t1, t2)) return false;
  }
  auto cvTarget = control()->get_cv_qualifiers(targetType);
  auto cvSource = control()->get_cv_qualifiers(sourceType);
  return check_cv_qualifiers(cvTarget, cvSource);
}

auto TypeChecker::Visitor::check_const_cast(ExpressionAST*& expression,
                                            const Type* targetType,
                                            ValueCategory targetVC) -> bool {
  if (!targetType) return false;

  auto sourceType = expression->type;
  const Type* T1 = nullptr;
  const Type* T2 = nullptr;

  if (auto targetPtr =
          type_cast<PointerType>(control()->remove_cv(targetType))) {
    auto sourcePtr = type_cast<PointerType>(control()->remove_cv(sourceType));
    if (!sourcePtr) return false;

    (void)ensure_prvalue(expression);
    adjust_cv(expression);

    T1 = sourcePtr->elementType();
    T2 = targetPtr->elementType();
  } else if (auto targetPtrm = type_cast<MemberObjectPointerType>(
                 control()->remove_cv(targetType))) {
    auto sourcePtrm =
        type_cast<MemberObjectPointerType>(control()->remove_cv(sourceType));
    if (!sourcePtrm) return false;

    if (!control()->is_same(sourcePtrm->classType(), targetPtrm->classType()))
      return false;

    (void)ensure_prvalue(expression);
    adjust_cv(expression);

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
    } else if (is_prvalue(expression) && (control()->is_class(sourceType) ||
                                          control()->is_array(sourceType))) {
      (void)temporary_materialization_conversion(expression);
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
    if (control()->is_same(control()->remove_cv(curr1),
                           control()->remove_cv(curr2))) {
      return true;
    }

    auto u1 = control()->remove_cv(curr1);
    auto u2 = control()->remove_cv(curr2);

    if (auto p1 = as_pointer(u1)) {
      if (auto p2 = as_pointer(u2)) {
        curr1 = p1->elementType();
        curr2 = p2->elementType();
        continue;
      }
    }

    if (auto m1 = type_cast<MemberObjectPointerType>(u1)) {
      if (auto m2 = type_cast<MemberObjectPointerType>(u2)) {
        if (!control()->is_same(m1->classType(), m2->classType())) return false;
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
    auto ptrToSource = control()->add_pointer(sourceType);
    auto ptrToTarget = control()->add_pointer(targetType);
    (void)ptrToSource;
    (void)ptrToTarget;
    if ((control()->is_object(control()->remove_cv(sourceType)) &&
         control()->is_object(control()->remove_cv(targetType))) ||
        (control()->is_function(sourceType) &&
         control()->is_function(targetType))) {
      if (casts_away_constness(sourceType, targetType)) return false;
      return true;
    }
    return false;
  }

  (void)ensure_prvalue(expression);
  adjust_cv(expression);
  sourceType = expression->type;

  if (control()->is_same(control()->remove_cv(sourceType),
                         control()->remove_cv(targetType)))
    return true;

  if (control()->is_pointer(sourceType) && control()->is_integral(targetType)) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = ImplicitCastKind::kIntegralConversion;
    cast->expression = expression;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if ((control()->is_integral(sourceType) || control()->is_enum(sourceType) ||
       control()->is_scoped_enum(sourceType)) &&
      control()->is_pointer(targetType)) {
    auto cast = ImplicitCastExpressionAST::create(arena());
    cast->castKind = ImplicitCastKind::kPointerConversion;
    cast->expression = expression;
    cast->type = targetType;
    cast->valueCategory = ValueCategory::kPrValue;
    expression = cast;
    return true;
  }

  if (control()->is_pointer(sourceType) && control()->is_pointer(targetType)) {
    auto srcPtr = as_pointer(sourceType);
    auto tgtPtr = as_pointer(targetType);
    if (srcPtr && tgtPtr &&
        casts_away_constness(srcPtr->elementType(), tgtPtr->elementType()))
      return false;
    return true;
  }

  if (control()->is_member_pointer(sourceType) &&
      control()->is_member_pointer(targetType)) {
    return true;
  }

  if (control()->is_null_pointer(sourceType) &&
      control()->is_integral(targetType))
    return true;

  return false;
}

auto TypeChecker::Visitor::casts_away_constness(const Type* sourceType,
                                                const Type* targetType)
    -> bool {
  auto srcCV = control()->get_cv_qualifiers(sourceType);
  auto tgtCV = control()->get_cv_qualifiers(targetType);

  if (!check_cv_qualifiers(tgtCV, srcCV)) return true;

  auto srcPtr = as_pointer(control()->remove_cv(sourceType));
  auto tgtPtr = as_pointer(control()->remove_cv(targetType));
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

void TypeChecker::Visitor::operator()(BuiltinBitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BuiltinOffsetofExpressionAST* ast) {
  ast->type = control()->getSizeType();

  auto classType =
      ast->typeId ? type_cast<ClassType>(ast->typeId->type) : nullptr;

  if (!classType) {
    error(ast->firstSourceLocation(), "expected a type");
    return;
  }

  if (!ast->identifier) {
    return;
  }

  auto symbol = classType->symbol();
  auto member = Lookup{scope()}.qualifiedLookup(symbol, ast->identifier);

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
      auto currentClass =
          type_cast<ClassType>(control()->remove_cvref(field->type()));

      if (!currentClass) {
        error(designator->firstSourceLocation(),
              std::format("expected a class or union type, but got '{}'",
                          to_string(field->type())));
        break;
      }

      auto member = Lookup{scope()}.qualifiedLookup(currentClass->symbol(),
                                                    dot->identifier);

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
      if (!control()->is_array(field->type()) &&
          !control()->is_pointer(field->type())) {
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

void TypeChecker::Visitor::operator()(TypeidExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(TypeidOfTypeExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(SpliceExpressionAST* ast) {
  if (!ast->splicer) return;
  if (!ast->splicer->expression) return;
  ast->type = ast->splicer->expression->type;
}

void TypeChecker::Visitor::operator()(GlobalScopeReflectExpressionAST* ast) {
  ast->type = control()->getBuiltinMetaInfoType();
}

void TypeChecker::Visitor::operator()(NamespaceReflectExpressionAST* ast) {
  ast->type = control()->getBuiltinMetaInfoType();
}

void TypeChecker::Visitor::operator()(TypeIdReflectExpressionAST* ast) {
  ast->type = control()->getBuiltinMetaInfoType();
}

void TypeChecker::Visitor::operator()(ReflectExpressionAST* ast) {
  ast->type = control()->getBuiltinMetaInfoType();
}

void TypeChecker::Visitor::operator()(LabelAddressExpressionAST* ast) {
  ast->type = control()->getPointerType(control()->getVoidType());
}

void TypeChecker::Visitor::operator()(UnaryExpressionAST* ast) {
  if (!ast->expression) return;

  if (auto symbol = check.lookupOperator(ast->expression->type, ast->op)) {
    ast->symbol = symbol;
    setResultTypeAndValueCategory(ast, symbol->type());
    return;
  }

  switch (ast->op) {
    case TokenKind::T_STAR: {
      auto pointerType = as_pointer(ast->expression->type);
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
          auto parentClass = field->parent();
          auto classType = type_cast<ClassType>(parentClass->type());

          ast->type =
              control()->getMemberObjectPointerType(classType, field->type());

          ast->valueCategory = ValueCategory::kPrValue;

          break;
        }

        if (auto function = symbol_cast<FunctionSymbol>(symbol);
            function && !function->isStatic()) {
          auto functionType = type_cast<FunctionType>(function->type());
          auto parentClass = function->parent();
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

      if (!control()->is_const(ast->expression->type)) {
        const auto ty = ast->expression->type;

        if (is_parsing_cxx() ? control()->is_arithmetic(ty)
                             : control()->is_arithmetic_or_unscoped_enum(ty)) {
          ast->type = ty;
          ast->valueCategory = ValueCategory::kLValue;
          break;
        }

        if (auto ptrTy = as_pointer(ty)) {
          if (!control()->is_void(ptrTy->elementType())) {
            ast->type = ptrTy;
            ast->valueCategory = ValueCategory::kLValue;
            break;
          }
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

      if (!control()->is_const(ast->expression->type)) {
        auto ty = ast->expression->type;

        if (is_parsing_cxx() ? control()->is_arithmetic(ty)
                             : control()->is_arithmetic_or_unscoped_enum(ty)) {
          ast->type = ty;
          ast->valueCategory = ValueCategory::kLValue;
          break;
        }

        if (auto ptrTy = as_pointer(ty)) {
          if (ptrTy && !control()->is_void(ptrTy->elementType())) {
            ast->type = ptrTy;
            ast->valueCategory = ValueCategory::kLValue;
            break;
          }
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

  if (ast->expression) {
    ast->value = control()->memoryLayout()->sizeOf(ast->expression->type);
  }
}

void TypeChecker::Visitor::operator()(SizeofTypeExpressionAST* ast) {
  ast->type = control()->getSizeType();

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

    std::vector<OverloadCandidate> candidates;

    for (auto ctor : classSymbol->constructors()) {
      if (ctor->canonical() != ctor) continue;

      auto type = type_cast<FunctionType>(ctor->type());
      if (!type) continue;

      if (type->parameterTypes().size() != args.size()) continue;

      OverloadCandidate cand{ctor};
      cand.viable = true;

      auto paramIt = type->parameterTypes().begin();
      for (auto arg : args) {
        auto paramType = *paramIt++;
        auto conv = check.checkImplicitConversion(arg, paramType);
        if (conv.rank == ConversionRank::kNone) {
          cand.viable = false;
          break;
        }
        cand.conversions.push_back(conv);
      }

      if (cand.viable) candidates.push_back(cand);
    }

    if (auto result = selectBestCandidate(candidates); result.best) {
      ast->constructorSymbol = result.best->symbol;

      for (size_t i = 0; i < args.size(); ++i) {
        check.applyImplicitConversion(result.best->conversions[i], args[i]);
      }
    }
  }
}

void TypeChecker::Visitor::operator()(DeleteExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(CastExpressionAST* ast) {
  if (!ast->typeId) return;

  auto fullTargetType = ast->typeId->type;
  ast->type = fullTargetType;

  if (auto refType = type_cast<LvalueReferenceType>(fullTargetType)) {
    ast->type = refType->elementType();
    ast->valueCategory = ValueCategory::kLValue;
  } else if (auto rrefType = type_cast<RvalueReferenceType>(fullTargetType)) {
    ast->type = rrefType->elementType();
    if (type_cast<FunctionType>(ast->type)) {
      ast->valueCategory = ValueCategory::kLValue;
    } else {
      ast->valueCategory = ValueCategory::kXValue;
    }
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }

  auto targetType = ast->type;
  auto targetVC = ast->valueCategory;

  if (auto expr = ast->expression;
      check_const_cast(expr, targetType, targetVC)) {
    ast->expression = expr;
    if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
    return;
  }

  if (auto expr = ast->expression;
      check_static_cast(expr, targetType, targetVC)) {
    ast->expression = expr;
    if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
    return;
  }

  {
    auto expr = ast->expression;
    auto cvAdjustedTarget = targetType;
    if (control()->is_pointer(targetType) &&
        control()->is_pointer(expr->type)) {
      if (check_static_cast(expr, targetType, targetVC)) {
        ast->expression = expr;
        if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
        return;
      }
    }
    (void)cvAdjustedTarget;
  }

  if (auto expr = ast->expression;
      check_reinterpret_cast(expr, targetType, targetVC)) {
    ast->expression = expr;
    if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
    return;
  }

  // Try reinterpret_cast permissively (C-style casts can cast away const).
  if (auto expr = ast->expression;
      check_reinterpret_cast_permissive(expr, targetType, targetVC)) {
    ast->expression = expr;
    if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
    return;
  }

  if (ast->valueCategory == ValueCategory::kPrValue) adjust_cv(ast);
}

auto TypeChecker::Visitor::check_reinterpret_cast_permissive(
    ExpressionAST*& expression, const Type* targetType, ValueCategory targetVC)
    -> bool {
  if (!expression || !expression->type) return false;

  auto sourceType = expression->type;

  if (targetVC == ValueCategory::kLValue ||
      targetVC == ValueCategory::kXValue) {
    if (!is_glvalue(expression)) return false;
    if ((control()->is_object(control()->remove_cv(sourceType)) &&
         control()->is_object(control()->remove_cv(targetType))) ||
        (control()->is_function(sourceType) &&
         control()->is_function(targetType))) {
      return true;
    }
    return false;
  }

  (void)ensure_prvalue(expression);
  adjust_cv(expression);
  sourceType = expression->type;

  if (control()->is_same(control()->remove_cv(sourceType),
                         control()->remove_cv(targetType)))
    return true;

  if (control()->is_pointer(sourceType) && control()->is_integral(targetType))
    return true;

  if ((control()->is_integral(sourceType) || control()->is_enum(sourceType) ||
       control()->is_scoped_enum(sourceType)) &&
      control()->is_pointer(targetType))
    return true;

  if (control()->is_pointer(sourceType) && control()->is_pointer(targetType))
    return true;

  if (control()->is_member_pointer(sourceType) &&
      control()->is_member_pointer(targetType))
    return true;

  if (control()->is_null_pointer(sourceType) &&
      control()->is_integral(targetType))
    return true;

  return false;
}

void TypeChecker::Visitor::operator()(ImplicitCastExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(BinaryExpressionAST* ast) {
  if (!ast->leftExpression) return;
  if (!ast->rightExpression) return;

  auto leftType = ast->leftExpression->type;
  auto rightType = ast->rightExpression->type;
  if (type_cast<AutoType>(control()->remove_cvref(leftType)) ||
      type_cast<AutoType>(control()->remove_cvref(rightType)))
    return;

  switch (ast->op) {
    case TokenKind::T_DOT_STAR:
      // TODO check for built-in .* operator
      break;

    case TokenKind::T_MINUS_GREATER_STAR:
      // TODO check for built-in ->* operator
      break;

    case TokenKind::T_STAR:
      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);
      break;

    case TokenKind::T_SLASH:
      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);
      break;

    case TokenKind::T_PLUS:
      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      check_addition(ast);
      break;

    case TokenKind::T_MINUS:
      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      check_subtraction(ast);
      break;

    case TokenKind::T_PERCENT:
      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                              ast->rightExpression);

      break;

    case TokenKind::T_LESS_LESS:
    case TokenKind::T_GREATER_GREATER:
      if (control()->is_class_or_union(ast->leftExpression->type) ||
          control()->is_class_or_union(ast->rightExpression->type)) {
        if (auto operatorFunc =
                check.lookupOperator(ast->leftExpression->type, ast->op,
                                     ast->rightExpression->type)) {
          ast->symbol = operatorFunc;
          ast->type = operatorFunc->type();
          setResultTypeAndValueCategory(ast, operatorFunc->type());
          break;
        }

        error(ast->opLoc,
              std::format("'operator {}' is not defined for types {} and {}",
                          Token::spell(ast->op),
                          to_string(ast->leftExpression->type),
                          to_string(ast->rightExpression->type)));
        break;
      }

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
    case TokenKind::T_GREATER: {
      ast->type = control()->getBoolType();

      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        setResultTypeAndValueCategory(ast, operatorFunc->type());
        break;
      }

      (void)lvalue_to_rvalue_conversion(ast->leftExpression);
      (void)function_to_pointer_conversion(ast->leftExpression);

      (void)lvalue_to_rvalue_conversion(ast->rightExpression);
      (void)function_to_pointer_conversion(ast->rightExpression);

      // handle array-to-pointer conversion if needed
      if (control()->is_pointer(ast->leftExpression->type)) {
        (void)array_to_pointer_conversion(ast->rightExpression);
      } else if (control()->is_pointer(ast->rightExpression->type)) {
        (void)array_to_pointer_conversion(ast->leftExpression);
      }

      if (usual_arithmetic_conversion(ast->leftExpression,
                                      ast->rightExpression)) {
        ast->type = control()->getBoolType();
        break;
      }

      if (control()->is_pointer(ast->leftExpression->type) &&
          control()->is_pointer(ast->rightExpression->type)) {
        auto compositeType =
            composite_pointer_type(ast->leftExpression, ast->rightExpression);
        (void)implicit_conversion(ast->leftExpression, compositeType);
        (void)implicit_conversion(ast->rightExpression, compositeType);
        break;
      }

      error(ast->firstSourceLocation(),
            std::format("invalid operands to binary expression ('{}' and '{}')",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));

      break;
    }

    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL: {
      ast->type = control()->getBoolType();

      if (auto operatorFunc = check.lookupOperator(
              ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
        ast->symbol = operatorFunc;
        // Get the return type from the function type
        if (auto funcType = type_cast<FunctionType>(operatorFunc->type())) {
          ast->type = funcType->returnType();
        } else {
          ast->type = operatorFunc->type();
        }
        break;
      }

      // apply lvalue-to-rvalue and function-to-pointer con
      // versions to both operands
      (void)lvalue_to_rvalue_conversion(ast->leftExpression);
      (void)function_to_pointer_conversion(ast->leftExpression);

      (void)lvalue_to_rvalue_conversion(ast->rightExpression);
      (void)function_to_pointer_conversion(ast->rightExpression);

      auto is_null_pointer = [this](ExpressionAST* expr) -> bool {
        if (control()->is_null_pointer(expr->type)) return true;

        // strip nested expressions
        while (auto nestedExpr = ast_cast<NestedExpressionAST>(expr)) {
          expr = nestedExpr->expression;
        }

        if (auto intLiteral = ast_cast<IntLiteralExpressionAST>(expr)) {
          if (intLiteral->literal->value() == "0") return true;
        }

        if (auto boolLiteral = ast_cast<BoolLiteralExpressionAST>(expr)) {
          if (!boolLiteral->isTrue) return true;
        }

        return false;
      };

      // handle array-to-pointer conversion if needed
      if (control()->is_pointer(ast->leftExpression->type) ||
          is_null_pointer(ast->leftExpression)) {
        (void)array_to_pointer_conversion(ast->rightExpression);
      } else if (control()->is_pointer(ast->rightExpression->type) ||
                 is_null_pointer(ast->rightExpression)) {
        (void)array_to_pointer_conversion(ast->leftExpression);
      }

      if (usual_arithmetic_conversion(ast->leftExpression,
                                      ast->rightExpression)) {
        ast->type = control()->getBoolType();
        break;
      }

      if (control()->is_pointer(ast->leftExpression->type) &&
          control()->is_same(ast->rightExpression->type,
                             control()->getBoolType())) {
        (void)implicit_conversion(ast->rightExpression,
                                  ast->leftExpression->type);
      } else if (control()->is_pointer(ast->rightExpression->type) &&
                 control()->is_same(ast->leftExpression->type,
                                    control()->getBoolType())) {
        (void)implicit_conversion(ast->leftExpression,
                                  ast->rightExpression->type);
      }

      if ((control()->is_pointer(ast->leftExpression->type) ||
           is_null_pointer(ast->leftExpression)) &&
          (control()->is_pointer(ast->rightExpression->type) ||
           is_null_pointer(ast->rightExpression))) {
        auto compositeType =
            composite_pointer_type(ast->leftExpression, ast->rightExpression);
        (void)implicit_conversion(ast->leftExpression, compositeType);
        (void)implicit_conversion(ast->rightExpression, compositeType);
        break;
      }

      error(ast->firstSourceLocation(),
            std::format("invalid operands to binary expression ('{}' and '{}')",
                        to_string(ast->leftExpression->type),
                        to_string(ast->rightExpression->type)));

      break;
    }

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

    ast->type = ast->iftrueExpression->type;

    ast->valueCategory = ast->iftrueExpression->valueCategory;

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

    (void)ensure_prvalue(ast->iftrueExpression);
    (void)ensure_prvalue(ast->iffalseExpression);

    ast->type = ast->iftrueExpression->type;
    ast->valueCategory = ValueCategory::kPrValue;
    return true;
  };

  auto check_compatible_pointers = [&] {
    if (!control()->is_pointer(ast->iftrueExpression->type) &&
        !control()->is_pointer(ast->iffalseExpression->type))
      return false;

    (void)ensure_prvalue(ast->iftrueExpression);
    (void)ensure_prvalue(ast->iffalseExpression);

    ast->type =
        composite_pointer_type(ast->iftrueExpression, ast->iffalseExpression);

    ast->valueCategory = ValueCategory::kPrValue;

    if (!ast->type) return false;

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

  if (is_parsing_c()) {
    // in C, both expressions must be prvalues
    (void)ensure_prvalue(ast->iftrueExpression);
    (void)ensure_prvalue(ast->iffalseExpression);
  }

  if (check_void_type()) return;
  if (check_same_type_and_value_category()) return;

  (void)array_to_pointer_conversion(ast->iftrueExpression);
  (void)function_to_pointer_conversion(ast->iftrueExpression);

  (void)array_to_pointer_conversion(ast->iffalseExpression);
  (void)function_to_pointer_conversion(ast->iffalseExpression);

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

void TypeChecker::Visitor::operator()(YieldExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(ThrowExpressionAST* ast) {
  ast->type = control()->getVoidType();
  ast->valueCategory = ValueCategory::kPrValue;
}

void TypeChecker::Visitor::operator()(AssignmentExpressionAST* ast) {
  if (!ast->leftExpression || !ast->rightExpression) return;

  if (auto operatorFunc = check.lookupOperator(
          ast->leftExpression->type, ast->op, ast->rightExpression->type)) {
    ast->symbol = operatorFunc;
    setResultTypeAndValueCategory(ast, operatorFunc->type());
    return;
  }

  if (!is_lvalue(ast->leftExpression)) {
    error(ast->opLoc, std::format("cannot assign to an rvalue of type '{}'",
                                  to_string(ast->leftExpression->type)));
    return;
  }

  ast->type = ast->leftExpression->type;

  if (is_parsing_c()) {
    ast->valueCategory = ValueCategory::kPrValue;
  } else {
    ast->valueCategory = ast->leftExpression->valueCategory;
  }

  (void)implicit_conversion(ast->rightExpression, ast->type);
}

void TypeChecker::Visitor::operator()(TargetExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(RightExpressionAST* ast) {}

void TypeChecker::Visitor::operator()(CompoundAssignmentExpressionAST* ast) {
  if (!ast->targetExpression || !ast->rightExpression) return;

  if (auto operatorFunc = check.lookupOperator(
          ast->targetExpression->type, ast->op, ast->rightExpression->type)) {
    ast->symbol = operatorFunc;
    setResultTypeAndValueCategory(ast, operatorFunc->type());
    return;
  }

  if (!is_lvalue(ast->targetExpression)) {
    error(ast->opLoc, std::format("cannot assign to an rvalue of type '{}'",
                                  to_string(ast->targetExpression->type)));
    return;
  }

  ast->leftExpression->type = ast->targetExpression->type;
  ast->leftExpression->valueCategory = ast->targetExpression->valueCategory;
  ast->type = ast->targetExpression->type;

  if (is_parsing_cxx()) {
    ast->valueCategory = ValueCategory::kLValue;
  } else {
    ast->valueCategory = ValueCategory::kPrValue;
  }

  if ((ast->op == TokenKind::T_PLUS_EQUAL ||
       ast->op == TokenKind::T_MINUS_EQUAL) &&
      control()->is_pointer(ast->targetExpression->type) &&
      control()->is_integral_or_unscoped_enum(ast->rightExpression->type)) {
    // pointer addition/subtraction

    (void)ensure_prvalue(ast->leftExpression);
    adjust_cv(ast->leftExpression);

    (void)ensure_prvalue(ast->rightExpression);
    adjust_cv(ast->rightExpression);

    (void)integral_promotion(ast->rightExpression);

    if (ast->adjustExpression) {
      ast->adjustExpression->type = ast->leftExpression->type;

      (void)implicit_conversion(ast->adjustExpression, ast->type);
    }

    return;
  }

  auto commonType =
      usual_arithmetic_conversion(ast->leftExpression, ast->rightExpression);

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

void TypeChecker::Visitor::operator()(DesignatedInitializerClauseAST* ast) {}

void TypeChecker::Visitor::operator()(TypeTraitExpressionAST* ast) {
  ast->type = control()->getBoolType();
  auto interp = ASTInterpreter{check.unit_};
  auto value = interp.evaluate(ast);
  if (value.has_value()) {
    ast->value = interp.toBool(*value);
  }
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
  auto cast = ImplicitCastExpressionAST::create(arena());
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
  auto cast = ImplicitCastExpressionAST::create(arena());
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
  auto cast = ImplicitCastExpressionAST::create(arena());
  cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
  cast->expression = expr;
  cast->type = control()->add_pointer(unref);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto TypeChecker::Visitor::integral_promotion(ExpressionAST*& expr,
                                              const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_integral(expr->type) && !control()->is_enum(expr->type))
    return false;

  auto make_implicit_cast = [&](const Type* type) {
    auto cast = ImplicitCastExpressionAST::create(arena());
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
      if (!destinationType) destinationType = control()->getIntType();

      if (destinationType->kind() == TypeKind::kInt ||
          destinationType->kind() == TypeKind::kUnsignedInt) {
        make_implicit_cast(destinationType);
        return true;
      }

      return false;
    }

    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar: {
      if (!destinationType) destinationType = control()->getIntType();

      if (destinationType->kind() == TypeKind::kInt ||
          destinationType->kind() == TypeKind::kUnsignedInt) {
        make_implicit_cast(destinationType);
        return true;
      }

      return false;
    }

    case TypeKind::kBool: {
      if (!destinationType) destinationType = control()->getIntType();
      if (destinationType->kind() == TypeKind::kInt) {
        make_implicit_cast(destinationType);
        return true;
      }

      return false;
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

auto TypeChecker::Visitor::floating_point_promotion(ExpressionAST*& expr,
                                                    const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control()->is_floating_point(expr->type)) return false;

  if (!destinationType) destinationType = control()->getDoubleType();

  if (!control()->is_floating_point(destinationType)) return false;

  if (expr->type->kind() != TypeKind::kFloat) return false;

  auto cast = ImplicitCastExpressionAST::create(arena());
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

  auto cast = ImplicitCastExpressionAST::create(arena());
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

  if (control()->is_same(expr->type, destinationType)) return true;
  if (!control()->is_floating_point(expr->type)) return false;
  if (!control()->is_floating_point(destinationType)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena());
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
    auto cast = ImplicitCastExpressionAST::create(arena());
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
    auto cast = ImplicitCastExpressionAST::create(arena());
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
    auto cast = ImplicitCastExpressionAST::create(arena());
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
    const auto pointerType = as_pointer(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = as_pointer(destinationType);
    if (!destinationPointerType) return false;

    auto sourceCv = control()->get_cv_qualifiers(pointerType->elementType());
    auto targetCv =
        control()->get_cv_qualifiers(destinationPointerType->elementType());

    if (!check_cv_qualifiers(targetCv, sourceCv)) return false;

    if (!control()->is_void(destinationPointerType->elementType()))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_to_base_class_pointer = [&] {
    const auto pointerType = as_pointer(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = as_pointer(destinationType);
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

  auto can_convert_from_void_pointer = [&] {
    if (!is_parsing_c()) return false;

    const auto pointerType = as_pointer(expr->type);
    if (!pointerType) return false;

    if (!control()->is_void(pointerType->elementType())) return false;

    const auto destinationPointerType = as_pointer(destinationType);
    if (!destinationPointerType) return false;

    auto sourceCv = control()->get_cv_qualifiers(pointerType->elementType());
    auto targetCv =
        control()->get_cv_qualifiers(destinationPointerType->elementType());

    if (!check_cv_qualifiers(targetCv, sourceCv)) return false;

    make_implicit_cast();

    return true;
  };

  if (can_convert_null_pointer_literal()) return true;
  if (can_convert_to_void_pointer()) return true;
  if (can_convert_from_void_pointer()) return true;
  if (can_convert_to_base_class_pointer()) return true;

  return false;
}

auto TypeChecker::Visitor::function_pointer_conversion(
    ExpressionAST*& expr, const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto can_remove_noexcept_from_function = [&] {
    const auto pointerType = as_pointer(expr->type);
    if (!pointerType) return false;

    const auto functionType =
        type_cast<FunctionType>(pointerType->elementType());

    if (!functionType) return false;

    if (functionType->isNoexcept()) return false;

    const auto destinationPointerType = as_pointer(destinationType);

    if (!destinationPointerType) return false;

    const auto destinationFunctionType =
        type_cast<FunctionType>(destinationPointerType->elementType());

    if (!destinationFunctionType) return false;

    if (!control()->is_same(control()->remove_noexcept(functionType),
                            destinationFunctionType))
      return false;

    auto cast = ImplicitCastExpressionAST::create(arena());
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

    auto cast = ImplicitCastExpressionAST::create(arena());
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

  auto cast = ImplicitCastExpressionAST::create(arena());
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

  auto cast = ImplicitCastExpressionAST::create(arena());
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
  auto type = get_qualification_combined_type(expr->type, destinationType);
  if (!type) return false;

  if (!control()->is_same(destinationType, type)) return false;

  auto cast = ImplicitCastExpressionAST::create(arena());
  cast->castKind = ImplicitCastKind::kQualificationConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = expr->valueCategory;
  expr = cast;

  return true;
}

auto TypeChecker::Visitor::user_defined_conversion(ExpressionAST*& expr,
                                                   const Type* destinationType)
    -> bool {
  if (!is_prvalue(expr)) return false;

  if (auto classType = as_class(expr->type)) {
    if (auto classSymbol = classType->symbol()) {
      for (auto convFunc : classSymbol->conversionFunctions()) {
        auto convFuncType = type_cast<FunctionType>(convFunc->type());
        if (!convFuncType) continue;

        auto returnType = convFuncType->returnType();
        if (!returnType) continue;

        // Check for an exact match first.
        if (control()->is_same(returnType, destinationType)) {
          auto cast = ImplicitCastExpressionAST::create(arena());
          cast->castKind = ImplicitCastKind::kUserDefinedConversion;
          cast->expression = expr;
          cast->type = destinationType;
          cast->valueCategory = ValueCategory::kPrValue;
          expr = cast;
          return true;
        }

        auto convPtrType = as_pointer(returnType);
        auto destPtrType = as_pointer(destinationType);
        if (convPtrType && destPtrType) {
          auto convInnerFunc =
              type_cast<FunctionType>(convPtrType->elementType());
          auto destInnerFunc =
              type_cast<FunctionType>(destPtrType->elementType());
          if (convInnerFunc && destInnerFunc &&
              type_cast<AutoType>(convInnerFunc->returnType())) {
            if (convInnerFunc->parameterTypes().size() ==
                    destInnerFunc->parameterTypes().size() &&
                convInnerFunc->isVariadic() == destInnerFunc->isVariadic()) {
              bool paramsMatch = true;
              for (size_t i = 0; i < convInnerFunc->parameterTypes().size();
                   ++i) {
                if (!control()->is_same(convInnerFunc->parameterTypes()[i],
                                        destInnerFunc->parameterTypes()[i])) {
                  paramsMatch = false;
                  break;
                }
              }
              if (paramsMatch) {
                auto cast = ImplicitCastExpressionAST::create(arena());
                cast->castKind = ImplicitCastKind::kUserDefinedConversion;
                cast->expression = expr;
                cast->type = destinationType;
                cast->valueCategory = ValueCategory::kPrValue;
                expr = cast;
                return true;
              }
            }
          }
        }
      }
    }
  }

  auto destUnqual = control()->remove_cv(destinationType);
  if (auto destClassType = type_cast<ClassType>(destUnqual)) {
    if (auto destClass = destClassType->symbol()) {
      for (auto ctor : destClass->convertingConstructors()) {
        auto funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (params.size() != 1) continue;

        auto paramType = params[0];
        auto paramUnref = control()->remove_reference(paramType);
        auto paramUnqual = control()->remove_cv(paramUnref);

        auto srcUnqual = control()->remove_cv(expr->type);
        if (control()->is_same(srcUnqual, paramUnqual) ||
            (control()->is_arithmetic(srcUnqual) &&
             control()->is_arithmetic(paramUnqual))) {
          auto cast = ImplicitCastExpressionAST::create(arena());
          cast->castKind = ImplicitCastKind::kUserDefinedConversion;
          cast->expression = expr;
          cast->type = destinationType;
          cast->valueCategory = ValueCategory::kPrValue;
          expr = cast;
          return true;
        }
      }
    }
  }

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

  auto savedValueCategory = expr->valueCategory;
  auto savedExpr = expr;
  auto didConvert = ensure_prvalue(expr);

  adjust_cv(expr);

  if (control()->is_same(expr->type, destinationType)) return true;
  if (integral_promotion(expr, destinationType)) return true;
  if (floating_point_promotion(expr, destinationType)) return true;
  if (integral_conversion(expr, destinationType)) return true;
  if (floating_point_conversion(expr, destinationType)) return true;
  if (floating_integral_conversion(expr, destinationType)) return true;
  if (pointer_conversion(expr, destinationType)) return true;
  if (pointer_to_member_conversion(expr, destinationType)) return true;
  if (boolean_conversion(expr, destinationType)) return true;
  if (function_pointer_conversion(expr, destinationType)) return true;
  if (qualification_conversion(expr, destinationType)) return true;
  if (user_defined_conversion(expr, destinationType)) return true;

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
      (void)floating_point_conversion(other, control()->getLongDoubleType());
      return control()->getLongDoubleType();
    }

    if (expr->type->kind() == TypeKind::kDouble ||
        other->type->kind() == TypeKind::kDouble) {
      (void)floating_point_conversion(expr, control()->getDoubleType());
      (void)floating_point_conversion(other, control()->getDoubleType());
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

void TypeChecker::check(DeclarationAST* ast) {
  if (!ast) return;

  if (auto staticAssert = ast_cast<StaticAssertDeclarationAST>(ast)) {
    Visitor{*this}.check_static_assert(staticAssert);
    return;
  }
}

void TypeChecker::check_init_declarator(InitDeclaratorAST* ast) {
  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  var->setInitializer(ast->initializer);

  deduce_array_size(var);
  deduce_auto_type(var);

  if (var->isConstexpr()) {
    var->setType(unit_->control()->add_const(var->type()));
  }

  check_initialization(var, ast);

  if (var->initializer()) {
    auto interp = ASTInterpreter{unit_};
    auto value = interp.evaluate(var->initializer());

    if (!value.has_value() && var->isConstexpr()) {
      if (auto* classType =
              type_cast<ClassType>(unit_->control()->remove_cv(var->type()))) {
        auto* classSym = classType->symbol();
        if (classSym) {
          std::vector<ConstValue> args;
          bool argsOk = true;
          if (auto* parenInit =
                  ast_cast<ParenInitializerAST>(var->initializer())) {
            for (auto node : ListView{parenInit->expressionList}) {
              auto argVal = interp.evaluate(node);
              if (!argVal) {
                argsOk = false;
                break;
              }
              args.push_back(std::move(*argVal));
            }
          } else if (auto* bracedInit =
                         ast_cast<BracedInitListAST>(var->initializer())) {
            for (auto node : ListView{bracedInit->expressionList}) {
              auto argVal = interp.evaluate(node);
              if (!argVal) {
                argsOk = false;
                break;
              }
              args.push_back(std::move(*argVal));
            }
          }
          if (argsOk) {
            for (auto* ctor : classSym->constructors()) {
              if (ctor->isConstexpr()) {
                value = interp.evaluateConstructor(ctor, classType,
                                                   std::move(args));
                break;
              }
            }
          }
        }
      }
    }

    var->setConstValue(value);
  }

  if (var->isConstexpr() && !var->constValue().has_value()) {
    error(var->location(), "constexpr variable must be initialized");
  }
}

void TypeChecker::deduce_array_size(VariableSymbol* var) {
  auto ty = type_cast<UnboundedArrayType>(var->type());
  if (!ty) return;

  auto initializer = var->initializer();
  if (!initializer) return;

  while (auto cast = ast_cast<ImplicitCastExpressionAST>(initializer)) {
    initializer = cast->expression;
  }

  BracedInitListAST* bracedInitList = nullptr;

  if (auto init = ast_cast<BracedInitListAST>(initializer)) {
    bracedInitList = init;
  } else if (auto init = ast_cast<EqualInitializerAST>(initializer)) {
    bracedInitList = ast_cast<BracedInitListAST>(init->expression);
  }

  if (bracedInitList) {
    const auto count =
        std::ranges::distance(ListView{bracedInitList->expressionList});

    if (count > 0) {
      const auto arrayType =
          unit_->control()->getBoundedArrayType(ty->elementType(), count);

      var->setType(arrayType);
    }

    return;
  }

  ExpressionAST* initExpr = nullptr;
  if (auto init = ast_cast<EqualInitializerAST>(initializer)) {
    initExpr = init->expression;
  }

  if (initExpr) {
    if (auto boundedArray = type_cast<BoundedArrayType>(initExpr->type)) {
      const auto arrayType = unit_->control()->getBoundedArrayType(
          ty->elementType(), boundedArray->size());

      var->setType(arrayType);
    }
  }
}

void TypeChecker::deduce_auto_type(VariableSymbol* var) {
  if (!type_cast<AutoType>(var->type())) return;

  if (!var->initializer()) {
    error(var->location(), "variable with 'auto' type must be initialized");
  } else {
    var->setType(unit_->control()->remove_cvref(var->initializer()->type));
  }
}

void TypeChecker::check_initialization(VariableSymbol* var,
                                       InitDeclaratorAST* ast) {
  auto control = unit_->control();
  if (control->is_reference(var->type())) return;

  auto targetType = control->remove_cv(var->type());

  if (control->is_class(targetType)) {
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

    BracedInitListAST* bracedInitList = nullptr;
    if (ast->initializer) {
      if (auto braced = ast_cast<BracedInitListAST>(ast->initializer)) {
        bracedInitList = braced;
      } else if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
        bracedInitList = ast_cast<BracedInitListAST>(equal->expression);
      }
    }

    if (isAggregate && bracedInitList) {
      check_braced_init_list(targetType, bracedInitList);
      return;
    }

    std::vector<ExpressionAST*> args;
    if (ast->initializer) {
      if (auto paren = ast_cast<ParenInitializerAST>(ast->initializer)) {
        for (auto it = paren->expressionList; it; it = it->next)
          args.push_back(it->value);
      } else if (auto braced = ast_cast<BracedInitListAST>(ast->initializer)) {
        for (auto it = braced->expressionList; it; it = it->next)
          args.push_back(it->value);
      } else if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
        args.push_back(equal->expression);
      }
    }

    std::vector<OverloadCandidate> candidates;

    for (auto ctor : classSymbol->constructors()) {
      if (ctor->canonical() != ctor) continue;

      auto type = type_cast<FunctionType>(ctor->type());
      if (!type) continue;

      if (type->parameterTypes().size() != args.size()) continue;

      OverloadCandidate cand{ctor};
      cand.viable = true;

      auto paramIt = type->parameterTypes().begin();
      for (auto arg : args) {
        auto paramType = *paramIt++;
        auto conv = checkImplicitConversion(arg, paramType);
        if (conv.rank == ConversionRank::kNone) {
          cand.viable = false;
          break;
        }
        cand.conversions.push_back(conv);
      }

      if (cand.viable) candidates.push_back(cand);
    }

    auto [bestPtr, ambiguous] = selectBestCandidate(candidates);

    if (!bestPtr) {
      // error(var->location(), "no matching constructor for initialization");
      return;
    } else if (ambiguous) {
      error(var->location(), "constructor call is ambiguous");
      return;
    }

    var->setConstructor(bestPtr->symbol);

    if (ast->initializer) {
      if (auto paren = ast_cast<ParenInitializerAST>(ast->initializer)) {
        size_t i = 0;
        for (auto it = paren->expressionList; it; it = it->next, ++i)
          applyImplicitConversion(bestPtr->conversions[i], it->value);
      } else if (auto braced = ast_cast<BracedInitListAST>(ast->initializer)) {
        size_t i = 0;
        for (auto it = braced->expressionList; it; it = it->next, ++i)
          applyImplicitConversion(bestPtr->conversions[i], it->value);
      } else if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
        applyImplicitConversion(bestPtr->conversions[0], equal->expression);
      }
    }
    return;
  }

  if (!ast->initializer) return;

  BracedInitListAST* bracedInitList = nullptr;

  if (auto init = ast_cast<BracedInitListAST>(ast->initializer)) {
    bracedInitList = init;
  } else if (auto init = ast_cast<EqualInitializerAST>(ast->initializer)) {
    bracedInitList = ast_cast<BracedInitListAST>(init->expression);
  }

  if (bracedInitList) {
    check_braced_init_list(targetType, bracedInitList);
  } else {
    (void)implicit_conversion(ast->initializer, targetType);
    var->setInitializer(ast->initializer);
  }
}

void TypeChecker::check_mem_initializers(
    CompoundStatementFunctionBodyAST* ast) {
  auto functionSymbol = symbol_cast<FunctionSymbol>(scope_);
  if (!functionSymbol) return;

  if (!functionSymbol->isConstructor()) return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
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

    if (auto symbol = Lookup{scope_}(name)) {
      if (auto type = symbol->type()) {
        type = control->remove_cv(type);
        if (auto classType = type_cast<ClassType>(type)) {
          for (auto base : classSymbol->baseClasses()) {
            if (base->symbol() &&
                control->is_same(base->symbol()->type(), classType)) {
              member = base;
              break;
            }
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

    if (control->is_class(targetType)) {
      auto classType = type_cast<ClassType>(control->remove_cv(targetType));
      if (!classType) continue;
      auto targetClassSymbol = classType->symbol();
      if (!targetClassSymbol) continue;

      std::vector<OverloadCandidate> candidates;

      for (auto ctor : targetClassSymbol->constructors()) {
        if (ctor->canonical() != ctor) continue;

        auto type = type_cast<FunctionType>(ctor->type());
        if (!type || type->parameterTypes().size() != args.size()) continue;

        OverloadCandidate cand{ctor};
        cand.viable = true;
        auto paramIt = type->parameterTypes().begin();
        for (auto arg : args) {
          auto paramType = *paramIt++;
          auto conv = checkImplicitConversion(*arg, paramType);
          if (conv.rank == ConversionRank::kNone) {
            cand.viable = false;
            break;
          }
          cand.conversions.push_back(conv);
        }
        if (cand.viable) candidates.push_back(cand);
      }

      auto [bestPtr, ambiguous] = selectBestCandidate(candidates);

      if (!bestPtr) {
        error(memInit->firstSourceLocation(), "no matching constructor");
        continue;
      }

      if (ambiguous) {
        error(memInit->firstSourceLocation(), "constructor call is ambiguous");
        continue;
      }

      memInit->constructor = bestPtr->symbol;

      for (size_t i = 0; i < args.size(); ++i) {
        applyImplicitConversion(bestPtr->conversions[i], *args[i]);
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

auto TypeChecker::is_narrowing_conversion(const Type* from, const Type* to)
    -> bool {
  if (unit_->language() != LanguageKind::kCXX) return false;

  auto control = unit_->control();

  from = control->remove_cv(from);
  to = control->remove_cv(to);

  if (control->is_same(from, to)) return false;

  if (control->is_floating_point(from) && control->is_integral(to)) return true;

  if (control->is_floating_point(from) && control->is_floating_point(to)) {
    auto fromSize = control->memoryLayout()->sizeOf(from);
    auto toSize = control->memoryLayout()->sizeOf(to);
    if (fromSize && toSize && *fromSize > *toSize) return true;
  }

  if (control->is_integral_or_unscoped_enum(from) &&
      control->is_floating_point(to))
    return true;

  if (control->is_integral_or_unscoped_enum(from) && control->is_integral(to)) {
    auto fromSize = control->memoryLayout()->sizeOf(from);
    auto toSize = control->memoryLayout()->sizeOf(to);
    if (fromSize && toSize) {
      if (*fromSize > *toSize) return true;
      if (*fromSize == *toSize &&
          control->is_signed(from) != control->is_signed(to))
        return true;
    }
  }

  return false;
}

void TypeChecker::warn_narrowing(SourceLocation loc, const Type* from,
                                 const Type* to) {
  warning(loc, std::format("narrowing conversion from '{}' to '{}' in "
                           "braced-init-list",
                           to_string(from), to_string(to)));
}

void TypeChecker::check_braced_init_list(const Type* type,
                                         BracedInitListAST* ast) {
  auto control = unit_->control();

  ast->type = type;

  if (control->is_array(type)) {
    // Array initialization
    auto elementType = control->remove_cv(control->get_element_type(type));
    size_t index = 0;
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto boundedArrayType = type_cast<BoundedArrayType>(type)) {
        if (index >= boundedArrayType->size()) {
          error(it->value->firstSourceLocation(),
                "excess elements in array initializer");
          break;
        }
      }

      if (auto nested = ast_cast<BracedInitListAST>(it->value)) {
        check_braced_init_list(elementType, nested);
      } else if (auto desig =
                     ast_cast<DesignatedInitializerClauseAST>(it->value)) {
        check_designated_initializer(elementType, desig);
      } else {
        auto sourceType = it->value->type;
        if (!implicit_conversion(it->value, elementType) ||
            !control->is_same(it->value->type, elementType)) {
          error(
              it->value->firstSourceLocation(),
              std::format("cannot initialize array element of type '{}' with "
                          "expression of type '{}'",
                          to_string(elementType), to_string(it->value->type)));
        } else if (sourceType &&
                   is_narrowing_conversion(sourceType, elementType)) {
          warn_narrowing(it->value->firstSourceLocation(), sourceType,
                         elementType);
        }
      }
      ++index;
    }
  } else if (control->is_class_or_union(type)) {
    // Struct/union aggregate initialization
    auto classType = type_cast<ClassType>(type);
    if (!classType || !classType->symbol()) return;
    auto classSymbol = classType->symbol();

    if (classType->isUnion()) {
      check_union_init(classSymbol, ast);
    } else {
      check_struct_init(classSymbol, ast);
    }
  } else {
    // Scalar initialization
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

    auto sourceType = expr->type;
    if (!implicit_conversion(expr, type)) {
      error(expr->firstSourceLocation(),
            std::format("cannot initialize type '{}' with expression of "
                        "type '{}'",
                        to_string(type), to_string(expr->type)));
    } else if (sourceType && is_narrowing_conversion(sourceType, type)) {
      warn_narrowing(expr->firstSourceLocation(), sourceType, type);
    }
  }
}

void TypeChecker::check_struct_init(ClassSymbol* classSymbol,
                                    BracedInitListAST* ast) {
  auto control = unit_->control();

  std::vector<FieldSymbol*> fields;
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    fields.push_back(field);
  }

  size_t fieldIndex = 0;

  for (auto it = ast->expressionList; it; it = it->next) {
    auto& expr = it->value;

    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(expr)) {
      check_designated_initializer(control->getClassType(classSymbol), desig);
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

    auto fieldType = control->remove_cv(fields[fieldIndex]->type());

    if (auto nested = ast_cast<BracedInitListAST>(expr)) {
      check_braced_init_list(fieldType, nested);
    } else {
      auto sourceType = expr->type;
      if (!implicit_conversion(expr, fieldType)) {
        error(expr->firstSourceLocation(),
              std::format("cannot initialize member '{}' of type '{}' with "
                          "expression of type '{}'",
                          to_string(fields[fieldIndex]->name()),
                          to_string(fieldType), to_string(expr->type)));
      } else if (sourceType && is_narrowing_conversion(sourceType, fieldType)) {
        warn_narrowing(expr->firstSourceLocation(), sourceType, fieldType);
      }
    }

    ++fieldIndex;
  }
}

void TypeChecker::check_union_init(ClassSymbol* classSymbol,
                                   BracedInitListAST* ast) {
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

  FieldSymbol* firstField = nullptr;
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    firstField = field;
    break;
  }

  if (!firstField) {
    error(expr->firstSourceLocation(), "union has no named members");
    return;
  }

  auto fieldType = control->remove_cv(firstField->type());

  if (auto nested = ast_cast<BracedInitListAST>(expr)) {
    check_braced_init_list(fieldType, nested);
  } else {
    auto sourceType = expr->type;
    if (!implicit_conversion(expr, fieldType)) {
      error(expr->firstSourceLocation(),
            std::format("cannot initialize member '{}' of type '{}' with "
                        "expression of type '{}'",
                        to_string(firstField->name()), to_string(fieldType),
                        to_string(expr->type)));
    } else if (sourceType && is_narrowing_conversion(sourceType, fieldType)) {
      warn_narrowing(expr->firstSourceLocation(), sourceType, fieldType);
    }
  }

  if (it->next) {
    error(it->next->value->firstSourceLocation(),
          "excess elements in union initializer");
  }
}

void TypeChecker::check_designated_initializer(
    const Type* currentType, DesignatedInitializerClauseAST* ast) {
  auto control = unit_->control();

  const Type* targetType = currentType;

  for (auto desigIt = ast->designatorList; desigIt; desigIt = desigIt->next) {
    auto designator = desigIt->value;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      auto classType = type_cast<ClassType>(control->remove_cv(targetType));
      if (!classType || !classType->symbol()) {
        error(dot->firstSourceLocation(),
              std::format("member designator on non-aggregate type '{}'",
                          to_string(targetType)));
        return;
      }

      auto member =
          Lookup{scope_}.qualifiedLookup(classType->symbol(), dot->identifier);
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
      targetType = control->remove_cv(field->type());
    } else if (auto subscript = ast_cast<SubscriptDesignatorAST>(designator)) {
      if (!control->is_array(targetType)) {
        error(subscript->firstSourceLocation(),
              std::format("array designator on non-array type '{}'",
                          to_string(targetType)));
        return;
      }
      targetType = control->remove_cv(control->get_element_type(targetType));
    }
  }

  if (!ast->initializer) return;

  ExpressionAST* initExpr = nullptr;
  if (auto equal = ast_cast<EqualInitializerAST>(ast->initializer)) {
    initExpr = equal->expression;
  } else {
    initExpr = ast->initializer;
  }

  if (!initExpr) return;

  if (auto nested = ast_cast<BracedInitListAST>(initExpr)) {
    check_braced_init_list(targetType, nested);
  } else {
    auto sourceType = initExpr->type;
    if (!implicit_conversion(initExpr, targetType)) {
      error(initExpr->firstSourceLocation(),
            std::format("cannot initialize type '{}' with expression of "
                        "type '{}'",
                        to_string(targetType), to_string(initExpr->type)));
    } else if (sourceType && is_narrowing_conversion(sourceType, targetType)) {
      warn_narrowing(initExpr->firstSourceLocation(), sourceType, targetType);
    }
  }

  ast->type = targetType;
}

void TypeChecker::Visitor::check_static_assert(
    StaticAssertDeclarationAST* ast) {
  auto loc = ast->firstSourceLocation();

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
    if (control()->is_class_or_union(ast->baseExpression->type)) {
      // todo: lookup operator-> in the class
    } else {
      (void)ensure_prvalue(ast->baseExpression);
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

  auto classSymbol = classType->symbol();

  auto symbol = Lookup{scope()}.qualifiedLookup(classSymbol, memberName);

  ast->symbol = symbol;

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
    auto pointerType = as_pointer(objectType);
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

void TypeChecker::check_return_statement(ReturnStatementAST* ast) {
  const Type* targetType = nullptr;
  for (auto current = scope_; current; current = current->parent()) {
    if (!current) continue;
    if (current->isFunction() || current->isLambda()) {
      if (auto functionType = type_cast<FunctionType>(current->type())) {
        targetType = functionType->returnType();
      }
    }
  }

  if (!targetType) return;

  auto seq = checkImplicitConversion(ast->expression, targetType);
  applyImplicitConversion(seq, ast->expression);
}

auto TypeChecker::implicit_conversion(ExpressionAST*& yyast,
                                      const Type* targetType) -> bool {
  Visitor visitor{*this};
  return visitor.implicit_conversion(yyast, targetType);
}

void TypeChecker::check_bool_condition(ExpressionAST*& expr) {
  Visitor visitor{*this};
  (void)visitor.implicit_conversion(expr, unit_->control()->getBoolType());
}

void TypeChecker::check_integral_condition(ExpressionAST*& expr) {
  auto control = unit_->control();
  if (!control->is_integral(expr->type) && !control->is_enum(expr->type))
    return;
  Visitor visitor{*this};
  (void)visitor.lvalue_to_rvalue_conversion(expr);
  visitor.adjust_cv(expr);
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
  ImplicitConversionSequence seq;
  if (!expr || !targetType) return seq;

  auto* control = unit_->control();

  const Type* currentType = expr->type;
  ValueCategory currentValCat = expr->valueCategory;

  auto addStep = [&](ImplicitConversionKind kind, const Type* type) {
    seq.steps.push_back({kind, type});
  };

  if (control->is_reference(targetType)) {
    if (auto rvalRef = type_cast<RvalueReferenceType>(targetType)) {
      if (currentValCat == ValueCategory::kLValue) {
        return seq;
      }
      seq.bindsToRvalueRef = true;
    }
    if (auto lvalRef = type_cast<LvalueReferenceType>(targetType)) {
      auto inner = lvalRef->elementType();
      bool isConst = false;
      if (auto qual = type_cast<QualType>(inner)) {
        isConst = qual->isConst();
      }
      if (!isConst && currentValCat != ValueCategory::kLValue) {
        return seq;  // kNone — non-const lvalue ref can't bind rvalue
      }
    }
  }

  if (control->is_array(control->remove_reference(currentType))) {
    auto unref = control->remove_reference(currentType);
    currentType = control->add_pointer(control->remove_extent(unref));
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitConversionKind::kArrayToPointer, currentType);
  } else if (control->is_function(control->remove_reference(currentType))) {
    auto unref = control->remove_reference(currentType);
    currentType = control->add_pointer(unref);
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitConversionKind::kFunctionToPointer, currentType);
  } else if (currentValCat != ValueCategory::kPrValue &&
             !control->is_reference(targetType)) {
    currentType = control->remove_reference(currentType);
    currentValCat = ValueCategory::kPrValue;
    addStep(ImplicitConversionKind::kLValueToRValue, currentType);
  }

  auto comparisonTargetType = control->remove_reference(targetType);

  auto unqualFrom = control->remove_cv(currentType);
  auto unqualTo = control->remove_cv(comparisonTargetType);

  if (control->is_same(unqualFrom, unqualTo)) {
    seq.rank = ConversionRank::kExactMatch;
    // todo: check cv qualifiers
    addStep(ImplicitConversionKind::kIdentity, comparisonTargetType);
    return seq;
  }

  // nullptr to pointer
  if (control->is_null_pointer(unqualFrom) && control->is_pointer(unqualTo)) {
    seq.rank = ConversionRank::kExactMatch;
    addStep(ImplicitConversionKind::kPointerConversion, targetType);
    return seq;
  }

  if (control->is_integral(unqualFrom) && control->is_pointer(unqualTo)) {
    if (auto* intLit = ast_cast<IntLiteralExpressionAST>(expr)) {
      if (intLit->literal && intLit->literal->components().value == 0) {
        seq.rank = ConversionRank::kConversion;
        addStep(ImplicitConversionKind::kPointerConversion, targetType);
        return seq;
      }
    }
  }

  // Pointer conversions
  if (control->is_pointer(unqualFrom) && control->is_pointer(unqualTo)) {
    auto fromPtr = as_pointer(unqualFrom);
    auto toPtr = as_pointer(unqualTo);

    auto fromPointee = fromPtr->elementType();
    auto toPointee = toPtr->elementType();

    auto fromCv = control->get_cv_qualifiers(fromPointee);
    auto toCv = control->get_cv_qualifiers(toPointee);

    if ((static_cast<int>(fromCv) & ~static_cast<int>(toCv)) == 0) {
      auto fromUnqual = control->remove_cv(fromPointee);
      auto toUnqual = control->remove_cv(toPointee);

      if (control->is_same(fromUnqual, toUnqual)) {
        seq.rank = ConversionRank::kExactMatch;
        addStep(ImplicitConversionKind::kQualificationConversion, targetType);
        return seq;
      }

      if (control->is_void(toUnqual)) {
        seq.rank = ConversionRank::kConversion;
        addStep(ImplicitConversionKind::kPointerConversion, targetType);
        return seq;
      }

      if (control->is_class(fromUnqual) && control->is_class(toUnqual)) {
        if (control->is_base_of(toUnqual, fromUnqual)) {
          seq.rank = ConversionRank::kConversion;
          addStep(ImplicitConversionKind::kPointerConversion, targetType);
          return seq;
        }
      }
    }
  }

  // Arithmetic conversions
  if (control->is_arithmetic(unqualFrom) && control->is_arithmetic(unqualTo)) {
    seq.rank = ConversionRank::kConversion;

    if (control->is_integral(unqualFrom) && control->is_integral(unqualTo)) {
      addStep(ImplicitConversionKind::kIntegralConversion, targetType);
      return seq;
    }

    if (control->is_floating_point(unqualFrom) &&
        control->is_floating_point(unqualTo)) {
      addStep(ImplicitConversionKind::kFloatingPointConversion, targetType);
      return seq;
    }

    addStep(ImplicitConversionKind::kFloatingIntegralConversion, targetType);
    return seq;
  }

  // Boolean conversions (pointer-to-bool, arithmetic-to-bool, etc.)
  if (control->is_same(unqualTo, control->getBoolType())) {
    seq.rank = ConversionRank::kConversion;
    addStep(ImplicitConversionKind::kBooleanConversion, targetType);
    return seq;
  }

  // User-defined conversion

  auto makeUserDefinedSeq =
      [&](FunctionSymbol* func,
          ConversionRank s2Rank) -> ImplicitConversionSequence {
    ImplicitConversionSequence uds;
    uds.kind = ConversionSequenceKind::kUserDefined;
    uds.rank = ConversionRank::kConversion;  // user-defined always ranks below
                                             // standard
    uds.userDefinedConversionFunction = func;
    uds.secondStandardConversionRank = s2Rank;
    uds.steps.push_back(
        {ImplicitConversionKind::kUserDefinedConversion, comparisonTargetType});
    return uds;
  };

  ImplicitConversionSequence bestUserDefined;

  if (auto destClassType = type_cast<ClassType>(unqualTo)) {
    if (auto destClass = destClassType->symbol()) {
      for (auto ctor : destClass->convertingConstructors()) {
        auto funcType = type_cast<FunctionType>(ctor->type());
        if (!funcType) continue;
        auto& params = funcType->parameterTypes();
        if (params.size() != 1) continue;

        auto paramType = params[0];

        auto paramUnref = control->remove_reference(paramType);
        auto paramUnqual = control->remove_cv(paramUnref);

        bool viable = false;
        ConversionRank s2Rank = ConversionRank::kNone;

        if (control->is_same(unqualFrom, paramUnqual)) {
          viable = true;
          s2Rank = ConversionRank::kExactMatch;
        } else if (control->is_arithmetic(unqualFrom) &&
                   control->is_arithmetic(paramUnqual)) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        } else if (control->is_pointer(unqualFrom) &&
                   control->is_pointer(paramUnqual)) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        } else if (control->is_null_pointer(unqualFrom) &&
                   control->is_pointer(paramUnqual)) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        }

        if (!viable) continue;

        auto uds = makeUserDefinedSeq(ctor, s2Rank);
        if (!bestUserDefined || uds.isBetterThan(bestUserDefined)) {
          bestUserDefined = uds;
        }
      }
    }
  }

  if (auto srcClassType = type_cast<ClassType>(unqualFrom)) {
    if (auto srcClass = srcClassType->symbol()) {
      for (auto convFunc : srcClass->conversionFunctions()) {
        auto convFuncType = type_cast<FunctionType>(convFunc->type());
        if (!convFuncType) continue;

        auto returnType = convFuncType->returnType();
        if (!returnType) continue;

        auto retUnqual = control->remove_cv(returnType);

        bool viable = false;
        ConversionRank s2Rank = ConversionRank::kNone;

        if (control->is_same(retUnqual, unqualTo)) {
          viable = true;
          s2Rank = ConversionRank::kExactMatch;
        } else if (control->is_arithmetic(retUnqual) &&
                   control->is_arithmetic(unqualTo)) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        } else if (control->is_pointer(retUnqual) &&
                   control->is_pointer(unqualTo)) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        } else if (control->is_same(unqualTo, control->getBoolType())) {
          viable = true;
          s2Rank = ConversionRank::kConversion;
        }

        if (!viable) continue;

        auto uds = makeUserDefinedSeq(convFunc, s2Rank);
        if (!bestUserDefined || uds.isBetterThan(bestUserDefined)) {
          bestUserDefined = uds;
        }
      }
    }
  }

  if (bestUserDefined) return bestUserDefined;

  return seq;
}

void TypeChecker::applyImplicitConversion(
    const ImplicitConversionSequence& sequence, ExpressionAST*& expr) {
  if (sequence.rank == ConversionRank::kNone) return;

  for (const auto& step : sequence.steps) {
    switch (step.kind) {
      case ImplicitConversionKind::kIdentity:
        break;  // Nothing to do (or maybe add cast for type adjustment?)
      case ImplicitConversionKind::kLValueToRValue: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kArrayToPointer: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kArrayToPointerConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kFunctionToPointer: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kIntegralConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kIntegralConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kFloatingPointConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kFloatingPointConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kFloatingIntegralConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kFloatingIntegralConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kPointerConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kPointerConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kBooleanConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind =
            ImplicitCastKind::kIntegralConversion;  // Use integral conversion
                                                    // for bool
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kQualificationConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kQualificationConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      case ImplicitConversionKind::kUserDefinedConversion: {
        auto cast = ImplicitCastExpressionAST::create(unit_->arena());
        cast->castKind = ImplicitCastKind::kUserDefinedConversion;
        cast->expression = expr;
        cast->type = step.type;
        cast->valueCategory = ValueCategory::kPrValue;
        expr = cast;
        break;
      }
      default:
        break;
    }
  }
}

auto TypeChecker::findOverloads(ScopeSymbol* scope, const Name* name) const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;

  if (!scope || !name) return result;

  auto symbol = Lookup{scope}.qualifiedLookup(scope, name);
  if (!symbol) return result;

  if (auto funcSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    result.push_back(funcSymbol);
    return result;
  }

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    for (auto func : overloadSet->functions()) {
      // Only include canonical symbols (skip redeclarations)
      if (func->canonical() == func) {
        result.push_back(func);
      }
    }
  }

  return result;
}

auto TypeChecker::selectBestOverload(
    const std::vector<FunctionSymbol*>& candidates, const Type* leftType,
    const Type* rightType) const -> FunctionSymbol* {
  auto control = unit_->control();

  if (candidates.empty()) return nullptr;

  auto remove_cvref = [&](const Type* type) {
    if (!type) return type;
    return control->remove_cvref(type);
  };

  enum Rank { kExactMatch = 0, kPromotion = 1, kConversion = 2, kNoMatch = 3 };

  auto getRank = [&](const Type* source, const Type* target) -> Rank {
    if (!source || !target) return source == target ? kExactMatch : kNoMatch;

    auto s = remove_cvref(source);
    auto t = remove_cvref(target);

    // Exact match after removing cv-ref from both
    if (control->is_same(s, t)) {
      return kExactMatch;
    }

    // Decay source (array-to-pointer, function-to-pointer)
    auto decayedSource = control->decay(source);
    if (control->is_same(decayedSource, t)) {
      return kExactMatch;
    }

    if (control->is_arithmetic(s) && control->is_arithmetic(t)) {
      if (control->is_same(t, control->getBoolType()) &&
          !control->is_same(s, control->getBoolType()))
        return kConversion;
      if (control->is_integral(s) && control->is_integral(t)) {
        return kConversion;
      }
      return kConversion;
    }

    // Pointer to const void* conversion
    if (control->is_pointer(t)) {
      auto targetElem = control->remove_pointer(t);
      if (control->is_void(targetElem)) {
        if (control->is_pointer(decayedSource) ||
            control->is_null_pointer(source)) {
          return kConversion;
        }
      }
    }

    // Nullptr to pointer
    if (control->is_null_pointer(source) && control->is_pointer(t)) {
      return kExactMatch;
    }

    return kNoMatch;
  };

  struct ViableCandidate {
    FunctionSymbol* symbol;
    Rank leftRank;
    Rank rightRank;

    auto totalRank() const -> int { return (int)leftRank + (int)rightRank; }
  };

  std::vector<ViableCandidate> viable;

  for (auto candidate : candidates) {
    auto funcType = type_cast<FunctionType>(candidate->type());
    if (!funcType) continue;

    auto params = funcType->parameterTypes();
    bool isMember = candidate->parent() && candidate->parent()->isClass();

    Rank leftRank = kNoMatch;
    Rank rightRank = kExactMatch;

    if (rightType) {
      // Binary operator
      if (isMember) {
        if (params.size() != 1) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !control->is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        leftRank = kExactMatch;
        rightRank = getRank(rightType, params[0]);
      } else {
        // Non-member binary: 2 parameters
        if (params.size() != 2) continue;
        leftRank = getRank(leftType, params[0]);
        rightRank = getRank(rightType, params[1]);
      }
    } else {
      // Unary operator
      if (isMember) {
        if (!params.empty()) continue;
        auto classType =
            type_cast<ClassType>(remove_cvref(candidate->parent()->type()));
        if (!classType ||
            !control->is_base_of(classType, remove_cvref(leftType))) {
          continue;
        }
        leftRank = kExactMatch;
      } else {
        if (params.size() != 1) continue;
        leftRank = getRank(leftType, params[0]);
      }
    }

    if (leftRank != kNoMatch && rightRank != kNoMatch) {
      viable.push_back({candidate, leftRank, rightRank});
    }
  }

  if (viable.empty()) return nullptr;

  auto best = &viable[0];
  for (size_t i = 1; i < viable.size(); ++i) {
    if (viable[i].totalRank() < best->totalRank()) {
      best = &viable[i];
    }
  }

  return best->symbol;
}

auto TypeChecker::lookupOperator(const Type* type, TokenKind op,
                                 const Type* rightType) -> FunctionSymbol* {
  auto control = unit_->control();

  auto name = control->getOperatorId(op);
  if (!name) return nullptr;

  if (auto classType = type_cast<ClassType>(control->remove_cvref(type))) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return nullptr;

    auto candidates = findOverloads(classSymbol, name);
    if (!candidates.empty()) {
      return selectBestOverload(candidates, type, rightType);
    }
  }

  if (scope_) {
    auto symbol = Lookup{scope_}(name);

    if (auto funcSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      std::vector<FunctionSymbol*> candidates = {funcSymbol};
      return selectBestOverload(candidates, type, rightType);
    }

    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
      std::vector<FunctionSymbol*> candidates;
      for (auto func : overloadSet->functions()) {
        candidates.push_back(func);
      }

      return selectBestOverload(candidates, type, rightType);
    }
  }

  if (scope_) {
    std::vector<const Type*> argTypes;
    argTypes.push_back(type);
    if (rightType) argTypes.push_back(rightType);

    auto adlCandidates = Lookup{scope_}.argumentDependentLookup(name, argTypes);

    if (!adlCandidates.empty()) {
      return selectBestOverload(adlCandidates, type, rightType);
    }
  }

  return nullptr;
}

auto TypeChecker::as_pointer(const Type* type) const -> const PointerType* {
  return type_cast<PointerType>(unit_->control()->remove_cv(type));
}

auto TypeChecker::as_class(const Type* type) const -> const ClassType* {
  return type_cast<ClassType>(unit_->control()->remove_cv(type));
}

}  // namespace cxx
