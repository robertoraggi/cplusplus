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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/implicit_conversion_sequence.h>
#include <cxx/overload_resolution.h>
#include <cxx/source_location.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

#include <unordered_set>

namespace cxx {
class TranslationUnit;

[[nodiscard]] auto isUntypedAfterError(ExpressionAST* expr) -> bool;

void markUntypedAfterError(ExpressionAST* expr);

[[nodiscard]] auto fold_concept_id(TranslationUnit* unit, ExpressionAST* ast)
    -> ExpressionAST*;

class TypeChecker {
 public:
  explicit TypeChecker(TranslationUnit* unit);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;

  [[nodiscard]] auto reportErrors() const -> bool { return reportErrors_; }
  void setReportErrors(bool reportErrors) { reportErrors_ = reportErrors; }

  void setScope(ScopeSymbol* scope) { scope_ = scope; }
  [[nodiscard]] auto scope() const -> ScopeSymbol* { return scope_; }

  void operator()(ExpressionAST* ast);

  void check(ExpressionAST* ast);

  void check(DeclarationAST* ast);

  void check_return_statement(ReturnStatementAST* ast);

  [[nodiscard]] auto isMoveEligibleOperand(ExpressionAST* expr,
                                           ScopeSymbol* functionScope) const
      -> bool;

  void treatMoveEligibleOperandAsRvalue(ExpressionAST*& expr,
                                        ScopeSymbol* functionScope,
                                        const Type* targetType);

  auto check_bool_condition(ExpressionAST*& ast) -> bool;
  void check_integral_condition(ExpressionAST*& ast);
  void check_init_declarator(InitDeclaratorAST* initDecl,
                             SpecifierAST* typeSpecifier);
  void check_condition_declaration(ConditionExpressionAST* ast);
  void check_field_initializer(FieldSymbol* field);
  void check_mem_initializers(CompoundStatementFunctionBodyAST* ast);
  void bind_template_parameter_base_initializers(
      CompoundStatementFunctionBodyAST* ast);
  void check_braced_init_list(const Type* type, BracedInitListAST* ast,
                              InitializationKind initializationKind);
  void append_default_arguments(FunctionSymbol* function,
                                List<ExpressionAST*>** list);

  void checkConstructorAccess(FunctionSymbol* constructor,
                              SourceLocation location);

  [[nodiscard]] auto check_class_initializer(
      const Type* targetType, ExpressionAST*& initializer,
      SourceLocation location, List<ExpressionAST*>** argumentList = nullptr)
      -> FunctionSymbol*;

  [[nodiscard]] auto deduceAutoType(const Type* declaredType,
                                    const Type* initializerType) -> const Type*;

  [[nodiscard]] static auto isPotentiallyThrowing(ExpressionAST* expr) -> bool;

  [[nodiscard]] auto deducePlaceholderType(const Type* declaredType,
                                           ExpressionAST* initializer)
      -> const Type*;

  [[nodiscard]] auto deduceClassTemplateSpecialization(
      SpecifierAST* typeSpecifier, const std::vector<ExpressionAST*>& arguments,
      bool isListInitialization, bool isCopyInitialization,
      SourceLocation location) -> const Type*;

  auto getInitDeclaratorLocation(InitDeclaratorAST* ast,
                                 VariableSymbol* var) const -> SourceLocation;

  [[nodiscard]] auto implicit_conversion(
      ExpressionAST*& expr, const Type* targetType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization) -> bool;

  [[nodiscard]] auto checkImplicitConversion(ExpressionAST* expr,
                                             const Type* targetType)
      -> ImplicitConversionSequence;

  void applyImplicitConversion(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  [[nodiscard]] auto lookupOperator(const Type* type, TokenKind op,
                                    const Type* rightType = nullptr,
                                    ExpressionAST* leftExpr = nullptr,
                                    ExpressionAST* rightExpr = nullptr)
      -> FunctionSymbol*;

  [[nodiscard]] auto trySelectOperator(
      const std::vector<FunctionSymbol*>& candidates, const Type* type,
      const Type* rightType) -> FunctionSymbol*;

  [[nodiscard]] auto collectOverloads(Symbol* symbol) const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto findOverloads(ScopeSymbol* scope, const Name* name) const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto selectBestOverload(
      const std::vector<FunctionSymbol*>& candidates, const Type* type,
      const Type* rightType, bool* ambiguous) const -> FunctionSymbol*;

  [[nodiscard]] auto wasLastOperatorLookupAmbiguous() const -> bool {
    return lastOperatorLookupAmbiguous_;
  }

  [[nodiscard]] auto wasLastOperatorRewritten() const -> bool {
    return lastOperatorRewritten_;
  }

  [[nodiscard]] auto wasLastOperatorReversed() const -> bool {
    return lastOperatorReversed_;
  }

  void warning(SourceLocation loc, std::string message);
  void error(SourceLocation loc, std::string message);
  void note(SourceLocation loc, std::string message);
  void useFunction(FunctionSymbol* function, SourceLocation loc);

  void useConversionFunction(ExpressionAST* expr);

  void requireFunctionDefinition(FunctionSymbol* function);

  [[nodiscard]] auto hasConstantValue(FieldSymbol* field) -> bool;

  [[nodiscard]] auto as_pointer(const Type* type) const -> const PointerType*;
  [[nodiscard]] auto as_class(const Type* type) const -> const ClassType*;

  struct AggregateInitGuard {
    AggregateInitGuard(const AggregateInitGuard&) = delete;
    auto operator=(const AggregateInitGuard&) -> AggregateInitGuard& = delete;

    TypeChecker& checker;
    ClassSymbol* classSymbol;
    bool entered;

    AggregateInitGuard(TypeChecker& checker, ClassSymbol* classSymbol)
        : checker(checker),
          classSymbol(classSymbol),
          entered(
              checker.aggregatesBeingInitialized_.insert(classSymbol).second) {}

    ~AggregateInitGuard() {
      if (entered) checker.aggregatesBeingInitialized_.erase(classSymbol);
    }

    [[nodiscard]] explicit operator bool() const { return entered; }
  };

 private:
  struct Visitor;

  TranslationUnit* unit_ = nullptr;
  ScopeSymbol* scope_ = nullptr;
  bool reportErrors_ = false;
  bool lastOperatorLookupAmbiguous_ = false;
  bool lastOperatorRewritten_ = false;
  bool lastOperatorReversed_ = false;
  std::unordered_set<ClassSymbol*> aggregatesBeingInitialized_;
};
}  // namespace cxx
