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
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

#include <optional>

namespace cxx {

class TranslationUnit;

class TypeChecker {
 public:
  explicit TypeChecker(TranslationUnit* unit);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto reportErrors() const -> bool { return reportErrors_; }
  void setReportErrors(bool reportErrors) { reportErrors_ = reportErrors; }

  void setScope(ScopeSymbol* scope) { scope_ = scope; }

  void operator()(ExpressionAST* ast);

  void check(ExpressionAST* ast);

  void check(DeclarationAST* ast);

  // todo: remove
  void check_return_statement(ReturnStatementAST* ast);

  void check_bool_condition(ExpressionAST*& ast);
  void check_integral_condition(ExpressionAST*& ast);
  void check_init_declarator(InitDeclaratorAST* initDecl);

  void deduce_array_size(VariableSymbol* var);
  void deduce_auto_type(VariableSymbol* var);
  void check_initialization(VariableSymbol* var, InitDeclaratorAST* ast);
  void check_mem_initializers(CompoundStatementFunctionBodyAST* ast);

  [[nodiscard]] auto collect_init_args(ExpressionAST* initializer)
      -> std::vector<ExpressionAST*>;

  void apply_init_conversions(
      ExpressionAST* initializer,
      const std::vector<ImplicitConversionSequence>& conversions);

  void check_braced_init_list(const Type* type, BracedInitListAST* ast);
  void check_struct_init(ClassSymbol* classSymbol, BracedInitListAST* ast);
  void check_union_init(ClassSymbol* classSymbol, BracedInitListAST* ast);
  void check_designated_initializer(const Type* currentType,
                                    DesignatedInitializerClauseAST* ast);

  [[nodiscard]] auto is_narrowing_conversion(const Type* from, const Type* to)
      -> bool;

  void warn_narrowing(SourceLocation loc, const Type* from, const Type* to);

  void check_element_init(ExpressionAST*& expr, const Type* targetType,
                          std::string errorMessage);

  [[nodiscard]] auto implicit_conversion(ExpressionAST*& expr,
                                         const Type* targetType) -> bool;

  [[nodiscard]] auto checkImplicitConversion(ExpressionAST* expr,
                                             const Type* targetType)
      -> ImplicitConversionSequence;

  void applyImplicitConversion(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  [[nodiscard]] static auto mapImplicitCastKind(ImplicitConversionKind kind)
      -> std::optional<ImplicitCastKind>;

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  [[nodiscard]] auto lookupOperator(const Type* type, TokenKind op,
                                    const Type* rightType = nullptr)
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

  void warning(SourceLocation loc, std::string message);
  void error(SourceLocation loc, std::string message);

  [[nodiscard]] auto as_pointer(const Type* type) const -> const PointerType*;
  [[nodiscard]] auto as_class(const Type* type) const -> const ClassType*;

 private:
  struct Visitor;

  TranslationUnit* unit_;
  ScopeSymbol* scope_ = nullptr;
  bool reportErrors_ = false;
  bool lastOperatorLookupAmbiguous_ = false;
};

}  // namespace cxx
