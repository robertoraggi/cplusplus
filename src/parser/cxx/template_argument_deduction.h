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
#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/type_deduction.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <span>
#include <vector>

namespace cxx {
class Arena;
class Control;
class TranslationUnit;

[[nodiscard]] auto call_deduction_parameter_type(const TypeTraits& traits,
                                                 const Type* P) -> const Type*;

[[nodiscard]] auto call_deduction_argument_type(const TypeTraits& traits,
                                                const Type* P, const Type* A,
                                                bool forwardsLvalue)
    -> const Type*;

class TemplateArgumentDeduction {
 public:
  explicit TemplateArgumentDeduction(TranslationUnit* unit);

  [[nodiscard]] auto deduce(FunctionSymbol* func, List<ExpressionAST*>* args,
                            List<TemplateArgumentAST*>* explicitTemplateArgs)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceForGuide(TemplateDeclarationAST* templateDecl,
                                    const FunctionType* functionType,
                                    List<ExpressionAST*>* args)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceFromTargetType(
      FunctionSymbol* func, const FunctionType* targetType,
      List<TemplateArgumentAST*>* explicitTemplateArgs = nullptr,
      bool matchReturnType = true)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceFromConversionTarget(FunctionSymbol* func,
                                                const Type* targetType)
      -> std::optional<List<TemplateArgumentAST*>*>;

 private:
  void begin(TemplateDeclarationAST* templateDecl);

  [[nodiscard]] auto specifyExplicitArguments(
      List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool;

  [[nodiscard]] auto isForwardingReference(const Type* P) const -> bool;

  [[nodiscard]] auto bindsLvalueToForwardingReference(
      const Type* P, ExpressionAST* argument) const -> bool;

  [[nodiscard]] auto callArgumentType(const Type* P, const Type* A,
                                      ExpressionAST* argument) const
      -> const Type*;

  [[nodiscard]] auto deduceFromCall(const FunctionType* functionType,
                                    List<ExpressionAST*>* args) -> bool;

  [[nodiscard]] auto deduceFromFunctionParameterPack(const Type* P,
                                                     List<ExpressionAST*>* args)
      -> bool;

  [[nodiscard]] auto deduceFromCallArgument(const Type* P,
                                            ExpressionAST* argument) -> bool;

  [[nodiscard]] auto deduceFromInitializerList(const Type* P,
                                               BracedInitListAST* list) -> bool;

  [[nodiscard]] auto deduceFromOverloadSet(const Type* P,
                                           const OverloadSetType* A,
                                           bool takesAddress) -> bool;

  [[nodiscard]] auto deduceFromArgumentType(const Type* P, const Type* A)
      -> bool;

  [[nodiscard]] auto baseClassesOf(ClassSymbol* classSymbol) const
      -> std::vector<ClassSymbol*>;

  [[nodiscard]] auto deduceFromBaseClass(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto deduceFromTypes(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto deducedArguments()
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deducedArgument(int slot,
                                     List<TemplateArgumentAST*>* argumentsSoFar)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto collectDeducedSoFar(
      List<TemplateArgumentAST*>* argumentsSoFar)
      -> std::optional<std::vector<TemplateArgument>>;

  [[nodiscard]] auto nonTypeParameterType(int slot) const -> const Type*;

  [[nodiscard]] auto valueSymbol(Symbol* value, const Type* valueType) const
      -> Symbol*;

  [[nodiscard]] auto symbolArgument(Symbol* symbol, const Type* type) const
      -> TemplateArgumentAST*;

  [[nodiscard]] auto packArgument(int slot) -> TemplateArgumentAST*;

  [[nodiscard]] auto typeArgument(const Type* type) const
      -> TemplateArgumentAST*;

  TranslationUnit* unit_;
  TypeTraits traits;
  Control* control_;
  Arena* arena_;
  TemplateDeclarationAST* templateDecl_ = nullptr;
  std::optional<TypeDeduction> deduction_;
  std::vector<TemplateArgumentAST*> explicitArguments_;
};
}  // namespace cxx
