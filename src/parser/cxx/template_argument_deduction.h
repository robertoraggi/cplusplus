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
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <vector>

namespace cxx {

class Arena;
class Control;
class TranslationUnit;

struct TemplateParameterInfo {
  enum class Kind {
    kUnknown,
    kType,
    kNonType,
    kTemplate,
    kConstraint,
  };

  const TypeParameterType* typeParameterType = nullptr;
  bool isPack = false;
  bool hasDefault = false;
  Kind kind = Kind::kUnknown;
};

class TemplateArgumentDeduction {
 public:
  explicit TemplateArgumentDeduction(TranslationUnit* unit);

  [[nodiscard]] auto deduce(FunctionSymbol* func, List<ExpressionAST*>* args,
                            List<TemplateArgumentAST*>* explicitTemplateArgs)
      -> std::optional<List<TemplateArgumentAST*>*>;

 private:
  void collectTemplateParameters(TemplateDeclarationAST* templateDecl);

  [[nodiscard]] auto substituteExplicitTemplateArguments(
      List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool;

  [[nodiscard]] auto isExplicitArgumentCompatible(
      const TemplateParameterInfo& info, TemplateArgumentAST* arg) -> bool;

  [[nodiscard]] auto isForwardingReference(const Type* paramType) -> bool;

  [[nodiscard]] auto deduceTypeFromType(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto deduceFromCallArgument(const Type* P, const Type* A,
                                            ExpressionAST* argExpr) -> bool;

  [[nodiscard]] auto deduceFromCall(const FunctionType* functionType,
                                    List<ExpressionAST*>* args) -> bool;

  [[nodiscard]] auto checkDeducedArguments() -> bool;

  [[nodiscard]] auto buildTemplateArgumentList()
      -> std::optional<List<TemplateArgumentAST*>*>;

  TranslationUnit* unit_;
  Control* control_;
  Arena* arena_;

  std::vector<TemplateParameterInfo> templateParams_;
  std::vector<TemplateArgumentAST*> explicitParamArg_;
  std::vector<std::vector<TemplateArgumentAST*>> explicitPackArgs_;
  std::vector<const Type*> deducedTypes_;
  std::vector<std::vector<const Type*>> deducedPacks_;
};

}  // namespace cxx
