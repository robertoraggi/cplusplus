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
#include <cxx/standard_conversion.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <vector>

namespace cxx {

class Arena;
class Control;
class TranslationUnit;

struct Candidate {
  FunctionSymbol* symbol = nullptr;
  std::vector<ImplicitConversionSequence> conversions;
  bool viable = false;
  bool exactCvMatch = true;
  bool fromTemplate = false;
};

struct OverloadResult {
  Candidate* best = nullptr;
  bool ambiguous = false;
};

struct ConstructorResult {
  std::vector<Candidate> candidates;
  Candidate* best = nullptr;
  bool ambiguous = false;
};

class OverloadResolution {
 public:
  explicit OverloadResolution(TranslationUnit* unit);

  [[nodiscard]] auto computeImplicitConversionSequence(ExpressionAST* expr,
                                                       const Type* targetType)
      -> ImplicitConversionSequence;

  void applyImplicitConversion(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  [[nodiscard]] auto selectBestViableFunction(
      std::vector<Candidate>& candidates, bool useCvTiebreaker = false,
      bool preferNonTemplate = false) -> OverloadResult;

  [[nodiscard]] auto resolveConstructor(ClassSymbol* classSymbol,
                                        const std::vector<ExpressionAST*>& args)
      -> ConstructorResult;

  [[nodiscard]] auto findCandidates(ScopeSymbol* scope, const Name* name) const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto collectCandidates(Symbol* symbol) const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto resolveBinaryOperator(
      const std::vector<FunctionSymbol*>& candidates, const Type* leftType,
      const Type* rightType, bool* ambiguous) const -> FunctionSymbol*;

  [[nodiscard]] auto lookupOperator(const Type* type, TokenKind op,
                                    const Type* rightType = nullptr)
      -> FunctionSymbol*;

  [[nodiscard]] auto wasLastLookupAmbiguous() const -> bool {
    return lastLookupAmbiguous_;
  }

  [[nodiscard]] auto initializerListElementType(const Type* targetType) const
      -> const Type*;

 private:
  [[nodiscard]] auto trySelectOperator(
      const std::vector<FunctionSymbol*>& candidates, const Type* type,
      const Type* rightType) -> FunctionSymbol*;

  TranslationUnit* unit_;
  Control* control_;
  Arena* arena_;
  StandardConversion stdconv_;
  bool lastLookupAmbiguous_ = false;
};

}  // namespace cxx
