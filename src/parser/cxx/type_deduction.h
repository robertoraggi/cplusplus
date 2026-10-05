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
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace cxx {
class Control;
class TranslationUnit;

class TypeDeduction {
 public:
  struct State {
    std::vector<Symbol*> deduced;
    bool skippedNonDeducedContext = false;
  };

  TypeDeduction(TranslationUnit* unit,
                TemplateDeclarationAST* templateDeclaration);

  [[nodiscard]] auto deduce(std::span<const TemplateArgument> P,
                            std::span<const TemplateArgument> A) -> bool;

  [[nodiscard]] auto deduce(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto deduceAllowingConversions(const Type* P, const Type* A)
      -> bool;

  struct Expansion {
    std::vector<int> slots;
    std::vector<Symbol*> previous;
    std::vector<ParameterPackSymbol*> packs;
    std::size_t expandingSlots = 0;
  };

  [[nodiscard]] auto beginExpansion(const TemplateArgument& pattern)
      -> Expansion;
  [[nodiscard]] auto beginElement(const Expansion& expansion, std::size_t index)
      -> bool;
  [[nodiscard]] auto endElement(Expansion& expansion) -> bool;
  [[nodiscard]] auto endExpansion(Expansion& expansion) -> bool;

  [[nodiscard]] auto deduceArrayBound(ExpressionAST* P, std::size_t size)
      -> bool;

  [[nodiscard]] auto vectorSizeIn(VectorSizeKind sizeKind,
                                  const VectorType* vector) const
      -> std::optional<std::size_t>;

  void specify(int slot, Symbol* argument);
  void specifyPackPrefix(int slot, ParameterPackSymbol* prefix);

  [[nodiscard]] auto parameterCount() const -> int {
    return static_cast<int>(parameters_.size());
  }

  [[nodiscard]] auto parameter(int slot) const -> TemplateParameterAST* {
    return parameters_[slot];
  }

  [[nodiscard]] auto deduced(int slot) const -> Symbol* {
    return deduced_[slot];
  }

  [[nodiscard]] auto isSpecified(int slot) const -> bool {
    return specified_[slot];
  }

  [[nodiscard]] auto slotOf(const Type* type) const -> int;

  [[nodiscard]] auto mentionsDeducibleParameter(const Type* type) const -> bool;

  [[nodiscard]] auto mentionedSlots(const Type* type) const -> std::vector<int>;

  [[nodiscard]] auto packSlots(const TemplateArgument& pattern) const
      -> std::vector<int>;

  [[nodiscard]] auto specifiedPackLength(int slot) const -> std::size_t;

  [[nodiscard]] auto state() const -> State;
  void restore(State state);

  [[nodiscard]] auto templateArguments() const
      -> std::optional<std::vector<TemplateArgument>>;

  [[nodiscard]] auto undeducedParameter() const -> TemplateParameterAST*;

  [[nodiscard]] auto skippedNonDeducedContext() const -> bool {
    return skippedNonDeducedContext_;
  }

 private:
  struct DeduceType;

  [[nodiscard]] auto isPackExpansion(const TemplateArgument& argument) const
      -> bool;

  struct CollectSlots;

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto slotOf(int depth, int index) const -> int;
  [[nodiscard]] auto slotOf(Symbol* parameter) const -> int;

  [[nodiscard]] auto record(int slot, Symbol* value) -> bool;
  [[nodiscard]] auto typeArgument(const Type* type) const -> Symbol*;
  [[nodiscard]] auto sizeArgument(std::size_t value) const -> Symbol*;

  [[nodiscard]] auto deduce(const TemplateArgument& P,
                            const TemplateArgument& A) -> bool;

  [[nodiscard]] auto deducePackExpansion(
      const TemplateArgument& pattern,
      std::span<const TemplateArgument> arguments) -> bool;

  [[nodiscard]] auto deducePackElements(
      const TemplateArgument& pattern,
      std::span<const TemplateArgument> arguments, Expansion& expansion)
      -> bool;

  [[nodiscard]] auto deduceTemplateName(int slot, Symbol* templateSymbol)
      -> bool;

  [[nodiscard]] auto deduceClassSpecialization(const ClassType* P,
                                               const Type* A) -> bool;

  [[nodiscard]] auto deduceTemplateParameterSpecialization(
      const TemplateTypeParameterSpecializationType* P, const Type* A) -> bool;

  [[nodiscard]] auto deduceSpecializationArguments(
      std::span<const TemplateArgument> P, Symbol* templateSymbol,
      std::span<const TemplateArgument> A) -> bool;

  [[nodiscard]] auto deduceNonTypeParameter(NonTypeParameterSymbol* parameter,
                                            const TemplateArgument& A) -> bool;

  [[nodiscard]] auto deduceTemplateArgumentValue(
      NonTypeParameterSymbol* parameter, const TemplateArgument& A) -> bool;

  [[nodiscard]] auto holdsSpecifiedArgument(int slot) const -> bool;

  [[nodiscard]] auto matchesSpecifiedArgument(int slot,
                                              const TemplateArgument& A) const
      -> bool;

  [[nodiscard]] auto hasCorrespondingParameterType(
      NonTypeParameterSymbol* parameter, const TemplateArgument& A) const
      -> bool;

  [[nodiscard]] auto deduceValue(const TemplateArgument& P,
                                 const TemplateArgument& A) -> bool;

  [[nodiscard]] auto deduceAtLevel(const Type* P, const Type* A,
                                   bool allowsQualificationConversion) -> bool;

  [[nodiscard]] auto deduceQualifiedType(const QualType* P, const Type* A,
                                         bool allowsQualificationConversion)
      -> bool;

  [[nodiscard]] auto deduceFunctionType(const FunctionType* P, const Type* A,
                                        bool allowsFunctionPointerConversion)
      -> bool;

  [[nodiscard]] auto deduceExceptionSpecification(
      const FunctionType* P, const FunctionType* A,
      bool allowsFunctionPointerConversion) -> bool;

  [[nodiscard]] auto deduceArrayBound(ExpressionAST* P, const Type* A) -> bool;

  [[nodiscard]] auto matchesNonDependentType(
      const Type* P, const Type* A, bool allowsQualificationConversion) const
      -> bool;

  [[nodiscard]] auto skipNonDeducedContext() -> bool;

  TranslationUnit* unit_ = nullptr;
  std::vector<TemplateParameterAST*> parameters_;
  std::vector<Symbol*> deduced_;
  std::vector<bool> specified_;
  std::vector<ParameterPackSymbol*> packPrefixes_;
  std::vector<int> expandingSlots_;
  bool skippedNonDeducedContext_ = false;
};

}  // namespace cxx
