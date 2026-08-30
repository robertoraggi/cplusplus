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
#include <cxx/initialization.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <vector>

namespace cxx {
class Arena;
class Control;
class TranslationUnit;

enum class ClassAdjustment { kNone, kDerivedToBase, kBaseToDerived };

enum class ConversionContext { kImplicit, kStandardOnly };

struct AggregateInitializerSlot {
  ExpressionAST* initializer = nullptr;
  const Type* elementType = nullptr;
};

class StandardConversion {
 public:
  explicit StandardConversion(TranslationUnit* unit, bool isC = false);

  [[nodiscard]] auto computeConversionSequence(
      ExpressionAST* expr, const Type* targetType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization,
      ConversionContext context = ConversionContext::kImplicit)
      -> ImplicitConversionSequence;

  void applyConversionSequence(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  [[nodiscard]] auto convertImplicitly(
      ExpressionAST*& expr, const Type* destinationType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization) -> bool;

  void prepareOperand(ExpressionAST*& expr);
  void promoteOperand(ExpressionAST*& expr);
  void decayOperand(ExpressionAST*& expr);

  [[nodiscard]] auto temporaryMaterialization(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto usualArithmeticConversion(ExpressionAST*& expr,
                                               ExpressionAST*& other)
      -> const Type*;

  [[nodiscard]] auto commonArithmeticType(const Type* a, const Type* b)
      -> const Type*;

  [[nodiscard]] auto compositePointerType(ExpressionAST*& expr,
                                          ExpressionAST*& other) -> const Type*;

  [[nodiscard]] auto classAdjustment(const Type* sourceType,
                                     const Type* targetType) const
      -> ClassAdjustment;

  [[nodiscard]] auto pointerConversionCastKind(const Type* sourceType,
                                               const Type* targetType) const
      -> ImplicitCastKind;

  auto convertToBaseClass(ExpressionAST*& expr, const Type* baseType) -> bool;

  auto convertToDerivedClass(ExpressionAST*& expr, const Type* derivedType)
      -> bool;

  void convertPointer(ExpressionAST*& expr, const Type* targetType);

  [[nodiscard]] auto convertClassOperandForBuiltinOperator(ExpressionAST*& expr)
      -> bool;

  [[nodiscard]] auto isNullPointerConstant(ExpressionAST* expr) const -> bool;

  void foldConstantRead(ExpressionAST*& expression);

  void appendDefaultArguments(FunctionSymbol* function,
                              List<ExpressionAST*>** list);

  void recordConversionFunction(ImplicitCastExpressionAST* cast,
                                const ImplicitConversionSequence& sequence);

 private:
  void applyStep(const ImplicitConversionSequence& sequence,
                 const ImplicitConversionSequence::Step& step,
                 ExpressionAST*& expr);

  void applyCopyConstruction(const ImplicitConversionSequence& sequence,
                             ExpressionAST*& expr);

  void appendTemporaryMaterialization(ImplicitConversionSequence& sequence);

  void recordUserDefinedConversion(ImplicitCastExpressionAST* cast,
                                   FunctionSymbol* function);

  [[nodiscard]] auto selectCopyConstructor(ExpressionAST* expr,
                                           const Type* destinationType)
      -> FunctionSymbol*;

  [[nodiscard]] auto listInitializationSequence(
      BracedInitListAST* bracedInitList, const Type* targetType,
      InitializationKind initializationKind) -> ImplicitConversionSequence;

  [[nodiscard]] auto isDesignatedInitializerList(
      BracedInitListAST* bracedInitList) const -> bool;

  [[nodiscard]] auto singleListElement(BracedInitListAST* bracedInitList) const
      -> ExpressionAST*;

  [[nodiscard]] auto initializesCharacterArrayFromStringLiteral(
      BracedInitListAST* bracedInitList, const Type* arrayType) const -> bool;

  [[nodiscard]] auto designatedAggregateSlot(
      const std::vector<Symbol*>& elements,
      DesignatedInitializerClauseAST* designated) const
      -> std::optional<std::size_t>;

  [[nodiscard]] auto aggregateInitializerSlots(
      BracedInitListAST* bracedInitList, ClassSymbol* classSymbol)
      -> std::optional<std::vector<AggregateInitializerSlot>>;

  [[nodiscard]] auto referenceBindingSequence(ExpressionAST* expr,
                                              const Type* targetType)
      -> std::optional<ImplicitConversionSequence>;

  [[nodiscard]] auto directReferenceBindingCastKind(
      const Type* referencedType, const Type* sourceType) const
      -> ImplicitCastKind;

  [[nodiscard]] auto computeConversionSequenceSteps(
      ExpressionAST* expr, const Type* targetType,
      InitializationKind initializationKind, ConversionContext context)
      -> ImplicitConversionSequence;

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  void resolveOverloadSet(ExpressionAST* expr, const Type* targetType);

  void setResolvedFunction(ExpressionAST* expr, FunctionSymbol* function);

  [[nodiscard]] auto pointeeClassAdjustment(const Type* sourceType,
                                            const Type* targetType) const
      -> ClassAdjustment;

  [[nodiscard]] auto ensurePrvalue(ExpressionAST*& expr) -> bool;

  void adjustCv(ExpressionAST* expr);

  [[nodiscard]] auto lvalueToRvalue(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto functionToPointer(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto arrayToPointer(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto integralPromotion(ExpressionAST*& expr,
                                       const Type* destinationType = nullptr)
      -> bool;

  [[nodiscard]] auto floatingPointPromotion(
      ExpressionAST*& expr, const Type* destinationType = nullptr) -> bool;

  [[nodiscard]] auto convertArithmetic(ExpressionAST*& expr,
                                       const Type* destinationType) -> bool;

  [[nodiscard]] auto integralConversion(ExpressionAST*& expr,
                                        const Type* destinationType) -> bool;

  [[nodiscard]] auto floatingPointConversion(ExpressionAST*& expr,
                                             const Type* destinationType)
      -> bool;

  [[nodiscard]] auto floatingIntegralConversion(ExpressionAST*& expr,
                                                const Type* destinationType)
      -> bool;

  void requireDefinitionOfDesignatedField(ExpressionAST* expr);

  void materializeConstructorArguments(ImplicitCastExpressionAST* cast,
                                       FunctionSymbol* constructor);

  [[nodiscard]] auto requiresCopyConstruction(ExpressionAST* expr,
                                              const Type* destinationType) const
      -> bool;

  [[nodiscard]] auto narrowsAggregateElement(BracedInitListAST* bracedInitList,
                                             const Type* targetType) -> bool;

  [[nodiscard]] auto listInitializes(BracedInitListAST* bracedInitList,
                                     const Type* targetType,
                                     InitializationKind initializationKind)
      -> bool;

  [[nodiscard]] static auto isCallableWithOneArgument(FunctionSymbol* ctor)
      -> bool;

  [[nodiscard]] auto instantiateConversionFunctionTemplate(
      FunctionSymbol* convFunc, const Type* targetType, ExpressionAST* expr)
      -> FunctionSymbol*;

  [[nodiscard]] auto hasUniqueNonVirtualBase(const ClassType* derived,
                                             const ClassType* base) -> bool;

  [[nodiscard]] auto compositeVoidPointerType(const Type* left,
                                              const Type* right) -> const Type*;

  [[nodiscard]] auto compositeFunctionPointerType(const Type* left,
                                                  const Type* right)
      -> const Type*;

  [[nodiscard]] auto compositeClassAdjustedType(const Type* type,
                                                const Type* classType)
      -> const Type*;

  [[nodiscard]] auto compositePointerClassType(const Type* left,
                                               const Type* right,
                                               bool contravariant)
      -> const Type*;

  void normalizeCompositePointerClass(const Type*& left, const Type*& right);

  [[nodiscard]] auto isMemberPointeeConvertible(const Type* source,
                                                const Type* target) const
      -> bool;

 private:
  TranslationUnit* unit_;
  TypeTraits traits;
  Control* control_;
  Arena* arena_;
  bool isC_ = false;
};
}  // namespace cxx
