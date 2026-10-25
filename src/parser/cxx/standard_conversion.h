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
struct Candidate;
struct ConstructorResult;

enum class ClassAdjustment { kNone, kDerivedToBase, kBaseToDerived };

[[nodiscard]] auto checkBaseClassConversion(TranslationUnit* unit,
                                            ScopeSymbol* accessingScope,
                                            ClassSymbol* derived,
                                            ClassSymbol* base,
                                            SourceLocation loc) -> bool;

enum class ConversionContext { kImplicit, kStandardOnly };

struct AggregateListConversion {
  bool viable = false;
  bool narrows = false;
  std::size_t elementCount = 0;
  ConversionRank elementRank = ConversionRank::kExactMatch;
};

enum class NonAddressableDesignation { kNone, kBitField, kVectorElement };

class StandardConversion {
 public:
  explicit StandardConversion(TranslationUnit* unit, bool isC = false);

  [[nodiscard]] auto computeConversionSequence(
      ExpressionAST* expr, const Type* targetType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization,
      ConversionContext context = ConversionContext::kImplicit)
      -> ImplicitConversionSequence;

  [[nodiscard]] auto conversionFunctionSequence(ExpressionAST* expr,
                                                FunctionSymbol* function)
      -> ImplicitConversionSequence;

  [[nodiscard]] auto isAccessible(const ImplicitConversionSequence& sequence)
      -> bool;

  void applyConversionSequence(const ImplicitConversionSequence& sequence,
                               ExpressionAST*& expr);

  void setAccessingScope(ScopeSymbol* accessingScope) {
    accessingScope_ = accessingScope;
  }

  [[nodiscard]] auto convertImplicitly(
      ExpressionAST*& expr, const Type* destinationType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization) -> bool;

  void atomicToNonAtomic(ExpressionAST*& expr);
  void prepareOperand(ExpressionAST*& expr);
  void bindDesignatedFunction(ExpressionAST* expr);
  void promoteOperand(ExpressionAST*& expr);
  [[nodiscard]] auto arithmeticPromotion(ExpressionAST*& expr) -> bool;
  void decayOperand(ExpressionAST*& expr);
  void discardedValue(ExpressionAST*& expr);
  void copyDiscardedClass(ExpressionAST*& expr);

  [[nodiscard]] auto temporaryMaterialization(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto usualArithmeticConversion(ExpressionAST*& expr,
                                               ExpressionAST*& other)
      -> const Type*;

  [[nodiscard]] auto splatOperand(ExpressionAST*& expr,
                                  const VectorType* vectorType) -> bool;

  [[nodiscard]] auto vectorOperandConversion(ExpressionAST*& expr,
                                             ExpressionAST*& other)
      -> std::optional<const Type*>;

  [[nodiscard]] auto isVectorBooleanConversion(const VectorType* source,
                                               const VectorType* target)
      -> bool;
  [[nodiscard]] auto canSplatIntoVector(ExpressionAST* expr,
                                        const VectorType* vectorType) -> bool;
  [[nodiscard]] auto canSplatIntoExtVector(const Type* scalarType,
                                           const Type* elementType) -> bool;
  [[nodiscard]] auto canSplatIntoGnuVector(ExpressionAST* expr,
                                           const Type* elementType) -> bool;
  [[nodiscard]] auto integerTypeOrder(const Type* a, const Type* b) -> int;

  [[nodiscard]] auto commonFloatingPointType(const Type* a, const Type* b)
      -> const Type*;
  [[nodiscard]] auto commonArithmeticType(const Type* a, const Type* b)
      -> const Type*;

  [[nodiscard]] auto compositePointerType(ExpressionAST*& expr,
                                          ExpressionAST*& other) -> const Type*;

  [[nodiscard]] auto classAdjustment(const Type* sourceType,
                                     const Type* targetType) -> ClassAdjustment;

  [[nodiscard]] auto pointerConversionCastKind(const Type* sourceType,
                                               const Type* targetType)
      -> ImplicitCastKind;

  auto convertToBaseClass(ExpressionAST*& expr, const Type* baseType) -> bool;

  auto convertToDerivedClass(ExpressionAST*& expr, const Type* derivedType)
      -> bool;

  void convertPointer(ExpressionAST*& expr, const Type* targetType);

  [[nodiscard]] auto constructClassByDirectInitialization(ExpressionAST*& expr,
                                                          const Type* classType)
      -> bool;

  [[nodiscard]] auto bestConversionTarget(
      ExpressionAST* expr, const std::vector<const Type*>& targets)
      -> const Type*;

  [[nodiscard]] auto convertClassOperandToIntegralOrEnumeration(
      ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto convertClassTargetForBuiltinCompoundAssignment(
      ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto convertClassOperandForBuiltinOperator(
      ExpressionAST*& expr, const Type* enumeration = nullptr) -> bool;

  [[nodiscard]] auto isNullPointerConstant(ExpressionAST* expr) const -> bool;

  void foldConstantRead(ExpressionAST*& expression);
  [[nodiscard]] auto readsConstantVariable(ImplicitCastExpressionAST* read,
                                           Symbol* variable) -> bool;
  [[nodiscard]] auto readOfPotentialResult(ImplicitCastExpressionAST* read,
                                           ExpressionAST* operand)
      -> ExpressionAST*;
  void foldConditionalRead(ExpressionAST*& expression,
                           ImplicitCastExpressionAST* read,
                           ConditionalExpressionAST* conditional);
  void foldCommaRead(ExpressionAST*& expression,
                     ImplicitCastExpressionAST* read,
                     BinaryExpressionAST* comma);

  void appendDefaultArguments(FunctionSymbol* function,
                              List<ExpressionAST*>** list,
                              SourceLocation location = {});

  void recordConversionFunction(ImplicitCastExpressionAST* cast,
                                const ImplicitConversionSequence& sequence);

  [[nodiscard]] auto parameterConversionSequence(
      FunctionSymbol* function, ExpressionAST* argument,
      const Type* parameterType,
      InitializationKind initializationKind =
          InitializationKind::kCopyInitialization,
      ConversionContext context = ConversionContext::kImplicit)
      -> ImplicitConversionSequence;

  void convertObjectAddressSpace(ExpressionAST*& object,
                                 FunctionSymbol* function);

  [[nodiscard]] auto referenceBinding(const Type* targetType,
                                      const Type* sourceType,
                                      ValueCategory valueCategory,
                                      NonAddressableDesignation designation)
      -> std::optional<ImplicitConversionSequence>;

  [[nodiscard]] static auto nonAddressableDesignation(ExpressionAST* expr)
      -> NonAddressableDesignation;

  [[nodiscard]] static auto designatesAddressableStorage(ExpressionAST* expr)
      -> bool;

 private:
  [[nodiscard]] auto objectClassType(const Type* type) const -> const Type*;

  [[nodiscard]] auto addressSpaceGenericBinding(FunctionSymbol* function,
                                                ExpressionAST* argument,
                                                const Type* parameterType)
      -> ImplicitConversionSequence;

  [[nodiscard]] auto conversionFunctionResultType(FunctionSymbol* function,
                                                  const Type* targetType) const
      -> const Type*;
  [[nodiscard]] auto userDefinedSequence(const Candidate& candidate,
                                         ExpressionAST* expr,
                                         const Type* targetType)
      -> ImplicitConversionSequence;
  [[nodiscard]] auto objectArgumentConversion(FunctionSymbol* function,
                                              ExpressionAST* expr)
      -> std::optional<ImplicitConversionSequence>;
  void addUserDefinedCandidate(
      std::vector<Candidate>& candidates, FunctionSymbol* function,
      std::optional<ImplicitConversionSequence> argument,
      ImplicitConversionSequence result,
      std::optional<ImplicitConversionSequence> object);
  void addConstructorCandidates(std::vector<Candidate>& candidates,
                                ExpressionAST* expr, const Type* targetType,
                                InitializationKind initializationKind);
  [[nodiscard]] auto constructorArgumentConversion(
      ExpressionAST* expr, const FunctionType* constructorType)
      -> ImplicitConversionSequence;
  void addConstructorCandidate(std::vector<Candidate>& candidates,
                               FunctionSymbol* constructor,
                               ClassSymbol* classSymbol, ExpressionAST* expr,
                               const Type* targetType);
  void addConversionFunctionCandidates(std::vector<Candidate>& candidates,
                                       ExpressionAST* expr,
                                       const Type* targetType,
                                       InitializationKind initializationKind);
  void addConversionFunctionCandidate(std::vector<Candidate>& candidates,
                                      FunctionSymbol* function,
                                      ExpressionAST* expr,
                                      const Type* targetType,
                                      InitializationKind initializationKind);
  [[nodiscard]] auto userDefinedConversionSequence(
      ExpressionAST* expr, const Type* targetType,
      InitializationKind initializationKind) -> ImplicitConversionSequence;

  [[nodiscard]] auto directReferenceBinding(ImplicitConversionSequence sequence,
                                            ValueCategory category)
      -> ImplicitConversionSequence;
  [[nodiscard]] auto classListInitializationSequence(
      ImplicitConversionSequence sequence, const Type* targetType)
      -> ImplicitConversionSequence;
  [[nodiscard]] auto aggregateListInitializationSequence(
      ImplicitConversionSequence sequence, BracedInitListAST* initializer,
      const Type* targetType) -> ImplicitConversionSequence;
  [[nodiscard]] auto listElementSequence(ExpressionAST* element,
                                         const Type* targetType,
                                         InitializationKind initializationKind)
      -> ImplicitConversionSequence;

  [[nodiscard]] auto overloadSetConversionSequence(ExpressionAST* expr,
                                                   const Type* targetType)
      -> std::optional<ImplicitConversionSequence>;
  [[nodiscard]] auto pointerConversionSequence(
      ImplicitConversionSequence sequence, const Type* sourceType,
      const Type* targetType) -> std::optional<ImplicitConversionSequence>;
  [[nodiscard]] auto memberPointerConversionSequence(
      ImplicitConversionSequence sequence, const Type* sourceType,
      const Type* targetType) -> std::optional<ImplicitConversionSequence>;

  void applyStep(const ImplicitConversionSequence& sequence,
                 const ImplicitConversionSequence::Step& step,
                 ExpressionAST*& expr);

  void applyCopyConstruction(const ImplicitConversionSequence& sequence,
                             ExpressionAST*& expr);

  void appendTemporaryMaterialization(ImplicitConversionSequence& sequence);

  void recordUserDefinedConversion(ImplicitCastExpressionAST* cast,
                                   FunctionSymbol* function);

  void applyConversionFunctionObjectSteps(
      const ImplicitConversionSequence& sequence, ExpressionAST*& expr);

  [[nodiscard]] auto selectCopyConstructor(
      ExpressionAST* expr, const Type* destinationType,
      InitializationKind initializationKind) -> FunctionSymbol*;

  [[nodiscard]] auto listInitializationSequence(
      BracedInitListAST* bracedInitList, const Type* targetType,
      InitializationKind initializationKind, ConversionContext context)
      -> ImplicitConversionSequence;

  [[nodiscard]] auto isDesignatedInitializerList(
      BracedInitListAST* bracedInitList) const -> bool;

  [[nodiscard]] auto aggregateListConversion(BracedInitListAST* bracedInitList,
                                             const Type* aggregateType)
      -> AggregateListConversion;

  [[nodiscard]] auto rankAggregateElements(BracedInitListAST* bracedInitList,
                                           const Type* aggregateType)
      -> AggregateListConversion;

  void appendDirectBindingSteps(ImplicitConversionSequence& seq,
                                const Type* referencedType,
                                const Type* sourceType,
                                ValueCategory sourceValueCategory);

  [[nodiscard]] auto directReferenceBindingCastKind(const Type* referencedType,
                                                    const Type* sourceType)
      -> ImplicitCastKind;

  [[nodiscard]] auto computeConversionSequenceSteps(
      ExpressionAST* expr, const Type* targetType,
      InitializationKind initializationKind, ConversionContext context)
      -> ImplicitConversionSequence;

  void wrapWithImplicitCast(ImplicitCastKind castKind, const Type* type,
                            ExpressionAST*& expr);

  void materializeClosureConversion(ClassSymbol* srcClass,
                                    const Type* targetType);

  [[nodiscard]] auto convertedClassSymbol(const Type* type) const
      -> ClassSymbol*;

  [[nodiscard]] auto accessingScope() const -> ScopeSymbol*;

  [[nodiscard]] auto baseConversionClasses(const Type* sourceType,
                                           const Type* targetType,
                                           ImplicitCastKind kind) const
      -> std::pair<ClassSymbol*, ClassSymbol*>;
  [[nodiscard]] auto isAccessibleBaseConversion(const Type* sourceType,
                                                const Type* targetType,
                                                ImplicitCastKind kind) const
      -> bool;

  void checkBaseConversion(const Type* sourceType, const Type* targetType,
                           ImplicitCastKind kind, SourceLocation loc);

  void checkUserDefinedConversionAccess(
      const ImplicitConversionSequence& sequence, ExpressionAST* expr);

  void applyResolvedFunction(ExpressionAST* expr, FunctionSymbol* function);

  void setResolvedFunction(ExpressionAST* expr, FunctionSymbol* function);

  [[nodiscard]] auto pointeeClassAdjustment(const Type* sourceType,
                                            const Type* targetType)
      -> ClassAdjustment;

  void adjustCv(ExpressionAST* expr);

  [[nodiscard]] auto readsValueDirectly(const Type* type) const -> bool;

  [[nodiscard]] auto valueTransformation(ExpressionAST* expr)
      -> std::optional<ImplicitConversionSequence::Step>;

  [[nodiscard]] auto floatingPointPromotion(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto convertArithmetic(ExpressionAST*& expr,
                                       const Type* destinationType) -> bool;

  [[nodiscard]] auto arithmeticConversionKind(const Type* source,
                                              const Type* target) const
      -> std::optional<ImplicitCastKind>;

  void requireDefinitionOfDesignatedField(ExpressionAST* expr);

  [[nodiscard]] static auto namedEntity(ExpressionAST* expr) -> Symbol*;

  void requireNamedFunction(ExpressionAST* expr);

  void materializeConstructorArguments(ImplicitCastExpressionAST* cast,
                                       FunctionSymbol* constructor);

  [[nodiscard]] auto requiresCopyConstruction(ExpressionAST* expr,
                                              const Type* destinationType) const
      -> bool;

  [[nodiscard]] auto selectListInitializationConstructor(
      BracedInitListAST* bracedInitList, const Type* targetType,
      InitializationKind initializationKind) -> ConstructorResult;

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
                                                const Type* target) -> bool;

 private:
  TranslationUnit* unit_;
  TypeTraits traits;
  Control* control_;
  Arena* arena_;
  std::vector<const Type*> rankedAggregates_;
  ScopeSymbol* accessingScope_ = nullptr;
  bool isC_ = false;
};
}  // namespace cxx
