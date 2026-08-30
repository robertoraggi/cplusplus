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
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <string>
#include <vector>

namespace cxx {

class Arena;
class Control;
class TranslationUnit;
class TypeChecker;

enum class InitializationKind {
  kCopyInitialization,
  kDirectInitialization,
  kCopyListInitialization,
  kDirectListInitialization,
};

[[nodiscard]] constexpr auto isDirectInitialization(InitializationKind kind)
    -> bool {
  return kind == InitializationKind::kDirectInitialization ||
         kind == InitializationKind::kDirectListInitialization;
}

[[nodiscard]] constexpr auto isListInitialization(InitializationKind kind)
    -> bool {
  return kind == InitializationKind::kCopyListInitialization ||
         kind == InitializationKind::kDirectListInitialization;
}

[[nodiscard]] constexpr auto asListInitialization(InitializationKind kind)
    -> InitializationKind {
  return isDirectInitialization(kind)
             ? InitializationKind::kDirectListInitialization
             : InitializationKind::kCopyListInitialization;
}

enum class InitializedEntityKind {
  kVariable,
  kMember,
  kBase,
  kArrayElement,
  kParameter,
  kReturnObject,
  kExceptionObject,
  kTemporary,
  kNewObject,
  kDelegating,
};

class InitializedEntity {
 public:
  InitializedEntity() = default;

  [[nodiscard]] static auto variable(const Type* type, Symbol* symbol,
                                     SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto member(const Type* type, Symbol* symbol,
                                   SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto base(const Type* type, SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto arrayElement(const Type* type,
                                         SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto parameter(const Type* type, Symbol* symbol,
                                      SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto returnObject(const Type* type,
                                         SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto exceptionObject(const Type* type,
                                            SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto temporary(const Type* type, SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto newObject(const Type* type, SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto delegating(const Type* type,
                                       SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] auto kind() const -> InitializedEntityKind { return kind_; }
  [[nodiscard]] auto type() const -> const Type* { return type_; }
  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }
  [[nodiscard]] auto location() const -> SourceLocation { return location_; }

  void setType(const Type* type) { type_ = type; }
  void setLocation(SourceLocation location) { location_ = location; }

  [[nodiscard]] auto description() const -> std::string;

 private:
  InitializedEntityKind kind_ = InitializedEntityKind::kTemporary;
  const Type* type_ = nullptr;
  Symbol* symbol_ = nullptr;
  SourceLocation location_;
};

enum class InitializerForm {
  kNone,
  kExpression,
  kEqual,
  kParen,
  kList,
};

class Initializer {
 public:
  Initializer() = default;
  explicit Initializer(ExpressionAST* node) : node_(node) {}

  [[nodiscard]] static auto withArgumentList(ExpressionAST* node,
                                             List<ExpressionAST*>** arguments)
      -> Initializer;

  [[nodiscard]] static auto stripImplicitCasts(ExpressionAST* expr)
      -> ExpressionAST*;

  [[nodiscard]] auto node() const -> ExpressionAST* { return node_; }
  void setNode(ExpressionAST* node) { node_ = node; }

  [[nodiscard]] explicit operator bool() const {
    return node_ != nullptr || argumentList_ != nullptr;
  }

  [[nodiscard]] auto argumentList() const -> List<ExpressionAST*>** {
    return argumentList_;
  }

  [[nodiscard]] auto form() const -> InitializerForm;

  [[nodiscard]] auto clause() const -> ExpressionAST*;

  [[nodiscard]] auto bracedInitList() const -> BracedInitListAST*;

  [[nodiscard]] auto initializationKind() const -> InitializationKind;

  [[nodiscard]] auto singleExpression() const -> ExpressionAST*;

  [[nodiscard]] auto arguments() const -> std::vector<ExpressionAST*>;

  [[nodiscard]] auto expressionListSlot() const -> List<ExpressionAST*>**;

  [[nodiscard]] auto conversionTarget() const -> ExpressionAST**;

  void propagateType() const;

 private:
  [[nodiscard]] auto stripped() const -> ExpressionAST*;
  [[nodiscard]] auto unwrapEqual() const -> ExpressionAST*;

  ExpressionAST* node_ = nullptr;
  List<ExpressionAST*>** argumentList_ = nullptr;
};

[[nodiscard]] auto memInitializerClause(Arena* arena,
                                        MemInitializerAST* memInitializer)
    -> ExpressionAST*;

[[nodiscard]] auto memInitializerListSlot(MemInitializerAST* memInitializer)
    -> List<ExpressionAST*>**;

[[nodiscard]] auto memInitializerArgumentSlots(
    MemInitializerAST* memInitializer) -> std::vector<ExpressionAST**>;

[[nodiscard]] auto memInitializerId(MemInitializerAST* memInitializer)
    -> UnqualifiedIdAST*;

[[nodiscard]] auto constantExpressionTarget(ExpressionAST*& initializer)
    -> ExpressionAST**;

[[nodiscard]] auto makeParenInitializer(Arena* arena, SourceLocation location,
                                        List<ExpressionAST*>* arguments)
    -> ParenInitializerAST*;

struct InitContext {
  TypeChecker& checker;
  TranslationUnit* unit;
  Control* control;
  TypeTraits traits;

  explicit InitContext(TypeChecker& checker);

  [[nodiscard]] auto isCxx() const -> bool;

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  [[nodiscard]] auto initializesFromSameTypePrvalue(
      ExpressionAST* expr, const Type* targetType) const -> bool;

  [[nodiscard]] auto isTargetTypeUnresolved(const Type* type) const -> bool;
};

enum class InitializationBullet {
  kNone,
  kListInitialization,
  kReferenceBinding,
  kCharacterArrayFromStringLiteral,
  kValueInitializationFromParens,
  kArrayFromExpressionList,
  kSameTypePrvalue,
  kConstructor,
  kParenthesizedAggregate,
  kUserDefinedConversion,
  kStandardConversion,
  kDefaultInitialization,
  kValueInitialization,
  kZeroInitialization,
};

enum class ListInitializationBullet {
  kNone,
  kDesignatedAggregate,
  kAggregateFromSameOrDerivedElement,
  kCharacterArrayFromStringLiteral,
  kAggregate,
  kEmptyListDefaultConstructor,
  kInitializerList,
  kConstructor,
  kEnumerationWithFixedUnderlyingType,
  kSingleElement,
  kReferenceToPrvalue,
  kEmptyListValueInitialization,
};

enum class InitializationFailure {
  kNone,
  kUnresolvedDestinationType,
  kDependent,
  kMissingInitializer,
  kReferenceWithoutInitializer,
  kReferenceDefaultInitialized,
  kNotConstDefaultConstructible,
  kNoViableConstructor,
  kAmbiguousConstructor,
  kExplicitConstructorInCopyInitialization,
  kTooManyInitializers,
  kNoConversion,
  kIncompleteType,
};

struct InitializationSequence {
  InitializationBullet bullet = InitializationBullet::kNone;
  ListInitializationBullet listBullet = ListInitializationBullet::kNone;
  InitializationFailure failure = InitializationFailure::kNone;
  InitializationKind kind = InitializationKind::kCopyInitialization;
  const Type* destinationType = nullptr;
  FunctionSymbol* constructor = nullptr;
  bool zeroInitializesFirst = false;
  ImplicitConversionSequence conversion;
  std::vector<ImplicitConversionSequence> argumentConversions;

  [[nodiscard]] explicit operator bool() const {
    return failure == InitializationFailure::kNone &&
           bullet != InitializationBullet::kNone;
  }
};

void diagnoseNarrowingListElement(InitContext& ctx, ExpressionAST* element,
                                  const Type* targetType);

[[nodiscard]] auto computeInitializationSequence(
    InitContext& ctx, const InitializedEntity& entity, InitializationKind kind,
    const Initializer& initializer) -> InitializationSequence;

[[nodiscard]] auto applyInitializationSequence(InitContext& ctx,
                                               InitializationSequence& sequence,
                                               const InitializedEntity& entity,
                                               Initializer& initializer)
    -> ExpressionAST*;

void diagnoseInitializationFailure(InitContext& ctx,
                                   const InitializationSequence& sequence,
                                   const InitializedEntity& entity,
                                   const Initializer& initializer);

}  // namespace cxx
