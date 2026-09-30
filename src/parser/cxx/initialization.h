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
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <string>
#include <vector>

namespace cxx {

[[nodiscard]] auto makeDefaultInitializer(TranslationUnit* unit,
                                          ExpressionAST* expression,
                                          SourceLocation location,
                                          ScopeSymbol* scope) -> ExpressionAST*;

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

enum class ArrayCopyPolicy {
  kBracedInitializerOnly,
  kElementwiseCopyAllowed,
};

class InitializedEntity {
 public:
  [[nodiscard]] static auto variable(const Type* type, Symbol* symbol,
                                     SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto member(const Type* type, Symbol* symbol,
                                   SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] static auto arrayElement(const Type* type,
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

  [[nodiscard]] static auto delegating(const Type* type,
                                       SourceLocation location)
      -> InitializedEntity;

  [[nodiscard]] auto arrayCopyPolicy() const -> ArrayCopyPolicy {
    return arrayCopyPolicy_;
  }
  void setArrayCopyPolicy(ArrayCopyPolicy policy) { arrayCopyPolicy_ = policy; }

  [[nodiscard]] auto type() const -> const Type* { return type_; }
  [[nodiscard]] auto location() const -> SourceLocation { return location_; }

  [[nodiscard]] auto description() const -> std::string;

 private:
  enum class Kind {
    kVariable,
    kMember,
    kArrayElement,
    kReturnObject,
    kExceptionObject,
    kTemporary,
    kDelegating,
  };

  InitializedEntity(Kind kind, const Type* type, SourceLocation location,
                    Symbol* symbol = nullptr)
      : kind_(kind), type_(type), symbol_(symbol), location_(location) {}

  Kind kind_ = Kind::kTemporary;
  ArrayCopyPolicy arrayCopyPolicy_ = ArrayCopyPolicy::kBracedInitializerOnly;
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

  [[nodiscard]] auto form() const -> InitializerForm;

  [[nodiscard]] auto clause() const -> ExpressionAST*;

  [[nodiscard]] auto bracedInitList() const -> BracedInitListAST*;

  [[nodiscard]] auto initializationKind() const -> InitializationKind;

  [[nodiscard]] auto singleExpression() const -> ExpressionAST*;

  [[nodiscard]] auto arguments() const -> std::vector<ExpressionAST*>;

 private:
  [[nodiscard]] auto unwrapEqual() const -> ExpressionAST*;

  ExpressionAST* node_ = nullptr;
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

[[nodiscard]] auto isWholeArrayCopy(const TypeTraits& traits,
                                    ExpressionAST* expression,
                                    const Type* arrayType) -> bool;

struct InitContext {
  TypeChecker& checker;
  TranslationUnit* unit;
  Control* control;
  TypeTraits traits;

  explicit InitContext(TypeChecker& checker);

  [[nodiscard]] auto isCxx() const -> bool;

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  [[nodiscard]] auto isTargetTypeUnresolved(const Type* type) const -> bool;
};

struct AggregateInitializerElement {
  std::size_t index = 0;
  Symbol* element = nullptr;
  const Type* type = nullptr;
  ExpressionAST* initializer = nullptr;
  bool elided = false;
};

struct AggregateInitializerPlan {
  std::vector<Symbol*> elements;
  std::vector<AggregateInitializerElement> initializedElements;
  const Type* arrayElementType = nullptr;
  std::size_t elementCount = 0;
  bool isUnion = false;
  bool isVector = false;
  bool valid = true;
};

struct MaterializedTemporary {
  ExpressionAST* expression = nullptr;
  bool conditional = false;

  [[nodiscard]] explicit operator bool() const { return expression != nullptr; }
};

[[nodiscard]] auto materializedTemporary(const TypeTraits& traits,
                                         ExpressionAST* initializer)
    -> MaterializedTemporary;

struct StringLiteralInitialization {
  const Type* destinationElementType = nullptr;
  const Type* sourceElementType = nullptr;
  std::size_t elementCount = 0;
  std::size_t minimumElements = 0;
  std::size_t availableElements = 0;
  bool bounded = false;
  bool compatible = false;

  [[nodiscard]] auto tooLong() const -> bool {
    return compatible && bounded && minimumElements > availableElements;
  }
};

[[nodiscard]] auto singleInitializerClause(BracedInitListAST* bracedInitList)
    -> ExpressionAST*;

struct ArrayElementRange {
  std::size_t begin = 0;
  std::size_t end = 0;
};

[[nodiscard]] auto designatedArrayIndex(TranslationUnit* unit,
                                        DesignatedInitializerClauseAST* clause)
    -> std::optional<std::size_t>;

[[nodiscard]] auto implicitlyInitializedElements(TranslationUnit* unit,
                                                 BracedInitListAST* list,
                                                 std::size_t elementCount)
    -> std::vector<ArrayElementRange>;

[[nodiscard]] auto stringLiteralInitialization(const TypeTraits& traits,
                                               bool isCxx,
                                               const Type* destinationType,
                                               ExpressionAST* source)
    -> std::optional<StringLiteralInitialization>;

[[nodiscard]] auto planAggregateInitialization(
    TranslationUnit* unit, const Type* aggregateType,
    BracedInitListAST* bracedInitList)
    -> std::optional<AggregateInitializerPlan>;

enum class InitializationStatus { kComplete, kDeferred, kFailed };

struct InitializationResult {
  InitializationStatus status = InitializationStatus::kDeferred;
  FunctionSymbol* constructor = nullptr;
};

[[nodiscard]] auto initialize(InitContext& ctx, const InitializedEntity& entity,
                              InitializationKind kind,
                              ExpressionAST*& initializer)
    -> InitializationResult;

void diagnoseNarrowingListElement(InitContext& ctx, ExpressionAST* element,
                                  const Type* targetType);

void diagnoseConversionFailure(InitContext& ctx,
                               const InitializedEntity& entity,
                               ExpressionAST* source);

}  // namespace cxx
