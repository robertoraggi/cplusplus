// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <utility>
#include <vector>

namespace cxx {

class Symbol {
 public:
  Symbol(SymbolKind kind, Scope* enclosingScope)
      : kind_(kind), enclosingScope_(enclosingScope) {}

  virtual ~Symbol() = default;

  [[nodiscard]] auto kind() const -> SymbolKind { return kind_; }

  [[nodiscard]] auto name() const -> const Name* { return name_; }
  void setName(const Name* name) { name_ = name; }

  [[nodiscard]] auto type() const -> const Type* { return type_; }
  void setType(const Type* type) { type_ = type; }

  [[nodiscard]] auto enclosingScope() const -> Scope* {
    return enclosingScope_;
  }
  void setEnclosingScope(Scope* enclosingScope) {
    enclosingScope_ = enclosingScope;
  }

  [[nodiscard]] auto insertionPoint() const -> int { return insertionPoint_; }
  void setInsertionPoint(int index) { insertionPoint_ = index; }

#define PROCESS_SYMBOL(S) \
  [[nodiscard]] auto is##S() const->bool { return kind_ == SymbolKind::k##S; }
  CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)
#undef PROCESS_SYMBOL

 private:
  SymbolKind kind_;
  const Name* name_ = nullptr;
  const Type* type_ = nullptr;
  Scope* enclosingScope_ = nullptr;
  int insertionPoint_ = 0;
};

class NamespaceSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kNamespace;

  explicit NamespaceSymbol(Scope* enclosingScope);
  ~NamespaceSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

  [[nodiscard]] auto isInline() const -> bool { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

  [[nodiscard]] auto unnamedNamespace() const -> NamespaceSymbol* {
    return unnamedNamespace_;
  }

  void setUnnamedNamespace(NamespaceSymbol* unnamedNamespace) {
    unnamedNamespace_ = unnamedNamespace;
  }

  [[nodiscard]] auto usingNamespaces() const
      -> const std::vector<NamespaceSymbol*>& {
    return usingNamespaces_;
  }

  void addUsingNamespace(NamespaceSymbol* usingNamespace) {
    usingNamespaces_.push_back(usingNamespace);
  }

 private:
  std::unique_ptr<Scope> scope_;
  NamespaceSymbol* unnamedNamespace_ = nullptr;
  std::vector<NamespaceSymbol*> usingNamespaces_;
  bool isInline_ = false;
};

class ConceptSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kConcept;

  explicit ConceptSymbol(Scope* enclosingScope);
  ~ConceptSymbol() override;

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
};

class ClassSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kClass;

  explicit ClassSymbol(Scope* enclosingScope);
  ~ClassSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

  [[nodiscard]] auto isComplete() const -> bool { return isComplete_; }
  void setComplete(bool isComplete) { isComplete_ = isComplete; }

 private:
  std::unique_ptr<Scope> scope_;
  TemplateParametersSymbol* templateParameters_ = nullptr;
  bool isComplete_ = false;
};

class UnionSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kUnion;

  explicit UnionSymbol(Scope* enclosingScope);
  ~UnionSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class EnumSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kEnum;

  explicit EnumSymbol(Scope* enclosingScope);
  ~EnumSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

  [[nodiscard]] auto underlyingType() const -> const Type* {
    return underlyingType_;
  }

  void setUnderlyingType(const Type* underlyingType) {
    underlyingType_ = underlyingType;
  }

 private:
  std::unique_ptr<Scope> scope_;
  const Type* underlyingType_ = nullptr;
};

class ScopedEnumSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kScopedEnum;

  explicit ScopedEnumSymbol(Scope* enclosingScope);
  ~ScopedEnumSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

  [[nodiscard]] auto underlyingType() const -> const Type* {
    return underlyingType_;
  }

  void setUnderlyingType(const Type* underlyingType) {
    underlyingType_ = underlyingType;
  }

 private:
  std::unique_ptr<Scope> scope_;
  const Type* underlyingType_ = nullptr;
};

class FunctionSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunction;

  explicit FunctionSymbol(Scope* enclosingScope);
  ~FunctionSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

 private:
  std::unique_ptr<Scope> scope_;
  TemplateParametersSymbol* templateParameters_ = nullptr;
};

class LambdaSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kLambda;

  explicit LambdaSymbol(Scope* enclosingScope);
  ~LambdaSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class FunctionParametersSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunctionParameters;

  explicit FunctionParametersSymbol(Scope* enclosingScope);
  ~FunctionParametersSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class TemplateParametersSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateParameters;

  explicit TemplateParametersSymbol(Scope* enclosingScope);
  ~TemplateParametersSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class BlockSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kBlock;

  explicit BlockSymbol(Scope* enclosingScope);
  ~BlockSymbol() override;

  [[nodiscard]] auto scope() const -> Scope* { return scope_.get(); }

 private:
  std::unique_ptr<Scope> scope_;
};

class TypeAliasSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTypeAlias;

  explicit TypeAliasSymbol(Scope* enclosingScope);
  ~TypeAliasSymbol() override;

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
};

class VariableSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kVariable;

  explicit VariableSymbol(Scope* enclosingScope);
  ~VariableSymbol() override;

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
};

class FieldSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kField;

  explicit FieldSymbol(Scope* enclosingScope);
  ~FieldSymbol() override;
};

class ParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kParameter;

  explicit ParameterSymbol(Scope* enclosingScope);
  ~ParameterSymbol() override;
};

class TypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTypeParameter;

  explicit TypeParameterSymbol(Scope* enclosingScope);
  ~TypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int { return index_; }
  void setIndex(int index) { index_ = index; }

  [[nodiscard]] auto depth() const -> int { return depth_; }
  void setDepth(int depth) { depth_ = depth; }

  [[nodiscard]] auto isParameterPack() const -> bool {
    return isParameterPack_;
  }

  void setParameterPack(bool isParameterPack) {
    isParameterPack_ = isParameterPack;
  }

 private:
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class NonTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kNonTypeParameter;

  explicit NonTypeParameterSymbol(Scope* enclosingScope);
  ~NonTypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int { return index_; }
  void setIndex(int index) { index_ = index; }

  [[nodiscard]] auto depth() const -> int { return depth_; }
  void setDepth(int depth) { depth_ = depth; }

  [[nodiscard]] auto objectType() const -> const Type* { return objectType_; }
  void setObjectType(const Type* objectType) { objectType_ = objectType; }

  [[nodiscard]] auto isParameterPack() const -> bool {
    return isParameterPack_;
  }

  void setParameterPack(bool isParameterPack) {
    isParameterPack_ = isParameterPack;
  }

 private:
  const Type* objectType_ = nullptr;
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class TemplateTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateTypeParameter;

  explicit TemplateTypeParameterSymbol(Scope* enclosingScope);
  ~TemplateTypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int { return index_; }
  void setIndex(int index) { index_ = index; }

  [[nodiscard]] auto depth() const -> int { return depth_; }
  void setDepth(int depth) { depth_ = depth; }

  [[nodiscard]] auto isParameterPack() const -> bool {
    return isParameterPack_;
  }

  void setParameterPack(bool isParameterPack) {
    isParameterPack_ = isParameterPack;
  }

 private:
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class ConstraintTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kConstraintTypeParameter;

  explicit ConstraintTypeParameterSymbol(Scope* enclosingScope);
  ~ConstraintTypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int { return index_; }
  void setIndex(int index) { index_ = index; }

  [[nodiscard]] auto depth() const -> int { return depth_; }
  void setDepth(int depth) { depth_ = depth; }

  [[nodiscard]] auto isParameterPack() const -> bool {
    return isParameterPack_;
  }

  void setParameterPack(bool isParameterPack) {
    isParameterPack_ = isParameterPack;
  }

 private:
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class EnumeratorSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kEnumerator;

  explicit EnumeratorSymbol(Scope* enclosingScope);
  ~EnumeratorSymbol() override;
};

template <typename Visitor>
auto visit(Visitor&& visitor, Symbol* symbol) {
#define PROCESS_SYMBOL(S) \
  case SymbolKind::k##S:  \
    return std::forward<Visitor>(visitor)(static_cast<S##Symbol*>(symbol));

  switch (symbol->kind()) {
    CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)
    default:
      cxx_runtime_error("invalid symbol kind");
  }  // switch

#undef PROCESS_SYMBOL
}

#define PROCESS_SYMBOL(S)                                \
  inline auto is##S##Symbol(Symbol* symbol)->bool {      \
    return symbol && symbol->kind() == SymbolKind::k##S; \
  }

CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)

#undef PROCESS_SYMBOL

template <typename T>
auto symbol_cast(Symbol* symbol) -> T* {
  if (symbol && symbol->kind() == T::Kind) return static_cast<T*>(symbol);
  return nullptr;
}

}  // namespace cxx
