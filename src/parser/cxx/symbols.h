// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <optional>
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

  [[nodiscard]] auto enclosingSymbol() const -> Symbol*;

  [[nodiscard]] auto insertionPoint() const -> int { return insertionPoint_; }
  void setInsertionPoint(int index) { insertionPoint_ = index; }

#define PROCESS_SYMBOL(S) \
  [[nodiscard]] auto is##S() const -> bool { return kind_ == SymbolKind::k##S; }
  CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)
#undef PROCESS_SYMBOL

 private:
  SymbolKind kind_;
  const Name* name_ = nullptr;
  const Type* type_ = nullptr;
  Scope* enclosingScope_ = nullptr;
  int insertionPoint_ = 0;
};

class ScopedSymbol : public Symbol {
 public:
  ScopedSymbol(SymbolKind kind, Scope* enclosingScope);
  ~ScopedSymbol() override;

  [[nodiscard]] auto scope() const -> Scope*;

 private:
  std::unique_ptr<Scope> scope_;
};

class NamespaceSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kNamespace;

  explicit NamespaceSymbol(Scope* enclosingScope);
  ~NamespaceSymbol() override;

  [[nodiscard]] auto isInline() const -> bool { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

  [[nodiscard]] auto unnamedNamespace() const -> NamespaceSymbol* {
    return unnamedNamespace_;
  }

  void setUnnamedNamespace(NamespaceSymbol* unnamedNamespace) {
    unnamedNamespace_ = unnamedNamespace;
  }

 private:
  NamespaceSymbol* unnamedNamespace_ = nullptr;
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

class BaseClassSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kBaseClass;

  explicit BaseClassSymbol(Scope* enclosingScope);
  ~BaseClassSymbol() override;

  [[nodiscard]] auto isVirtual() const -> bool { return isVirtual_; }
  void setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

  [[nodiscard]] auto accessSpecifier() const -> AccessSpecifier {
    return accessSpecifier_;
  }

  void setAccessSpecifier(AccessSpecifier accessSpecifier) {
    accessSpecifier_ = accessSpecifier;
  }

  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }
  void setSymbol(Symbol* symbol) { symbol_ = symbol; }

 private:
  Symbol* symbol_ = nullptr;
  AccessSpecifier accessSpecifier_ = AccessSpecifier::kPublic;
  bool isVirtual_ = false;
};

class ClassSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kClass;

  explicit ClassSymbol(Scope* enclosingScope);
  ~ClassSymbol() override;

  [[nodiscard]] auto isUnion() const -> bool;
  void setIsUnion(bool isUnion);

  [[nodiscard]] auto baseClasses() const
      -> const std::vector<BaseClassSymbol*>&;

  void addBaseClass(BaseClassSymbol* baseClass);

  [[nodiscard]] auto constructors() const
      -> const std::vector<FunctionSymbol*>&;

  void addConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol*;
  void setTemplateParameters(TemplateParametersSymbol* templateParameters);

  [[nodiscard]] auto isComplete() const -> bool;
  void setComplete(bool isComplete);

  [[nodiscard]] auto sizeInBytes() const -> std::size_t;

 private:
  std::vector<BaseClassSymbol*> baseClasses_;
  std::vector<FunctionSymbol*> constructors_;
  TemplateParametersSymbol* templateParameters_ = nullptr;
  std::size_t sizeInBytes_ = 0;
  bool isUnion_ = false;
  bool isComplete_ = false;
};

class EnumSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kEnum;

  explicit EnumSymbol(Scope* enclosingScope);
  ~EnumSymbol() override;

  [[nodiscard]] auto underlyingType() const -> const Type* {
    return underlyingType_;
  }

  void setUnderlyingType(const Type* underlyingType) {
    underlyingType_ = underlyingType;
  }

 private:
  const Type* underlyingType_ = nullptr;
};

class ScopedEnumSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kScopedEnum;

  explicit ScopedEnumSymbol(Scope* enclosingScope);
  ~ScopedEnumSymbol() override;

  [[nodiscard]] auto underlyingType() const -> const Type* {
    return underlyingType_;
  }

  void setUnderlyingType(const Type* underlyingType) {
    underlyingType_ = underlyingType;
  }

 private:
  const Type* underlyingType_ = nullptr;
};

class FunctionSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunction;

  explicit FunctionSymbol(Scope* enclosingScope);
  ~FunctionSymbol() override;

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

  [[nodiscard]] auto isStatic() const { return isStatic_; }
  void setStatic(bool isStatic) { isStatic_ = isStatic; }

  [[nodiscard]] auto isExtern() const { return isExtern_; }
  void setExtern(bool isExtern) { isExtern_ = isExtern; }

  [[nodiscard]] auto isFriend() const { return isFriend_; }
  void setFriend(bool isFriend) { isFriend_ = isFriend; }

  [[nodiscard]] auto isConstexpr() const { return isConstexpr_; }
  void setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

  [[nodiscard]] auto isConsteval() const { return isConsteval_; }
  void setConsteval(bool isConsteval) { isConsteval_ = isConsteval; }

  [[nodiscard]] auto isInline() const { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

  [[nodiscard]] auto isVirtual() const { return isVirtual_; }
  void setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

  [[nodiscard]] auto isExplicit() const { return isExplicit_; }
  void setExplicit(bool isExplicit) { isExplicit_ = isExplicit; }

  [[nodiscard]] auto isDeleted() const { return isDeleted_; }
  void setDeleted(bool isDeleted) { isDeleted_ = isDeleted; }

  [[nodiscard]] auto isDefaulted() const { return isDefaulted_; }
  void setDefaulted(bool isDefaulted) { isDefaulted_ = isDefaulted; }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
  bool isStatic_ = false;
  bool isExtern_ = false;
  bool isFriend_ = false;
  bool isConstexpr_ = false;
  bool isConsteval_ = false;
  bool isInline_ = false;
  bool isVirtual_ = false;
  bool isExplicit_ = false;
  bool isDeleted_ = false;
  bool isDefaulted_ = false;
};

class OverloadSetSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kOverloadSet;

  explicit OverloadSetSymbol(Scope* enclosingScope);
  ~OverloadSetSymbol() override;

  [[nodiscard]] auto functions() const -> const std::vector<FunctionSymbol*>& {
    return functions_;
  }

  void setFunctions(std::vector<FunctionSymbol*> functions) {
    functions_ = std::move(functions);
  }

  void addFunction(FunctionSymbol* function) { functions_.push_back(function); }

 private:
  std::vector<FunctionSymbol*> functions_;
};

class LambdaSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kLambda;

  explicit LambdaSymbol(Scope* enclosingScope);
  ~LambdaSymbol() override;

  [[nodiscard]] auto templateParameters() const
      -> const TemplateParametersSymbol* {
    return templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    templateParameters_ = templateParameters;
  }

  [[nodiscard]] auto isConstexpr() const { return isConstexpr_; }
  void setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

  [[nodiscard]] auto isConsteval() const { return isConsteval_; }
  void setConsteval(bool isConsteval) { isConsteval_ = isConsteval; }

  [[nodiscard]] auto isMutable() const { return isMutable_; }
  void setMutable(bool isMutable) { isMutable_ = isMutable; }

  [[nodiscard]] auto isStatic() const { return isStatic_; }
  void setStatic(bool isStatic) { isStatic_ = isStatic; }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
  bool isConstexpr_ = false;
  bool isConsteval_ = false;
  bool isMutable_ = false;
  bool isStatic_ = false;
};

class FunctionParametersSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunctionParameters;

  explicit FunctionParametersSymbol(Scope* enclosingScope);
  ~FunctionParametersSymbol() override;
};

class TemplateParametersSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateParameters;

  explicit TemplateParametersSymbol(Scope* enclosingScope);
  ~TemplateParametersSymbol() override;
};

class BlockSymbol final : public ScopedSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kBlock;

  explicit BlockSymbol(Scope* enclosingScope);
  ~BlockSymbol() override;
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

  [[nodiscard]] auto isStatic() const { return isStatic_; }
  void setStatic(bool isStatic) { isStatic_ = isStatic; }

  [[nodiscard]] auto isThreadLocal() const { return isThreadLocal_; }
  void setThreadLocal(bool isThreadLocal) { isThreadLocal_ = isThreadLocal; }

  [[nodiscard]] auto isExtern() const { return isExtern_; }
  void setExtern(bool isExtern) { isExtern_ = isExtern; }

  [[nodiscard]] auto isConstexpr() const { return isConstexpr_; }
  void setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

  [[nodiscard]] auto isConstinit() const { return isConstinit_; }
  void setConstinit(bool isConstinit) { isConstinit_ = isConstinit; }

  [[nodiscard]] auto isInline() const { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

 private:
  TemplateParametersSymbol* templateParameters_ = nullptr;
  bool isStatic_ = false;
  bool isThreadLocal_ = false;
  bool isExtern_ = false;
  bool isConstexpr_ = false;
  bool isConstinit_ = false;
  bool isInline_ = false;
};

class FieldSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kField;

  explicit FieldSymbol(Scope* enclosingScope);
  ~FieldSymbol() override;

  [[nodiscard]] auto isStatic() const { return isStatic_; }
  void setStatic(bool isStatic) { isStatic_ = isStatic; }

  [[nodiscard]] auto isThreadLocal() const { return isThreadLocal_; }
  void setThreadLocal(bool isThreadLocal) { isThreadLocal_ = isThreadLocal; }

  [[nodiscard]] auto isConstexpr() const { return isConstexpr_; }
  void setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

  [[nodiscard]] auto isConstinit() const { return isConstinit_; }
  void setConstinit(bool isConstinit) { isConstinit_ = isConstinit; }

  [[nodiscard]] auto isInline() const { return isInline_; }
  void setInline(bool isInline) { isInline_ = isInline; }

 private:
  bool isStatic_ = false;
  bool isThreadLocal_ = false;
  bool isConstexpr_ = false;
  bool isConstinit_ = false;
  bool isInline_ = false;
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

  [[nodiscard]] auto value() const -> const std::optional<ConstValue>& {
    return value_;
  }

  void setValue(const std::optional<ConstValue>& value) { value_ = value; }

 private:
  std::optional<ConstValue> value_;
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
  inline auto is##S##Symbol(Symbol* symbol) -> bool {    \
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
