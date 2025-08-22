// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <expected>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cxx {

class SymbolChainView;

template <typename S>
struct TemplateSpecialization {
  S* templateSymbol = nullptr;
  std::vector<TemplateArgument> arguments;
  S* symbol = nullptr;
};

[[nodiscard]] auto compare_args(const std::vector<TemplateArgument>& args1,
                                const std::vector<TemplateArgument>& args2)
    -> bool;

template <typename S>
class TemplateInfo {
 public:
  explicit TemplateInfo(S* templateSymbol) : templateSymbol_(templateSymbol) {}

  [[nodiscard]] auto specializations() const
      -> std::span<const TemplateSpecialization<S>> {
    return specializations_;
  }

  [[nodiscard]] auto findSpecialization(
      const std::vector<TemplateArgument>& arguments) const -> S* {
    for (const auto& specialization : specializations_) {
      const std::vector<TemplateArgument>& args = specialization.arguments;
      if (args == arguments) return specialization.symbol;
      if (args.size() != arguments.size()) continue;
      if (compare_args(args, arguments)) {
        // If the arguments match, return the specialization symbol
        return specialization.symbol;
      }
    }
    return nullptr;
  }

  void addSpecialization(std::vector<TemplateArgument> arguments,
                         S* specialization) {
    specializations_.push_back(
        {templateSymbol_, std::move(arguments), specialization});
  }

 private:
  S* templateSymbol_ = nullptr;
  std::vector<TemplateSpecialization<S>> specializations_;
};

class Symbol {
 public:
  class EnclosingSymbolIterator {
   public:
    using value_type = Symbol*;
    using difference_type = std::ptrdiff_t;

    EnclosingSymbolIterator() = default;
    explicit EnclosingSymbolIterator(ScopeSymbol* symbol) : symbol_(symbol) {}

    auto operator<=>(const EnclosingSymbolIterator&) const = default;

    auto operator*() const -> ScopeSymbol* { return symbol_; }
    auto operator++() -> EnclosingSymbolIterator&;
    auto operator++(int) -> EnclosingSymbolIterator;

   private:
    ScopeSymbol* symbol_ = nullptr;
  };

  Symbol(SymbolKind kind, ScopeSymbol* enclosingScope)
      : kind_(kind), parent_(enclosingScope) {}

  virtual ~Symbol() = default;

  [[nodiscard]] virtual auto asScopeSymbol() -> ScopeSymbol* { return nullptr; }

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto name() const -> const Name*;
  void setName(const Name* name);

  [[nodiscard]] auto type() const -> const Type*;
  void setType(const Type* type);

  [[nodiscard]] auto location() const -> SourceLocation;
  void setLocation(SourceLocation location);

  [[nodiscard]] auto parent() const -> ScopeSymbol*;
  void setParent(ScopeSymbol* parent);

  [[nodiscard]] auto enclosingNamespace() const -> NamespaceSymbol*;

  [[nodiscard]] auto enclosingNonTemplateParametersScope() const
      -> ScopeSymbol*;

  [[nodiscard]] auto enclosingSymbols() const {
    return std::ranges::subrange(EnclosingSymbolIterator{parent()},
                                 EnclosingSymbolIterator{});
  }

  [[nodiscard]] auto templateParameters() -> TemplateParametersSymbol*;

  [[nodiscard]] auto hasEnclosingSymbol(Symbol* symbol) const -> bool;

  [[nodiscard]] auto next() const -> Symbol*;

#define PROCESS_SYMBOL(S) \
  [[nodiscard]] auto is##S() const -> bool { return kind_ == SymbolKind::k##S; }
  CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)
#undef PROCESS_SYMBOL

  [[nodiscard]] auto isClassOrNamespace() const -> bool {
    return isClass() || isNamespace();
  }

  [[nodiscard]] auto isEnumOrScopedEnum() const -> bool {
    return isEnum() || isScopedEnum();
  }

 private:
  friend class ScopeSymbol;

  SymbolKind kind_;
  const Name* name_ = nullptr;
  const Type* type_ = nullptr;
  ScopeSymbol* parent_ = nullptr;
  Symbol* link_ = nullptr;
  SourceLocation location_;
};

class ScopeSymbol : public Symbol {
 public:
  ScopeSymbol(SymbolKind kind, ScopeSymbol* enclosingScope);
  ~ScopeSymbol() override;

  [[nodiscard]] auto asScopeSymbol() -> ScopeSymbol* override { return this; }

  [[nodiscard]] auto empty() const -> bool { return members_.empty(); }

  [[nodiscard]] auto members() const -> const std::vector<Symbol*>&;
  void addMember(Symbol* member);

  [[nodiscard]] auto usingDirectives() const {
    return std::views::all(usingDirectives_);
  }

  [[nodiscard]] auto find(const Name* name) const -> SymbolChainView;

  [[nodiscard]] auto find(const std::string_view& name) const
      -> SymbolChainView;

  [[nodiscard]] auto find(TokenKind op) const -> SymbolChainView;

  void addSymbol(Symbol* symbol);
  void addUsingDirective(ScopeSymbol* scope);

  [[nodiscard]] auto isTransparent() const -> bool;

  // internal
  void replaceSymbol(Symbol* symbol, Symbol* newSymbol);
  void reset();

 private:
  void rehash();

 private:
  std::vector<Symbol*> members_;
  std::vector<Symbol*> buckets_;
  std::vector<ScopeSymbol*> usingDirectives_;
};

class NamespaceSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kNamespace;

  explicit NamespaceSymbol(ScopeSymbol* enclosingScope);
  ~NamespaceSymbol() override;

  [[nodiscard]] auto isInline() const -> bool;
  void setInline(bool isInline);

  [[nodiscard]] auto unnamedNamespace() const -> NamespaceSymbol*;
  void setUnnamedNamespace(NamespaceSymbol* unnamedNamespace);

  [[nodiscard]] auto anonNamespaceIndex() const -> std::optional<int>;
  void setAnonNamespaceIndex(int index);

 private:
  NamespaceSymbol* unnamedNamespace_ = nullptr;
  int anonNamespaceIndex_ = -1;
  bool isInline_ = false;
};

class ConceptSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kConcept;

  explicit ConceptSymbol(ScopeSymbol* enclosingScope);
  ~ConceptSymbol() override;

 private:
};

class BaseClassSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kBaseClass;

  explicit BaseClassSymbol(ScopeSymbol* enclosingScope);
  ~BaseClassSymbol() override;

  [[nodiscard]] auto isVirtual() const -> bool;
  void setVirtual(bool isVirtual);

  [[nodiscard]] auto accessSpecifier() const -> AccessSpecifier;
  void setAccessSpecifier(AccessSpecifier accessSpecifier);

  [[nodiscard]] auto symbol() const -> Symbol*;
  void setSymbol(Symbol* symbol);

 private:
  Symbol* symbol_ = nullptr;
  AccessSpecifier accessSpecifier_ = AccessSpecifier::kPublic;
  bool isVirtual_ = false;
};

class ClassSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kClass;

  explicit ClassSymbol(ScopeSymbol* enclosingScope);
  ~ClassSymbol() override;

  [[nodiscard]] auto isUnion() const -> bool;
  void setIsUnion(bool isUnion);

  [[nodiscard]] auto baseClasses() const
      -> const std::vector<BaseClassSymbol*>&;

  void addBaseClass(BaseClassSymbol* baseClass);

  [[nodiscard]] auto constructors() const
      -> const std::vector<FunctionSymbol*>&;

  void addConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto conversionFunctions() const
      -> const std::vector<FunctionSymbol*>&;

  void addConversionFunction(FunctionSymbol* conversionFunction);

  [[nodiscard]] auto isFinal() const -> bool;
  void setFinal(bool isFinal);

  [[nodiscard]] auto isComplete() const -> bool;
  void setComplete(bool isComplete);

  [[nodiscard]] auto sizeInBytes() const -> int;
  void setSizeInBytes(int sizeInBytes);

  [[nodiscard]] auto alignment() const -> int;
  void setAlignment(int alignment);

  [[nodiscard]] auto hasBaseClass(Symbol* symbol) const -> bool;

  [[nodiscard]] auto flags() const -> std::uint32_t;
  void setFlags(std::uint32_t flags);

  [[nodiscard]] auto declaration() const -> SpecifierAST*;
  void setDeclaration(SpecifierAST* ast);

  [[nodiscard]] auto templateDeclaration() const -> TemplateDeclarationAST*;
  void setTemplateDeclaration(TemplateDeclarationAST* templateDeclaration);

  [[nodiscard]] auto specializations() const
      -> std::span<const TemplateSpecialization<ClassSymbol>>;

  [[nodiscard]] auto findSpecialization(
      const std::vector<TemplateArgument>& arguments) const -> ClassSymbol*;

  void addSpecialization(std::vector<TemplateArgument> arguments,
                         ClassSymbol* specialization);

  [[nodiscard]] auto isSpecialization() const -> bool {
    return templateClass_ != nullptr;
  }

  [[nodiscard]] auto templateArguments() const
      -> std::span<const TemplateArgument> {
    if (!templateClass_) return {};
    return templateClass_->specializations()[templateSepcializationIndex_]
        .arguments;
  }

  void setSpecializationInfo(ClassSymbol* templateClass, std::size_t index) {
    templateClass_ = templateClass;
    templateSepcializationIndex_ = index;
  }

  [[nodiscard]] auto templateClass() const -> ClassSymbol* {
    return templateClass_;
  }

  [[nodiscard]] auto templateSepcializationIndex() const -> std::size_t {
    return templateSepcializationIndex_;
  }

  [[nodiscard]] auto buildClassLayout(Control* control)
      -> std::expected<bool, std::string>;

 private:
  [[nodiscard]] auto hasBaseClass(Symbol* symbol,
                                  std::unordered_set<const ClassSymbol*>&) const
      -> bool;

 private:
  std::vector<BaseClassSymbol*> baseClasses_;
  std::vector<FunctionSymbol*> constructors_;
  std::vector<FunctionSymbol*> conversionFunctions_;
  std::unique_ptr<TemplateInfo<ClassSymbol>> templateInfo_;
  SpecifierAST* specifier_ = nullptr;
  TemplateDeclarationAST* templateDeclaration_ = nullptr;
  ClassSymbol* templateClass_ = nullptr;
  std::size_t templateSepcializationIndex_ = 0;
  int sizeInBytes_ = 0;
  int alignment_ = 0;
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isUnion_ : 1;
      std::uint32_t isFinal_ : 1;
      std::uint32_t isComplete_ : 1;
    };
  };
};

class EnumSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kEnum;

  explicit EnumSymbol(ScopeSymbol* enclosingScope);
  ~EnumSymbol() override;

  [[nodiscard]] bool hasFixedUnderlyingType() const;
  void setHasFixedUnderlyingType(bool hasFixedUnderlyingType);

  [[nodiscard]] auto underlyingType() const -> const Type*;
  void setUnderlyingType(const Type* underlyingType);

 private:
  const Type* underlyingType_ = nullptr;
  bool hasFixedUnderlyingType_ = false;
};

class ScopedEnumSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kScopedEnum;

  explicit ScopedEnumSymbol(ScopeSymbol* enclosingScope);
  ~ScopedEnumSymbol() override;

  [[nodiscard]] auto underlyingType() const -> const Type*;
  void setUnderlyingType(const Type* underlyingType);

 private:
  const Type* underlyingType_ = nullptr;
};

class FunctionSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunction;

  explicit FunctionSymbol(ScopeSymbol* enclosingScope);
  ~FunctionSymbol() override;

  [[nodiscard]] auto isDefined() const -> bool;
  void setDefined(bool isDefined);

  [[nodiscard]] auto isStatic() const -> bool;
  void setStatic(bool isStatic);

  [[nodiscard]] auto isExtern() const -> bool;
  void setExtern(bool isExtern);

  [[nodiscard]] auto isFriend() const -> bool;
  void setFriend(bool isFriend);

  [[nodiscard]] auto isConstexpr() const -> bool;
  void setConstexpr(bool isConstexpr);

  [[nodiscard]] auto isConsteval() const -> bool;
  void setConsteval(bool isConsteval);

  [[nodiscard]] auto isInline() const -> bool;
  void setInline(bool isInline);

  [[nodiscard]] auto isVirtual() const -> bool;
  void setVirtual(bool isVirtual);

  [[nodiscard]] auto isExplicit() const -> bool;
  void setExplicit(bool isExplicit);

  [[nodiscard]] auto isDeleted() const -> bool;
  void setDeleted(bool isDeleted);

  [[nodiscard]] auto isDefaulted() const -> bool;
  void setDefaulted(bool isDefaulted);

  [[nodiscard]] auto isConstructor() const -> bool;

  [[nodiscard]] auto hasCLinkage() const -> bool;
  [[nodiscard]] auto hasCxxLinkage() const -> bool;
  void setHasCxxLinkage(bool hasCLinkage);

  [[nodiscard]] auto declaration() const -> DeclarationAST*;
  void setDeclaration(DeclarationAST* declaration);

 private:
  DeclarationAST* declaration_ = nullptr;

  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isDefined_ : 1;
      std::uint32_t isStatic_ : 1;
      std::uint32_t isExtern_ : 1;
      std::uint32_t isFriend_ : 1;
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConsteval_ : 1;
      std::uint32_t isInline_ : 1;
      std::uint32_t isVirtual_ : 1;
      std::uint32_t isExplicit_ : 1;
      std::uint32_t isDeleted_ : 1;
      std::uint32_t isDefaulted_ : 1;
      std::uint32_t isExternC_ : 1;
    };
  };
};

class OverloadSetSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kOverloadSet;

  explicit OverloadSetSymbol(ScopeSymbol* enclosingScope);
  ~OverloadSetSymbol() override;

  [[nodiscard]] auto functions() const -> const std::vector<FunctionSymbol*>&;

  void setFunctions(std::vector<FunctionSymbol*> functions);
  void addFunction(FunctionSymbol* function);

 private:
  std::vector<FunctionSymbol*> functions_;
};

class LambdaSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kLambda;

  explicit LambdaSymbol(ScopeSymbol* enclosingScope);
  ~LambdaSymbol() override;

  [[nodiscard]] auto isConstexpr() const -> bool;
  void setConstexpr(bool isConstexpr);

  [[nodiscard]] auto isConsteval() const -> bool;
  void setConsteval(bool isConsteval);

  [[nodiscard]] auto isMutable() const -> bool;
  void setMutable(bool isMutable);

  [[nodiscard]] auto isStatic() const -> bool;
  void setStatic(bool isStatic);

 private:
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConsteval_ : 1;
      std::uint32_t isMutable_ : 1;
      std::uint32_t isStatic_ : 1;
    };
  };
};

class FunctionParametersSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kFunctionParameters;

  explicit FunctionParametersSymbol(ScopeSymbol* enclosingScope);
  ~FunctionParametersSymbol() override;
};

class TemplateParametersSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateParameters;

  explicit TemplateParametersSymbol(ScopeSymbol* enclosingScope);
  ~TemplateParametersSymbol() override;

  [[nodiscard]] auto isExplicitTemplateSpecialization() const -> bool;
  void setExplicitTemplateSpecialization(bool isExplicit);

 private:
  bool isExplicitTemplateSpecialization_ = false;
};

class BlockSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kBlock;

  explicit BlockSymbol(ScopeSymbol* enclosingScope);
  ~BlockSymbol() override;
};

class TypeAliasSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTypeAlias;

  explicit TypeAliasSymbol(ScopeSymbol* enclosingScope);
  ~TypeAliasSymbol() override;

  [[nodiscard]] auto templateDeclaration() const -> TemplateDeclarationAST*;
  void setTemplateDeclaration(TemplateDeclarationAST* declaration);

 private:
  TemplateDeclarationAST* templateDeclaration_ = nullptr;
};

class VariableSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kVariable;

  explicit VariableSymbol(ScopeSymbol* enclosingScope);
  ~VariableSymbol() override;

  [[nodiscard]] auto isStatic() const -> bool;
  void setStatic(bool isStatic);

  [[nodiscard]] auto isThreadLocal() const -> bool;
  void setThreadLocal(bool isThreadLocal);

  [[nodiscard]] auto isExtern() const -> bool;
  void setExtern(bool isExtern);

  [[nodiscard]] auto isConstexpr() const -> bool;
  void setConstexpr(bool isConstexpr);

  [[nodiscard]] auto isConstinit() const -> bool;
  void setConstinit(bool isConstinit);

  [[nodiscard]] auto isInline() const -> bool;
  void setInline(bool isInline);

  [[nodiscard]] auto templateDeclaration() const -> TemplateDeclarationAST*;
  void setTemplateDeclaration(TemplateDeclarationAST* declaration);

  [[nodiscard]] auto initializer() const -> ExpressionAST*;
  void setInitializer(ExpressionAST*);

  [[nodiscard]] auto constValue() const -> const std::optional<ConstValue>&;
  void setConstValue(std::optional<ConstValue> value);

 private:
  TemplateDeclarationAST* templateDeclaration_ = nullptr;
  ExpressionAST* initializer_ = nullptr;
  std::optional<ConstValue> constValue_;

  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isStatic_ : 1;
      std::uint32_t isThreadLocal_ : 1;
      std::uint32_t isExtern_ : 1;
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConstinit_ : 1;
      std::uint32_t isInline_ : 1;
    };
  };
};

class FieldSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kField;

  explicit FieldSymbol(ScopeSymbol* enclosingScope);
  ~FieldSymbol() override;

  [[nodiscard]] bool isBitField() const;
  void setBitField(bool isBitField);

  [[nodiscard]] auto bitFieldOffset() const -> int;
  void setBitFieldOffset(int bitFieldOffset);

  [[nodiscard]] auto bitFieldWidth() const -> const std::optional<ConstValue>&;
  void setBitFieldWidth(std::optional<ConstValue> bitFieldWidth);

  [[nodiscard]] auto isExtern() const -> bool;

  [[nodiscard]] auto isStatic() const -> bool;
  void setStatic(bool isStatic);

  [[nodiscard]] auto isThreadLocal() const -> bool;
  void setThreadLocal(bool isThreadLocal);

  [[nodiscard]] auto isConstexpr() const -> bool;
  void setConstexpr(bool isConstexpr);

  [[nodiscard]] auto isConstinit() const -> bool;
  void setConstinit(bool isConstinit);

  [[nodiscard]] auto isInline() const -> bool;
  void setInline(bool isInline);

  [[nodiscard]] auto isMutable() const -> bool;
  void setMutable(bool isMutable);

  [[nodiscard]] auto offset() const -> int;
  void setOffset(int offset);

  [[nodiscard]] auto alignment() const -> int;
  void setAlignment(int alignment);

 private:
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isBitField_ : 1;
      std::uint32_t isStatic_ : 1;
      std::uint32_t isThreadLocal_ : 1;
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConstinit_ : 1;
      std::uint32_t isInline_ : 1;
      std::uint32_t isMutable_ : 1;
    };
  };
  int offset_{};
  int alignment_{};
  int bitFieldOffset_{};
  std::optional<ConstValue> bitFieldWidth_;
};

class ParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kParameter;

  explicit ParameterSymbol(ScopeSymbol* enclosingScope);
  ~ParameterSymbol() override;
};

class ParameterPackSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kParameterPack;

  explicit ParameterPackSymbol(ScopeSymbol* enclosingScope);
  ~ParameterPackSymbol() override;

  [[nodiscard]] auto elements() const -> const std::vector<Symbol*>&;
  void addElement(Symbol* element);

 private:
  std::vector<Symbol*> elements_;
};

class TypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTypeParameter;

  explicit TypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~TypeParameterSymbol() override;
};

class NonTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kNonTypeParameter;

  explicit NonTypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~NonTypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int;
  void setIndex(int index);

  [[nodiscard]] auto depth() const -> int;
  void setDepth(int depth);

  [[nodiscard]] auto objectType() const -> const Type*;
  void setObjectType(const Type* objectType);

  [[nodiscard]] auto isParameterPack() const -> bool;
  void setParameterPack(bool isParameterPack);

 private:
  const Type* objectType_ = nullptr;
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class TemplateTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateTypeParameter;

  explicit TemplateTypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~TemplateTypeParameterSymbol() override;
};

class ConstraintTypeParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kConstraintTypeParameter;

  explicit ConstraintTypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~ConstraintTypeParameterSymbol() override;

  [[nodiscard]] auto index() const -> int;
  void setIndex(int index);

  [[nodiscard]] auto depth() const -> int;
  void setDepth(int depth);

  [[nodiscard]] auto isParameterPack() const -> bool;
  void setParameterPack(bool isParameterPack);

 private:
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
};

class EnumeratorSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kEnumerator;

  explicit EnumeratorSymbol(ScopeSymbol* enclosingScope);
  ~EnumeratorSymbol() override;

  [[nodiscard]] auto value() const -> const std::optional<ConstValue>&;
  void setValue(const std::optional<ConstValue>& value);

 private:
  std::optional<ConstValue> value_;
};

class UsingDeclarationSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kUsingDeclaration;

  explicit UsingDeclarationSymbol(ScopeSymbol* enclosingScope);
  ~UsingDeclarationSymbol() override;

  [[nodiscard]] auto declarator() const -> UsingDeclaratorAST*;
  void setDeclarator(UsingDeclaratorAST* declarator);

  [[nodiscard]] auto target() const -> Symbol*;
  void setTarget(Symbol* symbol);

 private:
  Symbol* target_ = nullptr;
  UsingDeclaratorAST* declarator_ = nullptr;
};

bool is_type(Symbol* symbol);

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

template <>
inline auto symbol_cast(Symbol* symbol) -> ScopeSymbol* {
  if (symbol) return symbol->asScopeSymbol();
  return nullptr;
}

}  // namespace cxx
