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
#include <cxx/const_value.h>
#include <cxx/diagnostic.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cxx {
class SymbolChainView;

class TemplateSpecialization {
 public:
  Symbol* templateSymbol = nullptr;
  std::vector<TemplateArgument> arguments;
  Symbol* symbol = nullptr;
  std::vector<Diagnostic> instantiationErrors;
  List<TemplateArgumentAST*>* pendingArgumentList = nullptr;
  SourceLocation pendingInstantiationLoc;
  bool isPendingInstantiation = false;
};

struct PendingBodyInstantiation {
  FunctionDefinitionAST* originalDefinition = nullptr;
  std::vector<TemplateArgument> templateArguments;
  ScopeSymbol* parentScope = nullptr;
  int depth = 0;
};

[[nodiscard]] auto compare_args(const std::vector<TemplateArgument>& args1,
                                const std::vector<TemplateArgument>& args2)
    -> bool;

[[nodiscard]] auto expand_template_arguments(
    std::span<const TemplateArgument> arguments)
    -> std::vector<TemplateArgument>;

[[nodiscard]] auto template_argument_type(const TemplateArgument& argument)
    -> const Type*;

[[nodiscard]] auto template_argument_value(const TemplateArgument& argument)
    -> std::optional<ConstValue>;

[[nodiscard]] auto template_name_symbol(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto templated_symbol(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto template_declaration_of(Symbol* symbol)
    -> TemplateDeclarationAST*;

[[nodiscard]] auto template_parameters_of(Symbol* symbol)
    -> TemplateParametersSymbol*;

[[nodiscard]] auto template_declaration_ast(Symbol* symbol) -> AST*;

[[nodiscard]] auto is_member_template(Symbol* symbol) -> bool;

[[nodiscard]] auto template_parameter_info(Symbol* symbol)
    -> std::optional<TypeParamInfo>;

template <typename S>
class MaybeRedecl {
 public:
  [[nodiscard]] auto canonical() const -> S* {
    return canonical_ ? canonical_
                      : const_cast<S*>(static_cast<const S*>(this));
  }

  void setCanonical(S* canonical) { canonical_ = canonical; }

  [[nodiscard]] auto definition() const -> S* { return definition_; }

  [[nodiscard]] auto resolvedDefinition() const -> S* {
    return definition_ ? definition_
                       : const_cast<S*>(static_cast<const S*>(this));
  }

  void setDefinition(S* definition) { definition_ = definition; }

  [[nodiscard]] auto redeclarations() const -> const std::vector<S*>& {
    return redeclarations_;
  }

  void addRedeclaration(S* redecl) {
    auto self = static_cast<S*>(this);
    if (!redecl || redecl == self) cxx_runtime_error("invalid redeclaration");
    if (canonical_)
      cxx_runtime_error("addRedeclaration called on non-canonical symbol");
    if (std::ranges::find(redeclarations_, redecl) != redeclarations_.end())
      cxx_runtime_error("duplicate redeclaration");
    redecl->setCanonical(self);
    redeclarations_.push_back(redecl);
  }

 private:
  S* canonical_ = nullptr;
  S* definition_ = nullptr;
  std::vector<S*> redeclarations_;
};

template <typename S, typename D>
class MaybeTemplate {
  struct Template {
    std::vector<TemplateSpecialization> specializations_;
    TemplateDeclarationAST* templateDeclaration_ = nullptr;
    TemplateParametersSymbol* templateParameters_ = nullptr;
    std::vector<std::vector<TemplateArgument>> externInstantiationDeclarations_;
  };

  struct TemplateSpecializationRef {
    S* primaryTemplateSymbol_ = nullptr;
    int templateSepcializationIndex_ = 0;
  };

  struct TemplateData : Template, TemplateSpecializationRef {};

 public:
  [[nodiscard]] auto findSpecialization(
      const std::vector<TemplateArgument>& arguments) const -> Symbol* {
    for (const auto& specialization : specializations()) {
      const std::vector<TemplateArgument>& args = specialization.arguments;
      if (args == arguments) return specialization.symbol;
      if (args.size() != arguments.size()) continue;
      if (compare_args(args, arguments)) {
        return specialization.symbol;
      }
    }
    return nullptr;
  }

  [[nodiscard]] auto templateDeclaration() const -> TemplateDeclarationAST* {
    if (!template_) return nullptr;
    return template_->templateDeclaration_;
  }

  void setTemplateDeclaration(TemplateDeclarationAST* templateDeclaration) {
    ensure_template();
    template_->templateDeclaration_ = templateDeclaration;
  }

  [[nodiscard]] auto templateParameters() const -> TemplateParametersSymbol* {
    if (!template_) return nullptr;
    return template_->templateParameters_;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    ensure_template();
    template_->templateParameters_ = templateParameters;
  }

  [[nodiscard]] auto isSpecialization() const -> bool {
    if (!template_) return false;
    return template_->primaryTemplateSymbol_ != nullptr;
  }

  void addSpecialization(std::vector<TemplateArgument> arguments,
                         S* specialization) {
    ensure_template();

    for (std::size_t i = 0; i < template_->specializations_.size(); ++i) {
      const auto& existing = template_->specializations_[i];
      if (existing.symbol != specialization) continue;
      if (existing.arguments.size() != arguments.size()) continue;
      if (existing.arguments != arguments &&
          !compare_args(existing.arguments, arguments))
        continue;
      specialization->setSpecializationInfo(static_cast<S*>(this), i);
      return;
    }

    auto index = int(template_->specializations_.size());

    specialization->setSpecializationInfo(static_cast<S*>(this), index);

    template_->specializations_.push_back({template_->primaryTemplateSymbol_,
                                           std::move(arguments),
                                           specialization});
  }

  [[nodiscard]] auto declaration() const -> D* { return declaration_; }

  void setDeclaration(D* ast) { declaration_ = ast; }

  [[nodiscard]] auto specializations() const
      -> std::span<const TemplateSpecialization> {
    if (!template_) return {};
    return template_->specializations_;
  }

  [[nodiscard]] auto mutableSpecializations()
      -> std::span<TemplateSpecialization> {
    if (!template_) return {};
    return template_->specializations_;
  }
  [[nodiscard]] auto templateArguments() const
      -> std::span<const TemplateArgument> {
    if (!template_) return {};
    if (!template_->primaryTemplateSymbol_) return {};

    return template_->primaryTemplateSymbol_
        ->specializations()[template_->templateSepcializationIndex_]
        .arguments;
  }

  void addExternInstantiationDeclaration(
      std::vector<TemplateArgument> arguments) {
    ensure_template();
    template_->externInstantiationDeclarations_.push_back(std::move(arguments));
  }

  [[nodiscard]] auto isExternInstantiationDeclared(
      std::span<const TemplateArgument> arguments) const -> bool {
    if (!template_) return false;
    for (const auto& args : template_->externInstantiationDeclarations_) {
      if (args.size() != arguments.size()) continue;
      if (std::equal(args.begin(), args.end(), arguments.begin()) ||
          compare_args(args, std::vector<TemplateArgument>(arguments.begin(),
                                                           arguments.end())))
        return true;
    }
    return false;
  }

  [[nodiscard]] auto isExplicitInstantiationDeclared() const -> bool {
    if (!isSpecialization()) return false;
    auto primary = primaryTemplateSymbol();
    if (!primary) return false;
    return primary->isExternInstantiationDeclared(templateArguments());
  }

  [[nodiscard]] auto primaryTemplateSymbol() const -> S* {
    if (!template_) return nullptr;
    return template_->primaryTemplateSymbol_;
  }

 private:
  void setSpecializationInfo(S* primaryTemplateSymbol, std::size_t index) {
    ensure_template();
    template_->primaryTemplateSymbol_ = primaryTemplateSymbol;
    template_->templateSepcializationIndex_ = index;
  }

  [[nodiscard]] auto templateSepcializationIndex() const -> std::size_t {
    if (!template_) return 0;
    return template_->templateSepcializationIndex_;
  }

 private:
  void ensure_template() {
    if (template_) return;
    template_ = std::make_unique<TemplateData>();
  }

  std::unique_ptr<TemplateData> template_;
  D* declaration_ = nullptr;
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
      : kind_(kind), parent_(nullptr) {
    setParent(enclosingScope);
  }

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

  [[nodiscard]] auto enclosingFunction() const -> FunctionSymbol*;

  [[nodiscard]] auto hasEnclosingSymbol(Symbol* symbol) const -> bool;

  [[nodiscard]] auto next() const -> Symbol*;

  [[nodiscard]] auto isHidden() const -> bool { return isHidden_; }
  void setHidden(bool isHidden) { isHidden_ = isHidden; }

  [[nodiscard]] auto abiTags() const -> std::span<const Identifier* const>;

  [[nodiscard]] auto abiTagList() const
      -> const std::vector<const Identifier*>* {
    return abiTags_;
  }

  void setAbiTags(const std::vector<const Identifier*>* abiTags);

  [[nodiscard]] auto canonical() const -> Symbol*;

  [[nodiscard]] auto definition() const -> Symbol*;

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
  const std::vector<const Identifier*>* abiTags_ = nullptr;
  SourceLocation location_;
  bool isHidden_ = false;
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

  [[nodiscard]] auto hasInlineNamespaces() const -> bool;
  void setHasInlineNamespaces(bool value);

  [[nodiscard]] auto unnamedNamespace() const -> NamespaceSymbol*;
  void setUnnamedNamespace(NamespaceSymbol* unnamedNamespace);

  [[nodiscard]] auto anonNamespaceIndex() const -> std::optional<int>;
  void setAnonNamespaceIndex(int index);

 private:
  NamespaceSymbol* unnamedNamespace_ = nullptr;
  int anonNamespaceIndex_ = -1;
  bool isInline_ = false;
  bool hasInlineNamespaces_ = false;
};

class ConceptSymbol final
    : public Symbol,
      public MaybeTemplate<ConceptSymbol, ConceptDefinitionAST> {
 public:
  constexpr static auto Kind = SymbolKind::kConcept;

  explicit ConceptSymbol(ScopeSymbol* enclosingScope);
  ~ConceptSymbol() override;

 private:
};

class DeductionGuideSymbol final
    : public Symbol,
      public MaybeTemplate<DeductionGuideSymbol, DeductionGuideAST> {
 public:
  constexpr static auto Kind = SymbolKind::kDeductionGuide;

  explicit DeductionGuideSymbol(ScopeSymbol* enclosingScope);
  ~DeductionGuideSymbol() override;

  [[nodiscard]] auto isExplicit() const -> bool;
  void setExplicit(bool isExplicit);

 private:
  bool isExplicit_ = false;
};

class ClassLayout {
 public:
  struct MemberInfo {
    std::uint64_t offset = 0;
    std::uint32_t index = 0;
    std::uint32_t bitOffset = 0;
    std::uint32_t bitWidth = 0;
    std::uint32_t allocUnitSizeBytes = 0;
  };

  ClassLayout() = default;

  [[nodiscard]] auto getFieldInfo(FieldSymbol* field) const
      -> std::optional<MemberInfo>;

  [[nodiscard]] auto getBaseInfo(ClassSymbol* base) const
      -> std::optional<MemberInfo>;

  void setFieldInfo(FieldSymbol* field, const MemberInfo& info);
  void setBaseInfo(ClassSymbol* base, const MemberInfo& info);

  void addVirtualBase(ClassSymbol* base) { virtualBases_.push_back(base); }
  [[nodiscard]] auto virtualBases() const -> const std::vector<ClassSymbol*>& {
    return virtualBases_;
  }

  void setSize(std::uint64_t size) { size_ = size; }
  void setAlignment(std::uint64_t alignment) { alignment_ = alignment; }

  void setNonVirtualSize(std::uint64_t size) { nonVirtualSize_ = size; }
  void setNonVirtualAlignment(std::uint64_t alignment) {
    nonVirtualAlignment_ = alignment;
  }
  [[nodiscard]] auto nonVirtualSize() const -> std::uint64_t {
    return nonVirtualSize_;
  }
  [[nodiscard]] auto nonVirtualAlignment() const -> std::uint64_t {
    return nonVirtualAlignment_;
  }
  void setHasVtable(bool hasVtable) { hasVtable_ = hasVtable; }
  void setHasDirectVtable(bool hasDirectVtable) {
    hasDirectVtable_ = hasDirectVtable;
  }
  void setVtableIndex(std::uint32_t vtableIndex) { vtableIndex_ = vtableIndex; }

  [[nodiscard]] auto size() const -> std::uint64_t { return size_; }
  [[nodiscard]] auto alignment() const -> std::uint64_t { return alignment_; }

  [[nodiscard]] auto empty() const -> bool {
    return fields_.empty() && bases_.empty();
  }

  [[nodiscard]] auto hasVtable() const -> bool { return hasVtable_; }
  [[nodiscard]] auto vtableIndex() const -> std::uint32_t {
    return vtableIndex_;
  }
  [[nodiscard]] auto hasDirectVtable() const -> bool {
    return hasDirectVtable_;
  }

 private:
  std::unordered_map<FieldSymbol*, MemberInfo> fields_;
  std::unordered_map<ClassSymbol*, MemberInfo> bases_;
  std::vector<ClassSymbol*> virtualBases_;
  std::uint64_t size_ = 0;
  std::uint64_t alignment_ = 1;
  std::uint64_t nonVirtualSize_ = 0;
  std::uint64_t nonVirtualAlignment_ = 1;
  std::uint32_t vtableIndex_ = 0;
  bool hasVtable_ = false;
  bool hasDirectVtable_ = false;
};

class VTableLayout {
 public:
  enum class SlotKind : std::uint8_t {
    kFunction,
    kCompleteDtor,
    kDeletingDtor,
  };

  struct Slot {
    FunctionSymbol* function = nullptr;
    SlotKind kind = SlotKind::kFunction;
    std::uint64_t thisAdjustment = 0;
    int vcallOffsetIndex = -1;
  };

  struct Group {
    ClassSymbol* base = nullptr;
    std::uint64_t offset = 0;
    std::vector<std::pair<ClassSymbol*, std::int64_t>> vbaseOffsets;
    std::vector<std::pair<FunctionSymbol*, std::int64_t>> vcallOffsets;
    std::vector<Slot> slots;

    [[nodiscard]] auto headerWordCount() const -> std::size_t {
      return vbaseOffsets.size() + vcallOffsets.size() + 2;
    }

    [[nodiscard]] auto wordCount() const -> std::size_t {
      return headerWordCount() + slots.size();
    }
  };

  Group primary;
  std::vector<Group> secondary;
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

class InjectedClassNameSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kInjectedClassName;

  explicit InjectedClassNameSymbol(ScopeSymbol* enclosingScope);
  ~InjectedClassNameSymbol() override;

  [[nodiscard]] auto classSymbol() const -> ClassSymbol* {
    return classSymbol_;
  }
  void setClassSymbol(ClassSymbol* classSymbol) { classSymbol_ = classSymbol; }

 private:
  ClassSymbol* classSymbol_ = nullptr;
};

class UnresolvedSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kUnresolved;

  explicit UnresolvedSymbol(ScopeSymbol* enclosingScope);
  ~UnresolvedSymbol() override;
};

class ClassSymbol final : public ScopeSymbol,
                          public MaybeTemplate<ClassSymbol, SpecifierAST>,
                          public MaybeRedecl<ClassSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kClass;

  using MaybeRedecl<ClassSymbol>::canonical;
  using MaybeRedecl<ClassSymbol>::setCanonical;
  using MaybeRedecl<ClassSymbol>::definition;
  using MaybeRedecl<ClassSymbol>::resolvedDefinition;
  using MaybeRedecl<ClassSymbol>::setDefinition;
  using MaybeRedecl<ClassSymbol>::redeclarations;
  using MaybeRedecl<ClassSymbol>::addRedeclaration;

  void setInstantiationSubstitution(int depth,
                                    std::vector<TemplateArgument> arguments) {
    instantiationSubstitutionDepth_ = depth;
    instantiationSubstitutionArguments_ = std::move(arguments);
  }

  [[nodiscard]] auto instantiationSubstitutionDepth() const -> int {
    return instantiationSubstitutionDepth_;
  }

  [[nodiscard]] auto instantiationSubstitutionArguments() const
      -> const std::vector<TemplateArgument>& {
    return instantiationSubstitutionArguments_;
  }

  explicit ClassSymbol(ScopeSymbol* enclosingScope);
  ~ClassSymbol() override;

  [[nodiscard]] auto isUnion() const -> bool;
  void setIsUnion(bool isUnion);

  [[nodiscard]] auto baseClasses() const
      -> const std::vector<BaseClassSymbol*>&;

  void addBaseClass(BaseClassSymbol* baseClass);

  [[nodiscard]] auto constructors() const -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto declaredConstructors() const
      -> const std::vector<FunctionSymbol*>&;

  void addConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto constructorOverloadSet() const -> OverloadSetSymbol* {
    return constructorOverloadSet_;
  }

  void setConstructorOverloadSet(OverloadSetSymbol* overloadSet) {
    constructorOverloadSet_ = overloadSet;
  }

  [[nodiscard]] auto deductionGuides() const
      -> const std::vector<DeductionGuideSymbol*>&;

  void addDeductionGuide(DeductionGuideSymbol* guide);

  [[nodiscard]] auto conversionFunctions() const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto implicitConversionFunctions() const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto destructor() const -> FunctionSymbol*;
  [[nodiscard]] auto defaultConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto copyConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto moveConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto copyAssignmentOperator() const -> FunctionSymbol*;
  [[nodiscard]] auto moveAssignmentOperator() const -> FunctionSymbol*;
  [[nodiscard]] auto hasUserDeclaredConstructors() const -> bool;

  [[nodiscard]] auto hasInheritedConstructors() const -> bool;
  [[nodiscard]] auto hasVirtualFunctions() const -> bool;
  [[nodiscard]] auto hasVirtualBaseClasses() const -> bool;

  [[nodiscard]] auto convertingConstructors() const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto isFinal() const -> bool;
  void setFinal(bool isFinal);

  [[nodiscard]] auto isComplete() const -> bool;
  void setComplete(bool isComplete);

  [[nodiscard]] auto isFriend() const -> bool;
  void setFriend(bool isFriend);

  [[nodiscard]] auto isPolymorphic() const -> bool;
  void setPolymorphic(bool isPolymorphic);

  [[nodiscard]] auto isAbstract() const -> bool;
  void setAbstract(bool isAbstract);

  [[nodiscard]] auto hasVirtualDestructor() const -> bool;
  void setHasVirtualDestructor(bool hasVirtualDestructor);

  [[nodiscard]] auto sizeInBytes() const -> int;
  void setSizeInBytes(int sizeInBytes);

  [[nodiscard]] auto alignment() const -> int;
  void setAlignment(int alignment);

  [[nodiscard]] auto hasBaseClass(Symbol* symbol) const -> bool;

  [[nodiscard]] auto baseClassOffset(ClassSymbol* base) const
      -> std::optional<std::uint64_t>;

  [[nodiscard]] auto hasVirtualBasePath(Symbol* symbol) const -> bool;

  [[nodiscard]] auto flags() const -> std::uint32_t;
  void setFlags(std::uint32_t flags);

  void setLayout(std::unique_ptr<ClassLayout> layout);

  [[nodiscard]] auto layout() const -> const ClassLayout*;

  void setVTableLayout(std::unique_ptr<VTableLayout> vtableLayout);

  [[nodiscard]] auto vtableLayout() const -> const VTableLayout*;

  [[nodiscard]] auto isClosureType() const -> bool;
  void setIsClosureType(bool isClosureType);

  [[nodiscard]] auto capturedThisField() const -> FieldSymbol*;
  void setCapturedThisField(FieldSymbol* capturedThisField);

  [[nodiscard]] auto closureDiscriminator() const -> int;
  void setClosureDiscriminator(int closureDiscriminator);

 private:
  [[nodiscard]] auto hasBaseClass(Symbol* symbol,
                                  std::unordered_set<const ClassSymbol*>&) const
      -> bool;

  [[nodiscard]] auto hasVirtualBasePath(
      Symbol* symbol, std::unordered_set<const ClassSymbol*>&) const -> bool;

 private:
  std::vector<BaseClassSymbol*> baseClasses_;
  std::vector<TemplateArgument> instantiationSubstitutionArguments_;
  int instantiationSubstitutionDepth_ = -1;
  OverloadSetSymbol* constructorOverloadSet_ = nullptr;
  std::vector<DeductionGuideSymbol*> deductionGuides_;
  std::unique_ptr<ClassLayout> layout_;
  std::unique_ptr<VTableLayout> vtableLayout_;
  FieldSymbol* capturedThisField_ = nullptr;
  int closureDiscriminator_ = 0;
  int sizeInBytes_ = 0;
  int alignment_ = 0;
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isUnion_ : 1;
      std::uint32_t isFinal_ : 1;
      std::uint32_t isComplete_ : 1;
      std::uint32_t isFriend_ : 1;
      std::uint32_t isPolymorphic_ : 1;
      std::uint32_t isAbstract_ : 1;
      std::uint32_t hasVirtualDestructor_ : 1;
      std::uint32_t isClosureType_ : 1;
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

  [[nodiscard]] auto isDefined() const -> bool;
  void setDefined(bool isDefined);

  [[nodiscard]] auto underlyingType() const -> const Type*;
  void setUnderlyingType(const Type* underlyingType);

 private:
  const Type* underlyingType_ = nullptr;
  bool hasFixedUnderlyingType_ = false;
  bool isDefined_ = false;
};

class ScopedEnumSymbol final : public ScopeSymbol {
 public:
  constexpr static auto Kind = SymbolKind::kScopedEnum;

  explicit ScopedEnumSymbol(ScopeSymbol* enclosingScope);
  ~ScopedEnumSymbol() override;

  [[nodiscard]] auto underlyingType() const -> const Type*;
  void setUnderlyingType(const Type* underlyingType);

  [[nodiscard]] auto isDefined() const -> bool;
  void setDefined(bool isDefined);

 private:
  const Type* underlyingType_ = nullptr;
  bool isDefined_ = false;
};

class FunctionSymbol final
    : public ScopeSymbol,
      public MaybeTemplate<FunctionSymbol, FunctionDefinitionAST>,
      public MaybeRedecl<FunctionSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kFunction;

  using MaybeRedecl<FunctionSymbol>::canonical;
  using MaybeRedecl<FunctionSymbol>::setCanonical;
  using MaybeRedecl<FunctionSymbol>::definition;
  using MaybeRedecl<FunctionSymbol>::resolvedDefinition;
  using MaybeRedecl<FunctionSymbol>::setDefinition;
  using MaybeRedecl<FunctionSymbol>::redeclarations;
  using MaybeRedecl<FunctionSymbol>::addRedeclaration;

  explicit FunctionSymbol(ScopeSymbol* enclosingScope);
  ~FunctionSymbol() override;

  [[nodiscard]] auto functionParameters() const -> FunctionParametersSymbol*;

  [[nodiscard]] auto isDefined() const -> bool;
  void setDefined(bool isDefined);

  [[nodiscard]] auto isStatic() const -> bool;
  void setStatic(bool isStatic);

  [[nodiscard]] auto isExtern() const -> bool;
  void setExtern(bool isExtern);

  [[nodiscard]] auto isFriend() const -> bool;
  void setFriend(bool isFriend);

  [[nodiscard]] auto isImplicitObjectMemberFunction() const -> bool;

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

  [[nodiscard]] auto isPure() const -> bool;
  void setPure(bool isPure);

  [[nodiscard]] auto isOverride() const -> bool;
  void setOverride(bool isOverride);

  [[nodiscard]] auto isFinal() const -> bool;
  void setFinal(bool isFinal);

  [[nodiscard]] auto hasNoPrototype() const -> bool;
  void setNoPrototype(bool hasNoPrototype);

  [[nodiscard]] auto hasExceptionSpecifier() const -> bool;
  void setExceptionSpecifier(bool hasExceptionSpecifier);

  [[nodiscard]] auto isDefinitionRequired() const -> bool;
  void setDefinitionRequired(bool isDefinitionRequired);

  [[nodiscard]] auto trailingRequiresClause() const -> RequiresClauseAST*;
  void setTrailingRequiresClause(RequiresClauseAST* requiresClause);

  [[nodiscard]] auto isConstructor() const -> bool;
  [[nodiscard]] auto isDestructor() const -> bool;

  [[nodiscard]] auto languageLinkage() const -> LanguageKind;
  void setLanguageLinkage(LanguageKind linkage);

  [[nodiscard]] auto hasCLinkage() const -> bool;

  [[nodiscard]] auto externalName() const -> const Identifier*;
  void setExternalName(const Identifier* externalName);

  [[nodiscard]] auto aliasName() const -> const Identifier*;
  void setAliasName(const Identifier* aliasName);

  [[nodiscard]] auto hasHiddenVisibility() const -> bool;
  void setHiddenVisibility(bool hasHiddenVisibility);

  [[nodiscard]] auto hasPendingBody() const -> bool;
  [[nodiscard]] auto pendingBody() const -> PendingBodyInstantiation*;
  void setPendingBody(std::unique_ptr<PendingBodyInstantiation> pending);
  void clearPendingBody();

  [[nodiscard]] auto vtableSlotIndex() const -> int { return vtableSlotIndex_; }
  void setVtableSlotIndex(int index) { vtableSlotIndex_ = index; }

  [[nodiscard]] auto delegatingConstructor() const -> FunctionSymbol* {
    return delegatingConstructor_;
  }
  void setDelegatingConstructor(FunctionSymbol* target) {
    delegatingConstructor_ = target;
  }

  [[nodiscard]] auto completeObjectVariant() const -> FunctionSymbol* {
    return completeObjectVariant_;
  }
  void setCompleteObjectVariant(FunctionSymbol* variant) {
    completeObjectVariant_ = variant;
  }

  [[nodiscard]] auto deletingDtorVariant() const -> FunctionSymbol* {
    return deletingDtorVariant_;
  }
  void setDeletingDtorVariant(FunctionSymbol* variant) {
    deletingDtorVariant_ = variant;
  }

  [[nodiscard]] auto structorPrincipal() const -> FunctionSymbol* {
    return structorPrincipal_;
  }
  void setStructorPrincipal(FunctionSymbol* principal) {
    structorPrincipal_ = principal;
  }

  [[nodiscard]] auto isStructorVariant() const -> bool {
    return structorPrincipal_ != nullptr;
  }

  [[nodiscard]] auto inheritedConstructor() const -> FunctionSymbol* {
    return inheritedConstructor_;
  }
  [[nodiscard]] auto inheritedConstructorOrigin() const -> FunctionSymbol* {
    auto origin = inheritedConstructor_;
    while (origin && origin->inheritedConstructor_)
      origin = origin->inheritedConstructor_;
    return origin;
  }
  void setInheritedConstructor(FunctionSymbol* constructor) {
    inheritedConstructor_ = constructor;
  }

  [[nodiscard]] auto isDeletingDtorVariant() const -> bool {
    return structorPrincipal_ &&
           structorPrincipal_->deletingDtorVariant() == this;
  }

 private:
  std::unique_ptr<PendingBodyInstantiation> pendingBody_;
  FunctionSymbol* completeObjectVariant_ = nullptr;
  FunctionSymbol* delegatingConstructor_ = nullptr;
  FunctionSymbol* deletingDtorVariant_ = nullptr;
  FunctionSymbol* structorPrincipal_ = nullptr;
  FunctionSymbol* inheritedConstructor_ = nullptr;
  int vtableSlotIndex_ = -1;
  const Identifier* externalName_ = nullptr;
  const Identifier* aliasName_ = nullptr;
  RequiresClauseAST* trailingRequiresClause_ = nullptr;
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
      std::uint32_t isPure_ : 1;
      std::uint32_t hasCLinkage_ : 1;
      std::uint32_t isOverride_ : 1;
      std::uint32_t isFinal_ : 1;
      std::uint32_t hasNoPrototype_ : 1;
      std::uint32_t hasHiddenVisibility_ : 1;
      std::uint32_t hasExceptionSpecifier_ : 1;
      std::uint32_t isDefinitionRequired_ : 1;
    };
  };
};

class OverloadSetSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kOverloadSet;

  explicit OverloadSetSymbol(ScopeSymbol* enclosingScope);
  ~OverloadSetSymbol() override;

  [[nodiscard]] auto functions() const -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto declaredFunctions() const
      -> const std::vector<FunctionSymbol*>&;

  void setFunctions(std::vector<FunctionSymbol*> functions);
  void addFunction(FunctionSymbol* function);

  [[nodiscard]] auto usingDeclarations() const
      -> const std::vector<UsingDeclarationSymbol*>&;

  void addUsingDeclaration(UsingDeclarationSymbol* usingDeclaration);

 private:
  std::vector<FunctionSymbol*> declaredFunctions_;
  std::vector<UsingDeclarationSymbol*> usingDeclarations_;
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

  [[nodiscard]] auto isTemplate() const -> bool;
  void setTemplate(bool isTemplate);

  [[nodiscard]] auto isInTemplate() const -> bool;
  void setInTemplate(bool isInTemplate);

 private:
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConsteval_ : 1;
      std::uint32_t isMutable_ : 1;
      std::uint32_t isStatic_ : 1;
      std::uint32_t isTemplate_ : 1;
      std::uint32_t isInTemplate_ : 1;
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

class TypeAliasSymbol final
    : public Symbol,
      public MaybeTemplate<TypeAliasSymbol, AliasDeclarationAST>,
      public MaybeRedecl<TypeAliasSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kTypeAlias;

  using MaybeRedecl<TypeAliasSymbol>::canonical;
  using MaybeRedecl<TypeAliasSymbol>::setCanonical;
  using MaybeRedecl<TypeAliasSymbol>::definition;
  using MaybeRedecl<TypeAliasSymbol>::resolvedDefinition;
  using MaybeRedecl<TypeAliasSymbol>::setDefinition;
  using MaybeRedecl<TypeAliasSymbol>::redeclarations;
  using MaybeRedecl<TypeAliasSymbol>::addRedeclaration;

  explicit TypeAliasSymbol(ScopeSymbol* enclosingScope);
  ~TypeAliasSymbol() override;

  [[nodiscard]] auto expansionTypeId() const -> TypeIdAST* {
    return expansionTypeId_;
  }

  void setExpansionTypeId(TypeIdAST* typeId) { expansionTypeId_ = typeId; }

 private:
  TemplateDeclarationAST* templateDeclaration_ = nullptr;
  TypeIdAST* expansionTypeId_ = nullptr;
};

class VariableSymbol final
    : public Symbol,
      public MaybeTemplate<VariableSymbol, SimpleDeclarationAST>,
      public MaybeRedecl<VariableSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kVariable;

  using MaybeRedecl<VariableSymbol>::canonical;
  using MaybeRedecl<VariableSymbol>::setCanonical;
  using MaybeRedecl<VariableSymbol>::definition;
  using MaybeRedecl<VariableSymbol>::resolvedDefinition;
  using MaybeRedecl<VariableSymbol>::setDefinition;
  using MaybeRedecl<VariableSymbol>::redeclarations;
  using MaybeRedecl<VariableSymbol>::addRedeclaration;

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

  [[nodiscard]] auto initializer() const -> ExpressionAST*;
  void setInitializer(ExpressionAST*);

  [[nodiscard]] auto constructor() const -> FunctionSymbol*;
  void setConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto constValue() const -> const std::optional<ConstValue>&;
  void setConstValue(std::optional<ConstValue> value);

 private:
  ExpressionAST* initializer_ = nullptr;
  FunctionSymbol* constructor_ = nullptr;
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

  [[nodiscard]] auto isNoUniqueAddress() const -> bool;
  void setNoUniqueAddress(bool isNoUniqueAddress);

  [[nodiscard]] auto offsetInClass() const -> std::optional<std::uint64_t>;

  [[nodiscard]] auto localOffset() const -> int;
  void setLocalOffset(int offset);

  [[nodiscard]] auto alignment() const -> int;
  void setAlignment(int alignment);

  [[nodiscard]] auto initializer() const -> ExpressionAST*;
  void setInitializer(ExpressionAST* initializer);

  [[nodiscard]] auto constructor() const -> FunctionSymbol*;
  void setConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto definition() const -> VariableSymbol* {
    return definition_;
  }
  void setDefinition(VariableSymbol* definition) { definition_ = definition; }

 private:
  VariableSymbol* definition_ = nullptr;
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
      std::uint32_t isNoUniqueAddress_ : 1;
    };
  };
  int localOffset_{};
  int alignment_{};
  int bitFieldOffset_{};
  std::optional<ConstValue> bitFieldWidth_;
  ExpressionAST* initializer_ = nullptr;
  FunctionSymbol* constructor_ = nullptr;
};

class ParameterSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kParameter;

  explicit ParameterSymbol(ScopeSymbol* enclosingScope);
  ~ParameterSymbol() override;

  [[nodiscard]] auto defaultArgument() const -> ExpressionAST*;
  void setDefaultArgument(ExpressionAST* expr);

 private:
  ExpressionAST* defaultArgument_ = nullptr;
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

  [[nodiscard]] auto typeConstraint() const -> TypeConstraintAST* {
    return typeConstraint_;
  }

  void setTypeConstraint(TypeConstraintAST* typeConstraint) {
    typeConstraint_ = typeConstraint;
  }

  [[nodiscard]] auto constraintExpression() const -> ExpressionAST* {
    return constraintExpression_;
  }

  void setConstraintExpression(ExpressionAST* constraintExpression) {
    constraintExpression_ = constraintExpression;
  }

 private:
  int index_ = 0;
  int depth_ = 0;
  bool isParameterPack_ = false;
  TypeConstraintAST* typeConstraint_ = nullptr;
  ExpressionAST* constraintExpression_ = nullptr;
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

  [[nodiscard]] auto introducedFunctions() const
      -> std::vector<FunctionSymbol*>;

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
  }

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

[[nodiscard]] inline auto is_global_namespace(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (!symbol->isNamespace()) return false;
  if (symbol->parent()) return false;
  return true;
}

[[nodiscard]] auto isEnclosedInTemplate(ScopeSymbol* scope) -> bool;
}  // namespace cxx
