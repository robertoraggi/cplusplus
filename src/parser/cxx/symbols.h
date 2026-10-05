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
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/diagnostic.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/time_trace.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cxx {
class SymbolChainView;
class TranslationUnit;

struct InstantiationError {
  SourceLocation location;
  std::string message;
  Severity severity = Severity::Error;
};

[[nodiscard]] auto instantiationErrorsOf(
    const std::vector<Diagnostic>& diagnostics)
    -> std::vector<InstantiationError>;

class TemplateSpecialization {
 public:
  std::vector<TemplateArgument> arguments;
  Symbol* symbol = nullptr;
  std::vector<InstantiationError> instantiationErrors;
  List<TemplateArgumentAST*>* pendingArgumentList = nullptr;
  SourceLocation pendingInstantiationLoc;
  bool isPendingInstantiation = false;
};

struct TemplateFriendship {
  std::vector<TemplateArgument> arguments;
  ClassSymbol* befriendingClass = nullptr;
};

enum class PendingInstantiationState {
  kUnresolved,
  kResolving,
  kRecursionDiagnosed,
  kResolved,
};

struct PendingInstantiation {
  AST* pattern = nullptr;
  AST* instance = nullptr;
  std::vector<TemplateArgument> templateArguments;
  ScopeSymbol* parentScope = nullptr;
  int depth = 0;
  PendingInstantiationState state = PendingInstantiationState::kUnresolved;
};

[[nodiscard]] auto compare_single_arg(TranslationUnit* unit,
                                      const TemplateArgument& lhs,
                                      const TemplateArgument& rhs) -> bool;

[[nodiscard]] auto compare_args(TranslationUnit* unit,
                                std::span<const TemplateArgument> args1,
                                std::span<const TemplateArgument> args2)
    -> bool;

[[nodiscard]] auto expand_template_arguments(
    std::span<const TemplateArgument> arguments)
    -> std::vector<TemplateArgument>;

[[nodiscard]] auto hash_template_arguments(
    std::span<const TemplateArgument> arguments) -> std::optional<std::size_t>;

struct ExpandedTemplateArgument {
  TemplateArgument value;
  std::size_t sourceIndex = 0;
};

[[nodiscard]] auto expand_template_arguments_with_sources(
    std::span<const TemplateArgument> arguments)
    -> std::vector<ExpandedTemplateArgument>;

[[nodiscard]] auto template_argument_type(const TemplateArgument& argument)
    -> const Type*;

[[nodiscard]] auto template_argument_as_type(const TemplateArgument& argument)
    -> const Type*;

[[nodiscard]] auto template_argument_value(const TemplateArgument& argument)
    -> std::optional<ConstValue>;

[[nodiscard]] auto template_argument_parameter_info(
    const TemplateArgument& argument) -> std::optional<TypeParamInfo>;

[[nodiscard]] auto class_template_of(ClassSymbol* classSymbol) -> ClassSymbol*;

[[nodiscard]] auto template_name_symbol(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto templated_symbol(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto resolve_using_declaration(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto names_functions(Symbol* symbol) -> bool;

[[nodiscard]] auto resolve_namespace_alias(Symbol* symbol) -> NamespaceSymbol*;

void add_extern_instantiation_declaration(
    Symbol* symbol, std::vector<TemplateArgument> arguments);

[[nodiscard]] auto template_declaration_of(Symbol* symbol)
    -> TemplateDeclarationAST*;

[[nodiscard]] auto find_specialization(
    TranslationUnit* unit, Symbol* templateSymbol,
    std::span<const TemplateArgument> arguments) -> Symbol*;

[[nodiscard]] auto specialization_entry_of(Symbol* templateSymbol,
                                           Symbol* specialization)
    -> TemplateSpecialization*;

void rebuild_specialization_index(Symbol* symbol);

[[nodiscard]] auto template_parameters_of(Symbol* symbol)
    -> TemplateParametersSymbol*;

[[nodiscard]] auto template_arguments_of(Symbol* symbol)
    -> std::span<const TemplateArgument>;

[[nodiscard]] auto primary_template_of(Symbol* symbol) -> Symbol*;

[[nodiscard]] auto pretty_function_name(FunctionSymbol* function)
    -> std::string;

[[nodiscard]] auto template_declaration_ast(Symbol* symbol) -> AST*;

[[nodiscard]] auto is_member_template(Symbol* symbol) -> bool;

[[nodiscard]] auto is_templated_class(ClassSymbol* classSymbol) -> bool;

[[nodiscard]] auto anonymous_member_class(FieldSymbol* field) -> ClassSymbol*;

[[nodiscard]] auto resolved_base_class(BaseClassSymbol* baseClass)
    -> ClassSymbol*;

[[nodiscard]] auto virtual_base_initialization_order(ClassSymbol* classSymbol)
    -> std::vector<ClassSymbol*>;

[[nodiscard]] auto is_anonymous_union_member(FieldSymbol* field) -> bool;

[[nodiscard]] auto has_variant_members(ClassSymbol* classSymbol) -> bool;

[[nodiscard]] auto is_inline_or_templated(FunctionSymbol* function) -> bool;

[[nodiscard]] auto has_static_storage_duration(Symbol* symbol) -> bool;

[[nodiscard]] auto closure_mangling_context(ClassSymbol* closure)
    -> FunctionSymbol*;

[[nodiscard]] auto type_has_internal_linkage(const Type* type) -> bool;

[[nodiscard]] auto is_specialized_on_internal_type(Symbol* symbol) -> bool;

[[nodiscard]] auto is_declared_with_internal_type(Symbol* symbol) -> bool;

[[nodiscard]] auto is_unnamed_namespace(Symbol* symbol) -> bool;

[[nodiscard]] auto is_in_unnamed_namespace(Symbol* symbol) -> bool;

[[nodiscard]] auto is_declared_extern(VariableSymbol* variable) -> bool;

[[nodiscard]] auto has_internal_linkage(Symbol* symbol) -> bool;

[[nodiscard]] auto is_non_static_member(Symbol* symbol) -> bool;

[[nodiscard]] auto introduces_variable(Symbol* symbol) -> bool;

[[nodiscard]] auto is_function_local_predefined_variable(Symbol* symbol)
    -> bool;

[[nodiscard]] auto required_parameter_count(FunctionSymbol* function,
                                            int parameterCount) -> int;

[[nodiscard]] auto is_callable_with_one_argument(FunctionSymbol* function)
    -> bool;

[[nodiscard]] auto default_template_argument(Symbol* symbol)
    -> TemplateParameterAST*;

void set_default_template_argument(Symbol* symbol,
                                   TemplateParameterAST* defaultArgument);

[[nodiscard]] auto names_current_instantiation(ClassSymbol* classSymbol,
                                               ScopeSymbol* scope) -> bool;

[[nodiscard]] auto names_current_instantiation(TranslationUnit* unit,
                                               SimpleTemplateIdAST* templateId,
                                               ClassSymbol* classSymbol)
    -> bool;

[[nodiscard]] auto template_parameter_info(Symbol* symbol)
    -> std::optional<TypeParamInfo>;

[[nodiscard]] auto is_template_parameter_pack(Symbol* symbol) -> bool;

class MaybeDefaultTemplateArgument {
 public:
  [[nodiscard]] auto defaultArgument() const -> TemplateParameterAST* {
    return defaultArgument_;
  }

  void setDefaultArgument(TemplateParameterAST* defaultArgument) {
    defaultArgument_ = defaultArgument;
  }

 private:
  TemplateParameterAST* defaultArgument_ = nullptr;
};

void indexTemplateArguments(
    std::span<const TemplateArgument> arguments, std::uint32_t position,
    std::unordered_map<std::size_t, std::vector<std::uint32_t>>& byArguments,
    std::vector<std::uint32_t>& unkeyed);

template <typename Match>
[[nodiscard]] auto findTemplateArguments(
    std::span<const TemplateArgument> arguments, std::size_t entryCount,
    const std::unordered_map<std::size_t, std::vector<std::uint32_t>>&
        byArguments,
    std::span<const std::uint32_t> unkeyed, Match isMatch)
    -> std::optional<std::size_t> {
  auto key = hash_template_arguments(arguments);

  if (!key.has_value()) {
    for (std::size_t index = 0; index < entryCount; ++index) {
      if (isMatch(index)) return index;
    }
    return std::nullopt;
  }

  if (auto bucket = byArguments.find(*key); bucket != byArguments.end()) {
    for (auto index : bucket->second) {
      if (isMatch(index)) return index;
    }
  }

  for (auto index : unkeyed) {
    if (isMatch(index)) return index;
  }
  return std::nullopt;
}

class SpecializationTable {
 public:
  [[nodiscard]] auto entries() const
      -> std::span<const TemplateSpecialization> {
    return entries_;
  }

  [[nodiscard]] auto find(TranslationUnit* unit,
                          std::span<const TemplateArgument> arguments) const
      -> Symbol*;

  [[nodiscard]] auto entryOf(const Symbol* specialization)
      -> TemplateSpecialization*;

  [[nodiscard]] auto add(TranslationUnit* unit,
                         std::vector<TemplateArgument> arguments,
                         Symbol* specialization) -> std::size_t;

  void restore(TemplateSpecialization specialization);

  void rebuildIndex();

 private:
  [[nodiscard]] auto findIndex(TranslationUnit* unit,
                               std::span<const TemplateArgument> arguments,
                               const Symbol* specialization) const
      -> std::optional<std::size_t>;

  [[nodiscard]] auto matches(TranslationUnit* unit, std::size_t index,
                             std::span<const TemplateArgument> arguments,
                             const Symbol* specialization) const -> bool;

  void index(std::size_t index);

  std::vector<TemplateSpecialization> entries_;
  std::unordered_map<std::size_t, std::vector<std::uint32_t>> byArguments_;
  std::unordered_map<const Symbol*, std::uint32_t> bySymbol_;
  std::vector<std::uint32_t> unkeyed_;
};

template <typename S>
class MaybeRedecl {
 public:
  [[nodiscard]] auto canonical() const -> S* {
    return canonical_ ? canonical_ : self();
  }

  [[nodiscard]] auto canonicalOrNull() const -> S* { return canonical_; }

  void setCanonical(S* canonical) { canonical_ = canonical; }

  [[nodiscard]] auto definition() const -> S* { return definition_; }

  [[nodiscard]] auto resolvedDefinition() const -> S* {
    return definition_ ? definition_ : self();
  }

  void setDefinition(S* definition) { definition_ = definition; }

  [[nodiscard]] auto redeclarations() const -> const std::vector<S*>& {
    return redeclarations_;
  }

  [[nodiscard]] auto declarations() const -> std::vector<S*> {
    auto first = canonical();
    std::vector<S*> result{first};
    std::ranges::copy(first->redeclarations_, std::back_inserter(result));
    return result;
  }

  void addRedeclaration(S* redecl) {
    if (!redecl || redecl == self()) cxx_runtime_error("invalid redeclaration");
    if (canonical_)
      cxx_runtime_error("addRedeclaration called on non-canonical symbol");
    if (std::ranges::contains(redeclarations_, redecl))
      cxx_runtime_error("duplicate redeclaration");
    redecl->setCanonical(self());
    redeclarations_.push_back(redecl);
  }

  void truncateRedeclarations(std::size_t count) {
    while (redeclarations_.size() > count) {
      redeclarations_.back()->setCanonical(nullptr);
      redeclarations_.pop_back();
    }
  }

 private:
  [[nodiscard]] auto self() const -> S* {
    return const_cast<S*>(static_cast<const S*>(this));
  }

  S* canonical_ = nullptr;
  S* definition_ = nullptr;
  std::vector<S*> redeclarations_;
};

template <typename S, typename D>
class MaybeTemplate {
  struct TemplateData {
    SpecializationTable specializations;
    mutable std::optional<std::vector<std::size_t>> declaredSpecializations;
    std::vector<std::vector<TemplateArgument>> externInstantiationDeclarations;
    TemplateDeclarationAST* templateDeclaration = nullptr;
    TemplateParametersSymbol* templateParameters = nullptr;
    S* primaryTemplateSymbol = nullptr;
    int specializationIndex = 0;
  };

 public:
  [[nodiscard]] auto declaration() const -> D* { return declaration_; }

  void setDeclaration(D* ast) { declaration_ = ast; }

  [[nodiscard]] auto templateDeclaration() const -> TemplateDeclarationAST* {
    return template_ ? template_->templateDeclaration : nullptr;
  }

  void setTemplateDeclaration(TemplateDeclarationAST* templateDeclaration) {
    if (this->templateDeclaration() == templateDeclaration) return;
    ensureTemplate().templateDeclaration = templateDeclaration;
    if (auto primary = primaryTemplateSymbol())
      primary->ensureTemplate().declaredSpecializations.reset();
  }

  [[nodiscard]] auto templateParameters() const -> TemplateParametersSymbol* {
    return template_ ? template_->templateParameters : nullptr;
  }

  void setTemplateParameters(TemplateParametersSymbol* templateParameters) {
    ensureTemplate().templateParameters = templateParameters;
  }

  [[nodiscard]] auto primaryTemplateSymbol() const -> S* {
    return template_ ? template_->primaryTemplateSymbol : nullptr;
  }

  [[nodiscard]] auto templateSpecializationIndex() const -> int {
    return template_ ? template_->specializationIndex : 0;
  }

  [[nodiscard]] auto isSpecialization() const -> bool {
    return primaryTemplateSymbol() != nullptr;
  }

  [[nodiscard]] auto isTemplatePattern() const -> bool {
    return templateParameters() != nullptr && !isSpecialization();
  }

  [[nodiscard]] auto templateArguments() const
      -> std::span<const TemplateArgument> {
    auto primary = primaryTemplateSymbol();
    if (!primary) return {};
    return primary->specializations()[templateSpecializationIndex()].arguments;
  }

  [[nodiscard]] auto specializations() const
      -> std::span<const TemplateSpecialization> {
    if (!template_) return {};
    return template_->specializations.entries();
  }

  [[nodiscard]] auto findSpecialization(
      TranslationUnit* unit, std::span<const TemplateArgument> arguments) const
      -> Symbol* {
    if (!template_) return nullptr;
    return template_->specializations.find(unit, arguments);
  }

  [[nodiscard]] auto declaredSpecializations(TimeTrace* trace = nullptr) const
      -> std::vector<TemplateSpecialization> {
    if (!template_) return {};
    auto& indices = template_->declaredSpecializations;
    auto entries = specializations();
    if (!indices) {
      if (trace)
        trace->count(TimeTrace::kSpecializationIndexVisits, entries.size());
      indices.emplace();
      for (std::size_t index = 0; index < entries.size(); ++index) {
        auto symbol = static_cast<S*>(entries[index].symbol);
        if constexpr (std::is_same_v<S, ClassSymbol>)
          symbol = symbol->resolvedDefinition();
        if (symbol->templateDeclaration()) indices->push_back(index);
      }
    }
    std::vector<TemplateSpecialization> result;
    result.reserve(indices->size());
    for (auto index : *indices) result.push_back(entries[index]);
    return result;
  }

  [[nodiscard]] auto specializationEntry(const Symbol* specialization)
      -> TemplateSpecialization* {
    if (!template_) return nullptr;
    return template_->specializations.entryOf(specialization);
  }

  void addSpecialization(TranslationUnit* unit,
                         std::vector<TemplateArgument> arguments,
                         S* specialization) {
    auto index = ensureTemplate().specializations.add(
        unit, std::move(arguments), specialization);
    specialization->restoreSpecializationInfo(static_cast<S*>(this),
                                              static_cast<int>(index));
  }

  void restoreSpecialization(TemplateSpecialization specialization) {
    ensureTemplate().specializations.restore(std::move(specialization));
    template_->declaredSpecializations.reset();
  }

  void rebuildSpecializationIndex() {
    if (!template_) return;
    template_->specializations.rebuildIndex();
  }

  void restoreSpecializationInfo(S* primaryTemplateSymbol, int index) {
    auto& data = ensureTemplate();
    data.primaryTemplateSymbol = primaryTemplateSymbol;
    data.specializationIndex = index;
    if (primaryTemplateSymbol && data.templateDeclaration)
      primaryTemplateSymbol->ensureTemplate().declaredSpecializations.reset();
  }

  void setPendingInstantiation(S* specialization,
                               List<TemplateArgumentAST*>* argumentList,
                               SourceLocation location, bool isPending) {
    auto entry = specializationEntry(specialization);
    if (!entry) return;
    entry->pendingArgumentList = argumentList;
    entry->pendingInstantiationLoc = location;
    entry->isPendingInstantiation = isPending;
  }

  void clearPendingInstantiation(S* specialization) {
    setPendingInstantiation(specialization, nullptr, {}, false);
  }

  void addExternInstantiationDeclaration(
      std::vector<TemplateArgument> arguments) {
    ensureTemplate().externInstantiationDeclarations.push_back(
        std::move(arguments));
  }

  [[nodiscard]] auto externInstantiationDeclarations() const
      -> std::span<const std::vector<TemplateArgument>> {
    if (!template_) return {};
    return template_->externInstantiationDeclarations;
  }

  [[nodiscard]] auto isExternInstantiationDeclared(
      TranslationUnit* unit, std::span<const TemplateArgument> arguments) const
      -> bool {
    return std::ranges::any_of(externInstantiationDeclarations(),
                               [&](const auto& declared) {
                                 return compare_args(unit, declared, arguments);
                               });
  }

  [[nodiscard]] auto isExplicitInstantiationDeclared(
      TranslationUnit* unit) const -> bool {
    auto primary = primaryTemplateSymbol();
    if (!primary) return false;
    return primary->isExternInstantiationDeclared(unit, templateArguments());
  }

 private:
  [[nodiscard]] auto ensureTemplate() -> TemplateData& {
    if (!template_) template_ = std::make_unique<TemplateData>();
    return *template_;
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
  [[nodiscard]] auto enclosingClass() const -> ClassSymbol*;

  [[nodiscard]] auto enclosingSymbols() const {
    return std::ranges::subrange(EnclosingSymbolIterator{parent()},
                                 EnclosingSymbolIterator{});
  }

  [[nodiscard]] auto enclosingFunction() const -> FunctionSymbol*;

  [[nodiscard]] auto enclosingFunctionOrSelf() const -> FunctionSymbol*;

  [[nodiscard]] auto hasEnclosingSymbol(Symbol* symbol) const -> bool;

  [[nodiscard]] auto next() const -> Symbol*;

  [[nodiscard]] auto isHidden() const -> bool { return isHidden_; }
  void setHidden(bool isHidden) { isHidden_ = isHidden; }

  [[nodiscard]] auto accessSpecifier() const -> AccessSpecifier {
    return accessSpecifier_;
  }
  void setAccessSpecifier(AccessSpecifier accessSpecifier) {
    accessSpecifier_ = accessSpecifier;
  }

  [[nodiscard]] auto abiTags() const -> std::span<const Identifier* const>;

  [[nodiscard]] auto abiTagList() const
      -> const std::vector<const Identifier*>* {
    return abiTags_;
  }

  void setAbiTags(const std::vector<const Identifier*>* abiTags);

  [[nodiscard]] auto isWeak() const -> bool;

  [[nodiscard]] auto isNoInline() const -> bool;

  [[nodiscard]] auto isAlwaysInline() const -> bool;

  [[nodiscard]] auto attributes() const -> const AttributeMap* {
    return attributes_;
  }
  void setAttributes(const AttributeMap* attributes) {
    attributes_ = attributes;
  }

  [[nodiscard]] auto isNodiscard() const -> bool { return isNodiscard_; }
  void setNodiscard(bool isNodiscard) { isNodiscard_ = isNodiscard; }

  [[nodiscard]] auto isUsed() const -> bool { return isUsed_; }
  void setUsed(bool isUsed) { isUsed_ = isUsed; }

  [[nodiscard]] auto isExcludedFromExplicitInstantiation() const -> bool {
    return isExcludedFromExplicitInstantiation_;
  }
  void setExcludedFromExplicitInstantiation(bool value) {
    isExcludedFromExplicitInstantiation_ = value;
  }

  [[nodiscard]] auto isTrivialAbi() const -> bool { return isTrivialAbi_; }
  void setTrivialAbi(bool isTrivialAbi) { isTrivialAbi_ = isTrivialAbi; }

  [[nodiscard]] auto hasDeducedReturnType() const -> bool {
    return hasDeducedReturnType_;
  }
  void setDeducedReturnType(bool hasDeducedReturnType) {
    hasDeducedReturnType_ = hasDeducedReturnType;
  }

  [[nodiscard]] auto canonical() const -> Symbol*;

  [[nodiscard]] auto definition() const -> Symbol*;

  [[nodiscard]] auto instantiationPattern() const -> Symbol* {
    return instantiationPattern_;
  }
  void setInstantiationPattern(Symbol* instantiationPattern) {
    instantiationPattern_ = instantiationPattern;
  }

#define PROCESS_SYMBOL(S) \
  [[nodiscard]] auto is##S() const -> bool { return kind_ == SymbolKind::k##S; }
  CXX_FOR_EACH_SYMBOL(PROCESS_SYMBOL)
#undef PROCESS_SYMBOL

  [[nodiscard]] auto isClassOrNamespace() const -> bool {
    return isClass() || isNamespace();
  }

  [[nodiscard]] auto isNamespaceName() const -> bool {
    return isNamespace() || isNamespaceAlias();
  }

  [[nodiscard]] auto isEnumOrScopedEnum() const -> bool {
    return isEnum() || isScopedEnum();
  }

  /**
   * Scratch space for a walk that has to number the symbols it reaches, such
   * as the archive encoder assigning references. Zero means unnumbered; a walk
   * clears what it stamped when it finishes, so the field is always zero
   * outside one.
   */
  [[nodiscard]] auto internalId() const -> std::uint32_t { return internalId_; }
  void setInternalId(std::uint32_t internalId) { internalId_ = internalId; }

 private:
  friend class ScopeSymbol;

  SymbolKind kind_;
  const Name* name_ = nullptr;
  const Type* type_ = nullptr;
  ScopeSymbol* parent_ = nullptr;
  Symbol* link_ = nullptr;
  Symbol* instantiationPattern_ = nullptr;
  const std::vector<const Identifier*>* abiTags_ = nullptr;
  const AttributeMap* attributes_ = nullptr;
  SourceLocation location_;
  std::uint32_t internalId_ = 0;
  bool isHidden_ = false;
  bool isNodiscard_ = false;
  bool isUsed_ = false;
  bool isExcludedFromExplicitInstantiation_ = false;
  bool isTrivialAbi_ = false;
  bool hasDeducedReturnType_ = false;
  AccessSpecifier accessSpecifier_ = AccessSpecifier::kPublic;
};

class ScopeSymbol : public Symbol {
 public:
  ScopeSymbol(SymbolKind kind, ScopeSymbol* enclosingScope);
  ~ScopeSymbol() override;

  [[nodiscard]] auto asScopeSymbol() -> ScopeSymbol* override { return this; }

  [[nodiscard]] auto empty() const -> bool { return members_.empty(); }

  [[nodiscard]] auto members() const -> const std::vector<Symbol*>&;

  [[nodiscard]] auto usingDirectives() const {
    return std::views::all(usingDirectives_);
  }

  [[nodiscard]] auto find(const Name* name) const -> SymbolChainView;

  [[nodiscard]] auto find(const std::string_view& name) const
      -> SymbolChainView;

  [[nodiscard]] auto find(TokenKind op) const -> SymbolChainView;

  void addSymbol(Symbol* symbol);

  /**
   * Appends a member without indexing it. A decoded scope is filled before the
   * names of its members are interned, so the caller answers for calling
   * `rebuildLookupTable` once the graph is fixed up (7.6, phase 7).
   */
  void addMember(Symbol* symbol);

  void addUsingDirective(ScopeSymbol* scope);

  [[nodiscard]] auto isTransparent() const -> bool;

  void replaceSymbol(Symbol* symbol, Symbol* newSymbol);
  void truncate(std::size_t count);
  void reset();

  /**
   * Rebuilds the name-lookup index from `members_`. A decoded scope is filled
   * before the names of its members are interned, so the index is rebuilt once
   * the graph is fixed up (7.6, phase 7).
   */
  void rebuildLookupTable();

 private:
  void rehash();

  [[nodiscard]] auto bucketOfHash(std::size_t hash) const -> std::size_t {
    return hash & (buckets_.size() - 1);
  }

  [[nodiscard]] auto bucketOf(const Symbol* symbol) const -> std::size_t;

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

  [[nodiscard]] auto getVirtualBaseInfo(ClassSymbol* base) const
      -> std::optional<MemberInfo>;

  [[nodiscard]] auto getBaseInfo(ClassSymbol* base, bool isVirtual) const
      -> std::optional<MemberInfo> {
    return isVirtual ? getVirtualBaseInfo(base) : getBaseInfo(base);
  }

  void setFieldInfo(FieldSymbol* field, const MemberInfo& info);
  void setBaseInfo(ClassSymbol* base, const MemberInfo& info);
  void setVirtualBaseInfo(ClassSymbol* base, const MemberInfo& info);

  /**
   * The layout tables in member order. The maps themselves are unordered, and
   * anything that has to produce the same bytes twice — an archive (7.5), a
   * layout dump — needs a total order that does not depend on hashing.
   */
  [[nodiscard]] auto sortedFieldInfos() const
      -> std::vector<std::pair<FieldSymbol*, MemberInfo>> {
    return sortedByIndex(fields_);
  }

  [[nodiscard]] auto sortedBaseInfos() const
      -> std::vector<std::pair<ClassSymbol*, MemberInfo>> {
    return sortedByIndex(bases_);
  }

  [[nodiscard]] auto sortedVirtualBaseInfos() const
      -> std::vector<std::pair<ClassSymbol*, MemberInfo>> {
    return sortedByIndex(virtualBaseInfos_);
  }

  void addVirtualBase(ClassSymbol* base) { virtualBases_.push_back(base); }
  [[nodiscard]] auto virtualBases() const -> const std::vector<ClassSymbol*>& {
    return virtualBases_;
  }

  void setSize(std::uint64_t size) { size_ = size; }
  void setAlignment(std::uint64_t alignment) { alignment_ = alignment; }

  void setDataSize(std::uint64_t dataSize) { dataSize_ = dataSize; }
  [[nodiscard]] auto dataSize() const -> std::uint64_t { return dataSize_; }

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
  void setAbiEmpty(bool abiEmpty) { abiEmpty_ = abiEmpty; }
  [[nodiscard]] auto isAbiEmpty() const -> bool { return abiEmpty_; }

  struct PaddingInfo {
    std::uint32_t index = 0;
    std::uint64_t offset = 0;
    std::uint64_t sizeInBytes = 0;
  };

  void addPadding(std::uint32_t index, std::uint64_t offset,
                  std::uint64_t sizeInBytes) {
    padding_.push_back({index, offset, sizeInBytes});
  }

  [[nodiscard]] auto padding() const -> const std::vector<PaddingInfo>& {
    return padding_;
  }

  void setPrimaryBase(ClassSymbol* primaryBase, bool isVirtual) {
    primaryBase_ = primaryBase;
    primaryBaseIsVirtual_ = isVirtual;
  }

  [[nodiscard]] auto primaryBase() const -> ClassSymbol* {
    return primaryBase_;
  }

  [[nodiscard]] auto primaryBaseIsVirtual() const -> bool {
    return primaryBaseIsVirtual_;
  }

 private:
  template <typename K>
  [[nodiscard]] static auto sortedByIndex(
      const std::unordered_map<K, MemberInfo>& entries)
      -> std::vector<std::pair<K, MemberInfo>> {
    std::vector<std::pair<K, MemberInfo>> result(entries.begin(),
                                                 entries.end());
    std::ranges::sort(result, {},
                      [](const auto& entry) { return entry.second.index; });
    return result;
  }

  std::unordered_map<FieldSymbol*, MemberInfo> fields_;
  std::unordered_map<ClassSymbol*, MemberInfo> bases_;
  std::unordered_map<ClassSymbol*, MemberInfo> virtualBaseInfos_;
  std::vector<ClassSymbol*> virtualBases_;
  std::vector<PaddingInfo> padding_;
  std::uint64_t size_ = 0;
  std::uint64_t dataSize_ = 0;
  std::uint64_t alignment_ = 1;
  std::uint64_t nonVirtualSize_ = 0;
  std::uint64_t nonVirtualAlignment_ = 1;
  std::uint32_t vtableIndex_ = 0;
  ClassSymbol* primaryBase_ = nullptr;
  bool hasVtable_ = false;
  bool hasDirectVtable_ = false;
  bool primaryBaseIsVirtual_ = false;
  bool abiEmpty_ = false;
};

class VTableLayout {
 public:
  enum class SlotKind : std::uint8_t {
    kFunction,
    kCompleteDtor,
    kDeletingDtor,
  };

  struct CallOffset {
    std::int64_t nonVirtual = 0;
    std::int64_t virtualOffset = 0;

    [[nodiscard]] auto isEmpty() const -> bool {
      return nonVirtual == 0 && virtualOffset == 0;
    }
  };

  struct Slot {
    FunctionSymbol* function = nullptr;
    SlotKind kind = SlotKind::kFunction;
    FunctionSymbol* introducingFunction = nullptr;
    CallOffset thisAdjustment;
    CallOffset returnAdjustment;
  };

  struct Offset {
    Symbol* subject = nullptr;
    std::int64_t value = 0;
  };

  struct Table {
    ClassSymbol* base = nullptr;
    std::uint64_t offset = 0;
    ClassSymbol* enclosingVirtualBase = nullptr;
    std::int64_t offsetToTop = 0;
    std::vector<Offset> offsets;
    std::vector<Slot> slots;

    [[nodiscard]] auto headerWordCount() const -> std::size_t {
      return offsets.size() + 2;
    }

    [[nodiscard]] auto wordCount() const -> std::size_t {
      return headerWordCount() + slots.size();
    }

    [[nodiscard]] auto offsetWordsBeforeAddressPoint(Symbol* subject) const
        -> std::int64_t;
  };

  struct Group {
    ClassSymbol* base = nullptr;
    std::uint64_t offset = 0;
    std::vector<Table> tables;

    [[nodiscard]] auto wordCount() const -> std::size_t;
    [[nodiscard]] auto addressPointIndex(std::size_t table) const
        -> std::size_t;
    [[nodiscard]] auto tableAt(std::uint64_t offset) const -> int;
  };

  struct EntryPoint {
    FunctionSymbol* function = nullptr;
    SlotKind kind = SlotKind::kFunction;
    CallOffset thisAdjustment;
    CallOffset returnAdjustment;
  };

  struct VTTEntry {
    std::int32_t group = -1;
    std::uint32_t table = 0;
  };

  struct SubVTT {
    ClassSymbol* base = nullptr;
    std::uint32_t index = 0;
  };

  Group main;
  std::vector<Group> constructionGroups;
  std::vector<VTTEntry> vtt;
  std::vector<std::int32_t> tableVTTIndices;
  std::vector<SubVTT> baseSubVTTs;
  std::vector<SubVTT> virtualBaseSubVTTs;
  std::vector<EntryPoint> adjustingEntryPoints;
  FunctionSymbol* keyFunction = nullptr;

  [[nodiscard]] auto primary() const -> const Table& {
    return main.tables.front();
  }

  [[nodiscard]] auto finalOverrider(FunctionSymbol* function) const
      -> FunctionSymbol*;
};

class BaseClassSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kBaseClass;

  explicit BaseClassSymbol(ScopeSymbol* enclosingScope);
  ~BaseClassSymbol() override;

  [[nodiscard]] auto isVirtual() const -> bool;
  void setVirtual(bool isVirtual);

  [[nodiscard]] auto symbol() const -> Symbol*;
  void setSymbol(Symbol* symbol);

 private:
  Symbol* symbol_ = nullptr;
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
  using MaybeRedecl<ClassSymbol>::definition;

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

  [[nodiscard]] auto visibleConversionFunctions() const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto destructor() const -> FunctionSymbol*;
  [[nodiscard]] auto defaultConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto copyConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto moveConstructor() const -> FunctionSymbol*;
  [[nodiscard]] auto copyAssignmentOperator() const -> FunctionSymbol*;
  [[nodiscard]] auto moveAssignmentOperator() const -> FunctionSymbol*;
  [[nodiscard]] auto isDefaultConstructor(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto isCopyConstructor(FunctionSymbol* function) const -> bool;
  [[nodiscard]] auto isMoveConstructor(FunctionSymbol* function) const -> bool;
  [[nodiscard]] auto isCopyAssignmentOperator(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto isMoveAssignmentOperator(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto hasUserDeclaredConstructors() const -> bool;
  void setHasUserDeclaredConstructors(bool value);

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

  [[nodiscard]] auto isAccessControlDisabled() const -> bool {
    return isAccessControlDisabled_;
  }
  void setAccessControlDisabled(bool value) {
    isAccessControlDisabled_ = value;
  }

  [[nodiscard]] auto sizeInBytes() const -> int;
  void setSizeInBytes(int sizeInBytes);

  [[nodiscard]] auto alignment() const -> int;
  void setAlignment(int alignment);

  [[nodiscard]] auto explicitAlignment() const -> int;
  void setExplicitAlignment(int alignment);

  [[nodiscard]] auto packAlignment() const -> int;
  void setPackAlignment(int alignment);

  [[nodiscard]] auto hasBaseClass(const Symbol* symbol) const -> bool;

  void addBefriendingClass(ClassSymbol* classSymbol);
  void addBefriendingClass(ClassSymbol* classSymbol,
                           std::vector<TemplateArgument> arguments);
  [[nodiscard]] auto befriendingClasses() const
      -> const std::vector<ClassSymbol*>& {
    return befriendingClasses_;
  }
  [[nodiscard]] auto templateFriendships() const
      -> const std::vector<TemplateFriendship>& {
    return templateFriendships_;
  }

  struct BaseSubobjectInfo {
    std::uint32_t subobjectCount = 0;
    std::uint32_t nonVirtualSubobjectCount = 0;
    std::uint64_t nonVirtualOffset = 0;
    std::uint32_t publicSubobjectCount = 0;
    std::uint32_t publicNonVirtualSubobjectCount = 0;
    std::uint64_t publicNonVirtualOffset = 0;

    [[nodiscard]] auto isUniqueSubobject() const -> bool {
      return subobjectCount == 1;
    }

    [[nodiscard]] auto hasPublicSubobjectInVirtualBase() const -> bool {
      return publicSubobjectCount > publicNonVirtualSubobjectCount;
    }
  };

  [[nodiscard]] auto baseSubobjectInfo(ClassSymbol* base) const
      -> BaseSubobjectInfo;

  [[nodiscard]] auto baseClassOffset(ClassSymbol* base) const
      -> std::optional<std::uint64_t>;

  struct BaseClassRepetition {
    bool nonDiamondRepeat = false;
    bool diamondShaped = false;
  };

  [[nodiscard]] auto baseClassRepetition() const -> BaseClassRepetition;

  [[nodiscard]] auto hasVirtualBasePath(Symbol* symbol) const -> bool;

  [[nodiscard]] auto flags() const -> std::uint32_t;
  void setFlags(std::uint32_t flags);

  void setLayout(std::unique_ptr<ClassLayout> layout);

  [[nodiscard]] auto layout() const -> const ClassLayout*;

  [[nodiscard]] auto hasVirtualBaseSubobjects() const -> bool;

  void setVTableLayout(std::unique_ptr<VTableLayout> vtableLayout);

  [[nodiscard]] auto vtableLayout() const -> const VTableLayout*;

  [[nodiscard]] auto isClosureType() const -> bool;
  void setIsClosureType(bool isClosureType);

  [[nodiscard]] auto hasLambdaCapture() const -> bool;
  void setHasLambdaCapture(bool hasLambdaCapture);

  [[nodiscard]] auto capturedThisField() const -> FieldSymbol*;
  void setCapturedThisField(FieldSymbol* capturedThisField);

  [[nodiscard]] auto functionCallOperator() const -> FunctionSymbol*;

  [[nodiscard]] auto closureDiscriminator() const -> int;
  void setClosureDiscriminator(int closureDiscriminator);

  [[nodiscard]] auto instantiationTemplate() const -> ClassSymbol*;

 private:
  std::vector<BaseClassSymbol*> baseClasses_;
  std::vector<ClassSymbol*> befriendingClasses_;
  std::vector<TemplateFriendship> templateFriendships_;
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
  int explicitAlignment_ = 0;
  int packAlignment_ = 0;
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isUnion_ : 1;
      std::uint32_t isFinal_ : 1;
      std::uint32_t isComplete_ : 1;
      std::uint32_t isFriend_ : 1;
      std::uint32_t isAccessControlDisabled_ : 1;
      std::uint32_t isPolymorphic_ : 1;
      std::uint32_t isAbstract_ : 1;
      std::uint32_t hasVirtualDestructor_ : 1;
      std::uint32_t isClosureType_ : 1;
      std::uint32_t hasLambdaCapture_ : 1;
      std::uint32_t hasUserDeclaredConstructors_ : 1;
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
  using MaybeRedecl<FunctionSymbol>::definition;

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

  [[nodiscard]] auto isNonStaticMemberFunction() const -> bool;

  [[nodiscard]] auto hasImplicitObjectParameter() const -> bool;

  [[nodiscard]] auto hasExplicitObjectParameter() const -> bool;
  void setExplicitObjectParameter(bool hasExplicitObjectParameter);

  [[nodiscard]] auto explicitObjectParameter() const -> ParameterSymbol*;

  [[nodiscard]] auto parameters() const -> std::vector<ParameterSymbol*>;

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

  [[nodiscard]] auto isNoReturn() const -> bool;
  void setNoReturn(bool isNoReturn);

  [[nodiscard]] auto builtinKind() const -> BuiltinFunctionKind;
  void setBuiltinKind(BuiltinFunctionKind builtinKind);

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

  [[nodiscard]] auto importModule() const -> const Identifier*;
  void setImportModule(const Identifier* importModule);

  [[nodiscard]] auto importName() const -> const Identifier*;
  void setImportName(const Identifier* importName);

  [[nodiscard]] auto exportName() const -> const Identifier*;
  void setExportName(const Identifier* exportName);

  [[nodiscard]] auto hasPendingBody() const -> bool;
  [[nodiscard]] auto hasUninstantiatedBody() const -> bool;
  [[nodiscard]] auto pendingBody() const -> PendingInstantiation*;
  void setPendingBody(std::unique_ptr<PendingInstantiation> pending);
  void clearPendingBody();

  [[nodiscard]] auto hasDeferredImplicitExceptionSpecification() const -> bool {
    return hasDeferredImplicitExceptionSpecification_;
  }
  void setDeferredImplicitExceptionSpecification(bool deferred) {
    hasDeferredImplicitExceptionSpecification_ = deferred;
  }

  [[nodiscard]] auto pendingExceptionSpecification() const
      -> PendingInstantiation*;
  void setPendingExceptionSpecification(
      std::unique_ptr<PendingInstantiation> pending);

  [[nodiscard]] auto vtableSlotIndex() const -> int { return vtableSlotIndex_; }
  void setVtableSlotIndex(int index) { vtableSlotIndex_ = index; }

  void addOverriddenFunction(FunctionSymbol* function);
  [[nodiscard]] auto overriddenFunctions() const
      -> const std::vector<FunctionSymbol*>& {
    return overriddenFunctions_;
  }
  [[nodiscard]] auto overrides(FunctionSymbol* function) const -> bool;

  void addBefriendingClass(ClassSymbol* classSymbol);
  void addBefriendingClass(ClassSymbol* classSymbol,
                           std::vector<TemplateArgument> arguments);
  [[nodiscard]] auto befriendingClasses() const
      -> const std::vector<ClassSymbol*>& {
    return befriendingClasses_;
  }
  [[nodiscard]] auto templateFriendships() const
      -> const std::vector<TemplateFriendship>& {
    return templateFriendships_;
  }

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

  [[nodiscard]] auto isStructor() const -> bool;

  [[nodiscard]] auto hasBaseObjectVariant() const -> bool;

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

  [[nodiscard]] auto hostScope() const -> ScopeSymbol* {
    return hostScope_ ? hostScope_ : parent();
  }

  void setHostScope(ScopeSymbol* hostScope) { hostScope_ = hostScope; }

  [[nodiscard]] auto hasFriendDefaultArgument() const -> bool {
    return hasFriendDefaultArgument_;
  }
  void setFriendDefaultArgument(bool value) {
    hasFriendDefaultArgument_ = value;
  }

  [[nodiscard]] auto hasFriendDefaultTemplateArgument() const -> bool {
    return hasFriendDefaultTemplateArgument_;
  }
  void setFriendDefaultTemplateArgument(bool value) {
    hasFriendDefaultTemplateArgument_ = value;
  }

 private:
  std::unique_ptr<PendingInstantiation> pendingBody_;
  std::unique_ptr<PendingInstantiation> pendingExceptionSpecification_;
  ScopeSymbol* hostScope_ = nullptr;
  FunctionSymbol* completeObjectVariant_ = nullptr;
  FunctionSymbol* delegatingConstructor_ = nullptr;
  FunctionSymbol* deletingDtorVariant_ = nullptr;
  FunctionSymbol* structorPrincipal_ = nullptr;
  FunctionSymbol* inheritedConstructor_ = nullptr;
  std::vector<FunctionSymbol*> overriddenFunctions_;
  std::vector<ClassSymbol*> befriendingClasses_;
  std::vector<TemplateFriendship> templateFriendships_;
  int vtableSlotIndex_ = -1;
  const Identifier* externalName_ = nullptr;
  const Identifier* aliasName_ = nullptr;
  const Identifier* importModule_ = nullptr;
  const Identifier* importName_ = nullptr;
  const Identifier* exportName_ = nullptr;
  RequiresClauseAST* trailingRequiresClause_ = nullptr;
  BuiltinFunctionKind builtinKind_ = BuiltinFunctionKind::T_NONE;
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
      std::uint32_t hasExplicitObjectParameter_ : 1;
      std::uint32_t isNoReturn_ : 1;
      std::uint32_t hasFriendDefaultArgument_ : 1;
      std::uint32_t hasFriendDefaultTemplateArgument_ : 1;
      std::uint32_t hasDeferredImplicitExceptionSpecification_ : 1;
    };
  };
};

class OverloadSetSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kOverloadSet;

  explicit OverloadSetSymbol(ScopeSymbol* enclosingScope);
  ~OverloadSetSymbol() override;

  [[nodiscard]] auto functions() const -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto functionsWithin(std::size_t functionCount,
                                     std::size_t usingDeclarationCount) const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto declaredFunctions() const
      -> const std::vector<FunctionSymbol*>&;

  void setFunctions(std::vector<FunctionSymbol*> functions);
  void addFunction(FunctionSymbol* function);
  void truncateFunctions(std::size_t count);

  [[nodiscard]] auto usingDeclarations() const
      -> const std::vector<UsingDeclarationSymbol*>&;

  void addUsingDeclaration(UsingDeclarationSymbol* usingDeclaration);

  [[nodiscard]] auto hasUnresolvedUsingDeclaration() const -> bool;

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

  [[nodiscard]] auto closureType() const -> ClassSymbol* {
    return closureType_;
  }
  void setClosureType(ClassSymbol* closureType) { closureType_ = closureType; }

 private:
  ClassSymbol* closureType_ = nullptr;

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

  [[nodiscard]] auto cvQualifiers() const -> CvQualifiers {
    return cvQualifiers_;
  }
  void setCvQualifiers(CvQualifiers value) { cvQualifiers_ = value; }

 private:
  CvQualifiers cvQualifiers_ = CvQualifiers::kNone;
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

  [[nodiscard]] auto isOutermostBlockScope() const -> bool {
    return isOutermostBlockScope_;
  }

  void setOutermostBlockScope(bool value) { isOutermostBlockScope_ = value; }

  [[nodiscard]] auto isFunctionTryHandler() const -> bool {
    return isFunctionTryHandler_;
  }

  void setFunctionTryHandler(bool value) { isFunctionTryHandler_ = value; }

 private:
  bool isOutermostBlockScope_ = false;
  bool isFunctionTryHandler_ = false;
};

class TypeAliasSymbol final
    : public Symbol,
      public MaybeTemplate<TypeAliasSymbol, AliasDeclarationAST>,
      public MaybeRedecl<TypeAliasSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kTypeAlias;

  using MaybeRedecl<TypeAliasSymbol>::canonical;
  using MaybeRedecl<TypeAliasSymbol>::definition;

  explicit TypeAliasSymbol(ScopeSymbol* enclosingScope);
  ~TypeAliasSymbol() override;

  [[nodiscard]] auto expansionTypeId() const -> TypeIdAST* {
    return expansionTypeId_;
  }

  void setExpansionTypeId(TypeIdAST* typeId) { expansionTypeId_ = typeId; }

 private:
  TypeIdAST* expansionTypeId_ = nullptr;
};

class VariableSymbol final
    : public Symbol,
      public MaybeTemplate<VariableSymbol, SimpleDeclarationAST>,
      public MaybeRedecl<VariableSymbol> {
 public:
  constexpr static auto Kind = SymbolKind::kVariable;

  using MaybeRedecl<VariableSymbol>::canonical;
  using MaybeRedecl<VariableSymbol>::definition;

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

  [[nodiscard]] auto isFunctionLocalPredefined() const -> bool;
  void setFunctionLocalPredefined(bool isFunctionLocalPredefined);

  [[nodiscard]] auto initializer() const -> ExpressionAST*;
  void setInitializer(ExpressionAST*);

  [[nodiscard]] auto constructor() const -> FunctionSymbol*;
  void setConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto constValue() const -> const std::optional<ConstValue>&;
  void setConstValue(std::optional<ConstValue> value);

  [[nodiscard]] auto explicitAlignment() const -> int;
  void setExplicitAlignment(int alignment);

 private:
  ExpressionAST* initializer_ = nullptr;
  FunctionSymbol* constructor_ = nullptr;
  std::optional<ConstValue> constValue_;

  int explicitAlignment_ = 0;
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isStatic_ : 1;
      std::uint32_t isThreadLocal_ : 1;
      std::uint32_t isExtern_ : 1;
      std::uint32_t isConstexpr_ : 1;
      std::uint32_t isConstinit_ : 1;
      std::uint32_t isInline_ : 1;
      std::uint32_t isFunctionLocalPredefined_ : 1;
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

  [[nodiscard]] auto explicitAlignment() const -> int;
  void setExplicitAlignment(int alignment);

  [[nodiscard]] auto isPacked() const -> bool;
  [[nodiscard]] auto effectiveAlignment() const -> int;

  [[nodiscard]] auto initializer() const -> ExpressionAST*;
  void setInitializer(ExpressionAST* initializer);

  [[nodiscard]] auto constructor() const -> FunctionSymbol*;
  void setConstructor(FunctionSymbol* constructor);

  [[nodiscard]] auto constValue() const -> const std::optional<ConstValue>&;
  void setConstValue(std::optional<ConstValue> value);

  [[nodiscard]] auto definition() const -> VariableSymbol* {
    return definition_;
  }
  void setDefinition(VariableSymbol* definition) { definition_ = definition; }

  [[nodiscard]] auto isDefinitionRequired() const -> bool {
    return isDefinitionRequired_;
  }
  void setDefinitionRequired(bool isDefinitionRequired) {
    isDefinitionRequired_ = isDefinitionRequired;
  }

  [[nodiscard]] auto hasPendingInitializer() const -> bool {
    return pendingInitializer_ != nullptr;
  }

  [[nodiscard]] auto hasInitializer() const -> bool {
    return initializer_ != nullptr || hasPendingInitializer();
  }
  [[nodiscard]] auto pendingInitializer() const -> PendingInstantiation*;
  void setPendingInitializer(std::unique_ptr<PendingInstantiation> pending);
  void clearPendingInitializer();

 private:
  VariableSymbol* definition_ = nullptr;
  std::unique_ptr<PendingInstantiation> pendingInitializer_;
  std::optional<ConstValue> constValue_;
  union {
    std::uint32_t flags_{};
    struct {
      std::uint32_t isDefinitionRequired_ : 1;
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
  int explicitAlignment_{};
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

  [[nodiscard]] auto hasDefaultArgument() const -> bool;

  [[nodiscard]] auto pendingDefaultArgument() const -> PendingInstantiation*;
  void setPendingDefaultArgument(std::unique_ptr<PendingInstantiation> pending);
  void clearPendingDefaultArgument();

  [[nodiscard]] auto defaultArgumentSource() const -> ParameterSymbol*;
  void setDefaultArgumentSource(ParameterSymbol* source);

  [[nodiscard]] auto isExplicitObject() const -> bool;
  void setExplicitObject(bool isExplicitObject);

  [[nodiscard]] auto isParameterPack() const -> bool;
  void setParameterPack(bool isParameterPack);

 private:
  ExpressionAST* defaultArgument_ = nullptr;
  std::unique_ptr<PendingInstantiation> pendingDefaultArgument_;
  ParameterSymbol* defaultArgumentSource_ = nullptr;
  bool isExplicitObject_ = false;
  bool isParameterPack_ = false;
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

class TypeParameterSymbol final : public Symbol,
                                  public MaybeDefaultTemplateArgument {
 public:
  constexpr static auto Kind = SymbolKind::kTypeParameter;

  explicit TypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~TypeParameterSymbol() override;
};

class NonTypeParameterSymbol final : public Symbol,
                                     public MaybeDefaultTemplateArgument {
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

class TemplateTypeParameterSymbol final : public Symbol,
                                          public MaybeDefaultTemplateArgument {
 public:
  constexpr static auto Kind = SymbolKind::kTemplateTypeParameter;

  explicit TemplateTypeParameterSymbol(ScopeSymbol* enclosingScope);
  ~TemplateTypeParameterSymbol() override;
};

class ConstraintTypeParameterSymbol final
    : public Symbol,
      public MaybeDefaultTemplateArgument {
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

class NamespaceAliasSymbol final : public Symbol {
 public:
  constexpr static auto Kind = SymbolKind::kNamespaceAlias;

  explicit NamespaceAliasSymbol(ScopeSymbol* enclosingScope);
  ~NamespaceAliasSymbol() override;

  [[nodiscard]] auto namespaceSymbol() const -> NamespaceSymbol*;
  void setNamespaceSymbol(NamespaceSymbol* namespaceSymbol);

 private:
  NamespaceSymbol* namespaceSymbol_ = nullptr;
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

  [[nodiscard]] auto targetFunctionCount() const -> int;
  void setTargetFunctionCount(int count);

  [[nodiscard]] auto targetUsingDeclarationCount() const -> int;
  void setTargetUsingDeclarationCount(int count);

  [[nodiscard]] auto introducedFunctions() const
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto isUnresolved() const -> bool;

 private:
  Symbol* target_ = nullptr;
  int targetFunctionCount_ = 0;
  int targetUsingDeclarationCount_ = 0;
  UsingDeclaratorAST* declarator_ = nullptr;
};

bool is_type(Symbol* symbol);

[[nodiscard]] auto is_class_or_enum_declaration(Symbol* symbol) -> bool;

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

[[nodiscard]] auto isDeclaredConstant(Symbol* symbol) -> bool;

[[nodiscard]] auto isUsableInConstantExpressions(Symbol* symbol) -> bool;
}  // namespace cxx
