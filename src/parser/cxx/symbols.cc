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

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <bit>
#include <format>
#include <functional>
#include <limits>
#include <ranges>
#include <unordered_set>

namespace cxx {

auto instantiationErrorsOf(const std::vector<Diagnostic>& diagnostics)
    -> std::vector<InstantiationError> {
  std::vector<InstantiationError> errors;
  errors.reserve(diagnostics.size());
  for (const auto& diagnostic : diagnostics) {
    errors.push_back(
        {diagnostic.location(), diagnostic.message(), diagnostic.severity()});
  }
  return errors;
}

auto toAccessSpecifier(TokenKind accessSpecifierToken,
                       AccessSpecifier defaultAccessSpecifier)
    -> AccessSpecifier {
  switch (accessSpecifierToken) {
    case TokenKind::T_PRIVATE:
      return AccessSpecifier::kPrivate;
    case TokenKind::T_PROTECTED:
      return AccessSpecifier::kProtected;
    case TokenKind::T_PUBLIC:
      return AccessSpecifier::kPublic;
    default:
      return defaultAccessSpecifier;
  }
}

auto defaultAccessSpecifierOfClassKey(TokenKind classKey) -> AccessSpecifier {
  if (classKey == TokenKind::T_CLASS) return AccessSpecifier::kPrivate;
  return AccessSpecifier::kPublic;
}

namespace {
[[nodiscard]] auto hasEquivalentParameterTypeList(FunctionSymbol* lhs,
                                                  FunctionSymbol* rhs) -> bool {
  auto lhsType = type_cast<FunctionType>(lhs->type());
  auto rhsType = type_cast<FunctionType>(rhs->type());
  if (!lhsType || !rhsType) return false;

  return lhsType->parameterTypes() == rhsType->parameterTypes() &&
         lhsType->isVariadic() == rhsType->isVariadic() &&
         lhsType->cvQualifiers() == rhsType->cvQualifiers() &&
         lhsType->refQualifier() == rhsType->refQualifier();
}

struct NonTypeParameterIdentity {
  int depth;
  int index;
  bool isPack;

  auto operator==(const NonTypeParameterIdentity&) const -> bool = default;
};

auto nonTypeParameterIdentity(Symbol* symbol)
    -> std::optional<NonTypeParameterIdentity> {
  auto parameter = symbol_cast<NonTypeParameterSymbol>(symbol);
  if (!parameter) return std::nullopt;
  return NonTypeParameterIdentity{parameter->depth(), parameter->index(),
                                  parameter->isParameterPack()};
}

struct HoldsAST {
  mutable const Type* firstExaminedType = nullptr;
  mutable std::vector<const Type*> examinedTypes;

  [[nodiscard]] auto alreadyExamined(const Type* type) const -> bool {
    if (type == firstExaminedType) return true;
    return std::ranges::contains(examinedTypes, type);
  }

  void recordExamined(const Type* type) const {
    if (!firstExaminedType) {
      firstExaminedType = type;
      return;
    }
    examinedTypes.push_back(type);
  }

  [[nodiscard]] auto operator()(const DecltypeType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedNameType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedBoundedArrayType*) const
      -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedUnderlyingType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedBuiltinType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedBitIntType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const UnresolvedVectorType*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const PointerType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const LvalueReferenceType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const RvalueReferenceType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const BoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const UnboundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const MemberObjectPointerType* type) const
      -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const MemberFunctionPointerType* type) const
      -> bool {
    return visit(*this, type->functionType());
  }

  [[nodiscard]] auto operator()(const PackExpansionType* type) const -> bool {
    return visit(*this, type->pattern());
  }

  [[nodiscard]] auto operator()(const VectorType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const ComplexType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const AtomicType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const TemplateTypeParameterType* type) const
      -> bool {
    if (alreadyExamined(type)) return false;
    if (std::ranges::any_of(type->templateParameters(), std::ref(*this)))
      return true;
    recordExamined(type);
    return false;
  }

  [[nodiscard]] auto operator()(const FunctionType* type) const -> bool {
    if (alreadyExamined(type)) return false;
    if (type->noexceptExpression()) return true;
    if (visit(*this, type->returnType())) return true;
    if (std::ranges::any_of(type->parameterTypes(), std::ref(*this)))
      return true;
    recordExamined(type);
    return false;
  }

  [[nodiscard]] auto operator()(const Type* type) const -> bool {
    return type && visit(*this, type);
  }

  template <typename T>
  [[nodiscard]] auto operator()(const T*) const -> bool {
    return false;
  }
};

[[nodiscard]] auto holds_ast(const Type* type) -> bool {
  return HoldsAST{}(type);
}

[[nodiscard]] auto same_argument_type(TranslationUnit* unit, const Type* lhs,
                                      const Type* rhs) -> bool {
  if (lhs == rhs) return true;
  if (!holds_ast(lhs) || !holds_ast(rhs)) return false;
  return unit->typeTraits().is_same(lhs, rhs);
}

[[nodiscard]] auto same_non_type_parameter(TranslationUnit* unit, Symbol* lhs,
                                           Symbol* rhs) -> bool {
  if (nonTypeParameterIdentity(lhs) != nonTypeParameterIdentity(rhs))
    return false;
  auto lhsParameter = symbol_cast<NonTypeParameterSymbol>(lhs);
  auto rhsParameter = symbol_cast<NonTypeParameterSymbol>(rhs);
  return same_argument_type(unit, unqualified_type(lhsParameter->objectType()),
                            unqualified_type(rhsParameter->objectType()));
}

auto compare_symbols(TranslationUnit* unit, Symbol* lhs, Symbol* rhs) -> bool {
  if (lhs == rhs) return true;
  if (!lhs || !rhs) return false;

  if (nonTypeParameterIdentity(lhs) || nonTypeParameterIdentity(rhs))
    return same_non_type_parameter(unit, lhs, rhs);

  auto lhsPack = symbol_cast<ParameterPackSymbol>(lhs);
  auto rhsPack = symbol_cast<ParameterPackSymbol>(rhs);
  if (lhsPack || rhsPack) {
    if (!lhsPack || !rhsPack) return false;
    if (lhsPack->elements().size() != rhsPack->elements().size()) return false;
    for (size_t i = 0; i < lhsPack->elements().size(); ++i) {
      if (!compare_symbols(unit, lhsPack->elements()[i],
                           rhsPack->elements()[i])) {
        return false;
      }
    }
    return true;
  }

  auto lhsTemplateName = template_name_symbol(lhs);
  auto rhsTemplateName = template_name_symbol(rhs);
  if (lhsTemplateName || rhsTemplateName) {
    if (!lhsTemplateName || !rhsTemplateName) return false;
    return lhsTemplateName->canonical() == rhsTemplateName->canonical();
  }

  auto lhsVar = symbol_cast<VariableSymbol>(lhs);
  auto rhsVar = symbol_cast<VariableSymbol>(rhs);
  if (bool(lhsVar) != bool(rhsVar)) return false;
  if (lhsVar && rhsVar) {
    if (lhsVar->constValue().has_value() != rhsVar->constValue().has_value())
      return false;

    if (lhsVar->constValue().has_value()) {
      if (!equivalent_values(lhsVar->constValue().value(),
                             rhsVar->constValue().value()))
        return false;
    } else if (!TemplateEquivalence{unit}.same(lhsVar->initializer(),
                                               rhsVar->initializer())) {
      return false;
    }
  }

  return same_argument_type(unit, lhs->type(), rhs->type());
}

auto compare_symbol_and_type(TranslationUnit* unit, Symbol* symbol,
                             const Type* type) -> bool {
  if (!symbol || !type) return false;
  if (symbol_cast<VariableSymbol>(symbol)) return false;
  if (symbol_cast<NonTypeParameterSymbol>(symbol)) return false;
  if (template_name_symbol(symbol)) return false;
  return same_argument_type(unit, symbol->type(), type);
}

auto compare_symbol_and_const(Symbol* symbol, const ConstValue& value) -> bool {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable) return false;
  if (!variable->constValue().has_value()) return false;
  return equivalent_values(variable->constValue().value(), value);
}

}  // namespace

auto compare_single_arg(TranslationUnit* unit, const TemplateArgument& lhs,
                        const TemplateArgument& rhs) -> bool {
  if (auto lhsType = std::get_if<const Type*>(&lhs)) {
    if (auto rhsType = std::get_if<const Type*>(&rhs)) {
      return same_argument_type(unit, *lhsType, *rhsType);
    }
    if (auto rhsSymbol = std::get_if<Symbol*>(&rhs)) {
      return compare_symbol_and_type(unit, *rhsSymbol, *lhsType);
    }
    return false;
  }

  if (auto lhsSymbol = std::get_if<Symbol*>(&lhs)) {
    if (auto rhsSymbol = std::get_if<Symbol*>(&rhs)) {
      return compare_symbols(unit, *lhsSymbol, *rhsSymbol);
    }
    if (auto rhsType = std::get_if<const Type*>(&rhs)) {
      return compare_symbol_and_type(unit, *lhsSymbol, *rhsType);
    }
    if (auto rhsValue = std::get_if<ConstValue>(&rhs)) {
      return compare_symbol_and_const(*lhsSymbol, *rhsValue);
    }
    return false;
  }

  if (auto lhsValue = std::get_if<ConstValue>(&lhs)) {
    if (auto rhsValue = std::get_if<ConstValue>(&rhs)) {
      return equivalent_values(*lhsValue, *rhsValue);
    }
    if (auto rhsSymbol = std::get_if<Symbol*>(&rhs)) {
      return compare_symbol_and_const(*rhsSymbol, *lhsValue);
    }
    return false;
  }

  if (auto lhsExpr = std::get_if<ExpressionAST*>(&lhs)) {
    auto rhsExpr = std::get_if<ExpressionAST*>(&rhs);
    if (!rhsExpr) return false;
    if (*lhsExpr == *rhsExpr) return true;
    return TemplateEquivalence{unit}.same(*lhsExpr, *rhsExpr);
  }

  return false;
}

auto compare_args(TranslationUnit* unit,
                  std::span<const TemplateArgument> args1,
                  std::span<const TemplateArgument> args2) -> bool {
  if (args1.size() != args2.size()) return false;

  for (size_t i = 0; i < args1.size(); ++i) {
    if (!compare_single_arg(unit, args1[i], args2[i])) return false;
  }

  return true;
}

auto SpecializationTable::find(
    TranslationUnit* unit, std::span<const TemplateArgument> arguments) const
    -> Symbol* {
  auto index = findIndex(unit, arguments, nullptr);
  if (!index) return nullptr;
  return entries_[*index].symbol;
}

auto SpecializationTable::entryOf(const Symbol* specialization)
    -> TemplateSpecialization* {
  auto it = bySymbol_.find(specialization);
  if (it == bySymbol_.end()) return nullptr;
  return &entries_[it->second];
}

auto SpecializationTable::add(TranslationUnit* unit,
                              std::vector<TemplateArgument> arguments,
                              Symbol* specialization) -> std::size_t {
  if (auto existing = findIndex(unit, arguments, specialization))
    return *existing;
  entries_.push_back(
      {.arguments = std::move(arguments), .symbol = specialization});
  index(entries_.size() - 1);
  return entries_.size() - 1;
}

void SpecializationTable::restore(TemplateSpecialization specialization) {
  entries_.push_back(std::move(specialization));
}

void SpecializationTable::rebuildIndex() {
  byArguments_.clear();
  bySymbol_.clear();
  unkeyed_.clear();
  for (std::size_t i = 0; i < entries_.size(); ++i) index(i);
}

auto SpecializationTable::matches(TranslationUnit* unit, std::size_t index,
                                  std::span<const TemplateArgument> arguments,
                                  const Symbol* specialization) const -> bool {
  const auto& entry = entries_[index];
  if (auto trace = unit->timeTrace())
    trace->count(TimeTrace::kSpecializationComparisons);
  if (specialization && entry.symbol != specialization) return false;
  if (std::ranges::equal(entry.arguments, arguments)) return true;
  return compare_args(unit, entry.arguments, arguments);
}

auto SpecializationTable::findIndex(TranslationUnit* unit,
                                    std::span<const TemplateArgument> arguments,
                                    const Symbol* specialization) const
    -> std::optional<std::size_t> {
  return findTemplateArguments(arguments, entries_.size(), byArguments_,
                               unkeyed_, [&](std::size_t index) {
                                 return matches(unit, index, arguments,
                                                specialization);
                               });
}

void SpecializationTable::index(std::size_t index) {
  auto position = static_cast<std::uint32_t>(index);
  bySymbol_.try_emplace(entries_[index].symbol, position);
  indexTemplateArguments(entries_[index].arguments, position, byArguments_,
                         unkeyed_);
}

void indexTemplateArguments(
    std::span<const TemplateArgument> arguments, std::uint32_t position,
    std::unordered_map<std::size_t, std::vector<std::uint32_t>>& byArguments,
    std::vector<std::uint32_t>& unkeyed) {
  auto key = hash_template_arguments(arguments);
  if (key.has_value())
    byArguments[*key].push_back(position);
  else
    unkeyed.push_back(position);
}

auto expand_template_arguments(std::span<const TemplateArgument> arguments)
    -> std::vector<TemplateArgument> {
  std::vector<TemplateArgument> expanded;
  for (auto& entry : expand_template_arguments_with_sources(arguments))
    expanded.push_back(std::move(entry.value));
  return expanded;
}

auto expand_template_arguments_with_sources(
    std::span<const TemplateArgument> arguments)
    -> std::vector<ExpandedTemplateArgument> {
  std::vector<ExpandedTemplateArgument> expanded;
  std::vector<ExpandedTemplateArgument> pending;
  pending.reserve(arguments.size());

  for (std::size_t index = arguments.size(); index > 0; --index)
    pending.push_back({arguments[index - 1], index - 1});

  while (!pending.empty()) {
    auto entry = std::move(pending.back());
    pending.pop_back();
    if (auto symbol = std::get_if<Symbol*>(&entry.value)) {
      if (auto pack = symbol_cast<ParameterPackSymbol>(*symbol)) {
        auto& elements = pack->elements();
        for (std::size_t index = elements.size(); index > 0; --index)
          pending.push_back(
              {TemplateArgument{elements[index - 1]}, entry.sourceIndex});
        continue;
      }
    }
    expanded.push_back(std::move(entry));
  }

  return expanded;
}

auto template_argument_type(const TemplateArgument& argument) -> const Type* {
  if (auto type = std::get_if<const Type*>(&argument)) return *type;
  if (auto symbol = std::get_if<Symbol*>(&argument)) return (*symbol)->type();
  return nullptr;
}

auto template_argument_as_type(const TemplateArgument& argument)
    -> const Type* {
  if (auto type = std::get_if<const Type*>(&argument)) return *type;
  auto symbol = std::get_if<Symbol*>(&argument);
  if (!symbol || !is_type(*symbol)) return nullptr;
  return (*symbol)->type();
}

auto template_argument_parameter_info(const TemplateArgument& argument)
    -> std::optional<TypeParamInfo> {
  if (auto symbol = std::get_if<Symbol*>(&argument))
    return template_parameter_info(*symbol);
  if (auto type = std::get_if<const Type*>(&argument))
    return getTypeParamInfo(*type);
  return std::nullopt;
}

auto class_template_of(ClassSymbol* classSymbol) -> ClassSymbol* {
  if (!classSymbol) return nullptr;
  if (classSymbol->isSpecialization())
    return classSymbol->primaryTemplateSymbol();
  if (classSymbol->templateDeclaration()) return classSymbol;
  return nullptr;
}

auto template_argument_value(const TemplateArgument& argument)
    -> std::optional<ConstValue> {
  if (auto value = std::get_if<ConstValue>(&argument)) return *value;

  if (auto symbol = std::get_if<Symbol*>(&argument)) {
    if (auto variable = symbol_cast<VariableSymbol>(*symbol)) {
      if (auto value = variable->constValue()) return *value;
    }
  }

  return std::nullopt;
}

namespace {
struct TemplateArgumentHasher {
  std::size_t seed;

  [[nodiscard]] auto operator()(const Type* type) -> bool {
    auto key = holds_ast(type) ? 0 : std::hash<const void*>{}(type);
    hash_combine(seed, key);
    return true;
  }

  [[nodiscard]] auto operator()(const ConstValue& constant) -> bool {
    auto value = std::get_if<ConstInt>(&constant);
    if (!value) return false;
    hash_combine(seed, value->hash());
    return true;
  }

  [[nodiscard]] auto operator()(ExpressionAST*) -> bool { return false; }

  [[nodiscard]] auto operator()(Symbol* symbol) -> bool {
    if (!symbol) return false;
    if (auto templateName = template_name_symbol(symbol)) {
      hash_combine(seed, std::hash<const void*>{}(templateName->canonical()));
      return true;
    }
    return visit(*this, symbol);
  }

  [[nodiscard]] auto operator()(NonTypeParameterSymbol* symbol) -> bool {
    hash_combine(seed, symbol->depth());
    hash_combine(seed, symbol->index());
    hash_combine(seed, symbol->isParameterPack());
    return (*this)(unqualified_type(symbol->objectType()));
  }

  [[nodiscard]] auto operator()(VariableSymbol* symbol) -> bool {
    if (auto value = symbol->constValue();
        value && std::holds_alternative<ConstInt>(*value))
      return (*this)(*value);
    if (!symbol->type()) return false;
    return (*this)(symbol->type());
  }

  [[nodiscard]] auto operator()(ParameterPackSymbol* pack) -> bool {
    if (pack->type()) return false;
    hash_combine(seed, pack->elements().size());
    for (auto element : pack->elements()) {
      if (!(*this)(element)) return false;
    }
    return true;
  }

  template <typename S>
  [[nodiscard]] auto operator()(S* symbol) -> bool {
    if (!symbol->type()) return false;
    return (*this)(symbol->type());
  }
};
}  // namespace

auto hash_template_arguments(std::span<const TemplateArgument> arguments)
    -> std::optional<std::size_t> {
  TemplateArgumentHasher hasher{arguments.size()};
  for (const auto& argument : arguments) {
    if (!std::visit(hasher, argument)) return std::nullopt;
  }
  return hasher.seed;
}

auto template_name_symbol(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;

  if (auto alias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (alias->isTemplatePattern()) return alias;
    if (auto classType = unqualified_cast<ClassType>(alias->type())) {
      return template_name_symbol(classType->symbol());
    }
    return nullptr;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (classSymbol->isSpecialization()) return nullptr;
    return classSymbol->templateParameters() ? classSymbol : nullptr;
  }

  if (symbol_cast<TemplateTypeParameterSymbol>(symbol)) return symbol;

  return nullptr;
}

auto resolve_using_declaration(Symbol* symbol) -> Symbol* {
  while (auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(symbol)) {
    symbol = usingDeclaration->target();
  }
  return symbol;
}

auto names_functions(Symbol* symbol) -> bool {
  if (!symbol) return true;
  if (symbol_cast<OverloadSetSymbol>(symbol)) return true;
  return symbol_cast<FunctionSymbol>(symbol) != nullptr;
}

auto resolve_namespace_alias(Symbol* symbol) -> NamespaceSymbol* {
  symbol = resolve_using_declaration(symbol);
  while (auto alias = symbol_cast<NamespaceAliasSymbol>(symbol)) {
    symbol = alias->namespaceSymbol();
  }
  return symbol_cast<NamespaceSymbol>(symbol);
}

auto templated_symbol(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;

  symbol = resolve_using_declaration(symbol);
  if (!symbol) return nullptr;

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    for (auto function : overloadSet->functions()) {
      if (auto result = templated_symbol(function)) return result;
    }
    return nullptr;
  }

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    auto classSymbol = injected->classSymbol();
    if (!classSymbol) return nullptr;
    if (auto primary = classSymbol->primaryTemplateSymbol())
      classSymbol = primary;
    return templated_symbol(classSymbol);
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (classSymbol->isSpecialization())
      return templated_symbol(classSymbol->primaryTemplateSymbol());
  }

  if (auto alias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (alias->isSpecialization())
      return templated_symbol(alias->primaryTemplateSymbol());
  }

  if (symbol_cast<TemplateTypeParameterSymbol>(symbol)) return symbol;
  if (template_parameters_of(symbol) || template_declaration_of(symbol))
    return symbol;
  return nullptr;
}

auto Symbol::EnclosingSymbolIterator::operator++() -> EnclosingSymbolIterator& {
  symbol_ = symbol_->parent();
  return *this;
}

auto Symbol::EnclosingSymbolIterator::operator++(int)
    -> EnclosingSymbolIterator {
  auto it = *this;
  symbol_ = symbol_->parent();
  return it;
}

auto Symbol::hasEnclosingSymbol(Symbol* symbol) const -> bool {
  for (auto enclosingSymbol : enclosingSymbols()) {
    if (enclosingSymbol == symbol) return true;
  }
  return false;
}

auto Symbol::kind() const -> SymbolKind { return kind_; }

auto Symbol::name() const -> const Name* { return name_; }

void Symbol::setName(const Name* name) { name_ = name; }

auto Symbol::type() const -> const Type* { return type_; }

void Symbol::setType(const Type* type) { type_ = type; }

auto Symbol::location() const -> SourceLocation { return location_; }

void Symbol::setLocation(SourceLocation location) { location_ = location; }

auto Symbol::parent() const -> ScopeSymbol* { return parent_; }

auto Symbol::isWeak() const -> bool {
  return findAttribute(canonical()->attributes(), "weak") != nullptr;
}

auto Symbol::isNoInline() const -> bool {
  return findAttribute(canonical()->attributes(), "noinline") != nullptr;
}

auto Symbol::isAlwaysInline() const -> bool {
  return findAttribute(canonical()->attributes(), "always_inline") != nullptr;
}

auto Symbol::abiTags() const -> std::span<const Identifier* const> {
  if (!abiTags_) return {};
  return *abiTags_;
}

void Symbol::setAbiTags(const std::vector<const Identifier*>* abiTags) {
  abiTags_ = abiTags;
}

void Symbol::setParent(ScopeSymbol* enclosingScope) {
  if (enclosingScope && enclosingScope->isTemplateParameters()) {
    switch (kind()) {
      case SymbolKind::kTypeParameter:
      case SymbolKind::kNonTypeParameter:
      case SymbolKind::kTemplateTypeParameter:
      case SymbolKind::kConstraintTypeParameter:
      case SymbolKind::kFunctionParameters:
      case SymbolKind::kTemplateParameters:
        break;
      default:
        cxx_runtime_error(std::format(
            "symbol kind '{}' may not have TemplateParametersSymbol as parent",
            static_cast<int>(kind())));
    }
  }
  parent_ = enclosingScope;
}

auto Symbol::next() const -> Symbol* {
  for (auto sym = link_; sym; sym = sym->link_) {
    if (sym->name_ == name_) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingNamespace() const -> NamespaceSymbol* {
  for (auto scope = parent(); scope; scope = scope->parent()) {
    if (auto ns = symbol_cast<NamespaceSymbol>(scope)) {
      return ns;
    }
  }
  return nullptr;
}

auto Symbol::enclosingClass() const -> ClassSymbol* {
  for (auto scope = parent(); scope; scope = scope->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
      return classSymbol;
    }
  }
  return nullptr;
}

auto is_unnamed_namespace(Symbol* symbol) -> bool {
  auto ns = symbol_cast<NamespaceSymbol>(symbol);
  return ns && ns->anonNamespaceIndex().has_value();
}

auto is_in_unnamed_namespace(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (is_unnamed_namespace(symbol)) return true;
  for (auto scope : symbol->enclosingSymbols()) {
    if (is_unnamed_namespace(scope)) return true;
  }
  return false;
}

auto is_declared_extern(VariableSymbol* variable) -> bool {
  return std::ranges::any_of(variable->declarations(),
                             &VariableSymbol::isExtern);
}

auto has_internal_linkage(Symbol* symbol) -> bool {
  if (!symbol) return false;

  if (is_in_unnamed_namespace(symbol)) return true;

  auto parent = symbol->parent();
  if (!parent || !parent->isNamespace()) return false;

  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return function->isStatic();

  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable) return false;

  if (variable->isStatic()) return true;
  if (variable->isInline() || is_declared_extern(variable)) return false;
  if (variable->templateParameters() || variable->isSpecialization())
    return false;

  const auto cv = cv_qualifiers(variable->type());
  return has_const(cv) && !has_volatile(cv);
}

auto Symbol::enclosingFunction() const -> FunctionSymbol* {
  for (auto scope = parent(); scope; scope = scope->parent()) {
    if (auto func = symbol_cast<FunctionSymbol>(scope)) return func;
    auto lambda = symbol_cast<LambdaSymbol>(scope);
    if (!lambda || !lambda->closureType()) continue;
    if (auto callOperator = lambda->closureType()->functionCallOperator())
      return callOperator;
  }
  return nullptr;
}

auto Symbol::enclosingFunctionOrSelf() const -> FunctionSymbol* {
  if (auto func = symbol_cast<FunctionSymbol>(const_cast<Symbol*>(this)))
    return func;
  return enclosingFunction();
}

namespace {
template <typename S, typename D>
void acceptsMaybeTemplate(const MaybeTemplate<S, D>&);

template <typename S>
concept Templatable = requires(const S& s) { acceptsMaybeTemplate(s); };

template <typename S>
void acceptsMaybeRedecl(const MaybeRedecl<S>&);

template <typename S>
concept Redeclarable = requires(const S& s) { acceptsMaybeRedecl(s); };

struct GetCanonical {
  template <Redeclarable S>
  auto operator()(S* symbol) const -> Symbol* {
    return symbol->canonical();
  }

  auto operator()(Symbol* symbol) const -> Symbol* { return symbol; }
};

struct GetDefinition {
  template <Redeclarable S>
  auto operator()(S* symbol) const -> Symbol* {
    return symbol->definition();
  }

  auto operator()(Symbol*) const -> Symbol* { return nullptr; }
};

struct FindSpecialization {
  TranslationUnit* unit;
  std::span<const TemplateArgument> arguments;

  template <Templatable S>
  auto operator()(S* symbol) const -> Symbol* {
    return symbol->findSpecialization(unit, arguments);
  }

  auto operator()(Symbol*) const -> Symbol* { return nullptr; }
};

struct GetSpecializationEntry {
  const Symbol* specialization;

  template <Templatable S>
  auto operator()(S* symbol) const -> TemplateSpecialization* {
    return symbol->specializationEntry(specialization);
  }

  auto operator()(Symbol*) const -> TemplateSpecialization* { return nullptr; }
};

struct GetTemplateDeclaration {
  template <Templatable S>
  auto operator()(S* symbol) const -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(Symbol*) const -> TemplateDeclarationAST* { return nullptr; }
};

struct RebuildSpecializationIndex {
  template <Templatable S>
  void operator()(S* symbol) {
    symbol->rebuildSpecializationIndex();
  }

  void operator()(Symbol*) {}
};

struct AddExternInstantiationDeclaration {
  std::vector<TemplateArgument> arguments;

  template <Templatable S>
  void operator()(S* symbol) {
    symbol->addExternInstantiationDeclaration(std::move(arguments));
  }

  void operator()(Symbol*) {}
};

struct GetTemplateArguments {
  template <Templatable S>
  auto operator()(S* symbol) const -> std::span<const TemplateArgument> {
    return symbol->templateArguments();
  }

  auto operator()(Symbol*) const -> std::span<const TemplateArgument> {
    return {};
  }
};

struct GetTemplateParameters {
  template <Templatable S>
  auto operator()(S* symbol) const -> TemplateParametersSymbol* {
    return symbol->templateParameters();
  }

  auto operator()(Symbol*) const -> TemplateParametersSymbol* {
    return nullptr;
  }
};

struct GetPrimaryTemplate {
  template <Templatable S>
  auto operator()(S* symbol) const -> Symbol* {
    return symbol->primaryTemplateSymbol();
  }

  auto operator()(Symbol*) const -> Symbol* { return nullptr; }
};

struct GetTemplateDeclarationAST {
  auto operator()(ClassSymbol* symbol) const -> AST* {
    return symbol->declaration();
  }

  auto operator()(VariableSymbol* symbol) const -> AST* {
    auto templateDeclaration = symbol->templateDeclaration();
    return templateDeclaration ? templateDeclaration->declaration : nullptr;
  }

  auto operator()(TypeAliasSymbol* symbol) const -> AST* {
    auto templateDeclaration = symbol->templateDeclaration();
    return templateDeclaration ? templateDeclaration->declaration : nullptr;
  }

  auto operator()(FunctionSymbol* symbol) const -> AST* {
    if (auto declaration = symbol->declaration()) return declaration;
    if (auto templateDeclaration = symbol->templateDeclaration())
      return templateDeclaration->declaration;
    return nullptr;
  }

  auto operator()(Symbol*) const -> AST* { return nullptr; }
};

struct GetTemplateParameterInfo {
  auto operator()(TypeParameterSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return getTypeParamInfo(symbol->type());
  }

  auto operator()(TemplateTypeParameterSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return getTypeParamInfo(symbol->type());
  }

  auto operator()(NonTypeParameterSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return TypeParamInfo{symbol->index(), symbol->depth(),
                         symbol->isParameterPack()};
  }

  auto operator()(ConstraintTypeParameterSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return TypeParamInfo{symbol->index(), symbol->depth(),
                         symbol->isParameterPack()};
  }

  auto operator()(TypeAliasSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return getTypeParamInfo(symbol->type());
  }

  auto operator()(VariableSymbol* symbol) const
      -> std::optional<TypeParamInfo> {
    return getTypeParamInfo(symbol->type());
  }

  auto operator()(Symbol*) const -> std::optional<TypeParamInfo> {
    return std::nullopt;
  }
};
}  // namespace

auto template_declaration_of(Symbol* symbol) -> TemplateDeclarationAST* {
  if (!symbol) return nullptr;
  return visit(GetTemplateDeclaration{}, symbol);
}

auto Symbol::canonical() const -> Symbol* {
  return visit(GetCanonical{}, const_cast<Symbol*>(this));
}

auto find_specialization(TranslationUnit* unit, Symbol* templateSymbol,
                         std::span<const TemplateArgument> arguments)
    -> Symbol* {
  if (!templateSymbol) return nullptr;
  return visit(FindSpecialization{unit, arguments}, templateSymbol);
}

auto specialization_entry_of(Symbol* templateSymbol, Symbol* specialization)
    -> TemplateSpecialization* {
  if (!templateSymbol || !specialization) return nullptr;
  return visit(GetSpecializationEntry{specialization}, templateSymbol);
}

void rebuild_specialization_index(Symbol* symbol) {
  if (!symbol) return;
  visit(RebuildSpecializationIndex{}, symbol);
}

void add_extern_instantiation_declaration(
    Symbol* symbol, std::vector<TemplateArgument> arguments) {
  if (!symbol) return;
  visit(AddExternInstantiationDeclaration{std::move(arguments)}, symbol);
}

auto template_parameters_of(Symbol* symbol) -> TemplateParametersSymbol* {
  if (!symbol) return nullptr;
  return visit(GetTemplateParameters{}, symbol);
}

auto template_arguments_of(Symbol* symbol)
    -> std::span<const TemplateArgument> {
  if (!symbol) return {};
  return visit(GetTemplateArguments{}, symbol);
}

auto primary_template_of(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  return visit(GetPrimaryTemplate{}, symbol);
}

auto template_declaration_ast(Symbol* symbol) -> AST* {
  if (!symbol) return nullptr;
  return visit(GetTemplateDeclarationAST{}, symbol);
}

auto template_parameter_info(Symbol* symbol) -> std::optional<TypeParamInfo> {
  if (!symbol) return std::nullopt;
  return visit(GetTemplateParameterInfo{}, symbol);
}

auto is_template_parameter_pack(Symbol* symbol) -> bool {
  auto info = template_parameter_info(symbol);
  return info && info->isPack;
}

auto is_callable_with_one_argument(FunctionSymbol* function) -> bool {
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType || functionType->parameterTypes().empty()) return false;

  const auto parameterCount =
      static_cast<int>(functionType->parameterTypes().size());

  return required_parameter_count(function, parameterCount) <= 1;
}

struct GetDefaultTemplateArgument {
  auto operator()(TypeParameterSymbol* symbol) const -> TemplateParameterAST* {
    return symbol->defaultArgument();
  }

  auto operator()(NonTypeParameterSymbol* symbol) const
      -> TemplateParameterAST* {
    return symbol->defaultArgument();
  }

  auto operator()(TemplateTypeParameterSymbol* symbol) const
      -> TemplateParameterAST* {
    return symbol->defaultArgument();
  }

  auto operator()(ConstraintTypeParameterSymbol* symbol) const
      -> TemplateParameterAST* {
    return symbol->defaultArgument();
  }

  auto operator()(Symbol*) const -> TemplateParameterAST* { return nullptr; }
};

struct SetDefaultTemplateArgument {
  TemplateParameterAST* defaultArgument;

  void operator()(TypeParameterSymbol* symbol) const {
    symbol->setDefaultArgument(defaultArgument);
  }

  void operator()(NonTypeParameterSymbol* symbol) const {
    symbol->setDefaultArgument(defaultArgument);
  }

  void operator()(TemplateTypeParameterSymbol* symbol) const {
    symbol->setDefaultArgument(defaultArgument);
  }

  void operator()(ConstraintTypeParameterSymbol* symbol) const {
    symbol->setDefaultArgument(defaultArgument);
  }

  void operator()(Symbol*) const {}
};

auto default_template_argument(Symbol* symbol) -> TemplateParameterAST* {
  if (!symbol) return nullptr;
  return visit(GetDefaultTemplateArgument{}, symbol);
}

void set_default_template_argument(Symbol* symbol,
                                   TemplateParameterAST* defaultArgument) {
  if (!symbol) return;
  visit(SetDefaultTemplateArgument{defaultArgument}, symbol);
}

auto required_parameter_count(FunctionSymbol* function, int parameterCount)
    -> int {
  if (!function) return parameterCount;

  auto parameters = function->parameters();
  if (parameters.empty()) return parameterCount;

  int defaultCount = 0;
  for (auto parameter : parameters | std::views::reverse) {
    if (!parameter->hasDefaultArgument()) break;
    ++defaultCount;
  }

  return parameterCount - defaultCount;
}

auto is_non_static_member(Symbol* symbol) -> bool {
  if (auto field = symbol_cast<FieldSymbol>(symbol)) return !field->isStatic();
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return !function->isStatic();
  if (symbol_cast<OverloadSetSymbol>(symbol)) {
    return std::ranges::any_of(
        views::each_function(symbol),
        [](FunctionSymbol* function) { return !function->isStatic(); });
  }
  return false;
}

auto is_function_local_predefined_variable(Symbol* symbol) -> bool {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  return variable && variable->isFunctionLocalPredefined();
}

auto introduces_variable(Symbol* symbol) -> bool {
  if (symbol_cast<VariableSymbol>(symbol)) return true;
  if (symbol_cast<ParameterSymbol>(symbol)) return true;
  if (auto field = symbol_cast<FieldSymbol>(symbol)) return field->isStatic();
  return false;
}

auto is_templated_class(ClassSymbol* classSymbol) -> bool {
  for (Symbol* symbol = classSymbol; symbol; symbol = symbol->parent()) {
    auto enclosingClass = symbol_cast<ClassSymbol>(symbol);
    if (!enclosingClass) continue;
    if (enclosingClass->templateParameters()) return true;
    if (enclosingClass->isSpecialization()) return true;
  }
  return false;
}

auto anonymous_member_class(FieldSymbol* field) -> ClassSymbol* {
  if (field->name()) return nullptr;
  auto classType = unqualified_cast<ClassType>(field->type());
  if (!classType) return nullptr;
  auto classSymbol = classType->symbol();
  if (classSymbol->name()) return nullptr;
  return classSymbol;
}

auto resolved_base_class(BaseClassSymbol* baseClass) -> ClassSymbol* {
  auto classSymbol = symbol_cast<ClassSymbol>(baseClass->symbol());
  if (!classSymbol) return nullptr;
  return classSymbol->resolvedDefinition();
}

auto virtual_base_initialization_order(ClassSymbol* classSymbol)
    -> std::vector<ClassSymbol*> {
  struct Frame {
    ClassSymbol* classSymbol = nullptr;
    bool isVirtual = false;
    std::size_t nextBase = 0;
  };

  std::vector<ClassSymbol*> order;
  std::vector<Frame> frames{{classSymbol}};
  while (!frames.empty()) {
    auto& frame = frames.back();
    const auto& bases = frame.classSymbol->baseClasses();
    if (frame.nextBase < bases.size()) {
      auto base = bases[frame.nextBase++];
      if (auto baseClass = resolved_base_class(base))
        frames.push_back({baseClass, base->isVirtual()});
      continue;
    }
    if (frame.isVirtual && !std::ranges::contains(order, frame.classSymbol))
      order.push_back(frame.classSymbol);
    frames.pop_back();
  }
  return order;
}

auto is_anonymous_union_member(FieldSymbol* field) -> bool {
  auto anonymous = anonymous_member_class(field);
  return anonymous && anonymous->isUnion();
}

auto has_variant_members(ClassSymbol* classSymbol) -> bool {
  if (classSymbol->isUnion()) return true;
  return std::ranges::any_of(
      views::members(classSymbol) | views::non_static_fields,
      is_anonymous_union_member);
}

auto is_inline_or_templated(FunctionSymbol* function) -> bool {
  if (!function) return false;
  if (!function->isSpecialization()) function = function->canonical();
  if (function->isInline()) return true;
  if (function->templateDeclaration() || function->isSpecialization())
    return true;
  for (auto scope = function->parent(); scope; scope = scope->parent()) {
    if (auto enclosingClass = symbol_cast<ClassSymbol>(scope))
      return is_templated_class(enclosingClass);
  }
  return false;
}

auto has_static_storage_duration(Symbol* symbol) -> bool {
  if (auto field = symbol_cast<FieldSymbol>(symbol))
    return field->isStatic() && !field->isThreadLocal();
  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable || variable->isThreadLocal()) return false;
  if (variable->isStatic() || variable->isExtern()) return true;
  return !variable->enclosingFunction();
}

auto closure_mangling_context(ClassSymbol* closure) -> FunctionSymbol* {
  auto context = closure->enclosingFunction();
  if (!is_inline_or_templated(context)) return nullptr;
  return context;
}

namespace {

[[nodiscard]] auto is_local_entity_with_internal_linkage(Symbol* symbol)
    -> bool {
  auto function = symbol->enclosingFunction();
  if (!function) return false;
  if (has_internal_linkage(function)) return true;
  return !is_inline_or_templated(function);
}

[[nodiscard]] auto is_unnamed_namespace_scope_type(Symbol* symbol) -> bool {
  if (symbol->name()) return false;
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol);
      classSymbol && classSymbol->isClosureType())
    return false;
  auto parent = symbol->parent();
  return parent && parent->isNamespace();
}

[[nodiscard]] auto class_has_internal_linkage(ClassSymbol* classSymbol)
    -> bool {
  if (is_in_unnamed_namespace(classSymbol)) return true;
  if (classSymbol->isClosureType() && !closure_mangling_context(classSymbol))
    return true;
  if (is_unnamed_namespace_scope_type(classSymbol)) return true;
  if (is_local_entity_with_internal_linkage(classSymbol)) return true;
  return is_specialized_on_internal_type(classSymbol);
}

[[nodiscard]] auto enum_has_internal_linkage(Symbol* enumSymbol) -> bool {
  if (is_in_unnamed_namespace(enumSymbol)) return true;
  if (is_unnamed_namespace_scope_type(enumSymbol)) return true;
  return is_local_entity_with_internal_linkage(enumSymbol);
}

struct TypeHasInternalLinkage {
  [[nodiscard]] auto operator()(const ClassType* type) const -> bool {
    return class_has_internal_linkage(type->symbol());
  }
  [[nodiscard]] auto operator()(const EnumType* type) const -> bool {
    return enum_has_internal_linkage(type->symbol());
  }
  [[nodiscard]] auto operator()(const ScopedEnumType* type) const -> bool {
    return enum_has_internal_linkage(type->symbol());
  }
  [[nodiscard]] auto operator()(const QualType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const PointerType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const LvalueReferenceType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const RvalueReferenceType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const BoundedArrayType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const UnboundedArrayType* type) const -> bool {
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const FunctionType* type) const -> bool {
    if (type_has_internal_linkage(type->returnType())) return true;
    return std::ranges::any_of(type->parameterTypes(),
                               type_has_internal_linkage);
  }
  [[nodiscard]] auto operator()(const MemberObjectPointerType* type) const
      -> bool {
    if (type_has_internal_linkage(type->classType())) return true;
    return type_has_internal_linkage(type->elementType());
  }
  [[nodiscard]] auto operator()(const MemberFunctionPointerType* type) const
      -> bool {
    if (type_has_internal_linkage(type->classType())) return true;
    return type_has_internal_linkage(type->functionType());
  }
  [[nodiscard]] auto operator()(const Type*) const -> bool { return false; }
};

[[nodiscard]] auto argument_has_internal_linkage(
    const TemplateArgument& argument) -> bool {
  if (auto type = std::get_if<const Type*>(&argument))
    return type_has_internal_linkage(*type);
  auto symbol = std::get_if<Symbol*>(&argument);
  if (!symbol || !*symbol) return false;
  if (auto pack = symbol_cast<ParameterPackSymbol>(*symbol)) {
    return std::ranges::any_of(pack->elements(), [](Symbol* element) {
      return argument_has_internal_linkage(TemplateArgument{element});
    });
  }
  if (!symbol_cast<TypeAliasSymbol>(*symbol)) return false;
  return type_has_internal_linkage((*symbol)->type());
}

}  // namespace

auto type_has_internal_linkage(const Type* type) -> bool {
  if (!type) return false;
  return visit(TypeHasInternalLinkage{}, type);
}

auto is_declared_with_internal_type(Symbol* symbol) -> bool {
  if (auto enclosingClass = symbol_cast<ClassSymbol>(symbol->parent()))
    return type_has_internal_linkage(enclosingClass->type());

  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function) return false;

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return false;

  if (!function->hasDeducedReturnType() &&
      type_has_internal_linkage(functionType->returnType()))
    return true;

  return std::ranges::any_of(functionType->parameterTypes(),
                             type_has_internal_linkage);
}

auto is_specialized_on_internal_type(Symbol* symbol) -> bool {
  for (auto scope = symbol; scope; scope = scope->parent()) {
    auto arguments = visit(GetTemplateArguments{}, scope);
    if (std::ranges::any_of(arguments, argument_has_internal_linkage))
      return true;
  }
  return false;
}

auto is_member_template(Symbol* symbol) -> bool {
  if (!template_declaration_of(symbol)) return false;
  return symbol_cast<ClassSymbol>(symbol->parent()) != nullptr;
}

namespace {
[[nodiscard]] auto denotes_template_parameter(Symbol* symbol, Symbol* parameter)
    -> bool {
  auto symbolInfo = template_parameter_info(symbol);
  auto parameterInfo = template_parameter_info(parameter);
  if (!symbolInfo || !parameterInfo) return false;
  if (symbolInfo->depth != parameterInfo->depth) return false;
  if (symbolInfo->index != parameterInfo->index) return false;
  return symbolInfo->isPack == parameterInfo->isPack;
}

[[nodiscard]] auto is_equivalent_to_template_parameter(
    TemplateArgumentAST* argument, Symbol* parameter) -> bool {
  if (auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument)) {
    if (!typeArgument->typeId) return false;
    return typeArgument->typeId->type == parameter->type();
  }

  if (auto expressionArgument =
          ast_cast<ExpressionTemplateArgumentAST>(argument)) {
    auto expression = expressionArgument->expression;
    if (auto pack = ast_cast<PackExpansionExpressionAST>(expression))
      expression = pack->expression;
    auto idExpression = ast_cast<IdExpressionAST>(expression);
    if (!idExpression) return false;
    return denotes_template_parameter(idExpression->symbol, parameter);
  }

  return false;
}
}  // namespace

auto names_current_instantiation(TranslationUnit* unit,
                                 SimpleTemplateIdAST* templateId,
                                 ClassSymbol* classSymbol) -> bool {
  if (!templateId || !classSymbol) return false;
  classSymbol = classSymbol->resolvedDefinition();
  auto templateParameters = classSymbol->templateParameters();
  if (!templateParameters) return false;

  if (classSymbol->isSpecialization()) {
    auto declaration = ast_cast<ClassSpecifierAST>(classSymbol->declaration());
    if (!declaration) return false;
    auto pattern = ast_cast<SimpleTemplateIdAST>(declaration->unqualifiedId);
    if (!pattern) return false;
    return TemplateEquivalence{unit}.same(templateId->templateArgumentList,
                                          pattern->templateArgumentList);
  }

  const auto& parameters = templateParameters->members();

  std::size_t index = 0;
  for (auto argument : ListView{templateId->templateArgumentList}) {
    if (index >= parameters.size()) return false;
    if (!is_equivalent_to_template_parameter(argument, parameters[index]))
      return false;
    ++index;
  }

  return index == parameters.size();
}

auto names_current_instantiation(ClassSymbol* classSymbol, ScopeSymbol* scope)
    -> bool {
  if (!classSymbol) return false;

  auto primary = classSymbol->resolvedDefinition();

  for (auto enclosing = scope; enclosing; enclosing = enclosing->parent()) {
    auto enclosingClass = symbol_cast<ClassSymbol>(enclosing);
    if (!enclosingClass) continue;

    auto candidate = enclosingClass->isSpecialization()
                         ? enclosingClass->primaryTemplateSymbol()
                         : enclosingClass;
    if (!candidate) continue;
    if (candidate->resolvedDefinition() == primary) return true;
  }

  return false;
}

auto Symbol::definition() const -> Symbol* {
  return visit(GetDefinition{}, const_cast<Symbol*>(this));
}

ScopeSymbol::ScopeSymbol(SymbolKind kind, ScopeSymbol* enclosingScope)
    : Symbol(kind, enclosingScope) {}

ScopeSymbol::~ScopeSymbol() {}

auto ScopeSymbol::members() const -> const std::vector<Symbol*>& {
  return members_;
}

void ScopeSymbol::reset() {
  truncate(0);
  usingDirectives_.clear();
}

void ScopeSymbol::truncate(std::size_t count) {
  if (count >= members_.size()) return;
  for (std::size_t i = count; i < members_.size(); ++i) {
    members_[i]->link_ = nullptr;
    members_[i]->setParent(nullptr);
  }
  members_.resize(count);
  if (members_.empty())
    buckets_.clear();
  else
    rehash();
}

auto ScopeSymbol::isTransparent() const -> bool {
  if (isTemplateParameters()) return true;
  if (isFunctionParameters()) return true;
  return false;
}

auto ScopeSymbol::bucketOf(const Symbol* symbol) const -> std::size_t {
  return bucketOfHash(symbol->name() ? symbol->name()->hashValue() : 0);
}

void ScopeSymbol::addMember(Symbol* symbol) {
  if (symbol->isTemplateParameters()) {
    cxx_runtime_error("trying to add a template parameters symbol to a scope");
    return;
  }

  if (isTemplateParameters()) {
    if (!(symbol->isTypeParameter() || symbol->isTemplateTypeParameter() ||
          symbol->isNonTypeParameter() ||
          symbol->isConstraintTypeParameter())) {
      cxx_runtime_error("invalid symbol in template parameters scope");
    }
  }

  if (!symbol->parent_ || symbol->isFunctionParameters()) {
    symbol->setParent(this);
  }

  members_.push_back(symbol);
}

void ScopeSymbol::addSymbol(Symbol* symbol) {
  const auto size = members_.size();

  addMember(symbol);

  if (members_.size() == size) return;

  if (3 * members_.size() >= 2 * buckets_.size()) {
    rehash();
  } else {
    const auto h = bucketOf(symbol);
    symbol->link_ = buckets_[h];
    buckets_[h] = symbol;
  }
}

void ScopeSymbol::rebuildLookupTable() {
  if (members_.empty()) return;
  rehash();
}

void ScopeSymbol::rehash() {
  const auto newSize =
      std::max(std::size_t(8), std::bit_ceil(members_.size() * 2));

  buckets_.assign(newSize, nullptr);

  for (auto symbol : members_) {
    const auto h = bucketOf(symbol);
    symbol->link_ = buckets_[h];
    buckets_[h] = symbol;
  }
}

void ScopeSymbol::replaceSymbol(Symbol* symbol, Symbol* newSymbol) {
  if (symbol == newSymbol) return;

  auto it = std::find(members_.begin(), members_.end(), symbol);

  if (it == members_.end()) return;

  *it = newSymbol;

  newSymbol->link_ = symbol->link_;

  const auto h = bucketOf(newSymbol);

  if (buckets_[h] == symbol) {
    buckets_[h] = newSymbol;
  } else {
    for (auto p = buckets_[h]; p; p = p->link_) {
      if (p->link_ == symbol) {
        p->link_ = newSymbol;
        break;
      }
    }
  }

  symbol->link_ = nullptr;
}

void ScopeSymbol::addUsingDirective(ScopeSymbol* scope) {
  usingDirectives_.push_back(scope);
}

auto ScopeSymbol::find(const Name* name) const -> SymbolChainView {
  if (!members_.empty()) {
    const auto h = bucketOfHash(name ? name->hashValue() : 0);
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      if (symbol->name() == name) {
        return SymbolChainView{symbol};
      }
    }
  }
  return SymbolChainView{nullptr};
}

auto ScopeSymbol::find(TokenKind op) const -> SymbolChainView {
  if (!members_.empty()) {
    const auto h = bucketOfHash(OperatorId::hash(op));
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<OperatorId>(symbol->name());
      if (id && id->op() == op) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

auto ScopeSymbol::find(const std::string_view& name) const -> SymbolChainView {
  if (!members_.empty()) {
    const auto h = bucketOfHash(Identifier::hash(name));
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->name() == name) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

NamespaceSymbol::NamespaceSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

NamespaceSymbol::~NamespaceSymbol() {}

auto NamespaceSymbol::isInline() const -> bool { return isInline_; }

void NamespaceSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto NamespaceSymbol::hasInlineNamespaces() const -> bool {
  return hasInlineNamespaces_;
}

void NamespaceSymbol::setHasInlineNamespaces(bool value) {
  hasInlineNamespaces_ = value;
}

auto NamespaceSymbol::unnamedNamespace() const -> NamespaceSymbol* {
  return unnamedNamespace_;
}

void NamespaceSymbol::setUnnamedNamespace(NamespaceSymbol* unnamedNamespace) {
  unnamedNamespace_ = unnamedNamespace;
}

auto NamespaceSymbol::anonNamespaceIndex() const -> std::optional<int> {
  if (anonNamespaceIndex_ < 0) return std::nullopt;
  return anonNamespaceIndex_;
}

void NamespaceSymbol::setAnonNamespaceIndex(int index) {
  anonNamespaceIndex_ = index;
}

ConceptSymbol::ConceptSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConceptSymbol::~ConceptSymbol() {}

DeductionGuideSymbol::DeductionGuideSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

DeductionGuideSymbol::~DeductionGuideSymbol() {}

auto DeductionGuideSymbol::isExplicit() const -> bool { return isExplicit_; }

void DeductionGuideSymbol::setExplicit(bool isExplicit) {
  isExplicit_ = isExplicit;
}

BaseClassSymbol::BaseClassSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

BaseClassSymbol::~BaseClassSymbol() {}

auto BaseClassSymbol::isVirtual() const -> bool { return isVirtual_; }

void BaseClassSymbol::setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

auto BaseClassSymbol::symbol() const -> Symbol* { return symbol_; }

void BaseClassSymbol::setSymbol(Symbol* symbol) { symbol_ = symbol; }

InjectedClassNameSymbol::InjectedClassNameSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

InjectedClassNameSymbol::~InjectedClassNameSymbol() {}

UnresolvedSymbol::UnresolvedSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

UnresolvedSymbol::~UnresolvedSymbol() {}

void ClassLayout::setFieldInfo(FieldSymbol* field, const MemberInfo& info) {
  fields_[field] = info;
}

void ClassLayout::setBaseInfo(ClassSymbol* base, const MemberInfo& info) {
  bases_[base] = info;
}

void ClassLayout::setVirtualBaseInfo(ClassSymbol* base,
                                     const MemberInfo& info) {
  virtualBaseInfos_[base] = info;
}

auto ClassLayout::getVirtualBaseInfo(ClassSymbol* base) const
    -> std::optional<MemberInfo> {
  auto it = virtualBaseInfos_.find(base);
  if (it != virtualBaseInfos_.end()) return it->second;
  return std::nullopt;
}

auto ClassLayout::getFieldInfo(FieldSymbol* field) const
    -> std::optional<MemberInfo> {
  auto it = fields_.find(field);
  if (it != fields_.end()) {
    return it->second;
  }
  return std::nullopt;
}

auto ClassLayout::getBaseInfo(ClassSymbol* base) const
    -> std::optional<MemberInfo> {
  auto it = bases_.find(base);
  if (it != bases_.end()) {
    return it->second;
  }
  return std::nullopt;
}

ClassSymbol::ClassSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

ClassSymbol::~ClassSymbol() {}

auto ClassSymbol::flags() const -> std::uint32_t { return flags_; }

void ClassSymbol::setFlags(std::uint32_t flags) { flags_ = flags; }

auto ClassSymbol::isUnion() const -> bool { return isUnion_; }

void ClassSymbol::setIsUnion(bool isUnion) { isUnion_ = isUnion; }

auto ClassSymbol::isFinal() const -> bool { return isFinal_; }

void ClassSymbol::setFinal(bool isFinal) { isFinal_ = isFinal; }

auto ClassSymbol::baseClasses() const -> const std::vector<BaseClassSymbol*>& {
  return baseClasses_;
}

void ClassSymbol::addBaseClass(BaseClassSymbol* baseClass) {
  baseClasses_.push_back(baseClass);
}

void ClassSymbol::addBefriendingClass(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (std::ranges::contains(befriendingClasses_, classSymbol)) return;
  befriendingClasses_.push_back(classSymbol);
}

void ClassSymbol::addBefriendingClass(ClassSymbol* classSymbol,
                                      std::vector<TemplateArgument> arguments) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  auto found = std::ranges::find_if(
      templateFriendships_, [&](const TemplateFriendship& friendship) {
        return friendship.befriendingClass == classSymbol &&
               friendship.arguments == arguments;
      });
  if (found != templateFriendships_.end()) return;
  templateFriendships_.push_back({std::move(arguments), classSymbol});
}

auto ClassSymbol::constructors() const -> std::vector<FunctionSymbol*> {
  return constructorOverloadSet_->functions();
}

auto ClassSymbol::declaredConstructors() const
    -> const std::vector<FunctionSymbol*>& {
  return constructorOverloadSet_->declaredFunctions();
}

void ClassSymbol::addConstructor(FunctionSymbol* constructor) {
  constructorOverloadSet_->addFunction(constructor);
}

auto ClassSymbol::deductionGuides() const
    -> const std::vector<DeductionGuideSymbol*>& {
  return deductionGuides_;
}

void ClassSymbol::addDeductionGuide(DeductionGuideSymbol* guide) {
  deductionGuides_.push_back(guide);
}

auto ClassSymbol::isComplete() const -> bool { return isComplete_; }

void ClassSymbol::setComplete(bool isComplete) { isComplete_ = isComplete; }

auto ClassSymbol::isFriend() const -> bool { return isFriend_; }

void ClassSymbol::setFriend(bool isFriend) { isFriend_ = isFriend; }

auto ClassSymbol::isPolymorphic() const -> bool { return isPolymorphic_; }

void ClassSymbol::setPolymorphic(bool isPolymorphic) {
  isPolymorphic_ = isPolymorphic;
}

auto ClassSymbol::isAbstract() const -> bool { return isAbstract_; }

void ClassSymbol::setAbstract(bool isAbstract) { isAbstract_ = isAbstract; }

auto ClassSymbol::hasVirtualDestructor() const -> bool {
  return hasVirtualDestructor_;
}

void ClassSymbol::setHasVirtualDestructor(bool hasVirtualDestructor) {
  hasVirtualDestructor_ = hasVirtualDestructor;
}

auto ClassSymbol::sizeInBytes() const -> int { return sizeInBytes_; }

void ClassSymbol::setSizeInBytes(int sizeInBytes) {
  sizeInBytes_ = sizeInBytes;
}

auto ClassSymbol::alignment() const -> int { return std::max(alignment_, 1); }

void ClassSymbol::setAlignment(int alignment) { alignment_ = alignment; }

auto ClassSymbol::explicitAlignment() const -> int {
  return explicitAlignment_;
}

void ClassSymbol::setExplicitAlignment(int alignment) {
  explicitAlignment_ = alignment;
}

auto ClassSymbol::packAlignment() const -> int { return packAlignment_; }

void ClassSymbol::setPackAlignment(int alignment) {
  packAlignment_ = alignment;
}

namespace {

[[nodiscard]] auto baseClassDefinitionOf(BaseClassSymbol* baseClass)
    -> const ClassSymbol* {
  auto baseClassType = type_cast<ClassType>(baseClass->symbol()->type());
  if (!baseClassType) return nullptr;
  return baseClassType->symbol();
}

struct BaseWalkState {
  const ClassSymbol* classSymbol = nullptr;
  bool throughVirtualBase = false;

  [[nodiscard]] auto operator==(const BaseWalkState&) const -> bool = default;
};

}  // namespace

auto ClassSymbol::hasBaseClass(const Symbol* symbol) const -> bool {
  std::vector<const ClassSymbol*> visited{this};

  for (std::size_t index = 0; index < visited.size(); ++index) {
    for (auto baseClass : visited[index]->baseClasses()) {
      if (baseClass->symbol() == symbol) return true;

      auto base = baseClassDefinitionOf(baseClass);
      if (!base) continue;
      if (std::ranges::contains(visited, base)) continue;
      visited.push_back(base);
    }
  }

  return false;
}

namespace {

[[nodiscard]] auto saturatingAdd(std::uint32_t lhs, std::uint32_t rhs)
    -> std::uint32_t {
  const auto limit = std::numeric_limits<std::uint32_t>::max();
  if (rhs > limit - lhs) return limit;
  return lhs + rhs;
}

struct SubobjectEdge {
  const ClassSymbol* base = nullptr;
  bool isVirtual = false;
  std::uint64_t offset = 0;
};

struct NonVirtualPaths {
  std::uint32_t count = 0;
  std::uint64_t offset = 0;
};

struct SubobjectCount {
  std::uint32_t total = 0;
  NonVirtualPaths nonVirtual;
};

class BaseSubobjectSearch {
 public:
  BaseSubobjectSearch(const ClassSymbol* base, bool publicOnly)
      : base_(base), publicOnly_(publicOnly) {}

  [[nodiscard]] auto countIn(const ClassSymbol* derived) -> SubobjectCount {
    SubobjectCount result{.nonVirtual = nonVirtualPathsFrom(derived)};
    result.total = result.nonVirtual.count;

    for (auto virtualBase : virtualBasesOf(derived)) {
      auto paths = nonVirtualPathsFrom(virtualBase);
      result.total = saturatingAdd(result.total, paths.count);
    }

    return result;
  }

 private:
  [[nodiscard]] auto edgesOf(const ClassSymbol* derived) const
      -> std::vector<SubobjectEdge> {
    std::vector<SubobjectEdge> edges;

    auto layout = derived->layout();

    for (auto baseClass : derived->baseClasses()) {
      if (publicOnly_ &&
          baseClass->accessSpecifier() != AccessSpecifier::kPublic)
        continue;

      auto baseSymbol = symbol_cast<ClassSymbol>(baseClass->symbol());
      if (!baseSymbol) continue;

      auto baseDefinition = baseSymbol->resolvedDefinition();

      if (baseClass->isVirtual()) {
        edges.push_back({.base = baseDefinition, .isVirtual = true});
        continue;
      }

      if (!layout) {
        edges.push_back({.base = baseDefinition});
        continue;
      }

      auto baseInfo = layout->getBaseInfo(baseDefinition);
      if (!baseInfo) continue;

      edges.push_back({.base = baseDefinition, .offset = baseInfo->offset});
    }

    return edges;
  }

  [[nodiscard]] auto nonVirtualPathsFrom(const ClassSymbol* anchor)
      -> NonVirtualPaths {
    if (anchor == base_) return {.count = 1};

    if (auto cached = cachedPathsFrom(anchor)) return *cached;

    nonVirtualPaths_.emplace_back(anchor, NonVirtualPaths{});

    NonVirtualPaths result;

    for (const auto& edge : edgesOf(anchor)) {
      if (edge.isVirtual) continue;

      auto paths = nonVirtualPathsFrom(edge.base);
      if (!paths.count) continue;

      result.count = saturatingAdd(result.count, paths.count);
      result.offset = edge.offset + paths.offset;
    }

    *cachedPathsFrom(anchor) = result;
    return result;
  }

  [[nodiscard]] auto cachedPathsFrom(const ClassSymbol* anchor)
      -> NonVirtualPaths* {
    auto cached =
        std::ranges::find(nonVirtualPaths_, anchor, &CachedPaths::first);
    if (cached == nonVirtualPaths_.end()) return nullptr;
    return &cached->second;
  }

  [[nodiscard]] auto virtualBasesOf(const ClassSymbol* derived) const
      -> std::vector<const ClassSymbol*> {
    std::vector<const ClassSymbol*> virtualBases;
    std::vector<const ClassSymbol*> visited{derived};

    for (std::size_t index = 0; index < visited.size(); ++index) {
      for (const auto& edge : edgesOf(visited[index])) {
        if (edge.isVirtual && !std::ranges::contains(virtualBases, edge.base))
          virtualBases.push_back(edge.base);

        if (std::ranges::contains(visited, edge.base)) continue;
        visited.push_back(edge.base);
      }
    }

    return virtualBases;
  }

  using CachedPaths = std::pair<const ClassSymbol*, NonVirtualPaths>;

  const ClassSymbol* base_;
  bool publicOnly_;
  std::vector<CachedPaths> nonVirtualPaths_;
};

struct BaseClassRepetitionSearch {
  std::unordered_set<const ClassSymbol*> virtualBases;
  std::unordered_set<const ClassSymbol*> nonVirtualBases;
  ClassSymbol::BaseClassRepetition result;

  void collect(const ClassSymbol* derived) {
    for (auto baseClass : derived->baseClasses()) {
      auto baseSymbol = symbol_cast<ClassSymbol>(baseClass->symbol());
      if (!baseSymbol) continue;

      auto baseDefinition = baseSymbol->resolvedDefinition();

      if (baseClass->isVirtual()) {
        if (!virtualBases.insert(baseDefinition).second) {
          result.diamondShaped = true;
        } else if (nonVirtualBases.contains(baseDefinition)) {
          result.nonDiamondRepeat = true;
        }
      } else if (!nonVirtualBases.insert(baseDefinition).second) {
        result.nonDiamondRepeat = true;
      } else if (virtualBases.contains(baseDefinition)) {
        result.nonDiamondRepeat = true;
      }

      collect(baseDefinition);
    }
  }
};

}  // namespace

auto ClassSymbol::baseSubobjectInfo(ClassSymbol* base) const
    -> BaseSubobjectInfo {
  auto derived = resolvedDefinition();
  auto baseDefinition = base->resolvedDefinition();

  auto all = BaseSubobjectSearch{baseDefinition, false}.countIn(derived);
  auto visible = BaseSubobjectSearch{baseDefinition, true}.countIn(derived);

  return {
      .subobjectCount = all.total,
      .nonVirtualSubobjectCount = all.nonVirtual.count,
      .nonVirtualOffset = all.nonVirtual.offset,
      .publicSubobjectCount = visible.total,
      .publicNonVirtualSubobjectCount = visible.nonVirtual.count,
      .publicNonVirtualOffset = visible.nonVirtual.offset,
  };
}

auto ClassSymbol::baseClassOffset(ClassSymbol* base) const
    -> std::optional<std::uint64_t> {
  auto info = baseSubobjectInfo(base);
  if (!info.isUniqueSubobject()) return std::nullopt;
  if (info.nonVirtualSubobjectCount != 1) return std::nullopt;
  return info.nonVirtualOffset;
}

auto ClassSymbol::baseClassRepetition() const -> BaseClassRepetition {
  BaseClassRepetitionSearch search;
  search.collect(resolvedDefinition());
  return search.result;
}

auto ClassSymbol::hasVirtualBasePath(Symbol* symbol) const -> bool {
  std::vector<BaseWalkState> visited{{.classSymbol = this}};

  for (std::size_t index = 0; index < visited.size(); ++index) {
    const auto state = visited[index];

    for (auto baseClass : state.classSymbol->baseClasses()) {
      const auto throughVirtualBase =
          state.throughVirtualBase || baseClass->isVirtual();
      if (throughVirtualBase && baseClass->symbol() == symbol) return true;

      auto base = baseClassDefinitionOf(baseClass);
      if (!base) continue;

      const BaseWalkState next{base, throughVirtualBase};
      if (std::ranges::contains(visited, next)) continue;
      visited.push_back(next);
    }
  }

  return false;
}

auto ClassSymbol::conversionFunctions() const -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;
  for (auto func : views::members(const_cast<ClassSymbol*>(this)) |
                       views::member_functions) {
    if (name_cast<ConversionFunctionId>(func->name())) result.push_back(func);
  }
  return result;
}

auto ClassSymbol::implicitConversionFunctions() const
    -> std::vector<FunctionSymbol*> {
  auto result = conversionFunctions();
  std::erase_if(result,
                [](FunctionSymbol* func) { return func->isExplicit(); });
  return result;
}

auto ClassSymbol::visibleConversionFunctions() const
    -> std::vector<FunctionSymbol*> {
  struct Declaration {
    ClassSymbol* declaringClass = nullptr;
    FunctionSymbol* function = nullptr;
  };

  std::vector<Declaration> declarations;
  std::vector<ClassSymbol*> pending{resolvedDefinition()};
  std::vector<ClassSymbol*> seen;

  while (!pending.empty()) {
    auto currentClass = pending.back();
    pending.pop_back();
    if (!currentClass) continue;
    if (std::ranges::find(seen, currentClass) != seen.end()) continue;
    seen.push_back(currentClass);

    for (auto base : currentClass->baseClasses()) {
      if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol()))
        pending.push_back(baseClass->resolvedDefinition());
    }

    for (auto func : currentClass->conversionFunctions())
      declarations.push_back({currentClass, func});
  }

  const auto isHidden = [&declarations](const Declaration& declaration) {
    return std::ranges::any_of(declarations, [&](const Declaration& other) {
      if (other.declaringClass == declaration.declaringClass) return false;
      if (other.function->name() != declaration.function->name()) return false;
      return other.declaringClass->hasBaseClass(declaration.declaringClass);
    });
  };

  std::vector<FunctionSymbol*> result;
  for (const auto& declaration : declarations) {
    if (isHidden(declaration)) continue;
    result.push_back(declaration.function);
  }
  return result;
}

auto ClassSymbol::destructor() const -> FunctionSymbol* {
  return views::find_function(members(), [](FunctionSymbol* func) {
    return name_cast<DestructorId>(func->name()) != nullptr;
  });
}

auto ClassSymbol::isDefaultConstructor(FunctionSymbol* function) const -> bool {
  if (!function || !function->isConstructor()) return false;
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return false;
  const auto parameterCount =
      static_cast<int>(functionType->parameterTypes().size());
  return required_parameter_count(function, parameterCount) == 0;
}

auto ClassSymbol::defaultConstructor() const -> FunctionSymbol* {
  for (auto ctor : constructors()) {
    if (ctor->canonical() != ctor) continue;
    if (isDefaultConstructor(ctor)) return ctor;
  }
  return nullptr;
}

namespace {

[[nodiscard]] auto isTemplated(FunctionSymbol* function) -> bool {
  return function->templateDeclaration() || function->isSpecialization();
}

[[nodiscard]] auto namesClass(const Type* type, const ClassSymbol* classSymbol)
    -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  return classType && classType->symbol() == classSymbol;
}

template <typename Reference>
[[nodiscard]] auto isReferenceToClass(const Type* type,
                                      const ClassSymbol* classSymbol) -> bool {
  auto reference = type_cast<Reference>(type);
  return reference && namesClass(reference->elementType(), classSymbol);
}

template <typename Reference>
[[nodiscard]] auto isCopyOrMoveConstructorFor(FunctionSymbol* constructor,
                                              const ClassSymbol* classSymbol)
    -> bool {
  if (isTemplated(constructor)) return false;
  auto functionType = type_cast<FunctionType>(constructor->type());
  if (!functionType) return false;
  const auto& parameters = functionType->parameterTypes();
  if (parameters.empty()) return false;
  if (!isReferenceToClass<Reference>(parameters.front(), classSymbol))
    return false;
  const auto parameterCount = static_cast<int>(parameters.size());
  return required_parameter_count(constructor, parameterCount) <= 1;
}

[[nodiscard]] auto soleParameterType(FunctionSymbol* function) -> const Type* {
  if (isTemplated(function)) return nullptr;
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType || functionType->parameterTypes().size() != 1)
    return nullptr;
  return functionType->parameterTypes().front();
}

[[nodiscard]] auto isCopyAssignmentFor(FunctionSymbol* function,
                                       const ClassSymbol* classSymbol) -> bool {
  auto parameterType = soleParameterType(function);
  if (!parameterType) return false;
  if (namesClass(parameterType, classSymbol)) return true;
  return isReferenceToClass<LvalueReferenceType>(parameterType, classSymbol);
}

[[nodiscard]] auto isMoveAssignmentFor(FunctionSymbol* function,
                                       const ClassSymbol* classSymbol) -> bool {
  auto parameterType = soleParameterType(function);
  return isReferenceToClass<RvalueReferenceType>(parameterType, classSymbol);
}

}  // namespace

auto ClassSymbol::isCopyConstructor(FunctionSymbol* function) const -> bool {
  if (!function || !function->isConstructor()) return false;
  return isCopyOrMoveConstructorFor<LvalueReferenceType>(function, this);
}

auto ClassSymbol::isMoveConstructor(FunctionSymbol* function) const -> bool {
  if (!function || !function->isConstructor()) return false;
  return isCopyOrMoveConstructorFor<RvalueReferenceType>(function, this);
}

auto ClassSymbol::isCopyAssignmentOperator(FunctionSymbol* function) const
    -> bool {
  return function && isCopyAssignmentFor(function, this);
}

auto ClassSymbol::isMoveAssignmentOperator(FunctionSymbol* function) const
    -> bool {
  return function && isMoveAssignmentFor(function, this);
}

auto ClassSymbol::copyConstructor() const -> FunctionSymbol* {
  for (auto ctor : constructors()) {
    if (isCopyConstructor(ctor)) return ctor;
  }
  return nullptr;
}

auto ClassSymbol::moveConstructor() const -> FunctionSymbol* {
  for (auto ctor : constructors()) {
    if (isMoveConstructor(ctor)) return ctor;
  }
  return nullptr;
}

auto ClassSymbol::copyAssignmentOperator() const -> FunctionSymbol* {
  return views::find_function(find(TokenKind::T_EQUAL),
                              [this](FunctionSymbol* function) {
                                return isCopyAssignmentOperator(function);
                              });
}

auto ClassSymbol::moveAssignmentOperator() const -> FunctionSymbol* {
  return views::find_function(find(TokenKind::T_EQUAL),
                              [this](FunctionSymbol* function) {
                                return isMoveAssignmentOperator(function);
                              });
}

auto ClassSymbol::hasUserDeclaredConstructors() const -> bool {
  return hasUserDeclaredConstructors_;
}

void ClassSymbol::setHasUserDeclaredConstructors(bool value) {
  hasUserDeclaredConstructors_ = value;
}

auto ClassSymbol::hasInheritedConstructors() const -> bool {
  return !constructorOverloadSet_->usingDeclarations().empty();
}

auto ClassSymbol::hasVirtualFunctions() const -> bool {
  return views::any_function(
      members(), [](FunctionSymbol* fn) { return fn->isVirtual(); });
}

auto ClassSymbol::hasVirtualBaseSubobjects() const -> bool {
  auto classLayout = resolvedDefinition()->layout();
  return classLayout && !classLayout->virtualBases().empty();
}

auto ClassSymbol::hasVirtualBaseClasses() const -> bool {
  for (auto base : baseClasses_) {
    if (base->isVirtual()) return true;
  }
  return false;
}

auto ClassSymbol::convertingConstructors() const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;
  for (auto ctor : constructors()) {
    if (ctor->isExplicit()) continue;
    auto funcType = type_cast<FunctionType>(ctor->type());
    if (!funcType) continue;
    if (funcType->parameterTypes().empty() && !funcType->isVariadic()) continue;
    result.push_back(ctor);
  }
  return result;
}

void ClassSymbol::setLayout(std::unique_ptr<ClassLayout> layout) {
  layout_ = std::move(layout);
}

auto ClassSymbol::layout() const -> const ClassLayout* { return layout_.get(); }

void ClassSymbol::setVTableLayout(std::unique_ptr<VTableLayout> vtableLayout) {
  vtableLayout_ = std::move(vtableLayout);
}

auto VTableLayout::Table::offsetWordsBeforeAddressPoint(Symbol* subject) const
    -> std::int64_t {
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    if (offsets[index].subject != subject) continue;
    return static_cast<std::int64_t>(offsets.size() - index + 2);
  }
  return 0;
}

auto VTableLayout::Group::wordCount() const -> std::size_t {
  std::size_t count = 0;
  for (const auto& table : tables) count += table.wordCount();
  return count;
}

auto VTableLayout::Group::addressPointIndex(std::size_t table) const
    -> std::size_t {
  std::size_t index = 0;
  for (std::size_t i = 0; i < table; ++i) index += tables[i].wordCount();
  return index + tables[table].headerWordCount();
}

auto VTableLayout::Group::tableAt(std::uint64_t offset) const -> int {
  for (std::size_t index = 0; index < tables.size(); ++index) {
    if (tables[index].offset == offset) return static_cast<int>(index);
  }
  return -1;
}

auto VTableLayout::finalOverrider(FunctionSymbol* function) const
    -> FunctionSymbol* {
  for (const auto& table : main.tables) {
    for (const auto& slot : table.slots) {
      if (!slot.function) continue;
      if (slot.introducingFunction == function ||
          function->overrides(slot.introducingFunction))
        return slot.function;
    }
  }
  return nullptr;
}

auto ClassSymbol::vtableLayout() const -> const VTableLayout* {
  return vtableLayout_.get();
}

auto ClassSymbol::isClosureType() const -> bool { return isClosureType_; }

void ClassSymbol::setIsClosureType(bool isClosureType) {
  isClosureType_ = isClosureType;
}

auto ClassSymbol::hasLambdaCapture() const -> bool { return hasLambdaCapture_; }

void ClassSymbol::setHasLambdaCapture(bool hasLambdaCapture) {
  hasLambdaCapture_ = hasLambdaCapture;
}

auto ClassSymbol::capturedThisField() const -> FieldSymbol* {
  return capturedThisField_;
}

auto ClassSymbol::functionCallOperator() const -> FunctionSymbol* {
  return views::find_function(members(), [](FunctionSymbol* function) {
    auto name = name_cast<OperatorId>(function->name());
    return name && name->op() == TokenKind::T_LPAREN;
  });
}

void ClassSymbol::setCapturedThisField(FieldSymbol* capturedThisField) {
  capturedThisField_ = capturedThisField;
}

auto ClassSymbol::closureDiscriminator() const -> int {
  return closureDiscriminator_;
}

void ClassSymbol::setClosureDiscriminator(int closureDiscriminator) {
  closureDiscriminator_ = closureDiscriminator;
}

auto ClassSymbol::instantiationTemplate() const -> ClassSymbol* {
  if (auto pattern = symbol_cast<ClassSymbol>(instantiationPattern()))
    return pattern->resolvedDefinition();
  if (auto primary = primaryTemplateSymbol())
    return primary->resolvedDefinition();
  return nullptr;
}

EnumSymbol::EnumSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

EnumSymbol::~EnumSymbol() {}

auto EnumSymbol::hasFixedUnderlyingType() const -> bool {
  return hasFixedUnderlyingType_;
}

void EnumSymbol::setHasFixedUnderlyingType(bool hasFixedUnderlyingType) {
  hasFixedUnderlyingType_ = hasFixedUnderlyingType;
}

auto EnumSymbol::isDefined() const -> bool { return isDefined_; }

void EnumSymbol::setDefined(bool isDefined) { isDefined_ = isDefined; }

auto EnumSymbol::underlyingType() const -> const Type* {
  return underlyingType_;
}

void EnumSymbol::setUnderlyingType(const Type* underlyingType) {
  underlyingType_ = underlyingType;
}

ScopedEnumSymbol::ScopedEnumSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

ScopedEnumSymbol::~ScopedEnumSymbol() {}

auto ScopedEnumSymbol::underlyingType() const -> const Type* {
  return underlyingType_;
}

void ScopedEnumSymbol::setUnderlyingType(const Type* underlyingType) {
  underlyingType_ = underlyingType;
}

auto ScopedEnumSymbol::isDefined() const -> bool { return isDefined_; }

void ScopedEnumSymbol::setDefined(bool isDefined) { isDefined_ = isDefined; }

FunctionSymbol::FunctionSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

FunctionSymbol::~FunctionSymbol() {}

auto FunctionSymbol::isDefined() const -> bool { return isDefined_; }

void FunctionSymbol::setDefined(bool isDefined) { isDefined_ = isDefined; }

auto FunctionSymbol::isStatic() const -> bool { return isStatic_; }

void FunctionSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto FunctionSymbol::isExtern() const -> bool { return isExtern_; }

void FunctionSymbol::setExtern(bool isExtern) { isExtern_ = isExtern; }

auto FunctionSymbol::isFriend() const -> bool { return isFriend_; }

void FunctionSymbol::setFriend(bool isFriend) { isFriend_ = isFriend; }

auto FunctionSymbol::isImplicitObjectMemberFunction() const -> bool {
  if (hasExplicitObjectParameter()) return false;
  return !isStatic() && !isFriend() && symbol_cast<ClassSymbol>(parent());
}

auto FunctionSymbol::isNonStaticMemberFunction() const -> bool {
  return isImplicitObjectMemberFunction() || hasExplicitObjectParameter();
}

auto FunctionSymbol::hasImplicitObjectParameter() const -> bool {
  if (isConstructor()) return false;
  if (isStatic()) return symbol_cast<ClassSymbol>(parent()) != nullptr;
  return isImplicitObjectMemberFunction();
}

auto FunctionSymbol::hasExplicitObjectParameter() const -> bool {
  return hasExplicitObjectParameter_;
}

void FunctionSymbol::setExplicitObjectParameter(
    bool hasExplicitObjectParameter) {
  hasExplicitObjectParameter_ = hasExplicitObjectParameter;
}

auto FunctionSymbol::explicitObjectParameter() const -> ParameterSymbol* {
  if (!hasExplicitObjectParameter()) return nullptr;
  auto parameters = functionParameters();
  if (!parameters) return nullptr;
  for (auto member : parameters->members()) {
    if (auto parameter = symbol_cast<ParameterSymbol>(member)) return parameter;
  }
  return nullptr;
}

auto FunctionSymbol::parameters() const -> std::vector<ParameterSymbol*> {
  std::vector<ParameterSymbol*> result;
  auto scope = functionParameters();
  if (!scope) return result;
  for (auto parameter : views::members(scope) | views::parameters)
    result.push_back(parameter);
  return result;
}

auto FunctionSymbol::isConstexpr() const -> bool {
  if (isConsteval_) return true;
  return isConstexpr_;
}

void FunctionSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto FunctionSymbol::isConsteval() const -> bool { return isConsteval_; }

void FunctionSymbol::setConsteval(bool isConsteval) {
  isConsteval_ = isConsteval;
}

auto FunctionSymbol::isInline() const -> bool { return isInline_; }

void FunctionSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto FunctionSymbol::isVirtual() const -> bool { return isVirtual_; }

void FunctionSymbol::setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

void FunctionSymbol::addOverriddenFunction(FunctionSymbol* function) {
  if (!function) return;
  if (std::ranges::contains(overriddenFunctions_, function)) return;
  overriddenFunctions_.push_back(function);
}

auto FunctionSymbol::overrides(FunctionSymbol* function) const -> bool {
  if (!function) return false;

  std::vector<const FunctionSymbol*> pending{this};
  std::unordered_set<const FunctionSymbol*> visited;

  while (!pending.empty()) {
    auto current = pending.back();
    pending.pop_back();
    if (!visited.insert(current).second) continue;

    for (auto overridden : current->overriddenFunctions_) {
      if (overridden == function) return true;
      pending.push_back(overridden);
    }
  }

  return false;
}

void FunctionSymbol::addBefriendingClass(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (std::ranges::contains(befriendingClasses_, classSymbol)) return;
  befriendingClasses_.push_back(classSymbol);
}

void FunctionSymbol::addBefriendingClass(
    ClassSymbol* classSymbol, std::vector<TemplateArgument> arguments) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  auto found = std::ranges::find_if(
      templateFriendships_, [&](const TemplateFriendship& friendship) {
        return friendship.befriendingClass == classSymbol &&
               friendship.arguments == arguments;
      });
  if (found != templateFriendships_.end()) return;
  templateFriendships_.push_back({std::move(arguments), classSymbol});
}

auto FunctionSymbol::isExplicit() const -> bool { return isExplicit_; }

void FunctionSymbol::setExplicit(bool isExplicit) { isExplicit_ = isExplicit; }

auto FunctionSymbol::isDeleted() const -> bool { return isDeleted_; }

void FunctionSymbol::setDeleted(bool isDeleted) { isDeleted_ = isDeleted; }

auto FunctionSymbol::isDefaulted() const -> bool { return isDefaulted_; }

void FunctionSymbol::setDefaulted(bool isDefaulted) {
  isDefaulted_ = isDefaulted;
}

auto FunctionSymbol::isPure() const -> bool { return isPure_; }

void FunctionSymbol::setPure(bool isPure) { isPure_ = isPure; }

auto FunctionSymbol::isOverride() const -> bool { return isOverride_; }

void FunctionSymbol::setOverride(bool isOverride) { isOverride_ = isOverride; }

auto FunctionSymbol::isFinal() const -> bool { return isFinal_; }

void FunctionSymbol::setFinal(bool isFinal) { isFinal_ = isFinal; }

auto FunctionSymbol::hasNoPrototype() const -> bool { return hasNoPrototype_; }

void FunctionSymbol::setNoPrototype(bool hasNoPrototype) {
  hasNoPrototype_ = hasNoPrototype;
}

auto FunctionSymbol::hasExceptionSpecifier() const -> bool {
  return hasExceptionSpecifier_;
}

void FunctionSymbol::setExceptionSpecifier(bool hasExceptionSpecifier) {
  hasExceptionSpecifier_ = hasExceptionSpecifier;
}

auto FunctionSymbol::trailingRequiresClause() const -> RequiresClauseAST* {
  return trailingRequiresClause_;
}

void FunctionSymbol::setTrailingRequiresClause(
    RequiresClauseAST* requiresClause) {
  trailingRequiresClause_ = requiresClause;
}

auto FunctionSymbol::isStructor() const -> bool {
  return isConstructor() || name_cast<DestructorId>(name());
}

auto FunctionSymbol::hasBaseObjectVariant() const -> bool {
  if (!isStructor()) return false;
  if (externalName() || hasCLinkage()) return false;
  if (completeObjectVariant()) return false;
  if (isStructorVariant()) return false;

  auto classSymbol = symbol_cast<ClassSymbol>(parent());
  if (classSymbol && classSymbol->hasVirtualBaseSubobjects()) return false;

  return true;
}

auto FunctionSymbol::isConstructor() const -> bool {
  auto p = symbol_cast<ClassSymbol>(parent());
  if (!p) return false;

  auto functionType = type_cast<FunctionType>(type());
  if (!functionType) return false;
  if (!functionType->returnType()) return false;
  if (functionType->returnType()->kind() != TypeKind::kVoid) {
    return false;
  }

  auto id = name_cast<Identifier>(name());
  if (!id) {
    return std::ranges::contains(p->constructors(),
                                 const_cast<FunctionSymbol*>(this));
  }

  if (p->name() == id) return true;

  if (auto pid = name_cast<Identifier>(p->name())) {
    if (pid->name() == id->name()) return true;
  }

  return false;
}

auto FunctionSymbol::isDestructor() const -> bool {
  if (name_cast<DestructorId>(name())) return true;
  return false;
}

auto FunctionSymbol::languageLinkage() const -> LanguageKind {
  return hasCLinkage_ ? LanguageKind::kC : LanguageKind::kCXX;
}

void FunctionSymbol::setLanguageLinkage(LanguageKind linkage) {
  hasCLinkage_ = (linkage == LanguageKind::kC);
}

auto FunctionSymbol::hasCLinkage() const -> bool { return hasCLinkage_; }

auto FunctionSymbol::isDefinitionRequired() const -> bool {
  return isDefinitionRequired_;
}

void FunctionSymbol::setDefinitionRequired(bool isDefinitionRequired) {
  isDefinitionRequired_ = isDefinitionRequired;
}

auto FunctionSymbol::isNoReturn() const -> bool { return isNoReturn_; }

void FunctionSymbol::setNoReturn(bool isNoReturn) { isNoReturn_ = isNoReturn; }

auto FunctionSymbol::builtinKind() const -> BuiltinFunctionKind {
  return builtinKind_;
}

void FunctionSymbol::setBuiltinKind(BuiltinFunctionKind builtinKind) {
  builtinKind_ = builtinKind;
}

auto FunctionSymbol::externalName() const -> const Identifier* {
  return externalName_;
}

void FunctionSymbol::setExternalName(const Identifier* externalName) {
  externalName_ = externalName;
}

auto FunctionSymbol::aliasName() const -> const Identifier* {
  return aliasName_;
}

void FunctionSymbol::setAliasName(const Identifier* aliasName) {
  aliasName_ = aliasName;
}

auto FunctionSymbol::importModule() const -> const Identifier* {
  return importModule_;
}

void FunctionSymbol::setImportModule(const Identifier* importModule) {
  importModule_ = importModule;
}

auto FunctionSymbol::importName() const -> const Identifier* {
  return importName_;
}

void FunctionSymbol::setImportName(const Identifier* importName) {
  importName_ = importName;
}

auto FunctionSymbol::exportName() const -> const Identifier* {
  return exportName_;
}

void FunctionSymbol::setExportName(const Identifier* exportName) {
  exportName_ = exportName;
}

auto FunctionSymbol::hasHiddenVisibility() const -> bool {
  return hasHiddenVisibility_;
}

void FunctionSymbol::setHiddenVisibility(bool hasHiddenVisibility) {
  hasHiddenVisibility_ = hasHiddenVisibility;
}

auto FunctionSymbol::hasPendingBody() const -> bool {
  return pendingBody_ != nullptr;
}

auto FunctionSymbol::hasUninstantiatedBody() const -> bool {
  if (hasPendingBody()) return true;
  return isSpecialization() && !declaration() && !isDefined();
}

auto FunctionSymbol::pendingBody() const -> PendingInstantiation* {
  return pendingBody_.get();
}

void FunctionSymbol::setPendingBody(
    std::unique_ptr<PendingInstantiation> pending) {
  pendingBody_ = std::move(pending);
}

void FunctionSymbol::clearPendingBody() { pendingBody_.reset(); }

auto FunctionSymbol::pendingExceptionSpecification() const
    -> PendingInstantiation* {
  return pendingExceptionSpecification_.get();
}

void FunctionSymbol::setPendingExceptionSpecification(
    std::unique_ptr<PendingInstantiation> pending) {
  pendingExceptionSpecification_ = std::move(pending);
}

auto FunctionSymbol::functionParameters() const -> FunctionParametersSymbol* {
  for (auto member : members()) {
    if (auto params = symbol_cast<FunctionParametersSymbol>(member))
      return params;
  }
  return nullptr;
}

OverloadSetSymbol::OverloadSetSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

OverloadSetSymbol::~OverloadSetSymbol() {}

auto OverloadSetSymbol::declaredFunctions() const
    -> const std::vector<FunctionSymbol*>& {
  return declaredFunctions_;
}

void OverloadSetSymbol::setFunctions(std::vector<FunctionSymbol*> functions) {
  declaredFunctions_ = std::move(functions);
}

void OverloadSetSymbol::addFunction(FunctionSymbol* function) {
  if (!function) return;

  auto canonical = function->canonical();

  for (auto existing : declaredFunctions_) {
    if (!existing) continue;
    if (existing->canonical() == canonical) return;
  }

  declaredFunctions_.push_back(function);
}

void OverloadSetSymbol::truncateFunctions(std::size_t count) {
  if (count >= declaredFunctions_.size()) return;
  declaredFunctions_.resize(count);
}

auto OverloadSetSymbol::usingDeclarations() const
    -> const std::vector<UsingDeclarationSymbol*>& {
  return usingDeclarations_;
}

void OverloadSetSymbol::addUsingDeclaration(
    UsingDeclarationSymbol* usingDeclaration) {
  if (!usingDeclaration) return;
  if (std::ranges::contains(usingDeclarations_, usingDeclaration)) return;
  usingDeclarations_.push_back(usingDeclaration);
}

auto OverloadSetSymbol::hasUnresolvedUsingDeclaration() const -> bool {
  return std::ranges::any_of(usingDeclarations_,
                             &UsingDeclarationSymbol::isUnresolved);
}

auto OverloadSetSymbol::functions() const -> std::vector<FunctionSymbol*> {
  return functionsWithin(declaredFunctions_.size(), usingDeclarations_.size());
}

auto OverloadSetSymbol::functionsWithin(std::size_t functionCount,
                                        std::size_t usingDeclarationCount) const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result(
      declaredFunctions_.begin(),
      declaredFunctions_.begin() + static_cast<std::ptrdiff_t>(functionCount));

  for (std::size_t i = 0; i < usingDeclarationCount; ++i) {
    for (auto introduced : usingDeclarations_[i]->introducedFunctions()) {
      auto canonical = introduced->canonical();

      const auto isHidden =
          std::ranges::any_of(result, [&](FunctionSymbol* declared) {
            return declared->canonical() == canonical ||
                   hasEquivalentParameterTypeList(declared, introduced);
          });

      if (!isHidden) result.push_back(introduced);
    }
  }

  return result;
}

LambdaSymbol::LambdaSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

LambdaSymbol::~LambdaSymbol() {}

auto LambdaSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void LambdaSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto LambdaSymbol::isConsteval() const -> bool { return isConsteval_; }

void LambdaSymbol::setConsteval(bool isConsteval) {
  isConsteval_ = isConsteval;
}

auto LambdaSymbol::isMutable() const -> bool { return isMutable_; }

void LambdaSymbol::setMutable(bool isMutable) { isMutable_ = isMutable; }

auto LambdaSymbol::isStatic() const -> bool { return isStatic_; }

void LambdaSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto LambdaSymbol::isTemplate() const -> bool { return isTemplate_; }

void LambdaSymbol::setTemplate(bool isTemplate) { isTemplate_ = isTemplate; }

auto LambdaSymbol::isInTemplate() const -> bool { return isInTemplate_; }

void LambdaSymbol::setInTemplate(bool isInTemplate) {
  isInTemplate_ = isInTemplate;
}

FunctionParametersSymbol::FunctionParametersSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

FunctionParametersSymbol::~FunctionParametersSymbol() {}

TemplateParametersSymbol::TemplateParametersSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

TemplateParametersSymbol::~TemplateParametersSymbol() {}

auto TemplateParametersSymbol::isExplicitTemplateSpecialization() const
    -> bool {
  return isExplicitTemplateSpecialization_;
}

void TemplateParametersSymbol::setExplicitTemplateSpecialization(
    bool isExplicit) {
  isExplicitTemplateSpecialization_ = isExplicit;
}

BlockSymbol::BlockSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

BlockSymbol::~BlockSymbol() {}

TypeAliasSymbol::TypeAliasSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeAliasSymbol::~TypeAliasSymbol() {}

VariableSymbol::VariableSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

VariableSymbol::~VariableSymbol() {}

auto VariableSymbol::isStatic() const -> bool { return isStatic_; }

void VariableSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto VariableSymbol::isThreadLocal() const -> bool { return isThreadLocal_; }

void VariableSymbol::setThreadLocal(bool isThreadLocal) {
  isThreadLocal_ = isThreadLocal;
}

auto VariableSymbol::isExtern() const -> bool { return isExtern_; }

void VariableSymbol::setExtern(bool isExtern) { isExtern_ = isExtern; }

auto VariableSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void VariableSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto VariableSymbol::isConstinit() const -> bool { return isConstinit_; }

void VariableSymbol::setConstinit(bool isConstinit) {
  isConstinit_ = isConstinit;
}

auto VariableSymbol::isInline() const -> bool { return isInline_; }

void VariableSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto VariableSymbol::isFunctionLocalPredefined() const -> bool {
  return isFunctionLocalPredefined_;
}

void VariableSymbol::setFunctionLocalPredefined(
    bool isFunctionLocalPredefined) {
  isFunctionLocalPredefined_ = isFunctionLocalPredefined;
}

auto VariableSymbol::initializer() const -> ExpressionAST* {
  return initializer_;
}

void VariableSymbol::setInitializer(ExpressionAST* initializer) {
  initializer_ = initializer;
}

auto VariableSymbol::constructor() const -> FunctionSymbol* {
  return constructor_;
}

void VariableSymbol::setConstructor(FunctionSymbol* constructor) {
  constructor_ = constructor;
}

auto VariableSymbol::constValue() const -> const std::optional<ConstValue>& {
  return constValue_;
}

void VariableSymbol::setConstValue(std::optional<ConstValue> value) {
  constValue_ = std::move(value);
}

auto VariableSymbol::explicitAlignment() const -> int {
  return explicitAlignment_;
}

void VariableSymbol::setExplicitAlignment(int alignment) {
  explicitAlignment_ = alignment;
}

auto FieldSymbol::constValue() const -> const std::optional<ConstValue>& {
  return constValue_;
}

void FieldSymbol::setConstValue(std::optional<ConstValue> value) {
  constValue_ = std::move(value);
}

FieldSymbol::FieldSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

auto FieldSymbol::pendingInitializer() const -> PendingInstantiation* {
  return pendingInitializer_.get();
}

void FieldSymbol::setPendingInitializer(
    std::unique_ptr<PendingInstantiation> pending) {
  pendingInitializer_ = std::move(pending);
}

void FieldSymbol::clearPendingInitializer() { pendingInitializer_.reset(); }

FieldSymbol::~FieldSymbol() {}

auto FieldSymbol::isBitField() const -> bool { return isBitField_; }

void FieldSymbol::setBitField(bool isBitField) { isBitField_ = isBitField; }

auto FieldSymbol::bitFieldOffset() const -> int { return bitFieldOffset_; }

void FieldSymbol::setBitFieldOffset(int bitFieldOffset) {
  bitFieldOffset_ = bitFieldOffset;
}

auto FieldSymbol::bitFieldWidth() const -> const std::optional<ConstValue>& {
  return bitFieldWidth_;
}

void FieldSymbol::setBitFieldWidth(std::optional<ConstValue> bitFieldWidth) {
  bitFieldWidth_ = std::move(bitFieldWidth);
}

auto FieldSymbol::isStatic() const -> bool { return isStatic_; }

auto FieldSymbol::isExtern() const -> bool {
  if (!isStatic_) return false;
  if (isInline_) return false;
  return !isConstexpr_;
}

void FieldSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto FieldSymbol::isThreadLocal() const -> bool { return isThreadLocal_; }

void FieldSymbol::setThreadLocal(bool isThreadLocal) {
  isThreadLocal_ = isThreadLocal;
}

auto FieldSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void FieldSymbol::setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

auto FieldSymbol::isConstinit() const -> bool { return isConstinit_; }

void FieldSymbol::setConstinit(bool isConstinit) { isConstinit_ = isConstinit; }

auto FieldSymbol::isInline() const -> bool { return isInline_; }

void FieldSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto FieldSymbol::isMutable() const -> bool { return isMutable_; }

void FieldSymbol::setMutable(bool isMutable) { isMutable_ = isMutable; }

auto FieldSymbol::isNoUniqueAddress() const -> bool {
  return isNoUniqueAddress_;
}

void FieldSymbol::setNoUniqueAddress(bool isNoUniqueAddress) {
  isNoUniqueAddress_ = isNoUniqueAddress;
}

auto FieldSymbol::offsetInClass() const -> std::optional<std::uint64_t> {
  if (isStatic()) return std::nullopt;
  auto classSymbol = symbol_cast<ClassSymbol>(parent());
  if (!classSymbol) return std::nullopt;
  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;
  auto info = layout->getFieldInfo(const_cast<FieldSymbol*>(this));
  if (!info) return std::nullopt;
  return info->offset;
}

auto FieldSymbol::localOffset() const -> int { return localOffset_; }

void FieldSymbol::setLocalOffset(int offset) { localOffset_ = offset; }

auto FieldSymbol::alignment() const -> int {
  return std::max(alignment_, explicitAlignment_);
}

void FieldSymbol::setAlignment(int alignment) { alignment_ = alignment; }

auto FieldSymbol::explicitAlignment() const -> int {
  return explicitAlignment_;
}

void FieldSymbol::setExplicitAlignment(int alignment) {
  explicitAlignment_ = alignment;
}

auto FieldSymbol::isPacked() const -> bool {
  if (findAttribute(attributes(), "packed")) return true;
  auto owner = symbol_cast<ClassSymbol>(parent());
  return owner && findAttribute(owner->attributes(), "packed");
}

auto FieldSymbol::effectiveAlignment() const -> int {
  auto effective = alignment();
  if (isPacked()) effective = std::max(explicitAlignment(), 1);
  auto owner = symbol_cast<ClassSymbol>(parent());
  if (!owner || owner->packAlignment() <= 0) return effective;
  return std::min(effective, owner->packAlignment());
}

auto FieldSymbol::initializer() const -> ExpressionAST* { return initializer_; }

void FieldSymbol::setInitializer(ExpressionAST* initializer) {
  initializer_ = initializer;
}

auto FieldSymbol::constructor() const -> FunctionSymbol* {
  return constructor_;
}

void FieldSymbol::setConstructor(FunctionSymbol* constructor) {
  constructor_ = constructor;
}

ParameterSymbol::ParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ParameterSymbol::~ParameterSymbol() {}

auto ParameterSymbol::defaultArgument() const -> ExpressionAST* {
  return defaultArgument_;
}

auto ParameterSymbol::isExplicitObject() const -> bool {
  return isExplicitObject_;
}

void ParameterSymbol::setExplicitObject(bool isExplicitObject) {
  isExplicitObject_ = isExplicitObject;
}

auto ParameterSymbol::isParameterPack() const -> bool {
  return isParameterPack_;
}

void ParameterSymbol::setParameterPack(bool isParameterPack) {
  isParameterPack_ = isParameterPack;
}

void ParameterSymbol::setDefaultArgument(ExpressionAST* expr) {
  defaultArgument_ = expr;
}

auto ParameterSymbol::hasDefaultArgument() const -> bool {
  if (defaultArgument_ || pendingDefaultArgument_) return true;
  return defaultArgumentSource_ && defaultArgumentSource_->hasDefaultArgument();
}

auto ParameterSymbol::pendingDefaultArgument() const -> PendingInstantiation* {
  return pendingDefaultArgument_.get();
}

void ParameterSymbol::setPendingDefaultArgument(
    std::unique_ptr<PendingInstantiation> pending) {
  pendingDefaultArgument_ = std::move(pending);
}

void ParameterSymbol::clearPendingDefaultArgument() {
  pendingDefaultArgument_.reset();
}

auto ParameterSymbol::defaultArgumentSource() const -> ParameterSymbol* {
  return defaultArgumentSource_;
}

void ParameterSymbol::setDefaultArgumentSource(ParameterSymbol* source) {
  defaultArgumentSource_ = source;
}

ParameterPackSymbol::ParameterPackSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ParameterPackSymbol::~ParameterPackSymbol() {}

auto ParameterPackSymbol::elements() const -> const std::vector<Symbol*>& {
  return elements_;
}

void ParameterPackSymbol::addElement(Symbol* element) {
  if (auto pack = symbol_cast<ParameterPackSymbol>(element)) {
    for (auto nested : pack->elements()) addElement(nested);
    return;
  }
  elements_.push_back(element);
}

TypeParameterSymbol::TypeParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeParameterSymbol::~TypeParameterSymbol() {}

NonTypeParameterSymbol::NonTypeParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

NonTypeParameterSymbol::~NonTypeParameterSymbol() {}

auto NonTypeParameterSymbol::index() const -> int { return index_; }

void NonTypeParameterSymbol::setIndex(int index) { index_ = index; }

auto NonTypeParameterSymbol::depth() const -> int { return depth_; }

void NonTypeParameterSymbol::setDepth(int depth) { depth_ = depth; }

auto NonTypeParameterSymbol::objectType() const -> const Type* {
  return objectType_;
}

void NonTypeParameterSymbol::setObjectType(const Type* objectType) {
  objectType_ = objectType;
}

auto NonTypeParameterSymbol::isParameterPack() const -> bool {
  return isParameterPack_;
}

void NonTypeParameterSymbol::setParameterPack(bool isParameterPack) {
  isParameterPack_ = isParameterPack;
}

TemplateTypeParameterSymbol::TemplateTypeParameterSymbol(
    ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TemplateTypeParameterSymbol::~TemplateTypeParameterSymbol() {}

ConstraintTypeParameterSymbol::ConstraintTypeParameterSymbol(
    ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConstraintTypeParameterSymbol::~ConstraintTypeParameterSymbol() {}

auto ConstraintTypeParameterSymbol::index() const -> int { return index_; }

void ConstraintTypeParameterSymbol::setIndex(int index) { index_ = index; }

auto ConstraintTypeParameterSymbol::depth() const -> int { return depth_; }

void ConstraintTypeParameterSymbol::setDepth(int depth) { depth_ = depth; }

auto ConstraintTypeParameterSymbol::isParameterPack() const -> bool {
  return isParameterPack_;
}

void ConstraintTypeParameterSymbol::setParameterPack(bool isParameterPack) {
  isParameterPack_ = isParameterPack;
}

EnumeratorSymbol::EnumeratorSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

EnumeratorSymbol::~EnumeratorSymbol() {}

auto EnumeratorSymbol::value() const -> const std::optional<ConstValue>& {
  return value_;
}

void EnumeratorSymbol::setValue(const std::optional<ConstValue>& value) {
  value_ = value;
}

NamespaceAliasSymbol::NamespaceAliasSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

NamespaceAliasSymbol::~NamespaceAliasSymbol() {}

auto NamespaceAliasSymbol::namespaceSymbol() const -> NamespaceSymbol* {
  return namespaceSymbol_;
}

void NamespaceAliasSymbol::setNamespaceSymbol(
    NamespaceSymbol* namespaceSymbol) {
  namespaceSymbol_ = namespaceSymbol;
}

UsingDeclarationSymbol::UsingDeclarationSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

UsingDeclarationSymbol::~UsingDeclarationSymbol() {}

auto UsingDeclarationSymbol::target() const -> Symbol* { return target_; }

void UsingDeclarationSymbol::setTarget(Symbol* symbol) {
  target_ = symbol;
  targetFunctionCount_ = 0;
  targetUsingDeclarationCount_ = 0;
  auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol);
  if (!overloadSet) return;
  targetFunctionCount_ =
      static_cast<int>(overloadSet->declaredFunctions().size());
  targetUsingDeclarationCount_ =
      static_cast<int>(overloadSet->usingDeclarations().size());
}

auto UsingDeclarationSymbol::targetFunctionCount() const -> int {
  return targetFunctionCount_;
}

void UsingDeclarationSymbol::setTargetFunctionCount(int count) {
  targetFunctionCount_ = count;
}

auto UsingDeclarationSymbol::targetUsingDeclarationCount() const -> int {
  return targetUsingDeclarationCount_;
}

void UsingDeclarationSymbol::setTargetUsingDeclarationCount(int count) {
  targetUsingDeclarationCount_ = count;
}

auto UsingDeclarationSymbol::introducedFunctions() const
    -> std::vector<FunctionSymbol*> {
  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(target_))
    return overloadSet->functionsWithin(targetFunctionCount_,
                                        targetUsingDeclarationCount_);
  if (auto function = symbol_cast<FunctionSymbol>(target_)) return {function};
  return {};
}

auto UsingDeclarationSymbol::isUnresolved() const -> bool {
  return target_ == nullptr;
}

auto UsingDeclarationSymbol::declarator() const -> UsingDeclaratorAST* {
  return declarator_;
}

void UsingDeclarationSymbol::setDeclarator(UsingDeclaratorAST* declarator) {
  declarator_ = declarator;
}

auto is_class_or_enum_declaration(Symbol* symbol) -> bool {
  if (!symbol) return false;
  switch (symbol->kind()) {
    case SymbolKind::kClass:
    case SymbolKind::kInjectedClassName:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum:
      return true;
    case SymbolKind::kUsingDeclaration: {
      auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(symbol);
      return is_class_or_enum_declaration(usingDeclaration->target());
    }
    default:
      return false;
  }
}

bool is_type(Symbol* symbol) {
  if (!symbol) return false;
  switch (symbol->kind()) {
    case SymbolKind::kTypeParameter:
    case SymbolKind::kConstraintTypeParameter:
    case SymbolKind::kTemplateTypeParameter:
    case SymbolKind::kTypeAlias:
    case SymbolKind::kClass:
    case SymbolKind::kInjectedClassName:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum:
      return true;
    case SymbolKind::kUsingDeclaration: {
      auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(symbol);
      return is_type(usingDeclaration->target());
    }
    default:
      return false;
  }
}

auto isDeclaredConstant(Symbol* symbol) -> bool {
  auto isConstQualified = [](const Type* type) {
    auto qualType = type_cast<QualType>(type);
    return qualType && qualType->isConst();
  };

  if (auto var = symbol_cast<VariableSymbol>(symbol))
    return var->isConstexpr() || isConstQualified(var->type());

  if (auto field = symbol_cast<FieldSymbol>(symbol))
    return field->isConstexpr() || isConstQualified(field->type());

  return false;
}

auto isUsableInConstantExpressions(Symbol* symbol) -> bool {
  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var || !var->constValue().has_value()) return false;
  return isDeclaredConstant(var);
}
}  // namespace cxx
