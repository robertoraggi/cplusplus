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
#include <cxx/binder.h>
#include <cxx/builtin_signature.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <optional>
#include <span>

namespace cxx {
namespace {
struct AssociatedNamespaceCollector {
  TranslationUnit* unit = nullptr;
  std::vector<NamespaceSymbol*>& namespaces;
  std::vector<ClassSymbol*>& classes;
  std::vector<const Type*>& visited;

  void collect(const Type* type) {
    if (!type) return;
    if (std::ranges::contains(visited, type)) return;
    visited.push_back(type);
    visit(*this, type);
  }

  void addNamespace(NamespaceSymbol* ns) {
    if (!ns) return;
    while (ns->isInline()) {
      auto parent = ns->enclosingNamespace();
      if (!parent) break;
      ns = parent;
    }
    for (auto current : inlineNamespaceSet(ns)) {
      if (std::ranges::contains(namespaces, current)) continue;
      namespaces.push_back(current);
    }
  }

  void addEnclosingClass(const Symbol* symbol) {
    if (auto cls = symbol_cast<ClassSymbol>(symbol->parent())) {
      cls = cls->resolvedDefinition();
      addClass(cls);
      addNamespace(cls->enclosingNamespace());
    }
  }

  void addClass(ClassSymbol* cls) {
    if (cls && !std::ranges::contains(classes, cls)) classes.push_back(cls);
  }

  void collect(const Symbol* symbol) {
    if (!symbol) return;
    collect(symbol->type());
    addEnclosingClass(symbol);
    addNamespace(symbol->enclosingNamespace());
  }

  void collect(const ConstValue& value) {
    if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
      if (*object) collect((*object)->type());
    }
  }

  void collect(const TemplateArgument& arg) {
    if (auto argType = std::get_if<const Type*>(&arg)) {
      collect(*argType);
      return;
    }

    if (auto argSymbol = std::get_if<Symbol*>(&arg)) {
      collect(*argSymbol);
      return;
    }

    if (auto argValue = std::get_if<ConstValue>(&arg)) {
      collect(*argValue);
      return;
    }

    if (auto argExpr = std::get_if<ExpressionAST*>(&arg)) {
      if (*argExpr) collect((*argExpr)->type);
      return;
    }
  }

  void operator()(const QualType* type) { collect(type->elementType()); }

  void operator()(const PointerType* type) { collect(type->elementType()); }
  void operator()(const LvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const ClassType* type) {
    auto classSymbol = type->symbol();
    if (!classSymbol) return;
    classSymbol = classSymbol->resolvedDefinition();
    addClass(classSymbol);
    addEnclosingClass(classSymbol);

    addNamespace(classSymbol->enclosingNamespace());

    for (const auto& base : classSymbol->baseClasses()) {
      if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol())) {
        if (auto baseType = type_cast<ClassType>(baseClass->type())) {
          collect(baseType);
        }
      }
    }

    for (const auto& arg : classSymbol->templateArguments()) collect(arg);
  }

  void operator()(const EnumType* type) {
    if (auto sym = type->symbol()) {
      addEnclosingClass(sym);
      addNamespace(sym->enclosingNamespace());
    }
  }

  void operator()(const ScopedEnumType* type) {
    if (auto sym = type->symbol()) {
      addEnclosingClass(sym);
      addNamespace(sym->enclosingNamespace());
    }
  }

  void operator()(const FunctionType* type) {
    collect(type->returnType());
    for (auto paramType : type->parameterTypes()) collect(paramType);
  }

  void operator()(const MemberObjectPointerType* type) {
    collect(type->classType());
    collect(type->elementType());
  }

  void operator()(const MemberFunctionPointerType* type) {
    collect(type->classType());
    collect(type->functionType());
  }

  void operator()(const Type*) {}
};
}  // namespace

namespace {
template <typename Predicate>
void collectQualifiedNamespaceDeclarations(NamespaceSymbol* scope,
                                           const Name* name, Predicate accept,
                                           std::vector<Symbol*>& found,
                                           std::vector<ScopeSymbol*>& visited) {
  if (!scope || std::ranges::contains(visited, scope)) return;
  auto inlineSet = inlineNamespaceSet(scope);

  const auto start = found.size();
  for (auto ns : inlineSet) {
    if (std::ranges::contains(visited, ns)) continue;
    std::vector<ScopeSymbol*> directVisited;
    if (auto symbol =
            detail::searchScope(ns, name, directVisited, accept, false))
      found.push_back(symbol);
  }
  for (auto ns : inlineSet) {
    if (!std::ranges::contains(visited, ns)) visited.push_back(ns);
  }
  if (found.size() != start) return;

  for (auto ns : inlineSet) {
    for (auto directive : ns->usingDirectives()) {
      collectQualifiedNamespaceDeclarations(
          symbol_cast<NamespaceSymbol>(directive), name, accept, found,
          visited);
    }
  }
}

auto uniqueLookupDeclaration(const std::vector<Symbol*>& found) -> Symbol* {
  if (found.empty()) return nullptr;
  auto first = found.front();
  for (auto symbol : found) {
    if (symbol == first) continue;
    auto ns = resolve_namespace_alias(first);
    if (ns && ns == resolve_namespace_alias(symbol)) continue;
    return nullptr;
  }
  return first;
}

auto lookupNamespaceHelper(ScopeSymbol* scope, const Identifier* id,
                           std::vector<ScopeSymbol*>& visited)
    -> NamespaceSymbol* {
  std::vector<Symbol*> found;
  collectQualifiedNamespaceDeclarations(
      symbol_cast<NamespaceSymbol>(scope), id,
      [](Symbol* symbol) { return resolve_namespace_alias(symbol) != nullptr; },
      found, visited);
  return resolve_namespace_alias(uniqueLookupDeclaration(found));
}

auto lookupTypeHelper(ScopeSymbol* scope, const Identifier* id,
                      std::vector<ScopeSymbol*>& visited,
                      bool tagsAreTypes = true,
                      bool discardHiddenClassNames = false,
                      bool followUsingDirectives = true,
                      bool* ambiguous = nullptr) -> Symbol* {
  if (auto cls = symbol_cast<ClassSymbol>(scope)) {
    scope = cls->resolvedDefinition();
  }

  if (std::ranges::contains(visited, scope)) return nullptr;
  visited.push_back(scope);

  Symbol* fallback = nullptr;
  bool foundOtherDeclaration = false;
  for (auto candidate : scope->find(id)) {
    if (candidate->isHidden()) continue;

    if (auto u = symbol_cast<UsingDeclarationSymbol>(candidate);
        u && u->target()) {
      candidate = resolve_using_declaration(candidate);
    }

    if (is_type(candidate) || candidate->isNamespaceName()) {
      if (!tagsAreTypes && (symbol_cast<ClassSymbol>(candidate) ||
                            symbol_cast<EnumSymbol>(candidate) ||
                            symbol_cast<ScopedEnumSymbol>(candidate)))
        continue;

      if (symbol_cast<TypeAliasSymbol>(candidate)) return candidate;
      if (!fallback) fallback = candidate;
    } else if (bindsName(candidate)) {
      foundOtherDeclaration = true;
    }
  }

  if (discardHiddenClassNames && foundOtherDeclaration &&
      is_class_or_enum_declaration(fallback))
    fallback = nullptr;

  if (fallback) return fallback;

  auto accept = [tagsAreTypes](Symbol* symbol) {
    if (!is_type(symbol) && !symbol->isNamespaceName()) return false;
    return tagsAreTypes || !is_class_or_enum_declaration(symbol);
  };
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
    auto result = lookupClassMember(classSymbol, id, accept);
    if (ambiguous) *ambiguous = result.ambiguous;
    return result.ambiguous ? nullptr : result.symbol;
  }

  if (followUsingDirectives) {
    std::vector<Symbol*> found;
    std::vector<ScopeSymbol*> namespaceVisited;
    collectQualifiedNamespaceDeclarations(symbol_cast<NamespaceSymbol>(scope),
                                          id, accept, found, namespaceVisited);
    auto result = uniqueLookupDeclaration(found);
    if (ambiguous && !found.empty() && !result) *ambiguous = true;
    return result;
  }

  return nullptr;
}

auto resolveTypeScope(Symbol* symbol) -> ScopeSymbol* {
  if (!symbol) return nullptr;

  switch (symbol->kind()) {
    case SymbolKind::kNamespace:
    case SymbolKind::kClass:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum:
      return symbol->asScopeSymbol();

    case SymbolKind::kNamespaceAlias:
      return resolve_namespace_alias(symbol);

    case SymbolKind::kInjectedClassName: {
      auto injected = symbol_cast<InjectedClassNameSymbol>(symbol);
      return injected->classSymbol();
    }

    case SymbolKind::kTypeAlias: {
      auto alias = symbol_cast<TypeAliasSymbol>(symbol);
      auto aliasedType = unqualified_type(alias->type());
      if (auto ct = type_cast<ClassType>(aliasedType)) return ct->symbol();
      return nullptr;
    }

    case SymbolKind::kUsingDeclaration: {
      auto ud = symbol_cast<UsingDeclarationSymbol>(symbol);
      if (!ud->target()) return nullptr;
      auto target = resolve_using_declaration(symbol);
      if (auto cls = symbol_cast<ClassSymbol>(target)) return cls;
      if (auto en = symbol_cast<EnumSymbol>(target)) return en;
      if (auto se = symbol_cast<ScopedEnumSymbol>(target)) return se;
      return nullptr;
    }

    default:
      return nullptr;
  }
}
}  // namespace

auto qualifiedLookupType(Symbol* scopeOrAlias, const Identifier* id)
    -> Symbol* {
  auto resolved = resolveTypeScope(scopeOrAlias);
  if (!resolved) return nullptr;
  std::vector<ScopeSymbol*> visited;
  if (auto ns = symbol_cast<NamespaceSymbol>(resolved)) {
    std::vector<Symbol*> found;
    collectQualifiedNamespaceDeclarations(
        ns, id,
        [](Symbol* symbol) {
          return is_type(symbol) || symbol->isNamespaceName();
        },
        found, visited);
    return uniqueLookupDeclaration(found);
  }
  return lookupTypeHelper(resolved, id, visited);
}

auto qualifiedLookupNamespace(Symbol* scopeOrAlias, const Identifier* id)
    -> NamespaceSymbol* {
  auto base = resolve_namespace_alias(scopeOrAlias);
  if (!base) return nullptr;
  std::vector<ScopeSymbol*> visited;
  return lookupNamespaceHelper(base, id, visited);
}

namespace {
auto isContainedBy(NamespaceSymbol* ns, ScopeSymbol* scope) -> bool {
  for (auto parent = ns->parent(); parent; parent = parent->parent()) {
    if (parent == scope) return true;
  }
  return false;
}

void collectActiveNominatedNamespaces(ScopeSymbol* scope,
                                      std::vector<NamespaceSymbol*>& out) {
  for (auto directive : scope->usingDirectives()) {
    auto ns = symbol_cast<NamespaceSymbol>(directive);
    if (!ns) continue;
    if (std::ranges::contains(out, ns)) continue;
    out.push_back(ns);
    collectActiveNominatedNamespaces(ns, out);
  }
}

auto mergeDeclarations(Control* control, ScopeSymbol* scope, const Name* name,
                       std::vector<Symbol*>& found, bool* ambiguous)
    -> Symbol* {
  std::vector<Symbol*> distinct;
  for (auto symbol : found) {
    if (!std::ranges::contains(distinct, symbol)) distinct.push_back(symbol);
  }

  if (distinct.size() == 1) return distinct.front();

  std::vector<FunctionSymbol*> functions;
  for (auto symbol : distinct) {
    if (!symbol_cast<FunctionSymbol>(symbol) &&
        !symbol_cast<OverloadSetSymbol>(symbol)) {
      if (ambiguous) *ambiguous = true;
      return distinct.front();
    }

    for (auto func : views::each_function(symbol)) {
      auto canonical = func->canonical();
      if (!std::ranges::contains(functions, canonical))
        functions.push_back(canonical);
    }
  }

  auto merged =
      control->newOverloadSetSymbol(scope, distinct.front()->location());
  merged->setName(name);
  for (auto func : functions) merged->addFunction(func);
  return merged;
}

}  // namespace

auto unqualifiedLookupType(Scope* lexicalScope, const Identifier* id,
                           bool tagsAreTypes, bool discardHiddenClassNames)
    -> Symbol* {
  std::vector<NamespaceSymbol*> nominated;
  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    if (!sc->symbol) continue;
    collectActiveNominatedNamespaces(sc->symbol, nominated);
    std::vector<Symbol*> found;
    bool ambiguous = false;
    auto search = [&](ScopeSymbol* scope) {
      std::vector<ScopeSymbol*> visited;
      if (auto symbol =
              lookupTypeHelper(scope, id, visited, tagsAreTypes,
                               discardHiddenClassNames, false, &ambiguous))
        found.push_back(symbol);
    };
    search(sc->symbol);
    for (auto ns : nominated)
      if (isContainedBy(ns, sc->symbol)) search(ns);
    if (ambiguous || !found.empty()) return uniqueLookupDeclaration(found);
  }
  return nullptr;
}

auto unqualifiedLookupNamespace(Scope* lexicalScope, const Identifier* id)
    -> NamespaceSymbol* {
  std::vector<NamespaceSymbol*> nominated;
  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    if (!sc->symbol) continue;
    collectActiveNominatedNamespaces(sc->symbol, nominated);
    std::vector<Symbol*> found;
    auto search = [&](ScopeSymbol* scope) {
      for (auto candidate : scope->find(id))
        if (auto ns = resolve_namespace_alias(candidate)) found.push_back(ns);
    };
    search(sc->symbol);
    for (auto ns : nominated)
      if (isContainedBy(ns, sc->symbol)) search(ns);
    if (!found.empty())
      return resolve_namespace_alias(uniqueLookupDeclaration(found));
  }
  return nullptr;
}

auto unqualifiedLookupIncludingInlineNamespaces(Control* control,
                                                Scope* lexicalScope,
                                                const Name* name,
                                                bool skipClassNames,
                                                bool* ambiguous) -> Symbol* {
  if (!name) return nullptr;

  auto accept = [skipClassNames](Symbol* symbol) {
    return !skipClassNames || !symbol_cast<ClassSymbol>(symbol);
  };

  std::vector<NamespaceSymbol*> nominated;
  std::vector<ScopeSymbol*> visited;

  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    auto scope = sc->symbol;
    if (!scope) continue;

    collectActiveNominatedNamespaces(scope, nominated);

    std::vector<Symbol*> found;

    if (auto cls = symbol_cast<ClassSymbol>(scope)) {
      auto result = lookupClassMember(cls, name, accept);
      if (result.ambiguous) {
        if (ambiguous) *ambiguous = true;
        return ambiguous ? result.symbol : nullptr;
      }
      if (result.symbol) found.push_back(result.symbol);
    } else if (auto symbol =
                   detail::searchScope(scope, name, visited, accept, false)) {
      found.push_back(symbol);
    }

    for (auto ns : nominated) {
      if (!isContainedBy(ns, scope)) continue;
      if (auto symbol = detail::searchScope(ns, name, visited, accept,
                                            /*followUsingDirectives=*/false)) {
        found.push_back(symbol);
      }
    }

    if (found.empty()) continue;

    return mergeDeclarations(control, scope, name, found, ambiguous);
  }

  return nullptr;
}

auto qualifiedLookupIncludingInlineNamespaces(Control* control,
                                              Symbol* scopeOrAlias,
                                              const Name* name, bool* ambiguous)
    -> Symbol* {
  auto ns = resolve_namespace_alias(scopeOrAlias);
  if (!ns) {
    if (auto cls = symbol_cast<ClassSymbol>(resolveTypeScope(scopeOrAlias))) {
      auto result = lookupClassMember(cls, name, [](Symbol*) { return true; });
      if (ambiguous) *ambiguous = result.ambiguous;
      return result.ambiguous && !ambiguous ? nullptr : result.symbol;
    }
    return qualifiedLookup(scopeOrAlias, name);
  }
  std::vector<Symbol*> found;
  std::vector<ScopeSymbol*> visited;
  collectQualifiedNamespaceDeclarations(
      ns, name, [](Symbol*) { return true; }, found, visited);
  if (found.empty()) return nullptr;
  bool isAmbiguous = false;
  auto result = mergeDeclarations(control, ns, name, found, &isAmbiguous);
  if (ambiguous) *ambiguous = isAmbiguous;
  return isAmbiguous && !ambiguous ? nullptr : result;
}

auto designatedFunction(Symbol* symbol) -> FunctionSymbol* {
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) return function;

  auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol);
  if (!overloadSet) return nullptr;

  const auto functions = overloadSet->functions();
  if (functions.size() != 1) return nullptr;

  auto function = functions.front();
  if (function->templateDeclaration() && !function->isSpecialization())
    return nullptr;

  return function;
}

auto isPureFriend(FunctionSymbol* func) -> bool {
  if (!func) return false;
  auto canonical = func->canonical();
  if (!canonical->isFriend()) return false;
  auto isClassParented = [](FunctionSymbol* f) {
    return f->parent() && f->parent()->isClass();
  };
  return std::ranges::all_of(canonical->declarations(), isClassParented);
}

auto inlineNamespaceSet(NamespaceSymbol* namespaceSymbol)
    -> std::vector<NamespaceSymbol*> {
  std::vector<NamespaceSymbol*> result;
  if (!namespaceSymbol) return result;

  result.push_back(namespaceSymbol);

  for (std::size_t i = 0; i < result.size(); ++i) {
    for (auto directive : result[i]->usingDirectives()) {
      auto nested = symbol_cast<NamespaceSymbol>(directive);
      if (!nested || !nested->isInline()) continue;
      if (std::ranges::contains(result, nested)) continue;
      result.push_back(nested);
    }
  }

  return result;
}

auto bindsName(Symbol* symbol) -> bool {
  if (!symbol) return false;

  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return !isPureFriend(function);

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    const auto& functions = overloadSet->declaredFunctions();
    if (functions.empty()) return true;
    return !std::ranges::all_of(functions, isPureFriend);
  }

  return true;
}

void addOverloadCandidate(std::vector<FunctionSymbol*>& candidates,
                          FunctionSymbol* function) {
  if (!function) return;
  if (function->isSpecialization()) return;
  auto canonical = function->canonical();
  if (std::ranges::contains(candidates, canonical)) return;
  candidates.push_back(canonical);
}

auto argumentDependentLookup(TranslationUnit* unit, const Name* name,
                             std::span<const Type* const> argumentTypes)
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;
  if (!name) return result;

  std::vector<NamespaceSymbol*> namespaces;
  std::vector<ClassSymbol*> classes;
  std::vector<const Type*> visited;

  AssociatedNamespaceCollector collector{unit, namespaces, classes, visited};
  for (auto argType : argumentTypes) {
    auto argClassType =
        type_cast<ClassType>(unit->typeTraits().remove_cvref(argType));
    if (argClassType)
      unit->typeTraits().requireCompleteClass(argClassType->symbol());
    collector.collect(argType);
  }

  auto addCandidate = [&](FunctionSymbol* func) {
    if (!func->templateDeclaration() && isDependent(unit, func->type())) return;
    if (isPureFriend(func)) {
      auto befriending = symbol_cast<ClassSymbol>(func->parent());
      if (!befriending || !std::ranges::contains(classes, befriending)) return;
    }
    addOverloadCandidate(result, func);
  };

  for (auto ns : namespaces) {
    for (auto symbol : ns->find(name)) {
      for (auto func : views::each_function(symbol)) addCandidate(func);
    }
  }

  return result;
}

auto isDeferredDependentLookupContext(TranslationUnit* unit,
                                      Symbol* lookupContext, ScopeSymbol* scope)
    -> bool {
  auto classSymbol = symbol_cast<ClassSymbol>(lookupContext);
  if (!classSymbol) return false;
  if (!isDependent(unit, classSymbol->type())) return false;
  if (classSymbol->isSpecialization()) return true;
  return !names_current_instantiation(classSymbol, scope);
}

auto isArgumentDependentCallee(Symbol* symbol) -> bool {
  auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol);
  return overloadSet && overloadSet->declaredFunctions().empty() &&
         overloadSet->usingDeclarations().empty();
}

namespace {
void declareGlobalFunction(TranslationUnit* unit, ScopeSymbol* globalScope,
                           const Name* name, FunctionSymbol* function) {
  Binder binder{unit};
  auto overloadSet = binder.overloadSetFor(globalScope, name, {});
  overloadSet->addFunction(function);
}

}  // namespace

namespace {

auto hasNewExtendedAlignment(TranslationUnit* unit, const Type* objectType)
    -> bool {
  if (!objectType) return false;
  auto memoryLayout = unit->control()->memoryLayout();
  auto alignment = memoryLayout->alignmentOf(objectType);
  if (!alignment) return false;
  return *alignment > memoryLayout->defaultNewAlignment();
}

}  // namespace

auto deallocationSignatureOf(TranslationUnit* unit, FunctionSymbol* fn)
    -> std::optional<DeallocationSignature> {
  if (!fn) return std::nullopt;
  auto funcType = type_cast<FunctionType>(fn->type());
  if (!funcType) return std::nullopt;
  if (fn->isSpecialization()) return std::nullopt;

  const std::span parameterTypes{funcType->parameterTypes()};
  if (parameterTypes.empty()) return std::nullopt;

  auto traits = unit->typeTraits();

  auto firstParameter = type_cast<PointerType>(parameterTypes[0]);
  if (!firstParameter) return std::nullopt;

  DeallocationSignature signature;
  auto rest = parameterTypes.subspan(1);
  std::size_t index = 0;

  auto enclosingClass = fn->enclosingClass();

  if (!rest.empty() && enclosingClass &&
      traits.is_destroying_delete_t(rest[0])) {
    if (!traits.is_same(firstParameter->elementType(), enclosingClass->type()))
      return std::nullopt;
    signature.isDestroying = true;
    index = 1;
  } else if (!traits.is_void(firstParameter->elementType())) {
    return std::nullopt;
  }

  if (index < rest.size() &&
      traits.is_same(rest[index], unit->control()->getSizeType())) {
    signature.hasSize = true;
    ++index;
  }

  if (index < rest.size() && traits.is_align_val_t(rest[index])) {
    signature.hasAlignment = true;
    ++index;
  }

  if (index != rest.size()) return std::nullopt;

  return signature;
}

namespace {

auto declareGlobalOperatorDelete(TranslationUnit* unit, const Name* name)
    -> FunctionSymbol* {
  auto control = unit->control();
  auto globalScope = unit->globalScope();
  auto voidType = control->getVoidType();
  auto fn = control->newFunctionSymbol(globalScope, {});
  fn->setName(name);
  fn->setType(
      control->getFunctionType(voidType, {control->getPointerType(voidType)}));
  fn->setLanguageLinkage(LanguageKind::kCXX);
  declareGlobalFunction(unit, globalScope, name, fn);
  return fn;
}

}  // namespace

auto resolveUsualOperatorDelete(TranslationUnit* unit, ClassSymbol* classSymbol,
                                const Type* objectType, bool isArrayDelete)
    -> FunctionSymbol* {
  auto control = unit->control();
  auto name = control->getOperatorId(isArrayDelete ? TokenKind::T_DELETE_ARRAY
                                                   : TokenKind::T_DELETE);

  std::vector<std::pair<FunctionSymbol*, DeallocationSignature>> candidates;

  auto collect = [&](Symbol* declarations) {
    for (auto fn : views::each_function(declarations)) {
      if (auto signature = deallocationSignatureOf(unit, fn))
        candidates.emplace_back(fn, *signature);
    }
  };

  auto inClassScope = false;

  if (classSymbol) {
    if (auto declarations =
            qualifiedLookup(classSymbol->resolvedDefinition(), name)) {
      inClassScope = true;
      collect(declarations);
    }
  }

  if (!inClassScope) {
    auto globalScope = unit->globalScope();
    auto declarations = qualifiedLookup(globalScope, name);
    if (!declarations) return declareGlobalOperatorDelete(unit, name);
    collect(declarations);
  }

  if (candidates.empty()) return nullptr;

  auto keepIf = [&](auto&& predicate) {
    decltype(candidates) kept;
    for (auto& candidate : candidates)
      if (predicate(candidate.second)) kept.push_back(candidate);
    if (!kept.empty()) candidates = std::move(kept);
    return !kept.empty();
  };

  (void)keepIf([](const DeallocationSignature& s) { return s.isDestroying; });

  const auto overAligned = hasNewExtendedAlignment(unit, objectType);
  (void)keepIf([&](const DeallocationSignature& s) {
    return s.hasAlignment == overAligned;
  });

  if (candidates.size() == 1) return candidates.front().first;

  if (inClassScope) {
    for (auto& [fn, signature] : candidates)
      if (!signature.hasSize) return fn;
    return nullptr;
  }

  auto traits = unit->typeTraits();
  auto selectSized =
      traits.is_complete(objectType) &&
      (!isArrayDelete || !traits.has_trivial_destructor(objectType));

  for (auto& [fn, signature] : candidates)
    if (signature.hasSize == selectSized) return fn;

  return candidates.front().first;
}

auto declareGlobalOperatorNew(TranslationUnit* unit, bool isArrayNew)
    -> FunctionSymbol* {
  auto control = unit->control();
  auto name = control->getOperatorId(isArrayNew ? TokenKind::T_NEW_ARRAY
                                                : TokenKind::T_NEW);

  auto sizeType = control->getSizeType();
  auto globalScope = unit->globalScope();

  auto matches = [&](FunctionSymbol* fn) {
    auto funcType = type_cast<FunctionType>(fn->type());
    if (!funcType || funcType->parameterTypes().size() != 1) return false;
    return unit->typeTraits().is_same(funcType->parameterTypes()[0], sizeType);
  };

  if (auto symbol = qualifiedLookup(globalScope, name)) {
    if (auto fn = views::find_function(views::each_function(symbol), matches))
      return fn;
  }

  auto voidType = control->getVoidType();
  auto fn = control->newFunctionSymbol(globalScope, {});
  fn->setName(name);
  fn->setType(
      control->getFunctionType(control->getPointerType(voidType), {sizeType}));
  fn->setLanguageLinkage(LanguageKind::kCXX);
  declareGlobalFunction(unit, globalScope, name, fn);
  return fn;
}

namespace {

auto findOrDeclareGlobalAllocationFunction(
    TranslationUnit* unit, TokenKind op, const Type* returnType,
    std::vector<const Type*> parameterTypes) -> FunctionSymbol* {
  if (std::ranges::contains(parameterTypes, nullptr)) return nullptr;

  auto control = unit->control();
  auto name = control->getOperatorId(op);
  auto globalScope = unit->globalScope();

  auto matches = [&](FunctionSymbol* fn) {
    auto funcType = type_cast<FunctionType>(fn->type());
    return funcType &&
           std::ranges::equal(funcType->parameterTypes(), parameterTypes);
  };

  if (auto fn = views::find_function(globalScope->find(name), matches))
    return fn;

  auto fn = control->newFunctionSymbol(globalScope, {});
  fn->setName(name);
  fn->setType(control->getFunctionType(returnType, std::move(parameterTypes)));
  fn->setLanguageLinkage(LanguageKind::kCXX);
  declareGlobalFunction(unit, globalScope, name, fn);
  return fn;
}

}  // namespace

auto resolveBuiltinOperatorDelete(TranslationUnit* unit,
                                  std::span<const Type* const> argumentTypes)
    -> FunctionSymbol* {
  auto control = unit->control();
  auto voidType = control->getVoidType();

  std::vector<const Type*> parameterTypes;
  parameterTypes.push_back(control->getPointerType(voidType));
  for (auto argumentType :
       argumentTypes.subspan(std::min<std::size_t>(1, argumentTypes.size())))
    parameterTypes.push_back(argumentType);

  return findOrDeclareGlobalAllocationFunction(
      unit, TokenKind::T_DELETE, voidType, std::move(parameterTypes));
}

auto resolveBuiltinOperatorNew(TranslationUnit* unit,
                               std::span<const Type* const> argumentTypes)
    -> FunctionSymbol* {
  auto control = unit->control();

  std::vector<const Type*> parameterTypes;
  parameterTypes.push_back(control->getSizeType());
  for (auto argumentType :
       argumentTypes.subspan(std::min<std::size_t>(1, argumentTypes.size())))
    parameterTypes.push_back(argumentType);

  return findOrDeclareGlobalAllocationFunction(
      unit, TokenKind::T_NEW, control->getPointerType(control->getVoidType()),
      std::move(parameterTypes));
}

auto resolveBuiltinLibcallSymbol(TranslationUnit* unit, const char* nameStr,
                                 const FunctionType* funcType)
    -> FunctionSymbol* {
  auto control = unit->control();
  auto name = control->getIdentifier(nameStr);
  auto globalScope = unit->globalScope();

  auto matches = [&](FunctionSymbol* fn) {
    return fn->hasCLinkage() && fn->type() == funcType;
  };

  if (auto fn = views::find_function(globalScope->find(name), matches))
    return fn;

  auto fn = control->newFunctionSymbol(globalScope, {});
  fn->setName(name);
  fn->setType(funcType);
  fn->setLanguageLinkage(LanguageKind::kC);
  globalScope->addSymbol(fn);
  return fn;
}

auto resolveBuiltinFunctionSymbol(TranslationUnit* unit, const Identifier* name,
                                  BuiltinFunctionKind kind) -> Symbol* {
  if (kind == BuiltinFunctionKind::T_NONE) return nullptr;

  auto globalScope = unit->globalScope();

  auto isBuiltin = [&](FunctionSymbol* fn) {
    return fn->builtinKind() == kind;
  };

  for (auto symbol : globalScope->find(name)) {
    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
      if (views::find_function(views::each_function(overloadSet), isBuiltin))
        return overloadSet;
      continue;
    }
    if (auto fn = symbol_cast<FunctionSymbol>(symbol)) {
      if (isBuiltin(fn)) return fn;
    }
  }

  auto signature = builtinSignatureOf(kind);
  if (!signature.count) return nullptr;

  auto control = unit->control();

  Symbol* result = nullptr;

  for (std::size_t index = 0; index < signature.count; ++index) {
    auto overload = decodeBuiltinSignature(control, kind, index);
    if (overload.status == BuiltinOverloadStatus::kUnavailable) continue;
    if (overload.status != BuiltinOverloadStatus::kOk) return result;

    auto fn = control->newFunctionSymbol(globalScope, {});
    fn->setName(name);
    fn->setType(overload.type);
    fn->setBuiltinKind(kind);
    fn->setConstexpr(contains(signature.flags, BuiltinFlags::kConstexpr));
    fn->setNoReturn(contains(signature.flags, BuiltinFlags::kNoReturn));
    fn->setExceptionSpecifier(
        contains(signature.flags, BuiltinFlags::kNoexcept));

    if (unit->language() == LanguageKind::kCXX)
      fn->setConsteval(contains(signature.flags, BuiltinFlags::kConsteval));

    declareGlobalFunction(unit, globalScope, name, fn);

    result = fn;
  }

  if (signature.count > 1) {
    for (auto symbol : globalScope->find(name)) {
      if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol))
        return overloadSet;
    }
  }

  return result;
}

auto lookupClassMember(ClassSymbol* scope, const Name* name,
                       const std::function<bool(Symbol*)>& accept)
    -> ClassMemberLookup {
  if (!scope || !name) return {};
  scope = scope->resolvedDefinition();
  if (!scope) return {};
  std::vector<ScopeSymbol*> visited;
  if (auto symbol =
          detail::searchScope(scope, name, visited, accept, false, false))
    return {symbol, false};
  if (scope->baseClasses().empty()) return {};

  struct Subobject {
    ClassSymbol* type;
    std::vector<int> bases;
  };
  struct LookupSet {
    Symbol* declaration = nullptr;
    std::vector<int> subobjects;
    bool ambiguous = false;
  };
  std::vector<Subobject> graph;
  std::vector<std::pair<ClassSymbol*, int>> virtualBases;
  std::vector<ClassSymbol*> path;
  auto build = [&](auto&& self, ClassSymbol* cls) -> int {
    cls = cls->resolvedDefinition();
    auto index = int(graph.size());
    graph.push_back({cls, {}});
    if (std::ranges::contains(path, cls)) return index;
    path.push_back(cls);
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass || isDependent(nullptr, baseClass->type())) continue;
      baseClass = baseClass->resolvedDefinition();
      int baseIndex = -1;
      if (base->isVirtual()) {
        for (auto [type, candidate] : virtualBases)
          if (type == baseClass) baseIndex = candidate;
      }
      if (baseIndex < 0) {
        baseIndex = self(self, baseClass);
        if (base->isVirtual()) virtualBases.emplace_back(baseClass, baseIndex);
      }
      graph[index].bases.push_back(baseIndex);
    }
    path.pop_back();
    return index;
  };
  build(build, scope);

  auto isBase = [&](auto&& self, int base, int derived) -> bool {
    if (base == derived) return true;
    for (auto next : graph[derived].bases)
      if (self(self, base, next)) return true;
    return false;
  };
  auto sameDeclaration = [&](Symbol* first, Symbol* second) {
    if (first == second) return true;
    if (is_type(first) && is_type(second))
      return first->type() == second->type();
    std::vector<FunctionSymbol*> left;
    std::vector<FunctionSymbol*> right;
    for (auto fn : views::each_function(first)) left.push_back(fn->canonical());
    for (auto fn : views::each_function(second))
      right.push_back(fn->canonical());
    if (left.empty() || right.empty()) return false;
    return std::ranges::is_permutation(left, right);
  };
  std::vector<std::optional<LookupSet>> cache(graph.size());
  auto lookup = [&](auto&& self, int index) -> LookupSet {
    if (cache[index]) return *cache[index];
    LookupSet result;
    std::vector<ScopeSymbol*> directVisited;
    result.declaration = detail::searchScope(
        graph[index].type, name, directVisited, accept, false, false);
    if (result.declaration) {
      result.subobjects.push_back(index);
    } else {
      for (auto base : graph[index].bases) {
        auto incoming = self(self, base);
        if (!incoming.declaration) continue;
        if (!result.declaration) {
          result = std::move(incoming);
          continue;
        }
        auto dominated = [&](const auto& first, const auto& second) {
          return std::ranges::all_of(first, [&](int a) {
            return std::ranges::any_of(
                second, [&](int b) { return isBase(isBase, a, b); });
          });
        };
        if (dominated(incoming.subobjects, result.subobjects)) continue;
        if (dominated(result.subobjects, incoming.subobjects)) {
          result = std::move(incoming);
          continue;
        }
        result.ambiguous =
            result.ambiguous || incoming.ambiguous ||
            !sameDeclaration(result.declaration, incoming.declaration);
        for (auto subobject : incoming.subobjects)
          if (!std::ranges::contains(result.subobjects, subobject))
            result.subobjects.push_back(subobject);
      }
    }
    cache[index] = result;
    return result;
  };
  auto result = lookup(lookup, 0);
  return {result.declaration, result.ambiguous};
}

}  // namespace cxx
