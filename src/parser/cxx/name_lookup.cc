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
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {
namespace {
struct AssociatedNamespaceCollector {
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
    if (ns && !std::ranges::contains(namespaces, ns)) namespaces.push_back(ns);
  }

  void addClass(ClassSymbol* cls) {
    if (cls && !std::ranges::contains(classes, cls)) classes.push_back(cls);
  }

  void collect(const Symbol* symbol) {
    if (!symbol) return;
    collect(symbol->type());
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
    if (std::ranges::contains(classes, classSymbol)) return;
    addClass(classSymbol);

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
    if (auto sym = type->symbol()) addNamespace(sym->enclosingNamespace());
  }

  void operator()(const ScopedEnumType* type) {
    if (auto sym = type->symbol()) addNamespace(sym->enclosingNamespace());
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
auto lookupNamespaceHelper(ScopeSymbol* scope, const Identifier* id,
                           std::vector<ScopeSymbol*>& visited)
    -> NamespaceSymbol* {
  if (std::ranges::contains(visited, scope)) return nullptr;
  visited.push_back(scope);

  for (auto candidate : scope->find(id) | views::namespaces) {
    return candidate;
  }

  for (auto u : scope->usingDirectives()) {
    if (auto ns = lookupNamespaceHelper(u, id, visited)) return ns;
  }

  return nullptr;
}

auto lookupTypeHelper(ScopeSymbol* scope, const Identifier* id,
                      std::vector<ScopeSymbol*>& visited,
                      bool tagsAreTypes = true,
                      bool discardHiddenClassNames = false) -> Symbol* {
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

    if (is_type(candidate) || candidate->isNamespace()) {
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

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (auto s = lookupTypeHelper(baseClass, id, visited, tagsAreTypes))
        return s;
    }
  }

  for (auto u : scope->usingDirectives()) {
    if (auto s = lookupTypeHelper(u, id, visited, tagsAreTypes)) return s;
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

    case SymbolKind::kInjectedClassName: {
      auto injected = symbol_cast<InjectedClassNameSymbol>(symbol);
      return injected->classSymbol();
    }

    case SymbolKind::kTypeAlias: {
      auto alias = symbol_cast<TypeAliasSymbol>(symbol);
      if (auto ct = type_cast<ClassType>(alias->type())) return ct->symbol();
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

auto unqualifiedLookupType(Scope* lexicalScope, const Identifier* id,
                           bool tagsAreTypes, bool discardHiddenClassNames)
    -> Symbol* {
  std::vector<ScopeSymbol*> visited;
  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    if (!sc->symbol) continue;
    if (auto s = lookupTypeHelper(sc->symbol, id, visited, tagsAreTypes,
                                  discardHiddenClassNames))
      return s;
  }
  return nullptr;
}

auto qualifiedLookupType(Symbol* scopeOrAlias, const Identifier* id)
    -> Symbol* {
  auto resolved = resolveTypeScope(scopeOrAlias);
  if (!resolved) return nullptr;
  std::vector<ScopeSymbol*> visited;
  return lookupTypeHelper(resolved, id, visited);
}

auto unqualifiedLookupNamespace(Scope* lexicalScope, const Identifier* id)
    -> NamespaceSymbol* {
  std::vector<ScopeSymbol*> visited;
  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    if (!sc->symbol) continue;
    if (auto ns = lookupNamespaceHelper(sc->symbol, id, visited)) return ns;
  }
  return nullptr;
}

auto qualifiedLookupNamespace(Symbol* scopeOrAlias, const Identifier* id)
    -> NamespaceSymbol* {
  auto base = symbol_cast<NamespaceSymbol>(scopeOrAlias);
  if (!base) return nullptr;
  std::vector<ScopeSymbol*> visited;
  return lookupNamespaceHelper(base, id, visited);
}

namespace {
void collectInlineNamespaceFunctions(ScopeSymbol* scope, const Name* name,
                                     std::vector<FunctionSymbol*>& out,
                                     std::vector<ScopeSymbol*>& visited) {
  if (std::ranges::contains(visited, scope)) return;
  visited.push_back(scope);

  auto add = [&](FunctionSymbol* func) {
    auto canonical = func->canonical();
    if (!std::ranges::contains(out, canonical)) out.push_back(canonical);
  };

  for (auto directive : scope->usingDirectives()) {
    auto ns = symbol_cast<NamespaceSymbol>(directive);
    if (!ns) continue;
    if (!ns->isInline() && ns->name()) continue;

    for (auto symbol : ns->find(name)) {
      if (symbol->isHidden()) continue;
      for (auto func : views::each_function(symbol)) add(func);
    }

    collectInlineNamespaceFunctions(ns, name, out, visited);
  }
}
}  // namespace

auto mergeInlineNamespaceOverloads(Control* control, NamespaceSymbol* scope,
                                   const Name* name, Symbol* primary)
    -> Symbol* {
  if (!control || !scope || !primary) return primary;

  const bool primaryIsFunction = symbol_cast<FunctionSymbol>(primary);
  if (!primaryIsFunction && !symbol_cast<OverloadSetSymbol>(primary))
    return primary;

  if (!scope->hasInlineNamespaces()) return primary;

  auto isOverloadable = [](Symbol* s) {
    return !s->isHidden() && (symbol_cast<FunctionSymbol>(s) ||
                              symbol_cast<OverloadSetSymbol>(s));
  };
  if (!std::ranges::any_of(scope->find(name), isOverloadable)) return primary;

  std::vector<FunctionSymbol*> functions;
  auto add = [&](FunctionSymbol* func) {
    auto canonical = func->canonical();
    if (!std::ranges::contains(functions, canonical))
      functions.push_back(canonical);
  };

  for (auto func : views::each_function(primary)) add(func);

  const auto directCount = functions.size();

  std::vector<ScopeSymbol*> visited;
  collectInlineNamespaceFunctions(scope, name, functions, visited);

  if (functions.size() == directCount) return primary;

  auto merged = control->newOverloadSetSymbol(scope, primary->location());
  merged->setName(name);
  for (auto func : functions) merged->addFunction(func);
  return merged;
}

auto qualifiedLookupIncludingInlineNamespaces(Control* control,
                                              Symbol* scopeOrAlias,
                                              const Name* name) -> Symbol* {
  auto symbol = qualifiedLookup(scopeOrAlias, name);
  auto ns = symbol_cast<NamespaceSymbol>(scopeOrAlias);
  if (!ns) return symbol;
  return mergeInlineNamespaceOverloads(control, ns, name, symbol);
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
  if (!isClassParented(canonical)) return false;
  for (auto redecl : canonical->redeclarations()) {
    if (!isClassParented(redecl)) return false;
  }
  return true;
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

  AssociatedNamespaceCollector collector{namespaces, classes, visited};
  for (auto argType : argumentTypes) collector.collect(argType);

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

auto resolveUsualOperatorDelete(TranslationUnit* unit, ClassSymbol* classSymbol,
                                bool isArrayDelete) -> FunctionSymbol* {
  auto control = unit->control();
  auto name = control->getOperatorId(isArrayDelete ? TokenKind::T_DELETE_ARRAY
                                                   : TokenKind::T_DELETE);

  auto matches = [&](FunctionSymbol* fn) {
    auto funcType = type_cast<FunctionType>(fn->type());
    if (!funcType || funcType->parameterTypes().size() != 1) return false;
    auto param = funcType->parameterTypes()[0];
    auto pointer = type_cast<PointerType>(param);
    return pointer && unit->typeTraits().is_void(pointer->elementType());
  };

  auto findUsual = [&](auto&& symbols) -> FunctionSymbol* {
    return views::find_function(symbols, matches);
  };

  if (classSymbol) {
    classSymbol = classSymbol->resolvedDefinition();
    if (auto fn = findUsual(classSymbol->find(name))) return fn;
  }

  auto globalScope = unit->globalScope();
  if (auto fn = findUsual(globalScope->find(name))) return fn;

  auto voidType = control->getVoidType();
  auto fn = control->newFunctionSymbol(globalScope, {});
  fn->setName(name);
  fn->setType(
      control->getFunctionType(voidType, {control->getPointerType(voidType)}));
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
}  // namespace cxx
