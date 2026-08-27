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

#include <cxx/access_control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>

namespace cxx {

namespace {

[[nodiscard]] auto normalize(ClassSymbol* classSymbol) -> ClassSymbol* {
  if (!classSymbol) return nullptr;
  return classSymbol->resolvedDefinition();
}

[[nodiscard]] auto templateArgumentsOf(auto* symbol)
    -> std::vector<TemplateArgument> {
  auto arguments = symbol->templateArguments();
  return std::vector<TemplateArgument>{arguments.begin(), arguments.end()};
}

[[nodiscard]] auto instantiationChainOf(ClassSymbol* classSymbol)
    -> std::vector<ClassSymbol*> {
  std::vector<ClassSymbol*> chain;
  std::vector<ClassSymbol*> pending{classSymbol};

  while (!pending.empty()) {
    auto current = normalize(pending.back());
    pending.pop_back();
    if (!current) continue;
    if (std::ranges::contains(chain, current)) continue;
    chain.push_back(current);
    pending.push_back(current->instantiationPattern());
    pending.push_back(current->primaryTemplateSymbol());
  }

  return chain;
}

[[nodiscard]] auto isFriendDeclaration(Symbol* symbol) -> bool {
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return function->isFriend();
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
    return classSymbol->isFriend();
  return false;
}

[[nodiscard]] auto isSameClassForAccess(ClassSymbol* lhs, ClassSymbol* rhs)
    -> bool {
  lhs = normalize(lhs);
  rhs = normalize(rhs);
  if (!lhs || !rhs) return false;
  if (lhs == rhs) return true;
  if (std::ranges::contains(instantiationChainOf(lhs), rhs)) return true;
  return std::ranges::contains(instantiationChainOf(rhs), lhs);
}

[[nodiscard]] auto derivesFromForAccess(ClassSymbol* derived, ClassSymbol* base)
    -> bool {
  std::vector<ClassSymbol*> visited;
  std::vector<ClassSymbol*> pending{derived};

  while (!pending.empty()) {
    auto current = normalize(pending.back());
    pending.pop_back();
    if (!current) continue;
    if (std::ranges::contains(visited, current)) continue;
    visited.push_back(current);

    for (auto baseClass : current->resolvedDefinition()->baseClasses()) {
      auto candidate = normalize(symbol_cast<ClassSymbol>(baseClass->symbol()));
      if (!candidate) continue;
      if (isSameClassForAccess(candidate, base)) return true;
      pending.push_back(candidate);
    }
  }

  return false;
}

[[nodiscard]] auto introduces(UsingDeclarationSymbol* usingDeclaration,
                              Symbol* member) -> bool {
  if (auto function = symbol_cast<FunctionSymbol>(member)) {
    return std::ranges::any_of(usingDeclaration->introducedFunctions(),
                               [&](FunctionSymbol* introduced) {
                                 return introduced->canonical() ==
                                        function->canonical();
                               });
  }

  auto target = usingDeclaration->target();
  if (!target) return false;
  return target->canonical() == member->canonical();
}

[[nodiscard]] auto isNonStaticMember(Symbol* member) -> bool {
  if (auto field = symbol_cast<FieldSymbol>(member)) return !field->isStatic();
  if (auto function = symbol_cast<FunctionSymbol>(member))
    return !function->isStatic();
  return false;
}

}  // namespace

auto usingDeclarationIntroducing(Symbol* member, ClassSymbol* designatingClass)
    -> UsingDeclarationSymbol* {
  if (!member || !member->name()) return nullptr;
  if (symbol_cast<UsingDeclarationSymbol>(member)) return nullptr;

  for (auto candidate :
       designatingClass->resolvedDefinition()->find(member->name())) {
    if (auto usingDeclaration =
            symbol_cast<UsingDeclarationSymbol>(candidate)) {
      if (introduces(usingDeclaration, member)) return usingDeclaration;
      continue;
    }

    auto overloadSet = symbol_cast<OverloadSetSymbol>(candidate);
    if (!overloadSet) continue;

    for (auto usingDeclaration : overloadSet->usingDeclarations()) {
      if (introduces(usingDeclaration, member)) return usingDeclaration;
    }
  }

  return nullptr;
}

[[nodiscard]] auto injectsMembersIntoEnclosingClass(ClassSymbol* classSymbol)
    -> bool {
  if (classSymbol->name()) return false;

  auto enclosingClass = symbol_cast<ClassSymbol>(classSymbol->parent());
  if (!enclosingClass) return false;

  for (auto field : views::members(enclosingClass) | views::non_static_fields) {
    if (field->name()) continue;
    auto fieldClass = type_cast<ClassType>(field->type());
    if (!fieldClass) continue;
    if (normalize(fieldClass->symbol()) == normalize(classSymbol)) return true;
  }

  return false;
}

auto declaredMemberOf(Symbol* member) -> DeclaredMember {
  if (!member) return {};
  if (isFriendDeclaration(member)) return {};

  auto access = member->accessSpecifier();

  for (auto declaringScope = member->parent(); declaringScope;
       declaringScope = declaringScope->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(declaringScope)) {
      if (!injectsMembersIntoEnclosingClass(classSymbol))
        return {normalize(classSymbol), access};
      access = std::max(access, classSymbol->accessSpecifier());
      continue;
    }

    if (declaringScope->isEnum() || declaringScope->isScopedEnum()) {
      access = std::max(access, declaringScope->accessSpecifier());
      continue;
    }

    return {};
  }

  return {};
}

auto declaringClassOf(Symbol* member) -> ClassSymbol* {
  return declaredMemberOf(member).declaringClass;
}

auto designatingClassOf(Symbol* member, ScopeSymbol* accessingScope)
    -> ClassSymbol* {
  auto declaringClass = declaringClassOf(member);
  if (!declaringClass) return nullptr;
  if (!accessingScope) return declaringClass;

  auto innermostClass = symbol_cast<ClassSymbol>(accessingScope);
  if (!innermostClass) innermostClass = accessingScope->enclosingClass();

  for (auto enclosingClass = innermostClass; enclosingClass;
       enclosingClass = enclosingClass->enclosingClass()) {
    auto normalized = normalize(enclosingClass);
    if (isSameClassForAccess(normalized, declaringClass)) return normalized;
    if (derivesFromForAccess(normalized, declaringClass)) return normalized;
  }

  return declaringClass;
}

auto isProtectedAccessRestricted(Symbol* member) -> bool {
  if (!member) return false;
  if (member->accessSpecifier() != AccessSpecifier::kProtected) return false;
  return isNonStaticMember(member);
}

AccessContext::AccessContext(TranslationUnit* unit, ScopeSymbol* accessingScope)
    : unit_(unit), accessingScope_(accessingScope) {}

auto AccessContext::classes() const -> std::span<ClassSymbol* const> {
  materialize();
  return classes_;
}

void AccessContext::materialize() const {
  if (materialized_) return;
  materialized_ = true;

  std::vector<ClassSymbol*> visitedClasses;
  std::vector<FunctionSymbol*> visitedFunctions;

  const auto grantAccessOf = [&](ClassSymbol* classSymbol) {
    classSymbol = normalize(classSymbol);
    if (!classSymbol) return;
    if (std::ranges::contains(classes_, classSymbol)) return;
    classes_.push_back(classSymbol);
  };

  const auto addEnclosingClass = [&](ClassSymbol* classSymbol) {
    classSymbol = normalize(classSymbol);
    if (!classSymbol) return;
    if (std::ranges::contains(enclosingClasses_, classSymbol)) return;
    enclosingClasses_.push_back(classSymbol);
  };

  const auto grantTemplateFriendships =
      [&](const std::vector<TemplateFriendship>& friendships,
          const std::vector<TemplateArgument>& arguments) {
        for (const auto& friendship : friendships) {
          if (friendship.arguments.size() != arguments.size()) continue;
          if (friendship.arguments != arguments) {
            if (!compare_args(unit_, friendship.arguments, arguments)) continue;
          }
          grantAccessOf(friendship.befriendingClass);
        }
      };

  const auto grantFriendshipsOfClass = [&](ClassSymbol* seed) {
    if (!seed) return;
    auto arguments = templateArgumentsOf(seed);
    std::vector<ClassSymbol*> pending{seed};
    while (!pending.empty()) {
      auto classSymbol = pending.back();
      pending.pop_back();
      if (!classSymbol) continue;
      if (std::ranges::contains(visitedClasses, classSymbol)) continue;
      visitedClasses.push_back(classSymbol);
      grantAccessOf(classSymbol);
      for (auto befriending : classSymbol->befriendingClasses())
        grantAccessOf(befriending);
      grantTemplateFriendships(classSymbol->templateFriendships(), arguments);
      pending.push_back(classSymbol->canonical());
      pending.push_back(classSymbol->resolvedDefinition());
      pending.push_back(classSymbol->instantiationPattern());
      pending.push_back(classSymbol->primaryTemplateSymbol());
    }
  };

  const auto grantFriendshipsOfFunction = [&](FunctionSymbol* seed) {
    if (!seed) return;
    auto arguments = templateArgumentsOf(seed);
    std::vector<FunctionSymbol*> pending{seed};
    while (!pending.empty()) {
      auto function = pending.back();
      pending.pop_back();
      if (!function) continue;
      function = function->canonical();
      if (std::ranges::contains(visitedFunctions, function)) continue;
      visitedFunctions.push_back(function);
      for (auto befriending : function->befriendingClasses())
        grantAccessOf(befriending);
      grantTemplateFriendships(function->templateFriendships(), arguments);
      pending.push_back(function->primaryTemplateSymbol());
    }
  };

  for (auto current = accessingScope_; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current)) {
      addEnclosingClass(classSymbol);
      grantFriendshipsOfClass(classSymbol);
      continue;
    }

    auto function = symbol_cast<FunctionSymbol>(current);
    if (!function) continue;

    grantFriendshipsOfFunction(function);

    auto functionClass = symbol_cast<ClassSymbol>(function->parent());
    if (functionClass && !function->isFriend()) {
      addEnclosingClass(functionClass);
      grantFriendshipsOfClass(functionClass);
    }
  }
}

auto AccessContext::isMemberOrFriendOf(ClassSymbol* classSymbol) const -> bool {
  materialize();
  return std::ranges::any_of(
      instantiationChainOf(classSymbol), [&](ClassSymbol* candidate) {
        return std::ranges::contains(classes_, candidate);
      });
}

auto AccessContext::isMemberOf(ClassSymbol* classSymbol) const -> bool {
  materialize();
  return std::ranges::any_of(
      instantiationChainOf(classSymbol), [&](ClassSymbol* candidate) {
        return std::ranges::contains(enclosingClasses_, candidate);
      });
}

auto AccessContext::isAccessibleBaseClassEdge(ClassSymbol* derived,
                                              BaseClassSymbol* baseClass) const
    -> bool {
  switch (baseClass->accessSpecifier()) {
    case AccessSpecifier::kPublic:
      return true;

    case AccessSpecifier::kPrivate:
      return isMemberOrFriendOf(derived);

    case AccessSpecifier::kProtected: {
      if (isMemberOrFriendOf(derived)) return true;
      return std::ranges::any_of(classes(), [&](ClassSymbol* candidate) {
        return derivesFromForAccess(candidate, derived);
      });
    }
  }

  return false;
}

auto AccessContext::isAccessibleBaseClass(ClassSymbol* derived,
                                          ClassSymbol* base) const -> bool {
  derived = normalize(derived);
  base = normalize(base);
  if (!derived || !base) return false;
  if (derived == base) return true;
  if (derived->isAccessControlDisabled()) return true;

  for (auto baseClass : derived->baseClasses()) {
    auto directBase = normalize(symbol_cast<ClassSymbol>(baseClass->symbol()));
    if (!directBase) continue;
    if (!isAccessibleBaseClassEdge(derived, baseClass)) continue;
    if (directBase == base) return true;
    if (isAccessibleBaseClass(directBase, base)) return true;
  }

  return false;
}

auto AccessContext::satisfiesProtectedObjectRestriction(
    Symbol* member, ClassSymbol* grantingClass, ClassSymbol* objectClass) const
    -> bool {
  if (!isProtectedAccessRestricted(member)) return true;
  if (!objectClass) return true;

  objectClass = normalize(objectClass);
  if (isSameClassForAccess(objectClass, grantingClass)) return true;

  return derivesFromForAccess(objectClass, grantingClass);
}

auto AccessContext::accessAsMemberOf(Symbol* member,
                                     ClassSymbol* designatingClass,
                                     AccessMemo& memo) const
    -> std::optional<AccessSpecifier> {
  designatingClass = normalize(designatingClass);
  if (!designatingClass) return std::nullopt;

  auto entry =
      std::ranges::find(memo, designatingClass, &AccessMemo::value_type::first);
  if (entry != memo.end()) return entry->second;

  memo.emplace_back(designatingClass, std::nullopt);
  const auto index = memo.size() - 1;

  auto declared = declaredMemberOf(member);
  if (isSameClassForAccess(declared.declaringClass, designatingClass)) {
    memo[index].second = declared.accessSpecifier;
    return memo[index].second;
  }

  if (auto usingDeclaration =
          usingDeclarationIntroducing(member, designatingClass)) {
    memo[index].second = usingDeclaration->accessSpecifier();
    return memo[index].second;
  }

  std::optional<AccessSpecifier> best;

  for (auto baseClass : designatingClass->baseClasses()) {
    auto base = normalize(symbol_cast<ClassSymbol>(baseClass->symbol()));
    if (!base) continue;

    auto accessInBase = accessAsMemberOf(member, base, memo);
    if (!accessInBase) continue;
    if (*accessInBase == AccessSpecifier::kPrivate) continue;

    auto combined = std::max(*accessInBase, baseClass->accessSpecifier());
    if (!best || combined < *best) best = combined;
  }

  memo[index].second = best;
  return best;
}

auto AccessContext::isProtectedMemberAccessible(Symbol* member,
                                                ClassSymbol* designatingClass,
                                                ClassSymbol* objectClass) const
    -> bool {
  for (auto grantingClass : classes()) {
    if (grantingClass != designatingClass) {
      AccessMemo memo;
      if (!accessAsMemberOf(member, grantingClass, memo)) continue;
    }

    if (satisfiesProtectedObjectRestriction(member, grantingClass, objectClass))
      return true;
  }

  return false;
}

auto AccessContext::isAccessibleWhenDesignatedIn(
    Symbol* member, ClassSymbol* designatingClass, ClassSymbol* objectClass,
    std::vector<ClassSymbol*>& visited) const -> bool {
  designatingClass = normalize(designatingClass);
  if (!designatingClass) return true;
  if (std::ranges::contains(visited, designatingClass)) return false;
  visited.push_back(designatingClass);
  if (designatingClass->isAccessControlDisabled()) return true;

  AccessMemo memo;
  auto access = accessAsMemberOf(member, designatingClass, memo);

  if (access == AccessSpecifier::kPublic) return true;

  if (access == AccessSpecifier::kPrivate) {
    if (isMemberOrFriendOf(designatingClass)) return true;
  }

  if (access == AccessSpecifier::kProtected) {
    if (isProtectedMemberAccessible(member, designatingClass, objectClass))
      return true;
  }

  if (usingDeclarationIntroducing(member, designatingClass)) return false;

  for (auto baseClass : designatingClass->baseClasses()) {
    auto base = normalize(symbol_cast<ClassSymbol>(baseClass->symbol()));
    if (!base) continue;
    if (!isAccessibleBaseClassEdge(designatingClass, baseClass)) continue;
    if (isAccessibleWhenDesignatedIn(member, base, objectClass, visited))
      return true;
  }

  return false;
}

auto AccessContext::isAccessible(Symbol* member, ClassSymbol* designatingClass,
                                 ClassSymbol* objectClass) const -> bool {
  if (!member) return true;

  if (symbol_cast<OverloadSetSymbol>(member)) {
    return std::ranges::any_of(
        views::each_function(member), [&](FunctionSymbol* function) {
          return isAccessible(function, designatingClass, objectClass);
        });
  }

  auto declaringClass = declaringClassOf(member);
  if (!declaringClass) return true;

  if (!designatingClass) designatingClass = declaringClass;

  std::vector<ClassSymbol*> visited;
  return isAccessibleWhenDesignatedIn(member, designatingClass, objectClass,
                                      visited);
}

}  // namespace cxx
