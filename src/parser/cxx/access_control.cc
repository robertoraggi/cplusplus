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
#include <cxx/dependent_types.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <optional>
#include <span>

namespace cxx {

namespace {

struct DeclaredMember {
  ClassSymbol* declaringClass = nullptr;
  AccessSpecifier accessSpecifier = AccessSpecifier::kPublic;
};

enum class Derivation {
  kUnrelated,
  kDerived,
  kDependent,
};

[[nodiscard]] auto normalize(ClassSymbol* classSymbol) -> ClassSymbol* {
  if (!classSymbol) return nullptr;
  return classSymbol->resolvedDefinition();
}

[[nodiscard]] auto baseClassOf(BaseClassSymbol* baseClass) -> ClassSymbol* {
  return normalize(symbol_cast<ClassSymbol>(baseClass->symbol()));
}

[[nodiscard]] auto innermostClassOf(ScopeSymbol* scope) -> ClassSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) return classSymbol;
  return scope->enclosingClass();
}

void addUnique(std::vector<ClassSymbol*>& classes, ClassSymbol* classSymbol) {
  classSymbol = normalize(classSymbol);
  if (!classSymbol) return;
  if (std::ranges::contains(classes, classSymbol)) return;
  classes.push_back(classSymbol);
}

[[nodiscard]] auto eitherOf(AccessResult lhs, AccessResult rhs)
    -> AccessResult {
  if (lhs == AccessResult::kAccessible) return lhs;
  if (rhs == AccessResult::kAccessible) return rhs;
  if (lhs == AccessResult::kDependent) return lhs;
  return rhs;
}

[[nodiscard]] auto bothOf(AccessResult lhs, AccessResult rhs) -> AccessResult {
  if (lhs == AccessResult::kInaccessible) return lhs;
  if (rhs == AccessResult::kInaccessible) return rhs;
  if (lhs == AccessResult::kDependent) return lhs;
  return rhs;
}

[[nodiscard]] auto accessGrantedBy(Derivation derivation) -> AccessResult {
  switch (derivation) {
    case Derivation::kDerived:
      return AccessResult::kAccessible;
    case Derivation::kDependent:
      return AccessResult::kDependent;
    case Derivation::kUnrelated:
      return AccessResult::kInaccessible;
  }
  return AccessResult::kInaccessible;
}

[[nodiscard]] auto enclosingScopeOf(ClassSymbol* classSymbol) -> ScopeSymbol* {
  auto scope = classSymbol->parent();
  while (symbol_cast<TemplateParametersSymbol>(scope)) scope = scope->parent();
  return scope;
}

[[nodiscard]] auto mayShareEnclosingScope(ClassSymbol* lhs, ClassSymbol* rhs)
    -> bool {
  auto lhsScope = enclosingScopeOf(lhs);
  auto rhsScope = enclosingScopeOf(rhs);
  if (lhsScope == rhsScope) return true;
  if (symbol_cast<NamespaceSymbol>(lhsScope)) return false;
  if (symbol_cast<NamespaceSymbol>(rhsScope)) return false;
  return true;
}

[[nodiscard]] auto mightInstantiateTo(TranslationUnit* unit,
                                      ClassSymbol* pattern,
                                      ClassSymbol* classSymbol) -> bool {
  if (pattern == classSymbol) return false;
  if (pattern->name() != classSymbol->name()) return false;
  if (!isEnclosedInDependentTemplate(unit, pattern, true)) return false;
  return mayShareEnclosingScope(pattern, classSymbol);
}

[[nodiscard]] auto derivationOf(TranslationUnit* unit, ClassSymbol* derived,
                                ClassSymbol* base) -> Derivation {
  derived = normalize(derived);
  base = normalize(base);
  if (!derived || !base) return Derivation::kUnrelated;
  if (derived == base) return Derivation::kDerived;

  auto derivation = Derivation::kUnrelated;
  std::vector<ClassSymbol*> visited{derived};

  for (std::size_t index = 0; index < visited.size(); ++index) {
    auto current = visited[index];

    if (mightInstantiateTo(unit, current, base))
      derivation = Derivation::kDependent;

    for (auto baseClass : current->baseClasses()) {
      if (isDependentBaseClass(unit, current, baseClass))
        derivation = Derivation::kDependent;

      auto candidate = baseClassOf(baseClass);
      if (candidate == base) return Derivation::kDerived;
      if (!candidate) continue;
      if (std::ranges::contains(visited, candidate)) continue;
      visited.push_back(candidate);
    }
  }

  return derivation;
}

template <typename Key, typename Value>
[[nodiscard]] auto cachedValue(std::vector<std::pair<Key, Value>>& cache,
                               const Key& key) -> Value* {
  auto cached = std::ranges::find(cache, key, &std::pair<Key, Value>::first);
  if (cached == cache.end()) return nullptr;
  return &cached->second;
}

[[nodiscard]] auto templateArgumentsOf(auto* symbol)
    -> std::vector<TemplateArgument> {
  auto arguments = symbol->templateArguments();
  return std::vector<TemplateArgument>{arguments.begin(), arguments.end()};
}

[[nodiscard]] auto hasDependentArgument(
    TranslationUnit* unit, const std::vector<TemplateArgument>& arguments)
    -> bool {
  for (const auto& argument : arguments) {
    if (isDependentTemplateArgument(unit, argument)) return true;
  }
  return false;
}

[[nodiscard]] auto isFriendDeclaration(Symbol* symbol) -> bool {
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return function->isFriend();
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
    return classSymbol->isFriend();
  return false;
}

[[nodiscard]] auto introduces(UsingDeclarationSymbol* usingDeclaration,
                              Symbol* member) -> bool {
  if (auto function = symbol_cast<FunctionSymbol>(member)) {
    for (auto introduced : usingDeclaration->introducedFunctions()) {
      if (introduced->canonical() == function->canonical()) return true;
    }
    return false;
  }

  auto target = usingDeclaration->target();
  if (!target) return false;
  return target->canonical() == member->canonical();
}

[[nodiscard]] auto usingDeclarationIntroducing(Symbol* member,
                                               ClassSymbol* classSymbol)
    -> UsingDeclarationSymbol* {
  if (!member || !member->name()) return nullptr;
  if (symbol_cast<UsingDeclarationSymbol>(member)) return nullptr;

  for (auto candidate : classSymbol->find(member->name())) {
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
    auto fieldClass = unqualified_cast<ClassType>(field->type());
    if (!fieldClass) continue;
    if (normalize(fieldClass->symbol()) == normalize(classSymbol)) return true;
  }

  return false;
}

[[nodiscard]] auto isInjectedIntoEnclosingClass(Symbol* member,
                                                ClassSymbol* classSymbol)
    -> bool {
  if (!injectsMembersIntoEnclosingClass(classSymbol)) return false;
  auto field = symbol_cast<FieldSymbol>(member);
  return field && !field->isStatic();
}

[[nodiscard]] auto declaredMemberOf(Symbol* member) -> DeclaredMember {
  if (!member) return {};
  if (isFriendDeclaration(member)) return {};

  auto access = member->accessSpecifier();

  for (auto declaringScope = member->parent(); declaringScope;
       declaringScope = declaringScope->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(declaringScope)) {
      if (!isInjectedIntoEnclosingClass(member, classSymbol))
        return {normalize(classSymbol), access};
      access = std::max(access, classSymbol->accessSpecifier());
      continue;
    }

    if (!declaringScope->isEnum() && !declaringScope->isScopedEnum()) break;

    access = std::max(access, declaringScope->accessSpecifier());
  }

  return {};
}

[[nodiscard]] auto inheritedAccess(std::optional<AccessSpecifier> accessInBase,
                                   AccessSpecifier baseAccess)
    -> std::optional<AccessSpecifier> {
  if (!accessInBase) return std::nullopt;
  if (*accessInBase == AccessSpecifier::kPrivate) return std::nullopt;
  return std::max(*accessInBase, baseAccess);
}

class EffectiveAccess {
 public:
  EffectiveAccess(Symbol* member, DeclaredMember declared)
      : member_(member), declared_(declared) {}

  [[nodiscard]] auto asMemberOf(ClassSymbol* classSymbol)
      -> std::optional<AccessSpecifier> {
    if (auto cached = cachedValue(cache_, classSymbol)) return *cached;
    cache_.emplace_back(classSymbol, std::nullopt);
    auto access = computeAsMemberOf(classSymbol);
    *cachedValue(cache_, classSymbol) = access;
    return access;
  }

 private:
  [[nodiscard]] auto computeAsMemberOf(ClassSymbol* classSymbol)
      -> std::optional<AccessSpecifier> {
    if (classSymbol == declared_.declaringClass)
      return declared_.accessSpecifier;

    if (auto usingDeclaration =
            usingDeclarationIntroducing(member_, classSymbol))
      return usingDeclaration->accessSpecifier();

    std::optional<AccessSpecifier> best;

    for (auto baseClass : classSymbol->baseClasses()) {
      auto base = baseClassOf(baseClass);
      if (!base) continue;

      auto access =
          inheritedAccess(asMemberOf(base), baseClass->accessSpecifier());
      if (!access) continue;
      if (best && *best <= *access) continue;
      best = access;
    }

    return best;
  }

  Symbol* member_;
  DeclaredMember declared_;
  std::vector<std::pair<ClassSymbol*, std::optional<AccessSpecifier>>> cache_;
};

class FriendshipCollector {
 public:
  FriendshipCollector(TranslationUnit* unit,
                      std::vector<ClassSymbol*>& friendClasses)
      : unit_(unit), friendClasses_(friendClasses) {}

  void collect(ClassSymbol* seed) {
    auto arguments = templateArgumentsOf(seed);
    std::vector<ClassSymbol*> visited;
    std::vector<ClassSymbol*> pending{seed};

    while (!pending.empty()) {
      auto classSymbol = pending.back();
      pending.pop_back();
      if (!classSymbol) continue;
      if (std::ranges::contains(visited, classSymbol)) continue;
      visited.push_back(classSymbol);

      addFriendClasses(classSymbol->befriendingClasses());
      addTemplateFriendships(classSymbol->templateFriendships(), arguments);

      pending.push_back(classSymbol->canonical());
      pending.push_back(classSymbol->resolvedDefinition());
      pending.push_back(classSymbol->instantiationTemplate());
      pending.push_back(classSymbol->primaryTemplateSymbol());
    }
  }

  void collect(FunctionSymbol* seed) {
    auto arguments = templateArgumentsOf(seed);

    for (auto function = seed; function;
         function = function->canonical()->primaryTemplateSymbol()) {
      auto canonical = function->canonical();
      addFriendClasses(canonical->befriendingClasses());
      addTemplateFriendships(canonical->templateFriendships(), arguments);
    }
  }

 private:
  void addFriendClasses(const std::vector<ClassSymbol*>& befriendingClasses) {
    for (auto befriendingClass : befriendingClasses)
      addUnique(friendClasses_, befriendingClass);
  }

  void addTemplateFriendships(
      const std::vector<TemplateFriendship>& friendships,
      const std::vector<TemplateArgument>& arguments) {
    if (friendships.empty()) return;

    const auto dependent = hasDependentArgument(unit_, arguments);

    for (const auto& friendship : friendships) {
      if (!dependent && !compare_args(unit_, friendship.arguments, arguments))
        continue;
      addUnique(friendClasses_, friendship.befriendingClass);
    }
  }

  TranslationUnit* unit_;
  std::vector<ClassSymbol*>& friendClasses_;
};

}  // namespace

class AccessContext::Query {
 public:
  Query(const AccessContext& context, Symbol* member, DeclaredMember declared,
        ClassSymbol* objectClass)
      : context_(context),
        effectiveAccess_(member, declared),
        member_(member),
        declaringClass_(declared.declaringClass),
        objectClass_(normalize(objectClass)) {
    context_.materialize();
  }

  [[nodiscard]] auto designatedIn(ClassSymbol* designatingClass)
      -> AccessResult {
    designatingClass = normalize(designatingClass);
    if (!designatingClass) return AccessResult::kAccessible;
    restrictsObject_ = restrictsObjectIn(designatingClass);
    return accessibleIn(designatingClass);
  }

 private:
  [[nodiscard]] auto restrictsObjectIn(ClassSymbol* designatingClass) -> bool {
    if (!objectClass_) return false;
    if (!member_) return true;
    if (!is_non_static_member(member_)) return false;
    auto access = effectiveAccess_.asMemberOf(designatingClass);
    return access == AccessSpecifier::kProtected;
  }

  [[nodiscard]] auto accessibleIn(ClassSymbol* namingClass) -> AccessResult {
    if (auto cached = cachedValue(accessible_, namingClass)) return *cached;
    accessible_.emplace_back(namingClass, AccessResult::kInaccessible);
    auto result = computeAccessibleIn(namingClass);
    *cachedValue(accessible_, namingClass) = result;
    return result;
  }

  [[nodiscard]] auto computeAccessibleIn(ClassSymbol* namingClass)
      -> AccessResult {
    if (namingClass->isAccessControlDisabled())
      return AccessResult::kAccessible;

    auto result = AccessResult::kInaccessible;

    if (auto access = effectiveAccess_.asMemberOf(namingClass))
      result = grants(namingClass, *access);

    if (result == AccessResult::kAccessible) return result;
    if (usingDeclarationIntroducing(member_, namingClass)) return result;

    for (auto baseClass : namingClass->baseClasses()) {
      auto base = baseClassOf(baseClass);
      if (!base) continue;
      if (isInventedMemberOf(base)) continue;

      auto inBase = accessibleIn(base);
      if (inBase == AccessResult::kInaccessible) continue;

      auto baseAccess =
          context_.baseClassAccess(namingClass, base, restrictedObjectClass());
      result = eitherOf(result, bothOf(baseAccess, inBase));
      if (result == AccessResult::kAccessible) return result;
    }

    if (result != AccessResult::kInaccessible) return result;
    if (!hasDependentBaseClass(context_.unit_, namingClass)) return result;
    return AccessResult::kDependent;
  }

  [[nodiscard]] auto isInventedMemberOf(ClassSymbol* classSymbol) const
      -> bool {
    if (member_) return false;
    return classSymbol == declaringClass_;
  }

  [[nodiscard]] auto grants(ClassSymbol* namingClass, AccessSpecifier access)
      -> AccessResult {
    if (access == AccessSpecifier::kPublic) return AccessResult::kAccessible;
    if (context_.isMemberOf(namingClass)) return AccessResult::kAccessible;
    if (context_.isFriendOf(namingClass)) return AccessResult::kAccessible;

    auto result = membershipAfterInstantiation(namingClass);
    if (access == AccessSpecifier::kPrivate) return result;

    result = eitherOf(result, grantsThroughDerivedClasses(
                                  context_.memberClasses_, namingClass));
    if (!friendsOfDerivedClassesGrantAccess()) return result;

    return eitherOf(result, grantsThroughDerivedClasses(context_.friendClasses_,
                                                        namingClass));
  }

  [[nodiscard]] auto restrictedObjectClass() const -> ClassSymbol* {
    if (!restrictsObject_) return nullptr;
    return objectClass_;
  }

  [[nodiscard]] auto membershipAfterInstantiation(ClassSymbol* namingClass)
      -> AccessResult {
    for (auto memberClass : context_.memberClasses_) {
      if (mightInstantiateTo(context_.unit_, memberClass, namingClass))
        return AccessResult::kDependent;
    }
    return AccessResult::kInaccessible;
  }

  [[nodiscard]] auto friendsOfDerivedClassesGrantAccess() const -> bool {
    if (!member_) return true;
    return restrictsObject_;
  }

  [[nodiscard]] auto grantsThroughDerivedClasses(
      std::span<ClassSymbol* const> grantingClasses, ClassSymbol* namingClass)
      -> AccessResult {
    auto result = AccessResult::kInaccessible;

    for (auto grantingClass : grantingClasses) {
      auto derivation =
          derivationOf(context_.unit_, grantingClass, namingClass);
      auto granted = bothOf(accessGrantedBy(derivation),
                            satisfiesObjectRestriction(grantingClass));
      result = eitherOf(result, granted);
      if (result == AccessResult::kAccessible) break;
    }

    return result;
  }

  [[nodiscard]] auto satisfiesObjectRestriction(ClassSymbol* grantingClass)
      -> AccessResult {
    if (!restrictsObject_) return AccessResult::kAccessible;
    return accessGrantedBy(
        derivationOf(context_.unit_, objectClass_, grantingClass));
  }

  const AccessContext& context_;
  EffectiveAccess effectiveAccess_;
  Symbol* member_;
  ClassSymbol* declaringClass_;
  ClassSymbol* objectClass_;
  bool restrictsObject_ = false;
  std::vector<std::pair<ClassSymbol*, AccessResult>> accessible_;
};

auto declaringClassOf(Symbol* member) -> ClassSymbol* {
  return declaredMemberOf(member).declaringClass;
}

auto implicitObjectClassOf(TranslationUnit* unit, Symbol* member,
                           ScopeSymbol* accessingScope) -> ClassSymbol* {
  auto declaringClass = declaringClassOf(member);
  if (!declaringClass) return nullptr;
  if (!accessingScope) return nullptr;

  for (auto classSymbol = innermostClassOf(accessingScope); classSymbol;
       classSymbol = classSymbol->enclosingClass()) {
    auto derivation = derivationOf(unit, classSymbol, declaringClass);
    if (derivation == Derivation::kDerived) return normalize(classSymbol);
  }

  return nullptr;
}

AccessContext::AccessContext(TranslationUnit* unit, ScopeSymbol* accessingScope)
    : unit_(unit), accessingScope_(accessingScope) {}

void AccessContext::materialize() const {
  if (materialized_) return;
  materialized_ = true;

  FriendshipCollector friendships{unit_, friendClasses_};

  for (auto scope = accessingScope_; scope; scope = scope->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
      addUnique(memberClasses_, classSymbol);
      friendships.collect(classSymbol);
      continue;
    }

    if (auto function = symbol_cast<FunctionSymbol>(scope))
      friendships.collect(function);
  }
}

auto AccessContext::isMemberOf(ClassSymbol* classSymbol) const -> bool {
  materialize();
  return std::ranges::contains(memberClasses_, normalize(classSymbol));
}

auto AccessContext::isFriendOf(ClassSymbol* classSymbol) const -> bool {
  materialize();
  return std::ranges::contains(friendClasses_, normalize(classSymbol));
}

auto AccessContext::baseClassAccess(ClassSymbol* derived, ClassSymbol* base,
                                    ClassSymbol* objectClass) const
    -> AccessResult {
  const auto key = BaseClassAccessKey{derived, base, objectClass};
  if (auto cached = cachedValue(baseClassAccess_, key)) return *cached;

  Query query{*this, nullptr, DeclaredMember{base, AccessSpecifier::kPublic},
              objectClass};
  auto result = query.designatedIn(derived);
  baseClassAccess_.emplace_back(key, result);
  return result;
}

auto AccessContext::checkAccess(Symbol* member, ClassSymbol* designatingClass,
                                ClassSymbol* objectClass) const
    -> AccessResult {
  if (!member) return AccessResult::kAccessible;

  if (symbol_cast<OverloadSetSymbol>(member)) {
    auto result = AccessResult::kInaccessible;
    for (auto function : views::each_function(member)) {
      result = eitherOf(result,
                        checkAccess(function, designatingClass, objectClass));
      if (result == AccessResult::kAccessible) break;
    }
    return result;
  }

  auto declared = declaredMemberOf(member);
  if (!declared.declaringClass) return AccessResult::kAccessible;

  if (!designatingClass) designatingClass = declared.declaringClass;

  Query query{*this, member, declared, objectClass};
  return query.designatedIn(designatingClass);
}

auto AccessContext::isAccessible(Symbol* member, ClassSymbol* designatingClass,
                                 ClassSymbol* objectClass) const -> bool {
  auto result = checkAccess(member, designatingClass, objectClass);
  return result != AccessResult::kInaccessible;
}

auto AccessContext::isAccessibleBaseClass(ClassSymbol* derived,
                                          ClassSymbol* base) const -> bool {
  derived = normalize(derived);
  base = normalize(base);
  if (!derived || !base) return false;
  if (derived == base) return true;
  return baseClassAccess(derived, base, nullptr) != AccessResult::kInaccessible;
}

namespace {

[[nodiscard]] auto deniedMemberOf(Symbol* member, ClassSymbol* designatingClass)
    -> DeclaredMember {
  auto declared = declaredMemberOf(member);
  designatingClass = normalize(designatingClass);
  if (!designatingClass) return declared;

  EffectiveAccess effectiveAccess{member, declared};
  auto access = effectiveAccess.asMemberOf(designatingClass)
                    .value_or(AccessSpecifier::kPrivate);

  if (usingDeclarationIntroducing(member, designatingClass))
    return {designatingClass, access};

  return {declared.declaringClass, access};
}

[[nodiscard]] auto accessKindOf(AccessSpecifier access) -> std::string_view {
  if (access == AccessSpecifier::kProtected) return "protected";
  return "private";
}

}  // namespace

auto checkMemberAccess(TranslationUnit* unit, ScopeSymbol* accessingScope,
                       Symbol* member, ClassSymbol* designatingClass,
                       ClassSymbol* objectClass, SourceLocation loc) -> bool {
  if (!unit->config().checkTypes) return true;

  AccessContext accessContext{unit, accessingScope};
  if (accessContext.isAccessible(member, designatingClass, objectClass))
    return true;

  auto denied = deniedMemberOf(member, designatingClass);
  if (!denied.declaringClass) return true;

  unit->error(
      loc, std::format("'{}' is a {} member of '{}'", to_string(member->name()),
                       accessKindOf(denied.accessSpecifier),
                       to_string(denied.declaringClass->type())));

  return false;
}

}  // namespace cxx
