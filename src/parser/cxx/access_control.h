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

#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace cxx {

struct DeclaredMember {
  ClassSymbol* declaringClass = nullptr;
  AccessSpecifier accessSpecifier = AccessSpecifier::kPublic;
};

[[nodiscard]] auto declaredMemberOf(Symbol* member) -> DeclaredMember;

[[nodiscard]] auto declaringClassOf(Symbol* member) -> ClassSymbol*;

[[nodiscard]] auto designatingClassOf(Symbol* member,
                                      ScopeSymbol* accessingScope)
    -> ClassSymbol*;

[[nodiscard]] auto isProtectedAccessRestricted(Symbol* member) -> bool;

[[nodiscard]] auto usingDeclarationIntroducing(Symbol* member,
                                               ClassSymbol* designatingClass)
    -> UsingDeclarationSymbol*;

class AccessContext {
 public:
  AccessContext(TranslationUnit* unit, ScopeSymbol* accessingScope);

  AccessContext(const AccessContext&) = delete;
  auto operator=(const AccessContext&) -> AccessContext& = delete;

  [[nodiscard]] auto accessingScope() const -> ScopeSymbol* {
    return accessingScope_;
  }

  [[nodiscard]] auto isAccessible(Symbol* member, ClassSymbol* designatingClass,
                                  ClassSymbol* objectClass) const -> bool;

  [[nodiscard]] auto isAccessibleBaseClass(ClassSymbol* derived,
                                           ClassSymbol* base) const -> bool;

  [[nodiscard]] auto classes() const -> std::span<ClassSymbol* const>;

 private:
  void materialize() const;

  [[nodiscard]] auto isMemberOrFriendOf(ClassSymbol* classSymbol) const -> bool;
  [[nodiscard]] auto isMemberOf(ClassSymbol* classSymbol) const -> bool;

  [[nodiscard]] auto isAccessibleBaseClassEdge(ClassSymbol* derived,
                                               BaseClassSymbol* baseClass) const
      -> bool;

  using AccessMemo =
      std::vector<std::pair<ClassSymbol*, std::optional<AccessSpecifier>>>;

  [[nodiscard]] auto accessAsMemberOf(Symbol* member,
                                      ClassSymbol* designatingClass,
                                      AccessMemo& memo) const
      -> std::optional<AccessSpecifier>;

  [[nodiscard]] auto isAccessibleWhenDesignatedIn(
      Symbol* member, ClassSymbol* designatingClass, ClassSymbol* objectClass,
      std::vector<ClassSymbol*>& visited) const -> bool;

  [[nodiscard]] auto isProtectedMemberAccessible(Symbol* member,
                                                 ClassSymbol* designatingClass,
                                                 ClassSymbol* objectClass) const
      -> bool;

  [[nodiscard]] auto satisfiesProtectedObjectRestriction(
      Symbol* member, ClassSymbol* grantingClass,
      ClassSymbol* objectClass) const -> bool;

  TranslationUnit* unit_;
  ScopeSymbol* accessingScope_;
  mutable std::vector<ClassSymbol*> classes_;
  mutable std::vector<ClassSymbol*> enclosingClasses_;
  mutable bool materialized_ = false;
};

}  // namespace cxx
