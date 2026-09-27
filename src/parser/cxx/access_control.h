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
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>

#include <tuple>
#include <utility>
#include <vector>

namespace cxx {

enum class AccessResult {
  kInaccessible,
  kAccessible,
  kDependent,
};

[[nodiscard]] auto declaringClassOf(Symbol* member) -> ClassSymbol*;

[[nodiscard]] auto implicitObjectClassOf(TranslationUnit* unit, Symbol* member,
                                         ScopeSymbol* accessingScope)
    -> ClassSymbol*;

[[nodiscard]] auto checkMemberAccess(TranslationUnit* unit,
                                     ScopeSymbol* accessingScope,
                                     Symbol* member,
                                     ClassSymbol* designatingClass,
                                     ClassSymbol* objectClass,
                                     SourceLocation loc) -> bool;

class AccessContext {
 public:
  AccessContext(TranslationUnit* unit, ScopeSymbol* accessingScope);

  AccessContext(const AccessContext&) = delete;
  auto operator=(const AccessContext&) -> AccessContext& = delete;

  [[nodiscard]] auto isAccessible(Symbol* member, ClassSymbol* designatingClass,
                                  ClassSymbol* objectClass) const -> bool;

  [[nodiscard]] auto isAccessibleBaseClass(ClassSymbol* derived,
                                           ClassSymbol* base) const -> bool;

 private:
  class Query;

  using BaseClassAccessKey =
      std::tuple<ClassSymbol*, ClassSymbol*, ClassSymbol*>;

  [[nodiscard]] auto checkAccess(Symbol* member, ClassSymbol* designatingClass,
                                 ClassSymbol* objectClass) const
      -> AccessResult;

  void materialize() const;

  [[nodiscard]] auto isMemberOf(ClassSymbol* classSymbol) const -> bool;
  [[nodiscard]] auto isFriendOf(ClassSymbol* classSymbol) const -> bool;

  [[nodiscard]] auto baseClassAccess(ClassSymbol* derived, ClassSymbol* base,
                                     ClassSymbol* objectClass) const
      -> AccessResult;

  TranslationUnit* unit_;
  ScopeSymbol* accessingScope_;
  mutable std::vector<ClassSymbol*> memberClasses_;
  mutable std::vector<ClassSymbol*> friendClasses_;
  mutable std::vector<std::pair<BaseClassAccessKey, AccessResult>>
      baseClassAccess_;
  mutable bool materialized_ = false;
};

}  // namespace cxx
