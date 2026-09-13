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
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>

#include <compare>
#include <span>
#include <string_view>
#include <vector>

namespace cxx {

class TranslationUnit;

struct Attribute {
  const Identifier* attributeNamespace = nullptr;
  const Identifier* name = nullptr;
  std::vector<const Identifier*> arguments;

  /**
   * Orders by spelling. Identifiers are interned, so equality is a pointer
   * comparison, but ordering by the pointers themselves would order by
   * allocation address and differ from one run to the next.
   */
  [[nodiscard]] auto operator<=>(const Attribute& other) const
      -> std::strong_ordering;

  auto operator==(const Attribute&) const -> bool = default;
};

using AttributeMap = std::vector<Attribute>;

[[nodiscard]] auto collectAttributes(TranslationUnit* unit,
                                     List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap;

[[nodiscard]] auto mergeAttributes(AttributeMap lhs, const AttributeMap* rhs)
    -> AttributeMap;

[[nodiscard]] auto findAttribute(const AttributeMap* attributes,
                                 std::string_view name) -> const Attribute*;

[[nodiscard]] auto attributeArgument(const AttributeMap* attributes,
                                     std::string_view name)
    -> const Identifier*;

enum class AttributeSyntax { kGnu, kCxx };

struct AttributeSpelling {
  AttributeSyntax syntax;
  std::string_view attributeNamespace;
  std::string_view name;
};

struct AttributeRef {
  const AttributeSpelling* spelling = nullptr;
  AttributeArgumentClauseAST* argumentClause = nullptr;
  SourceLocation location;

  explicit operator bool() const { return spelling != nullptr; }
};

/**
 * Returns the first attribute of the list matching one of the given spellings,
 * in source order, together with the spelling it matched.
 */
[[nodiscard]] auto findAttributeBySpelling(
    TranslationUnit* unit, List<AttributeSpecifierAST*>* attributeList,
    std::span<const AttributeSpelling> spellings) -> AttributeRef;

}  // namespace cxx
