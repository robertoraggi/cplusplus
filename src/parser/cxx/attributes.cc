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
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>

#include <algorithm>

namespace cxx {

namespace {

[[nodiscard]] auto spellingOf(const Identifier* id) -> std::string_view {
  return id ? std::string_view{id->name()} : std::string_view{};
}

}  // namespace

auto Attribute::operator<=>(const Attribute& other) const
    -> std::strong_ordering {
  if (auto cmp = spellingOf(attributeNamespace) <=>
                 spellingOf(other.attributeNamespace);
      cmp != 0)
    return cmp;

  if (auto cmp = spellingOf(name) <=> spellingOf(other.name); cmp != 0)
    return cmp;

  if (auto cmp = arguments.size() <=> other.arguments.size(); cmp != 0)
    return cmp;

  for (std::size_t i = 0; i < arguments.size(); ++i) {
    if (auto cmp = spellingOf(arguments[i]) <=> spellingOf(other.arguments[i]);
        cmp != 0)
      return cmp;
  }

  return std::strong_ordering::equal;
}

namespace {

[[nodiscard]] auto canonicalAttributeName(Control* control,
                                          const Identifier* id)
    -> const Identifier* {
  if (!id) return nullptr;
  const auto spelling = id->name();
  if (spelling.size() > 4 && spelling.starts_with("__") &&
      spelling.ends_with("__")) {
    return control->getIdentifier(
        std::string(spelling.substr(2, spelling.size() - 4)));
  }
  return id;
}

void collectStringArguments(TranslationUnit* unit,
                            AttributeArgumentClauseAST* clause,
                            std::vector<const Identifier*>& arguments) {
  for (auto expression : ListView{clause->expressionList}) {
    auto stringLiteral = ast_cast<StringLiteralExpressionAST>(expression);
    if (!stringLiteral || !stringLiteral->literal) continue;

    auto components = StringLiteral::Components::from(
        stringLiteral->literal->value(), StringLiteralEncoding::kNone);

    arguments.push_back(unit->control()->getIdentifier(components.value));
  }
}

[[nodiscard]] auto isStandardOrVendorNamespace(const Identifier* id) -> bool {
  if (!id) return true;
  const auto spelling = id->name();
  return spelling == "gnu" || spelling == "clang" || spelling == "__gnu__" ||
         spelling == "__clang__";
}

void addAttribute(AttributeMap& map, Attribute attribute) {
  if (!attribute.name) return;

  auto it = std::ranges::find_if(map, [&](const Attribute& entry) {
    return entry.name == attribute.name &&
           entry.attributeNamespace == attribute.attributeNamespace;
  });

  if (it == map.end()) {
    map.push_back(std::move(attribute));
    return;
  }

  if (it->arguments.empty()) it->arguments = std::move(attribute.arguments);
}

}  // namespace

auto collectAttributes(TranslationUnit* unit,
                       List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap {
  AttributeMap map;
  auto control = unit->control();

  for (auto specifier : ListView{attributes}) {
    if (specifier->attributes) {
      for (const auto& attribute : *specifier->attributes)
        addAttribute(map, attribute);
      continue;
    }

    List<AttributeAST*>* entries = nullptr;
    const Identifier* usingNamespace = nullptr;

    if (auto cxxAttribute = ast_cast<CxxAttributeAST>(specifier)) {
      entries = cxxAttribute->attributeList;
      if (auto prefix = cxxAttribute->attributeUsingPrefix) {
        usingNamespace = canonicalAttributeName(
            control, unit->identifier(prefix->attributeNamespaceLoc));
      }
    } else if (auto gccAttribute = ast_cast<GccAttributeAST>(specifier)) {
      entries = gccAttribute->attributeList;
    } else {
      continue;
    }

    AttributeMap collected;

    for (auto entry : ListView{entries}) {
      Attribute attribute;
      attribute.attributeNamespace = usingNamespace;

      if (auto simple =
              ast_cast<SimpleAttributeTokenAST>(entry->attributeToken)) {
        attribute.name = canonicalAttributeName(control, simple->identifier);
      } else if (auto scoped =
                     ast_cast<ScopedAttributeTokenAST>(entry->attributeToken)) {
        attribute.name = canonicalAttributeName(control, scoped->identifier);
        attribute.attributeNamespace =
            canonicalAttributeName(control, scoped->attributeNamespace);
      }

      if (auto clause = entry->attributeArgumentClause) {
        collectStringArguments(unit, clause, attribute.arguments);
      }

      addAttribute(collected, std::move(attribute));
    }

    std::ranges::sort(collected);

    specifier->attributes = control->getAttributes(collected);

    for (const auto& attribute : collected) addAttribute(map, attribute);
  }

  std::ranges::sort(map);

  return map;
}

auto mergeAttributes(AttributeMap lhs, const AttributeMap* rhs)
    -> AttributeMap {
  if (!rhs) return lhs;

  for (const auto& attribute : *rhs) addAttribute(lhs, attribute);

  std::ranges::sort(lhs);

  return lhs;
}

auto findAttribute(const AttributeMap* attributes, std::string_view name)
    -> const Attribute* {
  if (!attributes) return nullptr;

  auto it = std::ranges::find_if(*attributes, [&](const Attribute& attribute) {
    return attribute.name && attribute.name->name() == name &&
           isStandardOrVendorNamespace(attribute.attributeNamespace);
  });

  return it == attributes->end() ? nullptr : &*it;
}

auto attributeArgument(const AttributeMap* attributes, std::string_view name)
    -> const Identifier* {
  auto attribute = findAttribute(attributes, name);
  if (!attribute || attribute->arguments.empty()) return nullptr;
  return attribute->arguments.front();
}

namespace {

[[nodiscard]] auto matchesSpelling(const AttributeSpelling& spelling,
                                   AttributeSyntax syntax,
                                   const Identifier* attributeNamespace,
                                   const Identifier* name) -> bool {
  if (spelling.syntax != syntax) return false;
  if (spelling.name != spellingOf(name)) return false;
  return spelling.attributeNamespace == spellingOf(attributeNamespace);
}

}  // namespace

auto findAttributeBySpelling(TranslationUnit* unit,
                             List<AttributeSpecifierAST*>* attributeList,
                             std::span<const AttributeSpelling> spellings)
    -> AttributeRef {
  auto control = unit->control();

  for (auto specifier : ListView{attributeList}) {
    List<AttributeAST*>* entries = nullptr;
    auto syntax = AttributeSyntax::kCxx;
    const Identifier* usingNamespace = nullptr;

    if (auto cxxAttribute = ast_cast<CxxAttributeAST>(specifier)) {
      entries = cxxAttribute->attributeList;
      if (auto prefix = cxxAttribute->attributeUsingPrefix) {
        usingNamespace = canonicalAttributeName(
            control, unit->identifier(prefix->attributeNamespaceLoc));
      }
    } else if (auto gccAttribute = ast_cast<GccAttributeAST>(specifier)) {
      entries = gccAttribute->attributeList;
      syntax = AttributeSyntax::kGnu;
    } else {
      continue;
    }

    for (auto entry : ListView{entries}) {
      const Identifier* attributeNamespace = usingNamespace;
      const Identifier* name = nullptr;

      if (auto simple =
              ast_cast<SimpleAttributeTokenAST>(entry->attributeToken)) {
        name = canonicalAttributeName(control, simple->identifier);
      } else if (auto scoped =
                     ast_cast<ScopedAttributeTokenAST>(entry->attributeToken)) {
        name = canonicalAttributeName(control, scoped->identifier);
        attributeNamespace =
            canonicalAttributeName(control, scoped->attributeNamespace);
      }

      for (const auto& spelling : spellings) {
        if (!matchesSpelling(spelling, syntax, attributeNamespace, name))
          continue;

        return AttributeRef{&spelling, entry->attributeArgumentClause,
                            entry->firstSourceLocation()};
      }
    }
  }

  return {};
}

}  // namespace cxx
