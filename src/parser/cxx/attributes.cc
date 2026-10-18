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
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/private/attributes-priv.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <ranges>
#include <tuple>

namespace cxx {

auto canonicalAttributeSpelling(std::string_view name) -> std::string_view {
  if (name.size() > 4 && name.starts_with("__") && name.ends_with("__"))
    return name.substr(2, name.size() - 4);
  return name;
}

auto attributeInfos() -> std::span<const AttributeInfo> {
  return attribute_table::kAttributeInfos;
}

auto attributeInfo(AttributeKind kind) -> const AttributeInfo& {
  return attribute_table::kAttributeInfos[static_cast<std::size_t>(kind)];
}

namespace {

[[nodiscard]] constexpr auto spellingKey(const AttributeSpelling& spelling) {
  return std::tuple{spelling.syntax, spelling.attributeNamespace,
                    spelling.name};
}

static_assert(std::ranges::is_sorted(attribute_table::kSortedAttributeSpellings,
                                     {}, [](const AttributeSpelling* spelling) {
                                       return spellingKey(*spelling);
                                     }));

static_assert(std::ranges::all_of(
    std::views::iota(std::size_t{0},
                     std::size(attribute_table::kAttributeInfos)),
    [](std::size_t index) {
      return static_cast<std::size_t>(
                 attribute_table::kAttributeInfos[index].kind) == index;
    }));

}  // namespace

auto findAttributeSpelling(AttributeSyntax syntax,
                           std::string_view attributeNamespace,
                           std::string_view name) -> const AttributeSpelling* {
  const auto key = std::tuple{syntax, attributeNamespace, name};
  const auto& spellings = attribute_table::kSortedAttributeSpellings;
  auto it = std::ranges::lower_bound(
      spellings, key, {},
      [](const AttributeSpelling* spelling) { return spellingKey(*spelling); });
  if (it == std::end(spellings) || spellingKey(**it) != key) return nullptr;
  return *it;
}

auto isAttributeAvailable(const AttributeInfo& info, const MemoryLayout& layout)
    -> bool {
  switch (info.availability) {
    case AttributeAvailability::kAlways:
      return true;
    case AttributeAvailability::kWebAssembly:
      return layout.isWebAssembly();
    case AttributeAvailability::kDeviceProfile:
      return layout.deviceProfile() != DeviceProfile::kNone;
  }
  return false;
}

auto supportsGnuAttribute(std::string_view name, const MemoryLayout& layout)
    -> bool {
  auto spelling = findAttributeSpelling(AttributeSyntax::kGnu, "",
                                        canonicalAttributeSpelling(name));
  return spelling &&
         isAttributeAvailable(attributeInfo(spelling->kind), layout);
}

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
  return control->getIdentifier(
      std::string(canonicalAttributeSpelling(id->name())));
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

[[nodiscard]] auto specifierAttributes(TranslationUnit* unit,
                                       AttributeSpecifierAST* specifier)
    -> const AttributeMap* {
  if (specifier->attributes) return specifier->attributes;

  auto control = unit->control();

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
  } else if (auto assembly = ast_cast<AsmAttributeAST>(specifier)) {
    if (!assembly->literal) return nullptr;
    auto components = StringLiteral::Components::from(
        assembly->literal->value(), StringLiteralEncoding::kNone);
    Attribute attribute{
        .name = control->getIdentifier("asm"),
        .arguments = {control->getIdentifier(components.value)}};
    specifier->attributes = control->getAttributes({attribute});
    return specifier->attributes;
  } else {
    return nullptr;
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
  return specifier->attributes;
}

void addSpecifierAttributes(TranslationUnit* unit, AttributeMap& map,
                            AttributeSpecifierAST* specifier) {
  auto attributes = specifierAttributes(unit, specifier);
  if (!attributes) return;
  for (const auto& attribute : *attributes) addAttribute(map, attribute);
}

}  // namespace

auto collectAttributes(TranslationUnit* unit,
                       List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap {
  AttributeMap map;
  for (auto specifier : ListView{attributes})
    addSpecifierAttributes(unit, map, specifier);
  std::ranges::sort(map);
  return map;
}

auto collectGnuAttributes(TranslationUnit* unit,
                          List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap {
  AttributeMap map;
  for (auto specifier : ListView{attributes}) {
    if (!ast_cast<GccAttributeAST>(specifier) &&
        !ast_cast<AsmAttributeAST>(specifier))
      continue;
    addSpecifierAttributes(unit, map, specifier);
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

auto namesAttribute(const Attribute& attribute, AttributeKind kind) -> bool {
  if (!attribute.name) return false;
  return std::ranges::any_of(
      attributeInfo(kind).spellings, [&](const AttributeSpelling& spelling) {
        return spelling.name == attribute.name->name() &&
               spelling.attributeNamespace ==
                   spellingOf(attribute.attributeNamespace);
      });
}

auto findAttribute(const AttributeMap* attributes, AttributeKind kind)
    -> const Attribute* {
  if (!attributes) return nullptr;

  auto it = std::ranges::find_if(*attributes, [&](const Attribute& attribute) {
    return namesAttribute(attribute, kind);
  });

  return it == attributes->end() ? nullptr : &*it;
}

auto attributeArgument(const AttributeMap* attributes, AttributeKind kind)
    -> const Identifier* {
  auto attribute = findAttribute(attributes, kind);
  if (!attribute || attribute->arguments.empty()) return nullptr;
  return attribute->arguments.front();
}

auto knownAttributes(TranslationUnit* unit,
                     List<AttributeSpecifierAST*>* attributeList)
    -> std::vector<AttributeRef> {
  auto control = unit->control();
  std::vector<AttributeRef> result;

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

      auto spelling = findAttributeSpelling(
          syntax, spellingOf(attributeNamespace), spellingOf(name));
      if (!spelling) continue;

      result.push_back({spelling, entry->attributeArgumentClause,
                        entry->firstSourceLocation()});
    }
  }

  return result;
}

auto findAttributes(TranslationUnit* unit,
                    List<AttributeSpecifierAST*>* attributeList,
                    AttributeKind kind) -> std::vector<AttributeRef> {
  auto attributes = knownAttributes(unit, attributeList);
  std::erase_if(attributes, [kind](const AttributeRef& attribute) {
    return attribute.kind() != kind;
  });
  return attributes;
}

auto findAttribute(TranslationUnit* unit,
                   List<AttributeSpecifierAST*>* attributeList,
                   AttributeKind kind) -> AttributeRef {
  for (auto attribute : knownAttributes(unit, attributeList))
    if (attribute.kind() == kind) return attribute;
  return {};
}

}  // namespace cxx
