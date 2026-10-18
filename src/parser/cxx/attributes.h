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
#include <cxx/attribute_kind.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>

#include <compare>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace cxx {

class TranslationUnit;
class MemoryLayout;

[[nodiscard]] auto canonicalAttributeSpelling(std::string_view name)
    -> std::string_view;

[[nodiscard]] auto supportsGnuAttribute(std::string_view name,
                                        const MemoryLayout& layout) -> bool;

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

struct KernelWorkgroupSize {
  std::uint32_t x = 1;
  std::uint32_t y = 1;
  std::uint32_t z = 1;

  [[nodiscard]] auto operator==(const KernelWorkgroupSize&) const
      -> bool = default;
};

struct KernelBufferBinding {
  std::uint32_t binding = 0;

  [[nodiscard]] auto operator==(const KernelBufferBinding&) const
      -> bool = default;
};

enum class KernelInvocationIdKind { kGlobal, kLocal, kWorkgroup };

struct KernelInvocationId {
  KernelInvocationIdKind kind = KernelInvocationIdKind::kGlobal;

  [[nodiscard]] auto operator==(const KernelInvocationId&) const
      -> bool = default;
};

using KernelParameterBinding =
    std::variant<KernelBufferBinding, KernelInvocationId>;

[[nodiscard]] auto collectAttributes(TranslationUnit* unit,
                                     List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap;

[[nodiscard]] auto collectGnuAttributes(
    TranslationUnit* unit, List<AttributeSpecifierAST*>* attributes)
    -> AttributeMap;

[[nodiscard]] auto mergeAttributes(AttributeMap lhs, const AttributeMap* rhs)
    -> AttributeMap;

enum class AttributeSyntax { kGnu, kCxx, kAsmLabel };

enum class AttributeAvailability { kAlways, kWebAssembly, kDeviceProfile };

struct AttributeSpelling {
  AttributeSyntax syntax;
  std::string_view attributeNamespace;
  std::string_view name;
  AttributeKind kind;
};

struct AttributeInfo {
  AttributeKind kind;
  std::string_view name;
  std::span<const AttributeSpelling> spellings;
  AttributeAvailability availability;
  bool cplusplusOnly;
  std::span<const std::string_view> arguments;
  bool optionalArguments;
  std::string_view subjects;
  std::string_view documentation;
};

[[nodiscard]] auto attributeInfos() -> std::span<const AttributeInfo>;

[[nodiscard]] auto attributeInfo(AttributeKind kind) -> const AttributeInfo&;

[[nodiscard]] auto findAttributeSpelling(AttributeSyntax syntax,
                                         std::string_view attributeNamespace,
                                         std::string_view name)
    -> const AttributeSpelling*;

[[nodiscard]] auto isAttributeAvailable(const AttributeInfo& info,
                                        const MemoryLayout& layout) -> bool;

[[nodiscard]] auto namesAttribute(const Attribute& attribute,
                                  AttributeKind kind) -> bool;

[[nodiscard]] auto findAttribute(const AttributeMap* attributes,
                                 AttributeKind kind) -> const Attribute*;

[[nodiscard]] auto attributeArgument(const AttributeMap* attributes,
                                     AttributeKind kind) -> const Identifier*;

struct AttributeRef {
  const AttributeSpelling* spelling = nullptr;
  AttributeArgumentClauseAST* argumentClause = nullptr;
  SourceLocation location;

  [[nodiscard]] auto kind() const -> AttributeKind { return spelling->kind; }

  explicit operator bool() const { return spelling != nullptr; }
};

[[nodiscard]] auto knownAttributes(TranslationUnit* unit,
                                   List<AttributeSpecifierAST*>* attributeList)
    -> std::vector<AttributeRef>;

[[nodiscard]] auto findAttributes(TranslationUnit* unit,
                                  List<AttributeSpecifierAST*>* attributeList,
                                  AttributeKind kind)
    -> std::vector<AttributeRef>;

[[nodiscard]] auto findAttribute(TranslationUnit* unit,
                                 List<AttributeSpecifierAST*>* attributeList,
                                 AttributeKind kind) -> AttributeRef;

}  // namespace cxx
