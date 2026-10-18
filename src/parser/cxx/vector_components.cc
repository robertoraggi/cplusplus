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
#include <cxx/names.h>
#include <cxx/types.h>
#include <cxx/vector_components.h>

#include <array>
#include <cctype>
#include <optional>

namespace cxx {

namespace {

constexpr std::string_view kPositions = "xyzw";
constexpr std::string_view kColors = "rgba";
constexpr std::string_view kHexDigits = "0123456789abcdef";

struct HalvingSelection {
  std::string_view name;
  std::int64_t scale;
  std::int64_t start;
};

[[nodiscard]] auto halvingSelection(std::string_view name, std::int64_t half)
    -> std::optional<HalvingSelection> {
  const std::array selections{
      HalvingSelection{"lo", 1, 0},
      HalvingSelection{"hi", 1, half},
      HalvingSelection{"even", 2, 0},
      HalvingSelection{"odd", 2, 1},
  };
  for (const auto& selection : selections)
    if (selection.name == name) return selection;
  return std::nullopt;
}

[[nodiscard]] auto alphabetOf(std::string_view& spelling) -> std::string_view {
  if (spelling.size() > 1 && (spelling[0] == 's' || spelling[0] == 'S')) {
    spelling.remove_prefix(1);
    return kHexDigits;
  }
  if (kPositions.contains(spelling[0])) return kPositions;
  if (kColors.contains(spelling[0])) return kColors;
  return kHexDigits;
}

[[nodiscard]] auto laneOf(std::string_view alphabet, char c)
    -> std::optional<std::int64_t> {
  if (alphabet == kHexDigits)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  auto position = alphabet.find(c);
  if (position == std::string_view::npos) return std::nullopt;
  return static_cast<std::int64_t>(position);
}

}  // namespace

auto vectorComponents(const VectorType* type, std::string_view name)
    -> std::expected<VectorComponents, VectorComponentError> {
  const auto elementCount = static_cast<std::int64_t>(type->elementCount());
  const auto half = (elementCount + 1) / 2;

  VectorComponents components;
  if (auto selection = halvingSelection(name, half)) {
    for (std::int64_t index = 0; index < half; ++index)
      components.lanes.push_back(selection->start + selection->scale * index);
    return components;
  }

  auto spelling = name;
  const auto alphabet = alphabetOf(spelling);
  for (auto c : spelling) {
    auto lane = laneOf(alphabet, c);
    if (!lane)
      return std::unexpected(
          VectorComponentError{VectorComponentErrorKind::kIllegalName, c});
    components.lanes.push_back(*lane);
  }

  for (std::size_t index = 0; index < spelling.size(); ++index) {
    if (components.lanes[index] >= elementCount)
      return std::unexpected(VectorComponentError{
          VectorComponentErrorKind::kOutOfRange, spelling[index]});
  }

  std::array<bool, 16> seen{};
  for (auto lane : components.lanes) {
    if (seen[lane]) components.hasRepeatedLane = true;
    seen[lane] = true;
  }

  return components;
}

auto vectorComponentAccess(MemberExpressionAST* ast)
    -> std::optional<VectorComponentAccess> {
  if (!ast || ast->symbol || !ast->baseExpression) return std::nullopt;
  auto name = ast_cast<NameIdAST>(ast->unqualifiedId);
  if (!name || !name->identifier) return std::nullopt;
  auto objectType = ast->baseExpression->type;
  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointer = unqualified_cast<PointerType>(objectType);
    objectType = pointer ? pointer->elementType() : nullptr;
  }
  auto vector = unqualified_cast<VectorType>(objectType);
  if (!vector || vector->vectorKind() != VectorKind::kExt) return std::nullopt;
  auto components = vectorComponents(vector, name->identifier->name());
  if (!components) return std::nullopt;
  return VectorComponentAccess{vector, std::move(*components)};
}

}  // namespace cxx
