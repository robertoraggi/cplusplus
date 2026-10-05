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

#include <cxx/linker/relocate.h>

namespace cxx::linker {

namespace {

enum class Form { Uleb, Sleb, Fixed };

constexpr std::uint32_t kFixedSize = 4;
constexpr std::uint32_t kRangesTombstone = 0xfffffffe;
constexpr std::uint32_t kDebugTombstone = 0xffffffff;

[[nodiscard]] auto formOf(std::uint8_t type, Form& form) -> bool {
  switch (type) {
    case kFunctionIndexLeb:
    case kMemoryAddrLeb:
    case kTypeIndexLeb:
    case kGlobalIndexLeb:
    case kTagIndexLeb:
    case kTableNumberLeb:
      form = Form::Uleb;
      return true;
    case kTableIndexSleb:
    case kMemoryAddrSleb:
    case kMemoryAddrRelSleb:
    case kTableIndexRelSleb:
      form = Form::Sleb;
      return true;
    case kTableIndexI32:
    case kMemoryAddrI32:
    case kFunctionOffsetI32:
    case kSectionOffsetI32:
    case kGlobalIndexI32:
    case kMemoryAddrLocrelI32:
    case kFunctionIndexI32:
      form = Form::Fixed;
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto tombstone(const Chunk& chunk) -> std::uint32_t {
  if (chunk.kind != ChunkKind::Section) return 0;
  if (chunk.name == ".debug_ranges" || chunk.name == ".debug_loc") {
    return kRangesTombstone;
  }
  return chunk.name.starts_with(".debug_") ? kDebugTombstone : 0;
}

[[nodiscard]] auto isResolved(const Symbol& symbol, std::uint8_t type) -> bool {
  if (type == kFunctionOffsetI32) {
    return symbol.defined && symbol.chunk && symbol.chunk->live;
  }
  if (type == kSectionOffsetI32) return symbol.chunk != nullptr;
  if (symbol.kind == SymbolKind::Section) return true;
  return symbol.chunk ? symbol.chunk->live : symbol.live;
}

[[nodiscard]] auto memoryAddress(const Symbol& symbol, std::int32_t addend)
    -> std::uint32_t {
  return symbol.defined ? symbol.address() + addend : 0;
}

[[nodiscard]] auto valueOf(Layout& layout, const Chunk& chunk,
                           const Relocation& relocation, const Symbol& symbol)
    -> std::uint32_t {
  const auto addend = relocation.addend;
  switch (relocation.type) {
    case kFunctionIndexLeb:
    case kFunctionIndexI32:
      return symbol.outputIndex();
    case kTableIndexSleb:
    case kTableIndexI32:
      return symbol.tableIndex.value_or(0);
    case kTableIndexRelSleb:
      return symbol.tableIndex.value_or(0) - layout.tableBase;
    case kMemoryAddrLeb:
    case kMemoryAddrSleb:
    case kMemoryAddrI32:
    case kMemoryAddrRelSleb:
      return memoryAddress(symbol, addend);
    case kMemoryAddrLocrelI32:
      return memoryAddress(symbol, addend) - (chunk.offset + relocation.offset);
    case kGlobalIndexLeb:
    case kGlobalIndexI32:
      return symbol.kind == SymbolKind::Global ? symbol.index
                                               : symbol.gotIndex.value_or(0);
    case kTagIndexLeb:
      return symbol.index;
    case kFunctionOffsetI32:
      return symbol.chunk->offset + addend;
    case kSectionOffsetI32:
      return symbol.chunk->offset + addend;
    default:
      return 0;
  }
}

[[nodiscard]] auto unresolvedValue(const Chunk& chunk,
                                   const Relocation& relocation)
    -> std::uint32_t {
  const auto value = tombstone(chunk);
  return value ? value : static_cast<std::uint32_t>(relocation.addend);
}

[[nodiscard]] auto symbolValue(Layout& layout, const Chunk& chunk,
                               const Relocation& relocation) -> std::uint32_t {
  if (relocation.type == kTypeIndexLeb) {
    return layout.typeIndex(*chunk.object, relocation.index);
  }
  const Symbol& symbol = *chunk.object->symbols[relocation.index];
  if (isResolved(symbol, relocation.type)) {
    return valueOf(layout, chunk, relocation, symbol);
  }
  return unresolvedValue(chunk, relocation);
}

}  // namespace

auto relocate(Layout& layout, const Chunk& chunk, std::uint8_t* output,
              std::string& error) -> bool {
  for (const auto& relocation : chunk.relocations) {
    Form form = Form::Fixed;
    if (!formOf(relocation.type, form)) {
      error = concat(chunk.object->name, ": unsupported relocation type ",
                     std::to_string(relocation.type));
      return false;
    }
    const auto width = form == Form::Fixed ? kFixedSize : kPaddedLebSize;
    if (relocation.offset + width > chunk.content.size()) {
      error = concat(chunk.object->name, ": relocation out of range");
      return false;
    }
    const auto value = symbolValue(layout, chunk, relocation);
    auto* at = output + relocation.offset;
    if (form == Form::Uleb) patchUleb(at, value);
    if (form == Form::Sleb) patchSleb(at, static_cast<std::int32_t>(value));
    if (form == Form::Fixed) patchU32(at, value);
  }
  return true;
}

}  // namespace cxx::linker
