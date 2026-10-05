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

#include <cxx/linker/object.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cxx::linker {

enum class SymbolKind : std::uint8_t {
  Function,
  Data,
  Global,
  Section,
  Tag,
  Table,
};

struct Symbol {
  std::string name;
  SymbolKind kind = SymbolKind::Function;
  bool defined = false;
  bool weak = false;
  bool imported = false;
  bool exported = false;
  bool noStrip = false;
  bool live = false;
  Object* object = nullptr;
  Chunk* chunk = nullptr;
  std::uint32_t offset = 0;
  std::string importModule;
  std::string importField;
  FuncType type;
  GlobalDefinition global;
  std::uint32_t index = 0;
  std::optional<std::uint32_t> tableIndex;
  std::optional<std::uint32_t> gotIndex;

  [[nodiscard]] auto outputIndex() const -> std::uint32_t {
    return chunk ? chunk->index : index;
  }

  [[nodiscard]] auto address() const -> std::uint32_t {
    return (chunk ? chunk->offset : 0) + offset;
  }
};

class SymbolTable {
 public:
  [[nodiscard]] auto find(std::string_view name) const -> Symbol*;

  [[nodiscard]] auto add(Object& object, std::string& error) -> bool;

  [[nodiscard]] auto defineSynthetic(std::string_view name, SymbolKind kind)
      -> Symbol*;

  [[nodiscard]] auto takePending() -> std::vector<Symbol*>;

  [[nodiscard]] auto symbols() const -> const std::vector<Symbol*>& {
    return order_;
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<Symbol>> table_;
  std::vector<Symbol*> order_;
  std::vector<Symbol*> pending_;

  [[nodiscard]] auto addGlobal(Object& object, const SymbolInfo& info,
                               std::string& error) -> Symbol*;
};

}  // namespace cxx::linker
