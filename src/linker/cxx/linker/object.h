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

#include <cxx/linker/bytes.h>
#include <cxx/linker/wasm_format.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace cxx::linker {

struct Symbol;
class Object;

struct Relocation {
  std::uint8_t type = 0;
  std::uint32_t offset = 0;
  std::uint32_t index = 0;
  std::int32_t addend = 0;
};

enum class ChunkKind : std::uint8_t { Function, Segment, Section };

struct Chunk {
  Object* object = nullptr;
  ChunkKind kind = ChunkKind::Function;
  std::string name;
  ByteView content;
  std::vector<Relocation> relocations;
  std::uint32_t typeIndex = 0;
  std::uint32_t alignment = 0;
  std::uint32_t flags = 0;
  std::uint32_t index = 0;
  std::uint32_t offset = 0;
  bool live = false;
  bool discarded = false;
};

struct Import {
  std::string module;
  std::string field;
  std::uint32_t typeIndex = 0;
  std::uint8_t valueType = 0;
  bool isMutable = false;
};

struct GlobalDefinition {
  std::uint8_t valueType = kI32;
  bool isMutable = false;
  std::uint8_t initOpcode = kOpI32Const;
  std::uint64_t init = 0;
};

struct SymbolInfo {
  std::uint8_t kind = 0;
  std::uint32_t flags = 0;
  std::uint32_t index = 0;
  std::uint32_t offset = 0;
  std::string name;
};

struct InitFunction {
  std::uint32_t priority = 0;
  std::uint32_t symbol = 0;
};

class Object {
 public:
  Object(std::string name, ByteView content);
  Object(const Object&) = delete;
  auto operator=(const Object&) -> Object& = delete;
  ~Object();

  [[nodiscard]] auto parse(std::string& error) -> bool;

  [[nodiscard]] auto functionImportCount() const -> std::size_t {
    return imports[kExternalFunction].size();
  }

  std::string name;
  ByteView content;
  std::vector<FuncType> types;
  std::array<std::vector<Import>, 5> imports;
  std::vector<std::uint32_t> functionTypes;
  std::vector<GlobalDefinition> globals;
  std::vector<std::uint32_t> tags;
  std::vector<SymbolInfo> symbolInfos;
  std::vector<InitFunction> initFunctions;
  std::vector<Chunk> functions;
  std::vector<Chunk> segments;
  std::vector<Chunk> sections;
  std::vector<Symbol*> symbols;
  std::vector<std::unique_ptr<Symbol>> localSymbols;
  std::vector<std::uint32_t> typeMap;
};

}  // namespace cxx::linker
