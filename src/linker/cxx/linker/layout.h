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
#include <cxx/linker/symbols.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace cxx::linker {

struct LayoutOptions {
  std::string entry = "_start";
  std::vector<std::string> exports;
  std::uint32_t globalBase = 0;
  std::uint32_t stackSize = kPageSize;
  std::uint32_t initialMemory = 0;
  std::uint32_t maxMemory = 0;
  bool stackFirst = true;
  bool gcSections = true;
  bool noEntry = false;
  bool stripDebug = false;
  bool stripAll = false;
  bool allowUndefined = false;
};

struct OutputSection {
  std::string name;
  std::vector<Chunk*> chunks;
  std::uint32_t size = 0;
};

struct Layout {
  Layout();

  [[nodiscard]] auto typeIndex(const FuncType& type) -> std::uint32_t;
  [[nodiscard]] auto typeIndex(Object& object, std::uint32_t index)
      -> std::uint32_t;

  Object synthetic;
  std::deque<Bytes> syntheticBodies;
  std::vector<FuncType> types;
  std::vector<Symbol*> functionImports;
  std::vector<Symbol*> globalImports;
  std::vector<Symbol*> tagImports;
  std::vector<Chunk*> functions;
  std::vector<Symbol*> globals;
  std::vector<Symbol*> gotEntries;
  std::vector<Symbol*> tags;
  std::vector<Symbol*> tableEntries;
  std::vector<Symbol*> exports;
  std::vector<Chunk*> dataChunks;
  std::vector<OutputSection> customSections;
  std::uint32_t tableBase = 1;
  std::uint32_t dataStart = 0;
  std::uint32_t memoryPages = 0;
  std::uint32_t maxMemoryPages = 0;
  bool hasTable = false;
  bool hasNames = true;
};

[[nodiscard]] auto buildLayout(SymbolTable& symbols,
                               std::vector<std::unique_ptr<Object>>& objects,
                               const LayoutOptions& options, Layout& layout,
                               std::string& error) -> bool;

}  // namespace cxx::linker
