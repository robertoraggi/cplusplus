// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <sstream>
#include <string_view>

namespace cxx {

inline auto dump_symbol(Symbol* symbol) -> std::string {
  std::ostringstream out;
  out << symbol;
  return out.str();
}

struct Source {
  std::unique_ptr<MemoryLayout> memoryLayout;
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};

  explicit Source(std::string_view source, bool templateInstantiation = true) {
    // default to wasm32 memory layout
    memoryLayout = std::make_unique<MemoryLayout>(32);

    unit.control()->setMemoryLayout(memoryLayout.get());

    unit.setSource(std::string(source), "<test>");

    unit.parse({
        .checkTypes = true,
        .fuzzyTemplateResolution = false,
        .templateInstantiation = templateInstantiation,
        .reflect = true,
    });
  }

  auto control() -> Control* { return unit.control(); }
  auto ast() -> UnitAST* { return unit.ast(); }
  auto scope() -> ScopeSymbol* { return unit.globalScope(); }

  auto get(std::string_view name) -> Symbol* {
    Symbol* symbol = nullptr;
    auto id = unit.control()->getIdentifier(name);
    for (auto candidate : scope()->find(id)) {
      if (symbol) return nullptr;
      symbol = candidate;
    }
    return symbol;
  }

  template <typename S>
  auto getAs(std::string_view name) -> S* {
    auto symbol = get(name);
    return symbol_cast<S>(symbol);
  }
};

inline auto operator""_cxx(const char* source, std::size_t size) -> Source {
  return Source{std::string_view{source, size}};
}

inline auto operator""_cxx_no_templates(const char* source, std::size_t size)
    -> Source {
  // disable templates to allow overriding the template instantiation algorithm
  bool templateInstantiation = false;
  auto text = std::string_view{source, size};
  return Source{text, templateInstantiation};
}

struct LookupMember {
  Source& source;

  auto operator()(ScopeSymbol* scope, std::string_view name) -> Symbol* {
    auto id = source.control()->getIdentifier(name);
    for (auto candidate : scope->find(id)) {
      return candidate;
    }
    return nullptr;
  }
};

}  // namespace cxx