// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/symbol_printer.h>

// cxx
#include <cxx/name_printer.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>

#include <algorithm>
#include <ranges>

namespace cxx {

namespace {

struct DumpSymbols {
  std::ostream& out;
  int depth = 0;

  auto dumpScope(Scope* scope) {
    if (!scope) return;
    ++depth;
    auto symbols = scope->symbols();
    std::vector<Symbol*> sortedSymbols(begin(symbols), end(symbols));
    std::ranges::sort(sortedSymbols,
                      [](auto a, auto b) { return a->index() < b->index(); });
    std::ranges::for_each(sortedSymbols,
                          [&](auto symbol) { visit(*this, symbol); });
    --depth;
  }

  void operator()(NamespaceSymbol* symbol) {
    fmt::print(out, "{:{}}namespace {}\n", "", depth * 2,
               to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(ClassSymbol* symbol) {
    fmt::print(out, "{:{}}class {}\n", "", depth * 2,
               to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(UnionSymbol* symbol) {
    fmt::print(out, "{:{}}union {}\n", "", depth * 2,
               to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(EnumSymbol* symbol) {
    fmt::print(out, "{:{}}enum {}\n", "", depth * 2, to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(ScopedEnumSymbol* symbol) {
    fmt::print(out, "{:{}}enum class {}\n", "", depth * 2,
               to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(FunctionSymbol* symbol) {
    fmt::print(out, "{:{}}function {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(LambdaSymbol* symbol) {
    fmt::print(out, "{:{}}lambda {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(PrototypeSymbol* symbol) {
    fmt::print(out, "{:{}}prototype\n", "", depth * 2);
    dumpScope(symbol->scope());
  }

  void operator()(BlockSymbol* symbol) {
    fmt::print(out, "{:{}}block\n", "", depth * 2);
    dumpScope(symbol->scope());
  }

  void operator()(TypeAliasSymbol* symbol) {
    fmt::print(out, "{:{}}typealias {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
  }

  void operator()(VariableSymbol* symbol) {
    fmt::print(out, "{:{}}variable {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
  }

  void operator()(FieldSymbol* symbol) {
    fmt::print(out, "{:{}}field {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterSymbol* symbol) {
    fmt::print(out, "{:{}}parameter {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
  }

  void operator()(EnumeratorSymbol* symbol) {
    fmt::print(out, "{:{}}enumerator {}\n", "", depth * 2,
               to_string(symbol->type(), symbol->name()));
  }
};

}  // namespace

void dump(std::ostream& out, Symbol* symbol, int depth) {
  visit(DumpSymbols{out, depth}, symbol);
}

auto operator<<(std::ostream& out, Symbol* symbol) -> std::ostream& {
  dump(out, symbol);
  return out;
}

}  // namespace cxx