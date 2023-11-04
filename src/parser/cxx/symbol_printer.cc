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

    std::ranges::sort(sortedSymbols, [](auto a, auto b) {
      return a->insertionPoint() < b->insertionPoint();
    });

    std::ranges::for_each(sortedSymbols,
                          [&](auto symbol) { visit(*this, symbol); });

    --depth;
  }

  void indent() { fmt::print(out, "{:{}}", "", depth * 2); }

  void operator()(NamespaceSymbol* symbol) {
    indent();
    fmt::print(out, "namespace {}\n", to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(ClassSymbol* symbol) {
    indent();
    if (symbol->templateParameters()) {
      fmt::print(out, "template class {}\n", to_string(symbol->name()));
      dumpScope(symbol->templateParameters()->scope());
    } else {
      fmt::print(out, "class {}\n", to_string(symbol->name()));
    }
    dumpScope(symbol->scope());
  }

  void operator()(ConceptSymbol* symbol) {
    indent();
    fmt::print(out, "concept {}\n", to_string(symbol->name()));
    if (symbol->templateParameters())
      dumpScope(symbol->templateParameters()->scope());
  }

  void operator()(UnionSymbol* symbol) {
    indent();
    fmt::print(out, "union {}\n", to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(EnumSymbol* symbol) {
    indent();
    fmt::print(out, "enum {}", "", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      fmt::print(out, " : {}", to_string(underlyingType));
    }

    fmt::print(out, "\n");

    dumpScope(symbol->scope());
  }

  void operator()(ScopedEnumSymbol* symbol) {
    indent();
    fmt::print(out, "enum class {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      fmt::print(out, " : {}", to_string(underlyingType));
    }

    fmt::print(out, "\n");

    dumpScope(symbol->scope());
  }

  void operator()(FunctionSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) {
      fmt::print(out, "template ");
    }

    fmt::print(out, "function");

    if (symbol->isStatic()) fmt::print(" static");
    if (symbol->isExtern()) fmt::print(" extern");
    if (symbol->isFriend()) fmt::print(" friend");
    if (symbol->isConstexpr()) fmt::print(" constexpr");
    if (symbol->isConsteval()) fmt::print(" consteval");
    if (symbol->isInline()) fmt::print(" inline");
    if (symbol->isVirtual()) fmt::print(" virtual");
    if (symbol->isExplicit()) fmt::print(" explicit");
    if (symbol->isDeleted()) fmt::print(" deleted");
    if (symbol->isDefaulted()) fmt::print(" defaulted");

    fmt::print(out, " {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters()->scope());
    }

    dumpScope(symbol->scope());
  }

  void operator()(LambdaSymbol* symbol) {
    indent();

    fmt::print(out, "lambda");

    if (symbol->isConstexpr()) fmt::print(" constexpr");
    if (symbol->isConsteval()) fmt::print(" consteval");
    if (symbol->isMutable()) fmt::print(" mutable");
    if (symbol->isStatic()) fmt::print(" static");

    fmt::print(out, "{}\n", to_string(symbol->type(), symbol->name()));

    dumpScope(symbol->scope());
  }

  void operator()(TemplateParametersSymbol* symbol) {
    indent();
    fmt::print(out, "template parameters\n");
    dumpScope(symbol->scope());
  }

  void operator()(FunctionParametersSymbol* symbol) {
    indent();
    fmt::print(out, "parameters\n");
    dumpScope(symbol->scope());
  }

  void operator()(BlockSymbol* symbol) {
    indent();
    fmt::print(out, "block\n");
    dumpScope(symbol->scope());
  }

  void operator()(TypeAliasSymbol* symbol) {
    indent();
    if (symbol->templateParameters()) {
      fmt::print(out, "template typealias {}\n",
                 to_string(symbol->type(), symbol->name()));
      dumpScope(symbol->templateParameters()->scope());
    } else {
      fmt::print(out, "typealias {}\n",
                 to_string(symbol->type(), symbol->name()));
    }
  }

  void operator()(VariableSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) fmt::print(out, "template ");

    fmt::print(out, "variable");

    if (symbol->isStatic()) fmt::print(" static");
    if (symbol->isThreadLocal()) fmt::print(" thread_local");
    if (symbol->isExtern()) fmt::print(" extern");
    if (symbol->isConstexpr()) fmt::print(" constexpr");
    if (symbol->isConstinit()) fmt::print(" constinit");
    if (symbol->isInline()) fmt::print(" inline");

    fmt::print(" {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters()->scope());
    }
  }

  void operator()(FieldSymbol* symbol) {
    indent();

    fmt::print(out, "field");

    if (symbol->isStatic()) fmt::print(" static");
    if (symbol->isThreadLocal()) fmt::print(" thread_local");
    if (symbol->isConstexpr()) fmt::print(" constexpr");
    if (symbol->isConstinit()) fmt::print(" constinit");
    if (symbol->isInline()) fmt::print(" inline");

    fmt::print(" {}\n", to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterSymbol* symbol) {
    indent();
    fmt::print(out, "parameter {}\n",
               to_string(symbol->type(), symbol->name()));
  }

  void operator()(TypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    fmt::print(out, "parameter typename<{}, {}>{} {}\n", symbol->index(),
               symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(NonTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    fmt::print(out, "parameter object<{}, {}, {}>{} {}\n", symbol->index(),
               symbol->depth(), to_string(symbol->objectType()), pack,
               to_string(symbol->name()));
  }

  void operator()(TemplateTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    fmt::print(out, "parameter template<{}, {}>{} {}\n", symbol->index(),
               symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(ConstraintTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    fmt::print(out, "parameter constraint<{}, {}>{} {}\n", symbol->index(),
               symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(EnumeratorSymbol* symbol) {
    indent();
    fmt::print(out, "enumerator {}\n",
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