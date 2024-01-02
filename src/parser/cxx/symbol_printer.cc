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
#include <iostream>
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

  void indent() { out << cxx::format("{:{}}", "", depth * 2); }

  void operator()(NamespaceSymbol* symbol) {
    indent();
    out << cxx::format("namespace {}\n", to_string(symbol->name()));
    dumpScope(symbol->scope());
  }

  void operator()(BaseClassSymbol* symbol) {
    indent();
    out << cxx::format("base class {}\n", to_string(symbol->name()));
  }

  void operator()(ClassSymbol* symbol) {
    indent();
    std::string_view classKey = symbol->isUnion() ? "union" : "class";

    if (symbol->templateParameters()) {
      out << cxx::format("template {} {}\n", classKey,
                         to_string(symbol->name()));
      dumpScope(symbol->templateParameters()->scope());
    } else {
      out << cxx::format("{} {}\n", classKey, to_string(symbol->name()));
    }
    dumpScope(symbol->scope());
  }

  void operator()(ConceptSymbol* symbol) {
    indent();
    out << cxx::format("concept {}\n", to_string(symbol->name()));
    if (symbol->templateParameters())
      dumpScope(symbol->templateParameters()->scope());
  }

  void operator()(EnumSymbol* symbol) {
    indent();
    out << cxx::format("enum {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out << cxx::format(" : {}", to_string(underlyingType));
    }

    out << cxx::format("\n");

    dumpScope(symbol->scope());
  }

  void operator()(ScopedEnumSymbol* symbol) {
    indent();
    out << cxx::format("enum class {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out << cxx::format(" : {}", to_string(underlyingType));
    }

    out << cxx::format("\n");

    dumpScope(symbol->scope());
  }

  void operator()(OverloadSetSymbol* symbol) {
    for (auto function : symbol->functions()) {
      visit(*this, function);
    }
  }

  void operator()(FunctionSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) {
      out << cxx::format("template ");
    }

    out << cxx::format("function");

    if (symbol->isStatic()) out << " static";
    if (symbol->isExtern()) out << " extern";
    if (symbol->isFriend()) out << " friend";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConsteval()) out << " consteval";
    if (symbol->isInline()) out << " inline";
    if (symbol->isVirtual()) out << " virtual";
    if (symbol->isExplicit()) out << " explicit";
    if (symbol->isDeleted()) out << " deleted";
    if (symbol->isDefaulted()) out << " defaulted";

    out << cxx::format(" {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters()->scope());
    }

    dumpScope(symbol->scope());
  }

  void operator()(LambdaSymbol* symbol) {
    indent();

    out << cxx::format("lambda");

    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConsteval()) out << " consteval";
    if (symbol->isMutable()) out << " mutable";
    if (symbol->isStatic()) out << " static";

    out << cxx::format("{}\n", to_string(symbol->type(), symbol->name()));

    dumpScope(symbol->scope());
  }

  void operator()(TemplateParametersSymbol* symbol) {
    indent();
    out << cxx::format("template parameters\n");
    dumpScope(symbol->scope());
  }

  void operator()(FunctionParametersSymbol* symbol) {
    indent();
    out << cxx::format("parameters\n");
    dumpScope(symbol->scope());
  }

  void operator()(BlockSymbol* symbol) {
    indent();
    out << cxx::format("block\n");
    dumpScope(symbol->scope());
  }

  void operator()(TypeAliasSymbol* symbol) {
    indent();
    if (symbol->templateParameters()) {
      out << cxx::format("template typealias {}\n",
                         to_string(symbol->type(), symbol->name()));
      dumpScope(symbol->templateParameters()->scope());
    } else {
      out << cxx::format("typealias {}\n",
                         to_string(symbol->type(), symbol->name()));
    }
  }

  void operator()(VariableSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) out << cxx::format("template ");

    out << cxx::format("variable");

    if (symbol->isStatic()) out << " static";
    if (symbol->isThreadLocal()) out << " thread_local";
    if (symbol->isExtern()) out << " extern";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConstinit()) out << " constinit";
    if (symbol->isInline()) out << " inline";

    out << cxx::format(" {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters()->scope());
    }
  }

  void operator()(FieldSymbol* symbol) {
    indent();

    out << cxx::format("field");

    if (symbol->isStatic()) out << " static";
    if (symbol->isThreadLocal()) out << " thread_local";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConstinit()) out << " constinit";
    if (symbol->isInline()) out << " inline";

    out << cxx::format(" {}\n", to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterSymbol* symbol) {
    indent();
    out << cxx::format("parameter {}\n",
                       to_string(symbol->type(), symbol->name()));
  }

  void operator()(TypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << cxx::format("parameter typename<{}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(NonTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << cxx::format("parameter object<{}, {}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), to_string(symbol->objectType()), pack,
                       to_string(symbol->name()));
  }

  void operator()(TemplateTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << cxx::format("parameter template<{}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(ConstraintTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << cxx::format("parameter constraint<{}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(EnumeratorSymbol* symbol) {
    indent();
    out << cxx::format("enumerator {}\n",
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