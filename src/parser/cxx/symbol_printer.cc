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

// cxx
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <iostream>
#include <ranges>
#include <unordered_set>

namespace cxx {

namespace {

struct GetEnumeratorValue {
  auto operator()(bool value) const -> std::string {
    return value ? "true" : "false";
  }
  auto operator()(std::intmax_t value) const -> std::string {
    return std::to_string(value);
  }

  auto operator()(auto x) const -> std::string { return {}; }
};

struct DumpSymbols {
  std::ostream& out;
  int depth = 0;
  std::unordered_set<Symbol*> visited;

  auto constValueToString(const ConstValue& value) const -> std::string {
    if (auto n = std::get_if<std::intmax_t>(&value)) {
      return std::to_string(*n);
    }
    if (auto n = std::get_if<float>(&value)) {
      return std::format("{}", *n);
    }
    if (auto n = std::get_if<double>(&value)) {
      return std::format("{}", *n);
    }
    if (auto n = std::get_if<long double>(&value)) {
      return std::format("{}", *n);
    }
    if (std::holds_alternative<const StringLiteral*>(value)) {
      return "\"...\"";
    }
    if (std::holds_alternative<std::shared_ptr<Meta>>(value)) {
      return "<meta>";
    }
    if (std::holds_alternative<std::shared_ptr<InitializerList>>(value)) {
      return "<init-list>";
    }
    if (std::holds_alternative<std::shared_ptr<ConstObject>>(value)) {
      return "<const-object>";
    }
    return "<const>";
  }

  auto templateArgumentToString(const TemplateArgument& arg) const
      -> std::string {
    if (auto typeArg = std::get_if<const Type*>(&arg)) {
      return to_string(*typeArg);
    }

    if (auto symbolArg = std::get_if<Symbol*>(&arg)) {
      auto sym = *symbolArg;
      if (!sym) return "<null-symbol>";
      if (sym->isTypeAlias()) return to_string(sym->type());
      if (auto var = symbol_cast<VariableSymbol>(sym)) {
        if (auto cst = var->constValue()) return constValueToString(*cst);
      }
      if (sym->type()) return to_string(sym->type());
      return to_string(sym->name());
    }

    if (auto constArg = std::get_if<ConstValue>(&arg)) {
      return constValueToString(*constArg);
    }

    if (std::get_if<ExpressionAST*>(&arg)) {
      return "<expr>";
    }

    return "<arg>";
  }

  auto dumpScope(ScopeSymbol* scope) {
    if (!scope) return;

    ++depth;

    auto symbols = scope->members();

    std::vector<Symbol*> sortedSymbols(begin(symbols), end(symbols));

    std::ranges::for_each(sortedSymbols, [&](auto symbol) {
      // Skip non-canonical redeclarations
      if (symbol->canonical() != symbol) return;
      // Skip builtin function declarations
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->info() &&
          id->info()->kind() == IdentifierInfoKind::kBuiltinFunction) {
        return;
      }
      visit(*this, symbol);
    });

    --depth;
  }

  void dumpSpecializations(
      std::span<const TemplateSpecialization> specializations) {
    if (specializations.empty()) return;
    ++depth;
    indent();
    out << std::format("[specializations]\n");
    ++depth;
    for (auto specialization : specializations) {
      if (visited.insert(specialization.symbol).second) {
        visit(*this, specialization.symbol);
      }
    }
    depth -= 2;
  }

  template <typename T>
  void dumpRedeclarations(T* symbol) {
    auto& redecls = symbol->redeclarations();
    if (redecls.empty()) return;
    ++depth;
    indent();
    out << "[redeclarations]\n";
    ++depth;
    for (auto redecl : redecls) {
      visit(*this, redecl);
    }
    depth -= 2;
  }

  void indent() { out << std::format("{:{}}", "", depth * 2); }

  void operator()(NamespaceSymbol* symbol) {
    indent();
    out << "namespace";
    if (symbol->name()) out << std::format(" {}", to_string(symbol->name()));
    out << "\n";
    dumpScope(symbol);
  }

  void operator()(BaseClassSymbol* symbol) {
    indent();
    out << std::format("base class {}\n", to_string(symbol->name()));
  }

  void operator()(ClassSymbol* symbol) {
    indent();
    std::string_view classKey = symbol->isUnion() ? "union" : "class";

    if (symbol->templateParameters()) {
      out << std::format("template {} {}", classKey, to_string(symbol->name()));

      if (symbol->isSpecialization()) {
        out << '<';
        std::string_view sep = "";
        for (auto arg : symbol->templateArguments()) {
          out << std::format("{}{}", sep, templateArgumentToString(arg));
          sep = ", ";
        }
        out << '>';
      } else {
        out << '<';
        std::string_view sep = "";
        for (const auto& param : views::members(symbol->templateParameters())) {
          if (auto cstParam = symbol_cast<NonTypeParameterSymbol>(param)) {
            out << std::format("{}{}", sep, to_string(cstParam->objectType()));
          } else if (symbol_cast<TypeParameterSymbol>(param)) {
            out << std::format("{}{}", sep, to_string(param->type()));
          } else if (symbol_cast<TemplateTypeParameterSymbol>(param)) {
            out << std::format("{}{}", sep, to_string(param->type()));
          } else {
            out << std::format("{}{}", sep, to_string(param->type()));
          }
          if (param->isParameterPack()) out << "...";
          sep = ", ";
        }
        out << '>';
      }

      out << "\n";

      dumpScope(symbol->templateParameters());
    } else if (symbol->isSpecialization()) {
      out << std::format("{} {}", classKey, to_string(symbol->name()));
      out << "<";
      std::string_view sep = "";
      for (auto arg : symbol->templateArguments()) {
        out << std::format("{}{}", sep, templateArgumentToString(arg));
        sep = ", ";
      }
      out << std::format(">\n");
    } else {
      out << std::format("{} {}", classKey, to_string(symbol->name()));
      if (symbol->isFriend()) out << " friend";
      if (symbol->isHidden()) out << " hidden";
      out << "\n";
    }
    for (auto baseClass : symbol->baseClasses()) {
      ++depth;
      visit(*this, baseClass);
      --depth;
    }
    if (!symbol->constructors().empty()) {
      ++depth;
      for (auto constructor : symbol->constructors()) {
        // Skip non-canonical redeclarations
        if (constructor->canonical() != constructor) continue;
        visit(*this, constructor);
      }
      --depth;
    }
    dumpScope(symbol);
    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(ConceptSymbol* symbol) {
    indent();
    out << std::format("concept {}\n", to_string(symbol->name()));
    if (symbol->templateParameters()) dumpScope(symbol->templateParameters());
  }

  void operator()(EnumSymbol* symbol) {
    indent();
    out << std::format("enum {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out << std::format(" : {}", to_string(underlyingType));
    }

    out << std::format("\n");

    dumpScope(symbol);
  }

  void operator()(ScopedEnumSymbol* symbol) {
    indent();
    out << std::format("enum class {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out << std::format(" : {}", to_string(underlyingType));
    }

    out << std::format("\n");

    dumpScope(symbol);
  }

  void operator()(OverloadSetSymbol* symbol) {
    for (auto function : symbol->functions()) {
      // Skip non-canonical redeclarations
      if (function->canonical() != function) continue;
      visit(*this, function);
    }
  }

  void operator()(FunctionSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) {
      out << std::format("template ");
    }

    if (symbol->isConstructor()) {
      out << std::format("constructor");
    } else {
      out << std::format("function");
    }

    if (symbol->isStatic()) out << " static";
    if (symbol->isExtern()) out << " extern";
    if (symbol->isFriend()) out << " friend";
    if (symbol->isHidden()) out << " hidden";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConsteval()) out << " consteval";
    if (symbol->isInline()) out << " inline";
    if (symbol->isVirtual()) out << " virtual";
    if (symbol->isExplicit()) out << " explicit";
    if (symbol->isDeleted()) out << " deleted";
    if (symbol->isDefaulted()) out << " defaulted";
    if (symbol->hasCLinkage()) out << " extern \"C\"";

    out << std::format(" {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters());
    }

    dumpScope(symbol);
    dumpRedeclarations(symbol);
  }

  void operator()(LambdaSymbol* symbol) {
    indent();

    out << std::format("lambda");

    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConsteval()) out << " consteval";
    if (symbol->isMutable()) out << " mutable";
    if (symbol->isStatic()) out << " static";

    out << std::format("{}\n", to_string(symbol->type(), symbol->name()));

    dumpScope(symbol);
  }

  void operator()(TemplateParametersSymbol* symbol) {
    indent();
    out << std::format("template parameters\n");
    dumpScope(symbol);
  }

  void operator()(FunctionParametersSymbol* symbol) {
    indent();
    out << std::format("parameters\n");
    dumpScope(symbol);
  }

  void operator()(BlockSymbol* symbol) {
    indent();
    out << std::format("block\n");
    dumpScope(symbol);
  }

  void operator()(TypeAliasSymbol* symbol) {
    indent();
    if (symbol->templateParameters()) {
      out << std::format("template typealias {}\n",
                         to_string(symbol->type(), symbol->name()));
      dumpScope(symbol->templateParameters());
    } else {
      out << std::format("typealias {}\n",
                         to_string(symbol->type(), symbol->name()));
    }
    dumpRedeclarations(symbol);
  }

  void operator()(VariableSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) out << std::format("template ");

    out << std::format("variable");

    if (symbol->isStatic()) out << " static";
    if (symbol->isThreadLocal()) out << " thread_local";
    if (symbol->isExtern()) out << " extern";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConstinit()) out << " constinit";
    if (symbol->isInline()) out << " inline";

    out << std::format(" {}", to_string(symbol->type(), symbol->name()));

    if (!symbol->templateArguments().empty()) {
      out << "<";
      std::string_view sep = "";
      for (auto arg : symbol->templateArguments()) {
        out << std::format("{}{}", sep, templateArgumentToString(arg));
        sep = ", ";
      }
      out << std::format(">");
    }

    out << "\n";

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters());
    }

    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(FieldSymbol* symbol) {
    indent();

    if (symbol->isBitField())
      out << std::format("bitfield");
    else
      out << std::format("field");

    if (symbol->isStatic()) out << " static";
    if (symbol->isThreadLocal()) out << " thread_local";
    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConstinit()) out << " constinit";
    if (symbol->isInline()) out << " inline";

    out << std::format(" {}\n", to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterSymbol* symbol) {
    indent();
    out << std::format("parameter {}\n",
                       to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterPackSymbol* symbol) {
    indent();
    out << std::format("parameter pack {}\n",
                       to_string(symbol->type(), symbol->name()));
  }

  void operator()(TypeParameterSymbol* symbol) {
    auto type = type_cast<TypeParameterType>(symbol->type());
    std::string_view pack = type->isParameterPack() ? "..." : "";
    indent();
    out << std::format("parameter typename<{}, {}>{} {}\n", type->index(),
                       type->depth(), pack, to_string(symbol->name()));
  }

  void operator()(NonTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << std::format("parameter constant<{}, {}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), to_string(symbol->objectType()), pack,
                       to_string(symbol->name()));
  }

  void operator()(TemplateTypeParameterSymbol* symbol) {
    auto type = type_cast<TemplateTypeParameterType>(symbol->type());
    std::string_view pack = type->isParameterPack() ? "..." : "";
    indent();
    out << std::format("parameter template<{}, {}>{} {}\n", type->index(),
                       type->depth(), pack, to_string(symbol->name()));
  }

  void operator()(ConstraintTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out << std::format("parameter constraint<{}, {}>{} {}\n", symbol->index(),
                       symbol->depth(), pack, to_string(symbol->name()));
  }

  void operator()(EnumeratorSymbol* symbol) {
    indent();

    auto get_value = [](auto value) {
      return std::visit(GetEnumeratorValue{}, value);
    };

    const auto value = symbol->value().transform(get_value);

    out << std::format("enumerator {}",
                       to_string(symbol->type(), symbol->name()));

    if (value.has_value() && !value->empty()) {
      out << std::format(" = {}", *value);
    }

    out << "\n";
  }

  void operator()(UsingDeclarationSymbol* symbol) {
    indent();

    if (auto target = symbol->target()) {
      out << std::format("using {}\n",
                         to_string(target->type(), target->name()));
    } else {
      // unresolved symbol
      out << std::format("using unresolved {}\n", to_string(symbol->name()));
    }
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