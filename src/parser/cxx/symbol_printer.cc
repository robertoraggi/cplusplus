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

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
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

[[nodiscard]] auto templateParameterText(Symbol* parameter) -> std::string {
  auto nonTypeParameter = symbol_cast<NonTypeParameterSymbol>(parameter);
  if (!nonTypeParameter) return to_string(parameter->type());

  auto text = to_string(nonTypeParameter->objectType());
  if (nonTypeParameter->isParameterPack()) text += "...";
  return text;
}

struct DumpSymbols {
  std::ostream& out;
  int depth = 0;
  TranslationUnit* unit = nullptr;
  std::unordered_set<Symbol*> visited;

  [[nodiscard]] auto isPreambleSymbol(Symbol* symbol) const -> bool {
    if (!unit || !symbol) return false;
    auto loc = symbol->location();
    if (!loc) return false;
    return unit->tokenStartPosition(loc).fileName == "<builtins>";
  }

  auto dumpScope(ScopeSymbol* scope) {
    if (!scope) return;

    ++depth;

    auto symbols = scope->members();

    std::vector<Symbol*> sortedSymbols(begin(symbols), end(symbols));

    std::ranges::for_each(sortedSymbols, [&](auto symbol) {
      if (symbol->canonical() != symbol) return;
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->info() &&
          id->info()->kind() == IdentifierInfoKind::kBuiltinFunction) {
        return;
      }
      if (id && id->builtinTemplate() != BuiltinTemplateKind::T_NONE) return;
      if (isPreambleSymbol(symbol)) return;
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

  void operator()(NamespaceAliasSymbol* symbol) {
    indent();
    std::string aliasedName{"<unresolved>"};
    if (auto aliased = symbol->namespaceSymbol())
      aliasedName = to_string(aliased->name());
    out << std::format("namespace {} = {}\n", to_string(symbol->name()),
                       aliasedName);
  }

  void operator()(BaseClassSymbol* symbol) {
    indent();
    auto baseClass = symbol->symbol();
    if (baseClass && baseClass->type()) {
      out << std::format("base class {}\n", to_string(baseClass->type()));
      return;
    }
    out << std::format("base class {}\n", to_string(symbol->name()));
  }

  void operator()(InjectedClassNameSymbol* symbol) {
    indent();
    out << std::format("injected class name {}\n", to_string(symbol->name()));
  }

  void operator()(UnresolvedSymbol* symbol) {
    indent();
    out << std::format("unresolved {}\n", to_string(symbol->name()));
  }

  void operator()(ClassSymbol* symbol) {
    indent();
    std::string_view classKey = symbol->isUnion() ? "union" : "class";

    if (symbol->templateParameters()) {
      out << std::format("template {} {}", classKey, to_string(symbol->name()));

      if (symbol->isSpecialization()) {
        out << '<';
        std::string_view sep = "";
        for (const auto& arg :
             expand_template_arguments(symbol->templateArguments())) {
          out << std::format("{}{}", sep, to_string(arg));
          sep = ", ";
        }
        out << '>';
      } else {
        out << '<';
        std::string_view sep = "";
        for (const auto& param : views::members(symbol->templateParameters())) {
          out << std::format("{}{}", sep, templateParameterText(param));
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
      for (const auto& arg :
           expand_template_arguments(symbol->templateArguments())) {
        out << std::format("{}{}", sep, to_string(arg));
        sep = ", ";
      }
      out << std::format(">\n");
    } else {
      out << std::format("{} {}", classKey, to_string(symbol->name()));
      if (symbol->isFriend()) out << " friend";
      if (symbol->isHidden()) out << " hidden";
      if (symbol->isFinal()) out << " final";
      if (symbol->isPolymorphic()) out << " polymorphic";
      if (symbol->isAbstract()) out << " abstract";
      out << "\n";
    }
    for (auto baseClass : symbol->baseClasses()) {
      ++depth;
      visit(*this, baseClass);
      --depth;
    }
    if (!symbol->declaredConstructors().empty()) {
      ++depth;
      for (auto constructor : symbol->declaredConstructors()) {
        if (constructor->canonical() != constructor) continue;
        visit(*this, constructor);
      }
      --depth;
    }
    dumpScope(symbol);
    if (!symbol->deductionGuides().empty()) {
      ++depth;
      for (auto guide : symbol->deductionGuides()) {
        visit(*this, guide);
      }
      --depth;
    }
    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(ConceptSymbol* symbol) {
    indent();
    out << std::format("concept {}\n", to_string(symbol->name()));
    if (symbol->templateParameters()) dumpScope(symbol->templateParameters());
  }

  void operator()(DeductionGuideSymbol* symbol) {
    indent();
    if (auto funcType = type_cast<FunctionType>(symbol->type())) {
      std::string params;
      bool first = true;
      for (auto pt : funcType->parameterTypes()) {
        if (!first) params += ", ";
        params += to_string(pt);
        first = false;
      }
      out << std::format("deduction-guide {}({}) -> {}\n",
                         to_string(symbol->name()), params,
                         to_string(funcType->returnType()));
    } else {
      out << std::format("deduction-guide {}\n", to_string(symbol->name()));
    }
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
    for (auto usingDeclaration : symbol->usingDeclarations())
      visit(*this, usingDeclaration);

    for (auto function : symbol->declaredFunctions()) {
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
    if (symbol->isPure()) out << " pure";
    if (symbol->isOverride()) out << " override";
    if (symbol->isFinal()) out << " final";
    if (symbol->isExplicit()) out << " explicit";
    if (symbol->isDeleted()) out << " deleted";
    if (symbol->isDefaulted()) out << " defaulted";
    if (symbol->hasCLinkage()) out << " extern \"C\"";
    dumpAbiTags(symbol);

    out << std::format(" {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters());
    }

    dumpScope(symbol);
    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(LambdaSymbol* symbol) {
    indent();

    out << std::format("lambda");

    if (symbol->isConstexpr()) out << " constexpr";
    if (symbol->isConsteval()) out << " consteval";
    if (symbol->isMutable()) out << " mutable";
    if (symbol->isStatic()) out << " static";

    out << std::format(" {}\n", to_string(symbol->type(), symbol->name()));

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

  void dumpAbiTags(Symbol* symbol) {
    for (auto tag : symbol->abiTags()) {
      out << std::format(" [abi:{}]", tag->name());
    }
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
    dumpAbiTags(symbol);

    out << std::format(" {}", to_string(symbol->type(), symbol->name()));

    if (!symbol->templateArguments().empty()) {
      out << "<";
      std::string_view sep = "";
      for (const auto& arg :
           expand_template_arguments(symbol->templateArguments())) {
        out << std::format("{}{}", sep, to_string(arg));
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
    auto target = symbol->target();

    if (!target) {
      indent();
      out << std::format("using unresolved {}\n", to_string(symbol->name()));
      return;
    }

    for (auto introduced : symbol->introducedFunctions()) {
      indent();
      out << std::format("using {}\n",
                         to_string(introduced->type(), introduced->name()));
    }

    if (!symbol->introducedFunctions().empty()) return;

    indent();
    out << std::format("using {}\n", to_string(target->type(), target->name()));
  }
};
}  // namespace

void dump(std::ostream& out, Symbol* symbol, int depth) {
  visit(DumpSymbols{out, depth}, symbol);
}

void dump(std::ostream& out, Symbol* symbol, TranslationUnit* unit, int depth) {
  visit(DumpSymbols{out, depth, unit}, symbol);
}

auto operator<<(std::ostream& out, Symbol* symbol) -> std::ostream& {
  dump(out, symbol);
  return out;
}
}  // namespace cxx
