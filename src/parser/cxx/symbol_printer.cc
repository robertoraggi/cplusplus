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

#include <cxx/ast_rewriter.h>
#include <cxx/names.h>
#include <cxx/printing.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <ranges>
#include <unordered_set>

namespace cxx {
namespace {
struct GetEnumeratorValue {
  [[nodiscard]] auto operator()(bool value) const -> std::string {
    return value ? "true" : "false";
  }
  [[nodiscard]] auto operator()(ConstInt value) const -> std::string {
    return value.toString();
  }

  [[nodiscard]] auto operator()(auto x) const -> std::string { return {}; }
};

[[nodiscard]] auto templateParameterText(Symbol* parameter) -> std::string {
  auto nonTypeParameter = symbol_cast<NonTypeParameterSymbol>(parameter);
  auto text = nonTypeParameter ? to_string(nonTypeParameter->objectType(), "")
                               : to_string(parameter->type(), "");
  if (is_template_parameter_pack(parameter)) text += "...";
  return text;
}

struct DumpSymbols {
  std::format_context::iterator out;
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

    for (auto symbol : sortedSymbols) {
      if (symbol->canonical() != symbol) continue;
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->info() &&
          id->info()->kind() == IdentifierInfoKind::kBuiltinFunction) {
        continue;
      }
      if (id && id->builtinTemplate() != BuiltinTemplateKind::T_NONE) continue;
      if (isPreambleSymbol(symbol)) continue;
      visit(*this, symbol);
    }

    --depth;
  }

  void dumpSpecializations(
      std::span<const TemplateSpecialization> specializations) {
    if (specializations.empty()) return;
    ++depth;
    indent();
    out = std::format_to(out, "[specializations]\n");
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
    out = std::format_to(out, "{}", "[redeclarations]\n");
    ++depth;
    for (auto redecl : redecls) {
      visit(*this, redecl);
    }
    depth -= 2;
  }

  void indent() { out = std::format_to(out, "{:{}}", "", depth * 2); }

  void operator()(NamespaceSymbol* symbol) {
    indent();
    out = std::format_to(out, "{}", "namespace");
    if (symbol->name())
      out = std::format_to(out, " {}", to_string(symbol->name()));
    out = std::format_to(out, "{}", "\n");
    dumpScope(symbol);
  }

  void operator()(NamespaceAliasSymbol* symbol) {
    indent();
    std::string aliasedName{"<unresolved>"};
    if (auto aliased = symbol->namespaceSymbol())
      aliasedName = to_string(aliased->name());
    out = std::format_to(out, "namespace {} = {}\n", to_string(symbol->name()),
                         aliasedName);
  }

  void operator()(BaseClassSymbol* symbol) {
    indent();
    auto baseClass = symbol->symbol();
    if (baseClass && baseClass->type()) {
      out = std::format_to(out, "base class {}\n",
                           to_string(baseClass->type(), ""));
      return;
    }
    out = std::format_to(out, "base class {}\n", to_string(symbol->name()));
  }

  void operator()(InjectedClassNameSymbol* symbol) {
    indent();
    out = std::format_to(out, "injected class name {}\n",
                         to_string(symbol->name()));
  }

  void operator()(UnresolvedSymbol* symbol) {
    indent();
    out = std::format_to(out, "unresolved {}\n", to_string(symbol->name()));
  }

  void operator()(ClassSymbol* symbol) {
    indent();
    std::string_view classKey = symbol->isUnion() ? "union" : "class";

    if (symbol->templateParameters()) {
      out = std::format_to(out, "template {} {}", classKey,
                           to_string(symbol->name()));

      if (symbol->isSpecialization()) {
        out = std::format_to(out, "{}", '<');
        std::string_view sep = "";
        for (const auto& arg :
             expand_template_arguments(symbol->templateArguments())) {
          out = std::format_to(out, "{}{}", sep, to_string(arg));
          sep = ", ";
        }
        out = std::format_to(out, "{}", '>');
      } else {
        out = std::format_to(out, "{}", '<');
        std::string_view sep = "";
        for (const auto& param : views::members(symbol->templateParameters())) {
          out = std::format_to(out, "{}{}", sep, templateParameterText(param));
          sep = ", ";
        }
        out = std::format_to(out, "{}", '>');
      }

      out = std::format_to(out, "{}", "\n");

      dumpScope(symbol->templateParameters());
    } else if (symbol->isSpecialization()) {
      out = std::format_to(out, "{} {}", classKey, to_string(symbol->name()));
      out = std::format_to(out, "{}", "<");
      std::string_view sep = "";
      for (const auto& arg :
           expand_template_arguments(symbol->templateArguments())) {
        out = std::format_to(out, "{}{}", sep, to_string(arg));
        sep = ", ";
      }
      out = std::format_to(out, ">\n");
    } else if (symbol->isClosureType()) {
      out = std::format_to(out, "{} {}\n", classKey,
                           to_string(symbol->type(), ""));
    } else {
      out = std::format_to(out, "{} {}", classKey, to_string(symbol->name()));
      if (symbol->isFriend()) out = std::format_to(out, "{}", " friend");
      if (symbol->isHidden()) out = std::format_to(out, "{}", " hidden");
      if (symbol->isFinal()) out = std::format_to(out, "{}", " final");
      if (symbol->isPolymorphic())
        out = std::format_to(out, "{}", " polymorphic");
      if (symbol->isAbstract()) out = std::format_to(out, "{}", " abstract");
      out = std::format_to(out, "{}", "\n");
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
    out = std::format_to(out, "concept {}\n", to_string(symbol->name()));
    if (symbol->templateParameters()) dumpScope(symbol->templateParameters());
  }

  void operator()(DeductionGuideSymbol* symbol) {
    indent();
    if (auto funcType = type_cast<FunctionType>(symbol->type())) {
      std::string params;
      bool first = true;
      for (auto pt : funcType->parameterTypes()) {
        if (!first) params += ", ";
        params += to_string(pt, "");
        first = false;
      }
      out = std::format_to(out, "deduction-guide {}({}) -> {}\n",
                           to_string(symbol->name()), params,
                           to_string(funcType->returnType(), ""));
    } else {
      out = std::format_to(out, "deduction-guide {}\n",
                           to_string(symbol->name()));
    }
    if (symbol->templateParameters()) dumpScope(symbol->templateParameters());
  }

  void operator()(EnumSymbol* symbol) {
    indent();
    out = std::format_to(out, "enum {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out = std::format_to(out, " : {}", to_string(underlyingType, ""));
    }

    out = std::format_to(out, "\n");

    dumpScope(symbol);
  }

  void operator()(ScopedEnumSymbol* symbol) {
    indent();
    out = std::format_to(out, "enum class {}", to_string(symbol->name()));

    if (auto underlyingType = symbol->underlyingType()) {
      out = std::format_to(out, " : {}", to_string(underlyingType, ""));
    }

    out = std::format_to(out, "\n");

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
    if (unit) ASTRewriter::completePendingExceptionSpecification(unit, symbol);
    indent();

    if (symbol->templateParameters()) {
      out = std::format_to(out, "template ");
    }

    if (symbol->isConstructor()) {
      out = std::format_to(out, "constructor");
    } else {
      out = std::format_to(out, "function");
    }

    if (symbol->isStatic()) out = std::format_to(out, "{}", " static");
    if (symbol->isExtern()) out = std::format_to(out, "{}", " extern");
    if (symbol->isFriend()) out = std::format_to(out, "{}", " friend");
    if (symbol->isHidden()) out = std::format_to(out, "{}", " hidden");
    if (symbol->isConstexpr()) out = std::format_to(out, "{}", " constexpr");
    if (symbol->isConsteval()) out = std::format_to(out, "{}", " consteval");
    if (symbol->isInline()) out = std::format_to(out, "{}", " inline");
    if (symbol->isVirtual()) out = std::format_to(out, "{}", " virtual");
    if (symbol->isPure()) out = std::format_to(out, "{}", " pure");
    if (symbol->isOverride()) out = std::format_to(out, "{}", " override");
    if (symbol->isFinal()) out = std::format_to(out, "{}", " final");
    if (symbol->isExplicit()) out = std::format_to(out, "{}", " explicit");
    if (symbol->isDeleted()) out = std::format_to(out, "{}", " deleted");
    if (symbol->isDefaulted()) out = std::format_to(out, "{}", " defaulted");
    if (symbol->hasCLinkage()) out = std::format_to(out, "{}", " extern \"C\"");
    dumpAbiTags(symbol);

    out =
        std::format_to(out, " {}\n", to_string(symbol->type(), symbol->name()));

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters());
    }

    dumpScope(symbol);
    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(LambdaSymbol* symbol) {
    indent();

    out = std::format_to(out, "lambda");

    if (symbol->isConstexpr()) out = std::format_to(out, "{}", " constexpr");
    if (symbol->isConsteval()) out = std::format_to(out, "{}", " consteval");
    if (symbol->isMutable()) out = std::format_to(out, "{}", " mutable");
    if (symbol->isStatic()) out = std::format_to(out, "{}", " static");

    out =
        std::format_to(out, " {}\n", to_string(symbol->type(), symbol->name()));

    dumpScope(symbol);
    if (auto closure = symbol->closureType()) {
      ++depth;
      visit(*this, static_cast<Symbol*>(closure));
      --depth;
    }
  }

  void operator()(TemplateParametersSymbol* symbol) {
    indent();
    out = std::format_to(out, "template parameters\n");
    dumpScope(symbol);
  }

  void operator()(FunctionParametersSymbol* symbol) {
    indent();
    out = std::format_to(out, "parameters\n");
    dumpScope(symbol);
  }

  void operator()(BlockSymbol* symbol) {
    indent();
    out = std::format_to(out, "block\n");
    dumpScope(symbol);
  }

  void operator()(TypeAliasSymbol* symbol) {
    indent();
    if (symbol->templateParameters()) {
      out = std::format_to(out, "template typealias {}\n",
                           to_string(symbol->type(), symbol->name()));
      dumpScope(symbol->templateParameters());
    } else {
      out = std::format_to(out, "typealias {}\n",
                           to_string(symbol->type(), symbol->name()));
    }
    dumpRedeclarations(symbol);
  }

  void dumpAbiTags(Symbol* symbol) {
    for (auto tag : symbol->abiTags()) {
      out = std::format_to(out, " [abi:{}]", tag->name());
    }
  }

  void operator()(VariableSymbol* symbol) {
    indent();

    if (symbol->templateParameters()) out = std::format_to(out, "template ");

    out = std::format_to(out, "variable");

    if (symbol->isStatic()) out = std::format_to(out, "{}", " static");
    if (symbol->isThreadLocal())
      out = std::format_to(out, "{}", " thread_local");
    if (symbol->isExtern()) out = std::format_to(out, "{}", " extern");
    if (symbol->isConstexpr()) out = std::format_to(out, "{}", " constexpr");
    if (symbol->isConstinit()) out = std::format_to(out, "{}", " constinit");
    if (symbol->isInline()) out = std::format_to(out, "{}", " inline");
    dumpAbiTags(symbol);

    out = std::format_to(out, " {}", to_string(symbol->type(), symbol->name()));

    if (!symbol->templateArguments().empty()) {
      out = std::format_to(out, "{}", "<");
      std::string_view sep = "";
      for (const auto& arg :
           expand_template_arguments(symbol->templateArguments())) {
        out = std::format_to(out, "{}{}", sep, to_string(arg));
        sep = ", ";
      }
      out = std::format_to(out, ">");
    }

    out = std::format_to(out, "{}", "\n");

    if (symbol->templateParameters()) {
      dumpScope(symbol->templateParameters());
    }

    dumpSpecializations(symbol->specializations());
    dumpRedeclarations(symbol);
  }

  void operator()(FieldSymbol* symbol) {
    indent();

    if (symbol->isBitField())
      out = std::format_to(out, "bitfield");
    else
      out = std::format_to(out, "field");

    if (symbol->isStatic()) out = std::format_to(out, "{}", " static");
    if (symbol->isThreadLocal())
      out = std::format_to(out, "{}", " thread_local");
    if (symbol->isConstexpr()) out = std::format_to(out, "{}", " constexpr");
    if (symbol->isConstinit()) out = std::format_to(out, "{}", " constinit");
    if (symbol->isInline()) out = std::format_to(out, "{}", " inline");

    out =
        std::format_to(out, " {}\n", to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterSymbol* symbol) {
    indent();
    if (symbol->isParameterPack()) {
      out = std::format_to(out, "parameter {}... {}\n",
                           to_string(symbol->type(), ""),
                           to_string(symbol->name()));
      return;
    }
    out = std::format_to(out, "parameter {}\n",
                         to_string(symbol->type(), symbol->name()));
  }

  void operator()(ParameterPackSymbol* symbol) {
    indent();
    out = std::format_to(out, "parameter pack {}\n",
                         to_string(symbol->type(), symbol->name()));
  }

  void operator()(TypeParameterSymbol* symbol) {
    auto type = type_cast<TypeParameterType>(symbol->type());
    std::string_view pack = type->isParameterPack() ? "..." : "";
    indent();
    out =
        std::format_to(out, "parameter typename<{}, {}>{} {}\n", type->index(),
                       type->depth(), pack, to_string(symbol->name()));
  }

  void operator()(NonTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out = std::format_to(out, "parameter constant<{}, {}, {}>{} {}\n",
                         symbol->index(), symbol->depth(),
                         to_string(symbol->objectType(), ""), pack,
                         to_string(symbol->name()));
  }

  void operator()(TemplateTypeParameterSymbol* symbol) {
    auto type = type_cast<TemplateTypeParameterType>(symbol->type());
    std::string_view pack = type->isParameterPack() ? "..." : "";
    indent();
    out =
        std::format_to(out, "parameter template<{}, {}>{} {}\n", type->index(),
                       type->depth(), pack, to_string(symbol->name()));
  }

  void operator()(ConstraintTypeParameterSymbol* symbol) {
    std::string_view pack = symbol->isParameterPack() ? "..." : "";
    indent();
    out = std::format_to(out, "parameter constraint<{}, {}>{} {}\n",
                         symbol->index(), symbol->depth(), pack,
                         to_string(symbol->name()));
  }

  void operator()(EnumeratorSymbol* symbol) {
    indent();

    auto get_value = [](auto value) {
      return std::visit(GetEnumeratorValue{}, value);
    };

    const auto value = symbol->value().transform(get_value);

    out = std::format_to(out, "enumerator {}",
                         to_string(symbol->type(), symbol->name()));

    if (value.has_value() && !value->empty()) {
      out = std::format_to(out, " = {}", *value);
    }

    out = std::format_to(out, "{}", "\n");
  }

  void operator()(UsingDeclarationSymbol* symbol) {
    auto target = symbol->target();

    if (!target) {
      indent();
      out = std::format_to(out, "using unresolved {}\n",
                           to_string(symbol->name()));
      return;
    }

    for (auto introduced : symbol->introducedFunctions()) {
      indent();
      out = std::format_to(out, "using {}\n",
                           to_string(introduced->type(), introduced->name()));
    }

    if (!symbol->introducedFunctions().empty()) return;

    indent();
    out = std::format_to(out, "using {}\n",
                         to_string(target->type(), target->name()));
  }
};
}  // namespace

auto SymbolDump::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  if (!symbol) return out;
  DumpSymbols printer{out, depth, unit};
  visit(printer, symbol);
  return printer.out;
}
}  // namespace cxx
