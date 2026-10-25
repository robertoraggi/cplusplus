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

#include <cxx/ast_pretty_printer.h>
#include <cxx/names.h>
#include <cxx/printing.h>
#include <cxx/private/printing-priv.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

struct TemplateArgumentPrinter {
  detail::PrintOptions options;

  [[nodiscard]] auto const_value_to_string(const ConstValue& value) const
      -> std::string {
    if (auto v = std::get_if<ConstInt>(&value)) return v->toString();
    if (auto v = std::get_if<float>(&value)) return std::format("{}", *v);
    if (auto v = std::get_if<double>(&value)) return std::format("{}", *v);
    if (auto v = std::get_if<long double>(&value)) {
      return std::format("{}", *v);
    }
    if (std::holds_alternative<const StringLiteral*>(value)) return "\"...\"";
    if (std::holds_alternative<std::shared_ptr<Meta>>(value)) return "<meta>";
    if (std::holds_alternative<std::shared_ptr<InitializerList>>(value)) {
      return "<init-list>";
    }
    if (std::holds_alternative<std::shared_ptr<ConstObject>>(value)) {
      return "<const-object>";
    }
    return "<const>";
  }

  [[nodiscard]] auto pack_to_string(const ParameterPackSymbol* pack) const
      -> std::string {
    std::string s = "<";
    std::string_view sep = "";
    for (auto element : pack->elements()) {
      s += sep;
      s += detail::renderTemplateArgument(TemplateArgument{element}, options);
      sep = ", ";
    }
    s += '>';
    return s;
  }

  [[nodiscard]] auto operator()(const Type* type) const -> std::string {
    return detail::renderType(type, "", options);
  }

  [[nodiscard]] auto operator()(const ConstValue& value) const -> std::string {
    return const_value_to_string(value);
  }

  [[nodiscard]] auto operator()(Symbol* symbol) const -> std::string {
    if (!symbol) return "<null-symbol>";
    if (options.mode != detail::PrintMode::Internal) {
      if (auto pack = symbol_cast<ParameterPackSymbol>(symbol))
        return pack_to_string(pack);
    }
    if (symbol->isTypeAlias())
      return detail::renderType(symbol->type(), "", options);
    if (symbol->isVariable()) {
      auto var = static_cast<const VariableSymbol*>(symbol);
      if (auto cst = var->constValue()) return const_value_to_string(*cst);
    }
    if (auto type = symbol->type())
      return detail::renderType(type, "", options);
    return detail::renderName(symbol->name(), options);
  }

  [[nodiscard]] auto operator()(ExpressionAST* value) const -> std::string {
    if (!value || !options.unit) return {};
    return std::format("{}", formatAST(options.unit, value));
  }
};

struct NamePrinter {
  detail::PrintOptions options;

  [[nodiscard]] auto operator()(const Identifier* name) const -> std::string {
    if (options.mode != detail::PrintMode::Internal && !name->value().empty() &&
        name->isAnonymous())
      return "(anonymous)";
    return name->value();
  }

  [[nodiscard]] auto operator()(const OperatorId* name) const -> std::string {
    switch (name->op()) {
      case TokenKind::T_LPAREN:
        return "operator ()";
      case TokenKind::T_LBRACKET:
        return "operator []";
      case TokenKind::T_NEW_ARRAY:
        return "operator new[]";
      case TokenKind::T_DELETE_ARRAY:
        return "operator delete[]";
      default:
        return std::format("operator {}", Token::spell(name->op()));
    }  // switch
  }

  [[nodiscard]] auto operator()(const DestructorId* name) const -> std::string {
    return "~" + detail::renderName(name->name(), options);
  }

  [[nodiscard]] auto operator()(const LiteralOperatorId* name) const
      -> std::string {
    return std::format("operator \"\"{}", name->name());
  }

  [[nodiscard]] auto operator()(const ConversionFunctionId* name) const
      -> std::string {
    return std::format("operator {}",
                       detail::renderType(name->type(), "", options));
  }

  [[nodiscard]] auto operator()(const TemplateId* name) const -> std::string {
    std::string s = detail::renderName(name->name(), options);
    s += options.mode == detail::PrintMode::Internal ? " <" : "<";
    std::string_view sep = "";
    for (const auto& arg : expand_template_arguments(name->arguments())) {
      s += sep;
      s += detail::renderTemplateArgument(arg, options);
      sep = ", ";
    }
    s += '>';
    return s;
  }
};

}  // namespace

auto detail::renderName(const Name* name, PrintOptions options) -> std::string {
  if (!name) return {};
  return visit(NamePrinter{options}, name);
}

auto detail::renderTemplateArgument(const TemplateArgument& argument,
                                    PrintOptions options) -> std::string {
  auto text = std::visit(TemplateArgumentPrinter{options}, argument);
  if (text.empty()) return "?";
  return text;
}

auto NameFormat::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  return std::format_to(out, "{}", detail::renderName(name, {options, mode}));
}

auto TemplateArgumentFormat::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  return std::format_to(
      out, "{}", detail::renderTemplateArgument(argument, {options, mode}));
}

auto to_string(const Name* name, TypePrintOptions options) -> std::string {
  return std::format("{}", formatName(name, options));
}

auto to_string(const TemplateArgument& argument, TypePrintOptions options)
    -> std::string {
  return std::format("{}", formatTemplateArgument(argument, options));
}

auto to_pretty_string(const Name* name, TypePrintOptions options)
    -> std::string {
  return std::format("{}", formatPrettyName(name, options));
}

auto to_pretty_string(const TemplateArgument& argument,
                      TypePrintOptions options) -> std::string {
  return std::format("{}", formatPrettyTemplateArgument(argument, options));
}

auto to_diag_string(const Name* name, TypePrintOptions options) -> std::string {
  return std::format("{}", formatDiagName(name, options));
}

auto to_diag_string(const TemplateArgument& argument, TypePrintOptions options)
    -> std::string {
  return std::format("{}", formatDiagTemplateArgument(argument, options));
}

}  // namespace cxx
