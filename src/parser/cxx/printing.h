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

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <format>
#include <string_view>
#include <variant>

namespace cxx {

namespace detail {
enum class PrintMode { Internal, Source, Diagnostic };
}

struct TypeFormat {
  const Type* type;
  std::variant<std::string_view, const Name*> identifier;
  TypePrintOptions options;
  detail::PrintMode mode;

  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
};

struct NameFormat {
  const Name* name;
  TypePrintOptions options;
  detail::PrintMode mode;

  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
};

struct TemplateArgumentFormat {
  const TemplateArgument& argument;
  TypePrintOptions options;
  detail::PrintMode mode;

  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
};

struct SymbolFormat {
  Symbol* symbol;
  TypePrintOptions options;
  detail::PrintMode mode;

  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
};

struct SymbolDump {
  Symbol* symbol;
  TranslationUnit* unit = nullptr;
  int depth = 0;

  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
};

[[nodiscard]] inline auto formatType(
    const Type* type,
    std::variant<std::string_view, const Name*> identifier = std::string_view{},
    TypePrintOptions options = {}) -> TypeFormat {
  return {type, identifier, options, detail::PrintMode::Internal};
}

[[nodiscard]] inline auto formatName(const Name* name,
                                     TypePrintOptions options = {})
    -> NameFormat {
  return {name, options, detail::PrintMode::Internal};
}

[[nodiscard]] inline auto formatTemplateArgument(
    const TemplateArgument& argument, TypePrintOptions options = {})
    -> TemplateArgumentFormat {
  return {argument, options, detail::PrintMode::Internal};
}

[[nodiscard]] inline auto formatSymbol(Symbol* symbol,
                                       TypePrintOptions options = {})
    -> SymbolFormat {
  return {symbol, options, detail::PrintMode::Internal};
}

[[nodiscard]] inline auto formatPrettyType(
    const Type* type,
    std::variant<std::string_view, const Name*> identifier = std::string_view{},
    TypePrintOptions options = {}) -> TypeFormat {
  return {type, identifier, options, detail::PrintMode::Source};
}

[[nodiscard]] inline auto formatPrettyName(const Name* name,
                                           TypePrintOptions options = {})
    -> NameFormat {
  return {name, options, detail::PrintMode::Source};
}

[[nodiscard]] inline auto formatPrettyTemplateArgument(
    const TemplateArgument& argument, TypePrintOptions options = {})
    -> TemplateArgumentFormat {
  return {argument, options, detail::PrintMode::Source};
}

[[nodiscard]] inline auto formatPrettySymbol(Symbol* symbol,
                                             TypePrintOptions options = {})
    -> SymbolFormat {
  return {symbol, options, detail::PrintMode::Source};
}

[[nodiscard]] inline auto formatDiagType(
    const Type* type,
    std::variant<std::string_view, const Name*> identifier = std::string_view{},
    TypePrintOptions options = {}) -> TypeFormat {
  return {type, identifier, options, detail::PrintMode::Diagnostic};
}

[[nodiscard]] inline auto formatDiagName(const Name* name,
                                         TypePrintOptions options = {})
    -> NameFormat {
  return {name, options, detail::PrintMode::Diagnostic};
}

[[nodiscard]] inline auto formatDiagTemplateArgument(
    const TemplateArgument& argument, TypePrintOptions options = {})
    -> TemplateArgumentFormat {
  return {argument, options, detail::PrintMode::Diagnostic};
}

[[nodiscard]] inline auto formatDiagSymbol(Symbol* symbol,
                                           TypePrintOptions options = {})
    -> SymbolFormat {
  return {symbol, options, detail::PrintMode::Diagnostic};
}

[[nodiscard]] inline auto dumpSymbols(Symbol* symbol,
                                      TranslationUnit* unit = nullptr,
                                      int depth = 0) -> SymbolDump {
  return {symbol, unit, depth};
}

namespace detail {
template <typename T>
struct PrinterFormatter {
  [[nodiscard]] constexpr auto parse(std::format_parse_context& context) {
    return context.begin();
  }

  auto format(const T& value, std::format_context& context) const {
    return value.formatTo(context.out());
  }
};
}  // namespace detail

}  // namespace cxx

template <>
struct std::formatter<cxx::TypeFormat>
    : cxx::detail::PrinterFormatter<cxx::TypeFormat> {};
template <>
struct std::formatter<cxx::NameFormat>
    : cxx::detail::PrinterFormatter<cxx::NameFormat> {};
template <>
struct std::formatter<cxx::TemplateArgumentFormat>
    : cxx::detail::PrinterFormatter<cxx::TemplateArgumentFormat> {};
template <>
struct std::formatter<cxx::SymbolFormat>
    : cxx::detail::PrinterFormatter<cxx::SymbolFormat> {};
template <>
struct std::formatter<cxx::SymbolDump>
    : cxx::detail::PrinterFormatter<cxx::SymbolDump> {};
