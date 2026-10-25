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

#include <cxx/printing.h>

#include <functional>

namespace cxx::detail {

struct FormattingAction {
  const std::function<void(std::format_context::iterator&)>& action;
};

struct PrintOptions : TypePrintOptions {
  PrintMode mode;

  constexpr PrintOptions(TypePrintOptions options, PrintMode mode)
      : TypePrintOptions(options), mode(mode) {}
};

[[nodiscard]] auto renderType(const Type* type, const std::string& identifier,
                              PrintOptions options) -> std::string;
[[nodiscard]] auto renderType(const Type* type, const Name* identifier,
                              PrintOptions options) -> std::string;
[[nodiscard]] auto renderName(const Name* name, PrintOptions options)
    -> std::string;
[[nodiscard]] auto renderTemplateArgument(const TemplateArgument& argument,
                                          PrintOptions options) -> std::string;

}  // namespace cxx::detail

template <>
struct std::formatter<cxx::detail::FormattingAction> {
  [[nodiscard]] constexpr auto parse(std::format_parse_context& context) {
    return context.begin();
  }
  auto format(const cxx::detail::FormattingAction& value,
              std::format_context& context) const {
    auto out = context.out();
    value.action(out);
    return out;
  }
};
