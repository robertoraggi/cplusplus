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

#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/token.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>

namespace cxx {

auto NamePrinter::operator()(const Identifier* name) const -> std::string {
  return name->value();
}

auto NamePrinter::operator()(const OperatorId* name) const -> std::string {
  return fmt::format("operator {}", Token::spell(name->op()));
}

auto NamePrinter::operator()(const DestructorId* name) const -> std::string {
  return "~" + visit(*this, name->name());
}

auto NamePrinter::operator()(const LiteralOperatorId* name) const
    -> std::string {
  return fmt::format("operator \"\"{}", name->name());
}

auto NamePrinter::operator()(const ConversionFunctionId* name) const
    -> std::string {
  return fmt::format("operator {}", to_string(name->type()));
}

auto NamePrinter::operator()(const TemplateId* name) const -> std::string {
  std::string s = visit(*this, name->name());
  s += " <";
  for (auto&& arg : name->arguments()) {
    if (&arg != &name->arguments().front()) s += ", ";
    // s += to_string(arg);
  }
  s += '>';
  return s;
}

auto to_string(const Name* name) -> std::string {
  if (!name) return {};
  return visit(NamePrinter{}, name);
}

}  // namespace cxx