// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/token.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct {
  auto operator()(const Type* type) const -> std::string {
    return to_string(type);
  }

  auto operator()(const ConstValue& value) const -> std::string { return {}; }

  auto operator()(ExpressionAST* value) const -> std::string { return {}; }

} template_argument_to_string;

auto NamePrinter::operator()(const Identifier* name) const -> std::string {
  return name->value();
}

auto NamePrinter::operator()(const OperatorId* name) const -> std::string {
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

auto NamePrinter::operator()(const DestructorId* name) const -> std::string {
  return "~" + visit(*this, name->name());
}

auto NamePrinter::operator()(const LiteralOperatorId* name) const
    -> std::string {
  return std::format("operator \"\"{}", name->name());
}

auto NamePrinter::operator()(const ConversionFunctionId* name) const
    -> std::string {
  return std::format("operator {}", to_string(name->type()));
}

auto NamePrinter::operator()(const TemplateId* name) const -> std::string {
  std::string s = visit(*this, name->name());
  s += " <";
  for (auto&& arg : name->arguments()) {
    if (&arg != &name->arguments().front()) s += ", ";
    s += std::visit(template_argument_to_string, arg);
  }
  s += '>';
  return s;
}

auto to_string(const Name* name) -> std::string {
  if (!name) return {};
  return visit(NamePrinter{}, name);
}

}  // namespace cxx