// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "names.h"

#include "token.h"

namespace cxx {

std::string Identifier::toString() const { return string(); }

std::string OperatorName::toString() const {
  const auto o = op();
  std::string s;
  s += "operator ";
  s += Token::spell(o);
  if (o == TokenKind::T_LPAREN)
    s += ")";
  else if (o == TokenKind::T_LBRACKET)
    s += "]";
  return s;
}

std::string ConversionName::toString() const {
  TypeToString typeToString;
  return "operator " + typeToString(type(), nullptr);
}

std::string DestructorName::toString() const {
  return "~" + name()->toString();
}

std::string TemplateName::toString() const {
  TypeToString typeToString;
  auto s = name()->toString();
  s += '<';
  bool first = true;
  for (auto&& arg : argumentTypes()) {
    if (first)
      first = false;
    else
      s += ", ";
    s += typeToString(arg);
  }
  s += '>';
  return s;
}

std::string QualifiedName::toString() const {
  if (auto b = base()) return b->toString() + "::" + name()->toString();
  return "::" + name()->toString();
}

std::string DecltypeName::toString() const {
  TypeToString typeToString;
  return "decltype(" + typeToString(type(), nullptr) + ")";
}

}  // namespace cxx
