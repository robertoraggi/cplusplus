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

namespace cxx {

void NamePrinter::operator()(std::ostream& out, const Name* name) {
  if (!name) return;
  auto o = &out;
  std::swap(out_, o);
  accept(name);
  std::swap(out_, o);
}

void NamePrinter::accept(const Name* name) {
  if (!name) return;
  name->accept(this);
}

void NamePrinter::visit(const Identifier* name) {
  fmt::print(*out_, "{}", name->name());
}

void NamePrinter::visit(const OperatorNameId* name) {
  fmt::print(*out_, "operator {}", Token::spell(name->op()));
}

void NamePrinter::visit(const ConversionNameId* name) {
  fmt::print(*out_, "operator {}", name->type());
}

}  // namespace cxx
