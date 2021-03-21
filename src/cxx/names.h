// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/fully_specified_type.h>
#include <cxx/name_visitor.h>
#include <cxx/names_fwd.h>
#include <cxx/token.h>

#include <iosfwd>
#include <string>

namespace cxx {

class Name {
 public:
  virtual ~Name();

  virtual void accept(NameVisitor* visitor) const = 0;
};

class Identifier final : public Name {
 public:
  explicit Identifier(std::string name) : name_(std::move(name)) {}

  const std::string& name() const { return name_; }

  void accept(NameVisitor* visitor) const override { visitor->visit(this); }

 private:
  std::string name_;
};

class OperatorNameId final : public Name {
 public:
  explicit OperatorNameId(TokenKind op) : op_(op) {}

  TokenKind op() const { return op_; }

  void accept(NameVisitor* visitor) const override { visitor->visit(this); }

 private:
  TokenKind op_;
};

std::ostream& operator<<(std::ostream& out, const Name& name);

}  // namespace cxx
