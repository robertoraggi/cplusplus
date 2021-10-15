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

#include <cxx/name_visitor.h>
#include <cxx/names_fwd.h>
#include <cxx/qualified_type.h>
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

class ConversionNameId final : public Name {
 public:
  explicit ConversionNameId(const QualifiedType& type) : type_(type) {}

  const QualifiedType& type() const { return type_; }

  void accept(NameVisitor* visitor) const override { visitor->visit(this); }

 private:
  QualifiedType type_;
};

std::ostream& operator<<(std::ostream& out, const Name& name);

}  // namespace cxx

template <>
struct std::less<cxx::Identifier> {
  using is_transparent = void;

  bool operator()(const cxx::Identifier& id,
                  const cxx::Identifier& other) const {
    return id.name() < other.name();
  }

  bool operator()(const cxx::Identifier& id,
                  const std::string_view& name) const {
    return id.name() < name;
  }

  bool operator()(const std::string_view& name,
                  const cxx::Identifier& id) const {
    return name < id.name();
  }
};
