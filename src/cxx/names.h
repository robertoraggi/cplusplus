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

#include <string>
#include <tuple>
#include <variant>

#include "token.h"

namespace cxx {

class Identifier;
class DestructorId;
class OperatorId;
class TemplateId;

using Name =
    std::variant<std::monostate, const Identifier*, const DestructorId*,
                 const OperatorId*, const TemplateId*>;

inline size_t hashCode(const Name& name);
inline std::string toString(const Name& name);

class Identifier : public std::tuple<std::string> {
  mutable size_t hashCode_ = static_cast<size_t>(-1);

 public:
  using tuple::tuple;

  const std::string& toString() const { return std::get<0>(*this); }

  size_t hashCode() const {
    if (hashCode_ != static_cast<size_t>(-1)) return hashCode_;
    hashCode_ = std::hash<std::string>()(toString());
    return hashCode_;
  }
};

class OperatorId : public std::tuple<TokenKind> {
  size_t hashCode_ = 0;

 public:
  OperatorId() = default;

  explicit OperatorId(TokenKind op) : tuple(op) { hashCode_ = size_t(op); }

  TokenKind op() const { return std::get<0>(*this); }

  std::string toString() const { return std::string(Token::spell(op())); }

  size_t hashCode() const { return hashCode_; }
};

class DestructorId : public std::tuple<Name> {
 public:
  using tuple::tuple;

  const Name& name() const { return std::get<0>(*this); }

  std::string toString() const { return "~" + cxx::toString(name()); }

  size_t hashCode() const { return cxx::hashCode(name()); }
};

class TemplateId : public std::tuple<Name> {
 public:
  using tuple::tuple;

  const Name& name() const { return std::get<0>(*this); }

  std::string toString() const { return cxx::toString(name()) + "<...>"; }

  size_t hashCode() const { return cxx::hashCode(name()); }
};

inline size_t hashCode(const Name& name) {
  static struct Hash {
    size_t operator()(const std::monostate&) const { return 0; }
    size_t operator()(const Identifier* id) const { return id->hashCode(); }
    size_t operator()(const DestructorId* id) const { return id->hashCode(); }
    size_t operator()(const OperatorId* id) const { return id->hashCode(); }
    size_t operator()(const TemplateId* id) const { return id->hashCode(); }
  } hash;
  return std::visit(hash, name);
}

inline std::string toString(const Name& name) {
  static struct ToString {
    std::string operator()(const std::monostate&) const {
      return std::string();
    }
    std::string operator()(const Identifier* id) const {
      return id->toString();
    }
    std::string operator()(const DestructorId* id) const {
      return id->toString();
    }
    std::string operator()(const OperatorId* id) const {
      return id->toString();
    }
    std::string operator()(const TemplateId* id) const {
      return id->toString();
    }
  } toString;
  return std::visit(toString, name);
}

}  // namespace cxx
