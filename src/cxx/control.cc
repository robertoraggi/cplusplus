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

#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbol_factory.h>
#include <cxx/type_environment.h>

#include <set>
#include <unordered_set>

namespace cxx {

namespace {

struct NameHash {
  std::hash<std::string> hash_value;

  std::size_t operator()(const Identifier& id) const {
    return hash_value(id.name());
  }

  std::size_t operator()(const OperatorNameId& name) const {
    return std::hash<uint32_t>()(static_cast<uint32_t>(name.op()));
  }

  std::size_t operator()(const ConversionNameId& name) const {
    return std::hash<const void*>()(name.type().type());
  }
};

struct NameEqualTo {
  bool operator()(const Identifier& name, const Identifier& other) const {
    return name.name() == other.name();
  }

  bool operator()(const OperatorNameId& name,
                  const OperatorNameId& other) const {
    return name.op() == other.op();
  }

  bool operator()(const ConversionNameId& name,
                  const ConversionNameId& other) const {
    return name.type() == other.type();
  }
};

template <typename T>
using NameSet = std::unordered_set<T, NameHash, NameEqualTo>;

template <typename T>
struct LiteralLess {
  using is_transparent = void;

  bool operator()(const T& literal, const T& other) const {
    return literal.value() < other.value();
  }

  bool operator()(const T& literal, const std::string_view& value) const {
    return literal.value() < value;
  }

  bool operator()(const std::string_view& value, const T& literal) const {
    return value < literal.value();
  }
};

template <typename T>
using LiteralSet = std::set<T, LiteralLess<T>>;

}  // namespace

struct Control::Private {
  TypeEnvironment typeEnvironment_;
  SymbolFactory symbols_;
  LiteralSet<IntegerLiteral> integerLiterals_;
  LiteralSet<FloatLiteral> floatLiterals_;
  LiteralSet<StringLiteral> stringLiterals_;
  LiteralSet<CharLiteral> charLiterals_;
  std::set<Identifier> identifiers_;
  NameSet<OperatorNameId> operatorNameIds_;
  NameSet<ConversionNameId> conversionNameIds_;
};

Control::Control() : d(std::make_unique<Private>()) {}

Control::~Control() {}

const Identifier* Control::identifier(const std::string_view& name) {
  if (auto it = d->identifiers_.find(name); it != d->identifiers_.end())
    return &*it;

  return &*d->identifiers_.emplace(std::string{name}).first;
}

const OperatorNameId* Control::operatorNameId(TokenKind op) {
  return &*d->operatorNameIds_.emplace(op).first;
}

const ConversionNameId* Control::conversionNameId(const QualifiedType& type) {
  return &*d->conversionNameIds_.emplace(type).first;
}

const IntegerLiteral* Control::integerLiteral(const std::string_view& value) {
  if (auto it = d->integerLiterals_.find(value);
      it != d->integerLiterals_.end())
    return &*it;

  return &*d->integerLiterals_.emplace(std::string{value}).first;
}

const FloatLiteral* Control::floatLiteral(const std::string_view& value) {
  if (auto it = d->floatLiterals_.find(value); it != d->floatLiterals_.end())
    return &*it;

  return &*d->floatLiterals_.emplace(std::string{value}).first;
}

const StringLiteral* Control::stringLiteral(const std::string_view& value) {
  if (auto it = d->stringLiterals_.find(value); it != d->stringLiterals_.end())
    return &*it;

  return &*d->stringLiterals_.emplace(std::string{value}).first;
}

const CharLiteral* Control::charLiteral(const std::string_view& value) {
  if (auto it = d->charLiterals_.find(value); it != d->charLiterals_.end())
    return &*it;

  return &*d->charLiterals_.emplace(std::string{value}).first;
}

TypeEnvironment* Control::types() { return &d->typeEnvironment_; }

SymbolFactory* Control::symbols() { return &d->symbols_; }

}  // namespace cxx
