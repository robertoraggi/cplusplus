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

struct LiteralHash {
  std::hash<std::string> hash_value;

  std::size_t operator()(const StringLiteral& literal) const {
    return hash_value(literal.value());
  }

  std::size_t operator()(const NumericLiteral& literal) const {
    return hash_value(literal.value());
  }

  std::size_t operator()(const CharLiteral& literal) const {
    return hash_value(literal.value());
  }
};

struct LiteralEqualTo {
  bool operator()(const NumericLiteral& literal,
                  const NumericLiteral& other) const {
    return literal.value() == other.value();
  }

  bool operator()(const StringLiteral& literal,
                  const StringLiteral& other) const {
    return literal.value() == other.value();
  }

  bool operator()(const CharLiteral& literal, const CharLiteral& other) const {
    return literal.value() == other.value();
  }
};

template <typename T>
using LiteralMap = std::unordered_set<T, LiteralHash, LiteralEqualTo>;

}  // namespace

struct Control::Private {
  TypeEnvironment typeEnvironment_;
  SymbolFactory symbols_;
  LiteralMap<NumericLiteral> numericLiterals_;
  LiteralMap<StringLiteral> stringLiterals_;
  LiteralMap<CharLiteral> charLiterals_;
  NameSet<Identifier> identifiers_;
  NameSet<OperatorNameId> operatorNameIds_;
  NameSet<ConversionNameId> conversionNameIds_;
};

Control::Control() : d(std::make_unique<Private>()) {}

Control::~Control() {}

const Identifier* Control::identifier(std::string name) {
  return &*d->identifiers_.emplace(std::move(name)).first;
}

const OperatorNameId* Control::operatorNameId(TokenKind op) {
  return &*d->operatorNameIds_.emplace(op).first;
}

const ConversionNameId* Control::conversionNameId(const QualifiedType& type) {
  return &*d->conversionNameIds_.emplace(type).first;
}

const NumericLiteral* Control::numericLiteral(std::string value) {
  return &*d->numericLiterals_.emplace(std::move(value)).first;
}

const StringLiteral* Control::stringLiteral(std::string value) {
  return &*d->stringLiterals_.emplace(std::move(value)).first;
}

const CharLiteral* Control::charLiteral(std::string value) {
  return &*d->charLiterals_.emplace(std::move(value)).first;
}

TypeEnvironment* Control::types() { return &d->typeEnvironment_; }

SymbolFactory* Control::symbols() { return &d->symbols_; }

}  // namespace cxx
