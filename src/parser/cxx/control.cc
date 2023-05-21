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

#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbol_factory.h>
#include <cxx/type_environment.h>

#include <forward_list>
#include <set>
#include <unordered_set>

namespace cxx {

namespace {

struct NameHash {
  std::hash<std::string> hash_value;

  auto operator()(const Identifier& id) const -> std::size_t {
    return hash_value(id.name());
  }

  auto operator()(const OperatorNameId& name) const -> std::size_t {
    return std::hash<uint32_t>()(static_cast<uint32_t>(name.op()));
  }

  auto operator()(const ConversionNameId& name) const -> std::size_t {
    return std::hash<const void*>()(name.type().type());
  }
};

struct NameEqualTo {
  auto operator()(const Identifier& name, const Identifier& other) const
      -> bool {
    return name.name() == other.name();
  }

  auto operator()(const OperatorNameId& name, const OperatorNameId& other) const
      -> bool {
    return name.op() == other.op();
  }

  auto operator()(const ConversionNameId& name,
                  const ConversionNameId& other) const -> bool {
    return name.type() == other.type();
  }
};

template <typename T>
using NameSet = std::unordered_set<T, NameHash, NameEqualTo>;

template <typename T>
struct LiteralLess {
  using is_transparent = void;

  auto operator()(const T& literal, const T& other) const -> bool {
    return literal.value() < other.value();
  }

  auto operator()(const T& literal, const std::string_view& value) const
      -> bool {
    return literal.value() < value;
  }

  auto operator()(const std::string_view& value, const T& literal) const
      -> bool {
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
  LiteralSet<WideStringLiteral> wideStringLiterals_;
  LiteralSet<Utf8StringLiteral> utf8StringLiterals_;
  LiteralSet<Utf16StringLiteral> utf16StringLiterals_;
  LiteralSet<Utf32StringLiteral> utf32StringLiterals_;
  LiteralSet<CharLiteral> charLiterals_;
  std::forward_list<CommentLiteral> commentLiterals_;
  std::set<Identifier> identifiers_;
  NameSet<OperatorNameId> operatorNameIds_;
  NameSet<ConversionNameId> conversionNameIds_;
};

Control::Control() : d(std::make_unique<Private>()) {}

Control::~Control() = default;

auto Control::identifier(const std::string_view& name) -> const Identifier* {
  if (auto it = d->identifiers_.find(name); it != d->identifiers_.end()) {
    return &*it;
  }

  return &*d->identifiers_.emplace(std::string{name}).first;
}

auto Control::operatorNameId(TokenKind op) -> const OperatorNameId* {
  return &*d->operatorNameIds_.emplace(op).first;
}

auto Control::conversionNameId(const QualifiedType& type)
    -> const ConversionNameId* {
  return &*d->conversionNameIds_.emplace(type).first;
}

auto Control::integerLiteral(const std::string_view& value)
    -> const IntegerLiteral* {
  if (auto it = d->integerLiterals_.find(value);
      it != d->integerLiterals_.end()) {
    return &*it;
  }

  return &*d->integerLiterals_.emplace(std::string{value}).first;
}

auto Control::floatLiteral(const std::string_view& value)
    -> const FloatLiteral* {
  if (auto it = d->floatLiterals_.find(value); it != d->floatLiterals_.end()) {
    return &*it;
  }

  return &*d->floatLiterals_.emplace(std::string{value}).first;
}

auto Control::stringLiteral(const std::string_view& value)
    -> const StringLiteral* {
  if (auto it = d->stringLiterals_.find(value);
      it != d->stringLiterals_.end()) {
    return &*it;
  }

  return &*d->stringLiterals_.emplace(std::string{value}).first;
}

auto Control::wideStringLiteral(const std::string_view& value)
    -> const WideStringLiteral* {
  if (auto it = d->wideStringLiterals_.find(value);
      it != d->wideStringLiterals_.end()) {
    return &*it;
  }

  return &*d->wideStringLiterals_.emplace(std::string{value}).first;
}

auto Control::utf8StringLiteral(const std::string_view& value)
    -> const Utf8StringLiteral* {
  if (auto it = d->utf8StringLiterals_.find(value);
      it != d->utf8StringLiterals_.end()) {
    return &*it;
  }

  return &*d->utf8StringLiterals_.emplace(std::string{value}).first;
}

auto Control::utf16StringLiteral(const std::string_view& value)
    -> const Utf16StringLiteral* {
  if (auto it = d->utf16StringLiterals_.find(value);
      it != d->utf16StringLiterals_.end()) {
    return &*it;
  }

  return &*d->utf16StringLiterals_.emplace(std::string{value}).first;
}

auto Control::utf32StringLiteral(const std::string_view& value)
    -> const Utf32StringLiteral* {
  if (auto it = d->utf32StringLiterals_.find(value);
      it != d->utf32StringLiterals_.end()) {
    return &*it;
  }

  return &*d->utf32StringLiterals_.emplace(std::string{value}).first;
}

auto Control::charLiteral(const std::string_view& value) -> const CharLiteral* {
  if (auto it = d->charLiterals_.find(value); it != d->charLiterals_.end()) {
    return &*it;
  }

  return &*d->charLiterals_.emplace(std::string{value}).first;
}

auto Control::commentLiteral(const std::string_view& value)
    -> const CommentLiteral* {
  return &d->commentLiterals_.emplace_front(std::string{value});
}

auto Control::types() -> TypeEnvironment* { return &d->typeEnvironment_; }

auto Control::symbols() -> SymbolFactory* { return &d->symbols_; }

}  // namespace cxx
