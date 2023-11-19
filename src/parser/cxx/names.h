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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/token_fwd.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace cxx {

class Name {
 public:
  explicit Name(NameKind kind) : kind_(kind) {}
  virtual ~Name();

  auto kind() const -> NameKind { return kind_; }

 private:
  NameKind kind_;
};

class Identifier final : public Name, public std::tuple<std::string> {
 public:
  static constexpr auto Kind = NameKind::kIdentifier;

  explicit Identifier(std::string name) : Name(Kind), tuple(std::move(name)) {}

  auto isAnonymous() const -> bool { return name().at(0) == '$'; }
  auto name() const -> const std::string& { return std::get<0>(*this); }
  auto value() const -> const std::string& { return name(); }

  friend auto operator<(const Identifier& identifier,
                        const std::string_view& other) -> bool {
    return identifier.name() < other;
  }

  friend auto operator<(const std::string_view& other,
                        const Identifier& identifier) -> bool {
    return other < identifier.name();
  }
};

class OperatorId final : public Name, public std::tuple<TokenKind> {
 public:
  static constexpr auto Kind = NameKind::kOperatorId;

  explicit OperatorId(TokenKind op) : Name(Kind), tuple(op) {}

  auto op() const -> TokenKind { return std::get<0>(*this); }
};

class DestructorId final : public Name, public std::tuple<const Name*> {
 public:
  static constexpr auto Kind = NameKind::kDestructorId;

  explicit DestructorId(const Name* name) : Name(Kind), tuple(name) {}

  auto name() const -> const Name* { return std::get<0>(*this); }
};

class LiteralOperatorId final : public Name, public std::tuple<std::string> {
 public:
  static constexpr auto Kind = NameKind::kLiteralOperatorId;

  explicit LiteralOperatorId(std::string name)
      : Name(Kind), tuple(std::move(name)) {}

  auto name() const -> const std::string& { return std::get<0>(*this); }
};

class ConversionFunctionId final : public Name, public std::tuple<const Type*> {
 public:
  static constexpr auto Kind = NameKind::kConversionFunctionId;

  explicit ConversionFunctionId(const Type* type) : Name(Kind), tuple(type) {}

  auto type() const -> const Type* { return std::get<0>(*this); }
};

class TemplateId final
    : public Name,
      public std::tuple<const Name*, std::vector<TemplateArgument>> {
 public:
  static constexpr auto Kind = NameKind::kTemplateId;

  explicit TemplateId(const Name* name, std::vector<TemplateArgument> args)
      : Name(Kind), tuple(name, std::move(args)) {}

  auto name() const -> const Name* { return std::get<0>(*this); }

  auto arguments() const -> const std::vector<TemplateArgument>& {
    return std::get<1>(*this);
  }
};

template <typename Visitor>
auto visit(Visitor&& visitor, const Name* name) {
#define PROCESS_NAME(N) \
  case NameKind::k##N:  \
    return std::forward<Visitor>(visitor)(static_cast<const N*>(name));

  switch (name->kind()) {
    CXX_FOR_EACH_NAME(PROCESS_NAME)
    default:
      cxx_runtime_error("invalid name kind");
  }  // switch

#undef PROCESS_NAME
}

template <typename T>
auto name_cast(const Name* name) -> const T* {
  return name && name->kind() == T::Kind ? static_cast<const T*>(name)
                                         : nullptr;
}

}  // namespace cxx