// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/token_fwd.h>

#include <string>
#include <tuple>
#include <vector>

namespace cxx {

class IdentifierInfo {
 public:
  explicit IdentifierInfo(IdentifierInfoKind kind) : kind_(kind) {}

  virtual ~IdentifierInfo() = default;

  [[nodiscard]] auto kind() const -> IdentifierInfoKind { return kind_; }

 private:
  IdentifierInfoKind kind_;
};

class TypeTraitIdentifierInfo final : public IdentifierInfo {
 public:
  static constexpr auto Kind = IdentifierInfoKind::kTypeTrait;

  explicit TypeTraitIdentifierInfo(BuiltinTypeTraitKind trait)
      : IdentifierInfo(Kind), trait_(trait) {}

  [[nodiscard]] auto trait() const -> BuiltinTypeTraitKind { return trait_; }

 private:
  BuiltinTypeTraitKind trait_;
};

class UnaryBuiltinTypeInfo final : public IdentifierInfo {
 public:
  static constexpr auto Kind = IdentifierInfoKind::kUnaryBuiltinType;

  explicit UnaryBuiltinTypeInfo(UnaryBuiltinTypeKind builtinKind)
      : IdentifierInfo(Kind), builtinKind_(builtinKind) {}

  [[nodiscard]] auto builtinKind() const -> UnaryBuiltinTypeKind {
    return builtinKind_;
  }

 private:
  UnaryBuiltinTypeKind builtinKind_;
};

class Name {
 public:
  Name(NameKind kind, std::size_t hashValue)
      : kind_(kind), hashValue_(hashValue) {}

  virtual ~Name();

  [[nodiscard]] auto kind() const -> NameKind { return kind_; }
  [[nodiscard]] auto hashValue() const -> std::size_t { return hashValue_; }

 private:
  NameKind kind_;
  std::size_t hashValue_;
};

class Identifier final : public Name {
 public:
  static constexpr auto Kind = NameKind::kIdentifier;

  static auto hash(std::string_view name) -> std::size_t {
    return std::hash<std::string_view>{}(name);
  }

  explicit Identifier(std::string name)
      : Name(Kind, hash(name)), name_(std::move(name)) {}

  [[nodiscard]] auto isAnonymous() const -> bool { return name_.at(0) == '$'; }

  [[nodiscard]] auto name() const -> const std::string& { return name_; }

  [[nodiscard]] auto value() const -> const std::string& { return name(); }

  [[nodiscard]] auto isBuiltinTypeTrait() const -> bool;
  [[nodiscard]] auto builtinTypeTrait() const -> BuiltinTypeTraitKind;

  [[nodiscard]] auto info() const -> const IdentifierInfo* { return info_; }
  void setInfo(const IdentifierInfo* info) const { info_ = info; }

 private:
  std::string name_;
  mutable const IdentifierInfo* info_ = nullptr;
};

class OperatorId final : public Name {
 public:
  static constexpr auto Kind = NameKind::kOperatorId;

  static auto hash(TokenKind op) -> std::size_t {
    return std::hash<int>{}(static_cast<int>(op));
  }

  explicit OperatorId(TokenKind op) : Name(Kind, hash(op)), op_(op) {}

  [[nodiscard]] auto op() const -> TokenKind { return op_; }

 private:
  TokenKind op_;
};

class DestructorId final : public Name, public std::tuple<const Name*> {
 public:
  static constexpr auto Kind = NameKind::kDestructorId;

  explicit DestructorId(const Name* name)
      : Name(Kind, name ? name->hashValue() : 0), tuple(name) {}

  [[nodiscard]] auto name() const -> const Name* { return std::get<0>(*this); }
};

class LiteralOperatorId final : public Name {
 public:
  static constexpr auto Kind = NameKind::kLiteralOperatorId;

  static auto hash(std::string_view name) -> std::size_t {
    return std::hash<std::string_view>{}(name);
  }

  explicit LiteralOperatorId(std::string name)
      : Name(Kind, hash(name)), name_(std::move(name)) {}

  [[nodiscard]] auto name() const -> const std::string& { return name_; }

 private:
  std::string name_;
};

class ConversionFunctionId final : public Name {
 public:
  static constexpr auto Kind = NameKind::kConversionFunctionId;

  static auto hash(const Type* type) -> std::size_t {
    return std::hash<const void*>{}(type);
  }

  explicit ConversionFunctionId(const Type* type)
      : Name(Kind, hash(type)), type_(type) {}

  [[nodiscard]] auto type() const -> const Type* { return type_; }

 private:
  const Type* type_;
};

class TemplateId final : public Name {
 public:
  static constexpr auto Kind = NameKind::kTemplateId;

  static auto hash(const Name* name, const std::vector<TemplateArgument>& args)
      -> std::size_t;

  explicit TemplateId(const Name* name, std::vector<TemplateArgument> args)
      : Name(Kind, hash(name, args)), name_(name), args_(std::move(args)) {}

  [[nodiscard]] auto name() const -> const Name* { return name_; }

  [[nodiscard]] auto arguments() const -> const std::vector<TemplateArgument>& {
    return args_;
  }

 private:
  const Name* name_;
  std::vector<TemplateArgument> args_;
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

[[nodiscard]] auto get_name(Control* control, UnqualifiedIdAST* id)
    -> const Name*;

}  // namespace cxx

template <>
struct std::hash<cxx::Identifier> {
  using is_transparent = void;

  auto operator()(const cxx::Identifier& id) const -> std::size_t {
    return id.hashValue();
  }

  auto operator()(const std::string_view& name) const -> std::size_t {
    return cxx::Identifier::hash(name);
  }

  auto operator()(const std::string& name) const -> std::size_t {
    return cxx::Identifier::hash(name);
  }
};

template <>
struct std::hash<cxx::OperatorId> {
  using is_transparent = void;

  auto operator()(const cxx::OperatorId& id) const -> std::size_t {
    return id.hashValue();
  }

  auto operator()(const cxx::TokenKind& op) const -> std::size_t {
    return cxx::OperatorId::hash(op);
  }
};

template <>
struct std::hash<cxx::DestructorId> {
  using is_transparent = void;

  auto operator()(const cxx::DestructorId& id) const -> std::size_t {
    return id.hashValue();
  }

  auto operator()(const cxx::Name* name) const -> std::size_t {
    return name ? name->hashValue() : 0;
  }
};

template <>
struct std::hash<cxx::LiteralOperatorId> {
  using is_transparent = void;

  auto operator()(const cxx::LiteralOperatorId& id) const -> std::size_t {
    return id.hashValue();
  }

  auto operator()(const std::string_view& name) const -> std::size_t {
    return cxx::LiteralOperatorId::hash(name);
  }
};

template <>
struct std::hash<cxx::ConversionFunctionId> {
  using is_transparent = void;

  auto operator()(const cxx::ConversionFunctionId& id) const -> std::size_t {
    return id.hashValue();
  }

  auto operator()(const cxx::Type* type) const -> std::size_t {
    return cxx::ConversionFunctionId::hash(type);
  }
};

template <>
struct std::hash<cxx::TemplateId> {
  auto operator()(const cxx::TemplateId& id) const -> std::size_t {
    return id.hashValue();
  }
};

template <>
struct std::equal_to<cxx::Identifier> {
  using is_transparent = void;

  auto operator()(const cxx::Identifier& id, const cxx::Identifier& other) const
      -> bool {
    return id.name() == other.name();
  }

  auto operator()(const cxx::Identifier& id, const std::string_view& name) const
      -> bool {
    return id.name() == name;
  }

  auto operator()(const std::string_view& name, const cxx::Identifier& id) const
      -> bool {
    return id.name() == name;
  }
};

template <>
struct std::equal_to<cxx::OperatorId> {
  using is_transparent = void;

  auto operator()(const cxx::OperatorId& id, const cxx::OperatorId& other) const
      -> bool {
    return id.op() == other.op();
  }

  auto operator()(const cxx::OperatorId& id, const cxx::TokenKind& op) const
      -> bool {
    return id.op() == op;
  }

  auto operator()(const cxx::TokenKind& op, const cxx::OperatorId& id) const
      -> bool {
    return id.op() == op;
  }
};

template <>
struct std::equal_to<cxx::DestructorId> {
  using is_transparent = void;

  auto operator()(const cxx::DestructorId& id,
                  const cxx::DestructorId& other) const -> bool {
    return id.name() == other.name();
  }

  auto operator()(const cxx::DestructorId& id, const cxx::Name* name) const
      -> bool {
    return id.name() == name;
  }

  auto operator()(const cxx::Name* name, const cxx::DestructorId& id) const
      -> bool {
    return id.name() == name;
  }
};

template <>
struct std::equal_to<cxx::LiteralOperatorId> {
  using is_transparent = void;

  auto operator()(const cxx::LiteralOperatorId& id,
                  const cxx::LiteralOperatorId& other) const -> bool {
    return id.name() == other.name();
  }

  auto operator()(const cxx::LiteralOperatorId& id,
                  const std::string_view& name) const -> bool {
    return id.name() == name;
  }

  auto operator()(const std::string_view& name,
                  const cxx::LiteralOperatorId& id) const -> bool {
    return id.name() == name;
  }
};

template <>
struct std::equal_to<cxx::ConversionFunctionId> {
  using is_transparent = void;

  auto operator()(const cxx::ConversionFunctionId& id,
                  const cxx::ConversionFunctionId& other) const -> bool {
    return id.type() == other.type();
  }

  auto operator()(const cxx::ConversionFunctionId& id,
                  const cxx::Type* type) const -> bool {
    return id.type() == type;
  }

  auto operator()(const cxx::Type* type,
                  const cxx::ConversionFunctionId& id) const -> bool {
    return id.type() == type;
  }
};

template <>
struct std::equal_to<cxx::TemplateId> {
  auto operator()(const cxx::TemplateId& id, const cxx::TemplateId& other) const
      -> bool {
    return id.name() == other.name() && id.arguments() == other.arguments();
  }
};