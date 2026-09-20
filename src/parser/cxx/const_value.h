// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/const_int.h>
#include <cxx/cxx_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace cxx {
class Meta;
class InitializerList;
class ConstComplex;
class ConstObject;
class ConstAddress;
class ConstLabelAddress;

struct DefaultInitializerContext {
  SourceLocation location;
  ScopeSymbol* scope = nullptr;
};

struct IndeterminateValue {
  auto operator==(const IndeterminateValue&) const -> bool = default;
};

using ConstValue =
    std::variant<ConstInt, const StringLiteral*, float, double, long double,
                 std::shared_ptr<Meta>, std::shared_ptr<InitializerList>,
                 std::shared_ptr<ConstObject>, std::shared_ptr<ConstAddress>,
                 std::shared_ptr<ConstLabelAddress>,
                 std::shared_ptr<ConstComplex>, IndeterminateValue>;

class InitializerList {
 public:
  std::vector<std::tuple<ConstValue, const Type*>> elements;
};

class ConstComplex {
 public:
  ConstComplex() = default;

  ConstComplex(ConstValue real, ConstValue imag)
      : real_(std::move(real)), imag_(std::move(imag)) {}

  [[nodiscard]] auto real() const -> const ConstValue& { return real_; }
  [[nodiscard]] auto imag() const -> const ConstValue& { return imag_; }

  void setReal(ConstValue value) { real_ = std::move(value); }
  void setImag(ConstValue value) { imag_ = std::move(value); }

 private:
  ConstValue real_;
  ConstValue imag_;
};

class ConstObject {
 public:
  struct Member {
    const Symbol* symbol = nullptr;
    ConstValue value;
  };

  ConstObject() = default;

  explicit ConstObject(const Type* type) : type_(type) {}

  ConstObject(const Type* type, std::deque<Member> members)
      : type_(type), members_(std::move(members)) {}

  [[nodiscard]] auto type() const -> const Type* { return type_; }

  void setType(const Type* type) { type_ = type; }

  [[nodiscard]] auto members() const -> const std::deque<Member>& {
    return members_;
  }

  [[nodiscard]] auto mutableMembers() -> std::deque<Member>& {
    return members_;
  }

  [[nodiscard]] auto isUnion() const -> bool;

  auto addMember(const Symbol* symbol, ConstValue value) -> ConstValue*;

  void setMember(const Symbol* symbol, ConstValue value);

  [[nodiscard]] auto subobject(const Symbol* symbol) const -> const ConstValue*;

  [[nodiscard]] auto mutableSubobject(const Symbol* symbol) -> ConstValue*;

  [[nodiscard]] auto operator==(const ConstObject& other) const -> bool;

 private:
  const Type* type_ = nullptr;
  std::deque<Member> members_;
};

class Meta {
 public:
  struct ConstExpr {
    ExpressionAST* expression = nullptr;
    ConstValue value;
  };

  std::variant<const Type*, const Symbol*, ConstExpr> value;
};

class ConstAddress {
 public:
  ConstAddress() = default;

  explicit ConstAddress(Symbol* symbol, std::intmax_t offset = 0)
      : symbol_(symbol), offset_(offset) {}

  explicit ConstAddress(const StringLiteral* string, std::intmax_t offset = 0)
      : string_(string), offset_(offset) {}

  ConstAddress(std::shared_ptr<ConstObject> owner, Symbol* symbol,
               std::intmax_t offset = 0)
      : symbol_(symbol), owner_(std::move(owner)), offset_(offset) {}

  explicit ConstAddress(const Type* typeInfoFor) : typeInfoFor_(typeInfoFor) {}

  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }
  [[nodiscard]] auto typeInfoFor() const -> const Type* { return typeInfoFor_; }
  [[nodiscard]] auto owner() const -> const std::shared_ptr<ConstObject>& {
    return owner_;
  }
  [[nodiscard]] auto stringLiteral() const -> const StringLiteral* {
    return string_;
  }
  [[nodiscard]] auto offset() const -> std::intmax_t { return offset_; }

  [[nodiscard]] auto sameTarget(const ConstAddress& other) const -> bool;

  void setSymbol(Symbol* symbol) { symbol_ = symbol; }
  void setOwner(std::shared_ptr<ConstObject> owner) {
    owner_ = std::move(owner);
  }
  void setStringLiteral(const StringLiteral* string) { string_ = string; }
  void setTypeInfoFor(const Type* type) { typeInfoFor_ = type; }
  void setOffset(std::intmax_t offset) { offset_ = offset; }

 private:
  Symbol* symbol_ = nullptr;
  std::shared_ptr<ConstObject> owner_;
  const StringLiteral* string_ = nullptr;
  const Type* typeInfoFor_ = nullptr;
  std::intmax_t offset_ = 0;
};

class ConstLabelAddress {
 public:
  ConstLabelAddress() = default;

  explicit ConstLabelAddress(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] auto name() const -> const std::string& { return name_; }

  void setName(std::string name) { name_ = std::move(name); }

 private:
  std::string name_;
};

[[nodiscard]] auto isFullyInitialized(const ConstValue& value) -> bool;

}  // namespace cxx
