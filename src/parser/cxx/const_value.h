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
#include <cxx/cxx_fwd.h>
#include <cxx/literals_fwd.h>
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
class ConstObject;
class ConstAddress;
class ConstLabelAddress;

struct IndeterminateValue {
  auto operator==(const IndeterminateValue&) const -> bool = default;
};

using ConstValue =
    std::variant<std::intmax_t, const StringLiteral*, float, double,
                 long double, std::shared_ptr<Meta>,
                 std::shared_ptr<InitializerList>, std::shared_ptr<ConstObject>,
                 std::shared_ptr<ConstAddress>,
                 std::shared_ptr<ConstLabelAddress>, IndeterminateValue>;

class InitializerList {
 public:
  std::vector<std::tuple<ConstValue, const Type*>> elements;
};

class ConstObject {
 public:
  struct Field {
    const Symbol* symbol = nullptr;
    ConstValue value;
  };

  explicit ConstObject(const Type* type) : type_(type) {}

  ConstObject(const Type* type, std::deque<Field> fields)
      : type_(type), fields_(std::move(fields)) {}

  [[nodiscard]] auto type() const -> const Type* { return type_; }

  [[nodiscard]] auto fields() const -> const std::deque<Field>& {
    return fields_;
  }

  [[nodiscard]] auto mutableFields() -> std::deque<Field>& { return fields_; }

  void addField(const Symbol* symbol, ConstValue value) {
    fields_.push_back({symbol, std::move(value)});
  }

  [[nodiscard]] auto getField(const Symbol* symbol) const -> const ConstValue* {
    for (const auto& f : fields_) {
      if (f.symbol == symbol) return &f.value;
    }
    for (const auto& base : bases_) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&base)) {
        if (*obj) {
          if (auto found = (*obj)->getField(symbol)) return found;
        }
      }
    }
    return nullptr;
  }

  [[nodiscard]] auto getFieldMutable(const Symbol* symbol) -> ConstValue* {
    for (auto& f : fields_) {
      if (f.symbol == symbol) return &f.value;
    }
    for (auto& base : bases_) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&base)) {
        if (*obj) {
          if (auto found = (*obj)->getFieldMutable(symbol)) return found;
        }
      }
    }
    fields_.push_back({symbol, ConstValue{IndeterminateValue{}}});
    return &fields_.back().value;
  }

  void setField(const Symbol* symbol, ConstValue value) {
    for (auto& f : fields_) {
      if (f.symbol == symbol) {
        f.value = std::move(value);
        return;
      }
    }
    fields_.push_back({symbol, std::move(value)});
  }

  [[nodiscard]] auto bases() const -> const std::vector<ConstValue>& {
    return bases_;
  }

  [[nodiscard]] auto mutableBases() -> std::vector<ConstValue>& {
    return bases_;
  }

  void addBase(ConstValue base) { bases_.push_back(std::move(base)); }

  [[nodiscard]] auto operator==(const ConstObject& other) const -> bool;

 private:
  const Type* type_ = nullptr;
  std::deque<Field> fields_;
  std::vector<ConstValue> bases_;
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
  explicit ConstAddress(Symbol* symbol, std::intmax_t offset = 0)
      : symbol_(symbol), offset_(offset) {}

  explicit ConstAddress(const StringLiteral* string, std::intmax_t offset = 0)
      : string_(string), offset_(offset) {}

  ConstAddress(std::shared_ptr<ConstObject> owner, Symbol* symbol,
               std::intmax_t offset = 0)
      : symbol_(symbol), owner_(std::move(owner)), offset_(offset) {}

  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }
  [[nodiscard]] auto owner() const -> const std::shared_ptr<ConstObject>& {
    return owner_;
  }
  [[nodiscard]] auto stringLiteral() const -> const StringLiteral* {
    return string_;
  }
  [[nodiscard]] auto offset() const -> std::intmax_t { return offset_; }

 private:
  Symbol* symbol_ = nullptr;
  std::shared_ptr<ConstObject> owner_;
  const StringLiteral* string_ = nullptr;
  std::intmax_t offset_ = 0;
};

class ConstLabelAddress {
 public:
  explicit ConstLabelAddress(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] auto name() const -> const std::string& { return name_; }

 private:
  std::string name_;
};
}  // namespace cxx
