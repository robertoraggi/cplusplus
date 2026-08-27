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

#include <simdjson.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cxx::lsp {

[[noreturn]] void lsp_runtime_error(const std::string& msg);

class Json final {
 public:
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json, std::less<>>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool value) : value_(value) {}
  Json(double value) : value_(value) {}
  Json(std::string value) : value_(std::move(value)) {}
  Json(std::string_view value) : value_(std::string(value)) {}
  Json(const char* value) : value_(std::string(value)) {}

  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool> &&
             std::is_signed_v<T>)
  Json(T value) : value_(static_cast<std::int64_t>(value)) {}

  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool> &&
             std::is_unsigned_v<T>)
  Json(T value) : value_(static_cast<std::uint64_t>(value)) {}

  template <typename T>
  Json(const std::vector<T>& values) : value_(Array{}) {
    auto& array = std::get<Array>(value_);
    array.reserve(values.size());
    for (const auto& value : values) array.emplace_back(value);
  }

  template <typename T>
  Json(std::vector<T>&& values) : value_(Array{}) {
    auto& array = std::get<Array>(value_);
    array.reserve(values.size());
    for (auto& value : values) array.emplace_back(std::move(value));
  }

  [[nodiscard]] static auto array() -> Json;
  [[nodiscard]] static auto object() -> Json;
  [[nodiscard]] static auto parse(std::string_view source) -> Json;

  [[nodiscard]] auto dump(int indent = -1) const -> std::string;

  [[nodiscard]] auto is_null() const -> bool;
  [[nodiscard]] auto is_boolean() const -> bool;
  [[nodiscard]] auto is_number_integer() const -> bool;
  [[nodiscard]] auto is_number_float() const -> bool;
  [[nodiscard]] auto is_number() const -> bool;
  [[nodiscard]] auto is_string() const -> bool;
  [[nodiscard]] auto is_array() const -> bool;
  [[nodiscard]] auto is_object() const -> bool;

  [[nodiscard]] auto contains(std::string_view key) const -> bool;
  [[nodiscard]] auto size() const -> std::size_t;
  [[nodiscard]] auto empty() const -> bool;

  auto operator[](std::string_view key) -> Json&;
  auto at(std::string_view key) -> Json&;
  auto at(std::string_view key) const -> const Json&;
  auto at(std::size_t index) -> Json&;
  auto at(std::size_t index) const -> const Json&;

  template <typename T>
  auto emplace(std::string key, T&& value) -> Json& {
    ensure_object();
    auto& object = std::get<Object>(value_);
    auto result =
        object.insert_or_assign(std::move(key), Json(std::forward<T>(value)));
    return result.first->second;
  }

  template <typename... Args>
  auto emplace_back(Args&&... args) -> Json& {
    ensure_array();
    auto& array = std::get<Array>(value_);
    array.emplace_back(std::forward<Args>(args)...);
    return array.back();
  }

  auto erase(std::string_view key) -> std::size_t;

  template <typename T>
  [[nodiscard]] auto get() const -> T {
    if constexpr (std::is_same_v<T, Json>) {
      return *this;
    } else if constexpr (std::is_same_v<T, std::string>) {
      return std::get<std::string>(value_);
    } else if constexpr (std::is_same_v<T, bool>) {
      return std::get<bool>(value_);
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      if (auto value = std::get_if<std::int64_t>(&value_))
        return static_cast<T>(*value);
      return static_cast<T>(std::get<std::uint64_t>(value_));
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
      if (auto value = std::get_if<std::uint64_t>(&value_))
        return static_cast<T>(*value);
      return static_cast<T>(std::get<std::int64_t>(value_));
    } else if constexpr (std::is_floating_point_v<T>) {
      if (auto value = std::get_if<double>(&value_))
        return static_cast<T>(*value);
      if (auto value = std::get_if<std::int64_t>(&value_))
        return static_cast<T>(*value);
      return static_cast<T>(std::get<std::uint64_t>(value_));
    } else {
      static_assert(!sizeof(T), "unsupported Json::get type");
    }
  }

  friend auto operator==(const Json& value, std::string_view text) -> bool;
  friend auto operator==(std::string_view text, const Json& value) -> bool;
  friend auto operator!=(const Json& value, std::string_view text) -> bool;
  friend auto operator!=(std::string_view text, const Json& value) -> bool;

 private:
  using Value = std::variant<std::nullptr_t, bool, std::int64_t, std::uint64_t,
                             double, std::string, Array, Object>;

  explicit Json(Array value) : value_(std::move(value)) {}
  explicit Json(Object value) : value_(std::move(value)) {}

  void ensure_array();
  void ensure_object();

  static auto from_simdjson(simdjson::dom::element element) -> Json;
  void append_to(simdjson::builder::string_builder& builder, int indent,
                 int depth) const;

  Value value_{nullptr};
};

using json = Json;

}  // namespace cxx::lsp
