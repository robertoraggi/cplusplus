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

#include <cxx/lsp/json.h>

namespace cxx::lsp {

namespace {

void append_indent(simdjson::builder::string_builder& builder, int count) {
  for (int index = 0; index < count; ++index) builder.append(' ');
}

template <typename T>
auto simdjson_value(simdjson::simdjson_result<T> result) -> T {
  T value;
  const auto error = std::move(result).get(value);
  if (error) lsp_runtime_error(simdjson::error_message(error));
  return value;
}

}  // namespace

auto Json::array() -> Json { return Json(Array{}); }

auto Json::object() -> Json { return Json(Object{}); }

auto Json::parse(std::string_view source) -> Json {
  simdjson::dom::parser parser;
  simdjson::dom::element element;
  const auto error = parser.parse(source.data(), source.size()).get(element);
  if (error) lsp_runtime_error(simdjson::error_message(error));
  return from_simdjson(element);
}

auto Json::dump(int indent) const -> std::string {
  simdjson::builder::string_builder builder;
  append_to(builder, indent, 0);
  std::string_view result;
  const auto error = builder.view().get(result);
  if (error) lsp_runtime_error(simdjson::error_message(error));
  return std::string(result);
}

auto Json::is_null() const -> bool {
  return std::holds_alternative<std::nullptr_t>(value_);
}

auto Json::is_boolean() const -> bool {
  return std::holds_alternative<bool>(value_);
}

auto Json::is_number_integer() const -> bool {
  if (std::holds_alternative<std::int64_t>(value_)) return true;
  return std::holds_alternative<std::uint64_t>(value_);
}

auto Json::is_number_float() const -> bool {
  return std::holds_alternative<double>(value_);
}

auto Json::is_number() const -> bool {
  if (is_number_integer()) return true;
  return is_number_float();
}

auto Json::is_string() const -> bool {
  return std::holds_alternative<std::string>(value_);
}

auto Json::is_array() const -> bool {
  return std::holds_alternative<Array>(value_);
}

auto Json::is_object() const -> bool {
  return std::holds_alternative<Object>(value_);
}

auto Json::contains(std::string_view key) const -> bool {
  const auto object = std::get_if<Object>(&value_);
  if (!object) return false;
  return object->contains(key);
}

auto Json::size() const -> std::size_t {
  if (const auto array = std::get_if<Array>(&value_)) return array->size();
  if (const auto object = std::get_if<Object>(&value_)) return object->size();
  return 0;
}

auto Json::empty() const -> bool { return size() == 0; }

auto Json::operator[](std::string_view key) -> Json& {
  ensure_object();
  return std::get<Object>(value_)[std::string(key)];
}

auto Json::at(std::string_view key) -> Json& {
  return std::get<Object>(value_).at(std::string(key));
}

auto Json::at(std::string_view key) const -> const Json& {
  return std::get<Object>(value_).at(std::string(key));
}

auto Json::at(std::size_t index) -> Json& {
  return std::get<Array>(value_).at(index);
}

auto Json::at(std::size_t index) const -> const Json& {
  return std::get<Array>(value_).at(index);
}

auto Json::erase(std::string_view key) -> std::size_t {
  auto object = std::get_if<Object>(&value_);
  if (!object) return 0;
  return object->erase(std::string(key));
}

auto operator==(const Json& value, std::string_view text) -> bool {
  if (!value.is_string()) return false;
  return value.get<std::string>() == text;
}

auto operator==(std::string_view text, const Json& value) -> bool {
  return value == text;
}

auto operator!=(const Json& value, std::string_view text) -> bool {
  return !(value == text);
}

auto operator!=(std::string_view text, const Json& value) -> bool {
  return !(value == text);
}

void Json::ensure_array() {
  if (is_null()) value_ = Array{};
  if (!is_array()) lsp_runtime_error("JSON value is not an array");
}

void Json::ensure_object() {
  if (is_null()) value_ = Object{};
  if (!is_object()) lsp_runtime_error("JSON value is not an object");
}

auto Json::from_simdjson(simdjson::dom::element element) -> Json {
  switch (element.type()) {
    case simdjson::dom::element_type::ARRAY: {
      Array result;
      for (simdjson::dom::element item : simdjson_value(element.get_array()))
        result.push_back(from_simdjson(item));
      return Json(std::move(result));
    }

    case simdjson::dom::element_type::OBJECT: {
      Object result;
      for (const auto field : simdjson_value(element.get_object()))
        result.emplace(std::string(field.key), from_simdjson(field.value));
      return Json(std::move(result));
    }

    case simdjson::dom::element_type::INT64:
      return Json(simdjson_value(element.get_int64()));

    case simdjson::dom::element_type::UINT64:
      return Json(simdjson_value(element.get_uint64()));

    case simdjson::dom::element_type::DOUBLE:
      return Json(simdjson_value(element.get_double()));

    case simdjson::dom::element_type::STRING:
      return Json(simdjson_value(element.get_string()));

    case simdjson::dom::element_type::BOOL:
      return Json(simdjson_value(element.get_bool()));

    case simdjson::dom::element_type::NULL_VALUE:
      return Json();

    case simdjson::dom::element_type::BIGINT:
      lsp_runtime_error("JSON integer is out of range");
  }

  lsp_runtime_error("invalid JSON value");
}

void Json::append_to(simdjson::builder::string_builder& builder, int indent,
                     int depth) const {
  if (is_null()) {
    builder.append_null();
    return;
  }

  if (const auto value = std::get_if<bool>(&value_)) {
    builder.append(*value);
    return;
  }

  if (const auto value = std::get_if<std::int64_t>(&value_)) {
    builder.append(*value);
    return;
  }

  if (const auto value = std::get_if<std::uint64_t>(&value_)) {
    builder.append(*value);
    return;
  }

  if (const auto value = std::get_if<double>(&value_)) {
    builder.append(*value);
    return;
  }

  if (const auto value = std::get_if<std::string>(&value_)) {
    builder.escape_and_append_with_quotes(*value);
    return;
  }

  const bool pretty = indent >= 0;

  if (const auto array = std::get_if<Array>(&value_)) {
    builder.start_array();
    for (std::size_t index = 0; index < array->size(); ++index) {
      if (index != 0) builder.append_comma();
      if (pretty) {
        builder.append('\n');
        append_indent(builder, (depth + 1) * indent);
      }
      (*array)[index].append_to(builder, indent, depth + 1);
    }
    if (pretty && !array->empty()) {
      builder.append('\n');
      append_indent(builder, depth * indent);
    }
    builder.end_array();
    return;
  }

  const auto& object = std::get<Object>(value_);
  builder.start_object();
  std::size_t index = 0;
  for (const auto& [key, value] : object) {
    if (index != 0) builder.append_comma();
    if (pretty) {
      builder.append('\n');
      append_indent(builder, (depth + 1) * indent);
    }
    builder.escape_and_append_with_quotes(key);
    builder.append_colon();
    if (pretty) builder.append(' ');
    value.append_to(builder, indent, depth + 1);
    ++index;
  }
  if (pretty && !object.empty()) {
    builder.append('\n');
    append_indent(builder, depth * indent);
  }
  builder.end_object();
}

}  // namespace cxx::lsp
