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

#include <cxx/codegen/emitter_handles.h>
#include <cxx/source_location.h>
#include <emscripten/val.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace cxx::js {

using emscripten::val;

inline auto toVal(bool value) -> val { return val(value); }

inline auto toVal(double value) -> val { return val(value); }

template <typename T>
  requires std::is_integral_v<T>
auto toVal(T value) -> val {
  return val(static_cast<double>(value));
}

template <typename T>
  requires std::is_enum_v<T>
auto toVal(T value) -> val {
  return val(static_cast<int>(value));
}

template <typename Tag>
auto toVal(ir::Handle<Tag> handle) -> val {
  return val(static_cast<double>(ir::HandleAccess::id(handle)));
}

inline auto toVal(SourceLocation loc) -> val {
  return val(static_cast<double>(loc.index()));
}

inline auto toVal(std::string_view text) -> val {
  return val(std::string(text));
}

inline auto toVal(const std::string& bytes) -> val {
  auto result = val::global("Uint8Array").new_(bytes.size());
  result.call<void>(
      "set",
      val(emscripten::typed_memory_view(
          bytes.size(), reinterpret_cast<const std::uint8_t*>(bytes.data()))));
  return result;
}

template <typename T, typename Convert>
auto arrayVal(std::span<const T> items, Convert&& convert) -> val {
  auto result = val::array();
  for (const auto& item : items) result.call<void>("push", convert(item));
  return result;
}

template <typename T, typename Convert>
auto arrayVal(const std::vector<T>& items, Convert&& convert) -> val {
  return arrayVal(std::span<const T>(items), std::forward<Convert>(convert));
}

template <typename T, typename Convert>
auto optionalVal(const std::optional<T>& value, Convert&& convert) -> val {
  if (!value.has_value()) return val::undefined();
  return convert(*value);
}

inline auto toBool(const val& value) -> bool {
  return value.isUndefined() || value.isNull() ? false : value.as<bool>();
}

template <typename T>
auto toNumber(const val& value) -> T {
  if (!value.isNumber()) return T{};
  return static_cast<T>(value.as<double>());
}

template <typename T>
auto toEnum(const val& value) -> T {
  return static_cast<T>(toNumber<int>(value));
}

template <typename Tag>
auto toHandle(const val& value) -> ir::Handle<Tag> {
  if (!value.isNumber()) return ir::Handle<Tag>{};
  return ir::HandleAccess::make<Tag>(
      static_cast<std::uint32_t>(value.as<double>()));
}

template <typename Tag>
auto toHandleVector(const val& value) -> std::vector<ir::Handle<Tag>> {
  std::vector<ir::Handle<Tag>> result;
  if (!value.isArray()) return result;
  const auto size = value["length"].as<unsigned>();
  result.reserve(size);
  for (unsigned i = 0; i < size; ++i) result.push_back(toHandle<Tag>(value[i]));
  return result;
}

}  // namespace cxx::js
