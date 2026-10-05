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

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cxx::linker {

using Bytes = std::vector<std::uint8_t>;

template <typename... Parts>
[[nodiscard]] auto concat(const Parts&... parts) -> std::string {
  std::string result;
  (result.append(parts), ...);
  return result;
}
using ByteView = std::span<const std::uint8_t>;

class Reader {
 public:
  explicit Reader(ByteView bytes) : bytes_(bytes) {}

  [[nodiscard]] auto ok() const -> bool { return ok_; }
  [[nodiscard]] auto position() const -> std::size_t { return position_; }

  [[nodiscard]] auto remaining() const -> std::size_t {
    return bytes_.size() - position_;
  }

  void seek(std::size_t position);
  void skip(std::size_t count);

  [[nodiscard]] auto u8() -> std::uint8_t;
  [[nodiscard]] auto u32() -> std::uint32_t;
  [[nodiscard]] auto uleb() -> std::uint32_t;
  [[nodiscard]] auto sleb() -> std::int64_t;
  [[nodiscard]] auto bytes(std::size_t count) -> ByteView;
  [[nodiscard]] auto string() -> std::string;

 private:
  ByteView bytes_;
  std::size_t position_ = 0;
  bool ok_ = true;
};

class Buffer {
 public:
  [[nodiscard]] auto size() const -> std::size_t { return bytes_.size(); }
  [[nodiscard]] auto view() const -> ByteView { return bytes_; }
  [[nodiscard]] auto data() -> std::uint8_t* { return bytes_.data(); }

  void u8(std::uint8_t value) { bytes_.push_back(value); }
  void uleb(std::uint32_t value);
  void sleb(std::int64_t value);
  void string(std::string_view text);
  void bytes(ByteView data);
  void fixed(std::uint64_t value, std::size_t width);
  void zeros(std::size_t count) { bytes_.resize(bytes_.size() + count); }
  void section(std::uint8_t id, const Buffer& payload);

 private:
  Bytes bytes_;
};

constexpr std::size_t kPaddedLebSize = 5;

void patchUleb(std::uint8_t* at, std::uint32_t value);
void patchSleb(std::uint8_t* at, std::int32_t value);
void patchU32(std::uint8_t* at, std::uint32_t value);

[[nodiscard]] auto ulebSize(std::uint32_t value) -> std::size_t;

}  // namespace cxx::linker
