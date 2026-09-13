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
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class ByteWriter {
 public:
  void u8(std::uint8_t value) { data_.push_back(value); }

  void boolean(bool value) { u8(value ? 1 : 0); }

  void u32(std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
      data_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }

  void u64(std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8)
      data_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }

  void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }

  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }

  void f32(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }

  void f64(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  void f80(long double value) {
    const auto bytes = reinterpret_cast<const std::uint8_t*>(&value);
    u8(static_cast<std::uint8_t>(sizeof(long double)));
    for (std::size_t i = 0; i < sizeof(long double); ++i) u8(bytes[i]);
  }

  /**
   * A reference, an ordinal, an enumerator and a container count are all small
   * almost all of the time, so they travel as LEB128 rather than as four fixed
   * bytes. The envelope and the section directory stay fixed-width because they
   * are patched after the fact.
   */
  void varU32(std::uint32_t value) {
    while (value >= 0x80) {
      data_.push_back(static_cast<std::uint8_t>(value) | 0x80);
      value >>= 7;
    }
    data_.push_back(static_cast<std::uint8_t>(value));
  }

  void varU64(std::uint64_t value) {
    while (value >= 0x80) {
      data_.push_back(static_cast<std::uint8_t>(value) | 0x80);
      value >>= 7;
    }
    data_.push_back(static_cast<std::uint8_t>(value));
  }

  void varI64(std::int64_t value) {
    const auto zigzag = (static_cast<std::uint64_t>(value) << 1) ^
                        static_cast<std::uint64_t>(value >> 63);
    varU64(zigzag);
  }

  void varI32(std::int32_t value) { varI64(value); }

  void str(std::string_view value) {
    varU32(static_cast<std::uint32_t>(value.size()));
    data_.insert(data_.end(), value.begin(), value.end());
  }

  void bytes(const std::vector<std::uint8_t>& value) {
    varU32(static_cast<std::uint32_t>(value.size()));
    data_.insert(data_.end(), value.begin(), value.end());
  }

  [[nodiscard]] auto size() const -> std::size_t { return data_.size(); }

  [[nodiscard]] auto view() const -> const std::vector<std::uint8_t>& {
    return data_;
  }

  void patchU32(std::size_t offset, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      data_[offset++] = static_cast<std::uint8_t>((value >> shift) & 0xff);
    }
  }

  [[nodiscard]] auto reserveU32() -> std::size_t {
    const auto offset = data_.size();
    u32(0);
    return offset;
  }

  void append(const std::vector<std::uint8_t>& value) {
    data_.insert(data_.end(), value.begin(), value.end());
  }

  void append(const std::vector<std::uint8_t>& value, std::size_t offset,
              std::size_t size) {
    data_.insert(data_.end(),
                 value.begin() + static_cast<std::ptrdiff_t>(offset),
                 value.begin() + static_cast<std::ptrdiff_t>(offset + size));
  }

  /** Keeps the capacity, so one writer can serve every record in turn. */
  void clear() { data_.clear(); }

  void reserve(std::size_t capacity) { data_.reserve(capacity); }

  [[nodiscard]] auto take() -> std::vector<std::uint8_t> {
    return std::move(data_);
  }

 private:
  std::vector<std::uint8_t> data_;
};

class ByteReader {
 public:
  ByteReader() = default;

  explicit ByteReader(std::span<const std::uint8_t> data) : data_(data) {}

  [[nodiscard]] auto ok() const -> bool { return ok_; }

  void fail() { ok_ = false; }

  [[nodiscard]] auto position() const -> std::size_t { return pos_; }

  [[nodiscard]] auto remaining() const -> std::size_t {
    return pos_ <= data_.size() ? data_.size() - pos_ : 0;
  }

  [[nodiscard]] auto atEnd() const -> bool { return remaining() == 0; }

  /** Everything not consumed yet, viewed in place. */
  [[nodiscard]] auto rest() const -> std::span<const std::uint8_t> {
    return pos_ <= data_.size() ? data_.subspan(pos_)
                                : std::span<const std::uint8_t>{};
  }

  [[nodiscard]] auto subReader(std::size_t offset, std::size_t size)
      -> ByteReader {
    if (offset > data_.size() || size > data_.size() - offset) {
      ok_ = false;
      return ByteReader{};
    }
    return ByteReader{data_.subspan(offset, size)};
  }

  [[nodiscard]] auto u8() -> std::uint8_t {
    if (remaining() < 1) {
      ok_ = false;
      return 0;
    }
    return data_[pos_++];
  }

  [[nodiscard]] auto boolean() -> bool { return u8() != 0; }

  [[nodiscard]] auto u32() -> std::uint32_t {
    if (remaining() < 4) {
      ok_ = false;
      return 0;
    }
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8)
      value |= static_cast<std::uint32_t>(data_[pos_++]) << shift;
    return value;
  }

  [[nodiscard]] auto u64() -> std::uint64_t {
    if (remaining() < 8) {
      ok_ = false;
      return 0;
    }
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8)
      value |= static_cast<std::uint64_t>(data_[pos_++]) << shift;
    return value;
  }

  [[nodiscard]] auto i32() -> std::int32_t {
    return static_cast<std::int32_t>(u32());
  }

  [[nodiscard]] auto i64() -> std::int64_t {
    return static_cast<std::int64_t>(u64());
  }

  [[nodiscard]] auto f32() -> float {
    const auto bits = u32();
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  [[nodiscard]] auto f64() -> double {
    const auto bits = u64();
    double value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  [[nodiscard]] auto f80() -> long double {
    const auto width = u8();
    if (width != sizeof(long double)) {
      ok_ = false;
      for (std::uint8_t i = 0; ok_ && i < width; ++i) (void)u8();
      return 0;
    }
    if (remaining() < width) {
      ok_ = false;
      return 0;
    }
    long double value = 0;
    std::memcpy(&value, data_.data() + pos_, width);
    pos_ += width;
    return value;
  }

  [[nodiscard]] auto varU32() -> std::uint32_t {
    if (pos_ < data_.size()) {
      const auto byte = data_[pos_];
      if ((byte & 0x80) == 0) {
        ++pos_;
        return byte;
      }
    }
    const auto value = varU64Wide();
    if (value > std::numeric_limits<std::uint32_t>::max()) {
      ok_ = false;
      return 0;
    }
    return static_cast<std::uint32_t>(value);
  }

  [[nodiscard]] auto varU64() -> std::uint64_t {
    if (pos_ < data_.size()) {
      const auto byte = data_[pos_];
      if ((byte & 0x80) == 0) {
        ++pos_;
        return byte;
      }
    }
    return varU64Wide();
  }

  [[nodiscard]] auto varI64() -> std::int64_t {
    const auto zigzag = varU64();
    return static_cast<std::int64_t>((zigzag >> 1) ^ (~(zigzag & 1) + 1));
  }

  [[nodiscard]] auto varI32() -> std::int32_t {
    return static_cast<std::int32_t>(varI64());
  }

  [[nodiscard]] auto varCount(std::size_t elementSizeLowerBound)
      -> std::uint32_t {
    const auto value = varU32();
    if (!ok_) return 0;
    if (elementSizeLowerBound &&
        value > remaining() / elementSizeLowerBound + 1) {
      ok_ = false;
      return 0;
    }
    return value;
  }

  [[nodiscard]] auto str() -> std::string {
    const auto size = varU32();
    if (!ok_ || remaining() < size) {
      ok_ = false;
      return {};
    }
    std::string value(reinterpret_cast<const char*>(data_.data() + pos_), size);
    pos_ += size;
    return value;
  }

  [[nodiscard]] auto bytes() -> std::vector<std::uint8_t> {
    const auto value = byteSpan();
    return std::vector<std::uint8_t>(value.begin(), value.end());
  }

  /**
   * The same length-prefixed block as `bytes`, viewed in place: the reader does
   * not own the buffer, so the span lives as long as the data it was given.
   */
  [[nodiscard]] auto byteSpan() -> std::span<const std::uint8_t> {
    const auto size = varU32();
    if (!ok_ || remaining() < size) {
      ok_ = false;
      return {};
    }
    auto value = data_.subspan(pos_, size);
    pos_ += size;
    return value;
  }

  [[nodiscard]] auto count(std::size_t elementSizeLowerBound) -> std::uint32_t {
    const auto value = u32();
    if (!ok_) return 0;
    if (elementSizeLowerBound &&
        value > remaining() / elementSizeLowerBound + 1) {
      ok_ = false;
      return 0;
    }
    return value;
  }

 private:
  [[nodiscard]] auto varU64Wide() -> std::uint64_t {
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (pos_ >= data_.size()) {
        ok_ = false;
        return 0;
      }
      const auto byte = data_[pos_++];
      value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return value;
    }
    ok_ = false;
    return 0;
  }

  std::span<const std::uint8_t> data_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

}  // namespace cxx
