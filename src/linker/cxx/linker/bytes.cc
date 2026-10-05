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

#include <cxx/linker/bytes.h>

#include <cstring>

namespace cxx::linker {

void Reader::seek(std::size_t position) {
  position_ = position;
  ok_ = ok_ && position <= bytes_.size();
}

void Reader::skip(std::size_t count) { seek(position_ + count); }

auto Reader::u8() -> std::uint8_t {
  if (position_ >= bytes_.size()) {
    ok_ = false;
    return 0;
  }
  return bytes_[position_++];
}

auto Reader::u32() -> std::uint32_t {
  const auto data = bytes(4);
  std::uint32_t value = 0;
  if (ok_) std::memcpy(&value, data.data(), 4);
  return value;
}

auto Reader::uleb() -> std::uint32_t {
  std::uint64_t value = 0;
  for (int shift = 0; shift < 35; shift += 7) {
    const auto byte = u8();
    value |= std::uint64_t{byte & 0x7fu} << shift;
    if (!(byte & 0x80)) return static_cast<std::uint32_t>(value);
  }
  ok_ = false;
  return 0;
}

auto Reader::sleb() -> std::int64_t {
  std::uint64_t value = 0;
  for (int shift = 0; shift < 70; shift += 7) {
    const auto byte = u8();
    value |= std::uint64_t{byte & 0x7fu} << shift;
    if (byte & 0x80) continue;
    const bool negative = byte & 0x40;
    if (negative && shift + 7 < 64) value |= ~std::uint64_t{0} << (shift + 7);
    return static_cast<std::int64_t>(value);
  }
  ok_ = false;
  return 0;
}

auto Reader::bytes(std::size_t count) -> ByteView {
  if (count > remaining()) {
    ok_ = false;
    return {};
  }
  const auto result = bytes_.subspan(position_, count);
  position_ += count;
  return result;
}

auto Reader::string() -> std::string {
  const auto data = bytes(uleb());
  return {data.begin(), data.end()};
}

void Buffer::uleb(std::uint32_t value) {
  do {
    const auto byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    bytes_.push_back(value ? byte | 0x80 : byte);
  } while (value);
}

void Buffer::sleb(std::int64_t value) {
  for (bool more = true; more;) {
    const auto byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    more = !((value == 0 && !(byte & 0x40)) || (value == -1 && (byte & 0x40)));
    bytes_.push_back(more ? byte | 0x80 : byte);
  }
}

void Buffer::string(std::string_view text) {
  uleb(static_cast<std::uint32_t>(text.size()));
  bytes_.insert(bytes_.end(), text.begin(), text.end());
}

void Buffer::bytes(ByteView data) {
  bytes_.insert(bytes_.end(), data.begin(), data.end());
}

void Buffer::fixed(std::uint64_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    bytes_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

void Buffer::section(std::uint8_t id, const Buffer& payload) {
  u8(id);
  uleb(static_cast<std::uint32_t>(payload.size()));
  bytes(payload.view());
}

void patchUleb(std::uint8_t* at, std::uint32_t value) {
  for (std::size_t i = 0; i + 1 < kPaddedLebSize; ++i) {
    at[i] = static_cast<std::uint8_t>(value & 0x7f) | 0x80;
    value >>= 7;
  }
  at[kPaddedLebSize - 1] = static_cast<std::uint8_t>(value & 0x7f);
}

void patchSleb(std::uint8_t* at, std::int32_t value) {
  for (std::size_t i = 0; i + 1 < kPaddedLebSize; ++i) {
    at[i] = static_cast<std::uint8_t>(value & 0x7f) | 0x80;
    value >>= 7;
  }
  at[kPaddedLebSize - 1] = static_cast<std::uint8_t>(value & 0x7f);
}

void patchU32(std::uint8_t* at, std::uint32_t value) {
  std::memcpy(at, &value, 4);
}

auto ulebSize(std::uint32_t value) -> std::size_t {
  std::size_t size = 1;
  for (; value >= 0x80; value >>= 7) ++size;
  return size;
}

}  // namespace cxx::linker
