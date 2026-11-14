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

#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cxx {

class Int128;

class UInt128 {
 public:
  constexpr UInt128() = default;

  template <std::integral T>
  constexpr UInt128(T value)
      : low_(static_cast<std::uint64_t>(value)),
        high_(std::is_signed_v<T> && value < 0 ? ~std::uint64_t{0} : 0) {}

  template <std::floating_point T>
  explicit UInt128(T value) {
    constexpr auto kLowScale = static_cast<T>(18446744073709551616.0);
    const auto truncated = std::trunc(value);
    if (truncated < kLowScale) {
      low_ = static_cast<std::uint64_t>(truncated);
      return;
    }
    const auto high = std::trunc(truncated / kLowScale);
    high_ = static_cast<std::uint64_t>(high);
    low_ = static_cast<std::uint64_t>(truncated - high * kLowScale);
  }

  constexpr explicit UInt128(const Int128& value);

  [[nodiscard]] static constexpr auto fromParts(std::uint64_t high,
                                                std::uint64_t low) -> UInt128 {
    UInt128 result;
    result.high_ = high;
    result.low_ = low;
    return result;
  }

  [[nodiscard]] constexpr auto low() const -> std::uint64_t { return low_; }
  [[nodiscard]] constexpr auto high() const -> std::uint64_t { return high_; }

  constexpr explicit operator bool() const { return (low_ | high_) != 0; }

  template <std::integral T>
    requires(!std::same_as<T, bool>)
  constexpr explicit operator T() const {
    return static_cast<T>(low_);
  }

  template <std::floating_point T>
  explicit operator T() const {
    static_assert(std::numeric_limits<T>::digits <= 62);
    if (!high_) return static_cast<T>(low_);
    const int shift = 64 - std::countl_zero(high_);
    const auto top = (*this >> shift).low_;
    const auto discarded = *this & ((UInt128{1} << shift) - 1);
    const auto sticky = static_cast<bool>(discarded) ? 1u : 0u;
    return std::ldexp(static_cast<T>(top | sticky), shift);
  }

  friend constexpr auto operator==(const UInt128&, const UInt128&)
      -> bool = default;

  friend constexpr auto operator<=>(const UInt128& left, const UInt128& right)
      -> std::strong_ordering {
    if (left.high_ != right.high_) return left.high_ <=> right.high_;
    return left.low_ <=> right.low_;
  }

  friend constexpr auto operator+(const UInt128& left, const UInt128& right)
      -> UInt128 {
    const auto low = left.low_ + right.low_;
    const auto carry = low < left.low_ ? 1u : 0u;
    return fromParts(left.high_ + right.high_ + carry, low);
  }

  friend constexpr auto operator-(const UInt128& left, const UInt128& right)
      -> UInt128 {
    const auto borrow = left.low_ < right.low_ ? 1u : 0u;
    return fromParts(left.high_ - right.high_ - borrow, left.low_ - right.low_);
  }

  friend constexpr auto operator*(const UInt128& left, const UInt128& right)
      -> UInt128 {
    auto product = widening_product(left.low_, right.low_);
    product.high_ += left.low_ * right.high_ + left.high_ * right.low_;
    return product;
  }

  friend constexpr auto operator/(const UInt128& left, const UInt128& right)
      -> UInt128;

  friend constexpr auto operator%(const UInt128& left, const UInt128& right)
      -> UInt128;

  friend constexpr auto operator&(const UInt128& left, const UInt128& right)
      -> UInt128 {
    return fromParts(left.high_ & right.high_, left.low_ & right.low_);
  }

  friend constexpr auto operator|(const UInt128& left, const UInt128& right)
      -> UInt128 {
    return fromParts(left.high_ | right.high_, left.low_ | right.low_);
  }

  friend constexpr auto operator^(const UInt128& left, const UInt128& right)
      -> UInt128 {
    return fromParts(left.high_ ^ right.high_, left.low_ ^ right.low_);
  }

  friend constexpr auto operator<<(const UInt128& value, int count) -> UInt128 {
    if (count <= 0) return value;
    if (count >= 128) return {};
    if (count >= 64) return fromParts(value.low_ << (count - 64), 0);
    return fromParts(value.high_ << count | value.low_ >> (64 - count),
                     value.low_ << count);
  }

  friend constexpr auto operator>>(const UInt128& value, int count) -> UInt128 {
    if (count <= 0) return value;
    if (count >= 128) return {};
    if (count >= 64) return fromParts(0, value.high_ >> (count - 64));
    return fromParts(value.high_ >> count,
                     value.low_ >> count | value.high_ << (64 - count));
  }

  template <std::integral T>
  friend constexpr auto operator<<(const UInt128& value, T count) -> UInt128 {
    return value << static_cast<int>(count);
  }

  template <std::integral T>
  friend constexpr auto operator>>(const UInt128& value, T count) -> UInt128 {
    return value >> static_cast<int>(count);
  }

  constexpr auto operator~() const -> UInt128 {
    return fromParts(~high_, ~low_);
  }

  constexpr auto operator-() const -> UInt128 { return UInt128{} - *this; }

  constexpr auto operator++() -> UInt128& { return *this = *this + 1; }
  constexpr auto operator--() -> UInt128& { return *this = *this - 1; }
  constexpr auto operator++(int) -> UInt128 {
    auto previous = *this;
    ++*this;
    return previous;
  }
  constexpr auto operator--(int) -> UInt128 {
    auto previous = *this;
    --*this;
    return previous;
  }

  constexpr auto operator+=(const UInt128& other) -> UInt128& {
    return *this = *this + other;
  }
  constexpr auto operator-=(const UInt128& other) -> UInt128& {
    return *this = *this - other;
  }
  constexpr auto operator*=(const UInt128& other) -> UInt128& {
    return *this = *this * other;
  }
  constexpr auto operator/=(const UInt128& other) -> UInt128& {
    return *this = *this / other;
  }
  constexpr auto operator%=(const UInt128& other) -> UInt128& {
    return *this = *this % other;
  }
  constexpr auto operator&=(const UInt128& other) -> UInt128& {
    return *this = *this & other;
  }
  constexpr auto operator|=(const UInt128& other) -> UInt128& {
    return *this = *this | other;
  }
  constexpr auto operator^=(const UInt128& other) -> UInt128& {
    return *this = *this ^ other;
  }
  constexpr auto operator<<=(int count) -> UInt128& {
    return *this = *this << count;
  }
  constexpr auto operator>>=(int count) -> UInt128& {
    return *this = *this >> count;
  }

 private:
  struct Division;

  [[nodiscard]] static constexpr auto widening_product(std::uint64_t left,
                                                       std::uint64_t right)
      -> UInt128 {
    constexpr std::uint64_t kHalfMask = 0xffffffff;
    const auto ll = (left & kHalfMask) * (right & kHalfMask);
    const auto lh = (left & kHalfMask) * (right >> 32);
    const auto hl = (left >> 32) * (right & kHalfMask);
    const auto hh = (left >> 32) * (right >> 32);
    const auto middle = (ll >> 32) + (lh & kHalfMask) + (hl & kHalfMask);
    return fromParts(hh + (lh >> 32) + (hl >> 32) + (middle >> 32),
                     (middle << 32) | (ll & kHalfMask));
  }

  [[nodiscard]] static constexpr auto divide(const UInt128& dividend,
                                             const UInt128& divisor)
      -> Division;

  std::uint64_t low_ = 0;
  std::uint64_t high_ = 0;
};

struct UInt128::Division {
  UInt128 quotient;
  UInt128 remainder;
};

constexpr auto UInt128::divide(const UInt128& dividend, const UInt128& divisor)
    -> Division {
  Division result;
  if (!divisor) return result;
  if (!dividend.high_ && !divisor.high_) {
    result.quotient = dividend.low_ / divisor.low_;
    result.remainder = dividend.low_ % divisor.low_;
    return result;
  }
  for (int bit = 127; bit >= 0; --bit) {
    result.remainder = result.remainder << 1 | ((dividend >> bit) & 1);
    if (result.remainder < divisor) continue;
    result.remainder -= divisor;
    result.quotient |= UInt128{1} << bit;
  }
  return result;
}

constexpr auto operator/(const UInt128& left, const UInt128& right) -> UInt128 {
  return UInt128::divide(left, right).quotient;
}

constexpr auto operator%(const UInt128& left, const UInt128& right) -> UInt128 {
  return UInt128::divide(left, right).remainder;
}

class Int128 {
 public:
  constexpr Int128() = default;

  template <std::integral T>
  constexpr Int128(T value) : bits_(value) {}

  template <std::floating_point T>
  explicit Int128(T value)
      : bits_(std::signbit(value) ? -UInt128{-value} : UInt128{value}) {}

  constexpr explicit Int128(const UInt128& bits) : bits_(bits) {}

  [[nodiscard]] constexpr auto bits() const -> UInt128 { return bits_; }

  [[nodiscard]] constexpr auto isNegative() const -> bool {
    return (bits_.high() >> 63) != 0;
  }

  [[nodiscard]] constexpr auto magnitude() const -> UInt128 {
    return isNegative() ? -bits_ : bits_;
  }

  constexpr explicit operator bool() const { return static_cast<bool>(bits_); }

  template <std::integral T>
    requires(!std::same_as<T, bool>)
  constexpr explicit operator T() const {
    return static_cast<T>(bits_.low());
  }

  template <std::floating_point T>
  explicit operator T() const {
    const auto result = static_cast<T>(magnitude());
    return isNegative() ? -result : result;
  }

  friend constexpr auto operator==(const Int128&, const Int128&)
      -> bool = default;

  friend constexpr auto operator<=>(const Int128& left, const Int128& right)
      -> std::strong_ordering {
    if (left.isNegative() != right.isNegative())
      return left.isNegative() ? std::strong_ordering::less
                               : std::strong_ordering::greater;
    return left.bits_ <=> right.bits_;
  }

  friend constexpr auto operator+(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ + right.bits_};
  }

  friend constexpr auto operator-(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ - right.bits_};
  }

  friend constexpr auto operator*(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ * right.bits_};
  }

  friend constexpr auto operator/(const Int128& left, const Int128& right)
      -> Int128 {
    const auto quotient = Int128{left.magnitude() / right.magnitude()};
    return left.isNegative() != right.isNegative() ? -quotient : quotient;
  }

  friend constexpr auto operator%(const Int128& left, const Int128& right)
      -> Int128 {
    const auto remainder = Int128{left.magnitude() % right.magnitude()};
    return left.isNegative() ? -remainder : remainder;
  }

  friend constexpr auto operator&(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ & right.bits_};
  }

  friend constexpr auto operator|(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ | right.bits_};
  }

  friend constexpr auto operator^(const Int128& left, const Int128& right)
      -> Int128 {
    return Int128{left.bits_ ^ right.bits_};
  }

  friend constexpr auto operator<<(const Int128& value, int count) -> Int128 {
    return Int128{value.bits_ << count};
  }

  friend constexpr auto operator>>(const Int128& value, int count) -> Int128 {
    if (!value.isNegative()) return Int128{value.bits_ >> count};
    return Int128{~(~value.bits_ >> count)};
  }

  template <std::integral T>
  friend constexpr auto operator<<(const Int128& value, T count) -> Int128 {
    return value << static_cast<int>(count);
  }

  template <std::integral T>
  friend constexpr auto operator>>(const Int128& value, T count) -> Int128 {
    return value >> static_cast<int>(count);
  }

  constexpr auto operator~() const -> Int128 { return Int128{~bits_}; }

  constexpr auto operator-() const -> Int128 { return Int128{-bits_}; }

  constexpr auto operator++() -> Int128& { return *this = *this + 1; }
  constexpr auto operator--() -> Int128& { return *this = *this - 1; }
  constexpr auto operator++(int) -> Int128 {
    auto previous = *this;
    ++*this;
    return previous;
  }
  constexpr auto operator--(int) -> Int128 {
    auto previous = *this;
    --*this;
    return previous;
  }

  constexpr auto operator+=(const Int128& other) -> Int128& {
    return *this = *this + other;
  }
  constexpr auto operator-=(const Int128& other) -> Int128& {
    return *this = *this - other;
  }
  constexpr auto operator*=(const Int128& other) -> Int128& {
    return *this = *this * other;
  }
  constexpr auto operator/=(const Int128& other) -> Int128& {
    return *this = *this / other;
  }
  constexpr auto operator%=(const Int128& other) -> Int128& {
    return *this = *this % other;
  }
  constexpr auto operator&=(const Int128& other) -> Int128& {
    return *this = *this & other;
  }
  constexpr auto operator|=(const Int128& other) -> Int128& {
    return *this = *this | other;
  }
  constexpr auto operator^=(const Int128& other) -> Int128& {
    return *this = *this ^ other;
  }
  constexpr auto operator<<=(int count) -> Int128& {
    return *this = *this << count;
  }
  constexpr auto operator>>=(int count) -> Int128& {
    return *this = *this >> count;
  }

 private:
  UInt128 bits_;
};

constexpr UInt128::UInt128(const Int128& value) : UInt128(value.bits()) {}

}  // namespace cxx

template <>
class std::numeric_limits<cxx::UInt128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = false;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr int digits = 128;
  static constexpr int radix = 2;

  [[nodiscard]] static constexpr auto min() -> cxx::UInt128 { return {}; }
  [[nodiscard]] static constexpr auto lowest() -> cxx::UInt128 { return {}; }
  [[nodiscard]] static constexpr auto max() -> cxx::UInt128 {
    return ~cxx::UInt128{};
  }
};

template <>
class std::numeric_limits<cxx::Int128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr int digits = 127;
  static constexpr int radix = 2;

  [[nodiscard]] static constexpr auto min() -> cxx::Int128 {
    return cxx::Int128{cxx::UInt128{1} << 127};
  }
  [[nodiscard]] static constexpr auto lowest() -> cxx::Int128 { return min(); }
  [[nodiscard]] static constexpr auto max() -> cxx::Int128 {
    return cxx::Int128{~(cxx::UInt128{1} << 127)};
  }
};
