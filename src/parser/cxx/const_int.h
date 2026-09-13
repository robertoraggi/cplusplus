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
#include <cstdint>
#include <optional>

namespace cxx {

class ConstInt {
 public:
#if defined(__SIZEOF_INT128__)
  using Wide = __int128;
  using UWide = unsigned __int128;
#else
  using Wide = std::int64_t;
  using UWide = std::uint64_t;
#endif

  using Narrow = std::int64_t;
  using UNarrow = std::uint64_t;

  static constexpr int narrowWidth = static_cast<int>(sizeof(Narrow) * 8);
  static constexpr int maxWidth = static_cast<int>(sizeof(Wide) * 8);

  static constexpr auto isRepresentableWidth(int width) -> bool {
    return width > 0 && width <= maxWidth;
  }

  ConstInt() = default;

  static auto make(Wide value, int width, bool isSigned)
      -> std::optional<ConstInt> {
    if (!isRepresentableWidth(width)) return std::nullopt;
    ConstInt result;
    result.width_ = static_cast<std::uint8_t>(width);
    result.isSigned_ = isSigned;
    result.store(value);
    return result;
  }

  [[nodiscard]] auto width() const -> int { return width_; }
  [[nodiscard]] auto isSigned() const -> bool { return isSigned_; }
  [[nodiscard]] auto isNarrow() const -> bool { return width_ <= narrowWidth; }
  [[nodiscard]] auto isZero() const -> bool { return toUWide() == 0; }
  [[nodiscard]] auto isNegative() const -> bool {
    return isSigned_ && toWide() < 0;
  }

  [[nodiscard]] auto toWide() const -> Wide {
    return isNarrow() ? static_cast<Wide>(narrow_) : wide_;
  }

  [[nodiscard]] auto toUWide() const -> UWide {
    return static_cast<UWide>(toWide()) & widthMask();
  }

  [[nodiscard]] auto toIntMax() const -> std::intmax_t {
    return static_cast<std::intmax_t>(toWide());
  }

  [[nodiscard]] auto toUIntMax() const -> std::uintmax_t {
    return static_cast<std::uintmax_t>(toUWide());
  }

  [[nodiscard]] auto popcount() const -> int {
    auto bits = toUWide();
    if constexpr (maxWidth > narrowWidth) {
      return std::popcount(low(bits)) + std::popcount(high(bits));
    } else {
      return std::popcount(low(bits));
    }
  }

  [[nodiscard]] auto countTrailingZeros() const -> int {
    if (isZero()) return width_;
    auto bits = toUWide();
    if constexpr (maxWidth > narrowWidth) {
      if (auto lo = low(bits)) return std::countr_zero(lo);
      return narrowWidth + std::countr_zero(high(bits));
    } else {
      return std::countr_zero(low(bits));
    }
  }

  [[nodiscard]] auto countLeadingZeros() const -> int {
    if (isZero()) return width_;
    auto bits = toUWide();
    const auto leadingInStorage = [&]() -> int {
      if constexpr (maxWidth > narrowWidth) {
        if (auto hi = high(bits)) return std::countl_zero(hi);
        return narrowWidth + std::countl_zero(low(bits));
      } else {
        return std::countl_zero(low(bits));
      }
    }();
    return leadingInStorage - (maxWidth - width_);
  }

  friend auto operator+(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() + r.toUWide()));
  }

  friend auto operator-(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() - r.toUWide()));
  }

  friend auto operator*(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() * r.toUWide()));
  }

  friend auto operator/(ConstInt l, ConstInt r) -> ConstInt {
    if (l.isSigned_) return l.rebuild(l.toWide() / r.toWide());
    return l.rebuild(static_cast<Wide>(l.toUWide() / r.toUWide()));
  }

  friend auto operator%(ConstInt l, ConstInt r) -> ConstInt {
    if (l.isSigned_) return l.rebuild(l.toWide() % r.toWide());
    return l.rebuild(static_cast<Wide>(l.toUWide() % r.toUWide()));
  }

  friend auto operator<<(ConstInt l, ConstInt r) -> ConstInt {
    auto shift = r.toUIntMax();
    if (shift >= static_cast<std::uintmax_t>(l.width_)) return l.rebuild(0);
    return l.rebuild(static_cast<Wide>(l.toUWide() << shift));
  }

  friend auto operator>>(ConstInt l, ConstInt r) -> ConstInt {
    auto shift = r.toUIntMax();
    if (shift >= static_cast<std::uintmax_t>(l.width_))
      return l.rebuild(l.isNegative() ? -1 : 0);
    if (l.isSigned_) return l.rebuild(l.toWide() >> shift);
    return l.rebuild(static_cast<Wide>(l.toUWide() >> shift));
  }

  friend auto operator&(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() & r.toUWide()));
  }

  friend auto operator|(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() | r.toUWide()));
  }

  friend auto operator^(ConstInt l, ConstInt r) -> ConstInt {
    return l.rebuild(static_cast<Wide>(l.toUWide() ^ r.toUWide()));
  }

  auto operator~() const -> ConstInt {
    return rebuild(static_cast<Wide>(~toUWide()));
  }

  auto operator-() const -> ConstInt {
    return rebuild(static_cast<Wide>(UWide{0} - toUWide()));
  }

  auto operator==(const ConstInt& other) const -> bool {
    return toUWide() == other.toUWide();
  }

  auto operator<=>(const ConstInt& other) const -> std::strong_ordering {
    if (isSigned_) {
      auto l = toWide();
      auto r = other.toWide();
      if (l < r) return std::strong_ordering::less;
      if (l > r) return std::strong_ordering::greater;
      return std::strong_ordering::equal;
    }
    auto l = toUWide();
    auto r = other.toUWide();
    if (l < r) return std::strong_ordering::less;
    if (l > r) return std::strong_ordering::greater;
    return std::strong_ordering::equal;
  }

 private:
  static auto low(UWide bits) -> UNarrow { return static_cast<UNarrow>(bits); }

  static auto high(UWide bits) -> UNarrow {
    if constexpr (maxWidth > narrowWidth) {
      return static_cast<UNarrow>(bits >> (narrowWidth / 2) >>
                                  (narrowWidth / 2));
    } else {
      return 0;
    }
  }

  [[nodiscard]] auto widthMask() const -> UWide {
    if (width_ >= maxWidth) return ~UWide{0};
    return (UWide{1} << width_) - 1;
  }

  [[nodiscard]] auto rebuild(Wide value) const -> ConstInt {
    ConstInt result;
    result.width_ = width_;
    result.isSigned_ = isSigned_;
    result.store(value);
    return result;
  }

  void store(Wide value) {
    auto truncated = static_cast<UWide>(value) & widthMask();
    if (isSigned_ && width_ < maxWidth &&
        ((truncated >> (width_ - 1)) & 1) != 0) {
      truncated |= ~widthMask();
    }
    auto normalized = static_cast<Wide>(truncated);
    if (isNarrow()) {
      narrow_ = static_cast<Narrow>(normalized);
    } else {
      wide_ = normalized;
    }
  }

  union {
    Narrow narrow_{};
    Wide wide_;
  };

  std::uint8_t width_ = narrowWidth;
  bool isSigned_ = true;
};

}  // namespace cxx
