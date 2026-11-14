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

#include <cxx/device_profile_limits.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <compare>
#include <cstdlib>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cxx {
namespace {

[[nodiscard]] auto pointerBits(TripleArch arch) -> std::size_t {
  if (arch == TripleArch::kWasm32 || arch == TripleArch::kSPIRV32) return 32;
  return 64;
}

constexpr FloatingPointFormat kBinary128Format{.exponentBits = 15,
                                               .significandDigits = 113};

struct FloatingPointFormatOf {
  const MemoryLayout& memoryLayout;

  [[nodiscard]] auto operator()(const HalfType*) const
      -> std::optional<FloatingPointFormat> {
    return FloatingPointFormat{.exponentBits = 5, .significandDigits = 11};
  }

  [[nodiscard]] auto operator()(const Float16Type*) const
      -> std::optional<FloatingPointFormat> {
    return FloatingPointFormat{.exponentBits = 5, .significandDigits = 11};
  }

  [[nodiscard]] auto operator()(const FloatType*) const
      -> std::optional<FloatingPointFormat> {
    return FloatingPointFormat{.exponentBits = 8, .significandDigits = 24};
  }

  [[nodiscard]] auto operator()(const DoubleType*) const
      -> std::optional<FloatingPointFormat> {
    return FloatingPointFormat{.exponentBits = 11, .significandDigits = 53};
  }

  [[nodiscard]] auto operator()(const LongDoubleType*) const
      -> std::optional<FloatingPointFormat> {
    switch (memoryLayout.longDoubleMantissaDigits()) {
      case 53:
        return FloatingPointFormat{.exponentBits = 11, .significandDigits = 53};
      case 64:
        return FloatingPointFormat{.exponentBits = 15,
                                   .significandDigits = 64,
                                   .explicitIntegerBit = true};
      case 113:
        return kBinary128Format;
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] auto operator()(const Float128Type*) const
      -> std::optional<FloatingPointFormat> {
    return kBinary128Format;
  }

  [[nodiscard]] auto operator()(const QualType* type) const
      -> std::optional<FloatingPointFormat> {
    return visit(*this, type->elementType());
  }

  [[nodiscard]] auto operator()(const Type*) const
      -> std::optional<FloatingPointFormat> {
    return std::nullopt;
  }
};

[[nodiscard]] auto storageSizeInBytes(std::size_t numBits) -> std::size_t {
  auto bytes = (numBits + 7) / 8;
  std::size_t result = 1;
  while (result < bytes) result *= 2;
  return result;
}

[[nodiscard]] auto vectorWidthInBits(const MemoryLayout& memoryLayout,
                                     const VectorType* type)
    -> std::optional<std::size_t> {
  if (type->vectorKind() == VectorKind::kExt &&
      type->elementType()->kind() == TypeKind::kBool)
    return type->elementCount();

  auto elementSize = memoryLayout.sizeOf(type->elementType());
  if (!elementSize) return std::nullopt;

  return type->elementCount() * *elementSize * 8;
}

[[nodiscard]] auto atomicWidthInBits(const MemoryLayout& memoryLayout,
                                     const AtomicType* type)
    -> std::optional<std::size_t> {
  auto valueSize = memoryLayout.sizeOf(type->elementType());
  if (!valueSize) return std::nullopt;

  auto width = *valueSize * 8;
  if (!width) return std::size_t{8};
  if (width > memoryLayout.maxAtomicPromoteWidth()) return width;

  return std::bit_ceil(width);
}

struct SizeOf {
  const MemoryLayout& memoryLayout;

  auto operator()(const BuiltinVaListType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const BuiltinMetaInfoType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const VoidType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const NullptrType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const DecltypeAutoType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const AutoType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const BoolType* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const SignedCharType* type) const
      -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const ShortIntType* type) const
      -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const IntType* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const LongIntType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLong();
  }

  auto operator()(const LongLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongLong();
  }

  auto operator()(const Int128Type*) const -> std::optional<std::size_t> {
    return 16;
  }

  auto operator()(const UnsignedCharType* type) const
      -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const UnsignedShortIntType* type) const
      -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const UnsignedIntType* type) const
      -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const UnsignedLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLong();
  }

  auto operator()(const UnsignedLongLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongLong();
  }

  auto operator()(const UnsignedInt128Type*) const
      -> std::optional<std::size_t> {
    return 16;
  }

  auto operator()(const CharType* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const Char8Type* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const Char16Type* type) const -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const Char32Type* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const WideCharType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfWideChar();
  }

  auto operator()(const FloatType* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const DoubleType* type) const -> std::optional<std::size_t> {
    return 8;
  }

  auto operator()(const LongDoubleType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongDouble();
  }

  auto operator()(const HalfType* type) const -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const Float16Type* type) const -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const Float128Type* type) const
      -> std::optional<std::size_t> {
    return 16;
  }

  auto operator()(const QualType* type) const -> std::optional<std::size_t> {
    return visit(*this, type->elementType());
  }

  auto operator()(const BoundedArrayType* type) const
      -> std::optional<std::size_t> {
    auto elementSize = visit(*this, type->elementType());
    if (elementSize.has_value()) return *elementSize * type->size();
    return std::nullopt;
  }

  auto operator()(const UnboundedArrayType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const PointerType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const LvalueReferenceType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const RvalueReferenceType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const FunctionType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const ClassType* type) const -> std::optional<std::size_t> {
    auto classSymbol = type->definition();
    if (!classSymbol->isComplete()) return std::nullopt;
    return classSymbol->sizeInBytes();
  }

  auto operator()(const EnumType* type) const -> std::optional<std::size_t> {
    if (type->underlyingType()) {
      return visit(*this, type->underlyingType());
    }
    return 4;
  }

  auto operator()(const ScopedEnumType* type) const
      -> std::optional<std::size_t> {
    if (type->underlyingType()) {
      return visit(*this, type->underlyingType());
    }
    return 4;
  }

  auto operator()(const MemberObjectPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> std::optional<std::size_t> {
    return 2 * memoryLayout.sizeOfPointer();
  }

  auto operator()(const NamespaceType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const TypeParameterType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const TemplateTypeParameterType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const TemplateTypeParameterSpecializationType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const PackExpansionType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const DecltypeType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedNameType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedPackIndexType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedUnderlyingType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedBuiltinType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const OverloadSetType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const BitIntType* type) const -> std::optional<std::size_t> {
    return storageSizeInBytes(type->numBits());
  }

  auto operator()(const UnsignedBitIntType* type) const
      -> std::optional<std::size_t> {
    return storageSizeInBytes(type->numBits());
  }

  auto operator()(const UnresolvedBitIntType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const VectorType* type) const -> std::optional<std::size_t> {
    auto bits = vectorWidthInBits(memoryLayout, type);
    if (!bits) return std::nullopt;
    return storageSizeInBytes(*bits);
  }

  auto operator()(const UnresolvedVectorType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const ComplexType* type) const -> std::optional<std::size_t> {
    auto elementSize = memoryLayout.sizeOf(type->elementType());
    if (!elementSize) return std::nullopt;
    return *elementSize * 2;
  }

  auto operator()(const AtomicType* type) const -> std::optional<std::size_t> {
    auto width = atomicWidthInBits(memoryLayout, type);
    if (!width) return std::nullopt;
    return *width / 8;
  }

  auto operator()(const SveType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }
};

struct AlignmentOf {
  const MemoryLayout& memoryLayout;

  auto operator()(const QualType* type) const -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const EnumType* type) const -> std::optional<std::size_t> {
    if (type->underlyingType())
      return memoryLayout.alignmentOf(type->underlyingType());
    return 4;
  }

  auto operator()(const ScopedEnumType* type) const
      -> std::optional<std::size_t> {
    if (type->underlyingType())
      return memoryLayout.alignmentOf(type->underlyingType());
    return 4;
  }

  auto operator()(const ClassType* type) const -> std::optional<std::size_t> {
    auto classSymbol = type->definition();
    if (!classSymbol->isComplete()) return std::nullopt;
    return classSymbol->alignment();
  }

  auto operator()(const UnboundedArrayType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const BoundedArrayType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const LongDoubleType* type) const
      -> std::optional<std::size_t> {
    if (memoryLayout.floatingPointFormat(type) == kBinary128Format)
      return memoryLayout.binary128Alignment();
    return memoryLayout.sizeOfLongDouble();
  }

  auto operator()(const Float128Type*) const -> std::optional<std::size_t> {
    return memoryLayout.binary128Alignment();
  }

  auto operator()(const MemberObjectPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const VectorType* type) const -> std::optional<std::size_t> {
    auto size = memoryLayout.sizeOf(type);
    if (!size) return std::nullopt;
    auto maximum = memoryLayout.maxVectorAlignment();
    if (maximum && maximum < *size) return maximum;
    return size;
  }

  auto operator()(const ComplexType* type) const -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const AtomicType* type) const -> std::optional<std::size_t> {
    auto width = atomicWidthInBits(memoryLayout, type);
    if (!width) return std::nullopt;
    if (*width <= memoryLayout.maxAtomicPromoteWidth()) return *width / 8;
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const SveType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(auto type) const -> std::optional<std::size_t> {
    if (!type) return std::nullopt;
    return memoryLayout.sizeOf(type);
  }
};

[[nodiscard]] auto significant_bits(ConstInt::UWide value) -> int {
  int bits = 0;
  for (; value > std::numeric_limits<std::uint32_t>::max(); value >>= 32)
    bits += 32;
  return bits + std::bit_width(static_cast<std::uint32_t>(value));
}

[[nodiscard]] auto round_shift_right(ConstInt::UWide value, int count,
                                     std::strong_ordering excess)
    -> ConstInt::UWide {
  constexpr int kWidth = ConstInt::maxWidth;
  if (count > kWidth) return 0;
  const auto quotient = count == kWidth ? ConstInt::UWide(0) : value >> count;
  const auto remainder =
      count == kWidth ? value : value & ((ConstInt::UWide(1) << count) - 1);
  const auto half = ConstInt::UWide(1) << (count - 1);
  if (remainder > half) return quotient + 1;
  if (remainder < half) return quotient;
  if (excess > 0) return quotient + 1;
  if (excess < 0) return quotient;
  return quotient + (quotient & 1);
}

struct RoundedSignificand {
  int exponent = 0;
  ConstInt::UWide significand = 0;
};

[[nodiscard]] auto round_significand(ConstInt::UWide integer, int exponent,
                                     std::strong_ordering excess,
                                     const FloatingPointFormat& format)
    -> RoundedSignificand {
  const int digits = format.significandDigits;
  const int leadingExponent = exponent + significant_bits(integer) - 1;
  RoundedSignificand rounded;
  rounded.exponent = std::max(leadingExponent, 1 - format.maxExponent());
  const int shift = rounded.exponent - (digits - 1) - exponent;
  rounded.significand = shift <= 0 ? integer << -shift
                                   : round_shift_right(integer, shift, excess);
  if (rounded.significand >> digits) {
    rounded.significand >>= 1;
    ++rounded.exponent;
  }
  return rounded;
}

template <typename Float>
[[nodiscard]] auto round_to_format(
    Float magnitude, const FloatingPointFormat& format,
    std::strong_ordering excess = std::strong_ordering::equal)
    -> RoundedSignificand {
  constexpr int kDigits = std::numeric_limits<Float>::digits;
  int binaryExponent = 0;
  const auto fraction = std::frexp(magnitude, &binaryExponent);
  const auto integer =
      static_cast<ConstInt::UWide>(std::ldexp(fraction, kDigits));
  return round_significand(integer, binaryExponent - kDigits, excess, format);
}

class Magnitude {
 public:
  explicit Magnitude(std::uint32_t value) {
    if (value) limbs_.push_back(value);
  }

  [[nodiscard]] auto isZero() const -> bool { return limbs_.empty(); }

  [[nodiscard]] auto bitWidth() const -> int {
    if (limbs_.empty()) return 0;
    return static_cast<int>(32 * (limbs_.size() - 1)) +
           std::bit_width(limbs_.back());
  }

  void multiplyAdd(std::uint32_t factor, std::uint32_t addend) {
    std::uint64_t carry = addend;
    for (auto& limb : limbs_) {
      const auto product = std::uint64_t(limb) * factor + carry;
      limb = static_cast<std::uint32_t>(product);
      carry = product >> 32;
    }
    if (carry) limbs_.push_back(static_cast<std::uint32_t>(carry));
  }

  void multiplyByPowerOfFive(int count) {
    constexpr std::uint32_t kFiveToThe13 = 1220703125;
    for (; count >= 13; count -= 13) multiplyAdd(kFiveToThe13, 0);
    for (; count > 0; --count) multiplyAdd(5, 0);
  }

  void shiftLeft(int count) {
    const auto words = static_cast<std::size_t>(count / 32);
    const int bits = count % 32;
    if (bits) {
      std::uint32_t carry = 0;
      for (auto& limb : limbs_) {
        const auto shifted = limb << bits | carry;
        carry = limb >> (32 - bits);
        limb = shifted;
      }
      if (carry) limbs_.push_back(carry);
    }
    limbs_.insert(limbs_.begin(), words, 0);
  }

  [[nodiscard]] auto shiftRight(int count) -> bool {
    const auto words =
        std::min(static_cast<std::size_t>(count / 32), limbs_.size());
    const int bits = count % 32;
    bool inexact = false;
    for (std::size_t index = 0; index < words; ++index)
      inexact = inexact || limbs_[index];
    limbs_.erase(limbs_.begin(), limbs_.begin() + words);
    if (bits && !limbs_.empty()) {
      inexact = inexact || (limbs_.front() & ((1u << bits) - 1));
      for (std::size_t index = 0; index < limbs_.size(); ++index) {
        const auto next = index + 1 < limbs_.size() ? limbs_[index + 1] : 0;
        limbs_[index] = limbs_[index] >> bits | next << (32 - bits);
      }
    }
    normalize();
    return inexact;
  }

  [[nodiscard]] auto operator<=>(const Magnitude& other) const
      -> std::strong_ordering {
    if (limbs_.size() != other.limbs_.size())
      return limbs_.size() <=> other.limbs_.size();
    for (auto index = limbs_.size(); index-- > 0;) {
      if (limbs_[index] != other.limbs_[index])
        return limbs_[index] <=> other.limbs_[index];
    }
    return std::strong_ordering::equal;
  }

  void subtract(const Magnitude& other) {
    std::int64_t borrow = 0;
    for (std::size_t index = 0; index < limbs_.size(); ++index) {
      const std::int64_t subtrahend =
          index < other.limbs_.size() ? other.limbs_[index] : 0;
      auto difference = std::int64_t(limbs_[index]) - subtrahend - borrow;
      borrow = difference < 0;
      if (borrow) difference += std::int64_t(1) << 32;
      limbs_[index] = static_cast<std::uint32_t>(difference);
    }
    normalize();
  }

  [[nodiscard]] auto toUWide() const -> ConstInt::UWide {
    ConstInt::UWide value = 0;
    for (auto index = limbs_.size(); index-- > 0;)
      value = value << 32 | limbs_[index];
    return value;
  }

 private:
  void normalize() {
    while (!limbs_.empty() && !limbs_.back()) limbs_.pop_back();
  }

  std::vector<std::uint32_t> limbs_;
};

struct DividedMagnitude {
  ConstInt::UWide quotient = 0;
  bool inexact = false;
};

[[nodiscard]] auto divide(Magnitude dividend, const Magnitude& divisor)
    -> DividedMagnitude {
  DividedMagnitude result;
  for (int bit = dividend.bitWidth() - divisor.bitWidth(); bit >= 0; --bit) {
    auto shifted = divisor;
    shifted.shiftLeft(bit);
    if (dividend < shifted) continue;
    dividend.subtract(shifted);
    result.quotient |= ConstInt::UWide(1) << bit;
  }
  result.inexact = !dividend.isZero();
  return result;
}

[[nodiscard]] auto digit_value(char ch) -> int {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  return 16;
}

struct ScaledNumeral {
  Magnitude digits{0};
  int radix = 10;
  long exponent = 0;
};

[[nodiscard]] auto scaled_numeral(std::string_view spelling) -> ScaledNumeral {
  constexpr long kExponentLimit = 1'000'000;
  ScaledNumeral numeral;
  std::size_t pos = 0;
  if (spelling.size() > 1 && spelling[0] == '0' &&
      (spelling[1] == 'x' || spelling[1] == 'X')) {
    numeral.radix = 16;
    pos = 2;
  }
  const int digitScale = numeral.radix == 16 ? 4 : 1;
  bool inFraction = false;
  for (; pos < spelling.size(); ++pos) {
    const char ch = spelling[pos];
    if (ch == '\'') continue;
    if (ch == '.') {
      inFraction = true;
      continue;
    }
    const int digit = digit_value(ch);
    if (digit >= numeral.radix) break;
    numeral.digits.multiplyAdd(numeral.radix, digit);
    if (inFraction) numeral.exponent -= digitScale;
  }
  if (pos == spelling.size()) return numeral;
  const char marker = spelling[pos];
  const bool exponentMarker = numeral.radix == 16
                                  ? marker == 'p' || marker == 'P'
                                  : marker == 'e' || marker == 'E';
  if (!exponentMarker) return numeral;
  ++pos;
  bool negative = false;
  if (pos < spelling.size() && (spelling[pos] == '+' || spelling[pos] == '-'))
    negative = spelling[pos++] == '-';
  long written = 0;
  for (; pos < spelling.size(); ++pos) {
    const char ch = spelling[pos];
    if (ch == '\'') continue;
    if (ch < '0' || ch > '9') break;
    written = std::min(written * 10 + (ch - '0'), kExponentLimit);
  }
  numeral.exponent += negative ? -written : written;
  return numeral;
}

[[nodiscard]] auto round_numeral(ScaledNumeral numeral,
                                 const FloatingPointFormat& format)
    -> RoundedSignificand {
  const int targetBits = format.significandDigits + 3;
  const RoundedSignificand zero{.exponent = 1 - format.maxExponent()};
  const RoundedSignificand infinity{.exponent = format.maxExponent() + 1};
  if (numeral.digits.isZero()) return zero;

  auto binaryExponent = numeral.radix == 16 ? numeral.exponent : 0L;
  const auto decimalExponent = numeral.radix == 16 ? 0L : numeral.exponent;
  const auto estimate = static_cast<double>(numeral.digits.bitWidth()) +
                        static_cast<double>(binaryExponent) +
                        static_cast<double>(decimalExponent) * std::log2(10.0);
  if (estimate > format.maxExponent() + 3) return infinity;
  if (estimate < -format.maxExponent() - format.significandDigits - 3)
    return zero;

  bool inexact = false;
  ConstInt::UWide integer = 0;
  if (decimalExponent < 0) {
    Magnitude divisor{1};
    divisor.multiplyByPowerOfFive(static_cast<int>(-decimalExponent));
    const int scale =
        targetBits + divisor.bitWidth() - numeral.digits.bitWidth();
    if (scale >= 0)
      numeral.digits.shiftLeft(scale);
    else
      inexact = numeral.digits.shiftRight(-scale);
    const auto quotient = divide(numeral.digits, divisor);
    integer = quotient.quotient;
    inexact = inexact || quotient.inexact;
    binaryExponent = decimalExponent - scale;
  } else {
    numeral.digits.multiplyByPowerOfFive(static_cast<int>(decimalExponent));
    binaryExponent += decimalExponent;
    const int excessBits = numeral.digits.bitWidth() - targetBits;
    if (excessBits > 0) {
      inexact = numeral.digits.shiftRight(excessBits);
      binaryExponent += excessBits;
    }
    integer = numeral.digits.toUWide();
  }
  const auto excess =
      inexact ? std::strong_ordering::greater : std::strong_ordering::equal;
  return round_significand(integer, static_cast<int>(binaryExponent), excess,
                           format);
}

[[nodiscard]] auto host_long_double_format() -> FloatingPointFormat {
  using Limits = std::numeric_limits<long double>;
  return FloatingPointFormat{.exponentBits = std::bit_width(
                                 static_cast<unsigned>(Limits::max_exponent)),
                             .significandDigits = Limits::digits};
}

[[nodiscard]] auto holds_host_long_double(const FloatingPointFormat& format)
    -> bool {
  const auto host = host_long_double_format();
  return format.significandDigits >= host.significandDigits &&
         format.maxExponent() >= host.maxExponent();
}

[[nodiscard]] auto host_long_double(const RoundedSignificand& rounded,
                                    const FloatingPointFormat& format)
    -> long double {
  if (rounded.exponent > format.maxExponent())
    return std::numeric_limits<long double>::infinity();
  return std::ldexp(static_cast<long double>(rounded.significand),
                    rounded.exponent - (format.significandDigits - 1));
}

[[nodiscard]] auto sign_of(long double value) -> std::strong_ordering {
  if (value > 0) return std::strong_ordering::greater;
  if (value < 0) return std::strong_ordering::less;
  return std::strong_ordering::equal;
}

[[nodiscard]] auto stored_significand(ConstInt::UWide significand,
                                      const FloatingPointFormat& format)
    -> ConstInt::UWide {
  if (format.explicitIntegerBit) return significand;
  return significand & ((ConstInt::UWide(1) << format.fractionBits()) - 1);
}
}  // namespace

MemoryLayout::MemoryLayout(std::size_t bits) : bits_(bits) {
  sizeOfPointer_ = bits / 8;
  sizeOfLong_ = bits / 8;
  sizeOfLongLong_ = 8;
  sizeOfLongDouble_ = 8;
  longDoubleMantissaDigits_ = 53;
}

MemoryLayout::~MemoryLayout() = default;

auto to_string(DeviceProfile profile) -> std::string_view {
  switch (profile) {
    case DeviceProfile::kNone:
      return "none";
    case DeviceProfile::kVulkan11F32:
      return kVulkan11F32ProfileName;
  }
  std::unreachable();
}

MemoryLayout::MemoryLayout(const Triple& triple)
    : MemoryLayout(pointerBits(triple.arch())) {
  setTriple(triple.str());
  if (isSPIRV()) {
    setSizeOfLong(8);
    setSizeOfLongDouble(8, 53);
    setDefaultNewAlignment(8);
  }
}

auto MemoryLayout::bits() const -> std::size_t { return bits_; }

auto MemoryLayout::sizeOfSizeType() const -> std::size_t {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOfPointer() const -> std::size_t {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOfLong() const -> std::size_t { return sizeOfLong_; }

auto MemoryLayout::sizeOfLongLong() const -> std::size_t {
  return sizeOfLongLong_;
}

auto MemoryLayout::sizeOfLongDouble() const -> std::size_t {
  return sizeOfLongDouble_;
}

auto MemoryLayout::longDoubleMantissaDigits() const -> std::size_t {
  return longDoubleMantissaDigits_;
}

auto MemoryLayout::unwindExceptionSize() const -> std::size_t {
  const auto size = 3 * sizeof(std::uint64_t) + sizeOfPointer_;
  const auto alignment = 2 * sizeOfPointer_;
  return (size + alignment - 1) / alignment * alignment;
}

void MemoryLayout::setSizeOfPointer(std::size_t sizeOfPointer) {
  sizeOfPointer_ = sizeOfPointer;
}

void MemoryLayout::setSizeOfLong(std::size_t sizeOfLong) {
  sizeOfLong_ = sizeOfLong;
}

void MemoryLayout::setSizeOfLongDouble(std::size_t sizeOfLongDouble,
                                       std::size_t mantissaDigits) {
  sizeOfLongDouble_ = sizeOfLongDouble;
  longDoubleMantissaDigits_ = mantissaDigits;
}

auto MemoryLayout::sizeOfWideChar() const -> std::size_t {
  return sizeOfWideChar_;
}

auto MemoryLayout::isWideCharSigned() const -> bool {
  return wideCharIsSigned_;
}

void MemoryLayout::setWideCharUnderlyingType(std::size_t size, bool isSigned) {
  sizeOfWideChar_ = size;
  wideCharIsSigned_ = isSigned;
}

auto MemoryLayout::sizeOf(const Type* type) const
    -> std::optional<std::size_t> {
  if (!type) return std::nullopt;
  return visit(SizeOf{*this}, type);
}

auto MemoryLayout::alignmentOf(const Type* type) const
    -> std::optional<std::size_t> {
  if (!type) return std::nullopt;
  return visit(AlignmentOf{*this}, type);
}

auto MemoryLayout::variableAlignmentOf(const Type* type) const
    -> std::optional<std::size_t> {
  if (auto vla = unqualified_cast<UnresolvedBoundedArrayType>(type)) {
    auto alignment = variableAlignmentOf(vla->elementType());
    if (!alignment) return std::nullopt;
    return std::max(*alignment, largeArrayAlignment());
  }
  auto alignment = alignmentOf(type);
  if (!alignment || !largeArrayMinSize()) return alignment;
  if (!unqualified_cast<BoundedArrayType>(type)) return alignment;
  auto size = sizeOf(type);
  if (!size || *size < largeArrayMinSize()) return alignment;
  return std::max(*alignment, largeArrayAlignment());
}

auto MemoryLayout::largeArrayMinSize() const -> std::size_t {
  if (isX86_64() || isWebAssembly()) return 16;
  return 0;
}

auto MemoryLayout::largeArrayAlignment() const -> std::size_t {
  if (isX86_64() || isWebAssembly()) return 16;
  return 0;
}

auto MemoryLayout::floatingPointFormat(const Type* type) const
    -> std::optional<FloatingPointFormat> {
  if (!type) return std::nullopt;
  return visit(FloatingPointFormatOf{*this}, type);
}

auto FloatingPointFormat::fractionBits() const -> int {
  return explicitIntegerBit ? significandDigits : significandDigits - 1;
}

auto FloatingPointFormat::maxExponent() const -> int {
  return (1 << (exponentBits - 1)) - 1;
}

auto FloatingPointFormat::representsInteger(std::intmax_t value) const -> bool {
  if (value == 0) return true;
  const auto magnitude =
      value < 0 ? std::uint64_t(-(value + 1)) + 1 : std::uint64_t(value);
  const int width = std::bit_width(magnitude);
  const int significant = width - std::countr_zero(magnitude);
  if (significant > significandDigits) return false;
  return width - 1 <= maxExponent();
}

namespace {
template <typename Float>
[[nodiscard]] auto floatingRepresentation(Float value,
                                          const FloatingPointFormat& format)
    -> ConstInt::UWide {
  const auto integerBit = ConstInt::UWide(1) << (format.significandDigits - 1);
  const auto sign = ConstInt::UWide(std::signbit(value) ? 1 : 0)
                    << (format.exponentBits + format.fractionBits());
  const auto maxBiasedExponent =
      (ConstInt::UWide(1) << format.exponentBits) - 1;
  const auto infinity = (maxBiasedExponent << format.fractionBits()) |
                        stored_significand(integerBit, format);

  if (std::isnan(value)) {
    constexpr auto sourceDigits = std::numeric_limits<Float>::digits - 1;
    auto bytes = std::bit_cast<std::array<unsigned char, sizeof(Float)>>(value);
    ConstInt::UWide bits = 0;
    constexpr auto size = std::numeric_limits<Float>::digits == 64
                              ? std::size_t{10}
                              : sizeof(Float);
    for (std::size_t index = 0; index < size; ++index) {
      auto shift = std::endian::native == std::endian::little
                       ? index * 8
                       : (size - index - 1) * 8;
      auto offset = std::endian::native == std::endian::little
                        ? index
                        : bytes.size() - size + index;
      bits |= ConstInt::UWide(bytes[offset]) << shift;
    }
    auto payload = bits & ((ConstInt::UWide(1) << sourceDigits) - 1);
    auto targetDigits = format.significandDigits - 1;
    if (targetDigits < sourceDigits)
      payload >>= sourceDigits - targetDigits;
    else
      payload <<= targetDigits - sourceDigits;
    if (targetDigits != sourceDigits || !payload) payload |= integerBit >> 1;
    return sign | infinity | stored_significand(integerBit | payload, format);
  }
  if (std::isinf(value)) return sign | infinity;
  if (value == 0) return sign;

  const auto rounded = round_to_format(std::fabs(value), format);
  if (rounded.exponent > format.maxExponent()) return sign | infinity;

  const bool isNormal = (rounded.significand & integerBit) != 0;
  const auto biasedExponent =
      isNormal ? ConstInt::UWide(rounded.exponent + format.maxExponent()) : 0;
  return sign | (biasedExponent << format.fractionBits()) |
         stored_significand(rounded.significand, format);
}
}  // namespace

auto FloatingPointFormat::representation(float value) const -> ConstInt::UWide {
  return floatingRepresentation(value, *this);
}

auto FloatingPointFormat::representation(double value) const
    -> ConstInt::UWide {
  return floatingRepresentation(value, *this);
}

auto FloatingPointFormat::representation(long double value) const
    -> ConstInt::UWide {
  return floatingRepresentation(value, *this);
}

auto FloatingPointFormat::rounded(long double value,
                                  std::strong_ordering excess) const
    -> long double {
  if (holds_host_long_double(*this)) return value;
  if (!std::isfinite(value) || value == 0) return value;
  const auto magnitudeExcess = std::signbit(value) ? 0 <=> excess : excess;
  const auto result = round_to_format(std::fabs(value), *this, magnitudeExcess);
  return std::copysign(host_long_double(result, *this), value);
}

auto FloatingPointFormat::sum(long double left, long double right) const
    -> long double {
  const auto value = left + right;
  const auto rightPart = value - left;
  const auto error = (left - (value - rightPart)) + (right - rightPart);
  return rounded(value, sign_of(error));
}

auto FloatingPointFormat::difference(long double left, long double right) const
    -> long double {
  return sum(left, -right);
}

auto FloatingPointFormat::product(long double left, long double right) const
    -> long double {
  const auto value = left * right;
  return rounded(value, sign_of(std::fma(left, right, -value)));
}

auto FloatingPointFormat::quotient(long double left, long double right) const
    -> long double {
  const auto value = left / right;
  const auto remainder = sign_of(std::fma(-value, right, left));
  return rounded(value, std::signbit(right) ? 0 <=> remainder : remainder);
}

auto FloatingPointFormat::literalValue(std::string_view spelling) const
    -> long double {
  const auto format =
      holds_host_long_double(*this) ? host_long_double_format() : *this;
  return host_long_double(round_numeral(scaled_numeral(spelling), format),
                          format);
}

auto FloatingPointFormat::integerValue(const ConstInt& value) const
    -> long double {
  const auto magnitude = value.magnitude();
  if (!magnitude) return 0;
  const auto format =
      holds_host_long_double(*this) ? host_long_double_format() : *this;
  const auto result = host_long_double(
      round_significand(magnitude, 0, std::strong_ordering::equal, format),
      format);
  return value.isNegative() ? -result : result;
}

auto FloatingPointFormat::value(ConstInt::UWide representation) const
    -> double {
  const auto integerBit = ConstInt::UWide(1) << (significandDigits - 1);
  const auto fractionMask = (ConstInt::UWide(1) << fractionBits()) - 1;
  const auto maxBiasedExponent = (ConstInt::UWide(1) << exponentBits) - 1;
  const bool negative =
      ((representation >> (exponentBits + fractionBits())) & 1) != 0;
  const auto biasedExponent =
      (representation >> fractionBits()) & maxBiasedExponent;
  auto significand = representation & fractionMask;
  if (explicitIntegerBit) significand |= representation & integerBit;

  const auto sign = negative ? -1.0 : 1.0;

  if (biasedExponent == maxBiasedExponent) {
    if (significand & (integerBit - 1))
      return std::numeric_limits<double>::quiet_NaN();
    return sign * std::numeric_limits<double>::infinity();
  }

  if (biasedExponent != 0 && !explicitIntegerBit) significand |= integerBit;

  const int exponent = biasedExponent != 0
                           ? static_cast<int>(biasedExponent) - maxExponent()
                           : 1 - maxExponent();

  return sign * std::ldexp(static_cast<double>(significand),
                           exponent - (significandDigits - 1));
}

auto FloatingPointFormat::rangeContains(double value) const -> bool {
  if (!std::isfinite(value)) return true;
  const auto largest =
      std::ldexp(2.0 - std::ldexp(1.0, 1 - significandDigits), maxExponent());
  return std::fabs(value) <= largest;
}

auto MemoryLayout::triple() const -> const std::string& { return triple_; }

auto MemoryLayout::arch() const -> std::string_view {
  return std::string_view{triple_}.substr(0, triple_.find('-'));
}

auto MemoryLayout::targetArch() const -> TripleArch {
  return parseTripleArch(arch());
}

auto MemoryLayout::isAArch64() const -> bool {
  return targetArch() == TripleArch::kAArch64;
}

auto MemoryLayout::isX86_64() const -> bool {
  return targetArch() == TripleArch::kX86_64;
}

auto MemoryLayout::isWebAssembly() const -> bool {
  return arch().starts_with("wasm");
}

namespace {

class TargetDataLayoutWriter {
 public:
  TargetDataLayoutWriter(const MemoryLayout& memoryLayout, const Triple& triple)
      : memoryLayout_(memoryLayout), triple_(triple) {}

  [[nodiscard]] auto write() -> std::string {
    switch (triple_.arch()) {
      case TripleArch::kAArch64:
        writeAArch64();
        break;
      case TripleArch::kX86_64:
        writeX86_64();
        break;
      case TripleArch::kWasm32:
        writeWasm32();
        break;
      case TripleArch::kSPIRV32:
      case TripleArch::kSPIRV64:
        writeSPIRV();
        break;
      case TripleArch::kUnknown:
        break;
    }
    return std::move(text_);
  }

 private:
  void writeAArch64() {
    writeHeader();
    writeMicrosoftPointerAddressSpaces();
    if (triple_.objectFormat() == TripleObjectFormat::kCOFF) {
      writePointer();
      writeInteger(32);
    }
    if (triple_.objectFormat() == TripleObjectFormat::kELF) {
      writeInteger(8, 32);
      writeInteger(16, 32);
    }
    writeInteger(64);
    writeInt128();
    add("n32:64");
    add("S128");
    add("Fn32");
  }

  void writeX86_64() {
    writeHeader();
    writeMicrosoftPointerAddressSpaces();
    writeInteger(64);
    writeInt128();
    add("f80:128");
    add("n8:16:32:64");
    add("S128");
  }

  void writeWasm32() {
    writeHeader();
    writeNonDefaultPointer();
    add("p10:8:8");
    add("p20:8:8");
    writeInteger(64);
    writeInt128();
    writeBinary128();
    add("n32:64");
    add("S128");
    add("ni:1:10:20");
  }

  void writeSPIRV() {
    writeHeader();
    writeNonDefaultPointer();
    writeInteger(64);
    writeInt128();
    add("v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-"
        "v1024:1024");
    add("n8:16:32:64");
    add("G1");
  }

  void writeHeader() {
    add("e");
    writeMangling();
  }

  void writeMangling() {
    switch (triple_.objectFormat()) {
      case TripleObjectFormat::kMachO:
        add("m:o");
        break;
      case TripleObjectFormat::kCOFF:
        add("m:w");
        break;
      case TripleObjectFormat::kELF:
      case TripleObjectFormat::kWasm:
        add("m:e");
        break;
      case TripleObjectFormat::kSPIRV:
      case TripleObjectFormat::kUnknown:
        break;
    }
  }

  void writeMicrosoftPointerAddressSpaces() {
    add("p270:32:32");
    add("p271:32:32");
    add("p272:64:64");
  }

  void writePointer() {
    const auto bits = memoryLayout_.sizeOfPointer() * 8;
    add(std::format("p:{}:{}", bits, bits));
  }

  void writeNonDefaultPointer() {
    if (memoryLayout_.sizeOfPointer() == kDefaultPointerSize) return;
    writePointer();
  }

  void writeInteger(int bits) { add(std::format("i{}:{}", bits, bits)); }

  void writeInteger(int bits, int preferredBits) {
    add(std::format("i{}:{}:{}", bits, bits, preferredBits));
  }

  void writeInt128() {
    if (!memoryLayout_.hasInt128Type()) return;
    writeInteger(128);
  }

  void writeBinary128() {
    const auto alignmentBits = memoryLayout_.binary128Alignment() * 8;
    if (alignmentBits == kDefaultBinary128AlignmentBits) return;
    add(std::format("f128:{}", alignmentBits));
  }

  void add(std::string_view spec) {
    if (!text_.empty()) text_ += '-';
    text_ += spec;
  }

  static constexpr std::size_t kDefaultPointerSize = 8;
  static constexpr std::size_t kDefaultBinary128AlignmentBits = 128;

  const MemoryLayout& memoryLayout_;
  const Triple& triple_;
  std::string text_;
};

}  // namespace

auto MemoryLayout::targetDataLayout() const -> std::string {
  const Triple triple{triple_};
  return TargetDataLayoutWriter{*this, triple}.write();
}

auto MemoryLayout::isSPIRV() const -> bool {
  auto target = targetArch();
  return target == TripleArch::kSPIRV32 || target == TripleArch::kSPIRV64;
}

auto MemoryLayout::hasFloat16Type() const -> bool { return !isWebAssembly(); }

auto MemoryLayout::hasInt128Type() const -> bool { return !isSPIRV(); }

auto MemoryLayout::binary128Alignment() const -> std::size_t {
  if (Triple{triple_}.os() == TripleOS::kEmscripten) return 8;
  return 16;
}

auto MemoryLayout::isDarwin() const -> bool {
  std::string_view triple{triple_};
  return triple.find("apple") != std::string_view::npos ||
         triple.find("darwin") != std::string_view::npos ||
         triple.find("macos") != std::string_view::npos;
}

auto MemoryLayout::usesArmMemberPointerAbi() const -> bool {
  return isAArch64() || isWebAssembly();
}

auto MemoryLayout::unnamedBitFieldAlignsAggregate() const -> bool {
  if (isDarwin()) return false;
  return isAArch64();
}

auto MemoryLayout::structorsReturnThis() const -> bool {
  if (isWebAssembly()) return true;
  if (isAArch64()) return isDarwin();
  return false;
}

auto MemoryLayout::aliasesStructorVariants() const -> bool {
  if (isWebAssembly()) return true;
  return Triple{triple_}.os() == TripleOS::kLinux;
}

auto MemoryLayout::arrayCookieHasElementSize() const -> bool {
  if (isAArch64()) return isDarwin();
  return false;
}

auto MemoryLayout::defaultNewAlignment() const -> std::size_t {
  return defaultNewAlignment_;
}

void MemoryLayout::setDefaultNewAlignment(std::size_t defaultNewAlignment) {
  defaultNewAlignment_ = defaultNewAlignment;
}

auto MemoryLayout::alignedAttributeAlignment() const -> std::size_t {
  return 16;
}

auto MemoryLayout::maxAtomicInlineWidth() const -> std::size_t {
  if (isAArch64()) return 128;
  return 64;
}

auto MemoryLayout::maxAtomicPromoteWidth() const -> std::size_t {
  if (isAArch64() || isX86_64()) return 128;
  return 64;
}

auto MemoryLayout::maxVectorAlignment() const -> std::size_t {
  if (isAArch64()) return 16;
  return 0;
}

auto MemoryLayout::nullMemberObjectPointer() const -> std::int64_t {
  return -1;
}

auto MemoryLayout::classValueAbiKind() const -> ClassValueAbiKind {
  if (isWebAssembly()) return ClassValueAbiKind::kSingleScalar;
  if (isAArch64()) return ClassValueAbiKind::kAArch64;
  if (isX86_64()) return ClassValueAbiKind::kX86_64;
  return ClassValueAbiKind::kDefault;
}

auto MemoryLayout::abiPodDefinition() const -> AbiPodDefinition {
  if (isWebAssembly()) return AbiPodDefinition::kCxx11;
  if (!isDarwin()) return AbiPodDefinition::kCxx03UserProvided;
  if (isAArch64()) return AbiPodDefinition::kCxx11;
  return AbiPodDefinition::kCxx03UserDeclared;
}

auto MemoryLayout::extendsPromotableIntegersAtCalls() const -> bool {
  if (isAArch64()) return isDarwin();
  return true;
}

auto to_string(FramePointerKind kind) -> std::string_view {
  switch (kind) {
    case FramePointerKind::kNone:
      return "none";
    case FramePointerKind::kNonLeaf:
      return "non-leaf";
    case FramePointerKind::kAll:
      return "all";
  }
  return "none";
}

auto MemoryLayout::framePointerKind() const -> FramePointerKind {
  if (!isDarwin()) return FramePointerKind::kNone;
  if (isAArch64()) return FramePointerKind::kNonLeaf;
  return FramePointerKind::kAll;
}

void MemoryLayout::setTriple(std::string triple) {
  triple_ = std::move(triple);
}
}  // namespace cxx
