// clang-format off
// RUN: %cxx -verify -fcheck %s

template <long _Num, long _Den = 1>
struct ratio {
  static constexpr long num = _Num;
  static constexpr long den = _Den;
};

// Non-recursive gcd base case for testing NNS substitution in isolation.
template <long _Xp>
struct gcd_base {
  static const long value = _Xp;
};

template <class R1, class R2>
struct ratio_info {
  static constexpr long r1_num = R1::num;
  static constexpr long r1_den = R1::den;
  static constexpr long r2_num = R2::num;
  static constexpr long r2_den = R2::den;
  // Also test NNS substitution with template arguments.
  static const long gcd_r1_num = gcd_base<R1::num>::value;
  static const long gcd_r2_den = gcd_base<R2::den>::value;
};

// Verify NNS substitution with constexpr members.
using RI = ratio_info<ratio<2, 3>, ratio<5, 7>>;
static_assert(RI::r1_num == 2, "r1_num");
static_assert(RI::r1_den == 3, "r1_den");
static_assert(RI::r2_num == 5, "r2_num");
static_assert(RI::r2_den == 7, "r2_den");

// Verify const integral evaluation (C++ [expr.const]).
struct S {
  static const int x = 42;
  static const long y = 100;
};
static_assert(S::x == 42, "const integral member");
static_assert(S::y == 100, "const long member");

// Verify static const member through NNS + gcd_base.
// Verify static const member through NNS + gcd_base.
static_assert(RI::gcd_r1_num == 2, "gcd_base<R1::num>::value");
static_assert(RI::gcd_r2_den == 7, "gcd_base<R2::den>::value");
