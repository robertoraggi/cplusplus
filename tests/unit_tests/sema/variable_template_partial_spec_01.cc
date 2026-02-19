// clang-format off
// RUN: %cxx -verify -fcheck %s

// Variable template partial specialization: base case and recursive evaluation
template <long _Xp, long _Yp>
inline constexpr long __static_gcd = __static_gcd<_Yp, _Xp % _Yp>;

template <long _Xp>
inline constexpr long __static_gcd<_Xp, 0> = _Xp;

// Base case: second arg is 0, should select partial spec
static_assert(__static_gcd<5, 0> == 5, "base case: gcd(5,0) = 5");
static_assert(__static_gcd<0, 0> == 0, "base case: gcd(0,0) = 0");
static_assert(__static_gcd<42, 0> == 42, "base case: gcd(42,0) = 42");

// Recursive case: gcd(12, 8) -> gcd(8, 4) -> gcd(4, 0) -> 4
static_assert(__static_gcd<12, 8> == 4, "recursive: gcd(12,8) = 4");
static_assert(__static_gcd<6, 4> == 2, "recursive: gcd(6,4) = 2");
static_assert(__static_gcd<100, 75> == 25, "recursive: gcd(100,75) = 25");
static_assert(__static_gcd<17, 13> == 1, "recursive: gcd(17,13) = 1");

// Simple variable template (non-recursive, non-partial)
template <int N>
inline constexpr int doubled = N * 2;

static_assert(doubled<5> == 10, "doubled");
static_assert(doubled<0> == 0, "doubled zero");

// Negative abs via variable template (conditional in constexpr var template)
template <long _Xp>
inline constexpr long __static_abs = _Xp < 0 ? -_Xp : _Xp;
