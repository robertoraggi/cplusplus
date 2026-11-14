// REQUIRES: int128
// RUN: %cxx --target=arm64-apple-macosx14.0.0 -verify -fsyntax-only %s
// RUN: %cxx --target=x86_64-pc-windows-msvc -verify -fsyntax-only %s
// expected-no-diagnostics

using u128 = unsigned __int128;
using i128 = __int128;

constexpr u128 sticky = u128(1) << 127 | u128(1) << 74 | 1;
constexpr i128 negative = -i128(u128(1) << 126 | u128(1) << 73 | 1);

static_assert(static_cast<double>(u128(1) << 100 | 1) == 0x1p100);
static_assert(static_cast<double>(sticky) == 0x1.0000000000001p127);
static_assert(static_cast<float>(sticky) == 0x1p127f);
static_assert(static_cast<long double>(sticky) == 0x1.0000000000001p127L);
static_assert(static_cast<double>(negative) ==
              -0x1.0000000000001p126);
static_assert(static_cast<long double>(negative) ==
              -0x1.0000000000001p126L);

constexpr double implicit = sticky;
static_assert(implicit == 0x1.0000000000001p127);

constexpr long double implicitLong = sticky;
static_assert(implicitLong == 0x1.0000000000001p127L);
