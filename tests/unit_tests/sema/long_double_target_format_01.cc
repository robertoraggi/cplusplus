// RUN: %cxx --target=spirv32-unknown-unknown -verify -fsyntax-only %s
// RUN: %cxx --target=arm64-apple-macosx14.0.0 -verify -fsyntax-only %s
// RUN: %cxx --target=x86_64-pc-windows-msvc -verify -fsyntax-only %s
// expected-no-diagnostics

constexpr long double limit = 9007199254740992.0L;
static_assert(limit + 1.0L == limit);
static_assert(limit * 3.0L - 2.0L * limit == limit);
static_assert(static_cast<long double>(9007199254740993ULL) == limit);
static_assert(static_cast<long double>(9007199254740993LL) == limit);
static_assert(+(limit + 1.0L) == limit);
static_assert(sizeof(long double) == 8);
static_assert(__LDBL_MANT_DIG__ == 53);

constexpr long double above_half = 0x1.0000002p-53L;
static_assert(1.0L + above_half == 0x1.0000000000001p0L);
static_assert(-1.0L - above_half == -0x1.0000000000001p0L);
static_assert(1.0 + above_half == 0x1.0000000000001p0L);
static_assert(0x113d4a60b0e9dp-48L - 0x1000003p-77L == 0x113d4a60b0e9cfp-52L);
static_assert(0x1.3ee65c4fac5efp0L * 0x1.d67d902d899d9p0L ==
              0x1.250babc7d39b1p1L);
static_assert(0x1.355cf2eeb9895p0L * 0x1.766323e66b372p-1L ==
              0x1.c46d993cdda9bp-1L);
static_assert(0x195c677ccdca35p-52L / 0x5ff77fc0b1779p-50L ==
              0x10e9c4666981adp-52L);
static_assert(0x1.1b4a8cc7770c3p0L / 0x1.f7fbe80764304p0L ==
              0x1.1fcc0a00f7ca3p-1L);

constexpr auto accumulate(long double value, long double step) {
  value += step;
  return value;
}
static_assert(accumulate(1.0L, above_half) == 0x1.0000000000001p0L);

static_assert(1.00000000000000011102230246251565404236316680908203125L == 1.0L);
static_assert(1.000000000000000111022302462515654042363166809082031251L ==
              0x1.0000000000001p0L);
static_assert(1.0000000000000001110223024625156540423631668090820312499L ==
              1.0L);
static_assert(0x1.00000000000008000000000000000000001p0L ==
              0x1.0000000000001p0L);
static_assert(0x1.fffffffffffff7ffp1023L == 0x1.fffffffffffffp1023L);
static_assert(0x1.8p-1074L == 0x1p-1073L);
static_assert(0x1p-1074L * 0.5L == 0.0L);
