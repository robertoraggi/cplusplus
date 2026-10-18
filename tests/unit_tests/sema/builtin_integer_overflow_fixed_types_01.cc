// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

#ifdef CXX_HAS_INT128

#include <limits.h>

constexpr bool unsigned_add() { unsigned long long r = 0; bool o = __builtin_uaddll_overflow(~0ull, 1ull, &r); return o && r == 0; }
constexpr bool unsigned_add_fits() { unsigned long long r = 0; bool o = __builtin_uaddll_overflow(1ull, 2ull, &r); return !o && r == 3; }
constexpr bool signed_add() { int r = 0; bool o = __builtin_sadd_overflow(INT_MAX, 1, &r); return o && r == INT_MIN; }
constexpr bool signed_sub_long() { long r = 0; bool o = __builtin_ssubl_overflow(LONG_MIN, 1L, &r); return o && r == LONG_MAX; }
constexpr bool unsigned_sub() { unsigned r = 0; bool o = __builtin_usub_overflow(0u, 1u, &r); return o && r == ~0u; }
constexpr bool signed_mul_long_long() { long long r = 0; bool o = __builtin_smulll_overflow(1LL << 40, 1LL << 40, &r); return o && r == 0; }
constexpr bool unsigned_mul_long() { unsigned long r = 0; bool o = __builtin_umull_overflow(6ul, 7ul, &r); return !o && r == 42; }
constexpr bool narrow_result_type() { int r = 0; bool o = __builtin_smul_overflow(65536, 65536, &r); return o && r == 0; }

static_assert(unsigned_add() && unsigned_add_fits() && signed_add());
static_assert(signed_sub_long() && unsigned_sub() && signed_mul_long_long());
static_assert(unsigned_mul_long() && narrow_result_type());

constexpr unsigned carry_chain(unsigned a, unsigned b, unsigned c, unsigned* out) { return __builtin_addc(a, b, c, out); }
constexpr bool carry_out() { unsigned co = 7; unsigned r = carry_chain(0xffffffffu, 0, 1, &co); return r == 0 && co == 1; }
constexpr bool carry_none() { unsigned co = 7; unsigned r = carry_chain(1, 2, 3, &co); return r == 6 && co == 0; }
constexpr bool borrow_byte() { unsigned char bo = 7; unsigned char r = __builtin_subcb(5, 3, 3, &bo); return r == 255 && bo == 1; }
constexpr bool borrow_long_long() { unsigned long long bo = 7; unsigned long long r = __builtin_subcll(0, 0, 1, &bo); return r == ~0ull && bo == 1; }
constexpr bool add_carry_short() { unsigned short co = 7; unsigned short r = __builtin_addcs(65535, 1, 0, &co); return r == 0 && co == 1; }

static_assert(carry_out() && carry_none() && borrow_byte());
static_assert(borrow_long_long() && add_carry_short());

#endif
