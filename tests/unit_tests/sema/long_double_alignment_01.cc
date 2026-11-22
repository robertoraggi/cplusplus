// RUN: %cxx --target=wasm32-unknown-emscripten -verify -fsyntax-only %s
// RUN: %cxx --target=wasm32-wasip1 -verify -fsyntax-only %s
// expected-no-diagnostics

#ifdef __EMSCRIPTEN__
constexpr unsigned long binary128Alignment = 8;
#else
constexpr unsigned long binary128Alignment = 16;
#endif

struct LongDoubleMember {
  char c;
  long double value;
};

struct Float128Member {
  short s;
  __float128 value;
};

struct ComplexMember {
  char c;
  _Complex long double value;
};

static_assert(sizeof(long double) == 16);
static_assert(alignof(long double) == binary128Alignment);
static_assert(alignof(__float128) == binary128Alignment);
static_assert(alignof(_Complex long double) == binary128Alignment);
static_assert(sizeof(LongDoubleMember) == 16 + binary128Alignment);
static_assert(alignof(LongDoubleMember) == binary128Alignment);
static_assert(sizeof(Float128Member) == 16 + binary128Alignment);
static_assert(sizeof(ComplexMember) == 32 + binary128Alignment);
