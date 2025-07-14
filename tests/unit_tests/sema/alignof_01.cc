// RUN: %cxx -toolchain wasm32 -verify -fcheck %s

static_assert(alignof(bool) == 1);
static_assert(alignof(signed char) == 1);
static_assert(alignof(unsigned char) == 1);
static_assert(alignof(short) == 2);
static_assert(alignof(unsigned short) == 2);
static_assert(alignof(int) == 4);
static_assert(alignof(unsigned int) == 4);
static_assert(alignof(long long) == 8);
static_assert(alignof(unsigned long long) == 8);
static_assert(alignof(float) == 4);
static_assert(alignof(double) == 8);
static_assert(alignof(char) == 1);
static_assert(alignof(wchar_t) == 4);
static_assert(alignof(char8_t) == 1);
static_assert(alignof(char16_t) == 2);
static_assert(alignof(char32_t) == 4);

#ifdef __wasm32__
static_assert(alignof(long) == 4);
static_assert(alignof(unsigned long) == 4);
static_assert(alignof(long double) == 16);
#endif

static_assert(alignof(char[]) == 1);
static_assert(alignof(int[]) == 4);
static_assert(alignof(double[]) == 8);
static_assert(alignof(__int128_t) == 16);